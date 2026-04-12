#pragma once
// scenario_injector.hpp — Programmable market shock injection.
// Chronos / AETERNUS — C++ scenario injector.

#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_map>
#include <deque>
#include <optional>

namespace chronos {
namespace scenario {

// ── Types ────────────────────────────────────────────────────────────────────

using InstrumentId = uint32_t;
using Nanos = uint64_t;
using Price = double;
using Qty = double;

// ── Shock types ───────────────────────────────────────────────────────────────

enum class ShockType {
    PriceJump,
    SpreadWidening,
    LiquidityRemoval,
    VolatilitySpike,
    TradingHalt,
    FlashCrash,
    LiquidityCrisis,
    NewsEvent,
    RegimeChange,
    CircuitBreaker,
};

// ── Market state ──────────────────────────────────────────────────────────────

struct MarketState {
    InstrumentId instrument_id;
    double mid_price;
    double bid;
    double ask;
    double bid_depth;
    double ask_depth;
    double realized_vol;
    double spread_multiplier;
    double depth_multiplier;
    bool   is_halted;
    Nanos  last_update_ns;
};

// ── Shock event ───────────────────────────────────────────────────────────────

struct ShockEvent {
    Nanos       trigger_ns;     // When to apply
    InstrumentId instrument;
    ShockType   type;
    double      magnitude;      // Interpretation depends on type
    double      duration_ns;    // How long it lasts
    std::string description;
    bool        applied;
    Nanos       applied_at_ns;
};

// ── Shock builder ─────────────────────────────────────────────────────────────

class ShockBuilder {
public:
    static ShockEvent price_jump(
        InstrumentId inst, Nanos trigger_ns, double pct_change
    ) {
        return ShockEvent{trigger_ns, inst, ShockType::PriceJump, pct_change, 0, "price_jump", false, 0};
    }

    static ShockEvent spread_widening(
        InstrumentId inst, Nanos trigger_ns, double multiplier, double duration_ns
    ) {
        return ShockEvent{trigger_ns, inst, ShockType::SpreadWidening, multiplier, duration_ns, "spread_widening", false, 0};
    }

    static ShockEvent liquidity_removal(
        InstrumentId inst, Nanos trigger_ns, double fraction, double duration_ns
    ) {
        return ShockEvent{trigger_ns, inst, ShockType::LiquidityRemoval, fraction, duration_ns, "liquidity_removal", false, 0};
    }

    static ShockEvent volatility_spike(
        InstrumentId inst, Nanos trigger_ns, double vol_multiplier, double duration_ns
    ) {
        return ShockEvent{trigger_ns, inst, ShockType::VolatilitySpike, vol_multiplier, duration_ns, "vol_spike", false, 0};
    }

    static ShockEvent trading_halt(
        InstrumentId inst, Nanos trigger_ns, double duration_ns
    ) {
        return ShockEvent{trigger_ns, inst, ShockType::TradingHalt, 1.0, duration_ns, "halt", false, 0};
    }

    static ShockEvent flash_crash(
        InstrumentId inst, Nanos trigger_ns, double drawdown_pct, double duration_ns
    ) {
        return ShockEvent{trigger_ns, inst, ShockType::FlashCrash, drawdown_pct, duration_ns, "flash_crash", false, 0};
    }

    static ShockEvent news_event(
        InstrumentId inst, Nanos trigger_ns, double price_impact_pct
    ) {
        return ShockEvent{trigger_ns, inst, ShockType::NewsEvent, price_impact_pct, 0, "news", false, 0};
    }
};

// ── Callback types ────────────────────────────────────────────────────────────

using ShockCallback = std::function<void(const ShockEvent&, const MarketState&)>;
using MarketUpdateCallback = std::function<void(const MarketState&)>;

// ── Scenario injector ─────────────────────────────────────────────────────────

class ScenarioInjector {
public:
    explicit ScenarioInjector(uint64_t seed = 42) : rng_state_(seed ^ 0xDEAD_BEEF_C0FFEE) {}

    // Register instruments
    void add_instrument(InstrumentId id, double mid_price) {
        MarketState s;
        s.instrument_id = id;
        s.mid_price = mid_price;
        s.bid = mid_price * (1.0 - 0.0001);
        s.ask = mid_price * (1.0 + 0.0001);
        s.bid_depth = 1000.0;
        s.ask_depth = 1000.0;
        s.realized_vol = 0.20;
        s.spread_multiplier = 1.0;
        s.depth_multiplier = 1.0;
        s.is_halted = false;
        s.last_update_ns = 0;
        states_[id] = s;
    }

    // Register a shock event
    void schedule_shock(ShockEvent event) {
        pending_shocks_.push_back(std::move(event));
        std::sort(pending_shocks_.begin(), pending_shocks_.end(),
            [](const ShockEvent& a, const ShockEvent& b) { return a.trigger_ns < b.trigger_ns; });
    }

    // Schedule multiple shocks from a scenario script
    void schedule_flash_crash_sequence(InstrumentId inst, Nanos start_ns, double drawdown_pct) {
        // 1) Initial price jump down at start
        schedule_shock(ShockBuilder::price_jump(inst, start_ns, -drawdown_pct * 0.3));
        // 2) Spread widens
        schedule_shock(ShockBuilder::spread_widening(inst, start_ns + 1e9, 5.0, 60e9));
        // 3) Liquidity removal
        schedule_shock(ShockBuilder::liquidity_removal(inst, start_ns + 2e9, 0.8, 120e9));
        // 4) More price drop
        schedule_shock(ShockBuilder::price_jump(inst, start_ns + 10e9, -drawdown_pct * 0.7));
        // 5) Partial recovery
        schedule_shock(ShockBuilder::price_jump(inst, start_ns + 60e9, drawdown_pct * 0.6));
    }

    void schedule_liquidity_crisis(InstrumentId inst, Nanos start_ns, double duration_ns) {
        schedule_shock(ShockBuilder::liquidity_removal(inst, start_ns, 0.9, duration_ns));
        schedule_shock(ShockBuilder::spread_widening(inst, start_ns, 10.0, duration_ns));
        schedule_shock(ShockBuilder::volatility_spike(inst, start_ns, 3.0, duration_ns));
    }

    // Advance time and apply due shocks
    void advance(Nanos current_ns) {
        current_ns_ = current_ns;
        apply_due_shocks(current_ns);
        revert_expired_shocks(current_ns);
        apply_ambient_noise(current_ns);
    }

    // Set callbacks
    void on_shock(ShockCallback cb) { shock_cb_ = std::move(cb); }
    void on_market_update(MarketUpdateCallback cb) { update_cb_ = std::move(cb); }

    const MarketState* state(InstrumentId id) const {
        auto it = states_.find(id);
        return (it != states_.end()) ? &it->second : nullptr;
    }

    std::vector<ShockEvent> applied_shocks() const { return applied_shocks_; }
    size_t pending_shock_count() const { return pending_shocks_.size(); }
    size_t instrument_count() const { return states_.size(); }

    Nanos current_time_ns() const { return current_ns_; }

private:
    std::unordered_map<InstrumentId, MarketState> states_;
    std::vector<ShockEvent> pending_shocks_;
    std::vector<ShockEvent> applied_shocks_;
    std::vector<std::pair<ShockEvent, Nanos>> active_shocks_; // (shock, revert_at_ns)
    ShockCallback shock_cb_;
    MarketUpdateCallback update_cb_;
    Nanos current_ns_ = 0;
    uint64_t rng_state_;
    Nanos last_noise_ns_ = 0;

    double next_rand() {
        rng_state_ ^= rng_state_ << 13;
        rng_state_ ^= rng_state_ >> 7;
        rng_state_ ^= rng_state_ << 17;
        return static_cast<double>(rng_state_ >> 11) / static_cast<double>(1ULL << 53);
    }

    double next_normal() {
        double u1 = std::max(next_rand(), 1e-15);
        double u2 = next_rand();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979 * u2);
    }

    void apply_due_shocks(Nanos now) {
        auto it = pending_shocks_.begin();
        while (it != pending_shocks_.end() && it->trigger_ns <= now) {
            apply_shock(*it);
            applied_shocks_.push_back(*it);
            if (it->duration_ns > 0) {
                active_shocks_.push_back({*it, now + static_cast<Nanos>(it->duration_ns)});
            }
            it = pending_shocks_.erase(it);
        }
    }

    void revert_expired_shocks(Nanos now) {
        for (auto it = active_shocks_.begin(); it != active_shocks_.end(); ) {
            if (it->second <= now) {
                revert_shock(it->first);
                it = active_shocks_.erase(it);
            } else { ++it; }
        }
    }

    void apply_shock(const ShockEvent& shock) {
        auto it = states_.find(shock.instrument);
        if (it == states_.end()) return;
        auto& s = it->second;

        switch (shock.type) {
            case ShockType::PriceJump:
                s.mid_price *= (1.0 + shock.magnitude);
                s.mid_price = std::max(s.mid_price, 0.001);
                update_quotes(s);
                break;
            case ShockType::SpreadWidening:
                s.spread_multiplier = shock.magnitude;
                update_quotes(s);
                break;
            case ShockType::LiquidityRemoval:
                s.depth_multiplier = 1.0 - shock.magnitude;
                s.bid_depth *= (1.0 - shock.magnitude);
                s.ask_depth *= (1.0 - shock.magnitude);
                break;
            case ShockType::VolatilitySpike:
                s.realized_vol *= shock.magnitude;
                break;
            case ShockType::TradingHalt:
                s.is_halted = true;
                break;
            case ShockType::FlashCrash:
                s.mid_price *= (1.0 - shock.magnitude);
                s.spread_multiplier = 5.0;
                s.depth_multiplier = 0.2;
                update_quotes(s);
                break;
            case ShockType::LiquidityCrisis:
                s.spread_multiplier = 10.0;
                s.depth_multiplier = 0.05;
                update_quotes(s);
                break;
            case ShockType::NewsEvent:
                s.mid_price *= (1.0 + shock.magnitude);
                s.spread_multiplier = 2.0;
                update_quotes(s);
                break;
            default: break;
        }

        s.last_update_ns = current_ns_;
        if (shock_cb_) shock_cb_(shock, s);
        if (update_cb_) update_cb_(s);
    }

    void revert_shock(const ShockEvent& shock) {
        auto it = states_.find(shock.instrument);
        if (it == states_.end()) return;
        auto& s = it->second;

        switch (shock.type) {
            case ShockType::SpreadWidening:
                s.spread_multiplier = 1.0;
                update_quotes(s);
                break;
            case ShockType::LiquidityRemoval:
                s.depth_multiplier = 1.0;
                s.bid_depth = 1000.0;
                s.ask_depth = 1000.0;
                break;
            case ShockType::VolatilitySpike:
                s.realized_vol = 0.20;
                break;
            case ShockType::TradingHalt:
                s.is_halted = false;
                break;
            case ShockType::FlashCrash:
                s.spread_multiplier = 1.0;
                s.depth_multiplier = 1.0;
                update_quotes(s);
                break;
            default: break;
        }

        s.last_update_ns = current_ns_;
        if (update_cb_) update_cb_(s);
    }

    void update_quotes(MarketState& s) {
        double half_spread = s.mid_price * 0.0001 * s.spread_multiplier;
        s.bid = s.mid_price - half_spread;
        s.ask = s.mid_price + half_spread;
    }

    void apply_ambient_noise(Nanos now) {
        if (now <= last_noise_ns_) return;
        last_noise_ns_ = now;

        for (auto& [id, s] : states_) {
            if (s.is_halted) continue;
            double vol_per_ns = s.realized_vol / std::sqrt(252.0 * 6.5 * 3600.0 * 1e9);
            double ret = vol_per_ns * next_normal();
            s.mid_price *= (1.0 + ret);
            s.mid_price = std::max(s.mid_price, 0.001);
            update_quotes(s);
            if (update_cb_) update_cb_(s);
        }
    }
};

} // namespace scenario
} // namespace chronos
