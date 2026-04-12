// scenario_injector.cpp — Programmable market shock injection implementation.
// Chronos / AETERNUS — production C++ scenario injector.

#include "scenario_injector.hpp"
#include <cstdio>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace chronos {
namespace scenario {

// ── Utilities ─────────────────────────────────────────────────────────────────

static std::string shock_type_name(ShockType t) {
    switch (t) {
        case ShockType::PriceJump:        return "PriceJump";
        case ShockType::SpreadWidening:   return "SpreadWidening";
        case ShockType::LiquidityRemoval: return "LiquidityRemoval";
        case ShockType::VolatilitySpike:  return "VolatilitySpike";
        case ShockType::TradingHalt:      return "TradingHalt";
        case ShockType::FlashCrash:       return "FlashCrash";
        case ShockType::LiquidityCrisis:  return "LiquidityCrisis";
        case ShockType::NewsEvent:        return "NewsEvent";
        case ShockType::RegimeChange:     return "RegimeChange";
        case ShockType::CircuitBreaker:   return "CircuitBreaker";
        default:                          return "Unknown";
    }
}

static std::string format_shock(const ShockEvent& s) {
    std::ostringstream oss;
    oss << "[Shock type=" << shock_type_name(s.type)
        << " inst=" << s.instrument
        << " trigger_ns=" << s.trigger_ns
        << " magnitude=" << std::fixed << std::setprecision(4) << s.magnitude
        << " duration=" << s.duration_ns
        << " desc=" << s.description << "]";
    return oss.str();
}

static std::string format_market_state(const MarketState& s) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4)
        << "MarketState[id=" << s.instrument_id
        << " mid=" << s.mid_price
        << " bid=" << s.bid
        << " ask=" << s.ask
        << " spread_bps=" << (s.ask - s.bid) / s.mid_price * 10000.0
        << " bid_depth=" << std::setprecision(0) << s.bid_depth
        << " ask_depth=" << s.ask_depth
        << " vol=" << std::setprecision(3) << s.realized_vol
        << " halted=" << (s.is_halted ? "YES" : "NO") << "]";
    return oss.str();
}

// ── Scenario script builder ───────────────────────────────────────────────────

class ScenarioScript {
public:
    explicit ScenarioScript(ScenarioInjector& injector) : injector_(injector) {}

    ScenarioScript& add_shock(ShockEvent e) {
        shocks_.push_back(std::move(e));
        return *this;
    }

    ScenarioScript& schedule_all() {
        for (auto& s : shocks_) injector_.schedule_shock(s);
        shocks_.clear();
        return *this;
    }

    // Convenience: schedule correlated multi-asset crash
    ScenarioScript& multi_asset_crash(
        const std::vector<InstrumentId>& instruments,
        Nanos start_ns,
        double drawdown_pct,
        double contagion_factor = 0.3)
    {
        if (instruments.empty()) return *this;
        // Primary instrument: full crash
        injector_.schedule_flash_crash_sequence(instruments[0], start_ns, drawdown_pct);
        // Secondary instruments: contagion
        for (size_t i = 1; i < instruments.size(); ++i) {
            double contagion_drop = drawdown_pct * contagion_factor;
            injector_.schedule_shock(ShockBuilder::price_jump(instruments[i], start_ns + static_cast<Nanos>(i * 1e9), -contagion_drop));
            injector_.schedule_shock(ShockBuilder::spread_widening(instruments[i], start_ns, 2.0 + i, 30e9));
        }
        return *this;
    }

    // News shock that affects multiple correlated instruments
    ScenarioScript& correlated_news_shock(
        const std::vector<InstrumentId>& instruments,
        const std::vector<double>& betas,
        Nanos trigger_ns,
        double base_move_pct)
    {
        for (size_t i = 0; i < instruments.size() && i < betas.size(); ++i) {
            double move = base_move_pct * betas[i];
            injector_.schedule_shock(ShockBuilder::news_event(instruments[i], trigger_ns, move));
        }
        return *this;
    }

private:
    ScenarioInjector& injector_;
    std::vector<ShockEvent> shocks_;
};

// ── Pre-built scenario profiles ───────────────────────────────────────────────

struct ScenarioProfile {
    std::string name;
    std::string description;
    std::function<void(ScenarioInjector&, Nanos)> apply;
};

std::vector<ScenarioProfile> built_in_profiles() {
    return {
        {
            "flash_crash_2010",
            "Simulates the May 2010 Flash Crash: rapid 9% drop and recovery",
            [](ScenarioInjector& inj, Nanos start_ns) {
                inj.schedule_flash_crash_sequence(1, start_ns, 0.09);
            }
        },
        {
            "liquidity_crisis",
            "Simulates a 30-minute liquidity crisis: 90% depth removal, 10x spread",
            [](ScenarioInjector& inj, Nanos start_ns) {
                inj.schedule_liquidity_crisis(1, start_ns, 1800e9);
            }
        },
        {
            "earnings_shock_positive",
            "Large positive earnings beat: +5% gap up",
            [](ScenarioInjector& inj, Nanos start_ns) {
                inj.schedule_shock(ShockBuilder::news_event(1, start_ns, 0.05));
                inj.schedule_shock(ShockBuilder::volatility_spike(1, start_ns, 2.5, 3600e9));
            }
        },
        {
            "earnings_shock_negative",
            "Earnings miss: -8% gap down with elevated volatility",
            [](ScenarioInjector& inj, Nanos start_ns) {
                inj.schedule_shock(ShockBuilder::news_event(1, start_ns, -0.08));
                inj.schedule_shock(ShockBuilder::volatility_spike(1, start_ns, 3.0, 7200e9));
                inj.schedule_shock(ShockBuilder::spread_widening(1, start_ns, 3.0, 3600e9));
            }
        },
        {
            "circuit_breaker_halt",
            "5-minute trading halt followed by reopening",
            [](ScenarioInjector& inj, Nanos start_ns) {
                inj.schedule_shock(ShockBuilder::trading_halt(1, start_ns, 300e9));
                inj.schedule_shock(ShockBuilder::price_jump(1, start_ns + 300e9, -0.03));
            }
        },
        {
            "index_arbitrage_pressure",
            "Coordinated pressure on index components",
            [](ScenarioInjector& inj, Nanos start_ns) {
                for (uint32_t i = 1; i <= 5; ++i) {
                    inj.schedule_shock(ShockBuilder::price_jump(i, start_ns + static_cast<Nanos>(i * 500e6), -0.02));
                    inj.schedule_shock(ShockBuilder::liquidity_removal(i, start_ns, 0.4, 60e9));
                }
            }
        },
    };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#ifndef SCENARIO_NO_TESTS

namespace tests {

void test_basic_price_jump() {
    ScenarioInjector inj(42);
    inj.add_instrument(1, 100.0);

    double initial = inj.state(1)->mid_price;
    inj.schedule_shock(ShockBuilder::price_jump(1, 1000, 0.05));
    inj.advance(2000);

    double after = inj.state(1)->mid_price;
    assert(std::abs(after / initial - 1.05) < 0.01);
    printf("test_basic_price_jump: PASSED (%.4f -> %.4f)\n", initial, after);
}

void test_spread_widening_and_revert() {
    ScenarioInjector inj(42);
    inj.add_instrument(1, 100.0);

    auto spread_before = inj.state(1)->ask - inj.state(1)->bid;
    inj.schedule_shock(ShockBuilder::spread_widening(1, 100, 5.0, 1000));
    inj.advance(200);
    auto spread_during = inj.state(1)->ask - inj.state(1)->bid;
    assert(spread_during > spread_before * 3.0);

    inj.advance(2000); // After duration
    auto spread_after = inj.state(1)->ask - inj.state(1)->bid;
    assert(spread_after < spread_during);
    printf("test_spread_widening_and_revert: PASSED spread: %.4f -> %.4f -> %.4f\n",
        spread_before, spread_during, spread_after);
}

void test_trading_halt() {
    ScenarioInjector inj(42);
    inj.add_instrument(1, 50.0);

    inj.schedule_shock(ShockBuilder::trading_halt(1, 100, 500));
    inj.advance(200);
    assert(inj.state(1)->is_halted);

    inj.advance(700);
    assert(!inj.state(1)->is_halted);
    printf("test_trading_halt: PASSED\n");
}

void test_flash_crash_sequence() {
    ScenarioInjector inj(42);
    inj.add_instrument(1, 100.0);

    inj.schedule_flash_crash_sequence(1, 0, 0.10);
    inj.advance(15'000'000'000ULL); // 15 seconds

    // After crash, price should be lower
    double final_price = inj.state(1)->mid_price;
    assert(final_price < 100.0);
    printf("test_flash_crash_sequence: PASSED price=%.4f\n", final_price);
}

void test_liquidity_crisis() {
    ScenarioInjector inj(42);
    inj.add_instrument(1, 200.0);

    double initial_depth = inj.state(1)->bid_depth;
    inj.schedule_liquidity_crisis(1, 0, 1'000'000'000ULL);
    inj.advance(100'000'000);

    double crisis_depth = inj.state(1)->bid_depth;
    assert(crisis_depth < initial_depth);
    assert(inj.state(1)->spread_multiplier > 5.0);
    printf("test_liquidity_crisis: PASSED depth_ratio=%.2f\n", crisis_depth / initial_depth);
}

void test_shock_callback() {
    ScenarioInjector inj(42);
    inj.add_instrument(1, 75.0);

    int callback_count = 0;
    inj.on_shock([&](const ShockEvent& s, const MarketState&) {
        ++callback_count;
    });

    inj.schedule_shock(ShockBuilder::news_event(1, 100, 0.03));
    inj.advance(200);
    assert(callback_count == 1);
    printf("test_shock_callback: PASSED count=%d\n", callback_count);
}

void test_built_in_profiles() {
    auto profiles = built_in_profiles();
    assert(!profiles.empty());
    for (const auto& p : profiles) {
        assert(!p.name.empty());
        assert(!p.description.empty());
        assert(p.apply != nullptr);
        // Apply each profile to a fresh injector
        ScenarioInjector inj(42);
        inj.add_instrument(1, 100.0);
        for (uint32_t i = 2; i <= 5; ++i) inj.add_instrument(i, 50.0 * i);
        p.apply(inj, 1'000'000'000ULL);
        // Run for 10 minutes
        for (uint64_t t = 0; t < 600'000'000'000ULL; t += 1'000'000'000ULL) {
            inj.advance(t);
        }
        printf("test_built_in_profiles[%s]: PASSED\n", p.name.c_str());
    }
}

void test_multiple_instruments() {
    ScenarioInjector inj(42);
    for (uint32_t i = 1; i <= 5; ++i) inj.add_instrument(i, 100.0 * i);

    assert(inj.instrument_count() == 5);
    for (uint32_t i = 1; i <= 5; ++i) {
        assert(inj.state(i) != nullptr);
    }
    printf("test_multiple_instruments: PASSED\n");
}

void test_shock_ordering() {
    ScenarioInjector inj(42);
    inj.add_instrument(1, 100.0);

    // Schedule out of order
    inj.schedule_shock(ShockBuilder::price_jump(1, 3000, 0.01));
    inj.schedule_shock(ShockBuilder::price_jump(1, 1000, 0.01));
    inj.schedule_shock(ShockBuilder::price_jump(1, 2000, 0.01));

    // Advance step by step and check shocks are applied in order
    inj.advance(1500);
    assert(inj.applied_shocks().size() == 1);
    inj.advance(2500);
    assert(inj.applied_shocks().size() == 2);
    inj.advance(4000);
    assert(inj.applied_shocks().size() == 3);
    printf("test_shock_ordering: PASSED\n");
}

void run_all() {
    test_basic_price_jump();
    test_spread_widening_and_revert();
    test_trading_halt();
    test_flash_crash_sequence();
    test_liquidity_crisis();
    test_shock_callback();
    test_built_in_profiles();
    test_multiple_instruments();
    test_shock_ordering();
    printf("All scenario injector tests PASSED\n");
}

} // namespace tests

#endif // SCENARIO_NO_TESTS

} // namespace scenario
} // namespace chronos

#ifndef SCENARIO_NO_MAIN
int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--test") {
        chronos::scenario::tests::run_all();
        return 0;
    }
    printf("ScenarioInjector — Chronos/AETERNUS\n");
    printf("Usage: scenario_injector --test\n");
    return 0;
}
#endif
