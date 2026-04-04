#pragma once
// Pre-trade risk checks: limits on position size, order rate, notional, etc.
// All checks are O(1) and designed to add < 1µs latency.

#include "order.hpp"
#include <atomic>
#include <string>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <array>

namespace hft {

enum class RejectionReason : uint8_t {
    None             = 0,
    MaxOrderSize     = 1,
    MaxPositionSize  = 2,
    MaxNotional      = 3,
    MaxOrderRate     = 4,
    MaxDailyVolume   = 5,
    PriceBandViolation = 6,
    DuplicateOrder   = 7,
    InvalidPrice     = 8,
    MarketClosed     = 9,
    KillSwitch       = 10
};

inline const char* rejection_str(RejectionReason r) {
    switch (r) {
    case RejectionReason::None:             return "None";
    case RejectionReason::MaxOrderSize:     return "MaxOrderSize";
    case RejectionReason::MaxPositionSize:  return "MaxPositionSize";
    case RejectionReason::MaxNotional:      return "MaxNotional";
    case RejectionReason::MaxOrderRate:     return "MaxOrderRate";
    case RejectionReason::MaxDailyVolume:   return "MaxDailyVolume";
    case RejectionReason::PriceBandViolation: return "PriceBandViolation";
    case RejectionReason::DuplicateOrder:   return "DuplicateOrder";
    case RejectionReason::InvalidPrice:     return "InvalidPrice";
    case RejectionReason::MarketClosed:     return "MarketClosed";
    case RejectionReason::KillSwitch:       return "KillSwitch";
    }
    return "Unknown";
}

struct RiskLimits {
    Quantity max_order_size     = 100000;  // shares per order
    Quantity max_position_size  = 500000;  // net position per symbol
    double   max_notional       = 10e6;    // dollars per order
    uint32_t max_order_rate     = 1000;    // orders per second
    Quantity max_daily_volume   = 10000000;
    double   price_band_pct     = 0.05;    // 5% from reference price
    bool     kill_switch        = false;
};

struct PositionState {
    int64_t  net_qty;            // positive = long, negative = short
    Quantity bought_today;
    Quantity sold_today;
    double   avg_cost;
    double   realized_pnl;
    double   unrealized_pnl;
    Price    last_price;
    Timestamp last_update;

    PositionState() noexcept
        : net_qty(0), bought_today(0), sold_today(0),
          avg_cost(0.0), realized_pnl(0.0), unrealized_pnl(0.0),
          last_price(0), last_update(0) {}

    void on_fill(Side side, Quantity qty, Price price) noexcept {
        double p = price_to_double(price);
        if (side == Side::Buy) {
            // Update avg cost
            double total_cost = avg_cost * std::abs(net_qty) + p * qty;
            net_qty += static_cast<int64_t>(qty);
            avg_cost = (net_qty != 0) ? total_cost / std::abs(net_qty) : 0.0;
            bought_today += qty;
        } else {
            if (net_qty > 0) {
                // Closing long: realize PnL
                Quantity close_qty = std::min(static_cast<Quantity>(net_qty), qty);
                realized_pnl += (p - avg_cost) * close_qty;
            }
            net_qty -= static_cast<int64_t>(qty);
            sold_today += qty;
        }
        last_price = price;
    }

    void mark_to_market(Price mkt_price) noexcept {
        if (net_qty != 0 && avg_cost > 0) {
            double p = price_to_double(mkt_price);
            unrealized_pnl = (p - avg_cost) * net_qty;
        }
        last_price = mkt_price;
    }

    double total_pnl() const noexcept { return realized_pnl + unrealized_pnl; }
};

// Token bucket for order rate limiting
class TokenBucket {
public:
    explicit TokenBucket(double rate, double burst)
        : rate_(rate), burst_(burst), tokens_(burst),
          last_refill_(std::chrono::high_resolution_clock::now().time_since_epoch().count())
    {}

    bool consume(double n = 1.0) noexcept {
        refill();
        if (tokens_ >= n) { tokens_ -= n; return true; }
        return false;
    }

    double available() const noexcept { return tokens_; }

private:
    void refill() noexcept {
        int64_t now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        double dt = (now - last_refill_) * 1e-9; // seconds
        tokens_ = std::min(burst_, tokens_ + rate_ * dt);
        last_refill_ = now;
    }

    double  rate_;
    double  burst_;
    double  tokens_;
    int64_t last_refill_;
};

class RiskManager {
public:
    explicit RiskManager(const RiskLimits& limits = {})
        : limits_(limits),
          order_rate_bucket_(limits.max_order_rate, limits.max_order_rate * 2)
    {}

    // Returns RejectionReason::None if order passes all checks
    RejectionReason check_order(const Order& order) noexcept {
        if (limits_.kill_switch) return RejectionReason::KillSwitch;

        // 1. Order size
        if (order.qty > limits_.max_order_size) return RejectionReason::MaxOrderSize;

        // 2. Notional
        double notional = order.qty * price_to_double(order.price);
        if (notional > limits_.max_notional) return RejectionReason::MaxNotional;

        // 3. Order rate
        if (!order_rate_bucket_.consume()) return RejectionReason::MaxOrderRate;

        // 4. Position size
        auto pos_it = positions_.find(order.symbol);
        if (pos_it != positions_.end()) {
            const auto& pos = pos_it->second;
            int64_t new_pos = pos.net_qty;
            if (order.side == Side::Buy)  new_pos += static_cast<int64_t>(order.qty);
            else                          new_pos -= static_cast<int64_t>(order.qty);
            if (std::abs(new_pos) > static_cast<int64_t>(limits_.max_position_size))
                return RejectionReason::MaxPositionSize;
        }

        // 5. Price band check (only for limit orders with reference price)
        if (order.order_type == OrderType::Limit) {
            auto ref_it = ref_prices_.find(order.symbol);
            if (ref_it != ref_prices_.end() && ref_it->second > 0) {
                double ref = price_to_double(ref_it->second);
                double p   = price_to_double(order.price);
                if (std::fabs(p - ref) / ref > limits_.price_band_pct)
                    return RejectionReason::PriceBandViolation;
            }
        }

        // 6. Daily volume
        auto vol_it = daily_volume_.find(order.symbol);
        if (vol_it != daily_volume_.end() && vol_it->second >= limits_.max_daily_volume)
            return RejectionReason::MaxDailyVolume;

        return RejectionReason::None;
    }

    void on_fill(const char* symbol, Side side, Quantity qty, Price price) noexcept {
        auto& pos = positions_[symbol];
        pos.on_fill(side, qty, price);
        auto& vol = daily_volume_[symbol];
        vol += qty;
        ++total_fills_;
        total_fill_volume_ += qty;
    }

    void set_reference_price(const char* symbol, Price p) noexcept {
        ref_prices_[symbol] = p;
    }

    void set_kill_switch(bool v) noexcept { limits_.kill_switch = v; }
    bool kill_switch() const noexcept { return limits_.kill_switch; }

    const PositionState* get_position(const char* symbol) const noexcept {
        auto it = positions_.find(symbol);
        return it != positions_.end() ? &it->second : nullptr;
    }

    void reset_daily() noexcept {
        daily_volume_.clear();
        for (auto& [sym, pos] : positions_) {
            pos.bought_today = 0;
            pos.sold_today   = 0;
        }
    }

    void update_pnl(const char* symbol, Price mkt_price) noexcept {
        auto it = positions_.find(symbol);
        if (it != positions_.end()) it->second.mark_to_market(mkt_price);
    }

    double total_realized_pnl() const noexcept {
        double total = 0.0;
        for (auto& [sym, pos] : positions_) total += pos.realized_pnl;
        return total;
    }
    double total_unrealized_pnl() const noexcept {
        double total = 0.0;
        for (auto& [sym, pos] : positions_) total += pos.unrealized_pnl;
        return total;
    }

    uint64_t total_fills() const noexcept { return total_fills_; }
    uint64_t total_fill_volume() const noexcept { return total_fill_volume_; }

private:
    RiskLimits  limits_;
    TokenBucket order_rate_bucket_;
    std::unordered_map<std::string, PositionState> positions_;
    std::unordered_map<std::string, Quantity>      daily_volume_;
    std::unordered_map<std::string, Price>         ref_prices_;
    uint64_t total_fills_       = 0;
    uint64_t total_fill_volume_ = 0;
};

} // namespace hft
