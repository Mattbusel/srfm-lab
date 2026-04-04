#pragma once
// Order flow analytics: Tick Rule, Lee-Ready algorithm, VPIN (Volume-Synchronized
// Probability of Informed Trading), order flow imbalance, and toxicity metrics.

#include "order.hpp"
#include "orderbook.hpp"
#include <deque>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <string>

namespace hft {

// ============================================================
// Tick Rule trade classification
// Classifies trades as buyer- or seller-initiated
// ============================================================
enum class TradeInitiator : int8_t {
    Unknown  = 0,
    Buyer    = 1,   // trade at ask (uptick)
    Seller   = -1,  // trade at bid (downtick)
};

class TickRuleClassifier {
public:
    TradeInitiator classify(Price trade_price) {
        if (last_price_ == 0) { last_price_ = trade_price; return TradeInitiator::Unknown; }

        if (trade_price > last_price_) {
            last_price_ = trade_price;
            last_direction_ = TradeInitiator::Buyer;
        } else if (trade_price < last_price_) {
            last_price_ = trade_price;
            last_direction_ = TradeInitiator::Seller;
        }
        // If equal: use previous direction (tick continuation)
        return last_direction_;
    }

    void reset() { last_price_ = 0; last_direction_ = TradeInitiator::Unknown; }
    TradeInitiator last() const noexcept { return last_direction_; }

private:
    Price last_price_             = 0;
    TradeInitiator last_direction_ = TradeInitiator::Unknown;
};

// ============================================================
// Lee-Ready algorithm: combines quote rule and tick rule
// Quote rule: if trade > mid → buyer; if < mid → seller
// Reversal rule: use tick rule when trade == mid
// ============================================================
class LeeReadyClassifier {
public:
    TradeInitiator classify(Price trade_price, Price bid, Price ask) {
        Price mid = (bid + ask) / 2;
        if (trade_price > mid) return TradeInitiator::Buyer;
        if (trade_price < mid) return TradeInitiator::Seller;
        // At midpoint: use tick rule
        return tick_.classify(trade_price);
    }

    void reset() { tick_.reset(); }

private:
    TickRuleClassifier tick_;
};

// ============================================================
// VPIN (Volume-Synchronized Probability of Informed Trading)
// Easley-López de Prado-O'Hara (2012)
// Estimates the proportion of informed trades in a sample
//
// Algorithm:
//  1. Split volume into equal-sized buckets
//  2. Classify trades in each bucket as buy/sell
//  3. VPIN = avg(|V_buy - V_sell|) / V_bucket
// ============================================================
class VPINEstimator {
public:
    struct Config {
        size_t   sample_buckets = 50;   // number of buckets for VPIN estimate
        uint64_t bucket_volume  = 10000; // shares per bucket
    };

    explicit VPINEstimator(const Config& cfg = {}) : cfg_(cfg) {}

    // Update with a trade
    void update(Price trade_price, Price bid, Price ask, Quantity qty) {
        auto dir = lr_.classify(trade_price, bid, ask);

        double buy_frac  = 0.0;
        double sell_frac = 0.0;

        if (dir == TradeInitiator::Buyer) {
            buy_frac  = 1.0;
        } else if (dir == TradeInitiator::Seller) {
            sell_frac = 1.0;
        } else {
            buy_frac = sell_frac = 0.5;
        }

        uint64_t remaining = qty;
        while (remaining > 0) {
            uint64_t space = cfg_.bucket_volume - current_bucket_vol_;
            uint64_t fill  = std::min(remaining, space);
            remaining -= fill;
            current_bucket_vol_ += fill;
            current_buy_vol_  += static_cast<double>(fill) * buy_frac;
            current_sell_vol_ += static_cast<double>(fill) * sell_frac;

            if (current_bucket_vol_ >= cfg_.bucket_volume) {
                // Close bucket
                double imb = std::fabs(current_buy_vol_ - current_sell_vol_);
                imbalances_.push_back(imb);
                if (imbalances_.size() > cfg_.sample_buckets)
                    imbalances_.pop_front();
                total_buckets_++;
                current_bucket_vol_  = 0;
                current_buy_vol_     = 0.0;
                current_sell_vol_    = 0.0;
            }
        }
    }

    double vpin() const {
        if (imbalances_.empty()) return 0.0;
        double sum = std::accumulate(imbalances_.begin(), imbalances_.end(), 0.0);
        return sum / (imbalances_.size() * cfg_.bucket_volume);
    }

    bool   is_ready() const noexcept { return imbalances_.size() >= cfg_.sample_buckets / 2; }
    size_t buckets_completed() const noexcept { return total_buckets_; }

    // Alert threshold: VPIN above this indicates toxic flow
    bool is_toxic(double threshold = 0.5) const { return is_ready() && vpin() > threshold; }

private:
    Config              cfg_;
    LeeReadyClassifier  lr_;
    std::deque<double>  imbalances_;
    uint64_t            current_bucket_vol_ = 0;
    double              current_buy_vol_    = 0.0;
    double              current_sell_vol_   = 0.0;
    size_t              total_buckets_      = 0;
};

// ============================================================
// Order Flow Imbalance (OFI)
// Measures the net order book pressure
// OFI = sum(bid_size_change) - sum(ask_size_change) over window
// ============================================================
class OrderFlowImbalance {
public:
    explicit OrderFlowImbalance(size_t window = 100) : window_(window) {}

    struct BookSnapshot {
        Price    bid_px;
        Quantity bid_qty;
        Price    ask_px;
        Quantity ask_qty;
    };

    // Update with consecutive book snapshots
    void update(const BookSnapshot& prev, const BookSnapshot& curr) {
        double bid_delta = 0.0, ask_delta = 0.0;

        // Bid side contribution
        if (curr.bid_px == prev.bid_px)
            bid_delta = static_cast<double>(curr.bid_qty) - static_cast<double>(prev.bid_qty);
        else if (curr.bid_px > prev.bid_px)
            bid_delta = static_cast<double>(curr.bid_qty);
        else
            bid_delta = -static_cast<double>(prev.bid_qty);

        // Ask side contribution
        if (curr.ask_px == prev.ask_px)
            ask_delta = -(static_cast<double>(curr.ask_qty) - static_cast<double>(prev.ask_qty));
        else if (curr.ask_px < prev.ask_px)
            ask_delta = -static_cast<double>(curr.ask_qty);
        else
            ask_delta = static_cast<double>(prev.ask_qty);

        double ofi = bid_delta + ask_delta;
        history_.push_back(ofi);
        if (history_.size() > window_) history_.pop_front();
        cumulative_ofi_ += ofi;
    }

    double current_ofi() const {
        return history_.empty() ? 0.0 : history_.back();
    }
    double rolling_ofi() const {
        if (history_.empty()) return 0.0;
        return std::accumulate(history_.begin(), history_.end(), 0.0);
    }
    double normalized_ofi() const {
        double sum = 0.0;
        double sum_abs = 0.0;
        for (auto v : history_) { sum += v; sum_abs += std::fabs(v); }
        return sum_abs > 0 ? sum / sum_abs : 0.0;
    }
    double cumulative() const noexcept { return cumulative_ofi_; }

private:
    size_t             window_;
    std::deque<double> history_;
    double             cumulative_ofi_ = 0.0;
};

// ============================================================
// Trade Toxicity Index (composite indicator)
// Combines VPIN, OFI, and spread widening
// ============================================================
struct ToxicityIndicator {
    double vpin;
    double ofi;
    double spread_percentile;  // where current spread is in historical distribution
    double composite_score;    // weighted average
    bool   is_toxic;

    static ToxicityIndicator compute(
        const VPINEstimator& vpin_est,
        const OrderFlowImbalance& ofi_est,
        double current_spread, double median_spread,
        double vpin_w = 0.4, double ofi_w = 0.3, double spread_w = 0.3)
    {
        ToxicityIndicator t{};
        t.vpin = vpin_est.vpin();
        t.ofi  = std::fabs(ofi_est.normalized_ofi());
        t.spread_percentile = median_spread > 0 ? current_spread / median_spread : 1.0;

        // Normalize spread percentile to [0,1]
        double sp_norm = std::min(t.spread_percentile - 1.0, 1.0);
        sp_norm        = std::max(sp_norm, 0.0);

        t.composite_score = vpin_w * t.vpin + ofi_w * t.ofi + spread_w * sp_norm;
        t.is_toxic        = t.composite_score > 0.5;
        return t;
    }
};

// ============================================================
// Order Toxicity Monitor: wraps all the above
// ============================================================
class ToxicityMonitor {
public:
    explicit ToxicityMonitor(const std::string& symbol,
                               VPINEstimator::Config vpin_cfg = {})
        : symbol_(symbol), vpin_(vpin_cfg)
    {}

    void on_trade(Price price, Price bid, Price ask, Quantity qty, Timestamp /*ts*/) {
        vpin_.update(price, bid, ask, qty);
        tick_.classify(price);

        // Update OFI
        OrderFlowImbalance::BookSnapshot curr{bid, qty/2, ask, qty/2};
        if (prev_snap_.bid_px > 0) ofi_.update(prev_snap_, curr);
        prev_snap_ = curr;

        // Track spread distribution
        double sp = price_to_double(ask) - price_to_double(bid);
        spread_history_.push_back(sp);
        if (spread_history_.size() > 500) spread_history_.pop_front();

        ++update_count_;
    }

    ToxicityIndicator assess() const {
        double spread = prev_snap_.bid_px > 0 ?
            price_to_double(prev_snap_.ask_px) - price_to_double(prev_snap_.bid_px) : 0.0;
        double median_spread = 0.01;
        if (!spread_history_.empty()) {
            auto sorted = spread_history_;
            std::sort(sorted.begin(), sorted.end());
            median_spread = sorted[sorted.size()/2];
        }
        return ToxicityIndicator::compute(vpin_, ofi_, spread, median_spread);
    }

    double vpin() const { return vpin_.vpin(); }
    bool   is_toxic(double threshold = 0.5) const { return vpin_.is_toxic(threshold); }
    uint64_t updates() const noexcept { return update_count_; }
    const std::string& symbol() const noexcept { return symbol_; }

private:
    std::string         symbol_;
    VPINEstimator       vpin_;
    LeeReadyClassifier  lr_;
    TickRuleClassifier  tick_;
    OrderFlowImbalance  ofi_;
    OrderFlowImbalance::BookSnapshot prev_snap_{};
    std::deque<double>  spread_history_;
    uint64_t            update_count_ = 0;
};

// ============================================================
// Aggregate order flow metrics across multiple symbols
// ============================================================
class MarketWideFlow {
public:
    struct SymbolFlow {
        std::string    symbol;
        double         vpin;
        double         ofi;
        double         net_flow;   // buy_vol - sell_vol (last window)
        double         turnover;
        bool           is_toxic;
        uint64_t       trades;
    };

    void update(const std::string& sym,
                Price price, Price bid, Price ask, Quantity qty,
                bool is_buy)
    {
        auto& mon = monitors_[sym];
        if (!mon) mon = std::make_unique<ToxicityMonitor>(sym);
        mon->on_trade(price, bid, ask, qty, 0);

        auto& flow = flows_[sym];
        flow.symbol    = sym;
        flow.vpin      = mon->vpin();
        flow.ofi       = 0.0; // simplified
        double net     = is_buy ? static_cast<double>(qty) : -static_cast<double>(qty);
        flow.net_flow += net;
        flow.turnover += price_to_double(price) * qty;
        flow.trades++;
        flow.is_toxic  = mon->is_toxic();
    }

    std::vector<SymbolFlow> most_toxic(size_t n = 5) const {
        std::vector<SymbolFlow> v;
        for (auto& [sym, f] : flows_) v.push_back(f);
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b){ return a.vpin > b.vpin; });
        if (v.size() > n) v.resize(n);
        return v;
    }

    size_t symbol_count() const noexcept { return monitors_.size(); }

private:
    std::unordered_map<std::string, std::unique_ptr<ToxicityMonitor>> monitors_;
    std::unordered_map<std::string, SymbolFlow> flows_;
};

} // namespace hft
