#include "execution_analytics.hpp"
#include <cstring>
#include <cfloat>

namespace srfm::execution {

// ============================================================================
// ImplementationShortfall
// ============================================================================

void ImplementationShortfall::add_fill(const Fill& fill, const OrderInfo& /*order*/) {
    fills_.push_back(fill);
}

ISResult ImplementationShortfall::compute(const OrderInfo& order) const {
    ISResult r{};
    if (fills_.empty() || order.target_quantity <= 0) return r;

    r.arrival_price = order.arrival_mid;
    double total_qty = 0, total_cost = 0;
    for (const auto& f : fills_) {
        total_qty += f.quantity;
        total_cost += f.price * f.quantity;
    }
    r.avg_execution_price = (total_qty > 0) ? total_cost / total_qty : 0;

    int sign = (order.side == Side::Buy) ? 1 : -1;
    double shortfall = sign * (r.avg_execution_price - r.arrival_price);
    r.shortfall_bps = (r.arrival_price > 0) ? shortfall / r.arrival_price * 10000.0 : 0;
    r.shortfall_dollars = shortfall * total_qty;
    r.realized_gain_loss = -r.shortfall_dollars;

    // Delay cost: from decision to first fill
    if (!fills_.empty() && order.submit_time_ns > 0) {
        double first_fill_price = fills_.front().price;
        double delay = sign * (first_fill_price - order.decision_price);
        r.delay_cost_bps = (order.decision_price > 0) ? delay / order.decision_price * 10000.0 : 0;
    }

    // Trading cost: from arrival to execution
    r.trading_cost_bps = r.shortfall_bps - r.delay_cost_bps;

    // Opportunity cost for unfilled portion
    double unfilled = order.target_quantity - total_qty;
    if (unfilled > 0 && r.arrival_price > 0) {
        // Assume market moved in the direction of the trade
        r.opportunity_cost_bps = unfilled / order.target_quantity * std::abs(r.shortfall_bps) * 0.5;
    }

    return r;
}

void ImplementationShortfall::reset() { fills_.clear(); }

// ============================================================================
// VWAPTracker
// ============================================================================

void VWAPTracker::add_fill(const Fill& fill) { fills_.push_back(fill); }
void VWAPTracker::add_market_bar(const MarketBar& bar) { bars_.push_back(bar); }

VWAPResult VWAPTracker::compute() const {
    VWAPResult r{};
    if (fills_.empty()) return r;

    double fill_pv = 0, fill_qty = 0;
    for (const auto& f : fills_) {
        fill_pv += f.price * f.quantity;
        fill_qty += f.quantity;
    }
    r.execution_vwap = (fill_qty > 0) ? fill_pv / fill_qty : 0;
    r.total_quantity = fill_qty;

    double mkt_pv = 0, mkt_vol = 0;
    for (const auto& b : bars_) {
        mkt_pv += b.vwap * b.volume;
        mkt_vol += b.volume;
    }
    r.market_vwap = (mkt_vol > 0) ? mkt_pv / mkt_vol : r.execution_vwap;
    r.market_volume = mkt_vol;
    r.participation_rate = (mkt_vol > 0) ? fill_qty / mkt_vol : 0;

    r.slippage_bps = (r.market_vwap > 0) ? (r.execution_vwap - r.market_vwap) / r.market_vwap * 10000.0 : 0;

    return r;
}

void VWAPTracker::reset() { fills_.clear(); bars_.clear(); }

// ============================================================================
// MarketImpact
// ============================================================================

MarketImpact::MarketImpact() : config_(), pre_trade_mid_(0), post_trade_mid_(0) {}
MarketImpact::MarketImpact(const Config& cfg) : config_(cfg), pre_trade_mid_(0), post_trade_mid_(0) {}

void MarketImpact::add_fill(const Fill& fill) { fills_.push_back(fill); }
void MarketImpact::add_market_bar(const MarketBar& bar) { bars_.push_back(bar); }

ImpactResult MarketImpact::compute(Side side) const {
    ImpactResult r{};
    if (fills_.empty() || pre_trade_mid_ <= 0) return r;

    double total_qty = 0, total_cost = 0;
    for (const auto& f : fills_) {
        total_qty += f.quantity;
        total_cost += f.price * f.quantity;
    }
    double exec_vwap = (total_qty > 0) ? total_cost / total_qty : 0;

    int sign = (side == Side::Buy) ? 1 : -1;

    // Total impact: pre-trade to post-trade mid price change
    double total_move = sign * (post_trade_mid_ - pre_trade_mid_);
    r.total_impact_bps = total_move / pre_trade_mid_ * 10000.0;

    // Realized impact: exec price vs pre-trade
    double realized = sign * (exec_vwap - pre_trade_mid_);
    r.realized_impact_bps = realized / pre_trade_mid_ * 10000.0;

    // Permanent impact: post-trade mid vs pre-trade mid (after reversion)
    double reversion_mid = post_trade_mid_;
    if (static_cast<int>(bars_.size()) >= config_.reversion_window_bars) {
        // Use mid price N bars after trade completion
        int idx = std::min(config_.reversion_window_bars, static_cast<int>(bars_.size()) - 1);
        reversion_mid = (bars_[idx].high + bars_[idx].low) / 2.0;
    }
    double permanent = sign * (reversion_mid - pre_trade_mid_);
    r.permanent_impact_bps = permanent / pre_trade_mid_ * 10000.0;

    // Temporary = total - permanent
    r.temporary_impact_bps = r.total_impact_bps - r.permanent_impact_bps;

    // Post-trade reversion
    r.post_trade_reversion = r.total_impact_bps - r.permanent_impact_bps;

    // Participation rate
    double mkt_vol = 0;
    for (const auto& b : bars_) mkt_vol += b.volume;
    r.participation_rate = (mkt_vol > 0) ? total_qty / mkt_vol : 0;

    return r;
}

void MarketImpact::reset() {
    fills_.clear(); bars_.clear();
    pre_trade_mid_ = post_trade_mid_ = 0;
}

// ============================================================================
// SpreadCapture
// ============================================================================

void SpreadCapture::add_fill(const Fill& fill, double mid_at_fill) {
    fills_.push_back({fill, mid_at_fill});
}

SpreadResult SpreadCapture::compute() const {
    SpreadResult r{};
    r.n_fills = static_cast<int>(fills_.size());
    if (fills_.empty()) return r;

    double sum_quoted = 0, sum_effective = 0, sum_realized = 0;
    for (const auto& fm : fills_) {
        sum_quoted += fm.fill.quoted_spread;

        // Effective spread: 2 * |fill_price - mid|
        double eff = 2.0 * std::abs(fm.fill.price - fm.mid);
        sum_effective += eff;

        // Realized spread: for market maker perspective
        int sign = (fm.fill.side == Side::Buy) ? 1 : -1;
        double realized = 2.0 * sign * (fm.fill.price - fm.mid);
        sum_realized += realized;
    }

    r.avg_quoted_spread_bps = sum_quoted / r.n_fills;
    r.effective_spread_bps = sum_effective / r.n_fills;
    r.avg_realized_spread_bps = sum_realized / r.n_fills;
    r.spread_capture_ratio = (r.avg_quoted_spread_bps > 0) ? r.avg_realized_spread_bps / r.avg_quoted_spread_bps : 0;

    // Convert to bps if not already
    if (fills_[0].mid > 0) {
        double ref = fills_[0].mid;
        r.avg_quoted_spread_bps = r.avg_quoted_spread_bps / ref * 10000.0;
        r.effective_spread_bps = r.effective_spread_bps / ref * 10000.0;
        r.avg_realized_spread_bps = r.avg_realized_spread_bps / ref * 10000.0;
    }

    return r;
}

void SpreadCapture::reset() { fills_.clear(); }

// ============================================================================
// FillRateAnalyzer
// ============================================================================

void FillRateAnalyzer::add_order(const OrderInfo& order) { orders_.push_back(order); }

FillRateResult FillRateAnalyzer::compute() const {
    FillRateResult r{};
    r.total_orders = static_cast<int>(orders_.size());
    if (orders_.empty()) return r;

    double total_target = 0, total_filled = 0;
    std::vector<double> fill_times;
    int total_fills = 0, total_rejects = 0;

    for (const auto& o : orders_) {
        total_target += o.target_quantity;
        total_filled += o.filled_quantity;
        total_fills += o.num_fills;
        total_rejects += o.num_rejects;

        if (o.filled_quantity >= o.target_quantity - 1e-10) {
            ++r.fully_filled;
        } else if (o.filled_quantity > 0) {
            ++r.partially_filled;
        } else if (o.num_rejects > 0) {
            ++r.rejected;
        }

        if (o.num_fills > 0 && o.last_fill_time_ns > o.submit_time_ns) {
            double ms = static_cast<double>(o.last_fill_time_ns - o.submit_time_ns) / 1e6;
            fill_times.push_back(ms);
        }
    }

    r.fill_rate = (total_target > 0) ? total_filled / total_target : 0;
    r.partial_fill_rate = static_cast<double>(r.partially_filled) / r.total_orders;
    r.rejection_rate = static_cast<double>(total_rejects) / (total_fills + total_rejects + 1);
    r.avg_fills_per_order = static_cast<double>(total_fills) / r.total_orders;

    if (!fill_times.empty()) {
        double sum = 0;
        for (auto t : fill_times) sum += t;
        r.avg_fill_time_ms = sum / fill_times.size();

        std::vector<double> sorted = fill_times;
        std::sort(sorted.begin(), sorted.end());
        r.median_fill_time_ms = sorted[sorted.size() / 2];
    }

    return r;
}

void FillRateAnalyzer::reset() { orders_.clear(); }

// ============================================================================
// TimingAnalyzer
// ============================================================================

void TimingAnalyzer::add_fill(const Fill& fill, uint64_t order_start_ns, uint64_t order_end_ns) {
    double duration = static_cast<double>(order_end_ns - order_start_ns);
    double norm_time = (duration > 0) ? static_cast<double>(fill.timestamp_ns - order_start_ns) / duration : 0.5;
    norm_time = std::clamp(norm_time, 0.0, 1.0);
    fills_.push_back({fill, norm_time});
}

TimingResult TimingAnalyzer::compute() const {
    TimingResult r{};
    if (fills_.empty()) return r;

    double total_qty = 0;
    for (const auto& f : fills_) total_qty += f.fill.quantity;

    // Time-weighted slippage
    double tw_slip = 0;
    for (const auto& f : fills_) {
        if (arrival_price_ > 0) {
            double slip = (f.fill.price - arrival_price_) / arrival_price_ * 10000.0;
            tw_slip += slip * f.fill.quantity / total_qty;
        }
    }
    r.time_weighted_slippage_bps = tw_slip;

    // Front/back loading
    double front_qty = 0, back_qty = 0;
    for (const auto& f : fills_) {
        if (f.normalized_time < 1.0 / 3.0) front_qty += f.fill.quantity;
        else if (f.normalized_time > 2.0 / 3.0) back_qty += f.fill.quantity;
    }
    r.front_loading_ratio = (total_qty > 0) ? front_qty / total_qty : 0;
    r.back_loading_ratio = (total_qty > 0) ? back_qty / total_qty : 0;

    // Urgency: how front-loaded
    double weighted_time = 0;
    for (const auto& f : fills_) {
        weighted_time += f.normalized_time * f.fill.quantity / total_qty;
    }
    r.urgency_score = 1.0 - weighted_time; // 1 = all at start, 0 = all at end

    // Consistency: coefficient of variation of fill times
    double mean_t = 0;
    for (const auto& f : fills_) mean_t += f.normalized_time;
    mean_t /= fills_.size();
    double var_t = 0;
    for (const auto& f : fills_) {
        double d = f.normalized_time - mean_t;
        var_t += d * d;
    }
    var_t /= fills_.size();
    double ideal_var = 1.0 / 12.0; // variance of uniform on [0,1]
    r.consistency_score = 1.0 - std::min(var_t / ideal_var, 1.0);

    return r;
}

void TimingAnalyzer::reset() { fills_.clear(); arrival_price_ = 0; }

// ============================================================================
// CostDecomposer
// ============================================================================

CostDecomposer::CostDecomposer() : config_(), final_mid_(0), has_order_(false) {}
CostDecomposer::CostDecomposer(const Config& cfg) : config_(cfg), final_mid_(0), has_order_(false) {}

void CostDecomposer::set_order(const OrderInfo& order) {
    order_ = order;
    has_order_ = true;
}

void CostDecomposer::add_fill(const Fill& fill, double mid_at_fill, double market_vwap_at_fill) {
    fills_.push_back({fill, mid_at_fill, market_vwap_at_fill});
}

CostDecomp CostDecomposer::compute() const {
    CostDecomp r{};
    if (!has_order_ || fills_.empty()) return r;

    double total_qty = 0, total_cost = 0;
    double spread_cost = 0, impact_cost = 0, timing_cost = 0;
    int sign = (order_.side == Side::Buy) ? 1 : -1;

    for (const auto& cf : fills_) {
        double qty = cf.fill.quantity;
        total_qty += qty;
        total_cost += cf.fill.price * qty;

        // Spread cost: half-spread * quantity
        double half_spread = std::abs(cf.fill.price - cf.mid);
        spread_cost += half_spread * qty;

        // Impact cost: execution price vs market VWAP at that time
        double impact = sign * (cf.fill.price - cf.market_vwap);
        impact_cost += std::max(impact, 0.0) * qty;

        // Timing: market VWAP vs arrival price
        double timing = sign * (cf.market_vwap - order_.arrival_mid);
        timing_cost += timing * qty;
    }

    double ref_price = order_.arrival_mid;
    if (ref_price <= 0) ref_price = 1.0;

    r.spread_cost_bps = spread_cost / (total_qty * ref_price) * 10000.0;
    r.impact_cost_bps = impact_cost / (total_qty * ref_price) * 10000.0;
    r.timing_cost_bps = timing_cost / (total_qty * ref_price) * 10000.0;

    // Opportunity cost for unfilled
    double unfilled = order_.target_quantity - total_qty;
    if (unfilled > 0 && final_mid_ > 0) {
        double opp = sign * (final_mid_ - ref_price) * unfilled;
        r.opportunity_cost_bps = std::max(opp, 0.0) / (order_.target_quantity * ref_price) * 10000.0;
    }

    // Commission
    r.commission_bps = config_.commission_per_share / ref_price * 10000.0;

    r.total_cost_bps = r.spread_cost_bps + r.impact_cost_bps + r.timing_cost_bps
                     + r.opportunity_cost_bps + r.commission_bps;

    return r;
}

void CostDecomposer::reset() {
    fills_.clear(); final_mid_ = 0; has_order_ = false;
}

// ============================================================================
// TCAEngine
// ============================================================================

void TCAEngine::add_trade_report(const TCATradeReport& report) { reports_.push_back(report); }

TCAAggregateReport TCAEngine::aggregate() const {
    TCAAggregateReport r{};
    r.n_trades = static_cast<int>(reports_.size());
    if (reports_.empty()) return r;

    double sum_is = 0, sum_vwap = 0, sum_spread = 0, sum_cost = 0, sum_dollar = 0;
    r.worst_is_bps = -1e18;
    r.best_is_bps = 1e18;

    for (const auto& tr : reports_) {
        sum_is += tr.implementation_shortfall.shortfall_bps;
        sum_vwap += tr.vwap.slippage_bps;
        sum_spread += tr.spread.spread_capture_ratio;
        sum_cost += tr.cost.total_cost_bps;
        sum_dollar += tr.implementation_shortfall.shortfall_dollars;

        r.worst_is_bps = std::max(r.worst_is_bps, tr.implementation_shortfall.shortfall_bps);
        r.best_is_bps = std::min(r.best_is_bps, tr.implementation_shortfall.shortfall_bps);
    }

    int n = r.n_trades;
    r.avg_is_bps = sum_is / n;
    r.avg_vwap_slippage_bps = sum_vwap / n;
    r.avg_spread_capture = sum_spread / n;
    r.avg_total_cost_bps = sum_cost / n;
    r.total_dollar_cost = sum_dollar;

    // Std dev of IS
    double var = 0;
    for (const auto& tr : reports_) {
        double d = tr.implementation_shortfall.shortfall_bps - r.avg_is_bps;
        var += d * d;
    }
    r.is_std_dev = std::sqrt(var / n);

    return r;
}

void TCAEngine::reset() { reports_.clear(); }

// ============================================================================
// Execution Simulator
// ============================================================================

class ExecutionSimulator {
public:
    struct Config {
        double permanent_impact_coeff = 0.1;   // sqrt model: lambda * sigma * sqrt(Q/V)
        double temporary_impact_coeff = 0.05;
        double spread_bps = 2.0;
        double fill_probability = 0.85;
        double latency_ms = 1.0;
        double avg_daily_volume = 1e6;
        double daily_volatility = 0.02;
    };

    ExecutionSimulator() : config_() {}
    explicit ExecutionSimulator(const Config& cfg) : config_(cfg) {}

    struct SimFill {
        double price;
        double quantity;
        double slippage_bps;
        double impact_bps;
        bool filled;
    };

    SimFill simulate_market_order(Side side, double quantity, double mid_price) const {
        SimFill f{};
        f.filled = true;
        f.quantity = quantity;

        // Spread cost
        double half_spread = mid_price * config_.spread_bps / 10000.0 / 2.0;
        int sign = (side == Side::Buy) ? 1 : -1;

        // Temporary impact: linear model
        double participation = quantity / config_.avg_daily_volume;
        double temp_impact = config_.temporary_impact_coeff * config_.daily_volatility * mid_price
                           * std::sqrt(participation);

        // Permanent impact
        double perm_impact = config_.permanent_impact_coeff * config_.daily_volatility * mid_price
                           * std::sqrt(participation);

        double total_slip = half_spread + temp_impact;
        f.price = mid_price + sign * total_slip;
        f.slippage_bps = total_slip / mid_price * 10000.0;
        f.impact_bps = (temp_impact + perm_impact) / mid_price * 10000.0;

        return f;
    }

    SimFill simulate_limit_order(Side side, double quantity, double limit_price, double mid_price) const {
        SimFill f{};
        int sign = (side == Side::Buy) ? 1 : -1;
        double edge = sign * (mid_price - limit_price);

        // Fill probability depends on how aggressive the limit price is
        double fill_prob = config_.fill_probability;
        if (edge > 0) {
            // Passive order: lower fill probability
            fill_prob *= std::exp(-edge / (mid_price * config_.daily_volatility) * 5.0);
        } else {
            // Aggressive: almost certain fill
            fill_prob = std::min(fill_prob * 1.5, 1.0);
        }

        f.filled = (fill_prob > 0.5); // deterministic threshold for simulation
        if (f.filled) {
            f.quantity = quantity;
            f.price = limit_price;
            f.slippage_bps = sign * (limit_price - mid_price) / mid_price * 10000.0;
            f.impact_bps = 0; // limit orders don't cause impact (maker)
        }
        return f;
    }

    // Simulate TWAP execution
    struct TWAPResult {
        double avg_price;
        double total_quantity;
        double total_slippage_bps;
        int n_slices;
        int n_filled;
    };

    TWAPResult simulate_twap(Side side, double total_quantity, double start_mid,
                             int n_slices, double price_drift_per_slice) const {
        TWAPResult r{};
        r.n_slices = n_slices;
        double slice_qty = total_quantity / n_slices;
        double current_mid = start_mid;
        double total_cost = 0;

        for (int i = 0; i < n_slices; ++i) {
            auto fill = simulate_market_order(side, slice_qty, current_mid);
            if (fill.filled) {
                total_cost += fill.price * fill.quantity;
                r.total_quantity += fill.quantity;
                ++r.n_filled;
            }
            current_mid += price_drift_per_slice;
        }

        r.avg_price = (r.total_quantity > 0) ? total_cost / r.total_quantity : 0;
        r.total_slippage_bps = (start_mid > 0 && r.total_quantity > 0)
            ? ((side == Side::Buy ? 1 : -1) * (r.avg_price - start_mid)) / start_mid * 10000.0
            : 0;
        return r;
    }

    // Simulate VWAP execution
    TWAPResult simulate_vwap(Side side, double total_quantity, double start_mid,
                             const std::vector<double>& volume_profile) const {
        TWAPResult r{};
        int n = static_cast<int>(volume_profile.size());
        r.n_slices = n;
        double total_vol = 0;
        for (auto v : volume_profile) total_vol += v;
        if (total_vol <= 0 || n == 0) return r;

        double current_mid = start_mid;
        double total_cost = 0;

        for (int i = 0; i < n; ++i) {
            double slice_qty = total_quantity * volume_profile[i] / total_vol;
            auto fill = simulate_market_order(side, slice_qty, current_mid);
            if (fill.filled) {
                total_cost += fill.price * fill.quantity;
                r.total_quantity += fill.quantity;
                ++r.n_filled;
            }
            current_mid *= (1.0 + config_.daily_volatility / std::sqrt(252.0 * n) * 0.1);
        }

        r.avg_price = (r.total_quantity > 0) ? total_cost / r.total_quantity : 0;
        r.total_slippage_bps = (start_mid > 0 && r.total_quantity > 0)
            ? ((side == Side::Buy ? 1 : -1) * (r.avg_price - start_mid)) / start_mid * 10000.0
            : 0;
        return r;
    }

private:
    Config config_;
};

// ============================================================================
// Order Flow Toxicity (VPIN)
// ============================================================================

class VPINEstimator {
public:
    struct Config {
        int bucket_size = 50;       // volume per bucket
        int n_buckets = 50;         // rolling window
        double sigma = 0.02;        // daily vol for CDF
    };

    VPINEstimator() : config_(), current_bucket_vol_(0), current_buy_vol_(0) {}
    explicit VPINEstimator(const Config& cfg) : config_(cfg), current_bucket_vol_(0), current_buy_vol_(0) {}

    void add_trade(double price, double quantity, double prev_close) {
        // Classify using tick rule: if price > prev_close, it's a buy
        double buy_vol = 0;
        if (price > prev_close) buy_vol = quantity;
        else if (price < prev_close) buy_vol = 0;
        else buy_vol = quantity * 0.5; // at mid: split

        current_buy_vol_ += buy_vol;
        current_bucket_vol_ += quantity;

        if (current_bucket_vol_ >= config_.bucket_size) {
            double sell_vol = current_bucket_vol_ - current_buy_vol_;
            double imbalance = std::abs(current_buy_vol_ - sell_vol);
            bucket_imbalances_.push_back(imbalance);
            bucket_volumes_.push_back(current_bucket_vol_);

            if (static_cast<int>(bucket_imbalances_.size()) > config_.n_buckets) {
                bucket_imbalances_.pop_front();
                bucket_volumes_.pop_front();
            }

            current_bucket_vol_ = 0;
            current_buy_vol_ = 0;
        }
    }

    double vpin() const {
        if (bucket_imbalances_.empty()) return 0;
        double sum_imbalance = 0, sum_vol = 0;
        for (size_t i = 0; i < bucket_imbalances_.size(); ++i) {
            sum_imbalance += bucket_imbalances_[i];
            sum_vol += bucket_volumes_[i];
        }
        return (sum_vol > 0) ? sum_imbalance / sum_vol : 0;
    }

    // CDF of VPIN: probability of informed trading
    double vpin_cdf() const {
        double v = vpin();
        // Approximate CDF using normal distribution
        double z = v / (config_.sigma / std::sqrt(static_cast<double>(config_.n_buckets)));
        return 0.5 * std::erfc(-z * 0.7071067811865476);
    }

    void reset() {
        bucket_imbalances_.clear();
        bucket_volumes_.clear();
        current_bucket_vol_ = 0;
        current_buy_vol_ = 0;
    }

private:
    Config config_;
    std::deque<double> bucket_imbalances_;
    std::deque<double> bucket_volumes_;
    double current_bucket_vol_;
    double current_buy_vol_;
};

// ============================================================================
// Execution Quality Benchmarks
// ============================================================================

class ExecutionBenchmarks {
public:
    struct BenchmarkResult {
        double vs_arrival_bps;
        double vs_close_bps;
        double vs_vwap_bps;
        double vs_twap_bps;
        double vs_open_bps;
        double participation_rate;
        double execution_time_sec;
    };

    ExecutionBenchmarks() = default;

    void set_benchmarks(double open, double close, double vwap, double twap, double volume) {
        open_ = open; close_ = close; vwap_ = vwap; twap_ = twap; market_vol_ = volume;
    }

    BenchmarkResult evaluate(Side side, double arrival_mid, double exec_price,
                             double exec_qty, double exec_time_sec) const {
        BenchmarkResult r{};
        int sign = (side == Side::Buy) ? 1 : -1;

        r.vs_arrival_bps = sign * (exec_price - arrival_mid) / arrival_mid * 10000.0;
        r.vs_close_bps = sign * (exec_price - close_) / close_ * 10000.0;
        r.vs_vwap_bps = sign * (exec_price - vwap_) / vwap_ * 10000.0;
        r.vs_twap_bps = sign * (exec_price - twap_) / twap_ * 10000.0;
        r.vs_open_bps = sign * (exec_price - open_) / open_ * 10000.0;
        r.participation_rate = (market_vol_ > 0) ? exec_qty / market_vol_ : 0;
        r.execution_time_sec = exec_time_sec;

        return r;
    }

    // Rank execution quality: 1=best, 5=worst
    int quality_rank(const BenchmarkResult& r) const {
        double avg_slip = (std::abs(r.vs_arrival_bps) + std::abs(r.vs_vwap_bps)) / 2.0;
        if (avg_slip < 1.0) return 1;  // excellent
        if (avg_slip < 3.0) return 2;  // good
        if (avg_slip < 7.0) return 3;  // average
        if (avg_slip < 15.0) return 4; // poor
        return 5;                       // very poor
    }

private:
    double open_ = 0, close_ = 0, vwap_ = 0, twap_ = 0, market_vol_ = 0;
};

// ============================================================================
// Latency Tracker
// ============================================================================

class LatencyTracker {
public:
    LatencyTracker() : count_(0), sum_(0), sum_sq_(0), min_(UINT64_MAX), max_(0) {}

    void record(uint64_t latency_ns) {
        latencies_.push_back(latency_ns);
        if (static_cast<int>(latencies_.size()) > 10000) {
            latencies_.pop_front();
        }
        ++count_;
        sum_ += latency_ns;
        sum_sq_ += static_cast<double>(latency_ns) * latency_ns;
        if (latency_ns < min_) min_ = latency_ns;
        if (latency_ns > max_) max_ = latency_ns;
    }

    double mean_ns() const { return (count_ > 0) ? static_cast<double>(sum_) / count_ : 0; }
    double std_ns() const {
        if (count_ < 2) return 0;
        double m = mean_ns();
        return std::sqrt(sum_sq_ / count_ - m * m);
    }
    uint64_t min_ns() const { return min_; }
    uint64_t max_ns() const { return max_; }

    double percentile(double p) const {
        if (latencies_.empty()) return 0;
        auto sorted = std::vector<uint64_t>(latencies_.begin(), latencies_.end());
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
        return static_cast<double>(sorted[idx]);
    }

    double p50() const { return percentile(0.50); }
    double p95() const { return percentile(0.95); }
    double p99() const { return percentile(0.99); }
    double p999() const { return percentile(0.999); }

    void reset() {
        latencies_.clear();
        count_ = 0; sum_ = 0; sum_sq_ = 0;
        min_ = UINT64_MAX; max_ = 0;
    }

private:
    std::deque<uint64_t> latencies_;
    uint64_t count_;
    uint64_t sum_;
    double sum_sq_;
    uint64_t min_, max_;
};

} // namespace srfm::execution
