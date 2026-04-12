// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// market_microstructure.cpp — Market microstructure analytics
// =============================================================================
// Implements:
// - Trade classification (Lee-Ready, Tick rule, EMO)
// - Effective/realized spread decomposition
// - Kyle's lambda, Amihud illiquidity
// - Hasbrouck information share
// - Price impact models (linear, square-root)
// - Intraday volume/volatility patterns
// - Market efficiency metrics

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtel {
namespace microstructure {

static constexpr double kEps = 1e-12;

// ---------------------------------------------------------------------------
// Trade classification
// ---------------------------------------------------------------------------

enum class TradeSide { Buy, Sell, Unknown };

// Lee-Ready tick rule
TradeSide classify_tick_rule(double trade_price, double prev_price,
                              double prev_prev_price = 0.0)
{
    if (trade_price > prev_price)       return TradeSide::Buy;
    if (trade_price < prev_price)       return TradeSide::Sell;
    // Reverse tick rule for tied prices
    if (prev_prev_price > 0.0) {
        if (prev_price > prev_prev_price) return TradeSide::Buy;
        if (prev_price < prev_prev_price) return TradeSide::Sell;
    }
    return TradeSide::Unknown;
}

// Lee-Ready quote rule (uses bid/ask quotes)
TradeSide classify_quote_rule(double trade_price, double bid, double ask) {
    double mid = 0.5 * (bid + ask);
    if (trade_price > mid) return TradeSide::Buy;
    if (trade_price < mid) return TradeSide::Sell;
    return TradeSide::Unknown;
}

// EMO (Ellis-Michaely-O'Hara) combined rule
TradeSide classify_emo(double trade_price, double bid, double ask,
                       double prev_trade, double prev2_trade = 0.0)
{
    // First apply quote rule
    if (trade_price == ask)  return TradeSide::Buy;
    if (trade_price == bid)  return TradeSide::Sell;
    if (trade_price > bid && trade_price < ask) {
        // Apply tick rule for inside-quote trades
        return classify_tick_rule(trade_price, prev_trade, prev2_trade);
    }
    return classify_quote_rule(trade_price, bid, ask);
}

// ---------------------------------------------------------------------------
// Spread decomposition
// ---------------------------------------------------------------------------

struct SpreadComponents {
    double effective_spread;      // = 2 |trade_px - mid|
    double realized_spread_5min;  // = 2 * side * (trade_px - mid_t+5min)
    double price_impact_5min;     // = side * (mid_t+5min - mid_t)
    double adverse_selection;     // = price_impact (information component)
    double inventory_cost;        // = realized_spread (inventory component)
    double quoted_spread;         // = ask - bid
    double relative_spread;       // quoted / mid

    static SpreadComponents compute(double trade_px, double mid_at_trade,
                                    double mid_5min_later,
                                    double bid, double ask,
                                    TradeSide side)
    {
        SpreadComponents s{};
        double sign = (side == TradeSide::Buy) ? 1.0 : -1.0;
        s.effective_spread     = 2.0 * std::abs(trade_px - mid_at_trade);
        s.realized_spread_5min = 2.0 * sign * (trade_px - mid_5min_later);
        s.price_impact_5min    = sign * (mid_5min_later - mid_at_trade);
        s.adverse_selection    = s.price_impact_5min;
        s.inventory_cost       = s.realized_spread_5min;
        s.quoted_spread        = ask - bid;
        s.relative_spread      = (mid_at_trade > kEps) ?
            s.quoted_spread / mid_at_trade : 0.0;
        return s;
    }
};

// ---------------------------------------------------------------------------
// Price impact models
// ---------------------------------------------------------------------------

struct PriceImpactModel {
    double lambda;        // linear impact coefficient
    double sqrt_coeff;    // square-root impact coefficient
    double eta;           // decay coefficient

    // Almgren-Chriss linear impact
    double linear_impact(double trade_size, double adv) const {
        return lambda * trade_size / adv;
    }

    // Square-root market impact (Grinold-Kahn approximation)
    double sqrt_impact(double trade_size, double adv, double sigma) const {
        double pct_adv = trade_size / adv;
        return sqrt_coeff * sigma * std::sqrt(pct_adv);
    }

    // Combined impact
    double total_impact(double trade_size, double adv, double sigma) const {
        return linear_impact(trade_size, adv) + sqrt_impact(trade_size, adv, sigma);
    }
};

// Kyle's lambda (price impact per unit of signed order flow)
// Estimated via regression: Δp = λ × net_order_flow + ε
class KyleLambdaEstimator {
    std::deque<double> delta_p_;     // price changes
    std::deque<double> order_flow_;  // signed order flow
    int window_;

public:
    explicit KyleLambdaEstimator(int window = 50) : window_(window) {}

    void update(double price_change, double signed_flow) {
        delta_p_.push_back(price_change);
        order_flow_.push_back(signed_flow);
        if ((int)delta_p_.size() > window_) {
            delta_p_.pop_front();
            order_flow_.pop_front();
        }
    }

    double estimate() const {
        int n = (int)delta_p_.size();
        if (n < 5) return 0.0;

        double sum_xy = 0.0, sum_x2 = 0.0;
        double mean_x = 0.0, mean_y = 0.0;
        for (int i = 0; i < n; ++i) {
            mean_x += order_flow_[i];
            mean_y += delta_p_[i];
        }
        mean_x /= n; mean_y /= n;
        for (int i = 0; i < n; ++i) {
            double dx = order_flow_[i] - mean_x;
            sum_xy += dx * (delta_p_[i] - mean_y);
            sum_x2 += dx * dx;
        }
        return (sum_x2 > kEps) ? sum_xy / sum_x2 : 0.0;
    }
};

// ---------------------------------------------------------------------------
// Amihud illiquidity ratio
// ---------------------------------------------------------------------------

// ILLIQ = (1/T) Σ |r_t| / Volume_t
// Higher = more illiquid
class AmihudIlliquidity {
    std::deque<double> returns_;
    std::deque<double> volumes_;
    int window_;

public:
    explicit AmihudIlliquidity(int window = 20) : window_(window) {}

    void update(double abs_return, double dollar_volume) {
        returns_.push_back(abs_return);
        volumes_.push_back(dollar_volume);
        if ((int)returns_.size() > window_) {
            returns_.pop_front();
            volumes_.pop_front();
        }
    }

    double illiquidity() const {
        int n = (int)returns_.size();
        if (n < 2) return 0.0;
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            if (volumes_[i] > kEps)
                sum += returns_[i] / volumes_[i];
        }
        return sum / n * 1e6;  // scale to per million dollars
    }
};

// ---------------------------------------------------------------------------
// Intraday patterns
// ---------------------------------------------------------------------------

// U-shaped intraday volume pattern (based on 5-min buckets)
struct IntradayPattern {
    static constexpr int kBuckets = 78;  // 6.5h × 12 buckets/hour

    std::array<double, kBuckets> mean_volume{};
    std::array<double, kBuckets> mean_spread{};
    std::array<double, kBuckets> mean_vol{};    // realized vol per bucket
    std::array<int, kBuckets>    n_obs{};

    void update(int bucket_idx, double volume, double spread, double realized_vol) {
        if (bucket_idx < 0 || bucket_idx >= kBuckets) return;
        int n = ++n_obs[bucket_idx];
        double alpha = 1.0 / n;
        mean_volume[bucket_idx] += alpha * (volume - mean_volume[bucket_idx]);
        mean_spread[bucket_idx] += alpha * (spread - mean_spread[bucket_idx]);
        mean_vol[bucket_idx]    += alpha * (realized_vol - mean_vol[bucket_idx]);
    }

    // Volume adjustment factor: actual / expected (for VWAP scheduling)
    double volume_factor(int bucket_idx) const {
        if (bucket_idx < 0 || bucket_idx >= kBuckets) return 1.0;
        double total = 0.0;
        for (double v : mean_volume) total += v;
        if (total < kEps) return 1.0;
        double expected_frac = mean_volume[bucket_idx] / total;
        return expected_frac * kBuckets;  // normalized per bucket
    }

    // Optimal execution time (lowest spread + lowest vol period)
    int optimal_execution_bucket() const {
        int    best_b = 0;
        double best_v = std::numeric_limits<double>::max();
        for (int b = 0; b < kBuckets; ++b) {
            double cost = mean_spread[b] + mean_vol[b];
            if (cost < best_v) { best_v = cost; best_b = b; }
        }
        return best_b;
    }
};

// ---------------------------------------------------------------------------
// Market efficiency metrics
// ---------------------------------------------------------------------------

// Variance ratio test: if returns are i.i.d., VR(q) = 1
// VR > 1 suggests positive autocorrelation (momentum)
// VR < 1 suggests negative autocorrelation (mean reversion)
double variance_ratio(const std::vector<double>& prices, int q) {
    int n = (int)prices.size();
    if (n < q + 2) return 1.0;

    // 1-period returns
    std::vector<double> r1;
    for (int i = 1; i < n; ++i)
        r1.push_back((prices[i] - prices[i-1]) / (prices[i-1] + kEps));
    double mean1 = std::accumulate(r1.begin(), r1.end(), 0.0) / r1.size();
    double var1  = 0.0;
    for (double r : r1) var1 += (r-mean1)*(r-mean1);
    var1 /= r1.size();

    // q-period returns
    std::vector<double> rq;
    for (int i = q; i < n; ++i)
        rq.push_back((prices[i] - prices[i-q]) / (prices[i-q] + kEps));
    double meanq = std::accumulate(rq.begin(), rq.end(), 0.0) / rq.size();
    double varq  = 0.0;
    for (double r : rq) varq += (r-meanq)*(r-meanq);
    varq /= rq.size();

    // VR(q) = Var(rq) / (q * Var(r1))
    double denom = q * var1;
    return (denom > kEps) ? varq / denom : 1.0;
}

// Runs test for randomness
struct RunsTestResult {
    int    n_runs;
    int    n_pos;
    int    n_neg;
    double z_stat;    // test statistic
    bool   reject_iid; // reject i.i.d. at 5% level if |z| > 1.96
};

RunsTestResult runs_test(const std::vector<double>& returns) {
    RunsTestResult res{};
    int n = (int)returns.size();
    if (n < 4) return res;

    for (double r : returns) {
        if (r > 0) ++res.n_pos;
        else if (r < 0) ++res.n_neg;
    }

    int n1 = res.n_pos, n2 = res.n_neg;
    int n_total = n1 + n2;
    if (n_total < 2) return res;

    // Count runs
    int runs = 1;
    for (int i = 1; i < (int)returns.size(); ++i) {
        bool prev_pos = returns[i-1] >= 0;
        bool curr_pos = returns[i]   >= 0;
        if (prev_pos != curr_pos) ++runs;
    }
    res.n_runs = runs;

    // Expected runs and variance under H0
    double E_R  = 2.0 * n1 * n2 / n_total + 1.0;
    double var_R = 2.0 * n1 * n2 * (2.0*n1*n2 - n_total) /
                   ((double)n_total * n_total * (n_total - 1.0));
    double std_R = std::sqrt(std::max(0.0, var_R));

    res.z_stat    = (std_R > kEps) ? (runs - E_R) / std_R : 0.0;
    res.reject_iid= std::abs(res.z_stat) > 1.96;
    return res;
}

// ---------------------------------------------------------------------------
// TWAP/VWAP execution algorithms
// ---------------------------------------------------------------------------

struct TWAPSchedule {
    int     n_slices;
    double  total_size;
    std::vector<double> slice_sizes;
    std::vector<double> slice_timestamps;

    // Generate uniform TWAP schedule
    static TWAPSchedule create(double total_size, int n_slices,
                                double start_ts, double end_ts)
    {
        TWAPSchedule s{};
        s.n_slices    = n_slices;
        s.total_size  = total_size;
        double slice  = total_size / n_slices;
        double dt     = (end_ts - start_ts) / n_slices;
        for (int i = 0; i < n_slices; ++i) {
            s.slice_sizes.push_back(slice);
            s.slice_timestamps.push_back(start_ts + i * dt);
        }
        return s;
    }
};

struct VWAPSchedule {
    int     n_slices;
    double  total_size;
    std::vector<double> slice_sizes;
    std::vector<double> slice_timestamps;

    // Generate VWAP schedule from historical intraday volume pattern
    static VWAPSchedule create(double total_size, int n_slices,
                                const IntradayPattern& pattern,
                                int start_bucket, int end_bucket)
    {
        VWAPSchedule s{};
        s.n_slices   = n_slices;
        s.total_size = total_size;

        // Compute bucket volume shares
        std::vector<double> vshares;
        int range = end_bucket - start_bucket;
        int step  = std::max(1, range / n_slices);
        double total_share = 0.0;
        for (int b = start_bucket; b < end_bucket; b += step) {
            double share = pattern.volume_factor(b);
            vshares.push_back(share);
            total_share += share;
        }
        if (total_share < kEps) total_share = vshares.size();
        for (double& v : vshares) v /= total_share;

        for (int i = 0; i < (int)vshares.size(); ++i) {
            s.slice_sizes.push_back(vshares[i] * total_size);
            s.slice_timestamps.push_back(start_bucket + i * step * 300.0);
        }
        return s;
    }
};

// Participation-rate execution (pov = % of volume)
struct POVSchedule {
    double  target_participation;  // e.g. 0.10 = 10% of market volume
    double  total_size;
    double  executed;

    double next_slice(double market_volume_this_period) const {
        return target_participation * market_volume_this_period;
    }
};

// ---------------------------------------------------------------------------
// Microstructure statistics aggregator
// ---------------------------------------------------------------------------

struct MicrostructureStats {
    double mean_spread_bps;
    double std_spread_bps;
    double kyle_lambda;
    double amihud_illiq;
    double variance_ratio_5;
    double variance_ratio_10;
    double effective_spread;
    double adverse_selection_pct;
    double daily_volume;
    double turnover;
    double n_trades;
    double trade_size_mean;
    double trade_size_std;
    bool   reject_iid;
};

class MicrostructureAnalyzer {
    KyleLambdaEstimator kyle_est_{50};
    AmihudIlliquidity   amihud_{20};
    IntradayPattern     intraday_pattern_;

    std::deque<double> prices_;
    std::deque<double> spreads_bps_;
    std::deque<double> returns_;
    std::deque<double> trade_sizes_;

    double total_volume_  = 0.0;
    double n_trades_      = 0.0;

    int  window_ = 100;

public:
    void update_quote(double bid, double ask, double mid) {
        double spread_bps = (mid > kEps) ? (ask - bid) / mid * 1e4 : 0.0;
        spreads_bps_.push_back(spread_bps);
        if ((int)spreads_bps_.size() > window_) spreads_bps_.pop_front();

        if (!prices_.empty()) {
            double r = (prices_.back() > kEps) ?
                (mid - prices_.back()) / prices_.back() : 0.0;
            returns_.push_back(r);
            if ((int)returns_.size() > window_) returns_.pop_front();
        }
        prices_.push_back(mid);
        if ((int)prices_.size() > window_) prices_.pop_front();
    }

    void update_trade(double price, double size, TradeSide side) {
        double signed_size = (side == TradeSide::Buy) ? size : -size;
        double ret = !returns_.empty() ? returns_.back() : 0.0;
        kyle_est_.update(ret, signed_size);
        amihud_.update(std::abs(ret), price * size);
        trade_sizes_.push_back(size);
        if ((int)trade_sizes_.size() > window_) trade_sizes_.pop_front();
        total_volume_ += price * size;
        n_trades_     += 1.0;
    }

    MicrostructureStats compute() const {
        MicrostructureStats s{};
        int n_s = (int)spreads_bps_.size();
        if (n_s > 0) {
            double sum = 0.0, sum2 = 0.0;
            for (double sp : spreads_bps_) { sum += sp; sum2 += sp*sp; }
            s.mean_spread_bps = sum / n_s;
            s.std_spread_bps  = std::sqrt(std::max(0.0, sum2/n_s - s.mean_spread_bps*s.mean_spread_bps));
        }
        s.kyle_lambda  = kyle_est_.estimate();
        s.amihud_illiq = amihud_.illiquidity();

        std::vector<double> pv(prices_.begin(), prices_.end());
        if (pv.size() > 10) {
            s.variance_ratio_5  = variance_ratio(pv, 5);
            s.variance_ratio_10 = variance_ratio(pv, 10);
        }

        std::vector<double> rv(returns_.begin(), returns_.end());
        if (rv.size() > 10) {
            auto rt = runs_test(rv);
            s.reject_iid = rt.reject_iid;
        }

        s.daily_volume  = total_volume_;
        s.n_trades      = n_trades_;
        if (!trade_sizes_.empty()) {
            double sum = 0.0, sum2 = 0.0;
            for (double ts : trade_sizes_) { sum += ts; sum2 += ts*ts; }
            s.trade_size_mean = sum / trade_sizes_.size();
            s.trade_size_std  = std::sqrt(std::max(0.0,
                sum2/trade_sizes_.size() - s.trade_size_mean*s.trade_size_mean));
        }
        return s;
    }

    std::string summary() const {
        auto s = compute();
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "spread=%.2fbps std=%.2fbps lambda=%.4f amihud=%.4f "
            "VR(5)=%.3f VR(10)=%.3f n_trades=%.0f",
            s.mean_spread_bps, s.std_spread_bps, s.kyle_lambda,
            s.amihud_illiq, s.variance_ratio_5, s.variance_ratio_10,
            s.n_trades);
        return buf;
    }
};

// ---------------------------------------------------------------------------
// Information share (Hasbrouck)
// ---------------------------------------------------------------------------

// Simplified single-venue information share estimation
// Uses vector error correction model (VECM) framework
double estimate_information_share(const std::vector<double>& price1,
                                   const std::vector<double>& price2)
{
    int n = (int)std::min(price1.size(), price2.size());
    if (n < 10) return 0.5;

    // Estimate VECM: Δp1_t = α1 * (p1_{t-1} - p2_{t-1}) + ε1_t
    //                Δp2_t = α2 * (p1_{t-1} - p2_{t-1}) + ε2_t
    std::vector<double> dp1(n-1), dp2(n-1), spread(n-1);
    for (int i = 0; i < n-1; ++i) {
        dp1[i]    = price1[i+1] - price1[i];
        dp2[i]    = price2[i+1] - price2[i];
        spread[i] = price1[i] - price2[i];
    }
    int m = n - 1;

    // OLS for alpha1
    double cov1 = 0.0, var_s = 0.0, mean_dp1 = 0.0, mean_s = 0.0;
    for (int i = 0; i < m; ++i) { mean_dp1 += dp1[i]; mean_s += spread[i]; }
    mean_dp1 /= m; mean_s /= m;
    for (int i = 0; i < m; ++i) {
        cov1  += (dp1[i]-mean_dp1) * (spread[i]-mean_s);
        var_s += (spread[i]-mean_s) * (spread[i]-mean_s);
    }
    double alpha1 = (var_s > kEps) ? cov1 / var_s : 0.0;

    // OLS for alpha2
    double cov2 = 0.0, mean_dp2 = 0.0;
    for (int i = 0; i < m; ++i) mean_dp2 += dp2[i];
    mean_dp2 /= m;
    for (int i = 0; i < m; ++i)
        cov2 += (dp2[i]-mean_dp2) * (spread[i]-mean_s);
    double alpha2 = (var_s > kEps) ? cov2 / var_s : 0.0;

    // Information share: IS1 = α2^2 / (α1^2 + α2^2)
    double denom = alpha1*alpha1 + alpha2*alpha2;
    return (denom > kEps) ? alpha2*alpha2 / denom : 0.5;
}

}  // namespace microstructure
}  // namespace rtel
