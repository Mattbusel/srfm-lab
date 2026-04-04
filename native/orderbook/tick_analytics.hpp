#pragma once
// Tick data analytics: trade clustering, intraday seasonality,
// regime detection, and real-time alert generation.
// All algorithms operate in constant or bounded memory.

#include "order.hpp"
#include "order_flow.hpp"
#include "market_impact.hpp"
#include <vector>
#include <deque>
#include <array>
#include <string>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <chrono>

namespace hft {

// ============================================================
// Intraday time utilities
// ============================================================
// Returns fraction of trading day [0, 1] from nanosecond timestamp
inline double trading_day_fraction(int64_t ts_ns) {
    int64_t open_ns  = 34200LL * 1_000_000_000LL; // 9:30 AM
    int64_t close_ns = 57600LL * 1_000_000_000LL; // 4:00 PM
    int64_t duration = close_ns - open_ns;
    double frac = static_cast<double>(ts_ns - open_ns) / duration;
    return std::max(0.0, std::min(1.0, frac));
}

// Returns minute of day (0-389 for 9:30–4:00)
inline int minute_of_day(int64_t ts_ns) {
    int64_t open_ns = 34200LL * 1_000_000_000LL;
    return static_cast<int>((ts_ns - open_ns) / 60_000_000_000LL);
}

// ============================================================
// Intraday Seasonality Model
// Tracks average volume, spread, and volatility by minute-of-day
// Uses exponential smoothing for online updates
// ============================================================
class IntradaySeasonality {
public:
    static constexpr int MINUTES_PER_DAY = 390; // 9:30–4:00

    struct MinuteBucket {
        double avg_volume  = 0.0;
        double avg_spread  = 0.0;
        double avg_vol     = 0.0;  // realized volatility
        double avg_trades  = 0.0;
        int    n_days      = 0;
    };

    explicit IntradaySeasonality(double alpha = 0.1)  // EMA smoothing
        : alpha_(alpha) {}

    void update(int64_t ts_ns, double volume, double spread, double mid_return) {
        int min = minute_of_day(ts_ns);
        if (min < 0 || min >= MINUTES_PER_DAY) return;

        auto& b = buckets_[min];
        double w = (b.n_days == 0) ? 1.0 : alpha_;
        b.avg_volume = (1 - w) * b.avg_volume + w * volume;
        b.avg_spread = (1 - w) * b.avg_spread + w * spread;
        b.avg_vol    = (1 - w) * b.avg_vol    + w * std::fabs(mid_return);
        b.avg_trades = (1 - w) * b.avg_trades + w * 1.0;
        b.n_days++;
    }

    const MinuteBucket& get(int minute) const {
        minute = std::max(0, std::min(MINUTES_PER_DAY - 1, minute));
        return buckets_[minute];
    }

    // Normalized volume (relative to average for that minute)
    double normalized_volume(int64_t ts_ns, double volume) const {
        int min = minute_of_day(ts_ns);
        if (min < 0 || min >= MINUTES_PER_DAY) return 1.0;
        double avg = buckets_[min].avg_volume;
        return avg > 0 ? volume / avg : 1.0;
    }

    // Is the current minute in high-activity zone?
    bool is_high_volume_period(int64_t ts_ns, double threshold = 1.5) const {
        return normalized_volume(ts_ns, buckets_[minute_of_day(ts_ns)].avg_volume) >= threshold;
    }

    // Get the top-N highest volume minutes
    std::vector<int> top_volume_minutes(size_t n = 10) const {
        std::vector<int> idx(MINUTES_PER_DAY);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + std::min(n, (size_t)MINUTES_PER_DAY),
                          idx.end(), [&](int a, int b){
                              return buckets_[a].avg_volume > buckets_[b].avg_volume;
                          });
        idx.resize(std::min(n, (size_t)MINUTES_PER_DAY));
        return idx;
    }

    void print_summary() const {
        std::cout << "Intraday Seasonality (top 5 volume minutes):\n";
        for (int m : top_volume_minutes(5)) {
            int hour = 9 + (m + 30) / 60;
            int min  = (m + 30) % 60;
            const auto& b = buckets_[m];
            std::cout << "  " << std::setw(2) << hour << ":" << std::setw(2) << std::setfill('0') << min
                      << std::setfill(' ')
                      << "  vol=" << std::fixed << std::setprecision(0) << b.avg_volume
                      << "  spread=" << std::setprecision(4) << b.avg_spread
                      << "  n=" << b.n_days << "\n";
        }
    }

private:
    double      alpha_;
    std::array<MinuteBucket, MINUTES_PER_DAY> buckets_{};
};

// ============================================================
// Trade Clustering Analysis
// Identifies bursts of trading activity using inter-arrival times
// ============================================================
class TradeClusterer {
public:
    struct Cluster {
        int64_t  start_ns;
        int64_t  end_ns;
        int      trade_count;
        double   total_volume;
        double   total_notional;
        double   net_flow;      // positive = buy-dominated
        double   avg_price;
        double   price_impact;  // end_px - start_px
        double   start_px;
        double   end_px;
        bool     is_active;
    };

    struct Config {
        int64_t  gap_threshold_ns   = 5'000'000LL; // 5ms gap closes cluster
        double   min_cluster_volume = 100.0;
        int      min_trades         = 3;
        size_t   max_history        = 500;
    };

    explicit TradeClusterer(const Config& cfg = {}) : cfg_(cfg) {
        current_ = {};
        current_.is_active = false;
    }

    // Process incoming trade
    Cluster* on_trade(int64_t ts_ns, double price, double volume, bool is_buy) {
        if (!current_.is_active) {
            // Start new cluster
            current_             = {};
            current_.start_ns    = ts_ns;
            current_.start_px    = price;
            current_.is_active   = true;
        } else if (ts_ns - last_trade_ns_ > cfg_.gap_threshold_ns) {
            // Close current cluster if it meets criteria
            close_cluster(ts_ns);
        }

        current_.end_ns        = ts_ns;
        current_.end_px        = price;
        current_.trade_count++;
        current_.total_volume  += volume;
        current_.total_notional += volume * price;
        current_.net_flow      += is_buy ? volume : -volume;
        current_.avg_price      = current_.total_notional / current_.total_volume;
        current_.price_impact   = current_.end_px - current_.start_px;
        last_trade_ns_          = ts_ns;

        return nullptr;
    }

    void close_cluster(int64_t now_ns) {
        if (!current_.is_active) return;
        if (current_.trade_count >= cfg_.min_trades &&
            current_.total_volume >= cfg_.min_cluster_volume)
        {
            current_.is_active = false;
            history_.push_back(current_);
            if (history_.size() > cfg_.max_history) history_.pop_front();
            total_clusters_++;
        }
        current_.is_active = false;
        _ = now_ns;
    }

    // Statistics over recent clusters
    struct ClusterStats {
        double avg_cluster_volume;
        double avg_cluster_duration_ms;
        double avg_net_flow;
        double avg_price_impact;
        double cluster_rate_per_min;
        size_t n_clusters;
    };

    ClusterStats compute_stats() const {
        ClusterStats s{};
        s.n_clusters = history_.size();
        if (history_.empty()) return s;

        for (const auto& c : history_) {
            s.avg_cluster_volume      += c.total_volume;
            s.avg_cluster_duration_ms += (c.end_ns - c.start_ns) / 1e6;
            s.avg_net_flow            += c.net_flow;
            s.avg_price_impact        += c.price_impact;
        }
        double n = s.n_clusters;
        s.avg_cluster_volume      /= n;
        s.avg_cluster_duration_ms /= n;
        s.avg_net_flow            /= n;
        s.avg_price_impact        /= n;

        // Rate: clusters per minute (over history time span)
        if (history_.size() > 1) {
            double span_min = (history_.back().end_ns - history_.front().start_ns) / 60e9;
            s.cluster_rate_per_min = span_min > 0 ? n / span_min : 0;
        }
        return s;
    }

    const std::deque<Cluster>& history() const { return history_; }
    uint64_t total_clusters() const noexcept { return total_clusters_; }

private:
    Config          cfg_;
    Cluster         current_;
    int64_t         last_trade_ns_ = 0;
    std::deque<Cluster> history_;
    uint64_t        total_clusters_ = 0;
};

// ============================================================
// Market Regime Detector
// Uses rolling volatility and autocorrelation to classify regime
// ============================================================
class RegimeDetector {
public:
    enum class Regime {
        Trending_Up,
        Trending_Down,
        Mean_Reverting,
        High_Vol,
        Low_Vol,
        Unknown,
    };

    struct Config {
        size_t vol_window      = 60;    // minutes
        size_t trend_window    = 20;    // periods
        double high_vol_z      = 1.5;   // z-score for high vol
        double trend_threshold = 0.6;   // autocorrelation threshold
    };

    explicit RegimeDetector(const Config& cfg = {}) : cfg_(cfg) {}

    void update(double mid_return) {
        returns_.push_back(mid_return);
        if (returns_.size() > std::max(cfg_.vol_window, cfg_.trend_window) * 2)
            returns_.pop_front();

        if (returns_.size() < 10) return;

        // Rolling vol
        double vol = compute_vol(cfg_.vol_window);
        vol_history_.push_back(vol);
        if (vol_history_.size() > 100) vol_history_.pop_front();

        current_ = classify(vol);
    }

    Regime current_regime() const { return current_; }

    std::string regime_name() const {
        switch (current_) {
            case Regime::Trending_Up:    return "Trending_Up";
            case Regime::Trending_Down:  return "Trending_Down";
            case Regime::Mean_Reverting: return "Mean_Reverting";
            case Regime::High_Vol:       return "High_Vol";
            case Regime::Low_Vol:        return "Low_Vol";
            default:                     return "Unknown";
        }
    }

    // How long in current regime (in update steps)
    size_t regime_duration() const { return regime_duration_steps_; }

private:
    double compute_vol(size_t window) const {
        size_t n = std::min(returns_.size(), window);
        if (n < 2) return 0.0;
        double mean = 0, var = 0;
        for (size_t i = returns_.size() - n; i < returns_.size(); ++i)
            mean += returns_[i];
        mean /= n;
        for (size_t i = returns_.size() - n; i < returns_.size(); ++i) {
            double d = returns_[i] - mean;
            var += d * d;
        }
        return std::sqrt(var / (n - 1)) * std::sqrt(252.0 * 6.5 * 60);
    }

    double compute_autocorr(int lag, size_t window) const {
        size_t n = std::min(returns_.size(), window);
        if (n <= (size_t)lag + 2) return 0.0;
        size_t start = returns_.size() - n;
        double mean = 0;
        for (size_t i = start; i < returns_.size(); ++i) mean += returns_[i];
        mean /= n;
        double cov = 0, var = 0;
        for (size_t i = start + lag; i < returns_.size(); ++i) {
            cov += (returns_[i] - mean) * (returns_[i - lag] - mean);
            var += (returns_[i] - mean) * (returns_[i] - mean);
        }
        return var > 0 ? cov / var : 0.0;
    }

    Regime classify(double vol) {
        if (vol_history_.size() < 20) return Regime::Unknown;

        // Vol z-score
        double vol_mean = std::accumulate(vol_history_.begin(), vol_history_.end(), 0.0)
                         / vol_history_.size();
        double vol_var  = 0;
        for (auto v : vol_history_) vol_var += (v - vol_mean) * (v - vol_mean);
        double vol_std  = std::sqrt(vol_var / vol_history_.size());
        double vol_z    = vol_std > 0 ? (vol - vol_mean) / vol_std : 0.0;

        Regime new_regime;
        if (vol_z > cfg_.high_vol_z) {
            new_regime = Regime::High_Vol;
        } else if (vol_z < -cfg_.high_vol_z) {
            new_regime = Regime::Low_Vol;
        } else {
            // Autocorrelation-based trend/mean-reversion detection
            double ac1 = compute_autocorr(1, cfg_.trend_window);
            if (ac1 > cfg_.trend_threshold) {
                // Positive autocorr + recent net return
                double net = 0;
                size_t w = std::min(returns_.size(), cfg_.trend_window);
                for (size_t i = returns_.size() - w; i < returns_.size(); ++i)
                    net += returns_[i];
                new_regime = net > 0 ? Regime::Trending_Up : Regime::Trending_Down;
            } else if (ac1 < -0.2) {
                new_regime = Regime::Mean_Reverting;
            } else {
                new_regime = Regime::Unknown;
            }
        }

        if (new_regime == current_) {
            regime_duration_steps_++;
        } else {
            current_             = new_regime;
            regime_duration_steps_ = 1;
        }
        return current_;
    }

    Config              cfg_;
    std::deque<double>  returns_;
    std::deque<double>  vol_history_;
    Regime              current_            = Regime::Unknown;
    size_t              regime_duration_steps_ = 0;
};

// ============================================================
// Real-time Alert System
// Monitors multiple metrics and fires alerts on threshold breach
// ============================================================
struct Alert {
    enum class Severity { Info, Warning, Critical };
    enum class Type {
        VolSpike,
        VPINHigh,
        SpreadWidening,
        OrderFlowImbalance,
        PriceMovefast,
        LargeCluster,
        RegimeChange,
        DrawdownAlert,
    };

    Type        type;
    Severity    severity;
    std::string symbol;
    std::string message;
    double      value;
    double      threshold;
    int64_t     timestamp_ns;
};

class AlertMonitor {
public:
    struct Thresholds {
        double vol_spike_z         = 2.5;
        double vpin_high           = 0.6;
        double spread_widen_pct    = 2.0;  // 200% of median
        double ofi_high            = 0.7;
        double price_move_bps      = 20.0;
        double large_cluster_vol   = 5000.0;
        double drawdown_pct        = 0.02; // 2%
    };

    using AlertCallback = std::function<void(const Alert&)>;

    explicit AlertMonitor(const std::string& symbol, const Thresholds& thresh = {})
        : symbol_(symbol), thresh_(thresh)
    {}

    void set_callback(AlertCallback cb) { callback_ = std::move(cb); }

    void on_update(int64_t ts_ns, double mid, double vol, double vpin,
                   double spread, double median_spread,
                   double ofi, double cluster_vol,
                   double portfolio_nav)
    {
        mid_history_.push_back(mid);
        if (mid_history_.size() > 60) mid_history_.pop_front();

        // Vol spike
        vol_history_.push_back(vol);
        if (vol_history_.size() > 100) vol_history_.pop_front();
        if (vol_history_.size() > 10) {
            double mean = 0, std = 0;
            for (auto v : vol_history_) mean += v;
            mean /= vol_history_.size();
            for (auto v : vol_history_) std += (v - mean) * (v - mean);
            std = std::sqrt(std / vol_history_.size());
            double z = std > 0 ? (vol - mean) / std : 0;
            if (z > thresh_.vol_spike_z)
                fire({Alert::Type::VolSpike, Alert::Severity::Warning, symbol_,
                      "Volatility spike detected z=" + fmt(z), z, thresh_.vol_spike_z, ts_ns});
        }

        // VPIN
        if (vpin > thresh_.vpin_high)
            fire({Alert::Type::VPINHigh, Alert::Severity::Critical, symbol_,
                  "VPIN elevated: " + fmt(vpin), vpin, thresh_.vpin_high, ts_ns});

        // Spread widening
        if (median_spread > 0) {
            double spread_ratio = spread / median_spread;
            if (spread_ratio > thresh_.spread_widen_pct)
                fire({Alert::Type::SpreadWidening, Alert::Severity::Warning, symbol_,
                      "Spread widened " + fmt(spread_ratio) + "x", spread_ratio, thresh_.spread_widen_pct, ts_ns});
        }

        // OFI imbalance
        if (std::fabs(ofi) > thresh_.ofi_high)
            fire({Alert::Type::OrderFlowImbalance, Alert::Severity::Info, symbol_,
                  "OFI imbalance: " + fmt(ofi), ofi, thresh_.ofi_high, ts_ns});

        // Fast price move
        if (mid_history_.size() >= 5) {
            double prev = mid_history_[mid_history_.size() - 5];
            double move_bps = prev > 0 ? std::fabs(mid - prev) / prev * 10000.0 : 0;
            if (move_bps > thresh_.price_move_bps)
                fire({Alert::Type::PriceMovefast, Alert::Severity::Warning, symbol_,
                      "Fast move: " + fmt(move_bps) + " bps", move_bps, thresh_.price_move_bps, ts_ns});
        }

        // Large trade cluster
        if (cluster_vol > thresh_.large_cluster_vol)
            fire({Alert::Type::LargeCluster, Alert::Severity::Info, symbol_,
                  "Large cluster: " + fmt(cluster_vol) + " shares", cluster_vol, thresh_.large_cluster_vol, ts_ns});

        // Drawdown
        if (portfolio_nav > 0) {
            peak_nav_ = std::max(peak_nav_, portfolio_nav);
            double dd = peak_nav_ > 0 ? (peak_nav_ - portfolio_nav) / peak_nav_ : 0;
            if (dd > thresh_.drawdown_pct)
                fire({Alert::Type::DrawdownAlert, Alert::Severity::Critical, symbol_,
                      "Drawdown: " + fmt(dd * 100) + "%", dd, thresh_.drawdown_pct, ts_ns});
        }

        total_updates_++;
    }

    uint64_t total_alerts() const noexcept { return total_alerts_; }
    uint64_t total_updates() const noexcept { return total_updates_; }
    const std::vector<Alert>& recent_alerts() const { return recent_; }

private:
    static std::string fmt(double v) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4) << v;
        return ss.str();
    }

    void fire(Alert a) {
        total_alerts_++;
        recent_.push_back(a);
        if (recent_.size() > 100) recent_.erase(recent_.begin());
        if (callback_) callback_(a);
    }

    std::string        symbol_;
    Thresholds         thresh_;
    AlertCallback      callback_;
    std::deque<double> mid_history_;
    std::deque<double> vol_history_;
    double             peak_nav_     = 0.0;
    uint64_t           total_alerts_ = 0;
    uint64_t           total_updates_ = 0;
    std::vector<Alert> recent_;
};

// ============================================================
// Composite analytics engine: all metrics for a single symbol
// ============================================================
class TickAnalyticsEngine {
public:
    struct AnalyticsState {
        double mid;
        double spread;
        double vol;
        double vpin;
        double ofi;
        double imbalance;
        double cluster_vol;
        RegimeDetector::Regime regime;
        std::vector<Alert> new_alerts;
    };

    explicit TickAnalyticsEngine(const std::string& symbol)
        : symbol_(symbol),
          seasonality_(),
          clusterer_(),
          regime_(),
          flow_monitor_(symbol),
          alert_monitor_(symbol)
    {}

    void set_alert_callback(AlertMonitor::AlertCallback cb) {
        alert_monitor_.set_callback(std::move(cb));
    }

    // Process one trade tick
    AnalyticsState on_trade(int64_t ts_ns, Price price, Price bid, Price ask,
                             Quantity qty, bool is_buy)
    {
        AnalyticsState state{};
        state.mid    = (price_to_double(bid) + price_to_double(ask)) / 2.0;
        state.spread = price_to_double(ask) - price_to_double(bid);

        double px_d = price_to_double(price);

        // Seasonality
        seasonality_.update(ts_ns, qty, state.spread, 0.0);

        // Trade clustering
        clusterer_.on_trade(ts_ns, px_d, qty, is_buy);
        auto cs = clusterer_.compute_stats();
        state.cluster_vol = cs.avg_cluster_volume;

        // Order flow
        flow_monitor_.on_trade(price, bid, ask, qty, ts_ns);
        auto toxic = flow_monitor_.assess();
        state.vpin   = toxic.vpin;
        state.ofi    = toxic.ofi;
        state.imbalance = 0.0;

        // Volatility / regime
        if (prev_mid_ > 0) {
            double ret = std::log(state.mid / prev_mid_);
            regime_.update(ret);
            state.vol = std::fabs(ret) * std::sqrt(252.0 * 6.5 * 3600.0);
        }
        state.regime = regime_.current_regime();
        prev_mid_ = state.mid;

        // Alerts
        alert_monitor_.on_update(ts_ns, state.mid, state.vol, state.vpin,
                                  state.spread, 0.01, state.ofi,
                                  qty, 1000000.0);

        // Capture new alerts
        const auto& recent = alert_monitor_.recent_alerts();
        if (!recent.empty() && last_alert_count_ < recent.size()) {
            for (size_t i = last_alert_count_; i < recent.size(); ++i)
                state.new_alerts.push_back(recent[i]);
            last_alert_count_ = recent.size();
        }

        update_count_++;
        return state;
    }

    void print_summary() const {
        std::cout << "Analytics Engine: " << symbol_ << "\n";
        std::cout << "  Updates:    " << update_count_ << "\n";
        std::cout << "  Clusters:   " << clusterer_.total_clusters() << "\n";
        std::cout << "  Regime:     " << regime_.regime_name()
                  << " (" << regime_.regime_duration() << " steps)\n";
        std::cout << "  Alerts:     " << alert_monitor_.total_alerts() << "\n";
        seasonality_.print_summary();
    }

private:
    std::string          symbol_;
    IntradaySeasonality  seasonality_;
    TradeClusterer       clusterer_;
    RegimeDetector       regime_;
    ToxicityMonitor      flow_monitor_;
    AlertMonitor         alert_monitor_;
    double               prev_mid_       = 0.0;
    uint64_t             update_count_   = 0;
    size_t               last_alert_count_ = 0;
};

// ============================================================
// OHLCV bar builder with analytics
// ============================================================
struct OHLCVBar {
    int64_t  open_time_ns;
    int64_t  close_time_ns;
    double   open, high, low, close;
    double   vwap;
    uint64_t volume;
    uint64_t trade_count;
    double   buy_vol_pct;
    double   realized_vol;  // intra-bar
    double   avg_spread;
    int      bar_type;       // 0=time, 1=volume, 2=dollar
};

class BarBuilder {
public:
    enum class BarType { Time, Volume, Dollar };

    explicit BarBuilder(BarType type = BarType::Time,
                        int64_t bar_duration_ns = 60'000'000'000LL, // 1 minute
                        uint64_t volume_bar_size = 10000,
                        double   dollar_bar_size = 1'000'000.0)
        : bar_type_(type), bar_duration_ns_(bar_duration_ns),
          volume_bar_size_(volume_bar_size), dollar_bar_size_(dollar_bar_size)
    {
        reset_current();
    }

    using BarCallback = std::function<void(const OHLCVBar&)>;
    void set_callback(BarCallback cb) { callback_ = std::move(cb); }

    void on_trade(int64_t ts_ns, double price, uint64_t qty, bool is_buy,
                  double spread)
    {
        if (!bar_open_) {
            // Open new bar
            current_.open_time_ns = ts_ns;
            current_.open         = price;
            current_.high         = price;
            current_.low          = price;
            bar_open_             = true;
        }

        current_.high  = std::max(current_.high, price);
        current_.low   = std::min(current_.low, price);
        current_.close = price;
        current_.close_time_ns = ts_ns;
        current_.volume      += qty;
        current_.trade_count++;
        current_.vwap        += price * qty;
        current_.avg_spread  += spread;
        if (is_buy) current_.buy_vol_pct += qty;

        // Track realized vol (log returns within bar)
        if (prev_price_ > 0) {
            double ret = std::log(price / prev_price_);
            ret_sum_sq_ += ret * ret;
            ret_count_++;
        }
        prev_price_ = price;

        // Check if bar should close
        bool should_close = false;
        switch (bar_type_) {
            case BarType::Time:
                should_close = (ts_ns - current_.open_time_ns) >= bar_duration_ns_;
                break;
            case BarType::Volume:
                should_close = current_.volume >= volume_bar_size_;
                break;
            case BarType::Dollar:
                should_close = (current_.vwap >= dollar_bar_size_);  // vwap = sum(px*qty)
                break;
        }

        if (should_close) close_bar();
    }

    void close_bar() {
        if (!bar_open_ || current_.volume == 0) return;
        current_.vwap       /= current_.volume;
        current_.avg_spread /= current_.trade_count;
        current_.buy_vol_pct /= current_.volume;
        current_.realized_vol = ret_count_ > 1
            ? std::sqrt(ret_sum_sq_ / ret_count_ * 252.0 * 6.5 * 3600.0) : 0.0;
        current_.bar_type = static_cast<int>(bar_type_);

        total_bars_++;
        if (callback_) callback_(current_);

        bars_history_.push_back(current_);
        if (bars_history_.size() > 500) bars_history_.pop_front();

        reset_current();
    }

    const std::deque<OHLCVBar>& history() const { return bars_history_; }
    uint64_t total_bars() const noexcept { return total_bars_; }

private:
    void reset_current() {
        current_   = {};
        bar_open_  = false;
        prev_price_  = 0.0;
        ret_sum_sq_  = 0.0;
        ret_count_   = 0;
    }

    BarType       bar_type_;
    int64_t       bar_duration_ns_;
    uint64_t      volume_bar_size_;
    double        dollar_bar_size_;
    OHLCVBar      current_{};
    bool          bar_open_   = false;
    double        prev_price_ = 0.0;
    double        ret_sum_sq_ = 0.0;
    size_t        ret_count_  = 0;
    BarCallback   callback_;
    uint64_t      total_bars_ = 0;
    std::deque<OHLCVBar> bars_history_;
};

// ============================================================
// Microstructure noise model (Roll 1984 model)
// Estimates transaction cost from autocovariance of price changes
// ============================================================
class RollModel {
public:
    explicit RollModel(size_t window = 100) : window_(window) {}

    void update(double price) {
        prices_.push_back(price);
        if (prices_.size() > window_ + 1) prices_.pop_front();
    }

    // Roll spread estimate: 2 * sqrt(-autocovariance)
    double roll_spread() const {
        if (prices_.size() < 3) return 0.0;
        size_t n = prices_.size();
        std::vector<double> changes(n - 1);
        for (size_t i = 0; i + 1 < n; ++i)
            changes[i] = prices_[i + 1] - prices_[i];

        // Autocovariance at lag 1
        double mean = 0;
        for (auto c : changes) mean += c;
        mean /= changes.size();
        double cov = 0;
        for (size_t i = 1; i < changes.size(); ++i)
            cov += (changes[i] - mean) * (changes[i - 1] - mean);
        cov /= (changes.size() - 1);

        if (cov >= 0) return 0.0;  // Roll model requires negative autocov
        return 2.0 * std::sqrt(-cov);
    }

    // Realized spread from trade-by-trade data
    double realized_spread(double fill_px, double mid_px) const {
        return 2.0 * std::fabs(fill_px - mid_px);
    }

private:
    size_t             window_;
    std::deque<double> prices_;
};

// ============================================================
// Run integration demo
// ============================================================
void run_analytics_demo() {
    std::cout << "\n=== Tick Analytics Demo ===\n";

    TickAnalyticsEngine engine("DEMO");
    engine.set_alert_callback([](const Alert& a) {
        std::cout << "  ALERT [" << (a.severity == Alert::Severity::Critical ? "CRIT" :
                                     a.severity == Alert::Severity::Warning ? "WARN" : "INFO")
                  << "] " << a.message << "  val=" << a.value << "\n";
    });

    BarBuilder builder(BarBuilder::BarType::Time, 60'000'000'000LL);
    builder.set_callback([](const OHLCVBar& bar) {
        std::cout << "  BAR: open=" << std::fixed << std::setprecision(2) << bar.open
                  << " close=" << bar.close
                  << " vol="   << bar.volume
                  << " vwap="  << bar.vwap
                  << " trades=" << bar.trade_count << "\n";
    });

    RollModel roll;

    std::mt19937 rng(42);
    std::normal_distribution<double> norm(0, 1);
    std::uniform_real_distribution<double> unif(0, 1);

    double price = 150.0;
    int64_t ts_ns = 34200LL * 1'000'000'000LL;

    for (int step = 0; step < 5000; ++step) {
        ts_ns += 1'000'000; // 1ms
        price *= std::exp(norm(rng) * 0.001);
        price = std::max(price, 0.01);

        double spread = 0.02 + std::fabs(norm(rng)) * 0.01;
        double bid = price - spread / 2;
        double ask = price + spread / 2;
        bool is_buy = unif(rng) < 0.52;
        uint64_t qty = 100 + static_cast<uint64_t>(norm(rng) * norm(rng) * 200);
        qty = std::max((uint64_t)10, qty);

        engine.on_trade(ts_ns, double_to_price(price),
                        double_to_price(bid), double_to_price(ask),
                        qty, is_buy);

        builder.on_trade(ts_ns, price, qty, is_buy, spread);
        roll.update(price);
    }
    builder.close_bar();

    engine.print_summary();
    std::cout << "  Total bars:  " << builder.total_bars() << "\n";
    std::cout << "  Roll spread: " << std::setprecision(5) << roll.roll_spread() << "\n";
}

} // namespace hft
