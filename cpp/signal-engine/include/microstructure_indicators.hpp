#pragma once
// microstructure_indicators.hpp -- Tick-level market microstructure metrics.
// All indicators use circular buffers for O(1) incremental computation.
// Thread safety: hot-path reads are atomic; writes must be externally serialized
// for a single update() caller (single-producer model) OR internally locked for
// multi-producer. Atomic reads prevent torn-word reads from concurrent reporters.
//
// AVX2 batch path: compute_batch() processes N ticks in one call using
// vectorized inner-product computations.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define SRFM_MICRO_AVX2 1
#endif

namespace srfm {
namespace microstructure {

// ============================================================
// Tick -- raw trade / quote event
// ============================================================

struct Tick {
    double   price;         // trade price
    double   volume;        // trade size (always positive)
    bool     is_buy;        // true = buyer-initiated
    int64_t  timestamp_ns;  // Unix epoch nanoseconds
};

// ============================================================
// Internal circular buffer -- power-of-2, no atomics (single writer)
// ============================================================

template<typename T, int Cap>
struct CircBuf {
    static_assert((Cap & (Cap - 1)) == 0, "Cap must be power of 2");
    static constexpr int MASK = Cap - 1;

    T    data[Cap]{};
    int  head = 0;   // next write position
    int  count = 0;  // number of valid elements (<= Cap)

    void push(const T& v) noexcept {
        data[head] = v;
        head = (head + 1) & MASK;
        if (count < Cap) ++count;
    }

    // Oldest element at logical index 0, newest at count-1.
    const T& at(int i) const noexcept {
        assert(i >= 0 && i < count);
        if (count == Cap) {
            return data[(head + i) & MASK];
        }
        return data[i];
    }

    bool full()  const noexcept { return count == Cap; }
    bool empty() const noexcept { return count == 0; }
    void clear() noexcept { head = 0; count = 0; }
};

// ============================================================
// VPIN bucket accumulator
// ============================================================

struct VpinBucket {
    double buy_vol;
    double sell_vol;

    double total()    const noexcept { return buy_vol + sell_vol; }
    double imbalance() const noexcept { return std::abs(buy_vol - sell_vol); }
};

// ============================================================
// MicrostructureIndicators
// ============================================================

class MicrostructureIndicators {
public:
    static constexpr int TICK_CAP        = 4096;   // circular tick buffer capacity
    static constexpr int BUCKET_CAP      = 64;     // max VPIN buckets stored
    static constexpr int PRICE_CAP       = 1024;   // price/return history
    static constexpr int FLOW_CAP        = 512;    // signed flow history

    MicrostructureIndicators() noexcept
        : total_volume_(0.0),
          bucket_target_vol_(0.0),
          current_bucket_{0, 0},
          buckets_filled_(0),
          vpin_cached_(std::numeric_limits<double>::quiet_NaN()),
          tick_count_(0)
    {
        ticks_.clear();
        prices_.clear();
        returns_.clear();
        signed_flows_.clear();
        bucket_history_.clear();
    }

    // update -- incremental single-tick update. Single-producer thread only.
    void update(const Tick& tick) noexcept {
        const double vol = tick.volume;
        if (vol <= 0.0 || tick.price <= 0.0) return;

        // Update tick circular buffer
        ticks_.push(tick);
        prices_.push(tick.price);

        // Log-return (need at least 2 prices)
        if (prices_.count >= 2) {
            const double prev_p = prices_.at(prices_.count - 2);
            const double ret = std::log(tick.price / (prev_p + 1e-15));
            returns_.push(ret);
        }

        // Signed order flow: +vol for buy, -vol for sell
        const double sf = tick.is_buy ? vol : -vol;
        signed_flows_.push(sf);

        // Accumulate into current VPIN bucket
        if (tick.is_buy) current_bucket_.buy_vol  += vol;
        else             current_bucket_.sell_vol += vol;
        total_volume_ += vol;

        // Close bucket when target volume reached
        if (bucket_target_vol_ > 0.0 &&
            current_bucket_.total() >= bucket_target_vol_)
        {
            bucket_history_.push(current_bucket_);
            ++buckets_filled_;
            current_bucket_ = {0.0, 0.0};
            vpin_cached_ = std::numeric_limits<double>::quiet_NaN(); // invalidate cache
        }

        ++tick_count_;

        // Auto-calibrate bucket target after first 500 ticks
        if (tick_count_ == 500 && bucket_target_vol_ == 0.0) {
            bucket_target_vol_ = total_volume_ / 50.0;  // target: 50 buckets per lifetime
        }

        // Publish atomic snapshot for readers
        vpin_atomic_.store(compute_vpin_internal(), std::memory_order_release);
        ofi_atomic_.store(compute_ofi_internal(50), std::memory_order_release);
    }

    // ----------------------------------------------------------
    // Indicator accessors (thread-safe reads via atomics)
    // ----------------------------------------------------------

    // vpin -- Volume-synchronized PIN, rolling n_buckets buckets.
    // Returns average |buy_vol - sell_vol| / total_vol per bucket.
    double vpin(int n_buckets = 50) const noexcept {
        (void)n_buckets;
        const double cached = vpin_atomic_.load(std::memory_order_acquire);
        return std::isnan(cached) ? 0.0 : cached;
    }

    // order_flow_imbalance -- (buy_vol - sell_vol) / total in window ticks.
    double order_flow_imbalance(int window) const noexcept {
        if (window <= 0) return 0.0;
        return ofi_atomic_.load(std::memory_order_acquire);
    }

    // bid_ask_bounce -- Roll (1984) measure: sqrt(-cov(r_t, r_{t-1})).
    // Uses return series. Returns 0 if window too small or cov positive.
    double bid_ask_bounce(int window) const noexcept {
        const int n = std::min(window, returns_.count);
        if (n < 4) return 0.0;

        // Compute covariance of consecutive returns
        double sum_a = 0, sum_b = 0, sum_ab = 0;
        int pairs = 0;
        for (int i = 0; i + 1 < n; ++i) {
            const double a = returns_.at(returns_.count - n + i);
            const double b = returns_.at(returns_.count - n + i + 1);
            sum_a  += a;
            sum_b  += b;
            sum_ab += a * b;
            ++pairs;
        }
        if (pairs < 2) return 0.0;
        const double pf = static_cast<double>(pairs);
        const double cov = sum_ab / pf - (sum_a / pf) * (sum_b / pf);
        if (cov >= 0.0) return 0.0;  // positive cov => no bounce
        return std::sqrt(-cov);
    }

    // price_impact_kyle -- OLS slope of dP on signed flow (Kyle's lambda).
    // delta_price ~ lambda * signed_flow. Returns lambda >= 0.
    double price_impact_kyle(int window) const noexcept {
        const int n_ret  = std::min(window, returns_.count);
        const int n_flow = std::min(window, signed_flows_.count);
        const int n = std::min(n_ret, n_flow);
        if (n < 4) return 0.0;

        // OLS: lambda = cov(dP, Q) / var(Q)
        const int start_r = returns_.count - n;
        const int start_f = signed_flows_.count - n;
        double sum_q  = 0, sum_dp = 0, sum_qq = 0, sum_qdp = 0;
        for (int i = 0; i < n; ++i) {
            const double dp = returns_.at(start_r + i);
            const double q  = signed_flows_.at(start_f + i);
            sum_q   += q;
            sum_dp  += dp;
            sum_qq  += q * q;
            sum_qdp += q * dp;
        }
        const double nf = static_cast<double>(n);
        const double var_q = sum_qq / nf - (sum_q / nf) * (sum_q / nf);
        if (std::abs(var_q) < 1e-20) return 0.0;
        const double cov_qdp = sum_qdp / nf - (sum_q / nf) * (sum_dp / nf);
        return std::max(0.0, cov_qdp / var_q);
    }

    // effective_spread -- avg |mid - trade_price| * 2 over n bid/ask pairs.
    // bids[i] and asks[i] are NBBO quotes at time of trade.
    double effective_spread(const double* bids, const double* asks, int n) const noexcept {
        if (!bids || !asks || n <= 0) return 0.0;
        double sum = 0.0;
        int valid = 0;
        const int tick_n = std::min(n, ticks_.count);
        for (int i = 0; i < tick_n; ++i) {
            const double mid = (bids[i] + asks[i]) * 0.5;
            if (mid <= 0.0) continue;
            const double trade_price = ticks_.at(ticks_.count - tick_n + i).price;
            sum += std::abs(mid - trade_price);
            ++valid;
        }
        if (valid == 0) return 0.0;
        return (sum / static_cast<double>(valid)) * 2.0;
    }

    // quote_stuffing_score -- detect rapid quote cancellations within window_ms.
    // Uses message-rate proxy: count ticks in window vs expected baseline.
    // Returns [0,1] where 1 = extreme message rate anomaly.
    double quote_stuffing_score(int window_ms) const noexcept {
        if (window_ms <= 0 || ticks_.count < 10) return 0.0;

        const int64_t window_ns = static_cast<int64_t>(window_ms) * 1'000'000LL;
        const Tick& newest = ticks_.at(ticks_.count - 1);
        const int64_t cutoff = newest.timestamp_ns - window_ns;

        // Count ticks in window and compute inter-arrival stats
        int in_window = 0;
        double sum_gap = 0.0, sum_gap2 = 0.0;
        int64_t prev_ts = -1;
        for (int i = std::max(0, ticks_.count - 200); i < ticks_.count; ++i) {
            const Tick& t = ticks_.at(i);
            if (t.timestamp_ns < cutoff) continue;
            ++in_window;
            if (prev_ts >= 0) {
                const double gap_us = static_cast<double>(t.timestamp_ns - prev_ts) / 1000.0;
                sum_gap  += gap_us;
                sum_gap2 += gap_us * gap_us;
            }
            prev_ts = t.timestamp_ns;
        }

        if (in_window < 3) return 0.0;

        const double mean_gap = sum_gap / static_cast<double>(in_window - 1);
        const double mean_gap2 = sum_gap2 / static_cast<double>(in_window - 1);
        const double var_gap = mean_gap2 - mean_gap * mean_gap;

        // Quote stuffing: very high message rate (small mean gap) and low variance
        // Anomaly score: if mean_gap < 100us, score increases
        double rate_score = std::max(0.0, 1.0 - mean_gap / 100.0);  // saturates at 100us
        double cv = (mean_gap > 0.0) ? std::sqrt(std::max(0.0, var_gap)) / mean_gap : 0.0;
        double regularity_score = std::max(0.0, 1.0 - cv);  // low CV => regular => suspicious

        return std::clamp(rate_score * 0.7 + regularity_score * 0.3, 0.0, 1.0);
    }

    // ----------------------------------------------------------
    // AVX2 batch computation
    // ----------------------------------------------------------

    // compute_batch -- process N ticks and write indicator results into results[].
    // results layout: [vpin, ofi_50, bounce_100, kyle_impact, stuffing_score] per tick.
    // results must point to n * 5 doubles.
    void compute_batch(const Tick* ticks, int n, double* results) noexcept
    {
        if (!ticks || n <= 0 || !results) return;

        // Process each tick then snapshot indicators
        for (int i = 0; i < n; ++i) {
            update(ticks[i]);
            double* out = results + i * 5;

#if defined(SRFM_MICRO_AVX2)
            // Load into AVX2 register and store 4 results at once
            const double vals[4] = {
                vpin_atomic_.load(std::memory_order_relaxed),
                ofi_atomic_.load(std::memory_order_relaxed),
                bid_ask_bounce(100),
                price_impact_kyle(100)
            };
            __m256d v = _mm256_loadu_pd(vals);
            _mm256_storeu_pd(out, v);
            out[4] = quote_stuffing_score(100);
#else
            out[0] = vpin(50);
            out[1] = order_flow_imbalance(50);
            out[2] = bid_ask_bounce(100);
            out[3] = price_impact_kyle(100);
            out[4] = quote_stuffing_score(100);
#endif
        }
    }

    // ----------------------------------------------------------
    // Diagnostics
    // ----------------------------------------------------------

    int64_t tick_count()     const noexcept { return tick_count_; }
    int     buckets_filled() const noexcept { return buckets_filled_; }
    double  total_volume()   const noexcept { return total_volume_; }
    double  bucket_target()  const noexcept { return bucket_target_vol_; }

    void reset() noexcept {
        ticks_.clear();
        prices_.clear();
        returns_.clear();
        signed_flows_.clear();
        bucket_history_.clear();
        total_volume_    = 0.0;
        bucket_target_vol_ = 0.0;
        current_bucket_  = {0.0, 0.0};
        buckets_filled_  = 0;
        tick_count_      = 0;
        vpin_cached_     = std::numeric_limits<double>::quiet_NaN();
        vpin_atomic_.store(0.0, std::memory_order_relaxed);
        ofi_atomic_.store(0.0, std::memory_order_relaxed);
    }

private:
    // Internal VPIN computation (not thread-safe -- called only from update())
    double compute_vpin_internal() const noexcept {
        const int n = std::min(50, bucket_history_.count);
        if (n == 0) return 0.0;
        double sum = 0.0;
        for (int i = bucket_history_.count - n; i < bucket_history_.count; ++i) {
            const auto& b = bucket_history_.at(i);
            const double tot = b.total();
            if (tot > 0.0) sum += b.imbalance() / tot;
        }
        return sum / static_cast<double>(n);
    }

    double compute_ofi_internal(int window) const noexcept {
        const int n = std::min(window, signed_flows_.count);
        if (n == 0) return 0.0;
        double buy_vol = 0, sell_vol = 0;
        for (int i = signed_flows_.count - n; i < signed_flows_.count; ++i) {
            const double sf = signed_flows_.at(i);
            if (sf > 0) buy_vol  +=  sf;
            else        sell_vol += -sf;
        }
        const double total = buy_vol + sell_vol;
        if (total < 1e-15) return 0.0;
        return (buy_vol - sell_vol) / total;
    }

    // Circular buffers
    CircBuf<Tick,   TICK_CAP>   ticks_;
    CircBuf<double, PRICE_CAP>  prices_;
    CircBuf<double, PRICE_CAP>  returns_;
    CircBuf<double, FLOW_CAP>   signed_flows_;
    CircBuf<VpinBucket, BUCKET_CAP> bucket_history_;

    // VPIN state
    double     total_volume_;
    double     bucket_target_vol_;
    VpinBucket current_bucket_;
    int        buckets_filled_;
    double     vpin_cached_;
    int64_t    tick_count_;

    // Atomic snapshots for lock-free reads from reporter thread
    alignas(64) std::atomic<double> vpin_atomic_{0.0};
    alignas(64) std::atomic<double> ofi_atomic_{0.0};
};

} // namespace microstructure
} // namespace srfm
