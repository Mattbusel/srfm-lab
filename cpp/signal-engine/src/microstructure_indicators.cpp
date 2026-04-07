// microstructure_indicators.cpp -- Implementation stubs and batch utilities.
// The header contains the full class definition and inline methods.
// This file adds:
//   - Out-of-line helpers for complex multipass algorithms
//   - A standalone spread estimator for quote streams
//   - Batch OFI rolling computation with AVX2 inner product
//   - Self-test under SRFM_MICRO_SELFTEST

#include "microstructure_indicators.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>

namespace srfm {
namespace microstructure {

// ============================================================
// Standalone spread utilities (no state -- takes raw arrays)
// ============================================================

// Compute average effective spread over a set of trades and quotes.
// bids[i], asks[i], trade_prices[i], n = count.
double effective_spread_batch(
    const double* bids,
    const double* asks,
    const double* trade_prices,
    int n) noexcept
{
    if (!bids || !asks || !trade_prices || n <= 0) return 0.0;

#if defined(SRFM_MICRO_AVX2)
    // AVX2: process 4 elements at a time
    __m256d sum_v   = _mm256_setzero_pd();
    __m256d half    = _mm256_set1_pd(0.5);
    __m256d two     = _mm256_set1_pd(2.0);
    int valid_count = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vb  = _mm256_loadu_pd(bids + i);
        __m256d va  = _mm256_loadu_pd(asks + i);
        __m256d vtp = _mm256_loadu_pd(trade_prices + i);
        __m256d mid = _mm256_mul_pd(_mm256_add_pd(vb, va), half);
        __m256d diff = _mm256_sub_pd(vtp, mid);
        // abs via bit manipulation: clear sign bit
        __m256d abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff);
        sum_v = _mm256_add_pd(sum_v, abs_diff);
        valid_count += 4;
    }
    // Horizontal sum of AVX2 register
    __m128d lo = _mm256_castpd256_pd128(sum_v);
    __m128d hi = _mm256_extractf128_pd(sum_v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    __m128d sh = _mm_unpackhi_pd(s, s);
    double total = _mm_cvtsd_f64(_mm_add_sd(s, sh));
    // Scalar tail
    for (; i < n; ++i) {
        const double mid = (bids[i] + asks[i]) * 0.5;
        total += std::abs(trade_prices[i] - mid);
        ++valid_count;
    }
    return (valid_count > 0) ? (total / static_cast<double>(valid_count)) * 2.0 : 0.0;
#else
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        const double mid = (bids[i] + asks[i]) * 0.5;
        total += std::abs(trade_prices[i] - mid);
    }
    return (total / static_cast<double>(n)) * 2.0;
#endif
}

// Compute rolling OFI over a signed-volume array using AVX2 inner product.
// signed_vol[i] > 0 => buy, < 0 => sell.
// out[i] = OFI for window ending at i.
// out must have n elements.
void rolling_ofi_batch(
    const double* signed_vol,
    int n,
    int window,
    double* out) noexcept
{
    if (!signed_vol || n <= 0 || window <= 0 || !out) return;

    for (int i = 0; i < n; ++i) {
        const int start = std::max(0, i - window + 1);
        const int len   = i - start + 1;
        double buy_vol = 0.0, sell_vol = 0.0;

#if defined(SRFM_MICRO_AVX2)
        __m256d buy_acc  = _mm256_setzero_pd();
        __m256d sell_acc = _mm256_setzero_pd();
        __m256d zero     = _mm256_setzero_pd();
        int j = start;
        for (; j + 4 <= start + len; j += 4) {
            __m256d sv = _mm256_loadu_pd(signed_vol + j);
            // Buy: max(sv, 0)
            __m256d bv = _mm256_max_pd(sv, zero);
            // Sell: max(-sv, 0)
            __m256d neg_sv = _mm256_sub_pd(zero, sv);
            __m256d sv2 = _mm256_max_pd(neg_sv, zero);
            buy_acc  = _mm256_add_pd(buy_acc, bv);
            sell_acc = _mm256_add_pd(sell_acc, sv2);
        }
        // Horizontal sum
        {
            __m128d lo = _mm256_castpd256_pd128(buy_acc);
            __m128d hi = _mm256_extractf128_pd(buy_acc, 1);
            __m128d s  = _mm_add_pd(lo, hi);
            __m128d sh = _mm_unpackhi_pd(s, s);
            buy_vol = _mm_cvtsd_f64(_mm_add_sd(s, sh));
        }
        {
            __m128d lo = _mm256_castpd256_pd128(sell_acc);
            __m128d hi = _mm256_extractf128_pd(sell_acc, 1);
            __m128d s  = _mm_add_pd(lo, hi);
            __m128d sh = _mm_unpackhi_pd(s, s);
            sell_vol = _mm_cvtsd_f64(_mm_add_sd(s, sh));
        }
        // Scalar tail
        for (; j < start + len; ++j) {
            if (signed_vol[j] > 0) buy_vol  +=  signed_vol[j];
            else                   sell_vol += -signed_vol[j];
        }
#else
        for (int j = start; j <= i; ++j) {
            if (signed_vol[j] > 0) buy_vol  +=  signed_vol[j];
            else                   sell_vol += -signed_vol[j];
        }
#endif
        const double total = buy_vol + sell_vol;
        out[i] = (total > 1e-15) ? (buy_vol - sell_vol) / total : 0.0;
    }
}

// ============================================================
// Roll measure batch over a returns array
// ============================================================

// rolling_roll_measure -- compute bid-ask bounce via Roll (1984) for each index.
// returns_arr[i] = log(P_i / P_{i-1}).
// out[i] = sqrt(max(0, -cov)) over [i-window+1, i].
void rolling_roll_measure(
    const double* returns_arr,
    int n,
    int window,
    double* out) noexcept
{
    if (!returns_arr || n <= 0 || window < 4 || !out) return;

    for (int i = 0; i < n; ++i) {
        out[i] = 0.0;
        const int end = i;
        const int start = std::max(1, end - window + 1); // need at least index 1
        const int len = end - start + 1;
        if (len < 3) continue;

        double sum_a = 0, sum_b = 0, sum_ab = 0;
        int pairs = 0;
        for (int j = start; j < end; ++j) {
            const double a = returns_arr[j];
            const double b = returns_arr[j + 1];
            sum_a  += a;
            sum_b  += b;
            sum_ab += a * b;
            ++pairs;
        }
        if (pairs < 2) continue;
        const double pf = static_cast<double>(pairs);
        const double cov = sum_ab / pf - (sum_a / pf) * (sum_b / pf);
        out[i] = (cov < 0.0) ? std::sqrt(-cov) : 0.0;
    }
}

// ============================================================
// VPIN multi-window helper -- compute VPIN for several bucket windows
// ============================================================

// Compute VPIN for each requested window_size from bucket_imbalances.
// bucket_imbalances[i] = |buy_vol - sell_vol| / total_vol for bucket i.
// windows[w] = number of buckets to average over.
// out[w] = VPIN for that window.
void vpin_multi_window(
    const double* bucket_imbalances,
    int n_buckets,
    const int* windows,
    int n_windows,
    double* out) noexcept
{
    if (!bucket_imbalances || !windows || !out) return;

    for (int w = 0; w < n_windows; ++w) {
        const int k = std::min(windows[w], n_buckets);
        if (k == 0) { out[w] = 0.0; continue; }

        double sum = 0.0;
#if defined(SRFM_MICRO_AVX2)
        __m256d acc = _mm256_setzero_pd();
        int j = n_buckets - k;
        for (; j + 4 <= n_buckets; j += 4) {
            __m256d v = _mm256_loadu_pd(bucket_imbalances + j);
            acc = _mm256_add_pd(acc, v);
        }
        __m128d lo = _mm256_castpd256_pd128(acc);
        __m128d hi = _mm256_extractf128_pd(acc, 1);
        __m128d s  = _mm_add_pd(lo, hi);
        __m128d sh = _mm_unpackhi_pd(s, s);
        sum = _mm_cvtsd_f64(_mm_add_sd(s, sh));
        for (; j < n_buckets; ++j) sum += bucket_imbalances[j];
#else
        for (int j = n_buckets - k; j < n_buckets; ++j)
            sum += bucket_imbalances[j];
#endif
        out[w] = sum / static_cast<double>(k);
    }
}

// ============================================================
// Self-test
// ============================================================

#ifdef SRFM_MICRO_SELFTEST

static void check(bool cond, const char* label) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::abort();
    }
    std::fprintf(stdout, "PASS: %s\n", label);
}

void run_microstructure_selftest() {
    MicrostructureIndicators mi;
    check(mi.tick_count() == 0, "initial_tick_count");

    // Feed 100 ticks alternating buy/sell
    double price = 100.0;
    for (int i = 0; i < 200; ++i) {
        price += (i % 2 == 0) ? 0.01 : -0.005;
        Tick t;
        t.price        = price;
        t.volume       = 100.0 + static_cast<double>(i % 10) * 10.0;
        t.is_buy       = (i % 3 != 0);
        t.timestamp_ns = static_cast<int64_t>(i) * 1'000'000LL;
        mi.update(t);
    }

    check(mi.tick_count() == 200, "tick_count_after_200");

    const double ofi = mi.order_flow_imbalance(50);
    check(ofi >= -1.0 && ofi <= 1.0, "ofi_range");

    const double bounce = mi.bid_ask_bounce(50);
    check(bounce >= 0.0, "bounce_non_negative");

    const double kyle = mi.price_impact_kyle(50);
    check(kyle >= 0.0, "kyle_non_negative");

    // Test effective_spread standalone
    std::vector<double> bids(20, 99.90), asks(20, 100.10), tps(20, 100.00);
    double esp = effective_spread_batch(bids.data(), asks.data(), tps.data(), 20);
    // mid = 100, trade = 100, |mid-trade|*2 = 0
    check(esp < 0.01, "eff_spread_at_mid");

    // Test rolling OFI
    std::vector<double> sv = { 100, -50, 200, -100, 150, -80, 300, -200 };
    std::vector<double> ofi_out(sv.size(), 0.0);
    rolling_ofi_batch(sv.data(), static_cast<int>(sv.size()), 4, ofi_out.data());
    check(ofi_out[3] >= -1.0 && ofi_out[3] <= 1.0, "rolling_ofi_range");

    // Test VPIN multi-window
    std::vector<double> imbalances = { 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 };
    int wins[2] = { 3, 6 };
    double vpin_out[2] = {};
    vpin_multi_window(imbalances.data(), static_cast<int>(imbalances.size()),
                      wins, 2, vpin_out);
    check(vpin_out[0] > 0.5 && vpin_out[0] < 0.8, "vpin_window3");
    check(vpin_out[1] > 0.3 && vpin_out[1] < 0.7, "vpin_window6");

    std::fprintf(stdout, "All microstructure self-tests passed.\n");
}

#endif // SRFM_MICRO_SELFTEST

} // namespace microstructure
} // namespace srfm
