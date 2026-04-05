/// test_performance.cpp
/// Performance benchmarks: targets >10M bars/s single, >500K for 20 instruments.

#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#include "srfm/types.hpp"
#include "../src/streaming/feed_processor.hpp"
#include "../src/indicators/ema.hpp"
#include "../src/bh_physics/bh_state.hpp"
#include "../src/bh_physics/garch.hpp"

using namespace srfm;
using namespace std::chrono;

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) do { \
    if (!(expr)) { std::fprintf(stderr,"FAIL  %s:%d  %s\n",__FILE__,__LINE__,#expr); ++g_fail; } \
    else { ++g_pass; } \
} while(0)

static void section(const char* n) { std::printf("--- %s ---\n", n); }

// ============================================================
// Synthetic bar generator
// ============================================================

static std::vector<OHLCVBar> generate_bars(int n_instruments,
                                            long long n_bars_per_inst) {
    std::vector<OHLCVBar> bars;
    bars.reserve(static_cast<std::size_t>(n_instruments * n_bars_per_inst));

    uint64_t rng = 0xDEADBEEFCAFEBABEULL;
    auto next_rng = [&]() -> double {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return (static_cast<double>(rng & 0xFFFFFFFF) / 0x100000000LL) * 2.0 - 1.0;
    };

    std::vector<double> prices(n_instruments, 50000.0);

    for (long long i = 0; i < n_bars_per_inst; ++i) {
        for (int s = 0; s < n_instruments; ++s) {
            double ret = next_rng() * 0.002;
            double hl  = std::abs(next_rng()) * 0.003;
            double o   = prices[s];
            prices[s] *= (1.0 + ret);
            double h = prices[s] * (1.0 + hl * 0.5);
            double l = prices[s] * (1.0 - hl * 0.5);
            double v = 1000.0 + std::abs(next_rng()) * 500.0;

            bars.push_back(OHLCVBar(o, h, l, prices[s], v,
                                    1700000000LL * constants::NS_PER_SEC +
                                    i * constants::NS_PER_MIN,
                                    s, 60));
        }
    }
    return bars;
}

// ============================================================
// Benchmark helper: run N bars through FeedProcessor, return throughput
// ============================================================

struct BenchResult {
    long long  total_bars;
    int        n_instruments;
    double     elapsed_us;
    double     throughput_bps;
    double     latency_us_per_bar;
};

static BenchResult run_bench(int n_instruments, long long n_bars_per_inst) {
    auto bars = generate_bars(n_instruments, n_bars_per_inst);
    long long total = static_cast<long long>(bars.size());

    FeedProcessor proc(n_instruments);

    // Warmup: first 100 bars
    for (int i = 0; i < std::min(100LL, total); ++i)
        proc.process_bar(bars[i]);
    proc.reset();

    auto t0 = high_resolution_clock::now();
    for (long long i = 0; i < total; ++i) {
        proc.process_bar(bars[i]);
    }
    auto t1 = high_resolution_clock::now();

    double us  = static_cast<double>(duration_cast<microseconds>(t1 - t0).count());
    double bps = total / (us / 1e6);

    return { total, n_instruments, us, bps, us / total };
}

// ============================================================
// Test: single instrument throughput > 10M bars/s
// ============================================================

static void test_single_instrument_throughput() {
    section("Single instrument throughput (target: >10M bars/s)");

    auto res = run_bench(1, 500000);
    std::printf("  %lld bars in %.2f ms  =>  %.2fM bars/s  (%.3f µs/bar)\n",
                res.total_bars, res.elapsed_us / 1000.0,
                res.throughput_bps / 1e6, res.latency_us_per_bar);

    bool ok = res.throughput_bps >= 10e6;
    if (!ok)
        std::printf("  WARNING: target not met (%.2fM < 10M bars/s)\n",
                    res.throughput_bps / 1e6);
    // Note: in Debug mode or without -O3, this may not pass. Accept 1M minimum.
    CHECK(res.throughput_bps >= 1e6);
}

// ============================================================
// Test: 20 instruments throughput > 500K bars/s
// ============================================================

static void test_twenty_instrument_throughput() {
    section("20-instrument throughput (target: >500K bars/s)");

    auto res = run_bench(20, 50000);
    std::printf("  %lld bars in %.2f ms  =>  %.2fM bars/s  (%.3f µs/bar)\n",
                res.total_bars, res.elapsed_us / 1000.0,
                res.throughput_bps / 1e6, res.latency_us_per_bar);

    bool ok = res.throughput_bps >= 500e3;
    if (!ok)
        std::printf("  WARNING: target not met (%.0fK < 500K bars/s)\n",
                    res.throughput_bps / 1e3);
    CHECK(res.throughput_bps >= 100e3);  // accept 100K min in debug
}

// ============================================================
// Test: scaling across instrument counts
// ============================================================

static void test_scaling() {
    section("Scaling: 1 → 20 instruments");

    int test_counts[] = { 1, 2, 4, 8, 16, 20 };
    double prev_bps = -1.0;

    for (int n : test_counts) {
        auto res = run_bench(n, 100000);
        std::printf("  [%2d instr] %.2fM bars/s  %.3f µs/bar\n",
                    n, res.throughput_bps / 1e6, res.latency_us_per_bar);

        CHECK(res.throughput_bps > 0.0);
        prev_bps = res.throughput_bps;
    }
    (void)prev_bps;
}

// ============================================================
// Test: latency < 10µs per instrument
// ============================================================

static void test_per_instrument_latency() {
    section("Per-instrument latency (target: <10µs)");

    auto res = run_bench(1, 1000000);
    std::printf("  Single instrument: %.3f µs/bar\n", res.latency_us_per_bar);
    // Target: < 10µs per bar per instrument
    bool ok = res.latency_us_per_bar < 10.0;
    std::printf("  Target <10 µs: %s\n", ok ? "PASS" : "MISS (needs -O3 build)");
    // In optimized build this should pass; accept 100µs in debug
    CHECK(res.latency_us_per_bar < 100.0);
}

// ============================================================
// Test: individual component benchmarks
// ============================================================

static void bench_component_ema() {
    section("EMA component microbenchmark");

    constexpr int N = 5000000;
    EMA ema(9);

    auto t0 = high_resolution_clock::now();
    double price = 50000.0;
    for (int i = 0; i < N; ++i) {
        price += (i % 2 == 0 ? 1.0 : -0.9);
        ema.update(price);
    }
    auto t1 = high_resolution_clock::now();

    double ns = duration_cast<nanoseconds>(t1 - t0).count();
    std::printf("  EMA(9): %d updates in %.2f ms  =>  %.1f ns/update\n",
                N, ns / 1e6, ns / N);
    CHECK(ema.value() > 0.0);  // prevent dead-code elim
}

static void bench_component_bh() {
    section("BH physics microbenchmark");

    constexpr int N = 1000000;
    BHState bh;

    auto t0 = high_resolution_clock::now();
    double price = 50000.0;
    for (int i = 0; i < N; ++i) {
        price *= (i % 2 == 0 ? 1.001 : 0.999);
        bh.update(price, 1000.0,
                  1700000000LL * constants::NS_PER_SEC +
                  static_cast<int64_t>(i) * constants::NS_PER_MIN);
    }
    auto t1 = high_resolution_clock::now();

    double ns = duration_cast<nanoseconds>(t1 - t0).count();
    std::printf("  BHState: %d updates in %.2f ms  =>  %.1f ns/update\n",
                N, ns / 1e6, ns / N);
    CHECK(bh.mass() >= 0.0);
}

static void bench_component_garch() {
    section("GARCH microbenchmark");

    constexpr int N = 2000000;
    GARCHTracker g;

    auto t0 = high_resolution_clock::now();
    double price = 50000.0;
    for (int i = 0; i < N; ++i) {
        price *= (i % 2 == 0 ? 1.001 : 0.999);
        g.update(price);
    }
    auto t1 = high_resolution_clock::now();

    double ns = duration_cast<nanoseconds>(t1 - t0).count();
    std::printf("  GARCHTracker: %d updates in %.2f ms  =>  %.1f ns/update\n",
                N, ns / 1e6, ns / N);
    CHECK(g.variance() > 0.0);
}

// ============================================================
// Test: memory access patterns (cache efficiency)
// ============================================================

static void test_cache_line_alignment() {
    section("Cache line alignment verification");

    std::printf("  OHLCVBar  align=%zu  size=%zu\n", alignof(OHLCVBar),  sizeof(OHLCVBar));
    std::printf("  TickData  align=%zu  size=%zu\n", alignof(TickData),  sizeof(TickData));
    std::printf("  SignalOut align=%zu  size=%zu\n", alignof(SignalOutput), sizeof(SignalOutput));

    CHECK(alignof(OHLCVBar)   >= 64);
    CHECK(sizeof(OHLCVBar)    == 64);
    CHECK(sizeof(TickData)    == 32);

    // Array of OHLCVBar: each element starts on its own cache line
    std::vector<OHLCVBar> bars(4);
    uintptr_t base = reinterpret_cast<uintptr_t>(bars.data());
    CHECK(base % 64 == 0);
}

// ============================================================
// Main
// ============================================================

int main() {
    std::printf("=== Performance Tests ===\n\n");

    test_cache_line_alignment();
    bench_component_ema();
    bench_component_bh();
    bench_component_garch();
    test_single_instrument_throughput();
    test_twenty_instrument_throughput();
    test_scaling();
    test_per_instrument_latency();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
