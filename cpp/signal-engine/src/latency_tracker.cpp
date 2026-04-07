// latency_tracker.cpp -- Out-of-line utilities for LatencyTracker.
// Includes:
//   - Histogram merge (combine two trackers into one)
//   - CSV export
//   - Self-test under SRFM_LATENCY_SELFTEST

#include "latency_tracker.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cassert>
#include <thread>
#include <chrono>

namespace srfm {
namespace metrics {

// ============================================================
// Histogram merge -- add src into dst (both must track same ops)
// ============================================================

void merge_histograms(LatencyTracker& dst, const LatencyTracker& src) noexcept {
    // For each op in src, replicate its histogram counts into dst.
    for (int i = 0; i < src.op_count(); ++i) {
        const char* name = src.op_name(i);
        const LatencyStats s = src.get_stats(name);
        // Re-inject synthetic samples at each percentile boundary to approximate the merge.
        // This is an approximation -- a full merge would copy bucket arrays directly.
        // For now we inject known percentile points as individual samples.
        if (s.count == 0) continue;
        // Inject p50/p95/p99 samples proportionally to approximate the source distribution.
        // 50% of samples at p50, 45% at p95, 4% at p99, 1% at max.
        const uint64_t n50 = s.count / 2;
        const uint64_t n95 = (s.count * 45) / 100;
        const uint64_t n99 = (s.count * 4)  / 100;
        const uint64_t nmax = s.count - n50 - n95 - n99;
        for (uint64_t k = 0; k < n50;  ++k) dst.record_ns(name, s.p50);
        for (uint64_t k = 0; k < n95;  ++k) dst.record_ns(name, s.p95);
        for (uint64_t k = 0; k < n99;  ++k) dst.record_ns(name, s.p99);
        for (uint64_t k = 0; k < nmax; ++k) dst.record_ns(name, s.max);
    }
}

// ============================================================
// CSV export -- write all op stats to a CSV file.
// Header: op_name,count,p50_ns,p95_ns,p99_ns,min_ns,max_ns,mean_ns
// ============================================================

bool export_csv(const LatencyTracker& tracker, const char* path) {
    FILE* f = std::fopen(path, "w");
    if (!f) return false;

    std::fprintf(f, "op_name,count,p50_ns,p95_ns,p99_ns,min_ns,max_ns,mean_ns\n");
    for (int i = 0; i < tracker.op_count(); ++i) {
        const char* name = tracker.op_name(i);
        const LatencyStats s = tracker.get_stats(name);
        std::fprintf(f, "%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64
                        ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                     s.op_name, s.count,
                     s.p50, s.p95, s.p99,
                     s.min, s.max, s.mean);
    }

    std::fclose(f);
    return true;
}

// ============================================================
// Human-readable report -- print all stats to stdout
// ============================================================

void print_report(const LatencyTracker& tracker) {
    std::fprintf(stdout,
        "%-30s %10s %10s %10s %10s %10s %10s %10s\n",
        "op_name", "count", "p50_us", "p95_us", "p99_us",
        "min_us", "max_us", "mean_us");
    std::fprintf(stdout, "%s\n", std::string(100, '-').c_str());

    for (int i = 0; i < tracker.op_count(); ++i) {
        const char* name = tracker.op_name(i);
        const LatencyStats s = tracker.get_stats(name);
        if (s.count == 0) continue;
        // Convert ns to us for readability (with one decimal)
        auto ns_to_us = [](uint64_t ns) { return static_cast<double>(ns) / 1000.0; };
        std::fprintf(stdout,
            "%-30s %10" PRIu64 " %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f\n",
            s.op_name, s.count,
            ns_to_us(s.p50), ns_to_us(s.p95), ns_to_us(s.p99),
            ns_to_us(s.min), ns_to_us(s.max), ns_to_us(s.mean));
    }
}

// ============================================================
// Self-test
// ============================================================

#ifdef SRFM_LATENCY_SELFTEST

static void check(bool cond, const char* label) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::abort();
    }
    std::fprintf(stdout, "PASS: %s\n", label);
}

void run_latency_selftest() {
    LatencyTracker tracker;

    // Record some known latencies (in ns)
    const char* op = "signal_compute";
    tracker.record_ns(op,    1000);   // 1 us
    tracker.record_ns(op,    2000);   // 2 us
    tracker.record_ns(op,    3000);   // 3 us
    tracker.record_ns(op,   10000);   // 10 us
    tracker.record_ns(op,   20000);   // 20 us
    tracker.record_ns(op,  100000);   // 100 us
    tracker.record_ns(op, 1000000);   // 1 ms (outlier)

    LatencyStats s = tracker.get_stats(op);
    check(s.count == 7, "count_7");
    check(s.p50 > 0, "p50_nonzero");
    check(s.p99 >= s.p50, "p99_ge_p50");
    check(s.max >= s.min, "max_ge_min");

    // p99 should be close to the 1ms outlier bucket
    check(s.p99 >= 500000, "p99_near_1ms");

    // JSON export
    std::string json = tracker.to_json();
    check(json.find("signal_compute") != std::string::npos, "json_has_op");
    check(json.find("count") != std::string::npos, "json_has_count");
    std::fprintf(stdout, "JSON snippet: %.200s\n", json.c_str());

    // Test reset
    tracker.reset(op);
    LatencyStats s2 = tracker.get_stats(op);
    check(s2.count == 0, "count_after_reset");

    // Test multiple ops
    tracker.record_ns("order_send",   50000);
    tracker.record_ns("order_send",   80000);
    tracker.record_ns("market_data",   5000);
    tracker.record_ns("market_data",   6000);
    check(tracker.op_count() >= 2, "multiple_ops");

    LatencyStats ms = tracker.get_stats("market_data");
    check(ms.count == 2, "market_data_count");
    check(ms.max > ms.min || ms.max == ms.min, "market_data_minmax");

    // Test reset_all
    tracker.reset_all();
    LatencyStats s3 = tracker.get_stats("order_send");
    check(s3.count == 0, "all_reset");

    // Test ScopedTimer (basic -- just checks it doesn't crash)
    {
        ScopedTimer t(tracker, "scoped_test");
        // Simulate some work
        volatile int x = 0;
        for (int i = 0; i < 1000; ++i) x += i;
        (void)x;
    }
    LatencyStats st = tracker.get_stats("scoped_test");
    check(st.count == 1, "scoped_timer_count");

    // Test concurrent recording from 4 threads
    {
        tracker.reset_all();
        constexpr int N_THREADS = 4;
        constexpr int N_RECORDS = 500;
        std::vector<std::thread> threads;
        for (int t = 0; t < N_THREADS; ++t) {
            threads.emplace_back([&tracker, t]() {
                for (int i = 0; i < N_RECORDS; ++i) {
                    tracker.record_ns("concurrent_op",
                        static_cast<uint64_t>((t + 1) * (i + 1) * 1000));
                }
            });
        }
        for (auto& th : threads) th.join();

        LatencyStats cs = tracker.get_stats("concurrent_op");
        check(cs.count == N_THREADS * N_RECORDS, "concurrent_count");
    }

    std::fprintf(stdout, "All latency_tracker self-tests passed.\n");
}

#endif // SRFM_LATENCY_SELFTEST

} // namespace metrics
} // namespace srfm
