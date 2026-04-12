// telemetry.cpp — rdtsc cycle counter, lock-free stats, CSV/binary metrics export.
// Chronos / AETERNUS — production C++ telemetry implementation.

#include "telemetry.hpp"
#include <cstdio>
#include <cassert>
#include <thread>
#include <cmath>

namespace chronos {
namespace telemetry {

// ── TSC calibration ───────────────────────────────────────────────────────────

double calibrate_tsc_freq_ghz() {
    constexpr int CALIBRATION_MS = 20;
    uint64_t t0 = rdtsc();
    auto wall0 = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(CALIBRATION_MS));
    uint64_t t1 = rdtsc();
    auto wall1 = std::chrono::high_resolution_clock::now();
    double elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(wall1 - wall0).count();
    double cycles = static_cast<double>(t1 - t0);
    return cycles / elapsed_ns; // GHz (cycles per ns)
}

// ── Formatted output helpers ──────────────────────────────────────────────────

static std::string format_ns(uint64_t ns) {
    if (ns < 1000) return std::to_string(ns) + "ns";
    if (ns < 1'000'000) return std::to_string(ns / 1000) + "µs";
    if (ns < 1'000'000'000) return std::to_string(ns / 1'000'000) + "ms";
    return std::to_string(ns / 1'000'000'000) + "s";
}

static std::string histogram_ascii_bar(uint64_t value, uint64_t max_value, int width = 40) {
    if (max_value == 0) return std::string(width, ' ');
    int filled = static_cast<int>(static_cast<double>(value) / max_value * width);
    std::string bar(filled, '#');
    bar.resize(width, ' ');
    return "[" + bar + "]";
}

// ── Latency histogram printer ─────────────────────────────────────────────────

std::string print_histogram(const LatencyHistogram& h, int n_buckets = 20) {
    if (h.count() == 0) return "histogram: (empty)\n";

    std::ostringstream oss;
    oss << "Latency Histogram (n=" << h.count() << "):\n";

    uint64_t min_ns = h.min_ns();
    uint64_t max_ns = h.max_ns();
    if (min_ns == max_ns) {
        oss << "  All values: " << format_ns(min_ns) << "\n";
        return oss.str();
    }

    // Build display buckets
    uint64_t range = max_ns - min_ns;
    uint64_t bucket_width = range / n_buckets + 1;
    std::vector<uint64_t> counts(n_buckets, 0);
    uint64_t max_count = 0;

    // Use percentile sampling to estimate bucket distribution
    for (int i = 0; i < n_buckets; ++i) {
        double pct_lo = 100.0 * i / n_buckets;
        double pct_hi = 100.0 * (i + 1) / n_buckets;
        uint64_t p_lo = h.percentile(pct_lo);
        uint64_t p_hi = h.percentile(pct_hi);
        counts[i] = (p_hi > p_lo) ? (p_hi - p_lo) : 1;
        max_count = std::max(max_count, counts[i]);
    }

    for (int i = 0; i < n_buckets; ++i) {
        uint64_t bucket_min = min_ns + static_cast<uint64_t>(i) * bucket_width;
        uint64_t bucket_max = bucket_min + bucket_width;
        oss << std::setw(8) << format_ns(bucket_min) << " - "
            << std::setw(8) << format_ns(bucket_max) << " "
            << histogram_ascii_bar(counts[i], max_count, 30) << "\n";
    }

    oss << "  p50=" << format_ns(h.p50())
        << " p95=" << format_ns(h.p95())
        << " p99=" << format_ns(h.p99())
        << " p999=" << format_ns(h.p999()) << "\n";
    return oss.str();
}

// ── Per-event latency tracker ─────────────────────────────────────────────────

struct LatencyEvent {
    uint64_t start_cycles;
    uint32_t event_type;
    uint32_t flags;
};

class CycleTimer {
public:
    static uint64_t start() noexcept { return rdtsc(); }
    static uint64_t stop(uint64_t start_cycles) noexcept { return rdtscp() - start_cycles; }
    static double to_ns(uint64_t cycles, double tsc_ghz = 1.0) noexcept {
        return cycles_to_ns(cycles, tsc_ghz);
    }
};

// ── Scoped timer ──────────────────────────────────────────────────────────────

class ScopedTimer {
public:
    explicit ScopedTimer(LatencyHistogram& hist, double tsc_ghz = 1.0)
        : hist_(hist), start_(rdtsc()), tsc_ghz_(tsc_ghz) {}

    ~ScopedTimer() {
        uint64_t cycles = rdtscp() - start_;
        uint64_t ns = static_cast<uint64_t>(cycles_to_ns(cycles, tsc_ghz_));
        hist_.record(ns);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    LatencyHistogram& hist_;
    uint64_t start_;
    double tsc_ghz_;
};

// ── Stats dump ────────────────────────────────────────────────────────────────

std::string dump_stats_report(const TelemetryHub& hub) {
    std::ostringstream oss;
    oss << "=== Chronos Telemetry Report ===\n";
    oss << "Orders: " << hub.orders_total()
        << " | Fills: " << hub.fills_total()
        << " | Rate: " << std::fixed << std::setprecision(0) << hub.msg_rate_per_sec() << " msg/s\n";
    oss << "Order latency: " << hub.order_latency().summary() << "\n";
    oss << "Fill  latency: " << hub.fill_latency().summary() << "\n";
    oss << "Uptime: ";
    uint64_t up = hub.uptime_ns();
    if (up < 1e9) oss << up / 1'000'000 << "ms";
    else oss << up / 1'000'000'000 << "s";
    oss << "\n";
    return oss.str();
}

// ── Binary metrics file I/O ───────────────────────────────────────────────────

bool write_metrics_binary(const MetricsLog& log, const std::string& path) {
    auto bytes = log.to_binary();
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t written = fwrite(bytes.data(), 1, bytes.size(), f);
    fclose(f);
    return written == bytes.size();
}

bool write_metrics_csv(const MetricsLog& log, const std::string& path) {
    auto csv = log.to_csv();
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    fwrite(csv.data(), 1, csv.size(), f);
    fclose(f);
    return true;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#ifndef TELEMETRY_NO_TESTS

namespace tests {

void test_rdtsc() {
    uint64_t t0 = rdtsc();
    volatile int sum = 0;
    for (int i = 0; i < 10000; ++i) sum += i;
    uint64_t t1 = rdtsc();
    assert(t1 > t0);
    (void)sum;
    printf("test_rdtsc: PASSED (delta=%llu cycles)\n", (unsigned long long)(t1 - t0));
}

void test_atomic_stats() {
    AtomicStats stats;
    for (uint64_t v : {100ULL, 200ULL, 300ULL, 400ULL, 500ULL}) {
        stats.record(v);
    }
    assert(stats.get_count() == 5);
    assert(std::abs(stats.mean() - 300.0) < 0.01);
    assert(stats.get_min() == 100);
    assert(stats.get_max() == 500);
    printf("test_atomic_stats: PASSED\n");
}

void test_latency_histogram() {
    LatencyHistogram h;
    for (uint64_t i = 1; i <= 1000; ++i) h.record(i * 1000);
    assert(h.count() == 1000);
    assert(h.p50() < h.p99());
    assert(h.p99() < h.p999());
    printf("test_latency_histogram: p50=%lluns p99=%lluns p999=%lluns PASSED\n",
        (unsigned long long)h.p50(), (unsigned long long)h.p99(), (unsigned long long)h.p999());
}

void test_histogram_reset() {
    LatencyHistogram h;
    h.record(1000);
    h.record(2000);
    assert(h.count() == 2);
    h.reset();
    assert(h.count() == 0);
    printf("test_histogram_reset: PASSED\n");
}

void test_ring_buffer() {
    RingBuffer<uint64_t, 16> rb;
    assert(rb.empty());
    assert(rb.push(42));
    assert(rb.push(99));
    uint64_t v;
    assert(rb.pop(v) && v == 42);
    assert(rb.pop(v) && v == 99);
    assert(!rb.pop(v));
    printf("test_ring_buffer: PASSED\n");
}

void test_metrics_log() {
    MetricsLog log(64);
    MetricRecord r{now_ns(), 1, 0, 0, 3.14, 0.0};
    log.write(r);
    log.write(r);
    assert(log.total_written() == 2);
    std::string csv = log.to_csv();
    assert(csv.find("3.14") != std::string::npos);
    printf("test_metrics_log: PASSED\n");
}

void test_throughput_counter() {
    ThroughputCounter tc(0.1); // 100ms window
    tc.record(100);
    tc.record(200);
    assert(tc.total() == 300);
    printf("test_throughput_counter: PASSED total=%llu\n", (unsigned long long)tc.total());
}

void test_prometheus_exporter() {
    auto hist = std::make_shared<LatencyHistogram>();
    for (uint64_t i = 100; i <= 10000; i += 100) hist->record(i);

    std::atomic<uint64_t> counter{42};
    PrometheusExporter exp;
    exp.register_counter("test_counter", "A test counter", &counter);
    exp.register_histogram("test_latency", hist);

    std::string output = exp.exposition("test_");
    assert(output.find("test_test_counter") != std::string::npos);
    assert(output.find("42") != std::string::npos);
    assert(output.find("quantile") != std::string::npos);
    printf("test_prometheus_exporter: PASSED\n");
}

void test_telemetry_hub() {
    TelemetryHub hub;
    hub.record_order(500);
    hub.record_order(800);
    hub.record_fill(300, 0.05);
    hub.record_error();

    assert(hub.orders_total() == 2);
    assert(hub.fills_total() == 1);

    std::string prom = hub.prometheus_exposition();
    assert(prom.find("chronos_orders_total") != std::string::npos);
    assert(prom.find("2 ") != std::string::npos || prom.find("2\n") != std::string::npos);

    std::string csv = hub.metrics_csv();
    assert(!csv.empty());
    printf("test_telemetry_hub: PASSED\n");
}

void test_scoped_timer() {
    double tsc_ghz = calibrate_tsc_freq_ghz();
    LatencyHistogram hist;
    {
        ScopedTimer t(hist, tsc_ghz);
        volatile int sum = 0;
        for (int i = 0; i < 1000; ++i) sum += i;
        (void)sum;
    }
    assert(hist.count() == 1);
    assert(hist.min_ns() > 0);
    printf("test_scoped_timer: PASSED (%lluns elapsed)\n", (unsigned long long)hist.min_ns());
}

void run_all() {
    test_rdtsc();
    test_atomic_stats();
    test_latency_histogram();
    test_histogram_reset();
    test_ring_buffer();
    test_metrics_log();
    test_throughput_counter();
    test_prometheus_exporter();
    test_telemetry_hub();
    test_scoped_timer();
    printf("All telemetry tests PASSED\n");
}

} // namespace tests

#endif // TELEMETRY_NO_TESTS

} // namespace telemetry
} // namespace chronos

#ifndef TELEMETRY_NO_MAIN
int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--test") {
        chronos::telemetry::tests::run_all();
        return 0;
    }
    printf("Telemetry engine — Chronos/AETERNUS\n");
    printf("Usage: telemetry --test\n");
    return 0;
}
#endif
