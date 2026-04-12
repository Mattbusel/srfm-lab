// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// ring_buffer.cpp — Lock-Free SPMC Ring Buffer Implementation
// =============================================================================
// This file contains explicit instantiations of the RingBuffer template
// for common types, as well as utility functions for histogram reporting,
// cycle conversion, and throughput analysis.
// =============================================================================

#include "rtel/ring_buffer.hpp"
#include "rtel/shm_bus.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------
template class RingBuffer<MarketEvent>;
template class RingBuffer<StageEvent>;
template class SPMCQueue<MarketEvent>;
template class SPMCQueue<StageEvent>;

// ---------------------------------------------------------------------------
// CycleConverter — calibrates RDTSC frequency to convert cycles → ns
// ---------------------------------------------------------------------------
class CycleConverter {
public:
    static CycleConverter& instance() {
        static CycleConverter cc;
        return cc;
    }

    double cycles_per_ns() const noexcept { return cycles_per_ns_; }
    double ns_per_cycle()  const noexcept { return 1.0 / cycles_per_ns_; }

    uint64_t cycles_to_ns(uint64_t cycles) const noexcept {
        return static_cast<uint64_t>(cycles * ns_per_cycle());
    }
    uint64_t ns_to_cycles(uint64_t ns) const noexcept {
        return static_cast<uint64_t>(ns * cycles_per_ns_);
    }

private:
    CycleConverter() { calibrate(); }

    void calibrate() {
        constexpr int kRounds = 5;
        std::vector<double> samples;
        samples.reserve(kRounds);

        for (int r = 0; r < kRounds; ++r) {
            uint64_t c0 = rdtsc();
            uint64_t t0 = now_ns();
            // Sleep ~10ms
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            uint64_t c1 = rdtsc();
            uint64_t t1 = now_ns();
            double elapsed_ns = static_cast<double>(t1 - t0);
            double elapsed_cycles = static_cast<double>(c1 - c0);
            if (elapsed_ns > 0 && elapsed_cycles > 0) {
                samples.push_back(elapsed_cycles / elapsed_ns);
            }
        }
        if (!samples.empty()) {
            // Use median
            std::sort(samples.begin(), samples.end());
            cycles_per_ns_ = samples[samples.size() / 2];
        }
    }

    double cycles_per_ns_ = 3.0;  // default: assume ~3 GHz
};

// ---------------------------------------------------------------------------
// LatencyReport — pretty-print a histogram with cycle→ns conversion
// ---------------------------------------------------------------------------
struct LatencyReport {
    std::string label;
    uint64_t    count       = 0;
    double      mean_ns     = 0.0;
    uint64_t    min_ns      = 0;
    uint64_t    max_ns      = 0;
    uint64_t    p50_ns      = 0;
    uint64_t    p95_ns      = 0;
    uint64_t    p99_ns      = 0;
    uint64_t    p999_ns     = 0;

    static LatencyReport from_histogram(const LatencyHistogram& h,
                                         const std::string& lbl) {
        auto& cc = CycleConverter::instance();
        LatencyReport r;
        r.label  = lbl;
        r.count  = h.count();
        r.mean_ns= h.mean_cycles() * cc.ns_per_cycle();
        r.min_ns = cc.cycles_to_ns(
            h.min_cycles() == std::numeric_limits<uint64_t>::max()
                ? 0 : h.min_cycles());
        r.max_ns = cc.cycles_to_ns(h.max_cycles());
        r.p50_ns = cc.cycles_to_ns(h.p50());
        r.p95_ns = cc.cycles_to_ns(h.p95());
        r.p99_ns = cc.cycles_to_ns(h.p99());
        r.p999_ns= cc.cycles_to_ns(h.p999());
        return r;
    }

    void print() const {
        std::printf("[%s] n=%lu mean=%.1fns p50=%luns p95=%luns p99=%luns "
                    "p99.9=%luns min=%luns max=%luns\n",
                    label.c_str(), count, mean_ns,
                    p50_ns, p95_ns, p99_ns, p999_ns, min_ns, max_ns);
    }
};

// ---------------------------------------------------------------------------
// RollingPercentileWindow — maintains last N timestamps for percentile calc
// ---------------------------------------------------------------------------
class RollingPercentileWindow {
public:
    explicit RollingPercentileWindow(std::size_t window = 60000)
        : window_(window, 0), head_(0), size_(0) {}

    void record(uint64_t value_ns) noexcept {
        std::size_t idx = head_.fetch_add(1, std::memory_order_relaxed) % window_.size();
        window_[idx] = value_ns;
        if (size_.load(std::memory_order_relaxed) < window_.size())
            size_.fetch_add(1, std::memory_order_relaxed);
    }

    // Returns sorted snapshot for percentile calculation
    std::vector<uint64_t> sorted_snapshot() const {
        std::size_t n = std::min(size_.load(), window_.size());
        std::vector<uint64_t> v(window_.begin(), window_.begin() + n);
        std::sort(v.begin(), v.end());
        return v;
    }

    uint64_t percentile(double p) const {
        auto v = sorted_snapshot();
        if (v.empty()) return 0;
        std::size_t idx = static_cast<std::size_t>(p / 100.0 * v.size());
        idx = std::min(idx, v.size() - 1);
        return v[idx];
    }

    uint64_t p50()  const { return percentile(50.0); }
    uint64_t p95()  const { return percentile(95.0); }
    uint64_t p99()  const { return percentile(99.0); }
    uint64_t p999() const { return percentile(99.9); }

private:
    std::vector<uint64_t> window_;
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> size_{0};
};

// ---------------------------------------------------------------------------
// MultiConsumerStats — aggregate stats across multiple consumer cursors
// ---------------------------------------------------------------------------
struct MultiConsumerStats {
    std::size_t num_consumers       = 0;
    uint64_t    total_consumed      = 0;
    uint64_t    total_lag           = 0;   // slots behind producer
    uint64_t    max_lag             = 0;

    static MultiConsumerStats compute(
        const std::vector<uint64_t>& cursors,
        uint64_t write_pos)
    {
        MultiConsumerStats s;
        s.num_consumers = cursors.size();
        for (uint64_t c : cursors) {
            uint64_t lag = (write_pos > c) ? (write_pos - c) : 0;
            s.total_lag += lag;
            s.max_lag = std::max(s.max_lag, lag);
        }
        return s;
    }

    void print() const {
        std::printf("Consumers: %zu  MaxLag: %lu  AvgLag: %.1f\n",
                    num_consumers,
                    max_lag,
                    num_consumers > 0
                        ? static_cast<double>(total_lag) / num_consumers
                        : 0.0);
    }
};

// ---------------------------------------------------------------------------
// BenchmarkRingBuffer — self-contained benchmark for throughput/latency
// ---------------------------------------------------------------------------
struct BenchmarkResult {
    double   throughput_mpps    = 0.0;  // million pub/s
    double   mean_latency_ns    = 0.0;
    uint64_t p99_latency_ns     = 0;
    uint64_t total_ops          = 0;
    double   elapsed_s          = 0.0;
};

BenchmarkResult benchmark_ring_buffer(std::size_t capacity,
                                       std::size_t iterations,
                                       int n_consumers) {
    struct Item {
        uint64_t value = 0;
        uint64_t t0    = 0;
    };

    RingBuffer<Item> ring(capacity);
    std::atomic<bool> running{true};
    std::atomic<uint64_t> consumed_total{0};

    // Launch consumer threads
    std::vector<std::thread> consumers;
    for (int i = 0; i < n_consumers; ++i) {
        consumers.emplace_back([&, i]() {
            uint64_t cursor = ring.new_consumer_cursor();
            while (running.load(std::memory_order_relaxed)) {
                auto opt = ring.try_consume(cursor);
                if (opt) {
                    consumed_total.fetch_add(1, std::memory_order_relaxed);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    // Producer
    uint64_t wall_start = now_ns();
    for (std::size_t i = 0; i < iterations; ++i) {
        Item item;
        item.value = i;
        item.t0    = rdtsc();
        ring.publish_blocking(item);
    }
    uint64_t wall_end = now_ns();

    running.store(false, std::memory_order_release);
    for (auto& t : consumers) t.join();

    BenchmarkResult r;
    r.total_ops       = iterations;
    r.elapsed_s       = static_cast<double>(wall_end - wall_start) / 1e9;
    r.throughput_mpps = (r.elapsed_s > 0)
                        ? static_cast<double>(iterations) / r.elapsed_s / 1e6
                        : 0.0;
    auto rep = LatencyReport::from_histogram(ring.histogram(), "bench");
    r.mean_latency_ns = rep.mean_ns;
    r.p99_latency_ns  = rep.p99_ns;
    return r;
}

} // namespace aeternus::rtel
