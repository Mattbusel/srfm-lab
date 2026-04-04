#pragma once
// Latency probing utilities: cycle counter, histogram, percentile,
// and end-to-end latency measurement between producer and consumer.

#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <atomic>
#include <cassert>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64)
#   include <x86intrin.h>
#   define HFT_RDTSC() __rdtsc()
#   define HFT_RDTSCP(aux) __rdtscp(&(aux))
#else
#   define HFT_RDTSC() static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())
#   define HFT_RDTSCP(aux) HFT_RDTSC()
#endif

namespace hft {

// ============================================================
// CPU frequency estimation (for TSC → ns conversion)
// ============================================================
inline double estimate_tsc_freq_ghz(int warmup_ms = 100) {
    auto t0 = std::chrono::steady_clock::now();
    uint64_t c0 = HFT_RDTSC();
    // Busy wait
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0).count() < warmup_ms) {}
    uint64_t c1 = HFT_RDTSC();
    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return (c1 - c0) / elapsed_ns; // cycles/ns = GHz
}

// ============================================================
// Latency histogram (log2-bucketed)
// Buckets: [0,1), [1,2), [2,4), [4,8), ... [2^k, 2^(k+1))
// ============================================================
class LatencyHistogram {
public:
    static constexpr int N_BUCKETS = 64;

    void record(uint64_t ns) {
        total_samples_++;
        total_sum_ns_ += ns;
        if (ns < min_ns_) min_ns_ = ns;
        if (ns > max_ns_) max_ns_ = ns;

        int bucket = 0;
        if (ns > 0) {
            bucket = 63 - __builtin_clzll(ns);
            if (bucket >= N_BUCKETS) bucket = N_BUCKETS - 1;
        }
        buckets_[bucket]++;
    }

    double percentile(double p) const {
        if (total_samples_ == 0) return 0;
        uint64_t target = static_cast<uint64_t>(p / 100.0 * total_samples_);
        uint64_t cumul  = 0;
        for (int b = 0; b < N_BUCKETS; ++b) {
            cumul += buckets_[b];
            if (cumul >= target) {
                // Interpolate within bucket [2^b, 2^(b+1))
                uint64_t lo = (b == 0) ? 0 : (1ULL << (b - 1));
                uint64_t hi = 1ULL << b;
                return (lo + hi) / 2.0;
            }
        }
        return static_cast<double>(max_ns_);
    }

    uint64_t p50()  const { return static_cast<uint64_t>(percentile(50)); }
    uint64_t p95()  const { return static_cast<uint64_t>(percentile(95)); }
    uint64_t p99()  const { return static_cast<uint64_t>(percentile(99)); }
    uint64_t p999() const { return static_cast<uint64_t>(percentile(99.9)); }
    double   mean() const {
        return total_samples_ > 0 ? static_cast<double>(total_sum_ns_) / total_samples_ : 0;
    }
    uint64_t min()  const { return min_ns_ == UINT64_MAX ? 0 : min_ns_; }
    uint64_t max()  const { return max_ns_; }
    uint64_t count() const { return total_samples_; }

    void reset() {
        std::fill(buckets_.begin(), buckets_.end(), 0);
        total_samples_ = 0;
        total_sum_ns_  = 0;
        min_ns_        = UINT64_MAX;
        max_ns_        = 0;
    }

    void print(const std::string& name) const {
        std::cout << std::left << std::setw(24) << name
                  << std::right
                  << "  count=" << std::setw(8)  << total_samples_
                  << "  min="   << std::setw(7)  << min()
                  << "  p50="   << std::setw(7)  << p50()
                  << "  p95="   << std::setw(7)  << p95()
                  << "  p99="   << std::setw(7)  << p99()
                  << "  p99.9=" << std::setw(8)  << p999()
                  << "  max="   << std::setw(8)  << max()
                  << "  mean="  << std::setw(7)  << std::fixed << std::setprecision(1) << mean()
                  << "  ns\n";
    }

    // Merge two histograms
    LatencyHistogram operator+(const LatencyHistogram& o) const {
        LatencyHistogram res;
        for (int b = 0; b < N_BUCKETS; ++b) res.buckets_[b] = buckets_[b] + o.buckets_[b];
        res.total_samples_ = total_samples_ + o.total_samples_;
        res.total_sum_ns_  = total_sum_ns_  + o.total_sum_ns_;
        res.min_ns_        = std::min(min_ns_,  o.min_ns_);
        res.max_ns_        = std::max(max_ns_,  o.max_ns_);
        return res;
    }

private:
    std::array<uint64_t, N_BUCKETS> buckets_{};
    uint64_t total_samples_ = 0;
    uint64_t total_sum_ns_  = 0;
    uint64_t min_ns_        = UINT64_MAX;
    uint64_t max_ns_        = 0;
};

// ============================================================
// Latency probe: timestamps at multiple checkpoints
// ============================================================
struct LatencyProbe {
    static constexpr int MAX_CHECKPOINTS = 8;
    struct Checkpoint {
        const char* name;
        uint64_t    tsc;
    };

    Checkpoint checkpoints[MAX_CHECKPOINTS];
    int n = 0;

    void mark(const char* name) {
        if (n >= MAX_CHECKPOINTS) return;
        checkpoints[n++] = { name, HFT_RDTSC() };
    }

    // Print all inter-checkpoint latencies (requires TSC freq calibration)
    void print(double tsc_ghz = 3.0) const {
        if (n < 2) return;
        for (int i = 1; i < n; ++i) {
            double ns = (checkpoints[i].tsc - checkpoints[i-1].tsc) / tsc_ghz;
            std::cout << "  " << checkpoints[i-1].name << " → " << checkpoints[i].name
                      << ": " << std::fixed << std::setprecision(1) << ns << " ns\n";
        }
        double total = (checkpoints[n-1].tsc - checkpoints[0].tsc) / tsc_ghz;
        std::cout << "  TOTAL: " << total << " ns\n";
    }

    double elapsed_ns(int from, int to, double tsc_ghz = 3.0) const {
        if (from >= n || to >= n || from >= to) return 0;
        return (checkpoints[to].tsc - checkpoints[from].tsc) / tsc_ghz;
    }
};

// ============================================================
// ScopedTimer: RAII latency measurement
// ============================================================
class ScopedTimer {
public:
    explicit ScopedTimer(LatencyHistogram& hist, bool use_tsc = false, double tsc_ghz = 3.0)
        : hist_(hist), use_tsc_(use_tsc), tsc_ghz_(tsc_ghz)
    {
        if (use_tsc_) tsc_start_ = HFT_RDTSC();
        else          wall_start_ = std::chrono::high_resolution_clock::now();
    }

    ~ScopedTimer() {
        uint64_t ns;
        if (use_tsc_) {
            uint64_t tsc_end = HFT_RDTSC();
            ns = static_cast<uint64_t>((tsc_end - tsc_start_) / tsc_ghz_);
        } else {
            auto wall_end = std::chrono::high_resolution_clock::now();
            ns = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start_).count();
        }
        hist_.record(ns);
    }

private:
    LatencyHistogram&  hist_;
    bool               use_tsc_;
    double             tsc_ghz_;
    uint64_t           tsc_start_  = 0;
    std::chrono::high_resolution_clock::time_point wall_start_;
};

// ============================================================
// Round-trip latency (producer → consumer → producer)
// ============================================================
struct RTTProbe {
    std::atomic<uint64_t> send_tsc{0};
    std::atomic<uint64_t> recv_tsc{0};
    std::atomic<uint64_t> ack_tsc{0};
    char _pad[40];  // cache line isolation

    void send() {
        send_tsc.store(HFT_RDTSC(), std::memory_order_relaxed);
    }

    void receive() {
        recv_tsc.store(HFT_RDTSC(), std::memory_order_relaxed);
    }

    void ack() {
        ack_tsc.store(HFT_RDTSC(), std::memory_order_relaxed);
    }

    double one_way_ns(double tsc_ghz = 3.0) const {
        uint64_t s = send_tsc.load(std::memory_order_relaxed);
        uint64_t r = recv_tsc.load(std::memory_order_relaxed);
        return r > s ? (r - s) / tsc_ghz : 0.0;
    }

    double rtt_ns(double tsc_ghz = 3.0) const {
        uint64_t s = send_tsc.load(std::memory_order_relaxed);
        uint64_t a = ack_tsc.load(std::memory_order_relaxed);
        return a > s ? (a - s) / tsc_ghz : 0.0;
    }
};

// ============================================================
// Benchmark runner: measure wall-clock latency distribution
// ============================================================
template<typename Func>
LatencyHistogram benchmark_func(const std::string& name, Func&& f,
                                  size_t n_warmup = 1000,
                                  size_t n_measure = 100000)
{
    // Warmup
    for (size_t i = 0; i < n_warmup; ++i) f();

    LatencyHistogram hist;
    for (size_t i = 0; i < n_measure; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        auto t1 = std::chrono::high_resolution_clock::now();
        hist.record(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
    }
    hist.print(name);
    return hist;
}

// ============================================================
// Spinning wait utilities (for low-latency synchronization)
// ============================================================
inline void spin_pause() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#else
    // ARM: yield hint
    asm volatile("yield" ::: "memory");
#endif
}

template<typename Pred>
bool spin_wait(Pred pred, uint64_t timeout_ns = 1'000'000'000ULL) {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::nanoseconds(timeout_ns);
    while (!pred()) {
        spin_pause();
        if (std::chrono::steady_clock::now() > deadline) return false;
    }
    return true;
}

// ============================================================
// Cache warmup: touch memory pages before benchmarking
// ============================================================
template<typename T>
void warm_cache(const T* data, size_t n) {
    volatile T dummy = T{};
    for (size_t i = 0; i < n; i += 64 / sizeof(T))
        dummy = data[i];
    (void)dummy;
}

// ============================================================
// Print a comparison table of multiple histograms
// ============================================================
struct ComparisonTable {
    std::vector<std::pair<std::string, LatencyHistogram>> entries;

    void add(const std::string& name, const LatencyHistogram& h) {
        entries.push_back({name, h});
    }

    void print() const {
        if (entries.empty()) return;
        std::cout << std::left
                  << std::setw(24) << "Name"
                  << std::right
                  << std::setw(10) << "Count"
                  << std::setw(9) << "p50(ns)"
                  << std::setw(9) << "p95(ns)"
                  << std::setw(9) << "p99(ns)"
                  << std::setw(10) << "p99.9(ns)"
                  << std::setw(10) << "mean(ns)"
                  << "\n";
        std::cout << std::string(81, '-') << "\n";
        for (const auto& [name, h] : entries) {
            std::cout << std::left << std::setw(24) << name
                      << std::right
                      << std::setw(10) << h.count()
                      << std::setw(9)  << h.p50()
                      << std::setw(9)  << h.p95()
                      << std::setw(9)  << h.p99()
                      << std::setw(10) << h.p999()
                      << std::fixed << std::setprecision(1)
                      << std::setw(10) << h.mean()
                      << "\n";
        }
    }
};

} // namespace hft
