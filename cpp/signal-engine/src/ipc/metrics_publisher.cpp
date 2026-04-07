// metrics_publisher.cpp -- Out-of-line implementations for MetricsPublisher.
//
// The bulk of the class lives in the header.  This TU provides:
//   - A high-resolution wall-clock helper (RDTSC-backed on x86)
//   - A global singleton accessor
//   - Histogram-based percentile tracker for latency
//
// Compile: -std=c++17 -O3 -mavx2 -mfma

#include "ipc/metrics_publisher.hpp"
#include <cstdlib>
#include <cstring>
#include <cassert>

#if defined(_MSC_VER) || defined(__MINGW32__)
#  include <intrin.h>
#  pragma intrinsic(__rdtsc)
#endif

namespace srfm {
namespace ipc {

// ------------------------------------------------------------------
// RDTSC-based cycle counter (x86/x86_64)
// ------------------------------------------------------------------

/// Read the CPU timestamp counter.  Used for sub-nanosecond latency
/// measurements where std::chrono overhead is non-negligible.
inline uint64_t rdtsc() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned hi, lo;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(_MSC_VER)
    return __rdtsc();
#else
    // Fallback: use steady_clock
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(
            steady_clock::now().time_since_epoch()).count());
#endif
}

// ------------------------------------------------------------------
// LatencyHistogram -- HDR-style power-of-2 bucketed histogram
// ------------------------------------------------------------------

/// Tracks latency distribution with O(1) record and O(n_buckets) percentile.
/// Buckets are exponentially spaced: [0,1ns), [1,2), [2,4), [4,8), ...
class LatencyHistogram {
public:
    static constexpr int N_BUCKETS = 64;

    LatencyHistogram() noexcept { reset(); }

    void record(int64_t latency_ns) noexcept {
        if (latency_ns < 0) latency_ns = 0;
        int bucket = 0;
        if (latency_ns > 0) {
            // floor(log2(latency_ns)) clamped to N_BUCKETS-1
#if defined(__GNUC__) || defined(__clang__)
            bucket = 63 - __builtin_clzll(static_cast<unsigned long long>(latency_ns));
#else
            uint64_t v = static_cast<uint64_t>(latency_ns);
            while (v >>= 1) ++bucket;
#endif
        }
        if (bucket >= N_BUCKETS) bucket = N_BUCKETS - 1;
        ++counts_[static_cast<std::size_t>(bucket)];
        ++total_;
        if (latency_ns > max_ns_) max_ns_ = latency_ns;
        if (latency_ns < min_ns_ || total_ == 1) min_ns_ = latency_ns;
    }

    /// Returns the approximate percentile value (e.g. pct=0.99 for P99).
    int64_t percentile(double pct) const noexcept {
        if (total_ == 0) return 0;
        int64_t target = static_cast<int64_t>(pct * total_);
        int64_t acc = 0;
        for (int i = 0; i < N_BUCKETS; ++i) {
            acc += counts_[static_cast<std::size_t>(i)];
            if (acc >= target) {
                // Upper bound of bucket i is 2^(i+1) - 1
                return (static_cast<int64_t>(1) << (i + 1)) - 1;
            }
        }
        return max_ns_;
    }

    int64_t p50() const noexcept { return percentile(0.50); }
    int64_t p95() const noexcept { return percentile(0.95); }
    int64_t p99() const noexcept { return percentile(0.99); }
    int64_t min() const noexcept { return min_ns_; }
    int64_t max() const noexcept { return max_ns_; }
    int64_t total() const noexcept { return total_; }

    void reset() noexcept {
        counts_.fill(0);
        total_ = min_ns_ = max_ns_ = 0;
    }

    std::string to_json() const {
        char buf[1024];
        snprintf(buf, sizeof(buf),
            "{\"p50\":%lld,\"p95\":%lld,\"p99\":%lld,\"min\":%lld,\"max\":%lld,\"n\":%lld}",
            static_cast<long long>(p50()),
            static_cast<long long>(p95()),
            static_cast<long long>(p99()),
            static_cast<long long>(min_ns_),
            static_cast<long long>(max_ns_),
            static_cast<long long>(total_));
        return std::string(buf);
    }

private:
    std::array<int64_t, N_BUCKETS> counts_;
    int64_t total_  = 0;
    int64_t min_ns_ = 0;
    int64_t max_ns_ = 0;
};

// ------------------------------------------------------------------
// ScopedLatencyTimer -- RAII helper for measuring bar processing time
// ------------------------------------------------------------------

/// Usage:
///   {
///       ScopedLatencyTimer t(publisher);
///       // ... process bar ...
///   }  // timer records latency on destruction
class ScopedLatencyTimer {
public:
    explicit ScopedLatencyTimer(MetricsPublisher& pub) noexcept
        : pub_(pub)
        , start_ns_(MetricsPublisher::clock_ns())
    {}

    ~ScopedLatencyTimer() noexcept {
        int64_t elapsed = MetricsPublisher::clock_ns() - start_ns_;
        pub_.record_bar(elapsed);
    }

    ScopedLatencyTimer(const ScopedLatencyTimer&) = delete;
    ScopedLatencyTimer& operator=(const ScopedLatencyTimer&) = delete;

private:
    MetricsPublisher& pub_;
    int64_t           start_ns_;
};

// ------------------------------------------------------------------
// Global singleton
// ------------------------------------------------------------------

/// Returns a process-wide MetricsPublisher (lazy initialised).
/// Thread-safe via static local initialisation (C++11 magic statics).
MetricsPublisher& global_publisher() {
    static MetricsPublisher pub("/tmp/srfm_signal_status.json", 10);
    return pub;
}

// ------------------------------------------------------------------
// MetricsPublisher::clock_ns() -- static helper
// ------------------------------------------------------------------

// Implementation of the static clock helper declared but not defined
// in the header.  Uses std::chrono::steady_clock for portability;
// override with RDTSC variant on production Linux if < 20ns overhead
// per call is required.
int64_t MetricsPublisher::clock_ns() noexcept {
    using namespace std::chrono;
    return static_cast<int64_t>(
        duration_cast<nanoseconds>(
            steady_clock::now().time_since_epoch()).count());
}

// ------------------------------------------------------------------
// MetricsPublisher::render_latency_section() -- extended JSON helper
// ------------------------------------------------------------------

/// Appends a JSON latency histogram section into `buf`, starting at `pos`.
/// Returns new position after appending.
/// Free function -- appends latency histogram section to a JSON buffer.
static int render_latency_json(const LatencyHistogram& hist,
                                char*                   buf,
                                int                     pos,
                                std::size_t             cap) noexcept
{
    auto w = [&](const char* s) {
        std::size_t l = std::strlen(s);
        if (static_cast<std::size_t>(pos) + l < cap) {
            std::memcpy(buf + pos, s, l);
            pos += static_cast<int>(l);
        }
    };
    char tmp[32];

    w(",\"latency\":{");
    snprintf(tmp, sizeof(tmp), "%lld", static_cast<long long>(hist.p50())); w("\"p50\":"); w(tmp); w(",");
    snprintf(tmp, sizeof(tmp), "%lld", static_cast<long long>(hist.p95())); w("\"p95\":"); w(tmp); w(",");
    snprintf(tmp, sizeof(tmp), "%lld", static_cast<long long>(hist.p99())); w("\"p99\":"); w(tmp); w(",");
    snprintf(tmp, sizeof(tmp), "%lld", static_cast<long long>(hist.min())); w("\"min\":"); w(tmp); w(",");
    snprintf(tmp, sizeof(tmp), "%lld", static_cast<long long>(hist.max())); w("\"max\":"); w(tmp);
    w("}");
    return pos;
}

} // namespace ipc
} // namespace srfm
