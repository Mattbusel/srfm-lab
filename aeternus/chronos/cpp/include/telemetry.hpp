#pragma once
// telemetry.hpp — rdtsc cycle counter, lock-free stats, CSV/binary metrics export.
// Chronos / AETERNUS — production C++ telemetry.

#include <cstdint>
#include <cstring>
#include <atomic>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <functional>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <memory>

namespace chronos {
namespace telemetry {

// ── rdtsc cycle counter ───────────────────────────────────────────────────────

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
inline uint64_t rdtsc() { return __rdtsc(); }
inline uint64_t rdtscp() { unsigned int aux; return __rdtscp(&aux); }
#else
// Fallback: use high-resolution clock
inline uint64_t rdtsc() {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}
inline uint64_t rdtscp() { return rdtsc(); }
#endif

inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );
}

// ── Calibrate TSC frequency ───────────────────────────────────────────────────

double calibrate_tsc_freq_ghz();

inline double cycles_to_ns(uint64_t cycles, double tsc_ghz) {
    return static_cast<double>(cycles) / tsc_ghz;
}

// ── Lock-free stats accumulator ──────────────────────────────────────────────

struct AtomicStats {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> sum{0};
    std::atomic<uint64_t> max_val{0};
    std::atomic<uint64_t> min_val{UINT64_MAX};

    void record(uint64_t value) noexcept {
        count.fetch_add(1, std::memory_order_relaxed);
        sum.fetch_add(value, std::memory_order_relaxed);
        uint64_t cur_max = max_val.load(std::memory_order_relaxed);
        while (value > cur_max &&
               !max_val.compare_exchange_weak(cur_max, value,
                   std::memory_order_relaxed, std::memory_order_relaxed));
        uint64_t cur_min = min_val.load(std::memory_order_relaxed);
        while (value < cur_min &&
               !min_val.compare_exchange_weak(cur_min, value,
                   std::memory_order_relaxed, std::memory_order_relaxed));
    }

    double mean() const noexcept {
        uint64_t c = count.load(std::memory_order_relaxed);
        if (c == 0) return 0.0;
        return static_cast<double>(sum.load(std::memory_order_relaxed)) / c;
    }

    uint64_t get_max() const noexcept { return max_val.load(std::memory_order_relaxed); }
    uint64_t get_min() const noexcept {
        uint64_t v = min_val.load(std::memory_order_relaxed);
        return (v == UINT64_MAX) ? 0 : v;
    }
    uint64_t get_count() const noexcept { return count.load(std::memory_order_relaxed); }

    void reset() noexcept {
        count.store(0, std::memory_order_relaxed);
        sum.store(0, std::memory_order_relaxed);
        max_val.store(0, std::memory_order_relaxed);
        min_val.store(UINT64_MAX, std::memory_order_relaxed);
    }
};

// ── HDR Latency Histogram (lock-free) ────────────────────────────────────────

class LatencyHistogram {
public:
    static constexpr size_t BUCKET_COUNT = 1 << 20; // 1M buckets
    static constexpr uint32_t RESOLUTION_BITS = 8;  // 256ns resolution

    LatencyHistogram() : buckets_(BUCKET_COUNT) {}

    void record(uint64_t latency_ns) noexcept {
        size_t idx = (latency_ns >> RESOLUTION_BITS);
        if (idx >= BUCKET_COUNT) idx = BUCKET_COUNT - 1;
        buckets_[idx].fetch_add(1, std::memory_order_relaxed);
        stats_.record(latency_ns);
    }

    uint64_t percentile(double pct) const noexcept {
        uint64_t total = stats_.get_count();
        if (total == 0) return 0;
        uint64_t target = static_cast<uint64_t>(pct / 100.0 * total) + 1;
        uint64_t cumulative = 0;
        for (size_t i = 0; i < BUCKET_COUNT; ++i) {
            cumulative += buckets_[i].load(std::memory_order_relaxed);
            if (cumulative >= target) {
                uint64_t midpoint = (static_cast<uint64_t>(i) << RESOLUTION_BITS)
                    + (1ULL << (RESOLUTION_BITS - 1));
                return midpoint;
            }
        }
        return stats_.get_max();
    }

    uint64_t p50() const noexcept { return percentile(50.0); }
    uint64_t p95() const noexcept { return percentile(95.0); }
    uint64_t p99() const noexcept { return percentile(99.0); }
    uint64_t p999() const noexcept { return percentile(99.9); }
    uint64_t p9999() const noexcept { return percentile(99.99); }
    uint64_t min_ns() const noexcept { return stats_.get_min(); }
    uint64_t max_ns() const noexcept { return stats_.get_max(); }
    double   mean_ns() const noexcept { return stats_.mean(); }
    uint64_t count() const noexcept { return stats_.get_count(); }

    void reset() noexcept {
        for (auto& b : buckets_) b.store(0, std::memory_order_relaxed);
        stats_.reset();
    }

    std::string summary() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0)
            << "n=" << count()
            << " min=" << min_ns() << "ns"
            << " mean=" << mean_ns() << "ns"
            << " p50=" << p50() << "ns"
            << " p95=" << p95() << "ns"
            << " p99=" << p99() << "ns"
            << " p999=" << p999() << "ns"
            << " max=" << max_ns() << "ns";
        return oss.str();
    }

private:
    std::vector<std::atomic<uint64_t>> buckets_;
    AtomicStats stats_;
};

// ── Lock-free ring buffer ────────────────────────────────────────────────────

template<typename T, size_t CAPACITY>
class RingBuffer {
    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "CAPACITY must be power of two");
public:
    static constexpr size_t MASK = CAPACITY - 1;

    bool push(const T& item) noexcept {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (tail + 1) & MASK;
        if (next_tail == head_.load(std::memory_order_acquire)) return false;
        buffer_[tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(T& item) noexcept {
        size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) return false;
        item = buffer_[head];
        head_.store((head + 1) & MASK, std::memory_order_release);
        return true;
    }

    bool empty() const noexcept { return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed); }
    size_t approx_size() const noexcept {
        size_t h = head_.load(std::memory_order_relaxed);
        size_t t = tail_.load(std::memory_order_relaxed);
        return (t >= h) ? t - h : CAPACITY - h + t;
    }
    size_t capacity() const noexcept { return CAPACITY; }

private:
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    T buffer_[CAPACITY];
};

// ── Binary metrics record ────────────────────────────────────────────────────

#pragma pack(push, 1)
struct MetricRecord {
    uint64_t timestamp_ns;
    uint32_t metric_id;
    uint16_t flags;
    uint16_t padding;
    double   value;
    double   secondary;
};
#pragma pack(pop)

static_assert(sizeof(MetricRecord) == 32, "MetricRecord must be 32 bytes");

// ── Metrics log (in-memory / mmap-compatible) ────────────────────────────────

class MetricsLog {
public:
    explicit MetricsLog(size_t capacity)
        : buffer_(capacity), capacity_(capacity), write_pos_(0), total_(0) {}

    bool write(const MetricRecord& rec) noexcept {
        size_t pos = write_pos_.fetch_add(1, std::memory_order_relaxed) % capacity_;
        buffer_[pos] = rec;
        total_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    uint64_t total_written() const noexcept { return total_.load(std::memory_order_relaxed); }
    size_t capacity() const noexcept { return capacity_; }

    std::vector<MetricRecord> dump() const {
        size_t n = std::min(write_pos_.load(std::memory_order_relaxed), capacity_);
        return std::vector<MetricRecord>(buffer_.begin(), buffer_.begin() + n);
    }

    std::string to_csv() const {
        auto records = dump();
        std::ostringstream oss;
        oss << "timestamp_ns,metric_id,flags,value,secondary\n";
        for (const auto& r : records) {
            oss << r.timestamp_ns << ","
                << r.metric_id << ","
                << r.flags << ","
                << std::fixed << std::setprecision(6) << r.value << ","
                << r.secondary << "\n";
        }
        return oss.str();
    }

    // Binary serialization
    std::vector<uint8_t> to_binary() const {
        auto records = dump();
        std::vector<uint8_t> out(records.size() * sizeof(MetricRecord));
        std::memcpy(out.data(), records.data(), out.size());
        return out;
    }

private:
    std::vector<MetricRecord> buffer_;
    size_t capacity_;
    std::atomic<size_t> write_pos_;
    std::atomic<uint64_t> total_;
};

// ── Throughput counter ────────────────────────────────────────────────────────

class ThroughputCounter {
public:
    explicit ThroughputCounter(double window_secs = 1.0)
        : window_ns_(static_cast<uint64_t>(window_secs * 1e9)) {}

    void record(uint64_t count = 1) noexcept {
        total_.fetch_add(count, std::memory_order_relaxed);
        window_count_.fetch_add(count, std::memory_order_relaxed);

        uint64_t now = now_ns();
        uint64_t ws = window_start_.load(std::memory_order_relaxed);
        if (now - ws >= window_ns_) {
            uint64_t wc = window_count_.exchange(0, std::memory_order_relaxed);
            double rate = static_cast<double>(wc) / (window_ns_ / 1e9);
            uint64_t rate_bits;
            std::memcpy(&rate_bits, &rate, sizeof(rate));
            last_rate_.store(rate_bits, std::memory_order_relaxed);
            window_start_.store(now, std::memory_order_relaxed);
        }
    }

    double rate_per_second() const noexcept {
        uint64_t bits = last_rate_.load(std::memory_order_relaxed);
        double rate;
        std::memcpy(&rate, &bits, sizeof(rate));
        return rate;
    }

    uint64_t total() const noexcept { return total_.load(std::memory_order_relaxed); }

private:
    uint64_t window_ns_;
    std::atomic<uint64_t> total_{0};
    std::atomic<uint64_t> window_count_{0};
    std::atomic<uint64_t> window_start_{0};
    std::atomic<uint64_t> last_rate_{0};
};

// ── Prometheus-compatible exporter ───────────────────────────────────────────

class PrometheusExporter {
public:
    struct CounterEntry { std::string name; std::string help; std::atomic<uint64_t>* value; };
    struct GaugeEntry { std::string name; std::string help; std::atomic<uint64_t>* value; }; // bits
    struct HistogramEntry { std::string name; std::shared_ptr<LatencyHistogram> hist; };

    void register_counter(std::string name, std::string help, std::atomic<uint64_t>* val) {
        counters_.push_back({std::move(name), std::move(help), val});
    }

    void register_histogram(std::string name, std::shared_ptr<LatencyHistogram> hist) {
        histograms_.push_back({std::move(name), std::move(hist)});
    }

    std::string exposition(const std::string& prefix = "") const {
        std::ostringstream oss;
        uint64_t ts = now_ns() / 1'000'000; // ms timestamp
        for (const auto& c : counters_) {
            oss << "# HELP " << prefix << c.name << " " << c.help << "\n";
            oss << "# TYPE " << prefix << c.name << " counter\n";
            oss << prefix << c.name << " " << c.value->load(std::memory_order_relaxed) << " " << ts << "\n";
        }
        for (const auto& h : histograms_) {
            auto& hist = *h.hist;
            oss << "# HELP " << prefix << h.name << " Latency histogram\n";
            oss << "# TYPE " << prefix << h.name << " summary\n";
            oss << prefix << h.name << "_count " << hist.count() << " " << ts << "\n";
            oss << prefix << h.name << "_sum " << static_cast<uint64_t>(hist.mean_ns() * hist.count()) << " " << ts << "\n";
            for (auto [q, val] : std::initializer_list<std::pair<double, uint64_t>>{
                {0.5, hist.p50()}, {0.95, hist.p95()}, {0.99, hist.p99()}, {0.999, hist.p999()}
            }) {
                oss << prefix << h.name << "{quantile=\"" << q << "\"} " << val << " " << ts << "\n";
            }
        }
        return oss.str();
    }

private:
    std::vector<CounterEntry> counters_;
    std::vector<HistogramEntry> histograms_;
};

// ── Telemetry hub ─────────────────────────────────────────────────────────────

class TelemetryHub {
public:
    TelemetryHub() :
        metrics_log_(65536),
        order_latency_(std::make_shared<LatencyHistogram>()),
        fill_latency_(std::make_shared<LatencyHistogram>()) {
        exporter_.register_counter("orders_total", "Total orders submitted", &orders_total_);
        exporter_.register_counter("fills_total", "Total fills received", &fills_total_);
        exporter_.register_counter("errors_total", "Total errors", &errors_total_);
        exporter_.register_histogram("order_latency_ns", order_latency_);
        exporter_.register_histogram("fill_latency_ns", fill_latency_);
        start_ns_ = now_ns();
    }

    void record_order(uint64_t latency_ns) noexcept {
        orders_total_.fetch_add(1, std::memory_order_relaxed);
        order_latency_->record(latency_ns);
        msg_counter_.record(1);
        metrics_log_.write({now_ns(), 1, 0, 0, static_cast<double>(latency_ns), 0.0});
    }

    void record_fill(uint64_t latency_ns, double pnl) noexcept {
        fills_total_.fetch_add(1, std::memory_order_relaxed);
        fill_latency_->record(latency_ns);
        metrics_log_.write({now_ns(), 2, 1, 0, static_cast<double>(latency_ns), pnl});
    }

    void record_error() noexcept { errors_total_.fetch_add(1, std::memory_order_relaxed); }

    std::string prometheus_exposition(const std::string& prefix = "chronos_") const {
        return exporter_.exposition(prefix);
    }

    const LatencyHistogram& order_latency() const { return *order_latency_; }
    const LatencyHistogram& fill_latency() const { return *fill_latency_; }
    uint64_t orders_total() const { return orders_total_.load(std::memory_order_relaxed); }
    uint64_t fills_total() const { return fills_total_.load(std::memory_order_relaxed); }
    double msg_rate_per_sec() const { return msg_counter_.rate_per_second(); }
    std::string metrics_csv() const { return metrics_log_.to_csv(); }
    uint64_t uptime_ns() const { return now_ns() - start_ns_; }

private:
    std::atomic<uint64_t> orders_total_{0};
    std::atomic<uint64_t> fills_total_{0};
    std::atomic<uint64_t> errors_total_{0};
    ThroughputCounter msg_counter_;
    std::shared_ptr<LatencyHistogram> order_latency_;
    std::shared_ptr<LatencyHistogram> fill_latency_;
    MetricsLog metrics_log_;
    PrometheusExporter exporter_;
    uint64_t start_ns_;
};

} // namespace telemetry
} // namespace chronos
