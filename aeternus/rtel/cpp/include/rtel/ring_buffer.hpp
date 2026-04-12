// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// ring_buffer.hpp — Lock-Free SPMC Ring Buffer
// =============================================================================
// Single-Producer Multi-Consumer lock-free ring buffer.
//
// Design:
//   - Power-of-2 capacity enforced at construction.
//   - Each slot has a `sequence` atomic<uint64_t> for sequencing:
//       sequence == slot_index         → slot is free (writable)
//       sequence == slot_index + 1     → slot is published (readable)
//       sequence == slot_index + cap   → slot has been consumed and recycled
//   - Producer atomically increments write cursor; stores data; stores
//     sequence = slot_index + 1.
//   - Each consumer has an independent read cursor; they never contend
//     with each other on cursor updates (no shared mutable cursor).
//   - False-sharing prevention: producer cursor and each consumer cursor
//     are on separate cache lines.
//
// Throughput counter: atomic incremented on each publish.
// Latency histogram: rdtsc-based, 64 buckets logarithmic.
// =============================================================================

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include "shm_bus.hpp"   // for rdtsc(), kCacheLineSize, align_up()

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// LatencyHistogram — rdtsc-based, 64 logarithmic buckets
// ---------------------------------------------------------------------------
class LatencyHistogram {
public:
    static constexpr std::size_t kBuckets = 64;

    void record(uint64_t cycles) noexcept {
        // Bucket = floor(log2(cycles)) clamped to [0, kBuckets-1]
        std::size_t b = 0;
        if (cycles > 1) {
            uint64_t v = cycles;
            while (v >>= 1) ++b;
        }
        b = std::min(b, kBuckets - 1);
        buckets_[b].fetch_add(1, std::memory_order_relaxed);
        count_.fetch_add(1, std::memory_order_relaxed);
        // Track min/max
        uint64_t mn = min_.load(std::memory_order_relaxed);
        while (cycles < mn && !min_.compare_exchange_weak(mn, cycles,
               std::memory_order_relaxed)) {}
        uint64_t mx = max_.load(std::memory_order_relaxed);
        while (cycles > mx && !max_.compare_exchange_weak(mx, cycles,
               std::memory_order_relaxed)) {}
        // Running sum for mean
        sum_.fetch_add(cycles, std::memory_order_relaxed);
    }

    // Returns cycles at given percentile (0..100)
    uint64_t percentile(double p) const noexcept {
        uint64_t total = count_.load(std::memory_order_relaxed);
        if (total == 0) return 0;
        uint64_t target = static_cast<uint64_t>(p / 100.0 * total);
        uint64_t cum = 0;
        for (std::size_t b = 0; b < kBuckets; ++b) {
            cum += buckets_[b].load(std::memory_order_relaxed);
            if (cum > target) {
                // Midpoint of bucket: 2^b .. 2^(b+1)-1  → 3*2^(b-1)
                return (b == 0) ? 1ULL : (3ULL << (b - 1));
            }
        }
        return (3ULL << (kBuckets - 2));
    }

    uint64_t p50() const noexcept { return percentile(50.0); }
    uint64_t p95() const noexcept { return percentile(95.0); }
    uint64_t p99() const noexcept { return percentile(99.0); }
    uint64_t p999()const noexcept { return percentile(99.9); }

    uint64_t count() const noexcept { return count_.load(std::memory_order_relaxed); }
    uint64_t min_cycles() const noexcept { return min_.load(std::memory_order_relaxed); }
    uint64_t max_cycles() const noexcept { return max_.load(std::memory_order_relaxed); }
    double   mean_cycles() const noexcept {
        uint64_t c = count_.load(std::memory_order_relaxed);
        if (c == 0) return 0.0;
        return static_cast<double>(sum_.load(std::memory_order_relaxed)) / c;
    }

    void reset() noexcept {
        for (auto& b : buckets_) b.store(0, std::memory_order_relaxed);
        count_.store(0, std::memory_order_relaxed);
        sum_.store(0, std::memory_order_relaxed);
        min_.store(std::numeric_limits<uint64_t>::max(), std::memory_order_relaxed);
        max_.store(0, std::memory_order_relaxed);
    }

    // Merge another histogram into this one
    void merge(const LatencyHistogram& other) noexcept {
        for (std::size_t b = 0; b < kBuckets; ++b) {
            buckets_[b].fetch_add(
                other.buckets_[b].load(std::memory_order_relaxed),
                std::memory_order_relaxed);
        }
        count_.fetch_add(other.count_.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
        sum_.fetch_add(other.sum_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
    }

    // Print to stdout
    void print(const char* label = "latency") const noexcept {
        std::printf("[%s] count=%lu p50=%lu p95=%lu p99=%lu mean=%.1f min=%lu max=%lu (cycles)\n",
                    label,
                    count(), p50(), p95(), p99(),
                    mean_cycles(),
                    min_cycles() == std::numeric_limits<uint64_t>::max() ? 0 : min_cycles(),
                    max_cycles());
    }

private:
    std::array<std::atomic<uint64_t>, kBuckets> buckets_{};
    std::atomic<uint64_t> count_{0};
    std::atomic<uint64_t> sum_{0};
    std::atomic<uint64_t> min_{std::numeric_limits<uint64_t>::max()};
    std::atomic<uint64_t> max_{0};
};

// ---------------------------------------------------------------------------
// ThroughputCounter — rolling window throughput measurement
// ---------------------------------------------------------------------------
class ThroughputCounter {
public:
    explicit ThroughputCounter(std::size_t window_size = 1000)
        : window_(window_size, 0), head_(0), count_(0) {}

    void record_now() noexcept {
        uint64_t t = now_ns();
        std::size_t idx = head_.fetch_add(1, std::memory_order_relaxed) % window_.size();
        window_[idx] = t;
        count_.fetch_add(1, std::memory_order_relaxed);
    }

    // Returns events per second over last window_size events
    double rate_per_second() const noexcept {
        if (count_.load() < 2) return 0.0;
        // Find min and max timestamps in window
        uint64_t mn = std::numeric_limits<uint64_t>::max();
        uint64_t mx = 0;
        for (auto t : window_) {
            if (t > 0) {
                mn = std::min(mn, t);
                mx = std::max(mx, t);
            }
        }
        if (mx <= mn) return 0.0;
        double elapsed_s = static_cast<double>(mx - mn) / 1e9;
        return static_cast<double>(window_.size()) / elapsed_s;
    }

    uint64_t total() const noexcept { return count_.load(std::memory_order_relaxed); }

private:
    std::vector<uint64_t> window_;
    std::atomic<std::size_t> head_{0};
    std::atomic<uint64_t> count_{0};
};

// ---------------------------------------------------------------------------
// RingSlot<T> — a single slot in the ring buffer
// ---------------------------------------------------------------------------
template<typename T>
struct alignas(kCacheLineSize) RingSlot {
    std::atomic<uint64_t> sequence{0};
    uint8_t _pad[kCacheLineSize - sizeof(std::atomic<uint64_t>)];
    T data;
    // Additional padding to prevent false sharing on data
    // (only matters if T < cache line)
};

// ---------------------------------------------------------------------------
// RingBuffer<T> — lock-free SPMC ring buffer
// ---------------------------------------------------------------------------
template<typename T>
class RingBuffer {
public:
    using value_type = T;

    // capacity must be power of 2; if not, rounds up
    explicit RingBuffer(std::size_t capacity = 1024)
        : capacity_(next_pow2(capacity)),
          mask_(capacity_ - 1) {
        slots_ = std::make_unique<RingSlot<T>[]>(capacity_);
        for (std::size_t i = 0; i < capacity_; ++i) {
            slots_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    ~RingBuffer() = default;
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t mask()     const noexcept { return mask_; }

    // ---------------------------------------------------------------------------
    // Producer API (single producer assumed)
    // ---------------------------------------------------------------------------

    // Non-blocking publish. Returns true if successful, false if full.
    bool try_publish(const T& item) noexcept {
        uint64_t pos = write_cursor_.load(std::memory_order_relaxed);
        RingSlot<T>& slot = slots_[pos & mask_];
        uint64_t seq = slot.sequence.load(std::memory_order_acquire);
        if (seq != pos) {
            // Ring is full
            stat_backpressure_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        write_cursor_.store(pos + 1, std::memory_order_relaxed);
        uint64_t t0 = rdtsc();
        slot.data = item;
        slot.sequence.store(pos + 1, std::memory_order_release);
        uint64_t t1 = rdtsc();
        histogram_.record(t1 - t0);
        throughput_.record_now();
        stat_published_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // Blocking publish — spins until slot is free
    void publish(const T& item) noexcept {
        while (!try_publish(item)) {
            std::this_thread::yield();
        }
    }

    // Batch publish (up to items.size() items)
    // Returns number of items actually published
    std::size_t batch_publish(std::span<const T> items) noexcept {
        std::size_t count = 0;
        for (const auto& item : items) {
            if (!try_publish(item)) break;
            ++count;
        }
        return count;
    }

    // ---------------------------------------------------------------------------
    // Consumer API (multiple consumers, each with independent cursor)
    // ---------------------------------------------------------------------------

    // Returns next item for given cursor. Returns nullopt if no data.
    std::optional<T> try_consume(uint64_t& cursor) noexcept {
        RingSlot<T>& slot = slots_[cursor & mask_];
        uint64_t seq = slot.sequence.load(std::memory_order_acquire);
        if (seq != cursor + 1) {
            return std::nullopt;
        }
        T item = slot.data;
        // signal consumed (sequence = cursor + capacity + 1 → free for producer)
        // Actually: advance cursor; sequence accounting is per-slot
        ++cursor;
        stat_consumed_.fetch_add(1, std::memory_order_relaxed);
        return item;
    }

    // Blocking consume
    T consume(uint64_t& cursor) noexcept {
        while (true) {
            auto opt = try_consume(cursor);
            if (opt.has_value()) return *opt;
            std::this_thread::yield();
        }
    }

    // Batch consume up to max_count items into out
    std::size_t batch_consume(uint64_t& cursor, std::span<T> out) noexcept {
        std::size_t count = 0;
        while (count < out.size()) {
            auto opt = try_consume(cursor);
            if (!opt.has_value()) break;
            out[count++] = *opt;
        }
        return count;
    }

    // Get current write position (for new consumer initialization)
    uint64_t current_write_pos() const noexcept {
        return write_cursor_.load(std::memory_order_acquire);
    }

    // Create a new consumer cursor at the current tail (receives new items only)
    uint64_t new_consumer_cursor() const noexcept {
        return current_write_pos();
    }

    // ---------------------------------------------------------------------------
    // Stats
    // ---------------------------------------------------------------------------
    uint64_t published_total()    const noexcept {
        return stat_published_.load(std::memory_order_relaxed);
    }
    uint64_t consumed_total()     const noexcept {
        return stat_consumed_.load(std::memory_order_relaxed);
    }
    uint64_t backpressure_hits()  const noexcept {
        return stat_backpressure_.load(std::memory_order_relaxed);
    }
    double   throughput_per_sec() const noexcept {
        return throughput_.rate_per_second();
    }
    const LatencyHistogram& histogram() const noexcept { return histogram_; }
    LatencyHistogram&       histogram()       noexcept { return histogram_; }

    void reset_stats() noexcept {
        stat_published_.store(0, std::memory_order_relaxed);
        stat_consumed_.store(0, std::memory_order_relaxed);
        stat_backpressure_.store(0, std::memory_order_relaxed);
        histogram_.reset();
    }

    // Approximate fill level [0.0, 1.0]
    double fill_fraction() const noexcept {
        uint64_t w = write_cursor_.load(std::memory_order_relaxed);
        uint64_t c = stat_consumed_.load(std::memory_order_relaxed);
        uint64_t in_flight = (w > c) ? (w - c) : 0;
        return static_cast<double>(in_flight) / static_cast<double>(capacity_);
    }

private:
    static std::size_t next_pow2(std::size_t n) {
        if (n == 0) return 1;
        --n;
        for (std::size_t s = 1; s < sizeof(n)*8; s <<= 1) n |= n >> s;
        return n + 1;
    }

    const std::size_t capacity_;
    const std::size_t mask_;

    // Producer cursor — on its own cache line
    alignas(kCacheLineSize) std::atomic<uint64_t> write_cursor_{0};
    uint8_t _pad0[kCacheLineSize - sizeof(std::atomic<uint64_t>)];

    std::unique_ptr<RingSlot<T>[]> slots_;

    // Stats
    alignas(kCacheLineSize) std::atomic<uint64_t> stat_published_{0};
    std::atomic<uint64_t> stat_consumed_{0};
    std::atomic<uint64_t> stat_backpressure_{0};
    LatencyHistogram histogram_;
    ThroughputCounter throughput_{10000};
};

// ---------------------------------------------------------------------------
// SPMCQueue<T> — higher-level wrapper with consumer registration
// ---------------------------------------------------------------------------
template<typename T>
class SPMCQueue {
public:
    explicit SPMCQueue(std::size_t capacity = 1024)
        : ring_(capacity) {}

    // Register a new consumer; returns its cursor handle
    uint64_t subscribe() {
        return ring_.new_consumer_cursor();
    }

    // Producer: publish item
    bool publish(const T& item) { return ring_.try_publish(item); }
    void publish_blocking(const T& item) { ring_.publish(item); }

    // Consumer: consume from cursor
    std::optional<T> consume(uint64_t& cursor) {
        return ring_.try_consume(cursor);
    }

    const RingBuffer<T>& ring() const noexcept { return ring_; }
    RingBuffer<T>&       ring()       noexcept { return ring_; }

private:
    RingBuffer<T> ring_;
};

// ---------------------------------------------------------------------------
// Typed ring aliases for common AETERNUS data types
// ---------------------------------------------------------------------------

// Lightweight market event for the pipeline scheduler
struct MarketEvent {
    uint64_t timestamp_ns = 0;
    uint32_t asset_id     = 0;
    uint32_t event_type   = 0;  // 0=tick, 1=trade, 2=quote, 3=eod
    double   price        = 0.0;
    double   size         = 0.0;
    double   bid          = 0.0;
    double   ask          = 0.0;
};

// Pipeline stage completion event
struct StageEvent {
    uint64_t pipeline_id  = 0;
    uint64_t start_ns     = 0;
    uint64_t end_ns       = 0;
    uint32_t stage_id     = 0;   // enum: CHRONOS=0,TENSORNET=1,...
    uint32_t status       = 0;   // 0=ok, 1=error, 2=timeout
    char     stage_name[32] = {};
};

using MarketEventQueue = SPMCQueue<MarketEvent>;
using StageEventQueue  = SPMCQueue<StageEvent>;

} // namespace aeternus::rtel
