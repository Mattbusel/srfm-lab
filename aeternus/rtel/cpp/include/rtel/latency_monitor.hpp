// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// latency_monitor.hpp — End-to-End Pipeline Latency Monitor
// =============================================================================
// Monitors per-stage latency across the full AETERNUS pipeline.
// Features:
//   - Per-stage rdtsc-based timing (nanosecond resolution)
//   - Latency budget tracking: each stage has a budget in nanoseconds
//   - SLA violation alerts: callback fired when budget exceeded
//   - Rolling 1-minute percentile window (p50/p95/p99)
//   - Prometheus metrics export in text format (for scraping by prometheus)
// =============================================================================

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ring_buffer.hpp"
#include "shm_bus.hpp"

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// Stage IDs for the AETERNUS pipeline
// ---------------------------------------------------------------------------
enum class PipelineStage : uint8_t {
    MARKET_RECV    = 0,
    CHRONOS        = 1,
    NEURO_SDE      = 2,
    TENSORNET      = 3,
    OMNI_GRAPH     = 4,
    LUMINA         = 5,
    HYPER_AGENT    = 6,
    ORDER_SUBMIT   = 7,
    TOTAL_PIPELINE = 8,
    COUNT          = 9,
};

inline const char* stage_name(PipelineStage s) noexcept {
    switch (s) {
        case PipelineStage::MARKET_RECV:    return "market_recv";
        case PipelineStage::CHRONOS:        return "chronos";
        case PipelineStage::NEURO_SDE:      return "neuro_sde";
        case PipelineStage::TENSORNET:      return "tensornet";
        case PipelineStage::OMNI_GRAPH:     return "omni_graph";
        case PipelineStage::LUMINA:         return "lumina";
        case PipelineStage::HYPER_AGENT:    return "hyper_agent";
        case PipelineStage::ORDER_SUBMIT:   return "order_submit";
        case PipelineStage::TOTAL_PIPELINE: return "total_pipeline";
        default:                            return "unknown";
    }
}

// Default latency budgets in nanoseconds
inline constexpr uint64_t kStageBudgets[] = {
    /* MARKET_RECV   */ 10'000,    // 10µs
    /* CHRONOS       */ 50'000,    // 50µs
    /* NEURO_SDE     */ 200'000,   // 200µs
    /* TENSORNET     */ 300'000,   // 300µs
    /* OMNI_GRAPH    */ 200'000,   // 200µs
    /* LUMINA        */ 500'000,   // 500µs
    /* HYPER_AGENT   */ 100'000,   // 100µs
    /* ORDER_SUBMIT  */ 20'000,    // 20µs
    /* TOTAL_PIPELINE*/ 1'000'000, // 1ms
};

// ---------------------------------------------------------------------------
// StageLatency — single stage timing record
// ---------------------------------------------------------------------------
struct StageLatency {
    PipelineStage stage        = PipelineStage::COUNT;
    uint64_t      start_ns     = 0;
    uint64_t      end_ns       = 0;
    uint64_t      duration_ns  = 0;
    uint64_t      budget_ns    = 0;
    bool          over_budget  = false;
    uint64_t      pipeline_id  = 0;
};

// ---------------------------------------------------------------------------
// SLAViolation — fired when a stage exceeds its budget
// ---------------------------------------------------------------------------
struct SLAViolation {
    PipelineStage stage        = PipelineStage::COUNT;
    uint64_t      duration_ns  = 0;
    uint64_t      budget_ns    = 0;
    uint64_t      overage_ns   = 0;
    uint64_t      timestamp_ns = 0;
    uint64_t      pipeline_id  = 0;
    std::string   message;
};

// ---------------------------------------------------------------------------
// StageTimer — RAII scope timer for a pipeline stage
// ---------------------------------------------------------------------------
class LatencyMonitor;

class StageTimer {
public:
    StageTimer(LatencyMonitor& monitor, PipelineStage stage, uint64_t pipeline_id)
        : monitor_(monitor), stage_(stage), pipeline_id_(pipeline_id),
          start_ns_(now_ns()) {}

    ~StageTimer();

    // Early stop (if needed before destructor)
    void stop() noexcept;

private:
    LatencyMonitor& monitor_;
    PipelineStage   stage_;
    uint64_t        pipeline_id_;
    uint64_t        start_ns_;
    bool            stopped_ = false;
};

// ---------------------------------------------------------------------------
// RollingLatencyWindow — 1-minute sliding window of latency values
// ---------------------------------------------------------------------------
class RollingLatencyWindow {
public:
    static constexpr std::size_t kWindowSize = 60000;  // ~1 min at 1kHz

    explicit RollingLatencyWindow() : data_(kWindowSize, 0), head_(0), size_(0) {}

    void record(uint64_t latency_ns) noexcept {
        std::size_t idx = head_.fetch_add(1, std::memory_order_relaxed) % kWindowSize;
        data_[idx] = latency_ns;
        std::size_t sz = size_.load(std::memory_order_relaxed);
        if (sz < kWindowSize) size_.fetch_add(1, std::memory_order_relaxed);
    }

    // Returns sorted snapshot (expensive — use only for reporting)
    std::vector<uint64_t> sorted_window() const {
        std::size_t n = std::min(size_.load(), kWindowSize);
        std::vector<uint64_t> v(data_.begin(), data_.begin() + n);
        std::sort(v.begin(), v.end());
        return v;
    }

    uint64_t percentile(double p) const {
        auto v = sorted_window();
        if (v.empty()) return 0;
        std::size_t idx = static_cast<std::size_t>(p * 0.01 * v.size());
        return v[std::min(idx, v.size() - 1)];
    }

    uint64_t p50()  const { return percentile(50.0); }
    uint64_t p95()  const { return percentile(95.0); }
    uint64_t p99()  const { return percentile(99.0); }
    uint64_t p999() const { return percentile(99.9); }
    uint64_t count()const { return size_.load(std::memory_order_relaxed); }

private:
    std::vector<uint64_t> data_;
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> size_{0};
};

// ---------------------------------------------------------------------------
// LatencyMonitor — main latency monitoring class
// ---------------------------------------------------------------------------
class LatencyMonitor {
public:
    using ViolationCallback = std::function<void(const SLAViolation&)>;

    static LatencyMonitor& instance();

    LatencyMonitor(const LatencyMonitor&) = delete;
    LatencyMonitor& operator=(const LatencyMonitor&) = delete;

    // Set custom budgets (nanoseconds per stage)
    void set_budget(PipelineStage stage, uint64_t ns) noexcept;

    // Set SLA violation callback
    void set_violation_callback(ViolationCallback cb) {
        violation_cb_ = std::move(cb);
    }

    // Record a stage timing
    void record(const StageLatency& lat);

    // Create RAII stage timer
    StageTimer time_stage(PipelineStage stage, uint64_t pipeline_id) {
        return StageTimer(*this, stage, pipeline_id);
    }

    // Get stats for a specific stage
    struct StageStats {
        uint64_t count               = 0;
        double   mean_ns             = 0.0;
        uint64_t p50_ns              = 0;
        uint64_t p95_ns              = 0;
        uint64_t p99_ns              = 0;
        uint64_t budget_ns           = 0;
        uint64_t violations          = 0;
        double   violation_rate_pct  = 0.0;
    };
    StageStats get_stage_stats(PipelineStage stage) const;

    // Export Prometheus metrics (text format)
    std::string prometheus_export() const;

    // Print all stage stats to stdout
    void print_stats() const;

    // Reset all stats
    void reset();

private:
    LatencyMonitor();

    std::array<uint64_t, static_cast<std::size_t>(PipelineStage::COUNT)> budgets_{};
    std::array<RollingLatencyWindow, static_cast<std::size_t>(PipelineStage::COUNT)> windows_;
    std::array<LatencyHistogram, static_cast<std::size_t>(PipelineStage::COUNT)> histograms_;
    std::array<std::atomic<uint64_t>,
               static_cast<std::size_t>(PipelineStage::COUNT)> violation_counts_{};

    ViolationCallback violation_cb_;
    mutable std::mutex mutex_;
};

// StageTimer destructor — defined after LatencyMonitor is complete
inline StageTimer::~StageTimer() {
    stop();
}

inline void StageTimer::stop() noexcept {
    if (!stopped_) {
        stopped_ = true;
        uint64_t end_ns = now_ns();
        StageLatency lat{};
        lat.stage       = stage_;
        lat.start_ns    = start_ns_;
        lat.end_ns      = end_ns;
        lat.duration_ns = end_ns - start_ns_;
        lat.pipeline_id = pipeline_id_;
        monitor_.record(lat);
    }
}

// Convenience macros
#define RTEL_TIME_STAGE(monitor, stage, pid) \
    auto _stage_timer_##stage = (monitor).time_stage((stage), (pid))

} // namespace aeternus::rtel
