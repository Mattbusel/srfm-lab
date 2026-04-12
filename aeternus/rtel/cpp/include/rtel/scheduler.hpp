// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// scheduler.hpp — Real-Time Pipeline Scheduler
// =============================================================================
// Event-driven pipeline scheduler with deadline-aware scheduling.
//
// Pipeline flow:
//   MarketTick → Chronos → NeuroSDE || TensorNet → OmniGraph → Lumina → HyperAgent → OrderSubmit
//
// Threading model:
//   - Main thread runs the event loop (priority SCHED_FIFO if root).
//   - Thread pool (N workers, N = logical CPU count - 2) for parallel stages.
//   - CPU affinity: scheduler thread pinned to core 0; workers to cores 1..N.
//   - Watchdog thread monitors for stalled modules (configurable timeout).
//
// Latency targets:
//   - Market tick → HyperAgent action: < 1ms end-to-end (soft real-time).
//   - Chronos LOB update: < 50µs.
//   - Lumina inference: < 500µs.
//   - HyperAgent action: < 100µs.
//
// Priority queue: min-heap by deadline (earliest deadline first).
// =============================================================================

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "global_state_registry.hpp"
#include "module_wrapper.hpp"
#include "ring_buffer.hpp"
#include "shm_bus.hpp"

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// PipelineTask — a unit of work in the scheduler
// ---------------------------------------------------------------------------
struct PipelineTask {
    uint64_t     pipeline_id  = 0;
    uint64_t     deadline_ns  = 0;   // absolute nanoseconds (CLOCK_MONOTONIC)
    uint64_t     enqueue_ns   = 0;
    ModuleID     module_id    = ModuleID::COUNT;
    int          priority     = 0;   // lower = higher priority
    MarketEvent  trigger;            // triggering market event (if any)
    std::function<bool()> task;     // callable

    bool operator>(const PipelineTask& o) const noexcept {
        return deadline_ns > o.deadline_ns;
    }
};

// ---------------------------------------------------------------------------
// SchedulerConfig
// ---------------------------------------------------------------------------
struct SchedulerConfig {
    int        n_workers           = 4;
    bool       pin_cpus            = false;       // requires elevated privileges
    bool       real_time_priority  = false;       // SCHED_FIFO
    uint64_t   pipeline_timeout_ns = 2'000'000;  // 2ms default
    uint64_t   watchdog_interval_ns= 5'000'000;  // 5ms
    bool       enable_watchdog     = true;
    std::size_t market_queue_depth = 4096;
    std::size_t task_queue_depth   = 8192;
};

// ---------------------------------------------------------------------------
// PipelineMetrics — per-run latency breakdown
// ---------------------------------------------------------------------------
struct PipelineMetrics {
    uint64_t pipeline_id         = 0;
    uint64_t market_recv_ns      = 0;
    uint64_t chronos_start_ns    = 0;
    uint64_t chronos_end_ns      = 0;
    uint64_t neuro_sde_start_ns  = 0;
    uint64_t neuro_sde_end_ns    = 0;
    uint64_t tensornet_start_ns  = 0;
    uint64_t tensornet_end_ns    = 0;
    uint64_t omni_graph_start_ns = 0;
    uint64_t omni_graph_end_ns   = 0;
    uint64_t lumina_start_ns     = 0;
    uint64_t lumina_end_ns       = 0;
    uint64_t hyper_agent_start_ns= 0;
    uint64_t hyper_agent_end_ns  = 0;
    uint64_t order_submit_ns     = 0;
    uint64_t total_latency_ns    = 0;
    bool     sla_met             = false;
    uint32_t modules_ok          = 0;
    uint32_t modules_error       = 0;
    uint32_t modules_timeout     = 0;

    void compute_total() noexcept {
        if (market_recv_ns > 0 && order_submit_ns > 0) {
            total_latency_ns = order_submit_ns - market_recv_ns;
        }
    }
};

// ---------------------------------------------------------------------------
// ThreadPool — fixed-size worker thread pool with work-stealing
// ---------------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(int n_threads, bool pin_cpus = false);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Submit a task; returns a future for the result
    template<typename F>
    auto submit(F&& f) -> std::future<decltype(f())>;

    // Submit a fire-and-forget task (no future returned)
    void submit_detached(std::function<void()> f);

    int  n_threads() const noexcept { return static_cast<int>(workers_.size()); }
    bool is_running() const noexcept { return running_.load(); }

    // Drain all pending work
    void drain();

    // Stats
    uint64_t tasks_executed() const noexcept {
        return stat_executed_.load(std::memory_order_relaxed);
    }
    uint64_t tasks_queued() const noexcept {
        return stat_queued_.load(std::memory_order_relaxed);
    }

private:
    void worker_loop(int thread_id, bool pin, int cpu_id);

    std::vector<std::thread>    workers_;
    std::mutex                  mutex_;
    std::condition_variable     cv_;
    std::queue<std::function<void()>> tasks_;
    std::atomic<bool>           running_{true};
    std::atomic<uint64_t>       stat_executed_{0};
    std::atomic<uint64_t>       stat_queued_{0};
};

// ---------------------------------------------------------------------------
// Watchdog — detects stalled modules and triggers alerts
// ---------------------------------------------------------------------------
class Watchdog {
public:
    struct Alert {
        ModuleID     module_id;
        uint64_t     start_ns;
        uint64_t     elapsed_ns;
        std::string  message;
    };

    using AlertCallback = std::function<void(const Alert&)>;

    explicit Watchdog(uint64_t check_interval_ns = 5'000'000);
    ~Watchdog();

    void set_alert_callback(AlertCallback cb) { callback_ = std::move(cb); }

    // Register a module heartbeat
    void module_start(ModuleID id) noexcept;
    void module_end  (ModuleID id) noexcept;
    void set_timeout (ModuleID id, uint64_t timeout_ns) noexcept;

    bool is_running() const noexcept { return running_.load(); }
    void start();
    void stop();

    std::vector<Alert> pending_alerts();

private:
    void watchdog_loop();

    struct ModuleState {
        std::atomic<uint64_t> start_ns{0};
        std::atomic<bool>     running{false};
        uint64_t              timeout_ns = 5'000'000;
    };

    std::array<ModuleState, static_cast<std::size_t>(ModuleID::COUNT)> states_;
    uint64_t check_interval_ns_;
    std::atomic<bool> running_{false};
    std::thread watchdog_thread_;
    AlertCallback callback_;

    std::mutex alerts_mutex_;
    std::vector<Alert> pending_;
};

// ---------------------------------------------------------------------------
// Scheduler — main real-time pipeline scheduler
// ---------------------------------------------------------------------------
class Scheduler {
public:
    explicit Scheduler(SchedulerConfig cfg = {});
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Initialize: set up bus, registry, modules
    bool initialize();

    // Start the scheduler event loop (blocking)
    void run();

    // Start scheduler in background thread
    void start_async();

    // Stop gracefully
    void stop();

    bool is_running() const noexcept { return running_.load(); }

    // Inject a market event (called by market data feed or test harness)
    void inject_market_event(const MarketEvent& ev);

    // Set callback for completed pipeline runs
    using PipelineCallback = std::function<void(const PipelineMetrics&)>;
    void set_pipeline_callback(PipelineCallback cb) {
        pipeline_cb_ = std::move(cb);
    }

    // Set order submission callback
    using OrderCallback = std::function<void(const HyperAgentWrapper::Action&)>;
    void set_order_callback(OrderCallback cb) {
        order_cb_ = std::move(cb);
    }

    // Get stats
    struct Stats {
        uint64_t pipelines_run        = 0;
        uint64_t sla_violations       = 0;
        uint64_t market_events        = 0;
        uint64_t total_latency_sum_ns = 0;
        double   mean_latency_us      = 0.0;
        double   p99_latency_us       = 0.0;
    };
    Stats get_stats() const noexcept;

    // Print stats to stdout
    void print_stats() const;

private:
    void event_loop();
    bool run_pipeline(const MarketEvent& ev, PipelineMetrics& metrics);
    void submit_order(const HyperAgentWrapper::Action& act, PipelineMetrics& metrics);

    SchedulerConfig cfg_;
    std::unique_ptr<ThreadPool> pool_;
    std::unique_ptr<Watchdog>   watchdog_;

    std::atomic<bool> running_{false};
    std::thread       event_thread_;

    // Market event queue (SPSC: feed → scheduler)
    MarketEventQueue market_queue_;

    // Pipeline ID counter
    std::atomic<uint64_t> pipeline_id_{0};

    // SLA: target end-to-end latency in nanoseconds
    static constexpr uint64_t kSlaTargetNs = 1'000'000;  // 1ms

    PipelineCallback pipeline_cb_;
    OrderCallback    order_cb_;

    // Stats
    std::atomic<uint64_t> stat_pipelines_{0};
    std::atomic<uint64_t> stat_sla_violations_{0};
    std::atomic<uint64_t> stat_market_events_{0};
    std::atomic<uint64_t> stat_total_latency_{0};

    LatencyHistogram latency_hist_;
};

} // namespace aeternus::rtel
