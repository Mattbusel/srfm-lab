// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// scheduler.cpp — Real-Time Pipeline Scheduler Implementation
// =============================================================================

#include "rtel/scheduler.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <iostream>

#if defined(RTEL_PLATFORM_POSIX)
#  include <pthread.h>
#  include <sched.h>
#  include <sys/resource.h>
#endif

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// ThreadPool
// ---------------------------------------------------------------------------
ThreadPool::ThreadPool(int n_threads, bool pin_cpus) {
    for (int i = 0; i < n_threads; ++i) {
        int cpu_id = pin_cpus ? (i + 1) : -1;
        workers_.emplace_back([this, i, pin_cpus, cpu_id]() {
            worker_loop(i, pin_cpus, cpu_id);
        });
    }
}

ThreadPool::~ThreadPool() {
    running_.store(false, std::memory_order_release);
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void ThreadPool::worker_loop(int thread_id, bool pin, int cpu_id) {
    (void)thread_id;
#if defined(RTEL_PLATFORM_POSIX)
    if (pin && cpu_id >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }
#endif
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return !tasks_.empty() || !running_.load();
            });
            if (!running_.load() && tasks_.empty()) break;
            if (tasks_.empty()) continue;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        if (task) {
            task();
            stat_executed_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void ThreadPool::submit_detached(std::function<void()> f) {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        tasks_.push(std::move(f));
        stat_queued_.fetch_add(1, std::memory_order_relaxed);
    }
    cv_.notify_one();
}

template<typename F>
auto ThreadPool::submit(F&& f) -> std::future<decltype(f())> {
    using RetType = decltype(f());
    auto task_ptr = std::make_shared<std::packaged_task<RetType()>>(
        std::forward<F>(f));
    auto future = task_ptr->get_future();
    submit_detached([task_ptr]() { (*task_ptr)(); });
    return future;
}

void ThreadPool::drain() {
    while (true) {
        bool empty;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            empty = tasks_.empty();
        }
        if (empty) break;
        std::this_thread::yield();
    }
}

// Explicit instantiation for common types
template std::future<bool> ThreadPool::submit<std::function<bool()>>(
    std::function<bool()>&&);

// ---------------------------------------------------------------------------
// Watchdog
// ---------------------------------------------------------------------------
Watchdog::Watchdog(uint64_t check_interval_ns)
    : check_interval_ns_(check_interval_ns) {}

Watchdog::~Watchdog() { stop(); }

void Watchdog::module_start(ModuleID id) noexcept {
    auto& s = states_[static_cast<std::size_t>(id)];
    s.start_ns.store(now_ns(), std::memory_order_release);
    s.running.store(true, std::memory_order_release);
}

void Watchdog::module_end(ModuleID id) noexcept {
    auto& s = states_[static_cast<std::size_t>(id)];
    s.running.store(false, std::memory_order_release);
}

void Watchdog::set_timeout(ModuleID id, uint64_t timeout_ns) noexcept {
    states_[static_cast<std::size_t>(id)].timeout_ns = timeout_ns;
}

void Watchdog::start() {
    if (running_.exchange(true)) return;
    watchdog_thread_ = std::thread([this]() { watchdog_loop(); });
}

void Watchdog::stop() {
    if (!running_.exchange(false)) return;
    if (watchdog_thread_.joinable()) watchdog_thread_.join();
}

void Watchdog::watchdog_loop() {
    while (running_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(
            std::chrono::nanoseconds(check_interval_ns_));
        uint64_t now = now_ns();
        for (std::size_t i = 0; i < static_cast<std::size_t>(ModuleID::COUNT); ++i) {
            auto& s = states_[i];
            if (!s.running.load(std::memory_order_acquire)) continue;
            uint64_t start = s.start_ns.load(std::memory_order_relaxed);
            uint64_t elapsed = now - start;
            if (elapsed > s.timeout_ns) {
                Alert a;
                a.module_id  = static_cast<ModuleID>(i);
                a.start_ns   = start;
                a.elapsed_ns = elapsed;
                a.message    = std::string("Module ") +
                               module_name(static_cast<ModuleID>(i)) +
                               " stalled for " +
                               std::to_string(elapsed / 1000) + "us";
                {
                    std::lock_guard<std::mutex> lock(alerts_mutex_);
                    pending_.push_back(a);
                }
                if (callback_) callback_(a);
            }
        }
    }
}

std::vector<Watchdog::Alert> Watchdog::pending_alerts() {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    auto v = std::move(pending_);
    pending_.clear();
    return v;
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------
Scheduler::Scheduler(SchedulerConfig cfg)
    : cfg_(std::move(cfg)),
      market_queue_(cfg_.market_queue_depth) {}

Scheduler::~Scheduler() {
    stop();
}

bool Scheduler::initialize() {
    // Create thread pool
    pool_ = std::make_unique<ThreadPool>(cfg_.n_workers, cfg_.pin_cpus);

    // Create watchdog
    if (cfg_.enable_watchdog) {
        watchdog_ = std::make_unique<Watchdog>(cfg_.watchdog_interval_ns);
        watchdog_->set_alert_callback([](const Watchdog::Alert& a) {
            std::fprintf(stderr, "[WATCHDOG] %s\n", a.message.c_str());
        });
        // Set timeouts per module
        watchdog_->set_timeout(ModuleID::CHRONOS,    500'000);   // 500µs
        watchdog_->set_timeout(ModuleID::NEURO_SDE,  2'000'000); // 2ms
        watchdog_->set_timeout(ModuleID::TENSORNET,  3'000'000); // 3ms
        watchdog_->set_timeout(ModuleID::OMNI_GRAPH, 5'000'000); // 5ms
        watchdog_->set_timeout(ModuleID::LUMINA,     5'000'000); // 5ms
        watchdog_->set_timeout(ModuleID::HYPER_AGENT,1'000'000); // 1ms
        watchdog_->start();
    }

    // Initialize ShmBus channels
    ShmBus::instance().create_aeternus_channels();

    // Register all modules
    auto& reg = ModuleRegistry::instance();
    reg.register_module(std::make_unique<ChronosWrapper>());
    reg.register_module(std::make_unique<NeuroSDEWrapper>());
    reg.register_module(std::make_unique<TensorNetWrapper>());
    reg.register_module(std::make_unique<OmniGraphWrapper>());
    reg.register_module(std::make_unique<LuminaWrapper>());
    reg.register_module(std::make_unique<HyperAgentWrapper>());

    return reg.initialize_all();
}

bool Scheduler::run_pipeline(const MarketEvent& ev, PipelineMetrics& metrics) {
    metrics.pipeline_id   = pipeline_id_.fetch_add(1, std::memory_order_relaxed);
    metrics.market_recv_ns = ev.timestamp_ns ? ev.timestamp_ns : now_ns();

    // Inject market event into Chronos
    auto* chronos = dynamic_cast<ChronosWrapper*>(
        ModuleRegistry::instance().get(ModuleID::CHRONOS));
    if (chronos) chronos->inject_event(ev);

    // Read current world state
    WorldState state;
    GlobalStateRegistry::instance().read_world_state(state);

    auto& reg = ModuleRegistry::instance();
    auto order = reg.topological_order();

    for (ModuleID id : order) {
        ModuleBase* m = reg.get(id);
        if (!m || !m->enabled()) continue;

        uint64_t start_ns = now_ns();

        if (watchdog_) watchdog_->module_start(id);

        OutputBuffer out;
        bool ok = m->timed_forward(state, out);

        if (watchdog_) watchdog_->module_end(id);

        uint64_t end_ns = now_ns();

        // Update metrics
        switch (id) {
            case ModuleID::CHRONOS:
                metrics.chronos_start_ns = start_ns;
                metrics.chronos_end_ns   = end_ns;
                break;
            case ModuleID::NEURO_SDE:
                metrics.neuro_sde_start_ns = start_ns;
                metrics.neuro_sde_end_ns   = end_ns;
                break;
            case ModuleID::TENSORNET:
                metrics.tensornet_start_ns = start_ns;
                metrics.tensornet_end_ns   = end_ns;
                break;
            case ModuleID::OMNI_GRAPH:
                metrics.omni_graph_start_ns = start_ns;
                metrics.omni_graph_end_ns   = end_ns;
                break;
            case ModuleID::LUMINA:
                metrics.lumina_start_ns = start_ns;
                metrics.lumina_end_ns   = end_ns;
                break;
            case ModuleID::HYPER_AGENT:
                metrics.hyper_agent_start_ns = start_ns;
                metrics.hyper_agent_end_ns   = end_ns;
                break;
            default: break;
        }

        if (ok) {
            ++metrics.modules_ok;
            // Re-read state after each update
            GlobalStateRegistry::instance().read_world_state(state);
        } else {
            ++metrics.modules_error;
        }
    }

    metrics.order_submit_ns = now_ns();
    metrics.compute_total();
    metrics.sla_met = (metrics.total_latency_ns <= kSlaTargetNs);

    return metrics.modules_error == 0;
}

void Scheduler::submit_order(const HyperAgentWrapper::Action& act,
                               PipelineMetrics& metrics) {
    metrics.order_submit_ns = now_ns();
    if (order_cb_) order_cb_(act);
}

void Scheduler::event_loop() {
#if defined(RTEL_PLATFORM_POSIX) && defined(SCHED_FIFO)
    if (cfg_.real_time_priority) {
        struct sched_param sp{};
        sp.sched_priority = 50;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
            std::perror("Scheduler: set RT priority");
        }
    }
    if (cfg_.pin_cpus) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }
#endif

    uint64_t consumer_cursor = market_queue_.ring().new_consumer_cursor();

    while (running_.load(std::memory_order_acquire)) {
        auto ev_opt = market_queue_.ring().try_consume(consumer_cursor);
        if (!ev_opt.has_value()) {
            std::this_thread::yield();
            continue;
        }

        stat_market_events_.fetch_add(1, std::memory_order_relaxed);
        PipelineMetrics metrics{};
        bool ok = run_pipeline(*ev_opt, metrics);

        stat_pipelines_.fetch_add(1, std::memory_order_relaxed);
        stat_total_latency_.fetch_add(metrics.total_latency_ns,
                                       std::memory_order_relaxed);
        latency_hist_.record(metrics.total_latency_ns);

        if (!metrics.sla_met) {
            stat_sla_violations_.fetch_add(1, std::memory_order_relaxed);
        }

        if (pipeline_cb_) pipeline_cb_(metrics);
        (void)ok;
    }
}

void Scheduler::run() {
    running_.store(true, std::memory_order_release);
    event_loop();
}

void Scheduler::start_async() {
    running_.store(true, std::memory_order_release);
    event_thread_ = std::thread([this]() { event_loop(); });
}

void Scheduler::stop() {
    running_.store(false, std::memory_order_release);
    if (event_thread_.joinable()) event_thread_.join();
    if (watchdog_) watchdog_->stop();
    if (pool_) pool_->drain();
    ModuleRegistry::instance().shutdown_all();
}

void Scheduler::inject_market_event(const MarketEvent& ev) {
    market_queue_.publish(ev);
}

Scheduler::Stats Scheduler::get_stats() const noexcept {
    Stats s{};
    s.pipelines_run        = stat_pipelines_.load(std::memory_order_relaxed);
    s.sla_violations       = stat_sla_violations_.load(std::memory_order_relaxed);
    s.market_events        = stat_market_events_.load(std::memory_order_relaxed);
    s.total_latency_sum_ns = stat_total_latency_.load(std::memory_order_relaxed);
    if (s.pipelines_run > 0) {
        s.mean_latency_us = static_cast<double>(s.total_latency_sum_ns)
                          / s.pipelines_run / 1000.0;
    }
    s.p99_latency_us = static_cast<double>(latency_hist_.p99()) / 1000.0;
    return s;
}

void Scheduler::print_stats() const {
    auto s = get_stats();
    std::printf("=== Scheduler Stats ===\n");
    std::printf("  Pipelines:       %lu\n", s.pipelines_run);
    std::printf("  SLA violations:  %lu (%.1f%%)\n",
                s.sla_violations,
                s.pipelines_run > 0
                    ? 100.0 * s.sla_violations / s.pipelines_run : 0.0);
    std::printf("  Market events:   %lu\n", s.market_events);
    std::printf("  Mean latency:    %.1f µs\n", s.mean_latency_us);
    std::printf("  p99  latency:    %.1f µs\n", s.p99_latency_us);
}

} // namespace aeternus::rtel
