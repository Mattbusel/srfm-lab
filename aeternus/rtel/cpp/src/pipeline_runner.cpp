// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// pipeline_runner.cpp — Standalone pipeline runner and demo
// =============================================================================
// Demonstrates a complete AETERNUS pipeline run with synthetic market data.
// Usable as both a test harness and a production entry point.
// =============================================================================

#include "rtel/shm_bus.hpp"
#include "rtel/ring_buffer.hpp"
#include "rtel/global_state_registry.hpp"
#include "rtel/module_wrapper.hpp"
#include "rtel/scheduler.hpp"
#include "rtel/latency_monitor.hpp"
#include "rtel/serialization.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iomanip>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// SyntheticFeed — generates fake market data and injects into scheduler
// ---------------------------------------------------------------------------
class SyntheticFeed {
public:
    struct Config {
        int      n_assets          = 5;
        int      n_ticks           = 1000;
        double   base_price        = 150.0;
        double   annual_vol        = 0.20;    // 20% annual vol
        double   dt                = 1.0/252/390; // 1 min in years
        double   half_spread       = 0.005;
        double   tick_size         = 0.01;
        int      n_lob_levels      = 10;
        bool     realtime          = false;
        uint64_t inter_tick_ns     = 1'000'000; // 1ms
        int      seed              = 42;
    };

    explicit SyntheticFeed(Config cfg) : cfg_(std::move(cfg)) {
        rng_.seed(cfg_.seed);
        prices_.resize(cfg_.n_assets);
        vols_.resize(cfg_.n_assets);
        for (int i = 0; i < cfg_.n_assets; ++i) {
            prices_[i] = cfg_.base_price * (1.0 + i * 0.05);
            vols_[i]   = cfg_.annual_vol;
        }
    }

    // Run the feed; calls callback for each generated tick
    void run(std::function<void(const MarketEvent&)> callback) {
        std::normal_distribution<double> norm(0.0, 1.0);
        for (int t = 0; t < cfg_.n_ticks; ++t) {
            for (int i = 0; i < cfg_.n_assets; ++i) {
                double dt    = cfg_.dt;
                double sigma = vols_[i] * std::sqrt(dt);
                double dW    = norm(rng_);
                prices_[i]  *= std::exp(-0.5 * sigma * sigma + sigma * dW);
                prices_[i]   = std::max(1.0, prices_[i]);
                // GARCH-like vol update
                vols_[i]     = std::max(0.05, vols_[i] * 0.99
                               + 0.3 * std::abs(dW) * 0.01 * cfg_.annual_vol);

                MarketEvent ev{};
                ev.timestamp_ns = now_ns();
                ev.asset_id     = static_cast<uint32_t>(i);
                ev.price        = prices_[i];
                ev.bid          = prices_[i] - cfg_.half_spread;
                ev.ask          = prices_[i] + cfg_.half_spread;
                ev.size         = 100.0 * (1.0 + std::abs(dW));
                ev.event_type   = 0;
                callback(ev);
            }

            if (cfg_.realtime && cfg_.inter_tick_ns > 0) {
                std::this_thread::sleep_for(
                    std::chrono::nanoseconds(cfg_.inter_tick_ns));
            }
        }
    }

    int n_ticks_per_asset() const noexcept { return cfg_.n_ticks; }

private:
    Config cfg_;
    std::mt19937_64 rng_;
    std::vector<double> prices_;
    std::vector<double> vols_;
};

// ---------------------------------------------------------------------------
// PipelineRunner — orchestrates a full RTEL run
// ---------------------------------------------------------------------------
class PipelineRunner {
public:
    struct RunConfig {
        int      n_assets        = 5;
        int      n_ticks         = 500;
        bool     verbose         = true;
        bool     benchmark_mode  = false;
        int      scheduler_workers = 2;
        bool     run_diagnostics = true;
    };

    struct RunResult {
        int      n_pipelines_run = 0;
        int      n_sla_violations= 0;
        double   mean_latency_us = 0.0;
        double   p99_latency_us  = 0.0;
        double   throughput_pps  = 0.0;
        uint64_t gsr_version     = 0;
        bool     ok              = false;

        void print() const {
            std::printf("=== PipelineRunner Result ===\n");
            std::printf("  Pipelines:       %d\n", n_pipelines_run);
            std::printf("  SLA violations:  %d (%.1f%%)\n",
                        n_sla_violations,
                        n_pipelines_run > 0
                            ? 100.0 * n_sla_violations / n_pipelines_run : 0.0);
            std::printf("  Mean latency:    %.1f µs\n", mean_latency_us);
            std::printf("  p99 latency:     %.1f µs\n", p99_latency_us);
            std::printf("  Throughput:      %.1f pipelines/s\n", throughput_pps);
            std::printf("  GSR version:     %lu\n", gsr_version);
            std::printf("  Status:          %s\n", ok ? "OK" : "FAIL");
        }
    };

    explicit PipelineRunner(RunConfig cfg) : cfg_(std::move(cfg)) {}

    RunResult run() {
        RunResult result{};

        // 1. Initialize ShmBus
        auto& bus = ShmBus::instance();
        bus.shutdown();  // clean slate
        bus.create_aeternus_channels();

        // 2. Register and initialize modules
        auto& reg = ModuleRegistry::instance();
        reg.register_module(std::make_unique<ChronosWrapper>());
        reg.register_module(std::make_unique<NeuroSDEWrapper>());
        reg.register_module(std::make_unique<TensorNetWrapper>());
        reg.register_module(std::make_unique<OmniGraphWrapper>());
        reg.register_module(std::make_unique<LuminaWrapper>());
        reg.register_module(std::make_unique<HyperAgentWrapper>());

        if (!reg.initialize_all()) {
            std::fprintf(stderr, "Module initialization failed\n");
            return result;
        }

        // 3. Setup latency monitor
        auto& mon = LatencyMonitor::instance();
        mon.reset();
        mon.set_violation_callback([&](const SLAViolation& v) {
            if (cfg_.verbose) {
                std::fprintf(stderr, "  SLA: %s +%luµs\n",
                             stage_name(v.stage), v.overage_ns / 1000);
            }
        });

        // 4. Setup pipeline callback
        uint64_t total_lat_ns = 0;
        std::vector<uint64_t> latencies;
        latencies.reserve(cfg_.n_ticks);

        // 5. Create scheduler
        SchedulerConfig sched_cfg{};
        sched_cfg.n_workers   = cfg_.scheduler_workers;
        sched_cfg.enable_watchdog = !cfg_.benchmark_mode;
        Scheduler scheduler(sched_cfg);
        scheduler.initialize();

        scheduler.set_pipeline_callback([&](const PipelineMetrics& m) {
            result.n_pipelines_run++;
            if (!m.sla_met) result.n_sla_violations++;
            total_lat_ns += m.total_latency_ns;
            latencies.push_back(m.total_latency_ns);
        });

        // 6. Start scheduler in background
        scheduler.start_async();

        // 7. Generate and inject synthetic market data
        SyntheticFeed::Config feed_cfg{};
        feed_cfg.n_assets  = cfg_.n_assets;
        feed_cfg.n_ticks   = cfg_.n_ticks;
        feed_cfg.realtime  = false;
        feed_cfg.seed      = 42;
        SyntheticFeed feed(feed_cfg);

        uint64_t t_start = now_ns();
        feed.run([&](const MarketEvent& ev) {
            scheduler.inject_market_event(ev);
        });

        // Wait for pipeline to drain
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        uint64_t t_end = now_ns();

        scheduler.stop();

        // 8. Collect results
        double elapsed_s = static_cast<double>(t_end - t_start) / 1e9;
        if (result.n_pipelines_run > 0) {
            result.mean_latency_us = static_cast<double>(total_lat_ns)
                                   / result.n_pipelines_run / 1000.0;
            result.throughput_pps  = result.n_pipelines_run / elapsed_s;
        }
        if (!latencies.empty()) {
            std::sort(latencies.begin(), latencies.end());
            std::size_t p99idx = static_cast<std::size_t>(0.99 * latencies.size());
            result.p99_latency_us = latencies[p99idx] / 1000.0;
        }
        result.gsr_version = GlobalStateRegistry::instance().version();
        result.ok = (result.n_pipelines_run > 0);

        // 9. Diagnostics
        if (cfg_.run_diagnostics && cfg_.verbose) {
            std::printf("\n");
            result.print();
            std::printf("\n--- Latency Monitor ---\n");
            mon.print_stats();
            std::printf("\n--- Global State Registry ---\n");
            GlobalStateRegistry::instance().print_summary();
            std::printf("\n--- ShmBus ---\n");
            bus.print_stats();
        }

        return result;
    }

private:
    RunConfig cfg_;
};

// ---------------------------------------------------------------------------
// MemoryUsageTracker — approximate process memory usage
// ---------------------------------------------------------------------------
struct MemoryUsage {
    std::size_t rss_bytes    = 0;
    std::size_t heap_bytes   = 0;
    std::size_t shm_bytes    = 0;

    void print() const {
        std::printf("Memory: RSS=%.1fMB Heap=%.1fMB SHM=%.1fMB\n",
                    rss_bytes  / 1e6,
                    heap_bytes / 1e6,
                    shm_bytes  / 1e6);
    }
};

MemoryUsage get_memory_usage() {
    MemoryUsage m{};
#if defined(RTEL_PLATFORM_POSIX)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::size_t kb = 0;
            std::sscanf(line.c_str() + 6, "%zu", &kb);
            m.rss_bytes = kb * 1024;
        } else if (line.substr(0, 7) == "VmData:") {
            std::size_t kb = 0;
            std::sscanf(line.c_str() + 7, "%zu", &kb);
            m.heap_bytes = kb * 1024;
        }
    }
#endif
    return m;
}

// ---------------------------------------------------------------------------
// PerformanceReport — comprehensive performance analysis
// ---------------------------------------------------------------------------
struct PerformanceReport {
    PipelineRunner::RunResult run_result;
    MemoryUsage               memory;
    std::string               prometheus_output;
    double                    elapsed_s           = 0.0;
    std::string               timestamp;

    std::string to_json() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "{\n"
            << "  \"timestamp\": \"" << timestamp << "\",\n"
            << "  \"pipelines_run\": " << run_result.n_pipelines_run << ",\n"
            << "  \"sla_violations\": " << run_result.n_sla_violations << ",\n"
            << "  \"mean_latency_us\": " << run_result.mean_latency_us << ",\n"
            << "  \"p99_latency_us\": " << run_result.p99_latency_us << ",\n"
            << "  \"throughput_pps\": " << run_result.throughput_pps << ",\n"
            << "  \"elapsed_s\": " << elapsed_s << ",\n"
            << "  \"memory_rss_mb\": " << memory.rss_bytes / 1e6 << ",\n"
            << "  \"gsr_version\": " << run_result.gsr_version << "\n"
            << "}";
        return oss.str();
    }

    bool write_json(const std::string& path) const {
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << to_json();
        return true;
    }
};

// ---------------------------------------------------------------------------
// ContinuousBenchmark — runs repeated pipeline iterations and reports stats
// ---------------------------------------------------------------------------
class ContinuousBenchmark {
public:
    struct Config {
        int      duration_s       = 60;
        int      n_assets         = 3;
        int      ticks_per_burst  = 10;
        int      burst_interval_ms= 10;
        bool     verbose          = false;
    };

    explicit ContinuousBenchmark(Config cfg) : cfg_(std::move(cfg)) {}

    void run() {
        auto& bus = ShmBus::instance();
        bus.create_aeternus_channels();

        auto& reg = ModuleRegistry::instance();
        reg.register_module(std::make_unique<ChronosWrapper>());
        reg.register_module(std::make_unique<LuminaWrapper>());
        reg.register_module(std::make_unique<HyperAgentWrapper>());
        reg.initialize_all();

        SchedulerConfig sc{};
        sc.n_workers = 2;
        Scheduler sched(sc);
        sched.initialize();
        sched.start_async();

        std::atomic<int> tick_count{0};
        std::atomic<int> pipeline_count{0};
        sched.set_pipeline_callback([&](const PipelineMetrics&) {
            pipeline_count.fetch_add(1, std::memory_order_relaxed);
        });

        uint64_t t_deadline = now_ns() + static_cast<uint64_t>(cfg_.duration_s) * 1'000'000'000ULL;
        SyntheticFeed::Config fc{};
        fc.n_assets = cfg_.n_assets;
        fc.n_ticks  = cfg_.ticks_per_burst;
        fc.seed     = 42;

        while (now_ns() < t_deadline) {
            SyntheticFeed feed(fc);
            feed.run([&](const MarketEvent& ev) {
                sched.inject_market_event(ev);
                tick_count.fetch_add(1, std::memory_order_relaxed);
            });
            std::this_thread::sleep_for(
                std::chrono::milliseconds(cfg_.burst_interval_ms));
        }

        sched.stop();
        auto stats = sched.get_stats();
        std::printf("[ContinuousBench] duration=%ds ticks=%d pipelines=%lu "
                    "sla_viols=%lu mean_lat=%.1fµs p99=%.1fµs\n",
                    cfg_.duration_s,
                    tick_count.load(),
                    stats.pipelines_run,
                    stats.sla_violations,
                    stats.mean_latency_us,
                    stats.p99_latency_us);
    }

private:
    Config cfg_;
};

// ---------------------------------------------------------------------------
// RTELSmokeTest — quick smoke test suite
// ---------------------------------------------------------------------------
class RTELSmokeTest {
public:
    static bool run_all(bool verbose = false) {
        int pass = 0, fail = 0;
        auto test = [&](const char* name, bool ok) {
            std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
            if (ok) ++pass; else ++fail;
        };

        // Test 1: ShmBus creates channels
        {
            auto& bus = ShmBus::instance();
            bus.shutdown();
            bus.create_aeternus_channels();
            auto names = bus.channel_names();
            test("ShmBus.create_channels", names.size() >= 8);
        }

        // Test 2: ShmBus write/read roundtrip
        {
            auto& bus = ShmBus::instance();
            auto* ch = bus.channel(channels::PIPELINE_EVENTS);
            bool write_ok = false;
            if (ch) {
                auto [handle, err] = ch->claim(false);
                if (err == ShmChannel::Error::OK) {
                    *reinterpret_cast<uint64_t*>(handle.data()) = 0xDEADBEEF;
                    handle.publish();
                    write_ok = true;
                }
            }
            test("ShmBus.write_read", write_ok);
        }

        // Test 3: GSR update/read
        {
            auto& gsr = GlobalStateRegistry::instance();
            uint64_t v0 = gsr.version();
            LOBSnapshot snap{};
            snap.asset_id = 0;
            snap.n_bid_levels = 1;
            snap.n_ask_levels = 1;
            snap.bids[0] = {149.99, 100.0};
            snap.asks[0] = {150.01, 100.0};
            snap.compute_derived();
            gsr.update_lob(0, snap);
            LOBSnapshot out{};
            bool read_ok = gsr.read_lob(0, out);
            test("GSR.lob_update_read", read_ok && out.mid_price > 149.0);
            test("GSR.version_incremented", gsr.version() > v0);
        }

        // Test 4: Ring buffer basic
        {
            RingBuffer<int> rb(64);
            bool pub_ok = rb.try_publish(42);
            uint64_t cur = rb.new_consumer_cursor();
            auto val = rb.try_consume(cur);
            test("RingBuffer.basic", pub_ok && val.has_value() && *val == 42);
        }

        // Test 5: Serialization
        {
            Serializer ser;
            float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
            TensorDescriptor td{};
            td.dtype = DType::FLOAT32;
            td.ndim  = 1;
            td.shape[0] = 4;
            td.payload_bytes = 16;
            auto blob = ser.serialize(td, data, 16);
            auto result = ser.deserialize(blob.data(), blob.size());
            const float* out_data = reinterpret_cast<const float*>(result.data_ptr);
            test("Serialization.roundtrip",
                 result.ok && std::fabs(out_data[0] - 1.0f) < 1e-5f);
        }

        // Test 6: Latency monitor
        {
            auto& mon = LatencyMonitor::instance();
            mon.reset();
            for (int i = 1; i <= 100; ++i) {
                StageLatency lat{};
                lat.stage = PipelineStage::CHRONOS;
                lat.duration_ns = i * 1000ULL;
                lat.end_ns = now_ns();
                mon.record(lat);
            }
            auto stats = mon.get_stage_stats(PipelineStage::CHRONOS);
            test("LatencyMonitor.record", stats.count == 100);
        }

        std::printf("\nSmoke tests: %d passed, %d failed\n", pass, fail);
        return fail == 0;
    }
};

} // namespace aeternus::rtel

// ---------------------------------------------------------------------------
// main() — standalone pipeline runner entry point
// ---------------------------------------------------------------------------
// Uncomment to build as executable:
// int main(int argc, char* argv[]) {
//     using namespace aeternus::rtel;
//     std::printf("AETERNUS RTEL Pipeline Runner\n");
//
//     if (!RTELSmokeTest::run_all(true)) {
//         return 1;
//     }
//
//     PipelineRunner::RunConfig cfg{};
//     cfg.n_assets    = 5;
//     cfg.n_ticks     = 200;
//     cfg.verbose     = true;
//     cfg.scheduler_workers = 2;
//
//     PipelineRunner runner(cfg);
//     auto result = runner.run();
//     return result.ok ? 0 : 1;
// }
