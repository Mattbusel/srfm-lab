// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// rtel_extra.cpp — Additional utilities, benchmarking harness, diagnostics
// =============================================================================
// This file provides:
//   1. RTELBenchmarkHarness — runs end-to-end pipeline benchmarks
//   2. DiagnosticsReporter — aggregates all RTEL metrics into one report
//   3. HeartbeatPublisher — publishes periodic heartbeats to the ShmBus
//   4. MarketDataReplay — replays recorded market data through the pipeline
//   5. ConfigParser — simple INI/TOML-like configuration file parser
// =============================================================================

#include "rtel/shm_bus.hpp"
#include "rtel/ring_buffer.hpp"
#include "rtel/global_state_registry.hpp"
#include "rtel/module_wrapper.hpp"
#include "rtel/scheduler.hpp"
#include "rtel/latency_monitor.hpp"
#include "rtel/serialization.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// HeartbeatPublisher — periodic heartbeat to indicate RTEL is alive
// ---------------------------------------------------------------------------
class HeartbeatPublisher {
public:
    struct HeartbeatMsg {
        uint64_t timestamp_ns  = 0;
        uint64_t seq           = 0;
        uint32_t process_id    = 0;
        uint32_t module_mask   = 0;  // bitmask of active modules
        float    cpu_util_pct  = 0.0f;
        float    mem_util_pct  = 0.0f;
        char     hostname[16]  = {};
    };

    explicit HeartbeatPublisher(uint64_t interval_ns = 1'000'000'000ULL)
        : interval_ns_(interval_ns) {}

    ~HeartbeatPublisher() { stop(); }

    void start() {
        if (running_.exchange(true)) return;
        thread_ = std::thread([this]() { loop(); });
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (thread_.joinable()) thread_.join();
    }

    bool is_running() const noexcept { return running_.load(); }

    uint64_t heartbeats_sent() const noexcept {
        return seq_.load(std::memory_order_relaxed);
    }

private:
    void loop() {
        auto& bus = ShmBus::instance();
        while (running_.load(std::memory_order_acquire)) {
            auto* ch = bus.channel(channels::HEARTBEAT);
            if (ch) {
                auto [handle, err] = ch->claim(false);
                if (err == ShmChannel::Error::OK) {
                    HeartbeatMsg* msg = reinterpret_cast<HeartbeatMsg*>(handle.data());
                    msg->timestamp_ns = now_ns();
                    msg->seq          = seq_.fetch_add(1, std::memory_order_relaxed);
#if defined(RTEL_PLATFORM_POSIX)
                    msg->process_id   = static_cast<uint32_t>(getpid());
#endif
                    msg->module_mask  = 0x3F;  // all 6 modules active

                    TensorDescriptor td{};
                    td.dtype         = DType::UINT8;
                    td.ndim          = 1;
                    td.shape[0]      = sizeof(HeartbeatMsg);
                    td.payload_bytes = sizeof(HeartbeatMsg);
                    handle.set_tensor(td);
                    handle.set_flags(SlotHeader::FLAG_HEARTBEAT | SlotHeader::FLAG_VALID);
                    handle.publish();
                }
            }
            std::this_thread::sleep_for(std::chrono::nanoseconds(interval_ns_));
        }
    }

    uint64_t interval_ns_;
    std::atomic<bool>     running_{false};
    std::atomic<uint64_t> seq_{0};
    std::thread           thread_;
};

// ---------------------------------------------------------------------------
// RTELBenchmarkHarness — end-to-end pipeline benchmark
// ---------------------------------------------------------------------------
struct BenchmarkResult {
    std::string name;
    uint64_t    iterations      = 0;
    double      elapsed_s       = 0.0;
    double      throughput_ops  = 0.0;
    double      mean_latency_us = 0.0;
    double      p50_latency_us  = 0.0;
    double      p95_latency_us  = 0.0;
    double      p99_latency_us  = 0.0;
    double      p999_latency_us = 0.0;
    uint64_t    errors          = 0;

    void print() const {
        std::printf("=== Benchmark: %s ===\n", name.c_str());
        std::printf("  Iterations:  %lu\n",  iterations);
        std::printf("  Elapsed:     %.3fs\n", elapsed_s);
        std::printf("  Throughput:  %.1f ops/s\n", throughput_ops);
        std::printf("  Mean:        %.2f µs\n", mean_latency_us);
        std::printf("  p50:         %.2f µs\n", p50_latency_us);
        std::printf("  p95:         %.2f µs\n", p95_latency_us);
        std::printf("  p99:         %.2f µs\n", p99_latency_us);
        std::printf("  p99.9:       %.2f µs\n", p999_latency_us);
        std::printf("  Errors:      %lu\n",  errors);
    }

    std::string to_json() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "{"
            << "\"name\":\"" << name << "\","
            << "\"iterations\":" << iterations << ","
            << "\"elapsed_s\":" << elapsed_s << ","
            << "\"throughput_ops\":" << throughput_ops << ","
            << "\"mean_latency_us\":" << mean_latency_us << ","
            << "\"p50_us\":" << p50_latency_us << ","
            << "\"p95_us\":" << p95_latency_us << ","
            << "\"p99_us\":" << p99_latency_us << ","
            << "\"errors\":" << errors
            << "}";
        return oss.str();
    }
};

class RTELBenchmarkHarness {
public:
    RTELBenchmarkHarness() = default;

    // Benchmark: ShmBus write throughput
    BenchmarkResult bench_shm_write(std::size_t iterations = 100000,
                                     const std::string& channel = "bench.write") {
        BenchmarkResult r;
        r.name = "shm_write";
        r.iterations = iterations;

        auto& bus = ShmBus::instance();
        bus.register_channel({channel, 4096, 1024, true, false});
        auto* ch = bus.channel(channel);
        if (!ch) { r.errors = 1; return r; }

        std::vector<uint64_t> lats;
        lats.reserve(iterations);

        uint64_t t_start = now_ns();
        for (std::size_t i = 0; i < iterations; ++i) {
            uint64_t t0 = now_ns();
            auto [handle, err] = ch->claim(false);
            if (err != ShmChannel::Error::OK) {
                ++r.errors;
                continue;
            }
            *reinterpret_cast<uint64_t*>(handle.data()) = i;
            handle.publish();
            uint64_t t1 = now_ns();
            lats.push_back(t1 - t0);
        }
        uint64_t t_end = now_ns();

        r.elapsed_s = static_cast<double>(t_end - t_start) / 1e9;
        r.throughput_ops = iterations / r.elapsed_s;
        fill_percentiles(lats, r);
        return r;
    }

    // Benchmark: GSR update throughput
    BenchmarkResult bench_gsr_update(std::size_t iterations = 50000) {
        BenchmarkResult r;
        r.name = "gsr_update_lob";
        r.iterations = iterations;

        LOBSnapshot snap{};
        snap.asset_id = 0;
        snap.n_bid_levels = 5;
        snap.n_ask_levels = 5;
        for (int i = 0; i < 5; ++i) {
            snap.bids[i].price = 150.0 - (i+1)*0.01;
            snap.bids[i].size  = 100.0;
            snap.asks[i].price = 150.0 + (i+1)*0.01;
            snap.asks[i].size  = 100.0;
        }
        snap.compute_derived();

        std::vector<uint64_t> lats;
        lats.reserve(iterations);

        uint64_t t_start = now_ns();
        for (std::size_t i = 0; i < iterations; ++i) {
            uint64_t t0 = now_ns();
            snap.sequence = i;
            GlobalStateRegistry::instance().update_lob(0, snap);
            uint64_t t1 = now_ns();
            lats.push_back(t1 - t0);
        }
        uint64_t t_end = now_ns();

        r.elapsed_s = static_cast<double>(t_end - t_start) / 1e9;
        r.throughput_ops = iterations / r.elapsed_s;
        fill_percentiles(lats, r);
        return r;
    }

    // Benchmark: ring buffer SPMC
    BenchmarkResult bench_ring_buffer_spmc(std::size_t iterations = 500000,
                                            int n_consumers = 2) {
        BenchmarkResult r;
        r.name = "ring_buffer_spmc";
        r.iterations = iterations;

        RingBuffer<MarketEvent> rb(4096);
        std::atomic<bool> running{true};
        std::atomic<uint64_t> consumed{0};

        std::vector<std::thread> consumers;
        for (int c = 0; c < n_consumers; ++c) {
            consumers.emplace_back([&]() {
                uint64_t cur = rb.new_consumer_cursor();
                while (running.load()) {
                    if (rb.try_consume(cur)) {
                        consumed.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
        }

        std::vector<uint64_t> lats;
        lats.reserve(iterations);

        uint64_t t_start = now_ns();
        for (std::size_t i = 0; i < iterations; ++i) {
            MarketEvent ev{};
            ev.timestamp_ns = now_ns();
            ev.asset_id = 0;
            ev.price = 150.0;
            uint64_t t0 = now_ns();
            rb.publish_blocking(ev);
            uint64_t t1 = now_ns();
            lats.push_back(t1 - t0);
        }
        uint64_t t_end = now_ns();

        running.store(false);
        for (auto& t : consumers) t.join();

        r.elapsed_s = static_cast<double>(t_end - t_start) / 1e9;
        r.throughput_ops = iterations / r.elapsed_s;
        fill_percentiles(lats, r);
        return r;
    }

    // Run all benchmarks
    std::vector<BenchmarkResult> run_all(std::size_t iters = 50000) {
        std::vector<BenchmarkResult> results;
        results.push_back(bench_shm_write(iters));
        results.push_back(bench_gsr_update(iters));
        results.push_back(bench_ring_buffer_spmc(iters));
        return results;
    }

    void print_all(const std::vector<BenchmarkResult>& results) {
        for (const auto& r : results) r.print();
    }

private:
    void fill_percentiles(std::vector<uint64_t>& lats, BenchmarkResult& r) {
        if (lats.empty()) return;
        std::sort(lats.begin(), lats.end());
        auto pct = [&](double p) -> double {
            std::size_t idx = static_cast<std::size_t>(p / 100.0 * lats.size());
            idx = std::min(idx, lats.size() - 1);
            return lats[idx] / 1000.0;
        };
        r.mean_latency_us = std::accumulate(lats.begin(), lats.end(), 0ULL)
                           / static_cast<double>(lats.size()) / 1000.0;
        r.p50_latency_us  = pct(50.0);
        r.p95_latency_us  = pct(95.0);
        r.p99_latency_us  = pct(99.0);
        r.p999_latency_us = pct(99.9);
    }
};

// ---------------------------------------------------------------------------
// MarketDataReplay — replay recorded LOB ticks
// ---------------------------------------------------------------------------
class MarketDataReplay {
public:
    struct TickRecord {
        uint64_t timestamp_ns;
        uint32_t asset_id;
        double   bid_price;
        double   ask_price;
        double   bid_size;
        double   ask_size;
        double   last_trade_price;
        double   last_trade_size;
    };

    explicit MarketDataReplay(std::size_t max_ticks = 100000)
        : max_ticks_(max_ticks) {
        ticks_.reserve(max_ticks);
    }

    void add_tick(const TickRecord& tick) {
        if (ticks_.size() < max_ticks_) {
            ticks_.push_back(tick);
        }
    }

    // Generate synthetic tick data for N assets
    void generate_synthetic(uint32_t n_assets, std::size_t n_ticks,
                             double base_price = 150.0,
                             double vol = 0.02) {
        ticks_.clear();
        std::mt19937_64 rng(12345);
        std::normal_distribution<double> noise(0.0, vol * base_price / 100.0);

        std::vector<double> prices(n_assets);
        for (uint32_t i = 0; i < n_assets; ++i) {
            prices[i] = base_price * (1.0 + i * 0.05);
        }

        uint64_t ts = now_ns();
        for (std::size_t t = 0; t < n_ticks; ++t) {
            for (uint32_t i = 0; i < n_assets; ++i) {
                prices[i] += noise(rng);
                prices[i] = std::max(1.0, prices[i]);
                TickRecord rec{};
                rec.timestamp_ns     = ts;
                rec.asset_id         = i;
                rec.bid_price        = prices[i] - 0.005;
                rec.ask_price        = prices[i] + 0.005;
                rec.bid_size         = 100.0 + std::abs(noise(rng)) * 10;
                rec.ask_size         = 100.0 + std::abs(noise(rng)) * 10;
                rec.last_trade_price = prices[i];
                rec.last_trade_size  = 50.0;
                ticks_.push_back(rec);
            }
            ts += 1'000'000;  // 1ms between ticks
        }
        std::printf("[Replay] Generated %zu ticks for %u assets\n",
                    ticks_.size(), n_assets);
    }

    // Replay ticks at real-time speed or as fast as possible
    void replay(Scheduler& scheduler, bool realtime = false) {
        if (ticks_.empty()) return;

        uint64_t t0_wall = now_ns();
        uint64_t t0_data = ticks_[0].timestamp_ns;

        for (const auto& tick : ticks_) {
            if (realtime) {
                uint64_t wall_now  = now_ns() - t0_wall;
                uint64_t data_time = tick.timestamp_ns - t0_data;
                if (data_time > wall_now) {
                    std::this_thread::sleep_for(
                        std::chrono::nanoseconds(data_time - wall_now));
                }
            }

            MarketEvent ev{};
            ev.timestamp_ns = tick.timestamp_ns;
            ev.asset_id     = tick.asset_id;
            ev.price        = (tick.bid_price + tick.ask_price) * 0.5;
            ev.bid          = tick.bid_price;
            ev.ask          = tick.ask_price;
            ev.size         = tick.last_trade_size;
            ev.event_type   = 2;  // quote
            scheduler.inject_market_event(ev);
        }
    }

    std::size_t size() const noexcept { return ticks_.size(); }
    const std::vector<TickRecord>& ticks() const noexcept { return ticks_; }

private:
    std::size_t max_ticks_;
    std::vector<TickRecord> ticks_;
};

// ---------------------------------------------------------------------------
// DiagnosticsReporter — aggregates all RTEL diagnostics
// ---------------------------------------------------------------------------
class DiagnosticsReporter {
public:
    struct Report {
        uint64_t    timestamp_ns;
        std::string version;
        // ShmBus stats
        std::unordered_map<std::string, ChannelStats> channel_stats;
        // GSR stats
        GlobalStateRegistry::Stats gsr_stats;
        // Scheduler stats
        Scheduler::Stats scheduler_stats;
        // Latency monitor
        std::unordered_map<std::string, LatencyMonitor::StageStats> stage_stats;
        // Prometheus export
        std::string prometheus_metrics;

        void print() const {
            std::printf("=== AETERNUS RTEL Diagnostics ===\n");
            std::printf("Timestamp:  %lu ns\n", timestamp_ns);
            std::printf("Version:    %s\n\n", version.c_str());

            std::printf("--- GSR ---\n");
            std::printf("  Writes:         %lu\n", gsr_stats.total_writes);
            std::printf("  Reads:          %lu\n", gsr_stats.total_reads);
            std::printf("  SeqLockRetries: %lu\n\n", gsr_stats.seqlock_retries);

            std::printf("--- Scheduler ---\n");
            std::printf("  Pipelines:      %lu\n", scheduler_stats.pipelines_run);
            std::printf("  SLA Violations: %lu\n", scheduler_stats.sla_violations);
            std::printf("  Mean Latency:   %.1f µs\n\n", scheduler_stats.mean_latency_us);

            std::printf("--- ShmBus Channels ---\n");
            for (const auto& [name, stats] : channel_stats) {
                std::printf("  %-40s pub=%lu cons=%lu util=%.1f%%\n",
                            name.c_str(), stats.published_total,
                            stats.consumed_total, stats.utilization_pct);
            }
            std::printf("\n");
        }

        std::string to_json() const {
            std::ostringstream oss;
            oss << "{"
                << "\"timestamp_ns\":" << timestamp_ns << ","
                << "\"version\":\"" << version << "\","
                << "\"gsr\":{\"writes\":" << gsr_stats.total_writes
                << ",\"reads\":" << gsr_stats.total_reads << "},"
                << "\"scheduler\":{\"pipelines\":" << scheduler_stats.pipelines_run
                << ",\"sla_violations\":" << scheduler_stats.sla_violations
                << ",\"mean_latency_us\":" << scheduler_stats.mean_latency_us << "}"
                << "}";
            return oss.str();
        }
    };

    Report collect(Scheduler* scheduler = nullptr) const {
        Report r{};
        r.timestamp_ns  = now_ns();
        r.version       = "0.1.0";

        // ShmBus
        r.channel_stats = ShmBus::instance().all_stats();

        // GSR
        r.gsr_stats = GlobalStateRegistry::instance().get_stats();

        // Scheduler
        if (scheduler) {
            r.scheduler_stats = scheduler->get_stats();
        }

        // Latency monitor
        auto& mon = LatencyMonitor::instance();
        for (std::size_t i = 0; i < static_cast<std::size_t>(PipelineStage::COUNT); ++i) {
            PipelineStage stage = static_cast<PipelineStage>(i);
            auto stats = mon.get_stage_stats(stage);
            if (stats.count > 0) {
                r.stage_stats[stage_name(stage)] = stats;
            }
        }

        // Prometheus
        r.prometheus_metrics = mon.prometheus_export();
        return r;
    }

    // Write Prometheus metrics to file (for scraping)
    bool write_prometheus_file(const std::string& path,
                                Scheduler* scheduler = nullptr) const {
        auto report = collect(scheduler);
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << report.prometheus_metrics;
        // Add extra gauges
        f << "\n# HELP rtel_gsr_version Current GSR version\n";
        f << "# TYPE rtel_gsr_version gauge\n";
        f << "rtel_gsr_version " << GlobalStateRegistry::instance().version() << "\n";
        if (scheduler) {
            f << "\n# HELP rtel_pipelines_total Total pipeline executions\n";
            f << "# TYPE rtel_pipelines_total counter\n";
            f << "rtel_pipelines_total " << report.scheduler_stats.pipelines_run << "\n";
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// ConfigParser — simple key=value config file parser
// ---------------------------------------------------------------------------
class ConfigParser {
public:
    struct Section {
        std::string name;
        std::unordered_map<std::string, std::string> values;
    };

    bool parse_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        return parse_stream(f);
    }

    bool parse_string(const std::string& content) {
        std::istringstream iss(content);
        return parse_stream(iss);
    }

    std::string get(const std::string& section, const std::string& key,
                    const std::string& default_val = "") const {
        auto sit = sections_.find(section);
        if (sit == sections_.end()) return default_val;
        auto kit = sit->second.values.find(key);
        return (kit != sit->second.values.end()) ? kit->second : default_val;
    }

    int get_int(const std::string& section, const std::string& key,
                int default_val = 0) const {
        auto s = get(section, key);
        return s.empty() ? default_val : std::stoi(s);
    }

    double get_double(const std::string& section, const std::string& key,
                      double default_val = 0.0) const {
        auto s = get(section, key);
        return s.empty() ? default_val : std::stod(s);
    }

    bool get_bool(const std::string& section, const std::string& key,
                  bool default_val = false) const {
        auto s = get(section, key);
        if (s.empty()) return default_val;
        return (s == "true" || s == "1" || s == "yes" || s == "on");
    }

    // Build SchedulerConfig from config file
    SchedulerConfig to_scheduler_config() const {
        SchedulerConfig cfg{};
        cfg.n_workers          = get_int("scheduler", "n_workers", 4);
        cfg.pin_cpus           = get_bool("scheduler", "pin_cpus", false);
        cfg.real_time_priority = get_bool("scheduler", "real_time_priority", false);
        cfg.pipeline_timeout_ns= static_cast<uint64_t>(
            get_int("scheduler", "pipeline_timeout_ms", 2) * 1'000'000);
        cfg.enable_watchdog    = get_bool("scheduler", "enable_watchdog", true);
        return cfg;
    }

    void print() const {
        for (const auto& [name, sec] : sections_) {
            std::printf("[%s]\n", name.c_str());
            for (const auto& [k, v] : sec.values) {
                std::printf("  %s = %s\n", k.c_str(), v.c_str());
            }
        }
    }

private:
    std::unordered_map<std::string, Section> sections_;
    std::string current_section_ = "default";

    bool parse_stream(std::istream& in) {
        std::string line;
        sections_[current_section_] = {current_section_, {}};
        while (std::getline(in, line)) {
            // Trim
            auto trim = [](std::string s) {
                std::size_t start = s.find_first_not_of(" \t\r\n");
                std::size_t end   = s.find_last_not_of(" \t\r\n");
                return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
            };
            line = trim(line);
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;
            if (line[0] == '[') {
                // Section header
                auto end = line.find(']');
                if (end != std::string::npos) {
                    current_section_ = line.substr(1, end - 1);
                    if (sections_.find(current_section_) == sections_.end()) {
                        sections_[current_section_] = {current_section_, {}};
                    }
                }
            } else {
                // key = value
                auto eq = line.find('=');
                if (eq != std::string::npos) {
                    std::string key   = trim(line.substr(0, eq));
                    std::string value = trim(line.substr(eq + 1));
                    // Remove inline comment
                    auto comment = value.find('#');
                    if (comment != std::string::npos)
                        value = trim(value.substr(0, comment));
                    sections_[current_section_].values[key] = value;
                }
            }
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// OrderBookBuilder — builds LOBSnapshot from individual order events
// ---------------------------------------------------------------------------
class OrderBookBuilder {
public:
    enum class Side { BID, ASK };
    enum class Type { ADD, MODIFY, CANCEL };

    struct OrderEvent {
        uint64_t order_id;
        uint64_t timestamp_ns;
        Side     side;
        Type     type;
        double   price;
        double   size;
        uint32_t asset_id;
    };

    explicit OrderBookBuilder(uint32_t asset_id) : asset_id_(asset_id) {}

    void process(const OrderEvent& ev) {
        ++event_count_;
        switch (ev.type) {
            case Type::ADD:
                (ev.side == Side::BID ? bids_ : asks_)[ev.price] += ev.size;
                break;
            case Type::MODIFY:
                (ev.side == Side::BID ? bids_ : asks_)[ev.price] = ev.size;
                break;
            case Type::CANCEL:
                auto& book = (ev.side == Side::BID ? bids_ : asks_);
                book[ev.price] -= ev.size;
                if (book[ev.price] <= 0.0) book.erase(ev.price);
                break;
        }
        ++seq_;
    }

    LOBSnapshot build_snapshot() const {
        LOBSnapshot snap{};
        snap.asset_id   = asset_id_;
        snap.recv_ts_ns = now_ns();
        snap.sequence   = seq_;

        // Top bids (highest prices first)
        auto bit = bids_.rbegin();
        for (uint32_t i = 0; i < kMaxLOBLevels && bit != bids_.rend(); ++i, ++bit) {
            snap.bids[i].price = bit->first;
            snap.bids[i].size  = bit->second;
            ++snap.n_bid_levels;
        }
        // Top asks (lowest prices first)
        auto ait = asks_.begin();
        for (uint32_t i = 0; i < kMaxLOBLevels && ait != asks_.end(); ++i, ++ait) {
            snap.asks[i].price = ait->first;
            snap.asks[i].size  = ait->second;
            ++snap.n_ask_levels;
        }
        snap.compute_derived();
        return snap;
    }

    double best_bid() const {
        return bids_.empty() ? 0.0 : bids_.rbegin()->first;
    }
    double best_ask() const {
        return asks_.empty() ? 0.0 : asks_.begin()->first;
    }
    double mid_price() const {
        return (best_bid() + best_ask()) * 0.5;
    }
    uint64_t sequence() const noexcept { return seq_; }
    uint64_t event_count() const noexcept { return event_count_; }

    void clear() {
        bids_.clear();
        asks_.clear();
        seq_ = 0;
    }

private:
    uint32_t asset_id_;
    std::map<double, double> bids_;  // price → size (ascending)
    std::map<double, double> asks_;  // price → size (ascending)
    uint64_t seq_ = 0;
    uint64_t event_count_ = 0;

    // Need map include
    #include <map>
};

// ---------------------------------------------------------------------------
// Feature engineering utilities
// ---------------------------------------------------------------------------
namespace features {

// Compute rolling statistics over a window
struct RollingStats {
    double mean  = 0.0;
    double var   = 0.0;
    double std   = 0.0;
    double min   = 0.0;
    double max   = 0.0;
    double range = 0.0;
    double skew  = 0.0;
};

RollingStats compute_rolling(const double* data, std::size_t n) {
    if (n == 0) return {};
    RollingStats s{};
    s.min = *std::min_element(data, data + n);
    s.max = *std::max_element(data, data + n);
    s.range = s.max - s.min;

    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += data[i];
    s.mean = sum / n;

    double var_sum = 0.0;
    double skew_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double diff = data[i] - s.mean;
        var_sum  += diff * diff;
        skew_sum += diff * diff * diff;
    }
    s.var  = var_sum / n;
    s.std  = std::sqrt(s.var);
    double std3 = s.std * s.std * s.std;
    s.skew = (std3 > 1e-10) ? (skew_sum / n) / std3 : 0.0;
    return s;
}

// Exponential moving average
double ema(double current, double prev_ema, double alpha = 0.1) noexcept {
    return alpha * current + (1.0 - alpha) * prev_ema;
}

// ATR (Average True Range) proxy for LOB
double atr(const LOBSnapshot& snap, double prev_close, int period = 14) noexcept {
    double high = snap.asks[0].price;
    double low  = snap.bids[0].price;
    double tr   = std::max({high - low,
                            std::fabs(high - prev_close),
                            std::fabs(low  - prev_close)});
    return tr;
}

// VWAP deviation signal
double vwap_signal(const LOBSnapshot& snap) noexcept {
    double vwap = (snap.vwap_bid + snap.vwap_ask) * 0.5;
    return (snap.mid_price - vwap) / (snap.spread + 1e-10);
}

// Order book pressure (weighted imbalance using multiple levels)
double ob_pressure(const LOBSnapshot& snap, int levels = 5) noexcept {
    double bid_vol = 0.0, ask_vol = 0.0;
    int n = std::min((int)snap.n_bid_levels, std::min((int)snap.n_ask_levels, levels));
    for (int i = 0; i < n; ++i) {
        double weight = 1.0 / (i + 1.0);  // higher weight to closer levels
        bid_vol += snap.bids[i].size * weight;
        ask_vol += snap.asks[i].size * weight;
    }
    double total = bid_vol + ask_vol;
    return (total > 0.0) ? (bid_vol - ask_vol) / total : 0.0;
}

// Micro-price (weighted mid by top-of-book quantities)
double micro_price(const LOBSnapshot& snap) noexcept {
    if (snap.n_bid_levels == 0 || snap.n_ask_levels == 0) return snap.mid_price;
    double bs = snap.bids[0].size;
    double as = snap.asks[0].size;
    return (snap.bids[0].price * as + snap.asks[0].price * bs) / (bs + as);
}

// Kyle lambda proxy (price impact per unit volume)
double kyle_lambda_proxy(double price_change, double volume) noexcept {
    if (std::fabs(volume) < 1e-10) return 0.0;
    return price_change / volume;
}

} // namespace features

} // namespace aeternus::rtel
