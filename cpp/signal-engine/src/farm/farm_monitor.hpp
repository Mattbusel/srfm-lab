#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <deque>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <mutex>
#include <atomic>
#include <functional>
#include <sstream>
#include <iomanip>
#include <optional>
#include <limits>
#include <queue>

namespace srfm::farm {

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
struct BacktestResult;
struct DashboardSnapshot;
struct AnomalyAlert;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------
enum class BacktestPhase : uint8_t {
    DataLoad      = 0,
    SignalCompute = 1,
    ExecutionSim  = 2,
    MetricsCalc   = 3,
    Total         = 4,
    COUNT         = 5
};

[[nodiscard]] inline const char* phase_name(BacktestPhase p) noexcept {
    switch (p) {
        case BacktestPhase::DataLoad:      return "data_load";
        case BacktestPhase::SignalCompute:  return "signal_compute";
        case BacktestPhase::ExecutionSim:   return "execution_sim";
        case BacktestPhase::MetricsCalc:    return "metrics_calc";
        case BacktestPhase::Total:          return "total";
        default:                            return "unknown";
    }
}

enum class BacktestStatus : uint8_t {
    Queued    = 0,
    Running   = 1,
    Completed = 2,
    Failed    = 3,
    COUNT     = 4
};

[[nodiscard]] inline const char* status_name(BacktestStatus s) noexcept {
    switch (s) {
        case BacktestStatus::Queued:    return "queued";
        case BacktestStatus::Running:   return "running";
        case BacktestStatus::Completed: return "completed";
        case BacktestStatus::Failed:    return "failed";
        default:                        return "unknown";
    }
}

enum class AnomalyType : uint8_t {
    ZScoreOutlier        = 0,
    ImpossiblePerformance = 1,
    NegativeCost         = 2,
    RegimeInconsistency  = 3,
    COUNT                = 4
};

[[nodiscard]] inline const char* anomaly_name(AnomalyType a) noexcept {
    switch (a) {
        case AnomalyType::ZScoreOutlier:         return "z_score_outlier";
        case AnomalyType::ImpossiblePerformance:  return "impossible_performance";
        case AnomalyType::NegativeCost:           return "negative_cost";
        case AnomalyType::RegimeInconsistency:    return "regime_inconsistency";
        default:                                  return "unknown";
    }
}

// ---------------------------------------------------------------------------
// BacktestResult — the data flowing from workers into the monitor
// ---------------------------------------------------------------------------
struct BacktestResult {
    uint64_t    backtest_id{0};
    std::string strategy_name;
    std::string symbol;
    std::string timeframe;      // e.g. "1m", "5m", "1h"
    std::string regime_label;   // e.g. "trending", "mean_revert"

    double sharpe{0.0};
    double annualized_return{0.0};
    double max_drawdown{0.0};
    double sortino{0.0};
    double calmar{0.0};
    double win_rate{0.0};
    int    num_trades{0};

    double cost_bps{0.0};       // transaction cost in basis points
    double sharpe_no_cost{0.0}; // sharpe before costs

    BacktestStatus status{BacktestStatus::Completed};
    std::string    error_message;

    // Timing breakdown in microseconds for each phase
    std::array<double, static_cast<size_t>(BacktestPhase::COUNT)> phase_us{};

    // Additional param string for identification
    std::string params_json;
};

// ---------------------------------------------------------------------------
// AnomalyAlert
// ---------------------------------------------------------------------------
struct AnomalyAlert {
    uint64_t    backtest_id{0};
    AnomalyType type{AnomalyType::ZScoreOutlier};
    std::string description;
    double      severity{0.0};   // 0..1 scale
    double      value{0.0};      // the offending metric value
    double      threshold{0.0};  // the threshold it crossed
    uint64_t    timestamp_ns{0};
};

// ---------------------------------------------------------------------------
// PerformanceTimer
// ---------------------------------------------------------------------------
class PerformanceTimer {
public:
    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration  = std::chrono::nanoseconds;

    PerformanceTimer() noexcept;

    void   start() noexcept;
    void   stop() noexcept;
    void   reset() noexcept;

    [[nodiscard]] int64_t elapsed_ns() const noexcept;
    [[nodiscard]] double  elapsed_us() const noexcept;
    [[nodiscard]] double  elapsed_ms() const noexcept;
    [[nodiscard]] bool    is_running() const noexcept;

private:
    TimePoint start_;
    TimePoint stop_;
    bool      running_{false};
};

// ---------------------------------------------------------------------------
// TimingAccumulator — online statistics for timing samples
// ---------------------------------------------------------------------------
class TimingAccumulator {
public:
    TimingAccumulator() noexcept;

    void   add_sample(double value_us) noexcept;
    void   reset() noexcept;

    [[nodiscard]] uint64_t count()  const noexcept;
    [[nodiscard]] double   min()    const noexcept;
    [[nodiscard]] double   max()    const noexcept;
    [[nodiscard]] double   mean()   const noexcept;
    [[nodiscard]] double   variance() const noexcept;
    [[nodiscard]] double   stddev()  const noexcept;

    // Approximate percentiles via P-square or sorted reservoir
    [[nodiscard]] double   p50() const noexcept;
    [[nodiscard]] double   p95() const noexcept;
    [[nodiscard]] double   p99() const noexcept;

    [[nodiscard]] const std::vector<double>& reservoir() const noexcept;

private:
    static constexpr size_t kReservoirCap = 4096;

    uint64_t count_{0};
    double   min_{std::numeric_limits<double>::max()};
    double   max_{std::numeric_limits<double>::lowest()};

    // Welford online mean/variance
    double   welford_mean_{0.0};
    double   welford_m2_{0.0};

    // Reservoir sampling for percentiles
    std::vector<double> reservoir_;
    mutable bool        reservoir_sorted_{false};
    mutable std::vector<double> sorted_cache_;

    void ensure_sorted() const noexcept;
};

// ---------------------------------------------------------------------------
// BacktestProfiler
// ---------------------------------------------------------------------------
class BacktestProfiler {
public:
    static constexpr size_t kNumPhases = static_cast<size_t>(BacktestPhase::COUNT);

    BacktestProfiler();

    void begin_phase(BacktestPhase phase);
    void end_phase(BacktestPhase phase);
    void record_phase(BacktestPhase phase, double elapsed_us);

    void reset();

    [[nodiscard]] const TimingAccumulator& accumulator(BacktestPhase phase) const;
    [[nodiscard]] BacktestPhase            bottleneck() const;
    [[nodiscard]] double                   bottleneck_fraction() const;
    [[nodiscard]] std::string              summary() const;

    // Record all phases from a completed backtest result
    void ingest(const BacktestResult& result);

private:
    std::array<TimingAccumulator, kNumPhases> accumulators_;
    std::array<PerformanceTimer, kNumPhases>  timers_;
    mutable std::mutex                        mu_;
};

// ---------------------------------------------------------------------------
// ThroughputTracker
// ---------------------------------------------------------------------------
class ThroughputTracker {
public:
    // Rolling windows in seconds
    static constexpr std::array<double, 4> kWindows = {1.0, 10.0, 60.0, 300.0};

    ThroughputTracker();

    void record_completion(const std::string& strategy_type,
                           const std::string& symbol,
                           const std::string& timeframe);

    void prune_old_entries();

    [[nodiscard]] double throughput(double window_seconds) const;
    [[nodiscard]] double peak_throughput_1s() const noexcept;
    [[nodiscard]] double average_throughput() const;

    [[nodiscard]] std::unordered_map<std::string, double>
        throughput_by_strategy(double window_seconds) const;
    [[nodiscard]] std::unordered_map<std::string, double>
        throughput_by_symbol(double window_seconds) const;
    [[nodiscard]] std::unordered_map<std::string, double>
        throughput_by_timeframe(double window_seconds) const;

    [[nodiscard]] uint64_t total_completed() const noexcept;
    [[nodiscard]] std::string summary() const;

    void reset();

private:
    struct CompletionEvent {
        double      timestamp_s;
        std::string strategy_type;
        std::string symbol;
        std::string timeframe;
    };

    std::deque<CompletionEvent> events_;
    uint64_t                    total_completed_{0};
    double                      peak_throughput_1s_{0.0};
    double                      start_time_s_{0.0};
    bool                        started_{false};
    mutable std::mutex          mu_;

    [[nodiscard]] double now_seconds() const;
};

// ---------------------------------------------------------------------------
// ResourceMonitor
// ---------------------------------------------------------------------------
class ResourceMonitor {
public:
    ResourceMonitor();
    explicit ResourceMonitor(uint32_t num_workers);

    // Memory tracking (simulated: manual allocation accounting)
    void record_alloc(size_t bytes);
    void record_dealloc(size_t bytes);
    [[nodiscard]] size_t current_memory_bytes() const noexcept;
    [[nodiscard]] size_t peak_memory_bytes() const noexcept;

    // Queue depth
    void set_queue_depth(uint64_t depth);
    [[nodiscard]] uint64_t current_queue_depth() const noexcept;
    [[nodiscard]] uint64_t peak_queue_depth() const noexcept;
    [[nodiscard]] const std::deque<std::pair<double, uint64_t>>& queue_history() const noexcept;

    // Worker utilization
    void worker_started_task(uint32_t worker_id);
    void worker_finished_task(uint32_t worker_id);
    [[nodiscard]] double worker_utilization(uint32_t worker_id) const;
    [[nodiscard]] double average_utilization() const;

    // ETA estimation
    void set_total_backtests(uint64_t total);
    void set_completed_backtests(uint64_t completed);
    [[nodiscard]] double estimated_seconds_remaining() const;
    [[nodiscard]] std::string eta_string() const;

    [[nodiscard]] uint32_t num_workers() const noexcept;
    [[nodiscard]] std::string summary() const;

    void reset();

private:
    struct WorkerState {
        bool     busy{false};
        double   busy_start_s{0.0};
        double   total_busy_s{0.0};
        double   total_idle_s{0.0};
        double   last_transition_s{0.0};
        uint64_t tasks_completed{0};
    };

    uint32_t                    num_workers_{0};
    std::vector<WorkerState>    workers_;

    std::atomic<size_t>         current_memory_{0};
    size_t                      peak_memory_{0};

    uint64_t                    current_queue_depth_{0};
    uint64_t                    peak_queue_depth_{0};
    std::deque<std::pair<double, uint64_t>> queue_history_;
    static constexpr size_t     kMaxQueueHistory = 2048;

    uint64_t                    total_backtests_{0};
    uint64_t                    completed_backtests_{0};
    double                      first_completion_time_s_{0.0};
    bool                        has_first_completion_{false};

    mutable std::mutex          mu_;

    [[nodiscard]] double now_seconds() const;
};

// ---------------------------------------------------------------------------
// RunningHistogram — fixed-bin histogram for online streaming data
// ---------------------------------------------------------------------------
class RunningHistogram {
public:
    RunningHistogram();
    RunningHistogram(double low, double high, uint32_t num_bins);

    void   add(double value);
    void   reset();

    [[nodiscard]] uint64_t count() const noexcept;
    [[nodiscard]] uint64_t underflow() const noexcept;
    [[nodiscard]] uint64_t overflow() const noexcept;

    [[nodiscard]] double   bin_low(uint32_t bin) const noexcept;
    [[nodiscard]] double   bin_high(uint32_t bin) const noexcept;
    [[nodiscard]] double   bin_center(uint32_t bin) const noexcept;
    [[nodiscard]] uint64_t bin_count(uint32_t bin) const noexcept;
    [[nodiscard]] double   bin_frequency(uint32_t bin) const;

    [[nodiscard]] uint32_t num_bins() const noexcept;
    [[nodiscard]] double   low() const noexcept;
    [[nodiscard]] double   high() const noexcept;

    [[nodiscard]] std::string render_ascii(uint32_t width = 50) const;

private:
    double               low_{-3.0};
    double               high_{5.0};
    uint32_t             num_bins_{40};
    double               bin_width_{0.2};
    std::vector<uint64_t> bins_;
    uint64_t             count_{0};
    uint64_t             underflow_{0};
    uint64_t             overflow_{0};
};

// ---------------------------------------------------------------------------
// ResultAggregator
// ---------------------------------------------------------------------------
class ResultAggregator {
public:
    static constexpr size_t kTopK = 20;

    ResultAggregator();

    void ingest(const BacktestResult& result);
    void reset();

    // Online statistics (Welford's algorithm)
    [[nodiscard]] double sharpe_mean() const noexcept;
    [[nodiscard]] double sharpe_variance() const noexcept;
    [[nodiscard]] double sharpe_stddev() const noexcept;

    [[nodiscard]] double return_mean() const noexcept;
    [[nodiscard]] double return_variance() const noexcept;

    [[nodiscard]] double drawdown_mean() const noexcept;
    [[nodiscard]] double drawdown_variance() const noexcept;

    [[nodiscard]] uint64_t total_ingested() const noexcept;

    // Histogram
    [[nodiscard]] const RunningHistogram& sharpe_histogram() const noexcept;

    // Top-K / Worst-K
    [[nodiscard]] std::vector<BacktestResult> top_k_by_sharpe() const;
    [[nodiscard]] std::vector<BacktestResult> worst_k_by_drawdown() const;

    // Status counts
    [[nodiscard]] uint64_t count_by_status(BacktestStatus status) const noexcept;
    void record_status_change(BacktestStatus from, BacktestStatus to);
    void record_status(BacktestStatus status);

    [[nodiscard]] std::string summary() const;

private:
    // Welford accumulators
    struct WelfordState {
        uint64_t n{0};
        double   mean{0.0};
        double   m2{0.0};

        void add(double x) noexcept;
        [[nodiscard]] double variance() const noexcept;
        [[nodiscard]] double stddev() const noexcept;
    };

    WelfordState sharpe_acc_;
    WelfordState return_acc_;
    WelfordState dd_acc_;

    uint64_t total_ingested_{0};

    // Sharpe histogram: range [-3, 5] with 40 bins
    RunningHistogram sharpe_hist_;

    // Min-heap for top-K by sharpe (keep the K largest)
    struct SharpeMinCmp {
        bool operator()(const BacktestResult& a, const BacktestResult& b) const noexcept {
            return a.sharpe > b.sharpe; // min-heap: smallest on top
        }
    };
    std::priority_queue<BacktestResult, std::vector<BacktestResult>, SharpeMinCmp> top_k_sharpe_;

    // Max-heap for worst-K by drawdown (keep the K with largest DD)
    struct DrawdownMaxCmp {
        bool operator()(const BacktestResult& a, const BacktestResult& b) const noexcept {
            return a.max_drawdown > b.max_drawdown; // min-heap on DD => keeps largest
        }
    };
    std::priority_queue<BacktestResult, std::vector<BacktestResult>, DrawdownMaxCmp> worst_k_dd_;

    // Status counts
    std::array<std::atomic<uint64_t>, static_cast<size_t>(BacktestStatus::COUNT)> status_counts_{};

    mutable std::mutex mu_;
};

// ---------------------------------------------------------------------------
// AnomalyDetector
// ---------------------------------------------------------------------------
class AnomalyDetector {
public:
    struct Config {
        double z_score_threshold{3.0};
        double impossible_sharpe{5.0};
        int    impossible_min_trades{50};
        double negative_cost_threshold{0.0}; // sharpe_with_cost - sharpe_no_cost
        double regime_inconsistency_threshold{2.0}; // Sharpe diff
    };

    AnomalyDetector();
    explicit AnomalyDetector(const Config& cfg);

    // Run all checks on a result, return all anomalies found
    [[nodiscard]] std::vector<AnomalyAlert> check(
        const BacktestResult& result,
        double sharpe_mean,
        double sharpe_stddev) const;

    // Individual checks
    [[nodiscard]] std::optional<AnomalyAlert> check_zscore(
        const BacktestResult& result,
        double sharpe_mean,
        double sharpe_stddev) const;

    [[nodiscard]] std::optional<AnomalyAlert> check_impossible(
        const BacktestResult& result) const;

    [[nodiscard]] std::optional<AnomalyAlert> check_negative_cost(
        const BacktestResult& result) const;

    // Regime inconsistency requires history — must feed regime data
    void record_regime_result(const std::string& strategy,
                              const std::string& regime,
                              double sharpe);

    [[nodiscard]] std::optional<AnomalyAlert> check_regime_inconsistency(
        const BacktestResult& result) const;

    [[nodiscard]] const std::vector<AnomalyAlert>& alert_history() const noexcept;
    [[nodiscard]] uint64_t total_anomalies() const noexcept;

    void set_config(const Config& cfg);
    [[nodiscard]] const Config& config() const noexcept;

    void reset();

private:
    Config cfg_;

    // regime_data_[strategy][regime] = vector of sharpe values
    std::unordered_map<std::string,
        std::unordered_map<std::string, std::vector<double>>> regime_data_;

    std::vector<AnomalyAlert> history_;
    mutable std::mutex mu_;
};

// ---------------------------------------------------------------------------
// DashboardSnapshot — plain data struct for TUI / API consumption
// ---------------------------------------------------------------------------
struct DashboardSnapshot {
    // Status counts
    uint64_t queued{0};
    uint64_t running{0};
    uint64_t completed{0};
    uint64_t failed{0};

    // Throughput
    double throughput_1s{0.0};
    double throughput_10s{0.0};
    double throughput_60s{0.0};
    double throughput_300s{0.0};
    double peak_throughput{0.0};

    // Best result
    std::string best_strategy;
    std::string best_params;
    double      best_sharpe{std::numeric_limits<double>::lowest()};
    double      best_return{0.0};
    double      best_drawdown{0.0};

    // ETA
    double      eta_seconds{0.0};
    std::string eta_string;

    // Resources
    size_t   memory_bytes{0};
    size_t   peak_memory_bytes{0};
    double   avg_utilization{0.0};
    uint64_t queue_depth{0};

    // Anomalies
    uint64_t total_anomalies{0};
    std::vector<AnomalyAlert> recent_anomalies; // last 10

    // Sharpe distribution summary
    double sharpe_mean{0.0};
    double sharpe_stddev{0.0};
};

// ---------------------------------------------------------------------------
// FarmDashboard
// ---------------------------------------------------------------------------
class FarmDashboard {
public:
    FarmDashboard();

    void update(const DashboardSnapshot& snap);
    [[nodiscard]] const DashboardSnapshot& latest() const noexcept;

    // Text rendering
    [[nodiscard]] std::string render_status_line() const;
    [[nodiscard]] std::string render_throughput_line() const;
    [[nodiscard]] std::string render_best_result() const;
    [[nodiscard]] std::string render_eta() const;
    [[nodiscard]] std::string render_resources() const;
    [[nodiscard]] std::string render_anomalies() const;
    [[nodiscard]] std::string render_full() const;

private:
    DashboardSnapshot snap_;
    mutable std::mutex mu_;
};

// ---------------------------------------------------------------------------
// MetricLine — single Prometheus metric
// ---------------------------------------------------------------------------
struct MetricLine {
    std::string name;
    std::string type;        // "counter", "gauge", "histogram"
    std::string help;
    double      value{0.0};
    std::unordered_map<std::string, std::string> labels;

    [[nodiscard]] std::string render() const;
};

// ---------------------------------------------------------------------------
// MetricsExporter
// ---------------------------------------------------------------------------
class MetricsExporter {
public:
    MetricsExporter();

    // Build the full Prometheus exposition from a snapshot
    [[nodiscard]] std::string export_prometheus(const DashboardSnapshot& snap) const;

    // Individual metric helpers
    [[nodiscard]] MetricLine farm_backtests_total(uint64_t completed, uint64_t failed) const;
    [[nodiscard]] MetricLine farm_backtests_running(uint64_t running) const;
    [[nodiscard]] MetricLine farm_throughput_per_second(double tps) const;
    [[nodiscard]] MetricLine farm_best_sharpe(double sharpe) const;
    [[nodiscard]] MetricLine farm_queue_depth(uint64_t depth) const;
    [[nodiscard]] MetricLine farm_worker_utilization(double util) const;
    [[nodiscard]] MetricLine farm_anomalies_detected(uint64_t count) const;
    [[nodiscard]] MetricLine farm_memory_bytes(size_t bytes) const;
    [[nodiscard]] MetricLine farm_sharpe_mean(double mean) const;
    [[nodiscard]] MetricLine farm_sharpe_stddev(double sd) const;

    void set_prefix(const std::string& prefix);
    [[nodiscard]] const std::string& prefix() const noexcept;

private:
    std::string prefix_{"srfm_farm"};
};

// ---------------------------------------------------------------------------
// FarmMonitor — top-level orchestrator tying all sub-systems together
// ---------------------------------------------------------------------------
class FarmMonitor {
public:
    FarmMonitor();
    explicit FarmMonitor(uint32_t num_workers);

    // Lifecycle
    void start();
    void stop();

    // Ingest a completed backtest
    void ingest_result(const BacktestResult& result);

    // Queue management
    void on_backtest_queued();
    void on_backtest_started(uint32_t worker_id);
    void on_backtest_finished(uint32_t worker_id, const BacktestResult& result);
    void on_backtest_failed(uint32_t worker_id, const BacktestResult& result);

    // Memory tracking passthrough
    void record_alloc(size_t bytes);
    void record_dealloc(size_t bytes);

    // Set total expected
    void set_total_backtests(uint64_t total);

    // Snapshot
    [[nodiscard]] DashboardSnapshot snapshot() const;

    // Convenience renders
    [[nodiscard]] std::string render_dashboard() const;
    [[nodiscard]] std::string export_prometheus() const;

    // Accessors
    [[nodiscard]] const BacktestProfiler&   profiler()    const noexcept;
    [[nodiscard]] const ThroughputTracker&   throughput()  const noexcept;
    [[nodiscard]] const ResourceMonitor&     resources()   const noexcept;
    [[nodiscard]] const ResultAggregator&    aggregator()  const noexcept;
    [[nodiscard]] const AnomalyDetector&     anomalies()   const noexcept;
    [[nodiscard]] const FarmDashboard&       dashboard()   const noexcept;
    [[nodiscard]] const MetricsExporter&     exporter()    const noexcept;

    void reset();

private:
    BacktestProfiler  profiler_;
    ThroughputTracker throughput_;
    ResourceMonitor   resources_;
    ResultAggregator  aggregator_;
    AnomalyDetector   anomaly_detector_;
    FarmDashboard     dashboard_;
    MetricsExporter   exporter_;

    std::atomic<bool> running_{false};
    mutable std::mutex mu_;
};

} // namespace srfm::farm
