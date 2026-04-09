#include "farm_monitor.hpp"

#include <cassert>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace srfm::farm {

// ===================================================================
//  Utility helpers (file-local)
// ===================================================================
namespace {

[[nodiscard]] double steady_seconds() noexcept {
    using C = std::chrono::steady_clock;
    auto now = C::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

[[nodiscard]] uint64_t steady_nanos() noexcept {
    using C = std::chrono::steady_clock;
    auto now = C::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count());
}

[[nodiscard]] std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int idx = 0;
    double val = static_cast<double>(bytes);
    while (val >= 1024.0 && idx < 4) {
        val /= 1024.0;
        ++idx;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << val << " " << units[idx];
    return oss.str();
}

[[nodiscard]] std::string format_duration(double seconds) {
    if (seconds < 0.0) return "N/A";
    int h = static_cast<int>(seconds) / 3600;
    int m = (static_cast<int>(seconds) % 3600) / 60;
    int s = static_cast<int>(seconds) % 60;
    std::ostringstream oss;
    if (h > 0) oss << h << "h ";
    if (m > 0 || h > 0) oss << m << "m ";
    oss << s << "s";
    return oss.str();
}

[[nodiscard]] double percentile_sorted(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    double idx = p * static_cast<double>(sorted.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return sorted[lo];
    double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

} // anonymous namespace

// ===================================================================
//  PerformanceTimer
// ===================================================================

PerformanceTimer::PerformanceTimer() noexcept
    : start_(Clock::now()), stop_(start_), running_(false) {}

void PerformanceTimer::start() noexcept {
    start_ = Clock::now();
    running_ = true;
}

void PerformanceTimer::stop() noexcept {
    stop_ = Clock::now();
    running_ = false;
}

void PerformanceTimer::reset() noexcept {
    start_ = Clock::now();
    stop_ = start_;
    running_ = false;
}

int64_t PerformanceTimer::elapsed_ns() const noexcept {
    auto end = running_ ? Clock::now() : stop_;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
}

double PerformanceTimer::elapsed_us() const noexcept {
    return static_cast<double>(elapsed_ns()) / 1000.0;
}

double PerformanceTimer::elapsed_ms() const noexcept {
    return static_cast<double>(elapsed_ns()) / 1'000'000.0;
}

bool PerformanceTimer::is_running() const noexcept {
    return running_;
}

// ===================================================================
//  TimingAccumulator
// ===================================================================

TimingAccumulator::TimingAccumulator() noexcept {
    reservoir_.reserve(kReservoirCap);
}

void TimingAccumulator::add_sample(double value_us) noexcept {
    ++count_;
    if (value_us < min_) min_ = value_us;
    if (value_us > max_) max_ = value_us;

    // Welford update
    double delta = value_us - welford_mean_;
    welford_mean_ += delta / static_cast<double>(count_);
    double delta2 = value_us - welford_mean_;
    welford_m2_ += delta * delta2;

    // Reservoir sampling
    if (reservoir_.size() < kReservoirCap) {
        reservoir_.push_back(value_us);
    } else {
        // Simple modular replacement for approximate reservoir sampling.
        // Use a deterministic replacement strategy based on count to avoid
        // the cost of a random generator on every sample.
        size_t slot = count_ % kReservoirCap;
        // Replace with decreasing probability approximation
        if (slot < kReservoirCap) {
            // Use a simple hash of count to decide replacement
            uint64_t h = count_ * 2654435761ULL; // Knuth multiplicative hash
            size_t target = h % kReservoirCap;
            // Replace roughly kReservoirCap/count_ fraction of the time
            if ((h >> 32) % count_ < kReservoirCap) {
                reservoir_[target] = value_us;
            }
        }
    }
    reservoir_sorted_ = false;
}

void TimingAccumulator::reset() noexcept {
    count_ = 0;
    min_ = std::numeric_limits<double>::max();
    max_ = std::numeric_limits<double>::lowest();
    welford_mean_ = 0.0;
    welford_m2_ = 0.0;
    reservoir_.clear();
    reservoir_sorted_ = false;
    sorted_cache_.clear();
}

uint64_t TimingAccumulator::count() const noexcept { return count_; }
double TimingAccumulator::min() const noexcept {
    return count_ > 0 ? min_ : 0.0;
}
double TimingAccumulator::max() const noexcept {
    return count_ > 0 ? max_ : 0.0;
}
double TimingAccumulator::mean() const noexcept { return welford_mean_; }

double TimingAccumulator::variance() const noexcept {
    if (count_ < 2) return 0.0;
    return welford_m2_ / static_cast<double>(count_ - 1);
}

double TimingAccumulator::stddev() const noexcept {
    return std::sqrt(variance());
}

void TimingAccumulator::ensure_sorted() const noexcept {
    if (reservoir_sorted_) return;
    sorted_cache_ = reservoir_;
    std::sort(sorted_cache_.begin(), sorted_cache_.end());
    reservoir_sorted_ = true;
}

double TimingAccumulator::p50() const noexcept {
    ensure_sorted();
    return percentile_sorted(sorted_cache_, 0.50);
}

double TimingAccumulator::p95() const noexcept {
    ensure_sorted();
    return percentile_sorted(sorted_cache_, 0.95);
}

double TimingAccumulator::p99() const noexcept {
    ensure_sorted();
    return percentile_sorted(sorted_cache_, 0.99);
}

const std::vector<double>& TimingAccumulator::reservoir() const noexcept {
    return reservoir_;
}

// ===================================================================
//  BacktestProfiler
// ===================================================================

BacktestProfiler::BacktestProfiler() = default;

void BacktestProfiler::begin_phase(BacktestPhase phase) {
    std::lock_guard<std::mutex> lk(mu_);
    auto idx = static_cast<size_t>(phase);
    if (idx < kNumPhases) {
        timers_[idx].start();
    }
}

void BacktestProfiler::end_phase(BacktestPhase phase) {
    std::lock_guard<std::mutex> lk(mu_);
    auto idx = static_cast<size_t>(phase);
    if (idx < kNumPhases) {
        timers_[idx].stop();
        accumulators_[idx].add_sample(timers_[idx].elapsed_us());
    }
}

void BacktestProfiler::record_phase(BacktestPhase phase, double elapsed_us) {
    std::lock_guard<std::mutex> lk(mu_);
    auto idx = static_cast<size_t>(phase);
    if (idx < kNumPhases) {
        accumulators_[idx].add_sample(elapsed_us);
    }
}

void BacktestProfiler::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& a : accumulators_) a.reset();
    for (auto& t : timers_) t.reset();
}

const TimingAccumulator& BacktestProfiler::accumulator(BacktestPhase phase) const {
    auto idx = static_cast<size_t>(phase);
    return accumulators_[idx];
}

BacktestPhase BacktestProfiler::bottleneck() const {
    std::lock_guard<std::mutex> lk(mu_);
    BacktestPhase worst = BacktestPhase::DataLoad;
    double worst_mean = 0.0;
    // Only check the four sub-phases, not Total
    for (size_t i = 0; i < static_cast<size_t>(BacktestPhase::Total); ++i) {
        double m = accumulators_[i].mean();
        if (m > worst_mean) {
            worst_mean = m;
            worst = static_cast<BacktestPhase>(i);
        }
    }
    return worst;
}

double BacktestProfiler::bottleneck_fraction() const {
    std::lock_guard<std::mutex> lk(mu_);
    double total_mean = 0.0;
    double worst_mean = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(BacktestPhase::Total); ++i) {
        double m = accumulators_[i].mean();
        total_mean += m;
        if (m > worst_mean) worst_mean = m;
    }
    if (total_mean <= 0.0) return 0.0;
    return worst_mean / total_mean;
}

std::string BacktestProfiler::summary() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::ostringstream oss;
    oss << "=== Backtest Profiler ===\n";
    for (size_t i = 0; i < kNumPhases; ++i) {
        auto phase = static_cast<BacktestPhase>(i);
        const auto& acc = accumulators_[i];
        if (acc.count() == 0) continue;
        oss << "  " << std::setw(16) << std::left << phase_name(phase)
            << " n=" << acc.count()
            << " mean=" << std::fixed << std::setprecision(1) << acc.mean() << "us"
            << " p50=" << acc.p50() << "us"
            << " p95=" << acc.p95() << "us"
            << " p99=" << acc.p99() << "us"
            << " min=" << acc.min() << "us"
            << " max=" << acc.max() << "us"
            << "\n";
    }
    // Bottleneck info (only if we have data)
    bool has_data = false;
    for (size_t i = 0; i < static_cast<size_t>(BacktestPhase::Total); ++i) {
        if (accumulators_[i].count() > 0) { has_data = true; break; }
    }
    if (has_data) {
        // Find bottleneck without recursing into the public method (we already hold mu_)
        double worst_mean = 0.0;
        double total_mean = 0.0;
        BacktestPhase worst_phase = BacktestPhase::DataLoad;
        for (size_t i = 0; i < static_cast<size_t>(BacktestPhase::Total); ++i) {
            double m = accumulators_[i].mean();
            total_mean += m;
            if (m > worst_mean) {
                worst_mean = m;
                worst_phase = static_cast<BacktestPhase>(i);
            }
        }
        double frac = (total_mean > 0.0) ? (worst_mean / total_mean) : 0.0;
        oss << "  Bottleneck: " << phase_name(worst_phase)
            << " (" << std::fixed << std::setprecision(1) << (frac * 100.0) << "%)\n";
    }
    return oss.str();
}

void BacktestProfiler::ingest(const BacktestResult& result) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < kNumPhases; ++i) {
        if (result.phase_us[i] > 0.0) {
            accumulators_[i].add_sample(result.phase_us[i]);
        }
    }
}

// ===================================================================
//  ThroughputTracker
// ===================================================================

ThroughputTracker::ThroughputTracker() = default;

double ThroughputTracker::now_seconds() const {
    return steady_seconds();
}

void ThroughputTracker::record_completion(const std::string& strategy_type,
                                           const std::string& symbol,
                                           const std::string& timeframe) {
    std::lock_guard<std::mutex> lk(mu_);
    double ts = now_seconds();
    if (!started_) {
        start_time_s_ = ts;
        started_ = true;
    }
    events_.push_back(CompletionEvent{ts, strategy_type, symbol, timeframe});
    ++total_completed_;

    // Update peak throughput (1s window)
    // Count events in the last 1 second
    double cutoff = ts - 1.0;
    uint64_t count_1s = 0;
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
        if (it->timestamp_s < cutoff) break;
        ++count_1s;
    }
    double tps = static_cast<double>(count_1s);
    if (tps > peak_throughput_1s_) {
        peak_throughput_1s_ = tps;
    }
}

void ThroughputTracker::prune_old_entries() {
    std::lock_guard<std::mutex> lk(mu_);
    if (events_.empty()) return;
    // Keep events within the largest window (300s) plus a margin
    double cutoff = now_seconds() - 320.0;
    while (!events_.empty() && events_.front().timestamp_s < cutoff) {
        events_.pop_front();
    }
}

double ThroughputTracker::throughput(double window_seconds) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (events_.empty()) return 0.0;
    double now = steady_seconds();
    double cutoff = now - window_seconds;
    uint64_t count = 0;
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
        if (it->timestamp_s < cutoff) break;
        ++count;
    }
    if (window_seconds <= 0.0) return 0.0;
    return static_cast<double>(count) / window_seconds;
}

double ThroughputTracker::peak_throughput_1s() const noexcept {
    return peak_throughput_1s_;
}

double ThroughputTracker::average_throughput() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!started_ || total_completed_ == 0) return 0.0;
    double elapsed = now_seconds() - start_time_s_;
    if (elapsed <= 0.0) return 0.0;
    return static_cast<double>(total_completed_) / elapsed;
}

std::unordered_map<std::string, double>
ThroughputTracker::throughput_by_strategy(double window_seconds) const {
    std::lock_guard<std::mutex> lk(mu_);
    std::unordered_map<std::string, uint64_t> counts;
    double cutoff = steady_seconds() - window_seconds;
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
        if (it->timestamp_s < cutoff) break;
        counts[it->strategy_type]++;
    }
    std::unordered_map<std::string, double> result;
    for (const auto& [k, v] : counts) {
        result[k] = static_cast<double>(v) / window_seconds;
    }
    return result;
}

std::unordered_map<std::string, double>
ThroughputTracker::throughput_by_symbol(double window_seconds) const {
    std::lock_guard<std::mutex> lk(mu_);
    std::unordered_map<std::string, uint64_t> counts;
    double cutoff = steady_seconds() - window_seconds;
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
        if (it->timestamp_s < cutoff) break;
        counts[it->symbol]++;
    }
    std::unordered_map<std::string, double> result;
    for (const auto& [k, v] : counts) {
        result[k] = static_cast<double>(v) / window_seconds;
    }
    return result;
}

std::unordered_map<std::string, double>
ThroughputTracker::throughput_by_timeframe(double window_seconds) const {
    std::lock_guard<std::mutex> lk(mu_);
    std::unordered_map<std::string, uint64_t> counts;
    double cutoff = steady_seconds() - window_seconds;
    for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
        if (it->timestamp_s < cutoff) break;
        counts[it->timeframe]++;
    }
    std::unordered_map<std::string, double> result;
    for (const auto& [k, v] : counts) {
        result[k] = static_cast<double>(v) / window_seconds;
    }
    return result;
}

uint64_t ThroughputTracker::total_completed() const noexcept {
    return total_completed_;
}

std::string ThroughputTracker::summary() const {
    std::ostringstream oss;
    oss << "=== Throughput ===\n";
    oss << "  Total completed: " << total_completed_ << "\n";
    oss << std::fixed << std::setprecision(2);
    oss << "  1s:   " << throughput(1.0) << " bt/s\n";
    oss << "  10s:  " << throughput(10.0) << " bt/s\n";
    oss << "  60s:  " << throughput(60.0) << " bt/s\n";
    oss << "  300s: " << throughput(300.0) << " bt/s\n";
    oss << "  Peak: " << peak_throughput_1s_ << " bt/s\n";
    oss << "  Avg:  " << average_throughput() << " bt/s\n";
    return oss.str();
}

void ThroughputTracker::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    events_.clear();
    total_completed_ = 0;
    peak_throughput_1s_ = 0.0;
    start_time_s_ = 0.0;
    started_ = false;
}

// ===================================================================
//  ResourceMonitor
// ===================================================================

ResourceMonitor::ResourceMonitor() : ResourceMonitor(0) {}

ResourceMonitor::ResourceMonitor(uint32_t num_workers)
    : num_workers_(num_workers)
    , workers_(num_workers)
{
    double now = now_seconds();
    for (auto& w : workers_) {
        w.last_transition_s = now;
    }
}

double ResourceMonitor::now_seconds() const {
    return steady_seconds();
}

void ResourceMonitor::record_alloc(size_t bytes) {
    current_memory_.fetch_add(bytes, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mu_);
    size_t cur = current_memory_.load(std::memory_order_relaxed);
    if (cur > peak_memory_) peak_memory_ = cur;
}

void ResourceMonitor::record_dealloc(size_t bytes) {
    current_memory_.fetch_sub(bytes, std::memory_order_relaxed);
}

size_t ResourceMonitor::current_memory_bytes() const noexcept {
    return current_memory_.load(std::memory_order_relaxed);
}

size_t ResourceMonitor::peak_memory_bytes() const noexcept {
    return peak_memory_;
}

void ResourceMonitor::set_queue_depth(uint64_t depth) {
    std::lock_guard<std::mutex> lk(mu_);
    current_queue_depth_ = depth;
    if (depth > peak_queue_depth_) peak_queue_depth_ = depth;
    double ts = now_seconds();
    queue_history_.push_back({ts, depth});
    if (queue_history_.size() > kMaxQueueHistory) {
        queue_history_.pop_front();
    }
}

uint64_t ResourceMonitor::current_queue_depth() const noexcept {
    return current_queue_depth_;
}

uint64_t ResourceMonitor::peak_queue_depth() const noexcept {
    return peak_queue_depth_;
}

const std::deque<std::pair<double, uint64_t>>& ResourceMonitor::queue_history() const noexcept {
    return queue_history_;
}

void ResourceMonitor::worker_started_task(uint32_t worker_id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (worker_id >= num_workers_) return;
    auto& w = workers_[worker_id];
    double now = now_seconds();
    if (!w.busy) {
        double idle_duration = now - w.last_transition_s;
        w.total_idle_s += idle_duration;
        w.busy = true;
        w.busy_start_s = now;
        w.last_transition_s = now;
    }
}

void ResourceMonitor::worker_finished_task(uint32_t worker_id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (worker_id >= num_workers_) return;
    auto& w = workers_[worker_id];
    double now = now_seconds();
    if (w.busy) {
        double busy_duration = now - w.last_transition_s;
        w.total_busy_s += busy_duration;
        w.busy = false;
        w.last_transition_s = now;
        w.tasks_completed++;
    }

    // Track for ETA
    if (!has_first_completion_) {
        first_completion_time_s_ = now;
        has_first_completion_ = true;
    }
}

double ResourceMonitor::worker_utilization(uint32_t worker_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (worker_id >= num_workers_) return 0.0;
    const auto& w = workers_[worker_id];
    double now = now_seconds();
    double total_busy = w.total_busy_s;
    double total_idle = w.total_idle_s;
    // Account for current state
    double since_last = now - w.last_transition_s;
    if (w.busy) {
        total_busy += since_last;
    } else {
        total_idle += since_last;
    }
    double total = total_busy + total_idle;
    if (total <= 0.0) return 0.0;
    return total_busy / total;
}

double ResourceMonitor::average_utilization() const {
    if (num_workers_ == 0) return 0.0;
    double sum = 0.0;
    // Note: worker_utilization() locks mu_, so we can't hold it here.
    // Instead, inline the calculation under one lock.
    std::lock_guard<std::mutex> lk(mu_);
    double now = now_seconds();
    for (uint32_t i = 0; i < num_workers_; ++i) {
        const auto& w = workers_[i];
        double total_busy = w.total_busy_s;
        double total_idle = w.total_idle_s;
        double since_last = now - w.last_transition_s;
        if (w.busy) total_busy += since_last;
        else        total_idle += since_last;
        double total = total_busy + total_idle;
        if (total > 0.0) sum += total_busy / total;
    }
    return sum / static_cast<double>(num_workers_);
}

void ResourceMonitor::set_total_backtests(uint64_t total) {
    std::lock_guard<std::mutex> lk(mu_);
    total_backtests_ = total;
}

void ResourceMonitor::set_completed_backtests(uint64_t completed) {
    std::lock_guard<std::mutex> lk(mu_);
    completed_backtests_ = completed;
}

double ResourceMonitor::estimated_seconds_remaining() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!has_first_completion_ || completed_backtests_ == 0) return -1.0;
    if (completed_backtests_ >= total_backtests_) return 0.0;
    double now = now_seconds();
    double elapsed = now - first_completion_time_s_;
    if (elapsed <= 0.0) return -1.0;
    double rate = static_cast<double>(completed_backtests_) / elapsed;
    if (rate <= 0.0) return -1.0;
    double remaining = static_cast<double>(total_backtests_ - completed_backtests_);
    return remaining / rate;
}

std::string ResourceMonitor::eta_string() const {
    double eta = estimated_seconds_remaining();
    if (eta < 0.0) return "calculating...";
    return format_duration(eta);
}

uint32_t ResourceMonitor::num_workers() const noexcept {
    return num_workers_;
}

std::string ResourceMonitor::summary() const {
    std::ostringstream oss;
    oss << "=== Resources ===\n";
    oss << "  Memory: " << format_bytes(current_memory_bytes())
        << " (peak: " << format_bytes(peak_memory_bytes()) << ")\n";
    oss << "  Queue depth: " << current_queue_depth_
        << " (peak: " << peak_queue_depth_ << ")\n";
    oss << "  Workers: " << num_workers_ << "\n";
    oss << std::fixed << std::setprecision(1);
    oss << "  Avg utilization: " << (average_utilization() * 100.0) << "%\n";
    oss << "  ETA: " << eta_string() << "\n";
    return oss.str();
}

void ResourceMonitor::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    current_memory_.store(0, std::memory_order_relaxed);
    peak_memory_ = 0;
    current_queue_depth_ = 0;
    peak_queue_depth_ = 0;
    queue_history_.clear();
    total_backtests_ = 0;
    completed_backtests_ = 0;
    has_first_completion_ = false;
    double now = now_seconds();
    for (auto& w : workers_) {
        w = WorkerState{};
        w.last_transition_s = now;
    }
}

// ===================================================================
//  RunningHistogram
// ===================================================================

RunningHistogram::RunningHistogram()
    : RunningHistogram(-3.0, 5.0, 40) {}

RunningHistogram::RunningHistogram(double low, double high, uint32_t num_bins)
    : low_(low)
    , high_(high)
    , num_bins_(num_bins)
    , bin_width_((high - low) / static_cast<double>(num_bins))
    , bins_(num_bins, 0)
    , count_(0)
    , underflow_(0)
    , overflow_(0)
{}

void RunningHistogram::add(double value) {
    ++count_;
    if (value < low_) {
        ++underflow_;
        return;
    }
    if (value >= high_) {
        ++overflow_;
        return;
    }
    auto bin = static_cast<uint32_t>((value - low_) / bin_width_);
    if (bin >= num_bins_) bin = num_bins_ - 1;
    bins_[bin]++;
}

void RunningHistogram::reset() {
    count_ = 0;
    underflow_ = 0;
    overflow_ = 0;
    std::fill(bins_.begin(), bins_.end(), 0ULL);
}

uint64_t RunningHistogram::count() const noexcept { return count_; }
uint64_t RunningHistogram::underflow() const noexcept { return underflow_; }
uint64_t RunningHistogram::overflow() const noexcept { return overflow_; }

double RunningHistogram::bin_low(uint32_t bin) const noexcept {
    return low_ + static_cast<double>(bin) * bin_width_;
}

double RunningHistogram::bin_high(uint32_t bin) const noexcept {
    return low_ + static_cast<double>(bin + 1) * bin_width_;
}

double RunningHistogram::bin_center(uint32_t bin) const noexcept {
    return low_ + (static_cast<double>(bin) + 0.5) * bin_width_;
}

uint64_t RunningHistogram::bin_count(uint32_t bin) const noexcept {
    if (bin >= num_bins_) return 0;
    return bins_[bin];
}

double RunningHistogram::bin_frequency(uint32_t bin) const {
    if (count_ == 0 || bin >= num_bins_) return 0.0;
    return static_cast<double>(bins_[bin]) / static_cast<double>(count_);
}

uint32_t RunningHistogram::num_bins() const noexcept { return num_bins_; }
double RunningHistogram::low() const noexcept { return low_; }
double RunningHistogram::high() const noexcept { return high_; }

std::string RunningHistogram::render_ascii(uint32_t width) const {
    if (count_ == 0) return "(empty histogram)\n";

    uint64_t max_count = *std::max_element(bins_.begin(), bins_.end());
    if (max_count == 0) return "(all samples outside range)\n";

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    for (uint32_t i = 0; i < num_bins_; ++i) {
        double lo = bin_low(i);
        double hi = bin_high(i);
        uint64_t c = bins_[i];
        uint32_t bar_len = static_cast<uint32_t>(
            static_cast<double>(c) / static_cast<double>(max_count) * static_cast<double>(width));

        oss << "[" << std::setw(6) << lo << "," << std::setw(6) << hi << ") "
            << std::setw(6) << c << " ";
        for (uint32_t j = 0; j < bar_len; ++j) oss << '#';
        oss << "\n";
    }
    if (underflow_ > 0) oss << "  underflow: " << underflow_ << "\n";
    if (overflow_ > 0)  oss << "  overflow:  " << overflow_ << "\n";
    return oss.str();
}

// ===================================================================
//  ResultAggregator::WelfordState
// ===================================================================

void ResultAggregator::WelfordState::add(double x) noexcept {
    ++n;
    double delta = x - mean;
    mean += delta / static_cast<double>(n);
    double delta2 = x - mean;
    m2 += delta * delta2;
}

double ResultAggregator::WelfordState::variance() const noexcept {
    if (n < 2) return 0.0;
    return m2 / static_cast<double>(n - 1);
}

double ResultAggregator::WelfordState::stddev() const noexcept {
    return std::sqrt(variance());
}

// ===================================================================
//  ResultAggregator
// ===================================================================

ResultAggregator::ResultAggregator()
    : sharpe_hist_(-3.0, 5.0, 40)
{
    for (auto& c : status_counts_) c.store(0, std::memory_order_relaxed);
}

void ResultAggregator::ingest(const BacktestResult& result) {
    std::lock_guard<std::mutex> lk(mu_);
    ++total_ingested_;

    // Welford updates
    sharpe_acc_.add(result.sharpe);
    return_acc_.add(result.annualized_return);
    dd_acc_.add(result.max_drawdown);

    // Histogram
    sharpe_hist_.add(result.sharpe);

    // Top-K by Sharpe (min-heap, pop smallest when full)
    top_k_sharpe_.push(result);
    if (top_k_sharpe_.size() > kTopK) {
        top_k_sharpe_.pop();
    }

    // Worst-K by drawdown (min-heap on DD, pop smallest DD when full => keeps largest DD)
    worst_k_dd_.push(result);
    if (worst_k_dd_.size() > kTopK) {
        worst_k_dd_.pop();
    }

    // Status count
    auto si = static_cast<size_t>(result.status);
    if (si < static_cast<size_t>(BacktestStatus::COUNT)) {
        status_counts_[si].fetch_add(1, std::memory_order_relaxed);
    }
}

void ResultAggregator::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    sharpe_acc_ = WelfordState{};
    return_acc_ = WelfordState{};
    dd_acc_ = WelfordState{};
    total_ingested_ = 0;
    sharpe_hist_.reset();
    // Clear priority queues
    while (!top_k_sharpe_.empty()) top_k_sharpe_.pop();
    while (!worst_k_dd_.empty()) worst_k_dd_.pop();
    for (auto& c : status_counts_) c.store(0, std::memory_order_relaxed);
}

double ResultAggregator::sharpe_mean() const noexcept { return sharpe_acc_.mean; }
double ResultAggregator::sharpe_variance() const noexcept { return sharpe_acc_.variance(); }
double ResultAggregator::sharpe_stddev() const noexcept { return sharpe_acc_.stddev(); }

double ResultAggregator::return_mean() const noexcept { return return_acc_.mean; }
double ResultAggregator::return_variance() const noexcept { return return_acc_.variance(); }

double ResultAggregator::drawdown_mean() const noexcept { return dd_acc_.mean; }
double ResultAggregator::drawdown_variance() const noexcept { return dd_acc_.variance(); }

uint64_t ResultAggregator::total_ingested() const noexcept { return total_ingested_; }

const RunningHistogram& ResultAggregator::sharpe_histogram() const noexcept {
    return sharpe_hist_;
}

std::vector<BacktestResult> ResultAggregator::top_k_by_sharpe() const {
    std::lock_guard<std::mutex> lk(mu_);
    // Copy the heap and drain it
    auto copy = top_k_sharpe_;
    std::vector<BacktestResult> results;
    results.reserve(copy.size());
    while (!copy.empty()) {
        results.push_back(std::move(const_cast<BacktestResult&>(copy.top())));
        copy.pop();
    }
    // Sort descending by Sharpe
    std::sort(results.begin(), results.end(),
              [](const BacktestResult& a, const BacktestResult& b) {
                  return a.sharpe > b.sharpe;
              });
    return results;
}

std::vector<BacktestResult> ResultAggregator::worst_k_by_drawdown() const {
    std::lock_guard<std::mutex> lk(mu_);
    auto copy = worst_k_dd_;
    std::vector<BacktestResult> results;
    results.reserve(copy.size());
    while (!copy.empty()) {
        results.push_back(std::move(const_cast<BacktestResult&>(copy.top())));
        copy.pop();
    }
    // Sort descending by drawdown (worst first)
    std::sort(results.begin(), results.end(),
              [](const BacktestResult& a, const BacktestResult& b) {
                  return a.max_drawdown > b.max_drawdown;
              });
    return results;
}

uint64_t ResultAggregator::count_by_status(BacktestStatus status) const noexcept {
    auto si = static_cast<size_t>(status);
    if (si >= static_cast<size_t>(BacktestStatus::COUNT)) return 0;
    return status_counts_[si].load(std::memory_order_relaxed);
}

void ResultAggregator::record_status_change(BacktestStatus from, BacktestStatus to) {
    auto fi = static_cast<size_t>(from);
    auto ti = static_cast<size_t>(to);
    if (fi < static_cast<size_t>(BacktestStatus::COUNT)) {
        uint64_t cur = status_counts_[fi].load(std::memory_order_relaxed);
        if (cur > 0) status_counts_[fi].fetch_sub(1, std::memory_order_relaxed);
    }
    if (ti < static_cast<size_t>(BacktestStatus::COUNT)) {
        status_counts_[ti].fetch_add(1, std::memory_order_relaxed);
    }
}

void ResultAggregator::record_status(BacktestStatus status) {
    auto si = static_cast<size_t>(status);
    if (si < static_cast<size_t>(BacktestStatus::COUNT)) {
        status_counts_[si].fetch_add(1, std::memory_order_relaxed);
    }
}

std::string ResultAggregator::summary() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::ostringstream oss;
    oss << "=== Result Aggregator ===\n";
    oss << "  Total ingested: " << total_ingested_ << "\n";
    oss << std::fixed << std::setprecision(4);
    oss << "  Sharpe:  mean=" << sharpe_acc_.mean
        << " std=" << sharpe_acc_.stddev() << "\n";
    oss << "  Return:  mean=" << return_acc_.mean
        << " std=" << return_acc_.stddev() << "\n";
    oss << "  MaxDD:   mean=" << dd_acc_.mean
        << " std=" << dd_acc_.stddev() << "\n";
    oss << "  Status:  queued=" << status_counts_[0].load(std::memory_order_relaxed)
        << " running=" << status_counts_[1].load(std::memory_order_relaxed)
        << " completed=" << status_counts_[2].load(std::memory_order_relaxed)
        << " failed=" << status_counts_[3].load(std::memory_order_relaxed)
        << "\n";

    // Top-K preview
    auto copy = top_k_sharpe_;
    std::vector<BacktestResult> top;
    while (!copy.empty()) {
        top.push_back(copy.top());
        copy.pop();
    }
    std::sort(top.begin(), top.end(),
              [](const BacktestResult& a, const BacktestResult& b) {
                  return a.sharpe > b.sharpe;
              });
    if (!top.empty()) {
        oss << "  Best Sharpe: " << top[0].sharpe
            << " (" << top[0].strategy_name << " " << top[0].symbol << ")\n";
    }
    return oss.str();
}

// ===================================================================
//  AnomalyDetector
// ===================================================================

AnomalyDetector::AnomalyDetector() : AnomalyDetector(Config{}) {}

AnomalyDetector::AnomalyDetector(const Config& cfg) : cfg_(cfg) {}

std::vector<AnomalyAlert> AnomalyDetector::check(
    const BacktestResult& result,
    double sharpe_mean,
    double sharpe_stddev) const
{
    std::vector<AnomalyAlert> alerts;

    auto a1 = check_zscore(result, sharpe_mean, sharpe_stddev);
    if (a1) alerts.push_back(std::move(*a1));

    auto a2 = check_impossible(result);
    if (a2) alerts.push_back(std::move(*a2));

    auto a3 = check_negative_cost(result);
    if (a3) alerts.push_back(std::move(*a3));

    auto a4 = check_regime_inconsistency(result);
    if (a4) alerts.push_back(std::move(*a4));

    return alerts;
}

std::optional<AnomalyAlert> AnomalyDetector::check_zscore(
    const BacktestResult& result,
    double sharpe_mean,
    double sharpe_stddev) const
{
    if (sharpe_stddev <= 0.0) return std::nullopt;
    double z = std::abs(result.sharpe - sharpe_mean) / sharpe_stddev;
    if (z <= cfg_.z_score_threshold) return std::nullopt;

    AnomalyAlert alert;
    alert.backtest_id = result.backtest_id;
    alert.type = AnomalyType::ZScoreOutlier;
    alert.severity = std::min(1.0, z / 10.0);
    alert.value = result.sharpe;
    alert.threshold = cfg_.z_score_threshold;
    alert.timestamp_ns = steady_nanos();

    std::ostringstream oss;
    oss << "Z-score outlier: Sharpe=" << std::fixed << std::setprecision(3)
        << result.sharpe << " z=" << std::setprecision(2) << z
        << " (threshold=" << cfg_.z_score_threshold << ")"
        << " strategy=" << result.strategy_name
        << " — likely overfit or data issue";
    alert.description = oss.str();
    return alert;
}

std::optional<AnomalyAlert> AnomalyDetector::check_impossible(
    const BacktestResult& result) const
{
    if (result.sharpe <= cfg_.impossible_sharpe) return std::nullopt;
    if (result.num_trades >= cfg_.impossible_min_trades) return std::nullopt;

    AnomalyAlert alert;
    alert.backtest_id = result.backtest_id;
    alert.type = AnomalyType::ImpossiblePerformance;
    alert.severity = std::min(1.0, result.sharpe / 10.0);
    alert.value = result.sharpe;
    alert.threshold = cfg_.impossible_sharpe;
    alert.timestamp_ns = steady_nanos();

    std::ostringstream oss;
    oss << "Impossible performance: Sharpe=" << std::fixed << std::setprecision(3)
        << result.sharpe << " with only " << result.num_trades << " trades"
        << " (need >=" << cfg_.impossible_min_trades << ")"
        << " strategy=" << result.strategy_name
        << " — likely bug or insufficient data";
    alert.description = oss.str();
    return alert;
}

std::optional<AnomalyAlert> AnomalyDetector::check_negative_cost(
    const BacktestResult& result) const
{
    // If Sharpe with costs > Sharpe without costs, something is wrong
    double cost_impact = result.sharpe - result.sharpe_no_cost;
    if (cost_impact <= cfg_.negative_cost_threshold) return std::nullopt;
    // Only flag if cost_bps > 0 (if costs are zero, no issue)
    if (result.cost_bps <= 0.0) return std::nullopt;

    AnomalyAlert alert;
    alert.backtest_id = result.backtest_id;
    alert.type = AnomalyType::NegativeCost;
    alert.severity = std::min(1.0, std::abs(cost_impact));
    alert.value = cost_impact;
    alert.threshold = cfg_.negative_cost_threshold;
    alert.timestamp_ns = steady_nanos();

    std::ostringstream oss;
    oss << "Negative cost anomaly: Sharpe improves by " << std::fixed << std::setprecision(4)
        << cost_impact << " when costs=" << result.cost_bps << "bps are applied"
        << " (sharpe_no_cost=" << result.sharpe_no_cost
        << ", sharpe=" << result.sharpe << ")"
        << " strategy=" << result.strategy_name
        << " — likely data error or cost model bug";
    alert.description = oss.str();
    return alert;
}

void AnomalyDetector::record_regime_result(const std::string& strategy,
                                            const std::string& regime,
                                            double sharpe) {
    std::lock_guard<std::mutex> lk(mu_);
    regime_data_[strategy][regime].push_back(sharpe);
}

std::optional<AnomalyAlert> AnomalyDetector::check_regime_inconsistency(
    const BacktestResult& result) const
{
    std::lock_guard<std::mutex> lk(mu_);
    auto strat_it = regime_data_.find(result.strategy_name);
    if (strat_it == regime_data_.end()) return std::nullopt;

    const auto& regimes = strat_it->second;
    auto regime_it = regimes.find(result.regime_label);
    if (regime_it == regimes.end()) return std::nullopt;

    const auto& history = regime_it->second;
    if (history.size() < 3) return std::nullopt;

    // Compute mean of this strategy in this same regime
    double sum = 0.0;
    for (double s : history) sum += s;
    double regime_mean = sum / static_cast<double>(history.size());

    // Check if the current result is wildly inconsistent with prior results
    // in the SAME regime (the spec says "works in regime A but fails in A")
    double diff = std::abs(result.sharpe - regime_mean);
    if (diff <= cfg_.regime_inconsistency_threshold) return std::nullopt;

    AnomalyAlert alert;
    alert.backtest_id = result.backtest_id;
    alert.type = AnomalyType::RegimeInconsistency;
    alert.severity = std::min(1.0, diff / 5.0);
    alert.value = result.sharpe;
    alert.threshold = regime_mean;
    alert.timestamp_ns = steady_nanos();

    std::ostringstream oss;
    oss << "Regime inconsistency: strategy=" << result.strategy_name
        << " regime=" << result.regime_label
        << " sharpe=" << std::fixed << std::setprecision(3) << result.sharpe
        << " but regime mean=" << regime_mean
        << " (diff=" << std::setprecision(2) << diff
        << ", threshold=" << cfg_.regime_inconsistency_threshold << ")"
        << " — logic error or regime label mismatch";
    alert.description = oss.str();
    return alert;
}

const std::vector<AnomalyAlert>& AnomalyDetector::alert_history() const noexcept {
    return history_;
}

uint64_t AnomalyDetector::total_anomalies() const noexcept {
    return history_.size();
}

void AnomalyDetector::set_config(const Config& cfg) {
    std::lock_guard<std::mutex> lk(mu_);
    cfg_ = cfg;
}

const AnomalyDetector::Config& AnomalyDetector::config() const noexcept {
    return cfg_;
}

void AnomalyDetector::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    regime_data_.clear();
    history_.clear();
}

// ===================================================================
//  FarmDashboard
// ===================================================================

FarmDashboard::FarmDashboard() = default;

void FarmDashboard::update(const DashboardSnapshot& snap) {
    std::lock_guard<std::mutex> lk(mu_);
    snap_ = snap;
}

const DashboardSnapshot& FarmDashboard::latest() const noexcept {
    return snap_;
}

std::string FarmDashboard::render_status_line() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::ostringstream oss;
    oss << "Status: "
        << "queued=" << snap_.queued
        << " running=" << snap_.running
        << " completed=" << snap_.completed
        << " failed=" << snap_.failed
        << " total=" << (snap_.queued + snap_.running + snap_.completed + snap_.failed);
    return oss.str();
}

std::string FarmDashboard::render_throughput_line() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "Throughput: "
        << snap_.throughput_1s << "/s (1s) "
        << snap_.throughput_10s << "/s (10s) "
        << snap_.throughput_60s << "/s (60s) "
        << snap_.throughput_300s << "/s (5m) "
        << "peak=" << snap_.peak_throughput << "/s";
    return oss.str();
}

std::string FarmDashboard::render_best_result() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    if (snap_.best_sharpe > std::numeric_limits<double>::lowest() + 1.0) {
        oss << "Best: " << snap_.best_strategy
            << " sharpe=" << snap_.best_sharpe
            << " ret=" << (snap_.best_return * 100.0) << "%"
            << " dd=" << (snap_.best_drawdown * 100.0) << "%";
        if (!snap_.best_params.empty()) {
            oss << " params=" << snap_.best_params;
        }
    } else {
        oss << "Best: (no results yet)";
    }
    return oss.str();
}

std::string FarmDashboard::render_eta() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::ostringstream oss;
    oss << "ETA: " << snap_.eta_string
        << " (" << std::fixed << std::setprecision(0)
        << snap_.eta_seconds << "s)";
    return oss.str();
}

std::string FarmDashboard::render_resources() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::ostringstream oss;
    oss << "Resources: "
        << "mem=" << format_bytes(snap_.memory_bytes)
        << " peak=" << format_bytes(snap_.peak_memory_bytes)
        << " util=" << std::fixed << std::setprecision(1)
        << (snap_.avg_utilization * 100.0) << "%"
        << " qdepth=" << snap_.queue_depth;
    return oss.str();
}

std::string FarmDashboard::render_anomalies() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::ostringstream oss;
    oss << "Anomalies: " << snap_.total_anomalies << " detected";
    if (!snap_.recent_anomalies.empty()) {
        oss << "\n";
        for (const auto& a : snap_.recent_anomalies) {
            oss << "  [" << anomaly_name(a.type) << "] " << a.description << "\n";
        }
    }
    return oss.str();
}

std::string FarmDashboard::render_full() const {
    std::ostringstream oss;
    oss << "============================================================\n";
    oss << "              SRFM Backtest Farm Monitor\n";
    oss << "============================================================\n";
    oss << render_status_line() << "\n";
    oss << render_throughput_line() << "\n";
    oss << render_best_result() << "\n";
    oss << render_eta() << "\n";
    oss << render_resources() << "\n";
    oss << "------------------------------------------------------------\n";
    {
        std::lock_guard<std::mutex> lk(mu_);
        oss << "Sharpe dist: mean=" << std::fixed << std::setprecision(4)
            << snap_.sharpe_mean << " std=" << snap_.sharpe_stddev << "\n";
    }
    oss << "------------------------------------------------------------\n";
    oss << render_anomalies() << "\n";
    oss << "============================================================\n";
    return oss.str();
}

// ===================================================================
//  MetricLine
// ===================================================================

std::string MetricLine::render() const {
    std::ostringstream oss;
    oss << "# HELP " << name << " " << help << "\n";
    oss << "# TYPE " << name << " " << type << "\n";
    oss << name;
    if (!labels.empty()) {
        oss << "{";
        bool first = true;
        for (const auto& [k, v] : labels) {
            if (!first) oss << ",";
            oss << k << "=\"" << v << "\"";
            first = false;
        }
        oss << "}";
    }
    oss << " " << std::fixed << std::setprecision(6) << value << "\n";
    return oss.str();
}

// ===================================================================
//  MetricsExporter
// ===================================================================

MetricsExporter::MetricsExporter() = default;

std::string MetricsExporter::export_prometheus(const DashboardSnapshot& snap) const {
    std::ostringstream oss;

    oss << farm_backtests_total(snap.completed, snap.failed).render();
    oss << farm_backtests_running(snap.running).render();
    oss << farm_throughput_per_second(snap.throughput_1s).render();
    oss << farm_best_sharpe(snap.best_sharpe).render();
    oss << farm_queue_depth(snap.queue_depth).render();
    oss << farm_worker_utilization(snap.avg_utilization).render();
    oss << farm_anomalies_detected(snap.total_anomalies).render();
    oss << farm_memory_bytes(snap.memory_bytes).render();
    oss << farm_sharpe_mean(snap.sharpe_mean).render();
    oss << farm_sharpe_stddev(snap.sharpe_stddev).render();

    return oss.str();
}

MetricLine MetricsExporter::farm_backtests_total(uint64_t completed, uint64_t failed) const {
    MetricLine m;
    m.name = prefix_ + "_backtests_total";
    m.type = "counter";
    m.help = "Total number of completed and failed backtests";
    m.value = static_cast<double>(completed + failed);
    return m;
}

MetricLine MetricsExporter::farm_backtests_running(uint64_t running) const {
    MetricLine m;
    m.name = prefix_ + "_backtests_running";
    m.type = "gauge";
    m.help = "Number of currently running backtests";
    m.value = static_cast<double>(running);
    return m;
}

MetricLine MetricsExporter::farm_throughput_per_second(double tps) const {
    MetricLine m;
    m.name = prefix_ + "_throughput_per_second";
    m.type = "gauge";
    m.help = "Current backtest throughput in backtests per second";
    m.value = tps;
    return m;
}

MetricLine MetricsExporter::farm_best_sharpe(double sharpe) const {
    MetricLine m;
    m.name = prefix_ + "_best_sharpe";
    m.type = "gauge";
    m.help = "Best Sharpe ratio observed so far";
    m.value = sharpe;
    return m;
}

MetricLine MetricsExporter::farm_queue_depth(uint64_t depth) const {
    MetricLine m;
    m.name = prefix_ + "_queue_depth";
    m.type = "gauge";
    m.help = "Current number of backtests in queue";
    m.value = static_cast<double>(depth);
    return m;
}

MetricLine MetricsExporter::farm_worker_utilization(double util) const {
    MetricLine m;
    m.name = prefix_ + "_worker_utilization";
    m.type = "gauge";
    m.help = "Average worker utilization ratio (0-1)";
    m.value = util;
    return m;
}

MetricLine MetricsExporter::farm_anomalies_detected(uint64_t count) const {
    MetricLine m;
    m.name = prefix_ + "_anomalies_detected";
    m.type = "counter";
    m.help = "Total number of anomalies detected";
    m.value = static_cast<double>(count);
    return m;
}

MetricLine MetricsExporter::farm_memory_bytes(size_t bytes) const {
    MetricLine m;
    m.name = prefix_ + "_memory_bytes";
    m.type = "gauge";
    m.help = "Current tracked memory allocation in bytes";
    m.value = static_cast<double>(bytes);
    return m;
}

MetricLine MetricsExporter::farm_sharpe_mean(double mean) const {
    MetricLine m;
    m.name = prefix_ + "_sharpe_mean";
    m.type = "gauge";
    m.help = "Running mean of Sharpe ratios across all completed backtests";
    m.value = mean;
    return m;
}

MetricLine MetricsExporter::farm_sharpe_stddev(double sd) const {
    MetricLine m;
    m.name = prefix_ + "_sharpe_stddev";
    m.type = "gauge";
    m.help = "Running standard deviation of Sharpe ratios";
    m.value = sd;
    return m;
}

void MetricsExporter::set_prefix(const std::string& prefix) {
    prefix_ = prefix;
}

const std::string& MetricsExporter::prefix() const noexcept {
    return prefix_;
}

// ===================================================================
//  FarmMonitor
// ===================================================================

FarmMonitor::FarmMonitor() : FarmMonitor(0) {}

FarmMonitor::FarmMonitor(uint32_t num_workers)
    : resources_(num_workers)
{}

void FarmMonitor::start() {
    running_.store(true, std::memory_order_release);
}

void FarmMonitor::stop() {
    running_.store(false, std::memory_order_release);
}

void FarmMonitor::ingest_result(const BacktestResult& result) {
    // Profile
    profiler_.ingest(result);

    // Throughput
    throughput_.record_completion(result.strategy_name, result.symbol, result.timeframe);

    // Aggregation
    aggregator_.ingest(result);

    // Anomaly detection
    auto alerts = anomaly_detector_.check(
        result,
        aggregator_.sharpe_mean(),
        aggregator_.sharpe_stddev());

    if (!alerts.empty()) {
        // Record regime data for future checks
        anomaly_detector_.record_regime_result(
            result.strategy_name, result.regime_label, result.sharpe);
    }

    // Also always record regime data
    anomaly_detector_.record_regime_result(
        result.strategy_name, result.regime_label, result.sharpe);

    // Update completed count on resource monitor
    resources_.set_completed_backtests(aggregator_.total_ingested());
}

void FarmMonitor::on_backtest_queued() {
    aggregator_.record_status(BacktestStatus::Queued);
}

void FarmMonitor::on_backtest_started(uint32_t worker_id) {
    aggregator_.record_status_change(BacktestStatus::Queued, BacktestStatus::Running);
    resources_.worker_started_task(worker_id);
}

void FarmMonitor::on_backtest_finished(uint32_t worker_id, const BacktestResult& result) {
    aggregator_.record_status_change(BacktestStatus::Running, BacktestStatus::Completed);
    resources_.worker_finished_task(worker_id);
    ingest_result(result);
}

void FarmMonitor::on_backtest_failed(uint32_t worker_id, const BacktestResult& result) {
    aggregator_.record_status_change(BacktestStatus::Running, BacktestStatus::Failed);
    resources_.worker_finished_task(worker_id);

    // Still record throughput
    throughput_.record_completion(result.strategy_name, result.symbol, result.timeframe);
    resources_.set_completed_backtests(
        aggregator_.count_by_status(BacktestStatus::Completed) +
        aggregator_.count_by_status(BacktestStatus::Failed));
}

void FarmMonitor::record_alloc(size_t bytes) {
    resources_.record_alloc(bytes);
}

void FarmMonitor::record_dealloc(size_t bytes) {
    resources_.record_dealloc(bytes);
}

void FarmMonitor::set_total_backtests(uint64_t total) {
    resources_.set_total_backtests(total);
}

DashboardSnapshot FarmMonitor::snapshot() const {
    DashboardSnapshot snap;

    // Status
    snap.queued    = aggregator_.count_by_status(BacktestStatus::Queued);
    snap.running   = aggregator_.count_by_status(BacktestStatus::Running);
    snap.completed = aggregator_.count_by_status(BacktestStatus::Completed);
    snap.failed    = aggregator_.count_by_status(BacktestStatus::Failed);

    // Throughput
    snap.throughput_1s   = throughput_.throughput(1.0);
    snap.throughput_10s  = throughput_.throughput(10.0);
    snap.throughput_60s  = throughput_.throughput(60.0);
    snap.throughput_300s = throughput_.throughput(300.0);
    snap.peak_throughput = throughput_.peak_throughput_1s();

    // Best result
    auto top = aggregator_.top_k_by_sharpe();
    if (!top.empty()) {
        snap.best_strategy = top[0].strategy_name;
        snap.best_params   = top[0].params_json;
        snap.best_sharpe   = top[0].sharpe;
        snap.best_return   = top[0].annualized_return;
        snap.best_drawdown = top[0].max_drawdown;
    }

    // ETA
    snap.eta_seconds = resources_.estimated_seconds_remaining();
    snap.eta_string  = resources_.eta_string();

    // Resources
    snap.memory_bytes      = resources_.current_memory_bytes();
    snap.peak_memory_bytes = resources_.peak_memory_bytes();
    snap.avg_utilization   = resources_.average_utilization();
    snap.queue_depth       = resources_.current_queue_depth();

    // Anomalies
    snap.total_anomalies = anomaly_detector_.total_anomalies();
    const auto& hist = anomaly_detector_.alert_history();
    size_t start_idx = hist.size() > 10 ? hist.size() - 10 : 0;
    for (size_t i = start_idx; i < hist.size(); ++i) {
        snap.recent_anomalies.push_back(hist[i]);
    }

    // Sharpe stats
    snap.sharpe_mean   = aggregator_.sharpe_mean();
    snap.sharpe_stddev = aggregator_.sharpe_stddev();

    return snap;
}

std::string FarmMonitor::render_dashboard() const {
    auto snap = snapshot();
    FarmDashboard dash;
    dash.update(snap);
    return dash.render_full();
}

std::string FarmMonitor::export_prometheus() const {
    auto snap = snapshot();
    return exporter_.export_prometheus(snap);
}

const BacktestProfiler& FarmMonitor::profiler() const noexcept { return profiler_; }
const ThroughputTracker& FarmMonitor::throughput() const noexcept { return throughput_; }
const ResourceMonitor& FarmMonitor::resources() const noexcept { return resources_; }
const ResultAggregator& FarmMonitor::aggregator() const noexcept { return aggregator_; }
const AnomalyDetector& FarmMonitor::anomalies() const noexcept { return anomaly_detector_; }
const FarmDashboard& FarmMonitor::dashboard() const noexcept { return dashboard_; }
const MetricsExporter& FarmMonitor::exporter() const noexcept { return exporter_; }

void FarmMonitor::reset() {
    profiler_.reset();
    throughput_.reset();
    resources_.reset();
    aggregator_.reset();
    anomaly_detector_.reset();
}

} // namespace srfm::farm
