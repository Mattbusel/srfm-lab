// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// latency_monitor.cpp — Latency Monitor Implementation
// =============================================================================

#include "rtel/latency_monitor.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <sstream>

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// LatencyMonitor — singleton
// ---------------------------------------------------------------------------
LatencyMonitor& LatencyMonitor::instance() {
    static LatencyMonitor m;
    return m;
}

LatencyMonitor::LatencyMonitor() {
    // Initialize budgets from constants
    for (std::size_t i = 0; i < static_cast<std::size_t>(PipelineStage::COUNT); ++i) {
        if (i < sizeof(kStageBudgets) / sizeof(kStageBudgets[0])) {
            budgets_[i] = kStageBudgets[i];
        } else {
            budgets_[i] = 1'000'000;  // 1ms default
        }
        violation_counts_[i].store(0, std::memory_order_relaxed);
    }
}

void LatencyMonitor::set_budget(PipelineStage stage, uint64_t ns) noexcept {
    budgets_[static_cast<std::size_t>(stage)] = ns;
}

void LatencyMonitor::record(const StageLatency& lat) {
    std::size_t idx = static_cast<std::size_t>(lat.stage);
    if (idx >= static_cast<std::size_t>(PipelineStage::COUNT)) return;

    uint64_t budget = budgets_[idx];
    bool over = lat.duration_ns > budget;

    windows_[idx].record(lat.duration_ns);
    histograms_[idx].record(lat.duration_ns);

    if (over) {
        violation_counts_[idx].fetch_add(1, std::memory_order_relaxed);
        if (violation_cb_) {
            SLAViolation v{};
            v.stage        = lat.stage;
            v.duration_ns  = lat.duration_ns;
            v.budget_ns    = budget;
            v.overage_ns   = lat.duration_ns - budget;
            v.timestamp_ns = lat.end_ns;
            v.pipeline_id  = lat.pipeline_id;
            v.message      = std::string("Stage ") +
                             stage_name(lat.stage) +
                             " exceeded budget by " +
                             std::to_string(v.overage_ns / 1000) + "µs";
            violation_cb_(v);
        }
    }
}

LatencyMonitor::StageStats
LatencyMonitor::get_stage_stats(PipelineStage stage) const {
    std::size_t idx = static_cast<std::size_t>(stage);
    if (idx >= static_cast<std::size_t>(PipelineStage::COUNT)) return {};

    StageStats s{};
    s.count      = windows_[idx].count();
    s.p50_ns     = windows_[idx].p50();
    s.p95_ns     = windows_[idx].p95();
    s.p99_ns     = windows_[idx].p99();
    s.mean_ns    = histograms_[idx].mean_cycles();  // NB: already in ns (recorded as ns)
    s.budget_ns  = budgets_[idx];
    s.violations = violation_counts_[idx].load(std::memory_order_relaxed);
    if (s.count > 0) {
        s.violation_rate_pct = 100.0 * s.violations / s.count;
    }
    return s;
}

std::string LatencyMonitor::prometheus_export() const {
    std::ostringstream oss;
    oss << "# HELP rtel_stage_latency_ns Pipeline stage latency in nanoseconds\n";
    oss << "# TYPE rtel_stage_latency_ns summary\n";

    for (std::size_t i = 0; i < static_cast<std::size_t>(PipelineStage::COUNT); ++i) {
        PipelineStage stage = static_cast<PipelineStage>(i);
        const char* sn = stage_name(stage);
        auto s = get_stage_stats(stage);
        if (s.count == 0) continue;

        oss << "rtel_stage_latency_ns{stage=\"" << sn << "\",quantile=\"0.5\"} "
            << s.p50_ns << "\n";
        oss << "rtel_stage_latency_ns{stage=\"" << sn << "\",quantile=\"0.95\"} "
            << s.p95_ns << "\n";
        oss << "rtel_stage_latency_ns{stage=\"" << sn << "\",quantile=\"0.99\"} "
            << s.p99_ns << "\n";
        oss << "rtel_stage_latency_ns_count{stage=\"" << sn << "\"} "
            << s.count << "\n";
        oss << "rtel_stage_latency_ns_sum{stage=\"" << sn << "\"} "
            << static_cast<uint64_t>(s.mean_ns * s.count) << "\n";
    }

    oss << "\n# HELP rtel_sla_violations_total SLA violations per stage\n";
    oss << "# TYPE rtel_sla_violations_total counter\n";
    for (std::size_t i = 0; i < static_cast<std::size_t>(PipelineStage::COUNT); ++i) {
        PipelineStage stage = static_cast<PipelineStage>(i);
        auto s = get_stage_stats(stage);
        if (s.count == 0) continue;
        oss << "rtel_sla_violations_total{stage=\"" << stage_name(stage) << "\"} "
            << s.violations << "\n";
    }

    oss << "\n# HELP rtel_stage_budget_ns Configured budget per stage\n";
    oss << "# TYPE rtel_stage_budget_ns gauge\n";
    for (std::size_t i = 0; i < static_cast<std::size_t>(PipelineStage::COUNT); ++i) {
        PipelineStage stage = static_cast<PipelineStage>(i);
        oss << "rtel_stage_budget_ns{stage=\"" << stage_name(stage) << "\"} "
            << budgets_[i] << "\n";
    }

    return oss.str();
}

void LatencyMonitor::print_stats() const {
    std::printf("%-20s %10s %8s %8s %8s %10s %8s %s\n",
                "Stage", "Count", "p50µs", "p95µs", "p99µs", "Budget µs",
                "Viols", "ViolRate%");
    std::printf("%s\n", std::string(90, '-').c_str());
    for (std::size_t i = 0; i < static_cast<std::size_t>(PipelineStage::COUNT); ++i) {
        PipelineStage stage = static_cast<PipelineStage>(i);
        auto s = get_stage_stats(stage);
        if (s.count == 0) continue;
        std::printf("%-20s %10lu %8.1f %8.1f %8.1f %10.1f %8lu %6.2f%%\n",
                    stage_name(stage),
                    s.count,
                    s.p50_ns / 1000.0,
                    s.p95_ns / 1000.0,
                    s.p99_ns / 1000.0,
                    s.budget_ns / 1000.0,
                    s.violations,
                    s.violation_rate_pct);
    }
}

void LatencyMonitor::reset() {
    for (std::size_t i = 0; i < static_cast<std::size_t>(PipelineStage::COUNT); ++i) {
        violation_counts_[i].store(0, std::memory_order_relaxed);
        histograms_[i].reset();
    }
}

} // namespace aeternus::rtel
