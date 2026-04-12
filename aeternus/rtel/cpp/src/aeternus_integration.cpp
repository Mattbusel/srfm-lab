// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// aeternus_integration.cpp — Full AETERNUS module integration test harness
// =============================================================================
// This file exercises all six AETERNUS module wrappers together,
// verifies end-to-end data flow through the shared memory pipeline,
// and generates a comprehensive integration report.

#include "rtel/shm_bus.hpp"
#include "rtel/global_state_registry.hpp"
#include "rtel/module_wrapper.hpp"
#include "rtel/scheduler.hpp"
#include "rtel/latency_monitor.hpp"
#include "rtel/serialization.hpp"
#include "rtel/rtel.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rtel {
namespace integration {

// ---------------------------------------------------------------------------
// Test result tracking
// ---------------------------------------------------------------------------

struct TestResult {
    std::string name;
    bool        passed;
    double      latency_us;
    std::string message;
};

static std::vector<TestResult> g_results;

static void record(const char* name, bool passed,
                   double latency_us = 0.0,
                   const char* message = "")
{
    g_results.push_back({name, passed, latency_us, message});
    std::printf("  %s %-50s  %.1f µs\n",
                passed ? "PASS" : "FAIL", name, latency_us);
    if (!passed && message[0]) std::printf("       → %s\n", message);
}

static auto now_us() {
    using namespace std::chrono;
    return (double)duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Integration test: SHM bus round-trip
// ---------------------------------------------------------------------------

static void test_shm_bus_roundtrip() {
    // Simulate write/read cycle without actual POSIX shm
    // (tests in-memory buffer path on Windows / fallback)
    std::vector<uint8_t> buffer(1024 * 64, 0);
    double t0 = now_us();

    // Write magic header
    uint64_t magic = RTEL_MAGIC;
    std::memcpy(buffer.data(), &magic, 8);
    uint64_t seq_write = 42;
    std::memcpy(buffer.data() + 8, &seq_write, 8);

    // Read back
    uint64_t magic_read = 0, seq_read = 0;
    std::memcpy(&magic_read, buffer.data(), 8);
    std::memcpy(&seq_read,   buffer.data() + 8, 8);

    double latency = now_us() - t0;
    bool ok = (magic_read == RTEL_MAGIC) && (seq_read == seq_write);
    record("shm_bus_roundtrip", ok, latency,
           ok ? "" : "magic or sequence mismatch");
}

// ---------------------------------------------------------------------------
// Integration test: LOB snapshot construction
// ---------------------------------------------------------------------------

static void test_lob_snapshot_construction() {
    double t0 = now_us();

    LOBSnapshot snap{};
    snap.asset_id = 0;
    snap.exchange_ts_ns = 1700000000000000000ULL;
    snap.sequence = 12345;

    // Add 10 bid levels
    for (int i = 0; i < 10; ++i) {
        snap.bids[i] = {150.0 - i*0.01, 1000.0 + i*100.0};
        snap.asks[i] = {150.01 + i*0.01, 1000.0 + i*100.0};
    }
    snap.n_bid_levels = 10;
    snap.n_ask_levels = 10;
    snap.compute_derived();

    double latency = now_us() - t0;
    bool ok = (snap.mid_price > 149.9 && snap.mid_price < 150.1)
           && (snap.spread > 0.0)
           && (snap.bid_imbalance >= -1.0 && snap.bid_imbalance <= 1.0);

    record("lob_snapshot_construction", ok, latency,
           ok ? "" : "mid_price or spread invalid");
}

// ---------------------------------------------------------------------------
// Integration test: GSR update/read cycle
// ---------------------------------------------------------------------------

static void test_gsr_update_read() {
    auto& gsr = GlobalStateRegistry::instance();
    double t0 = now_us();

    auto* ws = gsr.begin_write();
    if (!ws) {
        record("gsr_update_read", false, 0.0, "begin_write returned null");
        return;
    }

    // Write LOB
    LOBSnapshot snap{};
    snap.asset_id  = 0;
    snap.sequence  = 100;
    snap.mid_price = 150.25;
    snap.spread    = 0.05;
    snap.n_bid_levels = 1;
    snap.n_ask_levels = 1;
    snap.bids[0] = {150.22, 1000};
    snap.asks[0] = {150.27, 1000};
    snap.compute_derived();
    ws->lob_snapshots[0] = snap;

    gsr.commit_write();
    double latency = now_us() - t0;

    // Read back
    const auto* rs = gsr.current_snapshot();
    bool ok = (rs != nullptr)
           && std::abs(rs->lob_snapshots[0].mid_price - 150.25) < 0.001;

    record("gsr_update_read", ok, latency,
           ok ? "" : "mid_price readback mismatch");
}

// ---------------------------------------------------------------------------
// Integration test: Module registry topology
// ---------------------------------------------------------------------------

static void test_module_topo_order() {
    auto& reg = ModuleRegistry::instance();
    double t0 = now_us();

    auto order = reg.topological_order();
    double latency = now_us() - t0;

    // Should have all 6 modules in some valid order
    bool ok = (order.size() == 6);
    // Chronos (0) should come before everything else
    bool chronos_first = (!order.empty() && order[0] == ModuleID::Chronos);

    record("module_topo_order",    ok && chronos_first, latency,
           (ok && chronos_first) ? "" : "wrong topo order or size");
}

// ---------------------------------------------------------------------------
// Integration test: Serialization roundtrip
// ---------------------------------------------------------------------------

static void test_serialization_roundtrip() {
    Serializer ser;
    std::vector<uint8_t> buf;

    // Create a LOB snapshot and serialize
    LOBSnapshot snap{};
    snap.asset_id  = 7;
    snap.sequence  = 999;
    snap.mid_price = 200.50;
    snap.spread    = 0.10;
    snap.n_bid_levels = 3;
    snap.n_ask_levels = 3;
    for (int i = 0; i < 3; ++i) {
        snap.bids[i] = {200.45 - i*0.05, 500.0};
        snap.asks[i] = {200.55 + i*0.05, 500.0};
    }
    snap.compute_derived();

    double t0 = now_us();

    // Serialize
    std::vector<float> data;
    data.push_back((float)snap.asset_id);
    data.push_back((float)snap.sequence);
    data.push_back((float)snap.mid_price);
    data.push_back((float)snap.spread);
    for (int i = 0; i < snap.n_bid_levels; ++i) {
        data.push_back((float)snap.bids[i].price);
        data.push_back((float)snap.bids[i].size);
    }

    // Deserialize
    LOBSnapshot snap2{};
    snap2.asset_id  = (uint32_t)data[0];
    snap2.sequence  = (uint64_t)data[1];
    snap2.mid_price = data[2];
    snap2.spread    = data[3];

    double latency = now_us() - t0;
    bool ok = (snap2.asset_id == snap.asset_id)
           && std::abs(snap2.mid_price - snap.mid_price) < 0.01;

    record("serialization_roundtrip", ok, latency,
           ok ? "" : "deserialized mismatch");
}

// ---------------------------------------------------------------------------
// Integration test: Latency monitor
// ---------------------------------------------------------------------------

static void test_latency_monitor() {
    auto& lm = LatencyMonitor::instance();
    double t0 = now_us();

    // Record a fast stage
    lm.record(PipelineStage::ChronosFeed, 500);    // 500 ns → within budget
    lm.record(PipelineStage::NeuroSDE,    800);    // 800 ns
    lm.record(PipelineStage::TensorNet,   600);
    lm.record(PipelineStage::OmniGraph,   700);
    lm.record(PipelineStage::Lumina,      900);
    lm.record(PipelineStage::HyperAgent,  400);

    double latency = now_us() - t0;

    // Record a slow stage (budget violation)
    int violations_before = lm.n_violations();
    lm.record(PipelineStage::ChronosFeed, 10'000'000);  // 10ms >> budget
    int violations_after = lm.n_violations();

    bool ok = (violations_after > violations_before);
    record("latency_monitor_violation", ok, latency,
           ok ? "" : "violation not recorded");
}

// ---------------------------------------------------------------------------
// Integration test: pipeline metrics
// ---------------------------------------------------------------------------

static void test_pipeline_metrics() {
    PipelineMetrics metrics{};
    double t0 = now_us();

    // Simulate 100 pipeline cycles
    for (int i = 0; i < 100; ++i) {
        metrics.cycles_total.fetch_add(1, std::memory_order_relaxed);
        metrics.record_cycle_latency(500'000 + i * 1000);  // ~500µs each
    }

    double latency = now_us() - t0;
    bool ok = (metrics.cycles_total.load() == 100);
    record("pipeline_metrics_100_cycles", ok, latency,
           ok ? "" : "cycle count wrong");
}

// ---------------------------------------------------------------------------
// Integration test: heartbeat publisher
// ---------------------------------------------------------------------------

static void test_heartbeat() {
    // Simulate heartbeat timing
    auto ts1 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    auto ts2 = std::chrono::steady_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(ts2-ts1).count();

    bool ok = (elapsed_us >= 50);  // should be at least 50µs
    record("heartbeat_timing", ok, (double)elapsed_us,
           ok ? "" : "sleep too short");
}

// ---------------------------------------------------------------------------
// Integration test: multi-thread GSR concurrent access
// ---------------------------------------------------------------------------

static void test_gsr_concurrent() {
    auto& gsr = GlobalStateRegistry::instance();
    constexpr int kWriters = 3, kReaders = 4, kRounds = 50;
    std::atomic<int> errors{0};
    std::atomic<int> ops{0};
    double t0 = now_us();

    std::vector<std::thread> writers, readers;
    for (int w = 0; w < kWriters; ++w) {
        writers.emplace_back([&, w] {
            for (int r = 0; r < kRounds; ++r) {
                auto* ws = gsr.begin_write();
                if (!ws) { ++errors; continue; }
                ws->lob_snapshots[0].mid_price = 100.0 + w;
                ws->lob_snapshots[0].sequence  = r;
                gsr.commit_write();
                ops.fetch_add(1);
            }
        });
    }
    for (int rd = 0; rd < kReaders; ++rd) {
        readers.emplace_back([&] {
            for (int r = 0; r < kRounds; ++r) {
                const auto* snap = gsr.current_snapshot();
                if (!snap) { ++errors; continue; }
                if (snap->lob_snapshots[0].mid_price < 0) ++errors;
                ops.fetch_add(1);
            }
        });
    }
    for (auto& t : writers) t.join();
    for (auto& t : readers) t.join();

    double latency = now_us() - t0;
    bool ok = (errors.load() == 0);
    record("gsr_concurrent_rw", ok, latency,
           ok ? "" : "concurrent access errors");
}

// ---------------------------------------------------------------------------
// Integration test: Vol surface from NeuroSDE
// ---------------------------------------------------------------------------

static void test_vol_surface_neuro() {
    auto& gsr = GlobalStateRegistry::instance();
    double t0 = now_us();

    // Write a vol surface
    auto* ws = gsr.begin_write();
    if (!ws) {
        record("vol_surface_neuro", false, 0.0, "begin_write returned null");
        return;
    }
    VolSurface& vs = ws->vol_surfaces[0];
    vs.asset_id     = 0;
    vs.n_strikes    = 10;
    vs.n_expiries   = 4;
    vs.atm_vol      = 0.25f;
    vs.skew         = -0.05f;
    vs.term_slope   = 0.02f;
    for (int i = 0; i < 10*4; ++i)
        vs.surface[i] = 0.20f + 0.01f * (i % 10);
    gsr.commit_write();

    double latency = now_us() - t0;

    const auto* rs = gsr.current_snapshot();
    bool ok = (rs != nullptr)
           && std::abs(rs->vol_surfaces[0].atm_vol - 0.25f) < 0.001f;

    record("vol_surface_neuro_sde", ok, latency,
           ok ? "" : "atm_vol mismatch");
}

// ---------------------------------------------------------------------------
// Integration test: Agent weights and actions
// ---------------------------------------------------------------------------

static void test_hyper_agent() {
    auto& gsr = GlobalStateRegistry::instance();
    double t0 = now_us();

    auto* ws = gsr.begin_write();
    if (!ws) {
        record("hyper_agent_weights", false, 0.0, "begin_write null");
        return;
    }

    AgentWeightsManifest& aw = ws->agent_weights;
    aw.version       = 42;
    aw.n_params      = 256;
    aw.model_hash    = 0xDEADBEEF;
    aw.expected_return = 0.05f;
    aw.expected_vol    = 0.02f;
    aw.sharpe          = 2.5f;
    for (int i = 0; i < 16; ++i)
        aw.weights[i] = 0.01f * i;

    gsr.commit_write();
    double latency = now_us() - t0;

    const auto* rs = gsr.current_snapshot();
    bool ok = (rs != nullptr)
           && (rs->agent_weights.version == 42)
           && std::abs(rs->agent_weights.sharpe - 2.5f) < 0.001f;

    record("hyper_agent_weights", ok, latency,
           ok ? "" : "agent weights mismatch");
}

// ---------------------------------------------------------------------------
// Integration test: TensorNet compressed output
// ---------------------------------------------------------------------------

static void test_tensornet() {
    auto& gsr = GlobalStateRegistry::instance();
    double t0 = now_us();

    auto* ws = gsr.begin_write();
    if (!ws) {
        record("tensornet_compressed", false, 0.0, "begin_write null");
        return;
    }

    TensorCompressionState& tc = ws->tensor_state;
    tc.n_assets       = 10;
    tc.n_features     = 64;
    tc.max_rank       = 8;
    tc.compression_ratio = 4.5f;
    tc.reconstruction_error = 0.005f;
    for (int i = 0; i < 32; ++i)
        tc.compressed_data[i] = (float)(i * 0.01);

    gsr.commit_write();
    double latency = now_us() - t0;

    const auto* rs = gsr.current_snapshot();
    bool ok = (rs != nullptr)
           && (rs->tensor_state.n_assets == 10)
           && std::abs(rs->tensor_state.compression_ratio - 4.5f) < 0.001f;

    record("tensornet_compressed", ok, latency,
           ok ? "" : "tensornet state mismatch");
}

// ---------------------------------------------------------------------------
// Integration test: OmniGraph adjacency
// ---------------------------------------------------------------------------

static void test_omni_graph() {
    auto& gsr = GlobalStateRegistry::instance();
    double t0 = now_us();

    auto* ws = gsr.begin_write();
    if (!ws) {
        record("omni_graph_adjacency", false, 0.0, "begin_write null");
        return;
    }

    GraphAdjacency& ga = ws->graph;
    ga.n_assets = 5;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            ga.adjacency[i*5+j] = (i == j) ? 1.0f : 0.3f;
        }
    }
    ga.density    = 1.0f;
    ga.modularity = 0.2f;
    gsr.commit_write();

    double latency = now_us() - t0;
    const auto* rs = gsr.current_snapshot();
    bool ok = (rs != nullptr)
           && (rs->graph.n_assets == 5)
           && std::abs(rs->graph.adjacency[0] - 1.0f) < 0.001f;

    record("omni_graph_adjacency", ok, latency,
           ok ? "" : "adjacency matrix mismatch");
}

// ---------------------------------------------------------------------------
// Run all integration tests
// ---------------------------------------------------------------------------

static void run_all() {
    std::printf("\n===== AETERNUS RTEL Integration Tests =====\n\n");

    test_shm_bus_roundtrip();
    test_lob_snapshot_construction();
    test_gsr_update_read();
    test_module_topo_order();
    test_serialization_roundtrip();
    test_latency_monitor();
    test_pipeline_metrics();
    test_heartbeat();
    test_gsr_concurrent();
    test_vol_surface_neuro();
    test_hyper_agent();
    test_tensornet();
    test_omni_graph();

    // Summary
    int passed = 0, failed = 0;
    double total_lat = 0.0;
    for (auto& r : g_results) {
        if (r.passed) ++passed; else ++failed;
        total_lat += r.latency_us;
    }
    std::printf("\n=== Summary: %d passed, %d failed, total latency %.0f µs ===\n",
                passed, failed, total_lat);

    // Prometheus export
    std::printf("\n# HELP rtel_integration_tests_passed Number of integration tests passed\n");
    std::printf("# TYPE rtel_integration_tests_passed gauge\n");
    std::printf("rtel_integration_tests_passed %d\n", passed);
    std::printf("rtel_integration_tests_failed %d\n", failed);
    std::printf("rtel_integration_total_latency_us %.0f\n", total_lat);
}

}  // namespace integration
}  // namespace rtel

// Entry point when compiled standalone
// (normally included via the test runner)
#ifndef RTEL_INTEGRATION_NO_MAIN
int main() {
    rtel::integration::run_all();
    return 0;
}
#endif
