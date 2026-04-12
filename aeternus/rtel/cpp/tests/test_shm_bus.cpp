// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// test_shm_bus.cpp — Tests for ShmBus, RingBuffer, GSR
// =============================================================================

#include "rtel/shm_bus.hpp"
#include "rtel/ring_buffer.hpp"
#include "rtel/global_state_registry.hpp"
#include "rtel/module_wrapper.hpp"
#include "rtel/scheduler.hpp"
#include "rtel/latency_monitor.hpp"
#include "rtel/serialization.hpp"

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <chrono>

namespace rtel = aeternus::rtel;

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------
static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(cond) do {                                          \
    if (!(cond)) {                                                       \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_fail;                                                        \
    } else { ++g_pass; }                                                 \
} while(0)

#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))
#define ASSERT_NE(a, b) ASSERT_TRUE((a) != (b))
#define ASSERT_GT(a, b) ASSERT_TRUE((a) > (b))
#define ASSERT_GE(a, b) ASSERT_TRUE((a) >= (b))

#define TEST(name) static void test_##name()
#define RUN_TEST(name) do { \
    std::printf("[ RUN ] %s\n", #name); \
    test_##name(); \
    std::printf("[ %s ] %s\n", g_fail == 0 ? "OK " : "FAIL", #name); \
} while(0)

// ---------------------------------------------------------------------------
// TEST: ShmBus channel registration and basic write/read
// ---------------------------------------------------------------------------
TEST(shm_bus_register_and_write) {
    auto& bus = rtel::ShmBus::instance();
    bus.shutdown();

    bool ok = bus.register_channel({
        "test.channel", 4096, 16, true, false
    });
    ASSERT_TRUE(ok);

    auto* ch = bus.channel("test.channel");
    ASSERT_NE(ch, nullptr);
    ASSERT_TRUE(ch->is_open());
}

TEST(shm_bus_write_read_roundtrip) {
    auto& bus = rtel::ShmBus::instance();
    bus.register_channel({"test.rw", 4096, 16, true, false});

    auto* ch = bus.channel("test.rw");
    ASSERT_NE(ch, nullptr);

    // Write
    auto [handle, err] = ch->claim(false);
    ASSERT_EQ(static_cast<int>(err), 0);
    ASSERT_TRUE(handle.valid());

    const char* msg = "hello rtel";
    std::memcpy(handle.data(), msg, std::strlen(msg) + 1);

    rtel::TensorDescriptor td{};
    td.dtype = rtel::DType::UINT8;
    td.ndim  = 1;
    td.shape[0] = std::strlen(msg) + 1;
    td.payload_bytes = td.shape[0];
    handle.set_tensor(td);
    handle.publish();

    // Read
    rtel::ReadCursor cur;
    rtel::TensorDescriptor td_out{};
    const void* data = ch->read_data(cur, &td_out);
    ASSERT_NE(data, nullptr);
    ASSERT_EQ(td_out.payload_bytes, td.payload_bytes);
    ASSERT_EQ(0, std::strcmp(static_cast<const char*>(data), msg));

    std::printf("  Read: '%s'\n", static_cast<const char*>(data));
}

TEST(shm_bus_backpressure) {
    auto& bus = rtel::ShmBus::instance();
    bus.register_channel({"test.bp", 1024, 4, true, false});  // 4-slot ring

    auto* ch = bus.channel("test.bp");
    ASSERT_NE(ch, nullptr);

    // Fill the ring
    int full_count = 0;
    for (int i = 0; i < 8; ++i) {
        auto [handle, err] = ch->claim(false);
        if (err == rtel::ShmChannel::Error::RING_FULL) {
            ++full_count;
        } else {
            handle.publish();
        }
    }
    // We expect some backpressure
    ASSERT_GT(full_count, 0);
    std::printf("  Backpressure triggered %d times\n", full_count);
}

// ---------------------------------------------------------------------------
// TEST: RingBuffer SPMC
// ---------------------------------------------------------------------------
TEST(ring_buffer_spmc_basic) {
    rtel::RingBuffer<int> rb(64);
    ASSERT_EQ(rb.capacity(), 64u);

    // Publish 10 items
    for (int i = 0; i < 10; ++i) {
        bool ok = rb.try_publish(i);
        ASSERT_TRUE(ok);
    }
    ASSERT_EQ(rb.published_total(), 10u);

    // Consumer 1
    uint64_t c1 = rb.new_consumer_cursor();
    for (int i = 0; i < 10; ++i) {
        auto v = rb.try_consume(c1);
        ASSERT_TRUE(v.has_value());
        ASSERT_EQ(*v, i);
    }
    // No more data
    auto v = rb.try_consume(c1);
    ASSERT_TRUE(!v.has_value());

    std::printf("  Ring buffer SPMC basic: OK\n");
}

TEST(ring_buffer_multi_consumer) {
    rtel::RingBuffer<int> rb(256);

    // Two independent consumers
    uint64_t c1 = rb.new_consumer_cursor();
    uint64_t c2 = rb.new_consumer_cursor();

    for (int i = 0; i < 50; ++i) rb.try_publish(i);

    int sum1 = 0, sum2 = 0;
    for (int i = 0; i < 50; ++i) {
        auto v1 = rb.try_consume(c1);
        auto v2 = rb.try_consume(c2);
        if (v1) sum1 += *v1;
        if (v2) sum2 += *v2;
    }
    ASSERT_EQ(sum1, sum2);  // both consumers see the same data
    std::printf("  Multi-consumer sum: %d == %d\n", sum1, sum2);
}

TEST(ring_buffer_throughput) {
    rtel::RingBuffer<rtel::MarketEvent> rb(4096);
    constexpr int N = 100000;

    uint64_t t0 = rtel::now_ns();
    for (int i = 0; i < N; ++i) {
        rtel::MarketEvent ev{};
        ev.asset_id = 0;
        ev.price    = 100.0 + i * 0.01;
        while (!rb.try_publish(ev)) std::this_thread::yield();
    }
    uint64_t t1 = rtel::now_ns();

    double elapsed_s = (t1 - t0) / 1e9;
    double mpps = N / elapsed_s / 1e6;
    std::printf("  Ring buffer throughput: %.2f Mops/s (%.3fs for %d ops)\n",
                mpps, elapsed_s, N);
    ASSERT_GT(mpps, 0.1);
}

// ---------------------------------------------------------------------------
// TEST: LatencyHistogram
// ---------------------------------------------------------------------------
TEST(latency_histogram) {
    rtel::LatencyHistogram hist;
    for (uint64_t i = 1; i <= 10000; ++i) hist.record(i);
    ASSERT_EQ(hist.count(), 10000u);
    ASSERT_GT(hist.p50(), 0u);
    ASSERT_GT(hist.p95(), hist.p50());
    ASSERT_GT(hist.p99(), hist.p95());
    std::printf("  Histogram p50=%lu p95=%lu p99=%lu mean=%.1f\n",
                hist.p50(), hist.p95(), hist.p99(), hist.mean_cycles());
}

// ---------------------------------------------------------------------------
// TEST: GlobalStateRegistry reads and writes
// ---------------------------------------------------------------------------
TEST(gsr_lob_update_and_read) {
    auto& gsr = rtel::GlobalStateRegistry::instance();

    rtel::LOBSnapshot snap{};
    snap.asset_id     = 0;
    snap.n_bid_levels = 5;
    snap.n_ask_levels = 5;
    for (int i = 0; i < 5; ++i) {
        snap.bids[i].price = 150.0 - (i+1) * 0.01;
        snap.bids[i].size  = 100.0;
        snap.asks[i].price = 150.0 + (i+1) * 0.01;
        snap.asks[i].size  = 100.0;
    }
    snap.compute_derived();
    gsr.update_lob(0, snap);

    rtel::LOBSnapshot out{};
    bool ok = gsr.read_lob(0, out);
    ASSERT_TRUE(ok);
    ASSERT_GT(out.mid_price, 149.0);
    ASSERT_LT(out.mid_price, 151.0);
    std::printf("  GSR LOB mid=%.4f spread=%.4f imbalance=%.3f\n",
                out.mid_price, out.spread, out.bid_imbalance);
}

TEST(gsr_version_monotonic) {
    auto& gsr = rtel::GlobalStateRegistry::instance();
    uint64_t v0 = gsr.version();

    rtel::LOBSnapshot snap{};
    snap.asset_id = 0;
    gsr.update_lob(0, snap);

    uint64_t v1 = gsr.version();
    ASSERT_GT(v1, v0);
    std::printf("  GSR version: %lu → %lu\n", v0, v1);
}

TEST(gsr_concurrent_reads) {
    auto& gsr = rtel::GlobalStateRegistry::instance();
    constexpr int N_READERS = 4;
    constexpr int N_OPS     = 10000;

    std::atomic<int> errors{0};
    std::vector<std::thread> readers;
    for (int r = 0; r < N_READERS; ++r) {
        readers.emplace_back([&]() {
            rtel::LOBSnapshot snap{};
            for (int i = 0; i < N_OPS; ++i) {
                bool ok = gsr.read_lob(0, snap);
                if (!ok) ++errors;
            }
        });
    }

    // Concurrent writer
    std::thread writer([&]() {
        rtel::LOBSnapshot snap{};
        snap.asset_id = 0;
        snap.n_bid_levels = 1;
        snap.n_ask_levels = 1;
        for (int i = 0; i < 100; ++i) {
            snap.bids[0].price = 150.0 + i * 0.001;
            snap.asks[0].price = 150.01 + i * 0.001;
            snap.compute_derived();
            gsr.update_lob(0, snap);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    for (auto& t : readers) t.join();
    writer.join();

    ASSERT_EQ(errors.load(), 0);
    std::printf("  GSR concurrent reads: %d readers × %d ops, errors=%d\n",
                N_READERS, N_OPS, errors.load());
}

// ---------------------------------------------------------------------------
// TEST: ModuleRegistry and pipeline execution
// ---------------------------------------------------------------------------
TEST(module_registry_topo_order) {
    auto& reg = rtel::ModuleRegistry::instance();
    reg.register_module(std::make_unique<rtel::ChronosWrapper>());
    reg.register_module(std::make_unique<rtel::TensorNetWrapper>());
    reg.register_module(std::make_unique<rtel::LuminaWrapper>());

    auto order = reg.topological_order();
    ASSERT_GT(order.size(), 0u);

    // Verify Chronos comes before TensorNet
    auto it_c = std::find(order.begin(), order.end(), rtel::ModuleID::CHRONOS);
    auto it_t = std::find(order.begin(), order.end(), rtel::ModuleID::TENSORNET);
    if (it_c != order.end() && it_t != order.end()) {
        ASSERT_TRUE(it_c < it_t);
    }
    std::printf("  Topo order: %zu modules\n", order.size());
}

// ---------------------------------------------------------------------------
// TEST: Serialization roundtrip
// ---------------------------------------------------------------------------
TEST(serialization_rtel_roundtrip) {
    rtel::Serializer ser;

    // Create a float32 tensor
    constexpr int N = 100;
    std::vector<float> data(N);
    for (int i = 0; i < N; ++i) data[i] = static_cast<float>(i) * 0.1f;

    rtel::TensorDescriptor td{};
    td.dtype         = rtel::DType::FLOAT32;
    td.ndim          = 1;
    td.shape[0]      = N;
    td.payload_bytes = N * sizeof(float);
    td.compute_c_strides();

    auto blob = ser.serialize(td, data.data(), data.size() * sizeof(float));
    ASSERT_TRUE(!blob.empty());
    ASSERT_TRUE(rtel::Serializer::validate_blob(blob.data(), blob.size()));

    auto result = ser.deserialize(blob.data(), blob.size());
    ASSERT_TRUE(result.ok);
    ASSERT_EQ(result.td.dtype, rtel::DType::FLOAT32);
    ASSERT_EQ(result.td.shape[0], (uint64_t)N);

    const float* out = reinterpret_cast<const float*>(result.data_ptr);
    ASSERT_LT(std::fabs(out[0] - 0.0f), 1e-5f);
    ASSERT_LT(std::fabs(out[N-1] - (N-1)*0.1f), 1e-4f);
    std::printf("  Serialization roundtrip: blob_size=%zu, data[0]=%.2f data[%d]=%.2f\n",
                blob.size(), out[0], N-1, out[N-1]);
}

TEST(serialization_numpy_roundtrip) {
    rtel::Serializer ser;

    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    rtel::TensorDescriptor td{};
    td.dtype = rtel::DType::FLOAT64;
    td.ndim  = 1;
    td.shape[0] = data.size();
    td.payload_bytes = data.size() * sizeof(double);

    auto blob = ser.serialize_numpy(td, data.data(), data.size() * sizeof(double));
    ASSERT_GE(blob.size(), 10u);
    ASSERT_EQ(blob[0], 0x93);
    ASSERT_EQ(blob[1], 'N');

    auto result = ser.deserialize_numpy(blob.data(), blob.size());
    ASSERT_TRUE(result.ok);
    const double* out = reinterpret_cast<const double*>(result.data_ptr);
    ASSERT_LT(std::fabs(out[0] - 1.0), 1e-10);
    ASSERT_LT(std::fabs(out[4] - 5.0), 1e-10);
    std::printf("  NumPy roundtrip: data[0]=%.1f data[4]=%.1f\n", out[0], out[4]);
}

// ---------------------------------------------------------------------------
// TEST: LatencyMonitor
// ---------------------------------------------------------------------------
TEST(latency_monitor_stage_timing) {
    auto& mon = rtel::LatencyMonitor::instance();
    mon.reset();

    // Simulate 100 pipeline runs
    for (int i = 0; i < 100; ++i) {
        rtel::StageLatency lat{};
        lat.stage       = rtel::PipelineStage::CHRONOS;
        lat.start_ns    = rtel::now_ns();
        lat.duration_ns = 10000 + (i * 100);  // 10µs..20µs
        lat.end_ns      = lat.start_ns + lat.duration_ns;
        lat.pipeline_id = i;
        mon.record(lat);
    }

    auto stats = mon.get_stage_stats(rtel::PipelineStage::CHRONOS);
    ASSERT_EQ(stats.count, 100u);
    ASSERT_GT(stats.p50_ns, 0u);
    std::printf("  LatencyMonitor Chronos p50=%luµs p99=%luµs\n",
                stats.p50_ns / 1000, stats.p99_ns / 1000);
}

TEST(latency_monitor_prometheus_export) {
    auto& mon = rtel::LatencyMonitor::instance();
    std::string metrics = mon.prometheus_export();
    ASSERT_TRUE(!metrics.empty());
    ASSERT_TRUE(metrics.find("rtel_stage_latency_ns") != std::string::npos);
    std::printf("  Prometheus export: %zu bytes\n", metrics.size());
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::printf("========================================\n");
    std::printf(" AETERNUS RTEL Test Suite\n");
    std::printf("========================================\n\n");

    // Initialize ShmBus
    rtel::ShmBus::instance().create_aeternus_channels();

    RUN_TEST(shm_bus_register_and_write);
    RUN_TEST(shm_bus_write_read_roundtrip);
    RUN_TEST(shm_bus_backpressure);
    RUN_TEST(ring_buffer_spmc_basic);
    RUN_TEST(ring_buffer_multi_consumer);
    RUN_TEST(ring_buffer_throughput);
    RUN_TEST(latency_histogram);
    RUN_TEST(gsr_lob_update_and_read);
    RUN_TEST(gsr_version_monotonic);
    RUN_TEST(gsr_concurrent_reads);
    RUN_TEST(module_registry_topo_order);
    RUN_TEST(serialization_rtel_roundtrip);
    RUN_TEST(serialization_numpy_roundtrip);
    RUN_TEST(latency_monitor_stage_timing);
    RUN_TEST(latency_monitor_prometheus_export);

    std::printf("\n========================================\n");
    std::printf(" Results: %d passed, %d failed\n", g_pass, g_fail);
    std::printf("========================================\n");
    return g_fail > 0 ? 1 : 0;
}
