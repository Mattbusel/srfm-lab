/// test_ring_buffer.cpp
/// Concurrent producer-consumer test for RingBuffer and MPSCRingBuffer.

#include <cstdio>
#include <cmath>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <numeric>
#include <cassert>

#include "srfm/types.hpp"
#include "srfm/ring_buffer.hpp"

using namespace srfm;
using namespace std::chrono;

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) do { \
    if (!(expr)) { std::fprintf(stderr,"FAIL  %s:%d  %s\n",__FILE__,__LINE__,#expr); ++g_fail; } \
    else { ++g_pass; } \
} while(0)

static void section(const char* n) { std::printf("--- %s ---\n", n); }

// ============================================================
// Basic single-threaded tests
// ============================================================

static void test_ringbuf_empty() {
    section("RingBuffer empty/full state");

    RingBuffer<int, 8> rb;
    CHECK(rb.empty());
    CHECK(rb.size() == 0);

    rb.push(1);
    CHECK(!rb.empty());
    CHECK(rb.size() == 1);

    int v;
    rb.pop(v);
    CHECK(rb.empty());
    CHECK(rb.size() == 0);
}

static void test_ringbuf_fifo_order() {
    section("RingBuffer FIFO order");

    RingBuffer<int, 16> rb;
    for (int i = 0; i < 8; ++i) {
        bool ok = rb.push(i);
        CHECK(ok);
    }

    for (int i = 0; i < 8; ++i) {
        int v;
        bool ok = rb.pop(v);
        CHECK(ok);
        CHECK(v == i);
    }
    CHECK(rb.empty());
}

static void test_ringbuf_full() {
    section("RingBuffer full behavior");

    RingBuffer<int, 4> rb;  // capacity = 3 (4 - 1)
    CHECK(rb.push(1));
    CHECK(rb.push(2));
    CHECK(rb.push(3));
    // Fourth push should fail (full)
    CHECK(!rb.push(4));
    CHECK(rb.size() == 3);
}

static void test_ringbuf_wrap() {
    section("RingBuffer wrap-around");

    RingBuffer<int, 8> rb;

    // Push 4, pop 4, push 4 more to exercise wrap-around
    for (int i = 0; i < 4; ++i) rb.push(i);
    for (int i = 0; i < 4; ++i) {
        int v; rb.pop(v);
        CHECK(v == i);
    }
    for (int i = 4; i < 8; ++i) rb.push(i);
    for (int i = 4; i < 8; ++i) {
        int v; rb.pop(v);
        CHECK(v == i);
    }
    CHECK(rb.empty());
}

static void test_ringbuf_peek() {
    section("RingBuffer peek");

    RingBuffer<int, 8> rb;
    CHECK(rb.peek() == nullptr);

    rb.push(42);
    const int* p = rb.peek();
    CHECK(p != nullptr && *p == 42);

    // Peek doesn't consume
    CHECK(rb.size() == 1);
    int v; rb.pop(v);
    CHECK(v == 42);
    CHECK(rb.peek() == nullptr);
}

static void test_ringbuf_optional_api() {
    section("RingBuffer optional pop API");

    RingBuffer<int, 8> rb;
    auto v = rb.pop();
    CHECK(!v.has_value());

    rb.push(99);
    auto v2 = rb.pop();
    CHECK(v2.has_value() && *v2 == 99);
}

static void test_ringbuf_drain() {
    section("RingBuffer drain");

    RingBuffer<int, 16> rb;
    for (int i = 0; i < 5; ++i) rb.push(i * 10);

    std::vector<int> collected;
    std::size_t cnt = rb.drain([&](int v) { collected.push_back(v); });

    CHECK(cnt == 5);
    CHECK(rb.empty());
    for (int i = 0; i < 5; ++i) CHECK(collected[i] == i * 10);
}

// ============================================================
// OHLCVBar through ring buffer
// ============================================================

static void test_ringbuf_ohlcv() {
    section("RingBuffer with OHLCVBar");

    RingBuffer<OHLCVBar, 64> rb;
    CHECK(rb.empty());

    // Push several bars
    for (int i = 0; i < 10; ++i) {
        OHLCVBar bar(100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0,
                     1700000000LL * constants::NS_PER_SEC +
                     static_cast<int64_t>(i) * constants::NS_PER_MIN,
                     0, 60);
        CHECK(rb.push(bar));
    }
    CHECK(rb.size() == 10);

    for (int i = 0; i < 10; ++i) {
        OHLCVBar bar;
        CHECK(rb.pop(bar));
        CHECK(bar.symbol_id == 0);
        CHECK(bar.close == 100.5 + i);
    }
}

// ============================================================
// Concurrent SPSC test
// ============================================================

static void test_spsc_concurrent() {
    section("SPSC concurrent producer-consumer");

    constexpr int N_ITEMS = 100000;
    RingBuffer<int, 4096> rb;

    std::atomic<long long> sum_produced{0}, sum_consumed{0};
    std::atomic<int>       items_consumed{0};

    auto producer = [&]() {
        for (int i = 0; i < N_ITEMS; ++i) {
            while (!rb.push(i)) {
                std::this_thread::yield();
            }
            sum_produced.fetch_add(i, std::memory_order_relaxed);
        }
    };

    auto consumer = [&]() {
        int v;
        int count = 0;
        while (count < N_ITEMS) {
            if (rb.pop(v)) {
                sum_consumed.fetch_add(v, std::memory_order_relaxed);
                ++count;
            } else {
                std::this_thread::yield();
            }
        }
        items_consumed.store(count, std::memory_order_relaxed);
    };

    auto t0 = high_resolution_clock::now();
    std::thread prod_thread(producer);
    std::thread cons_thread(consumer);
    prod_thread.join();
    cons_thread.join();
    auto t1 = high_resolution_clock::now();

    double us = duration_cast<microseconds>(t1 - t0).count();
    double throughput = N_ITEMS / (us / 1e6);

    std::printf("  SPSC: %d items in %.2f ms  %.2fM items/s\n",
                N_ITEMS, us / 1000.0, throughput / 1e6);

    CHECK(items_consumed.load() == N_ITEMS);
    CHECK(sum_produced.load()   == sum_consumed.load());
    CHECK(rb.empty());
}

// ============================================================
// MPSC concurrent test
// ============================================================

static void test_mpsc_concurrent() {
    section("MPSC concurrent producers");

    constexpr int N_PRODUCERS = 4;
    constexpr int ITEMS_PER   = 10000;
    constexpr int TOTAL        = N_PRODUCERS * ITEMS_PER;

    MPSCRingBuffer<int, 4096> rb;
    std::atomic<long long> sum_consumed{0};
    std::atomic<int>       items_consumed{0};

    std::vector<std::thread> producers;
    for (int p = 0; p < N_PRODUCERS; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < ITEMS_PER; ++i) {
                int val = p * ITEMS_PER + i;
                while (!rb.push(val)) {
                    std::this_thread::yield();
                }
            }
        });
    }

    std::thread consumer([&]() {
        int v;
        int count = 0;
        while (count < TOTAL) {
            if (rb.pop(v)) {
                sum_consumed.fetch_add(v, std::memory_order_relaxed);
                ++count;
            } else {
                std::this_thread::yield();
            }
        }
        items_consumed.store(count, std::memory_order_relaxed);
    });

    for (auto& t : producers) t.join();
    consumer.join();

    std::printf("  MPSC: %d total items consumed\n", items_consumed.load());

    CHECK(items_consumed.load() == TOTAL);

    // Expected sum: sum over all producers of sum(p*ITEMS + 0..ITEMS-1)
    long long expected_sum = 0;
    for (int p = 0; p < N_PRODUCERS; ++p) {
        for (int i = 0; i < ITEMS_PER; ++i) {
            expected_sum += p * ITEMS_PER + i;
        }
    }
    CHECK(sum_consumed.load() == expected_sum);
}

// ============================================================
// Latency measurement
// ============================================================

static void test_ringbuf_latency() {
    section("RingBuffer round-trip latency");

    constexpr int ITERS = 1000000;
    RingBuffer<int64_t, 16> rb;

    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        rb.push(static_cast<int64_t>(i));
        int64_t v;
        rb.pop(v);
    }
    auto t1 = high_resolution_clock::now();

    double ns_per_iter = duration_cast<nanoseconds>(t1 - t0).count()
                         / static_cast<double>(ITERS);
    std::printf("  Round-trip (push+pop): %.1f ns per iteration\n", ns_per_iter);
    // Should be < 100ns for a local ring buffer
    CHECK(ns_per_iter < 500.0);
}

// ============================================================
// Main
// ============================================================

int main() {
    std::printf("=== Ring Buffer Tests ===\n\n");

    test_ringbuf_empty();
    test_ringbuf_fifo_order();
    test_ringbuf_full();
    test_ringbuf_wrap();
    test_ringbuf_peek();
    test_ringbuf_optional_api();
    test_ringbuf_drain();
    test_ringbuf_ohlcv();
    test_spsc_concurrent();
    test_mpsc_concurrent();
    test_ringbuf_latency();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
