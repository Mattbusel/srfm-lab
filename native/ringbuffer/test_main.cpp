#include "ringbuffer.hpp"
#include "tick_store.cpp"
#include "compressor.cpp"
#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>
#include <chrono>
#include <cstdio>
#include <iomanip>

using namespace tickstore;
using namespace tickstore::compress;

static const char* TEMP_PATH = "/tmp/test_ring.bin";

void test_ringbuffer() {
    std::cout << "=== Test: Ring Buffer ===" << std::endl;
    // Remove file if exists
    std::remove(TEMP_PATH);

    RingBuffer<int64_t> rb(TEMP_PATH, 1024);
    assert(rb.is_open());
    assert(rb.empty());

    // Push sequential values
    for (int64_t i = 0; i < 512; ++i) {
        uint64_t seq = rb.push(i * 100);
        assert(seq == static_cast<uint64_t>(i));
    }
    assert(rb.size() == 512);

    // Read back
    for (int64_t i = 0; i < 512; ++i) {
        int64_t v;
        assert(rb.read(static_cast<uint64_t>(i), v));
        assert(v == i * 100);
    }

    // Snapshot
    std::vector<int64_t> snap(100);
    size_t got = rb.snapshot(10, snap.data(), 100);
    assert(got == 100);
    for (size_t i = 0; i < got; ++i) assert(snap[i] == (int64_t)((i+10) * 100));

    // Latest
    int64_t last;
    assert(rb.peek_latest(last));
    assert(last == 511 * 100);

    rb.sync();

    // Reopen: data should persist
    RingBuffer<int64_t> rb2(TEMP_PATH, 1024);
    assert(rb2.size() == 512);
    int64_t v0;
    assert(rb2.read(0, v0) && v0 == 0);

    std::cout << "  [PASS] ring buffer" << std::endl;
    std::remove(TEMP_PATH);
}

void test_tick_store() {
    std::cout << "=== Test: Tick Store ===" << std::endl;
    std::remove("/tmp/test_ticks.ring");

    TickStore store("/tmp/test_ticks.ring", 1 << 16);

    // Append trades
    for (int i = 0; i < 1000; ++i) {
        store.append_trade("AAPL", 15000000 + i * 10, 100 + i, i % 2, i);
    }

    // Append quotes
    for (int i = 0; i < 500; ++i) {
        store.append_quote("AAPL", 14999000, 1000, 15001000, 800);
    }

    assert(store.write_seq() == 1500);

    // Read back
    Tick t;
    assert(store.read(0, t));
    assert(t.tick_type == 0); // trade
    assert(t.qty == 100);

    // VWAP (trades only)
    double v = store.vwap("AAPL", 1000);
    std::cout << "  VWAP(AAPL, 1000): " << std::fixed << std::setprecision(2) << v << std::endl;
    assert(v > 0);

    // Scan
    size_t count = 0;
    store.scan(0, [&](uint64_t, const Tick& tk) {
        ++count;
        return count < 50;
    });
    assert(count == 50);

    store.sync();
    std::cout << "  [PASS] tick store (" << store.write_seq() << " ticks)" << std::endl;
    std::remove("/tmp/test_ticks.ring");
}

void test_compression() {
    std::cout << "=== Test: Compression ===" << std::endl;

    // Delta-encode prices
    {
        std::vector<int64_t> prices;
        int64_t p = 15000000;
        for (int i = 0; i < 1000; ++i) {
            p += (i % 3 == 0) ? 100 : ((i % 3 == 1) ? -50 : 0);
            prices.push_back(p);
        }
        auto enc = delta_encode_prices(prices.data(), prices.size());
        auto dec = delta_decode_prices(enc.data(), enc.size());
        assert(dec.size() == prices.size());
        for (size_t i = 0; i < prices.size(); ++i) assert(dec[i] == prices[i]);
        double ratio = static_cast<double>(prices.size() * 8) / enc.size();
        std::cout << "  Delta price compression ratio: " << std::setprecision(2) << ratio << "x" << std::endl;
    }

    // RLE timestamps
    {
        std::vector<uint64_t> ts;
        uint64_t t = 1700000000000000000ULL;
        for (int i = 0; i < 1000; ++i) {
            t += 1000000; // 1ms intervals (should RLE well)
            if (i % 100 == 0) t += 500000; // occasional jitter
            ts.push_back(t);
        }
        auto rle = rle_encode_timestamps(ts.data(), ts.size());
        auto dec = rle_decode_timestamps(rle);
        assert(dec.size() == ts.size());
        for (size_t i = 0; i < ts.size(); ++i) assert(dec[i] == ts[i]);
        double ratio = static_cast<double>(ts.size() * 8) / (rle.encoded.size() + 8);
        std::cout << "  RLE timestamp ratio: " << ratio << "x" << std::endl;
    }

    // LZ4 compression
    {
        // Create repetitive tick-like data
        std::vector<uint8_t> data(10000, 0);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = static_cast<uint8_t>(i % 32 + (i/100) % 16);

        auto compressed = lz4_compress(data.data(), data.size());
        auto decompressed = lz4_decompress(compressed.data(), compressed.size());
        assert(decompressed.size() == data.size());
        assert(std::equal(data.begin(), data.end(), decompressed.begin()));
        double ratio = static_cast<double>(data.size()) / compressed.size();
        std::cout << "  LZ4 ratio: " << ratio << "x" << std::endl;
    }

    // Full tick compression pipeline
    {
        std::vector<Tick> ticks(500);
        int64_t price = 15000000;
        uint64_t ts = 1700000000000000000ULL;
        for (auto& t : ticks) {
            t.timestamp = ts; ts += 1000000;
            t.price     = price + (price % 7 - 3) * 100;
            t.bid_price = price - 500;
            t.ask_price = price + 500;
            t.qty       = 100 + (price % 50);
            t.side      = price % 2;
            t.tick_type = 0;
            price += 100;
        }

        auto block = compress_ticks(ticks.data(), ticks.size());
        std::cout << "  Full tick compression ratio: " << block.compression_ratio << "x" << std::endl;

        auto recovered = decompress_ticks(block);
        assert(recovered.size() == ticks.size());
        for (size_t i = 0; i < ticks.size(); ++i) {
            assert(recovered[i].price     == ticks[i].price);
            assert(recovered[i].bid_price == ticks[i].bid_price);
            assert(recovered[i].ask_price == ticks[i].ask_price);
            assert(recovered[i].timestamp == ticks[i].timestamp);
            assert(recovered[i].qty       == ticks[i].qty);
        }
    }

    std::cout << "  [PASS] compression" << std::endl;
}

void bench_tick_store() {
    std::cout << "=== Bench: Tick Store ===" << std::endl;
    std::remove("/tmp/bench_ticks.ring");
    TickStore store("/tmp/bench_ticks.ring", 1 << 22); // 4M ticks

    const int N = 1000000;
    Tick t;
    t.set_symbol("MSFT");
    t.price     = 35000000;
    t.qty       = 100;
    t.tick_type = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        t.price += (i % 3 - 1) * 100;
        store.append(t);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  " << N << " appends in " << ms << " ms  ("
              << (N / ms * 1000.0) << " ticks/sec)" << std::endl;

    // Read bench
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        Tick out;
        store.read(static_cast<uint64_t>(i), out);
    }
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  " << N << " reads in " << ms << " ms  ("
              << (N / ms * 1000.0) << " ticks/sec)" << std::endl;

    std::remove("/tmp/bench_ticks.ring");
}

int main() {
    std::cout << "Ring Buffer & Tick Store Tests\n" << std::string(50,'=') << std::endl;
    test_ringbuffer();
    test_tick_store();
    test_compression();
    bench_tick_store();
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
