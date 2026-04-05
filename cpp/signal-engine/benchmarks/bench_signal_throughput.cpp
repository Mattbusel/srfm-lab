/// bench_signal_throughput.cpp
/// Hand-rolled benchmark for full signal computation pipeline.
/// Times 1, 5, 10, 20 instruments simultaneously.

#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cstring>

#include "srfm/types.hpp"
#include "../src/streaming/feed_processor.hpp"
#include "../src/indicators/ema.hpp"
#include "../src/indicators/atr.hpp"
#include "../src/indicators/rsi.hpp"
#include "../src/indicators/bollinger.hpp"
#include "../src/indicators/macd.hpp"
#include "../src/indicators/vwap.hpp"
#include "../src/bh_physics/bh_state.hpp"
#include "../src/bh_physics/garch.hpp"
#include "../src/bh_physics/ou_detector.hpp"
#include "../src/io/json_writer.hpp"
#include "../src/io/binary_protocol.hpp"

using namespace srfm;
using namespace std::chrono;

// ============================================================
// Benchmark timer helper
// ============================================================

struct Timer {
    high_resolution_clock::time_point start;
    Timer() : start(high_resolution_clock::now()) {}

    double elapsed_us() const {
        return static_cast<double>(
            duration_cast<microseconds>(high_resolution_clock::now() - start).count());
    }
    double elapsed_ms() const { return elapsed_us() / 1000.0; }
    double elapsed_sec() const { return elapsed_us() / 1e6; }
};

// ============================================================
// Benchmark statistics
// ============================================================

struct BenchStats {
    double min_us, max_us, mean_us, p99_us, throughput;
    long long n;

    static BenchStats compute(const std::vector<double>& latencies_us,
                               long long total_items) {
        BenchStats s{};
        s.n = total_items;
        if (latencies_us.empty()) return s;

        auto sorted = latencies_us;
        std::sort(sorted.begin(), sorted.end());

        s.min_us  = sorted.front();
        s.max_us  = sorted.back();
        s.mean_us = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
        s.p99_us  = sorted[static_cast<std::size_t>(sorted.size() * 0.99)];

        double total_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
        s.throughput = total_items / (total_us / 1e6);
        return s;
    }

    void print(const char* name) const {
        std::printf("  [%s]\n", name);
        std::printf("    Total items:  %lld\n", n);
        std::printf("    Throughput:   %.2fM items/s\n", throughput / 1e6);
        std::printf("    Mean latency: %.3f µs\n", mean_us);
        std::printf("    P99  latency: %.3f µs\n", p99_us);
        std::printf("    Min / Max:    %.3f / %.3f µs\n", min_us, max_us);
    }
};

// ============================================================
// Synthetic price series generator
// ============================================================

struct PriceSeries {
    std::vector<OHLCVBar> bars;

    PriceSeries(int n_instruments, long long n_bars, uint64_t seed = 42) {
        bars.reserve(static_cast<std::size_t>(n_instruments * n_bars));
        uint64_t rng = seed;
        auto next = [&]() -> double {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            return (static_cast<double>(rng & 0xFFFFFF) / double(0x1000000)) * 2.0 - 1.0;
        };

        std::vector<double> prices(n_instruments);
        for (int i = 0; i < n_instruments; ++i)
            prices[i] = 10000.0 + i * 5000.0;

        for (long long i = 0; i < n_bars; ++i) {
            for (int s = 0; s < n_instruments; ++s) {
                double ret = next() * 0.002;
                double hl  = std::abs(next()) * 0.003 + 0.0001;
                double o   = prices[s];
                prices[s] *= (1.0 + ret);
                bars.push_back(OHLCVBar(
                    o, prices[s] * (1.0 + hl), prices[s] * (1.0 - hl),
                    prices[s], 500.0 + std::abs(next()) * 1000.0,
                    1700000000LL * constants::NS_PER_SEC + i * constants::NS_PER_MIN,
                    s, 60));
            }
        }
    }
};

// ============================================================
// Benchmark 1: Full pipeline, single instrument
// ============================================================

static void bench_full_pipeline_1_instrument() {
    std::printf("\n=== Full Pipeline: 1 Instrument ===\n");

    constexpr long long N_BARS = 2000000;
    PriceSeries ps(1, N_BARS);

    FeedProcessor proc(1);

    // Warmup
    for (int i = 0; i < 1000 && i < static_cast<int>(ps.bars.size()); ++i)
        proc.process_bar(ps.bars[i]);
    proc.reset();

    // Measure in batches of 1000 bars to get latency distribution
    constexpr long long BATCH = 1000;
    std::vector<double> latencies;
    latencies.reserve(static_cast<std::size_t>(N_BARS / BATCH));

    Timer total_timer;
    for (long long i = 0; i < N_BARS; i += BATCH) {
        long long end = std::min(i + BATCH, N_BARS);
        Timer t;
        for (long long j = i; j < end; ++j)
            proc.process_bar(ps.bars[j]);
        latencies.push_back(t.elapsed_us() / (end - i));
    }

    auto stats = BenchStats::compute(latencies, N_BARS);
    stats.print("1 instrument");
}

// ============================================================
// Benchmark 2: Full pipeline, multiple instruments
// ============================================================

static void bench_full_pipeline_N_instruments(int n_instruments, long long n_bars_per) {
    std::printf("\n=== Full Pipeline: %d Instruments ===\n", n_instruments);

    PriceSeries ps(n_instruments, n_bars_per);
    long long total = static_cast<long long>(ps.bars.size());

    FeedProcessor proc(n_instruments);

    // Warmup
    for (int i = 0; i < std::min(1000LL, total); ++i)
        proc.process_bar(ps.bars[i]);
    proc.reset();

    constexpr long long BATCH = 1000;
    std::vector<double> latencies;
    latencies.reserve(static_cast<std::size_t>(total / BATCH + 1));

    for (long long i = 0; i < total; i += BATCH) {
        long long end = std::min(i + BATCH, total);
        Timer t;
        for (long long j = i; j < end; ++j)
            proc.process_bar(ps.bars[j]);
        latencies.push_back(t.elapsed_us() / (end - i));
    }

    auto stats = BenchStats::compute(latencies, total);
    {
        std::string lbl = std::to_string(n_instruments) + " instruments";
        stats.print(lbl.c_str());
    }
    (void)stats;

    // Print throughput
    double total_us = 0;
    for (double l : latencies) total_us += l;
    // Actually total_us / latencies.size() is mean per bar;
    // total throughput = total / total_elapsed
    // We don't have total elapsed directly from the mean; re-run timed:
    proc.reset();
    Timer big;
    for (long long j = 0; j < total; ++j) proc.process_bar(ps.bars[j]);
    double elapsed = big.elapsed_sec();
    double tput    = total / elapsed;
    std::printf("    Total elapsed: %.2f s  =>  %.2fM bars/s  (%.3f µs/bar)\n",
                elapsed, tput / 1e6, elapsed * 1e6 / total);
}

// ============================================================
// Benchmark 3: Individual indicators
// ============================================================

static void bench_individual_indicators() {
    std::printf("\n=== Individual Indicator Benchmarks ===\n");
    constexpr int N = 1000000;

    double price = 50000.0;
    double sum   = 0.0; // prevent dead-code elimination

    // EMA
    {
        EMA ema(9);
        Timer t;
        for (int i = 0; i < N; ++i) {
            price += (i % 2 == 0 ? 1.0 : -0.9);
            sum += ema.update(price);
        }
        double ns = t.elapsed_us() * 1000.0 / N;
        std::printf("  EMA(9):              %.1f ns/bar\n", ns);
    }

    // ATR
    {
        ATR atr(14);
        double h = price, l = price;
        Timer t;
        for (int i = 0; i < N; ++i) {
            h = price + 50.0; l = price - 50.0;
            price += (i % 2 == 0 ? 1.0 : -0.9);
            sum += atr.update(h, l, price);
        }
        double ns = t.elapsed_us() * 1000.0 / N;
        std::printf("  ATR(14):             %.1f ns/bar\n", ns);
    }

    // RSI
    {
        RSI rsi(14);
        Timer t;
        for (int i = 0; i < N; ++i) {
            price += (i % 2 == 0 ? 1.0 : -0.9);
            sum += rsi.update(price);
        }
        double ns = t.elapsed_us() * 1000.0 / N;
        std::printf("  RSI(14):             %.1f ns/bar\n", ns);
    }

    // Bollinger
    {
        BollingerBands bb(20, 2.0);
        Timer t;
        for (int i = 0; i < N; ++i) {
            price += (i % 2 == 0 ? 1.0 : -0.9);
            auto out = bb.update(price);
            sum += out.mid;
        }
        double ns = t.elapsed_us() * 1000.0 / N;
        std::printf("  Bollinger(20,2):     %.1f ns/bar\n", ns);
    }

    // MACD
    {
        MACD macd;
        Timer t;
        for (int i = 0; i < N; ++i) {
            price += (i % 2 == 0 ? 1.0 : -0.9);
            sum += macd.update(price).macd_line;
        }
        double ns = t.elapsed_us() * 1000.0 / N;
        std::printf("  MACD(12,26,9):       %.1f ns/bar\n", ns);
    }

    // BH state
    {
        BHState bh;
        int64_t ts = 1700000000LL * constants::NS_PER_SEC;
        Timer t;
        for (int i = 0; i < N; ++i) {
            price += (i % 2 == 0 ? 1.0 : -0.9);
            sum += bh.update(price, 1000.0, ts + i * constants::NS_PER_MIN).mass;
        }
        double ns = t.elapsed_us() * 1000.0 / N;
        std::printf("  BHState:             %.1f ns/bar\n", ns);
    }

    // GARCH
    {
        GARCHTracker g;
        Timer t;
        for (int i = 0; i < N; ++i) {
            price += (i % 2 == 0 ? 1.0 : -0.9);
            sum += g.update(price).variance;
        }
        double ns = t.elapsed_us() * 1000.0 / N;
        std::printf("  GARCH(1,1):          %.1f ns/bar\n", ns);
    }

    // Prevent dead-code elimination
    if (sum < 0.0) std::printf("  (never printed: %g)\n", sum);
}

// ============================================================
// Benchmark 4: JSON serialization throughput
// ============================================================

static void bench_json_serialization() {
    std::printf("\n=== JSON Serialization Benchmark ===\n");

    constexpr int N = 500000;
    SignalOutput sig{};
    sig.timestamp_ns   = 1700000000LL * constants::NS_PER_SEC;
    sig.symbol_id      = 0;
    sig.bh_mass        = 1.5;
    sig.rsi            = 55.3;
    sig.ema_fast       = 50123.4;
    sig.ema_slow       = 50100.0;
    sig.garch_vol_scale= 0.95;
    sig.ou_zscore      = -1.2;

    char buf[2048];
    Timer t;
    for (int i = 0; i < N; ++i) {
        sig.bar_count = i;
        JSONWriter::serialize(sig, buf, sizeof(buf));
    }
    double us = t.elapsed_us();
    std::printf("  %d JSON serializations in %.2f ms  =>  %.2fM/s  (%.1f µs each)\n",
                N, us / 1000.0, N / (us / 1e6) / 1e6, us / N);
}

// ============================================================
// Benchmark 5: Binary protocol encoding
// ============================================================

static void bench_binary_protocol() {
    std::printf("\n=== Binary Protocol Benchmark ===\n");

    constexpr int N = 1000000;
    SignalOutput sig{};
    sig.bh_mass   = 1.5;
    sig.rsi       = 55.0;
    sig.ema_fast  = 50000.0;
    BinaryMessage msgs[16];

    Timer t;
    for (int i = 0; i < N; ++i) {
        sig.timestamp_ns = i;
        BinaryEncoder::encode(sig, msgs, 16);
    }
    double us = t.elapsed_us();
    std::printf("  %d encodings in %.2f ms  =>  %.2fM/s  (%.1f ns each)\n",
                N, us / 1000.0, N / (us / 1e6) / 1e6, us * 1000.0 / N);
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    std::printf("SRFM Signal Engine Throughput Benchmarks\n");
    std::printf("==========================================\n");

    bench_full_pipeline_1_instrument();

    int instrument_counts[] = { 5, 10, 20 };
    for (int n : instrument_counts) {
        bench_full_pipeline_N_instruments(n, 200000);
    }

    bench_individual_indicators();
    bench_json_serialization();
    bench_binary_protocol();

    std::printf("\nDone.\n");
    return 0;
}
