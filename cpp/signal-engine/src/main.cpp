#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include "srfm/types.hpp"
#include "srfm/ring_buffer.hpp"
#include "streaming/feed_processor.hpp"
#include "io/csv_reader.hpp"
#include "io/json_writer.hpp"
#include "io/binary_protocol.hpp"

using namespace srfm;
using namespace std::chrono;

// ============================================================
// CLI modes
// ============================================================

enum class Mode {
    Run,       // process CSV and output signals
    Benchmark, // time 1M bar updates, report throughput
    Info,      // print configuration info
};

struct Config {
    Mode        mode            = Mode::Run;
    std::string input_csv       = "";
    std::string output_json     = "";
    std::string output_bin      = "";
    int         n_instruments   = 1;
    int         symbol_id       = 0;
    int         timeframe_sec   = 60;
    bool        verbose         = false;
    long long   benchmark_bars  = 1'000'000;
    int         benchmark_insts = 1;
};

static void print_usage(const char* prog) {
    std::printf("Usage: %s [options]\n", prog);
    std::printf("  -i FILE      Input CSV file (OHLCV format)\n");
    std::printf("  -o FILE      Output JSON file (NDJSON format)\n");
    std::printf("  -b FILE      Output binary protocol file\n");
    std::printf("  -n N         Number of instruments (default: 1)\n");
    std::printf("  -s ID        Symbol ID (default: 0)\n");
    std::printf("  -t SEC       Timeframe in seconds (default: 60)\n");
    std::printf("  --bench      Benchmark mode (1M bars)\n");
    std::printf("  --bench-n N  Instruments for benchmark (default: 1)\n");
    std::printf("  --bench-bars N  Bars per benchmark run (default: 1000000)\n");
    std::printf("  --info       Print configuration info\n");
    std::printf("  -v           Verbose output\n");
    std::printf("  -h           Print this help\n");
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            cfg.input_csv = argv[++i];
        else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc)
            cfg.output_json = argv[++i];
        else if (std::strcmp(argv[i], "-b") == 0 && i + 1 < argc)
            cfg.output_bin = argv[++i];
        else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            cfg.n_instruments = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc)
            cfg.symbol_id = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc)
            cfg.timeframe_sec = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bench") == 0)
            cfg.mode = Mode::Benchmark;
        else if (std::strcmp(argv[i], "--bench-n") == 0 && i + 1 < argc)
            cfg.benchmark_insts = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bench-bars") == 0 && i + 1 < argc)
            cfg.benchmark_bars = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--info") == 0)
            cfg.mode = Mode::Info;
        else if (std::strcmp(argv[i], "-v") == 0)
            cfg.verbose = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    return cfg;
}

// ============================================================
// Mode: info
// ============================================================

static void run_info() {
    std::printf("SRFM Signal Engine v1.0.0\n");
    std::printf("C++20 high-performance signal computation\n\n");

    std::printf("Struct sizes:\n");
    std::printf("  OHLCVBar:      %zu bytes (cache-line aligned: %s)\n",
                sizeof(OHLCVBar),
                alignof(OHLCVBar) >= 64 ? "yes" : "no");
    std::printf("  TickData:      %zu bytes\n", sizeof(TickData));
    std::printf("  SignalOutput:  %zu bytes\n", sizeof(SignalOutput));
    std::printf("  BinaryMessage: %zu bytes\n", sizeof(BinaryMessage));

    std::printf("\nISA features:\n");
#if defined(SRFM_AVX2)
    std::printf("  AVX2: YES\n");
#else
    std::printf("  AVX2: NO (scalar fallback)\n");
#endif
#if defined(SRFM_SSE2)
    std::printf("  SSE2: YES\n");
#else
    std::printf("  SSE2: NO\n");
#endif

    std::printf("\nConstants:\n");
    std::printf("  BH mass threshold:   %.3f\n", constants::BH_MASS_THRESH);
    std::printf("  BH EMA decay:        %.4f\n", constants::BH_EMA_DECAY);
    std::printf("  BH collapse rate:    %.4f\n", constants::BH_COLLAPSE_RATE);
    std::printf("  GARCH omega:         %.8f\n", constants::GARCH_OMEGA);
    std::printf("  GARCH alpha:         %.2f\n", constants::GARCH_ALPHA);
    std::printf("  GARCH beta:          %.2f\n", constants::GARCH_BETA);
    std::printf("  Target vol:          %.2f%%\n", constants::TARGET_VOL * 100.0);
    std::printf("  Per-inst risk:       %.2f%%\n", constants::PER_INST_RISK * 100.0);
    std::printf("  OU window:           %d bars\n", constants::OU_WINDOW);
    std::printf("  OU long threshold:   %.1f\n", constants::OU_ZSCORE_LONG);
    std::printf("  OU short threshold:  %.1f\n", constants::OU_ZSCORE_SHORT);
    std::printf("  Max instruments:     %d\n", MAX_INSTRUMENTS);
}

// ============================================================
// Mode: run (process CSV)
// ============================================================

static int run_csv(const Config& cfg) {
    if (cfg.input_csv.empty()) {
        std::fprintf(stderr, "Error: no input file specified (-i)\n");
        return 1;
    }

    // Load bars
    std::vector<OHLCVBar> bars;
    bars.reserve(500000);

    CSVReader reader(cfg.input_csv, cfg.symbol_id, cfg.timeframe_sec);
    if (!reader.open()) {
        std::fprintf(stderr, "Error: cannot open %s\n", cfg.input_csv.c_str());
        return 1;
    }

    auto t0 = high_resolution_clock::now();
    std::size_t n_bars = reader.read_all([&](const OHLCVBar& b) {
        bars.push_back(b);
    });
    auto t_load = high_resolution_clock::now();

    double load_ms = duration_cast<microseconds>(t_load - t0).count() / 1000.0;
    if (cfg.verbose)
        std::printf("Loaded %zu bars in %.2f ms (%.0f bars/s)\n",
                    n_bars, load_ms, n_bars / (load_ms / 1000.0));

    // Set up processor
    FeedProcessor proc(cfg.n_instruments);

    // Optional output writers
    JSONWriter json_writer;
    bool write_json = !cfg.output_json.empty();
    if (write_json && !json_writer.open(cfg.output_json)) {
        std::fprintf(stderr, "Error: cannot open output %s\n",
                     cfg.output_json.c_str());
        return 1;
    }

    BinaryPipeWriter bin_writer(cfg.output_bin);
    bool write_bin = !cfg.output_bin.empty();
    if (write_bin) bin_writer.open();

    // Process all bars
    auto t1 = high_resolution_clock::now();
    std::size_t count = 0;
    double last_log_ts = 0.0;

    for (const auto& bar : bars) {
        SignalOutput sig = proc.process_bar(bar);

        if (write_json) json_writer.write(sig);

        if (write_bin) {
            BinaryMessage msgs[16];
            int n_msg = BinaryEncoder::encode(sig, msgs, 16);
            bin_writer.write_batch(msgs, n_msg);
        }

        ++count;

        if (cfg.verbose && count % 10000 == 0) {
            std::printf("\r  Processed %zu / %zu bars...", count, n_bars);
            std::fflush(stdout);
        }
    }

    auto t2 = high_resolution_clock::now();
    double proc_us = duration_cast<microseconds>(t2 - t1).count();
    double proc_ms = proc_us / 1000.0;

    std::printf("\nProcessed %zu bars in %.2f ms\n", count, proc_ms);
    std::printf("  Throughput:  %.0f bars/s\n",    count / (proc_ms / 1000.0));
    std::printf("  Per bar:     %.2f µs\n",         proc_us / count);
    std::printf("  Parse errors: %d\n",             reader.error_count());

    if (write_json)
        std::printf("  JSON records: %zu → %s\n",
                    json_writer.records_written(), cfg.output_json.c_str());

    // Print last signal as sample
    if (count > 0 && cfg.verbose) {
        const SignalOutput& last = proc.last_signal(cfg.symbol_id);
        std::printf("\nLast signal (symbol %d):\n", cfg.symbol_id);
        std::printf("  BH mass=%.4f  active=%d  dir=%.4f  cf_scale=%.4f\n",
                    last.bh_mass, last.bh_active, last.bh_dir, last.cf_scale);
        std::printf("  EMA fast=%.4f  slow=%.4f\n",
                    last.ema_fast, last.ema_slow);
        std::printf("  ATR=%.4f  RSI=%.2f\n", last.atr, last.rsi);
        std::printf("  GARCH vol_scale=%.4f\n", last.garch_vol_scale);
        std::printf("  OU z=%.3f  half_life=%.1f\n",
                    last.ou_zscore, last.ou_half_life);
        std::printf("  Position size=%.4f\n", last.position_size);
    }

    return 0;
}

// ============================================================
// Mode: benchmark
// ============================================================

static void run_benchmark(const Config& cfg) {
    const long long N_BARS  = cfg.benchmark_bars;
    const int       N_INSTS = std::max(1, std::min(cfg.benchmark_insts,
                                                    MAX_INSTRUMENTS));

    std::printf("SRFM Signal Engine Benchmark\n");
    std::printf("  Instruments: %d\n", N_INSTS);
    std::printf("  Bars/run:    %lld\n", N_BARS);
    std::printf("\n");

    // Generate synthetic price series (geometric random walk)
    std::vector<OHLCVBar> bars(N_BARS);
    double price = 50000.0;  // BTC-like starting price
    int64_t ts   = 1700000000LL * constants::NS_PER_SEC;

    // Simple LCG for reproducible pseudo-random numbers
    uint64_t rng = 12345678901234ULL;
    auto next_rng = [&]() -> double {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<double>(rng >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
    };

    for (long long i = 0; i < N_BARS; ++i) {
        double ret  = next_rng() * 0.002;  // ~0.2% per bar
        double hl   = std::abs(next_rng()) * 0.003;
        double open = price;
        price      *= (1.0 + ret);
        double high = price * (1.0 + hl * 0.5);
        double low  = price * (1.0 - hl * 0.5);
        double vol  = 1000.0 + std::abs(next_rng()) * 500.0;

        bars[i] = OHLCVBar(open, high, low, price, vol,
                           ts + i * constants::NS_PER_MIN,
                           static_cast<int32_t>(i % N_INSTS));
    }

    // Warm up
    {
        FeedProcessor proc(N_INSTS);
        for (int i = 0; i < std::min(N_BARS, 100LL); ++i)
            proc.process_bar(bars[i]);
    }

    // Run benchmark
    struct RunResult {
        long long   bars;
        int         instruments;
        double      total_us;
        double      throughput_bps; // bars per second
        double      latency_us;     // per-bar latency
    };

    for (int n = 1; n <= N_INSTS; n = (n < 4) ? n + 1 : n * 2) {
        if (n > N_INSTS) n = N_INSTS;

        FeedProcessor proc(n);

        // Assign symbol_ids round-robin
        for (auto& b : bars) {
            b.symbol_id = static_cast<int32_t>(
                (&b - bars.data()) % n);
        }

        auto t0 = high_resolution_clock::now();
        for (long long i = 0; i < N_BARS; ++i) {
            proc.process_bar(bars[i]);
        }
        auto t1 = high_resolution_clock::now();

        double us     = duration_cast<microseconds>(t1 - t0).count();
        double bps    = N_BARS / (us / 1e6);
        double lat_us = us / N_BARS;

        std::printf("  [%2d instr] %lld bars in %.1f ms  "
                    "%.2fM bars/s  %.3f µs/bar\n",
                    n, N_BARS, us / 1000.0, bps / 1e6, lat_us);

        // Check target
        bool ok_single = (n == 1 && bps >= 10e6);
        bool ok_20     = (n == 20 && bps >= 500e3);
        if (n == 1)
            std::printf("    Target >10M bars/s: %s\n", ok_single ? "PASS" : "MISS");
        if (n == 20)
            std::printf("    Target >500K bars/s: %s\n", ok_20 ? "PASS" : "MISS");

        if (n == N_INSTS) break;
    }
}

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    switch (cfg.mode) {
    case Mode::Info:
        run_info();
        return 0;

    case Mode::Benchmark:
        run_benchmark(cfg);
        return 0;

    case Mode::Run:
    default:
        if (cfg.input_csv.empty()) {
            // No input: print info + help
            run_info();
            std::printf("\n");
            print_usage(argv[0]);
            return 0;
        }
        return run_csv(cfg);
    }
}
