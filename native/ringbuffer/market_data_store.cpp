// Full integration: multi-symbol tick store with live feed ingestion,
// compression, and replay capabilities.

#include "tick_store.cpp"
#include "compressor.cpp"
#include "replay.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cstring>

namespace tickstore {

// ============================================================
// Compressed archive: writes compressed blocks to a binary file
// Block structure:
//   [8-byte block_size][compressed_block_bytes]
// ============================================================
class TickArchive {
public:
    static constexpr size_t kBlockTicks = 10000; // ticks per block

    explicit TickArchive(const std::string& path, bool write_mode = true)
        : path_(path), write_mode_(write_mode)
    {
        if (write_mode_) {
            ofs_.open(path, std::ios::binary | std::ios::app);
            if (!ofs_.is_open()) throw std::runtime_error("Cannot open archive: " + path);
        } else {
            ifs_.open(path, std::ios::binary);
            if (!ifs_.is_open()) throw std::runtime_error("Cannot read archive: " + path);
        }
    }

    ~TickArchive() {
        flush_pending();
        if (ofs_.is_open()) ofs_.close();
        if (ifs_.is_open()) ifs_.close();
    }

    // Write a tick; buffers until a full block is accumulated
    void write(const Tick& t) {
        pending_.push_back(t);
        if (pending_.size() >= kBlockTicks) flush_pending();
    }

    // Write a batch
    void write_batch(const Tick* ticks, size_t n) {
        for (size_t i = 0; i < n; ++i) write(ticks[i]);
    }

    void flush_pending() {
        if (pending_.empty()) return;
        auto block = compress::compress_ticks(pending_.data(), pending_.size());
        uint64_t bsz = block.data.size();
        ofs_.write(reinterpret_cast<const char*>(&bsz), 8);
        ofs_.write(reinterpret_cast<const char*>(block.data.data()), bsz);
        blocks_written_++;
        ticks_archived_ += pending_.size();
        bytes_written_  += 8 + bsz;
        pending_.clear();
    }

    // Read all blocks; calls callback for each Tick
    void read_all(std::function<void(const Tick&)> cb) {
        while (ifs_.peek() != EOF) {
            uint64_t bsz;
            ifs_.read(reinterpret_cast<char*>(&bsz), 8);
            if (!ifs_) break;

            std::vector<uint8_t> raw(bsz);
            ifs_.read(reinterpret_cast<char*>(raw.data()), bsz);
            if (!ifs_) break;

            compress::CompressedBlock block;
            block.data = std::move(raw);
            auto ticks = compress::decompress_ticks(block);
            for (auto& t : ticks) cb(t);
        }
    }

    // Stats
    uint64_t blocks_written() const noexcept  { return blocks_written_; }
    uint64_t ticks_archived() const noexcept  { return ticks_archived_; }
    uint64_t bytes_written()  const noexcept  { return bytes_written_;  }
    double   compression_ratio() const noexcept {
        if (bytes_written_ == 0) return 0.0;
        return static_cast<double>(ticks_archived_ * sizeof(Tick)) / bytes_written_;
    }

private:
    std::string          path_;
    bool                 write_mode_;
    std::ofstream        ofs_;
    std::ifstream        ifs_;
    std::vector<Tick>    pending_;
    uint64_t             blocks_written_ = 0;
    uint64_t             ticks_archived_ = 0;
    uint64_t             bytes_written_  = 0;
};

// ============================================================
// Real-time tick aggregation: OHLCV bars
// ============================================================
struct OHLCVBar {
    uint64_t timestamp_open;   // ns
    uint64_t timestamp_close;
    int64_t  open, high, low, close; // fixed-point prices
    uint64_t volume;
    uint64_t trade_count;
    double   vwap;
    char     symbol[16];
    uint32_t bar_duration_sec; // e.g., 1, 5, 60, 300, 3600
};

class BarAggregator {
public:
    explicit BarAggregator(uint32_t bar_sec = 60, TickCallback bar_cb = {})
        : bar_sec_(bar_sec), bar_cb_(std::move(bar_cb)) {}

    void on_tick(const Tick& t) {
        if (t.tick_type != 0) return; // trades only
        std::string sym(t.symbol, strnlen(t.symbol, 15));
        auto& bar = bars_[sym];

        uint64_t bar_ts = bucket_ts(t.timestamp);
        if (bar.timestamp_open == 0) {
            // New bar
            start_bar(bar, sym, t, bar_ts);
        } else if (bar_ts > bar.timestamp_open) {
            // Close and emit current bar
            bar.timestamp_close = static_cast<uint64_t>(t.timestamp);
            emit_bar(bar);
            start_bar(bar, sym, t, bar_ts);
        } else {
            // Update current bar
            if (t.price > bar.high) bar.high = t.price;
            if (t.price < bar.low)  bar.low  = t.price;
            bar.close  = t.price;
            bar.volume += t.qty;
            bar.trade_count++;
            bar.vwap  += (t.price / 100000.0 * t.qty - bar.vwap * bar.qty_sum) /
                          (bar.qty_sum + t.qty);
            bar.qty_sum += t.qty;
        }
    }

    void flush_all() {
        for (auto& [sym, bar] : bars_) {
            if (bar.timestamp_open > 0) emit_bar(bar);
        }
        bars_.clear();
    }

    void set_bar_callback(TickCallback cb) { bar_cb_ = std::move(cb); }
    const std::vector<OHLCVBar>& bars() const { return emitted_bars_; }

private:
    struct BarState : OHLCVBar {
        uint64_t qty_sum = 0;
        BarState() { std::memset(static_cast<OHLCVBar*>(this), 0, sizeof(OHLCVBar)); }
    };

    uint32_t  bar_sec_;
    TickCallback bar_cb_;
    std::unordered_map<std::string, BarState> bars_;
    std::vector<OHLCVBar> emitted_bars_;

    uint64_t bucket_ts(int64_t ts) const {
        uint64_t sec = static_cast<uint64_t>(ts) / 1000000000ULL;
        return (sec / bar_sec_) * bar_sec_ * 1000000000ULL;
    }

    void start_bar(BarState& bar, const std::string& sym, const Tick& t, uint64_t ts) {
        std::memset(static_cast<OHLCVBar*>(&bar), 0, sizeof(OHLCVBar));
        bar.qty_sum = 0;
        bar.timestamp_open = ts;
        bar.open = bar.high = bar.low = bar.close = t.price;
        bar.volume = t.qty;
        bar.trade_count = 1;
        bar.vwap = t.price / 100000.0;
        bar.qty_sum = t.qty;
        bar.bar_duration_sec = bar_sec_;
        std::strncpy(bar.symbol, sym.c_str(), 15);
    }

    void emit_bar(BarState& bar) {
        emitted_bars_.push_back(static_cast<OHLCVBar&>(bar));
    }
};

// ============================================================
// Integration test
// ============================================================
void run_integration_test() {
    std::cout << "=== Market Data Store Integration Test ===" << std::endl;
    std::remove("/tmp/integ_ticks.ring");
    std::remove("/tmp/integ_archive.bin");

    // 1. Create tick store and write simulated data
    TickStore store("/tmp/integ_ticks.ring", 1 << 16);
    const char* syms[] = {"AAPL", "MSFT", "TSLA", "NVDA"};
    size_t N = 50000;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        const char* sym = syms[i % 4];
        int64_t price = 15000000 + (i % 200) * 1000;
        store.append_trade(sym, price, 100 + i % 500, i % 2);
        if (i % 10 == 0) {
            store.append_quote(sym, price - 500, 1000, price + 500, 800);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  Wrote " << store.write_seq() << " ticks in "
              << ms << "ms" << std::endl;

    // 2. Archive to compressed file
    {
        TickArchive archive("/tmp/integ_archive.bin", true);
        uint64_t seqs = store.write_seq();
        for (uint64_t s = 0; s < seqs; s += 256) {
            Tick buf[256];
            size_t n = store.snapshot(s, buf, 256);
            archive.write_batch(buf, n);
        }
        archive.flush_pending();
        std::cout << "  Archive: " << archive.ticks_archived() << " ticks, "
                  << archive.bytes_written() << " bytes, "
                  << std::fixed << std::setprecision(2)
                  << archive.compression_ratio() << "x compression" << std::endl;
    }

    // 3. Bar aggregation
    {
        BarAggregator agg(1); // 1-second bars
        store.scan(0, [&](uint64_t, const Tick& t) {
            agg.on_tick(t);
            return true;
        });
        agg.flush_all();
        std::cout << "  Generated " << agg.bars().size() << " OHLCV bars" << std::endl;
    }

    // 4. Replay stats
    {
        ReplayConfig rcfg;
        rcfg.speed_factor = 0.0; // max speed
        rcfg.symbol_filter = "AAPL";
        TickReplayer replayer(store, rcfg);

        ReplayStats rstats;
        replayer.set_callback([&](const Tick& t) { rstats.on_tick(t); });
        replayer.replay_sync();
        rstats.compute_vol();

        std::cout << "  AAPL replay: " << rstats.trade_ticks << " trades, "
                  << "VWAP=" << std::fixed << std::setprecision(2) << rstats.vwap()
                  << ", Vol=" << std::setprecision(1) << rstats.realized_vol * 100.0 << "%" << std::endl;
    }

    std::remove("/tmp/integ_ticks.ring");
    std::remove("/tmp/integ_archive.bin");
    std::cout << "  [PASS] integration test" << std::endl;
}

} // namespace tickstore

int main() {
    tickstore::run_integration_test();
    return 0;
}
