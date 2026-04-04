#include "ringbuffer.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <functional>
#include <atomic>
#include <chrono>

namespace tickstore {

// Tick represents a single market data event
struct Tick {
    uint64_t timestamp;    // nanoseconds since epoch
    char     symbol[16];   // null-terminated
    int64_t  price;        // fixed-point price * 100000
    uint64_t qty;          // quantity
    uint8_t  side;         // 0=buy, 1=sell
    uint8_t  tick_type;    // 0=trade, 1=bid, 2=ask, 3=midpoint
    uint8_t  exchange;     // exchange ID
    uint8_t  flags;        // misc flags
    uint64_t trade_id;     // unique trade ID
    uint64_t order_id;     // order ID
    int64_t  bid_price;    // best bid at time of tick
    int64_t  ask_price;    // best ask at time of tick
    uint64_t bid_qty;      // bid depth at BBO
    uint64_t ask_qty;      // ask depth at BBO
    uint32_t seq_num;      // exchange sequence number
    uint32_t venue_seq;    // venue sequence number

    Tick() noexcept { std::memset(this, 0, sizeof(Tick)); }

    void set_symbol(const char* s) noexcept {
        std::strncpy(symbol, s, 15);
        symbol[15] = '\0';
    }

    double price_d()     const noexcept { return price / 100000.0; }
    double bid_price_d() const noexcept { return bid_price / 100000.0; }
    double ask_price_d() const noexcept { return ask_price / 100000.0; }
    double spread_d()    const noexcept { return (ask_price - bid_price) / 100000.0; }
    double mid_d()       const noexcept { return (bid_price + ask_price) / 200000.0; }
};

static_assert(sizeof(Tick) == 128, "Tick must be 128 bytes for cache alignment");

// TickStore wraps RingBuffer<Tick> with query and aggregation APIs
class TickStore {
public:
    static constexpr size_t kDefaultCapacity = 1 << 20; // 1M ticks

    explicit TickStore(const std::string& path, size_t capacity = kDefaultCapacity)
        : ring_(path, capacity)
    {}

    // O(1) append
    uint64_t append(const Tick& t) noexcept {
        return ring_.push(t);
    }

    // Convenience: append a trade tick
    uint64_t append_trade(const char* symbol, int64_t price, uint64_t qty,
                           uint8_t side, uint64_t trade_id = 0) noexcept
    {
        Tick t;
        t.timestamp = now_ns();
        t.set_symbol(symbol);
        t.price     = price;
        t.qty       = qty;
        t.side      = side;
        t.tick_type = 0; // trade
        t.trade_id  = trade_id;
        t.seq_num   = seq_.fetch_add(1, std::memory_order_relaxed);
        return ring_.push(t);
    }

    // Append a quote tick
    uint64_t append_quote(const char* symbol,
                           int64_t bid, uint64_t bid_qty,
                           int64_t ask, uint64_t ask_qty) noexcept
    {
        Tick t;
        t.timestamp = now_ns();
        t.set_symbol(symbol);
        t.bid_price = bid;
        t.bid_qty   = bid_qty;
        t.ask_price = ask;
        t.ask_qty   = ask_qty;
        t.price     = (bid + ask) / 2;
        t.tick_type = 1; // quote
        t.seq_num   = seq_.fetch_add(1, std::memory_order_relaxed);
        return ring_.push(t);
    }

    // Read tick at sequence number seq
    bool read(uint64_t seq, Tick& out) const noexcept {
        return ring_.read(seq, out);
    }

    // Latest tick
    bool latest(Tick& out) const noexcept {
        return ring_.peek_latest(out);
    }

    // Iterate over ticks in [start_seq, end_seq)
    // Calls callback(seq, tick) for each; stops if callback returns false
    void scan(uint64_t start_seq,
              std::function<bool(uint64_t, const Tick&)> cb) const
    {
        uint64_t end_seq = ring_.write_seq();
        for (uint64_t s = start_seq; s < end_seq; ++s) {
            Tick t;
            if (!ring_.read(s, t)) break;
            if (!cb(s, t)) break;
        }
    }

    // VWAP over last N ticks of given symbol
    double vwap(const char* symbol, size_t n) const noexcept {
        uint64_t w = ring_.write_seq();
        if (w == 0) return 0.0;
        uint64_t start = (w > n) ? w - n : 0;

        double num = 0.0;
        uint64_t den = 0;
        for (uint64_t s = start; s < w; ++s) {
            Tick t;
            if (!ring_.read(s, t)) continue;
            if (std::strncmp(t.symbol, symbol, 15) != 0) continue;
            if (t.tick_type != 0) continue; // trades only
            num += t.price * static_cast<double>(t.qty);
            den += t.qty;
        }
        return den > 0 ? num / den / 100000.0 : 0.0;
    }

    // Count ticks in last window
    size_t count_recent(size_t window) const noexcept {
        uint64_t w = ring_.write_seq();
        return static_cast<size_t>(std::min(w, static_cast<uint64_t>(window)));
    }

    // Snapshot: copy up to `count` ticks starting from `start_seq`
    size_t snapshot(uint64_t start_seq, Tick* buf, size_t count) const noexcept {
        return ring_.snapshot(start_seq, buf, count);
    }

    uint64_t write_seq()    const noexcept { return ring_.write_seq(); }
    size_t   capacity()     const noexcept { return ring_.capacity(); }
    void     sync()               noexcept { ring_.sync(); }

private:
    RingBuffer<Tick>       ring_;
    std::atomic<uint32_t>  seq_{0};

    static uint64_t now_ns() noexcept {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
            .count());
    }
};

// Multi-symbol tick store: routes ticks to per-symbol ring buffers
class MultiSymbolStore {
public:
    MultiSymbolStore(const std::string& base_dir, size_t ticks_per_symbol = 1 << 18)
        : base_dir_(base_dir), ticks_per_sym_(ticks_per_symbol) {}

    TickStore& get_or_create(const std::string& symbol) {
        auto it = stores_.find(symbol);
        if (it != stores_.end()) return *it->second;
        std::string path = base_dir_ + "/" + symbol + ".ring";
        auto [ins, ok] = stores_.emplace(symbol,
            std::make_unique<TickStore>(path, ticks_per_sym_));
        return *ins->second;
    }

    void append_tick(const Tick& t) {
        std::string sym(t.symbol, strnlen(t.symbol, 15));
        get_or_create(sym).append(t);
    }

    size_t symbol_count() const noexcept { return stores_.size(); }

    void sync_all() noexcept {
        for (auto& [sym, store] : stores_) store->sync();
    }

private:
    std::string                                             base_dir_;
    size_t                                                  ticks_per_sym_;
    std::unordered_map<std::string, std::unique_ptr<TickStore>> stores_;
};

} // namespace tickstore
