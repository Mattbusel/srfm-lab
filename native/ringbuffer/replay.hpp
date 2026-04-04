#pragma once
// Tick replay engine: reads from a TickStore ring buffer and replays
// ticks at their original timestamps (or accelerated/decelerated).

#include "tick_store.cpp"
#include <functional>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

namespace tickstore {

using TickCallback = std::function<void(const Tick&)>;

struct ReplayConfig {
    double   speed_factor = 1.0;   // 1.0 = real-time, 10.0 = 10x faster, 0 = max speed
    bool     loop        = false;  // loop at end of data
    uint64_t start_seq   = 0;      // starting sequence number
    uint64_t end_seq     = UINT64_MAX; // ending sequence number
    uint64_t start_ts    = 0;      // filter: only replay ticks >= start_ts
    uint64_t end_ts      = UINT64_MAX;
    std::string symbol_filter;     // empty = all symbols
};

class TickReplayer {
public:
    TickReplayer(TickStore& store, const ReplayConfig& cfg = {})
        : store_(store), cfg_(cfg) {}

    ~TickReplayer() { stop(); }

    void set_callback(TickCallback cb) { callback_ = std::move(cb); }
    void set_heartbeat(std::function<void(uint64_t seq, uint64_t total)> hb)
    { heartbeat_ = std::move(hb); }

    void start() {
        if (running_.exchange(true)) return;
        replay_thread_ = std::thread([this]{ replay_loop(); });
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        if (replay_thread_.joinable()) replay_thread_.join();
    }

    bool is_running() const noexcept { return running_.load(); }
    uint64_t ticks_replayed() const noexcept { return ticks_replayed_.load(); }
    uint64_t ticks_skipped() const noexcept  { return ticks_skipped_.load(); }

    // Synchronous replay (no thread): returns when done
    void replay_sync() {
        uint64_t total = store_.write_seq();
        uint64_t end   = std::min(cfg_.end_seq, total);
        uint64_t ref_real_ns  = 0; // wall clock at first tick
        uint64_t ref_tick_ts  = 0; // tick timestamp at first tick

        for (uint64_t seq = cfg_.start_seq; seq < end; ++seq) {
            Tick t;
            if (!store_.read(seq, t)) break;

            // Timestamp filter
            if (t.timestamp < static_cast<int64_t>(cfg_.start_ts)) {
                ++ticks_skipped_; continue;
            }
            if (t.timestamp > static_cast<int64_t>(cfg_.end_ts)) break;

            // Symbol filter
            if (!cfg_.symbol_filter.empty()) {
                if (std::strncmp(t.symbol, cfg_.symbol_filter.c_str(), 15) != 0) {
                    ++ticks_skipped_; continue;
                }
            }

            // Timing
            if (cfg_.speed_factor > 0 && ref_real_ns == 0) {
                ref_real_ns = wall_ns();
                ref_tick_ts = static_cast<uint64_t>(t.timestamp);
            }
            if (cfg_.speed_factor > 0 && ref_real_ns > 0) {
                uint64_t tick_elapsed = static_cast<uint64_t>(t.timestamp) - ref_tick_ts;
                uint64_t real_elapsed = static_cast<uint64_t>(
                    tick_elapsed / cfg_.speed_factor);
                uint64_t target = ref_real_ns + real_elapsed;
                uint64_t now    = wall_ns();
                if (target > now) {
                    std::this_thread::sleep_for(
                        std::chrono::nanoseconds(target - now));
                }
            }

            if (callback_) callback_(t);
            ++ticks_replayed_;

            if (heartbeat_ && ticks_replayed_ % 10000 == 0)
                heartbeat_(seq, total);
        }
    }

private:
    TickStore&          store_;
    ReplayConfig        cfg_;
    TickCallback        callback_;
    std::function<void(uint64_t, uint64_t)> heartbeat_;
    std::thread         replay_thread_;
    std::atomic<bool>   running_{false};
    std::atomic<uint64_t> ticks_replayed_{0};
    std::atomic<uint64_t> ticks_skipped_{0};

    void replay_loop() {
        do {
            replay_sync();
        } while (cfg_.loop && running_.load());
        running_.store(false);
    }

    static uint64_t wall_ns() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count());
    }
};

// ============================================================
// Statistics collector that processes replayed ticks
// ============================================================
struct ReplayStats {
    uint64_t total_ticks     = 0;
    uint64_t trade_ticks     = 0;
    uint64_t quote_ticks     = 0;
    double   total_volume    = 0.0;
    double   vwap_num        = 0.0;
    uint64_t vwap_den        = 0;
    double   min_price       = 1e15;
    double   max_price       = 0.0;
    double   first_price     = 0.0;
    double   last_price      = 0.0;
    uint64_t first_ts        = 0;
    uint64_t last_ts         = 0;
    double   realized_vol    = 0.0; // annualized
    std::vector<double> log_returns;

    void on_tick(const Tick& t) {
        ++total_ticks;
        if (t.tick_type == 0) {
            ++trade_ticks;
            double p = t.price / 100000.0;
            double q = static_cast<double>(t.qty);
            total_volume += q;
            vwap_num     += p * q;
            vwap_den     += t.qty;
            if (p < min_price) min_price = p;
            if (p > max_price) max_price = p;
            if (first_price == 0.0) { first_price = p; first_ts = t.timestamp; }
            if (last_price > 0) log_returns.push_back(std::log(p / last_price));
            last_price = p;
            last_ts    = t.timestamp;
        } else {
            ++quote_ticks;
        }
    }

    double vwap() const { return vwap_den > 0 ? vwap_num / vwap_den : 0.0; }
    double return_pct() const {
        return first_price > 0 ? (last_price / first_price - 1.0) * 100.0 : 0.0;
    }
    double duration_hours() const {
        if (first_ts == 0 || last_ts == 0) return 0.0;
        return (last_ts - first_ts) / 3.6e12; // ns to hours
    }

    void compute_vol() {
        if (log_returns.size() < 2) return;
        const size_t n = log_returns.size();
        double mean = 0.0;
        for (auto r : log_returns) mean += r;
        mean /= n;
        double var = 0.0;
        for (auto r : log_returns) var += (r - mean) * (r - mean);
        var /= (n - 1);
        // Annualize: assuming ticks_per_year ~ 1M * 252
        const double tpy = 1e6 * 252.0;
        realized_vol = std::sqrt(var * tpy);
    }

    void print() const {
        printf("=== Replay Stats ===\n");
        printf("  Total ticks:  %llu\n", (unsigned long long)total_ticks);
        printf("  Trade ticks:  %llu\n", (unsigned long long)trade_ticks);
        printf("  Quote ticks:  %llu\n", (unsigned long long)quote_ticks);
        printf("  First price:  %.5f\n", first_price);
        printf("  Last price:   %.5f\n", last_price);
        printf("  Return:       %.2f%%\n", return_pct());
        printf("  VWAP:         %.5f\n", vwap());
        printf("  Volume:       %.0f\n", total_volume);
        printf("  Duration:     %.2fh\n", duration_hours());
        printf("  Realized vol: %.1f%%\n", realized_vol * 100.0);
    }
};

} // namespace tickstore
