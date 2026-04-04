#pragma once
#include "order.hpp"
#include "lockfree_queue.hpp"
#include <atomic>
#include <random>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <cmath>

namespace hft {

enum class FeedEventType : uint8_t {
    NewOrder    = 0,
    CancelOrder = 1,
    ModifyOrder = 2,
    Trade       = 3,
    QuoteUpdate = 4,
    Heartbeat   = 5
};

struct FeedEvent {
    FeedEventType type;
    uint64_t      sequence;
    Timestamp     timestamp;
    OrderId       order_id;
    char          symbol[16];
    Side          side;
    Price         price;
    Quantity      qty;
    Price         new_price;
    Quantity      new_qty;
    uint32_t      flags;

    FeedEvent() noexcept
        : type(FeedEventType::Heartbeat), sequence(0), timestamp(0),
          order_id(0), side(Side::Buy), price(0), qty(0),
          new_price(0), new_qty(0), flags(0)
    { symbol[0] = '\0'; }
};

struct FeedConfig {
    std::string symbol;
    double initial_price;      // starting mid price
    double daily_vol;          // annualized vol (e.g. 0.20 = 20%)
    double lambda_new;         // Poisson arrival rate for new orders (per second)
    double lambda_cancel;      // Poisson arrival rate for cancels
    double lambda_trade;       // Poisson rate for market orders / trades
    double tick_size;          // price granularity
    double spread_bps;         // initial bid-ask spread in bps
    double lot_size;           // minimum order quantity
    double max_qty;            // max order quantity
    int    depth_levels;       // simulated book depth
    int    seed;               // RNG seed
};

// Simulated market data feed using Poisson order arrivals
// and GBM price diffusion
class FeedHandler {
public:
    static constexpr size_t kQueueCap = 65536;
    using EventQueue = SPSCQueue<FeedEvent, kQueueCap>;

    explicit FeedHandler(const FeedConfig& cfg);
    ~FeedHandler();

    // Start/stop background simulation thread
    void start();
    void stop();
    bool is_running() const noexcept { return running_.load(std::memory_order_acquire); }

    // Drain events from the queue (consumer side)
    size_t drain(FeedEvent* buf, size_t max_events);

    // Single event peek/pop
    bool pop_event(FeedEvent& ev) { return queue_.pop(ev); }
    bool has_events() const       { return !queue_.empty(); }

    // Stats
    uint64_t events_generated() const noexcept { return events_gen_.load(); }
    uint64_t events_dropped()   const noexcept { return events_drop_.load(); }
    double   current_mid()      const noexcept { return mid_price_.load(); }

    const FeedConfig& config() const noexcept { return cfg_; }

    // For testing: inject a specific event
    bool inject(const FeedEvent& ev) { return queue_.push(ev); }

private:
    FeedConfig              cfg_;
    EventQueue              queue_;
    std::thread             sim_thread_;
    std::atomic<bool>       running_{false};
    std::atomic<uint64_t>   seq_gen_{1};
    std::atomic<uint64_t>   order_id_gen_{1000000};
    std::atomic<uint64_t>   events_gen_{0};
    std::atomic<uint64_t>   events_drop_{0};
    std::atomic<double>     mid_price_;

    void simulation_loop();
    void generate_new_order(std::mt19937_64& rng, double mid);
    void generate_cancel(std::mt19937_64& rng, uint64_t seq);
    void generate_trade(std::mt19937_64& rng, double mid);

    bool push_event(const FeedEvent& ev) {
        events_gen_.fetch_add(1, std::memory_order_relaxed);
        if (!queue_.push(ev)) {
            events_drop_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        return true;
    }

    uint64_t next_seq()      noexcept { return seq_gen_.fetch_add(1, std::memory_order_relaxed); }
    uint64_t next_order_id() noexcept { return order_id_gen_.fetch_add(1, std::memory_order_relaxed); }
};

} // namespace hft
