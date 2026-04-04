#pragma once
#include "orderbook.hpp"
#include "feed_handler.hpp"
#include "lockfree_queue.hpp"
#include <unordered_map>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <vector>
#include <string>

namespace hft {

struct EngineStats {
    uint64_t orders_received;
    uint64_t orders_matched;
    uint64_t orders_cancelled;
    uint64_t orders_rejected;
    uint64_t trades_executed;
    uint64_t total_volume;
    double   avg_fill_rate;       // fraction of orders fully filled
    double   throughput_ops;      // orders per second
    uint64_t latency_ns_min;      // min matching latency
    uint64_t latency_ns_max;
    uint64_t latency_ns_avg;
    uint64_t latency_ns_p99;
};

struct MatchedOrder {
    Order*            order;
    std::vector<Trade> trades;
    bool               success;
    std::string        reject_reason;
};

using MatchCallback  = std::function<void(const MatchedOrder&)>;

// Command submitted to the engine
struct EngineCommand {
    enum class Type : uint8_t {
        AddOrder    = 0,
        CancelOrder = 1,
        ModifyOrder = 2,
        Shutdown    = 3
    };
    Type     type;
    Order*   order;       // for AddOrder / ModifyOrder
    OrderId  order_id;    // for CancelOrder
    Price    new_price;   // for ModifyOrder
    Quantity new_qty;     // for ModifyOrder
};

// Single-symbol matching engine
// Runs a background thread consuming from command queue
class MatchingEngine {
public:
    static constexpr size_t kCmdQueueCap   = 65536;
    static constexpr size_t kTradeQueueCap = 65536;

    using CmdQueue   = SPSCQueue<EngineCommand, kCmdQueueCap>;
    using TradeQueue = SPSCQueue<Trade, kTradeQueueCap>;

    explicit MatchingEngine(const std::string& symbol);
    ~MatchingEngine();

    // Thread control
    void start();
    void stop();
    bool is_running() const noexcept { return running_.load(); }

    // Submit commands (from producer thread)
    bool submit_order(Order* order);
    bool submit_cancel(OrderId id);
    bool submit_modify(OrderId id, Price new_price, Quantity new_qty);

    // Drain trades (from consumer thread)
    size_t drain_trades(Trade* buf, size_t max);
    bool   pop_trade(Trade& t) { return trade_q_.pop(t); }

    // Read-only book access (not thread-safe; call from engine thread or after stop)
    const OrderBook& book() const noexcept { return *book_; }
    EngineStats      stats() const noexcept;

    // Callbacks
    void set_match_callback(MatchCallback cb) { match_cb_ = std::move(cb); }

    // Feed handler integration: consume events from feed and translate to commands
    void consume_feed(FeedHandler& feed, size_t max_per_call = 256);

private:
    std::unique_ptr<OrderBook>  book_;
    CmdQueue                    cmd_q_;
    TradeQueue                  trade_q_;
    std::thread                 engine_thread_;
    std::atomic<bool>           running_{false};
    MatchCallback               match_cb_;

    // Stats (updated by engine thread)
    std::atomic<uint64_t>       stat_rcvd_{0};
    std::atomic<uint64_t>       stat_matched_{0};
    std::atomic<uint64_t>       stat_cancelled_{0};
    std::atomic<uint64_t>       stat_rejected_{0};
    std::atomic<uint64_t>       stat_trades_{0};
    std::atomic<uint64_t>       stat_volume_{0};

    // Latency histogram (coarse buckets in ns)
    static constexpr size_t kLatBuckets = 1024;
    std::atomic<uint64_t>   lat_hist_[kLatBuckets]{};
    std::atomic<uint64_t>   lat_min_{UINT64_MAX};
    std::atomic<uint64_t>   lat_max_{0};
    std::atomic<uint64_t>   lat_sum_{0};
    std::atomic<uint64_t>   lat_count_{0};

    void engine_loop();
    void process_command(const EngineCommand& cmd);
    void record_latency(uint64_t ns);
};

// Multi-symbol engine dispatcher
class MultiSymbolEngine {
public:
    MultiSymbolEngine() = default;
    ~MultiSymbolEngine() { stop_all(); }

    MatchingEngine& get_or_create(const std::string& symbol);
    MatchingEngine* get(const std::string& symbol);

    void start_all();
    void stop_all();

    size_t symbol_count() const noexcept { return engines_.size(); }
    std::vector<std::string> symbols() const;

private:
    std::unordered_map<std::string, std::unique_ptr<MatchingEngine>> engines_;
};

} // namespace hft
