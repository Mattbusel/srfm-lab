#include "matching_engine.hpp"
#include <chrono>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace hft {

static uint64_t mono_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
        .count());
}

MatchingEngine::MatchingEngine(const std::string& symbol)
    : book_(std::make_unique<OrderBook>(symbol))
{
    for (size_t i = 0; i < kLatBuckets; ++i)
        lat_hist_[i].store(0, std::memory_order_relaxed);
}

MatchingEngine::~MatchingEngine() {
    stop();
}

void MatchingEngine::start() {
    if (running_.exchange(true)) return;
    engine_thread_ = std::thread([this]{ engine_loop(); });
}

void MatchingEngine::stop() {
    if (!running_.load()) return;
    EngineCommand cmd;
    cmd.type = EngineCommand::Type::Shutdown;
    // Spin-submit shutdown command
    while (!cmd_q_.push(cmd)) {
        std::this_thread::yield();
    }
    running_.store(false, std::memory_order_release);
    if (engine_thread_.joinable()) engine_thread_.join();
}

bool MatchingEngine::submit_order(Order* order) {
    if (!order) return false;
    EngineCommand cmd;
    cmd.type  = EngineCommand::Type::AddOrder;
    cmd.order = order;
    return cmd_q_.push(cmd);
}

bool MatchingEngine::submit_cancel(OrderId id) {
    EngineCommand cmd;
    cmd.type     = EngineCommand::Type::CancelOrder;
    cmd.order_id = id;
    return cmd_q_.push(cmd);
}

bool MatchingEngine::submit_modify(OrderId id, Price new_price, Quantity new_qty) {
    EngineCommand cmd;
    cmd.type      = EngineCommand::Type::ModifyOrder;
    cmd.order_id  = id;
    cmd.new_price = new_price;
    cmd.new_qty   = new_qty;
    return cmd_q_.push(cmd);
}

size_t MatchingEngine::drain_trades(Trade* buf, size_t max) {
    return batch_pop(trade_q_, buf, max);
}

void MatchingEngine::engine_loop() {
    // Set up trade callback to push to trade queue
    book_->set_trade_callback([this](const Trade& t) {
        trade_q_.push(t);
        stat_trades_.fetch_add(1, std::memory_order_relaxed);
        stat_volume_.fetch_add(t.qty, std::memory_order_relaxed);
    });

    EngineCommand cmd;
    while (true) {
        if (!cmd_q_.pop(cmd)) {
            // Spin-wait: yield CPU briefly
            std::this_thread::yield();
            continue;
        }
        if (cmd.type == EngineCommand::Type::Shutdown) break;
        process_command(cmd);
    }
}

void MatchingEngine::process_command(const EngineCommand& cmd) {
    uint64_t t0 = mono_ns();
    stat_rcvd_.fetch_add(1, std::memory_order_relaxed);

    switch (cmd.type) {
    case EngineCommand::Type::AddOrder: {
        Order* order = cmd.order;
        bool ok = book_->add_order(order);
        if (ok) {
            if (order->filled_qty > 0)
                stat_matched_.fetch_add(1, std::memory_order_relaxed);
            if (match_cb_) {
                MatchedOrder mo;
                mo.order   = order;
                mo.success = true;
                match_cb_(mo);
            }
        } else {
            stat_rejected_.fetch_add(1, std::memory_order_relaxed);
            if (match_cb_) {
                MatchedOrder mo;
                mo.order          = order;
                mo.success        = false;
                mo.reject_reason  = "Duplicate order ID";
                match_cb_(mo);
            }
        }
        break;
    }
    case EngineCommand::Type::CancelOrder: {
        bool ok = book_->cancel_order(cmd.order_id);
        if (ok) stat_cancelled_.fetch_add(1, std::memory_order_relaxed);
        break;
    }
    case EngineCommand::Type::ModifyOrder: {
        book_->modify_order(cmd.order_id, cmd.new_price, cmd.new_qty);
        break;
    }
    default:
        break;
    }

    uint64_t lat = mono_ns() - t0;
    record_latency(lat);
}

void MatchingEngine::record_latency(uint64_t ns) {
    // Bucket: log2-ish, each bucket covers [2^(i-1), 2^i) ns
    // Bucket 0: < 1ns, Bucket 1: 1-2ns, ... Bucket 20: ~1ms, ...
    size_t bucket = 0;
    uint64_t v = ns;
    while (v > 0 && bucket < kLatBuckets - 1) { v >>= 1; ++bucket; }
    lat_hist_[bucket].fetch_add(1, std::memory_order_relaxed);

    uint64_t prev_min = lat_min_.load(std::memory_order_relaxed);
    while (ns < prev_min && !lat_min_.compare_exchange_weak(prev_min, ns)) {}

    uint64_t prev_max = lat_max_.load(std::memory_order_relaxed);
    while (ns > prev_max && !lat_max_.compare_exchange_weak(prev_max, ns)) {}

    lat_sum_.fetch_add(ns, std::memory_order_relaxed);
    lat_count_.fetch_add(1, std::memory_order_relaxed);
}

EngineStats MatchingEngine::stats() const noexcept {
    EngineStats s{};
    s.orders_received  = stat_rcvd_.load();
    s.orders_matched   = stat_matched_.load();
    s.orders_cancelled = stat_cancelled_.load();
    s.orders_rejected  = stat_rejected_.load();
    s.trades_executed  = stat_trades_.load();
    s.total_volume     = stat_volume_.load();

    uint64_t cnt = lat_count_.load();
    if (cnt > 0) {
        s.latency_ns_min = lat_min_.load();
        s.latency_ns_max = lat_max_.load();
        s.latency_ns_avg = lat_sum_.load() / cnt;

        // p99: find bucket containing 99th percentile
        uint64_t target = cnt * 99 / 100;
        uint64_t cumulative = 0;
        for (size_t i = 0; i < kLatBuckets; ++i) {
            cumulative += lat_hist_[i].load();
            if (cumulative >= target) {
                s.latency_ns_p99 = (1ULL << i);
                break;
            }
        }
    }
    if (s.orders_received > 0)
        s.avg_fill_rate = static_cast<double>(s.orders_matched) / s.orders_received;
    return s;
}

void MatchingEngine::consume_feed(FeedHandler& feed, size_t max_per_call) {
    static thread_local std::vector<FeedEvent> buf(256);
    if (buf.size() < max_per_call) buf.resize(max_per_call);

    size_t n = feed.drain(buf.data(), max_per_call);
    for (size_t i = 0; i < n; ++i) {
        const FeedEvent& ev = buf[i];
        switch (ev.type) {
        case FeedEventType::NewOrder: {
            // Allocate order (in real system would use pool allocator)
            Order* order = new Order();
            order->id         = ev.order_id;
            std::strncpy(order->symbol, ev.symbol, 15);
            order->symbol[15] = '\0';
            order->side       = ev.side;
            order->price      = ev.price;
            order->qty        = ev.qty;
            order->timestamp  = ev.timestamp;
            order->order_type = static_cast<OrderType>(ev.flags & 0xFF);
            order->tif        = TimeInForce::GTC;
            submit_order(order);
            break;
        }
        case FeedEventType::CancelOrder:
            submit_cancel(ev.order_id);
            break;
        case FeedEventType::ModifyOrder:
            submit_modify(ev.order_id, ev.new_price, ev.new_qty);
            break;
        default:
            break;
        }
    }
}

// ---- MultiSymbolEngine ----

MatchingEngine& MultiSymbolEngine::get_or_create(const std::string& symbol) {
    auto it = engines_.find(symbol);
    if (it != engines_.end()) return *it->second;
    auto [ins, ok] = engines_.emplace(symbol, std::make_unique<MatchingEngine>(symbol));
    return *ins->second;
}

MatchingEngine* MultiSymbolEngine::get(const std::string& symbol) {
    auto it = engines_.find(symbol);
    return it != engines_.end() ? it->second.get() : nullptr;
}

void MultiSymbolEngine::start_all() {
    for (auto& [sym, eng] : engines_) eng->start();
}

void MultiSymbolEngine::stop_all() {
    for (auto& [sym, eng] : engines_) eng->stop();
}

std::vector<std::string> MultiSymbolEngine::symbols() const {
    std::vector<std::string> v;
    v.reserve(engines_.size());
    for (auto& [sym, _] : engines_) v.push_back(sym);
    return v;
}

} // namespace hft
