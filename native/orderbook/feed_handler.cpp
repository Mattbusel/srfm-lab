#include "feed_handler.hpp"
#include <chrono>
#include <cstring>
#include <algorithm>
#include <thread>

namespace hft {

static Timestamp feed_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
               .count();
}

FeedHandler::FeedHandler(const FeedConfig& cfg)
    : cfg_(cfg)
{
    mid_price_.store(cfg.initial_price);
}

FeedHandler::~FeedHandler() {
    stop();
}

void FeedHandler::start() {
    if (running_.exchange(true)) return; // already running
    sim_thread_ = std::thread([this]{ simulation_loop(); });
}

void FeedHandler::stop() {
    running_.store(false, std::memory_order_release);
    if (sim_thread_.joinable()) sim_thread_.join();
}

size_t FeedHandler::drain(FeedEvent* buf, size_t max_events) {
    return batch_pop(queue_, buf, max_events);
}

void FeedHandler::simulation_loop() {
    std::mt19937_64 rng(static_cast<uint64_t>(cfg_.seed));
    std::exponential_distribution<double> arrive_dist(cfg_.lambda_new);
    std::exponential_distribution<double> cancel_dist(cfg_.lambda_cancel);
    std::exponential_distribution<double> trade_dist(cfg_.lambda_trade);

    // GBM parameters
    const double dt     = 1.0 / (252.0 * 6.5 * 3600.0); // 1 second in trading time
    const double sigma  = cfg_.daily_vol / std::sqrt(252.0 * 6.5 * 3600.0);
    const double mu     = 0.0; // no drift for simulation

    std::normal_distribution<double> gbm_noise(0.0, 1.0);
    std::uniform_real_distribution<double> side_dist(0.0, 1.0);
    std::uniform_real_distribution<double> type_dist(0.0, 1.0);

    uint64_t tick       = 0;
    double   next_new    = arrive_dist(rng);
    double   next_cancel = cancel_dist(rng);
    double   next_trade  = trade_dist(rng);
    double   t_seconds   = 0.0;

    const double tick_interval_us = 100.0; // simulate at 100 µs resolution
    const double tick_interval_s  = tick_interval_us * 1e-6;
    const double tick_sigma       = cfg_.daily_vol * std::sqrt(tick_interval_s / (252.0 * 6.5 * 3600.0));

    while (running_.load(std::memory_order_relaxed)) {
        // Advance time
        t_seconds += tick_interval_s;
        ++tick;

        // GBM price update
        double mid = mid_price_.load(std::memory_order_relaxed);
        double dW  = gbm_noise(rng);
        mid *= std::exp((mu - 0.5 * tick_sigma * tick_sigma) * tick_interval_s
                        + tick_sigma * dW);
        // Round to tick size
        mid = std::round(mid / cfg_.tick_size) * cfg_.tick_size;
        mid = std::max(mid, cfg_.tick_size);
        mid_price_.store(mid, std::memory_order_relaxed);

        // Check new order arrivals
        if (t_seconds >= next_new) {
            generate_new_order(rng, mid);
            next_new = t_seconds + arrive_dist(rng);
        }

        // Check cancels
        if (t_seconds >= next_cancel) {
            generate_cancel(rng, tick);
            next_cancel = t_seconds + cancel_dist(rng);
        }

        // Check trades / market orders
        if (t_seconds >= next_trade) {
            generate_trade(rng, mid);
            next_trade = t_seconds + trade_dist(rng);
        }

        // Heartbeat every 1000 ticks
        if (tick % 1000 == 0) {
            FeedEvent hb;
            hb.type      = FeedEventType::Heartbeat;
            hb.sequence  = next_seq();
            hb.timestamp = feed_now_ns();
            push_event(hb);
        }

        // Throttle simulation to avoid burning CPU when not consuming
        if (queue_.size() > FeedHandler::kQueueCap * 3 / 4) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void FeedHandler::generate_new_order(std::mt19937_64& rng, double mid) {
    std::uniform_real_distribution<double> side_dist(0.0, 1.0);
    std::uniform_real_distribution<double> offset_dist(0.0, 5.0 * cfg_.tick_size);
    std::uniform_real_distribution<double> qty_dist(cfg_.lot_size, cfg_.max_qty);
    std::uniform_real_distribution<double> type_dist(0.0, 1.0);

    FeedEvent ev;
    ev.type      = FeedEventType::NewOrder;
    ev.sequence  = next_seq();
    ev.timestamp = feed_now_ns();
    ev.order_id  = next_order_id();
    std::strncpy(ev.symbol, cfg_.symbol.c_str(), 15);
    ev.symbol[15] = '\0';

    bool is_buy  = side_dist(rng) < 0.5;
    ev.side      = is_buy ? Side::Buy : Side::Sell;

    double half_spread = mid * cfg_.spread_bps / 20000.0;
    double offset      = offset_dist(rng);

    if (is_buy)
        ev.price = double_to_price(mid - half_spread - offset);
    else
        ev.price = double_to_price(mid + half_spread + offset);

    // Snap to tick
    Price tick_p = double_to_price(cfg_.tick_size);
    if (tick_p > 0) ev.price = (ev.price / tick_p) * tick_p;
    ev.price = std::max(ev.price, Price(1));

    double raw_qty = qty_dist(rng);
    raw_qty = std::round(raw_qty / cfg_.lot_size) * cfg_.lot_size;
    ev.qty   = static_cast<Quantity>(std::max(raw_qty, cfg_.lot_size));

    // Order type: 80% limit, 15% iceberg, 5% stop
    double t = type_dist(rng);
    if (t < 0.80)       ev.flags = static_cast<uint32_t>(OrderType::Limit);
    else if (t < 0.95)  ev.flags = static_cast<uint32_t>(OrderType::Iceberg);
    else                ev.flags = static_cast<uint32_t>(OrderType::Stop);

    push_event(ev);
}

void FeedHandler::generate_cancel(std::mt19937_64& rng, uint64_t seq) {
    // Cancel a "recent" order — use sequence to pick one
    uint64_t base_id = order_id_gen_.load(std::memory_order_relaxed);
    if (base_id < 1000010) return;

    std::uniform_int_distribution<uint64_t> id_dist(1000000, base_id - 1);
    FeedEvent ev;
    ev.type      = FeedEventType::CancelOrder;
    ev.sequence  = next_seq();
    ev.timestamp = feed_now_ns();
    ev.order_id  = id_dist(rng);
    std::strncpy(ev.symbol, cfg_.symbol.c_str(), 15);
    ev.symbol[15] = '\0';
    push_event(ev);
}

void FeedHandler::generate_trade(std::mt19937_64& rng, double mid) {
    std::uniform_real_distribution<double> qty_dist(cfg_.lot_size, cfg_.max_qty * 2.0);
    std::uniform_real_distribution<double> side_dist(0.0, 1.0);
    std::uniform_real_distribution<double> slip_dist(0.0, 2.0 * cfg_.tick_size);

    FeedEvent ev;
    ev.type      = FeedEventType::Trade;
    ev.sequence  = next_seq();
    ev.timestamp = feed_now_ns();
    ev.order_id  = next_order_id(); // aggressor order
    std::strncpy(ev.symbol, cfg_.symbol.c_str(), 15);
    ev.symbol[15] = '\0';

    bool is_buy  = side_dist(rng) < 0.5;
    ev.side      = is_buy ? Side::Buy : Side::Sell;

    double slip   = slip_dist(rng);
    double price  = is_buy ? mid + slip : mid - slip;
    ev.price      = double_to_price(std::max(price, cfg_.tick_size));
    double raw_qty = qty_dist(rng);
    raw_qty        = std::round(raw_qty / cfg_.lot_size) * cfg_.lot_size;
    ev.qty         = static_cast<Quantity>(std::max(raw_qty, cfg_.lot_size));

    push_event(ev);
}

} // namespace hft
