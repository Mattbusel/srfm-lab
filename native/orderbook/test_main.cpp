#include "orderbook.hpp"
#include "lockfree_queue.hpp"
#include "market_impact.hpp"
#include "feed_handler.hpp"
#include "matching_engine.hpp"

#include <iostream>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <vector>
#include <thread>

using namespace hft;

void test_basic_order_book() {
    std::cout << "=== Test: Basic Order Book ===" << std::endl;
    OrderBook book("AAPL");

    // Add some bids
    Order b1(1, "AAPL", Side::Buy, OrderType::Limit, TimeInForce::GTC,
              double_to_price(150.00), 100, 0);
    Order b2(2, "AAPL", Side::Buy, OrderType::Limit, TimeInForce::GTC,
              double_to_price(149.90), 200, 0);
    Order b3(3, "AAPL", Side::Buy, OrderType::Limit, TimeInForce::GTC,
              double_to_price(150.00), 150, 0); // same level as b1

    // Add some asks
    Order a1(4, "AAPL", Side::Sell, OrderType::Limit, TimeInForce::GTC,
              double_to_price(150.10), 100, 0);
    Order a2(5, "AAPL", Side::Sell, OrderType::Limit, TimeInForce::GTC,
              double_to_price(150.20), 300, 0);

    book.add_order(&b1);
    book.add_order(&b2);
    book.add_order(&b3);
    book.add_order(&a1);
    book.add_order(&a2);

    auto bb = book.best_bid();
    auto ba = book.best_ask();
    assert(bb.has_value() && price_to_double(*bb) == 150.00);
    assert(ba.has_value() && price_to_double(*ba) == 150.10);

    double spread = *book.spread();
    std::cout << "  Best bid: " << price_to_double(*bb)
              << "  Best ask: " << price_to_double(*ba)
              << "  Spread: "   << spread << std::endl;

    auto stats = book.stats();
    std::cout << "  Bid depth: " << stats.bid_depth
              << "  Ask depth: " << stats.ask_depth
              << "  Imbalance: " << std::fixed << std::setprecision(3)
              << stats.imbalance << std::endl;

    // Test cancel
    bool ok = book.cancel_order(2);
    assert(ok);
    assert(book.bid_levels_count() == 1); // only 150.00 remains

    std::cout << "  [PASS] basic order book" << std::endl;
}

void test_matching() {
    std::cout << "=== Test: Price-Time Priority Matching ===" << std::endl;
    OrderBook book("TSLA");

    // Seed book
    Order b1(1, "TSLA", Side::Buy, OrderType::Limit, TimeInForce::GTC,
              double_to_price(200.00), 100, 0);
    Order b2(2, "TSLA", Side::Buy, OrderType::Limit, TimeInForce::GTC,
              double_to_price(200.00), 50, 0);
    book.add_order(&b1);
    book.add_order(&b2);

    // Aggressive sell
    Order sell(3, "TSLA", Side::Sell, OrderType::Limit, TimeInForce::GTC,
               double_to_price(199.90), 120, 0);
    book.add_order(&sell);

    // b1 should be filled 100, b2 partially 20, sell fully 120
    assert(b1.status == OrderStatus::Filled);
    assert(b1.filled_qty == 100);
    assert(b2.filled_qty == 20);
    assert(b2.status == OrderStatus::PartialFill);
    assert(sell.status == OrderStatus::Filled);
    assert(sell.filled_qty == 120);

    auto trades = book.trade_history();
    assert(trades.size() == 2);
    std::cout << "  Trades: " << trades.size() << std::endl;
    for (auto& t : trades)
        std::cout << "    trade id=" << t.trade_id
                  << " price=" << price_to_double(t.price)
                  << " qty=" << t.qty << std::endl;

    std::cout << "  [PASS] price-time priority matching" << std::endl;
}

void test_ioc_fok() {
    std::cout << "=== Test: IOC / FOK ===" << std::endl;
    OrderBook book("SPY");

    Order a1(1, "SPY", Side::Sell, OrderType::Limit, TimeInForce::GTC,
              double_to_price(400.00), 50, 0);
    book.add_order(&a1);

    // IOC: partially fills, remainder cancelled
    Order ioc(2, "SPY", Side::Buy, OrderType::Limit, TimeInForce::IOC,
               double_to_price(400.00), 100, 0);
    book.add_order(&ioc);
    assert(ioc.filled_qty == 50);
    assert(ioc.status == OrderStatus::Cancelled);

    // FOK: can't fully fill → reject
    Order a2(3, "SPY", Side::Sell, OrderType::Limit, TimeInForce::GTC,
              double_to_price(400.00), 30, 0);
    book.add_order(&a2);

    Order fok(4, "SPY", Side::Buy, OrderType::Limit, TimeInForce::FOK,
               double_to_price(400.00), 100, 0);
    book.add_order(&fok);
    assert(fok.filled_qty == 0);
    assert(fok.status == OrderStatus::Cancelled);

    std::cout << "  [PASS] IOC/FOK" << std::endl;
}

void test_lockfree_queue() {
    std::cout << "=== Test: Lock-Free Queue ===" << std::endl;
    SPSCQueue<int, 1024> q;
    assert(q.empty());

    for (int i = 0; i < 512; ++i) assert(q.push(i));
    assert(q.size() == 512);

    int v;
    for (int i = 0; i < 512; ++i) {
        assert(q.pop(v));
        assert(v == i);
    }
    assert(q.empty());

    // Multi-threaded producer/consumer
    std::atomic<int> sum_produced{0}, sum_consumed{0};
    SPSCQueue<int, 4096> q2;
    std::thread producer([&]{
        for (int i = 1; i <= 100000; ++i) {
            while (!q2.push(i)) std::this_thread::yield();
            sum_produced.fetch_add(i, std::memory_order_relaxed);
        }
    });
    std::thread consumer([&]{
        int val, count = 0;
        while (count < 100000) {
            if (q2.pop(val)) {
                sum_consumed.fetch_add(val, std::memory_order_relaxed);
                ++count;
            } else {
                std::this_thread::yield();
            }
        }
    });
    producer.join();
    consumer.join();
    assert(sum_produced.load() == sum_consumed.load());
    std::cout << "  Producer sum=" << sum_produced.load()
              << " Consumer sum=" << sum_consumed.load() << std::endl;
    std::cout << "  [PASS] lock-free SPSC queue" << std::endl;
}

void test_market_impact() {
    std::cout << "=== Test: Market Impact Models ===" << std::endl;
    // Almgren-Chriss
    AlmgrenChrissParams p{};
    p.sigma   = 0.02;    // 2% daily vol
    p.gamma   = 2.5e-7;  // permanent impact
    p.eta     = 2.5e-6;  // temporary impact
    p.epsilon = 0.0625;  // 1/16 spread
    p.lambda  = 1e-6;    // risk aversion
    p.tau     = 1.0/6.5; // 1 hour in trading days
    p.N       = 5;       // 5 hourly slices

    AlmgrenChriss ac(p);
    auto sched = ac.optimal_schedule(1000000.0); // 1M shares
    std::cout << "  AC Expected cost: $" << std::fixed << std::setprecision(0)
              << sched.expected_cost << std::endl;
    std::cout << "  AC Variance cost: " << sched.variance_cost << std::endl;
    assert(sched.trade_list.size() == 5);
    double total_trades = 0;
    for (auto x : sched.trade_list) total_trades += x;
    assert(std::fabs(total_trades - 1000000.0) < 1.0);

    // Kyle Lambda
    KyleLambda kyle(50);
    for (int i = 0; i < 50; ++i) {
        double flow = (i % 2 == 0) ? 1000.0 : -1000.0;
        double dp   = (i % 2 == 0) ? 0.01 : -0.01;
        kyle.update(dp, flow);
    }
    double lambda = kyle.estimate();
    std::cout << "  Kyle lambda: " << lambda << std::endl;
    assert(lambda > 0);

    // Amihud
    AmihudIlliquidity amihud(50);
    for (int i = 0; i < 50; ++i) amihud.update(0.01, 1e6);
    std::cout << "  Amihud illiq: " << amihud.estimate() << std::endl;

    std::cout << "  [PASS] market impact" << std::endl;
}

void test_feed_handler() {
    std::cout << "=== Test: Feed Handler ===" << std::endl;
    FeedConfig cfg{};
    cfg.symbol        = "NVDA";
    cfg.initial_price = 500.0;
    cfg.daily_vol     = 0.30;
    cfg.lambda_new    = 1000.0;  // 1000 new orders/sec
    cfg.lambda_cancel = 800.0;
    cfg.lambda_trade  = 200.0;
    cfg.tick_size     = 0.01;
    cfg.spread_bps    = 10.0;
    cfg.lot_size      = 1.0;
    cfg.max_qty       = 1000.0;
    cfg.depth_levels  = 10;
    cfg.seed          = 42;

    FeedHandler feed(cfg);
    feed.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    feed.stop();

    uint64_t gen  = feed.events_generated();
    uint64_t drop = feed.events_dropped();
    std::cout << "  Events generated: " << gen
              << "  Dropped: " << drop << std::endl;
    assert(gen > 0);

    std::cout << "  [PASS] feed handler" << std::endl;
}

void test_matching_engine() {
    std::cout << "=== Test: Matching Engine ===" << std::endl;
    MatchingEngine engine("AMZN");
    engine.start();

    // Allocate some orders
    std::vector<std::unique_ptr<Order>> orders;
    auto make = [&](OrderId id, Side side, double price, Quantity qty) -> Order* {
        auto o = std::make_unique<Order>(id, "AMZN", side, OrderType::Limit,
                                         TimeInForce::GTC, double_to_price(price), qty, 0);
        Order* ptr = o.get();
        orders.push_back(std::move(o));
        return ptr;
    };

    // Add bids
    for (int i = 0; i < 10; ++i)
        engine.submit_order(make(i+1, Side::Buy, 3000.0 - i*0.10, 100 + i*10));

    // Add asks
    for (int i = 0; i < 10; ++i)
        engine.submit_order(make(100+i, Side::Sell, 3001.0 + i*0.10, 100 + i*10));

    // Aggressive buy
    engine.submit_order(make(200, Side::Buy, 3005.0, 500));

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto stats = engine.stats();
    std::cout << "  Orders received: " << stats.orders_received
              << "  Trades: "          << stats.trades_executed
              << "  Volume: "          << stats.total_volume
              << "  Lat p99: "         << stats.latency_ns_p99 << "ns" << std::endl;

    engine.stop();
    std::cout << "  [PASS] matching engine" << std::endl;
}

int main() {
    std::cout << "HFT Orderbook Tests\n" << std::string(50, '=') << std::endl;

    test_lockfree_queue();
    test_basic_order_book();
    test_matching();
    test_ioc_fok();
    test_market_impact();
    test_feed_handler();
    test_matching_engine();

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
