/// test_matching.cpp — Unit tests for the Chronos matching engine.
/// Minimal test harness (no external framework required).

#include "../include/lob_types.hpp"
#include "../include/price_level.hpp"
#include "../include/matching_engine.hpp"
#include "../include/simd_utils.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace chronos;

// ── Test helpers ──────────────────────────────────────────────────────────────

static int n_pass = 0, n_fail = 0;

#define TEST(name) \
    static void name(); \
    struct __Test_##name { \
        __Test_##name() { \
            try { name(); \
                std::cout << "[PASS] " #name "\n"; ++n_pass; \
            } catch (...) { \
                std::cout << "[FAIL] " #name "\n"; ++n_fail; \
            } \
        } \
    } __test_##name##_inst; \
    static void name()

#define ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "ASSERT FAILED: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
        throw std::runtime_error("assertion failed"); \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a)-(b)) < (tol))

// ── Helpers ───────────────────────────────────────────────────────────────────

static Order make_limit(OrderId id, Side side, double price, double qty, Nanos ts = 0, AgentId agent = 0) {
    Order o;
    o.id = id;
    o.instrument_id = 1;
    o.agent_id = agent;
    o.side = side;
    o.type = OrderType::Limit;
    o.tif = TimeInForce::GTC;
    o.price = to_tick(price);
    o.orig_qty = qty;
    o.leaves_qty = qty;
    o.timestamp_ns = ts;
    o.status = OrderStatus::New;
    return o;
}

static Order make_market(OrderId id, Side side, double qty, Nanos ts = 0, AgentId agent = 0) {
    Order o;
    o.id = id;
    o.instrument_id = 1;
    o.agent_id = agent;
    o.side = side;
    o.type = OrderType::Market;
    o.tif = TimeInForce::IOC;
    o.price = (side == Side::Bid) ? to_tick(1e12) : to_tick(0.0);
    o.orig_qty = qty;
    o.leaves_qty = qty;
    o.timestamp_ns = ts;
    o.status = OrderStatus::New;
    return o;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST(test_basic_types) {
    ASSERT(to_tick(100.0) == 100 * PRICE_SCALE);
    ASSERT_NEAR(from_tick(to_tick(100.0)), 100.0, 1e-6);
    ASSERT(to_tick(100.005) == to_tick(100.005));  // Round-trip.
    ASSERT(opposite(Side::Bid) == Side::Ask);
    ASSERT(opposite(Side::Ask) == Side::Bid);
}

TEST(test_order_apply_fill) {
    Order o = make_limit(1, Side::Bid, 100.0, 100.0);
    Qty actual = o.apply_fill(30.0);
    ASSERT_NEAR(actual, 30.0, 1e-9);
    ASSERT_NEAR(o.leaves_qty, 70.0, 1e-9);
    ASSERT_NEAR(o.filled_qty, 30.0, 1e-9);
    ASSERT(o.status == OrderStatus::PartialFill);

    o.apply_fill(70.0);
    ASSERT_NEAR(o.leaves_qty, 0.0, 1e-9);
    ASSERT(o.status == OrderStatus::Filled);
}

TEST(test_price_level_push_pop) {
    PriceLevel lvl(to_tick(100.0), Side::Bid);
    ASSERT(lvl.empty());

    Order o1 = make_limit(1, Side::Bid, 100.0, 50.0);
    Order o2 = make_limit(2, Side::Bid, 100.0, 30.0);
    lvl.push_back(o1);
    lvl.push_back(o2);

    ASSERT(!lvl.empty());
    ASSERT(lvl.order_count == 2);
    ASSERT_NEAR(lvl.total_qty, 80.0, 1e-9);

    Order* front = lvl.front();
    ASSERT(front != nullptr);
    ASSERT(front->id == 1);  // FIFO.

    lvl.pop_front();
    ASSERT(lvl.order_count == 1);
    front = lvl.front();
    ASSERT(front->id == 2);
}

TEST(test_price_level_fill_front) {
    PriceLevel lvl(to_tick(100.0), Side::Bid);
    Order o = make_limit(1, Side::Bid, 100.0, 100.0);
    lvl.push_back(o);

    Qty filled = lvl.fill_front(60.0);
    ASSERT_NEAR(filled, 60.0, 1e-9);
    ASSERT(!lvl.empty());  // Still has 40 remaining.

    filled = lvl.fill_front(40.0);
    ASSERT_NEAR(filled, 40.0, 1e-9);
    ASSERT(lvl.empty());  // Fully consumed.
}

TEST(test_price_level_remove_by_id) {
    PriceLevel lvl(to_tick(100.0), Side::Bid);
    for (int i = 1; i <= 5; ++i) {
        lvl.push_back(make_limit(i, Side::Bid, 100.0, 10.0));
    }
    ASSERT(lvl.order_count == 5);

    Order removed;
    bool found = lvl.remove_by_id(3, removed);
    ASSERT(found);
    ASSERT(removed.id == 3);
    ASSERT(lvl.order_count == 4);
    ASSERT_NEAR(lvl.total_qty, 40.0, 1e-9);
}

TEST(test_price_level_queue_position) {
    PriceLevel lvl(to_tick(100.0), Side::Bid);
    for (int i = 1; i <= 4; ++i) {
        lvl.push_back(make_limit(i, Side::Bid, 100.0, 10.0 * i));
    }
    ASSERT(lvl.queue_position(1) == 0);
    ASSERT(lvl.queue_position(2) == 1);
    ASSERT(lvl.queue_position(4) == 3);
    ASSERT(lvl.queue_position(99) == -1);  // Not found.

    // Qty ahead of order 3: 10 + 20 = 30.
    ASSERT_NEAR(lvl.qty_ahead(3), 30.0, 1e-9);
}

TEST(test_engine_basic_match) {
    auto engine = create_matching_engine(1);
    std::vector<Fill> fills;
    engine->set_fill_callback([&](const Fill& f) { fills.push_back(f); });

    // Bid at 100, ask at 100 → immediate fill.
    auto bid = make_limit(1, Side::Bid, 100.0, 10.0);
    auto ask = make_limit(2, Side::Ask, 100.0, 10.0);

    engine->submit(bid);
    engine->submit(ask);

    ASSERT(fills.size() == 1);
    ASSERT_NEAR(fills[0].qty, 10.0, 1e-9);
    ASSERT_NEAR(from_tick(fills[0].price), 100.0, 1e-4);
    ASSERT(engine->order_count() == 0);
}

TEST(test_engine_partial_fill) {
    auto engine = create_matching_engine(1);
    std::vector<Fill> fills;
    engine->set_fill_callback([&](const Fill& f) { fills.push_back(f); });

    engine->submit(make_limit(1, Side::Bid, 100.0, 5.0));
    engine->submit(make_limit(2, Side::Ask, 100.0, 10.0));

    ASSERT(fills.size() == 1);
    ASSERT_NEAR(fills[0].qty, 5.0, 1e-9);
    // Ask should have 5 remaining.
    ASSERT(engine->order_count() == 1);
}

TEST(test_engine_no_cross) {
    auto engine = create_matching_engine(1);
    std::vector<Fill> fills;
    engine->set_fill_callback([&](const Fill& f) { fills.push_back(f); });

    engine->submit(make_limit(1, Side::Bid, 99.0, 10.0));
    engine->submit(make_limit(2, Side::Ask, 101.0, 10.0));

    ASSERT(fills.empty());
    ASSERT(engine->order_count() == 2);
    ASSERT(to_tick(99.0) == engine->best_bid());
    ASSERT(to_tick(101.0) == engine->best_ask());
}

TEST(test_engine_market_order) {
    auto engine = create_matching_engine(1);
    std::vector<Fill> fills;
    engine->set_fill_callback([&](const Fill& f) { fills.push_back(f); });

    // Multiple ask levels.
    for (int i = 0; i < 5; ++i) {
        engine->submit(make_limit(100 + i, Side::Ask, 100.0 + i * 0.01, 20.0));
    }

    // Market buy for 60 units.
    engine->submit(make_market(200, Side::Bid, 60.0));

    double total_filled = 0.0;
    for (const auto& f : fills) total_filled += f.qty;
    ASSERT_NEAR(total_filled, 60.0, 1e-9);
}

TEST(test_engine_cancel) {
    auto engine = create_matching_engine(1);
    engine->submit(make_limit(1, Side::Bid, 100.0, 50.0));
    ASSERT(engine->order_count() == 1);
    ASSERT(engine->cancel(1));
    ASSERT(engine->order_count() == 0);
    ASSERT(!engine->cancel(1));  // Already cancelled.
}

TEST(test_engine_ioc_cancel) {
    auto engine = create_matching_engine(1);
    engine->submit(make_limit(1, Side::Ask, 100.0, 5.0));

    // IOC buy for 10, only 5 available.
    Order ioc = make_limit(2, Side::Bid, 100.0, 10.0);
    ioc.tif = TimeInForce::IOC;

    std::vector<Fill> fills;
    engine->set_fill_callback([&](const Fill& f) { fills.push_back(f); });
    engine->submit(ioc);

    double total = 0.0;
    for (const auto& f : fills) total += f.qty;
    ASSERT_NEAR(total, 5.0, 1e-9);
    ASSERT(engine->order_count() == 0);  // IOC remainder cancelled.
}

TEST(test_engine_fok_reject) {
    auto engine = create_matching_engine(1);
    engine->submit(make_limit(1, Side::Ask, 100.0, 5.0));

    // FOK buy for 10 — only 5 available → should be rejected.
    Order fok = make_limit(2, Side::Bid, 100.0, 10.0);
    fok.tif = TimeInForce::FOK;

    std::vector<Fill> fills;
    engine->set_fill_callback([&](const Fill& f) { fills.push_back(f); });
    engine->submit(fok);

    ASSERT(fills.empty());
    ASSERT(engine->order_count() == 1);  // Ask still resting.
}

TEST(test_engine_vwap_sweep) {
    auto engine = create_matching_engine(1);
    for (int i = 0; i < 5; ++i) {
        engine->submit(make_limit(i + 1, Side::Ask, 100.0 + i, 10.0));
    }
    // Sweep 30 units: fills at 100, 101, 102.
    double vwap = engine->vwap_sweep(Side::Bid, 30.0);
    ASSERT_NEAR(vwap, 101.0, 1e-4);
}

TEST(test_engine_snapshot) {
    auto engine = create_matching_engine(1);
    engine->submit(make_limit(1, Side::Bid, 99.0, 100.0));
    engine->submit(make_limit(2, Side::Ask, 101.0, 100.0));
    MarketSnapshot snap = engine->snapshot(5);
    ASSERT_NEAR(snap.mid_price(), 100.0, 1e-4);
    ASSERT_NEAR(snap.spread(), 2.0, 1e-4);
    ASSERT(snap.bid_levels == 1);
    ASSERT(snap.ask_levels == 1);
}

TEST(test_engine_modify) {
    auto engine = create_matching_engine(1);
    engine->submit(make_limit(1, Side::Bid, 99.0, 50.0));
    // Modify to 100.
    engine->modify(1, to_tick(100.0), 60.0);
    // Now a sell at 100 should match.
    std::vector<Fill> fills;
    engine->set_fill_callback([&](const Fill& f) { fills.push_back(f); });
    engine->submit(make_limit(2, Side::Ask, 100.0, 30.0));
    ASSERT(!fills.empty());
}

TEST(test_simd_find_bid) {
    int64_t prices[] = { 105, 103, 101, 99, 97 }; // descending bid prices
    size_t idx = simd::find_first_bid_match_scalar(prices, 5, 102);
    ASSERT(idx == 1);  // prices[1]=103 > 102, prices[2]=101 <= 102 → wait...
    // Actually: find first where prices[i] <= 102: that's prices[2]=101 at index 2.
    ASSERT(simd::find_first_bid_match_scalar(prices, 5, 102) == 2);
}

TEST(test_simd_find_ask) {
    int64_t prices[] = { 97, 99, 101, 103, 105 }; // ascending ask prices
    // find first >= 100: prices[2]=101 at index 2.
    size_t idx = simd::find_first_ask_match_scalar(prices, 5, 100);
    ASSERT(idx == 2);
}

TEST(test_price_level_for_each) {
    PriceLevel lvl(to_tick(100.0), Side::Ask);
    for (int i = 1; i <= 5; ++i) {
        lvl.push_back(make_limit(i, Side::Ask, 100.0, 10.0));
    }
    int count = 0;
    lvl.for_each([&](const Order& o) { ++count; });
    ASSERT(count == 5);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n=== Chronos Matching Engine Tests ===\n\n";
    // All TEST macros above run their constructors at static init time.
    std::cout << "\nResults: " << n_pass << " passed, " << n_fail << " failed.\n";
    return n_fail > 0 ? 1 : 0;
}
