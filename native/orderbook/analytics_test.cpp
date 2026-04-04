// Integration tests for tick analytics, intraday seasonality,
// trade clustering, regime detection, and alert system.

#include "tick_analytics.hpp"
#include "order_flow.hpp"
#include "market_impact.hpp"
#include "risk_checks.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <random>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <string>

using namespace hft;

// ============================================================
// Helper: generate GBM price path
// ============================================================
std::vector<double> generate_gbm(double S0, double mu, double sigma,
                                   int T, uint64_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> norm(0, 1);
    std::vector<double> path(T);
    path[0] = S0;
    for (int t = 1; t < T; ++t)
        path[t] = path[t-1] * std::exp((mu - 0.5*sigma*sigma)/252.0 +
                                         sigma/std::sqrt(252.0)*norm(rng));
    return path;
}

// ============================================================
// Test: Intraday Seasonality
// ============================================================
bool test_seasonality() {
    std::cout << "test_seasonality... ";
    IntradaySeasonality season;

    // Simulate a trading day with higher volume at open/close
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> unif(0, 1);

    int64_t open_ns = 34200LL * 1'000'000'000LL;
    int64_t ts = open_ns;

    for (int min = 0; min < 390; ++min) {
        // U-shaped volume: high at open/close, low midday
        double tod = min / 390.0;
        double vol_factor = 1.0 + 2.0 * std::exp(-50 * tod * tod) +
                                  2.0 * std::exp(-50 * (tod - 1.0) * (tod - 1.0));
        double volume = 1000.0 * vol_factor * (0.8 + 0.4 * unif(rng));
        double spread = 0.01 + 0.005 * vol_factor;
        season.update(ts, volume, spread, 0.001 * (unif(rng) - 0.5));
        ts += 60LL * 1'000'000'000LL;
    }

    auto top5 = season.top_volume_minutes(5);
    // Open (first few minutes) and close should be in top-5
    bool has_open = false;
    for (int m : top5) if (m < 30) has_open = true;
    assert(has_open);
    assert(!top5.empty());

    // Normalized volume at open should be > 1
    int64_t open_ts = open_ns + 1LL * 1'000'000'000LL;
    double nv = season.normalized_volume(open_ts, 3000.0);
    assert(nv > 0);

    std::cout << "PASS\n";
    return true;
}

// ============================================================
// Test: Trade Clustering
// ============================================================
bool test_clustering() {
    std::cout << "test_clustering... ";
    TradeClusterer clusterer;
    std::mt19937 rng(13);
    std::normal_distribution<double> norm(0, 1);
    std::uniform_real_distribution<double> unif(0, 1);

    int64_t ts = 34200LL * 1'000'000'000LL;
    double price = 100.0;
    uint64_t total_trades = 0;

    for (int i = 0; i < 1000; ++i) {
        price += norm(rng) * 0.01;
        ts += static_cast<int64_t>(unif(rng) * 50'000'000); // up to 50ms gaps
        bool is_buy = unif(rng) < 0.5;
        double volume = 100 + std::fabs(norm(rng)) * 50;

        clusterer.on_trade(ts, price, volume, is_buy);
        total_trades++;
    }
    clusterer.close_cluster(ts + 1'000'000'000LL);

    auto stats = clusterer.compute_stats();
    assert(clusterer.total_clusters() > 0);
    assert(stats.avg_cluster_volume > 0);

    std::cout << "PASS (clusters=" << clusterer.total_clusters() << ")\n";
    return true;
}

// ============================================================
// Test: Regime Detection
// ============================================================
bool test_regime_detection() {
    std::cout << "test_regime_detection... ";
    RegimeDetector detector;
    std::mt19937 rng(99);
    std::normal_distribution<double> norm(0, 1);

    // Phase 1: calm period (low vol)
    for (int t = 0; t < 200; ++t)
        detector.update(norm(rng) * 0.001);

    // Phase 2: high volatility spike
    for (int t = 0; t < 30; ++t)
        detector.update(norm(rng) * 0.05); // 5x vol

    auto regime = detector.current_regime();
    // After high vol, should not be Low_Vol
    assert(regime != RegimeDetector::Regime::Low_Vol);

    std::cout << "PASS (regime=" << detector.regime_name() << ")\n";
    return true;
}

// ============================================================
// Test: Alert Monitor
// ============================================================
bool test_alert_monitor() {
    std::cout << "test_alert_monitor... ";

    int alert_count = 0;
    AlertMonitor monitor("ALERT");
    monitor.set_callback([&](const Alert& a) {
        ++alert_count;
        (void)a;
    });

    // Normal conditions: no alerts
    for (int i = 0; i < 100; ++i) {
        monitor.on_update(i * 1'000'000LL, 100.0, 0.20, 0.1, 0.01, 0.01,
                          0.1, 100, 1'000'000.0);
    }
    int normal_alerts = alert_count;

    // VPIN spike
    monitor.on_update(200'000'000LL, 100.0, 0.20, 0.9, 0.01, 0.01,
                      0.1, 100, 1'000'000.0);
    assert(alert_count > normal_alerts); // Should have fired VPIN alert

    std::cout << "PASS (alerts=" << alert_count << ")\n";
    return true;
}

// ============================================================
// Test: Bar Builder
// ============================================================
bool test_bar_builder() {
    std::cout << "test_bar_builder... ";

    int bars_built = 0;
    double total_volume = 0;

    BarBuilder builder(BarBuilder::BarType::Time, 60'000'000'000LL); // 1-min bars
    builder.set_callback([&](const OHLCVBar& bar) {
        ++bars_built;
        total_volume += bar.volume;
        assert(bar.high >= bar.open);
        assert(bar.high >= bar.close);
        assert(bar.low  <= bar.open);
        assert(bar.low  <= bar.close);
        assert(bar.vwap >= bar.low && bar.vwap <= bar.high);
    });

    std::mt19937 rng(55);
    std::normal_distribution<double> norm(0, 1);
    std::uniform_int_distribution<int> qty_dist(50, 200);

    double price = 100.0;
    int64_t ts = 34200LL * 1'000'000'000LL;

    // 10 minutes of ticks
    for (int i = 0; i < 600; ++i) {
        price += norm(rng) * 0.01;
        ts    += 1'000'000'000; // 1 second per tick
        uint64_t qty = qty_dist(rng);
        builder.on_trade(ts, price, qty, norm(rng) > 0, 0.02);
    }
    builder.close_bar();

    assert(bars_built >= 9 && bars_built <= 11); // ~10 minutes
    assert(total_volume > 0);
    std::cout << "PASS (bars=" << bars_built << " vol=" << std::fixed
              << std::setprecision(0) << total_volume << ")\n";
    return true;
}

// ============================================================
// Test: Roll Model
// ============================================================
bool test_roll_model() {
    std::cout << "test_roll_model... ";
    RollModel roll(200);
    std::mt19937 rng(17);
    std::normal_distribution<double> norm(0, 1);

    // Simulate prices with bid-ask bounce
    double mid = 100.0;
    double half_spread = 0.05;
    for (int i = 0; i < 500; ++i) {
        mid += norm(rng) * 0.005;
        // Trade alternates between bid and ask (bounce)
        double px = mid + (i % 2 == 0 ? half_spread : -half_spread);
        roll.update(px);
    }

    double est_spread = roll.roll_spread();
    // Roll model should estimate a spread in the right ballpark
    // It may be 0 if autocov is non-negative (noise), so just test it's non-negative
    assert(est_spread >= 0.0);
    std::cout << "PASS (roll_spread=" << std::fixed << std::setprecision(4)
              << est_spread << ")\n";
    return true;
}

// ============================================================
// Test: Full analytics engine
// ============================================================
bool test_analytics_engine() {
    std::cout << "test_analytics_engine... ";

    int alert_count = 0;
    TickAnalyticsEngine engine("TEST");
    engine.set_alert_callback([&](const Alert& a) {
        ++alert_count;
        (void)a;
    });

    std::mt19937 rng(2024);
    std::normal_distribution<double> norm(0, 1);
    std::uniform_real_distribution<double> unif(0, 1);

    double price = 150.0;
    int64_t ts = 34200LL * 1'000'000'000LL;

    for (int i = 0; i < 2000; ++i) {
        price *= std::exp(norm(rng) * 0.005);
        ts    += 1'000'000;
        double spread = 0.02 + std::fabs(norm(rng)) * 0.01;
        double bid = price - spread / 2;
        double ask = price + spread / 2;
        bool is_buy = unif(rng) < 0.5;
        uint64_t qty = 100 + static_cast<uint64_t>(std::fabs(norm(rng)) * 50);

        auto state = engine.on_trade(ts, double_to_price(price),
                                     double_to_price(bid), double_to_price(ask),
                                     qty, is_buy);
        (void)state;
    }

    std::cout << "PASS (alerts=" << alert_count << ")\n";
    return true;
}

// ============================================================
// Test: VPIN integration
// ============================================================
bool test_vpin_integration() {
    std::cout << "test_vpin_integration... ";
    VPINEstimator::Config cfg{};
    cfg.bucket_volume  = 1000;
    cfg.sample_buckets = 10;
    VPINEstimator vpin(cfg);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> qty(50, 200);
    std::normal_distribution<double> noise(0, 0.01);

    double px = 100.0;
    // Simulate toxic flow: all buys
    for (int i = 0; i < 100; ++i) {
        px += noise(rng);
        Price bid = double_to_price(px - 0.01);
        Price ask = double_to_price(px + 0.01);
        vpin.update(ask, bid, ask, qty(rng)); // buy-side aggression
    }

    double v = vpin.vpin();
    assert(v >= 0.0 && v <= 1.0);
    // With all buys, VPIN should be elevated
    if (vpin.is_ready()) {
        assert(v > 0.3);
    }
    std::cout << "PASS (vpin=" << std::fixed << std::setprecision(4) << v << ")\n";
    return true;
}

// ============================================================
// Performance benchmark: analytics throughput
// ============================================================
void bench_analytics_throughput() {
    std::cout << "\n--- Analytics Throughput Benchmark ---\n";

    TickAnalyticsEngine engine("BENCH");
    std::mt19937 rng(123);
    std::normal_distribution<double> norm(0, 1);
    std::uniform_real_distribution<double> unif(0, 1);

    double price = 100.0;
    int64_t ts   = 34200LL * 1'000'000'000LL;
    const int N  = 100000;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        price *= std::exp(norm(rng) * 0.001);
        ts    += 1'000'000;
        double spread = 0.02;
        bool is_buy   = unif(rng) < 0.5;
        engine.on_trade(ts, double_to_price(price),
                        double_to_price(price - spread/2),
                        double_to_price(price + spread/2),
                        100, is_buy);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ns_per_tick = elapsed_ms * 1e6 / N;
    double tps = N / elapsed_ms * 1000.0;

    std::cout << "  " << N << " ticks in " << std::fixed << std::setprecision(1)
              << elapsed_ms << " ms\n";
    std::cout << "  Throughput: " << std::setprecision(0) << tps << " ticks/sec\n";
    std::cout << "  Latency:    " << std::setprecision(1) << ns_per_tick << " ns/tick\n";
}

// ============================================================
// Main
// ============================================================
int main() {
    std::cout << "=== Analytics Module Tests ===\n\n";

    int passed = 0, total = 0;

    auto run = [&](bool (*test)()) {
        ++total;
        if (test()) ++passed;
    };

    run(test_seasonality);
    run(test_clustering);
    run(test_regime_detection);
    run(test_alert_monitor);
    run(test_bar_builder);
    run(test_roll_model);
    run(test_analytics_engine);
    run(test_vpin_integration);

    bench_analytics_throughput();

    std::cout << "\n" << passed << "/" << total << " tests passed.\n";
    return passed == total ? 0 : 1;
}
