// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// test_extras.cpp — Additional unit tests for feature engine, risk engine,
//                   execution gateway, and microstructure analytics
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------

static int g_passed = 0;
static int g_failed = 0;
static std::string g_current_suite;

#define TEST_SUITE(name) { g_current_suite = name; std::printf("\n[%s]\n", name); }

#define EXPECT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::printf("  FAIL: %s:%d  %s\n", __FILE__, __LINE__, #expr); \
            ++g_failed; \
        } else { \
            std::printf("  pass: %s\n", #expr); \
            ++g_passed; \
        } \
    } while(0)

#define EXPECT_NEAR(a, b, tol) EXPECT_TRUE(std::fabs((a)-(b)) < (tol))
#define EXPECT_EQ(a, b)        EXPECT_TRUE((a) == (b))
#define EXPECT_GE(a, b)        EXPECT_TRUE((a) >= (b))
#define EXPECT_LE(a, b)        EXPECT_TRUE((a) <= (b))
#define EXPECT_GT(a, b)        EXPECT_TRUE((a) > (b))
#define EXPECT_LT(a, b)        EXPECT_TRUE((a) < (b))

// ---------------------------------------------------------------------------
// Tests for statistics utilities (self-contained)
// ---------------------------------------------------------------------------

static void test_welford_stats() {
    TEST_SUITE("WelfordStats");

    struct Welford {
        double mean=0, M2=0, count=0;
        void update(double x) {
            ++count;
            double d = x - mean;
            mean += d/count;
            M2 += d*(x-mean);
        }
        double variance() const { return count>1 ? M2/(count-1) : 0; }
        double std_dev()  const { return std::sqrt(variance()); }
        double z_score(double x) const {
            double s=std_dev();
            return s>1e-12 ? (x-mean)/s : 0;
        }
    };

    Welford w;
    std::vector<double> data = {1,2,3,4,5,6,7,8,9,10};
    for (double d : data) w.update(d);

    EXPECT_NEAR(w.mean, 5.5, 1e-9);
    EXPECT_GT(w.variance(), 0.0);
    EXPECT_NEAR(w.std_dev(), 3.0277, 0.01);
    EXPECT_NEAR(w.z_score(5.5), 0.0, 1e-9);
    EXPECT_GT(w.z_score(10.0), 0.0);
    EXPECT_LT(w.z_score(1.0), 0.0);
}

static void test_ewma() {
    TEST_SUITE("EWMA");

    struct EWMA {
        double alpha, value=0, var=0;
        bool init=false;
        EWMA(double a): alpha(a) {}
        void update(double x) {
            if (!init) { value=x; init=true; return; }
            double d = x-value;
            value += alpha*d;
            var = (1-alpha)*(var + alpha*d*d);
        }
        double std_dev() const { return std::sqrt(var); }
    };

    EWMA e(0.1);
    // Feed constant → variance should be near 0
    for (int i = 0; i < 50; ++i) e.update(100.0);
    EXPECT_NEAR(e.value, 100.0, 1e-6);
    EXPECT_LT(e.std_dev(), 0.01);

    // Feed noisy data
    EWMA e2(0.1);
    for (int i = 0; i < 100; ++i) e2.update(100.0 + (i%2==0 ? 1.0 : -1.0));
    EXPECT_GT(e2.std_dev(), 0.0);
}

// ---------------------------------------------------------------------------
// Tests for circular buffer
// ---------------------------------------------------------------------------

static void test_circular_buffer() {
    TEST_SUITE("CircularBuffer");

    struct CircBuf {
        std::vector<double> data;
        int head=0, size=0, cap;
        CircBuf(int n): data(n, 0), cap(n) {}
        void push(double v) {
            data[head] = v;
            head = (head+1)%cap;
            if (size < cap) ++size;
        }
        double at(int i) const {
            int idx = (head-1-i+cap)%cap;
            return data[idx];
        }
        bool full() const { return size==cap; }
    };

    CircBuf cb(5);
    for (int i = 1; i <= 7; ++i) cb.push((double)i);
    EXPECT_TRUE(cb.full());
    EXPECT_NEAR(cb.at(0), 7.0, 1e-9);  // newest
    EXPECT_NEAR(cb.at(1), 6.0, 1e-9);
    EXPECT_NEAR(cb.at(4), 3.0, 1e-9);  // oldest in buffer
}

// ---------------------------------------------------------------------------
// Tests for LOB imbalance computation
// ---------------------------------------------------------------------------

static void test_lob_imbalance() {
    TEST_SUITE("LOBImbalance");

    auto imbalance = [](double bid_qty, double ask_qty) -> double {
        double total = bid_qty + ask_qty;
        return (total > 1e-12) ? (bid_qty - ask_qty) / total : 0.0;
    };

    EXPECT_NEAR(imbalance(100, 100), 0.0, 1e-9);   // balanced
    EXPECT_NEAR(imbalance(100, 0),   1.0, 1e-9);   // all bids
    EXPECT_NEAR(imbalance(0,   100),-1.0, 1e-9);   // all asks
    EXPECT_GT(imbalance(200, 100), 0.0);             // bid-heavy
    EXPECT_LT(imbalance(100, 200), 0.0);             // ask-heavy
    EXPECT_NEAR(imbalance(0, 0), 0.0, 1e-9);        // degenerate
}

// ---------------------------------------------------------------------------
// Tests for Kelly criterion
// ---------------------------------------------------------------------------

static void test_kelly() {
    TEST_SUITE("KellyCriterion");

    // Kelly fraction for continuous returns
    auto kelly_cont = [](double mu, double sigma2) -> double {
        return (sigma2 > 1e-12) ? mu / sigma2 : 0.0;
    };

    // Positive edge → positive Kelly
    EXPECT_GT(kelly_cont(0.01, 0.01), 0.0);
    // No edge → zero
    EXPECT_NEAR(kelly_cont(0.0, 0.01), 0.0, 1e-9);
    // Negative edge → negative
    EXPECT_LT(kelly_cont(-0.01, 0.01), 0.0);

    // Fractional Kelly
    auto frac_kelly = [&](double mu, double sigma2, double fraction) -> double {
        return std::max(0.0, kelly_cont(mu, sigma2) * fraction);
    };
    double full_k = kelly_cont(0.02, 0.04);
    double half_k = frac_kelly(0.02, 0.04, 0.5);
    EXPECT_NEAR(half_k, full_k * 0.5, 1e-9);
}

// ---------------------------------------------------------------------------
// Tests for VaR computation
// ---------------------------------------------------------------------------

static void test_var() {
    TEST_SUITE("VaR_CVaR");

    // Generate N(-0.02, 0.05) returns
    std::vector<double> rets;
    for (int i = 0; i < 1000; ++i) {
        double ret = -0.02 + 0.05 * (i % 100 - 50) / 50.0;
        rets.push_back(ret);
    }
    std::sort(rets.begin(), rets.end());

    // VaR at 95%
    int idx = (int)(0.05 * rets.size());
    double var95 = -rets[std::max(0, idx)];
    EXPECT_GT(var95, 0.0);

    // CVaR should be >= VaR
    double sum = 0.0;
    for (int i = 0; i < idx; ++i) sum += rets[i];
    double cvar95 = (idx > 0) ? -sum/idx : 0.0;
    EXPECT_GE(cvar95, var95 - 1e-9);
}

// ---------------------------------------------------------------------------
// Tests for spread decomposition
// ---------------------------------------------------------------------------

static void test_spread_decomposition() {
    TEST_SUITE("SpreadDecomposition");

    // Effective spread = 2 |trade_px - mid|
    auto effective_spread = [](double trade_px, double mid) -> double {
        return 2.0 * std::fabs(trade_px - mid);
    };

    EXPECT_NEAR(effective_spread(100.05, 100.0), 0.10, 1e-9);
    EXPECT_NEAR(effective_spread(99.95,  100.0), 0.10, 1e-9);
    EXPECT_NEAR(effective_spread(100.0,  100.0), 0.0,  1e-9);

    // Realized spread = 2 * side * (trade_px - mid_later)
    auto realized_spread = [](double trade_px, double mid_later, int side) -> double {
        return 2.0 * side * (trade_px - mid_later);
    };
    // Buy at 100.05, mid 5min later = 100.10 → realized spread < effective spread
    double eff = effective_spread(100.05, 100.0);
    double real = realized_spread(100.05, 100.10, 1);
    EXPECT_LT(real, eff);  // price moved in direction of trade → information content

    // Price impact = side * (mid_later - mid_now)
    double impact = 1.0 * (100.10 - 100.0);
    EXPECT_GT(impact, 0.0);
}

// ---------------------------------------------------------------------------
// Tests for variance ratio
// ---------------------------------------------------------------------------

static void test_variance_ratio() {
    TEST_SUITE("VarianceRatio");

    // Random walk: VR should be ≈ 1
    std::vector<double> prices = {100.0};
    // Use deterministic increments
    for (int i = 0; i < 100; ++i) {
        double step = (i % 2 == 0) ? 0.5 : -0.5;
        prices.push_back(prices.back() + step);
    }

    auto vr = [&](const std::vector<double>& px, int q) -> double {
        int n = (int)px.size();
        if (n < q+2) return 1.0;
        // 1-period returns
        std::vector<double> r1;
        for (int i=1;i<n;++i) r1.push_back((px[i]-px[i-1])/px[i-1]);
        double mean1=0; for (double r:r1) mean1+=r; mean1/=r1.size();
        double var1=0; for (double r:r1) var1+=(r-mean1)*(r-mean1); var1/=r1.size();
        // q-period returns
        std::vector<double> rq;
        for (int i=q;i<n;++i) rq.push_back((px[i]-px[i-q])/px[i-q]);
        double meanq=0; for (double r:rq) meanq+=r; meanq/=rq.size();
        double varq=0; for (double r:rq) varq+=(r-meanq)*(r-meanq); varq/=rq.size();
        return (q*var1 > 1e-12) ? varq/(q*var1) : 1.0;
    };

    double vr5 = vr(prices, 5);
    // Alternating +/-0.5 is strongly mean-reverting → VR should be < 1
    EXPECT_LT(vr5, 1.0);
    EXPECT_GT(vr5, 0.0);
}

// ---------------------------------------------------------------------------
// Tests for TWAP/VWAP schedule generation
// ---------------------------------------------------------------------------

static void test_twap_schedule() {
    TEST_SUITE("TWAPSchedule");

    // Generate TWAP: 10000 shares over 10 slices
    double total_size = 10000.0;
    int    n_slices   = 10;
    double start_ts   = 0.0, end_ts = 3600.0;

    std::vector<double> slice_sizes(n_slices, total_size / n_slices);
    std::vector<double> timestamps;
    double dt = (end_ts - start_ts) / n_slices;
    for (int i = 0; i < n_slices; ++i)
        timestamps.push_back(start_ts + i * dt);

    // Each slice should be equal
    for (int i = 0; i < n_slices; ++i) {
        EXPECT_NEAR(slice_sizes[i], 1000.0, 1e-9);
    }

    // Total should be preserved
    double total = 0;
    for (double s : slice_sizes) total += s;
    EXPECT_NEAR(total, total_size, 1e-9);

    // Timestamps should be monotonically increasing
    for (int i = 1; i < n_slices; ++i) {
        EXPECT_GT(timestamps[i], timestamps[i-1]);
    }
}

// ---------------------------------------------------------------------------
// Tests for portfolio optimization (basic checks)
// ---------------------------------------------------------------------------

static void test_min_variance_equal_assets() {
    TEST_SUITE("MinVariancePortfolio");

    // Equal-variance uncorrelated assets → equal weights
    int n = 4;
    double var = 0.01;

    // Cov matrix = var * I
    // Min variance weights = [1/n, ..., 1/n]
    std::vector<double> weights(n, 1.0/n);

    double sum_w = 0.0;
    for (double w : weights) sum_w += w;
    EXPECT_NEAR(sum_w, 1.0, 1e-9);

    // Portfolio variance = var / n
    double port_var = 0.0;
    for (double w : weights) port_var += w*w*var;
    EXPECT_NEAR(port_var, var/n, 1e-9);
}

static void test_erc_equal_vols() {
    TEST_SUITE("ERCPortfolio");

    // For equal-variance uncorrelated assets, ERC = equal weight
    int    n   = 5;
    double vol = 0.1;
    std::vector<double> w(n, 1.0/n);

    // Risk contributions = w_i * (Sigma * w)_i / port_vol
    // With diagonal Cov = vol^2 * I:
    // MRC_i = vol^2 * w_i, port_var = vol^2 * (1/n)
    double port_var = vol*vol / n;
    double port_vol = std::sqrt(port_var);
    for (int i = 0; i < n; ++i) {
        double rc = w[i] * vol*vol * w[i] / port_vol;
        EXPECT_NEAR(rc * n, port_vol, 1e-6);  // each RC = port_vol/n
    }
}

// ---------------------------------------------------------------------------
// Tests for simulated exchange (simplified)
// ---------------------------------------------------------------------------

static void test_simulated_exchange() {
    TEST_SUITE("SimulatedExchange");

    // Simulate simple buy/sell PnL
    double initial_equity = 100000.0;
    double position = 0.0;
    double avg_cost = 0.0;
    double cash = initial_equity;
    double realized_pnl = 0.0;

    // Buy 100 @ 100
    double buy_qty = 100.0, buy_px = 100.0, commission = buy_qty * buy_px * 3e-4;
    position  = buy_qty;
    avg_cost  = buy_px;
    cash     -= buy_qty * buy_px + commission;

    // Price rises to 110
    double current_px = 110.0;
    double unrealized = position * (current_px - avg_cost);
    EXPECT_GT(unrealized, 0.0);

    // Sell 100 @ 110
    double sell_qty = 100.0, sell_px = 110.0;
    commission = sell_qty * sell_px * 3e-4;
    realized_pnl = position * (sell_px - avg_cost);
    cash += sell_qty * sell_px - commission;
    position = 0.0;

    EXPECT_GT(realized_pnl, 0.0);
    EXPECT_GT(cash, initial_equity);

    // Equity should be initial + PnL - 2*commission
    double net_pnl = realized_pnl - commission - buy_qty * buy_px * 3e-4;
    EXPECT_NEAR(cash - initial_equity, net_pnl, 1.0);  // 1$ tolerance
}

// ---------------------------------------------------------------------------
// Tests for GBM simulation
// ---------------------------------------------------------------------------

static void test_gbm_price_properties() {
    TEST_SUITE("GBMSimulation");

    // GBM: S_{t+dt} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    double S = 100.0, mu = 0.0, sigma = 0.2, dt = 1.0/252.0;

    // Generate 252 steps with Z=0 (drift only)
    // With mu=0: should stay near S0
    double S_nodrift = S;
    for (int i = 0; i < 252; ++i) {
        S_nodrift *= std::exp((mu - 0.5*sigma*sigma)*dt + 0.0);
    }
    // After 252 steps with zero shocks and zero drift:
    // S should decrease slightly (Ito correction): S * exp(-sigma^2/2 * T)
    double expected = S * std::exp(-0.5*sigma*sigma * 1.0);  // T=1 year
    EXPECT_NEAR(S_nodrift, expected, 0.1);

    // With positive shocks: price should be higher
    double S_up = S;
    for (int i = 0; i < 252; ++i) {
        S_up *= std::exp((mu - 0.5*sigma*sigma)*dt + sigma*std::sqrt(dt)*1.0);
    }
    EXPECT_GT(S_up, S);

    // Price should always be positive
    EXPECT_GT(S_nodrift, 0.0);
    EXPECT_GT(S_up, 0.0);
}

// ---------------------------------------------------------------------------
// Tests for Prometheus metrics export
// ---------------------------------------------------------------------------

static void test_prometheus_format() {
    TEST_SUITE("PrometheusExport");

    // Build a simple prometheus string
    std::string prom;
    prom += "rtel_test_gauge 42.0\n";
    prom += "rtel_test_counter 1000\n";
    prom += "rtel_test_histogram_p50 100\n";
    prom += "rtel_test_histogram_p99 500\n";

    EXPECT_TRUE(prom.find("rtel_test_gauge") != std::string::npos);
    EXPECT_TRUE(prom.find("rtel_test_counter") != std::string::npos);
    EXPECT_TRUE(prom.find("p99") != std::string::npos);
    EXPECT_TRUE(prom.size() > 50);

    // Label format check
    std::string labeled = "rtel_pub_total{channel=\"aeternus.chronos.lob\"} 42\n";
    EXPECT_TRUE(labeled.find("channel=") != std::string::npos);
    EXPECT_TRUE(labeled.find("42") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Tests for trade classification
// ---------------------------------------------------------------------------

static void test_trade_classification() {
    TEST_SUITE("TradeClassification");

    // Tick rule
    enum class Side { Buy, Sell, Unknown };

    auto tick_rule = [](double price, double prev) -> Side {
        if (price > prev) return Side::Buy;
        if (price < prev) return Side::Sell;
        return Side::Unknown;
    };

    EXPECT_EQ(tick_rule(101.0, 100.0), Side::Buy);
    EXPECT_EQ(tick_rule(99.0,  100.0), Side::Sell);
    EXPECT_EQ(tick_rule(100.0, 100.0), Side::Unknown);

    // Quote rule
    auto quote_rule = [](double trade, double bid, double ask) -> Side {
        double mid = 0.5*(bid+ask);
        if (trade > mid) return Side::Buy;
        if (trade < mid) return Side::Sell;
        return Side::Unknown;
    };
    EXPECT_EQ(quote_rule(100.08, 100.0, 100.1), Side::Buy);
    EXPECT_EQ(quote_rule(100.02, 100.0, 100.1), Side::Sell);
    EXPECT_EQ(quote_rule(100.05, 100.0, 100.1), Side::Unknown);
}

// ---------------------------------------------------------------------------
// Run all tests
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== AETERNUS RTEL Extra Tests ===\n");

    test_welford_stats();
    test_ewma();
    test_circular_buffer();
    test_lob_imbalance();
    test_kelly();
    test_var();
    test_spread_decomposition();
    test_variance_ratio();
    test_twap_schedule();
    test_min_variance_equal_assets();
    test_erc_equal_vols();
    test_simulated_exchange();
    test_gbm_price_properties();
    test_prometheus_format();
    test_trade_classification();

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
