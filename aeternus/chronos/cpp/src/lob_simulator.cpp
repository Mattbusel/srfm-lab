/// lob_simulator.cpp — Main simulation loop.
///
/// Generates synthetic order flow using Poisson arrivals with Hawkes clustering.
/// Drives mid-price via Heston SV model. Writes events to Arrow IPC format
/// (or CSV if Arrow not available).
///
/// Architecture:
///   1. Heston model generates mid-price path.
///   2. Hawkes process generates bid/ask order arrival times.
///   3. Order generator creates limit/market orders around the mid-price.
///   4. Matching engine processes all orders in timestamp order.
///   5. Events serialized to output.

#include "../include/lob_types.hpp"
#include "../include/matching_engine.hpp"
#include "../include/simd_utils.hpp"

#include <cmath>
#include <vector>
#include <queue>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <functional>
#include <cassert>
#include <memory>
#include <unordered_map>

namespace chronos {
namespace simulator {

// ── Hawkes Process ────────────────────────────────────────────────────────────

struct HawkesParams {
    double mu;     ///< Baseline intensity.
    double alpha;  ///< Jump size.
    double beta;   ///< Decay rate.
};

/// Simulate univariate Hawkes process using Ogata thinning.
/// Returns sorted event times in [0, horizon].
static std::vector<double> simulate_hawkes(
    double horizon,
    HawkesParams params,
    std::mt19937_64& rng
) {
    assert(params.alpha / params.beta < 1.0 && "Hawkes must be stationary");
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<double> times;
    double t = 0.0;
    double lambda_bar = params.mu;

    while (t < horizon) {
        double u = uni(rng);
        if (u < 1e-20) break;
        double dt = -std::log(u) / lambda_bar;
        t += dt;
        if (t >= horizon) break;

        // Compute actual intensity.
        double lambda_t = params.mu;
        for (double ti : times) {
            lambda_t += params.alpha * std::exp(-params.beta * (t - ti));
        }

        double u2 = uni(rng);
        if (u2 <= lambda_t / lambda_bar) {
            times.push_back(t);
            lambda_bar = lambda_t + params.alpha;
        } else {
            lambda_bar = lambda_t;
        }
    }
    return times;
}

// ── Heston Path ───────────────────────────────────────────────────────────────

struct HestonState {
    double s;   ///< Price.
    double v;   ///< Variance.
    double t;   ///< Time.
};

struct HestonSimParams {
    double mu, kappa, theta, sigma, rho, v0;
};

static std::vector<std::pair<double, double>> simulate_heston_path(
    double s0,
    HestonSimParams p,
    size_t n_steps,
    double horizon,
    std::mt19937_64& rng
) {
    std::vector<std::pair<double, double>> path; // (time, price)
    path.reserve(n_steps + 1);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    double dt = horizon / n_steps;
    double sqrt_dt = std::sqrt(dt);
    double rho_bar = std::sqrt(1.0 - p.rho * p.rho);

    double s = s0, v = p.v0;
    path.push_back({0.0, s});

    for (size_t i = 0; i < n_steps; ++i) {
        double u1 = std::max(uni(rng), 1e-20);
        double u2 = uni(rng);
        double u3 = std::max(uni(rng), 1e-20);
        double u4 = uni(rng);

        double mag1 = std::sqrt(-2.0 * std::log(u1));
        double n1 = mag1 * std::cos(2.0 * M_PI * u2);
        double mag2 = std::sqrt(-2.0 * std::log(u3));
        double n2 = mag2 * std::cos(2.0 * M_PI * u4);

        double dws = n1 * sqrt_dt;
        double dwv = (p.rho * n1 + rho_bar * n2) * sqrt_dt;

        double vp = std::max(v, 0.0);
        double sqv = std::sqrt(vp);

        double ds_log = (p.mu - 0.5 * vp) * dt + sqv * dws;
        s = s * std::exp(ds_log);
        v = v + p.kappa * (p.theta - vp) * dt + p.sigma * sqv * dwv;

        path.push_back({(i + 1) * dt, s});
    }
    return path;
}

// ── Simulation Config ─────────────────────────────────────────────────────────

struct SimConfig {
    // Time parameters.
    double horizon_secs = 3600.0;   ///< Simulation horizon in seconds.
    Nanos  start_ts_ns  = 0;

    // Instrument.
    InstId instrument_id = 1;
    double initial_price = 100.0;
    double tick_size     = 0.01;

    // Heston parameters.
    HestonSimParams heston = {
        .mu    = 0.0,
        .kappa = 2.0,
        .theta = 0.04,
        .sigma = 0.3,
        .rho   = -0.7,
        .v0    = 0.04,
    };

    // Hawkes parameters (bid side).
    HawkesParams hawkes_bid = { .mu = 5.0, .alpha = 2.0, .beta = 8.0 };
    // Hawkes parameters (ask side).
    HawkesParams hawkes_ask = { .mu = 5.0, .alpha = 2.0, .beta = 8.0 };
    // Noise (uniform Poisson baseline).
    double noise_intensity = 2.0;

    // Order parameters.
    double market_fraction  = 0.3;   ///< Fraction of orders that are market.
    double avg_qty          = 100.0;
    double qty_std          = 50.0;
    uint32_t max_levels     = 5;     ///< Max depth for limit order placement.
    double spread_ticks     = 2.0;   ///< Initial spread in ticks.

    // Output.
    std::string output_path = "sim_events.csv";
    bool        write_csv   = true;
    bool        verbose     = false;
};

// ── Order Generator ───────────────────────────────────────────────────────────

class OrderGenerator {
public:
    explicit OrderGenerator(const SimConfig& cfg, uint64_t seed)
        : cfg_(cfg)
        , rng_(seed)
        , uni_(0.0, 1.0)
        , id_counter_(1)
    {}

    OrderId next_id() { return id_counter_++; }

    double sample_qty() {
        double q = cfg_.avg_qty + cfg_.qty_std * (uni_(rng_) * 2.0 - 1.0);
        return std::max(1.0, q);
    }

    double sample_price_offset(double mid, Side side) {
        double half_spread = cfg_.spread_ticks * cfg_.tick_size / 2.0;
        double depth_offset = std::floor(uni_(rng_) * cfg_.max_levels) * cfg_.tick_size;
        if (side == Side::Bid) {
            return mid - half_spread - depth_offset;
        } else {
            return mid + half_spread + depth_offset;
        }
    }

    Order make_limit(Side side, double mid, Nanos ts) {
        Order o;
        o.id = next_id();
        o.instrument_id = cfg_.instrument_id;
        o.agent_id = 0; // background flow
        o.side = side;
        o.type = OrderType::Limit;
        o.tif = TimeInForce::GTC;
        o.timestamp_ns = ts;

        double raw_price = sample_price_offset(mid, side);
        // Round to tick.
        raw_price = std::round(raw_price / cfg_.tick_size) * cfg_.tick_size;
        o.price = to_tick(raw_price);
        double qty = sample_qty();
        o.orig_qty = qty;
        o.leaves_qty = qty;
        o.status = OrderStatus::New;
        return o;
    }

    Order make_market(Side side, Nanos ts) {
        Order o;
        o.id = next_id();
        o.instrument_id = cfg_.instrument_id;
        o.agent_id = 0;
        o.side = side;
        o.type = OrderType::Market;
        o.tif = TimeInForce::IOC;
        o.timestamp_ns = ts;
        o.price = (side == Side::Bid) ? to_tick(1e15) : to_tick(0.0);
        double qty = sample_qty();
        o.orig_qty = qty;
        o.leaves_qty = qty;
        o.status = OrderStatus::New;
        return o;
    }

    bool is_market() { return uni_(rng_) < cfg_.market_fraction; }
    bool is_buy() { return uni_(rng_) < 0.5; }

private:
    const SimConfig& cfg_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uni_;
    OrderId id_counter_;
};

// ── Event Record ──────────────────────────────────────────────────────────────

struct SimEventRecord {
    Nanos   timestamp_ns;
    char    type[16];     // "ORDER" | "FILL" | "SNAP"
    double  price;
    double  qty;
    int     side;         // 0 = bid, 1 = ask
    double  mid;
    double  spread;
    double  variance;
    uint64_t order_id;
};

// ── Main Simulator ────────────────────────────────────────────────────────────

class LobSimulator {
public:
    explicit LobSimulator(SimConfig cfg, uint64_t seed = 42)
        : cfg_(std::move(cfg))
        , engine_(create_matching_engine(cfg_.instrument_id))
        , gen_(cfg_, seed)
        , rng_(seed + 1)
        , heston_price_(cfg_.initial_price)
        , heston_variance_(cfg_.heston.v0)
        , sim_time_(0.0)
    {
        // Register fill callback.
        engine_->set_fill_callback([this](const Fill& f) {
            on_fill(f);
        });

        seed_initial_book();
    }

    /// Run the full simulation.
    void run() {
        if (cfg_.verbose) {
            std::cout << "[Chronos] Starting simulation: horizon=" << cfg_.horizon_secs
                      << "s, initial_price=" << cfg_.initial_price << "\n";
        }

        // Generate Heston price path.
        size_t n_heston_steps = static_cast<size_t>(cfg_.horizon_secs * 100); // 10ms steps
        auto heston_path = simulate_heston_path(
            cfg_.initial_price, cfg_.heston, n_heston_steps, cfg_.horizon_secs, rng_
        );

        // Generate order arrival times (Hawkes + noise).
        auto bid_times = simulate_hawkes(cfg_.horizon_secs, cfg_.hawkes_bid, rng_);
        auto ask_times = simulate_hawkes(cfg_.horizon_secs, cfg_.hawkes_ask, rng_);

        // Add noise traders (uniform Poisson).
        {
            std::poisson_distribution<int> poisson(cfg_.noise_intensity * cfg_.horizon_secs);
            int n_noise = poisson(rng_);
            std::uniform_real_distribution<double> uni_t(0.0, cfg_.horizon_secs);
            for (int i = 0; i < n_noise; ++i) {
                if (gen_.is_buy()) bid_times.push_back(uni_t(rng_));
                else ask_times.push_back(uni_t(rng_));
            }
        }

        // Sort all times.
        std::sort(bid_times.begin(), bid_times.end());
        std::sort(ask_times.begin(), ask_times.end());

        // Merge into a single event queue.
        struct Event {
            double t;
            Side side;
        };
        std::vector<Event> events;
        events.reserve(bid_times.size() + ask_times.size());
        for (double t : bid_times) events.push_back({t, Side::Bid});
        for (double t : ask_times) events.push_back({t, Side::Ask});
        std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
            return a.t < b.t;
        });

        // Open output file.
        std::ofstream out;
        if (cfg_.write_csv) {
            out.open(cfg_.output_path);
            if (out.is_open()) {
                out << "timestamp_ns,type,side,price,qty,mid,spread,variance,order_id\n";
            }
        }

        size_t heston_idx = 0;
        uint64_t event_count = 0;

        for (const auto& ev : events) {
            // Advance Heston to current time.
            size_t target_idx = static_cast<size_t>(ev.t / cfg_.horizon_secs * n_heston_steps);
            target_idx = std::min(target_idx, heston_path.size() - 1);
            if (target_idx > heston_idx) {
                heston_price_ = heston_path[target_idx].second;
                heston_idx = target_idx;
            }

            Nanos ts_ns = cfg_.start_ts_ns
                + static_cast<Nanos>(ev.t * 1e9);
            double mid = heston_price_;

            // Generate order.
            Order order = gen_.is_market()
                ? gen_.make_market(ev.side, ts_ns)
                : gen_.make_limit(ev.side, mid, ts_ns);

            // Submit to matching engine.
            engine_->submit(order);

            // Log order submission.
            if (cfg_.write_csv && out.is_open()) {
                MarketSnapshot snap = engine_->snapshot(5);
                double sp = snap.spread();
                out << ts_ns << ",ORDER,"
                    << (ev.side == Side::Bid ? "BID" : "ASK") << ","
                    << from_tick(order.price) << ","
                    << order.orig_qty << ","
                    << mid << ","
                    << sp << ","
                    << heston_variance_ << ","
                    << order.id << "\n";
            }

            ++event_count;

            // Periodic market snapshot.
            if (event_count % 1000 == 0) {
                MarketSnapshot snap = engine_->snapshot(5);
                if (cfg_.verbose) {
                    std::cout << "[t=" << ev.t << "s] fills=" << engine_->fill_count()
                              << " orders=" << engine_->order_count()
                              << " mid=" << snap.mid_price()
                              << " spread=" << snap.spread() << "\n";
                }

                if (cfg_.write_csv && out.is_open()) {
                    out << ts_ns << ",SNAP,,-,-,"
                        << snap.mid_price() << ","
                        << snap.spread() << ","
                        << heston_variance_ << ",-\n";
                }

                // Replenish thin books.
                replenish_book(mid, ts_ns);
            }
        }

        if (cfg_.verbose) {
            std::cout << "[Chronos] Simulation complete. Total fills: "
                      << engine_->fill_count()
                      << ", Total events: " << event_count << "\n";
        }
    }

    uint64_t total_fills() const { return engine_->fill_count(); }
    size_t active_orders() const { return engine_->order_count(); }

    MarketSnapshot snapshot() const {
        return engine_->snapshot(10);
    }

private:
    SimConfig cfg_;
    std::unique_ptr<IMatchingEngine> engine_;
    OrderGenerator gen_;
    std::mt19937_64 rng_;
    double heston_price_;
    double heston_variance_;
    double sim_time_;
    std::vector<SimEventRecord> fill_log_;

    void seed_initial_book() {
        double mid = cfg_.initial_price;
        double tick = cfg_.tick_size;
        Nanos ts = cfg_.start_ts_ns;
        uint64_t seed_id_base = 1'000'000'000ULL;

        for (int i = 1; i <= 5; ++i) {
            Order bid;
            bid.id = seed_id_base++;
            bid.instrument_id = cfg_.instrument_id;
            bid.agent_id = 0;
            bid.side = Side::Bid;
            bid.type = OrderType::Limit;
            bid.tif = TimeInForce::GTC;
            bid.price = to_tick(mid - i * tick);
            bid.orig_qty = 200.0;
            bid.leaves_qty = 200.0;
            bid.timestamp_ns = ts;
            bid.seq = 0;
            bid.status = OrderStatus::New;
            engine_->submit(bid);

            Order ask;
            ask.id = seed_id_base++;
            ask.instrument_id = cfg_.instrument_id;
            ask.agent_id = 0;
            ask.side = Side::Ask;
            ask.type = OrderType::Limit;
            ask.tif = TimeInForce::GTC;
            ask.price = to_tick(mid + i * tick);
            ask.orig_qty = 200.0;
            ask.leaves_qty = 200.0;
            ask.timestamp_ns = ts;
            ask.seq = 0;
            ask.status = OrderStatus::New;
            engine_->submit(ask);
        }
    }

    void replenish_book(double mid, Nanos ts) {
        MarketSnapshot snap = engine_->snapshot(5);
        double tick = cfg_.tick_size;
        static uint64_t replenish_id = 2'000'000'000ULL;

        if (snap.bid_levels < 2) {
            for (int i = 1; i <= 3; ++i) {
                Order b;
                b.id = replenish_id++;
                b.instrument_id = cfg_.instrument_id;
                b.agent_id = 0;
                b.side = Side::Bid;
                b.type = OrderType::Limit;
                b.tif = TimeInForce::GTC;
                b.price = to_tick(mid - i * tick);
                b.orig_qty = 100.0;
                b.leaves_qty = 100.0;
                b.timestamp_ns = ts;
                b.status = OrderStatus::New;
                engine_->submit(b);
            }
        }
        if (snap.ask_levels < 2) {
            for (int i = 1; i <= 3; ++i) {
                Order a;
                a.id = replenish_id++;
                a.instrument_id = cfg_.instrument_id;
                a.agent_id = 0;
                a.side = Side::Ask;
                a.type = OrderType::Limit;
                a.tif = TimeInForce::GTC;
                a.price = to_tick(mid + i * tick);
                a.orig_qty = 100.0;
                a.leaves_qty = 100.0;
                a.timestamp_ns = ts;
                a.status = OrderStatus::New;
                engine_->submit(a);
            }
        }
    }

    void on_fill(const Fill& f) {
        SimEventRecord rec;
        rec.timestamp_ns = f.timestamp_ns;
        std::strncpy(rec.type, "FILL", 15);
        rec.price = from_tick(f.price);
        rec.qty = f.qty;
        rec.side = (f.side == Side::Bid) ? 0 : 1;
        rec.mid = heston_price_;
        rec.spread = 0.0; // filled later
        rec.variance = heston_variance_;
        rec.order_id = f.aggressor_id;
        fill_log_.push_back(rec);
    }
};

// ── Entry point helper ────────────────────────────────────────────────────────

int run_simulation(const SimConfig& cfg, uint64_t seed) {
    LobSimulator sim(cfg, seed);
    sim.run();
    std::cout << "Total fills: " << sim.total_fills()
              << ", Active orders: " << sim.active_orders() << "\n";
    return 0;
}

} // namespace simulator
} // namespace chronos

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    using namespace chronos::simulator;
    SimConfig cfg;
    cfg.horizon_secs   = 3600.0;
    cfg.initial_price  = 100.0;
    cfg.tick_size      = 0.01;
    cfg.output_path    = "sim_events.csv";
    cfg.write_csv      = true;
    cfg.verbose        = true;

    // Parse simple CLI args.
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string key(argv[i]);
        std::string val(argv[i + 1]);
        if (key == "--horizon") cfg.horizon_secs = std::stod(val);
        else if (key == "--price") cfg.initial_price = std::stod(val);
        else if (key == "--tick") cfg.tick_size = std::stod(val);
        else if (key == "--out") cfg.output_path = val;
        else if (key == "--seed") { /* seed handled below */ }
    }

    uint64_t seed = 42;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string key(argv[i]);
        if (key == "--seed") seed = std::stoull(argv[i + 1]);
    }

    return run_simulation(cfg, seed);
}
