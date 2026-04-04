#include "market_impact.hpp"
// Implementation is header-only via templates/inline definitions.
// This .cpp exists for any future non-template implementations
// and to satisfy the build system's explicit file list.

namespace hft {

// Validate Almgren-Chriss parameters
bool validate_ac_params(const AlmgrenChrissParams& p) {
    if (p.sigma  <= 0) return false;
    if (p.eta    <= 0) return false;
    if (p.lambda <= 0) return false;
    if (p.tau    <= 0) return false;
    if (p.N      <= 0) return false;
    return true;
}

// Compute the price trajectory under optimal execution
// Returns vector of prices at each time step
std::vector<double> price_trajectory(
    const AlmgrenChrissParams& p,
    double S0,     // initial price
    double X,      // shares to execute
    unsigned seed)
{
    AlmgrenChriss ac(p);
    auto sched = ac.optimal_schedule(X);

    std::vector<double> prices(p.N + 1);
    prices[0] = S0;

    // Simple random walk with drift from market impact
    // Use LCG for reproducibility
    uint64_t rng = seed;
    auto randn = [&]() -> double {
        // Box-Muller transform
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double u1 = (rng >> 11) / static_cast<double>(1ULL << 53);
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double u2 = (rng >> 11) / static_cast<double>(1ULL << 53);
        u1 = std::max(u1, 1e-15);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    };

    for (int t = 0; t < p.N; ++t) {
        double n_t     = sched.trade_list[t];
        double perm    = p.gamma * n_t;
        double diffuse = p.sigma * std::sqrt(p.tau) * randn();
        prices[t+1]    = prices[t] - perm + diffuse;
    }
    return prices;
}

// Monte Carlo estimation of execution cost distribution
struct MonteCarloCost {
    double mean;
    double std_dev;
    double percentile_5;
    double percentile_95;
};

MonteCarloCost monte_carlo_cost(
    const AlmgrenChrissParams& p, double X,
    int num_paths, unsigned seed)
{
    AlmgrenChriss ac(p);
    auto sched = ac.optimal_schedule(X);

    std::vector<double> costs(num_paths);
    uint64_t rng = seed;

    for (int path = 0; path < num_paths; ++path) {
        double total_cost = 0.0;
        double S = 100.0; // normalized starting price

        for (int t = 0; t < p.N; ++t) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            double u1 = std::max((rng >> 11) / static_cast<double>(1ULL << 53), 1e-15);
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            double u2 = (rng >> 11) / static_cast<double>(1ULL << 53);
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);

            double n_t      = sched.trade_list[t];
            double v_t      = n_t / p.tau;
            double temp_imp = p.epsilon * (n_t > 0 ? 1 : -1) + p.eta * v_t;
            double exec_price = S - temp_imp;

            total_cost += n_t * exec_price;

            // Update mid-price
            double perm = p.gamma * n_t;
            double diff = p.sigma * std::sqrt(p.tau) * z;
            S += -perm + diff;
        }
        costs[path] = total_cost;
    }

    std::sort(costs.begin(), costs.end());
    double mean = std::accumulate(costs.begin(), costs.end(), 0.0) / num_paths;
    double var  = 0.0;
    for (auto c : costs) var += (c - mean) * (c - mean);
    var /= num_paths;

    MonteCarloCost mc{};
    mc.mean          = mean;
    mc.std_dev       = std::sqrt(var);
    mc.percentile_5  = costs[static_cast<size_t>(0.05 * num_paths)];
    mc.percentile_95 = costs[static_cast<size_t>(0.95 * num_paths)];
    return mc;
}

} // namespace hft
