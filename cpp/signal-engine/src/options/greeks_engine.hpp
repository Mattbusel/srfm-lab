#pragma once
#include <cstdint>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <limits>

namespace srfm::options {

enum class OptionType : uint8_t { Call = 0, Put = 1 };

struct GreeksResult {
    double price;
    double delta, gamma, vega, theta, rho;
    double vanna, volga, charm, vomma, speed, zomma, color;
};

struct OptionPosition {
    OptionType type;
    double strike;
    double expiry;     // time to expiry in years
    double iv;         // implied vol
    double quantity;   // signed: positive=long, negative=short
    double underlying; // current spot
    double rate;       // risk-free rate
};

struct PortfolioGreeks {
    double net_delta, net_gamma, net_vega, net_theta, net_rho;
    double net_vanna, net_volga, net_charm;
    double dollar_delta, dollar_gamma, dollar_vega, dollar_theta;
    int n_positions;
};

struct PnLExplain {
    double total_pnl;
    double theta_pnl;
    double delta_pnl;
    double gamma_pnl;
    double vega_pnl;
    double residual;
};

struct ScenarioResult {
    double base_value;
    double scenario_value;
    double pnl;
    PortfolioGreeks scenario_greeks;
};

struct SVIParams {
    double a, b, rho, m, sigma; // SVI raw parameterization
};

struct VolPoint {
    double strike;
    double expiry;
    double vol;
};

struct SkewMetrics {
    double atm_vol;
    double rr25;       // 25-delta risk reversal
    double bf25;       // 25-delta butterfly
    double term_slope; // term structure slope
    double skew_slope; // skew per unit moneyness
};

// ----------- Black-Scholes Engine -----------
class BSEngine {
public:
    static double norm_cdf(double x);
    static double norm_pdf(double x);

    static double call_price(double S, double K, double T, double r, double sigma);
    static double put_price(double S, double K, double T, double r, double sigma);
    static double price(double S, double K, double T, double r, double sigma, OptionType type);

    static GreeksResult greeks(double S, double K, double T, double r, double sigma, OptionType type);

    static double delta(double S, double K, double T, double r, double sigma, OptionType type);
    static double gamma(double S, double K, double T, double r, double sigma);
    static double vega(double S, double K, double T, double r, double sigma);
    static double theta(double S, double K, double T, double r, double sigma, OptionType type);
    static double rho(double S, double K, double T, double r, double sigma, OptionType type);
    static double vanna(double S, double K, double T, double r, double sigma);
    static double volga(double S, double K, double T, double r, double sigma);
    static double charm(double S, double K, double T, double r, double sigma, OptionType type);
    static double vomma(double S, double K, double T, double r, double sigma);
    static double speed(double S, double K, double T, double r, double sigma);
    static double zomma(double S, double K, double T, double r, double sigma);
    static double color(double S, double K, double T, double r, double sigma);

private:
    static void d1d2(double S, double K, double T, double r, double sigma, double& d1, double& d2);
};

// ----------- Implied Vol Solver -----------
class IVSolver {
public:
    struct Config {
        double tol = 1e-10;
        int max_iter_nr = 50;
        int max_iter_brent = 100;
        double vol_lower = 0.001;
        double vol_upper = 5.0;
    };

    IVSolver();
    explicit IVSolver(const Config& cfg);

    double solve(double market_price, double S, double K, double T, double r, OptionType type) const;
    double jaeckel_seed(double market_price, double S, double K, double T, double r, OptionType type) const;

private:
    double newton_raphson(double market_price, double S, double K, double T, double r, OptionType type, double seed) const;
    double brent(double market_price, double S, double K, double T, double r, OptionType type) const;
    Config config_;
};

// ----------- Portfolio Greeks Aggregator -----------
class PortfolioGreeksAgg {
public:
    PortfolioGreeksAgg() = default;

    void clear();
    void add_position(const OptionPosition& pos);
    void remove_position(int index);
    PortfolioGreeks compute() const;
    PnLExplain explain_pnl(double dS, double dT, double dSigma) const;
    ScenarioResult scenario(double spot_bump, double vol_bump, double time_bump) const;

    int position_count() const { return static_cast<int>(positions_.size()); }
    const OptionPosition& position(int i) const { return positions_[i]; }

private:
    std::vector<OptionPosition> positions_;
};

// ----------- Vol Surface -----------
class VolSurface {
public:
    struct Config {
        int max_strikes = 100;
        int max_expiries = 20;
    };

    VolSurface();
    explicit VolSurface(const Config& cfg);

    void add_point(double strike, double expiry, double vol);
    void clear();
    double interpolate(double strike, double expiry) const;
    SVIParams fit_svi(double expiry) const;
    double svi_vol(const SVIParams& params, double strike, double forward) const;
    SkewMetrics compute_skew(double spot, double expiry, double rate) const;

    int point_count() const { return static_cast<int>(points_.size()); }

private:
    double cubic_interp_1d(const std::vector<double>& xs, const std::vector<double>& ys, double x) const;
    Config config_;
    std::vector<VolPoint> points_;
};

// ----------- SIMD Helpers -----------
namespace simd {
    void vec_exp(const double* in, double* out, int n);
    void vec_log(const double* in, double* out, int n);
    void vec_norm_cdf(const double* in, double* out, int n);
    void vec_bs_call(const double* S, const double* K, const double* T,
                     const double* r, const double* sigma, double* out, int n);
}

} // namespace srfm::options
