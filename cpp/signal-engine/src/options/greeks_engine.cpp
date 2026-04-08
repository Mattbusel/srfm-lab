#include "greeks_engine.hpp"
#include <cstring>
#include <cfloat>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace srfm::options {

static constexpr double PI = 3.14159265358979323846;
static constexpr double SQRT_2PI = 2.5066282746310002;
static constexpr double INV_SQRT_2 = 0.7071067811865476;
static constexpr double ONE_OVER_SQRT_2PI = 0.3989422804014327;

// ============================================================================
// BSEngine: Normal distribution functions
// ============================================================================

double BSEngine::norm_pdf(double x) {
    return ONE_OVER_SQRT_2PI * std::exp(-0.5 * x * x);
}

double BSEngine::norm_cdf(double x) {
    // Abramowitz & Stegun approximation 7.1.26, max error 1.5e-7
    // For higher precision, use erfc-based
    return 0.5 * std::erfc(-x * INV_SQRT_2);
}

void BSEngine::d1d2(double S, double K, double T, double r, double sigma,
                     double& d1, double& d2) {
    double sqrt_t = std::sqrt(T);
    double sigma_sqrt_t = sigma * sqrt_t;
    d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / sigma_sqrt_t;
    d2 = d1 - sigma_sqrt_t;
}

double BSEngine::call_price(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(S - K, 0.0);
    if (sigma <= 0.0) return std::max(S - K * std::exp(-r * T), 0.0);
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double BSEngine::put_price(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(K - S, 0.0);
    if (sigma <= 0.0) return std::max(K * std::exp(-r * T) - S, 0.0);
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

double BSEngine::price(double S, double K, double T, double r, double sigma, OptionType type) {
    return (type == OptionType::Call) ? call_price(S, K, T, r, sigma) : put_price(S, K, T, r, sigma);
}

// ============================================================================
// BSEngine: Individual Greeks
// ============================================================================

double BSEngine::delta(double S, double K, double T, double r, double sigma, OptionType type) {
    if (T <= 0.0 || sigma <= 0.0) {
        if (type == OptionType::Call) return (S > K) ? 1.0 : 0.0;
        return (S < K) ? -1.0 : 0.0;
    }
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    if (type == OptionType::Call) return norm_cdf(d1);
    return norm_cdf(d1) - 1.0;
}

double BSEngine::gamma(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    return norm_pdf(d1) / (S * sigma * std::sqrt(T));
}

double BSEngine::vega(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    return S * norm_pdf(d1) * std::sqrt(T) * 0.01; // per 1% vol
}

double BSEngine::theta(double S, double K, double T, double r, double sigma, OptionType type) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    double sqrt_t = std::sqrt(T);
    double term1 = -S * norm_pdf(d1) * sigma / (2.0 * sqrt_t);
    if (type == OptionType::Call) {
        return (term1 - r * K * std::exp(-r * T) * norm_cdf(d2)) / 365.0;
    }
    return (term1 + r * K * std::exp(-r * T) * norm_cdf(-d2)) / 365.0;
}

double BSEngine::rho(double S, double K, double T, double r, double sigma, OptionType type) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    if (type == OptionType::Call) {
        return K * T * std::exp(-r * T) * norm_cdf(d2) * 0.01;
    }
    return -K * T * std::exp(-r * T) * norm_cdf(-d2) * 0.01;
}

double BSEngine::vanna(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    return -norm_pdf(d1) * d2 / sigma;
}

double BSEngine::volga(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    double v = vega(S, K, T, r, sigma) * 100.0; // un-scale
    return v * d1 * d2 / sigma;
}

double BSEngine::charm(double S, double K, double T, double r, double sigma, OptionType type) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    double sqrt_t = std::sqrt(T);
    double pdf_d1 = norm_pdf(d1);
    double term = 2.0 * (r * T - d2 * sigma * sqrt_t) / (2.0 * T * sigma * sqrt_t);
    if (type == OptionType::Call) {
        return -pdf_d1 * term / 365.0;
    }
    return -pdf_d1 * term / 365.0;
}

double BSEngine::vomma(double S, double K, double T, double r, double sigma) {
    return volga(S, K, T, r, sigma); // synonym
}

double BSEngine::speed(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0 || S <= 0.0) return 0.0;
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    double g = gamma(S, K, T, r, sigma);
    double sigma_sqrt_t = sigma * std::sqrt(T);
    return -g / S * (d1 / sigma_sqrt_t + 1.0);
}

double BSEngine::zomma(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    double g = gamma(S, K, T, r, sigma);
    return g * (d1 * d2 - 1.0) / sigma;
}

double BSEngine::color(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    double sqrt_t = std::sqrt(T);
    double pdf_d1 = norm_pdf(d1);
    double term = 2.0 * r * T - d2 * sigma * sqrt_t;
    return -pdf_d1 / (2.0 * S * T * sigma * sqrt_t) *
           (1.0 + d1 * term / (sigma * sqrt_t)) / 365.0;
}

// ============================================================================
// BSEngine: Full Greeks computation
// ============================================================================

GreeksResult BSEngine::greeks(double S, double K, double T, double r, double sigma, OptionType type) {
    GreeksResult g{};
    g.price = price(S, K, T, r, sigma, type);
    if (T <= 0.0 || sigma <= 0.0) {
        g.delta = delta(S, K, T, r, sigma, type);
        return g;
    }
    double d1, d2;
    d1d2(S, K, T, r, sigma, d1, d2);
    double sqrt_t = std::sqrt(T);
    double sigma_sqrt_t = sigma * sqrt_t;
    double pdf_d1 = norm_pdf(d1);
    double cdf_d1 = norm_cdf(d1);
    double cdf_d2 = norm_cdf(d2);
    double disc = std::exp(-r * T);

    // Delta
    g.delta = (type == OptionType::Call) ? cdf_d1 : cdf_d1 - 1.0;

    // Gamma
    g.gamma = pdf_d1 / (S * sigma_sqrt_t);

    // Vega (per 1%)
    g.vega = S * pdf_d1 * sqrt_t * 0.01;

    // Theta (per day)
    double theta_term1 = -S * pdf_d1 * sigma / (2.0 * sqrt_t);
    if (type == OptionType::Call) {
        g.theta = (theta_term1 - r * K * disc * cdf_d2) / 365.0;
    } else {
        g.theta = (theta_term1 + r * K * disc * norm_cdf(-d2)) / 365.0;
    }

    // Rho (per 1%)
    if (type == OptionType::Call) {
        g.rho = K * T * disc * cdf_d2 * 0.01;
    } else {
        g.rho = -K * T * disc * norm_cdf(-d2) * 0.01;
    }

    // Vanna
    g.vanna = -pdf_d1 * d2 / sigma;

    // Volga / Vomma
    double raw_vega = S * pdf_d1 * sqrt_t;
    g.volga = raw_vega * d1 * d2 / sigma;
    g.vomma = g.volga;

    // Charm (per day)
    double charm_term = 2.0 * (r * T - d2 * sigma_sqrt_t) / (2.0 * T * sigma_sqrt_t);
    g.charm = -pdf_d1 * charm_term / 365.0;

    // Speed
    g.speed = -g.gamma / S * (d1 / sigma_sqrt_t + 1.0);

    // Zomma
    g.zomma = g.gamma * (d1 * d2 - 1.0) / sigma;

    // Color (per day)
    double color_term = 2.0 * r * T - d2 * sigma_sqrt_t;
    g.color = -pdf_d1 / (2.0 * S * T * sigma_sqrt_t) *
              (1.0 + d1 * color_term / sigma_sqrt_t) / 365.0;

    return g;
}

// ============================================================================
// IVSolver Implementation
// ============================================================================

IVSolver::IVSolver() : config_() {}
IVSolver::IVSolver(const Config& cfg) : config_(cfg) {}

double IVSolver::jaeckel_seed(double market_price, double S, double K, double T, double r, OptionType type) const {
    // Rational approximation for initial implied vol guess
    double disc = std::exp(-r * T);
    double F = S / disc;
    double x = std::log(F / K);
    double intrinsic = (type == OptionType::Call) ? std::max(S - K * disc, 0.0) : std::max(K * disc - S, 0.0);
    double time_value = market_price - intrinsic;

    if (time_value <= 0.0) return 0.2; // fallback

    // Brenner-Subrahmanyam approximation: sigma ~ sqrt(2*pi/T) * C/S
    double approx = std::sqrt(2.0 * PI / T) * market_price / S;
    return std::clamp(approx, config_.vol_lower, config_.vol_upper);
}

double IVSolver::newton_raphson(double market_price, double S, double K, double T, double r,
                                 OptionType type, double seed) const {
    double sigma = seed;
    for (int i = 0; i < config_.max_iter_nr; ++i) {
        double model_price = BSEngine::price(S, K, T, r, sigma, type);
        double diff = model_price - market_price;
        if (std::abs(diff) < config_.tol) return sigma;

        double v = BSEngine::vega(S, K, T, r, sigma) * 100.0; // un-scale
        if (v < 1e-15) break; // vega too small, NR won't converge

        sigma -= diff / v;
        sigma = std::clamp(sigma, config_.vol_lower, config_.vol_upper);
    }
    return sigma;
}

double IVSolver::brent(double market_price, double S, double K, double T, double r, OptionType type) const {
    double a = config_.vol_lower, b = config_.vol_upper;
    double fa = BSEngine::price(S, K, T, r, a, type) - market_price;
    double fb = BSEngine::price(S, K, T, r, b, type) - market_price;

    if (fa * fb > 0) return 0.2; // no bracket

    if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }

    double c = a, fc = fa;
    bool mflag = true;
    double s = 0, d_prev = 0;

    for (int i = 0; i < config_.max_iter_brent; ++i) {
        if (std::abs(fb) < config_.tol) return b;
        if (std::abs(b - a) < config_.tol) return b;

        if (std::abs(fa - fc) > 1e-15 && std::abs(fb - fc) > 1e-15) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
              + b * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            s = b - fb * (b - a) / (fb - fa); // secant
        }

        bool cond1 = !((s > (3.0 * a + b) / 4.0) && (s < b));
        bool cond2 = mflag && (std::abs(s - b) >= std::abs(b - c) / 2.0);
        bool cond3 = !mflag && (std::abs(s - b) >= std::abs(c - d_prev) / 2.0);
        bool cond4 = mflag && (std::abs(b - c) < config_.tol);
        bool cond5 = !mflag && (std::abs(c - d_prev) < config_.tol);

        if (cond1 || cond2 || cond3 || cond4 || cond5) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = BSEngine::price(S, K, T, r, s, type) - market_price;
        d_prev = c;
        c = b; fc = fb;

        if (fa * fs < 0) { b = s; fb = fs; }
        else { a = s; fa = fs; }

        if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }
    }
    return b;
}

double IVSolver::solve(double market_price, double S, double K, double T, double r, OptionType type) const {
    if (market_price <= 0.0 || T <= 0.0 || S <= 0.0 || K <= 0.0) return 0.0;

    double seed = jaeckel_seed(market_price, S, K, T, r, type);
    double result = newton_raphson(market_price, S, K, T, r, type, seed);

    // Verify convergence
    double model = BSEngine::price(S, K, T, r, result, type);
    if (std::abs(model - market_price) > config_.tol * 100.0) {
        result = brent(market_price, S, K, T, r, type);
    }
    return result;
}

// ============================================================================
// PortfolioGreeksAgg Implementation
// ============================================================================

void PortfolioGreeksAgg::clear() { positions_.clear(); }

void PortfolioGreeksAgg::add_position(const OptionPosition& pos) {
    positions_.push_back(pos);
}

void PortfolioGreeksAgg::remove_position(int index) {
    if (index >= 0 && index < static_cast<int>(positions_.size())) {
        positions_.erase(positions_.begin() + index);
    }
}

PortfolioGreeks PortfolioGreeksAgg::compute() const {
    PortfolioGreeks pg{};
    pg.n_positions = static_cast<int>(positions_.size());

    for (const auto& pos : positions_) {
        GreeksResult g = BSEngine::greeks(pos.underlying, pos.strike, pos.expiry,
                                           pos.rate, pos.iv, pos.type);
        double q = pos.quantity;
        pg.net_delta += g.delta * q;
        pg.net_gamma += g.gamma * q;
        pg.net_vega += g.vega * q;
        pg.net_theta += g.theta * q;
        pg.net_rho += g.rho * q;
        pg.net_vanna += g.vanna * q;
        pg.net_volga += g.volga * q;
        pg.net_charm += g.charm * q;
        pg.dollar_delta += g.delta * q * pos.underlying;
        pg.dollar_gamma += 0.5 * g.gamma * q * pos.underlying * pos.underlying * 0.01;
        pg.dollar_vega += g.vega * q;
        pg.dollar_theta += g.theta * q;
    }
    return pg;
}

PnLExplain PortfolioGreeksAgg::explain_pnl(double dS, double dT, double dSigma) const {
    PnLExplain pnl{};
    for (const auto& pos : positions_) {
        GreeksResult g = BSEngine::greeks(pos.underlying, pos.strike, pos.expiry,
                                           pos.rate, pos.iv, pos.type);
        double q = pos.quantity;

        double theta_c = g.theta * q * (-dT * 365.0); // dT is negative for time passing
        double delta_c = g.delta * q * dS;
        double gamma_c = 0.5 * g.gamma * q * dS * dS;
        double vega_c = g.vega * q * dSigma * 100.0; // dSigma in decimal, vega per 1%

        // Actual P&L
        double new_price = BSEngine::price(pos.underlying + dS, pos.strike,
                                            pos.expiry + dT, pos.rate,
                                            pos.iv + dSigma, pos.type);
        double old_price = BSEngine::price(pos.underlying, pos.strike, pos.expiry,
                                            pos.rate, pos.iv, pos.type);
        double actual = (new_price - old_price) * q;

        pnl.theta_pnl += theta_c;
        pnl.delta_pnl += delta_c;
        pnl.gamma_pnl += gamma_c;
        pnl.vega_pnl += vega_c;
        pnl.total_pnl += actual;
    }
    pnl.residual = pnl.total_pnl - pnl.theta_pnl - pnl.delta_pnl - pnl.gamma_pnl - pnl.vega_pnl;
    return pnl;
}

ScenarioResult PortfolioGreeksAgg::scenario(double spot_bump, double vol_bump, double time_bump) const {
    ScenarioResult sr{};
    sr.base_value = 0;
    sr.scenario_value = 0;

    std::vector<OptionPosition> bumped;
    for (const auto& pos : positions_) {
        double base_px = BSEngine::price(pos.underlying, pos.strike, pos.expiry,
                                          pos.rate, pos.iv, pos.type);
        sr.base_value += base_px * pos.quantity;

        OptionPosition bp = pos;
        bp.underlying += spot_bump;
        bp.iv += vol_bump;
        bp.expiry += time_bump;
        if (bp.expiry < 0) bp.expiry = 0;
        if (bp.iv < 0.001) bp.iv = 0.001;

        double scen_px = BSEngine::price(bp.underlying, bp.strike, bp.expiry,
                                          bp.rate, bp.iv, bp.type);
        sr.scenario_value += scen_px * pos.quantity;
        bumped.push_back(bp);
    }
    sr.pnl = sr.scenario_value - sr.base_value;

    // Compute scenario greeks
    PortfolioGreeksAgg temp;
    for (const auto& bp : bumped) temp.add_position(bp);
    sr.scenario_greeks = temp.compute();

    return sr;
}

// ============================================================================
// VolSurface Implementation
// ============================================================================

VolSurface::VolSurface() : config_() {}
VolSurface::VolSurface(const Config& cfg) : config_(cfg) {}

void VolSurface::add_point(double strike, double expiry, double vol) {
    points_.push_back({strike, expiry, vol});
}

void VolSurface::clear() { points_.clear(); }

double VolSurface::cubic_interp_1d(const std::vector<double>& xs, const std::vector<double>& ys, double x) const {
    int n = static_cast<int>(xs.size());
    if (n == 0) return 0.0;
    if (n == 1) return ys[0];
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];

    // Find bracket
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xs[mid] > x) hi = mid;
        else lo = mid;
    }

    // Catmull-Rom style cubic with boundary handling
    int p0 = std::max(lo - 1, 0);
    int p1 = lo;
    int p2 = hi;
    int p3 = std::min(hi + 1, n - 1);

    double t = (x - xs[p1]) / (xs[p2] - xs[p1]);
    double t2 = t * t, t3 = t2 * t;

    double a0 = -0.5 * ys[p0] + 1.5 * ys[p1] - 1.5 * ys[p2] + 0.5 * ys[p3];
    double a1 = ys[p0] - 2.5 * ys[p1] + 2.0 * ys[p2] - 0.5 * ys[p3];
    double a2 = -0.5 * ys[p0] + 0.5 * ys[p2];
    double a3 = ys[p1];

    return a0 * t3 + a1 * t2 + a2 * t + a3;
}

double VolSurface::interpolate(double strike, double expiry) const {
    if (points_.empty()) return 0.2;

    // Collect unique expiries
    std::vector<double> expiries;
    for (const auto& p : points_) {
        bool found = false;
        for (auto e : expiries) { if (std::abs(e - p.expiry) < 1e-8) { found = true; break; } }
        if (!found) expiries.push_back(p.expiry);
    }
    std::sort(expiries.begin(), expiries.end());

    // For each expiry slice, interpolate in strike dimension
    std::vector<double> vols_at_expiries;
    for (double exp : expiries) {
        std::vector<double> strikes, vols;
        for (const auto& p : points_) {
            if (std::abs(p.expiry - exp) < 1e-8) {
                strikes.push_back(p.strike);
                vols.push_back(p.vol);
            }
        }
        // Sort by strike
        std::vector<int> idx(strikes.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return strikes[a] < strikes[b]; });
        std::vector<double> ss(strikes.size()), vs(vols.size());
        for (size_t i = 0; i < idx.size(); ++i) { ss[i] = strikes[idx[i]]; vs[i] = vols[idx[i]]; }

        vols_at_expiries.push_back(cubic_interp_1d(ss, vs, strike));
    }

    // Interpolate in expiry dimension
    return cubic_interp_1d(expiries, vols_at_expiries, expiry);
}

SVIParams VolSurface::fit_svi(double expiry) const {
    // Gather points at this expiry
    std::vector<double> strikes, total_vars;
    for (const auto& p : points_) {
        if (std::abs(p.expiry - expiry) < 1e-8) {
            strikes.push_back(p.strike);
            total_vars.push_back(p.vol * p.vol * expiry);
        }
    }

    // Simple SVI fit using grid search + Nelder-Mead-like refinement
    // w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    // where k = log(K/F)
    SVIParams best{0.04, 0.1, -0.3, 0.0, 0.1};
    if (strikes.size() < 3) return best;

    double best_err = std::numeric_limits<double>::max();
    // Grid search
    for (double a = 0.01; a <= 0.10; a += 0.01) {
        for (double b = 0.05; b <= 0.30; b += 0.05) {
            for (double rho = -0.9; rho <= 0.1; rho += 0.2) {
                for (double m = -0.2; m <= 0.2; m += 0.1) {
                    for (double sig = 0.05; sig <= 0.30; sig += 0.05) {
                        double err = 0;
                        for (size_t i = 0; i < strikes.size(); ++i) {
                            double k = std::log(strikes[i]); // simplified, assumes F~strike midpoint
                            double w = a + b * (rho * (k - m) + std::sqrt((k - m) * (k - m) + sig * sig));
                            double diff = w - total_vars[i];
                            err += diff * diff;
                        }
                        if (err < best_err) {
                            best_err = err;
                            best = {a, b, rho, m, sig};
                        }
                    }
                }
            }
        }
    }
    return best;
}

double VolSurface::svi_vol(const SVIParams& params, double strike, double forward) const {
    double k = std::log(strike / forward);
    double w = params.a + params.b * (params.rho * (k - params.m) +
               std::sqrt((k - params.m) * (k - params.m) + params.sigma * params.sigma));
    return (w > 0) ? std::sqrt(w) : 0.01;
}

SkewMetrics VolSurface::compute_skew(double spot, double expiry, double rate) const {
    SkewMetrics sm{};
    double F = spot * std::exp(rate * expiry);
    sm.atm_vol = interpolate(F, expiry);

    // 25-delta strikes approximation
    double sqrt_t = std::sqrt(expiry);
    double d25_call = F * std::exp(-0.675 * sm.atm_vol * sqrt_t + 0.5 * sm.atm_vol * sm.atm_vol * expiry);
    double d25_put = F * std::exp(0.675 * sm.atm_vol * sqrt_t + 0.5 * sm.atm_vol * sm.atm_vol * expiry);

    double vol_25c = interpolate(d25_call, expiry);
    double vol_25p = interpolate(d25_put, expiry);

    sm.rr25 = vol_25c - vol_25p;
    sm.bf25 = 0.5 * (vol_25c + vol_25p) - sm.atm_vol;

    // Term structure slope
    double vol_short = interpolate(F, expiry * 0.5);
    double vol_long = interpolate(F, expiry * 2.0);
    sm.term_slope = (vol_long - vol_short) / (expiry * 1.5);

    // Skew slope per unit moneyness
    double vol_up = interpolate(F * 1.05, expiry);
    double vol_down = interpolate(F * 0.95, expiry);
    sm.skew_slope = (vol_up - vol_down) / 0.10;

    return sm;
}

// ============================================================================
// SIMD Helpers
// ============================================================================

namespace simd {

void vec_exp(const double* in, double* out, int n) {
#ifdef __AVX2__
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d v = _mm256_loadu_pd(in + i);
        // Polynomial approx: e^x ~ 1 + x + x^2/2 + x^3/6 + x^4/24 + x^5/120
        __m256d one = _mm256_set1_pd(1.0);
        __m256d half = _mm256_set1_pd(0.5);
        __m256d sixth = _mm256_set1_pd(1.0 / 6.0);
        __m256d twentyfourth = _mm256_set1_pd(1.0 / 24.0);
        __m256d one_twenty = _mm256_set1_pd(1.0 / 120.0);
        // Range reduce: e^x = 2^k * e^r where r = x - k*ln2
        __m256d log2e = _mm256_set1_pd(1.4426950408889634);
        __m256d ln2 = _mm256_set1_pd(0.6931471805599453);
        __m256d k = _mm256_round_pd(_mm256_mul_pd(v, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d r = _mm256_sub_pd(v, _mm256_mul_pd(k, ln2));
        __m256d r2 = _mm256_mul_pd(r, r);
        __m256d r3 = _mm256_mul_pd(r2, r);
        __m256d r4 = _mm256_mul_pd(r3, r);
        __m256d r5 = _mm256_mul_pd(r4, r);
        __m256d result = _mm256_add_pd(one, r);
        result = _mm256_add_pd(result, _mm256_mul_pd(r2, half));
        result = _mm256_add_pd(result, _mm256_mul_pd(r3, sixth));
        result = _mm256_add_pd(result, _mm256_mul_pd(r4, twentyfourth));
        result = _mm256_add_pd(result, _mm256_mul_pd(r5, one_twenty));
        // Scale by 2^k using bit manipulation
        // For simplicity, store and use scalar ldexp
        double res[4], ks[4];
        _mm256_storeu_pd(res, result);
        _mm256_storeu_pd(ks, k);
        for (int j = 0; j < 4; ++j) {
            out[i + j] = std::ldexp(res[j], static_cast<int>(ks[j]));
        }
    }
    for (; i < n; ++i) out[i] = std::exp(in[i]);
#else
    for (int i = 0; i < n; ++i) out[i] = std::exp(in[i]);
#endif
}

void vec_log(const double* in, double* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = std::log(in[i]);
}

void vec_norm_cdf(const double* in, double* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = BSEngine::norm_cdf(in[i]);
}

void vec_bs_call(const double* S, const double* K, const double* T,
                  const double* r, const double* sigma, double* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = BSEngine::call_price(S[i], K[i], T[i], r[i], sigma[i]);
    }
}

void vec_bs_put(const double* S, const double* K, const double* T,
                 const double* r, const double* sigma, double* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = BSEngine::put_price(S[i], K[i], T[i], r[i], sigma[i]);
    }
}

void vec_delta(const double* S, const double* K, const double* T,
               const double* r, const double* sigma, const OptionType* types, double* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = BSEngine::delta(S[i], K[i], T[i], r[i], sigma[i], types[i]);
    }
}

void vec_gamma(const double* S, const double* K, const double* T,
               const double* r, const double* sigma, double* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = BSEngine::gamma(S[i], K[i], T[i], r[i], sigma[i]);
    }
}

void vec_vega(const double* S, const double* K, const double* T,
              const double* r, const double* sigma, double* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = BSEngine::vega(S[i], K[i], T[i], r[i], sigma[i]);
    }
}

void vec_iv_solve(const double* prices, const double* S, const double* K,
                  const double* T, const double* r, const OptionType* types,
                  double* out, int n) {
    IVSolver solver;
    for (int i = 0; i < n; ++i) {
        out[i] = solver.solve(prices[i], S[i], K[i], T[i], r[i], types[i]);
    }
}

} // namespace simd

// ============================================================================
// Additional Vol Surface utilities
// ============================================================================

namespace vol_utils {

// SABR model: alpha, beta, rho, nu
struct SABRParams {
    double alpha; // initial vol
    double beta;  // CEV exponent (0=normal, 1=lognormal)
    double rho;   // correlation
    double nu;    // vol of vol
};

// Hagan SABR approximation for implied vol
double sabr_implied_vol(const SABRParams& p, double F, double K, double T) {
    if (std::abs(F - K) < 1e-10) {
        // ATM formula
        double FK_beta = std::pow(F, 1.0 - p.beta);
        double term1 = p.alpha / FK_beta;
        double A = (1.0 - p.beta) * (1.0 - p.beta) * p.alpha * p.alpha / (24.0 * FK_beta * FK_beta);
        double B = 0.25 * p.rho * p.beta * p.nu * p.alpha / FK_beta;
        double C = (2.0 - 3.0 * p.rho * p.rho) * p.nu * p.nu / 24.0;
        return term1 * (1.0 + (A + B + C) * T);
    }

    double log_FK = std::log(F / K);
    double FK_mid = std::sqrt(F * K);
    double FK_beta = std::pow(FK_mid, 1.0 - p.beta);
    double z = p.nu / p.alpha * FK_beta * log_FK;
    double x_z = std::log((std::sqrt(1.0 - 2.0 * p.rho * z + z * z) + z - p.rho) / (1.0 - p.rho));

    if (std::abs(x_z) < 1e-10) return p.alpha / FK_beta;

    double prefix = p.alpha / (FK_beta * (1.0 + (1.0 - p.beta) * (1.0 - p.beta) / 24.0 * log_FK * log_FK
                    + std::pow(1.0 - p.beta, 4) / 1920.0 * std::pow(log_FK, 4)));
    double A = (1.0 - p.beta) * (1.0 - p.beta) * p.alpha * p.alpha / (24.0 * FK_beta * FK_beta);
    double B = 0.25 * p.rho * p.beta * p.nu * p.alpha / FK_beta;
    double C = (2.0 - 3.0 * p.rho * p.rho) * p.nu * p.nu / 24.0;

    return prefix * z / x_z * (1.0 + (A + B + C) * T);
}

// Local vol from implied vol surface (Dupire formula)
// sigma_local^2 = (dw/dT + r*K*dw/dK) / (1 - y/w * dw/dy + 0.25*(-0.25 - 1/w + y^2/w^2)*(dw/dy)^2 + 0.5*d^2w/dy^2)
// where w = sigma^2 * T (total variance), y = ln(K/F)
double dupire_local_vol(double K, double T, double sigma, double dSigma_dK, double dSigma_dT,
                        double d2Sigma_dK2, double r) {
    double w = sigma * sigma * T;
    double dw_dT = 2.0 * sigma * dSigma_dT * T + sigma * sigma;
    double y = std::log(K); // simplified, should be ln(K/F)
    double dw_dy = 2.0 * sigma * dSigma_dK * K * T; // chain rule
    double d2w_dy2 = 2.0 * T * (dSigma_dK * dSigma_dK * K * K + sigma * d2Sigma_dK2 * K * K + sigma * dSigma_dK * K);

    double numerator = dw_dT;
    double denom = 1.0 - y / w * dw_dy + 0.25 * (-0.25 - 1.0 / w + y * y / (w * w)) * dw_dy * dw_dy + 0.5 * d2w_dy2;

    if (std::abs(denom) < 1e-15) return sigma;
    double local_var = numerator / denom;
    return (local_var > 0) ? std::sqrt(local_var) : sigma;
}

// Sticky strike vs sticky delta vol computation
double sticky_strike_vol(double K, double T, const VolSurface& surface) {
    return surface.interpolate(K, T);
}

double sticky_delta_vol(double S, double K, double T, double r, const VolSurface& surface) {
    // Transform to moneyness
    double F = S * std::exp(r * T);
    double moneyness = K / F;
    // Use moneyness as strike input (requires surface calibrated in moneyness space)
    return surface.interpolate(moneyness, T);
}

// Variance swap fair strike: integral of sigma^2 across strikes
double variance_swap_strike(const VolSurface& surface, double spot, double T, double r, int n_points) {
    double F = spot * std::exp(r * T);
    double disc = std::exp(-r * T);
    double sum = 0;
    double dk = 2.0 * F / n_points;

    for (int i = 1; i <= n_points; ++i) {
        double K = dk * i;
        double vol = surface.interpolate(K, T);
        double w = vol * vol * T;
        // Contribution: 2/T * (1 - ln(K/F)) * sigma^2 * T / K^2 * dK (simplified)
        double otm_price = 0;
        if (K < F) {
            otm_price = BSEngine::put_price(spot, K, T, r, vol);
        } else {
            otm_price = BSEngine::call_price(spot, K, T, r, vol);
        }
        sum += 2.0 * disc / (K * K) * otm_price * dk;
    }
    return sum / T;
}

// Vega-weighted implied vol for portfolio
double portfolio_weighted_iv(const std::vector<OptionPosition>& positions) {
    double vega_sum = 0;
    double weighted_iv = 0;
    for (const auto& pos : positions) {
        double v = BSEngine::vega(pos.underlying, pos.strike, pos.expiry, pos.rate, pos.iv) * 100.0;
        double abs_vega = std::abs(v * pos.quantity);
        weighted_iv += pos.iv * abs_vega;
        vega_sum += abs_vega;
    }
    return (vega_sum > 0) ? weighted_iv / vega_sum : 0;
}

// Pin risk: gamma near expiry at strike
double pin_risk_metric(double S, double K, double T, double sigma) {
    if (T > 5.0 / 252.0) return 0; // only relevant near expiry
    double gamma = BSEngine::gamma(S, K, T, 0, sigma);
    double proximity = std::exp(-0.5 * std::pow((S - K) / (S * sigma * std::sqrt(T)), 2));
    return gamma * proximity;
}

// Smile adjustment for barrier options (rule of thumb)
double barrier_vol_adjustment(double spot, double barrier, double strike, double T, double atm_vol, double skew_per_delta) {
    double moneyness_barrier = barrier / spot - 1.0;
    double moneyness_strike = strike / spot - 1.0;
    double avg_moneyness = 0.5 * (moneyness_barrier + moneyness_strike);
    return atm_vol + skew_per_delta * avg_moneyness;
}

} // namespace vol_utils

// ============================================================================
// OptionChainAnalyzer: analyze entire option chains
// ============================================================================

class OptionChainAnalyzer {
public:
    struct ChainPoint {
        double strike;
        double expiry;
        OptionType type;
        double bid;
        double ask;
        double mid;
        double iv;
        double delta;
        double gamma;
        double vega;
        double theta;
        double open_interest;
        double volume;
    };

    void clear() { chain_.clear(); }

    void add_option(double strike, double expiry, OptionType type,
                    double bid, double ask, double spot, double rate) {
        ChainPoint cp;
        cp.strike = strike;
        cp.expiry = expiry;
        cp.type = type;
        cp.bid = bid;
        cp.ask = ask;
        cp.mid = 0.5 * (bid + ask);
        cp.open_interest = 0;
        cp.volume = 0;

        IVSolver solver;
        cp.iv = solver.solve(cp.mid, spot, strike, expiry, rate, type);
        auto greeks = BSEngine::greeks(spot, strike, expiry, rate, cp.iv, type);
        cp.delta = greeks.delta;
        cp.gamma = greeks.gamma;
        cp.vega = greeks.vega;
        cp.theta = greeks.theta;

        chain_.push_back(cp);
    }

    // Put-call parity check
    double parity_violation(double strike, double expiry, double spot, double rate) const {
        const ChainPoint* call = nullptr;
        const ChainPoint* put = nullptr;
        for (const auto& cp : chain_) {
            if (std::abs(cp.strike - strike) < 0.01 && std::abs(cp.expiry - expiry) < 0.001) {
                if (cp.type == OptionType::Call) call = &cp;
                else put = &cp;
            }
        }
        if (!call || !put) return 0;
        double disc = std::exp(-rate * expiry);
        double parity = call->mid - put->mid - spot + strike * disc;
        return parity;
    }

    // Max pain: strike with minimum total option value
    double max_pain(double spot) const {
        std::vector<double> unique_strikes;
        for (const auto& cp : chain_) {
            bool found = false;
            for (auto s : unique_strikes) if (std::abs(s - cp.strike) < 0.01) { found = true; break; }
            if (!found) unique_strikes.push_back(cp.strike);
        }
        std::sort(unique_strikes.begin(), unique_strikes.end());

        double min_pain = std::numeric_limits<double>::max();
        double pain_strike = spot;

        for (double K : unique_strikes) {
            double total_pain = 0;
            for (const auto& cp : chain_) {
                double intrinsic = 0;
                if (cp.type == OptionType::Call) {
                    intrinsic = std::max(K - cp.strike, 0.0);
                } else {
                    intrinsic = std::max(cp.strike - K, 0.0);
                }
                total_pain += intrinsic * cp.open_interest;
            }
            if (total_pain < min_pain) {
                min_pain = total_pain;
                pain_strike = K;
            }
        }
        return pain_strike;
    }

    // PCR: put-call ratio by volume or open interest
    double put_call_ratio_volume() const {
        double put_vol = 0, call_vol = 0;
        for (const auto& cp : chain_) {
            if (cp.type == OptionType::Put) put_vol += cp.volume;
            else call_vol += cp.volume;
        }
        return (call_vol > 0) ? put_vol / call_vol : 0;
    }

    double put_call_ratio_oi() const {
        double put_oi = 0, call_oi = 0;
        for (const auto& cp : chain_) {
            if (cp.type == OptionType::Put) put_oi += cp.open_interest;
            else call_oi += cp.open_interest;
        }
        return (call_oi > 0) ? put_oi / call_oi : 0;
    }

    // Net gamma exposure at a given spot level
    double net_gamma_exposure(double spot) const {
        double total = 0;
        for (const auto& cp : chain_) {
            double g = BSEngine::gamma(spot, cp.strike, cp.expiry, 0, cp.iv);
            // Positive for long calls/puts (market makers are short, so negate for dealer gamma)
            total += g * cp.open_interest * 100.0; // 100 shares per contract
        }
        return total;
    }

    // Gamma exposure profile across spot range
    void gamma_profile(double spot_low, double spot_high, int n_points,
                       std::vector<double>& spots, std::vector<double>& gammas) const {
        spots.resize(n_points);
        gammas.resize(n_points);
        double step = (spot_high - spot_low) / (n_points - 1);
        for (int i = 0; i < n_points; ++i) {
            spots[i] = spot_low + i * step;
            gammas[i] = net_gamma_exposure(spots[i]);
        }
    }

    // Implied move from ATM straddle
    double implied_move(double spot, double expiry) const {
        // Find nearest ATM call and put
        const ChainPoint* atm_call = nullptr;
        const ChainPoint* atm_put = nullptr;
        double min_dist = std::numeric_limits<double>::max();

        for (const auto& cp : chain_) {
            if (std::abs(cp.expiry - expiry) > 0.01) continue;
            double dist = std::abs(cp.strike - spot);
            if (dist < min_dist) {
                min_dist = dist;
                if (cp.type == OptionType::Call) atm_call = &cp;
                else atm_put = &cp;
            }
        }

        double straddle = 0;
        if (atm_call) straddle += atm_call->mid;
        if (atm_put) straddle += atm_put->mid;
        return straddle / spot; // as fraction of spot
    }

    int size() const { return static_cast<int>(chain_.size()); }

private:
    std::vector<ChainPoint> chain_;
};

} // namespace srfm::options
