#pragma once
// Numerical optimization methods for portfolio and parameter estimation.
// Gradient descent, L-BFGS, Nelder-Mead simplex, SGD with momentum.

#include "matrix.hpp"
#include <functional>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <deque>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace linalg {
namespace optim {

using ObjectiveFn  = std::function<double(const std::vector<double>&)>;
using GradientFn   = std::function<std::vector<double>(const std::vector<double>&)>;

struct OptimResult {
    std::vector<double> x;          // optimal point
    double              fval;        // objective value
    int                 iterations;
    bool                converged;
    std::string         message;
    std::vector<double> loss_history;
};

// ============================================================
// Numerical gradient via finite differences
// ============================================================
std::vector<double> numerical_gradient(const ObjectiveFn& f,
                                        const std::vector<double>& x,
                                        double h = 1e-6)
{
    std::vector<double> grad(x.size());
    std::vector<double> xp = x, xm = x;
    for (size_t i = 0; i < x.size(); ++i) {
        xp[i] = x[i] + h; xm[i] = x[i] - h;
        grad[i] = (f(xp) - f(xm)) / (2.0 * h);
        xp[i] = x[i]; xm[i] = x[i];
    }
    return grad;
}

// ============================================================
// Gradient Descent with line search (Armijo backtracking)
// ============================================================
struct GDConfig {
    double lr_init   = 0.01;
    double armijo_c  = 1e-4;
    double armijo_rho = 0.5;
    int    max_iter  = 1000;
    double tol       = 1e-7;
    bool   verbose   = false;
};

OptimResult gradient_descent(const ObjectiveFn& f,
                              const GradientFn& grad_f,
                              std::vector<double> x0,
                              const GDConfig& cfg = {})
{
    OptimResult res;
    res.x = x0;
    res.fval = f(x0);
    res.converged = false;

    for (res.iterations = 0; res.iterations < cfg.max_iter; ++res.iterations) {
        auto g = grad_f(res.x);
        double g_norm = 0.0;
        for (auto v : g) g_norm += v * v;
        g_norm = std::sqrt(g_norm);
        if (g_norm < cfg.tol) { res.converged = true; break; }

        // Armijo backtracking line search
        double alpha = cfg.lr_init;
        double f0    = res.fval;
        for (int ls = 0; ls < 30; ++ls) {
            std::vector<double> x_new(res.x.size());
            for (size_t i = 0; i < res.x.size(); ++i)
                x_new[i] = res.x[i] - alpha * g[i];
            double f_new = f(x_new);
            if (f_new <= f0 - cfg.armijo_c * alpha * g_norm * g_norm) {
                res.x    = x_new;
                res.fval = f_new;
                break;
            }
            alpha *= cfg.armijo_rho;
        }
        res.loss_history.push_back(res.fval);
    }
    return res;
}

// ============================================================
// Adam optimizer (SGD with adaptive moments)
// ============================================================
struct AdamConfig {
    double lr      = 1e-3;
    double beta1   = 0.9;
    double beta2   = 0.999;
    double eps     = 1e-8;
    int    max_iter = 10000;
    double tol      = 1e-7;
    bool   verbose  = false;
};

class AdamOptimizer {
public:
    explicit AdamOptimizer(size_t dim, const AdamConfig& cfg = {})
        : cfg_(cfg), m_(dim, 0.0), v_(dim, 0.0), t_(0) {}

    // One step: update x in-place, return new fval
    double step(std::vector<double>& x, const std::vector<double>& grad) {
        ++t_;
        double bc1 = 1.0 - std::pow(cfg_.beta1, t_);
        double bc2 = 1.0 - std::pow(cfg_.beta2, t_);
        for (size_t i = 0; i < x.size(); ++i) {
            m_[i] = cfg_.beta1 * m_[i] + (1.0 - cfg_.beta1) * grad[i];
            v_[i] = cfg_.beta2 * v_[i] + (1.0 - cfg_.beta2) * grad[i] * grad[i];
            double m_hat = m_[i] / bc1;
            double v_hat = v_[i] / bc2;
            x[i] -= cfg_.lr * m_hat / (std::sqrt(v_hat) + cfg_.eps);
        }
        return 0.0;
    }

    void reset() { std::fill(m_.begin(), m_.end(), 0.0); std::fill(v_.begin(), v_.end(), 0.0); t_ = 0; }

private:
    AdamConfig         cfg_;
    std::vector<double> m_, v_;
    int                t_;
};

OptimResult adam_optimize(const ObjectiveFn& f,
                           std::vector<double> x0,
                           const AdamConfig& cfg = {})
{
    OptimResult res;
    res.x = x0;
    res.fval = f(x0);
    AdamOptimizer adam(x0.size(), cfg);

    for (res.iterations = 0; res.iterations < cfg.max_iter; ++res.iterations) {
        auto g = numerical_gradient(f, res.x);
        adam.step(res.x, g);
        double new_f = f(res.x);
        res.loss_history.push_back(new_f);

        if (std::fabs(new_f - res.fval) < cfg.tol) {
            res.converged = true;
            res.fval = new_f;
            break;
        }
        res.fval = new_f;
    }
    return res;
}

// ============================================================
// L-BFGS (Limited-memory BFGS)
// ============================================================
struct LBFGSConfig {
    int    m        = 10;     // history size
    double ftol     = 1e-7;
    double gtol     = 1e-5;
    int    max_iter = 500;
    bool   verbose  = false;
};

OptimResult lbfgs(const ObjectiveFn& f,
                   const GradientFn& grad_f,
                   std::vector<double> x0,
                   const LBFGSConfig& cfg = {})
{
    OptimResult res;
    res.x  = x0;
    res.fval = f(x0);
    const size_t n = x0.size();

    std::deque<std::vector<double>> S, Y;
    std::deque<double> rho;

    auto g = grad_f(res.x);

    for (res.iterations = 0; res.iterations < cfg.max_iter; ++res.iterations) {
        double g_norm = 0.0;
        for (auto v : g) g_norm += v * v;
        if (std::sqrt(g_norm) < cfg.gtol) { res.converged = true; break; }

        // Two-loop recursion for Hessian-vector product
        std::vector<double> q = g;
        std::vector<double> alpha_vec(S.size());

        for (int i = static_cast<int>(S.size())-1; i >= 0; --i) {
            double si_q = 0.0;
            for (size_t j = 0; j < n; ++j) si_q += S[i][j] * q[j];
            alpha_vec[i] = rho[i] * si_q;
            for (size_t j = 0; j < n; ++j) q[j] -= alpha_vec[i] * Y[i][j];
        }

        // Initial Hessian approximation: gamma * I
        double gamma = 1.0;
        if (!S.empty()) {
            double sy = 0.0, yy = 0.0;
            for (size_t j = 0; j < n; ++j) { sy += S.back()[j] * Y.back()[j]; yy += Y.back()[j] * Y.back()[j]; }
            gamma = yy > 0 ? sy / yy : 1.0;
        }

        std::vector<double> r(n);
        for (size_t j = 0; j < n; ++j) r[j] = gamma * q[j];

        for (size_t i = 0; i < S.size(); ++i) {
            double yi_r = 0.0;
            for (size_t j = 0; j < n; ++j) yi_r += Y[i][j] * r[j];
            double beta = rho[i] * yi_r;
            for (size_t j = 0; j < n; ++j) r[j] += S[i][j] * (alpha_vec[i] - beta);
        }

        // r is search direction; negate for descent
        std::vector<double> d(n);
        for (size_t j = 0; j < n; ++j) d[j] = -r[j];

        // Wolfe condition line search
        double alpha_step = 1.0;
        double f0 = res.fval;
        double dg0 = 0.0;
        for (size_t j = 0; j < n; ++j) dg0 += d[j] * g[j];
        if (dg0 > 0) { d = g; for (auto& v : d) v = -v; dg0 = -g_norm*g_norm; }

        std::vector<double> x_new(n);
        for (int ls = 0; ls < 20; ++ls) {
            for (size_t j = 0; j < n; ++j) x_new[j] = res.x[j] + alpha_step * d[j];
            double f_new = f(x_new);
            if (f_new <= f0 + 1e-4 * alpha_step * dg0) break;
            alpha_step *= 0.5;
        }

        std::vector<double> s(n), y(n);
        auto g_new = grad_f(x_new);
        for (size_t j = 0; j < n; ++j) { s[j] = x_new[j] - res.x[j]; y[j] = g_new[j] - g[j]; }

        double sy = 0.0;
        for (size_t j = 0; j < n; ++j) sy += s[j] * y[j];
        if (sy > 0) {
            if (static_cast<int>(S.size()) == cfg.m) { S.pop_front(); Y.pop_front(); rho.pop_front(); }
            S.push_back(s); Y.push_back(y); rho.push_back(1.0 / sy);
        }

        res.x   = x_new;
        res.fval = f(x_new);
        g        = g_new;
        res.loss_history.push_back(res.fval);
    }
    return res;
}

// ============================================================
// Nelder-Mead Simplex (derivative-free)
// ============================================================
struct NelderMeadConfig {
    double alpha = 1.0;   // reflection
    double gamma = 2.0;   // expansion
    double rho   = 0.5;   // contraction
    double sigma = 0.5;   // shrink
    int    max_iter = 5000;
    double tol      = 1e-8;
    bool   verbose  = false;
};

OptimResult nelder_mead(const ObjectiveFn& f,
                         std::vector<double> x0,
                         const NelderMeadConfig& cfg = {})
{
    const size_t n = x0.size();
    const size_t m = n + 1;

    // Initialize simplex
    std::vector<std::vector<double>> simplex(m, x0);
    for (size_t i = 1; i < m; ++i) {
        double delta = (std::fabs(x0[i-1]) > 1e-8) ? 0.05 * x0[i-1] : 0.00025;
        simplex[i][i-1] += delta;
    }
    std::vector<double> fvals(m);
    for (size_t i = 0; i < m; ++i) fvals[i] = f(simplex[i]);

    OptimResult res;
    res.converged = false;

    auto centroid = [&](size_t skip) {
        std::vector<double> c(n, 0.0);
        for (size_t i = 0; i < m; ++i) {
            if (i == skip) continue;
            for (size_t j = 0; j < n; ++j) c[j] += simplex[i][j];
        }
        for (auto& v : c) v /= (m - 1);
        return c;
    };

    for (res.iterations = 0; res.iterations < cfg.max_iter; ++res.iterations) {
        // Sort by function value
        std::vector<size_t> idx(m);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return fvals[a] < fvals[b]; });

        // Check convergence
        double range = fvals[idx.back()] - fvals[idx[0]];
        if (range < cfg.tol) { res.converged = true; break; }

        size_t best  = idx[0];
        size_t worst = idx.back();
        size_t second_worst = idx[m-2];

        auto c = centroid(worst);

        // Reflection
        std::vector<double> xr(n);
        for (size_t j = 0; j < n; ++j) xr[j] = c[j] + cfg.alpha * (c[j] - simplex[worst][j]);
        double fr = f(xr);

        if (fr < fvals[best]) {
            // Expansion
            std::vector<double> xe(n);
            for (size_t j = 0; j < n; ++j) xe[j] = c[j] + cfg.gamma * (xr[j] - c[j]);
            double fe = f(xe);
            if (fe < fr) { simplex[worst] = xe; fvals[worst] = fe; }
            else         { simplex[worst] = xr; fvals[worst] = fr; }
        } else if (fr < fvals[second_worst]) {
            simplex[worst] = xr; fvals[worst] = fr;
        } else {
            // Contraction
            std::vector<double> xc(n);
            if (fr < fvals[worst]) {
                for (size_t j = 0; j < n; ++j) xc[j] = c[j] + cfg.rho * (xr[j] - c[j]);
                double fc = f(xc);
                if (fc < fr) { simplex[worst] = xc; fvals[worst] = fc; }
                else goto shrink;
            } else {
                for (size_t j = 0; j < n; ++j) xc[j] = c[j] + cfg.rho * (simplex[worst][j] - c[j]);
                double fc = f(xc);
                if (fc < fvals[worst]) { simplex[worst] = xc; fvals[worst] = fc; }
                else goto shrink;
            }
            goto next_iter;
            shrink:
            for (size_t i = 0; i < m; ++i) {
                if (i == best) continue;
                for (size_t j = 0; j < n; ++j)
                    simplex[i][j] = simplex[best][j] + cfg.sigma * (simplex[i][j] - simplex[best][j]);
                fvals[i] = f(simplex[i]);
            }
        }
        next_iter:
        res.loss_history.push_back(fvals[idx[0]]);
    }

    std::vector<size_t> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return fvals[a] < fvals[b]; });
    res.x    = simplex[idx[0]];
    res.fval = fvals[idx[0]];
    return res;
}

// ============================================================
// Constrained optimization: projected gradient for box constraints
// ============================================================
std::vector<double> project_box(const std::vector<double>& x,
                                  const std::vector<double>& lb,
                                  const std::vector<double>& ub)
{
    std::vector<double> xp(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        xp[i] = std::max(lb[i], std::min(ub[i], x[i]));
    return xp;
}

OptimResult projected_gradient(const ObjectiveFn& f,
                                 std::vector<double> x0,
                                 const std::vector<double>& lb,
                                 const std::vector<double>& ub,
                                 double lr = 0.01, int max_iter = 1000, double tol = 1e-7)
{
    OptimResult res;
    res.x = project_box(x0, lb, ub);
    res.fval = f(res.x);

    for (res.iterations = 0; res.iterations < max_iter; ++res.iterations) {
        auto g = numerical_gradient(f, res.x);
        double g_norm = 0.0;
        for (auto v : g) g_norm += v*v;
        if (std::sqrt(g_norm) < tol) { res.converged = true; break; }

        std::vector<double> x_new(res.x.size());
        for (size_t i = 0; i < res.x.size(); ++i)
            x_new[i] = res.x[i] - lr * g[i];
        x_new = project_box(x_new, lb, ub);

        double f_new = f(x_new);
        res.loss_history.push_back(f_new);
        if (std::fabs(f_new - res.fval) < tol) { res.converged = true; res.fval = f_new; res.x = x_new; break; }
        res.fval = f_new;
        res.x    = x_new;
    }
    return res;
}

// ============================================================
// Minimum variance portfolio via constrained optimization
// Minimize w^T Sigma w  subject to sum(w) = 1, w >= 0
// ============================================================
std::vector<double> min_var_constrained(const MatrixD& Sigma,
                                          int max_iter = 2000, double tol = 1e-8)
{
    const size_t n = Sigma.rows();
    auto f = [&](const std::vector<double>& w) {
        double var = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                var += w[i] * Sigma(i,j) * w[j];
        // Penalty for sum(w) != 1
        double sum_w = 0.0;
        for (auto v : w) sum_w += v;
        var += 1000.0 * (sum_w - 1.0) * (sum_w - 1.0);
        return var;
    };

    std::vector<double> lb(n, 0.0), ub(n, 1.0);
    std::vector<double> x0(n, 1.0/n);
    return projected_gradient(f, x0, lb, ub, 0.01, max_iter, tol).x;
}

} // namespace optim
} // namespace linalg
