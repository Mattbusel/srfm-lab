///////////////////////////////////////////////////////////////////////////////
// portfolio_optimizer.cpp
// Portfolio Optimization Engine - Implementation
///////////////////////////////////////////////////////////////////////////////

#include "portfolio_optimizer.hpp"
#include <random>
#include <iostream>
#include <cstring>

namespace signal_engine {
namespace portfolio {

///////////////////////////////////////////////////////////////////////////////
// Covariance Estimation
///////////////////////////////////////////////////////////////////////////////

Mat sample_covariance(const Mat& returns) {
    size_t n = returns.size();
    size_t k = returns[0].size();

    // Compute means
    Vec mu(k, 0.0);
    for (size_t t = 0; t < n; ++t) {
        for (size_t j = 0; j < k; ++j) {
            mu[j] += returns[t][j];
        }
    }
    for (size_t j = 0; j < k; ++j) mu[j] /= n;

    // Covariance
    auto cov = zeros(k, k);
    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < k; ++i) {
            double di = returns[t][i] - mu[i];
            for (size_t j = i; j < k; ++j) {
                double dj = returns[t][j] - mu[j];
                cov[i][j] += di * dj;
            }
        }
    }
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = i; j < k; ++j) {
            cov[i][j] /= (n - 1);
            cov[j][i] = cov[i][j];
        }
    }
    return cov;
}

Mat ewma_covariance(const Mat& returns, double lambda) {
    size_t n = returns.size();
    size_t k = returns[0].size();

    Vec mu(k, 0.0);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < k; ++j)
            mu[j] += returns[t][j];
    for (size_t j = 0; j < k; ++j) mu[j] /= n;

    auto cov = zeros(k, k);
    double weight_sum = 0.0;

    for (size_t t = 0; t < n; ++t) {
        double w = std::pow(lambda, static_cast<double>(n - 1 - t));
        weight_sum += w;
        for (size_t i = 0; i < k; ++i) {
            double di = returns[t][i] - mu[i];
            for (size_t j = i; j < k; ++j) {
                double dj = returns[t][j] - mu[j];
                cov[i][j] += w * di * dj;
            }
        }
    }

    for (size_t i = 0; i < k; ++i) {
        for (size_t j = i; j < k; ++j) {
            cov[i][j] /= weight_sum;
            cov[j][i] = cov[i][j];
        }
    }
    return cov;
}

LedoitWolfResult ledoit_wolf_shrinkage(const Mat& returns) {
    size_t n = returns.size();
    size_t p = returns[0].size();

    // Sample covariance
    auto S = sample_covariance(returns);

    // Shrinkage target: scaled identity
    double trace_S = 0.0;
    for (size_t i = 0; i < p; ++i) trace_S += S[i][i];
    double mu_target = trace_S / p;

    auto F = zeros(p, p);
    for (size_t i = 0; i < p; ++i) F[i][i] = mu_target;

    // Compute means for centering
    Vec means(p, 0.0);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < p; ++j)
            means[j] += returns[t][j];
    for (size_t j = 0; j < p; ++j) means[j] /= n;

    // Compute shrinkage intensity (Ledoit-Wolf formula)
    // delta = ||S - F||^2 (squared Frobenius norm of S - target)
    double delta = 0.0;
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double diff = S[i][j] - F[i][j];
            delta += diff * diff;
        }
    }

    // beta_bar = (1/n^2) * sum_t || (x_t - mu)(x_t - mu)^T - S ||^2
    double beta_sum = 0.0;
    for (size_t t = 0; t < n; ++t) {
        double norm_sq = 0.0;
        for (size_t i = 0; i < p; ++i) {
            double di = returns[t][i] - means[i];
            for (size_t j = 0; j < p; ++j) {
                double dj = returns[t][j] - means[j];
                double diff = di * dj - S[i][j];
                norm_sq += diff * diff;
            }
        }
        beta_sum += norm_sq;
    }
    double beta_bar = beta_sum / (n * n);

    // Shrinkage intensity
    double alpha = std::min(beta_bar / delta, 1.0);
    alpha = std::max(alpha, 0.0);

    // Shrunk covariance
    auto shrunk = zeros(p, p);
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < p; ++j) {
            shrunk[i][j] = alpha * F[i][j] + (1.0 - alpha) * S[i][j];
        }
    }

    return {shrunk, alpha, S, F};
}

Mat factor_model_covariance(const Mat& returns, size_t n_factors) {
    size_t n = returns.size();
    size_t k = returns[0].size();

    // Simple PCA-based factor model
    auto S = sample_covariance(returns);

    // Power iteration for top eigenvectors (simplified)
    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Mat eigenvectors(n_factors, Vec(k));
    Vec eigenvalues(n_factors);

    auto S_deflated = S;

    for (size_t f = 0; f < n_factors; ++f) {
        Vec v(k);
        for (size_t j = 0; j < k; ++j) v[j] = norm_dist(rng);
        double nv = norm2(v);
        for (size_t j = 0; j < k; ++j) v[j] /= nv;

        // Power iteration
        for (int iter = 0; iter < 100; ++iter) {
            auto Sv = mat_vec(S_deflated, v);
            double ev = norm2(Sv);
            if (ev < 1e-15) break;
            for (size_t j = 0; j < k; ++j) v[j] = Sv[j] / ev;
            eigenvalues[f] = ev;
        }

        eigenvectors[f] = v;

        // Deflate
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j)
                S_deflated[i][j] -= eigenvalues[f] * v[i] * v[j];
    }

    // Factor covariance: B * Lambda * B^T + D
    // B = eigenvectors^T, Lambda = diag(eigenvalues)
    auto cov = zeros(k, k);

    // Systematic part
    for (size_t f = 0; f < n_factors; ++f) {
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                cov[i][j] += eigenvalues[f] *
                    eigenvectors[f][i] * eigenvectors[f][j];
            }
        }
    }

    // Specific (idiosyncratic) part
    for (size_t i = 0; i < k; ++i) {
        double specific = S[i][i] - cov[i][i];
        if (specific < 1e-10) specific = 1e-10;
        cov[i][i] += specific;
    }

    return cov;
}

///////////////////////////////////////////////////////////////////////////////
// Transaction Cost Model
///////////////////////////////////////////////////////////////////////////////

double TransactionCostModel::compute_cost(
    const Vec& old_w, const Vec& new_w, double pv) const {

    double cost = 0.0;
    size_t k = old_w.size();

    for (size_t i = 0; i < k; ++i) {
        double trade = std::abs(new_w[i] - old_w[i]) * pv;
        cost += proportional_cost * trade;
        if (trade > 1e-10) cost += fixed_cost;

        // Market impact (square root model)
        if (i < impact_coefficients.size()) {
            cost += impact_coefficients[i] * std::sqrt(trade);
        }

        // Tax cost for sells with gains
        if (new_w[i] < old_w[i] && i < cost_basis.size()) {
            double sell_amount = (old_w[i] - new_w[i]) * pv;
            double gain_per_dollar = std::max(0.0,
                old_w[i] * pv - cost_basis[i]) / (old_w[i] * pv + 1e-15);
            cost += tax_rate * gain_per_dollar * sell_amount;
        }
    }

    return cost;
}

///////////////////////////////////////////////////////////////////////////////
// PortfolioOptimizer - Constraint Management
///////////////////////////////////////////////////////////////////////////////

void PortfolioOptimizer::set_box_constraints(const Vec& lower, const Vec& upper) {
    box_constraint_ = std::make_unique<BoxConstraint>();
    box_constraint_->lower = lower;
    box_constraint_->upper = upper;
}

void PortfolioOptimizer::add_group_constraint(const GroupConstraint& gc) {
    group_constraints_.push_back(gc);
}

void PortfolioOptimizer::set_turnover_constraint(const TurnoverConstraint& tc) {
    turnover_constraint_ = std::make_unique<TurnoverConstraint>(tc);
}

void PortfolioOptimizer::set_factor_exposure_constraint(
    const FactorExposureConstraint& fec) {
    factor_constraint_ = std::make_unique<FactorExposureConstraint>(fec);
}

void PortfolioOptimizer::set_cardinality_constraint(
    const CardinalityConstraint& cc) {
    cardinality_constraint_ = std::make_unique<CardinalityConstraint>(cc);
}

void PortfolioOptimizer::clear_constraints() {
    box_constraint_.reset();
    group_constraints_.clear();
    turnover_constraint_.reset();
    factor_constraint_.reset();
    cardinality_constraint_.reset();
}

///////////////////////////////////////////////////////////////////////////////
// Simplex and Box Projection
///////////////////////////////////////////////////////////////////////////////

Vec PortfolioOptimizer::project_onto_simplex(const Vec& v, double z) {
    size_t n = v.size();
    Vec u = v;
    std::sort(u.begin(), u.end(), std::greater<double>());

    double cumsum = 0.0;
    double rho_val = 0.0;
    size_t rho = 0;
    for (size_t j = 0; j < n; ++j) {
        cumsum += u[j];
        double t = (cumsum - z) / (j + 1);
        if (u[j] - t > 0) {
            rho = j + 1;
            rho_val = t;
        }
    }

    double theta = (std::accumulate(u.begin(), u.begin() + rho, 0.0) - z) / rho;

    Vec w(n);
    for (size_t i = 0; i < n; ++i) {
        w[i] = std::max(v[i] - theta, 0.0);
    }
    return w;
}

Vec PortfolioOptimizer::project_onto_box(const Vec& v, const BoxConstraint& box) {
    Vec w(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        w[i] = std::max(box.lower[i], std::min(box.upper[i], v[i]));
    }
    return w;
}

Vec PortfolioOptimizer::enforce_cardinality(
    const Vec& w, size_t max_assets, double min_weight) {

    size_t n = w.size();
    if (max_assets >= n) return w;

    // Sort by absolute weight
    std::vector<std::pair<double, size_t>> abs_weights(n);
    for (size_t i = 0; i < n; ++i) {
        abs_weights[i] = {std::abs(w[i]), i};
    }
    std::sort(abs_weights.begin(), abs_weights.end(),
              std::greater<std::pair<double, size_t>>());

    Vec result(n, 0.0);
    double total = 0.0;
    for (size_t i = 0; i < max_assets; ++i) {
        size_t idx = abs_weights[i].second;
        result[idx] = std::max(w[idx], min_weight);
        total += result[idx];
    }

    // Renormalize
    if (total > 0) {
        for (size_t i = 0; i < n; ++i) result[i] /= total;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
// Active Set QP Solver
///////////////////////////////////////////////////////////////////////////////

Vec PortfolioOptimizer::solve_qp_active_set(
    const Mat& Q, const Vec& c,
    const Mat& A_eq, const Vec& b_eq,
    const Mat& A_ineq, const Vec& b_ineq,
    int max_iter) {

    size_t n = c.size();
    size_t m_eq = b_eq.size();
    size_t m_ineq = b_ineq.size();

    // Simple projected gradient descent for constrained QP
    // min 0.5 * x^T Q x + c^T x
    // s.t. A_eq x = b_eq, A_ineq x <= b_ineq

    Vec x(n, 1.0 / n);  // Start at equal weights

    double step_size = 0.01;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Gradient: Q*x + c
        Vec grad = vec_add(mat_vec(Q, x), c);

        // Gradient step
        Vec x_new = vec_sub(x, vec_scale(grad, step_size));

        // Project onto equality constraint (sum = 1 for portfolio)
        if (m_eq > 0) {
            double sum_x = vec_sum(x_new);
            double correction = (sum_x - b_eq[0]) / n;
            for (size_t i = 0; i < n; ++i) x_new[i] -= correction;
        }

        // Project onto inequality constraints (box)
        if (box_constraint_) {
            x_new = project_onto_box(x_new, *box_constraint_);
            // Re-normalize
            double sum_x = vec_sum(x_new);
            if (sum_x > 0) {
                for (size_t i = 0; i < n; ++i) x_new[i] /= sum_x;
            }
        }

        // Check convergence
        double diff = norm2(vec_sub(x_new, x));
        x = x_new;
        if (diff < 1e-10) break;

        // Adaptive step size
        step_size *= 0.999;
    }

    return x;
}

///////////////////////////////////////////////////////////////////////////////
// Mean-Variance Optimization
///////////////////////////////////////////////////////////////////////////////

OptimizationResult PortfolioOptimizer::mean_variance(
    const Vec& mu, const Mat& Sigma,
    double target_return, double risk_aversion,
    const BoxConstraint* box) {

    size_t n = mu.size();

    if (box) {
        box_constraint_ = std::make_unique<BoxConstraint>(*box);
    }

    Vec weights;

    if (!std::isnan(target_return)) {
        // Minimize variance s.t. w^T mu = target, sum(w) = 1
        // Use projected gradient descent

        weights.assign(n, 1.0 / n);
        double lr = 0.001;

        for (int iter = 0; iter < 5000; ++iter) {
            // Gradient of 0.5 * w^T Sigma w
            Vec grad = mat_vec(Sigma, weights);

            // Lagrangian for return constraint
            double current_ret = dot(weights, mu);
            double lambda_ret = (current_ret - target_return);

            for (size_t i = 0; i < n; ++i) {
                grad[i] += lambda_ret * mu[i] * 100.0;
            }

            // Step
            for (size_t i = 0; i < n; ++i) {
                weights[i] -= lr * grad[i];
            }

            // Project: sum = 1
            double s = vec_sum(weights);
            for (size_t i = 0; i < n; ++i) weights[i] /= s;

            // Box constraints
            if (box_constraint_) {
                weights = project_onto_box(weights, *box_constraint_);
                s = vec_sum(weights);
                if (s > 0) for (size_t i = 0; i < n; ++i) weights[i] /= s;
            }
        }
    } else if (!std::isnan(risk_aversion)) {
        // Maximize w^T mu - (gamma/2) * w^T Sigma w
        // Equivalent to minimizing (gamma/2) * w^T Sigma w - w^T mu

        Vec c(n);
        for (size_t i = 0; i < n; ++i) c[i] = -mu[i];

        Mat Q = Sigma;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                Q[i][j] *= risk_aversion;

        weights = solve_qp_active_set(Q, c, {{Vec(n, 1.0)}}, {1.0}, {}, {});
    } else {
        // No target specified: use default risk aversion of 2.5
        return mean_variance(mu, Sigma, std::numeric_limits<double>::quiet_NaN(),
                            2.5, box);
    }

    // Compute results
    double ret = dot(weights, mu);
    double vol = std::sqrt(
        dot(weights, mat_vec(Sigma, weights)));
    double sharpe = (vol > 1e-15) ? ret / vol : 0.0;

    // Risk contributions
    auto Sw = mat_vec(Sigma, weights);
    Vec marginal_risk(n);
    Vec risk_contrib(n);
    for (size_t i = 0; i < n; ++i) {
        marginal_risk[i] = Sw[i] / vol;
        risk_contrib[i] = weights[i] * marginal_risk[i];
    }

    return {
        weights, ret, vol, sharpe,
        risk_contrib, marginal_risk,
        5000, true, "mean_variance",
        0.5 * vol * vol
    };
}

///////////////////////////////////////////////////////////////////////////////
// Minimum Variance
///////////////////////////////////////////////////////////////////////////////

OptimizationResult PortfolioOptimizer::minimum_variance(
    const Mat& Sigma, const BoxConstraint* box) {

    size_t n = Sigma.size();
    Vec mu(n, 0.0);  // Zero expected returns
    return mean_variance(mu, Sigma,
                        std::numeric_limits<double>::quiet_NaN(),
                        1.0, box);  // gamma=1 with zero mu gives min variance
}

///////////////////////////////////////////////////////////////////////////////
// Maximum Sharpe Ratio
///////////////////////////////////////////////////////////////////////////////

OptimizationResult PortfolioOptimizer::max_sharpe(
    const Vec& mu, const Mat& Sigma,
    double risk_free_rate, const BoxConstraint* box) {

    size_t n = mu.size();

    // Reformulation: minimize w^T Sigma w s.t. (mu - rf)^T w = 1
    // Then normalize
    Vec excess_mu(n);
    for (size_t i = 0; i < n; ++i) {
        excess_mu[i] = mu[i] - risk_free_rate;
    }

    // Analytical solution (unconstrained): w* = Sigma^{-1} (mu - rf)
    Vec w_star = cholesky_solve(Sigma, excess_mu);

    // Normalize to sum to 1
    double w_sum = vec_sum(w_star);
    if (std::abs(w_sum) > 1e-15) {
        for (size_t i = 0; i < n; ++i) w_star[i] /= w_sum;
    }

    // Apply box constraints if present
    if (box) {
        w_star = project_onto_box(w_star, *box);
        double s = vec_sum(w_star);
        if (s > 0) for (size_t i = 0; i < n; ++i) w_star[i] /= s;

        // Refine with projected gradient
        for (int iter = 0; iter < 2000; ++iter) {
            double ret = dot(w_star, excess_mu);
            double vol = std::sqrt(dot(w_star, mat_vec(Sigma, w_star)));
            if (vol < 1e-15) break;

            // Gradient of -Sharpe = -(mu_p * Sigma w - vol^2 * mu) / (vol^3)
            Vec Sw = mat_vec(Sigma, w_star);
            Vec grad(n);
            for (size_t i = 0; i < n; ++i) {
                grad[i] = -(excess_mu[i] * vol - ret * Sw[i] / vol) / (vol * vol);
            }

            double lr = 0.0001;
            for (size_t i = 0; i < n; ++i) w_star[i] -= lr * grad[i];

            w_star = project_onto_box(w_star, *box);
            double ws = vec_sum(w_star);
            if (ws > 0) for (size_t i = 0; i < n; ++i) w_star[i] /= ws;
        }
    }

    double ret = dot(w_star, mu);
    double vol = std::sqrt(dot(w_star, mat_vec(Sigma, w_star)));
    double sharpe = (vol > 1e-15) ? (ret - risk_free_rate) / vol : 0.0;

    auto Sw = mat_vec(Sigma, w_star);
    Vec marg(n), rc(n);
    for (size_t i = 0; i < n; ++i) {
        marg[i] = Sw[i] / vol;
        rc[i] = w_star[i] * marg[i];
    }

    return {w_star, ret, vol, sharpe, rc, marg, 1, true, "max_sharpe", -sharpe};
}

///////////////////////////////////////////////////////////////////////////////
// Risk Parity (ERC)
///////////////////////////////////////////////////////////////////////////////

OptimizationResult PortfolioOptimizer::risk_parity(
    const Mat& Sigma, const Vec& risk_budget,
    double tol, int max_iter) {

    size_t n = Sigma.size();
    Vec budget = risk_budget;
    if (budget.empty()) {
        budget.assign(n, 1.0 / n);
    }

    Vec w(n, 1.0 / n);

    for (int iter = 0; iter < max_iter; ++iter) {
        Vec w_old = w;

        double port_var = dot(w, mat_vec(Sigma, w));
        double port_vol = std::sqrt(port_var);

        Vec Sw = mat_vec(Sigma, w);
        Vec mrc(n);
        for (size_t i = 0; i < n; ++i) {
            mrc[i] = Sw[i] / port_vol;
        }

        Vec rc(n);
        for (size_t i = 0; i < n; ++i) {
            rc[i] = w[i] * mrc[i];
        }

        Vec target_rc(n);
        for (size_t i = 0; i < n; ++i) {
            target_rc[i] = budget[i] * port_vol;
        }

        // Multiplicative update
        for (size_t i = 0; i < n; ++i) {
            if (rc[i] > 1e-15) {
                w[i] *= std::pow(target_rc[i] / rc[i], 0.5);
            }
        }

        // Normalize
        double s = vec_sum(w);
        for (size_t i = 0; i < n; ++i) w[i] /= s;

        // Check convergence
        double max_diff = 0.0;
        for (size_t i = 0; i < n; ++i) {
            max_diff = std::max(max_diff, std::abs(w[i] - w_old[i]));
        }

        if (max_diff < tol) {
            double vol = std::sqrt(dot(w, mat_vec(Sigma, w)));
            auto Sw_final = mat_vec(Sigma, w);
            Vec mrc_final(n), rc_final(n);
            for (size_t i = 0; i < n; ++i) {
                mrc_final[i] = Sw_final[i] / vol;
                rc_final[i] = w[i] * mrc_final[i];
            }

            return {w, 0.0, vol, 0.0, rc_final, mrc_final,
                    iter, true, "risk_parity", 0.0};
        }
    }

    double vol = std::sqrt(dot(w, mat_vec(Sigma, w)));
    auto Sw = mat_vec(Sigma, w);
    Vec mrc(n), rc(n);
    for (size_t i = 0; i < n; ++i) {
        mrc[i] = Sw[i] / vol;
        rc[i] = w[i] * mrc[i];
    }

    return {w, 0.0, vol, 0.0, rc, mrc, max_iter, false, "risk_parity", 0.0};
}

///////////////////////////////////////////////////////////////////////////////
// Black-Litterman
///////////////////////////////////////////////////////////////////////////////

OptimizationResult PortfolioOptimizer::black_litterman(
    const Vec& market_weights, const Mat& Sigma,
    const Mat& P, const Vec& Q, const Mat& Omega,
    double tau, double risk_aversion,
    const BoxConstraint* box) {

    size_t n = market_weights.size();
    size_t k = Q.size();  // Number of views

    // Implied equilibrium returns: Pi = gamma * Sigma * w_mkt
    Vec Pi = vec_scale(mat_vec(Sigma, market_weights), risk_aversion);

    // Scaled covariance
    auto tau_Sigma = Sigma;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            tau_Sigma[i][j] *= tau;

    // Posterior precision
    auto tau_Sigma_inv = cholesky_inverse(tau_Sigma);

    // P^T * Omega^{-1}
    auto Omega_inv = cholesky_inverse(Omega);
    auto Pt = transpose(P);
    auto Pt_Omega_inv = mat_mul(Pt, Omega_inv);

    // Posterior precision: tau_Sigma_inv + P^T Omega^{-1} P
    auto Pt_Omega_inv_P = mat_mul(Pt_Omega_inv, P);
    auto M = tau_Sigma_inv;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            M[i][j] += Pt_Omega_inv_P[i][j];

    // Posterior covariance
    auto M_inv = cholesky_inverse(M);

    // Posterior mean: M_inv * (tau_Sigma_inv * Pi + P^T Omega^{-1} Q)
    Vec term1 = mat_vec(tau_Sigma_inv, Pi);
    Vec term2 = mat_vec(Pt_Omega_inv, Q);
    Vec rhs = vec_add(term1, term2);
    Vec mu_bl = mat_vec(M_inv, rhs);

    // Posterior covariance (for optimization)
    // Sigma_bl = Sigma + M_inv
    auto Sigma_bl = Sigma;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            Sigma_bl[i][j] += M_inv[i][j];

    // Optimize with BL posterior
    return mean_variance(mu_bl, Sigma_bl,
                        std::numeric_limits<double>::quiet_NaN(),
                        risk_aversion, box);
}

///////////////////////////////////////////////////////////////////////////////
// Maximum Diversification
///////////////////////////////////////////////////////////////////////////////

OptimizationResult PortfolioOptimizer::max_diversification(
    const Vec& sigma, const Mat& Sigma,
    const BoxConstraint* box) {

    size_t n = sigma.size();

    // Maximize DR = (w^T sigma) / sqrt(w^T Sigma w)
    // Equivalent to minimizing w^T Sigma w s.t. w^T sigma = 1
    // Then normalize

    Vec w = cholesky_solve(Sigma, sigma);

    // Normalize to sum to 1
    double s = vec_sum(w);
    if (std::abs(s) > 1e-15) {
        for (size_t i = 0; i < n; ++i) w[i] /= s;
    }

    // Handle negative weights (short constraint)
    if (box) {
        w = project_onto_box(w, *box);
        s = vec_sum(w);
        if (s > 0) for (size_t i = 0; i < n; ++i) w[i] /= s;
    } else {
        // Long-only: project
        for (size_t i = 0; i < n; ++i) w[i] = std::max(0.0, w[i]);
        s = vec_sum(w);
        if (s > 0) for (size_t i = 0; i < n; ++i) w[i] /= s;
    }

    double port_vol = std::sqrt(dot(w, mat_vec(Sigma, w)));
    double weighted_vol = dot(w, sigma);
    double div_ratio = (port_vol > 1e-15) ? weighted_vol / port_vol : 0.0;

    auto Sw = mat_vec(Sigma, w);
    Vec mrc(n), rc(n);
    for (size_t i = 0; i < n; ++i) {
        mrc[i] = Sw[i] / port_vol;
        rc[i] = w[i] * mrc[i];
    }

    return {w, 0.0, port_vol, 0.0, rc, mrc, 1, true,
            "max_diversification", -div_ratio};
}

///////////////////////////////////////////////////////////////////////////////
// Robust Markowitz (Resampled Efficient Frontier)
///////////////////////////////////////////////////////////////////////////////

OptimizationResult PortfolioOptimizer::robust_markowitz(
    const Mat& returns, int n_resample, double risk_aversion,
    const BoxConstraint* box) {

    size_t n = returns.size();
    size_t k = returns[0].size();

    // Use shrinkage covariance
    auto lw = ledoit_wolf_shrinkage(returns);
    auto Sigma_shrunk = lw.shrunk_cov;

    // Sample mean
    Vec mu(k, 0.0);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < k; ++j)
            mu[j] += returns[t][j];
    for (size_t j = 0; j < k; ++j) mu[j] /= n;

    // Resampled efficient frontier
    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    auto L = cholesky(Sigma_shrunk);

    Vec avg_weights(k, 0.0);

    for (int s = 0; s < n_resample; ++s) {
        // Generate bootstrap sample
        Mat boot_returns(n, Vec(k));
        for (size_t t = 0; t < n; ++t) {
            Vec z(k);
            for (size_t j = 0; j < k; ++j) z[j] = norm_dist(rng);
            auto noise = mat_vec(L, z);
            for (size_t j = 0; j < k; ++j) {
                boot_returns[t][j] = mu[j] + noise[j];
            }
        }

        // Estimate from bootstrap sample
        Vec mu_boot(k, 0.0);
        for (size_t t = 0; t < n; ++t)
            for (size_t j = 0; j < k; ++j)
                mu_boot[j] += boot_returns[t][j];
        for (size_t j = 0; j < k; ++j) mu_boot[j] /= n;

        auto Sigma_boot = sample_covariance(boot_returns);

        // Optimize with bootstrap estimates
        auto result = mean_variance(mu_boot, Sigma_boot,
            std::numeric_limits<double>::quiet_NaN(),
            risk_aversion, box);

        for (size_t j = 0; j < k; ++j) {
            avg_weights[j] += result.weights[j];
        }
    }

    // Average
    for (size_t j = 0; j < k; ++j) avg_weights[j] /= n_resample;

    // Normalize
    double ws = vec_sum(avg_weights);
    if (ws > 0) for (size_t j = 0; j < k; ++j) avg_weights[j] /= ws;

    double ret = dot(avg_weights, mu);
    double vol = std::sqrt(dot(avg_weights, mat_vec(Sigma_shrunk, avg_weights)));
    double sharpe = (vol > 1e-15) ? ret / vol : 0.0;

    auto Sw = mat_vec(Sigma_shrunk, avg_weights);
    Vec mrc(k), rc(k);
    for (size_t i = 0; i < k; ++i) {
        mrc[i] = Sw[i] / vol;
        rc[i] = avg_weights[i] * mrc[i];
    }

    return {avg_weights, ret, vol, sharpe, rc, mrc,
            n_resample, true, "robust_markowitz", 0.0};
}

///////////////////////////////////////////////////////////////////////////////
// Rebalancing
///////////////////////////////////////////////////////////////////////////////

PortfolioOptimizer::RebalanceResult PortfolioOptimizer::compute_rebalance(
    const Vec& current_weights, const Vec& target_weights,
    double portfolio_value, const TransactionCostModel& cost_model,
    double threshold) {

    size_t n = current_weights.size();

    // Check if rebalance is needed
    double max_deviation = 0.0;
    for (size_t i = 0; i < n; ++i) {
        max_deviation = std::max(max_deviation,
            std::abs(current_weights[i] - target_weights[i]));
    }

    bool trigger = max_deviation > threshold;

    if (!trigger) {
        return {current_weights, Vec(n, 0.0), 0.0, 0.0, false};
    }

    // Compute trades
    Vec trades(n);
    double turnover = 0.0;
    for (size_t i = 0; i < n; ++i) {
        trades[i] = target_weights[i] - current_weights[i];
        turnover += std::abs(trades[i]);
    }

    // Transaction cost
    double cost = cost_model.compute_cost(
        current_weights, target_weights, portfolio_value);

    return {target_weights, trades, turnover, cost, true};
}

Vec PortfolioOptimizer::threshold_rebalance(
    const Vec& current, const Vec& target, double threshold) {

    size_t n = current.size();
    Vec result = current;

    bool need_rebalance = false;
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(current[i] - target[i]) > threshold) {
            need_rebalance = true;
            break;
        }
    }

    if (need_rebalance) {
        // Partial rebalance: only adjust assets beyond threshold
        for (size_t i = 0; i < n; ++i) {
            if (std::abs(current[i] - target[i]) > threshold) {
                result[i] = target[i];
            }
        }
        // Normalize
        double s = vec_sum(result);
        if (s > 0) for (size_t i = 0; i < n; ++i) result[i] /= s;
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
// Risk Decomposition
///////////////////////////////////////////////////////////////////////////////

PortfolioOptimizer::RiskDecomposition PortfolioOptimizer::decompose_risk(
    const Vec& weights, const Mat& Sigma, double alpha) {

    size_t n = weights.size();

    double port_var = dot(weights, mat_vec(Sigma, weights));
    double port_vol = std::sqrt(port_var);

    Vec Sw = mat_vec(Sigma, weights);
    Vec marginal(n), component(n), pct(n);

    double sum_component = 0.0;
    Vec asset_vol(n);

    for (size_t i = 0; i < n; ++i) {
        marginal[i] = Sw[i] / port_vol;
        component[i] = weights[i] * marginal[i];
        asset_vol[i] = std::sqrt(Sigma[i][i]);
        sum_component += component[i];
    }

    for (size_t i = 0; i < n; ++i) {
        pct[i] = (sum_component > 1e-15) ? component[i] / sum_component * 100.0 : 0.0;
    }

    // Component VaR (parametric)
    double z = 2.326; // 99% quantile approximation
    if (alpha < 0.99) {
        // Simple normal quantile approximation
        // For common values:
        if (std::abs(alpha - 0.95) < 0.001) z = 1.645;
        else if (std::abs(alpha - 0.975) < 0.001) z = 1.960;
        else if (std::abs(alpha - 0.99) < 0.001) z = 2.326;
        else z = 1.645; // default to 95%
    }

    Vec comp_var(n);
    for (size_t i = 0; i < n; ++i) {
        comp_var[i] = weights[i] * z * Sw[i] / port_vol;
    }

    // Diversification ratio
    double weighted_vol = 0.0;
    for (size_t i = 0; i < n; ++i) {
        weighted_vol += std::abs(weights[i]) * asset_vol[i];
    }
    double div_ratio = (port_vol > 1e-15) ? weighted_vol / port_vol : 1.0;

    return {port_vol, marginal, component, pct, comp_var, div_ratio};
}

///////////////////////////////////////////////////////////////////////////////
// Efficient Frontier
///////////////////////////////////////////////////////////////////////////////

std::vector<PortfolioOptimizer::FrontierPoint>
PortfolioOptimizer::efficient_frontier(
    const Vec& mu, const Mat& Sigma, size_t n_points,
    const BoxConstraint* box) {

    size_t n = mu.size();

    // Find min and max feasible returns
    double min_ret = *std::min_element(mu.begin(), mu.end());
    double max_ret = *std::max_element(mu.begin(), mu.end());

    // Add margin
    double margin = (max_ret - min_ret) * 0.05;
    min_ret += margin;
    max_ret -= margin;

    std::vector<FrontierPoint> frontier;
    frontier.reserve(n_points);

    for (size_t i = 0; i < n_points; ++i) {
        double target = min_ret + (max_ret - min_ret) * i / (n_points - 1);

        auto result = mean_variance(mu, Sigma, target,
            std::numeric_limits<double>::quiet_NaN(), box);

        frontier.push_back({
            result.expected_return,
            result.volatility,
            result.sharpe_ratio,
            result.weights
        });
    }

    return frontier;
}

} // namespace portfolio
} // namespace signal_engine
