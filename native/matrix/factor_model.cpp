// Multi-factor model implementation: Fama-French 3-factor, PCA-based factor extraction,
// factor exposure estimation, risk attribution, and factor return forecasting.

#include "matrix.hpp"
#include "statistics.hpp"
#include "time_series.hpp"
#include "decomposition.cpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>

// ============================================================
// Factor definitions
// ============================================================
enum class FactorType {
    Market,      // Market excess return
    SMB,         // Small Minus Big (size)
    HML,         // High Minus Low (value)
    Mom,         // Momentum
    QMJ,         // Quality Minus Junk
    BAB,         // Betting Against Beta
    Custom,
};

struct Factor {
    std::string name;
    FactorType  type;
    std::vector<double> returns;  // time series of factor returns
    double      mean_return = 0.0;
    double      vol         = 0.0;
    double      sharpe      = 0.0;

    void compute_stats(double periods_per_year = 252.0) {
        if (returns.empty()) return;
        WelfordState ws;
        for (auto r : returns) ws.update(r);
        mean_return = ws.mean;
        vol         = ws.std_dev() * std::sqrt(periods_per_year);
        sharpe      = vol > 0 ? ws.mean / ws.std_dev() * std::sqrt(periods_per_year) : 0.0;
    }
};

// ============================================================
// Factor Exposure (Beta) Estimation
// Uses rolling OLS with optional Newey-West standard errors
// ============================================================
struct FactorExposure {
    std::string asset;
    std::vector<double> betas;       // one per factor
    std::vector<double> beta_se;     // standard errors
    std::vector<double> t_stats;
    double              alpha;       // annualized alpha
    double              r_squared;
    double              residual_vol;
    double              idio_sharpe;
};

struct OLSResult {
    std::vector<double> coef;
    std::vector<double> se;
    std::vector<double> tstat;
    double              r2;
    double              adj_r2;
    double              f_stat;
    double              residual_std;
};

// Ordinary least squares: y = X * beta + eps
OLSResult ols(const MatrixD& X, const std::vector<double>& y) {
    size_t T = X.rows(), K = X.cols();
    assert(y.size() == T);
    OLSResult res;

    // Add intercept column
    MatrixD Xa(T, K + 1);
    for (size_t t = 0; t < T; ++t) {
        Xa(t, 0) = 1.0;
        for (size_t k = 0; k < K; ++k) Xa(t, k + 1) = X(t, k);
    }
    size_t K1 = K + 1;

    // Normal equations: (X'X)^{-1} X'y
    MatrixD XtX(K1, K1);
    for (size_t i = 0; i < K1; ++i)
        for (size_t j = 0; j < K1; ++j)
            for (size_t t = 0; t < T; ++t)
                XtX(i, j) += Xa(t, i) * Xa(t, j);

    std::vector<double> Xty(K1, 0.0);
    for (size_t k = 0; k < K1; ++k)
        for (size_t t = 0; t < T; ++t)
            Xty[k] += Xa(t, k) * y[t];

    // Solve via LU
    auto perm = lu_inplace(XtX);
    auto beta = lu_solve(XtX, perm, Xty);
    res.coef  = beta;

    // Residuals
    std::vector<double> resid(T);
    double ss_res = 0, ss_tot = 0;
    double y_mean = 0;
    for (auto v : y) y_mean += v;
    y_mean /= T;

    for (size_t t = 0; t < T; ++t) {
        double yhat = 0;
        for (size_t k = 0; k < K1; ++k) yhat += Xa(t, k) * beta[k];
        resid[t] = y[t] - yhat;
        ss_res  += resid[t] * resid[t];
        double d = y[t] - y_mean;
        ss_tot  += d * d;
    }
    res.residual_std = std::sqrt(ss_res / (T - K1));
    res.r2     = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0;
    res.adj_r2 = 1.0 - (1.0 - res.r2) * (T - 1.0) / (T - K1 - 0.0);
    res.f_stat = (res.r2 / K) / ((1.0 - res.r2) / (T - K1));

    // Standard errors (HC0)
    MatrixD XtXinv = inverse(XtX);
    res.se.resize(K1);
    res.tstat.resize(K1);
    for (size_t k = 0; k < K1; ++k) {
        res.se[k]    = std::sqrt(XtXinv(k, k)) * res.residual_std;
        res.tstat[k] = res.se[k] > 0 ? beta[k] / res.se[k] : 0.0;
    }
    return res;
}

// Newey-West HAC standard errors (lags = 4)
std::vector<double> newey_west_se(const MatrixD& X, const std::vector<double>& resid,
                                    const MatrixD& XtXinv, int lags = 4)
{
    size_t T = X.rows(), K = X.cols();
    MatrixD S(K, K);  // HAC covariance

    // Lag 0
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < K; ++i)
            for (size_t j = 0; j < K; ++j)
                S(i, j) += resid[t] * resid[t] * X(t, i) * X(t, j);

    // Lags 1..L with Bartlett kernel
    for (int l = 1; l <= lags; ++l) {
        double w = 1.0 - l / (lags + 1.0);
        for (size_t t = l; t < T; ++t)
            for (size_t i = 0; i < K; ++i)
                for (size_t j = 0; j < K; ++j) {
                    double contrib = w * resid[t] * resid[t-l] * X(t, i) * X(t-l, j);
                    S(i, j) += contrib;
                    S(j, i) += contrib;
                }
    }

    // V = (XtX)^{-1} * S * (XtX)^{-1}
    MatrixD V(K, K);
    for (size_t i = 0; i < K; ++i)
        for (size_t j = 0; j < K; ++j)
            for (size_t k = 0; k < K; ++k)
                for (size_t l2 = 0; l2 < K; ++l2)
                    V(i, j) += XtXinv(i, k) * S(k, l2) * XtXinv(l2, j);

    std::vector<double> se(K);
    for (size_t k = 0; k < K; ++k) se[k] = std::sqrt(std::max(0.0, V(k, k)));
    return se;
}

// ============================================================
// Factor model: fit betas for multiple assets
// ============================================================
struct FactorModel {
    std::vector<Factor>         factors;
    std::vector<FactorExposure> exposures;
    std::string                 model_name;
    size_t                      n_assets = 0;
    size_t                      n_factors = 0;
    double                      periods_per_year = 252.0;

    // Fit the model: returns = alpha + sum(beta_k * F_k) + eps
    void fit(const std::vector<std::string>& asset_names,
             const MatrixD& asset_returns,   // T x N
             bool use_newey_west = true)
    {
        size_t T = asset_returns.rows();
        size_t N = asset_returns.cols();
        assert(N == asset_names.size());
        assert(!factors.empty());
        assert(factors[0].returns.size() == T);

        n_assets  = N;
        n_factors = factors.size();

        // Build factor matrix F: T x n_factors
        MatrixD F(T, n_factors);
        for (size_t t = 0; t < T; ++t)
            for (size_t k = 0; k < n_factors; ++k)
                F(t, k) = factors[k].returns[t];

        exposures.resize(N);
        for (size_t i = 0; i < N; ++i) {
            std::vector<double> y(T);
            for (size_t t = 0; t < T; ++t) y[t] = asset_returns(t, i);

            auto ols_res = ols(F, y);

            FactorExposure& exp = exposures[i];
            exp.asset = asset_names[i];
            exp.betas.assign(ols_res.coef.begin() + 1, ols_res.coef.end());
            exp.alpha     = ols_res.coef[0] * periods_per_year;
            exp.r_squared = ols_res.r2;

            // Residual vol
            WelfordState ws;
            std::vector<double> resid(T);
            for (size_t t = 0; t < T; ++t) {
                double yhat = ols_res.coef[0];
                for (size_t k = 0; k < n_factors; ++k)
                    yhat += ols_res.coef[k + 1] * F(t, k);
                resid[t] = y[t] - yhat;
                ws.update(resid[t]);
            }
            exp.residual_vol = ws.std_dev() * std::sqrt(periods_per_year);

            // SE and t-stats
            if (use_newey_west) {
                // Build augmented X with intercept
                MatrixD Fa(T, n_factors + 1);
                for (size_t t = 0; t < T; ++t) {
                    Fa(t, 0) = 1.0;
                    for (size_t k = 0; k < n_factors; ++k) Fa(t, k+1) = F(t, k);
                }
                MatrixD FaFa(n_factors+1, n_factors+1);
                for (size_t a = 0; a <= n_factors; ++a)
                    for (size_t b = 0; b <= n_factors; ++b)
                        for (size_t t = 0; t < T; ++t)
                            FaFa(a, b) += Fa(t, a) * Fa(t, b);
                auto FaFainv = inverse(FaFa);
                auto nw_se = newey_west_se(Fa, resid, FaFainv);
                exp.beta_se = std::vector<double>(nw_se.begin() + 1, nw_se.end());
                exp.t_stats.resize(n_factors);
                for (size_t k = 0; k < n_factors; ++k)
                    exp.t_stats[k] = exp.beta_se[k] > 0
                        ? exp.betas[k] / exp.beta_se[k] : 0.0;
            } else {
                exp.beta_se = std::vector<double>(ols_res.se.begin()+1, ols_res.se.end());
                exp.t_stats = std::vector<double>(ols_res.tstat.begin()+1, ols_res.tstat.end());
            }

            exp.idio_sharpe = exp.residual_vol > 0 ?
                exp.alpha / exp.residual_vol : 0.0;
        }
    }

    // Risk attribution: decompose total variance into factor + idiosyncratic
    struct RiskAttrib {
        double total_var;
        double factor_var;
        double idio_var;
        double factor_pct;
        std::vector<double> factor_contributions;  // per factor
    };

    RiskAttrib risk_attribution(size_t asset_idx, const MatrixD& factor_cov) const {
        assert(asset_idx < n_assets);
        const auto& exp = exposures[asset_idx];
        RiskAttrib r{};

        // Factor variance: beta' * F_cov * beta
        size_t K = n_factors;
        assert(factor_cov.rows() == K && factor_cov.cols() == K);

        double factor_var = 0;
        std::vector<double> contrib(K, 0.0);
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < K; ++j)
                factor_var += exp.betas[i] * factor_cov(i, j) * exp.betas[j];
        }
        // Per-factor marginal contribution (diagonal approximation)
        for (size_t k = 0; k < K; ++k)
            contrib[k] = exp.betas[k] * exp.betas[k] * factor_cov(k, k);

        double idio_var = std::pow(exp.residual_vol / std::sqrt(periods_per_year), 2);
        double total_var = factor_var + idio_var;

        r.total_var             = total_var;
        r.factor_var            = factor_var;
        r.idio_var              = idio_var;
        r.factor_pct            = total_var > 0 ? factor_var / total_var : 0.0;
        r.factor_contributions  = contrib;
        return r;
    }

    void print_summary() const {
        std::cout << "=== Factor Model: " << model_name << " ===\n";
        std::cout << "  Factors: " << n_factors << "  Assets: " << n_assets << "\n";
        for (size_t k = 0; k < factors.size(); ++k) {
            const auto& f = factors[k];
            std::cout << "  F" << k << " (" << f.name << ")"
                      << "  mean=" << std::fixed << std::setprecision(4) << f.mean_return
                      << "  vol="  << f.vol
                      << "  sharpe=" << f.sharpe << "\n";
        }
        std::cout << "\n  Asset Exposures:\n";
        std::cout << "  " << std::left << std::setw(8) << "Asset"
                  << std::right << std::setw(8) << "Alpha"
                  << std::setw(8) << "R2";
        for (size_t k = 0; k < n_factors; ++k)
            std::cout << std::setw(8) << ("B" + std::to_string(k));
        std::cout << "\n  " << std::string(8 + 8 + 8 + 8 * n_factors, '-') << "\n";

        for (const auto& exp : exposures) {
            std::cout << "  " << std::left << std::setw(8) << exp.asset
                      << std::right << std::setw(8) << std::setprecision(3) << exp.alpha
                      << std::setw(8) << exp.r_squared;
            for (double b : exp.betas)
                std::cout << std::setw(8) << b;
            std::cout << "\n";
        }
    }
};

// ============================================================
// PCA-based statistical factor extraction
// ============================================================
struct PCAFactorModel {
    MatrixD loadings;       // N x K: asset loadings on K factors
    MatrixD factor_returns; // T x K: time series of factor returns
    std::vector<double> eigenvalues;
    std::vector<double> variance_explained;
    size_t k_factors;

    void fit(const MatrixD& asset_returns, size_t k = 5) {
        k_factors = k;
        size_t T = asset_returns.rows();
        size_t N = asset_returns.cols();

        // Demean and standardize
        MatrixD X(T, N);
        std::vector<double> mu(N, 0.0), sig(N, 1.0);
        for (size_t j = 0; j < N; ++j) {
            WelfordState ws;
            for (size_t t = 0; t < T; ++t) ws.update(asset_returns(t, j));
            mu[j]  = ws.mean;
            sig[j] = ws.std_dev() > 1e-12 ? ws.std_dev() : 1.0;
        }
        for (size_t t = 0; t < T; ++t)
            for (size_t j = 0; j < N; ++j)
                X(t, j) = (asset_returns(t, j) - mu[j]) / sig[j];

        // Compute covariance matrix (N x N)
        MatrixD cov(N, N);
        simd::covariance_matrix(X, cov);

        // Eigendecomposition (top-k)
        auto [eigvecs, eigvals] = eigen_symmetric(cov);
        eigenvalues = eigvals;

        // Sort descending
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){
            return eigvals[a] > eigvals[b];
        });

        double total_var = std::accumulate(eigvals.begin(), eigvals.end(), 0.0);
        variance_explained.resize(k);

        loadings = MatrixD(N, k);
        for (size_t j = 0; j < k && j < N; ++j) {
            size_t col = idx[j];
            for (size_t i = 0; i < N; ++i)
                loadings(i, j) = eigvecs(i, col);
            variance_explained[j] = total_var > 0 ? eigvals[col] / total_var : 0.0;
        }

        // Factor returns: F = X * L (T x K)
        factor_returns = MatrixD(T, k);
        for (size_t t = 0; t < T; ++t)
            for (size_t j = 0; j < k; ++j)
                for (size_t i = 0; i < N; ++i)
                    factor_returns(t, j) += X(t, i) * loadings(i, j);
    }

    void print_variance_explained() const {
        std::cout << "PCA Variance Explained:\n";
        double cumul = 0.0;
        for (size_t k = 0; k < variance_explained.size(); ++k) {
            cumul += variance_explained[k];
            std::cout << "  PC" << k+1 << ": "
                      << std::fixed << std::setprecision(1)
                      << variance_explained[k] * 100.0 << "%"
                      << "  (cum=" << cumul * 100.0 << "%)\n";
        }
    }
};

// ============================================================
// Factor return forecasting: time-series regression + momentum
// ============================================================
struct FactorForecaster {
    size_t                   lookback;
    std::vector<std::deque<double>> histories;
    size_t                   n_factors;

    explicit FactorForecaster(size_t n_factors_, size_t lookback_ = 60)
        : lookback(lookback_), histories(n_factors_), n_factors(n_factors_) {}

    void update(const std::vector<double>& factor_returns) {
        assert(factor_returns.size() == n_factors);
        for (size_t k = 0; k < n_factors; ++k) {
            histories[k].push_back(factor_returns[k]);
            if (histories[k].size() > lookback) histories[k].pop_front();
        }
    }

    // AR(1) forecast for each factor
    std::vector<double> forecast_ar1() const {
        std::vector<double> preds(n_factors, 0.0);
        for (size_t k = 0; k < n_factors; ++k) {
            const auto& h = histories[k];
            if (h.size() < 5) continue;
            size_t T = h.size();
            // OLS: r_t = a + b * r_{t-1}
            double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
            for (size_t t = 1; t < T; ++t) {
                sum_x  += h[t-1];
                sum_y  += h[t];
                sum_xx += h[t-1] * h[t-1];
                sum_xy += h[t-1] * h[t];
            }
            double n = T - 1.0;
            double b = (n * sum_xy - sum_x * sum_y) /
                       (n * sum_xx - sum_x * sum_x + 1e-12);
            double a = (sum_y - b * sum_x) / n;
            preds[k] = a + b * h.back();
        }
        return preds;
    }

    // Momentum forecast: sign of mean return over lookback
    std::vector<double> forecast_momentum() const {
        std::vector<double> preds(n_factors, 0.0);
        for (size_t k = 0; k < n_factors; ++k) {
            const auto& h = histories[k];
            if (h.size() < 2) continue;
            double mean = 0;
            for (auto v : h) mean += v;
            mean /= h.size();
            preds[k] = mean > 0 ? 1.0 : -1.0;
        }
        return preds;
    }

    // Ensemble: blend AR1 and momentum
    std::vector<double> forecast_ensemble(double ar1_weight = 0.7) const {
        auto ar1 = forecast_ar1();
        auto mom = forecast_momentum();
        std::vector<double> blend(n_factors);
        for (size_t k = 0; k < n_factors; ++k)
            blend[k] = ar1_weight * ar1[k] + (1.0 - ar1_weight) * mom[k];
        return blend;
    }
};

// ============================================================
// Integration test
// ============================================================
void run_factor_model_test() {
    std::cout << "\n=== Factor Model Integration Test ===\n";

    const size_t T = 500, N = 10, K = 3;
    std::mt19937 rng(42);
    std::normal_distribution<double> norm(0, 1);

    // Generate K factors
    std::vector<Factor> factors(K);
    std::vector<std::string> fnames = {"Market", "SMB", "HML"};
    for (size_t k = 0; k < K; ++k) {
        factors[k].name    = fnames[k];
        factors[k].type    = (k == 0) ? FactorType::Market :
                             (k == 1) ? FactorType::SMB : FactorType::HML;
        factors[k].returns.resize(T);
        double mu = (k == 0) ? 0.0005 : 0.0001;
        for (auto& r : factors[k].returns)
            r = mu + norm(rng) * (k == 0 ? 0.01 : 0.005);
        factors[k].compute_stats();
    }

    // Generate asset returns: r_i = alpha_i + sum(beta_ik * F_k) + eps_i
    std::vector<double> true_betas_mkt  = {1.0, 0.9, 1.2, 0.8, 1.1, 0.95, 1.05, 0.85, 1.15, 1.0};
    std::vector<double> true_betas_smb  = {0.3, 0.5, 0.2, 0.7, 0.1, 0.4, 0.6, 0.3, 0.5, 0.2};
    std::vector<double> true_betas_hml  = {0.2, -0.1, 0.4, 0.3, -0.2, 0.1, 0.3, -0.1, 0.2, 0.4};
    std::vector<double> alphas          = {0.0002, -0.0001, 0.0003, 0.0, 0.0001,
                                            -0.0002, 0.0004, 0.0001, -0.0003, 0.0002};

    std::vector<std::string> asset_names;
    std::ostringstream oss;
    for (size_t i = 0; i < N; ++i) {
        oss.str("");
        oss << "ASSET" << i;
        asset_names.push_back(oss.str());
    }

    MatrixD asset_returns(T, N);
    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < N; ++i) {
            double r = alphas[i]
                     + true_betas_mkt[i] * factors[0].returns[t]
                     + true_betas_smb[i] * factors[1].returns[t]
                     + true_betas_hml[i] * factors[2].returns[t]
                     + norm(rng) * 0.008;  // idiosyncratic
            asset_returns(t, i) = r;
        }

    // Fit factor model
    FactorModel model;
    model.model_name = "Fama-French 3-Factor";
    model.factors    = factors;
    model.fit(asset_names, asset_returns, true);
    model.print_summary();

    // Risk attribution for first asset
    MatrixD fcov(K, K);
    for (size_t k = 0; k < K; ++k) {
        WelfordState ws;
        for (auto r : factors[k].returns) ws.update(r);
        fcov(k, k) = ws.variance(true);
    }
    auto attrib = model.risk_attribution(0, fcov);
    std::cout << "\n  Asset0 Risk Attribution:\n"
              << "    Factor:    " << std::fixed << std::setprecision(4)
              << attrib.factor_pct * 100 << "%\n"
              << "    Idiosync:  " << (1.0 - attrib.factor_pct) * 100 << "%\n";

    // PCA
    std::cout << "\n--- PCA Factor Extraction ---\n";
    PCAFactorModel pca;
    pca.fit(asset_returns, 3);
    pca.print_variance_explained();

    // Factor forecasting
    std::cout << "\n--- Factor Forecasting ---\n";
    FactorForecaster forecaster(K, 60);
    for (size_t t = 0; t < T; ++t) {
        std::vector<double> row(K);
        for (size_t k = 0; k < K; ++k) row[k] = factors[k].returns[t];
        forecaster.update(row);
    }
    auto preds = forecaster.forecast_ensemble();
    for (size_t k = 0; k < K; ++k)
        std::cout << "  " << fnames[k] << " forecast: "
                  << std::setprecision(6) << preds[k] << "\n";
}

int main() {
    run_factor_model_test();
    return 0;
}
