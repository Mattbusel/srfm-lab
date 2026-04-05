#include "ou_detector.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <numeric>

namespace srfm {

// ============================================================
// OUDetector
// ============================================================

OUDetector::OUDetector(int window) noexcept
    : window_(std::min(window, MAX_WIN))
    , ring_idx_(0)
    , count_(0)
    , theta_(0.01)
    , mu_(0.0)
    , sigma_(0.01)
    , half_life_(69.3)   // log(2)/0.01
    , zscore_(0.0)
    , mu_ema_(0.0)
    , resid_std_(1.0)
    , refit_interval_(10)
    , bars_since_refit_(0)
{
    std::memset(prices_, 0, sizeof(prices_));
}

void OUDetector::run_ols() noexcept {
    // Collect the current window of prices into a contiguous array
    int n = std::min(count_, window_);
    if (n < 10) return;  // need enough data

    // Build x = X[0..n-2], y = X[1..n-1] from ring buffer
    // Ring buffer: oldest at ring_idx_, newest at ring_idx_ - 1 (mod window_)
    //
    // We perform OLS: y_i = a + b * x_i
    // where x_i = X_{t-1}, y_i = X_t (AR(1) form of OU)

    double sum_x  = 0.0, sum_y  = 0.0;
    double sum_xx = 0.0, sum_xy = 0.0;
    int m = n - 1;  // number of pairs

    for (int i = 0; i < m; ++i) {
        // Index of X_{t-1+i}: oldest is at ring_idx_ in the ring
        int idx_x = (ring_idx_ + i)     % window_;
        int idx_y = (ring_idx_ + i + 1) % window_;
        double x  = prices_[idx_x];
        double y  = prices_[idx_y];
        sum_x  += x;
        sum_y  += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    double denom = static_cast<double>(m) * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < constants::EPSILON) return;

    double b = (static_cast<double>(m) * sum_xy - sum_x * sum_y) / denom;
    double a = (sum_y - b * sum_x) / static_cast<double>(m);

    // OU parameters from AR(1) fit
    // b = exp(-theta), so theta = -log(b) if b > 0 and b < 1
    if (b <= 0.0 || b >= 2.0) return;  // not a valid mean-reverting process

    double theta = -std::log(std::clamp(b, 1e-6, 1.0 - 1e-6));
    double mu    = (theta > constants::EPSILON) ? a / theta : sum_x / m;

    // Compute residual std dev
    double sum_resid2 = 0.0;
    for (int i = 0; i < m; ++i) {
        int    idx_x = (ring_idx_ + i)     % window_;
        int    idx_y = (ring_idx_ + i + 1) % window_;
        double x     = prices_[idx_x];
        double y     = prices_[idx_y];
        double pred  = a + b * x;
        double resid = y - pred;
        sum_resid2  += resid * resid;
    }
    double resid_var = sum_resid2 / m;
    double resid_sd  = std::sqrt(std::max(resid_var, constants::EPSILON));

    // OU sigma (instantaneous vol, not annualized)
    // sigma^2 = var_resid / (1 - exp(-2*theta)) * 2*theta  (for continuous-time)
    double sigma_sq_continuous = resid_var * 2.0 * theta /
                                 std::max(1.0 - std::exp(-2.0 * theta), constants::EPSILON);
    double sigma_cont = std::sqrt(std::max(sigma_sq_continuous, constants::EPSILON));

    // Update estimates
    theta_     = theta;
    mu_        = mu;
    sigma_     = sigma_cont;
    resid_std_ = resid_sd;
    half_life_ = (theta > constants::EPSILON) ? std::log(2.0) / theta : 1e6;

    // Cap half-life
    half_life_ = std::clamp(half_life_, 1.0, 10000.0);
}

OUOutput OUDetector::update(double price) noexcept {
    // Store price in ring buffer
    prices_[ring_idx_] = price;
    ring_idx_          = (ring_idx_ + 1) % window_;
    ++count_;

    // Update fast mu estimate via EMA (alpha = 2/(window+1))
    double alpha = 2.0 / (window_ + 1.0);
    if (count_ == 1) {
        mu_ema_ = price;
    } else {
        mu_ema_ = alpha * price + (1.0 - alpha) * mu_ema_;
    }

    // Re-run OLS periodically
    ++bars_since_refit_;
    if (bars_since_refit_ >= refit_interval_ && count_ >= window_) {
        run_ols();
        bars_since_refit_ = 0;
    } else if (count_ >= window_) {
        // Use fast EMA-based mu when not refitting
        mu_ = mu_ema_;
    }

    // Compute z-score
    double spread = price - mu_;
    double denom  = std::max(resid_std_, constants::EPSILON);
    zscore_       = spread / denom;

    bool long_sig  = zscore_ < constants::OU_ZSCORE_LONG;
    bool short_sig = zscore_ > constants::OU_ZSCORE_SHORT;

    return { theta_, mu_, sigma_, half_life_, zscore_,
             spread, long_sig, short_sig, is_warm() };
}

void OUDetector::reset() noexcept {
    std::memset(prices_, 0, sizeof(prices_));
    ring_idx_         = 0;
    count_            = 0;
    theta_            = 0.01;
    mu_               = 0.0;
    sigma_            = 0.01;
    half_life_        = 69.3;
    zscore_           = 0.0;
    mu_ema_           = 0.0;
    resid_std_        = 1.0;
    bars_since_refit_ = 0;
}

// ============================================================
// SpreadOUDetector
// ============================================================

SpreadOUDetector::SpreadOUDetector(int window) noexcept
    : ou_(window)
{}

OUOutput SpreadOUDetector::update(double spread) noexcept {
    return ou_.update(spread);
}

// ============================================================
// Standalone helpers
// ============================================================

/// Adf test approximation: returns t-stat for unit root test.
/// Negative values suggest stationarity (mean reversion).
double adf_tstat(const double* prices, int n) noexcept {
    if (n < 10) return 0.0;

    // y[t] - y[t-1] = alpha + beta * y[t-1] + eps
    // Test: H0: beta = 0 (unit root), H1: beta < 0 (stationary)
    double sum_y  = 0.0, sum_y2 = 0.0, sum_dy = 0.0, sum_ydy = 0.0;
    int m = n - 1;

    for (int i = 0; i < m; ++i) {
        double y  = prices[i];
        double dy = prices[i + 1] - prices[i];
        sum_y   += y;
        sum_y2  += y * y;
        sum_dy  += dy;
        sum_ydy += y * dy;
    }

    double denom = (double)m * sum_y2 - sum_y * sum_y;
    if (std::abs(denom) < constants::EPSILON) return 0.0;

    double beta = ((double)m * sum_ydy - sum_y * sum_dy) / denom;

    // Compute residual variance
    double alpha   = (sum_dy - beta * sum_y) / m;
    double sum_e2  = 0.0;
    for (int i = 0; i < m; ++i) {
        double dy_hat = alpha + beta * prices[i];
        double e      = (prices[i + 1] - prices[i]) - dy_hat;
        sum_e2       += e * e;
    }
    double s2  = sum_e2 / (m - 2);
    double se  = std::sqrt(s2 * m / denom + constants::EPSILON);
    return beta / se;
}

} // namespace srfm
