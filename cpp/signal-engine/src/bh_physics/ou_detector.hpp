#pragma once
#include "srfm/types.hpp"
#include <array>

namespace srfm {

struct OUOutput {
    double theta;       // mean-reversion speed (per bar)
    double mu;          // long-run mean (estimated)
    double sigma;       // noise volatility
    double half_life;   // log(2) / theta in bars
    double zscore;      // (current_price - mu) / sigma_of_residuals
    double spread;      // current price - mu
    bool   long_signal; // zscore < OU_ZSCORE_LONG
    bool   short_signal;// zscore > OU_ZSCORE_SHORT
    bool   is_warm;
};

/// Ornstein-Uhlenbeck process detector.
///
/// Uses rolling OLS regression over a 50-bar window to estimate:
///   dX_t = theta * (mu - X_t) * dt + sigma * dW_t
///   => X_t = a + b * X_{t-1} + eps
///   where b = 1 - theta, a = theta * mu
///
/// half_life = log(2) / theta
/// z-score = (X_t - mu) / sigma_residual
///
/// Signals: long when z < -1.5, short when z > 1.5
class OUDetector {
public:
    explicit OUDetector(int window = constants::OU_WINDOW) noexcept;

    /// Update with a new price. Returns OU state.
    OUOutput update(double price) noexcept;

    double theta()     const noexcept { return theta_; }
    double mu()        const noexcept { return mu_; }
    double sigma()     const noexcept { return sigma_; }
    double half_life() const noexcept { return half_life_; }
    double zscore()    const noexcept { return zscore_; }
    bool   is_warm()   const noexcept { return count_ >= window_; }

    void reset() noexcept;

private:
    void   run_ols() noexcept;

    int    window_;

    // Ring buffer for prices
    static constexpr int MAX_WIN = 200;
    double prices_[MAX_WIN];
    int    ring_idx_;
    int    count_;

    // Current estimates
    double theta_;
    double mu_;
    double sigma_;
    double half_life_;
    double zscore_;
    double mu_ema_;    // EMA of price for fast mu estimate
    double resid_std_; // std dev of OU residuals

    // OLS refit interval (don't refit every bar — too expensive)
    int    refit_interval_;
    int    bars_since_refit_;
};

// ============================================================
// Spread OU: for pairs / cointegration
// ============================================================

class SpreadOUDetector {
public:
    SpreadOUDetector(int window = constants::OU_WINDOW) noexcept;

    /// Update with the spread: spread = log(price_a) - hedge_ratio * log(price_b)
    OUOutput update(double spread) noexcept;

    bool is_warm() const noexcept { return ou_.is_warm(); }

private:
    OUDetector ou_;
};

} // namespace srfm
