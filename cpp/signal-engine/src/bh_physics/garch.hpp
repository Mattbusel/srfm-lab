#pragma once
#include "srfm/types.hpp"

namespace srfm {

struct GARCHOutput {
    double variance;     // current conditional variance
    double vol;          // sqrt(variance) = current daily vol
    double vol_scale;    // target_vol / sqrt(variance * ann_factor)
    double log_return;   // last log return
    bool   is_warm;
};

/// Online GARCH(1,1) tracker.
///
/// Model: sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}
/// Parameters: omega=0.000001, alpha=0.1, beta=0.85
///
/// vol_scale = target_vol / sqrt(variance * ann_factor)
/// Used to size positions so each instrument has consistent volatility.
class GARCHTracker {
public:
    GARCHTracker(double omega      = constants::GARCH_OMEGA,
                 double alpha      = constants::GARCH_ALPHA,
                 double beta       = constants::GARCH_BETA,
                 double target_vol = constants::TARGET_VOL,
                 double ann_factor = 252.0) noexcept;

    /// Update with a new close price. Returns GARCH output.
    GARCHOutput update(double close) noexcept;

    /// Update with pre-computed log return.
    GARCHOutput update_with_return(double log_ret) noexcept;

    double variance()  const noexcept { return variance_; }
    double vol_scale() const noexcept { return vol_scale_; }
    bool   is_warm()   const noexcept { return count_ >= warmup_bars_; }

    /// Returns the unconditional variance: omega / (1 - alpha - beta)
    double unconditional_variance() const noexcept;

    /// Returns the half-life of variance shocks (in bars)
    double shock_half_life() const noexcept;

    void reset() noexcept;

private:
    void   recompute_vol_scale() noexcept;

    double omega_;
    double alpha_;
    double beta_;
    double target_vol_;
    double ann_factor_;

    double variance_;
    double prev_return_;
    double vol_scale_;

    double prev_close_;
    int    count_;
    bool   has_prev_;

    static constexpr int warmup_bars_ = 20;
};

// ============================================================
// EGARCH (Exponential GARCH) — handles asymmetric volatility
// ============================================================

class EGARCHTracker {
public:
    EGARCHTracker(double omega = -0.1,
                  double alpha = 0.1,
                  double gamma = -0.05,  // asymmetry (leverage effect)
                  double beta  = 0.85) noexcept;

    double update(double log_return) noexcept;
    double log_variance() const noexcept { return log_var_; }
    double variance()     const noexcept;
    bool   is_warm()      const noexcept { return count_ >= 20; }
    void   reset()        noexcept;

private:
    double omega_, alpha_, gamma_, beta_;
    double log_var_;
    double prev_std_resid_;  // standardized residual
    int    count_;
};

} // namespace srfm
