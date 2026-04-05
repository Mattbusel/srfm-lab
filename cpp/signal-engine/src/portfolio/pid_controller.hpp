#pragma once
#include "srfm/types.hpp"

namespace srfm {

/// PID controller with anti-windup (clamped integral).
///
/// Gains: Kp=0.002, Ki=0.0001, Kd=0.001
/// Used for adaptive threshold tuning (stale_threshold, max_frac).
class PIDController {
public:
    PIDController(double kp          = 0.002,
                  double ki          = 0.0001,
                  double kd          = 0.001,
                  double output_min  = -1.0,
                  double output_max  =  1.0,
                  double integral_min= -10.0,
                  double integral_max=  10.0) noexcept;

    /// Update: setpoint = target, measurement = current value.
    /// Returns control output.
    double update(double setpoint, double measurement, double dt = 1.0) noexcept;

    /// Reset integral and derivative state.
    void reset() noexcept;

    double output()     const noexcept { return output_; }
    double error()      const noexcept { return prev_error_; }
    double integral()   const noexcept { return integral_; }
    double derivative() const noexcept { return derivative_; }

    void set_gains(double kp, double ki, double kd) noexcept;
    void set_output_limits(double min_val, double max_val) noexcept;

private:
    double kp_, ki_, kd_;
    double output_min_, output_max_;
    double integral_min_, integral_max_;
    double integral_;
    double prev_error_;
    double derivative_;
    double output_;
    bool   first_call_;
};

// ============================================================
// Dual PID controller: manages two adaptive thresholds.
// PID 1: stale_threshold (how long without update before considering signal stale)
// PID 2: max_frac (maximum fraction of capital to deploy)
// ============================================================

class AdaptiveThresholdController {
public:
    AdaptiveThresholdController() noexcept;

    struct Output {
        double stale_threshold;  // bars
        double max_frac;         // [0, 1]
    };

    /// Update based on current performance metrics.
    /// error_rate: fraction of signals that were wrong
    /// vol_ratio:  realized_vol / target_vol
    Output update(double error_rate, double vol_ratio, double dt = 1.0) noexcept;

    void reset() noexcept;

    double stale_threshold() const noexcept { return stale_threshold_; }
    double max_frac()        const noexcept { return max_frac_; }

private:
    PIDController stale_pid_;
    PIDController frac_pid_;
    double        stale_threshold_;
    double        max_frac_;
};

} // namespace srfm
