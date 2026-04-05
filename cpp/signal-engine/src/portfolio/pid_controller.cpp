#include "pid_controller.hpp"
#include <algorithm>
#include <cmath>

namespace srfm {

// ============================================================
// PIDController
// ============================================================

PIDController::PIDController(double kp, double ki, double kd,
                               double output_min, double output_max,
                               double integral_min, double integral_max) noexcept
    : kp_(kp)
    , ki_(ki)
    , kd_(kd)
    , output_min_(output_min)
    , output_max_(output_max)
    , integral_min_(integral_min)
    , integral_max_(integral_max)
    , integral_(0.0)
    , prev_error_(0.0)
    , derivative_(0.0)
    , output_(0.0)
    , first_call_(true)
{}

double PIDController::update(double setpoint, double measurement,
                              double dt) noexcept {
    double error = setpoint - measurement;

    // Proportional term
    double p_term = kp_ * error;

    // Integral term with anti-windup clamping
    integral_ += ki_ * error * dt;
    integral_  = std::clamp(integral_, integral_min_, integral_max_);
    double i_term = integral_;

    // Derivative term (on measurement to avoid derivative kick on setpoint changes)
    double d_term = 0.0;
    if (!first_call_ && dt > constants::EPSILON) {
        // Use error derivative (simpler; setpoint rarely changes abruptly here)
        double deriv_raw = (error - prev_error_) / dt;
        // Low-pass filter derivative: derivative_ = 0.7*derivative_ + 0.3*deriv
        derivative_ = 0.7 * derivative_ + 0.3 * deriv_raw;
        d_term = kd_ * derivative_;
    }

    output_     = std::clamp(p_term + i_term + d_term, output_min_, output_max_);
    prev_error_ = error;
    first_call_ = false;

    return output_;
}

void PIDController::reset() noexcept {
    integral_   = 0.0;
    prev_error_ = 0.0;
    derivative_ = 0.0;
    output_     = 0.0;
    first_call_ = true;
}

void PIDController::set_gains(double kp, double ki, double kd) noexcept {
    kp_ = kp; ki_ = ki; kd_ = kd;
}

void PIDController::set_output_limits(double min_val, double max_val) noexcept {
    output_min_ = min_val;
    output_max_ = max_val;
}

// ============================================================
// AdaptiveThresholdController
// ============================================================

AdaptiveThresholdController::AdaptiveThresholdController() noexcept
    // stale_threshold PID: target error_rate = 0.1 (10% stale)
    // Output range: [1, 50] bars
    : stale_pid_(0.002, 0.0001, 0.001, 1.0, 50.0, -5.0, 5.0)
    // max_frac PID: target vol_ratio = 1.0 (at target vol)
    // Output range: [0.01, 1.0]
    , frac_pid_(0.002, 0.0001, 0.001, 0.01, 1.0, -0.1, 0.1)
    , stale_threshold_(10.0)
    , max_frac_(0.2)
{}

AdaptiveThresholdController::Output
AdaptiveThresholdController::update(double error_rate, double vol_ratio,
                                     double dt) noexcept {
    // stale_threshold: increase when error_rate is high (signals stale more often)
    // setpoint = 0.05 (target 5% stale rate), measurement = error_rate
    double stale_adj = stale_pid_.update(0.05, error_rate, dt);
    stale_threshold_ = std::clamp(stale_threshold_ + stale_adj, 1.0, 50.0);

    // max_frac: reduce when vol_ratio > 1 (more volatile than target)
    // setpoint = 1.0 (at target vol), measurement = vol_ratio
    double frac_adj  = frac_pid_.update(1.0, vol_ratio, dt);
    max_frac_        = std::clamp(max_frac_ + frac_adj, 0.01, 1.0);

    return { stale_threshold_, max_frac_ };
}

void AdaptiveThresholdController::reset() noexcept {
    stale_pid_.reset();
    frac_pid_.reset();
    stale_threshold_ = 10.0;
    max_frac_        = 0.2;
}

// ============================================================
// Utility: Ziegler-Nichols auto-tune approximation
// ============================================================

struct PIDGains {
    double kp, ki, kd;
};

/// Compute Ziegler-Nichols gains from ultimate gain and period.
/// method: 0=classic, 1=no-overshoot, 2=some-overshoot
PIDGains ziegler_nichols(double ku, double tu, int method = 0) noexcept {
    switch (method) {
    case 1:  // No overshoot
        return { 0.20 * ku, 0.40 * ku / tu, 0.066 * ku * tu };
    case 2:  // Some overshoot
        return { 0.33 * ku, 0.67 * ku / tu, 0.11  * ku * tu };
    default: // Classic
        return { 0.60 * ku, 1.20 * ku / tu, 0.075 * ku * tu };
    }
}

/// Cohen-Coon tuning for first-order plus dead time systems.
PIDGains cohen_coon(double K, double tau, double theta) noexcept {
    double r = theta / tau;
    double kp = (1.0 / K) * (1.33 + r / 4.0) * (tau / theta);
    double ki = kp * theta / (0.9 + r / 12.0) / tau;
    double kd = kp * 4.0  / (11.0 + 2.0 * r) * theta / tau;
    return { kp, ki, kd };
}

} // namespace srfm
