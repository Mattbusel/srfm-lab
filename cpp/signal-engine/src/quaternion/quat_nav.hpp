#pragma once
#include "srfm/types.hpp"
#include <cmath>

namespace srfm {

/// Output from the quaternion navigation layer for a single bar.
struct QuatNavOutput {
    // Bar quaternion: unit quaternion representing this bar's position in
    // (t, Price, Volume, MarketImpact) 4-space.
    double bar_qw, bar_qx, bar_qy, bar_qz;

    // Running orientation Q_current: accumulated product of successive
    // inter-bar rotation quaternions.  Represents the market's current
    // heading in price-space.
    double qw, qx, qy, qz;

    // Angular velocity: angle (radians) swept by Q_current this bar.
    // High omega => rapid regime rotation.  Low omega => stable heading.
    double angular_velocity;

    // Geodesic deviation: angle (radians) between the SLERP-extrapolated
    // predicted next position and the actual next bar quaternion.
    // Corrected upward by BH mass concentration (curvature effect).
    // Valid from bar index >= 2; 0.0 before that.
    double geodesic_deviation;

    // Lorentz boost applied this bar (regime boundary crossing).
    bool   lorentz_boost_applied;
    double lorentz_boost_rapidity;
};

/// Quaternion navigation layer.
///
/// Maintains a rolling orientation quaternion Q_current that tracks the
/// market's rotational heading in the SRFM 4-space.  Each bar provides:
///
///   1. A unit quaternion q constructed from (dt, price, volume, MI).
///   2. The inter-bar rotation delta_q = q_new * q_prev^{-1}.
///   3. Q_current updated by delta_q (normalised after every step).
///   4. Angular velocity omega = angle(delta_q) / dt_bars.
///   5. Geodesic deviation: extrapolated prediction error, curvature-
///      corrected near BH mass concentrations.
///   6. Lorentz boosts on BH regime transitions expressed as quaternion
///      rotations rather than parameter discontinuities.
///
/// Normalisation invariant: |Q_current| == 1.0 is enforced after every
/// update.  Drift is caught immediately rather than compounding silently.
class QuatNav {
public:
    QuatNav() noexcept;

    /// Update with a new bar.  bh_mass is the current black hole mass from
    /// BHState; bh_was_active / bh_active are the active flags from the
    /// previous and current bars respectively (used for Lorentz boosts).
    QuatNavOutput update(double close,
                         double volume,
                         int64_t timestamp_ns,
                         double bh_mass,
                         bool bh_was_active,
                         bool bh_active) noexcept;

    void reset() noexcept;

    // Current orientation quaternion accessors
    double qw() const noexcept { return Q_[0]; }
    double qx() const noexcept { return Q_[1]; }
    double qy() const noexcept { return Q_[2]; }
    double qz() const noexcept { return Q_[3]; }

    int bar_count() const noexcept { return count_; }

    // ----------------------------------------------------------------
    // Static quaternion math helpers (exposed for testing)
    // ----------------------------------------------------------------

    /// Quaternion multiply: out = q1 * q2.  out may alias neither input.
    static void quat_mul(const double* q1, const double* q2,
                         double* out) noexcept;

    /// Unit quaternion inverse = conjugate (negate x,y,z).
    static void quat_inv(const double* q, double* out) noexcept;

    /// Normalise q in-place.  On degenerate zero input resets to identity.
    static void quat_normalize(double* q) noexcept;

    /// Rotation angle of q: 2 * arccos(clamp(|w|, 0, 1)).
    static double quat_angle(const double* q) noexcept;

    /// SLERP: out = slerp(q1, q2, t).  Stable for t in [0, 1].
    static void quat_slerp(const double* q1, const double* q2,
                           double t, double* out) noexcept;

    /// Geodesic angle between two unit quaternions (range [0, pi]).
    static double quat_geodesic_angle(const double* q1,
                                      const double* q2) noexcept;

    /// Extrapolate: apply the same rotation that maps q1 -> q2 once more.
    /// out = (q2 * q1^{-1}) * q2.  More stable than slerp(q1, q2, 2.0).
    static void quat_extrapolate(const double* q1, const double* q2,
                                 double* out) noexcept;

private:
    // Build a normalised bar quaternion from raw 4-components.
    static void build_bar_quat(double dt_norm, double price_norm,
                                double vol_norm, double mi_norm,
                                double* q_out) noexcept;

    // Rolling normalisers (slow-decay max, same pattern as BHState::norm_dx_max_)
    double norm_price_max_;   // rolling max of close
    double norm_vol_max_;     // rolling max of volume
    double norm_mi_max_;      // rolling max of market impact proxy
    double vol_ema_;          // volume EMA (for MI computation)
    double prev_close_;
    int64_t prev_ts_ns_;
    double norm_dt_ref_;      // reference dt in seconds (60s for 1m bars; auto-calibrated)

    // Current orientation quaternion Q_current [w, x, y, z]
    double Q_[4];

    // Previous bar quaternion (bar n)
    double q_prev_[4];

    // Bar n-1 quaternion (for geodesic extrapolation)
    double q_prev2_[4];

    bool prev_bh_active_;
    int  count_;

    // Curvature coefficient: geodesic_deviation *= (1 + CURVATURE_K * mass)
    static constexpr double CURVATURE_K  = 0.15;

    // Lorentz rapidity scale: η = arctanh(min(0.99, mass * LORENTZ_SCALE))
    static constexpr double LORENTZ_SCALE = 0.40;

    // Volume EMA alpha (period ~20)
    static constexpr double VOL_EMA_ALPHA = 2.0 / 21.0;
};

} // namespace srfm
