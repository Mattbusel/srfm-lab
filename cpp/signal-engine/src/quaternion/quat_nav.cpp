#include "quat_nav.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace srfm {

// ============================================================
// Static quaternion math helpers
// ============================================================

void QuatNav::quat_mul(const double* q1, const double* q2,
                       double* out) noexcept {
    const double w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
    const double w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
    out[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    out[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    out[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    out[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

void QuatNav::quat_inv(const double* q, double* out) noexcept {
    out[0] =  q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = -q[3];
}

void QuatNav::quat_normalize(double* q) noexcept {
    double n2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if (n2 < 1e-30) {
        // Degenerate: reset to identity quaternion
        q[0] = 1.0; q[1] = q[2] = q[3] = 0.0;
        return;
    }
    double inv_n = 1.0 / std::sqrt(n2);
    q[0] *= inv_n; q[1] *= inv_n; q[2] *= inv_n; q[3] *= inv_n;
}

double QuatNav::quat_angle(const double* q) noexcept {
    // Angle = 2 * arccos(|w|).  Clamp w to [-1,1] before arccos.
    double w = q[0];
    if (w < -1.0) w = -1.0;
    if (w >  1.0) w =  1.0;
    return 2.0 * std::acos(std::abs(w));
}

void QuatNav::quat_slerp(const double* q1, const double* q2,
                          double t, double* out) noexcept {
    double dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];

    // Choose shorter arc
    double q2a[4] = {q2[0], q2[1], q2[2], q2[3]};
    if (dot < 0.0) {
        dot = -dot;
        q2a[0] = -q2a[0]; q2a[1] = -q2a[1];
        q2a[2] = -q2a[2]; q2a[3] = -q2a[3];
    }
    if (dot > 1.0) dot = 1.0;

    double theta = std::acos(dot);

    if (theta < 1e-10) {
        // Nearly parallel: linear blend + normalise
        for (int i = 0; i < 4; ++i)
            out[i] = q1[i] + t * (q2a[i] - q1[i]);
    } else {
        double sin_theta = std::sin(theta);
        double s1 = std::sin((1.0 - t) * theta) / sin_theta;
        double s2 = std::sin(t * theta) / sin_theta;
        for (int i = 0; i < 4; ++i)
            out[i] = s1 * q1[i] + s2 * q2a[i];
    }
    quat_normalize(out);
}

double QuatNav::quat_geodesic_angle(const double* q1,
                                     const double* q2) noexcept {
    double dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];
    // Quaternions q and -q represent the same rotation; use |dot|.
    if (dot < 0.0) dot = -dot;
    if (dot > 1.0) dot = 1.0;
    return 2.0 * std::acos(dot);
}

void QuatNav::quat_extrapolate(const double* q1, const double* q2,
                                double* out) noexcept {
    // Compute the rotation delta that maps q1 -> q2:
    //   delta = q2 * q1^{-1}
    // Then apply delta to q2 to predict q3:
    //   q_pred = delta * q2 = (q2 * q1^{-1}) * q2
    double q1_inv[4];
    quat_inv(q1, q1_inv);

    double delta[4];
    quat_mul(q2, q1_inv, delta);
    quat_normalize(delta);

    quat_mul(delta, q2, out);
    quat_normalize(out);
}

// ============================================================
// Bar quaternion construction
// ============================================================

void QuatNav::build_bar_quat(double dt_norm, double price_norm,
                              double vol_norm, double mi_norm,
                              double* q_out) noexcept {
    // Raw components: w=time, x=price, y=volume, z=market_impact
    // Each is already normalised to (0, 1].
    // Small epsilon avoids zero vector on first bar.
    q_out[0] = dt_norm    + 1e-12;
    q_out[1] = price_norm + 1e-12;
    q_out[2] = vol_norm   + 1e-12;
    q_out[3] = mi_norm    + 1e-12;
    quat_normalize(q_out);
}

// ============================================================
// QuatNav constructor / reset
// ============================================================

QuatNav::QuatNav() noexcept
    : norm_price_max_(1e-8)
    , norm_vol_max_(1e-8)
    , norm_mi_max_(1e-8)
    , vol_ema_(0.0)
    , prev_close_(0.0)
    , prev_ts_ns_(0)
    , norm_dt_ref_(60.0)   // default: 1m bar = 60s
    , prev_bh_active_(false)
    , count_(0)
{
    // Identity quaternion
    Q_[0] = 1.0; Q_[1] = Q_[2] = Q_[3] = 0.0;
    q_prev_[0]  = 1.0; q_prev_[1]  = q_prev_[2]  = q_prev_[3]  = 0.0;
    q_prev2_[0] = 1.0; q_prev2_[1] = q_prev2_[2] = q_prev2_[3] = 0.0;
}

void QuatNav::reset() noexcept {
    norm_price_max_  = 1e-8;
    norm_vol_max_    = 1e-8;
    norm_mi_max_     = 1e-8;
    vol_ema_         = 0.0;
    prev_close_      = 0.0;
    prev_ts_ns_      = 0;
    norm_dt_ref_     = 60.0;
    prev_bh_active_  = false;
    count_           = 0;
    Q_[0] = 1.0; Q_[1] = Q_[2] = Q_[3] = 0.0;
    q_prev_[0]  = 1.0; q_prev_[1]  = q_prev_[2]  = q_prev_[3]  = 0.0;
    q_prev2_[0] = 1.0; q_prev2_[1] = q_prev2_[2] = q_prev2_[3] = 0.0;
}

// ============================================================
// Main update
// ============================================================

QuatNavOutput QuatNav::update(double close,
                               double volume,
                               int64_t timestamp_ns,
                               double bh_mass,
                               bool bh_was_active,
                               bool bh_active) noexcept {
    QuatNavOutput out{};
    out.lorentz_boost_applied  = false;
    out.lorentz_boost_rapidity = 0.0;

    // ── 1. Compute raw 4-space components ─────────────────────────────────

    // dt: time since last bar, normalised to reference bar duration.
    double dt_norm = 1.0;  // default: 1 bar
    if (count_ > 0 && prev_ts_ns_ > 0 && timestamp_ns > prev_ts_ns_) {
        double dt_sec = static_cast<double>(timestamp_ns - prev_ts_ns_)
                        / static_cast<double>(constants::NS_PER_SEC);
        dt_norm = dt_sec / norm_dt_ref_;
        // Auto-calibrate norm_dt_ref_ on first bar with real timestamps
        if (count_ == 1) norm_dt_ref_ = dt_sec;
    }

    // Price normalisation: rolling max with slow decay
    double px = close;
    if (px > norm_price_max_) {
        norm_price_max_ = px;
    } else {
        norm_price_max_ *= 0.99995;  // slower decay for price (broad range)
        if (norm_price_max_ < 1e-10) norm_price_max_ = 1e-10;
    }
    double price_norm = px / norm_price_max_;  // in (0, 1]

    // Volume normalisation
    if (volume > norm_vol_max_) {
        norm_vol_max_ = volume;
    } else {
        norm_vol_max_ *= 0.9999;
        if (norm_vol_max_ < 1e-10) norm_vol_max_ = 1e-10;
    }
    double vol_norm = volume / norm_vol_max_;

    // Market impact proxy: |log_return| * sqrt(volume / vol_ema)
    // This approximates Kyle's lambda scaled to [0, 1].
    double mi_raw = 0.0;
    if (count_ > 0 && prev_close_ > constants::EPSILON) {
        double log_ret = std::abs(std::log(close / prev_close_));
        vol_ema_ = VOL_EMA_ALPHA * volume + (1.0 - VOL_EMA_ALPHA) * vol_ema_;
        double vol_ratio = (vol_ema_ > constants::EPSILON)
                           ? std::sqrt(volume / vol_ema_)
                           : 1.0;
        mi_raw = log_ret * vol_ratio;
    } else {
        vol_ema_ = volume;
        mi_raw = 0.0;
    }
    if (mi_raw > norm_mi_max_) {
        norm_mi_max_ = mi_raw;
    } else {
        norm_mi_max_ *= 0.9998;
        if (norm_mi_max_ < 1e-12) norm_mi_max_ = 1e-12;
    }
    double mi_norm = (norm_mi_max_ > constants::EPSILON)
                     ? mi_raw / norm_mi_max_
                     : 0.0;

    // ── 2. Build bar quaternion ───────────────────────────────────────────
    double q_bar[4];
    build_bar_quat(dt_norm, price_norm, vol_norm, mi_norm, q_bar);

    out.bar_qw = q_bar[0];
    out.bar_qx = q_bar[1];
    out.bar_qy = q_bar[2];
    out.bar_qz = q_bar[3];

    // ── 3. Inter-bar rotation & angular velocity ──────────────────────────
    double angular_velocity = 0.0;

    if (count_ > 0) {
        // delta_q = q_bar * q_prev^{-1}
        double q_prev_inv[4];
        quat_inv(q_prev_, q_prev_inv);
        double delta_q[4];
        quat_mul(q_bar, q_prev_inv, delta_q);
        quat_normalize(delta_q);

        angular_velocity = quat_angle(delta_q);  // radians per bar

        // Update running orientation: Q_current = delta_q * Q_current
        double Q_new[4];
        quat_mul(delta_q, Q_, Q_new);
        quat_normalize(Q_new);
        std::memcpy(Q_, Q_new, 4 * sizeof(double));
    }

    // ── 4. Lorentz boost on regime boundary ──────────────────────────────
    // A transition in BH active state (false→true or true→false) represents
    // a regime frame change.  Express this as a rotation in the w-x (time-
    // price) plane rather than a hard parameter discontinuity.
    if (count_ > 0 && bh_was_active != bh_active) {
        // Rapidity η = arctanh(min(0.99, mass * LORENTZ_SCALE))
        double rapidity = std::atanh(
            std::min(0.99, bh_mass * LORENTZ_SCALE)
        );
        // Boost quaternion (rotation in w-x plane by angle η/2)
        double half_rap = rapidity * 0.5;
        double q_boost[4] = {
            std::cos(half_rap),
            std::sin(half_rap),
            0.0,
            0.0
        };
        quat_normalize(q_boost);

        double Q_boosted[4];
        quat_mul(q_boost, Q_, Q_boosted);
        quat_normalize(Q_boosted);
        std::memcpy(Q_, Q_boosted, 4 * sizeof(double));

        out.lorentz_boost_applied  = true;
        out.lorentz_boost_rapidity = rapidity;
    }

    // ── 5. Geodesic deviation ─────────────────────────────────────────────
    double geodesic_deviation = 0.0;

    if (count_ >= 2) {
        // Extrapolate: apply the same rotation that took q_prev2_ → q_prev_
        // to predict where q_bar "should" be on the inertial path.
        double q_predicted[4];
        quat_extrapolate(q_prev2_, q_prev_, q_predicted);

        // Deviation = geodesic angle between prediction and reality
        geodesic_deviation = quat_geodesic_angle(q_predicted, q_bar);

        // Curvature correction: gravitational well bends the geodesic
        // proportionally to BH mass concentration.
        if (bh_mass > 0.0) {
            geodesic_deviation *= (1.0 + CURVATURE_K * bh_mass);
        }
    }

    // ── 6. Advance history ────────────────────────────────────────────────
    std::memcpy(q_prev2_, q_prev_,  4 * sizeof(double));
    std::memcpy(q_prev_,  q_bar,    4 * sizeof(double));
    prev_close_     = close;
    prev_ts_ns_     = timestamp_ns;
    prev_bh_active_ = bh_active;
    ++count_;

    // ── 7. Fill output ────────────────────────────────────────────────────
    out.qw               = Q_[0];
    out.qx               = Q_[1];
    out.qy               = Q_[2];
    out.qz               = Q_[3];
    out.angular_velocity  = angular_velocity;
    out.geodesic_deviation = geodesic_deviation;

    return out;
}

} // namespace srfm
