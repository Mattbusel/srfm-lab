/// test_quat_nav.cpp
/// Tests for the quaternion navigation layer.
/// Framework: same hand-rolled framework used in test_bh_physics.cpp.

#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>
#include <vector>

#include "srfm/types.hpp"
#include "../src/quaternion/quat_nav.hpp"

using namespace srfm;

// ============================================================
// Minimal test framework (matches existing suite)
// ============================================================

static int g_pass = 0, g_fail = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); \
        ++g_fail; \
    } else { ++g_pass; } \
} while(0)

#define CHECK_CLOSE(a, b, tol) do { \
    double _a = (a), _b = (b), _t = (tol); \
    if (std::abs(_a - _b) > _t) { \
        std::fprintf(stderr, "FAIL  %s:%d  %s  got %.10g  expected %.10g  diff %.3e\n", \
                     __FILE__, __LINE__, #a, _a, _b, std::abs(_a - _b)); \
        ++g_fail; \
    } else { ++g_pass; } \
} while(0)

#define CHECK_CLOSE_REL(a, b, rel_tol) do { \
    double _a = (a), _b = (b); \
    double _denom = std::max(std::abs(_b), 1e-15); \
    double _rel = std::abs(_a - _b) / _denom; \
    if (_rel > (rel_tol)) { \
        std::fprintf(stderr, "FAIL  %s:%d  %s  got %.10g  expected %.10g  rel_err=%.3e\n", \
                     __FILE__, __LINE__, #a, _a, _b, _rel); \
        ++g_fail; \
    } else { ++g_pass; } \
} while(0)

static void section(const char* name) {
    std::printf("--- %s ---\n", name);
}

// Helpers
static double qnorm(const double* q) {
    return std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
}

static int64_t bar_ts(int bar_idx, int interval_sec = 60) {
    return (1700000000LL + static_cast<int64_t>(bar_idx) * interval_sec)
           * constants::NS_PER_SEC;
}

// ============================================================
// 1. Quaternion math: multiply then inverse => identity
// ============================================================

static void test_quat_mul_identity() {
    section("quat_mul: q * q^{-1} = identity");

    double q[4] = {0.6, 0.4, -0.5, 0.5};
    QuatNav::quat_normalize(q);

    double q_inv[4];
    QuatNav::quat_inv(q, q_inv);

    double result[4];
    QuatNav::quat_mul(q, q_inv, result);

    CHECK_CLOSE(result[0], 1.0, 1e-12);
    CHECK_CLOSE(result[1], 0.0, 1e-12);
    CHECK_CLOSE(result[2], 0.0, 1e-12);
    CHECK_CLOSE(result[3], 0.0, 1e-12);
}

// ============================================================
// 2. Quaternion normalisation: output is always unit norm
// ============================================================

static void test_quat_normalize_unit_norm() {
    section("quat_normalize: output is unit norm");

    double q1[4] = {3.0, 4.0, 0.0, 0.0};
    QuatNav::quat_normalize(q1);
    CHECK_CLOSE(qnorm(q1), 1.0, 1e-14);

    double q2[4] = {0.001, 0.001, 0.001, 0.001};
    QuatNav::quat_normalize(q2);
    CHECK_CLOSE(qnorm(q2), 1.0, 1e-14);

    // Degenerate zero input => identity
    double q3[4] = {0.0, 0.0, 0.0, 0.0};
    QuatNav::quat_normalize(q3);
    CHECK_CLOSE(q3[0], 1.0, 1e-14);
    CHECK_CLOSE(q3[1], 0.0, 1e-14);
}

// ============================================================
// 3. SLERP: endpoints and midpoint
// ============================================================

static void test_slerp_endpoints() {
    section("SLERP: t=0 returns q1, t=1 returns q2");

    double q1[4] = {1.0, 0.0, 0.0, 0.0};   // identity
    double q2[4] = {0.0, 1.0, 0.0, 0.0};   // 180° rotation around x
    QuatNav::quat_normalize(q1);
    QuatNav::quat_normalize(q2);

    double out0[4], out1[4];
    QuatNav::quat_slerp(q1, q2, 0.0, out0);
    QuatNav::quat_slerp(q1, q2, 1.0, out1);

    CHECK_CLOSE(out0[0], q1[0], 1e-10);
    CHECK_CLOSE(out0[1], q1[1], 1e-10);
    CHECK_CLOSE(out1[0], q2[0], 1e-10);
    CHECK_CLOSE(out1[1], q2[1], 1e-10);

    // Midpoint should be unit norm
    double out_mid[4];
    QuatNav::quat_slerp(q1, q2, 0.5, out_mid);
    CHECK_CLOSE(qnorm(out_mid), 1.0, 1e-12);
}

static void test_slerp_midpoint_equidistant() {
    section("SLERP: midpoint is equidistant from endpoints");

    double q1[4] = {0.6, 0.8, 0.0, 0.0};
    double q2[4] = {0.3, 0.1, 0.9, 0.3};
    QuatNav::quat_normalize(q1);
    QuatNav::quat_normalize(q2);

    double mid[4];
    QuatNav::quat_slerp(q1, q2, 0.5, mid);

    double d1 = QuatNav::quat_geodesic_angle(q1, mid);
    double d2 = QuatNav::quat_geodesic_angle(mid, q2);

    // Both distances should be equal within floating-point tolerance
    CHECK_CLOSE(d1, d2, 1e-10);

    // Output must be unit norm
    CHECK_CLOSE(qnorm(mid), 1.0, 1e-12);
}

// ============================================================
// 4. Geodesic angle: identical quaternions => 0
// ============================================================

static void test_geodesic_angle_identical() {
    section("geodesic_angle: identical quaternions => 0");

    double q[4] = {0.5, 0.5, 0.5, 0.5};
    QuatNav::quat_normalize(q);

    double angle = QuatNav::quat_geodesic_angle(q, q);
    CHECK_CLOSE(angle, 0.0, 1e-12);
}

static void test_geodesic_angle_orthogonal() {
    section("geodesic_angle: orthogonal quaternions => pi/2");

    // Two quaternions with zero dot product => geodesic = pi/2
    double q1[4] = {1.0, 0.0, 0.0, 0.0};
    double q2[4] = {0.0, 1.0, 0.0, 0.0};

    double angle = QuatNav::quat_geodesic_angle(q1, q2);
    CHECK_CLOSE(angle, M_PI / 2.0, 1e-10);
}

// ============================================================
// 5. Extrapolation: collinear sequence => near-zero deviation
// ============================================================

static void test_extrapolate_collinear() {
    section("quat_extrapolate: constant rotation => prediction ~= next bar");

    // Create a constant rotation: each bar adds the same delta_q.
    double axis_q[4] = {std::cos(0.05), std::sin(0.05), 0.0, 0.0};
    QuatNav::quat_normalize(axis_q);

    double q0[4] = {1.0, 0.0, 0.0, 0.0};  // start = identity
    double q1[4], q2_true[4];
    QuatNav::quat_mul(axis_q, q0, q1);   QuatNav::quat_normalize(q1);
    QuatNav::quat_mul(axis_q, q1, q2_true); QuatNav::quat_normalize(q2_true);

    double q2_pred[4];
    QuatNav::quat_extrapolate(q0, q1, q2_pred);

    // Prediction should closely match the true next quaternion
    double dev = QuatNav::quat_geodesic_angle(q2_pred, q2_true);
    CHECK_CLOSE(dev, 0.0, 1e-10);
}

// ============================================================
// 6. QuatNav: running orientation stays unit norm after many updates
// ============================================================

static void test_qutnav_normalisation_preserved() {
    section("QuatNav: |Q_current| == 1.0 after many updates");

    QuatNav nav;
    double price = 50000.0;

    for (int i = 0; i < 500; ++i) {
        // Varied price motion to stress all code paths
        if      (i % 7 == 0)  price *= 1.005;
        else if (i % 11 == 0) price *= 0.994;
        else                   price *= 1.0001;

        double vol = 1000.0 + 200.0 * (i % 13);
        auto out = nav.update(price, vol, bar_ts(i), 1.5, false, false);

        // Bar quaternion must be unit norm
        double q_bar[4] = {out.bar_qw, out.bar_qx, out.bar_qy, out.bar_qz};
        CHECK_CLOSE(qnorm(q_bar), 1.0, 1e-12);

        // Running orientation must be unit norm
        double Q[4] = {out.qw, out.qx, out.qy, out.qz};
        CHECK_CLOSE(qnorm(Q), 1.0, 1e-12);

        // Angular velocity must be non-negative
        CHECK(out.angular_velocity >= 0.0);
    }
}

// ============================================================
// 7. QuatNav: reset restores initial state
// ============================================================

static void test_qutnav_reset() {
    section("QuatNav: reset restores identity orientation");

    QuatNav nav;
    double price = 50000.0;
    for (int i = 0; i < 50; ++i) {
        price *= 1.002;
        nav.update(price, 1000.0, bar_ts(i), 1.0, false, i > 30);
    }

    nav.reset();

    CHECK(nav.bar_count() == 0);
    CHECK_CLOSE(nav.qw(), 1.0, 1e-14);
    CHECK_CLOSE(nav.qx(), 0.0, 1e-14);
    CHECK_CLOSE(nav.qy(), 0.0, 1e-14);
    CHECK_CLOSE(nav.qz(), 0.0, 1e-14);
}

// ============================================================
// 8. QuatNav: angular velocity is higher during volatile regime
// ============================================================

static void test_angular_velocity_volatile_vs_stable() {
    section("QuatNav: angular velocity higher in volatile regime");

    QuatNav nav_stable, nav_volatile;

    // Stable: tiny consistent moves
    double p_s = 50000.0;
    double sum_omega_stable = 0.0;
    for (int i = 0; i < 100; ++i) {
        p_s *= 1.0001;
        auto out = nav_stable.update(p_s, 1000.0, bar_ts(i), 0.5, false, false);
        if (i > 5) sum_omega_stable += out.angular_velocity;
    }

    // Volatile: large alternating moves
    double p_v = 50000.0;
    double sum_omega_volatile = 0.0;
    for (int i = 0; i < 100; ++i) {
        p_v *= (i % 2 == 0 ? 1.05 : 0.952);
        auto out = nav_volatile.update(p_v, 5000.0 + i * 100.0, bar_ts(i),
                                       2.0, false, false);
        if (i > 5) sum_omega_volatile += out.angular_velocity;
    }

    std::printf("  Mean omega stable=%.6f  volatile=%.6f\n",
                sum_omega_stable / 95, sum_omega_volatile / 95);
    CHECK(sum_omega_volatile > sum_omega_stable);
}

// ============================================================
// 9. QuatNav: geodesic deviation is higher with mass concentration
// ============================================================

static void test_geodesic_deviation_curvature_correction() {
    section("QuatNav: curvature correction amplifies geodesic deviation with mass");

    // Feed the same 3 bars to two navigators, one with BH mass, one without.
    // The deviation with mass should be >= that without mass.
    double prices[3]   = {50000.0, 50050.0, 50080.0};
    double vols[3]     = {1000.0,  1200.0,  900.0};

    QuatNav nav_no_mass, nav_with_mass;

    // Warm up with 2 bars (geodesic needs count >= 2)
    for (int i = 0; i < 2; ++i) {
        nav_no_mass.update(prices[i], vols[i], bar_ts(i), 0.0, false, false);
        nav_with_mass.update(prices[i], vols[i], bar_ts(i), 0.0, false, false);
    }

    // Third bar: apply mass to one, not the other
    auto out_no_mass   = nav_no_mass.update(prices[2], vols[2], bar_ts(2),
                                             0.0, false, false);
    auto out_with_mass = nav_with_mass.update(prices[2], vols[2], bar_ts(2),
                                               3.0, false, false);

    std::printf("  Geodesic deviation  no_mass=%.6f  with_mass=%.6f\n",
                out_no_mass.geodesic_deviation,
                out_with_mass.geodesic_deviation);

    // With mass > 0, curvature correction must amplify deviation
    CHECK(out_with_mass.geodesic_deviation >= out_no_mass.geodesic_deviation);
}

// ============================================================
// 10. Lorentz boost: Q_current changes on BH transition
// ============================================================

static void test_lorentz_boost_on_transition() {
    section("Lorentz boost: Q_current shifts on BH activation");

    QuatNav nav_no_boost, nav_boost;

    double price = 50000.0;
    // Warm up both identically for 10 bars (bh_active = false throughout)
    for (int i = 0; i < 10; ++i) {
        price *= 1.002;
        nav_no_boost.update(price, 1000.0, bar_ts(i), 1.5, false, false);
        nav_boost.update(price, 1000.0, bar_ts(i), 1.5, false, false);
    }

    // Bar 11: nav_boost sees a BH activation (was_active=false, active=true)
    price *= 1.003;
    auto out_no_boost = nav_no_boost.update(price, 1500.0, bar_ts(10), 2.0, false, false);
    auto out_boost    = nav_boost.update(price, 1500.0, bar_ts(10), 2.0, false, true);

    // Boost should have fired
    CHECK(out_boost.lorentz_boost_applied);
    CHECK(!out_no_boost.lorentz_boost_applied);
    CHECK(out_boost.lorentz_boost_rapidity > 0.0);

    // Q_current should differ between the two navigators
    double Q_no[4] = {out_no_boost.qw, out_no_boost.qx,
                      out_no_boost.qy, out_no_boost.qz};
    double Q_bst[4] = {out_boost.qw, out_boost.qx,
                       out_boost.qy, out_boost.qz};
    double diff_angle = QuatNav::quat_geodesic_angle(Q_no, Q_bst);
    CHECK(diff_angle > 1e-6);

    // Both orientations must be unit norm
    CHECK_CLOSE(qnorm(Q_no),  1.0, 1e-12);
    CHECK_CLOSE(qnorm(Q_bst), 1.0, 1e-12);

    std::printf("  Orientation divergence after boost: %.6f rad\n", diff_angle);
}

// ============================================================
// 11. Serialisation round-trip: QuatNavOutput fields are finite
// ============================================================

static void test_output_finite_after_stress() {
    section("QuatNavOutput: all fields are finite after 1000 stress bars");

    QuatNav nav;
    double price = 1.0;  // start tiny to stress normaliser

    for (int i = 0; i < 1000; ++i) {
        // Stress: huge price jump, alternating, extreme volume
        if (i == 200) price = 1e8;   // shock
        if (i == 400) price = 0.01;  // extreme drop
        price *= (i % 2 == 0 ? 1.03 : 0.971);
        double vol = (i % 50 == 0) ? 1e9 : 100.0 + i;
        bool was_active = (i % 30 >= 10 && i % 30 < 20);
        bool now_active = (i % 30 >= 11 && i % 30 < 21);

        auto out = nav.update(price, vol, bar_ts(i), 2.5 * was_active,
                              was_active, now_active);

        // All output fields must be finite
        CHECK(std::isfinite(out.bar_qw));
        CHECK(std::isfinite(out.bar_qx));
        CHECK(std::isfinite(out.bar_qy));
        CHECK(std::isfinite(out.bar_qz));
        CHECK(std::isfinite(out.qw));
        CHECK(std::isfinite(out.qx));
        CHECK(std::isfinite(out.qy));
        CHECK(std::isfinite(out.qz));
        CHECK(std::isfinite(out.angular_velocity));
        CHECK(std::isfinite(out.geodesic_deviation));
        CHECK(out.angular_velocity  >= 0.0);
        CHECK(out.geodesic_deviation >= 0.0);

        // Unit norms enforced
        double q_b[4] = {out.bar_qw, out.bar_qx, out.bar_qy, out.bar_qz};
        double q_Q[4] = {out.qw, out.qx, out.qy, out.qz};
        CHECK_CLOSE(qnorm(q_b), 1.0, 1e-11);
        CHECK_CLOSE(qnorm(q_Q), 1.0, 1e-11);
    }
}

// ============================================================
// 12. Quaternion multiply: non-commutativity check
// ============================================================

static void test_quat_mul_noncommutative() {
    section("quat_mul: non-commutativity (q1*q2 != q2*q1 in general)");

    double q1[4] = {0.6, 0.5, 0.4, 0.5};
    double q2[4] = {0.3, 0.7, 0.2, 0.6};
    QuatNav::quat_normalize(q1);
    QuatNav::quat_normalize(q2);

    double r1[4], r2[4];
    QuatNav::quat_mul(q1, q2, r1);
    QuatNav::quat_mul(q2, q1, r2);

    // Results should be different (general case)
    double diff = QuatNav::quat_geodesic_angle(r1, r2);
    CHECK(diff > 1e-6);
}

// ============================================================
// 13. SignalOutput integration: nav fields present
// ============================================================

static void test_signal_output_nav_fields_present() {
    section("SignalOutput: nav fields are present and zero-initialised");
    SignalOutput so;
    CHECK(so.nav_qw == 0.0);
    CHECK(so.nav_qx == 0.0);
    CHECK(so.nav_qy == 0.0);
    CHECK(so.nav_qz == 0.0);
    CHECK(so.nav_angular_vel  == 0.0);
    CHECK(so.nav_geodesic_dev == 0.0);
}

// ============================================================
// Main
// ============================================================

int main() {
    std::printf("=== QuatNav Tests ===\n\n");

    test_quat_mul_identity();
    test_quat_normalize_unit_norm();
    test_slerp_endpoints();
    test_slerp_midpoint_equidistant();
    test_geodesic_angle_identical();
    test_geodesic_angle_orthogonal();
    test_extrapolate_collinear();
    test_qutnav_normalisation_preserved();
    test_qutnav_reset();
    test_angular_velocity_volatile_vs_stable();
    test_geodesic_deviation_curvature_correction();
    test_lorentz_boost_on_transition();
    test_output_finite_after_stress();
    test_quat_mul_noncommutative();
    test_signal_output_nav_fields_present();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
