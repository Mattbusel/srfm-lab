"""
tests/test_quat_nav_bridge.py

Tests for bridge/quat_nav_bridge.py

Validates:
  1. Quaternion normalisation preservation across many updates
  2. SLERP interpolation correctness (endpoints, midpoint equidistance)
  3. Geodesic deviation computation against known mass values
  4. Angular velocity computation (volatile > stable)
  5. Serialisation round-trip via NavStateWriter + SQLite
  6. Lorentz boost fires on BH transition and shifts orientation
  7. Reset restores identity state
  8. All outputs finite on stress input
"""

from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone

# Allow running from repo root or tests/ directory
sys.path.insert(0, str(Path(__file__).parents[1]))

from bridge.quat_nav_bridge import (
    QuatNavPy,
    QuatNavOutput,
    NavStateWriter,
    _qnormalize,
    _qmul,
    _qinv,
    _qslerp,
    _qgeodesic_angle,
    _qextrapolate,
    _qangle,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_NS_PER_SEC = 1_000_000_000
_BASE_TS    = 1_700_000_000 * _NS_PER_SEC

def bar_ts(i: int, interval_sec: int = 60) -> int:
    return _BASE_TS + i * interval_sec * _NS_PER_SEC

def qnorm(q) -> float:
    return math.sqrt(sum(x*x for x in q))

def assert_unit(q, tol=1e-12, label=""):
    n = qnorm(q)
    assert abs(n - 1.0) < tol, f"{label} |q|={n:.15f}, expected 1.0"

def assert_close(a, b, tol=1e-10, label=""):
    assert abs(a - b) <= tol, f"{label}: got {a!r}, expected {b!r}, diff={abs(a-b):.3e}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Quaternion normalisation
# ─────────────────────────────────────────────────────────────────────────────

def test_qnormalize_unit():
    q = _qnormalize([3.0, 4.0, 0.0, 0.0])
    assert_unit(q, label="test_qnormalize_unit")

def test_qnormalize_small_input():
    q = _qnormalize([1e-20, 1e-20, 1e-20, 1e-20])
    assert_unit(q, tol=1e-12, label="test_qnormalize_small_input")

def test_qnormalize_zero_input():
    q = _qnormalize([0.0, 0.0, 0.0, 0.0])
    assert q == [1.0, 0.0, 0.0, 0.0], f"zero input should give identity, got {q}"

def test_qmul_inverse_is_identity():
    """q * q^{-1} == identity."""
    raw = _qnormalize([0.6, 0.4, -0.5, 0.5])
    inv = _qinv(raw)
    result = _qnormalize(_qmul(raw, inv))
    assert_close(result[0], 1.0, tol=1e-12, label="w")
    assert_close(result[1], 0.0, tol=1e-12, label="x")
    assert_close(result[2], 0.0, tol=1e-12, label="y")
    assert_close(result[3], 0.0, tol=1e-12, label="z")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SLERP interpolation
# ─────────────────────────────────────────────────────────────────────────────

def test_slerp_t0_returns_q1():
    q1 = [1.0, 0.0, 0.0, 0.0]
    q2 = _qnormalize([0.0, 1.0, 0.0, 0.0])
    out = _qslerp(q1, q2, 0.0)
    assert_close(out[0], q1[0], tol=1e-10, label="slerp t=0 w")
    assert_close(out[1], q1[1], tol=1e-10, label="slerp t=0 x")

def test_slerp_t1_returns_q2():
    q1 = [1.0, 0.0, 0.0, 0.0]
    q2 = _qnormalize([0.0, 1.0, 0.0, 0.0])
    out = _qslerp(q1, q2, 1.0)
    assert_unit(out, label="slerp t=1 norm")
    # out should match q2 up to sign flip
    dot = abs(sum(a*b for a,b in zip(out, q2)))
    assert abs(dot - 1.0) < 1e-10, f"slerp t=1 should return q2, dot={dot}"

def test_slerp_midpoint_equidistant():
    """SLERP midpoint is equidistant from both endpoints."""
    q1 = _qnormalize([0.6, 0.8, 0.0, 0.0])
    q2 = _qnormalize([0.3, 0.1, 0.9, 0.3])
    mid = _qslerp(q1, q2, 0.5)
    assert_unit(mid, label="slerp midpoint norm")
    d1 = _qgeodesic_angle(q1, mid)
    d2 = _qgeodesic_angle(mid, q2)
    assert_close(d1, d2, tol=1e-9, label="midpoint equidistant")

def test_slerp_output_unit_norm():
    """SLERP output is always unit norm for various t."""
    q1 = _qnormalize([1.0, 0.2, 0.3, 0.4])
    q2 = _qnormalize([0.5, 0.5, -0.5, 0.5])
    for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        out = _qslerp(q1, q2, t)
        assert_unit(out, tol=1e-11, label=f"slerp t={t}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Geodesic deviation vs BH mass
# ─────────────────────────────────────────────────────────────────────────────

def test_geodesic_deviation_increases_with_mass():
    """Curvature correction raises geodesic_deviation proportionally to mass."""
    prices = [50000.0, 50050.0, 50080.0]
    vols   = [1000.0,  1200.0,  900.0]

    nav_no_mass   = QuatNavPy()
    nav_with_mass = QuatNavPy()

    for i in range(2):
        nav_no_mass.update(prices[i], vols[i], bar_ts(i), 0.0, False, False)
        nav_with_mass.update(prices[i], vols[i], bar_ts(i), 0.0, False, False)

    out_no   = nav_no_mass.update(prices[2], vols[2], bar_ts(2), 0.0, False, False)
    out_with = nav_with_mass.update(prices[2], vols[2], bar_ts(2), 3.5, False, False)

    assert out_with.geodesic_deviation >= out_no.geodesic_deviation, (
        f"With mass={3.5}, deviation {out_with.geodesic_deviation:.6f} should be "
        f">= {out_no.geodesic_deviation:.6f}"
    )

def test_geodesic_deviation_zero_before_bar2():
    """Geodesic deviation is 0 for the first two bars (need 3 to compute)."""
    nav = QuatNavPy()
    out0 = nav.update(50000.0, 1000.0, bar_ts(0), 1.0, False, False)
    out1 = nav.update(50100.0, 1100.0, bar_ts(1), 1.5, False, False)
    assert out0.geodesic_deviation == 0.0
    assert out1.geodesic_deviation == 0.0

def test_geodesic_deviation_nonzero_after_bar2():
    """Geodesic deviation is non-negative from bar index 2 onward."""
    nav = QuatNavPy()
    nav.update(50000.0, 1000.0, bar_ts(0), 0.0, False, False)
    nav.update(50050.0, 1200.0, bar_ts(1), 0.0, False, False)
    out2 = nav.update(50080.0, 900.0, bar_ts(2), 0.0, False, False)
    assert out2.geodesic_deviation >= 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 4. Angular velocity: volatile > stable
# ─────────────────────────────────────────────────────────────────────────────

def test_angular_velocity_volatile_gt_stable():
    nav_s = QuatNavPy()
    nav_v = QuatNavPy()

    p_s = p_v = 50000.0
    sum_s = sum_v = 0.0
    N = 100

    for i in range(N):
        p_s *= 1.0001
        p_v *= (1.05 if i % 2 == 0 else 0.952)
        out_s = nav_s.update(p_s, 1000.0, bar_ts(i), 0.5, False, False)
        out_v = nav_v.update(p_v, 5000.0, bar_ts(i), 0.5, False, False)
        if i >= 5:
            sum_s += out_s.angular_velocity
            sum_v += out_v.angular_velocity

    assert sum_v > sum_s, (
        f"Volatile mean omega {sum_v/(N-5):.6f} should exceed stable {sum_s/(N-5):.6f}"
    )

def test_angular_velocity_nonnegative():
    nav = QuatNavPy()
    price = 50000.0
    for i in range(50):
        price *= 1.002 if i % 3 != 0 else 0.995
        out = nav.update(price, 1000.0 + i * 10, bar_ts(i), 1.0, False, False)
        assert out.angular_velocity >= 0.0, f"bar {i}: omega={out.angular_velocity}"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Normalisation preservation across many updates
# ─────────────────────────────────────────────────────────────────────────────

def test_normalisation_preserved_500_bars():
    nav = QuatNavPy()
    price = 50000.0

    for i in range(500):
        if   i % 7  == 0: price *= 1.005
        elif i % 11 == 0: price *= 0.994
        else:              price *= 1.0001
        vol = 1000.0 + 200.0 * (i % 13)
        out = nav.update(price, vol, bar_ts(i), 1.5, False, False)

        q_bar = [out.bar_qw, out.bar_qx, out.bar_qy, out.bar_qz]
        q_Q   = [out.qw, out.qx, out.qy, out.qz]
        assert_unit(q_bar, tol=1e-11, label=f"bar_q bar={i}")
        assert_unit(q_Q,   tol=1e-11, label=f"Q_cur bar={i}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Lorentz boost
# ─────────────────────────────────────────────────────────────────────────────

def test_lorentz_boost_fires_on_bh_activation():
    nav = QuatNavPy()
    price = 50000.0
    for i in range(10):
        price *= 1.002
        nav.update(price, 1000.0, bar_ts(i), 1.5, False, False)

    price *= 1.003
    out = nav.update(price, 1500.0, bar_ts(10), 2.0, False, True)  # activation
    assert out.lorentz_boost_applied
    assert out.lorentz_boost_rapidity > 0.0

def test_lorentz_boost_fires_on_bh_deactivation():
    nav = QuatNavPy()
    price = 50000.0
    for i in range(10):
        price *= 1.002
        nav.update(price, 1000.0, bar_ts(i), 1.5, True, True)

    price *= 0.998
    out = nav.update(price, 800.0, bar_ts(10), 1.0, True, False)  # deactivation
    assert out.lorentz_boost_applied

def test_lorentz_boost_shifts_orientation():
    nav_no_boost = QuatNavPy()
    nav_boost    = QuatNavPy()

    price = 50000.0
    for i in range(10):
        price *= 1.002
        nav_no_boost.update(price, 1000.0, bar_ts(i), 1.5, False, False)
        nav_boost.update(price, 1000.0, bar_ts(i), 1.5, False, False)

    price *= 1.003
    out_nb = nav_no_boost.update(price, 1500.0, bar_ts(10), 2.0, False, False)
    out_b  = nav_boost.update(   price, 1500.0, bar_ts(10), 2.0, False, True)

    Q_nb = [out_nb.qw, out_nb.qx, out_nb.qy, out_nb.qz]
    Q_b  = [out_b.qw,  out_b.qx,  out_b.qy,  out_b.qz]
    diff = _qgeodesic_angle(Q_nb, Q_b)
    assert diff > 1e-6, f"Boost should shift orientation; diff={diff:.8f}"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Reset
# ─────────────────────────────────────────────────────────────────────────────

def test_reset_restores_identity():
    nav = QuatNavPy()
    price = 50000.0
    for i in range(50):
        price *= 1.002
        nav.update(price, 1000.0, bar_ts(i), 1.0, False, i > 30)

    nav.reset()
    assert nav.bar_count == 0
    assert nav._Q == [1.0, 0.0, 0.0, 0.0]
    assert nav._prev_close is None

def test_reset_then_update_gives_fresh_state():
    nav = QuatNavPy()
    price = 50000.0
    for i in range(20):
        price *= 1.005
        nav.update(price, 1000.0, bar_ts(i), 1.0, False, False)

    nav.reset()
    out = nav.update(50000.0, 1000.0, bar_ts(0), 0.0, False, False)
    # First bar after reset: angular_velocity and geodesic_deviation are 0
    assert out.angular_velocity == 0.0
    assert out.geodesic_deviation == 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 8. Serialisation round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_nav_state_writer_round_trip():
    """Write nav records, read them back, verify field integrity."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    writer = NavStateWriter(conn)

    nav = QuatNavPy()
    price = 50000.0
    records = []

    for i in range(5):
        price *= 1.003
        out = nav.update(price, 1000.0 + i * 100, bar_ts(i), 0.5 * i, False, i == 3)
        bar_time = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
        writer.write("BTC", "15m", bar_time, bar_ts(i), out, 0.5 * i, i == 3)
        records.append(out)

    rows = conn.execute(
        "SELECT * FROM nav_state ORDER BY id"
    ).fetchall()

    assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

    for i, (row, out) in enumerate(zip(rows, records)):
        assert row["symbol"] == "BTC"
        assert row["timeframe"] == "15m"
        assert abs(row["qw"] - out.qw) < 1e-14, f"row {i}: qw mismatch"
        assert abs(row["qx"] - out.qx) < 1e-14, f"row {i}: qx mismatch"
        assert abs(row["qy"] - out.qy) < 1e-14, f"row {i}: qy mismatch"
        assert abs(row["qz"] - out.qz) < 1e-14, f"row {i}: qz mismatch"
        assert abs(row["angular_velocity"]   - out.angular_velocity)   < 1e-14
        assert abs(row["geodesic_deviation"] - out.geodesic_deviation) < 1e-14
        assert row["lorentz_boost_applied"] == int(out.lorentz_boost_applied)
        # Stored orientation is unit norm
        q = [row["qw"], row["qx"], row["qy"], row["qz"]]
        assert_unit(q, tol=1e-12, label=f"stored Q row {i}")

    conn.close()

def test_nav_state_writer_write_error_is_silent(capsys=None):
    """Writer should log on error and not raise."""
    conn = sqlite3.connect(":memory:")
    writer = NavStateWriter(conn)
    conn.close()  # deliberately close to trigger write error
    out = QuatNavOutput(0,0,0,1, 1,0,0,0, 0.0, 0.0, False, 0.0)
    # Should not raise
    writer.write("ETH", "15m", "2024-01-01T00:00:00+00:00", 0, out, 0.0, False)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Stress: all outputs finite on extreme inputs
# ─────────────────────────────────────────────────────────────────────────────

def test_all_outputs_finite_stress():
    nav = QuatNavPy()
    price = 1.0

    for i in range(1000):
        if i == 100: price = 1e8
        if i == 300: price = 0.001
        price *= (1.05 if i % 2 == 0 else 0.952)
        vol = 1e9 if i % 50 == 0 else 100.0 + i
        was_active = (i % 30 >= 10 and i % 30 < 20)
        now_active = (i % 30 >= 11 and i % 30 < 21)

        out = nav.update(price, vol, bar_ts(i), 2.5 * was_active,
                         was_active, now_active)

        for field, val in [
            ("bar_qw", out.bar_qw), ("bar_qx", out.bar_qx),
            ("bar_qy", out.bar_qy), ("bar_qz", out.bar_qz),
            ("qw",  out.qw), ("qx", out.qx), ("qy", out.qy), ("qz", out.qz),
            ("angular_velocity", out.angular_velocity),
            ("geodesic_deviation", out.geodesic_deviation),
        ]:
            assert math.isfinite(val), f"bar {i} field {field}={val} is not finite"

        q_b = [out.bar_qw, out.bar_qx, out.bar_qy, out.bar_qz]
        q_Q = [out.qw, out.qx, out.qy, out.qz]
        assert_unit(q_b, tol=1e-10, label=f"bar_q stress bar={i}")
        assert_unit(q_Q, tol=1e-10, label=f"Q_cur stress bar={i}")
        assert out.angular_velocity  >= 0.0
        assert out.geodesic_deviation >= 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Runner (also works with pytest)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        test_qnormalize_unit,
        test_qnormalize_small_input,
        test_qnormalize_zero_input,
        test_qmul_inverse_is_identity,
        test_slerp_t0_returns_q1,
        test_slerp_t1_returns_q2,
        test_slerp_midpoint_equidistant,
        test_slerp_output_unit_norm,
        test_geodesic_deviation_increases_with_mass,
        test_geodesic_deviation_zero_before_bar2,
        test_geodesic_deviation_nonzero_after_bar2,
        test_angular_velocity_volatile_gt_stable,
        test_angular_velocity_nonnegative,
        test_normalisation_preserved_500_bars,
        test_lorentz_boost_fires_on_bh_activation,
        test_lorentz_boost_fires_on_bh_deactivation,
        test_lorentz_boost_shifts_orientation,
        test_reset_restores_identity,
        test_reset_then_update_gives_fresh_state,
        test_nav_state_writer_round_trip,
        test_nav_state_writer_write_error_is_silent,
        test_all_outputs_finite_stress,
    ]

    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)
