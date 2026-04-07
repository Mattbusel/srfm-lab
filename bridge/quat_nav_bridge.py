"""
bridge/quat_nav_bridge.py

Quaternion navigation layer — pure Python implementation mirroring
cpp/signal-engine/src/quaternion/quat_nav.cpp.

Provides:
  QuatNavPy      — per-instrument navigation state machine
  NavStateWriter — writes QuatNavOutput to the live_trades.db nav_state table

Integration rules (LARSA v17):
  - Read-only observability: no nav signal gates entry/exit logic yet.
  - nav_state table is written every 15m bar; consumers may query it.
  - Normalisation invariant enforced: |Q_current| == 1.0 after every update.
"""

from __future__ import annotations

import math
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_EPSILON = 1e-10
_NS_PER_SEC = 1_000_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuatNavOutput:
    # Bar quaternion (unit quaternion for this bar's 4-space position)
    bar_qw: float
    bar_qx: float
    bar_qy: float
    bar_qz: float
    # Running orientation Q_current
    qw: float
    qx: float
    qy: float
    qz: float
    # Nav signals
    angular_velocity:   float   # radians per bar
    geodesic_deviation: float   # radians; curvature-corrected by BH mass
    # Lorentz boost metadata
    lorentz_boost_applied:  bool
    lorentz_boost_rapidity: float


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _qnorm(q: tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in q))


def _qnormalize(q: list[float]) -> list[float]:
    n2 = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    if n2 < 1e-30:
        return [1.0, 0.0, 0.0, 0.0]
    inv_n = 1.0 / math.sqrt(n2)
    return [x * inv_n for x in q]


def _qmul(q1: list[float], q2: list[float]) -> list[float]:
    """Hamilton product q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def _qinv(q: list[float]) -> list[float]:
    """Unit quaternion inverse = conjugate."""
    return [q[0], -q[1], -q[2], -q[3]]


def _qangle(q: list[float]) -> float:
    """Rotation angle of a unit quaternion: 2 * arccos(|w|)."""
    w = max(-1.0, min(1.0, q[0]))
    return 2.0 * math.acos(abs(w))


def _qslerp(q1: list[float], q2: list[float], t: float) -> list[float]:
    """SLERP(q1, q2, t). Stable for t in [0, 1]."""
    dot = sum(a * b for a, b in zip(q1, q2))
    q2a = list(q2)
    if dot < 0.0:
        dot = -dot
        q2a = [-x for x in q2a]
    dot = min(dot, 1.0)
    theta = math.acos(dot)
    if theta < 1e-10:
        out = [q1[i] + t * (q2a[i] - q1[i]) for i in range(4)]
    else:
        sin_theta = math.sin(theta)
        s1 = math.sin((1.0 - t) * theta) / sin_theta
        s2 = math.sin(t * theta) / sin_theta
        out = [s1 * q1[i] + s2 * q2a[i] for i in range(4)]
    return _qnormalize(out)


def _qgeodesic_angle(q1: list[float], q2: list[float]) -> float:
    """Geodesic angle (radians) between two unit quaternions."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = min(dot, 1.0)
    return 2.0 * math.acos(dot)


def _qextrapolate(q1: list[float], q2: list[float]) -> list[float]:
    """Predict q3 by applying the q1->q2 rotation one more time to q2."""
    delta = _qmul(q2, _qinv(q1))
    delta = _qnormalize(delta)
    return _qnormalize(_qmul(delta, q2))


# ─────────────────────────────────────────────────────────────────────────────
# QuatNavPy: per-instrument state machine
# ─────────────────────────────────────────────────────────────────────────────

class QuatNavPy:
    """
    Quaternion navigation state machine for one instrument / timeframe.

    Mirrors QuatNav in cpp/signal-engine/src/quaternion/quat_nav.cpp.
    Must produce numerically equivalent outputs for identical input sequences.

    Usage (inside LARSA live trader, after BH update):
        nav = QuatNavPy()
        out = nav.update(close, volume, timestamp_ns, bh_mass,
                         bh_was_active, bh_active)
    """

    # Curvature coefficient: geodesic_deviation *= (1 + K * mass)
    _CURVATURE_K   = 0.15

    # Lorentz rapidity scale: η = atanh(min(0.99, mass * SCALE))
    _LORENTZ_SCALE = 0.40

    # Volume EMA alpha (period ~20)
    _VOL_EMA_ALPHA = 2.0 / 21.0

    def __init__(self) -> None:
        self._norm_price_max: float = 1e-8
        self._norm_vol_max:   float = 1e-8
        self._norm_mi_max:    float = 1e-12
        self._vol_ema:        float = 0.0
        self._prev_close:     Optional[float] = None
        self._prev_ts_ns:     int = 0
        self._norm_dt_ref:    float = 60.0  # seconds; auto-calibrated on bar 1

        # Running orientation quaternion Q_current [w, x, y, z]
        self._Q: list[float] = [1.0, 0.0, 0.0, 0.0]

        # Previous bar quaternion (bar n)
        self._q_prev: list[float] = [1.0, 0.0, 0.0, 0.0]

        # Bar n-1 (for geodesic extrapolation)
        self._q_prev2: list[float] = [1.0, 0.0, 0.0, 0.0]

        self._prev_bh_active: bool = False
        self._count: int = 0

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    @property
    def bar_count(self) -> int:
        return self._count

    # ------------------------------------------------------------------

    def update(
        self,
        close: float,
        volume: float,
        timestamp_ns: int,
        bh_mass: float,
        bh_was_active: bool,
        bh_active: bool,
    ) -> QuatNavOutput:
        """
        Process one bar.  Returns QuatNavOutput with all nav signals.
        Normalisation invariant: |Q_current| == 1.0 on every return.
        """

        # ── 1. Normalised 4-space components ──────────────────────────────

        # dt: normalised time step
        dt_norm = 1.0
        if self._count > 0 and self._prev_ts_ns > 0 and timestamp_ns > self._prev_ts_ns:
            dt_sec = (timestamp_ns - self._prev_ts_ns) / _NS_PER_SEC
            dt_norm = dt_sec / self._norm_dt_ref
            if self._count == 1:
                self._norm_dt_ref = dt_sec  # auto-calibrate on first real interval

        # Price normalisation (rolling max, slow decay)
        if close > self._norm_price_max:
            self._norm_price_max = close
        else:
            self._norm_price_max *= 0.99995
            self._norm_price_max = max(self._norm_price_max, 1e-10)
        price_norm = close / self._norm_price_max

        # Volume normalisation
        if volume > self._norm_vol_max:
            self._norm_vol_max = volume
        else:
            self._norm_vol_max *= 0.9999
            self._norm_vol_max = max(self._norm_vol_max, 1e-10)
        vol_norm = volume / self._norm_vol_max

        # Market impact: |log_return| * sqrt(volume / vol_ema)
        mi_raw = 0.0
        if self._prev_close and self._prev_close > _EPSILON:
            log_ret = abs(math.log(close / self._prev_close))
            self._vol_ema = (
                self._VOL_EMA_ALPHA * volume
                + (1.0 - self._VOL_EMA_ALPHA) * self._vol_ema
            )
            vol_ratio = math.sqrt(volume / self._vol_ema) if self._vol_ema > _EPSILON else 1.0
            mi_raw = log_ret * vol_ratio
        else:
            self._vol_ema = volume

        if mi_raw > self._norm_mi_max:
            self._norm_mi_max = mi_raw
        else:
            self._norm_mi_max *= 0.9998
            self._norm_mi_max = max(self._norm_mi_max, 1e-12)
        mi_norm = mi_raw / self._norm_mi_max if self._norm_mi_max > _EPSILON else 0.0

        # ── 2. Bar quaternion ──────────────────────────────────────────────
        q_bar_raw = [
            dt_norm    + 1e-12,
            price_norm + 1e-12,
            vol_norm   + 1e-12,
            mi_norm    + 1e-12,
        ]
        q_bar = _qnormalize(q_bar_raw)

        # ── 3. Inter-bar rotation & angular velocity ───────────────────────
        angular_velocity = 0.0

        if self._count > 0:
            delta_q = _qnormalize(_qmul(q_bar, _qinv(self._q_prev)))
            angular_velocity = _qangle(delta_q)

            Q_new = _qnormalize(_qmul(delta_q, self._Q))
            self._Q = Q_new

        # ── 4. Lorentz boost on BH regime boundary ─────────────────────────
        lorentz_applied  = False
        lorentz_rapidity = 0.0

        if self._count > 0 and bh_was_active != bh_active:
            rapidity = math.atanh(min(0.99, bh_mass * self._LORENTZ_SCALE))
            half_rap = rapidity * 0.5
            q_boost = _qnormalize([math.cos(half_rap), math.sin(half_rap), 0.0, 0.0])
            self._Q = _qnormalize(_qmul(q_boost, self._Q))
            lorentz_applied  = True
            lorentz_rapidity = rapidity

        # ── 5. Geodesic deviation ──────────────────────────────────────────
        geodesic_deviation = 0.0

        if self._count >= 2:
            q_predicted = _qextrapolate(self._q_prev2, self._q_prev)
            geodesic_deviation = _qgeodesic_angle(q_predicted, q_bar)
            if bh_mass > 0.0:
                geodesic_deviation *= (1.0 + self._CURVATURE_K * bh_mass)

        # ── 6. Advance history ─────────────────────────────────────────────
        self._q_prev2       = list(self._q_prev)
        self._q_prev        = list(q_bar)
        self._prev_close    = close
        self._prev_ts_ns    = timestamp_ns
        self._prev_bh_active = bh_active
        self._count        += 1

        return QuatNavOutput(
            bar_qw=q_bar[0],
            bar_qx=q_bar[1],
            bar_qy=q_bar[2],
            bar_qz=q_bar[3],
            qw=self._Q[0],
            qx=self._Q[1],
            qy=self._Q[2],
            qz=self._Q[3],
            angular_velocity=angular_velocity,
            geodesic_deviation=geodesic_deviation,
            lorentz_boost_applied=lorentz_applied,
            lorentz_boost_rapidity=lorentz_rapidity,
        )


# ─────────────────────────────────────────────────────────────────────────────
# NavStateWriter: persists nav signals to live_trades.db
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_NAV_TABLE = """
CREATE TABLE IF NOT EXISTS nav_state (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol                  TEXT    NOT NULL,
    timeframe               TEXT    NOT NULL DEFAULT '15m',
    bar_time                TEXT    NOT NULL,
    timestamp_ns            INTEGER NOT NULL,
    -- bar quaternion (position in 4-space this bar)
    bar_qw                  REAL    NOT NULL,
    bar_qx                  REAL    NOT NULL,
    bar_qy                  REAL    NOT NULL,
    bar_qz                  REAL    NOT NULL,
    -- running orientation Q_current
    qw                      REAL    NOT NULL,
    qx                      REAL    NOT NULL,
    qy                      REAL    NOT NULL,
    qz                      REAL    NOT NULL,
    -- nav signals
    angular_velocity        REAL    NOT NULL,
    geodesic_deviation      REAL    NOT NULL,
    -- context from BH physics (denormalised for pattern miner queries)
    bh_mass                 REAL    NOT NULL DEFAULT 0.0,
    bh_active               INTEGER NOT NULL DEFAULT 0,
    -- Lorentz boost metadata
    lorentz_boost_applied   INTEGER NOT NULL DEFAULT 0,
    lorentz_boost_rapidity  REAL    NOT NULL DEFAULT 0.0,
    -- strategy version for traceability
    strategy_version        TEXT    NOT NULL DEFAULT 'larsa_v17'
)
"""

_INSERT_NAV = """
INSERT INTO nav_state
    (symbol, timeframe, bar_time, timestamp_ns,
     bar_qw, bar_qx, bar_qy, bar_qz,
     qw, qx, qy, qz,
     angular_velocity, geodesic_deviation,
     bh_mass, bh_active,
     lorentz_boost_applied, lorentz_boost_rapidity,
     strategy_version)
VALUES
    (?,?,?,?,  ?,?,?,?,  ?,?,?,?,  ?,?,  ?,?,  ?,?,  ?)
"""


class NavStateWriter:
    """
    Writes QuatNavOutput records to the nav_state table in live_trades.db.

    Thread safety: not thread-safe.  Call from the same thread that owns
    the db connection (same as the live trader's _on_15m_bar path).

    Usage:
        writer = NavStateWriter(conn)   # pass the live trader's db conn
        writer.write(sym, "15m", bar_time_iso, ts_ns, nav_out,
                     bh_mass, bh_active)
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.execute(_CREATE_NAV_TABLE)
        conn.commit()
        logger.info("NavStateWriter: nav_state table ready.")

    def write(
        self,
        symbol: str,
        timeframe: str,
        bar_time: str,
        timestamp_ns: int,
        out: QuatNavOutput,
        bh_mass: float,
        bh_active: bool,
        strategy_version: str = "larsa_v17",
    ) -> None:
        try:
            self._conn.execute(
                _INSERT_NAV,
                (
                    symbol, timeframe, bar_time, timestamp_ns,
                    out.bar_qw, out.bar_qx, out.bar_qy, out.bar_qz,
                    out.qw, out.qx, out.qy, out.qz,
                    out.angular_velocity, out.geodesic_deviation,
                    bh_mass, int(bh_active),
                    int(out.lorentz_boost_applied), out.lorentz_boost_rapidity,
                    strategy_version,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("NavStateWriter: write failed for %s: %s", symbol, exc)
