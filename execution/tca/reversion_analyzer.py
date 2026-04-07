# execution/tca/reversion_analyzer.py -- Post-trade price reversion analysis for SRFM
# Measures how much price impact reverses after trade completion.

from __future__ import annotations

import math
import sqlite3
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _ols(X: np.ndarray, y: np.ndarray):
    """Minimal OLS: returns (beta, rmse, r2)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    res = y - y_hat
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    rmse = math.sqrt(ss_res / max(len(y), 1))
    return beta, rmse, r2

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ReversionProfile:
    """
    Post-trade price reversion profile at multiple time horizons.
    All price changes expressed in basis points relative to fill price.
    """
    horizons: List[int]            # bars after fill (e.g. [1, 5, 15, 30, 60])
    reversion_bps: List[float]     # price change from fill at each horizon (sign: negative = reversion for buy)
    half_life_bars: float          # exponential decay half-life in bars
    permanent_impact: float        # asymptote of reversion curve (bps) -- non-reverting portion
    temporary_impact: float        # initial_impact - permanent_impact (bps) -- reverting portion
    t_stat: float                  # t-stat for significance of mean reversion
    fit_quality: float             # R-squared of exponential fit (0..1)
    n_obs: int                     # number of post-trade prices used
    symbol: str = ""
    side: str = ""
    trade_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Levenberg-Marquardt style nonlinear least squares (manual implementation)
# ---------------------------------------------------------------------------

def _exp_model(t: float, a: float, b: float, c: float) -> float:
    """Exponential decay: y = a * exp(-b * t) + c"""
    b_safe = max(b, 1e-9)
    return a * math.exp(-b_safe * t) + c


def _exp_jacobian(t: float, a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Partial derivatives of a*exp(-b*t)+c with respect to a, b, c."""
    b_safe = max(b, 1e-9)
    e = math.exp(-b_safe * t)
    da = e
    db = -a * t * e
    dc = 1.0
    return da, db, dc


def _fit_exponential_nls(
    t_vals: List[float],
    y_vals: List[float],
    max_iter: int = 200,
    lam: float = 1e-2,
    lam_factor: float = 10.0,
    tol: float = 1e-8,
) -> Tuple[float, float, float, float]:
    """
    Fit y = a * exp(-b * t) + c using Levenberg-Marquardt (simplified manual impl).

    Returns (a, b, c, r_squared).
    Initial guess: a = y[0] - y[-1], b = 1/len(y), c = y[-1].
    """
    n = len(t_vals)
    if n < 3:
        return 0.0, 0.0, float(sum(y_vals) / max(n, 1)), 0.0

    y0 = y_vals[0]
    y_end = y_vals[-1]
    a = y0 - y_end
    b = 1.0 / max(len(t_vals), 1)
    c = y_end

    def residuals(a_: float, b_: float, c_: float) -> List[float]:
        return [_exp_model(t, a_, b_, c_) - y for t, y in zip(t_vals, y_vals)]

    def sse(a_: float, b_: float, c_: float) -> float:
        return sum(r * r for r in residuals(a_, b_, c_))

    prev_sse = sse(a, b, c)

    for _ in range(max_iter):
        # Build Jacobian matrix J (n x 3) and residual vector r (n,)
        J = []
        r_vec = []
        for t, y in zip(t_vals, y_vals):
            da, db, dc = _exp_jacobian(t, a, b, c)
            J.append([da, db, dc])
            r_vec.append(_exp_model(t, a, b, c) - y)

        # J'J and J'r
        JtJ = [[0.0] * 3 for _ in range(3)]
        Jtr = [0.0] * 3
        for row_j, res_i in zip(J, r_vec):
            for ii in range(3):
                Jtr[ii] += row_j[ii] * res_i
                for jj in range(3):
                    JtJ[ii][jj] += row_j[ii] * row_j[jj]

        # Add damping: (J'J + lam * diag(J'J)) * delta = -J'r
        for ii in range(3):
            JtJ[ii][ii] += lam * (JtJ[ii][ii] + 1e-9)

        # Solve 3x3 system via Cramer's rule / numpy-free Gaussian elimination
        delta = _solve_3x3(JtJ, [-x for x in Jtr])
        if delta is None:
            break

        a_new = a + delta[0]
        b_new = max(b + delta[1], 1e-9)   # b must stay positive
        c_new = c + delta[2]

        new_sse = sse(a_new, b_new, c_new)
        if new_sse < prev_sse:
            a, b, c = a_new, b_new, c_new
            lam /= lam_factor
            if abs(prev_sse - new_sse) < tol:
                break
            prev_sse = new_sse
        else:
            lam *= lam_factor

    # Compute R-squared
    y_mean = sum(y_vals) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    ss_res = sum((_exp_model(t, a, b, c) - y) ** 2 for t, y in zip(t_vals, y_vals))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return a, b, c, max(r2, 0.0)


def _solve_3x3(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """
    Solve 3x3 linear system Ax = b using Gaussian elimination with partial pivoting.
    Returns None if matrix is singular.
    """
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(3):
        # Partial pivot
        max_row = col
        max_val = abs(M[col][col])
        for row in range(col + 1, 3):
            if abs(M[row][col]) > max_val:
                max_val = abs(M[row][col])
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]

        pivot = M[col][col]
        if abs(pivot) < 1e-15:
            return None

        for row in range(col + 1, 3):
            factor = M[row][col] / pivot
            for k in range(col, 4):
                M[row][k] -= factor * M[col][k]

    # Back substitution
    x = [0.0] * 3
    for i in range(2, -1, -1):
        x[i] = M[i][3]
        for j in range(i + 1, 3):
            x[i] -= M[i][j] * x[j]
        pivot = M[i][i]
        if abs(pivot) < 1e-15:
            return None
        x[i] /= pivot

    return x


# ---------------------------------------------------------------------------
# Reversion analyzer
# ---------------------------------------------------------------------------

class ReversionAnalyzer:
    """
    Measures post-trade price reversion to estimate temporary vs permanent impact.
    """

    def __init__(self, min_observations: int = 3) -> None:
        self.min_observations = min_observations

    def analyze(
        self,
        trade,
        post_trade_prices: List[float],
        horizons: List[int],
    ) -> ReversionProfile:
        """
        Compute the reversion profile for a single trade.

        Parameters
        ----------
        trade             : TradeRecord with fill_price and side
        post_trade_prices : list of prices at each bar after fill
                            (len >= max(horizons))
        horizons          : bar offsets to sample (e.g. [1, 5, 15, 30])

        Returns
        -------
        ReversionProfile with exponential fit parameters
        """
        fill_price = trade.fill_price
        side = trade.side.upper()
        # sign convention: for a BUY, positive IS = bad; reversion back down = negative bps
        sign = 1.0 if side == "BUY" else -1.0

        if fill_price <= 0.0 or len(post_trade_prices) < self.min_observations:
            return ReversionProfile(
                horizons=horizons,
                reversion_bps=[0.0] * len(horizons),
                half_life_bars=float("inf"),
                permanent_impact=0.0,
                temporary_impact=0.0,
                t_stat=0.0,
                fit_quality=0.0,
                n_obs=len(post_trade_prices),
                symbol=getattr(trade, "symbol", ""),
                side=side,
                trade_id=getattr(trade, "trade_id", None),
            )

        # Sample prices at each horizon
        reversion_bps: List[float] = []
        valid_horizons: List[int] = []
        for h in horizons:
            idx = h - 1  # 0-indexed
            if idx < len(post_trade_prices):
                price_h = post_trade_prices[idx]
                if price_h > 0.0:
                    # Reversion for buyer: price going down = impact reverting
                    change_bps = sign * (price_h - fill_price) / fill_price * 10_000.0
                    reversion_bps.append(change_bps)
                    valid_horizons.append(h)
                else:
                    reversion_bps.append(0.0)
                    valid_horizons.append(h)
            else:
                reversion_bps.append(0.0)
                valid_horizons.append(h)

        # Fit exponential decay to reversion curve
        t_vals = [float(h) for h in valid_horizons]
        y_vals = reversion_bps[:]

        a, b, c, r2 = _fit_exponential_nls(t_vals, y_vals)

        # half-life: t such that a*exp(-b*t) = a/2 => t = ln(2)/b
        if b > 1e-9:
            half_life = math.log(2.0) / b
        else:
            half_life = float("inf")

        # permanent_impact = c (asymptote); temporary = a (decaying portion)
        permanent_impact = c
        temporary_impact = a

        # t-statistic for significance of mean reversion
        t_stat = self._compute_t_stat(reversion_bps)

        return ReversionProfile(
            horizons=valid_horizons,
            reversion_bps=reversion_bps,
            half_life_bars=half_life,
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            t_stat=t_stat,
            fit_quality=r2,
            n_obs=len(post_trade_prices),
            symbol=getattr(trade, "symbol", ""),
            side=side,
            trade_id=getattr(trade, "trade_id", None),
        )

    @staticmethod
    def _compute_t_stat(values: List[float]) -> float:
        """
        One-sample t-statistic testing whether mean != 0.
        Returns 0.0 if insufficient data or zero variance.
        """
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        if variance <= 0.0:
            return 0.0
        se = math.sqrt(variance / n)
        return mean / se if se > 0.0 else 0.0

    def aggregate_reversion(
        self,
        profiles: List[ReversionProfile],
    ) -> Dict[str, float]:
        """
        Aggregate multiple ReversionProfiles to compute mean reversion statistics.

        Returns dict with keys: mean_half_life, mean_permanent_impact,
        mean_temporary_impact, mean_t_stat, pct_significant (|t|>2), n_profiles.
        """
        if not profiles:
            return {
                "mean_half_life": 0.0,
                "mean_permanent_impact": 0.0,
                "mean_temporary_impact": 0.0,
                "mean_t_stat": 0.0,
                "pct_significant": 0.0,
                "n_profiles": 0,
            }

        finite_hl = [p.half_life_bars for p in profiles if math.isfinite(p.half_life_bars)]
        mean_hl = sum(finite_hl) / len(finite_hl) if finite_hl else float("inf")
        n = len(profiles)

        return {
            "mean_half_life": mean_hl,
            "mean_permanent_impact": sum(p.permanent_impact for p in profiles) / n,
            "mean_temporary_impact": sum(p.temporary_impact for p in profiles) / n,
            "mean_t_stat": sum(p.t_stat for p in profiles) / n,
            "pct_significant": sum(1 for p in profiles if abs(p.t_stat) > 2.0) / n,
            "n_profiles": n,
        }

    def fit(
        self,
        times: List[float],
        impacts: List[float],
    ) -> Dict[str, float]:
        """
        Fit an exponential decay model: impact(t) = A * exp(-lambda * t) + C.
        Returns dict with keys: A, lambda, C, half_life, r_squared.
        Raises ValueError if fewer than 2 data points.
        """
        if len(times) < 2 or len(impacts) < 2:
            raise ValueError("Need at least 2 data points for exponential fit")
        n = min(len(times), len(impacts))
        t = np.array(times[:n], dtype=float)
        y = np.array(impacts[:n], dtype=float)
        # Linearise: log(y - C) ~ log(A) - lambda * t
        # Use simple OLS on log(|y|) as approximation (C=0)
        safe_y = np.maximum(np.abs(y), 1e-12)
        log_y = np.log(safe_y)
        X = np.column_stack([t, np.ones(n)])
        beta, _, _ = _ols(X, log_y)
        lam = float(-beta[0])
        A = float(math.exp(beta[1]))
        half_life = math.log(2) / max(lam, 1e-9)
        y_pred = A * np.exp(-lam * t)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        return {"A": A, "lambda": lam, "C": 0.0, "half_life": half_life, "r_squared": r2}


# ---------------------------------------------------------------------------
# SQLite persistence for reversion profiles
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS reversion_profiles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id      TEXT,
    symbol        TEXT NOT NULL,
    side          TEXT,
    horizons_json TEXT,
    reversion_json TEXT,
    half_life     REAL,
    permanent_impact REAL,
    temporary_impact REAL,
    t_stat        REAL,
    fit_quality   REAL,
    n_obs         INTEGER,
    created_at    TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_IDX_SYMBOL = """
CREATE INDEX IF NOT EXISTS idx_rev_symbol ON reversion_profiles (symbol);
"""

_CREATE_IDX_TRADE = """
CREATE INDEX IF NOT EXISTS idx_rev_trade ON reversion_profiles (trade_id);
"""


class ReversionDatabase:
    """
    SQLite-backed store for ReversionProfile objects.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        # Keep a persistent connection for :memory: databases to prevent data loss
        self._persistent_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_IDX_SYMBOL)
        conn.execute(_CREATE_IDX_TRADE)
        conn.commit()

    def insert(self, trade_id: str, profile: ReversionProfile) -> None:
        """Insert a ReversionProfile into the database."""
        import json
        horizons_json = json.dumps(profile.horizons)
        reversion_json = json.dumps(profile.reversion_bps)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reversion_profiles
                    (trade_id, symbol, side, horizons_json, reversion_json,
                     half_life, permanent_impact, temporary_impact,
                     t_stat, fit_quality, n_obs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    profile.symbol,
                    profile.side,
                    horizons_json,
                    reversion_json,
                    profile.half_life_bars if math.isfinite(profile.half_life_bars) else -1.0,
                    profile.permanent_impact,
                    profile.temporary_impact,
                    profile.t_stat,
                    profile.fit_quality,
                    profile.n_obs,
                ),
            )
            conn.commit()

    def get_by_symbol(self, symbol: str, n: int = 50) -> List[ReversionProfile]:
        """Retrieve the most recent n ReversionProfiles for a symbol."""
        import json
        rows = []
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM reversion_profiles
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (symbol, n),
            )
            rows = cursor.fetchall()

        profiles = []
        for row in rows:
            horizons = json.loads(row["horizons_json"] or "[]")
            reversion_bps = json.loads(row["reversion_json"] or "[]")
            hl = row["half_life"]
            profiles.append(ReversionProfile(
                horizons=horizons,
                reversion_bps=reversion_bps,
                half_life_bars=hl if hl >= 0.0 else float("inf"),
                permanent_impact=row["permanent_impact"],
                temporary_impact=row["temporary_impact"],
                t_stat=row["t_stat"],
                fit_quality=row["fit_quality"],
                n_obs=row["n_obs"],
                symbol=row["symbol"],
                side=row["side"] or "",
                trade_id=row["trade_id"],
            ))
        return profiles

    def aggregate_stats(self, symbol: str) -> Dict:
        """
        Return aggregated reversion stats for a symbol.
        """
        profiles = self.get_by_symbol(symbol, n=500)
        if not profiles:
            return {
                "symbol": symbol,
                "n_profiles": 0,
                "mean_half_life": None,
                "mean_permanent_impact": None,
                "mean_temporary_impact": None,
                "pct_significant": None,
            }

        analyzer = ReversionAnalyzer()
        stats = analyzer.aggregate_reversion(profiles)
        stats["symbol"] = symbol
        return stats

    def count(self) -> int:
        """Return total number of stored profiles."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM reversion_profiles"
            ).fetchone()
            return row["cnt"]

    def delete_old(self, keep_days: int = 90) -> int:
        """Delete profiles older than keep_days, return rows deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM reversion_profiles
                WHERE created_at < datetime('now', ?)
                """,
                (f"-{keep_days} days",),
            )
            conn.commit()
            return cursor.rowcount
