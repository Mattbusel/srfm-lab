"""
ShadowComparator
================
Compares shadow strategies against the live strategy equity curve.

Key responsibilities:
  - Rank shadows by alpha (excess return over live)
  - Decompose alpha by signal source (BH, OU, GARCH)
  - Promote shadow strategies that have beaten live for 30+ calendar days
    by creating a hypothesis entry in the DB for the genome

Public API
----------
rank_shadows_by_alpha    — sorted list of (shadow_id, alpha) tuples
alpha_decomposition      — per-signal contribution to total alpha
promote_shadow           — write hypothesis if shadow beats live for 30+ days
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH_ENV = "IDEA_ENGINE_DB"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "db" / "idea_engine.db"

PROMOTION_MIN_DAYS = 30
PROMOTION_MIN_ALPHA = 0.02    # 2 % excess return to qualify


# ---------------------------------------------------------------------------
# ShadowComparator
# ---------------------------------------------------------------------------

class ShadowComparator:
    """
    Compares shadow equity curves to the live strategy and promotes winners.

    Parameters
    ----------
    db_path : str | Path | None
        Path to ``idea_engine.db``.
    promotion_min_days : int
        Minimum number of calendar days a shadow must beat live before
        it can be promoted (default 30).
    promotion_min_alpha : float
        Minimum cumulative alpha required for promotion (default 0.02 = 2 %).
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        promotion_min_days: int = PROMOTION_MIN_DAYS,
        promotion_min_alpha: float = PROMOTION_MIN_ALPHA,
    ) -> None:
        self.db_path = Path(
            db_path or os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
        )
        self.promotion_min_days = promotion_min_days
        self.promotion_min_alpha = promotion_min_alpha

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    # ------------------------------------------------------------------
    # Equity curve helpers
    # ------------------------------------------------------------------

    def _load_shadow_equity(
        self,
        shadow_id: str,
        days: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Load (ts, virtual_equity) pairs for a shadow_id.

        Parameters
        ----------
        shadow_id : str
        days : int | None
            If given, only return last N days of data.

        Returns
        -------
        List of (iso_ts, equity) sorted ascending by ts.
        """
        query = (
            "SELECT ts, virtual_equity FROM shadow_runs "
            "WHERE shadow_id = ? ORDER BY ts ASC"
        )
        params: list[Any] = [shadow_id]

        if days is not None:
            cutoff = _days_ago_iso(days)
            query = (
                "SELECT ts, virtual_equity FROM shadow_runs "
                "WHERE shadow_id = ? AND ts >= ? ORDER BY ts ASC"
            )
            params.append(cutoff)

        with self._db() as con:
            rows = con.execute(query, params).fetchall()
        return [(row["ts"], float(row["virtual_equity"])) for row in rows]

    def _load_live_equity(
        self,
        days: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Load live strategy equity curve from ``live_equity`` table if present,
        or return empty list.
        """
        try:
            query = (
                "SELECT ts, equity FROM live_equity ORDER BY ts ASC"
            )
            params: list[Any] = []
            if days is not None:
                cutoff = _days_ago_iso(days)
                query = (
                    "SELECT ts, equity FROM live_equity "
                    "WHERE ts >= ? ORDER BY ts ASC"
                )
                params.append(cutoff)

            with self._db() as con:
                rows = con.execute(query, params).fetchall()
            return [(row["ts"], float(row["equity"])) for row in rows]
        except sqlite3.OperationalError:
            return []

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_shadows_by_alpha(
        self,
        live_equity_curve: list[float] | None = None,
        shadow_curves: dict[str, list[float]] | None = None,
        shadow_ids: list[str] | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Rank all shadows by their alpha versus the live strategy.

        Parameters
        ----------
        live_equity_curve : list[float] | None
            Pre-loaded live equity curve.  If ``None``, loads from DB.
        shadow_curves : dict[str, list[float]] | None
            Pre-loaded shadow curves {shadow_id -> [equity, ...]}.
            If ``None``, loads from DB for each ``shadow_ids`` entry.
        shadow_ids : list[str] | None
            Which shadows to compare.  If ``None``, loads all shadow_ids from DB.
        days : int
            Lookback window in days.

        Returns
        -------
        List of dicts sorted by alpha descending:
            [{"shadow_id": str, "genome_id": int, "alpha": float,
              "shadow_return": float, "live_return": float, "n_bars": int}, ...]
        """
        # Resolve live curve
        if live_equity_curve is None:
            live_rows = self._load_live_equity(days=days)
            live_equity_curve = [eq for _, eq in live_rows] if live_rows else []

        # Resolve shadow IDs
        if shadow_ids is None:
            shadow_ids = self._all_shadow_ids()

        results = []
        for sid in shadow_ids:
            genome_id = self._genome_id_for_shadow(sid)
            if shadow_curves and sid in shadow_curves:
                s_curve = shadow_curves[sid]
            else:
                s_rows = self._load_shadow_equity(sid, days=days)
                s_curve = [eq for _, eq in s_rows]

            if len(s_curve) < 2:
                continue

            s_return = (s_curve[-1] - s_curve[0]) / max(s_curve[0], 1.0)

            if len(live_equity_curve) >= 2:
                l_return = (live_equity_curve[-1] - live_equity_curve[0]) / max(live_equity_curve[0], 1.0)
            else:
                l_return = 0.0

            alpha = s_return - l_return
            results.append({
                "shadow_id": sid,
                "genome_id": genome_id,
                "alpha": alpha,
                "shadow_return": s_return,
                "live_return": l_return,
                "n_bars": len(s_curve),
            })

        results.sort(key=lambda r: r["alpha"], reverse=True)
        return results

    def _all_shadow_ids(self) -> list[str]:
        try:
            with self._db() as con:
                rows = con.execute(
                    "SELECT DISTINCT shadow_id FROM shadow_runs"
                ).fetchall()
            return [row["shadow_id"] for row in rows]
        except sqlite3.OperationalError:
            return []

    def _genome_id_for_shadow(self, shadow_id: str) -> int:
        try:
            with self._db() as con:
                row = con.execute(
                    "SELECT genome_id FROM shadow_runs WHERE shadow_id = ? LIMIT 1",
                    (shadow_id,),
                ).fetchone()
            return int(row["genome_id"]) if row else -1
        except sqlite3.OperationalError:
            return -1

    # ------------------------------------------------------------------
    # Alpha decomposition
    # ------------------------------------------------------------------

    def alpha_decomposition(
        self,
        shadow_id: str,
        live_equity_curve: list[float] | None = None,
        days: int = 30,
    ) -> dict[str, float]:
        """
        Decompose a shadow's alpha by signal source (BH, OU, GARCH).

        Approach: load the shadow's trade history, attribute each trade's P&L
        to the dominant signal_source that triggered it, then express each
        bucket as a fraction of total realised P&L.

        Parameters
        ----------
        shadow_id : str
        live_equity_curve : list[float] | None
        days : int

        Returns
        -------
        dict:
            total_alpha : float
            bh_contrib  : float (fraction of total P&L from BH signal)
            ou_contrib  : float
            garch_contrib : float
            combined_contrib : float
        """
        # Load trade history from shadow_runs state JSON
        state_json = self._latest_state_json(shadow_id)
        if state_json is None:
            return {"error": "no state found", "shadow_id": shadow_id}

        try:
            state_dict = json.loads(state_json)
        except json.JSONDecodeError:
            return {"error": "corrupt state JSON", "shadow_id": shadow_id}

        trades = state_dict.get("trades", [])
        if not trades:
            return {
                "shadow_id": shadow_id,
                "total_alpha": 0.0,
                "bh_contrib": 0.0,
                "ou_contrib": 0.0,
                "garch_contrib": 0.0,
                "combined_contrib": 0.0,
            }

        # Bucket P&L by signal source
        buckets: dict[str, float] = {"bh": 0.0, "ou": 0.0, "garch": 0.0, "combined": 0.0}
        total_pnl = 0.0
        for t in trades:
            src = str(t.get("signal_source", "combined")).lower()
            pnl = float(t.get("pnl", 0.0))
            if src in buckets:
                buckets[src] += pnl
            else:
                buckets["combined"] += pnl
            total_pnl += pnl

        denom = abs(total_pnl) if abs(total_pnl) > 1e-8 else 1.0

        # Total alpha vs live
        s_curve = self._load_shadow_equity(shadow_id, days=days)
        s_curve_vals = [e for _, e in s_curve]
        if live_equity_curve is None:
            live_rows = self._load_live_equity(days=days)
            live_equity_curve = [e for _, e in live_rows]

        if len(s_curve_vals) >= 2:
            s_ret = (s_curve_vals[-1] - s_curve_vals[0]) / max(s_curve_vals[0], 1.0)
        else:
            s_ret = 0.0

        if len(live_equity_curve) >= 2:
            l_ret = (live_equity_curve[-1] - live_equity_curve[0]) / max(live_equity_curve[0], 1.0)
        else:
            l_ret = 0.0

        return {
            "shadow_id": shadow_id,
            "total_alpha": s_ret - l_ret,
            "bh_contrib": buckets["bh"] / denom,
            "ou_contrib": buckets["ou"] / denom,
            "garch_contrib": buckets["garch"] / denom,
            "combined_contrib": buckets["combined"] / denom,
            "total_realised_pnl": total_pnl,
        }

    def _latest_state_json(self, shadow_id: str) -> str | None:
        try:
            with self._db() as con:
                row = con.execute(
                    "SELECT shadow_state_json FROM shadow_runs "
                    "WHERE shadow_id = ? AND shadow_state_json IS NOT NULL "
                    "ORDER BY created_at DESC LIMIT 1",
                    (shadow_id,),
                ).fetchone()
            return row["shadow_state_json"] if row else None
        except sqlite3.OperationalError:
            return None

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_shadow(
        self,
        shadow_id: str,
        live_equity_curve: list[float] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        If a shadow has beaten the live strategy for ``promotion_min_days`` or
        more *and* its alpha exceeds ``promotion_min_alpha``, create a hypothesis
        in the ``hypotheses`` table so the idea pipeline can formally test it.

        Parameters
        ----------
        shadow_id : str
        live_equity_curve : list[float] | None
        force : bool
            If True, skip time-period check and promote immediately.

        Returns
        -------
        dict with keys: promoted (bool), hypothesis_id (str | None), reason (str)
        """
        ranking = self.rank_shadows_by_alpha(
            live_equity_curve=live_equity_curve,
            shadow_ids=[shadow_id],
            days=self.promotion_min_days,
        )

        if not ranking:
            return {"promoted": False, "hypothesis_id": None,
                    "reason": "no data for shadow"}

        r = ranking[0]
        alpha = r["alpha"]
        n_bars = r["n_bars"]
        genome_id = r["genome_id"]

        # Estimate whether we have ~30 days of data
        # (rough: assume ≥ 30 bars is ≥ 30 days for daily data)
        has_enough_data = force or (n_bars >= self.promotion_min_days)
        has_enough_alpha = alpha >= self.promotion_min_alpha

        if not has_enough_data:
            return {
                "promoted": False,
                "hypothesis_id": None,
                "reason": f"insufficient data: {n_bars} bars < {self.promotion_min_days}",
            }

        if not has_enough_alpha:
            return {
                "promoted": False,
                "hypothesis_id": None,
                "reason": f"alpha {alpha:.4f} below threshold {self.promotion_min_alpha}",
            }

        # Fetch genome params for hypothesis
        genome_params = self._genome_params(genome_id)
        h_id = str(uuid.uuid4())
        description = (
            f"Shadow {shadow_id} (genome {genome_id}) beat live for "
            f"{n_bars} bars with alpha={alpha:.4f}. "
            f"Proposing genome adoption."
        )

        try:
            with self._db() as con:
                con.execute(
                    """
                    INSERT INTO hypotheses
                        (id, source, description, params, status, created_at)
                    VALUES (?, 'shadow_promotion', ?, ?, 'pending',
                            strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                    """,
                    (h_id, description, json.dumps(genome_params)),
                )
                con.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("Could not insert hypothesis: %s", exc)
            h_id = None  # type: ignore[assignment]

        # Record promotion in shadow_comparisons
        self._record_comparison(shadow_id, genome_id, n_bars, r["shadow_return"],
                                r["live_return"], alpha, promoted=1)

        logger.info("Promoted shadow %s → hypothesis %s (alpha=%.4f)", shadow_id, h_id, alpha)
        return {
            "promoted": True,
            "hypothesis_id": h_id,
            "reason": f"alpha={alpha:.4f} over {n_bars} bars",
            "genome_id": genome_id,
        }

    def auto_promote_winners(
        self,
        live_equity_curve: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Scan all active shadows and promote any that qualify.

        Returns list of promotion results.
        """
        shadow_ids = self._all_shadow_ids()
        results = []
        for sid in shadow_ids:
            r = self.promote_shadow(sid, live_equity_curve=live_equity_curve)
            if r.get("promoted"):
                results.append(r)
        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _record_comparison(
        self,
        shadow_id: str,
        genome_id: int,
        period_days: int,
        shadow_return: float,
        live_return: float,
        alpha: float,
        promoted: int = 0,
    ) -> None:
        try:
            with self._db() as con:
                con.execute(
                    """
                    INSERT INTO shadow_comparisons
                        (shadow_id, genome_id, period_days,
                         shadow_return, live_return, alpha, promoted)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (shadow_id, genome_id, period_days,
                     shadow_return, live_return, alpha, promoted),
                )
                con.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("shadow_comparisons write error: %s", exc)

    def _genome_params(self, genome_id: int) -> dict[str, Any]:
        try:
            with self._db() as con:
                row = con.execute(
                    "SELECT params FROM hall_of_fame WHERE id = ?",
                    (genome_id,),
                ).fetchone()
            return json.loads(row["params"]) if row else {}
        except (sqlite3.OperationalError, json.JSONDecodeError):
            return {}

    # ------------------------------------------------------------------
    # Performance statistics helpers
    # ------------------------------------------------------------------

    def sharpe_ratio(
        self,
        equity_curve: list[float],
        annualise: bool = True,
        bars_per_year: float = 252.0,
    ) -> float:
        """
        Compute Sharpe ratio from an equity curve.

        Parameters
        ----------
        equity_curve : list[float]
            Sequence of equity values.
        annualise : bool
            If True, multiply by sqrt(bars_per_year).
        bars_per_year : float
            Scaling factor (default 252 for daily bars).

        Returns
        -------
        Sharpe ratio (float).  Returns 0.0 for degenerate inputs.
        """
        if len(equity_curve) < 3:
            return 0.0
        arr = np.array(equity_curve, dtype=float)
        returns = np.diff(arr) / np.where(arr[:-1] > 1e-8, arr[:-1], 1.0)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        if sigma < 1e-12:
            return 0.0
        sharpe = mu / sigma
        if annualise:
            sharpe *= (bars_per_year ** 0.5)
        return float(sharpe)

    def max_drawdown(self, equity_curve: list[float]) -> float:
        """
        Compute maximum drawdown from an equity curve.

        Returns positive fraction (e.g. 0.15 = 15 % drawdown).
        """
        if len(equity_curve) < 2:
            return 0.0
        arr = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd = (peak - arr) / np.where(peak > 1e-8, peak, 1.0)
        return float(dd.max())

    def calmar_ratio(
        self,
        equity_curve: list[float],
        bars_per_year: float = 252.0,
    ) -> float:
        """
        Calmar ratio = annualised return / max drawdown.
        """
        if len(equity_curve) < 2:
            return 0.0
        total_ret = (equity_curve[-1] - equity_curve[0]) / max(equity_curve[0], 1e-8)
        annualised = total_ret * (bars_per_year / max(len(equity_curve) - 1, 1))
        mdd = self.max_drawdown(equity_curve)
        if mdd < 1e-8:
            return 0.0
        return float(annualised / mdd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_ago_iso(days: int) -> str:
    """Return ISO 8601 timestamp for N days ago."""
    import datetime
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
