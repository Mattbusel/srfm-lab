"""
SensitivityAnalyzer
===================
Variance-based sensitivity analysis of genome parameters over a collection
of counterfactual simulation results.

Provides:
  - ``sobol_indices``        — first-order and total-order Sobol S1/ST indices
  - ``tornado_chart_data``   — per-parameter impact range (for dashboard)
  - ``interaction_matrix``   — pairwise parameter interaction strengths
  - ``report``               — full report dict, persisted to idea_engine.db
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB path resolution (mirrors oracle.py)
# ---------------------------------------------------------------------------

DB_PATH_ENV = "IDEA_ENGINE_DB"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "db" / "idea_engine.db"

SCORE_COL = "improvement"


# ---------------------------------------------------------------------------
# Helper: build a results DataFrame-like structure from list of dicts
# ---------------------------------------------------------------------------

class _ResultsArray:
    """Thin wrapper around a list of result dicts for numerical ops."""

    def __init__(self, rows: list[dict[str, Any]], param_names: list[str]) -> None:
        self.param_names = param_names
        n = len(rows)
        d = len(param_names)
        self.X = np.zeros((n, d))   # param matrix
        self.Y = np.zeros(n)        # score vector

        for i, row in enumerate(rows):
            params = row.get("params_full", {})
            if isinstance(params, str):
                params = json.loads(params)
            for j, name in enumerate(param_names):
                self.X[i, j] = float(params.get(name, 0.0))
            self.Y[i] = float(row.get("improvement", 0.0))

        self.n = n
        self.d = d


# ---------------------------------------------------------------------------
# SensitivityAnalyzer class
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """
    Sensitivity analysis for counterfactual parameter sweeps.

    Parameters
    ----------
    db_path : str | Path | None
        Path to ``idea_engine.db``.
    param_names : list[str] | None
        Parameter names to analyse.  If ``None``, reads from
        ``parameter_space.PARAM_BOUNDS``.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        param_names: list[str] | None = None,
    ) -> None:
        self.db_path = Path(
            db_path or os.environ.get(DB_PATH_ENV, DEFAULT_DB_PATH)
        )
        if param_names is None:
            from .parameter_space import PARAM_BOUNDS
            self._param_names = list(PARAM_BOUNDS.keys())
        else:
            self._param_names = list(param_names)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_results(self, run_id: str) -> list[dict[str, Any]]:
        """Load counterfactual rows for a given baseline run_id."""
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT params_full, improvement FROM counterfactual_results "
                "WHERE baseline_run_id = ? AND improvement IS NOT NULL "
                "ORDER BY created_at",
                (run_id,),
            ).fetchall()
        finally:
            con.close()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Sobol indices (variance-based, first-order)
    # ------------------------------------------------------------------

    def sobol_indices(
        self,
        results_df: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute first-order (S1) and total-order (ST) Sobol sensitivity indices
        for each parameter.

        Uses a rank-based estimator (Saltelli 2002 / Jansen 1999) that works
        well without requiring a specific sampling design.

        Parameters
        ----------
        results_df : list of result dicts | None
            Pre-loaded results.  If ``None``, ``run_id`` must be given.
        run_id : str | None
            Baseline run_id to query from DB if ``results_df`` is None.

        Returns
        -------
        dict: param_name -> {"S1": float, "ST": float}
        """
        rows = results_df if results_df is not None else self._load_results(run_id or "")
        if len(rows) < 4:
            logger.warning("Too few results (%d) for reliable Sobol estimation", len(rows))
            return {name: {"S1": 0.0, "ST": 0.0} for name in self._param_names}

        arr = _ResultsArray(rows, self._param_names)
        Y = arr.Y
        X = arr.X
        n, d = arr.n, arr.d

        # Variance of output
        var_Y = float(np.var(Y, ddof=1))
        if var_Y < 1e-12:
            return {name: {"S1": 0.0, "ST": 0.0} for name in self._param_names}

        indices: dict[str, dict[str, float]] = {}
        for j in range(d):
            name = self._param_names[j]
            # Rank-based conditional variance estimator for S1
            # Sort by X[:, j] and compute running conditional variance
            sort_idx = np.argsort(X[:, j])
            X_sorted = X[sort_idx, j]
            Y_sorted = Y[sort_idx]

            # Bin into sqrt(n) groups
            n_bins = max(4, int(np.sqrt(n)))
            bin_edges = np.array_split(np.arange(n), n_bins)
            cond_means = np.array([Y_sorted[idx].mean() for idx in bin_edges])
            s1_raw = float(np.var(cond_means, ddof=1))
            s1 = np.clip(s1_raw / var_Y, 0.0, 1.0)

            # Total-order index via complementary variance
            # Partition data into two halves by median of X[:, j]
            med = np.median(X[:, j])
            mask_lo = X[:, j] <= med
            mask_hi = X[:, j] > med
            y_lo = Y[mask_lo]
            y_hi = Y[mask_hi]
            if len(y_lo) > 1 and len(y_hi) > 1:
                # Average within-group variance
                st_raw = (np.var(y_lo, ddof=1) + np.var(y_hi, ddof=1)) / 2.0
                st = np.clip(st_raw / var_Y, 0.0, 1.0)
            else:
                st = s1

            indices[name] = {"S1": float(s1), "ST": float(st)}

        return indices

    # ------------------------------------------------------------------
    # Tornado chart data
    # ------------------------------------------------------------------

    def tornado_chart_data(
        self,
        results_df: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
        n_percentile: float = 10.0,
    ) -> list[dict[str, Any]]:
        """
        For each parameter, compute the range of improvement scores observed
        between the bottom and top ``n_percentile`` percentiles of that param.

        This answers "how much does score change when this parameter moves
        from low to high?"  — the classic tornado / one-way sensitivity chart.

        Parameters
        ----------
        results_df : list[dict] | None
        run_id : str | None
        n_percentile : float
            Percentile cutoff (default 10 = bottom 10% vs top 10%).

        Returns
        -------
        List of dicts sorted by ``score_range`` descending:
            [{"param": name, "low_score": float, "high_score": float,
              "score_range": float, "low_p": float, "high_p": float}, ...]
        """
        rows = results_df if results_df is not None else self._load_results(run_id or "")
        if len(rows) < 4:
            return []

        arr = _ResultsArray(rows, self._param_names)
        X, Y = arr.X, arr.Y

        tornado: list[dict[str, Any]] = []
        for j, name in enumerate(self._param_names):
            x = X[:, j]
            lo_p = np.percentile(x, n_percentile)
            hi_p = np.percentile(x, 100.0 - n_percentile)

            mask_lo = x <= lo_p
            mask_hi = x >= hi_p

            if mask_lo.sum() == 0 or mask_hi.sum() == 0:
                continue

            score_lo = float(Y[mask_lo].mean())
            score_hi = float(Y[mask_hi].mean())
            score_range = abs(score_hi - score_lo)

            tornado.append({
                "param": name,
                "low_score": score_lo,
                "high_score": score_hi,
                "score_range": score_range,
                "low_p": float(lo_p),
                "high_p": float(hi_p),
            })

        tornado.sort(key=lambda d: d["score_range"], reverse=True)
        return tornado

    # ------------------------------------------------------------------
    # Interaction matrix
    # ------------------------------------------------------------------

    def interaction_matrix(
        self,
        results_df: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute pairwise parameter interaction strengths.

        For each pair (i, j), the interaction strength is the Pearson
        correlation between the product X_i * X_j and Y (after normalising
        each parameter to [0, 1]).  A large positive value means the
        combination of the two parameters has a strong joint effect on score.

        Parameters
        ----------
        results_df : list[dict] | None
        run_id : str | None

        Returns
        -------
        Nested dict: param_i -> param_j -> correlation coefficient
        """
        rows = results_df if results_df is not None else self._load_results(run_id or "")
        if len(rows) < 4:
            return {}

        arr = _ResultsArray(rows, self._param_names)
        X, Y = arr.X, arr.Y

        # Normalise X column-wise to [0, 1]
        X_norm = np.zeros_like(X)
        for j in range(arr.d):
            lo, hi = X[:, j].min(), X[:, j].max()
            rng = hi - lo
            if rng > 1e-12:
                X_norm[:, j] = (X[:, j] - lo) / rng
            else:
                X_norm[:, j] = 0.5

        Y_centered = Y - Y.mean()

        matrix: dict[str, dict[str, float]] = {}
        for i, name_i in enumerate(self._param_names):
            matrix[name_i] = {}
            for j, name_j in enumerate(self._param_names):
                if i == j:
                    matrix[name_i][name_j] = 1.0
                    continue
                interaction = X_norm[:, i] * X_norm[:, j]
                interaction_centered = interaction - interaction.mean()
                denom = (
                    np.linalg.norm(interaction_centered) * np.linalg.norm(Y_centered)
                )
                if denom < 1e-12:
                    matrix[name_i][name_j] = 0.0
                else:
                    corr = float(np.dot(interaction_centered, Y_centered) / denom)
                    matrix[name_i][name_j] = float(np.clip(corr, -1.0, 1.0))

        return matrix

    # ------------------------------------------------------------------
    # Top-k most influential parameters
    # ------------------------------------------------------------------

    def top_parameters(
        self,
        results_df: list[dict[str, Any]] | None = None,
        run_id: str | None = None,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Return the ``k`` most influential parameters sorted by total-order
        Sobol index.

        Returns
        -------
        List of (param_name, ST) tuples, sorted by ST descending.
        """
        indices = self.sobol_indices(results_df=results_df, run_id=run_id)
        ranked = sorted(indices.items(), key=lambda x: x[1]["ST"], reverse=True)
        return [(name, vals["ST"]) for name, vals in ranked[:k]]

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def report(
        self,
        run_id: str,
        results_df: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a full sensitivity report for ``run_id`` and persist it
        to the ``sensitivity_reports`` table.

        Parameters
        ----------
        run_id : str
        results_df : list[dict] | None
            Pre-loaded results, or ``None`` to query from DB.

        Returns
        -------
        dict with keys:
            - ``sobol``       : {param -> {S1, ST}}
            - ``tornado``     : list of tornado rows
            - ``interaction`` : {param_i -> {param_j -> corr}}
            - ``top_params``  : [(param, ST), ...]
            - ``n_samples``   : int
        """
        rows = results_df if results_df is not None else self._load_results(run_id)
        if not rows:
            logger.warning("No counterfactual results found for run_id=%s", run_id)
            return {"error": "no_results", "run_id": run_id}

        sobol = self.sobol_indices(results_df=rows)
        tornado = self.tornado_chart_data(results_df=rows)
        interaction = self.interaction_matrix(results_df=rows)
        top = self.top_parameters(results_df=rows)

        result: dict[str, Any] = {
            "run_id": run_id,
            "n_samples": len(rows),
            "sobol": sobol,
            "tornado": tornado,
            "interaction": interaction,
            "top_params": top,
        }

        self._persist_report(run_id, sobol, tornado)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_report(
        self,
        run_id: str,
        sobol: dict[str, dict[str, float]],
        tornado: list[dict[str, Any]],
    ) -> None:
        """Write per-parameter rows into ``sensitivity_reports``."""
        tornado_map = {row["param"]: row["score_range"] for row in tornado}

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db_path))
        try:
            con.execute("PRAGMA journal_mode=WAL")
            for param_name, idx in sobol.items():
                tornado_range = float(tornado_map.get(param_name, 0.0))
                con.execute(
                    """
                    INSERT INTO sensitivity_reports
                        (run_id, param_name, sobol_index, tornado_range)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        param_name,
                        idx.get("ST", 0.0),
                        tornado_range,
                    ),
                )
            con.commit()
        except sqlite3.OperationalError as exc:
            logger.warning("Could not persist sensitivity report: %s", exc)
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SensitivityAnalyzer(db={self.db_path}, "
            f"n_params={len(self._param_names)})"
        )
