"""
portfolio-optimizer/rebalancer.py

Portfolio rebalancing engine:
  - Drift detection: when should we rebalance?
  - Volatility-aware optimal rebalancing schedule
  - Transaction-cost-aware weight optimisation
  - Tax-aware rebalancing (minimise realised gains)
  - Hypothesis generation for rebalancing improvements

Rebalance events are persisted to the ``rebalance_events`` table.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DriftResult:
    """
    Result of a drift check between current and target weights.

    Attributes
    ----------
    needs_rebalance : bool
        True if any asset has drifted beyond the threshold.
    max_drift : float
        Largest absolute weight deviation.
    drift_amounts : dict[str, float]
        Per-asset signed weight deviations (current - target).
    total_turnover : float
        Sum of absolute weight deviations (round-trip cost proxy).
    """

    needs_rebalance: bool
    max_drift: float
    drift_amounts: dict[str, float]
    total_turnover: float


@dataclass
class RebalanceSchedule:
    """
    Recommended rebalancing schedule and parameters.

    Attributes
    ----------
    frequency_bars : int
        Recommended rebalancing interval in bars.
    frequency_label : str
        Human-readable frequency label ('daily', 'weekly', 'monthly', etc.).
    estimated_annual_cost : float
        Expected annual transaction cost as a fraction of AUM.
    vol_regime : str
        'low', 'normal', or 'high' volatility regime driving the schedule.
    next_rebalance_bar : int
        Bar index of the next recommended rebalance.
    rationale : str
        Explanation of the schedule recommendation.
    """

    frequency_bars: int
    frequency_label: str
    estimated_annual_cost: float
    vol_regime: str
    next_rebalance_bar: int
    rationale: str


@dataclass
class RebalanceTrade:
    """A single rebalancing trade."""

    asset: str
    current_weight: float
    target_weight: float
    trade_weight: float           # signed delta: target - current
    pre_tax_gain: float = 0.0     # gain realised if sold (for tax awareness)
    after_tax_gain: float = 0.0


# ---------------------------------------------------------------------------
# Core Rebalancer
# ---------------------------------------------------------------------------


class Rebalancer:
    """
    Portfolio rebalancing engine.

    Parameters
    ----------
    db_path : str, optional
        SQLite database path for persisting rebalance events.
    annualisation : int
        Bars per year for volatility annualisation.

    Examples
    --------
    >>> import numpy as np
    >>> current = np.array([0.30, 0.25, 0.25, 0.20])
    >>> target  = np.array([0.25, 0.25, 0.25, 0.25])
    >>> assets  = ['A', 'B', 'C', 'D']
    >>> reb = Rebalancer()
    >>> drift = reb.rebalance_need(current, target, assets, threshold=0.05)
    >>> print(drift.needs_rebalance, drift.max_drift)
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        annualisation: int = 252,
    ) -> None:
        self.db_path = db_path
        self.annualisation = annualisation
        self._events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def rebalance_need(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        asset_names: Optional[list[str]] = None,
        threshold: float = 0.05,
    ) -> DriftResult:
        """
        Check whether the portfolio has drifted beyond the rebalance threshold.

        Parameters
        ----------
        current_weights : np.ndarray
            Current portfolio weights.
        target_weights : np.ndarray
            Target portfolio weights.
        asset_names : list[str], optional
            Asset names for the drift report.
        threshold : float
            Maximum allowable absolute drift for any single asset before a
            rebalance is triggered.

        Returns
        -------
        DriftResult
        """
        cw = np.asarray(current_weights, dtype=float)
        tw = np.asarray(target_weights, dtype=float)
        self._check_weights(cw)
        self._check_weights(tw)

        names = asset_names or [f"asset_{i}" for i in range(len(cw))]
        delta = cw - tw
        max_drift = float(np.max(np.abs(delta)))
        total_turnover = float(np.sum(np.abs(delta)))

        return DriftResult(
            needs_rebalance=max_drift > threshold,
            max_drift=max_drift,
            drift_amounts=dict(zip(names, delta.tolist())),
            total_turnover=total_turnover,
        )

    # ------------------------------------------------------------------
    # Optimal rebalancing schedule
    # ------------------------------------------------------------------

    def optimal_rebalance_schedule(
        self,
        equity_curve: pd.Series,
        vol_estimate: Optional[float] = None,
        cost_per_trade: float = 0.001,
        n_assets: int = 5,
        current_bar: int = 0,
    ) -> RebalanceSchedule:
        """
        Recommend a rebalancing frequency based on portfolio volatility and
        estimated transaction costs.

        Principle:
          - Higher volatility → faster drift → more frequent rebalancing.
          - Higher transaction costs → less frequent rebalancing.
          - The optimal frequency minimises: drift_cost + transaction_cost.

        Parameters
        ----------
        equity_curve : pd.Series
            Equity curve used to estimate recent realised volatility.
        vol_estimate : float, optional
            Annualised volatility override.  If None, estimated from equity.
        cost_per_trade : float
            One-way transaction cost as fraction of trade value.
        n_assets : int
            Number of assets in the portfolio.
        current_bar : int
            Current bar index in the series.

        Returns
        -------
        RebalanceSchedule
        """
        # Estimate volatility
        if vol_estimate is not None:
            ann_vol = float(vol_estimate)
        else:
            arr = equity_curve.dropna().values
            if len(arr) > 1:
                rets = np.diff(arr) / arr[:-1]
                window = min(60, len(rets))
                ann_vol = float(np.std(rets[-window:]) * np.sqrt(self.annualisation))
            else:
                ann_vol = 0.15  # default 15%

        # Regime classification
        if ann_vol < 0.10:
            vol_regime = "low"
        elif ann_vol < 0.20:
            vol_regime = "normal"
        else:
            vol_regime = "high"

        # Round-trip cost per full rebalance
        round_trip_cost = 2.0 * cost_per_trade * n_assets * 0.1  # assume 10% avg turnover

        # Optimal frequency: minimise tracking error + cost
        # Tracking error grows as ~sqrt(T) * vol; cost per rebalance is fixed
        # Optimal T ∝ cost / vol²  (simplified)
        if ann_vol > 0:
            optimal_bars = max(
                1,
                int(round(round_trip_cost / (ann_vol**2 / self.annualisation))),
            )
        else:
            optimal_bars = 63  # quarterly default

        # Clamp to reasonable range
        optimal_bars = int(np.clip(optimal_bars, 1, 252))

        # Pick a human-readable label
        if optimal_bars <= 2:
            label = "daily"
        elif optimal_bars <= 7:
            label = "weekly"
        elif optimal_bars <= 22:
            label = "bi-weekly"
        elif optimal_bars <= 63:
            label = "monthly"
        elif optimal_bars <= 126:
            label = "quarterly"
        else:
            label = "semi-annual"

        annual_rebalances = max(1, self.annualisation // optimal_bars)
        estimated_annual_cost = float(annual_rebalances * round_trip_cost)

        rationale = (
            f"Annualised vol={ann_vol:.1%} ({vol_regime} regime). "
            f"Round-trip cost={round_trip_cost:.4f}. "
            f"Optimal rebalance every {optimal_bars} bars ({label}). "
            f"Estimated annual transaction cost: {estimated_annual_cost:.4f}."
        )

        return RebalanceSchedule(
            frequency_bars=optimal_bars,
            frequency_label=label,
            estimated_annual_cost=estimated_annual_cost,
            vol_regime=vol_regime,
            next_rebalance_bar=current_bar + optimal_bars,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Transaction-cost-aware optimisation
    # ------------------------------------------------------------------

    def transaction_cost_aware_optimize(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
        cov_matrix: np.ndarray,
        cost: float = 0.001,
        risk_aversion: float = 1.0,
        asset_names: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Find the optimal post-rebalance weights, balancing tracking error
        against the target with transaction costs.

        Objective: min  (w - w*)ᵀΣ(w - w*) + λ * cost * Σ|wᵢ - wᵢ_curr|

        where w* = target_weights, wᵢ_curr = current_weights.

        Parameters
        ----------
        target_weights : np.ndarray
            Desired target weights.
        current_weights : np.ndarray
            Current portfolio weights.
        cov_matrix : np.ndarray
            Asset covariance matrix.
        cost : float
            One-way transaction cost per unit of weight traded.
        risk_aversion : float
            Controls tracking-error penalty vs. cost penalty.
        asset_names : list[str], optional
            Asset names (for labelling only).

        Returns
        -------
        np.ndarray
            Optimal weights after accounting for transaction costs.
        """
        tw = np.asarray(target_weights, dtype=float)
        cw = np.asarray(current_weights, dtype=float)
        n = len(tw)

        self._check_weights(tw)
        self._check_weights(cw)

        def objective(w: np.ndarray) -> float:
            tracking = float((w - tw) @ cov_matrix @ (w - tw))
            tc_cost = cost * float(np.sum(np.abs(w - cw)))
            return risk_aversion * tracking + tc_cost

        bounds = Bounds(lb=0.0, ub=1.0)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = cw.copy()

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        w_opt = res.x / res.x.sum()
        self._check_weights(w_opt)
        return w_opt

    # ------------------------------------------------------------------
    # Tax-aware rebalancing
    # ------------------------------------------------------------------

    def tax_aware_rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        asset_cost_basis: np.ndarray,
        current_prices: np.ndarray,
        tax_rate: float = 0.15,
        asset_names: Optional[list[str]] = None,
    ) -> list[RebalanceTrade]:
        """
        Generate a tax-minimising rebalancing trade list.

        Strategy:
          - Sort sell candidates by ascending unrealised gain (sell lowest-gain
            positions first to minimise tax drag).
          - Buy positions that are underweight.

        Parameters
        ----------
        current_weights : np.ndarray
            Current portfolio weights.
        target_weights : np.ndarray
            Target portfolio weights.
        asset_cost_basis : np.ndarray
            Average cost basis per unit for each asset.
        current_prices : np.ndarray
            Current market price per unit for each asset.
        tax_rate : float
            Short-term capital gains tax rate.
        asset_names : list[str], optional
            Asset names.

        Returns
        -------
        list[RebalanceTrade]
            Trades ordered by tax efficiency (lowest gain sold first).
        """
        cw = np.asarray(current_weights, dtype=float)
        tw = np.asarray(target_weights, dtype=float)
        basis = np.asarray(asset_cost_basis, dtype=float)
        prices = np.asarray(current_prices, dtype=float)
        n = len(cw)

        self._check_weights(cw)
        self._check_weights(tw)

        names = asset_names or [f"asset_{i}" for i in range(n)]
        delta = tw - cw  # positive = buy, negative = sell

        # Unrealised gain per unit
        unrealised_gain = prices - basis
        tax_cost = unrealised_gain * tax_rate

        trades: list[RebalanceTrade] = []
        for i in range(n):
            if abs(delta[i]) < 1e-6:
                continue
            pre_tax = float(unrealised_gain[i] * abs(delta[i])) if delta[i] < 0 else 0.0
            after_tax = float(pre_tax * (1.0 - tax_rate)) if delta[i] < 0 else 0.0

            trades.append(
                RebalanceTrade(
                    asset=names[i],
                    current_weight=float(cw[i]),
                    target_weight=float(tw[i]),
                    trade_weight=float(delta[i]),
                    pre_tax_gain=pre_tax,
                    after_tax_gain=after_tax,
                )
            )

        # Sort sells by ascending pre-tax gain (sell least-appreciated first)
        sells = sorted(
            [t for t in trades if t.trade_weight < 0], key=lambda t: t.pre_tax_gain
        )
        buys = [t for t in trades if t.trade_weight >= 0]
        return sells + buys

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def generate_rebalancing_hypothesis(
        self,
        current_allocation: dict[str, float],
        optimal_allocation: dict[str, float],
        schedule: Optional[RebalanceSchedule] = None,
    ) -> dict[str, Any]:
        """
        Generate a Hypothesis suggesting the transition from the current
        portfolio allocation to the optimal one.

        Parameters
        ----------
        current_allocation : dict[str, float]
            Current {asset: weight} mapping.
        optimal_allocation : dict[str, float]
            Target {asset: weight} mapping from an optimiser.
        schedule : RebalanceSchedule, optional
            Recommended rebalancing schedule.

        Returns
        -------
        dict
            Hypothesis creation kwargs compatible with ``Hypothesis.create``.
        """
        assets = sorted(set(current_allocation) | set(optimal_allocation))
        curr = np.array([current_allocation.get(a, 0.0) for a in assets])
        opt = np.array([optimal_allocation.get(a, 0.0) for a in assets])
        delta = opt - curr
        turnover = float(np.sum(np.abs(delta)))

        # Estimate benefit
        predicted_sharpe_delta = float(min(0.30, turnover * 0.5))
        predicted_dd_delta = float(-min(0.10, turnover * 0.20))

        big_moves = {
            a: round(float(d), 4)
            for a, d in zip(assets, delta.tolist())
            if abs(d) > 0.03
        }

        schedule_info: dict[str, Any] = {}
        if schedule:
            schedule_info = {
                "recommended_frequency": schedule.frequency_label,
                "estimated_annual_cost": schedule.estimated_annual_cost,
                "vol_regime": schedule.vol_regime,
            }

        params: dict[str, Any] = {
            "action": "rebalance_portfolio",
            "current_allocation": current_allocation,
            "optimal_allocation": optimal_allocation,
            "significant_changes": big_moves,
            "total_turnover": round(turnover, 4),
            "schedule": schedule_info,
        }

        description = (
            f"Portfolio rebalancing hypothesis: total turnover {turnover:.1%}. "
            f"Significant weight changes: {big_moves}. "
            + (
                f"Recommended frequency: {schedule.frequency_label} "
                f"(vol regime: {schedule.vol_regime})."
                if schedule
                else ""
            )
        )

        return {
            "hypothesis_type": "PARAMETER_TWEAK",
            "parent_pattern_id": str(uuid.uuid4()),
            "parameters": params,
            "predicted_sharpe_delta": round(predicted_sharpe_delta, 4),
            "predicted_dd_delta": round(predicted_dd_delta, 4),
            "novelty_score": 0.50,
            "description": description,
        }

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def record_rebalance_event(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
        trigger: str,
        estimated_cost: Optional[float] = None,
        ts: Optional[str] = None,
    ) -> None:
        """
        Buffer a rebalance event for DB persistence.

        Parameters
        ----------
        old_weights : dict[str, float]
            Pre-rebalance weights.
        new_weights : dict[str, float]
            Post-rebalance weights.
        trigger : str
            One of ``'drift'``, ``'scheduled'``, ``'vol_spike'``, ``'manual'``.
        estimated_cost : float, optional
            Estimated transaction cost for this rebalance.
        ts : str, optional
            ISO-8601 timestamp; defaults to now.
        """
        allowed = {"drift", "scheduled", "vol_spike", "manual"}
        if trigger not in allowed:
            raise ValueError(f"trigger must be one of {allowed}, got {trigger!r}")

        self._events.append(
            {
                "ts": ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "old_weights": json.dumps(old_weights),
                "new_weights": json.dumps(new_weights),
                "trigger": trigger,
                "estimated_cost": estimated_cost,
            }
        )

    def flush_to_db(self) -> int:
        """
        Write buffered rebalance events to the ``rebalance_events`` table.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if not self.db_path or not self._events:
            return 0

        rows = [
            (
                e["ts"],
                e["old_weights"],
                e["new_weights"],
                e["trigger"],
                e["estimated_cost"],
            )
            for e in self._events
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO rebalance_events
                    (ts, old_weights, new_weights, trigger, estimated_cost)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        inserted = len(rows)
        self._events.clear()
        return inserted

    def fetch_events(self, trigger: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve rebalance events from the database.

        Parameters
        ----------
        trigger : str, optional
            Filter by trigger type.

        Returns
        -------
        pd.DataFrame
        """
        if not self.db_path:
            raise RuntimeError("db_path not configured")

        query = "SELECT * FROM rebalance_events"
        params: tuple = ()
        if trigger:
            query += " WHERE trigger = ?"
            params = (trigger,)
        query += " ORDER BY ts"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_weights(weights: np.ndarray, tol: float = 1e-4) -> None:
        """Raise if weights do not sum to 1.0."""
        total = float(np.sum(weights))
        if abs(total - 1.0) > tol:
            raise ValueError(
                f"Portfolio weights must sum to 1.0; got {total:.6f}"
            )
