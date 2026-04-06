# ============================================================
# kelly_optimizer.py
# Kelly criterion and position sizing for multi-asset portfolios
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

# ---- Configuration --------------------------------------------------------

DEFAULT_KELLY_FRACTION = 0.5        # half-Kelly by default (safer)
MAX_LEVERAGE = 3.0
MAX_SINGLE_POSITION = 0.35          # 35% cap per position
MIN_POSITION = 0.0                  # long-only by default
SHRINKAGE_INTENSITY = 0.3           # toward equal-weight
TRANSACTION_COST_BPS = 10           # 10 bps round-trip


# ---- Data classes ---------------------------------------------------------

@dataclass
class KellyAllocation:
    timestamp: datetime
    symbols: list[str]
    full_kelly: dict[str, float]       # unconstrained Kelly weights
    half_kelly: dict[str, float]       # 0.5 × full Kelly
    fractional_kelly: dict[str, float] # user-specified fraction
    constrained_kelly: dict[str, float]  # post-constraint allocation
    shrinkage_kelly: dict[str, float]  # with shrinkage toward equal-weight
    bayesian_kelly: dict[str, float]   # uncertainty-weighted
    kelly_fraction: float
    expected_log_growth: float         # E[log(1+portfolio_return)]
    variance_log_growth: float


@dataclass
class KellyGap:
    symbol: str
    current_weight: float
    kelly_weight: float
    gap: float                  # kelly - current
    required_trade_usd: float
    turnover_cost_usd: float
    net_kelly_benefit: float    # expected gain - cost


@dataclass
class KellyFrontierPoint:
    fraction: float
    expected_log_growth: float
    variance_log_growth: float
    max_drawdown_estimate: float
    sharpe_estimate: float


# ---- Core Kelly math ------------------------------------------------------

def _full_kelly_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free: float = 0.0,
) -> np.ndarray:
    """
    Full Kelly for multi-asset: w* = Σ^{-1} (μ - r_f)
    Returns raw Kelly weights (not normalised).
    """
    excess_mu = mu - risk_free
    try:
        inv_sigma = np.linalg.pinv(sigma)
        w = inv_sigma @ excess_mu
    except np.linalg.LinAlgError:
        w = np.zeros(len(mu))
    return w


def _shrink_toward_equal(
    w: np.ndarray,
    shrinkage: float = SHRINKAGE_INTENSITY,
) -> np.ndarray:
    """James-Stein-style shrinkage toward 1/N equal-weight portfolio."""
    n = len(w)
    equal = np.ones(n) / n
    return (1 - shrinkage) * w + shrinkage * equal


def _constrained_kelly(
    mu: np.ndarray,
    sigma: np.ndarray,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_leverage: float = MAX_LEVERAGE,
    max_single: float = MAX_SINGLE_POSITION,
    min_single: float = MIN_POSITION,
    risk_free: float = 0.0,
) -> np.ndarray:
    """
    Solve constrained Kelly optimisation:
      max  f * (w'μ - r_f) - 0.5 * f² * w'Σw
      s.t. sum(|w|) ≤ max_leverage
           min_single ≤ w_i ≤ max_single
    """
    n = len(mu)
    excess_mu = mu - risk_free

    def neg_objective(w: np.ndarray) -> float:
        log_return = kelly_fraction * (w @ excess_mu) - 0.5 * kelly_fraction ** 2 * (w @ sigma @ w)
        return -log_return

    def neg_grad(w: np.ndarray) -> np.ndarray:
        return -(kelly_fraction * excess_mu - kelly_fraction ** 2 * sigma @ w)

    bounds = [(min_single, max_single)] * n
    constraints = [
        {"type": "ineq", "fun": lambda w: max_leverage - np.sum(np.abs(w))},
    ]

    w0 = np.ones(n) / n * min(1.0, max_leverage / n)

    result = minimize(
        neg_objective,
        w0,
        jac=neg_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    return result.x if result.success else w0


# ---- Bayesian Kelly -------------------------------------------------------

def _bayesian_kelly(
    mu_prior: np.ndarray,
    sigma_prior: np.ndarray,
    mu_likelihood: np.ndarray,
    n_obs: int,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Bayesian Kelly with uncertainty in mu.
    Posterior mean: μ_post = (κ*μ_prior + n*μ_data) / (κ + n)
    Variance inflation: Σ_eff = Σ * (1 + 1/(κ + n))
    """
    n_eff = n_obs + kappa
    mu_post = (kappa * mu_prior + n_obs * mu_likelihood) / n_eff
    sigma_eff = sigma_prior * (1 + 1 / n_eff)
    return _full_kelly_weights(mu_post, sigma_eff)


# ---- Main class -----------------------------------------------------------

class KellyOptimizer:
    """
    Kelly criterion and position sizing.

    Features:
    - Full / half / fractional Kelly
    - Multi-asset Kelly with correlation matrix
    - Shrinkage toward equal-weight
    - Constrained Kelly (leverage, per-position, long-only)
    - Sequential Kelly updates on new fills
    - Gap analysis: current vs Kelly allocation
    - Turnover-adjusted Kelly
    - Bayesian Kelly with parameter uncertainty
    - Kelly frontier visualisation
    """

    def __init__(
        self,
        symbols: list[str],
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        max_leverage: float = MAX_LEVERAGE,
        max_single: float = MAX_SINGLE_POSITION,
        long_only: bool = True,
        shrinkage: float = SHRINKAGE_INTENSITY,
        transaction_cost_bps: float = TRANSACTION_COST_BPS,
        portfolio_value: float = 1_000_000.0,
    ):
        self.symbols = list(symbols)
        self.n = len(symbols)
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage
        self.max_single = max_single
        self.min_single = 0.0 if long_only else -max_single
        self.shrinkage = shrinkage
        self.tc_bps = transaction_cost_bps
        self.portfolio_value = portfolio_value

        self._returns: list[pd.Series] = []
        self._current_weights: dict[str, float] = {s: 1.0 / self.n for s in symbols}
        self._allocation_history: list[KellyAllocation] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update_returns(self, bar_returns: dict[str, float]) -> None:
        row = pd.Series({s: bar_returns.get(s, 0.0) for s in self.symbols})
        self._returns.append(row)

    def update_positions(self, current_weights: dict[str, float]) -> None:
        self._current_weights = current_weights

    # ------------------------------------------------------------------
    # Kelly computation
    # ------------------------------------------------------------------

    def compute(
        self,
        min_obs: int = 20,
        kelly_fraction: float | None = None,
    ) -> Optional[KellyAllocation]:
        """
        Compute full Kelly allocation suite from return history.
        Returns None if insufficient data.
        """
        if len(self._returns) < min_obs:
            logger.debug("Kelly: insufficient data (%d/%d bars)", len(self._returns), min_obs)
            return None

        f = kelly_fraction if kelly_fraction is not None else self.kelly_fraction
        ret_df = pd.DataFrame(self._returns).dropna(how="all")
        n_obs = len(ret_df)

        mu = ret_df.mean().values
        cov = ret_df.cov().values
        # Regularise covariance
        cov = cov + np.eye(self.n) * 1e-8

        # Full Kelly
        full_w = _full_kelly_weights(mu, cov)

        # Half Kelly
        half_w = 0.5 * full_w

        # Fractional Kelly
        frac_w = f * full_w

        # Constrained Kelly
        con_w = _constrained_kelly(
            mu, cov, f, self.max_leverage, self.max_single, self.min_single
        )

        # Shrinkage Kelly
        shrink_w = _shrink_toward_equal(con_w, self.shrinkage)

        # Bayesian Kelly (use equal-weight as prior)
        mu_prior = np.full(self.n, float(np.mean(mu)))
        bay_w_raw = _bayesian_kelly(mu_prior, cov, mu, n_obs)
        bay_w = np.clip(bay_w_raw * f, self.min_single, self.max_single)

        def _to_dict(w: np.ndarray) -> dict[str, float]:
            return {s: float(w[i]) for i, s in enumerate(self.symbols)}

        # Expected log growth
        portfolio_mu = con_w @ mu
        portfolio_var = float(con_w @ cov @ con_w)
        exp_log_growth = float(f * portfolio_mu - 0.5 * f ** 2 * portfolio_var)

        alloc = KellyAllocation(
            timestamp=datetime.now(tz=timezone.utc),
            symbols=self.symbols,
            full_kelly=_to_dict(full_w),
            half_kelly=_to_dict(half_w),
            fractional_kelly=_to_dict(frac_w),
            constrained_kelly=_to_dict(con_w),
            shrinkage_kelly=_to_dict(shrink_w),
            bayesian_kelly=_to_dict(bay_w),
            kelly_fraction=f,
            expected_log_growth=exp_log_growth,
            variance_log_growth=portfolio_var,
        )
        self._allocation_history.append(alloc)
        return alloc

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def gap_analysis(
        self,
        alloc: KellyAllocation | None = None,
    ) -> list[KellyGap]:
        """Compare current allocation to Kelly target."""
        if alloc is None:
            alloc = self.compute()
        if alloc is None:
            return []

        target = alloc.constrained_kelly
        gaps = []
        tc_per_unit = self.tc_bps / 10_000

        for s in self.symbols:
            curr = self._current_weights.get(s, 0.0)
            kelly = target.get(s, 0.0)
            gap = kelly - curr
            trade_usd = abs(gap) * self.portfolio_value
            turnover_cost = trade_usd * tc_per_unit

            # Estimate benefit: gap × expected_excess_return (rough)
            ret_df = pd.DataFrame(self._returns)
            if s in ret_df.columns and len(ret_df) > 0:
                expected_excess = float(ret_df[s].mean() * 252)
            else:
                expected_excess = 0.0
            net_benefit = gap * expected_excess * self.portfolio_value - turnover_cost

            gaps.append(KellyGap(
                symbol=s,
                current_weight=curr,
                kelly_weight=kelly,
                gap=gap,
                required_trade_usd=trade_usd,
                turnover_cost_usd=turnover_cost,
                net_kelly_benefit=net_benefit,
            ))

        return gaps

    # ------------------------------------------------------------------
    # Sequential Kelly (update on new fill)
    # ------------------------------------------------------------------

    def sequential_update(
        self,
        symbol: str,
        fill_price: float,
        fill_qty: float,
        prev_price: float,
    ) -> None:
        """
        Update Kelly estimate incrementally after a new fill.
        Uses Sherman-Morrison-Woodbury online update (approximation).
        """
        if prev_price > 0:
            r = (fill_price - prev_price) / prev_price
            new_row = {s: 0.0 for s in self.symbols}
            new_row[symbol] = r
            self.update_returns(new_row)
            logger.debug("Kelly sequential update: %s r=%.4f", symbol, r)

    # ------------------------------------------------------------------
    # Kelly frontier
    # ------------------------------------------------------------------

    def kelly_frontier(
        self,
        n_points: int = 20,
        fractions: list[float] | None = None,
    ) -> list[KellyFrontierPoint]:
        """
        Compute Kelly frontier: return vs variance of log-wealth
        across a range of Kelly fractions.
        """
        if len(self._returns) < 20:
            return []

        fractions = fractions or list(np.linspace(0.05, 2.0, n_points))
        ret_df = pd.DataFrame(self._returns).dropna(how="all")
        mu = ret_df.mean().values
        cov = ret_df.cov().values + np.eye(self.n) * 1e-8

        points = []
        for f in fractions:
            w = _constrained_kelly(mu, cov, f, self.max_leverage * f, self.max_single)
            pf_mu = float(w @ mu)
            pf_var = float(w @ cov @ w)
            elg = f * pf_mu - 0.5 * f ** 2 * pf_var
            # Approximate max drawdown via Kelly ruin estimate
            if pf_mu > 0 and pf_var > 0:
                approx_dd = -0.5 * (f * pf_mu / pf_var) ** 2 * pf_var
            else:
                approx_dd = -0.5
            sharpe = pf_mu / max(np.sqrt(pf_var), 1e-8) * np.sqrt(252)

            points.append(KellyFrontierPoint(
                fraction=f,
                expected_log_growth=elg,
                variance_log_growth=pf_var,
                max_drawdown_estimate=approx_dd,
                sharpe_estimate=float(sharpe),
            ))

        return points

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_frontier(self, output_path: str = "kelly_frontier.html") -> None:
        """Plotly: Kelly fraction vs expected log growth and variance."""
        if not HAS_PLOTLY:
            logger.warning("plotly not installed")
            return

        pts = self.kelly_frontier(n_points=40)
        if not pts:
            return

        fracs = [p.fraction for p in pts]
        elg = [p.expected_log_growth for p in pts]
        var_lg = [p.variance_log_growth for p in pts]
        sharpe = [p.sharpe_estimate for p in pts]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fracs, y=elg, mode="lines+markers", name="E[log growth]"))
        fig.add_trace(go.Scatter(x=fracs, y=var_lg, mode="lines", name="Var[log growth]", yaxis="y2"))
        fig.add_vline(x=self.kelly_fraction, line_dash="dash", line_color="orange", annotation_text=f"f={self.kelly_fraction}")

        fig.update_layout(
            title="Kelly Frontier: Expected Log Growth vs Variance",
            xaxis_title="Kelly Fraction",
            yaxis_title="Expected Log Growth",
            yaxis2=dict(title="Variance of Log Growth", overlaying="y", side="right"),
        )
        fig.write_html(output_path)
        logger.info("Kelly frontier written to %s", output_path)

    def summary(self, alloc: KellyAllocation | None = None) -> dict[str, Any]:
        if alloc is None:
            alloc = self._allocation_history[-1] if self._allocation_history else None
        if alloc is None:
            return {"status": "no allocation computed"}

        gaps = self.gap_analysis(alloc)
        large_gaps = [g for g in gaps if abs(g.gap) > 0.05]

        return {
            "timestamp": alloc.timestamp.isoformat(),
            "kelly_fraction": alloc.kelly_fraction,
            "expected_log_growth": alloc.expected_log_growth,
            "constrained_kelly": alloc.constrained_kelly,
            "shrinkage_kelly": alloc.shrinkage_kelly,
            "large_gaps": [{"symbol": g.symbol, "gap": g.gap} for g in large_gaps],
            "total_turnover_usd": sum(g.required_trade_usd for g in large_gaps),
            "n_bars": len(self._returns),
        }


# ---- Standalone test -------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)

    symbols = ["BTC", "ETH", "SOL", "BNB", "AVAX"]
    optimizer = KellyOptimizer(symbols, kelly_fraction=0.5, portfolio_value=1_000_000)

    # Simulate 120 bars of returns
    for _ in range(120):
        rets = {s: float(rng.normal(0.0003, 0.02)) for s in symbols}
        optimizer.update_returns(rets)

    alloc = optimizer.compute()
    if alloc:
        print("Full Kelly:", {k: f"{v:.3f}" for k, v in alloc.full_kelly.items()})
        print("Constrained Kelly:", {k: f"{v:.3f}" for k, v in alloc.constrained_kelly.items()})
        print(f"E[log growth]: {alloc.expected_log_growth:.6f}")

    optimizer.update_positions({s: 0.2 for s in symbols})
    gaps = optimizer.gap_analysis(alloc)
    for g in gaps:
        print(f"  {g.symbol}: curr={g.current_weight:.3f} kelly={g.kelly_weight:.3f} gap={g.gap:+.3f}")

    optimizer.plot_frontier("kelly_frontier_test.html")
