"""
stress_testing.py — Stress testing framework for SRFM strategies.

Historical scenario replay, synthetic stress tests, correlation breakdown,
liquidity crises, and fat-tail shock analysis.
"""

from __future__ import annotations

import json
import argparse
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Historical stress scenarios
# ---------------------------------------------------------------------------

STRESS_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "covid_crash_2020": {
        "name": "COVID-19 Market Crash (Feb–Mar 2020)",
        "description": "Fastest 30% drawdown in S&P 500 history. VIX spiked to 82.69.",
        "shock_bps": -3500,          # peak-to-trough basis points
        "vol_multiplier": 5.0,       # vol expansion factor
        "correlation_increase": 0.40, # correlation added to all pairs
        "duration_days": 33,
        "recovery_days": 148,
        "bid_ask_multiplier": 8.0,   # liquidity deterioration
        "drift_daily": -0.040,       # daily drift during crash
        "vol_daily": 0.065,
    },
    "gfc_2008": {
        "name": "Global Financial Crisis (Sep–Nov 2008)",
        "description": "Lehman Brothers collapse, systemic credit freeze.",
        "shock_bps": -5700,
        "vol_multiplier": 6.5,
        "correlation_increase": 0.50,
        "duration_days": 60,
        "recovery_days": 400,
        "bid_ask_multiplier": 12.0,
        "drift_daily": -0.025,
        "vol_daily": 0.055,
    },
    "dot_com_crash_2000": {
        "name": "Dot-Com Crash (Mar 2000 – Oct 2002)",
        "description": "NASDAQ fell 78% from peak. Slow-grind bear market.",
        "shock_bps": -7800,
        "vol_multiplier": 3.0,
        "correlation_increase": 0.20,
        "duration_days": 929,
        "recovery_days": 2000,
        "bid_ask_multiplier": 3.0,
        "drift_daily": -0.002,
        "vol_daily": 0.025,
    },
    "flash_crash_2010": {
        "name": "Flash Crash (May 6, 2010)",
        "description": "S&P fell 10% intraday, recovered within hours.",
        "shock_bps": -1000,
        "vol_multiplier": 10.0,
        "correlation_increase": 0.30,
        "duration_days": 1,
        "recovery_days": 5,
        "bid_ask_multiplier": 20.0,
        "drift_daily": -0.10,
        "vol_daily": 0.15,
    },
    "taper_tantrum_2013": {
        "name": "Taper Tantrum (May–Jun 2013)",
        "description": "Fed signals QE tapering; bonds sold off sharply.",
        "shock_bps": -800,
        "vol_multiplier": 2.5,
        "correlation_increase": 0.15,
        "duration_days": 45,
        "recovery_days": 120,
        "bid_ask_multiplier": 2.0,
        "drift_daily": -0.006,
        "vol_daily": 0.018,
    },
    "ukraine_invasion_2022": {
        "name": "Russia–Ukraine War (Feb 2022)",
        "description": "Commodity spike, energy shock, geopolitical risk premium.",
        "shock_bps": -1200,
        "vol_multiplier": 3.0,
        "correlation_increase": 0.25,
        "duration_days": 20,
        "recovery_days": 90,
        "bid_ask_multiplier": 4.0,
        "drift_daily": -0.012,
        "vol_daily": 0.030,
    },
    "crypto_winter_2022": {
        "name": "Crypto Winter / FTX Collapse (2022)",
        "description": "BTC fell 75% from ATH; FTX bankruptcy Nov 2022.",
        "shock_bps": -7500,
        "vol_multiplier": 4.0,
        "correlation_increase": 0.60,
        "duration_days": 365,
        "recovery_days": 400,
        "bid_ask_multiplier": 6.0,
        "drift_daily": -0.006,
        "vol_daily": 0.060,
    },
    "volmageddon_2018": {
        "name": "Volmageddon (Feb 5, 2018)",
        "description": "Short-vol ETPs (XIV) collapse. VIX doubled in one day.",
        "shock_bps": -1100,
        "vol_multiplier": 15.0,
        "correlation_increase": 0.35,
        "duration_days": 2,
        "recovery_days": 30,
        "bid_ask_multiplier": 10.0,
        "drift_daily": -0.045,
        "vol_daily": 0.080,
    },
    "rate_shock_2022": {
        "name": "Fed Rate Shock (2022 Rate Hike Cycle)",
        "description": "Fastest rate hike cycle in 40 years; 60/40 portfolios hit hardest.",
        "shock_bps": -2500,
        "vol_multiplier": 2.5,
        "correlation_increase": 0.15,
        "duration_days": 270,
        "recovery_days": 300,
        "bid_ask_multiplier": 2.5,
        "drift_daily": -0.003,
        "vol_daily": 0.020,
    },
    "black_monday_1987": {
        "name": "Black Monday (Oct 19, 1987)",
        "description": "S&P fell 22.6% in a single day.",
        "shock_bps": -2260,
        "vol_multiplier": 25.0,
        "correlation_increase": 0.70,
        "duration_days": 1,
        "recovery_days": 300,
        "bid_ask_multiplier": 30.0,
        "drift_daily": -0.226,
        "vol_daily": 0.20,
    },
}


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario_name: str
    description: str
    peak_drawdown_pct: float
    max_consecutive_losses: int
    total_pnl_bps: float
    sharpe_during: float
    survived: bool         # strategy capital > 0 at end
    margin_calls: int      # times equity fell below 20% initial
    worst_day_bps: float
    best_day_bps: float
    n_days: int

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario_name,
            "description": self.description,
            "peak_drawdown_pct": round(self.peak_drawdown_pct, 2),
            "max_consecutive_losses": self.max_consecutive_losses,
            "total_pnl_bps": round(self.total_pnl_bps, 2),
            "sharpe_during": round(self.sharpe_during, 3),
            "survived": self.survived,
            "margin_calls": self.margin_calls,
            "worst_day_bps": round(self.worst_day_bps, 2),
            "best_day_bps": round(self.best_day_bps, 2),
            "n_days": self.n_days,
        }


@dataclass
class StressTestReport:
    strategy_name: str
    n_scenarios: int
    survival_rate: float
    mean_drawdown_pct: float
    worst_scenario: str
    best_scenario: str
    results: List[ScenarioResult]
    correlation_breakdown: Dict[str, float]
    liquidity_impact: Dict[str, float]
    tail_risk_metrics: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "n_scenarios": self.n_scenarios,
            "survival_rate": round(self.survival_rate, 3),
            "mean_drawdown_pct": round(self.mean_drawdown_pct, 2),
            "worst_scenario": self.worst_scenario,
            "best_scenario": self.best_scenario,
            "scenario_results": [r.to_dict() for r in self.results],
            "correlation_breakdown": {k: round(v, 4) for k, v in self.correlation_breakdown.items()},
            "liquidity_impact": {k: round(v, 4) for k, v in self.liquidity_impact.items()},
            "tail_risk_metrics": {k: round(v, 4) for k, v in self.tail_risk_metrics.items()},
        }


# ---------------------------------------------------------------------------
# StressTester
# ---------------------------------------------------------------------------

class StressTester:
    """
    Stress testing engine for SRFM strategies.

    Runs historical and synthetic stress scenarios against a strategy's
    typical return distribution, measuring drawdowns, survival, and risk metrics.
    """

    def __init__(
        self,
        strategy_returns: Optional[pd.Series] = None,
        initial_capital: float = 1_000_000.0,
        position_size_pct: float = 0.02,  # 2% risk per trade
        strategy_name: str = "SRFM_Strategy",
    ):
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct

        if strategy_returns is None:
            self.strategy_returns = self._generate_default_returns()
        else:
            self.strategy_returns = strategy_returns.dropna()

        # Fit return distribution parameters
        self._fit_distribution()

    def _generate_default_returns(self, n: int = 500, seed: int = 42) -> pd.Series:
        """Generate synthetic strategy returns with fat tails."""
        rng = np.random.default_rng(seed)
        # Student-t with 5 dof for fat tails
        returns = stats.t.rvs(df=5, loc=0.0003, scale=0.008, size=n, random_state=rng)
        dates = pd.date_range("2023-01-01", periods=n, freq="1D")
        return pd.Series(returns, index=dates, name="strategy_returns")

    def _fit_distribution(self) -> None:
        """Fit normal and t-distribution to strategy returns."""
        rets = self.strategy_returns.values
        self.mean_daily = float(rets.mean())
        self.std_daily = float(rets.std(ddof=1))
        self.skew = float(stats.skew(rets))
        self.kurt = float(stats.kurtosis(rets))

        # Fit t-distribution
        try:
            df, loc, scale = stats.t.fit(rets)
            self.t_df = float(df)
            self.t_loc = float(loc)
            self.t_scale = float(scale)
        except Exception:
            self.t_df = 5.0
            self.t_loc = self.mean_daily
            self.t_scale = self.std_daily

    # ------------------------------------------------------------------
    # Scenario simulation
    # ------------------------------------------------------------------

    def _simulate_scenario_returns(
        self,
        scenario: Dict[str, Any],
        seed: int = 0,
    ) -> np.ndarray:
        """
        Generate daily returns for a stress scenario by combining:
        - Strategy's typical return distribution
        - Scenario-specific drift, vol multiplier, and shock
        """
        rng = np.random.default_rng(seed)
        n_days = max(int(scenario["duration_days"]), 1)
        recovery_days = max(int(scenario["recovery_days"]), 1)
        total_days = n_days + recovery_days

        vol_mult = scenario["vol_multiplier"]
        drift = scenario["drift_daily"]
        vol = scenario["vol_daily"]
        shock_bps = scenario["shock_bps"] / 10_000  # convert to fraction

        # Strategy typically has partial exposure to market
        # We model: strategy_return = alpha + beta * market_return
        # beta ~ 0.3 for a typical BH strategy (not fully correlated)
        beta = 0.35
        alpha = self.mean_daily

        # During stress: market follows scenario parameters
        stress_market = rng.normal(drift, vol, n_days)

        # Distribute shock across first few days
        shock_days = min(3, n_days)
        for d in range(shock_days):
            stress_market[d] += shock_bps / shock_days

        # Strategy return during stress: alpha + beta * market (reduced by vol regime)
        # Alpha decays during stress (BH strategy struggles in trending regimes initially)
        alpha_decay = np.exp(-np.arange(n_days) / max(n_days / 3, 1)) * alpha
        strategy_shock = alpha_decay + beta * stress_market
        # Add strategy-specific noise (higher vol during stress)
        strategy_noise_stress = rng.normal(0, self.std_daily * vol_mult * 0.5, n_days)
        stress_returns = strategy_shock + strategy_noise_stress

        # Recovery period: returns revert to normal
        recovery_fraction = np.linspace(1.0, 0.0, recovery_days)
        recovery_drift = (
            recovery_fraction * drift
            + (1 - recovery_fraction) * self.mean_daily
        )
        recovery_vol = (
            recovery_fraction * vol * vol_mult
            + (1 - recovery_fraction) * self.std_daily
        )
        recovery_returns = rng.normal(recovery_drift, recovery_vol)
        strategy_noise_rec = rng.normal(0, self.std_daily, recovery_days)
        recovery_rets = recovery_returns + strategy_noise_rec * 0.3

        return np.concatenate([stress_returns, recovery_rets])

    def _evaluate_return_stream(
        self,
        returns: np.ndarray,
        scenario: Dict[str, Any],
    ) -> ScenarioResult:
        """Compute risk metrics from a return stream."""
        capital = self.initial_capital
        equity_curve = [capital]
        daily_pnl_bps = []
        margin_calls = 0
        peak = capital

        for r in returns:
            pnl = capital * r * self.position_size_pct / self.std_daily if self.std_daily > 0 else capital * r
            capital += pnl
            capital = max(capital, 0.0)
            equity_curve.append(capital)
            daily_pnl_bps.append(r * 10_000)
            peak = max(peak, capital)
            if capital < self.initial_capital * 0.2:
                margin_calls += 1

        equity_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (running_max - equity_arr) / (running_max + 1e-9)
        peak_dd = float(drawdowns.max() * 100)

        pnl_arr = np.array(daily_pnl_bps)
        total_pnl = float(pnl_arr.sum())
        worst_day = float(pnl_arr.min()) if len(pnl_arr) > 0 else 0.0
        best_day = float(pnl_arr.max()) if len(pnl_arr) > 0 else 0.0

        sharpe = 0.0
        if pnl_arr.std() > 1e-10:
            sharpe = float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(252))

        # Consecutive losses
        max_consec = 0
        curr_consec = 0
        for p in pnl_arr:
            if p < 0:
                curr_consec += 1
                max_consec = max(max_consec, curr_consec)
            else:
                curr_consec = 0

        survived = capital > 0.0

        return ScenarioResult(
            scenario_name=scenario["name"],
            description=scenario["description"],
            peak_drawdown_pct=peak_dd,
            max_consecutive_losses=max_consec,
            total_pnl_bps=total_pnl,
            sharpe_during=sharpe,
            survived=survived,
            margin_calls=margin_calls,
            worst_day_bps=worst_day,
            best_day_bps=best_day,
            n_days=len(returns),
        )

    # ------------------------------------------------------------------
    # Historical scenario runner
    # ------------------------------------------------------------------

    def run_historical_scenarios(
        self,
        scenario_keys: Optional[List[str]] = None,
        seed_base: int = 42,
    ) -> List[ScenarioResult]:
        """
        Run all (or specified) historical stress scenarios.
        Returns list of ScenarioResult sorted by peak drawdown.
        """
        if scenario_keys is None:
            scenario_keys = list(STRESS_SCENARIOS.keys())

        results = []
        for i, key in enumerate(scenario_keys):
            if key not in STRESS_SCENARIOS:
                print(f"Warning: scenario '{key}' not found; skipping")
                continue
            scenario = STRESS_SCENARIOS[key]
            returns = self._simulate_scenario_returns(scenario, seed=seed_base + i)
            result = self._evaluate_return_stream(returns, scenario)
            results.append(result)

        results.sort(key=lambda r: r.peak_drawdown_pct, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Synthetic stress tests
    # ------------------------------------------------------------------

    def run_synthetic_stress(
        self,
        n_simulations: int = 1000,
        shock_sizes_bps: Optional[List[int]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Monte Carlo stress test with random shock magnitudes and durations.
        Returns DataFrame with simulation results.
        """
        if shock_sizes_bps is None:
            shock_sizes_bps = [-500, -1000, -2000, -3500, -5000, -7500]

        rng = np.random.default_rng(seed)
        rows = []

        for shock_bps in shock_sizes_bps:
            for sim in range(n_simulations // len(shock_sizes_bps)):
                duration = int(rng.integers(1, 200))
                vol_mult = float(rng.uniform(1.5, 10.0))
                drift_daily = shock_bps / (duration * 10_000)

                scenario = {
                    "name": f"Synthetic_shock_{shock_bps}bps",
                    "description": f"Synthetic {shock_bps}bps shock",
                    "shock_bps": shock_bps,
                    "vol_multiplier": vol_mult,
                    "correlation_increase": 0.30,
                    "duration_days": duration,
                    "recovery_days": duration * 2,
                    "bid_ask_multiplier": vol_mult,
                    "drift_daily": drift_daily,
                    "vol_daily": self.std_daily * vol_mult,
                }

                returns = self._simulate_scenario_returns(scenario, seed=sim * 100)
                result = self._evaluate_return_stream(returns, scenario)

                rows.append({
                    "shock_bps": shock_bps,
                    "duration_days": duration,
                    "vol_multiplier": vol_mult,
                    "peak_drawdown_pct": result.peak_drawdown_pct,
                    "total_pnl_bps": result.total_pnl_bps,
                    "sharpe_during": result.sharpe_during,
                    "survived": result.survived,
                    "margin_calls": result.margin_calls,
                    "worst_day_bps": result.worst_day_bps,
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Correlation breakdown test
    # ------------------------------------------------------------------

    def correlation_breakdown_test(
        self,
        base_correlation: float = 0.30,
        target_correlations: Optional[List[float]] = None,
        n_assets: int = 5,
        n_days: int = 252,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Test portfolio drawdown as correlations spike toward 1.0.
        Simulates crisis correlation: assets stop diversifying.
        Returns DataFrame with correlation level → portfolio metrics.
        """
        if target_correlations is None:
            target_correlations = [0.0, 0.20, 0.40, 0.60, 0.80, 0.95]

        rng = np.random.default_rng(seed)
        rows = []

        for target_corr in target_correlations:
            # Build correlation matrix (gradually increasing off-diagonal)
            stress_corr = (1 - target_corr) * base_correlation + target_corr
            corr_matrix = np.full((n_assets, n_assets), stress_corr)
            np.fill_diagonal(corr_matrix, 1.0)

            # Ensure PSD via eigenvalue floor
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            eigenvalues = np.clip(eigenvalues, 1e-8, None)
            corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            # Renormalize diagonal
            d = np.sqrt(np.diag(corr_matrix))
            corr_matrix = corr_matrix / np.outer(d, d)

            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                L = np.eye(n_assets)

            # Generate correlated returns
            z = rng.standard_normal((n_days, n_assets))
            rets = z @ L.T

            # Scale to strategy vol
            rets = rets * self.std_daily + self.mean_daily

            # Equal-weight portfolio
            port_rets = rets.mean(axis=1)

            # Metrics
            equity = np.cumprod(1 + port_rets * self.position_size_pct)
            running_max = np.maximum.accumulate(equity)
            dd = (running_max - equity) / (running_max + 1e-9)
            peak_dd = float(dd.max() * 100)

            port_std = float(port_rets.std(ddof=1))
            sharpe = float(port_rets.mean() / port_std * np.sqrt(252)) if port_std > 1e-10 else 0.0

            # Actual realized correlation
            realized_corr = float(np.corrcoef(rets.T)[np.triu_indices(n_assets, k=1)].mean())

            rows.append({
                "target_corr": target_corr,
                "realized_corr": realized_corr,
                "peak_drawdown_pct": peak_dd,
                "portfolio_vol_pct": port_std * 100 * np.sqrt(252),
                "sharpe": sharpe,
                "diversification_ratio": float(np.std(rets, axis=0).mean() / port_std)
                if port_std > 1e-10 else 1.0,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Liquidity stress test
    # ------------------------------------------------------------------

    def liquidity_stress(
        self,
        bid_ask_multipliers: Optional[List[float]] = None,
        position_size_multipliers: Optional[List[float]] = None,
        n_days: int = 252,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Test strategy performance under various liquidity conditions.
        Models: higher bid-ask spreads reduce gross returns.
        Returns DataFrame with liquidity conditions → performance metrics.
        """
        if bid_ask_multipliers is None:
            bid_ask_multipliers = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

        if position_size_multipliers is None:
            position_size_multipliers = [1.0]  # fixed position size by default

        rng = np.random.default_rng(seed)
        base_returns = stats.t.rvs(
            df=self.t_df,
            loc=self.t_loc,
            scale=self.t_scale,
            size=n_days,
            random_state=rng,
        )

        # Base half-spread cost: 1 bps per trade, 0.5 trades per day
        base_spread_cost_daily = 0.0001 * 0.5

        rows = []
        for ba_mult in bid_ask_multipliers:
            for ps_mult in position_size_multipliers:
                spread_cost = base_spread_cost_daily * ba_mult
                adj_returns = base_returns - spread_cost

                # Scale position size (larger positions → more market impact)
                impact_cost = 0.00005 * ps_mult * ba_mult * 0.5
                adj_returns -= impact_cost

                equity = np.cumprod(1 + adj_returns * self.position_size_pct * ps_mult)
                running_max = np.maximum.accumulate(equity)
                dd = (running_max - equity) / (running_max + 1e-9)
                peak_dd = float(dd.max() * 100)

                net_sharpe_ann = 0.0
                if adj_returns.std() > 1e-10:
                    net_sharpe_ann = float(adj_returns.mean() / adj_returns.std() * np.sqrt(252))

                gross_sharpe_ann = 0.0
                if base_returns.std() > 1e-10:
                    gross_sharpe_ann = float(base_returns.mean() / base_returns.std() * np.sqrt(252))

                rows.append({
                    "bid_ask_multiplier": ba_mult,
                    "position_size_multiplier": ps_mult,
                    "spread_cost_bps_daily": spread_cost * 10_000,
                    "impact_cost_bps_daily": impact_cost * 10_000,
                    "net_return_pct_ann": float(adj_returns.mean() * 252 * 100),
                    "gross_return_pct_ann": float(base_returns.mean() * 252 * 100),
                    "peak_drawdown_pct": peak_dd,
                    "net_sharpe_ann": net_sharpe_ann,
                    "gross_sharpe_ann": gross_sharpe_ann,
                    "strategy_profitable": adj_returns.mean() > 0,
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Tail risk metrics
    # ------------------------------------------------------------------

    def compute_tail_risk(
        self,
        confidence_levels: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute comprehensive tail risk metrics from strategy returns.
        Returns dict with VaR, CVaR, Expected Shortfall, tail ratios.
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99, 0.995]

        rets = self.strategy_returns.values
        metrics = {}

        for cl in confidence_levels:
            var = float(np.percentile(rets, (1 - cl) * 100))
            cvar = float(rets[rets <= var].mean()) if (rets <= var).any() else var
            metrics[f"var_{int(cl*100)}_bps"] = var * 10_000
            metrics[f"cvar_{int(cl*100)}_bps"] = cvar * 10_000

        # Tail ratio: average positive tail / average negative tail
        positive_tail = rets[rets > np.percentile(rets, 95)]
        negative_tail = rets[rets < np.percentile(rets, 5)]
        if len(negative_tail) > 0 and abs(negative_tail.mean()) > 1e-10:
            metrics["tail_ratio"] = float(positive_tail.mean() / abs(negative_tail.mean()))
        else:
            metrics["tail_ratio"] = 0.0

        # Maximum drawdown
        equity = np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(equity)
        dd = (running_max - equity) / (running_max + 1e-9)
        metrics["historical_max_drawdown_pct"] = float(dd.max() * 100)

        # Calmar ratio
        ann_return = float(rets.mean() * 252)
        max_dd = dd.max()
        metrics["calmar_ratio"] = float(ann_return / max_dd) if max_dd > 1e-10 else 0.0

        # Omega ratio (at 0 threshold)
        gains = rets[rets > 0].sum()
        losses = abs(rets[rets < 0].sum())
        metrics["omega_ratio"] = float(gains / losses) if losses > 1e-10 else float("inf")

        # Sortino ratio
        downside = rets[rets < 0]
        downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else self.std_daily
        metrics["sortino_ratio"] = float(rets.mean() / downside_std * np.sqrt(252))

        # Kurtosis and skew
        metrics["skewness"] = self.skew
        metrics["excess_kurtosis"] = self.kurt

        return metrics

    # ------------------------------------------------------------------
    # Remove-best stress test
    # ------------------------------------------------------------------

    def stress_remove_best(
        self,
        pct_to_remove: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Remove best N% of returns and recompute performance.
        Tests robustness: does performance rely on a few great days?
        """
        if pct_to_remove is None:
            pct_to_remove = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

        rets = self.strategy_returns.values.copy()
        rows = []

        for pct in pct_to_remove:
            n_remove = int(len(rets) * pct)
            if n_remove > 0:
                sorted_idx = np.argsort(rets)[::-1]  # highest first
                mask = np.ones(len(rets), dtype=bool)
                mask[sorted_idx[:n_remove]] = False
                adj_rets = rets[mask]
            else:
                adj_rets = rets

            if len(adj_rets) == 0:
                continue

            ann_return = float(adj_rets.mean() * 252 * 100)
            ann_vol = float(adj_rets.std(ddof=1) * np.sqrt(252) * 100) if len(adj_rets) > 1 else 0.0
            sharpe = float(adj_rets.mean() / adj_rets.std(ddof=1) * np.sqrt(252)) if adj_rets.std() > 1e-10 else 0.0

            equity = np.cumprod(1 + adj_rets)
            running_max = np.maximum.accumulate(equity)
            dd = (running_max - equity) / (running_max + 1e-9)
            max_dd = float(dd.max() * 100)

            rows.append({
                "pct_removed": pct * 100,
                "n_days_remaining": len(adj_rets),
                "ann_return_pct": ann_return,
                "ann_vol_pct": ann_vol,
                "sharpe": sharpe,
                "max_drawdown_pct": max_dd,
                "profitable": adj_rets.mean() > 0,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Fat tail simulation
    # ------------------------------------------------------------------

    def fat_tail_simulation(
        self,
        n_simulations: int = 10_000,
        horizon_days: int = 252,
        dof_range: Optional[List[float]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Simulate terminal wealth distribution under varying tail fatness (t-distribution dof).
        Lower dof → fatter tails → worse tail outcomes.
        Returns DataFrame with dof, var_95, cvar_95, survival_rate, etc.
        """
        if dof_range is None:
            dof_range = [3.0, 4.0, 5.0, 7.0, 10.0, 20.0, 1000.0]  # 1000 ≈ normal

        rng = np.random.default_rng(seed)
        rows = []

        for dof in dof_range:
            terminal_wealths = []
            for _ in range(n_simulations):
                daily_rets = stats.t.rvs(
                    df=dof,
                    loc=self.mean_daily,
                    scale=self.std_daily,
                    size=horizon_days,
                    random_state=rng,
                )
                equity = np.cumprod(1 + daily_rets * self.position_size_pct)
                terminal_wealths.append(float(equity[-1]))

            tw = np.array(terminal_wealths)
            var_5 = float(np.percentile(tw, 5))
            cvar_5 = float(tw[tw <= var_5].mean()) if (tw <= var_5).any() else var_5

            rows.append({
                "dof": dof,
                "mean_terminal_wealth": float(tw.mean()),
                "median_terminal_wealth": float(np.median(tw)),
                "var_5pct": var_5,
                "cvar_5pct": cvar_5,
                "p1_terminal_wealth": float(np.percentile(tw, 1)),
                "p99_terminal_wealth": float(np.percentile(tw, 99)),
                "survival_rate": float((tw > 0.5).mean()),  # > 50% of initial capital
                "std_terminal_wealth": float(tw.std(ddof=1)),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def run_full_stress_test(
        self,
        scenario_keys: Optional[List[str]] = None,
        seed: int = 42,
    ) -> StressTestReport:
        """Run full stress test suite and return comprehensive report."""
        hist_results = self.run_historical_scenarios(scenario_keys, seed_base=seed)

        # Correlation breakdown
        corr_df = self.correlation_breakdown_test(seed=seed)
        corr_impact = {}
        for _, row in corr_df.iterrows():
            key = f"corr_{row['target_corr']:.2f}"
            corr_impact[key] = row["peak_drawdown_pct"]

        # Liquidity stress
        liq_df = self.liquidity_stress(seed=seed)
        liq_impact = {}
        for _, row in liq_df.iterrows():
            key = f"ba_mult_{row['bid_ask_multiplier']:.0f}x"
            liq_impact[key] = row["net_sharpe_ann"]

        # Tail risk
        tail_risk = self.compute_tail_risk()

        # Survival rate
        survival_rate = float(sum(r.survived for r in hist_results) / max(len(hist_results), 1))

        # Worst / best
        if hist_results:
            worst = max(hist_results, key=lambda r: r.peak_drawdown_pct)
            best = min(hist_results, key=lambda r: r.peak_drawdown_pct)
            mean_dd = float(np.mean([r.peak_drawdown_pct for r in hist_results]))
        else:
            worst = ScenarioResult("none", "", 0, 0, 0, 0, True, 0, 0, 0, 0)
            best = worst
            mean_dd = 0.0

        return StressTestReport(
            strategy_name=self.strategy_name,
            n_scenarios=len(hist_results),
            survival_rate=survival_rate,
            mean_drawdown_pct=mean_dd,
            worst_scenario=worst.scenario_name,
            best_scenario=best.scenario_name,
            results=hist_results,
            correlation_breakdown=corr_impact,
            liquidity_impact=liq_impact,
            tail_risk_metrics=tail_risk,
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot_scenario_results(
        self,
        report: Optional[StressTestReport] = None,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """Generate comprehensive stress test visualization."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib not available; skipping plots")
            return

        if report is None:
            report = self.run_full_stress_test()

        fig = plt.figure(figsize=(22, 24))
        fig.suptitle(
            f"SRFM Stress Testing Report: {report.strategy_name}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

        # ---- 1. Historical scenario drawdowns ----
        ax1 = fig.add_subplot(gs[0, :2])
        if report.results:
            names = [r.scenario_name.replace("_", " ")[:30] for r in report.results]
            dds = [r.peak_drawdown_pct for r in report.results]
            colors = ["darkred" if d > 30 else "orange" if d > 15 else "gold"
                      for d in dds]
            bars = ax1.barh(names, dds, color=colors, alpha=0.8)
            ax1.set_title("Peak Drawdown by Historical Scenario", fontweight="bold")
            ax1.set_xlabel("Peak Drawdown (%)")
            ax1.axvline(x=20, color="k", linestyle="--", alpha=0.5, label="20% threshold")
            for bar, dd in zip(bars, dds):
                ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                         f"{dd:.1f}%", va="center", fontsize=7)
            ax1.legend(fontsize=8)
        else:
            ax1.text(0.5, 0.5, "No scenario results", ha="center", va="center",
                     transform=ax1.transAxes)

        # ---- 2. Survival and Sharpe ----
        ax2 = fig.add_subplot(gs[0, 2])
        if report.results:
            survived = sum(r.survived for r in report.results)
            not_survived = len(report.results) - survived
            ax2.pie(
                [survived, not_survived],
                labels=[f"Survived\n({survived})", f"Failed\n({not_survived})"],
                colors=["green", "red"],
                autopct="%1.0f%%",
                startangle=90,
            )
            ax2.set_title(f"Scenario Survival Rate\n{report.survival_rate:.0%}",
                          fontweight="bold")
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax2.transAxes)

        # ---- 3. Strategy return distribution ----
        ax3 = fig.add_subplot(gs[1, 0])
        rets = self.strategy_returns.values
        ax3.hist(rets * 100, bins=40, color="steelblue", edgecolor="white",
                 alpha=0.7, density=True)
        x = np.linspace(rets.min(), rets.max(), 300) * 100
        normal_pdf = stats.norm.pdf(x / 100, rets.mean(), rets.std()) / 100
        ax3.plot(x, normal_pdf, "r--", linewidth=2, label="Normal fit")
        ax3.set_title("Strategy Return Distribution", fontweight="bold")
        ax3.set_xlabel("Daily Return (%)")
        ax3.set_ylabel("Density")
        ax3.legend(fontsize=8)

        # ---- 4. Correlation breakdown ----
        ax4 = fig.add_subplot(gs[1, 1])
        corr_df = self.correlation_breakdown_test()
        if not corr_df.empty:
            ax4.plot(corr_df["target_corr"], corr_df["peak_drawdown_pct"],
                     "o-", color="darkred", linewidth=2)
            ax4.fill_between(corr_df["target_corr"], corr_df["peak_drawdown_pct"],
                             alpha=0.2, color="red")
            ax4.set_title("Drawdown vs Correlation Spike", fontweight="bold")
            ax4.set_xlabel("Target Correlation")
            ax4.set_ylabel("Peak Drawdown (%)")
        else:
            ax4.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax4.transAxes)

        # ---- 5. Liquidity stress ----
        ax5 = fig.add_subplot(gs[1, 2])
        liq_df = self.liquidity_stress()
        if not liq_df.empty:
            ax5.semilogx(liq_df["bid_ask_multiplier"], liq_df["net_sharpe_ann"],
                         "o-", color="navy", linewidth=2)
            ax5.axhline(y=0, color="k", linestyle="--", alpha=0.6)
            ax5.axhline(y=1.0, color="green", linestyle="--", alpha=0.6, label="Sharpe=1")
            ax5.set_title("Net Sharpe vs Bid-Ask Spread Multiple", fontweight="bold")
            ax5.set_xlabel("Bid-Ask Multiplier (log scale)")
            ax5.set_ylabel("Net Annualized Sharpe")
            ax5.legend(fontsize=8)
        else:
            ax5.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax5.transAxes)

        # ---- 6. Remove-best sensitivity ----
        ax6 = fig.add_subplot(gs[2, 0])
        rb_df = self.stress_remove_best()
        if not rb_df.empty:
            ax6_twin = ax6.twinx()
            ax6.bar(rb_df["pct_removed"], rb_df["ann_return_pct"],
                    width=0.8, color="steelblue", alpha=0.6, label="Ann Return %")
            ax6_twin.plot(rb_df["pct_removed"], rb_df["sharpe"],
                          "o-", color="red", linewidth=2, label="Sharpe")
            ax6.set_title("Performance After Removing Best Days", fontweight="bold")
            ax6.set_xlabel("% of Best Days Removed")
            ax6.set_ylabel("Ann Return (%)", color="steelblue")
            ax6_twin.set_ylabel("Sharpe", color="red")
            ax6.axhline(y=0, color="k", linewidth=0.8)
        else:
            ax6.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax6.transAxes)

        # ---- 7. Fat tail simulation ----
        ax7 = fig.add_subplot(gs[2, 1])
        ft_df = self.fat_tail_simulation(n_simulations=2000)
        if not ft_df.empty:
            ax7.plot(ft_df["dof"], ft_df["median_terminal_wealth"],
                     "o-", color="green", label="Median wealth", linewidth=2)
            ax7.fill_between(ft_df["dof"], ft_df["p1_terminal_wealth"],
                             ft_df["p99_terminal_wealth"],
                             alpha=0.2, color="green", label="1-99th pctile")
            ax7.plot(ft_df["dof"], ft_df["cvar_5pct"],
                     "o--", color="red", label="CVaR 5%", linewidth=2)
            ax7.axhline(y=1.0, color="k", linestyle="--", alpha=0.5)
            ax7.set_title("Terminal Wealth vs Tail Fatness (t-dof)", fontweight="bold")
            ax7.set_xlabel("t-distribution degrees of freedom (lower = fatter tails)")
            ax7.set_ylabel("Terminal Wealth (initial = 1.0)")
            ax7.legend(fontsize=8)
        else:
            ax7.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax7.transAxes)

        # ---- 8. Tail risk metrics ----
        ax8 = fig.add_subplot(gs[2, 2])
        tail_risk = report.tail_risk_metrics
        metrics_to_show = ["var_95_bps", "cvar_95_bps", "var_99_bps", "cvar_99_bps"]
        available = [m for m in metrics_to_show if m in tail_risk]
        if available:
            vals = [tail_risk[m] for m in available]
            labels = [m.replace("_bps", "").replace("_", " ").upper() for m in available]
            colors_bar = ["orange", "red", "darkred", "black"][:len(available)]
            ax8.barh(labels, vals, color=colors_bar, alpha=0.8)
            ax8.axvline(x=0, color="k", linewidth=0.8)
            ax8.set_title("Tail Risk Metrics (bps)", fontweight="bold")
            ax8.set_xlabel("Value (bps)")
        else:
            ax8.text(0.5, 0.5, "No tail metrics", ha="center", va="center",
                     transform=ax8.transAxes)

        # ---- 9. Synthetic stress heatmap ----
        ax9 = fig.add_subplot(gs[3, :])
        synth_df = self.run_synthetic_stress(n_simulations=200,
                                              shock_sizes_bps=[-500, -1000, -2000, -3500, -5000])
        if not synth_df.empty:
            pivot = synth_df.groupby("shock_bps")["peak_drawdown_pct"].describe()[
                ["mean", "50%", "95%"]
            ]
            pivot.columns = ["Mean DD%", "Median DD%", "95th pct DD%"]
            cell_text = [[f"{v:.1f}" for v in row] for row in pivot.values]
            table = ax9.table(
                cellText=cell_text,
                rowLabels=[f"{idx} bps" for idx in pivot.index],
                colLabels=list(pivot.columns),
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            ax9.axis("off")
            ax9.set_title("Synthetic Shock Drawdown Distribution",
                          fontweight="bold", y=0.85)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Stress test report saved to {output_path}")
        elif show:
            plt.show()
        plt.close(fig)

    def export_report(self, report: StressTestReport, output_path: str) -> None:
        """Export full stress test report to JSON."""
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"Stress test report saved to {output_path}")


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------

def compute_historical_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """Historical simulation VaR scaled to horizon."""
    var_1d = float(np.percentile(returns, (1 - confidence) * 100))
    return var_1d * np.sqrt(horizon_days)


def compute_parametric_var(
    mu: float,
    sigma: float,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """Parametric (Gaussian) VaR scaled to horizon."""
    z = stats.norm.ppf(1 - confidence)
    return -(mu * horizon_days + z * sigma * np.sqrt(horizon_days))


def kupiec_test(
    n_obs: int,
    n_violations: int,
    confidence: float = 0.95,
) -> Tuple[float, float, bool]:
    """
    Kupiec POF (proportion of failures) test for VaR model validity.
    Returns: (test_statistic, p_value, reject_null)
    Null: violation rate = 1 - confidence level.
    """
    p = 1 - confidence
    p_hat = n_violations / n_obs if n_obs > 0 else 0.0

    if p_hat == 0 or p_hat == 1:
        return 0.0, 1.0, False

    lr = -2 * (
        n_violations * np.log(p / p_hat)
        + (n_obs - n_violations) * np.log((1 - p) / (1 - p_hat))
    )
    lr = max(lr, 0.0)
    p_val = float(1 - stats.chi2.cdf(lr, df=1))
    reject = p_val < 0.05

    return float(lr), p_val, reject


def expected_shortfall(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Expected Shortfall (CVaR) at given confidence level."""
    var = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)


def generate_correlated_shocks(
    n_assets: int,
    n_days: int,
    correlation: float = 0.5,
    vol: float = 0.015,
    seed: int = 42,
) -> np.ndarray:
    """Generate correlated return shocks for multi-asset stress testing."""
    rng = np.random.default_rng(seed)
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)

    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        L = np.eye(n_assets)

    z = rng.standard_normal((n_days, n_assets))
    return (z @ L.T) * vol


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SRFM Stress Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stress_testing.py --output stress_report.png
  python stress_testing.py --scenarios covid_crash_2020 gfc_2008 --export stress.json
  python stress_testing.py --list-scenarios
""",
    )
    parser.add_argument("--scenarios", type=str, nargs="+",
                        help="Specific scenarios to run (default: all)")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="List available scenarios and exit")
    parser.add_argument("--capital", type=float, default=1_000_000.0,
                        help="Initial capital")
    parser.add_argument("--position-size", type=float, default=0.02,
                        help="Position size as fraction of capital per unit vol")
    parser.add_argument("--output", type=str, default="stress_report.png",
                        help="Output plot path")
    parser.add_argument("--export", type=str,
                        help="Export JSON results to path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.list_scenarios:
        print("Available stress scenarios:")
        for key, val in STRESS_SCENARIOS.items():
            print(f"  {key:<30} — {val['name']}")
        return

    # Create stress tester with synthetic strategy returns
    tester = StressTester(
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        strategy_name="SRFM_BH_Strategy",
    )

    print(f"Strategy return stats:")
    print(f"  Mean daily:  {tester.mean_daily*100:.4f}%")
    print(f"  Daily vol:   {tester.std_daily*100:.4f}%")
    print(f"  Skewness:    {tester.skew:.3f}")
    print(f"  Kurtosis:    {tester.kurt:.3f} (excess)")

    # Run full stress test
    print("\nRunning stress test scenarios...")
    report = tester.run_full_stress_test(
        scenario_keys=args.scenarios,
        seed=args.seed,
    )

    print(f"\n--- Stress Test Summary ---")
    print(f"  Scenarios tested:    {report.n_scenarios}")
    print(f"  Survival rate:       {report.survival_rate:.0%}")
    print(f"  Mean peak drawdown:  {report.mean_drawdown_pct:.1f}%")
    print(f"  Worst scenario:      {report.worst_scenario}")
    print(f"  Best scenario:       {report.best_scenario}")

    print(f"\n--- Tail Risk Metrics ---")
    for k, v in sorted(report.tail_risk_metrics.items()):
        print(f"  {k:<35} {v:.3f}")

    print(f"\n--- Top 5 Worst Scenarios ---")
    for r in report.results[:5]:
        survived = "✓" if r.survived else "✗"
        print(f"  [{survived}] {r.scenario_name[:35]:<35} "
              f"DD={r.peak_drawdown_pct:.1f}%  Sharpe={r.sharpe_during:.2f}")

    print(f"\nGenerating stress test report → {args.output}")
    tester.plot_scenario_results(report=report, output_path=args.output)

    if args.export:
        tester.export_report(report, args.export)


if __name__ == "__main__":
    main()
