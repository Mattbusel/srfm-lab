"""
Regime-specific scenario simulation — models market regime transitions and
simulates forward regime paths for conditional risk analysis.

Capabilities:
  - Markov chain regime transition model with calibrated probabilities
  - Forward simulation of regime paths
  - Regime-conditional return distributions and correlation structures
  - Regime-conditional VaR / CVaR
  - Transition scenario analysis ("what if regime switches from X to Y")
  - Monte Carlo over regime paths + asset returns
  - Optimal strategy allocation per regime
  - Regime misclassification cost
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

REGIMES = ["risk_on", "risk_off", "crisis", "recovery", "low_vol_grind"]


@dataclass
class RegimeParams:
    """Statistical parameters for a single regime."""
    name: str
    expected_return_annual: float          # annualised expected return
    volatility_annual: float               # annualised vol
    skewness: float = 0.0
    kurtosis_excess: float = 0.0
    correlation_matrix: Optional[np.ndarray] = None   # n_assets × n_assets
    avg_duration_days: int = 60
    description: str = ""


@dataclass
class RegimeTransitionSpec:
    """Specification for a conditional transition probability override."""
    from_regime: str
    to_regime: str
    base_probability: float
    vix_sensitivity: float = 0.0        # per point of VIX above 20
    spread_sensitivity: float = 0.0     # per 100bp of HY spread above 400


@dataclass
class RegimePath:
    """A simulated forward path of regime states."""
    regimes: List[str]                   # one per day
    transitions: List[Tuple[int, str, str]]   # (day, from, to)
    duration: int = 0

    def __post_init__(self):
        self.duration = len(self.regimes)


@dataclass
class RegimeConditionalRisk:
    """VaR / CVaR conditional on a regime."""
    regime: str
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_return: float
    volatility: float


@dataclass
class StrategyAllocation:
    """Optimal allocation for a given regime."""
    regime: str
    weights: Dict[str, float]
    expected_sharpe: float
    expected_return: float
    expected_vol: float


# ---------------------------------------------------------------------------
# Default regime parameter library
# ---------------------------------------------------------------------------

def _default_regime_params() -> Dict[str, RegimeParams]:
    """Build calibrated regime parameter set."""
    n = 5  # assets: equity, bond, commodity, crypto, cash
    params: Dict[str, RegimeParams] = {}

    # Risk-on: trending up, moderate vol
    corr_risk_on = np.array([
        [1.00, -0.20,  0.30,  0.60,  0.00],
        [-0.20, 1.00, -0.10, -0.15,  0.05],
        [0.30, -0.10,  1.00,  0.25,  0.00],
        [0.60, -0.15,  0.25,  1.00,  0.00],
        [0.00,  0.05,  0.00,  0.00,  1.00],
    ])
    params["risk_on"] = RegimeParams(
        name="risk_on", expected_return_annual=0.15, volatility_annual=0.14,
        skewness=-0.2, kurtosis_excess=0.5, correlation_matrix=corr_risk_on,
        avg_duration_days=90, description="Bull market, positive momentum",
    )

    # Risk-off: flight to safety
    corr_risk_off = np.array([
        [1.00, -0.50,  0.40,  0.70,  0.00],
        [-0.50, 1.00, -0.20, -0.30,  0.10],
        [0.40, -0.20,  1.00,  0.35,  0.00],
        [0.70, -0.30,  0.35,  1.00,  0.00],
        [0.00,  0.10,  0.00,  0.00,  1.00],
    ])
    params["risk_off"] = RegimeParams(
        name="risk_off", expected_return_annual=-0.10, volatility_annual=0.22,
        skewness=-0.8, kurtosis_excess=2.0, correlation_matrix=corr_risk_off,
        avg_duration_days=45, description="Risk-off, flight to quality",
    )

    # Crisis: high vol, contagion, correlations go to 1
    corr_crisis = np.array([
        [1.00, -0.30,  0.60,  0.85,  0.00],
        [-0.30, 1.00, -0.10, -0.25,  0.15],
        [0.60, -0.10,  1.00,  0.55,  0.00],
        [0.85, -0.25,  0.55,  1.00,  0.00],
        [0.00,  0.15,  0.00,  0.00,  1.00],
    ])
    params["crisis"] = RegimeParams(
        name="crisis", expected_return_annual=-0.35, volatility_annual=0.45,
        skewness=-1.5, kurtosis_excess=5.0, correlation_matrix=corr_crisis,
        avg_duration_days=20, description="Full crisis — contagion, liquidation",
    )

    # Recovery: snapping back from crisis
    corr_recovery = np.array([
        [1.00, -0.10,  0.25,  0.50,  0.00],
        [-0.10, 1.00, -0.05, -0.10,  0.05],
        [0.25, -0.05,  1.00,  0.20,  0.00],
        [0.50, -0.10,  0.20,  1.00,  0.00],
        [0.00,  0.05,  0.00,  0.00,  1.00],
    ])
    params["recovery"] = RegimeParams(
        name="recovery", expected_return_annual=0.25, volatility_annual=0.20,
        skewness=0.3, kurtosis_excess=1.0, correlation_matrix=corr_recovery,
        avg_duration_days=40, description="Post-crisis recovery rally",
    )

    # Low-vol grind: complacency
    corr_lowvol = np.array([
        [1.00, -0.15,  0.20,  0.40,  0.00],
        [-0.15, 1.00, -0.05, -0.10,  0.05],
        [0.20, -0.05,  1.00,  0.15,  0.00],
        [0.40, -0.10,  0.15,  1.00,  0.00],
        [0.00,  0.05,  0.00,  0.00,  1.00],
    ])
    params["low_vol_grind"] = RegimeParams(
        name="low_vol_grind", expected_return_annual=0.08, volatility_annual=0.08,
        skewness=-0.1, kurtosis_excess=0.2, correlation_matrix=corr_lowvol,
        avg_duration_days=120, description="Low volatility, grinding higher",
    )

    return params


# ---------------------------------------------------------------------------
# Regime transition model
# ---------------------------------------------------------------------------

class RegimeTransitionModel:
    """
    Markov chain model of regime transitions with calibrated probabilities.

    Transition matrix rows = from-regime, columns = to-regime.
    """

    def __init__(self, regime_params: Optional[Dict[str, RegimeParams]] = None,
                 seed: int = 42):
        self.regime_params = regime_params or _default_regime_params()
        self.regime_names = list(self.regime_params.keys())
        self.n_regimes = len(self.regime_names)
        self.rng = np.random.default_rng(seed)

        # Default daily transition matrix (rows must sum to 1)
        self.transition_matrix = self._build_default_transition_matrix()

    def _build_default_transition_matrix(self) -> np.ndarray:
        """Build default daily transition probabilities."""
        # Indexed by self.regime_names order
        # risk_on, risk_off, crisis, recovery, low_vol_grind
        T = np.array([
            [0.970, 0.015, 0.003, 0.002, 0.010],   # risk_on
            [0.020, 0.940, 0.025, 0.005, 0.010],   # risk_off
            [0.005, 0.020, 0.930, 0.040, 0.005],   # crisis
            [0.040, 0.010, 0.005, 0.930, 0.015],   # recovery
            [0.015, 0.010, 0.002, 0.003, 0.970],   # low_vol_grind
        ])
        # Normalize rows
        for i in range(T.shape[0]):
            T[i] /= T[i].sum()
        return T

    def _regime_index(self, name: str) -> int:
        return self.regime_names.index(name)

    def set_transition_prob(self, from_regime: str, to_regime: str,
                            prob: float) -> None:
        """Override a specific transition probability and re-normalize."""
        i = self._regime_index(from_regime)
        j = self._regime_index(to_regime)
        self.transition_matrix[i, j] = prob
        # Re-normalize row
        row_sum = self.transition_matrix[i].sum()
        if row_sum > 0:
            self.transition_matrix[i] /= row_sum

    def stationary_distribution(self) -> Dict[str, float]:
        """Compute stationary distribution of the Markov chain."""
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        # Find eigenvector for eigenvalue ≈ 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return {name: float(stationary[i]) for i, name in enumerate(self.regime_names)}

    # -----------------------------------------------------------------------
    # Forward simulation
    # -----------------------------------------------------------------------

    def simulate_regime_path(self, start_regime: str, n_days: int) -> RegimePath:
        """Simulate a single regime path forward n_days."""
        current = self._regime_index(start_regime)
        regimes: List[str] = [start_regime]
        transitions: List[Tuple[int, str, str]] = []

        for day in range(1, n_days):
            probs = self.transition_matrix[current]
            next_idx = int(self.rng.choice(self.n_regimes, p=probs))
            next_name = self.regime_names[next_idx]
            if next_idx != current:
                transitions.append((day, self.regime_names[current], next_name))
            current = next_idx
            regimes.append(next_name)

        return RegimePath(regimes=regimes, transitions=transitions)

    def simulate_paths(self, start_regime: str, n_days: int,
                       n_paths: int = 1000) -> List[RegimePath]:
        """Simulate many regime paths."""
        return [self.simulate_regime_path(start_regime, n_days) for _ in range(n_paths)]

    def regime_occupation_probabilities(self, start_regime: str, n_days: int,
                                        n_paths: int = 5000) -> Dict[str, np.ndarray]:
        """
        For each regime, compute the probability of being in that regime on each day.

        Returns dict of regime_name -> array of shape (n_days,).
        """
        counts = {name: np.zeros(n_days) for name in self.regime_names}
        for _ in range(n_paths):
            path = self.simulate_regime_path(start_regime, n_days)
            for day, regime in enumerate(path.regimes):
                counts[regime][day] += 1
        for name in counts:
            counts[name] /= n_paths
        return counts

    # -----------------------------------------------------------------------
    # Regime-conditional returns
    # -----------------------------------------------------------------------

    def sample_returns(self, regime: str, n_samples: int,
                       n_assets: int = 5) -> np.ndarray:
        """
        Sample daily returns conditional on being in a given regime.

        Returns array of shape (n_samples, n_assets).
        """
        rp = self.regime_params[regime]
        daily_mu = rp.expected_return_annual / 252
        daily_sigma = rp.volatility_annual / math.sqrt(252)

        means = np.full(n_assets, daily_mu)
        if rp.correlation_matrix is not None:
            cov = rp.correlation_matrix * (daily_sigma ** 2)
        else:
            cov = np.eye(n_assets) * (daily_sigma ** 2)

        samples = self.rng.multivariate_normal(means, cov, size=n_samples)
        return samples

    def regime_conditional_var_cvar(self, regime: str, portfolio_weights: np.ndarray,
                                    n_samples: int = 50_000,
                                    confidence: float = 0.95) -> RegimeConditionalRisk:
        """Compute VaR and CVaR conditional on being in a specific regime."""
        n_assets = len(portfolio_weights)
        returns = self.sample_returns(regime, n_samples, n_assets)
        port_returns = returns @ portfolio_weights

        sorted_ret = np.sort(port_returns)
        idx_95 = int((1 - 0.95) * n_samples)
        idx_99 = int((1 - 0.99) * n_samples)

        var_95 = float(sorted_ret[idx_95])
        var_99 = float(sorted_ret[idx_99])
        cvar_95 = float(np.mean(sorted_ret[:idx_95])) if idx_95 > 0 else var_95
        cvar_99 = float(np.mean(sorted_ret[:idx_99])) if idx_99 > 0 else var_99

        return RegimeConditionalRisk(
            regime=regime,
            var_95=var_95, var_99=var_99,
            cvar_95=cvar_95, cvar_99=cvar_99,
            expected_return=float(np.mean(port_returns)),
            volatility=float(np.std(port_returns)),
        )

    def all_regime_risk(self, portfolio_weights: np.ndarray) -> List[RegimeConditionalRisk]:
        """Compute risk metrics for all regimes."""
        return [self.regime_conditional_var_cvar(r, portfolio_weights)
                for r in self.regime_names]

    # -----------------------------------------------------------------------
    # Transition scenario analysis
    # -----------------------------------------------------------------------

    def transition_scenario(self, from_regime: str, to_regime: str,
                            portfolio_weights: np.ndarray,
                            transition_days: int = 5,
                            n_sims: int = 10_000) -> Dict[str, float]:
        """
        "What if we're about to transition from X to Y?"

        Simulates transition_days in from_regime then switches to to_regime
        for another transition_days. Returns P&L distribution stats.
        """
        n_assets = len(portfolio_weights)
        pnl_array = np.zeros(n_sims)

        for i in range(n_sims):
            # Phase 1: still in from_regime
            ret1 = self.sample_returns(from_regime, transition_days, n_assets)
            # Phase 2: now in to_regime
            ret2 = self.sample_returns(to_regime, transition_days, n_assets)
            combined = np.vstack([ret1, ret2])
            # Compound returns
            cumulative = np.prod(1.0 + combined, axis=0) - 1.0
            pnl_array[i] = float(cumulative @ portfolio_weights)

        sorted_pnl = np.sort(pnl_array)
        n = len(sorted_pnl)
        return {
            "from_regime": from_regime,
            "to_regime": to_regime,
            "mean_pnl_pct": float(np.mean(pnl_array) * 100),
            "std_pnl_pct": float(np.std(pnl_array) * 100),
            "var_95_pct": float(sorted_pnl[int(0.05 * n)] * 100),
            "cvar_95_pct": float(np.mean(sorted_pnl[:int(0.05 * n)]) * 100),
            "max_loss_pct": float(sorted_pnl[0] * 100),
            "prob_negative": float(np.mean(pnl_array < 0)),
        }

    # -----------------------------------------------------------------------
    # Monte Carlo over regime paths + returns
    # -----------------------------------------------------------------------

    def full_monte_carlo(self, start_regime: str, portfolio_weights: np.ndarray,
                         n_days: int = 252, n_paths: int = 2000) -> Dict[str, object]:
        """
        Full Monte Carlo: simulate regime paths, then sample returns
        conditional on the regime each day. Compound to get terminal P&L.
        """
        n_assets = len(portfolio_weights)
        terminal_pnl = np.zeros(n_paths)

        for p in range(n_paths):
            path = self.simulate_regime_path(start_regime, n_days)
            cumulative = np.ones(n_assets)
            for day in range(n_days):
                regime = path.regimes[day]
                daily_ret = self.sample_returns(regime, 1, n_assets)[0]
                cumulative *= (1.0 + daily_ret)
            port_ret = float((cumulative - 1.0) @ portfolio_weights)
            terminal_pnl[p] = port_ret

        sorted_pnl = np.sort(terminal_pnl)
        n = len(sorted_pnl)
        return {
            "mean_return_pct": float(np.mean(terminal_pnl) * 100),
            "std_return_pct": float(np.std(terminal_pnl) * 100),
            "median_return_pct": float(np.median(terminal_pnl) * 100),
            "var_95_pct": float(sorted_pnl[int(0.05 * n)] * 100),
            "var_99_pct": float(sorted_pnl[int(0.01 * n)] * 100),
            "cvar_95_pct": float(np.mean(sorted_pnl[:int(0.05 * n)]) * 100),
            "max_loss_pct": float(sorted_pnl[0] * 100),
            "max_gain_pct": float(sorted_pnl[-1] * 100),
            "prob_negative": float(np.mean(terminal_pnl < 0)),
            "terminal_distribution": terminal_pnl,
        }

    # -----------------------------------------------------------------------
    # Optimal strategy allocation per regime
    # -----------------------------------------------------------------------

    def optimal_allocation(self, regime: str, asset_names: List[str],
                           risk_free_rate: float = 0.04,
                           n_samples: int = 50_000) -> StrategyAllocation:
        """
        Compute mean-variance optimal allocation for a specific regime.

        Uses sample-based estimation of mean and covariance.
        """
        n_assets = len(asset_names)
        returns = self.sample_returns(regime, n_samples, n_assets)
        mu = np.mean(returns, axis=0) * 252           # annualise
        cov = np.cov(returns, rowvar=False) * 252

        # Minimum variance with target return (analytical Markowitz)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        ones = np.ones(n_assets)
        A = float(mu @ inv_cov @ ones)
        B = float(mu @ inv_cov @ mu)
        C = float(ones @ inv_cov @ ones)
        D = B * C - A * A

        if abs(D) < 1e-12:
            # Fallback: equal weight
            weights = ones / n_assets
        else:
            # Max Sharpe portfolio
            excess = mu - risk_free_rate
            raw_weights = inv_cov @ excess
            w_sum = raw_weights.sum()
            if abs(w_sum) > 1e-12:
                weights = raw_weights / w_sum
            else:
                weights = ones / n_assets

        # Clip extreme weights
        weights = np.clip(weights, -1.0, 2.0)
        weights /= np.abs(weights).sum()

        port_ret = float(weights @ mu)
        port_vol = float(math.sqrt(weights @ cov @ weights))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return StrategyAllocation(
            regime=regime,
            weights={name: float(w) for name, w in zip(asset_names, weights)},
            expected_sharpe=sharpe,
            expected_return=port_ret,
            expected_vol=port_vol,
        )

    def all_regime_allocations(self, asset_names: List[str],
                               risk_free_rate: float = 0.04) -> List[StrategyAllocation]:
        """Compute optimal allocation for every regime."""
        return [self.optimal_allocation(r, asset_names, risk_free_rate)
                for r in self.regime_names]

    # -----------------------------------------------------------------------
    # Regime misclassification cost
    # -----------------------------------------------------------------------

    def misclassification_cost(self, true_regime: str, assumed_regime: str,
                               portfolio_weights: np.ndarray,
                               n_sims: int = 20_000) -> Dict[str, float]:
        """
        Cost of being in the wrong regime: compute expected P&L difference
        between running the optimal strategy for assumed_regime while
        actually being in true_regime.
        """
        n_assets = len(portfolio_weights)

        # Returns under true regime
        true_returns = self.sample_returns(true_regime, n_sims, n_assets)
        true_pnl = true_returns @ portfolio_weights

        # What we would have gotten with the right allocation
        asset_names = [f"asset_{i}" for i in range(n_assets)]
        optimal = self.optimal_allocation(true_regime, asset_names)
        optimal_weights = np.array([optimal.weights[n] for n in asset_names])
        optimal_pnl = true_returns @ optimal_weights

        cost = float(np.mean(optimal_pnl) - np.mean(true_pnl))
        cost_std = float(np.std(optimal_pnl - true_pnl))

        return {
            "true_regime": true_regime,
            "assumed_regime": assumed_regime,
            "mean_cost_daily": cost,
            "cost_std": cost_std,
            "mean_cost_annual": cost * 252,
            "prob_underperform": float(np.mean(true_pnl < optimal_pnl)),
        }

    def misclassification_matrix(self, portfolio_weights: np.ndarray
                                  ) -> Dict[str, Dict[str, float]]:
        """Build full matrix of misclassification costs."""
        matrix: Dict[str, Dict[str, float]] = {}
        for true_r in self.regime_names:
            matrix[true_r] = {}
            for assumed_r in self.regime_names:
                if true_r == assumed_r:
                    matrix[true_r][assumed_r] = 0.0
                else:
                    result = self.misclassification_cost(true_r, assumed_r,
                                                         portfolio_weights,
                                                         n_sims=5000)
                    matrix[true_r][assumed_r] = result["mean_cost_annual"]
        return matrix

    # -----------------------------------------------------------------------
    # Regime surprise probability
    # -----------------------------------------------------------------------

    def regime_surprise_prob(self, current_regime: str,
                             horizon_days: int = 5) -> Dict[str, float]:
        """
        Probability of regime surprise: ending up in an unexpected regime
        within horizon_days, starting from current_regime.
        """
        # Raise transition matrix to the power of horizon_days
        idx = self._regime_index(current_regime)
        T_n = np.linalg.matrix_power(self.transition_matrix, horizon_days)
        probs = T_n[idx]

        surprise_probs: Dict[str, float] = {}
        for i, name in enumerate(self.regime_names):
            if name == current_regime:
                surprise_probs[name] = 0.0  # not a surprise
            else:
                surprise_probs[name] = float(probs[i])
        return surprise_probs

    def expected_regime_duration(self, regime: str) -> float:
        """Expected number of days to stay in a regime (geometric distribution)."""
        idx = self._regime_index(regime)
        stay_prob = self.transition_matrix[idx, idx]
        if stay_prob >= 1.0:
            return float("inf")
        return 1.0 / (1.0 - stay_prob)

    def all_expected_durations(self) -> Dict[str, float]:
        """Expected duration for every regime."""
        return {name: self.expected_regime_duration(name)
                for name in self.regime_names}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def regime_transition_summary(model: RegimeTransitionModel) -> Dict[str, object]:
    """Summarize the regime model."""
    return {
        "regimes": model.regime_names,
        "stationary_distribution": model.stationary_distribution(),
        "expected_durations": model.all_expected_durations(),
        "transition_matrix": model.transition_matrix.tolist(),
    }


def compare_regime_risk(model: RegimeTransitionModel,
                        portfolio_weights: np.ndarray) -> List[Dict[str, float]]:
    """Compare risk across all regimes for quick overview."""
    risks = model.all_regime_risk(portfolio_weights)
    rows: List[Dict[str, float]] = []
    for r in risks:
        rows.append({
            "regime": r.regime,
            "expected_return_daily": round(r.expected_return * 100, 4),
            "volatility_daily": round(r.volatility * 100, 4),
            "var_95": round(r.var_95 * 100, 4),
            "cvar_95": round(r.cvar_95 * 100, 4),
        })
    return rows
