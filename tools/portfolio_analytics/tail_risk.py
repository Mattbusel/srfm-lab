# ============================================================
# tail_risk.py
# Comprehensive tail risk monitoring: VaR, CVaR, EVT, stress
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma

logger = logging.getLogger(__name__)

# ---- Configuration --------------------------------------------------------

CONFIDENCE_LEVELS = [0.95, 0.99, 0.999]
HORIZONS = [1, 5, 10]   # trading days
GPD_THRESHOLD_PCT = 0.90  # tail threshold for GPD fitting

# ---- Stress Scenarios (historical return shocks) -------------------------

STRESS_SCENARIOS: dict[str, dict[str, float]] = {
    "GFC_2008": {
        "equity": -0.52,   # S&P 500 peak-to-trough 2007–09
        "credit": -0.35,
        "commodity": -0.60,
        "crypto": -0.80,   # hypothetical for illustration
    },
    "COVID_2020": {
        "equity": -0.34,   # S&P 500 Feb–Mar 2020
        "credit": -0.20,
        "commodity": -0.35,
        "crypto": -0.60,   # BTC ~-60% Mar 2020
    },
    "CRYPTO_CRASH_2022": {
        "equity": -0.20,   # 2022 bear market
        "credit": -0.10,
        "commodity": +0.20,
        "crypto": -0.77,   # BTC ~-77%
    },
    "RATE_SHOCK_2023": {
        "equity": -0.15,
        "credit": -0.15,
        "commodity": -0.10,
        "crypto": -0.30,
    },
    "CUSTOM": {
        "equity": -0.30,
        "credit": -0.20,
        "commodity": -0.20,
        "crypto": -0.50,
    },
}


# ---- Data classes ---------------------------------------------------------

@dataclass
class VaRResult:
    method: str
    confidence: float
    horizon: int                # days
    var_pct: float              # as decimal, e.g. -0.03 = -3%
    var_usd: float
    cvar_pct: float
    cvar_usd: float


@dataclass
class MarginalVaR:
    symbol: str
    marginal_var: float         # contribution to portfolio VaR
    component_var: float        # marginal_var × weight
    component_cvar: float
    weight: float
    beta_vs_portfolio: float


@dataclass
class EVTResult:
    threshold: float
    shape_xi: float             # GPD shape parameter
    scale_sigma: float          # GPD scale parameter
    return_level_99: float      # 99th percentile return level
    return_level_999: float
    n_exceedances: int
    fit_quality: float          # KS test p-value


@dataclass
class StressTestResult:
    scenario: str
    portfolio_shock: float      # estimated portfolio P&L
    position_shocks: dict[str, float]   # per-symbol estimated shock
    worst_position: str
    worst_shock: float


@dataclass
class DailyRiskReport:
    timestamp: datetime
    portfolio_value: float
    var_95_1d: VaRResult
    var_99_1d: VaRResult
    var_99_10d: VaRResult
    cvar_99_1d: VaRResult
    cornish_fisher_cvar: VaRResult
    evt_result: Optional[EVTResult]
    marginal_vars: list[MarginalVaR]
    stress_results: list[StressTestResult]
    skewness: float
    kurtosis: float
    exceeds_var_today: bool


# ---- GARCH vol estimate --------------------------------------------------

def _garch_ewma_vol(returns: np.ndarray, halflife: int = 20) -> np.ndarray:
    """Simple EWMA variance estimate (GARCH(1,1) special case)."""
    alpha = 1 - np.exp(-np.log(2) / halflife)
    var = np.zeros(len(returns))
    var[0] = returns[0] ** 2
    for t in range(1, len(returns)):
        var[t] = alpha * returns[t - 1] ** 2 + (1 - alpha) * var[t - 1]
    return np.sqrt(np.maximum(var, 1e-10))


# ---- Cornish-Fisher expansion --------------------------------------------

def cornish_fisher_quantile(p: float, mu: float, sigma: float, skew: float, kurt_excess: float) -> float:
    """
    Cornish-Fisher expansion for quantile of a distribution with
    given first four moments.
    """
    z = stats.norm.ppf(p)
    # CF correction
    z_cf = (
        z
        + (z ** 2 - 1) * skew / 6.0
        + (z ** 3 - 3 * z) * kurt_excess / 24.0
        - (2 * z ** 3 - 5 * z) * skew ** 2 / 36.0
    )
    return mu + sigma * z_cf


# ---- GPD fitting ---------------------------------------------------------

def _fit_gpd(exceedances: np.ndarray) -> tuple[float, float]:
    """
    Fit Generalised Pareto Distribution to tail exceedances via MLE.
    Returns (shape_xi, scale_sigma).
    """
    if len(exceedances) < 10:
        return 0.0, float(np.std(exceedances))

    def neg_log_likelihood(params: np.ndarray) -> float:
        xi, sigma = params
        if sigma <= 0:
            return 1e10
        if xi == 0:
            ll = -np.sum(stats.expon.logpdf(exceedances, scale=sigma))
        else:
            z = 1 + xi * exceedances / sigma
            if np.any(z <= 0):
                return 1e10
            ll = -np.sum(stats.genpareto.logpdf(exceedances, c=xi, scale=sigma))
        return ll

    # Method of moments starting point
    m1 = np.mean(exceedances)
    m2 = np.var(exceedances)
    xi0 = 0.5 * (m1 ** 2 / m2 - 1)
    sigma0 = 0.5 * m1 * (m1 ** 2 / m2 + 1)

    result = minimize(
        neg_log_likelihood,
        x0=[xi0, max(sigma0, 0.001)],
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-6},
    )
    xi, sigma = result.x
    return float(xi), float(max(sigma, 1e-8))


# ---- Main class ----------------------------------------------------------

class TailRiskMonitor:
    """
    Comprehensive tail risk system supporting:
    - Historical simulation VaR/CVaR (1d/5d/10d)
    - Parametric VaR (normal, Student-t, skewed-t)
    - Filtered historical simulation (GARCH-filtered)
    - Cornish-Fisher adjusted CVaR
    - Extreme value theory (GPD)
    - Stress scenarios
    - Marginal/Component VaR
    - Daily risk report generation
    """

    def __init__(
        self,
        symbols: list[str],
        positions: dict[str, float] | None = None,    # symbol → USD notional
        portfolio_value: float = 1_000_000.0,
        confidence_levels: list[float] = None,
        horizons: list[int] = None,
    ):
        self.symbols = list(symbols)
        self.positions = positions or {s: portfolio_value / len(symbols) for s in symbols}
        self.portfolio_value = portfolio_value

        self.confidence_levels = confidence_levels or CONFIDENCE_LEVELS
        self.horizons = horizons or HORIZONS

        self._returns: dict[str, list[float]] = {s: [] for s in symbols}
        self._portfolio_returns: list[float] = []
        self._timestamps: list[datetime] = []
        self._reports: list[DailyRiskReport] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update(
        self,
        symbol_returns: dict[str, float],
        positions: dict[str, float] | None = None,
    ) -> None:
        """Feed one bar of returns. Optionally update position sizes."""
        if positions is not None:
            self.positions = positions
            self.portfolio_value = sum(abs(v) for v in positions.values())

        for s in self.symbols:
            self._returns[s].append(symbol_returns.get(s, 0.0))

        # Portfolio return = weighted sum
        pf_r = sum(
            symbol_returns.get(s, 0.0) * abs(self.positions.get(s, 0.0)) / max(self.portfolio_value, 1.0)
            for s in self.symbols
        )
        self._portfolio_returns.append(pf_r)
        self._timestamps.append(datetime.now(tz=timezone.utc))

    # ------------------------------------------------------------------
    # VaR methods
    # ------------------------------------------------------------------

    def historical_var(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        n_obs: int = 252,
    ) -> VaRResult:
        """Historical simulation VaR/CVaR."""
        pf_r = np.array(self._portfolio_returns[-n_obs:])
        if len(pf_r) < 30:
            return self._empty_var("historical", confidence, horizon)

        # Scale to horizon
        pf_r_h = pf_r * np.sqrt(horizon)

        cutoff = np.percentile(pf_r_h, (1 - confidence) * 100)
        tail = pf_r_h[pf_r_h <= cutoff]
        cvar = float(np.mean(tail)) if len(tail) > 0 else cutoff

        return VaRResult(
            method="historical",
            confidence=confidence,
            horizon=horizon,
            var_pct=float(cutoff),
            var_usd=float(cutoff * self.portfolio_value),
            cvar_pct=float(cvar),
            cvar_usd=float(cvar * self.portfolio_value),
        )

    def parametric_var(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        distribution: str = "student_t",
    ) -> VaRResult:
        """
        Parametric VaR.
        distribution: 'normal' | 'student_t' | 'skewed_t'
        """
        pf_r = np.array(self._portfolio_returns[-252:])
        if len(pf_r) < 20:
            return self._empty_var(f"parametric_{distribution}", confidence, horizon)

        mu = float(np.mean(pf_r))
        sigma = float(np.std(pf_r))
        skew = float(stats.skew(pf_r))
        kurt = float(stats.kurtosis(pf_r))

        if distribution == "normal":
            z = stats.norm.ppf(1 - confidence)
            var_pct = mu + sigma * z
            # CVaR for normal
            phi_z = stats.norm.pdf(-z)
            cvar_pct = mu - sigma * phi_z / (1 - confidence)

        elif distribution == "student_t":
            # Fit degrees of freedom
            df_fit, loc_fit, scale_fit = stats.t.fit(pf_r)
            df_fit = max(df_fit, 2.1)
            z = stats.t.ppf(1 - confidence, df=df_fit)
            var_pct = float(loc_fit + scale_fit * z)
            # CVaR for t
            t_pdf_z = stats.t.pdf(z, df=df_fit)
            cvar_pct = float(
                loc_fit - scale_fit * (df_fit + z ** 2) / (df_fit - 1) * t_pdf_z / (1 - confidence)
            )

        else:  # skewed_t — use Cornish-Fisher
            z = stats.norm.ppf(1 - confidence)
            cf_q = cornish_fisher_quantile(1 - confidence, mu, sigma, skew, kurt)
            var_pct = cf_q
            # CVaR via numerical integration
            tail_pts = np.linspace(0.0001, 1 - confidence, 200)
            tail_q = np.array([
                cornish_fisher_quantile(p, mu, sigma, skew, kurt)
                for p in tail_pts
            ])
            cvar_pct = float(np.mean(tail_q))

        # Scale to horizon
        var_h = var_pct * np.sqrt(horizon)
        cvar_h = cvar_pct * np.sqrt(horizon)

        return VaRResult(
            method=f"parametric_{distribution}",
            confidence=confidence,
            horizon=horizon,
            var_pct=var_h,
            var_usd=var_h * self.portfolio_value,
            cvar_pct=cvar_h,
            cvar_usd=cvar_h * self.portfolio_value,
        )

    def filtered_historical_var(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> VaRResult:
        """
        Filtered Historical Simulation:
        1. Estimate GARCH vol
        2. Standardise returns: z_t = r_t / sigma_t
        3. Scale standardised residuals by current vol forecast
        """
        pf_r = np.array(self._portfolio_returns[-252:])
        if len(pf_r) < 40:
            return self.historical_var(confidence, horizon)

        vol_series = _garch_ewma_vol(pf_r)
        current_vol = vol_series[-1]
        std_residuals = pf_r / vol_series

        # Resample current distribution
        scaled = std_residuals * current_vol * np.sqrt(horizon)
        cutoff = np.percentile(scaled, (1 - confidence) * 100)
        tail = scaled[scaled <= cutoff]
        cvar = float(np.mean(tail)) if len(tail) > 0 else cutoff

        return VaRResult(
            method="filtered_historical",
            confidence=confidence,
            horizon=horizon,
            var_pct=float(cutoff),
            var_usd=float(cutoff * self.portfolio_value),
            cvar_pct=float(cvar),
            cvar_usd=float(cvar * self.portfolio_value),
        )

    def cornish_fisher_cvar(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> VaRResult:
        """CVaR adjusted for skewness and excess kurtosis via Cornish-Fisher."""
        pf_r = np.array(self._portfolio_returns[-252:])
        if len(pf_r) < 20:
            return self._empty_var("cornish_fisher", confidence, horizon)

        mu = float(np.mean(pf_r))
        sigma = float(np.std(pf_r))
        skew = float(stats.skew(pf_r))
        kurt_excess = float(stats.kurtosis(pf_r))  # excess kurtosis

        # Compute CF quantile
        var_pct = cornish_fisher_quantile(1 - confidence, mu, sigma, skew, kurt_excess)

        # CVaR = average of tail losses beyond VaR
        tail_probs = np.linspace(0.0001, 1 - confidence, 500)
        tail_quantiles = np.array([
            cornish_fisher_quantile(p, mu, sigma, skew, kurt_excess)
            for p in tail_probs
        ])
        cvar_pct = float(np.mean(tail_quantiles))

        var_h = var_pct * np.sqrt(horizon)
        cvar_h = cvar_pct * np.sqrt(horizon)

        return VaRResult(
            method="cornish_fisher",
            confidence=confidence,
            horizon=horizon,
            var_pct=var_h,
            var_usd=var_h * self.portfolio_value,
            cvar_pct=cvar_h,
            cvar_usd=cvar_h * self.portfolio_value,
        )

    # ------------------------------------------------------------------
    # Extreme Value Theory
    # ------------------------------------------------------------------

    def extreme_value_theory(
        self,
        threshold_pct: float = GPD_THRESHOLD_PCT,
    ) -> Optional[EVTResult]:
        """
        Fit Generalised Pareto Distribution to the left tail.
        Returns EVT result with shape/scale parameters and return levels.
        """
        pf_r = np.array(self._portfolio_returns)
        if len(pf_r) < 50:
            return None

        # Negative returns → losses
        losses = -pf_r
        threshold = np.quantile(losses, threshold_pct)
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 10:
            return None

        xi, sigma = _fit_gpd(exceedances)

        n = len(losses)
        n_u = len(exceedances)

        def return_level(p: float) -> float:
            """Return level for exceedance probability p."""
            if xi == 0:
                return threshold + sigma * np.log(n / (n_u * p))
            else:
                return threshold + (sigma / xi) * ((n / (n_u * p)) ** xi - 1)

        rl_99 = return_level(0.01)
        rl_999 = return_level(0.001)

        # Goodness-of-fit: KS test
        try:
            gpd_samples = stats.genpareto.rvs(c=xi, scale=sigma, size=len(exceedances), random_state=0)
            ks_stat, ks_p = stats.ks_2samp(exceedances, gpd_samples)
            fit_quality = float(ks_p)
        except Exception:
            fit_quality = 0.0

        return EVTResult(
            threshold=float(threshold),
            shape_xi=float(xi),
            scale_sigma=float(sigma),
            return_level_99=float(rl_99),
            return_level_999=float(rl_999),
            n_exceedances=len(exceedances),
            fit_quality=fit_quality,
        )

    # ------------------------------------------------------------------
    # Stress testing
    # ------------------------------------------------------------------

    def stress_test(
        self,
        scenario_name: str = "GFC_2008",
        custom_shocks: dict[str, float] | None = None,
        symbol_sectors: dict[str, str] | None = None,
    ) -> StressTestResult:
        """
        Apply historical stress shocks to current portfolio.

        symbol_sectors: {symbol: 'equity'|'credit'|'commodity'|'crypto'}
        If not provided, defaults to 'crypto' for all.
        """
        if scenario_name == "CUSTOM" and custom_shocks:
            shocks = custom_shocks
        else:
            shocks = STRESS_SCENARIOS.get(scenario_name, STRESS_SCENARIOS["GFC_2008"])

        sector_map = symbol_sectors or {s: "crypto" for s in self.symbols}

        position_shocks: dict[str, float] = {}
        portfolio_pnl = 0.0

        for s in self.symbols:
            sector = sector_map.get(s, "crypto")
            shock_pct = shocks.get(sector, shocks.get("equity", -0.30))
            pos_usd = self.positions.get(s, 0.0)
            pnl = pos_usd * shock_pct
            position_shocks[s] = pnl
            portfolio_pnl += pnl

        worst_sym = min(position_shocks, key=lambda k: position_shocks[k])
        worst_shock = position_shocks[worst_sym]

        return StressTestResult(
            scenario=scenario_name,
            portfolio_shock=portfolio_pnl,
            position_shocks=position_shocks,
            worst_position=worst_sym,
            worst_shock=worst_shock,
        )

    def run_all_stress_tests(
        self,
        symbol_sectors: dict[str, str] | None = None,
    ) -> list[StressTestResult]:
        results = []
        for scenario in STRESS_SCENARIOS:
            results.append(self.stress_test(scenario, symbol_sectors=symbol_sectors))
        return results

    # ------------------------------------------------------------------
    # Marginal and Component VaR
    # ------------------------------------------------------------------

    def marginal_var(
        self,
        confidence: float = 0.99,
        bump_pct: float = 0.01,
    ) -> list[MarginalVaR]:
        """
        Compute marginal VaR for each position by bumping its weight.
        Component VaR = w_i × marginal VaR_i
        """
        pf_r = np.array(self._portfolio_returns[-252:])
        if len(pf_r) < 30:
            return []

        # Base portfolio VaR
        base_var = float(np.percentile(pf_r, (1 - confidence) * 100))
        base_cvar_tail = pf_r[pf_r <= base_var]
        base_cvar = float(np.mean(base_cvar_tail)) if len(base_cvar_tail) > 0 else base_var

        results = []
        total_pv = max(self.portfolio_value, 1.0)

        for s in self.symbols:
            sym_r = np.array(self._returns[s][-252:])
            if len(sym_r) < len(pf_r):
                sym_r = np.pad(sym_r, (len(pf_r) - len(sym_r), 0), constant_values=0.0)

            w = abs(self.positions.get(s, 0.0)) / total_pv
            bumped_pf_r = pf_r + bump_pct * sym_r * w

            bumped_var = float(np.percentile(bumped_pf_r, (1 - confidence) * 100))
            marginal = (bumped_var - base_var) / bump_pct

            component_var = marginal * w
            component_cvar_bump = bumped_pf_r[bumped_pf_r <= bumped_var]
            component_cvar = (
                float(np.mean(component_cvar_bump)) * w if len(component_cvar_bump) > 0 else base_cvar * w
            )

            # Beta vs portfolio
            cov = np.cov(sym_r[-len(pf_r):], pf_r)
            pf_var_scalar = float(np.var(pf_r))
            beta_pf = float(cov[0, 1] / pf_var_scalar) if pf_var_scalar > 0 else 0.0

            results.append(MarginalVaR(
                symbol=s,
                marginal_var=marginal,
                component_var=component_var,
                component_cvar=component_cvar,
                weight=w,
                beta_vs_portfolio=beta_pf,
            ))

        return results

    # ------------------------------------------------------------------
    # Full risk report
    # ------------------------------------------------------------------

    def generate_daily_report(
        self,
        symbol_sectors: dict[str, str] | None = None,
    ) -> DailyRiskReport:
        """Generate comprehensive daily risk report."""
        pf_r = np.array(self._portfolio_returns)
        today_r = pf_r[-1] if len(pf_r) > 0 else 0.0

        var95 = self.historical_var(0.95, 1)
        var99 = self.historical_var(0.99, 1)
        var99_10d = self.historical_var(0.99, 10)
        cvar99 = self.filtered_historical_var(0.99, 1)
        cf_cvar = self.cornish_fisher_cvar(0.99, 1)
        evt = self.extreme_value_theory()
        mv = self.marginal_var(0.99)
        stress_results = self.run_all_stress_tests(symbol_sectors)

        skew = float(stats.skew(pf_r)) if len(pf_r) > 3 else 0.0
        kurt = float(stats.kurtosis(pf_r)) if len(pf_r) > 3 else 0.0

        report = DailyRiskReport(
            timestamp=datetime.now(tz=timezone.utc),
            portfolio_value=self.portfolio_value,
            var_95_1d=var95,
            var_99_1d=var99,
            var_99_10d=var99_10d,
            cvar_99_1d=cvar99,
            cornish_fisher_cvar=cf_cvar,
            evt_result=evt,
            marginal_vars=mv,
            stress_results=stress_results,
            skewness=skew,
            kurtosis=kurt,
            exceeds_var_today=today_r < var99.var_pct,
        )
        self._reports.append(report)

        logger.info(
            "Risk report: VaR(99,1d)=%.2f%%, CVaR=%.2f%%, skew=%.2f, kurt=%.2f",
            var99.var_pct * 100,
            cvar99.cvar_pct * 100,
            skew,
            kurt,
        )
        return report

    def _empty_var(self, method: str, confidence: float, horizon: int) -> VaRResult:
        return VaRResult(
            method=method,
            confidence=confidence,
            horizon=horizon,
            var_pct=0.0,
            var_usd=0.0,
            cvar_pct=0.0,
            cvar_usd=0.0,
        )

    def summary(self) -> dict[str, Any]:
        if not self._portfolio_returns:
            return {"status": "no data"}
        pf_r = np.array(self._portfolio_returns[-252:])
        return {
            "n_obs": len(pf_r),
            "ann_vol": float(np.std(pf_r) * np.sqrt(252)),
            "skewness": float(stats.skew(pf_r)) if len(pf_r) > 3 else 0.0,
            "excess_kurtosis": float(stats.kurtosis(pf_r)) if len(pf_r) > 3 else 0.0,
            "var_99_1d_pct": float(np.percentile(pf_r, 1)) * 100,
            "n_reports": len(self._reports),
        }


# ---- Standalone test -------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)

    symbols = ["BTC", "ETH", "SOL"]
    positions = {"BTC": 500_000, "ETH": 300_000, "SOL": 200_000}
    monitor = TailRiskMonitor(symbols, positions, portfolio_value=1_000_000)

    for _ in range(300):
        rets = {s: float(rng.normal(0.0002, 0.025)) for s in symbols}
        monitor.update(rets)

    report = monitor.generate_daily_report()
    print(f"VaR(99,1d): {report.var_99_1d.var_pct*100:.2f}%  ${report.var_99_1d.var_usd:,.0f}")
    print(f"CVaR(99,1d): {report.cvar_99_1d.cvar_pct*100:.2f}%")
    print(f"Cornish-Fisher CVaR: {report.cornish_fisher_cvar.cvar_pct*100:.2f}%")
    if report.evt_result:
        print(f"EVT 99% return level: {report.evt_result.return_level_99*100:.2f}%")
    print("Stress — GFC:", report.stress_results[0].portfolio_shock)
    print(f"Skew: {report.skewness:.3f}  Excess Kurt: {report.kurtosis:.3f}")
