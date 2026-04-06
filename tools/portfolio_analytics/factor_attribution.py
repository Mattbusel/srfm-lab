# ============================================================
# factor_attribution.py
# Factor attribution system for portfolio returns
# ============================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

# ---- Factor definitions --------------------------------------------------

# Traditional equity factors
FF_FACTORS = ["market", "smb", "hml", "mom", "quality", "low_vol"]

# Crypto-specific factors
CRYPTO_FACTORS = ["btc_beta", "eth_beta", "defi_factor"]

ALL_FACTORS = FF_FACTORS + CRYPTO_FACTORS

# Factor exposure alert threshold
FACTOR_BETA_LIMIT = 0.5

# Rolling OLS window (bars)
ROLLING_WINDOW = 60


# ---- Data classes ---------------------------------------------------------

@dataclass
class FactorLoading:
    symbol: str
    timestamp: datetime
    alpha: float
    betas: dict[str, float]     # factor_name → beta
    r_squared: float
    residual_std: float
    t_stats: dict[str, float]
    p_values: dict[str, float]


@dataclass
class ReturnAttribution:
    symbol: str
    timestamp: datetime
    total_return: float
    systematic_return: float    # sum of factor contributions
    idiosyncratic_return: float # alpha + residual
    factor_contributions: dict[str, float]  # factor_name → contribution
    factor_betas: dict[str, float]
    r_squared: float


@dataclass
class AttributionReport:
    period: str                     # 'daily' | 'weekly' | 'inception'
    start_date: datetime
    end_date: datetime
    portfolio_return: float
    benchmark_return: float
    active_return: float
    factor_contributions: dict[str, float]
    systematic_pct: float
    idiosyncratic_pct: float
    top_contributors: list[tuple[str, float]]
    bottom_contributors: list[tuple[str, float]]
    factor_alerts: list[str]


@dataclass
class FactorExposureAlert:
    timestamp: datetime
    symbol: str
    factor: str
    beta: float
    limit: float
    message: str


# ---- Live factor proxies --------------------------------------------------

class LiveFactorProxies:
    """
    Tracks live proxy instruments for factor return series.

    Equity factors:
      market   → SPY
      smb      → IWM - SPY spread proxy
      hml      → value basket proxy
      mom      → momentum basket
      quality  → quality basket
      low_vol  → SPLV

    Crypto factors:
      btc_beta → BTC/USDT
      eth_beta → ETH/USDT
      defi_factor → (AAVE + UNI + LINK) / 3 basket
    """

    def __init__(self) -> None:
        self._factor_returns: dict[str, list[float]] = {f: [] for f in ALL_FACTORS}
        self._dates: list[datetime] = []
        self._cumulative: dict[str, float] = {f: 0.0 for f in ALL_FACTORS}

    def update(self, proxy_prices: dict[str, float], prev_prices: dict[str, float]) -> dict[str, float]:
        """
        Compute factor returns from proxy price updates.

        proxy_prices keys expected:
          SPY, IWM, SPLV, BTC, ETH, AAVE, UNI, LINK
          + optional: HML_PROXY, MOM_PROXY, QUALITY_PROXY
        """
        def ret(sym: str) -> float:
            p0 = prev_prices.get(sym, np.nan)
            p1 = proxy_prices.get(sym, np.nan)
            if p0 and p1 and p0 > 0:
                return (p1 - p0) / p0
            return 0.0

        spy_r = ret("SPY")
        iwm_r = ret("IWM")
        splv_r = ret("SPLV")
        btc_r = ret("BTC")
        eth_r = ret("ETH")
        aave_r = ret("AAVE")
        uni_r = ret("UNI")
        link_r = ret("LINK")

        factor_rets = {
            "market":     spy_r,
            "smb":        iwm_r - spy_r,                       # small minus big proxy
            "hml":        proxy_prices.get("HML_RETURN", 0.0), # externally supplied
            "mom":        proxy_prices.get("MOM_RETURN", 0.0),
            "quality":    proxy_prices.get("QUALITY_RETURN", 0.0),
            "low_vol":    splv_r - spy_r,                       # low-vol factor
            "btc_beta":   btc_r,
            "eth_beta":   eth_r - btc_r,                        # ETH-specific (residual)
            "defi_factor": (aave_r + uni_r + link_r) / 3.0,
        }

        for f, r in factor_rets.items():
            self._factor_returns[f].append(r)
            self._cumulative[f] = self._cumulative.get(f, 0.0) + r

        self._dates.append(datetime.now(tz=timezone.utc))
        return factor_rets

    def as_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._factor_returns, index=self._dates)
        df.index.name = "timestamp"
        return df

    def get_latest(self) -> dict[str, float]:
        return {f: (v[-1] if v else 0.0) for f, v in self._factor_returns.items()}


# ---- Rolling OLS ---------------------------------------------------------

def _rolling_ols(
    y: np.ndarray,
    X: np.ndarray,
    factor_names: list[str],
) -> tuple[float, dict[str, float], float, float, dict[str, float], dict[str, float]]:
    """
    OLS regression: y = alpha + X @ beta + eps
    Returns (alpha, betas, r2, resid_std, t_stats, p_values)
    """
    n = len(y)
    k = X.shape[1]
    X_aug = np.column_stack([np.ones(n), X])

    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
    except np.linalg.LinAlgError:
        alpha_v = 0.0
        betas = {f: 0.0 for f in factor_names}
        return alpha_v, betas, 0.0, np.std(y), {f: 0.0 for f in factor_names}, {f: 1.0 for f in factor_names}

    alpha_v = float(coeffs[0])
    beta_vals = coeffs[1:]

    y_hat = X_aug @ coeffs
    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    resid_std = float(np.std(resid))

    # t-statistics
    dof = n - k - 1
    if dof > 0 and ss_res > 0:
        mse = ss_res / dof
        try:
            cov_mat = mse * np.linalg.pinv(X_aug.T @ X_aug)
            se = np.sqrt(np.maximum(np.diag(cov_mat), 1e-12))
            t_full = coeffs / se
        except Exception:
            t_full = np.zeros(k + 1)
    else:
        t_full = np.zeros(k + 1)
        dof = 1

    t_stats = {f: float(t_full[i + 1]) for i, f in enumerate(factor_names)}
    p_values = {f: float(2 * (1 - stats.t.cdf(abs(t_full[i + 1]), df=dof))) for i, f in enumerate(factor_names)}

    betas = {f: float(beta_vals[i]) for i, f in enumerate(factor_names)}
    return alpha_v, betas, r2, resid_std, t_stats, p_values


# ---- FactorModel ----------------------------------------------------------

class FactorModel:
    """
    Fama-French 3/5 + Momentum + Quality + Low-Vol + Crypto factors.
    Maintains rolling 60-bar OLS loadings for each symbol.
    """

    def __init__(
        self,
        symbols: list[str],
        factors: list[str] | None = None,
        rolling_window: int = ROLLING_WINDOW,
        beta_limit: float = FACTOR_BETA_LIMIT,
    ):
        self.symbols = list(symbols)
        self.factors = factors if factors is not None else ALL_FACTORS
        self.rolling_window = rolling_window
        self.beta_limit = beta_limit

        # History buffers
        self._symbol_returns: dict[str, list[float]] = {s: [] for s in symbols}
        self._factor_returns: dict[str, list[float]] = {f: [] for f in self.factors}
        self._loadings: dict[str, FactorLoading] = {}
        self._alerts: list[FactorExposureAlert] = []

    def update(
        self,
        symbol_returns: dict[str, float],
        factor_returns: dict[str, float],
    ) -> dict[str, FactorLoading]:
        """
        Feed one bar of symbol and factor returns.
        Returns updated loadings for all symbols.
        """
        ts = datetime.now(tz=timezone.utc)

        for s in self.symbols:
            self._symbol_returns[s].append(symbol_returns.get(s, np.nan))
        for f in self.factors:
            self._factor_returns[f].append(factor_returns.get(f, 0.0))

        if len(self._factor_returns[self.factors[0]]) < 20:
            return {}

        n_bars = len(self._factor_returns[self.factors[0]])
        start = max(0, n_bars - self.rolling_window)

        F = np.array([self._factor_returns[f][start:] for f in self.factors]).T

        for s in self.symbols:
            y_raw = np.array(self._symbol_returns[s][start:])
            # Skip if too many NaNs
            valid = ~np.isnan(y_raw)
            if valid.sum() < 15:
                continue
            y = y_raw[valid]
            Fv = F[valid]

            alpha_v, betas, r2, resid_std, t_stats, p_values = _rolling_ols(y, Fv, self.factors)

            loading = FactorLoading(
                symbol=s,
                timestamp=ts,
                alpha=alpha_v,
                betas=betas,
                r_squared=r2,
                residual_std=resid_std,
                t_stats=t_stats,
                p_values=p_values,
            )
            self._loadings[s] = loading
            self._check_exposure_alerts(loading)

        return dict(self._loadings)

    def _check_exposure_alerts(self, loading: FactorLoading) -> None:
        for f, b in loading.betas.items():
            if abs(b) > self.beta_limit:
                alert = FactorExposureAlert(
                    timestamp=loading.timestamp,
                    symbol=loading.symbol,
                    factor=f,
                    beta=b,
                    limit=self.beta_limit,
                    message=(
                        f"{loading.symbol}: {f} beta = {b:.3f} exceeds limit {self.beta_limit}"
                    ),
                )
                self._alerts.append(alert)
                logger.warning("Factor exposure alert: %s", alert.message)

    def get_loadings(self) -> dict[str, FactorLoading]:
        return dict(self._loadings)

    def get_alerts(self, n: int = 20) -> list[FactorExposureAlert]:
        return self._alerts[-n:]


# ---- FactorAttributionEngine ----------------------------------------------

class FactorAttributionEngine:
    """
    Decomposes portfolio returns into systematic (factor) and
    idiosyncratic components, generates attribution reports.
    """

    def __init__(
        self,
        symbols: list[str],
        weights: dict[str, float] | None = None,
        factors: list[str] | None = None,
        rolling_window: int = ROLLING_WINDOW,
    ):
        self.symbols = list(symbols)
        self.weights = weights or {s: 1.0 / len(symbols) for s in symbols}
        self.factors = factors if factors is not None else ALL_FACTORS

        self.factor_model = FactorModel(symbols, self.factors, rolling_window)
        self.proxy = LiveFactorProxies()

        self._attribution_history: list[ReturnAttribution] = []
        self._prev_proxy_prices: dict[str, float] = {}

    def update(
        self,
        symbol_returns: dict[str, float],
        proxy_prices: dict[str, float],
    ) -> list[ReturnAttribution]:
        """
        Feed one bar. Returns per-symbol attribution.
        """
        factor_rets = self.proxy.update(proxy_prices, self._prev_proxy_prices)
        self._prev_proxy_prices = dict(proxy_prices)

        loadings = self.factor_model.update(symbol_returns, factor_rets)
        ts = datetime.now(tz=timezone.utc)
        attributions = []

        for s in self.symbols:
            r = symbol_returns.get(s, 0.0)
            loading = loadings.get(s)
            if loading is None:
                continue

            factor_contributions = {}
            systematic_return = loading.alpha
            for f, b in loading.betas.items():
                contrib = b * factor_rets.get(f, 0.0)
                factor_contributions[f] = contrib
                systematic_return += contrib

            idiosyncratic_return = r - systematic_return

            attr = ReturnAttribution(
                symbol=s,
                timestamp=ts,
                total_return=r,
                systematic_return=systematic_return,
                idiosyncratic_return=idiosyncratic_return,
                factor_contributions=factor_contributions,
                factor_betas=dict(loading.betas),
                r_squared=loading.r_squared,
            )
            self._attribution_history.append(attr)
            attributions.append(attr)

        return attributions

    def generate_report(
        self,
        period: str = "daily",
        benchmark_returns: Optional[pd.Series] = None,
    ) -> AttributionReport:
        """
        Generate an attribution report for the requested period.
        period: 'daily' | 'weekly' | 'inception'
        """
        if not self._attribution_history:
            raise ValueError("No attribution history available")

        now = datetime.now(tz=timezone.utc)
        if period == "daily":
            cutoff = pd.Timestamp(now).floor("D")
        elif period == "weekly":
            cutoff = pd.Timestamp(now) - pd.Timedelta(days=7)
        else:
            cutoff = pd.Timestamp.min.tz_localize("UTC")

        relevant = [
            a for a in self._attribution_history
            if pd.Timestamp(a.timestamp) >= cutoff
        ]

        if not relevant:
            relevant = self._attribution_history[-100:]

        # Aggregate factor contributions (portfolio-weighted)
        agg_factor: dict[str, float] = {f: 0.0 for f in self.factors}
        total_portfolio_return = 0.0
        systematic_sum = 0.0
        idio_sum = 0.0

        by_symbol: dict[str, list[float]] = {s: [] for s in self.symbols}

        for attr in relevant:
            w = self.weights.get(attr.symbol, 0.0)
            for f, c in attr.factor_contributions.items():
                agg_factor[f] = agg_factor.get(f, 0.0) + c * w
            total_portfolio_return += attr.total_return * w
            systematic_sum += attr.systematic_return * w
            idio_sum += attr.idiosyncratic_return * w
            by_symbol[attr.symbol].append(attr.total_return * w)

        # Top/bottom contributors
        sym_totals = [(s, sum(v)) for s, v in by_symbol.items() if v]
        sym_totals.sort(key=lambda x: x[1], reverse=True)
        top_contributors = sym_totals[:5]
        bottom_contributors = sym_totals[-5:]

        # Factor alerts
        factor_alerts: list[str] = []
        all_loadings = self.factor_model.get_loadings()
        for s, loading in all_loadings.items():
            for f, b in loading.betas.items():
                if abs(b) > FACTOR_BETA_LIMIT:
                    factor_alerts.append(f"{s}: {f} beta={b:.2f} > {FACTOR_BETA_LIMIT}")

        total_abs = abs(systematic_sum) + abs(idio_sum)
        sys_pct = (abs(systematic_sum) / total_abs * 100) if total_abs > 0 else 0.0
        idio_pct = 100.0 - sys_pct

        bm_return = float(benchmark_returns.mean()) if benchmark_returns is not None else 0.0
        start_date = min(a.timestamp for a in relevant)
        end_date = max(a.timestamp for a in relevant)

        return AttributionReport(
            period=period,
            start_date=start_date,
            end_date=end_date,
            portfolio_return=total_portfolio_return,
            benchmark_return=bm_return,
            active_return=total_portfolio_return - bm_return,
            factor_contributions=agg_factor,
            systematic_pct=sys_pct,
            idiosyncratic_pct=idio_pct,
            top_contributors=top_contributors,
            bottom_contributors=bottom_contributors,
            factor_alerts=factor_alerts,
        )

    def plot_stacked_contributions(
        self,
        n_bars: int = 30,
        output_path: str = "factor_attribution.html",
    ) -> None:
        """Stacked bar chart of factor contributions over time."""
        if not HAS_PLOTLY:
            logger.warning("plotly not installed; cannot plot factor contributions")
            return

        history = self._attribution_history[-n_bars:]
        if not history:
            return

        timestamps = [a.timestamp.strftime("%m-%d %H:%M") for a in history]
        fig = go.Figure()

        for f in self.factors:
            values = [a.factor_contributions.get(f, 0.0) for a in history]
            fig.add_trace(go.Bar(name=f, x=timestamps, y=values))

        # Idiosyncratic as separate series
        idio_vals = [a.idiosyncratic_return for a in history]
        fig.add_trace(go.Bar(name="idiosyncratic", x=timestamps, y=idio_vals))

        fig.update_layout(
            barmode="relative",
            title="Factor Contribution Attribution",
            xaxis_title="Bar",
            yaxis_title="Return (%)",
            legend=dict(orientation="h"),
        )
        fig.write_html(output_path)
        logger.info("Factor attribution chart written to %s", output_path)

    def factor_exposure_summary(self) -> pd.DataFrame:
        """Return a DataFrame of current factor betas for all symbols."""
        rows = []
        for s, loading in self.factor_model.get_loadings().items():
            row = {"symbol": s, "alpha": loading.alpha, "r_squared": loading.r_squared}
            row.update(loading.betas)
            rows.append(row)
        return pd.DataFrame(rows).set_index("symbol")

    def get_crypto_factor_exposures(self) -> pd.DataFrame:
        """Convenience method for crypto-specific factor betas."""
        df = self.factor_exposure_summary()
        crypto_cols = [f for f in CRYPTO_FACTORS if f in df.columns]
        return df[crypto_cols]


# ---- Standalone test -------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    symbols = ["BTC", "ETH", "SOL", "BNB", "AVAX"]
    engine = FactorAttributionEngine(symbols)

    rng = np.random.default_rng(1)
    proxy_prices: dict[str, float] = {
        "SPY": 500.0, "IWM": 200.0, "SPLV": 60.0,
        "BTC": 60000.0, "ETH": 3000.0, "AAVE": 100.0, "UNI": 8.0, "LINK": 15.0,
    }

    for bar in range(100):
        sym_rets = {s: float(rng.normal(0.0002, 0.02)) for s in symbols}
        for k in proxy_prices:
            proxy_prices[k] *= 1.0 + float(rng.normal(0.0001, 0.005))
        engine.update(sym_rets, proxy_prices)

    report = engine.generate_report("daily")
    print(f"Portfolio return: {report.portfolio_return:.4f}")
    print(f"Systematic: {report.systematic_pct:.1f}%  Idiosyncratic: {report.idiosyncratic_pct:.1f}%")
    print("Top contributors:", report.top_contributors)
    print("Factor alerts:", report.factor_alerts)
    engine.plot_stacked_contributions(output_path="factor_test.html")
