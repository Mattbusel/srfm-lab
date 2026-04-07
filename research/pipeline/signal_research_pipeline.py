# research/pipeline/signal_research_pipeline.py
# SRFM -- Automated signal research pipeline
# Runs a full evaluation: data -> signal -> IC -> decay -> regime -> capacity -> report

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ResearchResult:
    """Full research evaluation of a single signal function."""

    signal_name: str
    ic_mean: float
    ic_std: float
    icir: float

    # IC decay half-life in bars (trading days)
    ic_decay_halflife: float

    # Conditional IC by market regime
    regime_conditional_ic: Dict[str, float] = field(
        default_factory=lambda: {"bull": 0.0, "bear": 0.0, "ranging": 0.0}
    )

    # Mean fraction of portfolio turning over per day
    turnover_daily: float = 0.0

    # Estimated signal capacity in USD
    capacity_estimate_usd: float = 0.0

    # Statistical significance: ICIR > 0.5 and p-value < 0.05
    is_significant: bool = False

    # Bailey-Lopez deflated Sharpe ratio correcting for multiple testing
    deflated_sharpe: float = 0.0

    # PROMOTE / WATCH / RETIRE
    recommendation: str = "WATCH"

    # Supplementary diagnostics
    ic_series: Optional[pd.Series] = field(default=None, repr=False)
    p_value: float = 1.0
    n_observations: int = 0
    category: str = "unknown"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "signal_name": self.signal_name,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "icir": self.icir,
            "ic_decay_halflife": self.ic_decay_halflife,
            "regime_bull": self.regime_conditional_ic.get("bull", 0.0),
            "regime_bear": self.regime_conditional_ic.get("bear", 0.0),
            "regime_ranging": self.regime_conditional_ic.get("ranging", 0.0),
            "turnover_daily": self.turnover_daily,
            "capacity_estimate_usd": self.capacity_estimate_usd,
            "is_significant": self.is_significant,
            "deflated_sharpe": self.deflated_sharpe,
            "recommendation": self.recommendation,
            "p_value": self.p_value,
            "n_observations": self.n_observations,
            "category": self.category,
        }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class SignalResearchPipeline:
    """
    Automated signal research pipeline.

    Usage:
        pipeline = SignalResearchPipeline()
        result = pipeline.run(signal_fn, universe, "2018-01-01", "2023-12-31")
    """

    def __init__(
        self,
        forward_return_horizon: int = 5,
        ic_method: str = "spearman",
        regime_window: int = 63,
        capacity_adv_fraction: float = 0.05,
        n_trials_correction: int = 1,
    ) -> None:
        self.forward_return_horizon = forward_return_horizon
        self.ic_method = ic_method
        self.regime_window = regime_window
        # fraction of ADV we can trade without market impact
        self.capacity_adv_fraction = capacity_adv_fraction
        # number of strategies tested (for deflated Sharpe)
        self.n_trials_correction = n_trials_correction

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        signal_fn: Callable,
        universe: List[str],
        start: str,
        end: str,
        prices: Optional[pd.DataFrame] = None,
        volumes: Optional[pd.DataFrame] = None,
    ) -> ResearchResult:
        """
        Run the full evaluation pipeline for a single signal function.

        Parameters
        ----------
        signal_fn : callable
            Function with signature f(prices, volumes) -> pd.DataFrame of scores,
            indexed by date, columns = tickers.
        universe : list of str
            Ticker symbols.
        start, end : str
            Date range in "YYYY-MM-DD" format.
        prices, volumes : pd.DataFrame, optional
            If provided, use these instead of generating synthetic data.

        Returns
        -------
        ResearchResult
        """
        signal_name = getattr(signal_fn, "__name__", str(signal_fn))
        logger.info("Starting research pipeline for signal: %s", signal_name)

        # 1. Load / generate data
        if prices is None or volumes is None:
            prices, volumes = self._generate_synthetic_data(universe, start, end)
        else:
            prices = prices.loc[start:end, [t for t in universe if t in prices.columns]]
            volumes = volumes.loc[start:end, [t for t in universe if t in volumes.columns]]

        # 2. Compute signal scores
        logger.info("Computing signal scores...")
        try:
            scores = signal_fn(prices, volumes)
        except Exception as exc:
            logger.error("Signal function raised: %s", exc)
            return self._failed_result(signal_name, str(exc))

        scores = self._align_and_clean(scores, prices)

        # 3. Compute forward returns
        returns = prices.pct_change()
        fwd_returns = returns.shift(-self.forward_return_horizon)

        # 4. Compute IC time series
        ic_series = self._compute_ic_series(scores, fwd_returns)
        if ic_series.empty or ic_series.dropna().shape[0] < 20:
            return self._failed_result(signal_name, "Insufficient IC observations")

        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        icir = ic_mean / ic_std if ic_std > 0 else 0.0

        # t-test on IC series
        t_stat, p_value = stats.ttest_1samp(ic_series.dropna(), 0.0)
        n_obs = int(ic_series.dropna().shape[0])

        # 5. IC decay half-life
        halflife = self._compute_ic_decay_halflife(scores, returns, max_lag=20)

        # 6. Regime conditioning
        regime_ic = self._regime_conditional_ic(scores, fwd_returns, prices)

        # 7. Turnover
        turnover = self._compute_turnover(scores)

        # 8. Capacity estimate
        capacity = self._estimate_capacity(scores, volumes)

        # 9. Statistical significance
        is_significant = (abs(icir) > 0.5) and (p_value < 0.05)

        # 10. Deflated Sharpe (Bailey-Lopez)
        deflated_sr = self._deflated_sharpe(ic_series, self.n_trials_correction)

        # 11. Recommendation
        recommendation = self._recommend(icir, p_value, deflated_sr, ic_mean)

        return ResearchResult(
            signal_name=signal_name,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_decay_halflife=halflife,
            regime_conditional_ic=regime_ic,
            turnover_daily=turnover,
            capacity_estimate_usd=capacity,
            is_significant=is_significant,
            deflated_sharpe=deflated_sr,
            recommendation=recommendation,
            ic_series=ic_series,
            p_value=p_value,
            n_observations=n_obs,
        )

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _generate_synthetic_data(
        self, universe: List[str], start: str, end: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic OHLCV data for backtesting signal functions."""
        dates = pd.bdate_range(start, end)
        n_dates = len(dates)
        n_tickers = len(universe)
        rng = np.random.default_rng(seed=42)

        # Geometric Brownian Motion prices
        daily_vol = 0.015
        drift = 0.0002
        log_returns = rng.normal(drift, daily_vol, size=(n_dates, n_tickers))
        # Add cross-sectional momentum factor
        factor = rng.normal(0, 0.005, size=(n_dates, 1))
        log_returns += factor * rng.uniform(0.3, 1.2, size=(1, n_tickers))

        price_paths = np.exp(np.cumsum(log_returns, axis=0))
        prices = pd.DataFrame(
            price_paths * 100.0, index=dates, columns=universe
        )

        # Volumes: log-normal with mean ~1M shares
        volumes = pd.DataFrame(
            np.exp(rng.normal(13.8, 0.5, size=(n_dates, n_tickers))),
            index=dates,
            columns=universe,
        )

        return prices, volumes

    def _align_and_clean(
        self, scores: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Align signal scores to the price index and forward-fill / drop all-NaN rows."""
        scores = scores.reindex(index=prices.index, columns=prices.columns)
        scores = scores.where(prices.notna())
        # Drop rows with fewer than 3 valid scores
        scores = scores[scores.notna().sum(axis=1) >= 3]
        return scores

    # ------------------------------------------------------------------
    # IC computation
    # ------------------------------------------------------------------

    def _compute_ic_series(
        self, scores: pd.DataFrame, fwd_returns: pd.DataFrame
    ) -> pd.Series:
        """Compute per-date cross-sectional IC between signal and forward returns."""
        common_dates = scores.index.intersection(fwd_returns.index)
        ic_values = {}

        for date in common_dates:
            s = scores.loc[date].dropna()
            r = fwd_returns.loc[date].reindex(s.index).dropna()
            s = s.reindex(r.index)
            if len(s) < 5:
                continue
            if self.ic_method == "spearman":
                rho, _ = stats.spearmanr(s, r)
            else:
                rho, _ = stats.pearsonr(s, r)
            ic_values[date] = rho

        return pd.Series(ic_values).sort_index()

    # ------------------------------------------------------------------
    # IC decay half-life
    # ------------------------------------------------------------------

    def _compute_ic_decay_halflife(
        self, scores: pd.DataFrame, returns: pd.DataFrame, max_lag: int = 20
    ) -> float:
        """
        Compute the half-life of IC decay by measuring IC at multiple forward horizons
        and fitting an exponential decay curve.
        """
        lags = list(range(1, max_lag + 1))
        ic_at_lag = []

        for lag in lags:
            fwd = returns.shift(-lag)
            common = scores.index.intersection(fwd.index)
            lag_ics = []
            for date in common:
                s = scores.loc[date].dropna()
                r = fwd.loc[date].reindex(s.index).dropna()
                s = s.reindex(r.index)
                if len(s) < 5:
                    continue
                rho, _ = stats.spearmanr(s, r)
                lag_ics.append(rho)
            ic_at_lag.append(np.nanmean(lag_ics) if lag_ics else 0.0)

        ic_array = np.array(ic_at_lag)
        if ic_array[0] == 0:
            return float("inf")

        # Normalize to 1 at lag=0, then fit decay
        ic_norm = ic_array / (ic_array[0] + 1e-9)
        lags_arr = np.array(lags, dtype=float)

        try:
            def exp_decay(t, halflife):
                return np.exp(-np.log(2) * t / halflife)

            popt, _ = curve_fit(
                exp_decay, lags_arr, ic_norm, p0=[5.0], bounds=(0.1, 200.0)
            )
            return float(popt[0])
        except Exception:
            return float("inf")

    # ------------------------------------------------------------------
    # Regime conditioning
    # ------------------------------------------------------------------

    def _regime_conditional_ic(
        self,
        scores: pd.DataFrame,
        fwd_returns: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute IC separately in bull, bear, and ranging regimes."""
        # Use rolling 63-day return of equal-weight index to classify regime
        eq_index = prices.pct_change().mean(axis=1)
        rolling_ret = eq_index.rolling(self.regime_window).mean() * 252

        bull_dates = rolling_ret[rolling_ret > 0.10].index
        bear_dates = rolling_ret[rolling_ret < -0.10].index
        ranging_dates = rolling_ret[
            (rolling_ret >= -0.10) & (rolling_ret <= 0.10)
        ].index

        regime_ic = {}
        for label, dates in [
            ("bull", bull_dates),
            ("bear", bear_dates),
            ("ranging", ranging_dates),
        ]:
            mask = scores.index.intersection(dates)
            if len(mask) < 10:
                regime_ic[label] = float("nan")
                continue
            ic_vals = []
            for date in mask:
                s = scores.loc[date].dropna()
                r = fwd_returns.loc[date].reindex(s.index).dropna()
                s = s.reindex(r.index)
                if len(s) < 5:
                    continue
                rho, _ = stats.spearmanr(s, r)
                ic_vals.append(rho)
            regime_ic[label] = float(np.nanmean(ic_vals)) if ic_vals else float("nan")

        return regime_ic

    # ------------------------------------------------------------------
    # Turnover
    # ------------------------------------------------------------------

    def _compute_turnover(self, scores: pd.DataFrame, top_n: int = 20) -> float:
        """
        Estimate daily turnover as the mean fraction of the long-short book
        that changes from one day to the next.
        """
        turnovers = []
        dates = scores.index.tolist()
        for i in range(1, len(dates)):
            prev = scores.loc[dates[i - 1]].dropna()
            curr = scores.loc[dates[i]].dropna()
            common = prev.index.intersection(curr.index)
            if len(common) < top_n * 2:
                continue
            prev_long = set(prev.reindex(common).nlargest(top_n).index)
            curr_long = set(curr.reindex(common).nlargest(top_n).index)
            turnover = len(prev_long.symmetric_difference(curr_long)) / (2 * top_n)
            turnovers.append(turnover)

        return float(np.mean(turnovers)) if turnovers else 0.5

    # ------------------------------------------------------------------
    # Capacity estimate
    # ------------------------------------------------------------------

    def _estimate_capacity(
        self, scores: pd.DataFrame, volumes: pd.DataFrame, adv_price: float = 50.0
    ) -> float:
        """
        Estimate capacity using mean daily volume of the top-quintile stocks.
        Capacity = sum(ADV_top_quintile) * adv_fraction * price.
        """
        adv = volumes.mean()
        q80 = float(np.nanpercentile(scores.mean(), 80))
        top_tickers = scores.mean()[scores.mean() >= q80].index
        mean_adv_top = float(adv.reindex(top_tickers).mean())
        n_top = len(top_tickers)
        capacity = mean_adv_top * n_top * adv_price * self.capacity_adv_fraction
        return float(capacity)

    # ------------------------------------------------------------------
    # Deflated Sharpe (Bailey-Lopez 2014)
    # ------------------------------------------------------------------

    def _deflated_sharpe(
        self, ic_series: pd.Series, n_trials: int = 1
    ) -> float:
        """
        Compute the Deflated Sharpe Ratio (DSR) which adjusts the observed
        Sharpe for the bias introduced by multiple testing and non-normality.

        Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
        """
        ic = ic_series.dropna()
        n = len(ic)
        if n < 4:
            return 0.0

        sr_obs = float(ic.mean() / ic.std())
        skew = float(stats.skew(ic))
        kurt = float(stats.kurtosis(ic))

        # Expected maximum SR under n_trials tests (Eq 8 from Bailey-Lopez)
        gamma_euler = 0.5772156649
        sr_expected = (
            (1 - gamma_euler) * stats.norm.ppf(1 - 1.0 / n_trials)
            + gamma_euler * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        ) if n_trials > 1 else 0.0

        # Variance adjustment for skew/kurtosis
        var_adj = np.sqrt(
            (1 - skew * sr_obs + (kurt - 1) / 4.0 * sr_obs ** 2) / (n - 1)
        )
        if var_adj <= 0:
            return 0.0

        dsr = stats.norm.cdf((sr_obs - sr_expected) / var_adj)
        return float(dsr)

    # ------------------------------------------------------------------
    # Recommendation logic
    # ------------------------------------------------------------------

    def _recommend(
        self, icir: float, p_value: float, deflated_sr: float, ic_mean: float
    ) -> str:
        if abs(icir) > 0.5 and p_value < 0.05 and deflated_sr > 0.65:
            return "PROMOTE"
        if abs(icir) < 0.2 or p_value > 0.20:
            return "RETIRE"
        return "WATCH"

    def _failed_result(self, signal_name: str, reason: str) -> ResearchResult:
        logger.warning("Pipeline failed for %s: %s", signal_name, reason)
        return ResearchResult(
            signal_name=signal_name,
            ic_mean=0.0,
            ic_std=0.0,
            icir=0.0,
            ic_decay_halflife=float("inf"),
            recommendation="RETIRE",
        )


# ---------------------------------------------------------------------------
# Built-in signal functions
# Each returns a pd.DataFrame of raw signal scores (same shape as prices)
# ---------------------------------------------------------------------------

def signal_momentum_1m(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """21-day price momentum (skip last day)."""
    returns = prices.pct_change()
    mom = returns.rolling(21).sum().shift(1)
    return mom


def signal_momentum_3m(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """63-day price momentum, skip 1 day."""
    returns = prices.pct_change()
    mom = returns.rolling(63).sum().shift(1)
    return mom


def signal_momentum_6m(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """126-day price momentum, skip 1 day."""
    returns = prices.pct_change()
    mom = returns.rolling(126).sum().shift(1)
    return mom


def signal_reversal_5d(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """5-day short-term reversal (negative of 5d return)."""
    returns = prices.pct_change()
    rev = -returns.rolling(5).sum()
    return rev


def signal_reversal_1d(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """1-day overnight reversal."""
    rev = -prices.pct_change()
    return rev


def signal_vol_breakout(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility breakout: ratio of recent 5d vol to trailing 63d vol.
    High values indicate expanding volatility (potential breakout).
    """
    returns = prices.pct_change()
    short_vol = returns.rolling(5).std()
    long_vol = returns.rolling(63).std()
    score = short_vol / (long_vol + 1e-9) - 1.0
    return score


def signal_vol_regime(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Vol regime signal: stocks in low-vol regime get positive scores.
    Uses rolling 21d realized vol ranked cross-sectionally.
    """
    returns = prices.pct_change()
    vol_21 = returns.rolling(21).std()
    # Rank inverted -- low vol -> high score
    score = vol_21.rank(axis=1, ascending=False, pct=True)
    return score


def signal_bh_mass_raw(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Black-Hole mass proxy: volume * absolute return (raw, no smoothing).
    Captures large moves with high conviction (volume confirmation).
    """
    returns = prices.pct_change().abs()
    mass = volumes * returns
    return mass.rank(axis=1, pct=True)


def signal_bh_mass_filtered(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Black-Hole mass filtered: 5d EMA smoothed version of BH mass.
    Reduces noise in the raw mass signal.
    """
    returns = prices.pct_change().abs()
    mass = volumes * returns
    smoothed = mass.ewm(span=5).mean()
    return smoothed.rank(axis=1, pct=True)


def signal_hurst_trend(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Hurst-based trending signal.
    Approximates Hurst exponent using variance ratio over 4 vs 16 day windows.
    H > 0.5 suggests trending; used as positive momentum signal.
    """
    log_prices = np.log(prices + 1e-9)
    var_4 = log_prices.diff().rolling(4).var()
    var_16 = log_prices.diff().rolling(16).var()
    hurst_approx = 0.5 * np.log(var_16 / (var_4 + 1e-9)) / np.log(4.0)
    # High hurst -> trend following signal
    trend_score = (hurst_approx - 0.5) * prices.pct_change().rolling(21).sum()
    return trend_score


def signal_hurst_revert(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Hurst-based mean-reversion signal.
    H < 0.5 suggests mean-reverting; combined with short-term reversal.
    """
    log_prices = np.log(prices + 1e-9)
    var_4 = log_prices.diff().rolling(4).var()
    var_16 = log_prices.diff().rolling(16).var()
    hurst_approx = 0.5 * np.log(var_16 / (var_4 + 1e-9)) / np.log(4.0)
    rev_5 = -prices.pct_change().rolling(5).sum()
    revert_score = (0.5 - hurst_approx) * rev_5
    return revert_score


def signal_rsi_extreme(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    RSI extreme signal: distance of RSI(14) from 50, used as reversal signal.
    RSI > 70 -> sell signal; RSI < 30 -> buy signal.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    # Score: negative RSI deviation from 50 (reversal logic)
    score = -(rsi - 50) / 50.0
    return score


def signal_macd_cross(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    MACD crossover signal.
    Score = MACD line - signal line; positive when fast EMA crosses above slow.
    """
    ema_fast = prices.ewm(span=12).mean()
    ema_slow = prices.ewm(span=26).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    return histogram / (prices + 1e-9)  # normalize by price level


def signal_vwap_deviation(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    VWAP deviation signal.
    Score = (price - rolling_VWAP) / rolling_VWAP; used as mean-reversion signal.
    """
    dollar_vol = prices * volumes
    rolling_vwap = dollar_vol.rolling(21).sum() / (volumes.rolling(21).sum() + 1e-9)
    deviation = (prices - rolling_vwap) / (rolling_vwap + 1e-9)
    # Mean reversion: negative deviation -> positive signal
    return -deviation


def signal_atr_expansion(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    ATR expansion signal.
    Ratio of current ATR(5) to ATR(21); high values -> potential breakout.
    Uses close-to-close range as proxy for true range.
    """
    returns = prices.pct_change().abs()
    atr_5 = returns.rolling(5).mean()
    atr_21 = returns.rolling(21).mean()
    expansion = atr_5 / (atr_21 + 1e-9)
    # Momentum signal conditioned on expansion
    direction = prices.pct_change().rolling(5).sum()
    return expansion * direction


# ---------------------------------------------------------------------------
# Signal Universe -- registry and batch runner
# ---------------------------------------------------------------------------

class SignalUniverse:
    """
    Registry of named signal functions with metadata.
    Supports batch evaluation and comparison.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, dict] = {}
        self._pipeline = SignalResearchPipeline()

    def register(
        self,
        name: str,
        fn: Callable,
        description: str = "",
        category: str = "unknown",
    ) -> None:
        """Register a signal function with the universe."""
        self._registry[name] = {
            "fn": fn,
            "description": description,
            "category": category,
        }

    def list_signals(self) -> pd.DataFrame:
        """Return a DataFrame summarising registered signals."""
        rows = [
            {"name": name, "category": meta["category"], "description": meta["description"]}
            for name, meta in self._registry.items()
        ]
        return pd.DataFrame(rows)

    def run_all(
        self,
        universe: List[str],
        start: str,
        end: str,
        prices: Optional[pd.DataFrame] = None,
        volumes: Optional[pd.DataFrame] = None,
        n_trials_correction: Optional[int] = None,
    ) -> List[ResearchResult]:
        """
        Run every registered signal through the research pipeline.

        The n_trials_correction defaults to the number of registered signals,
        applying a proper Bonferroni-style adjustment in the deflated Sharpe.
        """
        n = n_trials_correction or len(self._registry)
        self._pipeline.n_trials_correction = max(n, 1)

        # Pre-generate synthetic data once so all signals see identical data
        if prices is None or volumes is None:
            prices, volumes = self._pipeline._generate_synthetic_data(
                universe, start, end
            )

        results = []
        for name, meta in self._registry.items():
            logger.info("Evaluating signal: %s", name)
            result = self._pipeline.run(
                meta["fn"], universe, start, end, prices=prices, volumes=volumes
            )
            result.signal_name = name
            result.category = meta["category"]
            result.description = meta["description"]
            results.append(result)

        return results

    def compare(self, results: List[ResearchResult]) -> pd.DataFrame:
        """Return a sorted comparison DataFrame of all research results."""
        rows = [r.to_dict() for r in results]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values("icir", ascending=False).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# Pre-register all built-in signals
# ---------------------------------------------------------------------------

_BUILTIN_SIGNALS = [
    ("signal_momentum_1m",   signal_momentum_1m,   "1-month momentum, skip 1d",           "momentum"),
    ("signal_momentum_3m",   signal_momentum_3m,   "3-month momentum, skip 1d",           "momentum"),
    ("signal_momentum_6m",   signal_momentum_6m,   "6-month momentum, skip 1d",           "momentum"),
    ("signal_reversal_5d",   signal_reversal_5d,   "5-day short-term reversal",           "reversal"),
    ("signal_reversal_1d",   signal_reversal_1d,   "1-day overnight reversal",            "reversal"),
    ("signal_vol_breakout",  signal_vol_breakout,  "Vol expansion breakout",              "volatility"),
    ("signal_vol_regime",    signal_vol_regime,    "Low-vol regime score",                "volatility"),
    ("signal_bh_mass_raw",   signal_bh_mass_raw,   "BH mass raw (vol * |ret|)",           "bh"),
    ("signal_bh_mass_filtered", signal_bh_mass_filtered, "BH mass EMA smoothed",         "bh"),
    ("signal_hurst_trend",   signal_hurst_trend,   "Hurst-based trend following",         "hurst"),
    ("signal_hurst_revert",  signal_hurst_revert,  "Hurst-based mean reversion",          "hurst"),
    ("signal_rsi_extreme",   signal_rsi_extreme,   "RSI extreme reversal",                "oscillator"),
    ("signal_macd_cross",    signal_macd_cross,    "MACD histogram crossover",            "oscillator"),
    ("signal_vwap_deviation", signal_vwap_deviation, "VWAP deviation reversal",           "microstructure"),
    ("signal_atr_expansion", signal_atr_expansion, "ATR expansion + direction",           "volatility"),
]

# Module-level default universe for convenience
default_signal_universe = SignalUniverse()
for _name, _fn, _desc, _cat in _BUILTIN_SIGNALS:
    default_signal_universe.register(_name, _fn, _desc, _cat)
