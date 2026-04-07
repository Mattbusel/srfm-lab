"""
pairs_trading.py -- Statistical arbitrage pairs trading strategy.

References:
  - Engle & Granger (1987): Co-integration and error correction
  - Gatev, Goetzmann, Rouwenhorst (2006): Pairs trading: Performance of a
    relative-value arbitrage rule
  - Kalman (1960): A new approach to linear filtering and prediction

BH constants used for regime gating:
  BH_MASS_THRESH = 1.92
  BH_DECAY       = 0.924
  BH_COLLAPSE    = 0.992

LARSA v18 compatible.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# BH physics constants
# ---------------------------------------------------------------------------
BH_MASS_THRESH = 1.92
BH_DECAY       = 0.924
BH_COLLAPSE    = 0.992


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PairsSignal:
    """Output of generate_signal for a single bar pair."""
    symbol_a: str = ""
    symbol_b: str = ""
    action: str = "flat"          # "long_spread", "short_spread", "exit", "stop", "flat"
    zscore: float = 0.0
    spread: float = 0.0
    hedge_ratio: float = 1.0
    position_a: float = 0.0       # signed unit position in A (+1 long, -1 short)
    position_b: float = 0.0       # signed unit position in B
    confidence: float = 0.0       # 0..1 signal confidence


@dataclass
class PairBacktestResult:
    symbol_a: str = ""
    symbol_b: str = ""
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_trade_return: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    spread_series: pd.Series = field(default_factory=pd.Series)
    zscore_series: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[{self.symbol_a}/{self.symbol_b}] "
            f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
            f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
            f"Trades={self.n_trades}"
        )


@dataclass
class CointegrationResult:
    symbol_a: str = ""
    symbol_b: str = ""
    pvalue: float = 1.0
    correlation: float = 0.0
    hedge_ratio: float = 1.0
    half_life: float = np.nan
    is_cointegrated: bool = False

    def summary(self) -> str:
        status = "YES" if self.is_cointegrated else "NO"
        return (
            f"{self.symbol_a}/{self.symbol_b}: coint={status} "
            f"pval={self.pvalue:.4f} rho={self.correlation:.3f} "
            f"beta={self.hedge_ratio:.4f} half_life={self.half_life:.1f}d"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_stats(equity_curve: np.ndarray, trade_returns: List[float]) -> dict:
    n = len(equity_curve)
    initial = equity_curve[0]
    final   = equity_curve[-1]
    total_return = final / initial - 1.0
    n_years = max(1, n / 252)
    cagr = (final / initial) ** (1.0 / n_years) - 1.0
    rets = np.diff(equity_curve) / (equity_curve[:-1] + 1e-9)
    rets = np.concatenate([[0.0], rets])
    std  = rets.std()
    sharpe = rets.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down   = rets[rets < 0]
    sortino_d = np.std(down) if len(down) > 0 else 1e-9
    sortino   = rets.mean() / sortino_d * math.sqrt(252)
    pk  = np.maximum.accumulate(equity_curve)
    dd  = (equity_curve - pk) / (pk + 1e-9)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    wins   = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = len(wins) / len(trade_returns) if trade_returns else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    return dict(
        total_return=total_return, cagr=cagr, sharpe=sharpe, sortino=sortino,
        max_drawdown=mdd, calmar=calmar, win_rate=win_rate, profit_factor=pf,
        n_trades=len(trade_returns),
        avg_trade_return=float(np.mean(trade_returns)) if trade_returns else 0.0,
        returns=pd.Series(rets),
    )


def _half_life(spread: np.ndarray) -> float:
    """Estimate mean-reversion half-life via OLS regression of spread on lagged spread."""
    lag = spread[:-1]
    diff = np.diff(spread)
    if len(lag) < 10:
        return float("nan")
    beta = np.polyfit(lag, diff, 1)[0]
    if beta >= 0:
        return float("nan")
    return -math.log(2.0) / beta


# ---------------------------------------------------------------------------
# Kalman filter hedge ratio updater
# ---------------------------------------------------------------------------

class KalmanHedgeFilter:
    """
    1-D Kalman filter to track the hedge ratio beta online.

    State model:  beta_{t} = beta_{t-1} + process_noise
    Measurement:  price_a = beta * price_b + alpha + obs_noise

    We augment state to [beta, alpha] (2-D) for intercept estimation.

    Parameters
    ----------
    delta      : process noise covariance scaling (default 1e-4)
    obs_noise  : observation noise variance (default 1e-3)
    """

    def __init__(self, delta: float = 1e-4, obs_noise: float = 1e-3):
        self.delta     = delta
        self.obs_noise = obs_noise
        # State: [beta, alpha]
        self.x  = np.array([1.0, 0.0])          # initial state
        self.P  = np.eye(2) * 1.0                # initial covariance
        self.Q  = np.eye(2) * delta              # process noise
        self.R  = obs_noise                      # observation noise

    def update(self, price_a: float, price_b: float) -> Tuple[float, float]:
        """
        Process one observation. Returns (beta, alpha).
        """
        # Measurement matrix H = [price_b, 1]
        H = np.array([[price_b, 1.0]])

        # Predict
        x_pred = self.x.copy()
        P_pred = self.P + self.Q

        # Innovation
        y_hat = float(H @ x_pred)
        innov = price_a - y_hat
        S     = float(H @ P_pred @ H.T) + self.R
        if abs(S) < 1e-12:
            return float(self.x[0]), float(self.x[1])

        # Kalman gain
        K = (P_pred @ H.T) / S   # shape (2,1) -> (2,)
        K = K.flatten()

        # Update
        self.x = x_pred + K * innov
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred

        return float(self.x[0]), float(self.x[1])

    def reset(self):
        self.x = np.array([1.0, 0.0])
        self.P = np.eye(2) * 1.0


# ---------------------------------------------------------------------------
# EWM spread statistics tracker
# ---------------------------------------------------------------------------

class EWMSpreadStats:
    """
    Exponentially weighted mean and std for the spread -- used for z-score.

    Parameters
    ----------
    halflife : EWM half-life in bars (default 60)
    min_obs  : minimum observations before returning valid stats (default 20)
    """

    def __init__(self, halflife: int = 60, min_obs: int = 20):
        self.alpha   = 1.0 - math.exp(-math.log(2.0) / halflife)
        self.min_obs = min_obs
        self._mean   = 0.0
        self._var    = 0.0
        self._count  = 0
        self._m2_ewm = 0.0  # second moment for variance

    def update(self, value: float) -> Tuple[float, float]:
        """Returns (mean, std) after incorporating new value."""
        self._count += 1
        if self._count == 1:
            self._mean   = value
            self._var    = 0.0
            return self._mean, 1e-9
        old_mean    = self._mean
        self._mean  = self._alpha_ema(self._mean, value)
        delta       = value - old_mean
        delta2      = value - self._mean
        self._var   = (1.0 - self.alpha) * (self._var + self.alpha * delta * delta2)
        std = math.sqrt(max(self._var, 1e-12))
        if self._count < self.min_obs:
            return self._mean, 1e-9   -- not enough obs yet
        return self._mean, std

    def _alpha_ema(self, prev: float, new: float) -> float:
        return (1.0 - self.alpha) * prev + self.alpha * new

    def reset(self):
        self._mean  = 0.0
        self._var   = 0.0
        self._count = 0


# ---------------------------------------------------------------------------
# Core strategy
# ---------------------------------------------------------------------------

class PairsTradingStrategy:
    """
    Statistical arbitrage pairs trading strategy with Kalman hedge ratio tracking.

    Parameters
    ----------
    symbol_a       : ticker of first instrument
    symbol_b       : ticker of second instrument (the "driver")
    config         : dict with optional overrides for all parameters

    Key config keys
    ---------------
    entry_zscore   : z-score threshold to enter trade (default 2.0)
    exit_zscore    : z-score threshold to exit trade (default 0.5)
    stop_zscore    : z-score threshold for stop-loss exit (default 4.0)
    kalman_delta   : Kalman process noise scaling (default 1e-4)
    ewm_halflife   : half-life for EWM spread stats (default 60)
    """

    def __init__(self, symbol_a: str, symbol_b: str, config: dict):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        cfg = config or {}

        self.hedge_ratio   = cfg.get("hedge_ratio_init", 1.0)
        self.alpha_        = cfg.get("alpha_init", 0.0)
        self.spread_mean   = cfg.get("spread_mean", 0.0)
        self.spread_std    = cfg.get("spread_std", 1.0)
        self.entry_zscore  = cfg.get("entry_zscore", 2.0)
        self.exit_zscore   = cfg.get("exit_zscore", 0.5)
        self.stop_zscore   = cfg.get("stop_zscore", 4.0)
        self.kalman_gain   = cfg.get("kalman_gain", 0.05)  -- online Kalman hedge ratio update

        kalman_delta  = cfg.get("kalman_delta", 1e-4)
        obs_noise     = cfg.get("obs_noise", 1e-3)
        ewm_halflife  = cfg.get("ewm_halflife", 60)
        ewm_min_obs   = cfg.get("ewm_min_obs", 20)

        self._kalman    = KalmanHedgeFilter(delta=kalman_delta, obs_noise=obs_noise)
        self._ewm_stats = EWMSpreadStats(halflife=ewm_halflife, min_obs=ewm_min_obs)
        self._position  = "flat"   -- current position state

    # ------------------------------------------------------------------
    def update_hedge_ratio(self, price_a: float, price_b: float) -> float:
        """
        Kalman filter update of hedge ratio.

        State: [beta, alpha] where price_a = beta * price_b + alpha + noise.
        Returns updated beta (hedge ratio).
        """
        beta, alpha = self._kalman.update(price_a, price_b)
        self.hedge_ratio = beta
        self.alpha_      = alpha
        return beta

    # ------------------------------------------------------------------
    def compute_spread(self, price_a: float, price_b: float) -> float:
        """
        Compute the pair spread: price_a - hedge_ratio * price_b.
        Uses current hedge_ratio (update_hedge_ratio should be called first).
        """
        return price_a - self.hedge_ratio * price_b

    # ------------------------------------------------------------------
    def compute_zscore(self, spread: float) -> float:
        """
        Update EWM mean/std with new spread value and return z-score.
        z = (spread - ewm_mean) / ewm_std
        """
        mean, std = self._ewm_stats.update(spread)
        self.spread_mean = mean
        self.spread_std  = std
        if std < 1e-9:
            return 0.0
        return (spread - mean) / std

    # ------------------------------------------------------------------
    def generate_signal(self, bar_a: dict, bar_b: dict) -> PairsSignal:
        """
        Process one bar for each leg and return a PairsSignal.

        bar_a / bar_b must have a 'close' key.

        Signal logic:
          z > +entry_zscore  -> short spread: short A, long B
          z < -entry_zscore  -> long spread:  long A, short B
          |z| < exit_zscore  -> exit position
          |z| > stop_zscore  -> stop loss exit
        """
        price_a = float(bar_a["close"])
        price_b = float(bar_b["close"])

        # Update Kalman hedge ratio first, then compute spread and z-score
        self.update_hedge_ratio(price_a, price_b)
        spread  = self.compute_spread(price_a, price_b)
        zscore  = self.compute_zscore(spread)

        abs_z = abs(zscore)

        # Determine action based on position state and z-score thresholds
        action = "flat"
        pos_a  = 0.0
        pos_b  = 0.0
        confidence = min(1.0, abs_z / (self.entry_zscore + 1e-9))

        if self._position != "flat" and abs_z > self.stop_zscore:
            # Stop loss -- z has blown out in wrong direction
            action = "stop"
            self._position = "flat"

        elif self._position != "flat" and abs_z < self.exit_zscore:
            # Mean reversion target hit -- exit
            action = "exit"
            self._position = "flat"

        elif self._position == "flat" and zscore > self.entry_zscore:
            # Spread is too high -- short A (sell) long B (buy)
            action = "short_spread"
            pos_a  = -1.0
            pos_b  = +self.hedge_ratio
            self._position = "short_spread"

        elif self._position == "flat" and zscore < -self.entry_zscore:
            # Spread is too low -- long A (buy) short B (sell)
            action = "long_spread"
            pos_a  = +1.0
            pos_b  = -self.hedge_ratio
            self._position = "long_spread"

        elif self._position == "short_spread":
            # Maintain existing short spread position
            action = "short_spread"
            pos_a  = -1.0
            pos_b  = +self.hedge_ratio

        elif self._position == "long_spread":
            # Maintain existing long spread position
            action = "long_spread"
            pos_a  = +1.0
            pos_b  = -self.hedge_ratio

        return PairsSignal(
            symbol_a=self.symbol_a,
            symbol_b=self.symbol_b,
            action=action,
            zscore=zscore,
            spread=spread,
            hedge_ratio=self.hedge_ratio,
            position_a=pos_a,
            position_b=pos_b,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    def reset(self):
        """Reset all internal state for a fresh backtest run."""
        self._kalman.reset()
        self._ewm_stats.reset()
        self.hedge_ratio = 1.0
        self.alpha_      = 0.0
        self.spread_mean = 0.0
        self.spread_std  = 1.0
        self._position   = "flat"


# ---------------------------------------------------------------------------
# Cointegration Scanner
# ---------------------------------------------------------------------------

class CointegrationScanner:
    """
    Engle-Granger cointegration test for pairs selection.

    Scans a universe of price series and returns pairs satisfying:
      1. ADF p-value on residuals < max_pvalue (default 0.05)
      2. |Pearson correlation| > min_correlation (default 0.80)

    Parameters
    ----------
    max_pvalue       : maximum cointegration test p-value (default 0.05)
    min_correlation  : minimum absolute price correlation (default 0.80)
    min_half_life    : minimum spread half-life in days (default 5)
    max_half_life    : maximum spread half-life in days (default 252)
    """

    def __init__(
        self,
        max_pvalue: float      = 0.05,
        min_correlation: float = 0.80,
        min_half_life: float   = 5.0,
        max_half_life: float   = 252.0,
    ):
        self.max_pvalue      = max_pvalue
        self.min_correlation = min_correlation
        self.min_half_life   = min_half_life
        self.max_half_life   = max_half_life

    def _engle_granger_test(
        self, series_a: np.ndarray, series_b: np.ndarray
    ) -> Tuple[float, float]:
        """
        Run Engle-Granger OLS and ADF test on residuals.
        Returns (p_value, hedge_ratio).
        """
        if len(series_a) < 30:
            return 1.0, 1.0
        # OLS: series_a = beta * series_b + alpha + eps
        X = np.column_stack([series_b, np.ones(len(series_b))])
        try:
            beta, alpha = np.linalg.lstsq(X, series_a, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 1.0, 1.0

        residuals = series_a - beta * series_b - alpha

        # ADF test on residuals (simplified Dickey-Fuller without lag augmentation)
        pvalue = self._adf_pvalue(residuals)
        return pvalue, float(beta)

    def _adf_pvalue(self, series: np.ndarray) -> float:
        """
        Simplified ADF test p-value via OLS regression.
        delta_y = gamma * y_{t-1} + error
        H0: gamma = 0 (unit root, no cointegration)
        """
        y    = series
        y_l  = y[:-1]
        dy   = np.diff(y)
        if len(y_l) < 10:
            return 1.0
        # OLS for gamma coefficient
        X      = y_l.reshape(-1, 1)
        beta_  = np.linalg.lstsq(X, dy, rcond=None)[0][0]
        resid  = dy - beta_ * y_l
        se_sq  = (resid ** 2).sum() / max(1, len(resid) - 1)
        x2     = (y_l ** 2).sum()
        if x2 < 1e-12 or se_sq < 1e-12:
            return 1.0
        se     = math.sqrt(se_sq / x2)
        t_stat = beta_ / se
        # Use normal approximation -- ADF critical values vary but this is reasonable
        pvalue = float(scipy_stats.norm.cdf(t_stat))
        return pvalue

    def scan(self, prices: pd.DataFrame) -> List[CointegrationResult]:
        """
        Scan all pairs in prices DataFrame and return cointegrated pairs.

        Parameters
        ----------
        prices : DataFrame with one column per instrument, rows = dates

        Returns
        -------
        List of CointegrationResult sorted by p-value ascending.
        """
        cols    = list(prices.columns)
        results = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                sa = cols[i]
                sb = cols[j]
                arr_a = prices[sa].dropna().values
                arr_b = prices[sb].dropna().values
                # Align lengths
                min_len = min(len(arr_a), len(arr_b))
                if min_len < 30:
                    continue
                arr_a = arr_a[-min_len:]
                arr_b = arr_b[-min_len:]

                # Pearson correlation on prices (log prices for stability)
                log_a = np.log(arr_a + 1e-9)
                log_b = np.log(arr_b + 1e-9)
                rho   = float(np.corrcoef(log_a, log_b)[0, 1])

                if abs(rho) < self.min_correlation:
                    continue

                # Engle-Granger test
                pvalue, hedge_ratio = self._engle_granger_test(arr_a, arr_b)

                if pvalue > self.max_pvalue:
                    continue

                # Half-life estimate
                spread = arr_a - hedge_ratio * arr_b
                hl     = _half_life(spread)

                if not math.isfinite(hl):
                    continue
                if hl < self.min_half_life or hl > self.max_half_life:
                    continue

                results.append(CointegrationResult(
                    symbol_a=sa,
                    symbol_b=sb,
                    pvalue=pvalue,
                    correlation=rho,
                    hedge_ratio=hedge_ratio,
                    half_life=hl,
                    is_cointegrated=True,
                ))

        results.sort(key=lambda r: r.pvalue)
        return results

    def top_pairs(self, prices: pd.DataFrame, n: int = 10) -> List[CointegrationResult]:
        """Return top n cointegrated pairs by p-value."""
        return self.scan(prices)[:n]


# ---------------------------------------------------------------------------
# Pairs Trading Backtest
# ---------------------------------------------------------------------------

class PairsTradingBacktest:
    """
    Run PairsTradingStrategy over historical OHLCV data for a given pair.

    Parameters
    ----------
    config           : dict passed to PairsTradingStrategy
    initial_equity   : starting equity (default 1_000_000)
    commission_pct   : round-trip commission fraction (default 0.001)
    """

    def __init__(
        self,
        config: Optional[dict]  = None,
        initial_equity: float   = 1_000_000.0,
        commission_pct: float   = 0.001,
    ):
        self.config         = config or {}
        self.initial_equity = initial_equity
        self.commission_pct = commission_pct

    def run(
        self,
        bars_a: pd.DataFrame,
        bars_b: pd.DataFrame,
        symbol_a: str = "A",
        symbol_b: str = "B",
    ) -> PairBacktestResult:
        """
        Backtest the pairs strategy.

        bars_a, bars_b: DataFrames with 'close' column, same index.
        """
        # Align on common index
        common_idx = bars_a.index.intersection(bars_b.index)
        if len(common_idx) < 50:
            raise ValueError(f"Insufficient common data: {len(common_idx)} bars")
        a_close = bars_a.loc[common_idx, "close"].values.astype(float)
        b_close = bars_b.loc[common_idx, "close"].values.astype(float)

        strat = PairsTradingStrategy(symbol_a, symbol_b, self.config)

        n          = len(common_idx)
        equity     = self.initial_equity
        eq_curve   = np.full(n, self.initial_equity, dtype=float)
        spread_arr = np.zeros(n)
        zscore_arr = np.zeros(n)
        trade_ret  = []

        # Track open position
        open_pos_a    = 0.0   -- +1 long, -1 short
        open_pos_b    = 0.0
        entry_price_a = None
        entry_price_b = None

        for i in range(n):
            bar_a = {"close": a_close[i]}
            bar_b = {"close": b_close[i]}
            sig   = strat.generate_signal(bar_a, bar_b)

            spread_arr[i] = sig.spread
            zscore_arr[i] = sig.zscore

            if i == 0:
                eq_curve[i] = equity
                continue

            # P&L from carry of existing position (using previous bar prices)
            if open_pos_a != 0.0 or open_pos_b != 0.0:
                ret_a  = (a_close[i] - a_close[i - 1]) / (a_close[i - 1] + 1e-9)
                ret_b  = (b_close[i] - b_close[i - 1]) / (b_close[i - 1] + 1e-9)
                pnl    = open_pos_a * ret_a + open_pos_b * ret_b
                equity *= (1.0 + pnl)

            # Trade transitions
            new_pos_a = sig.position_a
            new_pos_b = sig.position_b

            if sig.action in ("exit", "stop", "flat"):
                if open_pos_a != 0.0 and entry_price_a is not None:
                    # Record closed trade return
                    tr = (open_pos_a * (a_close[i] - entry_price_a) / (entry_price_a + 1e-9)
                          + open_pos_b * (b_close[i] - (entry_price_b or 1.0)) / ((entry_price_b or 1.0) + 1e-9)
                          - self.commission_pct)
                    trade_ret.append(float(tr))
                open_pos_a    = 0.0
                open_pos_b    = 0.0
                entry_price_a = None
                entry_price_b = None

            elif sig.action in ("long_spread", "short_spread"):
                if open_pos_a == 0.0:
                    # New entry
                    open_pos_a    = new_pos_a
                    open_pos_b    = new_pos_b
                    entry_price_a = a_close[i]
                    entry_price_b = b_close[i]
                    equity        *= (1.0 - self.commission_pct / 2.0)

            eq_curve[i] = equity

        stats = _compute_stats(eq_curve, trade_ret)
        return PairBacktestResult(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            **{k: v for k, v in stats.items() if k != "returns"},
            equity_curve=pd.Series(eq_curve, index=common_idx),
            returns=pd.Series(stats["returns"].values, index=common_idx),
            spread_series=pd.Series(spread_arr, index=common_idx),
            zscore_series=pd.Series(zscore_arr, index=common_idx),
            params=self.config,
        )

    def run_portfolio(
        self,
        all_prices: Dict[str, pd.DataFrame],
        pairs: List[Tuple[str, str]],
    ) -> Dict[str, PairBacktestResult]:
        """
        Backtest multiple pairs simultaneously.

        all_prices : {symbol: DataFrame with 'close' column}
        pairs      : list of (symbol_a, symbol_b) tuples

        Returns dict mapping "A/B" -> PairBacktestResult.
        """
        results = {}
        for sa, sb in pairs:
            if sa not in all_prices or sb not in all_prices:
                continue
            try:
                res = self.run(all_prices[sa], all_prices[sb], sa, sb)
                results[f"{sa}/{sb}"] = res
            except Exception as e:
                warnings.warn(f"Backtest failed for {sa}/{sb}: {e}")
        return results

    def summary_table(
        self, results: Dict[str, PairBacktestResult]
    ) -> pd.DataFrame:
        """Return a DataFrame summary of all pair backtests."""
        rows = []
        for key, r in results.items():
            rows.append({
                "pair":          key,
                "total_return":  r.total_return,
                "cagr":          r.cagr,
                "sharpe":        r.sharpe,
                "max_drawdown":  r.max_drawdown,
                "win_rate":      r.win_rate,
                "n_trades":      r.n_trades,
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(99)
    n   = 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    # Simulate cointegrated pair
    common_trend = np.cumsum(rng.normal(0.0002, 0.008, n))
    noise_a      = np.cumsum(rng.normal(0, 0.004, n))
    noise_b      = np.cumsum(rng.normal(0, 0.004, n))
    close_a      = 100.0 * np.exp(common_trend + noise_a)
    close_b      = 50.0  * np.exp(common_trend * 0.9 + noise_b)

    bars_a = pd.DataFrame({"close": close_a}, index=idx)
    bars_b = pd.DataFrame({"close": close_b}, index=idx)

    # Test cointegration scanner
    prices_df = pd.DataFrame({"A": close_a, "B": close_b}, index=idx)
    scanner   = CointegrationScanner(max_pvalue=0.10, min_correlation=0.70)
    coint_res = scanner.scan(prices_df)
    print(f"Cointegrated pairs found: {len(coint_res)}")
    for r in coint_res:
        print(" ", r.summary())

    # Run backtest
    bt     = PairsTradingBacktest(config={"entry_zscore": 1.8, "exit_zscore": 0.3})
    result = bt.run(bars_a, bars_b, "A", "B")
    print(result.summary())
