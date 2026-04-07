"""
multi_asset_backtest.py -- Multi-asset portfolio backtesting for SRFM.

Handles synchronized bar processing across symbols, rolling correlation
tracking, portfolio-level VaR limits, and correlation-aware position sizing.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

BARS_PER_YEAR = 252 * 26  # 15-min bars, ~6.5h/day
CORR_WINDOW = 63           # rolling correlation window in bars
VAR_CONFIDENCE = 0.99      # VaR confidence level
HIGH_CORR_THRESHOLD = 0.7  # pairwise correlation threshold for size reduction
HIGH_CORR_REDUCTION = 0.30 # fraction to reduce both positions by


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MultiAssetResult:
    """Complete results from a multi-asset backtest run."""
    equity_curve: pd.Series
    positions: Dict[str, pd.Series]           # position size over time per symbol
    pnl_by_symbol: Dict[str, float]           # cumulative P&L per symbol
    correlation_matrix: pd.DataFrame          # final rolling correlation
    max_concurrent_positions: int
    gross_exposure_series: pd.Series
    returns_by_symbol: Dict[str, pd.Series]   # bar-by-bar returns per symbol
    regime_var_breaches: int                  # times VaR limit stopped new positions
    trade_counts: Dict[str, int]              # number of trades per symbol

    def summary(self) -> Dict[str, float]:
        """Return top-level performance metrics."""
        eq = self.equity_curve
        rets = eq.pct_change().dropna()
        total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0
        sharpe = (
            rets.mean() / rets.std() * np.sqrt(BARS_PER_YEAR)
            if rets.std() > 1e-10 else 0.0
        )
        peak = eq.cummax()
        drawdown = (eq - peak) / peak
        max_dd = float(drawdown.min())
        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "total_pnl": sum(self.pnl_by_symbol.values()),
            "var_breaches": self.regime_var_breaches,
            "max_concurrent_positions": self.max_concurrent_positions,
        }


# ---------------------------------------------------------------------------
# Correlation-aware position sizer
# ---------------------------------------------------------------------------

class CorrelationAwarePositionSizer:
    """
    Adjusts position sizes based on portfolio correlation structure.

    Rules:
      1. If any pairwise correlation > HIGH_CORR_THRESHOLD, reduce both
         positions by HIGH_CORR_REDUCTION (30%).
      2. Apply a diversification multiplier: size * (1 - avg_pairwise_corr).
      3. Base size is capital_per_symbol / price.
    """

    def __init__(
        self,
        base_capital: float = 10_000.0,
        max_position_frac: float = 0.25,   # max fraction of capital in one symbol
    ):
        self.base_capital = base_capital
        self.max_position_frac = max_position_frac
        self._reduction_log: Dict[str, List[str]] = defaultdict(list)

    def size_position(
        self,
        symbol: str,
        signal: float,           # raw signal strength, -1.0 to 1.0
        price: float,
        portfolio_corr: pd.DataFrame,   # n x n correlation matrix
        other_symbols: List[str],
    ) -> float:
        """
        Compute final position size in units (shares/contracts).

        Returns positive for long, negative for short, 0 for flat.
        """
        if price <= 0 or abs(signal) < 1e-8:
            return 0.0

        # Base notional allocation
        notional = self.base_capital * self.max_position_frac * abs(signal)

        # Compute reduction based on correlation with other active symbols
        reduction_factor = 1.0
        if not portfolio_corr.empty and symbol in portfolio_corr.columns:
            symbols_in_corr = [s for s in other_symbols if s in portfolio_corr.columns and s != symbol]
            if symbols_in_corr:
                pairwise = [
                    abs(float(portfolio_corr.loc[symbol, s]))
                    for s in symbols_in_corr
                    if symbol in portfolio_corr.index
                ]
                if pairwise:
                    avg_corr = float(np.mean(pairwise))
                    max_corr = float(np.max(pairwise))

                    # Rule 1: high correlation penalty
                    if max_corr > HIGH_CORR_THRESHOLD:
                        reduction_factor *= (1.0 - HIGH_CORR_REDUCTION)
                        self._reduction_log[symbol].append(
                            f"high_corr={max_corr:.3f} -> -{HIGH_CORR_REDUCTION*100:.0f}%"
                        )

                    # Rule 2: diversification multiplier
                    div_mult = max(0.1, 1.0 - avg_corr)
                    reduction_factor *= div_mult

        notional *= reduction_factor
        units = notional / price
        sign = 1.0 if signal > 0 else -1.0
        return sign * units

    def get_reduction_log(self, symbol: str) -> List[str]:
        return self._reduction_log.get(symbol, [])


# ---------------------------------------------------------------------------
# VaR calculator
# ---------------------------------------------------------------------------

class PortfolioVaR:
    """
    Computes parametric portfolio VaR given positions and returns history.

    Uses variance-covariance method with rolling covariance matrix.
    """

    def __init__(self, confidence: float = VAR_CONFIDENCE):
        self.confidence = confidence
        self._z = float(scipy_stats.norm.ppf(confidence))

    def compute(
        self,
        positions_notional: Dict[str, float],  # symbol -> notional value
        returns_window: pd.DataFrame,           # rows=bars, cols=symbols
    ) -> float:
        """
        Return estimated 1-bar VaR in dollar terms (positive = loss).

        Returns 0.0 if insufficient data.
        """
        symbols = [s for s in positions_notional if abs(positions_notional[s]) > 1e-6]
        if not symbols or returns_window.empty:
            return 0.0

        cols = [s for s in symbols if s in returns_window.columns]
        if not cols:
            return 0.0

        sub = returns_window[cols].dropna()
        if len(sub) < 5:
            return 0.0

        weights = np.array([positions_notional[s] for s in cols])
        cov = sub.cov().values
        port_var = float(weights @ cov @ weights)
        port_std = np.sqrt(max(port_var, 0.0))
        return self._z * port_std  # positive = downside risk


# ---------------------------------------------------------------------------
# Symbol context
# ---------------------------------------------------------------------------

@dataclass
class SymbolContext:
    """Holds per-symbol state during backtest iteration."""
    symbol: str
    bars: pd.DataFrame           # OHLCV indexed by timestamp
    signal_fn: Callable          # fn(bar_dict) -> float in [-1, 1]
    position: float = 0.0        # current position in units
    cash_from_symbol: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trade_count: int = 0
    bar_returns: List[float] = field(default_factory=list)
    position_series: Dict[pd.Timestamp, float] = field(default_factory=dict)
    _prev_price: float = 0.0

    def bar_at(self, ts: pd.Timestamp) -> Optional[dict]:
        if ts in self.bars.index:
            row = self.bars.loc[ts]
            return row.to_dict() if hasattr(row, "to_dict") else None
        return None


# ---------------------------------------------------------------------------
# Main multi-asset backtest
# ---------------------------------------------------------------------------

class MultiAssetBacktest:
    """
    Portfolio backtester supporting multiple symbols simultaneously.

    Design:
      - All symbol bars are synchronized on a merged timestamp index.
      - At each bar, signals are generated for each available symbol.
      - Portfolio VaR is checked before opening new positions.
      - Rolling 63-bar correlation is tracked for position sizing.
      - P&L is computed mark-to-market each bar.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        var_limit: float = 5_000.0,         # max 1-bar VaR in dollars
        corr_window: int = CORR_WINDOW,
        commission_bps: float = 5.0,        # commission in basis points
        slippage_bps: float = 3.0,
        max_position_frac: float = 0.20,    # max fraction of capital per symbol
    ):
        self.initial_capital = initial_capital
        self.var_limit = var_limit
        self.corr_window = corr_window
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.max_position_frac = max_position_frac

        self._symbols: Dict[str, SymbolContext] = {}
        self._sizer = CorrelationAwarePositionSizer(
            base_capital=initial_capital,
            max_position_frac=max_position_frac,
        )
        self._var_calc = PortfolioVaR(confidence=VAR_CONFIDENCE)

    def add_symbol(
        self,
        symbol: str,
        bars_df: pd.DataFrame,
        signal_fn: Callable,
    ) -> None:
        """
        Register a symbol for multi-asset simulation.

        bars_df must have columns [open, high, low, close, volume] and a
        DatetimeIndex. signal_fn(bar: dict) -> float in [-1, 1].
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(bars_df.columns)
        if missing:
            raise ValueError(f"bars_df for {symbol} missing columns: {missing}")

        df = bars_df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        self._symbols[symbol] = SymbolContext(
            symbol=symbol,
            bars=df,
            signal_fn=signal_fn,
        )
        logger.info("Added symbol %s with %d bars", symbol, len(df))

    def run(self, start: str, end: str) -> MultiAssetResult:
        """
        Execute the multi-asset backtest over [start, end].

        Returns MultiAssetResult with full equity curve, positions, P&L, etc.
        """
        if not self._symbols:
            raise ValueError("No symbols added -- call add_symbol() first.")

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # Build merged timeline across all symbols
        all_timestamps: pd.DatetimeIndex = pd.DatetimeIndex([])
        for ctx in self._symbols.values():
            mask = (ctx.bars.index >= start_ts) & (ctx.bars.index <= end_ts)
            all_timestamps = all_timestamps.union(ctx.bars.index[mask])
        all_timestamps = all_timestamps.sort_values()

        if len(all_timestamps) == 0:
            raise ValueError(f"No bars found in range [{start}, {end}].")

        # State
        cash = self.initial_capital
        equity_series: Dict[pd.Timestamp, float] = {}
        gross_exposure: Dict[pd.Timestamp, float] = {}
        max_concurrent = 0
        var_breaches = 0

        # Rolling returns buffer for correlation
        returns_buffer: Dict[str, List[float]] = {s: [] for s in self._symbols}
        corr_matrix: pd.DataFrame = pd.DataFrame()

        for ts in all_timestamps:
            # Gather available bars at this timestamp
            available: Dict[str, dict] = {}
            for sym, ctx in self._symbols.items():
                bar = ctx.bar_at(ts)
                if bar is not None:
                    available[sym] = bar

            if not available:
                continue

            # Update rolling returns and correlation matrix
            for sym, bar in available.items():
                ctx = self._symbols[sym]
                price = float(bar.get("close", 0))
                if price > 0 and ctx._prev_price > 0:
                    ret = (price - ctx._prev_price) / ctx._prev_price
                    returns_buffer[sym].append(ret)
                    ctx.bar_returns.append(ret)

            # Recompute rolling correlation if enough history
            min_buf = min(len(returns_buffer[s]) for s in self._symbols)
            if min_buf >= self.corr_window:
                window_data = {
                    s: returns_buffer[s][-self.corr_window:]
                    for s in self._symbols
                }
                ret_df = pd.DataFrame(window_data)
                corr_matrix = ret_df.corr()

            # Compute current portfolio notional for VaR check
            notional_map: Dict[str, float] = {}
            for sym, ctx in self._symbols.items():
                bar = ctx.bar_at(ts)
                if bar is None:
                    bar = {"close": ctx._prev_price}
                price = float(bar.get("close", ctx._prev_price))
                notional_map[sym] = ctx.position * price

            # Build returns DataFrame for VaR
            min_len = min((len(v) for v in returns_buffer.values()), default=0)
            returns_df = pd.DataFrame()
            if min_len >= 2:
                returns_df = pd.DataFrame(
                    {s: returns_buffer[s][-min(min_len, 252):] for s in self._symbols}
                )

            portfolio_var = self._var_calc.compute(notional_map, returns_df)
            var_exceeded = portfolio_var > self.var_limit

            if var_exceeded:
                var_breaches += 1

            # Generate signals and trade
            for sym, bar in available.items():
                ctx = self._symbols[sym]
                price = float(bar.get("close", 0))
                if price <= 0:
                    continue

                # Generate signal
                try:
                    signal = float(ctx.signal_fn(bar))
                except Exception as exc:
                    logger.warning("signal_fn error for %s at %s: %s", sym, ts, exc)
                    signal = 0.0

                signal = max(-1.0, min(1.0, signal))

                # Determine desired position
                other_syms = [s for s in self._symbols if s != sym]
                desired_position = self._sizer.size_position(
                    symbol=sym,
                    signal=signal,
                    price=price,
                    portfolio_corr=corr_matrix,
                    other_symbols=other_syms,
                )

                # If VaR exceeded, only allow position reductions
                if var_exceeded:
                    current_sign = np.sign(ctx.position)
                    desired_sign = np.sign(desired_position)
                    # Reduce or close only
                    if desired_sign == current_sign and abs(desired_position) > abs(ctx.position):
                        desired_position = ctx.position  # no new size increase
                    elif abs(desired_position) > abs(ctx.position) and current_sign == 0:
                        desired_position = 0.0  # no new positions

                # Execute trade
                delta = desired_position - ctx.position
                if abs(delta) > 1e-8:
                    fill_price = self._apply_slippage(price, delta)
                    cost = self._compute_commission(abs(delta), fill_price)
                    trade_value = delta * fill_price
                    cash -= trade_value + cost
                    ctx.position = desired_position
                    ctx.trade_count += 1

                # Mark-to-market unrealized P&L
                if ctx._prev_price > 0:
                    ctx.unrealized_pnl = ctx.position * (price - ctx._prev_price)

                ctx._prev_price = price
                ctx.position_series[ts] = ctx.position

            # Compute equity
            total_market_value = 0.0
            n_open = 0
            for sym, ctx in self._symbols.items():
                bar_d = ctx.bar_at(ts)
                price = float(bar_d["close"]) if bar_d else ctx._prev_price
                total_market_value += ctx.position * price
                if abs(ctx.position) > 1e-8:
                    n_open += 1

            equity = cash + total_market_value
            equity_series[ts] = equity
            gross_exp = sum(
                abs(ctx.position * (float(available[s]["close"]) if s in available else ctx._prev_price))
                for s, ctx in self._symbols.items()
            )
            gross_exposure[ts] = gross_exp
            max_concurrent = max(max_concurrent, n_open)

        # Compute final realized P&L per symbol
        pnl_by_symbol: Dict[str, float] = {}
        for sym, ctx in self._symbols.items():
            bar_prices = ctx.bars.loc[
                (ctx.bars.index >= start_ts) & (ctx.bars.index <= end_ts), "close"
            ]
            if len(bar_prices) >= 2:
                first_p = float(bar_prices.iloc[0])
                last_p = float(bar_prices.iloc[-1])
                # Approximate: position * price change, ignoring intermediate trades
                positions_ts = pd.Series(ctx.position_series)
                if not positions_ts.empty:
                    # Sum bar P&L
                    bar_ret_series = pd.Series(ctx.bar_returns)
                    pnl_by_symbol[sym] = float(bar_ret_series.sum() * self.initial_capital * self.max_position_frac)
                else:
                    pnl_by_symbol[sym] = 0.0
            else:
                pnl_by_symbol[sym] = 0.0

        # Build output Series
        eq_series = pd.Series(equity_series, name="equity")
        eq_series = eq_series.sort_index()

        positions_out: Dict[str, pd.Series] = {}
        for sym, ctx in self._symbols.items():
            positions_out[sym] = pd.Series(ctx.position_series, name=sym).sort_index()

        returns_by_symbol: Dict[str, pd.Series] = {}
        for sym, ctx in self._symbols.items():
            bar_idx = sorted(ctx.position_series.keys())
            if len(ctx.bar_returns) > 0:
                n = min(len(bar_idx), len(ctx.bar_returns))
                returns_by_symbol[sym] = pd.Series(
                    ctx.bar_returns[-n:], index=bar_idx[-n:], name=sym
                )

        gross_exp_series = pd.Series(gross_exposure, name="gross_exposure").sort_index()

        return MultiAssetResult(
            equity_curve=eq_series,
            positions=positions_out,
            pnl_by_symbol=pnl_by_symbol,
            correlation_matrix=corr_matrix,
            max_concurrent_positions=max_concurrent,
            gross_exposure_series=gross_exp_series,
            returns_by_symbol=returns_by_symbol,
            regime_var_breaches=var_breaches,
            trade_counts={sym: ctx.trade_count for sym, ctx in self._symbols.items()},
        )

    def _apply_slippage(self, price: float, delta: float) -> float:
        """Return fill price after slippage."""
        direction = 1.0 if delta > 0 else -1.0
        slip_frac = self.slippage_bps / 10_000.0
        return price * (1.0 + direction * slip_frac)

    def _compute_commission(self, qty: float, price: float) -> float:
        """Return commission in dollars."""
        comm_frac = self.commission_bps / 10_000.0
        return abs(qty) * price * comm_frac

    def get_correlation_history(self) -> pd.DataFrame:
        """
        Return a DataFrame showing rolling pairwise correlations over time
        for all symbol pairs, computed on stored bar returns.
        """
        symbols = list(self._symbols.keys())
        all_rets = {
            sym: pd.Series(ctx.bar_returns)
            for sym, ctx in self._symbols.items()
        }
        ret_df = pd.DataFrame(all_rets)
        if ret_df.empty or len(ret_df) < self.corr_window:
            return pd.DataFrame()

        # Compute rolling correlation for each pair
        pairs: Dict[str, pd.Series] = {}
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                if s1 in ret_df.columns and s2 in ret_df.columns:
                    rolling_corr = (
                        ret_df[s1]
                        .rolling(self.corr_window)
                        .corr(ret_df[s2])
                    )
                    pairs[f"{s1}_{s2}"] = rolling_corr

        return pd.DataFrame(pairs)

    def reset(self) -> None:
        """Clear all symbols and state for a fresh run."""
        self._symbols.clear()
        logger.info("MultiAssetBacktest reset.")


# ---------------------------------------------------------------------------
# Portfolio analytics helpers
# ---------------------------------------------------------------------------

class PortfolioAnalytics:
    """
    Post-run analytics for multi-asset results.

    Computes diversification metrics, risk attribution, and exposure analysis.
    """

    def __init__(self, result: MultiAssetResult):
        self.result = result

    def diversification_ratio(self) -> float:
        """
        DR = (sum of individual vols) / portfolio vol.

        Higher is better -- indicates genuine diversification.
        """
        eq = self.result.equity_curve
        port_vol = float(eq.pct_change().dropna().std())
        if port_vol < 1e-10:
            return 1.0

        sym_vols = []
        for sym, ret_series in self.result.returns_by_symbol.items():
            v = float(ret_series.std())
            sym_vols.append(v)

        if not sym_vols:
            return 1.0

        sum_vol = float(np.sum(sym_vols))
        return sum_vol / (port_vol + 1e-10)

    def marginal_var(
        self, symbol: str, returns_df: pd.DataFrame, notional_map: Dict[str, float]
    ) -> float:
        """
        Estimate marginal VaR contribution of a symbol using delta-normal method.
        """
        if symbol not in returns_df.columns:
            return 0.0

        cov = returns_df.cov()
        symbols = list(notional_map.keys())
        w = np.array([notional_map.get(s, 0.0) for s in symbols])
        port_var = float(w @ cov.reindex(index=symbols, columns=symbols).values @ w)
        port_std = np.sqrt(max(port_var, 0.0))

        if port_std < 1e-10 or symbol not in symbols:
            return 0.0

        idx = symbols.index(symbol)
        cov_vec = cov.reindex(index=symbols, columns=[symbol]).values.flatten()
        marginal = float((w @ cov_vec) / port_std)
        z = float(scipy_stats.norm.ppf(VAR_CONFIDENCE))
        return z * marginal

    def exposure_by_time(self) -> pd.DataFrame:
        """
        Return DataFrame of gross exposure per symbol over time.
        Rows are timestamps, columns are symbols.
        """
        frames: Dict[str, pd.Series] = {}
        for sym, pos_series in self.result.positions.items():
            ret_series = self.result.returns_by_symbol.get(sym, pd.Series(dtype=float))
            # Reconstruct approximate prices from returns
            frames[sym] = pos_series.abs()
        return pd.DataFrame(frames)

    def concentration_risk(self) -> pd.Series:
        """
        Compute Herfindahl-Hirschman Index (HHI) of gross exposure over time.

        HHI = sum(s_i^2) where s_i is fraction of gross exposure in symbol i.
        HHI close to 1/n = diversified, close to 1 = concentrated.
        """
        exp_df = self.exposure_by_time()
        if exp_df.empty:
            return pd.Series(dtype=float)

        row_sums = exp_df.sum(axis=1)
        fractions = exp_df.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
        hhi = (fractions ** 2).sum(axis=1)
        return hhi

    def pnl_attribution(self) -> pd.DataFrame:
        """
        Return a DataFrame showing % of total P&L attributed to each symbol.
        """
        total = sum(abs(v) for v in self.result.pnl_by_symbol.values())
        if total < 1e-10:
            return pd.DataFrame()

        rows = []
        for sym, pnl in self.result.pnl_by_symbol.items():
            rows.append({
                "symbol": sym,
                "pnl": pnl,
                "pct_of_total": pnl / (total + 1e-10) * 100.0,
            })
        return pd.DataFrame(rows).set_index("symbol")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_equal_weight_signal(threshold: float = 0.0) -> Callable:
    """
    Return a simple momentum signal function for testing.

    Generates +1 if close > open, -1 if close < open.
    """
    def signal_fn(bar: dict) -> float:
        o = float(bar.get("open", 0))
        c = float(bar.get("close", 0))
        if c > o * (1.0 + threshold):
            return 1.0
        elif c < o * (1.0 - threshold):
            return -1.0
        return 0.0
    return signal_fn


def build_zscore_signal(lookback: int = 20) -> Callable:
    """
    Return a mean-reversion signal based on z-score of close prices.

    Uses a closure to maintain a rolling buffer.
    Signal is clipped to [-1, 1].
    """
    price_buffer: List[float] = []

    def signal_fn(bar: dict) -> float:
        price = float(bar.get("close", 0))
        price_buffer.append(price)
        if len(price_buffer) < lookback:
            return 0.0
        window = price_buffer[-lookback:]
        mu = float(np.mean(window))
        sigma = float(np.std(window))
        if sigma < 1e-10:
            return 0.0
        z = (price - mu) / sigma
        # Reverse: buy when price is below mean, sell when above
        return float(np.clip(-z / 2.0, -1.0, 1.0))

    return signal_fn
