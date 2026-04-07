"""
stress_test_backtest.py -- Stress scenario backtesting for SRFM.

Provides a library of pre-built stress scenarios and a runner that compares
strategy performance under each scenario versus the baseline.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BARS_PER_YEAR = 252 * 26  # 15-min bars


# ---------------------------------------------------------------------------
# Scenario result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Performance metrics for a single stress scenario run."""
    scenario_name: str
    max_drawdown: float          # peak-to-trough, negative
    recovery_bars: int           # bars to recover from max drawdown trough
    sharpe_during: float         # Sharpe ratio during scenario
    worst_day_return: float      # worst single bar return
    n_stop_outs: int             # number of times a stop-loss was triggered
    total_return: float          # scenario total return
    equity_curve: pd.Series      # equity during scenario
    n_bars: int                  # total bars in scenario
    baseline_sharpe: float = 0.0 # Sharpe of unmodified run for comparison
    vol_ratio: float = 1.0       # realized vol / baseline vol

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "max_drawdown": self.max_drawdown,
            "recovery_bars": self.recovery_bars,
            "sharpe_during": self.sharpe_during,
            "worst_day_return": self.worst_day_return,
            "n_stop_outs": self.n_stop_outs,
            "total_return": self.total_return,
            "n_bars": self.n_bars,
            "baseline_sharpe": self.baseline_sharpe,
            "vol_ratio": self.vol_ratio,
        }


# ---------------------------------------------------------------------------
# Scenario modifier functions
# ---------------------------------------------------------------------------

def scenario_vol_spike(
    bars: pd.DataFrame,
    start_bar: int = 50,
    n_bars: int = 20,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Multiply bar high-low range (volatility proxy) by multiplier for n_bars
    starting at start_bar.

    Also amplifies bar-to-bar returns (open/close) proportionally.
    """
    df = bars.copy()
    end_bar = min(start_bar + n_bars, len(df))
    idx = df.index[start_bar:end_bar]

    # Scale high and low around the midpoint
    mid = (df.loc[idx, "high"] + df.loc[idx, "low"]) / 2.0
    half_range = (df.loc[idx, "high"] - df.loc[idx, "low"]) / 2.0
    df.loc[idx, "high"] = mid + half_range * multiplier
    df.loc[idx, "low"] = mid - half_range * multiplier
    df.loc[idx, "low"] = df.loc[idx, "low"].clip(lower=0.01)

    # Scale bar returns too
    open_prices = df.loc[idx, "open"].values
    close_prices = df.loc[idx, "close"].values
    bar_rets = (close_prices - open_prices) / np.maximum(open_prices, 1e-8)
    new_rets = bar_rets * multiplier
    df.loc[idx, "close"] = open_prices * (1.0 + new_rets)
    df.loc[idx, "close"] = df.loc[idx, "close"].clip(lower=0.01)

    logger.info(
        "scenario_vol_spike: bars %d-%d, multiplier=%.1fx", start_bar, end_bar, multiplier
    )
    return df


def scenario_gap_down(
    bars: pd.DataFrame,
    gap_bar: int = 100,
    gap_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Inject a sudden gap down of gap_pct at bar gap_bar.

    All prices (open/high/low/close) from gap_bar onward are shifted down.
    This simulates overnight gap risk.
    """
    df = bars.copy()
    if gap_bar >= len(df):
        logger.warning("gap_bar %d >= len(bars) %d, skipping", gap_bar, len(df))
        return df

    # Apply gap at gap_bar open
    gap_idx = df.index[gap_bar]
    shift = 1.0 - gap_pct

    # Shift all subsequent bars down by the gap
    idx_after = df.index[gap_bar:]
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df.loc[idx_after, col] = df.loc[idx_after, col] * shift

    df.loc[:, "low"] = df.loc[:, "low"].clip(lower=0.01)
    logger.info("scenario_gap_down: bar %d, gap=%.1f%%", gap_bar, gap_pct * 100)
    return df


def scenario_liquidity_crisis(
    bars: pd.DataFrame,
    start_bar: int = 50,
    n_bars: int = 40,
    spread_mult: float = 5.0,
    fill_rate: float = 0.50,
) -> pd.DataFrame:
    """
    Widen spreads and reduce fills to simulate liquidity crisis.

    Implemented by widening high-low ranges (spread proxy) and injecting
    noise into close prices (representing poor fill quality).
    fill_rate: fraction of intended trade that fills (rest carries over).
    """
    df = bars.copy()
    end_bar = min(start_bar + n_bars, len(df))
    idx = df.index[start_bar:end_bar]

    # Widen high-low range as spread proxy
    mid = (df.loc[idx, "high"] + df.loc[idx, "low"]) / 2.0
    half_rng = (df.loc[idx, "high"] - df.loc[idx, "low"]) / 2.0
    df.loc[idx, "high"] = mid + half_rng * spread_mult
    df.loc[idx, "low"] = (mid - half_rng * spread_mult).clip(lower=0.01)

    # Add "liquidity_crisis" metadata column for downstream handlers
    df["liquidity_mult"] = 1.0
    df.loc[idx, "liquidity_mult"] = spread_mult
    df["fill_rate"] = 1.0
    df.loc[idx, "fill_rate"] = fill_rate

    logger.info(
        "scenario_liquidity_crisis: bars %d-%d, spread_mult=%.1fx, fill_rate=%.0f%%",
        start_bar, end_bar, spread_mult, fill_rate * 100,
    )
    return df


def scenario_correlation_spike(
    bars_dict: Dict[str, pd.DataFrame],
    start_bar: int = 50,
    n_bars: int = 30,
    correlation_target: float = 0.95,
) -> Dict[str, pd.DataFrame]:
    """
    Force all assets in bars_dict to move together during [start_bar, end_bar].

    Replaces each asset's bar returns with a weighted blend of the asset's
    own returns and a shared market return, to achieve near-perfect correlation.

    Returns updated bars_dict.
    """
    modified = {sym: df.copy() for sym, df in bars_dict.items()}
    symbols = list(modified.keys())
    if not symbols:
        return modified

    # Build a common "market" return using the first symbol
    ref_sym = symbols[0]
    ref_df = modified[ref_sym]
    n_total = len(ref_df)
    end_bar = min(start_bar + n_bars, n_total)

    if start_bar >= n_total:
        return modified

    ref_idx = ref_df.index[start_bar:end_bar]
    ref_open = ref_df.loc[ref_idx, "open"].values
    ref_close = ref_df.loc[ref_idx, "close"].values
    market_ret = (ref_close - ref_open) / np.maximum(ref_open, 1e-8)

    alpha = correlation_target  # weight on market return

    for sym in symbols[1:]:
        df = modified[sym]
        if start_bar >= len(df):
            continue
        sym_idx = df.index[start_bar:min(start_bar + n_bars, len(df))]
        sym_open = df.loc[sym_idx, "open"].values
        sym_close = df.loc[sym_idx, "close"].values
        sym_ret = (sym_close - sym_open) / np.maximum(sym_open, 1e-8)

        # Blend: blended_ret = alpha * market_ret + (1-alpha) * sym_ret
        min_len = min(len(market_ret), len(sym_ret))
        blended = alpha * market_ret[:min_len] + (1.0 - alpha) * sym_ret[:min_len]
        new_close = sym_open[:min_len] * (1.0 + blended)
        df.loc[sym_idx[:min_len], "close"] = np.maximum(new_close, 0.01)

    logger.info(
        "scenario_correlation_spike: bars %d-%d, target_corr=%.2f",
        start_bar, end_bar, correlation_target,
    )
    return modified


def scenario_flash_crash(
    bars: pd.DataFrame,
    crash_bar: int = 100,
    drop_pct: float = 0.10,
    recovery_bars: int = 5,
) -> pd.DataFrame:
    """
    Inject a flash crash: price drops drop_pct, then recovers over recovery_bars bars.

    The crash happens over a single bar (bar crash_bar), and recovery
    is linear over the next recovery_bars bars.
    """
    df = bars.copy()
    if crash_bar >= len(df):
        return df

    crash_idx = df.index[crash_bar]
    pre_crash_close = float(df.iloc[crash_bar - 1]["close"]) if crash_bar > 0 else float(df.iloc[0]["close"])

    # Crash bar: open normal, close is crashed
    crash_low = pre_crash_close * (1.0 - drop_pct)
    df.loc[crash_idx, "low"] = max(crash_low * 0.99, 0.01)
    df.loc[crash_idx, "close"] = crash_low

    # Recovery: linearly recover from crash_low back to pre_crash_close
    rec_end = min(crash_bar + 1 + recovery_bars, len(df))
    rec_idx = df.index[crash_bar + 1: rec_end]
    n_rec = len(rec_idx)

    if n_rec > 0:
        recovery_prices = np.linspace(crash_low, pre_crash_close, n_rec + 1)[1:]
        for i, ridx in enumerate(rec_idx):
            df.loc[ridx, "open"] = recovery_prices[i] * 0.999
            df.loc[ridx, "close"] = recovery_prices[i]
            df.loc[ridx, "high"] = recovery_prices[i] * 1.002
            df.loc[ridx, "low"] = max(recovery_prices[i] * 0.998, 0.01)

    logger.info(
        "scenario_flash_crash: bar %d, drop=%.1f%%, recovery=%d bars",
        crash_bar, drop_pct * 100, recovery_bars,
    )
    return df


def scenario_bear_market(
    bars: pd.DataFrame,
    total_drop_pct: float = 0.40,
    n_bars: int = 252,
    start_bar: int = 0,
) -> pd.DataFrame:
    """
    Apply a gradual bear market drift of total_drop_pct over n_bars.

    Each bar's returns are reduced by a constant daily drift.
    """
    df = bars.copy()
    end_bar = min(start_bar + n_bars, len(df))
    actual_n = end_bar - start_bar

    if actual_n <= 0:
        return df

    # Drift per bar to achieve total_drop_pct
    per_bar_drift = (1.0 - total_drop_pct) ** (1.0 / actual_n) - 1.0

    idx = df.index[start_bar:end_bar]
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            current = df.loc[idx, col].values
            # Apply cumulative drift
            cumulative = (1.0 + per_bar_drift) ** np.arange(actual_n)
            df.loc[idx, col] = current * cumulative
            df.loc[idx, col] = df.loc[idx, col].clip(lower=0.01)

    logger.info(
        "scenario_bear_market: bars %d-%d, drift=%.4f/bar, total=%.1f%%",
        start_bar, end_bar, per_bar_drift, total_drop_pct * 100,
    )
    return df


# ---------------------------------------------------------------------------
# Simple inline backtest runner (self-contained for scenarios)
# ---------------------------------------------------------------------------

def _run_inline_backtest(
    bars: pd.DataFrame,
    signal_fn: Callable[[dict], float],
    initial_capital: float = 100_000.0,
    commission_bps: float = 5.0,
    slippage_bps: float = 3.0,
    max_position_frac: float = 0.25,
    stop_loss_pct: Optional[float] = None,   # e.g. 0.02 for 2% stop
) -> Tuple[pd.Series, int]:
    """
    Minimal bar-iteration backtest. Returns (equity_series, n_stop_outs).
    """
    cash = initial_capital
    position = 0.0
    entry_price = 0.0
    equity_log: Dict[pd.Timestamp, float] = {}
    n_stop_outs = 0

    for ts, row in bars.iterrows():
        bar = row.to_dict()
        price = float(bar.get("close", 0))
        if price <= 0:
            continue

        # Check stop loss
        if stop_loss_pct is not None and position != 0.0 and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price * np.sign(position)
            if pnl_pct < -stop_loss_pct:
                # Close position at stop
                fill = price * (1.0 - slippage_bps / 10_000.0) if position > 0 else price * (1.0 + slippage_bps / 10_000.0)
                cash += position * fill - abs(position) * fill * commission_bps / 10_000.0
                position = 0.0
                n_stop_outs += 1

        # Generate signal
        try:
            signal = float(signal_fn(bar))
        except Exception:
            signal = 0.0

        signal = max(-1.0, min(1.0, signal))

        # Position sizing
        notional = initial_capital * max_position_frac * abs(signal)
        desired = (notional / price) * (1.0 if signal > 0 else -1.0) if abs(signal) > 1e-8 else 0.0

        delta = desired - position
        if abs(delta) > 1e-8:
            direction = 1.0 if delta > 0 else -1.0
            fill = price * (1.0 + direction * slippage_bps / 10_000.0)
            comm = abs(delta) * fill * commission_bps / 10_000.0
            cash -= delta * fill + comm
            if position == 0.0 and desired != 0.0:
                entry_price = fill
            position = desired

        equity_log[ts] = cash + position * price

    return pd.Series(equity_log, name="equity").sort_index(), n_stop_outs


def _compute_scenario_stats(
    equity: pd.Series,
    n_stop_outs: int,
    scenario_name: str,
    baseline_sharpe: float = 0.0,
) -> ScenarioResult:
    """Build ScenarioResult from equity curve."""
    if equity.empty or len(equity) < 2:
        return ScenarioResult(
            scenario_name=scenario_name,
            max_drawdown=0.0,
            recovery_bars=0,
            sharpe_during=0.0,
            worst_day_return=0.0,
            n_stop_outs=n_stop_outs,
            total_return=0.0,
            equity_curve=equity,
            n_bars=len(equity),
            baseline_sharpe=baseline_sharpe,
        )

    rets = equity.pct_change().dropna()
    total_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1.0)

    mu = float(rets.mean())
    sigma = float(rets.std())
    sharpe = mu / sigma * np.sqrt(BARS_PER_YEAR) if sigma > 1e-10 else 0.0

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())
    worst_bar = float(rets.min())

    # Recovery: bars from trough to recovery of prior peak
    trough_idx = int(dd.argmin())
    recovery_bars = 0
    if max_dd < -1e-4:
        peak_before_trough = float(peak.iloc[trough_idx])
        for i in range(trough_idx, len(equity)):
            if float(equity.iloc[i]) >= peak_before_trough:
                recovery_bars = i - trough_idx
                break
        else:
            recovery_bars = len(equity) - trough_idx  # did not recover

    # Realized vol ratio vs baseline
    baseline_vol = 0.02  # default assumption
    vol_ratio = (sigma * np.sqrt(BARS_PER_YEAR)) / max(baseline_vol, 1e-10)

    return ScenarioResult(
        scenario_name=scenario_name,
        max_drawdown=max_dd,
        recovery_bars=recovery_bars,
        sharpe_during=sharpe,
        worst_day_return=worst_bar,
        n_stop_outs=n_stop_outs,
        total_return=total_ret,
        equity_curve=equity,
        n_bars=len(equity),
        baseline_sharpe=baseline_sharpe,
        vol_ratio=vol_ratio,
    )


# ---------------------------------------------------------------------------
# Main stress test runner
# ---------------------------------------------------------------------------

class StressTestBacktest:
    """
    Run strategy through a battery of stress scenarios.

    Usage:
      st = StressTestBacktest(bars_df, signal_fn)
      results = st.run_all_scenarios()
      comparison = st.compare_scenarios(results)
    """

    BUILTIN_SCENARIO_NAMES: List[str] = [
        "baseline",
        "vol_spike",
        "gap_down",
        "liquidity_crisis",
        "flash_crash",
        "bear_market",
    ]

    def __init__(
        self,
        bars_df: pd.DataFrame,
        signal_fn: Callable[[dict], float],
        initial_capital: float = 100_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 3.0,
        max_position_frac: float = 0.25,
        stop_loss_pct: Optional[float] = 0.05,
    ):
        """
        bars_df: OHLCV DataFrame with DatetimeIndex.
        signal_fn: fn(bar: dict) -> float in [-1, 1].
        """
        self.bars_df = bars_df.copy()
        self.bars_df.index = pd.to_datetime(self.bars_df.index)
        self.signal_fn = signal_fn
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.max_position_frac = max_position_frac
        self.stop_loss_pct = stop_loss_pct
        self._baseline_sharpe: float = 0.0

        # Custom scenario registry
        self._custom_scenarios: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}

    def register_scenario(
        self, name: str, modifier: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> None:
        """
        Register a custom scenario modifier.

        modifier: fn(bars: pd.DataFrame) -> pd.DataFrame (modified bars)
        """
        self._custom_scenarios[name] = modifier
        logger.info("Registered custom scenario: %s", name)

    def run_scenario(
        self,
        scenario_name: str,
        bars_modifier: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> ScenarioResult:
        """
        Run backtest under a single named scenario.

        If bars_modifier is None, uses built-in scenario by name.
        """
        if bars_modifier is None:
            if scenario_name in self._custom_scenarios:
                bars_modifier = self._custom_scenarios[scenario_name]
            else:
                bars_modifier = self._get_builtin_modifier(scenario_name)

        modified_bars = bars_modifier(self.bars_df)
        equity, n_stops = _run_inline_backtest(
            bars=modified_bars,
            signal_fn=self.signal_fn,
            initial_capital=self.initial_capital,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
            max_position_frac=self.max_position_frac,
            stop_loss_pct=self.stop_loss_pct,
        )

        return _compute_scenario_stats(
            equity=equity,
            n_stop_outs=n_stops,
            scenario_name=scenario_name,
            baseline_sharpe=self._baseline_sharpe,
        )

    def run_all_scenarios(self) -> Dict[str, ScenarioResult]:
        """
        Run all built-in plus registered custom scenarios.

        Returns dict of scenario_name -> ScenarioResult.
        """
        results: Dict[str, ScenarioResult] = {}

        # Run baseline first
        baseline_result = self.run_scenario("baseline")
        self._baseline_sharpe = baseline_result.sharpe_during
        results["baseline"] = baseline_result

        # Run all built-in scenarios (skip baseline, already done)
        for name in self.BUILTIN_SCENARIO_NAMES:
            if name == "baseline":
                continue
            try:
                results[name] = self.run_scenario(name)
                logger.info("Completed scenario: %s", name)
            except Exception as exc:
                logger.error("Scenario %s failed: %s", name, exc)

        # Run custom scenarios
        for name, modifier in self._custom_scenarios.items():
            try:
                results[name] = self.run_scenario(name, modifier)
                logger.info("Completed custom scenario: %s", name)
            except Exception as exc:
                logger.error("Custom scenario %s failed: %s", name, exc)

        return results

    def compare_scenarios(
        self, results: Dict[str, ScenarioResult]
    ) -> pd.DataFrame:
        """
        Build a comparison DataFrame from scenario results.

        Rows = scenarios, columns = key metrics.
        Also adds vs_baseline columns showing % change from baseline.
        """
        rows = [r.to_dict() for r in results.values()]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("scenario_name")

        # Add delta vs baseline columns for key metrics
        if "baseline" in df.index:
            for col in ["sharpe_during", "max_drawdown", "total_return", "worst_day_return"]:
                if col in df.columns:
                    baseline_val = float(df.loc["baseline", col])
                    df[f"{col}_vs_baseline"] = df[col] - baseline_val

        return df

    def worst_case_summary(self, results: Dict[str, ScenarioResult]) -> Dict[str, Any]:
        """
        Return the worst outcome across all scenarios for each key metric.
        """
        if not results:
            return {}

        max_dd = min(r.max_drawdown for r in results.values())
        min_sharpe = min(r.sharpe_during for r in results.values())
        worst_ret = min(r.worst_day_return for r in results.values())
        max_stops = max(r.n_stop_outs for r in results.values())

        worst_dd_scenario = min(results, key=lambda k: results[k].max_drawdown)
        worst_sharpe_scenario = min(results, key=lambda k: results[k].sharpe_during)

        return {
            "worst_max_drawdown": max_dd,
            "worst_max_drawdown_scenario": worst_dd_scenario,
            "worst_sharpe": min_sharpe,
            "worst_sharpe_scenario": worst_sharpe_scenario,
            "worst_single_bar_return": worst_ret,
            "max_stop_outs": max_stops,
        }

    def _get_builtin_modifier(
        self, scenario_name: str
    ) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """Return the modifier function for a built-in scenario name."""
        n_bars = len(self.bars_df)
        mid_bar = max(10, n_bars // 4)

        modifiers: Dict[str, Callable] = {
            "baseline": lambda df: df.copy(),
            "vol_spike": lambda df: scenario_vol_spike(
                df, start_bar=mid_bar, n_bars=20, multiplier=3.0
            ),
            "gap_down": lambda df: scenario_gap_down(
                df, gap_bar=min(100, n_bars // 3), gap_pct=0.05
            ),
            "liquidity_crisis": lambda df: scenario_liquidity_crisis(
                df, start_bar=mid_bar, n_bars=40, spread_mult=5.0, fill_rate=0.50
            ),
            "flash_crash": lambda df: scenario_flash_crash(
                df, crash_bar=min(100, n_bars // 3), drop_pct=0.10, recovery_bars=5
            ),
            "bear_market": lambda df: scenario_bear_market(
                df, total_drop_pct=0.40, n_bars=min(252, n_bars), start_bar=0
            ),
        }

        if scenario_name not in modifiers:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(modifiers.keys())} or register custom."
            )
        return modifiers[scenario_name]


# ---------------------------------------------------------------------------
# Multi-asset stress test
# ---------------------------------------------------------------------------

class MultiAssetStressTest:
    """
    Run stress scenarios across a portfolio of assets simultaneously.

    Supports the correlation_spike scenario which requires multiple bars DataFrames.
    """

    def __init__(
        self,
        assets: Dict[str, pd.DataFrame],   # symbol -> bars_df
        signal_fns: Dict[str, Callable],   # symbol -> signal_fn
        initial_capital: float = 100_000.0,
        stop_loss_pct: float = 0.05,
    ):
        self.assets = {sym: df.copy() for sym, df in assets.items()}
        self.signal_fns = signal_fns
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct

    def run_correlation_spike(
        self,
        start_bar: int = 50,
        n_bars: int = 30,
        correlation_target: float = 0.95,
    ) -> Dict[str, ScenarioResult]:
        """
        Apply correlation spike scenario to all assets and run each.
        """
        modified = scenario_correlation_spike(
            self.assets,
            start_bar=start_bar,
            n_bars=n_bars,
            correlation_target=correlation_target,
        )

        results: Dict[str, ScenarioResult] = {}
        for sym, bars in modified.items():
            sig_fn = self.signal_fns.get(sym)
            if sig_fn is None:
                continue
            equity, n_stops = _run_inline_backtest(
                bars=bars,
                signal_fn=sig_fn,
                initial_capital=self.initial_capital / max(len(self.assets), 1),
                stop_loss_pct=self.stop_loss_pct,
            )
            results[sym] = _compute_scenario_stats(equity, n_stops, f"correlation_spike_{sym}")

        return results

    def run_all_scenarios_per_asset(self) -> Dict[str, Dict[str, ScenarioResult]]:
        """
        Run all single-asset stress scenarios for each asset separately.

        Returns: {symbol: {scenario_name: ScenarioResult}}
        """
        all_results: Dict[str, Dict[str, ScenarioResult]] = {}
        per_asset_capital = self.initial_capital / max(len(self.assets), 1)

        for sym, bars in self.assets.items():
            sig_fn = self.signal_fns.get(sym)
            if sig_fn is None:
                continue
            runner = StressTestBacktest(
                bars_df=bars,
                signal_fn=sig_fn,
                initial_capital=per_asset_capital,
                stop_loss_pct=self.stop_loss_pct,
            )
            all_results[sym] = runner.run_all_scenarios()

        return all_results

    def portfolio_stress_summary(
        self,
        per_asset_results: Dict[str, Dict[str, ScenarioResult]],
    ) -> pd.DataFrame:
        """
        Aggregate per-asset scenario results into portfolio-level summary.

        Returns DataFrame: index=scenario_name, columns=metrics aggregated across assets.
        """
        if not per_asset_results:
            return pd.DataFrame()

        scenario_names = set()
        for asset_res in per_asset_results.values():
            scenario_names.update(asset_res.keys())

        rows = []
        for scenario in sorted(scenario_names):
            asset_sharpes = []
            asset_mdd = []
            asset_rets = []
            for sym, res_dict in per_asset_results.items():
                if scenario in res_dict:
                    r = res_dict[scenario]
                    asset_sharpes.append(r.sharpe_during)
                    asset_mdd.append(r.max_drawdown)
                    asset_rets.append(r.total_return)

            if not asset_sharpes:
                continue

            rows.append({
                "scenario": scenario,
                "avg_sharpe": float(np.mean(asset_sharpes)),
                "min_sharpe": float(np.min(asset_sharpes)),
                "avg_max_drawdown": float(np.mean(asset_mdd)),
                "worst_max_drawdown": float(np.min(asset_mdd)),
                "avg_total_return": float(np.mean(asset_rets)),
                "min_total_return": float(np.min(asset_rets)),
                "n_assets": len(asset_sharpes),
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("scenario")
