"""
tools/position_sizing_sandbox.py
=================================
Interactive sandbox to test different position sizing approaches on
historical data loaded from the SRFM SQLite database.

Usage:
    python tools/position_sizing_sandbox.py --method kelly --kelly-fraction 0.25
    python tools/position_sizing_sandbox.py --method all --output sizing_report.html
    python tools/position_sizing_sandbox.py --compare --output compare.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import warnings
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger("position_sizing_sandbox")
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_REPO_ROOT  = Path(__file__).parents[1]
_DB_PATH    = _REPO_ROOT / "execution" / "live_trades.db"
_DATA_DIR   = _REPO_ROOT / "data"

ANNUAL_BARS  = 252 * 26  # 15-min bars per year (approx)
RISK_FREE    = 0.0525    # annualised risk-free rate (2024 approx)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class SizingMethod(str, Enum):
    KELLY_VOL_TARGET    = "kelly_vol_target"
    EQUAL_WEIGHT        = "equal_weight"
    RISK_PARITY         = "risk_parity"
    SIGNAL_PROPORTIONAL = "signal_proportional"
    FIXED_PCT           = "fixed_pct"


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series, annual_bars: int = ANNUAL_BARS) -> float:
    if returns.std() < 1e-12:
        return 0.0
    excess = returns - RISK_FREE / annual_bars
    return float(excess.mean() / excess.std() * np.sqrt(annual_bars))


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max.replace(0, np.nan)
    return float(dd.min())


def _turnover(weights: pd.DataFrame) -> float:
    """Mean absolute daily weight change."""
    diffs = weights.diff().abs().sum(axis=1)
    return float(diffs.mean())


def _sortino(returns: pd.Series, annual_bars: int = ANNUAL_BARS) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std() < 1e-12:
        return 0.0
    excess = returns.mean() - RISK_FREE / annual_bars
    return float(excess / downside.std() * np.sqrt(annual_bars))


def _calmar(returns: pd.Series, equity: pd.Series) -> float:
    ann_ret = returns.mean() * ANNUAL_BARS
    mdd = abs(_max_drawdown(equity))
    if mdd < 1e-9:
        return 0.0
    return float(ann_ret / mdd)


def compute_metrics(returns: pd.Series, equity: pd.Series, weights: pd.DataFrame | None = None) -> dict:
    return {
        "sharpe":       _sharpe(returns),
        "sortino":      _sortino(returns),
        "calmar":       _calmar(returns, equity),
        "max_drawdown": _max_drawdown(equity),
        "total_return": float((equity.iloc[-1] / equity.iloc[0]) - 1.0) if len(equity) > 0 else 0.0,
        "ann_vol":      float(returns.std() * np.sqrt(ANNUAL_BARS)),
        "turnover":     _turnover(weights) if weights is not None else None,
        "n_bars":       len(returns),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv_from_db(conn: sqlite3.Connection, symbols: list[str] | None = None) -> pd.DataFrame:
    """
    Load OHLCV bar data from bar_data table if it exists.
    Falls back to synthetic data for testing.
    """
    try:
        query = "SELECT symbol, ts, open, high, low, close, volume FROM bar_data ORDER BY ts"
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            query = (
                f"SELECT symbol, ts, open, high, low, close, volume FROM bar_data "
                f"WHERE symbol IN ({placeholders}) ORDER BY ts"
            )
            rows = pd.read_sql_query(query, conn, params=symbols)
        else:
            rows = pd.read_sql_query(query, conn)
        rows["ts"] = pd.to_datetime(rows["ts"])
        rows = rows.set_index(["ts", "symbol"]).sort_index()
        return rows
    except Exception as exc:
        log.debug("load_ohlcv_from_db: table missing or error: %s", exc)
        return pd.DataFrame()


def load_signals_from_db(conn: sqlite3.Connection, symbols: list[str] | None = None) -> pd.DataFrame:
    """Load LARSA signal data (bh_mass, hurst, nav_omega, etc.) from nav_state."""
    try:
        query = (
            "SELECT ts, symbol, bh_mass_15m, bh_mass_1h, hurst_h, nav_omega, "
            "signal_strength FROM nav_state ORDER BY ts"
        )
        if symbols:
            placeholders = ",".join("?" * len(symbols))
            query = (
                "SELECT ts, symbol, bh_mass_15m, bh_mass_1h, hurst_h, nav_omega, "
                f"signal_strength FROM nav_state WHERE symbol IN ({placeholders}) ORDER BY ts"
            )
            df = pd.read_sql_query(query, conn, params=symbols)
        else:
            df = pd.read_sql_query(query, conn)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index(["ts", "symbol"]).sort_index()
        return df
    except Exception as exc:
        log.debug("load_signals_from_db: %s", exc)
        return pd.DataFrame()


def _generate_synthetic_ohlcv(
    symbols: list[str] = ("BTC", "ETH", "XRP"),
    n_bars: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing when DB has no data."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_bars, freq="15min", tz="UTC")
    rows = []
    for sym in symbols:
        base_price = {"BTC": 30000.0, "ETH": 2000.0, "XRP": 0.50}.get(sym, 100.0)
        returns    = rng.normal(0.0002, 0.01, n_bars)
        closes     = base_price * np.cumprod(1 + returns)
        highs      = closes * (1 + rng.uniform(0, 0.005, n_bars))
        lows       = closes * (1 - rng.uniform(0, 0.005, n_bars))
        opens      = np.roll(closes, 1)
        opens[0]   = base_price
        vols       = rng.exponential(1e6, n_bars)
        df_sym     = pd.DataFrame({
            "symbol": sym, "open": opens, "high": highs,
            "low": lows, "close": closes, "volume": vols,
        }, index=idx)
        df_sym.index.name = "ts"
        rows.append(df_sym.reset_index().set_index(["ts", "symbol"]))
    return pd.concat(rows).sort_index()


def _generate_synthetic_signals(ohlcv: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic LARSA signals aligned to OHLCV data."""
    rng = np.random.default_rng(seed)
    rows = []
    for (ts, sym), row in ohlcv.iterrows():
        rows.append({
            "ts": ts,
            "symbol": sym,
            "bh_mass_15m":    float(rng.uniform(0, 1)),
            "bh_mass_1h":     float(rng.uniform(0, 1)),
            "hurst_h":        float(rng.uniform(0.3, 0.7)),
            "nav_omega":      float(rng.exponential(0.001)),
            "signal_strength": float(rng.uniform(-1, 1)),
        })
    df = pd.DataFrame(rows).set_index(["ts", "symbol"]).sort_index()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sizing implementations
# ─────────────────────────────────────────────────────────────────────────────

def _close_pivot(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Return close prices as (ts x symbol) pivot."""
    close_col = ohlcv["close"] if "close" in ohlcv.columns else ohlcv.iloc[:, 3]
    return close_col.reset_index().pivot(index="ts", columns="symbol", values="close")


def _returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def _rolling_vol(returns: pd.DataFrame, window: int = 64) -> pd.DataFrame:
    return returns.rolling(window, min_periods=max(2, window // 4)).std().fillna(1e-6)


def _size_kelly_vol_target(
    returns: pd.DataFrame,
    signals: pd.DataFrame | None,
    kelly_fraction: float = 0.25,
    vol_target: float = 0.10,
    **kwargs,
) -> pd.DataFrame:
    """
    Kelly-scaled vol-targeting sizing.
    Weight = kelly_fraction * (vol_target / realized_vol) * sign(signal)
    Clipped to [-1, 1] and normalised so |weights|.sum() <= 1.
    """
    vol = _rolling_vol(returns)
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for sym in returns.columns:
        sig = 1.0
        if signals is not None and sym in signals.columns:
            sig = np.sign(signals[sym]).replace(0, 1)
        w = kelly_fraction * (vol_target / (vol[sym].clip(lower=1e-6) * np.sqrt(ANNUAL_BARS)))
        weights[sym] = (w * sig).clip(-1, 1)
    # normalise
    row_sum = weights.abs().sum(axis=1).clip(lower=1e-9)
    return (weights.T / row_sum).T.clip(-1, 1)


def _size_equal_weight(
    returns: pd.DataFrame,
    signals: pd.DataFrame | None,
    **kwargs,
) -> pd.DataFrame:
    n = len(returns.columns)
    weights = pd.DataFrame(1.0 / n, index=returns.index, columns=returns.columns)
    if signals is not None:
        for sym in weights.columns:
            if sym in signals.columns:
                weights[sym] *= np.sign(signals[sym]).replace(0, 1)
    return weights


def _size_risk_parity(
    returns: pd.DataFrame,
    signals: pd.DataFrame | None,
    lookback: int = 64,
    **kwargs,
) -> pd.DataFrame:
    """Inverse vol weighting (simple risk parity)."""
    vol = _rolling_vol(returns, lookback)
    inv_vol = 1.0 / vol.clip(lower=1e-9)
    row_sum = inv_vol.sum(axis=1).clip(lower=1e-9)
    weights = (inv_vol.T / row_sum).T
    if signals is not None:
        for sym in weights.columns:
            if sym in signals.columns:
                weights[sym] *= np.sign(signals[sym]).replace(0, 1)
    return weights.clip(-1, 1)


def _size_signal_proportional(
    returns: pd.DataFrame,
    signals: pd.DataFrame | None,
    scale: float = 1.0,
    **kwargs,
) -> pd.DataFrame:
    """Weight proportional to abs(signal), direction from signal sign."""
    if signals is None:
        return _size_equal_weight(returns, None)
    weights = signals.clip(-1, 1) * scale
    row_sum = weights.abs().sum(axis=1).clip(lower=1e-9)
    return (weights.T / row_sum).T


def _size_fixed_pct(
    returns: pd.DataFrame,
    signals: pd.DataFrame | None,
    fixed_pct: float = 0.10,
    **kwargs,
) -> pd.DataFrame:
    """Fixed percentage per symbol that has a signal."""
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for sym in returns.columns:
        if signals is not None and sym in signals.columns:
            weights[sym] = np.where(signals[sym].abs() > 0.05, fixed_pct, 0.0)
        else:
            weights[sym] = fixed_pct
    if signals is not None:
        for sym in weights.columns:
            if sym in signals.columns:
                weights[sym] *= np.sign(signals[sym]).replace(0, 1)
    return weights


_SIZER_MAP: dict[SizingMethod, Callable] = {
    SizingMethod.KELLY_VOL_TARGET:    _size_kelly_vol_target,
    SizingMethod.EQUAL_WEIGHT:        _size_equal_weight,
    SizingMethod.RISK_PARITY:         _size_risk_parity,
    SizingMethod.SIGNAL_PROPORTIONAL: _size_signal_proportional,
    SizingMethod.FIXED_PCT:           _size_fixed_pct,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main sandbox class
# ─────────────────────────────────────────────────────────────────────────────

class SizingSandbox:
    """
    Interactive sandbox to test position sizing approaches on historical LARSA data.

    Usage:
        sb = SizingSandbox(db_path="execution/live_trades.db")
        result = sb.simulate(SizingMethod.KELLY_VOL_TARGET, {"kelly_fraction": 0.25})
        comparison = sb.compare_methods([SizingMethod.KELLY_VOL_TARGET, SizingMethod.EQUAL_WEIGHT])
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        symbols: list[str] | None = None,
        synthetic_fallback: bool = True,
    ) -> None:
        self.db_path    = Path(db_path) if db_path else _DB_PATH
        self.symbols    = symbols
        self._ohlcv:   pd.DataFrame | None = None
        self._signals: pd.DataFrame | None = None
        self._prices:  pd.DataFrame | None = None
        self._returns: pd.DataFrame | None = None
        self._sig_pivot: pd.DataFrame | None = None
        self._synthetic_fallback = synthetic_fallback
        self._loaded = False

    # -- Data loading --

    def load(self, force: bool = False) -> "SizingSandbox":
        """Load OHLCV and signal data from SQLite (or synthetic fallback)."""
        if self._loaded and not force:
            return self
        conn = None
        if self.db_path.exists():
            try:
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            except Exception as exc:
                log.warning("Cannot open DB: %s", exc)

        if conn:
            self._ohlcv   = load_ohlcv_from_db(conn, self.symbols)
            self._signals = load_signals_from_db(conn, self.symbols)
            conn.close()

        # Fall back to synthetic if tables are empty
        if (self._ohlcv is None or self._ohlcv.empty) and self._synthetic_fallback:
            log.info("No OHLCV data in DB -- using synthetic data for demonstration.")
            syms = self.symbols or ["BTC", "ETH", "XRP"]
            self._ohlcv   = _generate_synthetic_ohlcv(syms)
            self._signals = _generate_synthetic_signals(self._ohlcv)

        # Build pivots
        if self._ohlcv is not None and not self._ohlcv.empty:
            self._prices  = _close_pivot(self._ohlcv)
            self._returns = _returns_from_prices(self._prices)

        if self._signals is not None and not self._signals.empty:
            try:
                sig_df = self._signals["signal_strength"]
                self._sig_pivot = sig_df.reset_index().pivot(
                    index="ts", columns="symbol", values="signal_strength"
                )
            except Exception:
                self._sig_pivot = None

        self._loaded = True
        log.info(
            "Loaded %d bars x %d symbols",
            len(self._prices) if self._prices is not None else 0,
            len(self._prices.columns) if self._prices is not None else 0,
        )
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
        if self._returns is None or self._returns.empty:
            raise RuntimeError("No OHLCV data available. Check DB path or use synthetic_fallback=True.")

    # -- Core simulation --

    def simulate(
        self,
        method: SizingMethod | str,
        params: dict | None = None,
    ) -> pd.DataFrame:
        """
        Run a sizing method on the full historical data.

        Returns a DataFrame with columns:
            weights_<sym>, portfolio_return, equity, drawdown
        """
        self._ensure_loaded()
        method  = SizingMethod(method) if isinstance(method, str) else method
        params  = params or {}
        sizer   = _SIZER_MAP[method]
        weights = sizer(self._returns, self._sig_pivot, **params)
        # Align to returns index
        weights = weights.reindex(self._returns.index).fillna(0.0)
        # Portfolio returns (shift weights by 1 -- applied next bar)
        shifted   = weights.shift(1).fillna(0.0)
        port_ret  = (shifted * self._returns).sum(axis=1)
        equity    = (1 + port_ret).cumprod()
        roll_max  = equity.cummax()
        drawdown  = (equity - roll_max) / roll_max.replace(0.0, np.nan)

        result = pd.DataFrame({"portfolio_return": port_ret, "equity": equity, "drawdown": drawdown})
        for sym in weights.columns:
            result[f"weight_{sym}"] = weights[sym]
        result.attrs["method"]  = method.value
        result.attrs["params"]  = params
        result.attrs["metrics"] = compute_metrics(port_ret, equity, weights)
        return result

    def compare_methods(
        self,
        methods_list: list[SizingMethod | str],
        params_list: list[dict] | None = None,
    ) -> dict:
        """
        Side-by-side comparison of multiple sizing methods.

        Returns a dict keyed by method name with metrics dicts.
        """
        self._ensure_loaded()
        params_list = params_list or [{} for _ in methods_list]
        results = {}
        for method, params in zip(methods_list, params_list):
            method = SizingMethod(method) if isinstance(method, str) else method
            try:
                df = self.simulate(method, params)
                results[method.value] = df.attrs.get("metrics", {})
                results[method.value]["equity_series"] = df["equity"]
            except Exception as exc:
                log.warning("compare_methods: %s failed: %s", method, exc)
                results[method.value] = {"error": str(exc)}
        return results

    def optimize_kelly_fraction(
        self,
        grid: list[float] | None = None,
        vol_target: float = 0.10,
        train_frac: float = 0.7,
    ) -> dict:
        """
        Grid-search Kelly fraction on the training portion of data.

        Returns a dict with best_fraction, all_scores, and train/test metrics.
        """
        self._ensure_loaded()
        grid = grid or [0.1, 0.25, 0.5, 0.75, 1.0]
        n_train = int(len(self._returns) * train_frac)
        train_ret = self._returns.iloc[:n_train]
        test_ret  = self._returns.iloc[n_train:]
        train_sig = (self._sig_pivot.iloc[:n_train] if self._sig_pivot is not None else None)
        test_sig  = (self._sig_pivot.iloc[n_train:]  if self._sig_pivot is not None else None)

        scores = {}
        for kf in grid:
            params = {"kelly_fraction": kf, "vol_target": vol_target}
            w_train = _size_kelly_vol_target(train_ret, train_sig, **params)
            shifted  = w_train.shift(1).fillna(0.0)
            port_ret = (shifted * train_ret).sum(axis=1)
            eq       = (1 + port_ret).cumprod()
            scores[kf] = _sharpe(port_ret)

        best_kf = max(scores, key=lambda k: scores[k])

        # Evaluate best on test set
        params = {"kelly_fraction": best_kf, "vol_target": vol_target}
        w_test   = _size_kelly_vol_target(test_ret, test_sig, **params)
        shifted  = w_test.shift(1).fillna(0.0)
        test_port_ret = (shifted * test_ret).sum(axis=1)
        test_eq  = (1 + test_port_ret).cumprod()
        test_metrics = compute_metrics(test_port_ret, test_eq, w_test)

        log.info("Best Kelly fraction: %.2f  (train Sharpe: %.2f)", best_kf, scores[best_kf])
        return {
            "best_fraction":  best_kf,
            "all_scores":     scores,
            "test_metrics":   test_metrics,
            "train_sharpes":  scores,
        }

    def regime_conditioned_sizing(
        self,
        regime_fn: Callable[[pd.DataFrame], pd.Series],
        sizing_fn: Callable[[str, pd.DataFrame, pd.DataFrame | None], pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Apply different sizing logic per market regime.

        regime_fn(prices) -> pd.Series[str] of regime labels aligned to returns index.
        sizing_fn(regime_label, returns_slice, signals_slice) -> weights DataFrame.

        Returns combined portfolio return series with equity curve.
        """
        self._ensure_loaded()
        regimes = regime_fn(self._prices)
        regimes = regimes.reindex(self._returns.index).fillna("UNKNOWN")
        unique_regimes = regimes.unique()

        all_weights = pd.DataFrame(0.0, index=self._returns.index, columns=self._returns.columns)
        for reg in unique_regimes:
            mask = regimes == reg
            ret_slice = self._returns[mask]
            sig_slice = (self._sig_pivot[mask] if self._sig_pivot is not None else None)
            if len(ret_slice) == 0:
                continue
            try:
                w_slice = sizing_fn(reg, ret_slice, sig_slice)
                all_weights.loc[mask] = w_slice.values

            except Exception as exc:
                log.warning("regime_conditioned_sizing: regime=%s failed: %s", reg, exc)

        shifted  = all_weights.shift(1).fillna(0.0)
        port_ret = (shifted * self._returns).sum(axis=1)
        equity   = (1 + port_ret).cumprod()
        drawdown = ((equity - equity.cummax()) / equity.cummax().replace(0.0, np.nan))

        result = pd.DataFrame({"portfolio_return": port_ret, "equity": equity, "drawdown": drawdown})
        for sym in all_weights.columns:
            result[f"weight_{sym}"] = all_weights[sym]
        result.attrs["metrics"] = compute_metrics(port_ret, equity, all_weights)
        return result

    # -- Reporting --

    def summary_table(self, comparison: dict) -> pd.DataFrame:
        """Convert compare_methods output to a clean summary DataFrame."""
        rows = []
        for method, metrics in comparison.items():
            if "error" in metrics:
                rows.append({"method": method, "error": metrics["error"]})
                continue
            row = {"method": method}
            for k in ("sharpe", "sortino", "calmar", "max_drawdown", "total_return", "ann_vol", "turnover"):
                row[k] = metrics.get(k)
            rows.append(row)
        return pd.DataFrame(rows).set_index("method")

    def to_html_report(self, output_path: str | Path, comparison: dict | None = None) -> None:
        """Write an HTML report with comparison tables and equity curves."""
        output_path = Path(output_path)
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=2, cols=1, subplot_titles=("Equity Curves", "Drawdown"))
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
            if comparison:
                for i, (method, metrics) in enumerate(comparison.items()):
                    eq = metrics.get("equity_series")
                    if eq is None:
                        continue
                    color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=method,
                                             line=dict(color=color)), row=1, col=1)
            fig.update_layout(title=f"Sizing Method Comparison -- {len(comparison or {})} methods",
                              height=600)
            table_html = ""
            if comparison:
                tbl = self.summary_table(comparison)
                table_html = tbl.to_html(float_format="{:.4f}".format)
            html_content = f"""<!DOCTYPE html>
<html>
<head><title>Position Sizing Report</title>
<style>body{{font-family:monospace;background:#1a1a2e;color:#eee;}}
table{{border-collapse:collapse;width:100%;}}
th,td{{border:1px solid #555;padding:6px;text-align:right;}}
th{{background:#16213e;}}
</style></head>
<body>
<h1>Position Sizing Sandbox Report</h1>
<h2>Equity Curve Comparison</h2>
{fig.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>Metrics Summary</h2>
{table_html}
</body></html>"""
            output_path.write_text(html_content, encoding="utf-8")
            log.info("Report written to %s", output_path)
        except ImportError:
            # Plain CSV fallback
            if comparison:
                tbl = self.summary_table(comparison)
                csv_path = output_path.with_suffix(".csv")
                tbl.to_csv(csv_path)
                log.info("Plotly not available -- CSV written to %s", csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Position sizing sandbox for LARSA")
    parser.add_argument("--method", type=str, default="kelly",
                        help="Sizing method: kelly|equal|risk_parity|signal|fixed|all")
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--vol-target", type=float, default=0.10)
    parser.add_argument("--fixed-pct", type=float, default=0.10)
    parser.add_argument("--compare", action="store_true",
                        help="Compare all sizing methods")
    parser.add_argument("--optimize-kelly", action="store_true",
                        help="Grid-search best Kelly fraction")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for HTML/CSV report")
    parser.add_argument("--db", type=str, default=str(_DB_PATH))
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols, e.g. BTC,ETH")
    return parser.parse_args()


_METHOD_ALIASES = {
    "kelly":       SizingMethod.KELLY_VOL_TARGET,
    "equal":       SizingMethod.EQUAL_WEIGHT,
    "risk_parity": SizingMethod.RISK_PARITY,
    "signal":      SizingMethod.SIGNAL_PROPORTIONAL,
    "fixed":       SizingMethod.FIXED_PCT,
}

ALL_METHODS = list(SizingMethod)


def main() -> None:
    args    = _parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    sb      = SizingSandbox(db_path=args.db, symbols=symbols).load()

    if args.optimize_kelly:
        result = sb.optimize_kelly_fraction()
        print(f"\nBest Kelly fraction: {result['best_fraction']}")
        print("Sharpe by fraction:")
        for kf, sharpe in result["all_scores"].items():
            print(f"  kf={kf:.2f}  sharpe={sharpe:.4f}")
        print("\nTest metrics:")
        for k, v in result["test_metrics"].items():
            if k != "equity_series":
                print(f"  {k}: {v}")
        return

    if args.compare or args.method == "all":
        methods    = ALL_METHODS
        params_map = [
            {"kelly_fraction": args.kelly_fraction, "vol_target": args.vol_target},
            {},
            {},
            {},
            {"fixed_pct": args.fixed_pct},
        ]
        comparison = sb.compare_methods(methods, params_map)
        tbl = sb.summary_table(comparison)
        print("\n" + "=" * 70)
        print("SIZING METHOD COMPARISON")
        print("=" * 70)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(tbl.to_string(float_format="{:.4f}".format))
        if args.output:
            sb.to_html_report(args.output, comparison)
        return

    # Single method
    method_key = args.method.lower()
    method = _METHOD_ALIASES.get(method_key)
    if method is None:
        try:
            method = SizingMethod(method_key)
        except ValueError:
            print(f"Unknown method '{args.method}'. Choose from: {list(_METHOD_ALIASES.keys())+['all']}")
            sys.exit(1)

    params: dict = {}
    if method == SizingMethod.KELLY_VOL_TARGET:
        params = {"kelly_fraction": args.kelly_fraction, "vol_target": args.vol_target}
    elif method == SizingMethod.FIXED_PCT:
        params = {"fixed_pct": args.fixed_pct}

    result = sb.simulate(method, params)
    metrics = result.attrs.get("metrics", {})
    print(f"\n{'='*50}")
    print(f"Method: {method.value}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    if args.output:
        comparison = {method.value: {**metrics, "equity_series": result["equity"]}}
        sb.to_html_report(args.output, comparison)


if __name__ == "__main__":
    main()
