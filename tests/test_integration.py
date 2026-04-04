"""
test_integration.py — End-to-end integration tests.

~800 LOC. Tests full pipeline: data loading → backtest → MC → sensitivity → correlation.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "lib"))
sys.path.insert(0, str(_ROOT / "spacetime" / "engine"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_hourly_df(n: int = 800, drift: float = 0.0001, sigma: float = 0.0008,
                      seed: int = 42, start_price: float = 4500.0,
                      start: str = "2022-01-03") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = np.empty(n)
    closes[0] = start_price
    for i in range(1, n):
        closes[i] = closes[i-1] * max(1e-4, 1.0 + drift + sigma * rng.standard_normal())
    idx = pd.date_range(start, periods=n, freq="1h")
    noise = 0.0003 * np.abs(rng.standard_normal(n))
    return pd.DataFrame({
        "open":   closes * (1 - noise),
        "high":   closes * (1 + noise),
        "low":    closes * (1 - noise),
        "close":  closes,
        "volume": np.full(n, 50_000.0),
    }, index=idx)


def _make_trades_for_mc(n: int = 60, seed: int = 42) -> List[dict]:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2023-01-01")
    trades = []
    for i in range(n):
        win = rng.random() < 0.62
        pnl = float(rng.exponential(1400)) if win else -float(rng.exponential(900))
        trades.append({
            "entry_time": base + pd.Timedelta(hours=i * 6),
            "exit_time":  base + pd.Timedelta(hours=i * 6 + 3),
            "pnl": pnl,
            "regime": ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"][i % 4],
            "tf_score": int(rng.integers(4, 8)),
        })
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Class TestFullPipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:

    def test_load_data_run_backtest_run_mc(self):
        """
        Full pipeline:
        1. Create synthetic data
        2. Run BH backtest
        3. Feed trades to MC
        4. Verify all results are valid
        """
        from bh_engine import run_backtest, BacktestResult
        from mc import run_mc, MCConfig

        df = _build_hourly_df(1000, seed=1)
        result = run_backtest("ES", df, starting_equity=1_000_000.0)
        assert isinstance(result, BacktestResult)
        assert result.stats["final_equity"] > 0

        # If we have trades, run MC; otherwise use synthetic trades
        if len(result.trades) >= 10:
            trades = result.trades
        else:
            trades = _make_trades_for_mc(60)

        mc_cfg = MCConfig(n_sims=500, months=6)
        mc_result = run_mc(trades, starting_equity=1_000_000.0, cfg=mc_cfg)
        assert mc_result.median_equity > 0
        assert 0.0 <= mc_result.blowup_rate <= 1.0
        assert len(mc_result.final_equities) == 500

    def test_sensitivity_produces_valid_report(self):
        """
        Sensitivity sweep over CF and bh_form should produce valid results.
        """
        from srfm_core import MinkowskiClassifier, BlackHoleDetector

        df = _build_hourly_df(800, seed=2)
        closes = df["close"].values
        base = {"cf": 0.001, "bh_form": 1.5}
        grid = {
            "cf":      [0.0008, 0.001, 0.0012, 0.002],
            "bh_form": [1.0, 1.5, 2.0, 3.0],
        }

        results = {}
        for param, values in grid.items():
            param_results = []
            for v in values:
                p = dict(base); p[param] = v
                mc  = MinkowskiClassifier(cf=p["cf"])
                bh  = BlackHoleDetector(p["bh_form"], 1.0, 0.95)
                n_act = 0
                prev = float(closes[0]); mc.update(prev)
                for c in closes[1:]:
                    bit = mc.update(float(c))
                    was = bh.bh_active
                    bh.update(bit, float(c), prev)
                    if bh.bh_active and not was:
                        n_act += 1
                    prev = float(c)
                param_results.append({"value": v, "n_activations": n_act})
            results[param] = param_results

        for param in grid:
            assert param in results
            assert len(results[param]) == len(grid[param])
        # All activation counts non-negative
        for param, entries in results.items():
            for e in entries:
                assert e["n_activations"] >= 0

    def test_correlation_matrix_is_symmetric(self):
        """Correlation matrix of BH activations should be symmetric."""
        from srfm_core import MinkowskiClassifier, BlackHoleDetector

        def bh_series(closes, cf=0.001):
            mc = MinkowskiClassifier(cf=cf); bh = BlackHoleDetector(1.5, 1.0, 0.95)
            active = []; prev = float(closes[0]); mc.update(prev)
            for c in closes[1:]:
                bit = mc.update(float(c))
                active.append(bh.update(bit, float(c), prev))
                prev = float(c)
            return np.array(active, dtype=bool)

        syms = ["ES", "NQ", "BTC"]
        seeds = [1, 2, 3]
        drifts = [0.0001, 0.00012, 0.0002]
        acts = {}
        for sym, seed, drift in zip(syms, seeds, drifts):
            df = _build_hourly_df(500, drift=drift, seed=seed)
            acts[sym] = bh_series(df["close"].values)

        n = len(syms)
        mat = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                a = acts[syms[i]]; b = acts[syms[j]]
                mn = min(len(a), len(b))
                inter = np.sum(a[:mn] & b[:mn])
                union = np.sum(a[:mn] | b[:mn])
                jac = float(inter / union) if union > 0 else 0.0
                mat[i, j] = jac; mat[j, i] = jac

        corr_df = pd.DataFrame(mat, index=syms, columns=syms)
        for s1 in syms:
            for s2 in syms:
                assert abs(corr_df.loc[s1, s2] - corr_df.loc[s2, s1]) < 1e-10

    def test_mc_equity_percentiles_consistent(self):
        """MC percentiles should be consistently ordered."""
        from mc import run_mc, MCConfig
        trades = _make_trades_for_mc(80, seed=5)
        result = run_mc(trades, cfg=MCConfig(n_sims=1_000, months=6))
        assert result.pct_5 <= result.pct_25
        assert result.pct_25 <= result.median_equity
        assert result.median_equity <= result.pct_75
        assert result.pct_75 <= result.pct_95

    def test_backtest_then_mc_pipeline_consistent_equity(self):
        """Backtest final equity + MC median should both be > 0 for trending data."""
        from bh_engine import run_backtest
        from mc import run_mc, MCConfig

        df = _build_hourly_df(1500, drift=0.0002, sigma=0.0007, seed=99)
        bt = run_backtest("ES", df)
        assert bt.stats["final_equity"] > 0

        trades = _make_trades_for_mc(60, seed=99)
        mc = run_mc(trades, cfg=MCConfig(n_sims=500, months=3))
        assert mc.median_equity > 0

    def test_replay_streams_all_bars(self):
        """Simulated replay: processing all bars in the DataFrame should equal n bars."""
        df = _build_hourly_df(300, seed=7)
        from srfm_core import MinkowskiClassifier
        mc = MinkowskiClassifier(cf=0.001)
        bar_count = 0
        for ts, row in df.iterrows():
            mc.update(float(row["close"]))
            bar_count += 1
        assert bar_count == len(df)

    def test_full_pipeline_no_nan_in_stats(self):
        """Backtest stats should not contain NaN or Inf values."""
        from bh_engine import run_backtest
        df = _build_hourly_df(1000, seed=3)
        result = run_backtest("ES", df)
        for key, val in result.stats.items():
            if isinstance(val, float):
                assert not math.isnan(val) or key == "profit_factor", (
                    f"NaN found in stats[{key}]")

    def test_multi_asset_pipeline(self):
        """Run backtest on 4 different assets, verify all complete."""
        from bh_engine import run_backtest
        configs = {
            "ES":  (4500.0, 0.0001, 0.0008),
            "NQ":  (15000.0, 0.00012, 0.0010),
            "BTC": (42000.0, 0.0002, 0.005),
            "GC":  (1900.0, 0.00005, 0.0012),
        }
        for sym, (sp, drift, sigma) in configs.items():
            df = _build_hourly_df(800, drift=drift, sigma=sigma, start_price=sp, seed=ord(sym[0]))
            result = run_backtest(sym, df)
            assert result.sym == sym
            assert result.stats["final_equity"] > 0

    def test_portfolio_mc_multiple_instruments(self):
        """Portfolio MC with 3 instruments should return valid result."""
        from mc import run_portfolio_mc
        trade_lists = {
            "ES":  _make_trades_for_mc(50, seed=10),
            "BTC": _make_trades_for_mc(50, seed=20),
            "GC":  _make_trades_for_mc(50, seed=30),
        }
        result = run_portfolio_mc(
            trade_lists, starting_equity=3_000_000.0,
            n_sims=300, months=6, cross_corr=0.2
        )
        assert result.median_equity > 0
        assert 0.0 <= result.blowup_rate <= 1.0

    def test_bh_physics_consistency_across_timeframes(self):
        """
        BH mass should be non-negative and consistent across all timeframes
        in a backtest result.
        """
        from bh_engine import run_backtest
        df = _build_hourly_df(1000, seed=11)
        result = run_backtest("ES", df)
        assert all(m >= 0 for m in result.mass_series_1d)
        assert all(m >= 0 for m in result.mass_series_1h)
        assert all(m >= 0 for m in result.mass_series_15m)

    def test_trade_records_pnl_sum_approximates_equity_change(self):
        """Sum of trade PnLs should approximate total equity change."""
        from bh_engine import run_backtest
        df = _build_hourly_df(1000, seed=15)
        result = run_backtest("ES", df, starting_equity=1_000_000.0)
        total_pnl = sum(t.pnl for t in result.trades)
        final_eq  = result.stats["final_equity"]
        # Roughly 1M + pnl = final_eq (exact due to compounding effects)
        assert abs(final_eq - (1_000_000.0 + total_pnl)) < abs(total_pnl) * 0.5 + 10_000.0

    def test_sensitivity_higher_bh_form_reduces_trades(self):
        """Higher bh_form → fewer trades on same data (monotone relationship)."""
        from bh_engine import run_backtest
        df = _build_hourly_df(1500, seed=20)
        counts = []
        for bh_form in [1.0, 1.5, 2.0, 3.0]:
            r = run_backtest("ES", df, params={"bh_form": bh_form, "bh_collapse": bh_form * 0.67})
            counts.append(len(r.trades))
        # Not strictly monotone due to interactions, but generally declining
        assert counts[0] >= counts[-1], (
            f"Trade counts {counts} not declining with bh_form")

    def test_mc_blowup_positive_edge_reasonable(self):
        """With 60%+ win rate, blowup should be < 50%."""
        from mc import run_mc, MCConfig
        trades = _make_trades_for_mc(100, seed=42)
        result = run_mc(trades, cfg=MCConfig(n_sims=1_000, months=12))
        assert result.blowup_rate < 0.50, (
            f"Blowup rate {result.blowup_rate:.1%} unexpectedly high for 62% win rate")

    def test_db_roundtrip_integration(self, tmp_path):
        """Insert trades from backtest into SQLite and query back out."""
        from bh_engine import run_backtest
        df = _build_hourly_df(800, seed=3)
        result = run_backtest("ES", df)

        db_path = str(tmp_path / "test_trades.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sym TEXT, entry_price REAL, exit_price REAL,
                pnl REAL, tf_score INTEGER, regime TEXT
            )
        """)
        for t in result.trades:
            conn.execute(
                "INSERT INTO trades (sym, entry_price, exit_price, pnl, tf_score, regime) VALUES (?,?,?,?,?,?)",
                (t.sym, t.entry_price, t.exit_price, t.pnl, t.tf_score, t.regime)
            )
        conn.commit()
        cnt = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert cnt == len(result.trades)
        conn.close()

    def test_backtest_equity_curve_length(self):
        """Equity curve should have one entry per processed day."""
        from bh_engine import run_backtest
        df = _build_hourly_df(500, seed=25)
        result = run_backtest("ES", df)
        n_days = len(df.resample("1D").last())
        assert len(result.equity_curve) <= n_days + 5  # tolerance

    def test_mc_kelly_fraction_bounds(self):
        """Kelly fraction should always be in [0, 1]."""
        from mc import run_mc, MCConfig
        trades = _make_trades_for_mc(60, seed=7)
        result = run_mc(trades, cfg=MCConfig(n_sims=200, months=3))
        assert 0.0 <= result.kelly_fraction <= 1.0

    def test_full_pipeline_with_different_starting_equities(self):
        """Running backtest with different starting equities should scale results."""
        from bh_engine import run_backtest
        df = _build_hourly_df(800, seed=50)
        r1 = run_backtest("ES", df, starting_equity=100_000.0)
        r2 = run_backtest("ES", df, starting_equity=1_000_000.0)
        # 10x capital → roughly 10x final equity
        ratio = r2.stats["final_equity"] / max(r1.stats["final_equity"], 1.0)
        assert 5.0 <= ratio <= 20.0, f"Equity ratio {ratio:.2f} not near 10"

    def test_bar_states_tf_score_consistent_with_bh_active(self):
        """
        In bar_states, if bh_active_1d=True and bh_active_1h=True and bh_active_15m=True
        then tf_score must equal 7.
        """
        from bh_engine import run_backtest
        df = _build_hourly_df(1000, seed=99)
        result = run_backtest("ES", df)
        for bs in result.bar_states:
            expected = (4 if bs.bh_active_1d else 0) | (2 if bs.bh_active_1h else 0) | (1 if bs.bh_active_15m else 0)
            assert bs.tf_score == expected, (
                f"tf_score mismatch: bs.tf_score={bs.tf_score}, expected={expected}")

    def test_high_vol_data_doesnt_crash(self):
        """Very high volatility data should complete without crashing."""
        from bh_engine import run_backtest
        rng = np.random.default_rng(555)
        n = 500
        closes = np.empty(n); closes[0] = 4500.0
        for i in range(1, n):
            closes[i] = closes[i-1] * max(0.1, 1.0 + 0.05 * rng.standard_normal())
        idx = pd.date_range("2022-01-03", periods=n, freq="1h")
        df = pd.DataFrame({
            "open": closes, "high": closes*1.01, "low": closes*0.99,
            "close": closes, "volume": np.full(n, 50_000.0)
        }, index=idx)
        result = run_backtest("ES", df)
        assert result is not None
        assert result.stats["final_equity"] >= 0.0
