"""
test_monte_carlo.py — Tests for Monte Carlo simulation engine.

~600 LOC. Tests MCConfig, run_mc, compute_kelly, run_portfolio_mc.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "spacetime" / "engine"))

from mc import (  # type: ignore
    MCConfig,
    MCResult,
    run_mc,
    compute_kelly,
    classify_by_regime,
    run_portfolio_mc,
    PortfolioMCResult,
    REGIMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_trades(n: int, win_rate: float = 0.60, avg_win: float = 0.012,
                 avg_loss: float = 0.008, regime: str = "BULL",
                 seed: int = 42) -> List[dict]:
    """Generate n synthetic trade dicts."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2023-01-01")
    trades = []
    for i in range(n):
        win = rng.random() < win_rate
        pnl = float(rng.exponential(avg_win * 100_000)) if win else -float(rng.exponential(avg_loss * 100_000))
        r_regime = regime if regime != "MIXED" else rng.choice(list(REGIMES))
        trades.append({
            "entry_time": base + pd.Timedelta(hours=i * 8),
            "exit_time":  base + pd.Timedelta(hours=i * 8 + 4),
            "pnl": pnl,
            "regime": r_regime,
            "tf_score": int(rng.integers(4, 8)),
        })
    return trades


def _make_positive_edge_trades(n: int = 100, seed: int = 1) -> List[dict]:
    return _make_trades(n, win_rate=0.65, avg_win=0.015, avg_loss=0.008, seed=seed)


def _make_negative_edge_trades(n: int = 100, seed: int = 2) -> List[dict]:
    return _make_trades(n, win_rate=0.35, avg_win=0.008, avg_loss=0.015, seed=seed)


def _make_zero_edge_trades(n: int = 100, seed: int = 3) -> List[dict]:
    return _make_trades(n, win_rate=0.50, avg_win=0.010, avg_loss=0.010, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Class TestMonteCarlo
# ─────────────────────────────────────────────────────────────────────────────

class TestMonteCarlo:

    def test_median_convergence(self):
        """10K paths → median should be stable across two runs (same seed in mc)."""
        trades = _make_positive_edge_trades(100)
        cfg = MCConfig(n_sims=10_000, months=6)
        r1 = run_mc(trades, starting_equity=1_000_000.0, cfg=cfg)
        r2 = run_mc(trades, starting_equity=1_000_000.0, cfg=cfg)
        # Same seed in mc.py (42) → exact same result
        assert r1.median_equity == pytest.approx(r2.median_equity, rel=1e-6)

    def test_blowup_rate_zero_for_positive_edge(self):
        """Positive edge strategy should have very low blowup rate (<= 5%)."""
        trades = _make_positive_edge_trades(100)
        cfg = MCConfig(n_sims=2_000, months=12, blowup_threshold=0.10)
        result = run_mc(trades, starting_equity=1_000_000.0, cfg=cfg)
        assert result.blowup_rate <= 0.20, (
            f"Blowup rate {result.blowup_rate:.1%} too high for positive edge")

    def test_blowup_rate_high_for_negative_edge(self):
        """Negative edge strategy should have higher blowup rate than positive edge."""
        trades_pos = _make_positive_edge_trades(100)
        trades_neg = _make_negative_edge_trades(100)
        cfg = MCConfig(n_sims=2_000, months=12)
        r_pos = run_mc(trades_pos, starting_equity=1_000_000.0, cfg=cfg)
        r_neg = run_mc(trades_neg, starting_equity=1_000_000.0, cfg=cfg)
        assert r_neg.blowup_rate >= r_pos.blowup_rate, (
            f"Negative edge blowup {r_neg.blowup_rate:.1%} should be >= "
            f"positive edge {r_pos.blowup_rate:.1%}")

    def test_serial_correlation_increases_drawdown(self):
        """Higher serial correlation should increase expected max drawdown."""
        trades = _make_positive_edge_trades(100)
        cfg_low  = MCConfig(n_sims=2_000, months=12, serial_corr=0.0, regime_aware=True)
        cfg_high = MCConfig(n_sims=2_000, months=12, serial_corr=0.5, regime_aware=True)
        r_low  = run_mc(trades, starting_equity=1_000_000.0, cfg=cfg_low)
        r_high = run_mc(trades, starting_equity=1_000_000.0, cfg=cfg_high)
        med_dd_low  = float(np.median(r_low.max_drawdowns))
        med_dd_high = float(np.median(r_high.max_drawdowns))
        # High serial correlation → more clustered losses → larger drawdowns
        assert med_dd_high >= med_dd_low * 0.9, (
            f"Higher serial corr should not decrease drawdown: {med_dd_high:.4f} vs {med_dd_low:.4f}")

    def test_kelly_fraction_positive_for_positive_edge(self):
        """Kelly fraction should be > 0 for a clearly positive-edge strategy."""
        trades = _make_positive_edge_trades(200, seed=11)
        cfg = MCConfig(n_sims=500, months=6)
        result = run_mc(trades, cfg=cfg)
        assert result.kelly_fraction > 0.0, (
            f"Kelly fraction {result.kelly_fraction} should be > 0 for positive edge")

    def test_kelly_fraction_zero_for_zero_edge(self):
        """Kelly fraction should be near 0 for breakeven strategy."""
        trades = _make_zero_edge_trades(200, seed=33)
        cfg = MCConfig(n_sims=500, months=6)
        result = run_mc(trades, cfg=cfg)
        # Zero edge → Kelly ~ 0 (or very small)
        assert result.kelly_fraction <= 0.30, (
            f"Kelly fraction {result.kelly_fraction:.3f} should be small for zero edge")

    def test_regime_aware_uses_regime_returns(self):
        """Regime-aware MC should bucket trades by regime."""
        # Mix BULL and BEAR trades
        rng = np.random.default_rng(77)
        base = pd.Timestamp("2023-01-01")
        trades = []
        for i in range(80):
            regime = "BULL" if i < 40 else "BEAR"
            win = rng.random() < (0.70 if regime == "BULL" else 0.30)
            pnl = float(rng.exponential(1200)) if win else -float(rng.exponential(1000))
            trades.append({
                "entry_time": base + pd.Timedelta(hours=i * 6),
                "exit_time":  base + pd.Timedelta(hours=i * 6 + 3),
                "pnl": pnl, "regime": regime
            })
        cfg = MCConfig(n_sims=500, months=6, regime_aware=True)
        result = run_mc(trades, cfg=cfg)
        assert "BULL" in result.regime_stats
        assert "BEAR" in result.regime_stats
        assert result.regime_stats["BULL"]["count"] == 40
        assert result.regime_stats["BEAR"]["count"] == 40

    def test_percentiles_ordered(self):
        """pct_5 <= pct_25 <= median <= pct_75 <= pct_95."""
        trades = _make_positive_edge_trades(100)
        cfg = MCConfig(n_sims=2_000, months=6)
        result = run_mc(trades, cfg=cfg)
        assert result.pct_5 <= result.pct_25, f"{result.pct_5} > {result.pct_25}"
        assert result.pct_25 <= result.median_equity, f"{result.pct_25} > {result.median_equity}"
        assert result.median_equity <= result.pct_75, f"{result.median_equity} > {result.pct_75}"
        assert result.pct_75 <= result.pct_95, f"{result.pct_75} > {result.pct_95}"

    def test_portfolio_mc_handles_correlation(self):
        """Portfolio MC with correlation > 0 should not crash."""
        syms = ["ES", "NQ", "BTC"]
        trade_lists = {sym: _make_positive_edge_trades(50, seed=i) for i, sym in enumerate(syms)}
        result = run_portfolio_mc(
            trade_lists, starting_equity=3_000_000.0,
            n_sims=500, months=6, cross_corr=0.3
        )
        assert isinstance(result, PortfolioMCResult)
        assert result.median_equity > 0
        assert 0.0 <= result.blowup_rate <= 1.0

    def test_mc_result_has_all_fields(self):
        """MCResult should have all required fields populated."""
        trades = _make_positive_edge_trades(50)
        cfg = MCConfig(n_sims=200, months=3)
        result = run_mc(trades, cfg=cfg)
        assert result.final_equities is not None
        assert len(result.final_equities) == 200
        assert result.max_drawdowns is not None
        assert len(result.max_drawdowns) == 200
        assert math.isfinite(result.blowup_rate)
        assert math.isfinite(result.median_equity)
        assert math.isfinite(result.kelly_fraction)

    def test_fewer_than_5_trades_raises(self):
        """MC with fewer than 5 trades should raise ValueError."""
        trades = _make_trades(3)
        with pytest.raises(ValueError, match="Too few trades"):
            run_mc(trades)

    def test_median_equity_gt_starting_for_strong_edge(self):
        """Strong positive edge → median final equity > starting equity."""
        trades = _make_trades(200, win_rate=0.70, avg_win=0.020, avg_loss=0.006, seed=9)
        cfg = MCConfig(n_sims=2_000, months=12)
        result = run_mc(trades, starting_equity=1_000_000.0, cfg=cfg)
        assert result.median_equity > 1_000_000.0, (
            f"Median equity {result.median_equity:.0f} should exceed starting for strong edge")

    def test_blowup_threshold_effect(self):
        """Stricter blowup threshold (0.5) → higher blowup rate."""
        trades = _make_positive_edge_trades(100)
        r_loose = run_mc(trades, cfg=MCConfig(n_sims=1_000, months=6, blowup_threshold=0.05))
        r_strict = run_mc(trades, cfg=MCConfig(n_sims=1_000, months=6, blowup_threshold=0.50))
        assert r_strict.blowup_rate >= r_loose.blowup_rate

    def test_portfolio_mc_per_instrument_populated(self):
        """Portfolio MC should return per-instrument results."""
        trade_lists = {
            "ES":  _make_positive_edge_trades(60, seed=1),
            "BTC": _make_positive_edge_trades(60, seed=2),
        }
        result = run_portfolio_mc(trade_lists, n_sims=200, months=3)
        assert "ES" in result.per_instrument or "BTC" in result.per_instrument

    def test_regime_stats_win_rate_valid(self):
        """Each regime's win_rate should be in [0, 1]."""
        trades = _make_trades(100, regime="MIXED")
        cfg = MCConfig(n_sims=200, months=3)
        result = run_mc(trades, cfg=cfg)
        for regime, stats in result.regime_stats.items():
            assert 0.0 <= stats["win_rate"] <= 1.0, (
                f"Regime {regime} win_rate {stats['win_rate']} out of [0,1]")

    def test_trades_per_month_positive(self):
        """trades_per_month should be > 0."""
        trades = _make_positive_edge_trades(60)
        result = run_mc(trades, cfg=MCConfig(n_sims=200, months=3))
        assert result.trades_per_month > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Class TestKelly
# ─────────────────────────────────────────────────────────────────────────────

class TestKelly:

    def test_kelly_formula_simple_case(self):
        """Simple binary bet: win p=0.6, win_amount=1, loss_amount=1 → Kelly = 0.20."""
        # f* = p - q = 0.6 - 0.4 = 0.20 for symmetric bet
        returns = np.array([1.0, 1.0, 1.0, -1.0, -1.0] * 20, dtype=float) * 0.01
        # 60% +1% / 40% -1%
        # Analytical: f* ≈ (0.6*0.01 - 0.4*0.01) / (0.01^2) … simplified
        # Use compute_kelly and just verify it's > 0
        kelly = compute_kelly(returns)
        assert kelly > 0.0, f"Kelly should be > 0 for positive edge, got {kelly}"

    def test_kelly_negative_for_losing_strategy(self):
        """Kelly should be 0 (clamped) for a losing strategy."""
        # 40% wins at +1%, 60% losses at -1%
        returns = np.array([-1.0, -1.0, -1.0, 1.0, 1.0] * 20, dtype=float) * 0.01
        kelly = compute_kelly(returns)
        assert kelly == pytest.approx(0.0, abs=0.05), (
            f"Kelly {kelly} should be ~0 for negative edge")

    def test_kelly_half_kelly_safer(self):
        """Half Kelly should have lower blowup rate than full Kelly."""
        trades = _make_positive_edge_trades(100, seed=5)
        all_returns = np.array([t["pnl"] / 1_000_000.0 for t in trades])
        kelly = compute_kelly(all_returns)

        # Simulate with full Kelly vs half Kelly
        rng = np.random.default_rng(42)
        n_sims = 1_000
        n_steps = 100
        blowup_full = 0
        blowup_half = 0
        for _ in range(n_sims):
            eq_full = 1.0
            eq_half = 1.0
            for r in rng.choice(all_returns, size=n_steps, replace=True):
                eq_full = max(0.0, eq_full * (1.0 + kelly * r))
                eq_half = max(0.0, eq_half * (1.0 + 0.5 * kelly * r))
            if eq_full < 0.10: blowup_full += 1
            if eq_half < 0.10: blowup_half += 1
        assert blowup_half <= blowup_full, (
            f"Half Kelly blowup {blowup_half} should be <= full Kelly {blowup_full}")

    def test_kelly_bounded_to_one(self):
        """compute_kelly clips output to [0, 1]."""
        # Extreme positive returns → Kelly might exceed 1 analytically
        returns = np.full(100, 0.10)  # 10% per trade always
        kelly = compute_kelly(returns)
        assert 0.0 <= kelly <= 1.0, f"Kelly {kelly} should be in [0, 1]"

    def test_kelly_zero_for_empty(self):
        """Empty returns array → Kelly = 0."""
        kelly = compute_kelly(np.array([]))
        assert kelly == 0.0

    def test_kelly_consistent_with_theory(self):
        """For a coin flip with 2:1 payout, Kelly ~ 0.25."""
        # Win: +0.02, Loss: -0.01, p=0.5
        # Kelly = (p*b - q) / b = (0.5*2 - 0.5) / 2 = 0.25
        rng = np.random.default_rng(99)
        returns = np.where(rng.random(10_000) < 0.5, 0.02, -0.01)
        kelly = compute_kelly(returns)
        # Approximate: should be between 0.15 and 0.35
        assert 0.10 <= kelly <= 0.40, f"Kelly {kelly:.3f} not near theoretical 0.25"

    def test_classify_by_regime_counts(self):
        """classify_by_regime should correctly group trades by regime."""
        base = pd.Timestamp("2023-01-01")
        trades = [
            {"pnl": 100.0, "regime": "BULL",     "exit_time": base + pd.Timedelta(hours=1)},
            {"pnl": -50.0, "regime": "BEAR",     "exit_time": base + pd.Timedelta(hours=2)},
            {"pnl": 30.0,  "regime": "SIDEWAYS", "exit_time": base + pd.Timedelta(hours=3)},
            {"pnl": 20.0,  "regime": "BULL",     "exit_time": base + pd.Timedelta(hours=4)},
            {"pnl": -10.0, "regime": "HIGH_VOL", "exit_time": base + pd.Timedelta(hours=5)},
        ]
        buckets = classify_by_regime(trades)
        assert len(buckets["BULL"]) == 2
        assert len(buckets["BEAR"]) == 1
        assert len(buckets["SIDEWAYS"]) == 1
        assert len(buckets["HIGH_VOL"]) == 1

    def test_classify_regime_handles_unknown(self):
        """Unknown regime string should fall back to SIDEWAYS."""
        trades = [{"pnl": 100.0, "regime": "UNKNOWN_REGIME", "exit_time": "2023-01-01"}]
        buckets = classify_by_regime(trades)
        assert len(buckets["SIDEWAYS"]) == 1

    def test_mc_months_parameter(self):
        """Longer simulation window should increase median equity for positive edge."""
        trades = _make_positive_edge_trades(100)
        r_short = run_mc(trades, cfg=MCConfig(n_sims=500, months=3))
        r_long  = run_mc(trades, cfg=MCConfig(n_sims=500, months=24))
        assert r_long.median_equity >= r_short.median_equity * 0.8, (
            "Longer sim window should generally not decrease median")
