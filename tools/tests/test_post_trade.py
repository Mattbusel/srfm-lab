"""
test_post_trade.py -- Tests for post-trade analytics and strategy improvement tools.

Covers PostTradeAnalyzer, ExitQualityAnalyzer, RegimePerformanceAnalyzer,
CapacityAnalyzer, LearningCurveAnalyzer, and HypothesisGenerator.
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
import os
import sys
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup -- allow imports from tools/ directory
# ---------------------------------------------------------------------------

TOOLS_DIR = Path(__file__).parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from post_trade_analyzer import (
    PostTradeRecord,
    PostTradeAnalyzer,
    ExitQualityAnalyzer,
    load_trades_from_db,
    save_trades_to_db,
    _sharpe,
    _win_rate,
    _avg,
    _bucket_label,
)
from regime_performance_analyzer import (
    RegimePerformanceAnalyzer,
    BH_THRESHOLDS,
    BH_LABELS,
    HURST_THRESHOLDS,
    HURST_LABELS,
)
from capacity_analyzer import (
    CapacityAnalyzer,
    SymbolCapacityProfile,
    _ac_impact_bps,
)
from learning_curve_analyzer import (
    LearningCurveAnalyzer,
    PerformancePeriod,
)
from hypothesis_generator import (
    HypothesisGenerator,
    GeneratedHypothesis,
    _cohens_d,
    _ttest_pvalue,
    _bootstrap_confidence,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_trade(
    trade_id: str = "T001",
    symbol: str = "BTC",
    pnl_pct: float = 0.5,
    hold_bars: int = 5,
    entry_reason: str = "bh_mass",
    exit_reason: str = "rl_exit",
    bh_mass: float = 0.6,
    hurst: float = 0.6,
    nav_omega: float = 0.8,
    garch_vol: float = 0.015,
    mfe: float = 0.8,
    mae: float = -0.3,
    side: str = "long",
    nav_filtered: bool = False,
    event_filtered: bool = False,
    days_offset: int = 0,
) -> PostTradeRecord:
    base = datetime(2024, 1, 1)
    entry_time = base + timedelta(days=days_offset)
    exit_time = entry_time + timedelta(hours=hold_bars)
    return PostTradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=50000.0,
        exit_price=50000.0 * (1 + pnl_pct / 100.0),
        qty=0.1,
        pnl=pnl_pct * 50.0,
        pnl_pct=pnl_pct,
        hold_bars=hold_bars,
        entry_reason=entry_reason,
        exit_reason=exit_reason,
        bh_mass_at_entry=bh_mass,
        hurst_at_entry=hurst,
        nav_omega_at_entry=nav_omega,
        garch_vol_at_entry=garch_vol,
        mfe=mfe,
        mae=mae,
        max_pnl_during=mfe,
        min_pnl_during=mae,
        side=side,
        nav_filtered=nav_filtered,
        event_filtered=event_filtered,
    )


def _make_trades(
    n: int = 100,
    seed: int = 42,
    win_rate: float = 0.55,
    avg_pnl: float = 0.3,
) -> list[PostTradeRecord]:
    rng = random.Random(seed)
    trades = []
    for i in range(n):
        win = rng.random() < win_rate
        pnl = abs(rng.gauss(avg_pnl, 0.4)) if win else -abs(rng.gauss(avg_pnl * 0.8, 0.3))
        mfe = abs(rng.gauss(0.8, 0.3))
        mae = -abs(rng.gauss(0.3, 0.1))
        holds = rng.randint(1, 20)
        hurst = rng.gauss(0.55, 0.12)
        hurst = max(0.2, min(0.9, hurst))
        bh = rng.gauss(0.55, 0.18)
        bh = max(0.05, min(0.98, bh))
        nav = rng.gauss(0.7, 0.2)
        nav = max(0.1, min(1.0, nav))
        garch = abs(rng.gauss(0.02, 0.008))
        garch = max(0.005, min(0.06, garch))
        symbol = rng.choice(["BTC", "ETH", "SOL", "BNB"])
        entry_reason = rng.choice(["bh_mass", "cf_cross", "hurst_signal"])
        exit_reason = rng.choice(["rl_exit", "time", "signal", "stop"])
        nav_filt = rng.random() < 0.08
        evt_filt = rng.random() < 0.06

        trades.append(_make_trade(
            trade_id=f"T{i:04d}",
            symbol=symbol,
            pnl_pct=round(pnl, 4),
            hold_bars=holds,
            entry_reason=entry_reason,
            exit_reason=exit_reason,
            bh_mass=round(bh, 4),
            hurst=round(hurst, 4),
            nav_omega=round(nav, 4),
            garch_vol=round(garch, 5),
            mfe=round(max(mfe, 0.01), 4),
            mae=round(min(mae, -0.01), 4),
            side="long",
            nav_filtered=nav_filt,
            event_filtered=evt_filt,
            days_offset=i,
        ))
    return trades


# ===========================================================================
# 1. PostTradeRecord tests
# ===========================================================================

class TestPostTradeRecord:

    def test_win_property_true(self):
        t = _make_trade(pnl_pct=0.5)
        assert t.win is True

    def test_win_property_false(self):
        t = _make_trade(pnl_pct=-0.3)
        assert t.win is False

    def test_exit_efficiency_at_peak(self):
        t = _make_trade(pnl_pct=0.8, mfe=0.8)
        assert abs(t.exit_efficiency - 1.0) < 1e-9

    def test_exit_efficiency_partial(self):
        t = _make_trade(pnl_pct=0.4, mfe=0.8)
        assert abs(t.exit_efficiency - 0.5) < 1e-6

    def test_exit_efficiency_zero_mfe(self):
        t = _make_trade(pnl_pct=0.4, mfe=0.0)
        assert t.exit_efficiency == 0.0

    def test_exit_efficiency_bounds(self):
        """Exit efficiency must be in [-1, 1]."""
        t = _make_trade(pnl_pct=2.0, mfe=0.5)
        assert t.exit_efficiency <= 1.0

        t2 = _make_trade(pnl_pct=-1.5, mfe=0.5)
        assert t2.exit_efficiency >= -1.0

    def test_mfe_mae_ratio_positive(self):
        t = _make_trade(mfe=1.2, mae=-0.4)
        assert abs(t.mfe_mae_ratio - 3.0) < 1e-6

    def test_mfe_mae_ratio_zero_mae(self):
        t = _make_trade(mfe=1.0, mae=0.0)
        assert math.isinf(t.mfe_mae_ratio)

    def test_pnl_pct_field(self):
        t = _make_trade(pnl_pct=-1.23)
        assert t.pnl_pct == pytest.approx(-1.23)

    def test_all_fields_accessible(self):
        t = _make_trade()
        for attr in [
            "trade_id", "symbol", "entry_time", "exit_time",
            "entry_price", "exit_price", "qty", "pnl", "pnl_pct",
            "hold_bars", "entry_reason", "exit_reason",
            "bh_mass_at_entry", "hurst_at_entry", "nav_omega_at_entry",
            "garch_vol_at_entry", "mfe", "mae", "side",
        ]:
            assert hasattr(t, attr)


# ===========================================================================
# 2. MFE / MAE analysis
# ===========================================================================

class TestMFEMAECalculation:

    def _make_mfe_mae_trades(self) -> list[PostTradeRecord]:
        return [
            _make_trade("T1", pnl_pct=0.8, mfe=1.0, mae=-0.2),
            _make_trade("T2", pnl_pct=0.3, mfe=0.9, mae=-0.5),
            _make_trade("T3", pnl_pct=-0.1, mfe=0.4, mae=-0.6),
            _make_trade("T4", pnl_pct=0.5, mfe=0.5, mae=-0.1),
            _make_trade("T5", pnl_pct=0.9, mfe=1.0, mae=-0.05),
        ]

    def test_mfe_mae_calculation_returns_dict(self):
        trades = self._make_mfe_mae_trades()
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_mfe_mae()
        assert isinstance(result, dict)
        assert "mfe" in result
        assert "mae" in result

    def test_mfe_mae_mean_positive(self):
        trades = self._make_mfe_mae_trades()
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_mfe_mae()
        assert result["mfe"]["mean"] > 0
        assert result["mae"]["mean"] > 0  # abs(MAE)

    def test_mfe_mae_ratio_computed(self):
        trades = self._make_mfe_mae_trades()
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_mfe_mae()
        assert "mfe_mae_ratio" in result
        assert result["mfe_mae_ratio"]["mean"] > 0

    def test_exit_efficiency_in_mfe_result(self):
        trades = self._make_mfe_mae_trades()
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_mfe_mae()
        assert "exit_efficiency" in result
        eff = result["exit_efficiency"]["mean"]
        assert 0.0 <= eff <= 1.0

    def test_poor_exits_counted(self):
        # T3 is a full reversal
        trades = self._make_mfe_mae_trades()
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_mfe_mae()
        assert result["poor_exits"]["full_reversals_n"] >= 1

    def test_mfe_percentiles_ordered(self):
        trades = _make_trades(n=50)
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_mfe_mae()
        assert result["mfe"]["median"] <= result["mfe"]["p75"] <= result["mfe"]["p95"]

    def test_total_trades_matches(self):
        trades = self._make_mfe_mae_trades()
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_mfe_mae()
        assert result["total_trades"] == 5


# ===========================================================================
# 3. Exit quality efficiency bounds
# ===========================================================================

class TestExitQualityEfficiencyBounds:

    def test_efficiency_always_le_one(self):
        """Exit efficiency must never exceed 1.0."""
        trades = _make_trades(n=200, seed=0)
        eqa = ExitQualityAnalyzer(trades)
        df = eqa.compute_exit_efficiencies()
        assert (df["exit_efficiency"] <= 1.0 + 1e-9).all(), \
            f"Max efficiency: {df['exit_efficiency'].max()}"

    def test_efficiency_always_ge_minus_one(self):
        """Exit efficiency floor is -1.0 (lost more than MFE, rare edge case)."""
        trades = _make_trades(n=200, seed=1)
        eqa = ExitQualityAnalyzer(trades)
        df = eqa.compute_exit_efficiencies()
        assert (df["exit_efficiency"] >= -1.0 - 1e-9).all()

    def test_efficiency_one_when_exited_at_mfe(self):
        t = _make_trade(pnl_pct=0.8, mfe=0.8)
        eqa = ExitQualityAnalyzer([t])
        df = eqa.compute_exit_efficiencies()
        assert abs(df["exit_efficiency"].iloc[0] - 1.0) < 1e-9

    def test_compare_rl_vs_simple_returns_dict(self):
        trades = _make_trades(n=100, seed=5)
        eqa = ExitQualityAnalyzer(trades)
        result = eqa.compare_rl_vs_simple_exits(simple_hold_bars=5)
        assert isinstance(result, dict)
        assert "rl_exit" in result

    def test_compare_rl_vs_simple_has_advantage_key(self):
        trades = _make_trades(n=100, seed=6)
        eqa = ExitQualityAnalyzer(trades)
        result = eqa.compare_rl_vs_simple_exits(simple_hold_bars=5)
        assert "rl_advantage_pnl_pct" in result

    def test_optimal_exit_bars_returns_dict(self):
        trades = _make_trades(n=100, seed=7)
        eqa = ExitQualityAnalyzer(trades)
        result = eqa.optimal_exit_bars()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_optimal_bars_are_positive_integers(self):
        trades = _make_trades(n=200, seed=8)
        eqa = ExitQualityAnalyzer(trades)
        result = eqa.optimal_exit_bars()
        for lb, info in result.items():
            if info.get("optimal_bars") is not None:
                assert isinstance(info["optimal_bars"], int)
                assert info["optimal_bars"] >= 1

    def test_exit_efficiency_df_has_expected_columns(self):
        trades = _make_trades(n=50, seed=9)
        eqa = ExitQualityAnalyzer(trades)
        df = eqa.compute_exit_efficiencies()
        for col in ["trade_id", "exit_reason", "exit_efficiency", "mfe", "mae"]:
            assert col in df.columns

    def test_exit_reason_efficiency_breakdown_returns_df(self):
        trades = _make_trades(n=100, seed=10)
        eqa = ExitQualityAnalyzer(trades)
        df = eqa.exit_reason_efficiency_breakdown()
        assert not df.empty


# ===========================================================================
# 4. Regime performance by Hurst
# ===========================================================================

class TestRegimePerformanceByHurst:

    def _make_hurst_stratified_trades(self) -> list[PostTradeRecord]:
        """Trending trades have systematically higher P&L."""
        trades = []
        rng = random.Random(42)
        for i in range(120):
            hurst = rng.gauss(0.65, 0.05) if i < 40 else \
                    rng.gauss(0.47, 0.05) if i < 80 else \
                    rng.gauss(0.33, 0.05)
            pnl = 0.6 if hurst > 0.55 else (-0.1 if hurst < 0.4 else 0.25)
            pnl += rng.gauss(0, 0.2)
            hurst = max(0.1, min(0.95, hurst))
            trades.append(_make_trade(
                trade_id=f"T{i:04d}",
                pnl_pct=round(pnl, 4),
                hurst=round(hurst, 4),
                days_offset=i,
            ))
        return trades

    def test_performance_by_hurst_regime_returns_df(self):
        trades = self._make_hurst_stratified_trades()
        rpa = RegimePerformanceAnalyzer(trades=trades)
        df = rpa.performance_by_hurst_regime()
        assert not df.empty

    def test_hurst_regime_has_all_labels(self):
        trades = self._make_hurst_stratified_trades()
        rpa = RegimePerformanceAnalyzer(trades=trades)
        df = rpa.performance_by_hurst_regime()
        for label in HURST_LABELS:
            assert label in df.index

    def test_trending_outperforms_mean_reverting(self):
        trades = self._make_hurst_stratified_trades()
        rpa = RegimePerformanceAnalyzer(trades=trades)
        df = rpa.performance_by_hurst_regime()
        trending_pnl = df.loc["trending", "avg_pnl_pct"]
        mr_pnl = df.loc["mean_reverting", "avg_pnl_pct"]
        assert trending_pnl > mr_pnl, \
            f"Trending {trending_pnl:.4f} should > MR {mr_pnl:.4f}"

    def test_n_column_sums_to_total(self):
        trades = self._make_hurst_stratified_trades()
        rpa = RegimePerformanceAnalyzer(trades=trades)
        df = rpa.performance_by_hurst_regime()
        assert df["n"].sum() == len(trades)

    def test_win_rate_in_range(self):
        trades = self._make_hurst_stratified_trades()
        rpa = RegimePerformanceAnalyzer(trades=trades)
        df = rpa.performance_by_hurst_regime()
        assert (df["win_rate"] >= 0.0).all()
        assert (df["win_rate"] <= 1.0).all()

    def test_joint_regime_analysis_returns_2d_df(self):
        trades = _make_trades(n=200, seed=11)
        rpa = RegimePerformanceAnalyzer(trades=trades)
        df = rpa.joint_regime_analysis("bh_mass", "hurst")
        assert df.shape[0] == len(BH_LABELS)
        assert df.shape[1] == len(HURST_LABELS)

    def test_best_regime_combination_returns_dict(self):
        trades = _make_trades(n=200, seed=12)
        rpa = RegimePerformanceAnalyzer(trades=trades)
        best = rpa.best_regime_combination(min_n=5)
        if best:  # may be empty if no combo has enough trades
            assert "bh_mass_level" in best
            assert "hurst_regime" in best
            assert "sharpe" in best

    def test_worst_regime_has_lower_sharpe_than_best(self):
        trades = _make_trades(n=300, seed=13)
        rpa = RegimePerformanceAnalyzer(trades=trades)
        best = rpa.best_regime_combination(min_n=5)
        worst = rpa.worst_regime_combination(min_n=5)
        if best and worst:
            assert best["sharpe"] >= worst["sharpe"]

    def test_full_regime_summary_sorted_by_sharpe(self):
        trades = _make_trades(n=300, seed=14)
        rpa = RegimePerformanceAnalyzer(trades=trades)
        df = rpa.full_regime_summary()
        if len(df) > 1:
            sharpes = df["sharpe"].values
            assert (sharpes[:-1] >= sharpes[1:]).all(), "Should be sorted descending"


# ===========================================================================
# 5. Capacity analyzer
# ===========================================================================

class TestCapacityAtSharpePositive:

    def test_capacity_at_sharpe_is_positive(self):
        ca = CapacityAnalyzer(strategy_sharpe=1.0, avg_position_usd=100_000)
        cap = ca.compute_capacity_at_sharpe(target_sharpe=0.5, symbol="BTC")
        assert cap > 0, f"Capacity should be positive, got {cap}"

    def test_capacity_decreases_with_higher_target_sharpe(self):
        ca = CapacityAnalyzer(strategy_sharpe=1.5)
        cap_05 = ca.compute_capacity_at_sharpe(0.5, symbol="BTC")
        cap_10 = ca.compute_capacity_at_sharpe(1.0, symbol="BTC")
        assert cap_05 > cap_10, "Higher target Sharpe = lower capacity"

    def test_capacity_zero_when_target_exceeds_base(self):
        ca = CapacityAnalyzer(strategy_sharpe=0.8)
        cap = ca.compute_capacity_at_sharpe(1.0, symbol="BTC")
        assert cap == 0.0

    def test_position_size_sensitivity_returns_dict(self):
        ca = CapacityAnalyzer(strategy_sharpe=1.0)
        sizes = [1e4, 1e5, 1e6]
        result = ca.position_size_sensitivity("BTC", sizes)
        assert len(result) == 3
        for pos in sizes:
            assert pos in result

    def test_impact_increases_with_position_size(self):
        ca = CapacityAnalyzer(strategy_sharpe=1.0)
        sizes = [1e4, 1e5, 1e6, 1e7]
        result = ca.position_size_sensitivity("BTC", sizes)
        impacts = [result[s].total_impact_bps for s in sizes]
        assert impacts[0] < impacts[-1], "Impact should increase with position size"

    def test_turnover_cost_positive(self):
        ca = CapacityAnalyzer(strategy_sharpe=1.0, avg_position_usd=100_000)
        cost = ca.turnover_cost_estimate(0.02, aum=1_000_000)
        assert cost > 0

    def test_turnover_cost_scales_with_aum(self):
        ca = CapacityAnalyzer(strategy_sharpe=1.0)
        cost_1m = ca.turnover_cost_estimate(0.02, aum=1_000_000)
        cost_10m = ca.turnover_cost_estimate(0.02, aum=10_000_000)
        assert cost_10m > cost_1m

    def test_ac_impact_zero_for_zero_position(self):
        perm, temp, total = _ac_impact_bps(0.0, 1e9, 10.0, 15.0)
        assert perm == 0.0 and temp == 0.0 and total == 0.0

    def test_ac_impact_sqrt_relationship(self):
        """Doubling position should less than double impact (sqrt scaling)."""
        _, _, impact_1 = _ac_impact_bps(1e5, 1e9, 10.0, 15.0)
        _, _, impact_4 = _ac_impact_bps(4e5, 1e9, 10.0, 15.0)
        ratio = impact_4 / impact_1
        assert 1.0 < ratio < 4.0, f"sqrt impact ratio should be ~2, got {ratio:.2f}"

    def test_optimal_hold_time_returns_dict(self):
        profile = SymbolCapacityProfile(
            symbol="BTC", daily_volume_usd=2e10, base_sharpe=1.0,
            signal_decay_halflife_bars=8.0,
        )
        ca = CapacityAnalyzer(
            profiles=[profile], strategy_sharpe=1.0, avg_position_usd=100_000
        )
        result = ca.optimal_hold_time_by_symbol()
        assert "BTC" in result
        assert result["BTC"]["optimal_bars_discrete"] >= 1

    def test_scaling_scenario_table_rows_match_aum(self):
        ca = CapacityAnalyzer(strategy_sharpe=1.0)
        aum_vals = [1e5, 1e6, 1e7]
        df = ca.scaling_scenario_table(aum_values=aum_vals, symbol="BTC")
        assert len(df) == len(aum_vals)


# ===========================================================================
# 6. Hypothesis generator
# ===========================================================================

class TestHypothesisGeneratorCreatesHypotheses:

    def test_generate_returns_list(self):
        trades = _make_trades(n=150, seed=20)
        hg = HypothesisGenerator(trades=trades, min_group_size=5)
        hyps = hg.generate()
        assert isinstance(hyps, list)

    def test_generate_creates_at_least_one_hypothesis(self):
        trades = _make_trades(n=200, seed=21)
        hg = HypothesisGenerator(trades=trades, min_group_size=5)
        hyps = hg.generate()
        assert len(hyps) >= 1, "Should generate at least one hypothesis"

    def test_hypotheses_have_required_fields(self):
        trades = _make_trades(n=200, seed=22)
        hg = HypothesisGenerator(trades=trades, min_group_size=5)
        hyps = hg.generate()
        for h in hyps:
            assert h.hypothesis_id
            assert h.condition
            assert h.predicted_improvement
            assert 0.0 <= h.confidence <= 1.0
            assert h.supporting_evidence_n > 0
            assert h.testable_signal_code

    def test_hypothesis_ids_are_unique(self):
        trades = _make_trades(n=200, seed=23)
        hg = HypothesisGenerator(trades=trades, min_group_size=5)
        hyps = hg.generate()
        ids = [h.hypothesis_id for h in hyps]
        assert len(ids) == len(set(ids)), "Duplicate hypothesis IDs found"

    def test_hypotheses_sorted_by_confidence(self):
        trades = _make_trades(n=200, seed=24)
        hg = HypothesisGenerator(trades=trades, min_group_size=5)
        hyps = hg.generate()
        if len(hyps) > 1:
            confs = [h.confidence for h in hyps]
            assert confs == sorted(confs, reverse=True)

    def test_to_dataframe_works(self):
        trades = _make_trades(n=150, seed=25)
        hg = HypothesisGenerator(trades=trades, min_group_size=5)
        hyps = hg.generate()
        df = hg.to_dataframe(hyps)
        if hyps:
            assert not df.empty
            assert "hypothesis_id" in df.columns

    def test_to_json_is_valid(self):
        import json
        trades = _make_trades(n=150, seed=26)
        hg = HypothesisGenerator(trades=trades, min_group_size=5)
        hyps = hg.generate()
        json_str = hg.to_json(hyps)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_hypothesis_type_is_valid(self):
        valid_types = {"filter", "exit_timing", "position_sizing", "entry_timing"}
        trades = _make_trades(n=200, seed=27)
        hg = HypothesisGenerator(trades=trades, min_group_size=5)
        hyps = hg.generate()
        for h in hyps:
            assert h.hypothesis_type in valid_types

    def test_p_value_in_range(self):
        trades = _make_trades(n=200, seed=28)
        hg = HypothesisGenerator(trades=trades, min_group_size=5)
        hyps = hg.generate()
        for h in hyps:
            assert 0.0 <= h.p_value <= 1.0

    def test_generate_with_override_trades(self):
        trades_a = _make_trades(n=100, seed=29)
        trades_b = _make_trades(n=150, seed=30)
        hg = HypothesisGenerator(trades=trades_a, min_group_size=5)
        hyps_a = hg.generate()
        hyps_b = hg.generate(trades=trades_b)
        # After override, instance trades should be restored
        assert hg.trades is trades_a


# ===========================================================================
# 7. PostTradeAnalyzer
# ===========================================================================

class TestPostTradeAnalyzer:

    def test_analyze_entry_quality_returns_dict(self):
        trades = _make_trades(n=100, seed=40)
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_entry_quality()
        assert isinstance(result, dict)
        assert "bh_mass_level" in result
        assert "hurst_regime" in result
        assert "entry_reason" in result

    def test_entry_quality_win_rates_in_range(self):
        trades = _make_trades(n=100, seed=41)
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_entry_quality()
        for level_data in result["bh_mass_level"].values():
            wr = level_data["win_rate"]
            assert 0.0 <= wr <= 1.0

    def test_analyze_exit_timing_returns_dict(self):
        trades = _make_trades(n=100, seed=42)
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_exit_timing()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_exit_timing_has_avg_hold_bars(self):
        trades = _make_trades(n=100, seed=43)
        analyzer = PostTradeAnalyzer(trades=trades)
        result = analyzer.analyze_exit_timing()
        for stats in result.values():
            assert "avg_hold_bars" in stats
            assert stats["avg_hold_bars"] > 0

    def test_cohort_analysis_returns_df(self):
        trades = _make_trades(n=100, seed=44)
        analyzer = PostTradeAnalyzer(trades=trades)
        df = analyzer.cohort_analysis()
        assert not df.empty
        assert "cohort_week" in df.columns
        assert "sharpe" in df.columns

    def test_find_missed_opportunities_returns_list(self):
        # Add some filtered trades with high BH mass
        trades = _make_trades(n=100, seed=45)
        for i in range(5):
            t = _make_trade(
                trade_id=f"FILT{i}",
                bh_mass=0.75,
                nav_filtered=True,
                pnl_pct=0.4,
                days_offset=i + 200,
            )
            trades.append(t)
        analyzer = PostTradeAnalyzer(trades=trades)
        missed = analyzer.find_missed_opportunities(bh_mass_threshold=0.7)
        assert isinstance(missed, list)
        assert len(missed) >= 5

    def test_summary_report_returns_dict(self):
        trades = _make_trades(n=80, seed=46)
        analyzer = PostTradeAnalyzer(trades=trades)
        report = analyzer.summary_report()
        assert "total_trades" in report
        assert report["total_trades"] == 80

    def test_empty_analyzer_no_crash(self):
        analyzer = PostTradeAnalyzer(trades=[])
        assert analyzer.analyze_mfe_mae() == {}
        assert analyzer.analyze_entry_quality()
        assert analyzer.analyze_exit_timing() == {}
        assert analyzer.summary_report()["error"]


# ===========================================================================
# 8. LearningCurveAnalyzer
# ===========================================================================

class TestLearningCurveAnalyzer:

    def test_rolling_sharpe_returns_df(self):
        trades = _make_trades(n=100, seed=50)
        lca = LearningCurveAnalyzer(trades=trades, window_trades=20)
        df = lca.rolling_sharpe_series(window=20)
        assert not df.empty
        assert "rolling_sharpe" in df.columns

    def test_rolling_sharpe_row_count(self):
        trades = _make_trades(n=80, seed=51)
        lca = LearningCurveAnalyzer(trades=trades, window_trades=20)
        df = lca.rolling_sharpe_series(window=20)
        assert len(df) == 80 - 20 + 1

    def test_estimate_learning_rate_returns_float(self):
        trades = _make_trades(n=100, seed=52)
        lca = LearningCurveAnalyzer(trades=trades, window_trades=20)
        rate = lca.estimate_learning_rate()
        assert isinstance(rate, float)
        assert not math.isnan(rate)

    def test_detect_regimes_returns_list(self):
        trades = _make_trades(n=150, seed=53)
        lca = LearningCurveAnalyzer(trades=trades, window_trades=20, min_regime_trades=10)
        regimes = lca.detect_performance_regimes()
        assert isinstance(regimes, list)

    def test_regime_labels_are_valid(self):
        trades = _make_trades(n=150, seed=54)
        lca = LearningCurveAnalyzer(trades=trades, window_trades=20, min_regime_trades=5)
        regimes = lca.detect_performance_regimes()
        for r in regimes:
            assert r.label in ("strong", "weak", "neutral")

    def test_rl_exit_improvement_returns_df(self):
        trades = _make_trades(n=100, seed=55)
        # Ensure some rl_exit trades exist
        for t in trades:
            if t.exit_reason != "rl_exit":
                pass  # mix is fine
        lca = LearningCurveAnalyzer(trades=trades, window_trades=15)
        df = lca.rl_exit_improvement(window=15)
        assert not df.empty

    def test_cumulative_pnl_curve_is_monotonic_trade_index(self):
        trades = _make_trades(n=50, seed=56)
        lca = LearningCurveAnalyzer(trades=trades)
        df = lca.cumulative_pnl_curve()
        indices = df["trade_index"].values
        assert (indices[1:] > indices[:-1]).all()

    def test_learning_summary_keys(self):
        trades = _make_trades(n=100, seed=57)
        lca = LearningCurveAnalyzer(trades=trades, window_trades=20)
        summary = lca.learning_summary()
        for key in [
            "total_trades", "learning_rate_sharpe_per_100_trades",
            "learning_direction", "strong_periods", "weak_periods",
        ]:
            assert key in summary

    def test_ml_convergence_proxy_returns_df(self):
        trades = _make_trades(n=200, seed=58)
        lca = LearningCurveAnalyzer(trades=trades)
        df = lca.ml_convergence_proxy(feature_attr="bh_mass_at_entry", bin_size=30)
        assert not df.empty
        assert "correlation" in df.columns


# ===========================================================================
# 9. SQLite persistence
# ===========================================================================

class TestSQLitePersistence:

    def test_save_and_load_roundtrip(self):
        trades = _make_trades(n=20, seed=60)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_trades_to_db(trades, db_path)
            loaded = load_trades_from_db(db_path)
            assert len(loaded) == len(trades)
        finally:
            os.unlink(db_path)

    def test_loaded_pnl_matches_saved(self):
        trades = _make_trades(n=10, seed=61)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_trades_to_db(trades, db_path)
            loaded = load_trades_from_db(db_path)
            orig_pnls = sorted(t.pnl_pct for t in trades)
            load_pnls = sorted(t.pnl_pct for t in loaded)
            assert orig_pnls == pytest.approx(load_pnls, abs=1e-6)
        finally:
            os.unlink(db_path)

    def test_upsert_does_not_duplicate(self):
        trades = _make_trades(n=10, seed=62)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            save_trades_to_db(trades, db_path)
            save_trades_to_db(trades, db_path)  # save again
            loaded = load_trades_from_db(db_path)
            assert len(loaded) == len(trades)
        finally:
            os.unlink(db_path)

    def test_load_from_empty_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            loaded = load_trades_from_db(db_path)
            assert loaded == []
        finally:
            os.unlink(db_path)


# ===========================================================================
# 10. Statistical utility tests
# ===========================================================================

class TestStatisticalUtilities:

    def test_sharpe_zero_for_constant_returns(self):
        pnls = [0.5] * 20
        assert _sharpe(pnls) == 0.0

    def test_sharpe_positive_for_positive_drift(self):
        rng = np.random.default_rng(0)
        pnls = list(rng.normal(0.1, 0.05, 100))
        assert _sharpe(pnls) > 0

    def test_win_rate_all_wins(self):
        assert _win_rate([1.0, 0.5, 0.1]) == pytest.approx(1.0)

    def test_win_rate_no_wins(self):
        assert _win_rate([-1.0, -0.5]) == pytest.approx(0.0)

    def test_avg_empty_list(self):
        assert _avg([]) == 0.0

    def test_cohens_d_zero_for_identical_groups(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 2.0, 3.0, 4.0]
        assert abs(_cohens_d(a, b)) < 1e-9

    def test_cohens_d_positive_when_a_gt_b(self):
        rng = np.random.default_rng(99)
        a = list(rng.normal(2.0, 0.3, 20))
        b = list(rng.normal(1.0, 0.3, 20))
        assert _cohens_d(a, b) > 0

    def test_ttest_pvalue_significant_for_large_diff(self):
        rng = np.random.default_rng(42)
        a = list(rng.normal(2.0, 0.1, 50))
        b = list(rng.normal(0.0, 0.1, 50))
        p = _ttest_pvalue(a, b)
        assert p < 0.01

    def test_ttest_pvalue_not_significant_for_same_dist(self):
        rng = np.random.default_rng(42)
        a = list(rng.normal(1.0, 1.0, 30))
        b = list(rng.normal(1.0, 1.0, 30))
        p = _ttest_pvalue(a, b)
        assert p > 0.05  # not guaranteed but very likely

    def test_bootstrap_confidence_above_half_when_a_better(self):
        rng = np.random.default_rng(0)
        a = list(rng.normal(1.0, 0.2, 50))
        b = list(rng.normal(0.0, 0.2, 50))
        conf = _bootstrap_confidence(a, b, n_boot=200)
        assert conf > 0.9

    def test_bucket_label_boundaries(self):
        assert _bucket_label(0.1, [0.2, 0.4, 0.6], ["low", "mid", "high"]) == "low"
        assert _bucket_label(0.3, [0.2, 0.4, 0.6], ["low", "mid", "high"]) == "mid"
        assert _bucket_label(0.8, [0.2, 0.4, 0.6], ["low", "mid", "high"]) == "high"


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
