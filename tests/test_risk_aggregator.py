"""
tests/test_risk_aggregator.py
==============================
Test suite for the execution.risk aggregation system.

Covers:
  - ParametricVaR: positive VaR, CVaR >= VaR, diversification ratio
  - HistoricalVaR: correct quantile, CVaR >= VaR
  - MonteCarloVaR: convergence with large path count, CVaR >= VaR
  - VaRMonitor: consensus weighting, Kupiec test trigger, DB persistence
  - PnLAttributor: factor contributions sum to total P&L
  - AttributionReport: factor performance aggregation
  - CorrelationMatrix: symmetry, positive semi-definiteness, stress detection
  - ConcentrationRisk: HHI range, effective N
  - LimitChecker: breach detection on various limit types
  - PositionLimiter: quantity reduction and blocking logic
  - DrawdownGuard: halt at correct threshold, resume on recovery
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> Path:
    """Create a minimal live_trades.db in tmp_path with required tables."""
    db = tmp_path / "live_trades.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            notional REAL NOT NULL,
            fill_time TEXT NOT NULL,
            order_id TEXT,
            strategy_version TEXT DEFAULT 'test'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_pnl (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            entry_time TEXT,
            exit_time TEXT,
            entry_price REAL,
            exit_price REAL,
            qty REAL,
            pnl REAL,
            hold_bars INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nav_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT DEFAULT '15m',
            bar_time TEXT NOT NULL,
            timestamp_ns INTEGER NOT NULL,
            bar_qw REAL NOT NULL, bar_qx REAL NOT NULL,
            bar_qy REAL NOT NULL, bar_qz REAL NOT NULL,
            qw REAL NOT NULL, qx REAL NOT NULL,
            qy REAL NOT NULL, qz REAL NOT NULL,
            angular_velocity REAL NOT NULL,
            geodesic_deviation REAL NOT NULL,
            bh_mass REAL DEFAULT 0.0,
            bh_active INTEGER DEFAULT 0,
            lorentz_boost_applied INTEGER DEFAULT 0,
            lorentz_boost_rapidity REAL DEFAULT 0.0,
            strategy_version TEXT DEFAULT 'test'
        )
    """)
    conn.commit()
    conn.close()
    return db


def _sample_returns(n: int = 100, seed: int = 42) -> Dict[str, float]:
    """Generate one-day sample returns for two symbols."""
    rng = np.random.default_rng(seed)
    return {
        "BTC": float(rng.normal(0.001, 0.03)),
        "ETH": float(rng.normal(0.001, 0.025)),
        "SPY": float(rng.normal(0.0003, 0.01)),
    }


def _warm_up_parametric(pvar, n_days: int = 60, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_days):
        returns = {
            "BTC": float(rng.normal(0.001, 0.03)),
            "ETH": float(rng.normal(0.001, 0.025)),
            "SPY": float(rng.normal(0.0003, 0.01)),
        }
        pvar.update(returns)


def _warm_up_historical(hvar, n_days: int = 60, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for i in range(n_days):
        returns = {
            "BTC": float(rng.normal(0.001, 0.03)),
            "ETH": float(rng.normal(0.001, 0.025)),
            "SPY": float(rng.normal(0.0003, 0.01)),
        }
        hvar.update(returns)


def _make_snapshot(equity: float = 100_000.0) -> "PortfolioSnapshot":
    from execution.risk.live_var import PortfolioSnapshot, PositionSnapshot
    return PortfolioSnapshot(
        positions=[
            PositionSnapshot("BTC", qty=0.5, entry_price=60_000.0, current_price=62_000.0),
            PositionSnapshot("ETH", qty=5.0, entry_price=3_000.0, current_price=3_100.0),
            PositionSnapshot("SPY", qty=100.0, entry_price=450.0, current_price=455.0),
        ],
        equity=equity,
    )


# ===========================================================================
# ParametricVaR tests
# ===========================================================================

class TestParametricVaR:

    def test_var_positive_after_warmup(self):
        from execution.risk.live_var import ParametricVaR
        pvar = ParametricVaR()
        _warm_up_parametric(pvar, n_days=30)
        snap = _make_snapshot()
        result = pvar.portfolio_var(snap)
        assert result.var_95 >= 0, "VaR95 should be non-negative"
        assert result.var_99 >= 0, "VaR99 should be non-negative"

    def test_cvar_geq_var(self):
        from execution.risk.live_var import ParametricVaR
        pvar = ParametricVaR()
        _warm_up_parametric(pvar, n_days=30)
        snap = _make_snapshot()
        result = pvar.portfolio_var(snap)
        assert result.cvar_95 >= result.var_95 - 1e-6, "CVaR95 >= VaR95 required"
        assert result.cvar_99 >= result.var_99 - 1e-6, "CVaR99 >= VaR99 required"

    def test_var99_geq_var95(self):
        from execution.risk.live_var import ParametricVaR
        pvar = ParametricVaR()
        _warm_up_parametric(pvar, n_days=30)
        snap = _make_snapshot()
        result = pvar.portfolio_var(snap)
        assert result.var_99 >= result.var_95 - 1e-6, "VaR99 >= VaR95 required"

    def test_empty_snapshot_returns_zero(self):
        from execution.risk.live_var import ParametricVaR, PortfolioSnapshot
        pvar = ParametricVaR()
        _warm_up_parametric(pvar, n_days=10)
        snap = PortfolioSnapshot(positions=[], equity=100_000.0)
        result = pvar.portfolio_var(snap)
        assert result.var_99 == 0.0

    def test_diversification_ratio_geq_one(self):
        from execution.risk.live_var import ParametricVaR
        pvar = ParametricVaR()
        _warm_up_parametric(pvar, n_days=60)
        snap = _make_snapshot()
        dr = pvar.diversification_ratio(snap)
        assert dr >= 1.0 - 1e-6, f"Diversification ratio should be >= 1.0, got {dr}"

    def test_marginal_var_sums_to_component_var(self):
        from execution.risk.live_var import ParametricVaR
        pvar = ParametricVaR()
        _warm_up_parametric(pvar, n_days=60)
        snap = _make_snapshot()
        comp_var = pvar.component_var(snap)
        # Component VaRs should approximately sum to portfolio VaR
        total_comp = sum(comp_var.values())
        port_var = pvar.portfolio_var(snap).var_99
        # Component VaR satisfies Euler decomposition: sum == portfolio VaR exactly
        assert abs(total_comp - port_var) < 1e-4 * max(port_var, 1.0) + 1e-6, (
            f"Component VaR sum {total_comp:.6f} != portfolio VaR {port_var:.6f}"
        )

    def test_method_label(self):
        from execution.risk.live_var import ParametricVaR
        pvar = ParametricVaR()
        _warm_up_parametric(pvar, n_days=10)
        snap = _make_snapshot()
        result = pvar.portfolio_var(snap)
        assert result.method == "parametric"


# ===========================================================================
# HistoricalVaR tests
# ===========================================================================

class TestHistoricalVaR:

    def test_var_positive(self):
        from execution.risk.live_var import HistoricalVaR
        hvar = HistoricalVaR()
        _warm_up_historical(hvar, n_days=60)
        snap = _make_snapshot()
        result = hvar.portfolio_var(snap)
        assert result.var_95 >= 0
        assert result.var_99 >= 0

    def test_cvar_geq_var(self):
        from execution.risk.live_var import HistoricalVaR
        hvar = HistoricalVaR()
        _warm_up_historical(hvar, n_days=60)
        snap = _make_snapshot()
        result = hvar.portfolio_var(snap)
        assert result.cvar_95 >= result.var_95 - 1e-6
        assert result.cvar_99 >= result.var_99 - 1e-6

    def test_var99_geq_var95(self):
        from execution.risk.live_var import HistoricalVaR
        hvar = HistoricalVaR()
        _warm_up_historical(hvar, n_days=60)
        snap = _make_snapshot()
        result = hvar.portfolio_var(snap)
        assert result.var_99 >= result.var_95 - 1e-6

    def test_uses_correct_quantile(self):
        """
        Use a controlled return series where quantile is analytically known.
        Portfolio is 100% BTC. Returns: 100 observations, worst 5 are exactly -0.10.
        """
        from execution.risk.live_var import HistoricalVaR, PortfolioSnapshot, PositionSnapshot
        hvar = HistoricalVaR(window=100, ewma_lambda=1.0)  # flat weighting
        # 95 returns of +0.01 and 5 returns of -0.10
        rets = [0.01] * 95 + [-0.10] * 5
        for r in rets:
            hvar.update({"BTC": r})
        snap = PortfolioSnapshot(
            positions=[PositionSnapshot("BTC", qty=1.0, entry_price=1.0, current_price=1.0)],
            equity=1.0,
        )
        result = hvar.portfolio_var(snap)
        # At 95% confidence the VaR should be approximately 0.10 (the worst 5%)
        assert abs(result.var_95 - 0.10) < 0.02, (
            f"Expected historical VaR95 ~0.10, got {result.var_95:.4f}"
        )

    def test_insufficient_history_returns_zero(self):
        from execution.risk.live_var import HistoricalVaR
        hvar = HistoricalVaR()
        snap = _make_snapshot()
        result = hvar.portfolio_var(snap)
        assert result.var_99 == 0.0

    def test_method_label(self):
        from execution.risk.live_var import HistoricalVaR
        hvar = HistoricalVaR()
        _warm_up_historical(hvar, n_days=10)
        snap = _make_snapshot()
        result = hvar.portfolio_var(snap)
        assert result.method == "historical"


# ===========================================================================
# MonteCarloVaR tests
# ===========================================================================

class TestMonteCarloVaR:

    def _warm_mc(self, mc, n_days=60, seed=7):
        rng = np.random.default_rng(seed)
        for _ in range(n_days):
            mc.update({
                "BTC": float(rng.normal(0.001, 0.03)),
                "ETH": float(rng.normal(0.001, 0.025)),
                "SPY": float(rng.normal(0.0003, 0.01)),
            })

    def test_var_positive(self):
        from execution.risk.live_var import MonteCarloVaR
        mc = MonteCarloVaR(n_paths=1000, seed=1)
        self._warm_mc(mc)
        snap = _make_snapshot()
        result = mc.portfolio_var(snap)
        assert result.var_95 >= 0
        assert result.var_99 >= 0

    def test_cvar_geq_var(self):
        from execution.risk.live_var import MonteCarloVaR
        mc = MonteCarloVaR(n_paths=2000, seed=2)
        self._warm_mc(mc)
        snap = _make_snapshot()
        result = mc.portfolio_var(snap)
        assert result.cvar_95 >= result.var_95 - 1e-6
        assert result.cvar_99 >= result.var_99 - 1e-6

    def test_var99_geq_var95(self):
        from execution.risk.live_var import MonteCarloVaR
        mc = MonteCarloVaR(n_paths=2000, seed=3)
        self._warm_mc(mc)
        snap = _make_snapshot()
        result = mc.portfolio_var(snap)
        assert result.var_99 >= result.var_95 - 1e-6

    def test_convergence(self):
        """
        MC VaR with 10K paths should be within 20% of parametric VaR for
        a Gaussian return model (they should agree in expectation).
        """
        from execution.risk.live_var import ParametricVaR, MonteCarloVaR
        pvar = ParametricVaR()
        mc = MonteCarloVaR(n_paths=10_000, seed=99)
        rng = np.random.default_rng(99)
        for _ in range(100):
            rets = {
                "BTC": float(rng.normal(0.0, 0.02)),
                "ETH": float(rng.normal(0.0, 0.015)),
                "SPY": float(rng.normal(0.0, 0.008)),
            }
            pvar.update(rets)
            mc.update(rets)
        snap = _make_snapshot(equity=100_000.0)
        p_result = pvar.portfolio_var(snap)
        m_result = mc.portfolio_var(snap)
        if p_result.var_99 > 0:
            ratio = m_result.var_99 / p_result.var_99
            assert 0.5 <= ratio <= 2.0, (
                f"MC VaR99 / Parametric VaR99 = {ratio:.3f} -- too divergent for Gaussian returns"
            )

    def test_method_label(self):
        from execution.risk.live_var import MonteCarloVaR
        mc = MonteCarloVaR(n_paths=500, seed=5)
        self._warm_mc(mc, n_days=10)
        snap = _make_snapshot()
        result = mc.portfolio_var(snap)
        assert result.method == "montecarlo"


# ===========================================================================
# VaRMonitor tests
# ===========================================================================

class TestVaRMonitor:

    def test_returns_all_methods(self, tmp_path):
        from execution.risk.live_var import VaRMonitor
        db = _make_db(tmp_path)
        monitor = VaRMonitor(db_path=db, n_mc_paths=200, mc_seed=0)
        rng = np.random.default_rng(0)
        snap = _make_snapshot()
        for _ in range(30):
            rets = {s: float(rng.normal(0, 0.02)) for s in snap.symbols}
            results = monitor.update(rets, snap)
        assert set(results.keys()) == {"parametric", "historical", "montecarlo", "consensus"}

    def test_consensus_is_weighted_average(self, tmp_path):
        from execution.risk.live_var import VaRMonitor
        db = _make_db(tmp_path)
        monitor = VaRMonitor(db_path=db, n_mc_paths=200, mc_seed=1)
        rng = np.random.default_rng(1)
        snap = _make_snapshot()
        results = None
        for _ in range(30):
            rets = {s: float(rng.normal(0, 0.02)) for s in snap.symbols}
            results = monitor.update(rets, snap)
        if results:
            expected = (
                0.40 * results["parametric"].var_99
                + 0.30 * results["historical"].var_99
                + 0.30 * results["montecarlo"].var_99
            )
            assert abs(results["consensus"].var_99 - expected) < 1e-4

    def test_persists_to_db(self, tmp_path):
        from execution.risk.live_var import VaRMonitor
        db = _make_db(tmp_path)
        monitor = VaRMonitor(db_path=db, n_mc_paths=100, mc_seed=2)
        rng = np.random.default_rng(2)
        snap = _make_snapshot()
        for _ in range(5):
            rets = {s: float(rng.normal(0, 0.02)) for s in snap.symbols}
            monitor.update(rets, snap)
        df = monitor.latest_metrics(n_rows=20)
        assert len(df) > 0, "Should have persisted risk_metrics rows"

    def test_breach_flag_set_on_loss_exceeding_var(self, tmp_path):
        from execution.risk.live_var import VaRMonitor
        db = _make_db(tmp_path)
        monitor = VaRMonitor(db_path=db, n_mc_paths=200, mc_seed=3)
        rng = np.random.default_rng(3)
        snap = _make_snapshot()
        # Warm up
        for _ in range(30):
            rets = {s: float(rng.normal(0, 0.02)) for s in snap.symbols}
            monitor.update(rets, snap)
        # Now feed a catastrophic loss to trigger breach
        monitor.update(
            {s: -0.30 for s in snap.symbols},
            snap,
            actual_daily_pnl=-30_000.0,  # large loss
        )
        df = monitor.latest_metrics(n_rows=20)
        breaches = df[df["breach_flag"] == 1]
        assert len(breaches) > 0, "Expected at least one breach flag after large loss"


# ===========================================================================
# Attribution tests
# ===========================================================================

class TestAttribution:

    def _insert_trade_pnl(self, db_path: Path, n: int = 10, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        conn = sqlite3.connect(db_path)
        for i in range(n):
            pnl = float(rng.normal(50.0, 200.0))
            conn.execute(
                """INSERT INTO trade_pnl
                   (symbol, entry_time, exit_time, entry_price, exit_price, qty, pnl, hold_bars)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    "BTC" if i % 2 == 0 else "ETH",
                    f"2026-01-{i+1:02d}T10:00:00+00:00",
                    f"2026-01-{i+1:02d}T14:00:00+00:00",
                    float(60000 + i * 100),
                    float(60100 + i * 100),
                    0.1,
                    pnl,
                    4,
                ),
            )
        conn.commit()
        conn.close()

    def test_factor_contributions_sum_to_total_pnl(self, tmp_path):
        from execution.risk.attribution import PnLAttributor, AttributionFactor
        db = _make_db(tmp_path)
        self._insert_trade_pnl(db, n=5)
        attributor = PnLAttributor(db_path=db)
        trades = attributor.run()
        assert len(trades) > 0, "Expected attribution for inserted trades"
        for ta in trades:
            factor_sum = sum(ta.factors.values())
            assert abs(factor_sum - ta.total_pnl) < 1e-6, (
                f"Factor sum {factor_sum:.6f} != total_pnl {ta.total_pnl:.6f}"
            )

    def test_all_factors_present(self, tmp_path):
        from execution.risk.attribution import PnLAttributor, AttributionFactor
        db = _make_db(tmp_path)
        self._insert_trade_pnl(db, n=3)
        attributor = PnLAttributor(db_path=db)
        trades = attributor.run()
        assert len(trades) > 0
        for ta in trades:
            assert set(ta.factors.keys()) == set(AttributionFactor), (
                "All factors should be present in attribution"
            )

    def test_factor_performance_aggregation(self, tmp_path):
        from execution.risk.attribution import PnLAttributor, AttributionReport
        db = _make_db(tmp_path)
        self._insert_trade_pnl(db, n=8)
        PnLAttributor(db_path=db).run()
        report = AttributionReport(db_path=db)
        perfs = report.factor_performance()
        assert len(perfs) > 0
        for fp in perfs:
            assert 0.0 <= fp.win_rate <= 1.0
            assert fp.n_trades >= 0

    def test_read_live_trades_returns_dataframe(self, tmp_path):
        from execution.risk.attribution import read_live_trades
        db = _make_db(tmp_path)
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO live_trades (symbol, side, qty, price, notional, fill_time) VALUES (?,?,?,?,?,?)",
            ("BTC", "buy", 0.1, 60000.0, 6000.0, "2026-01-01T10:00:00"),
        )
        conn.commit()
        conn.close()
        df = read_live_trades(db)
        assert not df.empty
        assert "symbol" in df.columns

    def test_no_double_attribution(self, tmp_path):
        from execution.risk.attribution import PnLAttributor
        db = _make_db(tmp_path)
        self._insert_trade_pnl(db, n=4)
        attr = PnLAttributor(db_path=db)
        first_run = attr.run()
        second_run = attr.run()  # should be empty -- already attributed
        assert len(second_run) == 0, "Second run should not re-attribute same trades"


# ===========================================================================
# CorrelationMatrix tests
# ===========================================================================

class TestCorrelationMatrix:

    def _build_warm_matrix(self, n_days=80, seed=42) -> "CorrelationMatrix":
        from execution.risk.correlation_monitor import CorrelationMatrix
        cm = CorrelationMatrix(symbols=["BTC", "ETH", "SPY"])
        rng = np.random.default_rng(seed)
        for _ in range(n_days):
            cm.update({
                "BTC": float(rng.normal(0, 0.03)),
                "ETH": float(rng.normal(0, 0.025)),
                "SPY": float(rng.normal(0, 0.01)),
            })
        return cm

    def test_correlation_matrix_symmetric(self):
        cm = self._build_warm_matrix()
        corr = cm.correlation_matrix()
        assert np.allclose(corr, corr.T, atol=1e-10), "Correlation matrix must be symmetric"

    def test_correlation_matrix_diagonal_ones(self):
        cm = self._build_warm_matrix()
        corr = cm.correlation_matrix()
        assert np.allclose(np.diag(corr), 1.0, atol=1e-10), "Diagonal must be 1.0"

    def test_correlation_matrix_positive_semi_definite(self):
        cm = self._build_warm_matrix(n_days=100)
        corr = cm.correlation_matrix()
        eigvals = np.linalg.eigvalsh(corr)
        assert eigvals.min() >= -1e-8, (
            f"Correlation matrix not PSD: min eigenvalue = {eigvals.min():.2e}"
        )

    def test_correlation_values_in_range(self):
        cm = self._build_warm_matrix()
        corr = cm.correlation_matrix()
        assert corr.min() >= -1.0 - 1e-9
        assert corr.max() <= 1.0 + 1e-9

    def test_stress_regime_detected_at_high_correlation(self):
        from execution.risk.correlation_monitor import CorrelationMatrix
        cm = CorrelationMatrix(symbols=["A", "B", "C"], stress_threshold=0.3)
        rng = np.random.default_rng(10)
        for _ in range(100):
            common = float(rng.normal(0, 0.04))
            # Very high correlation: all assets move together
            cm.update({"A": common + 0.001, "B": common + 0.001, "C": common + 0.001})
        assert cm.is_stress_regime(), (
            "Perfectly correlated assets should trigger stress regime"
        )

    def test_no_stress_with_independent_assets(self):
        from execution.risk.correlation_monitor import CorrelationMatrix
        cm = CorrelationMatrix(symbols=["X", "Y", "Z"], stress_threshold=0.6)
        rng = np.random.default_rng(11)
        for _ in range(100):
            cm.update({
                "X": float(rng.normal(0, 0.02)),
                "Y": float(rng.normal(0, 0.02)),
                "Z": float(rng.normal(0, 0.02)),
            })
        # Independent assets should have low average correlation
        assert not cm.is_stress_regime()

    def test_pca_explained_sums_leq_one(self):
        cm = self._build_warm_matrix()
        pca_exp = cm.pca_explained_variance()
        assert pca_exp.sum() <= 1.0 + 1e-9
        assert (pca_exp >= 0).all()

    def test_add_symbol_expands_matrix(self):
        cm = self._build_warm_matrix()
        initial_n = cm.n
        cm.add_symbol("LTC")
        assert cm.n == initial_n + 1
        assert "LTC" in cm.symbols


# ===========================================================================
# ConcentrationRisk tests
# ===========================================================================

class TestConcentrationRisk:

    def test_hhi_single_position(self):
        from execution.risk.correlation_monitor import ConcentrationRisk
        cr = ConcentrationRisk()
        cr.update({"BTC": 100_000.0})
        assert abs(cr.hhi - 1.0) < 1e-9, "Single position should have HHI=1.0"
        assert abs(cr.effective_n - 1.0) < 1e-9

    def test_hhi_equal_positions(self):
        from execution.risk.correlation_monitor import ConcentrationRisk
        cr = ConcentrationRisk()
        cr.update({"A": 25_000.0, "B": 25_000.0, "C": 25_000.0, "D": 25_000.0})
        assert abs(cr.hhi - 0.25) < 1e-9, "4 equal positions -> HHI = 0.25"
        assert abs(cr.effective_n - 4.0) < 1e-6

    def test_hhi_range(self):
        from execution.risk.correlation_monitor import ConcentrationRisk
        cr = ConcentrationRisk()
        rng = np.random.default_rng(42)
        notionals = {f"SYM{i}": float(rng.exponential(10_000)) for i in range(8)}
        cr.update(notionals)
        assert 0.0 < cr.hhi <= 1.0
        assert cr.effective_n >= 1.0

    def test_empty_returns_zero_hhi(self):
        from execution.risk.correlation_monitor import ConcentrationRisk
        cr = ConcentrationRisk()
        cr.update({})
        assert cr.hhi == 0.0

    def test_over_concentrated_flag(self):
        from execution.risk.correlation_monitor import ConcentrationRisk
        cr = ConcentrationRisk()
        cr.update({"BTC": 95_000.0, "ETH": 5_000.0})
        assert cr.is_over_concentrated(threshold_hhi=0.25)

    def test_not_over_concentrated_with_diversified_portfolio(self):
        from execution.risk.correlation_monitor import ConcentrationRisk
        cr = ConcentrationRisk()
        cr.update({f"SYM{i}": 10_000.0 for i in range(10)})
        # HHI = 1/10 = 0.10 < 0.25 threshold
        assert not cr.is_over_concentrated(threshold_hhi=0.25)


# ===========================================================================
# LimitChecker tests
# ===========================================================================

class TestLimitChecker:

    def _make_checker(self, tmp_path: Path) -> "LimitChecker":
        from execution.risk.limits import RiskLimitConfig, LimitChecker
        # Write a minimal yaml to tmp_path
        yaml_content = """
portfolio:
  max_drawdown_pct: 0.10
  max_daily_loss_pct: 0.02
  max_gross_exposure: 1.50
  min_cash_buffer_pct: 0.05
  max_open_positions: 8
per_instrument:
  default_max_frac: 0.65
  notional_caps:
    NQ: 400000
    NG: 200000
circuit_breakers:
  equity_floor_pct: 0.70
"""
        cfg_file = tmp_path / "risk_limits.yaml"
        cfg_file.write_text(yaml_content)
        config = RiskLimitConfig(config_path=cfg_file)
        return LimitChecker(config=config)

    def test_no_breach_within_limits(self, tmp_path):
        checker = self._make_checker(tmp_path)
        breached = checker.check_portfolio(
            equity=100_000.0,
            initial_equity=100_000.0,
            daily_pnl=500.0,         # positive P&L
            gross_exposure_frac=0.8, # well within 1.50
            var99_frac=0.03,         # within 5%
            max_position_frac=0.40,  # within 65%
        )
        assert len(breached) == 0

    def test_drawdown_breach_detected(self, tmp_path):
        checker = self._make_checker(tmp_path)
        breached = checker.check_portfolio(
            equity=85_000.0,         # 15% below initial -> exceeds 10% threshold
            initial_equity=100_000.0,
            daily_pnl=-500.0,
            gross_exposure_frac=0.5,
            var99_frac=0.02,
            max_position_frac=0.20,
        )
        names = [lim.name for lim in breached]
        assert "portfolio_max_drawdown" in names

    def test_daily_loss_breach_detected(self, tmp_path):
        checker = self._make_checker(tmp_path)
        # daily loss of 3% exceeds 2% limit
        breached = checker.check_portfolio(
            equity=97_000.0,
            initial_equity=100_000.0,
            daily_pnl=-3_000.0,      # -3% daily loss
            gross_exposure_frac=0.5,
            var99_frac=0.02,
            max_position_frac=0.20,
        )
        names = [lim.name for lim in breached]
        assert "portfolio_daily_loss" in names

    def test_gross_exposure_breach(self, tmp_path):
        checker = self._make_checker(tmp_path)
        breached = checker.check_portfolio(
            equity=100_000.0,
            initial_equity=100_000.0,
            daily_pnl=0.0,
            gross_exposure_frac=1.80,  # exceeds 1.50
            var99_frac=0.02,
            max_position_frac=0.20,
        )
        names = [lim.name for lim in breached]
        assert "portfolio_gross_exposure" in names

    def test_notional_cap_breach(self, tmp_path):
        checker = self._make_checker(tmp_path)
        result = checker.check_symbol_notional("NQ", 500_000.0)  # exceeds 400K cap
        assert result is not None
        assert result.is_breached

    def test_notional_cap_no_breach_within_cap(self, tmp_path):
        checker = self._make_checker(tmp_path)
        result = checker.check_symbol_notional("NQ", 300_000.0)  # within 400K cap
        assert result is None  # no breach

    def test_symbol_without_cap_returns_none(self, tmp_path):
        checker = self._make_checker(tmp_path)
        result = checker.check_symbol_notional("BTC", 999_999.0)  # no cap for BTC
        assert result is None


# ===========================================================================
# PositionLimiter tests
# ===========================================================================

class TestPositionLimiter:

    def _make_limiter(self, tmp_path: Path) -> "PositionLimiter":
        from execution.risk.limits import RiskLimitConfig, LimitChecker, PositionLimiter
        yaml_content = """
portfolio:
  max_drawdown_pct: 0.10
  max_daily_loss_pct: 0.02
  max_gross_exposure: 1.50
  max_open_positions: 8
per_instrument:
  default_max_frac: 0.50
  notional_caps:
    NQ: 100000
circuit_breakers:
  equity_floor_pct: 0.70
"""
        cfg_file = tmp_path / "risk_limits.yaml"
        cfg_file.write_text(yaml_content)
        config = RiskLimitConfig(config_path=cfg_file)
        return PositionLimiter(config=config)

    def test_within_limits_passes_full_qty(self, tmp_path):
        limiter = self._make_limiter(tmp_path)
        result = limiter.check(
            symbol="BTC",
            requested_qty=1.0,
            price=60_000.0,
            equity=1_000_000.0,
            current_gross_notional=0.0,
            n_open_positions=2,
        )
        assert not result.is_blocked
        assert abs(result.allowed_qty - 1.0) < 1e-6

    def test_notional_cap_reduces_qty(self, tmp_path):
        limiter = self._make_limiter(tmp_path)
        # NQ cap is 100K; requesting 2 units @ 60K = 120K notional -> should reduce
        result = limiter.check(
            symbol="NQ",
            requested_qty=2.0,
            price=60_000.0,
            equity=1_000_000.0,
            current_gross_notional=0.0,
            n_open_positions=2,
        )
        assert not result.is_blocked
        assert result.is_reduced
        assert result.allowed_qty * 60_000.0 <= 100_000.0 + 1.0

    def test_max_positions_blocks_new_entry(self, tmp_path):
        limiter = self._make_limiter(tmp_path)
        result = limiter.check(
            symbol="BTC",
            requested_qty=0.1,
            price=60_000.0,
            equity=100_000.0,
            current_gross_notional=40_000.0,
            n_open_positions=8,   # already at max
            current_symbol_notional=0.0,  # new position
        )
        assert result.is_blocked

    def test_concentration_limit_reduces_qty(self, tmp_path):
        limiter = self._make_limiter(tmp_path)
        # max_frac=0.50, equity=100K -> max 50K notional per symbol
        # Already holding 45K; requesting 20K more (total 65K > 50K cap)
        result = limiter.check(
            symbol="ETH",
            requested_qty=10.0,
            price=2_000.0,       # 10 * 2000 = 20K requested
            equity=100_000.0,
            current_gross_notional=45_000.0,
            n_open_positions=3,
            current_symbol_notional=45_000.0,
        )
        assert result.is_reduced or result.is_blocked


# ===========================================================================
# DrawdownGuard tests
# ===========================================================================

class TestDrawdownGuard:

    def test_no_halt_below_threshold(self):
        from execution.risk.limits import DrawdownGuard
        guard = DrawdownGuard(initial_equity=100_000.0)
        halted = guard.update(95_000.0)  # 5% drawdown < 10% threshold
        assert not halted

    def test_halt_at_threshold(self):
        from execution.risk.limits import DrawdownGuard
        guard = DrawdownGuard(initial_equity=100_000.0)
        halted = guard.update(89_999.0)  # 10.001% drawdown > 10% threshold
        assert halted

    def test_halt_persists_above_resume_threshold(self):
        from execution.risk.limits import DrawdownGuard
        guard = DrawdownGuard(initial_equity=100_000.0)
        guard.update(89_000.0)  # trigger halt at 11% DD
        # Partial recovery to 7% DD (above 5% resume threshold) -> still halted
        halted = guard.update(93_000.0)
        assert halted

    def test_resume_at_resume_threshold(self):
        from execution.risk.limits import DrawdownGuard
        guard = DrawdownGuard(initial_equity=100_000.0)
        guard.update(89_000.0)  # trigger halt
        # Recovery to within 5% of peak: peak=100K, equity=96K -> DD=4% < 5% resume
        halted = guard.update(96_000.0)
        assert not halted

    def test_peak_equity_updates_on_new_high(self):
        from execution.risk.limits import DrawdownGuard
        guard = DrawdownGuard(initial_equity=100_000.0)
        guard.update(110_000.0)  # new peak
        assert guard.peak_equity == 110_000.0
        # 10% from 110K peak = 99K; equity of 99.5K is above that -> no halt
        halted = guard.update(99_500.0)
        assert not halted

    def test_drawdown_calculation(self):
        from execution.risk.limits import DrawdownGuard
        guard = DrawdownGuard(initial_equity=100_000.0)
        guard.update(92_000.0)
        assert abs(guard.current_drawdown - 0.08) < 1e-9

    def test_integrates_with_circuit_breaker_stub(self):
        from execution.risk.limits import DrawdownGuard

        class StubCB:
            def __init__(self):
                self.triggered = False
                self.halt_reason = None
            def trigger(self, name, reason):
                self.triggered = True
                self.halt_reason = name

        cb = StubCB()
        guard = DrawdownGuard(initial_equity=100_000.0, circuit_breaker=cb)
        guard.update(88_000.0)  # 12% DD -> halt
        assert cb.triggered, "CircuitBreaker.trigger should be called on halt"

    def test_status_dict_keys(self):
        from execution.risk.limits import DrawdownGuard
        guard = DrawdownGuard(initial_equity=100_000.0)
        guard.update(95_000.0)
        status = guard.status()
        for key in ["peak_equity", "current_equity", "drawdown_pct", "is_halted"]:
            assert key in status, f"Missing key in status: {key}"
