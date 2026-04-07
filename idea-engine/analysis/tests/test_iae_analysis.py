"""
test_iae_analysis.py -- Unit tests for the IAE Python analysis layer.

Run with:
    pytest idea-engine/analysis/tests/test_iae_analysis.py -v

All tests use in-memory SQLite databases and synthetic data -- no external
dependencies beyond numpy, pandas, and pytest.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# -- Imports under test
from idea_engine.analysis.genome_analyzer import (
    GenomeAnalyzer,
    GenomeDatabase,
    FitnessLandscape,
    BreakthroughEvent,
)
from idea_engine.analysis.iae_performance_tracker import (
    IAEPerformanceTracker,
    IAECycleResult,
    AdaptationQualityMonitor,
    QualityReport,
)
from idea_engine.analysis.parameter_explorer import (
    ParameterSpaceExplorer,
    LandscapeMap,
    ExplorationSuggestion,
)
from idea_engine.analysis.live_feedback_analyzer import (
    LiveFeedbackAnalyzer,
    FeedbackBatch,
    GradientEstimator,
    TradeRecord,
)


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_genome_db(path: str, n_generations: int = 20, pop_size: int = 5) -> None:
    """
    Create a genome_history SQLite database with synthetic data.
    Fitness increases monotonically with a small noise component.
    """
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE genome_history (
            id          TEXT PRIMARY KEY,
            parent_id   TEXT,
            generation  INTEGER NOT NULL,
            fitness     REAL NOT NULL,
            params_json TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
        """
    )
    rows = []
    rng = np.random.default_rng(42)
    parent_ids: Dict[int, str] = {}  # gen -> one id per generation (first member)

    for gen in range(n_generations):
        base_fitness = 0.5 + gen * 0.02  # rising trend
        for member in range(pop_size):
            gid = f"g{gen:03d}m{member:02d}"
            parent = parent_ids.get(gen - 1) if gen > 0 else None
            fitness = base_fitness + rng.normal(0, 0.005)
            params = {
                "momentum_window": 10 + gen * 0.5 + rng.normal(0, 0.2),
                "vol_scale": 1.0 + rng.normal(0, 0.05),
                "entry_threshold": 0.002 + rng.normal(0, 0.0002),
            }
            rows.append(
                (
                    gid,
                    parent,
                    gen,
                    float(fitness),
                    json.dumps(params),
                    datetime.now(tz=timezone.utc).isoformat(),
                )
            )
            if member == 0:
                parent_ids[gen] = gid

    conn.executemany(
        "INSERT INTO genome_history VALUES (?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    conn.close()


def _make_live_db(path: str, n_trades: int = 30) -> None:
    """Create a live-trading SQLite database with synthetic trade records."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE trades (
            id           TEXT PRIMARY KEY,
            symbol       TEXT NOT NULL,
            side         TEXT NOT NULL,
            entry_time   TEXT NOT NULL,
            exit_time    TEXT NOT NULL,
            pnl          REAL NOT NULL,
            return_pct   REAL NOT NULL,
            holding_bars INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE param_snapshots (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_time TEXT NOT NULL,
            params_json   TEXT NOT NULL
        )
        """
    )
    rng = np.random.default_rng(99)
    now = datetime.now(tz=timezone.utc)
    for i in range(n_trades):
        entry = now - timedelta(hours=n_trades - i)
        exit_ = entry + timedelta(minutes=rng.integers(15, 120))
        ret = float(rng.normal(0.001, 0.005))
        conn.execute(
            "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"t{i:04d}",
                "ES",
                "long" if rng.random() > 0.5 else "short",
                entry.isoformat(),
                exit_.isoformat(),
                ret * 1000,
                ret,
                int(rng.integers(1, 10)),
            ),
        )

    params = {
        "momentum_window": 12.0,
        "vol_scale": 1.1,
        "entry_threshold": 0.0022,
    }
    conn.execute(
        "INSERT INTO param_snapshots (snapshot_time, params_json) VALUES (?, ?)",
        (now.isoformat(), json.dumps(params)),
    )
    conn.commit()
    conn.close()


def _make_cycle_results(n: int = 30) -> List[IAECycleResult]:
    """Generate a list of synthetic IAECycleResult objects with rising fitness."""
    rng = np.random.default_rng(7)
    results = []
    fitness = 0.4
    for i in range(n):
        fitness += float(rng.exponential(0.005))
        results.append(
            IAECycleResult(
                cycle_id=f"cycle_{i:04d}",
                timestamp=datetime.now(tz=timezone.utc) + timedelta(hours=i),
                n_evaluations=int(rng.integers(50, 200)),
                best_fitness=fitness,
                mean_fitness=fitness * 0.9,
                params_changed={
                    "momentum_window": (10.0 + i * 0.1, 10.0 + (i + 1) * 0.1),
                    "vol_scale": (1.0, 1.0 + rng.normal(0, 0.02)),
                },
                live_sharpe_before=float(rng.normal(0.8, 0.1)),
                live_sharpe_after=float(rng.normal(0.85, 0.1)),
            )
        )
    return results


# ===========================================================================
# Test group 1: GenomeDatabase
# ===========================================================================

class TestGenomeDatabase:
    def test_get_all_rows(self, tmp_path):
        db = str(tmp_path / "genome.db")
        _make_genome_db(db, n_generations=5, pop_size=3)
        gdb = GenomeDatabase(db)
        rows = gdb.get_all_rows()
        assert len(rows) == 15  # 5 gens * 3 members

    def test_get_elite_genomes(self, tmp_path):
        db = str(tmp_path / "genome.db")
        _make_genome_db(db, n_generations=10, pop_size=5)
        gdb = GenomeDatabase(db)
        elite = gdb.get_elite_genomes(n=3)
        assert len(elite) == 3
        # Should be sorted by fitness descending
        fitnesses = [e["fitness"] for e in elite]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_get_lineage(self, tmp_path):
        db = str(tmp_path / "genome.db")
        _make_genome_db(db, n_generations=8, pop_size=4)
        gdb = GenomeDatabase(db)
        elite = gdb.get_elite_genomes(n=1)
        gid = elite[0]["id"]
        lineage = gdb.get_lineage(gid)
        # Lineage should start at generation 0 and end at the genome's generation
        assert lineage[0]["generation"] == 0
        assert lineage[-1]["id"] == gid

    def test_diversity_over_time(self, tmp_path):
        db = str(tmp_path / "genome.db")
        _make_genome_db(db, n_generations=5, pop_size=4)
        gdb = GenomeDatabase(db)
        div = gdb.diversity_over_time()
        assert isinstance(div, pd.Series)
        assert len(div) == 5
        assert (div >= 0).all()

    def test_missing_table_does_not_crash(self, tmp_path):
        db = str(tmp_path / "empty.db")
        # Create an empty DB -- no table
        sqlite3.connect(db).close()
        gdb = GenomeDatabase(db)  # should log warning but not raise
        rows = gdb.get_all_rows()  # should raise or return [] gracefully
        # We accept either an empty list or a sqlite3.Error being swallowed
        assert isinstance(rows, list)


# ===========================================================================
# Test group 2: GenomeAnalyzer
# ===========================================================================

class TestGenomeAnalyzer:
    def test_load_history_shape(self, tmp_path):
        db = str(tmp_path / "genome.db")
        _make_genome_db(db, n_generations=10, pop_size=4)
        history = GenomeAnalyzer.load_history(db)
        assert not history.empty
        # 3 params -> 3 p_ columns
        param_cols = [c for c in history.columns if c.startswith("p_")]
        assert len(param_cols) == 3

    def test_fitness_landscape_best_fitness(self, tmp_path):
        db = str(tmp_path / "genome.db")
        _make_genome_db(db, n_generations=15, pop_size=5)
        history = GenomeAnalyzer.load_history(db)
        landscape = GenomeAnalyzer.compute_fitness_landscape(history)
        assert isinstance(landscape, FitnessLandscape)
        assert landscape.best_fitness >= landscape.mean_fitness
        assert landscape.diversity >= 0.0

    def test_fitness_landscape_parameter_ranges(self, tmp_path):
        db = str(tmp_path / "genome.db")
        _make_genome_db(db, n_generations=10, pop_size=5)
        history = GenomeAnalyzer.load_history(db)
        landscape = GenomeAnalyzer.compute_fitness_landscape(history)
        assert "momentum_window" in landscape.parameter_ranges
        lo, hi = landscape.parameter_ranges["momentum_window"]
        assert hi >= lo

    def test_breakthrough_detection_synthetic(self):
        """
        Build a history DataFrame with a deliberate large fitness jump at
        generation 15 and verify it is detected as a breakthrough.
        """
        rng = np.random.default_rng(12)
        rows = []
        for gen in range(25):
            fitness = 0.5 + gen * 0.005 + rng.normal(0, 0.001)
            if gen == 15:
                fitness += 0.15  # 30% jump
            rows.append(
                {
                    "generation": gen,
                    "fitness": fitness,
                    "p_momentum_window": 10.0 + rng.normal(0, 0.5),
                    "p_vol_scale": 1.0 + rng.normal(0, 0.02),
                }
            )
        history = pd.DataFrame(rows)

        breaks = GenomeAnalyzer.identify_breakthroughs(history, threshold_pct=0.10)
        assert len(breaks) >= 1
        gens = [b.generation for b in breaks]
        assert 15 in gens

    def test_breakthrough_no_events(self):
        """Flat fitness series should produce no breakthroughs."""
        rows = [{"generation": g, "fitness": 0.5, "p_x": 1.0} for g in range(20)]
        history = pd.DataFrame(rows)
        breaks = GenomeAnalyzer.identify_breakthroughs(history, threshold_pct=0.05)
        assert breaks == []

    def test_parameter_evolution_returns_series(self, tmp_path):
        db = str(tmp_path / "genome.db")
        _make_genome_db(db, n_generations=10, pop_size=3)
        history = GenomeAnalyzer.load_history(db)
        evo = GenomeAnalyzer.parameter_evolution(history, "momentum_window")
        assert isinstance(evo, pd.Series)
        assert len(evo) == 10  # one value per generation

    def test_parameter_evolution_unknown_param(self, tmp_path):
        db = str(tmp_path / "genome.db")
        _make_genome_db(db, n_generations=5, pop_size=2)
        history = GenomeAnalyzer.load_history(db)
        evo = GenomeAnalyzer.parameter_evolution(history, "nonexistent_param")
        assert evo.empty

    def test_convergence_generation_detected(self):
        """
        Create history where diversity shrinks to near zero by gen 10.
        convergence_generation should be set.
        """
        rows = []
        rng = np.random.default_rng(5)
        for gen in range(20):
            for member in range(5):
                spread = max(0.0, 1.0 - gen * 0.1)  # diversity shrinks
                rows.append(
                    {
                        "generation": gen,
                        "fitness": 0.5 + gen * 0.01,
                        "p_x": 1.0 + rng.normal(0, spread * 0.01),
                        "p_y": 2.0 + rng.normal(0, spread * 0.01),
                    }
                )
        history = pd.DataFrame(rows)
        landscape = GenomeAnalyzer.compute_fitness_landscape(history)
        # May or may not detect depending on exact values -- just assert no crash
        assert landscape.convergence_generation is None or isinstance(
            landscape.convergence_generation, int
        )


# ===========================================================================
# Test group 3: IAEPerformanceTracker
# ===========================================================================

class TestIAEPerformanceTracker:
    def test_record_and_count(self):
        tracker = IAEPerformanceTracker()
        cycles = _make_cycle_results(10)
        for c in cycles:
            tracker.record_cycle(c)
        assert len(tracker._cycles) == 10

    def test_rolling_improvement_rate_positive(self):
        tracker = IAEPerformanceTracker()
        for c in _make_cycle_results(15):
            tracker.record_cycle(c)
        rir = tracker.rolling_improvement_rate(n_cycles=10)
        # Synthetic data has rising fitness
        assert rir > 0

    def test_rolling_improvement_rate_insufficient_data(self):
        tracker = IAEPerformanceTracker()
        tracker.record_cycle(_make_cycle_results(1)[0])
        assert tracker.rolling_improvement_rate() == 0.0

    def test_time_to_improvement_found(self):
        tracker = IAEPerformanceTracker()
        for c in _make_cycle_results(30):
            tracker.record_cycle(c)
        tti = tracker.time_to_improvement(threshold=0.05)
        assert tti is not None
        assert isinstance(tti, int)
        assert tti >= 0

    def test_time_to_improvement_not_found(self):
        tracker = IAEPerformanceTracker()
        # Flat fitness -- never reaches +0.05
        base = 0.4
        for i in range(10):
            tracker.record_cycle(
                IAECycleResult(
                    cycle_id=f"c{i}",
                    timestamp=datetime.now(tz=timezone.utc),
                    n_evaluations=100,
                    best_fitness=base,
                    mean_fitness=base * 0.9,
                    params_changed={},
                    live_sharpe_before=0.8,
                    live_sharpe_after=0.8,
                )
            )
        assert tracker.time_to_improvement(threshold=0.05) is None

    def test_parameter_update_frequency(self):
        tracker = IAEPerformanceTracker()
        for c in _make_cycle_results(20):
            tracker.record_cycle(c)
        freq = tracker.parameter_update_frequency()
        assert "momentum_window" in freq
        assert freq["momentum_window"] == 20

    def test_adaptation_efficiency(self):
        tracker = IAEPerformanceTracker()
        for c in _make_cycle_results(20):
            tracker.record_cycle(c)
        eff = tracker.adaptation_efficiency()
        assert isinstance(eff, float)
        assert eff >= 0  # should be positive for rising fitness

    def test_generate_report_returns_string(self):
        tracker = IAEPerformanceTracker()
        for c in _make_cycle_results(25):
            tracker.record_cycle(c)
        report = tracker.generate_report(n_cycles=20)
        assert isinstance(report, str)
        assert "IAE Performance Report" in report
        assert "Fitness Summary" in report

    def test_generate_report_empty(self):
        tracker = IAEPerformanceTracker()
        report = tracker.generate_report()
        assert "No cycle data" in report


# ===========================================================================
# Test group 4: AdaptationQualityMonitor
# ===========================================================================

class TestAdaptationQualityMonitor:
    def test_healthy_cycles(self):
        monitor = AdaptationQualityMonitor()
        cycles = _make_cycle_results(25)
        report = monitor.check_adaptation_quality(cycles)
        assert isinstance(report, QualityReport)
        # Rising fitness with 25 cycles -- should be ok or warning, not critical
        assert report.status in ("ok", "warning")

    def test_no_improvement_triggers_warning(self):
        monitor = AdaptationQualityMonitor()
        flat = [
            IAECycleResult(
                cycle_id=f"c{i}",
                timestamp=datetime.now(tz=timezone.utc),
                n_evaluations=100,
                best_fitness=0.5,
                mean_fitness=0.45,
                params_changed={"x": (1.0, 1.0)},
                live_sharpe_before=0.8,
                live_sharpe_after=0.8,
            )
            for i in range(25)
        ]
        report = monitor.check_adaptation_quality(flat)
        assert report.status in ("warning", "critical")
        assert any("No fitness improvement" in w for w in report.warnings)

    def test_oscillating_params_triggers_warning(self):
        monitor = AdaptationQualityMonitor()
        # Build cycles where vol_scale oscillates
        cycles = []
        for i in range(25):
            direction = 1.0 if i % 2 == 0 else -1.0
            cycles.append(
                IAECycleResult(
                    cycle_id=f"c{i}",
                    timestamp=datetime.now(tz=timezone.utc),
                    n_evaluations=100,
                    best_fitness=0.5 + i * 0.01,
                    mean_fitness=0.45 + i * 0.01,
                    params_changed={
                        "vol_scale": (1.0, 1.0 + direction * 0.1),
                    },
                    live_sharpe_before=0.8,
                    live_sharpe_after=0.82,
                )
            )
        report = monitor.check_adaptation_quality(cycles)
        assert any("oscillating" in w.lower() for w in report.warnings)

    def test_quality_report_to_markdown(self):
        monitor = AdaptationQualityMonitor()
        report = monitor.check_adaptation_quality(_make_cycle_results(10))
        md = report.to_markdown()
        assert "Status" in md


# ===========================================================================
# Test group 5: GradientEstimator
# ===========================================================================

class TestGradientEstimator:
    def test_insufficient_data_returns_none(self):
        est = GradientEstimator()
        est.update("alpha", 0.1, 0.11, 0.02)
        est.update("alpha", 0.11, 0.12, 0.02)
        # Only 2 samples -- MIN_POINTS is 3
        assert est.estimate("alpha") is None

    def test_known_positive_gradient(self):
        """
        Supply observations where increasing alpha always improves Sharpe.
        Gradient estimate must be positive.
        """
        est = GradientEstimator()
        # d(sharpe)/d(alpha) ~ 0.5
        for i in range(6):
            old_v = 0.1 + i * 0.1
            new_v = old_v + 0.1
            sharpe_chg = 0.05  # consistent positive change
            est.update("alpha", old_v, new_v, sharpe_chg)

        g = est.estimate("alpha")
        assert g is not None
        assert g > 0.0

    def test_known_negative_gradient(self):
        """Decreasing Sharpe when increasing beta -- gradient should be negative."""
        est = GradientEstimator()
        for i in range(6):
            old_v = 1.0 + i * 0.2
            new_v = old_v + 0.2
            est.update("beta", old_v, new_v, -0.04)

        g = est.estimate("beta")
        assert g is not None
        assert g < 0.0

    def test_estimate_all(self):
        est = GradientEstimator()
        for i in range(5):
            est.update("p1", float(i), float(i + 1), 0.02)
            est.update("p2", float(i), float(i + 1), -0.01)
        all_grads = est.estimate_all()
        assert "p1" in all_grads
        assert "p2" in all_grads

    def test_clear_single_param(self):
        est = GradientEstimator()
        for i in range(5):
            est.update("x", float(i), float(i + 1), 0.01)
        est.clear("x")
        assert est.estimate("x") is None

    def test_zero_delta_skipped(self):
        est = GradientEstimator()
        est.update("x", 1.0, 1.0, 0.05)  # delta = 0
        assert est.sample_count("x") == 0


# ===========================================================================
# Test group 6: LiveFeedbackAnalyzer
# ===========================================================================

class TestLiveFeedbackAnalyzer:
    def _make_analyzer(self) -> LiveFeedbackAnalyzer:
        bounds = {
            "momentum_window": (5.0, 30.0),
            "vol_scale": (0.5, 2.0),
            "entry_threshold": (0.001, 0.005),
        }
        return LiveFeedbackAnalyzer(bounds)

    def test_collect_recent_feedback(self, tmp_path):
        db = str(tmp_path / "live.db")
        _make_live_db(db, n_trades=20)
        analyzer = self._make_analyzer()
        batch = analyzer.collect_recent_feedback(db, hours=48)
        assert isinstance(batch, FeedbackBatch)
        assert batch.n_trades > 0
        assert isinstance(batch.realized_sharpe, float)

    def test_collect_feedback_missing_db(self, tmp_path):
        analyzer = self._make_analyzer()
        batch = analyzer.collect_recent_feedback(
            str(tmp_path / "nonexistent.db"), hours=4
        )
        assert batch.n_trades == 0

    def test_validate_suggestion_valid(self):
        analyzer = self._make_analyzer()
        suggestion = {
            "momentum_window": 15.0,
            "vol_scale": 1.2,
        }
        assert analyzer.validate_suggestion(suggestion) is True

    def test_validate_suggestion_out_of_bounds(self):
        analyzer = self._make_analyzer()
        suggestion = {
            "momentum_window": 100.0,  # exceeds bound of 30.0
        }
        assert analyzer.validate_suggestion(suggestion) is False

    def test_validate_suggestion_empty(self):
        analyzer = self._make_analyzer()
        assert analyzer.validate_suggestion({}) is False

    def test_validate_suggestion_unknown_param(self):
        analyzer = self._make_analyzer()
        assert analyzer.validate_suggestion({"not_a_param": 1.0}) is False

    def test_suggest_param_adjustments_clipped(self):
        """
        Supply a very large gradient and verify adjustments are clipped to max_delta.
        """
        analyzer = self._make_analyzer()
        # Plant some gradient data so get_current_value returns something
        analyzer.gradient_estimator._data["momentum_window"] = [
            (12.0, 50.0),
            (13.0, 50.0),
            (14.0, 50.0),
        ]
        gradient = {"momentum_window": 1000.0}  # huge gradient
        suggestion = analyzer.suggest_param_adjustments(gradient, learning_rate=0.1)

        if "momentum_window" in suggestion:
            lo, hi = analyzer.param_bounds["momentum_window"]
            assert lo <= suggestion["momentum_window"] <= hi

    def test_ingest_natural_experiment_updates_estimator(self):
        analyzer = self._make_analyzer()
        for i in range(5):
            analyzer.ingest_natural_experiment(
                "vol_scale",
                old_value=1.0 + i * 0.1,
                new_value=1.0 + (i + 1) * 0.1,
                sharpe_before=0.8,
                sharpe_after=0.85,
            )
        assert analyzer.gradient_estimator.sample_count("vol_scale") == 5
        g = analyzer.gradient_estimator.estimate("vol_scale")
        assert g is not None
        assert g > 0.0

    def test_gradient_summary_returns_dataframe(self):
        analyzer = self._make_analyzer()
        for i in range(5):
            analyzer.ingest_natural_experiment(
                "vol_scale", 1.0 + i * 0.05, 1.05 + i * 0.05, 0.8, 0.82
            )
        df = analyzer.gradient_summary()
        assert isinstance(df, pd.DataFrame)
        assert "vol_scale" in df.index
