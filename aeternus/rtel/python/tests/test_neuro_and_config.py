"""
AETERNUS RTEL — Tests for neuro_interface, config, and monitoring.
"""
from __future__ import annotations

import json
import os
import pytest
import numpy as np

from rtel.neuro_interface import (
    VolSurfaceMessage, NeuroSDEInterface,
    CompressedTensor, TensorNetInterface,
    LuminaPrediction, LuminaInterface,
    AgentAction, HyperAgentInterface,
    GraphAdjacency, OmniGraphInterface,
    RTELModuleHub,
)
from rtel.config import (
    RTELConfig, ShmBusConfig, RiskConfig, ExecutionConfig,
    SignalConfig, PortfolioConfig, SimulationConfig, DataPipelineConfig,
)
from rtel.monitoring import (
    HealthRegistry, HealthStatus, MetricsAggregator, AlertRule,
    AlertManager, MonitoringDashboard, StructuredLogger, MonitoringSystem,
)


# ============================================================================
# Neuro-SDE Interface tests
# ============================================================================

class TestVolSurfaceMessage:
    def test_iv_access(self):
        surf = np.ones((10, 5), dtype=np.float32) * 0.25
        msg  = VolSurfaceMessage(0, 0.0, 10, 5, 0.25, 0.0, 0.0, surf)
        assert abs(msg.iv(0, 0) - 0.25) < 1e-6
        assert abs(msg.iv(9, 4) - 0.25) < 1e-6

    def test_out_of_bounds_returns_atm(self):
        surf = np.ones((5, 3), dtype=np.float32) * 0.20
        msg  = VolSurfaceMessage(0, 0.0, 5, 3, 0.20, 0.0, 0.0, surf)
        assert msg.iv(-1, 0) == 0.20
        assert msg.iv(100, 0) == 0.20

    def test_risk_reversal(self):
        surf = np.zeros((20, 4), dtype=np.float32)
        # Put wing higher vol than call wing
        for i in range(20):
            surf[i, 0] = 0.20 + 0.01 * (10 - i)  # smile with negative skew
        msg = VolSurfaceMessage(0, 0.0, 20, 4, 0.20, -0.1, 0.0, surf)
        rr = msg.risk_reversal(25.0)
        # rr = iv_high_strike - iv_low_strike; depends on smile shape
        assert isinstance(rr, float)


class TestNeuroSDEInterface:
    def test_on_vol_surface(self):
        iface = NeuroSDEInterface(n_assets=3)
        surf  = np.ones((10, 4), dtype=np.float32) * 0.25
        msg   = VolSurfaceMessage(0, 0.0, 10, 4, 0.25, 0.0, 0.0, surf)
        iface.on_vol_surface(msg)
        assert iface.latest_surface(0) is not None

    def test_callback_called(self):
        iface    = NeuroSDEInterface(n_assets=2)
        received = []
        iface.add_callback(lambda m: received.append(m))
        surf = np.ones((5, 3), dtype=np.float32) * 0.20
        msg  = VolSurfaceMessage(1, 0.0, 5, 3, 0.20, 0.0, 0.0, surf)
        iface.on_vol_surface(msg)
        assert len(received) == 1

    def test_generate_synthetic_valid(self):
        iface = NeuroSDEInterface(n_assets=5, n_strikes=15, n_expiries=6)
        msg   = iface.generate_synthetic(0, base_vol=0.20)
        assert msg.n_strikes == 15
        assert msg.n_expiries == 6
        assert msg.atm_vol > 0
        assert msg.surface.shape == (15, 6)

    def test_seed_all(self):
        iface = NeuroSDEInterface(n_assets=4)
        iface.seed_all()
        vols = iface.atm_vols()
        assert len(vols) == 4
        assert all(v > 0 for v in vols.values())


# ============================================================================
# TensorNet Interface tests
# ============================================================================

class TestTensorNetInterface:
    def test_on_tensor(self):
        iface = TensorNetInterface(n_assets=3)
        core1 = np.random.randn(1, 5, 2).astype(np.float32)
        core2 = np.random.randn(2, 4, 1).astype(np.float32)
        t = CompressedTensor(0, 0.0, (5, 4), [core1, core2],
                             compression_ratio=3.0, recon_error=0.001)
        iface.on_tensor(t)
        assert iface._tensors[0] is t

    def test_compression_stats(self):
        iface = TensorNetInterface(n_assets=3)
        for i in range(3):
            t = CompressedTensor(i, 0.0, (5, 4), [],
                                 compression_ratio=2.0 + i, recon_error=0.01)
            iface.on_tensor(t)
        stats = iface.compression_stats()
        assert stats["n_tensors"] == 3
        assert stats["mean_ratio"] > 0


# ============================================================================
# Lumina Interface tests
# ============================================================================

class TestLuminaInterface:
    def test_on_prediction(self):
        iface = LuminaInterface(n_assets=3)
        pred  = LuminaPrediction(0, 0.0, 0.8, 0.9, 60.0)
        iface.on_prediction(pred)
        assert iface.signal(0) == pytest.approx(0.8 * 0.9)

    def test_all_signals(self):
        iface = LuminaInterface(n_assets=3)
        for i in range(3):
            iface.on_prediction(LuminaPrediction(i, 0.0, 0.5*(i%2==0 or -1), 0.8, 60.0))
        sigs = iface.all_signals()
        assert len(sigs) == 3

    def test_rolling_ic_updates(self):
        iface = LuminaInterface(n_assets=1)
        for _ in range(5):
            iface.on_prediction(LuminaPrediction(0, 0.0, 0.5, 0.9, 60.0))
            iface.update_realized_return(0, 0.01)
        ic = iface.rolling_ic(0)
        assert isinstance(ic, float)

    def test_synthetic_prediction(self):
        iface = LuminaInterface(n_assets=2)
        pred  = iface.generate_synthetic(0, price_change=0.02)
        assert -1.0 <= pred.direction <= 1.0
        assert 0.0 <= pred.confidence <= 1.0


# ============================================================================
# HyperAgent Interface tests
# ============================================================================

class TestHyperAgentInterface:
    def test_on_action(self):
        iface  = HyperAgentInterface(n_assets=2)
        action = AgentAction(0, 0.0, 0.5, 0.8)
        iface.on_action(action)
        targets = iface.get_target_positions()
        assert targets[0] == pytest.approx(0.5)

    def test_synthetic_action_valid(self):
        iface  = HyperAgentInterface(n_assets=3)
        action = iface.generate_synthetic(0, signal=0.7)
        assert -1.0 <= action.position_target <= 1.0
        assert 0.0 <= action.urgency <= 1.0

    def test_n_actions_count(self):
        iface = HyperAgentInterface(n_assets=2)
        for _ in range(5):
            iface.on_action(AgentAction(0, 0.0, 0.1, 0.5))
        assert iface.n_actions() == 5


# ============================================================================
# OmniGraph Interface tests
# ============================================================================

class TestOmniGraphInterface:
    def test_on_adjacency(self):
        iface = OmniGraphInterface(n_assets=4)
        adj   = np.eye(4, dtype=np.float32)
        msg   = GraphAdjacency(0.0, 4, adj)
        iface.on_adjacency(msg)
        assert iface.latest() is not None

    def test_generate_synthetic(self):
        iface = OmniGraphInterface(n_assets=5)
        adj   = iface.generate_synthetic()
        assert adj.n_nodes == 5
        assert adj.adj_matrix.shape == (5, 5)
        # Diagonal should be 1
        for i in range(5):
            assert abs(adj.adj_matrix[i, i] - 1.0) < 1e-6

    def test_density(self):
        n   = 4
        adj_full = np.ones((n, n), dtype=np.float32)
        msg = GraphAdjacency(0.0, n, adj_full)
        assert msg.density() == pytest.approx(1.0)

        adj_diag = np.eye(n, dtype=np.float32)
        msg2 = GraphAdjacency(0.0, n, adj_diag)
        assert msg2.density() == pytest.approx(0.0)


# ============================================================================
# RTELModuleHub tests
# ============================================================================

class TestRTELModuleHub:
    def test_runs_without_error(self):
        hub = RTELModuleHub(n_assets=5)
        prices = {i: 100.0 + i for i in range(5)}
        for step in range(20):
            price_changes = {i: 0.001 * step * (1 + 0.1*i) for i in range(5)}
            hub.update_all(prices, price_changes)
        diag = hub.diagnostics()
        assert diag["step"] == 20
        assert diag["n_assets"] == 5

    def test_target_positions_range(self):
        hub = RTELModuleHub(n_assets=3)
        prices = {0: 100.0, 1: 200.0, 2: 50.0}
        hub.update_all(prices, {0: 0.01, 1: -0.02, 2: 0.005})
        targets = hub.target_positions()
        for v in targets.values():
            assert -1.0 <= v <= 1.0

    def test_atm_vols_populated(self):
        hub = RTELModuleHub(n_assets=4)
        prices = {i: 100.0 for i in range(4)}
        for _ in range(6):  # enough steps to trigger vol surface update
            hub.update_all(prices)
        vols = hub.atm_vols()
        assert len(vols) == 4


# ============================================================================
# Config tests
# ============================================================================

class TestShmBusConfig:
    def test_default_valid(self):
        cfg = ShmBusConfig()
        cfg.validate()  # should not raise

    def test_invalid_ring_capacity(self):
        cfg = ShmBusConfig(ring_capacity=100)  # not power of 2
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_valid_power_of_two(self):
        for cap in [1, 2, 4, 8, 64, 256, 1024, 4096]:
            cfg = ShmBusConfig(ring_capacity=cap)
            cfg.validate()


class TestRTELConfig:
    def test_default_valid(self):
        cfg = RTELConfig()
        cfg.validate()

    def test_to_json_roundtrip(self):
        cfg  = RTELConfig()
        json_str = cfg.to_json()
        cfg2 = RTELConfig.from_json(json_str)
        assert cfg2.mode == cfg.mode
        assert cfg2.risk.max_leverage == cfg.risk.max_leverage
        assert cfg2.portfolio.method == cfg.portfolio.method

    def test_for_simulation(self):
        cfg = RTELConfig.for_simulation(n_assets=5, n_steps=200)
        assert cfg.mode == "simulation"
        assert cfg.simulation.n_assets == 5
        assert cfg.simulation.n_steps == 200
        cfg.validate()

    def test_for_backtest(self):
        cfg = RTELConfig.for_backtest(n_assets=10)
        assert cfg.mode == "backtest"
        cfg.validate()

    def test_for_live(self):
        cfg = RTELConfig.for_live()
        assert cfg.mode == "live"
        assert cfg.risk.max_leverage <= 2.0  # conservative
        cfg.validate()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RTEL_MODE", "backtest")
        monkeypatch.setenv("RTEL_RISK_MAX_LEVERAGE", "3.0")
        cfg = RTELConfig()
        cfg.apply_env_overrides()
        assert cfg.mode == "backtest"
        assert cfg.risk.max_leverage == pytest.approx(3.0)

    def test_invalid_mode(self):
        cfg = RTELConfig(mode="invalid")
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_dict_roundtrip(self):
        cfg = RTELConfig.for_simulation(n_assets=3)
        d   = cfg.to_dict()
        cfg2 = RTELConfig.from_dict(d)
        assert cfg2.simulation.n_assets == 3
        assert cfg2.mode == "simulation"


class TestPortfolioConfig:
    def test_valid_methods(self):
        for method in ["erc", "mvo", "kelly", "min_var"]:
            cfg = PortfolioConfig(method=method)
            cfg.validate()

    def test_invalid_method(self):
        cfg = PortfolioConfig(method="invalid_method")
        with pytest.raises(AssertionError):
            cfg.validate()


# ============================================================================
# Monitoring tests
# ============================================================================

class TestHealthRegistry:
    def test_heartbeat_ok(self):
        reg = HealthRegistry()
        reg.heartbeat("component_a", HealthStatus.OK, "all good")
        health = reg.to_dict()
        assert health["component_a"]["status"] == HealthStatus.OK

    def test_register_and_check(self):
        reg = HealthRegistry()
        reg.register("test_comp", lambda: (HealthStatus.OK, "running"))
        health = reg.check_all()
        assert health["test_comp"].status == HealthStatus.OK

    def test_overall_critical_if_any_critical(self):
        reg = HealthRegistry()
        reg.heartbeat("ok_comp",  HealthStatus.OK)
        reg.heartbeat("bad_comp", HealthStatus.CRITICAL)
        assert reg.overall_status() == HealthStatus.CRITICAL

    def test_overall_ok_if_all_ok(self):
        reg = HealthRegistry()
        reg.heartbeat("a", HealthStatus.OK)
        reg.heartbeat("b", HealthStatus.OK)
        assert reg.overall_status() == HealthStatus.OK


class TestMetricsAggregator:
    def test_gauge(self):
        m = MetricsAggregator()
        m.set_gauge("test_gauge", 42.0)
        prom = m.export_prometheus()
        assert "42" in prom

    def test_counter_increments(self):
        m = MetricsAggregator()
        m.inc_counter("test_counter", 1.0)
        m.inc_counter("test_counter", 5.0)
        snap = m.snapshot()
        # The key may have labels or not
        total = sum(v for k, v in snap["counters"].items()
                    if "test_counter" in k)
        assert total == pytest.approx(6.0)

    def test_histogram_percentiles(self):
        m = MetricsAggregator()
        for i in range(100):
            m.observe_histogram("latency_ns", float(i * 1000))
        prom = m.export_prometheus()
        assert "p99" in prom
        assert "p50" in prom

    def test_labels(self):
        m = MetricsAggregator()
        m.set_gauge("rtel_pub", 100.0, labels={"channel": "lob"})
        prom = m.export_prometheus()
        assert "channel" in prom


class TestAlertManager:
    def test_alert_fires(self):
        m       = MetricsAggregator()
        h       = HealthRegistry()
        manager = AlertManager(m, h)

        m.set_gauge("error_count", 200.0)
        fired = []

        manager.add_rule(AlertRule(
            name="test_alert",
            condition=lambda m: m.snapshot()["gauges"].get("error_count", 0) > 100,
            severity="warning",
            message="Too many errors",
            cooldown_s=0.0,
        ))
        manager.add_handler(lambda a: fired.append(a))

        manager.evaluate()
        assert len(fired) == 1
        assert fired[0].rule_name == "test_alert"

    def test_alert_resolves(self):
        m       = MetricsAggregator()
        h       = HealthRegistry()
        manager = AlertManager(m, h)

        m.set_gauge("err", 200.0)
        manager.add_rule(AlertRule(
            name="err_alert",
            condition=lambda m: m.snapshot()["gauges"].get("err", 0) > 100,
            severity="warning",
            cooldown_s=0.0,
        ))
        manager.evaluate()
        assert len(manager.active_alerts()) == 1

        m.set_gauge("err", 50.0)
        manager.evaluate()
        assert len(manager.active_alerts()) == 0


class TestMonitoringSystem:
    def test_creates_without_error(self):
        sys = MonitoringSystem()
        assert sys.health is not None
        assert sys.metrics is not None

    def test_register_component(self):
        sys = MonitoringSystem()
        sys.register_component("test", lambda: (HealthStatus.OK, "ok"))
        health = sys.health.check_all()
        assert "test" in health

    def test_report_returns_string(self):
        sys    = MonitoringSystem()
        sys.health.heartbeat("comp", HealthStatus.OK)
        report = sys.report()
        assert isinstance(report, str)
        assert "AETERNUS" in report

    def test_prometheus_export(self):
        sys = MonitoringSystem()
        sys.metrics.set_gauge("rtel_test", 1.0)
        prom = sys.export_prometheus()
        assert "rtel_test" in prom


class TestStructuredLogger:
    def test_log_levels(self):
        sink   = []
        import collections
        deq    = collections.deque(maxlen=100)
        logger = StructuredLogger("test", sink=deq)
        logger.info("test_event", key="value", num=42)
        logger.warn("warn_event", code=500)
        assert len(deq) == 2
        assert deq[0]["event"] == "test_event"
        assert deq[0]["level"] == "INFO"

    def test_recent(self):
        import collections
        deq    = collections.deque(maxlen=100)
        logger = StructuredLogger("test2", sink=deq)
        for i in range(10):
            logger.debug(f"event_{i}")
        recent = logger.recent(5)
        assert len(recent) == 5
