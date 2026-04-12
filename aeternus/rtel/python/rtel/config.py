"""
AETERNUS Real-Time Execution Layer (RTEL)
config.py — Configuration management for all RTEL components

Provides:
- RTELConfig dataclass hierarchy for all subsystems
- INI/YAML/JSON config file loading
- Environment variable overrides
- Config validation with sensible defaults
- Runtime config update support
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SHM Bus config
# ---------------------------------------------------------------------------

@dataclass
class ShmBusConfig:
    slot_bytes:     int   = 65536       # 64 KB per slot
    ring_capacity:  int   = 1024        # must be power of 2
    shm_prefix:     str   = "/aeternus"
    use_posix_shm:  bool  = True
    use_hugepages:  bool  = False
    timeout_ms:     int   = 5000

    def validate(self) -> None:
        assert self.slot_bytes > 0
        assert self.ring_capacity > 0 and (self.ring_capacity & (self.ring_capacity-1)) == 0, \
            "ring_capacity must be power of 2"
        assert self.timeout_ms > 0

# ---------------------------------------------------------------------------
# Scheduler config
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    n_worker_threads:     int   = 4
    watchdog_timeout_ms:  int   = 1000
    cpu_affinity_start:   int   = 0
    use_realtime_priority: bool = False
    pipeline_budget_us:   int   = 1000   # µs

    def validate(self) -> None:
        assert self.n_worker_threads >= 1
        assert self.watchdog_timeout_ms > 0
        assert self.pipeline_budget_us > 0

# ---------------------------------------------------------------------------
# Risk config
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    max_position_usd:    float = 1_000_000.0
    max_leverage:        float = 4.0
    max_concentration:   float = 0.20
    max_drawdown:        float = 0.10
    daily_var_limit_usd: float = 50_000.0
    margin_rate:         float = 0.05
    kelly_fraction:      float = 0.25

    def validate(self) -> None:
        assert self.max_position_usd > 0
        assert self.max_leverage > 0
        assert 0 < self.max_concentration <= 1.0
        assert 0 < self.max_drawdown <= 1.0
        assert 0 < self.kelly_fraction <= 1.0

# ---------------------------------------------------------------------------
# Execution config
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    commission_bps:      float = 5.0
    slippage_bps:        float = 2.0
    min_trade_usd:       float = 100.0
    max_order_size_usd:  float = 100_000.0
    taker_fee_bps:       float = 3.0
    maker_fee_bps:       float = 1.0
    min_tick:            float = 0.01
    rebalance_frequency: int   = 5       # steps

    def validate(self) -> None:
        assert self.commission_bps >= 0
        assert self.slippage_bps >= 0
        assert self.min_trade_usd > 0
        assert self.rebalance_frequency >= 1

# ---------------------------------------------------------------------------
# Signal config
# ---------------------------------------------------------------------------

@dataclass
class SignalConfig:
    lookback_short:    int   = 5
    lookback_long:     int   = 20
    min_icir:          float = 0.1
    ic_window:         int   = 60
    signal_threshold:  float = 0.05
    use_momentum:      bool  = True
    use_mean_rev:      bool  = True
    use_lob_imbal:     bool  = True
    use_vol_surface:   bool  = True
    use_ema_cross:     bool  = True
    use_rsi:           bool  = True

    def validate(self) -> None:
        assert self.lookback_short >= 1
        assert self.lookback_long >= self.lookback_short
        assert self.ic_window >= 10
        assert 0 <= self.signal_threshold <= 1.0

# ---------------------------------------------------------------------------
# Portfolio optimization config
# ---------------------------------------------------------------------------

@dataclass
class PortfolioConfig:
    method:            str   = "erc"    # erc | mvo | kelly | min_var
    max_position_pct:  float = 0.15
    kelly_fraction:    float = 0.25
    cov_alpha:         float = 0.05     # EWMA decay for covariance
    rebalance_every:   int   = 5
    min_rebalance_pct: float = 0.005
    max_turnover:      float = 0.5

    def validate(self) -> None:
        assert self.method in {"erc", "mvo", "kelly", "min_var"}
        assert 0 < self.max_position_pct <= 1.0
        assert 0 < self.kelly_fraction <= 1.0
        assert self.cov_alpha > 0
        assert self.rebalance_every >= 1

# ---------------------------------------------------------------------------
# Data pipeline config
# ---------------------------------------------------------------------------

@dataclass
class DataPipelineConfig:
    n_assets:            int   = 10
    bar_duration_s:      float = 1.0
    max_spread_bps:      float = 100.0
    max_price_jump_pct:  float = 5.0
    anomaly_z_threshold: float = 4.0
    feature_seq_len:     int   = 32
    normalization_clip:  float = 4.0

    def validate(self) -> None:
        assert self.n_assets >= 1
        assert self.bar_duration_s > 0
        assert self.max_spread_bps > 0
        assert self.feature_seq_len >= 1

# ---------------------------------------------------------------------------
# Monitoring config
# ---------------------------------------------------------------------------

@dataclass
class MonitoringConfig:
    enable_prometheus:   bool  = True
    prometheus_port:     int   = 9090
    heartbeat_interval_s: float = 5.0
    stale_threshold_s:   float = 30.0
    alert_cooldown_s:    float = 60.0
    log_level:           str   = "INFO"
    enable_dashboard:    bool  = False

    def validate(self) -> None:
        assert 1 <= self.prometheus_port <= 65535
        assert self.heartbeat_interval_s > 0
        assert self.stale_threshold_s > 0
        assert self.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# ---------------------------------------------------------------------------
# Simulation config
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    n_assets:        int   = 10
    n_steps:         int   = 1000
    initial_capital: float = 1_000_000.0
    base_price:      float = 100.0
    mu:              float = 0.0001
    sigma:           float = 0.01
    correlation:     float = 0.3
    dt:              float = 1.0 / 252.0
    spread_bps:      float = 5.0
    seed:            int   = 42

    def validate(self) -> None:
        assert self.n_assets >= 1
        assert self.n_steps >= 1
        assert self.initial_capital > 0
        assert self.base_price > 0
        assert 0 <= self.correlation <= 1.0
        assert self.dt > 0

# ---------------------------------------------------------------------------
# Master RTELConfig
# ---------------------------------------------------------------------------

@dataclass
class RTELConfig:
    """Master configuration for the AETERNUS RTEL system."""

    # Subsystem configs
    shm_bus:     ShmBusConfig     = field(default_factory=ShmBusConfig)
    scheduler:   SchedulerConfig  = field(default_factory=SchedulerConfig)
    risk:        RiskConfig       = field(default_factory=RiskConfig)
    execution:   ExecutionConfig  = field(default_factory=ExecutionConfig)
    signal:      SignalConfig     = field(default_factory=SignalConfig)
    portfolio:   PortfolioConfig  = field(default_factory=PortfolioConfig)
    pipeline:    DataPipelineConfig = field(default_factory=DataPipelineConfig)
    monitoring:  MonitoringConfig = field(default_factory=MonitoringConfig)
    simulation:  SimulationConfig = field(default_factory=SimulationConfig)

    # Global settings
    mode:        str  = "simulation"   # simulation | live | backtest
    log_level:   str  = "INFO"
    data_dir:    str  = "/tmp/aeternus"
    model_dir:   str  = "/tmp/aeternus/models"
    enable_rtel: bool = True

    def validate(self) -> None:
        assert self.mode in {"simulation", "live", "backtest"}
        assert self.log_level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        self.shm_bus.validate()
        self.scheduler.validate()
        self.risk.validate()
        self.execution.validate()
        self.signal.validate()
        self.portfolio.validate()
        self.pipeline.validate()
        self.monitoring.validate()
        self.simulation.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RTELConfig":
        cfg = cls()
        for key, val in d.items():
            if key == "shm_bus" and isinstance(val, dict):
                cfg.shm_bus = ShmBusConfig(**{k: v for k, v in val.items()
                                              if k in ShmBusConfig.__dataclass_fields__})
            elif key == "scheduler" and isinstance(val, dict):
                cfg.scheduler = SchedulerConfig(**{k: v for k, v in val.items()
                                                   if k in SchedulerConfig.__dataclass_fields__})
            elif key == "risk" and isinstance(val, dict):
                cfg.risk = RiskConfig(**{k: v for k, v in val.items()
                                        if k in RiskConfig.__dataclass_fields__})
            elif key == "execution" and isinstance(val, dict):
                cfg.execution = ExecutionConfig(**{k: v for k, v in val.items()
                                                   if k in ExecutionConfig.__dataclass_fields__})
            elif key == "signal" and isinstance(val, dict):
                cfg.signal = SignalConfig(**{k: v for k, v in val.items()
                                            if k in SignalConfig.__dataclass_fields__})
            elif key == "portfolio" and isinstance(val, dict):
                cfg.portfolio = PortfolioConfig(**{k: v for k, v in val.items()
                                                   if k in PortfolioConfig.__dataclass_fields__})
            elif key == "pipeline" and isinstance(val, dict):
                cfg.pipeline = DataPipelineConfig(**{k: v for k, v in val.items()
                                                     if k in DataPipelineConfig.__dataclass_fields__})
            elif key == "monitoring" and isinstance(val, dict):
                cfg.monitoring = MonitoringConfig(**{k: v for k, v in val.items()
                                                     if k in MonitoringConfig.__dataclass_fields__})
            elif key == "simulation" and isinstance(val, dict):
                cfg.simulation = SimulationConfig(**{k: v for k, v in val.items()
                                                     if k in SimulationConfig.__dataclass_fields__})
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    @classmethod
    def from_json(cls, json_str: str) -> "RTELConfig":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> "RTELConfig":
        with open(path, "r") as f:
            content = f.read()
        return cls.from_json(content)

    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides (RTEL_* prefix)."""
        for key, val_str in os.environ.items():
            if not key.startswith("RTEL_"):
                continue
            parts = key[5:].lower().split("_", 1)
            if len(parts) == 2:
                section, field_name = parts
                obj = getattr(self, section, None)
                if obj is None:
                    continue
                if hasattr(obj, field_name):
                    old_val = getattr(obj, field_name)
                    try:
                        if isinstance(old_val, bool):
                            new_val = val_str.lower() in ("1", "true", "yes")
                        elif isinstance(old_val, int):
                            new_val = int(val_str)
                        elif isinstance(old_val, float):
                            new_val = float(val_str)
                        else:
                            new_val = val_str
                        setattr(obj, field_name, new_val)
                        logger.debug("Config override: %s = %s", key, val_str)
                    except (ValueError, TypeError) as e:
                        logger.warning("Invalid env override %s=%s: %s", key, val_str, e)
            elif len(parts) == 1:
                field_name = parts[0]
                if hasattr(self, field_name):
                    old_val = getattr(self, field_name)
                    try:
                        if isinstance(old_val, bool):
                            new_val = val_str.lower() in ("1", "true", "yes")
                        elif isinstance(old_val, int):
                            new_val = int(val_str)
                        elif isinstance(old_val, float):
                            new_val = float(val_str)
                        else:
                            new_val = val_str
                        setattr(self, field_name, new_val)
                    except (ValueError, TypeError):
                        pass

    @classmethod
    def for_simulation(cls, n_assets: int = 10, n_steps: int = 1000) -> "RTELConfig":
        """Create a simulation-mode config."""
        cfg = cls(mode="simulation")
        cfg.simulation.n_assets  = n_assets
        cfg.simulation.n_steps   = n_steps
        cfg.pipeline.n_assets    = n_assets
        cfg.signal.lookback_long = 20
        cfg.portfolio.method     = "erc"
        return cfg

    @classmethod
    def for_backtest(cls, n_assets: int = 20) -> "RTELConfig":
        """Create a backtest-mode config."""
        cfg = cls(mode="backtest")
        cfg.simulation.n_assets = n_assets
        cfg.pipeline.n_assets   = n_assets
        cfg.execution.commission_bps = 5.0
        cfg.execution.slippage_bps   = 2.0
        cfg.risk.max_leverage        = 2.0
        return cfg

    @classmethod
    def for_live(cls) -> "RTELConfig":
        """Create a live-trading config (conservative risk limits)."""
        cfg = cls(mode="live")
        cfg.risk.max_leverage        = 1.5
        cfg.risk.max_concentration   = 0.10
        cfg.risk.max_drawdown        = 0.05
        cfg.execution.commission_bps = 3.0
        cfg.monitoring.enable_prometheus = True
        cfg.monitoring.log_level     = "WARNING"
        return cfg

    def print_summary(self) -> None:
        print(f"=== AETERNUS RTEL Config ===")
        print(f"  Mode:       {self.mode}")
        print(f"  N assets:   {self.pipeline.n_assets}")
        print(f"  Opt method: {self.portfolio.method}")
        print(f"  Max lev:    {self.risk.max_leverage}x")
        print(f"  Max DD:     {self.risk.max_drawdown*100:.1f}%")
        print(f"  Commission: {self.execution.commission_bps}bps")
        print(f"  Kelly:      {self.risk.kelly_fraction}")
        print(f"  Monitoring: {self.monitoring.enable_prometheus}")
