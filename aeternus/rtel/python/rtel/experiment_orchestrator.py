"""
AETERNUS Real-Time Execution Layer (RTEL)
experiment_orchestrator.py — Cross-Module Experiment Orchestrator

Launches Chronos simulation, streams data through TensorNet→OmniGraph→Lumina→HyperAgent.
Collects metrics from all modules. Produces unified experiment report.

Usage:
    config = ExperimentConfig(
        n_assets=5,
        n_steps=1000,
        symbols=["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
    )
    orch = ExperimentOrchestrator(config)
    report = orch.run()
    report.print()
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .feature_store import FeatureSchema, FeatureSnapshot, FeatureStore
from .pipeline_client import PipelineClient, PipelineRun, StageMetrics
from .shm_reader import LobSnapshot, ShmReader
from .shm_writer import ShmWriter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name:              str = "aeternus_experiment"
    n_assets:          int = 5
    n_steps:           int = 1000
    symbols:           List[str] = field(default_factory=lambda: [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])
    seed:              int = 42
    base_price:        float = 150.0
    tick_size:         float = 0.01
    initial_vol:       float = 0.02
    bid_ask_spread:    float = 0.05
    n_lob_levels:      int = 10
    dt:                float = 1.0 / 252 / 390  # 1 minute in years
    pipeline_timeout_ms: float = 50.0
    output_dir:        Path = Path("experiments")
    save_checkpoints:  bool = True
    checkpoint_every:  int = 100
    verbose:           bool = False
    lumina_model:      str = "linear"  # "linear" | "mlp" | "transformer"
    agent_policy:      str = "momentum"  # "momentum" | "mean_reversion" | "rl"
    risk_aversion:     float = 1.0
    max_position:      float = 1.0    # max absolute position per asset
    transaction_cost:  float = 0.0005  # 5 bps


# ---------------------------------------------------------------------------
# MarketSimulator — generates synthetic LOB data
# ---------------------------------------------------------------------------
class MarketSimulator:
    """
    GBM-based market simulator with LOB.
    Produces LobSnapshot objects at each time step.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self._rng = rng
        n = cfg.n_assets

        # Initial state
        self.prices = np.array([cfg.base_price * (1 + i * 0.05) for i in range(n)])
        self.vols   = np.full(n, cfg.initial_vol)
        self.t      = 0

    def step(self) -> List[LobSnapshot]:
        n     = self.cfg.n_assets
        dt    = self.cfg.dt
        snaps = []
        for i in range(n):
            # GBM: dS = S * (mu*dt + sigma*dW)
            dW = self._rng.standard_normal() * np.sqrt(dt)
            self.prices[i] *= (1.0 + 0.05 * dt + self.vols[i] * dW)
            # Vol clustering (GARCH-like)
            shock = abs(dW)
            self.vols[i] = max(0.005, self.vols[i] * 0.99 + 0.3 * shock * 0.01)

            snap = self._build_lob_snapshot(i, self.prices[i], self.vols[i])
            snaps.append(snap)

        self.t += 1
        return snaps

    def _build_lob_snapshot(self, asset_id: int, price: float, vol: float) -> LobSnapshot:
        n_levels = self.cfg.n_lob_levels
        half_spread = self.cfg.bid_ask_spread / 2

        snap = LobSnapshot()
        snap.asset_id = asset_id
        snap.exchange_ts_ns = time.time_ns()
        snap.sequence = self.t

        # Build LOB levels with exponentially decaying sizes
        for j in range(n_levels):
            # Bid: price - (j+1)*tick - half_spread
            bid_price = price - half_spread - j * self.cfg.tick_size
            ask_price = price + half_spread + j * self.cfg.tick_size
            bid_size  = max(1.0, 100.0 * np.exp(-0.3 * j) * (1 + 0.2 * self._rng.standard_normal()))
            ask_size  = max(1.0, 100.0 * np.exp(-0.3 * j) * (1 + 0.2 * self._rng.standard_normal()))
            snap.bids.append((bid_price, bid_size))
            snap.asks.append((ask_price, ask_size))

        snap.mid_price    = price
        snap.spread       = 2 * half_spread
        bid_depth         = sum(s for _, s in snap.bids)
        ask_depth         = sum(s for _, s in snap.asks)
        total_depth       = bid_depth + ask_depth
        snap.bid_imbalance= (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0
        snap.vwap_bid     = sum(p*s for p,s in snap.bids) / bid_depth if bid_depth > 0 else price
        snap.vwap_ask     = sum(p*s for p,s in snap.asks) / ask_depth if ask_depth > 0 else price
        return snap


# ---------------------------------------------------------------------------
# Module stubs (lightweight stand-ins when actual modules unavailable)
# ---------------------------------------------------------------------------
class LuminaStub:
    """Linear regression model for return forecasting."""

    def __init__(self, n_assets: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights = rng.standard_normal(n_assets) * 0.01
        self.bias    = np.zeros(n_assets)

    def forward(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # features: [n_features]
        # Return: (return_forecast, risk_forecast, confidence)
        n = len(self.weights)
        feat_slice = features[:n] if len(features) >= n else np.pad(features, (0, n-len(features)))
        returns    = np.tanh(feat_slice * self.weights + self.bias)
        risks      = np.abs(returns) * 0.1 + 0.001
        confidence = np.minimum(1.0, np.abs(returns) * 2.0)
        return returns.astype(np.float32), risks.astype(np.float32), confidence.astype(np.float32)


class HyperAgentStub:
    """Simple momentum/mean-reversion policy."""

    def __init__(self, policy: str = "momentum", risk_aversion: float = 1.0,
                 max_position: float = 1.0):
        self.policy       = policy
        self.risk_aversion= risk_aversion
        self.max_position = max_position
        self.positions    = None

    def forward(self, returns: np.ndarray, risks: np.ndarray,
                confidence: np.ndarray) -> np.ndarray:
        n = len(returns)
        if self.positions is None:
            self.positions = np.zeros(n)

        if self.policy == "momentum":
            signal = returns * confidence
        elif self.policy == "mean_reversion":
            signal = -returns * confidence
        else:
            signal = returns

        # Kelly-like sizing
        risk_adj = risks + 1e-6
        sizing   = signal / (self.risk_aversion * risk_adj)
        sizing   = np.clip(sizing, -self.max_position, self.max_position)

        delta = sizing - self.positions
        self.positions += delta
        return delta.astype(np.float32)


# ---------------------------------------------------------------------------
# PortfolioTracker — tracks P&L and performance metrics
# ---------------------------------------------------------------------------
class PortfolioTracker:
    def __init__(self, n_assets: int, transaction_cost: float = 0.0005):
        self.n_assets        = n_assets
        self.transaction_cost= transaction_cost
        self.positions       = np.zeros(n_assets)
        self.pnl_history:    List[float] = []
        self.cost_history:   List[float] = []
        self.prices_history: List[np.ndarray] = []
        self.gross_pnl       = 0.0
        self.total_cost      = 0.0
        self.n_trades        = 0

    def update(self, deltas: np.ndarray, prices: np.ndarray,
               returns: Optional[np.ndarray] = None) -> float:
        n = min(len(deltas), self.n_assets)
        old_pos = self.positions[:n].copy()
        new_pos = old_pos + deltas[:n]
        self.positions[:n] = new_pos

        # Transaction costs
        cost = np.sum(np.abs(deltas[:n])) * self.transaction_cost * np.mean(prices[:n])
        self.total_cost += cost
        self.n_trades   += int(np.sum(np.abs(deltas[:n]) > 1e-6))

        # P&L (position × return)
        if returns is not None:
            step_pnl = float(np.dot(old_pos[:len(returns)], returns[:n]))
        else:
            step_pnl = 0.0

        net_pnl = step_pnl - cost
        self.gross_pnl += step_pnl
        self.pnl_history.append(net_pnl)
        self.cost_history.append(cost)
        return net_pnl

    @property
    def cumulative_pnl(self) -> float:
        return sum(self.pnl_history) if self.pnl_history else 0.0

    @property
    def sharpe_ratio(self) -> float:
        if len(self.pnl_history) < 2:
            return 0.0
        arr  = np.array(self.pnl_history)
        mean = arr.mean()
        std  = arr.std()
        return (mean / std * np.sqrt(252 * 390)) if std > 1e-10 else 0.0

    @property
    def max_drawdown(self) -> float:
        if not self.pnl_history:
            return 0.0
        cum = np.cumsum(self.pnl_history)
        peak = np.maximum.accumulate(cum)
        drawdown = peak - cum
        return float(drawdown.max())

    def summary(self) -> Dict[str, float]:
        return {
            "cumulative_pnl": round(self.cumulative_pnl, 6),
            "gross_pnl":      round(self.gross_pnl, 6),
            "total_cost":     round(self.total_cost, 6),
            "sharpe_ratio":   round(self.sharpe_ratio, 4),
            "max_drawdown":   round(self.max_drawdown, 6),
            "n_trades":       self.n_trades,
        }


# ---------------------------------------------------------------------------
# ExperimentReport — unified report from all modules
# ---------------------------------------------------------------------------
@dataclass
class ExperimentReport:
    config:          Dict[str, Any]
    portfolio:       Dict[str, float]
    latency:         Dict[str, Any]
    model_accuracy:  Dict[str, float]
    feature_stats:   Dict[str, Any]
    pipeline_stats:  Dict[str, Any]
    timestamp:       float = field(default_factory=time.time)

    def print(self) -> None:
        print("=" * 60)
        print(f" AETERNUS Experiment Report")
        print(f" {self.config.get('name', 'unnamed')}")
        print("=" * 60)
        print("\n--- Portfolio Performance ---")
        for k, v in self.portfolio.items():
            print(f"  {k:<25} {v:>12.4f}")
        print("\n--- Pipeline Latency ---")
        for k, v in self.latency.items():
            print(f"  {k:<25} {v}")
        print("\n--- Model Accuracy ---")
        for k, v in self.model_accuracy.items():
            print(f"  {k:<25} {v:.4f}")
        print("\n--- Feature Stats ---")
        print(f"  {self.feature_stats}")
        print("=" * 60)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config":         self.config,
            "portfolio":      self.portfolio,
            "latency":        self.latency,
            "model_accuracy": self.model_accuracy,
            "feature_stats":  self.feature_stats,
            "pipeline_stats": self.pipeline_stats,
            "timestamp":      self.timestamp,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Report saved to %s", path)


# ---------------------------------------------------------------------------
# ExperimentOrchestrator — main orchestrator
# ---------------------------------------------------------------------------
class ExperimentOrchestrator:
    """
    End-to-end AETERNUS experiment orchestrator.
    Coordinates simulation, data flow, inference, and metric collection.
    """

    def __init__(self, config: ExperimentConfig, base_path: Path = Path("/tmp")):
        self.cfg       = config
        self.base_path = base_path

        # Modules
        self._sim     = MarketSimulator(config)
        self._lumina  = LuminaStub(config.n_assets, config.seed)
        self._agent   = HyperAgentStub(
            config.agent_policy, config.risk_aversion, config.max_position)
        self._portfolio = PortfolioTracker(config.n_assets, config.transaction_cost)

        # Feature store
        self._store   = FeatureStore()

        # Metrics
        self._pipeline_latencies: List[int] = []
        self._stage_latencies: Dict[str, List[int]] = defaultdict(list)
        self._return_forecasts: List[np.ndarray] = []
        self._actual_returns:   List[np.ndarray] = []
        self._steps_run        = 0
        self._prev_prices      = np.array([config.base_price] * config.n_assets)

        logger.info("ExperimentOrchestrator '%s': %d assets, %d steps",
                    config.name, config.n_assets, config.n_steps)

    def run(self) -> ExperimentReport:
        """Run the full experiment and return a report."""
        logger.info("Starting experiment '%s'", self.cfg.name)
        t_start = time.perf_counter()

        for step in range(self.cfg.n_steps):
            self._run_step(step)

            if self.cfg.verbose and step % 100 == 0:
                print(f"  Step {step:4d}/{self.cfg.n_steps}: "
                      f"PnL={self._portfolio.cumulative_pnl:.4f} "
                      f"Sharpe={self._portfolio.sharpe_ratio:.3f}")

            if self.cfg.save_checkpoints and step % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(step)

        t_elapsed = time.perf_counter() - t_start
        logger.info("Experiment complete in %.2fs (%d steps, %.1f steps/s)",
                    t_elapsed, self.cfg.n_steps,
                    self.cfg.n_steps / t_elapsed)

        return self._build_report()

    def _run_step(self, step: int) -> None:
        t0 = time.perf_counter_ns()

        # 1. Market simulation → LOB snapshots
        t_lob = time.perf_counter_ns()
        snaps = self._sim.step()
        self._stage_latencies["market_sim"].append(time.perf_counter_ns() - t_lob)

        # 2. Update feature store
        t_feat = time.perf_counter_ns()
        for snap in snaps:
            self._store.update_from_lob(snap)
        self._stage_latencies["feature_update"].append(time.perf_counter_ns() - t_feat)

        # 3. Lumina inference
        t_lum = time.perf_counter_ns()
        features = self._store.group_vector("lob")
        returns, risks, confidence = self._lumina.forward(features)
        self._store.update_from_predictions(
            returns[:self.cfg.n_assets],
            risks[:self.cfg.n_assets],
            confidence[:self.cfg.n_assets],
        )
        self._stage_latencies["lumina"].append(time.perf_counter_ns() - t_lum)

        # 4. HyperAgent action
        t_agent = time.perf_counter_ns()
        deltas = self._agent.forward(
            returns[:self.cfg.n_assets],
            risks[:self.cfg.n_assets],
            confidence[:self.cfg.n_assets],
        )
        self._stage_latencies["agent"].append(time.perf_counter_ns() - t_agent)

        # 5. Compute actual returns for accuracy tracking
        current_prices = self._sim.prices[:self.cfg.n_assets].copy()
        actual_rets = (current_prices - self._prev_prices) / (self._prev_prices + 1e-10)
        self._prev_prices = current_prices.copy()
        self._return_forecasts.append(returns[:self.cfg.n_assets].copy())
        self._actual_returns.append(actual_rets.copy())

        # 6. Portfolio update
        self._portfolio.update(deltas, current_prices, actual_rets)

        # 7. Create feature snapshot periodically
        if step % 10 == 0:
            self._store.snapshot(pipeline_id=step)

        t1 = time.perf_counter_ns()
        self._pipeline_latencies.append(t1 - t0)
        self._steps_run += 1

    def _compute_forecast_accuracy(self) -> Dict[str, float]:
        if len(self._return_forecasts) < 2 or len(self._actual_returns) < 2:
            return {}
        forecasts = np.array(self._return_forecasts)
        actuals   = np.array(self._actual_returns)

        # Direction accuracy (sign agreement)
        direction_acc = float(np.mean(
            np.sign(forecasts) == np.sign(actuals)
        ))
        # Spearman rank correlation (per-step, averaged)
        try:
            from scipy.stats import spearmanr
            corrs = [spearmanr(forecasts[i], actuals[i]).correlation
                     for i in range(len(forecasts))]
            ic_mean = float(np.nanmean(corrs))
        except ImportError:
            # Fallback: Pearson IC
            valid_steps = [(f, a) for f, a in zip(forecasts, actuals)
                           if np.std(f) > 0 and np.std(a) > 0]
            if valid_steps:
                corrs = [np.corrcoef(f, a)[0, 1] for f, a in valid_steps]
                ic_mean = float(np.nanmean(corrs))
            else:
                ic_mean = 0.0

        mse = float(np.mean((forecasts - actuals) ** 2))
        return {
            "direction_accuracy": round(direction_acc, 4),
            "information_coefficient_mean": round(ic_mean, 4),
            "forecast_mse": round(mse, 8),
        }

    def _latency_stats(self) -> Dict[str, Any]:
        stats = {}
        all_lat = np.array(self._pipeline_latencies, dtype=np.int64)
        if len(all_lat) > 0:
            stats["pipeline"] = {
                "mean_us":  round(float(all_lat.mean()) / 1000, 2),
                "p50_us":   round(float(np.percentile(all_lat, 50)) / 1000, 2),
                "p99_us":   round(float(np.percentile(all_lat, 99)) / 1000, 2),
                "max_us":   round(float(all_lat.max()) / 1000, 2),
            }

        for stage, lats in self._stage_latencies.items():
            arr = np.array(lats, dtype=np.int64)
            stats[stage] = {
                "mean_us":  round(float(arr.mean()) / 1000, 2),
                "p99_us":   round(float(np.percentile(arr, 99)) / 1000, 2),
            }
        return stats

    def _build_report(self) -> ExperimentReport:
        cfg_dict = {
            "name":        self.cfg.name,
            "n_assets":    self.cfg.n_assets,
            "n_steps":     self.cfg.n_steps,
            "seed":        self.cfg.seed,
            "lumina_model":self.cfg.lumina_model,
            "agent_policy":self.cfg.agent_policy,
        }
        return ExperimentReport(
            config=cfg_dict,
            portfolio=self._portfolio.summary(),
            latency=self._latency_stats(),
            model_accuracy=self._compute_forecast_accuracy(),
            feature_stats=self._store.stats(),
            pipeline_stats={
                "steps_run":   self._steps_run,
                "sla_1ms_pct": round(
                    np.mean(np.array(self._pipeline_latencies) < 1_000_000) * 100, 2
                ) if self._pipeline_latencies else 0.0,
            },
        )

    def _save_checkpoint(self, step: int) -> None:
        path = self.cfg.output_dir / self.cfg.name / f"checkpoint_{step:06d}.json"
        snap = self._store.latest_snapshot()
        if snap:
            snap_path = self.cfg.output_dir / self.cfg.name / f"features_{step:06d}.json"
            try:
                snap_path.parent.mkdir(parents=True, exist_ok=True)
                with open(snap_path, "w") as f:
                    json.dump({
                        "version": snap.version,
                        "timestamp_ns": snap.timestamp_ns,
                        "portfolio": self._portfolio.summary(),
                    }, f)
            except Exception as e:
                logger.debug("Checkpoint save error: %s", e)

    def feature_store(self) -> FeatureStore:
        return self._store

    def portfolio(self) -> PortfolioTracker:
        return self._portfolio

    @classmethod
    def run_grid_search(
        cls,
        base_config: ExperimentConfig,
        param_grid: Dict[str, List[Any]],
        n_jobs: int = 1,
    ) -> List[Tuple[Dict[str, Any], ExperimentReport]]:
        """
        Run a grid search over experiment parameters.
        Returns list of (params, report) tuples sorted by Sharpe ratio.
        """
        import itertools
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        results = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            cfg = ExperimentConfig(
                name=f"{base_config.name}_{'_'.join(str(v) for v in combo)}",
                n_assets=base_config.n_assets,
                n_steps=base_config.n_steps,
                seed=base_config.seed,
                **{k: v for k, v in params.items()
                   if hasattr(base_config, k)},
            )
            orch = cls(cfg)
            report = orch.run()
            results.append((params, report))
            logger.info("Grid search: %s → Sharpe=%.4f", params,
                        report.portfolio.get("sharpe_ratio", 0))

        # Sort by Sharpe ratio
        results.sort(key=lambda x: x[1].portfolio.get("sharpe_ratio", 0), reverse=True)
        return results
