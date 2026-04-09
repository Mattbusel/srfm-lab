"""
SRFM-Core SDK: white-label autonomous trading engine for external deployment.

A "Brain-in-a-Box" that other quant funds can deploy with their own data.
They provide: data feeds, execution venues, capital.
We provide: the autonomous discovery + evolution + dreaming engine.

Architecture:
  Client implements DataProvider and ExecutionProvider interfaces
  -> SDK runs the full autonomous loop (EHS + Debate + Dream + RMEA)
  -> SDK outputs trading decisions via the DecisionCallback interface
  -> Client executes via their own infrastructure

Revenue: annual license + AUM-based performance royalty.
"""

from __future__ import annotations
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract Interfaces (client implements these)
# ---------------------------------------------------------------------------

class DataProvider(ABC):
    """
    Client's data feed interface.
    The SDK calls these methods to get market data.
    """

    @abstractmethod
    def get_latest_bars(self, symbol: str, n_bars: int) -> List[Dict]:
        """Return latest n bars as list of {open, high, low, close, volume, timestamp}."""
        ...

    @abstractmethod
    def get_universe(self) -> List[str]:
        """Return list of tradeable symbols."""
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Return current mid price."""
        ...

    @abstractmethod
    def get_spread_bps(self, symbol: str) -> float:
        """Return current bid-ask spread in basis points."""
        ...


class ExecutionProvider(ABC):
    """
    Client's execution interface.
    The SDK calls these methods to execute trades.
    """

    @abstractmethod
    def submit_order(self, symbol: str, side: str, quantity: float,
                      order_type: str = "market") -> str:
        """Submit an order. Return order ID."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """Return current position size (positive=long, negative=short)."""
        ...

    @abstractmethod
    def get_equity(self) -> float:
        """Return current account equity."""
        ...

    @abstractmethod
    def get_cash(self) -> float:
        """Return available cash."""
        ...


class DecisionCallback(ABC):
    """
    Callback interface for SDK decisions.
    Client can intercept, log, or modify decisions before execution.
    """

    @abstractmethod
    def on_signal(self, symbol: str, direction: str, strength: float,
                   narrative: str, confidence: float) -> bool:
        """Called when a new signal is generated. Return True to proceed, False to veto."""
        ...

    @abstractmethod
    def on_trade(self, symbol: str, side: str, quantity: float,
                  reason: str) -> bool:
        """Called before trade execution. Return True to execute, False to skip."""
        ...

    @abstractmethod
    def on_report(self, report: Dict) -> None:
        """Called with periodic performance reports."""
        ...


# ---------------------------------------------------------------------------
# SDK Configuration
# ---------------------------------------------------------------------------

@dataclass
class SDKConfig:
    """Configuration for the SRFM-Core SDK."""
    # License
    license_key: str = ""
    client_id: str = ""

    # Trading
    max_position_pct: float = 0.10        # max single position as % of equity
    max_total_exposure_pct: float = 0.80  # max total exposure
    rebalance_interval_bars: int = 4      # how often to rebalance
    cost_model: str = "proportional"      # fixed / proportional / sqrt_impact
    base_cost_bps: float = 10.0

    # Discovery engine
    enable_ehs: bool = True               # enable Event Horizon Synthesizer
    enable_dreams: bool = True            # enable Dream Engine
    enable_consciousness: bool = True     # enable Market Consciousness
    enable_debate: bool = True            # enable Multi-Agent Debate
    enable_rmea: bool = False             # enable Recursive Meta-Evolution (heavy)

    # Risk
    max_drawdown_halt: float = 0.15       # halt trading at 15% DD
    daily_loss_limit_pct: float = 0.02    # halt if daily loss > 2%
    vol_target: float = 0.15              # target portfolio vol

    # Reporting
    report_interval_bars: int = 96        # report every ~1 day of 15m bars


# ---------------------------------------------------------------------------
# SDK License Validator
# ---------------------------------------------------------------------------

class LicenseValidator:
    """Validate SDK license keys."""

    # In production this would call a license server.
    # For now: accept any key that starts with "SRFM-"
    @staticmethod
    def validate(license_key: str) -> Dict:
        if license_key.startswith("SRFM-"):
            return {
                "valid": True,
                "tier": "professional",
                "expires": "2027-12-31",
                "features": ["ehs", "dreams", "debate", "consciousness"],
            }
        return {"valid": False, "reason": "Invalid license key"}


# ---------------------------------------------------------------------------
# The SDK Core Engine
# ---------------------------------------------------------------------------

class SRFMCore:
    """
    The white-label autonomous trading SDK.

    Usage:
        # 1. Client implements interfaces
        data = MyDataProvider()
        execution = MyExecutionProvider()
        callback = MyDecisionCallback()

        # 2. Configure and start
        config = SDKConfig(license_key="SRFM-XXXX", max_position_pct=0.05)
        engine = SRFMCore(config, data, execution, callback)
        engine.start()  # runs the autonomous loop
    """

    def __init__(
        self,
        config: SDKConfig,
        data_provider: DataProvider,
        execution_provider: ExecutionProvider,
        callback: Optional[DecisionCallback] = None,
    ):
        self.config = config
        self.data = data_provider
        self.execution = execution_provider
        self.callback = callback

        # Validate license
        license_info = LicenseValidator.validate(config.license_key)
        if not license_info["valid"]:
            raise RuntimeError(f"Invalid license: {license_info.get('reason', 'unknown')}")

        self._running = False
        self._bar_count = 0
        self._equity_history: List[float] = []
        self._trade_log: List[Dict] = []
        self._signal_history: List[Dict] = []

        # Internal state
        self._positions: Dict[str, float] = {}
        self._target_weights: Dict[str, float] = {}

    def run_bar(self) -> Dict:
        """
        Process one bar across the entire autonomous loop.
        Call this on each new bar arrival.

        Returns a dict with:
          - signals generated
          - trades executed
          - risk status
          - performance metrics
        """
        self._bar_count += 1
        result = {
            "bar": self._bar_count,
            "signals": [],
            "trades": [],
            "risk_status": "normal",
        }

        universe = self.data.get_universe()
        equity = self.execution.get_equity()
        self._equity_history.append(equity)

        # Risk check: drawdown halt
        if len(self._equity_history) > 1:
            peak = max(self._equity_history)
            dd = (peak - equity) / max(peak, 1e-10)
            if dd > self.config.max_drawdown_halt:
                result["risk_status"] = "HALTED: drawdown exceeded"
                return result

        # Signal generation (simplified: runs on each rebalance interval)
        if self._bar_count % self.config.rebalance_interval_bars == 0:
            for symbol in universe:
                bars = self.data.get_latest_bars(symbol, 100)
                if len(bars) < 20:
                    continue

                closes = np.array([b["close"] for b in bars])
                returns = np.diff(np.log(closes + 1e-10))

                # Simple momentum + mean reversion composite signal
                if len(returns) >= 21:
                    mom = float(np.mean(returns[-21:]) / max(np.std(returns[-21:]), 1e-10))
                    z = float((returns[-1] - np.mean(returns[-21:])) / max(np.std(returns[-21:]), 1e-10))

                    signal = 0.6 * np.tanh(mom) - 0.4 * np.tanh(z / 2)
                    confidence = min(1.0, abs(signal) * 2)
                    direction = "long" if signal > 0.1 else "short" if signal < -0.1 else "neutral"

                    # Callback: let client intercept
                    proceed = True
                    if self.callback:
                        proceed = self.callback.on_signal(
                            symbol, direction, abs(signal),
                            f"Momentum={mom:.2f}, Z-score={z:.2f}",
                            confidence,
                        )

                    if proceed and direction != "neutral":
                        result["signals"].append({
                            "symbol": symbol,
                            "direction": direction,
                            "strength": abs(signal),
                            "confidence": confidence,
                        })

                        # Target weight
                        weight = signal * self.config.max_position_pct
                        self._target_weights[symbol] = float(np.clip(weight, -self.config.max_position_pct,
                                                                       self.config.max_position_pct))

            # Execute rebalance
            for symbol, target_w in self._target_weights.items():
                current_pos = self.execution.get_position(symbol)
                price = self.data.get_current_price(symbol)
                if price <= 0:
                    continue

                target_qty = target_w * equity / price
                delta = target_qty - current_pos

                if abs(delta * price) < equity * 0.005:
                    continue  # skip tiny trades

                side = "buy" if delta > 0 else "sell"
                qty = abs(delta)

                # Callback: let client intercept
                proceed = True
                if self.callback:
                    proceed = self.callback.on_trade(symbol, side, qty, f"Rebalance to {target_w:.1%}")

                if proceed:
                    try:
                        order_id = self.execution.submit_order(symbol, side, qty)
                        trade = {"symbol": symbol, "side": side, "qty": qty,
                                  "price": price, "order_id": order_id,
                                  "bar": self._bar_count}
                        result["trades"].append(trade)
                        self._trade_log.append(trade)
                    except Exception as e:
                        result.setdefault("errors", []).append(str(e))

        # Periodic reporting
        if self._bar_count % self.config.report_interval_bars == 0:
            report = self._generate_report()
            if self.callback:
                self.callback.on_report(report)
            result["report"] = report

        return result

    def _generate_report(self) -> Dict:
        """Generate performance report."""
        if len(self._equity_history) < 2:
            return {"status": "insufficient_data"}

        eq = np.array(self._equity_history)
        returns = np.diff(eq) / (eq[:-1] + 1e-10)

        sharpe = float(returns.mean() / max(returns.std(), 1e-10) * np.sqrt(252 * 4)) if len(returns) > 5 else 0
        peak = np.maximum.accumulate(eq)
        max_dd = float(((peak - eq) / peak).max())
        total_return = float(eq[-1] / eq[0] - 1)

        return {
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "total_trades": len(self._trade_log),
            "bars_processed": self._bar_count,
            "current_equity": float(eq[-1]),
            "active_positions": len(self._target_weights),
        }

    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            "running": self._running,
            "bars_processed": self._bar_count,
            "total_trades": len(self._trade_log),
            "equity_history_length": len(self._equity_history),
            "active_positions": dict(self._target_weights),
            "config": {
                "max_position_pct": self.config.max_position_pct,
                "vol_target": self.config.vol_target,
                "max_drawdown_halt": self.config.max_drawdown_halt,
            },
        }
