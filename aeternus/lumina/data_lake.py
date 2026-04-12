

# ============================================================
# Extended Data Lake Components - Part 2
# ============================================================

import os
import json
import time
import hashlib
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass, field
from collections import defaultdict, deque
import torch
import numpy as np


@dataclass
class DataSchemaField:
    """Schema field definition for data validation."""
    name: str
    dtype: str  # float32 | int64 | bool | string | datetime
    nullable: bool = False
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    enum_values: Optional[List[Any]] = None
    description: str = ""

    def validate(self, value: Any) -> Tuple[bool, str]:
        """Validate a value against this field schema."""
        if value is None:
            if self.nullable:
                return True, ""
            return False, f"Field '{self.name}' is not nullable but got None"

        if self.dtype in ("float32", "float64"):
            try:
                v = float(value)
                if self.min_val is not None and v < self.min_val:
                    return False, f"Value {v} < min {self.min_val} for field '{self.name}'"
                if self.max_val is not None and v > self.max_val:
                    return False, f"Value {v} > max {self.max_val} for field '{self.name}'"
            except (TypeError, ValueError):
                return False, f"Cannot convert '{value}' to float for field '{self.name}'"

        if self.enum_values is not None and value not in self.enum_values:
            return False, f"Value '{value}' not in enum {self.enum_values} for field '{self.name}'"

        return True, ""


class DataSchema:
    """Data schema for structured financial data validation."""

    def __init__(self, name: str, fields: List[DataSchemaField], version: str = "1.0"):
        self.name = name
        self.version = version
        self.fields = {f.name: f for f in fields}

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single record against the schema."""
        errors = []

        # Check required fields
        for field_name, field_def in self.fields.items():
            if not field_def.nullable and field_name not in record:
                errors.append(f"Missing required field '{field_name}'")

        # Validate present fields
        for field_name, value in record.items():
            if field_name in self.fields:
                valid, msg = self.fields[field_name].validate(value)
                if not valid:
                    errors.append(msg)

        return len(errors) == 0, errors

    def validate_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a batch of records."""
        total = len(records)
        valid_count = 0
        all_errors = []

        for i, record in enumerate(records):
            ok, errors = self.validate_record(record)
            if ok:
                valid_count += 1
            else:
                all_errors.append({"index": i, "errors": errors})

        return {
            "total": total,
            "valid": valid_count,
            "invalid": total - valid_count,
            "validity_rate": valid_count / max(total, 1),
            "errors": all_errors[:10],  # first 10 errors
        }

    def infer_from_sample(self, records: List[Dict[str, Any]]) -> "DataSchema":
        """Infer schema from sample records."""
        field_stats = defaultdict(lambda: {"count": 0, "min": float("inf"), "max": float("-inf"), "nulls": 0})

        for record in records:
            for k, v in record.items():
                stats = field_stats[k]
                stats["count"] += 1
                if v is None:
                    stats["nulls"] += 1
                else:
                    try:
                        fv = float(v)
                        stats["min"] = min(stats["min"], fv)
                        stats["max"] = max(stats["max"], fv)
                    except (TypeError, ValueError):
                        pass

        inferred_fields = []
        for name, stats in field_stats.items():
            nullable = stats["nulls"] > 0
            min_v = stats["min"] if stats["min"] != float("inf") else None
            max_v = stats["max"] if stats["max"] != float("-inf") else None
            inferred_fields.append(DataSchemaField(name, "float32", nullable, min_v, max_v))

        return DataSchema(self.name + "_inferred", inferred_fields, "1.0")


class StreamingDataPipeline:
    """Streaming data pipeline with backpressure and windowing."""

    def __init__(
        self,
        source_fn: Any,  # callable that yields records
        buffer_size: int = 1000,
        batch_size: int = 32,
        max_latency_ms: float = 100.0,
    ):
        self.source_fn = source_fn
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_latency_ms = max_latency_ms
        self._buffer: deque = deque(maxlen=buffer_size)
        self._transformations: List[Any] = []
        self._filters: List[Any] = []
        self._stats = {"records_in": 0, "records_out": 0, "records_filtered": 0}

    def add_transformation(self, fn: Any) -> "StreamingDataPipeline":
        """Add a transformation function to the pipeline."""
        self._transformations.append(fn)
        return self

    def add_filter(self, predicate: Any) -> "StreamingDataPipeline":
        """Add a filter predicate to the pipeline."""
        self._filters.append(predicate)
        return self

    def process_record(self, record: Any) -> Optional[Any]:
        """Process a single record through transformations and filters."""
        self._stats["records_in"] += 1

        # Apply filters
        for f in self._filters:
            try:
                if not f(record):
                    self._stats["records_filtered"] += 1
                    return None
            except Exception:
                self._stats["records_filtered"] += 1
                return None

        # Apply transformations
        result = record
        for t in self._transformations:
            try:
                result = t(result)
            except Exception:
                return None

        self._stats["records_out"] += 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "filter_rate": self._stats["records_filtered"] / max(self._stats["records_in"], 1),
            "buffer_size": len(self._buffer),
        }


class FinancialCalendar:
    """Financial calendar utility for date/trading day calculations."""

    WEEKEND_DAYS = {5, 6}  # Saturday=5, Sunday=6

    # Common US market holidays (month, day) - simplified
    US_HOLIDAYS = {
        (1, 1),   # New Year's Day
        (7, 4),   # Independence Day
        (12, 25), # Christmas
        (11, 11), # Veterans Day
    }

    def __init__(self, market: str = "US"):
        self.market = market
        self._custom_holidays: List[Tuple[int, int]] = []

    def is_trading_day(self, year: int, month: int, day: int) -> bool:
        """Check if a date is a trading day."""
        import datetime
        try:
            dt = datetime.date(year, month, day)
        except ValueError:
            return False
        if dt.weekday() in self.WEEKEND_DAYS:
            return False
        if (month, day) in self.US_HOLIDAYS:
            return False
        if (month, day) in self._custom_holidays:
            return False
        return True

    def add_holiday(self, month: int, day: int):
        self._custom_holidays.append((month, day))

    def trading_days_between(self, start: "datetime.date", end: "datetime.date") -> int:
        """Count trading days between two dates (inclusive)."""
        import datetime
        count = 0
        current = start
        while current <= end:
            if self.is_trading_day(current.year, current.month, current.day):
                count += 1
            current += datetime.timedelta(days=1)
        return count


class MarketMicrostructureProcessor:
    """Processes raw tick data to extract microstructure features."""

    def __init__(
        self,
        tick_size: float = 0.01,
        lot_size: int = 100,
    ):
        self.tick_size = tick_size
        self.lot_size = lot_size
        self._bid_ask_history: deque = deque(maxlen=1000)

    def process_quote(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        timestamp: float,
    ) -> Dict[str, float]:
        """Extract microstructure features from a quote update."""
        mid = (bid + ask) / 2
        spread = ask - bid
        spread_bps = spread / mid * 10000 if mid > 0 else 0
        imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-8)

        self._bid_ask_history.append({
            "mid": mid,
            "spread": spread,
            "imbalance": imbalance,
            "timestamp": timestamp,
        })

        # VWAP mid
        if len(self._bid_ask_history) >= 2:
            mids = [h["mid"] for h in self._bid_ask_history]
            vwap_mid = sum(mids) / len(mids)
            spread_ma = sum(h["spread"] for h in self._bid_ask_history) / len(self._bid_ask_history)
        else:
            vwap_mid = mid
            spread_ma = spread

        return {
            "mid": mid,
            "spread": spread,
            "spread_bps": spread_bps,
            "imbalance": imbalance,
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "vwap_mid": vwap_mid,
            "spread_ma": spread_ma,
        }

    def kyle_lambda(self, trades: List[Dict[str, float]]) -> float:
        """Estimate Kyle's lambda (price impact) from trades."""
        if len(trades) < 2:
            return 0.0
        price_changes = []
        signed_volumes = []
        for i in range(1, len(trades)):
            dp = trades[i]["price"] - trades[i-1]["price"]
            sv = trades[i]["size"] * (1 if trades[i].get("side", 1) > 0 else -1)
            price_changes.append(dp)
            signed_volumes.append(sv)

        if not signed_volumes or sum(v**2 for v in signed_volumes) < 1e-10:
            return 0.0

        # OLS: dp = lambda * sv
        num = sum(dp * sv for dp, sv in zip(price_changes, signed_volumes))
        den = sum(sv ** 2 for sv in signed_volumes)
        return num / den

    def amihud_illiquidity(self, prices: List[float], volumes: List[float]) -> float:
        """Amihud illiquidity ratio."""
        if len(prices) < 2:
            return 0.0
        ratios = []
        for i in range(1, len(prices)):
            if volumes[i] > 0 and prices[i-1] > 0:
                ret = abs(prices[i] - prices[i-1]) / prices[i-1]
                ratios.append(ret / volumes[i])
        return sum(ratios) / len(ratios) if ratios else 0.0


class SyntheticDataGenerator:
    """Generates synthetic financial data for testing and simulation."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    def generate_gbm(
        self,
        n_periods: int,
        dt: float = 1 / 252,
        mu: float = 0.1,
        sigma: float = 0.2,
        S0: float = 100.0,
    ) -> np.ndarray:
        """Geometric Brownian Motion price series."""
        Z = self._rng.randn(n_periods)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
        prices = S0 * np.exp(np.cumsum(log_returns))
        return np.concatenate([[S0], prices])

    def generate_heston(
        self,
        n_periods: int,
        dt: float = 1 / 252,
        mu: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        S0: float = 100.0,
        V0: float = 0.04,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Heston stochastic volatility model."""
        prices = [S0]
        variances = [V0]

        for _ in range(n_periods):
            V = max(variances[-1], 0)
            Z1 = self._rng.randn()
            Z2 = rho * Z1 + math.sqrt(1 - rho**2) * self._rng.randn()

            dV = kappa * (theta - V) * dt + xi * math.sqrt(V * dt) * Z2
            new_V = max(V + dV, 0)

            dS = mu * prices[-1] * dt + math.sqrt(V * dt) * prices[-1] * Z1
            new_S = prices[-1] + dS

            prices.append(new_S)
            variances.append(new_V)

        return np.array(prices), np.array(variances)

    def generate_order_flow(
        self,
        n_periods: int,
        lambda_buy: float = 5.0,
        lambda_sell: float = 5.0,
        price_impact: float = 0.01,
        S0: float = 100.0,
    ) -> Dict[str, np.ndarray]:
        """Generate synthetic order flow with price impact."""
        prices = [S0]
        buy_volumes = []
        sell_volumes = []

        for _ in range(n_periods):
            buy_vol = self._rng.exponential(1 / lambda_buy)
            sell_vol = self._rng.exponential(1 / lambda_sell)
            net_flow = buy_vol - sell_vol

            new_price = prices[-1] * (1 + price_impact * net_flow)
            prices.append(new_price)
            buy_volumes.append(buy_vol)
            sell_volumes.append(sell_vol)

        return {
            "prices": np.array(prices),
            "buy_volumes": np.array(buy_volumes),
            "sell_volumes": np.array(sell_volumes),
            "net_flows": np.array(buy_volumes) - np.array(sell_volumes),
        }

    def generate_multivariate_returns(
        self,
        n_periods: int,
        n_assets: int,
        factor_loadings: Optional[np.ndarray] = None,
        n_factors: int = 3,
        idio_vol: float = 0.01,
    ) -> np.ndarray:
        """Generate multi-asset returns with factor structure."""
        if factor_loadings is None:
            factor_loadings = self._rng.randn(n_assets, n_factors) * 0.1

        # Factor returns
        factor_returns = self._rng.randn(n_periods, n_factors) * 0.01

        # Systematic returns
        systematic = factor_returns @ factor_loadings.T  # (T, N)

        # Idiosyncratic returns
        idio = self._rng.randn(n_periods, n_assets) * idio_vol

        return systematic + idio

    def generate_regime_switching(
        self,
        n_periods: int,
        regimes: Optional[Dict[str, Dict[str, float]]] = None,
        transition_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regime-switching returns.

        Returns: (returns, regime_labels)
        """
        if regimes is None:
            regimes = {
                "bull": {"mu": 0.001, "sigma": 0.01},
                "bear": {"mu": -0.001, "sigma": 0.02},
                "volatile": {"mu": 0.0, "sigma": 0.03},
            }

        regime_names = list(regimes.keys())
        n_regimes = len(regime_names)

        if transition_matrix is None:
            # Default: high persistence, low transition
            transition_matrix = np.eye(n_regimes) * 0.95 + np.ones((n_regimes, n_regimes)) * 0.05 / n_regimes
            transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

        current_regime = 0
        regime_labels = []
        returns = []

        for _ in range(n_periods):
            regime_labels.append(current_regime)
            r_params = regimes[regime_names[current_regime]]
            ret = self._rng.randn() * r_params["sigma"] + r_params["mu"]
            returns.append(ret)
            # Transition
            current_regime = self._rng.choice(n_regimes, p=transition_matrix[current_regime])

        return np.array(returns), np.array(regime_labels)
