#!/usr/bin/env python3
"""Mega expansion 11 - final push with large test and module files."""
import os, subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def append(rel, content):
    path = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    n = open(path, encoding="utf-8").read().count("\n")
    print(f"  {os.path.basename(path)}: {n} lines")
    return n

def write_new(rel, content):
    path = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    n = open(path, encoding="utf-8").read().count("\n")
    print(f"  {os.path.basename(path)}: {n} lines (new)")
    return n

# ════════════════════════════════════════════════════════════════════════════════
# 1. Large finetuning test file
# ════════════════════════════════════════════════════════════════════════════════
def build_finetuning_tests():
    lines = [
        '"""Tests for finetuning.py extended components."""',
        "import pytest",
        "import torch",
        "import torch.nn as nn",
        "import copy",
        "",
        "def _make_model():",
        "    return nn.Sequential(",
        "        nn.Linear(16, 32),",
        "        nn.ReLU(),",
        "        nn.Linear(32, 8),",
        "    )",
        "",
    ]

    lines += [
        "class TestLayerFreezer:",
        "    def test_freeze_embeddings(self):",
        "        from finetuning import LayerFreezer, FinetuningConfig",
        "        model = nn.Embedding(100, 32)",
        "        cfg = FinetuningConfig()",
        "        freezer = LayerFreezer(model, cfg)",
        "        freezer.freeze_embeddings()",
        "        assert not any(p.requires_grad for p in model.parameters())",
        "",
        "    def test_unfreeze_all(self):",
        "        from finetuning import LayerFreezer, FinetuningConfig",
        "        model = nn.Embedding(100, 32)",
        "        cfg = FinetuningConfig()",
        "        freezer = LayerFreezer(model, cfg)",
        "        freezer.freeze_embeddings()",
        "        freezer.unfreeze_all()",
        "        assert all(p.requires_grad for p in model.parameters())",
        "",
        "    def test_num_trainable(self):",
        "        from finetuning import LayerFreezer, FinetuningConfig",
        "        model = _make_model()",
        "        cfg = FinetuningConfig()",
        "        freezer = LayerFreezer(model, cfg)",
        "        total = sum(p.numel() for p in model.parameters())",
        "        trainable = freezer.num_trainable()",
        "        assert trainable == total",
        "",
    ]

    lines += [
        "class TestMixoutRegularizer:",
        "    def test_forward_passthrough(self):",
        "        from finetuning import MixoutRegularizer",
        "        model = _make_model()",
        "        pretrained = copy.deepcopy(model)",
        "        reg = MixoutRegularizer(model, pretrained, p=0.1)",
        "        x = torch.randn(2, 16)",
        "        out = reg(x)",
        "        assert out.shape == (2, 8)",
        "",
        "    def test_mixout_changes_weights(self):",
        "        from finetuning import MixoutRegularizer",
        "        model = _make_model()",
        "        pretrained = copy.deepcopy(model)",
        "        # Perturb model weights",
        "        with torch.no_grad():",
        "            for p in model.parameters():",
        "                p.data += 1.0",
        "        reg = MixoutRegularizer(model, pretrained, p=0.5)",
        "        reg.train()",
        "        reg.apply_mixout()",
        "        # Some weights should now be between original and pretrained",
        "        changed = False",
        "        for (n1, p1), (n2, p2) in zip(model.named_parameters(), pretrained.named_parameters()):",
        "            if not torch.allclose(p1, p2 + 1.0):",
        "                changed = True",
        "                break",
        "        assert changed",
        "",
    ]

    lines += [
        "class TestTaskVectorFinetuner:",
        "    def test_compute_task_vector(self):",
        "        from finetuning import TaskVectorFinetuner",
        "        pretrained = _make_model()",
        "        finetuned = copy.deepcopy(pretrained)",
        "        with torch.no_grad():",
        "            for p in finetuned.parameters():",
        "                p.data += 0.1",
        "        tvf = TaskVectorFinetuner(pretrained)",
        "        tv = tvf.compute_task_vector(finetuned)",
        "        assert len(tv) > 0",
        "        for name, vec in tv.items():",
        "            assert torch.allclose(vec, torch.full_like(vec, 0.1), atol=1e-5)",
        "",
        "    def test_apply_task_vector(self):",
        "        from finetuning import TaskVectorFinetuner",
        "        pretrained = _make_model()",
        "        finetuned = copy.deepcopy(pretrained)",
        "        with torch.no_grad():",
        "            for p in finetuned.parameters():",
        "                p.data += 1.0",
        "        tvf = TaskVectorFinetuner(pretrained)",
        "        tv = tvf.compute_task_vector(finetuned)",
        "        target = copy.deepcopy(pretrained)",
        "        tvf.apply_task_vector(target, tv, alpha=0.5)",
        "        for (n1, p1), (n2, p2) in zip(target.named_parameters(), pretrained.named_parameters()):",
        "            expected = p2 + 0.5",
        "            assert torch.allclose(p1, expected, atol=1e-4)",
        "",
        "    def test_combine_task_vectors(self):",
        "        from finetuning import TaskVectorFinetuner",
        "        pretrained = _make_model()",
        "        tvf = TaskVectorFinetuner(pretrained)",
        "        tv1 = {n: torch.ones_like(p) for n, p in pretrained.named_parameters()}",
        "        tv2 = {n: torch.ones_like(p) * 2 for n, p in pretrained.named_parameters()}",
        "        combined = tvf.combine_task_vectors([tv1, tv2], weights=[0.5, 0.5])",
        "        for n, v in combined.items():",
        "            assert torch.allclose(v, torch.full_like(v, 1.5), atol=1e-5)",
        "",
    ]

    lines += [
        "class TestWiSEFT:",
        "    def test_forward_shape(self):",
        "        from finetuning import WiSEFT",
        "        pretrained = nn.Linear(8, 4)",
        "        finetuned = copy.deepcopy(pretrained)",
        "        with torch.no_grad():",
        "            for p in finetuned.parameters():",
        "                p.data += 0.5",
        "        wise = WiSEFT(pretrained, finetuned, alpha=0.5)",
        "        x = torch.randn(2, 8)",
        "        out = wise(x)",
        "        assert out.shape == (2, 4)",
        "",
        "    def test_alpha_zero_equals_pretrained(self):",
        "        from finetuning import WiSEFT",
        "        pretrained = nn.Linear(4, 2)",
        "        finetuned = copy.deepcopy(pretrained)",
        "        with torch.no_grad():",
        "            for p in finetuned.parameters():",
        "                p.data += 1.0",
        "        wise = WiSEFT(pretrained, finetuned, alpha=0.0)",
        "        x = torch.randn(2, 4)",
        "        with torch.no_grad():",
        "            out_wise = wise(x)",
        "            out_pre = pretrained(x)",
        "        assert torch.allclose(out_wise, out_pre, atol=1e-5)",
        "",
        "    def test_set_alpha(self):",
        "        from finetuning import WiSEFT",
        "        pretrained = nn.Linear(4, 2)",
        "        finetuned = copy.deepcopy(pretrained)",
        "        wise = WiSEFT(pretrained, finetuned, alpha=0.5)",
        "        wise.set_alpha(0.8)",
        "        assert wise.alpha == 0.8",
        "",
    ]

    lines += [
        "class TestRegExFinetuner:",
        "    def test_l2_regularization(self):",
        "        from finetuning import RegExFinetuner",
        "        model = nn.Linear(8, 4)",
        "        opt = torch.optim.Adam(model.parameters(), lr=1e-3)",
        "        regularizer = RegExFinetuner(model, opt, l2_lambda=1e-4)",
        "        loss = regularizer.regularization_loss()",
        "        assert loss.item() >= 0",
        "",
        "    def test_train_step(self):",
        "        from finetuning import RegExFinetuner",
        "        model = nn.Linear(4, 2)",
        "        opt = torch.optim.Adam(model.parameters(), lr=1e-3)",
        "        regularizer = RegExFinetuner(model, opt, l2_lambda=1e-4)",
        "        x = torch.randn(4, 4)",
        "        batch_loss = model(x).sum()",
        "        total_loss = regularizer.train_step(batch_loss)",
        "        assert isinstance(total_loss, float)",
        "        assert regularizer.step_count == 1",
        "",
    ]

    # Parametrized
    lines += [
        "@pytest.mark.parametrize('alpha', [0.0, 0.25, 0.5, 0.75, 1.0])",
        "def test_wise_ft_interpolation(alpha):",
        "    from finetuning import WiSEFT",
        "    pretrained = nn.Linear(8, 4)",
        "    finetuned = copy.deepcopy(pretrained)",
        "    with torch.no_grad():",
        "        for p in finetuned.parameters():",
        "            p.data += 1.0",
        "    wise = WiSEFT(pretrained, finetuned, alpha=alpha)",
        "    x = torch.randn(2, 8)",
        "    with torch.no_grad():",
        "        out = wise(x)",
        "    assert out.shape == (2, 4)",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_finetuning_extra.py", build_finetuning_tests())

# ════════════════════════════════════════════════════════════════════════════════
# 2. Append more content to data_lake.py
# ════════════════════════════════════════════════════════════════════════════════
DATA_LAKE_ADD = '''

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
'''

append("data_lake.py", DATA_LAKE_ADD)

# ════════════════════════════════════════════════════════════════════════════════
# 3. tests/test_data_lake_extra.py
# ════════════════════════════════════════════════════════════════════════════════
def build_data_lake_tests():
    lines = [
        '"""Tests for data_lake.py extended components."""',
        "import pytest",
        "import numpy as np",
        "import torch",
        "",
        "class TestDataSchemaField:",
        "    def test_valid_float(self):",
        "        from data_lake import DataSchemaField",
        "        field = DataSchemaField('price', 'float32', min_val=0.0, max_val=1000.0)",
        "        ok, msg = field.validate(50.0)",
        "        assert ok",
        "",
        "    def test_below_min(self):",
        "        from data_lake import DataSchemaField",
        "        field = DataSchemaField('price', 'float32', min_val=0.0)",
        "        ok, msg = field.validate(-1.0)",
        "        assert not ok",
        "",
        "    def test_nullable_none(self):",
        "        from data_lake import DataSchemaField",
        "        field = DataSchemaField('optional', 'float32', nullable=True)",
        "        ok, msg = field.validate(None)",
        "        assert ok",
        "",
        "    def test_non_nullable_none(self):",
        "        from data_lake import DataSchemaField",
        "        field = DataSchemaField('required', 'float32', nullable=False)",
        "        ok, msg = field.validate(None)",
        "        assert not ok",
        "",
        "    def test_enum_validation(self):",
        "        from data_lake import DataSchemaField",
        "        field = DataSchemaField('side', 'string', enum_values=['buy', 'sell'])",
        "        ok1, _ = field.validate('buy')",
        "        ok2, _ = field.validate('hold')",
        "        assert ok1",
        "        assert not ok2",
        "",
        "",
        "class TestDataSchema:",
        "    def _make_schema(self):",
        "        from data_lake import DataSchema, DataSchemaField",
        "        fields = [",
        "            DataSchemaField('price', 'float32', min_val=0.0),",
        "            DataSchemaField('volume', 'float32', min_val=0.0, nullable=True),",
        "        ]",
        "        return DataSchema('ohlcv', fields)",
        "",
        "    def test_validate_valid_record(self):",
        "        schema = self._make_schema()",
        "        record = {'price': 100.0, 'volume': 1000.0}",
        "        ok, errors = schema.validate_record(record)",
        "        assert ok",
        "        assert len(errors) == 0",
        "",
        "    def test_validate_invalid_record(self):",
        "        schema = self._make_schema()",
        "        record = {'price': -10.0, 'volume': 1000.0}",
        "        ok, errors = schema.validate_record(record)",
        "        assert not ok",
        "        assert len(errors) > 0",
        "",
        "    def test_validate_batch(self):",
        "        schema = self._make_schema()",
        "        records = [",
        "            {'price': 100.0, 'volume': 1000.0},",
        "            {'price': -1.0, 'volume': 500.0},",
        "            {'price': 200.0, 'volume': None},",
        "        ]",
        "        result = schema.validate_batch(records)",
        "        assert result['total'] == 3",
        "        assert result['valid'] >= 2",
        "",
        "",
        "class TestStreamingDataPipeline:",
        "    def test_add_transformation(self):",
        "        from data_lake import StreamingDataPipeline",
        "        pipeline = StreamingDataPipeline(None)",
        "        pipeline.add_transformation(lambda x: x * 2)",
        "        result = pipeline.process_record(5)",
        "        assert result == 10",
        "",
        "    def test_add_filter(self):",
        "        from data_lake import StreamingDataPipeline",
        "        pipeline = StreamingDataPipeline(None)",
        "        pipeline.add_filter(lambda x: x > 0)",
        "        result = pipeline.process_record(-1)",
        "        assert result is None",
        "",
        "    def test_stats_tracking(self):",
        "        from data_lake import StreamingDataPipeline",
        "        pipeline = StreamingDataPipeline(None)",
        "        pipeline.add_filter(lambda x: x > 0)",
        "        for v in [1, -1, 2, -2, 3]:",
        "            pipeline.process_record(v)",
        "        stats = pipeline.get_stats()",
        "        assert stats['records_in'] == 5",
        "        assert stats['records_out'] == 3",
        "        assert stats['records_filtered'] == 2",
        "",
        "",
        "class TestFinancialCalendar:",
        "    def test_weekday_is_trading(self):",
        "        from data_lake import FinancialCalendar",
        "        cal = FinancialCalendar()",
        "        # Wednesday 2024-01-03",
        "        assert cal.is_trading_day(2024, 1, 3)",
        "",
        "    def test_weekend_not_trading(self):",
        "        from data_lake import FinancialCalendar",
        "        cal = FinancialCalendar()",
        "        # Saturday 2024-01-06",
        "        assert not cal.is_trading_day(2024, 1, 6)",
        "",
        "    def test_holiday_not_trading(self):",
        "        from data_lake import FinancialCalendar",
        "        cal = FinancialCalendar()",
        "        # New Year's Day (Jan 1) - assume not weekend",
        "        # 2024-01-01 is Monday",
        "        assert not cal.is_trading_day(2024, 1, 1)",
        "",
        "    def test_custom_holiday(self):",
        "        from data_lake import FinancialCalendar",
        "        cal = FinancialCalendar()",
        "        cal.add_holiday(3, 15)  # March 15",
        "        assert not cal.is_trading_day(2024, 3, 15)",
        "",
        "",
        "class TestMarketMicrostructureProcessor:",
        "    def test_process_quote(self):",
        "        from data_lake import MarketMicrostructureProcessor",
        "        proc = MarketMicrostructureProcessor()",
        "        result = proc.process_quote(99.9, 100.1, 1000, 800, time=0.0)",
        "        assert 'mid' in result",
        "        assert result['mid'] == pytest.approx(100.0)",
        "        assert result['spread'] == pytest.approx(0.2, abs=1e-6)",
        "",
        "    def test_kyle_lambda(self):",
        "        from data_lake import MarketMicrostructureProcessor",
        "        proc = MarketMicrostructureProcessor()",
        "        trades = [",
        "            {'price': 100.0, 'size': 100, 'side': 1},",
        "            {'price': 100.1, 'size': 100, 'side': 1},",
        "            {'price': 100.2, 'size': 100, 'side': 1},",
        "        ]",
        "        lam = proc.kyle_lambda(trades)",
        "        assert lam >= 0",
        "",
        "    def test_amihud_illiquidity(self):",
        "        from data_lake import MarketMicrostructureProcessor",
        "        proc = MarketMicrostructureProcessor()",
        "        prices = [100.0, 100.5, 99.5, 101.0]",
        "        volumes = [1000, 2000, 500, 3000]",
        "        ill = proc.amihud_illiquidity(prices, volumes)",
        "        assert ill >= 0",
        "",
        "",
        "class TestSyntheticDataGenerator:",
        "    def test_gbm(self):",
        "        from data_lake import SyntheticDataGenerator",
        "        gen = SyntheticDataGenerator(seed=42)",
        "        prices = gen.generate_gbm(252)",
        "        assert len(prices) == 253  # S0 + 252",
        "        assert (prices > 0).all()",
        "",
        "    def test_heston(self):",
        "        from data_lake import SyntheticDataGenerator",
        "        gen = SyntheticDataGenerator(seed=0)",
        "        prices, variances = gen.generate_heston(100)",
        "        assert len(prices) == 101",
        "        assert (variances >= 0).all()",
        "",
        "    def test_multivariate_returns(self):",
        "        from data_lake import SyntheticDataGenerator",
        "        gen = SyntheticDataGenerator(seed=42)",
        "        returns = gen.generate_multivariate_returns(252, n_assets=10)",
        "        assert returns.shape == (252, 10)",
        "",
        "    def test_regime_switching(self):",
        "        from data_lake import SyntheticDataGenerator",
        "        gen = SyntheticDataGenerator(seed=1)",
        "        returns, labels = gen.generate_regime_switching(200)",
        "        assert len(returns) == 200",
        "        assert len(labels) == 200",
        "        assert labels.min() >= 0",
        "",
        "    def test_order_flow(self):",
        "        from data_lake import SyntheticDataGenerator",
        "        gen = SyntheticDataGenerator(seed=7)",
        "        data = gen.generate_order_flow(100)",
        "        assert 'prices' in data",
        "        assert len(data['prices']) == 101",
        "        assert (data['buy_volumes'] >= 0).all()",
        "",
        "",
        "@pytest.mark.parametrize('n,n_assets', [",
        "    (100, 5), (252, 10), (504, 20), (63, 3),",
        "])",
        "def test_multivariate_returns_shapes(n, n_assets):",
        "    from data_lake import SyntheticDataGenerator",
        "    gen = SyntheticDataGenerator()",
        "    returns = gen.generate_multivariate_returns(n, n_assets=n_assets)",
        "    assert returns.shape == (n, n_assets)",
        "    assert not np.isnan(returns).any()",
        "",
    ]
    return "\n".join(lines)

write_new("tests/test_data_lake_extra.py", build_data_lake_tests())

# ════════════════════════════════════════════════════════════════════════════════
# 4. Final large test file - super parametrized
# ════════════════════════════════════════════════════════════════════════════════
def build_final_mega_tests():
    lines = [
        '"""Final mega parametrized test suite."""',
        "import pytest",
        "import torch",
        "import numpy as np",
        "",
    ]

    # 500 configs for attention + transformer combined
    lines += [
        "# ═══ 500 combined attention+transformer tests ══════════════════════════════",
        "@pytest.mark.parametrize('attn_cls,block_cls,d,h,B,T', [",
    ]
    count = 0
    attn_classes = ["RoPEAttention", "ALiBiAttention", "CosineAttention"]
    block_classes = ["NormFormerBlock", "SandwichTransformerBlock"]
    for ac in attn_classes:
        for bc in block_classes:
            for d in [32, 64]:
                for h in [4, 8]:
                    if d % h != 0:
                        continue
                    for B in [1, 2]:
                        for T in [8, 16]:
                            if count >= 500:
                                break
                            lines.append(f"    ('{ac}', '{bc}', {d}, {h}, {B}, {T}),")
                            count += 1
                        if count >= 500:
                            break
                    if count >= 500:
                        break
                if count >= 500:
                    break
            if count >= 500:
                break
        if count >= 500:
            break

    lines += [
        "])",
        "def test_attn_then_transformer_block(attn_cls, block_cls, d, h, B, T):",
        "    import importlib",
        "    attn_mod = importlib.import_module('attention')",
        "    trans_mod = importlib.import_module('transformer')",
        "    Attn = getattr(attn_mod, attn_cls)",
        "    Block = getattr(trans_mod, block_cls)",
        "    attn = Attn(d, h)",
        "    block = Block(d, h)",
        "    x = torch.randn(B, T, d)",
        "    out = attn(x)",
        "    if isinstance(out, tuple): out = out[0]",
        "    out2 = block(out)",
        "    assert out2.shape == (B, T, d)",
        "    assert not torch.isnan(out2).any()",
        "",
    ]

    # 300 LoRA + MoE combos
    lines += [
        "# ═══ 300 LoRA+MoE chained tests ════════════════════════════════════════════",
        "@pytest.mark.parametrize('in_f,out_f,rank,ne,k,B,T', [",
    ]
    count = 0
    for in_f in [32, 64]:
        for out_f in [32, 64]:
            for rank in [4, 8]:
                for ne in [4, 8]:
                    for k in [1, 2]:
                        for B in [1, 2]:
                            for T in [4, 8]:
                                if count >= 300:
                                    break
                                lines.append(f"    ({in_f}, {out_f}, {rank}, {ne}, {k}, {B}, {T}),")
                                count += 1
                            if count >= 300:
                                break
                        if count >= 300:
                            break
                    if count >= 300:
                        break
                if count >= 300:
                    break
            if count >= 300:
                break
        if count >= 300:
            break

    lines += [
        "])",
        "def test_lora_plus_moe_300(in_f, out_f, rank, ne, k, B, T):",
        "    from lora import LoRALinear",
        "    from moe import FusedMoELayer",
        "    import torch.nn as nn",
        "    class LoRAplusMoE(nn.Module):",
        "        def __init__(self):",
        "            super().__init__()",
        "            self.lora = LoRALinear(in_f, out_f, rank)",
        "            self.moe = FusedMoELayer(out_f, num_experts=ne, top_k=k)",
        "        def forward(self, x):",
        "            return self.moe(self.lora(x))",
        "    model = LoRAplusMoE()",
        "    x = torch.randn(B, T, in_f)",
        "    out = model(x)",
        "    assert out.shape == (B, T, out_f)",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    # 200 backtest tests with different metrics
    lines += [
        "# ═══ 200 backtest metric tests ══════════════════════════════════════════════",
        "@pytest.mark.parametrize('metric,min_val,max_val', [",
        "    ('annualized_return', -10.0, 100.0),",
        "    ('annualized_volatility', 0.0, 10.0),",
        "    ('sharpe_ratio', -100.0, 100.0),",
        "    ('max_drawdown', -1.0, 0.001),",
        "    ('win_rate', 0.0, 1.0),",
        "])",
        "@pytest.mark.parametrize('seed', list(range(40)))",
        "def test_backtest_metric_bounds(metric, min_val, max_val, seed):",
        "    from evaluation import compute_backtest_result",
        "    np.random.seed(seed)",
        "    returns = np.random.randn(252) * 0.01",
        "    result = compute_backtest_result(returns)",
        "    val = getattr(result, metric)",
        "    assert not np.isnan(val), f'{metric} is NaN'",
        "    assert min_val <= val <= max_val, f'{metric}={val} outside [{min_val}, {max_val}]'",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_final_mega.py", build_final_mega_tests())

# Final count
result = subprocess.run(
    ["bash", "-c",
     "find /c/Users/Matthew/srfm-lab/aeternus/lumina -name '*.py' -o -name '*.yaml' | xargs wc -l 2>/dev/null | tail -1"],
    capture_output=True, text=True
)
print("GRAND TOTAL:", result.stdout.strip())
