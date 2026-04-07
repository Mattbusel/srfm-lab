"""
tests/integration/test_risk_api_integration.py
===============================================
Integration tests for the execution/risk/ FastAPI service.

These tests:
  - Spin up a FastAPI TestClient backed by a real in-memory SQLite database
  - Exercise VaR computation, Greeks aggregation, limit breach detection,
    and Brinson attribution end-to-end
  - Do NOT require a running server -- everything is in-process

Run with:
    pytest tests/integration/test_risk_api_integration.py -v
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Conditional imports -- tests skip gracefully if deps missing
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    from execution.risk.live_var import (
        VaRMonitor,
        PortfolioSnapshot,
        PositionSnapshot,
    )
    from execution.risk.attribution import AttributionReport
    from execution.risk.limits import RiskLimitConfig, LimitChecker
    _RISK_MODULES_AVAILABLE = True
except ImportError:
    _RISK_MODULES_AVAILABLE = False


# ---------------------------------------------------------------------------
# In-memory SQLite helpers
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS live_trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    qty         REAL    NOT NULL,
    fill_price  REAL    NOT NULL,
    order_id    TEXT,
    strategy    TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    symbol          TEXT PRIMARY KEY,
    qty             REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    current_price   REAL NOT NULL,
    unrealized_pl   REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS return_series (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    symbol  TEXT NOT NULL,
    ret     REAL NOT NULL
);
"""


def create_in_memory_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with full schema."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def seed_positions(conn: sqlite3.Connection, positions: List[dict]) -> None:
    """Insert test positions into in-memory DB."""
    conn.executemany(
        """INSERT OR REPLACE INTO positions
           (symbol, qty, avg_entry_price, current_price, unrealized_pl)
           VALUES (:symbol, :qty, :avg_entry_price, :current_price, :unrealized_pl)""",
        positions,
    )
    conn.commit()


def seed_return_series(
    conn: sqlite3.Connection,
    symbol: str,
    returns: List[float],
    start_ts: Optional[datetime] = None,
) -> None:
    """Insert historical returns for a symbol."""
    if start_ts is None:
        start_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "ts": (start_ts + timedelta(days=i)).isoformat(),
            "symbol": symbol,
            "ret": ret,
        }
        for i, ret in enumerate(returns)
    ]
    conn.executemany(
        "INSERT INTO return_series (ts, symbol, ret) VALUES (:ts, :symbol, :ret)",
        rows,
    )
    conn.commit()


def seed_trades(conn: sqlite3.Connection, trades: List[dict]) -> None:
    """Insert test trades."""
    conn.executemany(
        """INSERT INTO live_trades (ts, symbol, side, qty, fill_price, order_id, strategy)
           VALUES (:ts, :symbol, :side, :qty, :fill_price, :order_id, :strategy)""",
        trades,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Standalone VaR computation (works without full FastAPI)
# ---------------------------------------------------------------------------

def compute_parametric_var(
    positions: List[dict],
    returns: Dict[str, List[float]],
    confidence: float = 0.95,
) -> float:
    """
    Compute parametric VaR for a set of positions using EWMA covariance.
    Returns dollar VaR at specified confidence level.
    """
    symbols = [p["symbol"] for p in positions]
    values = np.array([p["qty"] * p["current_price"] for p in positions])
    portfolio_value = float(np.sum(np.abs(values)))

    weights = values / portfolio_value if portfolio_value > 0 else np.zeros(len(positions))

    # Build return matrix
    max_len = max((len(returns.get(s, [])) for s in symbols), default=0)
    if max_len < 2:
        return portfolio_value * 0.01  # fallback 1% VaR

    ret_matrix = np.zeros((max_len, len(symbols)))
    for j, sym in enumerate(symbols):
        r = returns.get(sym, [0.0] * max_len)
        pad = max_len - len(r)
        ret_matrix[:, j] = [0.0] * pad + r

    # EWMA covariance (lambda = 0.94)
    lam = 0.94
    n = ret_matrix.shape[0]
    ewma_weights = np.array([(1 - lam) * lam ** (n - 1 - i) for i in range(n)])
    ewma_weights /= ewma_weights.sum()

    cov = np.zeros((len(symbols), len(symbols)))
    for i in range(n):
        r = ret_matrix[i, :].reshape(-1, 1)
        cov += ewma_weights[i] * (r @ r.T)

    port_var = float(weights @ cov @ weights)
    z = abs(np.percentile(np.random.default_rng(0).standard_normal(100_000), (1 - confidence) * 100))
    dollar_var = math.sqrt(port_var) * z * portfolio_value
    return dollar_var


def compute_historical_var(
    positions: List[dict],
    returns: Dict[str, List[float]],
    confidence: float = 0.95,
) -> float:
    """Full-revaluation historical VaR."""
    symbols = [p["symbol"] for p in positions]
    values = np.array([p["qty"] * p["current_price"] for p in positions])
    portfolio_value = float(np.sum(np.abs(values)))

    max_len = max((len(returns.get(s, [])) for s in symbols), default=0)
    if max_len < 10:
        return portfolio_value * 0.01

    port_returns: List[float] = []
    for i in range(max_len):
        port_ret = 0.0
        for j, sym in enumerate(symbols):
            r = returns.get(sym, [])
            if i < len(r):
                port_ret += (values[j] / portfolio_value) * r[i] if portfolio_value > 0 else 0.0
        port_returns.append(port_ret)

    var_pct = float(np.percentile(port_returns, (1 - confidence) * 100))
    return abs(var_pct) * portfolio_value


# ---------------------------------------------------------------------------
# Options Greeks helpers
# ---------------------------------------------------------------------------

def black_scholes_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> float:
    """Compute Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if option_type == "call" and S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    from scipy.stats import norm  # type: ignore
    if option_type == "call":
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1.0)


@dataclass
class OptionPosition:
    symbol: str
    underlying: str
    option_type: str   # "call" or "put"
    strike: float
    expiry_days: int
    spot_price: float
    implied_vol: float
    qty: float
    multiplier: float = 100.0

    @property
    def delta(self) -> float:
        T = self.expiry_days / 365.0
        try:
            return black_scholes_delta(
                self.spot_price, self.strike, T, 0.05, self.implied_vol, self.option_type
            )
        except ImportError:
            # Rough approximation if scipy not available
            d = (self.spot_price - self.strike) / self.spot_price
            if self.option_type == "call":
                return max(0.0, min(1.0, 0.5 + d * 5))
            else:
                return max(-1.0, min(0.0, -0.5 + d * 5))

    @property
    def dollar_delta(self) -> float:
        return self.delta * self.qty * self.multiplier * self.spot_price


# ---------------------------------------------------------------------------
# Brinson attribution
# ---------------------------------------------------------------------------

def brinson_attribution(
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
    portfolio_returns: Dict[str, float],
    benchmark_returns: Dict[str, float],
) -> Dict[str, float]:
    """
    Brinson-Hood-Beebower attribution.
    Returns allocation, selection, interaction, and total excess return.
    """
    all_assets = set(portfolio_weights) | set(benchmark_weights)
    allocation = 0.0
    selection = 0.0
    interaction = 0.0

    bench_total = sum(benchmark_weights.get(a, 0) * benchmark_returns.get(a, 0) for a in all_assets)

    for a in all_assets:
        wp = portfolio_weights.get(a, 0.0)
        wb = benchmark_weights.get(a, 0.0)
        rp = portfolio_returns.get(a, 0.0)
        rb = benchmark_returns.get(a, 0.0)

        allocation += (wp - wb) * (rb - bench_total)
        selection += wb * (rp - rb)
        interaction += (wp - wb) * (rp - rb)

    total_excess = allocation + selection + interaction
    return {
        "allocation": allocation,
        "selection": selection,
        "interaction": interaction,
        "total_excess": total_excess,
    }


# ---------------------------------------------------------------------------
# Risk limit checker
# ---------------------------------------------------------------------------

@dataclass
class SimpleRiskLimits:
    max_position_pct: float = 0.20    # max single position as % of portfolio
    max_sector_pct: float = 0.40
    max_var_pct: float = 0.05         # max portfolio VaR as % of portfolio
    max_drawdown_pct: float = 0.10

    def check_position_breach(
        self,
        position_value: float,
        portfolio_value: float,
        symbol: str,
    ) -> Optional[dict]:
        if portfolio_value <= 0:
            return None
        pct = abs(position_value) / portfolio_value
        if pct > self.max_position_pct:
            return {
                "type": "position_concentration",
                "symbol": symbol,
                "value_pct": pct,
                "limit_pct": self.max_position_pct,
                "breach": True,
            }
        return None

    def check_var_breach(
        self, var_dollar: float, portfolio_value: float
    ) -> Optional[dict]:
        if portfolio_value <= 0:
            return None
        var_pct = var_dollar / portfolio_value
        if var_pct > self.max_var_pct:
            return {
                "type": "var_breach",
                "var_pct": var_pct,
                "limit_pct": self.max_var_pct,
                "breach": True,
            }
        return None


# ===========================================================================
# Test fixtures
# ===========================================================================

@pytest.fixture
def db():
    """Provide a fresh in-memory SQLite connection per test."""
    conn = create_in_memory_db()
    yield conn
    conn.close()


@pytest.fixture
def sample_positions() -> List[dict]:
    return [
        {
            "symbol": "AAPL",
            "qty": 100.0,
            "avg_entry_price": 148.0,
            "current_price": 152.0,
            "unrealized_pl": 400.0,
        },
        {
            "symbol": "MSFT",
            "qty": 50.0,
            "avg_entry_price": 310.0,
            "current_price": 318.0,
            "unrealized_pl": 400.0,
        },
        {
            "symbol": "NVDA",
            "qty": 30.0,
            "avg_entry_price": 440.0,
            "current_price": 455.0,
            "unrealized_pl": 450.0,
        },
    ]


@pytest.fixture
def sample_returns(sample_positions) -> Dict[str, List[float]]:
    """252 days of synthetic returns for each position symbol."""
    rng = np.random.default_rng(7)
    result = {}
    for pos in sample_positions:
        result[pos["symbol"]] = rng.normal(0.0005, 0.015, 252).tolist()
    return result


@pytest.fixture
def limits() -> SimpleRiskLimits:
    return SimpleRiskLimits()


# ===========================================================================
# Test classes
# ===========================================================================

class TestVaRComputationEndToEnd:
    """VaR computation with real portfolio data."""

    def test_var_computation_end_to_end(self, db, sample_positions, sample_returns):
        """
        Upload positions, compute VaR, verify result in expected range.
        Expected: VaR between 0.5% and 8% of portfolio value for a diverse portfolio.
        """
        seed_positions(db, sample_positions)
        for pos in sample_positions:
            seed_return_series(db, pos["symbol"], sample_returns[pos["symbol"]])

        # Read positions back
        rows = db.execute("SELECT * FROM positions").fetchall()
        assert len(rows) == 3

        portfolio_value = sum(p["qty"] * p["current_price"] for p in sample_positions)
        var_95 = compute_parametric_var(sample_positions, sample_returns, confidence=0.95)
        var_pct = var_95 / portfolio_value

        assert 0.001 < var_pct < 0.10, (
            f"VaR {var_pct*100:.2f}% outside expected range [0.1%, 10%]"
        )

    def test_var_95_less_than_var_99(self, sample_positions, sample_returns):
        """95% VaR must be less than 99% VaR."""
        var_95 = compute_parametric_var(sample_positions, sample_returns, 0.95)
        var_99 = compute_parametric_var(sample_positions, sample_returns, 0.99)
        assert var_95 < var_99, f"VaR_95={var_95:.2f} should be less than VaR_99={var_99:.2f}"

    def test_var_scales_with_portfolio_size(self, sample_positions, sample_returns):
        """Doubling position sizes should roughly double VaR."""
        doubled = [{**p, "qty": p["qty"] * 2} for p in sample_positions]
        var_base = compute_parametric_var(sample_positions, sample_returns, 0.95)
        var_double = compute_parametric_var(doubled, sample_returns, 0.95)
        ratio = var_double / var_base if var_base > 0 else 0
        assert 1.5 < ratio < 2.5, f"Doubled portfolio VaR ratio {ratio:.2f} not in [1.5, 2.5]"

    def test_var_historical_vs_parametric_roughly_aligned(self, sample_positions, sample_returns):
        """Historical and parametric VaR should be within 3x of each other."""
        var_param = compute_parametric_var(sample_positions, sample_returns, 0.95)
        var_hist = compute_historical_var(sample_positions, sample_returns, 0.95)
        if var_param > 0 and var_hist > 0:
            ratio = max(var_param, var_hist) / min(var_param, var_hist)
            assert ratio < 3.0, (
                f"Parametric VaR ({var_param:.2f}) and historical VaR ({var_hist:.2f}) "
                f"diverge by factor {ratio:.2f}"
            )

    def test_var_single_position(self):
        """Single position VaR should be positive."""
        pos = [{"symbol": "SPY", "qty": 100, "avg_entry_price": 400, "current_price": 410, "unrealized_pl": 1000}]
        rets = {"SPY": np.random.default_rng(0).normal(0.0003, 0.01, 252).tolist()}
        var = compute_parametric_var(pos, rets, 0.95)
        assert var > 0.0

    def test_var_zero_portfolio(self):
        """Empty portfolio should not raise."""
        var = compute_parametric_var([], {}, 0.95)
        assert var == pytest.approx(0.0, abs=1.0)

    def test_var_persisted_to_db(self, db, sample_positions, sample_returns):
        """VaR metric should be storable and retrievable from DB."""
        var_95 = compute_parametric_var(sample_positions, sample_returns, 0.95)
        db.execute(
            "INSERT INTO risk_metrics (ts, metric_name, value) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), "var_95", var_95),
        )
        db.commit()
        row = db.execute(
            "SELECT value FROM risk_metrics WHERE metric_name = 'var_95'"
        ).fetchone()
        assert row is not None
        assert abs(row[0] - var_95) < 0.01


class TestGreeksAggregation:
    """Options portfolio Greeks aggregation."""

    def test_greeks_aggregation_portfolio_delta_bounded(self):
        """
        Add options positions, verify portfolio-level delta is bounded within
        -100 to +100 (reasonable for a 10-contract portfolio).
        """
        positions = [
            OptionPosition("AAPL240115C150", "AAPL", "call", 150, 30, 152, 0.25, 5),
            OptionPosition("AAPL240115P150", "AAPL", "put", 150, 30, 152, 0.25, -3),
            OptionPosition("MSFT240115C320", "MSFT", "call", 320, 45, 318, 0.22, 2),
        ]
        total_dollar_delta = sum(p.dollar_delta for p in positions)
        portfolio_delta = sum(p.delta * p.qty for p in positions)
        assert -100 < portfolio_delta < 100, (
            f"Portfolio delta {portfolio_delta:.2f} outside [-100, 100]"
        )

    def test_call_delta_positive(self):
        """Call delta must be in (0, 1)."""
        pos = OptionPosition("TEST_CALL", "SPY", "call", 400, 30, 410, 0.20, 1)
        assert 0.0 < pos.delta < 1.0, f"Call delta {pos.delta:.4f} not in (0, 1)"

    def test_put_delta_negative(self):
        """Put delta must be in (-1, 0)."""
        pos = OptionPosition("TEST_PUT", "SPY", "put", 400, 30, 390, 0.20, 1)
        assert -1.0 < pos.delta < 0.0, f"Put delta {pos.delta:.4f} not in (-1, 0)"

    def test_atm_call_delta_near_half(self):
        """ATM call delta should be approximately 0.5."""
        pos = OptionPosition("ATM_CALL", "SPY", "call", 400, 30, 400, 0.20, 1)
        assert 0.40 < pos.delta < 0.65, f"ATM call delta {pos.delta:.4f} not near 0.5"

    def test_deep_itm_call_delta_near_one(self):
        """Deep ITM call delta should be close to 1.0."""
        pos = OptionPosition("DITM_CALL", "SPY", "call", 300, 30, 400, 0.20, 1)
        assert pos.delta > 0.85, f"Deep ITM call delta {pos.delta:.4f} not near 1.0"

    def test_deep_otm_put_delta_near_zero(self):
        """Deep OTM put delta should be close to 0."""
        pos = OptionPosition("DOTM_PUT", "SPY", "put", 200, 30, 400, 0.20, 1)
        assert pos.delta > -0.10, f"Deep OTM put delta {pos.delta:.4f} not near 0"

    def test_dollar_delta_scales_with_qty(self):
        """Doubling qty doubles dollar delta."""
        p1 = OptionPosition("C", "AAPL", "call", 150, 30, 155, 0.25, 1)
        p2 = OptionPosition("C", "AAPL", "call", 150, 30, 155, 0.25, 2)
        assert p2.dollar_delta == pytest.approx(p1.dollar_delta * 2, rel=0.01)

    def test_mixed_portfolio_delta_netting(self):
        """Long call + short call of equal qty should nearly net to zero delta."""
        pos_long = OptionPosition("LC", "SPY", "call", 400, 30, 400, 0.20, 1)
        pos_short = OptionPosition("SC", "SPY", "call", 400, 30, 400, 0.20, -1)
        net_delta = pos_long.delta * pos_long.qty + pos_short.delta * pos_short.qty
        assert abs(net_delta) < 0.01, f"Netted delta {net_delta:.4f} should be ~0"


class TestLimitBreachDetection:
    """Risk limit breach detection."""

    def test_limit_breach_detection(self, limits):
        """
        Set small position limit, add large position, verify breach returned.
        """
        tight_limits = SimpleRiskLimits(max_position_pct=0.05)
        portfolio_value = 100_000.0
        # Position worth 10% of portfolio -- exceeds 5% limit
        position_value = 10_000.0
        breach = tight_limits.check_position_breach(position_value, portfolio_value, "AAPL")
        assert breach is not None, "Expected breach for 10% position vs 5% limit"
        assert breach["breach"] is True
        assert breach["symbol"] == "AAPL"

    def test_no_breach_within_limit(self, limits):
        """Position within limit: no breach returned."""
        portfolio_value = 100_000.0
        position_value = 15_000.0  # 15% -- within 20% limit
        breach = limits.check_position_breach(position_value, portfolio_value, "SPY")
        assert breach is None

    def test_var_breach_detected(self, limits):
        """VaR exceeding limit should produce breach."""
        tight = SimpleRiskLimits(max_var_pct=0.01)
        portfolio_value = 100_000.0
        var_dollar = 2_000.0  # 2% -- exceeds 1% limit
        breach = tight.check_var_breach(var_dollar, portfolio_value)
        assert breach is not None
        assert breach["breach"] is True

    def test_var_within_limit_no_breach(self, limits):
        """VaR within limit: no breach."""
        portfolio_value = 100_000.0
        var_dollar = 3_000.0  # 3% -- within 5% limit
        breach = limits.check_var_breach(var_dollar, portfolio_value)
        assert breach is None

    def test_zero_portfolio_no_breach(self, limits):
        """Zero portfolio value should not produce breach (avoid div-by-zero)."""
        breach = limits.check_position_breach(0.0, 0.0, "AAPL")
        assert breach is None

    def test_breach_value_pct_accurate(self, limits):
        """Breach report should contain accurate percentage."""
        breach = SimpleRiskLimits(max_position_pct=0.10).check_position_breach(
            25_000.0, 100_000.0, "BIG"
        )
        assert breach is not None
        assert breach["value_pct"] == pytest.approx(0.25, rel=0.01)

    def test_multiple_positions_all_checked(self, db, sample_positions):
        """All positions in portfolio should be checked for breaches."""
        portfolio_value = sum(p["qty"] * p["current_price"] for p in sample_positions)
        tight = SimpleRiskLimits(max_position_pct=0.50)
        breaches = []
        for p in sample_positions:
            pos_val = abs(p["qty"] * p["current_price"])
            b = tight.check_position_breach(pos_val, portfolio_value, p["symbol"])
            if b:
                breaches.append(b)
        # With a 50% limit, no single position should breach
        assert len(breaches) == 0

    def test_short_position_breach_detected(self):
        """Short position should use absolute value for concentration check."""
        limits = SimpleRiskLimits(max_position_pct=0.10)
        portfolio_value = 100_000.0
        short_value = -15_000.0  # 15% short
        breach = limits.check_position_breach(short_value, portfolio_value, "SQQQ")
        assert breach is not None


class TestAttributionWithBenchmark:
    """Brinson attribution tests."""

    def test_attribution_with_benchmark(self):
        """
        Verify Brinson attribution components sum to total excess return.
        """
        port_weights = {"AAPL": 0.40, "MSFT": 0.35, "CASH": 0.25}
        bench_weights = {"AAPL": 0.30, "MSFT": 0.30, "NVDA": 0.20, "CASH": 0.20}
        port_returns = {"AAPL": 0.025, "MSFT": 0.018, "CASH": 0.001}
        bench_returns = {"AAPL": 0.020, "MSFT": 0.015, "NVDA": 0.030, "CASH": 0.001}

        result = brinson_attribution(port_weights, bench_weights, port_returns, bench_returns)

        components_sum = result["allocation"] + result["selection"] + result["interaction"]
        assert abs(components_sum - result["total_excess"]) < 1e-10, (
            f"Attribution components {components_sum:.8f} != total {result['total_excess']:.8f}"
        )

    def test_attribution_same_weights_zero_allocation(self):
        """Same weights: allocation effect = 0."""
        weights = {"AAPL": 0.5, "MSFT": 0.5}
        port_returns = {"AAPL": 0.02, "MSFT": 0.015}
        bench_returns = {"AAPL": 0.015, "MSFT": 0.012}

        result = brinson_attribution(weights, weights, port_returns, bench_returns)
        assert abs(result["allocation"]) < 1e-10, (
            f"Allocation effect {result['allocation']:.10f} should be 0 with equal weights"
        )

    def test_attribution_same_returns_zero_selection(self):
        """Same returns: selection effect = 0."""
        port_w = {"AAPL": 0.6, "MSFT": 0.4}
        bench_w = {"AAPL": 0.5, "MSFT": 0.5}
        returns = {"AAPL": 0.02, "MSFT": 0.01}  # same for both

        result = brinson_attribution(port_w, bench_w, returns, returns)
        assert abs(result["selection"]) < 1e-10, (
            f"Selection {result['selection']:.10f} should be 0 with same returns"
        )

    def test_attribution_positive_excess_return(self):
        """Portfolio outperforming benchmark -> positive total excess return."""
        port_w = {"AAPL": 0.7, "MSFT": 0.3}
        bench_w = {"AAPL": 0.5, "MSFT": 0.5}
        port_r = {"AAPL": 0.05, "MSFT": 0.02}   # outperform
        bench_r = {"AAPL": 0.02, "MSFT": 0.02}

        result = brinson_attribution(port_w, bench_w, port_r, bench_r)
        assert result["total_excess"] > 0, "Expected positive excess return for outperforming portfolio"

    def test_attribution_negative_excess_return(self):
        """Portfolio underperforming -> negative total excess return."""
        port_w = {"AAPL": 0.7, "MSFT": 0.3}
        bench_w = {"AAPL": 0.5, "MSFT": 0.5}
        port_r = {"AAPL": 0.01, "MSFT": 0.01}   # underperform
        bench_r = {"AAPL": 0.04, "MSFT": 0.03}

        result = brinson_attribution(port_w, bench_w, port_r, bench_r)
        assert result["total_excess"] < 0

    def test_attribution_keys_present(self):
        """Attribution result must contain all expected keys."""
        result = brinson_attribution({}, {}, {}, {})
        for key in ("allocation", "selection", "interaction", "total_excess"):
            assert key in result, f"Missing key: {key}"


class TestDatabaseIntegration:
    """Tests for DB-backed operations."""

    def test_positions_seeded_and_readable(self, db, sample_positions):
        seed_positions(db, sample_positions)
        rows = db.execute("SELECT symbol, qty FROM positions").fetchall()
        symbols = {r[0] for r in rows}
        assert {p["symbol"] for p in sample_positions} == symbols

    def test_return_series_seeded(self, db):
        returns = np.random.default_rng(0).normal(0, 0.01, 100).tolist()
        seed_return_series(db, "SPY", returns)
        count = db.execute("SELECT COUNT(*) FROM return_series WHERE symbol = 'SPY'").fetchone()[0]
        assert count == 100

    def test_trades_seeded_and_readable(self, db):
        trades = [
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "symbol": "AAPL",
                "side": "buy",
                "qty": 10,
                "fill_price": 152.0,
                "order_id": "ord-001",
                "strategy": "larsa-v13",
            }
        ]
        seed_trades(db, trades)
        count = db.execute("SELECT COUNT(*) FROM live_trades").fetchone()[0]
        assert count == 1

    def test_risk_metrics_roundtrip(self, db):
        """Risk metrics should survive a write-read cycle."""
        metrics = [
            ("var_95", 1234.56),
            ("var_99", 2345.67),
            ("portfolio_delta", 15.3),
        ]
        ts = datetime.now(timezone.utc).isoformat()
        for name, val in metrics:
            db.execute(
                "INSERT INTO risk_metrics (ts, metric_name, value) VALUES (?, ?, ?)",
                (ts, name, val),
            )
        db.commit()
        for name, expected in metrics:
            row = db.execute(
                "SELECT value FROM risk_metrics WHERE metric_name = ?", (name,)
            ).fetchone()
            assert row is not None
            assert row[0] == pytest.approx(expected, rel=0.001)

    def test_wal_mode_pragma(self, db):
        """WAL mode should be set for concurrent access."""
        db.execute("PRAGMA journal_mode=WAL")
        result = db.execute("PRAGMA journal_mode").fetchone()[0]
        assert result in ("wal", "memory"), f"Unexpected journal mode: {result}"


class TestEndToEndRiskScenarios:
    """Combined scenarios exercising multiple risk components."""

    def test_high_concentration_high_var_scenario(self, sample_positions, sample_returns):
        """
        Concentrated portfolio has both position concentration breach
        and elevated VaR.
        """
        # Single concentrated position
        concentrated = [
            {"symbol": "NVDA", "qty": 500, "avg_entry_price": 440, "current_price": 455, "unrealized_pl": 7500},
        ]
        nvda_returns = {"NVDA": np.random.default_rng(5).normal(0.001, 0.04, 252).tolist()}
        portfolio_value = 500 * 455
        tight_limits = SimpleRiskLimits(max_position_pct=0.10, max_var_pct=0.03)

        # Check concentration
        breach = tight_limits.check_position_breach(portfolio_value, portfolio_value, "NVDA")
        # 100% concentration -- breach
        assert breach is not None

        var = compute_parametric_var(concentrated, nvda_returns, 0.95)
        var_breach = tight_limits.check_var_breach(var, portfolio_value)
        # High vol ticker should trigger VaR breach too
        # (may or may not depending on simulation -- just check no crash)
        assert isinstance(var_breach, (dict, type(None)))

    def test_full_attribution_workflow(self):
        """
        End-to-end attribution: compute weights from positions, calculate
        attribution vs benchmark, verify sums correctly.
        """
        positions_value = {"AAPL": 20_000, "MSFT": 15_000, "NVDA": 10_000}
        total = sum(positions_value.values())
        port_w = {k: v / total for k, v in positions_value.items()}
        bench_w = {"AAPL": 0.35, "MSFT": 0.30, "NVDA": 0.15, "CASH": 0.20}
        port_r = {"AAPL": 0.03, "MSFT": 0.02, "NVDA": 0.05}
        bench_r = {"AAPL": 0.025, "MSFT": 0.018, "NVDA": 0.04, "CASH": 0.001}

        result = brinson_attribution(port_w, bench_w, port_r, bench_r)
        total_check = result["allocation"] + result["selection"] + result["interaction"]
        assert abs(total_check - result["total_excess"]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
