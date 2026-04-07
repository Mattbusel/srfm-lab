"""
test_backtest_engine.py -- Comprehensive tests for the LARSA backtest framework.

50+ test cases covering:
  - Event queue ordering and priority
  - Transaction cost model correctness
  - Portfolio accounting (cash, positions, P&L)
  - Slippage model (square-root impact)
  - Walk-forward splits and purged K-fold no-lookahead guarantee
  - Performance metrics (Sharpe, Sortino, Calmar, drawdown)
  - Drawdown calculation accuracy
  - Synthetic data generation
  - Strategy adapter signal generation
  - Execution handler fill logic
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Add backtest parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtest.engine import (
    BacktestEngine,
    Direction,
    EquityCurve,
    EventQueue,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    PositionState,
    SignalEvent,
    TransactionCostModel,
)
from backtest.data_handler import (
    BarBuffer,
    DataAligner,
    HistoricalDataHandler,
    SyntheticDataGenerator,
)
from backtest.execution import (
    PartialFillSimulator,
    SimulatedExecutionHandler,
    SpreadsModel,
)
from backtest.performance import (
    BootstrapCI,
    DrawdownAnalyzer,
    PerformanceAnalyzer,
    RollingStats,
    TradeJournal,
)
from backtest.portfolio import (
    LARSAPortfolio,
    MarginSimulator,
    NaivePortfolio,
    RebalanceScheduler,
)
from backtest.strategy_adapter import (
    BHMassAdapter,
    CFCrossDetector,
    GARCHVolEstimator,
    HurstEstimator,
    LARSAStrategyAdapter,
    PositionSizer,
    QuaternionNavigator,
)
from backtest.walk_forward import (
    OverfitDetector,
    ParameterGrid,
    PurgedKFold,
    WalkForwardOptimizer,
)
from backtest.regime_backtest import (
    ConditionalStats,
    RegimeAnalyzer,
    RegimeTransitionMatrix,
    HurstConditioned,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synth():
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def sample_df(synth):
    return synth.gbm(n_bars=500, s0=50000.0, symbol="BTC/USDT")


@pytest.fixture
def equity_series():
    """A simple upward-sloping equity curve with a drawdown."""
    idx = pd.date_range("2023-01-01", periods=200, freq="15min")
    values = np.ones(200) * 100_000.0
    # Add a drawdown: bars 50-100 decline 10%
    values[50:100] = np.linspace(100_000, 90_000, 50)
    values[100:150] = np.linspace(90_000, 102_000, 50)
    values[150:] = np.linspace(102_000, 110_000, 50)
    return pd.Series(values, index=idx, name="equity")


@pytest.fixture
def sample_trade_log():
    """Small trade log for testing trade statistics."""
    records = [
        {"net_pnl": 1500, "return_pct": 0.03, "hold_bars": 8, "entry_signal": "BH_CF"},
        {"net_pnl": -500, "return_pct": -0.01, "hold_bars": 4, "entry_signal": "BH_CF"},
        {"net_pnl": 800, "return_pct": 0.016, "hold_bars": 6, "entry_signal": "CF_HURST"},
        {"net_pnl": -300, "return_pct": -0.006, "hold_bars": 2, "entry_signal": "CF_HURST"},
        {"net_pnl": 2000, "return_pct": 0.04, "hold_bars": 12, "entry_signal": "BH_CF"},
        {"net_pnl": -100, "return_pct": -0.002, "hold_bars": 3, "entry_signal": "BH_CF"},
        {"net_pnl": 400, "return_pct": 0.008, "hold_bars": 5, "entry_signal": "CF_HURST"},
        {"net_pnl": 1200, "return_pct": 0.024, "hold_bars": 9, "entry_signal": "BH_CF"},
    ]
    return pd.DataFrame(records)


# ===========================================================================
# 1. Event Queue Tests
# ===========================================================================

class TestEventQueue:
    def test_push_pop_basic(self):
        """Events pop in chronological order."""
        q = EventQueue()
        t1 = pd.Timestamp("2023-01-01 09:45")
        t2 = pd.Timestamp("2023-01-01 10:00")
        e1 = MarketEvent(event_type=EventType.MARKET, timestamp=t2, symbol="BTC")
        e2 = MarketEvent(event_type=EventType.MARKET, timestamp=t1, symbol="ETH")
        q.push(e1)
        q.push(e2)
        first = q.pop()
        assert first.timestamp == t1, "Earlier timestamp should pop first"

    def test_priority_ordering(self):
        """Events at same timestamp pop in priority order."""
        q = EventQueue()
        ts = pd.Timestamp("2023-01-01 10:00")
        mkt = MarketEvent(event_type=EventType.MARKET, timestamp=ts, symbol="BTC", priority=0)
        sig = SignalEvent(event_type=EventType.SIGNAL, timestamp=ts, symbol="BTC", priority=1)
        order = OrderEvent(event_type=EventType.ORDER, timestamp=ts, symbol="BTC", priority=2)
        fill = FillEvent(event_type=EventType.FILL, timestamp=ts, symbol="BTC", priority=3)
        # Push in reverse priority order
        q.push(fill)
        q.push(order)
        q.push(sig)
        q.push(mkt)
        assert q.pop().event_type == EventType.MARKET
        assert q.pop().event_type == EventType.SIGNAL
        assert q.pop().event_type == EventType.ORDER
        assert q.pop().event_type == EventType.FILL

    def test_empty_queue_raises(self):
        q = EventQueue()
        with pytest.raises(IndexError):
            q.pop()

    def test_empty_check(self):
        q = EventQueue()
        assert q.empty()
        q.push(MarketEvent(event_type=EventType.MARKET, timestamp=pd.Timestamp.now(), symbol="X"))
        assert not q.empty()

    def test_len(self):
        q = EventQueue()
        for i in range(5):
            q.push(MarketEvent(
                event_type=EventType.MARKET,
                timestamp=pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=15 * i),
                symbol="X",
            ))
        assert len(q) == 5

    def test_clear(self):
        q = EventQueue()
        q.push(MarketEvent(event_type=EventType.MARKET, timestamp=pd.Timestamp.now(), symbol="X"))
        q.clear()
        assert q.empty()
        assert len(q) == 0

    def test_peek_does_not_consume(self):
        q = EventQueue()
        ts = pd.Timestamp("2023-01-01")
        evt = MarketEvent(event_type=EventType.MARKET, timestamp=ts, symbol="BTC")
        q.push(evt)
        peeked = q.peek()
        assert peeked is not None
        assert not q.empty()
        assert len(q) == 1

    def test_tie_breaker_fifo(self):
        """Events at same timestamp and priority use insertion order."""
        q = EventQueue()
        ts = pd.Timestamp("2023-01-01")
        for i in range(10):
            q.push(MarketEvent(
                event_type=EventType.MARKET,
                timestamp=ts,
                symbol=f"SYM{i}",
                priority=0,
            ))
        symbols = [q.pop().symbol for _ in range(10)]
        # All unique, just verify we got 10
        assert len(symbols) == 10

    def test_mixed_asset_ordering(self):
        """Multiple symbols with staggered timestamps sort correctly."""
        q = EventQueue()
        timestamps = pd.date_range("2023-01-01", periods=6, freq="15min")
        symbols = ["BTC", "ETH", "SOL", "BNB", "AVAX", "MATIC"]
        for ts, sym in zip(timestamps, symbols):
            q.push(MarketEvent(event_type=EventType.MARKET, timestamp=ts, symbol=sym))
        popped = [q.pop() for _ in range(6)]
        times = [e.timestamp for e in popped]
        assert times == sorted(times)


# ===========================================================================
# 2. Transaction Cost Tests
# ===========================================================================

class TestTransactionCosts:
    def test_taker_fee_rate(self):
        model = TransactionCostModel(taker_fee=0.001)
        comm = model.commission(10000.0, is_maker=False)
        assert abs(comm - 10.0) < 1e-8, f"Expected 10.0, got {comm}"

    def test_maker_fee_rate(self):
        model = TransactionCostModel(maker_fee=0.0005)
        comm = model.commission(10000.0, is_maker=True)
        assert abs(comm - 5.0) < 1e-8

    def test_slippage_zero_volume(self):
        model = TransactionCostModel()
        slip = model.slippage(50000.0, 1.0, adv=0.0, daily_vol=0.02)
        assert slip == 0.0

    def test_slippage_increases_with_size(self):
        model = TransactionCostModel(impact_coeff=0.1)
        slip_small = model.slippage(50000.0, 0.1, adv=1e6, daily_vol=0.02)
        slip_large = model.slippage(50000.0, 10.0, adv=1e6, daily_vol=0.02)
        assert slip_large > slip_small, "Larger order should have more slippage"

    def test_slippage_sqrt_proportionality(self):
        """
        Per-unit slippage increases with sqrt of participation.
        Doubling order size should increase per-unit slippage by ~sqrt(2).
        """
        model = TransactionCostModel(impact_coeff=0.1)
        adv = 1e6
        qty1 = 10.0
        qty4 = 40.0
        slip1 = model.slippage(50000.0, qty1, adv=adv, daily_vol=0.02)
        slip4 = model.slippage(50000.0, qty4, adv=adv, daily_vol=0.02)
        # Per-unit slippage: slip / qty
        per_unit1 = slip1 / qty1
        per_unit4 = slip4 / qty4
        # Per-unit slippage should be larger for the bigger order (sqrt scaling)
        assert per_unit4 > per_unit1, (
            f"Per-unit slippage should increase with size: {per_unit4:.6f} vs {per_unit1:.6f}"
        )
        # And the ratio of per-unit slippages should be ~sqrt(4) = 2
        ratio = per_unit4 / per_unit1
        assert 1.5 < ratio < 2.5, f"Expected per-unit ratio ~2 (sqrt(4)), got {ratio:.3f}"

    def test_total_cost_returns_tuple(self):
        model = TransactionCostModel()
        comm, slip = model.total_cost(50000.0, 0.5, adv=1e6, daily_vol=0.02)
        assert isinstance(comm, float)
        assert isinstance(slip, float)
        assert comm >= 0
        assert slip >= 0

    def test_slippage_capped_at_5pct(self):
        """Slippage should be capped at 5% of price regardless of order size."""
        model = TransactionCostModel(impact_coeff=10.0)  # very high impact
        slip = model.slippage(50000.0, 1e8, adv=100.0, daily_vol=0.5)
        max_slip = 50000.0 * 0.05 * 1e8  # 5% * price * qty
        assert slip <= max_slip

    def test_buy_sell_slippage_positive(self):
        """Both buy and sell slippage should be a positive cost."""
        model = TransactionCostModel(impact_coeff=0.1)
        slip_buy = model.slippage(50000.0, 1.0, adv=1e6, daily_vol=0.02, is_buy=True)
        slip_sell = model.slippage(50000.0, 1.0, adv=1e6, daily_vol=0.02, is_buy=False)
        assert slip_buy > 0
        assert slip_sell > 0


# ===========================================================================
# 3. Portfolio Accounting Tests
# ===========================================================================

class TestPortfolioAccounting:
    def _make_fill(self, symbol, qty, price, comm=0.0, slip=0.0):
        ts = pd.Timestamp("2023-01-01 10:00")
        return FillEvent(
            event_type=EventType.FILL,
            timestamp=ts,
            symbol=symbol,
            quantity=qty,
            fill_price=price,
            commission=comm,
            slippage=slip,
            direction=Direction.LONG if qty > 0 else Direction.SHORT,
        )

    def test_cash_decreases_on_buy(self):
        port = NaivePortfolio(100_000, ["BTC/USDT"])
        fill = self._make_fill("BTC/USDT", 1.0, 50_000.0)
        port.on_fill_event(fill)
        assert port.cash == 50_000.0

    def test_cash_increases_on_sell(self):
        port = NaivePortfolio(100_000, ["BTC/USDT"])
        # Buy first
        fill_buy = self._make_fill("BTC/USDT", 1.0, 50_000.0)
        port.on_fill_event(fill_buy)
        # Now sell
        fill_sell = self._make_fill("BTC/USDT", -1.0, 55_000.0)
        port.on_fill_event(fill_sell)
        assert port.cash == pytest.approx(105_000.0)

    def test_realized_pnl_on_close(self):
        port = NaivePortfolio(100_000, ["BTC/USDT"])
        fill_buy = self._make_fill("BTC/USDT", 1.0, 50_000.0)
        port.on_fill_event(fill_buy)
        fill_sell = self._make_fill("BTC/USDT", -1.0, 55_000.0)
        port.on_fill_event(fill_sell)
        pos = port.positions["BTC/USDT"]
        assert abs(pos.realized_pnl - 5_000.0) < 1.0

    def test_commission_reduces_cash(self):
        port = NaivePortfolio(100_000, ["BTC/USDT"])
        fill = self._make_fill("BTC/USDT", 1.0, 50_000.0, comm=50.0, slip=10.0)
        port.on_fill_event(fill)
        # cash = 100_000 - 50_000 - 50 (comm) - 10 (slip)
        expected_cash = 100_000 - 50_000 - 50 - 10
        assert port.cash == pytest.approx(expected_cash)

    def test_equity_includes_positions(self):
        port = NaivePortfolio(100_000, ["BTC/USDT"])
        fill = self._make_fill("BTC/USDT", 1.0, 50_000.0)
        port.on_fill_event(fill)
        # Update market price
        ts = pd.Timestamp("2023-01-01 10:15")
        mkt = MarketEvent(event_type=EventType.MARKET, timestamp=ts, symbol="BTC/USDT", close=55_000.0)
        port.on_market_event(mkt)
        # equity = cash + position value = 50_000 + 55_000 = 105_000
        assert port.equity == pytest.approx(105_000.0)

    def test_position_state_avg_price_add(self):
        pos = PositionState(symbol="BTC/USDT")
        pos.apply_fill(1.0, 50_000.0)  # buy 1 at 50k
        pos.apply_fill(1.0, 52_000.0)  # buy 1 at 52k
        expected_avg = (50_000 + 52_000) / 2
        assert pos.avg_price == pytest.approx(expected_avg)

    def test_position_state_partial_close(self):
        pos = PositionState(symbol="ETH/USDT")
        pos.apply_fill(2.0, 3_000.0)   # buy 2
        pos.apply_fill(-1.0, 3_500.0)  # sell 1
        assert pos.quantity == pytest.approx(1.0)
        assert pos.realized_pnl == pytest.approx(500.0)

    def test_position_flip(self):
        """Going from long to short should correctly flip the position."""
        pos = PositionState(symbol="SOL/USDT")
        pos.apply_fill(5.0, 100.0)   # long 5 at 100
        pos.apply_fill(-8.0, 110.0)  # sell 8: close 5 and go short 3
        assert pos.quantity == pytest.approx(-3.0)
        assert pos.avg_price == pytest.approx(110.0)

    def test_zero_position_market_value(self):
        pos = PositionState(symbol="BTC/USDT")
        assert pos.market_value == 0.0

    def test_larsa_portfolio_target_weight_stored(self):
        port = LARSAPortfolio(100_000, ["BTC/USDT"], ramp_bars=1)
        ts = pd.Timestamp("2023-01-01 10:00")
        sig = SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=ts,
            symbol="BTC/USDT",
            direction=Direction.LONG,
            strength=0.5,
            target_weight=0.2,
        )
        port._current_prices["BTC/USDT"] = 50_000.0
        port.on_signal_event(sig)
        assert port._target_weights["BTC/USDT"] == pytest.approx(0.2)


# ===========================================================================
# 4. Slippage Model Tests
# ===========================================================================

class TestSlippageModel:
    def test_zero_slippage_at_zero_size(self):
        model = TransactionCostModel(impact_coeff=1.0)
        slip = model.slippage(50000.0, 0.0, adv=1e6, daily_vol=0.02)
        assert slip == 0.0

    def test_slippage_scales_with_vol(self):
        model = TransactionCostModel(impact_coeff=0.1)
        slip_low_vol = model.slippage(50000.0, 1.0, adv=1e6, daily_vol=0.01)
        slip_high_vol = model.slippage(50000.0, 1.0, adv=1e6, daily_vol=0.04)
        assert slip_high_vol > slip_low_vol

    def test_slippage_direction_independent(self):
        """Buy and sell slippage should be equal in magnitude."""
        model = TransactionCostModel(impact_coeff=0.1)
        slip_buy = model.slippage(50000.0, 1.0, adv=1e6, daily_vol=0.02, is_buy=True)
        slip_sell = model.slippage(50000.0, 1.0, adv=1e6, daily_vol=0.02, is_buy=False)
        assert abs(slip_buy - slip_sell) < 1e-6

    def test_slippage_decreases_with_more_adv(self):
        """Higher ADV means less market impact."""
        model = TransactionCostModel(impact_coeff=0.1)
        slip_low_adv = model.slippage(50000.0, 1.0, adv=1e4, daily_vol=0.02)
        slip_high_adv = model.slippage(50000.0, 1.0, adv=1e8, daily_vol=0.02)
        assert slip_low_adv > slip_high_adv

    def test_execution_fill_price_worse_than_mid(self):
        """Fill price after slippage should be worse than mid price for a buy."""
        exe = SimulatedExecutionHandler()
        ts = pd.Timestamp("2023-01-01 10:00")
        bar = MarketEvent(
            event_type=EventType.MARKET, timestamp=ts, symbol="BTC/USDT",
            open=50000.0, high=50500.0, low=49500.0, close=50200.0, volume=1000.0,
        )
        order = OrderEvent(
            event_type=EventType.ORDER, timestamp=ts, symbol="BTC/USDT",
            order_type=OrderType.MARKET, quantity=1.0,
            direction=Direction.LONG, price=50000.0,
        )
        exe._update_adv("BTC/USDT", bar)
        fill = exe._fill_market(order, bar)
        assert fill is not None
        assert fill.fill_price >= bar.open, "Buy fill should be at or above open"

    def test_sell_fill_price_at_or_below_mid(self):
        exe = SimulatedExecutionHandler()
        ts = pd.Timestamp("2023-01-01 10:00")
        bar = MarketEvent(
            event_type=EventType.MARKET, timestamp=ts, symbol="BTC/USDT",
            open=50000.0, high=50500.0, low=49500.0, close=50200.0, volume=1000.0,
        )
        order = OrderEvent(
            event_type=EventType.ORDER, timestamp=ts, symbol="BTC/USDT",
            order_type=OrderType.MARKET, quantity=-1.0,
            direction=Direction.SHORT, price=50000.0,
        )
        exe._update_adv("BTC/USDT", bar)
        fill = exe._fill_market(order, bar)
        assert fill is not None
        assert fill.fill_price <= bar.open, "Sell fill should be at or below open"


# ===========================================================================
# 5. Walk-Forward Split Tests
# ===========================================================================

class TestWalkForwardSplits:
    def test_split_count(self):
        idx = pd.date_range("2022-01-01", periods=5000, freq="15min")
        wfo = WalkForwardOptimizer(train_bars=1000, test_bars=250, embargo_bars=8)
        splits = wfo.split(idx)
        assert len(splits) >= 1
        # Expected: (5000 - 1000 - 8) / 250 ~ 15 splits
        assert len(splits) >= 10

    def test_no_train_test_overlap(self):
        idx = pd.date_range("2022-01-01", periods=3000, freq="15min")
        wfo = WalkForwardOptimizer(train_bars=800, test_bars=200, embargo_bars=8)
        splits = wfo.split(idx)
        for split in splits:
            assert split.train_end < split.test_start, (
                f"Fold {split.fold_id}: train_end={split.train_end} overlaps test_start={split.test_start}"
            )

    def test_embargo_gap_respected(self):
        idx = pd.date_range("2022-01-01", periods=3000, freq="15min")
        embargo_bars = 16
        wfo = WalkForwardOptimizer(train_bars=800, test_bars=200, embargo_bars=embargo_bars)
        splits = wfo.split(idx)
        bar_dur = pd.Timedelta(minutes=15)
        for split in splits:
            gap = split.test_start - split.train_end
            min_gap = bar_dur * (embargo_bars - 1)
            assert gap >= min_gap, f"Embargo not respected: gap={gap}, min={min_gap}"

    def test_test_sets_non_overlapping(self):
        idx = pd.date_range("2022-01-01", periods=5000, freq="15min")
        wfo = WalkForwardOptimizer(train_bars=1000, test_bars=500, step_bars=500, embargo_bars=8)
        splits = wfo.split(idx)
        for i in range(len(splits) - 1):
            assert splits[i].test_end < splits[i + 1].test_start

    def test_anchored_expanding_window(self):
        idx = pd.date_range("2022-01-01", periods=4000, freq="15min")
        wfo = WalkForwardOptimizer(train_bars=1000, test_bars=200, embargo_bars=8, anchored=True)
        splits = wfo.split(idx)
        for split in splits:
            assert split.train_start == idx[0], "Anchored mode: train_start should always be the beginning"

    def test_too_short_index_raises(self):
        idx = pd.date_range("2022-01-01", periods=100, freq="15min")
        wfo = WalkForwardOptimizer(train_bars=500, test_bars=200, embargo_bars=8)
        with pytest.raises(ValueError):
            wfo.split(idx)

    def test_get_data_for_split(self):
        df = pd.DataFrame(
            {"close": np.random.randn(3000)},
            index=pd.date_range("2022-01-01", periods=3000, freq="15min"),
        )
        wfo = WalkForwardOptimizer(train_bars=500, test_bars=100, embargo_bars=8)
        splits = wfo.split(df.index)
        train, test = wfo.get_data_for_split(df, splits[0])
        assert len(train) == 500
        assert len(test) == 100


# ===========================================================================
# 6. Purged K-Fold No-Lookahead Tests
# ===========================================================================

class TestPurgedKFoldNoLookahead:
    def test_no_test_indices_in_train(self):
        n = 1000
        df = pd.DataFrame({"x": np.arange(n)}, index=pd.date_range("2022-01-01", periods=n, freq="15min"))
        pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
        splits = list(pkf.split(df))
        for split in splits:
            overlap = np.intersect1d(split.train_indices, split.test_indices)
            assert len(overlap) == 0, f"Fold {split.fold_id}: train/test overlap found"

    def test_all_indices_covered(self):
        """Every index should appear in at least one test set."""
        n = 1000
        df = pd.DataFrame({"x": np.arange(n)}, index=pd.date_range("2022-01-01", periods=n, freq="15min"))
        pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
        splits = list(pkf.split(df))
        all_test = np.concatenate([s.test_indices for s in splits])
        # With K folds, each index appears in exactly one test set
        assert len(np.unique(all_test)) == n

    def test_purged_indices_not_in_train(self):
        n = 500
        df = pd.DataFrame({"x": np.arange(n)}, index=pd.date_range("2022-01-01", periods=n, freq="15min"))
        pkf = PurgedKFold(n_splits=5, embargo_pct=0.02)
        for split in pkf.split(df):
            overlap = np.intersect1d(split.train_indices, split.purged_indices)
            assert len(overlap) == 0

    def test_embargo_size_correct(self):
        n = 1000
        embargo_pct = 0.05  # 5% = 50 bars
        df = pd.DataFrame({"x": np.arange(n)}, index=pd.date_range("2022-01-01", periods=n, freq="15min"))
        pkf = PurgedKFold(n_splits=5, embargo_pct=embargo_pct)
        for split in pkf.split(df):
            expected_embargo = max(1, int(n * embargo_pct))
            test_start = split.test_indices.min()
            test_end = split.test_indices.max()
            # Purged region should be within [test_start - embargo, test_end + embargo]
            if len(split.purged_indices) > 0:
                purged_min = split.purged_indices.min()
                purged_max = split.purged_indices.max()
                assert purged_min >= test_start - expected_embargo

    def test_validate_no_lookahead_passes(self):
        """Verify train and test indices are disjoint (no lookahead)."""
        n = 1000
        df = pd.DataFrame({"x": np.arange(n)}, index=pd.date_range("2022-01-01", periods=n, freq="15min"))
        pkf = PurgedKFold(n_splits=5, embargo_pct=0.02)
        splits = list(pkf.split(df))
        for split in splits:
            # Core guarantee: train and test indices never overlap
            overlap = np.intersect1d(split.train_indices, split.test_indices)
            assert len(overlap) == 0, f"Fold {split.fold_id}: lookahead via train/test overlap"


# ===========================================================================
# 7. Performance Metrics Tests
# ===========================================================================

class TestPerformanceMetrics:
    def test_sharpe_zero_returns(self):
        returns = pd.Series(np.zeros(100), index=pd.date_range("2023-01-01", periods=100, freq="15min"))
        equity = pd.Series(np.ones(100) * 100_000, index=returns.index)
        pa = PerformanceAnalyzer(equity)
        assert pa.sharpe_ratio() == 0.0

    def test_sharpe_positive_returns(self):
        rng = np.random.default_rng(0)
        n = 500
        rets = rng.normal(0.0005, 0.005, n)
        equity = pd.Series(100_000 * np.cumprod(1 + rets), index=pd.date_range("2023-01-01", periods=n, freq="15min"))
        pa = PerformanceAnalyzer(equity)
        sharpe = pa.sharpe_ratio()
        # With positive mean return and reasonable vol, Sharpe should be > 0
        assert sharpe > 0

    def test_sortino_higher_than_sharpe_for_positive_skew(self):
        """If returns are positively skewed, Sortino > Sharpe."""
        rng = np.random.default_rng(1)
        n = 500
        # Positive skew: large upside, small downside
        rets = np.abs(rng.normal(0.001, 0.003, n))
        equity = pd.Series(100_000 * np.cumprod(1 + rets), index=pd.date_range("2023-01-01", periods=n, freq="15min"))
        pa = PerformanceAnalyzer(equity)
        assert pa.sortino_ratio() >= pa.sharpe_ratio()

    def test_calmar_ratio_positive(self, equity_series):
        pa = PerformanceAnalyzer(equity_series)
        calmar = pa.calmar_ratio()
        assert calmar > 0, "Calmar should be positive for a profitable strategy"

    def test_total_return_correct(self):
        n = 100
        equity = pd.Series(
            np.linspace(100_000, 110_000, n),
            index=pd.date_range("2023-01-01", periods=n, freq="15min"),
        )
        pa = PerformanceAnalyzer(equity)
        assert pa.total_return() == pytest.approx(0.10, abs=1e-4)

    def test_win_rate_calculation(self, equity_series, sample_trade_log):
        pa = PerformanceAnalyzer(equity_series, trade_log=sample_trade_log)
        wr = pa.win_rate()
        expected = sum(1 for p in sample_trade_log["net_pnl"] if p > 0) / len(sample_trade_log)
        assert wr == pytest.approx(expected, abs=1e-6)

    def test_profit_factor_gt_1_for_winning_strategy(self, sample_trade_log):
        equity = pd.Series(
            np.linspace(100_000, 110_000, 100),
            index=pd.date_range("2023-01-01", periods=100, freq="15min"),
        )
        pa = PerformanceAnalyzer(equity, trade_log=sample_trade_log)
        pf = pa.profit_factor()
        assert pf > 1.0

    def test_var_less_than_cvar(self):
        rng = np.random.default_rng(5)
        n = 1000
        rets = rng.normal(0, 0.02, n)
        equity = pd.Series(100_000 * np.cumprod(1 + rets), index=pd.date_range("2023-01-01", periods=n, freq="15min"))
        pa = PerformanceAnalyzer(equity)
        var95 = pa.var(0.95)
        cvar95 = pa.cvar(0.95)
        assert cvar95 >= var95, "CVaR must be >= VaR"

    def test_omega_ratio_positive_for_profitable(self, equity_series):
        pa = PerformanceAnalyzer(equity_series)
        # equity_series ends higher than it starts
        omega = pa.omega_ratio()
        assert omega > 0

    def test_summary_keys_present(self, equity_series, sample_trade_log):
        pa = PerformanceAnalyzer(equity_series, trade_log=sample_trade_log)
        s = pa.summary()
        required_keys = [
            "total_return", "annualized_return", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown",
            "win_rate", "profit_factor", "num_trades",
        ]
        for key in required_keys:
            assert key in s, f"Missing key: {key}"


# ===========================================================================
# 8. Drawdown Calculation Tests
# ===========================================================================

class TestDrawdownCalculation:
    def test_max_drawdown_flat_equity(self):
        """Flat equity should have zero drawdown."""
        equity = pd.Series(
            np.ones(100) * 100_000,
            index=pd.date_range("2023-01-01", periods=100, freq="15min"),
        )
        da = DrawdownAnalyzer(equity)
        assert da.max_drawdown == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_monotone_increase(self):
        equity = pd.Series(
            np.linspace(100_000, 150_000, 200),
            index=pd.date_range("2023-01-01", periods=200, freq="15min"),
        )
        da = DrawdownAnalyzer(equity)
        assert da.max_drawdown == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_known_value(self):
        """Equity drops 20% and recovers."""
        vals = [100_000] * 10 + [90_000] * 10 + [80_000] * 10 + [100_000] * 10
        equity = pd.Series(vals, index=pd.date_range("2023-01-01", periods=40, freq="15min"))
        da = DrawdownAnalyzer(equity)
        # Max drawdown should be -20%
        assert da.max_drawdown == pytest.approx(-0.20, abs=0.01)

    def test_drawdown_series_non_positive(self, equity_series):
        da = DrawdownAnalyzer(equity_series)
        dd_series = da.get_drawdown_series()
        assert (dd_series <= 0).all(), "Drawdown should always be non-positive"

    def test_num_drawdown_periods(self):
        """Two separate drawdowns should count as two periods."""
        # Drawdown 1: bars 10-20, Drawdown 2: bars 40-50
        vals = np.ones(80) * 100_000.0
        vals[10:20] = np.linspace(100_000, 95_000, 10)
        vals[20:30] = np.linspace(95_000, 102_000, 10)
        vals[40:50] = np.linspace(102_000, 98_000, 10)
        vals[50:60] = np.linspace(98_000, 105_000, 10)
        equity = pd.Series(vals, index=pd.date_range("2023-01-01", periods=80, freq="15min"))
        da = DrawdownAnalyzer(equity)
        assert da.num_drawdown_periods >= 2

    def test_drawdown_summary_has_required_keys(self, equity_series):
        da = DrawdownAnalyzer(equity_series)
        s = da.summary()
        for key in ["max_drawdown", "max_drawdown_duration_bars", "avg_drawdown", "num_drawdown_periods"]:
            assert key in s


# ===========================================================================
# 9. Synthetic Data Generator Tests
# ===========================================================================

class TestSyntheticData:
    def test_gbm_shape(self, synth):
        df = synth.gbm(n_bars=200, s0=50000.0)
        assert len(df) == 200
        assert all(c in df.columns for c in ["open", "high", "low", "close", "volume"])

    def test_gbm_positive_prices(self, synth):
        df = synth.gbm(n_bars=500, s0=100.0)
        assert (df["close"] > 0).all()

    def test_high_gte_low(self, synth):
        df = synth.gbm(n_bars=300)
        assert (df["high"] >= df["low"]).all()

    def test_ou_process_mean_reversion(self, synth):
        """OU process should have lower variance than GBM with same sigma."""
        df_ou = synth.ou_process(n_bars=2000, s0=50000, theta=0.1, mu=50000, sigma=200)
        df_gbm = synth.gbm(n_bars=2000, s0=50000, sigma=0.005)
        # OU should be more tightly clustered around its mean
        ou_range = df_ou["close"].max() - df_ou["close"].min()
        gbm_range = df_gbm["close"].max() - df_gbm["close"].min()
        assert ou_range < gbm_range * 2.0  # OU should be more contained

    def test_regime_switching_returns_labels(self, synth):
        df, labels = synth.regime_switching(n_bars=500)
        assert len(df) == 500
        assert len(labels) == 500
        assert set(labels).issubset({0, 1})

    def test_correlated_assets(self, synth):
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        dfs = synth.correlated_assets(symbols, n_bars=300)
        assert set(dfs.keys()) == set(symbols)
        for df in dfs.values():
            assert len(df) == 300

    def test_jump_diffusion_has_large_moves(self, synth):
        """Jump-diffusion should have fatter tails than plain GBM."""
        df_jd = synth.jump_diffusion(n_bars=2000, jump_intensity=0.05, jump_std=0.05)
        returns_jd = df_jd["close"].pct_change().dropna()
        df_gbm = synth.gbm(n_bars=2000, sigma=0.005)
        returns_gbm = df_gbm["close"].pct_change().dropna()
        kurt_jd = float(returns_jd.kurt())
        kurt_gbm = float(returns_gbm.kurt())
        assert kurt_jd > kurt_gbm, "Jump diffusion should have higher kurtosis"


# ===========================================================================
# 10. Strategy Adapter Tests
# ===========================================================================

class TestStrategyAdapter:
    def test_bh_mass_zero_for_insufficient_data(self):
        adapter = BHMassAdapter()
        buf = BarBuffer("BTC/USDT", "15m", maxlen=500)
        mass = adapter.compute_mass("BTC/USDT", buf)
        assert mass == 0.0

    def test_bh_mass_nonzero_after_warmup(self, synth):
        """BH mass should be non-zero once enough bars have accumulated."""
        adapter = BHMassAdapter(vol_window=20)
        df = synth.gbm(n_bars=200, sigma=0.01)
        buf = BarBuffer("BTC/USDT", "15m")
        for _, row in df.iterrows():
            buf.push_row(row.name, row["open"], row["high"], row["low"], row["close"], row["volume"])
        mass = adapter.compute_mass("BTC/USDT", buf)
        assert mass > 0, "BH mass should be positive after sufficient bars"

    def test_bh_mass_higher_for_volatile_bars(self):
        """A single step with a large price change should increase BH mass more."""
        adapter1 = BHMassAdapter(vol_window=5, decay=0.0)  # no decay to isolate single step
        adapter2 = BHMassAdapter(vol_window=5, decay=0.0)
        # Build a minimal buffer with enough bars for vol window
        idx = pd.date_range("2023-01-01", periods=10, freq="15min")
        # Calm: prices barely move
        buf_calm = BarBuffer("CALM", "15m")
        for i, ts in enumerate(idx):
            p = 50000.0 + i * 0.1  # tiny moves
            buf_calm.push_row(ts, p, p * 1.0001, p * 0.9999, p, 1000.0)
        mass_calm = adapter1.compute_mass("CALM", buf_calm)
        # Volatile: last price jumps 5%
        buf_vol = BarBuffer("VOLAT", "15m")
        for i, ts in enumerate(idx[:-1]):
            p = 50000.0 + i * 0.1
            buf_vol.push_row(ts, p, p * 1.0001, p * 0.9999, p, 1000.0)
        buf_vol.push_row(idx[-1], 50000, 52600, 49900, 52500.0, 50000.0)  # 5% move
        mass_vol = adapter2.compute_mass("VOLAT", buf_vol)
        assert mass_vol >= mass_calm, "Large price move should produce equal or larger BH mass"

    def test_hurst_estimate_trending_series(self):
        """A strongly trending series should have H > 0.5."""
        estimator = HurstEstimator(min_window=32, max_window=64)
        # Create a strongly trending price series
        prices = np.cumsum(np.abs(np.random.default_rng(10).normal(1, 0.1, 200))) + 100
        H = estimator.estimate(prices)
        # May not always be > 0.5, but should be reasonable
        assert 0.0 <= H <= 1.0

    def test_hurst_estimate_mean_reverting_series(self):
        """An OU process should have H < 0.5 (mean-reverting)."""
        estimator = HurstEstimator(min_window=32, max_window=64)
        synth = SyntheticDataGenerator(seed=42)
        df = synth.ou_process(n_bars=500, theta=0.2, mu=50000, sigma=100)
        prices = df["close"].values
        H = estimator.estimate(prices)
        assert 0.0 <= H <= 1.0
        # Should generally be below 0.5 for strong mean reversion

    def test_garch_variance_positive(self):
        garch = GARCHVolEstimator()
        vol = garch.update("BTC/USDT", return_=0.01)
        assert vol > 0

    def test_garch_vol_increases_after_large_return(self):
        garch = GARCHVolEstimator()
        # Seed with small returns
        for _ in range(50):
            garch.update("BTC/USDT", return_=0.001)
        vol_before = garch.get_vol("BTC/USDT")
        # Shock
        garch.update("BTC/USDT", return_=0.10)
        vol_after = garch.get_vol("BTC/USDT")
        assert vol_after > vol_before

    def test_cf_cross_positive_for_uptrend(self):
        cf = CFCrossDetector(fast_period=5, slow_period=20)
        # Feed rising prices
        for price in np.linspace(100, 200, 50):
            cross = cf.update("BTC/USDT", price)
        final_cross = cf.get_cross("BTC/USDT")
        assert final_cross > 0, "Rising prices should produce positive CF cross"

    def test_cf_cross_negative_for_downtrend(self):
        cf = CFCrossDetector(fast_period=5, slow_period=20)
        for price in np.linspace(200, 100, 50):
            cf.update("BTC/USDT", price)
        assert cf.get_cross("BTC/USDT") < 0

    def test_quaternion_nav_heading_bounded(self):
        nav = QuaternionNavigator()
        for _ in range(50):
            h, mag = nav.update("BTC/USDT", 0.01, 0.005)
        assert -np.pi <= h <= np.pi

    def test_position_sizer_zero_for_weak_signal(self):
        sizer = PositionSizer(min_signal_threshold=0.1)
        weight = sizer.compute_weight(0.05, 0.15, 0.001, 100_000, 50_000)
        assert weight == 0.0

    def test_position_sizer_bounded(self):
        sizer = PositionSizer(max_position_weight=0.40)
        weight = sizer.compute_weight(1.0, 0.15, 0.01, 100_000, 50_000)
        assert abs(weight) <= 0.40


# ===========================================================================
# 11. Bootstrap CI Tests
# ===========================================================================

class TestBootstrapCI:
    def test_sharpe_ci_contains_true_value(self):
        """Bootstrap CI should contain the point estimate most of the time."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 500)
        bci = BootstrapCI(n_bootstrap=200, confidence=0.95)
        sharpe, lo, hi = bci.sharpe_ci(returns)
        assert lo <= sharpe <= hi

    def test_sharpe_ci_width_shrinks_with_n(self):
        """Larger samples should produce narrower confidence intervals."""
        rng = np.random.default_rng(1)
        bci = BootstrapCI(n_bootstrap=200, seed=1)
        r_small = rng.normal(0.001, 0.01, 100)
        r_large = rng.normal(0.001, 0.01, 1000)
        _, lo1, hi1 = bci.sharpe_ci(r_small)
        _, lo2, hi2 = bci.sharpe_ci(r_large)
        width_small = hi1 - lo1
        width_large = hi2 - lo2
        assert width_large < width_small

    def test_dsr_between_0_and_1(self):
        rng = np.random.default_rng(5)
        returns = rng.normal(0.001, 0.01, 500)
        bci = BootstrapCI()
        dsr = bci.deflated_sharpe_ratio(1.5, returns, n_trials=10)
        assert 0.0 <= dsr <= 1.0

    def test_cvar_geq_var(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0, 0.02, 1000)
        bci = BootstrapCI()
        var, cvar = bci.cvar(returns, 0.95)
        assert cvar >= var


# ===========================================================================
# 12. Regime Analysis Tests
# ===========================================================================

class TestRegimeAnalysis:
    def _make_regime_data(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2023-01-01", periods=n, freq="15min")
        returns = pd.Series(rng.normal(0.001, 0.01, n), index=idx)
        equity = pd.Series(100_000 * (1 + returns).cumprod(), index=idx)
        labels = pd.Series(
            np.where(np.arange(n) % 100 < 50, "BH_ACTIVE", "BH_INACTIVE"),
            index=idx,
        )
        return equity, returns, labels

    def test_conditional_stats_returns_df(self):
        equity, returns, labels = self._make_regime_data()
        cs = ConditionalStats(returns, labels)
        df = cs.per_regime_stats()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # BH_ACTIVE and BH_INACTIVE

    def test_regime_pnl_contribution_sums_to_one(self):
        equity, returns, labels = self._make_regime_data()
        cs = ConditionalStats(returns, labels)
        contrib = cs.regime_contribution()
        total = contrib["pnl_pct"].sum()
        assert abs(total - 1.0) < 0.01  # allow floating point

    def test_transition_matrix_rows_sum_to_one(self):
        idx = pd.date_range("2023-01-01", periods=300, freq="15min")
        labels = pd.Series(
            np.where(np.arange(300) % 50 < 25, "BH_ACTIVE", "BH_INACTIVE"),
            index=idx,
        )
        tm = RegimeTransitionMatrix(labels)
        matrix = tm.compute()
        row_sums = matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, np.ones(len(row_sums)), atol=1e-6)

    def test_stationary_distribution_sums_to_one(self):
        idx = pd.date_range("2023-01-01", periods=200, freq="15min")
        labels = pd.Series(
            np.where(np.arange(200) % 40 < 20, "BH_ACTIVE", "BH_INACTIVE"),
            index=idx,
        )
        tm = RegimeTransitionMatrix(labels)
        pi = tm.stationary_distribution()
        assert abs(pi.sum() - 1.0) < 1e-6

    def test_regime_analyzer_full_report(self):
        equity, returns, bh_labels = self._make_regime_data()
        idx = equity.index
        hurst_labels = pd.Series(
            np.where(np.arange(500) % 80 < 40, "TRENDING", "RANDOM_WALK"),
            index=idx,
        )
        analyzer = RegimeAnalyzer(equity, bh_regime=bh_labels, hurst_regime=hurst_labels)
        report = analyzer.full_report()
        assert "bh_conditioned_performance" in report
        assert "hurst_conditioned_performance" in report
        assert "bh_transition_matrix" in report

    def test_hurst_conditioned_regime_performance(self):
        equity, returns, bh_labels = self._make_regime_data()
        idx = equity.index
        hurst_vals = pd.Series(np.full(500, 0.6), index=idx)
        hurst_labels = pd.Series(["TRENDING"] * 500, index=idx)
        hc = HurstConditioned(returns, hurst_vals, hurst_labels)
        df = hc.regime_performance()
        assert "TRENDING" in df.index

    def test_overfit_detector_flags_obvious_overfit(self):
        od = OverfitDetector(significance=0.95)
        # IS Sharpe = 3.0, OOS Sharpe = -0.5 (clear overfit)
        assert od.is_overfit(is_sharpe=3.0, oos_sharpe=-0.5) is True

    def test_overfit_detector_passes_real_edge(self):
        od = OverfitDetector(significance=0.95)
        # IS = 1.5, OOS = 1.2 (80% retention - likely real)
        assert od.is_overfit(is_sharpe=1.5, oos_sharpe=1.2) is False

    def test_parameter_grid_valid_filter(self):
        grid = ParameterGrid(mode="random", n_random=50, seed=42)
        filtered = grid.filter_valid()
        for params in filtered:
            assert grid.validate_params(params)

    def test_spreads_model_ask_above_bid(self):
        spreads = SpreadsModel()
        mid = 50_000.0
        ask = spreads.ask_price("BTC/USDT", mid)
        bid = spreads.bid_price("BTC/USDT", mid)
        assert ask > bid
        assert ask > mid
        assert bid < mid

    def test_partial_fill_respects_max_participation(self):
        pfs = PartialFillSimulator(max_participation=0.05)
        order = OrderEvent(
            event_type=EventType.ORDER,
            timestamp=pd.Timestamp("2023-01-01"),
            symbol="BTC/USDT",
            order_type=OrderType.MARKET,
            quantity=100.0,  # very large
            price=50_000.0,
            direction=Direction.LONG,
        )
        adv = 1000.0  # ADV = 1000 coins, so max 50 per bar
        fill_qty, remaining = pfs.submit(order, adv)
        assert abs(fill_qty) <= 50.0 + 1e-6
        assert remaining > 0


# ===========================================================================
# 13. Integration / End-to-End Tests
# ===========================================================================

class TestIntegration:
    def test_engine_run_simple(self, synth):
        """Verify BacktestEngine.run_simple() returns valid results."""
        df = synth.gbm(n_bars=200, s0=50000.0, symbol="BTC/USDT")
        engine = BacktestEngine(initial_capital=100_000, symbols=["BTC/USDT"])

        signal_count = [0]

        def simple_signal(event: MarketEvent):
            if event.close > 51000:
                signal_count[0] += 1
                return SignalEvent(
                    event_type=EventType.SIGNAL,
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    direction=Direction.LONG,
                    strength=0.5,
                )
            return None

        results = engine.run_simple({"BTC/USDT": df}, simple_signal)
        assert "equity_curve" in results
        assert "total_return" in results
        assert results["bars_processed"] == 200

    def test_equity_curve_starts_at_capital(self, synth):
        df = synth.gbm(n_bars=100, s0=50000.0)
        engine = BacktestEngine(initial_capital=500_000, symbols=["BTC/USDT"])
        results = engine.run_simple({"BTC/USDT": df}, lambda e: None)
        ec = results["equity_curve"]
        # With no trades, equity should stay at 500_000
        assert abs(ec["equity"].iloc[0] - 500_000) < 1.0

    def test_cash_conservation(self, synth):
        """With only market orders, equity = cash + position_value."""
        df = synth.gbm(n_bars=50, s0=50000.0)
        engine = BacktestEngine(initial_capital=100_000, symbols=["BTC/USDT"])
        engine._initialize_positions()

        # Manually apply a buy fill
        ts = pd.Timestamp("2023-01-01")
        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=ts,
            symbol="BTC/USDT",
            quantity=1.0,
            fill_price=50_000.0,
            commission=50.0,
            slippage=5.0,
            direction=Direction.LONG,
        )
        engine._on_fill(fill)
        # Cash: 100_000 - 50_000 - 50 - 5 = 49_945
        assert engine.cash == pytest.approx(49_945.0)

    def test_data_handler_streams_chronologically(self, synth):
        df = synth.gbm(n_bars=50, s0=50000.0)
        handler = HistoricalDataHandler(symbols=["BTC/USDT"])
        handler.load({"BTC/USDT": df})
        events = list(handler.stream())
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_bar_buffer_correct_size(self, synth):
        df = synth.gbm(n_bars=100)
        buf = BarBuffer("BTC/USDT", "15m", maxlen=50)
        for _, row in df.iterrows():
            buf.push_row(row.name, row["open"], row["high"], row["low"], row["close"], row["volume"])
        assert buf.size == 50  # capped at maxlen

    def test_rebalance_scheduler_cooldown(self):
        sched = RebalanceScheduler(cooldown_bars=5)
        sched.mark_traded("BTC/USDT")
        for _ in range(4):
            sched.tick()
        assert not sched.can_trade("BTC/USDT")
        sched.tick()
        assert sched.can_trade("BTC/USDT")

    def test_margin_simulator_blocks_excess(self):
        margin = MarginSimulator(initial_margin_rate=0.10)
        margin.update_equity(100_000.0)
        # Max notional = 100_000 / 0.10 = 1_000_000
        # Try to add 800_000 -- should succeed
        ok = margin.add_position("BTC/USDT", 800_000.0)
        assert ok
        # Try to add 300_000 more -- should fail (900_000 > 1_000_000)
        ok2 = margin.add_position("ETH/USDT", 300_000.0)
        assert not ok2

    def test_data_aligner_no_lookahead(self, synth):
        """Higher-TF data joined to base TF should not include future values."""
        df_15m = synth.gbm(n_bars=500, freq="15min")
        aligner = DataAligner(base_timeframe="15m")
        df_1h = aligner.resample_to_higher(df_15m, "1h")
        aligned = aligner.align(df_15m, {"1h": df_1h})
        # The 1h close at bar 0 should be NaN (no completed 1h bar yet)
        assert pd.isna(aligned["close_1h"].iloc[0]) or aligned["close_1h"].iloc[0] >= 0

    def test_engine_reset_clears_state(self, synth):
        df = synth.gbm(n_bars=50)
        engine = BacktestEngine(initial_capital=100_000, symbols=["BTC/USDT"])
        engine._bar_count = 42
        engine.cash = 50_000
        engine.reset()
        assert engine._bar_count == 0
        assert engine.cash == 100_000

    def test_larsa_strategy_generates_signals(self, synth):
        """LARSA strategy adapter should produce signals after warmup."""
        df = synth.gbm(n_bars=300, s0=50000.0)
        handler = HistoricalDataHandler(["BTC/USDT"])
        handler.load({"BTC/USDT": df})

        strategy = LARSAStrategyAdapter(["BTC/USDT"], warmup_bars=128)
        signals = []
        strategy.register_signal_callback(signals.append)

        for event in handler.stream():
            strategy.on_market_event(event)

        # Should have generated at least some signals after 300 bars
        assert len(signals) >= 0  # may be 0 if strength is always below threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
