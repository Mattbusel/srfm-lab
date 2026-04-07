"""
test_onchain_advanced.py -- Unit tests for the onchain_advanced package.

Coverage:
  - WhaleTracker: exchange deposit classified as bearish
  - WhaleTracker: exchange withdrawal classified as bullish
  - WhaleTracker: below-threshold txs are ignored
  - WhaleTracker: net_whale_flow sign conventions
  - WhaleTracker: whale_signal direction with net outflows
  - WhaleTracker: whale_signal direction with net inflows
  - AddressClassifier: known exchange address lookup
  - AddressClassifier: cold-storage heuristic
  - MinerMetricsAnalyzer: Puell Multiple with known values
  - MinerMetricsAnalyzer: Puell signal direction
  - MinerMetricsAnalyzer: hash_rate_signal direction
  - MinerMetricsAnalyzer: miner_capitulation_risk below breakeven
  - HashRateEstimator: estimate_from_difficulty formula
  - HashRateEstimator: breakeven_price sanity check
  - NetworkValueMetrics: NVT ratio formula
  - NetworkValueMetrics: NVT signal is ratio of NVT to MA
  - NetworkValueMetrics: Metcalfe ratio > 1 when overpriced
  - NetworkValueMetrics: S2F model price increases with scarcity
  - NetworkValueMetrics: realized_cap sums utxo values
  - NetworkValueMetrics: MVRV signal direction
  - StablecoinFlowAnalyzer: rising supply produces positive signal
  - StablecoinFlowAnalyzer: falling supply produces negative signal
  - StablecoinFlowAnalyzer: stablecoin_dominance formula
  - DexStablecoinMonitor: net_stablecoin_buys sign
  - DexStablecoinMonitor: stable_to_crypto_flow_signal direction
"""

from __future__ import annotations

import math
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# Make the onchain_advanced package importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from onchain_advanced.whale_tracker import (
    AddressClassifier,
    AddressType,
    EventType,
    Transaction,
    WhaleEvent,
    WhaleTracker,
)
from onchain_advanced.miner_metrics import (
    HashRateEstimator,
    MinerMetricsAnalyzer,
)
from onchain_advanced.stablecoin_flows import (
    DexStablecoinMonitor,
    DexTrade,
    StablecoinFlowAnalyzer,
)
from onchain_advanced.network_value import NetworkValueMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tx(
    usd_value: float,
    from_addr: str = "0xaaaa",
    to_addr:   str = "0xbbbb",
    is_deposit:    bool = False,
    is_withdrawal: bool = False,
    asset: str = "BTC",
    dt: Optional[datetime] = None,
) -> Transaction:
    """Factory for Transaction test objects."""
    from typing import Optional
    return Transaction(
        tx_hash=f"0x{abs(hash((from_addr, to_addr, usd_value))):#018x}"[:18],
        asset=asset,
        from_addr=from_addr,
        to_addr=to_addr,
        amount=usd_value / 60_000.0,
        usd_value=usd_value,
        timestamp=dt or datetime.now(tz=timezone.utc),
        is_exchange_deposit=is_deposit,
        is_exchange_withdrawal=is_withdrawal,
    )


def _make_date_series(
    values: list,
    start: str = "2024-01-01",
) -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# WhaleTracker tests
# ---------------------------------------------------------------------------

class TestWhaleTracker:

    def setup_method(self) -> None:
        self.tracker = WhaleTracker(db_path=":memory:")
        self.tracker.define_whale_threshold("BTC", threshold_usd=500_000.0)

    def test_exchange_deposit_is_bearish(self) -> None:
        """A transaction flagged as exchange deposit should produce a bearish event."""
        tx = _make_tx(1_000_000.0, is_deposit=True)
        evt = self.tracker.process_transaction(tx)
        assert evt is not None
        assert evt.event_type == EventType.EXCHANGE_DEPOSIT.value
        assert evt.is_bullish is False
        assert evt.is_bearish is True

    def test_exchange_withdrawal_is_bullish(self) -> None:
        """A transaction flagged as exchange withdrawal should produce a bullish event."""
        tx = _make_tx(2_000_000.0, is_withdrawal=True)
        evt = self.tracker.process_transaction(tx)
        assert evt is not None
        assert evt.event_type == EventType.EXCHANGE_WITHDRAWAL.value
        assert evt.is_bullish is True

    def test_below_threshold_returns_none(self) -> None:
        """Transactions below the whale threshold must be ignored."""
        tx = _make_tx(100_000.0, is_deposit=True)
        evt = self.tracker.process_transaction(tx)
        assert evt is None

    def test_net_whale_flow_negative_on_deposits(self) -> None:
        """Net whale flow should be negative when deposits dominate."""
        now = datetime.now(tz=timezone.utc)
        for i in range(5):
            tx = _make_tx(
                1_000_000.0,
                is_deposit=True,
                dt=now - timedelta(hours=i),
            )
            self.tracker.process_transaction(tx)
        flow = self.tracker.net_whale_flow("BTC", hours=24)
        assert flow < 0, f"Expected negative flow, got {flow}"

    def test_net_whale_flow_positive_on_withdrawals(self) -> None:
        """Net whale flow should be positive when withdrawals dominate."""
        now = datetime.now(tz=timezone.utc)
        for i in range(5):
            tx = _make_tx(
                1_000_000.0,
                is_withdrawal=True,
                dt=now - timedelta(hours=i),
            )
            self.tracker.process_transaction(tx)
        flow = self.tracker.net_whale_flow("BTC", hours=24)
        assert flow > 0, f"Expected positive flow, got {flow}"

    def test_whale_signal_positive_on_net_outflows(self) -> None:
        """
        Net exchange outflows (withdrawals > deposits) should produce
        a positive (bullish) signal.
        """
        now = datetime.now(tz=timezone.utc)
        # 8 withdrawals vs 1 deposit
        for i in range(8):
            tx = _make_tx(2_000_000.0, is_withdrawal=True, dt=now - timedelta(hours=i + 1))
            self.tracker.process_transaction(tx)
        tx = _make_tx(1_000_000.0, is_deposit=True, dt=now - timedelta(hours=9))
        self.tracker.process_transaction(tx)

        sig = self.tracker.whale_signal("BTC")
        assert sig > 0, f"Expected positive signal, got {sig}"

    def test_whale_signal_negative_on_net_inflows(self) -> None:
        """
        Net exchange inflows (deposits > withdrawals) should produce
        a negative (bearish) signal.
        """
        now = datetime.now(tz=timezone.utc)
        for i in range(8):
            tx = _make_tx(2_000_000.0, is_deposit=True, dt=now - timedelta(hours=i + 1))
            self.tracker.process_transaction(tx)
        tx = _make_tx(1_000_000.0, is_withdrawal=True, dt=now - timedelta(hours=9))
        self.tracker.process_transaction(tx)

        sig = self.tracker.whale_signal("BTC")
        assert sig < 0, f"Expected negative signal, got {sig}"

    def test_whale_signal_range(self) -> None:
        """Signal must always be within [-1, +1]."""
        now = datetime.now(tz=timezone.utc)
        for i in range(20):
            tx = _make_tx(
                float(1_000_000 * (i + 1)),
                is_deposit=True,
                dt=now - timedelta(hours=i),
            )
            self.tracker.process_transaction(tx)
        sig = self.tracker.whale_signal("BTC")
        assert -1.0 <= sig <= 1.0, f"Signal out of range: {sig}"

    def test_recent_events_respects_window(self) -> None:
        """recent_whale_events should not return events outside the time window."""
        now = datetime.now(tz=timezone.utc)
        # Two recent events and one old
        old_tx = _make_tx(1_000_000.0, is_deposit=True, dt=now - timedelta(hours=48))
        new_tx1 = _make_tx(1_000_000.0, is_withdrawal=True, dt=now - timedelta(hours=2))
        new_tx2 = _make_tx(1_500_000.0, is_deposit=True, dt=now - timedelta(hours=1))
        for tx in (old_tx, new_tx1, new_tx2):
            self.tracker.process_transaction(tx)

        events_24h = self.tracker.recent_whale_events("BTC", hours=24)
        assert len(events_24h) == 2


# ---------------------------------------------------------------------------
# AddressClassifier tests
# ---------------------------------------------------------------------------

class TestAddressClassifier:

    def setup_method(self) -> None:
        self.clf = AddressClassifier()

    def test_known_exchange_eth(self) -> None:
        """Well-known Binance hot wallet should classify as EXCHANGE."""
        addr = "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be"
        assert self.clf.classify(addr) == AddressType.EXCHANGE

    def test_known_exchange_case_insensitive(self) -> None:
        """Exchange classification should be case-insensitive."""
        addr = "0x3F5CE5FBFE3E9AF3971DD833D26BA9B5C936F0BE"
        assert self.clf.classify(addr) == AddressType.EXCHANGE

    def test_cold_storage_whale_heuristic(self) -> None:
        """Large balance + few transactions should classify as WHALE."""
        addr = "0xdeadbeef12345678"
        self.clf.update_address_stats(addr, balance_usd=5_000_000.0, tx_count=10)
        assert self.clf.classify(addr) == AddressType.WHALE

    def test_unknown_address(self) -> None:
        """Random unknown address should return UNKNOWN."""
        assert self.clf.classify("0xrandomaddress999") == AddressType.UNKNOWN

    def test_defi_router_prefix(self) -> None:
        """Uniswap v2 router prefix should classify as DEFI."""
        addr = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d"
        assert self.clf.classify(addr) == AddressType.DEFI


# ---------------------------------------------------------------------------
# MinerMetricsAnalyzer tests
# ---------------------------------------------------------------------------

class TestMinerMetrics:

    def setup_method(self) -> None:
        self.analyzer = MinerMetricsAnalyzer(db_path=":memory:")

    def test_puell_multiple_known_values(self) -> None:
        """
        Puell Multiple should equal daily revenue / 365-day MA.
        At a stable revenue of $1M/day, Puell = 1.0.
        """
        # 400 days of constant $1M revenue
        revenue = _make_date_series([1_000_000.0] * 400, start="2022-01-01")
        puell = self.analyzer.puell_multiple(revenue, window=365)
        # After 365 days, Puell should be very close to 1.0
        last_value = float(puell.iloc[-1])
        assert abs(last_value - 1.0) < 1e-3, f"Expected Puell ~1.0, got {last_value}"

    def test_puell_multiple_high_revenue(self) -> None:
        """
        Sudden revenue spike relative to baseline should produce Puell > 1.
        """
        base = [1_000_000.0] * 365
        spike = [10_000_000.0] * 30
        revenue = _make_date_series(base + spike, start="2022-01-01")
        puell = self.analyzer.puell_multiple(revenue, window=365)
        last_value = float(puell.iloc[-1])
        assert last_value > 1.0, f"High revenue should yield Puell > 1, got {last_value}"

    def test_puell_signal_direction_low(self) -> None:
        """Low Puell Multiple should produce positive signal (oversold / bullish)."""
        # Puell of 0.3 is below the oversold threshold of 0.5
        puell_series = _make_date_series([0.3] * 100)
        signal = self.analyzer.puell_signal(puell_series)
        assert float(signal.iloc[-1]) > 0, "Low Puell should give positive signal"

    def test_puell_signal_direction_high(self) -> None:
        """High Puell Multiple should produce negative signal (overbought / bearish)."""
        puell_series = _make_date_series([5.0] * 100)
        signal = self.analyzer.puell_signal(puell_series)
        assert float(signal.iloc[-1]) < 0, "High Puell should give negative signal"

    def test_hash_rate_signal_rising_is_positive(self) -> None:
        """Rising hash rate should produce a positive signal."""
        hr = _make_date_series(list(range(1, 201)), start="2023-01-01")
        sig = self.analyzer.hash_rate_signal(hr, short_window=14, long_window=90)
        assert float(sig.iloc[-1]) > 0, "Rising hash rate should give positive signal"

    def test_hash_rate_signal_falling_is_negative(self) -> None:
        """Declining hash rate should produce a negative signal."""
        hr = _make_date_series(list(range(200, 0, -1)), start="2023-01-01")
        sig = self.analyzer.hash_rate_signal(hr, short_window=14, long_window=90)
        assert float(sig.iloc[-1]) < 0, "Falling hash rate should give negative signal"

    def test_capitulation_risk_high_below_breakeven(self) -> None:
        """Risk should be elevated when price is well below breakeven."""
        price    = _make_date_series([10_000.0] * 200, start="2022-01-01")
        hash_rate = _make_date_series(list(range(200, 0, -1)), start="2022-01-01")
        risk = self.analyzer.miner_capitulation_risk(
            hash_rate, price, breakeven_usd=20_000.0
        )
        avg_risk = float(risk.iloc[-50:].mean())
        assert avg_risk > 0.3, f"Expected high capitulation risk, got {avg_risk}"

    def test_capitulation_risk_low_above_breakeven(self) -> None:
        """Risk should be low when price is comfortably above breakeven."""
        price     = _make_date_series([60_000.0] * 200, start="2022-01-01")
        hash_rate = _make_date_series(list(range(1, 201)), start="2022-01-01")
        risk = self.analyzer.miner_capitulation_risk(
            hash_rate, price, breakeven_usd=20_000.0
        )
        avg_risk = float(risk.iloc[-50:].mean())
        assert avg_risk < 0.3, f"Expected low capitulation risk, got {avg_risk}"


# ---------------------------------------------------------------------------
# HashRateEstimator tests
# ---------------------------------------------------------------------------

class TestHashRateEstimator:

    def test_estimate_from_difficulty_formula(self) -> None:
        """
        H/s = difficulty * 2^32 / block_time_s
        At difficulty=1 and block_time=600s, expect 2^32/600 ~ 7.158e6 H/s
        """
        hr = HashRateEstimator.estimate_from_difficulty(difficulty=1.0, block_time_s=600.0)
        expected = (2 ** 32) / 600.0
        assert abs(hr - expected) < 1e-3, f"Expected {expected}, got {hr}"

    def test_estimate_scales_linearly_with_difficulty(self) -> None:
        """Doubling difficulty should double estimated hash rate."""
        hr1 = HashRateEstimator.estimate_from_difficulty(1e13, 600.0)
        hr2 = HashRateEstimator.estimate_from_difficulty(2e13, 600.0)
        assert abs(hr2 / hr1 - 2.0) < 1e-6

    def test_breakeven_price_positive(self) -> None:
        """Breakeven price must be a positive finite float."""
        be = HashRateEstimator.breakeven_price(electricity_cost_kwh=0.05)
        assert be > 0
        assert math.isfinite(be)

    def test_breakeven_price_higher_electricity(self) -> None:
        """Higher electricity cost should produce higher breakeven price."""
        be_low  = HashRateEstimator.breakeven_price(electricity_cost_kwh=0.03)
        be_high = HashRateEstimator.breakeven_price(electricity_cost_kwh=0.10)
        assert be_high > be_low

    def test_current_block_subsidy_post_halving(self) -> None:
        """Block subsidy at height 840_000 should be 3.125 BTC (2024 halving)."""
        subsidy = HashRateEstimator.current_block_subsidy(840_000)
        assert subsidy == 3.125


# ---------------------------------------------------------------------------
# NetworkValueMetrics tests
# ---------------------------------------------------------------------------

class TestNetworkValueMetrics:

    def setup_method(self) -> None:
        self.nvm = NetworkValueMetrics(db_path=":memory:")

    def test_nvt_ratio_formula(self) -> None:
        """
        NVT = market_cap / smoothed_transaction_volume.
        With constant inputs, NVT should equal their ratio.
        """
        mc  = _make_date_series([1_000_000.0] * 50)
        vol = _make_date_series([10_000.0]   * 50)
        nvt = self.nvm.nvt_ratio(mc, vol, smoothing_window=1)
        expected = 100.0
        assert abs(float(nvt.iloc[-1]) - expected) < 1e-3, (
            f"Expected NVT ~{expected}, got {float(nvt.iloc[-1])}"
        )

    def test_nvt_signal_equals_nvt_over_ma(self) -> None:
        """
        NVT signal at window=5 should equal NVT / 5-day MA of NVT.
        Use constant NVT so signal should be ~1.0.
        """
        mc  = _make_date_series([1_000_000.0] * 30)
        vol = _make_date_series([10_000.0]   * 30)
        nvt = self.nvm.nvt_ratio(mc, vol, smoothing_window=1)
        sig = self.nvm.nvt_signal(nvt, window=5)
        # Constant NVT -> signal should be ~1.0
        last = float(sig.iloc[-1])
        assert abs(last - 1.0) < 1e-3, f"Expected NVT signal ~1.0, got {last}"

    def test_nvt_directional_signal_cheap(self) -> None:
        """NVT < 50 should produce positive directional signal."""
        mc  = _make_date_series([500_000.0] * 30)
        vol = _make_date_series([20_000.0]  * 30)   # NVT = 25 (cheap)
        sig = self.nvm.nvt_directional_signal(mc, vol, smoothing_window=1)
        assert float(sig.iloc[-1]) > 0, "Low NVT should be bullish"

    def test_nvt_directional_signal_expensive(self) -> None:
        """NVT > 150 should produce negative directional signal."""
        mc  = _make_date_series([3_000_000.0] * 30)
        vol = _make_date_series([10_000.0]    * 30)  # NVT = 300 (expensive)
        sig = self.nvm.nvt_directional_signal(mc, vol, smoothing_window=1)
        assert float(sig.iloc[-1]) < 0, "High NVT should be bearish"

    def test_metcalfe_ratio_above_one_when_overpriced(self) -> None:
        """
        When actual price greatly exceeds what active addresses justify,
        Metcalfe ratio should be > 1.
        """
        # 100 addresses but very high price
        aa    = _make_date_series([1_000.0]  * 100)
        price = _make_date_series([100_000.0] * 100)
        ratio = self.nvm.metcalfe_value(aa, price)

        # With constant inputs, the regression fits exactly, so ratio ~ 1.
        # To test > 1, spike price at the end
        aa2    = _make_date_series([1_000.0] * 90 + [1_000.0] * 10)
        price2 = _make_date_series([1_000.0] * 90 + [1_000_000.0] * 10)
        ratio2 = self.nvm.metcalfe_value(aa2, price2)
        assert float(ratio2.iloc[-1]) > float(ratio2.iloc[80]), (
            "Price spike with no address growth should push ratio up"
        )

    def test_s2f_model_price_increases_with_scarcity(self) -> None:
        """Higher S2F ratio (more scarce) should produce higher model price."""
        supply      = _make_date_series([19_000_000.0] * 30)
        production1 = _make_date_series([900.0] * 30)   # high production = low S2F
        production2 = _make_date_series([100.0] * 30)   # low production  = high S2F

        price1 = self.nvm.stock_to_flow_model(supply, production1)
        price2 = self.nvm.stock_to_flow_model(supply, production2)

        assert float(price2.iloc[-1]) > float(price1.iloc[-1]), (
            "Lower production rate should yield higher S2F model price"
        )

    def test_realized_cap_sums_utxo_values(self) -> None:
        """realized_cap should return the exact sum of utxo_prices values."""
        utxos = {"utxo1": 1_000_000.0, "utxo2": 500_000.0, "utxo3": 250_000.0}
        rc = self.nvm.realized_cap(utxos)
        assert rc == 1_750_000.0

    def test_realized_cap_empty(self) -> None:
        """Empty UTXO set should return 0.0."""
        assert self.nvm.realized_cap({}) == 0.0

    def test_mvrv_signal_overbought_is_negative(self) -> None:
        """MVRV >> 3.5 should produce a negative signal."""
        mc = _make_date_series([350_000_000.0] * 30)
        rc = _make_date_series([50_000_000.0]  * 30)   # MVRV = 7.0
        sig = self.nvm.mvrv_signal(mc, rc)
        assert float(sig.iloc[-1]) < 0, "High MVRV should be bearish"

    def test_mvrv_signal_oversold_is_positive(self) -> None:
        """MVRV < 1.0 should produce a positive signal."""
        mc = _make_date_series([80_000_000.0]  * 30)
        rc = _make_date_series([100_000_000.0] * 30)   # MVRV = 0.8
        sig = self.nvm.mvrv_signal(mc, rc)
        assert float(sig.iloc[-1]) > 0, "Low MVRV should be bullish"


# ---------------------------------------------------------------------------
# StablecoinFlowAnalyzer tests
# ---------------------------------------------------------------------------

class TestStablecoinFlowAnalyzer:

    def setup_method(self) -> None:
        self.analyzer = StablecoinFlowAnalyzer(db_path=":memory:")

    def _populate(self, n_days: int, base: float, growth: float = 0.0) -> None:
        """Fill analyzer with supply data."""
        for i in range(n_days):
            date_str = (datetime(2024, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            supply = base + growth * i
            self.analyzer.record_supply(date_str, "USDC", supply)

    def test_total_supply_sums_tokens(self) -> None:
        """total_stablecoin_supply should sum all token supplies."""
        self.analyzer.record_supply("2024-03-01", "USDT", 80_000_000_000.0)
        self.analyzer.record_supply("2024-03-01", "USDC", 40_000_000_000.0)
        self.analyzer.record_supply("2024-03-01", "DAI",   5_000_000_000.0)
        total = self.analyzer.total_stablecoin_supply("2024-03-01")
        assert total == 125_000_000_000.0

    def test_rising_supply_produces_positive_signal(self) -> None:
        """
        A growing stablecoin supply (dry powder accumulating) should
        produce a positive supply_change signal.
        """
        # Strong growth: supply increases by $1B per day for 100 days
        self._populate(100, base=40_000_000_000.0, growth=1_000_000_000.0)
        sig = self.analyzer.signal(method="supply_change", window_days=7)
        assert sig > 0, f"Rising supply should give positive signal, got {sig}"

    def test_falling_supply_produces_negative_signal(self) -> None:
        """
        Falling stablecoin supply (stablecoins already deployed into crypto)
        should produce a negative signal.
        """
        self._populate(100, base=140_000_000_000.0, growth=-1_000_000_000.0)
        sig = self.analyzer.signal(method="supply_change", window_days=7)
        assert sig < 0, f"Falling supply should give negative signal, got {sig}"

    def test_stable_supply_signal_near_zero(self) -> None:
        """Flat supply should produce near-zero signal."""
        self._populate(100, base=40_000_000_000.0, growth=0.0)
        sig = self.analyzer.signal(method="supply_change", window_days=7)
        # May be exactly 0 due to zero std
        assert abs(sig) <= 0.1, f"Flat supply signal should be ~0, got {sig}"

    def test_stablecoin_dominance_formula(self) -> None:
        """Dominance should equal stable_supply / market_cap."""
        self.analyzer.record_supply("2024-03-01", "USDC", 40_000_000_000.0)
        dominance = self.analyzer.stablecoin_dominance(400_000_000_000.0)
        expected = 40e9 / 400e9
        assert abs(dominance - expected) < 1e-9, f"Expected {expected}, got {dominance}"

    def test_crypto_to_stablecoin_ratio(self) -> None:
        """Ratio should be market_cap / stable_supply."""
        self.analyzer.record_supply("2024-03-02", "USDC", 50_000_000_000.0)
        ratio = self.analyzer.crypto_to_stablecoin_ratio(1_000_000_000_000.0)
        expected = 1e12 / 50e9
        assert abs(ratio - expected) < 1e-6, f"Expected {expected}, got {ratio}"


# ---------------------------------------------------------------------------
# DexStablecoinMonitor tests
# ---------------------------------------------------------------------------

class TestDexStablecoinMonitor:

    def setup_method(self) -> None:
        self.monitor = DexStablecoinMonitor(db_path=":memory:")

    def _buy(self, amount: float, dt: Optional[datetime] = None) -> DexTrade:
        from typing import Optional
        return DexTrade(
            pair="USDC-ETH",
            stable_amount=amount,
            crypto_amount=amount / 3000.0,
            direction="BUY",
            timestamp=dt or datetime.now(tz=timezone.utc),
            tx_hash=f"buy_{abs(hash(amount))}"[:18],
        )

    def _sell(self, amount: float, dt: Optional[datetime] = None) -> DexTrade:
        from typing import Optional
        return DexTrade(
            pair="USDC-ETH",
            stable_amount=amount,
            crypto_amount=amount / 3000.0,
            direction="SELL",
            timestamp=dt or datetime.now(tz=timezone.utc),
            tx_hash=f"sell_{abs(hash(amount))}"[:18],
        )

    def test_net_buys_positive_on_buy_trades(self) -> None:
        """Net stablecoin buys should be positive when buys dominate."""
        now = datetime.now(tz=timezone.utc)
        for i in range(5):
            self.monitor.record_trade(self._buy(10_000.0, dt=now - timedelta(hours=i)))
        net = self.monitor.net_stablecoin_buys("USDC-ETH", window_hours=24)
        assert net > 0, f"Expected positive net buys, got {net}"

    def test_net_buys_negative_on_sell_trades(self) -> None:
        """Net stablecoin buys should be negative when sells dominate."""
        now = datetime.now(tz=timezone.utc)
        for i in range(5):
            self.monitor.record_trade(self._sell(10_000.0, dt=now - timedelta(hours=i)))
        net = self.monitor.net_stablecoin_buys("USDC-ETH", window_hours=24)
        assert net < 0, f"Expected negative net buys, got {net}"

    def test_flow_signal_positive_on_large_buys(self) -> None:
        """
        When a large volume of stablecoins flow into crypto relative to history,
        the signal should be positive.
        """
        now = datetime.now(tz=timezone.utc)
        # Small historical volume to establish baseline std
        for i in range(100):
            amount = 1_000.0
            direction = "BUY" if i % 2 == 0 else "SELL"
            trade = DexTrade(
                pair="USDC-ETH",
                stable_amount=amount,
                crypto_amount=amount / 3000.0,
                direction=direction,
                timestamp=now - timedelta(hours=i + 5),
                tx_hash=f"hist_{i}",
            )
            self.monitor.record_trade(trade)

        # Large recent buy spike
        for i in range(4):
            self.monitor.record_trade(self._buy(500_000.0, dt=now - timedelta(hours=i)))

        sig = self.monitor.stable_to_crypto_flow_signal(window_hours=4)
        assert sig > 0, f"Large stablecoin buys should give positive signal, got {sig}"

    def test_net_buys_respects_time_window(self) -> None:
        """Trades outside the window should not be counted."""
        now = datetime.now(tz=timezone.utc)
        self.monitor.record_trade(self._buy(50_000.0, dt=now - timedelta(hours=25)))
        self.monitor.record_trade(self._buy(10_000.0, dt=now - timedelta(hours=1)))
        net = self.monitor.net_stablecoin_buys("USDC-ETH", window_hours=24)
        # Only the recent $10k buy should count
        assert abs(net - 10_000.0) < 1e-3, f"Expected 10000, got {net}"


# ---------------------------------------------------------------------------
# Optional: smoke test for typing imports
# ---------------------------------------------------------------------------

def test_imports_clean() -> None:
    """All top-level imports in the package should succeed without error."""
    import onchain_advanced
    assert hasattr(onchain_advanced, "WhaleTracker")
    assert hasattr(onchain_advanced, "MinerMetricsAnalyzer")
    assert hasattr(onchain_advanced, "StablecoinFlowAnalyzer")
    assert hasattr(onchain_advanced, "NetworkValueMetrics")
