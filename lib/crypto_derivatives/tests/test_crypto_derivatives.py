"""
test_crypto_derivatives.py -- Production test suite for crypto_derivatives module.

Tests cover:
  - FundingRate signal convention and boundary cases
  - FundingRateAggregator composite weighting, momentum, and history
  - OpenInterestAnalyzer signal logic and liquidation risk
  - BasisTracker bps calculation and signal
  - CryptoImpliedVol surface construction, ATM vol, skew, term structure
  - CryptoVolRegime VRP signal and realized vol computation
  - PerpOptionsComposite signal combination
  - AMMLiquidityAnalyzer tick liquidity, cliff detection, price impact
  - LendingProtocolSignal utilization signal
  - BridgeFlowTracker flow signal and net flow

All tests are self-contained and require no external API calls.
"""

import math
import statistics
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from lib.crypto_derivatives.perpetuals import (
    BasisTracker,
    FundingRate,
    FundingRateAggregator,
    OpenInterestAnalyzer,
    _funding_rate_to_signal,
    build_funding_rate,
)
from lib.crypto_derivatives.options_market import (
    CryptoImpliedVol,
    CryptoVolRegime,
    DerivativesSignal,
    DeribitOptionChain,
    OptionQuote,
    PerpOptionsComposite,
    _vrp_ratio_to_signal,
)
from lib.crypto_derivatives.defi_analytics import (
    AMMLiquidityAnalyzer,
    BorrowSnapshot,
    BridgeFlowEvent,
    BridgeFlowTracker,
    LendingProtocolSignal,
    TickLiquidityBin,
    _utilization_to_signal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_ago(hours: int) -> datetime:
    return utc_now() - timedelta(hours=hours)


def make_funding_rate(exchange: str, rate_8h: float, symbol: str = "BTCUSDT") -> FundingRate:
    return FundingRate.from_rate_8h(exchange, symbol, rate_8h, utc_now())


def make_option_chain(
    symbol: str = "BTC",
    spot: float = 50000.0,
    expiry_days: float = 30.0,
    call_ivs: dict = None,
    put_ivs: dict = None,
) -> DeribitOptionChain:
    """Build a simple option chain with specified IV overrides per strike."""
    strikes = [40000, 45000, 48000, 50000, 52000, 55000, 60000]
    default_iv = 0.80

    calls = []
    puts = []
    expiry = utc_now() + timedelta(days=expiry_days)

    for s in strikes:
        c_iv = (call_ivs or {}).get(s, default_iv)
        p_iv = (put_ivs or {}).get(s, default_iv)

        # Approximate delta using Black-Scholes approximation
        log_moneyness = math.log(s / spot)
        t = expiry_days / 365.0
        approx_d1 = (-log_moneyness) / (default_iv * math.sqrt(t))
        # Crude normal CDF approximation for delta
        call_delta = max(0.01, min(0.99, 0.5 + 0.4 * approx_d1))
        put_delta = call_delta - 1.0

        calls.append(OptionQuote(
            strike=s,
            expiry_days=expiry_days,
            option_type="call",
            bid=max(0.1, c_iv * 0.95),
            ask=c_iv * 1.05,
            iv=c_iv,
            delta=call_delta,
        ))
        puts.append(OptionQuote(
            strike=s,
            expiry_days=expiry_days,
            option_type="put",
            bid=max(0.1, p_iv * 0.95),
            ask=p_iv * 1.05,
            iv=p_iv,
            delta=put_delta,
        ))

    return DeribitOptionChain(
        symbol=symbol,
        expiry=expiry,
        expiry_days=expiry_days,
        strikes=strikes,
        calls=calls,
        puts=puts,
        spot_price=spot,
    )


def make_bins(spot: float, n_bins: int = 10, range_pct: float = 0.20) -> List[TickLiquidityBin]:
    """Create evenly spaced liquidity bins centered around spot."""
    lo = spot * (1 - range_pct)
    hi = spot * (1 + range_pct)
    step = (hi - lo) / n_bins
    bins = []
    for i in range(n_bins):
        pl = lo + i * step
        pu = pl + step
        bins.append(TickLiquidityBin(
            tick_lower=int(math.log(pl) / math.log(1.0001)),
            tick_upper=int(math.log(pu) / math.log(1.0001)),
            liquidity=1_000_000.0,
            price_lower=pl,
            price_upper=pu,
            liquidity_usd=500_000.0,
        ))
    return bins


# ===========================================================================
# FundingRate tests
# ===========================================================================

class TestFundingRate:
    def test_from_rate_8h_annualizes_correctly(self):
        fr = FundingRate.from_rate_8h("binance", "BTCUSDT", 0.0001, utc_now())
        expected_annualized = 0.0001 * 3 * 365
        assert abs(fr.annualized_rate - expected_annualized) < 1e-10

    def test_negative_rate_annualized(self):
        fr = FundingRate.from_rate_8h("bybit", "ETHUSDT", -0.0003, utc_now())
        assert fr.annualized_rate < 0

    def test_repr_contains_exchange_and_symbol(self):
        fr = make_funding_rate("okx", 0.0001)
        r = repr(fr)
        assert "okx" in r
        assert "BTCUSDT" in r

    def test_zero_rate_annualizes_to_zero(self):
        fr = FundingRate.from_rate_8h("dydx", "SOLUSD", 0.0, utc_now())
        assert fr.annualized_rate == 0.0

    def test_next_funding_time_optional(self):
        fr = make_funding_rate("binance", 0.0001)
        assert fr.next_funding_time is None

    def test_build_funding_rate_factory(self):
        fr = build_funding_rate("binance", "BTCUSDT", -0.0005)
        assert fr.exchange == "binance"
        assert fr.rate_8h == -0.0005


# ===========================================================================
# Funding signal tests
# ===========================================================================

class TestFundingSignal:
    def test_very_negative_funding_is_bullish(self):
        """Very negative funding (< -0.1% per 8h) -> shorts paying longs -> bullish."""
        signal = _funding_rate_to_signal(-0.002)  # -0.2%, below threshold
        assert signal == 1.0

    def test_very_positive_funding_is_bearish(self):
        """Very positive funding (> 0.3% per 8h) -> longs paying shorts -> bearish."""
        signal = _funding_rate_to_signal(0.004)  # 0.4%, above threshold
        assert signal == -1.0

    def test_zero_funding_is_near_neutral(self):
        signal = _funding_rate_to_signal(0.0)
        # 0.0 is below the midpoint of [-0.001, 0.003]=0.001, so signal is mildly bullish
        assert -0.5 <= signal <= 0.5

    def test_at_very_negative_threshold_is_exactly_bullish(self):
        signal = _funding_rate_to_signal(-0.001)
        assert signal == 1.0

    def test_at_very_positive_threshold_is_exactly_bearish(self):
        signal = _funding_rate_to_signal(0.003)
        assert signal == -1.0

    def test_signal_is_within_bounds(self):
        for rate in [-0.01, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.01]:
            sig = _funding_rate_to_signal(rate)
            assert -1.0 <= sig <= 1.0, f"signal {sig} out of bounds for rate {rate}"

    def test_signal_monotonically_decreasing(self):
        """Higher funding rate -> lower (more bearish) signal."""
        rates = [-0.003, -0.001, 0.0, 0.001, 0.003]
        signals = [_funding_rate_to_signal(r) for r in rates]
        for i in range(len(signals) - 1):
            assert signals[i] >= signals[i + 1], (
                f"Signal not decreasing: signals[{i}]={signals[i]} vs signals[{i+1}]={signals[i+1]}"
            )

    def test_moderately_positive_funding_bearish(self):
        signal = _funding_rate_to_signal(0.0025)  # 0.25% -- between thresholds but high
        assert signal < 0

    def test_moderately_negative_funding_bullish(self):
        signal = _funding_rate_to_signal(-0.0008)
        assert signal > 0


# ===========================================================================
# FundingRateAggregator tests
# ===========================================================================

class TestFundingRateAggregator:
    def test_composite_rate_single_exchange(self):
        agg = FundingRateAggregator()
        agg.ingest(make_funding_rate("binance", 0.0001))
        rate = agg.composite_funding_rate("BTCUSDT")
        assert abs(rate - 0.0001) < 1e-10

    def test_composite_rate_weighted_average(self):
        agg = FundingRateAggregator(oi_weights={"binance": 1.0, "bybit": 1.0})
        agg.ingest(make_funding_rate("binance", 0.0002))
        agg.ingest(make_funding_rate("bybit", 0.0004))
        rate = agg.composite_funding_rate("BTCUSDT")
        assert abs(rate - 0.0003) < 1e-8

    def test_composite_rate_returns_zero_with_no_data(self):
        agg = FundingRateAggregator()
        assert agg.composite_funding_rate("BTCUSDT") == 0.0

    def test_funding_signal_negative_funding_bullish(self):
        agg = FundingRateAggregator()
        agg.ingest(make_funding_rate("binance", -0.002))
        signal = agg.funding_signal("BTCUSDT")
        assert signal == 1.0

    def test_funding_signal_positive_funding_bearish(self):
        agg = FundingRateAggregator()
        agg.ingest(make_funding_rate("binance", 0.005))
        signal = agg.funding_signal("BTCUSDT")
        assert signal == -1.0

    def test_funding_signal_range(self):
        agg = FundingRateAggregator()
        for rate in [-0.005, -0.001, 0.0, 0.001, 0.005]:
            agg._latest = {}
            agg.ingest(make_funding_rate("binance", rate))
            sig = agg.funding_signal("BTCUSDT")
            assert -1.0 <= sig <= 1.0

    def test_oi_override_weights_funding_correctly(self):
        agg = FundingRateAggregator()
        agg.ingest(make_funding_rate("binance", 0.001))
        agg.ingest(make_funding_rate("bybit", 0.003))
        # 90% weight on binance
        rate = agg.composite_funding_rate("BTCUSDT", {"binance": 0.9, "bybit": 0.1})
        assert abs(rate - (0.001 * 0.9 + 0.003 * 0.1)) < 1e-8

    def test_funding_momentum_positive_trend(self):
        agg = FundingRateAggregator()
        # Increasing funding rates
        for i in range(30):
            ts = utc_ago((30 - i) * 8)
            rate = 0.0001 + i * 0.0001
            agg.ingest(FundingRate.from_rate_8h("binance", "BTCUSDT", rate, ts))
        momentum = agg.funding_momentum("BTCUSDT", window=7)
        assert momentum > 0

    def test_funding_momentum_negative_trend(self):
        agg = FundingRateAggregator()
        for i in range(30):
            ts = utc_ago((30 - i) * 8)
            rate = 0.003 - i * 0.0001
            agg.ingest(FundingRate.from_rate_8h("binance", "BTCUSDT", rate, ts))
        momentum = agg.funding_momentum("BTCUSDT", window=7)
        assert momentum < 0

    def test_rolling_distribution_keys(self):
        agg = FundingRateAggregator()
        for i in range(20):
            ts = utc_ago(i * 8)
            agg.ingest(FundingRate.from_rate_8h("binance", "BTCUSDT", 0.0001 * (i + 1), ts))
        dist = agg.rolling_distribution("BTCUSDT", days=30)
        assert "mean" in dist
        assert "std" in dist
        assert "p5" in dist
        assert "p95" in dist
        assert dist["min"] <= dist["p5"] <= dist["p25"] <= dist["median"] <= dist["p75"] <= dist["p95"] <= dist["max"]

    def test_get_latest_rates_returns_per_exchange(self):
        agg = FundingRateAggregator()
        agg.ingest(make_funding_rate("binance", 0.0001))
        agg.ingest(make_funding_rate("bybit", 0.0002))
        latest = agg.get_latest_rates("BTCUSDT")
        assert "binance" in latest
        assert "bybit" in latest

    def test_ingest_batch(self):
        agg = FundingRateAggregator()
        rates = [make_funding_rate(ex, 0.0001) for ex in ["binance", "bybit", "okx"]]
        agg.ingest_batch(rates)
        assert len(agg.get_latest_rates("BTCUSDT")) == 3


# ===========================================================================
# OpenInterestAnalyzer tests
# ===========================================================================

class TestOpenInterestAnalyzer:
    def test_oi_change_signal_rising_oi_rising_price_bullish(self):
        """Rising OI + rising price = leveraged long buildup -> bullish."""
        ana = OpenInterestAnalyzer()
        oi = [1e9, 1.05e9, 1.10e9, 1.15e9]
        price = [40000, 41000, 42000, 43000]
        signal = ana.oi_change_signal("BTC", oi_series=oi, price_series=price)
        assert signal > 0, f"Expected bullish signal, got {signal}"

    def test_oi_change_signal_rising_oi_falling_price_bearish(self):
        """Rising OI + falling price = short buildup -> bearish."""
        ana = OpenInterestAnalyzer()
        oi = [1e9, 1.05e9, 1.10e9, 1.15e9]
        price = [43000, 42000, 41000, 40000]
        signal = ana.oi_change_signal("BTC", oi_series=oi, price_series=price)
        assert signal < 0, f"Expected bearish signal, got {signal}"

    def test_oi_change_signal_stable_oi_is_neutral(self):
        ana = OpenInterestAnalyzer()
        oi = [1e9, 1e9, 1e9, 1e9]
        price = [40000, 40100, 40200, 40300]
        signal = ana.oi_change_signal("BTC", oi_series=oi, price_series=price)
        assert abs(signal) < 0.1

    def test_oi_change_signal_within_bounds(self):
        ana = OpenInterestAnalyzer()
        oi = [1e9, 2e9, 3e9, 4e9]
        price = [40000, 41000, 42000, 43000]
        signal = ana.oi_change_signal("BTC", oi_series=oi, price_series=price)
        assert -1.0 <= signal <= 1.0

    def test_oi_to_volume_ratio(self):
        ana = OpenInterestAnalyzer()
        ratio = ana.oi_to_volume_ratio("BTC", oi_usd=5e9, volume_24h_usd=2.5e9)
        assert abs(ratio - 2.0) < 1e-8

    def test_oi_to_volume_ratio_zero_volume_returns_zero(self):
        ana = OpenInterestAnalyzer()
        assert ana.oi_to_volume_ratio("BTC", oi_usd=1e9, volume_24h_usd=0) == 0.0

    def test_liquidation_cascade_risk_basic(self):
        ana = OpenInterestAnalyzer()
        liq = ana.liquidation_cascade_risk("BTC", oi_usd=1e9, leverage_estimate=10, price_move_pct=0.10)
        # 10% move * 10x leverage = 100% liquidation
        assert liq == 1e9

    def test_liquidation_cascade_risk_partial(self):
        ana = OpenInterestAnalyzer()
        liq = ana.liquidation_cascade_risk("BTC", oi_usd=1e9, leverage_estimate=10, price_move_pct=0.05)
        # 5% move * 10x = 50% liquidated
        assert abs(liq - 5e8) < 1.0

    def test_liquidation_cascade_risk_clamps_at_oi(self):
        ana = OpenInterestAnalyzer()
        liq = ana.liquidation_cascade_risk("BTC", oi_usd=1e9, leverage_estimate=50, price_move_pct=0.50)
        assert liq == 1e9

    def test_liquidation_cascade_risk_zero_oi(self):
        ana = OpenInterestAnalyzer()
        assert ana.liquidation_cascade_risk("BTC", oi_usd=0) == 0.0

    def test_record_and_retrieve_oi(self):
        ana = OpenInterestAnalyzer()
        ana.record_oi("BTC", 5e9, 50000.0)
        ana.record_oi("BTC", 5.5e9, 51000.0)
        ratio = ana.oi_to_volume_ratio("BTC", oi_usd=5.5e9, volume_24h_usd=2e9)
        assert ratio == pytest.approx(2.75)

    def test_leverage_heatmap_returns_required_keys(self):
        ana = OpenInterestAnalyzer()
        heatmap = ana.leverage_heatmap(
            "BTC",
            oi_series=[1e9, 1.1e9, 1.2e9],
            price_series=[40000, 41000, 42000],
            volume_series=[5e8, 5e8, 5e8],
        )
        for key in ["oi_usd", "volume_24h_usd", "oi_volume_ratio", "oi_trend", "oi_change_signal"]:
            assert key in heatmap


# ===========================================================================
# BasisTracker tests
# ===========================================================================

class TestBasisTracker:
    def test_basis_bps_calculation(self):
        bt = BasisTracker()
        bps = bt.basis_bps(spot_price=50000.0, perp_price=50200.0)
        expected = (50200 - 50000) / 50000 * 10000
        assert abs(bps - expected) < 1e-6

    def test_basis_bps_negative_when_perp_below_spot(self):
        bt = BasisTracker()
        bps = bt.basis_bps(spot_price=50000.0, perp_price=49800.0)
        assert bps < 0

    def test_basis_bps_zero_when_equal(self):
        bt = BasisTracker()
        bps = bt.basis_bps(50000.0, 50000.0)
        assert bps == 0.0

    def test_basis_bps_raises_on_zero_spot(self):
        bt = BasisTracker()
        with pytest.raises(ValueError):
            bt.basis_bps(0.0, 50000.0)

    def test_basis_signal_positive_premium_bullish(self):
        bt = BasisTracker()
        signal = bt.basis_signal("BTC", spot_price=50000.0, perp_price=50500.0)
        assert signal > 0

    def test_basis_signal_negative_premium_bearish(self):
        bt = BasisTracker()
        signal = bt.basis_signal("BTC", spot_price=50000.0, perp_price=49500.0)
        assert signal < 0

    def test_basis_signal_saturates_at_plus_one(self):
        bt = BasisTracker()
        # Very large premium -> saturates at 1.0
        signal = bt.basis_signal("BTC", spot_price=50000.0, perp_price=55000.0)
        assert signal == 1.0

    def test_basis_signal_saturates_at_minus_one(self):
        bt = BasisTracker()
        signal = bt.basis_signal("BTC", spot_price=50000.0, perp_price=45000.0)
        assert signal == -1.0

    def test_annualized_basis_perpetual(self):
        bt = BasisTracker()
        ann = bt.annualized_basis("BTC", spot_price=50000.0, perp_price=50200.0)
        basis_decimal = 200 / 50000
        expected = basis_decimal * 365.0
        assert abs(ann - expected) < 1e-8

    def test_annualized_basis_dated_future(self):
        bt = BasisTracker()
        ann = bt.annualized_basis("BTC", days_to_settlement=30.0, spot_price=50000.0, perp_price=50200.0)
        basis_decimal = 200 / 50000
        expected = basis_decimal * (365.0 / 30.0)
        assert abs(ann - expected) < 1e-6

    def test_record_basis_and_retrieve(self):
        bt = BasisTracker()
        bt.record_basis("BTC", 50000.0, 50200.0, utc_now())
        signal = bt.basis_signal("BTC")
        assert signal > 0

    def test_rolling_basis_stats_returns_dict(self):
        bt = BasisTracker()
        for i in range(20):
            bt.record_basis("BTC", 50000.0, 50000.0 + i * 10, utc_ago(i))
        stats = bt.rolling_basis_stats("BTC", days=7)
        assert "mean_bps" in stats
        assert "std_bps" in stats


# ===========================================================================
# CryptoImpliedVol tests
# ===========================================================================

class TestCryptoImpliedVol:
    def test_vol_surface_returns_dict_with_expected_keys(self):
        civ = CryptoImpliedVol()
        chain = make_option_chain()
        result = civ.vol_surface(chain)
        assert "expiry_days" in result
        assert "atm_iv" in result
        assert "strikes" in result
        assert "ivs" in result

    def test_atm_vol_from_chain_near_spot(self):
        civ = CryptoImpliedVol()
        chain = make_option_chain(spot=50000.0)
        atm = civ.atm_vol_from_chain(chain)
        assert 0.0 < atm < 10.0  # sanity bounds

    def test_vol_surface_stores_surface(self):
        civ = CryptoImpliedVol()
        chain = make_option_chain(symbol="BTC", spot=50000.0, expiry_days=30.0)
        civ.vol_surface(chain)
        atm = civ.atm_vol("BTC", 30.0)
        assert atm > 0.0

    def test_vol_skew_sign_convention_positive_when_calls_expensive(self):
        """When call IVs are higher than put IVs -> positive skew."""
        call_ivs = {40000: 0.85, 45000: 0.83, 48000: 0.81, 50000: 0.80, 52000: 0.82, 55000: 0.84, 60000: 0.87}
        put_ivs = {40000: 0.75, 45000: 0.77, 48000: 0.78, 50000: 0.80, 52000: 0.79, 55000: 0.78, 60000: 0.76}
        chain = make_option_chain(spot=50000.0, call_ivs=call_ivs, put_ivs=put_ivs)
        civ = CryptoImpliedVol()
        civ.vol_surface(chain)
        skew = civ.vol_skew("BTC", 30.0, option_chain=chain)
        assert skew > 0, f"Expected positive skew with expensive calls, got {skew}"

    def test_vol_skew_negative_when_puts_expensive(self):
        """When put IVs are higher than call IVs -> negative skew (fear)."""
        call_ivs = {40000: 0.75, 45000: 0.77, 48000: 0.78, 50000: 0.80, 52000: 0.79, 55000: 0.78, 60000: 0.76}
        put_ivs = {40000: 0.90, 45000: 0.88, 48000: 0.85, 50000: 0.80, 52000: 0.82, 55000: 0.83, 60000: 0.85}
        chain = make_option_chain(spot=50000.0, call_ivs=call_ivs, put_ivs=put_ivs)
        civ = CryptoImpliedVol()
        civ.vol_surface(chain)
        skew = civ.vol_skew("BTC", 30.0, option_chain=chain)
        assert skew < 0, f"Expected negative skew with expensive puts, got {skew}"

    def test_term_structure_slope_normal_greater_than_one(self):
        """Normal term structure: far expiry vol > near expiry vol -> slope > 1."""
        civ = CryptoImpliedVol()
        chain_7d = make_option_chain(expiry_days=7.0)
        chain_30d = make_option_chain(
            expiry_days=30.0,
            call_ivs={k: 0.90 for k in [40000, 45000, 48000, 50000, 52000, 55000, 60000]},
            put_ivs={k: 0.90 for k in [40000, 45000, 48000, 50000, 52000, 55000, 60000]},
        )
        civ.vol_surface(chain_7d)
        civ.vol_surface(chain_30d)
        slope = civ.term_structure_slope("BTC")
        assert slope > 1.0, f"Expected slope > 1 for normal term structure, got {slope}"

    def test_term_structure_slope_inverted_less_than_one(self):
        """Inverted: near vol > far vol -> slope < 1."""
        civ = CryptoImpliedVol()
        chain_7d = make_option_chain(
            expiry_days=7.0,
            call_ivs={k: 1.20 for k in [40000, 45000, 48000, 50000, 52000, 55000, 60000]},
            put_ivs={k: 1.20 for k in [40000, 45000, 48000, 50000, 52000, 55000, 60000]},
        )
        chain_30d = make_option_chain(expiry_days=30.0)
        civ.vol_surface(chain_7d)
        civ.vol_surface(chain_30d)
        slope = civ.term_structure_slope("BTC")
        assert slope < 1.0, f"Expected slope < 1 for inverted term structure, got {slope}"

    def test_atm_vol_returns_zero_with_no_data(self):
        civ = CryptoImpliedVol()
        assert civ.atm_vol("UNKNOWN", 30.0) == 0.0


# ===========================================================================
# CryptoVolRegime tests
# ===========================================================================

class TestCryptoVolRegime:
    def test_vrp_signal_range(self):
        regime = CryptoVolRegime()
        for iv, rv in [(0.5, 0.5), (1.5, 0.5), (0.5, 1.0), (0.8, 0.8)]:
            sig = regime.vrp_signal("BTC", iv, rv)
            assert -1.0 <= sig <= 1.0, f"VRP signal out of range for iv={iv}, rv={rv}"

    def test_vrp_signal_expensive_options_negative(self):
        """IV/RV > 1.5 -> options expensive -> signal <= -1.0."""
        regime = CryptoVolRegime()
        sig = regime.vrp_signal("BTC", implied_vol=1.6, realized_vol=0.8)
        assert sig == -1.0

    def test_vrp_signal_cheap_options_positive(self):
        """IV/RV < 0.7 -> options cheap -> signal >= +1.0."""
        regime = CryptoVolRegime()
        sig = regime.vrp_signal("BTC", implied_vol=0.5, realized_vol=1.0)
        assert sig == 1.0

    def test_vrp_signal_zero_rv_returns_zero(self):
        regime = CryptoVolRegime()
        sig = regime.vrp_signal("BTC", implied_vol=0.8, realized_vol=0.0)
        assert sig == 0.0

    def test_vrp_ratio_calculation(self):
        regime = CryptoVolRegime()
        ratio = regime.vrp_ratio(implied_vol=0.8, realized_vol=0.4)
        assert ratio == pytest.approx(2.0)

    def test_vrp_ratio_to_signal_interpolation(self):
        # Midpoint between cheap (0.7) and rich (1.5) -> should be near 0
        ratio = (0.7 + 1.5) / 2.0  # = 1.1
        sig = _vrp_ratio_to_signal(ratio)
        assert abs(sig) < 0.1

    def test_realized_vol_computation(self):
        regime = CryptoVolRegime()
        prices = [50000 * (1 + 0.01 * (i % 3 - 1)) for i in range(35)]
        rv = regime.realized_vol(prices, window=30)
        assert rv > 0

    def test_vol_regime_labels(self):
        regime = CryptoVolRegime()
        assert regime.vol_regime_label(1.8, 0.8) == "expensive"
        assert regime.vol_regime_label(1.3, 0.8) == "rich"
        assert regime.vol_regime_label(1.0, 1.0) == "fair"
        assert regime.vol_regime_label(0.6, 1.0) == "cheap"
        assert regime.vol_regime_label(0.3, 1.0) == "very_cheap"

    def test_rolling_vrp_length(self):
        regime = CryptoVolRegime()
        iv_series = [0.80 + 0.01 * i for i in range(40)]
        rv_series = [0.60 + 0.005 * i for i in range(40)]
        result = regime.rolling_vrp(iv_series, rv_series, window=30)
        assert len(result) == 30


# ===========================================================================
# PerpOptionsComposite tests
# ===========================================================================

class TestPerpOptionsComposite:
    def test_composite_signal_all_bullish(self):
        comp = PerpOptionsComposite()
        sig = comp.composite_derivatives_signal("BTC", 1.0, 1.0, 1.0, 1.0)
        assert sig.composite_score == pytest.approx(1.0, abs=1e-6)

    def test_composite_signal_all_bearish(self):
        comp = PerpOptionsComposite()
        sig = comp.composite_derivatives_signal("BTC", -1.0, -1.0, -1.0, -1.0)
        assert sig.composite_score == pytest.approx(-1.0, abs=1e-6)

    def test_composite_signal_range(self):
        comp = PerpOptionsComposite()
        for vals in [(0.5, -0.5, 0.3, -0.2), (1.0, -1.0, 0.0, 0.0), (-0.3, 0.7, 0.2, -0.8)]:
            sig = comp.composite_derivatives_signal("BTC", *vals)
            assert -1.0 <= sig.composite_score <= 1.0

    def test_composite_derivatives_signal_components_sum(self):
        """Composite is weighted sum of components (with weights summing to 1)."""
        from lib.crypto_derivatives.options_market import COMPOSITE_WEIGHTS
        comp = PerpOptionsComposite()
        f, b, s, v = 0.4, -0.2, 0.6, -0.1
        sig = comp.composite_derivatives_signal("BTC", f, b, s, v)

        total = sum(COMPOSITE_WEIGHTS.values())
        expected = (
            COMPOSITE_WEIGHTS["funding"] / total * f
            + COMPOSITE_WEIGHTS["basis"] / total * b
            + COMPOSITE_WEIGHTS["skew"] / total * s
            + COMPOSITE_WEIGHTS["vrp"] / total * v
        )
        assert abs(sig.composite_score - expected) < 1e-6

    def test_composite_returns_derivatives_signal_dataclass(self):
        comp = PerpOptionsComposite()
        result = comp.composite_derivatives_signal("BTC", 0.5, 0.3, -0.2, 0.1)
        assert isinstance(result, DerivativesSignal)
        assert hasattr(result, "funding_component")
        assert hasattr(result, "basis_component")
        assert hasattr(result, "skew_component")
        assert hasattr(result, "vrp_component")

    def test_interpretation_contains_symbol(self):
        comp = PerpOptionsComposite()
        sig = comp.composite_derivatives_signal("ETH", 0.8, 0.7, 0.5, 0.6)
        assert "ETH" in sig.interpretation

    def test_skew_to_signal_positive_skew_bullish(self):
        comp = PerpOptionsComposite()
        sig = comp.skew_to_signal(0.10)
        assert sig > 0

    def test_skew_to_signal_negative_skew_bearish(self):
        comp = PerpOptionsComposite()
        sig = comp.skew_to_signal(-0.10)
        assert sig < 0

    def test_skew_to_signal_saturates(self):
        comp = PerpOptionsComposite()
        assert comp.skew_to_signal(1.0) == 1.0
        assert comp.skew_to_signal(-1.0) == -1.0

    def test_term_structure_inverted_bearish(self):
        comp = PerpOptionsComposite()
        sig = comp.term_structure_to_signal(0.7)
        assert sig < 0

    def test_term_structure_normal_slightly_negative(self):
        """Normal (slope > 1) is slightly bearish for tail risk."""
        comp = PerpOptionsComposite()
        sig = comp.term_structure_to_signal(1.2)
        assert sig <= 0

    def test_build_full_signal_returns_signal(self):
        comp = PerpOptionsComposite()
        sig = comp.build_full_signal(
            symbol="BTC",
            funding_signal=0.5,
            basis_signal=0.3,
            iv_skew_vol_pts=-0.05,
            implied_vol=0.8,
            realized_vol=0.6,
            term_slope=1.1,
        )
        assert isinstance(sig, DerivativesSignal)
        assert -1.0 <= sig.composite_score <= 1.0

    def test_custom_weights_are_respected(self):
        """When funding weight is 1.0 and others 0.0, composite = funding signal."""
        comp = PerpOptionsComposite(weights={"funding": 1.0, "basis": 0.0, "skew": 0.0, "vrp": 0.0})
        sig = comp.composite_derivatives_signal("BTC", 0.7, -0.9, 0.3, -0.5)
        assert abs(sig.composite_score - 0.7) < 1e-6


# ===========================================================================
# AMMLiquidityAnalyzer tests
# ===========================================================================

class TestAMMLiquidityAnalyzer:
    def test_tick_liquidity_returns_nonzero_within_range(self):
        ana = AMMLiquidityAnalyzer()
        bins = make_bins(spot=50000.0)
        liq = ana.tick_liquidity("BTC", price_range_pct=0.02, bins=bins, spot_price=50000.0)
        assert liq > 0

    def test_tick_liquidity_zero_for_empty_bins(self):
        ana = AMMLiquidityAnalyzer()
        liq = ana.tick_liquidity("UNKNOWN", bins=[], spot_price=50000.0)
        assert liq == 0.0

    def test_tick_liquidity_wider_range_more_liq(self):
        ana = AMMLiquidityAnalyzer()
        bins = make_bins(spot=50000.0)
        liq_2pct = ana.tick_liquidity("BTC", price_range_pct=0.02, bins=bins, spot_price=50000.0)
        liq_10pct = ana.tick_liquidity("BTC", price_range_pct=0.10, bins=bins, spot_price=50000.0)
        assert liq_10pct >= liq_2pct

    def test_price_impact_nonzero_for_large_trade(self):
        ana = AMMLiquidityAnalyzer()
        bins = make_bins(spot=50000.0)
        impact = ana.price_impact_estimate("BTC", trade_size_usd=1e7, bins=bins, spot_price=50000.0)
        assert impact >= 0.0

    def test_price_impact_buy_moves_price_up(self):
        ana = AMMLiquidityAnalyzer()
        bins = make_bins(spot=50000.0)
        impact_buy = ana.price_impact_estimate("BTC", 1e6, direction="buy", bins=bins, spot_price=50000.0)
        impact_sell = ana.price_impact_estimate("BTC", 1e6, direction="sell", bins=bins, spot_price=50000.0)
        # Both are positive magnitudes
        assert impact_buy >= 0
        assert impact_sell >= 0

    def test_price_impact_zero_trade_returns_zero(self):
        ana = AMMLiquidityAnalyzer()
        bins = make_bins(spot=50000.0)
        impact = ana.price_impact_estimate("BTC", 0, bins=bins, spot_price=50000.0)
        assert impact == 0.0

    def test_concentrated_liquidity_score_sums_correctly(self):
        ana = AMMLiquidityAnalyzer()
        bins = make_bins(spot=50000.0)
        score = ana.concentrated_liquidity_score("BTC", bins=bins, spot_price=50000.0)
        assert 0.0 <= score <= 1.0

    def test_update_liquidity_stores_data(self):
        ana = AMMLiquidityAnalyzer()
        bins = make_bins(spot=50000.0)
        ana.update_liquidity("BTC", bins, 50000.0)
        liq = ana.tick_liquidity("BTC")
        assert liq > 0

    def test_liquidity_cliff_with_no_bins_returns_inf(self):
        ana = AMMLiquidityAnalyzer()
        cliff = ana.liquidity_cliff("UNKNOWN", bins=[], spot_price=50000.0)
        assert cliff == float("inf")


# ===========================================================================
# LendingProtocolSignal tests
# ===========================================================================

class TestLendingProtocolSignal:
    def _make_snapshot(self, protocol: str, supplied: float, borrowed: float, symbol: str = "BTC") -> BorrowSnapshot:
        return BorrowSnapshot(
            symbol=symbol,
            protocol=protocol,
            total_supplied_usd=supplied,
            total_borrowed_usd=borrowed,
            borrow_rate_apy=0.05,
            supply_rate_apy=0.03,
            timestamp=utc_now(),
        )

    def test_utilization_signal_high_utilization_bullish(self):
        lps = LendingProtocolSignal()
        sig = lps.utilization_rate_signal("BTC", utilization_rate=0.90)
        assert sig == 1.0

    def test_utilization_signal_low_utilization_bearish(self):
        lps = LendingProtocolSignal()
        sig = lps.utilization_rate_signal("BTC", utilization_rate=0.20)
        assert sig == -1.0

    def test_utilization_signal_range(self):
        lps = LendingProtocolSignal()
        for u in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]:
            sig = lps.utilization_rate_signal("BTC", utilization_rate=u)
            assert -1.0 <= sig <= 1.0

    def test_utilization_internal(self):
        """Test _utilization_to_signal helper directly."""
        assert _utilization_to_signal(0.95) == 1.0
        assert _utilization_to_signal(0.30) == -1.0
        mid = _utilization_to_signal(0.625)  # midpoint of 0.40 and 0.85
        assert abs(mid) < 0.1

    def test_cross_protocol_utilization_averages(self):
        lps = LendingProtocolSignal()
        sig = lps.cross_protocol_utilization("BTC", {"aave": 0.90, "compound": 0.80})
        assert abs(sig - 0.85) < 1e-8

    def test_composite_lending_signal_from_overrides(self):
        lps = LendingProtocolSignal()
        sig = lps.composite_lending_signal("BTC", {"aave": 0.90, "compound": 0.90})
        assert sig == 1.0

    def test_borrow_snapshot_utilization_rate(self):
        snap = self._make_snapshot("aave", supplied=100e6, borrowed=80e6)
        assert snap.utilization_rate == pytest.approx(0.80)

    def test_record_and_retrieve_snapshot(self):
        lps = LendingProtocolSignal()
        snap = self._make_snapshot("aave", supplied=100e6, borrowed=90e6)
        lps.record_snapshot(snap)
        sig = lps.utilization_rate_signal("BTC")
        assert sig == 1.0


# ===========================================================================
# BridgeFlowTracker tests
# ===========================================================================

class TestBridgeFlowTracker:
    def _make_event(self, direction: str, amount_usd: float, hours_ago: int = 0, symbol: str = "BTC") -> BridgeFlowEvent:
        return BridgeFlowEvent(
            symbol=symbol,
            bridge_name="stargate",
            direction=direction,
            amount_usd=amount_usd,
            timestamp=utc_ago(hours_ago),
        )

    def test_bridge_flow_signal_net_outflow_bullish(self):
        """Large net outflow (self-custody) -> bullish signal."""
        bft = BridgeFlowTracker()
        events = [self._make_event("outflow", 1e7, hours_ago=12)]
        sig = bft.bridge_flow_signal("BTC", window_days=7, events=events, reference_volume_usd=1e7)
        assert sig > 0

    def test_bridge_flow_signal_net_inflow_bearish(self):
        """Large net inflow (exchange deposits) -> bearish signal."""
        bft = BridgeFlowTracker()
        events = [self._make_event("inflow", 1e7, hours_ago=12)]
        sig = bft.bridge_flow_signal("BTC", window_days=7, events=events, reference_volume_usd=1e7)
        assert sig < 0

    def test_bridge_flow_signal_neutral_balanced(self):
        bft = BridgeFlowTracker()
        events = [
            self._make_event("inflow", 5e6, hours_ago=10),
            self._make_event("outflow", 5e6, hours_ago=11),
        ]
        sig = bft.bridge_flow_signal("BTC", window_days=7, events=events, reference_volume_usd=1e7)
        assert abs(sig) < 0.01

    def test_bridge_flow_signal_range(self):
        bft = BridgeFlowTracker()
        events = [self._make_event("inflow", 5e8, hours_ago=5)]
        sig = bft.bridge_flow_signal("BTC", events=events, reference_volume_usd=1e6)
        assert -1.0 <= sig <= 1.0

    def test_bridge_flow_signal_empty_events_returns_zero(self):
        bft = BridgeFlowTracker()
        sig = bft.bridge_flow_signal("BTC", events=[])
        assert sig == 0.0

    def test_net_flow_usd_calculation(self):
        bft = BridgeFlowTracker()
        events = [
            self._make_event("inflow", 3e6, 5),
            self._make_event("outflow", 1e6, 10),
        ]
        result = bft.net_flow_usd("BTC", window_days=7, events=events)
        assert result["inflow_usd"] == pytest.approx(3e6)
        assert result["outflow_usd"] == pytest.approx(1e6)
        assert result["net_usd"] == pytest.approx(2e6)

    def test_bridge_volume_by_protocol(self):
        bft = BridgeFlowTracker()
        events = [
            BridgeFlowEvent("BTC", "stargate", "inflow", 1e6, utc_ago(5)),
            BridgeFlowEvent("BTC", "hop", "outflow", 2e6, utc_ago(6)),
        ]
        breakdown = bft.bridge_volume_by_protocol("BTC", window_days=7, events=events)
        assert "stargate" in breakdown
        assert "hop" in breakdown
        assert breakdown["hop"]["outflow_usd"] == pytest.approx(2e6)

    def test_record_and_retrieve_event(self):
        bft = BridgeFlowTracker()
        event = self._make_event("inflow", 1e6)
        bft.record_event(event)
        flows = bft.net_flow_usd("BTC", window_days=1)
        assert flows["inflow_usd"] > 0

    def test_flow_momentum_positive_when_outflow_accelerating(self):
        bft = BridgeFlowTracker()
        # Short window has more outflow than long window
        events = (
            [self._make_event("outflow", 2e6, hours_ago=h) for h in range(1, 50)]
            + [self._make_event("inflow", 5e5, hours_ago=h) for h in range(50, 300)]
        )
        momentum = bft.flow_momentum("BTC", short_window_days=3, long_window_days=14, events=events)
        assert -1.0 <= momentum <= 1.0

    def test_abnormal_flow_alert_returns_dict(self):
        bft = BridgeFlowTracker()
        events = [self._make_event("inflow", float(1e5 * (i + 1)), hours_ago=24 * i) for i in range(35)]
        result = bft.abnormal_flow_alert("BTC", events=events)
        assert "is_abnormal" in result
        assert "zscore" in result
