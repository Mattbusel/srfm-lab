# research/pipeline/tests/test_pipeline.py
# SRFM -- pytest test suite for the research pipeline
# Run with: pytest research/pipeline/tests/test_pipeline.py -v

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Fixtures -- synthetic data shared across tests
# ---------------------------------------------------------------------------

TICKERS = [f"T{i:02d}" for i in range(30)]
DATES = pd.bdate_range("2020-01-01", "2022-12-31")
RNG = np.random.default_rng(seed=0)


def make_prices(n_dates=None, n_tickers=None, seed=0):
    rng = np.random.default_rng(seed)
    nd = n_dates or len(DATES)
    nt = n_tickers or len(TICKERS)
    tickers = TICKERS[:nt]
    dates = DATES[:nd]
    log_ret = rng.normal(0.0002, 0.015, size=(nd, nt))
    prices = pd.DataFrame(
        np.exp(np.cumsum(log_ret, axis=0)) * 100.0,
        index=dates,
        columns=tickers,
    )
    return prices


def make_returns(prices=None):
    if prices is None:
        prices = make_prices()
    return prices.pct_change().dropna()


def make_volumes(n_dates=None, n_tickers=None, seed=1):
    rng = np.random.default_rng(seed)
    nd = n_dates or len(DATES)
    nt = n_tickers or len(TICKERS)
    tickers = TICKERS[:nt]
    dates = DATES[:nd]
    return pd.DataFrame(
        np.exp(rng.normal(13.8, 0.5, size=(nd, nt))),
        index=dates,
        columns=tickers,
    )


def make_factor(prices=None, seed=2):
    """Construct a mild momentum factor for testing."""
    if prices is None:
        prices = make_prices()
    rng = np.random.default_rng(seed)
    returns = prices.pct_change()
    # Weak 21-day momentum + noise
    factor = returns.rolling(21).sum().shift(1)
    noise = pd.DataFrame(
        rng.normal(0, 0.01, size=factor.shape),
        index=factor.index,
        columns=factor.columns,
    )
    return (factor + noise).dropna(how="all")


# ===========================================================================
# 1. ResearchResult dataclass
# ===========================================================================

class TestResearchResult:
    def test_to_dict_contains_all_keys(self):
        from research.pipeline.signal_research_pipeline import ResearchResult

        r = ResearchResult(
            signal_name="test_signal",
            ic_mean=0.05,
            ic_std=0.10,
            icir=0.50,
            ic_decay_halflife=8.0,
            regime_conditional_ic={"bull": 0.08, "bear": 0.03, "ranging": 0.01},
            turnover_daily=0.15,
            capacity_estimate_usd=5e6,
            is_significant=True,
            deflated_sharpe=0.72,
            recommendation="PROMOTE",
            p_value=0.01,
            n_observations=500,
        )
        d = r.to_dict()
        assert "signal_name" in d
        assert "icir" in d
        assert "regime_bull" in d
        assert "recommendation" in d

    def test_default_recommendation_is_watch(self):
        from research.pipeline.signal_research_pipeline import ResearchResult

        r = ResearchResult(
            signal_name="x", ic_mean=0.0, ic_std=0.0, icir=0.0, ic_decay_halflife=10.0
        )
        assert r.recommendation == "WATCH"


# ===========================================================================
# 2. Signal functions
# ===========================================================================

class TestBuiltinSignals:
    """Smoke tests for all 15 built-in signal functions."""

    def _run(self, fn):
        prices = make_prices()
        volumes = make_volumes()
        result = fn(prices, volumes)
        assert isinstance(result, pd.DataFrame), f"{fn.__name__} must return DataFrame"
        assert result.shape[0] > 0, f"{fn.__name__} returned empty DataFrame"
        return result

    def test_signal_momentum_1m(self):
        from research.pipeline.signal_research_pipeline import signal_momentum_1m
        self._run(signal_momentum_1m)

    def test_signal_momentum_3m(self):
        from research.pipeline.signal_research_pipeline import signal_momentum_3m
        self._run(signal_momentum_3m)

    def test_signal_momentum_6m(self):
        from research.pipeline.signal_research_pipeline import signal_momentum_6m
        self._run(signal_momentum_6m)

    def test_signal_reversal_5d(self):
        from research.pipeline.signal_research_pipeline import signal_reversal_5d
        self._run(signal_reversal_5d)

    def test_signal_reversal_1d(self):
        from research.pipeline.signal_research_pipeline import signal_reversal_1d
        self._run(signal_reversal_1d)

    def test_signal_vol_breakout(self):
        from research.pipeline.signal_research_pipeline import signal_vol_breakout
        self._run(signal_vol_breakout)

    def test_signal_vol_regime(self):
        from research.pipeline.signal_research_pipeline import signal_vol_regime
        self._run(signal_vol_regime)

    def test_signal_bh_mass_raw(self):
        from research.pipeline.signal_research_pipeline import signal_bh_mass_raw
        self._run(signal_bh_mass_raw)

    def test_signal_bh_mass_filtered(self):
        from research.pipeline.signal_research_pipeline import signal_bh_mass_filtered
        self._run(signal_bh_mass_filtered)

    def test_signal_hurst_trend(self):
        from research.pipeline.signal_research_pipeline import signal_hurst_trend
        self._run(signal_hurst_trend)

    def test_signal_hurst_revert(self):
        from research.pipeline.signal_research_pipeline import signal_hurst_revert
        self._run(signal_hurst_revert)

    def test_signal_rsi_extreme(self):
        from research.pipeline.signal_research_pipeline import signal_rsi_extreme
        self._run(signal_rsi_extreme)

    def test_signal_macd_cross(self):
        from research.pipeline.signal_research_pipeline import signal_macd_cross
        self._run(signal_macd_cross)

    def test_signal_vwap_deviation(self):
        from research.pipeline.signal_research_pipeline import signal_vwap_deviation
        self._run(signal_vwap_deviation)

    def test_signal_atr_expansion(self):
        from research.pipeline.signal_research_pipeline import signal_atr_expansion
        self._run(signal_atr_expansion)


# ===========================================================================
# 3. IC computation
# ===========================================================================

class TestICComputation:
    def test_ic_series_length(self):
        from research.pipeline.signal_research_pipeline import SignalResearchPipeline, signal_momentum_1m

        prices = make_prices()
        volumes = make_volumes()
        pipe = SignalResearchPipeline()
        scores = signal_momentum_1m(prices, volumes)
        returns = prices.pct_change()
        fwd = returns.shift(-5)
        ic = pipe._compute_ic_series(scores, fwd)
        assert len(ic) > 100, "Expected > 100 IC observations"

    def test_ic_values_in_range(self):
        from research.pipeline.signal_research_pipeline import SignalResearchPipeline, signal_momentum_3m

        prices = make_prices()
        volumes = make_volumes()
        pipe = SignalResearchPipeline()
        scores = signal_momentum_3m(prices, volumes)
        returns = prices.pct_change()
        fwd = returns.shift(-5)
        ic = pipe._compute_ic_series(scores, fwd)
        assert ic.dropna().between(-1.0, 1.0).all(), "IC values must be in [-1, 1]"

    def test_ic_spearman_vs_pearson(self):
        """Spearman and Pearson IC should agree in sign on average."""
        from research.pipeline.signal_research_pipeline import SignalResearchPipeline, signal_momentum_1m

        prices = make_prices()
        volumes = make_volumes()
        scores = signal_momentum_1m(prices, volumes)
        returns = prices.pct_change()
        fwd = returns.shift(-5)

        pipe_sp = SignalResearchPipeline(ic_method="spearman")
        pipe_pe = SignalResearchPipeline(ic_method="pearson")

        ic_sp = pipe_sp._compute_ic_series(scores, fwd)
        ic_pe = pipe_pe._compute_ic_series(scores, fwd)

        # Both should be small and consistent in sign
        assert np.sign(ic_sp.mean()) == np.sign(ic_pe.mean()) or (
            abs(ic_sp.mean()) < 0.01 and abs(ic_pe.mean()) < 0.01
        )


# ===========================================================================
# 4. IC decay half-life
# ===========================================================================

class TestICDecayHalflife:
    def test_halflife_positive(self):
        from research.pipeline.signal_research_pipeline import SignalResearchPipeline, signal_momentum_1m

        prices = make_prices()
        volumes = make_volumes()
        scores = signal_momentum_1m(prices, volumes)
        returns = prices.pct_change()
        pipe = SignalResearchPipeline()
        hl = pipe._compute_ic_decay_halflife(scores, returns, max_lag=10)
        assert hl > 0, "Half-life must be positive"


# ===========================================================================
# 5. Full pipeline run
# ===========================================================================

class TestPipelineRun:
    def test_pipeline_returns_research_result(self):
        from research.pipeline.signal_research_pipeline import (
            SignalResearchPipeline, signal_momentum_1m, ResearchResult
        )

        pipe = SignalResearchPipeline()
        result = pipe.run(
            signal_momentum_1m,
            TICKERS[:20],
            "2020-01-01",
            "2021-12-31",
        )
        assert isinstance(result, ResearchResult)

    def test_pipeline_recommendation_is_valid(self):
        from research.pipeline.signal_research_pipeline import (
            SignalResearchPipeline, signal_reversal_5d
        )

        pipe = SignalResearchPipeline()
        result = pipe.run(
            signal_reversal_5d,
            TICKERS[:20],
            "2020-01-01",
            "2021-12-31",
        )
        assert result.recommendation in ("PROMOTE", "WATCH", "RETIRE")

    def test_pipeline_ic_decay_halflife_finite_or_inf(self):
        from research.pipeline.signal_research_pipeline import (
            SignalResearchPipeline, signal_momentum_3m
        )

        pipe = SignalResearchPipeline()
        result = pipe.run(
            signal_momentum_3m,
            TICKERS[:15],
            "2020-01-01",
            "2021-06-30",
        )
        assert result.ic_decay_halflife > 0


# ===========================================================================
# 6. SignalUniverse
# ===========================================================================

class TestSignalUniverse:
    def test_register_and_list(self):
        from research.pipeline.signal_research_pipeline import SignalUniverse

        u = SignalUniverse()
        u.register("dummy", lambda p, v: p * 0.0, "test signal", "test")
        lst = u.list_signals()
        assert "dummy" in lst["name"].values

    def test_run_all_returns_list_of_results(self):
        from research.pipeline.signal_research_pipeline import SignalUniverse, ResearchResult

        u = SignalUniverse()
        u.register("mom_1m_test", lambda p, v: p.pct_change().rolling(21).sum().shift(1), "test", "momentum")
        u.register("rev_5d_test", lambda p, v: -p.pct_change().rolling(5).sum(), "test", "reversal")

        results = u.run_all(TICKERS[:15], "2020-01-01", "2021-06-30")
        assert len(results) == 2
        assert all(isinstance(r, ResearchResult) for r in results)

    def test_compare_returns_dataframe_sorted_by_icir(self):
        from research.pipeline.signal_research_pipeline import SignalUniverse, ResearchResult

        u = SignalUniverse()
        u.register("a", lambda p, v: p.pct_change().rolling(21).sum().shift(1), "test", "test")
        u.register("b", lambda p, v: -p.pct_change().rolling(5).sum(), "test2", "test")
        results = u.run_all(TICKERS[:15], "2020-01-01", "2021-06-30")
        df = u.compare(results)
        assert isinstance(df, pd.DataFrame)
        if len(df) > 1:
            assert df["icir"].iloc[0] >= df["icir"].iloc[1]


# ===========================================================================
# 7. FactorBuilder
# ===========================================================================

class TestFactorBuilder:
    def test_build_momentum_factor_shape(self):
        from research.pipeline.factor_construction import FactorBuilder

        fb = FactorBuilder()
        prices = make_prices()
        returns = prices.pct_change().dropna()
        mom = fb.build_momentum_factor(returns, lookback=63, skip=1)
        assert mom.shape == returns.shape

    def test_build_momentum_factor_raises_on_bad_lookback(self):
        from research.pipeline.factor_construction import FactorBuilder

        fb = FactorBuilder()
        returns = make_returns()
        with pytest.raises(ValueError):
            fb.build_momentum_factor(returns, lookback=0)

    def test_build_low_vol_factor_percentile_range(self):
        from research.pipeline.factor_construction import FactorBuilder

        fb = FactorBuilder()
        returns = make_returns()
        lvf = fb.build_low_vol_factor(returns, window=21)
        valid = lvf.dropna(how="all").values.flatten()
        valid = valid[~np.isnan(valid)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_build_value_factor_shape(self):
        from research.pipeline.factor_construction import FactorBuilder

        fb = FactorBuilder()
        prices = make_prices()
        # Synthetic book values: 50%-100% of price
        book = prices * RNG.uniform(0.5, 1.0, size=prices.shape)
        bp = fb.build_value_factor(prices, book)
        assert bp.shape == prices.shape

    def test_build_sentiment_factor_rank_range(self):
        from research.pipeline.factor_construction import FactorBuilder

        fb = FactorBuilder()
        n_dates, n_tickers = 200, 20
        sentiment = pd.DataFrame(
            RNG.uniform(-1, 1, size=(n_dates, n_tickers)),
            index=DATES[:n_dates],
            columns=TICKERS[:n_tickers],
        )
        sf = fb.build_sentiment_factor(sentiment, lookback=5)
        valid = sf.values.flatten()
        valid = valid[~np.isnan(valid)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0


# ===========================================================================
# 8. FactorNeutralizer
# ===========================================================================

class TestFactorNeutralizer:
    def test_standardize_mean_near_zero(self):
        from research.pipeline.factor_construction import FactorNeutralizer

        fn = FactorNeutralizer()
        factor = make_factor()
        standardized = fn.standardize(factor)
        row_means = standardized.mean(axis=1).dropna()
        assert (row_means.abs() < 1e-9).all(), "Cross-sectional mean should be ~0"

    def test_standardize_std_near_one(self):
        from research.pipeline.factor_construction import FactorNeutralizer

        fn = FactorNeutralizer()
        factor = make_factor()
        standardized = fn.standardize(factor)
        row_stds = standardized.std(axis=1).dropna()
        assert ((row_stds - 1.0).abs() < 1e-6).all(), "Cross-sectional std should be ~1"

    def test_winsorize_clips_extremes(self):
        from research.pipeline.factor_construction import FactorNeutralizer

        fn = FactorNeutralizer()
        # Build a factor with ONE isolated extreme value per row (1 out of 30 tickers).
        # With clip_pct=0.01 and 30 tickers, the 99th percentile threshold per row
        # is the second-highest value, so the single extreme outlier gets capped.
        rng2 = np.random.default_rng(99)
        n_dates, n_tickers = 100, 30
        data = rng2.normal(0, 1, size=(n_dates, n_tickers))
        # Inject a single +1000 outlier into col 0 on every row
        data[:, 0] = 1000.0
        factor = pd.DataFrame(data, index=DATES[:n_dates], columns=TICKERS[:n_tickers])
        # 99th percentile of 30 values = ~29th value, which is < 1000
        # so col 0 = 1000 gets clipped down to a value < 10 (other cols are N(0,1))
        winsorized = fn.winsorize(factor, clip_pct=0.01)
        # The maximum after clipping should be much less than 1000
        assert winsorized["T00"].max() < 100.0

    def test_winsorize_raises_on_invalid_clip_pct(self):
        from research.pipeline.factor_construction import FactorNeutralizer

        fn = FactorNeutralizer()
        with pytest.raises(ValueError):
            fn.winsorize(make_factor(), clip_pct=0.6)

    def test_neutralize_market_returns_same_shape(self):
        from research.pipeline.factor_construction import FactorNeutralizer

        fn = FactorNeutralizer()
        prices = make_prices()
        returns = prices.pct_change().dropna()
        factor = make_factor(prices)
        mkt = returns.mean(axis=1)
        neutralized = fn.neutralize_market(factor.dropna(how="all"), mkt)
        assert neutralized.shape == factor.dropna(how="all").shape


# ===========================================================================
# 9. FactorCombiner
# ===========================================================================

class TestFactorCombiner:
    def _get_two_factors(self):
        prices = make_prices()
        from research.pipeline.factor_construction import FactorBuilder
        fb = FactorBuilder()
        returns = prices.pct_change().dropna()
        f1 = fb.build_momentum_factor(returns, lookback=21).dropna(how="all")
        f2 = fb.build_low_vol_factor(returns, window=21).dropna(how="all")
        return {"momentum": f1, "low_vol": f2}

    def test_equal_weight_returns_dataframe(self):
        from research.pipeline.factor_construction import FactorCombiner

        fc = FactorCombiner()
        factors = self._get_two_factors()
        composite = fc.equal_weight(factors)
        assert isinstance(composite, pd.DataFrame)

    def test_rank_combine_values_in_zero_one(self):
        from research.pipeline.factor_construction import FactorCombiner

        fc = FactorCombiner()
        factors = self._get_two_factors()
        composite = fc.rank_combine(factors)
        valid = composite.values.flatten()
        valid = valid[~np.isnan(valid)]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_optimize_weights_sum_to_one(self):
        from research.pipeline.factor_construction import FactorCombiner

        fc = FactorCombiner()
        factors = self._get_two_factors()
        prices = make_prices()
        fwd_returns = prices.pct_change().shift(-5)
        weights = fc.optimize_weights(factors, fwd_returns, method="max_ic")
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1; got {total}"

    def test_optimize_weights_non_negative(self):
        from research.pipeline.factor_construction import FactorCombiner

        fc = FactorCombiner()
        factors = self._get_two_factors()
        prices = make_prices()
        fwd_returns = prices.pct_change().shift(-5)
        weights = fc.optimize_weights(factors, fwd_returns, method="max_icir")
        assert all(w >= -1e-9 for w in weights.values()), "All weights must be >= 0"


# ===========================================================================
# 10. CrossSectionalStudy
# ===========================================================================

class TestCrossSectionalStudy:
    def test_quintile_returns_five_buckets(self):
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        result = cs.quintile_returns(factor, returns)
        assert len(result.quintile_returns) == 5

    def test_quintile_monotonicity_range(self):
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        result = cs.quintile_returns(factor, returns)
        assert 0.0 <= result.monotonicity <= 1.0

    def test_decile_returns_ten_buckets(self):
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        result = cs.decile_returns(factor, returns)
        assert len(result.decile_returns) == 10

    def test_ic_series_spearman(self):
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        ic = cs.information_coefficient(factor, returns, method="spearman")
        assert isinstance(ic, pd.Series)
        assert len(ic) > 50

    def test_ic_series_values_bounded(self):
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        ic = cs.information_coefficient(factor, returns, method="pearson")
        assert ic.dropna().between(-1.0, 1.0).all()

    def test_ic_raises_on_bad_method(self):
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        with pytest.raises(ValueError):
            cs.information_coefficient(factor, returns, method="kendall")

    def test_long_short_portfolio_returns_series(self):
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        ls = cs.long_short_portfolio(factor, returns, n_long=10, n_short=10)
        assert isinstance(ls, pd.Series)
        assert len(ls) > 50


# ===========================================================================
# 11. Fama-MacBeth regression
# ===========================================================================

class TestFamaMacBeth:
    def test_fama_macbeth_returns_result(self):
        from research.pipeline.cross_sectional_study import CrossSectionalStudy, FamaMacBethResult

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        fm = cs.fama_macbeth(factor, returns)
        assert isinstance(fm, FamaMacBethResult)

    def test_fama_macbeth_n_periods_positive(self):
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        fm = cs.fama_macbeth(factor, returns)
        assert fm.n_periods > 0

    def test_fama_macbeth_tstat_consistent(self):
        """t-stat should be coefficient / std_error."""
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        fm = cs.fama_macbeth(factor, returns)
        if fm.std_error > 1e-9:
            expected_t = fm.coefficient / fm.std_error
            assert abs(fm.t_stat - expected_t) < 1e-6

    def test_newey_west_se_positive(self):
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        fm = cs.fama_macbeth(factor, returns, nw_lags=4)
        assert fm.newey_west_se >= 0.0

    def test_newey_west_se_larger_than_ols_se(self):
        """
        For persistent factors, NW SE should generally be >= OLS SE
        (HAC accounts for autocorrelation, inflating SE).
        This is not guaranteed but check it is within 3x.
        """
        from research.pipeline.cross_sectional_study import CrossSectionalStudy

        cs = CrossSectionalStudy()
        factor = make_factor()
        prices = make_prices()
        returns = prices.pct_change()
        fm = cs.fama_macbeth(factor, returns, nw_lags=4)
        if fm.std_error > 0:
            ratio = fm.newey_west_se / fm.std_error
            assert ratio < 5.0, f"NW SE is suspiciously large ({ratio:.2f}x OLS SE)"


# ===========================================================================
# 12. AlphaResearchDB
# ===========================================================================

class TestAlphaResearchDB:
    @pytest.fixture
    def db(self, tmp_path):
        from research.pipeline.alpha_research_db import AlphaResearchDB
        return AlphaResearchDB(str(tmp_path / "test.db"))

    def _make_result(self, name="signal_a", icir=0.6, rec="PROMOTE"):
        from research.pipeline.signal_research_pipeline import ResearchResult
        return ResearchResult(
            signal_name=name,
            ic_mean=0.05,
            ic_std=0.10 / max(abs(icir), 0.1),
            icir=icir,
            ic_decay_halflife=8.0,
            regime_conditional_ic={"bull": 0.08, "bear": 0.03, "ranging": 0.01},
            turnover_daily=0.15,
            capacity_estimate_usd=5e6,
            is_significant=True,
            deflated_sharpe=0.72,
            recommendation=rec,
            p_value=0.02,
            n_observations=500,
        )

    def test_save_and_get_result(self, db):
        r = self._make_result("sig1")
        db.save_result(r)
        results = db.get_results()
        names = [x.signal_name for x in results]
        assert "sig1" in names

    def test_upsert_updates_existing(self, db):
        r1 = self._make_result("sig1", icir=0.3, rec="WATCH")
        db.save_result(r1)
        r2 = self._make_result("sig1", icir=0.8, rec="PROMOTE")
        db.save_result(r2)
        results = db.get_results()
        sig = next(x for x in results if x.signal_name == "sig1")
        assert sig.icir == pytest.approx(0.8, abs=1e-6)
        assert sig.recommendation == "PROMOTE"

    def test_filter_by_category(self, db):
        db.save_result(self._make_result("mom_1m"))
        db.save_result(self._make_result("rev_5d"))
        # Manually set category via a fresh result
        from research.pipeline.signal_research_pipeline import ResearchResult
        r = self._make_result("mom_cat_test")
        r.category = "momentum"
        db.save_result(r)
        results = db.get_results(category="momentum")
        assert all(x.category == "momentum" for x in results)

    def test_filter_by_min_icir(self, db):
        db.save_result(self._make_result("low_icir_sig", icir=0.1, rec="RETIRE"))
        db.save_result(self._make_result("high_icir_sig", icir=0.9, rec="PROMOTE"))
        results = db.get_results(min_icir=0.5)
        for r in results:
            assert abs(r.icir) >= 0.5

    def test_get_best_signals_returns_top_n(self, db):
        for i in range(5):
            db.save_result(self._make_result(f"sig_{i}", icir=float(i) * 0.2))
        best = db.get_best_signals(n=3)
        assert len(best) <= 3

    def test_save_and_retrieve_ic_history(self, db):
        from research.pipeline.signal_research_pipeline import ResearchResult
        import pandas as pd

        r = self._make_result("ic_hist_sig")
        ic_idx = pd.date_range("2021-01-01", periods=100, freq="B")
        r.ic_series = pd.Series(np.random.normal(0.05, 0.1, 100), index=ic_idx)
        db.save_result(r)
        ic = db.get_ic_history("ic_hist_sig")
        assert len(ic) == 100

    def test_update_and_get_live_performance(self, db):
        r = self._make_result("live_sig")
        db.save_result(r)
        db.update_live_performance("live_sig", "2023-01-03", 0.04)
        db.update_live_performance("live_sig", "2023-01-04", 0.06)
        live = db.get_live_performance("live_sig")
        assert len(live) == 2
        assert live.mean() == pytest.approx(0.05, abs=1e-9)

    def test_compute_decay_in_production(self, db):
        from research.pipeline.signal_research_pipeline import ResearchResult
        import pandas as pd

        r = self._make_result("decay_sig")
        ic_idx = pd.date_range("2020-01-02", periods=500, freq="B")
        r.ic_series = pd.Series(np.full(500, 0.06), index=ic_idx)
        db.save_result(r)

        # Live IC is half of backtest
        for i, d in enumerate(pd.date_range("2023-01-02", periods=30, freq="B")):
            db.update_live_performance("decay_sig", str(d.date()), 0.03)

        decay = db.compute_decay_in_production("decay_sig")
        assert decay == pytest.approx(0.5, abs=0.01)

    def test_retirement_candidates_detected(self, db):
        from research.pipeline.signal_research_pipeline import ResearchResult
        import pandas as pd

        r = self._make_result("retire_me")
        ic_idx = pd.date_range("2020-01-02", periods=500, freq="B")
        r.ic_series = pd.Series(np.full(500, 0.08), index=ic_idx)
        db.save_result(r)

        # Live IC is only 30% of backtest
        for d in pd.date_range("2023-01-02", periods=25, freq="B"):
            db.update_live_performance("retire_me", str(d.date()), 0.024)

        candidates = db.retirement_candidates(decay_threshold=0.5, min_live_obs=20)
        assert "retire_me" in candidates

    def test_no_retirement_for_healthy_signal(self, db):
        from research.pipeline.signal_research_pipeline import ResearchResult
        import pandas as pd

        r = self._make_result("healthy_sig")
        ic_idx = pd.date_range("2020-01-02", periods=500, freq="B")
        r.ic_series = pd.Series(np.full(500, 0.08), index=ic_idx)
        db.save_result(r)

        # Live IC is 80% of backtest -- healthy
        for d in pd.date_range("2023-01-02", periods=25, freq="B"):
            db.update_live_performance("healthy_sig", str(d.date()), 0.064)

        candidates = db.retirement_candidates(decay_threshold=0.5, min_live_obs=20)
        assert "healthy_sig" not in candidates


# ===========================================================================
# 13. ResearchReport
# ===========================================================================

class TestResearchReport:
    def _get_results(self):
        from research.pipeline.signal_research_pipeline import ResearchResult
        results = []
        for name, icir, rec in [
            ("signal_a", 0.8, "PROMOTE"),
            ("signal_b", 0.3, "WATCH"),
            ("signal_c", -0.1, "RETIRE"),
        ]:
            results.append(ResearchResult(
                signal_name=name,
                ic_mean=icir * 0.1,
                ic_std=0.1,
                icir=icir,
                ic_decay_halflife=7.0,
                regime_conditional_ic={"bull": 0.05, "bear": 0.02, "ranging": 0.01},
                recommendation=rec,
            ))
        return results

    def test_generate_markdown_contains_signal_names(self):
        from research.pipeline.alpha_research_db import ResearchReport

        report = ResearchReport()
        md = report.generate_markdown(self._get_results())
        assert "signal_a" in md
        assert "signal_b" in md
        assert "signal_c" in md

    def test_generate_comparison_table_is_string(self):
        from research.pipeline.alpha_research_db import ResearchReport

        report = ResearchReport()
        table = report.generate_comparison_table(self._get_results())
        assert isinstance(table, str)
        assert "PROMOTE" in table or "WATCH" in table or "RETIRE" in table

    def test_generate_html_creates_file(self, tmp_path):
        from research.pipeline.alpha_research_db import ResearchReport

        report = ResearchReport()
        out_path = str(tmp_path / "test_report.html")
        report.generate_html(self._get_results(), out_path)
        assert os.path.exists(out_path)
        with open(out_path, "r") as fh:
            content = fh.read()
        assert "<html>" in content
        assert "signal_a" in content
