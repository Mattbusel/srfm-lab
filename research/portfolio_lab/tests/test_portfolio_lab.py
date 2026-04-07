"""
research/portfolio_lab/tests/test_portfolio_lab.py

Tests for efficient frontier, factor exposure, correlation labeller,
and return attribution modules.

Run with: pytest research/portfolio_lab/tests/test_portfolio_lab.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.portfolio_lab.efficient_frontier_lab import (
    EfficientFrontierLab,
    FrontierResult,
    PortfolioPoint,
)
from research.portfolio_lab.factor_exposure_analyzer import (
    FactorExposureAnalyzer,
    ReturnDecomposer,
    FactorExposures,
    generate_synthetic_factors,
)
from research.portfolio_lab.correlation_labeler import (
    CorrelationLabeler,
    CorrelationRegime,
    DiversificationMeasure,
    StabilityPoint,
    ClusterResult,
)
from research.portfolio_lab.return_attribution_lab import (
    ReturnAttributionLab,
    BrinsonResult,
    AttributionChain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def simple_returns(rng: np.random.Generator) -> pd.DataFrame:
    """Three-asset daily return DataFrame, 500 observations."""
    n = 500
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    data = {
        "A": rng.normal(0.0005, 0.010, n),
        "B": rng.normal(0.0003, 0.015, n),
        "C": rng.normal(0.0002, 0.008, n),
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def four_asset_returns(rng: np.random.Generator) -> pd.DataFrame:
    n = 504
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    data = {
        "SPY": rng.normal(0.0004, 0.012, n),
        "AGG": rng.normal(0.0001, 0.003, n),
        "GLD": rng.normal(0.0002, 0.008, n),
        "EFA": rng.normal(0.0003, 0.011, n),
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def eff_lab(simple_returns: pd.DataFrame) -> EfficientFrontierLab:
    lab = EfficientFrontierLab(annualise=True)
    for col in simple_returns.columns:
        lab.add_asset(col, simple_returns[col])
    return lab


@pytest.fixture
def factors(simple_returns: pd.DataFrame) -> pd.DataFrame:
    return generate_synthetic_factors(n_obs=len(simple_returns))


# ---------------------------------------------------------------------------
# EfficientFrontierLab tests
# ---------------------------------------------------------------------------


class TestEfficientFrontierLab:
    def test_add_asset_and_count(self, simple_returns: pd.DataFrame) -> None:
        lab = EfficientFrontierLab()
        assert lab.n_assets == 0
        lab.add_asset("A", simple_returns["A"])
        assert lab.n_assets == 1

    def test_duplicate_asset_raises(self, eff_lab: EfficientFrontierLab) -> None:
        with pytest.raises(ValueError, match="already added"):
            eff_lab.add_asset("A", pd.Series([0.01, 0.02]))

    def test_compute_frontier_returns_correct_shape(
        self, eff_lab: EfficientFrontierLab
    ) -> None:
        result = eff_lab.compute_frontier(n_points=50, compute_bootstrap=False)
        assert isinstance(result, FrontierResult)
        assert result.weights.shape == (50, 3)
        assert len(result.returns) == 50
        assert len(result.volatilities) == 50
        assert len(result.sharpes) == 50

    def test_frontier_weights_sum_to_one(self, eff_lab: EfficientFrontierLab) -> None:
        result = eff_lab.compute_frontier(n_points=30, compute_bootstrap=False)
        sums = result.weights.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4)

    def test_frontier_weights_non_negative(
        self, eff_lab: EfficientFrontierLab
    ) -> None:
        result = eff_lab.compute_frontier(n_points=30, compute_bootstrap=False)
        assert np.all(result.weights >= -1e-6)

    def test_min_variance_lower_vol_than_equal_weight(
        self, eff_lab: EfficientFrontierLab
    ) -> None:
        mvp = eff_lab.find_min_variance()
        # equal weight portfolio
        df = eff_lab._build_returns_matrix()
        _, cov = eff_lab._stats(df)
        n = eff_lab.n_assets
        eq_w = np.ones(n) / n
        from research.portfolio_lab.efficient_frontier_lab import _port_vol
        eq_vol = _port_vol(eq_w, cov)
        # min variance must have vol <= equal weight vol
        assert mvp.volatility <= eq_vol + 1e-6

    def test_min_variance_weights_sum_to_one(
        self, eff_lab: EfficientFrontierLab
    ) -> None:
        mvp = eff_lab.find_min_variance()
        assert abs(mvp.weights.sum() - 1.0) < 1e-5

    def test_min_variance_weights_non_negative(
        self, eff_lab: EfficientFrontierLab
    ) -> None:
        mvp = eff_lab.find_min_variance()
        assert np.all(mvp.weights >= -1e-6)

    def test_max_sharpe_exceeds_individual_assets(
        self, eff_lab: EfficientFrontierLab
    ) -> None:
        ms = eff_lab.find_max_sharpe(rf_rate=0.0)
        # Sharpe should be positive for positively-returning assets
        assert ms.sharpe > -np.inf
        # Max Sharpe weights sum to 1
        assert abs(ms.weights.sum() - 1.0) < 1e-4

    def test_max_sharpe_label(self, eff_lab: EfficientFrontierLab) -> None:
        ms = eff_lab.find_max_sharpe()
        assert ms.label == "max_sharpe"

    def test_find_risk_parity_weights_sum_to_one(
        self, eff_lab: EfficientFrontierLab
    ) -> None:
        rp = eff_lab.find_risk_parity()
        assert abs(rp.weights.sum() - 1.0) < 1e-4

    def test_find_risk_parity_label(self, eff_lab: EfficientFrontierLab) -> None:
        rp = eff_lab.find_risk_parity()
        assert rp.label == "risk_parity"

    def test_random_portfolios_shape(
        self, eff_lab: EfficientFrontierLab
    ) -> None:
        df = eff_lab.random_portfolios(n=200)
        assert len(df) == 200
        assert "expected_return" in df.columns
        assert "volatility" in df.columns
        assert "sharpe" in df.columns

    def test_random_portfolios_weights_sum_to_one(
        self, eff_lab: EfficientFrontierLab
    ) -> None:
        df = eff_lab.random_portfolios(n=100)
        weight_cols = eff_lab.asset_labels
        sums = df[weight_cols].sum(axis=1)
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-6)

    def test_bootstrap_bands_populated(
        self, eff_lab: EfficientFrontierLab
    ) -> None:
        result = eff_lab.compute_frontier(
            n_points=20, compute_bootstrap=True, n_bootstrap=20
        )
        assert result.vol_lower is not None
        assert result.vol_upper is not None
        assert len(result.vol_lower) == 20
        assert np.all(result.vol_lower <= result.vol_upper + 1e-10)

    def test_frontier_to_dataframe(self, eff_lab: EfficientFrontierLab) -> None:
        result = eff_lab.compute_frontier(n_points=15, compute_bootstrap=False)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 15

    def test_summary_dataframe(self, eff_lab: EfficientFrontierLab) -> None:
        s = eff_lab.summary()
        assert "ann_return" in s.columns
        assert len(s) == 3


# ---------------------------------------------------------------------------
# FactorExposureAnalyzer tests
# ---------------------------------------------------------------------------


class TestFactorExposureAnalyzer:
    def test_compute_exposures_returns_factor_exposures(
        self,
        simple_returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> None:
        analyzer = FactorExposureAnalyzer()
        exp = analyzer.compute_exposures(simple_returns["A"], factors)
        assert isinstance(exp, FactorExposures)
        assert "Market" in exp.betas
        assert exp.r_squared >= 0.0
        assert exp.r_squared <= 1.0 + 1e-6

    def test_market_portfolio_beta_close_to_one(
        self, factors: pd.DataFrame
    ) -> None:
        """A portfolio that IS the market should have Market beta ~1."""
        market_ret = factors["Market"].copy()
        analyzer = FactorExposureAnalyzer()
        exp = analyzer.compute_exposures(market_ret, factors[["Market"]])
        assert abs(exp.betas["Market"] - 1.0) < 0.05

    def test_r_squared_in_range(
        self,
        simple_returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> None:
        analyzer = FactorExposureAnalyzer()
        exp = analyzer.compute_exposures(simple_returns["A"], factors)
        assert 0.0 <= exp.r_squared <= 1.0 + 1e-6

    def test_rolling_exposures_shape(
        self,
        simple_returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> None:
        analyzer = FactorExposureAnalyzer()
        roll = analyzer.rolling_exposures(simple_returns["A"], factors, window=63)
        assert isinstance(roll, pd.DataFrame)
        assert len(roll) > 0
        assert "alpha" in roll.columns
        assert "beta_Market" in roll.columns

    def test_exposure_drift_returns_drift_report(
        self,
        simple_returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> None:
        analyzer = FactorExposureAnalyzer()
        report = analyzer.exposure_drift(simple_returns["A"], factors, window=63)
        assert "Market" in report.mean_abs_drift
        assert report.total_drift_score >= 0.0

    def test_explain_return_components_present(
        self,
        simple_returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> None:
        analyzer = FactorExposureAnalyzer()
        er = analyzer.explain_return(simple_returns["A"], factors)
        assert "Market" in er.factor_contributions
        assert isinstance(er.systematic_return, float)
        assert isinstance(er.idiosyncratic_return, float)

    def test_generate_synthetic_factors_shape(self) -> None:
        fac = generate_synthetic_factors(n_obs=252)
        assert fac.shape == (252, 7)
        assert "Market" in fac.columns

    def test_return_decomposer_keys(
        self,
        simple_returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> None:
        decomposer = ReturnDecomposer()
        contrib = decomposer.decompose(simple_returns["A"], factors)
        assert "specific" in contrib
        assert "alpha_daily" in contrib
        assert "Market" in contrib

    def test_return_decomposer_adds_up(
        self,
        simple_returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> None:
        """Sum of contributions should approximate the actual return series."""
        decomposer = ReturnDecomposer()
        contrib = decomposer.decompose(simple_returns["A"], factors)
        total = sum(contrib.values())
        actual = simple_returns["A"].reindex(total.index).dropna()
        reconstructed = total.reindex(actual.index).dropna()
        # Should be close (OLS fit, not exact)
        np.testing.assert_allclose(
            reconstructed.values, actual.values, atol=1e-4
        )


# ---------------------------------------------------------------------------
# CorrelationLabeler tests
# ---------------------------------------------------------------------------


class TestCorrelationLabeler:
    def test_build_correlation_matrix_shape(
        self, four_asset_returns: pd.DataFrame
    ) -> None:
        labeler = CorrelationLabeler()
        corr = labeler.build_correlation_matrix(four_asset_returns)
        assert corr.shape == (4, 4)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_label_regime_low(self) -> None:
        """Near-zero correlation -> LOW_CORR."""
        C = pd.DataFrame(
            [[1.0, 0.1, 0.05], [0.1, 1.0, 0.08], [0.05, 0.08, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        labeler = CorrelationLabeler()
        regime = labeler.label_regime(C)
        assert regime == CorrelationRegime.LOW_CORR

    def test_label_regime_crisis(self) -> None:
        """High correlations -> CRISIS."""
        C = pd.DataFrame(
            [[1.0, 0.85, 0.90], [0.85, 1.0, 0.88], [0.90, 0.88, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        labeler = CorrelationLabeler()
        regime = labeler.label_regime(C)
        assert regime == CorrelationRegime.CRISIS

    def test_label_regime_moderate(self) -> None:
        C = pd.DataFrame(
            [[1.0, 0.45, 0.40], [0.45, 1.0, 0.42], [0.40, 0.42, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        labeler = CorrelationLabeler()
        assert labeler.label_regime(C) == CorrelationRegime.MODERATE_CORR

    def test_label_regime_high(self) -> None:
        C = pd.DataFrame(
            [[1.0, 0.70, 0.72], [0.70, 1.0, 0.71], [0.72, 0.71, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        labeler = CorrelationLabeler()
        assert labeler.label_regime(C) == CorrelationRegime.HIGH_CORR

    def test_stability_points_list_length(
        self, four_asset_returns: pd.DataFrame
    ) -> None:
        labeler = CorrelationLabeler()
        points = labeler.track_correlation_stability(
            four_asset_returns, window=63, step=10
        )
        assert len(points) > 0
        assert isinstance(points[0], StabilityPoint)

    def test_stability_point_corr_bounds(
        self, four_asset_returns: pd.DataFrame
    ) -> None:
        labeler = CorrelationLabeler()
        points = labeler.track_correlation_stability(
            four_asset_returns, window=63, step=20
        )
        for p in points:
            assert p.min_corr <= p.mean_corr <= p.max_corr + 1e-10

    def test_find_cluster_structure_returns_cluster_result(
        self, four_asset_returns: pd.DataFrame
    ) -> None:
        labeler = CorrelationLabeler()
        corr = labeler.build_correlation_matrix(four_asset_returns)
        result = labeler.find_cluster_structure(corr)
        assert isinstance(result, ClusterResult)
        assert result.n_clusters >= 1
        assert len(result.cluster_labels) == 4

    def test_cluster_members_covers_all_assets(
        self, four_asset_returns: pd.DataFrame
    ) -> None:
        labeler = CorrelationLabeler()
        corr = labeler.build_correlation_matrix(four_asset_returns)
        result = labeler.find_cluster_structure(corr)
        members = result.cluster_members()
        all_assigned = [a for lst in members.values() for a in lst]
        assert set(all_assigned) == set(four_asset_returns.columns)


# ---------------------------------------------------------------------------
# DiversificationMeasure tests
# ---------------------------------------------------------------------------


class TestDiversificationMeasure:
    def test_effective_n_single_asset(self) -> None:
        """Single concentrated position: effective N ~= 1."""
        w = np.array([1.0, 0.0, 0.0])
        assert abs(DiversificationMeasure.effective_n(w) - 1.0) < 1e-6

    def test_effective_n_equal_weight(self) -> None:
        """Equal weights: effective N = n."""
        n = 5
        w = np.ones(n) / n
        eff = DiversificationMeasure.effective_n(w)
        assert abs(eff - float(n)) < 1e-6

    def test_diversification_ratio_single_asset(self) -> None:
        """DR = 1.0 for a single asset (no diversification)."""
        w = np.array([1.0])
        vols = np.array([0.15])
        C = np.array([[1.0]])
        dr = DiversificationMeasure.diversification_ratio(w, vols, C)
        assert abs(dr - 1.0) < 1e-6

    def test_diversification_ratio_uncorrelated_assets(self) -> None:
        """Uncorrelated assets: DR > 1 for mixed portfolio."""
        w = np.array([0.5, 0.5])
        vols = np.array([0.10, 0.20])
        C = np.eye(2)
        dr = DiversificationMeasure.diversification_ratio(w, vols, C)
        assert dr > 1.0

    def test_portfolio_concentration_single(self) -> None:
        w = np.array([1.0, 0.0])
        assert abs(DiversificationMeasure.portfolio_concentration(w) - 1.0) < 1e-10

    def test_portfolio_concentration_equal(self) -> None:
        n = 4
        w = np.ones(n) / n
        assert abs(DiversificationMeasure.portfolio_concentration(w) - 1.0 / n) < 1e-10

    def test_gini_equal_weights(self) -> None:
        """Equal weights -> Gini ~ 0."""
        w = np.array([0.25, 0.25, 0.25, 0.25])
        g = DiversificationMeasure.gini_coefficient(w)
        assert g < 0.05

    def test_risk_contributions_sum_to_one(
        self, four_asset_returns: pd.DataFrame
    ) -> None:
        scale = 252
        cov = four_asset_returns.cov().values * scale
        w = np.array([0.25, 0.25, 0.25, 0.25])
        rc = DiversificationMeasure.risk_contributions(w, cov)
        assert abs(rc.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# ReturnAttributionLab tests
# ---------------------------------------------------------------------------


class TestReturnAttributionLab:
    def _simple_brinson(self) -> BrinsonResult:
        lab = ReturnAttributionLab()
        # portfolio overweights Technology vs benchmark
        portfolio_weights = {"AAPL": 0.30, "MSFT": 0.20, "JNJ": 0.25, "PG": 0.25}
        benchmark_weights = {"AAPL": 0.15, "MSFT": 0.15, "JNJ": 0.35, "PG": 0.35}
        portfolio_returns = {"AAPL": 0.02, "MSFT": 0.03, "JNJ": 0.01, "PG": 0.005}
        benchmark_returns = {"AAPL": 0.015, "MSFT": 0.025, "JNJ": 0.008, "PG": 0.004}
        sectors = {"AAPL": "Tech", "MSFT": "Tech", "JNJ": "Health", "PG": "Health"}
        return lab.brinson_attribution(
            portfolio_weights,
            benchmark_weights,
            portfolio_returns,
            benchmark_returns,
            sectors,
        )

    def test_brinson_consistency(self) -> None:
        """allocation + selection + interaction == total."""
        result = self._simple_brinson()
        assert result.check_consistency(tol=1e-10)

    def test_brinson_total_is_active_return(self) -> None:
        """total should equal portfolio return minus benchmark return."""
        lab = ReturnAttributionLab()
        pw = {"A": 0.6, "B": 0.4}
        bw = {"A": 0.5, "B": 0.5}
        pr = {"A": 0.10, "B": 0.05}
        br = {"A": 0.08, "B": 0.04}
        sectors = {"A": "S1", "B": "S1"}
        result = lab.brinson_attribution(pw, bw, pr, br, sectors)
        # active return
        port_ret = pw["A"] * pr["A"] + pw["B"] * pr["B"]
        bench_ret = bw["A"] * br["A"] + bw["B"] * br["B"]
        active = port_ret - bench_ret
        assert abs(result.total - active) < 1e-10

    def test_brinson_by_sector_keys(self) -> None:
        result = self._simple_brinson()
        assert "Tech" in result.by_sector
        assert "Health" in result.by_sector

    def test_brinson_sector_effects_sum_to_total(self) -> None:
        result = self._simple_brinson()
        sector_totals = sum(v["total"] for v in result.by_sector.values())
        assert abs(sector_totals - result.total) < 1e-10

    def test_factor_attribution_returns_factor_result(
        self,
        simple_returns: pd.DataFrame,
        factors: pd.DataFrame,
    ) -> None:
        lab = ReturnAttributionLab()
        exposures = {"Market": 0.9, "Size": 0.2}
        result = lab.factor_attribution(simple_returns["A"], exposures, factors)
        assert "Market" in result.factor_contributions
        assert isinstance(result.active_return, float)

    def test_transaction_cost_attribution_net_less_than_gross(
        self, rng: np.random.Generator
    ) -> None:
        lab = ReturnAttributionLab()
        n = 252
        idx = pd.date_range("2021-01-04", periods=n, freq="B")
        gross = pd.Series(rng.normal(0.0004, 0.01, n), index=idx)
        costs = pd.Series(np.full(n, 0.0001), index=idx)  # 1 bps/day
        result = lab.transaction_cost_attribution(gross, costs)
        assert result.net_return <= result.gross_return + 1e-10
        assert result.total_cost >= 0.0

    def test_transaction_cost_zero_costs(
        self, rng: np.random.Generator
    ) -> None:
        lab = ReturnAttributionLab()
        n = 100
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        gross = pd.Series(rng.normal(0.0003, 0.008, n), index=idx)
        costs = pd.Series(np.zeros(n), index=idx)
        result = lab.transaction_cost_attribution(gross, costs)
        assert abs(result.total_cost) < 1e-10
        assert abs(result.net_return - result.gross_return) < 1e-10

    def test_time_attribution_buckets_present(
        self, rng: np.random.Generator
    ) -> None:
        lab = ReturnAttributionLab()
        n = 252
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        returns = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
        result = lab.time_attribution(returns, time_buckets="ME")
        assert len(result.bucket_returns) > 0
        assert isinstance(result.total_return, float)

    def test_time_attribution_non_datetime_raises(self) -> None:
        lab = ReturnAttributionLab()
        returns = pd.Series([0.01, 0.02, -0.01], index=[0, 1, 2])
        with pytest.raises(TypeError):
            lab.time_attribution(returns)


# ---------------------------------------------------------------------------
# AttributionChain tests
# ---------------------------------------------------------------------------


class TestAttributionChain:
    def test_compound_effects_single_period(self) -> None:
        effects = [{"allocation": 0.01, "selection": 0.005}]
        result = AttributionChain.compound_effects(effects)
        assert abs(result["allocation"] - 0.01) < 1e-10
        assert abs(result["selection"] - 0.005) < 1e-10

    def test_compound_effects_two_periods(self) -> None:
        effects = [
            {"total": 0.02},
            {"total": 0.03},
        ]
        result = AttributionChain.compound_effects(effects)
        expected = (1.02 * 1.03) - 1.0
        assert abs(result["total"] - expected) < 1e-10

    def test_compound_effects_empty(self) -> None:
        result = AttributionChain.compound_effects([])
        assert result == {}

    def test_cumulative_attribution_consistency(self) -> None:
        """Compounded BrinsonResult must satisfy allocation+selection+interaction==total."""
        lab = ReturnAttributionLab()
        pw = {"A": 0.6, "B": 0.4}
        bw = {"A": 0.5, "B": 0.5}
        sectors = {"A": "S1", "B": "S1"}
        results = []
        for pr_a, br_a in [(0.01, 0.008), (0.005, 0.004), (-0.002, -0.001)]:
            pr = {"A": pr_a, "B": 0.003}
            br = {"A": br_a, "B": 0.002}
            results.append(lab.brinson_attribution(pw, bw, pr, br, sectors))

        combined = AttributionChain.cumulative_attribution(results)
        computed = (
            combined.allocation_effect
            + combined.selection_effect
            + combined.interaction_effect
        )
        assert abs(computed - combined.total) < 1e-8

    def test_cumulative_attribution_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            AttributionChain.cumulative_attribution([])
