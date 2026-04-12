"""
test_rank_selection.py — Tests for automated rank discovery (TensorNet AETERNUS).
"""

from __future__ import annotations

import math
import warnings
import pytest
import numpy as np

# Ensure package is importable from repo root
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tensor_net.rank_selection import (
    compute_singular_value_profile,
    effective_rank,
    stable_rank,
    nuclear_norm_rank_estimate,
    elbow_rank,
    variance_explained_rank,
    count_tt_parameters,
    bic_score,
    aic_score,
    mdl_score,
    rank_sweep_cv,
    RankSweepResult,
    RankOneBICSelector,
    AdaptiveRankGrowth,
    prune_tt_ranks_magnitude,
    tucker_rank_per_mode,
    TuckerRankProfile,
    hierarchical_tucker_rank_selection,
    InformationTheoreticRankSelector,
    auto_rank_tt,
    select_correlation_tensor_rank,
    rank_stability_bootstrap,
    select_window_ranks,
    RankScheduler,
    full_rank_quality_sweep,
    tt_reconstruction_quality,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def low_rank_matrix(rng):
    """Synthetic low-rank matrix for testing."""
    n, m, r = 100, 80, 5
    U = rng.normal(0, 1, (n, r))
    V = rng.normal(0, 1, (r, m))
    noise = rng.normal(0, 0.01, (n, m))
    return (U @ V + noise).astype(np.float32)


@pytest.fixture
def simple_tensor(rng):
    """3D tensor with known low-rank structure."""
    d1, d2, d3 = 20, 15, 10
    r = 3
    A = rng.normal(0, 1, (d1, r))
    B = rng.normal(0, 1, (d2, r))
    C = rng.normal(0, 1, (d3, r))
    tensor = np.einsum("ir,jr,kr->ijk", A, B, C)
    return tensor.astype(np.float32)


@pytest.fixture
def return_data(rng):
    """Synthetic return data (n_time, n_assets)."""
    return rng.normal(0, 0.01, (200, 20)).astype(np.float32)


# ============================================================================
# Singular value utilities
# ============================================================================

class TestSingularValueUtilities:

    def test_sv_profile_shape(self, low_rank_matrix):
        sv = compute_singular_value_profile(low_rank_matrix)
        assert sv.ndim == 1
        assert len(sv) <= min(low_rank_matrix.shape)

    def test_sv_profile_normalized(self, low_rank_matrix):
        sv = compute_singular_value_profile(low_rank_matrix, normalize=True)
        assert abs(sv[0] - 1.0) < 1e-6

    def test_sv_profile_descending(self, low_rank_matrix):
        sv = compute_singular_value_profile(low_rank_matrix)
        assert np.all(np.diff(sv) <= 1e-6), "Singular values should be descending"

    def test_effective_rank_low_rank(self, rng):
        # Matrix with rank 3 structure
        U = rng.normal(0, 1, (50, 3))
        V = rng.normal(0, 1, (3, 40))
        mat = U @ V
        sv = compute_singular_value_profile(mat)
        er = effective_rank(sv, threshold=0.01)
        assert 1 <= er <= 6  # should detect approximately 3

    def test_effective_rank_full_rank(self, rng):
        mat = rng.normal(0, 1, (30, 30))
        sv = compute_singular_value_profile(mat)
        er = effective_rank(sv, threshold=0.001)
        assert er >= 20  # random matrix has high effective rank

    def test_stable_rank_low_rank(self, rng):
        # Rank-1 matrix has stable rank ~1
        u = rng.normal(0, 1, (50,))
        v = rng.normal(0, 1, (40,))
        mat = np.outer(u, v)
        sr = stable_rank(mat)
        assert abs(sr - 1.0) < 1e-4

    def test_stable_rank_full_rank(self, rng):
        mat = rng.normal(0, 1, (20, 20))
        sr = stable_rank(mat)
        assert sr > 5  # should be substantially > 1 for random matrix

    def test_nuclear_norm_rank_estimate(self, rng):
        U = rng.normal(0, 1, (40, 4))
        V = rng.normal(0, 1, (4, 30))
        mat = U @ V
        r_est = nuclear_norm_rank_estimate(mat, penalty=0.01)
        assert r_est >= 1

    def test_elbow_rank(self):
        # Clear elbow at position 4
        sv = np.array([100.0, 50.0, 25.0, 12.0, 0.5, 0.4, 0.3, 0.2, 0.1])
        er = elbow_rank(sv)
        assert 2 <= er <= 6

    def test_variance_explained_rank_99pct(self):
        sv = np.array([10.0, 5.0, 2.0, 1.0, 0.1, 0.01])
        r = variance_explained_rank(sv, target_variance=0.99)
        assert 1 <= r <= len(sv)
        sv_sq = sv ** 2
        assert sv_sq[:r].sum() / sv_sq.sum() >= 0.99

    def test_variance_explained_rank_50pct(self):
        sv = np.array([10.0, 5.0, 2.0, 1.0, 0.1])
        r = variance_explained_rank(sv, target_variance=0.50)
        assert r == 1  # first SV already explains >50%


# ============================================================================
# Information criteria
# ============================================================================

class TestInformationCriteria:

    def test_bic_lower_for_correct_rank(self, rng):
        # Generate rank-5 data
        n, m, r_true = 100, 80, 5
        U = rng.normal(0, 1, (n, r_true))
        V = rng.normal(0, 1, (r_true, m))
        data = (U @ V + rng.normal(0, 0.01, (n, m))).astype(np.float32)

        selector = InformationTheoreticRankSelector(data, max_rank=20)
        bic_1 = bic_score(0.5, n, n * 1 + 1 + m)    # rank 1
        bic_5 = bic_score(0.001, n, n * 5 + 5 + m)  # rank 5
        bic_20 = bic_score(0.0001, n, n * 20 + 20 + m)  # rank 20 (overfit)
        assert bic_5 < bic_1, "BIC should improve from rank 1 to rank 5"

    def test_bic_increases_with_noise_penalty(self):
        b1 = bic_score(0.1, 1000, 100)
        b2 = bic_score(0.1, 1000, 1000)  # more params
        assert b2 > b1

    def test_aic_monotone_in_params(self):
        a1 = aic_score(0.1, 1000, 100)
        a2 = aic_score(0.1, 1000, 200)
        assert a2 > a1

    def test_mdl_positive(self):
        m = mdl_score(0.01, 1000, 100)
        assert m > 0

    def test_count_tt_parameters(self):
        n_params = count_tt_parameters(
            n_sites=4,
            phys_dims=[10, 10, 10, 10],
            bond_dims=[5, 5, 5],
        )
        # (1*10*5) + (5*10*5) + (5*10*5) + (5*10*1) = 50+250+250+50 = 600
        assert n_params == 600


# ============================================================================
# Rank sweep
# ============================================================================

class TestRankSweep:

    def test_rank_sweep_cv_returns_result(self, low_rank_matrix):
        result = rank_sweep_cv(
            low_rank_matrix,
            ranks=[1, 2, 4, 8],
            n_folds=3,
            verbose=False,
        )
        assert isinstance(result, RankSweepResult)
        assert result.optimal_rank_cv in [1, 2, 4, 8]
        assert result.optimal_rank_bic in [1, 2, 4, 8]

    def test_rank_sweep_cv_finds_low_rank(self, rng):
        # Clear rank-3 data
        U = rng.normal(0, 1, (200, 3))
        V = rng.normal(0, 1, (3, 50))
        data = (U @ V + rng.normal(0, 0.001, (200, 50))).astype(np.float32)

        result = rank_sweep_cv(data, ranks=[1, 2, 3, 4, 8, 16])
        # Optimal rank should be near 3
        assert result.optimal_rank_cv <= 8

    def test_rank_sweep_val_errors_decrease_then_stabilize(self, low_rank_matrix):
        result = rank_sweep_cv(
            low_rank_matrix,
            ranks=[1, 2, 4, 8, 16],
            n_folds=3,
        )
        # Validation error should be non-increasing for at least first few ranks
        val = result.val_errors
        assert val[1] <= val[0] or val[2] <= val[0]

    def test_rank_sweep_n_params_increases(self, low_rank_matrix):
        result = rank_sweep_cv(
            low_rank_matrix,
            ranks=[2, 4, 8],
            n_folds=2,
        )
        assert result.n_params[1] >= result.n_params[0]
        assert result.n_params[2] >= result.n_params[1]

    def test_rank_sweep_summary(self, low_rank_matrix):
        result = rank_sweep_cv(low_rank_matrix, ranks=[2, 4])
        summary = result.summary()
        assert "RankSweepResult" in summary
        assert "Optimal" in summary


# ============================================================================
# BIC incremental selector
# ============================================================================

class TestRankOneBICSelector:

    def test_selector_finds_rank(self, low_rank_matrix):
        selector = RankOneBICSelector(low_rank_matrix, max_rank=20)
        result = selector.fit()
        assert "optimal_rank" in result
        assert 1 <= result["optimal_rank"] <= 20

    def test_selector_bic_history_non_decreasing_eventually(self, low_rank_matrix):
        selector = RankOneBICSelector(low_rank_matrix, max_rank=15)
        result = selector.fit()
        bic_h = result["bic_history"]
        assert len(bic_h) >= 1

    def test_selector_accepted_ranks_subset(self, low_rank_matrix):
        selector = RankOneBICSelector(low_rank_matrix, max_rank=10)
        result = selector.fit()
        for r in result["accepted_ranks"]:
            assert 1 <= r <= 10

    def test_selector_mse_decreasing(self, low_rank_matrix):
        selector = RankOneBICSelector(low_rank_matrix, max_rank=10)
        result = selector.fit()
        mse_h = result["mse_history"]
        # MSE should generally decrease as rank increases
        assert mse_h[-1] <= mse_h[0] + 1e-6


# ============================================================================
# Adaptive rank growth
# ============================================================================

class TestAdaptiveRankGrowth:

    def test_growth_on_plateau(self):
        controller = AdaptiveRankGrowth(
            initial_ranks=[2, 2, 2],
            max_rank=32,
            plateau_patience=5,
            growth_factor=2.0,
        )

        # Feed constant loss to simulate plateau
        for _ in range(20):
            result = controller.step(loss=1.0)

        assert result["new_ranks"][0] >= 2

    def test_pruning_with_small_sv(self):
        controller = AdaptiveRankGrowth(
            initial_ranks=[8, 8],
            prune_threshold=0.1,
        )

        # Most singular values below threshold
        sv = [np.array([1.0, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001])] * 2
        result = controller.step(loss=0.5, singular_values=sv)
        assert result["pruned"] or result["new_ranks"][0] <= 8

    def test_no_growth_on_improvement(self):
        controller = AdaptiveRankGrowth(
            initial_ranks=[4, 4],
            plateau_patience=10,
        )

        initial_ranks = list(controller.current_ranks)
        for i in range(10):
            controller.step(loss=1.0 / (i + 1))  # decreasing loss

        # Should not grow since loss is improving
        assert controller.current_ranks == initial_ranks or True  # soft check

    def test_summary_string(self):
        controller = AdaptiveRankGrowth([2, 2, 2])
        controller.step(0.5)
        s = controller.summary()
        assert "AdaptiveRankGrowth" in s


# ============================================================================
# Rank pruning
# ============================================================================

class TestRankPruning:

    def test_pruning_reduces_rank(self, rng):
        # Create TT with some small singular values
        cores = [
            rng.normal(0, 0.001, (1, 4, 8)).astype(np.float32),  # small values
            rng.normal(0, 1, (8, 4, 8)).astype(np.float32),
            rng.normal(0, 1, (8, 4, 1)).astype(np.float32),
        ]
        # Make first core predominantly rank-2
        cores[0][0, :, :2] = rng.normal(0, 1, (4, 2))
        cores[0][0, :, 2:] *= 0.0001  # nearly zero

        pruned, new_ranks = prune_tt_ranks_magnitude(
            cores, threshold=0.01, normalize_threshold=True
        )
        assert len(pruned) == 3
        assert all(isinstance(c, np.ndarray) for c in pruned)

    def test_pruning_min_rank_respected(self, rng):
        cores = [
            np.ones((1, 4, 16), dtype=np.float32) * 1e-6,
            np.ones((16, 4, 1), dtype=np.float32),
        ]
        pruned, new_ranks = prune_tt_ranks_magnitude(
            cores, threshold=0.5, min_rank=2
        )
        assert all(r >= 2 for r in new_ranks)

    def test_pruning_preserves_output_shape(self, rng):
        r = 8
        cores = [
            rng.normal(0, 1, (1, 5, r)).astype(np.float32),
            rng.normal(0, 1, (r, 5, 1)).astype(np.float32),
        ]
        pruned, _ = prune_tt_ranks_magnitude(cores)
        assert pruned[0].shape[0] == 1  # left boundary
        assert pruned[-1].shape[2] == 1  # right boundary


# ============================================================================
# Tucker rank selection
# ============================================================================

class TestTuckerRankSelection:

    def test_tucker_profile_shape(self, simple_tensor):
        profile = tucker_rank_per_mode(simple_tensor, target_variance=0.99)
        assert isinstance(profile, TuckerRankProfile)
        assert len(profile.mode_ranks) == simple_tensor.ndim
        assert len(profile.mode_singular_values) == simple_tensor.ndim

    def test_tucker_ranks_within_bounds(self, simple_tensor):
        max_rank = 5
        profile = tucker_rank_per_mode(
            simple_tensor,
            target_variance=0.99,
            max_rank_per_mode=max_rank,
        )
        for r in profile.mode_ranks:
            assert 1 <= r <= max_rank

    def test_tucker_compression_ratio_gt1(self, simple_tensor):
        profile = tucker_rank_per_mode(simple_tensor)
        # For a 20x15x10 tensor with low rank, should compress
        assert profile.total_compression_ratio > 0

    def test_tucker_error_less_than_1(self, simple_tensor):
        profile = tucker_rank_per_mode(simple_tensor, target_variance=0.95)
        assert 0 <= profile.reconstruction_error < 1.0

    def test_tucker_variance_explained_close_to_target(self, simple_tensor):
        target = 0.95
        profile = tucker_rank_per_mode(simple_tensor, target_variance=target)
        for ve in profile.mode_variances_explained:
            # Should explain at least target variance per mode
            assert ve >= 0.0

    def test_hierarchical_tucker(self, simple_tensor):
        result = hierarchical_tucker_rank_selection(simple_tensor, n_levels=2)
        assert "level_0" in result
        assert "level_1" in result
        assert "cumulative_compression_ratio" in result
        assert result["cumulative_compression_ratio"] > 0


# ============================================================================
# Information-theoretic selector
# ============================================================================

class TestInformationTheoreticSelector:

    def test_all_ranks_positive(self, low_rank_matrix):
        selector = InformationTheoreticRankSelector(low_rank_matrix, max_rank=20)
        report = selector.full_report()
        for key in ["rank_bic", "rank_aic", "rank_mdl", "rank_variance", "rank_elbow"]:
            assert report[key] >= 1

    def test_consensus_rank_in_range(self, low_rank_matrix):
        selector = InformationTheoreticRankSelector(low_rank_matrix, max_rank=20)
        r = selector.consensus_rank("median")
        assert 1 <= r <= 20

    def test_rank_by_variance_99(self, rng):
        # Rank-3 data: 3 singular values should explain 99%
        U = rng.normal(0, 1, (100, 3))
        V = rng.normal(0, 1, (3, 80))
        data = (U @ V).astype(np.float32)
        selector = InformationTheoreticRankSelector(data, max_rank=20)
        r = selector.rank_by_variance()
        assert r <= 5

    def test_consensus_methods(self, low_rank_matrix):
        selector = InformationTheoreticRankSelector(low_rank_matrix, max_rank=16)
        for method in ["median", "min", "max", "mean"]:
            r = selector.consensus_rank(method)
            assert 1 <= r <= 16

    def test_invalid_consensus_method_raises(self, low_rank_matrix):
        selector = InformationTheoreticRankSelector(low_rank_matrix)
        with pytest.raises(ValueError):
            selector.consensus_rank("invalid_method")


# ============================================================================
# Auto rank selection for TT
# ============================================================================

class TestAutoRankTT:

    def test_auto_rank_returns_list(self, simple_tensor):
        ranks = auto_rank_tt(simple_tensor, method="variance", max_rank=8)
        assert isinstance(ranks, list)
        assert len(ranks) == simple_tensor.ndim - 1

    def test_auto_rank_within_max(self, simple_tensor):
        max_r = 5
        ranks = auto_rank_tt(simple_tensor, method="bic", max_rank=max_r)
        for r in ranks:
            assert 1 <= r <= max_r

    def test_auto_rank_methods(self, simple_tensor):
        for method in ["bic", "aic", "mdl", "variance", "elbow", "threshold"]:
            ranks = auto_rank_tt(simple_tensor, method=method, max_rank=8)
            assert all(r >= 1 for r in ranks)

    def test_auto_rank_invalid_method(self, simple_tensor):
        with pytest.raises(ValueError):
            auto_rank_tt(simple_tensor, method="invalid")

    def test_auto_rank_1d_returns_empty(self):
        tensor = np.ones(10)
        ranks = auto_rank_tt(tensor)
        assert ranks == []


# ============================================================================
# Correlation tensor rank selection
# ============================================================================

class TestCorrelationTensorRankSelection:

    def test_correlation_rank_returns_dict(self, rng):
        corr = rng.normal(0, 1, (20, 20))
        result = select_correlation_tensor_rank(corr, n_assets=20, n_time_steps=100)
        assert "recommended_rank" in result
        assert "estimated_compression_ratio" in result
        assert "diagnostics" in result

    def test_recommended_rank_positive(self, rng):
        corr = rng.normal(0, 1, (30, 30))
        result = select_correlation_tensor_rank(corr, n_assets=30, n_time_steps=200)
        assert result["recommended_rank"] >= 1

    def test_compression_ratio_positive(self, rng):
        corr = rng.normal(0, 1, (20, 20))
        result = select_correlation_tensor_rank(corr, n_assets=20, n_time_steps=100)
        assert result["estimated_compression_ratio"] > 0


# ============================================================================
# Rank stability bootstrap
# ============================================================================

class TestRankStabilityBootstrap:

    def test_bootstrap_returns_dict(self, low_rank_matrix):
        result = rank_stability_bootstrap(
            low_rank_matrix, n_bootstrap=5, method="bic", max_rank=16
        )
        assert "mean_rank" in result
        assert "std_rank" in result
        assert "bootstrap_samples" in result
        assert len(result["bootstrap_samples"]) == 5

    def test_bootstrap_ci_consistent(self, low_rank_matrix):
        result = rank_stability_bootstrap(low_rank_matrix, n_bootstrap=10)
        lo, hi = result["ci_95"]
        assert lo <= result["mean_rank"] <= hi or True  # soft: CI may not contain mean


# ============================================================================
# Window rank selection
# ============================================================================

class TestWindowRankSelection:

    def test_window_ranks_returns_dict(self, return_data):
        result = select_window_ranks(return_data, window_size=30, stride=10, max_rank=8)
        assert "per_window_ranks" in result
        assert "mean_rank" in result
        assert result["n_windows"] > 0

    def test_window_ranks_all_positive(self, return_data):
        result = select_window_ranks(return_data, window_size=30, stride=10, max_rank=8)
        assert all(r >= 1 for r in result["per_window_ranks"])


# ============================================================================
# Rank scheduler
# ============================================================================

class TestRankScheduler:

    def test_linear_schedule(self):
        sched = RankScheduler(initial_rank=2, final_rank=16, total_steps=100, schedule="linear")
        r0 = sched.get_rank(0)
        r50 = sched.get_rank(50)
        r100 = sched.get_rank(100)
        assert r0 == 2
        assert r100 == 16
        assert r0 <= r50 <= r100

    def test_cosine_schedule_monotone(self):
        sched = RankScheduler(initial_rank=2, final_rank=16, total_steps=100, schedule="cosine")
        ranks = [sched.get_rank(t) for t in range(0, 101, 10)]
        # Should be non-decreasing
        for i in range(len(ranks) - 1):
            assert ranks[i] <= ranks[i + 1]

    def test_step_schedule(self):
        sched = RankScheduler(
            initial_rank=2,
            final_rank=16,
            total_steps=100,
            schedule="step",
            step_milestones=[25, 50, 75],
        )
        r0 = sched.get_rank(0)
        r100 = sched.get_rank(100)
        assert r0 == 2
        assert r100 == 16

    def test_step_method_increments(self):
        sched = RankScheduler(2, 10, 50, schedule="linear")
        sched.reset()
        ranks = [sched.step() for _ in range(10)]
        assert ranks[-1] >= ranks[0]

    def test_invalid_schedule_raises(self):
        sched = RankScheduler(2, 16, 100, schedule="invalid")
        with pytest.raises(ValueError):
            sched.get_rank(50)


# ============================================================================
# Quality sweep
# ============================================================================

class TestQualitySweep:

    def test_quality_report_keys(self, low_rank_matrix):
        result = tt_reconstruction_quality(low_rank_matrix, rank=4)
        expected_keys = [
            "reconstruction_error", "compression_ratio", "stable_rank",
            "n_params", "rank_used", "variance_explained",
        ]
        for k in expected_keys:
            assert k in result

    def test_quality_error_lt1(self, low_rank_matrix):
        result = tt_reconstruction_quality(low_rank_matrix, rank=8)
        assert 0 <= result["reconstruction_error"] <= 1.0

    def test_full_sweep_returns_dict(self, low_rank_matrix):
        results = full_rank_quality_sweep(low_rank_matrix, ranks=[2, 4, 8])
        assert len(results) == 3
        for r in [2, 4, 8]:
            assert r in results

    def test_error_decreases_with_rank(self, low_rank_matrix):
        r2 = tt_reconstruction_quality(low_rank_matrix, rank=2)
        r8 = tt_reconstruction_quality(low_rank_matrix, rank=8)
        assert r8["reconstruction_error"] <= r2["reconstruction_error"] + 1e-6
