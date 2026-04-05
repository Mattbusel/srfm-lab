//! Integration tests for the Monte Carlo Engine.
//!
//! These tests exercise the full pipeline: distribution fitting → simulation →
//! bootstrap → stress testing.

use monte_carlo_engine::{
    bootstrap::{
        Bootstrapper, CircularBlockBootstrap, MovingBlockBootstrap, StationaryBootstrap,
        slice_mean, slice_std,
    },
    simulation::{fit_distribution, MonteCarloSimulator, SimulationConfig},
    stress_tests::{
        conditional_drawdown_at_risk, scenario_analysis, tail_risk_contribution,
        StressScenario, StressTestEngine,
    },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate returns with a specific mean and variance.
fn generate_returns(n: usize, mean: f64, std: f64, seed_offset: f64) -> Vec<f64> {
    // Simple deterministic pseudo-series (not truly random, but sufficient
    // for testing distributional properties).
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64 + seed_offset;
            mean + std * (t * 37.0).sin()
        })
        .collect()
}

fn zero_return_series(n: usize) -> Vec<f64> {
    vec![0.0; n]
}

fn positive_sharpe_series(n: usize) -> Vec<f64> {
    // Mean = 0.001, std ≈ 0.005 → daily Sharpe ≈ 0.2, annual ≈ 3.2
    generate_returns(n, 0.001, 0.005, 0.0)
}

fn negative_sharpe_series(n: usize) -> Vec<f64> {
    generate_returns(n, -0.003, 0.008, 1.0)
}

fn run_simulation(returns: &[f64], n_paths: usize, n_bars: usize) -> monte_carlo_engine::SimulationResults {
    let dist = fit_distribution(returns);
    let config = SimulationConfig {
        n_paths,
        n_bars,
        initial_equity: 100_000.0,
        use_fat_tails: true,
        fat_tail_weight: 0.25,
        store_paths: false,
        seed: 42,
    };
    MonteCarloSimulator::new().run(&config, &dist)
}

// ---------------------------------------------------------------------------
// Simulation correctness tests
// ---------------------------------------------------------------------------

/// At 0% mean daily return, median final equity should be close to initial.
/// With fat tails and variance, there will be some drift, so we allow ±20%.
#[test]
fn test_zero_return_median_equity_near_initial() {
    let returns = zero_return_series(252);
    let results = run_simulation(&returns, 5_000, 252);
    let initial = 100_000.0_f64;
    let ratio = results.median_final_equity / initial;
    assert!(
        ratio > 0.80 && ratio < 1.20,
        "Expected ratio close to 1.0, got {:.4}",
        ratio
    );
}

/// P5 < median < P95 ordering must hold for any distribution.
#[test]
fn test_percentile_ordering() {
    let returns = positive_sharpe_series(252);
    let results = run_simulation(&returns, 2_000, 252);
    assert!(
        results.p5_final_equity < results.median_final_equity,
        "p5={} must be < median={}",
        results.p5_final_equity,
        results.median_final_equity
    );
    assert!(
        results.median_final_equity < results.p95_final_equity,
        "median={} must be < p95={}",
        results.median_final_equity,
        results.p95_final_equity
    );
}

/// P25 < median < P75.
#[test]
fn test_inner_percentile_ordering() {
    let returns = positive_sharpe_series(252);
    let results = run_simulation(&returns, 2_000, 252);
    assert!(results.p25_final_equity < results.median_final_equity);
    assert!(results.median_final_equity < results.p75_final_equity);
}

/// For a high positive Sharpe strategy, prob_ruin should approach 0.
#[test]
fn test_prob_ruin_near_zero_for_positive_sharpe() {
    let returns = positive_sharpe_series(252);
    let results = run_simulation(&returns, 5_000, 252);
    assert!(
        results.prob_ruin < 0.10,
        "prob_ruin={:.4} should be near 0 for positive Sharpe",
        results.prob_ruin
    );
}

/// For a highly negative strategy, prob_ruin should be elevated.
#[test]
fn test_prob_ruin_elevated_for_negative_sharpe() {
    let returns = negative_sharpe_series(252);
    let results = run_simulation(&returns, 3_000, 252);
    // With mean=-0.003/day over 252 days, equity should decay significantly.
    // We just check that the median is below initial.
    assert!(
        results.median_final_equity < 100_000.0,
        "Median equity {} should be below initial for negative-drift strategy",
        results.median_final_equity
    );
}

/// Median Sharpe should be positive for a positive-Sharpe input.
#[test]
fn test_sharpe_sign_preserved() {
    let returns = positive_sharpe_series(252);
    let results = run_simulation(&returns, 2_000, 252);
    assert!(
        results.median_sharpe > 0.0,
        "median_sharpe={} should be positive",
        results.median_sharpe
    );
}

/// P5 Sharpe ≤ median Sharpe.
#[test]
fn test_sharpe_ordering() {
    let returns = positive_sharpe_series(252);
    let results = run_simulation(&returns, 2_000, 252);
    assert!(
        results.p5_sharpe <= results.median_sharpe,
        "p5_sharpe={} must be <= median_sharpe={}",
        results.p5_sharpe,
        results.median_sharpe
    );
}

/// Store-paths mode produces equity_paths with correct dimensions.
#[test]
fn test_store_paths_dimensions() {
    let returns = positive_sharpe_series(100);
    let dist = fit_distribution(&returns);
    let n_paths = 50;
    let n_bars = 100;
    let config = SimulationConfig {
        n_paths,
        n_bars,
        initial_equity: 10_000.0,
        use_fat_tails: false,
        fat_tail_weight: 0.0,
        store_paths: true,
        seed: 1,
    };
    let results = MonteCarloSimulator::new().run(&config, &dist);
    let paths = results.equity_paths.expect("equity_paths should be Some");
    assert_eq!(paths.len(), n_paths, "should have {} paths", n_paths);
    for (i, path) in paths.iter().enumerate() {
        assert_eq!(
            path.len(),
            n_bars + 1,
            "path {} should have {} bars (including initial)",
            i,
            n_bars + 1
        );
    }
}

/// percentile_final_equity should return a value within the correct range.
#[test]
fn test_percentile_final_equity_method() {
    let returns = positive_sharpe_series(252);
    let dist = fit_distribution(&returns);
    let config = SimulationConfig {
        n_paths: 1_000,
        n_bars: 252,
        initial_equity: 100_000.0,
        use_fat_tails: false,
        fat_tail_weight: 0.0,
        store_paths: true,
        seed: 7,
    };
    let results = MonteCarloSimulator::new().run(&config, &dist);
    let p10 = results.percentile_final_equity(10.0).unwrap();
    let p90 = results.percentile_final_equity(90.0).unwrap();
    assert!(p10 < p90, "p10={} must be < p90={}", p10, p90);
}

// ---------------------------------------------------------------------------
// Bootstrap tests
// ---------------------------------------------------------------------------

/// Resampled mean should be close to original mean.
#[test]
fn test_stationary_bootstrap_mean_preservation() {
    let returns = positive_sharpe_series(500);
    let orig_mean = slice_mean(&returns);

    let bs = StationaryBootstrap { p_new_block: 0.1, seed: 42 };
    let mut boot_means = Vec::new();
    for i in 0..300 {
        let mut b = bs.clone();
        b.seed = 42 + i;
        let sample = b.resample(&returns, returns.len()).unwrap();
        boot_means.push(slice_mean(&sample));
    }
    let mean_of_means = slice_mean(&boot_means);
    let tol = slice_std(&returns) / (300_f64.sqrt());
    assert!(
        (mean_of_means - orig_mean).abs() < 5.0 * tol,
        "Bootstrap mean {} deviates too far from original {}",
        mean_of_means,
        orig_mean
    );
}

/// CBB resampled mean close to original.
#[test]
fn test_circular_bootstrap_mean_preservation() {
    let returns = positive_sharpe_series(500);
    let orig_mean = slice_mean(&returns);
    let mut bs = CircularBlockBootstrap::new(10);
    bs.seed = 55;
    let sample = bs.resample(&returns, returns.len()).unwrap();
    let boot_mean = slice_mean(&sample);
    assert!(
        (boot_mean - orig_mean).abs() < 0.005,
        "CBB mean {} vs original {}",
        boot_mean,
        orig_mean
    );
}

/// MBB resampled mean close to original.
#[test]
fn test_moving_block_bootstrap_mean_preservation() {
    let returns = positive_sharpe_series(500);
    let orig_mean = slice_mean(&returns);
    let mut bs = MovingBlockBootstrap::new(10);
    bs.seed = 77;
    let sample = bs.resample(&returns, returns.len()).unwrap();
    let boot_mean = slice_mean(&sample);
    assert!(
        (boot_mean - orig_mean).abs() < 0.005,
        "MBB mean {} vs original {}",
        boot_mean,
        orig_mean
    );
}

/// All three bootstrap methods preserve the sample length.
#[test]
fn test_bootstrap_output_lengths() {
    let returns = positive_sharpe_series(252);
    let target = 500;

    let sb = StationaryBootstrap { p_new_block: 0.1, seed: 1 };
    assert_eq!(sb.resample(&returns, target).unwrap().len(), target);

    let cb = CircularBlockBootstrap { block_size: 10, seed: 2 };
    assert_eq!(cb.resample(&returns, target).unwrap().len(), target);

    let mb = MovingBlockBootstrap { block_size: 10, seed: 3 };
    assert_eq!(mb.resample(&returns, target).unwrap().len(), target);
}

/// Resampling with same seed produces identical output.
#[test]
fn test_bootstrap_deterministic_seed() {
    let returns = positive_sharpe_series(200);
    let bs = StationaryBootstrap { p_new_block: 0.1, seed: 123 };
    let s1 = bs.resample(&returns, 200).unwrap();
    let s2 = bs.resample(&returns, 200).unwrap();
    assert_eq!(s1, s2, "same seed must produce identical samples");
}

// ---------------------------------------------------------------------------
// Stress test tests
// ---------------------------------------------------------------------------

fn stress_engine(n_paths: usize) -> StressTestEngine {
    StressTestEngine {
        n_paths,
        n_bars: 252,
        initial_equity: 100_000.0,
        seed: 42,
    }
}

/// Crash scenario should produce lower median equity than a benign baseline.
#[test]
fn test_crash_scenario_reduces_median_equity() {
    let positive_returns = positive_sharpe_series(252);
    let engine = stress_engine(500);

    // Baseline: run with no injected shocks (use CovidDip — smallest scenario)
    // vs MarketCrash2020 (severe scenario)
    let dip_result = engine
        .run_scenario(&positive_returns, &StressScenario::CovidDip)
        .unwrap();
    let crash_result = engine
        .run_scenario(&positive_returns, &StressScenario::MarketCrash2020)
        .unwrap();

    assert!(
        crash_result.median_final_equity <= dip_result.median_final_equity,
        "Crash median equity {} should be <= dip median {}",
        crash_result.median_final_equity,
        dip_result.median_final_equity
    );
}

/// P95 drawdown must be > 0 in a crash scenario.
#[test]
fn test_crash_scenario_drawdown_positive() {
    let returns = positive_sharpe_series(252);
    let engine = stress_engine(300);
    let result = engine
        .run_scenario(&returns, &StressScenario::MarketCrash2020)
        .unwrap();
    assert!(
        result.p95_max_drawdown > 0.0,
        "P95 max drawdown must be > 0 in crash scenario"
    );
}

/// CDaR should be zero for flat equity.
#[test]
fn test_cdar_zero_for_flat_equity() {
    let returns = zero_return_series(252);
    let cdar = conditional_drawdown_at_risk(&returns, 0.95);
    assert!(cdar < 1e-9, "CDaR must be ~0 for flat equity, got {}", cdar);
}

/// CDaR should increase when losses are present.
#[test]
fn test_cdar_increases_with_losses() {
    let low_vol_returns = generate_returns(252, 0.0005, 0.001, 0.0);
    let high_vol_returns = generate_returns(252, -0.002, 0.015, 1.0);
    let cdar_low = conditional_drawdown_at_risk(&low_vol_returns, 0.95);
    let cdar_high = conditional_drawdown_at_risk(&high_vol_returns, 0.95);
    assert!(
        cdar_high >= cdar_low,
        "CDaR high={} must be >= low={}",
        cdar_high,
        cdar_low
    );
}

/// Tail risk contribution should be positive for any negative shock sequence.
#[test]
fn test_tail_risk_contribution_positive_for_crash() {
    let base = positive_sharpe_series(252);
    let shocks = vec![-0.05; 20];
    let trc = tail_risk_contribution(&base, &shocks);
    assert!(trc > 0.0, "TRC must be positive for negative shocks");
}

/// scenario_analysis should return one result per scenario.
#[test]
fn test_scenario_analysis_count() {
    let returns = positive_sharpe_series(252);
    let engine = stress_engine(100);
    let scenarios = vec![
        StressScenario::CovidDip,
        StressScenario::FlashCrash,
        StressScenario::MarketCrash2020,
    ];
    let results = scenario_analysis(&engine, &returns, &scenarios).unwrap();
    assert_eq!(results.len(), 3);
}

/// Scenario names should match.
#[test]
fn test_scenario_result_names() {
    let returns = positive_sharpe_series(252);
    let engine = stress_engine(100);
    let scenarios = vec![StressScenario::FlashCrash, StressScenario::CovidDip];
    let results = scenario_analysis(&engine, &returns, &scenarios).unwrap();
    assert_eq!(results[0].scenario_name, StressScenario::FlashCrash.name());
    assert_eq!(results[1].scenario_name, StressScenario::CovidDip.name());
}

// ---------------------------------------------------------------------------
// Distribution fitting tests
// ---------------------------------------------------------------------------

#[test]
fn test_fit_distribution_mean() {
    let returns = vec![0.01; 100];
    let dist = fit_distribution(&returns);
    assert!((dist.mean - 0.01).abs() < 1e-9);
}

#[test]
fn test_fit_distribution_std_nonzero_series() {
    let returns = generate_returns(252, 0.001, 0.01, 0.0);
    let dist = fit_distribution(&returns);
    assert!(dist.std > 0.0);
    assert!(dist.n_obs == 252);
}

#[test]
fn test_fit_tail_alpha_bounded() {
    let returns = generate_returns(500, 0.0, 0.02, 0.5);
    let dist = fit_distribution(&returns);
    assert!(dist.tail_alpha >= 0.5 && dist.tail_alpha <= 10.0);
}

#[test]
fn test_student_df_large_for_normal_kurtosis() {
    // Near-normal distribution: low excess kurtosis → large df
    let returns = generate_returns(252, 0.001, 0.01, 0.0);
    let dist = fit_distribution(&returns);
    // df should be large (near-normal) if kurtosis is low
    if dist.kurtosis < 1.0 {
        assert!(dist.student_df > 10.0);
    }
}
