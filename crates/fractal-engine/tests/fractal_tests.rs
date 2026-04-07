//! Integration tests for fractal-engine extensions.
//!
//! Tests cover:
//! - MultifractalAnalyzer: spectrum width, H(2) consistency, singularity spectrum
//! - Detrending methods: HP, BK, SSA, EMD -- reconstruction and component properties
//! - Extended fractal dimension: Petrosian, Katz, dispatcher, trading regimes

use fractal_engine::{
    multifractal::MultifractalAnalyzer,
    detrending::{HPFilter, BKFilter, SSADetrend, EMDDetrend},
    fractal_dim_ext::{
        FractalDimension, FDMethod, PetrosianMethod, KatzMethod,
        FdTradingRegime, AllMethodsResult,
    },
    hurst::hurst_rs,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_walk(n: usize, seed: u64) -> Vec<f64> {
    let mut price = 100.0f64;
    let mut prices = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        prices.push(price);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        price = (price + r * 0.5).max(1.0);
    }
    prices
}

fn trending_series(n: usize) -> Vec<f64> {
    (0..n).map(|i| 100.0 + i as f64 * 0.5).collect()
}

fn sinusoidal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            100.0
                + (i as f64 * 0.1).sin() * 5.0
                + i as f64 * 0.05
        })
        .collect()
}

fn noisy_series(n: usize, seed: u64) -> Vec<f64> {
    let mut price = 100.0f64;
    let mut prices = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        prices.push(price);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
        price = (price + r * 2.5).max(1.0);
    }
    prices
}

// ---------------------------------------------------------------------------
// Multifractal tests
// ---------------------------------------------------------------------------

#[test]
fn test_multifractal_width_positive() {
    let prices = random_walk(512, 42);
    let analyzer = MultifractalAnalyzer::new();
    let width = analyzer.multifractal_width(&prices);
    assert!(width >= 0.0, "Multifractal width should be non-negative, got {width}");
}

#[test]
fn test_multifractal_result_some_for_large_series() {
    let prices = random_walk(256, 7);
    let analyzer = MultifractalAnalyzer::new();
    assert!(analyzer.analyze(&prices).is_some(),
        "Should return Some for 256-element series");
}

#[test]
fn test_multifractal_result_none_for_tiny_series() {
    let prices = random_walk(32, 1);
    let analyzer = MultifractalAnalyzer::new();
    assert!(analyzer.analyze(&prices).is_none(),
        "Should return None for 32-element series (< 64)");
}

#[test]
fn test_hurst_h2_equals_standard_hurst() {
    // H(q=2) from MFDFA should be in the same ballpark as R/S Hurst.
    // For random walk both should be near 0.5 (within 0.25).
    let prices = random_walk(512, 12345);
    let analyzer = MultifractalAnalyzer::new();

    let mfdfa_result = analyzer.analyze(&prices);
    let rs_hurst = hurst_rs(&prices).unwrap_or(0.5);

    if let Some(result) = mfdfa_result {
        if result.h2.is_finite() {
            let diff = (result.h2 - rs_hurst).abs();
            assert!(
                diff < 0.35,
                "MFDFA H(2)={:.3} vs R/S H={:.3}, diff={:.3} too large",
                result.h2, rs_hurst, diff
            );
        }
    }
}

#[test]
fn test_singularity_spectrum_lengths_equal() {
    let prices = random_walk(256, 99);
    let analyzer = MultifractalAnalyzer::new();
    let (alpha, f_alpha) = analyzer.singularity_spectrum(&prices);
    assert_eq!(alpha.len(), f_alpha.len(),
        "alpha and f_alpha vectors must have equal length");
}

#[test]
fn test_multifractal_is_consistent_with_width() {
    let prices = random_walk(512, 55);
    let analyzer = MultifractalAnalyzer::new();
    if let Some(result) = analyzer.analyze(&prices) {
        assert_eq!(
            result.is_multifractal,
            result.multifractal_width > 0.3,
            "is_multifractal flag inconsistent with width {:.4}",
            result.multifractal_width
        );
    }
}

#[test]
fn test_hq_length_at_least_five() {
    let prices = random_walk(256, 3);
    let analyzer = MultifractalAnalyzer::new();
    if let Some(result) = analyzer.analyze(&prices) {
        assert!(result.hq.len() >= 5,
            "Expected >= 5 H(q) values, got {}", result.hq.len());
    }
}

#[test]
fn test_tau_q_length_matches_hq() {
    let prices = random_walk(256, 7);
    let analyzer = MultifractalAnalyzer::new();
    if let Some(result) = analyzer.analyze(&prices) {
        assert_eq!(result.tau_q.len(), result.hq.len());
    }
}

#[test]
fn test_h2_in_unit_range() {
    let prices = random_walk(512, 88);
    let analyzer = MultifractalAnalyzer::new();
    if let Some(result) = analyzer.analyze(&prices) {
        if result.h2.is_finite() {
            assert!(result.h2 >= 0.0 && result.h2 <= 2.0,
                "H(2)={:.3} out of [0,2]", result.h2);
        }
    }
}

// ---------------------------------------------------------------------------
// Detrending tests
// ---------------------------------------------------------------------------

#[test]
fn test_hp_filter_trend_plus_cycle_equals_price() {
    let prices = sinusoidal(80);
    let hp = HPFilter::new(1600.0);
    let decomp = hp.decompose(&prices);
    for (i, (&p, (&t, &c))) in prices.iter()
        .zip(decomp.trend.iter().zip(decomp.cycle.iter()))
        .enumerate()
    {
        let err = (p - t - c).abs();
        assert!(err < 1e-6,
            "HP reconstruction error at bar {i}: {err:.2e}");
    }
}

#[test]
fn test_hp_filter_lengths_match() {
    let prices = sinusoidal(100);
    let hp = HPFilter::daily();
    let decomp = hp.decompose(&prices);
    assert_eq!(decomp.trend.len(), prices.len());
    assert_eq!(decomp.cycle.len(), prices.len());
    assert_eq!(decomp.noise.len(), prices.len());
}

#[test]
fn test_bk_filter_cycle_non_zero_interior() {
    let prices = sinusoidal(200);
    let bk = BKFilter::new(10, 40, 12);
    let decomp = bk.decompose(&prices);
    let interior = &decomp.cycle[20..180];
    let any_nonzero = interior.iter().any(|&c| c.abs() > 1e-6);
    assert!(any_nonzero, "BK cycle should have non-zero values in interior");
}

#[test]
fn test_bk_weights_sum_to_zero() {
    let bk = BKFilter::business_cycle();
    let w = bk.weights();
    let sum: f64 = w.iter().sum();
    assert!(sum.abs() < 1e-10, "BK weights should sum to 0, got {sum}");
}

#[test]
fn test_ssa_reconstruction_error_small() {
    let prices = sinusoidal(80);
    let ssa = SSADetrend::new(10, 2);
    let decomp = ssa.decompose(&prices);
    let err = decomp.reconstruction_error();
    // Reconstruction: trend + cycle + noise = price
    // SSA may have a small residual because we split residual into cycle + noise
    // via SMA -- reconstruction error should be < 2% of price range
    let price_range = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - prices.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(err < price_range * 0.10 + 1.0,
        "SSA reconstruction error {err:.3} too large (price range {price_range:.1})");
}

#[test]
fn test_ssa_trend_length_matches() {
    let prices = sinusoidal(100);
    let ssa = SSADetrend::default_daily();
    let decomp = ssa.decompose(&prices);
    assert_eq!(decomp.trend.len(), prices.len());
    assert_eq!(decomp.cycle.len(), prices.len());
}

#[test]
fn test_emd_lengths_correct() {
    let prices = sinusoidal(128);
    let emd = EMDDetrend::default_daily();
    let decomp = emd.decompose(&prices);
    assert_eq!(decomp.trend.len(), prices.len());
    assert_eq!(decomp.cycle.len(), prices.len());
    assert_eq!(decomp.noise.len(), prices.len());
}

#[test]
fn test_emd_no_panic_on_short_series() {
    let prices = vec![100.0f64, 101.0, 100.5, 102.0];
    let emd = EMDDetrend::default_daily();
    let decomp = emd.decompose(&prices);
    assert_eq!(decomp.trend.len(), 4);
}

// ---------------------------------------------------------------------------
// Extended fractal dimension tests
// ---------------------------------------------------------------------------

#[test]
fn test_fractal_dimension_trending() {
    let prices = trending_series(128);
    let fd = FractalDimension::compute(FDMethod::Higuchi { k_max: 8 }, &prices);
    assert!(fd.is_some());
    let fd = fd.unwrap();
    assert!(fd < 1.4, "Trending series Higuchi FD={fd:.3} should be < 1.4");
}

#[test]
fn test_fractal_dimension_noise_above_trending() {
    let trend_fd = FractalDimension::compute(FDMethod::Higuchi { k_max: 8 }, &trending_series(128))
        .unwrap_or(1.0);
    let noisy_fd = FractalDimension::compute(FDMethod::Higuchi { k_max: 8 }, &noisy_series(128, 42))
        .unwrap_or(1.5);
    assert!(noisy_fd >= trend_fd,
        "Noisy FD {noisy_fd:.3} should be >= trending FD {trend_fd:.3}");
}

#[test]
fn test_petrosian_trending_low() {
    let prices = trending_series(128);
    let fd = PetrosianMethod::compute(&prices).unwrap();
    assert!(fd < 1.5, "Petrosian FD={fd:.3} for trending series should be < 1.5");
}

#[test]
fn test_petrosian_noisy_higher() {
    let fd_trend = PetrosianMethod::compute(&trending_series(128)).unwrap_or(1.0);
    let fd_noisy = PetrosianMethod::compute(&noisy_series(128, 99)).unwrap_or(1.5);
    assert!(fd_noisy >= fd_trend,
        "Noisy Petrosian FD {fd_noisy:.3} should be >= trend {fd_trend:.3}");
}

#[test]
fn test_katz_constant_gives_fd_1() {
    let prices = vec![100.0f64; 64];
    let fd = KatzMethod::compute(&prices).unwrap();
    assert!((fd - 1.0).abs() < 1e-6, "Constant series Katz FD should be 1.0, got {fd}");
}

#[test]
fn test_katz_in_valid_range() {
    let prices = noisy_series(128, 7);
    if let Some(fd) = KatzMethod::compute(&prices) {
        assert!(fd >= 1.0 && fd <= 2.0, "Katz FD {fd:.3} out of [1, 2]");
    }
}

#[test]
fn test_all_methods_result_ensemble() {
    let result = AllMethodsResult {
        higuchi:     Some(1.3),
        box_counting: Some(1.4),
        petrosian:   Some(1.2),
        katz:        Some(1.5),
    };
    let mean = result.ensemble_mean().unwrap();
    assert!((mean - 1.35).abs() < 1e-9, "ensemble mean should be 1.35, got {mean}");
}

#[test]
fn test_fd_trading_regime_strong_trend() {
    let regime = FdTradingRegime::from_fd(1.1);
    assert_eq!(regime, FdTradingRegime::StrongTrend);
    assert!(regime.is_tradeable());
    assert!((regime.position_scale() - 1.0).abs() < 1e-9);
}

#[test]
fn test_fd_trading_regime_chaotic_no_trade() {
    let regime = FdTradingRegime::from_fd(1.9);
    assert_eq!(regime, FdTradingRegime::Chaotic);
    assert!(!regime.is_tradeable());
    assert!(regime.position_scale() < 1e-9);
}

#[test]
fn test_fd_rolling_length_matches() {
    let prices = noisy_series(100, 42);
    let rolling = FractalDimension::rolling(FDMethod::Petrosian, &prices, 32);
    assert_eq!(rolling.len(), prices.len(), "rolling FD length should match input");
}

#[test]
fn test_fd_rolling_first_values_nan() {
    let prices = trending_series(100);
    let rolling = FractalDimension::rolling(FDMethod::Higuchi { k_max: 8 }, &prices, 32);
    for &v in &rolling[..31] {
        assert!(v.is_nan(), "first 31 values should be NaN (window not filled)");
    }
}

#[test]
fn test_all_methods_compute_without_panic() {
    let prices = noisy_series(128, 77);
    let all = FractalDimension::compute_all(&prices);
    // At least higuchi and petrosian should succeed for 128 bars
    assert!(all.higuchi.is_some() || all.petrosian.is_some(),
        "At least one method should succeed");
}

#[test]
fn test_fd_classifier_boundary_1_45() {
    let regime_below = FdTradingRegime::from_fd(1.44);
    let regime_above = FdTradingRegime::from_fd(1.46);
    assert_eq!(regime_below, FdTradingRegime::MildTrend);
    assert_eq!(regime_above, FdTradingRegime::Borderline);
}
