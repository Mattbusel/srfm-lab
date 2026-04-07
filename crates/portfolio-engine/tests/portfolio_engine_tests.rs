// portfolio_engine_tests.rs -- Integration tests for portfolio-engine modules
// Covers rebalancer, risk_budget, drawdown_control, leverage_control,
// and performance_attribution.

use portfolio_engine::rebalancer::{Rebalancer, RebalanceConfig, OrderSide};
use portfolio_engine::risk_budget::{RiskBudgetOptimizer, RegimeState, RiskBudget};
use portfolio_engine::drawdown_control::DrawdownController;
use portfolio_engine::leverage_control::LeverageController;
use portfolio_engine::performance_attribution::AttributionEngine;

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_rebalance_config(
    targets: &[(&str, f64)],
    current: &[(&str, f64)],
    prices: &[(&str, f64)],
    cash: f64,
    max_turnover: f64,
) -> RebalanceConfig {
    RebalanceConfig {
        target_weights: targets.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        current_weights: current.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        current_prices: prices.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        cash,
        max_turnover,
        min_trade_size_usd: 50.0,
        transaction_cost_bps: 5.0,
    }
}

fn identity_cov(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

fn diag_cov(vols: &[f64]) -> Vec<Vec<f64>> {
    let n = vols.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { vols[i] * vols[i] } else { 0.0 })
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Rebalancer tests
// ---------------------------------------------------------------------------

#[test]
fn test_rebalancer_minimum_turnover() {
    // When targets match current, no orders should be generated.
    let cfg = make_rebalance_config(
        &[("AAPL", 0.5), ("MSFT", 0.5)],
        &[("AAPL", 0.5), ("MSFT", 0.5)],
        &[("AAPL", 150.0), ("MSFT", 300.0)],
        500.0,
        0.30,
    );
    let orders = Rebalancer::compute_orders(&cfg);
    assert!(
        orders.is_empty(),
        "No orders expected when weights already at target"
    );
}

#[test]
fn test_rebalancer_generates_orders_on_drift() {
    let cfg = make_rebalance_config(
        &[("AAPL", 0.60), ("MSFT", 0.40)],
        &[("AAPL", 0.40), ("MSFT", 0.60)],
        &[("AAPL", 150.0), ("MSFT", 300.0)],
        500.0,
        1.0,
    );
    let orders = Rebalancer::compute_orders(&cfg);
    assert!(!orders.is_empty(), "Orders expected when weights are drifted");
}

#[test]
fn test_rebalancer_netting() {
    // A portfolio that needs to sell one asset and buy another.
    // The buy should be financed from the sell (netting).
    let cfg = make_rebalance_config(
        &[("AAPL", 0.70), ("MSFT", 0.30)],
        &[("AAPL", 0.30), ("MSFT", 0.70)],
        &[("AAPL", 150.0), ("MSFT", 300.0)],
        1000.0,
        1.0,
    );
    let orders = Rebalancer::compute_orders(&cfg);
    let total_buy = Rebalancer::total_buy_value(&orders);
    let total_sell = Rebalancer::total_sell_value(&orders);
    // Buy value should not exceed sell value + cash.
    assert!(
        total_buy <= total_sell + cfg.cash + 1.0,
        "Buy value {} should not exceed sell + cash {}",
        total_buy,
        total_sell + cfg.cash
    );
}

#[test]
fn test_rebalancer_netting_buy_sell_counts() {
    let cfg = make_rebalance_config(
        &[("AAPL", 0.60), ("GOOG", 0.20), ("MSFT", 0.20)],
        &[("AAPL", 0.20), ("GOOG", 0.60), ("MSFT", 0.20)],
        &[("AAPL", 150.0), ("GOOG", 130.0), ("MSFT", 300.0)],
        1000.0,
        1.0,
    );
    let orders = Rebalancer::compute_orders(&cfg);
    let (buys, sells) = Rebalancer::split_orders(&orders);
    // There should be at least one buy and one sell.
    assert!(!buys.is_empty(), "Expected at least one buy order");
    assert!(!sells.is_empty(), "Expected at least one sell order");
}

#[test]
fn test_rebalancer_min_trade_size_filter() {
    // Tiny drift -- should produce no orders because delta < min_trade_size.
    let cfg = RebalanceConfig {
        target_weights: [("AAPL".to_string(), 0.5001)].into_iter().collect(),
        current_weights: [("AAPL".to_string(), 0.5000)].into_iter().collect(),
        current_prices: [("AAPL".to_string(), 150.0)].into_iter().collect(),
        cash: 100_000.0,
        max_turnover: 1.0,
        min_trade_size_usd: 50_000.0, // very large minimum
        transaction_cost_bps: 5.0,
    };
    let orders = Rebalancer::compute_orders(&cfg);
    // With a very large min trade size and tiny drift, no orders.
    assert!(
        orders.is_empty(),
        "Tiny drift below min_trade_size should produce no orders"
    );
}

#[test]
fn test_rebalancer_turnover_budget_respected() {
    let cfg = make_rebalance_config(
        &[("AAPL", 0.80), ("MSFT", 0.20)],
        &[("AAPL", 0.20), ("MSFT", 0.80)],
        &[("AAPL", 150.0), ("MSFT", 300.0)],
        2000.0,
        0.10, // only 10% turnover allowed
    );
    let total_value = Rebalancer::total_portfolio_value(&cfg);
    let orders = Rebalancer::compute_orders(&cfg);
    let turnover = Rebalancer::compute_turnover(&orders, total_value);
    assert!(
        turnover <= cfg.max_turnover + 1e-6,
        "Turnover {:.4} exceeds max_turnover {:.4}",
        turnover,
        cfg.max_turnover
    );
}

#[test]
fn test_rebalancer_drift_check_triggers() {
    let current: HashMap<String, f64> =
        [("AAPL".to_string(), 0.30), ("MSFT".to_string(), 0.70)]
            .into_iter()
            .collect();
    let target: HashMap<String, f64> =
        [("AAPL".to_string(), 0.50), ("MSFT".to_string(), 0.50)]
            .into_iter()
            .collect();
    assert!(
        Rebalancer::drift_check(&current, &target, 0.05),
        "20% drift should trigger rebalance with 5% threshold"
    );
}

#[test]
fn test_rebalancer_drift_check_no_trigger() {
    let current: HashMap<String, f64> =
        [("AAPL".to_string(), 0.501), ("MSFT".to_string(), 0.499)]
            .into_iter()
            .collect();
    let target: HashMap<String, f64> =
        [("AAPL".to_string(), 0.500), ("MSFT".to_string(), 0.500)]
            .into_iter()
            .collect();
    assert!(
        !Rebalancer::drift_check(&current, &target, 0.05),
        "0.1% drift should not trigger with 5% threshold"
    );
}

#[test]
fn test_rebalancer_simulate_post_weights_non_negative() {
    let cfg = make_rebalance_config(
        &[("AAPL", 0.60), ("MSFT", 0.40)],
        &[("AAPL", 0.40), ("MSFT", 0.60)],
        &[("AAPL", 150.0), ("MSFT", 300.0)],
        500.0,
        1.0,
    );
    let orders = Rebalancer::compute_orders(&cfg);
    let post_weights = Rebalancer::simulate_post_rebalance_weights(&orders, &cfg);
    for (sym, &w) in &post_weights {
        assert!(w >= 0.0, "Negative post-rebalance weight for {}: {}", sym, w);
    }
}

#[test]
fn test_rebalancer_estimate_cost_positive() {
    let cfg = make_rebalance_config(
        &[("AAPL", 0.60), ("MSFT", 0.40)],
        &[("AAPL", 0.40), ("MSFT", 0.60)],
        &[("AAPL", 150.0), ("MSFT", 300.0)],
        500.0,
        1.0,
    );
    let orders = Rebalancer::compute_orders(&cfg);
    if !orders.is_empty() {
        let cost = Rebalancer::estimate_cost(&orders, &cfg);
        assert!(cost > 0.0, "Cost should be positive when orders exist");
    }
}

#[test]
fn test_rebalancer_sells_before_buys_ordering() {
    // Sells should appear first to free up cash before buying.
    let cfg = make_rebalance_config(
        &[("AAPL", 0.60), ("MSFT", 0.40)],
        &[("AAPL", 0.40), ("MSFT", 0.60)],
        &[("AAPL", 150.0), ("MSFT", 300.0)],
        200.0,
        1.0,
    );
    let orders = Rebalancer::compute_orders(&cfg);
    if orders.len() >= 2 {
        // First order should be a sell if there are both buys and sells.
        let has_sell = orders.iter().any(|o| o.side == OrderSide::Sell);
        let has_buy = orders.iter().any(|o| o.side == OrderSide::Buy);
        if has_sell && has_buy {
            assert_eq!(
                orders[0].side,
                OrderSide::Sell,
                "First order should be a sell"
            );
        }
    }
}

#[test]
fn test_rebalancer_max_drift_computation() {
    let current: HashMap<String, f64> =
        [("AAPL".to_string(), 0.30), ("MSFT".to_string(), 0.70)]
            .into_iter()
            .collect();
    let target: HashMap<String, f64> =
        [("AAPL".to_string(), 0.50), ("MSFT".to_string(), 0.50)]
            .into_iter()
            .collect();
    let max_d = Rebalancer::max_drift(&current, &target);
    assert!((max_d - 0.20).abs() < 1e-9, "Max drift should be 0.20, got {}", max_d);
}

// ---------------------------------------------------------------------------
// Risk budget tests
// ---------------------------------------------------------------------------

#[test]
fn test_risk_budget_equal_contributions() {
    let budget = RiskBudgetOptimizer::equal_risk(5);
    assert_eq!(budget.len(), 5);
    let sum: f64 = budget.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "Budget sum should be 1.0");
    for &b in &budget {
        assert!((b - 0.20).abs() < 1e-10, "Each equal budget should be 0.20");
    }
}

#[test]
fn test_risk_budget_zero_assets() {
    let budget = RiskBudgetOptimizer::equal_risk(0);
    assert!(budget.is_empty());
}

#[test]
fn test_signal_weighted_proportional() {
    let signals = vec![1.0, 3.0];
    let budget = RiskBudgetOptimizer::signal_weighted_budget(&signals);
    assert!((budget[0] - 0.25).abs() < 1e-9);
    assert!((budget[1] - 0.75).abs() < 1e-9);
}

#[test]
fn test_signal_weighted_all_zero_falls_back() {
    let signals = vec![0.0, 0.0, 0.0];
    let budget = RiskBudgetOptimizer::signal_weighted_budget(&signals);
    for &b in &budget {
        assert!((b - 1.0 / 3.0).abs() < 1e-9);
    }
}

#[test]
fn test_signal_weighted_negative_clamped() {
    let signals = vec![-1.0, 2.0];
    let budget = RiskBudgetOptimizer::signal_weighted_budget(&signals);
    assert!((budget[0] - 0.0).abs() < 1e-9, "Negative signal should clamp to 0");
    assert!((budget[1] - 1.0).abs() < 1e-9);
}

#[test]
fn test_regime_adjusted_bull_low_vol_unchanged() {
    let base = vec![0.5, 0.3, 0.2];
    let adj = RiskBudgetOptimizer::regime_adjusted_budget(&base, &RegimeState::BullLowVol);
    for (a, b) in adj.iter().zip(base.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn test_regime_adjusted_bear_high_vol_flattens() {
    let base = vec![0.7, 0.2, 0.1];
    let adj = RiskBudgetOptimizer::regime_adjusted_budget(&base, &RegimeState::BearHighVol);
    let equal = 1.0 / 3.0;
    for &a in &adj {
        assert!(
            (a - equal).abs() < 0.25,
            "Bear high vol should flatten toward equal, got {}", a
        );
    }
}

#[test]
fn test_regime_adjusted_sums_to_one() {
    let base = vec![0.4, 0.35, 0.25];
    for regime in &[
        RegimeState::BullLowVol,
        RegimeState::BullHighVol,
        RegimeState::BearLowVol,
        RegimeState::BearHighVol,
        RegimeState::Neutral,
    ] {
        let adj = RiskBudgetOptimizer::regime_adjusted_budget(&base, regime);
        let sum: f64 = adj.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Regime {:?} budget sum {}", regime, sum);
    }
}

#[test]
fn test_realized_contributions_sum_to_one() {
    let weights = vec![0.25, 0.25, 0.25, 0.25];
    let cov = identity_cov(4);
    let rc = RiskBudgetOptimizer::compute_realized_contributions(&weights, &cov);
    let sum: f64 = rc.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "RC sum should be 1.0, got {}", sum);
}

#[test]
fn test_realized_contributions_equal_weights_identity() {
    let n = 4;
    let weights = vec![1.0 / n as f64; n];
    let cov = identity_cov(n);
    let rc = RiskBudgetOptimizer::compute_realized_contributions(&weights, &cov);
    for &r in &rc {
        assert!((r - 0.25).abs() < 1e-8);
    }
}

#[test]
fn test_realized_contributions_diag_cov() {
    // With diagonal covariance and equal weights, higher-vol assets contribute more.
    let weights = vec![0.5, 0.5];
    let cov = diag_cov(&[0.10, 0.20]);
    let rc = RiskBudgetOptimizer::compute_realized_contributions(&weights, &cov);
    assert!(rc[1] > rc[0], "Higher-vol asset should have higher risk contribution");
}

#[test]
fn test_budget_deviation_zero() {
    let v = vec![0.25, 0.25, 0.25, 0.25];
    let dev = RiskBudgetOptimizer::budget_deviation(&v, &v);
    assert!(dev.abs() < 1e-10);
}

#[test]
fn test_budget_deviation_known() {
    let r = vec![0.30, 0.70];
    let t = vec![0.50, 0.50];
    let dev = RiskBudgetOptimizer::budget_deviation(&r, &t);
    // sqrt((0.30-0.50)^2 + (0.70-0.50)^2) = sqrt(0.04 + 0.04) = sqrt(0.08)
    let expected = (0.08_f64).sqrt();
    assert!((dev - expected).abs() < 1e-9);
}

#[test]
fn test_risk_budget_from_vecs() {
    let syms: Vec<String> = vec!["A".to_string(), "B".to_string()];
    let budgets = vec![0.6, 0.4];
    let rb = RiskBudget::from_vecs(&syms, &budgets);
    assert_eq!(rb.allocations["A"], 0.6);
    assert_eq!(rb.allocations["B"], 0.4);
}

#[test]
fn test_optimize_to_budget_converges() {
    let target = vec![0.5, 0.5];
    let cov = identity_cov(2);
    let weights = RiskBudgetOptimizer::optimize_to_budget(&target, &cov, 100, 1e-8);
    // With identity cov and equal target, should converge to equal weights.
    assert!((weights[0] - 0.5).abs() < 0.01);
    assert!((weights[1] - 0.5).abs() < 0.01);
}

// ---------------------------------------------------------------------------
// Drawdown controller tests
// ---------------------------------------------------------------------------

#[test]
fn test_drawdown_controller_scalar_full_size() {
    let ctrl = DrawdownController::new(100_000.0);
    assert_eq!(ctrl.get_position_scalar(), 1.0);
}

#[test]
fn test_drawdown_controller_scalar_at_3pct_boundary() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(97_000.0); // 3% drawdown -- at boundary, should still be 1.0
    assert_eq!(ctrl.get_position_scalar(), 1.0);
}

#[test]
fn test_drawdown_controller_scalar_reduced_5pct() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(95_000.0); // 5% drawdown -> scalar 0.75
    assert_eq!(ctrl.get_position_scalar(), 0.75);
}

#[test]
fn test_drawdown_controller_scalar_half_10pct() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(90_000.0); // 10% drawdown -> scalar 0.50
    assert_eq!(ctrl.get_position_scalar(), 0.50);
}

#[test]
fn test_drawdown_controller_scalar_preservation_15pct() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(85_000.0); // 15% drawdown -> scalar 0.25
    assert_eq!(ctrl.get_position_scalar(), 0.25);
}

#[test]
fn test_drawdown_controller_recovery_mode() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(88_000.0); // 12% drawdown -> recovery mode
    assert!(ctrl.is_in_recovery_mode());
}

#[test]
fn test_drawdown_controller_not_recovery_shallow() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(96_000.0); // 4% drawdown -> not recovery mode
    assert!(!ctrl.is_in_recovery_mode());
}

#[test]
fn test_drawdown_controller_days_to_recover() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(90_000.0); // 10% drawdown
    let days = ctrl.days_to_recover(0.001); // 0.1% daily return
    assert!(days.is_some());
    let d = days.unwrap();
    assert!(d > 0.0, "Days to recover should be positive");
}

#[test]
fn test_drawdown_controller_days_to_recover_none_negative_rate() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(90_000.0);
    let days = ctrl.days_to_recover(-0.001);
    assert!(days.is_none(), "Negative rate should return None");
}

#[test]
fn test_drawdown_controller_at_peak_zero_days() {
    let ctrl = DrawdownController::new(100_000.0);
    let days = ctrl.days_to_recover(0.001);
    assert_eq!(days, Some(0.0), "At peak, should be 0 days to recover");
}

#[test]
fn test_drawdown_controller_tracks_max_drawdown() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(95_000.0); // 5%
    ctrl.update(98_000.0); // partial recovery
    ctrl.update(88_000.0); // 12% -- new max
    let state = ctrl.current_state();
    assert!(
        (state.max_drawdown_pct - 0.12).abs() < 1e-6,
        "Max drawdown should be 12%, got {}", state.max_drawdown_pct
    );
}

#[test]
fn test_drawdown_controller_peak_updates_on_new_high() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(110_000.0);
    let state = ctrl.current_state();
    assert_eq!(state.peak_equity, 110_000.0);
    assert!(state.is_at_peak());
}

#[test]
fn test_drawdown_controller_days_in_drawdown_counter() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(99_000.0); // below peak
    ctrl.update(98_000.0); // still below
    ctrl.update(97_000.0); // still below
    let state = ctrl.current_state();
    assert_eq!(state.days_in_drawdown, 3);
}

#[test]
fn test_drawdown_controller_reset() {
    let mut ctrl = DrawdownController::new(100_000.0);
    ctrl.update(85_000.0);
    ctrl.reset(100_000.0);
    assert_eq!(ctrl.get_position_scalar(), 1.0);
}

// ---------------------------------------------------------------------------
// Leverage controller tests
// ---------------------------------------------------------------------------

#[test]
fn test_leverage_vol_targeting_exact() {
    let lev = LeverageController::compute_target_leverage(0.10, 0.10, 2.0);
    assert!((lev - 1.0).abs() < 1e-9, "10% vol / 10% target = 1x leverage");
}

#[test]
fn test_leverage_vol_targeting_scales_up() {
    let lev = LeverageController::compute_target_leverage(0.05, 0.10, 2.0);
    assert!((lev - 2.0).abs() < 1e-9, "5% vol -> 2x leverage (capped at max)");
}

#[test]
fn test_leverage_vol_targeting_scales_down() {
    let lev = LeverageController::compute_target_leverage(0.20, 0.10, 2.0);
    assert!((lev - 0.5).abs() < 1e-9, "20% vol -> 0.5x leverage");
}

#[test]
fn test_leverage_vol_targeting_zero_vol() {
    let lev = LeverageController::compute_target_leverage(0.0, 0.10, 2.0);
    assert_eq!(lev, 2.0, "Zero vol should return max_leverage");
}

#[test]
fn test_leverage_bh_active_boosts() {
    let adj = LeverageController::regime_adjusted_leverage(1.0, true, 0.55, 3.0);
    assert!((adj - 1.2).abs() < 1e-9);
}

#[test]
fn test_leverage_mean_revert_reduces() {
    let adj = LeverageController::regime_adjusted_leverage(1.0, false, 0.35, 3.0);
    assert!((adj - 0.8).abs() < 1e-9);
}

#[test]
fn test_leverage_both_adjustments() {
    // BH active + mean-reverting: 1.0 * 1.2 * 0.8 = 0.96
    let adj = LeverageController::regime_adjusted_leverage(1.0, true, 0.35, 3.0);
    assert!((adj - 0.96).abs() < 1e-9);
}

#[test]
fn test_leverage_capped_at_max() {
    let adj = LeverageController::regime_adjusted_leverage(2.8, true, 0.60, 3.0);
    assert!(adj <= 3.0, "Leverage should not exceed max");
}

#[test]
fn test_margin_safety_check_safe() {
    assert!(LeverageController::margin_safety_check(150_000.0, 100_000.0, 2.0));
}

#[test]
fn test_margin_safety_check_breach() {
    assert!(!LeverageController::margin_safety_check(210_000.0, 100_000.0, 2.0));
}

#[test]
fn test_margin_safety_check_zero_equity() {
    assert!(!LeverageController::margin_safety_check(1_000.0, 0.0, 2.0));
}

#[test]
fn test_gross_exposure_fully_invested() {
    let weights = vec![0.4, 0.35, 0.25];
    let prices = vec![100.0, 200.0, 300.0];
    let exp = LeverageController::compute_gross_exposure(&weights, &prices, 100_000.0);
    assert!((exp - 100_000.0).abs() < 1e-6, "Fully invested should expose full equity");
}

#[test]
fn test_gross_exposure_levered() {
    let weights = vec![0.6, 0.6];
    let prices = vec![100.0, 200.0];
    let exp = LeverageController::compute_gross_exposure(&weights, &prices, 100_000.0);
    assert!((exp - 120_000.0).abs() < 1e-6, "1.2x levered should expose 120k");
}

#[test]
fn test_enforce_leverage_limit() {
    let weights = vec![0.8, 0.8]; // sum abs = 1.6
    let clamped = LeverageController::enforce_leverage_limit(&weights, 1.0);
    let sum_abs: f64 = clamped.iter().map(|w| w.abs()).sum();
    assert!((sum_abs - 1.0).abs() < 1e-9);
}

#[test]
fn test_scale_weights_to_leverage() {
    let weights = vec![0.4, 0.3, 0.3];
    let scaled = LeverageController::scale_weights_to_leverage(&weights, 1.5);
    let sum_abs: f64 = scaled.iter().map(|w| w.abs()).sum();
    assert!((sum_abs - 1.5).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Performance attribution tests
// ---------------------------------------------------------------------------

#[test]
fn test_bhb_allocation_effect_sign() {
    // Overweight asset with above-benchmark return -> positive allocation.
    let pw = vec![0.6, 0.4];
    let bw = vec![0.5, 0.5];
    let pr = vec![0.10, 0.05];
    let br = vec![0.08, 0.04];
    // r_b = 0.5*0.08 + 0.5*0.04 = 0.06
    // allocation[0] = (0.6-0.5) * (0.08-0.06) = 0.1 * 0.02 = 0.002 > 0
    let bhb = AttributionEngine::compute_attribution(&pw, &bw, &pr, &br);
    assert!(bhb.allocation[0] > 0.0, "Overweight in above-avg asset -> positive allocation");
}

#[test]
fn test_bhb_allocation_effect_negative_when_underweight_outperformer() {
    // Underweight an asset that beats benchmark -> negative allocation.
    let pw = vec![0.3, 0.7];
    let bw = vec![0.5, 0.5];
    let pr = vec![0.10, 0.05];
    let br = vec![0.12, 0.04];
    // r_b = 0.5*0.12 + 0.5*0.04 = 0.08
    // allocation[0] = (0.3-0.5)*(0.12-0.08) = -0.2 * 0.04 = -0.008 < 0
    let bhb = AttributionEngine::compute_attribution(&pw, &bw, &pr, &br);
    assert!(bhb.allocation[0] < 0.0, "Underweight in outperformer -> negative allocation");
}

#[test]
fn test_bhb_attribution_sums_to_total() {
    let pw = vec![0.6, 0.4];
    let bw = vec![0.5, 0.5];
    let pr = vec![0.10, 0.05];
    let br = vec![0.08, 0.04];
    let bhb = AttributionEngine::compute_attribution(&pw, &bw, &pr, &br);
    assert!(bhb.check_consistency(), "BHB components must sum to active return");
}

#[test]
fn test_bhb_zero_active_return_when_identical() {
    let w = vec![0.5, 0.5];
    let r = vec![0.08, 0.04];
    let bhb = AttributionEngine::compute_attribution(&w, &w, &r, &r);
    assert!(bhb.total_active_return.abs() < 1e-10, "Zero active return when portfolio = benchmark");
}

#[test]
fn test_bhb_selection_effect() {
    // Same weights but portfolio picks better stocks.
    let pw = vec![0.5, 0.5];
    let bw = vec![0.5, 0.5];
    let pr = vec![0.12, 0.06]; // outperform benchmark by 4% and 2%
    let br = vec![0.08, 0.04];
    let bhb = AttributionEngine::compute_attribution(&pw, &bw, &pr, &br);
    // Allocation = 0 (same weights)
    assert!(bhb.total_allocation().abs() < 1e-10);
    // Interaction = 0 (same weights)
    assert!(bhb.total_interaction().abs() < 1e-10);
    // Selection = total active return.
    assert!(bhb.total_selection() > 0.0);
}

#[test]
fn test_time_series_attribution_count() {
    let pw: Vec<Vec<f64>> = (0..10).map(|_| vec![0.6, 0.4]).collect();
    let bw: Vec<Vec<f64>> = (0..10).map(|_| vec![0.5, 0.5]).collect();
    let pr: Vec<Vec<f64>> = (0..10).map(|_| vec![0.10, 0.05]).collect();
    let br: Vec<Vec<f64>> = (0..10).map(|_| vec![0.08, 0.04]).collect();
    let records = AttributionEngine::time_series_attribution(&pw, &bw, &pr, &br);
    assert_eq!(records.len(), 10);
}

#[test]
fn test_cumulative_attribution_grows() {
    let pw: Vec<Vec<f64>> = (0..5).map(|_| vec![0.6, 0.4]).collect();
    let bw: Vec<Vec<f64>> = (0..5).map(|_| vec![0.5, 0.5]).collect();
    let pr: Vec<Vec<f64>> = (0..5).map(|_| vec![0.10, 0.05]).collect();
    let br: Vec<Vec<f64>> = (0..5).map(|_| vec![0.08, 0.04]).collect();
    let records = AttributionEngine::time_series_attribution(&pw, &bw, &pr, &br);
    let (cum_p, cum_b, _cum_active) = AttributionEngine::cumulative_attribution(&records);
    assert!(cum_p > cum_b, "Portfolio should outperform benchmark in this setup");
}

#[test]
fn test_factor_attribution_alpha() {
    // When returns are perfectly explained by factor, alpha should be near zero.
    let factor: Vec<f64> = vec![0.01, 0.02, -0.01, 0.03, 0.00, 0.02, -0.02];
    let port: Vec<f64> = factor.iter().map(|&f| f * 1.5).collect(); // pure factor exposure
    let x: Vec<Vec<f64>> = factor.iter().map(|&f| vec![f]).collect();
    let names = vec!["FACTOR1".to_string()];
    let fa = AttributionEngine::factor_attribution_ols(&port, &x, &names);
    assert!(fa.r_squared > 0.99, "R-squared should be near 1.0, got {}", fa.r_squared);
    assert!(fa.alpha.abs() < 0.001, "Alpha should be near zero, got {}", fa.alpha);
}

#[test]
fn test_asset_contributions_sum() {
    let weights = vec![0.4, 0.35, 0.25];
    let returns = vec![0.10, 0.05, -0.02];
    let contribs = AttributionEngine::asset_contributions(&weights, &returns);
    let total: f64 = contribs.iter().sum();
    let expected: f64 = weights.iter().zip(returns.iter()).map(|(&w, &r)| w * r).sum();
    assert!((total - expected).abs() < 1e-10);
}

#[test]
fn test_information_ratio_positive_skill() {
    // Consistently positive active returns -> positive IR.
    let ar: Vec<f64> = vec![0.001, 0.002, 0.001, 0.003, 0.001, 0.002];
    let ir = AttributionEngine::information_ratio(&ar);
    assert!(ir > 0.0, "Positive active returns should give positive IR");
}
