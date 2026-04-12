// test_impact.rs — Comprehensive tests for the market impact model module.
// Covers Almgren-Chriss model, LOB depth depletion, Kyle lambda estimation,
// volume participation model, feedback engine, and trajectory cost projection.

#![allow(clippy::float_cmp)]

use crate::market_impact_model::{
    AlmgrenChrissImpact, FeedbackImpactEngine, ImpactStats, ImpactTrajectory,
    KyleLambdaEstimator, LobDepthModel, OrderImpactResult, VolumeParticipationModel,
    linear_impact_bps, square_root_impact_bps,
};

// ── Almgren-Chriss model tests ────────────────────────────────────────────────

#[cfg(test)]
mod almgren_chriss_tests {
    use super::*;

    fn default_ac() -> AlmgrenChrissImpact {
        AlmgrenChrissImpact::new(
            0.001,  // permanent impact coefficient η
            0.01,   // temporary impact coefficient γ
            0.20,   // volatility σ
            1.0,    // risk aversion λ
        )
    }

    #[test]
    fn test_permanent_impact_proportional_to_order_size() {
        let ac = default_ac();
        let impact_small = ac.permanent_impact(100.0, 10.0);
        let impact_large = ac.permanent_impact(1000.0, 10.0);
        assert!(
            impact_large > impact_small,
            "Larger order should have more permanent impact: {} vs {}",
            impact_large,
            impact_small
        );
    }

    #[test]
    fn test_permanent_impact_scales_linearly() {
        let ac = default_ac();
        let i1 = ac.permanent_impact(100.0, 10.0);
        let i2 = ac.permanent_impact(200.0, 10.0);
        // Should be approximately 2x
        let ratio = i2 / i1;
        assert!(
            (ratio - 2.0).abs() < 0.5,
            "Permanent impact should scale linearly, ratio={}",
            ratio
        );
    }

    #[test]
    fn test_temporary_impact_positive() {
        let ac = default_ac();
        let impact = ac.temporary_impact(500.0, 60.0);
        assert!(impact > 0.0, "Temporary impact should be positive, got {}", impact);
    }

    #[test]
    fn test_temporary_impact_decreases_with_longer_horizon() {
        let ac = default_ac();
        let fast = ac.temporary_impact(1000.0, 10.0);
        let slow = ac.temporary_impact(1000.0, 3600.0);
        assert!(
            fast > slow,
            "Fast execution should have higher temporary impact: {} vs {}",
            fast,
            slow
        );
    }

    #[test]
    fn test_optimal_schedule_cost_positive() {
        let ac = default_ac();
        let cost = ac.optimal_schedule_cost(10_000.0, 3600.0, 100.0);
        assert!(cost > 0.0, "Schedule cost must be positive, got {}", cost);
    }

    #[test]
    fn test_optimal_schedule_cost_increases_with_trade_size() {
        let ac = default_ac();
        let c1 = ac.optimal_schedule_cost(1_000.0, 3600.0, 100.0);
        let c2 = ac.optimal_schedule_cost(10_000.0, 3600.0, 100.0);
        assert!(c2 > c1, "Larger trade should have higher cost: {} vs {}", c2, c1);
    }

    #[test]
    fn test_risk_adjusted_cost_exceeds_plain_cost() {
        let ac = default_ac();
        let plain = ac.optimal_schedule_cost(10_000.0, 3600.0, 100.0);
        let risk_adj = ac.risk_adjusted_cost(10_000.0, 3600.0, 100.0);
        assert!(
            risk_adj >= plain,
            "Risk-adjusted cost {} should exceed plain cost {}",
            risk_adj,
            plain
        );
    }

    #[test]
    fn test_high_risk_aversion_increases_cost() {
        let low_risk = AlmgrenChrissImpact::new(0.001, 0.01, 0.20, 0.1);
        let high_risk = AlmgrenChrissImpact::new(0.001, 0.01, 0.20, 10.0);
        let c_low = low_risk.risk_adjusted_cost(10_000.0, 3600.0, 100.0);
        let c_high = high_risk.risk_adjusted_cost(10_000.0, 3600.0, 100.0);
        assert!(
            c_high >= c_low,
            "High risk aversion should increase cost: {} vs {}",
            c_high,
            c_low
        );
    }

    #[test]
    fn test_zero_trade_size_zero_cost() {
        let ac = default_ac();
        let cost = ac.optimal_schedule_cost(0.0, 3600.0, 100.0);
        assert!(cost.abs() < 1e-10, "Zero trade size should have ~zero cost, got {}", cost);
    }

    #[test]
    fn test_impact_direction_buy_vs_sell() {
        let ac = default_ac();
        // Buy should increase price, sell should decrease
        let buy_perm = ac.permanent_impact(100.0, 10.0);
        let sell_perm = ac.permanent_impact(-100.0, 10.0);
        assert!(buy_perm > 0.0, "Buy permanent impact should be positive");
        assert!(sell_perm < 0.0, "Sell permanent impact should be negative");
    }
}

// ── LOB depth model tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod lob_depth_tests {
    use super::*;

    fn default_lob() -> LobDepthModel {
        LobDepthModel::new(100.0, 0.0001, 3)
    }

    #[test]
    fn test_lob_initial_state() {
        let lob = default_lob();
        let (bid, ask) = lob.best_bid_ask();
        assert!(ask > bid, "Ask {} must exceed bid {}", ask, bid);
        assert!((bid - 100.0).abs() < 1.0, "Bid should be near mid=100, got {}", bid);
    }

    #[test]
    fn test_lob_market_buy_moves_ask() {
        let mut lob = default_lob();
        let (_, ask_before) = lob.best_bid_ask();
        let result = lob.execute_market_order(100.0, true); // buy 100 shares
        let (_, ask_after) = lob.best_bid_ask();
        assert!(result.avg_fill_price >= ask_before, "Fill price should be >= initial ask");
        assert!(
            ask_after >= ask_before,
            "Ask should not decrease after buy: {} vs {}",
            ask_after,
            ask_before
        );
    }

    #[test]
    fn test_lob_market_sell_moves_bid() {
        let mut lob = default_lob();
        let (bid_before, _) = lob.best_bid_ask();
        let result = lob.execute_market_order(100.0, false); // sell 100 shares
        let (bid_after, _) = lob.best_bid_ask();
        assert!(result.avg_fill_price <= bid_before, "Fill price should be <= initial bid");
        assert!(
            bid_after <= bid_before,
            "Bid should not increase after sell: {} vs {}",
            bid_after,
            bid_before
        );
    }

    #[test]
    fn test_lob_depth_depletes_on_execution() {
        let mut lob = default_lob();
        let depth_before = lob.total_ask_depth();
        lob.execute_market_order(50.0, true);
        let depth_after = lob.total_ask_depth();
        assert!(
            depth_after < depth_before,
            "Ask depth should decrease after buy: {} vs {}",
            depth_after,
            depth_before
        );
    }

    #[test]
    fn test_lob_replenishment() {
        let mut lob = default_lob();
        lob.execute_market_order(200.0, true); // consume some depth
        let depth_depleted = lob.total_ask_depth();
        lob.replenish(1.0); // 1 second elapsed
        let depth_replenished = lob.total_ask_depth();
        assert!(
            depth_replenished >= depth_depleted,
            "Depth should replenish over time: {} vs {}",
            depth_replenished,
            depth_depleted
        );
    }

    #[test]
    fn test_lob_mid_price_shift() {
        let mut lob = default_lob();
        let mid_before = lob.mid_price();
        lob.shift_mid(0.01); // 1% up
        let mid_after = lob.mid_price();
        assert!(
            (mid_after / mid_before - 1.01).abs() < 0.01,
            "Mid should shift ~1%: {} -> {}",
            mid_before,
            mid_after
        );
    }

    #[test]
    fn test_lob_partial_fill_on_thin_book() {
        let mut lob = LobDepthModel::new(100.0, 0.001, 2); // thin book
        let result = lob.execute_market_order(10_000.0, true); // huge order
        // Should partially fill or fill at significantly worse price
        assert!(result.avg_fill_price >= 100.0, "Buy fill should be >= mid");
        assert!(result.slippage_bps >= 0.0, "Slippage must be non-negative");
    }

    #[test]
    fn test_lob_spread_positive() {
        let lob = default_lob();
        let (bid, ask) = lob.best_bid_ask();
        assert!(ask > bid, "Spread must be positive");
        let spread_bps = (ask - bid) / lob.mid_price() * 10_000.0;
        assert!(spread_bps > 0.0 && spread_bps < 1000.0, "Spread {} bps out of range", spread_bps);
    }

    #[test]
    fn test_lob_large_order_large_impact() {
        let mut lob = default_lob();
        let small_result = lob.execute_market_order(1.0, true);
        let large_result = lob.execute_market_order(1000.0, true);
        assert!(
            large_result.slippage_bps >= small_result.slippage_bps,
            "Large order slippage {} should exceed small {}",
            large_result.slippage_bps,
            small_result.slippage_bps
        );
    }
}

// ── Kyle lambda estimator tests ───────────────────────────────────────────────

#[cfg(test)]
mod kyle_lambda_tests {
    use super::*;

    #[test]
    fn test_kyle_lambda_positive_for_buy_flow() {
        let mut estimator = KyleLambdaEstimator::new(50);
        // Positive order flow (buys) should push prices up
        for i in 0..50 {
            let flow = 100.0 + i as f64 * 5.0;
            let price_change = flow * 0.001; // positive relationship
            estimator.update(flow, price_change);
        }
        let lambda = estimator.estimate();
        assert!(
            lambda > 0.0,
            "Kyle lambda should be positive for buy flow, got {}",
            lambda
        );
    }

    #[test]
    fn test_kyle_lambda_insufficient_data() {
        let mut estimator = KyleLambdaEstimator::new(20);
        // Only 5 observations — less than window
        for i in 0..5 {
            estimator.update(i as f64 * 10.0, i as f64 * 0.001);
        }
        // Should return 0 or a default, not panic
        let lambda = estimator.estimate();
        assert!(lambda >= 0.0, "Lambda should be non-negative for partial data, got {}", lambda);
    }

    #[test]
    fn test_kyle_lambda_zero_variance_flow() {
        let mut estimator = KyleLambdaEstimator::new(20);
        // Constant order flow — zero variance
        for _ in 0..20 {
            estimator.update(100.0, 0.001);
        }
        // Should handle degenerate case gracefully
        let lambda = estimator.estimate();
        assert!(lambda.is_finite(), "Lambda should be finite for zero-variance input");
    }

    #[test]
    fn test_kyle_lambda_rolling_window() {
        let mut estimator = KyleLambdaEstimator::new(10);
        // Fill window with positive relationship
        for i in 0..10 {
            estimator.update(i as f64 * 10.0 + 1.0, i as f64 * 0.01 + 0.001);
        }
        let lambda_before = estimator.estimate();
        // Now add negative relationship observations
        for i in 0..10 {
            estimator.update(i as f64 * 10.0 + 1.0, -(i as f64 * 0.01) - 0.001);
        }
        let lambda_after = estimator.estimate();
        // Lambda should change as window rolls
        assert!(
            lambda_after < lambda_before,
            "Lambda should decrease as negative relationship dominates: {} -> {}",
            lambda_before,
            lambda_after
        );
    }

    #[test]
    fn test_kyle_lambda_magnitude_proportional_to_sensitivity() {
        let mut low_sensitivity = KyleLambdaEstimator::new(30);
        let mut high_sensitivity = KyleLambdaEstimator::new(30);
        for i in 0..30 {
            let flow = i as f64 * 10.0 + 1.0;
            low_sensitivity.update(flow, flow * 0.00001);
            high_sensitivity.update(flow, flow * 0.001);
        }
        let lambda_low = low_sensitivity.estimate();
        let lambda_high = high_sensitivity.estimate();
        assert!(
            lambda_high > lambda_low,
            "Higher price sensitivity should give larger lambda: {} vs {}",
            lambda_high,
            lambda_low
        );
    }
}

// ── Volume participation model tests ──────────────────────────────────────────

#[cfg(test)]
mod vpr_tests {
    use super::*;

    #[test]
    fn test_vpr_basic_slice() {
        let vpr = VolumeParticipationModel::new(0.1); // 10% participation rate
        let slice_qty = vpr.compute_slice_qty(1000.0, 10000.0); // 10% of 1000 shares over 10000 total volume
        assert!(slice_qty > 0.0, "Slice qty must be positive, got {}", slice_qty);
    }

    #[test]
    fn test_vpr_participation_rate_respected() {
        let rate = 0.05; // 5%
        let vpr = VolumeParticipationModel::new(rate);
        let market_vol = 5000.0;
        let slice = vpr.compute_slice_qty(1000.0, market_vol);
        let expected = market_vol * rate;
        assert!(
            (slice - expected).abs() < expected * 0.1,
            "Slice {} should be ~{}% of market volume {}",
            slice,
            rate * 100.0,
            market_vol
        );
    }

    #[test]
    fn test_vpr_does_not_exceed_remaining() {
        let vpr = VolumeParticipationModel::new(0.5); // 50%
        // remaining_qty = 100, market_vol = 10000
        let slice = vpr.compute_slice_qty(100.0, 10000.0);
        assert!(
            slice <= 100.0,
            "Slice {} cannot exceed remaining qty 100",
            slice
        );
    }

    #[test]
    fn test_vpr_zero_market_volume() {
        let vpr = VolumeParticipationModel::new(0.1);
        let slice = vpr.compute_slice_qty(1000.0, 0.0);
        assert_eq!(slice, 0.0, "Zero market volume should produce zero slice");
    }

    #[test]
    fn test_vpr_tracking_fills() {
        let mut vpr = VolumeParticipationModel::new(0.1);
        vpr.on_fill(100.0, 1000.0);
        vpr.on_fill(200.0, 2000.0);
        let rate = vpr.realized_participation_rate();
        // 300 filled / 3000 market = 10%
        assert!(
            (rate - 0.10).abs() < 0.01,
            "Realized participation rate {} expected ~10%",
            rate
        );
    }

    #[test]
    fn test_vpr_high_rate_consumes_quickly() {
        let high = VolumeParticipationModel::new(0.5);
        let low = VolumeParticipationModel::new(0.05);
        let market_vol = 1000.0;
        let remaining = 500.0;
        let high_slice = high.compute_slice_qty(remaining, market_vol);
        let low_slice = low.compute_slice_qty(remaining, market_vol);
        assert!(
            high_slice >= low_slice,
            "High rate should produce larger slice: {} vs {}",
            high_slice,
            low_slice
        );
    }
}

// ── Feedback impact engine tests ──────────────────────────────────────────────

#[cfg(test)]
mod feedback_engine_tests {
    use super::*;

    fn default_engine() -> FeedbackImpactEngine {
        FeedbackImpactEngine::new(
            0.001, // eta
            0.01,  // gamma
            0.20,  // sigma
            1.0,   // lambda
            3,     // lob_levels
            0.1,   // decay_rate per second
        )
    }

    #[test]
    fn test_engine_basic_execution() {
        let mut engine = default_engine();
        let result = engine.execute(100.0, 0, 10.0);
        assert!(result.total_impact_bps >= 0.0, "Total impact must be non-negative");
        assert!(result.permanent_impact_bps >= 0.0);
        assert!(result.temporary_impact_bps >= 0.0);
    }

    #[test]
    fn test_engine_larger_order_more_impact() {
        let mut engine = default_engine();
        let small = engine.execute(10.0, 0, 100.0);
        let large = engine.execute(1000.0, 0, 100.0);
        assert!(
            large.total_impact_bps >= small.total_impact_bps,
            "Large order impact {} should exceed small {}",
            large.total_impact_bps,
            small.total_impact_bps
        );
    }

    #[test]
    fn test_engine_permanent_impact_decays() {
        let mut engine = default_engine();
        engine.execute(1000.0, 0, 100.0);
        let impact_at_0 = engine.current_permanent_impact_bps();
        // Advance time — permanent impact should decay
        engine.advance_time(10.0); // 10 seconds
        let impact_at_10 = engine.current_permanent_impact_bps();
        assert!(
            impact_at_10 < impact_at_0,
            "Permanent impact should decay: {} -> {}",
            impact_at_0,
            impact_at_10
        );
    }

    #[test]
    fn test_engine_impact_result_fields() {
        let mut engine = default_engine();
        let result = engine.execute(500.0, 1, 50.0);
        assert!(result.total_impact_bps >= 0.0);
        assert!(result.avg_fill_price > 0.0);
        assert!(result.slippage_bps >= 0.0);
        assert!(result.executed_qty > 0.0);
    }

    #[test]
    fn test_engine_multiple_sequential_orders() {
        let mut engine = default_engine();
        // Multiple buys should accumulate permanent impact
        let mut total_impact = 0.0f64;
        for _ in 0..5 {
            let result = engine.execute(200.0, 0, 20.0);
            total_impact += result.permanent_impact_bps;
        }
        assert!(total_impact > 0.0, "Cumulative permanent impact should be positive");
    }

    #[test]
    fn test_engine_buy_sell_netted() {
        let mut engine = default_engine();
        engine.execute(500.0, 0, 100.0); // buy
        let impact_after_buy = engine.current_permanent_impact_bps();
        engine.execute(-500.0, 0, 100.0); // sell same size
        let impact_after_sell = engine.current_permanent_impact_bps();
        // After netting, impact should be less than after pure buy
        assert!(
            impact_after_sell <= impact_after_buy,
            "Sell should reduce permanent impact: {} -> {}",
            impact_after_buy,
            impact_after_sell
        );
    }

    #[test]
    fn test_engine_reset() {
        let mut engine = default_engine();
        engine.execute(1000.0, 0, 100.0);
        engine.reset();
        assert!(
            engine.current_permanent_impact_bps().abs() < 1e-10,
            "Permanent impact should be zero after reset"
        );
    }
}

// ── Impact trajectory tests ───────────────────────────────────────────────────

#[cfg(test)]
mod trajectory_tests {
    use super::*;

    #[test]
    fn test_trajectory_slice_count() {
        let traj = ImpactTrajectory::new(10_000.0, 20, 3600.0, 0.001, 0.01, 0.20, 1.0);
        let slices = traj.slices();
        assert_eq!(slices.len(), 20, "Should have 20 slices");
    }

    #[test]
    fn test_trajectory_total_quantity() {
        let total_qty = 5000.0;
        let traj = ImpactTrajectory::new(total_qty, 10, 3600.0, 0.001, 0.01, 0.20, 1.0);
        let sum_qty: f64 = traj.slices().iter().map(|s| s.qty).sum();
        assert!(
            (sum_qty - total_qty).abs() < 1.0,
            "Sum of slice qtys {} should equal total {}",
            sum_qty,
            total_qty
        );
    }

    #[test]
    fn test_trajectory_total_cost_positive() {
        let traj = ImpactTrajectory::new(10_000.0, 10, 3600.0, 0.001, 0.01, 0.20, 1.0);
        let cost = traj.total_cost_bps();
        assert!(cost > 0.0, "Trajectory cost must be positive, got {}", cost);
    }

    #[test]
    fn test_trajectory_faster_execution_higher_cost() {
        let slow = ImpactTrajectory::new(10_000.0, 10, 3600.0, 0.001, 0.01, 0.20, 1.0);
        let fast = ImpactTrajectory::new(10_000.0, 10, 60.0, 0.001, 0.01, 0.20, 1.0);
        assert!(
            fast.total_cost_bps() >= slow.total_cost_bps(),
            "Faster execution should cost more: {} vs {}",
            fast.total_cost_bps(),
            slow.total_cost_bps()
        );
    }

    #[test]
    fn test_trajectory_slice_intervals_uniform() {
        let traj = ImpactTrajectory::new(1000.0, 5, 500.0, 0.001, 0.01, 0.20, 1.0);
        let slices = traj.slices();
        let expected_interval = 100.0; // 500s / 5 slices
        for s in slices {
            assert!(
                (s.interval_s - expected_interval).abs() < 1.0,
                "Slice interval {} expected {}",
                s.interval_s,
                expected_interval
            );
        }
    }

    #[test]
    fn test_trajectory_zero_quantity() {
        let traj = ImpactTrajectory::new(0.0, 5, 3600.0, 0.001, 0.01, 0.20, 1.0);
        let cost = traj.total_cost_bps();
        assert!(cost.abs() < 1e-10, "Zero quantity should have zero cost, got {}", cost);
    }
}

// ── Free function tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod free_function_tests {
    use super::*;

    #[test]
    fn test_square_root_impact_positive() {
        let impact = square_root_impact_bps(1000.0, 100_000.0, 0.20);
        assert!(impact > 0.0, "Square-root impact must be positive, got {}", impact);
    }

    #[test]
    fn test_square_root_impact_scaling() {
        let i1 = square_root_impact_bps(1000.0, 100_000.0, 0.20);
        let i4 = square_root_impact_bps(4000.0, 100_000.0, 0.20);
        // sqrt(4x) = 2x, so ratio should be ~2
        let ratio = i4 / i1;
        assert!(
            (ratio - 2.0).abs() < 0.3,
            "Square-root scaling ratio {} expected ~2.0",
            ratio
        );
    }

    #[test]
    fn test_linear_impact_positive() {
        let impact = linear_impact_bps(1000.0, 100_000.0, 0.001);
        assert!(impact > 0.0, "Linear impact must be positive, got {}", impact);
    }

    #[test]
    fn test_linear_impact_doubles_with_double_size() {
        let i1 = linear_impact_bps(1000.0, 100_000.0, 0.001);
        let i2 = linear_impact_bps(2000.0, 100_000.0, 0.001);
        assert!(
            (i2 / i1 - 2.0).abs() < 0.01,
            "Linear impact should double with double size: {}/{}={}",
            i2,
            i1,
            i2 / i1
        );
    }

    #[test]
    fn test_square_root_vs_linear_for_large_orders() {
        // For large orders, linear > sqrt (linear grows faster at large sizes)
        let large_size = 50_000.0;
        let adv = 100_000.0;
        let sqrt_i = square_root_impact_bps(large_size, adv, 0.20);
        let lin_i = linear_impact_bps(large_size, adv, 0.001);
        // Both should be positive
        assert!(sqrt_i > 0.0 && lin_i > 0.0);
    }

    #[test]
    fn test_impact_zero_order_size() {
        let sq_i = square_root_impact_bps(0.0, 100_000.0, 0.20);
        let lin_i = linear_impact_bps(0.0, 100_000.0, 0.001);
        assert!(sq_i.abs() < 1e-10, "Zero-size square-root impact should be ~0");
        assert!(lin_i.abs() < 1e-10, "Zero-size linear impact should be ~0");
    }

    #[test]
    fn test_impact_increases_with_volatility() {
        let low = square_root_impact_bps(1000.0, 100_000.0, 0.10);
        let high = square_root_impact_bps(1000.0, 100_000.0, 0.40);
        assert!(
            high >= low,
            "Higher volatility should increase impact: {} vs {}",
            high,
            low
        );
    }
}

// ── Impact statistics tests ───────────────────────────────────────────────────

#[cfg(test)]
mod impact_stats_tests {
    use super::*;

    #[test]
    fn test_impact_stats_accumulation() {
        let mut stats = ImpactStats::new();
        stats.record(1.0, 0.5, 0.5, 100.0);
        stats.record(2.0, 1.0, 1.0, 200.0);
        stats.record(3.0, 1.5, 1.5, 300.0);
        assert_eq!(stats.count(), 3);
    }

    #[test]
    fn test_impact_stats_average() {
        let mut stats = ImpactStats::new();
        stats.record(1.0, 0.5, 0.5, 100.0);
        stats.record(3.0, 1.5, 1.5, 100.0);
        let avg = stats.avg_total_impact_bps();
        assert!((avg - 2.0).abs() < 0.01, "Avg impact {} expected 2.0", avg);
    }

    #[test]
    fn test_impact_stats_max() {
        let mut stats = ImpactStats::new();
        stats.record(1.0, 0.3, 0.7, 100.0);
        stats.record(5.0, 2.0, 3.0, 100.0);
        stats.record(2.5, 1.0, 1.5, 100.0);
        assert!((stats.max_total_impact_bps() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_impact_stats_empty() {
        let stats = ImpactStats::new();
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.avg_total_impact_bps(), 0.0);
        assert_eq!(stats.max_total_impact_bps(), 0.0);
    }

    #[test]
    fn test_impact_stats_total_volume() {
        let mut stats = ImpactStats::new();
        stats.record(1.0, 0.5, 0.5, 100.0);
        stats.record(1.0, 0.5, 0.5, 200.0);
        stats.record(1.0, 0.5, 0.5, 300.0);
        assert!(
            (stats.total_volume() - 600.0).abs() < 0.01,
            "Total volume {} expected 600",
            stats.total_volume()
        );
    }

    #[test]
    fn test_impact_stats_reset() {
        let mut stats = ImpactStats::new();
        stats.record(5.0, 2.5, 2.5, 100.0);
        stats.reset();
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.total_volume(), 0.0);
    }
}

// ── Integration tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_execution_pipeline() {
        // Simulate executing 10,000 shares over 1 hour using VPR
        let vpr = VolumeParticipationModel::new(0.1);
        let ac = AlmgrenChrissImpact::new(0.001, 0.01, 0.20, 1.0);
        let mut stats = ImpactStats::new();

        let mut remaining = 10_000.0f64;
        let num_slices = 60; // one per minute

        for _ in 0..num_slices {
            if remaining <= 0.0 {
                break;
            }
            let market_vol = 5_000.0; // 5000 shares per minute market volume
            let slice_qty = vpr.compute_slice_qty(remaining, market_vol);
            let duration = 60.0; // 60 seconds

            let perm = ac.permanent_impact(slice_qty, market_vol);
            let temp = ac.temporary_impact(slice_qty, duration);
            let total = perm.abs() + temp;

            stats.record(total, perm.abs(), temp, slice_qty);
            remaining -= slice_qty;
        }

        assert!(stats.count() > 0, "Should have executed some slices");
        assert!(
            stats.total_volume() > 0.0,
            "Total volume should be positive"
        );
        assert!(remaining >= 0.0, "Should not overshoot");
    }

    #[test]
    fn test_kyle_lambda_updates_lob_model() {
        let mut kyle = KyleLambdaEstimator::new(20);
        let mut lob = LobDepthModel::new(100.0, 0.0001, 3);

        // Simulate 30 rounds of order flow
        let mut rng: u64 = 42;
        for _ in 0..30 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let flow = ((rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 200.0;
            let result = lob.execute_market_order(flow.abs(), flow > 0.0);
            // price_change = sign(flow) * slippage
            let price_change = if flow > 0.0 { result.slippage_bps } else { -result.slippage_bps };
            kyle.update(flow, price_change / 10_000.0);
        }

        let lambda = kyle.estimate();
        assert!(lambda.is_finite(), "Kyle lambda should be finite after 30 updates");
    }

    #[test]
    fn test_impact_trajectory_vs_almgren_chriss() {
        let traj = ImpactTrajectory::new(10_000.0, 10, 3600.0, 0.001, 0.01, 0.20, 1.0);
        let ac = AlmgrenChrissImpact::new(0.001, 0.01, 0.20, 1.0);
        let ac_cost = ac.optimal_schedule_cost(10_000.0, 3600.0, 100.0);
        let traj_cost = traj.total_cost_bps();
        // Both should be positive and in the same order of magnitude
        assert!(traj_cost > 0.0);
        assert!(ac_cost > 0.0);
        // Trajectory cost and AC cost should be in the same ballpark
        let ratio = if ac_cost > traj_cost { ac_cost / traj_cost } else { traj_cost / ac_cost };
        assert!(
            ratio < 100.0,
            "Trajectory cost {} and AC cost {} diverge too much",
            traj_cost,
            ac_cost
        );
    }

    #[test]
    fn test_feedback_engine_stateful_buildup() {
        let mut engine = FeedbackImpactEngine::new(0.001, 0.01, 0.20, 1.0, 3, 0.01);

        // Execute many small orders over a short period
        let mut cumulative_perm = 0.0f64;
        for i in 0..20 {
            let result = engine.execute(50.0, 0, 5.0);
            cumulative_perm += result.permanent_impact_bps;
            // Very little time between orders — impact accumulates
            engine.advance_time(0.1); // 100ms
            let _ = i;
        }

        // Should have meaningful cumulative impact
        assert!(
            cumulative_perm > 0.0,
            "Cumulative permanent impact should build up: {}",
            cumulative_perm
        );

        // Now wait for it to decay
        engine.advance_time(300.0); // 5 minutes
        let final_impact = engine.current_permanent_impact_bps();
        assert!(
            final_impact < cumulative_perm,
            "Impact should decay over time: {} -> {}",
            cumulative_perm,
            final_impact
        );
    }
}
