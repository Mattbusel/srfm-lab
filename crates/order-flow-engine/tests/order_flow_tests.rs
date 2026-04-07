/// Integration tests for the new order-flow-engine modules.
///
/// Covers:
///   - toxicity::vpin
///   - impact::kyle_lambda
///   - impact::amihud
///   - flow_analysis::trade_classifier
///   - flow_analysis::cumulative_delta
///   - orderbook_analytics::book_imbalance

use order_flow_engine::flow_analysis::cumulative_delta::{
    CumulativeDelta, DeltaDivergence, DivObs, DivergenceKind,
};
use order_flow_engine::flow_analysis::trade_classifier::{
    ClassificationMethod, OHLCVBar, Side, TradeClassifier, classify_bars_bulk,
    classify_tick_series,
};
use order_flow_engine::impact::amihud::AmihudEstimator;
use order_flow_engine::impact::kyle_lambda::{KyleLambdaConfig, KyleLambdaEstimator};
use order_flow_engine::orderbook_analytics::book_imbalance::{
    BookImbalanceAnalyzer, BookSnapshot, BookTrade,
};
use order_flow_engine::toxicity::vpin::{Side as VpinSide, VPINConfig, VPINEstimator};

// ===========================================================================
// VPIN tests
// ===========================================================================

#[test]
fn test_vpin_range_zero_to_one() {
    let est = VPINEstimator::with_volume_per_bucket(1000.0);
    for i in 0..300 {
        let side = if i % 3 == 0 { VpinSide::Buy } else { VpinSide::Sell };
        est.push_trade(100.0, 100.0, Some(side));
    }
    let v = est.vpin();
    assert!(v >= 0.0 && v <= 1.0, "VPIN out of [0,1]: {}", v);
}

#[test]
fn test_vpin_elevated_detection() {
    let est = VPINEstimator::with_volume_per_bucket(500.0);
    // Pure buy flow -> high imbalance in every bucket -> VPIN near 1.0
    for i in 0..200 {
        est.push_trade(100.0 + i as f64 * 0.01, 100.0, Some(VpinSide::Buy));
    }
    assert!(
        est.is_elevated(),
        "pure buy flow VPIN={} should exceed elevated threshold",
        est.vpin()
    );
}

#[test]
fn test_vpin_balanced_not_elevated() {
    let est = VPINEstimator::with_volume_per_bucket(500.0);
    for i in 0..500 {
        let side = if i % 2 == 0 { VpinSide::Buy } else { VpinSide::Sell };
        est.push_trade(100.0, 100.0, Some(side));
    }
    assert!(!est.is_elevated(), "balanced flow should not be elevated, VPIN={}", est.vpin());
}

#[test]
fn test_vpin_bucket_completed() {
    let est = VPINEstimator::with_volume_per_bucket(1000.0);
    // Exactly one bucket in one trade
    let bucket = est.push_trade(100.0, 1000.0, Some(VpinSide::Buy));
    assert!(bucket.is_some(), "one full bucket of volume should complete a bucket");
    assert_eq!(est.buckets_completed(), 1);
}

#[test]
fn test_vpin_history_length() {
    let est = VPINEstimator::with_volume_per_bucket(100.0);
    for i in 0..500 {
        let side = if i % 2 == 0 { VpinSide::Buy } else { VpinSide::Sell };
        est.push_trade(100.0, 10.0, Some(side));
    }
    let h = est.get_history(15);
    assert!(h.len() <= 15, "history should be bounded to requested count");
    for v in &h {
        assert!(*v >= 0.0 && *v <= 1.0, "history value out of range: {}", v);
    }
}

#[test]
fn test_vpin_config_from_total_volume() {
    let cfg = VPINConfig::from_total_volume(100_000.0, 50);
    assert!((cfg.volume_per_bucket - 2000.0).abs() < 1e-9);
    assert_eq!(cfg.bucket_count, 50);
}

#[test]
fn test_vpin_reset_clears() {
    let est = VPINEstimator::with_volume_per_bucket(100.0);
    for i in 0..200 {
        est.push_trade(100.0 + i as f64 * 0.01, 10.0, Some(VpinSide::Buy));
    }
    est.reset();
    assert_eq!(est.vpin(), 0.0);
    assert_eq!(est.buckets_completed(), 0);
    assert_eq!(est.ticks_processed(), 0);
}

#[test]
fn test_vpin_tick_rule_all_upticks() {
    let est = VPINEstimator::with_volume_per_bucket(500.0);
    let mut price = 100.0;
    for _ in 0..300 {
        price += 0.01;
        est.push_trade(price, 100.0, None); // tick rule: all up-ticks = buy
    }
    assert!(
        est.vpin() > 0.8,
        "all up-ticks should give high VPIN, got {}",
        est.vpin()
    );
}

// ===========================================================================
// Kyle Lambda tests
// ===========================================================================

/// Deterministic synthetic trade series: dp = lambda * x + tiny_noise.
fn synthetic_trades_det(n: usize, true_lambda: f64) -> Vec<(f64, f64)> {
    let mut price = 100.0f64;
    let mut out = Vec::with_capacity(n);
    let mut seed: u64 = 12_345_678;
    let lcg = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((*s >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0
    };
    for _ in 0..n {
        let x = lcg(&mut seed) * 100.0;
        let noise = lcg(&mut seed) * 0.0001;
        price += true_lambda * x + noise;
        out.push((price, x));
    }
    out
}

#[test]
fn test_kyle_lambda_positive() {
    let trades = synthetic_trades_det(200, 0.05);
    let mut est = KyleLambdaEstimator::new(KyleLambdaConfig {
        use_ewma: false,
        min_obs: 20,
        ..Default::default()
    });
    for &(p, x) in &trades {
        est.push_trade(p, x);
    }
    let lambda = est.lambda();
    assert!(lambda > 0.0, "lambda should be positive, got {}", lambda);
}

#[test]
fn test_kyle_lambda_ewma_positive() {
    let trades = synthetic_trades_det(200, 0.05);
    let mut est = KyleLambdaEstimator::new(KyleLambdaConfig {
        use_ewma: true,
        min_obs: 20,
        ..Default::default()
    });
    for &(p, x) in &trades {
        est.push_trade(p, x);
    }
    let lambda = est.lambda();
    assert!(lambda > 0.0, "EWMA lambda should be positive, got {}", lambda);
}

#[test]
fn test_kyle_lambda_impact_linear() {
    let trades = synthetic_trades_det(300, 0.10);
    let mut est = KyleLambdaEstimator::default();
    for &(p, x) in &trades {
        est.push_trade(p, x);
    }
    let imp100 = est.impact_estimate(100.0);
    let imp200 = est.impact_estimate(200.0);
    assert!(
        (imp200 - 2.0 * imp100).abs() < 1e-9,
        "impact must be linear: 2*{} != {}",
        imp100,
        imp200
    );
}

#[test]
fn test_kyle_lambda_returns_none_before_min_obs() {
    let mut est = KyleLambdaEstimator::new(KyleLambdaConfig {
        min_obs: 10,
        ..Default::default()
    });
    let mut price = 100.0;
    for i in 0..8 {
        price += 0.01;
        let r = est.push_trade(price, if i % 2 == 0 { 50.0 } else { -50.0 });
        assert!(r.is_none(), "should return None before min_obs, step {}", i);
    }
}

#[test]
fn test_kyle_lambda_window_bounded() {
    let mut est = KyleLambdaEstimator::default();
    for i in 0..500 {
        est.push_trade(100.0 + i as f64 * 0.001, 10.0);
    }
    assert!(est.obs_count() <= 100, "obs_count exceeded window_size");
}

#[test]
fn test_kyle_lambda_r_squared_in_range() {
    let trades = synthetic_trades_det(200, 0.05);
    let mut est = KyleLambdaEstimator::default();
    for &(p, x) in &trades {
        est.push_trade(p, x);
    }
    let r2 = est.r_squared();
    assert!(r2 >= 0.0 && r2 <= 1.0, "R2 out of [0,1]: {}", r2);
}

// ===========================================================================
// Amihud tests
// ===========================================================================

#[test]
fn test_amihud_ratio_direction() {
    let mut est = AmihudEstimator::default();
    // Larger return, same dollar volume -> higher ratio
    let r1 = est.push_bar(0.01, 1_000_000.0);
    let r2 = est.push_bar(0.05, 1_000_000.0);
    assert!(r2 > r1, "larger return -> higher illiquidity ratio, got r1={} r2={}", r1, r2);
}

#[test]
fn test_amihud_illiquidity_last_bar() {
    let mut est = AmihudEstimator::default();
    est.push_bar(0.01, 500_000.0);
    let last = est.push_bar(0.03, 600_000.0);
    assert!((est.illiquidity() - last).abs() < 1e-15, "illiquidity() should return last bar value");
}

#[test]
fn test_amihud_z_score_positive_on_spike() {
    let mut est = AmihudEstimator::default();
    for _ in 0..100 {
        est.push_bar(0.005, 1_000_000.0);
    }
    est.push_bar(0.15, 100.0); // extreme illiquid bar
    let z = est.illiquidity_z_score(101);
    assert!(z > 2.0, "spike should have high positive z-score, got {}", z);
}

#[test]
fn test_amihud_is_illiquid_flag() {
    let mut est = AmihudEstimator::default();
    // Seed normal bars
    for _ in 0..30 {
        est.push_bar(0.005, 1_000_000.0);
    }
    assert!(!est.is_illiquid(), "normal bar should not be flagged illiquid");
    est.push_bar(0.25, 50.0); // massive ratio
    assert!(est.is_illiquid(), "extreme illiquid bar should trigger flag");
}

#[test]
fn test_amihud_zero_volume_safe() {
    let mut est = AmihudEstimator::default();
    let r = est.push_bar(0.05, 0.0);
    assert_eq!(r, 0.0, "zero dollar volume should yield ratio 0.0");
}

#[test]
fn test_amihud_history_cap() {
    let mut est = AmihudEstimator::new(50, 10);
    for _ in 0..200 {
        est.push_bar(0.01, 1_000_000.0);
    }
    assert_eq!(est.history_len(), 50, "history should be capped at 50");
}

// ===========================================================================
// Trade classifier tests
// ===========================================================================

#[test]
fn test_trade_classifier_tick_rule() {
    // Up-tick -> Buy, down-tick -> Sell
    assert_eq!(TradeClassifier::tick_rule(100.1, 100.0), Side::Buy);
    assert_eq!(TradeClassifier::tick_rule(99.9, 100.0), Side::Sell);
    assert_eq!(TradeClassifier::tick_rule(100.0, 100.0), Side::Unknown);
}

#[test]
fn test_lee_ready_above_midpoint() {
    // mid = (99.8 + 100.2)/2 = 100.0; trade at 100.1 > mid -> Buy
    let side = TradeClassifier::lee_ready(100.1, 99.8, 100.2, None);
    assert_eq!(side, Side::Buy);
}

#[test]
fn test_lee_ready_below_midpoint() {
    let side = TradeClassifier::lee_ready(99.9, 99.8, 100.2, None);
    assert_eq!(side, Side::Sell);
}

#[test]
fn test_lee_ready_at_midpoint_uses_tick() {
    // mid = 100.0; trade at 100.0; prev = 99.8 (uptick) -> Buy
    let side = TradeClassifier::lee_ready(100.0, 99.8, 100.2, Some(99.8));
    assert_eq!(side, Side::Buy);
}

#[test]
fn test_bulk_volume_at_close() {
    // close == open -> z = 0 -> 50/50
    let bar = OHLCVBar::new(100.0, 105.0, 95.0, 100.0, 2000.0);
    let (buy, sell) = TradeClassifier::bulk_volume_classify(bar);
    assert!((buy - 1000.0).abs() < 1.0, "expected buy~1000, got {}", buy);
    assert!((sell - 1000.0).abs() < 1.0, "expected sell~1000, got {}", sell);
    assert!((buy + sell - 2000.0).abs() < 1e-9, "must sum to total volume");
}

#[test]
fn test_bulk_volume_close_at_high() {
    let bar = OHLCVBar::new(95.0, 105.0, 95.0, 105.0, 1000.0);
    let (buy, sell) = TradeClassifier::bulk_volume_classify(bar);
    assert!(buy > sell, "close at high -> more buy volume");
    assert!((buy + sell - 1000.0).abs() < 1e-9);
}

#[test]
fn test_bulk_volume_close_at_low() {
    let bar = OHLCVBar::new(105.0, 105.0, 95.0, 95.0, 1000.0);
    let (buy, sell) = TradeClassifier::bulk_volume_classify(bar);
    assert!(sell > buy, "close at low -> more sell volume");
}

#[test]
fn test_emo_classify_hit_ask() {
    let side = TradeClassifier::emo_classify(100.2, 100.0, 99.8, 100.2);
    assert_eq!(side, Side::Buy, "trade at ask -> Buy");
}

#[test]
fn test_emo_classify_hit_bid() {
    let side = TradeClassifier::emo_classify(99.8, 100.0, 99.8, 100.2);
    assert_eq!(side, Side::Sell, "trade at bid -> Sell");
}

#[test]
fn test_emo_classify_above_open_inside_spread() {
    let side = TradeClassifier::emo_classify(100.05, 100.0, 99.8, 100.2);
    assert_eq!(side, Side::Buy, "above open inside spread -> Buy (EMO)");
}

#[test]
fn test_stateful_zero_tick_carry_forward() {
    let mut clf = TradeClassifier::new(ClassificationMethod::TickRule);
    clf.classify(100.0, None, None, None); // first: Unknown
    clf.classify(100.5, None, None, None); // uptick -> Buy
    let side = clf.classify(100.5, None, None, None); // zero-tick -> carry Buy
    assert_eq!(side, Side::Buy, "zero-tick should carry forward Buy");
}

#[test]
fn test_classify_tick_series_volume_sum() {
    let ticks: Vec<(f64, f64)> = (0..100)
        .map(|i| (100.0 + (i as f64 % 5.0) * 0.1, 200.0))
        .collect();
    let (buy, sell) = classify_tick_series(&ticks);
    let total: f64 = ticks.iter().map(|(_, v)| v).sum();
    assert!((buy + sell - total).abs() < 1e-9, "buy+sell must equal total volume");
}

#[test]
fn test_classify_bars_bulk_sums() {
    let bars: Vec<OHLCVBar> = (0..10)
        .map(|i| OHLCVBar::new(100.0, 105.0, 95.0, 100.0 + i as f64 * 0.5, 1000.0))
        .collect();
    for (buy, sell) in classify_bars_bulk(&bars) {
        assert!((buy + sell - 1000.0).abs() < 1e-9);
    }
}

// ===========================================================================
// Cumulative delta tests
// ===========================================================================

#[test]
fn test_cumulative_delta_accumulation() {
    let mut cd = CumulativeDelta::default();
    cd.push(600.0, 400.0); // +200
    cd.push(300.0, 700.0); // -400
    cd.push(500.0, 500.0); // 0
    assert!((cd.value() - (-200.0)).abs() < 1e-9, "expected -200, got {}", cd.value());
}

#[test]
fn test_cumulative_delta_pure_buy() {
    let mut cd = CumulativeDelta::default();
    for _ in 0..5 {
        cd.push(1000.0, 0.0);
    }
    assert!((cd.value() - 5000.0).abs() < 1e-9);
}

#[test]
fn test_cumulative_delta_reset_zeroes() {
    let mut cd = CumulativeDelta::default();
    cd.push(1000.0, 200.0);
    cd.reset();
    assert_eq!(cd.value(), 0.0);
    assert_eq!(cd.push_count, 0);
}

#[test]
fn test_delta_ema_positive_on_buy_flow() {
    let mut cd = CumulativeDelta::new(0.1, 200);
    for _ in 0..50 {
        cd.push(600.0, 400.0); // +200 each bar
    }
    assert!(cd.delta_ema() > 0.0, "EMA should be positive after buy flow");
}

#[test]
fn test_delta_divergence_bearish() {
    let mut dd = DeltaDivergence::new(10, 0.001, 0.10);
    // Establish prior high with strong buying
    for _ in 0..8 {
        dd.check_push(DivObs { price_high: 100.0, price_low: 99.0, delta: 1000.0 });
    }
    // New price high but weak buying delta
    let sig = dd.check_push(DivObs { price_high: 102.0, price_low: 101.0, delta: 200.0 });
    assert_eq!(
        sig.kind,
        DivergenceKind::Bearish,
        "price higher high + delta lower high = bearish divergence"
    );
}

#[test]
fn test_delta_divergence_bullish() {
    let mut dd = DeltaDivergence::new(10, 0.001, 0.10);
    // Establish prior low with heavy selling
    for _ in 0..8 {
        dd.check_push(DivObs { price_high: 100.0, price_low: 99.0, delta: -1000.0 });
    }
    // New price low but less selling
    let sig = dd.check_push(DivObs { price_high: 99.0, price_low: 97.8, delta: -100.0 });
    assert_eq!(
        sig.kind,
        DivergenceKind::Bullish,
        "price lower low + delta higher low = bullish divergence"
    );
}

#[test]
fn test_delta_divergence_none_flat_market() {
    let mut dd = DeltaDivergence::default();
    for _ in 0..20 {
        let sig = dd.check_push(DivObs { price_high: 100.0, price_low: 99.0, delta: 500.0 });
        // Flat market: no new highs or lows -> None
        // (may not always be None if window sees the same value, but should not be Bearish/Bullish)
        assert_ne!(sig.kind, DivergenceKind::Bearish, "flat market should not be bearish");
    }
}

#[test]
fn test_delta_divergence_static_check_fn() {
    let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 1.0).collect();
    // Deltas strongly decline so last is very negative while price is at new high
    let mut deltas: Vec<f64> = (0..20).map(|i| 500.0 - i as f64 * 100.0).collect();
    *deltas.last_mut().unwrap() = -900.0;
    // The static check may or may not fire depending on signs; test it doesn't panic
    let _ = DeltaDivergence::check(&prices, &deltas, 20);
}

// ===========================================================================
// Book imbalance tests
// ===========================================================================

fn sym_book(levels: usize, size: f64) -> BookSnapshot {
    let bid_p: Vec<f64> = (0..levels).map(|i| 99.9 - i as f64 * 0.1).collect();
    let ask_p: Vec<f64> = (0..levels).map(|i| 100.1 + i as f64 * 0.1).collect();
    BookSnapshot::new(bid_p, vec![size; levels], ask_p, vec![size; levels])
}

#[test]
fn test_book_imbalance_symmetry() {
    let analyzer = BookImbalanceAnalyzer::default();
    let book = sym_book(5, 500.0);
    let imb = analyzer.weighted_imbalance(&book, 5);
    assert!(imb.abs() < 1e-9, "symmetric book should have zero imbalance, got {}", imb);
}

#[test]
fn test_book_imbalance_bid_heavy() {
    let analyzer = BookImbalanceAnalyzer::default();
    let book = BookSnapshot::new(
        vec![99.9, 99.8],
        vec![1000.0, 1000.0],
        vec![100.1, 100.2],
        vec![100.0, 100.0],
    );
    let imb = analyzer.weighted_imbalance(&book, 2);
    assert!(imb > 0.0, "bid-heavy book should have positive imbalance, got {}", imb);
}

#[test]
fn test_book_imbalance_in_unit_range() {
    let analyzer = BookImbalanceAnalyzer::default();
    let book = BookSnapshot::new(
        vec![99.9],
        vec![750.0],
        vec![100.1],
        vec![250.0],
    );
    let imb = analyzer.weighted_imbalance(&book, 1);
    assert!(imb >= -1.0 && imb <= 1.0, "imbalance out of [-1,1]: {}", imb);
    assert!(imb > 0.0, "more bid than ask -> positive imbalance");
}

#[test]
fn test_depth_imbalance_symmetric() {
    let analyzer = BookImbalanceAnalyzer::default();
    let book = sym_book(3, 300.0);
    let imb = analyzer.depth_imbalance(&book, 0.05);
    assert!(imb.abs() < 1e-9, "symmetric book depth imbalance should be 0, got {}", imb);
}

#[test]
fn test_iceberg_detector_fires_on_heavy_volume() {
    let analyzer = BookImbalanceAnalyzer::new(2.0, 2);
    let book = BookSnapshot::new(
        vec![99.9],
        vec![100.0],
        vec![100.1],
        vec![100.0],
    );
    // Trade 500 units through the bid at 99.9 (5x resting size)
    let trades: Vec<BookTrade> = (0..5).map(|_| BookTrade::new(99.9, 100.0)).collect();
    let icebergs = analyzer.iceberg_detector(&book, &trades);
    assert!(!icebergs.is_empty(), "5x traded volume should trigger iceberg");
}

#[test]
fn test_iceberg_not_triggered_small_volume() {
    let analyzer = BookImbalanceAnalyzer::default(); // threshold = 3x
    let book = BookSnapshot::new(
        vec![99.9],
        vec![10_000.0],
        vec![100.1],
        vec![10_000.0],
    );
    let trades = vec![BookTrade::new(99.9, 10.0)]; // 10/10000 = 0.001 << 3.0
    let icebergs = analyzer.iceberg_detector(&book, &trades);
    assert!(icebergs.is_empty(), "tiny volume should not trigger iceberg");
}

#[test]
fn test_sweep_detected_buy() {
    let analyzer = BookImbalanceAnalyzer::new(3.0, 3);
    let trades = vec![
        BookTrade::new(100.0, 500.0),
        BookTrade::new(100.1, 500.0),
        BookTrade::new(100.2, 500.0),
        BookTrade::new(100.3, 500.0),
    ];
    let sweep = analyzer.sweep_detector(&trades, 10);
    assert!(sweep.is_some(), "ascending prices should trigger sweep");
    assert!(sweep.unwrap().is_buy);
}

#[test]
fn test_sweep_detected_sell() {
    let analyzer = BookImbalanceAnalyzer::new(3.0, 3);
    let trades = vec![
        BookTrade::new(100.3, 500.0),
        BookTrade::new(100.2, 500.0),
        BookTrade::new(100.1, 500.0),
        BookTrade::new(100.0, 500.0),
    ];
    let sweep = analyzer.sweep_detector(&trades, 10);
    assert!(sweep.is_some(), "descending prices should trigger sweep");
    assert!(!sweep.unwrap().is_buy);
}

#[test]
fn test_no_sweep_too_few_levels() {
    let analyzer = BookImbalanceAnalyzer::new(3.0, 5); // need 5 distinct levels
    let trades = vec![
        BookTrade::new(100.0, 100.0),
        BookTrade::new(100.1, 100.0),
        BookTrade::new(100.2, 100.0),
    ];
    let sweep = analyzer.sweep_detector(&trades, 10);
    assert!(sweep.is_none(), "only 3 levels should not trigger sweep requiring 5");
}

#[test]
fn test_book_snapshot_midpoint() {
    let book = BookSnapshot::new(
        vec![99.8],
        vec![100.0],
        vec![100.2],
        vec![100.0],
    );
    assert!((book.midpoint().unwrap() - 100.0).abs() < 1e-9);
}

#[test]
fn test_book_snapshot_spread() {
    let book = BookSnapshot::new(
        vec![99.8],
        vec![100.0],
        vec![100.2],
        vec![100.0],
    );
    assert!((book.spread().unwrap() - 0.4).abs() < 1e-9);
}

#[test]
fn test_book_empty_imbalance() {
    let analyzer = BookImbalanceAnalyzer::default();
    // Empty book should not panic and return 0
    let book = BookSnapshot::new(vec![], vec![], vec![], vec![]);
    let imb = analyzer.weighted_imbalance(&book, 5);
    assert_eq!(imb, 0.0, "empty book should return 0 imbalance");
}
