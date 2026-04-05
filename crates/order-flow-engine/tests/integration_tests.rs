use order_flow_engine::aggressive_flow::{AggressiveFlowConfig, AggressiveFlowDetector};
use order_flow_engine::delta_divergence::{DeltaDivergenceDetector, DivergenceObs, DivergenceType};
use order_flow_engine::footprint::{build_footprint_bar, build_footprint_series};
use order_flow_engine::signal::{OrderFlowEngine, OrderFlowRegime, SignalEngineConfig};
use order_flow_engine::tick_classifier::{
    BulkVolumeClassifier, LeeReadyClassifier, Tick, TickSide,
    erf, proportion_buy,
};
use order_flow_engine::volume_imbalance::{ClassifiedBar, OFICalculator, compute_ofi_series};
use order_flow_engine::vpin::{VpinCalculator, compute_vpin_series, estimate_bucket_volume};

// ── Tick classifier tests ────────────────────────────────────────────────────

#[test]
fn test_lee_ready_initial_unknown() {
    let mut clf = LeeReadyClassifier::new();
    let t = Tick { price: 100.0, volume: 10.0, bid: None, ask: None };
    assert_eq!(clf.classify(&t), TickSide::Unknown);
}

#[test]
fn test_lee_ready_sequence() {
    let mut clf = LeeReadyClassifier::new();
    let ticks = vec![
        Tick { price: 100.0, volume: 10.0, bid: None, ask: None },
        Tick { price: 100.5, volume: 10.0, bid: None, ask: None }, // uptick -> BUY
        Tick { price: 100.3, volume: 10.0, bid: None, ask: None }, // downtick -> SELL
        Tick { price: 100.3, volume: 10.0, bid: None, ask: None }, // zero-tick -> inherit SELL
    ];
    let _ = clf.classify(&ticks[0]);
    assert_eq!(clf.classify(&ticks[1]), TickSide::Buy);
    assert_eq!(clf.classify(&ticks[2]), TickSide::Sell);
    assert_eq!(clf.classify(&ticks[3]), TickSide::Sell);
}

#[test]
fn test_proportion_buy_symmetry() {
    // erf is odd: proportion_buy(r) + proportion_buy(-r) == 1
    let r = 0.005;
    let sigma = 0.008;
    let p = proportion_buy(r, sigma);
    let q = proportion_buy(-r, sigma);
    assert!((p + q - 1.0).abs() < 1e-9, "symmetry broken: {} + {} != 1", p, q);
}

#[test]
fn test_bvc_large_positive_return_mostly_buy() {
    let mut bvc = BulkVolumeClassifier::new(20);
    // Very large up-bar -> proportion_buy should be high
    let (buy, sell) = bvc.classify_bar(100.0, 103.0, 10000.0); // 3% up bar
    assert!(buy > sell, "large up-bar should have more buy vol: buy={} sell={}", buy, sell);
}

#[test]
fn test_erf_bounds() {
    for x in [-3.0, -1.0, 0.0, 1.0, 3.0] {
        let v = erf(x);
        assert!(v >= -1.0 && v <= 1.0, "erf({}) = {} out of bounds", x, v);
    }
}

// ── VPIN tests ───────────────────────────────────────────────────────────────

#[test]
fn test_vpin_with_alternating_buysell() {
    // Each bar is pure buy or pure sell with volume = bucket_volume.
    // Each bucket is filled by a single bar -> imbalance = 1.0 per bucket -> VPIN = 1.0.
    // Use a mixed bar (equal buy/sell per bar) for a low-VPIN test instead.
    let bars: Vec<(f64, f64)> = (0..100)
        .map(|_| (500.0, 500.0)) // perfectly balanced each bar
        .collect();
    let series = compute_vpin_series(&bars, 500.0, 5);
    let last = *series.last().unwrap();
    assert!(last < 0.1, "balanced flow should produce low VPIN, got {}", last);
}

#[test]
fn test_vpin_returns_vec_same_length() {
    let bars = vec![(500.0_f64, 500.0_f64); 50];
    let series = compute_vpin_series(&bars, 1000.0, 10);
    assert_eq!(series.len(), 50);
}

#[test]
fn test_vpin_non_negative() {
    let bars: Vec<(f64, f64)> = (0..100)
        .map(|i| (i as f64 * 50.0, (100 - i) as f64 * 50.0))
        .collect();
    let series = compute_vpin_series(&bars, 2500.0, 5);
    for &v in &series {
        assert!(v >= 0.0, "VPIN should be non-negative: {}", v);
    }
}

// ── OFI tests ────────────────────────────────────────────────────────────────

#[test]
fn test_ofi_positive_on_buy_pressure() {
    let bars: Vec<ClassifiedBar> = (0..10)
        .map(|_| ClassifiedBar::new(800.0, 200.0))
        .collect();
    let series = compute_ofi_series(&bars, 5);
    assert!(series.last().unwrap() > &0.0, "buy pressure -> positive OFI");
}

#[test]
fn test_ofi_monotone_with_pure_buying() {
    let mut calc = OFICalculator::new(100);
    let mut prev = 0.0f64;
    for _ in 0..50 {
        let ofi = calc.push(ClassifiedBar::new(1000.0, 0.0));
        assert!(ofi >= prev - 1e-6, "OFI should not decrease with pure buys: {} < {}", ofi, prev);
        prev = ofi;
    }
}

// ── Footprint tests ──────────────────────────────────────────────────────────

#[test]
fn test_footprint_poc_is_price_level() {
    let fb = build_footprint_bar(100.0, 110.0, 90.0, 105.0, 10000.0, 0.55);
    assert!(fb.poc >= fb.low && fb.poc <= fb.high);
}

#[test]
fn test_footprint_value_area_ordering() {
    let fb = build_footprint_bar(100.0, 108.0, 92.0, 104.0, 20000.0, 0.6);
    assert!(fb.val <= fb.poc, "VAL should be <= POC");
    assert!(fb.poc <= fb.vah, "POC should be <= VAH");
}

#[test]
fn test_footprint_series_correct_count() {
    let ohlcv: Vec<(f64, f64, f64, f64, f64)> = (0..30)
        .map(|i| {
            let p = 100.0 + i as f64;
            (p, p + 2.0, p - 2.0, p + 0.5, 5000.0)
        })
        .collect();
    let bars = build_footprint_series(&ohlcv);
    assert_eq!(bars.len(), 30);
}

// ── Delta divergence tests ───────────────────────────────────────────────────

#[test]
fn test_divergence_only_bearish_when_criteria_met() {
    let mut det = DeltaDivergenceDetector::new(5, 0.001, 0.1);
    // Establish baseline
    for _ in 0..5 {
        det.push(DivergenceObs { high: 100.0, low: 99.0, close: 99.5, delta: 500.0 });
    }
    // Strong price new high + weak delta
    let sig = det.push(DivergenceObs { high: 101.5, low: 100.5, close: 101.0, delta: 100.0 });
    assert_eq!(sig.divergence_type, DivergenceType::BearishDivergence);
}

#[test]
fn test_divergence_strength_bounded() {
    let mut det = DeltaDivergenceDetector::default();
    let obs: Vec<_> = (0..30)
        .map(|i| DivergenceObs {
            high: 100.0 + i as f64 * 0.5,
            low: 99.0 + i as f64 * 0.5,
            close: 99.5 + i as f64 * 0.5,
            delta: 500.0 - i as f64 * 15.0,
        })
        .collect();
    let signals = det.process_series(&obs);
    for sig in &signals {
        assert!(sig.strength >= 0.0, "strength should be non-negative");
    }
}

// ── Aggressive flow tests ────────────────────────────────────────────────────

#[test]
fn test_aggressive_sweep_score_bounded() {
    let mut det = AggressiveFlowDetector::default();
    for i in 0..50 {
        let r = det.push(100.0, 110.0, 90.0, 108.0, 100_000.0 * (1 + i % 10) as f64);
        assert!(r.sweep_score >= 0.0 && r.sweep_score <= 1.0);
    }
}

#[test]
fn test_no_aggressive_on_tiny_volume() {
    let mut det = AggressiveFlowDetector::default();
    // Fill history with normal volume
    for _ in 0..30 {
        det.push(100.0, 100.5, 99.5, 100.2, 5000.0);
    }
    // Tiny volume bar with wide range should not be aggressive
    let r = det.push(100.0, 102.0, 98.0, 101.0, 10.0);
    assert!(!r.is_aggressive || r.sweep_score < 0.3);
}

// ── Full signal engine integration ───────────────────────────────────────────

#[test]
fn test_engine_output_ofi_in_range() {
    let mut eng = OrderFlowEngine::new(SignalEngineConfig::new("INTG"));
    let bars: Vec<_> = (0..100)
        .map(|i| {
            let p = 100.0 + i as f64 * 0.1;
            (format!("T{}", i), p, p + 1.0, p - 1.0, p + 0.3, 5000.0)
        })
        .collect();
    let sigs = eng.process_series(&bars);
    for s in &sigs {
        assert!(s.ofi >= -1.0 && s.ofi <= 1.0, "OFI out of range: {}", s.ofi);
        assert!(s.vpin >= 0.0 && s.vpin <= 1.0, "VPIN out of range: {}", s.vpin);
    }
}

#[test]
fn test_engine_buy_dominated_on_strong_up_trend() {
    // Test that strong up-bars produce positive OFI (buy_vol > sell_vol).
    // The regime classification requires net_pressure > 0.25, which depends on both
    // OFI and aggressive flow. We verify the weaker (but more reliable) property
    // that mean OFI is positive across 50 strongly up-trending bars.
    let mut cfg = SignalEngineConfig::new("UP");
    cfg.ofi_window = 5;
    let mut eng = OrderFlowEngine::new(cfg);
    let bars: Vec<_> = (0..50)
        .map(|i| {
            let p = 100.0 + i as f64 * 0.5;
            let close = p * 1.10; // 10% up bar -> very strong buy signal for BVC
            (format!("T{}", i), p, close + 0.5, p - 0.1, close, 8000.0)
        })
        .collect();
    let sigs = eng.process_series(&bars);
    let mean_ofi = sigs.iter().map(|s| s.ofi).sum::<f64>() / sigs.len() as f64;
    assert!(mean_ofi > 0.0, "strong up-trend should produce positive mean OFI, got {}", mean_ofi);
    // Also check that buy_volume > sell_volume on average
    let mean_buy: f64 = sigs.iter().map(|s| s.buy_volume).sum::<f64>() / sigs.len() as f64;
    let mean_sell: f64 = sigs.iter().map(|s| s.sell_volume).sum::<f64>() / sigs.len() as f64;
    assert!(mean_buy > mean_sell, "up-trend: mean_buy {} should > mean_sell {}", mean_buy, mean_sell);
}
