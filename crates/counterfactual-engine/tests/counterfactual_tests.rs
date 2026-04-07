//! Integration tests for the counterfactual-engine extensions.
//!
//! Tests cover:
//! - Propensity score matching and balance
//! - ATE/ATT estimation direction and magnitude
//! - "What if" scenario engine: trade counts, P&L deltas
//! - Causal graph: path finding and coefficient computation

use counterfactual_engine::{
    ATEEstimator, FeatureVec, LogisticRegression, PropensityScoreEstimator,
    Bar, BaselineParams, ParameterGrid, ScenarioSpec, WhatIfEngine,
    CausalEdge, CausalGraph, CausalNode, EdgeType, NodeType,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_treated(n: usize) -> Vec<FeatureVec> {
    // Treated: high bh_mass (3.0-4.5), trending (hurst 0.65-0.75)
    (0..n)
        .map(|i| {
            let phase = i as f64 / n as f64;
            FeatureVec::new(
                3.0 + phase * 1.5,
                0.65 + phase * 0.10,
                0.70 + phase * 0.15,
                9.0 + phase * 4.0,
                (i % 5) as f64,
            )
        })
        .collect()
}

fn make_control(n: usize) -> Vec<FeatureVec> {
    // Control: lower bh_mass (1.0-2.5), near-random walk (hurst 0.45-0.55)
    (0..n)
        .map(|i| {
            let phase = i as f64 / n as f64;
            FeatureVec::new(
                1.0 + phase * 1.5,
                0.45 + phase * 0.10,
                0.30 + phase * 0.20,
                8.0 + phase * 5.0,
                (i % 5) as f64,
            )
        })
        .collect()
}

fn make_trending_bars(n: usize) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(n);
    let mut price = 100.0f64;
    let mut state = 42u64;
    for i in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = (state as f64 / u64::MAX as f64) * 0.004 - 0.001;
        price = (price * (1.0 + r + 0.001)).max(1.0);
        let mut bar = Bar::new(i, price, 2.5, 0.65);
        bar.event_calendar_block = i % 7 == 3;
        bar.regime = "bull".to_string();
        bars.push(bar);
    }
    bars
}

fn make_blocked_bars(n: usize) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(n);
    let mut price = 100.0f64;
    for i in 0..n {
        price *= 1.002;
        let mut bar = Bar::new(i, price, 2.5, 0.65);
        bar.event_calendar_block = true;
        bars.push(bar);
    }
    bars
}

// ---------------------------------------------------------------------------
// Propensity score tests
// ---------------------------------------------------------------------------

#[test]
fn test_propensity_matching_balance() {
    let treated = make_treated(20);
    let control = make_control(20);

    let mut est = PropensityScoreEstimator::new();
    est.fit(&treated, &control);
    let pairs = est.match_units(&treated, &control, 0.6);

    // Balance report should have entries for all five features
    let report = est.balance_report(&treated, &control, &pairs);
    assert_eq!(report.len(), 5);

    // After matching with a wide caliper all SMDs should be finite
    for (feature, &smd) in &report {
        assert!(smd.is_finite() || pairs.is_empty(),
            "SMD for {feature} should be finite, got {smd}");
    }
}

#[test]
fn test_propensity_scores_separated() {
    // Treated and control have very different features
    let treated = make_treated(20);
    let control = make_control(20);

    let mut est = PropensityScoreEstimator::new();
    est.fit(&treated, &control);

    let mean_treated_score = est.treated_scores.iter().sum::<f64>() / est.treated_scores.len() as f64;
    let mean_control_score = est.control_scores.iter().sum::<f64>() / est.control_scores.len() as f64;

    // Treated units should have higher propensity scores
    assert!(
        mean_treated_score > mean_control_score,
        "treated mean score {mean_treated_score:.3} should exceed control {mean_control_score:.3}"
    );
}

#[test]
fn test_propensity_caliper_strictly_respected() {
    let treated = make_treated(15);
    let control = make_control(15);

    let mut est = PropensityScoreEstimator::new();
    est.fit(&treated, &control);

    let caliper = 0.10;
    let pairs = est.match_units(&treated, &control, caliper);

    for (ti, ci) in &pairs {
        let ts = est.estimate(&treated[*ti]);
        let cs = est.estimate(&control[*ci]);
        assert!(
            (ts - cs).abs() <= caliper + 1e-9,
            "Pair ({ti},{ci}) score diff {:.4} exceeds caliper {caliper}",
            (ts - cs).abs()
        );
    }
}

#[test]
fn test_no_duplicate_control_units() {
    let treated = make_treated(15);
    let control = make_control(15);

    let mut est = PropensityScoreEstimator::new();
    est.fit(&treated, &control);
    let pairs = est.match_units(&treated, &control, 0.8);

    let mut seen = std::collections::HashSet::new();
    for (_, ci) in &pairs {
        assert!(seen.insert(*ci), "control unit {ci} matched twice");
    }
}

// ---------------------------------------------------------------------------
// ATE tests
// ---------------------------------------------------------------------------

#[test]
fn test_ate_direction() {
    // Treated units earn more -> ATE should be positive
    let matched_pairs = vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)];
    let treated_outcomes = vec![0.10, 0.08, 0.12, 0.09, 0.11];
    let control_outcomes  = vec![0.02, 0.03, 0.01, 0.04, 0.02];

    let est = ATEEstimator::new();
    let (ate, se) = est.compute_ate(&matched_pairs, &treated_outcomes, &control_outcomes);

    assert!(ate > 0.0, "ATE should be positive when treated outperforms, got {ate:.4}");
    assert!(se >= 0.0, "SE should be non-negative, got {se:.4}");
}

#[test]
fn test_ate_negative_when_control_better() {
    let matched_pairs = vec![(0, 0), (1, 1), (2, 2)];
    let treated_outcomes = vec![-0.05, -0.03, -0.04];
    let control_outcomes  = vec![ 0.04,  0.03,  0.05];

    let est = ATEEstimator::new();
    let (ate, _) = est.compute_ate(&matched_pairs, &treated_outcomes, &control_outcomes);
    assert!(ate < 0.0, "ATE should be negative when control outperforms, got {ate:.4}");
}

#[test]
fn test_ate_zero_for_identical_outcomes() {
    let matched_pairs = vec![(0, 0), (1, 1)];
    let outcomes = vec![0.05, 0.07];
    let est = ATEEstimator::new();
    let (ate, _) = est.compute_ate(&matched_pairs, &outcomes, &outcomes);
    assert!(ate.abs() < 1e-9, "ATE should be 0 for identical outcomes, got {ate}");
}

#[test]
fn test_att_equals_ate_for_1_to_1() {
    let matched_pairs = vec![(0, 0), (1, 1), (2, 2)];
    let treated_outcomes = vec![0.10, 0.12, 0.09];
    let control_outcomes  = vec![0.05, 0.04, 0.06];
    let est = ATEEstimator::new();
    let (ate, _) = est.compute_ate(&matched_pairs, &treated_outcomes, &control_outcomes);
    let att = est.compute_att(&matched_pairs, &treated_outcomes, &control_outcomes);
    assert!((ate - att).abs() < 1e-9, "ATE ({ate:.6}) != ATT ({att:.6}) for 1:1 matching");
}

#[test]
fn test_t_test_significant() {
    let est = ATEEstimator::new();
    let (t, sig) = est.t_test(0.10, 0.02); // t = 5.0 >> 1.96
    assert!(sig, "t={t:.2} should be significant");
}

#[test]
fn test_t_test_not_significant() {
    let est = ATEEstimator::new();
    let (t, sig) = est.t_test(0.005, 0.01); // t = 0.5 < 1.96
    assert!(!sig, "t={t:.2} should not be significant");
}

#[test]
fn test_bootstrap_ci_width_reasonable() {
    let matched_pairs: Vec<(usize, usize)> = (0..10).map(|i| (i, i)).collect();
    let treated_outcomes: Vec<f64> = (0..10).map(|i| 0.05 + i as f64 * 0.002).collect();
    let control_outcomes: Vec<f64>  = (0..10).map(|i| 0.01 + i as f64 * 0.001).collect();

    let est = ATEEstimator::new();
    let (ate, _) = est.compute_ate(&matched_pairs, &treated_outcomes, &control_outcomes);
    let (lo, hi) = est.bootstrap_ci(&matched_pairs, &treated_outcomes, &control_outcomes, 1000, 0.05, 42);

    assert!(lo <= ate, "CI lower {lo:.4} should be <= ATE {ate:.4}");
    assert!(hi >= ate, "CI upper {hi:.4} should be >= ATE {ate:.4}");
    assert!(hi - lo > 0.0, "CI width should be positive");
}

// ---------------------------------------------------------------------------
// Scenario / "What if" engine tests
// ---------------------------------------------------------------------------

#[test]
fn test_what_if_lower_threshold_more_trades() {
    let bars = make_trending_bars(120);
    let params = BaselineParams::default(); // bh_mass_thresh = 2.0, bars have bh_mass = 2.5
    let engine = WhatIfEngine::new();

    // Lower threshold to 1.0 -- should fire on every bar
    let result = engine.run_counterfactual(
        &bars,
        &params,
        ScenarioSpec::lower_bh_threshold(1.0),
    );

    assert!(
        result.counterfactual_trades >= result.baseline_trades,
        "Lower threshold: cf_trades={} should be >= baseline={}",
        result.counterfactual_trades, result.baseline_trades
    );
}

#[test]
fn test_what_if_higher_threshold_fewer_trades() {
    let bars = make_trending_bars(120);
    let params = BaselineParams::default();
    let engine = WhatIfEngine::new();

    // Raise threshold above bh_mass = 2.5 -- should block all trades
    let result = engine.run_counterfactual(
        &bars,
        &params,
        ScenarioSpec::new("high_thresh").with_param("bh_mass_thresh", 3.5),
    );

    assert!(
        result.counterfactual_trades <= result.baseline_trades,
        "Higher threshold: cf_trades={} should be <= baseline={}",
        result.counterfactual_trades, result.baseline_trades
    );
}

#[test]
fn test_what_if_no_event_filter_more_trades() {
    let bars = make_blocked_bars(100);
    let params = BaselineParams::default();
    let engine = WhatIfEngine::new();

    let result = engine.run_counterfactual(&bars, &params, ScenarioSpec::no_event_filter());

    assert!(
        result.counterfactual_trades >= result.baseline_trades,
        "No filter: cf_trades={} should be >= baseline={}",
        result.counterfactual_trades, result.baseline_trades
    );
}

#[test]
fn test_what_if_min_hold_8_bars() {
    let bars = make_trending_bars(100);
    let params = BaselineParams::default();
    let engine = WhatIfEngine::new();

    let result = engine.run_counterfactual(&bars, &params, ScenarioSpec::min_hold(8));

    assert_eq!(result.name, "min_hold=8");
    assert!(result.baseline_pnl.is_finite());
    assert!(result.counterfactual_pnl.is_finite());
    assert!(result.pnl_delta.is_finite());
}

#[test]
fn test_pnl_delta_is_consistent() {
    let bars = make_trending_bars(80);
    let params = BaselineParams::default();
    let engine = WhatIfEngine::new();
    let result = engine.run_counterfactual(&bars, &params, ScenarioSpec::no_event_filter());

    let expected = result.counterfactual_pnl - result.baseline_pnl;
    assert!(
        (result.pnl_delta - expected).abs() < 1e-6,
        "pnl_delta {} != cf-baseline {expected}", result.pnl_delta
    );
}

#[test]
fn test_equity_curves_have_correct_length() {
    let bars = make_trending_bars(75);
    let params = BaselineParams::default();
    let engine = WhatIfEngine::new();
    let result = engine.run_counterfactual(&bars, &params, ScenarioSpec::new("len_test"));

    assert_eq!(result.baseline_equity_curve.len(), bars.len());
    assert_eq!(result.counterfactual_equity_curve.len(), bars.len());
}

#[test]
fn test_parameter_sweep_sorted_descending() {
    let bars = make_trending_bars(80);
    let params = BaselineParams::default();
    let engine = WhatIfEngine::new();

    let grid = ParameterGrid::new()
        .add_axis("bh_mass_thresh", vec![1.0, 1.5, 2.0, 2.5, 3.0]);

    let results = engine.run_parameter_sweep(&bars, &params, &grid);
    assert_eq!(results.len(), 5);

    for w in results.windows(2) {
        assert!(
            w[0].pnl_delta >= w[1].pnl_delta,
            "sweep not sorted: {} < {}", w[0].pnl_delta, w[1].pnl_delta
        );
    }
}

#[test]
fn test_grid_cartesian_product_size() {
    let grid = ParameterGrid::new()
        .add_axis("bh_mass_thresh", vec![1.5, 2.0, 2.5])
        .add_axis("min_hold_bars", vec![3.0, 8.0]);
    assert_eq!(grid.len(), 6);
    assert_eq!(grid.scenarios().len(), 6);
}

#[test]
fn test_batch_run_all_results_computed() {
    let bars = make_trending_bars(60);
    let params = BaselineParams::default();
    let engine = WhatIfEngine::new();
    let scenarios = counterfactual_engine::scenario_analysis::standard_larsa_scenarios();
    let n = scenarios.len();
    let results = engine.run_batch(&bars, &params, scenarios);
    assert_eq!(results.len(), n);
    for r in &results {
        assert!(r.baseline_pnl.is_finite(), "baseline P&L for '{}' not finite", r.name);
        assert!(r.counterfactual_pnl.is_finite(), "cf P&L for '{}' not finite", r.name);
    }
}

// ---------------------------------------------------------------------------
// Causal graph tests
// ---------------------------------------------------------------------------

#[test]
fn test_causal_path_coefficient() {
    let g = CausalGraph::larsa_default();
    let paths = g.find_paths("BH_MASS", "OUTCOME");
    assert!(!paths.is_empty(), "should find at least one path BH_MASS -> OUTCOME");

    for path in &paths {
        let coef = g.compute_path_coefficient(path);
        assert!(coef.is_finite(), "path coefficient should be finite");
    }
}

#[test]
fn test_causal_total_effect_bh_mass_positive() {
    let g = CausalGraph::larsa_default();
    let total = g.total_effect("BH_MASS", "OUTCOME");
    assert!(total > 0.0, "BH_MASS -> OUTCOME total effect should be positive, got {total:.4}");
}

#[test]
fn test_causal_total_effect_hurst_to_position_size() {
    let g = CausalGraph::larsa_default();
    let total = g.total_effect("HURST_H", "POSITION_SIZE");
    // HURST_H -> SIZE_MODIFIER (positive) -> POSITION_SIZE (positive)
    // Both Direct and Modulated edges are positive
    assert!(total > 0.0, "HURST_H -> POSITION_SIZE should be positive, got {total:.4}");
}

#[test]
fn test_causal_graph_nodes_and_edges_present() {
    let g = CausalGraph::larsa_default();
    assert!(g.node_count() >= 7, "expected >= 7 nodes, got {}", g.node_count());
    assert!(g.edge_count() >= 6, "expected >= 6 edges, got {}", g.edge_count());
}

#[test]
fn test_bh_mass_is_cause_of_outcome() {
    let g = CausalGraph::larsa_default();
    assert!(g.is_cause_of("BH_MASS", "OUTCOME"));
}

#[test]
fn test_outcome_is_not_cause_of_bh_mass() {
    let g = CausalGraph::larsa_default();
    assert!(!g.is_cause_of("OUTCOME", "BH_MASS"));
}

#[test]
fn test_causal_ancestors_of_outcome() {
    let g = CausalGraph::larsa_default();
    let anc = g.ancestors("OUTCOME");
    assert!(anc.contains("BH_MASS"), "BH_MASS should be an ancestor of OUTCOME");
    assert!(anc.contains("HURST_H"), "HURST_H should be an ancestor of OUTCOME");
}

#[test]
fn test_causal_descendants_of_bh_mass() {
    let g = CausalGraph::larsa_default();
    let desc = g.descendants("BH_MASS");
    assert!(desc.contains("ENTRY_SIGNAL"), "ENTRY_SIGNAL should be a descendant of BH_MASS");
    assert!(desc.contains("OUTCOME"), "OUTCOME should be a descendant of BH_MASS");
}

#[test]
fn test_topological_order_valid() {
    let g = CausalGraph::larsa_default();
    let order = g.topological_order();
    assert!(order.is_some(), "LARSA graph should have no cycles");
}

#[test]
fn test_blocking_edge_has_negative_strength() {
    let g = CausalGraph::larsa_default();
    let d = g.direct_effect("NAV_GEODESIC", "ENTRY_GATE");
    assert!(d < 0.0, "Blocking edge should have negative strength, got {d:.4}");
}

#[test]
fn test_path_product_two_hop() {
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::signal("X", ""));
    g.add_node(CausalNode::action("Y", ""));
    g.add_node(CausalNode::outcome("Z", ""));
    g.add_edge(CausalEdge::direct("X", "Y", 0.7)).unwrap();
    g.add_edge(CausalEdge::direct("Y", "Z", 0.5)).unwrap();

    let path = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    let coef = g.compute_path_coefficient(&path);
    assert!((coef - 0.35).abs() < 1e-9, "0.7 * 0.5 = 0.35, got {coef}");
}

#[test]
fn test_total_effect_sums_over_paths() {
    let mut g = CausalGraph::new();
    g.add_node(CausalNode::signal("A", ""));
    g.add_node(CausalNode::action("B", ""));
    g.add_node(CausalNode::action("C", ""));
    g.add_node(CausalNode::outcome("D", ""));
    g.add_edge(CausalEdge::direct("A", "B", 0.6)).unwrap();
    g.add_edge(CausalEdge::direct("B", "D", 0.5)).unwrap();
    g.add_edge(CausalEdge::direct("A", "C", 0.4)).unwrap();
    g.add_edge(CausalEdge::direct("C", "D", 0.8)).unwrap();

    let total = g.total_effect("A", "D");
    // path 1: A->B->D = 0.6 * 0.5 = 0.30
    // path 2: A->C->D = 0.4 * 0.8 = 0.32
    // total = 0.62
    assert!((total - 0.62).abs() < 1e-9, "expected 0.62, got {total}");
}
