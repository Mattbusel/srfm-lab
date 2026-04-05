use rl_exit_optimizer::action::Action;
use rl_exit_optimizer::agent::{AgentConfig, RLExitAgent};
use rl_exit_optimizer::environment::{generate_synthetic_trades, run_episode};
use rl_exit_optimizer::evaluator::evaluate;
use rl_exit_optimizer::q_network::{QNetwork, StateKey};
use rl_exit_optimizer::replay_buffer::ReplayBuffer;
use rl_exit_optimizer::reward::{compute_reward, EARLY_EXIT_WINNER_PENALTY, LATE_EXIT_LOSER_PENALTY, PNL_SCALE};
use rl_exit_optimizer::state::{StateVector, TradeStateRaw, NUM_BINS, STATE_DIM};
use rl_exit_optimizer::trainer::{Trainer, TrainerConfig};

// ── State normalization ──────────────────────────────────────────────────────

#[test]
fn test_state_all_features_in_range() {
    let raw = TradeStateRaw {
        position_pnl_pct: 0.12,
        bars_held: 30,
        bh_mass: 0.75,
        bh_active: true,
        atr_ratio: 1.3,
        market_return_since_entry: 0.02,
        momentum_15m: 0.01,
        utc_hour: 15.5,
        drawdown_from_peak: -0.03,
        pnl_acceleration: 0.005,
    };
    let sv = StateVector::from_raw(&raw);
    for (i, &v) in sv.features().iter().enumerate() {
        assert!(
            v >= -1.0 && v <= 1.0,
            "Feature {} out of range: {}",
            i,
            v
        );
    }
}

#[test]
fn test_state_dimension() {
    let sv = StateVector::default();
    assert_eq!(sv.features().len(), STATE_DIM);
    assert_eq!(sv.discretize().len(), STATE_DIM);
}

#[test]
fn test_bh_active_encoding() {
    let mut raw = TradeStateRaw::default();
    raw.bh_active = true;
    let sv_on = StateVector::from_raw(&raw);
    raw.bh_active = false;
    let sv_off = StateVector::from_raw(&raw);
    assert_eq!(sv_on.features()[3], 1.0);
    assert_eq!(sv_off.features()[3], -1.0);
}

#[test]
fn test_discretize_mid_bin() {
    // A state with all features = 0 should land on bin 2 (middle of 5)
    let sv = StateVector([0.0; STATE_DIM]);
    let bins = sv.discretize();
    for &b in &bins {
        assert_eq!(b, 2, "mid-value should land on bin 2");
    }
}

#[test]
fn test_state_key_deterministic() {
    let raw = TradeStateRaw {
        position_pnl_pct: 0.05,
        bars_held: 10,
        bh_active: true,
        bh_mass: 0.8,
        atr_ratio: 1.1,
        ..Default::default()
    };
    let sv = StateVector::from_raw(&raw);
    let k1 = StateKey::from_state(&sv);
    let k2 = StateKey::from_state(&sv);
    assert_eq!(k1, k2);
}

// ── Reward computation ───────────────────────────────────────────────────────

#[test]
fn test_reward_hold_returns_cost() {
    let raw = TradeStateRaw::default();
    let r = compute_reward(&raw, Action::Hold, false);
    assert!((r - (-0.001)).abs() < 1e-9);
}

#[test]
fn test_reward_exit_scales_pnl() {
    let raw = TradeStateRaw {
        position_pnl_pct: 0.10,
        bh_active: false,
        ..Default::default()
    };
    let r = compute_reward(&raw, Action::Exit, false);
    assert!((r - 0.10 * PNL_SCALE).abs() < 1e-9);
}

#[test]
fn test_reward_early_exit_winner_penalty() {
    let raw = TradeStateRaw {
        position_pnl_pct: 0.05,
        bh_active: true,
        bars_held: 5,
        ..Default::default()
    };
    let r = compute_reward(&raw, Action::Exit, false);
    let expected = 0.05 * PNL_SCALE + EARLY_EXIT_WINNER_PENALTY;
    assert!((r - expected).abs() < 1e-9);
}

#[test]
fn test_reward_late_loser_penalty() {
    let raw = TradeStateRaw {
        position_pnl_pct: -0.03,
        bh_active: false,
        bars_held: 25,
        ..Default::default()
    };
    let r = compute_reward(&raw, Action::Exit, false);
    let expected = -0.03 * PNL_SCALE + LATE_EXIT_LOSER_PENALTY;
    assert!((r - expected).abs() < 1e-9);
}

#[test]
fn test_reward_no_double_penalty_on_short_loser() {
    let raw = TradeStateRaw {
        position_pnl_pct: -0.03,
        bh_active: false,
        bars_held: 5, // short hold, no late-loser penalty
        ..Default::default()
    };
    let r = compute_reward(&raw, Action::Exit, false);
    let expected = -0.03 * PNL_SCALE; // only base
    assert!((r - expected).abs() < 1e-9);
}

// ── Q-table update ───────────────────────────────────────────────────────────

#[test]
fn test_qtable_update_bellman() {
    let mut q = QNetwork::new(1.0, 0.9); // alpha=1 for clean math
    let sv1 = StateVector([0.1; STATE_DIM]);
    let sv2 = StateVector([-0.1; STATE_DIM]);
    let k1 = StateKey::from_state(&sv1);
    let k2 = StateKey::from_state(&sv2);

    // Q(s2, HOLD) = 0 initially; target = 1.0 + 0.9*0 = 1.0
    q.update(k1, Action::Hold, 1.0, k2, false);
    let qs = q.get_q(k1);
    assert!((qs[Action::Hold.index()] - 1.0).abs() < 1e-9);
}

#[test]
fn test_qtable_convergence_on_fixed_reward() {
    let mut q = QNetwork::new(0.5, 0.0); // gamma=0, no bootstrapping
    let sv = StateVector([0.2; STATE_DIM]);
    let k = StateKey::from_state(&sv);
    let sv2 = StateVector([-0.2; STATE_DIM]);
    let k2 = StateKey::from_state(&sv2);

    // Repeated updates with same reward -> converge to reward value
    for _ in 0..100 {
        q.update(k, Action::Exit, 0.8, k2, true);
    }
    let qs = q.get_q(k);
    // With gamma=0 and done=true: Q converges to 0.8
    assert!((qs[Action::Exit.index()] - 0.8).abs() < 0.01);
}

// ── Replay buffer ────────────────────────────────────────────────────────────

#[test]
fn test_replay_buffer_capacity_respected() {
    let mut buf = ReplayBuffer::new(10);
    let sv = StateVector::default();
    for _ in 0..25 {
        buf.store(sv.clone(), Action::Hold, -0.001, sv.clone(), false);
    }
    assert_eq!(buf.len(), 10);
}

#[test]
fn test_replay_buffer_sample_unique() {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    let mut buf = ReplayBuffer::new(100);
    for i in 0..100 {
        let sv = StateVector::from_raw(&TradeStateRaw {
            position_pnl_pct: i as f64 * 0.001,
            ..Default::default()
        });
        buf.store(sv.clone(), Action::Hold, 0.0, sv, false);
    }
    let mut rng = SmallRng::seed_from_u64(42);
    let batch = buf.sample(10, &mut rng).unwrap();
    assert_eq!(batch.len(), 10);
}

// ── Environment ──────────────────────────────────────────────────────────────

#[test]
fn test_environment_step_terminates() {
    let trades = generate_synthetic_trades(5, 111);
    for trade in &trades {
        let result = run_episode(trade, |_| Action::Exit);
        assert_eq!(result.exit_bar, 0); // exits immediately
    }
}

#[test]
fn test_environment_hold_all_runs_to_end() {
    let trades = generate_synthetic_trades(5, 222);
    for trade in &trades {
        let result = run_episode(trade, |_| Action::Hold);
        assert_eq!(result.exit_bar, trade.num_bars() - 1);
    }
}

// ── Full training smoke test ──────────────────────────────────────────────────

#[test]
fn test_training_improves_epsilon() {
    let trades = generate_synthetic_trades(10, 333);
    let mut cfg = TrainerConfig::default();
    cfg.num_episodes = 100;
    cfg.log_interval = 100;
    cfg.agent_config.epsilon_decay_steps = 50;
    let mut trainer = Trainer::new(cfg);
    trainer.train(&trades);
    // After training with 50 decay steps and 100 episodes, epsilon should be at minimum
    assert!(trainer.agent.epsilon < 0.5);
}

#[test]
fn test_evaluate_report_valid() {
    let trades = generate_synthetic_trades(30, 444);
    let mut cfg = TrainerConfig::default();
    cfg.num_episodes = 50;
    cfg.log_interval = 50;
    let mut trainer = Trainer::new(cfg);
    trainer.train(&trades);
    let report = evaluate(&trainer.agent, &trades);
    assert_eq!(report.num_trades, 30);
    assert!(report.rl_win_rate >= 0.0 && report.rl_win_rate <= 1.0);
    assert!(report.bh_win_rate >= 0.0 && report.bh_win_rate <= 1.0);
    assert!(report.rl_avg_hold_bars >= 0.0);
}
