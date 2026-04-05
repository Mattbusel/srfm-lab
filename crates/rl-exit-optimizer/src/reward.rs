use crate::action::Action;
use crate::state::TradeStateRaw;

/// Per-bar holding cost (opportunity cost of tying up capital).
pub const HOLD_BASE_COST: f64 = -0.001;

/// Penalty for exiting a winner while BH signal is still active.
pub const EARLY_EXIT_WINNER_PENALTY: f64 = -0.5;

/// Penalty for staying in a losing trade too long (pnl < -2%, bars > 20).
pub const LATE_EXIT_LOSER_PENALTY: f64 = -1.0;

/// Scale factor to turn pnl fractions into reward units.
pub const PNL_SCALE: f64 = 10.0;

/// Threshold below which we consider the position a "loser".
pub const LOSER_PNL_THRESHOLD: f64 = -0.02;

/// Number of bars above which staying in a loser is penalized.
pub const LOSER_DURATION_THRESHOLD: u32 = 20;

/// Compute the scalar reward for a single (state, action) transition.
///
/// # Arguments
/// * `state`   – raw (un-normalized) trade state *before* the action
/// * `action`  – action chosen by the agent
/// * `terminal`– true when this bar is the last in the episode (forced exit)
pub fn compute_reward(state: &TradeStateRaw, action: Action, terminal: bool) -> f64 {
    match action {
        Action::Hold => {
            if terminal {
                // Forced exit at episode end: realize P&L, same as EXIT path
                realize_reward(state)
            } else {
                // Small negative reward for each bar we stay in
                HOLD_BASE_COST
            }
        }
        Action::Exit => realize_reward(state),
    }
}

/// Compute the reward for realizing the current P&L.
fn realize_reward(state: &TradeStateRaw) -> f64 {
    let pnl = state.position_pnl_pct;
    let mut reward = pnl * PNL_SCALE;

    // Penalty: exiting a winner early (positive P&L, BH still active)
    if pnl > 0.0 && state.bh_active {
        reward += EARLY_EXIT_WINNER_PENALTY;
    }

    // Penalty: exiting a loser after holding too long
    if pnl < LOSER_PNL_THRESHOLD && state.bars_held > LOSER_DURATION_THRESHOLD {
        reward += LATE_EXIT_LOSER_PENALTY;
    }

    reward
}

/// Reward summary returned alongside an action for logging/training.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RewardBreakdown {
    pub base_reward: f64,
    pub early_exit_penalty: f64,
    pub late_loser_penalty: f64,
    pub total: f64,
}

/// Compute reward and return a detailed breakdown.
pub fn compute_reward_detailed(
    state: &TradeStateRaw,
    action: Action,
    terminal: bool,
) -> RewardBreakdown {
    if action == Action::Hold && !terminal {
        return RewardBreakdown {
            base_reward: HOLD_BASE_COST,
            early_exit_penalty: 0.0,
            late_loser_penalty: 0.0,
            total: HOLD_BASE_COST,
        };
    }

    let pnl = state.position_pnl_pct;
    let base_reward = pnl * PNL_SCALE;

    let early_exit_penalty = if pnl > 0.0 && state.bh_active {
        EARLY_EXIT_WINNER_PENALTY
    } else {
        0.0
    };

    let late_loser_penalty =
        if pnl < LOSER_PNL_THRESHOLD && state.bars_held > LOSER_DURATION_THRESHOLD {
            LATE_EXIT_LOSER_PENALTY
        } else {
            0.0
        };

    let total = base_reward + early_exit_penalty + late_loser_penalty;

    RewardBreakdown {
        base_reward,
        early_exit_penalty,
        late_loser_penalty,
        total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TradeStateRaw;

    fn base_state() -> TradeStateRaw {
        TradeStateRaw {
            position_pnl_pct: 0.0,
            bars_held: 5,
            bh_mass: 0.6,
            bh_active: true,
            atr_ratio: 1.0,
            market_return_since_entry: 0.0,
            momentum_15m: 0.0,
            utc_hour: 14.0,
            drawdown_from_peak: 0.0,
            pnl_acceleration: 0.0,
        }
    }

    #[test]
    fn test_hold_cost() {
        let s = base_state();
        let r = compute_reward(&s, Action::Hold, false);
        assert!((r - HOLD_BASE_COST).abs() < 1e-9);
    }

    #[test]
    fn test_exit_flat_position() {
        let s = base_state(); // pnl=0, bh_active=true
        let r = compute_reward(&s, Action::Exit, false);
        // base = 0*10 = 0, early_exit_penalty = -0.5 (winner threshold is > 0, 0 is not > 0)
        // pnl=0 is not > 0, so no early exit penalty
        assert!((r - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_exit_winner_early_penalty() {
        let mut s = base_state();
        s.position_pnl_pct = 0.05; // 5% winner
        s.bh_active = true;
        let r = compute_reward(&s, Action::Exit, false);
        let expected = 0.05 * PNL_SCALE + EARLY_EXIT_WINNER_PENALTY;
        assert!((r - expected).abs() < 1e-9);
    }

    #[test]
    fn test_exit_winner_bh_dead_no_penalty() {
        let mut s = base_state();
        s.position_pnl_pct = 0.05;
        s.bh_active = false; // BH dead, exit is fine
        let r = compute_reward(&s, Action::Exit, false);
        let expected = 0.05 * PNL_SCALE; // no penalty
        assert!((r - expected).abs() < 1e-9);
    }

    #[test]
    fn test_exit_loser_late_penalty() {
        let mut s = base_state();
        s.position_pnl_pct = -0.05; // 5% loser
        s.bars_held = 25; // > 20
        s.bh_active = false;
        let r = compute_reward(&s, Action::Exit, false);
        let expected = -0.05 * PNL_SCALE + LATE_EXIT_LOSER_PENALTY;
        assert!((r - expected).abs() < 1e-9);
    }

    #[test]
    fn test_exit_loser_early_no_late_penalty() {
        let mut s = base_state();
        s.position_pnl_pct = -0.05;
        s.bars_held = 10; // <= 20, cutting loss quickly
        s.bh_active = false;
        let r = compute_reward(&s, Action::Exit, false);
        let expected = -0.05 * PNL_SCALE; // no late-loser penalty
        assert!((r - expected).abs() < 1e-9);
    }

    #[test]
    fn test_terminal_forced_exit() {
        let mut s = base_state();
        s.position_pnl_pct = 0.03;
        s.bh_active = true;
        // Terminal hold should behave like EXIT (forced close)
        let r_terminal = compute_reward(&s, Action::Hold, true);
        let r_exit = compute_reward(&s, Action::Exit, false);
        assert!((r_terminal - r_exit).abs() < 1e-9);
    }

    #[test]
    fn test_detailed_breakdown_consistency() {
        let mut s = base_state();
        s.position_pnl_pct = 0.04;
        s.bh_active = true;
        let bd = compute_reward_detailed(&s, Action::Exit, false);
        assert!((bd.total - (bd.base_reward + bd.early_exit_penalty + bd.late_loser_penalty)).abs() < 1e-9);
        let simple = compute_reward(&s, Action::Exit, false);
        assert!((bd.total - simple).abs() < 1e-9);
    }
}
