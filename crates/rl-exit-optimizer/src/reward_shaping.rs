// reward_shaping.rs -- Reward function design for the SRFM exit policy.
//
// Three reward components:
//   1. Action-conditioned shaped reward (HOLD / PARTIAL_EXIT / FULL_EXIT)
//   2. Potential-based shaping: phi(s) = signal_strength
//   3. Sparse-to-dense conversion via exponential backward assignment

// ---------------------------------------------------------------------------
// RewardConfig
// ---------------------------------------------------------------------------

/// Hyperparameters controlling the reward function shape.
#[derive(Debug, Clone)]
pub struct RewardConfig {
    /// Scale applied to raw PnL before computing rewards.
    pub base_pnl_scale: f64,
    /// Per-bar cost multiplied by bars_held for HOLD actions.
    pub hold_penalty: f64,
    /// Bonus added when taking PARTIAL_EXIT (reward decisive action).
    pub partial_exit_bonus: f64,
    /// Per-unit penalty applied when drawdown exceeds 2%.
    pub drawdown_penalty: f64,
    /// Bonus scaling factor for FULL_EXIT based on time held.
    pub time_bonus: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            base_pnl_scale:    10.0,
            hold_penalty:      1e-3,
            partial_exit_bonus: 0.05,
            drawdown_penalty:  2.0,
            time_bonus:        0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Action constants
// ---------------------------------------------------------------------------

/// Hold the position unchanged.
pub const ACTION_HOLD:         u8 = 0;
/// Exit half the position.
pub const ACTION_PARTIAL_EXIT: u8 = 1;
/// Exit the full position.
pub const ACTION_FULL_EXIT:    u8 = 2;

/// Drawdown level above which the drawdown penalty kicks in.
const DD_THRESHOLD: f64 = 0.02;

// ---------------------------------------------------------------------------
// RewardCalculator
// ---------------------------------------------------------------------------

/// Computes per-step shaped rewards for the exit policy.
pub struct RewardCalculator {
    pub config: RewardConfig,
}

impl RewardCalculator {
    /// Construct with custom config.
    pub fn new(config: RewardConfig) -> Self {
        RewardCalculator { config }
    }

    /// Construct with default config.
    pub fn default_config() -> Self {
        RewardCalculator { config: RewardConfig::default() }
    }

    /// Compute the shaped reward for one (state, action) step.
    ///
    /// Arguments:
    ///   action         -- 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT
    ///   pnl_change     -- change in unrealized PnL this bar (fraction)
    ///   current_dd     -- current drawdown from peak (positive fraction, e.g. 0.03 = 3%)
    ///   bars_held      -- number of bars in the position so far
    ///   signal_strength-- normalized signal strength in [0, 1]
    pub fn calculate(
        &self,
        action:          u8,
        pnl_change:      f64,
        current_dd:      f64,
        bars_held:       u32,
        _signal_strength: f64,
    ) -> f64 {
        let scaled_pnl = pnl_change * self.config.base_pnl_scale;

        match action {
            ACTION_HOLD => {
                // Reward the PnL change but penalise time cost and excess drawdown.
                let time_cost = self.config.hold_penalty * bars_held as f64;
                let dd_excess = (current_dd - DD_THRESHOLD).max(0.0);
                let dd_cost   = self.config.drawdown_penalty * dd_excess;
                scaled_pnl - time_cost - dd_cost
            }

            ACTION_PARTIAL_EXIT => {
                // Reward decisive action: realised PnL change + flat bonus.
                scaled_pnl + self.config.partial_exit_bonus
            }

            ACTION_FULL_EXIT => {
                // Reward with a time-based bonus that encourages quicker exits.
                // time_factor in [0, 1]: saturates at 5 bars.
                let time_factor = (bars_held as f64 / 5.0).min(1.0);
                scaled_pnl + self.config.time_bonus * time_factor
            }

            _ => scaled_pnl, // unknown action: bare PnL
        }
    }
}

// ---------------------------------------------------------------------------
// Potential-based reward shaping
// ---------------------------------------------------------------------------

/// Compute the potential-based shaped reward.
///
/// Formula: shaped = raw_reward + gamma * phi(s') - phi(s)
///
/// Here phi(s) = signal_strength, so the agent is encouraged to move toward
/// states with higher signal strength.
///
/// Arguments:
///   raw_reward    -- unshaped reward r_t
///   phi_s         -- potential of the current state phi(s)
///   phi_s_next    -- potential of the next state phi(s')
///   gamma         -- discount factor
pub fn shaped_reward(
    raw_reward: f64,
    phi_s:      f64,
    phi_s_next: f64,
    gamma:      f64,
) -> f64 {
    raw_reward + gamma * phi_s_next - phi_s
}

/// Compute the potential phi(s) = signal_strength for a given state.
///
/// Convenience wrapper so callers do not have to extract signal_strength manually.
#[inline]
pub fn potential(signal_strength: f64) -> f64 {
    signal_strength
}

// ---------------------------------------------------------------------------
// Sparse-to-dense reward conversion
// ---------------------------------------------------------------------------

/// Convert a terminal (sparse) reward to per-step dense rewards.
///
/// Uses exponential backward assignment: each step t receives a fraction of
/// the terminal reward discounted from the end of the episode:
///
///   r_t = terminal_reward * gamma^{T - t - 1}   for t in 0..T
///
/// The final step (t = T-1) receives terminal_reward * gamma^0 = terminal_reward.
///
/// Returns a Vec of length `bars_held` where index 0 is the first bar.
///
/// Arguments:
///   terminal_reward -- scalar reward realized at the end of the episode
///   bars_held       -- total episode length
///   gamma           -- discount factor applied per bar from the end
pub fn sparse_to_dense(terminal_reward: f64, bars_held: u32, gamma: f64) -> Vec<f64> {
    let n = bars_held as usize;
    if n == 0 {
        return Vec::new();
    }

    let mut rewards = Vec::with_capacity(n);
    // Work from the last bar backward.
    let mut discount = 1.0_f64;
    for _ in 0..n {
        rewards.push(terminal_reward * discount);
        discount *= gamma;
    }

    // Reverse so index 0 = first bar (smallest discount).
    rewards.reverse();
    rewards
}

/// Like `sparse_to_dense` but adds a small per-step base reward to prevent
/// the agent from receiving zero gradient on intermediate steps.
pub fn sparse_to_dense_with_base(
    terminal_reward: f64,
    bars_held:       u32,
    gamma:           f64,
    base_per_step:   f64,
) -> Vec<f64> {
    let mut dense = sparse_to_dense(terminal_reward, bars_held, gamma);
    for r in dense.iter_mut() {
        *r += base_per_step;
    }
    dense
}

// ---------------------------------------------------------------------------
// Advantage estimator (GAE-lambda, useful for actor-critic variants)
// ---------------------------------------------------------------------------

/// Compute Generalized Advantage Estimates (GAE-lambda) from a trajectory.
///
/// delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
/// A_t     = delta_t + (gamma * lambda) * delta_{t+1} + ...
///
/// Arguments:
///   rewards    -- slice of per-step rewards
///   values     -- slice of V(s_t) estimates (length == rewards.len())
///   next_value -- V(s_{T+1}), typically 0 if terminal
///   gamma      -- discount factor
///   lam        -- lambda for GAE smoothing (0 = TD, 1 = Monte Carlo)
pub fn gae_advantages(
    rewards:    &[f64],
    values:     &[f64],
    next_value: f64,
    gamma:      f64,
    lam:        f64,
) -> Vec<f64> {
    assert_eq!(rewards.len(), values.len(), "rewards and values must have equal length");
    let n = rewards.len();
    let mut advantages = vec![0.0_f64; n];
    let mut gae = 0.0_f64;

    for t in (0..n).rev() {
        let v_next = if t + 1 < n { values[t + 1] } else { next_value };
        let delta = rewards[t] + gamma * v_next - values[t];
        gae = delta + gamma * lam * gae;
        advantages[t] = gae;
    }

    advantages
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn calc() -> RewardCalculator {
        RewardCalculator::default_config()
    }

    // -- RewardCalculator tests ----------------------------------------------

    #[test]
    fn test_hold_has_time_penalty() {
        let rc = calc();
        // Zero PnL change, no drawdown, bars_held=10 -> penalty = hold_penalty * 10
        let r = rc.calculate(ACTION_HOLD, 0.0, 0.0, 10, 0.5);
        let expected = -(rc.config.hold_penalty * 10.0);
        assert!((r - expected).abs() < 1e-9, "got {}", r);
    }

    #[test]
    fn test_hold_excess_drawdown_penalty() {
        let rc = calc();
        // drawdown = 5% > 2%, excess = 3%, drawdown_penalty * 0.03
        let r = rc.calculate(ACTION_HOLD, 0.0, 0.05, 1, 0.5);
        let time_cost = rc.config.hold_penalty * 1.0;
        let dd_cost   = rc.config.drawdown_penalty * 0.03;
        let expected = -(time_cost + dd_cost);
        assert!((r - expected).abs() < 1e-9, "got {}", r);
    }

    #[test]
    fn test_hold_drawdown_below_threshold_no_penalty() {
        let rc = calc();
        // drawdown = 1% < 2%: no drawdown penalty.
        let r = rc.calculate(ACTION_HOLD, 0.0, 0.01, 1, 0.5);
        let expected = -(rc.config.hold_penalty * 1.0);
        assert!((r - expected).abs() < 1e-9, "got {}", r);
    }

    #[test]
    fn test_partial_exit_includes_bonus() {
        let rc = calc();
        let pnl_change = 0.02; // 2% pnl change
        let r = rc.calculate(ACTION_PARTIAL_EXIT, pnl_change, 0.0, 5, 0.5);
        let expected = pnl_change * rc.config.base_pnl_scale + rc.config.partial_exit_bonus;
        assert!((r - expected).abs() < 1e-9, "got {}", r);
    }

    #[test]
    fn test_full_exit_time_bonus_saturates() {
        let rc = calc();
        // bars_held >= 5 -> time_factor = 1.0
        let r5  = rc.calculate(ACTION_FULL_EXIT, 0.0, 0.0, 5, 0.5);
        let r10 = rc.calculate(ACTION_FULL_EXIT, 0.0, 0.0, 10, 0.5);
        assert!((r5 - r10).abs() < 1e-9, "should saturate: r5={} r10={}", r5, r10);
        assert!((r5 - rc.config.time_bonus).abs() < 1e-9);
    }

    #[test]
    fn test_full_exit_time_bonus_scales() {
        let rc = calc();
        // bars_held=1 -> time_factor = 0.2, bars_held=5 -> 1.0
        let r1 = rc.calculate(ACTION_FULL_EXIT, 0.0, 0.0, 1, 0.5);
        let r5 = rc.calculate(ACTION_FULL_EXIT, 0.0, 0.0, 5, 0.5);
        assert!(r1 < r5, "longer hold should get more time bonus");
    }

    // -- potential-based shaping tests ---------------------------------------

    #[test]
    fn test_shaped_reward_identity_at_same_potential() {
        // If phi(s) == phi(s'), shaping term = gamma * phi - phi = (gamma - 1) * phi.
        let phi = 0.7;
        let gamma = 0.99;
        let shaped = shaped_reward(1.0, phi, phi, gamma);
        let expected = 1.0 + (gamma - 1.0) * phi;
        assert!((shaped - expected).abs() < 1e-9);
    }

    #[test]
    fn test_shaped_reward_increasing_potential_adds() {
        // Moving to higher potential should increase the shaped reward.
        let r = shaped_reward(0.0, 0.2, 0.8, 0.99);
        // 0 + 0.99 * 0.8 - 0.2 = 0.592
        assert!(r > 0.0, "expected positive shaped reward, got {}", r);
    }

    // -- sparse_to_dense tests -----------------------------------------------

    #[test]
    fn test_sparse_to_dense_length() {
        let dense = sparse_to_dense(1.0, 5, 0.99);
        assert_eq!(dense.len(), 5);
    }

    #[test]
    fn test_sparse_to_dense_last_element_is_terminal() {
        let dense = sparse_to_dense(2.0, 4, 0.9);
        // Last element should be 2.0 * gamma^0 = 2.0
        assert!((dense[3] - 2.0).abs() < 1e-9, "last={}", dense[3]);
    }

    #[test]
    fn test_sparse_to_dense_first_element_most_discounted() {
        let dense = sparse_to_dense(1.0, 3, 0.5);
        // Expected: [0.25, 0.5, 1.0] (reversed exponential decay)
        assert!((dense[0] - 0.25).abs() < 1e-9);
        assert!((dense[1] - 0.50).abs() < 1e-9);
        assert!((dense[2] - 1.00).abs() < 1e-9);
    }

    #[test]
    fn test_sparse_to_dense_zero_bars() {
        let dense = sparse_to_dense(1.0, 0, 0.99);
        assert!(dense.is_empty());
    }

    // -- GAE tests -----------------------------------------------------------

    #[test]
    fn test_gae_td_lambda_zero_equals_td() {
        // lambda=0: GAE should equal one-step TD advantage.
        let rewards = vec![1.0, 0.0, -1.0];
        let values  = vec![0.5, 0.5,  0.5];
        let gamma = 0.99;
        let adv = gae_advantages(&rewards, &values, 0.0, gamma, 0.0);
        // A_t = r_t + gamma * V(s_{t+1}) - V(s_t) (since lambda=0, no lookahead)
        let a0_expected = 1.0 + gamma * values[1] - values[0];
        assert!((adv[0] - a0_expected).abs() < 1e-9);
    }

    #[test]
    fn test_gae_lambda_one_equals_monte_carlo() {
        // lambda=1: GAE computes full Monte Carlo return minus baseline.
        let rewards = vec![1.0, 1.0, 1.0];
        let values  = vec![0.0, 0.0, 0.0];
        let gamma = 1.0;
        let adv = gae_advantages(&rewards, &values, 0.0, gamma, 1.0);
        // With all-zero values and gamma=1: A_0 = 3, A_1 = 2, A_2 = 1
        assert!((adv[0] - 3.0).abs() < 1e-9);
        assert!((adv[1] - 2.0).abs() < 1e-9);
        assert!((adv[2] - 1.0).abs() < 1e-9);
    }
}
