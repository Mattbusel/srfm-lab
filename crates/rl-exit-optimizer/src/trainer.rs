use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::agent::{AgentConfig, RLExitAgent};
use crate::environment::{generate_synthetic_trades, run_episode, TradeRecord, TradeEnvironment};
use crate::state::StateVector;

/// Configuration for a training run.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Number of full passes (episodes) through the trade dataset.
    pub num_episodes: usize,
    /// Print stats every N episodes.
    pub log_interval: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Agent hyperparameters.
    pub agent_config: AgentConfig,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig {
            num_episodes: 5_000,
            log_interval: 1_000,
            seed: 42,
            agent_config: AgentConfig::default(),
        }
    }
}

/// Per-episode statistics used to build the training progress report.
#[derive(Debug, Clone, Default)]
struct EpisodeStats {
    episode_return: f64,
    realized_pnl: f64,
    natural_pnl: f64,
    exit_bar: usize,
    total_bars: usize,
    agent_exited_early: bool,
}

/// Summary printed every `log_interval` episodes.
#[derive(Debug, Clone)]
pub struct TrainingSummary {
    pub episode: usize,
    pub mean_return: f64,
    pub mean_pnl: f64,
    pub mean_natural_pnl: f64,
    /// % of winner trades held to full potential (didn't exit early while BH active)
    pub pct_winners_held_full: f64,
    /// % of loser trades cut before bar 20
    pub pct_losers_cut_early: f64,
    pub epsilon: f64,
    pub qtable_states: usize,
}

impl std::fmt::Display for TrainingSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[ep {:>6}] ret={:+.4} pnl={:+.4} nat={:+.4} win_held={:.1}% los_cut={:.1}% eps={:.4} Q_states={}",
            self.episode,
            self.mean_return,
            self.mean_pnl,
            self.mean_natural_pnl,
            self.pct_winners_held_full * 100.0,
            self.pct_losers_cut_early * 100.0,
            self.epsilon,
            self.qtable_states,
        )
    }
}

/// Trains the RL exit agent on a dataset of trade records.
pub struct Trainer {
    pub agent: RLExitAgent,
    pub config: TrainerConfig,
    pub summaries: Vec<TrainingSummary>,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        let agent = RLExitAgent::new(config.agent_config.clone());
        Trainer {
            agent,
            config,
            summaries: Vec::new(),
        }
    }

    /// Run the full training loop on `trades`.
    pub fn train(&mut self, trades: &[TradeRecord]) {
        let mut rng = SmallRng::seed_from_u64(self.config.seed);

        let mut window_stats: Vec<EpisodeStats> = Vec::new();
        let total_trades = trades.len();

        for episode in 0..self.config.num_episodes {
            // Cycle through trades
            let trade = &trades[episode % total_trades];

            let stats = self.run_train_episode(trade, &mut rng);
            window_stats.push(stats);

            // Log every log_interval episodes
            if (episode + 1) % self.config.log_interval == 0 || episode == 0 {
                let summary = self.compute_summary(episode + 1, &window_stats);
                println!("{}", summary);
                self.summaries.push(summary);
                window_stats.clear();
            }
        }
    }

    /// Run a single training episode on `trade`, updating the agent.
    fn run_train_episode(&mut self, trade: &TradeRecord, rng: &mut SmallRng) -> EpisodeStats {
        let mut env = TradeEnvironment::new(trade);
        let mut total_return = 0.0f64;
        let mut discount = 1.0f64;

        while !env.is_done() {
            let state_raw = env.current_state();
            let state_vec = StateVector::from_raw(&state_raw);

            let action = self.agent.select_action(&state_vec, rng);
            let (next_raw, reward, done) = env.step(action);
            let next_vec = StateVector::from_raw(&next_raw);

            self.agent
                .observe(state_vec, action, reward, next_vec, done, rng);

            total_return += discount * reward;
            discount *= 0.95;

            if done {
                break;
            }
        }

        EpisodeStats {
            episode_return: total_return,
            realized_pnl: env.realized_pnl(),
            natural_pnl: trade.natural_pnl_pct,
            exit_bar: env.bar_idx(),
            total_bars: trade.num_bars(),
            agent_exited_early: env.bar_idx() < trade.num_bars() - 1,
        }
    }

    fn compute_summary(&self, episode: usize, window: &[EpisodeStats]) -> TrainingSummary {
        if window.is_empty() {
            return TrainingSummary {
                episode,
                mean_return: 0.0,
                mean_pnl: 0.0,
                mean_natural_pnl: 0.0,
                pct_winners_held_full: 0.0,
                pct_losers_cut_early: 0.0,
                epsilon: self.agent.epsilon,
                qtable_states: self.agent.qtable_size(),
            };
        }

        let n = window.len() as f64;
        let mean_return = window.iter().map(|s| s.episode_return).sum::<f64>() / n;
        let mean_pnl = window.iter().map(|s| s.realized_pnl).sum::<f64>() / n;
        let mean_natural_pnl = window.iter().map(|s| s.natural_pnl).sum::<f64>() / n;

        // Winners: natural_pnl > 0 that agent held to last bar (didn't exit early)
        let winners: Vec<&EpisodeStats> = window
            .iter()
            .filter(|s| s.natural_pnl > 0.0)
            .collect();
        let pct_winners_held_full = if winners.is_empty() {
            0.0
        } else {
            winners.iter().filter(|s| !s.agent_exited_early).count() as f64
                / winners.len() as f64
        };

        // Losers: natural_pnl < -0.02 that agent exited before bar 20
        let losers: Vec<&EpisodeStats> = window
            .iter()
            .filter(|s| s.natural_pnl < -0.02)
            .collect();
        let pct_losers_cut_early = if losers.is_empty() {
            0.0
        } else {
            losers.iter().filter(|s| s.exit_bar <= 20).count() as f64 / losers.len() as f64
        };

        TrainingSummary {
            episode,
            mean_return,
            mean_pnl,
            mean_natural_pnl,
            pct_winners_held_full,
            pct_losers_cut_early,
            epsilon: self.agent.epsilon,
            qtable_states: self.agent.qtable_size(),
        }
    }

    /// Run evaluation episodes (epsilon=0) and return per-trade results.
    pub fn evaluate(&mut self, trades: &[TradeRecord]) -> Vec<crate::environment::EpisodeResult> {
        let saved_eps = self.agent.epsilon;
        self.agent.set_epsilon(0.0);

        let results = trades
            .iter()
            .map(|trade| {
                run_episode(trade, |raw| {
                    let sv = StateVector::from_raw(raw);
                    self.agent.predict(&sv)
                })
            })
            .collect();

        self.agent.set_epsilon(saved_eps);
        results
    }
}

/// Convenience function: train on synthetic data and return the trained agent.
pub fn train_on_synthetic(num_trades: usize, episodes: usize) -> RLExitAgent {
    let trades = generate_synthetic_trades(num_trades, 12345);
    let mut cfg = TrainerConfig::default();
    cfg.num_episodes = episodes;
    cfg.log_interval = episodes.max(1);
    let mut trainer = Trainer::new(cfg);
    trainer.train(&trades);
    trainer.agent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_runs_without_panic() {
        let trades = generate_synthetic_trades(10, 1);
        let mut cfg = TrainerConfig::default();
        cfg.num_episodes = 20;
        cfg.log_interval = 20;
        let mut trainer = Trainer::new(cfg);
        trainer.train(&trades);
    }

    #[test]
    fn test_qtable_grows_during_training() {
        let trades = generate_synthetic_trades(10, 2);
        let mut cfg = TrainerConfig::default();
        cfg.num_episodes = 50;
        cfg.log_interval = 50;
        let mut trainer = Trainer::new(cfg);
        trainer.train(&trades);
        assert!(trainer.agent.qtable_size() > 0);
    }

    #[test]
    fn test_summary_logged() {
        let trades = generate_synthetic_trades(5, 3);
        let mut cfg = TrainerConfig::default();
        cfg.num_episodes = 10;
        cfg.log_interval = 10;
        let mut trainer = Trainer::new(cfg);
        trainer.train(&trades);
        assert!(!trainer.summaries.is_empty());
    }

    #[test]
    fn test_evaluate_returns_results() {
        let trades = generate_synthetic_trades(5, 4);
        let mut cfg = TrainerConfig::default();
        cfg.num_episodes = 10;
        cfg.log_interval = 10;
        let mut trainer = Trainer::new(cfg);
        trainer.train(&trades);
        let results = trainer.evaluate(&trades);
        assert_eq!(results.len(), trades.len());
    }
}
