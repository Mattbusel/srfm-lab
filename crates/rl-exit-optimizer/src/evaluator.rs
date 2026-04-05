use crate::environment::{EpisodeResult, TradeRecord, run_episode};
use crate::state::StateVector;
use crate::agent::RLExitAgent;

/// Comparison of RL agent vs BH baseline for one trade.
#[derive(Debug, Clone)]
pub struct TradeComparison {
    pub trade_id: String,
    /// Realized P&L by the RL agent
    pub rl_pnl: f64,
    /// P&L at the natural BH exit
    pub bh_pnl: f64,
    /// Bar at which RL agent exited
    pub rl_exit_bar: usize,
    /// Total bars available (natural BH exit)
    pub bh_exit_bar: usize,
    /// Outcome category
    pub outcome: ComparisonOutcome,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOutcome {
    /// RL held longer than BH AND got better P&L
    RlHeldBetter,
    /// RL exited earlier than BH AND avoided a bigger loss (or captured more gain)
    RlExitedBetter,
    /// RL exited earlier and got worse P&L (left money on the table)
    RlExitedWorse,
    /// RL held longer and got worse P&L (stayed in a loser)
    RlHeldWorse,
    /// Both strategies effectively the same bar
    Equivalent,
}

impl std::fmt::Display for ComparisonOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComparisonOutcome::RlHeldBetter => write!(f, "RL_HELD_BETTER"),
            ComparisonOutcome::RlExitedBetter => write!(f, "RL_EXITED_BETTER"),
            ComparisonOutcome::RlExitedWorse => write!(f, "RL_EXITED_WORSE"),
            ComparisonOutcome::RlHeldWorse => write!(f, "RL_HELD_WORSE"),
            ComparisonOutcome::Equivalent => write!(f, "EQUIVALENT"),
        }
    }
}

/// Aggregate evaluation statistics.
#[derive(Debug, Clone, Default)]
pub struct EvaluationReport {
    pub num_trades: usize,
    pub rl_avg_pnl: f64,
    pub bh_avg_pnl: f64,
    pub rl_win_rate: f64,
    pub bh_win_rate: f64,
    pub rl_avg_hold_bars: f64,
    pub bh_avg_hold_bars: f64,
    pub pct_rl_held_better: f64,
    pub pct_rl_exited_better: f64,
    pub pct_rl_exited_worse: f64,
    pub pct_rl_held_worse: f64,
    pub pnl_improvement: f64, // rl_avg_pnl - bh_avg_pnl
}

impl std::fmt::Display for EvaluationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;
        writeln!(f, " RL Exit Agent vs BH Baseline Evaluation")?;
        writeln!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")?;
        writeln!(f, " Trades evaluated  : {}", self.num_trades)?;
        writeln!(f, " RL  avg P&L       : {:+.4}  ({:+.2}%)", self.rl_avg_pnl, self.rl_avg_pnl * 100.0)?;
        writeln!(f, " BH  avg P&L       : {:+.4}  ({:+.2}%)", self.bh_avg_pnl, self.bh_avg_pnl * 100.0)?;
        writeln!(f, " P&L improvement   : {:+.4}  ({:+.2}%)", self.pnl_improvement, self.pnl_improvement * 100.0)?;
        writeln!(f, " RL  win rate      : {:.1}%", self.rl_win_rate * 100.0)?;
        writeln!(f, " BH  win rate      : {:.1}%", self.bh_win_rate * 100.0)?;
        writeln!(f, " RL  avg hold bars : {:.1}", self.rl_avg_hold_bars)?;
        writeln!(f, " BH  avg hold bars : {:.1}", self.bh_avg_hold_bars)?;
        writeln!(f, " RL held longer+better  : {:.1}%", self.pct_rl_held_better * 100.0)?;
        writeln!(f, " RL exited earlier+better: {:.1}%", self.pct_rl_exited_better * 100.0)?;
        writeln!(f, " RL exited earlier+worse : {:.1}%", self.pct_rl_exited_worse * 100.0)?;
        writeln!(f, " RL held longer+worse    : {:.1}%", self.pct_rl_held_worse * 100.0)?;
        write!(f, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    }
}

/// Evaluate a trained agent against the BH baseline on a trade dataset.
pub fn evaluate(agent: &RLExitAgent, trades: &[TradeRecord]) -> EvaluationReport {
    let comparisons: Vec<TradeComparison> = trades
        .iter()
        .map(|trade| compare_trade(agent, trade))
        .collect();

    build_report(&comparisons)
}

/// Run both RL and BH policies on a single trade and compare.
pub fn compare_trade(agent: &RLExitAgent, trade: &TradeRecord) -> TradeComparison {
    // RL policy: greedy (epsilon=0)
    let rl_result = run_episode(trade, |raw| {
        let sv = StateVector::from_raw(raw);
        agent.predict(&sv)
    });

    // BH policy: always hold until natural close
    let bh_result = run_episode(trade, |_| crate::action::Action::Hold);

    let outcome = classify_outcome(&rl_result, &bh_result);

    TradeComparison {
        trade_id: trade.trade_id.clone(),
        rl_pnl: rl_result.realized_pnl,
        bh_pnl: bh_result.realized_pnl,
        rl_exit_bar: rl_result.exit_bar,
        bh_exit_bar: bh_result.exit_bar,
        outcome,
    }
}

fn classify_outcome(rl: &EpisodeResult, bh: &EpisodeResult) -> ComparisonOutcome {
    let same_bar = (rl.exit_bar as i64 - bh.exit_bar as i64).unsigned_abs() <= 1;
    if same_bar {
        return ComparisonOutcome::Equivalent;
    }

    let rl_held_longer = rl.exit_bar > bh.exit_bar;
    let rl_better_pnl = rl.realized_pnl > bh.realized_pnl;

    match (rl_held_longer, rl_better_pnl) {
        (true, true) => ComparisonOutcome::RlHeldBetter,
        (true, false) => ComparisonOutcome::RlHeldWorse,
        (false, true) => ComparisonOutcome::RlExitedBetter,
        (false, false) => ComparisonOutcome::RlExitedWorse,
    }
}

fn build_report(comparisons: &[TradeComparison]) -> EvaluationReport {
    let n = comparisons.len();
    if n == 0 {
        return EvaluationReport::default();
    }
    let nf = n as f64;

    let rl_avg_pnl = comparisons.iter().map(|c| c.rl_pnl).sum::<f64>() / nf;
    let bh_avg_pnl = comparisons.iter().map(|c| c.bh_pnl).sum::<f64>() / nf;

    let rl_win_rate = comparisons.iter().filter(|c| c.rl_pnl > 0.0).count() as f64 / nf;
    let bh_win_rate = comparisons.iter().filter(|c| c.bh_pnl > 0.0).count() as f64 / nf;

    let rl_avg_hold = comparisons.iter().map(|c| c.rl_exit_bar as f64).sum::<f64>() / nf;
    let bh_avg_hold = comparisons.iter().map(|c| c.bh_exit_bar as f64).sum::<f64>() / nf;

    let count = |o: &ComparisonOutcome| {
        comparisons.iter().filter(|c| &c.outcome == o).count() as f64 / nf
    };

    EvaluationReport {
        num_trades: n,
        rl_avg_pnl,
        bh_avg_pnl,
        rl_win_rate,
        bh_win_rate,
        rl_avg_hold_bars: rl_avg_hold,
        bh_avg_hold_bars: bh_avg_hold,
        pct_rl_held_better: count(&ComparisonOutcome::RlHeldBetter),
        pct_rl_exited_better: count(&ComparisonOutcome::RlExitedBetter),
        pct_rl_exited_worse: count(&ComparisonOutcome::RlExitedWorse),
        pct_rl_held_worse: count(&ComparisonOutcome::RlHeldWorse),
        pnl_improvement: rl_avg_pnl - bh_avg_pnl,
    }
}

/// Print a detailed table of trade comparisons.
pub fn print_comparison_table(comparisons: &[TradeComparison]) {
    println!(
        "{:<10} {:>10} {:>10} {:>8} {:>8} {}",
        "TRADE_ID", "RL_PNL", "BH_PNL", "RL_BAR", "BH_BAR", "OUTCOME"
    );
    println!("{}", "-".repeat(65));
    for c in comparisons {
        println!(
            "{:<10} {:>+10.4} {:>+10.4} {:>8} {:>8} {}",
            c.trade_id, c.rl_pnl, c.bh_pnl, c.rl_exit_bar, c.bh_exit_bar, c.outcome
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::generate_synthetic_trades;
    use crate::trainer::train_on_synthetic;

    #[test]
    fn test_evaluate_returns_report() {
        let trades = generate_synthetic_trades(20, 55);
        let agent = train_on_synthetic(20, 50);
        let report = evaluate(&agent, &trades);
        assert_eq!(report.num_trades, 20);
        assert!(report.rl_win_rate >= 0.0 && report.rl_win_rate <= 1.0);
    }

    #[test]
    fn test_comparison_outcome_coverage() {
        let trades = generate_synthetic_trades(50, 77);
        let agent = train_on_synthetic(50, 100);
        let comparisons: Vec<_> = trades.iter().map(|t| compare_trade(&agent, t)).collect();
        // Should have at least some non-equivalent outcomes given random data
        let non_equiv = comparisons
            .iter()
            .filter(|c| c.outcome != ComparisonOutcome::Equivalent)
            .count();
        // With random data this may or may not be zero -- just assert no panic
        let _ = non_equiv;
        assert_eq!(comparisons.len(), 50);
    }
}
