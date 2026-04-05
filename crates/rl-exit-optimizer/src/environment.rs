use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::reward::compute_reward;
use crate::state::{StateVector, TradeStateRaw};

/// One bar of OHLCV data plus derived features used to build state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub timestamp: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    /// ATR at this bar
    pub atr: f64,
    /// Market index close (e.g. SPY) for market_return calculation
    pub market_close: f64,
}

/// A historical trade record loaded from CSV.
/// Contains the full sequence of bars from entry to natural close.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub trade_id: String,
    pub symbol: String,
    pub entry_price: f64,
    pub entry_atr: f64,
    pub entry_market_close: f64,
    /// BH mass at each bar (same length as bars)
    pub bh_masses: Vec<f64>,
    /// Whether BH signal was active at each bar
    pub bh_actives: Vec<bool>,
    /// Bars from entry onwards
    pub bars: Vec<Bar>,
    /// Natural exit price (BH dies or position closed by system)
    pub natural_exit_price: f64,
    /// P&L at the natural exit
    pub natural_pnl_pct: f64,
}

impl TradeRecord {
    /// Number of bars in this trade's history.
    pub fn num_bars(&self) -> usize {
        self.bars.len()
    }

    /// P&L at a specific bar index relative to entry price.
    pub fn pnl_at(&self, bar_idx: usize) -> f64 {
        let close = self.bars[bar_idx].close;
        (close - self.entry_price) / self.entry_price
    }
}

/// Load trade records from a CSV file.
///
/// Expected CSV columns (all required):
///   trade_id, symbol, entry_price, entry_atr, entry_market_close,
///   timestamp, open, high, low, close, volume, atr, market_close,
///   bh_mass, bh_active, bar_index
///
/// Each row is one bar of one trade. Bars within a trade are ordered by bar_index.
pub fn load_trades_csv(path: &str) -> anyhow::Result<Vec<TradeRecord>> {
    use std::collections::HashMap;

    #[derive(Debug, Deserialize)]
    struct Row {
        trade_id: String,
        symbol: String,
        entry_price: f64,
        entry_atr: f64,
        entry_market_close: f64,
        timestamp: String,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        atr: f64,
        market_close: f64,
        bh_mass: f64,
        bh_active: bool,
        bar_index: usize,
        natural_exit_price: f64,
        natural_pnl_pct: f64,
    }

    let mut rdr = csv::Reader::from_path(path)?;
    let mut trade_map: HashMap<String, TradeRecord> = HashMap::new();

    for result in rdr.deserialize::<Row>() {
        let row = result?;

        let trade = trade_map
            .entry(row.trade_id.clone())
            .or_insert_with(|| TradeRecord {
                trade_id: row.trade_id.clone(),
                symbol: row.symbol.clone(),
                entry_price: row.entry_price,
                entry_atr: row.entry_atr,
                entry_market_close: row.entry_market_close,
                bh_masses: Vec::new(),
                bh_actives: Vec::new(),
                bars: Vec::new(),
                natural_exit_price: row.natural_exit_price,
                natural_pnl_pct: row.natural_pnl_pct,
            });

        trade.bars.push(Bar {
            timestamp: row.timestamp,
            open: row.open,
            high: row.high,
            low: row.low,
            close: row.close,
            volume: row.volume,
            atr: row.atr,
            market_close: row.market_close,
        });
        trade.bh_masses.push(row.bh_mass);
        trade.bh_actives.push(row.bh_active);
    }

    let mut trades: Vec<TradeRecord> = trade_map.into_values().collect();
    // Sort by trade_id for deterministic ordering
    trades.sort_by(|a, b| a.trade_id.cmp(&b.trade_id));
    Ok(trades)
}

/// Generate a synthetic trade dataset for testing/demonstration.
pub fn generate_synthetic_trades(n: usize, seed: u64) -> Vec<TradeRecord> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;
    use rand_distr::{Distribution, Normal};

    let mut rng = SmallRng::seed_from_u64(seed);
    let daily_vol = Normal::new(0.0, 0.008).unwrap();
    let mut trades = Vec::with_capacity(n);

    for i in 0..n {
        let entry_price = 100.0 + rng.gen::<f64>() * 50.0;
        let entry_atr = entry_price * 0.01;
        let n_bars: usize = rng.gen_range(5..=60);

        let mut bars = Vec::with_capacity(n_bars);
        let mut bh_masses = Vec::with_capacity(n_bars);
        let mut bh_actives = Vec::with_capacity(n_bars);

        let mut price = entry_price;
        let mut market_price = 400.0 + rng.gen::<f64>() * 20.0;

        // BH dies somewhere between bar 5 and n_bars
        let bh_death_bar: usize = rng.gen_range(5..=n_bars);

        for b in 0..n_bars {
            let ret: f64 = daily_vol.sample(&mut rng);
            let mkt_ret: f64 = daily_vol.sample(&mut rng);
            let open = price;
            price *= 1.0 + ret;
            market_price *= 1.0 + mkt_ret;
            let atr = entry_atr * (0.8 + rng.gen::<f64>() * 0.4);

            bars.push(Bar {
                timestamp: format!("2024-01-01T{:02}:00:00", b % 24),
                open,
                high: price.max(open) * (1.0 + rng.gen::<f64>() * 0.003),
                low: price.min(open) * (1.0 - rng.gen::<f64>() * 0.003),
                close: price,
                volume: 10_000.0 + rng.gen::<f64>() * 5_000.0,
                atr,
                market_close: market_price,
            });

            let bh_active = b < bh_death_bar;
            let bh_mass = if bh_active {
                0.5 + rng.gen::<f64>() * 0.5
            } else {
                rng.gen::<f64>() * 0.3
            };
            bh_masses.push(bh_mass);
            bh_actives.push(bh_active);
        }

        let natural_exit_price = price;
        let natural_pnl_pct = (natural_exit_price - entry_price) / entry_price;

        trades.push(TradeRecord {
            trade_id: format!("T{:04}", i),
            symbol: "SYNTH".to_string(),
            entry_price,
            entry_atr,
            entry_market_close: 400.0,
            bh_masses,
            bh_actives,
            bars,
            natural_exit_price,
            natural_pnl_pct,
        });
    }
    trades
}

/// Episode outcome after running one trade through the environment.
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    pub trade_id: String,
    /// Total discounted return accumulated during the episode
    pub episode_return: f64,
    /// P&L realized by the agent (fraction of entry)
    pub realized_pnl: f64,
    /// Bar on which the agent chose to exit (or last bar if forced)
    pub exit_bar: usize,
    /// Total bars in the trade
    pub total_bars: usize,
    /// True if agent chose to exit before natural trade close
    pub agent_exited_early: bool,
    /// Natural (BH-strategy) P&L for comparison
    pub natural_pnl: f64,
}

/// Stateful environment for one trade episode.
pub struct TradeEnvironment<'a> {
    trade: &'a TradeRecord,
    bar_idx: usize,
    peak_pnl: f64,
    prev_pnl: f64,
    done: bool,
}

impl<'a> TradeEnvironment<'a> {
    pub fn new(trade: &'a TradeRecord) -> Self {
        TradeEnvironment {
            trade,
            bar_idx: 0,
            peak_pnl: 0.0,
            prev_pnl: 0.0,
            done: false,
        }
    }

    /// Build the current `TradeStateRaw` for the current bar.
    pub fn current_state(&self) -> TradeStateRaw {
        let idx = self.bar_idx.min(self.trade.num_bars() - 1);
        let bar = &self.trade.bars[idx];
        let pnl = (bar.close - self.trade.entry_price) / self.trade.entry_price;

        let utc_hour = bar
            .timestamp
            .get(11..13)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(12.0);

        let market_return =
            (bar.market_close - self.trade.entry_market_close) / self.trade.entry_market_close;

        // Approximate 15m momentum as return over last 3 bars
        let momentum = if idx >= 3 {
            let prev_close = self.trade.bars[idx - 3].close;
            (bar.close - prev_close) / prev_close
        } else {
            0.0
        };

        TradeStateRaw {
            position_pnl_pct: pnl,
            bars_held: idx as u32,
            bh_mass: self.trade.bh_masses[idx],
            bh_active: self.trade.bh_actives[idx],
            atr_ratio: bar.atr / self.trade.entry_atr,
            market_return_since_entry: market_return,
            momentum_15m: momentum,
            utc_hour,
            drawdown_from_peak: pnl - self.peak_pnl,
            pnl_acceleration: pnl - self.prev_pnl,
        }
    }

    /// Step the environment forward.
    /// Returns (next_state, reward, done).
    pub fn step(&mut self, action: Action) -> (TradeStateRaw, f64, bool) {
        assert!(!self.done, "called step() on a done environment");

        let state = self.current_state();
        let last_bar = self.bar_idx >= self.trade.num_bars() - 1;
        let terminal = action == Action::Exit || last_bar;

        let reward = compute_reward(&state, action, terminal && action == Action::Hold);

        // Update peak and prev pnl
        let pnl = state.position_pnl_pct;
        if pnl > self.peak_pnl {
            self.peak_pnl = pnl;
        }
        self.prev_pnl = pnl;

        if terminal {
            self.done = true;
        } else {
            self.bar_idx += 1;
        }

        let next_state = if self.done {
            state.clone()
        } else {
            self.current_state()
        };

        (next_state, reward, self.done)
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn bar_idx(&self) -> usize {
        self.bar_idx
    }

    pub fn realized_pnl(&self) -> f64 {
        let idx = self.bar_idx.min(self.trade.num_bars() - 1);
        (self.trade.bars[idx].close - self.trade.entry_price) / self.trade.entry_price
    }
}

/// Run a full episode on a single trade using a provided action function.
pub fn run_episode<F>(trade: &TradeRecord, mut policy: F) -> EpisodeResult
where
    F: FnMut(&TradeStateRaw) -> Action,
{
    let mut env = TradeEnvironment::new(trade);
    let mut episode_return = 0.0;
    let mut discount = 1.0;
    let gamma = 0.95f64;

    while !env.is_done() {
        let state_raw = env.current_state();
        let state_vec = StateVector::from_raw(&state_raw);
        let _ = state_vec; // available for policy if needed
        let action = policy(&state_raw);
        let (_next, reward, done) = env.step(action);
        episode_return += discount * reward;
        discount *= gamma;
        if done {
            break;
        }
    }

    let exit_bar = env.bar_idx();
    let total_bars = trade.num_bars();
    let realized_pnl = env.realized_pnl();

    EpisodeResult {
        trade_id: trade.trade_id.clone(),
        episode_return,
        realized_pnl,
        exit_bar,
        total_bars,
        agent_exited_early: exit_bar < total_bars - 1,
        natural_pnl: trade.natural_pnl_pct,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_trades_generation() {
        let trades = generate_synthetic_trades(20, 42);
        assert_eq!(trades.len(), 20);
        for t in &trades {
            assert!(!t.bars.is_empty());
            assert_eq!(t.bars.len(), t.bh_masses.len());
        }
    }

    #[test]
    fn test_episode_terminates_on_exit() {
        let trades = generate_synthetic_trades(5, 1);
        let trade = &trades[0];
        let mut exit_count = 0;
        let result = run_episode(trade, |_| {
            exit_count += 1;
            if exit_count >= 3 {
                Action::Exit
            } else {
                Action::Hold
            }
        });
        assert!(result.exit_bar <= result.total_bars);
    }

    #[test]
    fn test_episode_terminates_at_last_bar() {
        let trades = generate_synthetic_trades(5, 2);
        let trade = &trades[0];
        // Always hold -> should hit last bar
        let result = run_episode(trade, |_| Action::Hold);
        assert_eq!(result.exit_bar, trade.num_bars() - 1);
    }

    #[test]
    fn test_pnl_at() {
        let trades = generate_synthetic_trades(3, 99);
        let t = &trades[0];
        let pnl = t.pnl_at(0);
        // First bar pnl is small
        assert!(pnl.abs() < 0.5);
    }
}
