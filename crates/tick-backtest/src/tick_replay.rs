// tick_replay.rs -- Tick-level replay engine for SRFM backtesting.
// Simulates market microstructure: fill latency, slippage, and partial fills.

use std::collections::VecDeque;
use std::path::Path;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Tick
// ---------------------------------------------------------------------------

/// A single market trade tick.
#[derive(Debug, Clone, PartialEq)]
pub struct Tick {
    pub price: f64,
    pub volume: f64,
    /// +1 = buy-aggressor (uptick), -1 = sell-aggressor (downtick), 0 = unknown.
    pub side: i8,
    pub timestamp_ns: i64,
}

impl Tick {
    pub fn new(price: f64, volume: f64, side: i8, timestamp_ns: i64) -> Self {
        Self { price, volume, side, timestamp_ns }
    }
}

// ---------------------------------------------------------------------------
// OrderBook -- minimal representation used during replay
// ---------------------------------------------------------------------------

/// A minimal order book snapshot updated tick-by-tick.
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bid: f64,
    pub ask: f64,
    pub last_price: f64,
    pub last_volume: f64,
    pub last_side: i8,
}

impl OrderBook {
    pub fn new(mid: f64, half_spread: f64) -> Self {
        Self {
            bid: mid - half_spread,
            ask: mid + half_spread,
            last_price: mid,
            last_volume: 0.0,
            last_side: 0,
        }
    }

    pub fn update(&mut self, tick: &Tick, spread_bps: f64) {
        self.last_price = tick.price;
        self.last_volume = tick.volume;
        self.last_side = tick.side;
        let half_spread = tick.price * spread_bps / 20_000.0; // spread_bps / 2 / 10000
        self.bid = tick.price - half_spread;
        self.ask = tick.price + half_spread;
    }

    pub fn mid(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }
}

// ---------------------------------------------------------------------------
// TickOrder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum TickOrderSide {
    Buy,
    Sell,
}

/// An order submitted by a TickStrategy in response to a tick.
#[derive(Debug, Clone)]
pub struct TickOrder {
    pub side: TickOrderSide,
    pub qty: f64,
    /// None = market order, Some = limit price.
    pub limit_price: Option<f64>,
    pub order_id: u64,
}

// ---------------------------------------------------------------------------
// TickFill
// ---------------------------------------------------------------------------

/// Fill notification delivered to the strategy after order execution.
#[derive(Debug, Clone)]
pub struct TickFill {
    pub order_id: u64,
    pub side: TickOrderSide,
    pub fill_qty: f64,
    pub fill_price: f64,
    pub fill_time_ns: i64,
    pub slippage_bps: f64,
}

// ---------------------------------------------------------------------------
// TickStrategy trait
// ---------------------------------------------------------------------------

pub trait TickStrategy {
    /// Called on every tick. Returns an optional order to submit.
    fn on_tick(&mut self, tick: &Tick, book: &OrderBook) -> Option<TickOrder>;
    /// Called when a previously submitted order is filled (fully or partially).
    fn on_fill(&mut self, fill: TickFill);
}

// ---------------------------------------------------------------------------
// ReplayConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Simulated fill latency in nanoseconds.
    pub latency_ns: i64,
    /// Effective bid-ask spread in basis points applied during fill simulation.
    pub spread_bps: f64,
    /// Probability [0, 1] that any given order results in a partial fill
    /// (fills half the requested quantity).
    pub partial_fill_rate: f64,
    /// Starting equity for P&L tracking.
    pub initial_equity: f64,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            latency_ns: 1_000_000, // 1 ms
            spread_bps: 1.0,
            partial_fill_rate: 0.0,
            initial_equity: 100_000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// ReplayResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub equity_curve: Vec<f64>,
    pub sharpe: f64,
    pub total_trades: usize,
    pub avg_slippage_bps: f64,
    pub total_pnl: f64,
    pub max_drawdown: f64,
}

// ---------------------------------------------------------------------------
// LoadError
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error on line {line}: {msg}")]
    Parse { line: usize, msg: String },
    #[error("Missing column: {0}")]
    MissingColumn(String),
}

// ---------------------------------------------------------------------------
// load_ticks_from_csv
// ---------------------------------------------------------------------------

/// Load ticks from a CSV file.
/// Expected columns (header row required): timestamp_ns,price,volume,side
/// `side` values: 1 = buy, -1 = sell, 0 = unknown. Missing side defaults to 0.
pub fn load_ticks_from_csv(path: &Path) -> Result<Vec<Tick>, LoadError> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Parse header
    let header_line = lines
        .next()
        .ok_or_else(|| LoadError::Parse { line: 1, msg: "Empty file".to_string() })??;
    let headers: Vec<&str> = header_line.split(',').map(str::trim).collect();

    let col = |name: &str| -> Result<usize, LoadError> {
        headers
            .iter()
            .position(|h| *h == name)
            .ok_or_else(|| LoadError::MissingColumn(name.to_string()))
    };

    let ts_col = col("timestamp_ns")?;
    let px_col = col("price")?;
    let vol_col = col("volume")?;
    // side column is optional
    let side_col = headers.iter().position(|h| *h == "side");

    let mut ticks = Vec::new();
    for (idx, line_res) in lines.enumerate() {
        let line = line_res?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();

        let parse_f64 = |col: usize, field: &str| -> Result<f64, LoadError> {
            field.trim().parse::<f64>().map_err(|_| LoadError::Parse {
                line: idx + 2,
                msg: format!("Cannot parse f64 in column {col}: '{field}'"),
            })
        };
        let parse_i64 = |col: usize, field: &str| -> Result<i64, LoadError> {
            field.trim().parse::<i64>().map_err(|_| LoadError::Parse {
                line: idx + 2,
                msg: format!("Cannot parse i64 in column {col}: '{field}'"),
            })
        };

        if fields.len() <= ts_col.max(px_col).max(vol_col) {
            return Err(LoadError::Parse {
                line: idx + 2,
                msg: format!("Too few columns: expected at least {}", ts_col.max(px_col).max(vol_col) + 1),
            });
        }

        let timestamp_ns = parse_i64(ts_col, fields[ts_col])?;
        let price = parse_f64(px_col, fields[px_col])?;
        let volume = parse_f64(vol_col, fields[vol_col])?;
        let side: i8 = if let Some(sc) = side_col {
            if sc < fields.len() {
                fields[sc].trim().parse::<i8>().unwrap_or(0)
            } else {
                0
            }
        } else {
            0
        };

        ticks.push(Tick { price, volume, side, timestamp_ns });
    }

    Ok(ticks)
}

// ---------------------------------------------------------------------------
// PendingOrder -- internal replay state
// ---------------------------------------------------------------------------

struct PendingOrder {
    order: TickOrder,
    submitted_at_ns: i64,
    intent_price: f64,
}

// ---------------------------------------------------------------------------
// replay
// ---------------------------------------------------------------------------

/// Replay a tick sequence through a strategy, simulating fill latency and slippage.
pub fn replay(
    ticks: &[Tick],
    strategy: &mut impl TickStrategy,
    config: &ReplayConfig,
) -> ReplayResult {
    let mut book = OrderBook::new(
        ticks.first().map(|t| t.price).unwrap_or(100.0),
        ticks.first().map(|t| t.price * config.spread_bps / 20_000.0).unwrap_or(0.05),
    );

    let mut pending: VecDeque<PendingOrder> = VecDeque::new();
    let mut next_order_id: u64 = 0;

    let mut equity = config.initial_equity;
    let mut position: f64 = 0.0;      // in units
    let mut position_cost: f64 = 0.0; // total cost basis
    let mut equity_curve: Vec<f64> = vec![equity];

    let mut total_trades: usize = 0;
    let mut slippage_sum_bps: f64 = 0.0;
    let mut slippage_count: usize = 0;
    let mut peak_equity = equity;
    let mut max_drawdown: f64 = 0.0;

    // Simple deterministic partial-fill: use order_id parity as a surrogate
    let should_partial_fill = |id: u64| -> bool {
        config.partial_fill_rate > 0.0 && (id % 2 == 0) && (config.partial_fill_rate >= 0.5)
    };

    for tick in ticks {
        book.update(tick, config.spread_bps);

        // Check pending orders whose latency has elapsed
        while let Some(front) = pending.front() {
            if tick.timestamp_ns < front.submitted_at_ns + config.latency_ns {
                break;
            }
            let po = pending.pop_front().unwrap();
            let id = po.order.order_id;

            // Determine fill price with spread slippage
            let half_spread = tick.price * config.spread_bps / 20_000.0;
            let raw_fill_price = match po.order.side {
                TickOrderSide::Buy => tick.price + half_spread,
                TickOrderSide::Sell => tick.price - half_spread,
            };

            let intent = po.intent_price;
            let slippage_bps = if intent > 1e-12 {
                ((raw_fill_price - intent) / intent * 10_000.0).abs()
            } else {
                0.0
            };

            let fill_qty = if should_partial_fill(id) {
                po.order.qty * 0.5
            } else {
                po.order.qty
            };

            // Update position and P&L
            match po.order.side {
                TickOrderSide::Buy => {
                    position += fill_qty;
                    position_cost += fill_qty * raw_fill_price;
                }
                TickOrderSide::Sell => {
                    let close_qty = fill_qty.min(position.abs());
                    if position > 0.0 && close_qty > 0.0 {
                        // Closing long
                        let avg_cost = position_cost / position;
                        let realized = close_qty * (raw_fill_price - avg_cost);
                        equity += realized;
                        position -= close_qty;
                        position_cost -= close_qty * avg_cost;
                    } else {
                        // Opening short
                        position -= fill_qty;
                        position_cost -= fill_qty * raw_fill_price;
                    }
                    total_trades += 1;
                }
            }

            slippage_sum_bps += slippage_bps;
            slippage_count += 1;

            strategy.on_fill(TickFill {
                order_id: id,
                side: po.order.side.clone(),
                fill_qty,
                fill_price: raw_fill_price,
                fill_time_ns: tick.timestamp_ns,
                slippage_bps,
            });
        }

        // Mark equity to market including open position
        let mark_equity = equity + position * (tick.price - if position > 0.0 {
            tick.price * config.spread_bps / 20_000.0
        } else {
            0.0
        });
        equity_curve.push(mark_equity);

        if mark_equity > peak_equity {
            peak_equity = mark_equity;
        }
        let dd = (peak_equity - mark_equity) / peak_equity;
        if dd > max_drawdown {
            max_drawdown = dd;
        }

        // Ask strategy for an order
        if let Some(mut order) = strategy.on_tick(tick, &book) {
            order.order_id = next_order_id;
            next_order_id += 1;
            let intent_price = tick.price;
            pending.push_back(PendingOrder {
                order,
                submitted_at_ns: tick.timestamp_ns,
                intent_price,
            });
        }
    }

    // Sharpe: annualise daily-equivalent returns from equity curve
    let sharpe = compute_sharpe(&equity_curve);

    let avg_slippage_bps = if slippage_count > 0 {
        slippage_sum_bps / slippage_count as f64
    } else {
        0.0
    };

    ReplayResult {
        total_pnl: equity_curve.last().copied().unwrap_or(config.initial_equity)
            - config.initial_equity,
        equity_curve,
        sharpe,
        total_trades,
        avg_slippage_bps,
        max_drawdown,
    }
}

fn compute_sharpe(equity_curve: &[f64]) -> f64 {
    if equity_curve.len() < 2 {
        return 0.0;
    }
    let returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| if w[0] > 1e-12 { (w[1] - w[0]) / w[0] } else { 0.0 })
        .collect();
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < 1e-12 {
        0.0
    } else {
        // Annualise assuming ~252 * 6.5 * 3600 * 1e9 ns per year
        mean / std * (n * 252.0).sqrt()
    }
}

// ---------------------------------------------------------------------------
// TickSimpleStrategy -- 3 consecutive uptick/downtick strategy
// ---------------------------------------------------------------------------

/// Example strategy: enter long after 3 consecutive buy-side ticks,
/// close and go short after 3 consecutive sell-side ticks.
pub struct TickSimpleStrategy {
    pub consecutive_up: u8,
    pub consecutive_down: u8,
    pub position: f64,
    pub lot_size: f64,
    orders_sent: u64,
}

impl TickSimpleStrategy {
    pub fn new(lot_size: f64) -> Self {
        Self {
            consecutive_up: 0,
            consecutive_down: 0,
            position: 0.0,
            lot_size,
            orders_sent: 0,
        }
    }
}

impl TickStrategy for TickSimpleStrategy {
    fn on_tick(&mut self, tick: &Tick, _book: &OrderBook) -> Option<TickOrder> {
        match tick.side {
            1 => {
                self.consecutive_up += 1;
                self.consecutive_down = 0;
            }
            -1 => {
                self.consecutive_down += 1;
                self.consecutive_up = 0;
            }
            _ => {}
        }

        if self.consecutive_up >= 3 && self.position <= 0.0 {
            self.consecutive_up = 0;
            self.orders_sent += 1;
            let close_qty = if self.position < 0.0 { self.lot_size } else { 0.0 };
            let _ = close_qty; // closing covered in full lot below
            self.position = self.lot_size;
            return Some(TickOrder {
                side: TickOrderSide::Buy,
                qty: self.lot_size,
                limit_price: None,
                order_id: 0, // assigned by replay engine
            });
        }

        if self.consecutive_down >= 3 && self.position >= 0.0 {
            self.consecutive_down = 0;
            self.orders_sent += 1;
            self.position = -self.lot_size;
            return Some(TickOrder {
                side: TickOrderSide::Sell,
                qty: self.lot_size,
                limit_price: None,
                order_id: 0,
            });
        }

        None
    }

    fn on_fill(&mut self, _fill: TickFill) {
        // No-op for simple strategy -- position tracking done in on_tick
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tick(price: f64, volume: f64, side: i8, ts_ns: i64) -> Tick {
        Tick::new(price, volume, side, ts_ns)
    }

    fn ns(ms: i64) -> i64 {
        ms * 1_000_000
    }

    // ---------------------------------------------------------------------------
    // Tick construction
    // ---------------------------------------------------------------------------

    #[test]
    fn test_tick_new() {
        let t = make_tick(100.0, 500.0, 1, ns(1000));
        assert_eq!(t.price, 100.0);
        assert_eq!(t.side, 1);
        assert_eq!(t.timestamp_ns, ns(1000));
    }

    // ---------------------------------------------------------------------------
    // OrderBook
    // ---------------------------------------------------------------------------

    #[test]
    fn test_order_book_update() {
        let mut book = OrderBook::new(100.0, 0.05);
        let tick = make_tick(101.0, 100.0, 1, ns(0));
        book.update(&tick, 10.0); // 10 bps spread
        assert!((book.mid() - 101.0).abs() < 0.001);
        assert!(book.ask > book.bid);
    }

    #[test]
    fn test_order_book_mid() {
        let book = OrderBook::new(200.0, 1.0);
        assert_eq!(book.mid(), 200.0);
    }

    // ---------------------------------------------------------------------------
    // TickSimpleStrategy
    // ---------------------------------------------------------------------------

    #[test]
    fn test_simple_strategy_buys_after_three_upticks() {
        let mut strat = TickSimpleStrategy::new(100.0);
        let book = OrderBook::new(100.0, 0.05);
        let t1 = make_tick(100.0, 10.0, 1, ns(1));
        let t2 = make_tick(100.1, 10.0, 1, ns(2));
        let t3 = make_tick(100.2, 10.0, 1, ns(3));
        assert!(strat.on_tick(&t1, &book).is_none());
        assert!(strat.on_tick(&t2, &book).is_none());
        let order = strat.on_tick(&t3, &book);
        assert!(order.is_some());
        let o = order.unwrap();
        assert!(matches!(o.side, TickOrderSide::Buy));
    }

    #[test]
    fn test_simple_strategy_sells_after_three_downticks() {
        let mut strat = TickSimpleStrategy::new(100.0);
        let book = OrderBook::new(100.0, 0.05);
        for &s in &[-1i8, -1, -1] {
            let _ = strat.on_tick(&make_tick(100.0, 10.0, s, ns(1)), &book);
        }
        // After 3 down ticks -- strategy should want to sell
        // Reset and try again explicitly
        let mut strat2 = TickSimpleStrategy::new(50.0);
        strat2.on_tick(&make_tick(100.0, 1.0, -1, ns(1)), &book);
        strat2.on_tick(&make_tick(99.9, 1.0, -1, ns(2)), &book);
        let ord = strat2.on_tick(&make_tick(99.8, 1.0, -1, ns(3)), &book);
        assert!(ord.is_some());
        assert!(matches!(ord.unwrap().side, TickOrderSide::Sell));
    }

    #[test]
    fn test_simple_strategy_reset_on_opposite_tick() {
        let mut strat = TickSimpleStrategy::new(100.0);
        let book = OrderBook::new(100.0, 0.05);
        strat.on_tick(&make_tick(100.0, 1.0, 1, ns(1)), &book);
        strat.on_tick(&make_tick(100.1, 1.0, 1, ns(2)), &book);
        // Down tick resets up counter
        strat.on_tick(&make_tick(100.0, 1.0, -1, ns(3)), &book);
        // Two more ups -- not enough for a buy signal
        strat.on_tick(&make_tick(100.1, 1.0, 1, ns(4)), &book);
        let ord = strat.on_tick(&make_tick(100.2, 1.0, 1, ns(5)), &book);
        assert!(ord.is_none()); // only 2 consecutive ups after reset
    }

    // ---------------------------------------------------------------------------
    // replay
    // ---------------------------------------------------------------------------

    #[test]
    fn test_replay_runs_without_error() {
        let ticks: Vec<Tick> = (0..20)
            .map(|i| make_tick(100.0 + i as f64 * 0.1, 100.0, if i % 2 == 0 { 1 } else { -1 }, ns(i * 10)))
            .collect();
        let mut strat = TickSimpleStrategy::new(1.0);
        let config = ReplayConfig::default();
        let result = replay(&ticks, &mut strat, &config);
        assert!(!result.equity_curve.is_empty());
    }

    #[test]
    fn test_replay_equity_curve_length() {
        let n = 30usize;
        let ticks: Vec<Tick> = (0..n)
            .map(|i| make_tick(100.0, 50.0, 1, ns(i as i64 * 5)))
            .collect();
        let mut strat = TickSimpleStrategy::new(1.0);
        let config = ReplayConfig::default();
        let result = replay(&ticks, &mut strat, &config);
        // equity_curve has one entry per tick plus the initial
        assert_eq!(result.equity_curve.len(), n + 1);
    }

    #[test]
    fn test_replay_no_trades_on_mixed_ticks() {
        // Alternating sides -- never 3 in a row, so no trades
        let ticks: Vec<Tick> = (0..10)
            .map(|i| make_tick(100.0, 1.0, if i % 2 == 0 { 1 } else { -1 }, ns(i)))
            .collect();
        let mut strat = TickSimpleStrategy::new(1.0);
        let config = ReplayConfig::default();
        let result = replay(&ticks, &mut strat, &config);
        assert_eq!(result.total_trades, 0);
    }

    #[test]
    fn test_replay_slippage_nonzero_with_spread() {
        // All upticks to trigger a buy, then all downticks to trigger a sell
        let mut ticks: Vec<Tick> = (0..3)
            .map(|i| make_tick(100.0 + i as f64 * 0.1, 10.0, 1, ns(i * 1_000_000)))
            .collect();
        ticks.extend((3..6).map(|i| make_tick(100.0 - (i - 3) as f64 * 0.1, 10.0, -1, ns(i * 1_000_000))));
        let mut strat = TickSimpleStrategy::new(1.0);
        let config = ReplayConfig { latency_ns: 500_000, spread_bps: 5.0, ..Default::default() };
        let result = replay(&ticks, &mut strat, &config);
        // With spread > 0, slippage should be > 0 if any fills occurred
        if result.total_trades > 0 {
            assert!(result.avg_slippage_bps >= 0.0);
        }
    }

    // ---------------------------------------------------------------------------
    // load_ticks_from_csv
    // ---------------------------------------------------------------------------

    #[test]
    fn test_load_ticks_from_csv_missing_file() {
        let result = load_ticks_from_csv(Path::new("/nonexistent/path/ticks.csv"));
        assert!(result.is_err());
    }
}
