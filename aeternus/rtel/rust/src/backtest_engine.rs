// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// backtest_engine.rs — Event-driven backtesting framework
// =============================================================================
//! Full backtesting harness for AETERNUS strategies.
//!
//! Architecture:
//! - `MarketEvent` stream drives execution
//! - `Strategy` trait defines signal generation
//! - `BacktestEngine` wires strategy + execution + risk
//! - `BacktestResult` aggregates metrics

use std::collections::HashMap;
use crate::now_ns;

// ---------------------------------------------------------------------------
// Market events
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Tick {
    pub timestamp_ns: u64,
    pub asset_id:     u32,
    pub bid:          f64,
    pub ask:          f64,
    pub bid_size:     f64,
    pub ask_size:     f64,
    pub last_trade:   f64,
    pub volume:       f64,
}

impl Tick {
    pub fn mid(&self) -> f64 { 0.5 * (self.bid + self.ask) }
    pub fn spread(&self) -> f64 { self.ask - self.bid }
}

#[derive(Debug, Clone)]
pub enum MarketEvent {
    Tick(Tick),
    BarClose { asset_id: u32, open: f64, high: f64, low: f64, close: f64, volume: f64, ts: u64 },
    NewsEvent { impact: f64, direction: f64, asset_id: u32, ts: u64 },
    SessionOpen { ts: u64 },
    SessionClose { ts: u64 },
}

// ---------------------------------------------------------------------------
// Strategy trait
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Signal {
    pub asset_id:   u32,
    pub direction:  f64,   // [-1, +1]
    pub confidence: f64,   // [0, 1]
    pub size_hint:  Option<f64>,
}

pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;
    fn on_tick(&mut self, tick: &Tick) -> Vec<Signal>;
    fn on_bar_close(&mut self, event: &MarketEvent) -> Vec<Signal> { let _ = event; vec![] }
    fn reset(&mut self) {}
}

// ---------------------------------------------------------------------------
// Momentum strategy
// ---------------------------------------------------------------------------

pub struct MomentumStrategy {
    lookback:   usize,
    threshold:  f64,
    price_history: HashMap<u32, Vec<f64>>,
}

impl MomentumStrategy {
    pub fn new(lookback: usize, threshold: f64) -> Self {
        Self { lookback, threshold, price_history: HashMap::new() }
    }
}

impl Strategy for MomentumStrategy {
    fn name(&self) -> &str { "momentum" }

    fn on_tick(&mut self, tick: &Tick) -> Vec<Signal> {
        let hist = self.price_history.entry(tick.asset_id).or_default();
        hist.push(tick.mid());
        if hist.len() > self.lookback * 2 { hist.remove(0); }
        if hist.len() < self.lookback { return vec![]; }

        let recent = &hist[hist.len() - self.lookback..];
        let oldest  = recent[0];
        let newest  = *recent.last().unwrap();
        if oldest.abs() < 1e-12 { return vec![]; }

        let ret = (newest - oldest) / oldest;
        if ret.abs() < self.threshold { return vec![]; }

        vec![Signal {
            asset_id:   tick.asset_id,
            direction:  ret.signum(),
            confidence: (ret.abs() / self.threshold).min(1.0),
            size_hint:  None,
        }]
    }

    fn reset(&mut self) { self.price_history.clear(); }
}

// ---------------------------------------------------------------------------
// Mean-reversion strategy
// ---------------------------------------------------------------------------

pub struct MeanReversionStrategy {
    window:     usize,
    z_threshold: f64,
    price_history: HashMap<u32, Vec<f64>>,
}

impl MeanReversionStrategy {
    pub fn new(window: usize, z_threshold: f64) -> Self {
        Self { window, z_threshold, price_history: HashMap::new() }
    }

    fn z_score(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 2.0 { return 0.0; }
        let mean: f64 = data.iter().sum::<f64>() / n;
        let var: f64  = data.iter().map(|&x| (x-mean).powi(2)).sum::<f64>() / (n-1.0);
        let std = var.sqrt();
        if std < 1e-12 { return 0.0; }
        (*data.last().unwrap() - mean) / std
    }
}

impl Strategy for MeanReversionStrategy {
    fn name(&self) -> &str { "mean_reversion" }

    fn on_tick(&mut self, tick: &Tick) -> Vec<Signal> {
        let hist = self.price_history.entry(tick.asset_id).or_default();
        hist.push(tick.mid());
        if hist.len() > self.window * 2 { hist.remove(0); }
        if hist.len() < self.window { return vec![]; }

        let recent = &hist[hist.len() - self.window..];
        let z = Self::z_score(recent);
        if z.abs() < self.z_threshold { return vec![]; }

        // Trade in opposite direction of z-score (mean reversion)
        vec![Signal {
            asset_id:   tick.asset_id,
            direction:  -z.signum(),
            confidence: ((z.abs() - self.z_threshold) / self.z_threshold).min(1.0),
            size_hint:  None,
        }]
    }

    fn reset(&mut self) { self.price_history.clear(); }
}

// ---------------------------------------------------------------------------
// Portfolio tracker
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct PortfolioState {
    pub cash:        f64,
    pub positions:   HashMap<u32, f64>,   // asset_id -> quantity
    pub avg_costs:   HashMap<u32, f64>,
    pub realized_pnl: f64,
    pub commissions: f64,
}

impl PortfolioState {
    pub fn new(initial_cash: f64) -> Self {
        Self { cash: initial_cash, ..Default::default() }
    }

    pub fn equity(&self, prices: &HashMap<u32, f64>) -> f64 {
        let unrealized: f64 = self.positions.iter().map(|(aid, &qty)| {
            let price = prices.get(aid).copied().unwrap_or(
                *self.avg_costs.get(aid).unwrap_or(&0.0)
            );
            qty * price
        }).sum();
        self.cash + unrealized
    }

    pub fn apply_trade(&mut self, asset_id: u32, qty: f64, price: f64, commission: f64) {
        let current_qty  = *self.positions.get(&asset_id).unwrap_or(&0.0);
        let current_cost = *self.avg_costs.get(&asset_id).unwrap_or(&0.0);

        if current_qty == 0.0 {
            self.positions.insert(asset_id, qty);
            self.avg_costs.insert(asset_id, price);
        } else if current_qty.signum() == qty.signum() {
            // Adding to position
            let new_qty  = current_qty + qty;
            let new_cost = (current_cost * current_qty.abs() + price * qty.abs()) / new_qty.abs();
            self.positions.insert(asset_id, new_qty);
            self.avg_costs.insert(asset_id, new_cost);
        } else {
            // Reducing/flipping
            let close_qty  = qty.abs().min(current_qty.abs());
            let sign = current_qty.signum();
            self.realized_pnl += sign * close_qty * (price - current_cost);
            let new_qty = current_qty + qty;
            if new_qty.abs() < 1e-12 {
                self.positions.remove(&asset_id);
                self.avg_costs.remove(&asset_id);
            } else {
                self.positions.insert(asset_id, new_qty);
                if new_qty.signum() != current_qty.signum() {
                    self.avg_costs.insert(asset_id, price);
                }
            }
        }

        // Update cash
        self.cash -= qty * price + commission;
        self.commissions += commission;
    }
}

// ---------------------------------------------------------------------------
// Backtest configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital:    f64,
    pub commission_bps:     f64,
    pub slippage_bps:       f64,
    pub max_position_pct:   f64,  // max single position as % of equity
    pub min_trade_size:     f64,
    pub leverage_limit:     f64,
    pub rebalance_frequency: usize,  // steps between rebalancing
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital:    1_000_000.0,
            commission_bps:     5.0,
            slippage_bps:       2.0,
            max_position_pct:   0.10,
            min_trade_size:     100.0,
            leverage_limit:     2.0,
            rebalance_frequency: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Backtest result
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct BacktestResult {
    pub total_return:      f64,
    pub annualized_return: f64,
    pub annualized_vol:    f64,
    pub sharpe_ratio:      f64,
    pub sortino_ratio:     f64,
    pub max_drawdown:      f64,
    pub calmar_ratio:      f64,
    pub total_trades:      u64,
    pub win_rate:          f64,
    pub avg_win:           f64,
    pub avg_loss:          f64,
    pub profit_factor:     f64,
    pub total_commissions: f64,
    pub equity_curve:      Vec<f64>,
    pub drawdown_series:   Vec<f64>,
    pub returns_series:    Vec<f64>,
    pub n_steps:           u64,
}

impl BacktestResult {
    pub fn compute_from_equity(equity_curve: Vec<f64>,
                                total_trades: u64,
                                win_rate: f64,
                                avg_win: f64,
                                avg_loss: f64,
                                total_commissions: f64,
                                steps_per_year: f64) -> Self
    {
        let n = equity_curve.len();
        if n < 2 { return Self::default(); }

        // Returns
        let returns: Vec<f64> = equity_curve.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let total_ret = (equity_curve.last().unwrap() - equity_curve[0]) / equity_curve[0];
        let n_years   = n as f64 / steps_per_year;
        let ann_ret   = (1.0 + total_ret).powf(1.0 / n_years) - 1.0;

        let mean_r: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let var_r:  f64 = returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>()
            / (returns.len() - 1) as f64;
        let std_r = var_r.sqrt();
        let ann_vol = std_r * steps_per_year.sqrt();

        let sharpe = if ann_vol > 1e-12 { ann_ret / ann_vol } else { 0.0 };

        // Sortino
        let downside_var: f64 = returns.iter()
            .filter(|&&r| r < 0.0)
            .map(|&r| r * r)
            .sum::<f64>() / returns.len() as f64;
        let downside_std = downside_var.sqrt() * steps_per_year.sqrt();
        let sortino = if downside_std > 1e-12 { ann_ret / downside_std } else { 0.0 };

        // Drawdown series
        let mut peak = equity_curve[0];
        let drawdown_series: Vec<f64> = equity_curve.iter().map(|&e| {
            peak = peak.max(e);
            if peak > 1e-12 { (peak - e) / peak } else { 0.0 }
        }).collect();
        let max_dd = drawdown_series.iter().cloned().fold(0.0f64, f64::max);

        let calmar = if max_dd > 1e-12 { ann_ret / max_dd } else { 0.0 };

        let profit_factor = if avg_loss.abs() > 1e-12 && win_rate < 1.0 {
            (win_rate * avg_win) / ((1.0 - win_rate) * avg_loss.abs())
        } else { 0.0 };

        Self {
            total_return:      total_ret,
            annualized_return: ann_ret,
            annualized_vol:    ann_vol,
            sharpe_ratio:      sharpe,
            sortino_ratio:     sortino,
            max_drawdown:      max_dd,
            calmar_ratio:      calmar,
            total_trades,
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            total_commissions,
            equity_curve,
            drawdown_series,
            returns_series: returns,
            n_steps: n as u64,
        }
    }

    pub fn print_summary(&self) {
        println!("=== Backtest Results ===");
        println!("  Total return:      {:.2}%", self.total_return * 100.0);
        println!("  Annual return:     {:.2}%", self.annualized_return * 100.0);
        println!("  Annual vol:        {:.2}%", self.annualized_vol * 100.0);
        println!("  Sharpe ratio:      {:.3}", self.sharpe_ratio);
        println!("  Sortino ratio:     {:.3}", self.sortino_ratio);
        println!("  Max drawdown:      {:.2}%", self.max_drawdown * 100.0);
        println!("  Calmar ratio:      {:.3}", self.calmar_ratio);
        println!("  Win rate:          {:.1}%", self.win_rate * 100.0);
        println!("  Profit factor:     {:.3}", self.profit_factor);
        println!("  Total trades:      {}", self.total_trades);
        println!("  Total commissions: ${:.2}", self.total_commissions);
        println!("  Steps:             {}", self.n_steps);
    }
}

// ---------------------------------------------------------------------------
// BacktestEngine
// ---------------------------------------------------------------------------

pub struct BacktestEngine {
    config:    BacktestConfig,
    portfolio: PortfolioState,
    strategies: Vec<Box<dyn Strategy>>,

    // Tracking
    equity_history:  Vec<f64>,
    trade_pnls:      Vec<f64>,
    n_trades:        u64,
    step_count:      u64,
    current_prices:  HashMap<u32, f64>,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        let initial = config.initial_capital;
        Self {
            portfolio:      PortfolioState::new(initial),
            config,
            strategies:     vec![],
            equity_history: vec![initial],
            trade_pnls:     vec![],
            n_trades:       0,
            step_count:     0,
            current_prices: HashMap::new(),
        }
    }

    pub fn add_strategy(&mut self, strategy: Box<dyn Strategy>) {
        self.strategies.push(strategy);
    }

    pub fn process_event(&mut self, event: &MarketEvent) {
        match event {
            MarketEvent::Tick(tick) => {
                self.current_prices.insert(tick.asset_id, tick.mid());
                self.process_tick(tick);
            }
            MarketEvent::BarClose { asset_id, close, .. } => {
                self.current_prices.insert(*asset_id, *close);
            }
            MarketEvent::SessionClose { .. } => {
                let equity = self.portfolio.equity(&self.current_prices);
                self.equity_history.push(equity);
            }
            _ => {}
        }
        self.step_count += 1;
    }

    fn process_tick(&mut self, tick: &Tick) {
        // Gather signals from all strategies
        let signals: Vec<Signal> = self.strategies.iter_mut()
            .flat_map(|s| s.on_tick(tick))
            .collect();

        for signal in signals {
            self.execute_signal(&signal, tick.mid(), tick.spread());
        }
    }

    fn execute_signal(&mut self, signal: &Signal, mid: f64, spread: f64) {
        if signal.confidence < 0.1 { return; }

        let equity = self.portfolio.equity(&self.current_prices);
        let max_pos_usd = equity * self.config.max_position_pct;

        // Current position value
        let current_qty   = *self.portfolio.positions.get(&signal.asset_id).unwrap_or(&0.0);
        let current_val   = current_qty * mid;
        let target_val    = signal.direction * signal.confidence * max_pos_usd;
        let trade_val     = target_val - current_val;

        if trade_val.abs() < self.config.min_trade_size { return; }

        // Slippage
        let slippage_frac = self.config.slippage_bps / 1e4;
        let fill_price = if trade_val > 0.0 {
            mid * (1.0 + slippage_frac)
        } else {
            mid * (1.0 - slippage_frac)
        };

        let qty = trade_val / fill_price;
        let commission = trade_val.abs() * self.config.commission_bps / 1e4;

        let prev_equity = equity;
        self.portfolio.apply_trade(signal.asset_id, qty, fill_price, commission);

        let new_equity = self.portfolio.equity(&self.current_prices);
        self.trade_pnls.push(new_equity - prev_equity);
        self.n_trades += 1;
    }

    pub fn finalize(self) -> BacktestResult {
        let n = self.equity_history.len();
        let initial = if n > 0 { self.equity_history[0] } else { 1.0 };

        // Win/loss stats from trade pnls
        let wins: Vec<f64> = self.trade_pnls.iter().filter(|&&p| p > 0.0).cloned().collect();
        let losses: Vec<f64> = self.trade_pnls.iter().filter(|&&p| p < 0.0).cloned().collect();
        let n_trades = self.trade_pnls.len() as u64;
        let win_rate = if n_trades > 0 { wins.len() as f64 / n_trades as f64 } else { 0.0 };
        let avg_win  = if !wins.is_empty() { wins.iter().sum::<f64>() / wins.len() as f64 } else { 0.0 };
        let avg_loss = if !losses.is_empty() { losses.iter().sum::<f64>() / losses.len() as f64 } else { 0.0 };

        BacktestResult::compute_from_equity(
            self.equity_history,
            n_trades,
            win_rate,
            avg_win,
            avg_loss,
            self.portfolio.commissions,
            252.0,  // assume daily steps; caller can adjust
        )
    }
}

// ---------------------------------------------------------------------------
// Synthetic data generator for backtesting
// ---------------------------------------------------------------------------

pub struct GBMGenerator {
    pub mu:      f64,
    pub sigma:   f64,
    pub dt:      f64,
    pub price:   f64,
    rng_state:   u64,
}

impl GBMGenerator {
    pub fn new(mu: f64, sigma: f64, dt: f64, initial_price: f64) -> Self {
        Self { mu, sigma, dt, price: initial_price, rng_state: 12345678901234567u64 }
    }

    // Xorshift64 RNG
    fn rand_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    // Box-Muller normal sample
    fn rand_normal(&mut self) -> f64 {
        let u1 = (self.rand_u64() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        let u2 = (self.rand_u64() as f64 + 1.0) / (u64::MAX as f64 + 2.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn next_tick(&mut self, asset_id: u32) -> Tick {
        let z     = self.rand_normal();
        let drift = (self.mu - 0.5 * self.sigma * self.sigma) * self.dt;
        let shock = self.sigma * self.dt.sqrt() * z;
        self.price *= (drift + shock).exp();

        let spread_half = self.price * 0.0005;
        let size_base   = 1000.0;
        let bid_size    = size_base * (1.0 + 0.2 * self.rand_normal().abs());
        let ask_size    = size_base * (1.0 + 0.2 * self.rand_normal().abs());

        Tick {
            timestamp_ns: now_ns(),
            asset_id,
            bid:      self.price - spread_half,
            ask:      self.price + spread_half,
            bid_size,
            ask_size,
            last_trade: self.price,
            volume:     (size_base * 10.0 + 100.0 * self.rand_normal().abs()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ticks(n: usize, n_assets: usize) -> Vec<Tick> {
        let mut gens: Vec<GBMGenerator> = (0..n_assets)
            .map(|i| GBMGenerator::new(0.0, 0.01, 1.0/252.0, 100.0 + i as f64 * 10.0))
            .collect();
        let mut ticks = Vec::with_capacity(n * n_assets);
        for _ in 0..n {
            for (i, gen) in gens.iter_mut().enumerate() {
                ticks.push(gen.next_tick(i as u32));
            }
        }
        ticks
    }

    #[test]
    fn test_gbm_generator() {
        let mut gen = GBMGenerator::new(0.0, 0.01, 1.0/252.0, 100.0);
        let mut prices = vec![];
        for i in 0..50 {
            let t = gen.next_tick(0);
            prices.push(t.mid());
        }
        assert!(prices.len() == 50);
        assert!(prices.iter().all(|&p| p > 0.0), "all prices positive");
        // Check prices are not all the same
        let min_p = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_p = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_p > min_p, "prices should vary");
    }

    #[test]
    fn test_momentum_strategy() {
        let mut strat = MomentumStrategy::new(5, 0.001);
        let mut gen = GBMGenerator::new(0.05, 0.01, 1.0/252.0, 100.0);
        let mut n_signals = 0;
        for _ in 0..100 {
            let tick = gen.next_tick(0);
            let signals = strat.on_tick(&tick);
            n_signals += signals.len();
        }
        // Not asserting specific count, just that it runs
        println!("Momentum signals generated: {}", n_signals);
    }

    #[test]
    fn test_mean_reversion_strategy() {
        let mut strat = MeanReversionStrategy::new(10, 1.5);
        let mut gen = GBMGenerator::new(0.0, 0.02, 1.0/252.0, 100.0);
        for _ in 0..50 {
            let tick = gen.next_tick(0);
            let _ = strat.on_tick(&tick);
        }
    }

    #[test]
    fn test_portfolio_state() {
        let mut port = PortfolioState::new(100_000.0);
        port.apply_trade(0, 100.0, 50.0, 1.0);
        assert!((port.cash - (100_000.0 - 100.0*50.0 - 1.0)).abs() < 1e-6);
        let mut prices = HashMap::new();
        prices.insert(0u32, 55.0f64);
        let equity = port.equity(&prices);
        assert!((equity - (100_000.0 - 100.0*50.0 - 1.0 + 100.0*55.0)).abs() < 1e-6);
    }

    #[test]
    fn test_backtest_engine_runs() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            rebalance_frequency: 1,
            ..Default::default()
        };
        let mut engine = BacktestEngine::new(config);
        engine.add_strategy(Box::new(MomentumStrategy::new(5, 0.002)));

        let mut gen = GBMGenerator::new(0.05, 0.015, 1.0/252.0, 100.0);
        for step in 0..300 {
            let tick = gen.next_tick(0);
            engine.process_event(&MarketEvent::Tick(tick));
            if step % 50 == 49 {
                engine.process_event(&MarketEvent::SessionClose { ts: now_ns() });
            }
        }

        let result = engine.finalize();
        result.print_summary();
        assert!(result.n_steps > 0);
        assert!(result.equity_curve.len() >= 2);
    }

    #[test]
    fn test_backtest_result_metrics() {
        let equity = vec![100_000.0, 102_000.0, 101_000.0, 105_000.0, 103_000.0, 108_000.0];
        let result = BacktestResult::compute_from_equity(
            equity, 10, 0.6, 500.0, -200.0, 100.0, 252.0
        );
        assert!(result.total_return > 0.0);
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
        assert!(result.win_rate > 0.0);
        println!("Sharpe: {:.3}, MaxDD: {:.2}%",
                 result.sharpe_ratio, result.max_drawdown * 100.0);
    }

    #[test]
    fn test_portfolio_pnl_tracking() {
        let mut port = PortfolioState::new(100_000.0);
        // Buy 100 at 100
        port.apply_trade(1, 100.0, 100.0, 0.0);
        // Sell 100 at 110 (profit 1000)
        port.apply_trade(1, -100.0, 110.0, 0.0);
        assert!((port.realized_pnl - 1000.0).abs() < 1e-6);
    }
}
