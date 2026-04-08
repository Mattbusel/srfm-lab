// engine.rs — Event-driven backtest engine, bar/tick modes, multi-asset, fill simulation, slippage
use crate::data::{Bar, BarSeries, Tick, TickSeries, MultiAssetData};
use crate::portfolio::{Portfolio, Position, Trade, TradeSide};
use crate::strategy::Signal;
use std::collections::HashMap;

/// Order types
#[derive(Clone, Debug, PartialEq)]
pub enum OrderType {
    Market,
    Limit { price: f64 },
    Stop { stop_price: f64 },
    StopLimit { stop_price: f64, limit_price: f64 },
    TrailingStop { trail_pct: f64 },
    MOC, // Market on close
    TWAP { duration_bars: usize },
    VWAP { duration_bars: usize },
    Iceberg { visible_qty: f64 },
}

/// Order state
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Order
#[derive(Clone, Debug)]
pub struct Order {
    pub id: u64,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub order_type: OrderType,
    pub status: OrderStatus,
    pub submit_time: u64,
    pub fill_time: Option<u64>,
    pub fill_price: Option<f64>,
    pub slippage: f64,
    pub commission: f64,
    pub time_in_force: TimeInForce,
    pub tag: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimeInForce {
    GTC,  // Good til cancelled
    DAY,
    IOC,  // Immediate or cancel
    FOK,  // Fill or kill
    GTD(u64), // Good til date
}

impl Order {
    pub fn market(id: u64, symbol: &str, side: TradeSide, qty: f64) -> Self {
        Self {
            id, symbol: symbol.to_string(), side, quantity: qty, filled_quantity: 0.0,
            order_type: OrderType::Market, status: OrderStatus::Pending,
            submit_time: 0, fill_time: None, fill_price: None,
            slippage: 0.0, commission: 0.0, time_in_force: TimeInForce::GTC, tag: String::new(),
        }
    }

    pub fn limit(id: u64, symbol: &str, side: TradeSide, qty: f64, price: f64) -> Self {
        let mut o = Self::market(id, symbol, side, qty);
        o.order_type = OrderType::Limit { price };
        o
    }

    pub fn stop(id: u64, symbol: &str, side: TradeSide, qty: f64, stop_price: f64) -> Self {
        let mut o = Self::market(id, symbol, side, qty);
        o.order_type = OrderType::Stop { stop_price };
        o
    }

    pub fn remaining(&self) -> f64 { self.quantity - self.filled_quantity }
    pub fn is_filled(&self) -> bool { self.status == OrderStatus::Filled }
    pub fn is_active(&self) -> bool { self.status == OrderStatus::Pending || self.status == OrderStatus::PartiallyFilled }
    pub fn notional(&self) -> f64 { self.fill_price.unwrap_or(0.0) * self.filled_quantity }
    pub fn total_cost(&self) -> f64 { self.notional() + self.slippage + self.commission }
}

/// Slippage models
#[derive(Clone, Debug)]
pub enum SlippageModel {
    Fixed(f64),
    Proportional(f64),
    SqrtImpact { coefficient: f64, daily_volume: f64 },
    TickBased { tick_size: f64, num_ticks: f64 },
    VolumeDependent { base_bps: f64, volume_factor: f64 },
    Zero,
}

impl SlippageModel {
    pub fn compute(&self, price: f64, quantity: f64, side: TradeSide, bar_volume: f64) -> f64 {
        let direction = match side { TradeSide::Buy => 1.0, TradeSide::Sell => -1.0 };
        match self {
            SlippageModel::Fixed(amount) => *amount * direction,
            SlippageModel::Proportional(bps) => price * bps / 10000.0 * direction,
            SlippageModel::SqrtImpact { coefficient, daily_volume } => {
                let participation = quantity / daily_volume.max(1.0);
                coefficient * price * participation.sqrt() * direction
            }
            SlippageModel::TickBased { tick_size, num_ticks } => {
                tick_size * num_ticks * direction
            }
            SlippageModel::VolumeDependent { base_bps, volume_factor } => {
                let vol_ratio = if bar_volume > 0.0 { quantity / bar_volume } else { 0.01 };
                let bps = base_bps + volume_factor * vol_ratio;
                price * bps / 10000.0 * direction
            }
            SlippageModel::Zero => 0.0,
        }
    }
}

/// Commission models
#[derive(Clone, Debug)]
pub enum CommissionModel {
    Fixed(f64),
    PerShare(f64),
    Percentage(f64),
    Tiered(Vec<(f64, f64)>), // (volume_threshold, rate)
    Zero,
}

impl CommissionModel {
    pub fn compute(&self, quantity: f64, price: f64) -> f64 {
        match self {
            CommissionModel::Fixed(fee) => *fee,
            CommissionModel::PerShare(rate) => quantity.abs() * rate,
            CommissionModel::Percentage(pct) => quantity.abs() * price * pct,
            CommissionModel::Tiered(tiers) => {
                let notional = quantity.abs() * price;
                for &(thresh, rate) in tiers.iter().rev() {
                    if notional >= thresh { return notional * rate; }
                }
                if let Some(&(_, rate)) = tiers.first() { notional * rate } else { 0.0 }
            }
            CommissionModel::Zero => 0.0,
        }
    }
}

/// Fill simulation
#[derive(Clone, Debug)]
pub struct FillSimulator {
    pub slippage: SlippageModel,
    pub commission: CommissionModel,
    pub partial_fill_prob: f64,
    pub rejection_prob: f64,
    pub latency_bars: usize,
}

impl FillSimulator {
    pub fn default_sim() -> Self {
        Self {
            slippage: SlippageModel::Proportional(1.0),
            commission: CommissionModel::PerShare(0.001),
            partial_fill_prob: 0.0,
            rejection_prob: 0.0,
            latency_bars: 0,
        }
    }

    pub fn realistic() -> Self {
        Self {
            slippage: SlippageModel::SqrtImpact { coefficient: 0.1, daily_volume: 1_000_000.0 },
            commission: CommissionModel::PerShare(0.005),
            partial_fill_prob: 0.05,
            rejection_prob: 0.01,
            latency_bars: 1,
        }
    }

    pub fn try_fill(&self, order: &mut Order, bar: &Bar) -> Option<Trade> {
        let fill_price = match &order.order_type {
            OrderType::Market => Some(match order.side {
                TradeSide::Buy => bar.open,
                TradeSide::Sell => bar.open,
            }),
            OrderType::Limit { price } => {
                match order.side {
                    TradeSide::Buy => if bar.low <= *price { Some((*price).min(bar.open)) } else { None },
                    TradeSide::Sell => if bar.high >= *price { Some((*price).max(bar.open)) } else { None },
                }
            }
            OrderType::Stop { stop_price } => {
                match order.side {
                    TradeSide::Buy => if bar.high >= *stop_price { Some((*stop_price).max(bar.open)) } else { None },
                    TradeSide::Sell => if bar.low <= *stop_price { Some((*stop_price).min(bar.open)) } else { None },
                }
            }
            OrderType::StopLimit { stop_price, limit_price } => {
                let triggered = match order.side {
                    TradeSide::Buy => bar.high >= *stop_price,
                    TradeSide::Sell => bar.low <= *stop_price,
                };
                if !triggered { return None; }
                match order.side {
                    TradeSide::Buy => if bar.low <= *limit_price { Some(*limit_price) } else { None },
                    TradeSide::Sell => if bar.high >= *limit_price { Some(*limit_price) } else { None },
                }
            }
            OrderType::MOC => Some(bar.close),
            _ => Some(bar.open), // simplified for TWAP/VWAP/Iceberg
        };

        let fp = match fill_price {
            Some(p) => p,
            None => return None,
        };

        let slip = self.slippage.compute(fp, order.quantity, order.side, bar.volume);
        let final_price = fp + slip;
        let comm = self.commission.compute(order.quantity, final_price);

        order.filled_quantity = order.quantity;
        order.fill_price = Some(final_price);
        order.fill_time = Some(bar.timestamp);
        order.slippage = slip.abs();
        order.commission = comm;
        order.status = OrderStatus::Filled;

        let signed_qty = match order.side {
            TradeSide::Buy => order.quantity,
            TradeSide::Sell => -order.quantity,
        };

        Some(Trade {
            timestamp: bar.timestamp,
            symbol: order.symbol.clone(),
            side: order.side,
            quantity: order.quantity,
            price: final_price,
            commission: comm,
            slippage: slip.abs(),
            order_id: order.id,
            pnl: 0.0,
            tag: order.tag.clone(),
        })
    }
}

/// Event types in the backtest engine
#[derive(Clone, Debug)]
pub enum BacktestEvent {
    BarUpdate { timestamp: u64, symbol: String, bar: Bar },
    TickUpdate { timestamp: u64, symbol: String, tick: Tick },
    OrderSubmitted { order: Order },
    OrderFilled { trade: Trade },
    OrderCancelled { order_id: u64 },
    SignalGenerated { timestamp: u64, signals: HashMap<String, Signal> },
    Rebalance { timestamp: u64 },
    EndOfDay { timestamp: u64 },
    EndOfBacktest,
}

/// Backtest configuration
#[derive(Clone, Debug)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub start_time: u64,
    pub end_time: u64,
    pub warmup_bars: usize,
    pub mode: BacktestMode,
    pub fill_sim: FillSimulator,
    pub max_positions: usize,
    pub max_leverage: f64,
    pub risk_free_rate: f64,
    pub benchmark_symbol: Option<String>,
    pub rebalance_frequency: Option<usize>, // every N bars
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BacktestMode {
    BarByBar,
    TickByTick,
    EventDriven,
}

impl BacktestConfig {
    pub fn default_config(capital: f64) -> Self {
        Self {
            initial_capital: capital,
            start_time: 0,
            end_time: u64::MAX,
            warmup_bars: 0,
            mode: BacktestMode::BarByBar,
            fill_sim: FillSimulator::default_sim(),
            max_positions: 100,
            max_leverage: 1.0,
            risk_free_rate: 0.0,
            benchmark_symbol: None,
            rebalance_frequency: None,
        }
    }
}

/// Callback trait for strategy
pub trait StrategyCallback {
    fn on_bar(&mut self, timestamp: u64, bars: &HashMap<&str, &Bar>, portfolio: &Portfolio) -> Vec<Order>;
    fn on_trade(&mut self, _trade: &Trade) {}
    fn on_end_of_day(&mut self, _timestamp: u64, _portfolio: &Portfolio) {}
    fn name(&self) -> &str { "unnamed" }
}

/// Main backtest engine
pub struct BacktestEngine {
    pub config: BacktestConfig,
    pub portfolio: Portfolio,
    pub pending_orders: Vec<Order>,
    pub order_history: Vec<Order>,
    pub event_log: Vec<BacktestEvent>,
    pub equity_curve: Vec<(u64, f64)>,
    pub next_order_id: u64,
    pub bar_count: usize,
    pub current_timestamp: u64,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        let portfolio = Portfolio::new(config.initial_capital);
        Self {
            config,
            portfolio,
            pending_orders: Vec::new(),
            order_history: Vec::new(),
            event_log: Vec::new(),
            equity_curve: Vec::new(),
            next_order_id: 1,
            bar_count: 0,
            current_timestamp: 0,
        }
    }

    pub fn submit_order(&mut self, mut order: Order) -> u64 {
        let id = self.next_order_id;
        self.next_order_id += 1;
        order.id = id;
        order.submit_time = self.current_timestamp;
        self.event_log.push(BacktestEvent::OrderSubmitted { order: order.clone() });
        self.pending_orders.push(order);
        id
    }

    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        if let Some(pos) = self.pending_orders.iter().position(|o| o.id == order_id) {
            let mut order = self.pending_orders.remove(pos);
            order.status = OrderStatus::Cancelled;
            self.event_log.push(BacktestEvent::OrderCancelled { order_id });
            self.order_history.push(order);
            return true;
        }
        false
    }

    pub fn cancel_all(&mut self) {
        let ids: Vec<u64> = self.pending_orders.iter().map(|o| o.id).collect();
        for id in ids { self.cancel_order(id); }
    }

    fn process_fills(&mut self, bars: &HashMap<&str, &Bar>) {
        let mut filled = Vec::new();
        let mut remaining = Vec::new();

        for mut order in self.pending_orders.drain(..) {
            if let Some(&bar) = bars.get(order.symbol.as_str()) {
                if let Some(trade) = self.config.fill_sim.try_fill(&mut order, bar) {
                    // Update portfolio
                    self.portfolio.process_trade(&trade);
                    self.event_log.push(BacktestEvent::OrderFilled { trade: trade.clone() });
                    filled.push(order);
                } else {
                    remaining.push(order);
                }
            } else {
                remaining.push(order);
            }
        }

        self.pending_orders = remaining;
        self.order_history.extend(filled);
    }

    fn update_portfolio_marks(&mut self, bars: &HashMap<String, &Bar>) {
        let mut prices = HashMap::new();
        for (sym, bar) in bars {
            prices.insert(sym.to_string(), bar.close);
        }
        self.portfolio.mark_to_market(&prices);
    }

    fn record_equity(&mut self, timestamp: u64) {
        let equity = self.portfolio.total_equity();
        self.equity_curve.push((timestamp, equity));
    }

    /// Run backtest on single asset bar data
    pub fn run_single(&mut self, data: &BarSeries, strategy: &mut dyn StrategyCallback) {
        let symbol = &data.symbol;
        for (i, bar) in data.bars.iter().enumerate() {
            if bar.timestamp < self.config.start_time { continue; }
            if bar.timestamp > self.config.end_time { break; }

            self.current_timestamp = bar.timestamp;
            self.bar_count += 1;

            let mut bars_map: HashMap<&str, &Bar> = HashMap::new();
            bars_map.insert(symbol.as_str(), bar);

            // Process pending orders
            self.process_fills(&bars_map);

            // Update marks
            let mut price_map = HashMap::new();
            price_map.insert(symbol.clone(), bar.close);
            self.portfolio.mark_to_market(&price_map);

            // Get new orders from strategy
            if self.bar_count > self.config.warmup_bars {
                let orders = strategy.on_bar(bar.timestamp, &bars_map, &self.portfolio);
                for order in orders {
                    self.submit_order(order);
                }
            }

            self.record_equity(bar.timestamp);
        }

        self.event_log.push(BacktestEvent::EndOfBacktest);
    }

    /// Run backtest on multi-asset data
    pub fn run_multi(&mut self, data: &MultiAssetData, strategy: &mut dyn StrategyCallback) {
        let num_bars = data.num_timestamps();
        for i in 0..num_bars {
            let ts = data.aligned_timestamps[i];
            if ts < self.config.start_time { continue; }
            if ts > self.config.end_time { break; }

            self.current_timestamp = ts;
            self.bar_count += 1;

            let mut bars_map: HashMap<&str, &Bar> = HashMap::new();
            for (sym, series) in &data.series {
                if i < series.bars.len() {
                    bars_map.insert(sym.as_str(), &series.bars[i]);
                }
            }

            // Process fills
            self.process_fills(&bars_map);

            // Update marks
            let mut prices = HashMap::new();
            for (&sym, bar) in &bars_map {
                prices.insert(sym.to_string(), bar.close);
            }
            self.portfolio.mark_to_market(&prices);

            // Strategy
            if self.bar_count > self.config.warmup_bars {
                let orders = strategy.on_bar(ts, &bars_map, &self.portfolio);
                for order in orders {
                    self.submit_order(order);
                }
            }

            self.record_equity(ts);

            // Rebalance check
            if let Some(freq) = self.config.rebalance_frequency {
                if self.bar_count % freq == 0 {
                    self.event_log.push(BacktestEvent::Rebalance { timestamp: ts });
                }
            }
        }

        self.event_log.push(BacktestEvent::EndOfBacktest);
    }

    pub fn equity_series(&self) -> (Vec<u64>, Vec<f64>) {
        let ts: Vec<u64> = self.equity_curve.iter().map(|(t, _)| *t).collect();
        let eq: Vec<f64> = self.equity_curve.iter().map(|(_, e)| *e).collect();
        (ts, eq)
    }

    pub fn returns_series(&self) -> Vec<f64> {
        let eqs: Vec<f64> = self.equity_curve.iter().map(|(_, e)| *e).collect();
        if eqs.len() < 2 { return vec![]; }
        eqs.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect()
    }

    pub fn total_return(&self) -> f64 {
        if self.equity_curve.is_empty() { return 0.0; }
        let first = self.equity_curve[0].1;
        let last = self.equity_curve.last().unwrap().1;
        (last - first) / first
    }

    pub fn num_trades(&self) -> usize {
        self.portfolio.trade_log.len()
    }

    pub fn num_bars_processed(&self) -> usize { self.bar_count }

    pub fn total_commission(&self) -> f64 {
        self.portfolio.trade_log.iter().map(|t| t.commission).sum()
    }

    pub fn total_slippage(&self) -> f64 {
        self.portfolio.trade_log.iter().map(|t| t.slippage).sum()
    }
}

/// Walk-forward backtest runner
pub struct WalkForwardRunner {
    pub train_window: usize,
    pub test_window: usize,
    pub step_size: usize,
}

impl WalkForwardRunner {
    pub fn new(train: usize, test: usize, step: usize) -> Self {
        Self { train_window: train, test_window: test, step_size: step }
    }

    pub fn generate_splits(&self, total_bars: usize) -> Vec<(usize, usize, usize, usize)> {
        let mut splits = Vec::new();
        let mut start = 0;
        while start + self.train_window + self.test_window <= total_bars {
            let train_end = start + self.train_window;
            let test_end = train_end + self.test_window;
            splits.push((start, train_end, train_end, test_end));
            start += self.step_size;
        }
        splits
    }
}

/// Expanding window runner
pub struct ExpandingWindowRunner {
    pub min_train: usize,
    pub test_window: usize,
    pub step_size: usize,
}

impl ExpandingWindowRunner {
    pub fn new(min_train: usize, test: usize, step: usize) -> Self {
        Self { min_train: min_train, test_window: test, step_size: step }
    }

    pub fn generate_splits(&self, total_bars: usize) -> Vec<(usize, usize, usize, usize)> {
        let mut splits = Vec::new();
        let mut train_end = self.min_train;
        while train_end + self.test_window <= total_bars {
            let test_end = train_end + self.test_window;
            splits.push((0, train_end, train_end, test_end));
            train_end += self.step_size;
        }
        splits
    }
}

/// Monte Carlo path generator (for equity curve simulation)
pub fn monte_carlo_paths(returns: &[f64], num_paths: usize, path_length: usize, seed: u64) -> Vec<Vec<f64>> {
    let n = returns.len();
    if n == 0 { return vec![vec![1.0; path_length]; num_paths]; }
    let mut paths = Vec::with_capacity(num_paths);
    let mut state = seed;

    for _ in 0..num_paths {
        let mut equity = vec![1.0; path_length + 1];
        for t in 0..path_length {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (state >> 32) as usize % n;
            equity[t + 1] = equity[t] * (1.0 + returns[idx]);
        }
        paths.push(equity[1..].to_vec());
    }
    paths
}

/// Multi-asset rebalancer
pub struct Rebalancer {
    pub target_weights: HashMap<String, f64>,
    pub rebalance_threshold: f64,
    pub last_rebalance_bar: usize,
    pub min_rebalance_interval: usize,
}

impl Rebalancer {
    pub fn new(targets: HashMap<String, f64>, threshold: f64, min_interval: usize) -> Self {
        Self { target_weights: targets, rebalance_threshold: threshold, last_rebalance_bar: 0, min_rebalance_interval: min_interval }
    }

    pub fn needs_rebalance(&self, current_weights: &HashMap<String, f64>, bar_count: usize) -> bool {
        if bar_count < self.last_rebalance_bar + self.min_rebalance_interval { return false; }
        for (sym, &target) in &self.target_weights {
            let current = current_weights.get(sym).copied().unwrap_or(0.0);
            if (current - target).abs() > self.rebalance_threshold { return true; }
        }
        // check for positions not in target
        for (sym, &w) in current_weights {
            if !self.target_weights.contains_key(sym) && w.abs() > self.rebalance_threshold { return true; }
        }
        false
    }

    pub fn generate_orders(&mut self, portfolio: &Portfolio, bar_count: usize) -> Vec<Order> {
        let current_weights = portfolio.weights();
        if !self.needs_rebalance(&current_weights, bar_count) { return vec![]; }
        self.last_rebalance_bar = bar_count;
        let trades = portfolio.trades_to_target(&self.target_weights);
        trades.into_iter().map(|(sym, qty)| {
            if qty > 0.0 { Order::market(0, &sym, TradeSide::Buy, qty) }
            else { Order::market(0, &sym, TradeSide::Sell, qty.abs()) }
        }).collect()
    }
}

/// TWAP order slicer
pub struct TWAPSlicer {
    pub total_quantity: f64,
    pub num_slices: usize,
    pub slices_sent: usize,
    pub symbol: String,
    pub side: TradeSide,
}

impl TWAPSlicer {
    pub fn new(symbol: &str, side: TradeSide, qty: f64, num_slices: usize) -> Self {
        Self { total_quantity: qty, num_slices, slices_sent: 0, symbol: symbol.to_string(), side }
    }

    pub fn next_slice(&mut self) -> Option<Order> {
        if self.slices_sent >= self.num_slices { return None; }
        let remaining_slices = self.num_slices - self.slices_sent;
        let remaining_qty = self.total_quantity - (self.total_quantity / self.num_slices as f64 * self.slices_sent as f64);
        let slice_qty = remaining_qty / remaining_slices as f64;
        self.slices_sent += 1;
        Some(Order::market(0, &self.symbol, self.side, slice_qty))
    }

    pub fn is_complete(&self) -> bool { self.slices_sent >= self.num_slices }
    pub fn progress(&self) -> f64 { self.slices_sent as f64 / self.num_slices as f64 }
}

/// VWAP order slicer (volume-weighted)
pub struct VWAPSlicer {
    pub total_quantity: f64,
    pub volume_profile: Vec<f64>, // expected volume fraction per period
    pub slices_sent: usize,
    pub filled_quantity: f64,
    pub symbol: String,
    pub side: TradeSide,
}

impl VWAPSlicer {
    pub fn new(symbol: &str, side: TradeSide, qty: f64, volume_profile: Vec<f64>) -> Self {
        let total: f64 = volume_profile.iter().sum();
        let normalized: Vec<f64> = volume_profile.iter().map(|&v| v / total).collect();
        Self { total_quantity: qty, volume_profile: normalized, slices_sent: 0, filled_quantity: 0.0, symbol: symbol.to_string(), side }
    }

    pub fn next_slice(&mut self) -> Option<Order> {
        if self.slices_sent >= self.volume_profile.len() { return None; }
        let frac = self.volume_profile[self.slices_sent];
        let qty = self.total_quantity * frac;
        self.slices_sent += 1;
        self.filled_quantity += qty;
        Some(Order::market(0, &self.symbol, self.side, qty))
    }

    pub fn is_complete(&self) -> bool { self.slices_sent >= self.volume_profile.len() }
}

/// Execution quality analysis
pub struct ExecutionAnalysis {
    pub orders: Vec<Order>,
}

impl ExecutionAnalysis {
    pub fn new(orders: Vec<Order>) -> Self { Self { orders } }

    pub fn total_slippage(&self) -> f64 {
        self.orders.iter().map(|o| o.slippage).sum()
    }

    pub fn total_commission(&self) -> f64 {
        self.orders.iter().map(|o| o.commission).sum()
    }

    pub fn avg_fill_price(&self) -> f64 {
        let total_notional: f64 = self.orders.iter()
            .filter(|o| o.is_filled())
            .map(|o| o.fill_price.unwrap_or(0.0) * o.filled_quantity)
            .sum();
        let total_qty: f64 = self.orders.iter()
            .filter(|o| o.is_filled())
            .map(|o| o.filled_quantity)
            .sum();
        if total_qty > 1e-10 { total_notional / total_qty } else { 0.0 }
    }

    pub fn fill_rate(&self) -> f64 {
        let filled = self.orders.iter().filter(|o| o.is_filled()).count();
        if self.orders.is_empty() { 0.0 } else { filled as f64 / self.orders.len() as f64 }
    }

    pub fn implementation_shortfall(&self, arrival_prices: &HashMap<String, f64>) -> f64 {
        let mut is = 0.0;
        for order in &self.orders {
            if !order.is_filled() { continue; }
            if let Some(&arrival) = arrival_prices.get(&order.symbol) {
                let fill = order.fill_price.unwrap_or(arrival);
                let cost = match order.side {
                    TradeSide::Buy => (fill - arrival) * order.filled_quantity,
                    TradeSide::Sell => (arrival - fill) * order.filled_quantity,
                };
                is += cost + order.commission;
            }
        }
        is
    }

    pub fn latency_stats(&self) -> (f64, f64, f64) {
        let latencies: Vec<u64> = self.orders.iter()
            .filter(|o| o.is_filled() && o.fill_time.is_some())
            .map(|o| o.fill_time.unwrap() - o.submit_time)
            .collect();
        if latencies.is_empty() { return (0.0, 0.0, 0.0); }
        let n = latencies.len() as f64;
        let mean = latencies.iter().sum::<u64>() as f64 / n;
        let max = latencies.iter().max().copied().unwrap_or(0) as f64;
        let min = latencies.iter().min().copied().unwrap_or(0) as f64;
        (mean, min, max)
    }
}

/// Event logger with structured output
pub struct EventLogger {
    pub events: Vec<(u64, String, String)>, // (timestamp, category, message)
    pub max_events: usize,
}

impl EventLogger {
    pub fn new(max: usize) -> Self { Self { events: Vec::new(), max_events: max } }

    pub fn log(&mut self, ts: u64, category: &str, msg: &str) {
        if self.events.len() < self.max_events {
            self.events.push((ts, category.to_string(), msg.to_string()));
        }
    }

    pub fn filter_category(&self, cat: &str) -> Vec<&(u64, String, String)> {
        self.events.iter().filter(|(_, c, _)| c == cat).collect()
    }

    pub fn to_csv(&self) -> String {
        let mut s = String::from("timestamp,category,message\n");
        for (ts, cat, msg) in &self.events {
            s.push_str(&format!("{},{},{}\n", ts, cat, msg));
        }
        s
    }

    pub fn len(&self) -> usize { self.events.len() }
}

/// Performance attribution over time
pub fn cumulative_attribution(returns: &[f64], factor_exposures: &[Vec<f64>], factor_returns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // returns[t], factor_exposures[t][f], factor_returns[t][f]
    // Returns cumulative contribution per factor
    let n = returns.len();
    if n == 0 { return vec![]; }
    let nf = factor_exposures[0].len();
    let mut cum = vec![vec![0.0; n]; nf];
    for t in 0..n {
        for f in 0..nf {
            let contrib = factor_exposures[t][f] * factor_returns[t][f];
            cum[f][t] = if t > 0 { cum[f][t - 1] + contrib } else { contrib };
        }
    }
    cum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let o = Order::market(1, "AAPL", TradeSide::Buy, 100.0);
        assert!(o.is_active());
        assert_eq!(o.remaining(), 100.0);
    }

    #[test]
    fn test_slippage_proportional() {
        let s = SlippageModel::Proportional(10.0); // 10 bps
        let slip = s.compute(100.0, 100.0, TradeSide::Buy, 10000.0);
        assert!((slip - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_fill_market_order() {
        let sim = FillSimulator::default_sim();
        let mut order = Order::market(1, "TEST", TradeSide::Buy, 100.0);
        let bar = Bar::new(1000, 50.0, 52.0, 49.0, 51.0, 10000.0);
        let trade = sim.try_fill(&mut order, &bar);
        assert!(trade.is_some());
        assert!(order.is_filled());
    }

    #[test]
    fn test_fill_limit_order_not_triggered() {
        let sim = FillSimulator::default_sim();
        let mut order = Order::limit(1, "TEST", TradeSide::Buy, 100.0, 45.0);
        let bar = Bar::new(1000, 50.0, 52.0, 49.0, 51.0, 10000.0);
        let trade = sim.try_fill(&mut order, &bar);
        assert!(trade.is_none());
    }

    #[test]
    fn test_fill_limit_order_triggered() {
        let sim = FillSimulator::default_sim();
        let mut order = Order::limit(1, "TEST", TradeSide::Buy, 100.0, 50.0);
        let bar = Bar::new(1000, 50.0, 52.0, 49.0, 51.0, 10000.0);
        let trade = sim.try_fill(&mut order, &bar);
        assert!(trade.is_some());
    }

    #[test]
    fn test_walk_forward() {
        let wf = WalkForwardRunner::new(100, 20, 20);
        let splits = wf.generate_splits(200);
        assert!(!splits.is_empty());
        for (ts, te, vs, ve) in &splits {
            assert_eq!(te - ts, 100);
            assert_eq!(ve - vs, 20);
        }
    }

    #[test]
    fn test_monte_carlo() {
        let returns = vec![0.01, -0.01, 0.02, -0.005, 0.015];
        let paths = monte_carlo_paths(&returns, 100, 50, 42);
        assert_eq!(paths.len(), 100);
        assert_eq!(paths[0].len(), 50);
    }

    #[test]
    fn test_commission_models() {
        assert!((CommissionModel::Fixed(5.0).compute(100.0, 50.0) - 5.0).abs() < 1e-10);
        assert!((CommissionModel::PerShare(0.01).compute(100.0, 50.0) - 1.0).abs() < 1e-10);
        assert!((CommissionModel::Percentage(0.001).compute(100.0, 50.0) - 5.0).abs() < 1e-10);
    }
}
