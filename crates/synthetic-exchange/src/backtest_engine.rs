//! backtest_engine.rs — Full event-driven backtest loop, P&L calculation,
//! transaction cost modeling, margin/leverage, performance metrics.
//!
//! Chronos / AETERNUS — production backtest engine.

use std::collections::{HashMap, VecDeque, BTreeMap};

// ── Types ────────────────────────────────────────────────────────────────────

pub type Price = f64;
pub type Qty = f64;
pub type Nanos = u64;
pub type OrderId = u64;
pub type InstrumentId = u32;

// ── PRNG ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Rng { state: u64 }
impl Rng {
    pub fn new(seed: u64) -> Self { Rng { state: seed ^ 0xfeed_face_dead_beef } }
    pub fn next_u64(&mut self) -> u64 { let mut x = self.state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.state = x; x }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Market events ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum MarketEvent {
    Quote { instrument: InstrumentId, bid: Price, ask: Price, bid_sz: Qty, ask_sz: Qty, timestamp_ns: Nanos },
    Trade { instrument: InstrumentId, price: Price, qty: Qty, is_buy: bool, timestamp_ns: Nanos },
    Bar { instrument: InstrumentId, open: Price, high: Price, low: Price, close: Price, volume: f64, timestamp_ns: Nanos },
    Dividend { instrument: InstrumentId, amount: f64, timestamp_ns: Nanos },
    Split { instrument: InstrumentId, ratio: f64, timestamp_ns: Nanos },
}

impl MarketEvent {
    pub fn timestamp(&self) -> Nanos {
        match self {
            MarketEvent::Quote { timestamp_ns, .. } => *timestamp_ns,
            MarketEvent::Trade { timestamp_ns, .. } => *timestamp_ns,
            MarketEvent::Bar { timestamp_ns, .. } => *timestamp_ns,
            MarketEvent::Dividend { timestamp_ns, .. } => *timestamp_ns,
            MarketEvent::Split { timestamp_ns, .. } => *timestamp_ns,
        }
    }

    pub fn instrument(&self) -> InstrumentId {
        match self {
            MarketEvent::Quote { instrument, .. } => *instrument,
            MarketEvent::Trade { instrument, .. } => *instrument,
            MarketEvent::Bar { instrument, .. } => *instrument,
            MarketEvent::Dividend { instrument, .. } => *instrument,
            MarketEvent::Split { instrument, .. } => *instrument,
        }
    }
}

// ── Order types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BacktestOrderType { Market, Limit, StopMarket, StopLimit }

#[derive(Debug, Clone, PartialEq)]
pub enum BacktestOrderStatus { Pending, Open, PartiallyFilled, Filled, Cancelled, Rejected, Expired }

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum BacktestSide { Buy, Sell }

#[derive(Debug, Clone)]
pub struct BacktestOrder {
    pub id: OrderId,
    pub instrument: InstrumentId,
    pub side: BacktestSide,
    pub order_type: BacktestOrderType,
    pub qty: Qty,
    pub filled_qty: Qty,
    pub price: Option<Price>,    // limit price
    pub stop_price: Option<Price>,
    pub status: BacktestOrderStatus,
    pub submitted_ns: Nanos,
    pub filled_ns: Option<Nanos>,
    pub avg_fill_price: Price,
    pub commission: f64,
}

impl BacktestOrder {
    pub fn market(id: OrderId, inst: InstrumentId, side: BacktestSide, qty: Qty, ns: Nanos) -> Self {
        BacktestOrder { id, instrument: inst, side, order_type: BacktestOrderType::Market, qty, filled_qty: 0.0, price: None, stop_price: None, status: BacktestOrderStatus::Pending, submitted_ns: ns, filled_ns: None, avg_fill_price: 0.0, commission: 0.0 }
    }

    pub fn limit(id: OrderId, inst: InstrumentId, side: BacktestSide, qty: Qty, price: Price, ns: Nanos) -> Self {
        BacktestOrder { id, instrument: inst, side, order_type: BacktestOrderType::Limit, qty, filled_qty: 0.0, price: Some(price), stop_price: None, status: BacktestOrderStatus::Pending, submitted_ns: ns, filled_ns: None, avg_fill_price: 0.0, commission: 0.0 }
    }
}

// ── Fill ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BacktestFill {
    pub order_id: OrderId,
    pub instrument: InstrumentId,
    pub side: BacktestSide,
    pub qty: Qty,
    pub price: Price,
    pub commission: f64,
    pub slippage: f64,
    pub timestamp_ns: Nanos,
    pub is_partial: bool,
}

// ── Transaction cost model ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TransactionCostModel {
    pub commission_per_share: f64,
    pub commission_per_trade: f64,
    pub commission_pct_notional: f64,  // bps
    pub slippage_bps: f64,
    pub market_impact_sqrt_factor: f64,
    pub min_commission: f64,
    pub adv: HashMap<InstrumentId, f64>,
}

impl Default for TransactionCostModel {
    fn default() -> Self {
        TransactionCostModel {
            commission_per_share: 0.005,
            commission_per_trade: 1.0,
            commission_pct_notional: 0.0,
            slippage_bps: 0.5,
            market_impact_sqrt_factor: 0.1,
            min_commission: 1.0,
            adv: HashMap::new(),
        }
    }
}

impl TransactionCostModel {
    pub fn institutional() -> Self {
        TransactionCostModel { commission_per_share: 0.001, commission_per_trade: 0.0, commission_pct_notional: 0.5, slippage_bps: 0.2, market_impact_sqrt_factor: 0.05, min_commission: 0.0, adv: HashMap::new() }
    }

    pub fn retail() -> Self {
        TransactionCostModel { commission_per_share: 0.0, commission_per_trade: 0.0, commission_pct_notional: 0.0, slippage_bps: 2.0, market_impact_sqrt_factor: 0.3, min_commission: 0.0, adv: HashMap::new() }
    }

    pub fn commission(&self, qty: Qty, price: Price) -> f64 {
        let c = self.commission_per_share * qty
            + self.commission_per_trade
            + self.commission_pct_notional / 10_000.0 * qty * price;
        c.max(self.min_commission)
    }

    pub fn slippage(&self, qty: Qty, price: Price, adv: f64) -> f64 {
        let base = price * self.slippage_bps / 10_000.0;
        let impact = if adv > 0.0 {
            price * self.market_impact_sqrt_factor * (qty / adv).sqrt()
        } else { 0.0 };
        base + impact
    }

    pub fn total_cost(&self, qty: Qty, price: Price, adv: f64) -> f64 {
        self.commission(qty, price) + self.slippage(qty, price, adv) * qty
    }
}

// ── Position tracker ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct Position {
    pub instrument: InstrumentId,
    pub qty: f64,              // signed (positive = long, negative = short)
    pub avg_entry_price: Price,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_commissions: f64,
    pub total_slippage: f64,
    pub trade_count: u64,
    pub last_price: Price,
}

impl Position {
    pub fn new(instrument: InstrumentId) -> Self { Position { instrument, ..Default::default() } }

    pub fn is_flat(&self) -> bool { self.qty.abs() < 0.01 }
    pub fn is_long(&self) -> bool { self.qty > 0.0 }
    pub fn is_short(&self) -> bool { self.qty < 0.0 }
    pub fn notional(&self) -> f64 { self.qty.abs() * self.last_price }

    pub fn update_price(&mut self, price: Price) {
        self.last_price = price;
        if self.qty.abs() > 0.01 {
            self.unrealized_pnl = (price - self.avg_entry_price) * self.qty;
        }
    }

    pub fn apply_fill(&mut self, fill: &BacktestFill) {
        let signed_qty = match fill.side { BacktestSide::Buy => fill.qty, BacktestSide::Sell => -fill.qty };
        let prev_qty = self.qty;

        // Position-weighted average entry price
        if (prev_qty >= 0.0 && signed_qty > 0.0) || (prev_qty <= 0.0 && signed_qty < 0.0) {
            // Adding to existing position
            let total = prev_qty.abs() + fill.qty;
            self.avg_entry_price = (prev_qty.abs() * self.avg_entry_price + fill.qty * fill.price) / total;
        } else if fill.qty.abs() >= self.qty.abs() {
            // Reversing position
            self.realized_pnl += (fill.price - self.avg_entry_price) * prev_qty;
            let residual = signed_qty + prev_qty;
            if residual.abs() > 0.01 { self.avg_entry_price = fill.price; }
        } else {
            // Partial close
            self.realized_pnl += (fill.price - self.avg_entry_price) * (-signed_qty);
        }

        self.qty += signed_qty;
        self.total_commissions += fill.commission;
        self.total_slippage += fill.slippage * fill.qty;
        self.trade_count += 1;
        self.last_price = fill.price;
        self.update_price(fill.price);
    }

    pub fn total_pnl(&self) -> f64 { self.realized_pnl + self.unrealized_pnl - self.total_commissions }
}

// ── Margin model ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MarginModel {
    pub initial_margin_pct: f64,   // fraction of notional
    pub maintenance_margin_pct: f64,
    pub leverage: f64,
}

impl Default for MarginModel {
    fn default() -> Self { MarginModel { initial_margin_pct: 0.05, maintenance_margin_pct: 0.03, leverage: 20.0 } }
}

impl MarginModel {
    pub fn required_initial_margin(&self, notional: f64) -> f64 { notional * self.initial_margin_pct }
    pub fn maintenance_margin(&self, notional: f64) -> f64 { notional * self.maintenance_margin_pct }
    pub fn max_position_notional(&self, capital: f64) -> f64 { capital * self.leverage }
    pub fn is_margin_call(&self, equity: f64, notional: f64) -> bool { equity < self.maintenance_margin(notional) }
}

// ── P&L tracker ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PnLTracker {
    pub equity_curve: Vec<(Nanos, f64)>,
    pub daily_returns: Vec<f64>,
    pub initial_capital: f64,
    pub peak_equity: f64,
    pub current_equity: f64,
    pub max_drawdown: f64,
    pub max_drawdown_start: Nanos,
    pub max_drawdown_end: Nanos,
    pub total_commissions: f64,
    pub total_slippage: f64,
    pub winning_trades: u64,
    pub losing_trades: u64,
}

impl PnLTracker {
    pub fn new(initial_capital: f64) -> Self {
        PnLTracker {
            equity_curve: Vec::new(),
            daily_returns: Vec::new(),
            initial_capital,
            peak_equity: initial_capital,
            current_equity: initial_capital,
            max_drawdown: 0.0,
            max_drawdown_start: 0,
            max_drawdown_end: 0,
            total_commissions: 0.0,
            total_slippage: 0.0,
            winning_trades: 0,
            losing_trades: 0,
        }
    }

    pub fn update(&mut self, equity: f64, timestamp_ns: Nanos) {
        self.current_equity = equity;
        self.equity_curve.push((timestamp_ns, equity));
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        let dd = (self.peak_equity - equity) / self.peak_equity.max(1.0);
        if dd > self.max_drawdown {
            self.max_drawdown = dd;
            self.max_drawdown_end = timestamp_ns;
        }
    }

    pub fn record_trade(&mut self, pnl: f64) {
        if pnl > 0.0 { self.winning_trades += 1; } else { self.losing_trades += 1; }
    }

    pub fn total_return(&self) -> f64 {
        (self.current_equity - self.initial_capital) / self.initial_capital
    }

    pub fn win_rate(&self) -> f64 {
        let total = self.winning_trades + self.losing_trades;
        if total == 0 { return 0.5; }
        self.winning_trades as f64 / total as f64
    }

    pub fn compute_daily_returns(&mut self) {
        if self.equity_curve.len() < 2 { return; }
        // Approximate: use whole series as daily buckets
        let n = self.equity_curve.len();
        let step = (n / 252).max(1);
        let mut returns = Vec::new();
        for i in (step..n).step_by(step) {
            let prev = self.equity_curve[i - step].1;
            let curr = self.equity_curve[i].1;
            if prev > 0.0 { returns.push((curr - prev) / prev); }
        }
        self.daily_returns = returns;
    }
}

// ── Performance metrics ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub annualized_vol: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration_days: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_trade_pnl: f64,
    pub total_trades: u64,
    pub total_commissions: f64,
    pub total_slippage: f64,
}

impl PerformanceMetrics {
    pub fn compute(tracker: &PnLTracker, trading_days: f64) -> Self {
        let returns = &tracker.daily_returns;
        let n = returns.len() as f64;
        if n < 2.0 || trading_days < 1.0 {
            return PerformanceMetrics {
                total_return: tracker.total_return(),
                max_drawdown: tracker.max_drawdown,
                win_rate: tracker.win_rate(),
                total_commissions: tracker.total_commissions,
                total_slippage: tracker.total_slippage,
                total_trades: tracker.winning_trades + tracker.losing_trades,
                ..Default::default()
            };
        }

        let mean_ret: f64 = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        let annualized_return = (1.0 + tracker.total_return()).powf(252.0 / trading_days) - 1.0;
        let annualized_vol = std_dev * 252f64.sqrt();

        let risk_free_daily = 0.05 / 252.0;
        let sharpe = if annualized_vol > 1e-12 {
            (annualized_return - 0.05) / annualized_vol
        } else { 0.0 };

        // Sortino: downside deviation
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < risk_free_daily).cloned().collect();
        let downside_variance = if downside_returns.len() > 1 {
            downside_returns.iter().map(|r| (r - risk_free_daily).powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else { variance };
        let downside_std = (downside_variance * 252.0).sqrt();
        let sortino = if downside_std > 1e-12 { (annualized_return - 0.05) / downside_std } else { 0.0 };

        let calmar = if tracker.max_drawdown > 1e-12 { annualized_return / tracker.max_drawdown } else { 0.0 };

        let total_trades = tracker.winning_trades + tracker.losing_trades;

        PerformanceMetrics {
            total_return: tracker.total_return(),
            annualized_return,
            annualized_vol,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            calmar_ratio: calmar,
            max_drawdown: tracker.max_drawdown,
            max_drawdown_duration_days: 0.0,
            win_rate: tracker.win_rate(),
            profit_factor: 0.0,
            avg_trade_pnl: 0.0,
            total_trades,
            total_commissions: tracker.total_commissions,
            total_slippage: tracker.total_slippage,
        }
    }
}

impl std::fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "Return={:.1}% Ann={:.1}% Vol={:.1}% Sharpe={:.2} Sortino={:.2} Calmar={:.2} MDD={:.1}% WinRate={:.1}% N={}",
            self.total_return * 100.0, self.annualized_return * 100.0, self.annualized_vol * 100.0,
            self.sharpe_ratio, self.sortino_ratio, self.calmar_ratio,
            self.max_drawdown * 100.0, self.win_rate * 100.0, self.total_trades)
    }
}

// ── Strategy interface ────────────────────────────────────────────────────────

pub trait Strategy: Send {
    fn on_event(&mut self, event: &MarketEvent, ctx: &mut BacktestContext) -> Vec<BacktestOrder>;
    fn on_fill(&mut self, fill: &BacktestFill, ctx: &mut BacktestContext);
    fn name(&self) -> &str;
}

pub struct BacktestContext {
    pub current_ns: Nanos,
    pub positions: HashMap<InstrumentId, Position>,
    pub cash: f64,
    pub equity: f64,
    pub margin_model: MarginModel,
    pub last_prices: HashMap<InstrumentId, Price>,
    pub order_counter: u64,
}

impl BacktestContext {
    pub fn new(initial_cash: f64) -> Self {
        BacktestContext {
            current_ns: 0,
            positions: HashMap::new(),
            cash: initial_cash,
            equity: initial_cash,
            margin_model: MarginModel::default(),
            last_prices: HashMap::new(),
            order_counter: 0,
        }
    }

    pub fn next_order_id(&mut self) -> OrderId { self.order_counter += 1; self.order_counter }

    pub fn position(&self, instrument: InstrumentId) -> f64 {
        self.positions.get(&instrument).map(|p| p.qty).unwrap_or(0.0)
    }

    pub fn last_price(&self, instrument: InstrumentId) -> Option<Price> {
        self.last_prices.get(&instrument).copied()
    }

    pub fn update_equity(&mut self) {
        let unrealized: f64 = self.positions.values()
            .map(|p| p.unrealized_pnl)
            .sum();
        let realized: f64 = self.positions.values()
            .map(|p| p.realized_pnl - p.total_commissions)
            .sum();
        self.equity = self.cash + unrealized + realized;
    }

    pub fn gross_notional(&self) -> f64 {
        self.positions.values().map(|p| p.notional()).sum()
    }
}

// ── Backtest engine ───────────────────────────────────────────────────────────

pub struct BacktestEngine {
    pub context: BacktestContext,
    pub strategy: Box<dyn Strategy>,
    pub cost_model: TransactionCostModel,
    pub pnl_tracker: PnLTracker,
    pub fills: Vec<BacktestFill>,
    pub orders: VecDeque<BacktestOrder>,
    pub order_history: Vec<BacktestOrder>,
    pub events_processed: u64,
    pub slippage_model_enabled: bool,
}

impl BacktestEngine {
    pub fn new(strategy: Box<dyn Strategy>, initial_capital: f64) -> Self {
        BacktestEngine {
            context: BacktestContext::new(initial_capital),
            strategy,
            cost_model: TransactionCostModel::default(),
            pnl_tracker: PnLTracker::new(initial_capital),
            fills: Vec::new(),
            orders: VecDeque::new(),
            order_history: Vec::new(),
            events_processed: 0,
            slippage_model_enabled: true,
        }
    }

    pub fn with_cost_model(mut self, model: TransactionCostModel) -> Self {
        self.cost_model = model; self
    }

    pub fn run(&mut self, events: &[MarketEvent]) -> BacktestResult {
        for event in events {
            self.process_event(event);
        }
        self.finalize_result()
    }

    fn process_event(&mut self, event: &MarketEvent) {
        self.context.current_ns = event.timestamp();
        self.events_processed += 1;

        // Update prices
        match event {
            MarketEvent::Quote { instrument, bid, ask, .. } => {
                let mid = (bid + ask) / 2.0;
                self.context.last_prices.insert(*instrument, mid);
                if let Some(pos) = self.context.positions.get_mut(instrument) {
                    pos.update_price(mid);
                }
            }
            MarketEvent::Trade { instrument, price, .. } => {
                self.context.last_prices.insert(*instrument, *price);
                if let Some(pos) = self.context.positions.get_mut(instrument) {
                    pos.update_price(*price);
                }
            }
            MarketEvent::Bar { instrument, close, .. } => {
                self.context.last_prices.insert(*instrument, *close);
                if let Some(pos) = self.context.positions.get_mut(instrument) {
                    pos.update_price(*close);
                }
            }
            MarketEvent::Dividend { instrument, amount, .. } => {
                let shares = self.context.position(*instrument);
                self.context.cash += shares * amount;
            }
            MarketEvent::Split { instrument, ratio, .. } => {
                if let Some(pos) = self.context.positions.get_mut(instrument) {
                    pos.qty *= ratio;
                    pos.avg_entry_price /= ratio;
                }
            }
        }

        // Try to fill pending orders
        self.attempt_fills(event);

        // Call strategy
        let new_orders = self.strategy.on_event(event, &mut self.context);
        for order in new_orders {
            self.orders.push_back(order);
        }

        // Update equity
        self.context.update_equity();
        self.pnl_tracker.update(self.context.equity, event.timestamp());
    }

    fn attempt_fills(&mut self, event: &MarketEvent) {
        let (fill_price, instrument) = match event {
            MarketEvent::Quote { instrument, bid, ask, .. } => {
                (Some((*bid, *ask)), *instrument)
            }
            MarketEvent::Trade { instrument, price, .. } => {
                (Some((*price, *price)), *instrument)
            }
            MarketEvent::Bar { instrument, open, close, .. } => {
                (Some((*open, *close)), *instrument)
            }
            _ => return,
        };

        let (bid, ask) = match fill_price { Some(p) => p, None => return };

        let mut filled_ids = Vec::new();
        let adv = self.cost_model.adv.get(&instrument).copied().unwrap_or(1_000_000.0);

        for (i, order) in self.orders.iter_mut().enumerate() {
            if order.instrument != instrument { continue; }
            if !matches!(order.status, BacktestOrderStatus::Pending | BacktestOrderStatus::Open) { continue; }

            let execution_price = match (&order.order_type, order.side) {
                (BacktestOrderType::Market, BacktestSide::Buy) => Some(ask),
                (BacktestOrderType::Market, BacktestSide::Sell) => Some(bid),
                (BacktestOrderType::Limit, BacktestSide::Buy) => {
                    if let Some(lp) = order.price { if ask <= lp { Some(ask) } else { None } } else { None }
                }
                (BacktestOrderType::Limit, BacktestSide::Sell) => {
                    if let Some(lp) = order.price { if bid >= lp { Some(bid) } else { None } } else { None }
                }
                _ => None,
            };

            if let Some(exec_price) = execution_price {
                let slippage = if self.slippage_model_enabled {
                    self.cost_model.slippage(order.qty, exec_price, adv)
                } else { 0.0 };
                let actual_price = match order.side {
                    BacktestSide::Buy => exec_price + slippage,
                    BacktestSide::Sell => exec_price - slippage,
                };
                let commission = self.cost_model.commission(order.qty, actual_price);

                let fill = BacktestFill {
                    order_id: order.id,
                    instrument: order.instrument,
                    side: order.side,
                    qty: order.qty,
                    price: actual_price,
                    commission,
                    slippage,
                    timestamp_ns: self.context.current_ns,
                    is_partial: false,
                };

                order.filled_qty = order.qty;
                order.avg_fill_price = actual_price;
                order.status = BacktestOrderStatus::Filled;
                order.filled_ns = Some(self.context.current_ns);
                order.commission = commission;

                // Update position
                let pos = self.context.positions.entry(instrument).or_insert_with(|| Position::new(instrument));
                pos.apply_fill(&fill);
                self.pnl_tracker.record_trade(pos.realized_pnl);
                self.pnl_tracker.total_commissions += commission;
                self.context.cash -= match order.side { BacktestSide::Buy => actual_price * order.qty + commission, BacktestSide::Sell => -(actual_price * order.qty - commission) };

                self.strategy.on_fill(&fill, &mut self.context);
                self.fills.push(fill);
                filled_ids.push(i);
            }
        }

        // Move filled orders to history
        for &i in filled_ids.iter().rev() {
            if i < self.orders.len() {
                let o = self.orders.remove(i).unwrap();
                self.order_history.push(o);
            }
        }
    }

    fn finalize_result(&mut self) -> BacktestResult {
        self.pnl_tracker.compute_daily_returns();
        let trading_days = if self.pnl_tracker.equity_curve.len() > 1 {
            let t_start = self.pnl_tracker.equity_curve.first().map(|(t, _)| *t).unwrap_or(0);
            let t_end = self.pnl_tracker.equity_curve.last().map(|(t, _)| *t).unwrap_or(0);
            let duration_ns = t_end.saturating_sub(t_start);
            duration_ns as f64 / (86_400_000_000_000.0)
        } else { 1.0 };

        let metrics = PerformanceMetrics::compute(&self.pnl_tracker, trading_days);

        BacktestResult {
            strategy_name: self.strategy.name().to_string(),
            initial_capital: self.pnl_tracker.initial_capital,
            final_equity: self.context.equity,
            metrics,
            equity_curve: self.pnl_tracker.equity_curve.clone(),
            events_processed: self.events_processed,
            total_fills: self.fills.len(),
            total_orders: self.order_history.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub strategy_name: String,
    pub initial_capital: f64,
    pub final_equity: f64,
    pub metrics: PerformanceMetrics,
    pub equity_curve: Vec<(Nanos, f64)>,
    pub events_processed: u64,
    pub total_fills: usize,
    pub total_orders: usize,
}

impl std::fmt::Display for BacktestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BacktestResult[{}]: capital={:.0}->{:.0} fills={} events={} | {}",
            self.strategy_name, self.initial_capital, self.final_equity,
            self.total_fills, self.events_processed, self.metrics)
    }
}

// ── Example strategy: buy-and-hold ───────────────────────────────────────────

pub struct BuyAndHoldStrategy {
    pub instrument: InstrumentId,
    pub invested: bool,
}

impl BuyAndHoldStrategy {
    pub fn new(instrument: InstrumentId) -> Self { BuyAndHoldStrategy { instrument, invested: false } }
}

impl Strategy for BuyAndHoldStrategy {
    fn name(&self) -> &str { "buy_and_hold" }

    fn on_event(&mut self, event: &MarketEvent, ctx: &mut BacktestContext) -> Vec<BacktestOrder> {
        if self.invested || event.instrument() != self.instrument { return Vec::new(); }
        if let Some(price) = ctx.last_price(self.instrument) {
            let qty = (ctx.cash / price).floor();
            if qty > 0.0 {
                self.invested = true;
                let id = ctx.next_order_id();
                return vec![BacktestOrder::market(id, self.instrument, BacktestSide::Buy, qty, ctx.current_ns)];
            }
        }
        Vec::new()
    }

    fn on_fill(&mut self, _fill: &BacktestFill, _ctx: &mut BacktestContext) {}
}

// ── Example strategy: simple moving average crossover ────────────────────────

pub struct SmaCrossoverStrategy {
    pub instrument: InstrumentId,
    fast: usize,
    slow: usize,
    prices: VecDeque<Price>,
    position: f64,
}

impl SmaCrossoverStrategy {
    pub fn new(instrument: InstrumentId, fast: usize, slow: usize) -> Self {
        SmaCrossoverStrategy { instrument, fast, slow, prices: VecDeque::new(), position: 0.0 }
    }

    fn sma(&self, n: usize) -> Option<f64> {
        if self.prices.len() < n { return None; }
        let sum: f64 = self.prices.iter().rev().take(n).sum();
        Some(sum / n as f64)
    }
}

impl Strategy for SmaCrossoverStrategy {
    fn name(&self) -> &str { "sma_crossover" }

    fn on_event(&mut self, event: &MarketEvent, ctx: &mut BacktestContext) -> Vec<BacktestOrder> {
        let price = match event {
            MarketEvent::Bar { close, instrument, .. } if *instrument == self.instrument => *close,
            MarketEvent::Trade { price, instrument, .. } if *instrument == self.instrument => *price,
            _ => return Vec::new(),
        };

        self.prices.push_back(price);
        if self.prices.len() > self.slow * 2 { self.prices.pop_front(); }

        let fast_sma = self.sma(self.fast);
        let slow_sma = self.sma(self.slow);

        match (fast_sma, slow_sma) {
            (Some(f), Some(s)) => {
                let signal: f64 = if f > s { 1.0 } else { -1.0 };
                if (signal - self.position).abs() > 0.5 {
                    let qty = ctx.cash / price * 0.9;
                    if qty < 1.0 { return Vec::new(); }
                    let id = ctx.next_order_id();
                    let side = if signal > 0.0 { BacktestSide::Buy } else { BacktestSide::Sell };
                    // Close existing first
                    self.position = signal;
                    return vec![BacktestOrder::market(id, self.instrument, side, qty.floor(), ctx.current_ns)];
                }
                Vec::new()
            }
            _ => Vec::new()
        }
    }

    fn on_fill(&mut self, fill: &BacktestFill, _ctx: &mut BacktestContext) {}
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bar_events(instrument: InstrumentId, n: usize, initial_price: f64, daily_ret: f64) -> Vec<MarketEvent> {
        let mut price = initial_price;
        let mut events = Vec::new();
        for i in 0..n {
            let ns = i as u64 * 86_400_000_000_000u64;
            events.push(MarketEvent::Bar {
                instrument, open: price, high: price * 1.01, low: price * 0.99,
                close: price, volume: 1_000_000.0, timestamp_ns: ns,
            });
            price *= 1.0 + daily_ret;
        }
        events
    }

    #[test]
    fn test_buy_and_hold_backtest() {
        let events = make_bar_events(1, 252, 100.0, 0.001);
        let strategy = Box::new(BuyAndHoldStrategy::new(1));
        let mut engine = BacktestEngine::new(strategy, 100_000.0);
        let result = engine.run(&events);
        assert!(result.final_equity > 100_000.0, "final_equity={}", result.final_equity);
        assert!(result.metrics.total_return > 0.0);
        println!("{}", result);
    }

    #[test]
    fn test_sma_crossover_backtest() {
        let mut events = make_bar_events(1, 252, 100.0, 0.0005);
        let strategy = Box::new(SmaCrossoverStrategy::new(1, 5, 20));
        let mut engine = BacktestEngine::new(strategy, 50_000.0);
        let result = engine.run(&events);
        assert!(result.events_processed == 252);
        assert!(result.final_equity > 0.0);
    }

    #[test]
    fn test_transaction_cost_commission() {
        let model = TransactionCostModel::default();
        let comm = model.commission(1000.0, 50.0);
        assert!(comm >= model.min_commission);
    }

    #[test]
    fn test_transaction_cost_slippage() {
        let model = TransactionCostModel::default();
        let slip = model.slippage(10_000.0, 100.0, 1_000_000.0);
        assert!(slip > 0.0);
    }

    #[test]
    fn test_position_apply_fill_long() {
        let mut pos = Position::new(1);
        let fill = BacktestFill { order_id: 1, instrument: 1, side: BacktestSide::Buy, qty: 100.0, price: 50.0, commission: 1.0, slippage: 0.0, timestamp_ns: 0, is_partial: false };
        pos.apply_fill(&fill);
        assert!((pos.qty - 100.0).abs() < 0.01);
        assert!((pos.avg_entry_price - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_position_realized_pnl() {
        let mut pos = Position::new(1);
        let buy = BacktestFill { order_id: 1, instrument: 1, side: BacktestSide::Buy, qty: 100.0, price: 50.0, commission: 0.0, slippage: 0.0, timestamp_ns: 0, is_partial: false };
        let sell = BacktestFill { order_id: 2, instrument: 1, side: BacktestSide::Sell, qty: 100.0, price: 55.0, commission: 0.0, slippage: 0.0, timestamp_ns: 1, is_partial: false };
        pos.apply_fill(&buy);
        pos.apply_fill(&sell);
        assert!((pos.realized_pnl - 500.0).abs() < 0.01, "realized_pnl={}", pos.realized_pnl);
    }

    #[test]
    fn test_performance_metrics_sharpe() {
        let mut tracker = PnLTracker::new(100_000.0);
        let mut rng = Rng::new(42);
        let mut equity = 100_000.0;
        for i in 0..252 {
            equity *= 1.0 + 0.0004 + rng.next_normal() * 0.01;
            tracker.update(equity, i as u64 * 86_400_000_000_000);
        }
        tracker.compute_daily_returns();
        let metrics = PerformanceMetrics::compute(&tracker, 252.0);
        assert!(metrics.sharpe_ratio.is_finite());
        assert!(metrics.annualized_vol > 0.0);
    }

    #[test]
    fn test_margin_model() {
        let model = MarginModel::default();
        let notional = 100_000.0;
        let init_margin = model.required_initial_margin(notional);
        let maint_margin = model.maintenance_margin(notional);
        assert!(init_margin > maint_margin);
        assert!(!model.is_margin_call(init_margin, notional));
        assert!(model.is_margin_call(maint_margin * 0.5, notional));
    }

    #[test]
    fn test_pnl_tracker_drawdown() {
        let mut tracker = PnLTracker::new(100_000.0);
        tracker.update(110_000.0, 1);
        tracker.update(95_000.0, 2);
        assert!((tracker.max_drawdown - (110_000.0 - 95_000.0) / 110_000.0).abs() < 0.001);
    }

    #[test]
    fn test_backtest_result_display() {
        let events = make_bar_events(1, 50, 200.0, 0.001);
        let strategy = Box::new(BuyAndHoldStrategy::new(1));
        let mut engine = BacktestEngine::new(strategy, 200_000.0);
        let result = engine.run(&events);
        let display = format!("{}", result);
        assert!(display.contains("buy_and_hold"));
    }

    #[test]
    fn test_fill_tracking() {
        let events = make_bar_events(1, 100, 50.0, 0.002);
        let strategy = Box::new(BuyAndHoldStrategy::new(1));
        let mut engine = BacktestEngine::new(strategy, 10_000.0);
        let result = engine.run(&events);
        assert!(result.total_fills > 0, "expected fills");
    }
}
