// portfolio.rs — Portfolio state, position management, P&L attribution, trade log
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// A single trade record
#[derive(Clone, Debug)]
pub struct Trade {
    pub timestamp: u64,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
    pub slippage: f64,
    pub order_id: u64,
    pub pnl: f64,
    pub tag: String,
}

impl Trade {
    pub fn notional(&self) -> f64 { self.quantity * self.price }
    pub fn total_cost(&self) -> f64 { self.notional() + self.commission + self.slippage }
    pub fn is_buy(&self) -> bool { self.side == TradeSide::Buy }
    pub fn is_sell(&self) -> bool { self.side == TradeSide::Sell }
    pub fn signed_quantity(&self) -> f64 {
        match self.side { TradeSide::Buy => self.quantity, TradeSide::Sell => -self.quantity }
    }
}

/// Position in a single asset
#[derive(Clone, Debug)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_cost: f64,
    pub market_price: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_bought: f64,
    pub total_sold: f64,
    pub num_trades: usize,
    pub max_quantity: f64,
    pub entry_time: u64,
    pub last_trade_time: u64,
}

impl Position {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            quantity: 0.0, avg_cost: 0.0, market_price: 0.0,
            realized_pnl: 0.0, unrealized_pnl: 0.0,
            total_bought: 0.0, total_sold: 0.0,
            num_trades: 0, max_quantity: 0.0,
            entry_time: 0, last_trade_time: 0,
        }
    }

    pub fn market_value(&self) -> f64 { self.quantity * self.market_price }
    pub fn cost_basis(&self) -> f64 { self.quantity * self.avg_cost }
    pub fn total_pnl(&self) -> f64 { self.realized_pnl + self.unrealized_pnl }
    pub fn is_long(&self) -> bool { self.quantity > 0.0 }
    pub fn is_short(&self) -> bool { self.quantity < 0.0 }
    pub fn is_flat(&self) -> bool { self.quantity.abs() < 1e-10 }
    pub fn notional_exposure(&self) -> f64 { self.quantity.abs() * self.market_price }

    pub fn return_pct(&self) -> f64 {
        let cost = self.cost_basis().abs();
        if cost < 1e-10 { 0.0 } else { self.total_pnl() / cost }
    }

    pub fn update_trade(&mut self, side: TradeSide, quantity: f64, price: f64, timestamp: u64) {
        self.num_trades += 1;
        self.last_trade_time = timestamp;

        let signed_qty = match side { TradeSide::Buy => quantity, TradeSide::Sell => -quantity };

        match side {
            TradeSide::Buy => self.total_bought += quantity * price,
            TradeSide::Sell => self.total_sold += quantity * price,
        }

        let new_qty = self.quantity + signed_qty;

        if (self.quantity >= 0.0 && signed_qty > 0.0) || (self.quantity <= 0.0 && signed_qty < 0.0) {
            // Increasing position
            let total_cost = self.quantity.abs() * self.avg_cost + quantity * price;
            let total_qty = self.quantity.abs() + quantity;
            self.avg_cost = if total_qty > 1e-10 { total_cost / total_qty } else { price };
        } else {
            // Reducing/reversing position
            let close_qty = quantity.min(self.quantity.abs());
            let realized = close_qty * (price - self.avg_cost) * if self.quantity > 0.0 { 1.0 } else { -1.0 };
            self.realized_pnl += realized;

            if new_qty.abs() > self.quantity.abs() {
                // Reversal
                self.avg_cost = price;
            }
        }

        self.quantity = new_qty;
        if self.quantity.abs() < 1e-10 { self.quantity = 0.0; }
        self.max_quantity = self.max_quantity.max(self.quantity.abs());

        if self.entry_time == 0 && !self.is_flat() {
            self.entry_time = timestamp;
        }
        if self.is_flat() { self.entry_time = 0; }
    }

    pub fn mark(&mut self, price: f64) {
        self.market_price = price;
        self.unrealized_pnl = self.quantity * (price - self.avg_cost);
    }
}

/// Full portfolio state
#[derive(Clone, Debug)]
pub struct Portfolio {
    pub cash: f64,
    pub initial_capital: f64,
    pub positions: HashMap<String, Position>,
    pub trade_log: Vec<Trade>,
    pub equity_history: Vec<f64>,
    pub cash_history: Vec<f64>,
    pub high_watermark: f64,
    pub drawdown: f64,
    pub max_drawdown: f64,
    pub total_commission: f64,
    pub total_slippage: f64,
}

impl Portfolio {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            cash: initial_capital,
            initial_capital,
            positions: HashMap::new(),
            trade_log: Vec::new(),
            equity_history: Vec::new(),
            cash_history: Vec::new(),
            high_watermark: initial_capital,
            drawdown: 0.0,
            max_drawdown: 0.0,
            total_commission: 0.0,
            total_slippage: 0.0,
        }
    }

    pub fn total_equity(&self) -> f64 {
        self.cash + self.positions.values().map(|p| p.market_value()).sum::<f64>()
    }

    pub fn total_exposure(&self) -> f64 {
        self.positions.values().map(|p| p.notional_exposure()).sum()
    }

    pub fn net_exposure(&self) -> f64 {
        self.positions.values().map(|p| p.market_value()).sum()
    }

    pub fn long_exposure(&self) -> f64 {
        self.positions.values().filter(|p| p.is_long()).map(|p| p.market_value()).sum()
    }

    pub fn short_exposure(&self) -> f64 {
        self.positions.values().filter(|p| p.is_short()).map(|p| p.market_value().abs()).sum()
    }

    pub fn leverage(&self) -> f64 {
        let eq = self.total_equity();
        if eq.abs() < 1e-10 { 0.0 } else { self.total_exposure() / eq }
    }

    pub fn gross_leverage(&self) -> f64 { self.leverage() }

    pub fn net_leverage(&self) -> f64 {
        let eq = self.total_equity();
        if eq.abs() < 1e-10 { 0.0 } else { self.net_exposure() / eq }
    }

    pub fn num_positions(&self) -> usize {
        self.positions.values().filter(|p| !p.is_flat()).count()
    }

    pub fn num_long(&self) -> usize { self.positions.values().filter(|p| p.is_long()).count() }
    pub fn num_short(&self) -> usize { self.positions.values().filter(|p| p.is_short()).count() }

    pub fn position(&self, symbol: &str) -> Option<&Position> { self.positions.get(symbol) }
    pub fn position_qty(&self, symbol: &str) -> f64 {
        self.positions.get(symbol).map_or(0.0, |p| p.quantity)
    }

    pub fn process_trade(&mut self, trade: &Trade) {
        let entry = self.positions.entry(trade.symbol.clone()).or_insert_with(|| Position::new(&trade.symbol));
        entry.update_trade(trade.side, trade.quantity, trade.price, trade.timestamp);

        // Cash update
        let cash_change = match trade.side {
            TradeSide::Buy => -(trade.quantity * trade.price + trade.commission),
            TradeSide::Sell => trade.quantity * trade.price - trade.commission,
        };
        self.cash += cash_change;
        self.total_commission += trade.commission;
        self.total_slippage += trade.slippage;
        self.trade_log.push(trade.clone());
    }

    pub fn mark_to_market(&mut self, prices: &HashMap<String, f64>) {
        for (sym, pos) in self.positions.iter_mut() {
            if let Some(&price) = prices.get(sym.as_str()) {
                pos.mark(price);
            }
        }
        let equity = self.total_equity();
        self.equity_history.push(equity);
        self.cash_history.push(self.cash);

        // Update drawdown
        if equity > self.high_watermark { self.high_watermark = equity; }
        self.drawdown = (self.high_watermark - equity) / self.high_watermark;
        if self.drawdown > self.max_drawdown { self.max_drawdown = self.drawdown; }
    }

    pub fn realized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.realized_pnl).sum()
    }

    pub fn unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl).sum()
    }

    pub fn total_pnl(&self) -> f64 { self.realized_pnl() + self.unrealized_pnl() }

    pub fn total_return(&self) -> f64 {
        (self.total_equity() - self.initial_capital) / self.initial_capital
    }

    pub fn returns(&self) -> Vec<f64> {
        if self.equity_history.len() < 2 { return vec![]; }
        self.equity_history.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect()
    }

    pub fn log_returns(&self) -> Vec<f64> {
        if self.equity_history.len() < 2 { return vec![]; }
        self.equity_history.windows(2).map(|w| {
            if w[0] > 0.0 && w[1] > 0.0 { (w[1] / w[0]).ln() } else { 0.0 }
        }).collect()
    }

    /// Position weights as fraction of equity
    pub fn weights(&self) -> HashMap<String, f64> {
        let eq = self.total_equity();
        if eq.abs() < 1e-10 { return HashMap::new(); }
        self.positions.iter()
            .filter(|(_, p)| !p.is_flat())
            .map(|(sym, p)| (sym.clone(), p.market_value() / eq))
            .collect()
    }

    /// Target portfolio: compute trades needed to reach target weights
    pub fn trades_to_target(&self, target_weights: &HashMap<String, f64>) -> Vec<(String, f64)> {
        let eq = self.total_equity();
        let mut trades = Vec::new();
        // close positions not in target
        for (sym, pos) in &self.positions {
            if !pos.is_flat() && !target_weights.contains_key(sym.as_str()) {
                trades.push((sym.clone(), -pos.quantity));
            }
        }
        // adjust to target
        for (sym, &target_w) in target_weights {
            let target_val = eq * target_w;
            let current_val = self.positions.get(sym).map_or(0.0, |p| p.market_value());
            let price = self.positions.get(sym).map_or(1.0, |p| p.market_price);
            if price.abs() > 1e-10 {
                let delta_qty = (target_val - current_val) / price;
                if delta_qty.abs() > 1e-6 {
                    trades.push((sym.clone(), delta_qty));
                }
            }
        }
        trades
    }

    /// Margin calculation (simplified: 50% initial, 25% maintenance)
    pub fn margin_used(&self) -> f64 {
        self.positions.values()
            .filter(|p| p.is_short())
            .map(|p| p.notional_exposure() * 0.5)
            .sum::<f64>()
    }

    pub fn buying_power(&self) -> f64 {
        self.cash - self.margin_used()
    }

    pub fn margin_call(&self) -> bool {
        let maintenance = self.positions.values()
            .filter(|p| p.is_short())
            .map(|p| p.notional_exposure() * 0.25)
            .sum::<f64>();
        self.cash < maintenance
    }

    /// P&L attribution by symbol
    pub fn pnl_by_symbol(&self) -> HashMap<String, f64> {
        self.positions.iter()
            .map(|(sym, pos)| (sym.clone(), pos.total_pnl()))
            .collect()
    }

    /// P&L attribution by sector (given sector map)
    pub fn pnl_by_sector(&self, sector_map: &HashMap<String, String>) -> HashMap<String, f64> {
        let mut result: HashMap<String, f64> = HashMap::new();
        for (sym, pos) in &self.positions {
            let sector = sector_map.get(sym).cloned().unwrap_or_else(|| "Unknown".to_string());
            *result.entry(sector).or_insert(0.0) += pos.total_pnl();
        }
        result
    }

    /// Factor attribution: compute return explained by factors
    pub fn factor_attribution(&self, factor_returns: &HashMap<String, Vec<f64>>, exposures: &HashMap<String, HashMap<String, f64>>) -> HashMap<String, f64> {
        let mut attribution: HashMap<String, f64> = HashMap::new();
        for (factor, returns) in factor_returns {
            let mut factor_pnl = 0.0;
            for (sym, pos) in &self.positions {
                if let Some(exp_map) = exposures.get(sym) {
                    if let Some(&exposure) = exp_map.get(factor) {
                        let avg_ret: f64 = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
                        factor_pnl += exposure * avg_ret * pos.notional_exposure();
                    }
                }
            }
            attribution.insert(factor.clone(), factor_pnl);
        }
        attribution
    }

    /// Trade statistics
    pub fn trade_stats(&self) -> TradeStats {
        let trades = &self.trade_log;
        if trades.is_empty() {
            return TradeStats::default();
        }
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl).collect();
        let winning: Vec<f64> = pnls.iter().filter(|&&p| p > 0.0).cloned().collect();
        let losing: Vec<f64> = pnls.iter().filter(|&&p| p < 0.0).cloned().collect();
        let win_rate = if trades.is_empty() { 0.0 } else { winning.len() as f64 / trades.len() as f64 };
        let avg_win = if winning.is_empty() { 0.0 } else { winning.iter().sum::<f64>() / winning.len() as f64 };
        let avg_loss = if losing.is_empty() { 0.0 } else { losing.iter().sum::<f64>() / losing.len() as f64 };
        let profit_factor = if avg_loss.abs() < 1e-15 { f64::INFINITY }
            else { winning.iter().sum::<f64>() / losing.iter().sum::<f64>().abs() };
        let expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss;
        let max_win = pnls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_loss = pnls.iter().cloned().fold(f64::INFINITY, f64::min);

        TradeStats {
            total_trades: trades.len(),
            winning_trades: winning.len(),
            losing_trades: losing.len(),
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            expectancy,
            max_win,
            max_loss,
            total_commission: self.total_commission,
            total_slippage: self.total_slippage,
        }
    }

    /// Snapshot as string
    pub fn summary(&self) -> String {
        format!(
            "Equity: {:.2}, Cash: {:.2}, Positions: {}, Leverage: {:.2}x, PnL: {:.2}, DD: {:.2}%",
            self.total_equity(), self.cash, self.num_positions(),
            self.leverage(), self.total_pnl(), self.drawdown * 100.0
        )
    }

    /// Reset portfolio to initial state
    pub fn reset(&mut self) {
        self.cash = self.initial_capital;
        self.positions.clear();
        self.trade_log.clear();
        self.equity_history.clear();
        self.cash_history.clear();
        self.high_watermark = self.initial_capital;
        self.drawdown = 0.0;
        self.max_drawdown = 0.0;
        self.total_commission = 0.0;
        self.total_slippage = 0.0;
    }
}

#[derive(Clone, Debug, Default)]
pub struct TradeStats {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub max_win: f64,
    pub max_loss: f64,
    pub total_commission: f64,
    pub total_slippage: f64,
}

/// Roundtrip trade (entry + exit)
#[derive(Clone, Debug)]
pub struct Roundtrip {
    pub symbol: String,
    pub side: TradeSide,
    pub entry_time: u64,
    pub exit_time: u64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub commission: f64,
    pub duration_bars: usize,
    pub mae: f64, // max adverse excursion
    pub mfe: f64, // max favorable excursion
}

/// Extract roundtrip trades from trade log
pub fn extract_roundtrips(trades: &[Trade]) -> Vec<Roundtrip> {
    let mut open: HashMap<String, Vec<&Trade>> = HashMap::new();
    let mut roundtrips = Vec::new();

    for trade in trades {
        let entry = open.entry(trade.symbol.clone()).or_default();
        if entry.is_empty() || entry[0].side == trade.side {
            entry.push(trade);
        } else {
            // closing
            if let Some(open_trade) = entry.first() {
                let pnl = match open_trade.side {
                    TradeSide::Buy => (trade.price - open_trade.price) * trade.quantity,
                    TradeSide::Sell => (open_trade.price - trade.price) * trade.quantity,
                };
                let pnl_pct = if open_trade.price.abs() > 1e-10 {
                    pnl / (open_trade.price * trade.quantity)
                } else { 0.0 };
                roundtrips.push(Roundtrip {
                    symbol: trade.symbol.clone(),
                    side: open_trade.side,
                    entry_time: open_trade.timestamp,
                    exit_time: trade.timestamp,
                    entry_price: open_trade.price,
                    exit_price: trade.price,
                    quantity: trade.quantity,
                    pnl,
                    pnl_pct,
                    commission: open_trade.commission + trade.commission,
                    duration_bars: 0,
                    mae: 0.0,
                    mfe: 0.0,
                });
            }
            entry.clear();
        }
    }
    roundtrips
}

/// Risk metrics for portfolio
pub struct PortfolioRisk {
    pub var_95: f64,
    pub var_99: f64,
    pub cvar_95: f64,
    pub max_position_pct: f64,
    pub hhi: f64, // concentration (Herfindahl-Hirschman)
}

pub fn compute_portfolio_risk(portfolio: &Portfolio) -> PortfolioRisk {
    let returns = portfolio.returns();
    let n = returns.len();

    // Sort returns for VaR
    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let var_95 = if n > 20 { -sorted[(n as f64 * 0.05) as usize] } else { 0.0 };
    let var_99 = if n > 100 { -sorted[(n as f64 * 0.01) as usize] } else { 0.0 };
    let cutoff = (n as f64 * 0.05) as usize;
    let cvar_95 = if cutoff > 0 { -sorted[..cutoff].iter().sum::<f64>() / cutoff as f64 } else { 0.0 };

    let weights = portfolio.weights();
    let max_pos = weights.values().map(|w| w.abs()).fold(0.0f64, f64::max);
    let hhi: f64 = weights.values().map(|w| w * w).sum();

    PortfolioRisk { var_95, var_99, cvar_95, max_position_pct: max_pos, hhi }
}

/// Portfolio snapshot at a point in time
#[derive(Clone, Debug)]
pub struct PortfolioSnapshot {
    pub timestamp: u64,
    pub equity: f64,
    pub cash: f64,
    pub positions: Vec<(String, f64, f64, f64)>, // (symbol, qty, price, pnl)
    pub leverage: f64,
    pub drawdown: f64,
}

impl Portfolio {
    pub fn snapshot(&self, timestamp: u64) -> PortfolioSnapshot {
        let positions: Vec<(String, f64, f64, f64)> = self.positions.iter()
            .filter(|(_, p)| !p.is_flat())
            .map(|(sym, p)| (sym.clone(), p.quantity, p.market_price, p.total_pnl()))
            .collect();
        PortfolioSnapshot {
            timestamp,
            equity: self.total_equity(),
            cash: self.cash,
            positions,
            leverage: self.leverage(),
            drawdown: self.drawdown,
        }
    }

    /// Correlation of portfolio returns with a benchmark
    pub fn correlation_with(&self, benchmark_returns: &[f64]) -> f64 {
        let my_returns = self.returns();
        let n = my_returns.len().min(benchmark_returns.len());
        if n < 2 { return 0.0; }
        let mr = &my_returns[..n];
        let br = &benchmark_returns[..n];
        let mm = mr.iter().sum::<f64>() / n as f64;
        let mb = br.iter().sum::<f64>() / n as f64;
        let cov: f64 = mr.iter().zip(br.iter()).map(|(&a, &b)| (a - mm) * (b - mb)).sum::<f64>() / n as f64;
        let sa = (mr.iter().map(|&a| (a - mm).powi(2)).sum::<f64>() / n as f64).sqrt();
        let sb = (br.iter().map(|&b| (b - mb).powi(2)).sum::<f64>() / n as f64).sqrt();
        if sa < 1e-15 || sb < 1e-15 { 0.0 } else { cov / (sa * sb) }
    }

    /// Sector exposure given sector mapping
    pub fn sector_exposure(&self, sector_map: &HashMap<String, String>) -> HashMap<String, f64> {
        let eq = self.total_equity();
        if eq.abs() < 1e-10 { return HashMap::new(); }
        let mut result: HashMap<String, f64> = HashMap::new();
        for (sym, pos) in &self.positions {
            if pos.is_flat() { continue; }
            let sector = sector_map.get(sym).cloned().unwrap_or_else(|| "Other".to_string());
            *result.entry(sector).or_insert(0.0) += pos.market_value() / eq;
        }
        result
    }

    /// Concentration: largest N positions as fraction of equity
    pub fn top_n_concentration(&self, n: usize) -> f64 {
        let eq = self.total_equity();
        if eq.abs() < 1e-10 { return 0.0; }
        let mut exposures: Vec<f64> = self.positions.values()
            .filter(|p| !p.is_flat())
            .map(|p| p.market_value().abs())
            .collect();
        exposures.sort_by(|a, b| b.partial_cmp(a).unwrap());
        exposures.iter().take(n).sum::<f64>() / eq
    }

    /// Time in market (fraction of bars with non-zero position)
    pub fn time_in_market(&self) -> f64 {
        let total = self.equity_history.len();
        if total == 0 { return 0.0; }
        // approximate: if we have positions, we're in market
        let in_market = self.equity_history.windows(2)
            .filter(|w| (w[1] - w[0]).abs() > 1e-10)
            .count();
        in_market as f64 / total as f64
    }

    /// Average holding period (in bars)
    pub fn avg_holding_period(&self) -> f64 {
        let roundtrips = extract_roundtrips(&self.trade_log);
        if roundtrips.is_empty() { return 0.0; }
        let total: u64 = roundtrips.iter().map(|r| r.exit_time - r.entry_time).sum();
        total as f64 / roundtrips.len() as f64
    }

    /// Sharpe of realized PnL stream
    pub fn trade_sharpe(&self) -> f64 {
        let pnls: Vec<f64> = self.trade_log.iter().map(|t| t.pnl).collect();
        if pnls.len() < 2 { return 0.0; }
        let mean = pnls.iter().sum::<f64>() / pnls.len() as f64;
        let std = (pnls.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / (pnls.len() - 1) as f64).sqrt();
        if std < 1e-15 { 0.0 } else { mean / std }
    }
}

/// Multi-currency portfolio support
#[derive(Clone, Debug)]
pub struct CurrencyConverter {
    pub rates: HashMap<String, f64>, // currency pair -> rate
    pub base_currency: String,
}

impl CurrencyConverter {
    pub fn new(base: &str) -> Self {
        Self { rates: HashMap::new(), base_currency: base.to_string() }
    }

    pub fn set_rate(&mut self, pair: &str, rate: f64) {
        self.rates.insert(pair.to_string(), rate);
    }

    pub fn convert(&self, amount: f64, from: &str, to: &str) -> f64 {
        if from == to { return amount; }
        let pair = format!("{}/{}", from, to);
        if let Some(&rate) = self.rates.get(&pair) {
            return amount * rate;
        }
        let inv_pair = format!("{}/{}", to, from);
        if let Some(&rate) = self.rates.get(&inv_pair) {
            return amount / rate;
        }
        // Try via base
        let to_base = format!("{}/{}", from, &self.base_currency);
        let from_base = format!("{}/{}", &self.base_currency, to);
        if let (Some(&r1), Some(&r2)) = (self.rates.get(&to_base), self.rates.get(&from_base)) {
            return amount * r1 * r2;
        }
        amount // fallback: no conversion
    }
}

/// Tax lot tracking (FIFO)
#[derive(Clone, Debug)]
pub struct TaxLot {
    pub entry_time: u64,
    pub quantity: f64,
    pub cost_basis: f64,
}

#[derive(Clone, Debug)]
pub struct TaxLotTracker {
    pub lots: HashMap<String, Vec<TaxLot>>,
}

impl TaxLotTracker {
    pub fn new() -> Self { Self { lots: HashMap::new() } }

    pub fn add_lot(&mut self, symbol: &str, time: u64, qty: f64, price: f64) {
        self.lots.entry(symbol.to_string()).or_default().push(TaxLot {
            entry_time: time, quantity: qty, cost_basis: price,
        });
    }

    pub fn sell_fifo(&mut self, symbol: &str, qty: f64, price: f64) -> f64 {
        let lots = match self.lots.get_mut(symbol) {
            Some(l) => l,
            None => return 0.0,
        };
        let mut remaining = qty;
        let mut realized_pnl = 0.0;
        while remaining > 1e-10 && !lots.is_empty() {
            let lot = &mut lots[0];
            let close_qty = remaining.min(lot.quantity);
            realized_pnl += close_qty * (price - lot.cost_basis);
            lot.quantity -= close_qty;
            remaining -= close_qty;
            if lot.quantity < 1e-10 { lots.remove(0); }
        }
        realized_pnl
    }

    pub fn unrealized_pnl(&self, symbol: &str, current_price: f64) -> f64 {
        self.lots.get(symbol).map_or(0.0, |lots| {
            lots.iter().map(|l| l.quantity * (current_price - l.cost_basis)).sum()
        })
    }

    pub fn total_cost_basis(&self, symbol: &str) -> f64 {
        self.lots.get(symbol).map_or(0.0, |lots| {
            lots.iter().map(|l| l.quantity * l.cost_basis).sum()
        })
    }

    pub fn total_quantity(&self, symbol: &str) -> f64 {
        self.lots.get(symbol).map_or(0.0, |lots| {
            lots.iter().map(|l| l.quantity).sum()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_buy_sell() {
        let mut pos = Position::new("TEST");
        pos.update_trade(TradeSide::Buy, 100.0, 50.0, 1000);
        assert_eq!(pos.quantity, 100.0);
        assert!((pos.avg_cost - 50.0).abs() < 1e-10);

        pos.mark(55.0);
        assert!((pos.unrealized_pnl - 500.0).abs() < 1e-10);

        pos.update_trade(TradeSide::Sell, 100.0, 55.0, 2000);
        assert!(pos.is_flat());
        assert!((pos.realized_pnl - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_trade() {
        let mut portfolio = Portfolio::new(100000.0);
        let trade = Trade {
            timestamp: 1000, symbol: "AAPL".to_string(), side: TradeSide::Buy,
            quantity: 100.0, price: 150.0, commission: 1.0, slippage: 0.0,
            order_id: 1, pnl: 0.0, tag: String::new(),
        };
        portfolio.process_trade(&trade);
        assert_eq!(portfolio.num_positions(), 1);
        assert!((portfolio.cash - (100000.0 - 150.0 * 100.0 - 1.0)).abs() < 1e-10);

        let mut prices = HashMap::new();
        prices.insert("AAPL".to_string(), 155.0);
        portfolio.mark_to_market(&prices);

        let eq = portfolio.total_equity();
        assert!((eq - (100000.0 - 1.0 + 500.0)).abs() < 1e-10);
    }

    #[test]
    fn test_weights() {
        let mut portfolio = Portfolio::new(100000.0);
        let t = Trade { timestamp: 0, symbol: "A".to_string(), side: TradeSide::Buy,
            quantity: 100.0, price: 100.0, commission: 0.0, slippage: 0.0,
            order_id: 1, pnl: 0.0, tag: String::new() };
        portfolio.process_trade(&t);
        let mut prices = HashMap::new();
        prices.insert("A".to_string(), 100.0);
        portfolio.mark_to_market(&prices);
        let w = portfolio.weights();
        assert!((w["A"] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_trades_to_target() {
        let mut portfolio = Portfolio::new(100000.0);
        let mut prices = HashMap::new();
        prices.insert("A".to_string(), 100.0);
        portfolio.mark_to_market(&prices);

        let mut target = HashMap::new();
        target.insert("A".to_string(), 0.5);
        let trades = portfolio.trades_to_target(&target);
        assert!(!trades.is_empty());
    }

    #[test]
    fn test_roundtrips() {
        let trades = vec![
            Trade { timestamp: 1, symbol: "X".into(), side: TradeSide::Buy, quantity: 10.0,
                price: 100.0, commission: 1.0, slippage: 0.0, order_id: 1, pnl: 0.0, tag: String::new() },
            Trade { timestamp: 2, symbol: "X".into(), side: TradeSide::Sell, quantity: 10.0,
                price: 110.0, commission: 1.0, slippage: 0.0, order_id: 2, pnl: 0.0, tag: String::new() },
        ];
        let rts = extract_roundtrips(&trades);
        assert_eq!(rts.len(), 1);
        assert!((rts[0].pnl - 100.0).abs() < 1e-10);
    }
}
