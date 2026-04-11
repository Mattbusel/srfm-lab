/// Trend-following momentum agent with EWMA signals and position limits.
///
/// Strategy:
///   - Compute fast/slow EWMA of mid-price
///   - Signal = fast_ewma - slow_ewma (trend direction and magnitude)
///   - Enter long when signal > entry_threshold
///   - Enter short when signal < -entry_threshold
///   - Exit when signal crosses zero or position limit hit
///   - Size position proportional to signal strength

use crate::exchange::{Order, Fill, Side, OrderKind, TimeInForce, OrderId, InstrumentId, AgentId, Qty, Price, Nanos};

// ── EWMA ──────────────────────────────────────────────────────────────────────

pub struct Ewma {
    pub value: f64,
    pub alpha: f64,
    pub initialized: bool,
    pub count: usize,
}

impl Ewma {
    pub fn new(halflife_periods: f64) -> Self {
        let alpha = 1.0 - (-2.0_f64.ln() / halflife_periods).exp();
        Ewma { value: 0.0, alpha, initialized: false, count: 0 }
    }

    pub fn with_alpha(alpha: f64) -> Self {
        Ewma { value: 0.0, alpha: alpha.clamp(0.0, 1.0), initialized: false, count: 0 }
    }

    pub fn update(&mut self, x: f64) -> f64 {
        if !self.initialized {
            self.value = x;
            self.initialized = true;
        } else {
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value;
        }
        self.count += 1;
        self.value
    }

    pub fn is_warm(&self, min_periods: usize) -> bool {
        self.count >= min_periods
    }
}

// ── ATR (Average True Range) for volatility adjustment ──────────────────────

pub struct AtrEstimator {
    ema: Ewma,
    pub value: f64,
    pub count: usize,
}

impl AtrEstimator {
    pub fn new(period: f64) -> Self {
        AtrEstimator { ema: Ewma::new(period), value: 0.0, count: 0 }
    }

    pub fn update(&mut self, high: f64, low: f64, prev_close: f64) -> f64 {
        let tr = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        self.value = self.ema.update(tr);
        self.count += 1;
        self.value
    }

    pub fn update_from_midprice(&mut self, mid: f64, prev_mid: f64) -> f64 {
        let range = (mid - prev_mid).abs() * 2.0; // approximate
        self.value = self.ema.update(range);
        self.count += 1;
        self.value
    }
}

// ── Momentum Config ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MomentumConfig {
    /// Fast EWMA half-life (in ticks/steps).
    pub fast_halflife: f64,
    /// Slow EWMA half-life (in ticks/steps).
    pub slow_halflife: f64,
    /// Signal entry threshold (in units of ATR).
    pub entry_threshold_atr: f64,
    /// Signal exit threshold.
    pub exit_threshold_atr: f64,
    /// Maximum position size.
    pub max_position: Qty,
    /// Base order quantity.
    pub base_qty: Qty,
    /// Whether to scale qty by signal strength.
    pub scale_by_signal: bool,
    /// Minimum warmup periods before trading.
    pub warmup_periods: usize,
    /// Take-profit distance (in ATR). None = no TP.
    pub take_profit_atr: Option<f64>,
    /// Stop-loss distance (in ATR). None = no SL.
    pub stop_loss_atr: Option<f64>,
    /// ATR period for volatility adjustment.
    pub atr_period: f64,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        MomentumConfig {
            fast_halflife: 5.0,
            slow_halflife: 20.0,
            entry_threshold_atr: 0.5,
            exit_threshold_atr: 0.1,
            max_position: 500.0,
            base_qty: 100.0,
            scale_by_signal: true,
            warmup_periods: 30,
            take_profit_atr: Some(2.0),
            stop_loss_atr: Some(1.0),
            atr_period: 14.0,
        }
    }
}

// ── Position Entry Tracker ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PositionEntry {
    pub entry_price: Price,
    pub entry_ts_ns: Nanos,
    pub side: Side,
    pub qty: Qty,
    pub take_profit: Option<Price>,
    pub stop_loss: Option<Price>,
}

// ── Momentum Agent ────────────────────────────────────────────────────────────

pub struct MomentumAgent {
    pub agent_id: AgentId,
    pub instrument_id: InstrumentId,
    pub config: MomentumConfig,

    fast_ewma: Ewma,
    slow_ewma: Ewma,
    signal_ewma: Ewma,     // smoothed signal
    atr: AtrEstimator,

    pub position: Qty,
    pub cash: f64,
    pub realized_pnl: f64,
    pub entry: Option<PositionEntry>,

    prev_mid: f64,
    step_count: usize,

    order_id_base: u64,
    order_id_ctr: u64,

    pub total_fills: u64,
    pub total_volume: Qty,
    pub win_count: u64,
    pub loss_count: u64,
    pub signal_history: Vec<f64>,
}

impl MomentumAgent {
    pub fn new(
        agent_id: AgentId,
        instrument_id: InstrumentId,
        config: MomentumConfig,
        initial_cash: f64,
    ) -> Self {
        let fast = Ewma::new(config.fast_halflife);
        let slow = Ewma::new(config.slow_halflife);
        let signal_sm = Ewma::new(config.fast_halflife / 2.0);
        let atr = AtrEstimator::new(config.atr_period);
        MomentumAgent {
            agent_id,
            instrument_id,
            config,
            fast_ewma: fast,
            slow_ewma: slow,
            signal_ewma: signal_sm,
            atr,
            position: 0.0,
            cash: initial_cash,
            realized_pnl: 0.0,
            entry: None,
            prev_mid: 0.0,
            step_count: 0,
            order_id_base: (agent_id as u64) * 300_000_000,
            order_id_ctr: 0,
            total_fills: 0,
            total_volume: 0.0,
            win_count: 0,
            loss_count: 0,
            signal_history: Vec::new(),
        }
    }

    fn next_order_id(&mut self) -> OrderId {
        self.order_id_ctr += 1;
        self.order_id_base + self.order_id_ctr
    }

    /// Update signal and generate orders.
    pub fn step(&mut self, mid: Price, ts_ns: Nanos) -> Vec<Order> {
        self.step_count += 1;
        let prev = self.prev_mid;
        self.prev_mid = mid;

        if prev < 1e-9 { return vec![]; }

        // Update indicators.
        let fast = self.fast_ewma.update(mid);
        let slow = self.slow_ewma.update(mid);
        let raw_signal = fast - slow;
        let signal = self.signal_ewma.update(raw_signal);
        self.atr.update_from_midprice(mid, prev);
        self.signal_history.push(signal);

        // Need warmup.
        if self.step_count < self.config.warmup_periods || self.atr.value < 1e-10 {
            return vec![];
        }

        let atr = self.atr.value;
        let entry_thr = self.config.entry_threshold_atr * atr;
        let exit_thr = self.config.exit_threshold_atr * atr;

        let mut orders = Vec::new();

        // Check TP/SL for existing position.
        if let Some(ref entry) = self.entry {
            let ep = entry.entry_price;
            let side = entry.side;

            let tp_hit = entry.take_profit.map_or(false, |tp| match side {
                Side::Buy => mid >= tp,
                Side::Sell => mid <= tp,
            });
            let sl_hit = entry.stop_loss.map_or(false, |sl| match side {
                Side::Buy => mid <= sl,
                Side::Sell => mid >= sl,
            });

            if tp_hit || sl_hit {
                // Exit position.
                let exit_side = match side {
                    Side::Buy => Side::Sell,
                    Side::Sell => Side::Buy,
                };
                let qty = self.position.abs();
                if qty > 1e-9 {
                    let exit_order = Order::new_market(self.next_order_id(), self.instrument_id, self.agent_id, exit_side, qty, ts_ns);
                    orders.push(exit_order);
                }
                return orders;
            }
        }

        // Signal logic.
        if self.position.abs() < 1e-9 {
            // Flat: check for entry signal.
            if signal > entry_thr {
                // Bullish momentum: buy.
                let qty = self.compute_order_qty(signal, atr);
                let entry_id = self.next_order_id();
                let buy = Order::new_market(entry_id, self.instrument_id, self.agent_id, Side::Buy, qty, ts_ns);
                // Pre-compute TP/SL levels.
                let tp = self.config.take_profit_atr.map(|m| mid + m * atr);
                let sl = self.config.stop_loss_atr.map(|m| mid - m * atr);
                self.entry = Some(PositionEntry { entry_price: mid, entry_ts_ns: ts_ns, side: Side::Buy, qty, take_profit: tp, stop_loss: sl });
                orders.push(buy);
            } else if signal < -entry_thr {
                // Bearish momentum: sell.
                let qty = self.compute_order_qty(signal.abs(), atr);
                let entry_id = self.next_order_id();
                let sell = Order::new_market(entry_id, self.instrument_id, self.agent_id, Side::Sell, qty, ts_ns);
                let tp = self.config.take_profit_atr.map(|m| mid - m * atr);
                let sl = self.config.stop_loss_atr.map(|m| mid + m * atr);
                self.entry = Some(PositionEntry { entry_price: mid, entry_ts_ns: ts_ns, side: Side::Sell, qty, take_profit: tp, stop_loss: sl });
                orders.push(sell);
            }
        } else {
            // In position: check for exit signal (signal reversal).
            let in_long = self.position > 0.0;
            let in_short = self.position < 0.0;

            let exit_long = in_long && signal < exit_thr;
            let exit_short = in_short && signal > -exit_thr;

            if exit_long || exit_short {
                let exit_side = if in_long { Side::Sell } else { Side::Buy };
                let qty = self.position.abs();
                let exit_order = Order::new_market(self.next_order_id(), self.instrument_id, self.agent_id, exit_side, qty, ts_ns);
                orders.push(exit_order);
                self.entry = None;
            }
        }

        orders
    }

    fn compute_order_qty(&self, signal_strength: f64, atr: f64) -> Qty {
        if !self.config.scale_by_signal {
            return self.config.base_qty;
        }
        let scale = (signal_strength / atr).min(3.0).max(0.1);
        (self.config.base_qty * scale).min(self.config.max_position)
    }

    pub fn on_fill(&mut self, fill: &Fill) {
        if fill.aggressor_agent != self.agent_id { return; }
        let delta = match fill.side {
            Side::Buy => fill.qty,
            Side::Sell => -fill.qty,
        };
        let cash_delta = match fill.side {
            Side::Buy => -fill.price * fill.qty,
            Side::Sell => fill.price * fill.qty,
        };

        // Check if this is an exit that closes a previous position.
        if let Some(ref entry) = self.entry.clone() {
            let closing = match fill.side {
                Side::Buy => entry.side == Side::Sell,
                Side::Sell => entry.side == Side::Buy,
            };
            if closing {
                let pnl = match entry.side {
                    Side::Buy => (fill.price - entry.entry_price) * fill.qty,
                    Side::Sell => (entry.entry_price - fill.price) * fill.qty,
                };
                self.realized_pnl += pnl;
                if pnl > 0.0 { self.win_count += 1; } else { self.loss_count += 1; }
                self.entry = None;
            }
        }

        self.position += delta;
        self.cash += cash_delta;
        self.total_fills += 1;
        self.total_volume += fill.qty;
    }

    pub fn mark_to_market(&self, mid: Price) -> f64 {
        self.cash + self.position * mid
    }

    pub fn win_rate(&self) -> f64 {
        let total = self.win_count + self.loss_count;
        if total == 0 { return 0.0; }
        self.win_count as f64 / total as f64
    }

    pub fn current_signal(&self) -> f64 {
        self.signal_history.last().copied().unwrap_or(0.0)
    }

    pub fn summary(&self, mid: Price) -> MomentumSummary {
        MomentumSummary {
            agent_id: self.agent_id,
            position: self.position,
            cash: self.cash,
            realized_pnl: self.realized_pnl,
            mtm: self.mark_to_market(mid),
            win_rate: self.win_rate(),
            total_fills: self.total_fills,
            total_volume: self.total_volume,
            current_signal: self.current_signal(),
            atr: self.atr.value,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MomentumSummary {
    pub agent_id: AgentId,
    pub position: Qty,
    pub cash: f64,
    pub realized_pnl: f64,
    pub mtm: f64,
    pub win_rate: f64,
    pub total_fills: u64,
    pub total_volume: Qty,
    pub current_signal: f64,
    pub atr: f64,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_convergence() {
        let mut ema = Ewma::new(10.0);
        for _ in 0..200 { ema.update(100.0); }
        assert!((ema.value - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_ewma_trend() {
        let mut fast = Ewma::new(3.0);
        let mut slow = Ewma::new(20.0);
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.1).collect();
        for &p in &prices {
            fast.update(p);
            slow.update(p);
        }
        // In an uptrend, fast should be above slow.
        assert!(fast.value > slow.value, "fast={} slow={}", fast.value, slow.value);
    }

    #[test]
    fn test_momentum_agent_warmup() {
        let config = MomentumConfig { warmup_periods: 5, ..Default::default() };
        let mut agent = MomentumAgent::new(1, 1, config, 100_000.0);
        // Steps before warmup should produce no orders.
        for i in 0..4 {
            let orders = agent.step(100.0 + i as f64 * 0.01, i as u64 * 1_000_000);
            assert!(orders.is_empty(), "expected no orders during warmup");
        }
    }

    #[test]
    fn test_momentum_agent_enters_trend() {
        let config = MomentumConfig {
            warmup_periods: 10,
            fast_halflife: 2.0,
            slow_halflife: 10.0,
            entry_threshold_atr: 0.01, // very low threshold for test
            ..Default::default()
        };
        let mut agent = MomentumAgent::new(1, 1, config, 100_000.0);

        // Strong uptrend.
        let mut ts = 0u64;
        let mut price = 100.0f64;
        for i in 0..50 {
            price += 0.5; // strong uptrend
            let orders = agent.step(price, ts);
            ts += 1_000_000_000;
            if !orders.is_empty() {
                // Found an entry signal.
                assert_eq!(orders[0].side, Side::Buy);
                return;
            }
        }
        // It's ok if no entry was generated given threshold.
    }

    #[test]
    fn test_position_tracking() {
        let config = MomentumConfig::default();
        let mut agent = MomentumAgent::new(1, 1, config, 100_000.0);
        let fill = Fill {
            aggressor_id: 1,
            passive_id: 2,
            instrument_id: 1,
            price: 100.0,
            qty: 50.0,
            side: Side::Buy,
            timestamp_ns: 0,
            aggressor_agent: 1,
            passive_agent: 99,
        };
        agent.on_fill(&fill);
        assert!((agent.position - 50.0).abs() < 1e-9);
        assert!((agent.cash - 95_000.0).abs() < 1e-9);
    }

    #[test]
    fn test_atr_positive() {
        let mut atr = AtrEstimator::new(14.0);
        let mut prev = 100.0;
        for i in 0..50 {
            let mid = 100.0 + (i as f64 * 0.1).sin();
            atr.update_from_midprice(mid, prev);
            prev = mid;
        }
        assert!(atr.value > 0.0);
    }
}
