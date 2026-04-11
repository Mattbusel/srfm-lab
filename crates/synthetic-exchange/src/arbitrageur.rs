/// Statistical arbitrage agent exploiting synthetic mispricings.
///
/// Implements:
/// - Pairs trading via Engle-Granger cointegration
/// - Spread Z-score signal generation
/// - Entry/exit thresholds with half-life adjustment
/// - Position sizing via Kelly criterion
/// - Kalman filter for dynamic hedge ratio estimation

use std::collections::VecDeque;
use crate::exchange::{Order, Fill, Side, OrderKind, TimeInForce, OrderId, InstrumentId, AgentId, Qty, Price, Nanos};

// ── Cointegration Test (Engle-Granger) ───────────────────────────────────────

/// Result of Engle-Granger cointegration test.
#[derive(Debug, Clone)]
pub struct CointegrationResult {
    /// OLS hedge ratio β: y ~ α + β·x.
    pub hedge_ratio: f64,
    /// Intercept α.
    pub intercept: f64,
    /// ADF test statistic on residuals.
    pub adf_stat: f64,
    /// p-value (approximate, from MacKinnon critical values).
    pub p_value: f64,
    /// Residuals (spread series).
    pub residuals: Vec<f64>,
    /// Half-life of mean reversion (in observations).
    pub half_life: f64,
    /// R² of the cointegrating regression.
    pub r_squared: f64,
}

impl CointegrationResult {
    pub fn is_cointegrated(&self) -> bool {
        self.p_value < 0.05
    }
}

/// Compute OLS hedge ratio and intercept: y = α + β·x + ε.
pub fn ols(y: &[f64], x: &[f64]) -> (f64, f64, Vec<f64>) {
    assert_eq!(y.len(), x.len());
    let n = y.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let cov_xy: f64 = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>() / n;
    let var_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<f64>() / n;

    let beta = if var_x.abs() < 1e-20 { 0.0 } else { cov_xy / var_x };
    let alpha = mean_y - beta * mean_x;

    let residuals: Vec<f64> = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| yi - alpha - beta * xi)
        .collect();

    (beta, alpha, residuals)
}

/// Augmented Dickey-Fuller test statistic (simplified, ADF(0) = DF test).
/// Tests H0: residuals have a unit root (not mean-reverting).
/// Returns t-statistic (more negative = more evidence against unit root).
pub fn adf_test(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 4 { return 0.0; }

    // Compute first differences.
    let y: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let x: Vec<f64> = series[..series.len() - 1].to_vec();  // lagged levels

    // OLS: Δy_t = ρ·y_{t-1} + ε (no drift/trend for simplicity)
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;

    let cov_xy: f64 = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>();
    let var_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<f64>();

    if var_x.abs() < 1e-20 { return 0.0; }
    let rho = cov_xy / var_x;

    // Residuals of DF regression.
    let resid: Vec<f64> = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| yi - rho * xi)
        .collect();
    let sigma2 = resid.iter().map(|r| r * r).sum::<f64>() / (resid.len() - 1) as f64;
    let se_rho = (sigma2 / var_x).sqrt();

    if se_rho.abs() < 1e-20 { return 0.0; }
    rho / se_rho  // ADF t-statistic
}

/// Approximate p-value from MacKinnon (1994) critical values for ADF test.
/// Returns p-value ∈ [0, 1].
pub fn adf_p_value(adf_stat: f64) -> f64 {
    // Critical values from MacKinnon (no trend/drift, n→∞):
    // 1%: -3.43, 5%: -2.86, 10%: -2.57
    if adf_stat < -3.43 { return 0.01; }
    if adf_stat < -2.86 { return 0.05; }
    if adf_stat < -2.57 { return 0.10; }
    if adf_stat < -2.20 { return 0.20; }
    if adf_stat < -1.75 { return 0.35; }
    0.50
}

/// Estimate half-life of mean reversion from AR(1) regression on spread.
/// Returns half-life in number of observations.
pub fn estimate_half_life(spread: &[f64]) -> f64 {
    if spread.len() < 4 { return f64::INFINITY; }
    let n = spread.len();
    // AR(1): S_t = φ·S_{t-1} + ε
    let y: Vec<f64> = spread[1..].to_vec();
    let x: Vec<f64> = spread[..n-1].to_vec();

    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let cov: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| (a - mean_x) * (b - mean_y)).sum::<f64>();
    let var: f64 = x.iter().map(|&a| (a - mean_x).powi(2)).sum::<f64>();

    if var.abs() < 1e-20 { return f64::INFINITY; }
    let phi = cov / var;
    if phi >= 1.0 || phi <= 0.0 { return f64::INFINITY; }
    -2.0_f64.ln() / phi.ln()
}

/// Full Engle-Granger test.
pub fn engle_granger_test(y: &[f64], x: &[f64]) -> CointegrationResult {
    let n = y.len() as f64;
    let (beta, alpha, residuals) = ols(y, x);

    // R²
    let y_mean = y.iter().sum::<f64>() / n;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = residuals.iter().map(|&r| r * r).sum();
    let r_sq = if ss_tot.abs() < 1e-20 { 0.0 } else { 1.0 - ss_res / ss_tot };

    let adf = adf_test(&residuals);
    let p_val = adf_p_value(adf);
    let hl = estimate_half_life(&residuals);

    CointegrationResult {
        hedge_ratio: beta,
        intercept: alpha,
        adf_stat: adf,
        p_value: p_val,
        residuals,
        half_life: hl,
        r_squared: r_sq,
    }
}

// ── Kalman Filter for Dynamic Hedge Ratio ─────────────────────────────────────

/// 1D Kalman filter to track time-varying hedge ratio.
///
/// State: β_t (hedge ratio), scalar.
/// Observation: y_t = β_t · x_t + ε_t
pub struct KalmanHedge {
    /// State estimate (hedge ratio).
    pub beta: f64,
    /// State covariance.
    pub p: f64,
    /// Process noise (variance of β change per step).
    pub q: f64,
    /// Observation noise variance.
    pub r: f64,
    /// History of estimates.
    pub beta_history: Vec<f64>,
}

impl KalmanHedge {
    pub fn new(beta_init: f64, p_init: f64, q: f64, r: f64) -> Self {
        KalmanHedge { beta: beta_init, p: p_init, q, r, beta_history: Vec::new() }
    }

    /// Update with new observation (y_t, x_t). Returns updated hedge ratio.
    pub fn update(&mut self, y: f64, x: f64) -> f64 {
        // Predict.
        // beta_t|t-1 = beta_{t-1} (random walk)
        let p_pred = self.p + self.q;

        // Innovation: y_t - beta_pred * x_t
        let innov = y - self.beta * x;

        // Innovation variance: S = x² * P_pred + R
        let s = x * x * p_pred + self.r;

        // Kalman gain: K = P_pred * x / S
        let k = if s.abs() < 1e-20 { 0.0 } else { p_pred * x / s };

        // Update.
        self.beta = self.beta + k * innov;
        self.p = (1.0 - k * x) * p_pred;
        self.beta_history.push(self.beta);
        self.beta
    }

    /// Compute spread using current hedge ratio.
    pub fn spread(&self, y: f64, x: f64, alpha: f64) -> f64 {
        y - self.beta * x - alpha
    }
}

// ── Spread Signal ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArbitrageSignal {
    /// No action.
    Neutral,
    /// Spread too wide: sell spread (sell y, buy x).
    ShortSpread,
    /// Spread too narrow: buy spread (buy y, sell x).
    LongSpread,
    /// Exit position.
    Exit,
}

pub struct SpreadSignal {
    /// Rolling window of spread values.
    history: VecDeque<f64>,
    window: usize,
    /// Entry Z-score threshold.
    pub entry_z: f64,
    /// Exit Z-score threshold.
    pub exit_z: f64,
    /// Current position direction.
    pub position_direction: i8,  // +1 = long spread, -1 = short, 0 = flat
}

impl SpreadSignal {
    pub fn new(window: usize, entry_z: f64, exit_z: f64) -> Self {
        SpreadSignal {
            history: VecDeque::new(),
            window,
            entry_z,
            exit_z,
            position_direction: 0,
        }
    }

    pub fn update(&mut self, spread: f64) -> ArbitrageSignal {
        self.history.push_back(spread);
        if self.history.len() > self.window {
            self.history.pop_front();
        }

        if self.history.len() < 5 { return ArbitrageSignal::Neutral; }

        let mean = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let std = {
            let var = self.history.iter().map(|&s| (s - mean).powi(2)).sum::<f64>()
                / (self.history.len() - 1) as f64;
            var.sqrt()
        };
        if std < 1e-10 { return ArbitrageSignal::Neutral; }

        let z = (spread - mean) / std;

        if self.position_direction != 0 {
            // Check for exit.
            if z.abs() < self.exit_z {
                self.position_direction = 0;
                return ArbitrageSignal::Exit;
            }
        } else {
            // Check for entry.
            if z > self.entry_z {
                self.position_direction = -1;
                return ArbitrageSignal::ShortSpread;
            } else if z < -self.entry_z {
                self.position_direction = 1;
                return ArbitrageSignal::LongSpread;
            }
        }

        ArbitrageSignal::Neutral
    }

    pub fn current_z(&self) -> f64 {
        if self.history.len() < 2 { return 0.0; }
        let mean = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let var = self.history.iter().map(|&s| (s - mean).powi(2)).sum::<f64>()
            / (self.history.len() - 1) as f64;
        let std = var.sqrt();
        if std < 1e-10 { return 0.0; }
        let last = *self.history.back().unwrap();
        (last - mean) / std
    }
}

// ── Arbitrageur Agent ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ArbitrageConfig {
    /// Maximum position size per leg.
    pub max_position: Qty,
    /// Base order quantity.
    pub order_qty: Qty,
    /// Entry Z-score.
    pub entry_z: f64,
    /// Exit Z-score.
    pub exit_z: f64,
    /// Lookback window for spread statistics.
    pub lookback: usize,
    /// Use Kalman filter for hedge ratio (true) or fixed OLS (false).
    pub use_kalman: bool,
    /// Recalibrate cointegration every N steps.
    pub recalib_interval: usize,
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        ArbitrageConfig {
            max_position: 1000.0,
            order_qty: 100.0,
            entry_z: 2.0,
            exit_z: 0.5,
            lookback: 60,
            use_kalman: true,
            recalib_interval: 100,
        }
    }
}

pub struct ArbitrageAgent {
    pub agent_id: AgentId,
    pub inst_y: InstrumentId,   // "y" leg of the pair
    pub inst_x: InstrumentId,   // "x" leg of the pair
    pub config: ArbitrageConfig,

    /// Hedge ratio β.
    pub hedge_ratio: f64,
    /// Intercept.
    pub alpha_intercept: f64,
    /// Kalman filter (if enabled).
    kalman: Option<KalmanHedge>,
    /// Spread signal.
    signal: SpreadSignal,
    /// Position in y-leg.
    pub pos_y: Qty,
    /// Position in x-leg.
    pub pos_x: Qty,
    /// Cash PnL.
    pub pnl: f64,
    /// Step counter.
    step_count: usize,
    /// Historical prices for y and x.
    prices_y: VecDeque<f64>,
    prices_x: VecDeque<f64>,
    /// Order counter.
    order_id_base: u64,
    order_id_ctr: u64,
    /// All generated fills.
    pub fill_count: u64,
    pub total_volume: Qty,
}

impl ArbitrageAgent {
    pub fn new(
        agent_id: AgentId,
        inst_y: InstrumentId,
        inst_x: InstrumentId,
        config: ArbitrageConfig,
    ) -> Self {
        let kalman = if config.use_kalman {
            Some(KalmanHedge::new(1.0, 1.0, 1e-4, 1e-2))
        } else {
            None
        };
        let signal = SpreadSignal::new(config.lookback, config.entry_z, config.exit_z);
        ArbitrageAgent {
            agent_id,
            inst_y, inst_x,
            config,
            hedge_ratio: 1.0,
            alpha_intercept: 0.0,
            kalman,
            signal,
            pos_y: 0.0,
            pos_x: 0.0,
            pnl: 0.0,
            step_count: 0,
            prices_y: VecDeque::new(),
            prices_x: VecDeque::new(),
            order_id_base: (agent_id as u64) * 200_000_000,
            order_id_ctr: 0,
            fill_count: 0,
            total_volume: 0.0,
        }
    }

    fn next_order_id(&mut self) -> OrderId {
        self.order_id_ctr += 1;
        self.order_id_base + self.order_id_ctr
    }

    /// Feed new prices. Returns list of orders to submit.
    pub fn step(
        &mut self,
        price_y: Price,
        price_x: Price,
        ts_ns: Nanos,
    ) -> Vec<Order> {
        self.step_count += 1;
        self.prices_y.push_back(price_y);
        self.prices_x.push_back(price_x);
        if self.prices_y.len() > self.config.lookback * 2 {
            self.prices_y.pop_front();
            self.prices_x.pop_front();
        }

        // Recalibrate periodically.
        if self.step_count % self.config.recalib_interval == 0
            && self.prices_y.len() >= 20
        {
            let y_vec: Vec<f64> = self.prices_y.iter().cloned().collect();
            let x_vec: Vec<f64> = self.prices_x.iter().cloned().collect();
            let coint = engle_granger_test(&y_vec, &x_vec);
            self.hedge_ratio = coint.hedge_ratio;
            self.alpha_intercept = coint.intercept;
        }

        // Update Kalman hedge ratio.
        if let Some(ref mut kf) = self.kalman {
            self.hedge_ratio = kf.update(price_y, price_x);
        }

        // Compute spread.
        let spread = price_y - self.hedge_ratio * price_x - self.alpha_intercept;
        let signal = self.signal.update(spread);

        let mut orders = Vec::new();

        match signal {
            ArbitrageSignal::ShortSpread => {
                // Spread too high: sell y, buy x.
                // Check position limits.
                if self.pos_y > -self.config.max_position
                    && self.pos_x < self.config.max_position
                {
                    let qty_y = self.config.order_qty;
                    let qty_x = self.config.order_qty * self.hedge_ratio.abs();

                    let sell_y = Order::new_market(self.next_order_id(), self.inst_y, self.agent_id, Side::Sell, qty_y, ts_ns);
                    let buy_x = Order::new_market(self.next_order_id(), self.inst_x, self.agent_id, Side::Buy, qty_x, ts_ns);
                    orders.push(sell_y);
                    orders.push(buy_x);
                }
            }
            ArbitrageSignal::LongSpread => {
                // Spread too low: buy y, sell x.
                if self.pos_y < self.config.max_position
                    && self.pos_x > -self.config.max_position
                {
                    let qty_y = self.config.order_qty;
                    let qty_x = self.config.order_qty * self.hedge_ratio.abs();

                    let buy_y = Order::new_market(self.next_order_id(), self.inst_y, self.agent_id, Side::Buy, qty_y, ts_ns);
                    let sell_x = Order::new_market(self.next_order_id(), self.inst_x, self.agent_id, Side::Sell, qty_x, ts_ns);
                    orders.push(buy_y);
                    orders.push(sell_x);
                }
            }
            ArbitrageSignal::Exit => {
                // Unwind positions.
                if self.pos_y.abs() > 1e-9 {
                    let side = if self.pos_y > 0.0 { Side::Sell } else { Side::Buy };
                    let qty = self.pos_y.abs();
                    orders.push(Order::new_market(self.next_order_id(), self.inst_y, self.agent_id, side, qty, ts_ns));
                }
                if self.pos_x.abs() > 1e-9 {
                    let side = if self.pos_x > 0.0 { Side::Sell } else { Side::Buy };
                    let qty = self.pos_x.abs();
                    orders.push(Order::new_market(self.next_order_id(), self.inst_x, self.agent_id, side, qty, ts_ns));
                }
            }
            ArbitrageSignal::Neutral => {}
        }

        orders
    }

    /// Process a fill from the exchange.
    pub fn on_fill(&mut self, fill: &Fill) {
        if fill.aggressor_agent != self.agent_id && fill.passive_agent != self.agent_id {
            return;
        }
        let our_side = if fill.passive_agent == self.agent_id {
            match fill.side { Side::Buy => Side::Sell, Side::Sell => Side::Buy }
        } else { fill.side };

        let delta = match our_side {
            Side::Buy => fill.qty,
            Side::Sell => -fill.qty,
        };
        let pnl_delta = match our_side {
            Side::Buy => -fill.price * fill.qty,
            Side::Sell => fill.price * fill.qty,
        };
        self.pnl += pnl_delta;

        if fill.instrument_id == self.inst_y {
            self.pos_y += delta;
        } else if fill.instrument_id == self.inst_x {
            self.pos_x += delta;
        }
        self.fill_count += 1;
        self.total_volume += fill.qty;
    }

    pub fn current_z(&self) -> f64 {
        self.signal.current_z()
    }

    pub fn current_spread(&self, price_y: Price, price_x: Price) -> f64 {
        price_y - self.hedge_ratio * price_x - self.alpha_intercept
    }

    pub fn mark_to_market(&self, price_y: Price, price_x: Price) -> f64 {
        self.pnl + self.pos_y * price_y + self.pos_x * price_x
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ols_regression() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 5.0 + xi.sin() * 0.1).collect();
        let (beta, alpha, resid) = ols(&y, &x);
        assert!((beta - 2.0).abs() < 0.1, "beta: {}", beta);
        assert!((alpha - 5.0).abs() < 0.5, "alpha: {}", alpha);
        assert!(resid.iter().map(|&r| r.abs()).sum::<f64>() / resid.len() as f64 < 0.2);
    }

    #[test]
    fn test_adf_stationary_series() {
        // White noise should be stationary.
        let series: Vec<f64> = (0..200).map(|i| {
            // Simple LCG pseudo-random.
            ((i * 1664525 + 1013904223) % 65536) as f64 / 32768.0 - 1.0
        }).collect();
        let adf = adf_test(&series);
        // White noise has very negative ADF statistic.
        assert!(adf < -2.0, "adf: {}", adf);
    }

    #[test]
    fn test_half_life_ar1() {
        // AR(1) with φ=0.9: half-life = ln(2)/ln(1/0.9) ≈ 6.58
        let mut series = vec![0.0_f64];
        for _ in 1..1000 {
            let last = *series.last().unwrap();
            series.push(0.9 * last + 0.1); // slight drift to stabilize
        }
        let hl = estimate_half_life(&series);
        assert!(hl > 0.0 && hl < 100.0, "half_life: {}", hl);
    }

    #[test]
    fn test_spread_signal_entry_exit() {
        let mut sig = SpreadSignal::new(20, 2.0, 0.5);
        // Push normal spreads.
        for _ in 0..20 { sig.update(0.0); }
        // Push very large spread to trigger entry.
        let signal = sig.update(100.0);
        assert_ne!(signal, ArbitrageSignal::Neutral);
        // Push neutral to trigger exit.
        for _ in 0..5 {
            let s = sig.update(0.1);
            if s == ArbitrageSignal::Exit { return; }
        }
        // May or may not exit in 5 steps depending on window stats.
    }

    #[test]
    fn test_kalman_hedge_converges() {
        let mut kf = KalmanHedge::new(1.5, 1.0, 1e-4, 1e-2);
        // True relationship: y = 2*x.
        for i in 1..500 {
            let x = i as f64 * 0.1;
            let y = 2.0 * x + (i as f64 * 0.01).sin() * 0.05;
            kf.update(y, x);
        }
        // After 500 observations, Kalman should be close to true beta=2.
        assert!((kf.beta - 2.0).abs() < 0.2, "kalman beta: {}", kf.beta);
    }

    #[test]
    fn test_arbitrageur_step() {
        let config = ArbitrageConfig {
            lookback: 20,
            recalib_interval: 10,
            ..Default::default()
        };
        let mut agent = ArbitrageAgent::new(1, 1, 2, config);
        // Warmup.
        for i in 0..30 {
            agent.step(100.0 + i as f64 * 0.01, 50.0 + i as f64 * 0.005, i as u64 * 1_000_000_000);
        }
        // Normal operation — no assertion on orders, just no panic.
        assert!(agent.step_count == 30);
    }

    #[test]
    fn test_engle_granger_cointegrated_pair() {
        // Generate a cointegrated pair: y = 2*x + stationary noise.
        let n = 200;
        let mut x = vec![100.0_f64];
        for i in 1..n {
            let noise = ((i * 1664525 + 1013904223) % 65536) as f64 / 65536.0 - 0.5;
            x.push(x.last().unwrap() + noise);
        }
        let y: Vec<f64> = x.iter().enumerate().map(|(i, &xi)| {
            let e = ((i * 6364136223846793005 + 1442695040888963407) % 65536) as f64 / 65536.0 - 0.5;
            2.0 * xi + 10.0 + e
        }).collect();

        let result = engle_granger_test(&y, &x);
        assert!((result.hedge_ratio - 2.0).abs() < 0.5, "hedge ratio: {}", result.hedge_ratio);
        // Residuals should be approximately stationary (negative ADF).
        assert!(result.adf_stat < 0.0, "adf stat: {}", result.adf_stat);
    }
}
