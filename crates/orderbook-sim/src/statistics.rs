/// Running statistics for the LOB: VWAP, realized vol, spread decomposition,
/// Amihud illiquidity, Kyle lambda, Roll estimator, trade arrival rates.

use std::collections::VecDeque;
use crate::lob_engine::{Side, Qty};

// ── Circular Buffer ───────────────────────────────────────────────────────────

/// Fixed-size circular buffer for running window statistics.
pub struct CircularBuffer<T: Copy + Default> {
    data: VecDeque<T>,
    capacity: usize,
}

impl<T: Copy + Default> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        CircularBuffer { data: VecDeque::with_capacity(capacity), capacity }
    }

    pub fn push(&mut self, value: T) {
        if self.data.len() == self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_full(&self) -> bool {
        self.data.len() == self.capacity
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn as_slice_vec(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }

    pub fn last(&self) -> Option<T> {
        self.data.back().copied()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

// ── Trade Record ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct TradeRecord {
    pub timestamp: u64,  // nanoseconds
    pub price: f64,
    pub qty: Qty,
    pub side: Side,
}

// ── Running VWAP ──────────────────────────────────────────────────────────────

/// Windowed VWAP calculator.
pub struct RunningVwap {
    window: CircularBuffer<TradeRecord>,
    notional_sum: f64,
    qty_sum: Qty,
}

impl RunningVwap {
    pub fn new(window_size: usize) -> Self {
        RunningVwap {
            window: CircularBuffer::new(window_size),
            notional_sum: 0.0,
            qty_sum: 0.0,
        }
    }

    pub fn add_trade(&mut self, trade: TradeRecord) {
        // If buffer is full, remove the oldest contribution.
        if self.window.is_full() {
            if let Some(old) = self.window.data.front() {
                self.notional_sum -= old.price * old.qty;
                self.qty_sum -= old.qty;
            }
        }
        self.notional_sum += trade.price * trade.qty;
        self.qty_sum += trade.qty;
        self.window.push(trade);
    }

    pub fn vwap(&self) -> Option<f64> {
        if self.qty_sum < 1e-9 { None } else { Some(self.notional_sum / self.qty_sum) }
    }

    pub fn total_qty(&self) -> Qty { self.qty_sum }
    pub fn count(&self) -> usize { self.window.len() }
}

// ── Realized Volatility ───────────────────────────────────────────────────────

/// Running realized volatility using log-returns over a rolling window.
pub struct RealizedVol {
    mid_prices: CircularBuffer<f64>,
    /// Annualization factor (e.g., 252 * trades_per_day, or 252*390*60 for per-minute).
    ann_factor: f64,
}

impl RealizedVol {
    pub fn new(window: usize, ann_factor: f64) -> Self {
        RealizedVol { mid_prices: CircularBuffer::new(window + 1), ann_factor }
    }

    pub fn update(&mut self, mid: f64) {
        self.mid_prices.push(mid);
    }

    /// Realized volatility (annualized) from current window.
    pub fn realized_vol(&self) -> f64 {
        let prices = self.mid_prices.as_slice_vec();
        if prices.len() < 2 { return 0.0; }
        let n = prices.len() - 1;
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        let mean = returns.iter().sum::<f64>() / n as f64;
        let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;
        (var * self.ann_factor).sqrt()
    }

    /// Realized variance (annualized).
    pub fn realized_variance(&self) -> f64 {
        let rv = self.realized_vol();
        rv * rv
    }

    /// Parkinson high-low vol estimator (needs H/L data; approximated here).
    pub fn yang_zhang_vol(&self) -> f64 {
        // Without OHLC, just return realized vol.
        self.realized_vol()
    }
}

// ── Bid-Ask Spread Decomposition ──────────────────────────────────────────────

/// Decompose bid-ask spread into adverse selection and order-processing components.
///
/// Based on Glosten-Harris (1988) decomposition.
#[derive(Debug, Clone, Default)]
pub struct SpreadDecomposition {
    /// Effective half-spread history.
    effective_half_spreads: Vec<f64>,
    /// Realized spreads (over a fixed horizon τ).
    realized_spreads: Vec<f64>,
    /// Price impact (adverse selection) component.
    price_impacts: Vec<f64>,
    /// Quote data.
    quotes: Vec<(f64, f64)>,   // (bid, ask) pairs
    /// Matched trades.
    trade_prices: Vec<f64>,
    trade_sides: Vec<Side>,
}

impl SpreadDecomposition {
    pub fn new() -> Self { Self::default() }

    pub fn add_quote(&mut self, bid: f64, ask: f64) {
        self.quotes.push((bid, ask));
    }

    pub fn add_trade(&mut self, price: f64, side: Side) {
        self.trade_prices.push(price);
        self.trade_sides.push(side);
    }

    /// Compute effective spread for a trade vs the current quote.
    pub fn effective_spread(&self, trade_price: f64, side: Side, bid: f64, ask: f64) -> f64 {
        let mid = (bid + ask) / 2.0;
        match side {
            Side::Bid => 2.0 * (trade_price - mid),   // buy: price above mid
            Side::Ask => 2.0 * (mid - trade_price),   // sell: price below mid
        }
    }

    /// Realized spread: effective spread minus price impact.
    /// Price impact = mid-price change from t to t+τ.
    pub fn realized_spread_from_pair(
        effective: f64,
        mid_at_trade: f64,
        mid_after_tau: f64,
        side: Side,
    ) -> f64 {
        let impact = match side {
            Side::Bid => mid_after_tau - mid_at_trade,
            Side::Ask => mid_at_trade - mid_after_tau,
        };
        effective - 2.0 * impact
    }

    /// Compute decomposition from stored data.
    pub fn compute(&mut self, tau_steps: usize) -> SpreadComponents {
        let n_trades = self.trade_prices.len().min(self.quotes.len());
        if n_trades == 0 {
            return SpreadComponents::default();
        }

        let mut eff_spreads = Vec::new();
        let mut price_impacts = Vec::new();
        let mut realized_spreads = Vec::new();

        for i in 0..n_trades {
            let (bid, ask) = self.quotes[i.min(self.quotes.len() - 1)];
            let mid = (bid + ask) / 2.0;
            let price = self.trade_prices[i];
            let side = self.trade_sides[i];

            let eff = self.effective_spread(price, side, bid, ask);
            eff_spreads.push(eff);

            // For price impact, look tau_steps ahead.
            let future_idx = (i + tau_steps).min(self.quotes.len() - 1);
            let (bid_f, ask_f) = self.quotes[future_idx];
            let mid_f = (bid_f + ask_f) / 2.0;

            let impact = match side {
                Side::Bid => 2.0 * (mid_f - mid),
                Side::Ask => 2.0 * (mid - mid_f),
            };
            price_impacts.push(impact);
            realized_spreads.push(eff - impact);
        }

        let mean_eff = mean(&eff_spreads);
        let mean_impact = mean(&price_impacts);
        let mean_realized = mean(&realized_spreads);

        SpreadComponents {
            mean_effective: mean_eff,
            mean_price_impact: mean_impact,
            mean_realized: mean_realized,
            adverse_selection_pct: if mean_eff.abs() > 1e-12 { mean_impact / mean_eff } else { 0.0 },
            n_obs: n_trades,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SpreadComponents {
    pub mean_effective: f64,
    pub mean_price_impact: f64,
    pub mean_realized: f64,
    pub adverse_selection_pct: f64,
    pub n_obs: usize,
}

// ── Amihud Illiquidity ────────────────────────────────────────────────────────

/// Amihud (2002) illiquidity ratio: |r| / volume.
/// Measures price impact per dollar volume.
pub struct AmihudEstimator {
    observations: Vec<(f64, f64)>,  // (|return|, dollar_volume)
}

impl AmihudEstimator {
    pub fn new() -> Self { AmihudEstimator { observations: Vec::new() } }

    pub fn add_observation(&mut self, abs_return: f64, dollar_volume: f64) {
        if dollar_volume > 1e-9 {
            self.observations.push((abs_return, dollar_volume));
        }
    }

    pub fn add_from_trades(&mut self, trades: &[TradeRecord], window_prices: &[f64]) {
        if trades.is_empty() || window_prices.len() < 2 { return; }
        let dollar_vol: f64 = trades.iter().map(|t| t.price * t.qty).sum();
        let p_start = window_prices[0];
        let p_end = window_prices[window_prices.len() - 1];
        if p_start > 1e-9 {
            let abs_ret = (p_end / p_start).ln().abs();
            self.add_observation(abs_ret, dollar_vol);
        }
    }

    /// Amihud illiquidity ratio: mean(|r_t| / DVOL_t).
    pub fn illiquidity(&self) -> f64 {
        if self.observations.is_empty() { return 0.0; }
        let n = self.observations.len() as f64;
        self.observations.iter().map(|(r, v)| r / v).sum::<f64>() / n
    }

    /// Normalized illiquidity (×10^6 for readability).
    pub fn illiquidity_scaled(&self) -> f64 {
        self.illiquidity() * 1e6
    }

    pub fn count(&self) -> usize { self.observations.len() }

    pub fn clear(&mut self) { self.observations.clear(); }
}

impl Default for AmihudEstimator {
    fn default() -> Self { Self::new() }
}

// ── Kyle Lambda ───────────────────────────────────────────────────────────────

/// Kyle (1985) lambda: price impact per unit of signed order flow.
///
/// Estimated via OLS: Δp = λ · Q + ε
/// where Q = signed order flow (buys minus sells) and Δp = price change.
pub struct KyleLambdaEstimator {
    price_changes: Vec<f64>,
    order_flows: Vec<f64>,
}

impl KyleLambdaEstimator {
    pub fn new() -> Self {
        KyleLambdaEstimator { price_changes: Vec::new(), order_flows: Vec::new() }
    }

    pub fn add_observation(&mut self, price_change: f64, signed_order_flow: f64) {
        self.price_changes.push(price_change);
        self.order_flows.push(signed_order_flow);
    }

    pub fn add_from_window(
        &mut self,
        trades: &[TradeRecord],
        mid_before: f64,
        mid_after: f64,
    ) {
        let signed_flow: f64 = trades.iter().map(|t| match t.side {
            Side::Bid => t.qty,
            Side::Ask => -t.qty,
        }).sum();
        self.add_observation(mid_after - mid_before, signed_flow);
    }

    /// Estimate lambda via OLS regression Δp ~ λ·Q.
    /// Returns (lambda, r_squared).
    pub fn estimate(&self) -> (f64, f64) {
        let n = self.price_changes.len();
        if n < 3 { return (0.0, 0.0); }

        let dp = &self.price_changes;
        let q = &self.order_flows;

        // OLS: λ = Cov(Δp, Q) / Var(Q)
        let mean_q = mean(q);
        let mean_dp = mean(dp);

        let cov = q.iter().zip(dp.iter())
            .map(|(&qi, &dpi)| (qi - mean_q) * (dpi - mean_dp))
            .sum::<f64>() / (n - 1) as f64;

        let var_q = q.iter().map(|&qi| (qi - mean_q).powi(2)).sum::<f64>() / (n - 1) as f64;

        if var_q.abs() < 1e-20 { return (0.0, 0.0); }

        let lambda = cov / var_q;

        // R² = (lambda * var_q / var_dp)
        let var_dp = dp.iter().map(|&d| (d - mean_dp).powi(2)).sum::<f64>() / (n - 1) as f64;
        let r_sq = if var_dp.abs() < 1e-20 { 0.0 } else { lambda * lambda * var_q / var_dp };

        (lambda, r_sq.min(1.0))
    }

    pub fn clear(&mut self) {
        self.price_changes.clear();
        self.order_flows.clear();
    }

    pub fn count(&self) -> usize { self.price_changes.len() }
}

impl Default for KyleLambdaEstimator {
    fn default() -> Self { Self::new() }
}

// ── Roll Estimator ────────────────────────────────────────────────────────────

/// Roll (1984) effective spread estimator.
///
/// Spread estimate: s = 2·√(−Cov(Δp_t, Δp_{t-1}))
/// Based on the serial covariance of price changes.
pub struct RollEstimator {
    price_changes: CircularBuffer<f64>,
}

impl RollEstimator {
    pub fn new(window: usize) -> Self {
        RollEstimator { price_changes: CircularBuffer::new(window) }
    }

    pub fn update(&mut self, price_change: f64) {
        self.price_changes.push(price_change);
    }

    /// Estimate effective spread using Roll's serial covariance method.
    pub fn effective_spread(&self) -> f64 {
        let changes = self.price_changes.as_slice_vec();
        let n = changes.len();
        if n < 4 { return 0.0; }

        // Serial covariance: Cov(Δp_t, Δp_{t-1}).
        let n_pairs = n - 1;
        let mean1 = changes[..n_pairs].iter().sum::<f64>() / n_pairs as f64;
        let mean2 = changes[1..].iter().sum::<f64>() / n_pairs as f64;

        let cov: f64 = changes[..n_pairs].iter().zip(changes[1..].iter())
            .map(|(&a, &b)| (a - mean1) * (b - mean2))
            .sum::<f64>() / (n_pairs - 1) as f64;

        if cov >= 0.0 {
            // Positive serial covariance → Roll's formula breaks down.
            // Fall back to range-based estimate.
            let range = changes.iter().map(|&c| c.abs()).sum::<f64>() / n as f64;
            return range * 2.0;
        }

        2.0 * (-cov).sqrt()
    }

    pub fn half_spread(&self) -> f64 {
        self.effective_spread() / 2.0
    }
}

// ── Trade Arrival Rate ────────────────────────────────────────────────────────

/// Estimate trade arrival rate (trades per second) using exponential moving average.
pub struct ArrivalRateEstimator {
    /// EMA of inter-arrival times in nanoseconds.
    ema_interarrival_ns: f64,
    /// Smoothing factor α.
    alpha: f64,
    last_ts: u64,
    count: u64,
}

impl ArrivalRateEstimator {
    pub fn new(alpha: f64) -> Self {
        ArrivalRateEstimator { ema_interarrival_ns: 1e9, alpha, last_ts: 0, count: 0 }
    }

    pub fn record_event(&mut self, timestamp_ns: u64) {
        if self.last_ts > 0 && timestamp_ns > self.last_ts {
            let dt = (timestamp_ns - self.last_ts) as f64;
            self.ema_interarrival_ns = self.alpha * dt + (1.0 - self.alpha) * self.ema_interarrival_ns;
        }
        self.last_ts = timestamp_ns;
        self.count += 1;
    }

    /// Estimated arrival rate in events per second.
    pub fn rate_per_second(&self) -> f64 {
        if self.ema_interarrival_ns < 1e-9 { return 0.0; }
        1e9 / self.ema_interarrival_ns
    }

    pub fn count(&self) -> u64 { self.count }
}

// ── Comprehensive LOB Statistics ──────────────────────────────────────────────

pub struct LobStatistics {
    pub symbol: String,
    pub vwap: RunningVwap,
    pub realized_vol: RealizedVol,
    pub spread_decomp: SpreadDecomposition,
    pub amihud: AmihudEstimator,
    pub kyle_lambda: KyleLambdaEstimator,
    pub roll: RollEstimator,
    pub buy_arrival: ArrivalRateEstimator,
    pub sell_arrival: ArrivalRateEstimator,
    pub all_arrival: ArrivalRateEstimator,
    /// Running tick-by-tick spread history.
    spreads: CircularBuffer<f64>,
    /// Count of all events.
    pub event_count: u64,
    /// Last known mid-price.
    pub last_mid: f64,
    /// Last known spread.
    pub last_spread: f64,
}

impl LobStatistics {
    pub fn new(symbol: impl Into<String>, vol_window: usize, vwap_window: usize) -> Self {
        LobStatistics {
            symbol: symbol.into(),
            vwap: RunningVwap::new(vwap_window),
            realized_vol: RealizedVol::new(vol_window, 252.0 * 390.0 * 60.0),
            spread_decomp: SpreadDecomposition::new(),
            amihud: AmihudEstimator::new(),
            kyle_lambda: KyleLambdaEstimator::new(),
            roll: RollEstimator::new(vol_window),
            buy_arrival: ArrivalRateEstimator::new(0.1),
            sell_arrival: ArrivalRateEstimator::new(0.1),
            all_arrival: ArrivalRateEstimator::new(0.1),
            spreads: CircularBuffer::new(vol_window),
            event_count: 0,
            last_mid: 0.0,
            last_spread: 0.0,
        }
    }

    pub fn update_quote(&mut self, bid: f64, ask: f64) {
        let mid = (bid + ask) / 2.0;
        let spread = ask - bid;

        if self.last_mid > 1e-9 {
            let dp = mid - self.last_mid;
            self.roll.update(dp);
            self.spread_decomp.add_quote(bid, ask);
        }

        self.realized_vol.update(mid);
        self.spreads.push(spread);
        self.last_mid = mid;
        self.last_spread = spread;
        self.event_count += 1;
    }

    pub fn record_trade(&mut self, trade: TradeRecord) {
        self.vwap.add_trade(trade);
        self.spread_decomp.add_trade(trade.price, trade.side);
        self.all_arrival.record_event(trade.timestamp);
        match trade.side {
            Side::Bid => self.buy_arrival.record_event(trade.timestamp),
            Side::Ask => self.sell_arrival.record_event(trade.timestamp),
        }
        self.event_count += 1;
    }

    pub fn update_kyle_lambda(&mut self, trades: &[TradeRecord], mid_before: f64, mid_after: f64) {
        self.kyle_lambda.add_from_window(trades, mid_before, mid_after);
    }

    pub fn update_amihud(&mut self, trades: &[TradeRecord], prices: &[f64]) {
        self.amihud.add_from_trades(trades, prices);
    }

    pub fn mean_spread(&self) -> f64 {
        let s = self.spreads.as_slice_vec();
        if s.is_empty() { return 0.0; }
        mean(&s)
    }

    pub fn snapshot(&self) -> StatSnapshot {
        let (lambda, r_sq) = self.kyle_lambda.estimate();
        StatSnapshot {
            symbol: self.symbol.clone(),
            vwap: self.vwap.vwap().unwrap_or(0.0),
            realized_vol: self.realized_vol.realized_vol(),
            realized_variance: self.realized_vol.realized_variance(),
            mean_spread: self.mean_spread(),
            roll_spread: self.roll.effective_spread(),
            amihud_illiquidity: self.amihud.illiquidity_scaled(),
            kyle_lambda: lambda,
            kyle_r_squared: r_sq,
            buy_rate: self.buy_arrival.rate_per_second(),
            sell_rate: self.sell_arrival.rate_per_second(),
            all_rate: self.all_arrival.rate_per_second(),
            last_mid: self.last_mid,
            last_spread: self.last_spread,
            event_count: self.event_count,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StatSnapshot {
    pub symbol: String,
    pub vwap: f64,
    pub realized_vol: f64,
    pub realized_variance: f64,
    pub mean_spread: f64,
    pub roll_spread: f64,
    pub amihud_illiquidity: f64,
    pub kyle_lambda: f64,
    pub kyle_r_squared: f64,
    pub buy_rate: f64,
    pub sell_rate: f64,
    pub all_rate: f64,
    pub last_mid: f64,
    pub last_spread: f64,
    pub event_count: u64,
}

// ── Utility ───────────────────────────────────────────────────────────────────

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn variance(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}

fn std_dev(v: &[f64]) -> f64 { variance(v).sqrt() }

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_vwap() {
        let mut vwap = RunningVwap::new(10);
        for i in 1..=5 {
            vwap.add_trade(TradeRecord {
                timestamp: i as u64 * 1_000_000,
                price: 100.0 + i as f64,
                qty: 10.0,
                side: Side::Bid,
            });
        }
        let v = vwap.vwap().unwrap();
        // (101+102+103+104+105)/5 = 103
        assert!((v - 103.0).abs() < 1e-9);
    }

    #[test]
    fn test_realized_vol() {
        let mut rv = RealizedVol::new(100, 252.0);
        for i in 0..200 {
            let p = 100.0 + (i as f64 * 0.01).sin() * 2.0;
            rv.update(p);
        }
        let vol = rv.realized_vol();
        assert!(vol >= 0.0);
    }

    #[test]
    fn test_kyle_lambda_positive() {
        let mut est = KyleLambdaEstimator::new();
        // Simulate: positive order flow → positive price change.
        for i in 0..50 {
            let flow = (i as f64 - 25.0) * 10.0;
            let dp = flow * 0.001 + (i as f64 * 0.001).sin() * 0.1;
            est.add_observation(dp, flow);
        }
        let (lambda, r_sq) = est.estimate();
        assert!(lambda > 0.0, "lambda should be positive: {}", lambda);
        assert!(r_sq >= 0.0 && r_sq <= 1.0);
    }

    #[test]
    fn test_roll_estimator() {
        let mut roll = RollEstimator::new(50);
        // Simulate bid-ask bounce with spread = 0.02.
        for i in 0..100 {
            let bounce: f64 = if i % 2 == 0 { 0.01 } else { -0.01 };
            roll.update(bounce);
        }
        let spread = roll.effective_spread();
        // Should recover ~0.02.
        assert!(spread > 0.0);
    }

    #[test]
    fn test_amihud_illiquidity() {
        let mut amihud = AmihudEstimator::new();
        for i in 1..=20 {
            amihud.add_observation(0.001 * i as f64, 1_000_000.0 * i as f64);
        }
        let ill = amihud.illiquidity();
        assert!(ill >= 0.0);
        assert!(ill < 1.0); // Should be tiny for large volumes.
    }

    #[test]
    fn test_arrival_rate() {
        let mut rate = ArrivalRateEstimator::new(0.1);
        // 10 events per second = 100ms inter-arrival.
        let base_ts = 1_000_000_000u64;
        for i in 0..20 {
            rate.record_event(base_ts + i * 100_000_000);
        }
        let r = rate.rate_per_second();
        // Should be close to 10.
        assert!(r > 5.0 && r < 20.0, "rate: {}", r);
    }

    #[test]
    fn test_lob_statistics_snapshot() {
        let mut stats = LobStatistics::new("SPY", 50, 20);
        for i in 0..100 {
            let mid = 400.0 + (i as f64 * 0.1).sin() * 2.0;
            stats.update_quote(mid - 0.01, mid + 0.01);
            let trade = TradeRecord {
                timestamp: i as u64 * 1_000_000,
                price: mid,
                qty: 100.0,
                side: if i % 2 == 0 { Side::Bid } else { Side::Ask },
            };
            stats.record_trade(trade);
        }
        let snap = stats.snapshot();
        assert!(snap.mean_spread > 0.0);
        assert!(snap.realized_vol >= 0.0);
        assert!(snap.vwap > 0.0);
    }

    #[test]
    fn test_spread_decomposition() {
        let mut decomp = SpreadDecomposition::new();
        for i in 0..50 {
            let mid = 100.0 + (i as f64 * 0.01).sin();
            decomp.add_quote(mid - 0.02, mid + 0.02);
            decomp.add_trade(mid + 0.01, Side::Bid);
        }
        let components = decomp.compute(5);
        assert!(components.mean_effective >= 0.0);
    }
}
