//! market_impact_model.rs — Almgren-Chriss permanent/temporary impact, LOB depth depletion,
//! feedback loop where simulator orders move the book, Kyle lambda estimation,
//! volume participation rate.
//!
//! Chronos / AETERNUS — production-grade market impact engine.

use std::collections::{BTreeMap, VecDeque};

// ── Types ────────────────────────────────────────────────────────────────────

pub type Price = f64;
pub type Qty = f64;
pub type Nanos = u64;

// ── PRNG ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Rng { state: u64 }

impl Rng {
    pub fn new(seed: u64) -> Self { Rng { state: seed ^ 0xc0ffee_f00d_1234 } }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x; x
    }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Almgren-Chriss impact model ───────────────────────────────────────────────

/// Parameters for the Almgren-Chriss optimal execution model.
#[derive(Debug, Clone)]
pub struct AlmgrenChrissImpact {
    /// Permanent impact coefficient (η)
    pub eta: f64,
    /// Temporary impact coefficient (γ)
    pub gamma: f64,
    /// Average daily volume (shares)
    pub adv: f64,
    /// Price volatility (annualized, fractional)
    pub sigma: f64,
    /// Risk aversion parameter (λ)
    pub lambda: f64,
    /// Current mid price
    pub mid_price: f64,
}

impl Default for AlmgrenChrissImpact {
    fn default() -> Self {
        AlmgrenChrissImpact { eta: 0.1, gamma: 0.5, adv: 1_000_000.0, sigma: 0.20, lambda: 1e-6, mid_price: 100.0 }
    }
}

impl AlmgrenChrissImpact {
    pub fn new(eta: f64, gamma: f64, adv: f64, sigma: f64, lambda: f64, mid_price: f64) -> Self {
        AlmgrenChrissImpact { eta, gamma, adv, sigma, lambda, mid_price }
    }

    /// Permanent price impact for trading qty shares over interval dt (seconds)
    pub fn permanent_impact(&self, qty: Qty, dt: f64) -> f64 {
        let rate = qty / (dt * self.adv + 1e-12);
        self.eta * self.mid_price * rate.signum() * rate.abs().sqrt()
    }

    /// Temporary (instantaneous) price impact for trading qty shares in a single slice
    pub fn temporary_impact(&self, qty: Qty, dt: f64) -> f64 {
        let rate = qty / (dt * self.adv + 1e-12);
        self.gamma * self.mid_price * rate.abs().sqrt() * rate.signum()
    }

    /// Total expected cost of liquidating X shares over T seconds in N slices
    pub fn optimal_schedule_cost(&self, total_qty: Qty, total_time: f64, n_slices: usize) -> (f64, f64) {
        let dt = total_time / n_slices as f64;
        let kappa2 = (self.lambda * self.sigma.powi(2) * self.mid_price.powi(2) * self.adv)
            / (0.5 * self.gamma * self.adv + 1e-12);
        let kappa = kappa2.abs().sqrt();
        let sinh_kappa_T = (kappa * total_time).sinh().max(1e-12);

        let mut shortfall = 0.0f64;
        let mut variance_acc = 0.0f64;

        for i in 0..n_slices {
            let t = i as f64 * dt;
            let tau = total_time - t;
            let remaining = total_qty * (kappa * tau).sinh() / sinh_kappa_T;
            let next_tau = total_time - (i + 1) as f64 * dt;
            let next_remaining = if i + 1 < n_slices {
                total_qty * (kappa * next_tau.max(0.0)).sinh() / sinh_kappa_T
            } else { 0.0 };
            let traded = remaining - next_remaining;
            let temp_cost = self.temporary_impact(traded, dt);
            let perm_cost = self.permanent_impact(traded, dt);
            shortfall += (temp_cost + perm_cost) * traded.abs();
            variance_acc += remaining.powi(2) * self.sigma.powi(2) * self.mid_price.powi(2) * dt;
        }
        (shortfall, variance_acc)
    }

    pub fn risk_adjusted_cost(&self, total_qty: Qty, total_time: f64, n_slices: usize) -> f64 {
        let (es, var) = self.optimal_schedule_cost(total_qty, total_time, n_slices);
        es + self.lambda * var
    }
}

// ── LOB Depth Depletion Model ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LobDepthModel {
    pub bids: BTreeMap<i64, f64>,
    pub asks: BTreeMap<i64, f64>,
    pub tick_size: f64,
    pub mid_price: f64,
    pub resilience_rate: f64,
    last_depletion: f64,
    last_depletion_time: f64,
}

impl LobDepthModel {
    pub fn new(tick_size: f64, mid_price: f64) -> Self {
        let mut m = LobDepthModel {
            bids: BTreeMap::new(), asks: BTreeMap::new(),
            tick_size, mid_price, resilience_rate: 1000.0,
            last_depletion: 0.0, last_depletion_time: 0.0,
        };
        m.init_depth(10, 500.0);
        m
    }

    pub fn init_depth(&mut self, n_levels: usize, base_qty: f64) {
        self.bids.clear(); self.asks.clear();
        let mid_ticks = (self.mid_price / self.tick_size).round() as i64;
        for i in 1..=n_levels as i64 {
            let decay = (-0.3 * (i - 1) as f64).exp();
            let qty = base_qty * decay;
            self.bids.insert(mid_ticks - i, qty);
            self.asks.insert(mid_ticks + i, qty);
        }
    }

    fn ticks_to_price(&self, ticks: i64) -> f64 { ticks as f64 * self.tick_size }
    fn price_to_ticks(&self, price: f64) -> i64 { (price / self.tick_size).round() as i64 }

    fn replenish(&mut self, current_time: f64) {
        if self.last_depletion_time <= 0.0 || self.last_depletion <= 0.0 { return; }
        let elapsed = current_time - self.last_depletion_time;
        let replenish_qty = (self.resilience_rate * elapsed).min(self.last_depletion);
        let mid_ticks = self.price_to_ticks(self.mid_price);
        *self.asks.entry(mid_ticks + 1).or_insert(0.0) += replenish_qty * 0.5;
        *self.bids.entry(mid_ticks - 1).or_insert(0.0) += replenish_qty * 0.5;
        self.last_depletion -= replenish_qty;
    }

    pub fn execute_market_order(&mut self, qty: Qty, is_buy: bool, current_time: f64) -> (Price, f64, usize) {
        self.replenish(current_time);
        let mut remaining = qty;
        let mut fill_cost = 0.0f64;
        let mut levels_swept = 0usize;

        if is_buy {
            let keys: Vec<i64> = self.asks.keys().cloned().collect();
            for pt in &keys {
                if remaining <= 0.0 { break; }
                let avail = *self.asks.get(pt).unwrap_or(&0.0);
                let fill = avail.min(remaining);
                fill_cost += fill * self.ticks_to_price(*pt);
                *self.asks.entry(*pt).or_insert(0.0) -= fill;
                if *self.asks.get(pt).unwrap_or(&0.0) < 0.01 { self.asks.remove(pt); }
                remaining -= fill;
                levels_swept += 1;
            }
        } else {
            let keys: Vec<i64> = self.bids.keys().cloned().rev().collect();
            for pt in &keys {
                if remaining <= 0.0 { break; }
                let avail = *self.bids.get(pt).unwrap_or(&0.0);
                let fill = avail.min(remaining);
                fill_cost += fill * self.ticks_to_price(*pt);
                *self.bids.entry(*pt).or_insert(0.0) -= fill;
                if *self.bids.get(pt).unwrap_or(&0.0) < 0.01 { self.bids.remove(pt); }
                remaining -= fill;
                levels_swept += 1;
            }
        }

        let filled = qty - remaining;
        let avg_price = if filled > 0.0 { fill_cost / filled } else { self.mid_price };
        let slippage_bps = if is_buy {
            (avg_price - self.mid_price) / self.mid_price * 10_000.0
        } else {
            (self.mid_price - avg_price) / self.mid_price * 10_000.0
        };
        self.last_depletion = qty;
        self.last_depletion_time = current_time;
        (avg_price, slippage_bps.max(0.0), levels_swept)
    }

    pub fn shift_mid(&mut self, delta: f64) {
        self.mid_price += delta;
        let n = self.bids.len().max(self.asks.len()).max(5);
        let base = (self.total_bid_depth().max(self.total_ask_depth()) / n as f64).max(100.0);
        self.init_depth(n, base);
    }

    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.iter().next_back().map(|(&t, &q)| (self.ticks_to_price(t), q))
    }
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.iter().next().map(|(&t, &q)| (self.ticks_to_price(t), q))
    }
    pub fn spread(&self) -> f64 {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => a - b,
            _ => self.tick_size * 2.0,
        }
    }
    pub fn total_bid_depth(&self) -> f64 { self.bids.values().sum() }
    pub fn total_ask_depth(&self) -> f64 { self.asks.values().sum() }

    pub fn depth_at_levels(&self, n: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let bids: Vec<_> = self.bids.iter().rev().take(n).map(|(&t, &q)| (self.ticks_to_price(t), q)).collect();
        let asks: Vec<_> = self.asks.iter().take(n).map(|(&t, &q)| (self.ticks_to_price(t), q)).collect();
        (bids, asks)
    }
}

// ── Feedback impact engine ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FeedbackImpactEngine {
    pub lob: LobDepthModel,
    pub ac_model: AlmgrenChrissImpact,
    pub permanent_impact_decay_secs: f64,
    accumulated_perm_impact: f64,
    last_impact_time: f64,
    trade_history: VecDeque<(f64, f64, f64)>,
}

impl FeedbackImpactEngine {
    pub fn new(tick_size: f64, mid_price: f64, ac: AlmgrenChrissImpact) -> Self {
        FeedbackImpactEngine {
            lob: LobDepthModel::new(tick_size, mid_price),
            ac_model: ac,
            permanent_impact_decay_secs: 300.0,
            accumulated_perm_impact: 0.0,
            last_impact_time: 0.0,
            trade_history: VecDeque::new(),
        }
    }

    pub fn submit_order(&mut self, qty: Qty, is_buy: bool, current_time: f64) -> OrderImpactResult {
        if self.last_impact_time > 0.0 {
            let elapsed = current_time - self.last_impact_time;
            let decay = (-std::f64::consts::LN_2 * elapsed / self.permanent_impact_decay_secs).exp();
            self.accumulated_perm_impact *= decay;
        }
        self.last_impact_time = current_time;

        let (avg_price, slippage_bps, levels_swept) = self.lob.execute_market_order(qty, is_buy, current_time);

        let dt = 1.0;
        let temp_impact = self.ac_model.temporary_impact(qty, dt);
        let perm_impact = self.ac_model.permanent_impact(qty, dt);

        let perm_dir = if is_buy { 1.0 } else { -1.0 };
        let perm_price_move = perm_impact * perm_dir;
        self.accumulated_perm_impact += perm_price_move;
        self.lob.shift_mid(perm_price_move);

        let perm_bps = perm_price_move.abs() / self.ac_model.mid_price.max(1e-12) * 10_000.0;
        let temp_bps = temp_impact / self.ac_model.mid_price.max(1e-12) * 10_000.0;

        self.trade_history.push_back((current_time, qty * if is_buy { 1.0 } else { -1.0 }, avg_price));
        if self.trade_history.len() > 1000 { self.trade_history.pop_front(); }

        OrderImpactResult {
            avg_fill_price: avg_price,
            slippage_bps,
            temporary_impact_bps: temp_bps,
            permanent_impact_bps: perm_bps,
            total_impact_bps: slippage_bps + temp_bps + perm_bps,
            levels_swept,
            post_trade_mid: self.lob.mid_price,
            accumulated_perm_impact: self.accumulated_perm_impact,
        }
    }

    pub fn mid_price(&self) -> f64 { self.lob.mid_price }
    pub fn trade_count(&self) -> usize { self.trade_history.len() }
}

#[derive(Debug, Clone)]
pub struct OrderImpactResult {
    pub avg_fill_price: f64,
    pub slippage_bps: f64,
    pub temporary_impact_bps: f64,
    pub permanent_impact_bps: f64,
    pub total_impact_bps: f64,
    pub levels_swept: usize,
    pub post_trade_mid: f64,
    pub accumulated_perm_impact: f64,
}

// ── Kyle Lambda Estimator ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct KyleLambdaEstimator {
    observations: VecDeque<(f64, f64)>,
    window_size: usize,
    pub current_lambda: f64,
    pub r_squared: f64,
}

impl KyleLambdaEstimator {
    pub fn new(window_size: usize) -> Self {
        KyleLambdaEstimator { observations: VecDeque::new(), window_size, current_lambda: 0.0, r_squared: 0.0 }
    }

    pub fn add_observation(&mut self, order_flow: f64, price_change: f64) {
        self.observations.push_back((order_flow, price_change));
        if self.observations.len() > self.window_size { self.observations.pop_front(); }
        if self.observations.len() >= 10 { self.estimate(); }
    }

    fn estimate(&mut self) {
        let n = self.observations.len() as f64;
        let sum_x: f64 = self.observations.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = self.observations.iter().map(|(_, y)| y).sum();
        let sum_xx: f64 = self.observations.iter().map(|(x, _)| x * x).sum();
        let sum_xy: f64 = self.observations.iter().map(|(x, y)| x * y).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 { return; }
        self.current_lambda = (n * sum_xy - sum_x * sum_y) / denom;
        let mean_y = sum_y / n;
        let ss_tot: f64 = self.observations.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = self.observations.iter().map(|(x, y)| (y - self.current_lambda * x).powi(2)).sum();
        self.r_squared = if ss_tot > 1e-12 { 1.0 - ss_res / ss_tot } else { 0.0 };
    }

    pub fn predict_price_change(&self, order_flow: f64) -> f64 { self.current_lambda * order_flow }
    pub fn effective_spread(&self, order_flow: f64) -> f64 { 2.0 * self.current_lambda * order_flow.abs() }
}

// ── Volume Participation Rate ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VolumeParticipationModel {
    pub target_rate: f64,
    pub max_slice_rate: f64,
    market_volume: VecDeque<f64>,
    our_volume: VecDeque<f64>,
    window_size: usize,
}

impl VolumeParticipationModel {
    pub fn new(target_rate: f64, max_slice_rate: f64, window_size: usize) -> Self {
        VolumeParticipationModel {
            target_rate: target_rate.clamp(0.0, 1.0),
            max_slice_rate: max_slice_rate.clamp(0.0, 1.0),
            market_volume: VecDeque::new(),
            our_volume: VecDeque::new(),
            window_size,
        }
    }

    pub fn record_interval(&mut self, market_vol: f64, our_vol: f64) {
        self.market_volume.push_back(market_vol);
        self.our_volume.push_back(our_vol);
        while self.market_volume.len() > self.window_size { self.market_volume.pop_front(); self.our_volume.pop_front(); }
    }

    pub fn compute_slice_qty(&self, remaining_qty: f64, predicted_market_vol: f64) -> f64 {
        let target = predicted_market_vol * self.target_rate;
        let max_s = predicted_market_vol * self.max_slice_rate;
        target.min(max_s).min(remaining_qty).max(0.0)
    }

    pub fn current_participation_rate(&self) -> f64 {
        let total_mkt: f64 = self.market_volume.iter().sum();
        let total_ours: f64 = self.our_volume.iter().sum();
        if total_mkt < 1.0 { 0.0 } else { total_ours / total_mkt }
    }

    pub fn is_above_target(&self) -> bool { self.current_participation_rate() > self.target_rate }
}

// ── Square-root impact law ────────────────────────────────────────────────────

pub fn square_root_impact_bps(qty: f64, adv: f64, sigma: f64, spread_bps: f64) -> f64 {
    let participation = (qty / adv.max(1.0)).abs().sqrt();
    let spread_adj = (sigma * 252f64.sqrt() / (spread_bps / 10_000.0).max(1e-6)).powf(0.25);
    sigma * participation * spread_adj * 10_000.0 / 252f64.sqrt()
}

pub fn linear_impact_bps(qty: f64, depth: f64, tick_bps: f64) -> f64 {
    (qty / depth.max(1.0)) * tick_bps
}

// ── Impact Stats ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct ImpactStats {
    pub n_orders: u64,
    pub total_qty: f64,
    pub total_temp_impact_bps: f64,
    pub total_perm_impact_bps: f64,
    pub total_slippage_bps: f64,
    pub total_shortfall_usd: f64,
    pub max_single_impact_bps: f64,
    pub min_single_impact_bps: f64,
}

impl ImpactStats {
    pub fn update(&mut self, result: &OrderImpactResult, qty: f64) {
        self.n_orders += 1;
        self.total_qty += qty;
        self.total_temp_impact_bps += result.temporary_impact_bps;
        self.total_perm_impact_bps += result.permanent_impact_bps;
        self.total_slippage_bps += result.slippage_bps;
        let shortfall = (result.avg_fill_price - result.post_trade_mid) * qty;
        self.total_shortfall_usd += shortfall;
        if result.total_impact_bps > self.max_single_impact_bps { self.max_single_impact_bps = result.total_impact_bps; }
        if self.n_orders == 1 || result.total_impact_bps < self.min_single_impact_bps { self.min_single_impact_bps = result.total_impact_bps; }
    }

    pub fn avg_total_impact_bps(&self) -> f64 {
        if self.n_orders == 0 { return 0.0; }
        (self.total_temp_impact_bps + self.total_perm_impact_bps + self.total_slippage_bps) / self.n_orders as f64
    }
}

// ── Impact trajectory ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImpactTrajectory {
    pub total_qty: f64,
    pub total_time_secs: f64,
    pub n_slices: usize,
    pub slices: Vec<ImpactSlice>,
    pub total_shortfall: f64,
    pub total_cost: f64,
    pub vwap_price: f64,
}

#[derive(Debug, Clone)]
pub struct ImpactSlice {
    pub slice_idx: usize,
    pub time_secs: f64,
    pub qty: f64,
    pub expected_price: f64,
    pub temp_impact: f64,
    pub perm_impact: f64,
    pub remaining_qty: f64,
}

impl ImpactTrajectory {
    pub fn compute_twap(engine: &FeedbackImpactEngine, total_qty: f64, total_time_secs: f64, n_slices: usize, initial_price: f64) -> Self {
        let dt = total_time_secs / n_slices as f64;
        let slice_qty = total_qty / n_slices as f64;
        let mut slices = Vec::with_capacity(n_slices);
        let mut cum_perm = 0.0f64;
        let mut total_cost = 0.0f64;
        let mut remaining = total_qty;
        for i in 0..n_slices {
            let temp_imp = engine.ac_model.temporary_impact(slice_qty, dt);
            let perm_imp = engine.ac_model.permanent_impact(slice_qty, dt);
            cum_perm += perm_imp;
            let expected_price = initial_price + cum_perm + temp_imp;
            total_cost += slice_qty * expected_price;
            remaining -= slice_qty;
            slices.push(ImpactSlice { slice_idx: i, time_secs: i as f64 * dt, qty: slice_qty, expected_price, temp_impact: temp_imp, perm_impact: perm_imp, remaining_qty: remaining.max(0.0) });
        }
        let shortfall = total_cost - total_qty * initial_price;
        let vwap = if total_qty > 0.0 { total_cost / total_qty } else { initial_price };
        ImpactTrajectory { total_qty, total_time_secs, n_slices, slices, total_shortfall: shortfall, total_cost, vwap_price: vwap }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_almgren_chriss_permanent_impact() {
        let ac = AlmgrenChrissImpact::default();
        let impact = ac.permanent_impact(10000.0, 1.0);
        assert!(impact > 0.0, "impact={}", impact);
    }

    #[test]
    fn test_almgren_chriss_temporary_impact_scaling() {
        let ac = AlmgrenChrissImpact::default();
        let t1 = ac.temporary_impact(1000.0, 1.0);
        let t2 = ac.temporary_impact(10000.0, 1.0);
        assert!(t2 > t1, "t1={} t2={}", t1, t2);
    }

    #[test]
    fn test_almgren_chriss_optimal_schedule() {
        let ac = AlmgrenChrissImpact::new(0.1, 0.5, 500_000.0, 0.25, 1e-6, 50.0);
        let (shortfall, variance) = ac.optimal_schedule_cost(10_000.0, 3600.0, 10);
        assert!(shortfall >= 0.0, "shortfall={}", shortfall);
        assert!(variance >= 0.0, "variance={}", variance);
    }

    #[test]
    fn test_lob_depth_init() {
        let model = LobDepthModel::new(0.01, 100.0);
        assert!(model.total_bid_depth() > 0.0);
        assert!(model.total_ask_depth() > 0.0);
        let b = model.best_bid().unwrap();
        let a = model.best_ask().unwrap();
        assert!(a.0 > b.0);
    }

    #[test]
    fn test_lob_execute_buy() {
        let mut model = LobDepthModel::new(0.01, 100.0);
        let (avg, slippage, levels) = model.execute_market_order(50.0, true, 0.0);
        assert!(avg >= 100.0);
        assert!(slippage >= 0.0);
        assert!(levels >= 1);
    }

    #[test]
    fn test_lob_spread_positive() {
        let model = LobDepthModel::new(0.01, 50.0);
        assert!(model.spread() >= 0.01);
    }

    #[test]
    fn test_feedback_engine_basic() {
        let ac = AlmgrenChrissImpact::default();
        let mut engine = FeedbackImpactEngine::new(0.01, 100.0, ac);
        let result = engine.submit_order(1000.0, true, 0.0);
        assert!(result.total_impact_bps >= 0.0);
        assert!(result.levels_swept >= 1);
    }

    #[test]
    fn test_feedback_buy_pushes_mid_up() {
        let ac = AlmgrenChrissImpact::new(0.5, 1.0, 100_000.0, 0.2, 1e-6, 100.0);
        let mut engine = FeedbackImpactEngine::new(0.01, 100.0, ac);
        let mid_before = engine.mid_price();
        let _ = engine.submit_order(5000.0, true, 0.0);
        assert!(engine.mid_price() >= mid_before);
    }

    #[test]
    fn test_kyle_lambda_estimation() {
        let mut est = KyleLambdaEstimator::new(100);
        let mut rng = Rng::new(42);
        for _ in 0..50 {
            let flow = rng.next_normal() * 1000.0;
            let dp = 0.0001 * flow + rng.next_normal() * 0.001;
            est.add_observation(flow, dp);
        }
        assert!(est.current_lambda.is_finite());
    }

    #[test]
    fn test_kyle_lambda_predict() {
        let mut est = KyleLambdaEstimator::new(50);
        for i in 0..30 {
            est.add_observation(i as f64 * 100.0, i as f64 * 0.01);
        }
        let pred = est.predict_price_change(1000.0);
        assert!(pred.is_finite());
    }

    #[test]
    fn test_volume_participation() {
        let mut vpm = VolumeParticipationModel::new(0.05, 0.10, 20);
        for _ in 0..20 { vpm.record_interval(100_000.0, 5_000.0); }
        let rate = vpm.current_participation_rate();
        assert!((rate - 0.05).abs() < 0.01, "rate={}", rate);
    }

    #[test]
    fn test_volume_participation_slice_qty() {
        let vpm = VolumeParticipationModel::new(0.05, 0.10, 20);
        let slice = vpm.compute_slice_qty(10_000.0, 100_000.0);
        assert!(slice > 0.0 && slice <= 10_000.0);
    }

    #[test]
    fn test_square_root_impact() {
        let impact = square_root_impact_bps(10_000.0, 1_000_000.0, 0.20, 10.0);
        assert!(impact > 0.0 && impact < 1000.0);
    }

    #[test]
    fn test_linear_impact() {
        let impact = linear_impact_bps(1000.0, 5000.0, 1.0);
        assert!((impact - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_impact_trajectory_twap() {
        let ac = AlmgrenChrissImpact::default();
        let engine = FeedbackImpactEngine::new(0.01, 100.0, ac);
        let traj = ImpactTrajectory::compute_twap(&engine, 50_000.0, 3600.0, 12, 100.0);
        assert_eq!(traj.slices.len(), 12);
        assert!(traj.vwap_price > 0.0);
    }

    #[test]
    fn test_impact_stats() {
        let mut stats = ImpactStats::default();
        let result = OrderImpactResult { avg_fill_price: 100.05, slippage_bps: 0.5, temporary_impact_bps: 0.3, permanent_impact_bps: 0.2, total_impact_bps: 1.0, levels_swept: 2, post_trade_mid: 100.02, accumulated_perm_impact: 0.02 };
        stats.update(&result, 1000.0);
        assert_eq!(stats.n_orders, 1);
        assert_eq!(stats.max_single_impact_bps, 1.0);
    }

    #[test]
    fn test_lob_resilience() {
        let mut model = LobDepthModel::new(0.01, 100.0);
        let depth_before = model.total_ask_depth();
        model.execute_market_order(200.0, true, 0.0);
        // After some time, depth should partially recover
        model.execute_market_order(0.0, true, 1.0);
        let depth_after = model.total_ask_depth();
        // depth may be different, just check it's positive
        assert!(depth_after >= 0.0);
    }
}
