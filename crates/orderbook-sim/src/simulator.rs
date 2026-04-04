/// Full market simulator with Hawkes-process order flow generation.

use chrono::{DateTime, Duration, Utc};
use rand::prelude::*;

use crate::fills::Fill;
use crate::order::{Order, OrderSide, OrderType};
use crate::orderbook::OrderBook;

// ── Hawkes Process ────────────────────────────────────────────────────────────

/// Parameters for a univariate Hawkes process.
#[derive(Debug, Clone, Copy)]
pub struct HawkesParams {
    /// Baseline (exogenous) intensity λ₀.
    pub baseline_intensity: f64,
    /// Decay rate β for the self-exciting kernel.
    pub decay: f64,
    /// Jump size α for each arriving event.
    pub jump_size: f64,
}

impl Default for HawkesParams {
    fn default() -> Self {
        HawkesParams {
            baseline_intensity: 5.0,
            decay: 10.0,
            jump_size: 3.0,
        }
    }
}

/// Simulate a univariate Hawkes process using Ogata's thinning algorithm.
/// Returns a sorted vector of event arrival times (in seconds from 0).
pub fn simulate_hawkes(n_events: usize, params: &HawkesParams, rng: &mut impl Rng) -> Vec<f64> {
    let mut times = Vec::with_capacity(n_events);
    let mut t = 0.0_f64;
    let mut lambda_star = params.baseline_intensity;

    while times.len() < n_events {
        // Sample next candidate event.
        let dt = -lambda_star.recip() * rng.gen::<f64>().ln();
        t += dt;

        // Update intensity at candidate time.
        let lambda_t = params.baseline_intensity
            + times
                .iter()
                .map(|&ti: &f64| params.jump_size * (-params.decay * (t - ti)).exp())
                .sum::<f64>();

        // Accept with probability λ(t) / λ*.
        let u: f64 = rng.gen();
        if u <= lambda_t / lambda_star {
            times.push(t);
        }
        // Upper bound for next interval.
        lambda_star = lambda_t + params.jump_size;
    }
    times
}

// ── Synthetic Order Flow ──────────────────────────────────────────────────────

/// Parameters for synthetic order-flow generation.
#[derive(Debug, Clone)]
pub struct OrderFlowParams {
    pub hawkes: HawkesParams,
    /// Mid price at simulation start.
    pub initial_mid: f64,
    /// Tick size.
    pub tick_size: f64,
    /// Fraction of orders that are market orders (rest are limits).
    pub market_order_fraction: f64,
    /// Max levels away from mid for limit orders.
    pub max_depth_levels: u32,
    /// Spread (in ticks) at which limit orders cluster.
    pub spread_ticks: u32,
    /// Average order size.
    pub avg_qty: f64,
    /// Std dev of order size.
    pub qty_std: f64,
}

impl Default for OrderFlowParams {
    fn default() -> Self {
        OrderFlowParams {
            hawkes: HawkesParams::default(),
            initial_mid: 100.0,
            tick_size: 0.01,
            market_order_fraction: 0.3,
            max_depth_levels: 5,
            spread_ticks: 2,
            avg_qty: 100.0,
            qty_std: 50.0,
        }
    }
}

/// Generate synthetic orders using a Hawkes process for inter-arrival times.
pub fn generate_order_flow(
    n_events: usize,
    params: &OrderFlowParams,
    start_time: DateTime<Utc>,
    rng: &mut impl Rng,
) -> Vec<Order> {
    let arrival_times = simulate_hawkes(n_events, &params.hawkes, rng);
    let mut orders = Vec::with_capacity(n_events);

    for (seq, &t_secs) in arrival_times.iter().enumerate() {
        let ts = start_time + Duration::milliseconds((t_secs * 1000.0) as i64);
        let is_buy: bool = rng.gen_bool(0.5);
        let side = if is_buy { OrderSide::Buy } else { OrderSide::Sell };
        let is_market = rng.gen_bool(params.market_order_fraction);

        // Qty ~ max(1, Normal(avg_qty, qty_std)).
        let qty = (params.avg_qty + params.qty_std * rng.gen::<f64>() * 2.0 - params.qty_std)
            .max(1.0);

        let order = if is_market {
            Order::new_market(seq as u64 + 1, side, qty, ts, seq as u64)
        } else {
            // Place limit at spread/2 + random depth offset.
            let half_spread = params.spread_ticks as f64 / 2.0 * params.tick_size;
            let depth_offset =
                rng.gen_range(0..=params.max_depth_levels) as f64 * params.tick_size;
            let price = match side {
                OrderSide::Buy => {
                    (params.initial_mid - half_spread - depth_offset) / params.tick_size
                }
                OrderSide::Sell => {
                    (params.initial_mid + half_spread + depth_offset) / params.tick_size
                }
            };
            let price = (price.round() * params.tick_size * 100.0).round() / 100.0;
            Order::new_limit(seq as u64 + 1, side, price, qty, ts, seq as u64)
        };
        orders.push(order);
    }
    orders
}

// ── Simulation Result ─────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct SimResult {
    pub equity_curve: Vec<f64>,
    pub fills: Vec<Fill>,
    pub spread_series: Vec<f64>,
    pub mid_price_series: Vec<f64>,
    pub slippage_total: f64,
}

// ── Full Simulator ────────────────────────────────────────────────────────────

/// Run a market simulation.
///
/// `strategy_fn` receives the current order book snapshot (mid, spread, imbalance)
/// and returns an optional order to submit.
pub fn run_simulation<F>(
    strategy_fn: F,
    duration_secs: u64,
    tick_size: f64,
    initial_cash: f64,
    initial_mid: f64,
    seed: u64,
) -> SimResult
where
    F: Fn(f64, f64, f64, u64) -> Option<(OrderSide, f64, f64, OrderType)>,
    // (mid, spread, imbalance, step) → Option<(side, qty, limit_price, order_type)>
{
    let mut rng = StdRng::seed_from_u64(seed);
    let start_time = Utc::now();

    let flow_params = OrderFlowParams {
        initial_mid,
        tick_size,
        ..OrderFlowParams::default()
    };

    // Generate background order flow for the simulation horizon.
    let n_background = (duration_secs as f64 * flow_params.hawkes.baseline_intensity * 2.0) as usize;
    let background_orders = generate_order_flow(n_background, &flow_params, start_time, &mut rng);

    let mut book = OrderBook::new();
    let mut cash = initial_cash;
    let mut position = 0.0_f64;
    let mut equity_curve = Vec::new();
    let mut spread_series = Vec::new();
    let mut mid_price_series = Vec::new();
    let mut strategy_fills: Vec<Fill> = Vec::new();

    // Sort background orders by timestamp.
    let mut all_orders = background_orders;
    all_orders.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    // Seed the book with some resting limit orders around initial_mid.
    let mut seed_id = 10_000_000u64;
    for tick in 1..=5u32 {
        let bid_price = initial_mid - tick as f64 * tick_size;
        let ask_price = initial_mid + tick as f64 * tick_size;
        let bid = Order::new_limit(seed_id, OrderSide::Buy, bid_price, 200.0, start_time, seed_id);
        seed_id += 1;
        let ask = Order::new_limit(seed_id, OrderSide::Sell, ask_price, 200.0, start_time, seed_id);
        seed_id += 1;
        book.add_order(bid);
        book.add_order(ask);
    }

    let step_duration_secs = 1.0_f64;
    let total_steps = duration_secs;
    let mut order_iter = all_orders.iter().peekable();

    for step in 0..total_steps {
        let step_end = start_time + Duration::seconds(step as i64 + 1);

        // Process all background orders up to this step.
        while let Some(order) = order_iter.peek() {
            if order.timestamp >= step_end {
                break;
            }
            let o = order_iter.next().unwrap().clone();
            book.add_order(o);
        }

        let mid = book.mid_price().unwrap_or(initial_mid);
        let spread = book.spread().unwrap_or(tick_size * 2.0);
        let imb = book.imbalance(5);

        // Replenish book if thin.
        if book.bid_level_count() < 3 {
            for tick in 1..=3u32 {
                let p = mid - tick as f64 * tick_size;
                let o = Order::new_limit(seed_id, OrderSide::Buy, p, 100.0, step_end, seed_id);
                seed_id += 1;
                book.add_order(o);
            }
        }
        if book.ask_level_count() < 3 {
            for tick in 1..=3u32 {
                let p = mid + tick as f64 * tick_size;
                let o = Order::new_limit(seed_id, OrderSide::Sell, p, 100.0, step_end, seed_id);
                seed_id += 1;
                book.add_order(o);
            }
        }

        // Strategy decision.
        if let Some((side, qty, limit_price, order_type)) =
            strategy_fn(mid, spread, imb, step)
        {
            let mut strat_order = match order_type {
                OrderType::Market => {
                    Order::new_market(seed_id, side, qty, step_end, seed_id)
                }
                _ => {
                    Order::new_limit(seed_id, side, limit_price, qty, step_end, seed_id)
                }
            };
            seed_id += 1;
            let fills = book.add_order(strat_order.clone());
            for fill in &fills {
                // Update cash and position.
                match side {
                    OrderSide::Buy => {
                        cash -= fill.price * fill.qty;
                        position += fill.qty;
                    }
                    OrderSide::Sell => {
                        cash += fill.price * fill.qty;
                        position -= fill.qty;
                    }
                }
                strategy_fills.push(fill.clone());
            }
        }

        let equity = cash + position * mid;
        equity_curve.push(equity);
        spread_series.push(spread);
        mid_price_series.push(mid);
    }

    // Compute total slippage for strategy fills.
    let slippage_total = if !strategy_fills.is_empty() {
        let avg_fill = strategy_fills.iter().map(|f| f.price * f.qty).sum::<f64>()
            / strategy_fills.iter().map(|f| f.qty).sum::<f64>();
        let avg_mid = mid_price_series.iter().sum::<f64>() / mid_price_series.len() as f64;
        (avg_fill - avg_mid).abs()
    } else {
        0.0
    };

    SimResult {
        equity_curve,
        fills: strategy_fills,
        spread_series,
        mid_price_series,
        slippage_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hawkes_produces_n_events() {
        let mut rng = StdRng::seed_from_u64(42);
        let params = HawkesParams::default();
        let times = simulate_hawkes(100, &params, &mut rng);
        assert_eq!(times.len(), 100);
        // Times should be increasing.
        for i in 1..times.len() {
            assert!(times[i] > times[i - 1]);
        }
    }

    #[test]
    fn generate_order_flow_count() {
        let mut rng = StdRng::seed_from_u64(1);
        let params = OrderFlowParams::default();
        let orders = generate_order_flow(50, &params, Utc::now(), &mut rng);
        assert_eq!(orders.len(), 50);
    }
}
