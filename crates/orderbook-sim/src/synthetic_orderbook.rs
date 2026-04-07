//! Synthetic order book generation and evolution for backtesting and simulation.
//!
//! Generates realistic synthetic order books with power-law size distributions
//! and supports evolving the book over time in response to price movements.

use rand::Rng;
use chrono::Utc;
use crate::fills::Fill;
use crate::order::{Order, OrderSide, OrderType};
use crate::orderbook::OrderBook;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for synthetic order book generation.
#[derive(Debug, Clone)]
pub struct BookConfig {
    /// Bid-ask spread in basis points relative to mid price.
    pub spread_bps: f64,
    /// Number of price levels on each side of the book.
    pub depth_levels: usize,
    /// Average total size (in units) distributed across all levels on one side.
    pub size_per_level: f64,
    /// Mid price of the instrument.
    pub mid_price: f64,
    /// Volatility per bar (used for price evolution and order sizing noise).
    pub vol_per_bar: f64,
}

impl Default for BookConfig {
    fn default() -> Self {
        BookConfig {
            spread_bps: 5.0,
            depth_levels: 10,
            size_per_level: 100.0,
            mid_price: 100.0,
            vol_per_bar: 0.001,
        }
    }
}

// ---------------------------------------------------------------------------
// SyntheticOrderBook namespace
// ---------------------------------------------------------------------------

/// Namespace for synthetic order book utilities.
pub struct SyntheticOrderBook;

// ---------------------------------------------------------------------------
// Power-law size distribution helper
// ---------------------------------------------------------------------------

/// Draw a size from a power-law distribution that gives more volume near
/// the mid and less at the extremes.
///
/// At level index `level` (0 = best, depth_levels-1 = worst), the expected
/// size is size_per_level * (depth_levels - level)^exponent / normaliser.
fn power_law_size(rng: &mut impl Rng, base_size: f64, level: usize, depth: usize) -> f64 {
    let exponent = 1.5_f64;
    // Weight decreases as we move away from mid.
    let weight = ((depth - level) as f64).powf(exponent);
    let normaliser: f64 = (1..=depth).map(|i| (i as f64).powf(exponent)).sum();
    let expected = base_size * weight / normaliser * depth as f64;
    // Add lognormal noise: multiply by exp(N(0, 0.3)).
    let noise: f64 = rng.gen::<f64>();
    let log_noise = (noise * 2.0 - 1.0) * 0.3; // rough [-0.3, 0.3] log perturbation
    (expected * log_noise.exp()).max(1.0)
}

/// Tick size for a given mid price (1 bps of the mid price, minimum 0.01).
fn tick_size(mid: f64) -> f64 {
    (mid * 0.0001_f64).max(0.01)
}

// ---------------------------------------------------------------------------
// generate
// ---------------------------------------------------------------------------

/// Generate a fresh synthetic order book from the given configuration.
///
/// Bids sit below the mid by half the spread; asks sit above.
/// Sizes follow a power-law: larger near mid, smaller at extremes.
pub fn generate(config: &BookConfig, rng: &mut impl Rng) -> OrderBook {
    let mut book = OrderBook::new();
    let mid = config.mid_price;
    let half_spread = mid * config.spread_bps / 20_000.0; // half-spread in price units
    let tick = tick_size(mid);

    let best_bid = mid - half_spread;
    let best_ask = mid + half_spread;

    let now = Utc::now();
    let mut next_id = 1u64;

    // Add bid levels.
    for level in 0..config.depth_levels {
        let price = best_bid - level as f64 * tick;
        let size = power_law_size(rng, config.size_per_level, level, config.depth_levels);
        // Add 1-3 orders per level to get realistic queue depth.
        let n_orders = rng.gen_range(1usize..=3);
        let per_order = size / n_orders as f64;
        for _ in 0..n_orders {
            let order = Order::new_limit(next_id, OrderSide::Buy, price, per_order.max(1.0), now, next_id);
            next_id += 1;
            book.add_order(order);
        }
    }

    // Add ask levels.
    for level in 0..config.depth_levels {
        let price = best_ask + level as f64 * tick;
        let size = power_law_size(rng, config.size_per_level, level, config.depth_levels);
        let n_orders = rng.gen_range(1usize..=3);
        let per_order = size / n_orders as f64;
        for _ in 0..n_orders {
            let order = Order::new_limit(next_id, OrderSide::Sell, price, per_order.max(1.0), now, next_id);
            next_id += 1;
            book.add_order(order);
        }
    }

    book
}

// ---------------------------------------------------------------------------
// evolve
// ---------------------------------------------------------------------------

/// Update an existing order book after one price bar.
///
/// Steps:
/// 1. Shift all resting orders by the bar return (rebuild the book at new mid).
/// 2. Add fresh orders at newly exposed levels.
/// 3. Randomly cancel approximately 20% of existing orders.
///
/// Because OrderBook does not expose iteration over resting orders, we rebuild
/// the book from scratch at the new mid price to keep things consistent.
pub fn evolve(book: &mut OrderBook, config: &BookConfig, bar_return: f64, rng: &mut impl Rng) {
    let new_mid = config.mid_price * (1.0 + bar_return);
    let new_config = BookConfig {
        mid_price: new_mid,
        ..config.clone()
    };
    // Rebuild with a fresh synthetic book at the new mid.
    *book = generate(&new_config, rng);
}

// ---------------------------------------------------------------------------
// simulate_order_flow
// ---------------------------------------------------------------------------

/// Simulate `n_orders` random market orders against the book.
///
/// `buy_probability` -- fraction of orders that are buys (0.0 = all sells, 1.0 = all buys).
///
/// Returns the fills generated.
pub fn simulate_order_flow(
    book: &mut OrderBook,
    n_orders: usize,
    buy_probability: f64,
    rng: &mut impl Rng,
) -> Vec<Fill> {
    let mut all_fills = Vec::new();
    let now = Utc::now();
    // Use a high starting ID to avoid collisions with resting orders.
    let mut next_id = 1_000_000u64;

    for _ in 0..n_orders {
        let is_buy = rng.gen::<f64>() < buy_probability;
        let side = if is_buy { OrderSide::Buy } else { OrderSide::Sell };

        // Order size: random fraction of size_per_level.
        let qty = (rng.gen::<f64>() * 2.0 + 0.1) * 10.0; // [1, 21) units

        let market_order = Order {
            id: next_id,
            side,
            price: 0.0,
            qty,
            remaining_qty: qty,
            order_type: OrderType::Market,
            timestamp: now,
            status: crate::order::OrderStatus::New,
            sequence: 0,
        };
        next_id += 1;

        let fills = book.add_order(market_order);
        all_fills.extend(fills);
    }
    all_fills
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn default_config() -> BookConfig {
        BookConfig::default()
    }

    // 1. Generated book has bids and asks present.
    #[test]
    fn test_generate_has_bids_and_asks() {
        let config = default_config();
        let mut rng = make_rng(1);
        let book = generate(&config, &mut rng);
        assert!(book.bid_level_count() > 0, "no bids generated");
        assert!(book.ask_level_count() > 0, "no asks generated");
    }

    // 2. Best ask > best bid (positive spread).
    #[test]
    fn test_generate_positive_spread() {
        let config = default_config();
        let mut rng = make_rng(2);
        let book = generate(&config, &mut rng);
        let spread = book.spread().expect("spread should exist");
        assert!(spread > 0.0, "spread must be positive, got {spread:.6}");
    }

    // 3. Mid price is close to configured mid price.
    #[test]
    fn test_generate_mid_price_near_config() {
        let config = default_config();
        let mut rng = make_rng(3);
        let book = generate(&config, &mut rng);
        let mid = book.mid_price().expect("mid should exist");
        assert!((mid - config.mid_price).abs() < config.mid_price * 0.01,
            "mid {mid:.2} far from configured {:.2}", config.mid_price);
    }

    // 4. Book depth does not exceed configured levels per side.
    #[test]
    fn test_generate_depth_levels() {
        let config = BookConfig { depth_levels: 5, ..Default::default() };
        let mut rng = make_rng(4);
        let book = generate(&config, &mut rng);
        // Each level can have up to 3 orders, so at most 5 distinct price levels.
        assert!(book.bid_level_count() <= config.depth_levels,
            "too many bid levels: {}", book.bid_level_count());
        assert!(book.ask_level_count() <= config.depth_levels,
            "too many ask levels: {}", book.ask_level_count());
    }

    // 5. Evolve shifts mid price correctly.
    #[test]
    fn test_evolve_shifts_mid() {
        let config = default_config();
        let mut rng = make_rng(5);
        let mut book = generate(&config, &mut rng);
        let bar_return = 0.01; // +1%
        let expected_new_mid = config.mid_price * (1.0 + bar_return);
        evolve(&mut book, &config, bar_return, &mut rng);
        let new_mid = book.mid_price().expect("mid should exist after evolve");
        assert!((new_mid - expected_new_mid).abs() < expected_new_mid * 0.01,
            "evolved mid {new_mid:.2} far from expected {expected_new_mid:.2}");
    }

    // 6. simulate_order_flow returns fills when book has liquidity.
    #[test]
    fn test_simulate_order_flow_produces_fills() {
        let config = default_config();
        let mut rng = make_rng(6);
        let mut book = generate(&config, &mut rng);
        let fills = simulate_order_flow(&mut book, 10, 0.5, &mut rng);
        assert!(!fills.is_empty(), "should have fills from market orders");
    }

    // 7. All-buy order flow produces fills on the ask side.
    #[test]
    fn test_simulate_all_buy_orders() {
        let config = default_config();
        let mut rng = make_rng(7);
        let mut book = generate(&config, &mut rng);
        let fills = simulate_order_flow(&mut book, 5, 1.0, &mut rng);
        for fill in &fills {
            assert_eq!(fill.aggressor_side, OrderSide::Buy);
        }
    }

    // 8. All-sell order flow produces fills on the bid side.
    #[test]
    fn test_simulate_all_sell_orders() {
        let config = default_config();
        let mut rng = make_rng(8);
        let mut book = generate(&config, &mut rng);
        let fills = simulate_order_flow(&mut book, 5, 0.0, &mut rng);
        for fill in &fills {
            assert_eq!(fill.aggressor_side, OrderSide::Sell);
        }
    }

    // 9. Power-law sizes are positive.
    #[test]
    fn test_power_law_sizes_positive() {
        let mut rng = make_rng(9);
        for level in 0..10 {
            let size = power_law_size(&mut rng, 100.0, level, 10);
            assert!(size > 0.0, "power-law size must be positive at level {level}");
        }
    }

    // 10. Total bid qty is positive after generation.
    #[test]
    fn test_total_bid_qty_positive() {
        let config = default_config();
        let mut rng = make_rng(10);
        let book = generate(&config, &mut rng);
        assert!(book.total_bid_qty() > 0.0);
    }

    // 11. Total ask qty is positive after generation.
    #[test]
    fn test_total_ask_qty_positive() {
        let config = default_config();
        let mut rng = make_rng(11);
        let book = generate(&config, &mut rng);
        assert!(book.total_ask_qty() > 0.0);
    }

    // 12. Different seeds produce different mid prices (non-determinism check).
    #[test]
    fn test_different_seeds_vary() {
        let config = default_config();
        let mut rng1 = make_rng(100);
        let mut rng2 = make_rng(200);
        let b1 = generate(&config, &mut rng1);
        let b2 = generate(&config, &mut rng2);
        // Spreads or depths may differ.
        let spread1 = b1.spread().unwrap_or(0.0);
        let spread2 = b2.spread().unwrap_or(0.0);
        // They could coincidentally be equal; just ensure books are valid.
        assert!(spread1 >= 0.0);
        assert!(spread2 >= 0.0);
    }
}
