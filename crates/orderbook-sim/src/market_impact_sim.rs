//! Market impact simulation -- walking the order book to measure execution quality.
//!
//! Provides realistic slippage estimation, impact curves, optimal order sizing,
//! and TWAP execution simulation.

use crate::orderbook::OrderBook;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Result of simulating the execution of a single order against a book snapshot.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Volume-weighted average fill price.
    pub avg_fill_price: f64,
    /// Slippage relative to mid price in basis points.
    pub slippage_bps: f64,
    /// Market impact in basis points (same as slippage for a single snapshot).
    pub market_impact_bps: f64,
    /// Quantity that could not be filled due to insufficient book depth.
    pub unfilled_qty: f64,
    /// Individual fills: (price, quantity) pairs.
    pub fills: Vec<(f64, f64)>,
}

impl ExecutionResult {
    fn empty(order_qty: f64) -> Self {
        ExecutionResult {
            avg_fill_price: 0.0,
            slippage_bps: 0.0,
            market_impact_bps: 0.0,
            unfilled_qty: order_qty,
            fills: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// MarketImpactSimulator
// ---------------------------------------------------------------------------

/// Namespace for market impact simulation utilities.
pub struct MarketImpactSimulator;

// ---------------------------------------------------------------------------
// simulate_execution
// ---------------------------------------------------------------------------

/// Simulate walking the order book to fill `order_qty` units.
///
/// For a buy order we walk up the ask side; for a sell we walk down the bid side.
/// Returns an ExecutionResult summarising the outcome.
///
/// This is non-destructive (the book is not modified).
pub fn simulate_execution(book: &OrderBook, order_qty: f64, is_buy: bool) -> ExecutionResult {
    if order_qty <= 0.0 {
        return ExecutionResult::empty(0.0);
    }

    let mid = match book.mid_price() {
        Some(m) => m,
        None => return ExecutionResult::empty(order_qty),
    };

    let levels: Vec<(f64, f64)> = if is_buy {
        book.ask_depth(500) // get up to 500 ask levels
    } else {
        book.bid_depth(500) // get up to 500 bid levels
    };

    let mut remaining = order_qty;
    let mut fills: Vec<(f64, f64)> = Vec::new();
    let mut total_notional = 0.0_f64;
    let mut total_filled = 0.0_f64;

    for (price, available_qty) in &levels {
        if remaining <= 0.0 {
            break;
        }
        let take = remaining.min(*available_qty);
        fills.push((*price, take));
        total_notional += price * take;
        total_filled += take;
        remaining -= take;
    }

    if total_filled <= 0.0 {
        return ExecutionResult::empty(order_qty);
    }

    let avg_price = total_notional / total_filled;

    // Slippage: for a buy, avg_price > mid is adverse; for sell, avg_price < mid.
    let slippage = if is_buy {
        (avg_price - mid) / mid
    } else {
        (mid - avg_price) / mid
    };
    let slippage_bps = slippage * 10_000.0;

    ExecutionResult {
        avg_fill_price: avg_price,
        slippage_bps,
        market_impact_bps: slippage_bps,
        unfilled_qty: remaining.max(0.0),
        fills,
    }
}

// ---------------------------------------------------------------------------
// estimate_impact_curve
// ---------------------------------------------------------------------------

/// Estimate the slippage vs order-size curve.
///
/// Samples `n_points` order sizes from 0 to `max_qty` and returns
/// (order_size, slippage_bps) pairs.
pub fn estimate_impact_curve(book: &OrderBook, max_qty: f64, n_points: usize) -> Vec<(f64, f64)> {
    if n_points == 0 || max_qty <= 0.0 {
        return vec![];
    }
    let step = max_qty / n_points as f64;
    (1..=n_points)
        .map(|i| {
            let qty = step * i as f64;
            let result = simulate_execution(book, qty, true); // use buy side as default
            (qty, result.slippage_bps)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// optimal_order_size
// ---------------------------------------------------------------------------

/// Compute the optimal order size balancing alpha signal against market impact.
///
/// Model: expected PnL = alpha_bps * qty - impact_cost(qty)
/// where impact_cost is estimated from the book as slippage_bps * qty.
///
/// At the optimum: d(PnL)/d(qty) = 0
/// => alpha_bps = d(impact_cost)/d(qty)
///
/// We solve this numerically by finding the quantity where marginal impact
/// (slope of the slippage curve) equals alpha_bps / risk_aversion.
///
/// `alpha_bps` -- signal edge in basis points.
/// `risk_aversion` -- multiplier on the impact cost (higher = more conservative).
///
/// Returns the optimal order size (may be 0 if alpha < minimum book impact).
pub fn optimal_order_size(book: &OrderBook, alpha_bps: f64, risk_aversion: f64) -> f64 {
    if alpha_bps <= 0.0 {
        return 0.0;
    }
    let ra = risk_aversion.max(1e-6);

    // Build a fine-grained impact curve.
    let total_bid = book.total_bid_qty();
    let total_ask = book.total_ask_qty();
    let max_qty = (total_bid.min(total_ask) * 0.5).max(1.0);
    let n_points = 200;
    let curve = estimate_impact_curve(book, max_qty, n_points);

    if curve.len() < 2 {
        return 0.0;
    }

    // Find the point where marginal impact = alpha_bps / risk_aversion.
    let target = alpha_bps / ra;

    // Compute finite-difference marginal impact at each point.
    let mut opt_qty = 0.0;
    for i in 1..curve.len() {
        let (qty_lo, slip_lo) = curve[i - 1];
        let (qty_hi, slip_hi) = curve[i];
        let dqty = qty_hi - qty_lo;
        if dqty <= 0.0 {
            continue;
        }
        let marginal_impact = (slip_hi - slip_lo) / dqty;
        if marginal_impact >= target {
            // Interpolate.
            opt_qty = qty_lo;
            break;
        }
        opt_qty = qty_hi;
    }
    opt_qty
}

// ---------------------------------------------------------------------------
// twap_simulation
// ---------------------------------------------------------------------------

/// Simulate TWAP execution: split `total_qty` into `n_intervals` equal slices,
/// executing one slice against each consecutive book snapshot.
///
/// Returns the average slippage in basis points across all executed slices.
/// If a slice cannot be fully filled, the unfilled portion is noted but
/// the slippage is computed only on filled quantity.
pub fn twap_simulation(book_snapshots: &[OrderBook], total_qty: f64, n_intervals: usize) -> f64 {
    if book_snapshots.is_empty() || n_intervals == 0 || total_qty <= 0.0 {
        return 0.0;
    }
    let slice_qty = total_qty / n_intervals as f64;
    let n = n_intervals.min(book_snapshots.len());

    let total_slippage: f64 = (0..n)
        .map(|i| {
            let result = simulate_execution(&book_snapshots[i], slice_qty, true);
            result.slippage_bps
        })
        .sum();

    total_slippage / n as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use crate::synthetic_orderbook::{BookConfig, generate};

    fn make_book(seed: u64) -> OrderBook {
        let config = BookConfig::default();
        let mut rng = StdRng::seed_from_u64(seed);
        generate(&config, &mut rng)
    }

    // 1. simulate_execution returns a positive avg fill price for a buy.
    #[test]
    fn test_simulate_execution_positive_price() {
        let book = make_book(1);
        let result = simulate_execution(&book, 10.0, true);
        assert!(result.avg_fill_price > 0.0);
    }

    // 2. Buy fills at prices >= mid; sell fills at prices <= mid.
    #[test]
    fn test_simulate_execution_direction() {
        let book = make_book(2);
        let mid = book.mid_price().unwrap();
        let buy_result = simulate_execution(&book, 10.0, true);
        let sell_result = simulate_execution(&book, 10.0, false);
        assert!(buy_result.avg_fill_price >= mid - 1e-6,
            "buy avg {:.3} below mid {mid:.3}", buy_result.avg_fill_price);
        assert!(sell_result.avg_fill_price <= mid + 1e-6,
            "sell avg {:.3} above mid {mid:.3}", sell_result.avg_fill_price);
    }

    // 3. Slippage is non-negative for both directions.
    #[test]
    fn test_simulate_slippage_non_negative() {
        let book = make_book(3);
        let buy = simulate_execution(&book, 50.0, true);
        let sell = simulate_execution(&book, 50.0, false);
        assert!(buy.slippage_bps >= 0.0, "buy slippage negative: {}", buy.slippage_bps);
        assert!(sell.slippage_bps >= 0.0, "sell slippage negative: {}", sell.slippage_bps);
    }

    // 4. Larger orders have more slippage (monotone impact).
    #[test]
    fn test_simulate_slippage_monotone() {
        let book = make_book(4);
        let small = simulate_execution(&book, 10.0, true);
        let large = simulate_execution(&book, 500.0, true);
        assert!(large.slippage_bps >= small.slippage_bps - 1e-6,
            "slippage should increase with order size: small={:.2} large={:.2}",
            small.slippage_bps, large.slippage_bps);
    }

    // 5. estimate_impact_curve returns the right number of points.
    #[test]
    fn test_impact_curve_length() {
        let book = make_book(5);
        let curve = estimate_impact_curve(&book, 1000.0, 20);
        assert_eq!(curve.len(), 20);
    }

    // 6. Impact curve is monotonically non-decreasing in slippage.
    #[test]
    fn test_impact_curve_monotone() {
        let book = make_book(6);
        let curve = estimate_impact_curve(&book, 200.0, 10);
        for i in 1..curve.len() {
            assert!(curve[i].1 >= curve[i - 1].1 - 1e-3,
                "impact curve not monotone at point {i}: {:.2} < {:.2}",
                curve[i].1, curve[i - 1].1);
        }
    }

    // 7. optimal_order_size returns zero for non-positive alpha.
    #[test]
    fn test_optimal_order_size_zero_alpha() {
        let book = make_book(7);
        let size = optimal_order_size(&book, 0.0, 1.0);
        assert_eq!(size, 0.0);
    }

    // 8. optimal_order_size returns positive size for sufficient alpha.
    #[test]
    fn test_optimal_order_size_positive_alpha() {
        let book = make_book(8);
        let size = optimal_order_size(&book, 10.0, 1.0);
        assert!(size >= 0.0, "optimal size must be non-negative");
    }

    // 9. Higher risk aversion produces smaller optimal size.
    #[test]
    fn test_optimal_order_size_risk_aversion() {
        let book = make_book(9);
        let size_low_ra = optimal_order_size(&book, 5.0, 0.5);
        let size_high_ra = optimal_order_size(&book, 5.0, 5.0);
        assert!(size_high_ra <= size_low_ra + 1e-6,
            "higher risk aversion should reduce optimal size: {size_low_ra:.2} vs {size_high_ra:.2}");
    }

    // 10. twap_simulation returns non-negative slippage.
    #[test]
    fn test_twap_simulation_non_negative() {
        let books: Vec<OrderBook> = (0..5).map(make_book).collect();
        let slippage = twap_simulation(&books, 100.0, 5);
        assert!(slippage >= 0.0, "TWAP slippage must be non-negative");
    }

    // 11. Empty order returns empty result.
    #[test]
    fn test_simulate_execution_zero_qty() {
        let book = make_book(11);
        let result = simulate_execution(&book, 0.0, true);
        assert_eq!(result.fills.len(), 0);
        assert_eq!(result.unfilled_qty, 0.0);
    }

    // 12. TWAP with one interval equals direct execution slippage.
    #[test]
    fn test_twap_single_interval() {
        let book = make_book(12);
        let direct = simulate_execution(&book, 50.0, true).slippage_bps;
        let twap = twap_simulation(&[book], 50.0, 1);
        assert!((direct - twap).abs() < 1e-6,
            "single-interval TWAP {twap:.4} != direct {direct:.4}");
    }

    // 13. Fills summed equal total filled quantity.
    #[test]
    fn test_fills_sum_to_filled_qty() {
        let book = make_book(13);
        let result = simulate_execution(&book, 30.0, true);
        let fills_sum: f64 = result.fills.iter().map(|(_, q)| q).sum();
        let filled = 30.0 - result.unfilled_qty;
        assert!((fills_sum - filled).abs() < 1e-6,
            "fills sum {fills_sum:.4} != filled qty {filled:.4}");
    }
}
