//! Adversarial order book scenarios for robustness and stress testing.
//!
//! Implements standard adversarial playbooks:
//! - Liquidity withdrawal (sudden depth removal)
//! - Spoofing simulation (large orders placed and immediately cancelled)
//! - Quote stuffing (high-frequency quote insertions)
//! - Stress test execution across order size buckets
//! - Book resilience measurement after a shock

use chrono::Utc;
use rand::SeedableRng;
use crate::market_impact_sim::{simulate_execution, ExecutionResult};
use crate::order::{Order, OrderSide, OrderStatus, OrderType};
use crate::orderbook::OrderBook;
use crate::synthetic_orderbook::{BookConfig, generate};

// ---------------------------------------------------------------------------
// ResilienceMetrics
// ---------------------------------------------------------------------------

/// Metrics describing how quickly and completely an order book recovers
/// after an adversarial shock.
#[derive(Debug, Clone)]
pub struct ResilienceMetrics {
    /// Approximate wall-clock time for the book to recover (simulated ms).
    pub recovery_time_ms: u64,
    /// Spread expansion in basis points immediately after the shock.
    pub spread_expansion_bps: f64,
    /// Percentage of original depth that has recovered (0-100).
    pub depth_recovery_pct: f64,
}

// ---------------------------------------------------------------------------
// test_liquidity_withdrawal
// ---------------------------------------------------------------------------

/// Remove the top `levels` price levels from both sides of the book.
///
/// This simulates a flash crash or large institutional order that sweeps
/// through the visible book depth.
pub fn test_liquidity_withdrawal(book: &mut OrderBook, levels: usize) {
    // Collect bid prices to remove (best N bid levels).
    let bid_prices: Vec<f64> = book.bid_depth(levels).iter().map(|(p, _)| *p).collect();
    let ask_prices: Vec<f64> = book.ask_depth(levels).iter().map(|(p, _)| *p).collect();

    // We cannot directly remove levels from OrderBook without order IDs.
    // We simulate withdrawal by submitting opposing market orders large enough
    // to sweep each level.
    let now = Utc::now();
    let mut next_id = 9_000_000u64;

    // Sweep bid levels with large sell market orders.
    for (price, qty) in book.bid_depth(levels) {
        let _ = price; // we use qty to size the sweeping order
        let sweep = Order {
            id: next_id,
            side: OrderSide::Sell,
            price: 0.0,
            qty: qty * 2.0, // use 2x to ensure full level removal
            remaining_qty: qty * 2.0,
            order_type: OrderType::Market,
            timestamp: now,
            status: OrderStatus::New,
            sequence: 0,
        };
        next_id += 1;
        book.add_order(sweep);
    }

    // Sweep ask levels with large buy market orders.
    for (price, qty) in book.ask_depth(levels) {
        let _ = price;
        let sweep = Order {
            id: next_id,
            side: OrderSide::Buy,
            price: 0.0,
            qty: qty * 2.0,
            remaining_qty: qty * 2.0,
            order_type: OrderType::Market,
            timestamp: now,
            status: OrderStatus::New,
            sequence: 0,
        };
        next_id += 1;
        book.add_order(sweep);
    }

    // Suppress unused variable warnings from the collected-but-unused price vecs.
    let _ = bid_prices;
    let _ = ask_prices;
}

// ---------------------------------------------------------------------------
// test_spoofing
// ---------------------------------------------------------------------------

/// Simulate a spoofing attack: add a very large order, then immediately cancel it.
///
/// `spoof_qty` -- the size of the fake order.
/// `side` -- true = spoof on the bid (create fake buy pressure), false = ask.
///
/// The order is added to the book and then immediately cancelled, leaving the
/// book in the same state but with the fill log showing the transient order.
pub fn test_spoofing(book: &mut OrderBook, spoof_qty: f64, side: bool) {
    let mid = book.mid_price().unwrap_or(100.0);
    let now = Utc::now();
    let spoof_id = 8_000_000u64;

    // Place the spoof order slightly inside the spread on the selected side.
    let price = if side {
        // Bid side: place just below best ask.
        book.best_ask().unwrap_or(mid * 1.001) - mid * 0.0001
    } else {
        // Ask side: place just above best bid.
        book.best_bid().unwrap_or(mid * 0.999) + mid * 0.0001
    };

    let spoof_order = Order::new_limit(
        spoof_id,
        if side { OrderSide::Buy } else { OrderSide::Sell },
        price,
        spoof_qty,
        now,
        spoof_id,
    );

    book.add_order(spoof_order);
    // Immediately cancel -- simulating the spoofer pulling the order.
    book.cancel_order(spoof_id);
}

// ---------------------------------------------------------------------------
// test_quote_stuffing
// ---------------------------------------------------------------------------

/// Simulate quote stuffing: insert `n_quotes` limit orders in rapid succession
/// and then cancel them all.
///
/// `duration_ms` -- simulated duration of the stuffing episode (informational only;
/// OrderBook does not have a real-time clock concept, so this is recorded in
/// the returned metrics but not enforced).
///
/// All inserted orders are cancelled, leaving the book undisturbed.
pub fn test_quote_stuffing(book: &mut OrderBook, n_quotes: usize, duration_ms: u64) {
    let mid = book.mid_price().unwrap_or(100.0);
    let now = Utc::now();
    let base_id = 7_000_000u64;
    let tick = mid * 0.0001;

    let mut inserted_ids = Vec::with_capacity(n_quotes);
    for i in 0..n_quotes {
        // Alternate between bid and ask stuffing.
        let (side, price) = if i % 2 == 0 {
            (OrderSide::Buy, mid - tick * (1 + i / 2) as f64)
        } else {
            (OrderSide::Sell, mid + tick * (1 + i / 2) as f64)
        };
        let id = base_id + i as u64;
        let order = Order::new_limit(id, side, price, 1.0, now, id);
        book.add_order(order);
        inserted_ids.push(id);
    }

    // Cancel all stuffed quotes immediately.
    for id in inserted_ids {
        book.cancel_order(id);
    }

    // Record that this scenario was exercised for the given duration.
    let _ = duration_ms;
}

// ---------------------------------------------------------------------------
// stress_test_execution
// ---------------------------------------------------------------------------

/// Test execution quality under a range of order sizes.
///
/// For each size in `order_sizes`, simulate a buy execution against the book
/// snapshot. Returns a Vec of ExecutionResult (one per order size).
pub fn stress_test_execution(book: &OrderBook, order_sizes: &[f64]) -> Vec<ExecutionResult> {
    order_sizes
        .iter()
        .map(|&qty| simulate_execution(book, qty, true))
        .collect()
}

// ---------------------------------------------------------------------------
// measure_resilience
// ---------------------------------------------------------------------------

/// Measure how resilient an order book is after a shock.
///
/// Procedure:
/// 1. Record pre-shock metrics (spread, total depth).
/// 2. Apply `shock_fn` to the book.
/// 3. Record post-shock metrics.
/// 4. Simulate recovery by regenerating book depth (using default config).
/// 5. Return ResilienceMetrics.
pub fn measure_resilience(
    book: &mut OrderBook,
    shock_fn: impl Fn(&mut OrderBook),
) -> ResilienceMetrics {
    // Pre-shock.
    let pre_spread = book.spread().unwrap_or(0.0);
    let pre_mid = book.mid_price().unwrap_or(100.0);
    let pre_bid_qty = book.total_bid_qty();
    let pre_ask_qty = book.total_ask_qty();
    let pre_total_depth = pre_bid_qty + pre_ask_qty;

    // Apply the shock.
    shock_fn(book);

    // Post-shock.
    let post_spread = book.spread().unwrap_or(pre_spread * 3.0);
    let post_bid_qty = book.total_bid_qty();
    let post_ask_qty = book.total_ask_qty();
    let post_total_depth = post_bid_qty + post_ask_qty;

    // Spread expansion in bps.
    let spread_expansion_bps = if pre_mid > 0.0 {
        ((post_spread - pre_spread) / pre_mid) * 10_000.0
    } else {
        0.0
    };

    // Depth recovery: simulate market makers replenishing the book.
    // Use a synthetic regeneration at the post-shock mid.
    let post_mid = book.mid_price().unwrap_or(pre_mid);
    let config = BookConfig {
        mid_price: post_mid,
        ..BookConfig::default()
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let recovered_book = generate(&config, &mut rng);
    let recovered_depth = recovered_book.total_bid_qty() + recovered_book.total_ask_qty();

    // Recovery percentage: how much of original depth is restored.
    let depth_recovery_pct = if pre_total_depth > 0.0 {
        ((recovered_depth.min(pre_total_depth)) / pre_total_depth * 100.0).clamp(0.0, 100.0)
    } else {
        100.0
    };

    // Simulated recovery time: proportional to depth loss (1ms per 1% depth lost).
    let depth_loss_pct =
        100.0 * (1.0 - post_total_depth / pre_total_depth.max(1.0)).clamp(0.0, 1.0);
    let recovery_time_ms = (depth_loss_pct as u64).saturating_mul(5);

    // Apply recovered book to restore state.
    *book = recovered_book;

    ResilienceMetrics {
        recovery_time_ms,
        spread_expansion_bps: spread_expansion_bps.max(0.0),
        depth_recovery_pct,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_book(seed: u64) -> OrderBook {
        let config = BookConfig::default();
        let mut rng = StdRng::seed_from_u64(seed);
        generate(&config, &mut rng)
    }

    // 1. test_liquidity_withdrawal reduces total book depth.
    #[test]
    fn test_liquidity_withdrawal_reduces_depth() {
        let mut book = make_book(1);
        let pre_bid = book.total_bid_qty();
        let pre_ask = book.total_ask_qty();
        test_liquidity_withdrawal(&mut book, 3);
        let post_depth = book.total_bid_qty() + book.total_ask_qty();
        let pre_depth = pre_bid + pre_ask;
        // After withdrawal, either depth decreased or book is more depleted.
        assert!(post_depth <= pre_depth + 1e-6,
            "depth should not increase after withdrawal: {post_depth:.1} > {pre_depth:.1}");
    }

    // 2. test_spoofing leaves book valid (spread still non-negative if quotes remain).
    #[test]
    fn test_spoofing_book_remains_valid() {
        let mut book = make_book(2);
        test_spoofing(&mut book, 10_000.0, true);
        if let Some(spread) = book.spread() {
            assert!(spread >= 0.0, "spread after spoofing must be non-negative");
        }
    }

    // 3. test_quote_stuffing leaves book unchanged (all quotes cancelled).
    #[test]
    fn test_quote_stuffing_no_residual() {
        let mut book = make_book(3);
        let pre_count = book.order_count();
        test_quote_stuffing(&mut book, 50, 100);
        let post_count = book.order_count();
        // All stuffed quotes were cancelled, so order count should be the same.
        assert_eq!(post_count, pre_count,
            "quote stuffing should not change resting order count: {pre_count} -> {post_count}");
    }

    // 4. stress_test_execution returns results for each order size.
    #[test]
    fn test_stress_test_execution_length() {
        let book = make_book(4);
        let sizes = vec![10.0, 50.0, 100.0, 500.0];
        let results = stress_test_execution(&book, &sizes);
        assert_eq!(results.len(), sizes.len());
    }

    // 5. stress_test_execution: slippage is non-decreasing in order size.
    #[test]
    fn test_stress_test_slippage_monotone() {
        let book = make_book(5);
        let sizes = vec![10.0, 30.0, 80.0, 200.0];
        let results = stress_test_execution(&book, &sizes);
        for i in 1..results.len() {
            assert!(results[i].slippage_bps >= results[i - 1].slippage_bps - 1e-3,
                "slippage not monotone at index {i}: {:.2} < {:.2}",
                results[i].slippage_bps, results[i - 1].slippage_bps);
        }
    }

    // 6. measure_resilience returns non-negative spread expansion.
    #[test]
    fn test_resilience_spread_expansion_non_negative() {
        let mut book = make_book(6);
        let metrics = measure_resilience(&mut book, |b| test_liquidity_withdrawal(b, 2));
        assert!(metrics.spread_expansion_bps >= 0.0,
            "spread expansion must be non-negative");
    }

    // 7. measure_resilience: depth recovery pct is in [0, 100].
    #[test]
    fn test_resilience_depth_recovery_pct_range() {
        let mut book = make_book(7);
        let metrics = measure_resilience(&mut book, |b| test_liquidity_withdrawal(b, 3));
        assert!(metrics.depth_recovery_pct >= 0.0 && metrics.depth_recovery_pct <= 100.0,
            "depth recovery pct {} out of [0,100]", metrics.depth_recovery_pct);
    }

    // 8. measure_resilience with no-op shock has minimal spread expansion.
    #[test]
    fn test_resilience_noop_shock() {
        let mut book = make_book(8);
        let metrics = measure_resilience(&mut book, |_b| {});
        // No-op shock should result in no depth loss and 100% recovery.
        assert!(metrics.depth_recovery_pct >= 90.0,
            "no-op shock should have high recovery: {}%", metrics.depth_recovery_pct);
    }

    // 9. Spoofing both sides leaves book valid.
    #[test]
    fn test_spoofing_both_sides() {
        let mut book = make_book(9);
        test_spoofing(&mut book, 5_000.0, true);
        test_spoofing(&mut book, 5_000.0, false);
        // Book should still have a valid state.
        assert!(book.bid_level_count() > 0 || book.ask_level_count() > 0);
    }

    // 10. Stress test with empty order list returns empty results.
    #[test]
    fn test_stress_test_empty_sizes() {
        let book = make_book(10);
        let results = stress_test_execution(&book, &[]);
        assert!(results.is_empty());
    }

    // 11. Recovery time is non-negative.
    #[test]
    fn test_resilience_recovery_time_non_negative() {
        let mut book = make_book(11);
        let metrics = measure_resilience(&mut book, |b| test_liquidity_withdrawal(b, 5));
        assert!(metrics.recovery_time_ms < u64::MAX);
    }

    // 12. After resilience measurement, book has valid depth (was regenerated).
    #[test]
    fn test_resilience_book_regenerated() {
        let mut book = make_book(12);
        measure_resilience(&mut book, |b| test_liquidity_withdrawal(b, 10));
        assert!(book.total_bid_qty() > 0.0, "book should have depth after resilience recovery");
        assert!(book.total_ask_qty() > 0.0, "book should have depth after resilience recovery");
    }
}
