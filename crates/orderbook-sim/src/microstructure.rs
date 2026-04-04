/// Market microstructure analytics.

use crate::fills::Fill;
use crate::order::OrderSide;

// ── Bid-Ask Spread Decomposition ─────────────────────────────────────────────

/// Decompose the bid-ask spread into three components using the
/// Glosten-Milgrom adverse-selection / inventory / order-processing framework.
///
/// This simplified decomposition uses:
/// - **Adverse selection** (informed trading): correlated with order imbalance.
/// - **Inventory**: proportional to net dealer position.
/// - **Order processing**: residual.
///
/// Returns `(adverse_selection, inventory, order_processing)` as fractions of half-spread.
pub fn bid_ask_spread_decomposition(
    fills: &[Fill],
    mid_prices: &[f64],
) -> (f64, f64, f64) {
    if fills.len() < 2 || mid_prices.len() < 2 {
        return (0.0, 0.0, 0.0);
    }
    let n = fills.len().min(mid_prices.len()) - 1;

    // Compute trade sign sequence (+1 = buy, -1 = sell).
    let signs: Vec<f64> = fills
        .iter()
        .map(|f| match f.aggressor_side {
            OrderSide::Buy => 1.0,
            OrderSide::Sell => -1.0,
        })
        .collect();

    // Price changes Δp_t.
    let delta_p: Vec<f64> = (0..n)
        .map(|i| mid_prices[i + 1] - mid_prices[i])
        .collect();

    // Adverse selection: E[Δp * sign] / E[Δp²] * sign variance
    let cov_sign_delta: f64 = (0..n).map(|i| signs[i] * delta_p[i]).sum::<f64>() / n as f64;

    // Inventory: net position proxy
    let net_sign: f64 = signs[..n].iter().sum::<f64>() / n as f64;

    // Half spread estimate (Roll).
    let half_spread = roll_spread_estimator(mid_prices).abs() / 2.0;

    let adverse = cov_sign_delta.abs().min(half_spread);
    let inventory = (net_sign.abs() * half_spread * 0.2).min(half_spread - adverse);
    let order_processing = (half_spread - adverse - inventory).max(0.0);

    (adverse, inventory, order_processing)
}

// ── Kyle's Lambda ─────────────────────────────────────────────────────────────

/// Kyle's λ: price impact per unit of signed order flow.
///
/// Regresses mid-price changes on signed volume: Δp = λ * Q + ε.
/// Returns λ.
pub fn kyle_lambda(fills: &[Fill], mid_prices: &[f64]) -> f64 {
    let n = fills.len().min(mid_prices.len().saturating_sub(1));
    if n < 2 {
        return 0.0;
    }

    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    let mut sum_xy = 0.0_f64;

    for i in 0..n {
        let sign = match fills[i].aggressor_side {
            OrderSide::Buy => 1.0,
            OrderSide::Sell => -1.0,
        };
        let x = sign * fills[i].qty; // signed volume
        let y = mid_prices[i + 1] - mid_prices[i]; // price change
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    let nf = n as f64;
    let denom = nf * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (nf * sum_xy - sum_x * sum_y) / denom
}

// ── Amihud Illiquidity ────────────────────────────────────────────────────────

/// Amihud (2002) illiquidity ratio: |r_t| / Volume_t, averaged over T days.
///
/// Higher values indicate lower liquidity (more price impact per dollar traded).
pub fn amihud_illiquidity(returns: &[f64], dollar_volumes: &[f64]) -> f64 {
    let n = returns.len().min(dollar_volumes.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = (0..n)
        .filter(|&i| dollar_volumes[i] > 0.0)
        .map(|i| returns[i].abs() / dollar_volumes[i])
        .sum();
    sum / n as f64
}

// ── Roll Spread Estimator ─────────────────────────────────────────────────────

/// Roll (1984) spread estimator.
///
/// Estimates the effective half-spread from the first-order serial covariance
/// of price changes: S = 2 * sqrt(-Cov(Δp_t, Δp_{t-1})).
/// Returns the full spread (positive value). If covariance is positive
/// (trending market), returns 0.
pub fn roll_spread_estimator(prices: &[f64]) -> f64 {
    let n = prices.len();
    if n < 3 {
        return 0.0;
    }

    let deltas: Vec<f64> = (0..n - 1).map(|i| prices[i + 1] - prices[i]).collect();
    let m = deltas.len();
    let mean_d = deltas.iter().sum::<f64>() / m as f64;

    let cov: f64 = (0..m - 1)
        .map(|i| (deltas[i] - mean_d) * (deltas[i + 1] - mean_d))
        .sum::<f64>()
        / (m - 1) as f64;

    if cov >= 0.0 {
        0.0
    } else {
        2.0 * (-cov).sqrt()
    }
}

// ── Effective & Realized Spread ───────────────────────────────────────────────

/// Effective spread = 2 * |fill_price - mid_price| (per-fill average).
pub fn effective_spread(fills: &[Fill], mid_prices: &[f64]) -> f64 {
    let n = fills.len().min(mid_prices.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = (0..n)
        .map(|i| 2.0 * (fills[i].price - mid_prices[i]).abs())
        .sum();
    sum / n as f64
}

/// Realized spread = 2 * sign * (fill_price - mid_price_{t+delay}).
///
/// Measures the dealer's profit after the information component has resolved.
/// `delay_bars` is the number of bars to look forward for the post-fill mid.
pub fn realized_spread(fills: &[Fill], mid_prices: &[f64], delay_bars: usize) -> f64 {
    let n = fills.len();
    if n == 0 || mid_prices.len() <= delay_bars {
        return 0.0;
    }
    let mut count = 0usize;
    let mut sum = 0.0_f64;
    for (i, fill) in fills.iter().enumerate() {
        let future_idx = i + delay_bars;
        if future_idx >= mid_prices.len() {
            break;
        }
        let sign = match fill.aggressor_side {
            OrderSide::Buy => 1.0,
            OrderSide::Sell => -1.0,
        };
        let mid_future = mid_prices[future_idx];
        sum += 2.0 * sign * (fill.price - mid_future);
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

// ── Price Impact Curve ────────────────────────────────────────────────────────

/// Estimate price impact as a function of order size using a power-law fit.
///
/// Fits: impact = a * qty^b  via OLS in log-log space.
/// Returns (a, b) parameters.
pub fn price_impact_curve(
    order_sizes: &[f64],
    price_impacts: &[f64],
) -> (f64, f64) {
    let n = order_sizes.len().min(price_impacts.len());
    if n < 2 {
        return (0.0, 0.5);
    }
    // Filter non-positive values (can't log).
    let pairs: Vec<(f64, f64)> = (0..n)
        .filter(|&i| order_sizes[i] > 0.0 && price_impacts[i] > 0.0)
        .map(|i| (order_sizes[i].ln(), price_impacts[i].ln()))
        .collect();
    let m = pairs.len() as f64;
    if m < 2.0 {
        return (0.0, 0.5);
    }
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / m;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / m;
    let ss_xx = pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>();
    let ss_xy = pairs.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum::<f64>();
    if ss_xx.abs() < 1e-12 {
        return (0.0, 0.5);
    }
    let b = ss_xy / ss_xx;
    let a = (mean_y - b * mean_x).exp();
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roll_spread_basic() {
        // Prices oscillating between bid and ask.
        let prices = vec![100.0, 100.1, 100.0, 100.1, 100.0, 100.1];
        let spread = roll_spread_estimator(&prices);
        assert!(spread > 0.0, "spread={spread}");
    }

    #[test]
    fn amihud_basic() {
        let returns = vec![0.01, -0.02, 0.005];
        let vols = vec![1e6, 2e6, 0.5e6];
        let ratio = amihud_illiquidity(&returns, &vols);
        assert!(ratio > 0.0);
    }
}
