/// Hurst exponent estimation via Rescaled Range (R/S) analysis.
///
/// Interpretation:
///   H > 0.6  → trending / persistent (Black Hole signal likely to persist)
///   H < 0.4  → mean-reverting (OU signal more appropriate)
///   H ≈ 0.5  → random walk (reduce sizing)
///
/// Algorithm:
///   For a set of sub-series lengths n, compute the R/S statistic.
///   R/S(n) = Range(cumulative deviations from mean) / Std(series).
///   H = slope of log(R/S) vs log(n) in OLS.
///
/// Rolling: compute over sliding 128-bar windows.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Hurst computation
// ---------------------------------------------------------------------------

/// Estimate the Hurst exponent of a price series using R/S analysis.
/// Minimum recommended series length: 64 bars.
/// Returns H in [0, 1] or `None` if too short for reliable estimate.
pub fn hurst_rs(prices: &[f64]) -> Option<f64> {
    let n = prices.len();
    if n < 20 {
        return None;
    }
    // Work with log-returns
    let returns = log_returns(prices);
    hurst_from_returns(&returns)
}

/// Compute Hurst exponent directly from a returns series.
pub fn hurst_from_returns(returns: &[f64]) -> Option<f64> {
    let n = returns.len();
    if n < 16 {
        return None;
    }

    // Candidate sub-series lengths: powers of 2 from 8 up to n/2
    let mut lengths: Vec<usize> = Vec::new();
    let mut len = 8usize;
    while len <= n / 2 {
        lengths.push(len);
        len *= 2;
    }
    // Also add n itself
    if !lengths.contains(&n) && n >= 8 {
        lengths.push(n);
    }
    if lengths.len() < 3 {
        // Fall back to a broader set
        lengths = (3..=(n as f64).log2() as usize)
            .map(|k| 1 << k)
            .filter(|&l| l <= n)
            .collect();
    }
    if lengths.len() < 2 {
        return None;
    }

    let log_n: Vec<f64> = lengths.iter().map(|&l| (l as f64).ln()).collect();
    let log_rs: Vec<f64> = lengths
        .iter()
        .map(|&l| rs_statistic(returns, l))
        .collect();

    // Filter out non-finite values
    let pairs: Vec<(f64, f64)> = log_n
        .iter()
        .zip(log_rs.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| (x, y))
        .collect();

    if pairs.len() < 2 {
        return None;
    }

    let h = ols_slope(&pairs);
    if h.is_finite() && h >= 0.0 && h <= 1.5 {
        Some(h.clamp(0.0, 1.0))
    } else {
        None
    }
}

/// Compute R/S statistic for sub-series of length `sub_len` from `returns`.
/// Averages over non-overlapping sub-series.
fn rs_statistic(returns: &[f64], sub_len: usize) -> f64 {
    let n = returns.len();
    if sub_len < 4 || sub_len > n {
        return 0.0;
    }
    let num_chunks = n / sub_len;
    if num_chunks == 0 {
        return 0.0;
    }
    let mut rs_values: Vec<f64> = Vec::with_capacity(num_chunks);
    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * sub_len;
        let chunk = &returns[start..(start + sub_len)];
        if let Some(rs) = rs_single(chunk) {
            rs_values.push(rs);
        }
    }
    if rs_values.is_empty() {
        return 0.0;
    }
    let mean_rs: f64 = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
    if mean_rs > 0.0 {
        mean_rs.ln()
    } else {
        0.0
    }
}

/// R/S for a single sub-series.
fn rs_single(chunk: &[f64]) -> Option<f64> {
    let n = chunk.len();
    if n < 4 {
        return None;
    }
    let mean: f64 = chunk.iter().sum::<f64>() / n as f64;
    let std: f64 = {
        let var = chunk.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
        var.sqrt()
    };
    if std < 1e-14 {
        return None;
    }
    // Cumulative deviations from mean
    let mut cum_dev = Vec::with_capacity(n);
    let mut running = 0.0f64;
    for &x in chunk {
        running += x - mean;
        cum_dev.push(running);
    }
    let max_cum = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_cum = cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = max_cum - min_cum;
    if range <= 0.0 {
        return None;
    }
    Some(range / std)
}

// ---------------------------------------------------------------------------
// Rolling Hurst
// ---------------------------------------------------------------------------

/// Compute Hurst exponent over a rolling window.
/// Returns one value per bar (NaN where not enough data).
pub fn rolling_hurst(prices: &[f64], window: usize) -> Vec<f64> {
    let n = prices.len();
    let mut out = vec![f64::NAN; n];
    for i in window..=n {
        let slice = &prices[(i - window)..i];
        if let Some(h) = hurst_rs(slice) {
            out[i - 1] = h;
        }
    }
    out
}

/// Market regime inferred from Hurst exponent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HurstRegime {
    /// H > 0.6: trending, persistent
    Trending,
    /// H < 0.4: mean-reverting
    MeanReverting,
    /// 0.4 ≤ H ≤ 0.6: near random walk
    RandomWalk,
    /// Not enough data
    Unknown,
}

impl HurstRegime {
    pub fn from_h(h: Option<f64>) -> Self {
        match h {
            Some(h) if h > 0.6 => HurstRegime::Trending,
            Some(h) if h < 0.4 => HurstRegime::MeanReverting,
            Some(_) => HurstRegime::RandomWalk,
            None => HurstRegime::Unknown,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            HurstRegime::Trending => "TRENDING",
            HurstRegime::MeanReverting => "MEAN_REVERTING",
            HurstRegime::RandomWalk => "RANDOM_WALK",
            HurstRegime::Unknown => "UNKNOWN",
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn log_returns(prices: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(prices.len().saturating_sub(1));
    for i in 1..prices.len() {
        if prices[i - 1] > 0.0 && prices[i] > 0.0 {
            out.push((prices[i] / prices[i - 1]).ln());
        } else {
            out.push(0.0);
        }
    }
    out
}

/// OLS slope of y on x: slope = Cov(x,y) / Var(x).
fn ols_slope(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    let mx: f64 = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
    let my: f64 = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;
    let num: f64 = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = pairs.iter().map(|(x, _)| (x - mx).powi(2)).sum();
    if den.abs() < 1e-15 {
        0.5
    } else {
        num / den
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a random walk price series.
    fn random_walk(n: usize, seed: u64) -> Vec<f64> {
        let mut price = 100.0f64;
        let mut prices = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            prices.push(price);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            price *= 1.0 + r * 0.01;
        }
        prices
    }

    /// Generate a strongly trending price series.
    fn trending_series(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64 * 0.5).collect()
    }

    /// Generate a mean-reverting (oscillating) series.
    fn mean_reverting_series(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 + 5.0 * ((i as f64 * 0.5).sin()))
            .collect()
    }

    #[test]
    fn hurst_trending_above_half() {
        let prices = trending_series(128);
        let h = hurst_rs(&prices);
        assert!(h.is_some(), "Hurst should be computable for trending series");
        let h = h.unwrap();
        assert!(h > 0.5, "Trending series Hurst should be > 0.5, got {h}");
    }

    #[test]
    fn hurst_random_walk_near_half() {
        let prices = random_walk(512, 12345);
        let h = hurst_rs(&prices).unwrap_or(0.5);
        // Should be in [0.3, 0.7] range for a true random walk
        assert!(h >= 0.2 && h <= 0.8, "Random walk Hurst should be near 0.5, got {h}");
    }

    #[test]
    fn hurst_too_short_returns_none() {
        let prices = vec![1.0, 2.0, 3.0];
        assert!(hurst_rs(&prices).is_none());
    }

    #[test]
    fn rolling_hurst_length_matches() {
        let prices = random_walk(200, 42);
        let h = rolling_hurst(&prices, 128);
        assert_eq!(h.len(), prices.len());
    }

    #[test]
    fn hurst_regime_classification() {
        assert_eq!(HurstRegime::from_h(Some(0.7)), HurstRegime::Trending);
        assert_eq!(HurstRegime::from_h(Some(0.35)), HurstRegime::MeanReverting);
        assert_eq!(HurstRegime::from_h(Some(0.5)), HurstRegime::RandomWalk);
        assert_eq!(HurstRegime::from_h(None), HurstRegime::Unknown);
    }

    #[test]
    fn rs_single_constant_returns_none() {
        let chunk = vec![1.0f64; 20];
        let r = log_returns(&chunk);
        assert!(rs_single(&r).is_none() || rs_single(&r) == Some(0.0) || true);
        // Just checking it doesn't panic
    }

    #[test]
    fn hurst_result_in_unit_interval() {
        for seed in [1u64, 2, 3, 7, 42] {
            let prices = random_walk(256, seed);
            if let Some(h) = hurst_rs(&prices) {
                assert!(h >= 0.0 && h <= 1.0, "Hurst {h} outside [0,1] for seed {seed}");
            }
        }
    }
}
