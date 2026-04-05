//! Bootstrap resampling methods for time-series return data.
//!
//! Three methods are implemented, all behind a unified [`Bootstrapper`] trait:
//!
//! | Method | Block size | Preserves autocorrelation |
//! |--------|-----------|---------------------------|
//! | [`StationaryBootstrap`] | Geometric random (Politis-Romano) | Yes |
//! | [`CircularBlockBootstrap`] | Fixed, circular wrap | Yes |
//! | [`MovingBlockBootstrap`] | Fixed, no wrap | Yes |
//!
//! All methods accept a target sample size `n` and resample from the original
//! return series, preserving the serial dependence structure.

use anyhow::{bail, Result};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Bootstrapper trait
// ---------------------------------------------------------------------------

/// Unified interface for all bootstrap resampling methods.
pub trait Bootstrapper: Send + Sync {
    /// Draw a bootstrap sample of length `n` from `returns`.
    ///
    /// The returned series has the same statistical properties as the original
    /// (mean, variance, autocorrelation structure) while providing a new
    /// independent realisation.
    fn resample(&self, returns: &[f64], n: usize) -> Result<Vec<f64>>;

    /// Draw `m` independent bootstrap samples, each of length `n`.
    fn resample_many(&self, returns: &[f64], n: usize, m: usize) -> Result<Vec<Vec<f64>>> {
        (0..m).map(|_| self.resample(returns, n)).collect()
    }
}

// ---------------------------------------------------------------------------
// Stationary Bootstrap (Politis & Romano 1994)
// ---------------------------------------------------------------------------

/// Stationary Bootstrap with geometrically distributed block lengths.
///
/// The expected block length equals `1 / p` where `p` is the probability of
/// starting a new block on any given draw. This makes the resampled series
/// stationary (in the weak sense) which is the key advantage over fixed-block
/// methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationaryBootstrap {
    /// Probability of starting a new block (= 1 / expected_block_length).
    pub p_new_block: f64,
    /// Random seed (0 = non-deterministic per call).
    pub seed: u64,
}

impl StationaryBootstrap {
    /// Create a new stationary bootstrap with given expected block length.
    ///
    /// # Arguments
    /// * `expected_block_len` — average number of observations per block.
    ///   Typical values: 5–20 for daily return data.
    pub fn new(expected_block_len: f64) -> Self {
        assert!(
            expected_block_len >= 1.0,
            "expected_block_len must be >= 1"
        );
        Self {
            p_new_block: 1.0 / expected_block_len,
            seed: 0,
        }
    }

    /// Return a geometrically-distributed block length.
    /// P(L = k) = (1 - p)^(k-1) * p, mean = 1/p.
    fn sample_block_length(p: f64, rng: &mut SmallRng) -> usize {
        let u: f64 = rng.gen();
        // Geometric distribution: k = ceil(log(u) / log(1-p))
        let k = (u.ln() / (1.0 - p).ln()).ceil() as usize;
        k.max(1)
    }
}

impl Bootstrapper for StationaryBootstrap {
    fn resample(&self, returns: &[f64], n: usize) -> Result<Vec<f64>> {
        if returns.is_empty() {
            bail!("returns slice is empty");
        }
        let seed = if self.seed == 0 {
            rand::random()
        } else {
            self.seed
        };
        let mut rng = SmallRng::seed_from_u64(seed);
        let t = returns.len();
        let mut sample = Vec::with_capacity(n);

        while sample.len() < n {
            // Pick a random starting index
            let start: usize = rng.gen_range(0..t);
            let block_len = Self::sample_block_length(self.p_new_block, &mut rng);

            for j in 0..block_len {
                if sample.len() >= n {
                    break;
                }
                // Wrap around the series (circular)
                let idx = (start + j) % t;
                sample.push(returns[idx]);
            }
        }

        Ok(sample)
    }
}

// ---------------------------------------------------------------------------
// Circular Block Bootstrap (Politis & Romano 1992)
// ---------------------------------------------------------------------------

/// Circular Block Bootstrap with fixed block size.
///
/// The series is treated as circular: after the last observation, sampling
/// wraps around to the beginning. This avoids end-of-sample edge effects that
/// afflict the standard MBB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularBlockBootstrap {
    /// Fixed block size (number of consecutive observations per block).
    pub block_size: usize,
    /// Random seed (0 = non-deterministic).
    pub seed: u64,
}

impl CircularBlockBootstrap {
    /// Create a new CBB sampler.
    ///
    /// # Arguments
    /// * `block_size` — number of consecutive observations per block.
    ///   Rule of thumb: `block_size ≈ n^(1/3)` for daily data.
    pub fn new(block_size: usize) -> Self {
        assert!(block_size >= 1, "block_size must be >= 1");
        Self {
            block_size,
            seed: 0,
        }
    }

    /// Heuristic: choose block size as `ceil(n^(1/3))`.
    pub fn auto_block_size(n: usize) -> usize {
        ((n as f64).cbrt().ceil() as usize).max(1)
    }
}

impl Bootstrapper for CircularBlockBootstrap {
    fn resample(&self, returns: &[f64], n: usize) -> Result<Vec<f64>> {
        if returns.is_empty() {
            bail!("returns slice is empty");
        }
        let seed = if self.seed == 0 {
            rand::random()
        } else {
            self.seed
        };
        let mut rng = SmallRng::seed_from_u64(seed);
        let t = returns.len();
        let mut sample = Vec::with_capacity(n);

        while sample.len() < n {
            let start: usize = rng.gen_range(0..t); // any index (circular)
            for j in 0..self.block_size {
                if sample.len() >= n {
                    break;
                }
                let idx = (start + j) % t; // circular wrap
                sample.push(returns[idx]);
            }
        }

        Ok(sample)
    }
}

// ---------------------------------------------------------------------------
// Moving Block Bootstrap (Kunsch 1989, Liu & Singh 1992)
// ---------------------------------------------------------------------------

/// Moving Block Bootstrap — standard fixed-block, non-circular method.
///
/// Blocks are sampled uniformly from the set of all overlapping blocks of
/// length `block_size` in the original series. Unlike CBB, blocks cannot
/// wrap around, so the last `block_size - 1` observations are undersampled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovingBlockBootstrap {
    /// Fixed block size.
    pub block_size: usize,
    /// Random seed (0 = non-deterministic).
    pub seed: u64,
}

impl MovingBlockBootstrap {
    pub fn new(block_size: usize) -> Self {
        assert!(block_size >= 1, "block_size must be >= 1");
        Self {
            block_size,
            seed: 0,
        }
    }
}

impl Bootstrapper for MovingBlockBootstrap {
    fn resample(&self, returns: &[f64], n: usize) -> Result<Vec<f64>> {
        if returns.is_empty() {
            bail!("returns slice is empty");
        }
        if returns.len() < self.block_size {
            bail!(
                "series length {} is shorter than block_size {}",
                returns.len(),
                self.block_size
            );
        }
        let seed = if self.seed == 0 {
            rand::random()
        } else {
            self.seed
        };
        let mut rng = SmallRng::seed_from_u64(seed);
        let t = returns.len();
        let max_start = t - self.block_size; // non-circular: last valid start
        let mut sample = Vec::with_capacity(n);

        while sample.len() < n {
            let start: usize = rng.gen_range(0..=max_start);
            for j in 0..self.block_size {
                if sample.len() >= n {
                    break;
                }
                sample.push(returns[start + j]);
            }
        }

        Ok(sample)
    }
}

// ---------------------------------------------------------------------------
// Bootstrap utility functions
// ---------------------------------------------------------------------------

/// Compute the mean of a slice.
pub fn slice_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the standard deviation of a slice.
pub fn slice_std(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean = slice_mean(data);
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    var.sqrt()
}

/// Compute the auto-correlation at lag `lag` for a series.
pub fn autocorrelation(data: &[f64], lag: usize) -> f64 {
    if lag >= data.len() {
        return 0.0;
    }
    let mean = slice_mean(data);
    let n = data.len();
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-12 {
        return 0.0;
    }
    let cov: f64 = (0..n - lag)
        .map(|i| (data[i] - mean) * (data[i + lag] - mean))
        .sum::<f64>()
        / n as f64;
    cov / var
}

/// Heuristic optimal block length using Politis-White (2004) plug-in rule.
/// Approximated as `max(1, ceil(1.75 * n^(1/3)))` for daily return data.
pub fn optimal_block_length(n: usize) -> usize {
    ((1.75 * (n as f64).cbrt()).ceil() as usize).max(1)
}

/// Create a [`StationaryBootstrap`] with the Politis-White optimal block length.
pub fn stationary_bootstrap_auto(returns: &[f64]) -> StationaryBootstrap {
    let b = optimal_block_length(returns.len());
    StationaryBootstrap::new(b as f64)
}

// ---------------------------------------------------------------------------
// Bootstrap-based confidence intervals
// ---------------------------------------------------------------------------

/// Bootstrap confidence interval for the mean return.
///
/// Returns (lower, upper) at the given confidence level (e.g. 0.95).
pub fn bootstrap_mean_ci(
    returns: &[f64],
    bootstrapper: &dyn Bootstrapper,
    n_resamples: usize,
    confidence: f64,
) -> Result<(f64, f64)> {
    if returns.is_empty() {
        bail!("returns is empty");
    }
    let original_mean = slice_mean(returns);
    let n = returns.len();
    let mut boot_means: Vec<f64> = Vec::with_capacity(n_resamples);

    for _ in 0..n_resamples {
        let sample = bootstrapper.resample(returns, n)?;
        boot_means.push(slice_mean(&sample) - original_mean); // centred
    }

    boot_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - confidence;
    let lo_idx = ((alpha / 2.0) * n_resamples as f64).round() as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_resamples as f64)
        .round()
        .min((n_resamples - 1) as f64) as usize;

    // Percentile-t interval (basic bootstrap)
    let lower = original_mean - boot_means[hi_idx];
    let upper = original_mean - boot_means[lo_idx];

    Ok((lower, upper))
}

/// Bootstrap standard error of the mean return.
pub fn bootstrap_std_error(
    returns: &[f64],
    bootstrapper: &dyn Bootstrapper,
    n_resamples: usize,
) -> Result<f64> {
    let n = returns.len();
    let means: Vec<f64> = (0..n_resamples)
        .map(|_| bootstrapper.resample(returns, n).map(|s| slice_mean(&s)))
        .collect::<Result<Vec<f64>>>()?;
    Ok(slice_std(&means))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_returns(n: usize) -> Vec<f64> {
        // Deterministic alternating series for reproducibility
        (0..n)
            .map(|i| if i % 3 == 0 { 0.01 } else { -0.005 })
            .collect()
    }

    #[test]
    fn test_stationary_bootstrap_length() {
        let returns = synthetic_returns(100);
        let bs = StationaryBootstrap::new(10.0);
        let sample = bs.resample(&returns, 200).unwrap();
        assert_eq!(sample.len(), 200);
    }

    #[test]
    fn test_circular_bootstrap_length() {
        let returns = synthetic_returns(100);
        let bs = CircularBlockBootstrap::new(10);
        let sample = bs.resample(&returns, 150).unwrap();
        assert_eq!(sample.len(), 150);
    }

    #[test]
    fn test_moving_block_bootstrap_length() {
        let returns = synthetic_returns(100);
        let bs = MovingBlockBootstrap::new(10);
        let sample = bs.resample(&returns, 100).unwrap();
        assert_eq!(sample.len(), 100);
    }

    #[test]
    fn test_stationary_bootstrap_mean_preservation() {
        let returns = synthetic_returns(252);
        let orig_mean = slice_mean(&returns);
        let bs = StationaryBootstrap { p_new_block: 0.1, seed: 42 };
        let mut boot_means = Vec::new();
        for i in 0..200 {
            let mut bs_i = bs.clone();
            bs_i.seed = 42 + i as u64;
            let sample = bs_i.resample(&returns, returns.len()).unwrap();
            boot_means.push(slice_mean(&sample));
        }
        let boot_mean_of_means = slice_mean(&boot_means);
        // Bootstrap mean should be within 3× std of original
        assert!(
            (boot_mean_of_means - orig_mean).abs() < 0.005,
            "boot mean {} far from orig {}",
            boot_mean_of_means,
            orig_mean
        );
    }

    #[test]
    fn test_block_length_heuristic() {
        let b = optimal_block_length(252);
        assert!(b >= 5 && b <= 15, "expected 5-15, got {}", b);
    }

    #[test]
    fn test_autocorrelation_zero_lag() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let ac = autocorrelation(&data, 0);
        assert!((ac - 1.0).abs() < 1e-9, "lag-0 autocorr must be 1");
    }

    #[test]
    fn test_bootstrap_ci_covers_mean() {
        let returns = synthetic_returns(252);
        let bs = StationaryBootstrap { p_new_block: 0.1, seed: 99 };
        let (lo, hi) = bootstrap_mean_ci(&returns, &bs, 500, 0.95).unwrap();
        let mean = slice_mean(&returns);
        assert!(lo <= mean && mean <= hi, "CI [{}, {}] must contain mean {}", lo, hi, mean);
    }

    #[test]
    fn test_moving_block_short_series_error() {
        let returns = vec![0.01; 5];
        let bs = MovingBlockBootstrap::new(10);
        assert!(bs.resample(&returns, 10).is_err());
    }
}
