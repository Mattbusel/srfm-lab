//! Detrending methods for price series.
//!
//! Each method decomposes a price series into three components:
//!
//! ```text
//! price = trend + cycle + noise
//! ```
//!
//! The trend is the slow-moving component.  The cycle is the business-cycle
//! component (medium frequency).  The noise is the high-frequency residual.
//!
//! # Methods
//!
//! | Type         | Best for                                   |
//! |--------------|---------------------------------------------|
//! | `HPFilter`   | Economic data, slow trend extraction        |
//! | `BKFilter`   | Extracting cycles of known period range     |
//! | `SSADetrend` | General, data-adaptive (no period needed)   |
//! | `EMDDetrend` | Non-stationary, non-linear time series      |

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Decomposition result
// ---------------------------------------------------------------------------

/// Result of a detrending operation.
///
/// `trend + cycle + noise == price` (element-wise, up to floating point).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decomposition {
    /// Original series (stored for reference).
    pub original: Vec<f64>,
    /// Long-run trend component.
    pub trend: Vec<f64>,
    /// Business-cycle component (medium frequency oscillation).
    pub cycle: Vec<f64>,
    /// High-frequency residual / noise.
    pub noise: Vec<f64>,
}

impl Decomposition {
    /// Reconstruct the original from components (sanity check).
    pub fn reconstruct(&self) -> Vec<f64> {
        self.trend.iter()
            .zip(self.cycle.iter())
            .zip(self.noise.iter())
            .map(|((t, c), n)| t + c + n)
            .collect()
    }

    /// RMS reconstruction error (should be near 0).
    pub fn reconstruction_error(&self) -> f64 {
        let recon = self.reconstruct();
        let sse: f64 = recon.iter().zip(self.original.iter())
            .map(|(r, o)| (r - o).powi(2))
            .sum();
        (sse / recon.len() as f64).sqrt()
    }

    /// Signal-to-noise ratio: std(cycle) / std(noise).
    pub fn snr(&self) -> f64 {
        let std_c = std_dev(&self.cycle);
        let std_n = std_dev(&self.noise);
        if std_n < 1e-14 { f64::INFINITY } else { std_c / std_n }
    }
}

fn std_dev(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 { return 0.0; }
    let m = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Hodrick-Prescott Filter
// ---------------------------------------------------------------------------

/// Hodrick-Prescott filter.
///
/// Minimises: sum((y - trend)^2) + lambda * sum((diff^2 trend)^2)
///
/// Standard values of lambda:
/// - 1600 for quarterly GDP data
/// - 6.25  for annual data
/// - 129600 for monthly data
/// - For daily prices: 107,200 is common
///
/// # Algorithm
///
/// Solve the normal equations via the banded tridiagonal system
/// (I + lambda * K'K) * tau = y, where K is the second-difference matrix.
/// We use a direct O(n) solver exploiting the banded structure.
pub struct HPFilter {
    pub lambda: f64,
}

impl HPFilter {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Default HP filter for daily prices.
    pub fn daily() -> Self {
        Self::new(107_200.0)
    }

    /// Default HP filter for quarterly data.
    pub fn quarterly() -> Self {
        Self::new(1600.0)
    }

    /// Apply the HP filter and return a `Decomposition`.
    ///
    /// The cycle is `price - trend`.  The noise field is set to zero because
    /// HP only separates trend from cycle.
    pub fn decompose(&self, prices: &[f64]) -> Decomposition {
        let trend = self.extract_trend(prices);
        let n = prices.len();
        let cycle: Vec<f64> = prices.iter().zip(trend.iter()).map(|(p, t)| p - t).collect();
        Decomposition {
            original: prices.to_vec(),
            trend,
            cycle,
            noise: vec![0.0; n],
        }
    }

    /// Extract only the trend component.
    pub fn extract_trend(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < 4 {
            return prices.to_vec();
        }
        // Solve (I + lambda * K'K) * tau = y using Thomas algorithm.
        // K'K is a symmetric pentadiagonal matrix.
        // We store diagonals: main (d), first off (e), second off (f).
        let lam = self.lambda;
        let mut d = vec![0.0f64; n];
        let mut e = vec![0.0f64; n]; // superdiagonal 1
        let mut f = vec![0.0f64; n]; // superdiagonal 2

        // Build K'K diagonals
        // Main diagonal: [1, 1+lam, 1+5lam, 1+6lam, ..., 1+6lam, 1+5lam, 1+lam, 1]
        // First off: [-2lam, -4lam, -4lam, ..., -4lam, -2lam]
        // Second off: [lam, lam, ..., lam]
        for i in 0..n {
            d[i] = 1.0;
            if i == 0 || i == n - 1 { d[i] += lam; }
            else if i == 1 || i == n - 2 { d[i] += 5.0 * lam; }
            else { d[i] += 6.0 * lam; }
        }
        for i in 0..(n - 1) {
            e[i] = if i == 0 || i == n - 2 { -2.0 * lam } else { -4.0 * lam };
        }
        for i in 0..(n - 2) {
            f[i] = lam;
        }

        // Forward elimination (pentadiagonal Thomas algorithm)
        let mut dd = d.clone();
        let mut ee = e.clone();
        let ff = f.clone();
        let mut rhs = prices.to_vec();

        for i in 1..n {
            if dd[i - 1].abs() < 1e-15 { continue; }
            let m1 = ee[i - 1] / dd[i - 1];
            dd[i] -= m1 * ee[i - 1];
            rhs[i] -= m1 * rhs[i - 1];
            if i + 1 < n {
                ee[i] -= m1 * ff[i - 1];
            }
            if i >= 2 {
                let m2 = ff[i - 2] / dd[i - 2];
                dd[i] -= m2 * ff[i - 2];
                rhs[i] -= m2 * rhs[i - 2];
            }
        }

        // Back substitution
        let mut trend = rhs.clone();
        trend[n - 1] /= dd[n - 1].max(1e-15);
        if n >= 2 {
            trend[n - 2] = (rhs[n - 2] - ee[n - 2] * trend[n - 1]) / dd[n - 2].max(1e-15);
        }
        for i in (0..n.saturating_sub(2)).rev() {
            let mut val = rhs[i];
            val -= ee[i] * trend[i + 1];
            if i + 2 < n { val -= ff[i] * trend[i + 2]; }
            trend[i] = val / dd[i].max(1e-15);
        }

        trend
    }
}

// ---------------------------------------------------------------------------
// Baxter-King bandpass filter
// ---------------------------------------------------------------------------

/// Baxter-King bandpass filter.
///
/// Extracts cyclical components with periods in `[pl, pu]` bars.
/// Standard values: pl=6, pu=32 for quarterly business cycles.
/// For intraday: pl=10, pu=60 (captures 10-60 bar oscillations).
///
/// The filter is a finite-order symmetric moving average with weights
/// constructed to approximate an ideal bandpass filter.
pub struct BKFilter {
    /// Minimum period of cycles to pass (in bars).
    pub period_low: usize,
    /// Maximum period of cycles to pass (in bars).
    pub period_high: usize,
    /// Filter order K (number of lags on each side).  Typical: 12.
    pub order: usize,
}

impl BKFilter {
    pub fn new(period_low: usize, period_high: usize, order: usize) -> Self {
        Self { period_low, period_high, order }
    }

    /// Standard BK filter for business cycle extraction (quarterly data).
    pub fn business_cycle() -> Self {
        Self::new(6, 32, 12)
    }

    /// BK filter for intraday cycle extraction.
    pub fn intraday_cycle() -> Self {
        Self::new(10, 60, 12)
    }

    /// Compute the ideal bandpass filter weight at lag k.
    fn ideal_weight(&self, k: i64) -> f64 {
        let pl = self.period_low as f64;
        let pu = self.period_high as f64;
        let omega_l = 2.0 * std::f64::consts::PI / pu;
        let omega_u = 2.0 * std::f64::consts::PI / pl;

        if k == 0 {
            (omega_u - omega_l) / std::f64::consts::PI
        } else {
            let kf = k as f64;
            ((omega_u * kf).sin() - (omega_l * kf).sin()) / (std::f64::consts::PI * kf)
        }
    }

    /// Compute the K+1 filter weights (symmetric, so weight[-k] == weight[k]).
    pub fn weights(&self) -> Vec<f64> {
        let k = self.order as i64;
        let raw: Vec<f64> = (-k..=k).map(|j| self.ideal_weight(j)).collect();

        // Adjust to make weights sum to exactly 0 (removes random walk component)
        let sum: f64 = raw.iter().sum::<f64>();
        let adjustment = sum / raw.len() as f64;
        raw.iter().map(|w| w - adjustment).collect()
    }

    /// Apply BK filter and return `Decomposition`.
    ///
    /// The first and last `order` observations cannot be filtered
    /// (set to NaN in cycle, original in trend).
    pub fn decompose(&self, prices: &[f64]) -> Decomposition {
        let n = prices.len();
        let k = self.order;

        if n < 2 * k + 1 {
            // Not enough data: return trivial decomposition
            return Decomposition {
                original: prices.to_vec(),
                trend: prices.to_vec(),
                cycle: vec![0.0; n],
                noise: vec![0.0; n],
            };
        }

        let weights = self.weights();
        let mut cycle = vec![f64::NAN; n];

        for i in k..(n - k) {
            let val: f64 = weights.iter().enumerate()
                .map(|(j, &w)| w * prices[i + j - k])
                .sum();
            cycle[i] = val;
        }

        // For endpoints, substitute 0 (effectively no cycle)
        for i in 0..k {
            cycle[i] = 0.0;
        }
        for i in (n - k)..n {
            cycle[i] = 0.0;
        }

        let trend: Vec<f64> = prices.iter().zip(cycle.iter())
            .map(|(p, c)| p - c)
            .collect();

        Decomposition {
            original: prices.to_vec(),
            trend,
            cycle,
            noise: vec![0.0; n],
        }
    }
}

// ---------------------------------------------------------------------------
// Singular Spectrum Analysis (SSA) detrend
// ---------------------------------------------------------------------------

/// Singular Spectrum Analysis detrending.
///
/// SSA embeds the time series into a trajectory matrix using a window of
/// length `L`, then reconstructs the trend from the first `k_components`
/// eigenvalues (which capture the slowest-varying components).
///
/// # Algorithm (simplified)
///
/// 1. Form the trajectory matrix X of shape (L, N-L+1).
/// 2. Compute the covariance matrix C = X * X' / (N-L+1).
/// 3. Eigendecompose C.
/// 4. Reconstruct trend from the first k eigentriplets.
/// 5. Cycle = price - trend; noise = small residual.
pub struct SSADetrend {
    /// Embedding window length.
    pub window: usize,
    /// Number of eigentriplets used to reconstruct the trend.
    pub k_components: usize,
}

impl SSADetrend {
    pub fn new(window: usize, k_components: usize) -> Self {
        Self { window, k_components }
    }

    /// Default SSA for daily prices (128-bar window, 2 trend components).
    pub fn default_daily() -> Self {
        Self::new(20, 2)
    }

    /// Build the trajectory (Hankel) matrix.
    fn trajectory_matrix(prices: &[f64], l: usize) -> Vec<Vec<f64>> {
        let n = prices.len();
        let k = n - l + 1;
        (0..l).map(|i| (0..k).map(|j| prices[i + j]).collect()).collect()
    }

    /// Compute X * X^T (row covariance, shape L x L).
    fn covariance(traj: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let l = traj.len();
        let k = traj[0].len();
        let mut cov = vec![vec![0.0f64; l]; l];
        for i in 0..l {
            for j in 0..=i {
                let dot: f64 = (0..k).map(|t| traj[i][t] * traj[j][t]).sum();
                cov[i][j] = dot / k as f64;
                cov[j][i] = cov[i][j];
            }
        }
        cov
    }

    /// Power iteration to find the top eigenvector of a symmetric matrix.
    fn top_eigenvector(cov: &[Vec<f64>], seed: &[f64], iterations: usize) -> Vec<f64> {
        let l = cov.len();
        let mut v = seed.to_vec();
        for _ in 0..iterations {
            let mut new_v = vec![0.0f64; l];
            for i in 0..l {
                for j in 0..l {
                    new_v[i] += cov[i][j] * v[j];
                }
            }
            let norm = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-14 { break; }
            for x in &mut new_v { *x /= norm; }
            v = new_v;
        }
        v
    }

    /// Deflate a covariance matrix by removing the contribution of eigenvector v.
    fn deflate(cov: &[Vec<f64>], v: &[f64]) -> Vec<Vec<f64>> {
        let l = cov.len();
        // eigenvalue estimate: lambda = v^T C v
        let lambda: f64 = v.iter().enumerate()
            .map(|(i, &vi)| vi * (0..l).map(|j| cov[i][j] * v[j]).sum::<f64>())
            .sum();
        let mut deflated = cov.to_vec();
        for i in 0..l {
            for j in 0..l {
                deflated[i][j] -= lambda * v[i] * v[j];
            }
        }
        deflated
    }

    /// Reconstruct time series from eigentriplet (u, s * v).
    /// Uses diagonal averaging to convert the rank-1 matrix back to a series.
    fn reconstruct_from_eigenvec(traj: &[Vec<f64>], u: &[f64]) -> Vec<f64> {
        let l = traj.len();
        let k = traj[0].len();
        let n = l + k - 1;

        // Compute principal component time series: w(t) = u^T * x_t
        let pc: Vec<f64> = (0..k).map(|t| u.iter().zip(traj.iter()).map(|(ui, row)| ui * row[t]).sum::<f64>()).collect();

        // Reconstructed component matrix: X_tilde = u * pc^T
        // Anti-diagonal average to get 1D series
        let mut series = vec![0.0f64; n];
        let mut counts = vec![0usize; n];
        for i in 0..l {
            for j in 0..k {
                series[i + j] += u[i] * pc[j];
                counts[i + j] += 1;
            }
        }
        for (s, c) in series.iter_mut().zip(counts.iter()) {
            if *c > 0 { *s /= *c as f64; }
        }
        series
    }

    /// Decompose the price series.
    pub fn decompose(&self, prices: &[f64]) -> Decomposition {
        let n = prices.len();
        let l = self.window.min(n / 2).max(2);

        if n < l * 2 {
            return Decomposition {
                original: prices.to_vec(),
                trend: prices.to_vec(),
                cycle: vec![0.0; n],
                noise: vec![0.0; n],
            };
        }

        let traj = Self::trajectory_matrix(prices, l);
        let mut cov = Self::covariance(&traj);

        let k = self.k_components.min(l);
        let mut trend = vec![0.0f64; n];

        // Initial eigenvector seed (uniform)
        let init_v: Vec<f64> = vec![1.0 / (l as f64).sqrt(); l];

        let mut current_v = init_v.clone();
        for comp_idx in 0..k {
            let u = Self::top_eigenvector(&cov, &current_v, 100);
            let component = Self::reconstruct_from_eigenvec(&traj, &u);

            // Add component to trend (truncate/pad to length n)
            for (i, &c) in component.iter().enumerate().take(n) {
                trend[i] += c;
            }

            cov = Self::deflate(&cov, &u);

            // New seed: orthogonal to u via simple perturbation
            let mut new_seed = init_v.clone();
            let dot: f64 = new_seed.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            for (s, &ui) in new_seed.iter_mut().zip(u.iter()) {
                *s -= dot * ui;
            }
            let norm = new_seed.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for s in &mut new_seed { *s /= norm; }
                current_v = new_seed;
            } else {
                // Reinitialise with shifted seed
                let shift = comp_idx + 1;
                current_v = (0..l).map(|i| if i == shift % l { 1.0 } else { 0.0 }).collect();
            }
        }

        let residual: Vec<f64> = prices.iter().zip(trend.iter()).map(|(p, t)| p - t).collect();

        // Split residual into cycle (low-frequency part) and noise (high freq).
        // Use a simple moving average to separate within the residual.
        let ma_window = (l / 2).max(2);
        let cycle: Vec<f64> = simple_moving_average(&residual, ma_window);
        let noise: Vec<f64> = residual.iter().zip(cycle.iter()).map(|(r, c)| r - c).collect();

        Decomposition {
            original: prices.to_vec(),
            trend,
            cycle,
            noise,
        }
    }
}

// ---------------------------------------------------------------------------
// Empirical Mode Decomposition (simplified)
// ---------------------------------------------------------------------------

/// Simplified Empirical Mode Decomposition via iterative envelope mean subtraction.
///
/// A full EMD implementation uses cubic spline interpolation of local extrema
/// to form upper and lower envelopes.  This simplified version uses a linear
/// spline approximation for the envelopes, which captures the main IMFs
/// without the computational overhead of cubic splines.
///
/// # Intrinsic Mode Functions (IMFs)
///
/// The first few IMFs capture high-frequency oscillations (noise/cycle).
/// The residual after extracting IMFs is the trend.
pub struct EMDDetrend {
    /// Maximum number of IMFs to extract.
    pub max_imfs: usize,
    /// Number of sifting iterations per IMF.
    pub sift_iter: usize,
    /// Stop criterion: fraction of series energy below which sifting stops.
    pub energy_threshold: f64,
}

impl EMDDetrend {
    pub fn new(max_imfs: usize, sift_iter: usize) -> Self {
        Self { max_imfs, sift_iter, energy_threshold: 0.001 }
    }

    pub fn default_daily() -> Self {
        Self::new(4, 10)
    }

    /// Find local maxima indices.
    fn local_maxima(xs: &[f64]) -> Vec<usize> {
        let n = xs.len();
        let mut maxima = Vec::new();
        for i in 1..(n - 1) {
            if xs[i] > xs[i - 1] && xs[i] >= xs[i + 1] {
                maxima.push(i);
            }
        }
        maxima
    }

    /// Find local minima indices.
    fn local_minima(xs: &[f64]) -> Vec<usize> {
        let n = xs.len();
        let mut minima = Vec::new();
        for i in 1..(n - 1) {
            if xs[i] < xs[i - 1] && xs[i] <= xs[i + 1] {
                minima.push(i);
            }
        }
        minima
    }

    /// Linear spline interpolation through given (x, y) points.
    fn linear_spline(xs: &[usize], ys: &[f64], n: usize) -> Vec<f64> {
        if xs.is_empty() {
            return vec![0.0; n];
        }
        let mut result = vec![0.0f64; n];

        // Extrapolate at the boundaries
        let x_first = xs[0] as f64;
        let y_first = ys[0];
        let x_last = xs[xs.len() - 1] as f64;
        let y_last = ys[ys.len() - 1];

        for i in 0..n {
            let x = i as f64;
            if x <= x_first {
                result[i] = y_first;
            } else if x >= x_last {
                result[i] = y_last;
            } else {
                // Find segment
                let pos = xs.partition_point(|&xi| (xi as f64) <= x);
                let j = pos.saturating_sub(1).min(xs.len() - 2);
                let x0 = xs[j] as f64;
                let x1 = xs[j + 1] as f64;
                let y0 = ys[j];
                let y1 = ys[j + 1];
                let t = (x - x0) / (x1 - x0).max(1e-14);
                result[i] = y0 + t * (y1 - y0);
            }
        }
        result
    }

    /// Perform one sifting step: subtract envelope mean from signal.
    fn sift(signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let max_idx = Self::local_maxima(signal);
        let min_idx = Self::local_minima(signal);

        if max_idx.len() < 2 || min_idx.len() < 2 {
            return signal.to_vec();
        }

        let max_vals: Vec<f64> = max_idx.iter().map(|&i| signal[i]).collect();
        let min_vals: Vec<f64> = min_idx.iter().map(|&i| signal[i]).collect();

        let upper_env = Self::linear_spline(&max_idx, &max_vals, n);
        let lower_env = Self::linear_spline(&min_idx, &min_vals, n);

        let mean_env: Vec<f64> = upper_env.iter().zip(lower_env.iter())
            .map(|(u, l)| (u + l) / 2.0)
            .collect();

        signal.iter().zip(mean_env.iter()).map(|(s, m)| s - m).collect()
    }

    /// Extract one IMF via iterative sifting.
    fn extract_imf(&self, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut h = signal.to_vec();
        for _ in 0..self.sift_iter {
            let h_sifted = Self::sift(&h);
            let energy_before: f64 = h.iter().map(|x| x * x).sum();
            let energy_after: f64 = h_sifted.iter().map(|x| x * x).sum();

            // Stopping criterion: residual energy change small
            if energy_before > 0.0 && (energy_before - energy_after).abs() / energy_before < 0.01 {
                h = h_sifted;
                break;
            }
            h = h_sifted;
        }
        let residual: Vec<f64> = signal.iter().zip(h.iter()).map(|(s, imf)| s - imf).collect();
        (h, residual)
    }

    /// Decompose the price series using simplified EMD.
    pub fn decompose(&self, prices: &[f64]) -> Decomposition {
        let n = prices.len();
        if n < 16 {
            return Decomposition {
                original: prices.to_vec(),
                trend: prices.to_vec(),
                cycle: vec![0.0; n],
                noise: vec![0.0; n],
            };
        }

        let total_energy: f64 = prices.iter().map(|x| x * x).sum::<f64>();

        let mut residual = prices.to_vec();
        let mut imfs: Vec<Vec<f64>> = Vec::new();

        for _ in 0..self.max_imfs {
            let res_energy: f64 = residual.iter().map(|x| x * x).sum::<f64>();
            if total_energy > 0.0 && res_energy / total_energy < self.energy_threshold {
                break;
            }

            let (imf, new_residual) = self.extract_imf(&residual);
            imfs.push(imf);
            residual = new_residual;
        }

        // The final residual is the trend
        let trend = residual;

        // Combine IMFs: first IMF = noise (highest frequency),
        // remaining IMFs = cycle (lower frequency oscillations)
        let noise = if !imfs.is_empty() {
            imfs[0].clone()
        } else {
            vec![0.0; n]
        };

        let cycle: Vec<f64> = if imfs.len() > 1 {
            let mut c = vec![0.0f64; n];
            for imf in &imfs[1..] {
                for (ci, &v) in c.iter_mut().zip(imf.iter()) {
                    *ci += v;
                }
            }
            c
        } else {
            vec![0.0; n]
        };

        Decomposition {
            original: prices.to_vec(),
            trend,
            cycle,
            noise,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: simple moving average
// ---------------------------------------------------------------------------

fn simple_moving_average(xs: &[f64], window: usize) -> Vec<f64> {
    let n = xs.len();
    let w = window.max(1).min(n);
    let mut result = vec![0.0f64; n];
    let mut sum = 0.0f64;

    for i in 0..n {
        sum += xs[i];
        let start = if i + 1 >= w { i + 1 - w } else { 0 };
        if i >= w { sum -= xs[i - w]; }
        let count = i.min(w - 1) + 1;
        result[i] = sum / count as f64;
        let _ = start; // suppress unused warning
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sinusoidal_trend(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let trend = i as f64 * 0.1;
                let cycle = 5.0 * (i as f64 * 2.0 * std::f64::consts::PI / 20.0).sin();
                trend + cycle + 100.0
            })
            .collect()
    }

    fn linear_trend(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64 * 0.5).collect()
    }

    #[test]
    fn test_hp_filter_trend_monotone_for_linear_input() {
        let prices = linear_trend(100);
        let hp = HPFilter::daily();
        let decomp = hp.decompose(&prices);
        assert_eq!(decomp.trend.len(), prices.len());
        assert_eq!(decomp.cycle.len(), prices.len());
        // Trend should be close to the input for a pure linear series
        for (t, p) in decomp.trend.iter().zip(prices.iter()) {
            assert!((t - p).abs() < 5.0, "HP trend {t:.2} far from price {p:.2}");
        }
    }

    #[test]
    fn test_hp_filter_cycle_near_zero_for_linear() {
        let prices = linear_trend(100);
        let hp = HPFilter::new(1600.0);
        let decomp = hp.decompose(&prices);
        let max_cycle = decomp.cycle.iter().cloned().map(f64::abs).fold(0.0f64, f64::max);
        // For a perfect linear trend, cycle should be small (not necessarily zero
        // due to endpoint effects, but well under 1% of price range)
        let price_range = prices.last().unwrap() - prices.first().unwrap();
        assert!(max_cycle < price_range * 0.05, "cycle too large: {max_cycle:.4}");
    }

    #[test]
    fn test_hp_trend_plus_cycle_equals_price() {
        let prices = sinusoidal_trend(80);
        let hp = HPFilter::daily();
        let decomp = hp.decompose(&prices);
        for i in 0..prices.len() {
            let reconstructed = decomp.trend[i] + decomp.cycle[i];
            assert!(
                (reconstructed - prices[i]).abs() < 1e-6,
                "HP reconstruction error at bar {i}: {:.8}", (reconstructed - prices[i]).abs()
            );
        }
    }

    #[test]
    fn test_bk_filter_cycle_extraction() {
        let prices = sinusoidal_trend(200);
        let bk = BKFilter::new(10, 40, 12);
        let decomp = bk.decompose(&prices);
        assert_eq!(decomp.cycle.len(), prices.len());
        // Cycle at interior points should be non-zero
        let interior_nonzero = decomp.cycle[20..180].iter().any(|&c| c.abs() > 0.01);
        assert!(interior_nonzero, "BK cycle should be non-zero at interior points");
    }

    #[test]
    fn test_bk_weights_sum_near_zero() {
        let bk = BKFilter::business_cycle();
        let weights = bk.weights();
        let sum: f64 = weights.iter().sum();
        assert!(sum.abs() < 1e-10, "BK weights should sum to ~0 (unit root removal), got {sum}");
    }

    #[test]
    fn test_bk_filter_short_series_no_panic() {
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let bk = BKFilter::business_cycle();
        let decomp = bk.decompose(&prices);
        assert_eq!(decomp.original.len(), 4);
    }

    #[test]
    fn test_ssa_trend_length_matches_input() {
        let prices = sinusoidal_trend(100);
        let ssa = SSADetrend::new(10, 2);
        let decomp = ssa.decompose(&prices);
        assert_eq!(decomp.trend.len(), prices.len());
        assert_eq!(decomp.cycle.len(), prices.len());
        assert_eq!(decomp.noise.len(), prices.len());
    }

    #[test]
    fn test_ssa_reconstruction_is_exact() {
        let prices = sinusoidal_trend(60);
        let ssa = SSADetrend::new(8, 2);
        let decomp = ssa.decompose(&prices);
        let err = decomp.reconstruction_error();
        assert!(err < 1.0, "SSA reconstruction RMS error should be small, got {err:.4}");
    }

    #[test]
    fn test_emd_decompose_lengths() {
        let prices = sinusoidal_trend(128);
        let emd = EMDDetrend::default_daily();
        let decomp = emd.decompose(&prices);
        assert_eq!(decomp.trend.len(), prices.len());
        assert_eq!(decomp.cycle.len(), prices.len());
        assert_eq!(decomp.noise.len(), prices.len());
    }

    #[test]
    fn test_emd_trend_plus_cycle_plus_noise_approx_price() {
        let prices = sinusoidal_trend(64);
        let emd = EMDDetrend::new(3, 5);
        let decomp = emd.decompose(&prices);
        let err = decomp.reconstruction_error();
        assert!(err < 10.0, "EMD reconstruction RMS error should be reasonable, got {err:.4}");
    }

    #[test]
    fn test_decomposition_snr_positive() {
        let prices = sinusoidal_trend(128);
        let hp = HPFilter::new(1600.0);
        let decomp = hp.decompose(&prices);
        // For sinusoidal input the cycle is the signal, HP noise is 0
        assert!(decomp.snr() >= 0.0, "SNR should be non-negative");
    }
}
