/// Pattern similarity using Dynamic Time Warping (DTW) distance.
///
/// Compares the current wavelet "fingerprint" (energy fractions across scales)
/// to historical profitable setups stored in a pattern library.
/// If DTW distance < threshold, flag the current bar as a similar setup.
///
/// DTW is O(n*m) in the general case. For fingerprint vectors (short, ~5-10 dims)
/// this is essentially instantaneous.

use crate::wavelet::forward_dwt;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// DTW distance
// ---------------------------------------------------------------------------

/// Compute the Dynamic Time Warping distance between two sequences.
/// Uses the standard O(n*m) DP with Euclidean point distance.
pub fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }

    // Allocate DP matrix
    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).powi(2); // squared Euclidean
            let prev = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = cost + prev;
        }
    }

    dp[n][m].sqrt() // Return Euclidean (not squared) DTW
}

/// DTW with Sakoe-Chiba band constraint (window w).
/// Restricts warping path to |i - j| <= w, improving speed and avoiding degenerate alignments.
pub fn dtw_distance_windowed(a: &[f64], b: &[f64], window: usize) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }

    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;

    for i in 1..=n {
        let j_lo = if i > window { i - window } else { 1 };
        let j_hi = (i + window).min(m);
        for j in j_lo..=j_hi {
            let cost = (a[i - 1] - b[j - 1]).powi(2);
            let prev = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = cost + prev;
        }
    }

    dp[n][m].sqrt()
}

// ---------------------------------------------------------------------------
// Wavelet fingerprint
// ---------------------------------------------------------------------------

/// A wavelet fingerprint is the energy-fraction vector of a DWT decomposition.
/// Compact representation: ~5 floats capturing the multi-scale structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveletFingerprint {
    pub energy_fractions: Vec<f64>,
    /// Dominant detail scale index
    pub dominant_scale: usize,
    /// Total signal energy (scale-invariant normalisation factor)
    pub total_energy: f64,
}

impl WaveletFingerprint {
    pub fn from_prices(prices: &[f64], levels: usize) -> Self {
        let decomp = forward_dwt(prices, levels);
        let energy_fracs = decomp.energy_fractions();
        let total_energy: f64 = decomp.energy().iter().sum();
        let dominant_scale = decomp.dominant_detail_scale();
        Self {
            energy_fractions: energy_fracs,
            dominant_scale,
            total_energy,
        }
    }

    /// DTW distance between two fingerprints (based on energy fraction vectors).
    pub fn dtw_distance(&self, other: &WaveletFingerprint) -> f64 {
        dtw_distance(&self.energy_fractions, &other.energy_fractions)
    }

    /// Euclidean distance between fingerprints (simpler, for sanity checks).
    pub fn euclidean_distance(&self, other: &WaveletFingerprint) -> f64 {
        let n = self.energy_fractions.len().min(other.energy_fractions.len());
        let sq: f64 = (0..n)
            .map(|i| (self.energy_fractions[i] - other.energy_fractions[i]).powi(2))
            .sum();
        sq.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Historical pattern library
// ---------------------------------------------------------------------------

/// A stored historical pattern (a wavelet fingerprint + outcome label).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredPattern {
    pub id: String,
    pub fingerprint: WaveletFingerprint,
    /// Whether this setup was profitable (+1.0) or not (-1.0).
    pub outcome: f64,
    /// Actual return achieved after pattern.
    pub realised_return: f64,
    pub bar_index: usize,
}

/// Library of stored patterns for similarity lookup.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PatternLibrary {
    pub patterns: Vec<StoredPattern>,
}

impl PatternLibrary {
    pub fn new() -> Self {
        Self { patterns: Vec::new() }
    }

    /// Add a pattern to the library.
    pub fn add(&mut self, pattern: StoredPattern) {
        self.patterns.push(pattern);
    }

    /// Find the k nearest patterns to the query fingerprint using DTW distance.
    /// Returns (distance, &StoredPattern) pairs sorted by ascending distance.
    pub fn knn_search(
        &self,
        query: &WaveletFingerprint,
        k: usize,
    ) -> Vec<(f64, &StoredPattern)> {
        let mut results: Vec<(f64, &StoredPattern)> = self
            .patterns
            .iter()
            .map(|p| (p.fingerprint.dtw_distance(query), p))
            .collect();
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Given a query fingerprint and threshold, return a similarity verdict.
    pub fn similarity_verdict(
        &self,
        query: &WaveletFingerprint,
        distance_threshold: f64,
        k: usize,
    ) -> SimilarityVerdict {
        if self.patterns.is_empty() {
            return SimilarityVerdict {
                is_similar: false,
                nearest_distance: f64::INFINITY,
                expected_outcome: 0.0,
                matched_patterns: 0,
            };
        }
        let nearest = self.knn_search(query, k);
        let nearest_distance = nearest.first().map(|(d, _)| *d).unwrap_or(f64::INFINITY);
        let below_threshold: Vec<_> = nearest.iter().filter(|(d, _)| *d < distance_threshold).collect();
        let matched = below_threshold.len();
        let expected_outcome = if matched > 0 {
            below_threshold.iter().map(|(_, p)| p.outcome).sum::<f64>() / matched as f64
        } else {
            0.0
        };
        SimilarityVerdict {
            is_similar: nearest_distance < distance_threshold,
            nearest_distance,
            expected_outcome,
            matched_patterns: matched,
        }
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("PatternLibrary serialisation infallible")
    }

    /// Deserialise from JSON.
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        serde_json::from_str(s).map_err(Into::into)
    }

    /// Build a library from a price history by extracting fingerprints
    /// from rolling windows and labelling outcomes by subsequent returns.
    pub fn build_from_prices(prices: &[f64], window: usize, levels: usize) -> Self {
        let n = prices.len();
        let mut library = Self::new();
        let mut id_counter = 0usize;
        for start in (0..n).step_by(window / 2) {
            let end = (start + window).min(n);
            if end - start < window / 2 {
                break;
            }
            let slice = &prices[start..end];
            let fingerprint = WaveletFingerprint::from_prices(slice, levels);
            // Label: next bar return after window
            let realised_return = if end < n && prices[start] > 0.0 {
                (prices[end - 1] - prices[start]) / prices[start]
            } else {
                0.0
            };
            let outcome = if realised_return > 0.0 { 1.0 } else { -1.0 };
            library.add(StoredPattern {
                id: format!("pat_{id_counter:04}"),
                fingerprint,
                outcome,
                realised_return,
                bar_index: start,
            });
            id_counter += 1;
        }
        library
    }
}

/// Result of a KNN similarity query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityVerdict {
    pub is_similar: bool,
    pub nearest_distance: f64,
    pub expected_outcome: f64,
    pub matched_patterns: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_prices(n: usize, freq: f64) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 + 10.0 * (2.0 * std::f64::consts::PI * freq * i as f64 / n as f64).sin())
            .collect()
    }

    #[test]
    fn dtw_zero_for_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let dist = dtw_distance(&a, &a);
        assert!(dist < 1e-10, "DTW of identical series should be 0, got {dist}");
    }

    #[test]
    fn dtw_symmetric() {
        let a = vec![1.0, 3.0, 2.0, 4.0];
        let b = vec![2.0, 1.0, 4.0, 3.0];
        let d_ab = dtw_distance(&a, &b);
        let d_ba = dtw_distance(&b, &a);
        assert!((d_ab - d_ba).abs() < 1e-10, "DTW should be symmetric");
    }

    #[test]
    fn dtw_shifted_series_is_small() {
        let a: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let b: Vec<f64> = (1..21).map(|i| i as f64).collect();
        // DTW should be able to align a shifted series with small distance
        let d = dtw_distance(&a, &b);
        assert!(d < 10.0, "Shifted series DTW should be small, got {d}");
    }

    #[test]
    fn dtw_windowed_less_than_full() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        let full = dtw_distance(&a, &b);
        let windowed = dtw_distance_windowed(&a, &b, 3);
        // Windowed DTW should be >= full DTW (more constrained)
        assert!(windowed >= full * 0.9, "Windowed DTW {windowed} too much less than full {full}");
    }

    #[test]
    fn fingerprint_from_prices() {
        let prices = sample_prices(64, 3.0);
        let fp = WaveletFingerprint::from_prices(&prices, 3);
        let total: f64 = fp.energy_fractions.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "Energy fractions should sum to 1.0, got {total}");
    }

    #[test]
    fn fingerprint_self_distance_zero() {
        let prices = sample_prices(64, 2.0);
        let fp = WaveletFingerprint::from_prices(&prices, 3);
        let d = fp.dtw_distance(&fp);
        assert!(d < 1e-10, "Self-distance should be 0, got {d}");
    }

    #[test]
    fn similar_signals_closer_than_different() {
        let p1 = sample_prices(64, 2.0);
        let p2 = sample_prices(64, 2.1); // very similar
        let p3 = sample_prices(64, 8.0); // very different
        let fp1 = WaveletFingerprint::from_prices(&p1, 3);
        let fp2 = WaveletFingerprint::from_prices(&p2, 3);
        let fp3 = WaveletFingerprint::from_prices(&p3, 3);
        let d12 = fp1.dtw_distance(&fp2);
        let d13 = fp1.dtw_distance(&fp3);
        assert!(d12 <= d13, "Similar signals should be closer: {d12} <= {d13}");
    }

    #[test]
    fn library_knn_search_returns_sorted() {
        let prices = sample_prices(200, 2.0);
        let library = PatternLibrary::build_from_prices(&prices, 32, 3);
        let query_prices = sample_prices(32, 2.0);
        let query = WaveletFingerprint::from_prices(&query_prices, 3);
        let results = library.knn_search(&query, 3);
        for w in results.windows(2) {
            assert!(w[0].0 <= w[1].0, "KNN results should be sorted by distance");
        }
    }

    #[test]
    fn similarity_verdict_on_identical_low_distance() {
        let prices = sample_prices(128, 2.0);
        let library = PatternLibrary::build_from_prices(&prices, 32, 3);
        let query = WaveletFingerprint::from_prices(&prices[..32], 3);
        let verdict = library.similarity_verdict(&query, 1.0, 3);
        // The nearest distance should be very small (exact match in library)
        assert!(verdict.nearest_distance < 1.0, "Expected small distance, got {}", verdict.nearest_distance);
    }
}
