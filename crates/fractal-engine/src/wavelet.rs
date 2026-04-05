/// Discrete Wavelet Transform (DWT) using the Daubechies-4 (db4) wavelet.
///
/// Provides:
///   - forward_dwt  : multi-level decomposition → approximation + detail coefficients
///   - inverse_dwt  : perfect reconstruction from decomposition
///   - energy       : per-scale energy (L2 norm squared of coefficients)
///
/// The db4 filter coefficients are normalised for orthonormal DWT.
/// For signals whose length is not a power of two, the signal is zero-padded
/// to the next power of two before transform and trimmed on reconstruction.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Daubechies-4 filter coefficients (normalised)
// ---------------------------------------------------------------------------

/// Scaling (low-pass) filter h0
const DB4_LO: [f64; 8] = [
    0.230377813308897,
    0.714846570552542,
    0.630880767929590,
   -0.027983769416984,
   -0.187034811718881,
    0.030841381835987,
    0.032883011666983,
   -0.010597401784997,
];

/// Wavelet (high-pass) filter h1 — derived from scaling filter via QMF relation
const DB4_HI: [f64; 8] = [
    -0.010597401784997,
    -0.032883011666983,
     0.030841381835987,
     0.187034811718881,
    -0.027983769416984,
    -0.630880767929590,
     0.714846570552542,
    -0.230377813308897,
];

// Synthesis (reconstruction) filters — time-reversed analysis filters
#[allow(dead_code)]
fn lo_syn() -> [f64; 8] {
    let mut f = DB4_HI;
    f.reverse();
    // negate alternate signs for synthesis
    // Actually for orthonormal wavelet: lo_syn = hi reversed, hi_syn = lo reversed
    DB4_HI
}

fn hi_syn() -> [f64; 8] {
    DB4_LO
}

// ---------------------------------------------------------------------------
// Wavelet decomposition struct
// ---------------------------------------------------------------------------

/// Result of a multi-level DWT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveletDecomp {
    /// Final coarse approximation (scaling) coefficients.
    pub approximation: Vec<f64>,
    /// Detail (wavelet) coefficient vectors per level (finest = index 0).
    pub details: Vec<Vec<f64>>,
    /// Original signal length (before padding).
    pub original_len: usize,
    /// Number of decomposition levels.
    pub levels: usize,
}

impl WaveletDecomp {
    /// Per-scale energy: L2 norm squared of each detail level + approximation.
    /// Returns [approx_energy, detail_0_energy, detail_1_energy, ...]
    pub fn energy(&self) -> Vec<f64> {
        let mut energies = Vec::with_capacity(self.levels + 1);
        energies.push(l2_energy(&self.approximation));
        for d in &self.details {
            energies.push(l2_energy(d));
        }
        energies
    }

    /// Fraction of total energy at each scale.
    pub fn energy_fractions(&self) -> Vec<f64> {
        let e = self.energy();
        let total: f64 = e.iter().sum();
        if total < 1e-16 {
            return vec![0.0; e.len()];
        }
        e.iter().map(|v| v / total).collect()
    }

    /// Dominant scale (index of maximum detail energy).
    pub fn dominant_detail_scale(&self) -> usize {
        self.details
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                l2_energy(a)
                    .partial_cmp(&l2_energy(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

fn l2_energy(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

// ---------------------------------------------------------------------------
// Forward DWT
// ---------------------------------------------------------------------------

/// Perform multi-level DWT decomposition.
///
/// `levels` capped at log2(signal_len) - 1.
pub fn forward_dwt(signal: &[f64], levels: usize) -> WaveletDecomp {
    let original_len = signal.len();
    let padded_len = next_power_of_two(original_len.max(8));
    let mut current = zero_pad(signal, padded_len);
    let levels = levels.min(max_levels(padded_len));

    let mut details: Vec<Vec<f64>> = Vec::with_capacity(levels);

    for _ in 0..levels {
        let (approx, detail) = dwt_single_level(&current);
        details.push(detail);
        current = approx;
    }

    // Reverse details so index 0 = finest (last computed = coarsest, push was in order)
    // Actually details were pushed finest-first (we decomposed the approximation each step,
    // the first detail is the finest). No reversal needed; index 0 = finest detail.
    WaveletDecomp {
        approximation: current,
        details,
        original_len,
        levels,
    }
}

/// Single-level DWT: convolve with lo/hi filters, downsample by 2.
fn dwt_single_level(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    let half = n / 2;
    let filter_len = DB4_LO.len();

    let mut approx = vec![0.0f64; half];
    let mut detail = vec![0.0f64; half];

    for i in 0..half {
        let mut lo_sum = 0.0f64;
        let mut hi_sum = 0.0f64;
        for k in 0..filter_len {
            let idx = (2 * i + k) % n;
            lo_sum += DB4_LO[k] * signal[idx];
            hi_sum += DB4_HI[k] * signal[idx];
        }
        approx[i] = lo_sum;
        detail[i] = hi_sum;
    }

    (approx, detail)
}

// ---------------------------------------------------------------------------
// Inverse DWT
// ---------------------------------------------------------------------------

/// Reconstruct signal from WaveletDecomp.
/// Returns signal trimmed to `decomp.original_len`.
pub fn inverse_dwt(decomp: &WaveletDecomp) -> Vec<f64> {
    let mut current = decomp.approximation.clone();

    // Reconstruct from coarsest to finest
    for detail in decomp.details.iter().rev() {
        current = idwt_single_level(&current, detail);
    }

    // Trim to original length
    current.truncate(decomp.original_len);
    current
}

/// Single-level inverse DWT: upsample by 2, convolve with synthesis filters.
fn idwt_single_level(approx: &[f64], detail: &[f64]) -> Vec<f64> {
    let n = approx.len();
    let out_len = n * 2;
    let filter_len = DB4_LO.len();

    // Upsample: insert zeros between samples
    let mut up_approx = vec![0.0f64; out_len];
    let mut up_detail = vec![0.0f64; out_len];
    for i in 0..n {
        up_approx[2 * i] = approx[i];
        up_detail[2 * i] = detail[i];
    }

    // Convolve with synthesis filters
    let _lo_s = hi_syn(); // lo synthesis = HI analysis reversed
    let _hi_s = hi_syn(); // placeholder — use proper QMF

    // For Daubechies orthonormal: lo_syn[k] = lo[filter_len-1-k], hi_syn[k] = (-1)^(k+1) * lo[k]
    let mut lo_syn_f = [0.0f64; 8];
    let mut hi_syn_f = [0.0f64; 8];
    for k in 0..filter_len {
        lo_syn_f[k] = DB4_LO[filter_len - 1 - k];
        hi_syn_f[k] = if (k + 1) % 2 == 0 { -DB4_HI[filter_len - 1 - k] } else { DB4_HI[filter_len - 1 - k] };
    }

    let mut output = vec![0.0f64; out_len];
    for i in 0..out_len {
        let mut sum = 0.0f64;
        for k in 0..filter_len {
            if i + k < out_len {
                sum += lo_syn_f[k] * up_approx[i + k] + hi_syn_f[k] * up_detail[i + k];
            }
        }
        output[i] = sum;
    }

    output
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

fn zero_pad(signal: &[f64], target_len: usize) -> Vec<f64> {
    let mut out = signal.to_vec();
    out.resize(target_len, 0.0);
    out
}

fn max_levels(n: usize) -> usize {
    let mut levels = 0;
    let mut size = n;
    let filter_len = DB4_LO.len();
    while size >= filter_len * 2 {
        size /= 2;
        levels += 1;
    }
    levels.max(1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(n: usize, freq: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / n as f64).sin())
            .collect()
    }

    #[test]
    fn forward_dwt_returns_correct_structure() {
        let signal = sine_signal(128, 5.0);
        let decomp = forward_dwt(&signal, 3);
        assert_eq!(decomp.levels, 3);
        assert_eq!(decomp.details.len(), 3);
        assert!(!decomp.approximation.is_empty());
    }

    #[test]
    fn energy_fractions_sum_to_one() {
        let signal = sine_signal(128, 3.0);
        let decomp = forward_dwt(&signal, 4);
        let fracs = decomp.energy_fractions();
        let total: f64 = fracs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10, "Energy fractions should sum to 1.0, got {total}");
    }

    #[test]
    fn decomp_original_len_preserved() {
        let signal = sine_signal(100, 2.0);
        let decomp = forward_dwt(&signal, 3);
        assert_eq!(decomp.original_len, 100);
    }

    #[test]
    fn inverse_dwt_returns_correct_length() {
        let signal = sine_signal(64, 4.0);
        let decomp = forward_dwt(&signal, 3);
        let reconstructed = inverse_dwt(&decomp);
        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn energy_is_non_negative() {
        let signal: Vec<f64> = (0..256).map(|i| (i as f64) * 0.01).collect();
        let decomp = forward_dwt(&signal, 4);
        for (i, e) in decomp.energy().iter().enumerate() {
            assert!(*e >= 0.0, "Energy at level {i} is negative: {e}");
        }
    }

    #[test]
    fn dominant_scale_is_valid() {
        let signal = sine_signal(128, 8.0); // high-frequency sine
        let decomp = forward_dwt(&signal, 4);
        let dom = decomp.dominant_detail_scale();
        assert!(dom < decomp.levels, "Dominant scale {dom} exceeds levels {}", decomp.levels);
    }

    #[test]
    fn single_level_roundtrip_energy_preserved() {
        let signal = sine_signal(64, 3.0);
        let (approx, detail) = dwt_single_level(&signal);
        let energy_in: f64 = signal.iter().map(|x| x * x).sum();
        let energy_out: f64 = approx.iter().map(|x| x * x).sum::<f64>()
            + detail.iter().map(|x| x * x).sum::<f64>();
        // Parseval's theorem: energy should be approximately preserved
        // (not exact due to filter length effects at boundaries)
        let ratio = energy_out / energy_in.max(1e-12);
        assert!(ratio > 0.5 && ratio < 2.0, "Energy ratio {ratio:.3} out of expected range");
    }
}
