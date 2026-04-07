// regime-analytics: Regime detection and analysis library.
// Provides HMM, Markov Switching, structural break detection, and regime-conditional statistics.

pub mod detection;
pub mod analysis;
pub mod hmm_regime;
pub mod regime_transition_model;
pub mod conditional_performance;

pub use detection::{
    hmm::{HiddenMarkovModel, HmmParams, HmmState, ViterbiResult, BaumWelchResult},
    markov_switching::{MarkovSwitchingModel, MsParams, FilteredProbabilities},
    structural_breaks::{CusumTest, ZivotAndrews, BaiPerron, BreakpointResult},
};
pub use analysis::{
    regime_stats::{RegimeStats, RegimeConditionalStats, PersistenceMatrix},
    regime_backtest::{RegimeBacktest, RegimePnl, RegimeTransitionCost},
};

/// Identify regime index from probability vector (argmax).
pub fn argmax(probs: &[f64]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Normalize a probability vector to sum to 1.
pub fn normalize_probs(v: &[f64]) -> Vec<f64> {
    let sum: f64 = v.iter().sum();
    if sum < 1e-300 {
        let n = v.len();
        return vec![1.0 / n as f64; n];
    }
    v.iter().map(|x| x / sum).collect()
}

/// Log-sum-exp trick for numerical stability.
pub fn log_sum_exp(log_vals: &[f64]) -> f64 {
    let max = log_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    max + log_vals.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}

/// Gaussian log-density: ln N(x | mu, sigma^2).
pub fn gaussian_log_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let sigma = sigma.max(1e-10);
    -0.5 * ((x - mu) / sigma).powi(2) - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// Gamma log-density: ln Gamma(x | shape, rate).
pub fn gamma_log_pdf(x: f64, shape: f64, rate: f64) -> f64 {
    if x <= 0.0 || shape <= 0.0 || rate <= 0.0 {
        return f64::NEG_INFINITY;
    }
    shape * rate.ln() - log_gamma(shape) + (shape - 1.0) * x.ln() - rate * x
}

/// Stirling approximation for ln(Gamma(a)).
pub fn log_gamma(a: f64) -> f64 {
    if a <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos approximation.
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if a < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * a).sin().ln()
            - log_gamma(1.0 - a);
    }
    let z = a - 1.0;
    let mut x = c[0];
    for i in 1..=(g as usize + 1) {
        x += c[i] / (z + i as f64);
    }
    let t = z + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln()
        + (z + 0.5) * t.ln()
        - t
        + x.ln()
}
