//! counterfactual-engine
//!
//! High-performance parameter sweeping and sensitivity analysis for the
//! Idea Automation Engine's Counterfactual Oracle.
//!
//! # Architecture
//!
//! ```text
//! lib.rs          — public API re-exports + core types
//! sampler.rs      — ParameterBounds, LHS, Sobol, NeighborhoodSampler
//! sensitivity.rs  — SobolAnalyzer, MorrisScreening, SensitivityReport
//! main.rs         — CLI: sample / sensitivity sub-commands
//! ```
//!
//! # Quick example
//!
//! ```no_run
//! use counterfactual_engine::{ParameterBounds, LatinHypercubeSampler, Sampler};
//!
//! let bounds = ParameterBounds::genome_defaults();
//! let mut sampler = LatinHypercubeSampler::new(bounds, Some(42));
//! let samples = sampler.sample(100);
//! println!("Generated {} samples", samples.len());
//! ```

pub mod sampler;
pub mod sensitivity;
pub mod causal_inference;
pub mod causal_graph;
pub mod scenario_analysis;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use sampler::{
    LatinHypercubeSampler, NeighborhoodSampler, ParameterBounds, ParameterSample,
    Sampler, SobolSampler,
};
pub use sensitivity::{
    MorrisResult, MorrisScreening, SensitivityReport, SobolAnalyzer, SobolIndices,
};
pub use causal_inference::{ATEEstimator, FeatureVec, LogisticRegression, PropensityScoreEstimator};
pub use causal_graph::{CausalEdge, CausalGraph, CausalNode, EdgeType, NodeType, PathAnalysis};
pub use scenario_analysis::{Bar, BaselineParams, ParameterGrid, ScenarioResult, ScenarioSpec, WhatIfEngine};

// ---------------------------------------------------------------------------
// Core result type
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};

/// A single simulation result as read from the DB or a JSON file.
///
/// Fields map directly to the `counterfactual_results` DB table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimResult {
    /// Parameters used for this run (full genome dict).
    pub params: std::collections::HashMap<String, f64>,
    /// Sharpe ratio.
    pub sharpe: f64,
    /// Maximum drawdown (positive fraction).
    pub max_dd: f64,
    /// Calmar ratio.
    pub calmar: f64,
    /// Total return over the backtest period.
    pub total_return: f64,
    /// Win rate (fraction of profitable trades).
    pub win_rate: f64,
    /// Number of trades executed.
    pub num_trades: i64,
    /// Composite improvement score vs baseline.
    pub improvement: f64,
}

impl SimResult {
    /// Composite improvement score:
    ///   0.4 × Δsharpe + 0.3 × Δcalmar − 0.3 × Δmax_dd
    pub fn improvement_vs(&self, baseline: &SimResult) -> f64 {
        let d_sharpe = self.sharpe - baseline.sharpe;
        let d_calmar = self.calmar - baseline.calmar;
        let d_maxdd = self.max_dd - baseline.max_dd;
        0.4 * d_sharpe + 0.3 * d_calmar - 0.3 * d_maxdd
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("DB error: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;

// ---------------------------------------------------------------------------
// Utility: load results from JSON file
// ---------------------------------------------------------------------------

use std::path::Path;

/// Load a `Vec<SimResult>` from a JSON array file.
pub fn load_results_from_file(path: &Path) -> Result<Vec<SimResult>> {
    let text = std::fs::read_to_string(path)?;
    let results: Vec<SimResult> = serde_json::from_str(&text)?;
    Ok(results)
}

/// Write a `Vec<ParameterSample>` to a JSON file.
pub fn write_samples_to_file(
    samples: &[ParameterSample],
    path: &Path,
) -> Result<()> {
    let text = serde_json::to_string_pretty(samples)?;
    std::fs::write(path, text)?;
    Ok(())
}

/// Write a `SensitivityReport` to a JSON file.
pub fn write_sensitivity_to_file(report: &SensitivityReport, path: &Path) -> Result<()> {
    let text = serde_json::to_string_pretty(report)?;
    std::fs::write(path, text)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_improvement_vs() {
        let baseline = SimResult {
            params: Default::default(),
            sharpe: 1.0,
            max_dd: 0.20,
            calmar: 2.0,
            total_return: 0.30,
            win_rate: 0.55,
            num_trades: 100,
            improvement: 0.0,
        };
        let variant = SimResult {
            sharpe: 1.5,   // +0.5
            max_dd: 0.15,  // -0.05 (better, contributes positively)
            calmar: 2.5,   // +0.5
            ..baseline.clone()
        };
        let score = variant.improvement_vs(&baseline);
        // 0.4*0.5 + 0.3*0.5 - 0.3*(-0.05) = 0.20 + 0.15 + 0.015 = 0.365
        assert!((score - 0.365).abs() < 1e-6, "score={}", score);
    }

    #[test]
    fn test_lhs_basic() {
        let bounds = ParameterBounds::genome_defaults();
        let mut sampler = LatinHypercubeSampler::new(bounds, Some(42));
        let samples = sampler.sample(50);
        assert_eq!(samples.len(), 50);
        // All values should be within bounds
        for s in &samples {
            for (name, val) in &s.params {
                let (lo, hi) = s.bounds_for(name).unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
                assert!(
                    *val >= lo && *val <= hi,
                    "param {name}={val} out of bounds [{lo},{hi}]"
                );
            }
        }
    }
}
