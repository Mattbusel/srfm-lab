/// Multi-objective fitness vector and the evaluator that fills it.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::genome::Genome;

// ---------------------------------------------------------------------------
// FitnessVec
// ---------------------------------------------------------------------------

/// All fitness metrics produced by a single backtest run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessVec {
    /// Annualised Sharpe ratio (higher is better).
    pub sharpe: f64,
    /// Maximum drawdown expressed as a positive fraction, e.g. 0.15 = 15 % (lower is better).
    pub max_dd: f64,
    /// Calmar ratio = annualised return / max_dd (higher is better).
    pub calmar: f64,
    /// Fraction of trades that are profitable (higher is better).
    pub win_rate: f64,
    /// Gross profit / gross loss (higher is better).
    pub profit_factor: f64,
    /// Number of completed round-trip trades.
    pub n_trades: u32,
    /// IS Sharpe − OOS Sharpe: measures overfitting (lower / closer to 0 is better).
    pub is_oos_spread: f64,
}

impl FitnessVec {
    /// A sentinel value used when evaluation fails.
    pub fn worst() -> Self {
        Self {
            sharpe: f64::NEG_INFINITY,
            max_dd: 1.0,
            calmar: f64::NEG_INFINITY,
            win_rate: 0.0,
            profit_factor: 0.0,
            n_trades: 0,
            is_oos_spread: f64::INFINITY,
        }
    }

    /// Returns `true` if the fitness values look plausible (not a sentinel or NaN-contaminated result).
    pub fn is_valid(&self) -> bool {
        self.sharpe.is_finite()
            && self.calmar.is_finite()
            && self.max_dd.is_finite()
            && self.max_dd >= 0.0
            && self.win_rate >= 0.0
            && self.win_rate <= 1.0
            && self.profit_factor >= 0.0
            && self.is_oos_spread.is_finite()
    }
}

impl std::fmt::Display for FitnessVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sharpe={:.3} calmar={:.3} dd={:.1}% wr={:.1}% pf={:.2} n={} spread={:.3}",
            self.sharpe,
            self.calmar,
            self.max_dd * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.n_trades,
            self.is_oos_spread,
        )
    }
}

// ---------------------------------------------------------------------------
// FitnessEvaluator
// ---------------------------------------------------------------------------

/// Configuration for how the backtest subprocess is launched.
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Absolute path to the srfm-lab root directory.
    pub lab_path: String,
    /// Maximum wall-clock time allowed per evaluation.
    pub timeout: Duration,
    /// If `true`, return a synthetic (random) fitness value instead of spawning a process.
    /// Useful for testing the GA mechanics without a working Python environment.
    pub dry_run: bool,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            lab_path: ".".to_string(),
            timeout: Duration::from_secs(120),
            dry_run: false,
        }
    }
}

/// Thread-safe evaluator: converts a `Genome` → `FitnessVec` by calling the
/// Python backtest as a subprocess.
///
/// The Python script is expected at `<lab_path>/tools/crypto_backtest_mc.py`
/// and must accept:
///   `--genome-config <json_path> --output-json <result_path>`
///
/// The result JSON must match the shape of `FitnessVec`.
#[derive(Debug, Clone)]
pub struct FitnessEvaluator {
    pub config: EvaluatorConfig,
}

impl FitnessEvaluator {
    pub fn new(config: EvaluatorConfig) -> Self {
        Self { config }
    }

    /// Evaluate a single genome.  Returns `FitnessVec::worst()` on any error so
    /// that the GA can continue even when some evaluations fail.
    pub fn evaluate(&self, genome: &Genome) -> FitnessVec {
        if self.config.dry_run {
            return self.synthetic_fitness(genome);
        }

        match self.evaluate_inner(genome) {
            Ok(fv) => fv,
            Err(e) => {
                eprintln!("[FitnessEvaluator] evaluation failed for {}: {:#}", genome.id, e);
                FitnessVec::worst()
            }
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn evaluate_inner(&self, genome: &Genome) -> Result<FitnessVec> {
        // Write the genome config to a temporary file.
        let tmp_dir = std::env::temp_dir();
        let cfg_path = tmp_dir.join(format!("genome_{}.json", genome.id));
        let out_path = tmp_dir.join(format!("fitness_{}.json", genome.id));

        std::fs::write(&cfg_path, genome.to_json())
            .with_context(|| format!("writing genome config to {:?}", cfg_path))?;

        let script = Path::new(&self.config.lab_path)
            .join("tools")
            .join("crypto_backtest_mc.py");

        // Spawn the subprocess.
        let mut child = Command::new("python")
            .arg(&script)
            .arg("--genome-config")
            .arg(&cfg_path)
            .arg("--output-json")
            .arg(&out_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawning python backtest subprocess")?;

        // Poll with a timeout.
        let deadline = std::time::Instant::now() + self.config.timeout;
        loop {
            match child.try_wait().context("polling subprocess")? {
                Some(status) => {
                    if !status.success() {
                        // Try to capture stderr for diagnostics.
                        let stderr = child
                            .stderr
                            .take()
                            .and_then(|mut s| {
                                let mut buf = String::new();
                                use std::io::Read;
                                s.read_to_string(&mut buf).ok();
                                Some(buf)
                            })
                            .unwrap_or_default();
                        return Err(anyhow!(
                            "backtest exited with {} — stderr: {}",
                            status,
                            stderr.trim()
                        ));
                    }
                    break;
                }
                None => {
                    if std::time::Instant::now() >= deadline {
                        let _ = child.kill();
                        return Err(anyhow!(
                            "backtest timed out after {:?}",
                            self.config.timeout
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(250));
                }
            }
        }

        // Parse result JSON.
        let result_str = std::fs::read_to_string(&out_path)
            .with_context(|| format!("reading output JSON from {:?}", out_path))?;

        let fv: FitnessVec =
            serde_json::from_str(&result_str).context("deserialising FitnessVec from output JSON")?;

        // Clean up temporaries.
        let _ = std::fs::remove_file(&cfg_path);
        let _ = std::fs::remove_file(&out_path);

        Ok(fv)
    }

    /// Deterministic synthetic fitness derived from parameter values so that
    /// tests and dry-runs have reproducible ordering.
    fn synthetic_fitness(&self, genome: &Genome) -> FitnessVec {
        // Use a simple hash of the parameters as a seed for pseudo-randomness.
        let sum: f64 = genome.parameters.iter().sum();
        let product: f64 = genome
            .parameters
            .iter()
            .fold(1.0_f64, |acc, x| acc + x * 0.1);

        // Map into plausible ranges.
        let sharpe = (sum.sin() * 2.0).clamp(-1.0, 3.5);
        let max_dd = ((product.cos().abs()) * 0.3 + 0.02).clamp(0.02, 0.50);
        let calmar = if max_dd > 0.0 { sharpe / max_dd } else { 0.0 };
        let win_rate = ((sum * 0.3).sin() * 0.2 + 0.55).clamp(0.30, 0.75);
        let profit_factor = (win_rate * 2.5).clamp(0.8, 4.0);
        let n_trades = ((sum.abs() * 10.0) as u32).max(5).min(500);
        let is_oos_spread = (sum * 0.05).abs().clamp(0.0, 0.5);

        FitnessVec {
            sharpe,
            max_dd,
            calmar,
            win_rate,
            profit_factor,
            n_trades,
            is_oos_spread,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn worst_fitness_is_invalid() {
        assert!(!FitnessVec::worst().is_valid());
    }

    #[test]
    fn synthetic_fitness_is_valid() {
        let mut rng = SmallRng::seed_from_u64(99);
        let g = Genome::new_random(&mut rng);
        let eval = FitnessEvaluator::new(EvaluatorConfig {
            dry_run: true,
            ..Default::default()
        });
        let fv = eval.evaluate(&g);
        assert!(fv.is_valid(), "synthetic fitness should be valid: {}", fv);
    }
}
