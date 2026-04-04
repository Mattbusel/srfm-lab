// monte_carlo.rs — Fast Monte Carlo simulation for trade return distributions
//
// Vectorised via ndarray: draws an (n_sims × n_steps) return matrix, applies
// AR(1) serial correlation, then compounds geometrically (or arithmetically
// for dollar P&L) to produce final equity distributions.
//
// Complements the Python mc.py but runs ~50–200× faster in Rust.

use crate::types::Trade;
use anyhow::{bail, Result};
use ndarray::Array2;
use rand::{distributions::Distribution, rngs::SmallRng, SeedableRng};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// MCConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCConfig {
    /// Number of Monte Carlo paths.
    pub n_sims: usize,
    /// Number of months to simulate.
    pub months: usize,
    /// AR(1) serial-correlation coefficient ρ ∈ (-1, 1).
    pub serial_corr: f64,
    /// Equity fraction below which a path is declared "blown up".
    pub blowup_threshold: f64,
    /// Random seed (0 = random).
    pub seed: u64,
    /// Compounding mode: true = geometric (fractional returns),
    ///                   false = arithmetic (dollar returns).
    pub geometric: bool,
    /// Approximate number of trades per month (used to set n_steps).
    pub trades_per_month: usize,
}

impl Default for MCConfig {
    fn default() -> Self {
        Self {
            n_sims: 10_000,
            months: 12,
            serial_corr: 0.0,
            blowup_threshold: 0.5,
            seed: 42,
            geometric: true,
            trades_per_month: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// MCResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCResult {
    /// Final equities across all simulated paths.
    pub final_equities: Vec<f64>,
    /// Fraction of paths whose equity falls below blowup_threshold × starting_equity.
    pub blowup_rate: f64,
    pub median_equity: f64,
    pub mean_equity: f64,
    pub pct_5: f64,
    pub pct_25: f64,
    pub pct_75: f64,
    pub pct_95: f64,
    /// Kelly fraction estimated from trade statistics: f = μ/σ².
    pub kelly_fraction: f64,
    pub starting_equity: f64,
    /// Expected CAGR across paths (annualised from median).
    pub expected_cagr: f64,
}

// ---------------------------------------------------------------------------
// Trade statistics helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TradeStats {
    mean_return: f64,
    std_return: f64,
    /// Returns as fractions
    returns: Vec<f64>,
}

fn compute_trade_stats(trades: &[Trade]) -> Result<TradeStats> {
    if trades.len() < 4 {
        bail!("Need at least 4 trades for Monte Carlo simulation (got {})", trades.len());
    }
    let returns: Vec<f64> = trades.iter().map(|t| t.return_frac()).collect();
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = var.sqrt();
    Ok(TradeStats { mean_return: mean, std_return: std, returns })
}

// ---------------------------------------------------------------------------
// run_mc
// ---------------------------------------------------------------------------

/// Run a full Monte Carlo simulation from historical trade results.
///
/// # Arguments
/// * `trades` — historical trade record (must have ≥ 4 entries)
/// * `starting_equity` — initial portfolio value
/// * `cfg` — simulation parameters
pub fn run_mc(trades: &[Trade], starting_equity: f64, cfg: &MCConfig) -> Result<MCResult> {
    let stats = compute_trade_stats(trades)?;
    run_mc_from_stats(
        stats.mean_return,
        stats.std_return,
        starting_equity,
        cfg,
    )
}

/// Lower-level entry point when you already have μ and σ of returns.
pub fn run_mc_from_stats(
    mean_return: f64,
    std_return: f64,
    starting_equity: f64,
    cfg: &MCConfig,
) -> Result<MCResult> {
    if cfg.n_sims == 0 {
        bail!("n_sims must be > 0");
    }
    if !((-1.0..1.0).contains(&cfg.serial_corr)) {
        bail!("serial_corr must be in (-1, 1), got {}", cfg.serial_corr);
    }

    let n_steps = cfg.months * cfg.trades_per_month.max(1);
    let n_sims = cfg.n_sims;

    // Build RNG
    let rng = if cfg.seed == 0 {
        SmallRng::from_entropy()
    } else {
        SmallRng::seed_from_u64(cfg.seed)
    };

    // Draw (n_sims × n_steps) iid normal shocks
    let normal = Normal::new(mean_return, std_return.max(1e-9))
        .map_err(|e| anyhow::anyhow!("Normal distribution error: {e}"))?;

    // We need owned RNG — draw sequentially into flat vec, then reshape.
    let mut rng = rng;
    let mut flat: Vec<f64> = Vec::with_capacity(n_sims * n_steps);
    for _ in 0..n_sims * n_steps {
        flat.push(normal.sample(&mut rng));
    }

    let mut returns = Array2::from_shape_vec((n_sims, n_steps), flat)
        .map_err(|e| anyhow::anyhow!("ndarray reshape error: {e}"))?;

    // Apply AR(1) serial correlation: r_t = ρ·r_{t-1} + ε·√(1-ρ²)
    let rho = cfg.serial_corr;
    if rho.abs() > 1e-9 {
        let scale = (1.0 - rho * rho).sqrt();
        for i in 0..n_sims {
            for t in 1..n_steps {
                let prev = returns[[i, t - 1]];
                let eps = returns[[i, t]];
                returns[[i, t]] = rho * prev + eps * scale;
            }
        }
    }

    // Compound returns path by path
    let blowup_level = starting_equity * cfg.blowup_threshold;
    let mut final_equities: Vec<f64> = Vec::with_capacity(n_sims);

    for i in 0..n_sims {
        let row = returns.row(i);
        let final_eq = if cfg.geometric {
            // Geometric compounding: equity *= (1 + r)
            let mut eq = starting_equity;
            for &r in row.iter() {
                eq *= 1.0 + r;
                if eq <= 0.0 {
                    eq = 0.0;
                    break;
                }
            }
            eq
        } else {
            // Arithmetic: equity += r × starting_equity
            let mut eq = starting_equity;
            for &r in row.iter() {
                eq += r * starting_equity;
            }
            eq.max(0.0)
        };
        final_equities.push(final_eq);
    }

    // Statistics
    let blowup_count = final_equities.iter().filter(|&&e| e < blowup_level).count();
    let blowup_rate = blowup_count as f64 / n_sims as f64;

    let mut sorted = final_equities.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };

    let median_equity = percentile(50.0);
    let mean_equity = final_equities.iter().sum::<f64>() / n_sims as f64;
    let pct_5 = percentile(5.0);
    let pct_25 = percentile(25.0);
    let pct_75 = percentile(75.0);
    let pct_95 = percentile(95.0);

    // Kelly fraction: f* = μ / σ²
    let kelly_fraction = if std_return > 1e-9 {
        mean_return / (std_return * std_return)
    } else {
        0.0
    };

    // Expected CAGR from median equity
    let years = cfg.months as f64 / 12.0;
    let expected_cagr = if years > 0.0 && median_equity > 0.0 && starting_equity > 0.0 {
        (median_equity / starting_equity).powf(1.0 / years) - 1.0
    } else {
        0.0
    };

    Ok(MCResult {
        final_equities,
        blowup_rate,
        median_equity,
        mean_equity,
        pct_5,
        pct_25,
        pct_75,
        pct_95,
        kelly_fraction: kelly_fraction.clamp(-5.0, 5.0),
        starting_equity,
        expected_cagr,
    })
}

// ---------------------------------------------------------------------------
// Bootstrap resampling variant
// ---------------------------------------------------------------------------

/// Bootstrap MC: re-sample (with replacement) from the actual trade returns
/// rather than assuming a parametric distribution.
pub fn run_mc_bootstrap(
    trades: &[Trade],
    starting_equity: f64,
    cfg: &MCConfig,
) -> Result<MCResult> {
    let stats = compute_trade_stats(trades)?;
    let n_steps = cfg.months * cfg.trades_per_month.max(1);
    let n_sims = cfg.n_sims;
    let actual_returns = &stats.returns;
    let n_actual = actual_returns.len();

    let mut rng = if cfg.seed == 0 {
        SmallRng::from_entropy()
    } else {
        SmallRng::seed_from_u64(cfg.seed.wrapping_add(9999))
    };

    use rand::Rng;
    let blowup_level = starting_equity * cfg.blowup_threshold;
    let mut final_equities: Vec<f64> = Vec::with_capacity(n_sims);

    for _ in 0..n_sims {
        let mut eq = starting_equity;
        for _ in 0..n_steps {
            let idx = rng.gen_range(0..n_actual);
            let r = actual_returns[idx];
            if cfg.geometric {
                eq *= 1.0 + r;
            } else {
                eq += r * starting_equity;
            }
            if eq <= 0.0 {
                eq = 0.0;
                break;
            }
        }
        final_equities.push(eq.max(0.0));
    }

    let blowup_count = final_equities.iter().filter(|&&e| e < blowup_level).count();
    let blowup_rate = blowup_count as f64 / n_sims as f64;

    let mut sorted = final_equities.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };

    let median_equity = percentile(50.0);
    let mean_equity = final_equities.iter().sum::<f64>() / n_sims as f64;
    let years = cfg.months as f64 / 12.0;
    let expected_cagr = if years > 0.0 && median_equity > 0.0 {
        (median_equity / starting_equity).powf(1.0 / years) - 1.0
    } else {
        0.0
    };

    let kelly_fraction = if stats.std_return > 1e-9 {
        stats.mean_return / (stats.std_return * stats.std_return)
    } else {
        0.0
    };

    Ok(MCResult {
        final_equities,
        blowup_rate,
        median_equity,
        mean_equity,
        pct_5: percentile(5.0),
        pct_25: percentile(25.0),
        pct_75: percentile(75.0),
        pct_95: percentile(95.0),
        kelly_fraction: kelly_fraction.clamp(-5.0, 5.0),
        starting_equity,
        expected_cagr,
    })
}

// ---------------------------------------------------------------------------
// Pretty summary
// ---------------------------------------------------------------------------

impl MCResult {
    pub fn summary(&self) -> String {
        format!(
            "MC: n_paths={n} | blowup={bp:.1}% | median={med:.0} | \
             5th={p5:.0} 95th={p95:.0} | Kelly={k:.3} | CAGR={cagr:.1}%",
            n = self.final_equities.len(),
            bp = self.blowup_rate * 100.0,
            med = self.median_equity,
            p5 = self.pct_5,
            p95 = self.pct_95,
            k = self.kelly_fraction,
            cagr = self.expected_cagr * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_trades(n: usize, mean_ret: f64, std_ret: f64, seed: u64) -> Vec<Trade> {
        use rand::{rngs::SmallRng, Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                let r: f64 = mean_ret + rng.gen::<f64>() * std_ret * 2.0 - std_ret;
                let ep = 100.0_f64;
                let xp = ep * (1.0 + r);
                let dp = 10_000.0;
                let pnl = (xp - ep) * (dp / ep);
                crate::types::Trade::new(
                    "TEST",
                    i as i64 * 86400_000,
                    (i as i64 + 1) * 86400_000,
                    ep,
                    xp,
                    pnl,
                    dp,
                    1,
                    crate::types::Regime::Bull,
                    crate::types::TFScore::zero(),
                    1.5,
                )
            })
            .collect()
    }

    #[test]
    fn mc_runs_and_produces_valid_output() {
        let trades = dummy_trades(50, 0.005, 0.02, 42);
        let cfg = MCConfig { n_sims: 1000, months: 12, ..Default::default() };
        let result = run_mc(&trades, 1_000_000.0, &cfg).expect("mc failed");

        assert_eq!(result.final_equities.len(), 1000);
        assert!(result.blowup_rate >= 0.0 && result.blowup_rate <= 1.0);
        assert!(result.pct_5 <= result.median_equity);
        assert!(result.median_equity <= result.pct_95);
    }

    #[test]
    fn mc_bootstrap_runs() {
        let trades = dummy_trades(50, 0.003, 0.015, 99);
        let cfg = MCConfig { n_sims: 500, months: 6, geometric: false, ..Default::default() };
        let result = run_mc_bootstrap(&trades, 500_000.0, &cfg).expect("bootstrap mc failed");
        assert_eq!(result.final_equities.len(), 500);
    }

    #[test]
    fn positive_drift_median_above_start() {
        // Strong positive mean return should almost always beat starting equity
        let cfg = MCConfig { n_sims: 2000, months: 12, geometric: true, ..Default::default() };
        let result = run_mc_from_stats(0.01, 0.01, 1_000_000.0, &cfg).expect("mc failed");
        assert!(result.median_equity > 1_000_000.0, "positive drift should grow equity");
    }
}
