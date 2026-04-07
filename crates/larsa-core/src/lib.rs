pub mod multi_timeframe;
pub mod signal_quality;
pub mod parameter_sensitivity;
pub mod live_performance;
pub mod regime_filter;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand::distributions::Uniform;

/// Core SRFM physics simulation in Rust.
/// Called from Python as: import larsa_core
///
/// All functions accept Vec<f64> (Python list or numpy array converted)
/// and return Vec<f64> results.

// ── Regime constants ──────────────────────────────────────────────────────────
const REGIME_SIDEWAYS: i32 = 0;
const REGIME_BULL: i32 = 1;
const REGIME_BEAR: i32 = 2;
const REGIME_HIGH_VOL: i32 = 3;

// ── Default simulation params (used by sensitivity_sweep) ────────────────────
const DEF_CF: f64 = 0.02;
const DEF_BH_FORM: f64 = 1.5;
const DEF_BH_DECAY: f64 = 0.95;
const DEF_BH_COLLAPSE: f64 = 1.0;
const DEF_CTL_REQ: i32 = 5;

// ─────────────────────────────────────────────────────────────────────────────
// Existing functions (preserved exactly)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute beta series: |Δclose/close| / cf
#[pyfunction]
fn beta_series(closes: Vec<f64>, cf: f64) -> PyResult<Vec<f64>> {
    if closes.len() < 2 {
        return Ok(vec![]);
    }
    let mut betas = Vec::with_capacity(closes.len() - 1);
    for i in 1..closes.len() {
        let b = (closes[i] - closes[i - 1]).abs() / (closes[i - 1] * cf + 1e-12);
        betas.push(b);
    }
    Ok(betas)
}

/// Compute BH mass series and active flags.
/// Returns (masses: Vec<f64>, active: Vec<bool>, ctl: Vec<i32>)
#[pyfunction]
fn bh_mass_series(
    closes: Vec<f64>,
    cf: f64,
    bh_form: f64,
    bh_decay: f64,
    _bh_collapse: f64,
    ctl_req: i32,
) -> PyResult<(Vec<f64>, Vec<bool>, Vec<i32>)> {
    let n = closes.len();
    let mut masses = vec![0.0f64; n];
    let mut active = vec![false; n];
    let mut ctls = vec![0i32; n];

    let mut mass = 0.0f64;
    let mut ctl = 0i32;
    let mut bh_is_active = false;

    for i in 1..n {
        let beta = (closes[i] - closes[i - 1]).abs() / (closes[i - 1] * cf + 1e-12);

        if beta < 1.0 {
            // TIMELIKE: mass accretes
            mass = mass * 0.97 + 0.03 * 1.0;
            ctl += 1;
        } else {
            // SPACELIKE: mass decays
            mass *= bh_decay;
            ctl = 0;
        }

        // BH formation
        if !bh_is_active && mass >= bh_form && ctl >= ctl_req {
            bh_is_active = true;
        }
        // BH collapse
        if bh_is_active && mass < _bh_collapse {
            bh_is_active = false;
        }

        masses[i] = mass;
        active[i] = bh_is_active;
        ctls[i] = ctl;
    }

    Ok((masses, active, ctls))
}

/// Full SRFM simulation: returns equity curve starting at 1.0
/// Returns (equity: Vec<f64>, positions: Vec<f64>, trades: i32)
#[pyfunction]
fn simulate(
    closes: Vec<f64>,
    cf: f64,
    bh_form: f64,
    bh_decay: f64,
    bh_collapse: f64,
    max_lev: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, i32)> {
    let n = closes.len();
    if n < 10 {
        return Err(PyValueError::new_err("Need at least 10 bars"));
    }

    let mut equity = vec![1.0f64; n];
    let mut positions = vec![0.0f64; n];
    let mut mass = 0.0f64;
    let mut ctl = 0i32;
    let mut bh_active = false;
    let mut last_pos = 0.0f64;
    let mut trades = 0i32;

    // Simple 20-bar EMA for trend direction
    let mut ema20 = closes[0];
    let alpha20 = 2.0 / 21.0;

    for i in 1..n {
        // Update EMA
        ema20 = ema20 * (1.0 - alpha20) + closes[i] * alpha20;

        // SRFM physics
        let beta = (closes[i] - closes[i - 1]).abs() / (closes[i - 1] * cf + 1e-12);
        if beta < 1.0 {
            mass = mass * 0.97 + 0.03;
            ctl += 1;
        } else {
            mass *= bh_decay;
            ctl = 0;
        }

        if !bh_active && mass >= bh_form && ctl >= 5 {
            bh_active = true;
        }
        if bh_active && mass < bh_collapse {
            bh_active = false;
        }

        // Position sizing
        let direction = if closes[i] > ema20 { 1.0 } else { -1.0 };
        let target = if bh_active {
            max_lev * direction
        } else if ctl >= 3 {
            max_lev * 0.5 * direction
        } else {
            0.0
        };

        if i >= 50 {
            // warmup
            positions[i] = target;
            if (target - last_pos).abs() > 0.01 {
                trades += 1;
                last_pos = target;
            }
        }

        // Portfolio update
        let ret = closes[i] / closes[i - 1] - 1.0;
        equity[i] = equity[i - 1] * (1.0 + positions[i - 1] * ret);
    }

    Ok((equity, positions, trades))
}

/// Compute Sharpe ratio from equity curve (annualized, hourly bars)
#[pyfunction]
fn sharpe(equity: Vec<f64>) -> f64 {
    if equity.len() < 2 {
        return 0.0;
    }
    let returns: Vec<f64> = equity.windows(2).map(|w| w[1] / w[0] - 1.0).collect();
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < 1e-10 {
        return 0.0;
    }
    mean / std * (252.0 * 24.0_f64).sqrt()
}

/// Compute max drawdown from equity curve
#[pyfunction]
fn max_drawdown(equity: Vec<f64>) -> f64 {
    let mut peak = equity[0];
    let mut max_dd = 0.0f64;
    for &v in &equity {
        if v > peak {
            peak = v;
        }
        let dd = (peak - v) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Parameter sweep (sequential)
/// Returns list of (cf, max_lev, sharpe, return_pct, max_dd) tuples
#[pyfunction]
fn sweep(
    closes: Vec<f64>,
    cf_values: Vec<f64>,
    lev_values: Vec<f64>,
) -> PyResult<Vec<(f64, f64, f64, f64, f64)>> {
    let mut results = Vec::new();

    for &cf in &cf_values {
        for &lev in &lev_values {
            let (equity, _, _) = simulate(closes.clone(), cf, 1.5, 0.95, 1.0, lev)?;
            let sh = sharpe(equity.clone());
            let ret_pct = (equity.last().unwrap_or(&1.0) - 1.0) * 100.0;
            let dd = max_drawdown(equity) * 100.0;
            results.push((cf, lev, sh, ret_pct, dd));
        }
    }

    Ok(results)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute max drawdown from an equity slice (non-pyfunction version).
fn max_dd_internal(equity: &[f64]) -> f64 {
    let mut peak = equity[0];
    let mut max_dd = 0.0f64;
    for &v in equity {
        if v > peak {
            peak = v;
        }
        let dd = if peak > 0.0 { (peak - v) / peak } else { 0.0 };
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Sharpe approximation (daily bars): mean_ret / std_ret * sqrt(252)
fn sharpe_approx(equity: &[f64]) -> f64 {
    if equity.len() < 2 {
        return 0.0;
    }
    let rets: Vec<f64> = equity.windows(2).map(|w| w[1] / w[0] - 1.0).collect();
    let n = rets.len() as f64;
    let mean = rets.iter().sum::<f64>() / n;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    mean / std * 252.0_f64.sqrt()
}

/// Core BH physics engine — shared between full_backtest and sensitivity_sweep.
/// Returns (masses, active, ctls, betas) aligned to closes (index 0 unused for betas).
fn run_bh_physics(
    closes: &[f64],
    cf: f64,
    bh_form: f64,
    bh_decay: f64,
    bh_collapse: f64,
    ctl_req: i32,
) -> (Vec<f64>, Vec<bool>, Vec<i32>, Vec<f64>) {
    let n = closes.len();
    let mut masses = vec![0.0f64; n];
    let mut active = vec![false; n];
    let mut ctls = vec![0i32; n];
    let mut betas = vec![0.0f64; n];

    let mut mass = 0.0f64;
    let mut ctl = 0i32;
    let mut bh_is_active = false;

    for i in 1..n {
        let beta = (closes[i] - closes[i - 1]).abs() / (closes[i - 1] * cf + 1e-12);
        betas[i] = beta;

        if beta < 1.0 {
            mass = mass * 0.97 + 0.03;
            ctl += 1;
        } else {
            mass *= bh_decay;
            ctl = 0;
        }

        if !bh_is_active && mass >= bh_form && ctl >= ctl_req {
            bh_is_active = true;
        }
        if bh_is_active && mass < bh_collapse {
            bh_is_active = false;
        }

        masses[i] = mass;
        active[i] = bh_is_active;
        ctls[i] = ctl;
    }

    (masses, active, ctls, betas)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. full_backtest
// ─────────────────────────────────────────────────────────────────────────────

/// A single completed trade.
struct TradeRecord {
    entry_bar: usize,
    exit_bar: usize,
    entry_price: f64,
    exit_price: f64,
    pnl_frac: f64,
    hold_bars: usize,
    mfe_frac: f64,
    mae_frac: f64,
    regime_at_entry: i32,
    bh_mass_at_entry: f64,
    tf_score: i32,
}

impl TradeRecord {
    fn into_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("entry_bar", self.entry_bar)?;
        d.set_item("exit_bar", self.exit_bar)?;
        d.set_item("entry_price", self.entry_price)?;
        d.set_item("exit_price", self.exit_price)?;
        d.set_item("pnl_frac", self.pnl_frac)?;
        d.set_item("hold_bars", self.hold_bars)?;
        d.set_item("mfe_frac", self.mfe_frac)?;
        d.set_item("mae_frac", self.mae_frac)?;
        d.set_item("regime_at_entry", self.regime_at_entry)?;
        d.set_item("bh_mass_at_entry", self.bh_mass_at_entry)?;
        d.set_item("tf_score", self.tf_score)?;
        Ok(d)
    }
}

/// Full backtest returning rich analytics.
/// Returns a Python dict with keys: equity_curve, positions, bh_masses, bh_active,
/// ctl_series, regime, trades.
#[pyfunction]
#[pyo3(signature = (
    closes, highs, lows, cf, bh_form, bh_decay, bh_collapse,
    ctl_req, long_only = false
))]
fn full_backtest(
    py: Python<'_>,
    closes: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    cf: f64,
    bh_form: f64,
    bh_decay: f64,
    bh_collapse: f64,
    ctl_req: i32,
    long_only: bool,
) -> PyResult<Bound<'_, PyDict>> {
    let n = closes.len();
    if n < 10 {
        return Err(PyValueError::new_err("Need at least 10 bars"));
    }
    if highs.len() != n || lows.len() != n {
        return Err(PyValueError::new_err("closes, highs, lows must be the same length"));
    }

    let (bh_masses, bh_active_vec, ctl_series, betas) =
        run_bh_physics(&closes, cf, bh_form, bh_decay, bh_collapse, ctl_req);

    // 20-bar EMA
    let alpha20 = 2.0 / 21.0;
    let mut ema20 = closes[0];
    let mut ema20_series = vec![closes[0]; n];
    for i in 1..n {
        ema20 = ema20 * (1.0 - alpha20) + closes[i] * alpha20;
        ema20_series[i] = ema20;
    }

    // Regime classification
    let mut regime = vec![REGIME_SIDEWAYS; n];
    for i in 1..n {
        let price = closes[i];
        let ema = ema20_series[i];
        let act = bh_active_vec[i];
        let beta = betas[i];
        regime[i] = if act && price > ema {
            REGIME_BULL
        } else if act && price < ema {
            REGIME_BEAR
        } else if !act && beta > 2.0 {
            REGIME_HIGH_VOL
        } else {
            REGIME_SIDEWAYS
        };
    }

    // Position sizing (same logic as simulate)
    let max_lev = 1.0_f64;
    let mut equity = vec![1.0f64; n];
    let mut positions = vec![0.0f64; n];
    let mut last_pos = 0.0f64;

    // Trade tracking state
    let mut in_trade = false;
    let mut trade_entry_bar = 0usize;
    let mut trade_entry_price = 0.0f64;
    let mut trade_pos = 0.0f64; // signed position at entry
    let mut trade_best_price = 0.0f64;
    let mut trade_worst_price = 0.0f64;
    let mut trade_regime_at_entry = REGIME_SIDEWAYS;
    let mut trade_mass_at_entry = 0.0f64;
    let mut trades: Vec<TradeRecord> = Vec::new();

    for i in 1..n {
        let ema = ema20_series[i];
        let direction = if closes[i] > ema { 1.0 } else { -1.0 };
        let direction = if long_only && direction < 0.0 { 0.0 } else { direction };

        let target = if bh_active_vec[i] {
            max_lev * direction
        } else if ctl_series[i] >= 3 {
            max_lev * 0.5 * direction
        } else {
            0.0
        };

        // Determine effective position (warmup: first 50 bars = flat)
        let pos_i = if i >= 50 { target } else { 0.0 };
        positions[i] = pos_i;

        // Trade entry / exit detection
        let pos_changed = (pos_i - last_pos).abs() > 0.01;

        if pos_changed {
            // Close existing trade
            if in_trade {
                let exit_price = closes[i];
                let entry_price = trade_entry_price;
                let mfe_frac = if trade_pos > 0.0 {
                    (trade_best_price - entry_price) / entry_price
                } else if trade_pos < 0.0 {
                    (entry_price - trade_worst_price) / entry_price
                } else {
                    0.0
                };
                let mae_frac = if trade_pos > 0.0 {
                    (entry_price - trade_worst_price) / entry_price
                } else if trade_pos < 0.0 {
                    (trade_best_price - entry_price) / entry_price
                } else {
                    0.0
                };
                let pnl_frac = if trade_pos > 0.0 {
                    (exit_price - entry_price) / entry_price
                } else if trade_pos < 0.0 {
                    (entry_price - exit_price) / entry_price
                } else {
                    0.0
                };
                trades.push(TradeRecord {
                    entry_bar: trade_entry_bar,
                    exit_bar: i,
                    entry_price,
                    exit_price,
                    pnl_frac,
                    hold_bars: i - trade_entry_bar,
                    mfe_frac,
                    mae_frac,
                    regime_at_entry: trade_regime_at_entry,
                    bh_mass_at_entry: trade_mass_at_entry,
                    tf_score: 1,
                });
                in_trade = false;
            }

            // Open new trade if position is non-zero
            if pos_i.abs() > 0.01 {
                in_trade = true;
                trade_entry_bar = i;
                trade_entry_price = closes[i];
                trade_pos = pos_i;
                trade_best_price = closes[i];
                trade_worst_price = closes[i];
                trade_regime_at_entry = regime[i];
                trade_mass_at_entry = bh_masses[i];
            }

            last_pos = pos_i;
        }

        // Update MFE/MAE extremes while in trade
        if in_trade {
            let h = highs[i];
            let l = lows[i];
            if h > trade_best_price {
                trade_best_price = h;
            }
            if l < trade_worst_price {
                trade_worst_price = l;
            }
        }

        // Equity update
        let ret = closes[i] / closes[i - 1] - 1.0;
        equity[i] = equity[i - 1] * (1.0 + positions[i - 1] * ret);
    }

    // Close any open trade at the end
    if in_trade {
        let exit_price = *closes.last().unwrap();
        let entry_price = trade_entry_price;
        let mfe_frac = if trade_pos > 0.0 {
            (trade_best_price - entry_price) / entry_price
        } else if trade_pos < 0.0 {
            (entry_price - trade_worst_price) / entry_price
        } else {
            0.0
        };
        let mae_frac = if trade_pos > 0.0 {
            (entry_price - trade_worst_price) / entry_price
        } else if trade_pos < 0.0 {
            (trade_best_price - entry_price) / entry_price
        } else {
            0.0
        };
        let pnl_frac = if trade_pos > 0.0 {
            (exit_price - entry_price) / entry_price
        } else if trade_pos < 0.0 {
            (entry_price - exit_price) / entry_price
        } else {
            0.0
        };
        trades.push(TradeRecord {
            entry_bar: trade_entry_bar,
            exit_bar: n - 1,
            entry_price,
            exit_price,
            pnl_frac,
            hold_bars: (n - 1) - trade_entry_bar,
            mfe_frac,
            mae_frac,
            regime_at_entry: trade_regime_at_entry,
            bh_mass_at_entry: trade_mass_at_entry,
            tf_score: 1,
        });
    }

    // Serialize trades as list of Python dicts
    let py_trades = pyo3::types::PyList::empty_bound(py);
    for tr in &trades {
        py_trades.append(tr.into_pydict(py)?)?;
    }

    // Build output dict
    let result = PyDict::new_bound(py);
    result.set_item("equity_curve", equity)?;
    result.set_item("positions", positions)?;
    result.set_item("bh_masses", bh_masses)?;
    result.set_item("bh_active", bh_active_vec)?;
    result.set_item("ctl_series", ctl_series)?;
    result.set_item("regime", regime)?;
    result.set_item("trades", py_trades)?;

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. mc_simulation
// ─────────────────────────────────────────────────────────────────────────────

/// Regime-naive Monte Carlo simulation over historical trade returns.
/// Returns (final_equities, max_drawdowns, blowup_count).
#[pyfunction]
fn mc_simulation(
    returns_vec: Vec<f64>,
    n_sims: i32,
    n_trades_per_sim: i32,
    position_frac: f64,
    serial_corr: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, i32)> {
    if returns_vec.is_empty() {
        return Err(PyValueError::new_err("returns_vec must not be empty"));
    }

    let n_sims = n_sims.max(1) as usize;
    let n_trades = n_trades_per_sim.max(1) as usize;

    // Split returns into negative and non-negative subsets for serial correlation
    let neg_returns: Vec<f64> = returns_vec.iter().copied().filter(|&r| r < 0.0).collect();
    let pos_returns: Vec<f64> = returns_vec.iter().copied().filter(|&r| r >= 0.0).collect();

    let mut rng = rand::thread_rng();
    let uniform01 = Uniform::new(0.0f64, 1.0f64);
    let full_dist = Uniform::new(0usize, returns_vec.len());

    let mut final_equities = Vec::with_capacity(n_sims);
    let mut max_drawdowns = Vec::with_capacity(n_sims);
    let mut blowup_count = 0i32;

    for _ in 0..n_sims {
        let mut equity = 1.0f64;
        let mut peak = 1.0f64;
        let mut max_dd = 0.0f64;
        let mut last_was_neg = false;
        let mut blowup = false;

        for _ in 0..n_trades {
            // Draw a return, optionally biased by serial correlation
            let drawn_return = if serial_corr > 1e-9
                && last_was_neg
                && !neg_returns.is_empty()
                && !pos_returns.is_empty()
            {
                // With probability (0.5 + serial_corr) draw from neg subset
                let p = rng.sample(uniform01);
                if p < (0.5 + serial_corr).min(1.0) {
                    let idx = rng.gen_range(0..neg_returns.len());
                    neg_returns[idx]
                } else {
                    let idx = rng.gen_range(0..pos_returns.len());
                    pos_returns[idx]
                }
            } else {
                let idx = rng.sample(full_dist);
                returns_vec[idx]
            };

            last_was_neg = drawn_return < 0.0;

            // Apply return to equity
            equity *= 1.0 + position_frac * drawn_return;

            if equity <= 0.0 {
                equity = 0.0;
                blowup = true;
                break;
            }

            if equity > peak {
                peak = equity;
            }
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        if blowup {
            blowup_count += 1;
        }

        final_equities.push(equity);
        max_drawdowns.push(max_dd);
    }

    Ok((final_equities, max_drawdowns, blowup_count))
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. sensitivity_sweep
// ─────────────────────────────────────────────────────────────────────────────

/// Sensitivity sweep over a single parameter.
/// Returns Vec of (perturbation_mult, final_equity, sharpe_approx, max_drawdown).
#[pyfunction]
fn sensitivity_sweep(
    py: Python<'_>,
    closes: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    param_name: String,
    base_value: f64,
    perturbations: Vec<f64>,
) -> PyResult<Vec<(f64, f64, f64, f64)>> {
    let mut results = Vec::with_capacity(perturbations.len());

    for &mult in &perturbations {
        let perturbed = base_value * mult;

        // Resolve which param is being swept; everything else stays at defaults
        let cf = if param_name == "cf" { perturbed } else { DEF_CF };
        let bh_form = if param_name == "bh_form" { perturbed } else { DEF_BH_FORM };
        let bh_decay = if param_name == "bh_decay" { perturbed } else { DEF_BH_DECAY };
        let bh_collapse = if param_name == "bh_collapse" { perturbed } else { DEF_BH_COLLAPSE };
        let ctl_req = if param_name == "ctl_req" {
            perturbed.round() as i32
        } else {
            DEF_CTL_REQ
        };

        let result_dict = full_backtest(
            py,
            closes.clone(),
            highs.clone(),
            lows.clone(),
            cf,
            bh_form,
            bh_decay,
            bh_collapse,
            ctl_req,
            false,
        )?;

        // Extract equity_curve from the returned dict
        let equity_curve: Vec<f64> = result_dict
            .get_item("equity_curve")?
            .ok_or_else(|| PyValueError::new_err("missing equity_curve"))?
            .extract()?;

        let final_eq = *equity_curve.last().unwrap_or(&1.0);
        let sh = sharpe_approx(&equity_curve);
        let dd = max_dd_internal(&equity_curve);

        results.push((mult, final_eq, sh, dd));
    }

    Ok(results)
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. bh_correlation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Jaccard and Pearson correlation between BH active series of two instruments.
/// Returns (jaccard: f64, pearson: f64).
#[pyfunction]
fn bh_correlation(
    closes_a: Vec<f64>,
    closes_b: Vec<f64>,
    cf_a: f64,
    cf_b: f64,
    bh_form: f64,
    bh_decay: f64,
    bh_collapse: f64,
) -> PyResult<(f64, f64)> {
    let ctl_req = DEF_CTL_REQ;

    let (_, active_a, _, _) = run_bh_physics(&closes_a, cf_a, bh_form, bh_decay, bh_collapse, ctl_req);
    let (_, active_b, _, _) = run_bh_physics(&closes_b, cf_b, bh_form, bh_decay, bh_collapse, ctl_req);

    // Align to the shorter series
    let len = active_a.len().min(active_b.len());
    if len == 0 {
        return Ok((0.0, 0.0));
    }

    let a: Vec<f64> = active_a[..len].iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
    let b: Vec<f64> = active_b[..len].iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();

    // Jaccard: |A ∩ B| / |A ∪ B|
    let both_active: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let either_active: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| if ai > 0.0 || bi > 0.0 { 1.0 } else { 0.0 })
        .sum();
    let jaccard = if either_active < 1.0 { 0.0 } else { both_active / either_active };

    // Pearson correlation
    let n = len as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let cov: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - mean_a) * (bi - mean_b))
        .sum::<f64>()
        / n;
    let std_a = (a.iter().map(|&ai| (ai - mean_a).powi(2)).sum::<f64>() / n).sqrt();
    let std_b = (b.iter().map(|&bi| (bi - mean_b).powi(2)).sum::<f64>() / n).sqrt();
    let pearson = if std_a < 1e-12 || std_b < 1e-12 {
        0.0
    } else {
        cov / (std_a * std_b)
    };

    Ok((jaccard, pearson))
}

// ─────────────────────────────────────────────────────────────────────────────
// Python module definition
// ─────────────────────────────────────────────────────────────────────────────

#[pymodule]
fn larsa_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Existing functions
    m.add_function(wrap_pyfunction!(beta_series, m)?)?;
    m.add_function(wrap_pyfunction!(bh_mass_series, m)?)?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(sweep, m)?)?;
    // New functions
    m.add_function(wrap_pyfunction!(full_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(mc_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(bh_correlation, m)?)?;
    m.add("__version__", "0.2.0")?;
    Ok(())
}
