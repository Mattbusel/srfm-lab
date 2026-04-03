use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Core SRFM physics simulation in Rust.
/// Called from Python as: import larsa_core
///
/// All functions accept Vec<f64> (Python list or numpy array converted)
/// and return Vec<f64> results.

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

/// Python module definition
#[pymodule]
fn larsa_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(beta_series, m)?)?;
    m.add_function(wrap_pyfunction!(bh_mass_series, m)?)?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(sweep, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
