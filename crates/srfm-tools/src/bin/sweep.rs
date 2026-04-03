/// sweep — Parallel SRFM parameter sweep using Rayon.
///
/// Usage:
///   sweep --csv data/NDX_hourly_poly.csv --cf-range 0.001,0.015,20 --lev-range 0.30,0.80,10
///
/// Output: TSV to stdout with columns: cf, max_lev, sharpe, return_pct, max_dd, trade_count

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// CSV loader
// ---------------------------------------------------------------------------

fn load_closes(path: &str) -> Vec<f64> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path, e));
    let mut lines = content.lines();

    // Parse header to find close column index
    let header = match lines.next() {
        Some(h) => h,
        None => return vec![],
    };
    let cols: Vec<&str> = header.split(',').collect();
    let close_idx = cols
        .iter()
        .position(|c| c.trim().to_lowercase() == "close")
        .unwrap_or_else(|| panic!("No 'close' column in {}", path));

    let mut closes = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if let Some(val_str) = fields.get(close_idx) {
            let v = val_str.trim();
            if v.is_empty() || v == "null" || v == "None" {
                continue;
            }
            if let Ok(f) = v.parse::<f64>() {
                if f > 0.0 {
                    closes.push(f);
                }
            }
        }
    }
    closes
}

// ---------------------------------------------------------------------------
// Simulation result
// ---------------------------------------------------------------------------

struct SimResult {
    sharpe: f64,
    return_pct: f64,
    max_dd: f64,
    trade_count: usize,
}

// ---------------------------------------------------------------------------
// SRFM core simulation
// ---------------------------------------------------------------------------

fn srfm_core(closes: &[f64], cf: f64, bh_form: f64, bh_decay: f64, max_lev: f64) -> SimResult {
    let n = closes.len();
    if n < 2 {
        return SimResult {
            sharpe: 0.0,
            return_pct: 0.0,
            max_dd: 0.0,
            trade_count: 0,
        };
    }

    let bh_collapse = 1.0_f64;
    let ema20_k = 2.0 / 21.0_f64;

    let mut equity = vec![1.0_f64; n];
    let mut positions = vec![0.0_f64; n];

    let mut mass = 0.0_f64;
    let mut ctl: usize = 0;
    let mut bh_active = false;
    let mut ema20 = closes[0];

    for i in 1..n {
        let prev = closes[i - 1];
        let cur = closes[i];

        // Update EMA-20
        ema20 = cur * ema20_k + ema20 * (1.0 - ema20_k);

        // SRFM physics
        let beta = if prev > 0.0 {
            (cur - prev).abs() / (prev * cf)
        } else {
            0.0
        };

        if beta < 1.0 {
            // TIMELIKE
            mass = mass * 0.97 + 0.03;
            ctl += 1;
        } else {
            // SPACELIKE
            mass *= bh_decay;
            ctl = 0;
        }

        // BH activation / collapse
        if bh_active {
            if mass < bh_collapse {
                bh_active = false;
            }
        } else if mass >= bh_form && ctl >= 5 {
            bh_active = true;
        }

        // Position sizing
        let sign = if cur > ema20 { 1.0 } else { -1.0 };
        let pos = if bh_active {
            max_lev * sign
        } else {
            max_lev * 0.5 * sign
        };
        positions[i] = pos;

        // Equity
        let bar_ret = if prev > 0.0 { cur / prev - 1.0 } else { 0.0 };
        equity[i] = equity[i - 1] * (1.0 + positions[i - 1] * bar_ret);
    }

    // Sharpe (annualised, hourly bars → 252*24 periods/year)
    let returns: Vec<f64> = (1..n)
        .map(|i| {
            if equity[i - 1] > 0.0 {
                equity[i] / equity[i - 1] - 1.0
            } else {
                0.0
            }
        })
        .collect();

    let mean_r = returns.iter().sum::<f64>() / returns.len() as f64;
    let var_r = returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>()
        / (returns.len() - 1).max(1) as f64;
    let std_r = var_r.sqrt();

    let sharpe = if std_r > 0.0 {
        (252.0_f64 * 24.0).sqrt() * mean_r / std_r
    } else {
        0.0
    };

    // Max drawdown
    let mut peak = equity[0];
    let mut max_dd = 0.0_f64;
    for &v in &equity {
        if v > peak {
            peak = v;
        }
        let dd = if peak > 0.0 { 1.0 - v / peak } else { 0.0 };
        if dd > max_dd {
            max_dd = dd;
        }
    }

    // Trade count (sign changes in positions)
    let mut trade_count = 0usize;
    for i in 1..positions.len() {
        let s_prev = positions[i - 1].signum();
        let s_cur = positions[i].signum();
        if (s_prev - s_cur).abs() > 0.5 {
            trade_count += 1;
        }
    }

    let return_pct = (equity[n - 1] - 1.0) * 100.0;

    SimResult {
        sharpe,
        return_pct,
        max_dd,
        trade_count,
    }
}

// ---------------------------------------------------------------------------
// Parameter range parser
// ---------------------------------------------------------------------------

/// Parse "start,end,n_steps" → Vec of n_steps evenly spaced values.
fn parse_range(s: &str, name: &str) -> Vec<f64> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        panic!("--{}-range must be start,end,n_steps (got {:?})", name, s);
    }
    let start: f64 = parts[0].parse().expect("range start");
    let end: f64 = parts[1].parse().expect("range end");
    let n: usize = parts[2].parse().expect("range n_steps");
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
        .collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Simple arg parser
    let get_flag = |flag: &str| -> Option<String> {
        for i in 0..args.len() {
            if args[i] == flag {
                if i + 1 < args.len() {
                    return Some(args[i + 1].clone());
                }
            }
        }
        None
    };

    let csv_path = get_flag("--csv").unwrap_or_else(|| {
        eprintln!("Usage: sweep --csv <file> --cf-range start,end,n --lev-range start,end,n");
        std::process::exit(1);
    });

    let cf_range_str = get_flag("--cf-range").unwrap_or_else(|| "0.001,0.015,20".to_string());
    let lev_range_str = get_flag("--lev-range").unwrap_or_else(|| "0.30,0.80,10".to_string());

    let bh_form: f64 = get_flag("--bh-form")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.5);
    let bh_decay: f64 = get_flag("--bh-decay")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.95);

    let cf_vals = parse_range(&cf_range_str, "cf");
    let lev_vals = parse_range(&lev_range_str, "lev");

    eprintln!(
        "Loading closes from {}...",
        csv_path
    );
    let closes = load_closes(&csv_path);
    eprintln!(
        "  {} bars loaded. Running {}x{} = {} combos...",
        closes.len(),
        cf_vals.len(),
        lev_vals.len(),
        cf_vals.len() * lev_vals.len()
    );

    // Build all (cf, lev) combos
    let combos: Vec<(f64, f64)> = cf_vals
        .iter()
        .flat_map(|&cf| lev_vals.iter().map(move |&lev| (cf, lev)))
        .collect();

    // Parallel sweep
    let mut results: Vec<(f64, f64, SimResult)> = combos
        .par_iter()
        .map(|&(cf, lev)| {
            let res = srfm_core(&closes, cf, bh_form, bh_decay, lev);
            (cf, lev, res)
        })
        .collect();

    // Sort by sharpe descending
    results.sort_by(|a, b| b.2.sharpe.partial_cmp(&a.2.sharpe).unwrap_or(std::cmp::Ordering::Equal));

    // TSV output
    println!("cf\tmax_lev\tsharpe\treturn_pct\tmax_dd\ttrade_count");
    for (cf, lev, r) in &results {
        println!(
            "{:.6}\t{:.4}\t{:.4}\t{:.2}\t{:.2}\t{}",
            cf, lev, r.sharpe, r.return_pct, r.max_dd * 100.0, r.trade_count
        );
    }
}
