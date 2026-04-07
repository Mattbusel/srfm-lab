//! signal_visualizer.rs -- WebAssembly-exported signal visualization data generator.
//!
//! Produces JSON arrays consumed directly by the SRFM dashboard charting layer.
//! All computation is self-contained: no HTTP calls, no DOM access.
//! Each exported function takes JSON input, validates it, computes, and returns JSON output.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Input types -- deserialized from JS caller
// ---------------------------------------------------------------------------

/// One OHLCV bar as received from the browser.
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
struct Bar {
    ts: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

/// One equity curve data point.
#[derive(Debug, Deserialize, Clone)]
struct EquityPoint {
    ts: i64,
    equity: f64,
}

// ---------------------------------------------------------------------------
// Output types -- serialized back to JS
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct BhTrajectoryPoint {
    ts: i64,
    bh_mass: f64,
    nav_curvature: f64,
    hurst: f64,
    regime: String,
}

#[derive(Debug, Serialize)]
struct HeatmapSlice {
    ts: i64,
    matrix: Vec<Vec<f64>>,
    labels: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DrawdownSegment {
    peak_ts: i64,
    trough_ts: i64,
    recovery_ts: Option<i64>,
    peak_equity: f64,
    trough_equity: f64,
    drawdown_pct: f64,
    duration_bars: usize,
    recovery_bars: Option<usize>,
}

#[derive(Debug, Serialize)]
struct FactorContribution {
    ts: i64,
    total_return: f64,
    bh_component: f64,
    nav_component: f64,
    hurst_component: f64,
    garch_component: f64,
    residual: f64,
}

// ---------------------------------------------------------------------------
// Main exported struct
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct SignalVisualizerWasm {
    /// Smoothing factor used in internal EMA computations.
    ema_alpha: f64,
    /// GARCH(1,1) omega parameter for volatility estimation.
    garch_omega: f64,
    /// GARCH alpha1 (ARCH term).
    garch_alpha: f64,
    /// GARCH beta1 (GARCH term).
    garch_beta: f64,
}

#[wasm_bindgen]
impl SignalVisualizerWasm {
    /// Construct a new visualizer with default GARCH(1,1) parameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> SignalVisualizerWasm {
        SignalVisualizerWasm {
            ema_alpha: 0.1,
            garch_omega: 1e-6,
            garch_alpha: 0.08,
            garch_beta: 0.90,
        }
    }

    /// Override GARCH parameters (useful for asset-specific calibration).
    pub fn set_garch_params(&mut self, omega: f64, alpha: f64, beta: f64) {
        self.garch_omega = omega;
        self.garch_alpha = alpha;
        self.garch_beta = beta;
    }

    /// Compute BH trajectory: returns JSON array of BhTrajectoryPoint.
    ///
    /// Each element contains BH mass, nav curvature, Hurst exponent, and regime label.
    /// Minimum 20 bars required; shorter slices return an empty array.
    pub fn compute_bh_trajectory(&self, bars_json: &str) -> String {
        let bars: Vec<Bar> = match serde_json::from_str(bars_json) {
            Ok(b) => b,
            Err(e) => return format!("{{\"error\":\"parse: {}\"}}", e),
        };
        if bars.len() < 20 {
            return "[]".to_string();
        }

        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        let n = closes.len();

        // BH mass: rolling accumulation matching SRFM logic
        let bh_masses = compute_bh_mass_series_internal(&closes, 0.01, 0.95);

        // Nav curvature: second derivative of log-price smoothed with EMA
        let log_prices: Vec<f64> = closes.iter().map(|c| c.ln()).collect();
        let curvatures = compute_nav_curvature(&log_prices, self.ema_alpha);

        // Hurst exponent via rescaled range over trailing 50-bar window
        let hursts = compute_rolling_hurst(&closes, 50);

        // GARCH conditional volatility
        let garch_vols = compute_garch_vols(
            &closes,
            self.garch_omega,
            self.garch_alpha,
            self.garch_beta,
        );

        let mut out: Vec<BhTrajectoryPoint> = Vec::with_capacity(n);
        for i in 0..n {
            let regime = classify_regime(
                bh_masses[i],
                curvatures[i],
                hursts[i],
                garch_vols[i],
            );
            out.push(BhTrajectoryPoint {
                ts: bars[i].ts,
                bh_mass: bh_masses[i],
                nav_curvature: curvatures[i],
                hurst: hursts[i],
                regime,
            });
        }

        serde_json::to_string(&out).unwrap_or_else(|_| "[]".to_string())
    }

    /// Compute rolling correlation heatmap data.
    ///
    /// Computes the NxN correlation matrix of [close, volume, returns, log_volume, atr]
    /// over each rolling window of `window` bars and returns one slice per step.
    pub fn compute_signal_heatmap(&self, bars_json: &str, window: usize) -> String {
        let bars: Vec<Bar> = match serde_json::from_str(bars_json) {
            Ok(b) => b,
            Err(e) => return format!("{{\"error\":\"parse: {}\"}}", e),
        };
        let win = window.max(5).min(bars.len());
        if bars.len() < win {
            return "[]".to_string();
        }

        let labels = vec![
            "close".to_string(),
            "volume".to_string(),
            "returns".to_string(),
            "log_vol".to_string(),
            "atr".to_string(),
        ];
        let n_signals = labels.len();

        // Precompute signal series
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();
        let n = closes.len();

        let mut returns = vec![0.0f64; n];
        for i in 1..n {
            returns[i] = closes[i] / closes[i - 1] - 1.0;
        }

        let log_vols: Vec<f64> = volumes
            .iter()
            .map(|v| if *v > 0.0 { v.ln() } else { 0.0 })
            .collect();

        let atrs = compute_atr_series(&bars);

        let series: Vec<&Vec<f64>> = vec![&closes, &volumes, &returns, &log_vols, &atrs];

        let mut slices: Vec<HeatmapSlice> = Vec::with_capacity(n - win + 1);
        for start in 0..=(n - win) {
            let end = start + win;
            let ts = bars[end - 1].ts;

            let mut windows: Vec<Vec<f64>> = series
                .iter()
                .map(|s| s[start..end].to_vec())
                .collect();

            // Z-score normalize each signal window before correlating
            for w in windows.iter_mut() {
                zscore_normalize(w);
            }

            let mut matrix = vec![vec![0.0f64; n_signals]; n_signals];
            for i in 0..n_signals {
                for j in 0..n_signals {
                    if i == j {
                        matrix[i][j] = 1.0;
                    } else if i < j {
                        let r = pearson_correlation(&windows[i], &windows[j]);
                        matrix[i][j] = r;
                        matrix[j][i] = r;
                    }
                }
            }

            slices.push(HeatmapSlice { ts, matrix, labels: labels.clone() });
        }

        serde_json::to_string(&slices).unwrap_or_else(|_| "[]".to_string())
    }

    /// Compute drawdown profile from an equity curve.
    ///
    /// Returns array of DrawdownSegment, each describing one peak-to-trough-to-recovery cycle.
    /// Only segments with drawdown >= 0.5% are included to reduce noise.
    pub fn compute_drawdown_profile(&self, equity_json: &str) -> String {
        let points: Vec<EquityPoint> = match serde_json::from_str(equity_json) {
            Ok(p) => p,
            Err(e) => return format!("{{\"error\":\"parse: {}\"}}", e),
        };
        if points.len() < 2 {
            return "[]".to_string();
        }

        let mut segments: Vec<DrawdownSegment> = Vec::new();
        let mut peak_idx = 0usize;
        let mut peak_equity = points[0].equity;
        let mut in_drawdown = false;
        let mut trough_idx = 0usize;
        let mut trough_equity = peak_equity;

        for i in 1..points.len() {
            let eq = points[i].equity;

            if eq > peak_equity {
                // New peak: if we were in a drawdown, record it (without recovery since price
                // surpassed old peak at bar i)
                if in_drawdown {
                    let dd_pct = (peak_equity - trough_equity) / peak_equity;
                    if dd_pct >= 0.005 {
                        segments.push(DrawdownSegment {
                            peak_ts: points[peak_idx].ts,
                            trough_ts: points[trough_idx].ts,
                            recovery_ts: Some(points[i].ts),
                            peak_equity,
                            trough_equity,
                            drawdown_pct: dd_pct,
                            duration_bars: trough_idx - peak_idx,
                            recovery_bars: Some(i - trough_idx),
                        });
                    }
                    in_drawdown = false;
                }
                peak_idx = i;
                peak_equity = eq;
                trough_idx = i;
                trough_equity = eq;
            } else if eq < trough_equity {
                // Deeper trough
                in_drawdown = true;
                trough_idx = i;
                trough_equity = eq;
            }
        }

        // Handle open drawdown at end of series
        if in_drawdown {
            let dd_pct = (peak_equity - trough_equity) / peak_equity;
            if dd_pct >= 0.005 {
                segments.push(DrawdownSegment {
                    peak_ts: points[peak_idx].ts,
                    trough_ts: points[trough_idx].ts,
                    recovery_ts: None,
                    peak_equity,
                    trough_equity,
                    drawdown_pct: dd_pct,
                    duration_bars: trough_idx - peak_idx,
                    recovery_bars: None,
                });
            }
        }

        serde_json::to_string(&segments).unwrap_or_else(|_| "[]".to_string())
    }

    /// Decompose bar-by-bar returns into BH / Nav / Hurst / GARCH factor components.
    ///
    /// Uses OLS attribution: regress each return against the four signal values,
    /// with the residual as the unexplained component.
    pub fn compute_factor_contributions(&self, bars_json: &str) -> String {
        let bars: Vec<Bar> = match serde_json::from_str(bars_json) {
            Ok(b) => b,
            Err(e) => return format!("{{\"error\":\"parse: {}\"}}", e),
        };
        if bars.len() < 30 {
            return "[]".to_string();
        }

        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        let n = closes.len();

        let bh_masses = compute_bh_mass_series_internal(&closes, 0.01, 0.95);
        let log_prices: Vec<f64> = closes.iter().map(|c| c.ln()).collect();
        let curvatures = compute_nav_curvature(&log_prices, self.ema_alpha);
        let hursts = compute_rolling_hurst(&closes, 50);
        let garch_vols = compute_garch_vols(
            &closes,
            self.garch_omega,
            self.garch_alpha,
            self.garch_beta,
        );

        // Compute rolling 30-bar OLS betas, then attribute each bar's return
        let window = 30usize;
        let mut out: Vec<FactorContribution> = Vec::with_capacity(n);

        // First window-1 bars: not enough history, emit zeros
        for i in 0..(window.min(n)) {
            out.push(FactorContribution {
                ts: bars[i].ts,
                total_return: 0.0,
                bh_component: 0.0,
                nav_component: 0.0,
                hurst_component: 0.0,
                garch_component: 0.0,
                residual: 0.0,
            });
        }

        for i in window..n {
            let ret = closes[i] / closes[i - 1] - 1.0;

            // Build factor matrix for the trailing window (exclude current bar from regression)
            let y: Vec<f64> = (i - window + 1..i)
                .map(|k| closes[k] / closes[k - 1] - 1.0)
                .collect();

            let x_bh: Vec<f64> = (i - window..i - 1).map(|k| bh_masses[k]).collect();
            let x_nav: Vec<f64> = (i - window..i - 1).map(|k| curvatures[k]).collect();
            let x_hurst: Vec<f64> = (i - window..i - 1).map(|k| hursts[k]).collect();
            let x_garch: Vec<f64> = (i - window..i - 1).map(|k| garch_vols[k]).collect();

            let b_bh = simple_ols_beta(&x_bh, &y);
            let b_nav = simple_ols_beta(&x_nav, &y);
            let b_hurst = simple_ols_beta(&x_hurst, &y);
            let b_garch = simple_ols_beta(&x_garch, &y);

            let bh_comp = b_bh * bh_masses[i - 1];
            let nav_comp = b_nav * curvatures[i - 1];
            let hurst_comp = b_hurst * hursts[i - 1];
            let garch_comp = b_garch * garch_vols[i - 1];
            let explained = bh_comp + nav_comp + hurst_comp + garch_comp;
            let residual = ret - explained;

            out.push(FactorContribution {
                ts: bars[i].ts,
                total_return: ret,
                bh_component: bh_comp,
                nav_component: nav_comp,
                hurst_component: hurst_comp,
                garch_component: garch_comp,
                residual,
            });
        }

        serde_json::to_string(&out).unwrap_or_else(|_| "[]".to_string())
    }
}

// ---------------------------------------------------------------------------
// Internal computation helpers
// ---------------------------------------------------------------------------

/// Compute BH mass series using the SRFM accumulation model.
/// cf = curvature factor, decay = mass decay on volatile bars.
fn compute_bh_mass_series_internal(closes: &[f64], cf: f64, decay: f64) -> Vec<f64> {
    let n = closes.len();
    let mut masses = vec![0.0f64; n];
    let mut mass = 0.0f64;
    for i in 1..n {
        let beta = (closes[i] - closes[i - 1]).abs() / (closes[i - 1] * cf + 1e-12);
        if beta < 1.0 {
            mass = mass * 0.97 + 0.03;
        } else {
            mass *= decay;
        }
        masses[i] = mass;
    }
    masses
}

/// Compute NAV curvature as the EMA-smoothed second derivative of a series.
fn compute_nav_curvature(log_prices: &[f64], alpha: f64) -> Vec<f64> {
    let n = log_prices.len();
    let mut curvatures = vec![0.0f64; n];
    if n < 3 {
        return curvatures;
    }

    let mut ema_raw = 0.0f64;
    for i in 2..n {
        let d2 = log_prices[i] - 2.0 * log_prices[i - 1] + log_prices[i - 2];
        ema_raw = alpha * d2 + (1.0 - alpha) * ema_raw;
        curvatures[i] = ema_raw;
    }
    curvatures
}

/// Compute rolling Hurst exponent via rescaled range (R/S) analysis.
/// Returns 0.5 (random walk) when the window is too short to estimate.
fn compute_rolling_hurst(closes: &[f64], window: usize) -> Vec<f64> {
    let n = closes.len();
    let win = window.min(n);
    let mut hursts = vec![0.5f64; n];

    for i in win..n {
        let slice = &closes[(i - win)..i];
        hursts[i] = hurst_rs(slice);
    }
    hursts
}

/// Rescaled range Hurst estimator over a single slice.
/// H = log(R/S) / log(n); returns 0.5 on degenerate input.
fn hurst_rs(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 4 {
        return 0.5;
    }

    let mean = xs.iter().sum::<f64>() / n as f64;
    let mut cumdev = 0.0f64;
    let mut cumdevs = Vec::with_capacity(n);
    for x in xs {
        cumdev += x - mean;
        cumdevs.push(cumdev);
    }

    let range = cumdevs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - cumdevs.iter().cloned().fold(f64::INFINITY, f64::min);

    let variance = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    if std_dev < 1e-12 || range < 1e-12 {
        return 0.5;
    }

    let rs = range / std_dev;
    let h = rs.ln() / (n as f64).ln();
    h.clamp(0.0, 1.0)
}

/// GARCH(1,1) conditional variance series.
/// Returns per-bar conditional standard deviation.
fn compute_garch_vols(closes: &[f64], omega: f64, alpha: f64, beta: f64) -> Vec<f64> {
    let n = closes.len();
    let mut vols = vec![0.0f64; n];
    if n < 2 {
        return vols;
    }

    let mut h = 1e-4f64; // initial variance
    for i in 1..n {
        let ret = closes[i] / closes[i - 1] - 1.0;
        h = omega + alpha * ret * ret + beta * h;
        vols[i] = h.sqrt();
    }
    vols
}

/// Classify a bar into a regime label based on BH mass, curvature, Hurst, and GARCH vol.
fn classify_regime(bh_mass: f64, curvature: f64, hurst: f64, garch_vol: f64) -> String {
    if bh_mass > 0.7 && hurst > 0.6 {
        "strong_trend".to_string()
    } else if bh_mass > 0.4 && hurst > 0.5 {
        "trend".to_string()
    } else if hurst < 0.45 && garch_vol > 0.015 {
        "volatile_mean_revert".to_string()
    } else if hurst < 0.45 {
        "mean_revert".to_string()
    } else if garch_vol > 0.02 {
        "volatile".to_string()
    } else if curvature.abs() < 1e-5 {
        "flat".to_string()
    } else {
        "neutral".to_string()
    }
}

/// Compute ATR series (Average True Range) over the bar slice.
fn compute_atr_series(bars: &[Bar]) -> Vec<f64> {
    let n = bars.len();
    let mut atrs = vec![0.0f64; n];
    if n < 2 {
        return atrs;
    }

    let period = 14usize;
    let mut tr_ema = (bars[0].high - bars[0].low).abs();
    let alpha = 2.0 / (period as f64 + 1.0);

    for i in 1..n {
        let hl = bars[i].high - bars[i].low;
        let hc = (bars[i].high - bars[i - 1].close).abs();
        let lc = (bars[i].low - bars[i - 1].close).abs();
        let tr = hl.max(hc).max(lc);
        tr_ema = alpha * tr + (1.0 - alpha) * tr_ema;
        atrs[i] = tr_ema;
    }
    atrs
}

/// Pearson correlation between two same-length slices.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let num: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - mx) * (b - my)).sum();
    let dx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = y.iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();
    if dx < 1e-12 || dy < 1e-12 {
        0.0
    } else {
        (num / (dx * dy)).clamp(-1.0, 1.0)
    }
}

/// Z-score normalize a mutable slice in place (mean 0, std 1).
fn zscore_normalize(v: &mut Vec<f64>) {
    let n = v.len() as f64;
    if n < 2.0 {
        return;
    }
    let mean = v.iter().sum::<f64>() / n;
    let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    if std < 1e-12 {
        for x in v.iter_mut() {
            *x = 0.0;
        }
    } else {
        for x in v.iter_mut() {
            *x = (*x - mean) / std;
        }
    }
}

/// Simple univariate OLS slope (beta) of y on x (no intercept adjustment here --
/// we use demeaned vectors so the intercept is implicitly zero).
fn simple_ols_beta(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n as f64;
    let my = y.iter().sum::<f64>() / n as f64;
    let num: f64 = x[..n]
        .iter()
        .zip(y[..n].iter())
        .map(|(xi, yi)| (xi - mx) * (yi - my))
        .sum();
    let denom: f64 = x[..n].iter().map(|xi| (xi - mx).powi(2)).sum();
    if denom < 1e-14 {
        0.0
    } else {
        num / denom
    }
}
