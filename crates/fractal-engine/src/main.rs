/// fractal-engine CLI
///
/// Loads a price series from CSV, computes Hurst exponent, fractal dimension,
/// wavelet decomposition, and prints market regime + pattern matches.
///
/// Usage:
///   fractal-engine --data <csv> [--window <n>] [--levels <n>] [--output <json>]

use anyhow::{Context, Result};
use clap::Parser;
use fractal_engine::wavelet::forward_dwt;
use fractal_engine::pattern_library::scan_patterns;
use fractal_engine::regime_detector::{current_regime, rolling_regime, RegimeStats};
use fractal_engine::similarity::WaveletFingerprint;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(
    name = "fractal-engine",
    about = "Wavelet-based fractal pattern recognition and regime detection"
)]
struct Args {
    /// Path to OHLCV or price CSV file
    #[arg(long, default_value = "data/crypto_trades.csv")]
    data: String,

    /// Rolling window for Hurst calculation
    #[arg(long, default_value_t = 128)]
    hurst_window: usize,

    /// DWT decomposition levels
    #[arg(long, default_value_t = 4)]
    levels: usize,

    /// Higuchi k_max parameter
    #[arg(long, default_value_t = 8)]
    k_max: usize,

    /// Output JSON path (stdout if not provided)
    #[arg(long)]
    output: Option<String>,

    /// Print rolling regime CSV to stderr
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

/// Full analysis output.
#[derive(Debug, Serialize, Deserialize)]
struct FractalAnalysisOutput {
    total_bars: usize,
    current_regime: String,
    hurst: Option<f64>,
    hurst_interpretation: String,
    fractal_dimension: Option<f64>,
    fd_interpretation: String,
    dominant_wavelet_scale: usize,
    energy_fractions: Vec<f64>,
    pattern_matches: Vec<PatternMatchOutput>,
    regime_stats: RegimeStatsOutput,
    size_multiplier: f64,
    strategy_bias: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PatternMatchOutput {
    pattern: String,
    confidence: f64,
    description: String,
    bar_index: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegimeStatsOutput {
    trending_pct: f64,
    mean_reverting_pct: f64,
    choppy_pct: f64,
    transitioning_pct: f64,
    unknown_pct: f64,
    mean_size_multiplier: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("Loading price data from: {}", args.data);
    let prices = load_prices_from_csv(&args.data)?;
    eprintln!("Loaded {} bars", prices.len());

    if prices.len() < 32 {
        anyhow::bail!("Need at least 32 bars for analysis. Got {}", prices.len());
    }

    // Current regime
    let snap = current_regime(&prices, args.hurst_window, args.k_max);
    let hurst = snap.hurst;
    let fd = snap.fractal_dimension;

    // Wavelet decomposition
    let decomp = forward_dwt(&prices, args.levels);
    let energy_fracs = decomp.energy_fractions();
    let dominant_scale = decomp.dominant_detail_scale();

    // Pattern detection
    let raw_patterns = scan_patterns(&prices);
    let pattern_matches: Vec<PatternMatchOutput> = raw_patterns
        .iter()
        .take(5)
        .map(|p| PatternMatchOutput {
            pattern: p.pattern.as_str().to_string(),
            confidence: p.confidence,
            description: p.description.clone(),
            bar_index: p.bar_index,
        })
        .collect();

    // Rolling regime stats
    let snapshots = rolling_regime(&prices, args.hurst_window, 64, args.k_max);
    let stats = RegimeStats::from_snapshots(&snapshots);

    // Similarity library
    let _fp = WaveletFingerprint::from_prices(&prices, args.levels);

    let hurst_interp = match hurst {
        Some(h) if h > 0.6 => format!("TRENDING (H={h:.3}) — persistent, BH signal appropriate"),
        Some(h) if h < 0.4 => format!("MEAN_REVERTING (H={h:.3}) — OU signal more appropriate"),
        Some(h) => format!("RANDOM_WALK (H={h:.3}) — reduce sizing"),
        None => "UNKNOWN — insufficient data".to_string(),
    };

    let fd_interp = match fd {
        Some(f) if f < 1.3 => format!("SMOOTH_TREND (FD={f:.3}) — enter signal"),
        Some(f) if f < 1.5 => format!("TRADEABLE (FD={f:.3}) — enter with normal sizing"),
        Some(f) if f < 1.65 => format!("BROWNIAN (FD={f:.3}) — reduce sizing"),
        Some(f) => format!("NOISY (FD={f:.3}) — avoid entering"),
        None => "UNKNOWN — insufficient data".to_string(),
    };

    let output = FractalAnalysisOutput {
        total_bars: prices.len(),
        current_regime: snap.regime.as_str().to_string(),
        hurst,
        hurst_interpretation: hurst_interp,
        fractal_dimension: fd,
        fd_interpretation: fd_interp,
        dominant_wavelet_scale: dominant_scale,
        energy_fractions: energy_fracs,
        pattern_matches,
        regime_stats: RegimeStatsOutput {
            trending_pct: stats.trending_pct,
            mean_reverting_pct: stats.mean_reverting_pct,
            choppy_pct: stats.choppy_pct,
            transitioning_pct: stats.transitioning_pct,
            unknown_pct: stats.unknown_pct,
            mean_size_multiplier: stats.mean_size_multiplier,
        },
        size_multiplier: snap.size_multiplier,
        strategy_bias: snap.strategy_bias,
    };

    let json = serde_json::to_string_pretty(&output)?;

    match &args.output {
        Some(path) => {
            std::fs::write(path, &json)?;
            eprintln!("Results written to: {path}");
        }
        None => println!("{json}"),
    }

    // Print summary to stderr
    eprintln!("\n=== Fractal Analysis Summary ===");
    eprintln!("Regime          : {}", output.current_regime);
    eprintln!("Hurst           : {}", output.hurst_interpretation);
    eprintln!("Fractal Dim     : {}", output.fd_interpretation);
    eprintln!("Strategy Bias   : {}", output.strategy_bias);
    eprintln!("Size Multiplier : {:.2}", output.size_multiplier);
    if !output.pattern_matches.is_empty() {
        eprintln!("\nTop Patterns:");
        for pm in &output.pattern_matches {
            eprintln!("  [{:.2}] {} — {}", pm.confidence, pm.pattern, pm.description);
        }
    }
    eprintln!("\nRegime distribution over history:");
    eprintln!("  Trending:       {:.1}%", output.regime_stats.trending_pct);
    eprintln!("  Mean-Reverting: {:.1}%", output.regime_stats.mean_reverting_pct);
    eprintln!("  Choppy:         {:.1}%", output.regime_stats.choppy_pct);
    eprintln!("  Transitioning:  {:.1}%", output.regime_stats.transitioning_pct);

    Ok(())
}

// ---------------------------------------------------------------------------
// CSV loader (minimal — reads first numeric column as close price)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PriceRow {
    #[serde(alias = "close", alias = "price", alias = "Close", alias = "Price")]
    close: Option<f64>,
    #[serde(alias = "open", alias = "Open")]
    open: Option<f64>,
}

fn load_prices_from_csv(path: &str) -> Result<Vec<f64>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("Cannot open {path}"))?;

    let mut prices = Vec::new();
    for result in reader.deserialize::<PriceRow>() {
        match result {
            Ok(row) => {
                if let Some(p) = row.close.or(row.open) {
                    if p > 0.0 {
                        prices.push(p);
                    }
                }
            }
            Err(_) => {
                // Try raw record
            }
        }
    }

    if prices.is_empty() {
        // Fallback: read all numeric values from first column
        let mut reader2 = csv::Reader::from_path(path)
            .with_context(|| format!("Cannot open {path}"))?;
        for result in reader2.records() {
            let rec = result?;
            for field in rec.iter() {
                if let Ok(v) = field.parse::<f64>() {
                    if v > 0.0 {
                        prices.push(v);
                        break;
                    }
                }
            }
        }
    }

    if prices.is_empty() {
        anyhow::bail!("Could not parse any prices from {path}");
    }

    Ok(prices)
}
