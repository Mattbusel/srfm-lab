use clap::Parser;

use order_flow_engine::signal::{load_ohlcv_csv, OrderFlowEngine, OrderFlowSignal, SignalEngineConfig};
use order_flow_engine::vpin::estimate_bucket_volume;

/// Order Flow Engine -- compute order flow signals bar-by-bar from OHLCV data.
#[derive(Parser, Debug)]
#[command(name = "order-flow-engine", version, about)]
struct Cli {
    /// Input OHLCV CSV file (columns: timestamp, open, high, low, close, volume)
    #[arg(long)]
    input: Option<String>,

    /// Output signal CSV file. Defaults to stdout if not specified.
    #[arg(long)]
    output: Option<String>,

    /// Ticker symbol label
    #[arg(long, default_value = "UNKNOWN")]
    symbol: String,

    /// OFI rolling window (number of bars)
    #[arg(long, default_value_t = 10)]
    ofi_window: usize,

    /// VPIN number of buckets for rolling average
    #[arg(long, default_value_t = 20)]
    vpin_buckets: usize,

    /// VPIN bucket volume. 0 = auto-estimate from median bar volume.
    #[arg(long, default_value_t = 0.0)]
    vpin_bucket_volume: f64,

    /// Delta divergence lookback window
    #[arg(long, default_value_t = 14)]
    divergence_window: usize,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // ── Load data ─────────────────────────────────────────────────────────
    let rows = if let Some(ref path) = cli.input {
        println!("Loading OHLCV from: {}", path);
        load_ohlcv_csv(path)?
    } else {
        println!("No input file provided. Generating synthetic OHLCV data...");
        generate_synthetic_ohlcv(500, 42)
    };

    println!("Loaded {} bars for symbol '{}'.", rows.len(), cli.symbol);

    // ── Auto-estimate VPIN bucket volume if requested ─────────────────────
    let bucket_volume = if cli.vpin_bucket_volume > 0.0 {
        cli.vpin_bucket_volume
    } else {
        let volumes: Vec<f64> = rows.iter().map(|(_, _, _, _, _, v)| *v).collect();
        let bv = estimate_bucket_volume(&volumes);
        println!("Auto-estimated VPIN bucket volume: {:.0}", bv);
        bv
    };

    // ── Build engine and process ──────────────────────────────────────────
    let mut engine_config = SignalEngineConfig::new(&cli.symbol);
    engine_config.ofi_window = cli.ofi_window;
    engine_config.vpin_bucket_volume = bucket_volume;
    engine_config.vpin_n_buckets = cli.vpin_buckets;
    engine_config.divergence_window = cli.divergence_window;

    let mut engine = OrderFlowEngine::new(engine_config);
    let signals = engine.process_series(&rows);

    // ── Summary stats ─────────────────────────────────────────────────────
    print_summary(&signals);

    // ── Write output ──────────────────────────────────────────────────────
    let csv_content = signals_to_csv(&signals);

    if let Some(ref out_path) = cli.output {
        std::fs::write(out_path, &csv_content)?;
        println!("Signals written to: {}", out_path);
    } else {
        println!("\n--- Signal CSV ---");
        print!("{}", csv_content);
    }

    Ok(())
}

fn signals_to_csv(signals: &[OrderFlowSignal]) -> String {
    let mut out = String::new();
    out.push_str(OrderFlowSignal::csv_header());
    out.push('\n');
    for sig in signals {
        out.push_str(&sig.to_csv_row());
        out.push('\n');
    }
    out
}

fn print_summary(signals: &[OrderFlowSignal]) {
    if signals.is_empty() {
        return;
    }
    let n = signals.len() as f64;
    let mean_ofi = signals.iter().map(|s| s.ofi).sum::<f64>() / n;
    let mean_vpin = signals.iter().map(|s| s.vpin).sum::<f64>() / n;
    let informed_count = signals
        .iter()
        .filter(|s| {
            s.regime == order_flow_engine::signal::OrderFlowRegime::InformedTradingDetected
        })
        .count();
    let divergence_count = signals.iter().filter(|s| s.delta_divergence).count();
    let filtered_count = signals.iter().filter(|s| s.filter_entry).count();

    println!("\n── Order Flow Summary ──────────────────────────────");
    println!("  Bars processed       : {}", signals.len());
    println!("  Mean OFI             : {:+.4}", mean_ofi);
    println!("  Mean VPIN            : {:.4}", mean_vpin);
    println!("  Informed trading bars: {} ({:.1}%)", informed_count, informed_count as f64 / n * 100.0);
    println!("  Divergence signals   : {}", divergence_count);
    println!("  Entry filters (VPIN) : {}", filtered_count);
    println!("────────────────────────────────────────────────────\n");
}

/// Generate synthetic OHLCV data for demo/testing.
fn generate_synthetic_ohlcv(n: usize, seed: u64) -> Vec<(String, f64, f64, f64, f64, f64)> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut price = 100.0f64;
    let mut rows = Vec::with_capacity(n);

    for i in 0..n {
        let ret: f64 = rng.gen_range(-0.015..0.015);
        let open = price;
        price *= 1.0 + ret;
        let close = price;
        let range: f64 = rng.gen_range(0.005..0.03) * open;
        let high = open.max(close) + range * 0.4;
        let low = open.min(close) - range * 0.4;

        // Occasionally inject volume spikes
        let volume = if rng.gen_bool(0.05) {
            rng.gen_range(50_000.0..200_000.0)
        } else {
            rng.gen_range(1_000.0..15_000.0)
        };

        let timestamp = format!("2024-01-01T{:02}:{:02}:00", (i / 60) % 24, i % 60);
        rows.push((timestamp, open, high, low, close, volume));
    }
    rows
}
