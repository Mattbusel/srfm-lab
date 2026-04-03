//! srfm-stream: Real-time SRFM state computation from stdin price ticks.
//!
//! Reads price ticks from stdin (CSV or JSON), outputs SRFM state per tick.
//! Universal adapter — pipe to anything.
//!
//! Usage:
//!   echo "4502.25,4505.50,4501.00,4503.75,12345" | srfm-stream
//!   cat prices.csv | srfm-stream --cf 0.005 --format json
//!   cat prices.csv | srfm-stream | jq '.bh_active'

use clap::Parser;
use std::io::{self, BufRead, Write};

#[derive(Parser, Debug)]
#[command(name = "srfm-stream", about = "Real-time SRFM state from stdin ticks")]
struct Args {
    /// Critical fraction (BH formation threshold scale)
    #[arg(long, default_value = "0.005")]
    cf: f64,

    /// BH formation mass threshold
    #[arg(long = "bh-form", default_value = "1.5")]
    bh_form: f64,

    /// BH decay factor (spacelike)
    #[arg(long = "bh-decay", default_value = "0.95")]
    bh_decay: f64,

    /// Maximum leverage
    #[arg(long = "max-lev", default_value = "0.65")]
    max_lev: f64,

    /// Output format: json | tsv | compact
    #[arg(long, default_value = "json")]
    format: String,

    /// Only output when BH state changes
    #[arg(long)]
    quiet: bool,

    /// Replay a CSV file (path) instead of stdin
    #[arg(long)]
    replay: Option<String>,
}

struct SRFMEngine {
    cf: f64,
    bh_form: f64,
    bh_decay: f64,
    max_lev: f64,
    mass: f64,
    ctl: usize,
    prev_close: Option<f64>,
    equity: f64,
    bar: usize,
}

#[derive(Debug, Clone)]
struct SRFMTick {
    bar: usize,
    close: f64,
    beta: f64,
    bit: &'static str,
    bh_mass: f64,
    bh_active: bool,
    ctl: usize,
    position: f64,
    equity: f64,
}

impl SRFMEngine {
    fn new(cf: f64, bh_form: f64, bh_decay: f64, max_lev: f64) -> Self {
        SRFMEngine {
            cf,
            bh_form,
            bh_decay,
            max_lev,
            mass: 0.0,
            ctl: 0,
            prev_close: None,
            equity: 1.0,
            bar: 0,
        }
    }

    fn step(&mut self, close: f64, ema20: f64) -> SRFMTick {
        self.bar += 1;

        let (beta, bit) = if let Some(prev) = self.prev_close {
            let b = (close - prev).abs() / (prev * self.cf + 1e-12);
            if b < 1.0 {
                self.mass = self.mass * 0.97 + 0.03;
                self.ctl += 1;
                (b, "TIMELIKE")
            } else {
                self.mass *= self.bh_decay;
                self.ctl = 0;
                (b, "SPACELIKE")
            }
        } else {
            (0.0, "INIT")
        };

        let bh_active = self.mass >= self.bh_form && self.ctl >= 5;
        let direction = if close >= ema20 { 1.0 } else { -1.0 };
        let position = if bh_active {
            self.max_lev * direction
        } else if self.ctl >= 3 {
            self.max_lev * 0.5 * direction
        } else {
            0.0
        };

        // Update equity
        if let Some(prev) = self.prev_close {
            let ret = close / prev - 1.0;
            self.equity *= 1.0 + position * ret;
        }

        let tick = SRFMTick {
            bar: self.bar,
            close,
            beta,
            bit,
            bh_mass: self.mass,
            bh_active,
            ctl: self.ctl,
            position,
            equity: self.equity,
        };

        self.prev_close = Some(close);
        tick
    }
}

fn parse_line(line: &str) -> Option<f64> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    // Try JSON first: {"close": 4503.75, ...}
    if line.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(c) = v.get("close").and_then(|x| x.as_f64()) {
                return Some(c);
            }
        }
        return None;
    }

    // CSV: open,high,low,close,volume — take index 3 (close)
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() >= 4 {
        if let Ok(f) = fields[3].trim().parse::<f64>() {
            return Some(f);
        }
    }
    // Single value
    if let Ok(f) = line.parse::<f64>() {
        return Some(f);
    }

    None
}

fn format_tick(tick: &SRFMTick, fmt: &str) -> String {
    match fmt {
        "tsv" => format!(
            "{}\t{:.4}\t{:.4}\t{}\t{:.4}\t{}\t{}\t{:.4}\t{:.6}",
            tick.bar,
            tick.close,
            tick.beta,
            tick.bit,
            tick.bh_mass,
            tick.bh_active,
            tick.ctl,
            tick.position,
            tick.equity
        ),
        "compact" => format!(
            "bar={} c={:.2} β={:.3} {} m={:.3} BH={} ctl={} pos={:.3} eq={:.4}",
            tick.bar,
            tick.close,
            tick.beta,
            tick.bit,
            tick.bh_mass,
            if tick.bh_active { "ON " } else { "off" },
            tick.ctl,
            tick.position,
            tick.equity
        ),
        _ => {
            // json (default)
            serde_json::json!({
                "bar": tick.bar,
                "close": tick.close,
                "beta": tick.beta,
                "bit": tick.bit,
                "bh_mass": tick.bh_mass,
                "bh_active": tick.bh_active,
                "ctl": tick.ctl,
                "position": tick.position,
                "equity": tick.equity,
            })
            .to_string()
        }
    }
}

/// Compute running EMA-20 given history and current close
struct Ema20 {
    value: f64,
    k: f64,
    initialized: bool,
}

impl Ema20 {
    fn new() -> Self {
        Ema20 {
            value: 0.0,
            k: 2.0 / 21.0,
            initialized: false,
        }
    }
    fn update(&mut self, close: f64) -> f64 {
        if !self.initialized {
            self.value = close;
            self.initialized = true;
        } else {
            self.value = close * self.k + self.value * (1.0 - self.k);
        }
        self.value
    }
}

fn process_lines<R: BufRead>(
    reader: R,
    engine: &mut SRFMEngine,
    ema: &mut Ema20,
    fmt: &str,
    quiet: bool,
) {
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    let mut prev_bh = false;

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => break,
        };

        // Skip header lines
        if line.trim().to_lowercase().contains("close") && line.contains(',') {
            continue;
        }

        if let Some(close) = parse_line(&line) {
            let ema_val = ema.update(close);
            let tick = engine.step(close, ema_val);

            let should_print = if quiet {
                tick.bh_active != prev_bh
            } else {
                true
            };

            if should_print {
                let s = format_tick(&tick, fmt);
                writeln!(out, "{}", s).ok();
            }
            prev_bh = tick.bh_active;
        }
    }
}

fn main() {
    let args = Args::parse();

    let mut engine = SRFMEngine::new(args.cf, args.bh_form, args.bh_decay, args.max_lev);
    let mut ema = Ema20::new();

    if let Some(ref path) = args.replay {
        // Replay mode: read from file
        let file = std::fs::File::open(path).expect("Cannot open replay file");
        let reader = io::BufReader::new(file);
        process_lines(reader, &mut engine, &mut ema, &args.format, args.quiet);
    } else {
        // Stdin mode
        let stdin = io::stdin();
        let reader = stdin.lock();
        process_lines(reader, &mut engine, &mut ema, &args.format, args.quiet);
    }
}
