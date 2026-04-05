//! counterfactual-engine CLI
//!
//! # Usage
//!
//! ```text
//! counterfactual-engine sample --n 100 --method lhs --output samples.json
//! counterfactual-engine sensitivity --results results.json --output sensitivity.json
//! counterfactual-engine neighborhood --center center.json --n 50 --radius 0.15 --output samples.json
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::Value;

use counterfactual_engine::{
    load_results_from_file, write_samples_to_file, write_sensitivity_to_file,
    LatinHypercubeSampler, NeighborhoodSampler, ParameterBounds, ParameterSample, Sampler,
    SobolSampler,
};
use counterfactual_engine::sensitivity::build_report;

// ---------------------------------------------------------------------------
// CLI definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(
    name = "counterfactual-engine",
    version = "0.1.0",
    about = "Fast parameter sweeping and sensitivity analysis for the IAE Counterfactual Oracle",
    long_about = None,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level (repeat for more: -v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Generate a parameter sample set
    Sample(SampleArgs),
    /// Run sensitivity analysis on a results file
    Sensitivity(SensitivityArgs),
    /// Generate a neighborhood sample around a center genome
    Neighborhood(NeighborhoodArgs),
}

// ---------------------------------------------------------------------------
// sample sub-command
// ---------------------------------------------------------------------------

#[derive(Debug, ValueEnum, Clone)]
enum SampleMethod {
    Lhs,
    Sobol,
}

#[derive(Debug, Parser)]
struct SampleArgs {
    /// Number of samples to generate
    #[arg(short, long, default_value = "100")]
    n: usize,

    /// Sampling method (lhs | sobol)
    #[arg(short, long, default_value = "lhs", value_enum)]
    method: SampleMethod,

    /// Output JSON file path
    #[arg(short, long, default_value = "samples.json")]
    output: PathBuf,

    /// RNG seed (lhs only)
    #[arg(short, long)]
    seed: Option<u64>,

    /// Optional JSON file with custom parameter bounds
    /// Format: [{"name": "bh_form", "min": 1.70, "max": 1.98, "integer": false}, ...]
    #[arg(long)]
    bounds: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// sensitivity sub-command
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
struct SensitivityArgs {
    /// Input JSON file of simulation results (array of SimResult)
    #[arg(short, long)]
    results: PathBuf,

    /// Output JSON file for the sensitivity report
    #[arg(short, long, default_value = "sensitivity.json")]
    output: PathBuf,

    /// Comma-separated list of parameter names to analyse.
    /// Defaults to all 15 genome parameters.
    #[arg(long)]
    params: Option<String>,

    /// Number of pairs for Morris screening
    #[arg(long, default_value = "500")]
    morris_pairs: usize,
}

// ---------------------------------------------------------------------------
// neighborhood sub-command
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
struct NeighborhoodArgs {
    /// JSON file with center genome params (dict of param -> value)
    #[arg(short, long)]
    center: PathBuf,

    /// Number of samples
    #[arg(short, long, default_value = "50")]
    n: usize,

    /// Perturbation radius as fraction of range (default 0.15)
    #[arg(short, long, default_value = "0.15")]
    radius: f64,

    /// Output file
    #[arg(short, long, default_value = "samples.json")]
    output: PathBuf,

    /// RNG seed
    #[arg(short, long)]
    seed: Option<u64>,

    /// Optional custom bounds file
    #[arg(long)]
    bounds: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Simple stderr logging
    if cli.verbose > 0 {
        eprintln!("[counterfactual-engine] verbose={}", cli.verbose);
    }

    match cli.command {
        Commands::Sample(args) => run_sample(args, cli.verbose),
        Commands::Sensitivity(args) => run_sensitivity(args, cli.verbose),
        Commands::Neighborhood(args) => run_neighborhood(args, cli.verbose),
    }
}

// ---------------------------------------------------------------------------
// sample handler
// ---------------------------------------------------------------------------

fn run_sample(args: SampleArgs, verbose: u8) -> Result<()> {
    let bounds = load_bounds(args.bounds.as_deref())?;

    if verbose > 0 {
        eprintln!(
            "[sample] method={:?}, n={}, bounds_dim={}, output={}",
            args.method,
            args.n,
            bounds.dim(),
            args.output.display()
        );
    }

    let samples: Vec<ParameterSample> = match args.method {
        SampleMethod::Lhs => {
            let mut sampler = LatinHypercubeSampler::new(bounds, args.seed);
            sampler.sample(args.n)
        }
        SampleMethod::Sobol => {
            let mut sampler = SobolSampler::new(bounds);
            sampler.sample(args.n)
        }
    };

    write_samples_to_file(&samples, &args.output)
        .with_context(|| format!("writing samples to {}", args.output.display()))?;

    eprintln!(
        "Generated {} samples → {}",
        samples.len(),
        args.output.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// sensitivity handler
// ---------------------------------------------------------------------------

fn run_sensitivity(args: SensitivityArgs, verbose: u8) -> Result<()> {
    let results = load_results_from_file(&args.results)
        .with_context(|| format!("loading results from {}", args.results.display()))?;

    if verbose > 0 {
        eprintln!(
            "[sensitivity] n_results={}, output={}",
            results.len(),
            args.output.display()
        );
    }

    if results.is_empty() {
        bail!("Results file is empty — nothing to analyse.");
    }

    // Determine parameter names
    let param_names_owned: Vec<String> = if let Some(p) = args.params {
        p.split(',').map(|s| s.trim().to_owned()).collect()
    } else {
        // Use keys from first result, fall back to genome defaults
        let first_keys: Vec<String> = results[0].params.keys().cloned().collect();
        if first_keys.is_empty() {
            ParameterBounds::genome_defaults()
                .param_names()
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            first_keys
        }
    };

    let param_names_ref: Vec<&str> = param_names_owned.iter().map(String::as_str).collect();
    let report = build_report(&results, &param_names_ref);

    write_sensitivity_to_file(&report, &args.output)
        .with_context(|| format!("writing report to {}", args.output.display()))?;

    // Print summary to stderr
    eprintln!("Sensitivity report ({} samples):", report.n_samples);
    eprintln!("  Top parameters by Sobol ST:");
    for (i, idx) in report.sobol.iter().take(5).enumerate() {
        eprintln!("    {}. {} — ST={:.4}, S1={:.4}", i + 1, idx.param, idx.st, idx.s1);
    }
    eprintln!("  Report → {}", args.output.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// neighborhood handler
// ---------------------------------------------------------------------------

fn run_neighborhood(args: NeighborhoodArgs, verbose: u8) -> Result<()> {
    let center_text = std::fs::read_to_string(&args.center)
        .with_context(|| format!("reading center file {}", args.center.display()))?;
    let center_value: Value = serde_json::from_str(&center_text)?;

    let center_params: HashMap<String, f64> = match center_value {
        Value::Object(map) => map
            .into_iter()
            .filter_map(|(k, v)| v.as_f64().map(|f| (k, f)))
            .collect(),
        _ => bail!("Center file must be a JSON object mapping param names to numbers"),
    };

    let bounds = load_bounds(args.bounds.as_deref())?;

    if verbose > 0 {
        eprintln!(
            "[neighborhood] n={}, radius={:.3}, center_params={}, output={}",
            args.n,
            args.radius,
            center_params.len(),
            args.output.display()
        );
    }

    let mut sampler = NeighborhoodSampler::new(bounds, &center_params, args.radius, args.seed);
    let samples = sampler.sample(args.n);

    write_samples_to_file(&samples, &args.output)
        .with_context(|| format!("writing samples to {}", args.output.display()))?;

    eprintln!(
        "Generated {} neighborhood samples → {}",
        samples.len(),
        args.output.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Bounds loading helper
// ---------------------------------------------------------------------------

fn load_bounds(path: Option<&std::path::Path>) -> Result<ParameterBounds> {
    match path {
        None => Ok(ParameterBounds::genome_defaults()),
        Some(p) => {
            use counterfactual_engine::sampler::ParamBound;
            let text = std::fs::read_to_string(p)
                .with_context(|| format!("reading bounds file {}", p.display()))?;
            let raw: Vec<Value> = serde_json::from_str(&text)?;
            let mut bounds = Vec::new();
            for v in raw {
                let name = v["name"].as_str()
                    .ok_or_else(|| anyhow::anyhow!("bounds entry missing 'name'"))?
                    .to_owned();
                let min = v["min"].as_f64()
                    .ok_or_else(|| anyhow::anyhow!("bounds entry missing 'min' for {name}"))?;
                let max = v["max"].as_f64()
                    .ok_or_else(|| anyhow::anyhow!("bounds entry missing 'max' for {name}"))?;
                let integer = v["integer"].as_bool().unwrap_or(false);
                let mut b = ParamBound::new(&name, min, max);
                b.integer = integer;
                bounds.push(b);
            }
            Ok(ParameterBounds::new(bounds))
        }
    }
}
