//! CLI entry point for the online-learning crate.
//!
//! Usage:
//!   online-learning train --data data.csv --model ftrl --output model.json [OPTIONS]
//!   online-learning eval  --data data.csv --model-file model.json
//!
//! Supported models: ftrl, passive_aggressive, sgd_vanilla, sgd_momentum, sgd_adam

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use online_learning::{
    ftrl::{FtrlBuilder},
    passive_aggressive::PaBuilder,
    sgd::{SgdBuilder, SgdMode},
    train_online, load_csv, normalize_samples, ModelState, OnlineLearner,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "online-learning",
    version = "0.1.0",
    about = "Online learning algorithms for SRFM-Lab (FTRL, Passive-Aggressive, SGD)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Train an online model on a CSV dataset and save weights to JSON.
    Train(TrainArgs),

    /// Evaluate a saved model on a CSV dataset (no weight updates).
    Eval(EvalArgs),
}

#[derive(Parser, Debug)]
struct TrainArgs {
    /// Path to training CSV file (last column = label).
    #[arg(long)]
    data: String,

    /// Model type: ftrl | passive_aggressive | sgd_vanilla | sgd_momentum | sgd_adam
    #[arg(long, default_value = "ftrl")]
    model: String,

    /// Path to write the trained model JSON.
    #[arg(long, default_value = "model.json")]
    output: String,

    /// Learning rate (alpha for FTRL; lr for SGD).
    #[arg(long, default_value_t = 0.1)]
    lr: f64,

    /// L1 regularisation (FTRL only).
    #[arg(long, default_value_t = 0.0)]
    lambda1: f64,

    /// L2 regularisation (FTRL / SGD).
    #[arg(long, default_value_t = 0.0)]
    lambda2: f64,

    /// Aggressiveness C (Passive-Aggressive only).
    #[arg(long, default_value_t = 1.0)]
    pa_c: f64,

    /// Epsilon insensitivity (Passive-Aggressive only).
    #[arg(long, default_value_t = 0.1)]
    pa_epsilon: f64,

    /// Mini-batch size (SGD only).
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Z-score normalise features before training.
    #[arg(long, default_value_t = false)]
    normalize: bool,

    /// Log loss every N samples (0 = no logging).
    #[arg(long, default_value_t = 1000)]
    log_every: usize,
}

#[derive(Parser, Debug)]
struct EvalArgs {
    /// Path to evaluation CSV file.
    #[arg(long)]
    data: String,

    /// Path to saved model JSON.
    #[arg(long)]
    model_file: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Train(args) => run_train(args),
        Command::Eval(args)  => run_eval(args),
    }
}

// ---------------------------------------------------------------------------
// Train command
// ---------------------------------------------------------------------------

fn run_train(args: TrainArgs) -> Result<()> {
    println!("Loading data from: {}", args.data);
    let mut samples = load_csv(&args.data)
        .with_context(|| format!("Failed to load CSV: {}", args.data))?;

    if samples.is_empty() {
        anyhow::bail!("CSV file contains no samples");
    }

    let n_features = samples[0].features.len();
    println!(
        "Loaded {} samples with {} features",
        samples.len(), n_features
    );

    if args.normalize {
        let (means, stds) = normalize_samples(&mut samples);
        println!(
            "Normalised features. Mean[0]={:.4}, Std[0]={:.4}",
            means.first().unwrap_or(&0.0),
            stds.first().unwrap_or(&1.0)
        );
    }

    let model_name = args.model.to_lowercase();

    // Build the appropriate model and run training
    let state = match model_name.as_str() {
        "ftrl" => {
            let mut model = FtrlBuilder::default()
                .alpha(args.lr)
                .lambda1(args.lambda1)
                .lambda2(args.lambda2)
                .n_features(n_features)
                .build();
            let (_errors, metrics) = train_online(&mut model, &samples, args.log_every);
            print_metrics("FTRL-Proximal", &metrics);
            build_state(&model, "ftrl", &args)
        }

        "passive_aggressive" | "pa" => {
            let mut model = PaBuilder::default()
                .c(args.pa_c)
                .epsilon(args.pa_epsilon)
                .n_features(n_features)
                .build();
            let (_errors, metrics) = train_online(&mut model, &samples, args.log_every);
            print_metrics("Passive-Aggressive II", &metrics);
            build_state(&model, "passive_aggressive", &args)
        }

        "sgd_vanilla" => {
            let mut model = SgdBuilder::default()
                .lr(args.lr)
                .mode(SgdMode::Vanilla)
                .l2(args.lambda2)
                .batch_size(args.batch_size)
                .n_features(n_features)
                .build();
            let (_errors, metrics) = train_online(&mut model, &samples, args.log_every);
            print_metrics("SGD Vanilla", &metrics);
            build_state(&model, "sgd_vanilla", &args)
        }

        "sgd_momentum" => {
            let mut model = SgdBuilder::default()
                .lr(args.lr)
                .mode(SgdMode::Momentum)
                .l2(args.lambda2)
                .batch_size(args.batch_size)
                .n_features(n_features)
                .build();
            let (_errors, metrics) = train_online(&mut model, &samples, args.log_every);
            print_metrics("SGD Momentum", &metrics);
            build_state(&model, "sgd_momentum", &args)
        }

        "sgd_adam" | "sgd" => {
            let mut model = SgdBuilder::default()
                .lr(args.lr)
                .mode(SgdMode::Adam)
                .l2(args.lambda2)
                .batch_size(args.batch_size)
                .n_features(n_features)
                .build();
            let (_errors, metrics) = train_online(&mut model, &samples, args.log_every);
            print_metrics("SGD Adam", &metrics);
            build_state(&model, "sgd_adam", &args)
        }

        other => {
            anyhow::bail!(
                "Unknown model '{}'. Choose from: ftrl, passive_aggressive, sgd_vanilla, sgd_momentum, sgd_adam",
                other
            );
        }
    };

    state.save_to_file(&args.output)
        .with_context(|| format!("Failed to write model to: {}", args.output))?;

    println!("Model saved to: {}", args.output);
    Ok(())
}

// ---------------------------------------------------------------------------
// Eval command
// ---------------------------------------------------------------------------

fn run_eval(args: EvalArgs) -> Result<()> {
    println!("Loading model from: {}", args.model_file);
    let state = ModelState::load_from_file(&args.model_file)
        .with_context(|| format!("Failed to load model: {}", args.model_file))?;

    println!("Algorithm: {}, n_seen: {}", state.algorithm, state.n_seen);

    println!("Loading eval data from: {}", args.data);
    let samples = load_csv(&args.data)
        .with_context(|| format!("Failed to load CSV: {}", args.data))?;

    // Reconstruct a model with the saved weights using a simple wrapper
    let weights = state.weights.clone();
    let bias = state.bias;
    let n_features = weights.len();

    // Evaluate by predicting from saved weights (no training)
    let errors: Vec<f64> = samples
        .iter()
        .map(|s| {
            let n = s.features.len().min(n_features);
            let pred: f64 = weights[..n].iter().zip(s.features[..n].iter()).map(|(w, x)| w * x).sum::<f64>() + bias;
            s.label - pred
        })
        .collect();

    let metrics = online_learning::TrainingMetrics::compute_from_errors(&errors);
    println!("Eval results on {} samples:", samples.len());
    println!("  MAE  = {:.6}", metrics.mean_absolute_error);
    println!("  RMSE = {:.6}", metrics.root_mean_squared_error);

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_state<L: OnlineLearner>(learner: &L, algorithm: &str, args: &TrainArgs) -> ModelState {
    let mut hyperparams = HashMap::new();
    hyperparams.insert("lr".to_string(), args.lr);
    hyperparams.insert("lambda1".to_string(), args.lambda1);
    hyperparams.insert("lambda2".to_string(), args.lambda2);
    hyperparams.insert("pa_c".to_string(), args.pa_c);
    hyperparams.insert("pa_epsilon".to_string(), args.pa_epsilon);
    hyperparams.insert("batch_size".to_string(), args.batch_size as f64);

    ModelState {
        algorithm: algorithm.to_string(),
        weights: learner.weights().to_vec(),
        bias: learner.bias(),
        n_seen: learner.n_seen(),
        hyperparams,
        metrics: online_learning::TrainingMetrics::default(),
        saved_at: chrono::Utc::now().to_rfc3339(),
    }
}

fn print_metrics(name: &str, metrics: &online_learning::TrainingMetrics) {
    println!(
        "{} | n={} | MAE={:.6} | RMSE={:.6}",
        name, metrics.n_updates, metrics.mean_absolute_error, metrics.root_mean_squared_error
    );
}
