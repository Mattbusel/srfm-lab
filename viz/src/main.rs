mod spacetime;
mod wells;
mod experiments;
mod equity;
mod convergence;
mod diff;
mod snap;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "srfm-viz", about = "SRFM Trading Strategy Visualization Tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Render spacetime diagram from OHLCV CSV
    Spacetime {
        #[arg(long)]
        csv: String,
        #[arg(long, default_value = "0.005")]
        cf: f64,
        #[arg(long)]
        out: String,
    },
    /// Render GitHub-style calendar heatmap of wells
    Wells {
        #[arg(long)]
        json: String,
        #[arg(long)]
        out: String,
    },
    /// Render experiments comparison chart
    Experiments {
        #[arg(long)]
        json: String,
        #[arg(long)]
        out: String,
    },
    /// Render equity curve chart
    Equity {
        #[arg(long)]
        json: String,
        #[arg(long)]
        out: String,
    },
    /// Render convergence grouped bar chart
    Convergence {
        #[arg(long)]
        json: String,
        #[arg(long)]
        out: String,
    },
    /// Semantic diff between two strategy Python files
    Diff {
        #[arg(help = "First strategy file")]
        file_a: String,
        #[arg(help = "Second strategy file")]
        file_b: String,
    },
    /// Snapshot current strategy state from a results JSON (single-line output)
    Snap {
        #[arg(long, help = "JSON file to snapshot")]
        json: String,
        #[arg(long, default_value = "auto", help = "Snapshot type: experiments|backtest|auto")]
        r#type: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Spacetime { csv, cf, out } => spacetime::run(&csv, cf, &out)?,
        Commands::Wells { json, out } => wells::run(&json, &out)?,
        Commands::Experiments { json, out } => experiments::run(&json, &out)?,
        Commands::Equity { json, out } => equity::run(&json, &out)?,
        Commands::Convergence { json, out } => convergence::run(&json, &out)?,
        Commands::Diff { file_a, file_b } => diff::run_diff(&file_a, &file_b)?,
        Commands::Snap { json, r#type } => snap::run_snap(&json, &r#type)?,
    }
    Ok(())
}
