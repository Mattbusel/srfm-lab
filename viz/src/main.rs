mod spacetime;
mod wells;
mod experiments;
mod equity;
mod convergence;

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Spacetime { csv, cf, out } => spacetime::run(&csv, cf, &out)?,
        Commands::Wells { json, out } => wells::run(&json, &out)?,
        Commands::Experiments { json, out } => experiments::run(&json, &out)?,
        Commands::Equity { json, out } => equity::run(&json, &out)?,
        Commands::Convergence { json, out } => convergence::run(&json, &out)?,
    }
    Ok(())
}
