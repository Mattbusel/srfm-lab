# Quick Start -- Get Running in 10 Minutes

## Prerequisites

- Python 3.11+
- Node.js 18+
- Go 1.21+
- Rust 1.75+
- Docker + Docker Compose (optional, for PostgreSQL)

---

## Step 1: Clone and Install Dependencies

```bash
# From srfm-lab/
make install          # Python deps, Rust build, Node deps

# Or manually:
pip install -r requirements.txt
cd spacetime/web && npm install && cd ../..
cd terminal && npm install && cd ..
cargo build --release
```

## Step 2: Fetch Market Data

```bash
# Set your Polygon.io API key
export POLYGON_API_KEY=your_key_here

# Fetch daily bars for the 8 core instruments
python scripts/fetch_polygon.py --symbols ES NQ CL GC ZB NG VX --timeframe 1d --start 2020-01-01

# Or fetch all configured instruments
python scripts/fetch_polygon.py --all --timeframe 1d
```

Data is saved to `data/bars/1d/{symbol}/YYYY-MM.parquet`.

## Step 3: Run a Backtest

```bash
# Run LARSA v16 backtest (2020-2024, 8 instruments)
make backtest s=larsa-v16

# Or use the Spacetime Arena UI:
make run-api          # Start FastAPI on port 8000
make run-terminal     # Start Arena UI on port 5173
# Open http://localhost:5173 → Select instruments → Run Backtest
```

## Step 4: View Results

```bash
# Compare all runs for a strategy
make compare s=larsa-v16

# View HTML report (opens browser)
open results/larsa-v16/*/report.html
```

## Step 5: Run Monte Carlo

```bash
# Via CLI (quickest)
python -m spacetime.engine.mc --run-id 1 --n-sims 1000

# Via UI: In Arena, after a backtest, click "Run MC" in the results panel
```

## Step 6: Start Live Paper Trading

```bash
# Set Alpaca paper trading keys
export ALPACA_API_KEY=your_paper_key
export ALPACA_API_SECRET=your_paper_secret
export ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Start the live trader (paper mode)
make run-live         # Runs tools/live_trader_alpaca.py

# Monitor in the terminal
make run-terminal     # Open http://localhost:5174
```

---

## What Each Command Does

| Command | What it runs |
|---------|-------------|
| `make install` | `pip install`, `cargo build`, `npm install` |
| `make backtest s=X` | `lean backtest strategies/X` → `results/X/{ts}/` |
| `make run-api` | `uvicorn spacetime.api.main:app --reload` |
| `make run-terminal` | `cd spacetime/web && npm run dev` |
| `make run-live` | `python tools/live_trader_alpaca.py` |
| `make run-gateway` | `cd cmd/gateway && go run .` |
| `make docs` | Build and serve this documentation |

---

## Troubleshooting

**"Module srfm_core not found"**: Run `cd extensions && maturin develop` to build the Rust extension.

**"No parquet files found"**: Run the data fetch step first. The engine requires at least 100 bars of history.

**"Alpaca connection refused"**: Check that your API keys are set and you're using the paper URL for paper trading.

**"Port 8000 already in use"**: Another FastAPI instance is running. Kill it with `pkill -f uvicorn`.
