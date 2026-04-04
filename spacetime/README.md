# Spacetime Arena

SRFM research, backtesting, and live monitoring platform built on LARSA v16 BH physics.

## Install

```bash
cd spacetime
pip install -r requirements.txt
```

## Run the API server

```bash
cd spacetime
python -m uvicorn api.main:app --host 0.0.0.0 --port 8765 --reload
```

API docs available at `http://localhost:8765/docs`

## Run a backtest from CLI

```python
from spacetime.engine.data_loader import load_yfinance
from spacetime.engine.bh_engine import run_backtest

df = load_yfinance("SPY", "2020-01-01", "2024-12-31", interval="1h")
result = run_backtest("ES", df, long_only=True)

print(result.stats)
print(f"Trades: {len(result.trades)}")
```

Or using existing CSV data:

```python
from spacetime.engine.data_loader import load_csv
from spacetime.engine.bh_engine import run_backtest

df = load_csv("data/ES_hourly_real.csv")
result = run_backtest("ES", df)
print(result.stats)
```

## Key API endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/instruments`  | List all instrument configs |
| POST | `/api/backtest`     | Run BH backtest |
| POST | `/api/mc`           | Monte Carlo simulation |
| POST | `/api/sensitivity`  | Parameter sensitivity analysis |
| GET  | `/api/correlation`  | BH activation correlation matrix |
| GET  | `/api/trades`       | Query archaeology DB |
| POST | `/api/archaeology`  | Import QC trade CSV into DB |
| POST | `/api/report`       | Generate PDF research report |
| WS   | `/ws/live`          | Stream live trader state |
| WS   | `/ws/replay`        | Bar-by-bar signal replay |

## Directory structure

```
spacetime/
  engine/
    bh_engine.py      Universal BH backtester
    data_loader.py    yfinance / Alpaca / CSV ingestion
    mc.py             Regime-aware Monte Carlo
    sensitivity.py    Parameter sensitivity analyzer
    correlation.py    BH activation correlation matrix
    archaeology.py    QC trade CSV → SQLite DB
    replay.py         Bar-by-bar signal replay engine
  api/
    main.py           FastAPI server
  reports/
    generator.py      PDF report generator (ReportLab)
  cache/              Parquet data cache + live_state.json
  db/
    trades.db         SQLite archaeology database
  logs/
    api.log           API server logs
```
