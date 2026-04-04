# Python Engine Module Reference

## `spacetime.engine.bh_engine`

The core BH backtesting engine.

### `run_backtest(instrument, start, end, timeframe, **params) -> BacktestResult`

Run a complete BH backtest.

**Parameters**:
- `instrument`: Symbol string (e.g., `"ES"`)
- `start`, `end`: ISO date strings
- `timeframe`: `"15m"`, `"1h"`, or `"1d"`
- `bh_form`: Formation threshold (default 1.5)
- `bh_decay`: Per-bar mass decay (default 0.95)
- `bh_collapse`: Collapse threshold (default 1.0)
- `min_tf_score`: Minimum tf_score for entry (default 2)
- `pos_floor`: Minimum position fraction (default 0.0)
- `initial_equity`: Starting capital (default 100_000)
- `include_1h`, `include_15m`: Whether to use sub-daily TFs (default True)

**Returns**: `BacktestResult` with attributes:
- `.summary`: dict of performance metrics
- `.trades`: pandas DataFrame of all trades
- `.equity_curve`: pandas Series (date → equity)
- `.bh_timeline`: DataFrame of mass, active, regime at each bar

### `BHState`

The core BH state machine.

```python
bh = BHState(cf=0.005, bh_form=1.5, bh_collapse=1.0, decay=0.95)

for log_return in returns:
    bh.update(log_return)
    print(f"mass={bh.mass:.3f} active={bh.active} dir={bh.direction}")
```

---

## `spacetime.engine.mc`

Monte Carlo simulation.

### `run_mc(trades_df, n_sims, initial_equity, regime_aware, ar1_rho) -> MCResult`

**Parameters**:
- `trades_df`: DataFrame with columns `pnl_pct`, `regime_at_entry`
- `n_sims`: Number of simulation paths (default 1000)
- `regime_aware`: Use regime-conditional sampling (default True)
- `ar1_rho`: AR(1) serial correlation coefficient (default: estimated from data)

**Returns**: `MCResult` with:
- `.percentile_paths`: dict `{p05, p25, p50, p75, p95}` → equity curves
- `.blowup_probability`: float
- `.terminal_equity_stats`: dict

---

## `spacetime.engine.sensitivity`

Parameter sensitivity analysis.

### `run_sensitivity(instrument, start, end, base_params, sweep_params) -> SensitivityResult`

Sweeps each parameter in `sweep_params` while holding others fixed. Returns Sharpe surface.

---

## `spacetime.engine.correlation`

BH activation correlation analysis.

### `compute_bh_correlations(symbols, timeframe, lookback_days) -> pd.DataFrame`

Returns pairwise BH activation correlation matrix.

---

## `spacetime.engine.data_loader`

Data loading utilities.

### `load_bars(symbol, timeframe, start, end) -> pd.DataFrame`

Loads OHLCV data from `data/bars/{timeframe}/{symbol}/*.parquet`.
Returns DataFrame with columns: `timestamp, open, high, low, close, volume, log_return`.

### `list_available_instruments(timeframe) -> list[str]`

Returns list of symbols that have parquet data available.
