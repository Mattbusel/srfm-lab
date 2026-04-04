# Spacetime Arena API Reference

Base URL: `http://localhost:8000`

---

## `GET /api/instruments`

Returns all configured instruments with their BH parameters.

**Response**:
```json
[
  {
    "symbol": "ES",
    "name": "S&P 500 E-mini",
    "asset_class": "equity_index",
    "cf_15m": 0.0003,
    "cf_1h": 0.001,
    "cf_1d": 0.005,
    "bh_form": 1.5
  }
]
```

---

## `POST /api/backtest`

Run a BH backtest on a single instrument.

**Request body**:
```json
{
  "instrument": "ES",
  "start": "2020-01-01",
  "end": "2024-01-01",
  "timeframe": "1d",
  "bh_form": 1.5,
  "bh_decay": 0.95,
  "min_tf_score": 2,
  "pos_floor": 0.0,
  "initial_equity": 100000,
  "include_15m": true,
  "include_1h": true,
  "include_1d": true
}
```

**Response**:
```json
{
  "run_id": "bt_20240101_143022",
  "summary": {
    "total_return_pct": 287.4,
    "cagr": 0.402,
    "sharpe": 1.84,
    "sortino": 2.31,
    "max_drawdown_pct": 0.218,
    "calmar": 1.84,
    "win_rate": 0.58,
    "profit_factor": 2.3,
    "n_trades": 412
  },
  "equity_curve": [[timestamp, equity], ...],
  "trades": [...],
  "bh_timeline": [...]
}
```

---

## `POST /api/mc`

Run Monte Carlo simulation on a completed backtest.

**Request body**:
```json
{
  "run_id": "bt_20240101_143022",
  "n_sims": 1000,
  "regime_aware": true,
  "ar1_rho": null
}
```

**Response**:
```json
{
  "percentiles": {
    "p05": [...equity_curve...],
    "p25": [...],
    "p50": [...],
    "p75": [...],
    "p95": [...]
  },
  "blowup_probability": 0.003,
  "terminal_equity_distribution": {
    "p05": 82000,
    "p25": 118000,
    "p50": 156000,
    "p75": 210000,
    "p95": 340000
  },
  "max_drawdown_distribution": {
    "p05": 0.08,
    "p50": 0.18,
    "p95": 0.38
  }
}
```

---

## `POST /api/sensitivity`

Compute Sharpe sensitivity to parameter perturbations.

**Request body**:
```json
{
  "instrument": "ES",
  "start": "2020-01-01",
  "end": "2024-01-01",
  "base_params": {"bh_form": 1.5, "min_tf_score": 2},
  "params": ["bh_form", "bh_decay", "min_tf_score"],
  "n_perturbations": 20
}
```

**Response**:
```json
{
  "surfaces": {
    "bh_form": {
      "values": [1.0, 1.25, 1.5, 1.75, 2.0, ...],
      "sharpe": [1.62, 1.71, 1.84, 1.79, 1.68, ...],
      "cagr": [...],
      "max_dd": [...]
    }
  },
  "tornado": [
    {"param": "bh_form", "low": 1.62, "high": 1.79, "base": 1.84},
    {"param": "min_tf_score", "low": 1.55, "high": 1.81, "base": 1.84}
  ]
}
```

---

## `GET /api/correlation`

Get current BH activation correlations between all instruments.

**Query params**: `timeframe=1d` (default), `lookback=252`

**Response**:
```json
{
  "matrix": {
    "ES": {"ES": 1.0, "NQ": 0.65, "CL": 0.18, "GC": 0.12},
    "NQ": {"ES": 0.65, "NQ": 1.0, ...}
  },
  "computed_at": "2024-01-15T12:00:00Z"
}
```

---

## `GET /api/trades`

List trades with filtering.

**Query params**: `run_id`, `instrument`, `regime`, `min_tf_score`, `limit=100`, `offset=0`

**Response**: Array of trade objects (see schema `03_trades.sql` for full field list).

---

## `POST /api/archaeology`

Find historical patterns matching current BH state.

**Request body**:
```json
{
  "instrument": "ES",
  "tf_score": 7,
  "direction": 1,
  "regime": "BULL",
  "lookback_years": 5,
  "n_matches": 20
}
```

**Response**:
```json
{
  "matches": [
    {
      "timestamp": "2021-11-05T00:00:00Z",
      "mass_1d": 2.45,
      "tf_score": 7,
      "subsequent_5d_return": 0.023,
      "subsequent_10d_return": 0.041
    }
  ],
  "statistics": {
    "median_5d_return": 0.018,
    "win_rate_5d": 0.72,
    "median_10d_return": 0.031
  }
}
```

---

## `POST /api/report`

Generate a full HTML report for a backtest run.

**Request body**: `{ "run_id": "bt_...", "include_mc": true, "include_sensitivity": true }`

**Response**: JSON with `report_url` pointing to the generated HTML file.

---

## `WebSocket /ws/live`

Live BH state stream. Connect once; receive updates as bars close.

**Message format**:
```json
{
  "type": "bar_close",
  "timestamp": "2024-01-15T14:30:00Z",
  "instrument": "ES",
  "bh_state": {
    "mass_1d": 2.34, "mass_1h": 1.89, "mass_15m": 0.45,
    "active_1d": true, "active_1h": true, "active_15m": false,
    "tf_score": 6, "regime": "BULL"
  }
}
```

---

## `WebSocket /ws/replay`

Bar-by-bar replay of a backtest. Useful for reviewing entries/exits interactively.

**Connection params**: `?run_id=bt_...&speed=10` (speed = bars per second)

**Message format**: Same as `/ws/live` but sourced from historical data.
