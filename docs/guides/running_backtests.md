# Running Backtests

## Overview

SRFM Lab supports two backtesting paths:

1. **LEAN backtests** — Full QuantConnect LEAN engine, supports futures data, realistic fill simulation, used for the primary LARSA strategies.
2. **Spacetime Arena backtests** — Custom Python engine, faster, supports arbitrary parquet data, best for parameter sweeps and research.

---

## LEAN Backtests

### Basic run

```bash
make backtest s=larsa-v16
```

This runs `lean backtest strategies/larsa-v16 --output results/larsa-v16/{timestamp}`.

### With parameter override

LEAN strategies read parameters from a `config.json`. To override:

```bash
# Edit strategies/larsa-v16/config.json temporarily, then:
make backtest s=larsa-v16

# Or use param sweep for a range:
make sweep s=larsa-v16 param=BH_FORM min=1.0 max=2.5 step=0.25
```

### Comparing multiple runs

```bash
make compare s=larsa-v16          # Compare all runs under results/larsa-v16/
make compare2 s1=larsa-v14 s2=larsa-v16   # Side-by-side comparison
```

---

## Spacetime Arena Backtests

### Via the UI

1. Start the API: `make run-api`
2. Start the frontend: `make run-terminal`
3. Navigate to `http://localhost:5173`
4. In the **Backtest** panel:
   - Select instrument(s)
   - Set date range
   - Adjust BH parameters (bh_form, min_tf_score, pos_floor)
   - Click **Run**

### Via the API directly

```bash
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "instrument": "ES",
    "start": "2020-01-01",
    "end": "2024-01-01",
    "timeframe": "1d",
    "bh_form": 1.5,
    "bh_decay": 0.95,
    "min_tf_score": 2,
    "initial_equity": 100000
  }'
```

### Via Python (for batch runs)

```python
import sys
sys.path.insert(0, 'spacetime')
from engine.bh_engine import run_backtest

result = run_backtest(
    instrument='ES',
    start='2020-01-01',
    end='2024-01-01',
    timeframe='1d',
    bh_form=1.5,
    min_tf_score=2,
    initial_equity=100_000,
)
print(result.summary())
print(result.trades.head(20))
```

---

## Parameter Sweep

The `sweep` command runs a grid search over a single parameter:

```bash
make sweep s=larsa-v16 param=BH_FORM min=1.0 max=3.0 step=0.25
```

This generates results for `bh_form ∈ {1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0}` and outputs a summary table + 3D surface plot.

For multi-parameter sweeps:

```python
# tools/param_sweep.py supports --grid flag
python tools/param_sweep.py strategies/larsa-v16 \
    --grid BH_FORM 1.0:2.5:0.5 MIN_TF_SCORE 1:4:1 \
    --output results/sweeps/grid_20240101.csv
```

---

## Sensitivity Analysis

After running a backtest, compute sensitivity to all parameters:

```bash
# Via API
POST /api/sensitivity
{
  "run_id": 1,
  "params": ["bh_form", "bh_decay", "min_tf_score"],
  "n_perturbations": 20
}
```

This varies each parameter ±25% from its base value and computes the resulting Sharpe, CAGR, and max drawdown. The output is a tornado chart showing which parameters matter most.

---

## Understanding Backtest Output

### Equity curve

The equity curve shows the growth of $100,000 (or your `initial_equity`) over the backtest period. Key things to look for:

- **Drawdown depth**: How far does the curve fall from its peak?
- **Drawdown recovery**: How long does it take to recover?
- **Regime labels**: Are drawdowns associated with SIDEWAYS or HIGH_VOL regimes?
- **Smoothness**: Does the curve grow steadily or in lumpy steps?

### Trade table columns

| Column | Description |
|--------|-------------|
| `entry_time` / `exit_time` | UTC timestamps |
| `side` | `long` or `short` |
| `pnl_pct` | Return on invested capital (not portfolio) |
| `tf_score` | 0-7 BH confluence score at entry |
| `regime_at_entry` | BULL / BEAR / SIDEWAYS / HIGH_VOLATILITY |
| `bh_mass_1d_at_entry` | Raw BH mass on daily TF at entry |
| `mfe_pct` | Maximum favorable excursion (peak unrealized gain) |
| `mae_pct` | Maximum adverse excursion (peak unrealized loss) |
| `hold_bars` | Number of bars held |
| `exit_reason` | `bh_collapse`, `regime_change`, `target`, `stop` |

### Performance metrics

| Metric | Target range |
|--------|-------------|
| Sharpe ratio | > 1.5 |
| CAGR | > 20% |
| Max drawdown | < 30% |
| Calmar ratio | > 1.0 |
| Win rate | 50-65% |
| Profit factor | > 1.8 |

---

## Avoiding Overfitting

1. **Use out-of-sample testing**: Never tune parameters on data after 2022. Keep 2023-present as OOS.
2. **Check parameter stability**: Good parameters should be in the middle of a flat region on the sensitivity surface, not at an edge.
3. **Use MC to stress-test**: A strategy that looks good in MC P5 is more robust than one that only looks good in the median.
4. **Check walk-forward consistency**: Run Q15 from `05_queries.sql` to see if performance is consistent across the 4 quarters of the backtest.
5. **Minimum trades**: Require at least 100 trades for any statistical conclusion. With fewer, parameter conclusions are noise.
