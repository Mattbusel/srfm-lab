# Strategy Builder Guide

## Overview

The Spacetime Arena strategy builder provides a visual interface for configuring and testing BH-based trading strategies. Access it at `http://localhost:5173` after starting the API and frontend.

---

## Interface Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Spacetime Arena                                [Live] [Research] │
├──────────────┬───────────────────────────────────────────────────┤
│              │                                                    │
│  Instruments │         Main Chart Area                           │
│  ──────────  │  (Equity curve / BH mass / Regime timeline)      │
│  [ ] ES      │                                                    │
│  [ ] NQ      ├───────────────────────────────────────────────────┤
│  [ ] CL      │  Trade Table                                      │
│  [ ] GC      │  (entry, exit, P&L, tf_score, regime)            │
│  [ ] ZB      ├───────────────────────────────────────────────────┤
│  [ ] BTC     │  Summary Metrics                                  │
│  ...         │  Sharpe / CAGR / Max DD / Win Rate / Profit Factor│
│              │                                                    │
├──────────────┴───────────────────────────────────────────────────┤
│  Parameters          [Run Backtest]  [Run MC]  [Sensitivity]     │
│  bh_form: [1.5]      Start: [2020-01-01]  End: [2024-01-01]     │
│  min_tf_score: [2]   Initial equity: [$100,000]                  │
│  pos_floor: [0.0]    Timeframe: [1d] [1h] [15m]                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step: Running a Backtest

1. **Select instruments** in the left panel. Ctrl+click for multiple.
2. **Set date range** and `initial_equity`.
3. **Adjust BH parameters**:
   - `bh_form`: Higher = fewer but stronger signals (1.0–3.0)
   - `min_tf_score`: Minimum confluence score for entry (0–7)
   - `pos_floor`: Minimum position size when BH is active (0–0.5)
4. **Click "Run Backtest"**. Results appear within 2-5 seconds.
5. **Inspect the trade table**: Sort by `tf_score` to see if higher scores outperform.
6. **Click "Run MC"** to see the distribution of possible outcomes.
7. **Click "Sensitivity"** to see which parameter matters most.

---

## Reading the BH Timeline

The BH timeline panel (below the equity curve) shows for each bar:
- **Mass bars** (green = active, gray = inactive): Height = BH mass value
- **Regime shading**: BULL (light green), BEAR (light red), SIDEWAYS (yellow), HIGH_VOL (orange)
- **Entry/exit markers**: Triangles at entry (▲ long, ▼ short), circles at exit

Click any trade in the trade table to zoom the chart to that trade's entry/exit window.

---

## MC Fan Chart

After running MC, the equity curve area shows percentile bands:
- Dark green fill = P25–P75 (interquartile range, most likely outcomes)
- Light green fill = P5–P95 (95th percentile range)
- Red dashed line = blowup threshold (50% of initial equity)

If the blowup threshold line is visible inside the P5 band, the strategy has > 5% blowup probability — reduce position sizes or raise `min_tf_score`.

---

## Saving and Loading Configurations

Configurations are saved as JSON in your browser's local storage. To share:
1. Click the "Export Config" button (top right of parameters panel)
2. Copy the JSON or download as file
3. Share with teammates
4. They can paste into "Import Config"

Configurations can also be saved as YAML files and used directly with the CLI:

```bash
# Run a named configuration
make backtest s=larsa-v16
# Or with Spacetime Arena:
python -m spacetime.engine.bh_engine --config path/to/config.yaml
```
