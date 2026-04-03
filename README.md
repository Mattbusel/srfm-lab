# SRFM Lab — Relativistic Trading Research

A local LEAN-powered research environment for developing, backtesting, and iterating
on trading strategies built on **Special Relativistic Financial Mechanics (SRFM)**.

> Mad scientist workshop. Everything automated, everything measurable, rapid iteration at scale.

---

## Quick Start

```bash
# Create a new experiment (copies template)
make new name=my-experiment

# Edit ONE thing in your experiment
vim strategies/my-experiment/main.py

# Run a backtest
make backtest s=my-experiment

# Compare against baseline
make compare2 s1=larsa-v1 s2=my-experiment

# Sweep a parameter
make sweep s=my-experiment param=BH_FORM min=0.5 max=3.0 step=0.25
```

---

## Architecture

```
Price Bar
    │
    ▼
MinkowskiClassifier ─── ds² = c²dt² − dx² ──► TIMELIKE / SPACELIKE
    │
    ▼
BlackHoleDetector ────── mass accretion ──────► BHState (ABSENT/FORMING/ACTIVE/COLLAPSE)
    │                    + well memory
    ▼
GeodesicAnalyzer ─────── rapidity, deviation ► causal_fraction, geodesic_deviation
    │
    ▼
RegimeDetector ───────── combine above ───────► TRENDING / RANGING / CRISIS / RECOVERY
    │
    ▼
HawkingMonitor ────────── T_H = 1/(8πM) ─────► stability scalar → position size
    │
    ▼
AgentEnsemble ─────────── D3QN + DDQN + TD3QN ► consensus signal (+1 / 0 / -1)
    │
    ▼
RiskManager ──────────── stops, drawdown ─────► final entry / exit decision
    │
    ▼
LEAN MarketOrder
```

---

## Repo Structure

```
srfm-lab/
├── strategies/
│   ├── larsa-v1/          # 274% production baseline — DO NOT MODIFY
│   ├── larsa-v2/          # Active development
│   ├── templates/         # Starting point for new experiments
│   └── graveyard/         # Failed experiments + post-mortem notes
├── lib/
│   ├── srfm_core.py       # SRFM physics (Minkowski, BH, Geodesic, Hawking, Lens, ProperTime)
│   ├── agents.py          # D3QN, DDQN, TD3QN ensemble agents
│   ├── regime.py          # Regime detection
│   └── risk.py            # Risk management, stops, circuit breakers
├── tools/
│   ├── batch_runner.py    # Parallel backtesting with variant configs
│   ├── compare.py         # Side-by-side result comparison + equity charts
│   ├── param_sweep.py     # Parameter sensitivity surface
│   ├── regime_analyzer.py # Historical regime timeline analysis
│   └── well_detector.py   # Standalone BH well detection on price CSV
├── scripts/               # Shell wrappers (called by Makefile)
├── research/notebooks/    # Jupyter via `lean research` or Docker
├── data/                  # LEAN auto-populates
├── results/               # Backtest outputs
└── .github/workflows/     # Auto-backtest on push to strategies/*/main.py
```

---

## Core Principles

1. **Every experiment is reproducible**: strategy code + LEAN version + data = same result.
2. **One change at a time**: copy template, modify ONE constant or ONE component, backtest.
3. **Failed experiments go to graveyard with a note**: document why they failed.
4. **`lib/` is the moat**: portable SRFM physics that works across any LEAN strategy.
5. **Batch testing is the default**: 10 variants in parallel beats 10 sequential runs.
6. **Results are always compared**: never evaluate a strategy in isolation.

---

## SRFM Physics Reference

| Component | Key Formula | Interpretation |
|-----------|-------------|----------------|
| MinkowskiClassifier | `ds² = c²dt² − dx²` | TIMELIKE = ordered, causal; SPACELIKE = anomalous velocity |
| BlackHoleDetector | mass accretes on spacelike moves | Well forms when mass ≥ `BH_FORM` |
| HawkingMonitor | `T_H = 1 / (8πM)` | Cold well = stable signal; hot well = reduce size |
| GravitationalLens | `μ = (u²+2) / (u√(u²+4))` | BH amplifies signal from other indicators |
| ProperTimeClock | `dτ² = dt² − (dx/c)²` | Proper time gates entries; spacelike moves don't advance it |
| GeodesicAnalyzer | RMS deviation from mean return | High deviation = curved spacetime = crisis |

---

## Parameters to Tune

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CF` | 1.2 | Minkowski speed-of-light (higher → fewer spacelike) |
| `BH_FORM` | 1.5 | Mass to declare BH active |
| `BH_COLLAPSE` | 0.4 | Mass at which BH collapses |
| `MASS_DECAY` | 0.92 | Per-bar exponential mass decay |
| `PROPER_TIME_MIN` | 5.0 | Proper time bars between entries |

Sweep these with `make sweep` before touching strategy logic.

---

## Baseline: LARSA v1

- **Return**: 274% (verified QC Cloud)
- **Instrument**: ES (S&P 500 E-mini futures)
- **Period**: [see strategies/larsa-v1/main.py]
- **Do not modify this directory** — it is the reference.

---

## References

- SRFM papers: [Zenodo — add links]
- LEAN documentation: https://www.lean.io/docs
- QuantConnect: https://www.quantconnect.com

---

*License: Proprietary. All rights reserved. This repository contains unpublished
research and strategy IP. Do not distribute.*
