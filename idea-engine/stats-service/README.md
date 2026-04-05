# Stats Service

The Stats Service provides advanced quantitative analysis for the Idea Automation Engine (IAE).
It combines R (statistical modeling, reporting) and Julia (high-performance optimization, time-series)
to evaluate, validate, and improve trading strategy genomes produced by the evolutionary engine.

## What It Does

- **Regime Detection** — Fits a 3-state Hidden Markov Model (BULL/BEAR/NEUTRAL) on equity return
  series using R's `depmixS4` package. Each bar receives a posterior regime probability.
- **Robustness Testing** — Bootstrap Sharpe confidence intervals, White's Reality Check for data
  snooping bias, Benjamini-Hochberg multiple-hypothesis correction.
- **Factor Attribution** — OLS Fama-French-style factor decomposition of strategy returns.
- **Walk-Forward Analysis** — Proper in-sample/out-of-sample WFA with parameter stability and
  efficiency ratio metrics.
- **Bayesian Optimization** — Gaussian-process-based hyperparameter search (Julia), parallel grid
  search, and NSGA-II multi-objective Pareto optimization.
- **Time-Series Analysis** — Hurst exponent, fractional differencing, auto-ARIMA, wavelet
  decomposition, cointegration tests (Julia).
- **Causal Discovery** — Granger causality matrix, transfer entropy, PC algorithm, FCI for hidden
  confounders (Julia).
- **Reporting** — Full tearsheets and JSON stat reports persisted to `idea_engine.db`.

## Directory Layout

```
stats-service/
├── r/
│   ├── analysis.R            # Core statistical analysis
│   ├── walk_forward_analysis.R
│   └── reporting.R
├── julia/
│   ├── optimizer.jl          # Bayesian / Pareto optimization
│   ├── time_series.jl        # HMM, Hurst, wavelets, cointegration
│   └── causal_fast.jl        # Granger, transfer entropy, PC/FCI
├── output/                   # Transient JSON output from R/Julia
├── schema_extension.sql      # SQLite tables: stats_reports, optimization_runs
└── run_r_analysis.py         # Python orchestration wrapper
```

## Dependencies

- **R** ≥ 4.3: `RSQLite`, `depmixS4`, `tidyverse`, `quantmod`, `PerformanceAnalytics`,
  `forecast`, `boot`, `car`
- **Julia** ≥ 1.10: `Turing`, `AbstractGPs`, `PyCall`, `FFTW`, `Wavelets`, `HypothesisTests`
- **Python** ≥ 3.11: standard library only for the wrapper

## Usage

```python
from stats_service.run_r_analysis import run_r_analysis, run_julia_optimizer

# Run full statistical analysis on a backtest run
report = run_r_analysis("run_abc123", script="analysis.R")

# Bayesian optimisation over parameter bounds
best = run_julia_optimizer(
    param_bounds={"fast_period": [5, 50], "slow_period": [20, 200]},
    method="bayesian",
)
```
