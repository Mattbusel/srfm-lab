# Optimization -- Three-Layer Parameter Stack

## Overview

The `optimization/` package implements a three-layer parameter optimization stack that
tunes SRFM strategy parameters at multiple timescales and granularities. Each layer
addresses a different aspect of the parameter search problem:

- **Layer 1 -- Bayesian Optimizer**: efficient single-objective search over the core
  BH signal parameters using Gaussian Process surrogate models.
- **Layer 2 -- Hyperparameter Search**: multi-objective Pareto optimization balancing
  Sharpe ratio against maximum drawdown using NSGA-II.
- **Layer 3 -- Regime Parameter Optimizer**: per-regime Optuna studies that adapt
  parameters dynamically as market regimes shift.

All three layers feed results into the IAE GenomeEngine as seed populations, closing
the loop between static optimization and live adaptive parameter evolution.

---

## Bayesian Optimizer -- `bayesian_optimizer.py`

### Purpose

The Bayesian optimizer is the primary workhorse for BH signal parameter search. It is
efficient in terms of function evaluations (each evaluation requires running a full
backtest) and naturally handles noisy objectives by modeling uncertainty explicitly.

### Gaussian Process Surrogate

The surrogate model is a Gaussian Process (GP) with a Matern 5/2 kernel:

```
k(x, x') = sigma^2 * (1 + sqrt(5)*r + (5/3)*r^2) * exp(-sqrt(5)*r)
```

where `r = ||x - x'|| / length_scale`. The Matern 5/2 kernel is preferred over the
squared exponential (RBF) because it produces smoother but less infinitely
differentiable functions, which better models the jagged Sharpe-vs-parameter surface
typical of financial strategies.

The GP is fit using maximum marginal likelihood (MML) optimization via L-BFGS-B each
time a new observation is added.

### Expected Improvement Acquisition

The acquisition function is Expected Improvement (EI):

```
EI(x) = (mu(x) - f_best - xi) * Phi(Z) + sigma(x) * phi(Z)
Z = (mu(x) - f_best - xi) / sigma(x)
```

`xi` (default 0.01) is the exploration-exploitation trade-off parameter. Higher `xi`
encourages exploration of uncertain regions. The next candidate point is the argmax
of EI over the parameter space, computed by multi-start L-BFGS-B.

### Warm Start from Prior Runs

`BayesianOptimizer` accepts a `prior_results_db` path pointing to a DuckDB database of
previous optimization runs. On initialization it loads all historical (params, objective)
pairs and seeds the GP with these observations before the first new evaluation. This
means repeat optimizations (e.g. monthly re-runs) converge in far fewer evaluations
than cold starts.

### DuckDB Result Caching

Every evaluation is written to the DuckDB cache immediately after the backtest
completes:

```sql
CREATE TABLE bo_results (
    run_id       VARCHAR,
    strategy_id  VARCHAR,
    params_json  JSON,
    sharpe       DOUBLE,
    max_dd       DOUBLE,
    calmar       DOUBLE,
    n_trades     INTEGER,
    evaluated_at TIMESTAMP
);
```

DuckDB's columnar storage makes it fast to query "all runs with CF in [0.01, 0.02] and
Sharpe > 1.5" during warm-start loading, without scanning the full result set.

### BH Parameter Search Space

The Bayesian optimizer is specifically tuned for the core BH signal parameters:

| Parameter | Range | Type | Description |
|---|---|---|---|
| `CF` | [0.001, 0.05] | continuous | Compression factor -- controls signal sensitivity |
| `BH_FORM` | [1.5, 2.5] | continuous | BH formation multiplier |
| `MIN_HOLD` | [1, 48] | integer | Minimum holding period in bars |

`BH_DECAY` and `GARCH_TARGET_VOL` are handled by the hyperparameter search layer rather
than the Bayesian optimizer, because they interact strongly with each other and benefit
from multi-objective treatment.

---

## Hyperparameter Search -- `hyperparameter_search.py`

### NSGA-II Multi-Objective Optimization

`HyperparameterSearch` uses the NSGA-II (Non-dominated Sorting Genetic Algorithm II)
to simultaneously optimize two competing objectives:

- **Maximize** annualized Sharpe ratio
- **Minimize** maximum drawdown

NSGA-II maintains a population of candidate parameter vectors and iteratively applies
selection (based on Pareto rank and crowding distance), crossover, and mutation. The
result is an approximation of the Pareto-optimal front -- the set of parameter vectors
for which no other vector is strictly better on both objectives.

### Hypervolume Indicator

The quality of the Pareto front approximation is tracked using the hypervolume indicator
`HV(A, r)`, which measures the volume of the objective space dominated by the front `A`
relative to a reference point `r = (0.0, 1.0)` (Sharpe=0, MaxDD=100%). A larger
hypervolume indicates a better-quality front. Hypervolume is computed after each
generation and plotted in the optimization log.

### Sobol Low-Discrepancy Initial Sampling

Rather than initializing the NSGA-II population randomly (which can leave large regions
of the parameter space unsampled), `HyperparameterSearch` uses a Sobol sequence to
generate the initial population. Sobol sequences are quasi-random sequences that fill
the parameter space more uniformly than pseudo-random samples, which leads to better
Pareto front coverage in early generations.

The initial population size is `2 * n_params * 10` by default, ensuring at least 10
samples per parameter dimension.

### Full Parameter Space for NSGA-II

| Parameter | Range | Type |
|---|---|---|
| `CF` | [0.001, 0.05] | continuous |
| `BH_FORM` | [1.5, 2.5] | continuous |
| `BH_DECAY` | [0.85, 0.98] | continuous |
| `MIN_HOLD` | [1, 48] | integer |
| `GARCH_TARGET_VOL` | [0.5, 2.0] | continuous |

After the final generation, the Pareto front is filtered by a minimum Sharpe threshold
(default 1.0) and the knee-point solution (maximum product of normalized objectives) is
selected as the recommended parameter set for deployment.

---

## Regime Parameter Optimizer -- `regime_parameter_optimizer.py`

### Per-Regime Optuna Studies

Different market regimes (trending, mean-reverting, high-volatility, low-liquidity)
favor different parameter configurations. `RegimeParameterOptimizer` maintains a
separate Optuna `Study` for each detected regime label, stored in a regime-partitioned
SQLite database.

Each study optimizes the same objective function (Sharpe on regime-specific data
segments) but only evaluates on data windows that were classified as that regime by the
IAE regime detector.

### Regime-Conditional Best Parameters

`RegimeParameterOptimizer.best_params(regime_label)` returns the Optuna best-trial
parameters for the requested regime. If a regime has fewer than `min_trials` evaluated
(default 30), it falls back to the global Bayesian optimizer best parameters with a
conservative CF multiplier of 0.8.

### Automatic Refit on Regime Change

`RegimeParameterOptimizer` subscribes to regime change events from the IAE
`RegimeDetector`. When a regime transition is detected:

1. The current regime's study is checkpointed (best params saved to the SQLite store).
2. The new regime's study is loaded.
3. If the new regime's study has stale best params (last trial older than
   `refit_staleness_days`, default 7), an async refit is triggered in a background
   thread.
4. The live strategy engine is notified via a `ParamUpdateEvent` with the new
   regime's best parameters.

This mechanism ensures the strategy is always running parameters that are appropriate
for the current regime without blocking the execution path.

---

## Sensitivity Analysis

### Sobol Indices

`sensitivity_analysis.py` computes Sobol sensitivity indices to identify which
parameters contribute most to output variance. For each parameter `x_i`:

- **First-order Sobol index `S_i`**: fraction of output variance explained by `x_i`
  alone.
- **Total-effect Sobol index `S_Ti`**: fraction of output variance explained by `x_i`
  including all interactions with other parameters.

The difference `S_Ti - S_i` measures how much of parameter `x_i`'s influence comes
from interactions rather than direct effects. High interaction terms suggest parameters
should be tuned jointly, not independently.

Sobol indices are estimated via the Saltelli sampling scheme using `N * (2D + 2)`
model evaluations, where `N` is the base sample size (default 512) and `D` is the
number of parameters.

### Morris Method for Screening

Before running the full Sobol analysis (which is expensive), `morris_screen()` applies
the Morris method to rank parameters by their mean elementary effect `mu*`. Parameters
with `mu* < 0.01 * max(mu*)` are flagged as non-influential and excluded from the
full Sobol run. This typically reduces the parameter count from 5 to 3, cutting Sobol
evaluation cost by ~60%.

### Variance Decomposition Report

The output of sensitivity analysis is a `SensitivityReport`:

```python
@dataclass
class SensitivityReport:
    first_order:   dict[str, float]   # S_i per parameter
    total_effect:  dict[str, float]   # S_Ti per parameter
    interactions:  dict[str, float]   # S_Ti - S_i per parameter
    morris_mu:     dict[str, float]   # Morris mean elementary effect
    morris_sigma:  dict[str, float]   # Morris std -- nonlinearity indicator
    top_params:    list[str]          # ranked by total effect
```

Sensitivity reports are generated monthly and stored in DuckDB alongside optimizer
results so that parameter importance trends can be tracked over time.

---

## IAE Genome Bridge

Optimization results feed back into the IAE `GenomeEngine` as seed populations. This
creates a closed loop:

```
Backtest -> Optimizer -> Best Params
                              |
                              v
                    GenomeEngine.seed_population(params)
                              |
                              v
                    Live adaptive evolution starts from
                    a near-optimal initial genome rather
                    than random initialization
```

`RegimeParameterOptimizer.export_genome_seed(regime_label)` produces a `GenomeSeed`
dataclass with `n_individuals` parameter vectors sampled from the Optuna Pareto front
for that regime. These are written to the IAE seed file at
`iae/genome_seeds/{regime_label}.json`.

The `GenomeEngine` reads seed files at startup and at each regime transition, giving
the adaptive layer a warm start that dramatically reduces convergence time in live
trading.

---

## Constraint Handling

### Monotonicity Constraints

The optimizer enforces that higher `CF` values correspond to tighter signal thresholds.
This is implemented as a penalty term added to the objective when the monotonicity
constraint is violated:

```
penalized_sharpe = sharpe - lambda * max(0, CF_violation)
```

`lambda = 10.0` by default, which makes severe violations uncompetitive but allows
mild violations to survive early in the search.

### Sum Constraints on Time Weights

For strategies that use time-weighted signal combinations, the weights must sum to 1.0.
`HyperparameterSearch` enforces this via a repair operator applied after each
crossover: the raw weight vector is normalized by its sum before evaluation.

### Conflict Detection

`ConstraintChecker.validate(params)` checks for parameter combinations known to cause
numerical instability or degenerate behavior:

- `MIN_HOLD >= 1 / CF` -- holding period longer than signal mean-reversion time
- `BH_DECAY < 0.85` combined with `GARCH_TARGET_VOL > 1.5` -- unstable variance targeting
- `BH_FORM > 2.3` combined with `CF < 0.005` -- formation too aggressive for tight threshold

Conflicting configurations are assigned a Sharpe of `-inf` and excluded from Pareto
front updates, preventing the optimizer from wasting evaluations in degenerate regions.
