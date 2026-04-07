# IAE Genome Evolution Pipeline

## Overview

The Idea Architecture Engine (IAE) genome evolution pipeline implements NSGA-II -- a
multi-objective genetic algorithm -- to evolve trading strategy parameter sets (genomes)
toward Pareto-optimal performance. The pipeline is split across two components:

- **Rust** (`crates/idea-genome-engine/`) -- core genetic operators, constraint handling,
  and fitness evaluation workers
- **Go** (`idea-engine/`) -- orchestration, lineage tracking, stagnation detection, and
  coordination with the Elixir parameter layer

The pipeline runs continuously in the background, cycling through populations and
broadcasting improved genomes to the live trading system via the ParameterCoordinator
service at `:8781`.

---

## Genome Representation

A genome is a **fixed-length float vector** where each index maps to a specific tunable
strategy parameter. This encoding allows all genetic operators (crossover, mutation,
selection) to operate uniformly without special-casing per-parameter logic.

### Parameter Map

| Index | Parameter              | Domain         | Description                                      |
|-------|------------------------|----------------|--------------------------------------------------|
| 0     | `CF`                   | [0.1, 5.0]     | Confidence factor -- scales position sizing      |
| 1     | `BH_FORM`              | [0.0, 1.0]     | Black hole formation threshold                   |
| 2     | `BH_DECAY`             | [0.001, 0.5]   | Black hole decay rate per bar                    |
| 3     | `MIN_HOLD`             | [1, 120]       | Minimum hold bars before exit is considered      |
| 4     | `GARCH_TARGET_VOL`     | [0.005, 0.10]  | GARCH(1,1) target volatility for vol scaling     |
| 5     | `OU_FRAC`              | [0.0, 1.0]     | Ornstein-Uhlenbeck mean-reversion fraction       |
| 6     | `BLOCKED_HOURS`        | bitmask [0,1]  | 24-bit mask -- hours where trading is blocked    |
| 7     | `BOOST_HOURS`          | bitmask [0,1]  | 24-bit mask -- hours where sizing is boosted     |
| 8     | `WINNER_PROTECTION_PCT`| [0.0, 1.0]     | Trailing stop as fraction of peak unrealized PnL |
| ...   | (additional params)    | varies         | Strategy-specific extensions                     |

The bitmask parameters (`BLOCKED_HOURS`, `BOOST_HOURS`) are encoded as floats in [0, 1]
per bit position and thresholded at 0.5 during decoding -- this keeps the genome fully
continuous for operator compatibility while preserving discrete semantics at runtime.

---

## NSGA-II Objectives

The pipeline optimizes simultaneously across three objectives, requiring no single
aggregate fitness function:

### 1. Weighted Sharpe Ratio (maximize)

A **3-period weighted Sharpe** divides the backtest window into three temporal segments
and computes an importance-weighted combination:

```
Sharpe_weighted = 0.5 * Sharpe_recent + 0.3 * Sharpe_mid + 0.2 * Sharpe_early
```

Where `recent` is the most recent third of the evaluation window, `mid` is the middle
third, and `early` is the earliest third. The forward-weighting penalizes genomes that
were strong historically but have degraded in current market conditions.

### 2. Maximum Drawdown (minimize)

```
MaxDD = max over [0,T] of (peak(t) - trough(t)) / peak(t)
```

Computed on the equity curve produced by the Python subprocess fitness evaluator. Lower
is better -- NSGA-II treats this as a minimization objective.

### 3. Profit Factor (maximize)

```
PF = sum(winning trades) / |sum(losing trades)|
```

A profit factor > 1.5 is the informal floor for non-dominated solutions to be considered
for live deployment.

---

## Pareto Front and Elitism

NSGA-II maintains a population where solutions are ranked by **Pareto dominance** rather
than a scalar score. A solution A dominates B if A is at least as good on all objectives
and strictly better on at least one.

### Non-Dominated Sorting

The population is partitioned into fronts F1, F2, F3, ... where:
- F1 -- the Pareto front: no solution in F1 is dominated by any other
- F2 -- solutions dominated only by F1 members
- Fn -- solutions dominated by all prior fronts

**Elitism** is enforced by filling the next generation's survivor pool front-by-front
until capacity is reached. If a front partially fits, members are ranked by crowding
distance to preserve diversity on the Pareto surface.

### Crowding Distance

For a solution i on front Fk, crowding distance is the sum over objectives of the
normalized distance to its nearest neighbors:

```
CD(i) = sum_m [ (f_m(i+1) - f_m(i-1)) / (f_m_max - f_m_min) ]
```

Solutions with larger crowding distance are preferred when breaking ties, which keeps the
front spread out and prevents convergence to a single point.

---

## Crossover Strategies

Implemented in `crates/idea-genome-engine/src/crossover_strategies.rs`.

### Uniform Crossover

Each gene is independently taken from parent A or parent B with probability 0.5. Produces
maximum disruption -- useful for escaping local optima in early generations.

### BLX-alpha (Blend Crossover)

For gene index i with values `a_i` and `b_i` (where `a_i <= b_i`):

```
gene = U(a_i - alpha * range, b_i + alpha * range)
```

Where `range = b_i - a_i` and `alpha = 0.5` by default. Allows offspring to explore
slightly outside the parent interval.

### SBX (Simulated Binary Crossover)

Mimics one-point binary crossover behavior on real-valued genes. Uses a distribution
index `eta_c` (typically 2--5) to control the spread of offspring around parents. Higher
`eta_c` produces offspring closer to the parents.

### Arithmetic Crossover

Simple weighted average:

```
child = alpha * parent_A + (1 - alpha) * parent_B
```

Where `alpha` is drawn from U(0,1). Always produces offspring within the convex hull of
the parents.

### Differential Evolution (DE/rand/1)

Selects three distinct random individuals r1, r2, r3 from the population:

```
mutant = r1 + F * (r2 - r3)
child[i] = mutant[i]  if U(0,1) < CR
           parent[i]  otherwise
```

DE crossover is particularly effective for tight parameter interactions such as
`BH_FORM`/`BH_DECAY` coupling.

---

## Mutation Strategies

Implemented in `crates/idea-genome-engine/src/mutation_strategies.rs`.

### Gaussian Mutation

```
gene' = gene + N(0, sigma)
```

Standard perturbation -- sigma is typically 1--5% of the gene's domain range.

### Cauchy Mutation

```
gene' = gene + Cauchy(0, gamma)
```

The Cauchy distribution's heavy tails enable large jumps that escape local optima more
readily than Gaussian noise. Useful during stagnation periods.

### Polynomial Mutation

Uses the polynomial probability distribution (distribution index `eta_m = 20` default).
Perturbation magnitude is bounded and decreases as the gene approaches its domain limits,
making this naturally bounds-respectful.

### Adaptive Mutation (Self-Adaptive Sigma)

The mutation step size `sigma` is itself encoded in an auxiliary gene that co-evolves
with the strategy parameters:

```
sigma' = sigma * exp(tau' * N(0,1) + tau * N_i(0,1))
gene'  = gene + sigma' * N(0,1)
```

Where `tau = 1/sqrt(2*n)` and `tau' = 1/sqrt(2*sqrt(n))` for genome length n. This
allows the optimizer to autonomously increase exploration during flat fitness landscapes.

### NonUniform Mutation (Cooling Schedule)

Perturbation magnitude decays as generation count increases:

```
delta(g) = (1 - U(0,1)^((1 - g/G)^b)) * (upper - gene)
```

Where g is current generation, G is max generations, and b controls the non-uniformity
degree. Implements an annealing-like schedule without an explicit temperature parameter.

---

## Selection Strategies

Implemented in `crates/idea-genome-engine/src/selection_strategies.rs`.

### Tournament Selection

k individuals are drawn at random; the one with the best Pareto rank (or, if tied, the
larger crowding distance) is selected. k = 2 is the default (binary tournament).

### Rank-Proportional Selection

Probability of selection is proportional to the individual's rank in the combined
Pareto-and-crowding ordering. Less aggressive than tournament, maintains more diversity
from lower fronts.

### Boltzmann Selection

Selection probability is scaled by a temperature T that decays over generations:

```
P(i) = exp(fitness(i) / T) / Z
```

High T early -- broad exploration. Low T late -- exploitation of best regions. The
scalar `fitness` used here is supplied by the fitness aggregator (see below).

### Niche Selection (Fitness Sharing)

Modifies apparent fitness by dividing by a sharing factor that counts how many neighbors
are nearby in objective space:

```
fitness_shared(i) = fitness(i) / sum_j sh(d(i,j))
sh(d) = 1 - (d/sigma_share)^alpha  if d < sigma_share, else 0
```

Niche selection is activated when the diversity metric drops below a threshold (see
Stagnation Detection).

---

## Constraint Handling

Implemented in `crates/idea-genome-engine/src/constraint_handler.rs`.

### Bounds Constraints

All genes are clipped to their declared domain `[lower, upper]` immediately after any
operator application. This is the first constraint pass.

### Sum Constraints

Some parameter groups must satisfy a sum constraint (e.g., hour-allocation fractions
summing to 1.0). Violated individuals are normalized in-place.

### Monotonicity Constraints

Certain parameter pairs must be ordered (e.g., a fast MA period < slow MA period).
Violated pairs are swapped.

### Conflict Detection

Logical contradictions are detected -- for example, `BLOCKED_HOURS` and `BOOST_HOURS`
masks should not have overlapping bits. Conflicts trigger a repair operator that clears
the conflicting bits from the lower-priority mask.

### Penalty and Death Penalty

Soft violations incur a penalty added to the dominance-comparison logic. Hard violations
(structurally infeasible genomes that survive repair) are assigned a **death penalty**:
their objectives are set to the worst possible values so they are sorted to the final
Pareto front and eliminated by elitism in the next generation.

---

## Population Configuration

| Parameter             | Value  | Notes                                      |
|-----------------------|--------|--------------------------------------------|
| Population size       | 200    | Maintained across generations              |
| Generations           | 100    | Per evolution run                          |
| Evaluation workers    | 4      | Python subprocess fitness evaluators       |
| Crossover probability | 0.85   | Per pair selected for mating               |
| Mutation probability  | 0.02   | Per gene (uniform application rate)        |
| Elitism               | Yes    | Full Pareto front F1 is always preserved   |

Fitness evaluation is delegated to Python subprocesses that run the full backtest engine,
returning Sharpe, MaxDD, and PF as JSON over stdout. The 4-worker pool keeps GPU/CPU
utilization near 100% during evaluation phases.

---

## Lineage Tracking

`idea-engine/genome_store.go` persists every genome that enters the population along with
its parent IDs, generation index, and objective values. The store supports:

- **BFS traversal** -- trace any genome's ancestry back to generation 0
- **Generational pruning** -- genomes older than `max_lineage_depth` generations that
  are no longer represented in the current population are archived to disk and evicted
  from the in-memory store to bound memory usage
- **Ancestry queries** -- the genome inspector CLI can display a genome's lineage tree
  with generation-over-generation objective improvement

---

## Stagnation Detection

`idea-engine/genome_analyzer.go` monitors the evolution trajectory and detects plateau
conditions:

- **Fitness plateau** -- if the best Sharpe_weighted has not improved by > 0.01 for 15
  consecutive generations, stagnation is declared
- **Diversity collapse** -- if the average pairwise L2 distance in genome space drops
  below a threshold, diversity injection is triggered regardless of fitness trend

On stagnation, the pipeline injects a configurable fraction (default 20%) of fresh
randomly generated individuals into the population, displacing the lowest-ranked members
of the last Pareto front. Cauchy mutation is also temporarily elevated for the following
5 generations.

---

## Fitness Aggregator

`idea-engine/fitness_aggregator.go` converts the multi-objective NSGA-II output into a
scalar fitness score used by Boltzmann selection and the genome inspector's `best`
command:

```
scalar_fitness = Sharpe_weighted * (1 - MaxDD) * log(PF + 1) / (pareto_rank^0.5)
```

The Pareto rank divisor ensures F1 members always score higher than F2 members regardless
of raw objective values, while the multiplicative terms differentiate within a front.

---

## Output and Live Application

After each evolution run, the top genome by scalar fitness from Pareto front F1 is
submitted to the **ParameterCoordinator** (Elixir, `:8781`) via HTTP POST. The
coordinator performs:

1. **Schema validation** -- all parameter names and types are checked against the live
   strategy schema
2. **Delta validation** -- parameter changes exceeding a maximum per-update delta are
   rejected to prevent destabilizing step changes during live trading
3. **Broadcasting** -- approved updates are published to all registered strategy
   processes via Elixir PubSub

Genomes rejected by the coordinator are logged with rejection reason and flagged in the
genome store for post-hoc analysis.

---

## Genome Inspector CLI

`cmd/genome-inspector/main.go` provides an ANSI-colored terminal interface for
inspecting the genome store and evolution history.

### Commands

| Command                         | Description                                              |
|---------------------------------|----------------------------------------------------------|
| `genome-inspector list`         | List all genomes in the current population with ranks    |
| `genome-inspector best`         | Show the top genome by scalar fitness with parameter map |
| `genome-inspector compare A B`  | Side-by-side diff of two genomes by ID                   |
| `genome-inspector history ID`   | Ancestry lineage tree for a specific genome              |
| `genome-inspector stats`        | Population diversity, stagnation status, run metrics     |

Output uses ANSI color codes: green for improvements over the previous generation, red
for regressions, yellow for neutral changes. The `compare` command highlights genes that
differ by more than 10% of their domain range.

---

## Key Files

| Path                                                     | Purpose                              |
|----------------------------------------------------------|--------------------------------------|
| `crates/idea-genome-engine/src/crossover_strategies.rs` | All crossover operator implementations|
| `crates/idea-genome-engine/src/mutation_strategies.rs`  | All mutation operator implementations|
| `crates/idea-genome-engine/src/selection_strategies.rs` | Selection strategy implementations   |
| `crates/idea-genome-engine/src/constraint_handler.rs`   | Constraint repair and penalty logic  |
| `idea-engine/genome_store.go`                           | Lineage persistence and BFS traversal|
| `idea-engine/genome_analyzer.go`                        | Stagnation and diversity detection   |
| `idea-engine/fitness_aggregator.go`                     | Multi-objective to scalar conversion |
| `cmd/genome-inspector/main.go`                          | CLI inspection tool                  |
