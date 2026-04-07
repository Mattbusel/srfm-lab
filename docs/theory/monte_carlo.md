# Monte Carlo Methodology

## Overview

The SRFM lab uses Monte Carlo simulation to estimate the distribution of future strategy performance. The key feature is **regime-aware sampling**: rather than shuffling all trades independently, we sample in a way that preserves the serial correlation structure and regime clustering observed in real markets.

---

## 1. Why Regime-Aware MC vs Naive Shuffle

### The problem with naive shuffling

A standard MC simulation shuffles the historical trade P&L series and computes equity curves from many random orderings. This assumes trades are i.i.d. (independent and identically distributed). Real trading systems violate this assumption in two important ways:

1. **Regime clustering**: Winning trades cluster in momentum regimes (BULL/BEAR); losing trades cluster in choppy SIDEWAYS and HIGH_VOLATILITY regimes. A naive shuffle destroys this clustering.

2. **Serial correlation of losses**: After a losing trade, the conditional probability of another losing trade is higher than the unconditional probability. This is especially true during market dislocations where the BH signal fires incorrectly multiple times before a regime regime.

If we ignore these effects, MC underestimates the probability and severity of loss streaks.

### The regime-aware approach

We split the trade distribution by regime and sample conditionally:

```python
regimes = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY']
trade_pool = {regime: [t for t in trades if t.regime == regime] for regime in regimes}

# Sample regime sequence from empirical transition matrix
regime_sequence = sample_markov_chain(transition_matrix, n_trades)

# Sample trades from the appropriate pool
simulated_trades = [random.choice(trade_pool[r]) for r in regime_sequence]
```

This preserves:
- Win rate differences across regimes
- Return distribution skew by regime
- Average regime duration (from the Markov chain)

---

## 2. AR(1) Serial Correlation Model for Loss Streaks

Within each regime, we model residual serial correlation using an AR(1) process on the trade outcomes.

### The model

Let `X_t ∈ {0, 1}` be the outcome of trade t (1 = winner, 0 = loser). We model:

```
X_t = p + ρ × (X_{t-1} - p) + ε_t
```

where:
- `p` = unconditional win probability
- `ρ` = AR(1) coefficient (serial correlation)
- `ε_t ~ Bernoulli(0.5) noise term (demeaned)

**Estimation**: Fit `ρ` from the historical trade sequence using OLS or method-of-moments:

```python
import numpy as np
from statsmodels.regression.linear_model import OLS

X = np.array(trade_outcomes, dtype=float)
p = X.mean()
X_demeaned = X - p
rho = np.corrcoef(X_demeaned[:-1], X_demeaned[1:])[0, 1]
```

**Interpretation**:
- `ρ > 0`: Positive serial correlation -- wins beget wins, losses beget losses
- `ρ < 0`: Negative serial correlation -- alternating wins/losses (mean-reverting outcomes)
- `ρ = 0`: No serial correlation (standard naive MC assumption)

For most trend-following strategies, ρ is in the range [0.05, 0.25], meaning loss streaks are more clustered than a naive model would predict.

### Generating AR(1) trade sequences

```python
def sample_ar1_outcomes(p: float, rho: float, n: int) -> np.ndarray:
    """Sample binary outcomes with AR(1) serial correlation."""
    outcomes = np.zeros(n)
    outcomes[0] = np.random.random() < p
    for t in range(1, n):
        # Conditional probability of win given previous outcome
        p_cond = p + rho * (outcomes[t-1] - p)
        p_cond = np.clip(p_cond, 0.01, 0.99)
        outcomes[t] = np.random.random() < p_cond
    return outcomes
```

---

## 3. Kelly Criterion Derivation from Log-Utility

### The log-utility framework

The Kelly criterion maximizes the **expected logarithm of wealth**, which is equivalent to maximizing the long-run geometric growth rate. For a binary bet with:
- `p` = probability of winning
- `b` = edge ratio (average win / average loss, both positive)

The Kelly fraction is:

```
f* = p - (1 - p) / b
```

For a trade with `p = 0.58` and `b = 1.8` (edge ratio):
```
f* = 0.58 - 0.42 / 1.8 = 0.58 - 0.233 = 0.347 = 34.7%
```

### Deriving from first principles

Given a fraction `f` of wealth risked on each trade, after `n` trades with `W` wins (each gaining `f × b`) and `L` losses (each losing `f`):

```
Terminal wealth = (1 + f×b)^W × (1 - f)^L
```

The expected log-growth per trade:

```
G(f) = p × ln(1 + f×b) + (1-p) × ln(1 - f)
```

Maximizing `G(f)` by setting `dG/df = 0`:

```
p×b / (1 + f×b) - (1-p) / (1 - f) = 0
```

Solving:

```
f* = p - (1-p)/b  ≡  Kelly fraction
```

### Why we use half-Kelly

Full Kelly has several practical problems:
1. **Parameter estimation error**: The true p and b are estimated from a finite sample. If p is overestimated by even 5%, full Kelly can result in catastrophic drawdowns.
2. **Finite sampling path**: Kelly is an asymptotic result. For any finite horizon, the variance of outcomes at full Kelly is high.
3. **Non-stationarity**: The true edge changes over time. Using historical estimates assumes stationarity.

Using **half-Kelly** (f = f*/2) approximately halves the expected growth rate but reduces the variance of outcomes by 4x, providing much better risk-adjusted returns in practice.

```
f_half = f*/2 = (p - (1-p)/b) / 2
```

For our example: f_half = 17.4%

---

## 4. Interpreting MC Percentile Bands

### The percentile chart

The MC output shows:
- **P5 band** (5th percentile): "Bad luck" scenario. 95% of outcomes are better than this.
- **P25 band** (25th percentile): "Below average" outcome.
- **P50 band** (50th percentile, median): The most likely outcome path.
- **P75 band** (75th percentile): "Above average" outcome.
- **P95 band** (95th percentile): "Good luck" scenario.

### What each percentile means for decision-making

**P5 path**: This is not a doomsday scenario but rather the typical "bad year." If the P5 terminal equity is above your minimum acceptable level, the strategy is robust to realistic bad luck.

**P50 path**: Your "base case." This is what you should plan around. Note that the P50 path often falls below the historical equity curve because:
1. The historical curve is a cherry-picked realization (survivorship bias)
2. Future conditions may differ from backtest conditions

**P95 path**: This is not the ceiling -- there will always be paths above P95 in a large enough simulation. Use P95 as a sanity check: if the P95 path seems unrealistically good, the historical trade distribution may be overfitted.

### Reading the fan chart

```
            P95
        -------/
P75   ------/
       ------
Actual  ----
P50   ---
P25  -/
P5  /
```

A narrow fan = consistent strategy (low variance of outcomes)
A wide fan = high variance strategy (outcome depends heavily on luck)

For LARSA, the typical fan narrows over time as sample size grows (central limit theorem effect), meaning longer backtests show more predictable performance.

---

## 5. Blowup Probability

### Definition

Blowup is defined as terminal equity falling below **50% of initial equity** (a 50% drawdown). This represents a scenario severe enough to require strategy redesign.

```python
blowup_prob = sum(1 for eq in terminal_equities if eq < 0.5 * initial_equity) / n_sims
```

### Interpretation

| Blowup Probability | Interpretation                           |
|-------------------|------------------------------------------|
| < 0.1%            | Negligible: consider strategy robust     |
| 0.1% - 1%         | Acceptable: review position sizing       |
| 1% - 5%           | Elevated: reduce leverage                |
| > 5%              | Dangerous: fundamental strategy issue    |

For LARSA with current parameters, blowup probability is typically < 0.5% over a 2-year forward horizon.

### Blowup vs maximum drawdown

Blowup probability is not the same as P(max_drawdown > 50%). The blowup condition requires **terminal** equity to be below 50%, not just an intermediate trough. A strategy can have a 60% drawdown and recover to above 50% initial equity -- that is not a blowup.

The more conservative `max_drawdown_blowup_prob` uses:
```python
max_dd_blowup = sum(1 for path in paths if min(path) < 0.5 * initial_equity) / n_sims
```

This is typically 3-5x higher than the terminal equity definition.

### What blowup probability means for sizing

The risk budget for a given blowup probability target can be computed using the Ruin Probability formula (for continuous compounding):

```
P(ruin) ≈ exp(-2 × E[g] × R / Var[g])
```

where:
- `E[g]` = expected log growth per period
- `R` = ruin boundary (fraction of initial equity)  
- `Var[g]` = variance of log growth

This gives a closed-form size that achieves a target blowup probability.
