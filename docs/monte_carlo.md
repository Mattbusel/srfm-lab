# Monte Carlo Engine

## Overview

`crates/monte-carlo-engine/` is the SRFM lab's stochastic simulation library, providing
four interconnected modules for risk quantification and derivatives pricing:

- **GBM paths** (`gbm_paths.rs`) -- path generators for equities, crypto, and
  multi-asset portfolios
- **VaR calculator** (`var_calculator.rs`) -- portfolio-level Value at Risk and
  Expected Shortfall
- **Option pricer** (`option_pricer.rs`) -- European, American, and barrier option
  pricing via Monte Carlo
- **Variance reduction** -- antithetic variates, control variates, and stratified
  sampling infrastructure shared across all modules

The engine is callable from Python via PyO3 bindings (`crypto_backtest_mc.py`) and has
a parallel independent implementation in Julia (`NumericalMethods.jl`) using quasi-Monte
Carlo sequences.

---

## Stochastic Process Implementations

### Geometric Brownian Motion (`gbm_paths.rs`)

The standard continuous-time model for asset prices:

```
dS = mu * S * dt + sigma * S * dW
```

Discretized in log space for numerical stability:

```
S(t+dt) = S(t) * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
```

Where Z ~ N(0,1). Each call to the path generator produces N paths of T steps, stored
as a contiguous `[N x T]` float64 matrix. The `rayon` crate parallelizes path generation
across CPU cores.

#### Antithetic Variates

For every random draw Z, the antithetic path uses -Z. The N paths are therefore always
generated as N/2 pairs. Antithetic variates halve variance for symmetric payoffs at no
extra simulation cost.

### Merton Jump-Diffusion

Extends GBM with a compound Poisson jump component:

```
dS/S = (mu - lambda * k_bar) * dt + sigma * dW + J * dN
```

Where:
- `lambda` -- average number of jumps per year (Poisson intensity)
- `J` -- jump size, log-normally distributed: log(1+J) ~ N(mu_J, sigma_J^2)
- `k_bar = exp(mu_J + 0.5 * sigma_J^2) - 1` -- mean jump size (compensator)
- `dN` -- Poisson increment, 1 with probability `lambda * dt`, 0 otherwise

Jump-diffusion paths capture the fat tails and sudden dislocations characteristic of
crypto markets (exchange hacks, de-pegging events, regulatory announcements).

### Heston Stochastic Volatility

The Heston model couples asset price to a mean-reverting variance process (CIR):

```
dS = mu * S * dt + sqrt(v) * S * dW_S
dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW_v
corr(dW_S, dW_v) = rho
```

Parameters:
- `kappa` -- speed of mean reversion for variance
- `theta` -- long-run variance (theta = sigma_LR^2)
- `xi` -- volatility of variance ("vol of vol")
- `rho` -- correlation between price and variance shocks (typically negative for
  equities, near zero for crypto)

Euler-Maruyama discretization with full truncation (variance is floored at zero) to
prevent the Feller condition violation issues that cause negative variance in naive
discretizations.

### Correlated Multi-Asset Paths

For a portfolio of n assets with covariance matrix Sigma:

```
S_i(t+dt) = S_i(t) * exp((mu_i - 0.5*sigma_i^2)*dt + sigma_i*sqrt(dt)*(L*Z)_i)
```

Where L is the lower Cholesky factor of the correlation matrix: `Sigma = L * L^T`.
Cholesky decomposition is computed once at initialization and reused across all path
batches. For n assets, each step requires one L*Z matrix-vector product (O(n^2)).

---

## VaR Calculator (`var_calculator.rs`)

### Portfolio VaR

Value at Risk at confidence level c is the (1-c) quantile of the simulated portfolio
P&L distribution:

```
VaR_c = -Quantile(PnL, 1 - c)
```

Three confidence levels are computed simultaneously from the same path set:

| Level  | Confidence | Interpretation                                      |
|--------|------------|-----------------------------------------------------|
| 95%    | c = 0.95   | Day-to-day risk limit for position sizing           |
| 99%    | c = 0.99   | Regulatory capital floor reference                  |
| 99.9%  | c = 0.999  | Stress-level tail risk, extreme scenario planning   |

### CVaR (Expected Shortfall)

CVaR -- also called Conditional VaR or Expected Shortfall -- is the mean loss in the
tail beyond VaR:

```
CVaR_c = -E[PnL | PnL < -VaR_c]
       = -(1/(1-c)) * integral from -inf to -VaR_c of x * f(x) dx
```

CVaR is coherent (satisfies subadditivity) while VaR is not, making CVaR the preferred
risk measure for portfolio optimization. The implementation computes CVaR directly from
the sorted simulated P&L array.

### Component VaR

Decomposes portfolio VaR into per-instrument contributions:

```
CVaR_i = rho(R_i, R_portfolio) * VaR_i
```

Where `rho` is the correlation between instrument i's return and the portfolio return
across simulated paths. Component VaR sums to total portfolio VaR, enabling attribution
of risk budget consumption to individual positions.

### Marginal VaR

Sensitivity of portfolio VaR to a unit increase in position size for instrument i:

```
MVaR_i = d(VaR_portfolio) / d(w_i)
       = rho(R_i, R_portfolio) * VaR_portfolio / sigma_portfolio
```

Used by the position sizer to answer: "if I add one unit to position i, how much does
portfolio VaR increase?"

---

## Option Pricer (`option_pricer.rs`)

### European Options (GBM)

Standard risk-neutral pricing:

```
C = e^(-r*T) * E[max(S_T - K, 0)]
P = e^(-r*T) * E[max(K - S_T, 0)]
```

Computed as the discounted mean payoff across all simulated terminal prices S_T. The
Black-Scholes closed-form solution is also available and serves as the control variate
for variance reduction (see below).

### American Options (Longstaff-Schwartz)

The Longstaff-Schwartz least-squares Monte Carlo (LSM) algorithm prices American options
by backward induction:

1. Simulate N paths forward to expiry
2. At each exercise date t working backward from T-1 to 1:
   a. Identify in-the-money paths
   b. Regress continuation value against basis functions of current spot price
   c. Exercise where immediate payoff > estimated continuation value
3. Price = discounted expected cash flows under optimal stopping rule

**Basis functions** -- 3rd degree polynomial in the current asset price S_t:

```
E[V(t+1) | S_t] ~ beta_0 + beta_1*S_t + beta_2*S_t^2 + beta_3*S_t^3
```

The regression is fit via ordinary least squares on in-the-money paths only at each
exercise date. American option prices are consistently 1--3% above equivalent European
prices for at-the-money crypto options with time-to-expiry > 30 days.

### Barrier Options

**Knock-out** -- option expires worthless if the underlying ever crosses the barrier B
during [0, T]:

```
Payoff = max(S_T - K, 0) * 1{max(S_t) < B for all t in [0,T]}   (up-and-out call)
```

**Knock-in** -- option only activates if the barrier is crossed:

```
Payoff = max(S_T - K, 0) * 1{max(S_t) >= B for some t in [0,T]} (up-and-in call)
```

Barrier monitoring is performed at each simulated time step (daily for 252-step paths).
Continuous barrier approximation corrections are not currently applied -- discrete
monitoring introduces a small upward bias in knock-out prices.

---

## Simulation Configuration

### Default Path Counts

| Use Case                          | Path Count | Notes                              |
|-----------------------------------|------------|------------------------------------|
| Standard VaR / option pricing     | 10,000     | Sufficient for 99% VaR             |
| Stress scenarios                  | 100,000    | Required for 99.9% tail accuracy   |
| American option pricing (LSM)     | 50,000     | Regression stability requirement   |
| Real-time intraday VaR            | 5,000      | Latency budget constraint          |

---

## Variance Reduction Techniques

### Antithetic Variates

For any estimator theta_hat = (1/N) * sum f(Z_i), the antithetic estimator is:

```
theta_hat_AV = (1/N) * sum [ (f(Z_i) + f(-Z_i)) / 2 ]
```

Variance reduction factor = (1 + rho(f(Z), f(-Z))) / 2, where rho is the correlation
between paired outcomes. For monotone payoffs like vanilla calls, rho < 0 and variance
is always reduced.

### Control Variates (Black-Scholes as Control)

For European calls, the Black-Scholes price C_BS is known analytically. The control
variate estimator:

```
C_MC_CV = C_MC + beta * (C_BS - E_MC[C_BS])
```

The optimal beta minimizes variance: `beta* = Cov(C_MC, C_MC_BS) / Var(C_MC_BS)`.
Variance reduction of 70--90% is typical for near-the-money options.

### Stratified Sampling

The [0,1] uniform space is divided into N_strata equal intervals. One sample is drawn
uniformly from each stratum and transformed to the desired distribution. This ensures
uniform coverage of the distribution's support and eliminates the clustering that random
sampling produces in finite samples.

---

## Integration Points

### Python (PyO3)

`crypto_backtest_mc.py` calls the Rust engine through PyO3 bindings:

```python
from monte_carlo_engine import run_gbm_paths, compute_var, price_european

paths = run_gbm_paths(S0=50000.0, mu=0.0, sigma=0.65, T=252, n_paths=10000)
var_95, cvar_95 = compute_var(paths, confidence=0.95)
```

The PyO3 layer converts Python floats and numpy arrays to Rust types with zero-copy
where possible (numpy arrays are shared as slices).

### Julia (NumericalMethods.jl)

`NumericalMethods.jl` contains an **independent** quasi-Monte Carlo implementation using:
- **Halton sequences** -- low-discrepancy sequences based on prime-base van der Corput
  sequences. For dimension d, the d-th Halton sequence uses the d-th prime as its base.
- **Sobol sequences** -- a scrambled Sobol implementation for higher-dimensional problems
  (multi-asset paths, exotic option pricing)

Quasi-MC converges at O(log(N)^d / N) versus O(1/sqrt(N)) for standard MC, making it
substantially more efficient for smooth payoffs in moderate dimensions (d < 20).

---

## Performance Benchmarks

| Operation                                    | Single Thread | Rayon (multi-core) |
|----------------------------------------------|---------------|--------------------|
| 10,000 GBM paths, 252 steps                  | ~8 ms         | ~2 ms              |
| 10,000 Heston paths, 252 steps               | ~35 ms        | ~9 ms              |
| 10,000 jump-diffusion paths, 252 steps       | ~12 ms        | ~3 ms              |
| Portfolio VaR (10k paths, 5 assets)          | ~15 ms        | ~4 ms              |
| American option LSM (50k paths, 50 steps)    | ~180 ms       | ~45 ms             |

Benchmarked on an AMD Ryzen 9 7950X (16c/32t). The rayon speedup is slightly sub-linear
due to cache pressure from the large path matrices at high path counts.

---

## Stress Scenarios

The engine includes pre-defined historical and hypothetical stress scenarios that scale
input parameters to replicate specific market conditions.

### Historical Scenarios

| Scenario           | Period        | Key Parameters                                          |
|--------------------|---------------|---------------------------------------------------------|
| COVID crash        | Mar 2020      | sigma x3.5, mu = -0.12/day for 10 days                 |
| LUNA collapse      | May 2022      | Jump intensity x10, mu_J = -0.40, sigma_J = 0.25       |
| FTX contagion      | Nov 2022      | Correlation spike to 0.95 across all crypto assets     |

### Hypothetical Scenarios

| Scenario             | Description                                                |
|----------------------|------------------------------------------------------------|
| Flash crash -30%     | Instantaneous -30% price shock, no diffusion component     |
| Sustained bear -60%  | 180-day bear trend: mu = -0.005/day, sigma = 0.08/day      |

Stress scenarios always use 100,000 paths regardless of the default path count setting
to ensure the extreme tail (99.9% CVaR) is estimated with sufficient samples.

---

## Key Files

| Path                                                     | Purpose                              |
|----------------------------------------------------------|--------------------------------------|
| `crates/monte-carlo-engine/src/gbm_paths.rs`            | GBM, Merton, Heston, multi-asset     |
| `crates/monte-carlo-engine/src/var_calculator.rs`       | VaR, CVaR, Component VaR, Marginal   |
| `crates/monte-carlo-engine/src/option_pricer.rs`        | European, American LSM, barrier      |
| `crypto_backtest_mc.py`                                  | Python PyO3 integration layer        |
| `NumericalMethods.jl`                                    | Julia quasi-MC (Halton/Sobol)        |
