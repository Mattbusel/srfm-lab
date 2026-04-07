# Research Validation -- Out-of-Sample Framework

## Overview

The `research/validation/` package implements a rigorous out-of-sample validation
framework for all SRFM strategies and signals. The central design principle is that no
reported metric should be inflated by in-sample overfitting or multiple-testing bias.
The framework achieves this through three complementary mechanisms: combinatorial
purged cross-validation (CPCV), deflated Sharpe ratio (DSR) correction, and
Benjamini-Hochberg false discovery rate (FDR) control applied to every statistical test.

All validators are composable -- they can run independently or be chained inside a
walk-forward loop where CPCV folds drive the outer loop and individual tests run
within each fold.

---

## Purged K-Fold Cross-Validation -- `out_of_sample_validator.py`

### Lopez de Prado CPCV

Standard K-Fold CV applied to financial time series produces leaky folds because
overlapping label windows create serial correlation between train and test sets. The
CPCV method (Advances in Financial Machine Learning, Chapter 12) resolves this by:

1. Splitting the time series into `n_splits` non-overlapping blocks.
2. For each test combination of `n_test_splits` blocks, removing all training samples
   whose labels overlap with the test window.
3. Inserting an **embargo gap** of `embargo_pct` of the total series length on each
   side of every test block before discarding training samples.

The combinatorial structure means the number of distinct train/test paths is
`C(n_splits, n_test_splits)`, which for typical values (n_splits=6, n_test_splits=2)
produces 15 independent OOS paths. Each path gets its own Sharpe ratio, and the
distribution of path-level Sharpes feeds directly into the DSR calculation.

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_splits` | 6 | Number of time blocks |
| `n_test_splits` | 2 | Blocks held out per fold |
| `embargo_pct` | 0.01 | Fraction of series length used as embargo gap |
| `min_train_pct` | 0.50 | Reject folds where training set is below this fraction |

### Embargo Gap

The embargo gap prevents the most common form of leakage in financial ML: features
computed from a rolling window that spans the train/test boundary. If labels have a
holding period of H bars, the embargo must be at least H bars wide. The default
`embargo_pct=0.01` is conservative; strategies with long holding periods (e.g. weekly
rebalance) should increase this.

---

## Deflated Sharpe Ratio -- `out_of_sample_validator.py`

### The Multiple Testing Problem

When a researcher tests many parameter configurations and reports the best Sharpe ratio,
the reported Sharpe is upward-biased simply because the maximum of many draws from a
distribution exceeds the true mean. The Deflated Sharpe Ratio (Bailey & Lopez de Prado,
2014) corrects for this bias.

### DSR Calculation

Given:
- `SR` -- observed annualized Sharpe ratio of the best strategy
- `SR_benchmark` -- Sharpe ratio of a naive benchmark (default 0.0)
- `T` -- number of return observations
- `skewness` / `kurtosis` -- of the return series
- `N` -- number of independent trials tested (configurations evaluated)

The DSR is:

```
PSR(SR*) = Phi(  (SR - SR*) * sqrt(T - 1)
               / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2)  )
```

where `SR*` is the expected maximum Sharpe ratio across N independent trials (computed
from the expected maximum of a normal distribution). DSR is the probability that the
true Sharpe exceeds the benchmark, after adjusting for the number of trials tested.

A strategy passes validation if `DSR >= 0.95`.

### Minimum Track Record Length

`out_of_sample_validator.py` also computes the minimum number of observations required
to achieve statistical significance at the 95% level given the observed Sharpe and
return distribution moments. This is reported in the `ValidationReport` and used to
flag strategies with insufficient history.

### Haircut Sharpe

The Haircut Sharpe (Chordia et al.) further discounts the reported Sharpe by the
proportion attributable to data mining. SRFM uses a conservative haircut of 50% of the
excess Sharpe above zero when the number of trials exceeds 100.

---

## Causal Inference -- `causal_inference.py`

### Purpose

Correlation between a signal and future returns is not sufficient for a trading edge.
`causal_inference.py` provides four methods to probe whether the signal-return
relationship is causal or spurious.

### Granger Causality

`granger_test(x, y, max_lags)` fits a Vector Autoregression (VAR) with automatic lag
selection via AIC. It then tests whether lagged values of `x` Granger-cause `y` using
an F-test on the restricted vs. unrestricted model. The output is a p-value per lag
order and a recommended lag order from AIC minimization.

Limitations: Granger causality tests linear predictability, not structural causality.
Results should be read alongside the PSM and DiD tests.

### Propensity Score Matching (PSM)

`psm_test(features, treatment, outcome)` estimates the average treatment effect (ATE)
of a binary signal on forward returns by matching treated (signal=1) and control
(signal=0) observations on estimated propensity scores. Steps:

1. Fit a logistic regression to estimate `P(treatment=1 | features)`.
2. Match each treated obs to its nearest-neighbor control on propensity score.
3. Compute the difference in `outcome` across matched pairs.
4. Report the mean matched difference and a t-test p-value.

### Difference-in-Differences (DiD)

`did_test(panel, event_dates)` estimates the causal effect of a regime event (e.g. Fed
announcement, index rebalance) on returns using a two-way fixed-effects model:

```
return_it = alpha_i + gamma_t + beta * (treated_i * post_t) + epsilon_it
```

`beta` is the DiD estimator. Heteroskedasticity-robust standard errors (HC3) are used.

### Two-Stage Least Squares (2SLS)

`tsls_test(endog, exog, instruments)` addresses endogeneity between the signal and
returns (common when the signal is itself a price-derived quantity). The first stage
regresses the endogenous signal on instruments; the second stage uses fitted values
from stage 1 as regressors. Instruments must satisfy the exclusion restriction.

---

## Market Efficiency Tests -- `market_efficiency_tests.py`

These tests check whether the return series targeted by a strategy exhibits the
anomalies the strategy claims to exploit.

### Variance Ratio Test (Lo-MacKinlay)

`variance_ratio_test(returns, lags)` computes the ratio `VR(q) = Var(r_q) / (q * Var(r_1))`
for each lag `q`. Under the random walk null, `VR(q) = 1`. A Z-statistic is computed
using heteroskedasticity-consistent standard errors (as in Lo-MacKinlay 1988). The test
is run for lags `[2, 4, 8, 16]` and results are reported jointly.

### Runs Test

`runs_test(returns)` counts the number of consecutive positive and negative return
runs. Under the null of independence, the number of runs is normally distributed with
known mean and variance. A Z-statistic above 1.96 (in absolute value) indicates
non-random structure. This is a fast non-parametric complement to the variance ratio test.

### GPH Long Memory (Geweke-Porter-Hudak)

`gph_test(returns, bandwidth_exp)` estimates the fractional integration parameter `d`
of the return series using the log-periodogram regression of Geweke and Porter-Hudak
(1983). The bandwidth is set to `T^bandwidth_exp` where `bandwidth_exp` defaults to
0.50. A significant positive `d` (e.g. d > 0.10, p < 0.05) is evidence of long memory,
which may indicate a persistent mean-reversion or momentum signal.

### Spectral Density Analysis

`spectral_test(returns, nperseg)` uses Welch's method to estimate the power spectral
density of the return series. Peaks at specific frequencies are reported and can be
used to tune the periodicity assumptions in BH signal parameters.

---

## Performance Persistence -- `performance_persistence.py`

### Contingency Table

`contingency_persistence(rolling_sharpes, n_periods)` partitions strategies (or
parameter configurations) into winners (top half by Sharpe) and losers (bottom half)
in period T, then checks whether winners repeat in period T+1. A 2x2 contingency table
is constructed and tested with Chi-squared and Fisher's exact test. The cross-product
ratio (CPR) exceeding 1.0 is evidence of persistence.

### Spearman IC Stability

`ic_stability(factor_returns, forward_returns, window)` computes the rolling information
coefficient (Spearman rank correlation between predicted and realized returns) over a
rolling window. Stability is quantified as the mean IC minus one standard deviation of
IC (IC_IR). An IC_IR above 0.50 is considered stable.

### Information Ratio Rolling Window

`rolling_ir(returns, benchmark_returns, window)` computes the rolling Information Ratio
(`IR = mean(active_return) / std(active_return)`) over a moving window. Regime breaks
in IR are detected using a CUSUM statistic. A persistent IR above 0.50 on a 252-day
rolling window is the primary persistence acceptance criterion.

---

## Benjamini-Hochberg FDR Correction

All p-values produced by the modules above are collected into a flat list and passed
through `bh_correction(p_values, q=0.10)` before any test is flagged as significant.

The BH procedure:

1. Sort p-values in ascending order: `p_(1) <= p_(2) <= ... <= p_(m)`.
2. Find the largest `k` such that `p_(k) <= (k / m) * q`.
3. Reject the null for all hypotheses `1, ..., k`.

With `q = 0.10` (the default FDR threshold), at most 10% of rejected hypotheses are
expected to be false positives. This is less conservative than Bonferroni but controls
the expected proportion of false discoveries rather than the probability of any false
discovery.

Q-values (the BH-adjusted p-values) are stored in `ValidationReport` alongside the
raw p-values so downstream consumers can apply a different threshold without re-running
tests.

---

## Walk-Forward Integration

The canonical validation workflow is:

```
for fold in cpcv.generate_folds(returns, features):
    train_data, test_data, embargo = fold
    model.fit(train_data)
    predictions = model.predict(test_data)
    fold_report = validator.evaluate(predictions, test_data)
    reports.append(fold_report)

final_report = ValidationReport.aggregate(reports)
final_report.apply_bh_correction(q=0.10)
```

Within each fold, causal inference and efficiency tests run on the test window only.
DSR is computed once across all fold Sharpes, treating each fold path as an independent
trial (N = number of CPCV paths).

---

## ValidationReport Dataclass

`ValidationReport` is the output container for a complete validation run.

```python
@dataclass
class ValidationReport:
    strategy_id:       str
    n_folds:           int
    fold_sharpes:      list[float]
    fold_max_dd:       list[float]
    dsr:               float
    dsr_pass:          bool           # True if DSR >= 0.95
    min_track_length:  int            # minimum observations needed
    granger_pvals:     dict[int, float]
    psm_ate:           float
    psm_pval:          float
    did_beta:          float
    did_pval:          float
    tsls_coef:         float
    tsls_pval:         float
    vr_stats:          dict[int, float]
    runs_z:            float
    gph_d:             float
    gph_pval:          float
    persistence_cpr:   float
    ic_ir:             float
    bh_q_values:       dict[str, float]   # test name -> q-value
    created_at:        datetime
```

`ValidationReport` provides `to_json()` and `from_json()` methods for persistence, and
`summary()` which returns a human-readable table of pass/fail status per test.
