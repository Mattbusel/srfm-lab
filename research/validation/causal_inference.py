"""
research/validation/causal_inference.py -- causal analysis tools for SRFM.

Tests whether relationships between features and returns are causal, not merely
correlational. Key application: does BH mass CAUSE future returns, or is the
relationship spurious?

Methods implemented:
  - Granger causality (VAR-based F-test)
  - Propensity score matching (logistic regression + nearest-neighbor)
  - Difference-in-differences (DiD)
  - Instrumental variables / 2SLS
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GrangerResult:
    """Result of Granger causality test."""
    f_stat: float
    p_value: float
    lag: int
    does_x_cause_y: bool
    # F-stats at each tested lag for diagnostics
    lag_f_stats: List[float] = field(default_factory=list)
    lag_p_values: List[float] = field(default_factory=list)


@dataclass
class PSMResult:
    """Result of propensity score matching."""
    treated_outcomes: pd.Series
    matched_control_outcomes: pd.Series
    ate: float          # Average treatment effect on the treated
    p_value: float
    std_error: float
    t_stat: float
    n_treated: int
    n_matched: int
    # Propensity score balance diagnostics
    standardized_mean_diff_before: float = 0.0
    standardized_mean_diff_after: float = 0.0


@dataclass
class DiffInDiffResult:
    """Result of difference-in-differences estimation."""
    ate: float
    std_error: float
    t_stat: float
    p_value: float
    # Component means for transparency
    pre_treated_mean: float = 0.0
    post_treated_mean: float = 0.0
    pre_control_mean: float = 0.0
    post_control_mean: float = 0.0
    # Parallel trends pre-test (should be ~0 if assumption holds)
    parallel_trends_stat: Optional[float] = None


@dataclass
class IVResult:
    """Result of instrumental variable (2SLS) estimation."""
    beta_iv: float          # IV estimate of causal effect
    std_error: float
    t_stat: float
    p_value: float
    # First-stage diagnostics
    first_stage_f: float    # Should be > 10 for strong instrument
    first_stage_r2: float
    # OLS estimate for comparison
    beta_ols: float
    beta_ols_std_error: float
    # Hausman test: significant means IV and OLS differ (endogeneity present)
    hausman_stat: float
    hausman_p_value: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    OLS via normal equations. Returns (coefficients, residuals, r_squared).
    X should NOT include the intercept column -- we add it here.
    """
    n = len(y)
    X_full = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(X_full.shape[1])
    y_hat = X_full @ beta
    residuals = y - y_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return beta, residuals, r2


def _ols_std_errors(X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """HC0 heteroskedasticity-robust standard errors."""
    n = len(residuals)
    X_full = np.column_stack([np.ones(n), X])
    k = X_full.shape[1]
    try:
        XtX_inv = np.linalg.inv(X_full.T @ X_full)
        # Sandwich estimator: (X'X)^{-1} X' diag(e^2) X (X'X)^{-1}
        meat = X_full.T @ np.diag(residuals ** 2) @ X_full
        var_beta = XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.diag(var_beta))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return se


def _logistic_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _logistic_fit(X: np.ndarray, y: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """
    Fit logistic regression via L-BFGS-B. X should NOT include intercept.
    Returns coefficient vector (including intercept as first element).
    """
    n = len(y)
    X_full = np.column_stack([np.ones(n), X])
    k = X_full.shape[1]

    def neg_log_likelihood(beta: np.ndarray) -> float:
        p = _logistic_sigmoid(X_full @ beta)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def gradient(beta: np.ndarray) -> np.ndarray:
        p = _logistic_sigmoid(X_full @ beta)
        return X_full.T @ (p - y)

    result = minimize(
        neg_log_likelihood,
        x0=np.zeros(k),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    return result.x


def _f_test(rss_restricted: float, rss_unrestricted: float, df_restricted: int,
            df_unrestricted: int, n: int) -> Tuple[float, float]:
    """
    F-test for joint significance of additional regressors.
    Returns (f_stat, p_value).
    """
    q = df_restricted - df_unrestricted  # number of restrictions
    if q <= 0 or rss_unrestricted < 1e-15:
        return 0.0, 1.0
    f = ((rss_restricted - rss_unrestricted) / q) / (rss_unrestricted / (n - df_unrestricted - 1))
    p = float(1.0 - stats.f.cdf(f, q, n - df_unrestricted - 1))
    return float(f), p


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CausalAnalyzer:
    """
    Causal inference tools for quantitative research.

    All methods are stateless -- instantiate once and call methods freely.

    Usage in SRFM context:
      analyzer = CausalAnalyzer()
      result = analyzer.granger_causality(bh_mass_series, forward_returns, max_lag=10)
      if result.does_x_cause_y:
          # BH mass has predictive causal content -- proceed to signal construction
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """
        Parameters
        ----------
        alpha : float
            Significance level for hypothesis tests (default 0.05).
        """
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Granger Causality
    # ------------------------------------------------------------------

    def granger_causality(
        self,
        x: pd.Series,
        y: pd.Series,
        max_lag: int = 5,
    ) -> GrangerResult:
        """
        Test whether x Granger-causes y using VAR regression F-test.

        For each lag L from 1 to max_lag:
          Restricted model:  y_t = a0 + sum_{i=1}^{L} a_i * y_{t-i} + eps
          Unrestricted model: y_t = a0 + sum_{i=1}^{L} a_i * y_{t-i}
                                       + sum_{i=1}^{L} b_i * x_{t-i} + eps

        The best lag is selected by AIC on the unrestricted model.
        The reported result is for the AIC-optimal lag.

        Parameters
        ----------
        x : pd.Series
            Potential cause variable (must be same length as y, aligned).
        y : pd.Series
            Effect variable.
        max_lag : int
            Maximum number of lags to test.

        Returns
        -------
        GrangerResult
        """
        if len(x) != len(y):
            raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")
        if max_lag < 1:
            raise ValueError("max_lag must be >= 1")

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        # Remove NaNs jointly
        valid = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[valid]
        y_arr = y_arr[valid]
        n_total = len(y_arr)

        if n_total < 2 * max_lag + 10:
            warnings.warn(
                f"Sample too small for max_lag={max_lag}; reducing to {max(1, n_total // 4)}",
                UserWarning,
                stacklevel=2,
            )
            max_lag = max(1, n_total // 4)

        lag_f_stats: List[float] = []
        lag_p_values: List[float] = []
        aic_values: List[float] = []

        for lag in range(1, max_lag + 1):
            n = n_total - lag  # usable observations

            # Build lagged y matrix (for restricted model)
            Y_lags = np.column_stack([y_arr[lag - i - 1: n_total - i - 1] for i in range(lag)])
            y_dep = y_arr[lag:]

            # Restricted model: y ~ y_lags
            _, resid_r, _ = _ols_fit(Y_lags, y_dep)
            rss_r = float(np.sum(resid_r ** 2))
            df_r = lag  # number of slope parameters (excl intercept)

            # Unrestricted model: y ~ y_lags + x_lags
            X_lags = np.column_stack([x_arr[lag - i - 1: n_total - i - 1] for i in range(lag)])
            XY_lags = np.column_stack([Y_lags, X_lags])
            _, resid_u, _ = _ols_fit(XY_lags, y_dep)
            rss_u = float(np.sum(resid_u ** 2))
            df_u = 2 * lag  # y_lags + x_lags

            f_stat, p_val = _f_test(rss_r, rss_u, df_r, df_u, n)
            lag_f_stats.append(f_stat)
            lag_p_values.append(p_val)

            # AIC on unrestricted model
            k_params = 2 * lag + 1  # intercept + 2*lag slope params
            aic = n * np.log(max(rss_u / n, 1e-15)) + 2 * k_params
            aic_values.append(aic)

        best_lag = int(np.argmin(aic_values)) + 1  # 1-indexed
        best_f = lag_f_stats[best_lag - 1]
        best_p = lag_p_values[best_lag - 1]

        return GrangerResult(
            f_stat=best_f,
            p_value=best_p,
            lag=best_lag,
            does_x_cause_y=best_p < self.alpha,
            lag_f_stats=lag_f_stats,
            lag_p_values=lag_p_values,
        )

    # ------------------------------------------------------------------
    # Propensity Score Matching
    # ------------------------------------------------------------------

    def propensity_score_matching(
        self,
        treated_df: pd.DataFrame,
        control_df: pd.DataFrame,
        covariates: List[str],
        outcome_col: str = "outcome",
        caliper: Optional[float] = None,
        random_seed: int = 42,
    ) -> PSMResult:
        """
        Estimate ATT via propensity score matching.

        Steps:
          1. Pool treated/control, fit logistic regression to predict treatment.
          2. For each treated unit, find nearest-neighbor control on logit(p).
          3. Compute ATE as mean(treated outcomes) - mean(matched control outcomes).
          4. Paired t-test for statistical significance.

        Parameters
        ----------
        treated_df : pd.DataFrame
            Treated group rows. Must contain covariates and outcome_col.
        control_df : pd.DataFrame
            Control group rows. Must contain covariates and outcome_col.
        covariates : List[str]
            Column names to use as matching covariates.
        outcome_col : str
            Column name of the outcome variable.
        caliper : float, optional
            Maximum allowed distance on logit(propensity) scale. Units matched
            outside caliper are dropped.
        random_seed : int
            Seed for tie-breaking.

        Returns
        -------
        PSMResult
        """
        rng = np.random.default_rng(random_seed)

        for col in covariates + [outcome_col]:
            if col not in treated_df.columns:
                raise ValueError(f"Column '{col}' missing from treated_df")
            if col not in control_df.columns:
                raise ValueError(f"Column '{col}' missing from control_df")

        # Build pooled dataset
        treated_df = treated_df.copy()
        control_df = control_df.copy()
        treated_df["_treatment"] = 1
        control_df["_treatment"] = 0

        pooled = pd.concat([treated_df, control_df], ignore_index=True)
        pooled = pooled.dropna(subset=covariates + [outcome_col])

        X_cov = pooled[covariates].values.astype(float)
        # Standardize covariates for logistic regression stability
        X_mean = X_cov.mean(axis=0)
        X_std = np.where(X_cov.std(axis=0) > 1e-10, X_cov.std(axis=0), 1.0)
        X_cov_std = (X_cov - X_mean) / X_std
        T = pooled["_treatment"].values.astype(float)

        # Fit propensity score model
        beta = _logistic_fit(X_cov_std, T)
        X_full = np.column_stack([np.ones(len(T)), X_cov_std])
        propensity = _logistic_sigmoid(X_full @ beta)
        propensity = np.clip(propensity, 1e-6, 1 - 1e-6)
        logit_p = np.log(propensity / (1 - propensity))

        pooled = pooled.copy()
        pooled["_propensity"] = propensity
        pooled["_logit_p"] = logit_p

        treated_rows = pooled[pooled["_treatment"] == 1].copy()
        control_rows = pooled[pooled["_treatment"] == 0].copy()

        if len(treated_rows) == 0 or len(control_rows) == 0:
            raise ValueError("Need at least one treated and one control unit after NA removal")

        # Compute standardized mean difference BEFORE matching
        def _smd(t_vals: np.ndarray, c_vals: np.ndarray) -> float:
            pooled_std = np.sqrt((np.var(t_vals) + np.var(c_vals)) / 2 + 1e-15)
            return float(np.abs(np.mean(t_vals) - np.mean(c_vals)) / pooled_std)

        smd_before = float(np.mean([
            _smd(treated_rows[c].values, control_rows[c].values) for c in covariates
        ]))

        # Nearest-neighbor matching on logit propensity (with replacement by default)
        control_logit = control_rows["_logit_p"].values
        control_outcomes = control_rows[outcome_col].values
        treated_logit = treated_rows["_logit_p"].values
        treated_outcomes = treated_rows[outcome_col].values

        matched_control_outcomes_list: List[float] = []
        kept_treated_outcomes_list: List[float] = []

        for i, (t_logit, t_out) in enumerate(zip(treated_logit, treated_outcomes)):
            dists = np.abs(control_logit - t_logit)
            # Break ties randomly
            min_dist = dists.min()
            candidates = np.where(dists == min_dist)[0]
            match_idx = int(rng.choice(candidates))

            if caliper is not None and min_dist > caliper:
                continue  # Outside caliper -- drop this treated unit

            matched_control_outcomes_list.append(float(control_outcomes[match_idx]))
            kept_treated_outcomes_list.append(float(t_out))

        if len(kept_treated_outcomes_list) == 0:
            raise ValueError("No matches within caliper -- try widening caliper")

        matched_control_arr = np.array(matched_control_outcomes_list)
        kept_treated_arr = np.array(kept_treated_outcomes_list)

        # ATE on treated (ATT)
        diffs = kept_treated_arr - matched_control_arr
        ate = float(np.mean(diffs))
        se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        t_stat = ate / se if se > 1e-15 else 0.0
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(diffs) - 1)))

        # SMD after matching -- compare treated covariate means to matched control means
        matched_control_rows = []
        for t_logit in treated_logit[: len(kept_treated_arr)]:
            dists = np.abs(control_logit - t_logit)
            match_idx = int(np.argmin(dists))
            matched_control_rows.append(control_rows.iloc[match_idx][covariates].values.astype(float))

        if len(matched_control_rows) > 0:
            matched_control_cov = np.vstack(matched_control_rows)
            treated_cov_sub = treated_rows[covariates].values[: len(kept_treated_arr)]
            smd_after = float(np.mean([
                _smd(treated_cov_sub[:, j], matched_control_cov[:, j])
                for j in range(len(covariates))
            ]))
        else:
            smd_after = smd_before

        return PSMResult(
            treated_outcomes=pd.Series(kept_treated_arr),
            matched_control_outcomes=pd.Series(matched_control_arr),
            ate=ate,
            p_value=p_val,
            std_error=se,
            t_stat=t_stat,
            n_treated=len(treated_rows),
            n_matched=len(kept_treated_arr),
            standardized_mean_diff_before=smd_before,
            standardized_mean_diff_after=smd_after,
        )

    # ------------------------------------------------------------------
    # Difference-in-Differences
    # ------------------------------------------------------------------

    def diff_in_diff(
        self,
        pre_treated: pd.Series,
        post_treated: pd.Series,
        pre_control: pd.Series,
        post_control: pd.Series,
        pre_pre_treated: Optional[pd.Series] = None,
        pre_pre_control: Optional[pd.Series] = None,
    ) -> DiffInDiffResult:
        """
        Classic 2x2 difference-in-differences estimator.

        ATE = (E[Y_post | T=1] - E[Y_pre | T=1]) - (E[Y_post | T=0] - E[Y_pre | T=0])

        Optionally tests parallel trends assumption using a pre-treatment period:
          If pre_pre_treated and pre_pre_control are given, a placebo DiD is run
          on the two pre-treatment periods. A significant placebo suggests
          parallel trends may not hold.

        Parameters
        ----------
        pre_treated : pd.Series
            Pre-treatment outcomes for treated group.
        post_treated : pd.Series
            Post-treatment outcomes for treated group.
        pre_control : pd.Series
            Pre-treatment outcomes for control group.
        post_control : pd.Series
            Post-treatment outcomes for control group.
        pre_pre_treated : pd.Series, optional
            Earlier pre-treatment period for treated (parallel trends test).
        pre_pre_control : pd.Series, optional
            Earlier pre-treatment period for control (parallel trends test).

        Returns
        -------
        DiffInDiffResult
        """
        pre_t_mean = float(np.nanmean(pre_treated))
        post_t_mean = float(np.nanmean(post_treated))
        pre_c_mean = float(np.nanmean(pre_control))
        post_c_mean = float(np.nanmean(post_control))

        ate = (post_t_mean - pre_t_mean) - (post_c_mean - pre_c_mean)

        # Standard error via regression formulation
        # Stack all observations with group (D) and time (T) indicators
        n_pre_t = len(pre_treated.dropna())
        n_post_t = len(post_treated.dropna())
        n_pre_c = len(pre_control.dropna())
        n_post_c = len(post_control.dropna())

        # Delta method: Var(DiD) = Var(post_t)/n + Var(pre_t)/n + Var(post_c)/n + Var(pre_c)/n
        var_did = (
            np.nanvar(post_treated, ddof=1) / max(n_post_t, 1)
            + np.nanvar(pre_treated, ddof=1) / max(n_pre_t, 1)
            + np.nanvar(post_control, ddof=1) / max(n_post_c, 1)
            + np.nanvar(pre_control, ddof=1) / max(n_pre_c, 1)
        )
        se = float(np.sqrt(max(var_did, 1e-15)))
        t_stat = ate / se if se > 1e-15 else 0.0
        df = max(n_pre_t + n_post_t + n_pre_c + n_post_c - 4, 1)
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df)))

        # Parallel trends test (placebo DiD on pre periods)
        pt_stat = None
        if pre_pre_treated is not None and pre_pre_control is not None:
            placebo = self.diff_in_diff(
                pre_treated=pre_pre_treated,
                post_treated=pre_treated,
                pre_control=pre_pre_control,
                post_control=pre_control,
            )
            pt_stat = placebo.t_stat

        return DiffInDiffResult(
            ate=ate,
            std_error=se,
            t_stat=t_stat,
            p_value=p_val,
            pre_treated_mean=pre_t_mean,
            post_treated_mean=post_t_mean,
            pre_control_mean=pre_c_mean,
            post_control_mean=post_c_mean,
            parallel_trends_stat=pt_stat,
        )

    # ------------------------------------------------------------------
    # Instrumental Variables / 2SLS
    # ------------------------------------------------------------------

    def instrumental_variable(
        self,
        y: pd.Series,
        x: pd.Series,
        z: pd.Series,
        controls: Optional[pd.DataFrame] = None,
    ) -> IVResult:
        """
        Two-stage least squares (2SLS) IV estimator.

        First stage:  x = a + b*z + c'*controls + v
        Second stage: y = d + e*x_hat + f'*controls + eps

        The IV estimator consistently estimates the causal effect of x on y
        when z is a valid instrument (relevant, exogenous, exclusion restriction).

        Hausman test: H0: OLS is consistent (no endogeneity). Rejection means
        endogeneity is present and IV estimates differ significantly from OLS.

        Parameters
        ----------
        y : pd.Series
            Outcome variable.
        x : pd.Series
            Endogenous regressor.
        z : pd.Series
            Instrument (must satisfy relevance and exogeneity).
        controls : pd.DataFrame, optional
            Exogenous control variables included in both stages.

        Returns
        -------
        IVResult
        """
        # Align and drop NAs
        if controls is not None:
            data = pd.concat([y, x, z, controls], axis=1).dropna()
            y_arr = data.iloc[:, 0].values.astype(float)
            x_arr = data.iloc[:, 1].values.astype(float)
            z_arr = data.iloc[:, 2].values.astype(float)
            ctrl_arr = data.iloc[:, 3:].values.astype(float)
        else:
            data = pd.concat([y, x, z], axis=1).dropna()
            y_arr = data.iloc[:, 0].values.astype(float)
            x_arr = data.iloc[:, 1].values.astype(float)
            z_arr = data.iloc[:, 2].values.astype(float)
            ctrl_arr = np.empty((len(y_arr), 0))

        n = len(y_arr)
        if n < 10:
            raise ValueError(f"Need at least 10 observations for IV, got {n}")

        # Build first-stage regressors: [z, controls]
        if ctrl_arr.shape[1] > 0:
            fs_X = np.column_stack([z_arr, ctrl_arr])
        else:
            fs_X = z_arr.reshape(-1, 1)

        # First stage: x ~ z + controls
        beta_fs, resid_fs, r2_fs = _ols_fit(fs_X, x_arr)
        n_fs = len(x_arr)
        X_fs_full = np.column_stack([np.ones(n_fs), fs_X])
        x_hat = X_fs_full @ beta_fs

        # First-stage F-stat for instrument relevance
        # Compare x ~ controls (restricted) vs x ~ z + controls (unrestricted)
        if ctrl_arr.shape[1] > 0:
            _, resid_r_fs, _ = _ols_fit(ctrl_arr, x_arr)
        else:
            resid_r_fs = x_arr - np.mean(x_arr)
        rss_r_fs = float(np.sum(resid_r_fs ** 2))
        rss_u_fs = float(np.sum(resid_fs ** 2))
        df_r_fs = ctrl_arr.shape[1]
        df_u_fs = ctrl_arr.shape[1] + 1  # + instrument
        f1_stat, _ = _f_test(rss_r_fs, rss_u_fs, df_r_fs, df_u_fs, n)

        # Second stage: y ~ x_hat + controls
        if ctrl_arr.shape[1] > 0:
            ss_X = np.column_stack([x_hat, ctrl_arr])
        else:
            ss_X = x_hat.reshape(-1, 1)
        beta_ss, resid_ss, _ = _ols_fit(ss_X, y_arr)
        se_ss = _ols_std_errors(ss_X, resid_ss)

        beta_iv = float(beta_ss[1])   # coefficient on x_hat (index 0 = intercept)
        se_iv = float(se_ss[1])
        t_iv = beta_iv / se_iv if se_iv > 1e-15 else 0.0
        p_iv = float(2 * (1 - stats.t.cdf(abs(t_iv), df=n - ss_X.shape[1] - 1)))

        # OLS for comparison: y ~ x + controls
        if ctrl_arr.shape[1] > 0:
            ols_X = np.column_stack([x_arr, ctrl_arr])
        else:
            ols_X = x_arr.reshape(-1, 1)
        beta_ols_vec, resid_ols, _ = _ols_fit(ols_X, y_arr)
        se_ols_vec = _ols_std_errors(ols_X, resid_ols)
        beta_ols_val = float(beta_ols_vec[1])
        se_ols_val = float(se_ols_vec[1])

        # Hausman test: H0: beta_OLS == beta_IV (no endogeneity)
        # Test stat: (beta_IV - beta_OLS)^2 / |Var(beta_IV) - Var(beta_OLS)|
        var_diff = abs(se_iv ** 2 - se_ols_val ** 2)
        if var_diff > 1e-15:
            hausman = (beta_iv - beta_ols_val) ** 2 / var_diff
            hausman_p = float(1 - stats.chi2.cdf(hausman, df=1))
        else:
            hausman = 0.0
            hausman_p = 1.0

        return IVResult(
            beta_iv=beta_iv,
            std_error=se_iv,
            t_stat=t_iv,
            p_value=p_iv,
            first_stage_f=float(f1_stat),
            first_stage_r2=float(r2_fs),
            beta_ols=beta_ols_val,
            beta_ols_std_error=se_ols_val,
            hausman_stat=float(hausman),
            hausman_p_value=hausman_p,
        )

    # ------------------------------------------------------------------
    # SRFM-specific application: BH mass -> returns causality
    # ------------------------------------------------------------------

    def test_bh_mass_causality(
        self,
        bh_mass: pd.Series,
        forward_returns: pd.Series,
        max_lag: int = 10,
        size_proxy: Optional[pd.Series] = None,
    ) -> dict:
        """
        Test whether BH mass Granger-causes forward returns (vs mere correlation).

        If size_proxy is provided, also runs IV with size as an instrument for BH mass,
        exploiting the fact that galaxy size is correlated with BH mass but may be
        more exogenous to short-term return shocks.

        Parameters
        ----------
        bh_mass : pd.Series
            Log BH mass series.
        forward_returns : pd.Series
            Forward return series (same index as bh_mass).
        max_lag : int
            Maximum lag for Granger test.
        size_proxy : pd.Series, optional
            Galaxy size or similar instrument.

        Returns
        -------
        dict with keys: granger, iv (if size_proxy given), summary_text
        """
        granger_result = self.granger_causality(bh_mass, forward_returns, max_lag=max_lag)

        results: dict = {"granger": granger_result}

        if size_proxy is not None:
            iv_result = self.instrumental_variable(
                y=forward_returns,
                x=bh_mass,
                z=size_proxy,
            )
            results["iv"] = iv_result
            iv_causal = iv_result.p_value < self.alpha
            instrument_strong = iv_result.first_stage_f > 10.0
            endogenous = iv_result.hausman_p_value < self.alpha
        else:
            iv_causal = None
            instrument_strong = None
            endogenous = None

        lines = [
            "=== BH Mass Causality Analysis ===",
            f"Granger causality (lag={granger_result.lag}): "
            f"F={granger_result.f_stat:.3f}, p={granger_result.p_value:.4f}, "
            f"causes={'YES' if granger_result.does_x_cause_y else 'NO'}",
        ]
        if iv_causal is not None:
            iv_res = results["iv"]
            lines += [
                f"IV (2SLS) beta={iv_res.beta_iv:.4f}, p={iv_res.p_value:.4f}, "
                f"causal={'YES' if iv_causal else 'NO'}",
                f"First-stage F={iv_res.first_stage_f:.2f} "
                f"({'strong' if instrument_strong else 'WEAK'} instrument)",
                f"Hausman p={iv_res.hausman_p_value:.4f} "
                f"({'endogeneity detected' if endogenous else 'no endogeneity'})",
            ]

        results["summary_text"] = "\n".join(lines)
        return results
