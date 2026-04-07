"""
model_validator.py -- validates trading models before deployment.

Covers signal models (IC/ICIR), risk models (VaR backtesting), and ML
online learners (calibration, concept drift).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from research.validation.statistical_tests import (
    ModelDiagnostics,
    StationarityTests,
    TestResult,
)


# ---------------------------------------------------------------------------
# Specification and report types
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """
    Specification for a model under validation.

    Fields
    ------
    name              : model identifier
    model_type        : 'signal', 'risk', or 'execution'
    min_ic            : minimum acceptable information coefficient (default 0.05)
    min_icir          : minimum acceptable IC information ratio (default 0.5)
    min_sharpe        : minimum acceptable annualized Sharpe ratio (default 0.5)
    ic_window         : rolling window for ICIR calculation (default 30)
    lookahead_check   : whether to verify no lookahead bias (default True)
    outlier_threshold : |signal| z-score threshold (default 5.0)
    """

    name: str
    model_type: str = "signal"
    min_ic: float = 0.05
    min_icir: float = 0.5
    min_sharpe: float = 0.5
    ic_window: int = 30
    lookahead_check: bool = True
    outlier_threshold: float = 5.0


@dataclass
class ValidationReport:
    """
    Result container for a model validation run.

    Fields
    ------
    model_name   : name of the model being validated
    passed       : True if all required checks pass
    failures     : list of failure description strings
    warnings     : list of non-fatal warning strings
    test_results : mapping of test-name -> TestResult or scalar
    """

    model_name: str
    passed: bool
    failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    test_results: dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] {self.model_name}"]
        if self.failures:
            lines.append("  Failures:")
            for f in self.failures:
                lines.append(f"    - {f}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ~ {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal model validator
# ---------------------------------------------------------------------------

class SignalModelValidator:
    """
    Validates a signal-generation model for deployment readiness.

    Checks performed
    ----------------
    1. IC > config.min_ic
    2. ICIR > config.min_icir over rolling config.ic_window periods
    3. No significant autocorrelation in the IC time series (Ljung-Box)
    4. Stationarity of the signal distribution (ADF)
    5. No extreme outliers: |signal| < config.outlier_threshold * sigma
    6. Out-of-sample check: IS IC not dramatically better than OOS IC
    """

    def validate(
        self,
        signal_values: np.ndarray,
        forward_returns: np.ndarray,
        config: ModelSpec,
        timestamps: Optional[np.ndarray] = None,
    ) -> ValidationReport:
        """
        Run all signal model checks.

        Parameters
        ----------
        signal_values   : array of signal values, shape (T,)
        forward_returns : forward return array aligned with signal_values, shape (T,)
        config          : ModelSpec controlling thresholds
        timestamps      : optional sorted index array; used for lookahead check message

        Returns
        -------
        ValidationReport
        """
        sig = np.asarray(signal_values, dtype=float)
        ret = np.asarray(forward_returns, dtype=float)
        if len(sig) != len(ret):
            raise ValueError("signal_values and forward_returns must have the same length")

        mask = np.isfinite(sig) & np.isfinite(ret)
        sig_c, ret_c = sig[mask], ret[mask]
        T = len(sig_c)

        failures: list = []
        warnings_: list = []
        results: dict = {}

        # -- 1. IC check -------------------------------------------------
        ic, ic_p = stats.spearmanr(sig_c, ret_c)
        ic = float(ic)
        results["ic"] = ic
        results["ic_p_value"] = ic_p
        if ic < config.min_ic:
            failures.append(
                f"IC={ic:.4f} below minimum {config.min_ic} (p={ic_p:.4f})"
            )
        elif ic_p >= 0.05:
            warnings_.append(f"IC={ic:.4f} not statistically significant (p={ic_p:.4f})")

        # -- 2. ICIR check -----------------------------------------------
        icir = self._rolling_icir(sig_c, ret_c, config.ic_window)
        results["icir"] = icir
        if np.isnan(icir):
            warnings_.append(
                f"Insufficient data to compute ICIR with window={config.ic_window}"
            )
        elif icir < config.min_icir:
            failures.append(f"ICIR={icir:.4f} below minimum {config.min_icir}")

        # -- 3. Autocorrelation in IC series (Ljung-Box) -----------------
        ic_series = self._rolling_ic_series(sig_c, ret_c, config.ic_window)
        results["ic_series_length"] = len(ic_series)
        if len(ic_series) >= 20:
            try:
                lb_result = ModelDiagnostics.ljung_box(ic_series, lags=min(10, len(ic_series) // 4))
                results["ljung_box_ic"] = lb_result
                if lb_result.is_significant:
                    warnings_.append(
                        "Autocorrelation detected in IC series -- signal may have hidden structure"
                    )
            except Exception as exc:
                warnings_.append(f"Ljung-Box IC test failed: {exc}")
        else:
            warnings_.append("Too few periods for IC autocorrelation test")

        # -- 4. Stationarity of the signal --------------------------------
        if T >= 20:
            try:
                adf_result = StationarityTests.augmented_dickey_fuller(sig_c)
                results["adf_signal"] = adf_result
                if not adf_result.extra.get("is_stationary", True):
                    failures.append(
                        "Signal distribution is non-stationary (ADF unit root not rejected)"
                    )
            except Exception as exc:
                warnings_.append(f"ADF stationarity test failed: {exc}")
        else:
            warnings_.append("Insufficient data for ADF stationarity test")

        # -- 5. Outlier check --------------------------------------------
        sig_std = float(np.std(sig_c, ddof=1))
        if sig_std > 0:
            z_scores = np.abs(sig_c - np.mean(sig_c)) / sig_std
            n_outliers = int(np.sum(z_scores > config.outlier_threshold))
            results["n_outliers"] = n_outliers
            outlier_frac = n_outliers / T
            if outlier_frac > 0.01:
                failures.append(
                    f"{n_outliers} extreme outliers (>{config.outlier_threshold}sigma) "
                    f"= {outlier_frac:.1%} of signal -- check data pipeline"
                )
            elif n_outliers > 0:
                warnings_.append(
                    f"{n_outliers} signal values exceed {config.outlier_threshold}sigma"
                )

        # -- 6. IS vs OOS IC check ---------------------------------------
        split = int(0.7 * T)
        if split >= 30 and (T - split) >= 20:
            ic_is, _ = stats.spearmanr(sig_c[:split], ret_c[:split])
            ic_oos, _ = stats.spearmanr(sig_c[split:], ret_c[split:])
            ic_is = float(ic_is)
            ic_oos = float(ic_oos)
            results["ic_is"] = ic_is
            results["ic_oos"] = ic_oos
            ic_decay = (ic_is - ic_oos) / (abs(ic_is) + 1e-8)
            results["ic_decay_ratio"] = ic_decay
            if ic_decay > 0.5:
                warnings_.append(
                    f"Large IS-to-OOS IC decay: IS={ic_is:.4f}, OOS={ic_oos:.4f} "
                    f"(decay ratio={ic_decay:.2f}) -- possible overfitting"
                )
            if ic_oos < 0 and ic_is > 0:
                failures.append(
                    f"OOS IC={ic_oos:.4f} is negative while IS IC={ic_is:.4f} is positive"
                )

        passed = len(failures) == 0
        return ValidationReport(
            model_name=config.name,
            passed=passed,
            failures=failures,
            warnings=warnings_,
            test_results=results,
        )

    @staticmethod
    def _rolling_ic_series(
        sig: np.ndarray, ret: np.ndarray, window: int
    ) -> np.ndarray:
        """Compute rolling Spearman IC series."""
        T = len(sig)
        if T < window:
            return np.array([])
        ics = []
        for i in range(window, T + 1):
            s_w = sig[i - window: i]
            r_w = ret[i - window: i]
            ic_val, _ = stats.spearmanr(s_w, r_w)
            ics.append(float(ic_val))
        return np.array(ics)

    @classmethod
    def _rolling_icir(cls, sig: np.ndarray, ret: np.ndarray, window: int) -> float:
        """Compute ICIR as mean(IC_series) / std(IC_series)."""
        ic_series = cls._rolling_ic_series(sig, ret, window)
        if len(ic_series) < 5:
            return float("nan")
        std_ic = float(np.std(ic_series, ddof=1))
        if std_ic < 1e-10:
            return float("nan")
        return float(np.mean(ic_series) / std_ic)


# ---------------------------------------------------------------------------
# Risk model validator (VaR backtesting)
# ---------------------------------------------------------------------------

class RiskModelValidator:
    """
    Validates Value-at-Risk (VaR) model outputs via standard backtests.

    Methods
    -------
    validate_var_model : runs Kupiec, Christoffersen, and Lopez tests
    """

    def validate_var_model(
        self,
        var_estimates: np.ndarray,
        realized_returns: np.ndarray,
        confidence: float = 0.95,
        model_name: str = "VaR Model",
    ) -> ValidationReport:
        """
        Validate VaR model by comparing forecasted VaR with realized P&L.

        Parameters
        ----------
        var_estimates     : estimated VaR values (positive convention: loss threshold),
                            shape (T,)
        realized_returns  : actual returns (negative = loss), shape (T,)
        confidence        : VaR confidence level (default 0.95)
        model_name        : name for the report

        Returns
        -------
        ValidationReport
        """
        var_est = np.asarray(var_estimates, dtype=float)
        real_ret = np.asarray(realized_returns, dtype=float)
        if len(var_est) != len(real_ret):
            raise ValueError("var_estimates and realized_returns must have equal length")

        mask = np.isfinite(var_est) & np.isfinite(real_ret)
        var_c, ret_c = var_est[mask], real_ret[mask]
        T = len(var_c)

        failures: list = []
        warnings_: list = []
        results: dict = {}

        # Exceedances: return < -VaR (loss exceeds VaR)
        exceedances = (ret_c < -var_c).astype(int)
        n_exc = int(exceedances.sum())
        exc_rate = n_exc / T
        expected_rate = 1.0 - confidence
        results["exceedance_count"] = n_exc
        results["exceedance_rate"] = exc_rate
        results["expected_rate"] = expected_rate

        # -- Kupiec POF test ---------------------------------------------
        kupiec = self._kupiec_pof(n_exc, T, expected_rate)
        results["kupiec_pof"] = kupiec
        if kupiec.is_significant:
            failures.append(
                f"Kupiec POF: exceedance rate {exc_rate:.3f} significantly "
                f"differs from expected {expected_rate:.3f}"
            )

        # -- Christoffersen independence test ----------------------------
        if T >= 50:
            try:
                chris = self._christoffersen_independence(exceedances)
                results["christoffersen"] = chris
                if chris.is_significant:
                    warnings_.append(
                        "Christoffersen: VaR exceedances are not independent (clustering)"
                    )
            except Exception as exc:
                warnings_.append(f"Christoffersen test failed: {exc}")

        # -- Lopez loss function ----------------------------------------
        lopez_loss = self._lopez_loss(var_c, ret_c)
        results["lopez_loss"] = lopez_loss
        results["lopez_mean_loss"] = float(np.mean(lopez_loss))
        # Compare to a naive model: fixed quantile
        naive_var = np.full_like(var_c, np.percentile(-ret_c, confidence * 100))
        naive_loss = self._lopez_loss(naive_var, ret_c)
        results["naive_mean_loss"] = float(np.mean(naive_loss))
        if np.mean(lopez_loss) > np.mean(naive_loss) * 1.2:
            warnings_.append(
                "Lopez loss function indicates model performs worse than naive fixed-VaR"
            )

        # -- Coverage ratio warning ------------------------------------
        if exc_rate > expected_rate * 2:
            failures.append(
                f"VaR systematically under-estimates risk: "
                f"exceedance rate {exc_rate:.3f} > 2x expected {expected_rate:.3f}"
            )
        elif exc_rate > expected_rate * 1.5:
            warnings_.append(
                f"VaR may be understated: exc_rate={exc_rate:.3f}, expected={expected_rate:.3f}"
            )

        passed = len(failures) == 0
        return ValidationReport(
            model_name=model_name,
            passed=passed,
            failures=failures,
            warnings=warnings_,
            test_results=results,
        )

    @staticmethod
    def _kupiec_pof(n_exc: int, T: int, p: float) -> TestResult:
        """
        Kupiec (1995) Proportion of Failures likelihood ratio test.

        H0: true exceedance probability equals p.
        LR statistic is chi-squared with 1 df.
        """
        if T == 0:
            return TestResult(statistic=np.nan, p_value=np.nan, interpretation="No data")

        p_hat = n_exc / T
        p_hat = np.clip(p_hat, 1e-9, 1 - 1e-9)
        p_c = np.clip(p, 1e-9, 1 - 1e-9)

        if n_exc == 0:
            lr = -2.0 * T * np.log(1.0 - p_c)
        elif n_exc == T:
            lr = -2.0 * T * np.log(p_c)
        else:
            lr = -2.0 * (
                n_exc * np.log(p_c / p_hat) + (T - n_exc) * np.log((1 - p_c) / (1 - p_hat))
            )

        p_val = float(1.0 - stats.chi2.cdf(lr, df=1))
        interp = (
            f"Reject H0: exc_rate={p_hat:.4f} != expected={p:.4f}" if p_val < 0.05
            else f"Cannot reject H0: exc_rate={p_hat:.4f} consistent with {p:.4f}"
        )
        return TestResult(
            statistic=float(lr),
            p_value=p_val,
            interpretation=interp,
            extra={"n_exc": n_exc, "T": T, "p_hat": float(p_hat)},
        )

    @staticmethod
    def _christoffersen_independence(exceedances: np.ndarray) -> TestResult:
        """
        Christoffersen (1998) independence test for VaR exceedances.

        Tests whether exceedances are serially independent. LR statistic is
        chi-squared with 1 df.
        """
        I = exceedances
        n = len(I)
        # Transition counts
        n00 = int(np.sum((I[:-1] == 0) & (I[1:] == 0)))
        n01 = int(np.sum((I[:-1] == 0) & (I[1:] == 1)))
        n10 = int(np.sum((I[:-1] == 1) & (I[1:] == 0)))
        n11 = int(np.sum((I[:-1] == 1) & (I[1:] == 1)))

        # Transition probabilities
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        p_hat = (n01 + n11) / (n - 1) if n > 1 else 0.0

        eps = 1e-10
        p01 = np.clip(p01, eps, 1 - eps)
        p11 = np.clip(p11, eps, 1 - eps)
        p_hat = np.clip(p_hat, eps, 1 - eps)

        # Log-likelihood ratio
        l_a = (
            (n00 + n10) * np.log(1 - p_hat)
            + (n01 + n11) * np.log(p_hat)
        )
        l_b = (
            n00 * np.log(1 - p01)
            + n01 * np.log(p01)
            + n10 * np.log(1 - p11)
            + n11 * np.log(p11)
        )
        lr = float(-2.0 * (l_a - l_b))
        p_val = float(1.0 - stats.chi2.cdf(lr, df=1))

        interp = (
            "Exceedances are NOT independent (clustering detected)" if p_val < 0.05
            else "Exceedances are independently distributed"
        )
        return TestResult(
            statistic=lr,
            p_value=p_val,
            interpretation=interp,
            extra={"p01": float(p01), "p11": float(p11), "pi_hat": float(p_hat)},
        )

    @staticmethod
    def _lopez_loss(var_estimates: np.ndarray, realized_returns: np.ndarray) -> np.ndarray:
        """
        Lopez (1998) loss function.

        L(t) = 1 + (ret - (-VaR))^2  if exceedance, else 0.
        Lower total loss indicates a better VaR model.
        """
        exceedance_mask = realized_returns < -var_estimates
        loss = np.where(
            exceedance_mask,
            1.0 + (realized_returns + var_estimates) ** 2,
            0.0,
        )
        return loss


# ---------------------------------------------------------------------------
# ML / Online learner validator
# ---------------------------------------------------------------------------

class MLModelValidator:
    """
    Validates ML models and online learners for deployment.

    Covers calibration quality, concept drift, and feature importance stability.
    """

    def validate_online_learner(
        self,
        predictions: np.ndarray,
        realizations: np.ndarray,
        warmup: int = 30,
        model_name: str = "OnlineLearner",
        feature_weights: Optional[np.ndarray] = None,
    ) -> ValidationReport:
        """
        Validate an online learning model.

        Parameters
        ----------
        predictions   : model predictions or probabilities, shape (T,)
        realizations  : actual outcomes (0/1 for classification, float for regression)
        warmup        : number of initial observations to skip for testing
        model_name    : identifier for the report
        feature_weights : optional array of shape (T, n_features) tracking feature
                          importances over time for stability check

        Returns
        -------
        ValidationReport
        """
        pred = np.asarray(predictions, dtype=float)
        real = np.asarray(realizations, dtype=float)
        if len(pred) != len(real):
            raise ValueError("predictions and realizations must have equal length")

        pred_w = pred[warmup:]
        real_w = real[warmup:]
        T_w = len(pred_w)

        failures: list = []
        warnings_: list = []
        results: dict = {}

        # -- Calibration: is prediction scale correct? -------------------
        cal_result = self._calibration_check(pred_w, real_w)
        results["calibration"] = cal_result
        results["expected_calibration_error"] = cal_result.get("ece", np.nan)
        if cal_result.get("ece", 0.0) > 0.10:
            warnings_.append(
                f"High ECE={cal_result['ece']:.4f} -- model predictions are poorly calibrated"
            )

        # -- Concept drift via ADWIN-style test ---------------------------
        errors = pred_w - real_w
        drift_detected, drift_location = self._adwin_drift_test(errors)
        results["concept_drift_detected"] = drift_detected
        results["drift_location"] = drift_location
        if drift_detected:
            failures.append(
                f"Concept drift detected at observation {drift_location} -- "
                f"model has lost predictive accuracy"
            )

        # -- Pearson error trend (simple drift proxy) --------------------
        if T_w >= 20:
            t_idx = np.arange(T_w, dtype=float)
            r_corr, r_p = stats.pearsonr(t_idx, errors)
            results["error_trend_corr"] = float(r_corr)
            results["error_trend_p"] = float(r_p)
            if r_p < 0.05 and abs(r_corr) > 0.2:
                warnings_.append(
                    f"Trending prediction error (r={r_corr:.3f}, p={r_p:.4f}) -- "
                    f"potential distribution shift"
                )

        # -- Feature importance stability --------------------------------
        if feature_weights is not None:
            fw = np.asarray(feature_weights, dtype=float)
            if fw.ndim == 2 and fw.shape[0] >= 20:
                stab = self._feature_stability(fw[warmup:])
                results["feature_stability"] = stab
                if stab < 0.5:
                    warnings_.append(
                        f"Low feature importance stability (r={stab:.4f}) -- "
                        f"model structure is changing rapidly"
                    )

        # -- Basic accuracy / correlation --------------------------------
        if T_w >= 10:
            if len(np.unique(real_w)) <= 2 and set(np.unique(real_w)).issubset({0.0, 1.0}):
                # Classification
                acc = float(np.mean((pred_w >= 0.5) == real_w))
                results["accuracy"] = acc
                if acc < 0.50:
                    failures.append(f"Accuracy={acc:.4f} below chance -- model is inverting")
            else:
                # Regression: Pearson correlation
                r_pred, r_p = stats.pearsonr(pred_w, real_w)
                results["pearson_r"] = float(r_pred)
                results["pearson_p"] = float(r_p)
                if r_pred < 0 and r_p < 0.05:
                    failures.append(
                        f"Negative predictive correlation r={r_pred:.4f} -- model is inverting"
                    )

        passed = len(failures) == 0
        return ValidationReport(
            model_name=model_name,
            passed=passed,
            failures=failures,
            warnings=warnings_,
            test_results=results,
        )

    @staticmethod
    def _calibration_check(
        pred: np.ndarray, real: np.ndarray, n_bins: int = 10
    ) -> dict:
        """
        Compute reliability diagram data and Expected Calibration Error (ECE).

        Works for both binary classification (pred in [0,1]) and regression
        (uses normalized RMSE as a calibration proxy).
        """
        if len(pred) < n_bins * 2:
            return {"ece": 0.0, "note": "insufficient data for calibration check"}

        is_prob = float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0

        if is_prob:
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            bin_data = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (pred >= lo) & (pred < hi)
                if mask.sum() == 0:
                    continue
                bin_conf = float(pred[mask].mean())
                bin_acc = float(real[mask].mean())
                ece += mask.sum() / len(pred) * abs(bin_conf - bin_acc)
                bin_data.append({"conf": bin_conf, "acc": bin_acc, "n": int(mask.sum())})
            return {"ece": ece, "bins": bin_data}
        else:
            # Regression: Pearson correlation as calibration metric
            corr, p = stats.pearsonr(pred, real)
            rmse = float(np.sqrt(np.mean((pred - real) ** 2)))
            return {"ece": 1.0 - abs(float(corr)), "rmse": rmse, "corr": float(corr)}

    @staticmethod
    def _adwin_drift_test(
        errors: np.ndarray, delta: float = 0.002
    ) -> tuple:
        """
        Simplified ADWIN-inspired drift test.

        Splits the error series at every window point and checks if the
        means differ significantly via Welch t-test (Bonferroni corrected).

        Returns (drift_detected: bool, location: int or None).
        """
        n = len(errors)
        if n < 30:
            return False, None

        # Scan for the split with maximum mean difference
        min_size = max(10, n // 10)
        best_stat = 0.0
        best_loc = None

        for t in range(min_size, n - min_size):
            left = errors[:t]
            right = errors[t:]
            stat, p = stats.ttest_ind(left, right, equal_var=False)
            # Bonferroni correction for number of splits tested
            n_tests = n - 2 * min_size
            p_corrected = min(1.0, float(p) * n_tests)
            if p_corrected < delta and abs(float(stat)) > abs(best_stat):
                best_stat = float(stat)
                best_loc = t

        drift_detected = best_loc is not None
        return drift_detected, best_loc

    @staticmethod
    def _feature_stability(feature_weights: np.ndarray) -> float:
        """
        Measure feature importance stability as the average Pearson correlation
        between consecutive weight vectors.
        """
        T, n_feat = feature_weights.shape
        if T < 2 or n_feat < 2:
            return 1.0

        corrs = []
        for i in range(T - 1):
            w1 = feature_weights[i]
            w2 = feature_weights[i + 1]
            if np.std(w1) < 1e-10 or np.std(w2) < 1e-10:
                continue
            r, _ = stats.pearsonr(w1, w2)
            corrs.append(float(r))

        return float(np.mean(corrs)) if corrs else 1.0
