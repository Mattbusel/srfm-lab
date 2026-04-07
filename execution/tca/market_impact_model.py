# execution/tca/market_impact_model.py -- Market impact estimation models for SRFM TCA
# Implements linear, square-root, and nonlinear impact models plus an ensemble.
# All numerical work uses numpy only -- no scipy dependency.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """Fitted parameters for a market impact model."""
    eta: float            # scaling coefficient
    alpha: float = 0.5    # exponent (for nonlinear model)
    sigma_weight: bool = True  # whether sigma was included in fitting
    n_obs: int = 0
    rmse: float = 0.0
    r_squared: float = 0.0
    model_type: str = "linear"


@dataclass
class CalibrationResult:
    """Result of cross-validated model calibration."""
    params: ModelParams
    cv_rmse: float            # mean cross-validated RMSE
    cv_rmse_std: float        # std of CV RMSE across folds
    n_folds: int
    n_obs: int
    in_sample_rmse: float
    in_sample_r2: float
    fold_rmses: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OLS helper (numpy only)
# ---------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Solve ordinary least squares: beta = (X'X)^{-1} X'y.
    Returns (beta, rmse, r_squared).
    X should include a constant column if desired.
    """
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        beta = np.zeros(X.shape[1])

    y_hat = X @ beta
    res = y - y_hat
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    rmse = math.sqrt(ss_res / max(len(y), 1))
    return beta, rmse, r2


def _build_feature_matrix(
    participation_rates: np.ndarray,
    sigmas: np.ndarray,
    advs: np.ndarray,
    qtys: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    Build the regressor matrix for the chosen impact model type.

    mode="linear"   : X = [sigma * sqrt(participation), 1]
    mode="sqrt"     : X = [sigma * sqrt(qty / adv), 1]
    """
    n = len(participation_rates)
    if mode == "linear":
        feature = sigmas * np.sqrt(np.clip(participation_rates, 1e-9, 1.0))
    elif mode == "sqrt":
        ratio = np.clip(qtys / np.where(advs > 0, advs, 1.0), 1e-9, 1.0)
        feature = sigmas * np.sqrt(ratio)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.column_stack([feature, np.ones(n)])


# ---------------------------------------------------------------------------
# Linear impact model  -- impact = eta * sigma * sqrt(participation_rate)
# ---------------------------------------------------------------------------

class LinearImpactModel:
    """
    Linear market impact model: impact_bps = eta * sigma * sqrt(prate).
    Parameters fit via OLS on historical TCA results.
    """

    def __init__(self) -> None:
        self.params: Optional[ModelParams] = None

    def fit(
        self,
        participation_rates: List[float],
        sigmas: List[float],
        observed_impacts_bps: List[float],
    ) -> ModelParams:
        """
        Fit model via OLS.

        Parameters
        ----------
        participation_rates   : fraction of market volume (0..1)
        sigmas                : daily return volatility for each trade
        observed_impacts_bps  : realized market impact in basis points
        """
        pr = np.array(participation_rates, dtype=float)
        sig = np.array(sigmas, dtype=float)
        y = np.array(observed_impacts_bps, dtype=float)
        n = len(y)
        if n < 2:
            self.params = ModelParams(eta=1.0, model_type="linear", n_obs=n)
            return self.params

        X = _build_feature_matrix(pr, sig, np.ones(n), np.ones(n), "linear")
        beta, rmse, r2 = _ols(X, y)
        eta = float(beta[0]) if beta[0] > 0.0 else 0.1
        self.params = ModelParams(
            eta=eta, alpha=0.5, n_obs=n, rmse=rmse, r_squared=r2, model_type="linear"
        )
        return self.params

    def predict(
        self, participation_rate: float, sigma: float
    ) -> float:
        """Return predicted impact in basis points."""
        eta = self.params.eta if self.params else 1.0
        return eta * sigma * math.sqrt(max(participation_rate, 0.0)) * 10_000.0

    def predict_pre_trade(
        self, qty: float, adv: float, sigma: float
    ) -> float:
        """Pre-trade impact estimate given order size and market conditions."""
        if adv <= 0.0:
            return 0.0
        prate = min(qty / adv, 1.0)
        return self.predict(prate, sigma)


# ---------------------------------------------------------------------------
# Square-root impact model  -- Almgren-Chriss style
# ---------------------------------------------------------------------------

class SqrtImpactModel:
    """
    Square-root market impact model: impact_bps = eta * sigma * sqrt(Q / ADV).
    Almgren-Chriss (2001) inspired.
    """

    def __init__(self) -> None:
        self.params: Optional[ModelParams] = None

    def fit(
        self,
        qtys: List[float],
        advs: List[float],
        sigmas: List[float],
        observed_impacts_bps: List[float],
    ) -> ModelParams:
        """Fit eta via OLS."""
        q = np.array(qtys, dtype=float)
        adv = np.array(advs, dtype=float)
        sig = np.array(sigmas, dtype=float)
        y = np.array(observed_impacts_bps, dtype=float)
        n = len(y)
        if n < 2:
            self.params = ModelParams(eta=1.0, model_type="sqrt", n_obs=n)
            return self.params

        pr = q / np.where(adv > 0, adv, 1.0)
        X = _build_feature_matrix(pr, sig, adv, q, "sqrt")
        beta, rmse, r2 = _ols(X, y)
        eta = float(beta[0]) if beta[0] > 0.0 else 0.5
        self.params = ModelParams(
            eta=eta, alpha=0.5, n_obs=n, rmse=rmse, r_squared=r2, model_type="sqrt"
        )
        return self.params

    def predict(self, qty: float, adv: float, sigma: float) -> float:
        """Return predicted impact in basis points."""
        eta = self.params.eta if self.params else 0.5
        if adv <= 0.0:
            return 0.0
        ratio = max(qty / adv, 0.0)
        return eta * sigma * math.sqrt(ratio) * 10_000.0

    def predict_pre_trade(
        self, symbol: str, qty: float, adv: float, sigma: float
    ) -> float:
        """Pre-trade estimate (symbol ignored, for interface consistency)."""
        return self.predict(qty, adv, sigma)


# ---------------------------------------------------------------------------
# Nonlinear impact model  -- impact = eta * sigma * (Q/ADV)^alpha
# ---------------------------------------------------------------------------

class NonlinearImpactModel:
    """
    Nonlinear market impact: impact_bps = eta * sigma * (Q/ADV)^alpha.
    Fit via log-linearization: log(impact) = log(eta) + log(sigma) + alpha * log(Q/ADV).
    """

    def __init__(self) -> None:
        self.params: Optional[ModelParams] = None

    def fit(
        self,
        qtys: List[float],
        advs: List[float],
        sigmas: List[float],
        observed_impacts_bps: List[float],
    ) -> ModelParams:
        """
        Fit eta and alpha via OLS on log-linearized model.
        Observations with non-positive impact are excluded.
        """
        q = np.array(qtys, dtype=float)
        adv = np.array(advs, dtype=float)
        sig = np.array(sigmas, dtype=float)
        y_raw = np.array(observed_impacts_bps, dtype=float)

        # Filter valid observations
        mask = (y_raw > 0) & (q > 0) & (adv > 0) & (sig > 0)
        if np.sum(mask) < 2:
            self.params = ModelParams(eta=1.0, alpha=0.5, model_type="nonlinear", n_obs=0)
            return self.params

        y = np.log(y_raw[mask])
        log_ratio = np.log(q[mask] / adv[mask])
        log_sig = np.log(sig[mask])
        n = len(y)

        # Model: log(impact) = log(eta) + log(sigma) + alpha * log(Q/ADV)
        # Treat log_sig as offset -- X = [log_ratio, 1]
        X = np.column_stack([log_ratio, np.ones(n)])
        y_adj = y - log_sig   # subtract log_sigma contribution
        beta, rmse, r2 = _ols(X, y_adj)

        alpha_fit = float(beta[0])
        log_eta = float(beta[1])
        eta_fit = math.exp(log_eta) if math.isfinite(log_eta) else 1.0
        alpha_fit = max(0.1, min(alpha_fit, 2.0))  # clamp to reasonable range

        self.params = ModelParams(
            eta=eta_fit,
            alpha=alpha_fit,
            n_obs=n,
            rmse=rmse,
            r_squared=r2,
            model_type="nonlinear",
        )
        return self.params

    def predict(self, qty: float, adv: float, sigma: float) -> float:
        """Return predicted impact in basis points."""
        if self.params is None:
            return 0.0
        if adv <= 0.0 or qty <= 0.0 or sigma <= 0.0:
            return 0.0
        ratio = qty / adv
        impact = self.params.eta * sigma * (ratio ** self.params.alpha) * 10_000.0
        return max(impact, 0.0)

    def predict_pre_trade(
        self, symbol: str, qty: float, adv: float, sigma: float
    ) -> float:
        """Pre-trade estimate."""
        return self.predict(qty, adv, sigma)


# ---------------------------------------------------------------------------
# Ensemble model
# ---------------------------------------------------------------------------

class ImpactModelEnsemble:
    """
    Weighted ensemble of LinearImpactModel, SqrtImpactModel, NonlinearImpactModel.
    Weights are inversely proportional to each model's recent RMSE on a holdout set.
    """

    def __init__(self) -> None:
        self.linear = LinearImpactModel()
        self.sqrt = SqrtImpactModel()
        self.nonlinear = NonlinearImpactModel()
        self._weights: List[float] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        self._rmses: List[float] = [float("inf"), float("inf"), float("inf")]
        self._fitted = False

    def fit(
        self,
        qtys: List[float],
        advs: List[float],
        sigmas: List[float],
        participation_rates: List[float],
        observed_impacts_bps: List[float],
    ) -> None:
        """
        Fit all three sub-models and compute ensemble weights from in-sample RMSE.
        In production, use ImpactCalibrator.cross_validate for out-of-sample weights.
        """
        n = len(observed_impacts_bps)
        if n < 3:
            self._fitted = False
            return

        lp = self.linear.fit(participation_rates, sigmas, observed_impacts_bps)
        sp = self.sqrt.fit(qtys, advs, sigmas, observed_impacts_bps)
        nlp = self.nonlinear.fit(qtys, advs, sigmas, observed_impacts_bps)

        self._rmses = [
            lp.rmse if math.isfinite(lp.rmse) else 1e6,
            sp.rmse if math.isfinite(sp.rmse) else 1e6,
            nlp.rmse if math.isfinite(nlp.rmse) else 1e6,
        ]
        self._weights = self._rmse_to_weights(self._rmses)
        self._fitted = True

    @staticmethod
    def _rmse_to_weights(rmses: List[float]) -> List[float]:
        """Convert RMSEs to inverse-RMSE weights, normalized to sum to 1."""
        inv = [1.0 / max(r, 1e-9) for r in rmses]
        total = sum(inv)
        if total <= 0.0:
            return [1.0 / len(rmses)] * len(rmses)
        return [v / total for v in inv]

    def predict(
        self,
        qty: float,
        adv: float,
        sigma: float,
        participation_rate: Optional[float] = None,
    ) -> float:
        """Return ensemble-weighted impact prediction in basis points."""
        if participation_rate is None:
            participation_rate = qty / adv if adv > 0.0 else 0.0
        preds = [
            self.linear.predict(participation_rate, sigma),
            self.sqrt.predict(qty, adv, sigma),
            self.nonlinear.predict(qty, adv, sigma),
        ]
        return sum(w * p for w, p in zip(self._weights, preds))

    def predict_pre_trade(
        self, symbol: str, qty: float, adv: float, sigma: float
    ) -> float:
        """Pre-trade cost estimate using the ensemble."""
        return self.predict(qty, adv, sigma)

    @property
    def weights(self) -> List[float]:
        return list(self._weights)

    @property
    def model_rmses(self) -> List[float]:
        return list(self._rmses)


# ---------------------------------------------------------------------------
# Impact calibrator with cross-validation
# ---------------------------------------------------------------------------

class ImpactCalibrator:
    """
    Fits market impact model parameters from a list of TCAResult objects
    and provides cross-validated performance metrics.
    """

    def __init__(self, model_type: str = "ensemble") -> None:
        """
        Parameters
        ----------
        model_type : "linear", "sqrt", "nonlinear", or "ensemble"
        """
        self.model_type = model_type
        self.ensemble = ImpactModelEnsemble()
        self._last_params: Optional[ModelParams] = None

    def calibrate(self, results) -> ModelParams:
        """
        Fit model parameters from a list of TCAResult objects.

        Parameters
        ----------
        results : List[TCAResult] -- must have market_impact_bps populated

        Returns
        -------
        ModelParams for the best-fit model
        """
        qtys, advs, sigmas, prates, impacts = self._extract_arrays(results)
        n = len(impacts)
        if n < 2:
            params = ModelParams(eta=1.0, n_obs=n, model_type=self.model_type)
            self._last_params = params
            return params

        if self.model_type == "ensemble":
            self.ensemble.fit(qtys, advs, sigmas, prates, impacts)
            best_rmse = min(self.ensemble.model_rmses)
            best_idx = self.ensemble.model_rmses.index(best_rmse)
            models = [self.ensemble.linear, self.ensemble.sqrt, self.ensemble.nonlinear]
            params = models[best_idx].params or ModelParams(eta=1.0, model_type="unknown")
        elif self.model_type == "linear":
            model = LinearImpactModel()
            params = model.fit(prates, sigmas, impacts)
        elif self.model_type == "sqrt":
            model = SqrtImpactModel()
            params = model.fit(qtys, advs, sigmas, impacts)
        elif self.model_type == "nonlinear":
            model = NonlinearImpactModel()
            params = model.fit(qtys, advs, sigmas, impacts)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self._last_params = params
        return params

    def cross_validate(
        self,
        results,
        n_folds: int = 5,
    ) -> CalibrationResult:
        """
        K-fold cross-validation of the impact model.

        Parameters
        ----------
        results  : List[TCAResult]
        n_folds  : number of CV folds (default 5)

        Returns
        -------
        CalibrationResult with per-fold RMSE statistics
        """
        qtys, advs, sigmas, prates, impacts = self._extract_arrays(results)
        n = len(impacts)
        if n < n_folds * 2:
            params = self.calibrate(results)
            return CalibrationResult(
                params=params,
                cv_rmse=params.rmse,
                cv_rmse_std=0.0,
                n_folds=1,
                n_obs=n,
                in_sample_rmse=params.rmse,
                in_sample_r2=params.r_squared,
                fold_rmses=[params.rmse],
            )

        indices = np.arange(n)
        rng = np.random.default_rng(42)
        rng.shuffle(indices)
        folds = np.array_split(indices, n_folds)
        fold_rmses: List[float] = []

        def _sub(arr: List[float], idx: np.ndarray) -> List[float]:
            return [arr[i] for i in idx]

        for fold_idx in range(n_folds):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate(
                [folds[i] for i in range(n_folds) if i != fold_idx]
            )

            test_impacts = _sub(impacts, test_idx)

            if self.model_type == "ensemble":
                fold_model = ImpactModelEnsemble()
                fold_model.fit(
                    _sub(qtys, train_idx),
                    _sub(advs, train_idx),
                    _sub(sigmas, train_idx),
                    _sub(prates, train_idx),
                    _sub(impacts, train_idx),
                )
                preds = [
                    fold_model.predict(qtys[i], advs[i], sigmas[i], prates[i])
                    for i in test_idx
                ]
            elif self.model_type == "sqrt":
                fold_model_s = SqrtImpactModel()
                fold_model_s.fit(
                    _sub(qtys, train_idx),
                    _sub(advs, train_idx),
                    _sub(sigmas, train_idx),
                    _sub(impacts, train_idx),
                )
                preds = [
                    fold_model_s.predict(qtys[i], advs[i], sigmas[i])
                    for i in test_idx
                ]
            else:
                fold_model_l = LinearImpactModel()
                fold_model_l.fit(
                    _sub(prates, train_idx),
                    _sub(sigmas, train_idx),
                    _sub(impacts, train_idx),
                )
                preds = [
                    fold_model_l.predict(prates[i], sigmas[i])
                    for i in test_idx
                ]

            fold_rmse = math.sqrt(
                sum((p - a) ** 2 for p, a in zip(preds, test_impacts))
                / max(len(test_impacts), 1)
            )
            fold_rmses.append(fold_rmse)

        cv_rmse = sum(fold_rmses) / len(fold_rmses)
        cv_std = (
            math.sqrt(
                sum((r - cv_rmse) ** 2 for r in fold_rmses) / len(fold_rmses)
            )
            if len(fold_rmses) > 1
            else 0.0
        )

        params = self.calibrate(results)
        return CalibrationResult(
            params=params,
            cv_rmse=cv_rmse,
            cv_rmse_std=cv_std,
            n_folds=n_folds,
            n_obs=n,
            in_sample_rmse=params.rmse,
            in_sample_r2=params.r_squared,
            fold_rmses=fold_rmses,
        )

    def predict_pre_trade(
        self, symbol: str, qty: float, adv: float, sigma: float
    ) -> float:
        """
        Predict pre-trade cost estimate for an order.

        Parameters
        ----------
        symbol : ticker (used for logging, not currently segmented by symbol)
        qty    : order size in shares
        adv    : average daily volume in shares
        sigma  : daily return volatility (fractional, e.g. 0.02 for 2%)

        Returns
        -------
        Predicted market impact in basis points
        """
        if self.model_type == "ensemble":
            return self.ensemble.predict_pre_trade(symbol, qty, adv, sigma)
        if self._last_params is None:
            return 0.0
        prate = qty / adv if adv > 0.0 else 0.0
        if self.model_type == "linear":
            return (self._last_params.eta * sigma
                    * math.sqrt(max(prate, 0.0)) * 10_000.0)
        if self.model_type in ("sqrt", "nonlinear"):
            ratio = max(qty / adv, 0.0) if adv > 0.0 else 0.0
            return (self._last_params.eta * sigma
                    * (ratio ** self._last_params.alpha) * 10_000.0)
        return 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_arrays(results) -> Tuple[
        List[float], List[float], List[float], List[float], List[float]
    ]:
        """
        Extract (qtys, advs, sigmas, prates, impacts) lists from TCAResult objects.
        Uses embedded fields where available, falls back to sensible defaults.
        """
        qtys: List[float] = []
        advs: List[float] = []
        sigmas: List[float] = []
        prates: List[float] = []
        impacts: List[float] = []

        for r in results:
            impact = r.market_impact_bps
            if not math.isfinite(impact) or impact <= 0:
                continue
            prate = getattr(r, "participation_rate", 0.0) or 0.0
            qty = 100.0        # placeholder -- real system would carry order qty
            adv = 1_000_000.0  # placeholder -- real system would carry ADV
            sigma = 0.02       # placeholder -- real system would carry realized vol

            qtys.append(qty)
            advs.append(adv)
            sigmas.append(sigma)
            prates.append(prate)
            impacts.append(impact)

        return qtys, advs, sigmas, prates, impacts
