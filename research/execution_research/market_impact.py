"""
market_impact.py — Market Impact Analysis and Model Calibration
===============================================================

Provides calibration of market impact models from live trade data:
  - Linear model:       impact = alpha + beta × Q/V
  - Square-root model:  impact = alpha + beta × sqrt(Q/V)
  - Almgren-Chriss:     calibrates (eta, gamma) from empirical fills

Also computes:
  - Kyle's lambda  (price impact per unit order flow)
  - Amihud illiquidity ratio series
"""

from __future__ import annotations

import math
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit, minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImpactModel:
    """A calibrated market impact model."""
    model_type: str              # "linear" | "sqrt" | "almgren_chriss"
    alpha: float = 0.0
    beta: float = 0.0
    eta: float = 0.0             # A-C temporary impact
    gamma: float = 0.0           # A-C permanent impact
    r_squared: float = 0.0
    n_observations: int = 0
    calibration_date: str = ""
    residual_std: float = 0.0
    notes: str = ""

    def predict(self, participation_rate: float) -> float:
        """
        Predict impact in bps for a given participation rate Q/V.

        Parameters
        ----------
        participation_rate : float
            Q / V  (trade size / ADV), e.g. 0.05 = 5% of ADV.

        Returns
        -------
        float
            Predicted impact in bps.
        """
        if self.model_type == "linear":
            return self.alpha + self.beta * participation_rate
        elif self.model_type == "sqrt":
            return self.alpha + self.beta * math.sqrt(max(0.0, participation_rate))
        elif self.model_type == "almgren_chriss":
            # Simplified: temporary only
            return self.eta * math.sqrt(max(0.0, participation_rate)) * 10_000
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


@dataclass
class CalibrationResult:
    """Result of a model calibration run."""
    model: ImpactModel
    actual_bps: np.ndarray
    predicted_bps: np.ndarray
    participation_rates: np.ndarray
    residuals: np.ndarray
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class MarketImpactCalibrator:
    """
    Calibrates market impact models from historical trade/fill data.

    The input `trades` DataFrame should have columns:
      - participation_rate: float  (Q / ADV_in_shares, or notional / ADV_USD)
      - actual_impact_bps: float   (observed IS or arrival-price slippage)
      - adv: float                 (average daily volume in USD)
      - notional: float            (trade size in USD)
      - daily_vol: float           (daily return vol)
      - order_flow: float          (optional, net buy/sell signed volume)
      - price_change: float        (optional, price Δ per unit time)

    The presence of adv and notional allows computing participation_rate
    internally if not already in the DataFrame.
    """

    MIN_OBSERVATIONS = 10

    def __init__(self) -> None:
        self._calibrated_models: dict[str, ImpactModel] = {}

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _prepare_data(
        trades: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract (participation_rates, actual_impact_bps) from trades DataFrame.

        Handles missing columns gracefully.
        """
        df = trades.copy()

        # Compute participation_rate if not present
        if "participation_rate" not in df.columns:
            if "notional" in df.columns and "adv" in df.columns:
                df["participation_rate"] = df["notional"] / df["adv"].replace(0, np.nan)
            else:
                raise ValueError(
                    "trades must have 'participation_rate' column or "
                    "both 'notional' and 'adv' columns"
                )

        if "actual_impact_bps" not in df.columns:
            raise ValueError("trades must have 'actual_impact_bps' column")

        # Drop NaN and non-positive participation rates
        df = df.dropna(subset=["participation_rate", "actual_impact_bps"])
        df = df[df["participation_rate"] > 0]

        if len(df) < MarketImpactCalibrator.MIN_OBSERVATIONS:
            raise ValueError(
                f"Need at least {MarketImpactCalibrator.MIN_OBSERVATIONS} observations, "
                f"got {len(df)}"
            )

        return df["participation_rate"].values, df["actual_impact_bps"].values

    @staticmethod
    def _compute_r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # -----------------------------------------------------------------------
    # Linear calibration: impact = alpha + beta × Q/V
    # -----------------------------------------------------------------------

    def calibrate_linear(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame | None = None,
    ) -> tuple[float, float]:
        """
        Calibrate linear impact model: impact_bps = alpha + beta × (Q/V).

        Parameters
        ----------
        trades : pd.DataFrame
        market_data : optional

        Returns
        -------
        (alpha, beta) : tuple[float, float]
            Intercept and slope in bps.
        """
        participation_rates, actual_bps = self._prepare_data(trades)

        X = np.column_stack([np.ones_like(participation_rates), participation_rates])
        slope, intercept, r_value, p_value, se = stats.linregress(participation_rates, actual_bps)

        alpha = float(intercept)
        beta = float(slope)
        predicted = alpha + beta * participation_rates
        r2 = self._compute_r_squared(actual_bps, predicted)

        model = ImpactModel(
            model_type="linear",
            alpha=alpha,
            beta=beta,
            r_squared=r2,
            n_observations=len(participation_rates),
            residual_std=float(np.std(actual_bps - predicted)),
            notes=f"p_value={p_value:.4f}",
        )
        self._calibrated_models["linear"] = model

        logger.info(
            "Linear impact model: alpha=%.2f bps, beta=%.2f bps/(Q/V), R²=%.3f",
            alpha, beta, r2,
        )
        return alpha, beta

    # -----------------------------------------------------------------------
    # Square-root calibration: impact = alpha + beta × sqrt(Q/V)
    # -----------------------------------------------------------------------

    def calibrate_sqrt(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame | None = None,
    ) -> tuple[float, float]:
        """
        Calibrate square-root impact model: impact_bps = alpha + beta × sqrt(Q/V).

        The square-root model is empirically the most robust for equity and
        crypto markets.

        Parameters
        ----------
        trades : pd.DataFrame
        market_data : optional

        Returns
        -------
        (alpha, beta) : tuple[float, float]
        """
        participation_rates, actual_bps = self._prepare_data(trades)

        sqrt_rates = np.sqrt(participation_rates)

        def sqrt_model(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
            return alpha + beta * np.sqrt(x)

        try:
            popt, pcov = curve_fit(
                sqrt_model,
                participation_rates,
                actual_bps,
                p0=[0.0, 10.0],
                maxfev=5000,
            )
            alpha, beta = float(popt[0]), float(popt[1])
        except RuntimeError:
            # Fall back to linear regression on sqrt(x)
            slope, intercept, _, _, _ = stats.linregress(sqrt_rates, actual_bps)
            alpha, beta = float(intercept), float(slope)
            warnings.warn("curve_fit failed for sqrt model; fell back to linregress")

        predicted = alpha + beta * sqrt_rates
        r2 = self._compute_r_squared(actual_bps, predicted)

        model = ImpactModel(
            model_type="sqrt",
            alpha=alpha,
            beta=beta,
            r_squared=r2,
            n_observations=len(participation_rates),
            residual_std=float(np.std(actual_bps - predicted)),
        )
        self._calibrated_models["sqrt"] = model

        logger.info(
            "Sqrt impact model: alpha=%.2f bps, beta=%.2f bps/sqrt(Q/V), R²=%.3f",
            alpha, beta, r2,
        )
        return alpha, beta

    # -----------------------------------------------------------------------
    # Almgren-Chriss calibration
    # -----------------------------------------------------------------------

    def calibrate_almgren_chriss(
        self,
        trades: pd.DataFrame,
        market_data: pd.DataFrame | None = None,
    ) -> tuple[float, float]:
        """
        Calibrate Almgren-Chriss parameters (eta, gamma).

        The A-C model:
          temporary_impact = eta × (trade_rate)     [in price units]
          permanent_impact = gamma × (trade_rate)   [in price units]

        We estimate (eta, gamma) by matching observed IS to model predictions.

        Parameters
        ----------
        trades : pd.DataFrame
            Must have: participation_rate, actual_impact_bps, daily_vol (optional)
        market_data : pd.DataFrame, optional

        Returns
        -------
        (eta, gamma) : tuple[float, float]
        """
        participation_rates, actual_bps = self._prepare_data(trades)

        if "daily_vol" in trades.columns:
            sigma_vals = trades.dropna(subset=["participation_rate", "actual_impact_bps"])["daily_vol"].values
        else:
            sigma_vals = np.full_like(participation_rates, 0.02)  # default 2%

        def ac_model(
            x: tuple[np.ndarray, np.ndarray],
            eta: float,
            gamma: float,
        ) -> np.ndarray:
            prate, sigma = x
            temp = eta * np.sqrt(np.maximum(prate, 0)) * sigma
            perm = gamma * prate * sigma
            return (temp + perm) * 10_000  # convert to bps

        try:
            popt, _ = curve_fit(
                ac_model,
                (participation_rates, sigma_vals),
                actual_bps,
                p0=[0.1, 0.1],
                bounds=(0, np.inf),
                maxfev=10000,
            )
            eta, gamma = float(popt[0]), float(popt[1])
        except RuntimeError:
            # Simple moment matching fallback
            mean_pr = float(np.mean(participation_rates))
            mean_sig = float(np.mean(sigma_vals))
            mean_impact = float(np.mean(actual_bps)) / 10_000  # to fraction
            eta = mean_impact / (math.sqrt(mean_pr) * mean_sig + 1e-12)
            gamma = eta * 0.5
            warnings.warn("A-C curve_fit failed; using moment-matching fallback")

        predicted = (
            eta * np.sqrt(np.maximum(participation_rates, 0)) * sigma_vals
            + gamma * participation_rates * sigma_vals
        ) * 10_000

        r2 = self._compute_r_squared(actual_bps, predicted)

        model = ImpactModel(
            model_type="almgren_chriss",
            eta=eta,
            gamma=gamma,
            r_squared=r2,
            n_observations=len(participation_rates),
            residual_std=float(np.std(actual_bps - predicted)),
        )
        self._calibrated_models["almgren_chriss"] = model

        logger.info(
            "A-C model: eta=%.4f, gamma=%.4f, R²=%.3f",
            eta, gamma, r2,
        )
        return eta, gamma

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict_impact(
        self,
        model: ImpactModel,
        trade: dict[str, float],
    ) -> float:
        """
        Predict market impact for a single trade.

        Parameters
        ----------
        model : ImpactModel
        trade : dict
            Must contain 'participation_rate' or ('notional', 'adv').
            May contain 'daily_vol'.

        Returns
        -------
        float
            Predicted impact in bps.
        """
        if "participation_rate" in trade:
            pr = float(trade["participation_rate"])
        elif "notional" in trade and "adv" in trade:
            adv = float(trade["adv"])
            pr = float(trade["notional"]) / adv if adv > 0 else 0.0
        else:
            raise ValueError("trade must have 'participation_rate' or ('notional', 'adv')")

        if model.model_type == "almgren_chriss":
            sigma = float(trade.get("daily_vol", 0.02))
            return (
                model.eta * math.sqrt(max(0.0, pr)) * sigma
                + model.gamma * pr * sigma
            ) * 10_000
        else:
            return model.predict(pr)

    # -----------------------------------------------------------------------
    # Static impact metrics
    # -----------------------------------------------------------------------

    @staticmethod
    def normalized_trade_size(dollar_size: float, adv: float) -> float:
        """
        Compute participation rate = dollar_size / ADV.

        Parameters
        ----------
        dollar_size : float
        adv : float

        Returns
        -------
        float
            Participation rate (0–1+).
        """
        if adv <= 0:
            raise ValueError(f"adv must be positive, got {adv}")
        return dollar_size / adv

    @staticmethod
    def kyle_lambda(
        price_changes: np.ndarray | pd.Series,
        order_flow: np.ndarray | pd.Series,
    ) -> float:
        """
        Estimate Kyle's lambda: price impact per unit of net order flow.

        Lambda = Cov(ΔP, x) / Var(x)
        where ΔP = price change, x = signed order flow (buy - sell volume).

        Parameters
        ----------
        price_changes : array-like
            Series of price changes (ΔP).
        order_flow : array-like
            Series of signed order flow (net buy volume, can be negative).

        Returns
        -------
        float
            Kyle's lambda (price change per unit order flow).
        """
        dp = np.asarray(price_changes, dtype=float)
        x = np.asarray(order_flow, dtype=float)

        if len(dp) != len(x):
            raise ValueError("price_changes and order_flow must have the same length")
        if len(dp) < 5:
            raise ValueError("Need at least 5 observations for Kyle's lambda")

        var_x = float(np.var(x))
        if var_x == 0:
            logger.warning("order_flow has zero variance — cannot compute Kyle's lambda")
            return 0.0

        cov = float(np.cov(dp, x)[0, 1])
        lam = cov / var_x

        logger.info("Kyle's lambda = %.6f", lam)
        return lam

    @staticmethod
    def amihud_illiquidity(
        daily_returns: np.ndarray | pd.Series,
        daily_volume: np.ndarray | pd.Series,
    ) -> pd.Series:
        """
        Compute the Amihud (2002) illiquidity ratio.

        ILLIQ_t = |R_t| / (Volume_t × Price_t)

        Approximated as: ILLIQ_t = |R_t| / DollarVolume_t

        Parameters
        ----------
        daily_returns : array-like
            Daily return series (fractional, e.g. 0.02 = 2%).
        daily_volume : array-like
            Daily dollar volume series.

        Returns
        -------
        pd.Series
            Amihud ratio series (higher = more illiquid).
        """
        ret = np.asarray(daily_returns, dtype=float)
        vol = np.asarray(daily_volume, dtype=float)

        if len(ret) != len(vol):
            raise ValueError("daily_returns and daily_volume must have the same length")

        with np.errstate(divide="ignore", invalid="ignore"):
            illiq = np.where(vol > 0, np.abs(ret) / vol, np.nan)

        index = getattr(daily_returns, "index", None)
        return pd.Series(illiq, index=index, name="amihud_illiquidity")

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot_impact_calibration(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        save_path: str | Path,
        title: str = "Market Impact Calibration",
    ) -> None:
        """
        Scatter plot of actual vs predicted impact with identity line.

        Parameters
        ----------
        actual : np.ndarray
            Actual observed impact in bps.
        predicted : np.ndarray
            Model-predicted impact in bps.
        save_path : str | Path
        title : str
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed — cannot plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: actual vs predicted
        ax = axes[0]
        ax.scatter(predicted, actual, alpha=0.4, s=20, color="steelblue")
        mn = min(predicted.min(), actual.min())
        mx = max(predicted.max(), actual.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="y=x")
        ax.set_xlabel("Predicted Impact (bps)")
        ax.set_ylabel("Actual Impact (bps)")
        ax.set_title(f"{title}\nActual vs Predicted")
        ax.legend()

        r2 = self._compute_r_squared(actual, predicted)
        ax.text(
            0.05, 0.95, f"R² = {r2:.3f}",
            transform=ax.transAxes, verticalalignment="top",
            fontsize=10, color="black",
        )

        # Right: residuals
        ax2 = axes[1]
        residuals = actual - predicted
        ax2.scatter(predicted, residuals, alpha=0.4, s=20, color="coral")
        ax2.axhline(0, color="black", linewidth=1)
        ax2.set_xlabel("Predicted Impact (bps)")
        ax2.set_ylabel("Residual (bps)")
        ax2.set_title("Residuals")

        fig.tight_layout()
        plt.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("Calibration plot saved to %s", save_path)

    def plot_impact_vs_size(
        self,
        model: ImpactModel,
        adv: float,
        save_path: str | Path,
        max_notional: float = 5_000_000,
        n_points: int = 200,
        sigma: float = 0.02,
    ) -> None:
        """
        Plot predicted impact vs trade size (USD notional) for a given ADV.

        Parameters
        ----------
        model : ImpactModel
        adv : float
            Average daily volume in USD (for computing participation rate).
        save_path : str | Path
        max_notional : float
            Maximum trade size to plot.
        n_points : int
        sigma : float
            Daily vol (used for A-C model).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed — cannot plot")
            return

        notionals = np.linspace(0, max_notional, n_points)
        prates = notionals / adv

        if model.model_type == "linear":
            impacts = model.alpha + model.beta * prates
        elif model.model_type == "sqrt":
            impacts = model.alpha + model.beta * np.sqrt(np.maximum(prates, 0))
        elif model.model_type == "almgren_chriss":
            impacts = (
                model.eta * np.sqrt(np.maximum(prates, 0)) * sigma
                + model.gamma * prates * sigma
            ) * 10_000
        else:
            raise ValueError(f"Unknown model_type: {model.model_type}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(notionals / 1e6, impacts, linewidth=2, color="steelblue",
                label=f"{model.model_type} model")
        ax.axvline(0.2, color="red", linestyle="--", linewidth=1, label="$200K Alpaca limit")
        ax.set_xlabel("Trade Size ($M notional)")
        ax.set_ylabel("Predicted Impact (bps)")
        ax.set_title(f"Impact vs Trade Size  |  ADV = ${adv/1e6:.1f}M")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plt.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("Impact vs size plot saved to %s", save_path)

    # -----------------------------------------------------------------------
    # Model inventory
    # -----------------------------------------------------------------------

    def get_model(self, model_type: str) -> ImpactModel:
        """Return a previously calibrated model."""
        if model_type not in self._calibrated_models:
            raise KeyError(f"Model '{model_type}' has not been calibrated yet")
        return self._calibrated_models[model_type]

    def summary(self) -> pd.DataFrame:
        """Return DataFrame summarising all calibrated models."""
        rows = []
        for name, m in self._calibrated_models.items():
            rows.append({
                "model": name,
                "alpha": m.alpha,
                "beta": m.beta,
                "eta": m.eta,
                "gamma": m.gamma,
                "r_squared": m.r_squared,
                "n_obs": m.n_observations,
                "residual_std": m.residual_std,
            })
        return pd.DataFrame(rows)
