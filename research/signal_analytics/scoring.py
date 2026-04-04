"""
research/signal_analytics/scoring.py
======================================
Signal scoring and ranking models for portfolio construction.

Provides:
  - Composite signal scoring (IC-weighted, rank-weighted)
  - Dynamic signal weighting based on rolling IC
  - Signal decay-adjusted position sizing
  - Risk-parity signal weighting
  - Ensemble combination optimisation
  - Signal concentration and diversification scoring
  - BH-specific scoring: optimal mass/tf_score filter selection

Usage example
-------------
>>> scorer = SignalScorer()
>>> weights = scorer.ic_weighted_combination(trades, signal_cols)
>>> positions = scorer.decay_adjusted_sizing(delta_scores, decay_model)
>>> scorer.plot_signal_weight_history(weight_history, save_path="results/weights.png")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EnsembleOptimisationResult:
    """Result of optimal ensemble signal weighting."""
    weights: Dict[str, float]       # signal_name -> weight
    combined_ic: float              # IC of the combined signal
    individual_ics: Dict[str, float]
    improvement_over_equal_weight: float
    n_signals: int


@dataclass
class DecayAdjustedPositions:
    """Position sizes adjusted for signal decay."""
    positions: pd.Series            # signal -> position_size
    raw_signals: pd.Series
    half_life: float
    horizon: int
    scale_factor: float


@dataclass
class RiskParityWeights:
    """Risk parity signal weights."""
    weights: pd.Series              # signal -> weight
    signal_volatilities: pd.Series  # signal -> vol
    effective_n: float              # diversification score


# ---------------------------------------------------------------------------
# SignalScorer
# ---------------------------------------------------------------------------

class SignalScorer:
    """Signal scoring and ensemble construction.

    Parameters
    ----------
    ic_window     : rolling window for IC estimation (bars)
    min_ic        : minimum IC to include a signal
    decay_penalty : weight given to signal decay in scoring
    """

    def __init__(
        self,
        ic_window: int = 60,
        min_ic: float = 0.0,
        decay_penalty: float = 0.5,
    ) -> None:
        self.ic_window = ic_window
        self.min_ic = min_ic
        self.decay_penalty = decay_penalty

    # ------------------------------------------------------------------ #
    # IC-weighted signal combination
    # ------------------------------------------------------------------ #

    def ic_weighted_combination(
        self,
        trades: pd.DataFrame,
        signal_cols: List[str],
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        method: str = "spearman",
        normalise: bool = True,
    ) -> Dict[str, float]:
        """Compute IC-proportional weights for combining multiple signals.

        Weight_i = max(0, IC_i) / sum(max(0, IC_j))

        Signals with negative IC are assigned zero weight.

        Parameters
        ----------
        trades      : trade records
        signal_cols : list of signal column names
        return_col  : P&L column
        dollar_pos_col: position column
        method      : IC correlation method
        normalise   : if True, normalise each signal to unit std before combination

        Returns
        -------
        Dict[signal_name -> weight]
        """
        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        ics: dict[str, float] = {}
        for col in signal_cols:
            if col not in df.columns:
                ics[col] = float("nan")
                continue
            sub = df[[col, "_ret"]].dropna()
            if len(sub) < 5:
                ics[col] = float("nan")
                continue
            r, _ = stats.spearmanr(sub[col], sub["_ret"])
            ics[col] = float(r)

        # Weight proportional to positive IC
        valid = {k: max(0, v) for k, v in ics.items() if not np.isnan(v)}
        total_ic = sum(valid.values())
        if total_ic == 0:
            # Equal weight as fallback
            n = len(valid)
            weights = {k: 1.0 / n for k in valid} if n > 0 else {}
        else:
            weights = {k: v / total_ic for k, v in valid.items()}

        return weights

    # ------------------------------------------------------------------ #
    # Rolling IC-weighted combination
    # ------------------------------------------------------------------ #

    def rolling_ic_weights(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        window: int = 60,
        method: str = "spearman",
        smoothing: float = 0.0,
    ) -> pd.DataFrame:
        """Compute time-varying IC weights for signal combination.

        At each time step, weights are computed from the trailing IC
        of each signal over the past *window* periods.

        Parameters
        ----------
        signal_df : DataFrame[time x signals] of signal values
        returns_df: DataFrame[time] or [time x signals] of returns
        window    : rolling window for IC estimation
        method    : IC correlation method
        smoothing : EWM smoothing halflife for weights (0 = no smoothing)

        Returns
        -------
        pd.DataFrame[time x signals] of weight time-series
        """
        common_idx = signal_df.index.intersection(returns_df.index)
        sig = signal_df.loc[common_idx]
        ret = returns_df.loc[common_idx] if isinstance(returns_df, pd.DataFrame) else returns_df.loc[common_idx]

        if isinstance(ret, pd.Series):
            # Same return for all signals
            ret_df = pd.DataFrame({col: ret for col in sig.columns})
        else:
            cols = sig.columns.intersection(ret.columns)
            sig = sig[cols]
            ret_df = ret[cols]

        weight_records: list[pd.Series] = []
        idx_records: list = []

        for i in range(window - 1, len(common_idx)):
            t_slice = common_idx[i - window + 1 : i + 1]
            sig_w = sig.loc[t_slice]
            ret_w = ret_df.loc[t_slice]

            ic_dict: dict[str, float] = {}
            for col in sig.columns:
                sv = sig_w[col].dropna().values
                rv = ret_w[col].dropna().values if col in ret_w.columns else ret_df.iloc[i - window + 1 : i + 1].mean(axis=1).values
                n = min(len(sv), len(rv))
                if n < 3:
                    ic_dict[col] = 0.0
                    continue
                if method == "spearman":
                    r, _ = stats.spearmanr(sv[:n], rv[:n])
                else:
                    r, _ = stats.pearsonr(sv[:n], rv[:n])
                ic_dict[col] = max(0.0, float(r))

            total = sum(ic_dict.values())
            if total > 0:
                w_series = pd.Series({k: v / total for k, v in ic_dict.items()})
            else:
                w_series = pd.Series({k: 1.0 / len(ic_dict) for k in ic_dict})

            weight_records.append(w_series)
            idx_records.append(common_idx[i])

        if not weight_records:
            return pd.DataFrame()

        weight_df = pd.DataFrame(weight_records, index=idx_records)

        if smoothing > 0:
            weight_df = weight_df.ewm(halflife=smoothing).mean()

        return weight_df

    # ------------------------------------------------------------------ #
    # Decay-adjusted position sizing
    # ------------------------------------------------------------------ #

    def decay_adjusted_sizing(
        self,
        delta_scores: pd.Series,
        half_life: float,
        horizon: int,
        base_position: float = 1.0,
    ) -> DecayAdjustedPositions:
        """Scale positions by the expected IC at the holding horizon.

        Position_i = base_position * IC(horizon) / IC(0) * |delta_score_i| / mean(|delta_score|)

        Parameters
        ----------
        delta_scores   : signal values (delta_score or ensemble_signal)
        half_life      : signal half-life in bars
        horizon        : target holding horizon in bars
        base_position  : base position size in dollar terms

        Returns
        -------
        DecayAdjustedPositions
        """
        # IC scale factor: how much IC remains at holding horizon
        decay_rate = np.log(2) / half_life if half_life > 0 and not np.isnan(half_life) else 0.0
        ic_ratio = np.exp(-decay_rate * horizon)

        # Normalise delta_scores
        abs_scores = delta_scores.abs()
        mean_abs = float(abs_scores.mean())
        if mean_abs > 0:
            normalised = abs_scores / mean_abs
        else:
            normalised = abs_scores

        positions = delta_scores.apply(np.sign) * normalised * base_position * ic_ratio

        return DecayAdjustedPositions(
            positions=positions,
            raw_signals=delta_scores,
            half_life=half_life,
            horizon=horizon,
            scale_factor=float(ic_ratio),
        )

    # ------------------------------------------------------------------ #
    # Risk-parity signal weights
    # ------------------------------------------------------------------ #

    def risk_parity_weights(
        self,
        signal_df: pd.DataFrame,
        risk_budget: Optional[Dict[str, float]] = None,
    ) -> RiskParityWeights:
        """Compute risk-parity weights: allocate equal risk contribution per signal.

        Each signal's weight is inversely proportional to its volatility
        so that each signal contributes equally to portfolio risk.

        Parameters
        ----------
        signal_df    : DataFrame[time x signals]
        risk_budget  : optional dict of target risk fractions (default equal)

        Returns
        -------
        RiskParityWeights
        """
        vols = signal_df.std(ddof=1)
        vols = vols.replace(0, np.nan).dropna()

        if len(vols) == 0:
            return RiskParityWeights(
                weights=pd.Series(dtype=float),
                signal_volatilities=pd.Series(dtype=float),
                effective_n=float("nan"),
            )

        # Inverse vol weights
        inv_vols = 1.0 / vols
        if risk_budget:
            budget = pd.Series(risk_budget).reindex(vols.index).fillna(1.0)
            inv_vols = inv_vols * budget

        weights = inv_vols / inv_vols.sum()

        # Effective N: 1/HHI
        hhi = float((weights**2).sum())
        effective_n = 1 / hhi if hhi > 0 else float("nan")

        return RiskParityWeights(
            weights=weights,
            signal_volatilities=vols,
            effective_n=effective_n,
        )

    # ------------------------------------------------------------------ #
    # Optimal ensemble via mean-variance optimisation
    # ------------------------------------------------------------------ #

    def optimal_ensemble_weights(
        self,
        trades: pd.DataFrame,
        signal_cols: List[str],
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        regularisation: float = 0.01,
    ) -> EnsembleOptimisationResult:
        """Find signal weights maximising combined IC using mean-variance optimisation.

        Objective: max w' * IC_vec - reg * w' * Sigma * w
        Subject to: sum(w) = 1, w >= 0

        Parameters
        ----------
        trades         : trade records
        signal_cols    : signal columns to combine
        return_col     : P&L column
        dollar_pos_col : position column
        regularisation : L2 regularisation coefficient

        Returns
        -------
        EnsembleOptimisationResult
        """
        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        avail = [c for c in signal_cols if c in df.columns]
        sub = df[avail + ["_ret"]].dropna()

        if len(sub) < 10 or len(avail) < 2:
            # Fallback to equal weight
            eq_w = {c: 1.0 / len(avail) for c in avail}
            return EnsembleOptimisationResult(
                weights=eq_w,
                combined_ic=float("nan"),
                individual_ics={c: float("nan") for c in avail},
                improvement_over_equal_weight=0.0,
                n_signals=len(avail),
            )

        # Individual ICs
        ics: dict[str, float] = {}
        for col in avail:
            r, _ = stats.spearmanr(sub[col], sub["_ret"])
            ics[col] = float(r)

        # Correlation matrix of signals
        sig_corr = sub[avail].corr().values
        ic_vec = np.array([ics[c] for c in avail])

        # Mean-variance: maximise IC - lambda * variance of IC
        n_sig = len(avail)

        def neg_objective(w: np.ndarray) -> float:
            portfolio_ic = float(np.dot(w, ic_vec))
            variance = float(w @ sig_corr @ w)
            return -(portfolio_ic - regularisation * variance)

        # Constraints: weights sum to 1, non-negative
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n_sig
        w0 = np.ones(n_sig) / n_sig

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    neg_objective, w0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 500, "ftol": 1e-9},
                )
            opt_w = res.x
            opt_w = np.maximum(opt_w, 0)
            opt_w /= opt_w.sum()
        except Exception:
            opt_w = w0

        weights = dict(zip(avail, opt_w.tolist()))

        # Combined IC
        combined_sig = sum(opt_w[i] * sub[avail[i]] for i in range(n_sig))
        r_comb, _ = stats.spearmanr(combined_sig, sub["_ret"])
        combined_ic = float(r_comb)

        # Equal-weight combined IC for comparison
        eq_combined = sub[avail].mean(axis=1)
        r_eq, _ = stats.spearmanr(eq_combined, sub["_ret"])
        improvement = combined_ic - float(r_eq)

        return EnsembleOptimisationResult(
            weights=weights,
            combined_ic=combined_ic,
            individual_ics=ics,
            improvement_over_equal_weight=improvement,
            n_signals=n_sig,
        )

    # ------------------------------------------------------------------ #
    # BH optimal filter selection
    # ------------------------------------------------------------------ #

    def optimal_bh_filters(
        self,
        trades: pd.DataFrame,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
    ) -> Dict[str, object]:
        """Find optimal BH signal filter thresholds via grid search.

        Jointly optimises mass_min and tf_score_min to maximise ICIR.

        Parameters
        ----------
        trades        : trade records with tf_score and mass columns
        return_col    : P&L column
        dollar_pos_col: position column

        Returns
        -------
        Dict with optimal_mass_min, optimal_tf_score_min, best_icir,
              best_ic, best_n_trades, grid_results (DataFrame)
        """
        if "mass" not in trades.columns or "tf_score" not in trades.columns:
            return {}

        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        if "ensemble_signal" not in df.columns:
            return {}

        mass_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        tf_thresholds = [0, 1, 2, 3, 4, 5]

        records: list[dict] = []
        best_icir = float("-inf")
        best_mass, best_tf = 0.0, 0

        for mass_min in mass_thresholds:
            for tf_min in tf_thresholds:
                sub = df[(df["mass"] >= mass_min) & (df["tf_score"] >= tf_min)]
                sub_clean = sub[["ensemble_signal", "_ret"]].dropna()
                if len(sub_clean) < 10:
                    records.append({
                        "mass_min": mass_min, "tf_min": tf_min,
                        "n_trades": len(sub_clean), "ic": float("nan"), "icir": float("nan"),
                    })
                    continue

                r, _ = stats.spearmanr(sub_clean["ensemble_signal"], sub_clean["_ret"])
                ic = float(r)

                # Approximate ICIR from sub-windows
                window_ics: list[float] = []
                step = max(1, len(sub_clean) // 10)
                for i in range(0, len(sub_clean) - step, step):
                    w = sub_clean.iloc[i : i + step]
                    r_w, _ = stats.spearmanr(w["ensemble_signal"], w["_ret"])
                    window_ics.append(float(r_w))

                if len(window_ics) >= 2:
                    ic_s = pd.Series(window_ics)
                    icir_val = float(ic_s.mean() / ic_s.std(ddof=1)) if ic_s.std(ddof=1) > 0 else float("nan")
                else:
                    icir_val = float("nan")

                records.append({
                    "mass_min": mass_min, "tf_min": tf_min,
                    "n_trades": len(sub_clean), "ic": ic, "icir": icir_val,
                })

                if not np.isnan(icir_val) and icir_val > best_icir:
                    best_icir = icir_val
                    best_mass = mass_min
                    best_tf = tf_min

        grid_df = pd.DataFrame(records)
        best_row = grid_df[
            (grid_df["mass_min"] == best_mass) & (grid_df["tf_min"] == best_tf)
        ]

        return {
            "optimal_mass_min": best_mass,
            "optimal_tf_score_min": best_tf,
            "best_icir": best_icir,
            "best_ic": float(best_row["ic"].values[0]) if len(best_row) > 0 else float("nan"),
            "best_n_trades": int(best_row["n_trades"].values[0]) if len(best_row) > 0 else 0,
            "grid_results": grid_df,
        }

    # ------------------------------------------------------------------ #
    # Adaptive signal sizing
    # ------------------------------------------------------------------ #

    def adaptive_signal_sizes(
        self,
        delta_scores: pd.Series,
        rolling_ic: pd.Series,
        base_size: float = 1.0,
        clip: float = 3.0,
    ) -> pd.Series:
        """Scale signal sizes by rolling IC to suppress weak-signal periods.

        position_size = base_size * |delta_score| * IC_rolling / max(IC_rolling)

        Parameters
        ----------
        delta_scores : current signal values
        rolling_ic   : rolling IC series (must be aligned with delta_scores)
        base_size    : base dollar position size
        clip         : maximum z-score clip for delta_score

        Returns
        -------
        pd.Series of adjusted position sizes
        """
        # Align
        common = delta_scores.index.intersection(rolling_ic.index)
        if len(common) == 0:
            return delta_scores.abs() * base_size

        ds = delta_scores.loc[common]
        ric = rolling_ic.loc[common].fillna(0).clip(0, None)  # Only positive IC amplifies

        # Normalise delta_score to z-score, clip
        ds_std = ds.std(ddof=1)
        if ds_std > 0:
            ds_z = (ds / ds_std).clip(-clip, clip)
        else:
            ds_z = ds

        # Normalise rolling IC
        max_ric = float(ric.max())
        if max_ric > 0:
            ic_scale = ric / max_ric
        else:
            ic_scale = pd.Series(1.0, index=ric.index)

        sizes = ds_z.abs() * ic_scale * base_size
        return sizes

    # ------------------------------------------------------------------ #
    # Signal scoring summary
    # ------------------------------------------------------------------ #

    def score_all_signals(
        self,
        trades: pd.DataFrame,
        signal_cols: Optional[List[str]] = None,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
    ) -> pd.DataFrame:
        """Compute composite quality scores for all available signals.

        Scoring dimensions:
          - IC magnitude and statistical significance
          - ICIR (risk-adjusted IC)
          - Positive IC fraction
          - Signal stability (low autocorrelation = fast-decaying = lower score)

        Parameters
        ----------
        trades      : trade records
        signal_cols : signal columns to score (auto-detect if None)
        return_col  : P&L column
        dollar_pos_col: position column

        Returns
        -------
        pd.DataFrame[signal x (ic, icir_approx, pct_pos, stability_score, composite_score)]
        """
        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        if signal_cols is None:
            exclude = {return_col, dollar_pos_col, "hold_bars", "entry_price",
                       "exit_price", "exit_time", "sym", "regime", "_ret"}
            signal_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                           if c not in exclude]

        records: list[dict] = []
        for col in signal_cols:
            if col not in df.columns:
                continue
            sub = df[[col, "_ret"]].dropna()
            n = len(sub)
            if n < 5:
                records.append({"signal": col, "composite_score": 0.0, "n_obs": n})
                continue

            r, p = stats.spearmanr(sub[col], sub["_ret"])
            ic = float(r)
            t = ic * np.sqrt(n - 2) / max(np.sqrt(max(1 - ic**2, 1e-10)), 1e-10)

            # Approximate ICIR from sub-windows
            step = max(5, n // 10)
            window_ics: list[float] = []
            for i in range(0, n - step, step):
                w = sub.iloc[i : i + step]
                r_w, _ = stats.spearmanr(w[col], w["_ret"])
                window_ics.append(float(r_w))

            icir_approx = float("nan")
            pct_pos = float("nan")
            if len(window_ics) >= 2:
                ic_s = pd.Series(window_ics)
                icir_approx = float(ic_s.mean() / ic_s.std(ddof=1)) if ic_s.std(ddof=1) > 0 else float("nan")
                pct_pos = float((ic_s > 0).mean())

            # Signal autocorrelation (high = slow decay = good for holding)
            sig_vals = sub[col].values
            n_s = len(sig_vals)
            mean_s = sig_vals.mean()
            var_s = np.var(sig_vals, ddof=1)
            acf1 = float("nan")
            if var_s > 0 and n_s > 2:
                cov1 = np.mean((sig_vals[:n_s-1] - mean_s) * (sig_vals[1:] - mean_s))
                acf1 = cov1 / var_s

            # Composite score (0-100):
            # IC component: ic * 100, capped at 100
            ic_score = max(-100, min(100, ic * 500))  # Scale: IC=0.2 -> 100
            icir_score = max(-50, min(50, icir_approx * 25)) if not np.isnan(icir_approx) else 0
            sig_score = max(0, min(100, float(t) * 15)) if not np.isnan(t) else 0
            composite = float(
                max(0, min(100, 0.4 * ic_score + 0.3 * icir_score + 0.3 * sig_score))
            )

            records.append({
                "signal": col,
                "ic": ic,
                "icir_approx": icir_approx,
                "pct_pos_ic": pct_pos,
                "t_stat": float(t),
                "p_value": float(p),
                "signal_acf1": acf1,
                "composite_score": composite,
                "n_obs": n,
            })

        return pd.DataFrame(records).set_index("signal").sort_values("composite_score", ascending=False)

    # ------------------------------------------------------------------ #
    # Visualisations
    # ------------------------------------------------------------------ #

    def plot_signal_weight_history(
        self,
        weight_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
        title: str = "Rolling Signal Weights",
    ) -> plt.Figure:
        """Stacked area chart of signal weight time-series.

        Parameters
        ----------
        weight_df : output of rolling_ic_weights() — DataFrame[time x signals]
        save_path : optional save path
        title     : figure title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        palette = plt.cm.Set1(np.linspace(0, 1, len(weight_df.columns)))

        bottoms = np.zeros(len(weight_df))
        for i, col in enumerate(weight_df.columns):
            vals = weight_df[col].fillna(0).values
            ax.fill_between(
                weight_df.index, bottoms, bottoms + vals,
                alpha=0.7, color=palette[i], label=col,
            )
            bottoms += vals

        ax.set_ylim(0, 1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_optimal_filter_heatmap(
        self,
        grid_results: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Heatmap of ICIR across mass_min x tf_score_min grid.

        Parameters
        ----------
        grid_results : 'grid_results' DataFrame from optimal_bh_filters()
        save_path    : optional save path
        """
        import seaborn as sns

        pivot = grid_results.pivot(index="mass_min", columns="tf_min", values="icir")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".2f", center=0, cmap="RdYlGn",
            cbar_kws={"label": "ICIR"},
            linewidths=0.5,
        )
        ax.set_title("ICIR by Mass Threshold and TF-Score Threshold")
        ax.set_xlabel("Min TF-Score")
        ax.set_ylabel("Min Mass")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_ensemble_weights_pie(
        self,
        ensemble_result: EnsembleOptimisationResult,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Pie chart of optimal ensemble signal weights.

        Parameters
        ----------
        ensemble_result : EnsembleOptimisationResult from optimal_ensemble_weights()
        save_path       : optional save path
        """
        weights = ensemble_result.weights
        labels = list(weights.keys())
        sizes = [weights[k] for k in labels]

        # Filter out near-zero weights
        threshold = 0.01
        filtered = [(l, s) for l, s in zip(labels, sizes) if s >= threshold]
        if filtered:
            labels, sizes = zip(*filtered)
        else:
            labels, sizes = list(labels), list(sizes)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
        )
        ax.set_title(
            f"Optimal Signal Weights\n"
            f"Combined IC={ensemble_result.combined_ic:.4f}  "
            f"Improvement={ensemble_result.improvement_over_equal_weight:.4f}"
        )
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_score_summary(
        self,
        score_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
        n_top: int = 15,
    ) -> plt.Figure:
        """Horizontal bar chart of composite signal scores.

        Parameters
        ----------
        score_df : output of score_all_signals()
        save_path: optional save path
        n_top    : number of top signals to show
        """
        top = score_df.head(n_top)
        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(top) * 0.4)))

        # Composite score
        ax1 = axes[0]
        colors = plt.cm.RdYlGn(top["composite_score"].values / 100)
        ax1.barh(range(len(top)), top["composite_score"].values, color=colors, alpha=0.85)
        ax1.set_yticks(range(len(top)))
        ax1.set_yticklabels(top.index.tolist(), fontsize=9)
        ax1.set_xlabel("Composite Score (0-100)")
        ax1.set_title("Signal Composite Score")

        # IC bar chart
        ax2 = axes[1]
        ic_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top["ic"].values]
        ax2.barh(range(len(top)), top["ic"].values, color=ic_colors, alpha=0.85)
        ax2.set_yticks(range(len(top)))
        ax2.set_yticklabels(top.index.tolist(), fontsize=9)
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("IC (Spearman)")
        ax2.set_title("Signal IC")

        fig.suptitle("Signal Scoring Dashboard", fontsize=12)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig
