"""
research/param_explorer/visualization.py
==========================================
Publication-quality dashboards and interactive visualisations for the
parameter space explorer.

Classes
-------
ParamExplorerDashboard : Stateful builder that holds references to results
                         and emits multi-panel matplotlib figures or HTML.

Stand-alone functions
---------------------
create_sensitivity_dashboard  : Multi-panel SA figure (OAT + Sobol + Morris)
create_landscape_dashboard    : Multi-panel landscape figure
create_bayesian_opt_dashboard : Multi-panel BO figure
interactive_param_explorer    : Plotly HTML (with matplotlib fallback)
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from research.param_explorer.space import ParamSpace
from research.param_explorer.sensitivity import (
    OATResult,
    SobolResult,
    MorrisResult,
    plot_oat_curves,
    plot_sobol_indices,
    plot_morris_mu_star_sigma,
)
from research.param_explorer.landscape import (
    LandscapeGrid,
    Basin,
    ObjectiveLandscape,
    plot_landscape_heatmap,
    plot_landscape_3d,
    plot_contour,
    roughness_index,
    flatness_score,
)
from research.param_explorer.bayesian_opt import (
    BayesOptResult,
    MOBayesOptResult,
    plot_convergence,
    plot_surrogate_1d,
    plot_pareto_front,
)

logger = logging.getLogger(__name__)

_PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    import plotly.subplots as psp
    import plotly.express as px
    _PLOTLY_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PALETTE = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "purple": "#8172B2",
    "brown": "#937860",
    "pink": "#DA8BC3",
    "grey": "#8C8C8C",
    "teal": "#64B5CD",
    "yellow": "#CCB974",
}


# ---------------------------------------------------------------------------
# Stand-alone dashboard functions
# ---------------------------------------------------------------------------

def create_sensitivity_dashboard(
    sensitivity_result: Dict[str, Any],
    save_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (18, 14),
) -> plt.Figure:
    """
    Create a multi-panel sensitivity analysis dashboard.

    Expects *sensitivity_result* to be a dict with some of the keys:
    'oat' (OATResult), 'sobol' (SobolResult), 'morris' (MorrisResult).

    Parameters
    ----------
    sensitivity_result : dict
    save_dir : str | Path | None
        Directory to save individual PNGs and a combined figure.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    oat: Optional[OATResult] = sensitivity_result.get("oat")
    sobol: Optional[SobolResult] = sensitivity_result.get("sobol")
    morris: Optional[MorrisResult] = sensitivity_result.get("morris")

    panels_needed = int(oat is not None) + int(sobol is not None) + int(morris is not None)
    if panels_needed == 0:
        raise ValueError("sensitivity_result must contain at least one of 'oat', 'sobol', 'morris'.")

    save_dir_path = Path(save_dir) if save_dir is not None else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    # --- Combined figure layout ---
    n_rows = 2
    n_cols = 3
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    fig.suptitle("Parameter Sensitivity Analysis Dashboard", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel 1: OAT sensitivity ranges (bar chart) ──────────────────
    ax_oat_bar = fig.add_subplot(gs[0, 0])
    if oat is not None:
        names_oat = sorted(oat.param_names, key=lambda n: oat.sensitivity_rank[n])
        ranges = [oat.sensitivity_range[n] for n in names_oat]
        colors_oat = [PALETTE["blue"] if oat.sensitivity_rank[n] <= 3 else PALETTE["grey"]
                      for n in names_oat]
        bars = ax_oat_bar.barh(names_oat, ranges, color=colors_oat, edgecolor="white", alpha=0.85)
        ax_oat_bar.set_xlabel("Objective range (max-min)", fontsize=9)
        ax_oat_bar.set_title("OAT Sensitivity Ranking", fontsize=10, fontweight="bold")
        ax_oat_bar.grid(axis="x", alpha=0.3)
    else:
        ax_oat_bar.set_visible(False)

    # ── Panel 2: Sobol Si / STi ───────────────────────────────────────
    ax_sobol = fig.add_subplot(gs[0, 1])
    if sobol is not None:
        names_s = sorted(sobol.Si.keys(), key=lambda n: sobol.STi[n], reverse=True)
        x_s = np.arange(len(names_s))
        si_vals = [sobol.Si[n] for n in names_s]
        sti_vals = [sobol.STi[n] for n in names_s]
        si_err = [sobol.Si_conf[n] for n in names_s]
        sti_err = [sobol.STi_conf[n] for n in names_s]
        w = 0.35
        ax_sobol.bar(x_s - w / 2, si_vals, w, label="Si", color=PALETTE["blue"],
                     alpha=0.85, yerr=si_err, capsize=2)
        ax_sobol.bar(x_s + w / 2, sti_vals, w, label="STi", color=PALETTE["orange"],
                     alpha=0.85, yerr=sti_err, capsize=2)
        ax_sobol.set_xticks(x_s)
        ax_sobol.set_xticklabels(names_s, rotation=30, ha="right", fontsize=7)
        ax_sobol.set_ylabel("Index", fontsize=9)
        ax_sobol.set_title("Sobol Sensitivity Indices", fontsize=10, fontweight="bold")
        ax_sobol.legend(fontsize=7)
        ax_sobol.grid(axis="y", alpha=0.3)
    else:
        ax_sobol.set_visible(False)

    # ── Panel 3: Morris μ*/σ ─────────────────────────────────────────
    ax_morris = fig.add_subplot(gs[0, 2])
    if morris is not None:
        names_m = list(morris.mu_star.keys())
        mu_s = [morris.mu_star[n] for n in names_m]
        sig_s = [morris.sigma[n] for n in names_m]
        scatter = ax_morris.scatter(mu_s, sig_s, c=range(len(names_m)),
                                    cmap="tab10", s=60, edgecolors="black", lw=0.5, zorder=3)
        for i, name in enumerate(names_m):
            ax_morris.annotate(name, (mu_s[i], sig_s[i]),
                               xytext=(5, 3), textcoords="offset points", fontsize=7)
        max_val = max(max(mu_s, default=1), max(sig_s, default=1)) * 1.1
        ax_morris.plot([0, max_val], [0, max_val], ls="--", color="grey", lw=0.8, alpha=0.5)
        ax_morris.set_xlabel("μ*", fontsize=9)
        ax_morris.set_ylabel("σ", fontsize=9)
        ax_morris.set_title("Morris μ*/σ", fontsize=10, fontweight="bold")
        ax_morris.grid(True, alpha=0.3)
    else:
        ax_morris.set_visible(False)

    # ── Panel 4: OAT curve for top parameter ─────────────────────────
    ax_oat_curve = fig.add_subplot(gs[1, 0])
    if oat is not None and oat.param_names:
        top_param = min(oat.param_names, key=lambda n: oat.sensitivity_rank[n])
        x_vals = oat.values[top_param]
        y_vals = oat.objectives[top_param]
        ax_oat_curve.plot(x_vals, y_vals, lw=2, color=PALETTE["blue"])
        ax_oat_curve.fill_between(x_vals, y_vals, y_vals.mean(), alpha=0.15, color=PALETTE["blue"])
        ax_oat_curve.set_xlabel(top_param, fontsize=9)
        ax_oat_curve.set_ylabel("Objective", fontsize=9)
        ax_oat_curve.set_title(f"OAT Curve: {top_param} (rank 1)", fontsize=10, fontweight="bold")
        ax_oat_curve.grid(True, alpha=0.3)
    else:
        ax_oat_curve.set_visible(False)

    # ── Panel 5: Sobol interaction (STi - Si) ────────────────────────
    ax_interact = fig.add_subplot(gs[1, 1])
    if sobol is not None:
        interaction = {n: max(0.0, sobol.STi[n] - sobol.Si[n]) for n in sobol.Si}
        names_int = sorted(interaction.keys(), key=lambda n: interaction[n], reverse=True)
        int_vals = [interaction[n] for n in names_int]
        ax_interact.bar(names_int, int_vals, color=PALETTE["purple"], alpha=0.85, edgecolor="white")
        ax_interact.set_xticklabels(names_int, rotation=30, ha="right", fontsize=7)
        ax_interact.set_ylabel("STi − Si", fontsize=9)
        ax_interact.set_title("Interaction Effects (STi − Si)", fontsize=10, fontweight="bold")
        ax_interact.grid(axis="y", alpha=0.3)
    else:
        ax_interact.set_visible(False)

    # ── Panel 6: Summary text ─────────────────────────────────────────
    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")
    summary_lines = ["Summary\n" + "─" * 25]
    if oat is not None:
        top3 = oat.top_k(3)
        summary_lines.append(f"OAT Top-3:")
        for nm, rng in top3:
            summary_lines.append(f"  {nm}: Δ={rng:.3g}")
    if sobol is not None:
        summary_lines.append(f"\nSobol (Var(Y)={sobol.var_y:.3g}):")
        for nm, val in sobol.top_k(3):
            summary_lines.append(f"  {nm}: STi={val:.3f}")
    if morris is not None:
        summary_lines.append(f"\nMorris (r={morris.n_trajectories}):")
        for nm, val in morris.top_k(3):
            summary_lines.append(f"  {nm}: μ*={val:.3g}")

    ax_text.text(0.05, 0.95, "\n".join(summary_lines),
                 transform=ax_text.transAxes, fontsize=9,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.5))

    if save_dir_path is not None:
        combined_path = save_dir_path / "sensitivity_dashboard.png"
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        logger.info("Sensitivity dashboard saved to %s", combined_path)

        # Also save individual plots
        if oat is not None:
            plot_oat_curves(oat, save_dir_path / "oat_curves.png")
        if sobol is not None:
            plot_sobol_indices(sobol, save_dir_path / "sobol_indices.png")
        if morris is not None:
            plot_morris_mu_star_sigma(morris, save_dir_path / "morris_mu_star_sigma.png")

    return fig


def create_landscape_dashboard(
    landscape_grid: LandscapeGrid,
    save_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (16, 10),
    basins: Optional[List[Basin]] = None,
) -> plt.Figure:
    """
    Multi-panel landscape analysis dashboard.

    Includes heatmap, contour, 3-D surface, and summary statistics.

    Parameters
    ----------
    landscape_grid : LandscapeGrid
    save_dir : str | Path | None
    figsize : tuple
    basins : list[Basin] | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    save_dir_path = Path(save_dir) if save_dir is not None else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Objective Landscape: {landscape_grid.p1_name} × {landscape_grid.p2_name}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Heatmap ─────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 0])
    Z = landscape_grid.Z
    p1v = landscape_grid.p1_values
    p2v = landscape_grid.p2_values
    im = ax_heat.pcolormesh(p2v, p1v, Z, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax_heat, label="Objective")
    i_b, j_b = landscape_grid.global_best_idx
    ax_heat.plot(p2v[j_b], p1v[i_b], "r*", ms=12, label="Best")
    if basins:
        for k, basin in enumerate(basins[:5]):
            ax_heat.plot(basin.p2_center, basin.p1_center, "wo", ms=7,
                         markeredgecolor="black", markeredgewidth=1)
    ax_heat.set_xlabel(landscape_grid.p2_name, fontsize=9)
    ax_heat.set_ylabel(landscape_grid.p1_name, fontsize=9)
    ax_heat.set_title("Heatmap", fontsize=10, fontweight="bold")
    ax_heat.legend(fontsize=8)

    # ── Panel 2: Contour ─────────────────────────────────────────────
    ax_cont = fig.add_subplot(gs[0, 1])
    z_finite = Z[~np.isnan(Z)]
    levels = np.linspace(z_finite.min(), z_finite.max(), 16)
    cf = ax_cont.contourf(p2v, p1v, Z, levels=levels, cmap="viridis")
    ax_cont.contour(p2v, p1v, Z, levels=levels, colors="k", linewidths=0.3, alpha=0.3)
    plt.colorbar(cf, ax=ax_cont, label="Objective")
    ax_cont.plot(p2v[j_b], p1v[i_b], "r*", ms=12)
    ax_cont.set_xlabel(landscape_grid.p2_name, fontsize=9)
    ax_cont.set_ylabel(landscape_grid.p1_name, fontsize=9)
    ax_cont.set_title("Contour", fontsize=10, fontweight="bold")

    # ── Panel 3: 3-D surface ─────────────────────────────────────────
    ax_3d = fig.add_subplot(gs[0, 2], projection="3d")
    P1 = landscape_grid.P1
    P2 = landscape_grid.P2
    Z_plot = Z.copy()
    Z_plot[np.isnan(Z_plot)] = float(np.nanmean(Z_plot))
    surf = ax_3d.plot_surface(P2, P1, Z_plot, cmap="viridis", alpha=0.85, edgecolor="none")
    ax_3d.set_xlabel(landscape_grid.p2_name, fontsize=7)
    ax_3d.set_ylabel(landscape_grid.p1_name, fontsize=7)
    ax_3d.set_zlabel("Obj", fontsize=7)
    ax_3d.set_title("3-D Surface", fontsize=10, fontweight="bold")
    ax_3d.view_init(elev=25, azim=-60)
    ax_3d.tick_params(labelsize=6)

    # ── Panel 4: Row-wise objective slices ───────────────────────────
    ax_slices = fig.add_subplot(gs[1, 0])
    stride = max(1, landscape_grid.n1 // 5)
    cmap_lines = plt.cm.get_cmap("plasma")
    for i in range(0, landscape_grid.n1, stride):
        ax_slices.plot(
            p2v, Z[i, :],
            lw=1.2, alpha=0.7,
            color=cmap_lines(i / landscape_grid.n1),
            label=f"{landscape_grid.p1_name}={p1v[i]:.3g}",
        )
    ax_slices.set_xlabel(landscape_grid.p2_name, fontsize=9)
    ax_slices.set_ylabel("Objective", fontsize=9)
    ax_slices.set_title("Row Slices", fontsize=10, fontweight="bold")
    ax_slices.legend(fontsize=6, ncol=2)
    ax_slices.grid(True, alpha=0.3)

    # ── Panel 5: Column-wise slices ──────────────────────────────────
    ax_slicesc = fig.add_subplot(gs[1, 1])
    stride2 = max(1, landscape_grid.n2 // 5)
    for j in range(0, landscape_grid.n2, stride2):
        ax_slicesc.plot(
            p1v, Z[:, j],
            lw=1.2, alpha=0.7,
            color=cmap_lines(j / landscape_grid.n2),
            label=f"{landscape_grid.p2_name}={p2v[j]:.3g}",
        )
    ax_slicesc.set_xlabel(landscape_grid.p1_name, fontsize=9)
    ax_slicesc.set_ylabel("Objective", fontsize=9)
    ax_slicesc.set_title("Column Slices", fontsize=10, fontweight="bold")
    ax_slicesc.legend(fontsize=6, ncol=2)
    ax_slicesc.grid(True, alpha=0.3)

    # ── Panel 6: Metrics summary ─────────────────────────────────────
    ax_metrics = fig.add_subplot(gs[1, 2])
    ax_metrics.axis("off")
    rough = roughness_index(landscape_grid)
    flat_10 = flatness_score(landscape_grid, threshold=0.10)
    flat_25 = flatness_score(landscape_grid, threshold=0.25)
    best_p = landscape_grid.global_best_params
    summary = [
        "Landscape Metrics\n" + "─" * 22,
        f"Global best: {landscape_grid.global_best_value:.5g}",
        f"  {landscape_grid.p1_name} = {best_p[landscape_grid.p1_name]:.4g}",
        f"  {landscape_grid.p2_name} = {best_p[landscape_grid.p2_name]:.4g}",
        f"Roughness index: {rough:.4g}",
        f"Flatness (10%): {flat_10:.1%}",
        f"Flatness (25%): {flat_25:.1%}",
        f"Grid: {landscape_grid.n1}×{landscape_grid.n2}",
    ]
    if basins:
        summary.append(f"Basins found: {len(basins)}")
        for k, b in enumerate(basins[:3]):
            summary.append(f"  B{k+1}: val={b.peak_value:.4g}, area={b.area_fraction:.1%}")

    ax_metrics.text(
        0.05, 0.95, "\n".join(summary),
        transform=ax_metrics.transAxes, fontsize=9,
        va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.5),
    )

    if save_dir_path is not None:
        combined_path = save_dir_path / "landscape_dashboard.png"
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        logger.info("Landscape dashboard saved to %s", combined_path)
        plot_landscape_heatmap(landscape_grid, save_dir_path / "landscape_heatmap.png", basins=basins)
        plot_landscape_3d(landscape_grid, save_dir_path / "landscape_3d.png")
        plot_contour(landscape_grid, save_dir_path / "landscape_contour.png")

    return fig


def create_bayesian_opt_dashboard(
    bayes_result: BayesOptResult,
    save_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (16, 10),
) -> plt.Figure:
    """
    Multi-panel Bayesian optimisation results dashboard.

    Includes convergence curves, score distribution, parameter correlation
    with objective, and the best parameter bar chart.

    Parameters
    ----------
    bayes_result : BayesOptResult
    save_dir : str | Path | None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    save_dir_path = Path(save_dir) if save_dir is not None else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Bayesian Optimisation Dashboard — {bayes_result.param_space_name}\n"
        f"(acq={bayes_result.acquisition.value}, n_init={bayes_result.n_init}, "
        f"n_iter={bayes_result.n_iter}, best={bayes_result.best_score:.4g})",
        fontsize=12, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    iters = np.arange(1, len(bayes_result.history_y) + 1)
    init_mask = iters <= bayes_result.n_init

    # ── Panel 1: Convergence ─────────────────────────────────────────
    ax_conv = fig.add_subplot(gs[0, 0])
    ax_conv.scatter(iters[init_mask], bayes_result.history_y[init_mask],
                    s=18, color=PALETTE["blue"], alpha=0.7, label="Init")
    ax_conv.scatter(iters[~init_mask], bayes_result.history_y[~init_mask],
                    s=18, color=PALETTE["orange"], alpha=0.7, label="BO")
    ax_conv.plot(iters, bayes_result.convergence, color="black", lw=2, label="Best")
    ax_conv.axvline(bayes_result.n_init + 0.5, ls="--", color="grey", lw=0.8)
    ax_conv.set_xlabel("Eval #", fontsize=9)
    ax_conv.set_ylabel("Score", fontsize=9)
    ax_conv.set_title("Convergence", fontsize=10, fontweight="bold")
    ax_conv.legend(fontsize=7)
    ax_conv.grid(True, alpha=0.3)

    # ── Panel 2: Score histogram ──────────────────────────────────────
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_hist.hist(bayes_result.history_y[~init_mask], bins=20, color=PALETTE["orange"],
                 alpha=0.7, edgecolor="white", label="BO")
    ax_hist.hist(bayes_result.history_y[init_mask], bins=8, color=PALETTE["blue"],
                 alpha=0.6, edgecolor="white", label="Init")
    ax_hist.axvline(bayes_result.best_score, ls="--", color="red", lw=1.5,
                    label=f"Best={bayes_result.best_score:.3g}")
    ax_hist.set_xlabel("Score", fontsize=9)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.set_title("Score Distribution", fontsize=10, fontweight="bold")
    ax_hist.legend(fontsize=7)
    ax_hist.grid(True, alpha=0.3)

    # ── Panel 3: Best params bar chart ───────────────────────────────
    ax_params = fig.add_subplot(gs[0, 2])
    param_names = list(bayes_result.best_params.keys())
    # Normalise to [0,1] range by looking at history
    X_hist = bayes_result.history_X
    param_vals_unit = []
    for j in range(min(len(param_names), X_hist.shape[1])):
        param_vals_unit.append(
            X_hist[np.argmax(bayes_result.history_y), j]
        )
    if len(param_vals_unit) == len(param_names):
        bars = ax_params.barh(
            param_names, param_vals_unit,
            color=PALETTE["green"], alpha=0.85, edgecolor="white",
        )
        ax_params.set_xlim(0, 1)
        ax_params.set_xlabel("Unit-space value", fontsize=9)
        ax_params.set_title("Best Params (unit space)", fontsize=10, fontweight="bold")
        ax_params.axvline(0.5, ls="--", color="grey", lw=0.8, alpha=0.5)
        ax_params.grid(axis="x", alpha=0.3)
    else:
        ax_params.text(0.5, 0.5, "N/A", ha="center", va="center")

    # ── Panel 4: Per-param acquisition trace ─────────────────────────
    ax_trace = fig.add_subplot(gs[1, 0])
    if X_hist.shape[1] > 0:
        n_p = min(5, X_hist.shape[1])
        cmap_t = plt.cm.get_cmap("tab10")
        for j in range(n_p):
            label = param_names[j] if j < len(param_names) else f"dim{j}"
            ax_trace.plot(iters, X_hist[:, j], lw=1, alpha=0.7,
                          color=cmap_t(j), label=label)
        ax_trace.axvline(bayes_result.n_init + 0.5, ls="--", color="grey", lw=0.8)
        ax_trace.set_xlabel("Eval #", fontsize=9)
        ax_trace.set_ylabel("Unit value", fontsize=9)
        ax_trace.set_title("Param Traces", fontsize=10, fontweight="bold")
        ax_trace.legend(fontsize=6, ncol=2)
        ax_trace.grid(True, alpha=0.3)
    else:
        ax_trace.set_visible(False)

    # ── Panel 5: Surrogate uncertainty over iterations ────────────────
    ax_uncert = fig.add_subplot(gs[1, 1])
    if len(bayes_result.history_y) > bayes_result.n_init:
        bo_y = bayes_result.history_y[~init_mask]
        # Compute rolling std as proxy for uncertainty
        window = max(3, len(bo_y) // 10)
        rolled_std = pd.rolling_std_approx(bo_y, window)
        ax_uncert.plot(iters[~init_mask], rolled_std, color=PALETTE["purple"], lw=2)
        ax_uncert.set_xlabel("Eval #", fontsize=9)
        ax_uncert.set_ylabel("Rolling score std", fontsize=9)
        ax_uncert.set_title("Exploration Uncertainty (proxy)", fontsize=10, fontweight="bold")
        ax_uncert.grid(True, alpha=0.3)
    else:
        ax_uncert.set_visible(False)

    # ── Panel 6: Summary ─────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[1, 2])
    ax_sum.axis("off")
    top5 = bayes_result.top_k(5)
    summary_lines = [
        "Best Results\n" + "─" * 20,
    ]
    for rank, (p, s) in enumerate(top5, 1):
        summary_lines.append(f"#{rank}: score={s:.4g}")
        for k, v in list(p.items())[:3]:
            summary_lines.append(f"   {k}={_fmt(v)}")
    ax_sum.text(
        0.05, 0.97, "\n".join(summary_lines),
        transform=ax_sum.transAxes, fontsize=8, va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.5),
    )

    if save_dir_path is not None:
        combined_path = save_dir_path / "bayesopt_dashboard.png"
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        logger.info("BO dashboard saved to %s", combined_path)
        plot_convergence(bayes_result, save_dir_path / "convergence.png")

    return fig


def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.4g}"
    except Exception:
        return str(v)


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple manual rolling std."""
    result = np.zeros_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        result[i] = float(np.std(arr[lo : i + 1], ddof=0))
    return result


# Monkey-patch helper namespace so we can import it
class pd:
    @staticmethod
    def rolling_std_approx(arr: np.ndarray, window: int) -> np.ndarray:
        return _rolling_std(arr, window)


# ---------------------------------------------------------------------------
# Interactive explorer
# ---------------------------------------------------------------------------

def interactive_param_explorer(
    param_space: ParamSpace,
    objective_fn: Callable[[Dict[str, Any]], float],
    n_samples: int = 300,
    save_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> Union[str, plt.Figure]:
    """
    Build an interactive parameter explorer.

    If ``plotly`` is available, generates an HTML file with:
    - Parallel coordinates plot of all evaluated points coloured by score
    - Scatter matrix of top 4 parameters vs objective
    - 3-D scatter of top 3 parameters vs objective

    Falls back to a matplotlib multi-panel figure if plotly is not installed.

    Parameters
    ----------
    param_space : ParamSpace
    objective_fn : callable
    n_samples : int
    save_path : str | Path | None
        Path for the HTML file (plotly) or PNG (matplotlib).
    seed : int

    Returns
    -------
    str (HTML path) or matplotlib.figure.Figure
    """
    X = param_space.sample_latin_hypercube(n_samples, seed=seed)
    param_list = param_space.decode_matrix(X)
    y = np.array([objective_fn(p) for p in param_list])

    if _PLOTLY_AVAILABLE:
        return _interactive_plotly(param_space, X, param_list, y, save_path)
    else:
        return _interactive_matplotlib(param_space, X, param_list, y, save_path)


def _interactive_plotly(
    param_space: ParamSpace,
    X: np.ndarray,
    param_list: List[Dict[str, Any]],
    y: np.ndarray,
    save_path: Optional[Union[str, Path]],
) -> str:
    """Build a plotly HTML interactive explorer."""
    import plotly.graph_objects as go
    import plotly.subplots as psp
    import pandas as _pd

    records = []
    for i, p in enumerate(param_list):
        row = {k: float(v) if _is_numeric_val(v) else str(v) for k, v in p.items()}
        row["__score__"] = float(y[i])
        records.append(row)
    df = _pd.DataFrame(records)

    param_names = param_space.names
    numeric_names = [
        n for n in param_names
        if n in df.columns and _pd.api.types.is_numeric_dtype(df[n])
    ]

    # Parallel coordinates
    fig_parallel = go.Figure(
        go.Parcoords(
            line=dict(
                color=df["__score__"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Score"),
            ),
            dimensions=[
                dict(
                    label=n,
                    values=df[n],
                    range=[float(df[n].min()), float(df[n].max())],
                )
                for n in numeric_names
            ] + [dict(label="Score", values=df["__score__"])],
        )
    )
    fig_parallel.update_layout(
        title="Parallel Coordinates — Parameter Space Explorer",
        height=500,
    )

    # Top parameters by Spearman correlation with score
    from scipy.stats import spearmanr
    corrs = []
    for n in numeric_names:
        rho, _ = spearmanr(df[n], df["__score__"])
        corrs.append((n, abs(float(rho))))
    corrs.sort(key=lambda kv: kv[1], reverse=True)
    top4 = [n for n, _ in corrs[:4]]

    # Scatter matrix
    fig_scatter = go.Figure(
        go.Splom(
            dimensions=[
                dict(label=n, values=df[n])
                for n in top4 + ["__score__"]
            ],
            marker=dict(
                color=df["__score__"],
                size=4,
                colorscale="Viridis",
                showscale=True,
            ),
            diagonal_visible=True,
        )
    )
    fig_scatter.update_layout(
        title="Scatter Matrix — Top 4 Parameters",
        height=600,
    )

    # 3-D scatter (top 3 params)
    top3 = [n for n, _ in corrs[:3]]
    if len(top3) >= 3:
        fig_3d = go.Figure(
            go.Scatter3d(
                x=df[top3[0]], y=df[top3[1]], z=df[top3[2]],
                mode="markers",
                marker=dict(
                    size=4, color=df["__score__"],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Score"),
                ),
                text=[f"Score={s:.4g}" for s in y],
            )
        )
        fig_3d.update_layout(
            title=f"3-D Scatter: {top3[0]} × {top3[1]} × {top3[2]}",
            scene=dict(
                xaxis_title=top3[0],
                yaxis_title=top3[1],
                zaxis_title=top3[2],
            ),
            height=600,
        )
    else:
        fig_3d = None

    # Combine into HTML
    html_parts = [
        "<html><head><title>Parameter Space Explorer</title></head><body>",
        "<h1>Parameter Space Explorer</h1>",
        f"<p>Space: {param_space.name} | Samples: {len(y)} | "
        f"Best score: {float(y.max()):.4g}</p>",
        fig_parallel.to_html(full_html=False, include_plotlyjs="cdn"),
        fig_scatter.to_html(full_html=False, include_plotlyjs=False),
    ]
    if fig_3d is not None:
        html_parts.append(fig_3d.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append("</body></html>")
    html_content = "\n".join(html_parts)

    out_path = save_path or Path("param_explorer.html")
    out_path = Path(out_path)
    out_path.write_text(html_content, encoding="utf-8")
    logger.info("Interactive explorer saved to %s", out_path)
    return str(out_path)


def _is_numeric_val(v: Any) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


def _interactive_matplotlib(
    param_space: ParamSpace,
    X: np.ndarray,
    param_list: List[Dict[str, Any]],
    y: np.ndarray,
    save_path: Optional[Union[str, Path]],
) -> plt.Figure:
    """Matplotlib fallback for the interactive explorer."""
    param_names = param_space.names
    d = min(len(param_names), 6)
    selected = param_names[:d]
    selected_idx = [param_space._name_to_idx[n] for n in selected]
    X_sel = X[:, selected_idx]

    n_panels = d + 1
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)
    fig.suptitle(
        f"Parameter Space Explorer — {param_space.name}\n"
        f"({len(y)} LHS samples)",
        fontsize=12, fontweight="bold",
    )

    cmap = plt.cm.get_cmap("viridis")
    norm = Normalize(vmin=y.min(), vmax=y.max())
    colors = cmap(norm(y))

    for i, name in enumerate(selected):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        ax.scatter(X_sel[:, i], y, c=colors, s=12, alpha=0.6)
        ax.set_xlabel(f"{name} (unit)", fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(f"Score vs {name}", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Last panel: score histogram
    row, col = divmod(d, n_cols)
    ax_hist = axes[row][col]
    ax_hist.hist(y, bins=20, color=PALETTE["blue"], alpha=0.8, edgecolor="white")
    ax_hist.axvline(float(y.max()), ls="--", color="red", lw=1.5,
                    label=f"Best={float(y.max()):.3g}")
    ax_hist.set_xlabel("Score", fontsize=9)
    ax_hist.set_title("Score Distribution", fontsize=9)
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_panels, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = save_path or Path("param_explorer.png")
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info("Interactive explorer (matplotlib) saved to %s", out_path)
    return fig


# ---------------------------------------------------------------------------
# ParamExplorerDashboard class
# ---------------------------------------------------------------------------

class ParamExplorerDashboard:
    """
    Stateful dashboard builder that holds references to all analysis results
    and can emit various multi-panel figures.

    Parameters
    ----------
    param_space : ParamSpace
    output_dir : str | Path | None
    """

    def __init__(
        self,
        param_space: ParamSpace,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.param_space = param_space
        self.output_dir = Path(output_dir) if output_dir is not None else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._sensitivity_result: Optional[Dict[str, Any]] = None
        self._landscape_grid: Optional[LandscapeGrid] = None
        self._bayes_result: Optional[BayesOptResult] = None
        self._basins: Optional[List[Basin]] = None

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_sensitivity(self, result: Dict[str, Any]) -> "ParamExplorerDashboard":
        self._sensitivity_result = result
        return self

    def set_landscape(
        self,
        grid: LandscapeGrid,
        basins: Optional[List[Basin]] = None,
    ) -> "ParamExplorerDashboard":
        self._landscape_grid = grid
        self._basins = basins
        return self

    def set_bayes_result(self, result: BayesOptResult) -> "ParamExplorerDashboard":
        self._bayes_result = result
        return self

    # ------------------------------------------------------------------
    # Dashboard emitters
    # ------------------------------------------------------------------

    def sensitivity_dashboard(self) -> plt.Figure:
        if self._sensitivity_result is None:
            raise RuntimeError("No sensitivity result set. Call set_sensitivity() first.")
        return create_sensitivity_dashboard(
            self._sensitivity_result,
            save_dir=self.output_dir / "sensitivity",
        )

    def landscape_dashboard(self) -> plt.Figure:
        if self._landscape_grid is None:
            raise RuntimeError("No landscape grid set. Call set_landscape() first.")
        return create_landscape_dashboard(
            self._landscape_grid,
            save_dir=self.output_dir / "landscape",
            basins=self._basins,
        )

    def bayes_dashboard(self) -> plt.Figure:
        if self._bayes_result is None:
            raise RuntimeError("No Bayes result set. Call set_bayes_result() first.")
        return create_bayesian_opt_dashboard(
            self._bayes_result,
            save_dir=self.output_dir / "bayesopt",
        )

    def full_dashboard(self) -> List[plt.Figure]:
        """Emit all available dashboards."""
        figs = []
        if self._sensitivity_result is not None:
            figs.append(self.sensitivity_dashboard())
        if self._landscape_grid is not None:
            figs.append(self.landscape_dashboard())
        if self._bayes_result is not None:
            figs.append(self.bayes_dashboard())
        return figs

    def save_all(self) -> None:
        """Save all available dashboards to output_dir."""
        for fig in self.full_dashboard():
            plt.close(fig)
        logger.info("All dashboards saved to %s", self.output_dir)
