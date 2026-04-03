"""
live_controls.py — LARSA Live Parameter Controls Dashboard (v6)

Enhanced Streamlit dashboard with real-time parameter sliders that
automatically re-run the arena and refresh all charts.

Run with:
    streamlit run tools/live_controls.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LARSA Live Controls",
    page_icon="🎛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Arena import — handled gracefully so the UI still loads on import failure
# ---------------------------------------------------------------------------
_ARENA_IMPORT_ERROR: Optional[str] = None

try:
    sys.path.insert(0, os.path.dirname(__file__))
    from arena_v2 import run_v2, load_ohlcv, generate_synthetic, CONFIGS  # type: ignore
    _ARENA_OK = True
except Exception as exc:
    _ARENA_OK = False
    _ARENA_IMPORT_ERROR = str(exc)

# PID sizer (optional — graceful fallback if not found)
try:
    from pid_sizer import PIDSizer, _rolling_drawdown  # type: ignore
    _PID_OK = True
except Exception:
    _PID_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BEST_CONFIG_PATH = os.path.join(RESULTS_DIR, "best_config.json")
EXPORT_CONFIG_PATH = os.path.join(RESULTS_DIR, "exported_config.json")

DEFAULT_BASELINE = {
    "cf": 0.005,
    "bh_form": 1.5,
    "bh_decay": 0.95,
    "max_lev": 0.65,
}

REGIME_COLORS = {
    "BULL": "#2ecc71",
    "BEAR": "#e74c3c",
    "SIDEWAYS": "#95a5a6",
    "HIGH_VOLATILITY": "#9b59b6",
}


# ---------------------------------------------------------------------------
# Helper: load best known config (from autotuner results)
# ---------------------------------------------------------------------------
def _load_best_config() -> dict:
    if os.path.exists(BEST_CONFIG_PATH):
        try:
            with open(BEST_CONFIG_PATH) as f:
                data = json.load(f)
            # Return only the keys we care about, with defaults for missing ones
            return {
                "cf":       float(data.get("cf",       DEFAULT_BASELINE["cf"])),
                "bh_form":  float(data.get("bh_form",  DEFAULT_BASELINE["bh_form"])),
                "bh_decay": float(data.get("bh_decay", DEFAULT_BASELINE["bh_decay"])),
                "max_lev":  float(data.get("max_lev",  DEFAULT_BASELINE["max_lev"])),
            }
        except Exception:
            pass
    return dict(DEFAULT_BASELINE)


# ---------------------------------------------------------------------------
# Arena runner (cached by parameter hash)
# ---------------------------------------------------------------------------
def _param_hash(**kwargs) -> str:
    key = json.dumps(kwargs, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()


@st.cache_data(show_spinner=False)
def _run_arena(
    cf: float,
    bh_form: float,
    bh_decay: float,
    max_lev: float,
    solo_cap: float,
    conv_cap: float,
    n_bars: int,
    seed: int,
    _cache_key: str,  # forces cache invalidation when params change
) -> dict:
    """Run arena_v2 with synthetic data and return results dict."""
    if not _ARENA_OK:
        return {}

    bars = generate_synthetic(n_bars=n_bars, seed=seed)
    cfg = {
        "cf":          cf,
        "bh_form":     bh_form,
        "bh_collapse": 1.0,
        "bh_decay":    bh_decay,
    }
    broker, bar_log = run_v2(bars, cfg, max_leverage=max_lev, exp_flags="ABCD", verbose=True)
    stats = broker.stats()

    return {
        "equity_curve": list(broker.equity_curve),
        "stats":        stats,
        "bar_log":      bar_log,
        "trade_log":    list(broker.trade_log),
    }


# ---------------------------------------------------------------------------
# PID multiplier series from equity curve
# ---------------------------------------------------------------------------
def _compute_pid_series(equity_curve: list, target_dd: float) -> list:
    """Compute PID multipliers from an equity curve."""
    if not _PID_OK:
        return [1.0] * len(equity_curve)
    pid = PIDSizer(target_dd=target_dd)
    peak = 0.0
    mults = []
    for pv in equity_curve:
        dd, peak = _rolling_drawdown(pv, peak)
        mults.append(pid.update(dd))
    return mults


# ---------------------------------------------------------------------------
# Metric display helpers
# ---------------------------------------------------------------------------
def _delta_color(val: float, baseline: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "normal" if val >= baseline else "inverse"
    return "normal" if val <= baseline else "inverse"


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _fig_equity(equity_curve: list, label: str = "Strategy", color: str = "#2ecc71") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve, mode="lines", name=label,
        line=dict(color=color, width=2),
    ))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Bar",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def _fig_equity_comparison(
    curve_a: list, label_a: str,
    curve_b: list, label_b: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=curve_a, mode="lines", name=label_a,
        line=dict(color="#3498db", width=2),
    ))
    fig.add_trace(go.Scatter(
        y=curve_b, mode="lines", name=label_b,
        line=dict(color="#e67e22", width=2),
    ))
    fig.update_layout(
        title="Equity Comparison",
        xaxis_title="Bar",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        height=420,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def _fig_regime_timeline(bar_log: list) -> go.Figure:
    """Colored bar chart of regime per bar (downsampled for performance)."""
    if not bar_log:
        return go.Figure().update_layout(title="No bar log available", template="plotly_dark")

    # Build numeric regime index + hover text
    regime_order = ["BULL", "SIDEWAYS", "HIGH_VOLATILITY", "BEAR"]
    regime_idx = {r: i for i, r in enumerate(regime_order)}

    bars_x = list(range(len(bar_log)))
    bars_y = [regime_idx.get(b.get("regime", "SIDEWAYS"), 1) for b in bar_log]
    bar_colors = [REGIME_COLORS.get(b.get("regime", "SIDEWAYS"), "#95a5a6") for b in bar_log]

    fig = go.Figure()
    for regime, color in REGIME_COLORS.items():
        idxs = [i for i, b in enumerate(bar_log) if b.get("regime") == regime]
        if idxs:
            fig.add_trace(go.Bar(
                x=idxs,
                y=[1] * len(idxs),
                name=regime,
                marker_color=color,
                width=1.2,
                showlegend=True,
            ))

    fig.update_layout(
        title="Regime Timeline",
        xaxis_title="Bar",
        yaxis=dict(showticklabels=False),
        barmode="stack",
        template="plotly_dark",
        height=200,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.15),
    )
    return fig


def _fig_bh_mass(bar_log: list, bh_form_threshold: float) -> go.Figure:
    if not bar_log:
        return go.Figure().update_layout(title="No bar log available", template="plotly_dark")

    masses = [b.get("bh_mass", 0.0) for b in bar_log]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=masses, mode="lines", name="BH Mass",
        line=dict(color="#f39c12", width=1.5),
    ))
    fig.add_hline(
        y=bh_form_threshold,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Formation threshold ({bh_form_threshold})",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Black Hole Mass",
        xaxis_title="Bar",
        yaxis_title="BH Mass",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def _fig_position_sizing(
    equity_curve: list,
    pid_enabled: bool,
    pid_target_dd: float,
) -> go.Figure:
    """Bar chart of PID multiplier; also shows raw (1.0 line) if PID enabled."""
    fig = go.Figure()

    if pid_enabled and _PID_OK:
        mults = _compute_pid_series(equity_curve, pid_target_dd)
        # Downsample for chart performance
        step = max(1, len(mults) // 500)
        xs = list(range(0, len(mults), step))
        ys = [mults[i] for i in xs]
        fig.add_trace(go.Bar(
            x=xs, y=ys, name="PID Multiplier",
            marker_color="#3498db", opacity=0.75,
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="#95a5a6", annotation_text="Raw (1.0)")
    else:
        n = len(equity_curve)
        fig.add_trace(go.Scatter(
            x=[0, n - 1], y=[1.0, 1.0], mode="lines",
            name="Raw (fixed)", line=dict(color="#95a5a6", dash="dash"),
        ))

    fig.update_layout(
        title="Position Size Multiplier" + (" (PID active)" if pid_enabled else " (PID disabled)"),
        xaxis_title="Bar",
        yaxis_title="Multiplier",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Sidebar: parameter controls ----------------------------------------
    st.sidebar.title("LARSA Live Controls")
    st.sidebar.markdown("---")

    cf = st.sidebar.slider("cf (curvature factor)", 0.001, 0.015, 0.005, step=0.001, format="%.3f")
    bh_form = st.sidebar.slider("bh_form (BH formation threshold)", 1.0, 3.0, 1.5, step=0.1)
    bh_decay = st.sidebar.slider("bh_decay", 0.80, 0.99, 0.95, step=0.01)
    max_lev = st.sidebar.slider("max_lev", 0.30, 0.90, 0.65, step=0.05)
    solo_cap = st.sidebar.slider("solo_cap (v6)", 0.10, 0.45, 0.25, step=0.05)
    conv_cap = st.sidebar.slider("conv_cap", 0.40, 0.80, 0.60, step=0.05)

    st.sidebar.markdown("---")
    pid_enabled = st.sidebar.checkbox("Enable PID Sizer", value=False)
    pid_target_dd = 0.15
    if pid_enabled:
        pid_target_dd = st.sidebar.slider(
            "pid_target_dd (target max drawdown)", 0.10, 0.30, 0.15, step=0.01, format="%.2f"
        )
        if not _PID_OK:
            st.sidebar.warning("pid_sizer.py could not be imported. PID disabled.")

    st.sidebar.markdown("---")
    n_bars = st.sidebar.selectbox("Simulation bars", [5000, 10000, 20000], index=1)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    # --- Export config button -----------------------------------------------
    if st.sidebar.button("Export Config"):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        export = {
            "cf": cf, "bh_form": bh_form, "bh_decay": bh_decay,
            "max_lev": max_lev, "solo_cap": solo_cap, "conv_cap": conv_cap,
            "pid_enabled": pid_enabled, "pid_target_dd": pid_target_dd,
        }
        with open(EXPORT_CONFIG_PATH, "w") as f:
            json.dump(export, f, indent=2)
        st.sidebar.success(f"Config saved to {EXPORT_CONFIG_PATH}")

    # --- Run AutoTuner button -----------------------------------------------
    if st.sidebar.button("Run AutoTuner (20 trials)"):
        autotuner_path = os.path.join(os.path.dirname(__file__), "autotuner.py")
        if os.path.exists(autotuner_path):
            subprocess.Popen([sys.executable, autotuner_path, "--trials", "20", "--quick"])
            st.sidebar.info("AutoTuner launched in background.")
        else:
            st.sidebar.error("autotuner.py not found in tools/")

    # --- Main area ----------------------------------------------------------
    st.title("LARSA Live Controls Dashboard")

    if not _ARENA_OK:
        st.error(
            f"Could not import arena_v2. The dashboard cannot run backtests.\n\n"
            f"Error: {_ARENA_IMPORT_ERROR}"
        )
        st.info("Check that all lib/ dependencies are installed and accessible.")
        return

    # Build cache key from current parameters
    cache_key = _param_hash(
        cf=cf, bh_form=bh_form, bh_decay=bh_decay, max_lev=max_lev,
        solo_cap=solo_cap, conv_cap=conv_cap, n_bars=n_bars, seed=seed,
    )

    # Run arena with current params
    with st.spinner("Running arena..."):
        result = _run_arena(
            cf=cf, bh_form=bh_form, bh_decay=bh_decay, max_lev=max_lev,
            solo_cap=solo_cap, conv_cap=conv_cap,
            n_bars=n_bars, seed=seed, _cache_key=cache_key,
        )

    if not result:
        st.error("Arena run returned no results.")
        return

    stats = result["stats"]
    equity_curve = result["equity_curve"]
    bar_log = result["bar_log"]

    # Load baseline for comparison
    baseline_cfg = _load_best_config()
    baseline_key = _param_hash(
        cf=baseline_cfg["cf"], bh_form=baseline_cfg["bh_form"],
        bh_decay=baseline_cfg["bh_decay"], max_lev=baseline_cfg["max_lev"],
        solo_cap=0.25, conv_cap=0.60,
        n_bars=n_bars, seed=seed,
    )
    with st.spinner("Running baseline..."):
        baseline_result = _run_arena(
            cf=baseline_cfg["cf"], bh_form=baseline_cfg["bh_form"],
            bh_decay=baseline_cfg["bh_decay"], max_lev=baseline_cfg["max_lev"],
            solo_cap=0.25, conv_cap=0.60,
            n_bars=n_bars, seed=seed, _cache_key=baseline_key,
        )

    baseline_stats = baseline_result.get("stats", {})

    # --- KPI strip ----------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    sharpe = stats.get("sharpe", 0.0)
    baseline_sharpe = baseline_stats.get("sharpe", 0.0)
    ret_pct = stats.get("total_return_pct", 0.0)
    max_dd = stats.get("max_drawdown_pct", 0.0)

    with col1:
        delta_sharpe = sharpe - baseline_sharpe
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe:.3f}",
            delta=f"{delta_sharpe:+.3f} vs baseline",
            delta_color="normal" if delta_sharpe >= 0 else "inverse",
        )

    with col2:
        baseline_ret = baseline_stats.get("total_return_pct", 0.0)
        delta_ret = ret_pct - baseline_ret
        st.metric(
            label="Total Return",
            value=f"{ret_pct:+.2f}%",
            delta=f"{delta_ret:+.2f}% vs baseline",
            delta_color="normal" if delta_ret >= 0 else "inverse",
        )

    with col3:
        baseline_dd = baseline_stats.get("max_drawdown_pct", 0.0)
        delta_dd = max_dd - baseline_dd
        dd_color = "inverse" if max_dd > 25 else "normal"
        st.metric(
            label="Max Drawdown",
            value=f"{max_dd:.2f}%",
            delta=f"{delta_dd:+.2f}% vs baseline",
            delta_color="inverse" if delta_dd > 0 else "normal",
        )

    st.markdown("---")

    # --- Tabs ---------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Equity Curve",
        "Regime Timeline",
        "BH Mass",
        "Position Sizing",
        "Comparison",
    ])

    with tab1:
        st.plotly_chart(_fig_equity(equity_curve, label="Strategy"), use_container_width=True)
        st.caption(
            f"Trades: {stats.get('trade_count', 0)}  |  "
            f"Win rate: {stats.get('win_rate', 0):.1%}  |  "
            f"Final equity: ${stats.get('final_equity', 0):,.0f}"
        )

    with tab2:
        if bar_log:
            st.plotly_chart(_fig_regime_timeline(bar_log), use_container_width=True)
        else:
            st.info("No bar log available. Set verbose=True in run_v2 to enable regime logging.")

    with tab3:
        if bar_log:
            st.plotly_chart(_fig_bh_mass(bar_log, bh_form_threshold=bh_form), use_container_width=True)
        else:
            st.info("No bar log available.")

    with tab4:
        st.plotly_chart(
            _fig_position_sizing(equity_curve, pid_enabled, pid_target_dd),
            use_container_width=True,
        )
        if pid_enabled and _PID_OK:
            mults = _compute_pid_series(equity_curve, pid_target_dd)
            st.caption(
                f"PID target DD: {pid_target_dd:.0%}  |  "
                f"Avg multiplier: {float(np.mean(mults)):.3f}  |  "
                f"Min: {float(np.min(mults)):.3f}  |  "
                f"Max: {float(np.max(mults)):.3f}"
            )
        elif pid_enabled and not _PID_OK:
            st.warning("PID sizer not available — pid_sizer.py could not be imported.")

    with tab5:
        baseline_curve = baseline_result.get("equity_curve", [])
        if baseline_curve:
            st.plotly_chart(
                _fig_equity_comparison(
                    equity_curve, "Current Config",
                    baseline_curve, "Baseline",
                ),
                use_container_width=True,
            )
            # Side-by-side stat table
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Current Config")
                st.json(stats)
            with c2:
                st.subheader("Baseline")
                st.json(baseline_stats)
        else:
            st.info("Baseline run produced no results.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__" or True:
    # Streamlit re-executes the whole file on each interaction;
    # the guard above is intentionally always True so main() runs.
    main()
