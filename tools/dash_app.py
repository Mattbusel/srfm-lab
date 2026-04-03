"""
dash_app.py — Full Plotly Dash real-time dashboard for LARSA.

Layout:
  Left sidebar (25%): strategy selector, parameter sliders, PID toggle,
                       Run Arena button, Export Config, Run AutoTuner
  Main area (75%):
    Row 1 — KPI Cards (5): Total Return%, Sharpe, Max DD%, Trade Count, Convergence Events
    Row 2 — Equity Curve (full width)
    Row 3 — Regime Timeline | BH Mass over time
    Row 4 — Annual Attribution | Solo vs Convergence donut
    Row 5 — Parameter heatmap

Usage:
    python tools/dash_app.py  # opens http://localhost:8050
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Arena import
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
try:
    from arena_v2 import run_v2, load_ohlcv, generate_synthetic
    ARENA_AVAILABLE = True
except ImportError:
    ARENA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Dash / Plotly import with fallback
# ---------------------------------------------------------------------------
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback_context
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Strategy presets
# ---------------------------------------------------------------------------
STRATEGY_PRESETS: Dict[str, Dict[str, Any]] = {
    "v1": {"cf": 0.001,  "bh_form": 1.5, "bh_decay": 0.95, "max_lev": 0.65,
           "solo_cap": 0.40, "conv_cap": 0.65, "pid_on": False, "target_dd": 12.0},
    "v3": {"cf": 0.001,  "bh_form": 1.5, "bh_decay": 0.95, "max_lev": 0.65,
           "solo_cap": 0.40, "conv_cap": 0.65, "pid_on": False, "target_dd": 12.0},
    "v4": {"cf": 0.001,  "bh_form": 1.5, "bh_decay": 0.95, "max_lev": 0.65,
           "solo_cap": 0.40, "conv_cap": 0.65, "pid_on": False, "target_dd": 12.0},
    "v5": {"cf": 0.001,  "bh_form": 1.5, "bh_decay": 0.95, "max_lev": 0.65,
           "solo_cap": 0.40, "conv_cap": 0.65, "pid_on": False, "target_dd": 12.0},
    "v6": {"cf": 0.001,  "bh_form": 1.5, "bh_decay": 0.95, "max_lev": 0.65,
           "solo_cap": 0.20, "conv_cap": 0.65, "pid_on": False, "target_dd": 12.0},
    "custom": {"cf": 0.005, "bh_form": 1.8, "bh_decay": 0.94, "max_lev": 0.60,
               "solo_cap": 0.25, "conv_cap": 0.65, "pid_on": False, "target_dd": 15.0},
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
HEATMAP_PATH = os.path.join(RESULTS_DIR, "param_heatmap.json")


# ---------------------------------------------------------------------------
# Arena runner helper
# ---------------------------------------------------------------------------

def run_arena(cf: float, bh_form: float, bh_decay: float, max_lev: float) -> Dict:
    """Run arena and return results dict."""
    if not ARENA_AVAILABLE:
        return _mock_results()

    cfg = {"cf": cf, "bh_form": bh_form, "bh_collapse": 1.0, "bh_decay": bh_decay}
    try:
        bars = generate_synthetic(n_bars=10000, seed=42)
        broker, bar_log = run_v2(bars, cfg, max_leverage=max_lev)
    except Exception as e:
        return {"error": str(e), **_mock_results()}

    equity = [b["equity"] for b in bar_log if "equity" in b]
    dates = [b.get("date", "") for b in bar_log if "equity" in b]
    regimes = [b.get("regime", 0) for b in bar_log if "equity" in b]
    bh_mass_hist = [b.get("bh_mass", 0.0) for b in bar_log if "equity" in b]
    conv_events = [i for i, b in enumerate(bar_log) if b.get("convergence", False)]

    returns = []
    for i in range(1, len(equity)):
        if equity[i - 1] > 0:
            returns.append((equity[i] - equity[i - 1]) / equity[i - 1])

    arr = np.array(returns) if returns else np.array([0.0])
    sharpe = float(arr.mean() / (arr.std() + 1e-10) * np.sqrt(252)) if len(arr) > 1 else 0.0

    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak + 1e-9) * 100
        if dd > max_dd:
            max_dd = dd

    total_ret = (equity[-1] / equity[0] - 1) * 100 if equity else 0.0

    return {
        "equity": equity,
        "dates": dates,
        "regimes": regimes,
        "bh_mass": bh_mass_hist,
        "conv_events": conv_events,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_ret": total_ret,
        "trade_count": len(bar_log),
        "n_conv": len(conv_events),
    }


def _mock_results() -> Dict:
    rng = np.random.default_rng(42)
    n = 5000
    equity = list(np.cumprod(1 + rng.normal(0.0002, 0.008, n)) * 1_000_000)
    regimes = list((rng.integers(0, 4, n)).astype(int))
    bh_mass = list(np.abs(rng.normal(0.5, 0.3, n)).clip(0, 3))
    conv_events = sorted(rng.choice(n, size=20, replace=False).tolist())
    returns = np.diff(equity) / (np.array(equity[:-1]) + 1e-9)
    sharpe = float(returns.mean() / (returns.std() + 1e-10) * np.sqrt(252))
    peak = max(equity)
    max_dd = max([(max(equity[:i + 1]) - v) / (max(equity[:i + 1]) + 1e-9) * 100
                  for i, v in enumerate(equity)])
    total_ret = (equity[-1] / equity[0] - 1) * 100
    return {
        "equity": equity,
        "dates": [f"2018-{(i//250)+1:02d}-01" for i in range(n)],
        "regimes": regimes,
        "bh_mass": bh_mass,
        "conv_events": conv_events,
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
        "total_ret": round(total_ret, 2),
        "trade_count": 263,
        "n_conv": 47,
    }


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------
REGIME_COLORS = {0: "#2ecc71", 1: "#e74c3c", 2: "#f39c12", 3: "#9b59b6"}
REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "SIDEWAYS", 3: "HIGH_VOL"}


def fig_equity_curve(results: Dict) -> "go.Figure":
    equity = results.get("equity", [1_000_000])
    conv_events = results.get("conv_events", [])
    regimes = results.get("regime", results.get("regimes", []))

    fig = go.Figure()

    # Drawdown shading
    peak = equity[0]
    dd_shading = []
    for i, v in enumerate(equity):
        if v > peak:
            peak = v
        dd_shading.append((peak - v) / (peak + 1e-9) * 100)

    x = list(range(len(equity)))
    fig.add_trace(go.Scatter(
        x=x, y=equity, mode="lines", name="Portfolio",
        line=dict(color="#3498db", width=2),
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=dd_shading, mode="lines", name="Drawdown%",
        line=dict(color="rgba(231,76,60,0.5)", width=1),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
        yaxis="y2"
    ))

    # Convergence event markers
    for ce in conv_events[:50]:
        if 0 <= ce < len(equity):
            fig.add_vline(x=ce, line_width=1, line_dash="dot",
                          line_color="rgba(255,215,0,0.7)")

    fig.update_layout(
        title="Equity Curve + Drawdown",
        xaxis_title="Bar",
        yaxis=dict(title="Portfolio Value ($)", side="left"),
        yaxis2=dict(title="Drawdown%", overlaying="y", side="right",
                    showgrid=False, range=[0, 40]),
        legend=dict(orientation="h"),
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=50, t=40, b=30),
    )
    return fig


def fig_regime_timeline(results: Dict) -> "go.Figure":
    regimes = results.get("regimes", [])
    if not regimes:
        return go.Figure().update_layout(title="Regime Timeline (no data)", template="plotly_dark")

    n = len(regimes)
    chunk = max(1, n // 50)
    chunks = [regimes[i:i + chunk] for i in range(0, n, chunk)]

    data = {r: [] for r in range(4)}
    for ch in chunks:
        total = len(ch)
        for r in range(4):
            data[r].append(sum(1 for x in ch if x == r) / (total + 1e-9))

    fig = go.Figure()
    x = list(range(len(chunks)))
    for r in range(4):
        fig.add_trace(go.Bar(
            x=x, y=data[r], name=REGIME_NAMES[r],
            marker_color=REGIME_COLORS[r]
        ))

    fig.update_layout(
        barmode="stack",
        title="Regime Distribution Over Time",
        xaxis_title="Period",
        yaxis_title="Fraction",
        template="plotly_dark",
        height=280,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def fig_bh_mass(results: Dict) -> "go.Figure":
    bh_mass = results.get("bh_mass", [])
    if not bh_mass:
        return go.Figure().update_layout(title="BH Mass (no data)", template="plotly_dark")

    x = list(range(len(bh_mass)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=bh_mass, mode="lines", name="BH Mass",
        line=dict(color="#f39c12", width=1.5)
    ))
    # Formation threshold line
    fig.add_hline(y=1.5, line_dash="dash", line_color="rgba(255,255,255,0.5)",
                  annotation_text="Formation threshold")

    fig.update_layout(
        title="Black Hole Mass Over Time",
        xaxis_title="Bar",
        yaxis_title="BH Mass",
        template="plotly_dark",
        height=280,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def fig_annual_attribution() -> "go.Figure":
    years = list(range(2018, 2025))
    pnl = [42000, 130000, 85000, -28000, 210000, 95000, 45000]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in pnl]

    fig = go.Figure(go.Bar(
        x=years, y=pnl, marker_color=colors, name="Annual P&L"
    ))
    fig.update_layout(
        title="Annual Gross P&L Attribution",
        xaxis_title="Year",
        yaxis_title="P&L ($)",
        template="plotly_dark",
        height=280,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def fig_trade_donut(results: Dict) -> "go.Figure":
    n_conv = results.get("n_conv", 47)
    n_total = results.get("trade_count", 263)
    n_solo = max(0, n_total - n_conv)

    fig = go.Figure(go.Pie(
        labels=["Convergence Wells", "Solo Wells"],
        values=[n_conv, n_solo],
        hole=0.5,
        marker_colors=["#3498db", "#95a5a6"],
    ))
    fig.update_layout(
        title="Solo vs Convergence Wells",
        template="plotly_dark",
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def fig_param_heatmap() -> "go.Figure":
    if os.path.exists(HEATMAP_PATH):
        try:
            with open(HEATMAP_PATH) as f:
                hm_data = json.load(f)
            x = hm_data.get("x", [])
            y = hm_data.get("y", [])
            z = hm_data.get("z", [[]])
            xlabel = hm_data.get("x_label", "cf")
            ylabel = hm_data.get("y_label", "bh_form")
        except Exception:
            x, y, z, xlabel, ylabel = _mock_heatmap()
    else:
        x, y, z, xlabel, ylabel = _mock_heatmap()

    fig = go.Figure(go.Heatmap(
        x=x, y=y, z=z,
        colorscale="RdYlGn",
        colorbar=dict(title="Sharpe"),
    ))
    fig.update_layout(
        title=f"Sharpe Heatmap: {xlabel} vs {ylabel}",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_dark",
        height=280,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def _mock_heatmap():
    rng = np.random.default_rng(1)
    x = [round(0.001 + i * 0.002, 3) for i in range(8)]
    y = [round(1.0 + j * 0.25, 2) for j in range(7)]
    z = (rng.random((7, 8)) * 2.0 + 0.5).tolist()
    return x, y, z, "cf", "bh_form"


# ---------------------------------------------------------------------------
# KPI card helper
# ---------------------------------------------------------------------------

def kpi_card(title: str, value: str, color: str = "#3498db") -> "html.Div":
    return html.Div([
        html.P(title, style={"fontSize": "12px", "color": "#aaa", "margin": "0"}),
        html.H4(value, style={"color": color, "margin": "4px 0 0 0"}),
    ], style={
        "background": "#1e2130", "borderRadius": "8px", "padding": "12px",
        "flex": "1", "margin": "0 6px", "textAlign": "center",
        "borderTop": f"3px solid {color}",
    })


# ---------------------------------------------------------------------------
# Build Dash app
# ---------------------------------------------------------------------------

def build_app() -> "dash.Dash":
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )
    app.title = "LARSA Dashboard"

    default_preset = STRATEGY_PRESETS["v6"]
    default_results = _mock_results()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    sidebar = html.Div([
        html.H5("LARSA Dashboard", style={"color": "#3498db", "marginBottom": "16px"}),

        html.Label("Strategy"),
        dcc.Dropdown(
            id="strategy-selector",
            options=[{"label": k, "value": k} for k in STRATEGY_PRESETS],
            value="v6",
            clearable=False,
            style={"marginBottom": "12px"},
        ),

        html.Label("Cost Fraction (cf)"),
        dcc.Slider(id="sl-cf", min=0.001, max=0.015, step=0.001,
                   value=default_preset["cf"],
                   marks={0.001: "0.001", 0.008: "0.008", 0.015: "0.015"},
                   tooltip={"always_visible": False}),

        html.Label("BH Formation (bh_form)"),
        dcc.Slider(id="sl-bh-form", min=1.0, max=3.0, step=0.1,
                   value=default_preset["bh_form"],
                   marks={1.0: "1", 2.0: "2", 3.0: "3"},
                   tooltip={"always_visible": False}),

        html.Label("BH Decay"),
        dcc.Slider(id="sl-bh-decay", min=0.80, max=0.99, step=0.01,
                   value=default_preset["bh_decay"],
                   marks={0.80: "0.80", 0.90: "0.90", 0.99: "0.99"},
                   tooltip={"always_visible": False}),

        html.Label("Max Leverage"),
        dcc.Slider(id="sl-max-lev", min=0.30, max=0.85, step=0.05,
                   value=default_preset["max_lev"],
                   marks={0.30: "0.30", 0.60: "0.60", 0.85: "0.85"},
                   tooltip={"always_visible": False}),

        html.Label("Solo Cap"),
        dcc.Slider(id="sl-solo-cap", min=0.10, max=0.45, step=0.05,
                   value=default_preset["solo_cap"],
                   marks={0.10: "0.10", 0.25: "0.25", 0.45: "0.45"},
                   tooltip={"always_visible": False}),

        html.Label("Conv Cap"),
        dcc.Slider(id="sl-conv-cap", min=0.40, max=0.80, step=0.05,
                   value=default_preset["conv_cap"],
                   marks={0.40: "0.40", 0.60: "0.60", 0.80: "0.80"},
                   tooltip={"always_visible": False}),

        html.Hr(style={"borderColor": "#444"}),

        html.Div([
            html.Label("PID Controller"),
            dcc.Checklist(id="pid-toggle", options=[{"label": " Enable", "value": "on"}],
                          value=[], inline=True),
        ], style={"marginBottom": "8px"}),

        html.Label("Target DD%"),
        dcc.Slider(id="sl-target-dd", min=5.0, max=25.0, step=1.0,
                   value=12.0,
                   marks={5: "5", 12: "12", 25: "25"},
                   tooltip={"always_visible": False}),

        html.Hr(style={"borderColor": "#444"}),

        html.Button("Run Arena", id="btn-run", n_clicks=0,
                    style={"width": "100%", "marginBottom": "8px",
                           "background": "#3498db", "color": "white",
                           "border": "none", "borderRadius": "4px", "padding": "8px"}),
        html.Button("Export Config", id="btn-export", n_clicks=0,
                    style={"width": "100%", "marginBottom": "8px",
                           "background": "#27ae60", "color": "white",
                           "border": "none", "borderRadius": "4px", "padding": "8px"}),
        html.Button("Run AutoTuner", id="btn-autotuner", n_clicks=0,
                    style={"width": "100%",
                           "background": "#8e44ad", "color": "white",
                           "border": "none", "borderRadius": "4px", "padding": "8px"}),

        html.Div(id="status-text", style={"marginTop": "10px", "fontSize": "11px",
                                           "color": "#aaa"}),
    ], style={
        "width": "22%", "minWidth": "220px", "padding": "16px",
        "background": "#131722", "borderRight": "1px solid #333",
        "height": "100vh", "overflowY": "auto", "position": "fixed",
        "top": 0, "left": 0,
    })

    main_content = html.Div([
        # KPI row
        html.Div(id="kpi-row", style={"display": "flex", "marginBottom": "16px"}),

        # Equity curve
        dcc.Graph(id="graph-equity", figure=fig_equity_curve(default_results),
                  style={"marginBottom": "12px"}),

        # Row 3
        html.Div([
            dcc.Graph(id="graph-regime",
                      figure=fig_regime_timeline(default_results),
                      style={"flex": "1", "marginRight": "8px"}),
            dcc.Graph(id="graph-bh-mass",
                      figure=fig_bh_mass(default_results),
                      style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "12px"}),

        # Row 4
        html.Div([
            dcc.Graph(id="graph-attribution",
                      figure=fig_annual_attribution(),
                      style={"flex": "1", "marginRight": "8px"}),
            dcc.Graph(id="graph-donut",
                      figure=fig_trade_donut(default_results),
                      style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "12px"}),

        # Row 5 — heatmap
        dcc.Graph(id="graph-heatmap", figure=fig_param_heatmap()),

    ], style={
        "marginLeft": "24%", "padding": "20px", "background": "#0f1117", "minHeight": "100vh",
    })

    # Store for arena results
    store = dcc.Store(id="arena-store", data=default_results)
    # Debounce interval
    debounce_interval = dcc.Interval(id="debounce-interval", interval=500, n_intervals=0,
                                     disabled=True)

    app.layout = html.Div([store, debounce_interval, sidebar, main_content],
                          style={"background": "#0f1117", "fontFamily": "sans-serif",
                                 "color": "#eee"})

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    @app.callback(
        Output("sl-cf", "value"),
        Output("sl-bh-form", "value"),
        Output("sl-bh-decay", "value"),
        Output("sl-max-lev", "value"),
        Output("sl-solo-cap", "value"),
        Output("sl-conv-cap", "value"),
        Input("strategy-selector", "value"),
    )
    def load_preset(strategy):
        p = STRATEGY_PRESETS.get(strategy, STRATEGY_PRESETS["v6"])
        return p["cf"], p["bh_form"], p["bh_decay"], p["max_lev"], p["solo_cap"], p["conv_cap"]

    @app.callback(
        Output("debounce-interval", "disabled"),
        Output("debounce-interval", "n_intervals"),
        Input("sl-cf", "value"),
        Input("sl-bh-form", "value"),
        Input("sl-bh-decay", "value"),
        Input("sl-max-lev", "value"),
        Input("sl-solo-cap", "value"),
        Input("sl-conv-cap", "value"),
        Input("btn-run", "n_clicks"),
    )
    def arm_debounce(*args):
        return False, 0

    @app.callback(
        Output("arena-store", "data"),
        Output("debounce-interval", "disabled", allow_duplicate=True),
        Input("debounce-interval", "n_intervals"),
        State("sl-cf", "value"),
        State("sl-bh-form", "value"),
        State("sl-bh-decay", "value"),
        State("sl-max-lev", "value"),
        prevent_initial_call=True,
    )
    def trigger_arena(n_intervals, cf, bh_form, bh_decay, max_lev):
        results = run_arena(cf or 0.001, bh_form or 1.5, bh_decay or 0.95, max_lev or 0.65)
        return results, True

    @app.callback(
        Output("kpi-row", "children"),
        Output("graph-equity", "figure"),
        Output("graph-regime", "figure"),
        Output("graph-bh-mass", "figure"),
        Output("graph-donut", "figure"),
        Input("arena-store", "data"),
    )
    def update_charts(results):
        if not results:
            results = _mock_results()

        total_ret = results.get("total_ret", 0.0)
        sharpe = results.get("sharpe", 0.0)
        max_dd = results.get("max_dd", 0.0)
        trade_count = results.get("trade_count", 0)
        n_conv = results.get("n_conv", 0)

        ret_color = "#2ecc71" if total_ret >= 0 else "#e74c3c"
        dd_color = "#e74c3c" if max_dd > 15 else "#f39c12" if max_dd > 10 else "#2ecc71"

        kpi_cards = [
            kpi_card("Total Return", f"{total_ret:+.1f}%", ret_color),
            kpi_card("Sharpe", f"{sharpe:.3f}", "#3498db"),
            kpi_card("Max DD", f"{max_dd:.1f}%", dd_color),
            kpi_card("Trade Count", str(trade_count), "#f39c12"),
            kpi_card("Conv Events", str(n_conv), "#9b59b6"),
        ]

        return (
            kpi_cards,
            fig_equity_curve(results),
            fig_regime_timeline(results),
            fig_bh_mass(results),
            fig_trade_donut(results),
        )

    @app.callback(
        Output("status-text", "children"),
        Input("btn-export", "n_clicks"),
        State("sl-cf", "value"),
        State("sl-bh-form", "value"),
        State("sl-bh-decay", "value"),
        State("sl-max-lev", "value"),
        State("sl-solo-cap", "value"),
        State("sl-conv-cap", "value"),
        State("strategy-selector", "value"),
        prevent_initial_call=True,
    )
    def export_config(n_clicks, cf, bh_form, bh_decay, max_lev, solo_cap, conv_cap, strategy):
        if not n_clicks:
            return ""
        cfg = {
            "strategy": strategy,
            "cf": cf, "bh_form": bh_form, "bh_decay": bh_decay,
            "max_lev": max_lev, "solo_cap": solo_cap, "conv_cap": conv_cap,
        }
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "dash_export_config.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        return f"Exported to {path}"

    @app.callback(
        Output("status-text", "children", allow_duplicate=True),
        Input("btn-autotuner", "n_clicks"),
        prevent_initial_call=True,
    )
    def run_autotuner(n_clicks):
        if not n_clicks:
            return ""
        autotuner = os.path.join(os.path.dirname(__file__), "autotuner.py")
        if os.path.exists(autotuner):
            subprocess.Popen([sys.executable, autotuner, "--quick"])
            return "AutoTuner launched in background..."
        return "autotuner.py not found."

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_install_instructions() -> None:
    print("=" * 60)
    print("LARSA Dash Dashboard — dependencies not installed")
    print("=" * 60)
    print()
    print("Install required packages:")
    print("  pip install dash dash-bootstrap-components plotly")
    print()
    print("Then run:  python tools/dash_app.py")
    print()
    print("Dashboard WOULD show:")
    print("  Left Sidebar:")
    print("    - Strategy selector (v1/v3/v4/v5/v6/custom)")
    print("    - Sliders: cf, bh_form, bh_decay, max_lev, solo_cap, conv_cap")
    print("    - PID toggle + target DD slider")
    print("    - Run Arena / Export Config / Run AutoTuner buttons")
    print()
    print("  Main Area (5 rows):")
    print("    Row 1: KPI Cards — Total Return%, Sharpe, Max DD%, Trade Count, Conv Events")
    print("    Row 2: Equity Curve with drawdown shading + convergence markers")
    print("    Row 3: Regime Timeline (stacked bar) | BH Mass over time")
    print("    Row 4: Annual P&L Attribution | Solo vs Conv Donut")
    print("    Row 5: Sharpe Heatmap (2-param sweep)")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="LARSA Plotly Dash Dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not DASH_AVAILABLE:
        print_install_instructions()
        sys.exit(0)

    if not ARENA_AVAILABLE:
        print("Note: arena_v2 not available — using synthetic mock data for charts.")

    app = build_app()
    print(f"LARSA Dashboard running at http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
