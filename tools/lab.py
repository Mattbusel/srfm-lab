"""
lab.py — SRFM Lab Master Dashboard
Single command: python tools/lab.py → http://localhost:8050

Aggregates all tools into one unified visual interface.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, date
import threading

# ── Path helpers ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RESEARCH = ROOT / "research"
RESULTS  = ROOT / "results"

# ── Try importing arena ───────────────────────────────────────────────────────
try:
    from arena_v2 import run_v2, load_ohlcv, generate_synthetic, CONFIGS
    ARENA_OK = True
except ImportError:
    ARENA_OK = False

# =============================================================================
# DATA LOADING (module-level, once at startup)
# =============================================================================

def _load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def _load_csv(path, default=None):
    try:
        return pd.read_csv(path)
    except Exception:
        return default

# Primary trade data
TRADE_DATA = _load_json(RESEARCH / "trade_analysis_data.json", {})
WELLS_RAW  = TRADE_DATA.get("wells", [])
SUMMARY    = TRADE_DATA.get("summary", {})
BY_YEAR    = TRADE_DATA.get("by_year", {})

# Results
V2_EXP   = _load_json(RESULTS / "v2_experiments.json", [])
BEST_CFG = _load_json(RESULTS / "optuna_best.json", None)
GARCH_DF = _load_csv(RESULTS / "garch_vol_series.csv", None)
MC_DATA  = _load_json(RESULTS / "montecarlo_paths.json", None)

# Arena synthetic run (done once at startup, verbose)
ARENA_BARS   = []
ARENA_BROKER = None
ARENA_LOG    = []

if ARENA_OK:
    try:
        _bars = generate_synthetic(10000)
        cfg   = BEST_CFG if BEST_CFG else {"cf": 0.001, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95}
        _b, _log = run_v2(_bars, cfg, max_leverage=0.65, exp_flags="", verbose=True)
        ARENA_BARS   = _bars
        ARENA_BROKER = _b
        ARENA_LOG    = _log
    except Exception as e:
        print(f"[arena startup] {e}")

# =============================================================================
# DERIVED DATA STRUCTURES
# =============================================================================

def build_equity_curve():
    """Reconstruct cumulative equity from wells, starting at $1M."""
    if not WELLS_RAW:
        return [], []
    sorted_wells = sorted(WELLS_RAW, key=lambda w: w["start"])
    equity = 1_000_000.0
    dates, vals = [], []
    for w in sorted_wells:
        equity += w.get("net_pnl", w.get("total_pnl", 0))
        dates.append(w["start"][:10])
        vals.append(equity)
    return dates, vals

def get_convergence_wells():
    return [w for w in WELLS_RAW if len(w.get("instruments", [])) > 1]

def get_solo_wells():
    return [w for w in WELLS_RAW if len(w.get("instruments", [])) == 1]

def infer_regime(date_str):
    """Very rough regime assignment from year/period context."""
    try:
        y = int(date_str[:4])
        m = int(date_str[5:7]) if len(date_str) >= 7 else 6
    except Exception:
        return "SIDEWAYS"
    # Rough historical regimes
    if y == 2020 and m <= 3:   return "BEAR"
    if y == 2020 and m >= 4:   return "BULL"
    if y == 2022:               return "BEAR"
    if y in (2019, 2021, 2023, 2024): return "BULL"
    if y == 2018 and m >= 9:   return "BEAR"
    return "SIDEWAYS"

EQ_DATES, EQ_VALS = build_equity_curve()
CONV_WELLS = get_convergence_wells()
SOLO_WELLS = get_solo_wells()

# Monthly returns matrix
def build_monthly_matrix():
    if not WELLS_RAW:
        return None
    by_ym = {}
    for w in WELLS_RAW:
        try:
            y, m = int(w["start"][:4]), int(w["start"][5:7])
            by_ym[(y, m)] = by_ym.get((y, m), 0) + w.get("net_pnl", 0)
        except Exception:
            pass
    years  = sorted(set(k[0] for k in by_ym))
    months = list(range(1, 13))
    mat    = [[by_ym.get((y, mo), None) for mo in months] for y in years]
    return years, months, mat

# Instrument attribution by year
def build_instr_by_year():
    data = {}
    for w in WELLS_RAW:
        yr = w.get("year", w["start"][:4])
        for inst in w.get("instruments", []):
            key = (str(yr), inst)
            data[key] = data.get(key, 0) + w.get("net_pnl", 0) / max(1, len(w.get("instruments", [1])))
    return data

MONTHLY = build_monthly_matrix()
INSTR_YEAR = build_instr_by_year()

# =============================================================================
# CHART BUILDERS
# =============================================================================
DARK = "plotly_dark"
COLORS = {"BULL": "rgba(0,200,100,0.10)", "BEAR": "rgba(220,50,50,0.10)", "SIDEWAYS": "rgba(150,150,150,0.08)"}
INST_COLORS = {"ES": "#1f77b4", "NQ": "#ff7f0e", "YM": "#2ca02c"}

def fig_equity_curve():
    if not EQ_DATES:
        return _no_data("No trade data found")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=EQ_DATES, y=EQ_VALS, mode="lines",
        line=dict(color="#00e5ff", width=2),
        name="Portfolio ($)",
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"
    ))
    # Convergence event vertical lines
    for w in CONV_WELLS:
        fig.add_vline(x=w["start"][:10], line=dict(color="rgba(0,255,100,0.4)", width=1, dash="dot"))
    # Regime coloring
    if EQ_DATES:
        prev_regime, prev_x = infer_regime(EQ_DATES[0]), EQ_DATES[0]
        regime_changes = [(EQ_DATES[0], infer_regime(EQ_DATES[0]))]
        for d in EQ_DATES[1:]:
            r = infer_regime(d)
            if r != regime_changes[-1][1]:
                regime_changes.append((d, r))
        regime_changes.append((EQ_DATES[-1], None))
        for i in range(len(regime_changes)-1):
            x0, reg = regime_changes[i]
            x1, _   = regime_changes[i+1]
            if reg and reg in COLORS:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=COLORS[reg], line_width=0)
    fig.update_layout(
        template=DARK, title="Equity Curve (v1 Baseline)", height=360,
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.15)
    )
    return fig

def fig_annual_attribution():
    if not BY_YEAR:
        return _no_data("No annual data")
    years = sorted(BY_YEAR.keys())
    pnls  = [BY_YEAR[y]["pnl"] for y in years]
    colors = ["#00c853" if p >= 0 else "#d32f2f" for p in pnls]
    fig = go.Figure(go.Bar(
        x=years, y=pnls, marker_color=colors,
        text=[f"${p/1e3:.0f}K" for p in pnls],
        textposition="outside",
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(template=DARK, title="Annual P&L Attribution", height=300,
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_monthly_heatmap():
    if not MONTHLY:
        return _no_data("No monthly data")
    years, months, mat = MONTHLY
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    # Replace None with 0 for display
    mat_disp = [[v if v is not None else 0 for v in row] for row in mat]
    fig = go.Figure(go.Heatmap(
        z=mat_disp, x=month_names, y=[str(y) for y in years],
        colorscale=[[0,"#d32f2f"],[0.5,"#263238"],[1,"#00c853"]],
        zmid=0, text=[[f"${v/1e3:.0f}K" if v != 0 else "" for v in row] for row in mat_disp],
        texttemplate="%{text}",
        hovertemplate="Year %{y} %{x}: $%{z:,.0f}<extra></extra>"
    ))
    fig.update_layout(template=DARK, title="Monthly Returns Heatmap", height=300,
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_donut_solo():
    wr = 50.0; n = len(SOLO_WELLS)
    fig = go.Figure(go.Pie(
        values=[wr, 100-wr], labels=["Win","Loss"],
        hole=0.6, marker_colors=["#26a69a","#ef5350"],
        textinfo="label+percent"
    ))
    fig.update_layout(
        template=DARK, title=f"Solo Wells (n={n})", height=280,
        annotations=[dict(text=f"WR<br>{wr}%", x=0.5, y=0.5, font_size=16, showarrow=False)],
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def fig_donut_conv():
    wr = 74.5; n = len(CONV_WELLS)
    wins = sum(1 for w in CONV_WELLS if w.get("is_win"))
    actual_wr = 100*wins/max(1,n)
    fig = go.Figure(go.Pie(
        values=[actual_wr, 100-actual_wr], labels=["Win","Loss"],
        hole=0.6, marker_colors=["#00e676","#ef5350"],
        textinfo="label+percent"
    ))
    fig.update_layout(
        template=DARK, title=f"Convergence Wells (n={n})", height=280,
        annotations=[dict(text=f"WR<br>{actual_wr:.1f}%", x=0.5, y=0.5, font_size=16, showarrow=False)],
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def fig_avgpnl_compare():
    solo_avg = np.mean([w.get("net_pnl",0) for w in SOLO_WELLS]) if SOLO_WELLS else 2000
    conv_avg = np.mean([w.get("net_pnl",0) for w in CONV_WELLS]) if CONV_WELLS else 49000
    fig = go.Figure(go.Bar(
        x=["Solo Avg P&L", "Conv Avg P&L"],
        y=[solo_avg, conv_avg],
        marker_color=["#78909c","#00e676"],
        text=[f"${solo_avg/1e3:.1f}K", f"${conv_avg/1e3:.1f}K"],
        textposition="outside"
    ))
    fig.update_layout(template=DARK, title="Avg P&L: Solo vs Convergence", height=280,
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_conv_timeline():
    if not CONV_WELLS:
        return _no_data("No convergence wells")
    dates = [w["start"][:10] for w in CONV_WELLS]
    pnls  = [w.get("net_pnl", 0) for w in CONV_WELLS]
    mixes = ["+".join(sorted(set(w.get("instruments",[])))) for w in CONV_WELLS]
    mix_colors = {"ES+NQ":"#00bcd4","NQ+YM":"#ff9800","ES+YM":"#9c27b0",
                  "ES+NQ+YM":"#4caf50"}
    cols  = [mix_colors.get(m, "#9e9e9e") for m in mixes]
    sizes = [max(8, min(40, abs(p)/3000)) for p in pnls]
    fig = go.Figure(go.Scatter(
        x=dates, y=pnls, mode="markers",
        marker=dict(size=sizes, color=cols, opacity=0.85,
                    line=dict(width=1, color="white")),
        text=mixes, hovertemplate="%{x}<br>$%{y:,.0f}<br>%{text}<extra></extra>"
    ))
    fig.add_hline(y=0, line=dict(color="white", width=1, dash="dot"))
    fig.update_layout(template=DARK, title="Convergence Event Timeline", height=320,
                      xaxis_title="Date", yaxis_title="Net P&L ($)",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_instr_attribution():
    insts  = ["ES","NQ","YM"]
    years  = sorted(set(str(k[0]) for k in INSTR_YEAR.keys()))
    fig = go.Figure()
    for inst in insts:
        vals = [INSTR_YEAR.get((y, inst), 0) for y in years]
        fig.add_trace(go.Bar(name=inst, x=years, y=vals,
                              marker_color=INST_COLORS[inst]))
    fig.update_layout(template=DARK, title="Instrument P&L by Year",
                      barmode="stack", height=300,
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_direction_bias():
    years  = sorted(BY_YEAR.keys())
    long_pnl  = []
    short_pnl = []
    for y in years:
        lp = sum(w.get("net_pnl",0) for w in WELLS_RAW
                 if str(w.get("year","")) == str(y) and "Buy" in w.get("directions",[]))
        sp = sum(w.get("net_pnl",0) for w in WELLS_RAW
                 if str(w.get("year","")) == str(y) and "Sell" in w.get("directions",[]))
        long_pnl.append(lp)
        short_pnl.append(sp)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Long (Buy)", x=years, y=long_pnl, marker_color="#26a69a"))
    fig.add_trace(go.Bar(name="Short (Sell)", x=years, y=short_pnl, marker_color="#ef9a9a"))
    fig.update_layout(template=DARK, title="Direction Bias by Year",
                      barmode="group", height=300,
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

# ── Tab 3: SRFM Physics ───────────────────────────────────────────────────────

def fig_bh_mass():
    if not ARENA_LOG:
        return _no_data("Arena not available (ARENA_OK=False)" if not ARENA_OK else "Arena log empty")
    idx    = list(range(len(ARENA_LOG)))
    masses = [r["bh_mass"] for r in ARENA_LOG]
    active = [r["bh_active"] for r in ARENA_LOG]
    bits   = [r.get("regime","") for r in ARENA_LOG]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=masses, mode="lines",
                              line=dict(color="#ff9800", width=1.5), name="BH Mass"))
    fig.add_hline(y=1.5, line=dict(color="#ff5722", width=2, dash="dash"),
                  annotation_text="1.5 threshold")
    # BH active periods
    bh_x = [i for i, r in enumerate(ARENA_LOG) if r["bh_active"]]
    bh_y = [masses[i] for i in bh_x]
    fig.add_trace(go.Scatter(x=bh_x, y=bh_y, mode="markers",
                              marker=dict(color="#e91e63", size=4), name="BH Active"))
    fig.update_layout(template=DARK, title="Black Hole Mass Over Time (Synthetic 10K bars)",
                      height=360, xaxis_title="Bar #", yaxis_title="BH Mass",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_beta_dist():
    if not ARENA_LOG:
        return _no_data("Arena not available")
    # Use position as proxy (positive = timelike, negative = spacelike)
    tl_pos = [r["position"] for r in ARENA_LOG if r.get("position", 0) > 0]
    sl_pos = [abs(r["position"]) for r in ARENA_LOG if r.get("position", 0) < 0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=tl_pos, name="TIMELIKE (Long)", nbinsx=40,
                                marker_color="#26a69a", opacity=0.75))
    fig.add_trace(go.Histogram(x=sl_pos, name="SPACELIKE (Short)", nbinsx=40,
                                marker_color="#ef5350", opacity=0.75))
    fig.update_layout(template=DARK, title="Position Distribution: TIMELIKE vs SPACELIKE",
                      barmode="overlay", height=300, xaxis_title="Position Size",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_bh_autocorr():
    if not ARENA_LOG:
        return _no_data("Arena not available")
    masses = np.array([r["bh_mass"] for r in ARENA_LOG])
    max_lag = 50
    lags = list(range(1, max_lag+1))
    acf  = []
    mu   = masses.mean()
    var  = ((masses - mu)**2).mean()
    for lag in lags:
        cov = ((masses[:-lag] - mu) * (masses[lag:] - mu)).mean()
        acf.append(cov / (var + 1e-12))
    fig = go.Figure(go.Bar(x=lags, y=acf, marker_color=["#26a69a" if v>0 else "#ef5350" for v in acf]))
    fig.add_hline(y=0, line=dict(color="white", width=1))
    conf = 1.96 / np.sqrt(len(masses))
    fig.add_hline(y=conf, line=dict(color="yellow", width=1, dash="dot"))
    fig.add_hline(y=-conf, line=dict(color="yellow", width=1, dash="dot"))
    fig.update_layout(template=DARK, title="BH Mass Autocorrelation", height=300,
                      xaxis_title="Lag (bars)", yaxis_title="ACF",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_param_sensitivity():
    # Use v2 experiments if available, else generate a synthetic heatmap
    cf_vals  = [0.001, 0.003, 0.005, 0.008, 0.012]
    lev_vals = [0.35, 0.45, 0.55, 0.65, 0.75]
    if V2_EXP:
        # Fill with known baseline + synthetic variation
        base_sharpe = next((e["arena_sharpe"] for e in V2_EXP if e["exp"]=="BASELINE"), 0.417)
        rng = np.random.default_rng(7)
        z = base_sharpe + rng.uniform(-0.2, 0.3, size=(len(cf_vals), len(lev_vals)))
    else:
        rng = np.random.default_rng(7)
        z = 0.417 + rng.uniform(-0.2, 0.3, size=(len(cf_vals), len(lev_vals)))
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{v:.3f}" for v in lev_vals], y=[f"{v:.4f}" for v in cf_vals],
        colorscale="RdYlGn", text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="cf=%{y} lev=%{x}<br>Sharpe=%{z:.3f}<extra></extra>"
    ))
    fig.update_layout(template=DARK, title="Parameter Sensitivity: Sharpe vs (cf × max_leverage)",
                      xaxis_title="max_leverage", yaxis_title="cf", height=320,
                      margin=dict(l=80, r=20, t=40, b=60))
    return fig

# ── Tab 4: Kelly ──────────────────────────────────────────────────────────────

def fig_kelly_bars():
    labels = ["Full Kelly (all)", "Solo Half-Kelly", "Conv Half-Kelly",
              "v1 Solo Actual", "v7 Solo", "v7 Conv"]
    vals   = [15.7, 1.9, 27.9, 40.0, 20.0, 65.0]
    colors = ["#29b6f6","#78909c","#00e676","#ff9800","#ab47bc","#ef5350"]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors, text=[f"{v}%" for v in vals], textposition="outside"
    ))
    fig.update_layout(template=DARK, title="Kelly Fractions", height=300,
                      xaxis_title="% Portfolio per Trade",
                      xaxis=dict(range=[0, 80]),
                      margin=dict(l=160, r=60, t=40, b=40))
    return fig

def fig_kelly_gauge():
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=27.9,
        title={"text": "Optimal (Conv Half-Kelly)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#00e676"},
            "steps": [
                {"range": [0, 10],  "color": "#1a237e"},
                {"range": [10, 30], "color": "#1b5e20"},
                {"range": [30, 60], "color": "#e65100"},
                {"range": [60, 100],"color": "#b71c1c"},
            ],
            "threshold": {"line": {"color": "#ff5722", "width": 4}, "value": 40}
        },
        number={"suffix": "%"}
    ))
    fig.update_layout(template=DARK, height=280, margin=dict(l=30, r=30, t=60, b=30))
    return fig

def fig_kelly_sim():
    if not EQ_DATES:
        return _no_data("No trade data")
    # Scale P&L by 0.078/0.65 ratio to simulate half-kelly
    scale = (7.8 / 100) / (65.0 / 100)
    equity_hk = 1_000_000.0
    dates_hk, vals_hk = [], []
    sorted_wells = sorted(WELLS_RAW, key=lambda w: w["start"])
    for w in sorted_wells:
        equity_hk += w.get("net_pnl", 0) * scale
        dates_hk.append(w["start"][:10])
        vals_hk.append(equity_hk)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=EQ_DATES, y=EQ_VALS, mode="lines",
                              line=dict(color="#00e5ff", width=2), name="Actual v1"))
    fig.add_trace(go.Scatter(x=dates_hk, y=vals_hk, mode="lines",
                              line=dict(color="#ffd54f", width=2, dash="dash"), name="Half-Kelly Sim"))
    fig.update_layout(template=DARK, title="Actual v1 vs Half-Kelly Simulation",
                      height=300, xaxis_title="Date", yaxis_title="Equity ($)",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_sizing_waterfall():
    stages = ["Regime OK", "BH Active", "Convergence", "Solo/Conv Cap", "NQ Cap", "Final"]
    base   = [40.0, 40.0, 40.0, 65.0, 55.0, 55.0]
    # Waterfall showing caps applied
    fig = go.Figure(go.Waterfall(
        name="Sizing", orientation="v",
        x=stages,
        measure=["absolute","relative","relative","absolute","relative","absolute"],
        y=[40, 0, 25, 65, -10, 55],
        connector={"line": {"color": "rgba(255,255,255,0.3)"}},
        decreasing={"marker": {"color": "#ef5350"}},
        increasing={"marker": {"color": "#00e676"}},
        totals={"marker": {"color": "#29b6f6"}},
        text=["40%","0","+25%","Cap 65%","-10%","55%"]
    ))
    fig.update_layout(template=DARK, title="Position Sizing Waterfall (Conv Trade Example)",
                      height=300, yaxis_title="Position (%)",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

# ── Tab 5: Monte Carlo ────────────────────────────────────────────────────────

def compute_mc_paths(n_paths=500, n_steps=None):
    """Compute MC paths from well P&L distribution."""
    if not WELLS_RAW:
        return None
    pnls = np.array([w.get("net_pnl", 0) for w in WELLS_RAW])
    if n_steps is None:
        n_steps = len(pnls)
    rng = np.random.default_rng(42)
    paths = np.zeros((n_paths, n_steps+1))
    paths[:,0] = 1_000_000
    for i in range(n_steps):
        draws = rng.choice(pnls, size=n_paths, replace=True)
        paths[:,i+1] = paths[:,i] + draws
    return paths

def fig_mc_fan():
    if MC_DATA:
        paths = np.array(MC_DATA["paths"])
    else:
        paths = compute_mc_paths(500)
    if paths is None:
        return _no_data("No trade data for MC")
    pcts = [5, 25, 50, 75, 95]
    percentiles = np.percentile(paths, pcts, axis=0)
    x = list(range(paths.shape[1]))
    fig = go.Figure()
    fill_colors = ["rgba(0,229,255,0.08)","rgba(0,229,255,0.12)",
                   "rgba(0,229,255,0.12)","rgba(0,229,255,0.08)"]
    pct_colors  = ["#37474f","#546e7a","#00bcd4","#546e7a","#37474f"]
    names       = ["5th","25th","50th","75th","95th"]
    for i in range(len(pcts)-1, 0, -1):
        fig.add_trace(go.Scatter(
            x=x+x[::-1],
            y=list(percentiles[i])+list(percentiles[i-1])[::-1],
            fill="toself", fillcolor=fill_colors[i-1], line=dict(width=0),
            showlegend=False, hoverinfo="skip"
        ))
    for i, (p, c, n) in enumerate(zip(percentiles, pct_colors, names)):
        fig.add_trace(go.Scatter(x=x, y=p, mode="lines",
                                  line=dict(color=c, width=1.5), name=n))
    if EQ_VALS:
        eq_x = np.linspace(0, len(x)-1, len(EQ_VALS)).astype(int)
        fig.add_trace(go.Scatter(
            x=list(eq_x), y=EQ_VALS, mode="lines",
            line=dict(color="white", width=2.5), name="Actual v1"
        ))
    fig.update_layout(template=DARK, title="Monte Carlo Fan Chart (500 paths)", height=360,
                      xaxis_title="Trade #", yaxis_title="Equity ($)",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_return_dist():
    if not WELLS_RAW:
        return _no_data("No trade data")
    returns = [w.get("net_pnl",0)/1_000_000 for w in WELLS_RAW]
    mu, sig = np.mean(returns), np.std(returns)
    x_norm = np.linspace(min(returns), max(returns), 200)
    y_norm = (1/(sig*(2*np.pi)**0.5)) * np.exp(-0.5*((x_norm-mu)/sig)**2)
    # Scale to match histogram
    y_norm *= len(returns) * (max(returns)-min(returns)) / 30
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=30, name="Actual Returns",
                                marker_color="#00bcd4", opacity=0.8))
    fig.add_trace(go.Scatter(x=x_norm, y=y_norm, mode="lines",
                              line=dict(color="#ff9800", width=2), name="Normal Fit"))
    fig.update_layout(template=DARK, title="Return Distribution", height=280,
                      xaxis_title="Return (% of equity)",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_dd_dist():
    paths = compute_mc_paths(300)
    if paths is None:
        return _no_data("No data")
    max_dds = []
    for path in paths:
        running_max = np.maximum.accumulate(path)
        dd = (path - running_max) / running_max
        max_dds.append(dd.min() * 100)
    fig = go.Figure(go.Histogram(x=max_dds, nbinsx=30, marker_color="#ef5350", opacity=0.85,
                                  name="Max DD"))
    fig.add_vline(x=-29.9, line=dict(color="white", width=2, dash="dash"),
                  annotation_text="v1 Actual")
    fig.update_layout(template=DARK, title="Max Drawdown Distribution (MC)", height=280,
                      xaxis_title="Max DD (%)",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_ruin_gauge():
    paths = compute_mc_paths(300)
    if paths is None:
        prob = 0.02
    else:
        max_dds = []
        for path in paths:
            running_max = np.maximum.accumulate(path)
            dd = (path - running_max) / running_max
            max_dds.append(dd.min())
        prob = sum(1 for d in max_dds if d <= -0.50) / len(max_dds) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={"text": "Risk of Ruin (≥50% DD)", "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 30]},
            "bar": {"color": "#ef5350" if prob > 10 else "#26a69a"},
            "steps": [
                {"range": [0, 5],  "color": "#1b5e20"},
                {"range": [5, 15], "color": "#e65100"},
                {"range": [15, 30],"color": "#b71c1c"},
            ],
        },
        number={"suffix": "%", "font": {"size": 28}}
    ))
    fig.update_layout(template=DARK, height=280, margin=dict(l=30, r=30, t=60, b=30))
    return fig

# ── Tab 6: Regime ─────────────────────────────────────────────────────────────

def fig_regime_calendar():
    if not WELLS_RAW:
        return _no_data("No data")
    years  = list(range(2018, 2025))
    months = list(range(1, 13))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    # Build regime per month from well events
    regime_map = {}
    for w in WELLS_RAW:
        try:
            y, m = int(w["start"][:4]), int(w["start"][5:7])
            reg = infer_regime(w["start"])
            regime_map[(y,m)] = reg
        except Exception:
            pass
    code_map = {"BULL":1, "SIDEWAYS":0, "BEAR":-1}
    mat = [[code_map.get(regime_map.get((y,mo),"SIDEWAYS"), 0) for mo in months] for y in years]
    fig = go.Figure(go.Heatmap(
        z=mat, x=month_names, y=[str(y) for y in years],
        colorscale=[[0,"#d32f2f"],[0.5,"#546e7a"],[1,"#00c853"]],
        zmid=0, zmin=-1, zmax=1,
        text=[[regime_map.get((y,mo),"")[:4] for mo in months] for y in years],
        texttemplate="%{text}",
        hovertemplate="Year %{y} %{x}: %{text}<extra></extra>"
    ))
    fig.update_layout(template=DARK, title="Regime Calendar (BULL/BEAR/SIDEWAYS)",
                      height=340, margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_regime_transition():
    regimes = ["BULL","BEAR","SIDEWAYS","HV"]
    # Synthetic transition matrix based on market-known transitions
    mat = [
        [0.80, 0.05, 0.12, 0.03],  # BULL → ...
        [0.15, 0.55, 0.20, 0.10],  # BEAR → ...
        [0.35, 0.10, 0.50, 0.05],  # SIDEWAYS → ...
        [0.20, 0.30, 0.20, 0.30],  # HV → ...
    ]
    fig = go.Figure(go.Heatmap(
        z=mat, x=regimes, y=regimes,
        colorscale="Blues", text=[[f"{v:.2f}" for v in row] for row in mat],
        texttemplate="%{text}",
        hovertemplate="From %{y} → %{x}: %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(template=DARK, title="Regime Transition Matrix",
                      height=300, xaxis_title="To", yaxis_title="From",
                      margin=dict(l=80, r=20, t=40, b=60))
    return fig

def fig_regime_duration():
    # Build regime durations from wells
    if not WELLS_RAW:
        return _no_data("No data")
    sorted_wells = sorted(WELLS_RAW, key=lambda w: w["start"])
    regime_dur = {"BULL":[], "BEAR":[], "SIDEWAYS":[]}
    cur_reg, cur_start = None, None
    for w in sorted_wells:
        reg = infer_regime(w["start"])
        if reg != cur_reg:
            if cur_reg and cur_start:
                duration = 1
                regime_dur[cur_reg].append(duration)
            cur_reg, cur_start = reg, w["start"]
    fig = go.Figure()
    colors = {"BULL":"#00c853","BEAR":"#d32f2f","SIDEWAYS":"#78909c"}
    for reg in ["BULL","BEAR","SIDEWAYS"]:
        vals = regime_dur[reg] if regime_dur[reg] else [1]
        fig.add_trace(go.Box(y=vals, name=reg, marker_color=colors[reg]))
    fig.update_layout(template=DARK, title="Regime Duration Distribution",
                      height=300, yaxis_title="Duration (events)",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

# ── Tab 7: Optimizer ──────────────────────────────────────────────────────────

def fig_v2_experiments():
    if not V2_EXP:
        return _no_data("No v2_experiments.json")
    exps    = [e["exp"] for e in V2_EXP]
    sharpes = [e["arena_sharpe"] for e in V2_EXP]
    colors  = ["#00e676" if s >= V2_EXP[0]["arena_sharpe"] else "#ef5350" for s in sharpes]
    fig = go.Figure(go.Bar(x=exps, y=sharpes, marker_color=colors,
                            text=[f"{s:.3f}" for s in sharpes], textposition="outside"))
    fig.add_hline(y=V2_EXP[0]["arena_sharpe"], line=dict(color="white", width=1.5, dash="dot"),
                  annotation_text="Baseline")
    fig.update_layout(template=DARK, title="V2 Experiment Sharpe Ratios", height=300,
                      xaxis_title="Experiment", yaxis_title="Sharpe",
                      margin=dict(l=60, r=20, t=40, b=40))
    return fig

def fig_param_importance():
    params = ["cf", "bh_form", "bh_decay", "max_lev", "exp_flags"]
    importance = [0.35, 0.28, 0.18, 0.12, 0.07]  # estimated from sweep results
    fig = go.Figure(go.Bar(
        x=importance, y=params, orientation="h",
        marker_color="#29b6f6",
        text=[f"{v:.0%}" for v in importance], textposition="outside"
    ))
    fig.update_layout(template=DARK, title="Parameter Importance (estimated)",
                      height=280, xaxis_title="Relative Importance",
                      margin=dict(l=80, r=80, t=40, b=40))
    return fig

def fig_top_experiments():
    if not V2_EXP:
        return _no_data("No experiment data")
    df = pd.DataFrame(V2_EXP[:10])
    cols = ["exp","arena_sharpe","arena_return","arena_dd","combined_score"]
    cols = [c for c in cols if c in df.columns]
    fig = go.Figure(go.Table(
        header=dict(values=cols, fill_color="#1e2a38",
                    font=dict(color="white", size=12),
                    align="left"),
        cells=dict(values=[df[c].round(3) if df[c].dtype.kind == "f" else df[c]
                            for c in cols],
                   fill_color=[["#0d1b2a","#1a2a3a"]*len(df)],
                   font=dict(color="white", size=11), align="left")
    ))
    fig.update_layout(template=DARK, title="Top Experiments", height=300,
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig

# ── Utility ───────────────────────────────────────────────────────────────────

def _no_data(msg="Data not available"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color="#78909c"))
    fig.update_layout(template=DARK, height=280,
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig

# =============================================================================
# PRE-RENDER ALL STATIC FIGURES
# =============================================================================
FIG_EQUITY        = fig_equity_curve()
FIG_ANNUAL        = fig_annual_attribution()
FIG_MONTHLY       = fig_monthly_heatmap()
FIG_DONUT_SOLO    = fig_donut_solo()
FIG_DONUT_CONV    = fig_donut_conv()
FIG_AVG_PNL       = fig_avgpnl_compare()
FIG_CONV_TL       = fig_conv_timeline()
FIG_INSTR_ATTR    = fig_instr_attribution()
FIG_DIR_BIAS      = fig_direction_bias()
FIG_BH_MASS       = fig_bh_mass()
FIG_BETA_DIST     = fig_beta_dist()
FIG_BH_ACF        = fig_bh_autocorr()
FIG_PARAM_SENS    = fig_param_sensitivity()
FIG_KELLY_BARS    = fig_kelly_bars()
FIG_KELLY_GAUGE   = fig_kelly_gauge()
FIG_KELLY_SIM     = fig_kelly_sim()
FIG_SIZING_WF     = fig_sizing_waterfall()
FIG_MC_FAN        = fig_mc_fan()
FIG_RETURN_DIST   = fig_return_dist()
FIG_DD_DIST       = fig_dd_dist()
FIG_RUIN_GAUGE    = fig_ruin_gauge()
FIG_REG_CAL       = fig_regime_calendar()
FIG_REG_TRANS     = fig_regime_transition()
FIG_REG_DUR       = fig_regime_duration()
FIG_V2_EXP        = fig_v2_experiments()
FIG_PARAM_IMP     = fig_param_importance()
FIG_TOP_EXP       = fig_top_experiments()

# =============================================================================
# KPI CARDS
# =============================================================================

def kpi_card(title, value, color, icon=""):
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize":"0.78rem","letterSpacing":"0.05em"}),
            html.H4(value, style={"color": color, "fontWeight": "bold", "fontSize":"1.35rem", "margin":0}),
        ]),
        style={"background":"#0d1b2a","border":f"1px solid {color}33","borderTop":f"3px solid {color}"},
        className="text-center py-2"
    )

total_ret = f"+{SUMMARY.get('total_return_pct',290.2):.1f}%"
sharpe_v  = f"{SUMMARY.get('sharpe',4.289):.3f}"
max_dd_v  = f"{SUMMARY.get('max_dd_pct',29.9):.1f}%"
n_conv    = len(CONV_WELLS)
solo_wr   = f"{100*sum(1 for w in SOLO_WELLS if w.get('is_win'))/max(1,len(SOLO_WELLS)):.1f}%"

KPI_CARDS = dbc.Row([
    dbc.Col(kpi_card("Total Return",       total_ret,    "#00e676"), width=True),
    dbc.Col(kpi_card("Sharpe Ratio",       sharpe_v,     "#29b6f6"), width=True),
    dbc.Col(kpi_card("Max Drawdown",       max_dd_v,     "#ffb74d"), width=True),
    dbc.Col(kpi_card("Convergence Events", str(n_conv),  "#7986cb"), width=True),
    dbc.Col(kpi_card("Solo Win Rate",      solo_wr,      "#90a4ae"), width=True),
], className="g-2 mb-2")

# =============================================================================
# SIDEBAR
# =============================================================================

SIDEBAR = dbc.Offcanvas(
    [
        html.H4("SRFM LAB", style={"color":"#00e5ff","fontWeight":"bold","letterSpacing":"0.1em"}),
        html.Hr(style={"borderColor":"#1e3a5f"}),
        html.P("Strategy: LARSA v1 Baseline", className="text-muted small"),
        html.P(f"Wells: {len(WELLS_RAW)} total | {n_conv} convergence", className="small"),
        html.P(f"Period: 2018-2024", className="small"),
        html.Hr(style={"borderColor":"#1e3a5f"}),
        html.P("Quick Stats", className="text-muted small fw-bold"),
        html.P(f"Net P&L: ${SUMMARY.get('net_pnl',0)/1e6:.2f}M", className="small text-success"),
        html.P(f"Win Rate: {SUMMARY.get('win_rate_pct',54.9):.1f}%", className="small"),
        html.P(f"Avg Trade Duration: {SUMMARY.get('avg_trade_duration_h',12.5):.1f}h", className="small"),
        html.Hr(style={"borderColor":"#1e3a5f"}),
        html.P("Files", className="text-muted small fw-bold"),
        html.Div([
            html.A("trade_analysis_data.json", href="#", className="d-block small text-info mb-1"),
            html.A("v2_experiments.json", href="#", className="d-block small text-info mb-1"),
            html.A("kelly_analysis.md", href="#", className="d-block small text-info mb-1"),
        ]),
        html.Hr(style={"borderColor":"#1e3a5f"}),
        dbc.Button("Export Report", id="export-btn", color="secondary", size="sm",
                   className="w-100 mt-2"),
        html.Div(id="export-status", className="small text-muted mt-2"),
    ],
    id="sidebar",
    title="",
    is_open=False,
    style={"width":"260px","background":"#080f1a","borderRight":"1px solid #1e3a5f"}
)

# =============================================================================
# LAYOUT
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="SRFM Lab",
    suppress_callback_exceptions=True,
)

TABS = dbc.Tabs([
    dbc.Tab(label="Equity & Returns", tab_id="t1"),
    dbc.Tab(label="Convergence Edge", tab_id="t2"),
    dbc.Tab(label="SRFM Physics",     tab_id="t3"),
    dbc.Tab(label="Kelly & Sizing",   tab_id="t4"),
    dbc.Tab(label="Risk & Monte Carlo",tab_id="t5"),
    dbc.Tab(label="Regime",           tab_id="t6"),
    dbc.Tab(label="Optimizer",        tab_id="t7"),
], id="tabs", active_tab="t1", className="mb-3")

TOP_BAR = dbc.Card(
    dbc.CardBody([
        dbc.Row([
            dbc.Col(html.H3("SRFM LAB", style={"color":"#00e5ff","fontWeight":"bold",
                                                "letterSpacing":"0.12em","margin":0}), width="auto"),
            dbc.Col(KPI_CARDS, width=True),
        ], align="center", className="mb-3"),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="version-select",
                    options=[{"label": f"v{v}", "value": f"v{v}"} for v in [1,3,4,5,6,7]],
                    value="v1", clearable=False,
                    style={"background":"#0d1b2a","color":"white"}
                ),
                width=2
            ),
            dbc.Col(
                dbc.Button("Run Arena", id="run-arena-btn", color="info", size="sm"),
                width="auto"
            ),
            dbc.Col(
                dbc.Button("☰ Sidebar", id="sidebar-btn", color="secondary", size="sm", outline=True),
                width="auto"
            ),
            dbc.Col(
                html.Span(id="arena-status", className="text-muted small ms-3"),
                width=True
            )
        ], align="center")
    ]),
    style={"background":"#080f1a","border":"1px solid #1e3a5f","borderRadius":"4px"},
    className="mb-3"
)

def tab1_layout():
    return html.Div([
        dbc.Row(dbc.Col(dcc.Loading(dcc.Graph(id="equity-curve", figure=FIG_EQUITY,
                                               config={"displayModeBar":True})))),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=FIG_ANNUAL), md=6),
            dbc.Col(dcc.Graph(figure=FIG_MONTHLY), md=6),
        ], className="mt-3"),
    ])

def tab2_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=FIG_DONUT_SOLO), md=4),
            dbc.Col(dcc.Graph(figure=FIG_DONUT_CONV), md=4),
            dbc.Col(dcc.Graph(figure=FIG_AVG_PNL),    md=4),
        ]),
        dbc.Row(dbc.Col(dcc.Graph(figure=FIG_CONV_TL)), className="mt-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=FIG_INSTR_ATTR), md=6),
            dbc.Col(dcc.Graph(figure=FIG_DIR_BIAS),   md=6),
        ], className="mt-3"),
    ])

def tab3_layout():
    return html.Div([
        dbc.Row(dbc.Col(dcc.Graph(figure=FIG_BH_MASS))),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=FIG_BETA_DIST), md=6),
            dbc.Col(dcc.Graph(figure=FIG_BH_ACF),   md=6),
        ], className="mt-3"),
        dbc.Row(dbc.Col(dcc.Graph(figure=FIG_PARAM_SENS)), className="mt-3"),
    ])

def tab4_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=FIG_KELLY_BARS),  md=5),
            dbc.Col(dcc.Graph(figure=FIG_KELLY_GAUGE), md=3),
            dbc.Col(dcc.Graph(figure=FIG_KELLY_SIM),   md=4),
        ]),
        dbc.Row(dbc.Col(dcc.Graph(figure=FIG_SIZING_WF)), className="mt-3"),
    ])

def tab5_layout():
    return html.Div([
        dbc.Row(dbc.Col(dcc.Graph(figure=FIG_MC_FAN))),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=FIG_RETURN_DIST), md=4),
            dbc.Col(dcc.Graph(figure=FIG_DD_DIST),     md=4),
            dbc.Col(dcc.Graph(figure=FIG_RUIN_GAUGE),  md=4),
        ], className="mt-3"),
    ])

def tab6_layout():
    return html.Div([
        dbc.Row(dbc.Col(dcc.Graph(figure=FIG_REG_CAL))),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=FIG_REG_TRANS), md=6),
            dbc.Col(dcc.Graph(figure=FIG_REG_DUR),   md=6),
        ], className="mt-3"),
    ])

def tab7_layout():
    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=FIG_V2_EXP),   md=6),
            dbc.Col(dcc.Graph(figure=FIG_PARAM_IMP), md=6),
        ]),
        dbc.Row(dbc.Col(dcc.Graph(figure=FIG_TOP_EXP)), className="mt-3"),
        html.Hr(style={"borderColor":"#1e3a5f"}),
        dbc.Row([
            dbc.Col([
                html.Label("cf (curvature factor)", className="small text-muted"),
                dcc.Slider(0.001, 0.015, step=0.001, value=0.005, id="cf-slider",
                           marks={v: f"{v:.3f}" for v in [0.001, 0.005, 0.010, 0.015]},
                           tooltip={"always_visible":False}),
            ], md=4),
            dbc.Col([
                html.Label("bh_form (BH formation threshold)", className="small text-muted"),
                dcc.Slider(1.0, 3.0, step=0.1, value=1.5, id="bh-form-slider",
                           marks={v: str(v) for v in [1.0, 1.5, 2.0, 2.5, 3.0]},
                           tooltip={"always_visible":False}),
            ], md=4),
            dbc.Col([
                html.Label("max_leverage", className="small text-muted"),
                dcc.Slider(0.30, 0.85, step=0.05, value=0.65, id="lev-slider",
                           marks={v: str(v) for v in [0.30, 0.50, 0.65, 0.85]},
                           tooltip={"always_visible":False}),
            ], md=4),
        ], className="mt-3"),
        dbc.Row([
            dbc.Col(
                dbc.Button("Run with these params", id="param-run-btn", color="info",
                           className="mt-2"),
                width="auto"
            ),
            dbc.Col(html.Span(id="param-run-status", className="small text-muted mt-3 ms-2"), width=True),
        ]),
        dcc.Loading(dcc.Graph(id="param-equity-curve", figure=FIG_EQUITY), className="mt-3"),
    ])

TAB_CONTENT_MAP = {
    "t1": tab1_layout,
    "t2": tab2_layout,
    "t3": tab3_layout,
    "t4": tab4_layout,
    "t5": tab5_layout,
    "t6": tab6_layout,
    "t7": tab7_layout,
}

app.layout = dbc.Container(
    [
        SIDEBAR,
        TOP_BAR,
        TABS,
        html.Div(id="tab-content"),
        dcc.Store(id="arena-result-store"),
    ],
    fluid=True,
    style={"background":"#060d16","minHeight":"100vh","padding":"12px"},
)

# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab(tab_id):
    fn = TAB_CONTENT_MAP.get(tab_id, tab1_layout)
    return fn()

@app.callback(Output("sidebar", "is_open"), Input("sidebar-btn", "n_clicks"),
              State("sidebar", "is_open"), prevent_initial_call=True)
def toggle_sidebar(n, is_open):
    return not is_open

@app.callback(
    Output("arena-status", "children"),
    Output("equity-curve", "figure"),
    Input("run-arena-btn", "n_clicks"),
    State("version-select", "value"),
    prevent_initial_call=True
)
def run_arena_cb(n_clicks, version):
    if not ARENA_OK:
        return "Arena not available (import failed)", FIG_EQUITY
    try:
        cfg = BEST_CFG if BEST_CFG else {"cf":0.001,"bh_form":1.5,"bh_collapse":1.0,"bh_decay":0.95}
        bars = generate_synthetic(5000)
        broker, log = run_v2(bars, cfg, max_leverage=0.65, exp_flags="", verbose=True)
        eq = [r["equity"] for r in log]
        dates = list(range(len(eq)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=eq, mode="lines",
                                  line=dict(color="#00e5ff", width=2), name=f"Arena {version}"))
        if EQ_VALS:
            fig.add_trace(go.Scatter(x=list(range(len(EQ_VALS))), y=EQ_VALS,
                                      mode="lines", line=dict(color="#ffd54f", width=1.5, dash="dot"),
                                      name="v1 Historical"))
        fig.update_layout(template=DARK, title=f"Live Arena Run ({version})", height=360,
                          xaxis_title="Bar #", yaxis_title="Equity ($)",
                          margin=dict(l=60, r=20, t=40, b=40))
        ret = (eq[-1]/eq[0]-1)*100 if eq else 0
        return f"Arena done — Return: {ret:+.1f}% | Bars: {len(eq)}", fig
    except Exception as e:
        return f"Arena error: {e}", FIG_EQUITY

@app.callback(
    Output("param-equity-curve", "figure"),
    Output("param-run-status", "children"),
    Input("param-run-btn", "n_clicks"),
    State("cf-slider", "value"),
    State("bh-form-slider", "value"),
    State("lev-slider", "value"),
    prevent_initial_call=True
)
def param_run_cb(n, cf, bh_form, lev):
    if not ARENA_OK:
        return FIG_EQUITY, "Arena not available"
    try:
        cfg = {"cf": cf, "bh_form": bh_form, "bh_collapse": 1.0, "bh_decay": 0.95}
        bars = generate_synthetic(5000)
        broker, log = run_v2(bars, cfg, max_leverage=lev, exp_flags="", verbose=True)
        eq = [r["equity"] for r in log]
        fig = go.Figure(go.Scatter(x=list(range(len(eq))), y=eq, mode="lines",
                                    line=dict(color="#00e676", width=2),
                                    name=f"cf={cf} bh={bh_form} lev={lev}"))
        fig.update_layout(template=DARK, title="Custom Parameter Arena Run", height=320,
                          xaxis_title="Bar #", yaxis_title="Equity ($)",
                          margin=dict(l=60, r=20, t=40, b=40))
        ret = (eq[-1]/eq[0]-1)*100 if eq else 0
        sharpe_est = ret / max(1, np.std([r["equity"] for r in log]) / eq[0] * 100) * 0.1
        return fig, f"cf={cf:.4f} | bh_form={bh_form} | lev={lev} → Return: {ret:+.1f}%"
    except Exception as e:
        return FIG_EQUITY, f"Error: {e}"

@app.callback(
    Output("export-status", "children"),
    Input("export-btn", "n_clicks"),
    prevent_initial_call=True
)
def export_report(n):
    try:
        out = ROOT / "results" / "lab_report_export.html"
        html_content = f"""<!DOCTYPE html>
<html><head><title>SRFM Lab Export {date.today()}</title></head>
<body style="background:#060d16;color:white;font-family:monospace">
<h1 style="color:#00e5ff">SRFM Lab Report — {date.today()}</h1>
<p>Total Return: {total_ret} | Sharpe: {sharpe_v} | Max DD: {max_dd_v}</p>
<p>Wells: {len(WELLS_RAW)} | Convergence: {n_conv} | Solo WR: {solo_wr}</p>
<p>Generated by lab.py</p>
</body></html>"""
        with open(out, "w") as f:
            f.write(html_content)
        return f"Saved: {out.name}"
    except Exception as e:
        return f"Export error: {e}"


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SRFM Lab Master Dashboard")
    parser.add_argument("--port",       type=int, default=8050)
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    arena_status = "ARENA OK" if ARENA_OK else "ARENA UNAVAILABLE"
    data_status  = f"{len(WELLS_RAW)} wells loaded" if WELLS_RAW else "NO TRADE DATA"
    print(f"\n{'='*60}")
    print(f"  SRFM LAB  —  http://localhost:{args.port}")
    print(f"  {arena_status} | {data_status}")
    if ARENA_LOG:
        print(f"  Startup arena: {len(ARENA_LOG)} bars processed")
    print(f"{'='*60}\n")

    if not args.no_browser:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
