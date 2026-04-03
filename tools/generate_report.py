"""
SRFM Lab — Static HTML Report Generator
Run: python tools/generate_report.py
Output: results/strategy_report.html
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import json
import os
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_TRADES = "C:/Users/Matthew/Downloads/Calm Orange Mule_trades.csv"
V3_TRADES = "C:/Users/Matthew/Downloads/Measured Red Anguilline_trades.csv"
REGIMES   = os.path.join(BASE_DIR, "results", "regimes_ES.csv")
WELLS_JSON = os.path.join(BASE_DIR, "research", "trade_analysis_data.json")
EXPERIMENTS = os.path.join(BASE_DIR, "results", "v2_experiments.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "strategy_report.html")

INSTRUMENT_COLORS = {"ES": "#f39c12", "NQ": "#9b59b6", "YM": "#1abc9c"}
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
PLOT_BG = "#161b22"
GRID_COLOR = "#30363d"
TEXT_COLOR = "#e6edf3"

# ── Data loading ───────────────────────────────────────────────────────────────

def load_trades(path, label):
    if not path or not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["version"] = label
    for col in ["Entry Time", "entry_time", "EntryTime"]:
        if col in df.columns:
            df["entry_dt"] = pd.to_datetime(df[col], errors="coerce")
            break
    if "entry_dt" in df.columns:
        df["year"] = df["entry_dt"].dt.year
        df["month"] = df["entry_dt"].dt.month
    for col in ["P&L", "Net Profit", "pnl", "NetProfit"]:
        if col in df.columns:
            df["pnl"] = pd.to_numeric(df[col], errors="coerce")
            break
    for col in ["Symbols", "Symbol", "symbol"]:
        if col in df.columns:
            df["instrument"] = df[col].str[:2].str.upper()
            break
    for col in ["Direction", "direction", "Type"]:
        if col in df.columns:
            df["direction"] = df[col]
            break
    for entry_col, exit_col in [("Entry Time", "Exit Time"), ("EntryTime", "ExitTime")]:
        if entry_col in df.columns and exit_col in df.columns:
            df["exit_dt"] = pd.to_datetime(df[exit_col], errors="coerce")
            df["duration_h"] = (df["exit_dt"] - df["entry_dt"]).dt.total_seconds() / 3600
            break
    print(f"  [OK] {label}: {len(df)} trades loaded from {os.path.basename(path)}")
    return df


def load_json(path):
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    print(f"  [OK] {os.path.basename(path)} loaded")
    return data


def load_regimes(path):
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"  [OK] regimes_ES.csv: {len(df)} bars")
    return df


# ── Plot styling helper ────────────────────────────────────────────────────────

def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR, family="Segoe UI, sans-serif"),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR),
    )
    base.update(kwargs)
    return base


def fig_to_html(fig, first=False):
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if first else False,
        config={"displayModeBar": True, "responsive": True},
    )


# ── Metric card ───────────────────────────────────────────────────────────────

def metric_card(label, v1_val, v3_val, format_str="{:.1f}%", higher_better=True):
    delta = v3_val - v1_val
    is_better = (delta > 0) == higher_better
    delta_class = "positive" if is_better else "negative"
    delta_sign = "+" if delta > 0 else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{format_str.format(v3_val)}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-delta {delta_class}">{delta_sign}{format_str.format(delta)} vs v1</div>
    </div>
    """


# ── Chart builders ─────────────────────────────────────────────────────────────

def build_equity_comparison(v1, v3):
    """Section 1: Equity curves overlaid."""
    fig = go.Figure()
    for df, name, color in [(v1, "v1", "#58a6ff"), (v3, "v3", "#3fb950")]:
        if df.empty or "pnl" not in df.columns:
            continue
        d = df.sort_values("entry_dt").copy()
        d["cum_pnl"] = d["pnl"].cumsum()
        fig.add_trace(go.Scatter(
            x=d["entry_dt"], y=d["cum_pnl"],
            name=name, line=dict(color=color, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Cum P&L: $%{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(**dark_layout(title="Equity Curve: v1 vs v3", height=400,
                                    yaxis_tickprefix="$", yaxis_tickformat=",.0f"))
    return fig


def build_annual_pnl_bar(v1, v3):
    """Section 1: Annual P&L grouped bar."""
    fig = go.Figure()
    for df, name, color in [(v1, "v1", "#58a6ff"), (v3, "v3", "#3fb950")]:
        if df.empty or "year" not in df.columns:
            continue
        annual = df.groupby("year")["pnl"].sum().reset_index()
        fig.add_trace(go.Bar(
            x=annual["year"], y=annual["pnl"], name=name,
            marker_color=color,
            hovertemplate="Year %{x}<br>P&L: $%{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(**dark_layout(title="Annual P&L: v1 vs v3", height=380,
                                    barmode="group", yaxis_tickprefix="$", yaxis_tickformat=",.0f"))
    return fig


def compute_metrics(df):
    if df.empty or "pnl" not in df.columns:
        return {}
    cum = df.sort_values("entry_dt")["pnl"].cumsum()
    total = cum.iloc[-1] if len(cum) else 0
    peak = cum.cummax()
    dd_series = (cum - peak)
    max_dd = dd_series.min()
    returns = df["pnl"]
    sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
    wr = df["pnl"].gt(0).mean() * 100 if "pnl" in df.columns else 0
    trades = len(df)
    # Approximate return% assuming ~$1M starting capital
    ret_pct = total / 1_000_000 * 100
    dd_pct = max_dd / 1_000_000 * 100
    return {"return_pct": ret_pct, "dd_pct": dd_pct, "sharpe": sharpe, "wr": wr, "trades": trades}


def build_trade_scatter(v3):
    """Section 2: Trade scatter duration vs P&L."""
    if v3.empty:
        return go.Figure()
    df = v3.dropna(subset=["duration_h", "pnl", "instrument"]).copy()
    df["color"] = df["instrument"].map(INSTRUMENT_COLORS).fillna("#888")
    df["abs_pnl"] = df["pnl"].abs()
    # Build one trace per instrument for proper legend
    fig = go.Figure()
    for instr, color in INSTRUMENT_COLORS.items():
        sub = df[df["instrument"] == instr]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["duration_h"], y=sub["pnl"],
            mode="markers", name=instr,
            marker=dict(
                color=color,
                size=(sub["abs_pnl"] / sub["abs_pnl"].max() * 30 + 4).clip(4, 34),
                opacity=0.75,
                line=dict(width=0.5, color="rgba(255,255,255,0.13)"),
            ),
            hovertemplate=f"<b>{instr}</b><br>Duration: %{{x:.1f}}h<br>P&L: $%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(**dark_layout(
        title="v3 Trade Scatter: Duration vs P&L",
        height=420,
        xaxis_title="Duration (hours)",
        yaxis_title="P&L ($)",
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
    ))
    return fig


def build_pnl_histogram(v3):
    """Section 2: P&L histogram per instrument."""
    if v3.empty:
        return go.Figure()
    fig = go.Figure()
    for instr, color in INSTRUMENT_COLORS.items():
        sub = v3[v3["instrument"] == instr]["pnl"].dropna()
        if sub.empty:
            continue
        fig.add_trace(go.Histogram(
            x=sub, name=instr, marker_color=color,
            opacity=0.65, nbinsx=30,
            hovertemplate=f"<b>{instr}</b><br>P&L: $%{{x:,.0f}}<br>Count: %{{y}}<extra></extra>",
        ))
    fig.update_layout(**dark_layout(
        title="P&L Distribution by Instrument",
        height=380, barmode="overlay",
        xaxis_title="P&L ($)", yaxis_title="Count",
        xaxis_tickprefix="$", xaxis_tickformat=",.0f",
    ))
    return fig


def build_nq_notional(v3):
    """Section 2: NQ notional over time with killer trades flagged."""
    if v3.empty:
        return go.Figure()
    nq = v3[v3["instrument"] == "NQ"].dropna(subset=["entry_dt", "pnl"]).sort_values("entry_dt")
    if nq.empty:
        return go.Figure()

    killer_dates = {
        "2024-10-14": ("Oct-14 2024", -265000),
        "2024-06-18": ("Jun-18 2024", -177000),
        "2024-12-09": ("Dec-09 2024", -121000),
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nq["entry_dt"], y=nq["pnl"],
        mode="markers+lines", name="NQ P&L",
        marker=dict(color="#9b59b6", size=5),
        line=dict(color="#9b59b6", width=1, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>P&L: $%{y:,.0f}<extra></extra>",
    ))

    # Annotate killer trades
    for date_str, (label, approx_pnl) in killer_dates.items():
        matches = nq[nq["entry_dt"].dt.strftime("%Y-%m-%d") == date_str]
        if matches.empty:
            continue
        row = matches.iloc[0]
        fig.add_annotation(
            x=row["entry_dt"], y=row["pnl"],
            text=f"💀 {label}<br>${row['pnl']:,.0f}",
            showarrow=True, arrowhead=2, arrowcolor="#f85149",
            font=dict(color="#f85149", size=11),
            bgcolor=CARD_BG, bordercolor="#f85149", borderwidth=1,
            ax=40, ay=-40,
        )

    fig.update_layout(**dark_layout(
        title="NQ P&L Over Time (Killer Trades Flagged)",
        height=420,
        xaxis_title="Entry Date",
        yaxis_title="P&L ($)",
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
    ))
    return fig


def build_top_wins_losses(v3):
    """Section 2: Top 10 wins and losses side by side."""
    if v3.empty or "pnl" not in v3.columns:
        return go.Figure()
    wins = v3.nlargest(10, "pnl")[["entry_dt", "pnl", "instrument"]].copy()
    losses = v3.nsmallest(10, "pnl")[["entry_dt", "pnl", "instrument"]].copy()

    def label(row):
        dt = row["entry_dt"].strftime("%Y-%m-%d") if pd.notna(row["entry_dt"]) else "?"
        return f"{row['instrument']} {dt}"

    wins["label"] = wins.apply(label, axis=1)
    losses["label"] = losses.apply(label, axis=1)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Top 10 Wins", "Top 10 Losses"),
                        horizontal_spacing=0.12)
    fig.add_trace(go.Bar(
        x=wins["pnl"], y=wins["label"], orientation="h",
        marker_color="#3fb950", name="Wins",
        hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=losses["pnl"], y=losses["label"], orientation="h",
        marker_color="#f85149", name="Losses",
        hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(**dark_layout(title="Top 10 Wins & Losses (v3)", height=420, showlegend=False))
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)})
    return fig


def build_regime_donut(regimes):
    """Section 3: Regime time distribution donut."""
    if regimes.empty:
        return go.Figure()
    counts = regimes["regime"].value_counts()
    colors = ["#58a6ff", "#3fb950", "#f39c12", "#f85149", "#9b59b6"]
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.values.tolist(),
        hole=0.5,
        marker=dict(colors=colors[:len(counts)]),
        hovertemplate="%{label}<br>%{value:,} bars (%{percent})<extra></extra>",
    ))
    fig.update_layout(**dark_layout(title="Regime Time Distribution", height=380))
    return fig


def build_regime_persistence(regimes):
    """Section 3: Average run length per regime."""
    if regimes.empty:
        return go.Figure()
    # Compute run lengths
    regimes = regimes.copy()
    regimes["run_id"] = (regimes["regime"] != regimes["regime"].shift()).cumsum()
    runs = regimes.groupby(["run_id", "regime"]).size().reset_index(name="run_len")
    avg_runs = runs.groupby("regime")["run_len"].mean().reset_index()
    avg_runs.columns = ["regime", "avg_run"]
    colors = ["#58a6ff", "#3fb950", "#f39c12", "#f85149", "#9b59b6"]
    fig = go.Figure(go.Bar(
        x=avg_runs["regime"], y=avg_runs["avg_run"],
        marker_color=colors[:len(avg_runs)],
        hovertemplate="%{x}<br>Avg run: %{y:.1f} bars<extra></extra>",
    ))
    fig.update_layout(**dark_layout(title="Regime Persistence (Avg Run Length)", height=360,
                                    yaxis_title="Avg bars per run"))
    return fig


def build_regime_winrate(regimes, v3):
    """Section 3: Win rate by regime."""
    if regimes.empty or v3.empty or "entry_dt" not in v3.columns:
        return go.Figure()
    reg = regimes.copy()
    reg["date"] = pd.to_datetime(reg["date"]).dt.tz_localize(None).dt.floor("h")
    trades = v3.copy()
    trades["date"] = trades["entry_dt"].dt.tz_localize(None).dt.floor("h") if trades["entry_dt"].dt.tz is not None else trades["entry_dt"].dt.floor("h")
    merged = trades.merge(reg[["date", "regime"]], on="date", how="left")
    if "regime" not in merged.columns or merged["regime"].isna().all():
        return go.Figure()
    wr = merged.groupby("regime")["pnl"].apply(lambda x: (x > 0).mean() * 100).reset_index()
    wr.columns = ["regime", "win_rate"]
    colors = ["#58a6ff", "#3fb950", "#f39c12", "#f85149", "#9b59b6"]
    fig = go.Figure(go.Bar(
        x=wr["regime"], y=wr["win_rate"],
        marker_color=colors[:len(wr)],
        hovertemplate="%{x}<br>Win rate: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#8b949e",
                  annotation_text="50% baseline", annotation_position="top right")
    fig.update_layout(**dark_layout(title="Win Rate by Regime (v3)", height=360,
                                    yaxis_title="Win Rate (%)", yaxis_range=[0, 100]))
    return fig


def build_well_scatter(wells_data):
    """Section 4: Well scatter."""
    if not wells_data:
        return go.Figure()
    wells = wells_data.get("wells", [])
    if not wells:
        return go.Figure()
    df = pd.DataFrame(wells)
    if "duration_h" not in df.columns or "total_pnl" not in df.columns:
        return go.Figure()
    df["n_instr"] = df["instruments"].apply(len)
    color_map = {1: "#58a6ff", 2: "#f39c12"}
    df["color"] = df["n_instr"].apply(lambda n: color_map.get(n, "#f85149"))
    df["size"] = (df["total_pnl"].abs() / df["total_pnl"].abs().max() * 30 + 4).clip(4, 34)
    color_label = {1: "1 instrument", 2: "2 instruments", 3: "3+ instruments"}
    fig = go.Figure()
    for n in sorted(df["n_instr"].unique()):
        sub = df[df["n_instr"] == n]
        fig.add_trace(go.Scatter(
            x=sub["duration_h"], y=sub["total_pnl"],
            mode="markers", name=color_label.get(n, f"{n} instruments"),
            marker=dict(color=sub["color"].iloc[0], size=sub["size"], opacity=0.75,
                        line=dict(width=0.5, color="rgba(255,255,255,0.13)")),
            hovertemplate="Duration: %{x:.1f}h<br>P&L: $%{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(**dark_layout(title="Well Scatter: Duration vs P&L (v1)", height=420,
                                    xaxis_title="Duration (hours)", yaxis_title="Total P&L ($)",
                                    yaxis_tickprefix="$", yaxis_tickformat=",.0f"))
    return fig


def build_well_waterfall(wells_data):
    """Section 4: Annual P&L waterfall (cumulative by well)."""
    if not wells_data:
        return go.Figure()
    wells = wells_data.get("wells", [])
    if not wells:
        return go.Figure()
    df = pd.DataFrame(wells).sort_values("start")
    cum = df["total_pnl"].cumsum()
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in df["total_pnl"]]
    fig = go.Figure(go.Bar(
        x=list(range(len(df))), y=df["total_pnl"],
        marker_color=colors,
        hovertemplate="Well #%{x}<br>P&L: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=cum.tolist(),
        mode="lines", name="Cumulative", line=dict(color="#58a6ff", width=2),
        hovertemplate="After well #%{x}<br>Cum: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(**dark_layout(title="Well P&L Waterfall (v1, chronological)", height=400,
                                    xaxis_title="Well #", yaxis_title="P&L ($)",
                                    yaxis_tickprefix="$", yaxis_tickformat=",.0f"))
    return fig


def build_top_wells(wells_data):
    """Section 4: Top 10 / Bottom 10 wells."""
    if not wells_data:
        return go.Figure()
    wells = wells_data.get("wells", [])
    if not wells:
        return go.Figure()
    df = pd.DataFrame(wells)
    df["label"] = pd.to_datetime(df["start"]).dt.strftime("%Y-%m-%d") + " " + df["instruments"].apply(lambda x: "/".join(x))
    top = df.nlargest(10, "total_pnl")
    bot = df.nsmallest(10, "total_pnl")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Top 10 Wells", "Bottom 10 Wells"),
                        horizontal_spacing=0.12)
    fig.add_trace(go.Bar(
        x=top["total_pnl"], y=top["label"], orientation="h",
        marker_color="#3fb950", name="Top",
        hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=bot["total_pnl"], y=bot["label"], orientation="h",
        marker_color="#f85149", name="Bottom",
        hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
    ), row=1, col=2)
    fig.update_layout(**dark_layout(title="Top & Bottom 10 Wells (v1)", height=420, showlegend=False))
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)})
    return fig


def build_experiment_heatmap(experiments):
    """Section 5: 4×4 flag matrix heatmap."""
    if not experiments:
        return go.Figure()
    flags = ["A", "B", "C", "D"]
    # Build 4x4: row=flag on x-axis, col=flag on y-axis
    # For a cell (r,c), find experiment containing both flags[r] and flags[c], or just flags[r] if r==c
    matrix = [[None] * 4 for _ in range(4)]
    exp_map = {}
    for e in experiments:
        f = e.get("flags", "").replace(" ", "")
        exp_map[f] = e.get("combined_score", 0)

    for i, fi in enumerate(flags):
        for j, fj in enumerate(flags):
            if i == j:
                score = exp_map.get(fi, None)
            else:
                # Combination of exactly these two flags
                key1 = fi + fj
                key2 = fj + fi
                score = exp_map.get(key1, exp_map.get(key2, None))
            matrix[i][j] = score if score is not None else 0

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=flags, y=flags,
        colorscale="RdYlGn",
        text=[[f"{v:.3f}" if v is not None else "" for v in row] for row in matrix],
        texttemplate="%{text}",
        hovertemplate="Flag %{y} × %{x}<br>Score: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Score", tickfont=dict(color=TEXT_COLOR)),
    ))
    fig.update_layout(**dark_layout(title="Experiment Flag Matrix (combined_score)", height=420))
    return fig


def build_experiment_scatter(experiments):
    """Section 5: arena_sharpe vs synth_sharpe scatter."""
    if not experiments:
        return go.Figure()
    df = pd.DataFrame(experiments)
    # Exclude leverage variants for clarity — keep pure flag combos
    base = df[~df["exp"].str.contains("lev", na=False)].copy()
    fig = go.Figure(go.Scatter(
        x=base["synth_sharpe"], y=base["arena_sharpe"],
        mode="markers+text",
        marker=dict(
            color=base["combined_score"],
            colorscale="RdYlGn",
            size=10,
            colorbar=dict(title="Score"),
            line=dict(width=0.5, color="rgba(255,255,255,0.27)"),
        ),
        text=base["exp"], textposition="top center",
        textfont=dict(color=TEXT_COLOR, size=9),
        hovertemplate="<b>%{text}</b><br>Synth Sharpe: %{x:.3f}<br>Arena Sharpe: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#8b949e")
    fig.add_vline(x=0, line_dash="dash", line_color="#8b949e")
    fig.update_layout(**dark_layout(
        title="Arena Sharpe vs Synth Sharpe by Experiment",
        height=440,
        xaxis_title="Synth Sharpe",
        yaxis_title="Arena Sharpe",
    ))
    return fig


# ── HTML template ──────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SRFM Lab — Strategy Report</title>
<style>
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
  h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
  h2 {{ color: #3fb950; margin-top: 40px; }}
  h3 {{ color: #79c0ff; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin: 20px 0; }}
  .metric-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }}
  .metric-value {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
  .metric-label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
  .metric-delta {{ font-size: 14px; margin-top: 4px; }}
  .positive {{ color: #3fb950; }}
  .negative {{ color: #f85149; }}
  .chart-container {{ margin: 20px 0; background: #161b22; border-radius: 8px; padding: 10px; }}
  .insight-box {{ background: #161b22; border-left: 3px solid #58a6ff; padding: 12px 16px; margin: 16px 0; border-radius: 0 8px 8px 0; }}
  .warning-box {{ background: #161b22; border-left: 3px solid #f85149; padding: 12px 16px; margin: 16px 0; border-radius: 0 8px 8px 0; }}
  @media (max-width: 900px) {{ .metric-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
</style>
</head>
<body>
<h1>SRFM Lab — Strategy Report</h1>
<p style="color:#8b949e">Generated: {timestamp} | v1: 274% | v2: 175% | v3: 200%</p>

{metric_cards}

{charts}

</body>
</html>
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("SRFM Lab Report Generator")
    print("=" * 40)

    # Load data
    print("\nLoading data...")
    v1 = load_trades(V1_TRADES, "v1")
    v3 = load_trades(V3_TRADES, "v3")
    regimes = load_regimes(REGIMES)
    wells_data = load_json(WELLS_JSON)
    experiments = load_json(EXPERIMENTS)

    # Compute metrics
    m1 = compute_metrics(v1)
    m3 = compute_metrics(v3)

    print("\nBuilding charts...")
    chart_blocks = []
    first_chart = True

    def add_section(title, level=2):
        chart_blocks.append(f"<h{level}>{title}</h{level}>")

    def add_chart(fig, label):
        nonlocal first_chart
        if fig is None or (hasattr(fig, "data") and len(fig.data) == 0):
            chart_blocks.append(f'<p style="color:#8b949e">[Chart skipped: {label} — no data]</p>')
            print(f"  [SKIP] {label} (no data)")
            return
        html = fig_to_html(fig, first=first_chart)
        chart_blocks.append(f'<div class="chart-container">{html}</div>')
        first_chart = False
        print(f"  [CHART] {label}")

    def add_box(html):
        chart_blocks.append(html)

    # ── Section 1: Version Comparison ────────────────────────────────────────
    add_section("Section 1: Version Comparison")
    add_chart(build_equity_comparison(v1, v3), "Equity curve comparison")
    add_chart(build_annual_pnl_bar(v1, v3), "Annual P&L grouped bar")

    # ── Section 2: Trade Deep Dive (v3) ──────────────────────────────────────
    add_section("Section 2: Trade Deep Dive (v3)")
    add_chart(build_trade_scatter(v3), "Trade scatter")
    add_chart(build_pnl_histogram(v3), "P&L histogram")
    add_chart(build_nq_notional(v3), "NQ notional over time")
    add_box("""<div class="warning-box">
  <strong>🔴 NQ Killing Zone:</strong> Oct-14 2024 (-$265k), Jun-18 2024 (-$177k), Dec-09 2024 (-$121k).
  Total: -$563k from 3 trades. NQ $20/pt multiplier means 0.65 leverage on $3M portfolio = ~$2M notional.
  v4 fix: hard $400k notional cap.
</div>""")
    add_chart(build_top_wins_losses(v3), "Top 10 wins & losses")

    # ── Section 3: Regime Analysis ────────────────────────────────────────────
    add_section("Section 3: Regime Analysis")
    add_chart(build_regime_donut(regimes), "Regime donut")
    add_chart(build_regime_persistence(regimes), "Regime persistence")
    add_chart(build_regime_winrate(regimes, v3), "Win rate by regime")
    add_box("""<div class="insight-box">
  <strong>💡 BULL→BEAR never happens directly</strong> (0.01%, 6 events in 52,560 bars).
  Path is always BULL→SIDEWAYS→BEAR. SIDEWAYS avg run = 9.5 bars.
  BEAR gate: block longs when rhb &gt; 5.
</div>""")

    # ── Section 4: Well Analysis (v1) ─────────────────────────────────────────
    add_section("Section 4: Well Analysis (v1)")
    add_chart(build_well_scatter(wells_data), "Well scatter")
    add_chart(build_well_waterfall(wells_data), "Well waterfall")
    add_chart(build_top_wells(wells_data), "Top/bottom wells")

    # ── Section 5: Experiment Matrix ──────────────────────────────────────────
    add_section("Section 5: Experiment Matrix")
    add_chart(build_experiment_heatmap(experiments), "Experiment heatmap")
    add_chart(build_experiment_scatter(experiments), "Arena vs Synth Sharpe scatter")

    # ── Metric cards ──────────────────────────────────────────────────────────
    cards_html = '<div class="metric-grid">'
    if m1 and m3:
        cards_html += metric_card("Net Return", m1.get("return_pct", 0), m3.get("return_pct", 0), "{:.1f}%")
        cards_html += metric_card("Max Drawdown", m1.get("dd_pct", 0), m3.get("dd_pct", 0), "{:.1f}%", higher_better=False)
        cards_html += metric_card("Sharpe Ratio", m1.get("sharpe", 0), m3.get("sharpe", 0), "{:.2f}", higher_better=True)
        cards_html += metric_card("Win Rate", m1.get("wr", 0), m3.get("wr", 0), "{:.1f}%")
        cards_html += metric_card("Total Trades", m1.get("trades", 0), m3.get("trades", 0), "{:.0f}", higher_better=False)
    cards_html += "</div>"

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = HTML_TEMPLATE.format(
        timestamp=timestamp,
        metric_cards=cards_html,
        charts="\n".join(chart_blocks),
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nReport written: {OUTPUT_PATH}")
    print(f"File size: {size_kb:.1f} KB ({size_kb/1024:.2f} MB)")


if __name__ == "__main__":
    main()
