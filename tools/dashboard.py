"""
SRFM Trading Lab — Interactive Dashboard
Run with: streamlit run tools/dashboard.py
"""

import os
import json
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SRFM Trading Lab Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Color palette ──────────────────────────────────────────────────────────
COLORS = {
    "v1": "#2ecc71",
    "v2": "#e74c3c",
    "v3": "#3498db",
    "ES": "#f39c12",
    "NQ": "#9b59b6",
    "YM": "#1abc9c",
    "BULL": "rgba(46,204,113,0.15)",
    "BEAR": "rgba(231,76,60,0.15)",
    "SIDEWAYS": "rgba(149,165,166,0.10)",
    "HIGH_VOLATILITY": "rgba(155,89,182,0.15)",
}

REGIME_LABEL_COLORS = {
    "BULL": "#2ecc71",
    "BEAR": "#e74c3c",
    "SIDEWAYS": "#95a5a6",
    "HIGH_VOLATILITY": "#9b59b6",
}

VERSION_NAMES = {
    "v1": "v1 (Calm Orange Mule / 274%)",
    "v2": "v2 (Fluorescent Yellow / 175%)",
    "v3": "v3 (Measured Red Anguilline / 200%)",
}

TRADE_FILES = {
    "v1": "C:/Users/Matthew/Downloads/Calm Orange Mule_trades.csv",
    "v2": None,
    "v3": "C:/Users/Matthew/Downloads/Measured Red Anguilline_trades.csv",
}

# ─── Data loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_trades(version: str) -> pd.DataFrame:
    path = TRADE_FILES.get(version)
    if path is None or not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    # Parse dates
    for col in ["Entry Time", "Exit Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    # Duration in hours
    if "Entry Time" in df.columns and "Exit Time" in df.columns:
        df["duration_h"] = (df["Exit Time"] - df["Entry Time"]).dt.total_seconds() / 3600
    # Extract instrument from Symbols
    if "Symbols" in df.columns:
        def extract_instrument(sym):
            sym = str(sym).strip().strip('"')
            if sym.startswith("NQ"):
                return "NQ"
            elif sym.startswith("ES"):
                return "ES"
            elif sym.startswith("YM"):
                return "YM"
            return sym[:2] if len(sym) >= 2 else sym
        df["instrument"] = df["Symbols"].apply(extract_instrument)
    # Extract year
    if "Entry Time" in df.columns:
        df["year"] = df["Entry Time"].dt.year
        df["date"] = df["Entry Time"].dt.date
    # Numeric
    if "P&L" in df.columns:
        df["pnl"] = pd.to_numeric(df["P&L"], errors="coerce")
    if "Entry Price" in df.columns:
        df["entry_price"] = pd.to_numeric(df["Entry Price"], errors="coerce")
    if "Quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    if "IsWin" in df.columns:
        df["is_win"] = df["IsWin"].astype(bool)
    df["version"] = version
    return df


@st.cache_data
def load_regimes() -> pd.DataFrame:
    path = "C:/Users/Matthew/srfm-lab/results/regimes_ES.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df


@st.cache_data
def load_wells() -> list:
    path = "C:/Users/Matthew/srfm-lab/research/trade_analysis_data.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("wells", [])


def build_equity_curve(df: pd.DataFrame, initial: float = 1_000_000.0) -> pd.DataFrame:
    """Build cumulative equity from trades sorted by entry time."""
    if df.empty or "pnl" not in df.columns:
        return pd.DataFrame()
    d = df.sort_values("Entry Time").copy()
    d["cum_pnl"] = d["pnl"].cumsum()
    d["equity"] = initial + d["cum_pnl"]
    d["peak"] = d["equity"].cummax()
    d["drawdown_pct"] = (d["equity"] - d["peak"]) / d["peak"] * 100
    return d


def rolling_metrics(df: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    """Compute rolling Sharpe, win rate for a trades DataFrame."""
    if df.empty:
        return pd.DataFrame()
    d = df.sort_values("Entry Time").copy()
    d["is_win_int"] = d["is_win"].astype(int) if "is_win" in d.columns else (d["pnl"] > 0).astype(int)
    d["roll_winrate"] = d["is_win_int"].rolling(window, min_periods=1).mean() * 100
    pnl_std = d["pnl"].rolling(window, min_periods=2).std()
    pnl_mean = d["pnl"].rolling(window, min_periods=2).mean()
    d["roll_sharpe"] = np.where(pnl_std > 0, pnl_mean / pnl_std * np.sqrt(window), np.nan)
    return d


def add_regime_bands(fig, regimes: pd.DataFrame, row=1, col=1):
    """Add regime colored background bands to a figure."""
    if regimes.empty:
        return
    reg = regimes.copy()
    reg = reg.sort_values("date")
    # Group consecutive same-regime rows
    reg["grp"] = (reg["regime"] != reg["regime"].shift()).cumsum()
    for _, grp_df in reg.groupby("grp"):
        regime = grp_df["regime"].iloc[0]
        color = COLORS.get(regime, "rgba(200,200,200,0.1)")
        x0 = str(grp_df["date"].iloc[0])
        x1 = str(grp_df["date"].iloc[-1])
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=color,
            layer="below",
            line_width=0,
            row=row, col=col,
        )


def filter_trades_by_date_instrument(df: pd.DataFrame, start_dt, end_dt, instrument: str) -> pd.DataFrame:
    if df.empty:
        return df
    if start_dt and "Entry Time" in df.columns:
        df = df[df["Entry Time"].dt.date >= start_dt]
    if end_dt and "Entry Time" in df.columns:
        df = df[df["Entry Time"].dt.date <= end_dt]
    if instrument != "All" and "instrument" in df.columns:
        df = df[df["instrument"] == instrument]
    return df


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("SRFM Lab Controls")
    st.markdown("---")

    selected_versions = st.multiselect(
        "Backtest Versions",
        options=["v1", "v2", "v3"],
        default=["v1", "v3"],
        format_func=lambda v: VERSION_NAMES[v],
    )
    if not selected_versions:
        selected_versions = ["v1", "v3"]

    st.markdown("---")
    st.subheader("Date Range")
    date_start = st.date_input("From", value=date(2018, 1, 1))
    date_end = st.date_input("To", value=date(2024, 12, 31))

    st.markdown("---")
    instrument_filter = st.selectbox("Instrument", ["All", "ES", "NQ", "YM"])

    st.markdown("---")
    cf_value = st.slider("CF Value", min_value=0.001, max_value=0.010,
                         value=0.005, step=0.001, format="%.3f")
    st.caption(f"Selected CF: {cf_value:.3f}")

    st.markdown("---")
    st.caption("Run: `streamlit run tools/dashboard.py`")


# ─── Load data ──────────────────────────────────────────────────────────────
all_trades = {}
for v in ["v1", "v2", "v3"]:
    all_trades[v] = load_trades(v)

regimes = load_regimes()
wells = load_wells()

# ─── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔍 Trade Analysis",
    "🗺️ Regime Intelligence",
    "💧 Well Analysis",
    "⚖️ Strategy Comparison",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Overview")

    # ── Chart 1: Equity Curve Comparison ────────────────────────────────────
    st.subheader("Chart 1 — Equity Curve Comparison")

    fig_eq = go.Figure()

    # Add regime bands first (background)
    if not regimes.empty:
        add_regime_bands(fig_eq, regimes)

    has_equity = False
    for v in selected_versions:
        df = filter_trades_by_date_instrument(
            all_trades[v].copy(), date_start, date_end, instrument_filter
        )
        eq = build_equity_curve(df)
        if eq.empty:
            st.caption(f"{VERSION_NAMES[v]}: no trade data available")
            continue
        has_equity = True
        fig_eq.add_trace(go.Scatter(
            x=eq["Entry Time"],
            y=eq["equity"],
            mode="lines",
            name=VERSION_NAMES[v],
            line=dict(color=COLORS[v], width=2),
            customdata=np.stack([
                eq["drawdown_pct"].round(2),
                eq.get("regime", pd.Series(["N/A"] * len(eq))).fillna("N/A") if "regime" in eq.columns else ["N/A"] * len(eq),
            ], axis=-1) if True else None,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Equity: $%{y:,.0f}<br>"
                "Drawdown: %{customdata[0]:.1f}%<br>"
                "<extra>" + VERSION_NAMES[v] + "</extra>"
            ),
        ))

    if not has_equity:
        st.info("No equity data available for selected versions/filters. Check that trade CSV files exist.")
    else:
        # Legend for regime colors
        for regime, color in REGIME_LABEL_COLORS.items():
            fig_eq.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=regime, showlegend=True,
            ))

        fig_eq.update_layout(
            title="Equity Curve — All Versions (with Regime Bands)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── Chart 2: Drawdown Comparison ─────────────────────────────────────────
    st.subheader("Chart 2 — Drawdown Comparison")

    fig_dd = go.Figure()
    has_dd = False
    for v in selected_versions:
        df = filter_trades_by_date_instrument(
            all_trades[v].copy(), date_start, date_end, instrument_filter
        )
        eq = build_equity_curve(df)
        if eq.empty:
            continue
        has_dd = True
        fig_dd.add_trace(go.Scatter(
            x=eq["Entry Time"],
            y=eq["drawdown_pct"],
            mode="lines",
            fill="tozeroy",
            name=VERSION_NAMES[v],
            line=dict(color=COLORS[v], width=1.5),
            fillcolor=COLORS[v].replace(")", ",0.2)").replace("rgb", "rgba") if COLORS[v].startswith("rgb") else COLORS[v] + "33",
        ))

    # Threshold lines
    for thresh, label in [(-10, "-10%"), (-20, "-20%"), (-30, "-30%")]:
        fig_dd.add_hline(y=thresh, line_dash="dash", line_color="red",
                         annotation_text=label, annotation_position="right")

    if has_dd:
        fig_dd.update_layout(
            title="Drawdown Comparison",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            height=350,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.info("No drawdown data available.")

    # ── Chart 3: Annual P&L Grouped Bars ─────────────────────────────────────
    st.subheader("Chart 3 — Annual P&L Grouped Bars")

    annual_frames = []
    for v in selected_versions:
        df = filter_trades_by_date_instrument(
            all_trades[v].copy(), date_start, date_end, instrument_filter
        )
        if df.empty or "pnl" not in df.columns:
            continue
        ann = df.groupby("year")["pnl"].sum().reset_index()
        ann["version"] = v
        annual_frames.append(ann)

    if annual_frames:
        ann_df = pd.concat(annual_frames, ignore_index=True)
        fig_annual = go.Figure()
        for v in selected_versions:
            vdata = ann_df[ann_df["version"] == v]
            if vdata.empty:
                continue
            fig_annual.add_trace(go.Bar(
                x=vdata["year"],
                y=vdata["pnl"],
                name=VERSION_NAMES[v],
                marker_color=[COLORS[v] if p >= 0 else "#e74c3c" for p in vdata["pnl"]],
                text=[f"${p/1000:.0f}k" for p in vdata["pnl"]],
                textposition="outside",
            ))

        # Delta annotations (v3 - v1)
        if "v1" in selected_versions and "v3" in selected_versions:
            v1_ann = ann_df[ann_df["version"] == "v1"].set_index("year")["pnl"]
            v3_ann = ann_df[ann_df["version"] == "v3"].set_index("year")["pnl"]
            common_years = v1_ann.index.intersection(v3_ann.index)
            for yr in common_years:
                delta = v3_ann[yr] - v1_ann[yr]
                fig_annual.add_annotation(
                    x=yr, y=max(v1_ann[yr], v3_ann[yr]) + 20000,
                    text=f"Δ${delta/1000:+.0f}k",
                    showarrow=False, font=dict(size=9, color="#3498db"),
                )

        fig_annual.update_layout(
            title="Annual P&L by Version",
            xaxis_title="Year",
            yaxis_title="P&L ($)",
            barmode="group",
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_annual, use_container_width=True)
    else:
        st.info("No annual P&L data available.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRADE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Trade Analysis")

    # Combine selected version trades
    combined_frames = []
    for v in selected_versions:
        df = filter_trades_by_date_instrument(
            all_trades[v].copy(), date_start, date_end, instrument_filter
        )
        if not df.empty:
            combined_frames.append(df)
    combined = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()

    # ── Chart 4: Trade Scatter ────────────────────────────────────────────────
    st.subheader("Chart 4 — Trade Scatter (Duration vs P&L)")

    if not combined.empty and "duration_h" in combined.columns and "pnl" in combined.columns:
        scatter_df = combined.dropna(subset=["duration_h", "pnl", "instrument"])
        scatter_df = scatter_df[np.isfinite(scatter_df["duration_h"]) & np.isfinite(scatter_df["pnl"])]

        fig_scatter = go.Figure()
        for instr in ["ES", "NQ", "YM"]:
            sub = scatter_df[scatter_df["instrument"] == instr]
            if sub.empty:
                continue
            hover_text = []
            for _, row in sub.iterrows():
                et = str(row.get("Entry Time", ""))[:19]
                direction = row.get("Direction", "N/A")
                hover_text.append(
                    f"Entry: {et}<br>Instr: {instr}<br>Dir: {direction}<br>"
                    f"P&L: ${row['pnl']:,.0f}<br>Dur: {row['duration_h']:.1f}h"
                )
            fig_scatter.add_trace(go.Scatter(
                x=sub["duration_h"],
                y=sub["pnl"],
                mode="markers",
                name=instr,
                marker=dict(
                    size=np.clip(np.abs(sub["pnl"]) / 5000, 4, 20),
                    color=COLORS[instr],
                    opacity=0.65,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover_text,
                hovertemplate="%{text}<extra>" + instr + "</extra>",
            ))

        # Reference lines
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        fig_scatter.add_vline(x=24, line_dash="dash", line_color="gray", line_width=1,
                              annotation_text="24h", annotation_position="top")

        # Quadrant annotations
        for txt, ax, ay in [
            ("Long Winners", 0.75, 0.85),
            ("Short Winners", 0.05, 0.85),
            ("Short Losers", 0.05, 0.15),
            ("Long Losers", 0.75, 0.15),
        ]:
            fig_scatter.add_annotation(
                xref="paper", yref="paper", x=ax, y=ay,
                text=txt, showarrow=False,
                font=dict(size=10, color="gray"),
                opacity=0.6,
            )

        fig_scatter.update_layout(
            title="Trade Scatter: Duration vs P&L",
            xaxis_title="Duration (hours)",
            yaxis_title="P&L ($)",
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No trade scatter data available.")

    # ── Chart 5: Rolling Metrics ──────────────────────────────────────────────
    st.subheader("Chart 5 — Rolling Metrics (28-bar window)")

    roll_traces_added = False
    fig_roll = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        subplot_titles=["Rolling Sharpe & Win Rate", "Rolling Drawdown"],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    for v in selected_versions:
        df = filter_trades_by_date_instrument(
            all_trades[v].copy(), date_start, date_end, instrument_filter
        )
        if df.empty:
            continue
        rm = rolling_metrics(df)
        if rm.empty:
            continue
        roll_traces_added = True
        fig_roll.add_trace(
            go.Scatter(x=rm["Entry Time"], y=rm["roll_sharpe"],
                       name=f"{v} Sharpe", line=dict(color=COLORS[v], width=1.5)),
            row=1, col=1, secondary_y=False,
        )
        fig_roll.add_trace(
            go.Scatter(x=rm["Entry Time"], y=rm["roll_winrate"],
                       name=f"{v} Win%", line=dict(color=COLORS[v], width=1.5, dash="dot")),
            row=1, col=1, secondary_y=True,
        )
        eq = build_equity_curve(df)
        if not eq.empty:
            fig_roll.add_trace(
                go.Scatter(x=eq["Entry Time"], y=eq["drawdown_pct"],
                           name=f"{v} DD", fill="tozeroy",
                           line=dict(color=COLORS[v], width=1),
                           fillcolor=COLORS[v] + "33"),
                row=2, col=1,
            )

    if roll_traces_added:
        fig_roll.update_yaxes(title_text="Rolling Sharpe", row=1, col=1, secondary_y=False)
        fig_roll.update_yaxes(title_text="Win Rate (%)", row=1, col=1, secondary_y=True)
        fig_roll.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig_roll.update_layout(height=600, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.info("No rolling metrics data available.")

    # ── Chart 6: NQ Notional Exposure ─────────────────────────────────────────
    st.subheader("Chart 6 — NQ Notional Exposure Over Time")

    v3_trades = all_trades["v3"]
    if not v3_trades.empty and "instrument" in v3_trades.columns:
        nq_trades = v3_trades[v3_trades["instrument"] == "NQ"].copy()
        if not nq_trades.empty and "entry_price" in nq_trades.columns and "quantity" in nq_trades.columns:
            nq_trades = nq_trades.dropna(subset=["entry_price", "quantity"])
            nq_trades["notional"] = nq_trades["entry_price"] * nq_trades["quantity"] * 20
            nq_trades = nq_trades.sort_values("Entry Time")

            fig_notional = go.Figure()
            # Color by threshold
            colors_notional = ["#e74c3c" if n > 500_000 else "#2ecc71" for n in nq_trades["notional"]]
            fig_notional.add_trace(go.Scatter(
                x=nq_trades["Entry Time"],
                y=nq_trades["notional"],
                mode="markers+lines",
                marker=dict(color=colors_notional, size=5),
                line=dict(color="#95a5a6", width=1),
                name="NQ Notional",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Notional: $%{y:,.0f}<extra></extra>",
            ))

            # Reference lines
            fig_notional.add_hline(y=500_000, line_dash="dash", line_color="red",
                                   annotation_text="$500k Danger Zone", annotation_position="right")
            fig_notional.add_hline(y=300_000, line_dash="dash", line_color="orange",
                                   annotation_text="$300k Recommended Cap", annotation_position="right")

            # Highlight killer trades
            killer_dates = ["2024-10-14", "2024-06-18", "2024-12-09", "2024-12-11"]
            for kd in killer_dates:
                fig_notional.add_vline(
                    x=kd, line_dash="dot", line_color="red",
                    annotation_text=kd[5:], annotation_position="top",
                )

            fig_notional.update_layout(
                title="NQ Notional Exposure (v3 Trades) — Entry Price × Qty × 20",
                xaxis_title="Date",
                yaxis_title="Notional ($)",
                height=400,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_notional, use_container_width=True)
        else:
            st.info("Insufficient NQ trade data for notional calculation.")
    else:
        st.info("No v3 NQ trade data available.")

    # ── Chart 7: P&L Waterfall ────────────────────────────────────────────────
    st.subheader("Chart 7 — P&L Waterfall (Top 20 Biggest Trades)")

    if not combined.empty and "pnl" in combined.columns:
        top20 = combined.nlargest(10, "pnl")
        bot20 = combined.nsmallest(10, "pnl")
        waterfall_df = pd.concat([top20, bot20]).sort_values("pnl", ascending=False)
        waterfall_df = waterfall_df.dropna(subset=["pnl"])

        labels = []
        for _, row in waterfall_df.iterrows():
            instr = row.get("instrument", "?")
            et = str(row.get("Entry Time", ""))[:10]
            labels.append(f"{instr} {et}")

        fig_wf = go.Figure(go.Waterfall(
            name="P&L",
            orientation="v",
            measure=["relative"] * len(waterfall_df),
            x=labels,
            y=waterfall_df["pnl"].values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            text=[f"${p/1000:.0f}k" for p in waterfall_df["pnl"].values],
            textposition="outside",
        ))
        fig_wf.update_layout(
            title="P&L Waterfall — Top 20 Biggest Trades",
            xaxis_title="Trade",
            yaxis_title="P&L ($)",
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_wf, use_container_width=True)
    else:
        st.info("No waterfall data available.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — REGIME INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Regime Intelligence")

    if regimes.empty:
        st.warning("No regime data found at results/regimes_ES.csv")
    else:
        # ── Chart 8: Regime Pie + Timeline ────────────────────────────────────
        st.subheader("Chart 8 — Regime Distribution & Timeline")
        col_pie, col_timeline = st.columns([1, 2])

        with col_pie:
            regime_counts = regimes["regime"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=regime_counts.index,
                values=regime_counts.values,
                marker_colors=[REGIME_LABEL_COLORS.get(r, "#cccccc") for r in regime_counts.index],
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                title="Regime Time Distribution",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_timeline:
            # Gantt-style regime timeline
            reg_sorted = regimes.sort_values("date").copy()
            reg_sorted["grp"] = (reg_sorted["regime"] != reg_sorted["regime"].shift()).cumsum()

            gantt_rows = []
            for _, grp_df in reg_sorted.groupby("grp"):
                regime = grp_df["regime"].iloc[0]
                gantt_rows.append({
                    "regime": regime,
                    "start": grp_df["date"].iloc[0],
                    "end": grp_df["date"].iloc[-1],
                })
            gantt_df = pd.DataFrame(gantt_rows)

            fig_gantt = go.Figure()
            y_map = {"BULL": 1, "BEAR": 2, "SIDEWAYS": 3, "HIGH_VOLATILITY": 4}
            for _, row in gantt_df.iterrows():
                regime = row["regime"]
                y_pos = y_map.get(regime, 0)
                fig_gantt.add_trace(go.Scatter(
                    x=[row["start"], row["end"], row["end"], row["start"], row["start"]],
                    y=[y_pos - 0.4, y_pos - 0.4, y_pos + 0.4, y_pos + 0.4, y_pos - 0.4],
                    fill="toself",
                    fillcolor=REGIME_LABEL_COLORS.get(regime, "#cccccc"),
                    line=dict(width=0),
                    mode="lines",
                    name=regime,
                    showlegend=False,
                    hovertemplate=f"<b>{regime}</b><br>{row['start']:%Y-%m-%d} → {row['end']:%Y-%m-%d}<extra></extra>",
                ))

            # Add legend items
            for regime, color in REGIME_LABEL_COLORS.items():
                fig_gantt.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(size=10, color=color, symbol="square"),
                    name=regime, showlegend=True,
                ))

            fig_gantt.update_layout(
                title="Regime Timeline (2018–2024)",
                yaxis=dict(
                    tickvals=list(y_map.values()),
                    ticktext=list(y_map.keys()),
                    range=[0, 5],
                ),
                height=400,
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

        # ── Chart 9: Regime Transition Heatmap ────────────────────────────────
        st.subheader("Chart 9 — Regime Transition Heatmap")

        reg_for_trans = regimes.sort_values("date").copy()
        reg_for_trans["grp"] = (reg_for_trans["regime"] != reg_for_trans["regime"].shift()).cumsum()
        regime_seq = reg_for_trans.groupby("grp")["regime"].first().values

        regime_list = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"]
        trans_matrix = pd.DataFrame(0, index=regime_list, columns=regime_list)
        for i in range(len(regime_seq) - 1):
            fr = regime_seq[i]
            to = regime_seq[i + 1]
            if fr in regime_list and to in regime_list:
                trans_matrix.loc[fr, to] += 1

        fig_trans = go.Figure(go.Heatmap(
            z=trans_matrix.values,
            x=trans_matrix.columns.tolist(),
            y=trans_matrix.index.tolist(),
            colorscale="Blues",
            text=trans_matrix.values,
            texttemplate="%{text}",
            hovertemplate="From: %{y}<br>To: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_trans.update_layout(
            title="Regime Transition Heatmap",
            xaxis_title="To Regime",
            yaxis_title="From Regime",
            height=400,
        )
        st.plotly_chart(fig_trans, use_container_width=True)

        # ── Chart 10: P&L by Regime Box Plots ─────────────────────────────────
        st.subheader("Chart 10 — P&L by Regime (Box Plots)")

        if not combined.empty and "pnl" in combined.columns and "Entry Time" in combined.columns:
            # Join trades with regime by nearest date
            reg_daily = regimes.copy()
            reg_daily["date_only"] = pd.to_datetime(reg_daily["date"]).dt.date

            combined_with_regime = combined.copy()
            combined_with_regime["date_only"] = combined_with_regime["Entry Time"].dt.date
            combined_with_regime["date_ts"] = pd.to_datetime(combined_with_regime["date_only"])
            reg_daily2 = regimes.copy()
            reg_daily2["date_ts"] = pd.to_datetime(reg_daily2["date"])
            reg_daily2 = reg_daily2.sort_values("date_ts").drop_duplicates("date_ts")

            merged = pd.merge_asof(
                combined_with_regime.sort_values("date_ts"),
                reg_daily2[["date_ts", "regime"]],
                on="date_ts",
                direction="nearest",
            )

            fig_box = go.Figure()
            for regime in regime_list:
                sub = merged[merged["regime"] == regime]
                if sub.empty:
                    continue
                fig_box.add_trace(go.Box(
                    y=sub["pnl"],
                    name=regime,
                    marker_color=REGIME_LABEL_COLORS.get(regime, "#cccccc"),
                    boxmean="sd",
                ))
            fig_box.add_hline(y=0, line_dash="dash", line_color="black")
            fig_box.update_layout(
                title="P&L Distribution by Regime",
                yaxis_title="P&L ($)",
                height=450,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # ── Chart 11: Win Rate by Regime ─────────────────────────────────
            st.subheader("Chart 11 — Win Rate by Regime")

            fig_wr = go.Figure()
            for v in selected_versions:
                v_df = filter_trades_by_date_instrument(
                    all_trades[v].copy(), date_start, date_end, instrument_filter
                )
                if v_df.empty:
                    continue
                v_df["date_ts"] = pd.to_datetime(v_df["Entry Time"].dt.date)
                v_merged = pd.merge_asof(
                    v_df.sort_values("date_ts"),
                    reg_daily2[["date_ts", "regime"]],
                    on="date_ts",
                    direction="nearest",
                )
                if "is_win" not in v_merged.columns and "pnl" in v_merged.columns:
                    v_merged["is_win"] = v_merged["pnl"] > 0

                wr_by_regime = v_merged.groupby("regime")["is_win"].mean() * 100
                wr_by_regime = wr_by_regime.reindex(regime_list).dropna()

                fig_wr.add_trace(go.Bar(
                    x=wr_by_regime.index,
                    y=wr_by_regime.values,
                    name=VERSION_NAMES[v],
                    marker_color=COLORS[v],
                    text=[f"{w:.1f}%" for w in wr_by_regime.values],
                    textposition="outside",
                ))

            fig_wr.add_hline(y=50, line_dash="dash", line_color="gray",
                             annotation_text="50% baseline")
            fig_wr.update_layout(
                title="Win Rate by Regime",
                xaxis_title="Regime",
                yaxis_title="Win Rate (%)",
                barmode="group",
                height=400,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_wr, use_container_width=True)
        else:
            st.info("No combined trade data available for regime analysis.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — WELL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Well Analysis")

    if not wells:
        st.warning("No well data found at research/trade_analysis_data.json")
    else:
        wells_df = pd.DataFrame(wells)
        wells_df["start_dt"] = pd.to_datetime(wells_df["start"], utc=True, errors="coerce")
        wells_df["date"] = wells_df["start_dt"].dt.date
        wells_df["week"] = wells_df["start_dt"].dt.isocalendar().week.astype(int)
        wells_df["dayofweek"] = wells_df["start_dt"].dt.dayofweek
        wells_df["year"] = wells_df["start_dt"].dt.year
        wells_df["n_instruments"] = wells_df["instruments"].apply(
            lambda x: len(x) if isinstance(x, list) else 1
        )
        wells_df["total_pnl"] = pd.to_numeric(wells_df.get("total_pnl", wells_df.get("net_pnl", 0)), errors="coerce").fillna(0)

        # ── Chart 12: Well Calendar Heatmap ───────────────────────────────────
        st.subheader("Chart 12 — Well Calendar Heatmap (GitHub-style)")

        # Aggregate to daily P&L
        daily_wells = wells_df.groupby("date").agg(
            total_pnl=("total_pnl", "sum"),
            n_wells=("total_pnl", "count"),
        ).reset_index()
        daily_wells["date"] = pd.to_datetime(daily_wells["date"])
        daily_wells["week"] = daily_wells["date"].dt.isocalendar().week.astype(int)
        daily_wells["dow"] = daily_wells["date"].dt.dayofweek
        daily_wells["year"] = daily_wells["date"].dt.year

        fig_cal = px.scatter(
            daily_wells,
            x="week",
            y="dow",
            color="total_pnl",
            size=np.clip(np.abs(daily_wells["total_pnl"]) / 10000 + 3, 3, 20),
            color_continuous_scale="RdYlGn",
            facet_row="year",
            hover_data={"date": True, "total_pnl": True, "n_wells": True},
            title="Well P&L Calendar Heatmap",
            labels={"dow": "Day", "week": "Week of Year"},
        )
        fig_cal.update_yaxes(
            tickvals=[0, 1, 2, 3, 4],
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri"],
        )
        fig_cal.update_layout(height=800, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_cal, use_container_width=True)

        # ── Chart 13: Well Scatter (duration vs P&L by instrument count) ──────
        st.subheader("Chart 13 — Well Scatter (Duration vs P&L by Instrument Count)")

        color_map = {1: "#3498db", 2: "#f39c12", 3: "#e74c3c"}
        label_map = {1: "Single (1 instr)", 2: "Dual (2 instr)", 3: "Triple (3 instr)"}

        fig_well_scatter = go.Figure()
        for n_instr in [1, 2, 3]:
            sub = wells_df[wells_df["n_instruments"] == n_instr]
            if sub.empty:
                continue
            dur = sub.get("duration_h", pd.Series([1.0] * len(sub))).fillna(1.0)
            pnl_vals = sub["total_pnl"]
            fig_well_scatter.add_trace(go.Scatter(
                x=dur,
                y=pnl_vals,
                mode="markers",
                name=label_map[n_instr],
                marker=dict(
                    size=np.clip(np.abs(pnl_vals) / 5000, 5, 25),
                    color=color_map[n_instr],
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>Duration: %{x:.1f}h<br>P&L: $%{y:,.0f}<extra></extra>"
                ),
                text=[
                    f"{', '.join(row.get('instruments', []))} — {row.get('start','')[:10]}"
                    for _, row in sub.iterrows()
                ],
            ))

        fig_well_scatter.add_hline(y=0, line_dash="dash", line_color="black")
        fig_well_scatter.update_layout(
            title="Well Scatter: Duration vs P&L (by Instrument Count)",
            xaxis_title="Well Duration (hours)",
            yaxis_title="Total P&L ($)",
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_well_scatter, use_container_width=True)

        # ── Chart 14: Instrument Convergence Matrix ───────────────────────────
        st.subheader("Chart 14 — Instrument Convergence Matrix")

        pairs = [("ES", "NQ"), ("ES", "YM"), ("NQ", "YM")]
        pair_freq = {p: 0 for p in pairs}
        pair_pnl = {p: [] for p in pairs}

        for _, row in wells_df.iterrows():
            instrs = set(row.get("instruments", []))
            for p in pairs:
                if p[0] in instrs and p[1] in instrs:
                    pair_freq[p] += 1
                    pair_pnl[p].append(row["total_pnl"])

        pair_labels = [f"{a}×{b}" for a, b in pairs]
        freqs = [pair_freq[p] for p in pairs]
        avg_pnls = [np.mean(pair_pnl[p]) if pair_pnl[p] else 0 for p in pairs]

        # Build 3×3 heatmap matrix
        instruments = ["ES", "NQ", "YM"]
        freq_matrix = pd.DataFrame(0.0, index=instruments, columns=instruments)
        pnl_matrix = pd.DataFrame(0.0, index=instruments, columns=instruments)
        for i, p in enumerate(pairs):
            freq_matrix.loc[p[0], p[1]] = freqs[i]
            freq_matrix.loc[p[1], p[0]] = freqs[i]
            pnl_matrix.loc[p[0], p[1]] = avg_pnls[i]
            pnl_matrix.loc[p[1], p[0]] = avg_pnls[i]

        col_freq, col_pnl = st.columns(2)
        with col_freq:
            fig_cfreq = go.Figure(go.Heatmap(
                z=freq_matrix.values,
                x=freq_matrix.columns.tolist(),
                y=freq_matrix.index.tolist(),
                colorscale="Blues",
                text=freq_matrix.values.astype(int),
                texttemplate="%{text}",
                hovertemplate="<b>%{y} × %{x}</b><br>Co-occurrences: %{z}<extra></extra>",
            ))
            fig_cfreq.update_layout(title="Co-occurrence Frequency", height=350)
            st.plotly_chart(fig_cfreq, use_container_width=True)

        with col_pnl:
            fig_cpnl = go.Figure(go.Heatmap(
                z=pnl_matrix.values,
                x=pnl_matrix.columns.tolist(),
                y=pnl_matrix.index.tolist(),
                colorscale="RdYlGn",
                text=[[f"${v/1000:.0f}k" for v in row] for row in pnl_matrix.values],
                texttemplate="%{text}",
                hovertemplate="<b>%{y} × %{x}</b><br>Avg P&L: $%{z:,.0f}<extra></extra>",
            ))
            fig_cpnl.update_layout(title="Average P&L per Convergence", height=350)
            st.plotly_chart(fig_cpnl, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — STRATEGY COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Strategy Comparison")

    def compute_version_stats(v: str, start_dt, end_dt, instr: str) -> dict:
        df = filter_trades_by_date_instrument(
            all_trades[v].copy(), start_dt, end_dt, instr
        )
        if df.empty or "pnl" not in df.columns:
            return None
        eq = build_equity_curve(df)
        total_return_pct = (eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1) * 100 if len(eq) > 1 else 0
        max_dd = eq["drawdown_pct"].min() if not eq.empty else 0
        win_rate = (df["pnl"] > 0).mean() * 100
        avg_pnl = df["pnl"].mean()
        pnl_std = df["pnl"].std()
        sharpe = (avg_pnl / pnl_std * np.sqrt(252)) if pnl_std > 0 else 0
        trade_count = len(df)
        return {
            "version": v,
            "Return%": total_return_pct,
            "MaxDD%": abs(max_dd),
            "Sharpe": sharpe,
            "WinRate%": win_rate,
            "AvgTradePnL": avg_pnl,
            "TradeCount": trade_count,
        }

    stats_list = []
    for v in ["v1", "v2", "v3"]:
        s = compute_version_stats(v, date_start, date_end, instrument_filter)
        if s:
            stats_list.append(s)

    # ── Chart 15: Radar Chart ─────────────────────────────────────────────────
    st.subheader("Chart 15 — Version Comparison Radar")

    radar_axes = ["Return%", "MaxDD%", "Sharpe", "WinRate%", "AvgTradePnL", "TradeCount"]

    if stats_list:
        stats_df = pd.DataFrame(stats_list).set_index("version")

        # Normalize each axis to 0-1 range
        norm_df = stats_df[radar_axes].copy()
        for col in radar_axes:
            col_min = norm_df[col].min()
            col_max = norm_df[col].max()
            if col_max > col_min:
                norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
            else:
                norm_df[col] = 0.5

        fig_radar = go.Figure()
        for v, row in norm_df.iterrows():
            raw_row = stats_df.loc[v]
            vals = row.tolist()
            vals.append(vals[0])  # close the polygon
            axes_closed = radar_axes + [radar_axes[0]]
            hover_texts = [
                f"{ax}: {raw_row[ax]:.1f}" for ax in radar_axes
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=axes_closed,
                fill="toself",
                name=VERSION_NAMES[v],
                line=dict(color=COLORS[v]),
                fillcolor=COLORS[v] + "33",
                hovertemplate="<br>".join(hover_texts) + "<extra>" + VERSION_NAMES[v] + "</extra>",
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Version Comparison Radar",
            height=500,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Summary table
        st.dataframe(
            stats_df.style.format({
                "Return%": "{:.1f}%",
                "MaxDD%": "{:.1f}%",
                "Sharpe": "{:.2f}",
                "WinRate%": "{:.1f}%",
                "AvgTradePnL": "${:,.0f}",
                "TradeCount": "{:.0f}",
            }),
            use_container_width=True,
        )
    else:
        st.info("No stats available for radar chart.")

    # ── Chart 16: Bubble Chart ────────────────────────────────────────────────
    st.subheader("Chart 16 — Trade Count × Win Rate × Avg P&L Bubble Chart")

    bubble_rows = []
    for v in ["v1", "v2", "v3"]:
        df = filter_trades_by_date_instrument(
            all_trades[v].copy(), date_start, date_end, instrument_filter
        )
        if df.empty or "pnl" not in df.columns:
            continue
        for yr in sorted(df["year"].dropna().unique()):
            yr_df = df[df["year"] == yr]
            bubble_rows.append({
                "version": v,
                "year": int(yr),
                "trade_count": len(yr_df),
                "win_rate": (yr_df["pnl"] > 0).mean() * 100,
                "avg_pnl": yr_df["pnl"].mean(),
            })

    if bubble_rows:
        bdf = pd.DataFrame(bubble_rows)
        bdf["bubble_size"] = np.clip(np.abs(bdf["avg_pnl"]) / 1000, 5, 60)

        fig_bubble = go.Figure()
        for v in ["v1", "v2", "v3"]:
            sub = bdf[bdf["version"] == v]
            if sub.empty:
                continue
            fig_bubble.add_trace(go.Scatter(
                x=sub["trade_count"],
                y=sub["win_rate"],
                mode="markers+text",
                name=VERSION_NAMES[v],
                marker=dict(
                    size=sub["bubble_size"],
                    color=COLORS[v],
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=sub["year"].astype(str),
                textposition="top center",
                hovertemplate=(
                    "<b>%{text} — " + VERSION_NAMES[v] + "</b><br>"
                    "Trades: %{x}<br>Win Rate: %{y:.1f}%<br>"
                    "Avg P&L: $" + sub["avg_pnl"].apply(lambda x: f"{x:,.0f}").reset_index(drop=True).astype(str) +
                    "<extra></extra>"
                ) if False else (
                    "<b>%{text}</b><br>Trades: %{x}<br>Win Rate: %{y:.1f}%<extra></extra>"
                ),
            ))

        fig_bubble.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% win rate")
        fig_bubble.update_layout(
            title="Trade Count × Win Rate × Avg P&L (bubble size = avg P&L)",
            xaxis_title="Trade Count",
            yaxis_title="Win Rate (%)",
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.info("No bubble chart data available.")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"SRFM Trading Lab Dashboard | CF={cf_value:.3f} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
