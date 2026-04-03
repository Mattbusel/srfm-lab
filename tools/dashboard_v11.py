import sys
sys.stdout.reconfigure(encoding='utf-8')

import os, json, subprocess
from pathlib import Path

# ── Try to import streamlit; fall back to opening browser ───────────────────
try:
    import streamlit as st
    _STREAMLIT = True
except ImportError:
    _STREAMLIT = False

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
TOOLS  = ROOT / "tools"
RESULTS = ROOT / "results"

TOOL_CONFIGS = [
    {"key": "suite_v11",        "label": "Suite v11",        "script": "tools/suite_v11.py",        "json": "results/suite_v11.json",        "png": "results/suite_v11.png"},
    {"key": "stress_test",      "label": "Stress Test",      "script": "tools/stress_test.py",       "json": "results/stress_test.json",       "png": None},
    {"key": "margin_sim",       "label": "Margin Sim",       "script": "tools/margin_sim.py",        "json": "results/margin_sim.json",        "png": None},
    {"key": "corr_monitor",     "label": "Corr Monitor",     "script": "tools/corr_monitor.py",      "json": "results/corr_monitor.json",      "png": None},
    {"key": "equity_paths",     "label": "Equity Paths",     "script": "tools/equity_paths.py",      "json": "results/equity_paths.json",      "png": "results/equity_paths.png"},
    {"key": "size_replay",      "label": "Size Replay",      "script": "tools/size_replay.py",       "json": "results/size_replay.json",       "png": None},
    {"key": "regime_stress",    "label": "Regime Stress",    "script": "tools/regime_stress.py",     "json": "results/regime_stress.json",     "png": None},
    {"key": "risk_sensitivity", "label": "Risk Sensitivity", "script": "tools/risk_sensitivity.py",  "json": "results/risk_sensitivity.json",  "png": "results/risk_sensitivity.png"},
    {"key": "dd_decomp",        "label": "DD Decomp",        "script": "tools/dd_decomp.py",         "json": "results/dd_decomp.json",         "png": "results/dd_decomp.png"},
    {"key": "fee_impact",       "label": "Fee Impact",       "script": "tools/fee_impact.py",        "json": "results/fee_impact.json",        "png": "results/fee_impact.png"},
    {"key": "live_check",       "label": "Live Check",       "script": "tools/live_check.py",        "json": "results/live_check.json",        "png": "results/live_check.png"},
]

def load_json(path):
    p = ROOT / path
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def run_tool(script):
    return subprocess.run(
        [sys.executable, str(ROOT / script)],
        capture_output=True, text=True, cwd=str(ROOT)
    )

# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT MODE
# ─────────────────────────────────────────────────────────────────────────────
if _STREAMLIT:
    st.set_page_config(
        page_title="LARSA v11 Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .verdict-ready    { color: #00c853; font-weight: bold; }
    .verdict-caution  { color: #ffd600; font-weight: bold; }
    .verdict-notready { color: #f44336; font-weight: bold; }
    .metric-box { background: #1e1e2e; border-radius: 8px; padding: 12px; margin: 4px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("LARSA v11 Lab")
    st.sidebar.markdown("---")

    selected = st.sidebar.radio(
        "Select analysis",
        [tc["label"] for tc in TOOL_CONFIGS],
        index=0,
    )
    st.sidebar.markdown("---")

    # Per-tool run button
    active_tc = next(tc for tc in TOOL_CONFIGS if tc["label"] == selected)
    if st.sidebar.button(f"Run: {active_tc['label']}"):
        with st.spinner(f"Running {active_tc['script']}..."):
            proc = run_tool(active_tc["script"])
        if proc.returncode == 0:
            st.sidebar.success("Done.")
        else:
            st.sidebar.error("Failed — see output below.")
            st.sidebar.code(proc.stderr[-2000:] if proc.stderr else "No stderr")

    st.sidebar.markdown("---")

    # Run All
    if st.sidebar.button("Run ALL tools"):
        progress = st.sidebar.progress(0)
        for i, tc in enumerate(TOOL_CONFIGS):
            st.sidebar.write(f"Running {tc['label']}...")
            run_tool(tc["script"])
            progress.progress((i+1)/len(TOOL_CONFIGS))
        st.sidebar.success("All tools complete.")
        st.rerun()

    # ── Status overview ───────────────────────────────────────────────────────
    st.sidebar.markdown("**Result status:**")
    for tc in TOOL_CONFIGS:
        exists = (ROOT / tc["json"]).exists()
        icon = "✓" if exists else "○"
        st.sidebar.markdown(f"  {icon} {tc['label']}")

    # ── Main panel ────────────────────────────────────────────────────────────
    st.title(f"LARSA v11 — {selected}")

    data = load_json(active_tc["json"])

    if data is None:
        st.info(f"No results yet. Click **Run: {selected}** in the sidebar to generate them.")
    else:
        # ── Suite v11 ─────────────────────────────────────────────────────────
        if active_tc["key"] == "suite_v11":
            if isinstance(data, dict):
                cols = st.columns(4)
                for i, (k, v) in enumerate(data.items()):
                    if isinstance(v, (int, float)):
                        cols[i % 4].metric(k, f"{v:.2f}" if isinstance(v, float) else str(v))
            st.json(data)

        # ── Risk sensitivity ──────────────────────────────────────────────────
        elif active_tc["key"] == "risk_sensitivity":
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Optimal PDR (max Sharpe)",
                          f"{data.get('efficient_frontier_pdr', 'N/A'):.4f}")
                st.metric("Optimal Sharpe",
                          f"{data.get('efficient_frontier_sharpe', 0):.2f}")
            with col2:
                ruin = data.get("ruin_breakeven_pdr")
                st.metric("Ruin PDR (p90 DD≥30%)", f"{ruin:.4f}" if ruin else "Not reached")
                st.metric("Worlds per PDR", data.get("n_worlds", "?"))
            if "sweep" in data:
                import pandas as pd
                df = pd.DataFrame(data["sweep"])
                df = df.round(3)
                st.dataframe(df, use_container_width=True)

        # ── DD Decomp ─────────────────────────────────────────────────────────
        elif active_tc["key"] == "dd_decomp":
            col1, col2, col3 = st.columns(3)
            v11 = data.get("v11", {})
            col1.metric("v11 Return%",  f"{v11.get('ret%', 0):+.1f}%")
            col2.metric("v11 Sharpe",   f"{v11.get('sharpe', 0):.2f}")
            col3.metric("v11 Max DD%",  f"{v11.get('max_dd%', 0):.1f}%")
            st.metric("Time underwater", f"{data.get('underwater_fraction', 0)*100:.1f}%")
            st.metric("Avg recovery (1% DD)", f"{data.get('avg_recovery_1pct', 0):.0f} bars")
            if "drawdowns" in data:
                st.markdown(f"**Drawdowns > 3%: {len(data['drawdowns'])}**")
                for i, dd in enumerate(data["drawdowns"], 1):
                    with st.expander(f"DD #{i}: {dd['dd_pct']:.1f}% at bar {dd['peak_idx']}"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Duration", f"{dd['duration']} bars")
                        c2.metric("Recovery", f"{dd.get('recovery_bars','N/A')} bars" if dd.get("recovered") else "Not recovered")
                        c3.metric("ATR elevated", "Yes" if dd.get("atr_elevated") else "No")
                        ic = dd.get("inst_contrib", {})
                        st.write(f"ES: ${ic.get('ES',0):+,.0f}  NQ: ${ic.get('NQ',0):+,.0f}  YM: ${ic.get('YM',0):+,.0f}")

        # ── Fee impact ────────────────────────────────────────────────────────
        elif active_tc["key"] == "fee_impact":
            import pandas as pd
            col1, col2 = st.columns(2)
            col1.metric("Sharpe=0 at fee mult", f"{data.get('sharpe_breakeven_mult', 'N/A')}×")
            col2.metric("B&H underperform at", f"{data.get('bh_underperform_mult', 'N/A')}×")
            col1.metric("v11/v9 trade ratio", f"{data.get('v11_vs_v9_trade_ratio', 0):.2f}×")
            col2.metric("B&H median return", f"{data.get('bh_median_ret', 0):+.1f}%")
            if "version_comparison" in data:
                st.markdown("**Version comparison at base fees:**")
                df = pd.DataFrame(data["version_comparison"]).round(2)
                st.dataframe(df, use_container_width=True)
            if "fee_sweep_v11" in data:
                st.markdown("**Fee sweep (v11):**")
                df = pd.DataFrame(data["fee_sweep_v11"]).round(2)
                st.dataframe(df, use_container_width=True)

        # ── Live check ────────────────────────────────────────────────────────
        elif active_tc["key"] == "live_check":
            base = data.get("baseline", {})
            st.markdown(f"**Baseline (perfect):** ret={base.get('med_ret',0):+.1f}%  "
                        f"Sharpe={base.get('med_sharpe',0):.2f}  DD={base.get('med_dd',0):.1f}%")
            st.markdown("---")
            for sc in data.get("scenarios", []):
                v = sc.get("verdict", "?")
                color_cls = {"READY": "verdict-ready", "CAUTION": "verdict-caution",
                             "NOT READY": "verdict-notready"}.get(v, "")
                with st.expander(f"{sc['name']}  —  {v}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Return%",        f"{sc['med_ret']:+.1f}%")
                    c2.metric("Sharpe",         f"{sc['med_sharpe']:.2f}")
                    c3.metric("Max DD%",        f"{sc['med_dd']:.1f}%")
                    c1.metric("Ret Degrade%",   f"{sc['ret_degrade']:+.1f}%")
                    c2.metric("Sharpe Drop",    f"{sc['sh_degrade']:+.2f}")
                    c3.metric("DD Increase%",   f"{sc['dd_increase']:+.1f}%")
                    st.markdown(f"Delay={sc['delay']} bars | Slippage={sc['slip']*100:.3f}% | Reject={sc['reject_rate']*100:.0f}%")

        # ── Generic fallback ──────────────────────────────────────────────────
        else:
            st.json(data)

    # ── Chart panel ───────────────────────────────────────────────────────────
    if active_tc.get("png"):
        png_path = ROOT / active_tc["png"]
        if png_path.exists():
            st.markdown("---")
            st.image(str(png_path), use_container_width=True)
        elif data is not None:
            st.info("Chart not yet generated — run the tool to create it.")

# ─────────────────────────────────────────────────────────────────────────────
#  FALLBACK: open in browser via subprocess
# ─────────────────────────────────────────────────────────────────────────────
else:
    import webbrowser, threading, time

    def _launch():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8501")

    print("Streamlit not found — launching via 'streamlit run'...")
    threading.Thread(target=_launch, daemon=True).start()
    os.execv(
        sys.executable,
        [sys.executable, "-m", "streamlit", "run", __file__, "--server.port", "8501"]
    )
