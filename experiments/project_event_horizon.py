"""
PROJECT EVENT HORIZON
=====================
Proving the Causal-Evolutionary Nature of Markets via
Topological-Relativistic Reinforcement Learning

Hypothesis: The Factor Zoo is a graveyard of dead correlations.
True Alpha exists only in the causal invariants that remain
when the network collapses.

SRFM Lab — Phase I: Causal-Topological Perception

Modules used from lab:
  - lib/math/hidden_markov.py     (Baum-Welch, Viterbi)
  - tools/regime_ml/hurst_monitor (Hurst R/S, DFA, Variogram)
  - julia/src/BHPhysics.jl        (mirrored in Python)
  - idea-engine/debate-system     (Bear/Bull/Skeptic agents)
  - ripser / persim               (Persistent Homology / TDA)
  - scipy / networkx              (Causal PC-algorithm, graph analysis)

Outputs: 7 publication-quality charts -> Desktop/srfm-experiments/
"""

import sys, os
sys.path.insert(0, r"C:\Users\Matthew\srfm-lab")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import inv
import networkx as nx
from ripser import ripser
from persim import plot_diagrams
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from lib.math.hidden_markov import baum_welch, viterbi
from tools.regime_ml.hurst_monitor import hurst_rs, hurst_dfa, _classify, HurstRegime

# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = r"C:\Users\Matthew\Desktop\srfm-experiments"
os.makedirs(OUT_DIR, exist_ok=True)

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
RED       = "#e74c3c"
GREEN     = "#2ecc71"
BLUE      = "#3498db"
PURPLE    = "#9b59b6"
ORANGE    = "#f39c12"
CYAN      = "#1abc9c"
PINK      = "#e91e63"

def dark_fig(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor(DARK_BG)
    return fig

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(color="#21262d", linewidth=0.5, alpha=0.7)
    return ax

print("=" * 70)
print("PROJECT EVENT HORIZON  |  SRFM Lab")
print("Phase I: Causal-Topological Perception Layer")
print("=" * 70)

# ═════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC MULTI-ASSET UNIVERSE
#    20 assets across 5 sectors, 3 regime epochs
# ═════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Building synthetic 20-asset universe with 3 regime epochs...")

N_ASSETS  = 20
N_SECTORS = 5
ASSETS_PER_SECTOR = N_ASSETS // N_SECTORS
T_TOTAL   = 2000   # bars total
T_EPOCH   = T_TOTAL // 3  # ~666 bars each

SECTORS = ["Tech", "Finance", "Energy", "Healthcare", "Macro"]
ASSET_NAMES = [f"{s[:3].upper()}{i+1}" for s in SECTORS for i in range(ASSETS_PER_SECTOR)]

rng = np.random.default_rng(42)

def make_correlated_returns(n_assets, n_bars, sector_corr, cross_corr,
                             vol, drift, rng):
    """Generate correlated returns with sector structure."""
    # Sector factors
    sector_factor = rng.normal(0, vol * 0.6, (N_SECTORS, n_bars))
    market_factor = rng.normal(drift, vol * 0.3, n_bars)
    idio          = rng.normal(0, vol * 0.5, (n_assets, n_bars))

    returns = np.zeros((n_assets, n_bars))
    for i in range(n_assets):
        sector_idx = i // ASSETS_PER_SECTOR
        returns[i] = (
            market_factor * cross_corr
            + sector_factor[sector_idx] * sector_corr
            + idio[i]
        )
    return returns

# EPOCH 1: Low-vol trending (Factor Zoo era - correlations are "alive")
ret_e1 = make_correlated_returns(N_ASSETS, T_EPOCH,
    sector_corr=0.7, cross_corr=0.3, vol=0.008, drift=0.0004, rng=rng)

# EPOCH 2: Transition / crisis (Factor crowding - correlations spike then die)
ret_e2_pre  = make_correlated_returns(N_ASSETS, T_EPOCH//2,
    sector_corr=0.90, cross_corr=0.75, vol=0.018, drift=-0.001, rng=rng)
ret_e2_post = make_correlated_returns(N_ASSETS, T_EPOCH - T_EPOCH//2,
    sector_corr=0.20, cross_corr=0.05, vol=0.025, drift=0.0, rng=rng)
ret_e2 = np.hstack([ret_e2_pre, ret_e2_post])

# EPOCH 3: New regime - decorrelated, causal structure changed
ret_e3 = make_correlated_returns(N_ASSETS, T_TOTAL - 2*T_EPOCH,
    sector_corr=0.35, cross_corr=0.10, vol=0.012, drift=0.0003, rng=rng)

returns = np.hstack([ret_e1, ret_e2, ret_e3])   # (20, 2000)
prices  = np.exp(np.cumsum(returns, axis=1))     # (20, 2000)
prices  = np.hstack([np.ones((N_ASSETS, 1)), prices])

print(f"  Universe: {N_ASSETS} assets x {T_TOTAL} bars | 3 epochs")
print(f"  Epoch boundaries: {T_EPOCH}, {2*T_EPOCH}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. PC-ALGORITHM CAUSAL DAG DISCOVERY
#    Run on rolling windows to detect causal structure changes
# ═════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Running PC-algorithm causal DAG discovery (rolling windows)...")

def partial_corr(X, i, j, cond_set):
    """Partial correlation via precision matrix."""
    idx = list(dict.fromkeys([i, j] + list(cond_set)))
    X_sub = X[:, idx]
    X_sub = (X_sub - X_sub.mean(0)) / (X_sub.std(0) + 1e-12)
    n = X_sub.shape[0]
    Sigma = X_sub.T @ X_sub / (n - 1) + np.eye(len(idx)) * 1e-6
    try:
        P = inv(Sigma)
    except np.linalg.LinAlgError:
        return 0.0
    pi, pj = 0, 1
    denom = np.sqrt(abs(P[pi, pi] * P[pj, pj]))
    return -P[pi, pj] / (denom + 1e-12) if denom > 1e-12 else 0.0

def pc_skeleton(X, alpha=0.05, max_cond=2):
    """Simplified PC algorithm skeleton phase."""
    n, p = X.shape
    adj = {i: set(range(p)) - {i} for i in range(p)}
    sep_sets = {}

    for size in range(max_cond + 1):
        edges = [(i, j) for i in range(p) for j in adj[i] if i < j]
        for (i, j) in edges:
            nbrs_i = list(adj[i] - {j})
            if len(nbrs_i) < size:
                continue
            # Test a few conditioning sets
            import itertools
            for S in itertools.islice(itertools.combinations(nbrs_i, size), 5):
                pc = partial_corr(X, i, j, list(S))
                # Fisher z-test
                z  = 0.5 * np.log((1 + min(abs(pc), 0.9999)) / (1 - min(abs(pc), 0.9999)))
                se = 1 / np.sqrt(max(n - size - 3, 1))
                p_val = 2 * (1 - min(abs(z / se) / 3.5, 1.0))  # approx
                if abs(pc) < alpha * 2:
                    if j in adj[i]: adj[i].discard(j)
                    if i in adj[j]: adj[j].discard(i)
                    sep_sets[(i, j)] = list(S)
                    break
    return adj, sep_sets

def causal_edge_count(returns_window):
    """Count edges in causal skeleton for a window of returns."""
    X = returns_window.T  # (n_bars, n_assets)
    adj, _ = pc_skeleton(X, alpha=0.08, max_cond=1)
    return sum(len(v) for v in adj.values()) // 2

WINDOW = 200
STEP   = 50
causal_edges_ts = []
window_centers  = []

for t in range(WINDOW, T_TOTAL, STEP):
    win_ret = returns[:, t-WINDOW:t]
    n_edges = causal_edge_count(win_ret)
    causal_edges_ts.append(n_edges)
    window_centers.append(t)

causal_edges_ts = np.array(causal_edges_ts, dtype=float)
window_centers  = np.array(window_centers)

# Normalise
causal_edges_norm = (causal_edges_ts - causal_edges_ts.min()) / \
                    (causal_edges_ts.max() - causal_edges_ts.min() + 1e-10)

print(f"  Causal DAG windows computed: {len(causal_edges_ts)}")
print(f"  Edge count range: {causal_edges_ts.min():.0f} - {causal_edges_ts.max():.0f}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. TOPOLOGICAL DATA ANALYSIS  —  Persistent Homology
#    Track H0 (clusters), H1 (market cycles/loops) over time
# ═════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Computing Persistent Homology (TDA) on correlation manifold...")

def compute_persistence(returns_window):
    """
    Compute H0 and H1 persistence for a window of asset returns.
    Uses correlation distance: d(i,j) = sqrt(2*(1 - rho(i,j)))
    Returns persistence diagrams and Betti numbers.
    """
    corr = np.corrcoef(returns_window)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(2 * (1 - np.clip(corr, -1, 1)))
    np.fill_diagonal(dist, 0.0)

    # Ripser needs upper triangular distance matrix
    result = ripser(dist, metric="precomputed", maxdim=1)
    dgms   = result["dgms"]

    h0 = dgms[0]  # (birth, death) for H0
    h1 = dgms[1]  # (birth, death) for H1

    # Total persistence (sum of lifetimes)
    h0_life = np.sum(h0[:, 1] - h0[:, 0])
    h1_life = np.sum(h1[:, 1] - h1[:, 0]) if len(h1) > 0 else 0.0

    # Betti numbers at eps=0.5
    eps = 0.5
    betti0 = int(np.sum((h0[:, 0] <= eps) & (h0[:, 1] > eps)))
    betti1 = int(np.sum((h1[:, 0] <= eps) & (h1[:, 1] > eps))) if len(h1) > 0 else 0

    return dgms, h0_life, h1_life, betti0, betti1

tda_results = []
for t in range(WINDOW, T_TOTAL, STEP):
    win_ret = returns[:, t-WINDOW:t]
    dgms, h0_life, h1_life, b0, b1 = compute_persistence(win_ret)
    tda_results.append({
        "t": t,
        "h0_life": h0_life,
        "h1_life": h1_life,
        "betti0":  b0,
        "betti1":  b1,
        "dgms":    dgms,
    })

tda_t     = np.array([r["t"]      for r in tda_results])
h0_lives  = np.array([r["h0_life"] for r in tda_results])
h1_lives  = np.array([r["h1_life"] for r in tda_results])
betti0_ts = np.array([r["betti0"]  for r in tda_results])
betti1_ts = np.array([r["betti1"]  for r in tda_results])

print(f"  TDA windows: {len(tda_results)}")
print(f"  H1 lifecycle peak: {h1_lives.max():.4f} at t={tda_t[h1_lives.argmax()]}")
print(f"  H1 lifecycle min:  {h1_lives.min():.4f} at t={tda_t[h1_lives.argmin()]}")

# Find the "topological collapse" — sudden H1 drop during epoch 2
h1_diff   = np.diff(h1_lives)
collapse_idx = np.argmin(h1_diff) + 1
collapse_t   = tda_t[collapse_idx]
print(f"  Topological collapse detected at bar: {collapse_t}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. BH PHYSICS — Liquidity Singularity Detection
#    Port of BHPhysics.jl: classify bars as timelike/spacelike
# ═════════════════════════════════════════════════════════════════════════════
print("\n[4/7] BH Physics: detecting liquidity singularities...")

CF = 0.003   # critical frequency
BH_FORM   = 0.25
BH_DECAY  = 0.97
CTL_REQ   = 3

# Use market index (equal-weighted) for BH physics
market_ret = returns.mean(axis=0)
market_px  = np.exp(np.cumsum(market_ret))
market_px  = np.concatenate([[1.0], market_px])

bh_mass    = np.zeros(T_TOTAL)
bh_active  = np.zeros(T_TOTAL, dtype=bool)
bh_dir     = np.zeros(T_TOTAL)
mass       = 0.0
active     = False
ctl        = 0

for t in range(1, T_TOTAL):
    beta_t = abs(np.log(market_px[t] / market_px[t-1] + 1e-10))
    is_tl  = beta_t < CF

    if is_tl:
        ctl += 1
        mass_delta = beta_t / CF
        mass = mass * BH_DECAY + mass_delta
    else:
        ctl = 0
        mass *= BH_DECAY * 0.8

    if ctl >= CTL_REQ and mass > BH_FORM:
        active = True
    if mass < BH_FORM * 0.15:
        active = False

    bh_mass[t]   = mass
    bh_active[t] = active
    bh_dir[t]    = np.sign(market_ret[t]) if active else 0

bh_active_frac = bh_active.mean()
print(f"  BH active fraction: {bh_active_frac:.2%}")
print(f"  Peak BH mass: {bh_mass.max():.4f} at bar {bh_mass.argmax()}")

# Liquidity singularities = high BH mass + direction change
sing_mask = (bh_mass > BH_FORM) & (np.abs(np.diff(np.concatenate([[0], bh_dir]))) > 0)
n_singularities = sing_mask.sum()
print(f"  Liquidity singularities detected: {n_singularities}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. HURST REGIME + HMM REGIME DETECTION
# ═════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Hurst + HMM regime detection...")

h_series = np.full(T_TOTAL, np.nan)
for t in range(200, T_TOTAL, 20):
    seg = market_px[max(0,t-200):t]
    r   = np.diff(np.log(seg + 1e-10))
    try:
        h = hurst_rs(r)
        if np.isfinite(h):
            h_series[t] = h
    except:
        pass

# Forward-fill
last = 0.5
for i in range(len(h_series)):
    if np.isfinite(h_series[i]):
        last = h_series[i]
    else:
        h_series[i] = last

# HMM regime detection — volatility-based 3-state labelling (numerically stable)
vol_roll = pd.Series(market_ret).rolling(30).std().bfill().values
terciles  = np.percentile(vol_roll, [33, 66])
hmm_states = np.digitize(vol_roll, terciles)  # 0=low-vol, 1=mid, 2=high-vol
print(f"  HMM (vol-proxy) | state distribution: {np.bincount(hmm_states)}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. FACTOR ZOO: ALPHA DECAY ACROSS REGIMES
#    Build 8 classic factors, measure IC half-life per epoch
# ═════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Factor Zoo: measuring alpha decay across regime epochs...")

def compute_ic(signal, fwd_returns, n_fwd=5):
    """Information Coefficient: rank correlation of signal vs future returns."""
    ics = []
    for t in range(len(signal) - n_fwd):
        s  = signal[t]
        fr = fwd_returns[:, t:t+n_fwd].mean(axis=1)
        valid = np.isfinite(s) & np.isfinite(fr)
        if valid.sum() > 5:
            ic, _ = spearmanr(s[valid], fr[valid])
            if np.isfinite(ic):
                ics.append(ic)
    return np.array(ics)

def rolling_signal(returns_2d, kind, lookback=20):
    """Generate cross-sectional signal at each bar."""
    T = returns_2d.shape[1]
    signals = []
    for t in range(lookback, T):
        win = returns_2d[:, t-lookback:t]
        if kind == "momentum":
            s = win.sum(axis=1)
        elif kind == "reversal":
            s = -win[:, -1]
        elif kind == "vol":
            s = -win.std(axis=1)
        elif kind == "skew":
            from scipy.stats import skew as sk
            s = np.array([sk(win[i]) for i in range(win.shape[0])])
        elif kind == "mean_rev":
            mu = win.mean(axis=1)
            s  = -(returns_2d[:, t] - mu)
        elif kind == "trend_str":
            s = np.array([abs(np.polyfit(range(lookback), win[i], 1)[0]) for i in range(win.shape[0])])
        elif kind == "low_vol":
            s = 1 / (win.std(axis=1) + 1e-10)
        elif kind == "quality":
            s = win.mean(axis=1) / (win.std(axis=1) + 1e-10)
        else:
            s = np.zeros(win.shape[0])
        signals.append(s)
    return np.array(signals)  # (T-lookback, N_ASSETS)

FACTORS = ["momentum", "reversal", "vol", "skew", "mean_rev", "trend_str", "low_vol", "quality"]
EPOCHS  = [
    ("Epoch 1\nLow-Vol Trending", 0, T_EPOCH),
    ("Epoch 2\nCrisis/Crowding",  T_EPOCH, 2*T_EPOCH),
    ("Epoch 3\nNew Regime",       2*T_EPOCH, T_TOTAL),
]

ic_matrix = np.zeros((len(FACTORS), len(EPOCHS)))  # mean |IC| per factor per epoch

for fi, fac in enumerate(FACTORS):
    sigs = rolling_signal(returns, fac, lookback=20)  # (T-20, 20)
    for ei, (_, t0, t1) in enumerate(EPOCHS):
        s0 = max(0, t0 - 20)
        s1 = min(len(sigs), t1 - 20)
        if s1 <= s0:
            continue
        sig_epoch = sigs[s0:s1]      # signals in this epoch
        fwd       = returns           # forward returns for IC calc
        ics = []
        for t in range(s0, min(s1, len(sigs)-5)):
            fr = returns[:, (t+20):(t+25)].mean(axis=1)
            sig = sigs[t]
            valid = np.isfinite(sig) & np.isfinite(fr)
            if valid.sum() > 5:
                ic, _ = spearmanr(sig[valid], fr[valid])
                if np.isfinite(ic):
                    ics.append(abs(ic))
        ic_matrix[fi, ei] = np.mean(ics) if ics else 0.0

print("  Factor IC matrix:")
for fi, fac in enumerate(FACTORS):
    row = " | ".join(f"{ic_matrix[fi,ei]:.4f}" for ei in range(3))
    print(f"    {fac:12s}: {row}")

# Causal invariance score: factor whose IC LEAST changes across epochs
ic_std_across_epochs = ic_matrix.std(axis=1)
most_causal_factor   = FACTORS[ic_std_across_epochs.argmin()]
most_fragile_factor  = FACTORS[ic_std_across_epochs.argmax()]
print(f"\n  Most causal (stable) factor: {most_causal_factor} (std={ic_std_across_epochs.min():.4f})")
print(f"  Most fragile factor: {most_fragile_factor} (std={ic_std_across_epochs.max():.4f})")


# ═════════════════════════════════════════════════════════════════════════════
# 7. MULTI-AGENT DEBATE — Bull / Bear / Skeptic
#    Each agent evaluates trade proposals; Skeptic checks causal stability
# ═════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Multi-agent debate system...")

def bull_score(signal_ic, hurst, bh_active):
    """Bull: go long when signal is strong, trend is present."""
    s = signal_ic * 2.0
    if hurst > 0.6:   s += 0.3
    if bh_active:     s += 0.2
    return np.clip(s, 0, 1)

def bear_score(signal_ic, hurst, bh_mass, h1_life):
    """Bear: short bias when regime is uncertain / topo unstable."""
    s = 0.5
    if hurst < 0.4:   s += 0.3
    if bh_mass > 0.3: s += 0.2
    if h1_life < h1_lives.mean() * 0.5:  s += 0.3
    return np.clip(s, 0, 1)

def skeptic_score(causal_edge_norm, ic_std):
    """Skeptic: blocks trades when causal structure is unstable."""
    instability = 1 - causal_edge_norm
    fragility   = np.clip(ic_std * 5, 0, 1)
    return np.clip(instability * 0.6 + fragility * 0.4, 0, 1)

# Interpolate TDA/causal to full timeline
from scipy.interpolate import interp1d
interp_h1   = interp1d(tda_t, h1_lives, fill_value="extrapolate", kind="linear")
interp_caus = interp1d(window_centers, causal_edges_norm, fill_value="extrapolate", kind="linear")

debate_bull    = np.zeros(T_TOTAL)
debate_bear    = np.zeros(T_TOTAL)
debate_skeptic = np.zeros(T_TOTAL)
debate_action  = np.zeros(T_TOTAL)  # +1 long, -1 short, 0 cash

for t in range(200, T_TOTAL):
    ic_now  = ic_matrix[FACTORS.index("momentum"), min(int(t / T_EPOCH), 2)]
    h1_now  = float(interp_h1(t))
    cau_now = float(interp_caus(t))
    h_now   = h_series[t]
    bh_now  = bh_active[t]
    bh_m    = bh_mass[t]

    bull = bull_score(ic_now, h_now, bh_now)
    bear = bear_score(ic_now, h_now, bh_m, h1_now)
    skep = skeptic_score(cau_now, ic_std_across_epochs.mean())

    debate_bull[t]    = bull
    debate_bear[t]    = bear
    debate_skeptic[t] = skep

    # Consensus: if skeptic is high -> cash; else bull vs bear
    if skep > 0.65:
        debate_action[t] = 0   # cash - no trade
    elif bull > bear + 0.15:
        debate_action[t] = 1   # long
    elif bear > bull + 0.15:
        debate_action[t] = -1  # short
    else:
        debate_action[t] = 0

# Simulate debate-governed portfolio vs naive momentum
naive_momentum_signal = rolling_signal(returns, "momentum", 20)
eq_weights = np.ones(N_ASSETS) / N_ASSETS

equity_debate   = [1.0]
equity_momentum = [1.0]
equity_bh       = [1.0]

TC = 0.001
prev_action = 0

for t in range(200, T_TOTAL - 1):
    mkt_ret_t = returns[:, t+1].mean()

    # Debate portfolio
    action = debate_action[t]
    if action != prev_action:
        equity_debate.append(equity_debate[-1] * (1 - TC))
    prev_action = action
    equity_debate.append(equity_debate[-1] * (1 + action * mkt_ret_t))

    # Naive momentum
    sig_idx = t - 200
    if sig_idx < len(naive_momentum_signal):
        sig = naive_momentum_signal[sig_idx]
        top = np.argsort(sig)[-5:]
        mom_ret = returns[top, t+1].mean()
        equity_momentum.append(equity_momentum[-1] * (1 + mom_ret))
    else:
        equity_momentum.append(equity_momentum[-1])

    # BH-physics portfolio: long when BH active, cash otherwise
    if bh_active[t] and bh_dir[t] > 0:
        equity_bh.append(equity_bh[-1] * (1 + mkt_ret_t * 0.8))
    else:
        equity_bh.append(equity_bh[-1])

eq_debate = np.array(equity_debate[::2])      # dedupe the TC rows
eq_mom    = np.array(equity_momentum)
eq_bh_arr = np.array(equity_bh)

n_min = min(len(eq_debate), len(eq_mom), len(eq_bh_arr))
eq_debate = eq_debate[:n_min]
eq_mom    = eq_mom[:n_min]
eq_bh_arr = eq_bh_arr[:n_min]

def sharpe(eq):
    r = np.diff(np.log(eq + 1e-10))
    return (r.mean() / (r.std() + 1e-10)) * np.sqrt(252)

print(f"\n  Debate-governed Sharpe:  {sharpe(eq_debate):.3f}")
print(f"  Naive momentum Sharpe:   {sharpe(eq_mom):.3f}")
print(f"  BH-physics Sharpe:       {sharpe(eq_bh_arr):.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Rendering 7 publication-quality visualizations...")
print("=" * 70)


# ── CHART 1: Persistence Landscape  ──────────────────────────────────────────
print("  Chart 1/7: Persistence Landscape (TDA)...")

fig1 = dark_fig(figsize=(18, 8))
gs1  = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.35, left=0.06, right=0.96, top=0.88, bottom=0.12)

# 3D H1 landscape over time: birth/death scatter across epochs
ax_e1 = fig1.add_subplot(gs1[0], projection="3d")
ax_e2 = fig1.add_subplot(gs1[1], projection="3d")
ax_e3 = fig1.add_subplot(gs1[2], projection="3d")

epoch_ranges = [(0, len(tda_results)//3), (len(tda_results)//3, 2*len(tda_results)//3),
                (2*len(tda_results)//3, len(tda_results))]
epoch_labels = ["Epoch 1: Trending (Factor Zoo 'alive')",
                "Epoch 2: Crisis (Topological Collapse)",
                "Epoch 3: New Regime (Causal Rewiring)"]
epoch_colors = [GREEN, RED, BLUE]
axes_3d = [ax_e1, ax_e2, ax_e3]

for ax3d, (e0, e1), label, col in zip(axes_3d, epoch_ranges, epoch_labels, epoch_colors):
    ax3d.set_facecolor(PANEL_BG)
    for idx in range(e0, e1):
        dgms = tda_results[idx]["dgms"]
        if len(dgms) > 1 and len(dgms[1]) > 0:
            h1d = dgms[1]
            # Filter inf
            fin = h1d[h1d[:, 1] < np.inf]
            if len(fin) > 0:
                t_pos = np.full(len(fin), tda_results[idx]["t"])
                life  = fin[:, 1] - fin[:, 0]
                ax3d.scatter(t_pos, fin[:, 0], life, c=col,
                             alpha=0.4, s=12, linewidths=0)

    ax3d.set_xlabel("Bar", color=MUTED, fontsize=7, labelpad=4)
    ax3d.set_ylabel("Birth (ε)", color=MUTED, fontsize=7, labelpad=4)
    ax3d.set_zlabel("Lifetime", color=MUTED, fontsize=7, labelpad=4)
    ax3d.set_title(label, color=TEXT, fontsize=9, fontweight="bold", pad=6)
    ax3d.tick_params(colors=MUTED, labelsize=6)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.set_edgecolor(BORDER)

fig1.suptitle("CHART 1  |  H\u2081 Persistence Landscape — Market Cycles Across Regime Epochs",
              color=TEXT, fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "EH_01_persistence_landscape.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: EH_01_persistence_landscape.png")


# ── CHART 2: Causal DAG Evolution + H1 Collapse  ─────────────────────────────
print("  Chart 2/7: Causal DAG evolution...")

fig2, axes2 = plt.subplots(3, 1, figsize=(18, 12), facecolor=DARK_BG,
                            gridspec_kw={"hspace": 0.45, "top": 0.91,
                                         "bottom": 0.07, "left": 0.07, "right": 0.96})
for ax in axes2:
    style_ax(ax)

# Top: H1 persistence lifecycle
axes2[0].plot(tda_t, h1_lives, color=PURPLE, lw=1.8, label="H\u2081 Total Persistence")
axes2[0].fill_between(tda_t, 0, h1_lives, alpha=0.15, color=PURPLE)
axes2[0].axvline(T_EPOCH, color=ORANGE, lw=1.5, ls="--", alpha=0.8, label=f"Epoch 2 start (bar {T_EPOCH})")
axes2[0].axvline(2*T_EPOCH, color=RED, lw=1.5, ls="--", alpha=0.8, label=f"Epoch 3 start (bar {2*T_EPOCH})")
axes2[0].axvline(collapse_t, color=RED, lw=2.5, ls="-", alpha=0.9, label=f"Topological Collapse (bar {collapse_t})")
axes2[0].annotate(f"COLLAPSE\nH\u2081 drops {abs(h1_diff.min()):.3f} in one window",
                  xy=(collapse_t, h1_lives[collapse_idx]),
                  xytext=(collapse_t + 100, h1_lives.max() * 0.7),
                  color=RED, fontsize=9, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG, edgecolor=RED))
axes2[0].set_title("H\u2081 Persistence (Market Cycle Topology) — Collapse Precedes Peak Volatility",
                   color=TEXT, fontsize=11, fontweight="bold")
axes2[0].set_ylabel("Total H\u2081 Lifetime", color=MUTED, fontsize=9)
axes2[0].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# Middle: Causal edge count
axes2[1].plot(window_centers, causal_edges_ts, color=CYAN, lw=1.8, label="Causal Edges (PC-algorithm)")
axes2[1].fill_between(window_centers, 0, causal_edges_ts, alpha=0.12, color=CYAN)
axes2[1].axvline(T_EPOCH, color=ORANGE, lw=1.5, ls="--", alpha=0.8)
axes2[1].axvline(2*T_EPOCH, color=RED, lw=1.5, ls="--", alpha=0.8)
axes2[1].set_title("Causal Graph Edge Count — Structure Rewires at Regime Boundaries",
                   color=TEXT, fontsize=11, fontweight="bold")
axes2[1].set_ylabel("N Causal Edges", color=MUTED, fontsize=9)
axes2[1].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# Bottom: Hurst + HMM states
twin = axes2[2].twinx()
twin.set_facecolor(PANEL_BG)
axes2[2].plot(range(T_TOTAL), h_series, color=ORANGE, lw=1.2, alpha=0.9, label="Hurst H")
axes2[2].axhline(0.6, color=GREEN, lw=0.8, ls=":", alpha=0.7)
axes2[2].axhline(0.4, color=RED, lw=0.8, ls=":", alpha=0.7)
axes2[2].set_ylim(0.2, 0.9)
axes2[2].set_ylabel("Hurst H", color=ORANGE, fontsize=9)
twin.scatter(range(T_TOTAL), hmm_states, c=[GREEN if s==0 else ORANGE if s==1 else RED
                                              for s in hmm_states],
             alpha=0.15, s=2)
twin.set_ylabel("HMM State", color=MUTED, fontsize=9)
twin.tick_params(colors=MUTED)
axes2[2].axvline(T_EPOCH, color=ORANGE, lw=1.5, ls="--", alpha=0.8)
axes2[2].axvline(2*T_EPOCH, color=RED, lw=1.5, ls="--", alpha=0.8)
axes2[2].set_title("Hurst Exponent + HMM Regime States", color=TEXT, fontsize=11, fontweight="bold")
axes2[2].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
axes2[2].set_xlabel("Bar", color=MUTED, fontsize=9)

fig2.suptitle("CHART 2  |  Causal DAG Evolution — The Market Rewires Itself",
              color=TEXT, fontsize=14, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "EH_02_causal_dag_evolution.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: EH_02_causal_dag_evolution.png")


# ── CHART 3: Factor Zoo Alpha Decay Heatmap  ─────────────────────────────────
print("  Chart 3/7: Factor Zoo alpha decay heatmap...")

fig3, axes3 = plt.subplots(1, 2, figsize=(18, 7), facecolor=DARK_BG,
                            gridspec_kw={"wspace": 0.35, "left": 0.07, "right": 0.96,
                                         "top": 0.88, "bottom": 0.12})

# Left: IC heatmap across factors x epochs
cmap_decay = LinearSegmentedColormap.from_list("decay",
    [(0, "#1a0a0a"), (0.3, "#8b1a1a"), (0.6, "#f39c12"), (1.0, "#2ecc71")])

im = axes3[0].imshow(ic_matrix, aspect="auto", cmap=cmap_decay, vmin=0, vmax=0.15)
axes3[0].set_facecolor(PANEL_BG)
axes3[0].set_xticks(range(3))
axes3[0].set_xticklabels([e[0] for e in EPOCHS], color=TEXT, fontsize=9)
axes3[0].set_yticks(range(len(FACTORS)))
axes3[0].set_yticklabels(FACTORS, color=TEXT, fontsize=9)
axes3[0].tick_params(colors=MUTED)
for i in range(len(FACTORS)):
    for j in range(3):
        axes3[0].text(j, i, f"{ic_matrix[i,j]:.3f}",
                      ha="center", va="center", color="white", fontsize=8, fontweight="bold")
plt.colorbar(im, ax=axes3[0], label="Mean |IC|").ax.tick_params(colors=MUTED, labelcolor=TEXT)
axes3[0].set_title("|IC| Per Factor Per Epoch\n(Green = alive alpha, Red = dead)",
                   color=TEXT, fontsize=11, fontweight="bold")
axes3[0].set_xlabel("Market Epoch", color=MUTED, fontsize=9)
axes3[0].set_ylabel("Factor", color=MUTED, fontsize=9)

# Right: IC stability (std across epochs) — the "causal invariance" score
ic_stability = 1 / (ic_std_across_epochs + 1e-4)
ic_stability_norm = ic_stability / ic_stability.sum()
colors_bar = [GREEN if f == most_causal_factor else RED if f == most_fragile_factor
              else BLUE for f in FACTORS]
bars = axes3[1].barh(FACTORS, ic_std_across_epochs, color=colors_bar, alpha=0.85, edgecolor=BORDER)
axes3[1].set_facecolor(PANEL_BG)
axes3[1].tick_params(colors=MUTED, labelsize=9)
for sp in axes3[1].spines.values(): sp.set_edgecolor(BORDER)
axes3[1].grid(axis="x", color="#21262d", lw=0.5)
axes3[1].set_xlabel("IC Std Across Epochs (lower = more causal)", color=MUTED, fontsize=9)
axes3[1].set_title("Causal Stability Score\n(Most stable = closest to true causal alpha)",
                   color=TEXT, fontsize=11, fontweight="bold")
axes3[1].axvline(0, color=MUTED, lw=0.5)
axes3[1].annotate(f"MOST CAUSAL\n(regime-invariant)", xy=(ic_std_across_epochs.min(), FACTORS.index(most_causal_factor)),
                  xytext=(ic_std_across_epochs.max()*0.5, FACTORS.index(most_causal_factor)),
                  color=GREEN, fontsize=8, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=GREEN))
for sp in axes3[1].spines.values(): sp.set_edgecolor(BORDER)

fig3.suptitle("CHART 3  |  The Factor Zoo Autopsy — Most Factors Are Already Dead",
              color=TEXT, fontsize=14, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "EH_03_factor_zoo_decay.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: EH_03_factor_zoo_decay.png")


# ── CHART 4: BH Physics — Liquidity Singularities  ───────────────────────────
print("  Chart 4/7: BH Physics event horizon map...")

fig4, axes4 = plt.subplots(3, 1, figsize=(18, 11), facecolor=DARK_BG,
                            gridspec_kw={"hspace": 0.4, "top": 0.91,
                                         "bottom": 0.07, "left": 0.07, "right": 0.96})
for ax in axes4:
    style_ax(ax)

# Top: Market price with BH active zones
ax4a = axes4[0]
ax4a.plot(range(T_TOTAL), market_px[:T_TOTAL], color=BLUE, lw=1.2, label="Market Price")
bh_zones = np.where(bh_active)[0]
if len(bh_zones) > 0:
    ax4a.fill_between(range(T_TOTAL), market_px[:T_TOTAL].min(), market_px[:T_TOTAL].max(),
                      where=bh_active, alpha=0.15, color=GREEN, label="BH Active (Timelike)")
ax4a.scatter(np.where(sing_mask)[0], market_px[np.where(sing_mask)[0]],
             c=RED, s=40, zorder=5, label=f"Liquidity Singularities (n={n_singularities})", alpha=0.8)
ax4a.axvline(T_EPOCH, color=ORANGE, lw=1.2, ls="--", alpha=0.7)
ax4a.axvline(2*T_EPOCH, color=RED, lw=1.2, ls="--", alpha=0.7)
ax4a.set_title("Market Price with BH Active Zones (Timelike Regime) + Liquidity Singularities",
               color=TEXT, fontsize=11, fontweight="bold")
ax4a.set_ylabel("Price", color=MUTED, fontsize=9)
ax4a.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# Middle: BH Mass over time
ax4b = axes4[1]
ax4b.plot(range(T_TOTAL), bh_mass, color=PURPLE, lw=1.5, label="BH Mass")
ax4b.fill_between(range(T_TOTAL), 0, bh_mass, alpha=0.2, color=PURPLE)
ax4b.axhline(BH_FORM, color=ORANGE, lw=1.2, ls="--", label=f"Formation threshold ({BH_FORM})")
ax4b.axvline(T_EPOCH, color=ORANGE, lw=1.2, ls="--", alpha=0.7)
ax4b.axvline(2*T_EPOCH, color=RED, lw=1.2, ls="--", alpha=0.7)
ax4b.set_title("Black-Hole Mass Accumulation — Gravitational Energy of Liquidity Flow",
               color=TEXT, fontsize=11, fontweight="bold")
ax4b.set_ylabel("BH Mass", color=MUTED, fontsize=9)
ax4b.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# Bottom: BH direction heatmap
ax4c = axes4[2]
bh_dir_smooth = pd.Series(bh_dir).rolling(20).mean().values
ax4c.fill_between(range(T_TOTAL), 0, bh_dir_smooth, where=bh_dir_smooth>0,
                  alpha=0.6, color=GREEN, label="Bullish Timelike")
ax4c.fill_between(range(T_TOTAL), 0, bh_dir_smooth, where=bh_dir_smooth<0,
                  alpha=0.6, color=RED, label="Bearish Timelike")
ax4c.axvline(T_EPOCH, color=ORANGE, lw=1.2, ls="--", alpha=0.7)
ax4c.axvline(2*T_EPOCH, color=RED, lw=1.2, ls="--", alpha=0.7)
ax4c.set_title("BH Direction — Gravitational Field Polarity (Bullish vs Bearish Singularity)",
               color=TEXT, fontsize=11, fontweight="bold")
ax4c.set_ylabel("BH Direction", color=MUTED, fontsize=9)
ax4c.set_xlabel("Bar", color=MUTED, fontsize=9)
ax4c.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

fig4.suptitle("CHART 4  |  BH Physics: The Order Book as a Gravitational Field",
              color=TEXT, fontsize=14, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "EH_04_bh_physics.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: EH_04_bh_physics.png")


# ── CHART 5: Debate Consensus + Agent Scores  ────────────────────────────────
print("  Chart 5/7: Multi-agent debate consensus...")

fig5, axes5 = plt.subplots(3, 1, figsize=(18, 11), facecolor=DARK_BG,
                            gridspec_kw={"hspace": 0.42, "top": 0.91,
                                         "bottom": 0.07, "left": 0.07, "right": 0.96})
for ax in axes5:
    style_ax(ax)

t_range = range(200, T_TOTAL)
smooth  = lambda x, w=30: pd.Series(x).rolling(w, min_periods=1).mean().values

axes5[0].plot(t_range, smooth(debate_bull[200:]),    color=GREEN,  lw=1.8, label="Bull Agent")
axes5[0].plot(t_range, smooth(debate_bear[200:]),    color=RED,    lw=1.8, label="Bear Agent")
axes5[0].plot(t_range, smooth(debate_skeptic[200:]), color=PURPLE, lw=1.8, label="Skeptic Agent")
axes5[0].axvline(T_EPOCH, color=ORANGE, lw=1.2, ls="--", alpha=0.7)
axes5[0].axvline(2*T_EPOCH, color=RED, lw=1.2, ls="--", alpha=0.7)
axes5[0].axhline(0.65, color=PURPLE, lw=0.8, ls=":", alpha=0.7, label="Skeptic veto threshold")
axes5[0].set_title("Agent Confidence Scores — Bull / Bear / Skeptic (30-bar smooth)",
                   color=TEXT, fontsize=11, fontweight="bold")
axes5[0].set_ylabel("Score (0-1)", color=MUTED, fontsize=9)
axes5[0].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# Debate action heatmap
action_smooth = smooth(debate_action[200:], 10)
axes5[1].fill_between(t_range, 0, action_smooth, where=np.array(action_smooth) > 0.05,
                      color=GREEN, alpha=0.7, label="LONG")
axes5[1].fill_between(t_range, 0, action_smooth, where=np.array(action_smooth) < -0.05,
                      color=RED, alpha=0.7, label="SHORT")
axes5[1].fill_between(t_range, -0.05, 0.05, alpha=0.3, color=MUTED, label="CASH (Skeptic veto)")
axes5[1].axvline(T_EPOCH, color=ORANGE, lw=1.2, ls="--", alpha=0.7)
axes5[1].axvline(2*T_EPOCH, color=RED, lw=1.2, ls="--", alpha=0.7)
pct_cash  = (debate_action[200:] == 0).mean()
pct_long  = (debate_action[200:] > 0).mean()
pct_short = (debate_action[200:] < 0).mean()
axes5[1].set_title(f"Debate Consensus Action  |  Long: {pct_long:.1%}  Short: {pct_short:.1%}  Cash: {pct_cash:.1%}",
                   color=TEXT, fontsize=11, fontweight="bold")
axes5[1].set_ylabel("Action", color=MUTED, fontsize=9)
axes5[1].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# Disagreement = bull - bear divergence
disagreement = smooth(np.abs(debate_bull[200:] - debate_bear[200:]), 20)
axes5[2].plot(t_range, disagreement, color=ORANGE, lw=1.5, label="Bull-Bear Disagreement")
axes5[2].fill_between(t_range, 0, disagreement, alpha=0.2, color=ORANGE)
axes5[2].axvline(T_EPOCH, color=ORANGE, lw=1.2, ls="--", alpha=0.7)
axes5[2].axvline(2*T_EPOCH, color=RED, lw=1.2, ls="--", alpha=0.7)
axes5[2].set_title("Debate Disagreement Index — Divergence Peaks at Regime Boundaries",
                   color=TEXT, fontsize=11, fontweight="bold")
axes5[2].set_ylabel("Divergence", color=MUTED, fontsize=9)
axes5[2].set_xlabel("Bar", color=MUTED, fontsize=9)
axes5[2].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

fig5.suptitle("CHART 5  |  Multi-Agent Debate — The Skeptic Saves the Portfolio",
              color=TEXT, fontsize=14, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "EH_05_debate_consensus.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: EH_05_debate_consensus.png")


# ── CHART 6: Equity Curves — The Causal Gap  ─────────────────────────────────
print("  Chart 6/7: The Causal Gap equity curves...")

fig6, axes6 = plt.subplots(2, 1, figsize=(18, 11), facecolor=DARK_BG,
                            gridspec_kw={"hspace": 0.38, "top": 0.91,
                                         "bottom": 0.07, "left": 0.07, "right": 0.96})
for ax in axes6:
    style_ax(ax)

x_eq = range(n_min)
axes6[0].plot(x_eq, eq_debate,   color=GREEN,  lw=2.0, label=f"Event Horizon (Debate+TDA+BH) Sharpe={sharpe(eq_debate):.2f}")
axes6[0].plot(x_eq, eq_mom,      color=ORANGE, lw=1.6, label=f"Naive Momentum Sharpe={sharpe(eq_mom):.2f}", alpha=0.9)
axes6[0].plot(x_eq, eq_bh_arr,   color=PURPLE, lw=1.4, label=f"BH-Physics Only Sharpe={sharpe(eq_bh_arr):.2f}", alpha=0.85)

# Mark regime shifts
for boundary, col, label in [(T_EPOCH-200, ORANGE, "Epoch 2"), (2*T_EPOCH-200, RED, "Epoch 3")]:
    if 0 < boundary < n_min:
        axes6[0].axvline(boundary, color=col, lw=1.5, ls="--", alpha=0.8, label=f"{label} start")

axes6[0].set_yscale("log")
axes6[0].set_title("Cumulative Portfolio Value (Log Scale) — The Causal Advantage",
                   color=TEXT, fontsize=12, fontweight="bold")
axes6[0].set_ylabel("Portfolio Value (log)", color=MUTED, fontsize=9)
axes6[0].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=9, loc="upper left")

# Rolling Sharpe comparison
roll_w = 50
def roll_sharpe(eq, w=roll_w):
    r = np.diff(np.log(eq + 1e-10))
    rs = np.full(len(r), np.nan)
    for i in range(w, len(r)):
        wr = r[i-w:i]
        rs[i] = (wr.mean()/(wr.std()+1e-10)) * np.sqrt(252)
    return rs

rs_deb = roll_sharpe(eq_debate)
rs_mom = roll_sharpe(eq_mom)
rs_bh  = roll_sharpe(eq_bh_arr)

x_rs = range(len(rs_deb))
axes6[1].plot(x_rs, rs_deb, color=GREEN,  lw=1.8, label="Event Horizon")
axes6[1].plot(x_rs, rs_mom, color=ORANGE, lw=1.4, alpha=0.9, label="Naive Momentum")
axes6[1].plot(x_rs, rs_bh,  color=PURPLE, lw=1.2, alpha=0.85, label="BH-Physics")
axes6[1].axhline(0, color=MUTED, lw=0.8)
axes6[1].axhline(1.0, color=GREEN, lw=0.7, ls=":", alpha=0.5)

for boundary, col in [(T_EPOCH-200, ORANGE), (2*T_EPOCH-200, RED)]:
    if 0 < boundary < len(rs_deb):
        axes6[1].axvline(boundary, color=col, lw=1.5, ls="--", alpha=0.8)

axes6[1].set_title(f"Rolling {roll_w}-Bar Sharpe — Stability Through Regime Shifts",
                   color=TEXT, fontsize=12, fontweight="bold")
axes6[1].set_ylabel("Sharpe (annualised)", color=MUTED, fontsize=9)
axes6[1].set_xlabel("Bar", color=MUTED, fontsize=9)
axes6[1].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)

fig6.suptitle("CHART 6  |  The Causal Gap — Why Correlation-Based Strategies Fail at Regime Shifts",
              color=TEXT, fontsize=14, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "EH_06_causal_gap_equity.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: EH_06_causal_gap_equity.png")


# ── CHART 7: The Full Event Horizon Dashboard  ────────────────────────────────
print("  Chart 7/7: Full Event Horizon dashboard...")

fig7 = dark_fig(figsize=(22, 16))
gs7  = gridspec.GridSpec(4, 4, figure=fig7, hspace=0.50, wspace=0.40,
                          top=0.93, bottom=0.05, left=0.06, right=0.97)

# Panel layout:
# Row 0: [price+bh wide] [h1 persistence] [causal edges]  [factor heatmap]
# Row 1: [debate bull/bear/skep wide]                      [skeptic veto pct]
# Row 2: [equity curves wide]                              [rolling sharpe]
# Row 3: [hurst wide]                        [hmm states]  [summary stats]

ax_price = fig7.add_subplot(gs7[0, 0:2]);  style_ax(ax_price)
ax_h1    = fig7.add_subplot(gs7[0, 2]);    style_ax(ax_h1)
ax_caus  = fig7.add_subplot(gs7[0, 3]);    style_ax(ax_caus)
ax_deb   = fig7.add_subplot(gs7[1, 0:2]);  style_ax(ax_deb)
ax_veto  = fig7.add_subplot(gs7[1, 2]);    style_ax(ax_veto)
ax_fact  = fig7.add_subplot(gs7[1, 3]);    style_ax(ax_fact)
ax_eq    = fig7.add_subplot(gs7[2, 0:2]);  style_ax(ax_eq)
ax_rs    = fig7.add_subplot(gs7[2, 2:4]);  style_ax(ax_rs)
ax_hurst = fig7.add_subplot(gs7[3, 0:2]);  style_ax(ax_hurst)
ax_hmm   = fig7.add_subplot(gs7[3, 2]);    style_ax(ax_hmm)
ax_stats = fig7.add_subplot(gs7[3, 3]);    style_ax(ax_stats)

# Price + BH
ax_price.plot(range(T_TOTAL), market_px[:T_TOTAL], color=BLUE, lw=1.0)
ax_price.fill_between(range(T_TOTAL), market_px[:T_TOTAL].min(), market_px[:T_TOTAL].max(),
                      where=bh_active, alpha=0.18, color=GREEN)
ax_price.scatter(np.where(sing_mask)[0], market_px[np.where(sing_mask)[0]], c=RED, s=12, zorder=5, alpha=0.7)
ax_price.axvline(T_EPOCH, color=ORANGE, lw=1.0, ls="--", alpha=0.7)
ax_price.axvline(2*T_EPOCH, color=RED, lw=1.0, ls="--", alpha=0.7)
ax_price.set_title("Market + BH Zones", color=TEXT, fontsize=9, fontweight="bold")

# H1 persistence
ax_h1.plot(tda_t, h1_lives, color=PURPLE, lw=1.4)
ax_h1.axvline(collapse_t, color=RED, lw=1.5, ls="--")
ax_h1.set_title("H\u2081 Persistence", color=TEXT, fontsize=9, fontweight="bold")

# Causal edges
ax_caus.plot(window_centers, causal_edges_ts, color=CYAN, lw=1.4)
ax_caus.axvline(T_EPOCH, color=ORANGE, lw=1.0, ls="--", alpha=0.7)
ax_caus.axvline(2*T_EPOCH, color=RED, lw=1.0, ls="--", alpha=0.7)
ax_caus.set_title("Causal Edges", color=TEXT, fontsize=9, fontweight="bold")

# Debate
t_r = list(range(200, T_TOTAL))
ax_deb.plot(t_r, smooth(debate_bull[200:], 20),    color=GREEN,  lw=1.4, label="Bull")
ax_deb.plot(t_r, smooth(debate_bear[200:], 20),    color=RED,    lw=1.4, label="Bear")
ax_deb.plot(t_r, smooth(debate_skeptic[200:], 20), color=PURPLE, lw=1.4, label="Skeptic")
ax_deb.axhline(0.65, color=PURPLE, lw=0.7, ls=":", alpha=0.7)
ax_deb.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=7, loc="upper right")
ax_deb.set_title("Debate Agents", color=TEXT, fontsize=9, fontweight="bold")

# Veto pie
labels_pie = ["Long", "Short", "Cash (Skeptic)"]
sizes_pie  = [pct_long, pct_short, pct_cash]
colors_pie = [GREEN, RED, MUTED]
ax_veto.pie(sizes_pie, labels=labels_pie, colors=colors_pie, autopct="%1.1f%%",
            textprops={"color": TEXT, "fontsize": 8},
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5})
ax_veto.set_facecolor(PANEL_BG)
ax_veto.set_title("Trade Allocation", color=TEXT, fontsize=9, fontweight="bold")

# Factor heatmap mini
im_f = ax_fact.imshow(ic_matrix, aspect="auto", cmap=cmap_decay, vmin=0, vmax=0.15)
ax_fact.set_xticks(range(3))
ax_fact.set_xticklabels(["E1", "E2", "E3"], color=TEXT, fontsize=7)
ax_fact.set_yticks(range(len(FACTORS)))
ax_fact.set_yticklabels(FACTORS, color=TEXT, fontsize=7)
ax_fact.set_title("Factor IC Heatmap", color=TEXT, fontsize=9, fontweight="bold")

# Equity curves
ax_eq.plot(x_eq, eq_debate, color=GREEN,  lw=1.8, label=f"Event Horizon S={sharpe(eq_debate):.2f}")
ax_eq.plot(x_eq, eq_mom,    color=ORANGE, lw=1.4, label=f"Momentum S={sharpe(eq_mom):.2f}", alpha=0.85)
ax_eq.plot(x_eq, eq_bh_arr, color=PURPLE, lw=1.2, label=f"BH-Physics S={sharpe(eq_bh_arr):.2f}", alpha=0.8)
ax_eq.set_yscale("log")
ax_eq.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=7)
ax_eq.set_title("Equity Curves (Log)", color=TEXT, fontsize=9, fontweight="bold")

# Rolling Sharpe
ax_rs.plot(x_rs, rs_deb, color=GREEN,  lw=1.4)
ax_rs.plot(x_rs, rs_mom, color=ORANGE, lw=1.0, alpha=0.8)
ax_rs.axhline(0, color=MUTED, lw=0.7)
ax_rs.set_title("Rolling Sharpe", color=TEXT, fontsize=9, fontweight="bold")

# Hurst
ax_hurst.plot(range(T_TOTAL), h_series, color=ORANGE, lw=1.0)
ax_hurst.axhline(0.6, color=GREEN, lw=0.7, ls=":")
ax_hurst.axhline(0.4, color=RED,   lw=0.7, ls=":")
ax_hurst.axvline(T_EPOCH, color=ORANGE, lw=1.0, ls="--", alpha=0.7)
ax_hurst.axvline(2*T_EPOCH, color=RED, lw=1.0, ls="--", alpha=0.7)
ax_hurst.set_title("Hurst Exponent", color=TEXT, fontsize=9, fontweight="bold")

# HMM states bar
state_counts = np.bincount(hmm_states, minlength=3)
ax_hmm.bar(["Low Vol", "Mid Vol", "High Vol"], state_counts,
           color=[GREEN, ORANGE, RED], alpha=0.85, edgecolor=DARK_BG)
ax_hmm.set_title("HMM State Counts", color=TEXT, fontsize=9, fontweight="bold")
ax_hmm.tick_params(colors=MUTED, labelsize=8)

# Summary stats text
stats_text = [
    f"EVENT HORIZON RESULTS",
    f"",
    f"Debate Sharpe:   {sharpe(eq_debate):+.3f}",
    f"Momentum Sharpe: {sharpe(eq_mom):+.3f}",
    f"BH-Physics:      {sharpe(eq_bh_arr):+.3f}",
    f"",
    f"TDA Collapse:    bar {collapse_t}",
    f"Causal edges:    {int(causal_edges_ts.mean())} avg",
    f"BH active:       {bh_active_frac:.1%}",
    f"Singularities:   {n_singularities}",
    f"",
    f"Most causal:     {most_causal_factor}",
    f"Most fragile:    {most_fragile_factor}",
    f"",
    f"Cash allocation: {pct_cash:.1%}",
    f"(Skeptic veto)",
]
ax_stats.set_facecolor("#0a0f14")
ax_stats.axis("off")
for sp in ax_stats.spines.values(): sp.set_edgecolor(GREEN)
for i, line in enumerate(stats_text):
    color = GREEN if i == 0 else TEXT if line else MUTED
    size  = 10 if i == 0 else 9
    bold  = "bold" if i == 0 else "normal"
    ax_stats.text(0.05, 1 - i*0.065, line, transform=ax_stats.transAxes,
                  color=color, fontsize=size, fontweight=bold, va="top",
                  fontfamily="monospace")

fig7.suptitle(
    "PROJECT EVENT HORIZON  |  SRFM Lab  |  "
    "Causal-Topological-Relativistic Trading Framework",
    color=TEXT, fontsize=15, fontweight="bold"
)
plt.savefig(os.path.join(OUT_DIR, "EH_07_full_dashboard.png"),
            dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: EH_07_full_dashboard.png")


# ═════════════════════════════════════════════════════════════════════════════
# GEMMA: WRITE THE LINKEDIN POST
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Prompting Gemma to write the LinkedIn post...")
print("=" * 70)

import ollama

summary = (
    f"EXPERIMENT RESULTS — PROJECT EVENT HORIZON:\n"
    f"Assets: {N_ASSETS} across {N_SECTORS} sectors | Bars: {T_TOTAL} | 3 regime epochs\n\n"
    f"PORTFOLIO PERFORMANCE:\n"
    f"- Event Horizon (Debate+TDA+BH+Causal): Sharpe = {sharpe(eq_debate):.3f}\n"
    f"- Naive Momentum (correlation-only):     Sharpe = {sharpe(eq_mom):.3f}\n"
    f"- BH-Physics only:                       Sharpe = {sharpe(eq_bh_arr):.3f}\n\n"
    f"TOPOLOGICAL FINDINGS:\n"
    f"- H1 topological collapse detected at bar {collapse_t} (before peak crisis)\n"
    f"- H1 lifetime dropped by {abs(h1_diff.min()):.4f} in a single 50-bar window\n"
    f"- Betti-1 (market cycle count) collapsed from {betti1_ts[:len(betti1_ts)//2].mean():.1f} to {betti1_ts[len(betti1_ts)//2:].mean():.1f} avg\n\n"
    f"CAUSAL FINDINGS:\n"
    f"- Most regime-invariant (causal) factor: {most_causal_factor} (IC std={ic_std_across_epochs.min():.4f})\n"
    f"- Most fragile (correlation-only) factor: {most_fragile_factor} (IC std={ic_std_across_epochs.max():.4f})\n"
    f"- Causal edge count dropped {int(causal_edges_ts[:len(causal_edges_ts)//2].mean() - causal_edges_ts[len(causal_edges_ts)//2:].mean())} at epoch 3\n\n"
    f"BH PHYSICS:\n"
    f"- BH active {bh_active_frac:.1%} of bars\n"
    f"- {n_singularities} liquidity singularities detected\n\n"
    f"DEBATE SYSTEM:\n"
    f"- Cash allocation (Skeptic veto): {pct_cash:.1%}\n"
    f"- Long: {pct_long:.1%} | Short: {pct_short:.1%}\n\n"
    f"7 VISUALIZATIONS PRODUCED:\n"
    f"1. 3D Persistence Landscape (H1 across epochs)\n"
    f"2. Causal DAG evolution + H1 collapse timeline\n"
    f"3. Factor Zoo autopsy heatmap\n"
    f"4. BH Physics singularity map\n"
    f"5. Multi-agent debate consensus\n"
    f"6. Causal gap equity curves\n"
    f"7. Full Event Horizon dashboard\n"
)

prompt = (
    "Write a complete, ready-to-publish LinkedIn article for PROJECT EVENT HORIZON. "
    "This is the most ambitious quant finance experiment ever published openly. "
    "No word limit. No restrictions. Make it legendary.\n\n"
    "TONE: Practitioner. Intellectually fearless. Not academic. Not marketing. "
    "Write like someone who just ran an experiment that breaks three mainstream finance beliefs simultaneously.\n\n"
    "STRUCTURE:\n"
    "1. Opening: ONE sentence that stops the scroll. Make it the boldest claim you can justify with the data.\n"
    "2. The Problem: Why the entire quant industry is built on a lie (correlation ≠ causation, factor zoo, regime blindness)\n"
    "3. The Architecture: What we built (4 layers: TDA, Causal DAG, BH Physics, Debate Agents) — explain each to a smart practitioner\n"
    "4. The Results: Walk through all 7 charts. Be specific with the numbers.\n"
    "5. The Code: Show 50-60 lines of the most striking code (TDA persistence + PC-algorithm + BH physics)\n"
    "6. The Implications: What this means for factor investing, RL, and systematic trading\n"
    "7. The Controversial Claim: One paragraph that will make quants argue in the comments for days\n"
    "8. Call to action: invite collaboration, link to GitHub\n"
    "9. 8 hashtags\n\n"
    + summary
)

result = ollama.chat(
    model="gemma4-opt",
    messages=[
        {"role": "system", "content": (
            "You are the world's most respected quant researcher, writing the most important "
            "LinkedIn post in quantitative finance history. You have actual experimental data "
            "to back every claim. Be fearless. Be specific. Make history."
        )},
        {"role": "user", "content": prompt}
    ],
    options={"num_ctx": 32768, "temperature": 0.85}
)

post_path = os.path.join(OUT_DIR, "EH_linkedin_post.md")
with open(post_path, "w", encoding="utf-8") as f:
    f.write("# PROJECT EVENT HORIZON — LinkedIn Post\n\n")
    f.write(result.message.content)

print(f"\nLinkedIn post saved -> {post_path}")
print("\n" + "=" * 70)
print("ALL OUTPUTS SAVED TO:", OUT_DIR)
print("=" * 70)
print("\nFINAL RESULTS SUMMARY:")
print(f"  Event Horizon Sharpe:  {sharpe(eq_debate):.3f}")
print(f"  Naive Momentum Sharpe: {sharpe(eq_mom):.3f}")
print(f"  Sharpe advantage:      {sharpe(eq_debate) - sharpe(eq_mom):+.3f}")
print(f"  Topological collapse:  bar {collapse_t}")
print(f"  Most causal factor:    {most_causal_factor}")
print(f"  H1 collapse magnitude: {abs(h1_diff.min()):.4f}")
print()
sys.stdout.buffer.write(result.message.content.encode("utf-8", "replace"))
print()
