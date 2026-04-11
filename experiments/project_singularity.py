"""
PROJECT SINGULARITY — Phase II of Project Event Horizon
========================================================
Causal Erasure and the Relativistic Topology of Market Collapse.

Hypothesis: Market crashes are not driven by volatility — they are driven
by Causal Erasure. As liquidity density reaches a critical threshold (the
Singularity), the do(X) effect size of all market interventions approaches
zero. The market dies before the price moves.

New components vs Phase I:
  1. Student-T HMM with fat-tail emissions   (lib/math/hidden_markov.py)
  2. Bayesian credibility debate agents       (debate-system/agents/)
  3. Cross-asset contagion network (Wormhole) (networkx + BH physics)
  4. Causal do-calculus intervention sim      (scipy + structural eqns)
  5. Epistemic D3QN agent with uncertainty    (research/agent_training/)
  6. Ricci curvature of LOB manifold          (scipy Laplacian)
  7. Causal Erasure countdown metric          (novel)

Outputs: 7 new charts + full dashboard -> Desktop/srfm-experiments/
"""

import sys, os
sys.path.insert(0, r"C:\Users\Matthew\srfm-lab")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigvalsh
from scipy.stats import t as student_t, spearmanr
from scipy.optimize import minimize
import networkx as nx
from ripser import ripser
import warnings
warnings.filterwarnings("ignore")

from tools.regime_ml.hurst_monitor import hurst_rs, _classify, HurstRegime

OUT_DIR = r"C:\Users\Matthew\Desktop\srfm-experiments"
os.makedirs(OUT_DIR, exist_ok=True)

DARK_BG = "#0d1117"; PANEL_BG = "#161b22"; BORDER = "#30363d"
TEXT = "#e6edf3";    MUTED = "#8b949e"
RED = "#e74c3c";     GREEN = "#2ecc71"; BLUE = "#3498db"
PURPLE = "#9b59b6";  ORANGE = "#f39c12"; CYAN = "#1abc9c"
PINK = "#e91e63";    GOLD = "#f1c40f"

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.grid(color="#21262d", lw=0.5, alpha=0.7)
    return ax

print("=" * 70)
print("PROJECT SINGULARITY  |  SRFM Lab  |  Phase II")
print("Causal Erasure and the Relativistic Topology of Market Collapse")
print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# 1. ENHANCED SYNTHETIC UNIVERSE  (30 assets, 5 layers including crypto/DeFi)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[1/8] Building 30-asset multi-layer universe (TradFi + Crypto + DeFi)...")

N_TRADFI  = 20   # equities / macro
N_CRYPTO  = 6    # BTC, ETH, SOL, BNB, AVAX, MATIC
N_DEFI    = 4    # Uniswap TVL proxy, Aave borrow rate, COMP, MKR
N_ASSETS  = N_TRADFI + N_CRYPTO + N_DEFI
T_TOTAL   = 2500
CRISIS_START = 800
CRISIS_END   = 1100
rng = np.random.default_rng(7)

ASSET_NAMES = (
    [f"EQ{i+1}" for i in range(10)]
    + [f"FI{i+1}" for i in range(5)]
    + [f"CM{i+1}" for i in range(5)]
    + ["BTC", "ETH", "SOL", "BNB", "AVAX", "MATIC"]
    + ["UNI_TVL", "AAVE_RATE", "COMP", "MKR"]
)

def make_universe(T, crisis_start, crisis_end, rng):
    returns = np.zeros((N_ASSETS, T))

    # Phase 1: Low-vol regime (0 → crisis_start)
    for i in range(N_TRADFI):
        base = rng.normal(0.0003, 0.007, T)
        corr_shock = rng.normal(0, 0.004, T)
        returns[i] = base + corr_shock * (0.3 if i < 10 else 0.15)

    # Crypto: higher vol, BTC-driven correlation
    btc_factor = rng.normal(0.0005, 0.020, T)
    for j, i in enumerate(range(N_TRADFI, N_TRADFI + N_CRYPTO)):
        idio = rng.normal(0, 0.015 + j * 0.005, T)
        returns[i] = btc_factor * (0.7 - j * 0.05) + idio

    # DeFi: correlated with crypto but with fat-tail spikes
    for j, i in enumerate(range(N_TRADFI + N_CRYPTO, N_ASSETS)):
        idio = rng.standard_t(df=3, size=T) * 0.012
        returns[i] = btc_factor * 0.5 + idio

    # Phase 2: CRISIS (crisis_start → crisis_end)
    # Correlations spike → contagion → causal erasure
    crisis_len = crisis_end - crisis_start
    crisis_factor = rng.normal(-0.002, 0.025, crisis_len)  # strong down factor
    for i in range(N_ASSETS):
        # All assets converge to same factor (correlation = 1 → causal erasure)
        corr_weight = min(0.95, 0.5 + i * 0.02)   # stable coins last to fall
        idio = rng.normal(0, 0.005, crisis_len)
        returns[i, crisis_start:crisis_end] = (
            crisis_factor * corr_weight + idio
        )

    # Phase 3: New regime (decorrelated, causal rewiring)
    post_len = T - crisis_end
    for i in range(N_ASSETS):
        new_regime = rng.normal(0.0002, 0.010 + rng.uniform(0, 0.008), post_len)
        returns[i, crisis_end:] = new_regime

    return returns

returns = make_universe(T_TOTAL, CRISIS_START, CRISIS_END, rng)
prices  = np.exp(np.cumsum(returns, axis=1))
prices  = np.hstack([np.ones((N_ASSETS, 1)), prices])
market_ret = returns[:N_TRADFI].mean(axis=0)   # TradFi index
crypto_ret = returns[N_TRADFI:N_TRADFI+N_CRYPTO].mean(axis=0)
defi_ret   = returns[N_TRADFI+N_CRYPTO:].mean(axis=0)

print(f"  {N_ASSETS} assets | {T_TOTAL} bars | Crisis: [{CRISIS_START}, {CRISIS_END}]")
print(f"  TradFi:{N_TRADFI}  Crypto:{N_CRYPTO}  DeFi:{N_DEFI}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. STUDENT-T HMM (FAT-TAIL REGIME DETECTION)
#    Compares Gaussian vs Student-T emission log-likelihoods
# ═════════════════════════════════════════════════════════════════════════════
print("\n[2/8] Student-T HMM: fat-tail regime detection...")

def student_t_hmm(obs, n_states=3, n_iter=60, df_init=5.0):
    """
    HMM with Student-T emissions fit via EM.
    State params: (mu, scale, df) per state.
    """
    T = len(obs)
    obs = np.asarray(obs, dtype=float)

    # Initialise
    pi  = np.ones(n_states) / n_states
    A   = np.ones((n_states, n_states)) / n_states
    # Kmeans-like init
    pcts = np.percentile(obs, np.linspace(0, 100, n_states + 2)[1:-1])
    mu    = pcts.copy()
    scale = np.full(n_states, obs.std() * 0.5 + 1e-6)
    df    = np.full(n_states, df_init)

    log_liks = []

    def log_emission(t_val, mu_k, sc_k, df_k):
        """Log student-T pdf."""
        return student_t.logpdf(t_val, df=df_k, loc=mu_k, scale=sc_k)

    for it in range(n_iter):
        # E-step: forward-backward
        log_B = np.zeros((T, n_states))
        for k in range(n_states):
            log_B[:, k] = log_emission(obs, mu[k], scale[k], df[k])

        # Forward pass (log-space)
        log_alpha = np.full((T, n_states), -np.inf)
        log_alpha[0] = np.log(pi + 1e-300) + log_B[0]
        log_A = np.log(A + 1e-300)
        for t in range(1, T):
            for k in range(n_states):
                vals = log_alpha[t-1] + log_A[:, k]
                m = vals.max()
                log_alpha[t, k] = m + np.log(np.sum(np.exp(vals - m)) + 1e-300) + log_B[t, k]

        ll = np.max(log_alpha[-1]) + np.log(np.sum(np.exp(log_alpha[-1] - log_alpha[-1].max())))
        log_liks.append(float(ll))

        # Backward pass
        log_beta = np.zeros((T, n_states))
        for t in range(T-2, -1, -1):
            for k in range(n_states):
                vals = log_A[k, :] + log_B[t+1, :] + log_beta[t+1, :]
                m = vals.max()
                log_beta[t, k] = m + np.log(np.sum(np.exp(vals - m)) + 1e-300)

        # Posteriors
        log_gamma = log_alpha + log_beta
        log_gamma -= log_gamma.max(axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # M-step
        pi = gamma[0] + 1e-10
        pi /= pi.sum()

        for k in range(n_states):
            g_k = gamma[:, k]
            g_sum = g_k.sum() + 1e-10
            mu[k]    = (g_k * obs).sum() / g_sum
            # Scale via weighted MAD
            resid    = obs - mu[k]
            scale[k] = np.sqrt((g_k * resid**2).sum() / g_sum) + 1e-6
            # df: keep fixed (MLE for df is expensive) — use moment matching
            kurt = (g_k * resid**4).sum() / (g_sum * scale[k]**4 + 1e-10)
            df[k] = max(2.5, min(30.0, 6.0 / (kurt - 3.0 + 1e-10) + 4.0)) if kurt > 3 else df_init

        # Renormalise A
        A_new = np.zeros((n_states, n_states))
        for t in range(T-1):
            for k in range(n_states):
                for j in range(n_states):
                    A_new[k, j] += gamma[t, k] * np.exp(
                        log_A[k, j] + log_B[t+1, j] + log_beta[t+1, j]
                        - log_alpha[t, k] - log_B[t+1, j] + 1e-300
                    )
        A_new = np.clip(A_new, 1e-10, None)
        A = A_new / A_new.sum(axis=1, keepdims=True)

        if it > 5 and abs(log_liks[-1] - log_liks[-2]) < 1e-4:
            break

    # Viterbi decode
    vit = np.argmax(gamma, axis=1)

    return {"states": vit, "mu": mu, "scale": scale, "df": df,
            "log_liks": log_liks, "gamma": gamma}

# Fit Student-T HMM on market returns
print("  Fitting Student-T HMM...")
t_hmm = student_t_hmm(market_ret, n_states=3, n_iter=50)
t_states = t_hmm["states"]

# Compare to Gaussian vol-proxy
vol_roll = pd.Series(market_ret).rolling(30).std().bfill().values
gauss_states = np.digitize(vol_roll, np.percentile(vol_roll, [33, 66]))

# Tail thickness per state
print(f"  Student-T df per state: {[f'{d:.1f}' for d in t_hmm['df']]}")
print(f"  State distribution: {np.bincount(t_states)}")
# Find crisis state (highest scale)
crisis_state_idx = int(np.argmax(t_hmm["scale"]))
print(f"  Crisis state (max scale): state {crisis_state_idx} | df={t_hmm['df'][crisis_state_idx]:.1f} | scale={t_hmm['scale'][crisis_state_idx]:.5f}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. RICCI CURVATURE OF THE CORRELATION MANIFOLD
#    High Ricci curvature = market is "bending" toward a singularity
# ═════════════════════════════════════════════════════════════════════════════
print("\n[3/8] Computing Ricci curvature of correlation manifold...")

def ricci_curvature_approx(returns_window):
    """
    Approximate Ollivier-Ricci curvature for the asset graph.
    Uses the Laplacian eigenvalue spectrum as a proxy:
    kappa ≈ 1 - (lambda_1 / lambda_max)
    Higher kappa = more curved = more connected = approaching singularity.
    Also returns the spectral gap (lambda_2 - lambda_1).
    """
    corr = np.corrcoef(returns_window)
    np.fill_diagonal(corr, 1.0)
    # Adjacency: threshold at 0.3
    adj  = np.where(np.abs(corr) > 0.30, np.abs(corr), 0.0)
    np.fill_diagonal(adj, 0.0)
    deg  = adj.sum(axis=1)
    D    = np.diag(deg)
    L    = D - adj   # Laplacian
    try:
        eigs = eigvalsh(L)
        eigs = np.sort(np.abs(eigs))
        lam1 = eigs[1] if len(eigs) > 1 else 0.0   # Fiedler value
        lam_max = eigs[-1] if eigs[-1] > 0 else 1.0
        kappa = 1.0 - lam1 / lam_max
        spectral_gap = eigs[1] - eigs[0] if len(eigs) > 1 else 0.0
    except Exception:
        kappa, spectral_gap = 0.5, 0.0
    # Also compute avg correlation (crisis proxy)
    avg_corr = (np.abs(corr).sum() - N_ASSETS) / (N_ASSETS * (N_ASSETS - 1))
    return float(kappa), float(spectral_gap), float(avg_corr)

WINDOW = 150
STEP   = 25
ricci_ts  = []
sgap_ts   = []
acorr_ts  = []
h1_ts     = []
t_centers = []

for t in range(WINDOW, T_TOTAL, STEP):
    win = returns[:, t-WINDOW:t]
    kappa, sgap, acorr = ricci_curvature_approx(win)
    ricci_ts.append(kappa)
    sgap_ts.append(sgap)
    acorr_ts.append(acorr)
    # TDA H1
    corr = np.corrcoef(win)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(2 * (1 - np.clip(corr, -1, 1)))
    np.fill_diagonal(dist, 0.0)
    try:
        res = ripser(dist, metric="precomputed", maxdim=1)
        h1 = res["dgms"][1]
        h1_life = float(np.sum(h1[:, 1] - h1[:, 0])) if len(h1) > 0 else 0.0
    except Exception:
        h1_life = 0.0
    h1_ts.append(h1_life)
    t_centers.append(t)

ricci_ts = np.array(ricci_ts)
sgap_ts  = np.array(sgap_ts)
acorr_ts = np.array(acorr_ts)
h1_ts    = np.array(h1_ts)
t_centers = np.array(t_centers)

crisis_mask = (t_centers >= CRISIS_START) & (t_centers <= CRISIS_END)
print(f"  Ricci curvature: pre-crisis mean={ricci_ts[t_centers<CRISIS_START].mean():.4f} | crisis mean={ricci_ts[crisis_mask].mean():.4f}")
print(f"  H1 persistence: pre-crisis={h1_ts[t_centers<CRISIS_START].mean():.4f} | crisis={h1_ts[crisis_mask].mean():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. CROSS-ASSET CONTAGION WORMHOLE NETWORK
#    Build multi-layer graph: TradFi ↔ Crypto ↔ DeFi
#    Detect "wormholes" — edges that transmit collapse across layers
# ═════════════════════════════════════════════════════════════════════════════
print("\n[4/8] Building cross-asset contagion wormhole network...")

def build_contagion_network(returns_window, threshold=0.35):
    """
    Build a multi-layer contagion network.
    Intra-layer edges: strong correlation within asset class.
    Inter-layer edges (Wormholes): cross-class correlation > threshold.
    Returns: graph, wormhole edges, contagion centrality.
    """
    corr = np.corrcoef(returns_window)
    np.fill_diagonal(corr, 1.0)

    G = nx.Graph()
    layers = {"TradFi": list(range(N_TRADFI)),
              "Crypto": list(range(N_TRADFI, N_TRADFI + N_CRYPTO)),
              "DeFi":   list(range(N_TRADFI + N_CRYPTO, N_ASSETS))}

    for name, idx_list in layers.items():
        for i in idx_list:
            G.add_node(i, layer=name, name=ASSET_NAMES[i])

    wormhole_edges = []
    intra_edges = []
    for i in range(N_ASSETS):
        for j in range(i+1, N_ASSETS):
            c = abs(corr[i, j])
            if c > threshold:
                layer_i = G.nodes[i]["layer"]
                layer_j = G.nodes[j]["layer"]
                G.add_edge(i, j, weight=c, is_wormhole=(layer_i != layer_j))
                if layer_i != layer_j:
                    wormhole_edges.append((i, j, c))
                else:
                    intra_edges.append((i, j, c))

    # Contagion centrality: betweenness weighted by cross-layer edges
    try:
        centrality = nx.betweenness_centrality(G, weight="weight", normalized=True)
    except Exception:
        centrality = {i: 0.0 for i in range(N_ASSETS)}

    return G, wormhole_edges, intra_edges, centrality, corr

# Compute for 3 windows: pre-crisis, peak-crisis, post-crisis
windows = {
    "Pre-Crisis":  (max(0, CRISIS_START - WINDOW), CRISIS_START),
    "Peak Crisis": (CRISIS_START, CRISIS_END),
    "Post-Crisis": (CRISIS_END, min(T_TOTAL, CRISIS_END + WINDOW)),
}

network_stats = {}
for label, (t0, t1) in windows.items():
    win = returns[:, t0:t1]
    G, wh_edges, intra, centrality, corr_mat = build_contagion_network(win)
    top_wormhole = sorted(wh_edges, key=lambda x: x[2], reverse=True)[:3]
    network_stats[label] = {
        "G": G, "wormholes": wh_edges, "intra": intra,
        "centrality": centrality, "corr": corr_mat,
        "n_wormholes": len(wh_edges),
        "n_intra": len(intra),
        "top_wormhole": top_wormhole,
        "avg_wormhole_strength": np.mean([w for _, _, w in wh_edges]) if wh_edges else 0.0,
    }
    print(f"  {label}: {len(wh_edges)} wormholes | {len(intra)} intra-layer edges | avg wormhole strength={network_stats[label]['avg_wormhole_strength']:.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. CAUSAL DO-CALCULUS INTERVENTION SIMULATOR
#    Q: If we inject +2σ liquidity into crypto, does TradFi topology recover?
# ═════════════════════════════════════════════════════════════════════════════
print("\n[5/8] Causal intervention simulator (do-calculus)...")

def causal_intervention(returns_window, intervene_on, intervention_size, target_layer):
    """
    Simulate do(X = intervention_size) for assets in intervene_on.
    Measures causal effect on target_layer assets via structural equations.

    Uses linear SEM: Y = B*X + noise
    Intervention: set X[intervene_on] = intervention_size (ignores natural causes)
    Measures: change in target correlation structure (H1 persistence).
    """
    n_bars, n_assets = returns_window.T.shape

    # Estimate structural coefficients via regression (proxy for SEM)
    X_obs = returns_window.T.copy()   # (T, N)
    intervene_on = list(intervene_on)
    target_layer = list(target_layer)
    B = np.zeros((n_assets, n_assets))
    for j in range(n_assets):
        if j not in intervene_on:
            X_causes = X_obs[:, intervene_on]
            if X_causes.shape[1] > 0:
                try:
                    coef, *_ = np.linalg.lstsq(X_causes, X_obs[:, j], rcond=None)
                    B[j, list(intervene_on)] = coef
                except Exception:
                    pass

    # Observational world
    corr_obs = np.corrcoef(returns_window)
    np.fill_diagonal(corr_obs, 1.0)
    dist_obs = np.sqrt(2 * (1 - np.clip(corr_obs, -1, 1)))
    np.fill_diagonal(dist_obs, 0.0)
    try:
        h1_obs = float(np.sum(np.diff(ripser(dist_obs, metric="precomputed", maxdim=1)["dgms"][1], axis=1)))
    except Exception:
        h1_obs = 0.0

    # Interventional world: do(X = intervention_size)
    X_int = X_obs.copy()
    X_int[:, list(intervene_on)] = intervention_size

    # Propagate intervention through SEM
    X_post = X_int.copy()
    for j in target_layer:
        X_post[:, j] = X_int[:, j] + B[j] @ (X_int - X_obs).T.mean(axis=1)

    corr_int = np.corrcoef(X_post.T)
    np.fill_diagonal(corr_int, 1.0)
    dist_int = np.sqrt(2 * (1 - np.clip(corr_int, -1, 1)))
    np.fill_diagonal(dist_int, 0.0)
    try:
        h1_int = float(np.sum(np.diff(ripser(dist_int, metric="precomputed", maxdim=1)["dgms"][1], axis=1)))
    except Exception:
        h1_int = 0.0

    return h1_obs, h1_int, h1_int - h1_obs

# Run interventions at different crisis severities
intervene_crypto = set(range(N_TRADFI, N_TRADFI + N_CRYPTO))
target_tradfi    = list(range(N_TRADFI))
intervention_sizes = [-3.0, -1.0, 0.0, 1.0, 2.0, 3.0]  # in sigma units
crisis_window = returns[:, CRISIS_START:CRISIS_END]

do_results = []
print("  Running do(X) interventions on Crypto → measuring TradFi H1 response...")
for size in intervention_sizes:
    h1_obs, h1_int, delta = causal_intervention(
        crisis_window, intervene_crypto, size * crisis_window[list(intervene_crypto)].std(), target_tradfi)
    do_results.append({"size": size, "h1_obs": h1_obs, "h1_int": h1_int, "delta": delta})
    print(f"    do(Crypto={size:+.0f}σ): H1 obs={h1_obs:.4f} → int={h1_int:.4f} | Δ={delta:+.4f}")

# The Causal Erasure check: does intervention SIZE matter less during peak crisis?
# If delta ≈ 0 regardless of size → causal erasure confirmed
delta_range = max(r["delta"] for r in do_results) - min(r["delta"] for r in do_results)
causal_erasure_confirmed = delta_range < 0.05
print(f"\n  Intervention effect range: {delta_range:.4f}")
print(f"  Causal Erasure confirmed: {causal_erasure_confirmed} (delta range < 0.05 means do(X) has no effect)")


# ═════════════════════════════════════════════════════════════════════════════
# 6. BAYESIAN CREDIBILITY DEBATE AGENTS
#    Three agents with track-record-weighted credibility votes
# ═════════════════════════════════════════════════════════════════════════════
print("\n[6/8] Bayesian credibility debate agents...")

class BayesianAgent:
    """
    Debate agent with Bayesian Beta-distribution credibility tracking.
    Prior: Beta(alpha=5, beta=5) → credibility = 0.5
    Update: +alpha on correct call, +beta on wrong call.
    """
    def __init__(self, name, specialization):
        self.name = name
        self.spec = specialization
        self.alpha = 5.0
        self.beta  = 5.0

    @property
    def credibility(self):
        return self.alpha / (self.alpha + self.beta)

    def update_credibility(self, was_correct: bool):
        if was_correct:
            self.alpha += 1.0
        else:
            self.beta  += 1.0

    def vote(self, features: dict) -> tuple:
        raise NotImplementedError


class QuantResearcher(BayesianAgent):
    """Votes based on statistical signal strength and IC."""
    def __init__(self):
        super().__init__("QuantResearcher", "Statistical validation")

    def vote(self, f):
        score = 0.5
        score += 0.2 * np.clip(f.get("ic", 0) / 0.3, -1, 1)
        score += 0.15 * (1 if f.get("t_stat", 0) > 2.0 else -0.1)
        score -= 0.2 * f.get("h1_collapse", 0)
        return ("FOR" if score > 0.55 else "AGAINST" if score < 0.45 else "ABSTAIN",
                np.clip(abs(score - 0.5) * 2, 0, 1))


class RegimeSpecialist(BayesianAgent):
    """Votes based on regime stability and Hurst."""
    def __init__(self):
        super().__init__("RegimeSpecialist", "Regime-conditional performance")

    def vote(self, f):
        score = 0.5
        h = f.get("hurst", 0.5)
        if h > 0.6:   score += 0.25
        elif h < 0.4: score -= 0.25
        if f.get("regime_change", False): score -= 0.30
        score -= 0.3 * f.get("ricci_curvature", 0)
        return ("FOR" if score > 0.55 else "AGAINST" if score < 0.45 else "ABSTAIN",
                np.clip(abs(score - 0.5) * 2, 0, 1))


class RiskManager(BayesianAgent):
    """Hard veto: blocks trades when tail risk / causal erasure is extreme."""
    def __init__(self):
        super().__init__("RiskManager", "Tail risk + causal erasure")

    def vote(self, f):
        score = 0.5
        df_val = f.get("t_df", 10.0)
        if df_val < 4.0:  score -= 0.35   # fat tails → dangerous
        if f.get("causal_erasure", False): score -= 0.50
        score += 0.15 * f.get("bh_active", 0)
        n_worm = f.get("n_wormholes", 0)
        if n_worm > 10: score -= 0.20
        return ("AGAINST" if score < 0.40 else "FOR" if score > 0.60 else "ABSTAIN",
                np.clip(abs(score - 0.5) * 2, 0, 1))


agents = [QuantResearcher(), RegimeSpecialist(), RiskManager()]

# Run debate at each timestep
h_series = np.full(T_TOTAL, 0.5)
for t in range(200, T_TOTAL, 20):
    seg = prices[0, max(0, t-200):t]
    r   = np.diff(np.log(seg + 1e-10))
    try:
        h = hurst_rs(r)
        if np.isfinite(h): h_series[t] = h
    except: pass
# forward fill
last = 0.5
for i in range(len(h_series)):
    if h_series[i] != 0.5 or i == 0:
        last = h_series[i]
    else:
        h_series[i] = last

# Interpolate ricci / H1 to full timeline
from scipy.interpolate import interp1d
interp_ricci = interp1d(t_centers, ricci_ts, fill_value="extrapolate", kind="linear")
interp_h1    = interp1d(t_centers, h1_ts,    fill_value="extrapolate", kind="linear")

debate_actions  = np.zeros(T_TOTAL)
credibility_log = {a.name: [] for a in agents}
agreement_log   = []
veto_log        = []

ic_series = pd.Series(market_ret).rolling(30).apply(
    lambda x: float(spearmanr(x, np.arange(len(x)))[0]) if len(x) > 5 else 0.0
).fillna(0).values

for t in range(200, T_TOTAL):
    rc     = float(interp_ricci(t))
    h1_val = float(interp_h1(t))
    h_val  = float(h_series[t])
    t_df   = float(t_hmm["df"][t_states[t]])
    ic_now = float(ic_series[t])
    wh_now = network_stats["Peak Crisis"]["n_wormholes"] if CRISIS_START <= t <= CRISIS_END else network_stats["Pre-Crisis"]["n_wormholes"]

    # Detect causal erasure: high ricci + low H1 + low df
    erasure = (rc > 0.75) and (h1_val < h1_ts.mean() * 0.5) and (t_df < 5.0)

    features = {
        "ic":             ic_now,
        "t_stat":         abs(ic_now) * np.sqrt(30),
        "h1_collapse":    1.0 if h1_val < h1_ts.mean() * 0.5 else 0.0,
        "hurst":          h_val,
        "regime_change":  abs(h_val - 0.5) > 0.15,
        "ricci_curvature": rc,
        "t_df":           t_df,
        "causal_erasure": erasure,
        "bh_active":      1.0 if h_val > 0.6 else 0.0,
        "n_wormholes":    wh_now,
    }

    votes = []
    weights = []
    for agent in agents:
        vote_dir, conf = agent.vote(features)
        credibility_log[agent.name].append(agent.credibility)
        val = 1 if vote_dir == "FOR" else -1 if vote_dir == "AGAINST" else 0
        weight = agent.credibility * conf
        votes.append(val * weight)
        weights.append(weight)

    total_weight = sum(weights) + 1e-10
    consensus    = sum(votes) / total_weight

    # Track record update (next bar's sign as ground truth)
    if t < T_TOTAL - 1:
        actual_sign = np.sign(market_ret[t+1])
        for agent, v in zip(agents, votes):
            agent.update_credibility(np.sign(v) == actual_sign)

    veto = erasure or (features["t_df"] < 3.5)
    veto_log.append(veto)

    if veto:
        debate_actions[t] = 0
    elif consensus > 0.15:
        debate_actions[t] = 1
    elif consensus < -0.15:
        debate_actions[t] = -1
    else:
        debate_actions[t] = 0

    agreement_log.append(consensus)

for a in agents:
    print(f"  {a.name}: final credibility = {a.credibility:.3f}")

pct_veto = np.mean(veto_log)
pct_long = (debate_actions[200:] > 0).mean()
pct_short = (debate_actions[200:] < 0).mean()
print(f"  Causal Erasure vetoes: {pct_veto:.1%} | Long: {pct_long:.1%} | Short: {pct_short:.1%}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. EPISTEMIC D3QN AGENT WITH UNCERTAINTY DECOMPOSITION
#    Uses ensemble of Q-networks to separate aleatoric vs epistemic variance
# ═════════════════════════════════════════════════════════════════════════════
print("\n[7/8] Training Epistemic Ensemble D3QN agent...")

N_ENSEMBLE = 5
N_FEATURES = 8

def get_features(t, returns, market_ret, h_series, ricci, h1):
    """8-feature state for RL agent."""
    mom   = market_ret[max(0,t-10):t].sum() if t >= 10 else 0.0
    vol   = market_ret[max(0,t-20):t].std() if t >= 20 else 0.01
    h     = float(h_series[t])
    rc    = float(ricci)
    h1v   = float(h1)
    cr_ret = returns[N_TRADFI:N_TRADFI+N_CRYPTO, t].mean() if t > 0 else 0.0
    df_ret = returns[N_TRADFI+N_CRYPTO:, t].mean() if t > 0 else 0.0
    crisis = 1.0 if CRISIS_START <= t <= CRISIS_END else 0.0
    return np.array([
        np.clip(mom, -0.1, 0.1),
        np.clip(vol, 0, 0.1),
        np.clip(h - 0.5, -0.5, 0.5),
        np.clip(rc, 0, 1),
        np.clip(h1v / (h1_ts.max() + 1e-6), 0, 1),
        np.clip(cr_ret, -0.1, 0.1),
        np.clip(df_ret, -0.1, 0.1),
        crisis,
    ], dtype=np.float32)

class EnsembleQNet:
    """Ensemble of N linear Q-networks for epistemic uncertainty estimation."""
    def __init__(self, n_features, n_actions=3, n_ensemble=5, lr=0.01):
        self.n = n_ensemble
        self.n_actions = n_actions
        # Each member: W1(n_f→32) + W2(32→n_a)
        self.W1 = [rng.normal(0, 0.1, (32, n_features)) for _ in range(n_ensemble)]
        self.W2 = [rng.normal(0, 0.1, (n_actions, 32))  for _ in range(n_ensemble)]
        self.b1 = [np.zeros(32) for _ in range(n_ensemble)]
        self.b2 = [np.zeros(n_actions) for _ in range(n_ensemble)]
        self.lr = lr

    def forward(self, x, member):
        h = np.tanh(self.W1[member] @ x + self.b1[member])
        return self.W2[member] @ h + self.b2[member]

    def predict_all(self, x):
        qs = np.array([self.forward(x, m) for m in range(self.n)])
        mean_q = qs.mean(axis=0)
        # Aleatoric: irreducible noise in environment
        # Epistemic: disagreement between ensemble members
        epistemic_var = qs.var(axis=0)
        aleatoric_var = np.abs(qs - mean_q).mean(axis=0)
        return mean_q, epistemic_var, aleatoric_var

    def update(self, x, action, target, member):
        """SGD update for one member."""
        h = np.tanh(self.W1[member] @ x + self.b1[member])
        q = self.W2[member] @ h + self.b2[member]
        err = q[action] - target
        dW2 = np.zeros_like(self.W2[member])
        dW2[action] = err * h
        dh  = err * self.W2[member][action]
        dh_act = dh * (1 - h**2)
        dW1 = np.outer(dh_act, x)
        self.W2[member] -= self.lr * dW2
        self.W1[member] -= self.lr * dW1

ensemble = EnsembleQNet(N_FEATURES, n_actions=3, n_ensemble=N_ENSEMBLE)

# Train: 3 passes over timeline
GAMMA = 0.95; EPSILON = 0.20
for epoch in range(3):
    EPSILON = max(0.05, EPSILON - 0.05)
    for t in range(50, T_TOTAL - 1):
        rc    = float(interp_ricci(t))
        h1v   = float(interp_h1(t))
        feat  = get_features(t, returns, market_ret, h_series, rc, h1v)
        q_m, ep_var, al_var = ensemble.predict_all(feat)
        mean_ep = ep_var.mean()

        # High epistemic uncertainty → cash (action 2)
        if mean_ep > 0.01 or (rng.random() < EPSILON):
            action = rng.integers(0, 3)
        else:
            action = int(np.argmax(q_m))

        # Reward
        actual_ret = market_ret[t+1]
        if action == 0:   reward = actual_ret - 0.0001
        elif action == 1: reward = -actual_ret - 0.0001
        else:             reward = -0.00005  # cash

        feat_next = get_features(t+1, returns, market_ret, h_series,
                                  float(interp_ricci(min(t+1, T_TOTAL-1))),
                                  float(interp_h1(min(t+1, T_TOTAL-1))))
        q_next, _, _ = ensemble.predict_all(feat_next)
        target = reward + GAMMA * q_next.max()

        member = rng.integers(0, N_ENSEMBLE)
        ensemble.update(feat, action, target, member)

# Evaluate: collect epistemic uncertainty + actions
ep_uncertainty = []
al_uncertainty = []
ens_actions    = []
for t in range(50, T_TOTAL):
    rc  = float(interp_ricci(t))
    h1v = float(interp_h1(t))
    feat = get_features(t, returns, market_ret, h_series, rc, h1v)
    q_m, ep_var, al_var = ensemble.predict_all(feat)
    ep_uncertainty.append(ep_var.mean())
    al_uncertainty.append(al_var.mean())

    if ep_var.mean() > 0.008:
        ens_actions.append(2)   # cash — epistemic uncertainty too high
    else:
        ens_actions.append(int(np.argmax(q_m)))

ep_uncertainty = np.array(ep_uncertainty)
al_uncertainty = np.array(al_uncertainty)

# Check: epistemic uncertainty spikes at regime shifts?
ep_crisis  = ep_uncertainty[max(0,CRISIS_START-50):CRISIS_END-50].mean()
ep_stable  = ep_uncertainty[:CRISIS_START-50].mean()
print(f"  Epistemic uncertainty: stable={ep_stable:.5f} | crisis={ep_crisis:.5f} | ratio={ep_crisis/(ep_stable+1e-10):.1f}x")

# Portfolio simulation
equity_ens = [1.0]; equity_deb = [1.0]; equity_bh  = [1.0]
TC = 0.001
prev_e = 2; prev_d = 0

for t in range(50, T_TOTAL - 1):
    ret = market_ret[t+1]
    # Ensemble agent
    ae = ens_actions[t - 50]
    if ae != prev_e: equity_ens[-1] *= (1 - TC)
    prev_e = ae
    if ae == 0:   equity_ens.append(equity_ens[-1] * (1 + ret))
    elif ae == 1: equity_ens.append(equity_ens[-1] * (1 - ret))
    else:         equity_ens.append(equity_ens[-1])

    # Debate agent
    ad = debate_actions[t]
    if ad != prev_d: equity_deb[-1] *= (1 - TC)
    prev_d = ad
    equity_deb.append(equity_deb[-1] * (1 + ad * ret))

    # BH baseline: long in trending market
    h_now = h_series[t]
    if h_now > 0.6:   equity_bh.append(equity_bh[-1] * (1 + ret * 0.8))
    else:             equity_bh.append(equity_bh[-1])

n_min = min(len(equity_ens), len(equity_deb), len(equity_bh))
eq_ens = np.array(equity_ens[:n_min])
eq_deb = np.array(equity_deb[:n_min])
eq_bh  = np.array(equity_bh[:n_min])

def sharpe(eq):
    r = np.diff(np.log(np.clip(eq, 1e-6, None)))
    return (r.mean() / (r.std() + 1e-10)) * np.sqrt(252)

print(f"\n  Ensemble D3QN Sharpe:     {sharpe(eq_ens):.3f}")
print(f"  Bayesian Debate Sharpe:   {sharpe(eq_deb):.3f}")
print(f"  BH Baseline Sharpe:       {sharpe(eq_bh):.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. SINGULARITY COUNTDOWN METRIC
#    Composite score: how many bars until causal erasure?
# ═════════════════════════════════════════════════════════════════════════════
print("\n[8/8] Computing Singularity Countdown metric...")

def singularity_score(ricci, h1, avg_corr, t_df, n_wormholes):
    """
    Composite 0-1 score: how close is the market to the singularity?
    1.0 = full causal erasure / singularity confirmed.
    """
    s  = 0.0
    s += 0.30 * np.clip(ricci, 0, 1)
    s += 0.25 * np.clip(1 - h1 / (h1_ts.max() + 1e-6), 0, 1)
    s += 0.20 * np.clip(avg_corr, 0, 1)
    s += 0.15 * np.clip(1 - (t_df - 2) / 18.0, 0, 1)  # low df = fat tails
    s += 0.10 * np.clip(n_wormholes / 30.0, 0, 1)
    return float(np.clip(s, 0, 1))

# Compute at each window
sing_scores = []
for i, t in enumerate(t_centers):
    wh_n = network_stats["Peak Crisis"]["n_wormholes"] if CRISIS_START <= t <= CRISIS_END \
           else network_stats["Pre-Crisis"]["n_wormholes"]
    t_df_now = float(t_hmm["df"][t_states[min(int(t), T_TOTAL-1)]])
    sc = singularity_score(ricci_ts[i], h1_ts[i], acorr_ts[i], t_df_now, wh_n)
    sing_scores.append(sc)

sing_scores = np.array(sing_scores)
# Find first bar where score > 0.75
alarm_bars = t_centers[sing_scores > 0.75]
first_alarm = int(alarm_bars[0]) if len(alarm_bars) > 0 else None
print(f"  Singularity alarm threshold (0.75) first crossed at bar: {first_alarm}")
print(f"  Crisis start was at bar: {CRISIS_START}")
if first_alarm and first_alarm < CRISIS_START:
    print(f"  EARLY WARNING: {CRISIS_START - first_alarm} bars BEFORE crisis onset")
elif first_alarm:
    print(f"  WARNING: {first_alarm - CRISIS_START} bars AFTER crisis onset")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Rendering 7 publication-quality visualizations...")
print("=" * 70)

smooth = lambda x, w=20: pd.Series(x).rolling(w, min_periods=1).mean().values


# ── CHART 1: Ricci Curvature + H1 Collapse + Singularity Score ───────────────
print("  Chart 1/7: Ricci curvature manifold...")
fig1, axes1 = plt.subplots(3, 1, figsize=(18, 12), facecolor=DARK_BG,
    gridspec_kw={"hspace": 0.42, "top": 0.91, "bottom": 0.07, "left": 0.07, "right": 0.96})
for ax in axes1: style_ax(ax)

cmap_ricci = LinearSegmentedColormap.from_list("ricci",
    [(0,"#0d2137"),(0.4,BLUE),(0.7,ORANGE),(1.0,RED)])

ax1a, ax1b, ax1c = axes1

ax1a.plot(t_centers, ricci_ts, color=ORANGE, lw=1.8, label="Ricci Curvature κ")
ax1a.fill_between(t_centers, 0, ricci_ts, alpha=0.2, color=ORANGE)
ax1a.axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color=RED, label="Crisis window")
ax1a.axhline(0.75, color=RED, lw=1.2, ls="--", alpha=0.8, label="Singularity threshold")
if first_alarm:
    ax1a.axvline(first_alarm, color=GOLD, lw=2.0, ls="-", label=f"Alarm bar {first_alarm}")
ax1a.set_title("Ricci Curvature κ — The Market Bends Before It Breaks",
               color=TEXT, fontsize=11, fontweight="bold")
ax1a.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
ax1a.set_ylabel("κ (curvature)", color=MUTED)

ax1b.plot(t_centers, h1_ts, color=PURPLE, lw=1.8, label="H₁ Persistence")
ax1b.fill_between(t_centers, 0, h1_ts, alpha=0.2, color=PURPLE)
ax1b.axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color=RED)
if first_alarm:
    ax1b.axvline(first_alarm, color=GOLD, lw=2.0, ls="-", alpha=0.9)
ax1b.set_title("H₁ Topological Persistence — Dimension Loss", color=TEXT, fontsize=11, fontweight="bold")
ax1b.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
ax1b.set_ylabel("H₁ lifetime", color=MUTED)

sc = ax1c.scatter(t_centers, sing_scores, c=sing_scores, cmap=cmap_ricci, s=30, zorder=3)
ax1c.plot(t_centers, sing_scores, color=MUTED, lw=0.8, alpha=0.5)
ax1c.fill_between(t_centers, 0, sing_scores, alpha=0.15, color=PINK)
ax1c.axhline(0.75, color=RED, lw=1.5, ls="--", label="SINGULARITY ALARM (0.75)")
ax1c.axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color=RED)
if first_alarm:
    ax1c.axvline(first_alarm, color=GOLD, lw=2.0, ls="-")
    ax1c.annotate(f"SINGULARITY ALARM\nbar {first_alarm}\n({CRISIS_START-first_alarm} bars early)",
                  xy=(first_alarm, 0.75), xytext=(first_alarm+80, 0.55),
                  color=GOLD, fontsize=9, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG, edgecolor=GOLD))
plt.colorbar(sc, ax=ax1c, label="Singularity Score").ax.tick_params(colors=MUTED, labelcolor=TEXT)
ax1c.set_title("Singularity Countdown Score (composite)", color=TEXT, fontsize=11, fontweight="bold")
ax1c.set_xlabel("Bar", color=MUTED); ax1c.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
ax1c.set_ylabel("Score 0-1", color=MUTED)

fig1.suptitle("CHART 1  |  Ricci Curvature + H₁ Collapse + Singularity Countdown",
              color=TEXT, fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "SG_01_ricci_singularity.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: SG_01_ricci_singularity.png")


# ── CHART 2: Wormhole Contagion Network ──────────────────────────────────────
print("  Chart 2/7: Wormhole contagion network...")
fig2, axes2 = plt.subplots(1, 3, figsize=(22, 8), facecolor=DARK_BG,
    gridspec_kw={"wspace": 0.15, "left": 0.03, "right": 0.97, "top": 0.88, "bottom": 0.05})

layer_colors = {"TradFi": BLUE, "Crypto": ORANGE, "DeFi": CYAN}
layer_pos_offset = {"TradFi": 0, "Crypto": 2.5, "DeFi": 5}

for ax, (label, stats) in zip(axes2, network_stats.items()):
    ax.set_facecolor(PANEL_BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

    G = stats["G"]
    # Layout: nodes arranged by layer
    pos = {}
    for layer_name, idx_list in [("TradFi", list(range(N_TRADFI))),
                                   ("Crypto", list(range(N_TRADFI, N_TRADFI+N_CRYPTO))),
                                   ("DeFi",   list(range(N_TRADFI+N_CRYPTO, N_ASSETS)))]:
        angle_step = 2 * np.pi / max(len(idx_list), 1)
        r = 1.0
        cx = layer_pos_offset[layer_name]
        for k, node in enumerate(idx_list):
            a = k * angle_step
            pos[node] = (cx + r * np.cos(a) * 0.8, r * np.sin(a))

    # Draw intra-layer edges
    for i, j, w in stats["intra"]:
        x1, y1 = pos[i]; x2, y2 = pos[j]
        c = layer_colors.get(G.nodes[i]["layer"], MUTED)
        ax.plot([x1,x2],[y1,y2], color=c, alpha=min(w*0.5,0.3), lw=0.8)

    # Draw wormholes
    for i, j, w in stats["wormholes"]:
        x1,y1 = pos[i]; x2,y2 = pos[j]
        ax.plot([x1,x2],[y1,y2], color=RED, alpha=min(w, 0.7), lw=1.5, zorder=3)

    # Draw nodes
    for node in G.nodes():
        layer = G.nodes[node]["layer"]
        c = layer_colors.get(layer, MUTED)
        cen = stats["centrality"].get(node, 0.0)
        ax.scatter(*pos[node], s=60 + cen*300, color=c, zorder=5, edgecolors=DARK_BG, linewidths=0.5)

    ax.set_title(f"{label}\n{stats['n_wormholes']} wormholes | {stats['n_intra']} intra edges",
                 color=TEXT, fontsize=10, fontweight="bold")
    # Legend
    for ln, lc in layer_colors.items():
        ax.scatter([], [], color=lc, s=40, label=ln)
    ax.plot([], [], color=RED, lw=2, label="Wormhole")
    ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=7, loc="lower right")

fig2.suptitle("CHART 2  |  Cross-Asset Contagion Wormholes — TradFi ↔ Crypto ↔ DeFi",
              color=TEXT, fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "SG_02_wormhole_network.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: SG_02_wormhole_network.png")


# ── CHART 3: Student-T vs Gaussian HMM + Tail Thickness ─────────────────────
print("  Chart 3/7: Student-T HMM fat-tail regimes...")
fig3, axes3 = plt.subplots(2, 2, figsize=(18, 10), facecolor=DARK_BG,
    gridspec_kw={"hspace": 0.42, "wspace": 0.30, "top": 0.91, "bottom": 0.08, "left": 0.07, "right": 0.96})
axes3 = axes3.flatten()
for ax in axes3: style_ax(ax)

state_colors_t = [GREEN, ORANGE, RED]
# Panel 1: Returns coloured by HMM state
for k in range(3):
    mask = t_states == k
    axes3[0].scatter(np.where(mask)[0], market_ret[mask], c=state_colors_t[k],
                     s=4, alpha=0.6, label=f"State {k} (df={t_hmm['df'][k]:.1f})")
axes3[0].axvspan(CRISIS_START, CRISIS_END, alpha=0.1, color=RED)
axes3[0].set_title("Market Returns Coloured by Student-T HMM State",
                   color=TEXT, fontsize=10, fontweight="bold")
axes3[0].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# Panel 2: df over time (fat-tail thickness)
df_series = np.array([t_hmm["df"][s] for s in t_states])
axes3[1].plot(range(T_TOTAL), df_series, color=PURPLE, lw=1.2, alpha=0.7)
axes3[1].fill_between(range(T_TOTAL), 0, df_series, alpha=0.15, color=PURPLE)
axes3[1].axhline(5.0, color=RED, lw=1.2, ls="--", label="df=5 (heavy tails)")
axes3[1].axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color=RED)
axes3[1].set_title("Student-T Degrees of Freedom Over Time\n(Low df = Fat Tails = Crash Risk)",
                   color=TEXT, fontsize=10, fontweight="bold")
axes3[1].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
axes3[1].set_ylabel("df", color=MUTED)

# Panel 3: Distribution comparison Gaussian vs T
x_range = np.linspace(-0.08, 0.08, 300)
crisis_state = crisis_state_idx
gauss_pdf = (1/(t_hmm["scale"][crisis_state]*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range - t_hmm["mu"][crisis_state])/t_hmm["scale"][crisis_state])**2)
t_pdf = student_t.pdf(x_range, df=t_hmm["df"][crisis_state], loc=t_hmm["mu"][crisis_state], scale=t_hmm["scale"][crisis_state])
axes3[2].plot(x_range, t_pdf, color=RED, lw=2.0, label=f"Student-T (df={t_hmm['df'][crisis_state]:.1f})")
axes3[2].plot(x_range, gauss_pdf, color=BLUE, lw=2.0, ls="--", label="Gaussian (df=∞)")
axes3[2].fill_between(x_range, 0, t_pdf, where=np.abs(x_range) > 0.04, alpha=0.3, color=RED, label="Fat tail region")
axes3[2].set_title("Crisis State: Gaussian vs Student-T Emission\n(Red = mass in tails Gaussian ignores)",
                   color=TEXT, fontsize=10, fontweight="bold")
axes3[2].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
axes3[2].set_xlabel("Return", color=MUTED)

# Panel 4: Actual return histogram vs both fits
crisis_rets = market_ret[CRISIS_START:CRISIS_END]
axes3[3].hist(crisis_rets, bins=40, color=MUTED, alpha=0.5, density=True, label="Actual crisis returns")
x_h = np.linspace(crisis_rets.min(), crisis_rets.max(), 200)
mu_c, sc_c, df_c = t_hmm["mu"][crisis_state], t_hmm["scale"][crisis_state], t_hmm["df"][crisis_state]
axes3[3].plot(x_h, student_t.pdf(x_h, df=df_c, loc=mu_c, scale=sc_c), color=RED, lw=2.0, label=f"Student-T fit (df={df_c:.1f})")
axes3[3].plot(x_h, (1/(sc_c*np.sqrt(2*np.pi)))*np.exp(-0.5*((x_h-mu_c)/sc_c)**2), color=BLUE, lw=2.0, ls="--", label="Gaussian fit")
axes3[3].set_title("Crisis Returns: Actual vs Model Fit\n(Gaussian misses extreme tails)",
                   color=TEXT, fontsize=10, fontweight="bold")
axes3[3].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
axes3[3].set_xlabel("Return", color=MUTED)

fig3.suptitle("CHART 3  |  Student-T HMM — The Fat Tails Gaussian Models Ignore",
              color=TEXT, fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "SG_03_student_t_hmm.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: SG_03_student_t_hmm.png")


# ── CHART 4: Causal do-Calculus Intervention Response ────────────────────────
print("  Chart 4/7: Causal do-calculus intervention...")
fig4, axes4 = plt.subplots(1, 2, figsize=(18, 7), facecolor=DARK_BG,
    gridspec_kw={"wspace": 0.30, "left": 0.07, "right": 0.96, "top": 0.88, "bottom": 0.12})
for ax in axes4: style_ax(ax)

sizes = [r["size"] for r in do_results]
deltas = [r["delta"] for r in do_results]
h1_obs_vals = [r["h1_obs"] for r in do_results]
h1_int_vals = [r["h1_int"] for r in do_results]

bar_colors = [GREEN if d > 0.01 else RED if d < -0.01 else MUTED for d in deltas]
axes4[0].bar(sizes, deltas, color=bar_colors, alpha=0.85, edgecolor=DARK_BG, width=0.5)
axes4[0].axhline(0, color=MUTED, lw=0.8)
axes4[0].axhline(0.05, color=GREEN, lw=1.0, ls=":", alpha=0.7)
axes4[0].axhline(-0.05, color=RED, lw=1.0, ls=":", alpha=0.7)
axes4[0].fill_between([-3.5, 3.5], -0.05, 0.05, alpha=0.08, color=MUTED, label="Causal Erasure zone (Δ≈0)")
axes4[0].set_xlabel("Intervention Size (σ)", color=MUTED, fontsize=10)
axes4[0].set_ylabel("ΔH₁ (topology change)", color=MUTED, fontsize=10)
axes4[0].set_title(f"do(Crypto=Xσ) → ΔH₁ in TradFi\nΔ range={delta_range:.4f} | Erasure={'YES' if causal_erasure_confirmed else 'NO'}",
                   color=TEXT, fontsize=11, fontweight="bold")
axes4[0].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
if causal_erasure_confirmed:
    axes4[0].text(0, max(deltas)*0.7,
                  "CAUSAL ERASURE CONFIRMED\ndo(X) has no effect on topology",
                  ha="center", color=RED, fontsize=10, fontweight="bold",
                  bbox=dict(boxstyle="round", facecolor=PANEL_BG, edgecolor=RED))

axes4[1].plot(sizes, h1_obs_vals, color=BLUE, lw=2.0, marker="o", ms=8, label="H₁ Observational")
axes4[1].plot(sizes, h1_int_vals, color=ORANGE, lw=2.0, marker="s", ms=8, ls="--", label="H₁ Interventional do(X)")
axes4[1].fill_between(sizes, h1_obs_vals, h1_int_vals, alpha=0.2, color=PURPLE, label="Causal Gap")
axes4[1].set_xlabel("Intervention Size (σ)", color=MUTED, fontsize=10)
axes4[1].set_ylabel("H₁ Persistence", color=MUTED, fontsize=10)
axes4[1].set_title("Observational vs Interventional Topology\nFlat lines = interventions are meaningless",
                   color=TEXT, fontsize=11, fontweight="bold")
axes4[1].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)

fig4.suptitle("CHART 4  |  Causal Erasure — do(X) Becomes Powerless at the Singularity",
              color=TEXT, fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "SG_04_causal_erasure.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: SG_04_causal_erasure.png")


# ── CHART 5: Epistemic Uncertainty Clouds ────────────────────────────────────
print("  Chart 5/7: Epistemic uncertainty decomposition...")
fig5, axes5 = plt.subplots(3, 1, figsize=(18, 12), facecolor=DARK_BG,
    gridspec_kw={"hspace": 0.42, "top": 0.91, "bottom": 0.07, "left": 0.07, "right": 0.96})
for ax in axes5: style_ax(ax)

t_ep = np.arange(50, T_TOTAL)

axes5[0].fill_between(t_ep, smooth(ep_uncertainty, 15), alpha=0.7, color=RED, label="Epistemic (model ignorance)")
axes5[0].fill_between(t_ep, smooth(al_uncertainty, 15), alpha=0.5, color=BLUE, label="Aleatoric (market noise)")
axes5[0].axvspan(CRISIS_START, CRISIS_END, alpha=0.1, color=RED)
if first_alarm: axes5[0].axvline(first_alarm, color=GOLD, lw=2.0, ls="-", label=f"Singularity alarm")
axes5[0].set_title("Epistemic vs Aleatoric Uncertainty — Agent Knows It Doesn't Know",
                   color=TEXT, fontsize=11, fontweight="bold")
axes5[0].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
axes5[0].set_ylabel("Variance", color=MUTED)

# Uncertainty ratio (epistemic / total)
total_unc = ep_uncertainty + al_uncertainty + 1e-10
ep_ratio  = ep_uncertainty / total_unc
axes5[1].plot(t_ep, smooth(ep_ratio, 20), color=PURPLE, lw=1.8, label="Epistemic fraction")
axes5[1].fill_between(t_ep, 0, smooth(ep_ratio, 20), alpha=0.2, color=PURPLE)
axes5[1].axhline(0.5, color=ORANGE, lw=1.0, ls="--", alpha=0.7, label="50% epistemic threshold")
axes5[1].axvspan(CRISIS_START, CRISIS_END, alpha=0.1, color=RED)
if first_alarm: axes5[1].axvline(first_alarm, color=GOLD, lw=2.0, ls="-", alpha=0.9)
axes5[1].set_title("Epistemic Fraction — When the Agent is More Confused Than the Market Is Noisy",
                   color=TEXT, fontsize=11, fontweight="bold")
axes5[1].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# Actions heatmap
actions_arr = np.array(ens_actions)
axes5[2].fill_between(t_ep, 0, [1 if a==0 else 0 for a in actions_arr], color=GREEN, alpha=0.7, label="LONG")
axes5[2].fill_between(t_ep, 0, [-1 if a==1 else 0 for a in actions_arr], color=RED, alpha=0.7, label="SHORT")
ep_cash = smooth([1 if a==2 else 0 for a in actions_arr], 20)
axes5[2].fill_between(t_ep, 0, ep_cash, color=MUTED, alpha=0.5, label="CASH (epistemic veto)")
axes5[2].axvspan(CRISIS_START, CRISIS_END, alpha=0.1, color=RED)
cash_pct = (np.array(ens_actions) == 2).mean()
long_pct = (np.array(ens_actions) == 0).mean()
short_pct= (np.array(ens_actions) == 1).mean()
axes5[2].set_title(f"Ensemble Agent Actions | Cash: {cash_pct:.1%}  Long: {long_pct:.1%}  Short: {short_pct:.1%}",
                   color=TEXT, fontsize=11, fontweight="bold")
axes5[2].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
axes5[2].set_xlabel("Bar", color=MUTED)

fig5.suptitle("CHART 5  |  Epistemic Uncertainty — The Agent Knows When It Doesn't Know",
              color=TEXT, fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "SG_05_epistemic_uncertainty.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: SG_05_epistemic_uncertainty.png")


# ── CHART 6: Bayesian Debate Credibility Evolution ───────────────────────────
print("  Chart 6/7: Bayesian credibility evolution...")
fig6, axes6 = plt.subplots(2, 2, figsize=(18, 10), facecolor=DARK_BG,
    gridspec_kw={"hspace": 0.42, "wspace": 0.30, "top": 0.91, "bottom": 0.08, "left": 0.07, "right": 0.96})
axes6 = axes6.flatten()
for ax in axes6: style_ax(ax)

agent_colors = [CYAN, ORANGE, RED]
agent_cred_t = range(200, T_TOTAL)
for i, (agent, col) in enumerate(zip(agents, agent_colors)):
    creds = credibility_log[agent.name]
    axes6[0].plot(agent_cred_t[:len(creds)], creds, color=col, lw=1.8, label=f"{agent.name} ({agent.credibility:.3f})")
axes6[0].axhline(0.5, color=MUTED, lw=0.8, ls="--")
axes6[0].axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color=RED)
axes6[0].set_title("Agent Credibility (Bayesian Beta track record)",
                   color=TEXT, fontsize=10, fontweight="bold")
axes6[0].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
axes6[0].set_ylabel("Credibility", color=MUTED)

# Consensus over time
consensus_arr = np.array(agreement_log)
axes6[1].plot(agent_cred_t[:len(consensus_arr)], smooth(consensus_arr, 20), color=GREEN, lw=1.8)
axes6[1].fill_between(agent_cred_t[:len(consensus_arr)], 0, smooth(consensus_arr, 20),
                      where=smooth(consensus_arr, 20) > 0, color=GREEN, alpha=0.3)
axes6[1].fill_between(agent_cred_t[:len(consensus_arr)], 0, smooth(consensus_arr, 20),
                      where=smooth(consensus_arr, 20) < 0, color=RED, alpha=0.3)
axes6[1].axhline(0, color=MUTED, lw=0.8)
axes6[1].axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color=RED)
axes6[1].set_title("Weighted Debate Consensus (credibility × confidence)",
                   color=TEXT, fontsize=10, fontweight="bold")
axes6[1].set_ylabel("Consensus score", color=MUTED)

# Veto rate over time
veto_arr   = np.array(veto_log, dtype=float)
veto_roll  = pd.Series(veto_arr).rolling(50, min_periods=1).mean().values
axes6[2].plot(agent_cred_t[:len(veto_roll)], veto_roll, color=PURPLE, lw=1.8, label="Causal Erasure veto rate")
axes6[2].fill_between(agent_cred_t[:len(veto_roll)], 0, veto_roll, alpha=0.2, color=PURPLE)
axes6[2].axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color=RED)
axes6[2].set_title("Causal Erasure Veto Rate (50-bar rolling)",
                   color=TEXT, fontsize=10, fontweight="bold")
axes6[2].set_ylabel("Veto rate", color=MUTED)
axes6[2].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

# Final credibility bar
names = [a.name for a in agents]
creds_final = [a.credibility for a in agents]
bars = axes6[3].bar(names, creds_final, color=agent_colors, alpha=0.85, edgecolor=DARK_BG)
axes6[3].axhline(0.5, color=MUTED, lw=1.0, ls="--")
axes6[3].set_ylim(0, 1)
axes6[3].set_title("Final Agent Credibility Scores\n(Earned via Bayesian track record)",
                   color=TEXT, fontsize=10, fontweight="bold")
for bar, cred in zip(bars, creds_final):
    axes6[3].text(bar.get_x() + bar.get_width()/2, cred + 0.02, f"{cred:.3f}",
                  ha="center", color=TEXT, fontsize=10, fontweight="bold")
axes6[3].tick_params(colors=TEXT, labelsize=9)

fig6.suptitle("CHART 6  |  Bayesian Credibility Debate — Agents Earn Their Authority",
              color=TEXT, fontsize=13, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "SG_06_bayesian_debate.png"),
            dpi=160, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: SG_06_bayesian_debate.png")


# ── CHART 7: Full Project Singularity Dashboard ───────────────────────────────
print("  Chart 7/7: Full Singularity dashboard...")
fig7 = plt.figure(figsize=(24, 18), facecolor=DARK_BG)
gs7  = gridspec.GridSpec(4, 4, figure=fig7, hspace=0.52, wspace=0.38,
                          top=0.93, bottom=0.05, left=0.06, right=0.97)

panels = {
    "price":   fig7.add_subplot(gs7[0, 0:2]),
    "ricci":   fig7.add_subplot(gs7[0, 2]),
    "sing":    fig7.add_subplot(gs7[0, 3]),
    "worm":    fig7.add_subplot(gs7[1, 0]),
    "tdf":     fig7.add_subplot(gs7[1, 1]),
    "causal":  fig7.add_subplot(gs7[1, 2]),
    "ep":      fig7.add_subplot(gs7[1, 3]),
    "eq":      fig7.add_subplot(gs7[2, 0:2]),
    "cred":    fig7.add_subplot(gs7[2, 2]),
    "veto":    fig7.add_subplot(gs7[2, 3]),
    "ep_rat":  fig7.add_subplot(gs7[3, 0:2]),
    "hmm_st":  fig7.add_subplot(gs7[3, 2]),
    "stats":   fig7.add_subplot(gs7[3, 3]),
}
for ax in panels.values(): style_ax(ax)

# Price
market_px_series = np.exp(np.cumsum(market_ret))
panels["price"].plot(range(T_TOTAL), market_px_series, color=BLUE, lw=1.0)
panels["price"].axvspan(CRISIS_START, CRISIS_END, alpha=0.2, color=RED)
if first_alarm: panels["price"].axvline(first_alarm, color=GOLD, lw=1.5, ls="--")
panels["price"].set_title("Market Price", color=TEXT, fontsize=9, fontweight="bold")

# Ricci
panels["ricci"].plot(t_centers, ricci_ts, color=ORANGE, lw=1.2)
panels["ricci"].axhline(0.75, color=RED, lw=0.8, ls="--")
panels["ricci"].axvspan(CRISIS_START, CRISIS_END, alpha=0.15, color=RED)
panels["ricci"].set_title("Ricci κ", color=TEXT, fontsize=9, fontweight="bold")

# Singularity score
panels["sing"].plot(t_centers, sing_scores, color=PINK, lw=1.2)
panels["sing"].fill_between(t_centers, 0, sing_scores, alpha=0.2, color=PINK)
panels["sing"].axhline(0.75, color=RED, lw=0.8, ls="--")
panels["sing"].set_title("Singularity Score", color=TEXT, fontsize=9, fontweight="bold")

# Wormhole count
wh_counts = [network_stats["Pre-Crisis"]["n_wormholes"],
             network_stats["Peak Crisis"]["n_wormholes"],
             network_stats["Post-Crisis"]["n_wormholes"]]
panels["worm"].bar(["Pre", "Crisis", "Post"], wh_counts, color=[GREEN, RED, BLUE], edgecolor=DARK_BG, alpha=0.85)
panels["worm"].set_title("Wormhole Count", color=TEXT, fontsize=9, fontweight="bold")
panels["worm"].tick_params(colors=TEXT, labelsize=8)

# T df over time
panels["tdf"].plot(range(T_TOTAL), df_series, color=PURPLE, lw=1.0, alpha=0.8)
panels["tdf"].axhline(5.0, color=RED, lw=0.8, ls="--")
panels["tdf"].axvspan(CRISIS_START, CRISIS_END, alpha=0.15, color=RED)
panels["tdf"].set_title("Student-T df", color=TEXT, fontsize=9, fontweight="bold")

# Causal erasure delta
panels["causal"].bar(sizes, deltas, color=[GREEN if d>0.01 else RED if d<-0.01 else MUTED for d in deltas],
                     edgecolor=DARK_BG, alpha=0.85, width=0.5)
panels["causal"].axhline(0, color=MUTED, lw=0.7)
panels["causal"].set_title("do(X) → ΔH₁", color=TEXT, fontsize=9, fontweight="bold")
panels["causal"].set_xlabel("Intervention σ", color=MUTED, fontsize=7)

# Epistemic uncertainty
panels["ep"].fill_between(t_ep, smooth(ep_uncertainty, 15), alpha=0.7, color=RED, label="Epistemic")
panels["ep"].fill_between(t_ep, smooth(al_uncertainty, 15), alpha=0.5, color=BLUE, label="Aleatoric")
panels["ep"].axvspan(CRISIS_START, CRISIS_END, alpha=0.1, color=RED)
panels["ep"].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=6)
panels["ep"].set_title("Uncertainty Split", color=TEXT, fontsize=9, fontweight="bold")

# Equity curves
x_eq = range(n_min)
panels["eq"].plot(x_eq, eq_ens, color=GREEN, lw=1.8, label=f"Epistemic D3QN S={sharpe(eq_ens):.2f}")
panels["eq"].plot(x_eq, eq_deb, color=CYAN,  lw=1.4, label=f"Bayesian Debate S={sharpe(eq_deb):.2f}")
panels["eq"].plot(x_eq, eq_bh,  color=ORANGE,lw=1.2, label=f"BH Baseline S={sharpe(eq_bh):.2f}")
panels["eq"].set_yscale("log")
panels["eq"].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=7)
panels["eq"].set_title("Equity Curves (Log)", color=TEXT, fontsize=9, fontweight="bold")
e2_x = CRISIS_START - 50
e3_x = CRISIS_END - 50
if 0 < e2_x < n_min: panels["eq"].axvline(e2_x, color=ORANGE, lw=1.0, ls="--", alpha=0.7)
if 0 < e3_x < n_min: panels["eq"].axvline(e3_x, color=RED, lw=1.0, ls="--", alpha=0.7)

# Credibility
for i, (agent, col) in enumerate(zip(agents, agent_colors)):
    creds = credibility_log[agent.name]
    panels["cred"].plot(agent_cred_t[:len(creds)], creds, color=col, lw=1.2, label=agent.name[:10])
panels["cred"].axhline(0.5, color=MUTED, lw=0.7, ls="--")
panels["cred"].legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=6)
panels["cred"].set_title("Agent Credibility", color=TEXT, fontsize=9, fontweight="bold")

# Veto
panels["veto"].plot(agent_cred_t[:len(veto_roll)], veto_roll, color=PURPLE, lw=1.2)
panels["veto"].fill_between(agent_cred_t[:len(veto_roll)], 0, veto_roll, alpha=0.2, color=PURPLE)
panels["veto"].axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color=RED)
panels["veto"].set_title("Erasure Veto Rate", color=TEXT, fontsize=9, fontweight="bold")

# Epistemic ratio
ep_rat_t = np.arange(50, 50 + len(ep_ratio))
panels["ep_rat"].plot(ep_rat_t, smooth(ep_ratio, 20), color=PURPLE, lw=1.5)
panels["ep_rat"].fill_between(ep_rat_t, 0, smooth(ep_ratio, 20), alpha=0.2, color=PURPLE)
panels["ep_rat"].axhline(0.5, color=ORANGE, lw=0.8, ls="--")
panels["ep_rat"].axvspan(CRISIS_START, CRISIS_END, alpha=0.12, color=RED)
if first_alarm: panels["ep_rat"].axvline(first_alarm, color=GOLD, lw=1.5, ls="-")
panels["ep_rat"].set_title("Epistemic Fraction (spikes at singularity)", color=TEXT, fontsize=9, fontweight="bold")
panels["ep_rat"].set_xlabel("Bar", color=MUTED, fontsize=8)

# HMM state distribution
panels["hmm_st"].bar(range(3), [np.bincount(t_states)[i] if i < len(np.bincount(t_states)) else 0 for i in range(3)],
                     color=[GREEN, ORANGE, RED], edgecolor=DARK_BG, alpha=0.85)
panels["hmm_st"].set_xticks(range(3))
panels["hmm_st"].set_xticklabels([f"S{i}\ndf={t_hmm['df'][i]:.1f}" for i in range(3)], color=TEXT, fontsize=8)
panels["hmm_st"].set_title("T-HMM States", color=TEXT, fontsize=9, fontweight="bold")

# Stats
early_warning_bars = (CRISIS_START - first_alarm) if first_alarm and first_alarm < CRISIS_START else 0
stats_lines = [
    "PROJECT SINGULARITY",
    "",
    f"D3QN Sharpe:    {sharpe(eq_ens):+.3f}",
    f"Debate Sharpe:  {sharpe(eq_deb):+.3f}",
    f"BH Sharpe:      {sharpe(eq_bh):+.3f}",
    "",
    f"Early warning:  {early_warning_bars} bars",
    f"Alarm at:       bar {first_alarm}",
    f"Crisis at:      bar {CRISIS_START}",
    "",
    f"Causal erasure: {'YES' if causal_erasure_confirmed else 'NO'}",
    f"Δ range:        {delta_range:.4f}",
    "",
    f"Wormholes peak: {network_stats['Peak Crisis']['n_wormholes']}",
    f"Crisis df:      {t_hmm['df'][crisis_state_idx]:.1f}",
    "",
    f"Ep. unc ratio:  {ep_crisis/(ep_stable+1e-10):.1f}x spike",
]
panels["stats"].set_facecolor("#0a0f14")
panels["stats"].axis("off")
for sp in panels["stats"].spines.values(): sp.set_edgecolor(CYAN)
for i, line in enumerate(stats_lines):
    c  = CYAN if i == 0 else (GREEN if "+" in line else RED if "NO" in line or (": " in line and i > 5) else TEXT)
    sz = 10 if i == 0 else 8
    fw = "bold" if i == 0 else "normal"
    panels["stats"].text(0.05, 1 - i*0.062, line, transform=panels["stats"].transAxes,
                         color=c, fontsize=sz, fontweight=fw, va="top", fontfamily="monospace")

fig7.suptitle("PROJECT SINGULARITY  |  SRFM Lab  |  Causal Erasure & Relativistic Market Topology",
              color=TEXT, fontsize=15, fontweight="bold")
plt.savefig(os.path.join(OUT_DIR, "SG_07_full_dashboard.png"),
            dpi=180, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("    Saved: SG_07_full_dashboard.png")


# ═════════════════════════════════════════════════════════════════════════════
# GEMMA: WRITE THE LINKEDIN POST
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Prompting Gemma to write the Phase II LinkedIn post...")
print("=" * 70)

import ollama

summary = (
    f"PROJECT SINGULARITY — Phase II Results\n\n"
    f"UNIVERSE: {N_ASSETS} assets (TradFi:{N_TRADFI} + Crypto:{N_CRYPTO} + DeFi:{N_DEFI}) | {T_TOTAL} bars\n"
    f"Crisis window: bars {CRISIS_START}-{CRISIS_END}\n\n"
    f"PERFORMANCE:\n"
    f"- Epistemic D3QN Ensemble:  Sharpe = {sharpe(eq_ens):.3f}\n"
    f"- Bayesian Debate System:   Sharpe = {sharpe(eq_deb):.3f}\n"
    f"- BH Physics Baseline:      Sharpe = {sharpe(eq_bh):.3f}\n\n"
    f"SINGULARITY DETECTION:\n"
    f"- First singularity alarm: bar {first_alarm}\n"
    f"- Crisis onset: bar {CRISIS_START}\n"
    f"- Early warning lead: {early_warning_bars} bars BEFORE crisis\n"
    f"- Ricci curvature at alarm: {ricci_ts[np.argmin(np.abs(t_centers - first_alarm))]:.4f}\n\n"
    f"CAUSAL ERASURE:\n"
    f"- do(X) intervention delta range: {delta_range:.4f}\n"
    f"- Confirmed: {causal_erasure_confirmed}\n"
    f"- Meaning: {'even a +3σ liquidity injection into Crypto had near-ZERO effect on TradFi topology' if causal_erasure_confirmed else 'some causal structure remains'}\n\n"
    f"STUDENT-T HMM:\n"
    f"- Crisis state df: {t_hmm['df'][crisis_state_idx]:.1f} (below 5 = dangerously fat tails)\n"
    f"- Gaussian model would have missed {100*(1-student_t.cdf(3, df=t_hmm['df'][crisis_state_idx])):.1f}% of extreme tail events\n\n"
    f"WORMHOLE CONTAGION:\n"
    f"- Pre-crisis wormholes: {network_stats['Pre-Crisis']['n_wormholes']}\n"
    f"- Peak-crisis wormholes: {network_stats['Peak Crisis']['n_wormholes']}\n"
    f"- Post-crisis wormholes: {network_stats['Post-Crisis']['n_wormholes']}\n\n"
    f"EPISTEMIC UNCERTAINTY:\n"
    f"- Crisis epistemic spike: {ep_crisis/(ep_stable+1e-10):.1f}x above stable baseline\n"
    f"- Agent cash allocation (epistemic veto): {cash_pct:.1%}\n\n"
    f"BAYESIAN DEBATE CREDIBILITY:\n"
    + "\n".join(f"  - {a.name}: {a.credibility:.3f}" for a in agents)
    + f"\n\n7 CHARTS PRODUCED:\n"
    f"1. Ricci curvature + H1 collapse + singularity countdown\n"
    f"2. Cross-asset wormhole network (TradFi/Crypto/DeFi)\n"
    f"3. Student-T HMM fat tail regime detection\n"
    f"4. do-calculus causal erasure measurement\n"
    f"5. Epistemic vs aleatoric uncertainty decomposition\n"
    f"6. Bayesian credibility evolution\n"
    f"7. Full Singularity dashboard\n"
)

prompt = (
    "Write the Phase II LinkedIn post for PROJECT SINGULARITY. "
    "This is a sequel to Phase I (Project Event Horizon). "
    "Phase I proved markets have topology. "
    "Phase II proves that at the singularity, do(X) causal interventions become POWERLESS.\n\n"
    "The post must be legendary. No word limit. No restrictions.\n\n"
    "TONE: A researcher who just found something that breaks central bank thinking. "
    "Calm but earth-shattering. Speak to quants, ML researchers, and macro traders.\n\n"
    "STRUCTURE:\n"
    "1. ONE opening sentence. The most radical claim you can back with the data.\n"
    "2. Phase I recap in 3 sentences (what we already proved)\n"
    "3. The Phase II question: can intervention save a market at the singularity?\n"
    "4. Architecture: 5 new components (Ricci curvature, Student-T HMM, Wormhole network, do-calculus, Epistemic D3QN)\n"
    "5. Walk through all 7 charts with specific numbers\n"
    "6. The Causal Erasure finding — what it means for central banks, market makers, quant funds\n"
    "7. Code sample: Student-T HMM + Ricci curvature (40-50 lines)\n"
    "8. The Implication: what does 'causal erasure' mean for systematic trading?\n"
    "9. The most controversial claim — one paragraph that will get 500 comments\n"
    "10. Call to action + GitHub\n"
    "11. 8 hashtags\n\n"
    + summary
)

result = ollama.chat(
    model="gemma4-opt",
    messages=[
        {"role": "system", "content": (
            "You are the lead researcher on Project Event Horizon / Project Singularity. "
            "You have just proved that central bank interventions are mathematically powerless during a market singularity. "
            "Write with the confidence of someone who has the data to back every claim."
        )},
        {"role": "user", "content": prompt}
    ],
    options={"num_ctx": 32768, "temperature": 0.85}
)

post_path = os.path.join(OUT_DIR, "SG_linkedin_post.md")
with open(post_path, "w", encoding="utf-8") as f:
    f.write("# PROJECT SINGULARITY — LinkedIn Post\n\n")
    f.write(result.message.content)

print(f"LinkedIn post saved -> {post_path}")
print("\n" + "=" * 70)
print("ALL OUTPUTS -> Desktop/srfm-experiments/")
print("=" * 70)
print(f"\nKEY FINDINGS:")
print(f"  Singularity early warning:  {early_warning_bars} bars before crisis")
print(f"  Causal erasure confirmed:   {causal_erasure_confirmed}")
print(f"  do(X) delta range:          {delta_range:.5f}")
print(f"  Epistemic spike at crisis:  {ep_crisis/(ep_stable+1e-10):.1f}x")
print(f"  Crisis state df:            {t_hmm['df'][crisis_state_idx]:.1f}")
print(f"  Peak wormholes:             {network_stats['Peak Crisis']['n_wormholes']}")
print()
sys.stdout.buffer.write(result.message.content.encode("utf-8", "replace"))
print()
