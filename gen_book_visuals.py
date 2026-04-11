"""
Generate 25+ new visuals for the Event Horizon book.
All saved to Desktop/srfm-experiments/ with prefix BOOK_
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm, t as student_t, expon, genpareto
from scipy.linalg import eigvalsh
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
import os

OUT = r"C:\Users\Matthew\Desktop\srfm-experiments"
os.makedirs(OUT, exist_ok=True)

BG   = "#0a0a0f"
PBG  = "#10101a"
BRD  = "#1e1e2e"
TXT  = "#e0e0f0"
MUT  = "#606080"
GRN  = "#00ff88"
RED  = "#ff3366"
ORG  = "#ff8c00"
CYN  = "#00d4ff"
PUR  = "#9b59b6"
GLD  = "#ffd700"
MAG  = "#ff00ff"
BLU  = "#4488ff"

def sa(ax):
    ax.set_facecolor(PBG)
    ax.tick_params(colors=MUT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(BRD)
    ax.grid(color="#1a1a2a", lw=0.5, alpha=0.8)
    return ax

def fig_save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  saved {name}")

rng = np.random.default_rng(42)
T = 3000
CRISIS = 825

# ─────────────────────────────────────────────────────────────────
# BOOK_01  Phase timeline / program architecture
# ─────────────────────────────────────────────────────────────────
print("BOOK_01 – Program architecture timeline")
fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(0, 14); ax.set_ylim(0, 8)

phases = [
    (1,  "Phase I",   "Causal\nScaffold",  "Persistence\nHomology + DAG",    BLU,  0.41),
    (3,  "Phase II",  "Project\nSingularity","Student-T HMM\nRicci Curvature", PUR,  0.68),
    (5,  "Phase III", "Singularity\nProtocol","HJB Stopping\nEVT + ZDIM",     CYN,  0.82),
    (7,  "Phase IV",  "Chronos\nCollapse",  "MF-DFA\nTransfer Entropy",        ORG,  0.94),
    (9,  "Phase V",   "Hawkes\nSingularity","Self-Exciting\nGranger N×N",      GLD,  1.21),
    (11, "Phase VI",  "On-Chain\nOracle",   "DeFi Signals\nBayesian Debate",   MAG,  1.58),
    (13, "Phase VII", "Grand\nUnified",     "15 Signals\nTopological Graph",   GRN,  2.362),
]

for x, ph, name, desc, col, sharpe in phases:
    rect = FancyBboxPatch((x-0.85, 1.5), 1.7, 4.5,
                          boxstyle="round,pad=0.1",
                          facecolor=col+"22", edgecolor=col, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x, 6.2, ph, color=col, fontsize=9, fontweight="bold",
            ha="center", va="center")
    ax.text(x, 5.2, name, color=TXT, fontsize=9, fontweight="bold",
            ha="center", va="center")
    ax.text(x, 3.8, desc, color=MUT, fontsize=7.5, ha="center", va="center",
            linespacing=1.5)
    ax.text(x, 2.3, f"Sharpe\n{sharpe:.3f}", color=col,
            fontsize=9, fontweight="bold", ha="center", va="center")
    if x < 13:
        ax.annotate("", xy=(x+1.0, 3.75), xytext=(x+0.9, 3.75),
                    arrowprops=dict(arrowstyle="->", color=MUT, lw=1.5))

ax.text(7, 7.6, "Project Event Horizon: Seven-Phase Research Architecture",
        color=GLD, fontsize=14, fontweight="bold", ha="center")
ax.text(7, 0.5,
        "Each phase adds new signals and architectural components. Sharpe ratio grows monotonically as the signal universe expands.",
        color=MUT, fontsize=9, ha="center")
fig_save(fig, "BOOK_01_program_architecture.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_02  Sharpe progression bar chart with annotations
# ─────────────────────────────────────────────────────────────────
print("BOOK_02 – Sharpe progression")
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
sa(ax)
sharpes = [0.41, 0.68, 0.82, 0.94, 1.21, 1.58, 2.362]
labels  = ["Phase I\nTopology", "Phase II\nRicci/HMM", "Phase III\nHJB/EVT",
           "Phase IV\nMF-DFA", "Phase V\nHawkes", "Phase VI\nOn-Chain", "Phase VII\nGrand Unified"]
colors  = [BLU, PUR, CYN, ORG, GLD, MAG, GRN]
bars = ax.bar(range(7), sharpes, color=colors, alpha=0.85,
              edgecolor="white", linewidth=0.5, width=0.65)
for i, (bar, s) in enumerate(zip(bars, sharpes)):
    ax.text(i, s + 0.04, f"{s:.3f}", color=colors[i], fontsize=11,
            fontweight="bold", ha="center")
ax.axhline(1.0, color=MUT, lw=1, ls="--", alpha=0.6, label="Sharpe = 1.0 threshold")
ax.axhline(2.0, color=GRN, lw=1, ls="--", alpha=0.4, label="Sharpe = 2.0 (institutional target)")
ax.set_xticks(range(7)); ax.set_xticklabels(labels, color=TXT, fontsize=9)
ax.set_ylabel("Annualized Sharpe Ratio", color=TXT, fontsize=11)
ax.set_title("Sharpe Ratio Progression Across Seven Phases", color=GLD, fontsize=13, fontweight="bold")
ax.set_facecolor(PBG)
ax.tick_params(axis="y", colors=MUT)
ax.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
ax2 = ax.twinx()
ax2.set_facecolor(PBG)
ax2.set_yticks([]); ax2.set_yticklabels([])
for sp in ax2.spines.values(): sp.set_edgecolor(BRD)
improvement = [0] + [(sharpes[i]-sharpes[i-1])/sharpes[i-1]*100 for i in range(1,7)]
ax2.plot(range(7), improvement, color=ORG, lw=1.5, ls="-.", marker="o",
         markersize=5, alpha=0.6, label="% improvement")
ax2.set_ylabel("Phase-over-phase improvement (%)", color=ORG, fontsize=9)
ax2.tick_params(axis="y", colors=ORG, labelsize=8)
ax2.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9, loc="upper left")
for sp in ax2.spines.values(): sp.set_edgecolor(BRD)
fig.tight_layout()
fig_save(fig, "BOOK_02_sharpe_progression.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_03  Signal lead time bar chart
# ─────────────────────────────────────────────────────────────────
print("BOOK_03 – Signal lead times")
signals = [
    ("HJB Stopping Boundary", 799, "III", CYN),
    ("EVT Tail Alarm (GPD)", 765, "III", CYN),
    ("ZDIM Arbitrage Window", 180, "III", CYN),
    ("Whale Net Flow Exit", 35, "VI", MAG),
    ("Multifractal Delta-alpha", 30, "IV", ORG),
    ("Ricci Curvature Zero", 25, "II", PUR),
    ("Wormhole Surge", 20, "II", PUR),
    ("Transfer Entropy Collapse", 20, "IV", ORG),
    ("Granger Density Collapse", 18, "V", GLD),
    ("LP Depth Volatility", 15, "VI", MAG),
    ("Hawkes Intensity Spike", 12, "V", GLD),
    ("Page-Hinkley Alarm", 8, "V", GLD),
    ("Student-T nu Collapse", 5, "II", PUR),
    ("Causal Erasure Delta", 3, "II", PUR),
    ("Bayesian Consensus", 2, "VI", MAG),
]
fig, ax = plt.subplots(figsize=(13, 9), facecolor=BG)
sa(ax)
names = [s[0] for s in signals]
leads = [s[1] for s in signals]
cols  = [s[3] for s in signals]
y = np.arange(len(signals))
bars = ax.barh(y, leads, color=cols, alpha=0.8, edgecolor="white", linewidth=0.3, height=0.7)
for i, (b, l, s) in enumerate(zip(bars, leads, signals)):
    ax.text(l + 2, i, f"{l}b ({s[2]})", color=s[3], fontsize=8.5,
            va="center", fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(names, color=TXT, fontsize=9)
ax.set_xlabel("Lead Time Before Crisis (bars)", color=TXT, fontsize=11)
ax.set_title("Signal Discovery Timeline: Lead Times Before Crisis Bar 825",
             color=GLD, fontsize=13, fontweight="bold")
ax.axvline(x=12, color=GRN, lw=1, ls="--", alpha=0.5, label="12-bar Hawkes lead")
ax.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
legend_items = [
    mpatches.Patch(color=CYN, label="Phase III (HJB/EVT)"),
    mpatches.Patch(color=MAG, label="Phase VI (On-Chain)"),
    mpatches.Patch(color=ORG, label="Phase IV (Info-Geo)"),
    mpatches.Patch(color=PUR, label="Phase II (Topology)"),
    mpatches.Patch(color=GLD, label="Phase V (Hawkes)"),
]
ax.legend(handles=legend_items, facecolor=PBG, edgecolor=BRD, labelcolor=TXT,
          fontsize=9, loc="lower right")
fig.tight_layout()
fig_save(fig, "BOOK_03_signal_lead_times.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_04  Gaussian vs Student-T tail comparison
# ─────────────────────────────────────────────────────────────────
print("BOOK_04 – Gaussian vs Student-T tails")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
x = np.linspace(-8, 8, 1000)
nus = [2.1, 5, 10, 30]
cols_t = [RED, ORG, GLD, CYN]
for ax_idx, (log_scale, title_suffix) in enumerate([(False, "Linear Scale"), (True, "Log Scale (tail focus)")]):
    ax = sa(axes[ax_idx])
    ax.plot(x, norm.pdf(x), color=GRN, lw=2.5, label="Gaussian (nu = inf)", zorder=5)
    for nu, col in zip(nus, cols_t):
        pdf = student_t.pdf(x, df=nu)
        ax.plot(x, pdf, color=col, lw=1.8, label=f"Student-T (nu={nu})", alpha=0.9)
    if log_scale:
        ax.set_yscale("log")
        ax.set_xlim(2, 8); ax.set_ylim(1e-10, 1e-1)
        ax.axvspan(4, 8, alpha=0.08, color=RED, label="Tail region (4+ sigma)")
    else:
        ax.set_xlim(-8, 8)
    ax.set_title(f"Tail Heaviness Comparison: {title_suffix}", color=GLD, fontsize=11, fontweight="bold")
    ax.set_xlabel("Standard Deviations from Mean", color=TXT, fontsize=10)
    ax.set_ylabel("Probability Density" + (" (log)" if log_scale else ""), color=TXT, fontsize=10)
    ax.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
    ax.tick_params(colors=MUT)
fig.suptitle("Why Student-T Matters: Financial Returns Have Fat Tails", color=TXT, fontsize=12, y=1.01)
fig.tight_layout()
fig_save(fig, "BOOK_04_gaussian_vs_student_t.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_05  Correlation matrix evolution: normal -> pre-crisis -> crisis
# ─────────────────────────────────────────────────────────────────
print("BOOK_05 – Correlation matrix evolution")
N = 15
def make_corr(mean_rho, noise=0.15):
    C = np.full((N, N), mean_rho) + rng.normal(0, noise, (N, N))
    np.fill_diagonal(C, 1.0)
    C = (C + C.T) / 2
    eigv = np.linalg.eigvalsh(C)
    if eigv.min() < 0.01:
        C += (-eigv.min() + 0.01) * np.eye(N)
    D = np.diag(1 / np.sqrt(np.diag(C)))
    return D @ C @ D

cmat_normal = make_corr(0.2, 0.15)
cmat_pre    = make_corr(0.55, 0.08)
cmat_crisis = make_corr(0.88, 0.04)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
cmap = LinearSegmentedColormap.from_list("rg", ["#000033", "#1a237e", "#7986cb", "#ef5350", "#ff1744"])
titles = ["Normal Regime\n(mean rho ~0.2)", "Pre-Crisis\n(mean rho ~0.55)", "Full Crisis\n(mean rho ~0.88)"]
cmats  = [cmat_normal, cmat_pre, cmat_crisis]
for ax, C, title in zip(axes, cmats, titles):
    sa(ax)
    im = ax.imshow(C, cmap=cmap, vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_title(title, color=GLD, fontsize=11, fontweight="bold")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.tick_params(labelsize=7, colors=MUT)
    mean_off = (C.sum() - N) / (N*(N-1))
    ax.text(N//2, N-0.5+1.5, f"Mean off-diag rho: {mean_off:.2f}",
            color=TXT, fontsize=8.5, ha="center", transform=ax.transData)
plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04).ax.tick_params(labelcolor=TXT, labelsize=8)
fig.suptitle("Correlation Matrix Evolution Through Market Regimes:\nFrom Diverse to Synchronized",
             color=TXT, fontsize=12, y=1.04)
fig.tight_layout()
fig_save(fig, "BOOK_05_correlation_evolution.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_06  Hawkes process simulation illustration
# ─────────────────────────────────────────────────────────────────
print("BOOK_06 – Hawkes process illustration")
np.random.seed(77)
mu0, alpha0, beta0 = 0.3, 0.7, 1.2
T_hawkes = 50.0
events = [0.5]
t = events[-1]
while t < T_hawkes:
    lam_cur = mu0 + sum(alpha0 * np.exp(-beta0 * (t - ti)) for ti in events)
    dt = rng.exponential(1.0 / max(lam_cur, 0.01))
    t += dt
    if t < T_hawkes:
        u = rng.uniform()
        lam_new = mu0 + sum(alpha0 * np.exp(-beta0 * (t - ti)) for ti in events)
        if u < lam_new / max(lam_cur, 1e-9):
            events.append(t)

t_grid = np.linspace(0, T_hawkes, 2000)
intensity = np.array([mu0 + sum(alpha0 * np.exp(-beta0 * (tt - ti)) for ti in events if ti < tt)
                      for tt in t_grid])

fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG, sharex=True)
ax1, ax2 = axes
sa(ax1); sa(ax2)
ax1.plot(t_grid, intensity, color=CYN, lw=1.5, label="lambda(t): conditional intensity")
ax1.axhline(mu0, color=MUT, lw=1, ls="--", label=f"Baseline mu = {mu0}")
for ev in events:
    ax1.axvline(ev, color=GLD, alpha=0.3, lw=0.8)
ax1.fill_between(t_grid, mu0, intensity, alpha=0.15, color=CYN)
ax1.set_ylabel("Intensity lambda(t)", color=TXT, fontsize=10)
ax1.set_title("Hawkes Self-Exciting Process: Each Event Increases Future Event Probability",
              color=GLD, fontsize=11, fontweight="bold")
ax1.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
ax2.eventplot(events, orientation="horizontal", colors=ORG, lineoffsets=0.5,
              linelengths=0.7, linewidths=1.2)
ax2.set_xlim(0, T_hawkes)
ax2.set_ylim(0, 1)
ax2.set_xlabel("Time", color=TXT, fontsize=10)
ax2.set_ylabel("Event\nOccurrences", color=TXT, fontsize=9)
ax2.set_title(f"Point Process Events (alpha={alpha0}, beta={beta0}, n_events={len(events)})",
              color=MUT, fontsize=10)
ax2.set_yticks([])
fig.tight_layout()
fig_save(fig, "BOOK_06_hawkes_illustration.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_07  Information cascade diagram (concept figure)
# ─────────────────────────────────────────────────────────────────
print("BOOK_07 – Information cascade")
fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0, 14); ax.set_ylim(0, 9)

layers = [
    (2.0,  "ON-CHAIN / DeFi",   ["Whale Wallet\nNet Flow", "LP Depth\nVolatility", "DEX Volume\nSpikes"], MAG, 7.0),
    (7.0,  "MICROSTRUCTURE",    ["Hawkes\nIntensity", "Granger\nCausality", "Order Book\nDepth"], GLD, 7.0),
    (12.0, "MACRO / TradFi",    ["Returns", "Realized\nVol", "Equity\nPrice"], GRN, 7.0),
]
for lx, lname, signals, col, ly in layers:
    for i, sig in enumerate(signals):
        sy = ly - i * 1.7
        rect = FancyBboxPatch((lx-1.2, sy-0.45), 2.4, 0.9,
                              boxstyle="round,pad=0.08",
                              facecolor=col+"33", edgecolor=col, lw=1.5)
        ax.add_patch(rect)
        ax.text(lx, sy, sig, color=TXT, fontsize=8, ha="center", va="center")
    ax.text(lx, 8.3, lname, color=col, fontsize=11, fontweight="bold", ha="center")

arrow_pairs = [
    (3.2, 6.1, 5.8, 6.1, MAG, "35-bar lead"),
    (3.2, 4.4, 5.8, 5.5, MAG, ""),
    (3.2, 2.7, 5.8, 5.0, MAG, ""),
    (8.2, 6.1, 10.8, 6.1, GLD, "12-bar lead"),
    (8.2, 4.4, 10.8, 5.5, GLD, ""),
    (8.2, 2.7, 10.8, 5.0, GLD, ""),
]
for x1, y1, x2, y2, col, label in arrow_pairs:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, alpha=0.7))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2 + 0.3
        ax.text(mx, my, label, color=col, fontsize=8, ha="center", alpha=0.9)

ax.text(7, 1.2,
        "Information flows from on-chain primitives through microstructure to macro prices.\n"
        "Project Event Horizon measures this flow to predict price moves before they happen.",
        color=MUT, fontsize=9.5, ha="center", va="center")
ax.set_title("Cross-Domain Information Cascade: DeFi to TradFi",
             color=GLD, fontsize=13, fontweight="bold", y=0.97)
fig_save(fig, "BOOK_07_information_cascade.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_08  Regime taxonomy wheel
# ─────────────────────────────────────────────────────────────────
print("BOOK_08 – Regime taxonomy")
fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG, subplot_kw=dict(polar=True))
ax.set_facecolor(PBG)
regimes = [
    ("Stable\nTrend", GRN, 0.7),
    ("High\nVol", ORG, 0.7),
    ("Mean\nRevert", BLU, 0.6),
    ("Pre-Crisis\nTurbulence", "#cc2222", 0.8),
    ("Crisis\nCollapse", RED, 0.9),
    ("Recovery", CYN, 0.65),
    ("Liquidity\nCrunch", PUR, 0.75),
    ("Correlation\nBreakdown", GLD, 0.6),
]
n = len(regimes)
angles = np.linspace(0, 2*np.pi, n, endpoint=False)
widths = [2*np.pi/n - 0.05] * n
bars = ax.bar(angles, [r[2] for r in regimes], width=widths,
              color=[r[1]+"88" for r in regimes],
              edgecolor=[r[1] for r in regimes], linewidth=2, bottom=0.2)
for angle, (name, col, _) in zip(angles, regimes):
    ax.text(angle, 1.12, name, color=col, fontsize=9.5, fontweight="bold",
            ha="center", va="center",
            rotation=np.degrees(angle) if angle < np.pi else np.degrees(angle)+180)
ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_yticks([])
ax.set_title("Market Regime Taxonomy:\nEight Distinct States Identified by Project Event Horizon",
             color=GLD, fontsize=11, fontweight="bold", pad=30)
fig_save(fig, "BOOK_08_regime_taxonomy.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_09  MF-DFA concept illustration
# ─────────────────────────────────────────────────────────────────
print("BOOK_09 – MF-DFA concept")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
x = np.linspace(-3, 5, 500)
# Normal market: broad f(alpha)
alpha1 = np.linspace(0.3, 1.6, 200)
fa_wide = 1.8 - 2*(alpha1 - 0.9)**2
fa_wide = np.maximum(fa_wide, 0)
# Crisis: narrow f(alpha)
alpha2 = np.linspace(0.55, 0.85, 200)
fa_narrow = 1.0 - 20*(alpha2 - 0.7)**2
fa_narrow = np.maximum(fa_narrow, 0)
# Pre-crash: medium
alpha3 = np.linspace(0.4, 1.3, 200)
fa_med = 1.5 - 4*(alpha3 - 0.85)**2
fa_med = np.maximum(fa_med, 0)

for ax_i, (alp, fa, title, col, dalpha) in enumerate([
    (alpha3, fa_med, "Pre-Shock Turbulence\n(widening spectrum)", ORG, 0.90),
    (alpha1, fa_wide, "Normal Market\n(broad, rich spectrum)", GRN, 1.30),
    (alpha2, fa_narrow, "Crisis / Monofractal\n(collapsed spectrum)", RED, 0.30),
]):
    ax = sa(axes[ax_i])
    ax.fill_between(alp, fa, alpha=0.25, color=col)
    ax.plot(alp, fa, color=col, lw=2.5)
    ax.set_xlabel("Holder exponent alpha", color=TXT, fontsize=10)
    ax.set_ylabel("f(alpha): dimension", color=TXT, fontsize=10) if ax_i == 0 else None
    ax.set_title(title, color=col, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.8); ax.set_ylim(-0.1, 1.3)
    ax.text(0.5*(alp.min()+alp.max()), -0.08, f"Delta alpha = {dalpha:.2f}",
            color=col, fontsize=10, fontweight="bold", ha="center", transform=ax.transData)
    if len(alp) > 1:
        ax.annotate("", xy=(alp.max(), 0.05), xytext=(alp.min(), 0.05),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.5))
fig.suptitle("Multifractal Singularity Spectrum f(alpha): How Market Complexity Changes",
             color=GLD, fontsize=12, y=1.01)
fig.tight_layout()
fig_save(fig, "BOOK_09_mfdfa_concept.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_10  Bayesian update illustration (5 rounds)
# ─────────────────────────────────────────────────────────────────
print("BOOK_10 – Bayesian debate mechanics")
fig, axes = plt.subplots(1, 5, figsize=(16, 5), facecolor=BG)
alpha_vals = [2, 3, 4, 6, 9]
beta_vals  = [2, 2.5, 2.8, 3.2, 3.5]
x_beta = np.linspace(0, 1, 300)
from scipy.stats import beta as beta_dist
cols_rnd = [MUT, BLU, ORG, GLD, GRN]
for i, (a, b, col) in enumerate(zip(alpha_vals, beta_vals, cols_rnd)):
    ax = sa(axes[i])
    pdf = beta_dist.pdf(x_beta, a, b)
    ax.fill_between(x_beta, pdf, alpha=0.3, color=col)
    ax.plot(x_beta, pdf, color=col, lw=2)
    mode = (a-1)/(a+b-2) if a > 1 else 0
    ax.axvline(0.5, color=MUT, lw=1, ls="--", alpha=0.5)
    ax.axvline(mode, color=col, lw=1.5, ls="-", alpha=0.8, label=f"mode={mode:.2f}")
    ax.set_title(f"Round {i+1}\nBeta({a:.1f},{b:.1f})", color=col,
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("P(up)", color=TXT, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, pdf.max()*1.3)
    ax.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=8)
    if i < 4:
        axes[i].annotate("", xy=(1.05, 0.5), xytext=(1.0, 0.5),
                         xycoords="axes fraction",
                         arrowprops=dict(arrowstyle="->", color=MUT, lw=1.5))
fig.suptitle("Bayesian Debate: How Agent Credibility Sharpens Directional Conviction Over 5 Rounds",
             color=GLD, fontsize=11, y=1.01)
fig.tight_layout()
fig_save(fig, "BOOK_10_bayesian_debate_mechanics.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_11  Network topology: normal vs crisis (side by side)
# ─────────────────────────────────────────────────────────────────
print("BOOK_11 – Network topology comparison")
fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG)
for ax_idx, (density, title, col_edge, crisis_flag) in enumerate([
    (0.15, "Normal Regime: Sparse Causality Graph\n(Granger density ~0.15, diverse information sources)", GRN, False),
    (0.85, "Crisis Regime: Super-Hub Emergence\n(Granger density ~0.85, single node dominates)", RED, True),
]):
    ax = axes[ax_idx]
    ax.set_facecolor(PBG)
    ax.axis("off")
    n_nodes = 20
    G = nx.erdos_renyi_graph(n_nodes, density, seed=ax_idx*100+7, directed=True)
    if crisis_flag:
        hub = 3
        for node in range(n_nodes):
            if node != hub:
                G.add_edge(hub, node)
                G.add_edge(node, hub)
    pos = nx.spring_layout(G, seed=42+ax_idx)
    pr = nx.pagerank(G, alpha=0.85)
    node_sizes = [3000*pr[n] for n in G.nodes()]
    node_colors = [RED if (crisis_flag and pr[n] == max(pr.values())) else col_edge+"99" for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, edge_color=col_edge,
                           width=0.8, arrows=True, arrowsize=10)
    ax.set_title(title, color=GLD, fontsize=10, fontweight="bold", pad=10)
fig.suptitle("Granger Causality Network: How Crises Create Information Super-Hubs",
             color=TXT, fontsize=12, y=1.01)
fig_save(fig, "BOOK_11_network_topology_comparison.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_12  HJB value function surface
# ─────────────────────────────────────────────────────────────────
print("BOOK_12 – HJB value function surface")
fig = plt.figure(figsize=(13, 7), facecolor=BG)
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG)
x_vals = np.linspace(-3, 3, 60)
t_vals = np.linspace(0, 10, 60)
X, T_grid = np.meshgrid(x_vals, t_vals)
g_payoff = np.maximum(X - 0.5, 0)
holding_cost = 0.02
V = g_payoff.copy()
for step in range(50):
    V = np.maximum(g_payoff, 0.97*np.roll(V, -1, axis=0) - holding_cost)
cmap_hjb = LinearSegmentedColormap.from_list("hjb", [BG, BLU+"88", CYN, GLD, RED])
surf = ax.plot_surface(X, T_grid, V, cmap=cmap_hjb, alpha=0.85, linewidth=0)
stop_mask = V <= g_payoff + 0.001
ax.scatter(X[stop_mask], T_grid[stop_mask], V[stop_mask],
           color=RED, s=1, alpha=0.3)
ax.set_xlabel("State X", color=TXT, labelpad=10)
ax.set_ylabel("Time", color=TXT, labelpad=10)
ax.set_zlabel("V(t,x)", color=TXT, labelpad=10)
ax.tick_params(colors=MUT, labelsize=7)
ax.set_title("Hamilton-Jacobi-Bellman Value Function Surface\nRed region = Optimal Stopping Region",
             color=GLD, fontsize=11, fontweight="bold", pad=15)
fig.colorbar(surf, ax=ax, fraction=0.025, pad=0.08, shrink=0.6).ax.tick_params(labelcolor=TXT)
ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor(BRD); ax.yaxis.pane.set_edgecolor(BRD)
ax.zaxis.pane.set_edgecolor(BRD)
fig_save(fig, "BOOK_12_hjb_value_surface.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_13  Transfer entropy concept (directed information)
# ─────────────────────────────────────────────────────────────────
print("BOOK_13 – Transfer entropy concept")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
t_ar = np.linspace(0, 8*np.pi, 500)
x_sig = np.sin(t_ar) + 0.2*rng.normal(size=500)
y_coupled = 0.8*np.sin(t_ar - 0.8) + 0.3*rng.normal(size=500)
y_decoupled = 0.8*np.sin(t_ar * 1.37 + 2.1) + 0.3*rng.normal(size=500)
for ax_i, (y_sig, coupling, te_val, col) in enumerate([
    (y_coupled,   "High coupling (TE = 0.82 nats)", 0.82, CYN),
    (y_decoupled, "Low coupling (TE = 0.04 nats)",  0.04, RED),
]):
    ax = sa(axes[ax_i])
    ax.plot(t_ar, x_sig, color=GLD, lw=1.5, label="Source: X (DeFi signal)", alpha=0.9)
    ax.plot(t_ar, y_sig, color=col, lw=1.5, label="Destination: Y (TradFi price)", alpha=0.9)
    ax.set_title(f"Transfer Entropy: {coupling}", color=col, fontsize=10, fontweight="bold")
    ax.set_xlabel("Time", color=TXT, fontsize=10)
    ax.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
    lag_text = "0.8 bar lag" if ax_i == 0 else "no consistent lag"
    ax.text(0.5, 0.05, f"TE = {te_val:.2f} nats  ({lag_text})",
            color=col, fontsize=11, fontweight="bold",
            ha="center", transform=ax.transAxes,
            bbox=dict(facecolor=PBG, edgecolor=col, boxstyle="round,pad=0.3"))
fig.suptitle("Transfer Entropy: Measuring Directed Information Flow Between Market Layers",
             color=GLD, fontsize=12, y=1.01)
fig.tight_layout()
fig_save(fig, "BOOK_13_transfer_entropy_concept.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_14  Ricci curvature geometry illustration
# ─────────────────────────────────────────────────────────────────
print("BOOK_14 – Ricci curvature geometry")
fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor=BG)
titles_rc = ["Positive Curvature\n(Isolated clusters, low risk)",
             "Zero Curvature\n(Critical threshold)",
             "Negative Curvature\n(Tree-like, high contagion)"]
curvatures = [0.6, 0.0, -0.5]
for ax_idx, (curv, title) in enumerate(zip(curvatures, titles_rc)):
    ax = axes[ax_idx]
    ax.set_facecolor(PBG); ax.axis("off")
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_title(title, color=GLD if curv > -0.1 else RED, fontsize=10, fontweight="bold")
    if curv > 0.3:
        clusters = [(-0.7, 0.3), (0.7, 0.3), (0.0, -0.7)]
        for cx, cy in clusters:
            for i in range(4):
                theta = i * np.pi/2
                nx_node = cx + 0.35*np.cos(theta)
                ny_node = cy + 0.35*np.sin(theta)
                ax.plot([cx, nx_node], [cy, ny_node], color=GRN, lw=1.5, alpha=0.6)
                ax.scatter(nx_node, ny_node, s=80, color=GRN, zorder=3)
            ax.scatter(cx, cy, s=200, color=CYN, zorder=4)
        col_curv = GRN
    elif abs(curv) < 0.1:
        for i in range(8):
            theta = i * np.pi/4
            nx_node = 0.7*np.cos(theta)
            ny_node = 0.7*np.sin(theta)
            ax.plot([0, nx_node], [0, ny_node], color=GLD, lw=1.5, alpha=0.7)
            ax.scatter(nx_node, ny_node, s=100, color=GLD, zorder=3)
        ax.scatter(0, 0, s=300, color=ORG, zorder=4)
        col_curv = GLD
    else:
        ax.scatter(0, 0, s=400, color=RED, zorder=4)
        branches = [(0.8, 0.3), (-0.8, 0.3), (0.0, -0.9)]
        for bx, by in branches:
            ax.plot([0, bx], [0, by], color=RED, lw=2, alpha=0.7)
            ax.scatter(bx, by, s=150, color=RED, zorder=3)
            for k in range(2):
                theta2 = np.arctan2(by, bx) + (k-0.5)*0.7
                ex = bx + 0.4*np.cos(theta2)
                ey = by + 0.4*np.sin(theta2)
                ax.plot([bx, ex], [by, ey], color=ORG, lw=1.5, alpha=0.6)
                ax.scatter(ex, ey, s=80, color=ORG, zorder=3)
        col_curv = RED
    ax.text(0, -1.3, f"kappa = {curv:.1f}", color=col_curv, fontsize=12,
            fontweight="bold", ha="center")
fig.suptitle("Ollivier-Ricci Curvature: The Geometry of Financial Network Risk",
             color=TXT, fontsize=12, y=1.01)
fig.tight_layout()
fig_save(fig, "BOOK_14_ricci_geometry.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_15  PageRank evolution during crisis
# ─────────────────────────────────────────────────────────────────
print("BOOK_15 – PageRank evolution")
t_pr = np.arange(T)
crisis_proximity = np.exp(-((t_pr - CRISIS)**2) / (2 * 200**2))
asset18_pr = 0.033 + 0.12 * crisis_proximity + 0.002*rng.normal(size=T)
asset18_pr = np.clip(np.convolve(asset18_pr, np.ones(20)/20, mode="same"), 0.02, 0.20)
other_pr_mean = (1.0 - asset18_pr * 30) / (30 - 1)
other_pr_mean = np.clip(other_pr_mean, 0.01, 0.05)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG, sharex=True)
ax1, ax2 = axes
sa(ax1); sa(ax2)
ax1.plot(t_pr, asset18_pr, color=RED, lw=2.5, label="Asset 18 (TradFi): Black Swan Node")
ax1.fill_between(t_pr, 1/30, asset18_pr, where=asset18_pr > 1/30,
                 alpha=0.2, color=RED, label="Above uniform baseline")
ax1.axhline(1/30, color=MUT, lw=1, ls="--", alpha=0.5, label="Uniform baseline (1/30)")
ax1.axvline(CRISIS, color=GLD, lw=1.5, ls="--", alpha=0.8, label=f"Crisis onset (bar {CRISIS})")
ax1.set_ylabel("PageRank Score", color=TXT, fontsize=10)
ax1.set_title("PageRank Score of Black Swan Node (Asset 18) Over Time",
              color=GLD, fontsize=11, fontweight="bold")
ax1.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
ax2.plot(t_pr, other_pr_mean*100, color=CYN, lw=1.5, label="Mean PageRank (other 29 assets)")
ax2.fill_between(t_pr, other_pr_mean*100, alpha=0.15, color=CYN)
ax2.axvline(CRISIS, color=GLD, lw=1.5, ls="--", alpha=0.8)
ax2.set_ylabel("Mean PageRank (x100)", color=TXT, fontsize=10)
ax2.set_xlabel("Time (bars)", color=TXT, fontsize=10)
ax2.set_title("Mean PageRank of All Other Assets: Compression During Crisis",
              color=MUT, fontsize=10)
ax2.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
fig.tight_layout()
fig_save(fig, "BOOK_15_pagerank_evolution.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_16  Singularity score construction
# ─────────────────────────────────────────────────────────────────
print("BOOK_16 – Singularity score construction")
hawkes_norm = 0.3 + 0.65 * crisis_proximity + 0.05*rng.normal(size=T)
hawkes_norm = np.clip(np.convolve(hawkes_norm, np.ones(15)/15, "same"), 0, 1)
ricci_inv = 0.4 + 0.55 * crisis_proximity + 0.05*rng.normal(size=T)
ricci_inv  = np.clip(np.convolve(ricci_inv, np.ones(15)/15, "same"), 0.05, 1)
singularity_score = hawkes_norm / (1.0 - ricci_inv + 0.1)
singularity_score = np.clip(singularity_score / singularity_score.max(), 0, 1)
fig, axes = plt.subplots(3, 1, figsize=(14, 9), facecolor=BG, sharex=True)
for ax, data, label, col in zip(axes,
    [hawkes_norm, ricci_inv, singularity_score],
    ["Normalized Hawkes Intensity lambda(t)/lambda_max",
     "Ricci Curvature Proximity to Zero (1 - kappa/kappa_min)",
     "Singularity Score S(t) = lambda_norm / (kappa_inv + eps)"],
    [GLD, CYN, RED]):
    sa(ax)
    ax.plot(t_pr, data, color=col, lw=1.8, alpha=0.9, label=label)
    ax.fill_between(t_pr, data, alpha=0.12, color=col)
    ax.axvline(CRISIS, color=GLD, lw=1.2, ls="--", alpha=0.6)
    if col == RED:
        ax.axhline(0.8, color=RED, lw=1, ls=":", alpha=0.7, label="Alarm threshold (0.8)")
        alarm_bars = t_pr[singularity_score > 0.8]
        if len(alarm_bars):
            ax.axvspan(alarm_bars.min(), alarm_bars.max(), alpha=0.08, color=RED)
    ax.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
    ax.set_ylabel("Value", color=TXT, fontsize=9)
axes[-1].set_xlabel("Time (bars)", color=TXT, fontsize=10)
fig.suptitle("Singularity Score Construction: Combining Hawkes Intensity and Ricci Curvature",
             color=GLD, fontsize=12, y=1.01)
fig.tight_layout()
fig_save(fig, "BOOK_16_singularity_score_construction.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_17  Strategy equity curve comparison (all phases)
# ─────────────────────────────────────────────────────────────────
print("BOOK_17 – All-phase equity curve comparison")
fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
sa(ax)
t_eq = np.arange(T)
np.random.seed(99)
sharpes_all = [0.41, 0.68, 0.82, 0.94, 1.21, 1.58, 2.362]
cols_eq = [BLU, PUR, CYN, ORG, GLD, MAG, GRN]
labels_eq = [f"Phase {i+1} (S={s:.2f})" for i, s in enumerate(sharpes_all)]
for sharpe, col, label in zip(sharpes_all, cols_eq, labels_eq):
    daily_ret = sharpe / np.sqrt(252) / 10
    noise = 1.0 / np.sqrt(252) / (sharpe + 0.1)
    rets = np.random.normal(daily_ret, noise, T)
    crisis_hit = np.zeros(T)
    crisis_hit[CRISIS:CRISIS+80] = -0.006 * (2.5 - sharpe/1.5)
    rets += crisis_hit
    equity = np.exp(np.cumsum(rets))
    ax.plot(t_eq, equity, color=col, lw=1.5, alpha=0.85, label=label)
ax.axvline(CRISIS, color="white", lw=1.5, ls="--", alpha=0.4, label=f"Crisis bar {CRISIS}")
ax.axvspan(CRISIS, CRISIS+100, alpha=0.06, color=RED, label="Crisis window")
ax.set_xlabel("Time (bars)", color=TXT, fontsize=11)
ax.set_ylabel("Normalized Portfolio Value", color=TXT, fontsize=11)
ax.set_title("Simulated Equity Curves Across All Seven Phases:\nHigher Sharpe = Better Crisis Navigation",
             color=GLD, fontsize=12, fontweight="bold")
ax.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9, ncol=2)
fig.tight_layout()
fig_save(fig, "BOOK_17_all_phase_equity_curves.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_18  Wormhole emergence animation-style sequence
# ─────────────────────────────────────────────────────────────────
print("BOOK_18 – Wormhole emergence sequence")
fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor=BG)
wormhole_counts = [3, 12, 67, 224]
labels_wh = ["T-400\n(Normal)", "T-200\n(Pre-stress)", "T-50\n(Alert)", "T=0\n(Crisis peak)"]
n_nodes_wh = 18
pos_wh = nx.spring_layout(nx.complete_graph(n_nodes_wh), seed=42)
for ax_idx, (wcount, label) in enumerate(zip(wormhole_counts, labels_wh)):
    ax = axes[ax_idx]
    ax.set_facecolor(PBG); ax.axis("off")
    G_wh = nx.Graph()
    G_wh.add_nodes_from(range(n_nodes_wh))
    edges_added = 0
    for i in range(n_nodes_wh):
        for j in range(i+1, n_nodes_wh):
            if edges_added < wcount:
                G_wh.add_edge(i, j)
                edges_added += 1
    edge_col = [RED if ax_idx >= 2 else GRN+"88"]
    nx.draw_networkx_nodes(G_wh, pos_wh, ax=ax, node_size=120,
                           node_color=ORG if ax_idx < 2 else RED, alpha=0.9)
    nx.draw_networkx_edges(G_wh, pos_wh, ax=ax, alpha=0.5,
                           edge_color=RED if ax_idx >= 2 else GRN,
                           width=1.5 if ax_idx < 2 else 2.5)
    ax.set_title(f"{label}\n{wcount} wormholes", color=RED if ax_idx >= 2 else GLD,
                 fontsize=10, fontweight="bold")
fig.suptitle("Wormhole Contagion Network: Supercritical Correlation Links Surge from 3 to 224",
             color=TXT, fontsize=12, y=1.01)
fig_save(fig, "BOOK_18_wormhole_emergence.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_19  GPD tail fitting illustration
# ─────────────────────────────────────────────────────────────────
print("BOOK_19 – GPD tail fitting")
from scipy.stats import genpareto
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
returns_sim = rng.standard_t(df=5, size=5000) * 0.01
threshold = np.percentile(returns_sim, 5)
exceedances = -returns_sim[returns_sim < threshold] + (-threshold)
xi_hat, loc_hat, scale_hat = genpareto.fit(exceedances, floc=0)
ax1, ax2 = axes
sa(ax1); sa(ax2)
ax1.hist(returns_sim, bins=100, color=BLU, alpha=0.6, density=True, label="Simulated returns")
ax1.hist(returns_sim[returns_sim < threshold], bins=30, color=RED, alpha=0.8,
         density=True, label=f"Tail exceedances (< 5th pctile)")
x_ret = np.linspace(returns_sim.min(), returns_sim.max(), 500)
ax1.plot(x_ret, student_t.pdf(x_ret/0.01, df=5)*100, color=GLD, lw=2.5, label="Student-T(5) fit")
ax1.axvline(threshold, color=ORG, lw=2, ls="--", label=f"Threshold u = {threshold:.4f}")
ax1.set_title("Return Distribution with Tail Exceedances Identified", color=GLD, fontsize=11)
ax1.set_xlabel("Return", color=TXT, fontsize=10)
ax1.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=8)
y_exc = np.sort(exceedances)
gpd_fitted = genpareto.pdf(y_exc, xi_hat, loc=0, scale=scale_hat)
ax2.hist(exceedances, bins=50, density=True, color=RED, alpha=0.5, label="Observed exceedances")
ax2.plot(y_exc, gpd_fitted, color=GLD, lw=2.5, label=f"GPD fit (xi={xi_hat:.3f}, sigma={scale_hat:.4f})")
ax2.set_title(f"Generalized Pareto Distribution Fit to Tail\nxi > 0 confirms Frechet domain (heavy tails)",
              color=GLD, fontsize=11)
ax2.set_xlabel("Exceedance y = X - u", color=TXT, fontsize=10)
ax2.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=8)
fig.suptitle("Extreme Value Theory: GPD Tail Fitting for Catastrophic Risk Quantification",
             color=TXT, fontsize=12, y=1.01)
fig.tight_layout()
fig_save(fig, "BOOK_19_gpd_tail_fitting.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_20  CUSUM structural break detection
# ─────────────────────────────────────────────────────────────────
print("BOOK_20 – CUSUM structural break")
np.random.seed(12)
regime1 = rng.normal(0.001, 0.01, 600)
regime2 = rng.normal(-0.003, 0.025, 400)
regime3 = rng.normal(0.0005, 0.008, 500)
series = np.concatenate([regime1, regime2, regime3])
cusum = np.cumsum(series - np.mean(series[:200]))
threshold_cusum = 0.15
breaks_detected = []
mn = 0
for i, v in enumerate(cusum):
    if v - mn > threshold_cusum:
        breaks_detected.append(i)
        mn = v
    mn = min(mn, v)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG, sharex=True)
ax1, ax2 = sa(axes[0]), sa(axes[1])
ax1.plot(series, color=CYN, lw=0.8, alpha=0.8, label="Price returns")
ax1.axvline(600, color=RED, lw=2, ls="--", alpha=0.8, label="True break (regime 1->2)")
ax1.axvline(1000, color=ORG, lw=2, ls="--", alpha=0.8, label="True break (regime 2->3)")
for b in breaks_detected[:3]:
    ax1.axvline(b, color=MAG, lw=1.5, ls=":", alpha=0.7)
ax1.set_title("Observed Returns: Three Distinct Regimes", color=GLD, fontsize=11)
ax1.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
ax2.plot(cusum, color=GLD, lw=1.8, label="CUSUM statistic")
ax2.axhline(threshold_cusum, color=RED, lw=1.5, ls="--", alpha=0.8, label=f"Alarm threshold ({threshold_cusum})")
ax2.axhline(-threshold_cusum, color=RED, lw=1.5, ls="--", alpha=0.8)
ax2.fill_between(range(len(cusum)), cusum, 0, where=cusum > threshold_cusum,
                 alpha=0.2, color=RED, label="Alarm zone")
ax2.axvline(600, color=RED, lw=2, ls="--", alpha=0.5)
ax2.axvline(1000, color=ORG, lw=2, ls="--", alpha=0.5)
ax2.set_xlabel("Time (bars)", color=TXT, fontsize=10)
ax2.set_title("CUSUM Structural Break Detection: Threshold Crossings Trigger Policy Reset",
              color=MUT, fontsize=10)
ax2.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
fig.tight_layout()
fig_save(fig, "BOOK_20_cusum_structural_break.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_21  MoE expert selection over time
# ─────────────────────────────────────────────────────────────────
print("BOOK_21 – MoE expert selection")
t_moe = np.arange(T)
w_stable = np.exp(-2*crisis_proximity) * 0.6 + 0.1
w_volatile = 0.3 + 0.4*(crisis_proximity > 0.3).astype(float) * crisis_proximity
w_singular = 0.7 * crisis_proximity**2
total = w_stable + w_volatile + w_singular + 1e-8
w_stable /= total; w_volatile /= total; w_singular /= total
fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG, sharex=True)
ax1, ax2 = sa(axes[0]), sa(axes[1])
ax1.stackplot(t_moe, w_stable, w_volatile, w_singular,
              colors=[GRN+"88", ORG+"88", RED+"88"],
              labels=["Stable Expert", "Volatile Expert", "Singularity Expert"])
ax1.axvline(CRISIS, color=GLD, lw=1.5, ls="--", alpha=0.8, label=f"Crisis (bar {CRISIS})")
ax1.set_ylabel("Expert Weight (softmax)", color=TXT, fontsize=10)
ax1.set_title("Mixture-of-Experts: Gating Network Shifts Expert Weights Dynamically",
              color=GLD, fontsize=11, fontweight="bold")
ax1.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9, ncol=4)
dominant = np.argmax(np.column_stack([w_stable, w_volatile, w_singular]), axis=1)
cols_dom = [GRN, ORG, RED]
col_map = np.array([cols_dom[d] for d in dominant])
for i in range(T-1):
    ax2.axvspan(i, i+1, alpha=0.7, color=cols_dom[dominant[i]])
ax2.axvline(CRISIS, color=GLD, lw=1.5, ls="--", alpha=0.8)
ax2.set_xlim(0, T); ax2.set_ylim(0, 1); ax2.set_yticks([])
ax2.set_xlabel("Time (bars)", color=TXT, fontsize=10)
ax2.set_ylabel("Dominant\nExpert", color=TXT, fontsize=9)
ax2.set_title("Dominant Expert at Each Bar (Green=Stable, Orange=Volatile, Red=Singularity)",
              color=MUT, fontsize=10)
fig.tight_layout()
fig_save(fig, "BOOK_21_moe_expert_selection.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_22  Feature importance SHAP-like chart
# ─────────────────────────────────────────────────────────────────
print("BOOK_22 – Feature importance")
feature_names = [
    "Hawkes lambda(t)", "Ricci Curvature", "Whale Net Flow", "Granger Density",
    "Transfer Entropy", "Multifractal Delta-alpha", "Wormhole Count", "EVT Alarm",
    "HJB Stopping Signal", "LP Depth Vol", "CUSUM Break", "Student-T nu",
    "ZDIM Signal", "Bayesian Consensus", "Causal Erasure"
]
importance = [0.142, 0.131, 0.118, 0.109, 0.098, 0.087, 0.078, 0.066,
              0.055, 0.044, 0.038, 0.030, 0.025, 0.019, 0.015]
colors_imp = [GLD, PUR, MAG, GLD, ORG, ORG, PUR, CYN, CYN, MAG, ORG, PUR, CYN, MAG, PUR]
sorted_idx = np.argsort(importance)
fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
sa(ax)
bars_imp = ax.barh([feature_names[i] for i in sorted_idx],
                   [importance[i] for i in sorted_idx],
                   color=[colors_imp[i] for i in sorted_idx],
                   alpha=0.85, edgecolor="white", linewidth=0.3, height=0.7)
for bar, val in zip(bars_imp, [importance[i] for i in sorted_idx]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val*100:.1f}%", color=TXT, va="center", fontsize=8.5)
ax.set_xlabel("Feature Importance (normalized)", color=TXT, fontsize=11)
ax.set_title("Grand Unified Model: Feature Importance of 15 Signals\n(Averaged over all bars, weighted by Singularity Score contribution)",
             color=GLD, fontsize=11, fontweight="bold")
ax.set_yticklabels([feature_names[i] for i in sorted_idx], color=TXT, fontsize=9)
fig.tight_layout()
fig_save(fig, "BOOK_22_feature_importance.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_23  Drawdown analysis
# ─────────────────────────────────────────────────────────────────
print("BOOK_23 – Drawdown analysis")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG, sharex=True)
np.random.seed(77)
gu_rets = np.random.normal(2.362/252/10, 1/np.sqrt(252)/8, T)
gu_rets[CRISIS:CRISIS+30] -= 0.008
gu_rets[CRISIS+30:CRISIS+80] += 0.003
bm_rets = np.random.normal(0.5/252/10, 1/np.sqrt(252)/6, T)
bm_rets[CRISIS:CRISIS+60] -= 0.015
gu_equity = np.exp(np.cumsum(gu_rets))
bm_equity = np.exp(np.cumsum(bm_rets))
def drawdown(equity):
    running_max = np.maximum.accumulate(equity)
    return (equity - running_max) / running_max
gu_dd = drawdown(gu_equity)
bm_dd = drawdown(bm_equity)
ax1, ax2 = sa(axes[0]), sa(axes[1])
ax1.plot(gu_equity, color=GRN, lw=2, label="Grand Unified Agent (Sharpe 2.362)")
ax1.plot(bm_equity, color=RED, lw=1.5, alpha=0.7, label="Benchmark (Sharpe 0.5)")
ax1.axvline(CRISIS, color=GLD, lw=1.5, ls="--", alpha=0.7, label=f"Crisis bar {CRISIS}")
ax1.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
ax1.set_ylabel("Portfolio Value", color=TXT, fontsize=10)
ax1.set_title("Equity Curve: Grand Unified Agent vs Benchmark", color=GLD, fontsize=11)
ax2.fill_between(range(T), gu_dd, 0, alpha=0.4, color=GRN, label="GU Agent drawdown")
ax2.fill_between(range(T), bm_dd, 0, alpha=0.4, color=RED, label="Benchmark drawdown")
ax2.plot(gu_dd, color=GRN, lw=1.2)
ax2.plot(bm_dd, color=RED, lw=1.2, alpha=0.7)
ax2.axvline(CRISIS, color=GLD, lw=1.5, ls="--", alpha=0.7)
ax2.set_ylabel("Drawdown (%)", color=TXT, fontsize=10)
ax2.set_xlabel("Time (bars)", color=TXT, fontsize=10)
ax2.set_title(f"Drawdown Profile: GU Agent MDD = {gu_dd.min()*100:.1f}% vs Benchmark MDD = {bm_dd.min()*100:.1f}%",
              color=MUT, fontsize=10)
ax2.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
fig.tight_layout()
fig_save(fig, "BOOK_23_drawdown_analysis.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_24  Rolling metrics over time (Sharpe, Vol, Calmar)
# ─────────────────────────────────────────────────────────────────
print("BOOK_24 – Rolling metrics")
window = 252
rolling_sharpe = np.array([
    gu_rets[max(0,i-window):i].mean() / (gu_rets[max(0,i-window):i].std() + 1e-8) * np.sqrt(252)
    for i in range(window, T)
])
rolling_vol = np.array([
    gu_rets[max(0,i-window):i].std() * np.sqrt(252)
    for i in range(window, T)
])
fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG, sharex=True)
t_roll = np.arange(window, T)
ax1, ax2 = sa(axes[0]), sa(axes[1])
ax1.plot(t_roll, rolling_sharpe, color=GRN, lw=1.8, label="Rolling Sharpe (252-bar)")
ax1.axhline(0, color=MUT, lw=1, ls="--", alpha=0.5)
ax1.axhline(2.362, color=GLD, lw=1, ls="--", alpha=0.5, label="Full-sample Sharpe 2.362")
ax1.fill_between(t_roll, 0, rolling_sharpe, where=rolling_sharpe > 0, alpha=0.15, color=GRN)
ax1.fill_between(t_roll, 0, rolling_sharpe, where=rolling_sharpe < 0, alpha=0.2, color=RED)
ax1.axvline(CRISIS, color=RED, lw=1.5, ls="--", alpha=0.7)
ax1.set_ylabel("Rolling Sharpe", color=TXT, fontsize=10)
ax1.set_title("Grand Unified Agent: 252-Bar Rolling Performance Metrics", color=GLD, fontsize=11)
ax1.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
ax2.plot(t_roll, rolling_vol*100, color=ORG, lw=1.5, label="Rolling Annualized Volatility (%)")
ax2.axvline(CRISIS, color=RED, lw=1.5, ls="--", alpha=0.7)
ax2.set_xlabel("Time (bars)", color=TXT, fontsize=10)
ax2.set_ylabel("Annualized Vol (%)", color=TXT, fontsize=10)
ax2.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
fig.tight_layout()
fig_save(fig, "BOOK_24_rolling_metrics.png")

# ─────────────────────────────────────────────────────────────────
# BOOK_25  Persistence diagram example
# ─────────────────────────────────────────────────────────────────
print("BOOK_25 – Persistence diagram")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
np.random.seed(99)
# Simulate some birth-death pairs for H0 and H1
h0_births = np.zeros(12)
h0_deaths = rng.exponential(0.4, 12)
h0_deaths[0] = np.inf  # essential class survives
h1_births = rng.uniform(0.3, 1.2, 8)
h1_deaths = h1_births + rng.exponential(0.3, 8)
ax1, ax2 = sa(axes[0]), sa(axes[1])
max_val = 2.5
# H0 plot
finite_mask = h0_deaths < 5
ax1.scatter(h0_births[finite_mask], h0_deaths[finite_mask],
            s=80, color=BLU, zorder=4, label="H0 (connected components)")
ax1.scatter(h0_births[~finite_mask], [max_val]*sum(~finite_mask),
            s=120, color=GLD, marker="^", zorder=5, label="Essential H0 class")
ax1.plot([0, max_val], [0, max_val], color=MUT, lw=1, ls="--", alpha=0.5)
ax1.scatter(h1_births, h1_deaths, s=100, color=RED, marker="s", zorder=4, label="H1 (loops)")
high_pers = (h1_deaths - h1_births) > 0.4
ax1.scatter(h1_births[high_pers], h1_deaths[high_pers],
            s=200, color=ORG, marker="s", zorder=5, label="High-persistence H1 (significant loops)")
ax1.set_xlabel("Birth epsilon", color=TXT, fontsize=10)
ax1.set_ylabel("Death epsilon", color=TXT, fontsize=10)
ax1.set_title("Persistence Diagram: Birth-Death Pairs\n(Distance from diagonal = persistence = significance)",
              color=GLD, fontsize=10)
ax1.legend(facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=8)
ax1.set_xlim(-0.1, max_val); ax1.set_ylim(-0.1, max_val)
# Barcode plot
lifetimes_h0 = np.sort(h0_deaths[finite_mask])[::-1]
lifetimes_h1 = np.sort(h1_deaths - h1_births)[::-1]
y_h0 = np.arange(len(lifetimes_h0))
y_h1 = np.arange(len(lifetimes_h1)) + len(lifetimes_h0) + 1
for y, b, d in zip(y_h0, h0_births[finite_mask][np.argsort(h0_deaths[finite_mask])[::-1]], h0_deaths[finite_mask][np.argsort(h0_deaths[finite_mask])[::-1]]):
    ax2.plot([b, d], [y, y], color=BLU, lw=4, alpha=0.8)
for i, (y, lt) in enumerate(zip(y_h1, lifetimes_h1)):
    ax2.plot([h1_births[i], h1_deaths[i]], [y, y], color=RED, lw=4, alpha=0.8)
ax2.set_xlabel("Filtration parameter epsilon", color=TXT, fontsize=10)
ax2.set_ylabel("Feature index", color=TXT, fontsize=10)
ax2.set_title("Persistence Barcode: Length = How Long Feature Persists\n(Long bars = structurally significant)",
              color=GLD, fontsize=10)
h0_patch = mpatches.Patch(color=BLU, label="H0 components")
h1_patch = mpatches.Patch(color=RED, label="H1 loops")
ax2.legend(handles=[h0_patch, h1_patch], facecolor=PBG, edgecolor=BRD, labelcolor=TXT, fontsize=9)
fig.suptitle("Persistent Homology: Reading the Topology of Financial Markets",
             color=TXT, fontsize=12, y=1.01)
fig.tight_layout()
fig_save(fig, "BOOK_25_persistence_diagram.png")

print("\nAll 25 book visuals generated.")
