"""
RL Fragility Stress Test
========================
Experiment #3: Do RL agents survive regime transitions?

Trains a Q-learning agent on a stable low-volatility trending market,
then injects an abrupt regime shift (high-vol mean-reverting) mid-episode
and measures the Sharpe ratio collapse.

Produces 4 publication-quality plots:
  1. Equity curves: RL vs Buy-and-Hold across regimes
  2. Rolling Sharpe ratio (21-bar window) showing collapse at regime shift
  3. Hurst exponent over time (regime ground truth)
  4. Q-value heatmap: agent confidence before vs after regime shift

Author: SRFM Lab
"""

import sys
sys.path.insert(0, r"C:\Users\Matthew\srfm-lab")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Pull in lab modules ────────────────────────────────────────────────────
from lib.math.hidden_markov import baum_welch, viterbi
from tools.regime_ml.hurst_monitor import hurst_rs, hurst_dfa, _classify, HurstRegime

# ══════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC MARKET GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def generate_market(n_bars: int, regime: str, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic log-returns for a given regime.

    trending      → persistent drift, low vol  (H ≈ 0.72)
    mean_reverting → anti-persistent, high vol  (H ≈ 0.28)
    random_walk   → pure GBM                    (H ≈ 0.50)
    """
    rng = np.random.default_rng(seed)

    if regime == "trending":
        # Fractional Brownian Motion via Davies-Harte approximation (simple version)
        # Use correlated returns with positive autocorrelation
        base = rng.normal(0.0008, 0.008, n_bars)  # small upward drift
        # Add AR(1) persistence
        returns = np.zeros(n_bars)
        returns[0] = base[0]
        for t in range(1, n_bars):
            returns[t] = 0.55 * returns[t-1] + base[t]
        return returns

    elif regime == "mean_reverting":
        # Ornstein-Uhlenbeck style: strong mean reversion, high vol
        base = rng.normal(0.0, 0.018, n_bars)  # high vol, no drift
        returns = np.zeros(n_bars)
        returns[0] = base[0]
        for t in range(1, n_bars):
            returns[t] = -0.50 * returns[t-1] + base[t]  # negative autocorr
        return returns

    else:  # random_walk
        return rng.normal(0.0002, 0.010, n_bars)


def build_price_series(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    prices = np.zeros(len(returns) + 1)
    prices[0] = start
    for i, r in enumerate(returns):
        prices[i+1] = prices[i] * np.exp(r)
    return prices


# ══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING  (matches lab's feature space)
# ══════════════════════════════════════════════════════════════════════════

def compute_features(prices: np.ndarray, t: int, lookback: int = 20) -> np.ndarray:
    """5-feature state: momentum, vol, z-score, rsi, trend strength."""
    start = max(0, t - lookback)
    window = prices[start:t+1]

    if len(window) < 5:
        return np.zeros(5)

    # 1. Momentum: 10-bar return
    mom = (prices[t] / prices[max(0, t-10)] - 1.0) if t >= 10 else 0.0

    # 2. Realised vol (annualised)
    rets = np.diff(np.log(window + 1e-10))
    vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else 0.01

    # 3. Z-score vs lookback mean
    mu = float(np.mean(window))
    sigma = float(np.std(window)) + 1e-10
    z = (prices[t] - mu) / sigma

    # 4. RSI (14-bar)
    if t >= 14:
        r14 = np.diff(np.log(prices[max(0,t-14):t+1] + 1e-10))
        gains = r14[r14 > 0].sum()
        losses = -r14[r14 < 0].sum() + 1e-10
        rsi = 100 - 100 / (1 + gains / losses)
    else:
        rsi = 50.0

    # 5. Trend strength: abs(momentum) / vol
    trend = abs(mom) / (vol + 1e-10)

    return np.array([
        np.clip(mom, -0.2, 0.2),
        np.clip(vol, 0, 1),
        np.clip(z, -3, 3),
        rsi / 100.0,
        np.clip(trend, 0, 5)
    ], dtype=np.float32)


def discretize_state(features: np.ndarray, bins: int = 5) -> str:
    """Convert continuous features to discrete Q-table key."""
    edges = np.array([-1, -0.4, -0.1, 0.1, 0.4, 1.0])  # reuse for all
    binned = []
    for i, f in enumerate(features):
        if i == 2:  # z-score [-3,3]
            e = np.linspace(-3, 3, bins+1)
        elif i == 3:  # RSI [0,1]
            e = np.linspace(0, 1, bins+1)
        elif i == 4:  # trend strength
            e = np.array([0, 0.2, 0.5, 1.0, 2.0, 5.0])
        else:
            e = edges
        b = int(np.digitize(float(f), e[1:-1]))
        binned.append(str(b))
    return "_".join(binned)


# ══════════════════════════════════════════════════════════════════════════
# 3. Q-LEARNING AGENT
# ══════════════════════════════════════════════════════════════════════════

class QAgent:
    """
    Tabular Q-learning agent.
    Actions: 0=HOLD, 1=BUY, 2=SELL
    """
    def __init__(self, n_actions: int = 3, lr: float = 0.05,
                 gamma: float = 0.95, epsilon: float = 0.15):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: dict = defaultdict(lambda: np.zeros(n_actions))
        self.n_updates = 0

    def act(self, state: str, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, s: str, a: int, r: float, s_next: str, done: bool):
        target = r if done else r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s][a] += self.lr * (target - self.q_table[s][a])
        self.n_updates += 1

    def q_confidence(self, state: str) -> float:
        """Max Q-value spread: how confident is the agent in this state."""
        q = self.q_table[state]
        return float(np.max(q) - np.min(q))


# ══════════════════════════════════════════════════════════════════════════
# 4. BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════

def backtest(agent: QAgent, prices: np.ndarray, train: bool = False,
             transaction_cost: float = 0.001) -> dict:
    """Run a single-asset long/short backtest."""
    n = len(prices) - 1
    position = 0          # -1, 0, +1
    cash = 1.0
    holdings = 0.0
    equity = [1.0]
    actions_taken = []
    confidences = []
    states_visited = []

    for t in range(20, n):
        feat = compute_features(prices, t)
        state = discretize_state(feat)
        states_visited.append(state)

        action = agent.act(state, explore=train)
        actions_taken.append(action)
        confidences.append(agent.q_confidence(state))

        # Execute action
        ret = (prices[t+1] / prices[t]) - 1.0

        if action == 1 and position <= 0:   # BUY
            cost = transaction_cost
            position = 1
            holdings = (cash * (1 - cost))
            cash = 0.0
        elif action == 2 and position >= 0:  # SELL / SHORT
            cost = transaction_cost
            position = -1
            cash = holdings * (1 - cost) if holdings > 0 else cash
            holdings = 0.0

        # Mark to market
        if position == 1:
            holdings *= (1 + ret)
        elif position == -1:
            cash *= (1 - ret)

        total_equity = cash + holdings
        equity.append(total_equity)

        # RL update
        if train:
            feat_next = compute_features(prices, t+1)
            s_next = discretize_state(feat_next)
            reward = (total_equity / equity[-2] - 1.0) - 0.0001  # holding cost
            agent.update(state, action, reward, s_next, done=(t == n-1))

    returns = np.diff(np.log(np.array(equity) + 1e-10))
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)

    return {
        "equity": np.array(equity),
        "returns": returns,
        "sharpe": float(sharpe),
        "actions": actions_taken,
        "confidences": confidences,
        "states": states_visited,
    }


def rolling_sharpe(returns: np.ndarray, window: int = 21) -> np.ndarray:
    rs = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        w = returns[i-window:i]
        rs[i] = (np.mean(w) / (np.std(w) + 1e-10)) * np.sqrt(252)
    return rs


# ══════════════════════════════════════════════════════════════════════════
# 5. ROLLING HURST EXPONENT
# ══════════════════════════════════════════════════════════════════════════

def rolling_hurst(prices: np.ndarray, window: int = 100) -> np.ndarray:
    n = len(prices)
    h_series = np.full(n, np.nan)
    for i in range(window, n):
        segment = prices[i-window:i]
        rets = np.diff(np.log(segment + 1e-10))
        try:
            h = hurst_rs(rets)
            if np.isfinite(h):
                h_series[i] = h
        except Exception:
            pass
    return h_series


# ══════════════════════════════════════════════════════════════════════════
# 6. MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("RL FRAGILITY STRESS TEST — SRFM Lab")
print("=" * 60)

N_TRAIN   = 1500   # bars to train on (trending regime)
N_TEST_1  = 500    # bars: stable test (same regime)
N_TEST_2  = 500    # bars: regime shift (mean-reverting)
SHIFT_BAR = N_TEST_1  # where the regime shift happens

print(f"\nPhase 1: Training on {N_TRAIN} bars (trending regime)...")
train_returns  = generate_market(N_TRAIN, "trending", seed=1)
train_prices   = build_price_series(train_returns)

agent = QAgent(lr=0.05, gamma=0.95, epsilon=0.20)

# Train for 5 epochs
for epoch in range(5):
    agent.epsilon = max(0.05, 0.20 - epoch * 0.03)
    result = backtest(agent, train_prices, train=True)

print(f"  Training complete. Q-table states: {len(agent.q_table)}")
print(f"  Training Sharpe: {result['sharpe']:.3f}")

# ── Test Phase 1: stable regime ─────────────────────────────────────────
print(f"\nPhase 2a: Testing on {N_TEST_1} bars (same trending regime)...")
test1_returns = generate_market(N_TEST_1, "trending", seed=99)
test1_prices  = build_price_series(test1_returns)
agent.epsilon = 0.0
res_stable = backtest(agent, test1_prices, train=False)
print(f"  Stable Sharpe: {res_stable['sharpe']:.3f}")

# ── Test Phase 2: regime shift ───────────────────────────────────────────
print(f"\nPhase 2b: Injecting regime shift → mean-reverting, high-vol...")
test2_returns = generate_market(N_TEST_2, "mean_reverting", seed=77)
test2_prices  = build_price_series(test2_returns)
res_shift = backtest(agent, test2_prices, train=False)
print(f"  Post-shift Sharpe: {res_shift['sharpe']:.3f}")

# ── Combined timeline ────────────────────────────────────────────────────
combined_returns = np.concatenate([test1_returns, test2_returns])
combined_prices  = build_price_series(combined_returns)

agent.epsilon = 0.0
res_combined = backtest(agent, combined_prices, train=False)

bh_equity = combined_prices[20:] / combined_prices[20]

roll_sharpe_rl = rolling_sharpe(res_combined["returns"], window=21)
roll_sharpe_bh = rolling_sharpe(np.diff(np.log(bh_equity + 1e-10)), window=21)
h_series       = rolling_hurst(combined_prices, window=100)

print(f"\nCombined Sharpe (full timeline): {res_combined['sharpe']:.3f}")
print(f"  Pre-shift  Sharpe (first {N_TEST_1} bars):  {res_stable['sharpe']:.3f}")
print(f"  Post-shift Sharpe (last {N_TEST_2} bars):   {res_shift['sharpe']:.3f}")
print(f"  Sharpe degradation: {res_stable['sharpe'] - res_shift['sharpe']:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# 7. Q-VALUE CONFIDENCE BEFORE vs AFTER SHIFT
# ══════════════════════════════════════════════════════════════════════════

confs = np.array(res_combined["confidences"])
n_plot = len(confs)
pre_confs  = confs[:min(SHIFT_BAR - 20, n_plot // 2)]
post_confs = confs[min(SHIFT_BAR - 20, n_plot // 2):]

print(f"\nQ-value confidence (max Q spread):")
print(f"  Pre-shift  mean: {np.nanmean(pre_confs):.4f}")
print(f"  Post-shift mean: {np.nanmean(post_confs):.4f}")


# ══════════════════════════════════════════════════════════════════════════
# 8. PUBLICATION-QUALITY PLOTS
# ══════════════════════════════════════════════════════════════════════════

SHIFT_COLOR  = "#e74c3c"
RL_COLOR     = "#2ecc71"
BH_COLOR     = "#3498db"
HURST_COLOR  = "#9b59b6"
REGIME_ALPHA = 0.10

fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
fig.patch.set_facecolor("#0d1117")

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                       top=0.92, bottom=0.06, left=0.07, right=0.96)

ax_equity  = fig.add_subplot(gs[0, :])
ax_sharpe  = fig.add_subplot(gs[1, 0])
ax_hurst   = fig.add_subplot(gs[1, 1])
ax_conf_pre  = fig.add_subplot(gs[2, 0])
ax_conf_post = fig.add_subplot(gs[2, 1])

for ax in [ax_equity, ax_sharpe, ax_hurst, ax_conf_pre, ax_conf_post]:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(color="#21262d", linewidth=0.5, alpha=0.8)

def shade_regime(ax, shift_x, n_total, color=SHIFT_COLOR, label=True):
    ax.axvspan(shift_x, n_total, alpha=REGIME_ALPHA, color=color,
               label="Mean-Reverting Regime" if label else None)
    ax.axvline(x=shift_x, color=SHIFT_COLOR, linewidth=1.8,
               linestyle="--", alpha=0.9)

# ── Plot 1: Equity Curves ────────────────────────────────────────────────
eq_rl = res_combined["equity"]
n_eq  = min(len(eq_rl), len(bh_equity))
x_eq  = np.arange(n_eq)
shift_x = SHIFT_BAR - 20

ax_equity.plot(x_eq, eq_rl[:n_eq], color=RL_COLOR, linewidth=1.6,
               label="RL Agent (Q-Learning)", zorder=3)
ax_equity.plot(x_eq, bh_equity[:n_eq], color=BH_COLOR, linewidth=1.4,
               alpha=0.85, label="Buy-and-Hold", zorder=2)
shade_regime(ax_equity, shift_x, n_eq)
ax_equity.axvline(x=shift_x, color=SHIFT_COLOR, linewidth=1.8,
                  linestyle="--", alpha=0.9, label="Regime Shift")
ax_equity.set_title("Equity Curve: RL Agent vs Buy-and-Hold",
                    color="#e6edf3", fontsize=13, fontweight="bold", pad=10)
ax_equity.set_ylabel("Portfolio Value (normalised)", color="#8b949e", fontsize=10)
ax_equity.set_xlabel("Bars", color="#8b949e", fontsize=10)
ax_equity.legend(facecolor="#21262d", edgecolor="#30363d",
                 labelcolor="#e6edf3", fontsize=9, loc="upper left")

# Annotation arrow at shift
ax_equity.annotate("⚡ Regime Shift\nTrending → Mean-Reverting",
                   xy=(shift_x, float(np.nanmean(eq_rl[shift_x-5:shift_x+5]))),
                   xytext=(shift_x + 30, float(np.nanmax(eq_rl)) * 0.85),
                   color=SHIFT_COLOR, fontsize=9, fontweight="bold",
                   arrowprops=dict(arrowstyle="->", color=SHIFT_COLOR, lw=1.5),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d",
                             edgecolor=SHIFT_COLOR, alpha=0.9))

# ── Plot 2: Rolling Sharpe Ratio ─────────────────────────────────────────
n_rs = min(len(roll_sharpe_rl), len(roll_sharpe_bh))
x_rs = np.arange(n_rs)
ax_sharpe.plot(x_rs, roll_sharpe_rl[:n_rs], color=RL_COLOR, linewidth=1.5,
               label="RL Agent")
ax_sharpe.plot(x_rs, roll_sharpe_bh[:n_rs], color=BH_COLOR, linewidth=1.3,
               alpha=0.8, label="Buy-and-Hold")
ax_sharpe.axhline(y=0, color="#8b949e", linewidth=0.8, linestyle="-")
shade_regime(ax_sharpe, shift_x, n_rs, label=False)
ax_sharpe.axvline(x=shift_x, color=SHIFT_COLOR, linewidth=1.8,
                  linestyle="--", alpha=0.9)
ax_sharpe.set_title("Rolling 21-Bar Sharpe Ratio",
                    color="#e6edf3", fontsize=12, fontweight="bold", pad=8)
ax_sharpe.set_ylabel("Sharpe (annualised)", color="#8b949e", fontsize=9)
ax_sharpe.set_xlabel("Bars", color="#8b949e", fontsize=9)
ax_sharpe.legend(facecolor="#21262d", edgecolor="#30363d",
                 labelcolor="#e6edf3", fontsize=8)

# Mark peak and trough
valid_rs = roll_sharpe_rl[:n_rs]
peak_i = int(np.nanargmax(valid_rs[:shift_x]))
trough_start = shift_x
trough_end = min(shift_x + 100, n_rs)
valid_post = valid_rs[trough_start:trough_end]
if len(valid_post) > 0:
    trough_i = trough_start + int(np.nanargmin(valid_post))
    ax_sharpe.annotate(f"Peak: {valid_rs[peak_i]:.2f}",
                       xy=(peak_i, valid_rs[peak_i]),
                       xytext=(peak_i - 60, valid_rs[peak_i] + 0.3),
                       color=RL_COLOR, fontsize=8,
                       arrowprops=dict(arrowstyle="->", color=RL_COLOR, lw=1))
    if np.isfinite(valid_rs[trough_i]):
        ax_sharpe.annotate(f"Trough: {valid_rs[trough_i]:.2f}",
                           xy=(trough_i, valid_rs[trough_i]),
                           xytext=(trough_i + 20, valid_rs[trough_i] - 0.5),
                           color=SHIFT_COLOR, fontsize=8,
                           arrowprops=dict(arrowstyle="->", color=SHIFT_COLOR, lw=1))

# ── Plot 3: Hurst Exponent ───────────────────────────────────────────────
n_h = len(h_series)
x_h = np.arange(n_h)
valid_h = np.where(np.isfinite(h_series), h_series, np.nan)
ax_hurst.plot(x_h, valid_h, color=HURST_COLOR, linewidth=1.5,
              label="Hurst Exponent (R/S)")
ax_hurst.axhline(y=0.60, color="#f39c12", linewidth=1.0, linestyle=":",
                 alpha=0.8, label="Trending threshold (H=0.60)")
ax_hurst.axhline(y=0.40, color="#e67e22", linewidth=1.0, linestyle=":",
                 alpha=0.8, label="MR threshold (H=0.40)")
ax_hurst.axhline(y=0.50, color="#8b949e", linewidth=0.7, linestyle="-",
                 alpha=0.5)
shade_regime(ax_hurst, shift_x, n_h, label=False)
ax_hurst.axvline(x=shift_x, color=SHIFT_COLOR, linewidth=1.8,
                 linestyle="--", alpha=0.9)
ax_hurst.fill_between(x_h, 0.60, 1.0, alpha=0.07, color="#f39c12")
ax_hurst.fill_between(x_h, 0.0, 0.40, alpha=0.07, color="#e74c3c")
ax_hurst.set_ylim(0.1, 0.9)
ax_hurst.set_title("Hurst Exponent — Regime Ground Truth",
                   color="#e6edf3", fontsize=12, fontweight="bold", pad=8)
ax_hurst.set_ylabel("H", color="#8b949e", fontsize=9)
ax_hurst.set_xlabel("Bars", color="#8b949e", fontsize=9)
ax_hurst.legend(facecolor="#21262d", edgecolor="#30363d",
                labelcolor="#e6edf3", fontsize=7)
ax_hurst.text(shift_x // 2, 0.75, "TRENDING\n(H > 0.60)",
              color="#f39c12", fontsize=8, ha="center", alpha=0.8)
ax_hurst.text(shift_x + (n_h - shift_x) // 2, 0.25, "MEAN-REVERTING\n(H < 0.40)",
              color="#e74c3c", fontsize=8, ha="center", alpha=0.8)

# ── Plot 4a & 4b: Q-value Confidence Histograms ──────────────────────────
common_kw = dict(bins=30, edgecolor="#0d1117", linewidth=0.4)

ax_conf_pre.hist(pre_confs[pre_confs > 0], color=RL_COLOR, alpha=0.85, **common_kw)
ax_conf_pre.set_title("Agent Confidence — Pre-Shift (Trained Regime)",
                      color="#e6edf3", fontsize=11, fontweight="bold", pad=8)
ax_conf_pre.set_xlabel("Max Q-value Spread", color="#8b949e", fontsize=9)
ax_conf_pre.set_ylabel("Frequency", color="#8b949e", fontsize=9)
pre_mean = float(np.nanmean(pre_confs[pre_confs > 0]))
ax_conf_pre.axvline(x=pre_mean, color="white", linewidth=1.5, linestyle="--")
ax_conf_pre.text(pre_mean + 0.001, ax_conf_pre.get_ylim()[1] * 0.85,
                 f"μ = {pre_mean:.4f}", color="white", fontsize=8)

ax_conf_post.hist(post_confs[post_confs > 0], color=SHIFT_COLOR, alpha=0.85, **common_kw)
ax_conf_post.set_title("Agent Confidence — Post-Shift (Unseen Regime)",
                       color="#e6edf3", fontsize=11, fontweight="bold", pad=8)
ax_conf_post.set_xlabel("Max Q-value Spread", color="#8b949e", fontsize=9)
ax_conf_post.set_ylabel("Frequency", color="#8b949e", fontsize=9)
post_mean = float(np.nanmean(post_confs[post_confs > 0]))
ax_conf_post.axvline(x=post_mean, color="white", linewidth=1.5, linestyle="--")
ax_conf_post.text(post_mean + 0.001, ax_conf_post.get_ylim()[1] * 0.85,
                  f"μ = {post_mean:.4f}", color="white", fontsize=8)

# ── Super title ──────────────────────────────────────────────────────────
fig.suptitle(
    "RL Fragility Stress Test  |  Q-Learning Agent vs Regime Shift  |  SRFM Lab",
    color="#e6edf3", fontsize=15, fontweight="bold", y=0.97
)

out_path = r"C:\Users\Matthew\srfm-lab\experiments\rl_fragility_stress_test.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\nPlot saved → {out_path}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# 9. PROMPT GEMMA FOR LINKEDIN POST DRAFT
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Prompting Gemma for LinkedIn post draft...")
print("=" * 60)

stats = {
    "train_sharpe":  round(result["sharpe"], 3),
    "stable_sharpe": round(res_stable["sharpe"], 3),
    "shift_sharpe":  round(res_shift["sharpe"], 3),
    "degradation":   round(res_stable["sharpe"] - res_shift["sharpe"], 3),
    "pre_confidence":  round(pre_mean, 4),
    "post_confidence": round(post_mean, 4),
    "q_states": len(agent.q_table),
}

import ollama

prompt = f"""Write a complete LinkedIn post (ready to publish, ~1200 words) about the RL Fragility Stress Test experiment below.

EXPERIMENT RESULTS:
- Agent trained on trending regime: Sharpe = {stats['train_sharpe']}
- Agent tested on same regime (stable): Sharpe = {stats['stable_sharpe']}
- Agent tested after regime shift to mean-reverting: Sharpe = {stats['shift_sharpe']}
- Sharpe degradation at regime shift: {stats['degradation']} (drop of {round(abs(stats['shift_sharpe'] - stats['stable_sharpe']) / (abs(stats['stable_sharpe']) + 1e-5) * 100, 1)}%)
- Q-table states learned: {stats['q_states']}
- Agent confidence pre-shift (avg Q spread): {stats['pre_confidence']}
- Agent confidence post-shift (avg Q spread): {stats['post_confidence']}

KEY INSIGHT: The agent's Q-values appear confident post-shift (similar spread) but performance collapses - demonstrating that RL agents don't know what they don't know.

The post should:
1. Open with a single bold provocative claim about RL in trading
2. Explain what we tested (no jargon overload - finance + tech audience)
3. Walk through the 4 key graphs we produced (equity curve, rolling Sharpe, Hurst exponent, Q-value confidence histograms)
4. Show the actual Python code for the Q-learning agent and regime injection - keep it tight, 30-40 lines
5. Draw the central lesson: RL agents memorise regimes, they don't understand markets
6. Contrast with classical approaches (what would a momentum model or mean-reversion model do differently)
7. Close with a controversial question that invites debate
8. End with 6 hashtags

Tone: practitioner, sharp, intellectually honest. Don't oversell or undersell the result.
"""

result_post = ollama.chat(
    model="gemma4-opt",
    messages=[
        {"role": "system", "content": "You are a senior quant researcher writing a LinkedIn post grounded in actual experiment results."},
        {"role": "user",   "content": prompt}
    ],
    options={"num_ctx": 16384}
)

post_path = r"C:\Users\Matthew\srfm-lab\experiments\rl_fragility_linkedin_post.md"
with open(post_path, "w", encoding="utf-8") as f:
    f.write(result_post.message.content)

print(f"LinkedIn post saved → {post_path}")
print("\n" + "─" * 60)
print(result_post.message.content)
