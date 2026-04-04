"""
02_regime_classification_study.py — Regime Classification Comparison Study

Implements and compares 4 regime detectors:
  1. BH-based (existing SRFM physics)
  2. HMM 3-state (Gaussian emissions via hmmlearn or manual Baum-Welch)
  3. Volatility threshold (high/mid/low vol bands)
  4. Moving average based (price vs MA200)

Analysis:
  - Compute regime probabilities over 2021-2026 history
  - Agreement matrix between detectors
  - Backtest returns conditional on each detector's regime
  - Find which detector best predicts next-day returns
  - Outputs: research/outputs/regime_comparison.png, regime_stats.json

Run: python research/notebooks/02_regime_classification_study.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT / "lib"))

from srfm_core import MinkowskiClassifier, BlackHoleDetector

OUTPUTS = _ROOT / "research" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate_daily(sym: str = "ES", n: int = 1260) -> pd.DataFrame:
    """Load daily OHLCV. n=1260 ≈ 5 years of trading days."""
    candidates = [
        _ROOT / "data" / f"{sym}_daily.csv",
        _ROOT / "data" / f"{sym}_daily_real.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                df.columns = [c.lower() for c in df.columns]
                if "close" in df.columns:
                    if "volume" not in df.columns:
                        df["volume"] = 1_000_000.0
                    return df.sort_index().dropna(subset=["close"])
            except Exception:
                pass

    # Generate synthetic daily ES
    print(f"  Generating synthetic daily {sym}: {n} bars")
    rng = np.random.default_rng(42)
    closes = np.empty(n)
    closes[0] = 3800.0
    # Mixed regime: bull + occasional crash
    for i in range(n):
        regime_prob = rng.random()
        if regime_prob < 0.60:    mu, sig = 0.0003, 0.008   # bull
        elif regime_prob < 0.80:  mu, sig = -0.0002, 0.012  # bear
        elif regime_prob < 0.95:  mu, sig = 0.0001, 0.005   # sideways
        else:                     mu, sig = -0.003, 0.025    # crisis
        ret = mu + sig * rng.standard_normal()
        closes[i] = closes[max(0, i-1)] * max(0.01, 1.0 + np.clip(ret, -0.10, 0.10))

    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    noise = 0.005 * np.abs(rng.standard_normal(n))
    return pd.DataFrame({
        "open":   closes * (1 - noise / 2),
        "high":   closes * (1 + noise),
        "low":    closes * (1 - noise),
        "close":  closes,
        "volume": np.full(n, 1_000_000.0),
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Detector 1: BH-based regime
# ─────────────────────────────────────────────────────────────────────────────

def detect_bh_regime(df: pd.DataFrame, cf: float = 0.005) -> pd.Series:
    """
    BH-based regime: uses bh_dir and bh_mass.
    Returns Series of regime strings: BULL, BEAR, SIDEWAYS, HIGH_VOL.
    """
    mc  = MinkowskiClassifier(cf=cf)
    bh  = BlackHoleDetector(bh_form=1.5, bh_collapse=1.0, bh_decay=0.95)
    closes = df["close"].values
    highs  = df["high"].values if "high" in df.columns else closes * 1.01
    lows   = df["low"].values  if "low" in df.columns  else closes * 0.99

    regimes = []
    mc.update(float(closes[0]))
    regimes.append("SIDEWAYS")

    for i in range(1, len(closes)):
        bit = mc.update(float(closes[i]))
        bh.update(bit, float(closes[i]), float(closes[i-1]))
        atr = (highs[i] - lows[i]) if len(highs) > i else abs(closes[i] - closes[i-1])
        atr_ratio = atr / (closes[i] * 0.01 + 1e-9)

        if atr_ratio > 3.0:
            regimes.append("HIGH_VOL")
        elif bh.bh_active and bh.bh_dir > 0:
            regimes.append("BULL")
        elif bh.bh_active and bh.bh_dir < 0:
            regimes.append("BEAR")
        else:
            regimes.append("SIDEWAYS")

    return pd.Series(regimes, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Detector 2: 3-state HMM (manual EM / Viterbi)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianHMM3State:
    """
    3-state Gaussian HMM fitted via EM (Baum-Welch).
    States: 0=BULL, 1=BEAR, 2=SIDEWAYS
    Observations: daily log returns.
    """

    def __init__(self, n_iter: int = 50, tol: float = 1e-4):
        self.n_iter = n_iter
        self.tol    = tol
        self.n_states = 3
        # Parameters (initialized in fit)
        self.pi  = np.array([1/3, 1/3, 1/3])
        self.A   = np.full((3, 3), 1/3)
        self.mu  = np.array([0.001, -0.001, 0.0])
        self.sig = np.array([0.008, 0.015, 0.004])

    def _emission(self, x: float) -> np.ndarray:
        """Gaussian emission probabilities for observation x."""
        probs = np.zeros(self.n_states)
        for k in range(self.n_states):
            probs[k] = (1.0 / (self.sig[k] * math.sqrt(2 * math.pi))
                        * math.exp(-0.5 * ((x - self.mu[k]) / self.sig[k])**2))
        return np.maximum(probs, 1e-300)

    def _forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(obs)
        alpha = np.zeros((T, self.n_states))
        scale = np.zeros(T)
        alpha[0] = self.pi * self._emission(obs[0])
        scale[0] = alpha[0].sum()
        alpha[0] /= max(scale[0], 1e-300)
        for t in range(1, T):
            em = self._emission(obs[t])
            alpha[t] = (alpha[t-1] @ self.A) * em
            scale[t] = alpha[t].sum()
            alpha[t] /= max(scale[t], 1e-300)
        return alpha, scale

    def _backward(self, obs: np.ndarray, scale: np.ndarray) -> np.ndarray:
        T = len(obs)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0
        for t in range(T-2, -1, -1):
            em = self._emission(obs[t+1])
            beta[t] = (self.A @ (em * beta[t+1]))
            beta[t] /= max(scale[t+1], 1e-300)
        return beta

    def fit(self, obs: np.ndarray) -> "GaussianHMM3State":
        """Fit HMM using Baum-Welch EM."""
        T = len(obs)
        prev_ll = -np.inf

        # Initialize mu from quantiles
        self.mu  = np.array([np.percentile(obs, 70),
                              np.percentile(obs, 10),
                              np.percentile(obs, 50)])
        self.sig = np.array([max(obs.std(), 1e-6)] * 3)

        for iteration in range(self.n_iter):
            # E-step
            alpha, scale = self._forward(obs)
            beta          = self._backward(obs, scale)
            ll = float(np.sum(np.log(scale + 1e-300)))

            # Posterior state probabilities
            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)

            # xi: joint state transitions
            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                em_next = self._emission(obs[t+1])
                num = alpha[t:t+1].T * self.A * em_next * beta[t+1]
                denom = num.sum() + 1e-300
                xi[t] = num / denom

            # M-step
            self.pi = gamma[0] + 1e-10
            self.pi /= self.pi.sum()

            new_A = xi.sum(axis=0)
            row_sums = new_A.sum(axis=1, keepdims=True)
            self.A = new_A / (row_sums + 1e-300)

            for k in range(self.n_states):
                gk = gamma[:, k]
                gk_sum = gk.sum() + 1e-10
                self.mu[k]  = (gk * obs).sum() / gk_sum
                self.sig[k] = math.sqrt((gk * (obs - self.mu[k])**2).sum() / gk_sum)
                self.sig[k] = max(self.sig[k], 1e-6)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Viterbi decoding: returns most likely state sequence."""
        T = len(obs)
        delta = np.zeros((T, self.n_states))
        psi   = np.zeros((T, self.n_states), dtype=int)
        delta[0] = np.log(self.pi + 1e-300) + np.log(self._emission(obs[0]) + 1e-300)
        for t in range(1, T):
            log_em = np.log(self._emission(obs[t]) + 1e-300)
            for j in range(self.n_states):
                vals = delta[t-1] + np.log(self.A[:, j] + 1e-300)
                psi[t, j]   = int(np.argmax(vals))
                delta[t, j] = vals[psi[t, j]] + log_em[j]
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

    def state_name(self, state: int) -> str:
        # Assign names by mu sign
        if self.mu[state] > 0.0002:
            return "BULL"
        elif self.mu[state] < -0.0002:
            return "BEAR"
        else:
            return "SIDEWAYS"


def detect_hmm_regime(df: pd.DataFrame, n_iter: int = 30) -> pd.Series:
    """Fit 3-state Gaussian HMM to daily log returns, return regime labels."""
    rets = np.diff(np.log(df["close"].values + 1e-9))
    rets = np.clip(rets, -0.15, 0.15)

    hmm = GaussianHMM3State(n_iter=n_iter)
    hmm.fit(rets)
    states = hmm.predict(rets)

    # Map state indices to names
    names = [hmm.state_name(s) for s in states]
    # First bar: pad with first regime
    names = [names[0]] + names
    return pd.Series(names, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Detector 3: Volatility threshold
# ─────────────────────────────────────────────────────────────────────────────

def detect_vol_regime(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Classify each day by realized volatility (21-day rolling std of log returns).
    LOW vol = bottom 25th pct → BULL (or sideways in calm)
    HIGH vol = top 25th pct  → HIGH_VOL
    MID vol  = middle         → SIDEWAYS
    Direction from price trend.
    """
    log_rets = np.log(df["close"] / df["close"].shift(1)).dropna()
    rvol = log_rets.rolling(window).std() * math.sqrt(252)
    rvol = rvol.reindex(df.index)

    p25 = rvol.quantile(0.25)
    p75 = rvol.quantile(0.75)

    # Simple trend: 50d SMA
    sma50  = df["close"].rolling(50).mean()
    ma_dir = (df["close"] > sma50).astype(int)

    regimes = []
    for i, idx in enumerate(df.index):
        v = rvol.loc[idx]
        if pd.isna(v):
            regimes.append("SIDEWAYS")
            continue
        if v > p75:
            regimes.append("HIGH_VOL")
        elif v < p25:
            # Low vol — use direction
            regimes.append("BULL" if ma_dir.loc[idx] else "BEAR")
        else:
            regimes.append("SIDEWAYS")

    return pd.Series(regimes, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Detector 4: Moving average based
# ─────────────────────────────────────────────────────────────────────────────

def detect_ma_regime(df: pd.DataFrame) -> pd.Series:
    """
    MA200-based regime:
      - Price > MA200 + price trending up (MA50 > MA200) → BULL
      - Price < MA200 → BEAR
      - Price near MA200 or MA50 crossing → SIDEWAYS
    """
    close  = df["close"]
    ma50   = close.rolling(50).mean()
    ma200  = close.rolling(200).mean()
    ma_cross = (ma50 > ma200).astype(int)
    above200  = (close > ma200).astype(int)

    regimes = []
    for i, idx in enumerate(df.index):
        c  = close.loc[idx]
        m2 = ma200.loc[idx]
        m5 = ma50.loc[idx]
        if pd.isna(m2) or pd.isna(m5):
            regimes.append("SIDEWAYS")
            continue
        dist200_pct = (c - m2) / (m2 + 1e-9)
        if c > m2 and m5 > m2 and dist200_pct > 0.02:
            regimes.append("BULL")
        elif c < m2 and m5 < m2 and dist200_pct < -0.02:
            regimes.append("BEAR")
        else:
            regimes.append("SIDEWAYS")

    return pd.Series(regimes, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison analysis
# ─────────────────────────────────────────────────────────────────────────────

REGIME_VALUES = {"BULL": 2, "SIDEWAYS": 1, "BEAR": 0, "HIGH_VOL": 3}


def agreement_matrix(detector_regimes: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Compute pairwise agreement (% matching regime) between detectors.
    """
    detectors = list(detector_regimes.keys())
    n = len(detectors)
    mat = np.eye(n)
    common_idx = detector_regimes[detectors[0]].index
    for k in detectors[1:]:
        common_idx = common_idx.intersection(detector_regimes[k].index)

    for i in range(n):
        for j in range(i+1, n):
            r1 = detector_regimes[detectors[i]].reindex(common_idx)
            r2 = detector_regimes[detectors[j]].reindex(common_idx)
            agree = float((r1 == r2).mean())
            mat[i, j] = agree
            mat[j, i] = agree

    return pd.DataFrame(mat, index=detectors, columns=detectors)


def conditional_returns(
    df: pd.DataFrame,
    regime_series: pd.Series,
    forward_days: int = 1,
) -> Dict[str, dict]:
    """
    Compute next-day returns conditional on each regime.
    Returns: {regime: {mean, std, n, sharpe, pct_positive}}
    """
    fwd_rets = df["close"].pct_change(forward_days).shift(-forward_days)
    result = {}
    for regime in sorted(regime_series.unique()):
        mask = regime_series == regime
        rets  = fwd_rets[mask].dropna().values
        if len(rets) == 0:
            continue
        result[regime] = {
            "mean":         float(np.mean(rets)),
            "std":          float(np.std(rets)),
            "n":            len(rets),
            "sharpe":       float(np.mean(rets) / (np.std(rets) + 1e-10) * math.sqrt(252)),
            "pct_positive": float(np.mean(rets > 0)),
        }
    return result


def information_coefficient(
    regime_series: pd.Series,
    fwd_returns: pd.Series,
) -> float:
    """
    IC: rank correlation between regime signal and forward returns.
    Maps regime to numeric value for correlation.
    """
    reg_num = regime_series.map(REGIME_VALUES).fillna(1).reindex(fwd_returns.index)
    common = reg_num.dropna().index.intersection(fwd_returns.dropna().index)
    if len(common) < 10:
        return 0.0
    r1 = reg_num.reindex(common).values
    r2 = fwd_returns.reindex(common).values
    # Spearman rank correlation
    from scipy.stats import spearmanr
    corr, _ = spearmanr(r1, r2)
    return float(corr) if math.isfinite(float(corr)) else 0.0


def backtest_regime_conditional(
    df: pd.DataFrame,
    regime_series: pd.Series,
    target_regime: str = "BULL",
    long_only: bool = True,
) -> dict:
    """
    Simple backtest: go long when target_regime matches, flat otherwise.
    Returns: {total_return, cagr, sharpe, max_drawdown, n_trades}
    """
    fwd_ret = df["close"].pct_change(1).shift(-1)
    in_regime = (regime_series == target_regime)
    strat_ret = fwd_ret * in_regime.astype(float)

    equity = np.cumprod(1.0 + strat_ret.fillna(0).values)
    if len(equity) == 0:
        return {}

    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / (peak + 1e-9)
    years  = max(0.01, len(df) / 252)
    total_ret = float(equity[-1] - 1.0)
    cagr   = float(equity[-1] ** (1 / years) - 1.0)
    sr     = strat_ret.fillna(0)
    sharpe = float(sr.mean() / (sr.std() + 1e-10) * math.sqrt(252))
    return {
        "total_return":  round(total_ret, 4),
        "cagr":          round(cagr, 4),
        "sharpe":        round(sharpe, 3),
        "max_drawdown":  round(float(dd.min()), 4),
        "n_signals":     int(in_regime.sum()),
        "signal_pct":    round(float(in_regime.mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_regime_comparison(
    df: pd.DataFrame,
    detector_regimes: Dict[str, pd.Series],
    agreement_mat: pd.DataFrame,
    ax_regimes,
    ax_agreement,
):
    """Plot regime timeseries side-by-side and agreement heatmap."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    COLORS = {"BULL": "lime", "BEAR": "red", "SIDEWAYS": "gray", "HIGH_VOL": "orange"}

    n_det = len(detector_regimes)
    times = np.arange(len(df))
    price = df["close"].values / df["close"].iloc[0]  # normalized

    for i, (name, reg) in enumerate(detector_regimes.items()):
        ax = ax_regimes[i]
        ax.plot(times, price, color="black", linewidth=0.6, alpha=0.7)
        reg_idx = reg.reindex(df.index).fillna("SIDEWAYS")
        for t, (ts, regime_label) in enumerate(zip(df.index, reg_idx)):
            color = COLORS.get(str(regime_label), "gray")
            ax.axvspan(t, t+1, alpha=0.25, color=color)
        ax.set_title(f"{name} Regime", fontsize=8)
        ax.set_ylabel("Price (norm.)")
        ax.grid(alpha=0.2)

        # Legend
        patches = [mpatches.Patch(color=c, label=r) for r, c in COLORS.items()]
        ax.legend(handles=patches, fontsize=5, loc="upper left")

    # Agreement heatmap
    mat = agreement_mat.values
    names = list(agreement_mat.columns)
    im = ax_agreement.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
    ax_agreement.set_xticks(range(len(names)))
    ax_agreement.set_yticks(range(len(names)))
    ax_agreement.set_xticklabels(names, rotation=30, fontsize=7)
    ax_agreement.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax_agreement.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax_agreement.set_title("Detector Agreement Matrix", fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("02_regime_classification_study.py")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAS_PLOT = True
    except ImportError:
        print("[WARN] matplotlib not available")
        HAS_PLOT = False

    print("\nLoading data...")
    df = load_or_generate_daily("ES", n=1260)
    print(f"  {len(df)} days, {df.index[0].date()} → {df.index[-1].date()}")

    print("\nFitting regime detectors...")
    detectors = {}

    print("  1. BH-based...")
    detectors["BH"] = detect_bh_regime(df)

    print("  2. HMM 3-state (EM)...")
    detectors["HMM"] = detect_hmm_regime(df, n_iter=30)

    print("  3. Volatility threshold...")
    detectors["VolThresh"] = detect_vol_regime(df)

    print("  4. MA200...")
    detectors["MA200"] = detect_ma_regime(df)

    # Agreement matrix
    print("\nComputing agreement matrix...")
    agree_mat = agreement_matrix(detectors)
    print(agree_mat.round(3).to_string())

    # Forward returns
    fwd_rets_1d = df["close"].pct_change(1).shift(-1)

    # IC per detector
    print("\nInformation Coefficients (Spearman, 1-day forward):")
    ics = {}
    for name, reg in detectors.items():
        ic = information_coefficient(reg, fwd_rets_1d)
        ics[name] = round(ic, 5)
        print(f"  {name:12s}: IC = {ic:.5f}")

    # Conditional returns
    print("\nConditional returns by regime:")
    cond_rets = {}
    for name, reg in detectors.items():
        cond = conditional_returns(df, reg, forward_days=1)
        cond_rets[name] = cond
        print(f"\n  [{name}]")
        for regime, stats in cond.items():
            print(f"    {regime:12s}: mean={stats['mean']:.4f}, "
                  f"Sharpe={stats['sharpe']:.2f}, n={stats['n']}")

    # Regime-conditional backtests
    print("\nBacktest — Long in BULL regime only:")
    backtest_results = {}
    for name, reg in detectors.items():
        bt = backtest_regime_conditional(df, reg, target_regime="BULL")
        backtest_results[name] = bt
        if bt:
            print(f"  {name:12s}: return={bt.get('total_return',0):.2%}, "
                  f"Sharpe={bt.get('sharpe',0):.2f}, "
                  f"signals={bt.get('n_signals',0)} ({bt.get('signal_pct',0):.1%})")

    # ── Plotting ──────────────────────────────────────────────────────────────
    if HAS_PLOT:
        fig, axes = plt.subplots(5, 1, figsize=(16, 16))
        # 4 regime plots
        det_items = list(detectors.items())
        for i, (name, reg) in enumerate(det_items):
            ax = axes[i]
            times = np.arange(len(df))
            price = df["close"].values / df["close"].iloc[0]
            ax.plot(times, price, "k-", linewidth=0.7, alpha=0.8)
            COLORS = {"BULL": "lime", "BEAR": "red", "SIDEWAYS": "gray", "HIGH_VOL": "orange"}
            reg2 = reg.reindex(df.index).fillna("SIDEWAYS").values
            for t in range(len(times)):
                c = COLORS.get(str(reg2[t]), "gray")
                ax.axvspan(t, t+1, alpha=0.2, color=c)
            ax.set_title(f"{name} Regime Classification", fontsize=10)
            ax.set_ylabel("Norm. Price"); ax.grid(alpha=0.2)

        # Agreement heatmap in last axis
        ax_last = axes[4]
        mat = agree_mat.values
        names_list = list(agree_mat.columns)
        im = ax_last.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax_last.set_xticks(range(len(names_list))); ax_last.set_xticklabels(names_list)
        ax_last.set_yticks(range(len(names_list))); ax_last.set_yticklabels(names_list)
        for i in range(len(names_list)):
            for j in range(len(names_list)):
                ax_last.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=9)
        ax_last.set_title("Detector Agreement Matrix (% Same Regime)")
        plt.colorbar(im, ax=ax_last)

        plt.tight_layout()
        out = OUTPUTS / "regime_comparison.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\nPlot → {out}")
        plt.close()

    # ── Save stats ────────────────────────────────────────────────────────────
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (float, np.floating)):
            v = float(obj)
            return v if math.isfinite(v) else None
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        return obj

    stats_out = {
        "n_days": len(df),
        "agreement_matrix": {
            k: {k2: float(v2) for k2, v2 in v.items()}
            for k, v in agree_mat.to_dict().items()
        },
        "information_coefficients": ics,
        "conditional_returns": _clean(cond_rets),
        "backtest_results": _clean(backtest_results),
    }
    out_json = OUTPUTS / "regime_stats.json"
    with open(out_json, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"Stats → {out_json}")

    print("\n" + "=" * 60)
    print("SUMMARY: Best predictor of next-day returns")
    best = max(ics, key=lambda k: abs(ics[k]))
    print(f"  Best IC: {best} ({ics[best]:.5f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
