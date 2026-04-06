"""
entropy_analyzer.py — Market information entropy measures.

Implements five entropy measures on rolling price/return windows:

1.  Shannon entropy    — binned return distribution
2.  Sample entropy     (SampEn) — template-matching complexity
3.  Permutation entropy (PE)    — ordinal pattern distribution, O(n)
4.  Approximate entropy (ApEn)  — Pincus 1991
5.  Transfer entropy   — directed information flow BTC → alt (or any pair)

Regime signal
    High entropy → market is chaotic/random → reduce position sizes
    Low entropy  → market has structure    → allow larger positions
    Threshold configurable; default: reduce if PE > 0.85, increase if PE < 0.55

Rolling window: 200 bars (configurable).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WINDOW = 200
_HIGH_ENTROPY_THRESHOLD = 0.85   # permutation entropy (normalised)
_LOW_ENTROPY_THRESHOLD  = 0.55


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EntropyResult:
    symbol: str
    bar_index: int
    shannon: float         # bits; normalised by log2(bins)
    sample_entropy: float  # SampEn; higher = more complex
    perm_entropy: float    # normalised 0–1
    approx_entropy: float  # ApEn
    transfer_entropy: Optional[float]  # X→Y; None if source not available
    regime: str            # "chaotic", "random_walk", "structured"
    size_scalar: float     # suggested position-size multiplier [0.5–1.5]


def _entropy_regime(pe: float) -> Tuple[str, float]:
    """Map normalised permutation entropy to a regime label and size scalar."""
    if pe > _HIGH_ENTROPY_THRESHOLD:
        return "chaotic", 0.5
    if pe < _LOW_ENTROPY_THRESHOLD:
        return "structured", 1.4
    # linear interpolation in between
    frac = (pe - _LOW_ENTROPY_THRESHOLD) / (_HIGH_ENTROPY_THRESHOLD - _LOW_ENTROPY_THRESHOLD)
    scalar = 1.4 - frac * (1.4 - 0.5)
    return "random_walk", round(float(scalar), 3)


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------


def shannon_entropy(returns: np.ndarray, bins: int = 20) -> float:
    """
    Binned Shannon entropy of return distribution, normalised to [0, 1]
    (divides by log2(bins)).
    """
    if len(returns) < bins:
        return float("nan")
    counts, _ = np.histogram(returns, bins=bins)
    counts = counts[counts > 0]
    total = counts.sum()
    probs = counts / total
    raw = -float(np.sum(probs * np.log2(probs)))
    return raw / math.log2(bins)


# ---------------------------------------------------------------------------
# Permutation entropy
# ---------------------------------------------------------------------------


def permutation_entropy(
    series: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalise: bool = True,
) -> float:
    """
    Permutation entropy (Bandt & Pompe 2002).

    Parameters
    ----------
    series    : 1-D array of prices or returns
    order     : embedding dimension (3–7 typical)
    delay     : time delay (1 = consecutive)
    normalise : divide by log(order!) to get [0, 1] value
    """
    n = len(series)
    if n < (order - 1) * delay + 1:
        return float("nan")

    counter: Counter = Counter()
    for i in range(n - (order - 1) * delay):
        pattern = tuple(np.argsort(series[i: i + order * delay: delay]))
        counter[pattern] += 1

    total = sum(counter.values())
    probs = np.array([v / total for v in counter.values()])
    pe = -float(np.sum(probs * np.log(probs)))
    if normalise:
        max_pe = math.log(math.factorial(order))
        pe = pe / max_pe if max_pe > 0 else 0.0
    return pe


# ---------------------------------------------------------------------------
# Sample entropy
# ---------------------------------------------------------------------------


def sample_entropy(
    series: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
) -> float:
    """
    Sample entropy (Richman & Moorman 2000).

    Parameters
    ----------
    series   : 1-D array
    m        : template length
    r_factor : tolerance = r_factor * std(series)
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < m + 2:
        return float("nan")

    r = r_factor * float(x.std())
    if r < 1e-12:
        return float("nan")

    def _count_templates(template_len: int) -> int:
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if np.all(np.abs(x[i: i + template_len] - x[j: j + template_len]) <= r):
                    count += 1
        return count

    # Use a vectorised approach to avoid O(n^2) Python loops on large arrays
    # Limit to first 300 points for speed
    x_sub = x[:min(n, 300)]
    ns = len(x_sub)

    def _fast_count(ml: int) -> int:
        cnt = 0
        for i in range(ns - ml):
            diffs = np.abs(x_sub[i + 1: ns - ml + 1] - x_sub[i])
            for k in range(1, ml):
                diffs = np.minimum(diffs, np.abs(
                    x_sub[i + k + 1: ns - ml + k + 1] - x_sub[i + k]
                ))
            cnt += int(np.sum(diffs <= r))
        return cnt

    A = _fast_count(m + 1)
    B = _fast_count(m)
    if B == 0:
        return float("nan")
    return -math.log(A / B) if A > 0 else float("inf")


# ---------------------------------------------------------------------------
# Approximate entropy
# ---------------------------------------------------------------------------


def approximate_entropy(
    series: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2,
) -> float:
    """
    Approximate entropy (ApEn, Pincus 1991).
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < m + 2:
        return float("nan")

    r = r_factor * float(x.std())
    if r < 1e-12:
        return float("nan")

    # Truncate for speed
    x = x[:min(n, 300)]
    n = len(x)

    def _phi(template_len: int) -> float:
        cnt = np.zeros(n - template_len + 1)
        for i in range(n - template_len + 1):
            for j in range(n - template_len + 1):
                if np.max(np.abs(x[i: i + template_len] - x[j: j + template_len])) <= r:
                    cnt[i] += 1
        cnt = cnt / (n - template_len + 1)
        cnt = cnt[cnt > 0]
        return float(np.sum(np.log(cnt))) / (n - template_len + 1)

    return abs(_phi(m) - _phi(m + 1))


# ---------------------------------------------------------------------------
# Transfer entropy
# ---------------------------------------------------------------------------


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    k: int = 1,
    bins: int = 10,
) -> float:
    """
    Estimate transfer entropy T(source → target) via histogram binning.

    TE(X→Y) = H(Y_t | Y_{t-k}) − H(Y_t | Y_{t-k}, X_{t-k})

    Parameters
    ----------
    source : return series of the source (e.g., BTC)
    target : return series of the target asset
    k      : lag in bars
    bins   : histogram bins for each marginal
    """
    n = min(len(source), len(target))
    if n < bins * 3 + k:
        return float("nan")

    src = np.asarray(source[-n:], dtype=float)
    tgt = np.asarray(target[-n:], dtype=float)

    # Discretise into bins
    src_b = np.digitize(src, np.histogram_bin_edges(src, bins=bins)[1:-1])
    tgt_b = np.digitize(tgt, np.histogram_bin_edges(tgt, bins=bins)[1:-1])

    # Trim to aligned windows
    yt   = tgt_b[k:]
    yt_k = tgt_b[:-k]
    xt_k = src_b[:-k]
    n2 = len(yt)

    def _joint_entropy(*arrs: np.ndarray) -> float:
        stacked = np.stack(arrs, axis=1)
        keys = [tuple(row) for row in stacked]
        counter: Counter = Counter(keys)
        total = sum(counter.values())
        probs = np.array([v / total for v in counter.values()])
        return -float(np.sum(probs * np.log2(probs + 1e-15)))

    def _entropy_1d(arr: np.ndarray) -> float:
        c: Counter = Counter(arr.tolist())
        total = sum(c.values())
        p = np.array([v / total for v in c.values()])
        return -float(np.sum(p * np.log2(p + 1e-15)))

    # H(Y_t | Y_{t-k}) = H(Y_t, Y_{t-k}) - H(Y_{t-k})
    H_yt_ytk = _joint_entropy(yt, yt_k) - _entropy_1d(yt_k)
    # H(Y_t | Y_{t-k}, X_{t-k}) = H(Y_t, Y_{t-k}, X_{t-k}) - H(Y_{t-k}, X_{t-k})
    H_yt_ytk_xtk = _joint_entropy(yt, yt_k, xt_k) - _joint_entropy(yt_k, xt_k)

    te = H_yt_ytk - H_yt_ytk_xtk
    return max(0.0, float(te))


# ---------------------------------------------------------------------------
# Per-symbol state
# ---------------------------------------------------------------------------


@dataclass
class _SymbolState:
    symbol: str
    window: int
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=400))
    returns: Deque[float] = field(default_factory=lambda: deque(maxlen=400))
    bar_count: int = 0


# ---------------------------------------------------------------------------
# EntropyAnalyzer
# ---------------------------------------------------------------------------


class EntropyAnalyzer:
    """
    Compute rolling entropy measures for one or multiple symbols.

    Parameters
    ----------
    window : int
        Rolling window length (bars).
    perm_order : int
        Embedding order for permutation entropy (3–6).
    shannon_bins : int
        Number of histogram bins for Shannon entropy.
    source_symbol : str
        Symbol whose returns drive transfer-entropy computation
        (e.g., "BTC").  Set to None to disable TE.
    recompute_every : int
        Recompute every N bars (entropy is expensive).
    """

    def __init__(
        self,
        window: int = _DEFAULT_WINDOW,
        perm_order: int = 4,
        shannon_bins: int = 20,
        source_symbol: Optional[str] = "BTC",
        recompute_every: int = 10,
    ) -> None:
        self.window = window
        self.perm_order = perm_order
        self.shannon_bins = shannon_bins
        self.source_symbol = source_symbol
        self.recompute_every = recompute_every

        self._states: Dict[str, _SymbolState] = {}
        self._latest: Dict[str, EntropyResult] = {}

    # ------------------------------------------------------------------
    # Stream interface
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        price: float,
    ) -> Optional[EntropyResult]:
        """
        Ingest a new price bar.

        Returns an EntropyResult every `recompute_every` bars once the
        buffer is full, else None.
        """
        st = self._get_state(symbol)

        if len(st.prices) > 0 and st.prices[-1] > 0:
            ret = math.log(price / st.prices[-1])
        else:
            ret = 0.0

        st.prices.append(price)
        st.returns.append(ret)
        st.bar_count += 1

        if st.bar_count % self.recompute_every != 0:
            return None
        if len(st.returns) < self.window:
            return None

        result = self._compute(symbol, st)
        self._latest[symbol] = result
        return result

    def get_latest(self, symbol: str) -> Optional[EntropyResult]:
        return self._latest.get(symbol)

    # ------------------------------------------------------------------
    # Compute all measures
    # ------------------------------------------------------------------

    def _compute(self, symbol: str, st: _SymbolState) -> EntropyResult:
        rets = np.array(list(st.returns)[-self.window:], dtype=float)
        prices_arr = np.array(list(st.prices)[-self.window:], dtype=float)

        se = shannon_entropy(rets, bins=self.shannon_bins)
        pe = permutation_entropy(prices_arr, order=self.perm_order)
        samp = sample_entropy(rets)
        apen = approximate_entropy(rets)

        # Transfer entropy: source → this symbol
        te: Optional[float] = None
        if self.source_symbol and self.source_symbol != symbol:
            src_state = self._states.get(self.source_symbol)
            if src_state and len(src_state.returns) >= self.window:
                src_rets = np.array(list(src_state.returns)[-self.window:], dtype=float)
                te = transfer_entropy(src_rets, rets)

        # Use PE as primary regime driver; fall back to Shannon if PE is nan
        primary = pe if math.isfinite(pe) else (se if math.isfinite(se) else 0.7)
        regime_str, size_scalar = _entropy_regime(primary)

        return EntropyResult(
            symbol=symbol,
            bar_index=st.bar_count,
            shannon=se,
            sample_entropy=samp,
            perm_entropy=pe,
            approx_entropy=apen,
            transfer_entropy=te,
            regime=regime_str,
            size_scalar=size_scalar,
        )

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def feed_series(
        self,
        symbol: str,
        prices: Sequence[float],
    ) -> List[EntropyResult]:
        results = []
        for p in prices:
            r = self.update(symbol, float(p))
            if r is not None:
                results.append(r)
        return results

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(symbol=symbol, window=self.window)
        return self._states[symbol]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["EntropyAnalyzer — latest results:"]
        for sym, r in sorted(self._latest.items()):
            te_str = f"{r.transfer_entropy:.4f}" if r.transfer_entropy is not None else "n/a"
            lines.append(
                f"  {sym:10s}  PE={r.perm_entropy:.3f}  SE={r.shannon:.3f}  "
                f"SampEn={r.sample_entropy:.3f}  TE={te_str}  "
                f"regime={r.regime}  size_x={r.size_scalar:.2f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import csv
    from pathlib import Path

    csv_path = Path(__file__).parent.parent.parent / "data" / "NDX_hourly_poly.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(open(csv_path)))
        prices = [float(r.get("close", r.get("Close", 0))) for r in rows[:600]]
    else:
        rng = np.random.default_rng(1)
        prices = (15000.0 + np.cumsum(rng.normal(0, 30, 600))).tolist()

    analyzer = EntropyAnalyzer(window=200, recompute_every=10, source_symbol=None)
    results = analyzer.feed_series("NDX", prices)
    print(f"Entropy computations: {len(results)}")
    if results:
        r = results[-1]
        print(
            f"Last: PE={r.perm_entropy:.3f}  SE={r.shannon:.3f}  "
            f"SampEn={r.sample_entropy:.3f}  ApEn={r.approx_entropy:.3f}  "
            f"regime={r.regime}  size_x={r.size_scalar}"
        )
    print(analyzer.summary())


if __name__ == "__main__":
    _demo()
