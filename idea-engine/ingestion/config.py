"""
idea-engine/ingestion/config.py
────────────────────────────────
All path and parameter configuration for the ingestion layer.

All paths are resolved relative to the srfm-lab root so that the engine
can be run from any working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

# ── Repository root ──────────────────────────────────────────────────────────

# idea-engine/ingestion/config.py  →  up 2 levels  →  srfm-lab/
_THIS_FILE  = Path(__file__).resolve()
SRFM_ROOT   = _THIS_FILE.parent.parent.parent          # srfm-lab/
ENGINE_ROOT = _THIS_FILE.parent.parent                 # srfm-lab/idea-engine/

# ── Database ─────────────────────────────────────────────────────────────────

DB_PATH: Path = ENGINE_ROOT / "idea_engine.db"

# ── Backtest output ──────────────────────────────────────────────────────────

BACKTEST_OUTPUT_DIR: Path   = SRFM_ROOT / "tools" / "backtest_output"
CRYPTO_TRADES_CSV:   Path   = BACKTEST_OUTPUT_DIR / "crypto_trades.csv"
CRYPTO_BH_MC_PNG:    Path   = BACKTEST_OUTPUT_DIR / "crypto_bh_mc.png"
LIVE_TRADES_DB:      Path   = BACKTEST_OUTPUT_DIR / "live_trades.db"

# ── Walk-forward ─────────────────────────────────────────────────────────────

WALK_FORWARD_DIR: Path = SRFM_ROOT / "research" / "walk_forward"

# ── Miner configuration ──────────────────────────────────────────────────────

# Minimum number of trades a group must have to be considered by any miner
MIN_GROUP_SAMPLE: int = 10

# Significance threshold for raw p-values (before multiple-testing correction)
RAW_P_VALUE_THRESHOLD: float = 0.10

# Minimum absolute effect size (Cohen's d or Cliff's delta) to report a pattern
MIN_EFFECT_SIZE: float = 0.20

# Bootstrap filter significance level (after BH correction)
BOOTSTRAP_ALPHA: float = 0.05

# Stationary bootstrap block length (None = automatic via Politis-Romano)
BOOTSTRAP_BLOCK_LENGTH: Optional[int] = None

# Number of bootstrap resamples
BOOTSTRAP_N_RESAMPLES: int = 2_000

# ── Time-of-day miner ────────────────────────────────────────────────────────

TOD_MIN_TRADES_PER_BIN:   int   = 8
TOD_KRUSKAL_ALPHA:         float = 0.05
TOD_DUNN_ALPHA:            float = 0.10   # post-hoc, adjusted

# ── Regime cluster miner ─────────────────────────────────────────────────────

REGIME_HDBSCAN_MIN_CLUSTER_SIZE: int   = 15
REGIME_HDBSCAN_MIN_SAMPLES:      int   = 5
REGIME_KMEANS_K_RANGE:           tuple = (2, 8)   # (min_k, max_k) for elbow
REGIME_FEATURE_COLS: list = [
    "d_bh_mass",
    "h_bh_mass",
    "m15_bh_mass",
    "tf_score",
    "atr",
    "garch_vol",
    "ou_zscore",
]

# ── BH physics miner ─────────────────────────────────────────────────────────

BH_EARLY_WARNING_LOW:  float = 1.50   # mass threshold for "approaching activation"
BH_EARLY_WARNING_HIGH: float = 1.92   # mass threshold — above this = fully active
BH_TRAJECTORY_LOOKBACK: int  = 5      # bars to look back for mass trajectory features

# ── Drawdown miner ────────────────────────────────────────────────────────────

DD_SIGNIFICANT_THRESHOLD: float = 0.05   # drawdowns worse than -5 % are "significant"
DD_CLUSTER_DISTANCE_DAYS: int   = 10     # drawdown events within N days are clustered


# ── Pipeline defaults ─────────────────────────────────────────────────────────

PIPELINE_DEFAULT_CONFIG: Dict[str, Any] = {
    "loaders": {
        "backtest":     True,
        "live_trades":  True,
        "walk_forward": True,
    },
    "miners": {
        "time_of_day":    True,
        "regime_cluster": True,
        "bh_physics":     True,
        "drawdown":       True,
    },
    "filter": {
        "alpha":          BOOTSTRAP_ALPHA,
        "min_effect":     MIN_EFFECT_SIZE,
        "n_resamples":    BOOTSTRAP_N_RESAMPLES,
    },
}
