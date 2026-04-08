"""
Mean reversion hypothesis templates.
"""

TEMPLATES = [
    {
        "id": "ou_zscore_entry",
        "name": "OU Z-Score Entry",
        "description": "Enter when spread z-score exceeds threshold, sized by half-life quality",
        "hypothesis": "Ornstein-Uhlenbeck spread deviations above {entry_z}σ provide positive expectation when half-life < {max_half_life} periods",
        "parameters": {
            "entry_z": 2.0,
            "exit_z": 0.5,
            "max_half_life": 30,
            "min_half_life": 3,
        },
        "conditions": [
            {"type": "zscore_above", "value": "entry_z"},
            {"type": "half_life_range", "min": "min_half_life", "max": "max_half_life"},
            {"type": "cointegration_confirmed", "pvalue": 0.05},
        ],
        "exit_conditions": [
            {"type": "zscore_below", "value": "exit_z"},
            {"type": "half_life_exceeded", "multiplier": 2.0},
        ],
        "edge": "mean_reversion",
        "tags": ["pairs", "stat_arb", "cointegration"],
    },
    {
        "id": "kalman_spread_fade",
        "name": "Kalman Spread Fade",
        "description": "Fade extreme Kalman-filtered spread deviations with time-varying hedge ratio",
        "hypothesis": "Time-varying hedge ratio (Kalman) spread >2σ reverts within {expected_hold} periods in stable cointegration regimes",
        "parameters": {
            "entry_z": 2.0,
            "exit_z": 0.3,
            "expected_hold": 10,
            "kalman_delta": 1e-4,
        },
        "conditions": [
            {"type": "kalman_zscore_extreme", "threshold": 2.0},
            {"type": "beta_stable", "max_beta_change": 0.3},
        ],
        "edge": "mean_reversion",
        "tags": ["pairs", "kalman", "dynamic_hedge"],
    },
    {
        "id": "tar_outer_regime",
        "name": "TAR Outer Regime Entry",
        "description": "Trade mean reversion only in outer TAR regime where kappa is strongest",
        "hypothesis": "Threshold AR model outer regime shows 3x stronger mean reversion than inner regime — larger positions warranted outside ±{threshold}σ",
        "parameters": {
            "threshold": 1.5,
            "min_kappa_ratio": 2.0,
        },
        "conditions": [
            {"type": "tar_outer_regime", "threshold": "threshold"},
            {"type": "tar_kappa_ratio_above", "value": "min_kappa_ratio"},
        ],
        "edge": "mean_reversion",
        "tags": ["nonlinear", "threshold", "regime"],
    },
    {
        "id": "bollinger_reversal",
        "name": "Bollinger Band Reversal",
        "description": "Classic Bollinger band mean reversion with volume confirmation",
        "hypothesis": "Price touching BB outer band with declining volume signals exhaustion and reversion toward mean",
        "parameters": {
            "bb_window": 20,
            "n_std": 2.0,
            "volume_decay_threshold": 0.7,
        },
        "conditions": [
            {"type": "outside_bollinger", "n_std": 2.0},
            {"type": "volume_declining", "threshold": "volume_decay_threshold"},
        ],
        "edge": "mean_reversion",
        "tags": ["bollinger", "technical", "volume"],
    },
    {
        "id": "basket_residual_trade",
        "name": "Multi-Asset Basket Residual",
        "description": "Trade basket residual when multiple cointegration vectors align",
        "hypothesis": "Johansen cointegration with r>1 provides multiple hedge ratios — residual from combined basket mean-reverts faster",
        "parameters": {
            "min_coint_vectors": 2,
            "entry_z": 1.8,
        },
        "conditions": [
            {"type": "johansen_rank_above", "value": "min_coint_vectors"},
            {"type": "basket_zscore_extreme", "threshold": "entry_z"},
        ],
        "edge": "mean_reversion",
        "tags": ["basket", "johansen", "multi_asset"],
    },
]
