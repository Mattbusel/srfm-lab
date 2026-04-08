"""
Liquidation cascade hypothesis templates.
"""

TEMPLATES = [
    {
        "id": "cascade_front_run",
        "name": "Cascade Front-Run",
        "description": "Position ahead of expected liquidation cascade given OI and leverage data",
        "hypothesis": "High cascade risk index (>0.6) with elevated OI and extreme funding predicts forced unwind within {window} periods",
        "parameters": {
            "window": 12,
            "cascade_risk_threshold": 0.6,
            "oi_ratio_threshold": 1.5,
            "min_funding_abs": 0.005,
        },
        "conditions": [
            {"type": "cascade_risk_high", "threshold": "cascade_risk_threshold"},
            {"type": "oi_elevated", "ratio": "oi_ratio_threshold"},
            {"type": "funding_extreme", "abs_threshold": "min_funding_abs"},
        ],
        "edge": "liquidation_cascade",
        "tags": ["crypto", "liquidation", "leverage", "front_run"],
    },
    {
        "id": "short_squeeze_setup",
        "name": "Short Squeeze Setup",
        "description": "Enter long on short squeeze conditions: rising price + high short OI + negative funding",
        "hypothesis": "Short squeeze score >0.5 with negative funding and rising price creates forced covering feedback loop",
        "parameters": {
            "squeeze_threshold": 0.5,
            "min_price_move_5d": 0.05,
        },
        "conditions": [
            {"type": "short_squeeze_score_above", "threshold": "squeeze_threshold"},
            {"type": "price_rising_5d", "min_pct": "min_price_move_5d"},
            {"type": "funding_negative"},
        ],
        "edge": "liquidation_cascade",
        "direction": "long",
        "tags": ["short_squeeze", "crypto", "funding"],
    },
    {
        "id": "long_squeeze_setup",
        "name": "Long Squeeze Setup",
        "description": "Short on long squeeze: falling price + high long OI + positive funding",
        "hypothesis": "Long squeeze score >0.5 + falling price + positive funding triggers margin cascade selling",
        "parameters": {
            "squeeze_threshold": 0.5,
            "max_price_move_5d": -0.05,
        },
        "conditions": [
            {"type": "long_squeeze_score_above", "threshold": "squeeze_threshold"},
            {"type": "price_falling_5d", "max_pct": "max_price_move_5d"},
            {"type": "funding_positive"},
        ],
        "edge": "liquidation_cascade",
        "direction": "short",
        "tags": ["long_squeeze", "crypto", "funding"],
    },
    {
        "id": "post_cascade_recovery",
        "name": "Post-Cascade Recovery",
        "description": "Enter after OI depletion signals cascade complete and market stabilizes",
        "hypothesis": "OI depletion score >0.3 followed by declining volatility signals cascade exhaustion — recovery trade available",
        "parameters": {
            "depletion_threshold": 0.3,
            "vol_decline_window": 8,
            "min_oi_drop_pct": 0.10,
        },
        "conditions": [
            {"type": "oi_depletion_above", "threshold": "depletion_threshold"},
            {"type": "volatility_declining", "window": "vol_decline_window"},
            {"type": "no_new_cascade_signals"},
        ],
        "exit_conditions": [
            {"type": "vol_spike_above", "multiple": 2.0},
            {"type": "oi_rising_again"},
        ],
        "edge": "liquidation_cascade",
        "tags": ["recovery", "post_cascade", "vol"],
    },
    {
        "id": "reflexivity_amplifier",
        "name": "Reflexivity Loop Amplifier",
        "description": "Trend-follow when reflexivity score is high — price → liq → price feedback detected",
        "hypothesis": "Reflexivity score >0.3 signals active feedback loop — price moves self-reinforcing, trend continuation likely",
        "parameters": {
            "reflexivity_threshold": 0.3,
            "min_trend_strength": 0.03,
        },
        "conditions": [
            {"type": "reflexivity_above", "threshold": "reflexivity_threshold"},
            {"type": "strong_trend", "threshold": "min_trend_strength"},
        ],
        "edge": "liquidation_cascade",
        "tags": ["reflexivity", "trend_follow", "momentum"],
    },
]
