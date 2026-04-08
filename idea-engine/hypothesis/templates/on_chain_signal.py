"""
On-chain signal hypothesis templates for crypto markets.
"""

TEMPLATES = [
    {
        "id": "exchange_outflow_accumulation",
        "name": "Exchange Outflow Accumulation",
        "description": "Large net outflows from exchanges signal accumulation — bullish",
        "hypothesis": "Exchange outflow > {threshold_btc} BTC/day sustained for {min_days} days signals institutional accumulation → price appreciation",
        "parameters": {
            "threshold_btc": 5000,
            "min_days": 3,
            "lookback_days": 14,
        },
        "conditions": [
            {"type": "exchange_outflow_above", "value": "threshold_btc"},
            {"type": "sustained", "days": "min_days"},
            {"type": "price_not_parabolic", "max_7d_return": 0.30},
        ],
        "edge": "on_chain",
        "direction": "long",
        "tags": ["on_chain", "bitcoin", "accumulation", "exchange_flow"],
    },
    {
        "id": "whale_accumulation",
        "name": "Whale Address Accumulation",
        "description": "Large wallets (>1000 BTC) increasing holdings — bullish signal",
        "hypothesis": "Whale address count growth >2% per week over {lookback} weeks precedes major moves",
        "parameters": {
            "lookback": 4,
            "min_weekly_growth": 0.02,
            "whale_threshold_btc": 1000,
        },
        "conditions": [
            {"type": "whale_count_growing", "threshold": "min_weekly_growth"},
            {"type": "sustained_weeks", "count": "lookback"},
        ],
        "edge": "on_chain",
        "direction": "long",
        "tags": ["on_chain", "whale", "smart_money"],
    },
    {
        "id": "stablecoin_supply_growth",
        "name": "Stablecoin Supply Growth Signal",
        "description": "Growing stablecoin supply on exchanges = dry powder waiting to deploy",
        "hypothesis": "Stablecoin supply on exchanges growing >5% per week signals incoming crypto buying pressure",
        "parameters": {
            "min_weekly_growth_pct": 5.0,
            "lookback_weeks": 2,
        },
        "conditions": [
            {"type": "stablecoin_exchange_supply_growing", "threshold": "min_weekly_growth_pct"},
            {"type": "not_at_local_top", "lookback": 30},
        ],
        "edge": "on_chain",
        "direction": "long",
        "tags": ["on_chain", "stablecoin", "liquidity"],
    },
    {
        "id": "miner_capitulation",
        "name": "Miner Capitulation Bottom",
        "description": "Miner outflows spike and hash rate drops — capitulation bottom signal",
        "hypothesis": "Miner outflow spike + hash rate declining >10% from peak signals miner capitulation → price bottom formation",
        "parameters": {
            "miner_outflow_spike_multiple": 3.0,
            "hash_rate_decline_pct": 0.10,
        },
        "conditions": [
            {"type": "miner_outflow_spike", "multiple": "miner_outflow_spike_multiple"},
            {"type": "hash_rate_declining", "threshold": "hash_rate_decline_pct"},
        ],
        "edge": "on_chain",
        "direction": "long",
        "signal_type": "contrarian",
        "tags": ["on_chain", "miner", "capitulation", "bottom"],
    },
    {
        "id": "nupl_extreme",
        "name": "NUPL Extreme Signal",
        "description": "Net Unrealized Profit/Loss at extremes signals market cycle tops/bottoms",
        "hypothesis": "NUPL > 0.75 (euphoria) signals cycle top within 60 days. NUPL < -0.25 (capitulation) signals cycle bottom",
        "parameters": {
            "top_threshold": 0.75,
            "bottom_threshold": -0.25,
        },
        "conditions": [
            {"type": "nupl_extreme", "top": "top_threshold", "bottom": "bottom_threshold"},
        ],
        "edge": "on_chain",
        "signal_type": "cycle",
        "tags": ["on_chain", "nupl", "cycle", "profit_loss"],
    },
    {
        "id": "sopr_retest",
        "name": "SOPR Retest Signal",
        "description": "Spent Output Profit Ratio retesting 1.0 from above = support; from below = resistance",
        "hypothesis": "SOPR dip to 1.0 in uptrend followed by bounce confirms bull market support. SOPR bounce to 1.0 in downtrend confirms resistance",
        "parameters": {
            "sopr_band": 0.02,
        },
        "conditions": [
            {"type": "sopr_near_one", "band": "sopr_band"},
            {"type": "trend_confirmed"},
        ],
        "edge": "on_chain",
        "tags": ["on_chain", "sopr", "support_resistance"],
    },
    {
        "id": "dormant_coins_moving",
        "name": "Old Coin Movement Warning",
        "description": "Long-dormant coins moving signals distribution by long-term holders",
        "hypothesis": "Coin Days Destroyed (CDD) spike >5x 90d average indicates old hands distributing — bearish for 7-30 days",
        "parameters": {
            "cdd_spike_multiple": 5.0,
            "lookback_90d": True,
        },
        "conditions": [
            {"type": "cdd_spike", "multiple": "cdd_spike_multiple"},
        ],
        "edge": "on_chain",
        "direction": "short",
        "signal_type": "distribution",
        "tags": ["on_chain", "cdd", "old_coins", "distribution"],
    },
]
