"""
Market making hypothesis templates.
"""

TEMPLATES = [
    {
        "id": "as_optimal_spread",
        "name": "Avellaneda-Stoikov Optimal Quoting",
        "description": "Quote at AS reservation price with optimal spread; scale by inventory risk",
        "hypothesis": "AS model optimal quotes at ±{half_spread}σ reservation price earn spread income while controlling inventory risk via gamma={gamma}",
        "parameters": {
            "gamma": 0.1,
            "k": 1.5,
            "A": 140.0,
            "max_inventory": 5.0,
            "quote_refresh_seconds": 5,
        },
        "conditions": [
            {"type": "low_adverse_selection", "max_vpin": 0.6},
            {"type": "volatility_regime", "max_vol_pct": 80},
            {"type": "sufficient_order_flow"},
        ],
        "edge": "market_making",
        "tags": ["market_making", "avellaneda_stoikov", "inventory"],
    },
    {
        "id": "inventory_skew",
        "name": "Inventory Skew Management",
        "description": "Skew quotes toward inventory reduction when position exceeds threshold",
        "hypothesis": "Widening bid (if long) and narrowing ask reduces inventory risk by {inventory_decay_pct}% per period at low additional cost",
        "parameters": {
            "inventory_threshold": 3.0,
            "skew_bps": 5,
            "max_skew_bps": 20,
            "inventory_decay_pct": 15,
        },
        "conditions": [
            {"type": "inventory_above_threshold", "threshold": "inventory_threshold"},
            {"type": "spread_wide_enough", "min_bps": "skew_bps"},
        ],
        "edge": "market_making",
        "tags": ["market_making", "inventory", "skew"],
    },
    {
        "id": "low_toxicity_mm",
        "name": "Low Toxicity Market Making",
        "description": "Increase quote size when VPIN is low, indicating uninformed flow",
        "hypothesis": "VPIN < {vpin_threshold} indicates predominantly uninformed order flow — safe to widen size, earn more spread",
        "parameters": {
            "vpin_threshold": 0.35,
            "size_multiplier": 2.0,
        },
        "conditions": [
            {"type": "vpin_below", "threshold": "vpin_threshold"},
            {"type": "kyle_lambda_low"},
            {"type": "vol_regime_normal"},
        ],
        "edge": "market_making",
        "tags": ["market_making", "vpin", "toxicity"],
    },
    {
        "id": "glosten_milgrom_spread_widening",
        "name": "Informed Flow Spread Widening",
        "description": "Widen spread when adverse selection probability is elevated",
        "hypothesis": "Glosten-Milgrom adverse selection component >30% of spread → widen quotes to break even against informed traders",
        "parameters": {
            "max_adverse_selection_frac": 0.30,
            "spread_widening_multiplier": 2.5,
        },
        "conditions": [
            {"type": "adverse_selection_high", "threshold": "max_adverse_selection_frac"},
        ],
        "edge": "market_making",
        "tags": ["market_making", "adverse_selection", "glosten_milgrom"],
    },
    {
        "id": "almgren_chriss_execution",
        "name": "Almgren-Chriss Optimal Liquidation",
        "description": "Execute large orders on AC optimal schedule to minimize implementation shortfall",
        "hypothesis": "AC schedule with risk aversion {risk_aversion} minimizes E[cost] + λ*Var[cost] over {T_periods} periods",
        "parameters": {
            "risk_aversion": 1e-6,
            "T_periods": 20,
            "max_participation_rate": 0.15,
        },
        "conditions": [
            {"type": "large_order", "min_pct_adv": 0.05},
            {"type": "sufficient_time_horizon"},
        ],
        "edge": "market_making",
        "tags": ["execution", "almgren_chriss", "implementation_shortfall"],
    },
]
