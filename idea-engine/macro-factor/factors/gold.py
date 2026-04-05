"""
macro-factor/factors/gold.py
─────────────────────────────
Gold Factor — Safe Haven vs Liquidity Regime Detector.

Financial rationale
───────────────────
Gold serves as a dual indicator in the macro framework:

  SAFE HAVEN role (when gold rises and crypto falls):
    → Pure risk-off / deflationary fear.  Investors are fleeing to safety.
    → Macro regime: RISK_OFF or CRISIS.

  LIQUIDITY INDICATOR role (when gold AND crypto both rise):
    → Dollar debasement / monetary expansion narrative.
    → Both are scarce assets benefiting from the same macro tailwind.
    → More sustainable rally — not purely speculative.
    → Macro regime: RISK_ON with inflation premium.

  GOLD FALLING + CRYPTO FALLING:
    → Broad risk-off with dollar strengthening (DXY rising).
    → Everything-sells-off environment.

  GOLD FALLING + CRYPTO RISING:
    → Pure speculative/liquidity-driven crypto move.
    → Less sustainable — often followed by correction.

Key signals:
  1. Gold 20-day momentum (GLD ETF).
  2. Gold vs 200d MA — structural trend.
  3. Gold/Crypto correlation regime (rolling 30d Pearson correlation).
     Positive correlation = liquidity regime (both benefit).
     Negative correlation = safe-haven regime (gold safe, crypto risky).

Ticker: GLD (SPDR Gold Shares ETF)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_GLD_TICKER        = "GLD"
_BTC_TICKER        = "BTC-USD"
_MOMENTUM_WINDOW   = 20
_MA_LONG_WINDOW    = 200
_CORR_WINDOW       = 30
_LOOKBACK_DAYS     = 300


@dataclass
class GoldResult:
    gld_price: float
    gold_momentum_20d: float         # % return over 20d
    gold_ma200_distance: float       # (price - MA200) / MA200
    gold_crypto_corr_30d: float      # rolling 30d correlation with BTC
    correlation_regime: str          # "LIQUIDITY" | "SAFE_HAVEN" | "MIXED"
    signal: float                    # [-1, +1]: +1 = gold bullish for crypto context
    computed_at: str


def _fetch(ticker: str, days: int) -> pd.Series:
    """Download close prices with simulation fallback."""
    try:
        hist = yf.Ticker(ticker).history(period=f"{days}d", interval="1d", auto_adjust=True)
        if not hist.empty:
            return hist["Close"].dropna()
    except Exception as exc:
        logger.warning("Gold: yfinance failed for %s: %s", ticker, exc)

    rng   = np.random.default_rng(seed=hash(ticker) % 50000)
    n     = days
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
    base  = 185.0 if ticker == "GLD" else 50000.0
    prices = [base]
    vol    = 0.012 if ticker == "GLD" else 0.025
    for _ in range(n - 1):
        prices.append(max(50.0, prices[-1] * (1 + rng.normal(0, vol))))
    return pd.Series(prices, index=dates, name=ticker)


def compute_gold(lookback_days: int = _LOOKBACK_DAYS) -> GoldResult:
    """Compute gold macro signals and gold/crypto correlation regime.

    Returns
    -------
    GoldResult with price momentum, MA signal, correlation regime, and composite signal.
    """
    gld = _fetch(_GLD_TICKER, lookback_days)
    btc = _fetch(_BTC_TICKER, lookback_days)

    if len(gld) < _MA_LONG_WINDOW + 10:
        raise ValueError("Insufficient data for gold analysis")

    current_gld = float(gld.iloc[-1])

    # 20d momentum
    if len(gld) > _MOMENTUM_WINDOW:
        past = float(gld.iloc[-(_MOMENTUM_WINDOW + 1)])
        mom = (current_gld - past) / past if past != 0 else 0.0
    else:
        mom = 0.0

    # MA200 distance
    ma200   = float(gld.rolling(_MA_LONG_WINDOW, min_periods=100).mean().iloc[-1])
    ma_dist = (current_gld - ma200) / ma200 if ma200 != 0 else 0.0

    # Rolling 30d correlation: gold vs BTC log returns
    common = gld.index.intersection(btc.index)
    if len(common) >= _CORR_WINDOW + 5:
        gld_ret = gld.loc[common].pct_change().dropna()
        btc_ret = btc.loc[common].pct_change().dropna()
        # Align
        common2 = gld_ret.index.intersection(btc_ret.index)
        if len(common2) >= _CORR_WINDOW:
            corr = float(
                gld_ret.loc[common2].iloc[-_CORR_WINDOW:].corr(btc_ret.loc[common2].iloc[-_CORR_WINDOW:])
            )
        else:
            corr = 0.0
    else:
        corr = 0.0

    # Regime classification
    if corr > 0.4:
        regime = "LIQUIDITY"    # both rising together → monetary debasement tailwind
    elif corr < -0.2:
        regime = "SAFE_HAVEN"   # gold up, crypto down → risk-off
    else:
        regime = "MIXED"

    # Signal logic:
    # Gold momentum + correlation regime combine to determine crypto context
    mom_signal = float(np.clip(mom / 0.05, -1.0, 1.0))  # 5% gold move = ±1

    if regime == "LIQUIDITY":
        # Gold rising AND correlated with crypto → bullish for crypto
        regime_bonus = 0.30 if mom > 0 else -0.10
    elif regime == "SAFE_HAVEN":
        # Gold rising as safe haven → crypto bearish
        regime_bonus = -0.25 if mom > 0 else 0.10
    else:
        regime_bonus = 0.0

    ma_signal = float(np.clip(ma_dist / 0.08, -1.0, 1.0))

    signal = float(np.clip(
        0.4 * mom_signal + 0.4 * regime_bonus + 0.2 * ma_signal,
        -1.0, 1.0,
    ))

    return GoldResult(
        gld_price=round(current_gld, 2),
        gold_momentum_20d=round(mom, 4),
        gold_ma200_distance=round(ma_dist, 4),
        gold_crypto_corr_30d=round(corr, 4),
        correlation_regime=regime,
        signal=round(signal, 4),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def gold_summary(result: GoldResult) -> str:
    direction = "BULLISH" if result.signal > 0 else "BEARISH" if result.signal < 0 else "NEUTRAL"
    return (
        f"GLD={result.gld_price:.1f} mom20d={result.gold_momentum_20d:+.2%} "
        f"MA200={result.gold_ma200_distance:+.2%} corr30d={result.gold_crypto_corr_30d:+.2f} "
        f"({result.correlation_regime}) — {direction} signal={result.signal:+.2f}"
    )
