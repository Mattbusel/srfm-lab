"""
On-Chain Signal Integration (T2-9)
Fetches and processes on-chain data as BH mass modifiers for crypto instruments.

Signals:
  1. Exchange net flow (negative = bullish accumulation)
  2. Perpetual funding rate extremes
  3. Simple NUPL proxy via price/200d ratio

Data sources: CryptoQuant free tier, CoinGecko (no API key required for basic data).
Falls back to neutral signal if unavailable.
"""
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional
import urllib.request
import json

log = logging.getLogger(__name__)

@dataclass
class OnChainConfig:
    refresh_interval_seconds: int = 3600  # refresh hourly
    max_funding_rate: float = 0.003  # 0.3% = extreme
    nupl_bull_threshold: float = 0.75  # price / 200d > 1.75
    nupl_bear_threshold: float = 0.85  # price / 200d < 0.85
    mass_modifier_cap: float = 0.35  # max ±0.35 BH mass modifier

class OnChainSignalProvider:
    """
    Provides BH mass modifier signals from on-chain data.

    Returns mass_modifier in [-0.35, +0.35]:
      positive = bullish on-chain conditions → boost BH mass
      negative = bearish on-chain conditions → suppress BH mass
    """

    def __init__(self, cfg: OnChainConfig = None):
        self.cfg = cfg or OnChainConfig()
        self._cache: dict[str, dict] = {}  # sym → {modifier, last_price, last_200d_price, ts}
        self._lock = threading.Lock()
        self._last_refresh = 0.0
        self._funding_rates: dict[str, float] = {}  # sym → latest funding rate

    def get_mass_modifier(self, sym: str, current_price: float) -> float:
        """
        Returns a BH mass modifier in [-0.35, +0.35].
        Uses cached data; returns 0.0 if no data available.
        """
        with self._lock:
            cached = self._cache.get(sym, {})

        modifier = 0.0

        # 1. Funding rate signal (works for BTC/ETH/etc with perp markets)
        fr = self._funding_rates.get(sym, 0.0)
        if abs(fr) > self.cfg.max_funding_rate:
            # Extreme positive funding = overleveraged longs = bearish contrarian
            # Extreme negative funding = overleveraged shorts = bullish contrarian
            fr_signal = -float(fr) / (self.cfg.max_funding_rate * 3)
            modifier += max(-0.15, min(0.15, fr_signal))

        # 2. Simple price/200d SMA proxy for NUPL
        price_history = cached.get("price_history", [])
        if len(price_history) >= 200:
            sma_200 = sum(price_history[-200:]) / 200
            ratio = current_price / (sma_200 + 1e-10)
            if ratio > self.cfg.nupl_bull_threshold:
                modifier += 0.10  # strong uptrend
            elif ratio < self.cfg.nupl_bear_threshold:
                modifier -= 0.10  # below 200d SMA = bearish

        return max(-self.cfg.mass_modifier_cap, min(self.cfg.mass_modifier_cap, modifier))

    def update_price(self, sym: str, price: float):
        """Call this on every bar to maintain the price history (used for SMA proxy)."""
        with self._lock:
            if sym not in self._cache:
                self._cache[sym] = {"price_history": []}
            hist = self._cache[sym]["price_history"]
            hist.append(price)
            if len(hist) > 300:
                hist.pop(0)

    def update_funding_rate(self, sym: str, funding_rate: float):
        """Update funding rate for a symbol (called from market data feed if available)."""
        with self._lock:
            self._funding_rates[sym] = funding_rate

    def try_fetch_funding_rates(self) -> None:
        """
        Attempt to fetch BTC/ETH funding rates from a public endpoint.
        Runs without error if network is unavailable.
        """
        now = time.time()
        if now - self._last_refresh < self.cfg.refresh_interval_seconds:
            return
        self._last_refresh = now

        # Try Binance public funding rate endpoint (no auth required)
        syms_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
        for sym, pair in syms_map.items():
            try:
                url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={pair}&limit=1"
                req = urllib.request.Request(url, headers={"User-Agent": "srfm-lab/1.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data and isinstance(data, list):
                        fr = float(data[0].get("fundingRate", 0))
                        with self._lock:
                            self._funding_rates[sym] = fr
            except Exception as e:
                log.debug("Funding rate fetch failed for %s: %s", sym, e)
