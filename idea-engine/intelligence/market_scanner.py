"""
Market Scanner: real-time screening across the entire investable universe.

Like a Bloomberg terminal scanner but powered by the Event Horizon system:
  - Scans ALL assets in the universe simultaneously
  - Runs the AssetIntelligenceEngine on each
  - Ranks by composite score (best opportunities first)
  - Filters by regime, signal strength, liquidity, risk
  - Detects "setup alerts" (specific technical/fundamental conditions)
  - Identifies cross-asset themes (what's moving together and why)
  - Generates a "Morning Brief" with top opportunities and risks

This is what the PM looks at FIRST every morning.
"""

from __future__ import annotations
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import numpy as np


@dataclass
class ScanResult:
    """Result of scanning one asset."""
    symbol: str
    asset_class: str
    current_price: float
    change_24h_pct: float
    overall_score: float          # -1 to +1
    confidence: float
    action: str                   # strong_buy / buy / hold / sell / strong_sell
    regime: str
    technical_score: float
    momentum: float
    rsi: float
    vol_percentile: float
    liquidity_tier: str
    key_signal: str               # dominant signal name
    highlights: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


@dataclass
class SetupAlert:
    """A specific trading setup detected."""
    symbol: str
    setup_type: str               # "oversold_bounce" / "breakout" / "divergence" / etc
    direction: str                # "long" / "short"
    strength: float               # 0-1
    description: str
    timeframe: str                # "intraday" / "swing" / "position"


@dataclass
class MarketTheme:
    """A cross-asset theme detected."""
    theme_name: str
    description: str
    affected_assets: List[str]
    direction: str
    confidence: float


@dataclass
class MorningBrief:
    """The daily morning brief for the PM."""
    date: str
    market_regime: str
    fear_greed_label: str

    # Top opportunities
    top_longs: List[ScanResult]
    top_shorts: List[ScanResult]

    # Alerts
    setup_alerts: List[SetupAlert]
    themes: List[MarketTheme]

    # Risk warnings
    risk_warnings: List[str]

    # Summary
    executive_summary: str


class MarketScanner:
    """
    Scan the entire universe and produce a ranked list of opportunities.
    """

    def __init__(self, universe: List[str] = None):
        self.universe = universe or [
            "BTC", "ETH", "SOL", "AVAX", "LINK", "AAVE", "LTC",
            "SPY", "QQQ", "GLD", "TLT", "NVDA", "AAPL", "TSLA",
        ]
        self._scan_history: List[Dict] = []

    def scan(
        self,
        price_data: Dict[str, np.ndarray],       # symbol -> (T,) prices
        volume_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[ScanResult]:
        """Scan all assets and rank by opportunity."""
        results = []

        for symbol in self.universe:
            if symbol not in price_data:
                continue

            prices = price_data[symbol]
            volumes = volume_data.get(symbol) if volume_data else None

            if len(prices) < 50:
                continue

            result = self._analyze_asset(symbol, prices, volumes)
            results.append(result)

        # Sort by absolute score (best opportunities first)
        results.sort(key=lambda r: abs(r.overall_score) * r.confidence, reverse=True)
        return results

    def detect_setups(self, price_data: Dict[str, np.ndarray]) -> List[SetupAlert]:
        """Detect specific trading setups across the universe."""
        alerts = []

        for symbol in self.universe:
            if symbol not in price_data:
                continue

            prices = price_data[symbol]
            if len(prices) < 100:
                continue

            returns = np.diff(np.log(prices + 1e-10))

            # Setup 1: Oversold bounce (RSI < 25 + positive momentum flip)
            if len(returns) >= 15:
                changes = np.diff(prices[-15:])
                gains = np.maximum(changes, 0).mean()
                losses = np.maximum(-changes, 0).mean()
                rsi = 100 - 100 / (1 + gains / max(losses, 1e-10))

                if rsi < 25 and returns[-1] > 0:
                    alerts.append(SetupAlert(
                        symbol=symbol,
                        setup_type="oversold_bounce",
                        direction="long",
                        strength=min(1.0, (25 - rsi) / 25),
                        description=f"{symbol} RSI={rsi:.0f} with positive momentum flip",
                        timeframe="swing",
                    ))

            # Setup 2: Breakout (new 20-day high on expanding volume)
            if len(prices) >= 21:
                if prices[-1] >= prices[-20:].max():
                    if volume_data and symbol in volume_data:
                        vol_ratio = volumes[-5:].mean() / max(volumes[-20:].mean(), 1e-10) if volumes is not None else 1
                    else:
                        vol_ratio = 1.0
                    if vol_ratio > 1.2:
                        alerts.append(SetupAlert(
                            symbol=symbol,
                            setup_type="volume_breakout",
                            direction="long",
                            strength=min(1.0, vol_ratio / 2),
                            description=f"{symbol} new 20-day high on {vol_ratio:.1f}x average volume",
                            timeframe="swing",
                        ))

            # Setup 3: Death cross (50 SMA crosses below 200 SMA)
            if len(prices) >= 201:
                sma50 = prices[-50:].mean()
                sma200 = prices[-200:].mean()
                sma50_prev = prices[-51:-1].mean()
                sma200_prev = prices[-201:-1].mean()
                if sma50 < sma200 and sma50_prev > sma200_prev:
                    alerts.append(SetupAlert(
                        symbol=symbol,
                        setup_type="death_cross",
                        direction="short",
                        strength=0.6,
                        description=f"{symbol} 50 SMA crossed below 200 SMA (death cross)",
                        timeframe="position",
                    ))

            # Setup 4: Bollinger squeeze (bandwidth at 6-month low)
            if len(prices) >= 126:
                bw_20 = float(prices[-20:].std() / max(prices[-20:].mean(), 1e-10))
                bw_history = [float(prices[i-20:i].std() / max(prices[i-20:i].mean(), 1e-10))
                               for i in range(20, min(126, len(prices)))]
                if bw_history and bw_20 < np.percentile(bw_history, 10):
                    alerts.append(SetupAlert(
                        symbol=symbol,
                        setup_type="bollinger_squeeze",
                        direction="long",  # breakout direction TBD
                        strength=0.5,
                        description=f"{symbol} Bollinger bandwidth at 6-month low (squeeze forming)",
                        timeframe="swing",
                    ))

        alerts.sort(key=lambda a: a.strength, reverse=True)
        return alerts

    def detect_themes(self, scan_results: List[ScanResult]) -> List[MarketTheme]:
        """Detect cross-asset themes from scan results."""
        themes = []

        # Theme 1: Broad market direction
        n_bullish = sum(1 for r in scan_results if r.overall_score > 0.2)
        n_bearish = sum(1 for r in scan_results if r.overall_score < -0.2)
        total = len(scan_results)

        if n_bullish > total * 0.7:
            themes.append(MarketTheme(
                "Broad Bullish", "Majority of assets are bullish across all sectors",
                [r.symbol for r in scan_results if r.overall_score > 0.2][:10],
                "bullish", min(1.0, n_bullish / total),
            ))
        elif n_bearish > total * 0.7:
            themes.append(MarketTheme(
                "Broad Bearish", "Majority of assets are bearish across all sectors",
                [r.symbol for r in scan_results if r.overall_score < -0.2][:10],
                "bearish", min(1.0, n_bearish / total),
            ))

        # Theme 2: Crypto vs equities divergence
        crypto = [r for r in scan_results if r.asset_class == "crypto"]
        equity = [r for r in scan_results if r.asset_class == "equity"]
        if crypto and equity:
            crypto_avg = float(np.mean([r.overall_score for r in crypto]))
            equity_avg = float(np.mean([r.overall_score for r in equity]))
            if crypto_avg - equity_avg > 0.3:
                themes.append(MarketTheme(
                    "Crypto Outperforming", "Crypto assets leading over equities",
                    [r.symbol for r in crypto if r.overall_score > 0], "bullish_crypto", 0.7,
                ))
            elif equity_avg - crypto_avg > 0.3:
                themes.append(MarketTheme(
                    "Risk-Off Rotation", "Capital rotating from crypto to equities/bonds",
                    [r.symbol for r in equity if r.overall_score > 0], "risk_off", 0.7,
                ))

        # Theme 3: Volatility expansion
        high_vol = [r for r in scan_results if r.vol_percentile > 0.8]
        if len(high_vol) > total * 0.5:
            themes.append(MarketTheme(
                "Volatility Expansion", "Majority of assets at elevated volatility percentile",
                [r.symbol for r in high_vol], "volatile", 0.8,
            ))

        return themes

    def generate_morning_brief(
        self,
        scan_results: List[ScanResult],
        setup_alerts: List[SetupAlert],
        themes: List[MarketTheme],
        fear_greed_label: str = "neutral",
    ) -> MorningBrief:
        """Generate the daily morning brief."""
        top_longs = [r for r in scan_results if r.action in ("strong_buy", "buy")][:5]
        top_shorts = [r for r in scan_results if r.action in ("strong_sell", "sell")][:5]

        # Risk warnings
        warnings = []
        if any(r.vol_percentile > 0.9 for r in scan_results):
            warnings.append("Multiple assets at extreme volatility percentile")
        if fear_greed_label in ("extreme_greed",):
            warnings.append("Internal fear/greed at extreme greed - contrarian caution")
        if len(setup_alerts) == 0:
            warnings.append("No clear setups detected - consider reducing activity")

        # Determine market regime from consensus
        regimes = [r.regime for r in scan_results]
        regime_counts = defaultdict(int)
        for r in regimes:
            regime_counts[r] += 1
        market_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "unknown"

        # Executive summary
        summary_parts = [f"Market regime: {market_regime}."]
        if top_longs:
            summary_parts.append(f"Top long: {top_longs[0].symbol} (score {top_longs[0].overall_score:+.2f}).")
        if top_shorts:
            summary_parts.append(f"Top short: {top_shorts[0].symbol} (score {top_shorts[0].overall_score:+.2f}).")
        summary_parts.append(f"{len(setup_alerts)} setup alerts detected.")
        if themes:
            summary_parts.append(f"Key theme: {themes[0].theme_name}.")

        return MorningBrief(
            date=time.strftime("%Y-%m-%d"),
            market_regime=market_regime,
            fear_greed_label=fear_greed_label,
            top_longs=top_longs,
            top_shorts=top_shorts,
            setup_alerts=setup_alerts[:10],
            themes=themes,
            risk_warnings=warnings,
            executive_summary=" ".join(summary_parts),
        )

    def _analyze_asset(self, symbol: str, prices: np.ndarray,
                         volumes: Optional[np.ndarray]) -> ScanResult:
        """Quick analysis of one asset for scanning."""
        T = len(prices)
        returns = np.diff(np.log(prices + 1e-10))

        # RSI
        if len(returns) >= 15:
            changes = np.diff(prices[-15:])
            gains = np.maximum(changes, 0).mean()
            losses = np.maximum(-changes, 0).mean()
            rsi = 100 - 100 / (1 + gains / max(losses, 1e-10))
        else:
            rsi = 50

        # Momentum
        if len(returns) >= 21:
            mom = float(returns[-21:].mean() / max(returns[-21:].std(), 1e-8))
            momentum = float(np.tanh(mom))
        else:
            momentum = 0.0

        # Technical score
        score = 0.0
        if rsi < 30: score += 0.3
        elif rsi > 70: score -= 0.3
        score += momentum * 0.4

        if T >= 200:
            sma200 = prices[-200:].mean()
            if prices[-1] > sma200:
                score += 0.15
            else:
                score -= 0.15

        # Vol percentile
        if len(returns) >= 252:
            current_vol = float(returns[-21:].std())
            all_vols = [float(returns[i-21:i].std()) for i in range(21, len(returns))]
            vol_pct = float(np.mean(np.array(all_vols) <= current_vol))
        else:
            vol_pct = 0.5

        # Regime
        vol = float(returns[-21:].std() * math.sqrt(252)) if len(returns) >= 21 else 0.15
        trend = float(returns[-21:].mean() * 252) if len(returns) >= 21 else 0
        if vol > 0.50: regime = "crisis"
        elif vol > 0.25: regime = "high_vol"
        elif trend > 0.20: regime = "trending_up"
        elif trend < -0.20: regime = "trending_down"
        else: regime = "ranging"

        # Action
        if score > 0.5: action = "strong_buy"
        elif score > 0.2: action = "buy"
        elif score < -0.5: action = "strong_sell"
        elif score < -0.2: action = "sell"
        else: action = "hold"

        confidence = min(1.0, abs(score) * 2)

        # Asset class guess
        crypto_symbols = {"BTC", "ETH", "SOL", "AVAX", "LINK", "AAVE", "LTC", "XRP"}
        asset_class = "crypto" if symbol in crypto_symbols else "equity"

        # 24h change
        change_24h = float((prices[-1] / prices[-2] - 1) * 100) if T >= 2 else 0

        # Highlights
        highlights = []
        if rsi < 30: highlights.append(f"RSI oversold ({rsi:.0f})")
        if rsi > 70: highlights.append(f"RSI overbought ({rsi:.0f})")
        if momentum > 0.5: highlights.append("Strong positive momentum")
        if T >= 20 and prices[-1] >= prices[-20:].max(): highlights.append("New 20-day high")

        # Risks
        risks = []
        if vol_pct > 0.9: risks.append("Extreme vol percentile")
        if vol > 0.50: risks.append("Crisis-level volatility")

        return ScanResult(
            symbol=symbol,
            asset_class=asset_class,
            current_price=float(prices[-1]),
            change_24h_pct=change_24h,
            overall_score=float(np.clip(score, -1, 1)),
            confidence=confidence,
            action=action,
            regime=regime,
            technical_score=score,
            momentum=momentum,
            rsi=rsi,
            vol_percentile=vol_pct,
            liquidity_tier="tier1" if asset_class == "equity" else "tier2",
            key_signal="momentum" if abs(momentum) > 0.3 else "rsi" if abs(rsi - 50) > 20 else "neutral",
            highlights=highlights,
            risks=risks,
        )
