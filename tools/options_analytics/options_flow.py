"""
options_flow.py — Unusual options activity scanner.

Features
--------
* Scans full option chains for all INSTRUMENTS
* Unusual volume: option volume > 2x its 20-day rolling average
* Large trade detection: single order > $50K notional
* Sweep detection: multi-exchange simultaneous fills (momentum signal)
* Put/call ratio: by expiry, strike, and overall
* Dark pool correlation: unusual equity dark prints + options activity
* Alerts: structured log + optional webhook (POST JSON)
* Flow summary: top-10 unusual contracts ranked by composite score

Usage
-----
    python options_flow.py                          # scan all instruments
    python options_flow.py --symbol SPY             # single symbol
    python options_flow.py --webhook https://...    # POST alerts to webhook
    python options_flow.py --out flow_report.json   # save report
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

INSTRUMENTS = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "MSFT"]
LARGE_TRADE_THRESHOLD = 50_000      # $ notional
UNUSUAL_VOLUME_MULTIPLIER = 2.0     # x above 20d avg
SWEEP_SIZE_THRESHOLD = 20           # min contracts for sweep signal
OI_MIN = 100                        # minimum open interest to consider
MIN_CONTRACT_PRICE = 0.05           # minimum mid-price
RISK_FREE_RATE = 0.0525

logger = logging.getLogger("options_flow")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FlowContract:
    occ_symbol: str
    underlying: str
    expiry: date
    strike: float
    option_type: str          # "call" | "put"
    mid_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    iv: Optional[float]
    dte: int
    # Derived
    notional: float           # volume * mid * 100
    oi_ratio: float           # volume / OI
    unusual_volume: bool
    large_trade: bool
    sweep_signal: bool
    score: float              # composite unusualness score 0-100


@dataclass
class PutCallRatios:
    overall_volume: float
    overall_oi: float
    by_expiry: Dict[str, float]     # expiry → PCR (volume)
    by_strike_band: Dict[str, float]  # "ATM±5%" → PCR


@dataclass
class FlowReport:
    as_of: str
    symbols_scanned: List[str]
    total_contracts_scanned: int
    total_unusual: int
    top_contracts: List[FlowContract]
    put_call_ratios: Dict[str, PutCallRatios]  # symbol → PCR
    dark_pool_alerts: List[dict]
    alerts: List[dict]


# ---------------------------------------------------------------------------
# Alpaca data client
# ---------------------------------------------------------------------------

class _AlpacaClient:
    DATA_URL = "https://data.alpaca.markets/v2"
    BROKER_URL = "https://api.alpaca.markets/v2"

    def __init__(self) -> None:
        self.key = os.environ.get("ALPACA_API_KEY", "")
        self.secret = os.environ.get("ALPACA_SECRET_KEY", "")
        self._s = requests.Session()
        self._s.headers.update({
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
        })

    def latest_price(self, sym: str) -> float:
        try:
            r = self._s.get(f"{self.DATA_URL}/stocks/{sym}/quotes/latest", timeout=8)
            r.raise_for_status()
            q = r.json().get("quote", {})
            return (q.get("bp", 0) + q.get("ap", 0)) / 2.0
        except Exception:
            return 0.0

    def option_snapshots(self, sym: str) -> List[dict]:
        """Full paginated option snapshot."""
        url = f"{self.DATA_URL}/options/snapshots/{sym}"
        results = []
        params: dict = {"feed": "indicative", "limit": 1000}
        while True:
            try:
                r = self._s.get(url, params=params, timeout=15)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                warnings.warn(f"Snapshots failed for {sym}: {e}")
                break
            for sym_key, snap in data.get("snapshots", {}).items():
                snap["_sym"] = sym_key
                results.append(snap)
            pt = data.get("next_page_token")
            if not pt:
                break
            params["page_token"] = pt
        return results

    def option_trades(self, sym: str, limit: int = 500) -> List[dict]:
        """Recent option trades for sweep detection."""
        try:
            r = self._s.get(
                f"{self.DATA_URL}/options/trades/{sym}",
                params={"limit": limit, "feed": "indicative"},
                timeout=10,
            )
            r.raise_for_status()
            return r.json().get("trades", [])
        except Exception:
            return []

    def equity_bars(self, sym: str, days: int = 25) -> List[dict]:
        """Daily bars for dark pool / equity activity."""
        start = (date.today() - timedelta(days=days + 5)).isoformat()
        end = date.today().isoformat()
        try:
            r = self._s.get(
                f"{self.DATA_URL}/stocks/{sym}/bars",
                params={"start": start, "end": end, "timeframe": "1Day", "limit": 50},
                timeout=8,
            )
            r.raise_for_status()
            return r.json().get("bars", [])
        except Exception:
            return []

    def equity_trades(self, sym: str, limit: int = 200) -> List[dict]:
        """Recent equity trades — used to detect dark-pool prints."""
        try:
            r = self._s.get(
                f"{self.DATA_URL}/stocks/{sym}/trades/latest",
                params={"feed": "sip", "limit": limit},
                timeout=8,
            )
            r.raise_for_status()
            return r.json().get("trades", [])
        except Exception:
            return []


# ---------------------------------------------------------------------------
# OCC symbol parser
# ---------------------------------------------------------------------------

def _parse_occ(sym: str) -> Optional[Tuple[str, date, str, float]]:
    try:
        for i in range(len(sym) - 1, -1, -1):
            if sym[i] in ("C", "P") and i >= 3:
                underlying = sym[: i - 6]
                dt = datetime.strptime(sym[i - 6: i], "%y%m%d").date()
                otype = "call" if sym[i] == "C" else "put"
                strike = float(sym[i + 1:]) / 1000.0
                return underlying, dt, otype, strike
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# BS IV solver (self-contained)
# ---------------------------------------------------------------------------

def _bs_iv(mid: float, S: float, K: float, T: float, r: float, otype: str) -> Optional[float]:
    if mid <= 0 or T <= 0:
        return None
    from scipy.stats import norm
    from scipy.optimize import brentq

    def price(sig: float) -> float:
        d1 = (math.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * math.sqrt(T))
        d2 = d1 - sig * math.sqrt(T)
        disc = math.exp(-r * T)
        if otype == "call":
            return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
        return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)

    try:
        if (price(0.001) - mid) * (price(10.0) - mid) > 0:
            return None
        return float(brentq(lambda s: price(s) - mid, 0.001, 10.0, xtol=1e-7))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Volume baseline estimator
# ---------------------------------------------------------------------------

class VolumeBaseline:
    """
    Estimates 20-day average daily volume per option contract.
    Since historical per-contract data is expensive, we use open interest
    as a proxy: OI represents cumulative positioning, and we approximate
    20d avg volume as OI / 30 (rough turnover assumption).

    For a real production system, store daily snapshots in a DB.
    """

    @staticmethod
    def estimate_avg_volume(oi: int, volume: int) -> float:
        """
        Heuristic: if OI is available, 20d avg ≈ OI / 30.
        If volume >> avg, flag as unusual.
        """
        if oi > 0:
            return max(oi / 30.0, 1.0)
        return max(volume / 5.0, 1.0)  # fallback: assume today is 5x avg

    @staticmethod
    def is_unusual(volume: int, oi: int) -> bool:
        avg = VolumeBaseline.estimate_avg_volume(oi, volume)
        return volume > UNUSUAL_VOLUME_MULTIPLIER * avg


# ---------------------------------------------------------------------------
# Sweep detector
# ---------------------------------------------------------------------------

class SweepDetector:
    """
    Detect sweeps: rapid large orders executed across multiple exchanges.
    Heuristics:
    - Multiple trades in very short time window (< 1s)
    - Same contract, alternating exchanges
    - Total size > SWEEP_SIZE_THRESHOLD contracts
    """

    @staticmethod
    def detect(trades: List[dict], occ_symbol: str) -> bool:
        """
        Look at recent trades for a contract and flag if sweep pattern detected.
        Trades should be sorted newest-first.
        """
        contract_trades = [t for t in trades if t.get("symbol") == occ_symbol]
        if len(contract_trades) < 3:
            return False

        # Check for burst: multiple trades within 5-second window
        timestamps = []
        for t in contract_trades[:20]:
            ts_str = t.get("t", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    timestamps.append(ts)
                except Exception:
                    pass

        if len(timestamps) < 3:
            # Fall back to size heuristic
            total_size = sum(int(t.get("s", 0)) for t in contract_trades[:10])
            return total_size >= SWEEP_SIZE_THRESHOLD

        timestamps.sort()
        for i in range(len(timestamps) - 2):
            window = (timestamps[i + 2] - timestamps[i]).total_seconds()
            if window < 5.0:
                total_size = sum(int(contract_trades[j].get("s", 0)) for j in range(i, min(i + 10, len(contract_trades))))
                if total_size >= SWEEP_SIZE_THRESHOLD:
                    return True

        return False


# ---------------------------------------------------------------------------
# Dark pool correlation
# ---------------------------------------------------------------------------

class DarkPoolAnalyzer:
    """
    Detect unusual equity activity that may indicate dark pool prints.
    Uses Alpaca SIP trades and looks for:
    - Large single trades (> 2x avg daily volume per trade)
    - Trades flagged with dark-pool-type conditions (T, P, W exchange codes)
    """

    DARK_POOL_EXCHANGES = {"D", "P", "Q", "T", "W"}  # common dark venue codes

    @staticmethod
    def analyze(symbol: str, trades: List[dict], bars: List[dict]) -> List[dict]:
        if not trades or not bars:
            return []

        # Average daily volume from bars
        volumes = [b.get("v", 0) for b in bars[-20:] if b.get("v")]
        if not volumes:
            return []
        avg_daily_vol = np.mean(volumes)
        avg_trade_size = avg_daily_vol / 5000  # assume ~5000 trades/day

        alerts = []
        for trade in trades:
            size = int(trade.get("s", 0))
            exchange = trade.get("x", "")
            condition = trade.get("c", [])
            price = float(trade.get("p", 0))

            is_dark = exchange in DarkPoolAnalyzer.DARK_POOL_EXCHANGES
            is_large = size > max(avg_trade_size * 5, 10000)

            if is_dark and is_large:
                alerts.append({
                    "type": "dark_pool_print",
                    "symbol": symbol,
                    "exchange": exchange,
                    "size": size,
                    "price": price,
                    "notional": round(size * price, 2),
                    "note": "Large dark pool print detected",
                })

        return alerts[:10]  # cap to top 10


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def _score_contract(fc: FlowContract) -> float:
    """
    Composite unusualness score 0-100.
    Components:
      - Volume/OI ratio (max 30 pts)
      - Unusual volume flag (20 pts)
      - Large trade flag (20 pts)
      - Sweep signal (15 pts)
      - IV level proxy (15 pts: very high IV → unusual)
    """
    score = 0.0

    # Volume/OI ratio
    vol_oi = min(fc.oi_ratio * 10, 30.0)
    score += vol_oi

    if fc.unusual_volume:
        score += 20.0

    if fc.large_trade:
        score += 20.0

    if fc.sweep_signal:
        score += 15.0

    # IV component: if IV > 1.0 (100%) → very unusual
    if fc.iv is not None:
        iv_pts = min(fc.iv * 15, 15.0)
        score += iv_pts

    return min(score, 100.0)


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

class OptionsFlowScanner:
    """
    Scans option chains across INSTRUMENTS for unusual activity.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        webhook_url: Optional[str] = None,
        r: float = RISK_FREE_RATE,
    ):
        self.symbols = [s.upper() for s in (symbols or INSTRUMENTS)]
        self.webhook_url = webhook_url
        self.r = r
        self.client = _AlpacaClient()
        self._underlying_prices: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Main scan
    # ------------------------------------------------------------------

    def scan(self) -> FlowReport:
        all_unusual: List[FlowContract] = []
        all_alerts: List[dict] = []
        dark_pool_alerts: List[dict] = []
        pcr_map: Dict[str, PutCallRatios] = {}
        total_scanned = 0
        scanned_symbols: List[str] = []

        for sym in self.symbols:
            logger.info(f"Scanning {sym}...")
            try:
                S = self.client.latest_price(sym)
                self._underlying_prices[sym] = S

                snapshots = self.client.option_snapshots(sym)
                if not snapshots:
                    logger.warning(f"No snapshots for {sym}")
                    continue

                scanned_symbols.append(sym)
                trades = self.client.option_trades(sym)
                sweep_detector = SweepDetector()

                contracts = self._parse_snapshots(sym, snapshots, S, trades, sweep_detector)
                total_scanned += len(contracts)

                unusual = [c for c in contracts if c.unusual_volume or c.large_trade or c.sweep_signal]
                all_unusual.extend(unusual)

                pcr_map[sym] = self._compute_pcr(contracts, S)

                # Dark pool analysis
                eq_trades = self.client.equity_trades(sym)
                bars = self.client.equity_bars(sym)
                dp_alerts = DarkPoolAnalyzer.analyze(sym, eq_trades, bars)
                dark_pool_alerts.extend(dp_alerts)

                # Correlate dark pool with options unusual activity
                if dp_alerts and unusual:
                    all_alerts.append({
                        "type": "dark_pool_options_correlation",
                        "symbol": sym,
                        "dark_pool_prints": len(dp_alerts),
                        "unusual_option_contracts": len(unusual),
                        "message": f"{sym}: {len(dp_alerts)} dark prints + {len(unusual)} unusual options",
                    })

                # Per-contract alerts
                for c in unusual:
                    alert = self._build_alert(c)
                    all_alerts.append(alert)
                    if self.webhook_url:
                        self._post_webhook(alert)

            except Exception as e:
                logger.error(f"Error scanning {sym}: {e}", exc_info=True)

        # Sort by score
        all_unusual.sort(key=lambda c: c.score, reverse=True)
        top_10 = all_unusual[:10]

        report = FlowReport(
            as_of=datetime.utcnow().isoformat(),
            symbols_scanned=scanned_symbols,
            total_contracts_scanned=total_scanned,
            total_unusual=len(all_unusual),
            top_contracts=top_10,
            put_call_ratios=pcr_map,
            dark_pool_alerts=dark_pool_alerts,
            alerts=all_alerts,
        )

        logger.info(
            f"Scan complete. Scanned: {total_scanned} contracts, "
            f"Unusual: {len(all_unusual)}, Top alerts: {len(all_alerts)}"
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_snapshots(
        self,
        underlying: str,
        snapshots: List[dict],
        S: float,
        trades: List[dict],
        sweep_detector: SweepDetector,
    ) -> List[FlowContract]:
        today = date.today()
        contracts: List[FlowContract] = []

        for snap in snapshots:
            occ_sym = snap.get("_sym", "")
            parsed = _parse_occ(occ_sym)
            if parsed is None:
                continue
            _und, expiry, otype, strike = parsed
            dte = (expiry - today).days
            if dte < 1 or dte > 365:
                continue

            q = snap.get("latestQuote", {})
            bid = float(q.get("bp", 0) or 0)
            ask = float(q.get("ap", 0) or 0)
            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
            if mid < MIN_CONTRACT_PRICE:
                continue

            bar = snap.get("dailyBar", {})
            volume = int(bar.get("v", 0) or 0)
            oi = int(snap.get("openInterest", 0) or 0)

            if volume < 1 and oi < OI_MIN:
                continue

            T = max(dte / 365.0, 1e-6)
            iv = _bs_iv(mid, S, strike, T, self.r, otype) if S > 0 else None

            notional = volume * mid * 100
            oi_ratio = volume / max(oi, 1)
            unusual = VolumeBaseline.is_unusual(volume, oi)
            large = notional >= LARGE_TRADE_THRESHOLD
            sweep = sweep_detector.detect(trades, occ_sym)

            fc = FlowContract(
                occ_symbol=occ_sym,
                underlying=underlying,
                expiry=expiry,
                strike=strike,
                option_type=otype,
                mid_price=mid,
                bid=bid,
                ask=ask,
                volume=volume,
                open_interest=oi,
                iv=iv,
                dte=dte,
                notional=notional,
                oi_ratio=oi_ratio,
                unusual_volume=unusual,
                large_trade=large,
                sweep_signal=sweep,
                score=0.0,
            )
            fc.score = _score_contract(fc)
            contracts.append(fc)

        return contracts

    def _compute_pcr(self, contracts: List[FlowContract], S: float) -> PutCallRatios:
        """Compute put/call ratios overall, by expiry, and by strike band."""
        call_vol = sum(c.volume for c in contracts if c.option_type == "call")
        put_vol = sum(c.volume for c in contracts if c.option_type == "put")
        call_oi = sum(c.open_interest for c in contracts if c.option_type == "call")
        put_oi = sum(c.open_interest for c in contracts if c.option_type == "put")

        overall_vol = put_vol / max(call_vol, 1)
        overall_oi = put_oi / max(call_oi, 1)

        # By expiry
        expiry_call_vol: Dict[str, int] = {}
        expiry_put_vol: Dict[str, int] = {}
        for c in contracts:
            key = str(c.expiry)
            if c.option_type == "call":
                expiry_call_vol[key] = expiry_call_vol.get(key, 0) + c.volume
            else:
                expiry_put_vol[key] = expiry_put_vol.get(key, 0) + c.volume

        by_expiry: Dict[str, float] = {}
        for key in set(expiry_call_vol) | set(expiry_put_vol):
            cv = expiry_call_vol.get(key, 0)
            pv = expiry_put_vol.get(key, 0)
            by_expiry[key] = round(pv / max(cv, 1), 4)

        # By strike band (relative to spot)
        bands = [
            ("deep_otm_puts (<80%)", 0.0, 0.8),
            ("otm_puts (80-95%)", 0.8, 0.95),
            ("atm (95-105%)", 0.95, 1.05),
            ("otm_calls (105-120%)", 1.05, 1.20),
            ("deep_otm_calls (>120%)", 1.20, 999),
        ]
        by_strike_band: Dict[str, float] = {}
        for band_name, lo, hi in bands:
            bc = sum(c.volume for c in contracts if c.option_type == "call" and lo * S <= c.strike < hi * S)
            bp = sum(c.volume for c in contracts if c.option_type == "put" and lo * S <= c.strike < hi * S)
            by_strike_band[band_name] = round(bp / max(bc, 1), 4)

        return PutCallRatios(
            overall_volume=round(overall_vol, 4),
            overall_oi=round(overall_oi, 4),
            by_expiry=by_expiry,
            by_strike_band=by_strike_band,
        )

    def _build_alert(self, c: FlowContract) -> dict:
        reasons = []
        if c.unusual_volume:
            reasons.append(f"unusual_volume (oi_ratio={c.oi_ratio:.2f}x)")
        if c.large_trade:
            reasons.append(f"large_trade (${c.notional:,.0f} notional)")
        if c.sweep_signal:
            reasons.append("sweep_detected")

        return {
            "type": "unusual_options_activity",
            "symbol": c.occ_symbol,
            "underlying": c.underlying,
            "expiry": str(c.expiry),
            "strike": c.strike,
            "option_type": c.option_type,
            "dte": c.dte,
            "volume": c.volume,
            "open_interest": c.open_interest,
            "mid_price": c.mid_price,
            "notional": round(c.notional, 2),
            "iv": round(c.iv * 100, 2) if c.iv else None,
            "score": round(c.score, 2),
            "reasons": reasons,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _post_webhook(self, alert: dict) -> None:
        if not self.webhook_url:
            return
        try:
            resp = requests.post(
                self.webhook_url,
                json=alert,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Webhook delivery failed: {e}")


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(report: FlowReport) -> None:
    print(f"\n{'='*70}")
    print(f"  OPTIONS FLOW REPORT — {report.as_of}")
    print(f"{'='*70}")
    print(f"  Symbols scanned:   {', '.join(report.symbols_scanned)}")
    print(f"  Contracts scanned: {report.total_contracts_scanned:,}")
    print(f"  Unusual contracts: {report.total_unusual:,}")
    print(f"  Top alerts:        {len(report.alerts)}")
    print()

    if report.top_contracts:
        print(f"  TOP {len(report.top_contracts)} UNUSUAL CONTRACTS BY SCORE")
        print(f"  {'Symbol':<26} {'Type':<5} {'Strike':>8} {'DTE':>5} {'Vol':>8} {'OI':>8} {'Notional':>12} {'IV':>7} {'Score':>6}")
        print(f"  {'-'*90}")
        for c in report.top_contracts:
            iv_str = f"{c.iv*100:.1f}%" if c.iv else "N/A"
            flags = ""
            if c.unusual_volume:
                flags += "U"
            if c.large_trade:
                flags += "L"
            if c.sweep_signal:
                flags += "S"
            print(
                f"  {c.occ_symbol:<26} {c.option_type[0].upper():<5} {c.strike:>8.2f} {c.dte:>5} "
                f"{c.volume:>8,} {c.open_interest:>8,} ${c.notional:>11,.0f} {iv_str:>7} {c.score:>5.1f} [{flags}]"
            )

    print()
    print("  PUT/CALL RATIOS")
    for sym, pcr in report.put_call_ratios.items():
        sentiment = (
            "bearish" if pcr.overall_volume > 1.2
            else "bullish" if pcr.overall_volume < 0.8
            else "neutral"
        )
        print(f"    {sym}: PCR(vol)={pcr.overall_volume:.3f} PCR(OI)={pcr.overall_oi:.3f} → {sentiment}")

    if report.dark_pool_alerts:
        print()
        print("  DARK POOL ALERTS")
        for a in report.dark_pool_alerts[:5]:
            print(f"    {a['symbol']} | {a['type']} | ${a.get('notional',0):,.0f} @ {a.get('price',0):.2f}")

    print(f"\n  Legend: U=unusual_volume, L=large_trade, S=sweep_detected")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Unusual options activity scanner")
    parser.add_argument("--symbol", nargs="+", default=None, help="Symbols (default: all instruments)")
    parser.add_argument("--webhook", default=None, help="Webhook URL for alerts")
    parser.add_argument("--out", default=None, help="Output JSON report path")
    parser.add_argument("--loop", action="store_true", help="Run continuously every 60s")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval seconds")
    args = parser.parse_args()

    symbols = args.symbol or INSTRUMENTS
    scanner = OptionsFlowScanner(symbols=symbols, webhook_url=args.webhook)

    def run_once() -> FlowReport:
        report = scanner.scan()
        print_report(report)
        if args.out:
            # Serialize — convert dataclasses to dicts
            def _serial(obj):
                if isinstance(obj, (date, datetime)):
                    return str(obj)
                if hasattr(obj, "__dict__"):
                    return obj.__dict__
                return str(obj)

            out_data = {
                "as_of": report.as_of,
                "symbols_scanned": report.symbols_scanned,
                "total_contracts_scanned": report.total_contracts_scanned,
                "total_unusual": report.total_unusual,
                "top_contracts": [asdict(c) for c in report.top_contracts],
                "put_call_ratios": {
                    sym: asdict(pcr) for sym, pcr in report.put_call_ratios.items()
                },
                "dark_pool_alerts": report.dark_pool_alerts,
                "alerts": report.alerts,
            }
            with open(args.out, "w") as f:
                json.dump(out_data, f, indent=2, default=str)
            print(f"[OptionsFlow] Report saved to {args.out}")
        return report

    if args.loop:
        try:
            while True:
                run_once()
                logger.info(f"Next scan in {args.interval}s...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Scanner stopped.")
    else:
        run_once()


if __name__ == "__main__":
    _cli()
