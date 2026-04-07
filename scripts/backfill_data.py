#!/usr/bin/env python3
"""
backfill_data.py -- Historical data backfill utility for SRFM.

Fetches OHLCV data from Alpaca (equities) or Binance/CoinGecko (crypto)
and stores it to DuckDB via the coordinator data store API.

Supports checkpointing to resume interrupted backfills and gap verification.

Usage:
  python scripts/backfill_data.py --symbol BTC-USD --start 2022-01-01 --end 2024-01-01
  python scripts/backfill_data.py --symbol AAPL --start 2023-01-01 --end 2024-01-01 --timeframe 1d
  python scripts/backfill_data.py --symbol BTC-USD --verify --start 2022-01-01 --end 2024-01-01
  python scripts/backfill_data.py --symbol BTC-USD --resume
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
CHECKPOINT_DIR = REPO_ROOT / ".backfill_checkpoints"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s -- %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(str(LOGS_DIR / "backfill.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = build_logger("srfm.backfill")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COORDINATOR_BASE = os.environ.get("SRFM_COORDINATOR_URL", "http://localhost:8000")
ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://data.alpaca.markets")
ALPACA_KEY = os.environ.get("APCA_API_KEY_ID", "")
ALPACA_SECRET = os.environ.get("APCA_API_SECRET_KEY", "")
BINANCE_BASE = os.environ.get("BINANCE_BASE_URL", "https://api.binance.com")
HTTP_TIMEOUT = 30
RATE_LIMIT_SLEEP = 1.0  # seconds between API calls (conservative)
ALPACA_MAX_BARS_PER_REQUEST = 10000
BINANCE_MAX_BARS_PER_REQUEST = 1000
COINGECKO_MAX_DAYS = 365

# Timeframe aliases
TIMEFRAME_MINUTES: Dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

CRYPTO_SYMBOLS = {"BTC", "ETH", "BNB", "SOL", "ADA", "AVAX", "MATIC", "DOGE", "XRP", "DOT"}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get_json(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[int, Optional[Dict]]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            body = resp.read()
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return exc.code, {"error": str(exc)}
    except (urllib.error.URLError, OSError) as exc:
        log.debug("http_get_json %s: %s", url, exc)
        return 0, None


def http_post_json(url: str, payload: Dict) -> Tuple[int, Optional[Dict]]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return exc.code, {"error": str(exc)}
    except (urllib.error.URLError, OSError) as exc:
        log.debug("http_post_json %s: %s", url, exc)
        return 0, None


# ---------------------------------------------------------------------------
# Progress bar (simple, no tqdm dep)
# ---------------------------------------------------------------------------

def _progress_bar(current: int, total: int, bar_width: int = 40) -> str:
    if total == 0:
        return "[----------] 0/0"
    frac = current / total
    filled = int(frac * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)
    pct = frac * 100
    return f"[{bar}] {pct:5.1f}% ({current}/{total})"


def _print_progress(symbol: str, current: int, total: int, label: str = "") -> None:
    bar = _progress_bar(current, total)
    label_str = f" | {label}" if label else ""
    line = f"\r{symbol}: {bar}{label_str}    "
    sys.stdout.write(line)
    sys.stdout.flush()
    if current >= total:
        print()  # newline when done


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _checkpoint_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("-", "_")
    return CHECKPOINT_DIR / f"{safe}.json"


def load_checkpoint(symbol: str) -> Optional[Dict]:
    cp = _checkpoint_path(symbol)
    if cp.exists():
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def save_checkpoint(symbol: str, data: Dict) -> None:
    cp = _checkpoint_path(symbol)
    try:
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        log.warning("Could not save checkpoint for %s: %s", symbol, exc)


def delete_checkpoint(symbol: str) -> None:
    cp = _checkpoint_path(symbol)
    try:
        cp.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Date range chunking
# ---------------------------------------------------------------------------

def _date_chunks(
    start_dt: datetime,
    end_dt: datetime,
    chunk_minutes: int,
    bars_per_chunk: int,
) -> Generator[Tuple[datetime, datetime], None, None]:
    """Yield (chunk_start, chunk_end) pairs covering [start_dt, end_dt]."""
    delta = timedelta(minutes=chunk_minutes * bars_per_chunk)
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + delta, end_dt)
        yield current, chunk_end
        current = chunk_end


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _is_crypto(symbol: str) -> bool:
    base = symbol.upper().split("-")[0].split("/")[0]
    return base in CRYPTO_SYMBOLS or "-USD" in symbol.upper() or "/USDT" in symbol.upper()


# ---------------------------------------------------------------------------
# DataBackfiller
# ---------------------------------------------------------------------------

class DataBackfiller:
    """
    Fetches historical OHLCV data and stores it via the SRFM coordinator.

    Source routing:
    - Equities (no "-USD" suffix)  --> Alpaca Data API
    - Crypto                       --> Binance klines, with CoinGecko fallback
    """

    def __init__(self, base_url: str = COORDINATOR_BASE) -> None:
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backfill_symbol(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "15m",
    ) -> bool:
        """
        Backfill OHLCV data for one symbol from start to end (ISO date strings).
        Returns True if the backfill completed successfully.
        """
        symbol = symbol.upper()
        log.info("Starting backfill: %s | %s -> %s | tf=%s", symbol, start, end, timeframe)

        if timeframe not in TIMEFRAME_MINUTES:
            log.error("Unknown timeframe '%s'. Valid: %s", timeframe, list(TIMEFRAME_MINUTES.keys()))
            return False

        start_dt = _parse_date(start)
        end_dt = _parse_date(end)
        if start_dt >= end_dt:
            log.error("start must be before end")
            return False

        tf_minutes = TIMEFRAME_MINUTES[timeframe]
        is_crypto = _is_crypto(symbol)

        # Estimate total chunks for progress display
        total_minutes = int((end_dt - start_dt).total_seconds() / 60)
        bars_per_req = BINANCE_MAX_BARS_PER_REQUEST if is_crypto else ALPACA_MAX_BARS_PER_REQUEST
        total_chunks = max(1, math.ceil(total_minutes / (tf_minutes * bars_per_req)))
        chunk_count = 0
        bars_stored = 0

        chunks = list(_date_chunks(start_dt, end_dt, tf_minutes, bars_per_req))
        total_chunks = len(chunks)

        for chunk_start, chunk_end in chunks:
            chunk_count += 1
            _print_progress(
                symbol, chunk_count, total_chunks,
                f"{chunk_start.date()} -> {chunk_end.date()} | bars stored: {bars_stored}",
            )

            # Save checkpoint
            save_checkpoint(symbol, {
                "symbol": symbol,
                "start": start,
                "end": end,
                "timeframe": timeframe,
                "last_completed_chunk_start": chunk_start.isoformat(),
                "bars_stored": bars_stored,
            })

            # Fetch bars
            if is_crypto:
                bars = self._fetch_binance(symbol, chunk_start, chunk_end, timeframe)
            else:
                bars = self._fetch_alpaca(symbol, chunk_start, chunk_end, timeframe)

            if bars is None:
                log.error("Fetch failed for chunk %s -> %s", chunk_start.date(), chunk_end.date())
                return False

            if not bars:
                log.debug("No bars returned for chunk %s -> %s (market closed?)", chunk_start.date(), chunk_end.date())
                time.sleep(RATE_LIMIT_SLEEP)
                continue

            # Store to coordinator
            stored = self._store_bars(symbol, timeframe, bars)
            if not stored:
                log.error("Failed to store bars for chunk %s -> %s", chunk_start.date(), chunk_end.date())
                return False

            bars_stored += len(bars)
            time.sleep(RATE_LIMIT_SLEEP)

        delete_checkpoint(symbol)
        log.info("Backfill complete for %s: %d bars stored.", symbol, bars_stored)
        return True

    def resume_from_checkpoint(self, symbol: str) -> bool:
        """Resume an interrupted backfill using the saved checkpoint."""
        symbol = symbol.upper()
        cp = load_checkpoint(symbol)
        if not cp:
            log.error("No checkpoint found for %s", symbol)
            return False

        last_start = cp.get("last_completed_chunk_start")
        start_str = cp.get("start")
        end_str = cp.get("end")
        timeframe = cp.get("timeframe", "15m")

        if last_start:
            # Resume from just after last completed chunk
            resume_dt = datetime.fromisoformat(last_start)
            resume_str = resume_dt.strftime("%Y-%m-%d")
            log.info(
                "Resuming backfill for %s from %s (original start: %s)",
                symbol, resume_str, start_str,
            )
            return self.backfill_symbol(symbol, resume_str, end_str, timeframe)
        else:
            log.info("Checkpoint found but no progress recorded -- starting fresh for %s", symbol)
            return self.backfill_symbol(symbol, start_str, end_str, timeframe)

    def verify_completeness(self, symbol: str, start: str, end: str, timeframe: str = "15m") -> Dict:
        """
        Check for gaps in stored data. Returns a dict with:
          - total_expected: expected bar count (approximate, excluding weekends for equities)
          - total_found: bars found in DB
          - gaps: list of (gap_start, gap_end) ISO strings
          - complete: bool
        """
        symbol = symbol.upper()
        log.info("Verifying completeness for %s %s -> %s tf=%s...", symbol, start, end, timeframe)

        payload = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "timeframe": timeframe,
        }
        status, resp = http_post_json(f"{self.base_url}/data/verify_gaps", payload)
        if status == 200 and resp:
            gaps = resp.get("gaps", [])
            total_expected = resp.get("total_expected", 0)
            total_found = resp.get("total_found", 0)
            complete = len(gaps) == 0
            log.info(
                "Verify result: expected=%d found=%d gaps=%d complete=%s",
                total_expected, total_found, len(gaps), complete,
            )
            if gaps:
                log.warning("Gaps detected:")
                for g in gaps[:20]:  # show first 20
                    log.warning("  %s -> %s", g.get("start"), g.get("end"))
                if len(gaps) > 20:
                    log.warning("  ... and %d more gaps", len(gaps) - 20)
            return {
                "total_expected": total_expected,
                "total_found": total_found,
                "gaps": gaps,
                "complete": complete,
            }
        else:
            log.warning("Gap verification endpoint returned HTTP %d", status)
            return {
                "total_expected": 0,
                "total_found": 0,
                "gaps": [],
                "complete": False,
                "error": f"HTTP {status}",
            }

    # ------------------------------------------------------------------
    # Private: data fetching
    # ------------------------------------------------------------------

    def _fetch_alpaca(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        timeframe: str,
    ) -> Optional[List[Dict]]:
        """Fetch bars from Alpaca Data API."""
        if not ALPACA_KEY:
            log.warning("Alpaca keys not set -- cannot fetch equity data")
            return []

        # Map SRFM timeframe to Alpaca format
        tf_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", "1h": "1Hour", "4h": "4Hour", "1d": "1Day"}
        alpaca_tf = tf_map.get(timeframe, "15Min")

        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (
            f"{ALPACA_BASE}/v2/stocks/{symbol}/bars"
            f"?timeframe={alpaca_tf}&start={start_str}&end={end_str}&limit={ALPACA_MAX_BARS_PER_REQUEST}"
        )
        headers = {
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }
        status, data = http_get_json(url, headers=headers)
        if status == 200 and data:
            raw_bars = data.get("bars", [])
            return self._normalize_alpaca_bars(raw_bars, symbol)
        elif status == 429:
            log.warning("Alpaca rate limit hit -- sleeping 5s")
            time.sleep(5)
            return None
        elif status == 422:
            log.warning("Alpaca rejected request (422) for %s %s-%s", symbol, start_str, end_str)
            return []
        log.error("Alpaca bars API returned HTTP %d for %s", status, symbol)
        return None

    def _normalize_alpaca_bars(self, raw: List[Dict], symbol: str) -> List[Dict]:
        bars = []
        for b in raw:
            bars.append({
                "symbol": symbol,
                "ts": b.get("t"),
                "open": b.get("o"),
                "high": b.get("h"),
                "low": b.get("l"),
                "close": b.get("c"),
                "volume": b.get("v"),
                "vwap": b.get("vw"),
            })
        return bars

    def _fetch_binance(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        timeframe: str,
    ) -> Optional[List[Dict]]:
        """Fetch klines from Binance. Falls back to CoinGecko for longer timeframes."""
        # Convert symbol: BTC-USD -> BTCUSDT
        binance_sym = symbol.replace("-USD", "USDT").replace("/", "").upper()
        tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}
        binance_tf = tf_map.get(timeframe, "15m")

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        url = (
            f"{BINANCE_BASE}/api/v3/klines"
            f"?symbol={binance_sym}&interval={binance_tf}"
            f"&startTime={start_ms}&endTime={end_ms}&limit={BINANCE_MAX_BARS_PER_REQUEST}"
        )
        status, data = http_get_json(url)
        if status == 200 and data is not None:
            return self._normalize_binance_bars(data, symbol)
        elif status == 429 or status == 418:
            log.warning("Binance rate limit hit -- sleeping 10s")
            time.sleep(10)
            return None
        elif status == 400:
            # Symbol might not exist -- try CoinGecko as fallback for daily data
            if timeframe == "1d":
                log.info("Binance returned 400 for %s -- trying CoinGecko fallback", symbol)
                return self._fetch_coingecko(symbol, start_dt, end_dt)
            log.warning("Binance returned 400 for %s %s", symbol, binance_sym)
            return []
        log.error("Binance klines returned HTTP %d for %s", status, symbol)
        return None

    def _normalize_binance_bars(self, raw: List, symbol: str) -> List[Dict]:
        bars = []
        for b in raw:
            if not isinstance(b, list) or len(b) < 6:
                continue
            ts_ms = b[0]
            ts_str = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")
            bars.append({
                "symbol": symbol,
                "ts": ts_str,
                "open": float(b[1]),
                "high": float(b[2]),
                "low": float(b[3]),
                "close": float(b[4]),
                "volume": float(b[5]),
                "vwap": None,
            })
        return bars

    def _fetch_coingecko(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> Optional[List[Dict]]:
        """Fetch daily OHLCV from CoinGecko (free tier)."""
        # Map symbol to CoinGecko coin ID (simple lookup)
        cg_map = {
            "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
            "SOL": "solana", "ADA": "cardano", "AVAX": "avalanche-2",
            "MATIC": "matic-network", "DOGE": "dogecoin", "XRP": "ripple",
            "DOT": "polkadot",
        }
        base = symbol.upper().split("-")[0].split("/")[0]
        coin_id = cg_map.get(base)
        if not coin_id:
            log.warning("No CoinGecko mapping for %s -- skipping", symbol)
            return []

        days = max(1, (end_dt - start_dt).days)
        days = min(days, COINGECKO_MAX_DAYS)
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
        status, data = http_get_json(url)
        if status == 200 and data:
            return self._normalize_coingecko_bars(data, symbol)
        log.error("CoinGecko returned HTTP %d for %s", status, symbol)
        return None

    def _normalize_coingecko_bars(self, raw: List, symbol: str) -> List[Dict]:
        bars = []
        for b in raw:
            if not isinstance(b, list) or len(b) < 5:
                continue
            ts_ms = b[0]
            ts_str = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")
            bars.append({
                "symbol": symbol,
                "ts": ts_str,
                "open": float(b[1]),
                "high": float(b[2]),
                "low": float(b[3]),
                "close": float(b[4]),
                "volume": None,
                "vwap": None,
            })
        return bars

    # ------------------------------------------------------------------
    # Private: storage
    # ------------------------------------------------------------------

    def _store_bars(self, symbol: str, timeframe: str, bars: List[Dict]) -> bool:
        """POST bars to the coordinator data store endpoint."""
        if not bars:
            return True
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": bars,
        }
        status, _ = http_post_json(f"{self.base_url}/data/bars", payload)
        if status in (200, 201, 204):
            return True
        log.error("Data store returned HTTP %d for %d bars of %s", status, len(bars), symbol)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRFM historical data backfill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to backfill (e.g. BTC-USD, AAPL)")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (required unless --resume)")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--timeframe", type=str, default="15m",
                        choices=list(TIMEFRAME_MINUTES.keys()),
                        help="Bar timeframe (default: 15m)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint instead of starting fresh")
    parser.add_argument("--verify", action="store_true",
                        help="Verify completeness of existing data (no new fetch)")
    parser.add_argument("--coordinator-url", type=str, default=COORDINATOR_BASE,
                        help=f"Coordinator base URL (default: {COORDINATOR_BASE})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbol = args.symbol.upper()
    end_str = args.end or date.today().isoformat()

    backfiller = DataBackfiller(base_url=args.coordinator_url)

    if args.verify:
        if not args.start:
            print("Error: --start is required with --verify", file=sys.stderr)
            sys.exit(2)
        result = backfiller.verify_completeness(symbol, args.start, end_str, args.timeframe)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("complete") else 1)

    if args.resume:
        ok = backfiller.resume_from_checkpoint(symbol)
    else:
        if not args.start:
            print("Error: --start is required (or use --resume)", file=sys.stderr)
            sys.exit(2)
        ok = backfiller.backfill_symbol(symbol, args.start, end_str, args.timeframe)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
