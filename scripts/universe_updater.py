#!/usr/bin/env python3
"""
universe_updater.py -- Update the SRFM trading universe.

Reads and writes config/instruments.yaml. After any modification,
triggers a hot-reload on the live trader via the coordinator API.
Always creates a backup of instruments.yaml before modifying.

Usage:
  python scripts/universe_updater.py add BTC-USD --class crypto --sector defi
  python scripts/universe_updater.py remove BTC-USD --reason "low-volume"
  python scripts/universe_updater.py adv BTC-USD --adv 5000000000
  python scripts/universe_updater.py cf BTC-USD --cf-long 0.15 --cf-short 0.10
  python scripts/universe_updater.py validate
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
INSTRUMENTS_FILE = CONFIG_DIR / "instruments.yaml"
BACKUPS_DIR = REPO_ROOT / "backups" / "instruments"
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

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
    return logger


log = build_logger("srfm.universe_updater")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COORDINATOR_BASE = os.environ.get("SRFM_COORDINATOR_URL", "http://localhost:8000")
HTTP_TIMEOUT = 10

# Required fields for each instrument entry
REQUIRED_INSTRUMENT_FIELDS = [
    "symbol",
    "asset_class",
    "sector",
    "enabled",
    "adv",
]

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def http_post(url: str, payload: Dict) -> Tuple[int, bytes]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        log.debug("http_post %s: %s", url, exc)
        return 0, b""


# ---------------------------------------------------------------------------
# YAML I/O (uses PyYAML with a plain-text fallback for reading)
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict:
    """Load instruments YAML. Returns dict with 'universe' list."""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return {"universe": []}
        if "universe" not in data:
            data["universe"] = []
        return data
    except ImportError:
        pass
    # Minimal fallback: return empty-ish structure
    log.warning("PyYAML not available -- using minimal fallback (add-only operations supported)")
    return {"universe": [], "_raw_path": str(path)}


def _dump_yaml(data: Dict, path: Path) -> None:
    """Write instruments YAML."""
    try:
        import yaml  # type: ignore
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return
    except ImportError:
        pass
    # Minimal fallback: JSON-in-YAML (valid YAML)
    log.warning("PyYAML not available -- writing as JSON-compatible YAML")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Backup helper
# ---------------------------------------------------------------------------

def _backup_instruments(src: Path) -> Path:
    """Copy instruments.yaml to backups/instruments/ with timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUPS_DIR / f"instruments_{ts}.yaml"
    try:
        shutil.copy2(str(src), str(dst))
        log.info("Backup created: %s", dst)
    except OSError as exc:
        log.warning("Could not create backup: %s", exc)
    return dst


# ---------------------------------------------------------------------------
# Hot-reload trigger
# ---------------------------------------------------------------------------

def _trigger_hot_reload(base_url: str) -> bool:
    log.info("Triggering live trader hot-reload...")
    status, body = http_post(
        f"{base_url}/universe/reload",
        {"timestamp": datetime.utcnow().isoformat()},
    )
    if status in (200, 204):
        log.info("Hot-reload accepted by live trader.")
        return True
    log.warning("Hot-reload returned HTTP %d -- live trader may need manual restart", status)
    return False


# ---------------------------------------------------------------------------
# UniverseUpdater
# ---------------------------------------------------------------------------

class UniverseUpdater:
    """
    Manages the SRFM trading universe in config/instruments.yaml.
    All modifications are preceded by a backup and followed by a hot-reload.
    """

    def __init__(
        self,
        instruments_file: Path = INSTRUMENTS_FILE,
        base_url: str = COORDINATOR_BASE,
    ) -> None:
        self.instruments_file = instruments_file
        self.base_url = base_url

    def _load(self) -> Dict:
        if not self.instruments_file.exists():
            log.warning("instruments.yaml not found -- starting with empty universe")
            return {"universe": []}
        return _load_yaml(self.instruments_file)

    def _save(self, data: Dict) -> None:
        self.instruments_file.parent.mkdir(parents=True, exist_ok=True)
        if self.instruments_file.exists():
            _backup_instruments(self.instruments_file)
        _dump_yaml(data, self.instruments_file)
        log.info("instruments.yaml saved.")

    def _find_symbol(self, data: Dict, symbol: str) -> Tuple[int, Optional[Dict]]:
        """Return (index, entry) for a symbol, or (-1, None) if not found."""
        universe: List[Dict] = data.get("universe", [])
        sym_upper = symbol.upper()
        for idx, entry in enumerate(universe):
            if entry.get("symbol", "").upper() == sym_upper:
                return idx, entry
        return -1, None

    # ------------------------------------------------------------------
    # add_symbol
    # ------------------------------------------------------------------

    def add_symbol(
        self,
        symbol: str,
        asset_class: str,
        sector: str,
        adv: float = 0.0,
        cf_long: float = 0.10,
        cf_short: float = 0.10,
    ) -> bool:
        """Add a new symbol to instruments.yaml."""
        symbol = symbol.upper()
        data = self._load()
        idx, existing = self._find_symbol(data, symbol)

        if existing is not None:
            if existing.get("enabled", True):
                log.warning("Symbol %s already exists and is enabled -- use update commands", symbol)
                return False
            else:
                log.info("Re-enabling existing disabled symbol %s", symbol)
                existing["enabled"] = True
                existing["asset_class"] = asset_class
                existing["sector"] = sector
                if adv:
                    existing["adv"] = adv
                existing["cf_long"] = cf_long
                existing["cf_short"] = cf_short
                existing["updated_at"] = datetime.utcnow().isoformat()
                data["universe"][idx] = existing
        else:
            entry: Dict[str, Any] = {
                "symbol": symbol,
                "asset_class": asset_class,
                "sector": sector,
                "enabled": True,
                "adv": adv,
                "cf_long": cf_long,
                "cf_short": cf_short,
                "added_at": datetime.utcnow().isoformat(),
            }
            data["universe"].append(entry)
            log.info("Added symbol %s (class=%s sector=%s)", symbol, asset_class, sector)

        self._save(data)
        _trigger_hot_reload(self.base_url)
        return True

    # ------------------------------------------------------------------
    # remove_symbol
    # ------------------------------------------------------------------

    def remove_symbol(self, symbol: str, reason: str) -> bool:
        """Disable a symbol (marks enabled=False, records reason)."""
        symbol = symbol.upper()
        data = self._load()
        idx, entry = self._find_symbol(data, symbol)

        if entry is None:
            log.error("Symbol %s not found in universe", symbol)
            return False

        entry["enabled"] = False
        entry["disabled_reason"] = reason
        entry["disabled_at"] = datetime.utcnow().isoformat()
        data["universe"][idx] = entry
        log.info("Symbol %s disabled (reason: %s)", symbol, reason)

        self._save(data)
        _trigger_hot_reload(self.base_url)
        return True

    # ------------------------------------------------------------------
    # update_adv
    # ------------------------------------------------------------------

    def update_adv(self, symbol: str, new_adv: float) -> bool:
        """Update the average daily volume for a symbol."""
        symbol = symbol.upper()
        data = self._load()
        idx, entry = self._find_symbol(data, symbol)

        if entry is None:
            log.error("Symbol %s not found in universe", symbol)
            return False

        old_adv = entry.get("adv", 0.0)
        entry["adv"] = new_adv
        entry["adv_updated_at"] = datetime.utcnow().isoformat()
        data["universe"][idx] = entry
        log.info("ADV updated for %s: %.2f -> %.2f", symbol, old_adv, new_adv)

        self._save(data)
        _trigger_hot_reload(self.base_url)
        return True

    # ------------------------------------------------------------------
    # set_cf_parameters
    # ------------------------------------------------------------------

    def set_cf_parameters(self, symbol: str, cf_long: float, cf_short: float) -> bool:
        """Set capacity factor (CF) parameters for a symbol."""
        symbol = symbol.upper()
        data = self._load()
        idx, entry = self._find_symbol(data, symbol)

        if entry is None:
            log.error("Symbol %s not found in universe", symbol)
            return False

        old_long = entry.get("cf_long", None)
        old_short = entry.get("cf_short", None)
        entry["cf_long"] = cf_long
        entry["cf_short"] = cf_short
        entry["cf_updated_at"] = datetime.utcnow().isoformat()
        data["universe"][idx] = entry
        log.info(
            "CF params updated for %s: cf_long %.4f->%.4f cf_short %.4f->%.4f",
            symbol, old_long or 0, cf_long, old_short or 0, cf_short,
        )

        self._save(data)
        _trigger_hot_reload(self.base_url)
        return True

    # ------------------------------------------------------------------
    # validate_universe
    # ------------------------------------------------------------------

    def validate_universe(self) -> List[str]:
        """
        Validate all enabled symbols have required fields.
        Returns a list of error strings (empty list = valid).
        """
        data = self._load()
        universe = data.get("universe", [])
        errors: List[str] = []

        if not universe:
            errors.append("Universe is empty -- no symbols configured")
            return errors

        symbols_seen: set = set()
        for entry in universe:
            if not entry.get("enabled", True):
                continue
            sym = entry.get("symbol", "<unknown>")

            # Duplicate check
            if sym in symbols_seen:
                errors.append(f"Duplicate symbol: {sym}")
            symbols_seen.add(sym)

            # Required fields
            for field in REQUIRED_INSTRUMENT_FIELDS:
                if field not in entry:
                    errors.append(f"{sym}: missing required field '{field}'")

            # ADV sanity
            adv = entry.get("adv", 0.0)
            if isinstance(adv, (int, float)) and adv < 0:
                errors.append(f"{sym}: negative ADV ({adv})")

            # CF sanity
            for cf_key in ("cf_long", "cf_short"):
                cf_val = entry.get(cf_key)
                if cf_val is not None:
                    if not (0.0 <= float(cf_val) <= 1.0):
                        errors.append(f"{sym}: {cf_key} out of range [0,1]: {cf_val}")

        if errors:
            log.warning("Universe validation found %d issue(s):", len(errors))
            for err in errors:
                log.warning("  - %s", err)
        else:
            enabled = sum(1 for e in universe if e.get("enabled", True))
            log.info("Universe valid: %d enabled symbol(s) of %d total.", enabled, len(universe))

        return errors

    # ------------------------------------------------------------------
    # list_universe
    # ------------------------------------------------------------------

    def list_universe(self) -> None:
        """Print the current universe to stdout."""
        data = self._load()
        universe = data.get("universe", [])
        if not universe:
            print("Universe is empty.")
            return
        print(f"\n{'Symbol':<20} {'Class':<12} {'Sector':<20} {'Enabled':<8} {'ADV':>15}")
        print("-" * 80)
        for entry in sorted(universe, key=lambda e: e.get("symbol", "")):
            sym = entry.get("symbol", "")
            cls = entry.get("asset_class", "")
            sec = entry.get("sector", "")
            enabled = "YES" if entry.get("enabled", True) else "no"
            adv = entry.get("adv", 0)
            adv_str = f"{adv:>15,.0f}" if isinstance(adv, (int, float)) else str(adv)
            print(f"{sym:<20} {cls:<12} {sec:<20} {enabled:<8} {adv_str}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRFM universe updater",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--coordinator-url",
        type=str,
        default=COORDINATOR_BASE,
        help=f"Coordinator base URL (default: {COORDINATOR_BASE})",
    )

    sub = parser.add_subparsers(dest="command", help="Sub-command")
    sub.required = True

    # add
    add_p = sub.add_parser("add", help="Add a symbol to the universe")
    add_p.add_argument("symbol", type=str)
    add_p.add_argument("--class", dest="asset_class", type=str, required=True)
    add_p.add_argument("--sector", type=str, required=True)
    add_p.add_argument("--adv", type=float, default=0.0, help="Average daily volume")
    add_p.add_argument("--cf-long", type=float, default=0.10)
    add_p.add_argument("--cf-short", type=float, default=0.10)

    # remove
    rm_p = sub.add_parser("remove", help="Disable a symbol")
    rm_p.add_argument("symbol", type=str)
    rm_p.add_argument("--reason", type=str, required=True)

    # adv
    adv_p = sub.add_parser("adv", help="Update average daily volume")
    adv_p.add_argument("symbol", type=str)
    adv_p.add_argument("--adv", type=float, required=True)

    # cf
    cf_p = sub.add_parser("cf", help="Set capacity factor parameters")
    cf_p.add_argument("symbol", type=str)
    cf_p.add_argument("--cf-long", type=float, required=True, dest="cf_long")
    cf_p.add_argument("--cf-short", type=float, required=True, dest="cf_short")

    # validate
    sub.add_parser("validate", help="Validate all instruments in universe")

    # list
    sub.add_parser("list", help="List current universe")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    updater = UniverseUpdater(base_url=args.coordinator_url)
    ok = True

    if args.command == "add":
        ok = updater.add_symbol(
            args.symbol,
            asset_class=args.asset_class,
            sector=args.sector,
            adv=args.adv,
            cf_long=args.cf_long,
            cf_short=args.cf_short,
        )
    elif args.command == "remove":
        ok = updater.remove_symbol(args.symbol, reason=args.reason)
    elif args.command == "adv":
        ok = updater.update_adv(args.symbol, new_adv=args.adv)
    elif args.command == "cf":
        ok = updater.set_cf_parameters(args.symbol, cf_long=args.cf_long, cf_short=args.cf_short)
    elif args.command == "validate":
        errors = updater.validate_universe()
        ok = len(errors) == 0
        if not ok:
            sys.exit(1)
    elif args.command == "list":
        updater.list_universe()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
