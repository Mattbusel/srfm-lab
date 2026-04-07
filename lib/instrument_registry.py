"""
lib/instrument_registry.py
===========================
Registry of all tradeable instruments with per-instrument metadata,
CF calibration, and order validation.

Loads from config/instruments.yaml (the canonical source of truth for the
instrument universe). Supports hot-reload without restarting the live trader.

Key types:
    Instrument          -- per-instrument dataclass
    InstrumentRegistry  -- registry with lookup, filtering, and validation

Usage:
    from lib.instrument_registry import InstrumentRegistry

    reg = InstrumentRegistry()
    btc = reg.get("BTC")
    cryptos = reg.get_crypto()
    ok, msg = reg.validate_order("BTC", qty=0.001, price=65000.0)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("instrument_registry")

_REPO_ROOT        = Path(__file__).parents[1]
_INSTRUMENTS_PATH = _REPO_ROOT / "config" / "instruments.yaml"


# ---------------------------------------------------------------------------
# Instrument dataclass
# ---------------------------------------------------------------------------

@dataclass
class CfCalibration:
    """Per-timeframe centrifugal force thresholds for one instrument."""
    cf_15m: float = 0.005
    cf_1h:  float = 0.015
    cf_4h:  float = 0.015   # derived from cf_1h * 1.5 if not present in YAML
    cf_1d:  float = 0.050
    bh_form:     float = 2.0
    bh_collapse: float = 0.992
    bh_decay:    float = 0.924
    min_hold_bars: int = 6


@dataclass
class Instrument:
    """
    Full metadata record for a single tradeable instrument.

    Fields:
        symbol          -- canonical short symbol (BTC, SPY, ES, etc.)
        name            -- human-readable name
        asset_class     -- "crypto" | "equity" | "equity_index" | "commodity" | "bond" | "volatility" | "forex"
        base_currency   -- base currency (BTC for BTC/USD)
        quote_currency  -- quote currency (USD)
        alpaca_ticker   -- ticker string used on Alpaca (e.g. "BTC/USD" or "SPY")
        instrument_type -- "crypto" | "stock" | "forex"
        tick_size       -- minimum price increment
        lot_size        -- minimum quantity increment (1.0 for stocks, fractional for crypto)
        min_qty         -- minimum order quantity
        qty_precision   -- decimal places for quantity
        price_precision -- decimal places for price
        max_position_usd -- maximum notional USD value in any single position
        margin_rate     -- fraction of notional required as margin (0 = no margin product)
        corr_group      -- correlation group name (equity_us, crypto_large, etc.)
        cf_calibration  -- per-timeframe CF and BH calibration
        is_active       -- whether this instrument is currently traded
        disabled_reason -- reason for disabling (if is_active = False)
        disabled_at     -- timestamp of disabling
        exchange        -- exchange string ("alpaca", "ibkr", etc.)
        options_overlay -- whether to trade ATM options overlay
        notes           -- free-form notes from YAML
    """
    symbol:          str
    name:            str
    asset_class:     str
    base_currency:   str
    quote_currency:  str
    alpaca_ticker:   Optional[str]
    instrument_type: str
    tick_size:       float
    lot_size:        float        = 1.0
    min_qty:         float        = 0.0
    qty_precision:   int          = 2
    price_precision: int          = 2
    max_position_usd: float       = 500_000.0
    margin_rate:     float        = 0.0
    corr_group:      str          = "unknown"
    cf_calibration:  CfCalibration = field(default_factory=CfCalibration)
    is_active:       bool         = True
    disabled_reason: Optional[str]= None
    disabled_at:     Optional[datetime] = None
    exchange:        str          = "alpaca"
    options_overlay: bool         = False
    notes:           str          = ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def round_qty(self, qty: float) -> float:
        """Round quantity to instrument precision."""
        factor = 10 ** self.qty_precision
        return round(qty * factor) / factor

    def round_price(self, price: float) -> float:
        """Round price to instrument tick size."""
        if self.tick_size <= 0:
            return price
        return round(price / self.tick_size) * self.tick_size

    def effective_min_qty(self) -> float:
        """Effective minimum quantity -- max of lot_size and min_qty."""
        return max(self.lot_size, self.min_qty)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.disabled_at:
            d["disabled_at"] = self.disabled_at.isoformat()
        return d


# ---------------------------------------------------------------------------
# YAML loading helper
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML from path; returns empty dict if file missing."""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        log.warning("instrument_registry: YAML not found at %s", path)
        return {}
    except ImportError:
        log.error("instrument_registry: PyYAML not installed -- pip install pyyaml")
        return {}


def _infer_qty_precision(tick_size: float) -> int:
    """Infer decimal precision from tick size (e.g. 0.001 -> 3)."""
    if tick_size <= 0:
        return 2
    s = f"{tick_size:.10f}".rstrip("0")
    if "." in s:
        return len(s.split(".")[1])
    return 0


def _infer_price_precision(tick_size: float) -> int:
    return _infer_qty_precision(tick_size)


def _parse_instrument(symbol: str, data: dict[str, Any]) -> Instrument:
    """Parse one instrument entry from YAML into an Instrument dataclass."""
    tick_size       = float(data.get("tick_size", 0.01))
    asset_class     = str(data.get("asset_class", "unknown"))
    price_precision = _infer_price_precision(tick_size)

    # lot_size: for crypto default to tick_size (fractional trading), else 1 share
    if "lot_size" in data:
        lot_size = float(data["lot_size"])
    elif asset_class == "crypto":
        lot_size = tick_size if tick_size < 1.0 else 0.0001
    else:
        lot_size = 1.0

    qty_precision = _infer_qty_precision(lot_size if lot_size > 0 else tick_size)

    # min_qty defaults to lot_size
    default_min_qty = lot_size if lot_size > 0 else tick_size

    cf = CfCalibration(
        cf_15m        = float(data.get("cf_15m", 0.005)),
        cf_1h         = float(data.get("cf_1h",  0.015)),
        cf_4h         = float(data.get("cf_4h",  float(data.get("cf_1h", 0.015)) * 1.5)),
        cf_1d         = float(data.get("cf_1d",  0.050)),
        bh_form       = float(data.get("bh_form",     2.0)),
        bh_collapse   = float(data.get("bh_collapse", 0.992)),
        bh_decay      = float(data.get("bh_decay",    0.924)),
        min_hold_bars = int(data.get("min_hold_bars", 6)),
    )

    # Asset-class specific position cap defaults
    default_cap = {
        "crypto":       500_000.0,
        "equity":       300_000.0,
        "equity_index": 300_000.0,
        "commodity":    200_000.0,
        "volatility":   150_000.0,
        "bond":         400_000.0,
        "forex":        100_000.0,
    }.get(asset_class, 250_000.0)

    return Instrument(
        symbol          = symbol,
        name            = str(data.get("name", symbol)),
        asset_class     = asset_class,
        base_currency   = str(data.get("base_currency", "USD")),
        quote_currency  = str(data.get("quote_currency", "USD")),
        alpaca_ticker   = data.get("alpaca_ticker"),
        instrument_type = str(data.get("type", "stock")),
        tick_size       = tick_size,
        lot_size        = lot_size,
        min_qty         = float(data.get("min_qty", default_min_qty)),
        qty_precision   = qty_precision,
        price_precision = price_precision,
        max_position_usd= float(data.get("max_position_usd", default_cap)),
        margin_rate     = float(data.get("margin_rate", 0.0)),
        corr_group      = str(data.get("corr_group", "unknown")),
        cf_calibration  = cf,
        is_active       = bool(data.get("is_active", True)),
        exchange        = str(data.get("exchange", "alpaca")),
        options_overlay = bool(data.get("options_overlay", False)),
        notes           = str(data.get("notes", "")),
    )


# ---------------------------------------------------------------------------
# InstrumentRegistry
# ---------------------------------------------------------------------------

class InstrumentRegistry:
    """
    Thread-safe registry of all tradeable instruments.

    Loads from config/instruments.yaml and supports hot-reload.
    All read operations are safe to call from multiple threads.
    """

    def __init__(
        self,
        yaml_path: Optional[Path] = None,
        watch: bool = True,
    ) -> None:
        self._path      = Path(yaml_path) if yaml_path else _INSTRUMENTS_PATH
        self._lock      = threading.RLock()
        self._instruments: dict[str, Instrument] = {}
        self._corr_groups: dict[str, Any]         = {}
        self._last_mtime: float                   = 0.0
        self._disabled_log: list[dict[str, Any]]  = []   # audit trail
        self._observer  = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._load()

        if watch:
            self._start_watch()

    # ------------------------------------------------------------------
    # Public API -- reads
    # ------------------------------------------------------------------

    def get(self, symbol: str) -> Instrument:
        """
        Return Instrument for symbol. Raises KeyError if not found.
        Symbol lookup is case-insensitive.
        """
        with self._lock:
            inst = self._instruments.get(symbol.upper())
            if inst is None:
                raise KeyError(f"instrument_registry: unknown symbol '{symbol}'")
            return inst

    def get_or_none(self, symbol: str) -> Optional[Instrument]:
        """Return Instrument or None if not found."""
        with self._lock:
            return self._instruments.get(symbol.upper())

    def get_active(self) -> list[Instrument]:
        """Return all active instruments."""
        with self._lock:
            return [i for i in self._instruments.values() if i.is_active]

    def get_all(self) -> list[Instrument]:
        """Return all instruments regardless of active state."""
        with self._lock:
            return list(self._instruments.values())

    def get_by_class(self, asset_class: str) -> list[Instrument]:
        """Return active instruments of the given asset class."""
        with self._lock:
            return [
                i for i in self._instruments.values()
                if i.asset_class == asset_class and i.is_active
            ]

    def get_crypto(self) -> list[Instrument]:
        """Return all active crypto instruments."""
        return self.get_by_class("crypto")

    def get_equity(self) -> list[Instrument]:
        """Return all active equity / equity_index instruments."""
        with self._lock:
            return [
                i for i in self._instruments.values()
                if i.asset_class in ("equity", "equity_index") and i.is_active
            ]

    def get_by_corr_group(self, group: str) -> list[Instrument]:
        """Return active instruments in the given correlation group."""
        with self._lock:
            return [
                i for i in self._instruments.values()
                if i.corr_group == group and i.is_active
            ]

    def get_cf_calibration(self, symbol: str) -> dict[str, Any]:
        """
        Return CF calibration dict for symbol.
        Raises KeyError if symbol unknown.
        """
        inst = self.get(symbol)
        cf   = inst.cf_calibration
        return {
            "cf_15m":         cf.cf_15m,
            "cf_1h":          cf.cf_1h,
            "cf_4h":          cf.cf_4h,
            "cf_1d":          cf.cf_1d,
            "bh_form":        cf.bh_form,
            "bh_collapse":    cf.bh_collapse,
            "bh_decay":       cf.bh_decay,
            "min_hold_bars":  cf.min_hold_bars,
        }

    def get_correlation_groups(self) -> dict[str, Any]:
        """Return the correlation groups metadata from YAML."""
        with self._lock:
            return dict(self._corr_groups)

    def symbols(self) -> list[str]:
        """Return sorted list of all known symbols."""
        with self._lock:
            return sorted(self._instruments.keys())

    def active_symbols(self) -> list[str]:
        """Return sorted list of active symbol strings."""
        with self._lock:
            return sorted(s for s, i in self._instruments.items() if i.is_active)

    # ------------------------------------------------------------------
    # Public API -- mutations
    # ------------------------------------------------------------------

    def add_instrument(self, instrument: Instrument) -> None:
        """
        Add or replace an instrument in the registry.
        Validates the instrument before inserting.
        Raises ValueError on invalid data.
        """
        self._validate_instrument(instrument)
        with self._lock:
            self._instruments[instrument.symbol.upper()] = instrument
        log.info("instrument_registry: added/updated %s", instrument.symbol)

    def disable_instrument(self, symbol: str, reason: str = "") -> None:
        """
        Mark instrument as inactive. Logs the disable event.
        Raises KeyError if symbol unknown.
        """
        with self._lock:
            sym = symbol.upper()
            inst = self._instruments.get(sym)
            if inst is None:
                raise KeyError(f"instrument_registry: cannot disable unknown symbol '{symbol}'")
            inst.is_active       = False
            inst.disabled_reason = reason
            inst.disabled_at     = datetime.now(timezone.utc)
        self._disabled_log.append({
            "symbol":    sym,
            "reason":    reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action":    "disable",
        })
        log.warning("instrument_registry: disabled %s -- reason: %s", symbol, reason)

    def enable_instrument(self, symbol: str) -> None:
        """Re-enable a previously disabled instrument."""
        with self._lock:
            sym  = symbol.upper()
            inst = self._instruments.get(sym)
            if inst is None:
                raise KeyError(f"instrument_registry: unknown symbol '{symbol}'")
            inst.is_active       = True
            inst.disabled_reason = None
            inst.disabled_at     = None
        self._disabled_log.append({
            "symbol":    sym,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action":    "enable",
        })
        log.info("instrument_registry: enabled %s", symbol)

    def hot_reload(self) -> bool:
        """
        Reload instrument definitions from YAML without restarting.
        Active/disabled state changes made at runtime are preserved for
        symbols already known; newly added symbols get their YAML state.

        Returns True if reload succeeded.
        """
        try:
            raw = _load_yaml(self._path)
        except Exception as exc:
            log.error("instrument_registry: hot_reload read error -- %s", exc)
            return False

        instruments_raw = raw.get("instruments", {})
        if not instruments_raw:
            log.warning("instrument_registry: hot_reload -- no instruments found in YAML")
            return False

        new_registry: dict[str, Instrument] = {}
        for sym, data in instruments_raw.items():
            sym_upper = sym.upper()
            try:
                new_inst = _parse_instrument(sym_upper, data)
            except Exception as exc:
                log.error("instrument_registry: parse error for %s -- %s", sym, exc)
                continue
            # Preserve runtime active/disable state
            with self._lock:
                existing = self._instruments.get(sym_upper)
                if existing is not None and not existing.is_active:
                    new_inst.is_active       = False
                    new_inst.disabled_reason = existing.disabled_reason
                    new_inst.disabled_at     = existing.disabled_at
            new_registry[sym_upper] = new_inst

        with self._lock:
            self._instruments   = new_registry
            self._corr_groups   = raw.get("correlation_groups", {})

        log.info(
            "instrument_registry: hot_reload complete -- %d instruments (%d active)",
            len(new_registry),
            sum(1 for i in new_registry.values() if i.is_active),
        )
        return True

    # ------------------------------------------------------------------
    # Order validation
    # ------------------------------------------------------------------

    def validate_order(
        self,
        symbol:    str,
        qty:       float,
        price:     float,
        notional:  Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Validate a proposed order.

        Checks:
          - Symbol is known and active
          - qty >= effective_min_qty
          - qty is a multiple of lot_size (within floating point tolerance)
          - notional (qty * price) <= max_position_usd
          - price > 0

        Returns (True, "") if valid, (False, error_message) otherwise.
        """
        inst = self.get_or_none(symbol)
        if inst is None:
            return False, f"unknown symbol '{symbol}'"
        if not inst.is_active:
            return False, f"symbol '{symbol}' is disabled: {inst.disabled_reason}"
        if price <= 0:
            return False, f"price must be > 0, got {price}"
        if qty <= 0:
            return False, f"qty must be > 0, got {qty}"

        min_q = inst.effective_min_qty()
        if min_q > 0 and qty < min_q * 0.9999:  # 0.01% tolerance for float repr
            return False, (
                f"qty {qty} < min_qty {min_q} for {symbol}"
            )

        if inst.lot_size > 0 and inst.lot_size < 1.0:
            # Crypto fractional: just check qty precision
            rounded = inst.round_qty(qty)
            if abs(rounded - qty) > inst.lot_size * 0.01:
                return False, (
                    f"qty {qty} not aligned to lot_size {inst.lot_size} for {symbol}"
                )

        notional_val = notional if notional is not None else qty * price
        if notional_val > inst.max_position_usd:
            return False, (
                f"notional {notional_val:.0f} > max_position_usd {inst.max_position_usd:.0f} for {symbol}"
            )

        return True, ""

    def get_disabled_log(self) -> list[dict[str, Any]]:
        """Return audit log of disable/enable events."""
        return list(self._disabled_log)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Initial load from YAML."""
        raw = _load_yaml(self._path)
        instruments_raw = raw.get("instruments", {})
        registry: dict[str, Instrument] = {}
        for sym, data in instruments_raw.items():
            sym_upper = sym.upper()
            try:
                registry[sym_upper] = _parse_instrument(sym_upper, data)
            except Exception as exc:
                log.error("instrument_registry: failed to parse '%s' -- %s", sym, exc)

        with self._lock:
            self._instruments = registry
            self._corr_groups = raw.get("correlation_groups", {})
            try:
                self._last_mtime = self._path.stat().st_mtime
            except FileNotFoundError:
                self._last_mtime = 0.0

        log.info(
            "instrument_registry: loaded %d instruments (%d active) from %s",
            len(registry),
            sum(1 for i in registry.values() if i.is_active),
            self._path,
        )

    @staticmethod
    def _validate_instrument(inst: Instrument) -> None:
        """Raise ValueError if instrument data is invalid."""
        if not inst.symbol:
            raise ValueError("instrument symbol cannot be empty")
        if inst.tick_size < 0:
            raise ValueError(f"tick_size must be >= 0, got {inst.tick_size} for {inst.symbol}")
        if inst.lot_size < 0:
            raise ValueError(f"lot_size must be >= 0, got {inst.lot_size} for {inst.symbol}")
        if inst.max_position_usd <= 0:
            raise ValueError(f"max_position_usd must be > 0 for {inst.symbol}")
        if inst.cf_calibration.bh_form <= 0:
            raise ValueError(f"bh_form must be > 0 for {inst.symbol}")

    def _start_watch(self) -> None:
        """Start watchdog or polling."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            registry_ref = self

            class _Handler(FileSystemEventHandler):
                def on_modified(self, event):
                    if not event.is_directory and Path(event.src_path).resolve() == registry_ref._path.resolve():
                        time.sleep(0.05)
                        registry_ref.hot_reload()

                def on_created(self, event):
                    self.on_modified(event)

            observer = Observer()
            observer.schedule(_Handler(), str(self._path.parent), recursive=False)
            observer.daemon = True
            observer.start()
            self._observer = observer
            log.info("instrument_registry: watchdog monitoring %s", self._path)
        except ImportError:
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(30):
            try:
                mtime = self._path.stat().st_mtime
                if mtime > self._last_mtime:
                    self._last_mtime = mtime
                    self.hot_reload()
            except Exception as exc:
                log.debug("instrument_registry: poll error -- %s", exc)

    def stop(self) -> None:
        self._stop_event.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=3)
            except Exception:
                pass
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
