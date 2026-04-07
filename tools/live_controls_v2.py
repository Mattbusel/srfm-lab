"""
tools/live_controls_v2.py
==========================
Enhanced live control interface for the running LARSA v18 trader.

This extends tools/live_controls.py with a CLI for hot-reload controls,
emergency actions, and parameter management via the coordination layer.

Usage:
    python tools/live_controls_v2.py --status
    python tools/live_controls_v2.py --propose-params params.json --confirm
    python tools/live_controls_v2.py --override BTC=1.5 --confirm
    python tools/live_controls_v2.py --block-symbol ETH 2 --confirm
    python tools/live_controls_v2.py --flatten-all --confirm
    python tools/live_controls_v2.py --pause 30 --confirm
    python tools/live_controls_v2.py --reset-circuit alpaca --confirm
    python tools/live_controls_v2.py --drain coordination --confirm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import aiohttp

# ── Rich (optional) ───────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH = True
    _console = Console()
except ImportError:
    _RICH = False
    _console = None  # type: ignore

log = logging.getLogger("live_controls_v2")
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Paths / endpoints ─────────────────────────────────────────────────────────
_REPO_ROOT        = Path(__file__).parents[1]
_OVERRIDES_FILE   = _REPO_ROOT / "config" / "signal_overrides.json"
_BLOCKED_FILE     = _REPO_ROOT / "config" / "blocked_symbols.json"
_PAUSE_FILE       = _REPO_ROOT / "config" / "trading_pause.json"
_COORD_URL        = "http://127.0.0.1:8781"
_RISK_URL         = "http://127.0.0.1:8791"
_OBS_URL          = "http://127.0.0.1:9091"

# Destructive action keywords (all require --confirm)
_DESTRUCTIVE_ACTIONS = frozenset([
    "flatten_all",
    "block_symbol",
    "override",
    "pause",
    "reset_circuit",
    "drain",
    "propose_params",
])

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _get(session: aiohttp.ClientSession, url: str, timeout: float = 5.0) -> dict:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                return await resp.json(content_type=None)
            return {"error": f"HTTP {resp.status}", "url": url}
    except Exception as exc:
        return {"error": str(exc), "url": url}


async def _post(session: aiohttp.ClientSession, url: str, data: dict, timeout: float = 10.0) -> dict:
    try:
        async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            body = await resp.json(content_type=None)
            if resp.status in (200, 201, 202):
                return {"ok": True, "response": body}
            return {"ok": False, "status": resp.status, "response": body}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Parameter validation
# ─────────────────────────────────────────────────────────────────────────────

_PARAM_SCHEMA: dict[str, tuple[type, float, float]] = {
    "cf_15m":             (float, 0.001, 0.10),
    "cf_1h":              (float, 0.001, 0.10),
    "cf_4h":              (float, 0.001, 0.20),
    "bh_decay":           (float, 0.80, 0.999),
    "max_lev":            (float, 0.05, 1.0),
    "kelly_fraction":     (float, 0.01, 1.0),
    "vol_target":         (float, 0.01, 0.50),
    "min_hold_minutes":   (float, 1.0, 1440.0),
    "hurst_trend_thresh": (float, 0.50, 0.80),
    "hurst_mr_thresh":    (float, 0.20, 0.50),
    "ml_signal_thresh":   (float, 0.0, 1.0),
    "nav_omega_thresh":   (float, 0.0, 1.0),
    "granger_boost":      (float, 1.0, 3.0),
}


def validate_params(params: dict) -> list[str]:
    """Return list of validation error strings. Empty = valid."""
    errors: list[str] = []
    for key, value in params.items():
        if key not in _PARAM_SCHEMA:
            errors.append(f"Unknown parameter: '{key}' (not in schema)")
            continue
        expected_type, lo, hi = _PARAM_SCHEMA[key]
        try:
            v = expected_type(value)
        except (TypeError, ValueError):
            errors.append(f"Parameter '{key}': cannot convert '{value}' to {expected_type.__name__}")
            continue
        if not (lo <= v <= hi):
            errors.append(f"Parameter '{key}': value {v} out of range [{lo}, {hi}]")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Config file helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json_file(path: Path, default: Any = None) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Cannot read %s: %s", path, exc)
    return default if default is not None else {}


def _save_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────

async def action_propose_params(params_file: str, confirm: bool) -> int:
    """Validate and propose parameter update to coordination layer."""
    _require_confirm("propose_params", confirm)
    path = Path(params_file)
    if not path.exists():
        _err(f"Params file not found: {path}")
        return 1
    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _err(f"Cannot parse JSON: {exc}")
        return 1

    errors = validate_params(params)
    if errors:
        _err("Parameter validation FAILED:")
        for e in errors:
            _err(f"  -- {e}")
        return 1

    _info(f"Validation passed ({len(params)} params). Proposing to coordination layer...")
    async with aiohttp.ClientSession() as session:
        result = await _post(session, f"{_COORD_URL}/params/propose", {"params": params})
    if result.get("ok"):
        _ok(f"Params proposed successfully: {result.get('response', {})}")
        return 0
    else:
        _err(f"Coordination layer rejected: {result}")
        return 1


async def action_override_signal(override_str: str, confirm: bool) -> int:
    """
    Hot-reload signal override for a symbol.
    Format: SYMBOL=multiplier  e.g. BTC=1.5 or ETH=0.0 (disable)
    Writes config/signal_overrides.json and signals coordination layer.
    """
    _require_confirm("override", confirm)
    if "=" not in override_str:
        _err(f"Override must be SYMBOL=multiplier, got: '{override_str}'")
        return 1
    sym, mult_str = override_str.split("=", 1)
    sym = sym.strip().upper()
    try:
        multiplier = float(mult_str.strip())
    except ValueError:
        _err(f"Multiplier must be a float, got: '{mult_str}'")
        return 1
    if not (0.0 <= multiplier <= 5.0):
        _err(f"Multiplier {multiplier} out of safe range [0, 5]")
        return 1

    overrides = _load_json_file(_OVERRIDES_FILE, {})
    overrides[sym] = {
        "multiplier": multiplier,
        "set_at": datetime.now(timezone.utc).isoformat(),
        "set_by": "live_controls_v2",
    }
    _save_json_file(_OVERRIDES_FILE, overrides)
    _ok(f"Override written: {sym} x{multiplier:.2f}")

    # Signal coordination layer to hot-reload
    async with aiohttp.ClientSession() as session:
        result = await _post(session, f"{_COORD_URL}/overrides/reload", {"symbol": sym, "multiplier": multiplier})
    if result.get("ok"):
        _ok("Coordination layer acknowledged reload.")
    else:
        _warn(f"Coordination layer not reachable (file written, reload will pick up on next bar): {result.get('error','')}")
    return 0


async def action_block_symbol(symbol: str, hours: float, confirm: bool) -> int:
    """Temporarily block trading a symbol for N hours."""
    _require_confirm("block_symbol", confirm)
    symbol = symbol.upper()
    unblock_at = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()

    blocked = _load_json_file(_BLOCKED_FILE, {})
    blocked[symbol] = {
        "blocked_until": unblock_at,
        "blocked_at": datetime.now(timezone.utc).isoformat(),
        "blocked_by": "live_controls_v2",
        "hours": hours,
    }
    _save_json_file(_BLOCKED_FILE, blocked)
    _ok(f"Symbol {symbol} blocked until {unblock_at} ({hours:.1f}h)")

    async with aiohttp.ClientSession() as session:
        result = await _post(session, f"{_COORD_URL}/symbols/block",
                              {"symbol": symbol, "unblock_at": unblock_at})
    if result.get("ok"):
        _ok("Coordination layer acknowledged block.")
    else:
        _warn(f"Coordination layer not reachable (file written): {result.get('error','')}")
    return 0


async def action_flatten_all(confirm: bool) -> int:
    """EMERGENCY: immediately close all open positions."""
    _require_confirm("flatten_all", confirm)
    _warn("Sending FLATTEN ALL to coordination layer...")
    async with aiohttp.ClientSession() as session:
        result = await _post(session, f"{_COORD_URL}/emergency/flatten_all",
                              {"source": "live_controls_v2",
                               "timestamp": datetime.now(timezone.utc).isoformat()})
    if result.get("ok"):
        _ok(f"Flatten-all acknowledged: {result.get('response', {})}")
        return 0
    else:
        _err(f"Flatten-all FAILED: {result}")
        return 1


async def action_pause(minutes: float, confirm: bool) -> int:
    """Pause new entry signals for N minutes."""
    _require_confirm("pause", confirm)
    resume_at = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
    pause_data = {
        "paused": True,
        "resume_at": resume_at,
        "paused_at": datetime.now(timezone.utc).isoformat(),
        "paused_by": "live_controls_v2",
        "minutes": minutes,
    }
    _save_json_file(_PAUSE_FILE, pause_data)
    _ok(f"Trading paused for {minutes:.0f} minutes (resumes {resume_at})")

    async with aiohttp.ClientSession() as session:
        result = await _post(session, f"{_COORD_URL}/trading/pause",
                              {"resume_at": resume_at, "minutes": minutes})
    if result.get("ok"):
        _ok("Coordination layer acknowledged pause.")
    else:
        _warn(f"Coordination layer not reachable (file written): {result.get('error','')}")
    return 0


async def action_reset_circuit(exchange: str, confirm: bool) -> int:
    """Reset circuit breaker for a given exchange via coordination layer."""
    _require_confirm("reset_circuit", confirm)
    exchange = exchange.lower()
    _warn(f"Resetting circuit breaker for: {exchange}")
    async with aiohttp.ClientSession() as session:
        result = await _post(session, f"{_COORD_URL}/circuit_breakers/reset",
                              {"exchange": exchange,
                               "reset_by": "live_controls_v2",
                               "timestamp": datetime.now(timezone.utc).isoformat()})
    if result.get("ok"):
        _ok(f"Circuit breaker '{exchange}' reset: {result.get('response', {})}")
        return 0
    else:
        _err(f"Circuit breaker reset FAILED: {result}")
        return 1


async def action_drain(service: str, confirm: bool) -> int:
    """Initiate graceful drain of a service."""
    _require_confirm("drain", confirm)
    service = service.lower()
    _warn(f"Initiating graceful drain for service: {service}")
    target_url_map = {
        "coordination": f"{_COORD_URL}/drain",
        "risk":         f"{_RISK_URL}/drain",
        "observability": f"{_OBS_URL}/drain",
    }
    url = target_url_map.get(service, f"{_COORD_URL}/drain/{service}")
    async with aiohttp.ClientSession() as session:
        result = await _post(session, url, {"service": service,
                                             "initiated_by": "live_controls_v2"})
    if result.get("ok"):
        _ok(f"Drain acknowledged for '{service}': {result.get('response', {})}")
        return 0
    else:
        _err(f"Drain FAILED for '{service}': {result}")
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Status display
# ─────────────────────────────────────────────────────────────────────────────

async def action_status() -> int:
    """Show rich terminal status of all controls and their current state."""
    async with aiohttp.ClientSession() as session:
        coord_state    = await _get(session, f"{_COORD_URL}/state")
        coord_params   = await _get(session, f"{_COORD_URL}/params")
        coord_breakers = await _get(session, f"{_COORD_URL}/circuit_breakers")
        risk_summary   = await _get(session, f"{_RISK_URL}/summary")
        obs_metrics    = await _get(session, f"{_OBS_URL}/metrics")

    overrides = _load_json_file(_OVERRIDES_FILE, {})
    blocked   = _load_json_file(_BLOCKED_FILE, {})
    pause_info = _load_json_file(_PAUSE_FILE, {})

    now_utc = datetime.now(timezone.utc)

    if _RICH and _console:
        _render_status_rich(
            coord_state, coord_params, coord_breakers,
            risk_summary, obs_metrics, overrides, blocked, pause_info, now_utc,
        )
    else:
        _render_status_plain(
            coord_state, coord_params, coord_breakers,
            risk_summary, obs_metrics, overrides, blocked, pause_info, now_utc,
        )
    return 0


def _render_status_rich(
    coord_state, coord_params, coord_breakers,
    risk_summary, obs_metrics, overrides, blocked, pause_info, now_utc,
) -> None:
    from rich.panel import Panel
    from rich.layout import Layout

    # Coordination state
    tbl_coord = Table(title="Coordination State", box=box.SIMPLE, padding=(0, 1))
    tbl_coord.add_column("Key", style="cyan")
    tbl_coord.add_column("Value", style="white")
    for k, v in (coord_state or {}).items():
        tbl_coord.add_row(str(k), str(v))
    if not coord_state or "error" in coord_state:
        tbl_coord.add_row("[red]OFFLINE[/red]", str(coord_state.get("error", "")))
    _console.print(tbl_coord)

    # Params
    tbl_params = Table(title="Live Parameters", box=box.SIMPLE, padding=(0, 1))
    tbl_params.add_column("Param", style="cyan")
    tbl_params.add_column("Value", style="white", justify="right")
    tbl_params.add_column("Range", style="dim")
    param_vals = (coord_params or {}).get("values", {})
    for k, v in param_vals.items():
        rng = _PARAM_SCHEMA.get(k)
        rng_str = f"[{rng[1]}, {rng[2]}]" if rng else ""
        tbl_params.add_row(str(k), str(v), rng_str)
    _console.print(tbl_params)

    # Circuit breakers
    tbl_cb = Table(title="Circuit Breakers", box=box.SIMPLE, padding=(0, 1))
    tbl_cb.add_column("Exchange", style="cyan")
    tbl_cb.add_column("State")
    for name, active in (coord_breakers or {}).items():
        style = "[red]TRIPPED[/red]" if active else "[green]OK[/green]"
        tbl_cb.add_row(str(name), style)
    if not coord_breakers or "error" in coord_breakers:
        tbl_cb.add_row("[dim]N/A[/dim]", "")
    _console.print(tbl_cb)

    # Signal overrides
    tbl_ov = Table(title="Signal Overrides (config/signal_overrides.json)", box=box.SIMPLE, padding=(0, 1))
    tbl_ov.add_column("Symbol", style="cyan")
    tbl_ov.add_column("Multiplier", justify="right")
    tbl_ov.add_column("Set At", style="dim")
    for sym, info in (overrides or {}).items():
        tbl_ov.add_row(sym, str(info.get("multiplier", "?")), str(info.get("set_at", ""))[:19])
    if not overrides:
        tbl_ov.add_row("[dim]none[/dim]", "", "")
    _console.print(tbl_ov)

    # Blocked symbols
    tbl_bl = Table(title="Blocked Symbols", box=box.SIMPLE, padding=(0, 1))
    tbl_bl.add_column("Symbol", style="cyan")
    tbl_bl.add_column("Blocked Until")
    tbl_bl.add_column("Status")
    for sym, info in (blocked or {}).items():
        until = info.get("blocked_until", "")
        try:
            until_dt = datetime.fromisoformat(until)
            still_blocked = until_dt > now_utc
        except Exception:
            still_blocked = True
        status = "[red]ACTIVE[/red]" if still_blocked else "[dim]EXPIRED[/dim]"
        tbl_bl.add_row(sym, until[:19], status)
    if not blocked:
        tbl_bl.add_row("[dim]none[/dim]", "", "")
    _console.print(tbl_bl)

    # Pause state
    if pause_info:
        resume_at = pause_info.get("resume_at", "")
        try:
            resume_dt = datetime.fromisoformat(resume_at)
            paused = resume_dt > now_utc
        except Exception:
            paused = False
        if paused:
            _console.print(Panel(
                f"[red]TRADING PAUSED[/red] -- resumes {resume_at[:19]}",
                border_style="red", title="Pause State",
            ))
        else:
            _console.print(Panel("[green]Trading active (pause expired)[/green]", title="Pause State"))
    else:
        _console.print(Panel("[green]Trading active (no pause)[/green]", title="Pause State"))

    # Risk
    tbl_risk = Table(title="Risk Summary", box=box.SIMPLE, padding=(0, 1))
    tbl_risk.add_column("Metric", style="cyan")
    tbl_risk.add_column("Value", justify="right")
    for k, v in (risk_summary or {}).items():
        if not isinstance(v, (dict, list)):
            tbl_risk.add_row(str(k), str(v))
    if not risk_summary or "error" in risk_summary:
        tbl_risk.add_row("[red]OFFLINE[/red]", "")
    _console.print(tbl_risk)


def _render_status_plain(
    coord_state, coord_params, coord_breakers,
    risk_summary, obs_metrics, overrides, blocked, pause_info, now_utc,
) -> None:
    sep = "-" * 60
    print(sep)
    print("LIVE CONTROLS V2 -- STATUS")
    print(sep)

    print("\nCOORDINATION STATE")
    if coord_state and "error" not in coord_state:
        for k, v in coord_state.items():
            print(f"  {k}: {v}")
    else:
        print(f"  OFFLINE: {coord_state.get('error','')}")

    print("\nLIVE PARAMETERS")
    param_vals = (coord_params or {}).get("values", {})
    for k, v in param_vals.items():
        print(f"  {k}: {v}")
    if not param_vals:
        print("  (coordination offline or no params)")

    print("\nCIRCUIT BREAKERS")
    for name, active in (coord_breakers or {}).items():
        print(f"  {name}: {'TRIPPED' if active else 'OK'}")
    if not coord_breakers or "error" in coord_breakers:
        print("  N/A (offline)")

    print("\nSIGNAL OVERRIDES")
    for sym, info in (overrides or {}).items():
        print(f"  {sym}: x{info.get('multiplier','?')}  set_at={str(info.get('set_at',''))[:19]}")
    if not overrides:
        print("  (none)")

    print("\nBLOCKED SYMBOLS")
    for sym, info in (blocked or {}).items():
        until = info.get("blocked_until", "")
        try:
            still = datetime.fromisoformat(until) > now_utc
        except Exception:
            still = True
        print(f"  {sym}: until={until[:19]}  {'ACTIVE' if still else 'expired'}")
    if not blocked:
        print("  (none)")

    print("\nPAUSE STATE")
    if pause_info:
        resume_at = pause_info.get("resume_at", "")
        try:
            paused = datetime.fromisoformat(resume_at) > now_utc
        except Exception:
            paused = False
        print(f"  {'PAUSED until ' + resume_at[:19] if paused else 'Trading active (pause expired)'}")
    else:
        print("  Trading active (no pause)")

    print("\nRISK SUMMARY")
    for k, v in (risk_summary or {}).items():
        if not isinstance(v, (dict, list)):
            print(f"  {k}: {v}")
    if not risk_summary or "error" in risk_summary:
        print("  OFFLINE")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_confirm(action: str, confirmed: bool) -> None:
    if action in _DESTRUCTIVE_ACTIONS and not confirmed:
        print(f"ERROR: Action '{action}' is destructive and requires --confirm flag.")
        sys.exit(1)


def _ok(msg: str) -> None:
    if _RICH and _console:
        _console.print(f"[green]OK[/green]  {msg}")
    else:
        print(f"OK  {msg}")


def _warn(msg: str) -> None:
    if _RICH and _console:
        _console.print(f"[yellow]WARN[/yellow]  {msg}")
    else:
        print(f"WARN  {msg}")


def _err(msg: str) -> None:
    if _RICH and _console:
        _console.print(f"[red]ERR[/red]  {msg}")
    else:
        print(f"ERR  {msg}")


def _info(msg: str) -> None:
    if _RICH and _console:
        _console.print(f"[cyan]INFO[/cyan]  {msg}")
    else:
        print(f"INFO  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LARSA v18 live controls CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/live_controls_v2.py --status
  python tools/live_controls_v2.py --override BTC=1.5 --confirm
  python tools/live_controls_v2.py --block-symbol ETH 4 --confirm
  python tools/live_controls_v2.py --pause 30 --confirm
  python tools/live_controls_v2.py --flatten-all --confirm
  python tools/live_controls_v2.py --reset-circuit alpaca --confirm
  python tools/live_controls_v2.py --drain coordination --confirm
  python tools/live_controls_v2.py --propose-params params.json --confirm
""",
    )
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument("--status", action="store_true",
                       help="Show status of all controls")
    mutex.add_argument("--propose-params", metavar="JSON_FILE",
                       help="Validate and propose parameter update to coordination layer")
    mutex.add_argument("--override", metavar="SYMBOL=MULTIPLIER",
                       help="Set signal multiplier override for a symbol (0=disable, 1=normal, 1.5=boost)")
    mutex.add_argument("--block-symbol", metavar=("SYMBOL", "HOURS"), nargs="+",
                       help="Temporarily block trading a symbol for N hours (default 1h)")
    mutex.add_argument("--flatten-all", action="store_true",
                       help="EMERGENCY: close all open positions immediately")
    mutex.add_argument("--pause", metavar="MINUTES", type=float,
                       help="Pause new entry signals for N minutes")
    mutex.add_argument("--reset-circuit", metavar="EXCHANGE",
                       help="Reset circuit breaker for exchange (alpaca/binance/polygon)")
    mutex.add_argument("--drain", metavar="SERVICE",
                       help="Initiate graceful drain of a service")

    parser.add_argument("--confirm", action="store_true",
                        help="Confirm destructive actions (required for all write/emergency ops)")
    parser.add_argument("--coord-url", default=_COORD_URL)
    parser.add_argument("--risk-url",  default=_RISK_URL)
    parser.add_argument("--obs-url",   default=_OBS_URL)
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    global _COORD_URL, _RISK_URL, _OBS_URL
    _COORD_URL = args.coord_url
    _RISK_URL  = args.risk_url
    _OBS_URL   = args.obs_url

    rc = 0
    if args.status:
        rc = asyncio.run(action_status())

    elif args.propose_params:
        rc = asyncio.run(action_propose_params(args.propose_params, args.confirm))

    elif args.override:
        rc = asyncio.run(action_override_signal(args.override, args.confirm))

    elif args.block_symbol:
        parts  = args.block_symbol
        symbol = parts[0]
        hours  = float(parts[1]) if len(parts) > 1 else 1.0
        rc = asyncio.run(action_block_symbol(symbol, hours, args.confirm))

    elif args.flatten_all:
        rc = asyncio.run(action_flatten_all(args.confirm))

    elif args.pause is not None:
        rc = asyncio.run(action_pause(args.pause, args.confirm))

    elif args.reset_circuit:
        rc = asyncio.run(action_reset_circuit(args.reset_circuit, args.confirm))

    elif args.drain:
        rc = asyncio.run(action_drain(args.drain, args.confirm))

    sys.exit(rc)


if __name__ == "__main__":
    main()
