#!/usr/bin/env python
"""
tools/srfm_config.py -- Validator for .srfm strategy config files.

Usage:
  python tools/srfm_config.py strategies/larsa-v4/strategy.srfm
  python tools/srfm_config.py --diff strategies/larsa-v1/strategy.srfm strategies/larsa-v4/strategy.srfm
  python tools/srfm_config.py --generate strategies/larsa-v4/main.py
"""

import argparse
import ast
import configparser
import io
import re
import sys
from pathlib import Path

# -- Parameter documentation --------------------------------------------------
PARAM_DOCS = {
    # physics
    "cf_es": {
        "desc": "Critical fraction for ES -- hourly |return| threshold for SPACELIKE classification",
        "formula": "hourly |return| > cf_es  ->  SPACELIKE",
        "implication": lambda v: (
            f"hourly |return| > {float(v)*100:.3f}% triggers SPACELIKE\n"
            f"    ES median hourly |return| ~0.067% -> ~{min(99, max(1, int(87 + (float(v) - 0.001)/0.001 * 8)))}% TIMELIKE rate"
            if float(v) > 0 else "invalid"
        ),
        "type": float,
        "range": (0.0001, 0.01),
    },
    "cf_nq": {
        "desc": "Critical fraction for NQ -- SPACELIKE threshold",
        "formula": "hourly |return| > cf_nq  ->  SPACELIKE",
        "implication": lambda v: (
            f"NQ hourly |return| > {float(v)*100:.3f}% triggers SPACELIKE\n"
            f"    NQ is more volatile than ES; higher cf = fewer SPACELIKE events"
        ),
        "type": float,
        "range": (0.0001, 0.02),
    },
    "cf_ym": {
        "desc": "Critical fraction for YM -- SPACELIKE threshold",
        "formula": "hourly |return| > cf_ym  ->  SPACELIKE",
        "implication": lambda v: f"YM hourly |return| > {float(v)*100:.3f}% triggers SPACELIKE",
        "type": float,
        "range": (0.0001, 0.02),
    },
    "bh_form": {
        "desc": "BH formation threshold -- mass must exceed this to activate a black hole",
        "formula": "bh_active = (bh_mass >= bh_form)",
        "implication": lambda v: (
            f"BH forms when mass crosses {v} (mass accretes at ~0.05/bar TIMELIKE)\n"
            f"    Expected bars to formation: ~{int((float(v) - 0.0) / 0.05)} TIMELIKE bars"
        ),
        "type": float,
        "range": (0.5, 5.0),
    },
    "bh_collapse": {
        "desc": "BH collapse threshold -- mass below this collapses the BH and may trigger exit",
        "formula": "bh_active = False when bh_mass < bh_collapse",
        "implication": lambda v: f"BH collapses (position exit signal) when mass drops below {v}",
        "type": float,
        "range": (0.1, 3.0),
    },
    "bh_decay": {
        "desc": "BH mass decay multiplier per SPACELIKE bar",
        "formula": "bh_mass *= bh_decay  (each SPACELIKE bar)",
        "implication": lambda v: (
            f"Mass halves in ~{int(0.693 / (1 - float(v))):.0f} bars of SPACELIKE activity"
        ),
        "type": float,
        "range": (0.5, 0.999),
    },
    # sizing
    "max_leverage": {
        "desc": "Maximum leverage per instrument as fraction of portfolio value",
        "formula": "position_value <= max_leverage * portfolio_value",
        "implication": lambda v: (
            f"{float(v)*100:.0f}% of portfolio per instrument, "
            f"{float(v)*3*100:.0f}% combined max (3 instruments)"
        ),
        "type": float,
        "range": (0.0, 2.0),
    },
    "solo_bh_cap": {
        "desc": "Position size cap when only one BH is active (single-conviction entry)",
        "formula": "if n_active_bh == 1: target = min(target, solo_bh_cap)",
        "implication": lambda v: (
            f"Single-BH positions capped at {float(v)*100:.0f}% (reduces variance ~30%)"
        ),
        "type": float,
        "range": (0.0, 1.0),
    },
    "nq_notional_cap": {
        "desc": "Hard dollar cap on NQ notional exposure ($)",
        "formula": "nq_contracts = min(contracts, nq_notional_cap / (nq_price * 20))",
        "implication": lambda v: (
            f"At $1M portfolio: {float(v)/1e6*100:.0f}% cap\n"
            f"    At $2M portfolio: {float(v)/2e6*100:.0f}% cap\n"
            f"    At $3M portfolio: {float(v)/3e6*100:.0f}% cap  <- key protection"
        ),
        "type": float,
        "range": (50000, 5000000),
    },
    "conv_size_enabled": {
        "desc": "Enable convergence-aware position sizing (scales with BH mass)",
        "formula": "target *= (bh_mass / bh_form) when enabled",
        "implication": lambda v: (
            "Position size scales with BH mass -- larger positions at higher conviction"
            if str(v).lower() == "true" else
            "Fixed position size regardless of BH mass"
        ),
        "type": bool,
        "range": None,
    },
    # gates
    "tl_req_normal": {
        "desc": "Minimum consecutive TIMELIKE bars required before entry (normal volatility)",
        "formula": "entry allowed if tl_confirm >= tl_req_normal",
        "implication": lambda v: f"Entry requires {v} consecutive TIMELIKE bars -- filters whipsaw",
        "type": int,
        "range": (1, 20),
    },
    "tl_req_high_vol": {
        "desc": "Minimum TIMELIKE bars required in HIGH_VOLATILITY regime",
        "formula": "entry allowed if tl_confirm >= tl_req_high_vol (when HIGH_VOL)",
        "implication": lambda v: (
            f"High-vol regime requires only {v} TIMELIKE bar(s) -- aggressive entry during vol spikes"
        ),
        "type": int,
        "range": (1, 10),
    },
    "pos_floor_trigger_ctl": {
        "desc": "CTL (consecutive TIMELIKE length) threshold to activate position floor",
        "formula": "if ctl >= pos_floor_trigger_ctl: lock pos_floor = current_target",
        "implication": lambda v: (
            f"Position floor locks at ctl={v} (~{int(int(v)*1.5)} bars into a TL run)\n"
            f"    Prevents sizing down during convergence events"
        ),
        "type": int,
        "range": (1, 20),
    },
    "pos_floor_retention": {
        "desc": "Fraction of peak position retained as floor when floor is active",
        "formula": "pos_floor = pos_floor_retention * peak_target",
        "implication": lambda v: (
            f"Floor retains {float(v)*100:.0f}% of peak position -- "
            f"allows {(1-float(v))*100:.0f}% reduction before floor binds"
        ),
        "type": float,
        "range": (0.0, 1.0),
    },
    "pos_floor_decay": {
        "desc": "Floor decay multiplier per bar after floor trigger -- gradually relaxes constraint",
        "formula": "pos_floor *= pos_floor_decay (each bar after trigger)",
        "implication": lambda v: (
            f"Floor halves in ~{int(0.693 / (1 - float(v))):.0f} bars -- "
            f"releases position control after sustained SPACELIKE"
        ),
        "type": float,
        "range": (0.5, 0.999),
    },
    "bear_long_gate_rhb": {
        "desc": "Block LONG entries if consecutive BEAR regime bars (rhb) exceeds this threshold",
        "formula": "if regime == BEAR and rhb >= bear_long_gate_rhb: skip long entry",
        "implication": lambda v: (
            f"Longs blocked after {v}+ consecutive BEAR bars\n"
            f"    BEAR avg run ~29.5 bars -> blocks ~{int((29.5-int(v))/29.5*100)}% of BEAR-regime long exposure"
        ),
        "type": int,
        "range": (1, 50),
    },
    # risk
    "max_drawdown_gate": {
        "desc": "Portfolio drawdown fraction that triggers full liquidation and cooldown",
        "formula": "if drawdown >= max_drawdown_gate: liquidate all, cooldown 3 bars",
        "implication": lambda v: (
            f"Liquidates all positions at {float(v)*100:.0f}% portfolio drawdown\n"
            f"    Cooldown: 3 bars, then ramp back over 5 bars"
        ),
        "type": float,
        "range": (0.01, 0.50),
    },
    "daily_loss_limit": {
        "desc": "Intraday loss limit as fraction of portfolio -- halts trading for the day",
        "formula": "if daily_pnl / portfolio <= -daily_loss_limit: halt until next open",
        "implication": lambda v: (
            f"Trading halts if daily loss exceeds {float(v)*100:.0f}% of portfolio"
        ),
        "type": float,
        "range": (0.001, 0.20),
    },
    "trail_pct_at_150pct_gain": {
        "desc": "Trailing stop percentage activated when position gain exceeds 150%",
        "formula": "if gain > 1.5: trail_stop = peak_gain * (1 - trail_pct_at_150pct_gain)",
        "implication": lambda v: (
            f"Trailing stop of {float(v)*100:.0f}% activates at 150% gain -- protects windfalls"
        ),
        "type": float,
        "range": (0.01, 0.50),
    },
    # instruments
    "es_multiplier": {
        "desc": "ES futures dollar multiplier ($ per point)",
        "formula": "es_pnl = (exit - entry) * quantity * es_multiplier",
        "implication": lambda v: f"Each ES point = ${v}. 1 contract at 5000 = ${int(float(v)*5000):,} notional",
        "type": int,
        "range": (50, 50),
    },
    "nq_multiplier": {
        "desc": "NQ futures dollar multiplier ($ per point)",
        "formula": "nq_pnl = (exit - entry) * quantity * nq_multiplier",
        "implication": lambda v: (
            f"Each NQ point = ${v}. 1 contract at 20000 = ${int(float(v)*20000):,} notional\n"
            f"    High multiplier amplifies both gains and losses vs ES"
        ),
        "type": int,
        "range": (20, 20),
    },
    "ym_multiplier": {
        "desc": "YM futures dollar multiplier ($ per point)",
        "formula": "ym_pnl = (exit - entry) * quantity * ym_multiplier",
        "implication": lambda v: f"Each YM point = ${v}. 1 contract at 40000 = ${int(float(v)*40000):,} notional",
        "type": int,
        "range": (5, 5),
    },
}

REQUIRED_PARAMS = {
    "physics": ["cf_es", "cf_nq", "bh_form", "bh_collapse", "bh_decay"],
    "sizing": ["max_leverage"],
    "gates": ["tl_req_normal"],
    "risk": ["max_drawdown_gate"],
    "instruments": ["es_multiplier", "nq_multiplier", "ym_multiplier"],
}


def parse_srfm(path: Path) -> tuple[dict, configparser.ConfigParser]:
    """Parse a .srfm file. Returns (meta, config_parser)."""
    text = path.read_text(encoding="utf-8")
    meta = {}
    section_lines = []
    in_section = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if "=" in stripped and not stripped.startswith("["):
            # Top-level key=value (before any section)
            if not in_section:
                k, _, v = stripped.partition("=")
                meta[k.strip()] = v.strip()
                continue
        if stripped.startswith("["):
            in_section = True
        section_lines.append(line)

    ini_text = "[DEFAULT]\n" + "\n".join(section_lines) if section_lines else "[DEFAULT]\n"
    # configparser needs a proper fallback section
    ini_text = "\n".join(section_lines)

    cp = configparser.RawConfigParser()
    cp.optionxform = str  # preserve case
    cp.read_string(ini_text)
    return meta, cp


def validate(path: Path) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings)."""
    errors = []
    warnings = []
    meta, cp = parse_srfm(path)

    all_params = {}
    for section in cp.sections():
        for k, v in cp.items(section):
            all_params[k] = v

    # Type checks
    for k, v in all_params.items():
        doc = PARAM_DOCS.get(k)
        if doc is None:
            continue
        expected_type = doc["type"]
        try:
            if expected_type == bool:
                if v.lower() not in ("true", "false", "1", "0"):
                    errors.append(f"{k} = {v!r} -- expected bool (true/false)")
            elif expected_type == int:
                int(v)
            elif expected_type == float:
                float(v)
        except ValueError:
            errors.append(f"{k} = {v!r} -- expected {expected_type.__name__}")

        # Range check
        if doc["range"] and expected_type in (int, float):
            try:
                num = expected_type(v)
                lo, hi = doc["range"]
                if not (lo <= num <= hi):
                    errors.append(f"{k} = {v} -- out of range [{lo}, {hi}]")
            except Exception:
                pass

    # Consistency checks
    max_lev = float(all_params.get("max_leverage", 0.65))
    solo_cap = float(all_params.get("solo_bh_cap", max_lev))
    if solo_cap > max_lev:
        errors.append(f"solo_bh_cap ({solo_cap}) > max_leverage ({max_lev}) -- solo cap exceeds max leverage")

    nq_cap = float(all_params.get("nq_notional_cap", 999e6))
    if nq_cap < 50000:
        warnings.append(f"nq_notional_cap ${nq_cap:,.0f} very low -- may prevent any NQ positions")
    if nq_cap < 500000 and "nq_notional_cap" in all_params:
        warnings.append(f"nq_notional_cap too conservative for <$500k portfolios (will cap at {nq_cap/500000*100:.0f}%)")

    if "pos_floor_retention" not in all_params:
        warnings.append("pos_floor_retention not set (default 0.70 assumed)")

    bh_form = float(all_params.get("bh_form", 1.5))
    bh_collapse = float(all_params.get("bh_collapse", 1.0))
    if bh_collapse >= bh_form:
        errors.append(f"bh_collapse ({bh_collapse}) >= bh_form ({bh_form}) -- BH collapses immediately on formation")

    return errors, warnings


def cmd_validate(path: Path):
    meta, cp = parse_srfm(path)
    errors, warnings = validate(path)

    width = 55
    label = f"{path.parent.name}/{path.name}"
    print("-" * width)
    print(f"  SRFM Config Validator -- {label}")
    print("-" * width)

    for section in cp.sections():
        print(f"  [{section}]")
        for k, v in cp.items(section):
            doc = PARAM_DOCS.get(k)
            if doc:
                try:
                    impl = doc["implication"](v)
                    first_line = impl.split("\n")[0]
                    rest_lines = impl.split("\n")[1:]
                    pad = " " * (len(k) + len(str(v)) + 7)
                    print(f"  {k} = {v}  ->  {first_line}")
                    for line in rest_lines:
                        print(f"  {pad}{line}")
                except Exception:
                    print(f"  {k} = {v}")
            else:
                print(f"  {k} = {v}")
        print()

    # Count params
    total_params = sum(len(list(cp.items(s))) for s in cp.sections())
    if not errors and not warnings:
        print(f"  Config valid ({total_params} params, 0 errors, 0 warnings)")
    elif not errors:
        print(f"  Config valid ({total_params} params, 0 errors, {len(warnings)} warning(s))")
    else:
        print(f"  Config INVALID ({total_params} params, {len(errors)} error(s), {len(warnings)} warning(s))")

    for e in errors:
        print(f"  (X) {e}")
    for w in warnings:
        print(f"  ! {w}")
    print("-" * width)


def cmd_diff(path1: Path, path2: Path):
    _, cp1 = parse_srfm(path1)
    _, cp2 = parse_srfm(path2)

    all1 = {k: v for s in cp1.sections() for k, v in cp1.items(s)}
    all2 = {k: v for s in cp2.sections() for k, v in cp2.items(s)}

    all_keys = sorted(set(all1) | set(all2))
    changed = [(k, all1.get(k, "(missing)"), all2.get(k, "(missing)"))
               for k in all_keys if all1.get(k) != all2.get(k)]

    width = 65
    print("-" * width)
    print(f"  SRFM Diff: {path1.parent.name} vs {path2.parent.name}")
    print("-" * width)
    if not changed:
        print("  (no differences)")
    else:
        print(f"  {len(changed)} parameter(s) changed:\n")
        for k, v1, v2 in changed:
            doc = PARAM_DOCS.get(k)
            desc = f"  -- {doc['desc']}" if doc else ""
            print(f"  {k}:")
            print(f"    {path1.parent.name:12s}  {v1}")
            print(f"    {path2.parent.name:12s}  {v2}{desc}")
            print()
    print("-" * width)


def cmd_generate(py_path: Path):
    """Extract self.xxx = <literal> assignments from a Python strategy file."""
    text = py_path.read_text(encoding="utf-8")
    # Find assignments like self.bh_form = 1.5 or self.max_leverage = 0.65
    pattern = re.compile(r"self\.(\w+)\s*=\s*([0-9.eE+\-]+|True|False|'[^']*'|\"[^\"]*\")")
    found = {}
    for m in pattern.finditer(text):
        name, val = m.group(1), m.group(2)
        # Only capture known SRFM params
        if name in PARAM_DOCS:
            found[name] = val

    if not found:
        print(f"No recognized SRFM parameters found in {py_path}")
        return

    # Bucket into sections
    sections = {"physics": [], "sizing": [], "gates": [], "risk": [], "instruments": []}
    section_map = {
        "cf_es": "physics", "cf_nq": "physics", "cf_ym": "physics",
        "bh_form": "physics", "bh_collapse": "physics", "bh_decay": "physics",
        "max_leverage": "sizing", "solo_bh_cap": "sizing",
        "nq_notional_cap": "sizing", "conv_size_enabled": "sizing",
        "tl_req_normal": "gates", "tl_req_high_vol": "gates",
        "pos_floor_trigger_ctl": "gates", "pos_floor_retention": "gates",
        "pos_floor_decay": "gates", "bear_long_gate_rhb": "gates",
        "max_drawdown_gate": "risk", "daily_loss_limit": "risk",
        "trail_pct_at_150pct_gain": "risk",
        "es_multiplier": "instruments", "nq_multiplier": "instruments", "ym_multiplier": "instruments",
    }
    for k, v in found.items():
        sec = section_map.get(k, "sizing")
        sections[sec].append((k, v))

    out_path = py_path.parent / "strategy.srfm"
    lines = [
        f"# Generated from {py_path.name}",
        f"# Generated: 2026-04-03",
        "",
        f"version = auto",
        f"class = {py_path.stem}",
        "",
    ]
    for sec, params in sections.items():
        if params:
            lines.append(f"[{sec}]")
            for k, v in params:
                lines.append(f"{k} = {v}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {out_path} ({len(found)} params extracted)")


def main():
    parser = argparse.ArgumentParser(description="SRFM Config Validator")
    parser.add_argument("config", nargs="?", help=".srfm config file to validate")
    parser.add_argument("--diff", nargs=2, metavar=("FILE1", "FILE2"), help="Diff two configs")
    parser.add_argument("--generate", metavar="PY_FILE", help="Generate .srfm from Python strategy")
    args = parser.parse_args()

    if args.diff:
        cmd_diff(Path(args.diff[0]), Path(args.diff[1]))
    elif args.generate:
        cmd_generate(Path(args.generate))
    elif args.config:
        cmd_validate(Path(args.config))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
