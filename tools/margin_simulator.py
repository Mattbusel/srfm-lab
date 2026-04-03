"""
margin_simulator.py — Simulate IB futures margin model on top of V8Broker.

Reconstructs the QC "Insufficient buying power" / margin call death loop that
turned a $20M peak into a $222K final equity over 7 years.

Usage:
    python tools/margin_simulator.py

Outputs:
    results/margin_sim_seahorse.json
    results/margin_sim_synth.json
"""

import ast
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# IB Futures Margin Constants (approximate initial margin per contract)
# ---------------------------------------------------------------------------
MARGIN_PER_CONTRACT: Dict[str, float] = {
    "ES": 6_600.0,
    "NQ": 17_600.0,
    "YM": 4_400.0,
}
CONTRACT_MULT: Dict[str, float] = {
    "ES": 50.0,
    "NQ": 20.0,
    "YM": 5.0,
}

MARGIN_BUFFER = 0.50   # must keep margin < 50% of equity
MARGIN_CALL_THRESH = 1.00  # margin call if margin > 100% equity

# ---------------------------------------------------------------------------
# Utility: parse Seahorse trades CSV
# ---------------------------------------------------------------------------
SYM_RE = re.compile(r"(ES|NQ|YM)")


def parse_seahorse_csv(path: str) -> List[dict]:
    """Parse QC trades CSV into a list of dicts with typed fields."""
    trades = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym_raw = row.get("Symbols", "").strip().strip('"')
            m = SYM_RE.match(sym_raw)
            if not m:
                continue
            sym = m.group(1)

            def _f(k: str) -> float:
                v = row.get(k, "").strip()
                return float(v) if v else 0.0

            entry_ts = row.get("Entry Time", "").strip()
            exit_ts  = row.get("Exit Time",  "").strip()

            trades.append({
                "sym":        sym,
                "entry_time": entry_ts,
                "exit_time":  exit_ts,
                "direction":  1 if row.get("Direction", "").strip() == "Buy" else -1,
                "entry_price": _f("Entry Price"),
                "exit_price":  _f("Exit Price"),
                "qty":         _f("Quantity"),
                "pnl":         _f("P&L"),
                "fees":        _f("Fees"),
            })
    return trades


# ---------------------------------------------------------------------------
# Reconstruct an hourly equity curve from Seahorse trades
# ---------------------------------------------------------------------------

def reconstruct_equity_curve(
    trades: List[dict],
    initial_equity: float = 1_000_000.0,
) -> Tuple[List[str], List[float], List[dict]]:
    """
    Walk the trades in time order and reconstruct an approximate hourly
    equity curve.  Each closed trade contributes its P&L + fees at exit time.

    Returns (timestamps, equity_curve, annotated_trades).
    """
    # Sort by exit time so we apply P&L chronologically
    closed = sorted(
        [t for t in trades if t["exit_time"]],
        key=lambda x: x["exit_time"]
    )

    equity = initial_equity
    timestamps = []
    equity_curve = []
    annotated = []

    for t in closed:
        equity += t["pnl"] - t["fees"]
        timestamps.append(t["exit_time"])
        equity_curve.append(equity)
        annotated.append({**t, "running_equity": equity})

    return timestamps, equity_curve, annotated


# ---------------------------------------------------------------------------
# Margin-aware broker — wraps position snapshots from the CSV
# ---------------------------------------------------------------------------

class MarginBroker:
    """
    Replay the Seahorse trades with IB margin enforcement.

    Instead of re-running the signal engine, we use the ACTUAL quantities
    from QC and apply margin scaling / forced liquidation.
    """

    def __init__(self, initial_equity: float = 1_000_000.0):
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.equity_curve: List[float] = [initial_equity]
        self.timestamps: List[str] = [""]

        # Open positions: sym -> {qty, entry_price, direction}
        self.open_positions: Dict[str, dict] = {}

        # Event logs
        self.margin_calls: List[dict] = []
        self.margin_scale_events: int = 0
        self.all_events: List[dict] = []

    # ------------------------------------------------------------------
    def _total_margin_required(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.open_positions.items():
            total += pos["qty"] * MARGIN_PER_CONTRACT[sym]
        return total

    def _notional_exposure(self, prices: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for sym, pos in self.open_positions.items():
            p = prices.get(sym, pos["entry_price"])
            out[sym] = pos["qty"] * CONTRACT_MULT[sym] * p
        return out

    # ------------------------------------------------------------------
    def open_trade(self, sym: str, qty: float, price: float,
                   direction: int, ts: str,
                   prices_snapshot: Dict[str, float]) -> dict:
        """
        Try to open a position.  Apply margin scaling if needed.
        Returns event dict.
        """
        original_qty = qty

        # Proposed total margin if we open at full qty
        existing_margin = self._total_margin_required(prices_snapshot)
        new_margin_add  = qty * MARGIN_PER_CONTRACT.get(sym, 6600.0)
        proposed_total  = existing_margin + new_margin_add

        scaled = False
        if proposed_total > self.equity * MARGIN_BUFFER:
            # How much room do we have?
            room = max(0.0, self.equity * MARGIN_BUFFER - existing_margin)
            margin_per = MARGIN_PER_CONTRACT.get(sym, 6600.0)
            allowed_qty = int(room / margin_per) if margin_per > 0 else 0
            if allowed_qty < qty:
                qty = float(allowed_qty)
                scaled = True
                self.margin_scale_events += 1

        if qty <= 0:
            event = {
                "type": "MARGIN_BLOCKED", "ts": ts, "sym": sym,
                "requested_qty": original_qty, "allowed_qty": 0.0,
                "equity": self.equity,
                "margin_required": self._total_margin_required(prices_snapshot),
            }
            self.all_events.append(event)
            return event

        # Apply fees
        fee = qty * 2.0 * 2  # $2/contract/side, round trip approximation
        self.equity -= fee

        # Merge into open positions (simple: just track qty and avg price)
        if sym in self.open_positions:
            p = self.open_positions[sym]
            total_qty = p["qty"] + qty
            avg_price = (p["entry_price"] * p["qty"] + price * qty) / total_qty
            self.open_positions[sym] = {
                "qty": total_qty, "entry_price": avg_price,
                "direction": direction,
            }
        else:
            self.open_positions[sym] = {
                "qty": qty, "entry_price": price,
                "direction": direction,
            }

        event = {
            "type": "MARGIN_SCALED" if scaled else "OPEN",
            "ts": ts, "sym": sym,
            "requested_qty": original_qty, "actual_qty": qty,
            "equity": self.equity,
            "margin_total": self._total_margin_required(prices_snapshot),
        }
        self.all_events.append(event)
        return event

    # ------------------------------------------------------------------
    def close_trade(self, sym: str, qty: float, exit_price: float,
                    pnl: float, fees: float, ts: str,
                    prices_snapshot: Dict[str, float]) -> dict:
        """Apply a realized P&L close."""
        net = pnl - fees
        self.equity += net

        if sym in self.open_positions:
            pos = self.open_positions[sym]
            pos["qty"] -= qty
            if pos["qty"] <= 0:
                del self.open_positions[sym]

        # Check margin call AFTER applying P&L (equity might have dropped)
        margin_req = self._total_margin_required(prices_snapshot)
        event = {"type": "CLOSE", "ts": ts, "sym": sym,
                 "qty": qty, "pnl": pnl, "fees": fees,
                 "equity": self.equity, "margin_req": margin_req}

        if margin_req > self.equity * MARGIN_CALL_THRESH and margin_req > 0:
            forced_loss = self._force_liquidate(prices_snapshot, ts)
            mc_event = {
                "type": "MARGIN_CALL",
                "ts": ts,
                "equity_before": self.equity - forced_loss,
                "equity_after": self.equity,
                "margin_required": margin_req,
                "forced_loss": forced_loss,
                "open_syms": list(self.open_positions.keys()),
            }
            self.margin_calls.append(mc_event)
            self.all_events.append(mc_event)

        self.equity_curve.append(self.equity)
        self.timestamps.append(ts)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        return event

    # ------------------------------------------------------------------
    def _force_liquidate(self, prices: Dict[str, float], ts: str) -> float:
        """Liquidate all open positions at current prices."""
        total_loss = 0.0
        for sym in list(self.open_positions.keys()):
            pos = self.open_positions[sym]
            p_curr = prices.get(sym, pos["entry_price"])
            p_entry = pos["entry_price"]
            direction = pos["direction"]
            qty = pos["qty"]
            mult = CONTRACT_MULT.get(sym, 50.0)
            pnl = (p_curr - p_entry) * direction * qty * mult
            fee = qty * 4.0  # $2/side × 2
            net = pnl - fee
            self.equity += net
            total_loss += net  # negative = loss
            del self.open_positions[sym]
        return total_loss


# ---------------------------------------------------------------------------
# Replay Seahorse with margin enforcement
# ---------------------------------------------------------------------------

def replay_with_margin(
    trades: List[dict],
    initial_equity: float = 1_000_000.0,
) -> Tuple[MarginBroker, dict]:
    """
    Replay QC Seahorse trades chronologically through the margin broker.

    We maintain a rolling price dict by using entry/exit prices from the
    trade records themselves.
    """
    broker = MarginBroker(initial_equity)
    prices: Dict[str, float] = {"ES": 2690.75, "NQ": 6486.25, "YM": 24801.0}

    # Sort all trade events (open + close) by time
    events: List[dict] = []
    for t in trades:
        if t["entry_time"]:
            events.append({"kind": "open", "ts": t["entry_time"], "trade": t})
        if t["exit_time"]:
            events.append({"kind": "close", "ts": t["exit_time"], "trade": t})

    events.sort(key=lambda x: x["ts"])

    for ev in events:
        t = ev["trade"]
        sym = t["sym"]

        if ev["kind"] == "open":
            prices[sym] = t["entry_price"]
            broker.open_trade(
                sym=sym,
                qty=t["qty"],
                price=t["entry_price"],
                direction=t["direction"],
                ts=ev["ts"],
                prices_snapshot=dict(prices),
            )
        else:  # close
            prices[sym] = t["exit_price"]
            broker.close_trade(
                sym=sym,
                qty=t["qty"],
                exit_price=t["exit_price"],
                pnl=t["pnl"],
                fees=t["fees"],
                ts=ev["ts"],
                prices_snapshot=dict(prices),
            )

    stats = _broker_stats(broker)
    return broker, stats


def _broker_stats(broker: MarginBroker) -> dict:
    curve = np.array(broker.equity_curve)
    if len(curve) < 2:
        return {}

    peak = curve.max()
    dd_from_peak = (peak - curve.min()) / (peak + 1e-9)

    total_mc_loss = sum(
        mc["forced_loss"] for mc in broker.margin_calls if mc["forced_loss"] < 0
    )
    worst_mc = min(
        broker.margin_calls, key=lambda x: x["forced_loss"], default=None
    )

    return {
        "initial_equity": broker.initial_equity,
        "final_equity": float(curve[-1]),
        "peak_equity": float(broker.peak_equity),
        "total_return_pct": float((curve[-1] - curve[0]) / curve[0] * 100),
        "peak_return_pct":  float((broker.peak_equity - curve[0]) / curve[0] * 100),
        "max_drawdown_pct": float(dd_from_peak * 100),
        "margin_calls_count": len(broker.margin_calls),
        "margin_scale_events": broker.margin_scale_events,
        "total_mc_forced_loss": float(total_mc_loss),
        "worst_mc": worst_mc,
    }


# ---------------------------------------------------------------------------
# Synthetic world margin stress test
# ---------------------------------------------------------------------------

def load_arena_synth_prices(n_bars: int = 10_000, seed: int = 42) -> Dict[str, List[float]]:
    """
    Generate correlated ES/NQ/YM synthetic price series (mirrors arena_v8).
    """
    rng = np.random.default_rng(seed)
    corr = np.array([[1.00, 0.92, 0.88],
                     [0.92, 1.00, 0.85],
                     [0.88, 0.85, 1.00]])
    vols = np.array([0.15, 0.20, 0.14]) / np.sqrt(252 * 6.5)
    L = np.linalg.cholesky(np.outer(vols, vols) * corr)

    starts = {"ES": 4000.0, "NQ": 14000.0, "YM": 33000.0}
    regime_probs = [0.55, 0.15, 0.25, 0.05]
    regime_mus   = [0.0003, -0.0002, 0.00005, -0.001]
    regime_sigs  = [0.8, 1.2, 0.6, 2.5]

    regimes = []
    while len(regimes) < n_bars:
        r = rng.choice(4, p=regime_probs)
        regimes.extend([r] * int(rng.exponential(200)))
    regimes = regimes[:n_bars]

    prices = {s: [v] for s, v in starts.items()}
    for i in range(n_bars):
        r = regimes[i]
        mu = regime_mus[r]
        sig = regime_sigs[r]
        z = L @ rng.standard_normal(3)
        for j, sym in enumerate(["ES", "NQ", "YM"]):
            ret = mu + sig * vols[j] * z[j] * np.sqrt(252 * 6.5)
            prices[sym].append(prices[sym][-1] * (1 + ret))

    return {s: v[1:] for s, v in prices.items()}


def run_synth_margin_world(seed: int, n_bars: int = 5_000,
                            initial_equity: float = 1_000_000.0) -> dict:
    """
    Simulate a synthetic world where we use the margin broker with
    a simple momentum-based position sizer (mirrors arena logic).
    """
    prices_series = load_arena_synth_prices(n_bars, seed=seed)
    syms = ["ES", "NQ", "YM"]
    n = min(len(prices_series[s]) for s in syms)

    equity = initial_equity
    positions: Dict[str, float] = {s: 0.0 for s in syms}  # in fractions
    open_pos: Dict[str, dict] = {}  # sym -> {qty, entry_price}

    margin_calls: List[dict] = []
    scale_events = 0
    equity_curve = [equity]

    # Simple 20-bar momentum signal
    price_hist: Dict[str, List[float]] = {s: [] for s in syms}

    for bar in range(n):
        bar_prices = {s: prices_series[s][bar] for s in syms}

        # Mark to market
        for sym in syms:
            if sym in open_pos:
                pos = open_pos[sym]
                prev_p = pos.get("last_price", pos["entry_price"])
                curr_p = bar_prices[sym]
                pnl = (curr_p - prev_p) * pos["direction"] * pos["qty"] * CONTRACT_MULT[sym]
                equity += pnl
                open_pos[sym]["last_price"] = curr_p

        # Compute signals
        for sym in syms:
            price_hist[sym].append(bar_prices[sym])

        targets: Dict[str, float] = {}
        for sym in syms:
            ph = price_hist[sym]
            if len(ph) < 20:
                targets[sym] = 0.0
                continue
            momentum = (ph[-1] - ph[-20]) / (ph[-20] + 1e-9)
            vol = float(np.std(np.diff(ph[-20:])) / (ph[-1] + 1e-9))
            signal = np.tanh(momentum / (vol * 4 + 1e-9))
            targets[sym] = float(np.clip(signal, -0.5, 0.5))

        # Normalize portfolio exposure to max 1.0
        total_exp = sum(abs(v) for v in targets.values())
        if total_exp > 1.0:
            scale = 1.0 / total_exp
            targets = {s: v * scale for s, v in targets.items()}

        # Compute desired contracts
        desired_contracts: Dict[str, int] = {}
        for sym in syms:
            tgt_frac = targets[sym]
            price = bar_prices[sym]
            mult = CONTRACT_MULT[sym]
            n_contracts = int(round(abs(tgt_frac) * equity / (price * mult + 1e-9)))
            desired_contracts[sym] = n_contracts

        # Compute total margin if we open at desired
        total_margin = sum(
            desired_contracts[sym] * MARGIN_PER_CONTRACT[sym]
            for sym in syms
        )

        # Scale down if margin > 50% of equity
        if total_margin > equity * MARGIN_BUFFER and total_margin > 0:
            scale_factor = (equity * MARGIN_BUFFER) / total_margin
            desired_contracts = {
                s: int(v * scale_factor) for s, v in desired_contracts.items()
            }
            scale_events += 1

        # Check margin call: current margin > equity
        current_margin = sum(
            (open_pos[sym]["qty"] if sym in open_pos else 0) * MARGIN_PER_CONTRACT[sym]
            for sym in syms
        )
        if current_margin > equity and current_margin > 0:
            # Force liquidate
            for sym in list(open_pos.keys()):
                pos = open_pos[sym]
                fee = pos["qty"] * 4.0
                equity -= fee
            open_pos.clear()
            positions = {s: 0.0 for s in syms}
            margin_calls.append({
                "bar": bar,
                "equity": equity,
                "margin_required": current_margin,
            })
            equity_curve.append(equity)
            continue

        # Apply position changes
        for sym in syms:
            desired_qty = desired_contracts[sym]
            direction = 1 if targets[sym] >= 0 else -1

            curr_qty = open_pos[sym]["qty"] if sym in open_pos else 0
            if desired_qty == 0 and curr_qty > 0:
                # Close
                fee = curr_qty * 4.0
                equity -= fee
                del open_pos[sym]
            elif desired_qty > 0 and desired_qty != curr_qty:
                # Open or resize
                price = bar_prices[sym]
                fee = abs(desired_qty - curr_qty) * 4.0
                equity -= fee
                open_pos[sym] = {
                    "qty": desired_qty,
                    "entry_price": price,
                    "direction": direction,
                    "last_price": price,
                }

        equity_curve.append(equity)

    curve = np.array(equity_curve)
    peak = curve.max()
    dd = (peak - curve.min()) / (peak + 1e-9)

    return {
        "seed": seed,
        "n_bars": n_bars,
        "initial_equity": initial_equity,
        "final_equity": float(curve[-1]),
        "peak_equity": float(peak),
        "total_return_pct": float((curve[-1] - initial_equity) / initial_equity * 100),
        "max_drawdown_pct": float(dd * 100),
        "margin_calls_count": len(margin_calls),
        "margin_scale_events": scale_events,
        "margin_calls": margin_calls[:10],  # first 10
    }


# ---------------------------------------------------------------------------
# ASCII visualization
# ---------------------------------------------------------------------------

def _sparkline(values: List[float], width: int = 60) -> str:
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn + 1e-9
    # Downsample to width
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    return "".join(blocks[int((v - mn) / rng * 8)] for v in sampled)


def print_margin_report(
    broker: MarginBroker,
    stats: dict,
    label: str = "Seahorse",
) -> None:
    W = 72
    print()
    print("=" * W)
    print(f"  MARGIN SIMULATOR — {label}")
    print("=" * W)

    curve = broker.equity_curve
    print(f"\n  Equity curve ({len(curve)} bars):")
    print(f"  {_sparkline(curve)}")

    print(f"\n  Initial equity :  ${stats['initial_equity']:>15,.0f}")
    print(f"  Peak equity    :  ${stats['peak_equity']:>15,.0f}  "
          f"(+{stats['peak_return_pct']:.1f}%)")
    print(f"  Final equity   :  ${stats['final_equity']:>15,.0f}  "
          f"({stats['total_return_pct']:+.1f}%)")
    print(f"  Max drawdown   :  {stats['max_drawdown_pct']:.1f}%")
    print()
    print(f"  Margin calls      : {stats['margin_calls_count']}")
    print(f"  Margin scale evts : {stats['margin_scale_events']}")
    print(f"  Total MC loss     : ${stats['total_mc_forced_loss']:>12,.0f}")

    if stats.get("worst_mc"):
        wmc = stats["worst_mc"]
        print(f"\n  Worst margin call:")
        print(f"    Time           : {wmc.get('ts', 'N/A')}")
        print(f"    Equity before  : ${wmc.get('equity_before', 0):>12,.0f}")
        print(f"    Equity after   : ${wmc.get('equity_after', 0):>12,.0f}")
        print(f"    Margin req     : ${wmc.get('margin_required', 0):>12,.0f}")
        print(f"    Forced loss    : ${wmc.get('forced_loss', 0):>12,.0f}")

    if broker.margin_calls:
        print(f"\n  First 5 margin calls:")
        print(f"  {'Timestamp':<28} {'Equity Before':>14} {'Margin Req':>14} {'Forced Loss':>12}")
        print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*12}")
        for mc in broker.margin_calls[:5]:
            print(f"  {mc.get('ts','?'):<28} "
                  f"${mc.get('equity_before',0):>13,.0f} "
                  f"${mc.get('margin_required',0):>13,.0f} "
                  f"${mc.get('forced_loss',0):>11,.0f}")

    print("=" * W)


def print_synth_summary(results: List[dict]) -> None:
    W = 72
    print()
    print("=" * W)
    print("  MARGIN SIMULATOR — Synthetic Worlds")
    print("=" * W)
    print(f"\n  {'World':>5}  {'Return%':>8}  {'Peak%':>8}  {'MaxDD%':>8}  "
          f"{'MrgCalls':>8}  {'ScaleEvt':>8}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in results:
        print(f"  {r['seed']:>5}  "
              f"{r['total_return_pct']:>7.1f}%  "
              f"{r['peak_return_pct'] if 'peak_return_pct' in r else 0:>7.1f}%  "
              f"{r['max_drawdown_pct']:>7.1f}%  "
              f"{r['margin_calls_count']:>8}  "
              f"{r['margin_scale_events']:>8}")

    mc_counts = [r["margin_calls_count"] for r in results]
    returns = [r["total_return_pct"] for r in results]
    print(f"\n  Avg margin calls : {np.mean(mc_counts):.1f}")
    print(f"  Max margin calls : {max(mc_counts)}")
    print(f"  Avg total return : {np.mean(returns):.1f}%")
    print(f"  Worlds with MC   : {sum(1 for x in mc_counts if x > 0)}/{len(results)}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    seahorse_path = (
        "C:/Users/Matthew/Downloads/"
        "Hyper Active Red Orange Seahorse_trades.csv"
    )

    print("\n[margin_simulator] Loading Seahorse trades CSV...")
    trades = parse_seahorse_csv(seahorse_path)
    print(f"  Loaded {len(trades)} trades  "
          f"({min(t['entry_time'] for t in trades)} → "
          f"{max(t['exit_time'] for t in trades if t['exit_time'])})")

    print("\n[margin_simulator] Replaying with IB margin enforcement...")
    broker, stats = replay_with_margin(trades, initial_equity=1_000_000.0)
    print_margin_report(broker, stats, label="Seahorse (reconstructed with margin)")

    # Save seahorse results
    out_sea = {
        "label": "Seahorse_margin_replay",
        "stats": stats,
        "margin_calls": broker.margin_calls,
        "margin_scale_events": broker.margin_scale_events,
        "equity_curve_sample": broker.equity_curve[::max(1, len(broker.equity_curve)//200)],
    }
    sea_path = os.path.join(RESULTS_DIR, "margin_sim_seahorse.json")
    with open(sea_path, "w", encoding="utf-8") as f:
        json.dump(out_sea, f, indent=2, default=str)
    print(f"\n  Saved → {sea_path}")

    # Synthetic worlds
    print("\n[margin_simulator] Running 5 synthetic worlds with margin model...")
    synth_results = []
    for seed in range(5):
        r = run_synth_margin_world(seed=seed, n_bars=8_000, initial_equity=1_000_000.0)
        synth_results.append(r)
        print(f"  Seed {seed}: return={r['total_return_pct']:+.1f}%  "
              f"MC={r['margin_calls_count']}  scale={r['margin_scale_events']}")

    print_synth_summary(synth_results)

    synth_path = os.path.join(RESULTS_DIR, "margin_sim_synth.json")
    with open(synth_path, "w", encoding="utf-8") as f:
        json.dump(synth_results, f, indent=2, default=str)
    print(f"\n  Saved → {synth_path}")

    print("\n[margin_simulator] Done.")


if __name__ == "__main__":
    main()
