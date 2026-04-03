#!/usr/bin/env python
"""beta.py — SRFM spacetime classification in one pipe.

Usage:
    echo "4500 4505 0.005" | python tools/beta.py
    echo "4500 4550 0.005" | python tools/beta.py
    tail -100 data/NDX_hourly_poly.csv | python tools/beta.py --stream --cf 0.005
"""
import sys
import math
import io

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ANSI
def ansi(code, text, use_color):
    if not use_color:
        return text
    return f"\033[{code}m{text}\033[0m"

GREEN  = "32"
RED    = "31"
YELLOW = "33"
BOLD   = "1"


def compute_beta(prev, curr, cf):
    return abs(curr - prev) / prev / cf


def classify(beta):
    return "TIMELIKE" if beta <= 1.0 else "SPACELIKE"


def bh_delta(beta):
    """Black-hole mass delta direction from SRFM spacetime model."""
    if beta <= 1.0:
        # BH accretes: mass increases proportional to proximity to boundary
        delta = round(beta * 0.1, 3)
        return f"BH accretes +{delta:.3f}"
    else:
        # BH decays: Hawking-style evaporation scaled by how spacelike
        decay = round(1.0 / beta, 3)
        return f"BH decays ×{decay:.3f}"


def format_single(beta, use_color):
    cls = classify(beta)
    delta_str = bh_delta(beta)
    check = "[ok]" if cls == "TIMELIKE" else "[x]"
    color = GREEN if cls == "TIMELIKE" else RED
    cls_colored = ansi(color, cls, use_color)
    check_colored = ansi(color, check, use_color)
    beta_str = ansi(BOLD, f"b={beta:.3f}", use_color)
    return f"{beta_str}  {cls_colored}  {check_colored}  ({delta_str})"


def stream_mode(cf, use_color):
    """Read CSV lines (timestamp,close) from stdin, output running β + cumulative BH mass."""
    prev_ts = None
    prev_close = None
    bh_mass = 1.0
    ctl_count = 0  # consecutive timelike count

    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Try to parse timestamp,close  (skip header-like lines)
        parts = line.split(',')
        if len(parts) < 2:
            continue
        ts_str = parts[0].strip()
        try:
            close = float(parts[1].strip())
        except ValueError:
            continue  # skip header

        if prev_close is None:
            prev_ts = ts_str
            prev_close = close
            continue

        beta = compute_beta(prev_close, close, cf)
        cls = classify(beta)

        if cls == "TIMELIKE":
            bh_mass *= (1.0 + beta * 0.05)
            ctl_count += 1
            bh_state = "ACTIVE"
        else:
            bh_mass *= (1.0 - 0.02)  # slight decay
            ctl_count = 0
            bh_state = "DECAYING"

        color = GREEN if cls == "TIMELIKE" else RED
        cls_pad = f"{cls:<9}"
        cls_colored = ansi(color, cls_pad, use_color)
        bh_colored = ansi(YELLOW if bh_state == "DECAYING" else GREEN, f"BH={bh_state}", use_color)

        print(f"{ts_str:<18}  β={beta:.3f}  {cls_colored}  mass={bh_mass:.3f}  CTL={ctl_count}  {bh_colored}")
        sys.stdout.flush()

        prev_ts = ts_str
        prev_close = close


def main():
    use_color = sys.stdout.isatty()
    stream = "--stream" in sys.argv

    # Parse --cf
    cf = 0.005
    if "--cf" in sys.argv:
        idx = sys.argv.index("--cf")
        cf = float(sys.argv[idx + 1])

    if stream:
        stream_mode(cf, use_color)
        return

    # Single or argv mode
    # Check if values on argv (not flags)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) >= 3:
        prev_close = float(args[0])
        curr_close = float(args[1])
        cf = float(args[2])
    elif len(args) == 2:
        prev_close = float(args[0])
        curr_close = float(args[1])
        # cf already set
    else:
        # Read from stdin
        line = sys.stdin.read().strip()
        parts = line.split()
        if len(parts) < 3:
            print("Usage: echo 'prev_close curr_close cf' | python tools/beta.py", file=sys.stderr)
            sys.exit(1)
        prev_close = float(parts[0])
        curr_close = float(parts[1])
        cf = float(parts[2])

    beta = compute_beta(prev_close, curr_close, cf)
    print(format_single(beta, use_color))


if __name__ == "__main__":
    main()
