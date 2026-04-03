"""
pid_sizer.py — PID controller for dynamic position sizing.

The controller treats drawdown as the process variable and targets a
maximum acceptable drawdown (setpoint). When drawdown exceeds the target
the multiplier is reduced; when the portfolio is recovering or performing
well the multiplier can rise above 1.0 (up to max_mult).

Standalone usage (reads bar_number,portfolio_value CSV from stdin):
    echo "0,1000000\\n100,950000\\n200,900000" | python tools/pid_sizer.py

Output columns: bar,pv,drawdown,P,I,D,multiplier

Simulation mode (built-in test scenarios):
    python tools/pid_sizer.py --simulate drawdown
    python tools/pid_sizer.py --simulate choppy
    python tools/pid_sizer.py --simulate trending
"""

from __future__ import annotations

import sys
import math
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Core PID controller
# ---------------------------------------------------------------------------

class PIDSizer:
    """
    PID controller for dynamic position sizing.

    Reduces position size when drawdown exceeds target.
    Allows increased size when performance is strong.

    Args:
        target_dd: target max drawdown fraction (default 0.15 = 15%)
        kp: proportional gain (default 2.0)
        ki: integral gain (default 0.1)
        kd: derivative gain (default 0.5)
        min_mult: minimum size multiplier (default 0.3)
        max_mult: maximum size multiplier (default 1.5)
    """

    INTEGRAL_CLAMP = 5.0   # anti-windup clamp for integral term

    def __init__(
        self,
        target_dd: float = 0.15,
        kp: float = 2.0,
        ki: float = 0.1,
        kd: float = 0.5,
        min_mult: float = 0.3,
        max_mult: float = 1.5,
    ) -> None:
        self.target_dd = target_dd
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_mult = min_mult
        self.max_mult = max_mult

        # Internal state
        self._integral: float = 0.0
        self._prev_error: Optional[float] = None
        self._multiplier: float = 1.0
        self._p: float = 0.0
        self._i: float = 0.0
        self._d: float = 0.0

    # ------------------------------------------------------------------
    def update(
        self,
        current_dd: float,
        peak: float = 0.0,
        portfolio_value: float = 0.0,
    ) -> float:
        """
        Compute the new position-size multiplier.

        Args:
            current_dd: current drawdown as a fraction in [0, 1].
                        0.0 = no drawdown, 0.30 = 30% below peak.
            peak: (optional) unused directly; kept for API symmetry.
            portfolio_value: (optional) unused directly.

        Returns:
            Multiplier to apply to raw position size, clamped to
            [min_mult, max_mult].
        """
        # Error: positive when drawdown exceeds target (bad), negative when
        # we are performing better than target (good).
        error = current_dd - self.target_dd

        # --- Proportional ---
        self._p = self.kp * error

        # --- Integral (with anti-windup clamp) ---
        self._integral += error
        self._integral = max(
            -self.INTEGRAL_CLAMP,
            min(self.INTEGRAL_CLAMP, self._integral),
        )
        self._i = self.ki * self._integral

        # --- Derivative ---
        if self._prev_error is None:
            self._d = 0.0
        else:
            self._d = self.kd * (error - self._prev_error)
        self._prev_error = error

        # PID output: positive output → reduce size, negative → allow increase
        pid_output = self._p + self._i + self._d

        # Map to multiplier: 1.0 at setpoint, decreases above, increases below
        raw = 1.0 - pid_output
        self._multiplier = max(self.min_mult, min(self.max_mult, raw))
        return self._multiplier

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset all internal state to initial conditions."""
        self._integral = 0.0
        self._prev_error = None
        self._multiplier = 1.0
        self._p = 0.0
        self._i = 0.0
        self._d = 0.0

    # ------------------------------------------------------------------
    def state(self) -> Dict[str, float]:
        """Return current internal state as a dictionary."""
        return {
            "P": round(self._p, 6),
            "I": round(self._i, 6),
            "D": round(self._d, 6),
            "integral": round(self._integral, 6),
            "multiplier": round(self._multiplier, 6),
            "target_dd": self.target_dd,
        }

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        s = self.state()
        return (
            f"PIDSizer(target_dd={self.target_dd}, kp={self.kp}, ki={self.ki}, "
            f"kd={self.kd}, mult={s['multiplier']:.3f}, "
            f"P={s['P']:.4f}, I={s['I']:.4f}, D={s['D']:.4f})"
        )


# ---------------------------------------------------------------------------
# Utility: compute drawdown from a running portfolio value
# ---------------------------------------------------------------------------

def _rolling_drawdown(pv: float, peak: float) -> tuple[float, float]:
    """Return (drawdown_fraction, new_peak)."""
    new_peak = max(peak, pv)
    dd = (new_peak - pv) / new_peak if new_peak > 0 else 0.0
    return dd, new_peak


# ---------------------------------------------------------------------------
# Simulation scenarios
# ---------------------------------------------------------------------------

def simulate_pid(scenario: str = "drawdown") -> None:
    """
    Test scenarios: 'drawdown', 'choppy', 'trending'.

    Prints a timestep-by-timestep table and an ASCII chart of the
    multiplier over time.
    """
    n = 200
    pid = PIDSizer()

    # Build synthetic portfolio-value series
    if scenario == "drawdown":
        # Gradual 25% drawdown then recovery
        pvs = []
        pv = 1_000_000.0
        for i in range(n):
            if i < 100:
                pv *= (1 - 0.0025)   # steady drawdown
            else:
                pv *= (1 + 0.003)    # recovery
            pvs.append(pv)
    elif scenario == "choppy":
        # Rapid oscillation around -10%
        import math as _math
        pvs = []
        base = 900_000.0
        for i in range(n):
            oscillation = 50_000 * _math.sin(i * 0.4)
            pvs.append(base + oscillation)
    elif scenario == "trending":
        # Steady growth with occasional dips
        pvs = []
        pv = 1_000_000.0
        for i in range(n):
            if i % 30 == 0 and i > 0:
                pv *= 0.94   # periodic dip
            else:
                pv *= 1.0015  # steady uptrend
            pvs.append(pv)
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}. Use 'drawdown', 'choppy', or 'trending'.")

    print(f"\n=== PID Sizer Simulation: {scenario.upper()} ===\n")
    print(f"{'Bar':>4}  {'PV':>12}  {'DD%':>7}  {'P':>8}  {'I':>8}  {'D':>8}  {'Mult':>6}")
    print("-" * 65)

    peak = 0.0
    multipliers = []

    for i, pv in enumerate(pvs):
        dd, peak = _rolling_drawdown(pv, peak)
        mult = pid.update(dd)
        s = pid.state()
        multipliers.append(mult)
        if i % 10 == 0:
            print(f"{i:>4}  {pv:>12,.0f}  {dd*100:>6.2f}%  {s['P']:>8.4f}  {s['I']:>8.4f}  {s['D']:>8.4f}  {mult:>6.3f}")

    # ASCII chart
    _ascii_chart(multipliers, title=f"Position Multiplier — {scenario}", lo=0.0, hi=1.6)


def _ascii_chart(values: list, title: str = "", lo: float = 0.0, hi: float = 2.0, width: int = 80, height: int = 20) -> None:
    """Render a simple ASCII line chart."""
    print(f"\n  {title}")
    print(f"  {hi:.1f} |")

    # Downsample to width
    n = len(values)
    step = max(1, n // width)
    sampled = [values[i] for i in range(0, n, step)][:width]

    rows = []
    for row_idx in range(height):
        threshold = hi - (hi - lo) * row_idx / (height - 1)
        line = ""
        for v in sampled:
            line += "*" if abs(v - threshold) < (hi - lo) / height else " "
        rows.append((threshold, line))

    for threshold, line in rows:
        label = f"{threshold:>5.2f}" if row_idx % 5 == 0 else "     "
        print(f"  {threshold:>5.2f} |{line}")

    print(f"  {'':>5} +{'-'*len(sampled)}")
    print(f"       0{' '*(len(sampled)//2 - 3)}bar{' '*(len(sampled)//2 - 3)}{len(values)-1}")
    print(f"\n  min={min(values):.3f}  max={max(values):.3f}  final={values[-1]:.3f}\n")


# ---------------------------------------------------------------------------
# CLI — stdin CSV mode
# ---------------------------------------------------------------------------

def _cli_stdin() -> None:
    """Read bar_number,portfolio_value from stdin, emit PID output per bar."""
    pid = PIDSizer()
    peak = 0.0
    print("bar,pv,drawdown,P,I,D,multiplier")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            bar = int(parts[0])
            pv = float(parts[1])
        except ValueError:
            continue
        dd, peak = _rolling_drawdown(pv, peak)
        mult = pid.update(dd, peak, pv)
        s = pid.state()
        print(f"{bar},{pv:.2f},{dd:.6f},{s['P']:.6f},{s['I']:.6f},{s['D']:.6f},{mult:.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PID position sizer. Without --simulate, reads bar,pv CSV from stdin."
    )
    parser.add_argument(
        "--simulate",
        metavar="SCENARIO",
        nargs="?",
        const="drawdown",
        help="Run a built-in simulation. Scenarios: drawdown, choppy, trending (default: drawdown)",
    )
    args = parser.parse_args()

    if args.simulate is not None:
        simulate_pid(args.simulate)
    else:
        _cli_stdin()
