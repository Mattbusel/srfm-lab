"""
tools/evaluate_rl_policy.py
============================
Evaluates the trained Q-table exit policy against held-out test trades.

Usage:
  python tools/evaluate_rl_policy.py [--qtable config/rl_exit_qtable.json]
                                     [--db execution/live_trades.db]
                                     [--output rl_evaluation.html]
                                     [--synthetic 300]

Produces:
  - Console report with metric tables
  - Optional HTML report with Plotly charts (--output)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

_REPO_ROOT   = Path(__file__).parents[1]
_QTABLE_PATH = _REPO_ROOT / "config" / "rl_exit_qtable.json"
_DB_PATH     = _REPO_ROOT / "execution" / "live_trades.db"

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("evaluate_rl_policy")


# ---------------------------------------------------------------------------
# Re-use the discretizer from train_rl_exit without circular import
# ---------------------------------------------------------------------------

class _StateDiscretizer:
    PNL_EDGES  = [-0.02, -0.005, 0.005, 0.02]
    BARS_EDGES = [3, 8, 16, 32]
    MASS_EDGES = [0.5, 1.0, 1.92, 3.0]
    ATR_EDGES  = [0.5, 0.8, 1.2, 2.0]

    @staticmethod
    def _bin(value: float, edges: list) -> int:
        for i, edge in enumerate(edges):
            if value < edge:
                return i
        return len(edges)

    def state_key(
        self,
        pnl_pct:   float,
        bars_held: int,
        bh_mass:   float,
        bh_active: bool,
        atr_ratio: float,
    ) -> str:
        f0 = self._bin(pnl_pct,   self.PNL_EDGES)
        f1 = self._bin(bars_held, self.BARS_EDGES)
        f2 = self._bin(bh_mass,   self.MASS_EDGES)
        f3 = 4 if bh_active else 0
        f4 = self._bin(atr_ratio, self.ATR_EDGES)
        return f"{f0},{f1},{f2},{f3},{f4}"


_DISC = _StateDiscretizer()


# ---------------------------------------------------------------------------
# Episode result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Outcome of running a policy through one trade episode."""
    exit_bar:    int        # bar index at which the policy exited
    exit_pnl:   float      # P&L at exit
    max_pnl:    float      # maximum P&L seen during episode
    min_pnl:    float      # minimum P&L seen during episode
    n_bars:     int        # total bars in episode
    hit_stoploss: bool     # whether P&L went below -2% threshold
    bars_saved:  int       # n_bars - exit_bar (bars saved vs hold-to-end)
    policy_name: str       = "unknown"


# ---------------------------------------------------------------------------
# Policy definitions
# ---------------------------------------------------------------------------

def policy_rl(
    qtable: dict[str, list[float]],
    pnl_pct: float,
    bars_held: int,
    bh_mass: float,
    bh_active: bool,
    atr_ratio: float,
) -> bool:
    """
    RL Q-table policy.
    Returns True (exit) when Q_exit > Q_hold.
    Hard stop at -3%.
    """
    if pnl_pct < -0.03:
        return True
    key = _DISC.state_key(pnl_pct, bars_held, bh_mass, bh_active, atr_ratio)
    qs  = qtable.get(key)
    if qs is not None and len(qs) == 2:
        return float(qs[1]) > float(qs[0])
    # Heuristic fallback (mirrors live_trader)
    if bh_active and pnl_pct > 0.005:
        return False
    if not bh_active and bars_held > 16:
        return True
    if pnl_pct < -0.015:
        return True
    return False


def policy_always_hold(
    pnl_pct: float, bars_held: int, bh_mass: float,
    bh_active: bool, atr_ratio: float
) -> bool:
    """Never exit until the episode ends (hold-to-end)."""
    return False


def policy_exit_at_1pct(
    pnl_pct: float, bars_held: int, bh_mass: float,
    bh_active: bool, atr_ratio: float
) -> bool:
    """Take profit immediately when P&L >= +1%."""
    return pnl_pct >= 0.01


def policy_time_based(
    pnl_pct: float, bars_held: int, bh_mass: float,
    bh_active: bool, atr_ratio: float
) -> bool:
    """Exit after 8 bars if near breakeven (P&L in [-0.5%, +0.5%]) or earlier if losing."""
    if bars_held >= 8 and abs(pnl_pct) < 0.005:
        return True
    if pnl_pct < -0.015:
        return True
    return False


# ---------------------------------------------------------------------------
# PolicyEvaluator
# ---------------------------------------------------------------------------

class PolicyEvaluator:
    """
    Loads a Q-table and evaluates multiple policies on a test episode set.

    Metrics computed per policy:
      - mean_exit_pnl        : average P&L at exit bar
      - mean_max_pnl         : average max possible P&L in episode
      - pnl_capture_rate     : mean_exit_pnl / mean_max_pnl (>1 = beat avg max)
      - stoploss_hit_rate    : fraction of episodes where P&L hit <= -2%
      - stoploss_avoided_rate: fraction of stoploss episodes where policy exited before
      - mean_bars_saved      : average bars saved vs always-hold
      - mean_exit_bar        : average bar at which policy exits
    """

    STOPLOSS_THRESHOLD = -0.02

    def __init__(self, qtable_path: Path | str = _QTABLE_PATH) -> None:
        self._qtable: dict[str, list[float]] = {}
        self._qtable_path = Path(qtable_path)
        self._load_qtable()

    def _load_qtable(self) -> None:
        if self._qtable_path.exists():
            self._qtable = json.loads(self._qtable_path.read_text())
            log.info("Loaded Q-table: %s (%d states)", self._qtable_path, len(self._qtable))
        else:
            log.warning("Q-table not found at %s -- RL policy will use heuristic fallback",
                        self._qtable_path)

    def _run_episode(
        self,
        episode: list[dict],
        policy_fn: Callable,
        policy_name: str = "policy",
    ) -> EpisodeResult:
        """
        Step through an episode bar by bar, calling policy_fn at each step.
        policy_fn(pnl_pct, bars_held, bh_mass, bh_active, atr_ratio) -> bool
        """
        n_bars   = len(episode)
        max_pnl  = -math.inf
        min_pnl  =  math.inf
        exit_bar = n_bars - 1
        exit_pnl = episode[-1]["pnl_pct"] if episode else 0.0

        for i, bar in enumerate(episode):
            pnl  = bar["pnl_pct"]
            max_pnl = max(max_pnl, pnl)
            min_pnl = min(min_pnl, pnl)
            should_exit = policy_fn(
                pnl,
                bar["bars_held"],
                bar["bh_mass"],
                bar["bh_active"],
                bar["atr_ratio"],
            )
            if should_exit or i == n_bars - 1:
                exit_bar = i
                exit_pnl = pnl
                break

        hit_stoploss = min_pnl <= self.STOPLOSS_THRESHOLD

        return EpisodeResult(
            exit_bar=    exit_bar,
            exit_pnl=    exit_pnl,
            max_pnl=     max_pnl,
            min_pnl=     min_pnl,
            n_bars=      n_bars,
            hit_stoploss= hit_stoploss,
            bars_saved=  (n_bars - 1) - exit_bar,
            policy_name= policy_name,
        )

    def _compute_metrics(self, results: list[EpisodeResult]) -> dict:
        """Aggregate EpisodeResult list into a metrics dict."""
        if not results:
            return {}

        exit_pnls    = [r.exit_pnl    for r in results]
        max_pnls     = [r.max_pnl     for r in results]
        bars_saved   = [r.bars_saved  for r in results]
        exit_bars    = [r.exit_bar    for r in results]
        stoploss_eps = [r for r in results if r.hit_stoploss]

        # Among episodes that would hit stoploss, how many did the policy exit before?
        stoploss_avoided = sum(
            1 for r in stoploss_eps if r.exit_pnl > self.STOPLOSS_THRESHOLD
        ) if stoploss_eps else 0

        mean_max = float(np.mean(max_pnls)) if max_pnls else 0.0

        return {
            "n_episodes":           len(results),
            "mean_exit_pnl":        round(float(np.mean(exit_pnls)), 6),
            "median_exit_pnl":      round(float(np.median(exit_pnls)), 6),
            "std_exit_pnl":         round(float(np.std(exit_pnls)), 6),
            "mean_max_pnl":         round(mean_max, 6),
            "pnl_capture_rate":     round(
                float(np.mean(exit_pnls)) / (abs(mean_max) + 1e-9), 4
            ),
            "stoploss_hit_rate":    round(len(stoploss_eps) / len(results), 4),
            "stoploss_avoided_rate": round(
                stoploss_avoided / max(len(stoploss_eps), 1), 4
            ),
            "mean_bars_saved":      round(float(np.mean(bars_saved)), 2),
            "mean_exit_bar":        round(float(np.mean(exit_bars)), 2),
            "pct_profitable":       round(
                sum(1 for r in results if r.exit_pnl > 0) / len(results), 4
            ),
        }

    def evaluate_policy(
        self,
        episodes:    list[list[dict]],
        policy_fn:   Callable,
        policy_name: str = "policy",
    ) -> dict:
        """Run policy on all episodes and return metrics dict."""
        results = [
            self._run_episode(ep, policy_fn, policy_name)
            for ep in episodes
        ]
        metrics = self._compute_metrics(results)
        metrics["policy_name"] = policy_name
        return metrics

    def compare_policies(
        self,
        test_episodes: list[list[dict]],
        qtable_policy: Optional[dict[str, list[float]]] = None,
    ) -> dict:
        """
        Evaluate RL Q-table policy against three baselines on the same test set.

        Parameters
        ----------
        test_episodes : list of episode histories
        qtable_policy : Q-table dict (defaults to self._qtable)

        Returns
        -------
        dict mapping policy_name -> metrics_dict, plus a "ranking" list
        """
        qtable = qtable_policy if qtable_policy is not None else self._qtable

        def rl_fn(pnl, bars, mass, active, atr):
            return policy_rl(qtable, pnl, bars, mass, active, atr)

        policies = {
            "rl_qtable":   rl_fn,
            "always_hold": policy_always_hold,
            "exit_at_1pct": policy_exit_at_1pct,
            "time_based":   policy_time_based,
        }

        all_metrics: dict[str, dict] = {}
        for name, fn in policies.items():
            m = self.evaluate_policy(test_episodes, fn, policy_name=name)
            all_metrics[name] = m
            log.info(
                "Policy %-15s: mean_pnl=%+.4f  capture=%.3f  stoploss_avoided=%.2f  "
                "bars_saved=%.1f  pct_profitable=%.1f%%",
                name,
                m["mean_exit_pnl"],
                m["pnl_capture_rate"],
                m["stoploss_avoided_rate"],
                m["mean_bars_saved"],
                m["pct_profitable"] * 100,
            )

        # Rank by mean_exit_pnl
        ranking = sorted(
            all_metrics.keys(),
            key=lambda k: all_metrics[k]["mean_exit_pnl"],
            reverse=True,
        )
        all_metrics["_ranking_by_mean_pnl"] = ranking

        # Compute RL improvement vs always_hold baseline
        rl_pnl   = all_metrics["rl_qtable"]["mean_exit_pnl"]
        hold_pnl = all_metrics["always_hold"]["mean_exit_pnl"]
        all_metrics["_rl_improvement_vs_hold"] = round(rl_pnl - hold_pnl, 6)

        return all_metrics

    def value_by_state(self) -> dict[str, dict]:
        """
        Compute policy value (Q_exit - Q_hold) for every state in the Q-table.
        Returns dict: state_key_str -> {"q_hold": ..., "q_exit": ..., "preference": ...}
        """
        out: dict[str, dict] = {}
        for key_str, qs in self._qtable.items():
            if len(qs) != 2:
                continue
            out[key_str] = {
                "q_hold":   round(qs[0], 6),
                "q_exit":   round(qs[1], 6),
                "q_diff":   round(qs[1] - qs[0], 6),
                "preference": "EXIT" if qs[1] > qs[0] else "HOLD",
            }
        return out


# ---------------------------------------------------------------------------
# HTML report generator
# ---------------------------------------------------------------------------

def generate_rl_report(
    evaluator: PolicyEvaluator,
    test_episodes: list[list[dict]],
    output: str = "rl_evaluation.html",
) -> None:
    """
    Generate an HTML report with Plotly charts of policy comparison and
    Q-table value landscape.

    Falls back to a text report if plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        _PLOTLY = True
    except ImportError:
        _PLOTLY = False
        log.warning("plotly not installed -- generating text report instead")

    comparison = evaluator.compare_policies(test_episodes)
    state_vals  = evaluator.value_by_state()

    if not _PLOTLY:
        _write_text_report(comparison, state_vals, output.replace(".html", ".txt"))
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    policy_names = [k for k in comparison if not k.startswith("_")]
    metrics_keys = ["mean_exit_pnl", "pnl_capture_rate", "stoploss_avoided_rate",
                    "mean_bars_saved", "pct_profitable"]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Mean Exit P&L by Policy",
            "P&L Capture Rate by Policy",
            "Stop-Loss Avoidance Rate",
            "Mean Bars Saved vs Hold-to-End",
            "Q-value Spread (EXIT - HOLD) Distribution",
            "Policy Win Rates (% Profitable)",
        ],
    )

    colors = {"rl_qtable": "#2ecc71", "always_hold": "#e74c3c",
              "exit_at_1pct": "#3498db", "time_based": "#f39c12"}

    # Chart 1: Mean exit P&L
    fig.add_trace(go.Bar(
        x=policy_names,
        y=[comparison[p]["mean_exit_pnl"] for p in policy_names],
        marker_color=[colors.get(p, "#95a5a6") for p in policy_names],
        name="Mean Exit P&L",
        showlegend=False,
    ), row=1, col=1)

    # Chart 2: P&L capture rate
    fig.add_trace(go.Bar(
        x=policy_names,
        y=[comparison[p]["pnl_capture_rate"] for p in policy_names],
        marker_color=[colors.get(p, "#95a5a6") for p in policy_names],
        name="Capture Rate",
        showlegend=False,
    ), row=1, col=2)

    # Chart 3: Stoploss avoidance
    fig.add_trace(go.Bar(
        x=policy_names,
        y=[comparison[p]["stoploss_avoided_rate"] for p in policy_names],
        marker_color=[colors.get(p, "#95a5a6") for p in policy_names],
        name="SL Avoid Rate",
        showlegend=False,
    ), row=2, col=1)

    # Chart 4: Bars saved
    fig.add_trace(go.Bar(
        x=policy_names,
        y=[comparison[p]["mean_bars_saved"] for p in policy_names],
        marker_color=[colors.get(p, "#95a5a6") for p in policy_names],
        name="Bars Saved",
        showlegend=False,
    ), row=2, col=2)

    # Chart 5: Q-diff distribution
    q_diffs = [v["q_diff"] for v in state_vals.values()]
    if q_diffs:
        fig.add_trace(go.Histogram(
            x=q_diffs,
            nbinsx=40,
            marker_color="#9b59b6",
            name="Q-diff distribution",
            showlegend=False,
        ), row=3, col=1)

    # Chart 6: Win rate
    fig.add_trace(go.Bar(
        x=policy_names,
        y=[comparison[p]["pct_profitable"] * 100 for p in policy_names],
        marker_color=[colors.get(p, "#95a5a6") for p in policy_names],
        name="% Profitable",
        showlegend=False,
    ), row=3, col=2)

    fig.update_layout(
        title_text="RL Exit Policy Evaluation Report",
        height=900,
        template="plotly_dark",
    )

    # Summary table as HTML
    ranking  = comparison.get("_ranking_by_mean_pnl", policy_names)
    rl_impr  = comparison.get("_rl_improvement_vs_hold", 0.0)
    table_rows = ""
    for p in ranking:
        m = comparison[p]
        table_rows += (
            f"<tr><td>{p}</td>"
            f"<td>{m['mean_exit_pnl']:+.4f}</td>"
            f"<td>{m['pnl_capture_rate']:.3f}</td>"
            f"<td>{m['stoploss_avoided_rate']:.2f}</td>"
            f"<td>{m['mean_bars_saved']:.1f}</td>"
            f"<td>{m['pct_profitable']*100:.1f}%</td></tr>\n"
        )

    html_table = f"""
    <table border='1' style='border-collapse:collapse;margin:20px auto;font-family:monospace'>
    <thead><tr>
      <th>Policy</th><th>Mean Exit P&L</th><th>Capture Rate</th>
      <th>SL Avoided</th><th>Bars Saved</th><th>Win Rate</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
    </table>
    <p style='text-align:center;font-family:monospace'>
      RL improvement vs always-hold: <b>{rl_impr:+.4f}</b>
    </p>
    """

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    full_html = f"""<!DOCTYPE html>
<html>
<head><meta charset='utf-8'><title>RL Exit Policy Evaluation</title>
<style>body{{background:#1a1a2e;color:#eee;}}
table td,table th{{padding:6px 12px;}}
thead{{background:#2c3e50;}}
tr:nth-child(even){{background:#2c3e50;}}
</style></head>
<body>
<h1 style='text-align:center;font-family:sans-serif'>RL Exit Policy Evaluation</h1>
{html_table}
{chart_html}
</body></html>"""

    Path(output).write_text(full_html, encoding="utf-8")
    log.info("HTML report saved: %s", output)


def _write_text_report(
    comparison: dict,
    state_vals: dict,
    output: str,
) -> None:
    """Fallback text report when plotly is unavailable."""
    lines: list[str] = ["RL Exit Policy Evaluation Report", "=" * 60, ""]

    ranking = comparison.get("_ranking_by_mean_pnl", [])
    for p in ranking:
        if p.startswith("_"):
            continue
        m = comparison[p]
        lines.append(f"Policy: {p}")
        for k, v in m.items():
            if k != "policy_name":
                lines.append(f"  {k}: {v}")
        lines.append("")

    rl_impr = comparison.get("_rl_improvement_vs_hold", 0.0)
    lines.append(f"RL vs always-hold improvement: {rl_impr:+.6f}")
    lines.append("")
    lines.append(f"Q-table states: {len(state_vals)}")

    q_diffs = sorted(v["q_diff"] for v in state_vals.values())
    if q_diffs:
        lines.append(f"Q-diff median: {np.median(q_diffs):.4f}")
        lines.append(f"Q-diff range: [{min(q_diffs):.4f}, {max(q_diffs):.4f}]")

    Path(output).write_text("\n".join(lines), encoding="utf-8")
    log.info("Text report saved: %s", output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate trained RL exit policy against baselines"
    )
    p.add_argument("--qtable",    type=str, default=str(_QTABLE_PATH))
    p.add_argument("--db",        type=str, default=str(_DB_PATH))
    p.add_argument("--output",    type=str, default="rl_evaluation.html")
    p.add_argument("--synthetic", type=int, default=300,
                   help="Number of synthetic test episodes to generate")
    p.add_argument("--seed",      type=int, default=99)
    p.add_argument("--no-report", action="store_true",
                   help="Skip HTML report generation (print metrics only)")
    return p.parse_args()


def _generate_test_episodes(n: int, seed: int) -> list[list[dict]]:
    """
    Generate test episodes using the same GBM-regime logic as TradeDataLoader
    but with a separate seed so test set is independent of training set.
    """
    # Import the loader from train_rl_exit if available, else duplicate minimally
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from train_rl_exit import TradeDataLoader
        rng_state = random.getstate()
        np_state  = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
        loader = TradeDataLoader()
        eps    = loader.generate_synthetic_episodes(n)
        random.setstate(rng_state)
        np.random.set_state(np_state)
        return eps
    except ImportError:
        log.warning("Could not import TradeDataLoader -- generating minimal test set")
        rng = np.random.default_rng(seed)
        episodes = []
        for _ in range(n):
            n_bars = int(rng.integers(5, 30))
            pnls   = np.cumsum(rng.normal(0, 0.005, n_bars))
            bars   = []
            for i in range(n_bars):
                mass = max(0.0, 1.5 - 0.04 * i)
                bars.append({
                    "pnl_pct":   float(pnls[i]),
                    "bars_held": i,
                    "bh_mass":   float(mass),
                    "bh_active": mass > 1.0,
                    "atr_ratio": float(1.0 + rng.normal(0, 0.15)),
                })
            episodes.append(bars)
        return episodes


def main() -> None:
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    evaluator = PolicyEvaluator(qtable_path=args.qtable)

    log.info("Generating %d test episodes (seed=%d)...", args.synthetic, args.seed)
    test_eps = _generate_test_episodes(args.synthetic, seed=args.seed)
    log.info("Running policy comparison on %d episodes...", len(test_eps))

    comparison = evaluator.compare_policies(test_eps)

    # Print summary
    print("\n" + "=" * 70)
    print("  RL EXIT POLICY EVALUATION SUMMARY")
    print("=" * 70)
    ranking = comparison.get("_ranking_by_mean_pnl", [])
    for rank, pname in enumerate(ranking, 1):
        if pname.startswith("_"):
            continue
        m = comparison[pname]
        print(
            f"  #{rank} {pname:<16} "
            f"mean_pnl={m['mean_exit_pnl']:+.4f}  "
            f"capture={m['pnl_capture_rate']:.3f}  "
            f"sl_avoid={m['stoploss_avoided_rate']:.2f}  "
            f"bars_saved={m['mean_bars_saved']:.1f}  "
            f"win%={m['pct_profitable']*100:.1f}"
        )
    rl_impr = comparison.get("_rl_improvement_vs_hold", 0.0)
    print(f"\n  RL vs always-hold P&L improvement: {rl_impr:+.6f}")
    print("=" * 70 + "\n")

    if not args.no_report:
        generate_rl_report(evaluator, test_eps, output=args.output)


if __name__ == "__main__":
    main()
