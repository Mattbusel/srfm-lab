#!/usr/bin/env python
"""heat.py — Any JSON with a numeric score per experiment → ANSI heatmap.

Usage:
    cat results/v2_experiments.json | python tools/heat.py
    cat results/v2_experiments.json | python tools/heat.py --field arena_sharpe
    cat results/tournament/leaderboard.csv | python tools/heat.py --field sharpe --top 20
    python tools/heat.py results/v2_experiments.json
"""
import sys
import json
import csv
import io
import os
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ANSI background colors
BG_GREEN      = "\033[42m"
BG_DKGREEN    = "\033[102m"
BG_YELLOW     = "\033[43m"
BG_RED        = "\033[41m"
BG_DKRED      = "\033[101m"
BG_RESET      = "\033[0m"
BOLD          = "\033[1m"
DIM           = "\033[2m"
RESET         = "\033[0m"

SCORE_FIELDS = ["combined_score", "score", "sharpe", "arena_sharpe", "net_pnl", "pnl"]
LABEL_FIELDS = ["exp", "flags", "name", "label", "experiment"]


def ansi(bg, text, use_color):
    if not use_color:
        return text
    return f"{bg}{text}{BG_RESET}"


def score_to_block(score, use_color):
    """Map score to block character + color."""
    if score > 0.3:
        char = "████"
        bg   = BG_GREEN
    elif score > 0.1:
        char = "▓▓▓▓"
        bg   = BG_DKGREEN
    elif score > -0.1:
        char = "▒▒▒▒"
        bg   = BG_YELLOW
    elif score > -0.3:
        char = "░░░░"
        bg   = BG_RED
    else:
        char = "····"
        bg   = BG_DKRED
    return ansi(bg, char, use_color)


def score_to_bar(score, max_abs, width=20, use_color=True):
    """Horizontal bar for ranked list."""
    if max_abs == 0:
        filled = 0
    else:
        filled = int(abs(score) / max_abs * width)
    bar = "█" * filled + " " * (width - filled)
    if score > 0.1:
        bg = BG_GREEN
    elif score > -0.1:
        bg = BG_YELLOW
    else:
        bg = BG_RED
    return ansi(bg, bar, use_color)


def detect_field(records, candidates):
    for c in candidates:
        if any(c in r for r in records):
            return c
    return None


def load_data(path_or_stream):
    """Load JSON or CSV, return list of dicts."""
    if isinstance(path_or_stream, str):
        with open(path_or_stream, encoding="utf-8-sig") as f:
            content = f.read()
    else:
        content = path_or_stream.read()

    content = content.strip()

    # Try JSON
    if content.startswith("[") or content.startswith("{"):
        data = json.loads(content)
        if isinstance(data, dict):
            # Maybe it's {records: [...]}
            for key in ("records", "data", "results", "experiments"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]
        return data

    # Try CSV
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    # Convert numeric strings
    out = []
    for row in rows:
        new = {}
        for k, v in row.items():
            v = v.strip() if v else ""
            try:
                new[k] = float(v)
            except (ValueError, TypeError):
                new[k] = v
        out.append(new)
    return out


def is_flag_experiment(records, label_field):
    """Check if records are 2^N flag combinations (flags like 'A', 'AB', 'BCD')."""
    if label_field != "flags" and label_field != "exp":
        return False
    labels = [str(r.get(label_field, "")).strip() for r in records]
    single = [l for l in labels if len(l) == 1 and l.isalpha()]
    return len(single) >= 2


def render_matrix(records, score_field, label_field, use_color, top_n=None):
    """Render NxN matrix for flag experiments."""
    # Find all single-flag labels
    all_labels = sorted(set(str(r.get(label_field, "")).strip() for r in records))
    singles = sorted(set(l for l in all_labels if len(l) == 1 and l.isalpha()))

    if not singles:
        return False

    # Build lookup: label -> score
    score_map = {}
    for r in records:
        lbl = str(r.get(label_field, "")).strip()
        score_map[lbl] = float(r.get(score_field, 0) or 0)

    n = len(singles)
    col_w = 4

    print(f"\n  EXPERIMENT HEAT MAP ({score_field})")

    # Header
    header = "  ┌" + "────┬" * n + "────┐"
    # Actually build proper header
    top = "  ┌" + ("────┬" * (n - 1)) + "────┬" + "────┐"
    label_row = "  │    │" + "".join(f" {s:>2} │" for s in singles)
    sep = "  ├────┼" + ("────┼" * (n - 1)) + "────┤"

    print("  ┌────┬" + "────┬" * (n - 1) + "────┐")
    print("  │    │" + "".join(f" {s:>2} │" for s in singles))
    print("  ├────┼" + "────┼" * (n - 1) + "────┤")

    combo_scores = {}
    for r in records:
        lbl = str(r.get(label_field, "")).strip()
        if len(lbl) == 2:
            combo_scores[lbl] = float(r.get(score_field, 0) or 0)
            combo_scores[lbl[1] + lbl[0]] = combo_scores[lbl]  # symmetric

    side_notes = []
    for i, row_flag in enumerate(singles):
        row_parts = [f"  │ {row_flag:>2} │"]
        row_combos = []
        for col_flag in singles:
            if row_flag == col_flag:
                s = score_map.get(row_flag, 0)
                block = score_to_block(s, use_color)
                row_parts.append(f"{block}│")
            else:
                combo = "".join(sorted([row_flag, col_flag]))
                s = combo_scores.get(combo, score_map.get(row_flag, 0))
                block = score_to_block(s, use_color)
                row_parts.append(f"{block}│")
                row_combos.append((combo, s))

        # Side notes: show combos from this row
        side = []
        seen = set()
        for combo, s in row_combos:
            if combo not in seen:
                seen.add(combo)
                sign = "+" if s >= 0 else ""
                note = f"{combo[0]}+{combo[1]}: {sign}{s:.2f}"
                side.append(note)
        side_str = "  " + "  ".join(side[:3]) if side else ""

        print("".join(row_parts) + side_str)
        if i < len(singles) - 1:
            print("  ├────┼" + "────┼" * (n - 1) + "────┤")

    print("  └────┴" + "────┴" * (n - 1) + "────┘")

    # Legend
    print()
    print("  Score: " +
          score_to_block(0.4, use_color) + " >+0.3  " +
          score_to_block(0.2, use_color) + " >+0.1  " +
          score_to_block(0.0, use_color) + " ~0  " +
          score_to_block(-0.2, use_color) + " <-0.1  " +
          score_to_block(-0.4, use_color) + " <-0.3")

    # Best / worst
    all_scores = [(str(r.get(label_field, "")), float(r.get(score_field, 0) or 0)) for r in records]
    all_scores.sort(key=lambda x: x[1], reverse=True)
    best_lbl, best_s = all_scores[0]
    worst_lbl, worst_s = all_scores[-1]

    baseline_records = [r for r in records if str(r.get(label_field, "")).strip().upper() in ("BASELINE", "BASE", "")]
    baseline_s = float(baseline_records[0].get(score_field, 0)) if baseline_records else 0.0

    print(f"  BEST: {best_lbl} ({best_s:+.3f})   WORST: {worst_lbl} ({worst_s:+.3f})   BASELINE: {baseline_s:.3f}")

    # Combo winner
    if combo_scores:
        best_combo = max(combo_scores.items(), key=lambda x: x[1])
        print(f"  BEST COMBO: {best_combo[0][0]}+{best_combo[0][1]} ({best_combo[1]:+.3f})")

    print()
    return True


def render_bars(records, score_field, label_field, use_color, top_n=None, sort=True):
    """Render horizontal bar chart."""
    scored = []
    for r in records:
        lbl = str(r.get(label_field, r.get("exp", r.get("name", "?"))))
        try:
            s = float(r.get(score_field, 0) or 0)
        except (ValueError, TypeError):
            s = 0.0
        scored.append((lbl, s, r))

    if sort:
        scored.sort(key=lambda x: x[1], reverse=True)

    if top_n:
        scored = scored[:top_n]

    max_abs = max((abs(s) for _, s, _ in scored), default=1) or 1
    max_lbl = max((len(l) for l, _, _ in scored), default=10)

    print(f"\n  RANKED: {score_field}")
    print("  " + "─" * (max_lbl + 28))
    for lbl, s, _ in scored:
        bar = score_to_bar(s, max_abs, width=20, use_color=use_color)
        sign = "+" if s >= 0 else ""
        print(f"  {lbl:<{max_lbl}}  {bar}  {sign}{s:.3f}")
    print("  " + "─" * (max_lbl + 28))

    best  = max(scored, key=lambda x: x[1])
    worst = min(scored, key=lambda x: x[1])
    print(f"  BEST: {best[0]} ({best[1]:+.3f})   WORST: {worst[0]} ({worst[1]:+.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="heat.py — experiment heatmap")
    parser.add_argument("file", nargs="?", default=None)
    parser.add_argument("--field", default=None, help="Score field to use")
    parser.add_argument("--sort", action="store_true", default=True)
    parser.add_argument("--no-sort", dest="sort", action="store_false")
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()

    use_color = sys.stdout.isatty()

    # Load data
    if args.file:
        records = load_data(args.file)
    elif not sys.stdin.isatty():
        records = load_data(sys.stdin)
    else:
        print("Provide a file argument or pipe JSON/CSV to stdin", file=sys.stderr)
        sys.exit(1)

    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    # Detect fields
    score_field = args.field or detect_field(records, SCORE_FIELDS) or list(records[0].keys())[-1]
    label_field = detect_field(records, LABEL_FIELDS) or list(records[0].keys())[0]

    # Try matrix first for flag experiments
    if is_flag_experiment(records, label_field):
        rendered = render_matrix(records, score_field, label_field, use_color, args.top)
        if rendered:
            return

    # Fall back to bar chart
    render_bars(records, score_field, label_field, use_color, args.top, args.sort)


if __name__ == "__main__":
    main()
