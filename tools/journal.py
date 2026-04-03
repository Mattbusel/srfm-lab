#!/usr/bin/env python
"""
tools/journal.py — Append-only experiment log. JOURNAL.md is the database.

Usage:
  python tools/journal.py add "v4 submitted to QC" --tags submitted,v4
  python tools/journal.py result --version v4 --return 285 --dd 22 --sharpe 4.8 --trades 390
  python tools/journal.py log --n 5
  python tools/journal.py search "NQ"
  python tools/journal.py runs
  python tools/journal.py export --out results/journal.csv
"""

import argparse
import csv
import os
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JOURNAL_PATH = REPO_ROOT / "JOURNAL.md"

HEADER = "# SRFM Lab Journal\n"
DIVIDER = "---"


def _read_journal() -> str:
    if not JOURNAL_PATH.exists():
        return HEADER + "\n" + DIVIDER + "\n\n" + DIVIDER + "\n*Last updated: " + str(date.today()) + "*\n"
    return JOURNAL_PATH.read_text(encoding="utf-8")


def _write_journal(content: str):
    JOURNAL_PATH.write_text(content, encoding="utf-8")


def _update_footer(content: str) -> str:
    footer_pat = re.compile(r"\*Last updated:.*?\*", re.MULTILINE)
    new_footer = f"*Last updated: {date.today()}*"
    if footer_pat.search(content):
        return footer_pat.sub(new_footer, content)
    return content.rstrip() + "\n" + new_footer + "\n"


def _parse_entries(content: str) -> list[dict]:
    """Parse all ## entries from JOURNAL.md into dicts."""
    entries = []
    pattern = re.compile(
        r"^## (\d{4}-\d{2}-\d{2})\s+(.*?)\s+\[([^\]]+)\]\s*$(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    for m in pattern.finditer(content):
        entry_date, message, tag, body = m.group(1), m.group(2), m.group(3), m.group(4)
        result = ""
        result_m = re.search(r"\*\*Result\*\*:(.*)", body)
        if result_m:
            result = result_m.group(1).strip()
        finding = ""
        finding_m = re.search(r"\*\*Finding\*\*:(.*)", body)
        if finding_m:
            finding = finding_m.group(1).strip()
        entries.append({
            "date": entry_date,
            "message": message,
            "tag": tag,
            "result": result,
            "finding": finding,
            "raw": m.group(0),
        })
    return entries


def cmd_add(args):
    message = args.message
    tags = args.tags or ""
    tag = tags.split(",")[0].upper() if tags else "NOTE"
    today = str(date.today())

    entry = f"""## {today}  {message}  [{tag}]
**Finding**: (edit in JOURNAL.md)
**Next**: (edit in JOURNAL.md)

"""

    content = _read_journal()
    # Insert after the header line and the first ---
    insert_after = HEADER + "\n" + DIVIDER + "\n"
    if insert_after in content:
        content = content.replace(insert_after, insert_after + "\n" + entry, 1)
    else:
        # Fallback: insert after header
        content = content.replace(HEADER, HEADER + "\n" + DIVIDER + "\n\n" + entry, 1)

    content = _update_footer(content)
    _write_journal(content)
    print(f"Added entry: {today}  {message}  [{tag}]")


def cmd_result(args):
    version = args.version
    ret = args.ret
    dd = args.dd
    sharpe = args.sharpe
    trades = args.trades

    result_line = f"**Result**: {ret}% return, Sharpe {sharpe}, DD {dd}%, {trades} trades"

    content = _read_journal()
    entries = _parse_entries(content)

    # Find most recent entry matching the version
    target = None
    for e in entries:
        if version.lower() in e["message"].lower() or version.lower() in e["tag"].lower():
            target = e
            break

    if target is None:
        print(f"No entry found for version '{version}'. Add an entry first with: journal.py add")
        sys.exit(1)

    old_raw = target["raw"]
    # Replace or inject the Result line
    if "**Result**:" in old_raw:
        new_raw = re.sub(r"\*\*Result\*\*:.*", result_line, old_raw)
    else:
        # Insert after the ## header line
        lines = old_raw.split("\n")
        lines.insert(1, result_line)
        new_raw = "\n".join(lines)

    content = content.replace(old_raw, new_raw, 1)
    content = _update_footer(content)
    _write_journal(content)
    print(f"Updated result for '{version}': {result_line}")


def cmd_log(args):
    n = args.n
    content = _read_journal()
    entries = _parse_entries(content)
    shown = entries[:n]
    width = 55
    print("-" * width)
    print(f"  SRFM Journal — last {len(shown)} entries")
    print("-" * width)
    for e in shown:
        tag_str = f"[{e['tag']}]"
        print(f"  {e['date']}  {e['message']}  {tag_str}")
        if e["result"]:
            print(f"    Result: {e['result']}")
        if e["finding"] and "(edit in JOURNAL.md)" not in e["finding"]:
            # Trim long findings
            finding = e["finding"][:80] + ("…" if len(e["finding"]) > 80 else "")
            print(f"    Finding: {finding}")
        print()
    print("-" * width)


def cmd_search(args):
    query = args.query.lower()
    content = _read_journal()
    entries = _parse_entries(content)
    results = [e for e in entries if query in e["raw"].lower()]
    print(f"Search '{args.query}' — {len(results)} match(es)")
    for e in results:
        tag_str = f"[{e['tag']}]"
        print(f"  {e['date']}  {e['message']}  {tag_str}")
        # Show matching lines from raw
        for line in e["raw"].split("\n"):
            if query in line.lower() and not line.startswith("##"):
                print(f"    {line.strip()}")
        print()


def cmd_runs(args):
    content = _read_journal()
    entries = _parse_entries(content)
    submitted = [e for e in entries if "SUBMITTED" in e["tag"].upper()]
    width = 60
    print("-" * width)
    print(f"  QC Submitted Runs — {len(submitted)} total")
    print("-" * width)
    for e in submitted:
        print(f"  {e['date']}  {e['message']}")
        if e["result"]:
            print(f"    {e['result']}")
        else:
            print(f"    (no result logged)")
        print()
    print("-" * width)


def cmd_export(args):
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = _read_journal()
    entries = _parse_entries(content)

    # Parse numeric fields from result string
    def extract(pattern, text, default=""):
        m = re.search(pattern, text)
        return m.group(1) if m else default

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "message", "tag", "return_pct", "sharpe", "dd_pct", "trades", "finding"])
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "date": e["date"],
                "message": e["message"],
                "tag": e["tag"],
                "return_pct": extract(r"([\d.]+)% return", e["result"]),
                "sharpe": extract(r"Sharpe ([\d.]+)", e["result"]),
                "dd_pct": extract(r"DD ([\d.]+)%", e["result"]),
                "trades": extract(r"([\d]+) trades", e["result"]),
                "finding": e["finding"],
            })

    print(f"Exported {len(entries)} entries to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SRFM Lab Journal")
    sub = parser.add_subparsers(dest="cmd")

    p_add = sub.add_parser("add", help="Add a new journal entry")
    p_add.add_argument("message", help="Entry message/title")
    p_add.add_argument("--tags", help="Comma-separated tags (first tag used as status)")

    p_result = sub.add_parser("result", help="Log a result for a version")
    p_result.add_argument("--version", required=True)
    p_result.add_argument("--return", dest="ret", required=True, type=float)
    p_result.add_argument("--dd", required=True, type=float)
    p_result.add_argument("--sharpe", required=True, type=float)
    p_result.add_argument("--trades", required=True, type=int)

    p_log = sub.add_parser("log", help="Show last N entries")
    p_log.add_argument("--n", type=int, default=5)

    p_search = sub.add_parser("search", help="Search entries")
    p_search.add_argument("query")

    sub.add_parser("runs", help="Show all submitted QC runs")

    p_export = sub.add_parser("export", help="Export to CSV")
    p_export.add_argument("--out", default="results/journal.csv")

    args = parser.parse_args()

    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "result":
        cmd_result(args)
    elif args.cmd == "log":
        cmd_log(args)
    elif args.cmd == "search":
        cmd_search(args)
    elif args.cmd == "runs":
        cmd_runs(args)
    elif args.cmd == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
