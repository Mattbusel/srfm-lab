#!/usr/bin/env bash
# run_wave4_backtest.sh
# Runs the Wave 4 signal-module comparison backtest.
#
# Usage:
#   bash tools/run_wave4_backtest.sh              # full run (includes ML)
#   bash tools/run_wave4_backtest.sh --no-ml      # skip ML training (faster)
#
# The script expects to be run from the repo root (srfm-lab/) or from tools/.

set -euo pipefail

# ── Resolve repo root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Python path: make sure tools/ is importable ──────────────────────────────
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ── Default cache path ───────────────────────────────────────────────────────
CACHE="$REPO_ROOT/tools/backtest_output/crypto_data_cache.pkl"

echo "============================================================"
echo "  Wave 4 Backtest — Signal Module Comparison"
echo "  Repo:   $REPO_ROOT"
echo "  Cache:  $CACHE"
echo "============================================================"

# Pass all CLI args through (e.g. --no-ml)
python "$REPO_ROOT/tools/backtest_wave4.py" --cache "$CACHE" "$@"
