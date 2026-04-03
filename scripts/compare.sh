#!/usr/bin/env bash
# compare.sh — Compare last N backtests for a strategy
# Usage: ./scripts/compare.sh larsa-v1
#        ./scripts/compare.sh larsa-v1 --chart
#        ./scripts/compare.sh larsa-v1 larsa-v2

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: compare.sh <strategy> [strategy2 ...] [--chart]"
    exit 1
fi

PATHS=()
FLAGS=()
for arg in "$@"; do
    if [[ "$arg" == --* ]]; then
        FLAGS+=("$arg")
    else
        PATHS+=("results/${arg}")
    fi
done

echo "==> Comparing: ${PATHS[*]}"
python tools/compare.py "${PATHS[@]}" "${FLAGS[@]}"
