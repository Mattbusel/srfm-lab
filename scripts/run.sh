#!/usr/bin/env bash
# run.sh — Quick backtest wrapper
# Usage: ./scripts/run.sh larsa-v1
#        ./scripts/run.sh larsa-v1 --docker

set -euo pipefail

STRATEGY="${1:?Usage: run.sh <strategy-name>}"
shift
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="results/${STRATEGY}/${TIMESTAMP}"

echo "==> Backtesting strategies/${STRATEGY}  →  ${OUTPUT}"
lean backtest "strategies/${STRATEGY}" --output "${OUTPUT}" "$@"

echo ""
echo "==> Result: ${OUTPUT}"
echo "==> To compare: make compare s=${STRATEGY}"
