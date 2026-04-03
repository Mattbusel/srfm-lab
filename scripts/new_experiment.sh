#!/usr/bin/env bash
# new_experiment.sh — Scaffold a new strategy from the template
# Usage: ./scripts/new_experiment.sh my-experiment

set -euo pipefail

NAME="${1:?Usage: new_experiment.sh <experiment-name>}"
SRC="strategies/templates"
DST="strategies/${NAME}"

if [ -d "${DST}" ]; then
    echo "[ERROR] ${DST} already exists."
    exit 1
fi

cp -r "${SRC}" "${DST}"
echo "==> Created ${DST}"
echo ""
echo "Next steps:"
echo "  1. Edit ${DST}/main.py — change ONE thing from the template"
echo "  2. make backtest s=${NAME}"
echo "  3. make compare s=${NAME}"
echo ""
echo "If it fails: mv ${DST} strategies/graveyard/${NAME} and document why."
