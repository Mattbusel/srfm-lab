#!/usr/bin/env bash
# sweep.sh — Parameter sweep shortcut
# Usage: ./scripts/sweep.sh larsa-v1 BH_FORM 0.5 3.0 0.25
#        ./scripts/sweep.sh larsa-v1 CF 0.8 2.0 0.1 --metric SharpeRatio

set -euo pipefail

STRATEGY="${1:?Usage: sweep.sh <strategy> <param> <min> <max> <step>}"
PARAM="${2:?Missing parameter name}"
MIN="${3:?Missing min value}"
MAX="${4:?Missing max value}"
STEP="${5:?Missing step size}"
shift 5

echo "==> Sweeping ${PARAM} ∈ [${MIN}, ${MAX}] step=${STEP} on ${STRATEGY}"
python tools/param_sweep.py "strategies/${STRATEGY}" "${PARAM}" "${MIN}" "${MAX}" "${STEP}" "$@"
