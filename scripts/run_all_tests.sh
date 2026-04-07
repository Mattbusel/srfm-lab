#!/bin/bash
# scripts/run_all_tests.sh
# ========================
# Run the full SRFM Lab test suite across all languages.
# Exits with code 0 only if every suite passes.
#
# Usage:
#   ./scripts/run_all_tests.sh [--fast] [--no-rust] [--no-julia] [--no-r]
#
# Flags:
#   --fast      Skip slow ML and backtesting test suites
#   --no-rust   Skip Rust crate tests
#   --no-julia  Skip Julia tests
#   --no-r      Skip R tests

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/test_runs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/test_run_${TIMESTAMP}.log"

RUN_RUST=true
RUN_JULIA=true
RUN_R=true
RUN_SLOW=true
FAST=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --fast)      FAST=true; RUN_SLOW=false ;;
        --no-rust)   RUN_RUST=false ;;
        --no-julia)  RUN_JULIA=false ;;
        --no-r)      RUN_R=false ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # no colour

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

pass() { echo -e "${GREEN}[PASS]${NC} $1"; ((PASS_COUNT++)); }
fail() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)); }
skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; ((SKIP_COUNT++)); }
section() { echo -e "\n${BOLD}${CYAN}=== $1 ===${NC}"; }

run_suite() {
    local name="$1"
    shift
    echo -e "\n${CYAN}-- Running: ${name}${NC}"
    if "$@" >> "$LOG_FILE" 2>&1; then
        pass "$name"
    else
        fail "$name"
        echo -e "${RED}   See $LOG_FILE for details${NC}"
    fi
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mkdir -p "$LOG_DIR"
echo "SRFM Lab Test Run -- ${TIMESTAMP}" > "$LOG_FILE"
echo "Repo: $REPO_ROOT" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

cd "$REPO_ROOT"

section "SRFM Lab Full Test Suite"
echo "Log: $LOG_FILE"
echo "Timestamp: $TIMESTAMP"
if [[ "$FAST" == "true" ]]; then
    echo -e "${YELLOW}Fast mode: skipping slow suites${NC}"
fi

# ---------------------------------------------------------------------------
# Python tests
# ---------------------------------------------------------------------------

section "Python Tests"

# Core library and unit tests
PYTHON_CORE_DIRS=(
    "tests/"
    "lib/options/tests/"
    "execution/risk/tests/"
    "execution/tests/"
)

# Slow suites (ML, backtesting, optimisation)
PYTHON_SLOW_DIRS=(
    "ml/tests/"
    "backtest/tests/"
    "optimization/tests/"
    "research/signal_analytics/tests/"
)

PYTHON_OTHER_DIRS=(
    "data/tests/"
    "infra/observability/tests/"
)

# Build the pytest invocation
PYTEST_DIRS=()
for d in "${PYTHON_CORE_DIRS[@]}"; do
    [[ -d "$d" ]] && PYTEST_DIRS+=("$d")
done
for d in "${PYTHON_OTHER_DIRS[@]}"; do
    [[ -d "$d" ]] && PYTEST_DIRS+=("$d")
done
if [[ "$RUN_SLOW" == "true" ]]; then
    for d in "${PYTHON_SLOW_DIRS[@]}"; do
        [[ -d "$d" ]] && PYTEST_DIRS+=("$d")
    done
fi

if [[ ${#PYTEST_DIRS[@]} -gt 0 ]]; then
    run_suite "Python core + integration tests" \
        python -m pytest "${PYTEST_DIRS[@]}" -x -q \
            --tb=short \
            --no-header \
            -p no:warnings \
            --timeout=120
else
    skip "Python tests (no test directories found)"
fi

# ---------------------------------------------------------------------------
# Rust tests
# ---------------------------------------------------------------------------

section "Rust Tests"

if [[ "$RUN_RUST" == "true" ]]; then
    if command -v cargo &>/dev/null; then
        if [[ -f "Cargo.toml" ]]; then
            run_suite "Rust workspace tests" \
                cargo test --workspace --quiet
        else
            skip "Rust tests (no Cargo.toml found)"
        fi
    else
        skip "Rust tests (cargo not on PATH)"
    fi
else
    skip "Rust tests (--no-rust)"
fi

# ---------------------------------------------------------------------------
# Go tests
# ---------------------------------------------------------------------------

section "Go Tests"

if command -v go &>/dev/null; then
    GO_MODULES=()
    for mod_dir in cmd/ market-data/ bridge/; do
        if [[ -f "${mod_dir}go.mod" ]]; then
            GO_MODULES+=("$mod_dir")
        fi
    done
    if [[ ${#GO_MODULES[@]} -gt 0 ]]; then
        for mod in "${GO_MODULES[@]}"; do
            run_suite "Go tests: $mod" \
                bash -c "cd $mod && go test ./... -count=1 -timeout 60s"
        done
    else
        skip "Go tests (no go.mod found in cmd/ market-data/ bridge/)"
    fi
else
    skip "Go tests (go not on PATH)"
fi

# ---------------------------------------------------------------------------
# Julia tests
# ---------------------------------------------------------------------------

section "Julia Tests"

if [[ "$RUN_JULIA" == "true" ]]; then
    if command -v julia &>/dev/null; then
        if [[ -f "julia/tests/runtests.jl" ]]; then
            run_suite "Julia test suite" \
                julia --project=julia julia/tests/runtests.jl
        else
            skip "Julia tests (julia/tests/runtests.jl not found)"
        fi
    else
        skip "Julia tests (julia not on PATH)"
    fi
else
    skip "Julia tests (--no-julia)"
fi

# ---------------------------------------------------------------------------
# R tests
# ---------------------------------------------------------------------------

section "R Tests"

if [[ "$RUN_R" == "true" ]]; then
    if command -v Rscript &>/dev/null; then
        if [[ -d "r/tests" ]] || [[ -d "R/tests" ]]; then
            R_TEST_DIR="r/tests"
            [[ -d "R/tests" ]] && R_TEST_DIR="R/tests"
            run_suite "R testthat suite" \
                Rscript -e "testthat::test_dir('${R_TEST_DIR}/')"
        else
            skip "R tests (no r/tests or R/tests directory)"
        fi
    else
        skip "R tests (Rscript not on PATH)"
    fi
else
    skip "R tests (--no-r)"
fi

# ---------------------------------------------------------------------------
# C++ / native tests
# ---------------------------------------------------------------------------

section "C++ / Native Tests"

if [[ -d "cpp" ]] && [[ -f "cpp/build/Release/run_tests" ]]; then
    run_suite "C++ signal engine tests" \
        cpp/build/Release/run_tests
elif [[ -d "build" ]] && command -v ctest &>/dev/null; then
    run_suite "CMake CTest suite" \
        ctest --test-dir build --output-on-failure --timeout 60
else
    skip "C++ tests (build artifacts not found -- run deploy.sh first)"
fi

# ---------------------------------------------------------------------------
# TypeScript / dashboard tests
# ---------------------------------------------------------------------------

section "TypeScript Tests"

if command -v npm &>/dev/null && [[ -f "dashboard/package.json" ]]; then
    run_suite "TypeScript dashboard tests" \
        bash -c "cd dashboard && npm test -- --watchAll=false --passWithNoTests"
elif command -v bun &>/dev/null && [[ -f "dashboard/package.json" ]]; then
    run_suite "TypeScript dashboard tests (bun)" \
        bash -c "cd dashboard && bun test"
else
    skip "TypeScript tests (npm/bun not found or no dashboard/package.json)"
fi

# ---------------------------------------------------------------------------
# Elixir tests
# ---------------------------------------------------------------------------

section "Elixir Tests"

if command -v mix &>/dev/null && [[ -f "coordination/mix.exs" ]]; then
    run_suite "Elixir coordination tests" \
        bash -c "cd coordination && mix test --no-start"
else
    skip "Elixir tests (mix not on PATH or no coordination/mix.exs)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

section "Test Summary"

TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo "Total suites: $TOTAL"
echo -e "${GREEN}Passed: ${PASS_COUNT}${NC}"
echo -e "${YELLOW}Skipped: ${SKIP_COUNT}${NC}"
echo -e "${RED}Failed: ${FAIL_COUNT}${NC}"
echo ""
echo "Full log: $LOG_FILE"

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo -e "\n${RED}${BOLD}TEST RUN FAILED${NC} -- $FAIL_COUNT suite(s) failed."
    exit 1
fi

echo -e "\n${GREEN}${BOLD}ALL TESTS PASSED${NC}"
exit 0
