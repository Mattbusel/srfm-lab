#!/bin/bash
# scripts/deploy.sh
# =================
# Build and deploy all SRFM Lab components in one shot.
#
# Build order:
#   1. Rust crates           (cargo build --release --workspace)
#   2. Zig native layer      (cd native && make)
#   3. C++ signal engine     (cmake --build build --config Release)
#   4. Julia sysimage        (julia --project=julia julia/Makefile --target sysimage)
#   5. Python deps           (pip install -r requirements.txt)
#   6. Go services           (go build in each cmd/)
#   7. TypeScript dashboard  (npm ci && npm run build)
#   8. Elixir coordination   (mix deps.get && mix compile)
#   9. Smoke tests
#  10. Restart services via coordination layer API
#
# Usage:
#   ./scripts/deploy.sh [--skip-build] [--no-restart] [--env staging|prod]
#
# Flags:
#   --skip-build   Skip all compilation steps (deploy pre-built artifacts only)
#   --no-restart   Build everything but do not restart services
#   --env          Target environment (default: staging)
#   --dry-run      Print commands, do not execute

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
DEPLOY_LOG="${LOG_DIR}/deploy_$(date +%Y%m%d_%H%M%S).log"
ENV_FILE="${REPO_ROOT}/.env"

SKIP_BUILD=false
NO_RESTART=false
DEPLOY_ENV="staging"
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=true ;;
        --no-restart) NO_RESTART=true ;;
        --dry-run)    DRY_RUN=true ;;
        --env)        shift; DEPLOY_ENV="$1" ;;
        --env=*)      DEPLOY_ENV="${arg#*=}" ;;
    esac
done

mkdir -p "$LOG_DIR"
[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

STEP=0
ERRORS=()

step() {
    STEP=$((STEP + 1))
    echo -e "\n${BOLD}${CYAN}[${STEP}]${NC} $1" | tee -a "$DEPLOY_LOG"
}

ok()    { echo -e "${GREEN}[OK]${NC}    $1" | tee -a "$DEPLOY_LOG"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1" | tee -a "$DEPLOY_LOG"; }
error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$DEPLOY_LOG" >&2; ERRORS+=("$1"); }

run() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}[DRY]${NC} $*" | tee -a "$DEPLOY_LOG"
        return 0
    fi
    echo "+ $*" >> "$DEPLOY_LOG"
    if "$@" >> "$DEPLOY_LOG" 2>&1; then
        return 0
    else
        local rc=$?
        error "Command failed (exit $rc): $*"
        return $rc
    fi
}

run_in_dir() {
    local dir="$1"
    shift
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}[DRY]${NC} (in $dir) $*" | tee -a "$DEPLOY_LOG"
        return 0
    fi
    echo "+ (cd $dir) $*" >> "$DEPLOY_LOG"
    if (cd "$dir" && "$@" >> "$DEPLOY_LOG" 2>&1); then
        return 0
    else
        local rc=$?
        error "Command failed in $dir (exit $rc): $*"
        return $rc
    fi
}

skip_check() {
    local name="$1"
    local condition="$2"
    if ! eval "$condition" 2>/dev/null; then
        warn "Skipping $name: prerequisite not met ($condition)"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}  SRFM Lab Deployment Script            ${NC}"
echo -e "${BOLD}  Environment: ${DEPLOY_ENV}            ${NC}"
echo -e "${BOLD}  $(date)                               ${NC}"
echo -e "${BOLD}========================================${NC}"
echo "Deploy log: $DEPLOY_LOG"
[[ "$DRY_RUN" == "true" ]] && echo -e "${YELLOW}DRY-RUN MODE: no changes will be made${NC}"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Rust crates
# ---------------------------------------------------------------------------

step "Building Rust crates"

if [[ "$SKIP_BUILD" == "false" ]]; then
    if skip_check "Rust" "command -v cargo"; then
        :
    elif [[ -f "Cargo.toml" ]]; then
        run cargo build --release --workspace
        ok "Rust crates built"
    else
        warn "No Cargo.toml found -- skipping Rust build"
    fi
else
    warn "Build skipped (--skip-build)"
fi

# ---------------------------------------------------------------------------
# 2. Zig native layer
# ---------------------------------------------------------------------------

step "Building Zig native layer"

if [[ "$SKIP_BUILD" == "false" ]]; then
    if [[ -d "native" ]] && [[ -f "native/Makefile" ]]; then
        if command -v zig &>/dev/null || command -v make &>/dev/null; then
            run_in_dir "native" make
            ok "Zig native layer built"
        else
            warn "make/zig not available -- skipping native build"
        fi
    else
        warn "native/Makefile not found -- skipping native build"
    fi
fi

# ---------------------------------------------------------------------------
# 3. C++ signal engine
# ---------------------------------------------------------------------------

step "Building C++ signal engine"

if [[ "$SKIP_BUILD" == "false" ]]; then
    if [[ -d "cpp" ]]; then
        if command -v cmake &>/dev/null; then
            # Configure if not already configured
            if [[ ! -d "cpp/build" ]]; then
                run cmake -S cpp -B cpp/build \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
            fi
            run cmake --build cpp/build --config Release --parallel 4
            ok "C++ signal engine built"
        else
            warn "cmake not available -- skipping C++ build"
        fi
    else
        warn "cpp/ directory not found -- skipping C++ build"
    fi
fi

# ---------------------------------------------------------------------------
# 4. Julia sysimage
# ---------------------------------------------------------------------------

step "Compiling Julia sysimage"

if [[ "$SKIP_BUILD" == "false" ]]; then
    if command -v julia &>/dev/null && [[ -d "julia" ]]; then
        JULIA_SYSIMAGE="${REPO_ROOT}/julia/sysimage.so"
        if [[ -f "julia/build_sysimage.jl" ]]; then
            run julia --project=julia julia/build_sysimage.jl
            ok "Julia sysimage compiled: $JULIA_SYSIMAGE"
        elif [[ -f "julia/Makefile" ]]; then
            run_in_dir "julia" make sysimage
            ok "Julia sysimage via Makefile"
        else
            # Precompile all packages
            run julia --project=julia -e 'using Pkg; Pkg.precompile()'
            ok "Julia packages precompiled (no sysimage Makefile found)"
        fi
    else
        warn "Julia not available or julia/ not found -- skipping sysimage"
    fi
fi

# ---------------------------------------------------------------------------
# 5. Python dependencies
# ---------------------------------------------------------------------------

step "Installing Python dependencies"

if [[ -f "requirements.txt" ]]; then
    if command -v pip &>/dev/null; then
        run pip install -r requirements.txt --quiet
        ok "Python requirements installed"
    elif command -v pip3 &>/dev/null; then
        run pip3 install -r requirements.txt --quiet
        ok "Python requirements installed (pip3)"
    else
        error "pip not found -- cannot install Python dependencies"
    fi
fi

if [[ -f "pyproject.toml" ]]; then
    if command -v pip &>/dev/null; then
        run pip install -e . --quiet --no-deps
        ok "Package installed in editable mode"
    fi
fi

# ---------------------------------------------------------------------------
# 6. Go services
# ---------------------------------------------------------------------------

step "Building Go services"

if [[ "$SKIP_BUILD" == "false" ]]; then
    if command -v go &>/dev/null; then
        GO_DIRS=()
        for d in cmd/* market-data/ bridge/; do
            if [[ -f "${d}go.mod" ]] || [[ -f "${d}main.go" ]]; then
                GO_DIRS+=("$d")
            fi
        done
        if [[ ${#GO_DIRS[@]} -gt 0 ]]; then
            for go_dir in "${GO_DIRS[@]}"; do
                run_in_dir "$go_dir" go build -o "${go_dir##*/}" ./...
                ok "Go service built: $go_dir"
            done
        else
            warn "No Go service directories found"
        fi
    else
        warn "go not on PATH -- skipping Go build"
    fi
fi

# ---------------------------------------------------------------------------
# 7. TypeScript dashboard
# ---------------------------------------------------------------------------

step "Building TypeScript dashboard"

if [[ "$SKIP_BUILD" == "false" ]]; then
    if [[ -d "dashboard" ]] && [[ -f "dashboard/package.json" ]]; then
        if command -v npm &>/dev/null; then
            run_in_dir "dashboard" npm ci --silent
            run_in_dir "dashboard" npm run build
            ok "TypeScript dashboard built"
        elif command -v bun &>/dev/null; then
            run_in_dir "dashboard" bun install
            run_in_dir "dashboard" bun run build
            ok "TypeScript dashboard built (bun)"
        else
            warn "npm/bun not found -- skipping dashboard build"
        fi
    else
        warn "dashboard/ not found or no package.json -- skipping"
    fi
fi

# ---------------------------------------------------------------------------
# 8. Elixir coordination layer
# ---------------------------------------------------------------------------

step "Building Elixir coordination layer"

if [[ "$SKIP_BUILD" == "false" ]]; then
    if command -v mix &>/dev/null && [[ -f "coordination/mix.exs" ]]; then
        run_in_dir "coordination" mix deps.get
        run_in_dir "coordination" mix compile
        ok "Elixir coordination layer compiled"
    else
        warn "mix not available or coordination/mix.exs missing -- skipping"
    fi
fi

# ---------------------------------------------------------------------------
# 9. Smoke tests
# ---------------------------------------------------------------------------

step "Running smoke tests"

SMOKE_FAILED=0

# Python import sanity
if command -v python &>/dev/null; then
    if python -c "import sys; sys.path.insert(0,'${REPO_ROOT}'); import execution.risk.live_var" \
       >> "$DEPLOY_LOG" 2>&1; then
        ok "Python execution.risk imports OK"
    else
        warn "Python import smoke test failed -- check $DEPLOY_LOG"
        SMOKE_FAILED=1
    fi
fi

# Rust binary smoke test
RUST_BIN="${REPO_ROOT}/target/release/srfm_signal_engine"
if [[ -f "$RUST_BIN" ]]; then
    if "$RUST_BIN" --version >> "$DEPLOY_LOG" 2>&1; then
        ok "Rust binary smoke test passed"
    else
        warn "Rust binary smoke test failed"
        SMOKE_FAILED=1
    fi
fi

# DB schema smoke test
if command -v python &>/dev/null; then
    python - << 'PYEOF' >> "$DEPLOY_LOG" 2>&1
import sqlite3, sys
db_path = "execution/live_trades.db"
try:
    conn = sqlite3.connect(db_path, timeout=3)
    conn.execute("PRAGMA integrity_check")
    conn.close()
    print("DB integrity OK")
except Exception as e:
    print(f"DB check: {e}")
    sys.exit(1)
PYEOF
    if [[ $? -eq 0 ]]; then
        ok "SQLite DB integrity check passed"
    else
        warn "SQLite DB integrity check failed"
        SMOKE_FAILED=1
    fi
fi

if [[ $SMOKE_FAILED -eq 0 ]]; then
    ok "All smoke tests passed"
else
    warn "Some smoke tests failed -- check $DEPLOY_LOG"
fi

# ---------------------------------------------------------------------------
# 10. Restart services via coordination API
# ---------------------------------------------------------------------------

step "Restarting services via coordination layer"

if [[ "$NO_RESTART" == "false" ]]; then
    COORD_URL="${SRFM_COORD_URL:-http://localhost:8781}"

    # Check if coordination layer is reachable
    if curl -sf --max-time 3 "${COORD_URL}/health" > /dev/null 2>&1; then
        ok "Coordination layer reachable at $COORD_URL"

        # Signal a rolling restart via the coordination API
        if [[ "$DRY_RUN" == "false" ]]; then
            RESTART_PAYLOAD='{"action":"rolling_restart","source":"deploy.sh","env":"'"${DEPLOY_ENV}"'"}'
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
                --max-time 10 \
                -X POST "${COORD_URL}/admin/restart" \
                -H "Content-Type: application/json" \
                -d "$RESTART_PAYLOAD" 2>/dev/null || echo "000")

            if [[ "$HTTP_CODE" == "200" ]] || [[ "$HTTP_CODE" == "202" ]]; then
                ok "Rolling restart initiated (HTTP $HTTP_CODE)"
            else
                warn "Coordination restart returned HTTP $HTTP_CODE"
                warn "Services may need to be restarted manually: ./scripts/start_all_services.sh"
            fi
        else
            echo -e "${YELLOW}[DRY]${NC} Would POST to ${COORD_URL}/admin/restart"
        fi
    else
        warn "Coordination layer not reachable at $COORD_URL"
        warn "Start services manually: ./scripts/start_all_services.sh"
    fi
else
    info "Service restart skipped (--no-restart)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}  Deployment Summary                    ${NC}"
echo -e "${BOLD}========================================${NC}"
echo "  Environment : $DEPLOY_ENV"
echo "  Steps run   : $STEP"
echo "  Log file    : $DEPLOY_LOG"

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}${BOLD}Errors encountered:${NC}"
    for err in "${ERRORS[@]}"; do
        echo -e "  ${RED}x${NC} $err"
    done
    echo ""
    echo -e "${RED}${BOLD}DEPLOYMENT INCOMPLETE${NC} -- review errors above"
    exit 1
fi

echo ""
echo -e "${GREEN}${BOLD}DEPLOYMENT SUCCESSFUL${NC}"
echo ""
exit 0
