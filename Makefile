# SRFM Lab Makefile
# Usage:  make help  (shows all targets)
#
# Quick start:
#   make install
#   make backtest s=larsa-v16
#   make run-api
#   make run-terminal

SHELL     := bash
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
PYTHON    := python
PIP       := pip
CARGO     := cargo
GO        := go
NPM       := npm

.DEFAULT_GOAL := help

# ─────────────────────────────────────────────────────────────────────────────
# Install / build
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: install
install: ## Install all dependencies (Python, Rust, Node, Go)
	$(PIP) install -r requirements.txt
	$(CARGO) build --release
	cd spacetime/web && $(NPM) install && cd ../..
	cd terminal && $(NPM) install && cd ..
	@echo "All dependencies installed."

.PHONY: install-python
install-python: ## Install Python dependencies only
	$(PIP) install -r requirements.txt

.PHONY: build
build: build-rust build-ext build-web ## Build all artifacts

.PHONY: build-rust
build-rust: ## Build all Rust crates
	$(CARGO) build --release --all

.PHONY: build-ext
build-ext: ## Build PyO3 Python extension via maturin
	cd extensions && python -m maturin develop --release
	@echo "Extension installed: srfm_core"

.PHONY: build-web
build-web: ## Build both React apps for production
	cd spacetime/web && $(NPM) run build
	cd terminal && $(NPM) run build

# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: test
test: test-python test-rust test-go ## Run all tests

.PHONY: test-python
test-python: ## Run Python test suite with coverage
	pytest tests/ --cov=spacetime --cov-report=term-missing -v --tb=short

.PHONY: test-rust
test-rust: ## Run Rust tests
	$(CARGO) test --all

.PHONY: test-go
test-go: ## Run Go tests
	$(GO) test ./cmd/... -v -race

.PHONY: test-quick
test-quick: ## Run fast tests only (no integration)
	pytest tests/ -m "not slow" -v --tb=short

# ─────────────────────────────────────────────────────────────────────────────
# Code quality
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: lint
lint: ## Lint Python, Rust, Go
	ruff check spacetime/ tools/ scripts/
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	test -z "$$(gofmt -l ./cmd/...)"

.PHONY: fmt
fmt: ## Format all code
	ruff format spacetime/ tools/ scripts/
	$(CARGO) fmt --all
	gofmt -w ./cmd/...
	cd spacetime/web && $(NPM) run lint -- --fix
	cd terminal && $(NPM) run lint -- --fix 2>/dev/null || true

.PHONY: typecheck
typecheck: ## Run mypy type checks
	mypy spacetime/engine/ spacetime/api/ --ignore-missing-imports

# ─────────────────────────────────────────────────────────────────────────────
# Running services
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: run-api
run-api: ## Start Spacetime Arena FastAPI server (port 8000)
	PYTHONPATH=. uvicorn spacetime.api.main:app \
		--host 0.0.0.0 --port 8000 --reload \
		--log-level info

.PHONY: run-terminal
run-terminal: ## Start Spacetime Arena Vite dev server (port 5173)
	cd spacetime/web && $(NPM) run dev -- --port 5173

.PHONY: run-terminal-app
run-terminal-app: ## Start portfolio Terminal Vite dev server (port 5174)
	cd terminal && $(NPM) run dev -- --port 5174

.PHONY: run-gateway
run-gateway: ## Start Go gateway (port 9000)
	$(GO) run ./cmd/gateway/

.PHONY: run-all
run-all: ## Start API + Arena + Gateway concurrently (requires tmux or separate terminals)
	@echo "Starting all services..."
	@echo "  API:      http://localhost:8000"
	@echo "  Arena:    http://localhost:5173"
	@echo "  Terminal: http://localhost:5174"
	@echo "  Gateway:  http://localhost:9000"
	@echo ""
	@echo "Run in separate terminals:"
	@echo "  make run-api"
	@echo "  make run-terminal"
	@echo "  make run-gateway"

# ─────────────────────────────────────────────────────────────────────────────
# Backtesting
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: backtest
backtest: ## Run LEAN backtest: make backtest s=larsa-v16
ifndef s
	$(error Usage: make backtest s=<strategy-name>)
endif
	lean backtest strategies/$(s) --output results/$(s)/$(TIMESTAMP)
	@echo ""
	@echo "Done. To compare: make compare s=$(s)"

.PHONY: local
local: ## Run Spacetime Arena local backtest: make local s=larsa-v16
ifndef s
	$(error Usage: make local s=larsa-v16)
endif
	PYTHONPATH=. $(PYTHON) -c "
from spacetime.engine.bh_engine import run_backtest_from_config
result = run_backtest_from_config('strategies/$(s)')
print(result.summary())
"

.PHONY: sweep
sweep: ## Parameter sweep: make sweep s=larsa-v16 param=BH_FORM min=1.0 max=2.5 step=0.25
ifndef s
	$(error Usage: make sweep s=<strategy> param=<PARAM> min=<min> max=<max> step=<step>)
endif
	$(PYTHON) tools/param_sweep.py strategies/$(s) $(param) $(min) $(max) $(step)

.PHONY: compare
compare: ## Compare all runs for a strategy: make compare s=larsa-v16
ifndef s
	$(error Usage: make compare s=<strategy-name>)
endif
	$(PYTHON) tools/compare.py results/$(s)

.PHONY: compare2
compare2: ## Compare two strategies: make compare2 s1=larsa-v14 s2=larsa-v16
ifndef s1
	$(error Usage: make compare2 s1=larsa-v1 s2=larsa-v2)
endif
	$(PYTHON) tools/compare.py results/$(s1) results/$(s2) --chart

.PHONY: sensitivity
sensitivity: ## Sensitivity analysis: make sensitivity s=larsa-v16
ifndef s
	$(error Usage: make sensitivity s=larsa-v16)
endif
	PYTHONPATH=. $(PYTHON) tools/risk_sensitivity.py --strategy $(s)

# ─────────────────────────────────────────────────────────────────────────────
# Live trading
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: run-live
run-live: ## Start live Alpaca trader (reads .env for keys)
	@if [ -f .env ]; then export $$(cat .env | xargs); fi; \
	$(PYTHON) tools/live_trader_alpaca.py

.PHONY: run-paper
run-paper: ## Same as run-live but forces paper trading URL
	@if [ -f .env ]; then export $$(cat .env | xargs); fi; \
	ALPACA_BASE_URL=https://paper-api.alpaca.markets \
	$(PYTHON) tools/live_trader_alpaca.py

.PHONY: livecheck
livecheck: ## Check live state and current positions
	$(PYTHON) tools/live_check.py

# ─────────────────────────────────────────────────────────────────────────────
# Data management
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: data-fetch
data-fetch: ## Fetch bar data from Polygon: make data-fetch start=2020-01-01
	$(PYTHON) scripts/fetch_polygon.py --all --timeframes 1d 1h 15m \
		--start $(or $(start),2020-01-01)

.PHONY: data-init
data-init: ## Download LEAN security master
	lean data download --dataset "US Futures Security Master"

.PHONY: duckdb-setup
duckdb-setup: ## Initialize DuckDB analytics database
	$(PYTHON) warehouse/duckdb/setup.py

.PHONY: duckdb-bh
duckdb-bh: ## Compute and export BH state timeseries
	$(PYTHON) warehouse/duckdb/setup.py --export-bh

.PHONY: duckdb-reports
duckdb-reports: ## Export CSV reports from DuckDB
	$(PYTHON) warehouse/duckdb/setup.py --reports

# ─────────────────────────────────────────────────────────────────────────────
# Database / warehouse
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: db-migrate
db-migrate: ## Run all pending migrations
	@for f in warehouse/migrations/*.sql; do \
		echo "Applying $$f..."; \
		psql $$DATABASE_URL -f "$$f" 2>&1 | tail -3; \
	done

.PHONY: db-seed
db-seed: ## Seed instruments and reference data
	psql $$DATABASE_URL -f warehouse/schema/06_seed_data.sql

.PHONY: db-reset
db-reset: ## Drop and recreate all tables (DESTRUCTIVE)
	@echo "WARNING: This will destroy all data. Ctrl-C to cancel."; sleep 5
	psql $$DATABASE_URL -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	$(MAKE) db-migrate
	$(MAKE) db-seed

.PHONY: db-refresh-views
db-refresh-views: ## Refresh all materialized views
	psql $$DATABASE_URL -c "REFRESH MATERIALIZED VIEW CONCURRENTLY run_daily_metrics;"
	psql $$DATABASE_URL -c "REFRESH MATERIALIZED VIEW CONCURRENTLY instrument_correlation_matrix;"

# ─────────────────────────────────────────────────────────────────────────────
# Reports and analytics
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: report
report: ## Generate strategy HTML report
	$(PYTHON) tools/report_builder.py
	@echo "Report -> results/lab_report.md"

.PHONY: html-report
html-report: ## Generate interactive HTML tearsheet
	$(PYTHON) tools/generate_report.py
	@echo "Report -> results/strategy_report.html"

.PHONY: tearsheet
tearsheet: ## QuantStats HTML tearsheet
	$(PYTHON) tools/quantstats_report.py --json research/trade_analysis_data.json
	@echo "Open results/tearsheet.html in browser"

.PHONY: r-tearsheet
r-tearsheet: ## R PerformanceAnalytics tearsheet
	Rscript scripts/r_tearsheet.R

.PHONY: r-garch
r-garch: ## R DCC-GARCH volatility forecasts
	Rscript scripts/r_garch.R

.PHONY: r-tables
r-tables: ## R GT publication-quality tables
	Rscript scripts/r_tables.R

.PHONY: r-install
r-install: ## Install R packages
	Rscript scripts/install_r_packages.R

# ─────────────────────────────────────────────────────────────────────────────
# Strategy utilities
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: new
new: ## Create new strategy from template: make new name=experiment-1
ifndef name
	$(error Usage: make new name=<experiment-name>)
endif
	./scripts/new_experiment.sh $(name)

.PHONY: research
research: ## Open Jupyter notebook for a strategy: make research s=larsa-v16
ifndef s
	$(error Usage: make research s=<strategy-name>)
endif
	lean research strategies/$(s)

# ─────────────────────────────────────────────────────────────────────────────
# Forensics / diagnostic tools
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: what
what: ## What was happening on a date: make what d=2024-10-14
ifndef d
	@echo "Usage: make what d=2024-10-14"
else
	$(PYTHON) tools/what.py $(d)
endif

.PHONY: blame
blame: ## P&L attribution: make blame from=2024-01-01 to=2024-12-31
ifndef from
	@echo "Usage: make blame from=2024-01-01 to=2024-12-31"
else
	$(PYTHON) tools/blame.py --from $(from) --to $(to)
endif

.PHONY: why
why: ## Why did a trade happen: make why date=2024-10-14 instrument=NQ
ifndef date
	@echo "Usage: make why date=2024-10-14 instrument=NQ"
else
	$(PYTHON) tools/why.py --date $(date) --instrument $(or $(instrument),ES)
endif

.PHONY: when
when: ## Find trade by dollar P&L: make when p=-265000
ifndef p
	@echo "Usage: make when p=-265000"
else
	$(PYTHON) tools/when.py $(p)
endif

.PHONY: odds
odds: ## Historical win probability given current state
	$(PYTHON) tools/odds.py --regime BULL --bh_active true --convergence 2

.PHONY: anatomy
anatomy: ## Dissect top N trades
	$(PYTHON) tools/anatomy.py --top 5
	@echo "Written to ANATOMY.md"

.PHONY: kelly
kelly: ## Compute Kelly optimal sizing from trade history
	$(PYTHON) tools/kelly.py

.PHONY: deathloop
deathloop: ## QC forensics on a trade CSV: make deathloop trades=file.csv
ifndef trades
	$(error Usage: make deathloop trades=path/to/trades.csv)
endif
	$(PYTHON) tools/deathloop_detective.py --trades "$(trades)"

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: viz-build
viz-build: ## Build Rust visualization tool
	cd viz && $(CARGO) build --release
	@echo "srfm-viz built -> viz/target/release/srfm-viz"

.PHONY: graphics
graphics: viz-build ## Generate all SVG graphics via Rust viz tool
	@mkdir -p results/graphics
	./viz/target/release/srfm-viz spacetime \
		--csv data/NDX_hourly_poly.csv --cf 0.005 \
		--out results/graphics/spacetime.svg
	@echo "Graphics -> results/graphics/"

.PHONY: regime-graph
regime-graph: ## Graphviz regime state machine diagram
	$(PYTHON) tools/regime_graph.py
	@echo "Regime graph -> results/graphics/regime_states.dot/.svg"

.PHONY: regime-line
regime-line: ## 7-year regime in one colored line
	cat data/NDX_hourly_poly.csv | $(PYTHON) tools/regime.py

.PHONY: network
network: ## D3 regime network diagram (browser)
	$(PYTHON) tools/regime_network.py 2>/dev/null || true
	@echo "Open tools/web/regime_force.html in browser"

.PHONY: gallery
gallery: ## 3D well gallery (browser)
	$(PYTHON) -m http.server 8080 --directory tools/web &
	@echo "Open http://localhost:8080/well_gallery.html"

.PHONY: animate
animate: ## Manim SRFM physics animation
	manim -pql tools/manim_srfm.py SRFM
	@echo "Or: python tools/manim_srfm.py (SVG fallback)"

.PHONY: tui
tui: ## Go terminal TUI dashboard
	$(GO) run ./cmd/srfm-tui/

.PHONY: dashboard
dashboard: ## Streamlit dashboard (port 8501)
	streamlit run tools/dashboard.py --server.port 8501

.PHONY: dash
dash: ## Live Plotly Dash dashboard
	$(PYTHON) tools/dash_app.py

# ─────────────────────────────────────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: docker-up
docker-up: ## Start all services via docker-compose
	docker-compose up -d
	@echo "Services started:"
	@echo "  API:      http://localhost:8000"
	@echo "  Arena:    http://localhost:5173"
	@echo "  Gateway:  http://localhost:9000"
	@echo "  PgAdmin:  http://localhost:5050 (if configured)"

.PHONY: docker-down
docker-down: ## Stop all docker-compose services
	docker-compose down

.PHONY: docker-logs
docker-logs: ## Tail logs from all docker-compose services
	docker-compose logs -f

.PHONY: docker-build
docker-build: ## Rebuild all docker images
	docker-compose build

# ─────────────────────────────────────────────────────────────────────────────
# Advanced analytics tools
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: stress
stress: ## Stress test analysis
	$(PYTHON) tools/stress_test.py

.PHONY: margin
margin: ## Margin simulation
	$(PYTHON) tools/margin_sim.py

.PHONY: corrmon
corrmon: ## Correlation monitor
	$(PYTHON) tools/corr_monitor.py

.PHONY: paths
paths: ## Monte Carlo equity paths (1000 sims)
	$(PYTHON) tools/equity_paths.py --paths 1000

.PHONY: paths-full
paths-full: ## Monte Carlo equity paths (10000 sims)
	$(PYTHON) tools/equity_paths.py --paths 10000

.PHONY: replay
replay: ## Position size replay
	$(PYTHON) tools/size_replay.py

.PHONY: regime-stress
regime-stress: ## Regime stress test
	$(PYTHON) tools/regime_stress.py

.PHONY: dddecomp
dddecomp: ## Drawdown decomposition
	$(PYTHON) tools/dd_decomp.py

.PHONY: fees
fees: ## Fee impact analysis
	$(PYTHON) tools/fee_impact.py

.PHONY: hmm
hmm: ## HMM regime detection
	$(PYTHON) tools/hmm_regime.py --csv data/NDX_hourly_poly.csv

.PHONY: garch
garch: ## Python GARCH volatility model
	$(PYTHON) tools/garch_vol.py --csv data/NDX_hourly_poly.csv

.PHONY: features
features: ## Feature mining / ML alpha search
	$(PYTHON) tools/feature_mine.py --csv data/NDX_hourly_poly.csv

.PHONY: explain
explain: ## SHAP feature importance explainer
	$(PYTHON) tools/explainer.py

.PHONY: autotune
autotune: ## Optuna Bayesian hyperparameter optimization
	$(PYTHON) tools/autotuner.py --trials 100 --csv data/NDX_hourly_poly.csv

.PHONY: turbo
turbo: ## Rust turbo arena (50k synthetic bars, 10k trials)
	$(CARGO) build --manifest-path crates/srfm-tools/Cargo.toml --bin srfm_turbo --release
	./crates/srfm-tools/target/release/srfm_turbo --synthetic 50000 --trials 10000

.PHONY: sweep-rust
sweep-rust: ## Rust parallel parameter sweep
	$(CARGO) run --manifest-path crates/srfm-tools/Cargo.toml --bin sweep --release -- \
		--cf-range 0.001,0.015,20 --lev-range 0.30,0.80,10

.PHONY: wells
wells: ## Well detection: make wells ticker=ES csv=data/ES_hourly.csv
ifndef ticker
	$(error Usage: make wells ticker=ES csv=data/ES_hourly.csv)
endif
	$(PYTHON) tools/well_detector.py --csv $(csv) --ticker $(ticker) --plot --save-csv

.PHONY: regimes
regimes: ## Regime analysis: make regimes csv=data/ES_hourly.csv ticker=ES
ifndef csv
	$(error Usage: make regimes csv=data/ES_hourly.csv ticker=ES)
endif
	$(PYTHON) tools/regime_analyzer.py --csv $(csv) --ticker $(ticker) --plot --save-csv

.PHONY: pulse
pulse: ## BH mass progress bar (Rust): pipe CSV prices
	cat data/NDX_hourly_poly.csv | cut -d, -f5 | tail -50 | \
		./crates/srfm-tools/target/release/pulse --cf 0.005

.PHONY: drift
drift: ## Rolling correlation (Rust): make drift
	./crates/srfm-tools/target/release/drift \
		data/ES_hourly_real.csv data/NQ_hourly_real.csv --window 60 --summary

.PHONY: wasm
wasm: ## Build WebAssembly module from larsa-wasm crate
	cd crates/larsa-wasm && wasm-pack build --target web
	@echo "WASM built: crates/larsa-wasm/pkg/"

.PHONY: query
query: ## Natural language query: make query q="best performing strategies"
ifndef q
	$(PYTHON) tools/query.py schema
else
	$(PYTHON) tools/query.py ask "$(q)"
endif

.PHONY: profile
profile: ## DuckDB analytics profile
	$(PYTHON) tools/query.py profile

.PHONY: journal
journal: ## Show experiment journal
	$(PYTHON) tools/journal.py log --n 10

.PHONY: autopsy
autopsy: ## Strategy autopsy / forensic analysis
	$(PYTHON) tools/srfm_autopsy.py

.PHONY: heat
heat: ## Experiment heatmap (terminal)
	cat results/v2_experiments.json | $(PYTHON) tools/heat.py

.PHONY: well
well: ## Pipe prices → print wells
	cat data/NDX_hourly_poly.csv | $(PYTHON) tools/well.py --cf 0.005

.PHONY: edge
edge: ## v1 vs v3 P&L comparison
	$(PYTHON) tools/edge.py \
		"C:/Users/Matthew/Downloads/Calm Orange Mule_trades.csv" \
		"C:/Users/Matthew/Downloads/Measured Red Anguilline_trades.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Test suites
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: suite-v11
suite-v11: ## v11 quick analysis suite
	$(PYTHON) tools/suite_v11.py

.PHONY: suite-v11-full
suite-v11-full: ## v11 full 10-world analysis suite
	$(PYTHON) tools/suite_v11.py --full

.PHONY: test-all
test-all: suite-v11 stress margin corrmon replay regime-stress sensitivity dddecomp fees livecheck ## Run all analysis tools
	$(PYTHON) tools/dashboard_v11.py
	@echo "All analysis complete."

# ─────────────────────────────────────────────────────────────────────────────
# Docs
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: docs
docs: ## Serve documentation locally (port 8088)
	$(PYTHON) -m http.server 8088 --directory docs &
	@echo "Docs at http://localhost:8088"

# ─────────────────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove build artifacts (NOT results)
	$(CARGO) clean 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.ruff_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf spacetime/web/dist terminal/dist 2>/dev/null || true

.PHONY: clean-results
clean-results: ## Delete all backtest results (DESTRUCTIVE — prompts first)
	@echo "This will delete all backtest results. Are you sure? (ctrl-c to cancel)"; sleep 5
	rm -rf results/*/[0-9]*

# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: stats
stats: ## Run R statistical validation tests
	Rscript research/statistical_validation.R 2>/dev/null || Rscript scripts/r_tables.R
	@echo "Statistical validation -> results/graphics/"

.PHONY: arena-multi
arena-multi: ## Multi-instrument arena convergence test
	$(PYTHON) tools/arena_multi.py --mode synth --n-worlds 5 --n-bars 20000

.PHONY: larsa-core
larsa-core: ## Build larsa-core PyO3 extension
	python -m maturin build --release --manifest-path crates/larsa-core/Cargo.toml
	pip install --force-reinstall crates/larsa-core/target/wheels/larsa_core-*.whl

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "SRFM Lab — Makefile"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "%-28s %s\n", "Target", "Description"} \
		/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@echo ""
