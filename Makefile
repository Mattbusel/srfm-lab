# SRFM Lab Makefile
# Usage examples:
#   make backtest s=larsa-v1
#   make sweep s=larsa-v1 param=BH_FORM min=0.5 max=3.0 step=0.25
#   make compare s=larsa-v1
#   make new name=experiment-1
#   make research s=larsa-v1
#   make wells ticker=ES start=20200101 end=20240101

SHELL   := bash
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# ── Backtest ──────────────────────────────────────────────────────────────────
.PHONY: backtest
backtest:
ifndef s
	$(error Usage: make backtest s=<strategy-name>)
endif
	lean backtest strategies/$(s) --output results/$(s)/$(TIMESTAMP)
	@echo ""
	@echo "Done. To compare: make compare s=$(s)"

# ── Sweep ─────────────────────────────────────────────────────────────────────
.PHONY: sweep
sweep:
ifndef s
	$(error Usage: make sweep s=<strategy> param=<PARAM> min=<min> max=<max> step=<step>)
endif
	python tools/param_sweep.py strategies/$(s) $(param) $(min) $(max) $(step)

# ── Compare ───────────────────────────────────────────────────────────────────
.PHONY: compare
compare:
ifndef s
	$(error Usage: make compare s=<strategy-name>)
endif
	python tools/compare.py results/$(s)

# ── Compare two strategies ────────────────────────────────────────────────────
.PHONY: compare2
compare2:
ifndef s1
	$(error Usage: make compare2 s1=larsa-v1 s2=larsa-v2)
endif
	python tools/compare.py results/$(s1) results/$(s2) --chart

# ── New experiment ────────────────────────────────────────────────────────────
.PHONY: new
new:
ifndef name
	$(error Usage: make new name=<experiment-name>)
endif
	./scripts/new_experiment.sh $(name)

# ── Research (Jupyter via lean research) ─────────────────────────────────────
.PHONY: research
research:
ifndef s
	$(error Usage: make research s=<strategy-name>)
endif
	lean research strategies/$(s)

# ── Well detection ────────────────────────────────────────────────────────────
.PHONY: wells
wells:
ifndef ticker
	$(error Usage: make wells ticker=ES csv=data/ES_hourly.csv)
endif
	python tools/well_detector.py --csv $(csv) --ticker $(ticker) --plot --save-csv

# ── Regime analysis ───────────────────────────────────────────────────────────
.PHONY: regimes
regimes:
ifndef csv
	$(error Usage: make regimes csv=data/ES_hourly.csv ticker=ES)
endif
	python tools/regime_analyzer.py --csv $(csv) --ticker $(ticker) --plot --save-csv

# ── Batch run ─────────────────────────────────────────────────────────────────
.PHONY: batch
batch:
ifndef s
	$(error Usage: make batch s=larsa-v1 variants=variants.json)
endif
	python tools/batch_runner.py strategies/$(s) $(variants)

# ── Data download ─────────────────────────────────────────────────────────────
.PHONY: data-init
data-init:
	lean data download --dataset "US Futures Security Master"
	@echo "Security master downloaded. Run 'lean data download --dataset ...' for price data."

# ── Install dependencies ──────────────────────────────────────────────────────
.PHONY: install
install:
	pip install lean numpy pandas matplotlib seaborn scipy

# ── Clean results ─────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo "This will delete all backtest results. Are you sure? (ctrl-c to cancel)"
	@sleep 3
	rm -rf results/*/[0-9]*

# ── Build Rust viz tool ───────────────────────────────────────────────────────
.PHONY: viz-build
viz-build:
	cd viz && cargo build --release
	@echo "srfm-viz built -> viz/target/release/srfm-viz"

# ── Build srfm-tools (pulse, drift, snap, srfm) ───────────────────────────────
.PHONY: tools-build
tools-build:
	CARGO_HOME=C:/Users/Matthew/.cargo CARGO_TARGET_DIR=C:/Users/Matthew/srfm-lab/crates/srfm-tools/target \
		C:/Users/Matthew/.cargo/bin/cargo.exe build --release --manifest-path crates/srfm-tools/Cargo.toml
	@echo "Binaries -> crates/srfm-tools/target/release/"

.PHONY: pulse
pulse:
	cat data/NDX_hourly_poly.csv | cut -d, -f5 | tail -50 | ./crates/srfm-tools/target/release/pulse --cf 0.005

.PHONY: drift
drift:
	./crates/srfm-tools/target/release/drift data/ES_hourly_real.csv data/NQ_hourly_real.csv --window 60 --summary

# ── Generate SRFM primitives (Python) ────────────────────────────────────────
.PHONY: primitives
primitives:
	python tools/primitive_builder.py --csv data/NDX_hourly_poly.csv --cf 0.005
	@echo "Primitives -> results/primitives.md"

# ── Generate lab report (Python) ─────────────────────────────────────────────
.PHONY: report
report:
	python tools/report_builder.py
	@echo "Report -> results/lab_report.md"

# ── Generate all graphics (Rust viz) ─────────────────────────────────────────
.PHONY: graphics
graphics: viz-build
	@mkdir -p results/graphics
	./viz/target/release/srfm-viz spacetime --csv data/NDX_hourly_poly.csv --cf 0.005 --out results/graphics/spacetime.svg
	./viz/target/release/srfm-viz wells --json research/trade_analysis_data.json --out results/graphics/wells_calendar.svg
	./viz/target/release/srfm-viz experiments --json results/v2_experiments.json --out results/graphics/experiments.svg
	./viz/target/release/srfm-viz equity --json research/trade_analysis_data.json --out results/graphics/equity.svg
	./viz/target/release/srfm-viz convergence --json research/trade_analysis_data.json --out results/graphics/convergence.svg
	@echo "Graphics -> results/graphics/"

# ── Full pipeline: primitives + report + graphics ────────────────────────────
.PHONY: pipeline
pipeline: primitives report graphics
	@echo ""
	@echo "Pipeline complete:"
	@echo "  results/primitives.md"
	@echo "  results/lab_report.md"
	@echo "  results/graphics/*.svg"

# ── Run experiment suite ──────────────────────────────────────────────────────
.PHONY: experiments
experiments:
	python tools/experiment_runner.py --quick
	@echo "Experiments -> results/v2_experiments.md"

.PHONY: experiments-full
experiments-full:
	python tools/experiment_runner.py
	@echo "Full experiments -> results/v2_experiments.md"

# ── Multi-instrument arena (convergence testing) ──────────────────────────────
.PHONY: arena-multi
arena-multi:
	python tools/arena_multi.py --mode synth --n-worlds 5 --n-bars 20000

# ── v3 design research ────────────────────────────────────────────────────────
.PHONY: v3-design
v3-design:
	@echo "v3 design doc -> results/v3_design.md"
	@echo "Run: python tools/experiment_runner.py then review results/v3_design.md"

.PHONY: regime-graph
regime-graph:
	python tools/regime_graph.py
	@echo "Regime graph -> results/graphics/regime_states.dot/.svg"

# ── Analytics query ───────────────────────────────────────────────────────────
.PHONY: query
query:
ifndef q
	python tools/query.py schema
else
	python tools/query.py ask "$(q)"
endif

# ── Analytics profile ─────────────────────────────────────────────────────────
.PHONY: profile
profile:
	python tools/query.py profile

# ── Statistical validation (R) ───────────────────────────────────────────────
.PHONY: stats
stats:
	Rscript research/statistical_validation.R
	@echo "Statistical validation -> results/graphics/stat_*.png"
	@echo "Report -> results/statistical_report.md"

.PHONY: html-report
html-report:
	python tools/generate_report.py
	@echo "Report -> results/strategy_report.html"

.PHONY: dashboard
dashboard:
	streamlit run tools/dashboard.py --server.port 8501

# ── Quick context tools ───────────────────────────────────────────────────────
.PHONY: what
what:
ifndef d
	@echo "Usage: make what d=2024-10-14"
else
	python tools/what.py $(d)
endif

.PHONY: blame
blame:
ifndef from
	@echo "Usage: make blame from=2024-01-01 to=2024-12-31"
else
	python tools/blame.py --from $(from) --to $(to) --csv "C:/Users/Matthew/Downloads/Measured Red Anguilline_trades.csv"
endif

.PHONY: mirror
mirror:
ifndef d
	@echo "Usage: make mirror d=2023-11-01"
else
	python tools/mirror.py --well $(d)
endif

.PHONY: cost
cost:
ifndef start
	@echo "Usage: make cost start=2022-01-15 end=2022-07-15"
else
	python tools/cost.py --start $(start) --end $(end)
endif

.PHONY: anatomy
anatomy:
	python tools/anatomy.py --top 5
	@echo "Written to ANATOMY.md"

.PHONY: heat
heat:
	cat results/v2_experiments.json | python tools/heat.py

.PHONY: well
well:
	cat data/NDX_hourly_poly.csv | python tools/well.py --cf 0.005

.PHONY: edge
edge:
	python tools/edge.py "C:/Users/Matthew/Downloads/Calm Orange Mule_trades.csv" "C:/Users/Matthew/Downloads/Measured Red Anguilline_trades.csv"

.PHONY: when
when:
ifndef p
	@echo "Usage: make when p=-265000"
else
	python tools/when.py $(p)
endif

.PHONY: regime-line
regime-line:
	cat data/NDX_hourly_poly.csv | python tools/regime.py

.PHONY: odds
odds:
	python tools/odds.py --regime BULL --bh_active true --convergence 2

.PHONY: journal
journal:
	python tools/journal.py log --n 10

.PHONY: why
why:
ifndef date
	@echo "Usage: make why date=2024-10-14 instrument=NQ"
else
	python tools/why.py --date $(date) --instrument $(instrument)
endif

.PHONY: config-check
config-check:
	python tools/srfm_config.py strategies/larsa-v4/strategy.srfm

.PHONY: help
help:
	@echo ""
	@echo "SRFM Lab — Makefile commands:"
	@echo ""
	@echo "── Strategy ──────────────────────────────────────────────"
	@echo "  make backtest s=larsa-v4          Run QC backtest"
	@echo "  make config-check                 Validate .srfm config"
	@echo ""
	@echo "── Analysis Pipeline ─────────────────────────────────────"
	@echo "  make pipeline                     Full: primitives+report+graphics"
	@echo "  make primitives                   SRFM physics metrics"
	@echo "  make report                       Lab report markdown"
	@echo "  make html-report                  Interactive HTML report"
	@echo "  make dashboard                    Streamlit dashboard (port 8501)"
	@echo "  make graphics                     All SVG graphics (Rust)"
	@echo ""
	@echo "── Forensics Tools ───────────────────────────────────────"
	@echo "  make what d=2024-10-14            What was happening on a date"
	@echo "  make blame from=2024-01-01 to=2024-12-31  P&L attribution"
	@echo "  make when p=-265000               Find trade by dollar amount"
	@echo "  make why date=2024-10-14 instrument=NQ    Why a trade happened"
	@echo "  make odds                         Historical win probability"
	@echo "  make edge                         v1 vs v3 P&L comparison"
	@echo "  make mirror d=2023-11-01          Bull well -> bear setup"
	@echo "  make cost start=X end=Y           Quantify flat period patience"
	@echo "  make anatomy                      Dissect top 5 trades"
	@echo ""
	@echo "── Data Streams ──────────────────────────────────────────"
	@echo "  make well                         Pipe prices -> print wells"
	@echo "  make regime-line                  7-year regime in one colored line"
	@echo "  make heat                         Experiment heatmap (terminal)"
	@echo "  make pulse                        BH mass progress bar (Rust)"
	@echo "  make drift                        Rolling ES/NQ correlation (Rust)"
	@echo ""
	@echo "── Experiment Tools ──────────────────────────────────────"
	@echo "  make experiments                  Quick experiment suite"
	@echo "  make experiments-full             Full 10-world suite"
	@echo "  make arena-multi                  Multi-instrument arena"
	@echo "  make profile                      DuckDB analytics profile"
	@echo "  make query q=\"...\"              Natural language query"
	@echo ""
	@echo "── Build ──────────────────────────────────────────────────"
	@echo "  make viz-build                    Build Rust srfm-viz"
	@echo "  make tools-build                  Build Rust srfm-tools"
	@echo "  make install                      Install Python deps"
	@echo "  make stats                        Run R statistical tests"
	@echo "  make journal                      Show experiment journal"
	@echo "  make regime-graph                 Graphviz state machine"
	@echo ""
