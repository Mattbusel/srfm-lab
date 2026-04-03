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

.PHONY: help
help:
	@echo ""
	@echo "SRFM Lab — Makefile commands:"
	@echo ""
	@echo "  make backtest s=larsa-v1                      Run a backtest"
	@echo "  make sweep s=larsa-v1 param=BH_FORM min=0.5 max=3.0 step=0.25"
	@echo "  make compare s=larsa-v1                       Compare all results for a strategy"
	@echo "  make compare2 s1=larsa-v1 s2=larsa-v2         Compare two strategies with chart"
	@echo "  make new name=experiment-1                    Scaffold new experiment from template"
	@echo "  make research s=larsa-v1                      Open Jupyter notebook"
	@echo "  make wells ticker=ES csv=data/ES_hourly.csv   Run well detector"
	@echo "  make regimes csv=data/ES_hourly.csv ticker=ES Run regime analyzer"
	@echo "  make batch s=larsa-v1 variants=variants.json  Batch run variants"
	@echo "  make data-init                                 Download LEAN security master"
	@echo "  make install                                   Install Python dependencies"
	@echo "  make viz-build                                Build Rust viz tool"
	@echo "  make primitives                               Compute SRFM physics primitives"
	@echo "  make report                                   Generate lab report markdown"
	@echo "  make graphics                                 Generate all SVG graphics"
	@echo "  make pipeline                                 Run full analysis pipeline"
	@echo "  make experiments                              Run experiment suite (quick)"
	@echo "  make experiments-full                         Run experiment suite (full, 10 synth worlds)"
	@echo "  make stats                                    R statistical validation (requires R + packages)"
	@echo ""
