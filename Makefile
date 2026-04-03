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
	@echo ""
