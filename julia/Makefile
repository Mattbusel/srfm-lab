# SRFM Julia Quantitative Trading System
# Build targets for precompilation, testing, and maintenance

JULIA        ?= julia
JULIA_FLAGS  ?= --project=. --threads=auto
PRECOMP_DIR  := precompile_cache
SYSIMAGE     := srfm_julia.so

.PHONY: all precompile test test-verbose lint clean sysimage deps check-julia help

# Default target
all: deps precompile test

# ---------------------------------------------------------------------------
# Dependency check and installation
# ---------------------------------------------------------------------------

deps:
	@echo "==> Installing/verifying Julia dependencies..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		import Pkg; \
		Pkg.instantiate(); \
		Pkg.resolve(); \
		println("Dependencies OK")'

# ---------------------------------------------------------------------------
# Precompilation
# ---------------------------------------------------------------------------

precompile:
	@echo "==> Precompiling SRFM Julia modules..."
	@mkdir -p $(PRECOMP_DIR)
	$(JULIA) $(JULIA_FLAGS) -e '\
		include("AdvancedOptions.jl"); \
		include("LiveRisk.jl"); \
		include("MLSignals.jl"); \
		include("Backtesting.jl"); \
		include("PortfolioOptimization.jl"); \
		include("TimeSeriesModels.jl"); \
		include("RiskAttribution.jl"); \
		include("MarketMicrostructure.jl"); \
		println("All modules loaded successfully")'

# ---------------------------------------------------------------------------
# System image (PackageCompiler.jl) - dramatically reduces startup time
# ---------------------------------------------------------------------------

sysimage:
	@echo "==> Building system image $(SYSIMAGE)..."
	@echo "==> Requires PackageCompiler.jl to be installed."
	$(JULIA) $(JULIA_FLAGS) -e '\
		using PackageCompiler; \
		create_sysimage( \
			[:Distributions, :LinearAlgebra, :Statistics, :Optim], \
			sysimage_path="$(SYSIMAGE)", \
			precompile_execution_file="tests/runtests.jl" \
		); \
		println("System image built: $(SYSIMAGE)")'

# Run with sysimage
run-sysimage:
	$(JULIA) -J$(SYSIMAGE) $(JULIA_FLAGS) -e 'include("tests/runtests.jl")'

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

test:
	@echo "==> Running SRFM Julia test suite..."
	$(JULIA) $(JULIA_FLAGS) tests/runtests.jl

test-verbose:
	@echo "==> Running SRFM Julia test suite (verbose)..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		using Test; \
		Test.runtests(nothing, verbose=true); \
		include("tests/runtests.jl")'

test-options:
	@echo "==> Running AdvancedOptions tests only..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		include("AdvancedOptions.jl"); \
		using .AdvancedOptions; \
		using Test; \
		include("tests/runtests.jl")'

test-risk:
	@echo "==> Running LiveRisk tests only..."
	$(JULIA) $(JULIA_FLAGS) -e 'include("tests/runtests.jl")'

test-ml:
	@echo "==> Running MLSignals tests only..."
	$(JULIA) $(JULIA_FLAGS) -e 'include("tests/runtests.jl")'

test-backtest:
	@echo "==> Running Backtesting tests only..."
	$(JULIA) $(JULIA_FLAGS) -e 'include("tests/runtests.jl")'

# ---------------------------------------------------------------------------
# New analytics module targets
# ---------------------------------------------------------------------------

portfolio:
	@echo "==> Checking PortfolioOptimization module..."
	$(JULIA) $(JULIA_FLAGS) -e 'include("PortfolioOptimization.jl"); using .PortfolioOptimization; println("OK")'

timeseries:
	@echo "==> Checking TimeSeriesModels module..."
	$(JULIA) $(JULIA_FLAGS) -e 'include("TimeSeriesModels.jl"); using .TimeSeriesModels; println("OK")'

riskattribution:
	@echo "==> Checking RiskAttribution module..."
	$(JULIA) $(JULIA_FLAGS) -e 'include("RiskAttribution.jl"); using .RiskAttribution; println("OK")'

microstructure:
	@echo "==> Checking MarketMicrostructure module..."
	$(JULIA) $(JULIA_FLAGS) -e 'include("MarketMicrostructure.jl"); using .MarketMicrostructure; println("OK")'

test_portfolio:
	@echo "==> Running PortfolioOptimization tests..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		include("PortfolioOptimization.jl"); \
		using .PortfolioOptimization; \
		PortfolioOptimization.run_tests()'

test_timeseries:
	@echo "==> Running TimeSeriesModels tests..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		include("TimeSeriesModels.jl"); \
		using .TimeSeriesModels; \
		TimeSeriesModels.run_tests()'

test_riskattribution:
	@echo "==> Running RiskAttribution tests..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		include("RiskAttribution.jl"); \
		using .RiskAttribution; \
		RiskAttribution.run_tests()'

test_microstructure:
	@echo "==> Running MarketMicrostructure tests..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		include("MarketMicrostructure.jl"); \
		using .MarketMicrostructure; \
		MarketMicrostructure.run_tests()'

test_analytics: test_portfolio test_timeseries test_riskattribution test_microstructure
	@echo "==> All analytics module tests complete."

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

bench-options:
	@echo "==> Benchmarking AdvancedOptions..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		include("AdvancedOptions.jl"); \
		using .AdvancedOptions; \
		@time for _ in 1:1000; \
			heston_call_price(100.0, 100.0, 0.05, 0.0, 1.0, 2.0, 0.04, 0.3, -0.7, 0.04); \
		end; \
		println("1000 Heston pricing calls complete")'

bench-risk:
	@echo "==> Benchmarking LiveRisk..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		include("LiveRisk.jl"); \
		using .LiveRisk; \
		using Random; \
		R = randn(500, 10); \
		w = ones(10) / 10; \
		@time for _ in 1:100; \
			parametric_var(w, R, 0.99); \
		end; \
		println("100 parametric VaR computations complete")'

bench-backtest:
	@echo "==> Benchmarking Backtesting engine..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		include("Backtesting.jl"); \
		using .Backtesting; \
		using Random; \
		prices = exp.(cumsum(randn(1000, 5) * 0.01, dims=1)); \
		signals = randn(1000, 5); \
		config = BacktestConfig(); \
		@time run_backtest(prices, signals, config); \
		println("1000-bar backtest complete")'

bench-all: bench-options bench-risk bench-backtest

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:
	@echo "==> Linting Julia source files..."
	@for f in AdvancedOptions.jl LiveRisk.jl MLSignals.jl Backtesting.jl \
	           PortfolioOptimization.jl TimeSeriesModels.jl RiskAttribution.jl MarketMicrostructure.jl; do \
		echo "Checking $$f..."; \
		$(JULIA) $(JULIA_FLAGS) -e "include(\"$$f\"); println(\"$$f OK\")"; \
	done

format:
	@echo "==> Formatting Julia source files (requires JuliaFormatter.jl)..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		using JuliaFormatter; \
		for f in ["AdvancedOptions.jl","LiveRisk.jl","MLSignals.jl","Backtesting.jl", \
		          "PortfolioOptimization.jl","TimeSeriesModels.jl","RiskAttribution.jl","MarketMicrostructure.jl"]; \
			format_file(f); \
			println("Formatted: $$f"); \
		end'

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

docs:
	@echo "==> Generating documentation (requires Documenter.jl)..."
	$(JULIA) $(JULIA_FLAGS) -e '\
		using Documenter; \
		include("AdvancedOptions.jl"); \
		include("LiveRisk.jl"); \
		include("MLSignals.jl"); \
		include("Backtesting.jl"); \
		using .AdvancedOptions, .LiveRisk, .MLSignals, .Backtesting; \
		makedocs(sitename="SRFM Julia Quant", modules=[AdvancedOptions, LiveRisk, MLSignals, Backtesting])'

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	@echo "==> Cleaning build artifacts..."
	rm -rf $(PRECOMP_DIR)
	rm -f $(SYSIMAGE)
	rm -rf .julia_cache
	find . -name "*.ji" -delete
	find . -name "*.o" -delete
	@echo "Clean complete."

clean-all: clean
	rm -rf Manifest.toml

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

check-julia:
	@$(JULIA) --version
	@echo "Julia path: $$(which $(JULIA))"

repl:
	@echo "==> Starting Julia REPL with all modules loaded..."
	$(JULIA) $(JULIA_FLAGS) -i -e '\
		include("AdvancedOptions.jl"); \
		include("LiveRisk.jl"); \
		include("MLSignals.jl"); \
		include("Backtesting.jl"); \
		using .AdvancedOptions, .LiveRisk, .MLSignals, .Backtesting; \
		println("All SRFM modules loaded. Happy trading!")'

# LOC count
wc:
	@echo "==> Line counts:"
	@wc -l AdvancedOptions.jl LiveRisk.jl MLSignals.jl Backtesting.jl \
	       PortfolioOptimization.jl TimeSeriesModels.jl RiskAttribution.jl MarketMicrostructure.jl \
	       tests/runtests.jl Makefile

help:
	@echo "SRFM Julia Quantitative Trading System - Build targets:"
	@echo ""
	@echo "  all                - Install deps, precompile, and run tests (default)"
	@echo "  deps               - Install/verify Julia dependencies"
	@echo "  precompile         - Precompile all modules"
	@echo "  sysimage           - Build PackageCompiler system image (fast startup)"
	@echo "  test               - Run full test suite"
	@echo "  test-verbose       - Run tests with verbose output"
	@echo "  bench-all          - Run all benchmarks"
	@echo "  bench-options      - Benchmark AdvancedOptions"
	@echo "  bench-risk         - Benchmark LiveRisk"
	@echo "  bench-backtest     - Benchmark Backtesting engine"
	@echo "  lint               - Syntax-check all modules"
	@echo "  format             - Auto-format (requires JuliaFormatter.jl)"
	@echo "  docs               - Generate docs (requires Documenter.jl)"
	@echo "  clean              - Remove build artifacts"
	@echo "  clean-all          - Remove artifacts + Manifest.toml"
	@echo "  repl               - Start REPL with all modules pre-loaded"
	@echo "  wc                 - Show line counts"
	@echo "  check-julia        - Print Julia version and path"
	@echo ""
	@echo "Analytics module targets:"
	@echo "  portfolio          - Load-check PortfolioOptimization"
	@echo "  timeseries         - Load-check TimeSeriesModels"
	@echo "  riskattribution    - Load-check RiskAttribution"
	@echo "  microstructure     - Load-check MarketMicrostructure"
	@echo "  test_portfolio     - Run PortfolioOptimization tests"
	@echo "  test_timeseries    - Run TimeSeriesModels tests"
	@echo "  test_riskattribution - Run RiskAttribution tests"
	@echo "  test_microstructure - Run MarketMicrostructure tests"
	@echo "  test_analytics     - Run all analytics module tests"
