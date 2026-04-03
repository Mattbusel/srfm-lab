#!/usr/bin/env bash
set -e
echo "=== SRFM Lab Pipeline ==="
echo ""
echo "[1/3] Computing primitives..."
python tools/primitive_builder.py "$@"
echo ""
echo "[2/3] Building lab report..."
python tools/report_builder.py
echo ""
echo "[3/3] Building Rust viz + graphics..."
cd viz && cargo build --release --quiet && cd ..
mkdir -p results/graphics
./viz/target/release/srfm-viz spacetime --csv data/NDX_hourly_poly.csv --cf 0.005 --out results/graphics/spacetime.svg 2>/dev/null || echo "  [skip] spacetime (needs data)"
./viz/target/release/srfm-viz wells --json research/trade_analysis_data.json --out results/graphics/wells_calendar.svg 2>/dev/null || echo "  [skip] wells"
./viz/target/release/srfm-viz experiments --json results/v2_experiments.json --out results/graphics/experiments.svg 2>/dev/null || echo "  [skip] experiments"
./viz/target/release/srfm-viz equity --json research/trade_analysis_data.json --out results/graphics/equity.svg 2>/dev/null || echo "  [skip] equity"
./viz/target/release/srfm-viz convergence --json research/trade_analysis_data.json --out results/graphics/convergence.svg 2>/dev/null || echo "  [skip] convergence"
echo ""
echo "Done. Outputs:"
echo "  results/primitives.md"
echo "  results/lab_report.md"
echo "  results/graphics/*.svg"
