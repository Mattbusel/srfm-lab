#!/usr/bin/env bash
echo "Running SRFM statistical validation..."
Rscript research/statistical_validation.R
echo "Done. Charts -> results/graphics/stat_*.png"
echo "Report -> results/statistical_report.md"
