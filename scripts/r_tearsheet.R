#!/usr/bin/env Rscript
# r_tearsheet.R — PerformanceAnalytics institutional tearsheet for LARSA
#
# Usage:
#   Rscript scripts/r_tearsheet.R                    # uses trade_analysis_data.json
#   Rscript scripts/r_tearsheet.R results/returns.csv
#
# Packages: PerformanceAnalytics, ggplot2, patchwork, gt, jsonlite, xts
# Auto-install if missing:
pkgs <- c("PerformanceAnalytics", "ggplot2", "patchwork", "gt", "jsonlite", "xts")
new_pkgs <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(new_pkgs)) install.packages(new_pkgs, repos = "https://cran.rstudio.com/")

suppressPackageStartupMessages({
  library(jsonlite)
})

# ── Load trade data ──────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
DATA_PATH <- "research/trade_analysis_data.json"

if (!file.exists(DATA_PATH)) {
  cat("ERROR: Missing", DATA_PATH, "\n")
  cat("Run: python tools/trade_forensics.py first\n")
  quit(status = 1)
}

cat("Loading trade data from", DATA_PATH, "\n")
data <- fromJSON(DATA_PATH)
wells <- as.data.frame(data$wells)

# ── Reconstruct daily returns from wells ($1M capital) ───────────────────────
CAPITAL <- 1e6

# Parse dates
wells$date <- as.Date(substr(wells$start, 1, 10))
wells$ret   <- wells$total_pnl / CAPITAL

# Aggregate P&L by date
daily_pnl <- aggregate(ret ~ date, data = wells, FUN = sum)
daily_pnl <- daily_pnl[order(daily_pnl$date), ]

# Fill in calendar dates so xts is continuous
all_dates <- seq(min(daily_pnl$date), max(daily_pnl$date), by = "day")
ret_full  <- merge(data.frame(date = all_dates), daily_pnl, by = "date", all.x = TRUE)
ret_full$ret[is.na(ret_full$ret)] <- 0

# ── Convert to xts ───────────────────────────────────────────────────────────
has_xts <- requireNamespace("xts", quietly = TRUE)
returns_xts <- NULL
if (has_xts) {
  suppressPackageStartupMessages(library(xts))
  returns_xts <- xts(ret_full$ret, order.by = ret_full$date)
  colnames(returns_xts) <- "LARSA"
}

# ── Performance metrics ──────────────────────────────────────────────────────
r    <- ret_full$ret
n    <- length(r)
nyrs <- as.numeric(diff(range(ret_full$date))) / 365.25

# CAGR
cum_ret <- prod(1 + r) - 1
cagr    <- (1 + cum_ret)^(1 / nyrs) - 1

# Annualised vol (252-day calendar scaling for daily)
ann_vol <- sd(r) * sqrt(252)

# Sharpe (assuming 0 risk-free)
sharpe <- mean(r) / sd(r) * sqrt(252)

# Sortino (downside deviation)
neg_r   <- r[r < 0]
downside_sd <- sqrt(mean(neg_r^2)) * sqrt(252)
sortino <- mean(r) / sqrt(mean(pmin(r, 0)^2)) * sqrt(252)

# Max drawdown
cum_curve <- cumprod(1 + r)
roll_max  <- cummax(cum_curve)
drawdowns <- (cum_curve - roll_max) / roll_max
max_dd    <- min(drawdowns)

# Calmar
calmar <- cagr / abs(max_dd)

# Win rate
win_rate <- mean(r > 0)

# Omega ratio (threshold = 0)
pos_sum <- sum(r[r > 0])
neg_sum <- abs(sum(r[r < 0]))
omega   <- if (neg_sum > 0) pos_sum / neg_sum else Inf

# ── Print results ────────────────────────────────────────────────────────────
cat("\n")
cat("════════════════════════════════════════════════════════\n")
cat("  LARSA v1 — Institutional Tearsheet                   \n")
cat("════════════════════════════════════════════════════════\n")
metrics <- data.frame(
  Metric = c("CAGR", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
             "Max Drawdown", "Win Rate", "Omega Ratio", "Ann. Volatility",
             "Cumulative Return", "Years"),
  Value  = c(
    sprintf("%.2f%%", cagr * 100),
    sprintf("%.3f",   sharpe),
    sprintf("%.3f",   sortino),
    sprintf("%.3f",   calmar),
    sprintf("%.2f%%", max_dd * 100),
    sprintf("%.2f%%", win_rate * 100),
    sprintf("%.3f",   omega),
    sprintf("%.2f%%", ann_vol * 100),
    sprintf("%.2f%%", cum_ret * 100),
    sprintf("%.1f",   nyrs)
  )
)

for (i in seq_len(nrow(metrics))) {
  cat(sprintf("  %-22s %s\n", metrics$Metric[i], metrics$Value[i]))
}
cat("════════════════════════════════════════════════════════\n\n")

# ── PerformanceAnalytics charts (if available) ───────────────────────────────
has_pa <- requireNamespace("PerformanceAnalytics", quietly = TRUE)
if (has_pa && !is.null(returns_xts)) {
  suppressPackageStartupMessages(library(PerformanceAnalytics))
  cat("Generating PerformanceAnalytics tearsheet -> results/r_tearsheet.pdf\n")
  dir.create("results", showWarnings = FALSE)
  tryCatch({
    pdf("results/r_tearsheet.pdf", width = 11, height = 8.5)
    charts.PerformanceSummary(returns_xts, main = "LARSA v1 — Performance Summary",
                               colorset = rich6equal, lwd = 2)
    dev.off()
    cat("PDF saved: results/r_tearsheet.pdf\n")
  }, error = function(e) {
    cat("PDF generation failed:", conditionMessage(e), "\n")
  })
} else {
  cat("PerformanceAnalytics not available; skipping PDF chart.\n")
}

# ── GT formatted table (if available) ────────────────────────────────────────
has_gt <- requireNamespace("gt", quietly = TRUE)
if (has_gt) {
  suppressPackageStartupMessages(library(gt))
  cat("\nFormatting GT table...\n")
  tryCatch({
    tbl <- gt(metrics) |>
      tab_header(title = "LARSA v1 Performance Metrics",
                 subtitle = paste("Period:", format(min(daily_pnl$date)), "to",
                                  format(max(daily_pnl$date)))) |>
      cols_label(Metric = "Metric", Value = "Value") |>
      tab_style(style = cell_fill(color = "#1a1a2e"),
                locations = cells_body()) |>
      tab_style(style = cell_text(color = "#e0e0ff", weight = "bold"),
                locations = cells_body(columns = "Metric")) |>
      tab_style(style = cell_text(color = "#00ff88"),
                locations = cells_body(columns = "Value"))
    cat(as_raw_html(tbl), file = "results/r_tearsheet_table.html")
    cat("GT table saved: results/r_tearsheet_table.html\n")
  }, error = function(e) {
    cat("GT table failed:", conditionMessage(e), "\n")
  })
}

# ── Save plain-text tearsheet ─────────────────────────────────────────────────
dir.create("results", showWarnings = FALSE)
sink("results/r_tearsheet.txt")
cat("LARSA v1 — R Tearsheet\n")
cat(paste("Generated:", Sys.time()), "\n\n")
for (i in seq_len(nrow(metrics))) {
  cat(sprintf("  %-22s %s\n", metrics$Metric[i], metrics$Value[i]))
}
sink()
cat("Text tearsheet saved: results/r_tearsheet.txt\n")
