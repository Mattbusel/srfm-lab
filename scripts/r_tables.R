#!/usr/bin/env Rscript
# r_tables.R — Publication-quality GT tables for LARSA forensics
#
# Usage: Rscript scripts/r_tables.R
# Output: results/larsa_tables.html (GT tables) or results/larsa_tables.md
#
# Auto-install if missing:
pkgs <- c("gt", "jsonlite", "dplyr")
new_pkgs <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(new_pkgs)) install.packages(new_pkgs, repos = "https://cran.rstudio.com/")

suppressPackageStartupMessages(library(jsonlite))

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH <- "research/trade_analysis_data.json"
if (!file.exists(DATA_PATH)) {
  cat("ERROR: Missing", DATA_PATH, "\n")
  quit(status = 1)
}

data  <- fromJSON(DATA_PATH)
wells <- as.data.frame(data$wells)
wells$date <- as.Date(substr(wells$start, 1, 10))
# Instruments column may be a list-column; extract first instrument
wells$instrument1 <- sapply(wells$instruments, function(x) x[1])
wells$n_instruments <- sapply(wells$instruments, length)
wells$is_convergence <- wells$n_instruments > 1

has_gt   <- requireNamespace("gt",   quietly = TRUE)
has_dplyr <- requireNamespace("dplyr", quietly = TRUE)

# ── Table 1: Annual attribution ───────────────────────────────────────────────
by_year <- do.call(rbind, lapply(names(data$by_year), function(yr) {
  d <- data$by_year[[yr]]
  data.frame(
    Year       = as.integer(yr),
    Trades     = d$count,
    WR_pct     = round(d$wins / d$count * 100, 1),
    Return_k   = round(d$pnl / 1000, 1),
    stringsAsFactors = FALSE
  )
}))
by_year <- by_year[order(by_year$Year), ]
by_year$Cumulative_k <- round(cumsum(by_year$Return_k), 1)
colnames(by_year) <- c("Year", "Trades", "Win Rate %", "Return $k", "Cumulative $k")

# ── Table 2: Instrument attribution ──────────────────────────────────────────
inst_df <- do.call(rbind, lapply(names(data$by_instrument), function(inst) {
  d <- data$by_instrument[[inst]]
  data.frame(
    Instrument = inst,
    Trades     = d$count,
    WR_pct     = round(d$wins / d$count * 100, 1),
    PnL_k      = round(d$pnl / 1000, 1),
    stringsAsFactors = FALSE
  )
}))
colnames(inst_df) <- c("Instrument", "Trades", "Win Rate %", "Net P&L $k")

# ── Table 3: Solo vs convergence ─────────────────────────────────────────────
solo_wells <- wells[!wells$is_convergence, ]
conv_wells <- wells[wells$is_convergence,  ]

solo_wr  <- mean(solo_wells$is_win) * 100
conv_wr  <- mean(conv_wells$is_win) * 100
solo_avg <- mean(solo_wells$total_pnl)
conv_avg <- mean(conv_wells$total_pnl)
solo_tot <- sum(solo_wells$total_pnl)
conv_tot <- sum(conv_wells$total_pnl)

compare_df <- data.frame(
  Type         = c("Solo (1 instrument)", "Convergence (2-3 instruments)"),
  N_Wells      = c(nrow(solo_wells), nrow(conv_wells)),
  Win_Rate_pct = c(round(solo_wr, 1), round(conv_wr, 1)),
  Avg_PnL_k    = c(round(solo_avg / 1000, 1), round(conv_avg / 1000, 1)),
  Total_PnL_k  = c(round(solo_tot / 1000, 1), round(conv_tot / 1000, 1)),
  stringsAsFactors = FALSE
)
colnames(compare_df) <- c("Type", "N Wells", "Win Rate %", "Avg P&L $k", "Total P&L $k")

# ── Table 4: Top 10 wells ─────────────────────────────────────────────────────
wells_sorted <- wells[order(-abs(wells$total_pnl)), ]
top10 <- head(wells_sorted, 10)
top10_df <- data.frame(
  Date        = format(top10$date),
  Instruments = sapply(top10$instruments, paste, collapse = "+"),
  Direction   = sapply(top10$directions, function(x) paste(unique(x), collapse = "/")),
  Duration_h  = top10$duration_h,
  PnL_k       = round(top10$total_pnl / 1000, 1),
  Win         = ifelse(top10$is_win, "WIN", "LOSS"),
  stringsAsFactors = FALSE
)
colnames(top10_df) <- c("Date", "Instruments", "Direction", "Duration (h)", "P&L $k", "Result")

# ── Render ────────────────────────────────────────────────────────────────────
dir.create("results", showWarnings = FALSE)

if (has_gt) {
  suppressPackageStartupMessages(library(gt))
  cat("Rendering GT HTML tables -> results/larsa_tables.html\n")

  render_table <- function(df, title, subtitle = "") {
    t <- gt(df) |>
      tab_header(title = title, subtitle = if (nchar(subtitle) > 0) subtitle else NULL) |>
      tab_style(
        style     = list(cell_fill(color = "#0d1117"), cell_text(color = "#c9d1d9")),
        locations = cells_body()
      ) |>
      tab_style(
        style     = list(cell_fill(color = "#161b22"), cell_text(color = "#58a6ff", weight = "bold")),
        locations = cells_column_labels()
      ) |>
      tab_style(
        style     = list(cell_fill(color = "#1f6feb"), cell_text(color = "white")),
        locations = cells_title()
      ) |>
      opt_table_font(font = list(google_font("JetBrains Mono"), default_fonts()))
    as_raw_html(t)
  }

  html_parts <- c(
    '<!DOCTYPE html><html><head>',
    '<meta charset="UTF-8">',
    '<title>LARSA Forensics Tables</title>',
    '<style>body{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:20px;}',
    'h1{color:#58a6ff;} .tbl-section{margin-bottom:40px;}</style>',
    '</head><body>',
    '<h1>LARSA v1 — Forensics Tables</h1>',
    paste0('<p>Generated: ', Sys.time(), '</p>'),
    '<div class="tbl-section">',
    render_table(by_year,     "1. Annual Attribution", "Year-by-year performance"),
    '</div><div class="tbl-section">',
    render_table(inst_df,     "2. Instrument Attribution", "ES / NQ / YM breakdown"),
    '</div><div class="tbl-section">',
    render_table(compare_df,  "3. Solo vs Convergence", "The smoking gun"),
    '</div><div class="tbl-section">',
    render_table(top10_df,    "4. Top 10 Wells by Absolute P&L"),
    '</div>',
    '</body></html>'
  )

  writeLines(html_parts, "results/larsa_tables.html")
  cat("Saved: results/larsa_tables.html\n")

} else {
  # Markdown fallback
  cat("gt not installed — writing markdown tables -> results/larsa_tables.md\n")

  md_table <- function(df) {
    header <- paste("|", paste(colnames(df), collapse = " | "), "|")
    sep    <- paste("|", paste(rep("---", ncol(df)), collapse = " | "), "|")
    rows   <- apply(df, 1, function(r) paste("|", paste(r, collapse = " | "), "|"))
    c(header, sep, rows, "")
  }

  lines <- c(
    "# LARSA v1 — Forensics Tables",
    paste("Generated:", Sys.time()), "",
    "## 1. Annual Attribution", "",
    md_table(by_year),
    "## 2. Instrument Attribution", "",
    md_table(inst_df),
    "## 3. Solo vs Convergence (Smoking Gun)", "",
    md_table(compare_df),
    "## 4. Top 10 Wells by Absolute P&L", "",
    md_table(top10_df)
  )
  writeLines(lines, "results/larsa_tables.md")
  cat("Saved: results/larsa_tables.md\n")
  cat(paste(lines, collapse = "\n"), "\n")
}
