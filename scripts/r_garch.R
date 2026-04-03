#!/usr/bin/env Rscript
# r_garch.R — DCC-GARCH multivariate volatility model for ES/NQ/YM
#
# Fits GARCH(1,1) per instrument, then DCC for dynamic correlation.
# Output: 5-day ahead vol forecasts + correlation matrix
#
# Usage: Rscript scripts/r_garch.R
# Packages: rugarch, rmgarch, jsonlite
#
# Auto-install if missing:
pkgs <- c("rugarch", "rmgarch", "jsonlite", "xts")
new_pkgs <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(new_pkgs)) install.packages(new_pkgs, repos = "https://cran.rstudio.com/")

suppressPackageStartupMessages(library(jsonlite))

# ── Load trade data ───────────────────────────────────────────────────────────
DATA_PATH <- "research/trade_analysis_data.json"
if (!file.exists(DATA_PATH)) {
  cat("ERROR: Missing", DATA_PATH, "\n")
  quit(status = 1)
}

data  <- fromJSON(DATA_PATH)
wells <- as.data.frame(data$wells)
wells$date <- as.Date(substr(wells$start, 1, 10))

CAPITAL <- 1e6

# ── Reconstruct per-instrument daily return series ────────────────────────────
make_series <- function(inst) {
  w <- wells[sapply(wells$instruments, function(x) inst %in% x), ]
  if (nrow(w) == 0) return(NULL)
  daily <- aggregate(total_pnl ~ date, data = w, FUN = sum)
  daily$ret <- daily$total_pnl / CAPITAL
  daily[order(daily$date), c("date", "ret")]
}

es_raw <- make_series("ES")
nq_raw <- make_series("NQ")
ym_raw <- make_series("YM")

# Merge on common dates
all_dates <- Reduce(intersect, list(
  as.character(es_raw$date),
  as.character(nq_raw$date),
  as.character(ym_raw$date)
))

if (length(all_dates) < 30) {
  # Fall back: use all dates with 0-fill
  all_dates_full <- seq(min(wells$date), max(wells$date), by = "day")
  merge_fill <- function(raw) {
    m <- merge(data.frame(date = all_dates_full), raw, by = "date", all.x = TRUE)
    m$ret[is.na(m$ret)] <- 0
    m
  }
  es_m <- merge_fill(es_raw)
  nq_m <- merge_fill(nq_raw)
  ym_m <- merge_fill(ym_raw)
  cat("Using zero-filled daily returns (", nrow(es_m), "observations)\n")
} else {
  filter_dates <- function(raw) raw[as.character(raw$date) %in% all_dates, ]
  es_m <- filter_dates(es_raw)
  nq_m <- filter_dates(nq_raw)
  ym_m <- filter_dates(ym_raw)
  cat("Common-date observations:", length(all_dates), "\n")
}

es_ret <- es_m$ret
nq_ret <- nq_m$ret
ym_ret <- ym_m$ret

# ── Rolling realized vol fallback ────────────────────────────────────────────
rolling_vol <- function(x, w = 20) {
  n <- length(x)
  sapply(seq_len(n), function(i) {
    start <- max(1, i - w + 1)
    sd(x[start:i]) * sqrt(252)
  })
}

rv_es <- rolling_vol(es_ret)
rv_nq <- rolling_vol(nq_ret)
rv_ym <- rolling_vol(ym_ret)

forecast_vol_fallback <- data.frame(
  Instrument = c("ES", "NQ", "YM"),
  Current_AnnVol = sprintf("%.2f%%", c(tail(rv_es, 1), tail(rv_nq, 1), tail(rv_ym, 1)) * 100),
  Avg20d_AnnVol  = sprintf("%.2f%%", c(mean(tail(rv_es, 5)), mean(tail(rv_nq, 5)), mean(tail(rv_ym, 5))) * 100)
)

simple_corr <- cor(cbind(es_ret, nq_ret, ym_ret))
rownames(simple_corr) <- colnames(simple_corr) <- c("ES", "NQ", "YM")

# ── GARCH fitting (if rugarch available) ─────────────────────────────────────
has_rugarch  <- requireNamespace("rugarch",  quietly = TRUE)
has_rmgarch  <- requireNamespace("rmgarch",  quietly = TRUE)

results_lines <- character(0)
results_lines <- c(results_lines,
  "# LARSA GARCH Volatility Forecasts",
  paste("Generated:", Sys.time()), "")

if (has_rugarch) {
  suppressPackageStartupMessages(library(rugarch))
  cat("Fitting GARCH(1,1) per instrument...\n")

  spec <- ugarchspec(
    variance.model  = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model      = list(armaOrder = c(0, 0)),
    distribution.model = "std"
  )

  fit_garch <- function(ret, name) {
    tryCatch({
      fit <- ugarchfit(spec, data = ret, solver = "hybrid")
      fcast <- ugarchforecast(fit, n.ahead = 5)
      vols  <- as.numeric(sigma(fcast)) * sqrt(252) * 100
      cat(sprintf("  %s: 5-day vol forecast: %.2f%% %.2f%% %.2f%% %.2f%% %.2f%%\n",
                  name, vols[1], vols[2], vols[3], vols[4], vols[5]))
      list(fit = fit, forecast = fcast, vols = vols)
    }, error = function(e) {
      cat("  ", name, "GARCH failed:", conditionMessage(e), "\n")
      NULL
    })
  }

  g_es <- fit_garch(es_ret, "ES")
  g_nq <- fit_garch(nq_ret, "NQ")
  g_ym <- fit_garch(ym_ret, "YM")

  results_lines <- c(results_lines, "## GARCH(1,1) 5-Day Vol Forecasts", "")
  results_lines <- c(results_lines, "| Day | ES Ann.Vol | NQ Ann.Vol | YM Ann.Vol |")
  results_lines <- c(results_lines, "|-----|-----------|-----------|-----------|")
  for (d in 1:5) {
    es_v <- if (!is.null(g_es)) sprintf("%.2f%%", g_es$vols[d]) else "N/A"
    nq_v <- if (!is.null(g_nq)) sprintf("%.2f%%", g_nq$vols[d]) else "N/A"
    ym_v <- if (!is.null(g_ym)) sprintf("%.2f%%", g_ym$vols[d]) else "N/A"
    results_lines <- c(results_lines,
      sprintf("| T+%d | %s | %s | %s |", d, es_v, nq_v, ym_v))
  }
  results_lines <- c(results_lines, "")

  # DCC model
  if (has_rmgarch && !is.null(g_es) && !is.null(g_nq) && !is.null(g_ym)) {
    suppressPackageStartupMessages(library(rmgarch))
    cat("Fitting DCC-GARCH model...\n")
    tryCatch({
      dcc_spec <- dccspec(
        uspec    = multispec(replicate(3, spec)),
        dccOrder = c(1, 1),
        distribution = "mvnorm"
      )
      ret_mat  <- cbind(es_ret, nq_ret, ym_ret)
      colnames(ret_mat) <- c("ES", "NQ", "YM")
      dcc_fit  <- dccfit(dcc_spec, data = ret_mat)
      dcc_fore <- dccforecast(dcc_fit, n.ahead = 5)
      R_fore   <- rcor(dcc_fore)
      R_t1     <- R_fore[, , 1, 1]  # day 1 correlation matrix
      rownames(R_t1) <- colnames(R_t1) <- c("ES", "NQ", "YM")

      results_lines <- c(results_lines,
        "## DCC Dynamic Correlation Forecast (T+1)", "",
        "```")
      for (i in 1:3) {
        results_lines <- c(results_lines,
          paste(sprintf("%6s", round(R_t1[i, ], 3)), collapse = "  "))
      }
      results_lines <- c(results_lines, "```", "")
    }, error = function(e) {
      cat("DCC fitting failed:", conditionMessage(e), "\n")
    })
  }

} else {
  cat("rugarch not installed — using rolling realized vol fallback.\n")
  results_lines <- c(results_lines,
    "## Rolling Realized Volatility (20-bar, annualised)", "")
  results_lines <- c(results_lines, "| Instrument | Current AnnVol | Avg 5-day |")
  results_lines <- c(results_lines, "|-----------|---------------|----------|")
  for (i in seq_len(nrow(forecast_vol_fallback))) {
    results_lines <- c(results_lines,
      sprintf("| %s | %s | %s |",
              forecast_vol_fallback$Instrument[i],
              forecast_vol_fallback$Current_AnnVol[i],
              forecast_vol_fallback$Avg20d_AnnVol[i]))
  }
  results_lines <- c(results_lines, "")
}

# ── Simple correlation ────────────────────────────────────────────────────────
results_lines <- c(results_lines, "## Simple Correlation Matrix (full sample)", "", "```")
for (i in 1:3) {
  results_lines <- c(results_lines,
    paste(sprintf("%8s", round(simple_corr[i, ], 4)), collapse = ""))
}
results_lines <- c(results_lines, "```", "")

# ── Save ──────────────────────────────────────────────────────────────────────
dir.create("results", showWarnings = FALSE)
writeLines(results_lines, "results/garch_forecasts.md")
cat("\nSaved: results/garch_forecasts.md\n")
cat(paste(results_lines, collapse = "\n"), "\n")
