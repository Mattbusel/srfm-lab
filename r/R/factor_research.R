# factor_research.R
# Fama-French factor construction and analysis for SRFM research.
# Implements: FF3/FF5 construction from scratch, cross-sectional regression,
#             IC/ICIR, factor decay, Newey-West SEs, factor zoo comparison.
# Dependencies: base R, xts, zoo, sandwich, lmtest, ggplot2, dplyr

library(xts)
library(zoo)
library(ggplot2)
library(dplyr, warn.conflicts = FALSE)

# Fallback: sandwich / lmtest may not be installed in all environments
.has_sandwich  <- requireNamespace("sandwich",  quietly = TRUE)
.has_lmtest    <- requireNamespace("lmtest",    quietly = TRUE)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Factor Construction from Scratch
# ─────────────────────────────────────────────────────────────────────────────

#' build_ff_factors
#'
#' Construct Fama-French 3 (and optional 5) factors from a panel of stock data.
#'
#' @param panel data.frame with columns:
#'   date, stock_id, ret (return), mktcap (market cap at previous month-end),
#'   beme (book-to-market), op (operating profitability), inv (investment)
#'   All columns in daily or monthly frequency.
#' @param rf_series xts or named numeric: risk-free rate series (dates matching panel)
#' @param factors character vector: which factors to build
#'   c("MKT","SMB","HML","RMW","CMA")
#' @return list with:
#'   $factors: xts of factor returns
#'   $portfolio_stats: data.frame of portfolio characteristics
build_ff_factors <- function(
  panel,
  rf_series,
  factors = c("MKT", "SMB", "HML", "RMW", "CMA")
) {
  stopifnot(is.data.frame(panel))
  required_cols <- c("date", "stock_id", "ret", "mktcap", "beme")
  missing_cols <- setdiff(required_cols, names(panel))
  if (length(missing_cols) > 0) {
    stop("panel is missing columns: ", paste(missing_cols, collapse = ", "))
  }

  panel <- panel[!is.na(panel$ret) & !is.na(panel$mktcap) & !is.na(panel$beme), ]
  panel <- panel[panel$mktcap > 0, ]
  dates <- sort(unique(panel$date))

  # Align rf to dates
  if (is.xts(rf_series)) {
    rf_df <- data.frame(date = as.Date(index(rf_series)), rf = as.numeric(rf_series))
  } else {
    rf_df <- data.frame(date = as.Date(names(rf_series)), rf = as.numeric(rf_series))
  }

  factor_list <- lapply(dates, function(d) {
    sub <- panel[panel$date == d, ]
    if (nrow(sub) < 10) return(NULL)

    # Size breakpoint: NYSE median market cap
    # (If exchange info not available, use full-sample median as proxy)
    size_bp <- median(sub$mktcap, na.rm = TRUE)

    # B/M breakpoints: 30th / 70th percentile
    beme_30 <- quantile(sub$beme, 0.30, na.rm = TRUE)
    beme_70 <- quantile(sub$beme, 0.70, na.rm = TRUE)

    # Size portfolios
    small <- sub[sub$mktcap < size_bp, ]
    big   <- sub[sub$mktcap >= size_bp, ]

    # 6 portfolios: S/L, S/M, S/H, B/L, B/M, B/H
    SL <- small[small$beme < beme_30, ]
    SM <- small[small$beme >= beme_30 & small$beme < beme_70, ]
    SH <- small[small$beme >= beme_70, ]
    BL <- big[big$beme < beme_30, ]
    BM <- big[big$beme >= beme_30 & big$beme < beme_70, ]
    BH <- big[big$beme >= beme_70, ]

    wt_ret <- function(df) {
      if (nrow(df) == 0) return(NA_real_)
      w <- df$mktcap / sum(df$mktcap)
      sum(w * df$ret)
    }

    r_SL <- wt_ret(SL); r_SM <- wt_ret(SM); r_SH <- wt_ret(SH)
    r_BL <- wt_ret(BL); r_BM <- wt_ret(BM); r_BH <- wt_ret(BH)

    # SMB = average small minus average big
    smb <- mean(c(r_SL, r_SM, r_SH), na.rm = TRUE) -
           mean(c(r_BL, r_BM, r_BH), na.rm = TRUE)
    # HML = average high B/M minus average low B/M
    hml <- mean(c(r_SH, r_BH), na.rm = TRUE) -
           mean(c(r_SL, r_BL), na.rm = TRUE)

    # Market excess return
    mkt_ret <- weighted.mean(sub$ret, sub$mktcap, na.rm = TRUE)
    rf_val  <- {
      rf_row <- rf_df[rf_df$date == d, "rf"]
      if (length(rf_row) == 0 || is.na(rf_row[1])) 0.0 else rf_row[1]
    }
    mkt <- mkt_ret - rf_val

    result <- c(date = as.numeric(d), MKT = mkt, SMB = smb, HML = hml,
                RF = rf_val)

    # RMW (profitability) and CMA (investment) — only if columns available
    if ("RMW" %in% factors && "op" %in% names(sub)) {
      op_50 <- median(sub$op, na.rm = TRUE)
      robust  <- sub[sub$op >= op_50, ]
      weak    <- sub[sub$op <  op_50, ]
      result["RMW"] <- wt_ret(robust) - wt_ret(weak)
    }
    if ("CMA" %in% factors && "inv" %in% names(sub)) {
      inv_50  <- median(sub$inv, na.rm = TRUE)
      conserv <- sub[sub$inv <= inv_50, ]
      aggr    <- sub[sub$inv >  inv_50, ]
      result["CMA"] <- wt_ret(conserv) - wt_ret(aggr)
    }

    result
  })

  factor_list <- Filter(Negate(is.null), factor_list)
  if (length(factor_list) == 0) {
    stop("No valid dates in panel to construct factors.")
  }

  fmat <- do.call(rbind, factor_list)
  factor_dates <- as.Date(fmat[, "date"], origin = "1970-01-01")
  factor_vals  <- fmat[, setdiff(colnames(fmat), "date"), drop = FALSE]

  factors_xts <- xts(factor_vals, order.by = factor_dates)
  return(list(factors = factors_xts))
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Cross-Sectional Regression (Fama-MacBeth)
# ─────────────────────────────────────────────────────────────────────────────

#' fama_macbeth_regression
#'
#' Run Fama-MacBeth two-pass cross-sectional regression.
#' Pass 1: time-series regression of each stock on factors → betas
#' Pass 2: cross-sectional regression of returns on betas each period → lambdas
#'
#' @param ret_panel matrix n_periods × n_stocks of returns
#' @param factor_returns matrix n_periods × n_factors of factor returns
#' @param newey_west_lags integer, lags for Newey-West SE (0 = OLS SE)
#' @return list with:
#'   $lambdas: matrix of period-by-period risk premia
#'   $mean_lambdas: vector of average risk premia
#'   $se: standard errors (Newey-West if requested)
#'   $t_stats: t-statistics
#'   $betas: matrix n_stocks × n_factors of estimated betas
fama_macbeth_regression <- function(
  ret_panel,
  factor_returns,
  newey_west_lags = 4L
) {
  n_t <- nrow(ret_panel)
  n_s <- ncol(ret_panel)
  n_f <- ncol(factor_returns)

  stopifnot(nrow(factor_returns) == n_t)

  # Pass 1: Time-series beta estimation
  betas <- matrix(NA_real_, n_s, n_f + 1L)   # +1 for intercept
  for (j in seq_len(n_s)) {
    y_j <- ret_panel[, j]
    valid <- !is.na(y_j)
    if (sum(valid) < n_f + 5L) next
    X_j <- cbind(1, factor_returns[valid, ])
    b_j <- tryCatch(
      solve(t(X_j) %*% X_j, t(X_j) %*% y_j[valid]),
      error = function(e) rep(NA_real_, n_f + 1L)
    )
    betas[j, ] <- b_j
  }
  colnames(betas) <- c("alpha", paste0("beta_", colnames(factor_returns)))

  # Pass 2: Cross-sectional regression each period
  lambdas <- matrix(NA_real_, n_t, n_f)
  colnames(lambdas) <- colnames(factor_returns)

  for (t in seq_len(n_t)) {
    y_t   <- ret_panel[t, ]
    valid <- !is.na(y_t) & apply(!is.na(betas[, -1, drop = FALSE]), 1, all)
    if (sum(valid) < n_f + 3L) next

    X_cs    <- betas[valid, -1, drop = FALSE]   # betas, no intercept in cross-section
    y_cs    <- y_t[valid]
    b_cs    <- tryCatch(
      solve(t(X_cs) %*% X_cs, t(X_cs) %*% y_cs),
      error = function(e) rep(NA_real_, n_f)
    )
    lambdas[t, ] <- b_cs
  }

  mean_lams <- colMeans(lambdas, na.rm = TRUE)

  # Standard errors
  if (newey_west_lags > 0) {
    se <- apply(lambdas, 2, function(l) {
      newey_west_se(l[!is.na(l)], lags = newey_west_lags)
    })
  } else {
    n_valid <- colSums(!is.na(lambdas))
    se <- apply(lambdas, 2, function(l) sd(l, na.rm = TRUE) / sqrt(sum(!is.na(l))))
  }

  t_stats <- mean_lams / (se + 1e-10)

  list(
    lambdas      = lambdas,
    mean_lambdas = mean_lams,
    se           = se,
    t_stats      = t_stats,
    betas        = betas,
    n_periods    = n_t,
    n_stocks     = n_s
  )
}


#' newey_west_se
#'
#' Compute Newey-West HAC standard error for a vector.
#'
#' @param x numeric vector
#' @param lags integer, number of lags (Bartlett kernel)
#' @return scalar SE estimate
newey_west_se <- function(x, lags = 4L) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (n < 3L) return(sd(x) / sqrt(n))

  x_c <- x - mean(x)
  gamma0 <- mean(x_c^2)

  nw_sum <- gamma0
  for (l in seq_len(lags)) {
    gamma_l <- mean(x_c[(l+1):n] * x_c[1:(n-l)])
    w_l <- 1.0 - l / (lags + 1.0)   # Bartlett weight
    nw_sum <- nw_sum + 2 * w_l * gamma_l
  }

  sqrt(max(nw_sum, 0) / n)
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Information Coefficient Analysis
# ─────────────────────────────────────────────────────────────────────────────

#' compute_ic_series
#'
#' Compute Information Coefficient (IC) time series for a factor signal.
#' IC_t = rank_correlation(factor_t, return_{t+1})
#'
#' @param factor_panel matrix n_periods × n_stocks of factor values
#' @param ret_panel matrix n_periods × n_stocks of forward returns
#' @param method character "spearman" or "pearson"
#' @return list with $ic (vector), $ic_dates (factor_dates)
compute_ic_series <- function(factor_panel, ret_panel, method = "spearman") {
  stopifnot(dim(factor_panel) == dim(ret_panel))
  n_t <- nrow(factor_panel)

  ic <- numeric(n_t)
  for (t in seq_len(n_t)) {
    f_t  <- factor_panel[t, ]
    r_t1 <- ret_panel[t, ]
    valid <- !is.na(f_t) & !is.na(r_t1)
    if (sum(valid) < 10L) {
      ic[t] <- NA_real_
      next
    }
    ic[t] <- tryCatch(
      cor(f_t[valid], r_t1[valid], method = method),
      error = function(e) NA_real_
    )
  }
  ic
}

#' compute_icir
#'
#' Compute ICIR = mean(IC) / sd(IC) (annualised if annual_periods specified).
#'
#' @param ic_series numeric vector
#' @param annual_periods integer (252=daily, 12=monthly)
#' @return list with mean_ic, sd_ic, icir, t_stat
compute_icir <- function(ic_series, annual_periods = 252L) {
  ic <- ic_series[!is.na(ic_series)]
  if (length(ic) < 2L) {
    return(list(mean_ic = NA, sd_ic = NA, icir = NA, t_stat = NA))
  }
  mu  <- mean(ic)
  sig <- sd(ic)
  icir <- if (sig > 0) mu / sig * sqrt(annual_periods) else NA_real_
  t_stat <- mu / (sig / sqrt(length(ic)) + 1e-10)
  list(mean_ic = mu, sd_ic = sig, icir = icir, t_stat = t_stat,
       n_obs = length(ic))
}


#' factor_decay_analysis
#'
#' Compute IC at multiple forward return horizons to measure factor signal decay.
#'
#' @param factor_panel matrix n_periods × n_stocks
#' @param ret_panel matrix n_periods × n_stocks (single-period returns)
#' @param horizons integer vector of forward horizons to test (e.g. 1,5,10,22,44)
#' @return data.frame with columns: horizon, mean_ic, icir, t_stat, se
factor_decay_analysis <- function(
  factor_panel,
  ret_panel,
  horizons = c(1L, 5L, 10L, 22L, 44L, 66L)
) {
  results <- lapply(horizons, function(h) {
    n_t <- nrow(ret_panel)
    ret_h <- matrix(NA_real_, n_t, ncol(ret_panel))
    for (t in seq_len(n_t)) {
      end_t <- min(t + h - 1L, n_t)
      # Compound returns
      ret_h[t, ] <- colMeans(ret_panel[t:end_t, , drop = FALSE], na.rm = TRUE) * h
    }
    ic_h <- compute_ic_series(factor_panel, ret_h)
    ic_stats <- compute_icir(ic_h)
    data.frame(
      horizon  = h,
      mean_ic  = ic_stats$mean_ic,
      icir     = ic_stats$icir,
      t_stat   = ic_stats$t_stat,
      se       = if (!is.null(ic_stats$sd_ic)) ic_stats$sd_ic / sqrt(max(ic_stats$n_obs, 1)) else NA_real_,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, results)
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Factor Zoo Comparison
# ─────────────────────────────────────────────────────────────────────────────

#' factor_zoo_comparison
#'
#' Compare multiple factors for redundancy / incremental contribution.
#' Uses pairwise IC correlation, spanning tests, and multiple testing correction.
#'
#' @param factor_panels named list of matrices (n_periods × n_stocks) per factor
#' @param ret_panel matrix n_periods × n_stocks of returns
#' @param alpha_fdr numeric, FDR level for multiple testing (Benjamini-Hochberg)
#' @return list with:
#'   $ic_table: IC stats per factor
#'   $ic_correlations: pairwise IC correlation matrix
#'   $spanning_results: GRS-type spanning test results
#'   $adjusted_p_values: BH-adjusted p-values
factor_zoo_comparison <- function(factor_panels, ret_panel, alpha_fdr = 0.05) {
  factor_names <- names(factor_panels)
  n_factors    <- length(factor_names)

  # IC series per factor
  ic_series_list <- lapply(factor_panels, function(fp) {
    compute_ic_series(fp, ret_panel)
  })
  names(ic_series_list) <- factor_names

  # IC statistics
  ic_table <- do.call(rbind, lapply(factor_names, function(nm) {
    stats <- compute_icir(ic_series_list[[nm]])
    data.frame(
      factor   = nm,
      mean_ic  = stats$mean_ic,
      sd_ic    = stats$sd_ic,
      icir     = stats$icir,
      t_stat   = stats$t_stat,
      p_value  = if (!is.na(stats$t_stat))
                   2 * pnorm(-abs(stats$t_stat)) else NA_real_,
      stringsAsFactors = FALSE
    )
  }))

  # Benjamini-Hochberg correction
  ic_table$p_adjusted <- p.adjust(ic_table$p_value, method = "BH")
  ic_table$significant <- ic_table$p_adjusted < alpha_fdr

  # Pairwise IC correlations (rank correlation of IC series)
  ic_corr <- matrix(1.0, n_factors, n_factors,
                    dimnames = list(factor_names, factor_names))
  for (i in seq_len(n_factors)) {
    for (j in seq_len(n_factors)) {
      if (i == j) next
      ic_i <- ic_series_list[[i]]
      ic_j <- ic_series_list[[j]]
      valid <- !is.na(ic_i) & !is.na(ic_j)
      if (sum(valid) >= 5L) {
        ic_corr[i, j] <- cor(ic_i[valid], ic_j[valid], method = "spearman")
      } else {
        ic_corr[i, j] <- NA_real_
      }
    }
  }

  # Spanning test: regress one factor's IC on others
  spanning <- do.call(rbind, lapply(factor_names, function(nm) {
    y <- ic_series_list[[nm]]
    other_ics <- do.call(cbind, ic_series_list[setdiff(factor_names, nm)])
    valid <- complete.cases(cbind(y, other_ics))
    if (sum(valid) < 10L) {
      return(data.frame(factor = nm, alpha = NA_real_, alpha_t = NA_real_,
                         r2 = NA_real_, stringsAsFactors = FALSE))
    }
    fit  <- lm(y[valid] ~ other_ics[valid, , drop = FALSE])
    summ <- summary(fit)
    data.frame(
      factor  = nm,
      alpha   = coef(fit)[1],
      alpha_t = coef(summ)[1, "t value"],
      r2      = summ$r.squared,
      stringsAsFactors = FALSE
    )
  }))

  list(
    ic_table         = ic_table,
    ic_correlations  = ic_corr,
    spanning_results = spanning,
    ic_series        = ic_series_list
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Visualisation of Factor Research
# ─────────────────────────────────────────────────────────────────────────────

#' plot_ic_series
#'
#' Plot IC time series with rolling average and zero line.
#'
#' @param ic_series numeric vector or xts
#' @param dates Date vector (required if ic_series is numeric)
#' @param rolling_window integer for rolling mean overlay
#' @param factor_name character label
#' @return ggplot2 object
plot_ic_series <- function(ic_series, dates = NULL, rolling_window = 12L,
                            factor_name = "Factor") {
  if (is.xts(ic_series)) {
    dates <- as.Date(index(ic_series))
    ic    <- as.numeric(ic_series)
  } else {
    ic <- as.numeric(ic_series)
    if (is.null(dates)) dates <- seq_along(ic)
  }

  roll_ic <- zoo::rollmean(ic, k = rolling_window, fill = NA, align = "right")

  df <- data.frame(date = dates, ic = ic, roll_ic = as.numeric(roll_ic),
                    stringsAsFactors = FALSE)

  stats <- compute_icir(ic)
  subtitle_txt <- sprintf(
    "Mean IC=%.4f | ICIR=%.2f | t-stat=%.2f",
    ifelse(is.na(stats$mean_ic), 0, stats$mean_ic),
    ifelse(is.na(stats$icir), 0, stats$icir),
    ifelse(is.na(stats$t_stat), 0, stats$t_stat)
  )

  ggplot(df, aes(x = date)) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.4) +
    geom_col(aes(y = ic),
             fill = ifelse(df$ic >= 0, "#2196F3", "#e63946"),
             alpha = 0.5, na.rm = TRUE) +
    geom_line(aes(y = roll_ic), color = "#ff6b35", linewidth = 1.0, na.rm = TRUE) +
    labs(
      title    = paste0("IC Series: ", factor_name),
      subtitle = subtitle_txt,
      x        = "Date",
      y        = "Information Coefficient"
    ) +
    theme_minimal(base_size = 10) +
    theme(panel.grid.minor = element_blank())
}


#' plot_factor_decay
#'
#' Plot IC vs horizon decay curve.
#'
#' @param decay_df data.frame from factor_decay_analysis
#' @param factor_name character label
#' @return ggplot2 object
plot_factor_decay <- function(decay_df, factor_name = "Factor") {
  ggplot(decay_df, aes(x = horizon, y = mean_ic)) +
    geom_ribbon(aes(ymin = mean_ic - 1.96 * se,
                    ymax = mean_ic + 1.96 * se),
                fill = "#2196F3", alpha = 0.2, na.rm = TRUE) +
    geom_line(color = "#2196F3", linewidth = 1.0) +
    geom_point(color = "#1a1a2e", size = 2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    labs(
      title = paste0("Signal Decay: ", factor_name),
      x     = "Horizon (periods)",
      y     = "Mean IC"
    ) +
    scale_x_continuous(breaks = decay_df$horizon) +
    theme_minimal(base_size = 10)
}


#' plot_factor_zoo_heatmap
#'
#' Heatmap of pairwise IC correlations between factors.
#'
#' @param ic_corr matrix from factor_zoo_comparison$ic_correlations
#' @return ggplot2 object
plot_factor_zoo_heatmap <- function(ic_corr) {
  n <- nrow(ic_corr)
  factor_names <- rownames(ic_corr)

  df_melt <- data.frame(
    factor1 = rep(factor_names, each = n),
    factor2 = rep(factor_names, times = n),
    corr    = as.vector(ic_corr),
    stringsAsFactors = FALSE
  )

  ggplot(df_melt, aes(x = factor1, y = factor2, fill = corr)) +
    geom_tile(color = "white") +
    geom_text(aes(label = round(corr, 2)), size = 3, na.rm = TRUE) +
    scale_fill_gradient2(
      low    = "#e63946",
      mid    = "white",
      high   = "#2196F3",
      midpoint = 0,
      limits = c(-1, 1),
      name   = "IC Corr"
    ) +
    labs(title = "Factor IC Correlation Matrix", x = NULL, y = NULL) +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}
