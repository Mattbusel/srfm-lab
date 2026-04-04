# visualization.R
# Research-grade ggplot2 visualization suite for SRFM.
# Implements: equity curves + regime bands, rolling metrics,
#             correlation heatmaps, return distributions,
#             drawdown analysis, factor exposure charts.
# Dependencies: ggplot2, xts, zoo, scales, tidyr (optional)

library(ggplot2)
library(xts)
library(zoo)
library(scales)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Theme
# ─────────────────────────────────────────────────────────────────────────────

#' srfm_theme
#'
#' Clean, publication-ready ggplot2 theme for SRFM research.
srfm_theme <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
  theme(
    panel.grid.minor    = element_blank(),
    panel.border        = element_rect(colour = "#cccccc", fill = NA, linewidth = 0.3),
    axis.title          = element_text(size = base_size - 1, colour = "#333333"),
    axis.text           = element_text(size = base_size - 2, colour = "#555555"),
    plot.title          = element_text(size = base_size + 1, face = "bold", colour = "#1a1a2e"),
    plot.subtitle       = element_text(size = base_size - 1, colour = "#666666"),
    plot.caption        = element_text(size = base_size - 3, colour = "#999999"),
    legend.text         = element_text(size = base_size - 2),
    legend.title        = element_text(size = base_size - 1, face = "bold"),
    strip.text          = element_text(size = base_size - 1, face = "bold"),
    plot.margin         = margin(8, 8, 8, 8)
  )
}

# Consistent palette
SRFM_COLORS <- c(
  blue    = "#2196F3",
  red     = "#e63946",
  green   = "#4caf50",
  orange  = "#ff9800",
  purple  = "#9c27b0",
  teal    = "#009688",
  pink    = "#e91e63",
  indigo  = "#3f51b5",
  grey    = "#607d8b",
  amber   = "#ffc107"
)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Equity Curves with Regime Bands
# ─────────────────────────────────────────────────────────────────────────────

#' plot_equity_curve
#'
#' Plot cumulative equity curve(s) with optional regime shading.
#'
#' @param returns_list named list of xts return series (or single xts)
#' @param regimes xts or numeric vector of regime labels (integer or factor)
#' @param regime_colors named character vector mapping regime labels to colors
#' @param log_scale logical, use log scale for y-axis
#' @param start_value numeric, initial portfolio value (default 1.0)
#' @param title character
#' @param benchmark xts, benchmark return series for comparison
#' @return ggplot2 object
plot_equity_curve <- function(
  returns_list,
  regimes       = NULL,
  regime_colors = NULL,
  log_scale     = FALSE,
  start_value   = 1.0,
  title         = "Equity Curve",
  benchmark     = NULL
) {
  # Standardize to named list
  if (is.xts(returns_list)) {
    returns_list <- list(Strategy = returns_list)
  }

  # Build equity curve data
  all_df <- do.call(rbind, lapply(names(returns_list), function(nm) {
    r    <- as.numeric(returns_list[[nm]])
    r    <- replace(r, is.na(r), 0)
    cum  <- start_value * cumprod(1 + r)
    dates <- if (is.xts(returns_list[[nm]])) as.Date(index(returns_list[[nm]])) else
              seq_along(r)
    data.frame(date = dates, equity = cum, strategy = nm,
               stringsAsFactors = FALSE)
  }))

  # Regime rectangles
  regime_df <- NULL
  if (!is.null(regimes)) {
    r_vec  <- as.integer(as.numeric(regimes))
    r_vec  <- replace(r_vec, is.na(r_vec), 0L)
    dates  <- if (is.xts(regimes)) as.Date(index(regimes)) else seq_along(r_vec)
    n_obs  <- length(r_vec)
    regime_df <- .build_regime_rect_df(dates, r_vec)
  }

  # Default regime colors
  if (!is.null(regime_df) && is.null(regime_colors)) {
    unique_regs <- unique(regime_df$regime)
    palette     <- c("#e8f4f8", "#ffe4e1", "#e8f5e9", "#fff8e1", "#f3e5f5")
    regime_colors <- setNames(
      palette[seq_along(unique_regs)],
      as.character(unique_regs)
    )
  }

  # Colour palette for strategies
  strat_names <- unique(all_df$strategy)
  strat_colors <- setNames(
    unname(SRFM_COLORS[seq_along(strat_names)]),
    strat_names
  )

  p <- ggplot()

  # Add regime shading
  if (!is.null(regime_df)) {
    p <- p + geom_rect(
      data = regime_df,
      aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf,
          fill = as.character(regime)),
      alpha = 0.3,
      inherit.aes = FALSE
    ) +
    scale_fill_manual(values = regime_colors, name = "Regime", guide = "legend")
  }

  p <- p +
    geom_line(
      data = all_df,
      aes(x = date, y = equity, color = strategy),
      linewidth = 0.9
    ) +
    scale_color_manual(values = strat_colors, name = "Strategy")

  # Benchmark
  if (!is.null(benchmark)) {
    bm_r    <- as.numeric(benchmark)
    bm_r    <- replace(bm_r, is.na(bm_r), 0)
    bm_cum  <- start_value * cumprod(1 + bm_r)
    bm_dates <- if (is.xts(benchmark)) as.Date(index(benchmark)) else seq_along(bm_r)
    bm_df    <- data.frame(date = bm_dates, equity = bm_cum)
    p <- p + geom_line(data = bm_df, aes(x = date, y = equity),
                        color = "#607d8b", linetype = "dashed", linewidth = 0.6)
  }

  p <- p +
    scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
    labs(title = title, x = "Date", y = "Cumulative Return") +
    srfm_theme()

  if (log_scale) p <- p + scale_y_log10(labels = scales::comma_format())

  p
}


#' .build_regime_rect_df
#'
#' Build a data.frame of (xmin, xmax, regime) for geom_rect.
.build_regime_rect_df <- function(dates, regimes) {
  n  <- length(dates)
  if (n == 0) return(data.frame())
  starts <- c(1L, which(diff(regimes) != 0) + 1L)
  ends   <- c(starts[-1] - 1L, n)

  data.frame(
    xmin   = as.Date(dates[starts]),
    xmax   = as.Date(dates[ends]),
    regime = regimes[starts],
    stringsAsFactors = FALSE
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Rolling Metrics Chart
# ─────────────────────────────────────────────────────────────────────────────

#' plot_rolling_metrics
#'
#' Multi-panel chart of rolling performance metrics.
#'
#' @param returns xts or numeric vector of portfolio returns
#' @param window integer, rolling window (bars)
#' @param metrics character vector: "sharpe", "vol", "return", "drawdown", "hit_rate"
#' @param annual_periods integer (252 for daily)
#' @return ggplot2 object
plot_rolling_metrics <- function(
  returns,
  window         = 63L,
  metrics        = c("return", "vol", "sharpe", "drawdown"),
  annual_periods = 252L,
  title          = "Rolling Performance Metrics"
) {
  if (is.xts(returns)) {
    dates <- as.Date(index(returns))
    r     <- as.numeric(returns)
  } else {
    dates <- seq_along(returns)
    r     <- as.numeric(returns)
  }
  n <- length(r)

  .roll <- function(fun, window) {
    out <- rep(NA_real_, n)
    for (i in window:n) {
      out[i] <- fun(r[(i - window + 1):i])
    }
    out
  }

  metric_data <- list()

  if ("return" %in% metrics) {
    metric_data[["Ann. Return"]] <- .roll(function(x) mean(x) * annual_periods, window)
  }
  if ("vol" %in% metrics) {
    metric_data[["Ann. Vol"]] <- .roll(function(x) sd(x) * sqrt(annual_periods), window)
  }
  if ("sharpe" %in% metrics) {
    metric_data[["Sharpe"]] <- .roll(function(x) {
      mu  <- mean(x) * annual_periods
      sig <- sd(x) * sqrt(annual_periods)
      if (sig > 0) mu / sig else NA_real_
    }, window)
  }
  if ("drawdown" %in% metrics) {
    cum <- cumprod(1 + replace(r, is.na(r), 0))
    rm_max <- cummax(cum)
    dd <- (cum - rm_max) / (rm_max + 1e-10)
    metric_data[["Drawdown"]] <- dd
  }
  if ("hit_rate" %in% metrics) {
    metric_data[["Hit Rate"]] <- .roll(function(x) mean(x > 0), window)
  }

  # Build long data.frame
  df_long <- do.call(rbind, lapply(names(metric_data), function(nm) {
    data.frame(
      date   = as.Date(dates),
      value  = metric_data[[nm]],
      metric = nm,
      stringsAsFactors = FALSE
    )
  }))
  df_long <- df_long[!is.na(df_long$value), ]

  n_metrics <- length(unique(df_long$metric))
  metric_colors <- setNames(unname(SRFM_COLORS[seq_len(n_metrics)]),
                             unique(df_long$metric))

  ggplot(df_long, aes(x = date, y = value)) +
    geom_line(aes(color = metric), linewidth = 0.8) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.3) +
    facet_wrap(~ metric, scales = "free_y", ncol = 1) +
    scale_color_manual(values = metric_colors, guide = "none") +
    scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
    labs(
      title    = title,
      subtitle = sprintf("Rolling %d-bar window", window),
      x        = "Date",
      y        = NULL
    ) +
    srfm_theme()
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

#' plot_correlation_heatmap
#'
#' Correlation matrix heatmap with hierarchical clustering order.
#'
#' @param returns matrix n_obs × n_assets or xts
#' @param labels character vector of asset names
#' @param method character: "pearson" or "spearman"
#' @param cluster logical, whether to reorder by hierarchical clustering
#' @param show_values logical, display correlation values in cells
#' @return ggplot2 object
plot_correlation_heatmap <- function(
  returns,
  labels      = NULL,
  method      = "pearson",
  cluster     = TRUE,
  show_values = TRUE,
  title       = "Correlation Matrix"
) {
  R <- as.matrix(returns)
  n <- ncol(R)

  if (is.null(labels)) {
    labels <- if (!is.null(colnames(R))) colnames(R) else paste0("A", seq_len(n))
  }
  colnames(R) <- labels

  corr <- cor(R, method = method, use = "pairwise.complete.obs")
  corr <- (corr + t(corr)) / 2

  if (cluster) {
    dist_mat <- as.dist(1 - corr)
    hc       <- hclust(dist_mat, method = "ward.D2")
    ord      <- hc$order
    labels   <- labels[ord]
    corr     <- corr[ord, ord]
  }

  # Melt correlation matrix
  df_melt <- data.frame(
    x    = rep(labels, each = n),
    y    = rep(labels, times = n),
    corr = as.vector(corr),
    stringsAsFactors = FALSE
  )
  df_melt$x <- factor(df_melt$x, levels = labels)
  df_melt$y <- factor(df_melt$y, levels = rev(labels))

  p <- ggplot(df_melt, aes(x = x, y = y, fill = corr)) +
    geom_tile(color = "white", linewidth = 0.3) +
    scale_fill_gradient2(
      low      = SRFM_COLORS["red"],
      mid      = "white",
      high     = SRFM_COLORS["blue"],
      midpoint = 0,
      limits   = c(-1, 1),
      name     = "Correlation"
    )

  if (show_values && n <= 20) {
    p <- p + geom_text(
      aes(label = round(corr, 2)),
      size  = if (n <= 10) 3.5 else 2.5,
      color = ifelse(abs(df_melt$corr) > 0.7, "white", "black")
    )
  }

  p <- p +
    labs(title = title, x = NULL, y = NULL) +
    srfm_theme() +
    theme(
      axis.text.x  = element_text(angle = 45, hjust = 1, size = 8),
      axis.text.y  = element_text(size = 8),
      legend.position = "right"
    )

  p
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Return Distribution Charts
# ─────────────────────────────────────────────────────────────────────────────

#' plot_return_distribution
#'
#' Return distribution chart with normal overlay, VaR/CVaR markers,
#' and statistical summary.
#'
#' @param returns_list named list of return series (xts or numeric)
#' @param annual_periods integer
#' @param show_normal logical
#' @param alpha numeric, VaR level
#' @return ggplot2 object
plot_return_distribution <- function(
  returns_list,
  annual_periods = 252L,
  show_normal    = TRUE,
  alpha          = 0.05,
  title          = "Return Distribution"
) {
  if (is.xts(returns_list) || is.numeric(returns_list)) {
    returns_list <- list(Strategy = returns_list)
  }

  all_df <- do.call(rbind, lapply(names(returns_list), function(nm) {
    r <- as.numeric(returns_list[[nm]])
    r <- r[!is.na(r)]
    data.frame(return = r, strategy = nm, stringsAsFactors = FALSE)
  }))

  n_strats <- length(unique(all_df$strategy))
  strat_colors <- setNames(unname(SRFM_COLORS[seq_len(n_strats)]),
                             unique(all_df$strategy))

  p <- ggplot(all_df, aes(x = return, fill = strategy, color = strategy)) +
    geom_histogram(aes(y = after_stat(density)), bins = 60, alpha = 0.4,
                   position = "identity") +
    geom_density(alpha = 0, linewidth = 1.0)

  if (show_normal) {
    # Add normal fit per strategy
    for (nm in names(returns_list)) {
      r_nm <- as.numeric(returns_list[[nm]])
      r_nm <- r_nm[!is.na(r_nm)]
      mu_n  <- mean(r_nm)
      sd_n  <- sd(r_nm)
      x_seq <- seq(min(r_nm), max(r_nm), length.out = 300)
      norm_df <- data.frame(
        return   = x_seq,
        density  = dnorm(x_seq, mean = mu_n, sd = sd_n),
        strategy = nm
      )
      p <- p + geom_line(data = norm_df,
                          aes(x = return, y = density, color = strategy),
                          linetype = "dashed", linewidth = 0.8, inherit.aes = FALSE)
    }
  }

  # VaR markers
  var_df <- do.call(rbind, lapply(names(returns_list), function(nm) {
    r <- as.numeric(returns_list[[nm]])
    r <- r[!is.na(r)]
    data.frame(var    = quantile(r, alpha),
               cvar   = mean(r[r <= quantile(r, alpha)]),
               strategy = nm, stringsAsFactors = FALSE)
  }))

  p <- p +
    geom_vline(data = var_df, aes(xintercept = var, color = strategy),
               linetype = "solid", linewidth = 0.7, alpha = 0.8) +
    geom_vline(data = var_df, aes(xintercept = cvar, color = strategy),
               linetype = "dotted", linewidth = 0.7, alpha = 0.8)

  # Stats annotation
  stats_df <- do.call(rbind, lapply(names(returns_list), function(nm) {
    r <- as.numeric(returns_list[[nm]])
    r <- r[!is.na(r)]
    ann_sharpe <- (mean(r) * annual_periods) / (sd(r) * sqrt(annual_periods))
    data.frame(
      strategy = nm,
      label    = sprintf("%s | Sharpe=%.2f | Skew=%.2f | Kurt=%.2f",
                         nm, ann_sharpe,
                         sum((r - mean(r))^3) / (length(r) * sd(r)^3),
                         sum((r - mean(r))^4) / (length(r) * sd(r)^4) - 3),
      stringsAsFactors = FALSE
    )
  }))

  p <- p +
    scale_fill_manual(values = strat_colors, guide = "none") +
    scale_color_manual(values = strat_colors, name = "Strategy") +
    scale_x_continuous(labels = scales::percent_format(accuracy = 0.01)) +
    labs(
      title    = title,
      subtitle = paste0("Vertical lines: VaR (solid) and CVaR (dotted) at ",
                         scales::percent(alpha)),
      x        = "Return",
      y        = "Density"
    ) +
    srfm_theme()

  p
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Drawdown Chart
# ─────────────────────────────────────────────────────────────────────────────

#' plot_drawdown_analysis
#'
#' Drawdown chart showing underwater equity curve for multiple strategies.
#'
#' @param returns_list named list of return series
#' @param title character
#' @param top_drawdowns integer, annotate N worst drawdowns
#' @return ggplot2 object
plot_drawdown_analysis <- function(
  returns_list,
  title          = "Drawdown Analysis",
  top_drawdowns  = 5L
) {
  if (is.xts(returns_list) || is.numeric(returns_list)) {
    returns_list <- list(Strategy = returns_list)
  }

  all_df <- do.call(rbind, lapply(names(returns_list), function(nm) {
    r <- as.numeric(returns_list[[nm]])
    r <- replace(r, is.na(r), 0)
    dates <- if (is.xts(returns_list[[nm]])) as.Date(index(returns_list[[nm]])) else
              seq_along(r)
    cum <- cumprod(1 + r)
    rm_max <- cummax(cum)
    dd <- (cum - rm_max) / (rm_max + 1e-10)
    data.frame(date = as.Date(dates), drawdown = dd, strategy = nm,
               stringsAsFactors = FALSE)
  }))

  n_strats <- length(unique(all_df$strategy))
  strat_colors <- setNames(unname(SRFM_COLORS[seq_len(n_strats)]),
                             unique(all_df$strategy))

  p <- ggplot(all_df, aes(x = date, y = drawdown, fill = strategy, color = strategy)) +
    geom_area(alpha = 0.3, position = "identity") +
    geom_line(linewidth = 0.7) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.4) +
    scale_fill_manual(values = strat_colors, guide = "none") +
    scale_color_manual(values = strat_colors, name = "Strategy") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
    labs(title = title, x = "Date", y = "Drawdown") +
    srfm_theme()

  # Annotate worst drawdowns for first strategy
  nm1    <- names(returns_list)[1]
  dd1_df <- all_df[all_df$strategy == nm1, ]
  if (nrow(dd1_df) > 0 && top_drawdowns > 0) {
    worst_idx <- order(dd1_df$drawdown)[seq_len(min(top_drawdowns, nrow(dd1_df)))]
    ann_df    <- dd1_df[worst_idx, ]
    ann_df$label <- scales::percent(ann_df$drawdown, accuracy = 0.1)
    p <- p + ggrepel::geom_text_repel(
      data    = ann_df,
      aes(x = date, y = drawdown, label = label),
      size    = 2.8,
      color   = "#333333",
      segment.color = "#aaaaaa",
      inherit.aes = FALSE,
      na.rm = TRUE
    )
  }

  p
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Factor Exposure Chart
# ─────────────────────────────────────────────────────────────────────────────

#' plot_factor_exposures
#'
#' Bar chart of factor exposures (betas) for a strategy or portfolio.
#'
#' @param exposures named numeric vector or data.frame of factor betas
#' @param error_bars optional, named numeric vector of SEs
#' @param title character
#' @return ggplot2 object
plot_factor_exposures <- function(
  exposures,
  error_bars = NULL,
  title      = "Factor Exposures",
  confidence = 0.95
) {
  if (is.numeric(exposures)) {
    df <- data.frame(
      factor   = names(exposures),
      exposure = as.numeric(exposures),
      stringsAsFactors = FALSE
    )
  } else {
    df <- as.data.frame(exposures)
  }

  if (!is.null(error_bars)) {
    df$se <- as.numeric(error_bars[df$factor])
    z     <- qnorm(1 - (1 - confidence)/2)
    df$lo <- df$exposure - z * df$se
    df$hi <- df$exposure + z * df$se
  } else {
    df$lo <- NA_real_
    df$hi <- NA_real_
  }

  df$positive <- df$exposure >= 0
  df$factor   <- factor(df$factor, levels = df$factor[order(df$exposure)])

  p <- ggplot(df, aes(x = factor, y = exposure, fill = positive)) +
    geom_col(width = 0.7, alpha = 0.85) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.4)

  if (!all(is.na(df$lo))) {
    p <- p + geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.25,
                            color = "#333333", linewidth = 0.6)
  }

  p <- p +
    scale_fill_manual(values = c(`FALSE` = SRFM_COLORS["red"],
                                  `TRUE`  = SRFM_COLORS["blue"]),
                      guide  = "none") +
    coord_flip() +
    labs(title = title, x = "Factor", y = "Exposure (Beta)") +
    srfm_theme()

  p
}


#' plot_rolling_factor_exposures
#'
#' Time-series of rolling factor exposures.
#'
#' @param rolling_betas data.frame with columns: date, factor, exposure
#'   (as returned by rolling regression)
#' @param factors character vector of factors to plot
#' @param title character
#' @return ggplot2 object
plot_rolling_factor_exposures <- function(
  rolling_betas,
  factors = NULL,
  title   = "Rolling Factor Exposures"
) {
  df <- rolling_betas
  if (!is.null(factors)) df <- df[df$factor %in% factors, ]
  df$date <- as.Date(df$date)

  n_factors <- length(unique(df$factor))
  fact_colors <- setNames(unname(SRFM_COLORS[seq_len(n_factors)]),
                            unique(df$factor))

  ggplot(df, aes(x = date, y = exposure, color = factor)) +
    geom_line(linewidth = 0.8) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.3) +
    facet_wrap(~ factor, scales = "free_y", ncol = 2) +
    scale_color_manual(values = fact_colors, guide = "none") +
    scale_x_date(date_labels = "%Y-%m") +
    labs(title = title, x = "Date", y = "Factor Exposure") +
    srfm_theme()
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Additional Chart Types
# ─────────────────────────────────────────────────────────────────────────────

#' plot_monthly_returns_heatmap
#'
#' Calendar heatmap of monthly returns (year × month grid).
#'
#' @param returns xts daily return series
#' @param title character
#' @return ggplot2 object
plot_monthly_returns_heatmap <- function(returns, title = "Monthly Returns Heatmap") {
  if (!is.xts(returns)) {
    stop("returns must be an xts object")
  }

  # Aggregate to monthly
  monthly <- xts::apply.monthly(returns, function(x) prod(1 + as.numeric(x)) - 1)
  dates   <- as.Date(index(monthly))
  df <- data.frame(
    year  = as.integer(format(dates, "%Y")),
    month = as.integer(format(dates, "%m")),
    ret   = as.numeric(monthly),
    stringsAsFactors = FALSE
  )
  df$month_label <- month.abb[df$month]
  df$month_label <- factor(df$month_label, levels = month.abb)

  max_abs <- max(abs(df$ret), na.rm = TRUE)

  ggplot(df, aes(x = month_label, y = factor(year, levels = rev(sort(unique(year)))),
                  fill = ret)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = scales::percent(ret, accuracy = 0.1)),
              size = 2.8, color = ifelse(abs(df$ret) > max_abs * 0.6, "white", "black"),
              na.rm = TRUE) +
    scale_fill_gradient2(
      low      = SRFM_COLORS["red"],
      mid      = "white",
      high     = SRFM_COLORS["green"],
      midpoint = 0,
      limits   = c(-max_abs, max_abs),
      labels   = scales::percent_format(accuracy = 1),
      name     = "Return"
    ) +
    labs(title = title, x = "Month", y = "Year") +
    srfm_theme() +
    theme(axis.text.x = element_text(size = 9))
}


#' plot_scatter_returns
#'
#' Scatter plot of strategy vs benchmark returns with regression line.
#'
#' @param strategy_returns numeric or xts
#' @param benchmark_returns numeric or xts
#' @param strategy_name character
#' @param benchmark_name character
#' @return ggplot2 object
plot_scatter_returns <- function(
  strategy_returns,
  benchmark_returns,
  strategy_name  = "Strategy",
  benchmark_name = "Benchmark"
) {
  r_s <- as.numeric(strategy_returns)
  r_b <- as.numeric(benchmark_returns)
  n   <- min(length(r_s), length(r_b))
  r_s <- r_s[seq_len(n)]
  r_b <- r_b[seq_len(n)]

  valid <- !is.na(r_s) & !is.na(r_b)
  df    <- data.frame(benchmark = r_b[valid], strategy = r_s[valid])

  fit   <- lm(strategy ~ benchmark, data = df)
  alpha <- coef(fit)[1]
  beta  <- coef(fit)[2]
  r2    <- summary(fit)$r.squared

  label_txt <- sprintf("α=%.4f | β=%.2f | R²=%.2f", alpha, beta, r2)

  ggplot(df, aes(x = benchmark, y = strategy)) +
    geom_point(color = SRFM_COLORS["blue"], alpha = 0.4, size = 1.2) +
    geom_smooth(method = "lm", color = SRFM_COLORS["red"],
                fill = "#ffcdd2", linewidth = 1.0, se = TRUE) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.4) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.4) +
    annotate("text",
             x = min(df$benchmark) * 0.9,
             y = max(df$strategy) * 0.9,
             label = label_txt, hjust = 0, size = 3.2, color = "#333333") +
    scale_x_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    labs(
      title = sprintf("%s vs %s Returns", strategy_name, benchmark_name),
      x     = paste(benchmark_name, "Return"),
      y     = paste(strategy_name, "Return")
    ) +
    srfm_theme()
}


#' plot_mc_fan
#'
#' Monte Carlo equity curve fan chart.
#'
#' @param returns xts or numeric, historical returns for simulation
#' @param n_paths integer, number of MC paths
#' @param n_periods integer, number of forward periods
#' @param quantiles numeric vector of quantile levels for fan
#' @param seed integer
#' @return ggplot2 object
plot_mc_fan <- function(
  returns,
  n_paths   = 1000L,
  n_periods = 252L,
  quantiles = c(0.05, 0.25, 0.50, 0.75, 0.95),
  seed      = 42L,
  title     = "Monte Carlo Equity Fan"
) {
  set.seed(seed)
  r <- as.numeric(returns)
  r <- r[!is.na(r)]

  mu  <- mean(r)
  sig <- sd(r)

  # Block bootstrap simulation
  mc_paths <- matrix(0.0, n_periods + 1L, n_paths)
  mc_paths[1L, ] <- 1.0

  for (path in seq_len(n_paths)) {
    r_sim <- sample(r, n_periods, replace = TRUE)  # simple resample
    mc_paths[2:(n_periods + 1L), path] <- cumprod(1 + r_sim)
  }

  # Quantile ribbons
  q_mat <- t(apply(mc_paths, 1, quantile, probs = quantiles, na.rm = TRUE))
  df_q  <- as.data.frame(q_mat)
  colnames(df_q) <- paste0("q", gsub("\\.", "_", as.character(quantiles * 100)))
  df_q$period <- 0:n_periods

  # Median path
  df_q$median <- q_mat[, which(abs(quantiles - 0.5) < 0.01)[1]]

  p <- ggplot(df_q, aes(x = period))

  # Add quantile ribbons
  ribbon_levels <- list(
    c("q5", "q95", 0.10),
    c("q25", "q75", 0.20)
  )
  fill_clrs <- c(SRFM_COLORS["blue"], SRFM_COLORS["teal"])
  for (i in seq_along(ribbon_levels)) {
    lo_nm <- ribbon_levels[[i]][1]
    hi_nm <- ribbon_levels[[i]][2]
    alpha_val <- as.numeric(ribbon_levels[[i]][3])
    if (lo_nm %in% names(df_q) && hi_nm %in% names(df_q)) {
      rib_df <- data.frame(period = df_q$period,
                            lo = df_q[[lo_nm]], hi = df_q[[hi_nm]])
      p <- p + geom_ribbon(data = rib_df,
                            aes(x = period, ymin = lo, ymax = hi),
                            fill = fill_clrs[i], alpha = alpha_val,
                            inherit.aes = FALSE)
    }
  }

  # Median line
  p <- p +
    geom_line(aes(y = median), color = SRFM_COLORS["blue"], linewidth = 1.2) +
    geom_hline(yintercept = 1.0, linetype = "dashed", color = "grey50",
               linewidth = 0.5) +
    scale_y_continuous(labels = scales::comma_format(prefix = "x")) +
    labs(
      title    = title,
      subtitle = sprintf("%d paths, %d periods | μ=%.2f%% σ=%.2f%% (annualised)",
                         n_paths, n_periods, mu * 252 * 100, sig * sqrt(252) * 100),
      x        = "Period",
      y        = "Cumulative Return Multiple"
    ) +
    srfm_theme()

  p
}


#' plot_turnover_analysis
#'
#' Plot portfolio turnover and its relationship to transaction costs.
#'
#' @param bt BacktestResult from run_backtest
#' @param rolling_window integer
#' @return ggplot2 object
plot_turnover_analysis <- function(bt, rolling_window = 21L) {
  dates <- as.Date(index(bt$returns))
  to    <- as.numeric(bt$turnover)
  costs <- as.numeric(bt$tc_costs)

  roll_to    <- zoo::rollmean(to,    k = rolling_window, fill = NA, align = "right")
  roll_costs <- zoo::rollmean(costs, k = rolling_window, fill = NA, align = "right")

  df <- data.frame(
    date    = dates,
    turnover = as.numeric(roll_to),
    tc_cost  = as.numeric(roll_costs),
    stringsAsFactors = FALSE
  )
  df <- df[!is.na(df$turnover), ]

  p_to <- ggplot(df, aes(x = date, y = turnover)) +
    geom_line(color = SRFM_COLORS["blue"], linewidth = 0.8) +
    geom_hline(yintercept = mean(df$turnover, na.rm = TRUE),
               linetype = "dashed", color = SRFM_COLORS["red"], linewidth = 0.5) +
    scale_x_date(date_labels = "%Y-%m") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    labs(title = "Rolling Portfolio Turnover",
         x = "Date", y = "Turnover (one-way)") +
    srfm_theme()

  p_to
}
