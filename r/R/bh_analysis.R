# bh_analysis.R
# Black-Hole (BH) physics state reconstruction from price series.
# Implements: BH mass series, formation detection, regime classification.
# Dependencies: base R, xts, ggplot2, zoo
#
# The BH metaphor maps market microstructure to gravitational dynamics:
#   - "Mass" M(t) = function of price momentum + volatility (analogous to accretion)
#   - "Event horizon" = threshold beyond which trend is self-reinforcing
#   - "Formation" = regime where mass accelerates past threshold
#   - "Evaporation" = Hawking-like dissipation of accumulated momentum

library(xts)
library(zoo)
library(ggplot2)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Mass Series Construction
# ─────────────────────────────────────────────────────────────────────────────

#' compute_bh_mass
#'
#' Compute the Black-Hole mass series from a price series.
#' Mass(t) = exponentially weighted momentum * volatility-adjusted return
#'
#' @param prices xts or numeric vector of prices
#' @param lookback integer, momentum lookback window
#' @param vol_window integer, volatility estimation window
#' @param alpha_decay numeric, EWM decay factor (0 < alpha < 1)
#' @param threshold numeric, event-horizon threshold for mass
#' @return xts object with columns: price, log_return, mass, vol, momentum, regime
compute_bh_mass <- function(
  prices,
  lookback    = 20L,
  vol_window  = 20L,
  alpha_decay = 0.94,
  threshold   = 1.5
) {
  stopifnot(is.numeric(prices) || is.xts(prices))
  if (!is.xts(prices)) {
    prices <- xts(prices, order.by = seq.Date(Sys.Date() - length(prices) + 1,
                                               Sys.Date(), by = "day"))
  }
  idx <- index(prices)
  p   <- as.numeric(prices)
  n   <- length(p)

  # Log returns
  log_ret <- c(NA_real_, diff(log(p)))

  # Rolling volatility (annualised, sqrt(252))
  vol <- rep(NA_real_, n)
  for (i in (vol_window):n) {
    vol[i] <- sd(log_ret[(i - vol_window + 1):i], na.rm = TRUE) * sqrt(252)
  }

  # Momentum: sum of log returns over lookback
  momentum <- rep(NA_real_, n)
  for (i in (lookback):n) {
    momentum[i] <- sum(log_ret[(i - lookback + 1):i], na.rm = TRUE)
  }

  # EWM accretion: recursive M_t = alpha_decay * M_{t-1} + (1-alpha_decay) * |momentum| / vol
  mass <- rep(0.0, n)
  for (i in seq_along(mass)) {
    if (!is.na(momentum[i]) && !is.na(vol[i]) && vol[i] > 1e-10) {
      accreted <- abs(momentum[i]) / vol[i]
      mass[i] <- alpha_decay * mass[max(1L, i - 1L)] + (1 - alpha_decay) * accreted
    } else if (i > 1) {
      mass[i] <- alpha_decay * mass[i - 1L]
    }
  }

  # Signed mass: carries sign of momentum
  signed_mass <- mass * sign(replace(momentum, is.na(momentum), 0))

  # Regime classification
  regime <- classify_bh_regime(signed_mass, threshold = threshold)

  out <- xts(
    data.frame(
      price       = p,
      log_return  = log_ret,
      mass        = signed_mass,
      abs_mass    = mass,
      vol         = vol,
      momentum    = momentum,
      regime      = as.numeric(regime)
    ),
    order.by = idx
  )
  attr(out, "threshold") <- threshold
  attr(out, "lookback")  <- lookback
  return(out)
}


#' compute_mass_derivative
#'
#' Compute rate of change (accretion rate) of the BH mass series.
#' d(mass)/dt > 0 => absorbing energy (trending)
#' d(mass)/dt < 0 => evaporating (reverting)
#'
#' @param mass_series xts with 'mass' column
#' @param smooth_window integer, smoothing window for noise reduction
#' @return xts with columns: mass, mass_derivative, accretion_rate
compute_mass_derivative <- function(mass_series, smooth_window = 5L) {
  m <- coredata(mass_series[, "mass"])
  n <- length(m)

  # Central difference
  deriv <- rep(NA_real_, n)
  for (i in 2:(n-1)) {
    deriv[i] <- (m[i+1] - m[i-1]) / 2
  }
  deriv[1] <- m[2] - m[1]
  deriv[n] <- m[n] - m[n-1]

  # Smooth the derivative
  if (smooth_window > 1) {
    kern   <- rep(1/smooth_window, smooth_window)
    deriv_smooth <- stats::filter(deriv, kern, sides = 2)
    deriv_smooth <- as.numeric(deriv_smooth)
    # Fill NAs at edges
    deriv_smooth[is.na(deriv_smooth)] <- deriv[is.na(deriv_smooth)]
  } else {
    deriv_smooth <- deriv
  }

  # Accretion rate (positive = gaining mass)
  accretion <- pmax(deriv_smooth, 0)

  xts(
    data.frame(
      mass            = as.numeric(m),
      mass_derivative = deriv_smooth,
      accretion_rate  = accretion,
      evaporation_rate = pmax(-deriv_smooth, 0)
    ),
    order.by = index(mass_series)
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Formation Detection
# ─────────────────────────────────────────────────────────────────────────────

#' detect_bh_formations
#'
#' Identify BH formation events: periods where mass crosses event-horizon
#' and sustains above threshold for min_duration bars.
#'
#' @param bh_data xts output of compute_bh_mass
#' @param min_duration integer, minimum bars above threshold
#' @param min_mass_peak numeric, minimum peak mass in a formation
#' @return data.frame with: start, end, peak_mass, direction, duration, n_formations
detect_bh_formations <- function(bh_data, min_duration = 5L, min_mass_peak = 1.0) {
  threshold <- attr(bh_data, "threshold")
  if (is.null(threshold)) threshold <- 1.5

  mass <- as.numeric(bh_data[, "abs_mass"])
  dates <- index(bh_data)
  n <- length(mass)

  # States: 0 = no formation, 1 = active formation
  above <- mass > threshold & !is.na(mass)

  formations <- list()
  in_formation <- FALSE
  form_start <- NA

  for (i in seq_len(n)) {
    if (!in_formation && above[i]) {
      in_formation <- TRUE
      form_start <- i
    } else if (in_formation && !above[i]) {
      # Formation ended
      form_end <- i - 1L
      duration <- form_end - form_start + 1L
      if (duration >= min_duration) {
        peak_mass <- max(mass[form_start:form_end], na.rm = TRUE)
        if (peak_mass >= min_mass_peak) {
          # Determine direction from signed mass
          signed_mass_slice <- as.numeric(bh_data[form_start:form_end, "mass"])
          direction <- ifelse(mean(signed_mass_slice) > 0, "long", "short")

          formations[[length(formations) + 1]] <- list(
            start     = dates[form_start],
            end       = dates[form_end],
            peak_mass = peak_mass,
            direction = direction,
            duration  = duration,
            start_idx = form_start,
            end_idx   = form_end
          )
        }
      }
      in_formation <- FALSE
      form_start <- NA
    }
  }
  # Handle formation still open at end
  if (in_formation && !is.na(form_start)) {
    form_end <- n
    duration <- form_end - form_start + 1L
    if (duration >= min_duration) {
      peak_mass <- max(mass[form_start:form_end], na.rm = TRUE)
      signed_mass_slice <- as.numeric(bh_data[form_start:form_end, "mass"])
      direction <- ifelse(mean(signed_mass_slice, na.rm = TRUE) > 0, "long", "short")
      formations[[length(formations) + 1]] <- list(
        start     = dates[form_start],
        end       = dates[form_end],
        peak_mass = peak_mass,
        direction = direction,
        duration  = duration,
        start_idx = form_start,
        end_idx   = form_end
      )
    }
  }

  if (length(formations) == 0) {
    return(data.frame(
      start=character(0), end=character(0),
      peak_mass=numeric(0), direction=character(0),
      duration=integer(0), stringsAsFactors=FALSE
    ))
  }

  out <- do.call(rbind, lapply(formations, as.data.frame))
  out$start <- as.Date(unlist(out$start))
  out$end   <- as.Date(unlist(out$end))
  attr(out, "n_formations") <- nrow(out)
  return(out)
}


#' formation_entry_signals
#'
#' Generate entry/exit signals based on BH formation transitions.
#'
#' @param bh_data xts output of compute_bh_mass
#' @param formations data.frame from detect_bh_formations
#' @return xts with columns: signal (1=long, -1=short, 0=flat)
formation_entry_signals <- function(bh_data, formations) {
  n    <- nrow(bh_data)
  dates <- index(bh_data)
  sig  <- rep(0L, n)

  for (k in seq_len(nrow(formations))) {
    form <- formations[k, ]
    # Find index bounds
    start_i <- which(dates >= form$start)[1]
    end_i   <- tail(which(dates <= form$end), 1)
    if (is.na(start_i) || length(end_i) == 0) next

    dir_val <- if (form$direction == "long") 1L else -1L
    sig[start_i:end_i] <- dir_val
  }

  xts(data.frame(signal = sig), order.by = dates)
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Regime Classification
# ─────────────────────────────────────────────────────────────────────────────

#' classify_bh_regime
#'
#' Classify market regime based on BH mass dynamics.
#' Regimes:
#'   1 = "Dormant"     : |mass| < threshold/2  (flat, no energy)
#'   2 = "Forming"     : threshold/2 <= |mass| < threshold
#'   3 = "Active"      : |mass| >= threshold, accreting
#'   4 = "Evaporating" : |mass| >= threshold, dissipating
#'
#' @param signed_mass numeric vector of signed BH mass
#' @param threshold numeric, event-horizon level
#' @param deriv numeric vector of mass derivative (optional)
#' @return factor vector of regime labels
classify_bh_regime <- function(signed_mass, threshold = 1.5, deriv = NULL) {
  n <- length(signed_mass)
  abs_mass <- abs(signed_mass)

  if (is.null(deriv)) {
    deriv <- c(0, diff(abs_mass))
  }

  regime <- character(n)
  for (i in seq_len(n)) {
    m <- abs_mass[i]
    d <- deriv[i]
    if (is.na(m)) {
      regime[i] <- "Dormant"
    } else if (m < threshold / 2) {
      regime[i] <- "Dormant"
    } else if (m < threshold) {
      regime[i] <- "Forming"
    } else if (d >= 0) {
      regime[i] <- "Active"
    } else {
      regime[i] <- "Evaporating"
    }
  }

  factor(regime, levels = c("Dormant", "Forming", "Active", "Evaporating"))
}


#' rolling_regime_stats
#'
#' Compute rolling statistics of regime transitions.
#'
#' @param bh_data xts output of compute_bh_mass
#' @param window integer, rolling window for regime stats
#' @return xts with columns: pct_active, pct_dormant, regime_entropy, transitions
rolling_regime_stats <- function(bh_data, window = 60L) {
  regime <- as.numeric(bh_data[, "regime"])
  n <- length(regime)
  dates <- index(bh_data)

  pct_active   <- rep(NA_real_, n)
  pct_dormant  <- rep(NA_real_, n)
  entropy      <- rep(NA_real_, n)
  transitions  <- rep(NA_integer_, n)

  for (i in window:n) {
    r_slice <- regime[(i - window + 1):i]
    # 1=Dormant, 2=Forming, 3=Active, 4=Evaporating
    pct_active[i]  <- mean(r_slice >= 3, na.rm = TRUE)
    pct_dormant[i] <- mean(r_slice == 1, na.rm = TRUE)

    # Shannon entropy of regime distribution
    props <- table(r_slice) / window
    ent <- -sum(props * log(props + 1e-10))
    entropy[i] <- ent

    # Transitions count
    trans <- sum(diff(r_slice) != 0, na.rm = TRUE)
    transitions[i] <- trans
  }

  xts(
    data.frame(
      pct_active   = pct_active,
      pct_dormant  = pct_dormant,
      regime_entropy = entropy,
      transitions  = transitions
    ),
    order.by = dates
  )
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Mass Series Plots
# ─────────────────────────────────────────────────────────────────────────────

#' plot_bh_mass_series
#'
#' Plot BH mass series with regime coloring and event-horizon threshold.
#'
#' @param bh_data xts output of compute_bh_mass
#' @param formations data.frame from detect_bh_formations (optional)
#' @param title character, plot title
#' @return ggplot2 object
plot_bh_mass_series <- function(bh_data, formations = NULL, title = "BH Mass Series") {
  df <- data.frame(
    date       = as.Date(index(bh_data)),
    price      = as.numeric(bh_data[, "price"]),
    mass       = as.numeric(bh_data[, "mass"]),
    abs_mass   = as.numeric(bh_data[, "abs_mass"]),
    regime     = factor(as.numeric(bh_data[, "regime"]),
                        levels = 1:4,
                        labels = c("Dormant", "Forming", "Active", "Evaporating")),
    stringsAsFactors = FALSE
  )

  threshold <- attr(bh_data, "threshold")
  if (is.null(threshold)) threshold <- 1.5

  regime_colors <- c(
    "Dormant"     = "#7f7f7f",
    "Forming"     = "#ffaa00",
    "Active"      = "#e63946",
    "Evaporating" = "#2196F3"
  )

  # Top panel: price
  p_price <- ggplot(df, aes(x = date, y = price)) +
    geom_line(color = "#1a1a2e", linewidth = 0.6) +
    scale_x_date(date_labels = "%Y-%m") +
    labs(title = title, x = NULL, y = "Price") +
    theme_minimal(base_size = 10) +
    theme(panel.grid.minor = element_blank())

  # Add formation shading if provided
  if (!is.null(formations) && nrow(formations) > 0) {
    form_df <- formations
    form_df$start <- as.Date(form_df$start)
    form_df$end   <- as.Date(form_df$end)
    form_df$fill_color <- ifelse(form_df$direction == "long", "#c8f7c5", "#f7c5c5")

    for (k in seq_len(nrow(form_df))) {
      p_price <- p_price +
        annotate("rect",
                 xmin = form_df$start[k], xmax = form_df$end[k],
                 ymin = -Inf, ymax = Inf,
                 alpha = 0.15, fill = form_df$fill_color[k])
    }
  }

  # Bottom panel: mass with regime coloring
  p_mass <- ggplot(df, aes(x = date, y = abs_mass)) +
    geom_line(aes(color = regime), linewidth = 0.7) +
    geom_hline(yintercept = threshold, linetype = "dashed",
               color = "red", linewidth = 0.5, alpha = 0.8) +
    geom_hline(yintercept = threshold / 2, linetype = "dotted",
               color = "#ffaa00", linewidth = 0.4, alpha = 0.6) +
    annotate("text", x = min(df$date), y = threshold * 1.05,
             label = "Event Horizon", hjust = 0, size = 3, color = "red") +
    scale_color_manual(values = regime_colors) +
    scale_x_date(date_labels = "%Y-%m") +
    labs(x = "Date", y = "|Mass|", color = "Regime") +
    theme_minimal(base_size = 10) +
    theme(panel.grid.minor = element_blank(),
          legend.position = "bottom")

  # Signed mass panel
  p_signed <- ggplot(df, aes(x = date, y = mass)) +
    geom_line(color = "#2196F3", linewidth = 0.5, alpha = 0.7) +
    geom_hline(yintercept = 0, color = "black", linewidth = 0.3) +
    geom_hline(yintercept = c(-threshold, threshold), linetype = "dashed",
               color = "red", linewidth = 0.4, alpha = 0.7) +
    scale_x_date(date_labels = "%Y-%m") +
    labs(x = "Date", y = "Signed Mass") +
    theme_minimal(base_size = 10)

  # Combine using cowplot or just return main mass plot
  list(price = p_price, mass = p_mass, signed_mass = p_signed)
}


#' plot_regime_timeline
#'
#' Plot regime distribution as colored horizontal bands over time.
#'
#' @param bh_data xts output of compute_bh_mass
#' @return ggplot2 object
plot_regime_timeline <- function(bh_data) {
  df <- data.frame(
    date   = as.Date(index(bh_data)),
    regime = factor(as.numeric(bh_data[, "regime"]),
                    levels = 1:4,
                    labels = c("Dormant", "Forming", "Active", "Evaporating")),
    stringsAsFactors = FALSE
  )

  regime_colors <- c(
    "Dormant"     = "#d3d3d3",
    "Forming"     = "#ffd966",
    "Active"      = "#e63946",
    "Evaporating" = "#457b9d"
  )

  ggplot(df, aes(x = date, y = 1, fill = regime)) +
    geom_tile(height = 1) +
    scale_fill_manual(values = regime_colors) +
    scale_x_date(date_labels = "%Y-%m") +
    labs(title = "BH Regime Timeline", x = "Date", y = NULL, fill = "Regime") +
    theme_minimal(base_size = 10) +
    theme(
      axis.text.y  = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid   = element_blank(),
      legend.position = "bottom"
    )
}


#' summarize_bh_regimes
#'
#' Compute summary statistics per regime (returns, Sharpe, duration).
#'
#' @param bh_data xts output of compute_bh_mass
#' @return data.frame with per-regime statistics
summarize_bh_regimes <- function(bh_data) {
  regime_labels <- c("Dormant", "Forming", "Active", "Evaporating")
  df <- data.frame(
    regime = as.numeric(bh_data[, "regime"]),
    log_return = as.numeric(bh_data[, "log_return"]),
    stringsAsFactors = FALSE
  )
  df <- df[!is.na(df$log_return), ]

  out <- do.call(rbind, lapply(1:4, function(r) {
    sub <- df[df$regime == r, ]
    if (nrow(sub) < 2) {
      return(data.frame(
        regime     = regime_labels[r],
        n_obs      = nrow(sub),
        mean_ret   = NA_real_,
        sd_ret     = NA_real_,
        sharpe     = NA_real_,
        pct_positive = NA_real_
      ))
    }
    mu  <- mean(sub$log_return, na.rm = TRUE)
    sig <- sd(sub$log_return, na.rm = TRUE)
    data.frame(
      regime      = regime_labels[r],
      n_obs       = nrow(sub),
      mean_ret    = mu * 252,
      sd_ret      = sig * sqrt(252),
      sharpe      = if (sig > 0) mu / sig * sqrt(252) else NA_real_,
      pct_positive = mean(sub$log_return > 0, na.rm = TRUE)
    )
  }))

  rownames(out) <- NULL
  out
}
