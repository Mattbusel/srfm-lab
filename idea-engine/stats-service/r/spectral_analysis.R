# =============================================================================
# spectral_analysis.R
# Spectral and frequency domain analysis for crypto/quant trading
# Base R only -- no external packages
# =============================================================================
# Financial intuition: Price series contain cycles at multiple frequencies.
# Spectral analysis decomposes variance into frequency components, helping
# identify dominant trading cycles (e.g., intraday, weekly, monthly rhythms)
# that can be exploited for mean-reversion or momentum strategies.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. PERIODOGRAM AND SPECTRAL DENSITY
# -----------------------------------------------------------------------------

#' Compute raw periodogram of a time series
#' The periodogram is the squared modulus of the DFT, measuring how much
#' variance is attributable to each frequency
#' @param x numeric time series (demeaned internally)
#' @return list with freq, spec (periodogram ordinates), and period
raw_periodogram <- function(x) {
  x <- x - mean(x, na.rm = TRUE)
  n <- length(x)
  # Frequencies: 0, 1/n, 2/n, ..., floor(n/2)/n
  n_freq <- floor(n / 2)
  freq   <- seq_len(n_freq) / n

  # Compute DFT coefficients manually
  spec <- numeric(n_freq)
  for (k in seq_len(n_freq)) {
    omega <- 2 * pi * k / n
    re_k <- sum(x * cos(omega * seq_len(n)))
    im_k <- sum(x * sin(omega * seq_len(n)))
    spec[k] <- (re_k^2 + im_k^2) / n
  }

  list(freq = freq, spec = spec, period = 1 / freq, n = n)
}

#' Fast periodogram using R's built-in FFT
fast_periodogram <- function(x) {
  x <- x - mean(x, na.rm = TRUE)
  n <- length(x)
  dft <- fft(x)
  n_freq <- floor(n / 2)
  # Power spectrum: |X(k)|^2 / n, k = 1..floor(n/2)
  spec <- Mod(dft[2:(n_freq + 1)])^2 / n
  freq <- seq_len(n_freq) / n
  list(freq = freq, spec = spec, period = 1 / freq, n = n)
}

#' Find dominant frequencies (peaks in periodogram)
#' @param pgram output of fast_periodogram
#' @param n_peaks number of peaks to return
dominant_frequencies <- function(pgram, n_peaks = 5) {
  spec <- pgram$spec
  freq <- pgram$freq

  # Simple peak finding: local maxima
  is_peak <- c(FALSE,
               spec[2:(length(spec)-1)] > spec[1:(length(spec)-2)] &
               spec[2:(length(spec)-1)] > spec[3:length(spec)],
               FALSE)
  peak_idx   <- which(is_peak)
  peak_power <- spec[peak_idx]
  top_peaks  <- head(peak_idx[order(-peak_power)], n_peaks)

  data.frame(
    rank   = seq_along(top_peaks),
    freq   = freq[top_peaks],
    period = 1 / freq[top_peaks],
    power  = spec[top_peaks],
    pct_var = 100 * spec[top_peaks] / sum(spec)
  )
}

# -----------------------------------------------------------------------------
# 2. SMOOTHED PERIODOGRAM (SPECTRAL DENSITY ESTIMATION)
# -----------------------------------------------------------------------------

#' Bartlett window for spectral smoothing
#' Weights: triangular taper that reduces leakage
#' @param m bandwidth (half-window)
bartlett_window <- function(m) {
  k <- -m:m
  w <- 1 - abs(k) / (m + 1)
  w / sum(w)
}

#' Parzen window (smoother, better bias-variance tradeoff)
parzen_window <- function(m) {
  k <- -m:m
  u <- abs(k) / m
  w <- ifelse(u <= 0.5,
              1 - 6 * u^2 + 6 * u^3,
              2 * (1 - u)^3)
  w / sum(w)
}

#' Tukey-Hanning window (commonly used in econometrics)
tukey_window <- function(m) {
  k <- -m:m
  w <- 0.5 * (1 + cos(pi * k / m))
  w / sum(w)
}

#' Smoothed spectral density estimate
#' @param x time series
#' @param window_type "bartlett", "parzen", or "tukey"
#' @param bandwidth smoothing bandwidth
smoothed_spectrum <- function(x, window_type = "parzen", bandwidth = 10) {
  pgram <- fast_periodogram(x)
  n_freq <- length(pgram$spec)

  w <- switch(window_type,
    bartlett = bartlett_window(bandwidth),
    parzen   = parzen_window(bandwidth),
    tukey    = tukey_window(bandwidth),
    stop("Unknown window type")
  )
  half_w <- length(w) %/% 2

  # Convolve periodogram with window
  spec_smooth <- numeric(n_freq)
  for (k in seq_len(n_freq)) {
    idx <- k + (-half_w:half_w)
    # Reflect at boundaries
    idx <- pmax(1, pmin(n_freq, idx))
    spec_smooth[k] <- sum(w * pgram$spec[idx])
  }

  list(freq = pgram$freq, spec = spec_smooth,
       spec_raw = pgram$spec, period = pgram$period,
       bandwidth = bandwidth, window = window_type)
}

# -----------------------------------------------------------------------------
# 3. CROSS-SPECTRUM AND COHERENCE
# -----------------------------------------------------------------------------

#' Compute cross-spectrum between two series
#' Cross-spectrum reveals how two asset returns co-vary at each frequency --
#' useful for identifying lead-lag relationships between crypto pairs
#' @param x,y time series of equal length
cross_spectrum <- function(x, y, bandwidth = 10) {
  x <- x - mean(x, na.rm = TRUE)
  y <- y - mean(y, na.rm = TRUE)
  n <- length(x)
  n_freq <- floor(n / 2)

  # DFT of both series
  dft_x <- fft(x)[2:(n_freq + 1)]
  dft_y <- fft(y)[2:(n_freq + 1)]
  freq   <- seq_len(n_freq) / n

  # Raw cross-periodogram (complex)
  cross_pgram <- Conj(dft_x) * dft_y / n

  # Smooth real and imaginary parts separately with Parzen window
  w <- parzen_window(bandwidth)
  half_w <- length(w) %/% 2

  smooth_fn <- function(v) {
    n_f <- length(v)
    out <- numeric(n_f)
    for (k in seq_len(n_f)) {
      idx <- pmax(1, pmin(n_f, k + (-half_w:half_w)))
      out[k] <- sum(w * v[idx])
    }
    out
  }

  co_spec  <- smooth_fn(Re(cross_pgram))   # co-spectrum
  quad_spec <- smooth_fn(Im(cross_pgram))  # quadrature spectrum
  spec_x   <- smooth_fn(Mod(dft_x)^2 / n)
  spec_y   <- smooth_fn(Mod(dft_y)^2 / n)

  # Coherence: |cross|^2 / (spec_x * spec_y) in [0,1]
  coherence <- (co_spec^2 + quad_spec^2) / (spec_x * spec_y + 1e-20)

  # Phase: angle of cross-spectrum (lead-lag in radians)
  phase <- atan2(quad_spec, co_spec)
  # Convert to time units: lag = phase / (2*pi*freq)
  lag_periods <- phase / (2 * pi * freq)

  # Gain: amplitude of y's response to x
  gain <- sqrt(co_spec^2 + quad_spec^2) / (spec_x + 1e-20)

  list(freq = freq, period = 1/freq,
       co_spec = co_spec, quad_spec = quad_spec,
       coherence = pmin(coherence, 1), phase = phase,
       gain = gain, lag_periods = lag_periods,
       spec_x = spec_x, spec_y = spec_y)
}

#' Identify frequency bands where coherence is significant
#' @param cs output of cross_spectrum
#' @param threshold coherence threshold for significance
significant_coherence_bands <- function(cs, threshold = 0.5) {
  sig_idx <- which(cs$coherence > threshold)
  if (length(sig_idx) == 0) {
    cat("No frequency bands with coherence above threshold.\n")
    return(invisible(NULL))
  }
  cat(sprintf("Significant coherence (threshold=%.2f) at frequencies:\n", threshold))
  df <- data.frame(
    freq       = cs$freq[sig_idx],
    period     = cs$period[sig_idx],
    coherence  = cs$coherence[sig_idx],
    phase_rad  = cs$phase[sig_idx],
    lag_periods = cs$lag_periods[sig_idx]
  )
  df <- df[order(-df$coherence), ]
  print(head(df, 10))
  invisible(df)
}

# -----------------------------------------------------------------------------
# 4. GRANGER CAUSALITY VIA FREQUENCY DOMAIN (Geweke 1982)
# -----------------------------------------------------------------------------

#' Frequency-domain Granger causality decomposition
#' Geweke (1982): decomposes Granger causality into frequency contributions
#' Intuition: does high-frequency (intraday) BTC activity cause ETH moves,
#' or is it the low-frequency (weekly) trend that matters more?
#' @param x,y bivariate time series
#' @param p VAR order
freq_domain_granger <- function(x, y, p = 5, n_freq = 128) {
  n <- length(x)
  Y <- cbind(x, y)

  # Fit bivariate VAR(p) via OLS
  # Build regressor matrix
  T_eff <- n - p
  X_mat <- matrix(0, T_eff, 2 * p + 1)  # constant + p lags of each
  X_mat[, 1] <- 1
  for (lag in 1:p) {
    X_mat[, 1 + lag]     <- x[(p - lag + 1):(n - lag)]
    X_mat[, 1 + p + lag] <- y[(p - lag + 1):(n - lag)]
  }
  Y_mat <- Y[(p + 1):n, ]

  # OLS for each equation
  B1 <- solve(t(X_mat) %*% X_mat) %*% t(X_mat) %*% Y_mat[, 1]
  B2 <- solve(t(X_mat) %*% X_mat) %*% t(X_mat) %*% Y_mat[, 2]

  resid1 <- Y_mat[, 1] - X_mat %*% B1
  resid2 <- Y_mat[, 2] - X_mat %*% B2
  Sigma  <- cov(cbind(resid1, resid2))

  # Coefficient matrices A_1,...,A_p
  A_mats <- lapply(seq_len(p), function(lag) {
    matrix(c(B1[1 + lag], B2[1 + lag],
             B1[1 + p + lag], B2[1 + p + lag]), 2, 2, byrow = TRUE)
  })

  # Compute spectral density matrix at each frequency
  freqs <- seq(0, 0.5, length.out = n_freq)
  gc_1_to_2 <- numeric(n_freq)
  gc_2_to_1 <- numeric(n_freq)

  for (k in seq_along(freqs)) {
    omega <- 2 * pi * freqs[k]
    # Transfer function A(omega) = I - sum A_l * exp(-i*omega*l)
    A_omega <- diag(2)
    for (lag in seq_len(p)) {
      A_omega <- A_omega - A_mats[[lag]] * exp(-1i * omega * lag)
    }
    H <- solve(A_omega)  # spectral transfer matrix
    S <- H %*% Sigma %*% Conj(t(H))  # spectral density matrix

    # Granger causality measures at frequency omega
    # GC from x to y: log(S_yy / (S_yy - |H_yx|^2 * Sigma_xx))
    gc_1_to_2[k] <- log(abs(S[2,2]) / abs(S[2,2] - abs(H[2,1])^2 * Sigma[1,1] + 1e-20))
    gc_2_to_1[k] <- log(abs(S[1,1]) / abs(S[1,1] - abs(H[1,2])^2 * Sigma[2,2] + 1e-20))
  }

  # Total Granger causality = integral over all frequencies
  total_gc_1to2 <- mean(gc_1_to_2) * 2  # normalize by 2*pi / 2pi
  total_gc_2to1 <- mean(gc_2_to_1) * 2

  cat("=== Frequency-Domain Granger Causality ===\n")
  cat(sprintf("Total GC x->y: %.4f\n", total_gc_1to2))
  cat(sprintf("Total GC y->x: %.4f\n", total_gc_2to1))

  list(freq = freqs, gc_x_to_y = gc_1_to_2, gc_y_to_x = gc_2_to_1,
       total_gc_x_to_y = total_gc_1to2, total_gc_y_to_x = total_gc_2to1,
       dominant_gc_freq_x = freqs[which.max(gc_1_to_2)],
       dominant_gc_freq_y = freqs[which.max(gc_2_to_1)])
}

# -----------------------------------------------------------------------------
# 5. BAND-PASS FILTERING
# -----------------------------------------------------------------------------

#' Hodrick-Prescott filter
#' Separates trend (low-freq) from cycle (high-freq) component
#' lambda: smoothing parameter (1600 for quarterly, 14400 for monthly,
#' 100 for daily, 6.25 for annual)
#' @param x time series
#' @param lambda smoothing parameter
hp_filter <- function(x, lambda = 1600) {
  n <- length(x)
  # Build difference matrix D_2 (second-order)
  # HP filter minimizes sum(cycle^2) + lambda * sum((D^2 trend)^2)
  # Solution: trend = (I + lambda * t(D2) %*% D2)^{-1} %*% x
  D1 <- diff(diag(n), differences = 1)
  D2 <- diff(diag(n), differences = 2)
  A  <- diag(n) + lambda * t(D2) %*% D2
  trend <- solve(A, x)
  cycle <- x - trend
  list(trend = trend, cycle = cycle, lambda = lambda)
}

#' Christiano-Fitzgerald (CF) band-pass filter
#' Isolates cycles between specified period bounds
#' Useful for extracting the "business cycle" or "trading cycle" frequency
#' @param x time series
#' @param pl lower period bound (e.g., 6 months = 6)
#' @param pu upper period bound (e.g., 32 months = 32)
cf_filter <- function(x, pl = 6, pu = 32) {
  n <- length(x)
  # Optimal weights for random walk assumption
  # B_k = (sin(k*2*pi/pl) - sin(k*2*pi/pu)) / (pi*k)
  # B_0 = 2*pi/pl - 2*pi/pu  ... actually use frequency cutoffs
  omega_u <- 2 * pi / pl  # upper frequency (lower period)
  omega_l <- 2 * pi / pu  # lower frequency (upper period)

  # CF weights (symmetric, finite approximation)
  K <- floor(n / 2)
  b <- numeric(K + 1)
  b[1] <- (omega_u - omega_l) / pi  # B_0
  for (k in 1:K) {
    b[k + 1] <- (sin(k * omega_u) - sin(k * omega_l)) / (pi * k)
  }

  # Apply filter (two-sided convolution, truncated at edges)
  cycle <- numeric(n)
  for (t in seq_len(n)) {
    val <- b[1] * x[t]
    for (k in 1:K) {
      t_pos <- t + k; t_neg <- t - k
      if (t_pos <= n) val <- val + b[k+1] * x[t_pos]
      if (t_neg >= 1) val <- val + b[k+1] * x[t_neg]
    }
    cycle[t] <- val
  }

  trend <- x - cycle
  list(cycle = cycle, trend = trend, pl = pl, pu = pu)
}

#' Butterworth band-pass filter (recursive IIR)
#' @param x time series
#' @param low_cutoff lower frequency cutoff (0 to 0.5)
#' @param high_cutoff upper frequency cutoff (0 to 0.5)
#' @param order filter order
butterworth_bandpass <- function(x, low_cutoff = 0.05, high_cutoff = 0.25,
                                  order = 2) {
  # Pre-warp frequencies for bilinear transform
  w1 <- tan(pi * low_cutoff)
  w2 <- tan(pi * high_cutoff)
  w0 <- sqrt(w1 * w2)
  bw <- w2 - w1

  # Simplified 2nd-order implementation
  # Transfer function in z-domain via bilinear transform
  b_coef <- c(bw^2, 0, -bw^2)  # numerator
  a_coef <- c(1 + bw + w0^2,
              2 * w0^2 - 2,
              1 - bw + w0^2)

  # Normalize
  b_coef <- b_coef / a_coef[1]
  a_coef <- a_coef / a_coef[1]

  # Filter via direct form II
  n  <- length(x)
  y  <- numeric(n)
  d1 <- 0; d2 <- 0
  for (i in seq_len(n)) {
    w_i <- x[i] - a_coef[2] * d1 - a_coef[3] * d2
    y[i] <- b_coef[1] * w_i + b_coef[2] * d1 + b_coef[3] * d2
    d2 <- d1; d1 <- w_i
  }
  y
}

# -----------------------------------------------------------------------------
# 6. CYCLE EXTRACTION AND TURNING POINT DATING
# -----------------------------------------------------------------------------

#' NBER-style turning point algorithm (Bry-Boschan procedure)
#' Identifies peaks and troughs in the cycle component
#' @param cycle extracted cycle series
#' @param min_cycle_length minimum number of periods per cycle phase
#' @param window local extrema search window
bry_boschan <- function(cycle, min_cycle_length = 5, window = 3) {
  n <- length(cycle)
  # Step 1: Find candidate peaks and troughs
  peaks   <- logical(n)
  troughs <- logical(n)

  for (i in (window+1):(n-window)) {
    neighborhood <- cycle[(i-window):(i+window)]
    if (cycle[i] == max(neighborhood) && cycle[i] > cycle[i-1]) {
      peaks[i] <- TRUE
    }
    if (cycle[i] == min(neighborhood) && cycle[i] < cycle[i-1]) {
      troughs[i] <- TRUE
    }
  }

  # Step 2: Enforce alternation of peaks and troughs
  turns <- data.frame(
    t    = c(which(peaks), which(troughs)),
    type = c(rep("peak", sum(peaks)), rep("trough", sum(troughs))),
    val  = c(cycle[peaks], cycle[troughs])
  )
  turns <- turns[order(turns$t), ]

  # Remove non-alternating
  if (nrow(turns) > 0) {
    keep <- logical(nrow(turns))
    keep[1] <- TRUE
    for (i in 2:nrow(turns)) {
      if (turns$type[i] != turns$type[i-1]) {
        keep[i] <- TRUE
      } else {
        # Keep the more extreme one
        prev_keep <- max(which(keep[1:(i-1)]))
        if ((turns$type[i] == "peak"   && turns$val[i] > turns$val[prev_keep]) ||
            (turns$type[i] == "trough" && turns$val[i] < turns$val[prev_keep])) {
          keep[prev_keep] <- FALSE
          keep[i] <- TRUE
        }
      }
    }
    turns <- turns[keep, ]
  }

  # Step 3: Apply minimum cycle length rule
  if (nrow(turns) > 1) {
    valid <- logical(nrow(turns))
    valid[1] <- TRUE
    for (i in 2:nrow(turns)) {
      if (turns$t[i] - turns$t[i-1] >= min_cycle_length) valid[i] <- TRUE
    }
    turns <- turns[valid, ]
  }

  # Expansion/contraction periods
  expansions  <- list()
  contractions <- list()
  if (nrow(turns) >= 2) {
    for (i in 2:nrow(turns)) {
      dur <- turns$t[i] - turns$t[i-1]
      if (turns$type[i] == "peak") {
        expansions[[length(expansions)+1]] <- list(
          start = turns$t[i-1], end = turns$t[i], duration = dur)
      } else {
        contractions[[length(contractions)+1]] <- list(
          start = turns$t[i-1], end = turns$t[i], duration = dur)
      }
    }
  }

  list(peaks = which(peaks), troughs = which(troughs),
       turns = turns,
       expansions = expansions, contractions = contractions,
       avg_expansion  = if (length(expansions)  > 0) mean(sapply(expansions,  `[[`, "duration")) else NA,
       avg_contraction = if (length(contractions) > 0) mean(sapply(contractions, `[[`, "duration")) else NA)
}

# -----------------------------------------------------------------------------
# 7. WAVELET ANALYSIS
# -----------------------------------------------------------------------------

#' Morlet wavelet (complex): psi(t) = pi^{-1/4} * exp(i*omega_0*t) * exp(-t^2/2)
#' Provides time-frequency decomposition; unlike FFT, wavelet analysis can
#' detect non-stationary cycles where frequency changes over time
#' @param t time vector
#' @param omega_0 central frequency (default 6, good for financial data)
morlet_wavelet <- function(t, omega_0 = 6) {
  pi^(-1/4) * exp(1i * omega_0 * t) * exp(-t^2 / 2)
}

#' Continuous Wavelet Transform using Morlet wavelet
#' @param x time series
#' @param scales wavelet scales (related to period by: period ≈ 4*pi*s/(omega_0+sqrt(2+omega_0^2)))
#' @param omega_0 Morlet parameter
cwt_morlet <- function(x, scales = 2^seq(1, 8, by = 0.25), omega_0 = 6) {
  n <- length(x)
  n_scales <- length(scales)
  W <- matrix(0 + 0i, n_scales, n)

  # Compute via FFT for efficiency
  # Psi_hat(omega) = sqrt(2*pi*s) * H(omega) * psi_hat_0(s*omega)
  # where H(omega) = 1 for omega>0 (analytic signal)
  x_hat <- fft(x)
  omega_arr <- 2 * pi * c(0:(n/2), -(floor(n/2)-1):(-1)) / n

  for (j in seq_len(n_scales)) {
    s <- scales[j]
    # Fourier transform of normalized Morlet at scale s
    psi_hat <- sqrt(2 * pi * s) *
               pi^(-1/4) * exp(-(s * omega_arr - omega_0)^2 / 2)
    psi_hat[omega_arr < 0] <- 0  # analytic signal (one-sided)
    # Inverse FFT gives wavelet coefficients
    W[j, ] <- fft(x_hat * psi_hat, inverse = TRUE) / n
  }

  # Period for each scale
  period <- 4 * pi * scales / (omega_0 + sqrt(2 + omega_0^2))

  # Wavelet power spectrum
  power <- Mod(W)^2

  list(W = W, power = power, scales = scales, period = period,
       n = n, omega_0 = omega_0)
}

#' Wavelet coherence between two series
#' @param cwt_x,cwt_y CWT outputs from cwt_morlet for two series
wavelet_coherence <- function(cwt_x, cwt_y, smooth_scale = 2) {
  Wxy <- cwt_x$W * Conj(cwt_y$W)  # cross-wavelet transform
  Wxx <- cwt_x$W * Conj(cwt_x$W)
  Wyy <- cwt_y$W * Conj(cwt_y$W)

  # Smooth over time (simple moving average)
  smooth_time <- function(mat, k = smooth_scale) {
    n <- ncol(mat)
    out <- mat
    if (k > 1) {
      for (j in seq_len(nrow(mat))) {
        row <- mat[j, ]
        row_smooth <- stats::filter(Re(row), rep(1/k, k), sides = 2)
        row_smooth[is.na(row_smooth)] <- Re(row)[is.na(row_smooth)]
        out[j, ] <- row_smooth
      }
    }
    out
  }

  S_xy <- smooth_time(Wxy)
  S_xx <- smooth_time(Wxx)
  S_yy <- smooth_time(Wyy)

  coherence <- Mod(S_xy)^2 / (Re(S_xx) * Re(S_yy) + 1e-20)
  phase      <- atan2(Im(Wxy), Re(Wxy))

  list(cross_wavelet = Wxy, coherence = pmin(coherence, 1),
       phase = phase, period = cwt_x$period, power_x = cwt_x$power,
       power_y = cwt_y$power)
}

#' Global wavelet spectrum (average power across time)
global_wavelet_spectrum <- function(cwt_result) {
  gws <- rowMeans(cwt_result$power)
  list(period = cwt_result$period, gws = gws,
       dominant_period = cwt_result$period[which.max(gws)])
}

# -----------------------------------------------------------------------------
# 8. COMPREHENSIVE SPECTRAL ANALYSIS PIPELINE
# -----------------------------------------------------------------------------

#' Full spectral analysis report for a financial time series
#' @param x time series (e.g., log-returns or price)
#' @param label descriptive name for the series
#' @param y optional second series for cross-spectral analysis
spectral_report <- function(x, label = "Series", y = NULL,
                             hp_lambda = 100, cf_low = 5, cf_high = 40) {
  cat("=============================================================\n")
  cat(sprintf("SPECTRAL ANALYSIS: %s\n", label))
  cat(sprintf("n = %d observations\n\n", length(x)))

  # 1. Raw periodogram and dominant frequencies
  cat("--- Dominant Frequencies (Periodogram) ---\n")
  pgram  <- fast_periodogram(x)
  dom_f  <- dominant_frequencies(pgram, n_peaks = 5)
  print(dom_f)

  # 2. Smoothed spectrum
  spec_smooth <- smoothed_spectrum(x, window_type = "parzen", bandwidth = 8)

  # 3. HP filter
  cat("\n--- HP Filter Cycle Statistics ---\n")
  hp_res <- hp_filter(x, lambda = hp_lambda)
  cat(sprintf("Cycle: mean=%.4f, sd=%.4f, AR(1)=%.3f\n",
              mean(hp_res$cycle),
              sd(hp_res$cycle),
              cor(hp_res$cycle[-1], hp_res$cycle[-length(hp_res$cycle)])))

  # 4. CF filter
  cat(sprintf("\n--- CF Band-Pass Filter [%d, %d] periods ---\n", cf_low, cf_high))
  cf_res <- cf_filter(x, pl = cf_low, pu = cf_high)
  cat(sprintf("BP Cycle: sd=%.4f  (%.1f%% of total variance)\n",
              sd(cf_res$cycle),
              100 * var(cf_res$cycle) / var(x)))

  # 5. Turning points in HP cycle
  cat("\n--- Cycle Turning Points ---\n")
  bp_res <- bry_boschan(hp_res$cycle)
  cat(sprintf("Peaks:   %d found\n", length(bp_res$peaks)))
  cat(sprintf("Troughs: %d found\n", length(bp_res$troughs)))
  if (!is.na(bp_res$avg_expansion)) {
    cat(sprintf("Avg expansion duration:   %.1f periods\n", bp_res$avg_expansion))
    cat(sprintf("Avg contraction duration: %.1f periods\n", bp_res$avg_contraction))
  }

  # 6. Wavelet power spectrum
  cat("\n--- Wavelet Analysis ---\n")
  scales <- 2^seq(1, log2(length(x)/4), by = 0.5)
  if (length(scales) > 1) {
    cwt <- cwt_morlet(x, scales = scales)
    gws <- global_wavelet_spectrum(cwt)
    cat(sprintf("Dominant wavelet period: %.1f\n", gws$dominant_period))
  }

  # 7. Cross-spectral analysis if second series provided
  if (!is.null(y) && length(y) == length(x)) {
    cat("\n--- Cross-Spectral Analysis ---\n")
    cs <- cross_spectrum(x, y)
    sig_bands <- significant_coherence_bands(cs, threshold = 0.4)

    cat("\n--- Frequency-Domain Granger Causality ---\n")
    gc_res <- freq_domain_granger(x, y, p = 5)
  }

  invisible(list(pgram = pgram, spec_smooth = spec_smooth,
                 hp = hp_res, cf = cf_res, turning_points = bp_res))
}

# =============================================================================
# EXAMPLE
# =============================================================================
# set.seed(123)
# n <- 500
# # Simulate a price series with a 20-period cycle + noise
# t_idx <- seq_len(n)
# x <- cumsum(rnorm(n, 0, 1)) + 2*sin(2*pi*t_idx/20) + 0.5*sin(2*pi*t_idx/7)
# y <- 0.8*x + rnorm(n, 0, 1)  # correlated pair
# result <- spectral_report(x, label="BTC_cycle", y=y)

# =============================================================================
# ADDITIONAL: EXTENDED SPECTRAL TOOLS
# =============================================================================

#' Harmonic regression: fit known periods to time series
#' @param x time series
#' @param periods vector of periods to include (e.g., c(7, 30, 365) for weekly/monthly/annual)
harmonic_regression <- function(x, periods) {
  n <- length(x); t_idx <- seq_len(n)
  X <- matrix(1, n, 1 + 2*length(periods))
  for (i in seq_along(periods)) {
    X[, 2*i]   <- cos(2*pi*t_idx/periods[i])
    X[, 2*i+1] <- sin(2*pi*t_idx/periods[i])
  }
  b <- solve(t(X)%*%X) %*% t(X) %*% x
  fitted <- X %*% b; resid <- x - fitted
  r2 <- 1 - sum(resid^2)/sum((x-mean(x))^2)
  amplitudes <- sapply(seq_along(periods), function(i) sqrt(b[2*i]^2 + b[2*i+1]^2))
  names(amplitudes) <- paste0("T=", periods)
  phases <- sapply(seq_along(periods), function(i) atan2(b[2*i+1], b[2*i]))
  names(phases) <- paste0("T=", periods)
  cat("=== Harmonic Regression ===\n")
  cat(sprintf("R2 = %.4f\n", r2))
  cat("Amplitudes by period:\n"); print(round(amplitudes, 5))
  cat("Phases (radians):\n"); print(round(phases, 3))
  invisible(list(b=b, fitted=fitted, resid=resid, r2=r2,
                 amplitudes=amplitudes, phases=phases))
}

#' Lomb-Scargle periodogram for unevenly sampled data
#' Critical for on-chain metrics that may have irregular timestamps
#' @param t_obs observation times (numeric)
#' @param x observed values
lomb_scargle <- function(t_obs, x, n_freq=200) {
  n  <- length(t_obs); x_c <- x - mean(x)
  t_span <- max(t_obs) - min(t_obs)
  freq   <- seq(1/t_span, n/(2*t_span), length.out=n_freq)
  power  <- numeric(n_freq)
  for (k in seq_along(freq)) {
    omega <- 2*pi*freq[k]
    tau_v <- atan2(sum(sin(2*omega*t_obs)), sum(cos(2*omega*t_obs))) / (2*omega)
    cc <- cos(omega*(t_obs-tau_v)); ss <- sin(omega*(t_obs-tau_v))
    power[k] <- 0.5 * (sum(x_c*cc)^2/(sum(cc^2)+1e-10) +
                        sum(x_c*ss)^2/(sum(ss^2)+1e-10))
  }
  power_norm <- power / (var(x_c)+1e-10)
  dom_freq   <- freq[which.max(power_norm)]
  cat(sprintf("Lomb-Scargle dominant period: %.2f\n", 1/dom_freq))
  invisible(list(freq=freq, power=power_norm, dominant_freq=dom_freq,
                 dominant_period=1/dom_freq))
}

#' Multi-taper spectral estimator using sine tapers
#' Reduces spectral leakage compared to single-window periodogram
#' @param x time series
#' @param K number of tapers (default 4; more = smoother but lower resolution)
multitaper_spectrum <- function(x, K=4) {
  n <- length(x); x <- x - mean(x)
  n_freq <- floor(n/2); freq <- seq_len(n_freq)/n
  # Sine tapers: orthonormal, good approximation to DPSS
  tapers <- matrix(0, n, K)
  for (k in seq_len(K)) tapers[,k] <- sqrt(2/(n+1)) * sin(pi*k*seq_len(n)/(n+1))
  # Accumulate power across tapers
  spec_mt <- numeric(n_freq)
  for (k in seq_len(K)) {
    x_t  <- x * tapers[,k]
    dft_k <- fft(x_t)[2:(n_freq+1)]
    spec_mt <- spec_mt + Mod(dft_k)^2 / n
  }
  spec_mt <- spec_mt / K
  dom_period <- 1/freq[which.max(spec_mt)]
  cat(sprintf("Multitaper spectrum (K=%d): dominant period=%.1f\n", K, dom_period))
  invisible(list(freq=freq, spec=spec_mt, dominant_period=dom_period))
}

#' Time-varying spectrum via short-time Fourier transform (spectrogram)
#' Shows how dominant frequencies evolve over time -- useful for detecting
#' regime changes in crypto cycle behavior
#' @param x time series
#' @param window number of points per STFT window
#' @param overlap fraction of overlap between windows
spectrogram <- function(x, window=64, overlap=0.5) {
  n <- length(x); step <- max(1, round(window*(1-overlap)))
  starts <- seq(1, n-window+1, by=step)
  n_freq  <- floor(window/2)
  spec_mat <- matrix(0, n_freq, length(starts))
  # Hanning window taper
  taper <- 0.5 * (1 - cos(2*pi*seq(0,1,length.out=window)))
  for (i in seq_along(starts)) {
    seg <- x[starts[i]:(starts[i]+window-1)] - mean(x[starts[i]:(starts[i]+window-1)])
    dft <- fft(seg*taper)[2:(n_freq+1)]
    spec_mat[, i] <- Mod(dft)^2 / window
  }
  freq <- seq_len(n_freq) / window
  # Time-varying dominant frequency
  dom_freq_ts <- freq[apply(spec_mat, 2, which.max)]
  cat(sprintf("Spectrogram: %d windows, %d freqs\n", length(starts), n_freq))
  cat(sprintf("Dominant freq range: [%.4f, %.4f]\n", min(dom_freq_ts), max(dom_freq_ts)))
  invisible(list(spec=spec_mat, freq=freq, starts=starts,
                 dom_freq_ts=dom_freq_ts, window=window))
}

#' STL decomposition: Seasonal-Trend decomposition using Loess (manual implementation)
#' Decomposes series into trend, seasonal, and residual components
#' @param x time series
#' @param period seasonal period (e.g., 7 for weekly in daily data)
stl_decomp_manual <- function(x, period=7, n_iter=5) {
  n <- length(x)
  trend <- as.numeric(stats::filter(x, rep(1/period, period), sides=2))
  # Handle NAs at edges via linear interpolation
  trend_interp <- trend
  na_idx <- which(is.na(trend))
  if (length(na_idx) > 0) {
    trend_interp[na_idx] <- approx(which(!is.na(trend)), trend[!is.na(trend)],
                                    xout=na_idx)$y
  }
  trend_interp[is.na(trend_interp)] <- mean(x)
  seasonal <- numeric(n)
  for (iter in seq_len(n_iter)) {
    detrended <- x - trend_interp
    # Seasonal: average within each period position
    seasonal_avg <- tapply(detrended, (seq_len(n)-1) %% period + 1, mean, na.rm=TRUE)
    seasonal <- seasonal_avg[(seq_len(n)-1) %% period + 1]
    seasonal <- seasonal - mean(seasonal)  # zero-sum constraint
    deseasoned <- x - seasonal
    # Smooth trend
    trend_interp <- as.numeric(stats::filter(deseasoned, rep(1/period, period), sides=2))
    trend_interp <- approx(which(!is.na(trend_interp)), trend_interp[!is.na(trend_interp)],
                            xout=seq_len(n))$y
    trend_interp[is.na(trend_interp)] <- mean(deseasoned)
  }
  remainder <- x - trend_interp - seasonal
  cat(sprintf("STL: trend_var=%.4f, seasonal_var=%.4f, resid_var=%.4f\n",
              var(trend_interp, na.rm=T), var(seasonal), var(remainder, na.rm=T)))
  invisible(list(trend=trend_interp, seasonal=seasonal, remainder=remainder))
}

#' Instantaneous frequency via Hilbert transform analog (FFT-based)
instantaneous_frequency <- function(x) {
  n <- length(x)
  X_fft <- fft(x)
  # Zero out negative frequencies to get analytic signal
  X_anal <- X_fft
  if (n %% 2 == 0) {
    X_anal[(n/2+2):n] <- 0
    X_anal[n/2+1] <- X_anal[n/2+1] / 2
  } else {
    X_anal[ceiling(n/2+1):n] <- 0
  }
  x_anal <- fft(X_anal, inverse=TRUE) / n
  inst_phase <- atan2(Im(x_anal), Re(x_anal))
  # Unwrapped phase difference = instantaneous frequency
  phase_diff <- diff(inst_phase)
  phase_diff[phase_diff > pi]  <- phase_diff[phase_diff > pi]  - 2*pi
  phase_diff[phase_diff < -pi] <- phase_diff[phase_diff < -pi] + 2*pi
  inst_freq <- phase_diff / (2*pi)
  cat(sprintf("Instantaneous freq: mean=%.4f, sd=%.4f\n", mean(inst_freq), sd(inst_freq)))
  invisible(list(inst_phase=inst_phase, inst_freq=inst_freq))
}
