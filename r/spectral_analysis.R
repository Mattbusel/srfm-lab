# spectral_analysis.R
#
# Spectral and wavelet analysis of financial time series.
#
# Key methods:
#
#   1. Periodogram: raw |DFT|² — noisy spectral estimator
#   2. Smoothed periodogram: kernel smoothing (Daniell, Bartlett)
#   3. Multitaper (MTM): average over Slepian-tapered periodograms — reduces variance
#   4. Continuous Wavelet Transform (CWT): time-frequency localisation
#   5. Wavelet coherence: cross-CWT normalised by individual CWTs
#   6. Cross-spectrum: co-movement at each frequency
#   7. Evolutionary spectrum: time-varying frequency content
#
# The power spectrum S(f) satisfies:
#   Var(X) = ∫₀^{1/2} S(f) df   [Parseval's theorem]
#
# Financial applications:
#   - Detect daily/weekly seasonality in trading signals
#   - Identify lead-lag relationships via cross-spectrum phase
#   - BTC-ETH wavelet coherence for pairs trading
#
# Packages: multitaper, WaveletComp, astsa, ggplot2, dplyr, zoo

suppressPackageStartupMessages({
  library(multitaper)    # Thomson multitaper spectral estimate
  library(WaveletComp)   # CWT, wavelet power, coherence
  library(astsa)         # Shumway-Stoffer time series (mvspec)
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(zoo)
  library(RColorBrewer)
  library(scales)
})

# ─────────────────────────────────────────────────────────────────────────────
# PERIODOGRAM: RAW AND SMOOTHED
# ─────────────────────────────────────────────────────────────────────────────

#' Compute raw periodogram via FFT.
#'
#' For a zero-mean time series x_1,...,x_n:
#'   I(ω_j) = (1/n) |Σ_{t=1}^n x_t e^{-2πi ω_j t}|²
#'
#' where ω_j = j/n for j=1,...,⌊n/2⌋ are the Fourier frequencies.
#'
#' @param x     time series vector
#' @param detrend  if TRUE, remove linear trend first
#' @param taper    fraction of data to taper with cosine bell (0 = none, 0.1 typical)
#' @return  data.frame of (frequency, period, power)
raw_periodogram <- function(x, detrend = TRUE, taper = 0.1) {
  n <- length(x)

  # Detrend
  if (detrend) {
    x <- residuals(lm(x ~ seq_along(x)))
  }

  # Cosine bell taper (reduces Gibbs ringing at boundaries)
  if (taper > 0) {
    m <- floor(taper * n / 2)
    taper_vec <- rep(1.0, n)
    for (i in 1:m) {
      bell <- 0.5 * (1 - cos(pi * i / m))
      taper_vec[i]       <- bell
      taper_vec[n - i + 1] <- bell
    }
    x <- x * taper_vec
  }

  # FFT
  X <- fft(x)
  n_freq <- floor(n / 2)
  freqs  <- (1:n_freq) / n

  # One-sided periodogram (multiply by 2 for frequencies 1 to n/2-1)
  power <- (Mod(X[2:(n_freq+1)])^2) / n
  power[1:(n_freq-1)] <- 2 * power[1:(n_freq-1)]

  data.frame(
    frequency = freqs,
    period    = 1 / freqs,
    power     = power
  )
}

#' Daniell smoothed periodogram.
#'
#' Modified Daniell kernel: weights = 1/2m at endpoints, 1/m elsewhere.
#' Running average of the periodogram with bandwidth m.
#'
#' @param pgram  data.frame from raw_periodogram
#' @param spans  vector of bandwidths (applied sequentially), e.g. c(5, 5)
smooth_periodogram_daniell <- function(pgram, spans = c(3, 5)) {
  power <- pgram$power

  for (m in spans) {
    n_pts <- length(power)
    half_m <- floor(m / 2)

    # Modified Daniell kernel weights
    w <- rep(1/(m), m)
    w[1] <- w[m] <- 1/(2*m)
    w <- w / sum(w)  # normalise

    # Convolve
    power_smooth <- numeric(n_pts)
    for (i in 1:n_pts) {
      i_lo <- max(1, i - half_m)
      i_hi <- min(n_pts, i + half_m)
      ww <- w[(i - i_lo + 1):(i - i_hi + length(w))]
      ww <- ww[1:(i_hi - i_lo + 1)]
      ww <- ww / sum(ww)
      power_smooth[i] <- sum(power[i_lo:i_hi] * ww)
    }
    power <- power_smooth
  }

  pgram$power_smooth <- power
  return(pgram)
}

#' Bartlett's method: average periodograms of non-overlapping segments.
#'
#' Reduces variance at the cost of frequency resolution.
#'
#' @param x        time series
#' @param L        number of segments
#' @return  data.frame of smoothed periodogram
bartlett_periodogram <- function(x, L = 8) {
  n       <- length(x)
  seg_len <- floor(n / L)
  n_freq  <- floor(seg_len / 2)

  # Compute periodogram for each segment
  pgrams <- matrix(0, n_freq, L)
  for (l in 1:L) {
    seg <- x[((l-1)*seg_len + 1):(l*seg_len)]
    pg  <- raw_periodogram(seg, detrend = TRUE, taper = 0)
    pgrams[, l] <- pg$power[1:n_freq]
  }

  freqs <- (1:n_freq) / seg_len

  data.frame(
    frequency   = freqs,
    period      = 1 / freqs,
    power       = rowMeans(pgrams),
    power_se    = apply(pgrams, 1, sd) / sqrt(L)
  )
}

# ─────────────────────────────────────────────────────────────────────────────
# MULTITAPER SPECTRAL ESTIMATION (MTM)
# ─────────────────────────────────────────────────────────────────────────────

#' Multitaper spectral estimate using Thomson (1982) Slepian sequences.
#'
#' Uses K = 2NW - 1 Slepian (DPSS) tapers with time-bandwidth product NW.
#' Each taper j gives a "taper spectrum" Jₖ(f), and the MTM estimate is:
#'   Ŝ(f) = Σₖ λₖ|Jₖ(f)|² / Σₖ λₖ
#'
#' Advantages over single-taper:
#'   - Much lower spectral leakage
#'   - Improved variance (bias-variance tradeoff controlled by NW)
#'   - Produces F-test for periodic components
#'
#' @param x    time series
#' @param nw   time-bandwidth product (typically 2-4)
#' @param k    number of tapers (default 2*nw - 1)
#' @return  multitaper object from multitaper package
compute_mtm <- function(x, nw = 4, k = NULL, dt = 1) {
  if (is.null(k)) k <- 2 * nw - 1

  cat(sprintf("Multitaper spectral estimate: NW=%.1f, K=%d tapers\n", nw, k))
  cat(sprintf("  Frequency resolution: ±%.4f cycles (bandwidth = %.4f)\n",
              nw / length(x), 2*nw/length(x)))

  mt <- spec.mtm(x, nw = nw, k = k, deltat = dt,
                  plot = FALSE, verbose = FALSE)

  # Identify significant spectral peaks via F-test
  # F-statistic for line component at each frequency
  cat(sprintf("  Significant periodicities detected at p<0.05:\n"))
  sig_freqs <- mt$freq[mt$mtm$Ftest > qf(0.95, 2, 2*k-2)]
  if (length(sig_freqs) == 0) {
    cat("    None detected.\n")
  } else {
    for (f in sig_freqs[1:min(5, length(sig_freqs))]) {
      cat(sprintf("    f=%.6f (period=%.2f obs)\n", f, 1/f))
    }
  }

  return(mt)
}

#' Convert multitaper output to data.frame for plotting.
mtm_to_df <- function(mt) {
  data.frame(
    frequency = mt$freq,
    period    = 1 / mt$freq,
    power     = mt$spec,
    Ftest     = mt$mtm$Ftest
  )
}

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SPECTRUM AND COHERENCE
# ─────────────────────────────────────────────────────────────────────────────

#' Cross-spectrum between two time series.
#'
#' Ŝ_{XY}(f) = Σₜ ĉ_{XY}(τ) e^{-2πifτ}
#'
#' where ĉ_{XY}(τ) = (1/n) Σ_t (x_{t+τ} - x̄)(y_t - ȳ).
#'
#' Cross-spectral density components:
#'   Co-spectrum:  C(f) = Re[Ŝ_{XY}(f)]    (in-phase co-movement)
#'   Quadrature:   Q(f) = Im[Ŝ_{XY}(f)]    (out-of-phase)
#'   Coherence:    K²(f) = |Ŝ_{XY}(f)|² / (Ŝ_{XX}(f) · Ŝ_{YY}(f))   ∈ [0,1]
#'   Phase:        φ(f) = arctan(Q(f) / C(f))  [lead-lag at frequency f]
#'
#' @param x, y  aligned time series vectors
#' @param spans smoothing bandwidth (Daniell)
cross_spectrum <- function(x, y, spans = c(3, 5), taper = 0.1) {
  n <- length(x)
  stopifnot(length(y) == n)

  # Detrend and taper both series
  x <- residuals(lm(x ~ seq_along(x)))
  y <- residuals(lm(y ~ seq_along(y)))

  n_freq <- floor(n / 2)
  freqs  <- (1:n_freq) / n

  Xf <- fft(x)[2:(n_freq+1)]
  Yf <- fft(y)[2:(n_freq+1)]

  # Cross-periodogram
  Sxy <- Xf * Conj(Yf) / n
  Sxx <- Mod(Xf)^2 / n
  Syy <- Mod(Yf)^2 / n

  # Smooth each component with Daniell kernel
  smooth_daniell <- function(z, m) {
    z_smooth <- filter(z, rep(1/m, m), method = "convolution",
                       sides = 2, circular = TRUE)
    z_smooth
  }

  sm <- spans[1]
  co_raw  <- Re(Sxy)
  qu_raw  <- Im(Sxy)
  sxx_sm  <- smooth_daniell(Sxx, sm)
  syy_sm  <- smooth_daniell(Syy, sm)
  co_sm   <- smooth_daniell(co_raw, sm)
  qu_sm   <- smooth_daniell(qu_raw, sm)

  coherence <- (co_sm^2 + qu_sm^2) / pmax(sxx_sm * syy_sm, 1e-30)
  phase     <- atan2(qu_sm, co_sm)

  data.frame(
    frequency  = freqs,
    period     = 1 / freqs,
    co_spectrum = co_sm,
    quadrature  = qu_sm,
    coherence   = pmin(sqrt(pmax(coherence, 0)), 1),
    phase       = phase * 180 / pi  # degrees
  )
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTINUOUS WAVELET TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────

#' Continuous Wavelet Transform using WaveletComp::analyze.wavelet.
#'
#' Morlet wavelet: ψ(t) = π^{-1/4} e^{iω₀t} e^{-t²/2}
#' Default ω₀ = 6 gives good time-frequency resolution tradeoff.
#'
#' The CWT at scale s and time t is:
#'   W(s,t) = (1/√s) ∫ x(τ) ψ*((τ-t)/s) dτ
#'
#' Power: |W(s,t)|² — indicates energy at scale s and time t.
#'
#' @param x        time series (or data frame with column `value`)
#' @param dt       time step (1 = one observation)
#' @param lowerPeriod  minimum period to analyse
#' @param upperPeriod  maximum period to analyse
compute_cwt <- function(x, dt = 1,
                         lowerPeriod = 4,
                         upperPeriod = NULL,
                         dj = 1/12,  # 12 suboctaves per octave
                         date.format = NULL) {
  n <- length(x)
  if (is.null(upperPeriod)) upperPeriod <- n / 3

  # WaveletComp requires a data frame
  df <- data.frame(x = x)

  wt <- analyze.wavelet(
    my.data     = df,
    my.series   = "x",
    loess.span  = 0,  # no detrending (do it manually if needed)
    dt          = dt,
    dj          = dj,
    lowerPeriod = lowerPeriod,
    upperPeriod = upperPeriod,
    make.pval   = TRUE,
    n.sim       = 0,  # no permutation test (set > 0 for significance)
    verbose     = FALSE
  )

  cat(sprintf("CWT computed: %d periods × %d time steps\n",
              length(wt$Period), n))
  cat(sprintf("  Period range: [%.1f, %.1f] observations\n",
              min(wt$Period), max(wt$Period)))

  # Identify time-averaged dominant period
  global_power <- rowMeans(wt$Power)
  peak_period  <- wt$Period[which.max(global_power)]
  cat(sprintf("  Dominant period: %.2f observations\n", peak_period))

  return(wt)
}

#' Global wavelet spectrum: time-averaged power at each scale.
global_wavelet_spectrum <- function(wt) {
  data.frame(
    period = wt$Period,
    power  = rowMeans(wt$Power)
  )
}

# ─────────────────────────────────────────────────────────────────────────────
# WAVELET COHERENCE
# ─────────────────────────────────────────────────────────────────────────────

#' Compute wavelet coherence between two time series.
#'
#' Coherence: R²(s,t) = |S(W_{XY}(s,t))|² / (S(|W_X(s,t)|²) · S(|W_Y(s,t)|²))
#'
#' where S is a smoothing operator in time and scale.
#'
#' High coherence at (scale, time) = strong co-movement at that period during
#' that time window.
#'
#' @param x, y  aligned time series
#' @return  WaveletComp object
compute_wavelet_coherence <- function(x, y, dt = 1,
                                       lowerPeriod = 4,
                                       upperPeriod = NULL) {
  n <- length(x)
  if (is.null(upperPeriod)) upperPeriod <- n / 3

  df <- data.frame(x = x, y = y)

  wc <- analyze.coherency(
    my.data     = df,
    my.pair     = c("x", "y"),
    loess.span  = 0,
    dt          = dt,
    dj          = 1/12,
    lowerPeriod = lowerPeriod,
    upperPeriod = upperPeriod,
    make.pval   = TRUE,
    n.sim       = 0,
    verbose     = FALSE
  )

  cat("Wavelet coherence computed.\n")

  # Mean coherence
  mean_coh <- mean(wc$Coherence, na.rm = TRUE)
  cat(sprintf("  Mean coherence: %.4f\n", mean_coh))

  return(wc)
}

# ─────────────────────────────────────────────────────────────────────────────
# EVOLUTIONARY SPECTRUM
# ─────────────────────────────────────────────────────────────────────────────

#' Evolutionary (time-varying) spectrum via short-time Fourier transform.
#'
#' Sliding window DFT with Hann window:
#'   STFT(t, f) = Σ_τ w(τ - t) x(τ) e^{-2πifτ}
#'
#' Produces a spectrogram: power as function of (time, frequency).
#'
#' @param x           time series
#' @param window_size  DFT window width
#' @param hop         step size between windows
#' @param taper       fraction to taper (Hann window)
evolutionary_spectrum <- function(x, window_size = 64, hop = NULL, taper = 0.5) {
  n  <- length(x)
  if (is.null(hop)) hop <- window_size %/% 4  # 75% overlap

  n_freq <- window_size %/% 2
  freqs  <- (1:n_freq) / window_size

  t_starts <- seq(1, n - window_size + 1, by = hop)
  n_windows <- length(t_starts)

  # Hann window
  hann <- 0.5 * (1 - cos(2 * pi * (0:(window_size-1)) / (window_size - 1)))

  power_matrix <- matrix(0, n_freq, n_windows)
  t_centers    <- numeric(n_windows)

  for (k in seq_along(t_starts)) {
    idx <- t_starts[k]:(t_starts[k] + window_size - 1)
    seg <- x[idx] * hann
    X   <- fft(seg)
    power_matrix[, k] <- (Mod(X[2:(n_freq+1)])^2) / window_size
    t_centers[k]      <- t_starts[k] + window_size / 2
  }

  return(list(
    power     = power_matrix,
    frequency = freqs,
    time      = t_centers,
    period    = 1 / freqs
  ))
}

# ─────────────────────────────────────────────────────────────────────────────
# DETECT PERIODICITIES IN TRADING SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

#' Detect daily and weekly seasonality in a high-frequency signal.
#'
#' For intraday signal with dt = 1 minute:
#'   Daily period     = 390 min (NYSE session) or 1440 min
#'   Weekly period    = 5 × 390 = 1950 min
#'
#' @param signal  time series of trading signal or volume
#' @param dt      time step in minutes
#' @param session_minutes  trading session length in minutes
detect_seasonality <- function(signal, dt = 1, session_minutes = 390) {
  n <- length(signal)
  T_total <- n * dt  # total time in minutes

  cat(sprintf("Seasonality detection: n=%d, dt=%d min, T=%.1f min\n",
              n, dt, T_total))

  # Raw periodogram
  pgram <- raw_periodogram(signal, detrend = TRUE, taper = 0.1)
  pgram_sm <- smooth_periodogram_daniell(pgram, spans = c(5, 5))

  # Key periods to check
  periods_check <- c(
    session_minutes / dt,         # daily (1 session)
    5 * session_minutes / dt,     # weekly
    30 / dt,                      # 30-minute cycle
    60 / dt                       # hourly cycle
  )
  period_names <- c("Daily", "Weekly", "30-min", "Hourly")

  cat("\nSpectral power at key periods:\n")
  for (i in seq_along(periods_check)) {
    p_target <- periods_check[i]
    f_target <- 1 / p_target

    # Find closest frequency bin
    idx <- which.min(abs(pgram_sm$frequency - f_target))
    if (idx >= 1 && idx <= nrow(pgram_sm)) {
      pwr <- pgram_sm$power_smooth[idx]
      max_pwr <- max(pgram_sm$power_smooth)
      cat(sprintf("  %-10s (period=%7.1f obs): power=%.4e (%.2f%% of max)\n",
                  period_names[i], p_target, pwr, 100*pwr/max_pwr))
    }
  }

  return(list(periodogram = pgram_sm))
}

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

#' Plot raw and smoothed periodogram.
plot_periodogram <- function(pgram, title = "Periodogram",
                              log_scale = TRUE,
                              mark_periods = NULL) {
  df <- pgram
  y_col <- if ("power_smooth" %in% names(df)) "power_smooth" else "power"

  p <- ggplot(df, aes(x = frequency)) +
    geom_line(aes(y = power), color = "gray70", linewidth = 0.5, alpha = 0.8) +
    theme_minimal() +
    labs(title = title, x = "Frequency (cycles/observation)", y = "Spectral power")

  if ("power_smooth" %in% names(df)) {
    p <- p + geom_line(aes(y = power_smooth), color = "steelblue", linewidth = 1.2)
  }

  if (log_scale) {
    p <- p + scale_y_log10(labels = label_number_si())
  }

  if (!is.null(mark_periods)) {
    for (period in mark_periods) {
      p <- p + geom_vline(xintercept = 1/period, color = "red",
                           linetype = "dashed", alpha = 0.7) +
        annotate("text", x = 1/period, y = Inf, vjust = 2, hjust = -0.1,
                 label = paste0("T=", period), color = "red", size = 3)
    }
  }
  return(p)
}

#' Plot multitaper spectrum with significance markers.
plot_mtm <- function(mt_df, title = "Multitaper Spectrum", f_crit = 0.95) {
  F_crit <- qf(f_crit, 2, 2 * (nrow(mt_df) > 0) * 1)  # approximate

  p <- ggplot(mt_df, aes(x = frequency)) +
    geom_line(aes(y = power), color = "steelblue", linewidth = 1) +
    scale_y_log10() +
    labs(title = title, x = "Frequency", y = "Power (log scale)") +
    theme_minimal()

  # Mark peaks where Ftest is significant
  if ("Ftest" %in% names(mt_df)) {
    sig_pts <- mt_df %>% filter(Ftest > qf(0.95, 2, 2))
    if (nrow(sig_pts) > 0) {
      p <- p + geom_vline(data = sig_pts, aes(xintercept = frequency),
                           color = "red", alpha = 0.4, linetype = "dashed")
    }
  }
  return(p)
}

#' Plot CWT scalogram (time-frequency power).
plot_cwt_scalogram <- function(wt, title = "Wavelet Power Spectrum") {
  wt.image(wt, show.date = FALSE,
           color.key = "quantile",
           n.levels = 250,
           legend.params = list(lab = "wavelet power levels"),
           main = title,
           verbose = FALSE)
}

#' Plot wavelet coherence.
plot_wavelet_coherence <- function(wc, title = "Wavelet Coherence") {
  wc.image(wc, show.date = FALSE,
           main = title,
           which.image = "wc",
           color.key = "interval",
           n.levels = 250,
           legend.params = list(lab = "coherence levels"),
           verbose = FALSE,
           graphics.reset = FALSE)
}

#' Plot spectrogram (evolutionary spectrum).
plot_spectrogram <- function(evol_spec, title = "Spectrogram") {
  df_sp <- expand.grid(
    time      = evol_spec$time,
    frequency = evol_spec$frequency
  )
  df_sp$power <- as.vector(t(evol_spec$power))

  ggplot(df_sp, aes(x = time, y = frequency, fill = log(power + 1e-10))) +
    geom_raster() +
    scale_fill_viridis_c(name = "log Power", option = "magma") +
    labs(title = title, x = "Time (observations)", y = "Frequency") +
    theme_minimal()
}

#' Plot cross-spectrum: coherence and phase.
plot_cross_spectrum <- function(cs, title = "Cross-Spectrum") {
  p1 <- ggplot(cs, aes(x = frequency, y = coherence)) +
    geom_line(color = "steelblue", linewidth = 1) +
    geom_hline(yintercept = 0.5, color = "red", linetype = "dashed") +
    labs(title = paste(title, "— Coherence"),
         x = "Frequency", y = "Coherence K(f)") +
    ylim(0, 1) +
    theme_minimal()

  p2 <- ggplot(cs, aes(x = frequency, y = phase)) +
    geom_line(color = "purple", linewidth = 1) +
    geom_hline(yintercept = 0, color = "black", linetype = "dotted") +
    labs(title = paste(title, "— Phase"),
         x = "Frequency", y = "Phase (degrees)") +
    theme_minimal()

  grid.arrange(p1, p2, ncol = 1)
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────────────────────────────────────

demo_spectral <- function() {
  set.seed(42)
  cat("================================================================\n")
  cat("Spectral Analysis Demo: Financial Time Series\n")
  cat("================================================================\n\n")

  # Synthetic time series with known structure:
  # 1. BTC-like price with daily (T=22) and weekly (T=110) periodicities
  # 2. ETH-like with high coherence to BTC at weekly cycle

  n <- 500  # ~2 years of daily data

  # Create periodic components
  t_seq   <- 1:n
  daily   <- 0.3 * sin(2*pi*t_seq/22)       # ~monthly cycle
  weekly  <- 0.5 * sin(2*pi*t_seq/110)      # ~semi-annual
  trend   <- cumsum(rnorm(n, 0, 0.02))
  noise   <- rnorm(n, 0, 0.5)

  btc_returns <- daily + weekly + noise
  eth_returns <- 0.8 * btc_returns + 0.3 * rnorm(n, 0, 0.5)  # correlated

  cat("Synthetic series: n=%d, periods embedded at T=22 and T=110\n\n")

  # 1. Raw periodogram
  cat("1. Periodogram\n")
  cat("──────────────\n")
  pgram_raw <- raw_periodogram(btc_returns, detrend = TRUE, taper = 0.1)
  pgram_sm  <- smooth_periodogram_daniell(pgram_raw, spans = c(5, 5))

  # Find top spectral peaks
  top5 <- pgram_sm %>%
    filter(!is.nan(power_smooth)) %>%
    slice_max(power_smooth, n = 5)
  cat("Top 5 spectral peaks:\n")
  for (i in 1:nrow(top5)) {
    cat(sprintf("  freq=%.5f, period=%.2f, power=%.4e\n",
                top5$frequency[i], top5$period[i], top5$power_smooth[i]))
  }

  # 2. Bartlett
  cat("\n2. Bartlett Smoothed Periodogram (L=8)\n")
  cat("────────────────────────────────────────\n")
  pgram_bart <- bartlett_periodogram(btc_returns, L = 8)
  top5_bart  <- pgram_bart %>% slice_max(power, n = 3)
  cat("Top 3 peaks (Bartlett):\n")
  for (i in 1:nrow(top5_bart)) {
    cat(sprintf("  freq=%.5f, period=%.2f\n", top5_bart$frequency[i], top5_bart$period[i]))
  }

  # 3. Multitaper
  cat("\n3. Multitaper Spectral Estimation (NW=4)\n")
  cat("─────────────────────────────────────────\n")
  mt <- compute_mtm(btc_returns, nw = 4)
  mt_df <- mtm_to_df(mt)

  # 4. CWT
  cat("\n4. Continuous Wavelet Transform\n")
  cat("────────────────────────────────\n")
  wt_btc <- compute_cwt(btc_returns, dt = 1, lowerPeriod = 4, upperPeriod = 200)
  gws_btc <- global_wavelet_spectrum(wt_btc)
  cat(sprintf("  Global wavelet spectrum peak period: %.2f\n",
              gws_btc$period[which.max(gws_btc$power)]))

  # 5. Wavelet coherence: BTC-ETH
  cat("\n5. Wavelet Coherence: BTC-ETH\n")
  cat("─────────────────────────────\n")
  wc <- compute_wavelet_coherence(btc_returns, eth_returns,
                                   lowerPeriod = 4, upperPeriod = 200)

  # 6. Cross-spectrum
  cat("\n6. Cross-Spectrum: BTC-ETH\n")
  cat("───────────────────────────\n")
  cs <- cross_spectrum(btc_returns, eth_returns, spans = c(5, 5))
  peak_coh_freq <- cs$frequency[which.max(cs$coherence)]
  cat(sprintf("  Peak coherence at freq=%.5f (period=%.2f)\n",
              peak_coh_freq, 1/peak_coh_freq))

  max_cs_coh <- max(cs$coherence, na.rm = TRUE)
  cat(sprintf("  Maximum coherence: %.4f\n", max_cs_coh))

  # 7. Evolutionary spectrum
  cat("\n7. Evolutionary Spectrum\n")
  cat("─────────────────────────\n")
  evol <- evolutionary_spectrum(btc_returns, window_size = 64, hop = 8)
  cat(sprintf("  Spectrogram: %d freq bins × %d time windows\n",
              nrow(evol$power), ncol(evol$power)))

  # 8. Seasonality detection
  cat("\n8. Seasonality Detection\n")
  cat("─────────────────────────\n")
  # Simulate daily signal with 22-day and 5-day cycles
  signal_hf <- sin(2*pi*t_seq/22) + 0.5*sin(2*pi*t_seq/5) + rnorm(n)*0.5
  season <- detect_seasonality(signal_hf, dt = 1, session_minutes = 22)

  # Plots
  cat("\nGenerating plots...\n")

  # Periodogram plot
  p_pgram <- plot_periodogram(pgram_sm,
                               title = "BTC Periodogram (Daniell smoothed)",
                               mark_periods = c(22, 110))
  ggsave("spectral_periodogram.png", p_pgram, width = 8, height = 4)

  # MTM plot
  p_mtm <- plot_mtm(mt_df, title = "BTC Multitaper Spectrum")
  ggsave("spectral_mtm.png", p_mtm, width = 8, height = 4)

  # Cross-spectrum
  png("spectral_cross.png", width = 700, height = 600)
  plot_cross_spectrum(cs, title = "BTC-ETH Cross-Spectrum")
  dev.off()

  # CWT scalogram
  png("spectral_cwt_btc.png", width = 900, height = 500)
  plot_cwt_scalogram(wt_btc, title = "BTC Wavelet Power Spectrum")
  dev.off()

  # Wavelet coherence
  png("spectral_wc_btc_eth.png", width = 900, height = 500)
  plot_wavelet_coherence(wc, title = "BTC-ETH Wavelet Coherence")
  dev.off()

  # Spectrogram
  p_spec <- plot_spectrogram(evol, title = "BTC Evolutionary Spectrum")
  ggsave("spectral_spectrogram.png", p_spec, width = 8, height = 4)

  cat("Saved: spectral_periodogram.png, spectral_mtm.png,\n")
  cat("       spectral_cross.png, spectral_cwt_btc.png,\n")
  cat("       spectral_wc_btc_eth.png, spectral_spectrogram.png\n")
  cat("\nSpectral analysis demo complete.\n")

  return(list(pgram = pgram_sm, mt = mt, wt = wt_btc, cs = cs, wc = wc))
}

if (!interactive()) {
  demo_spectral()
}
