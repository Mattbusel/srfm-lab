# statistical_arbitrage.R
# Statistical arbitrage tools for the SRFM quantitative trading system.
# Dependencies: zoo, xts, stats
# All functions follow base R + zoo/xts conventions.

suppressPackageStartupMessages({
  library(zoo)
  library(xts)
})


# ---------------------------------------------------------------------------
# pca_factors
# ---------------------------------------------------------------------------

#' Extract PCA factors from an asset returns matrix
#'
#' @param returns_df  Matrix or data.frame (dates x assets) of returns.
#'                    Rows are observations, columns are assets.
#' @param n_factors   Integer number of principal components to extract
#'                    (default 5).
#' @return List with components:
#'   \describe{
#'     \item{factor_scores}{Matrix (n_obs x n_factors): factor realisations.}
#'     \item{loadings}{Matrix (n_assets x n_factors): factor loadings
#'           (eigenvectors).}
#'     \item{eigenvalues}{Numeric vector of length n_factors.}
#'     \item{variance_explained}{Numeric vector: fraction of variance
#'           explained by each factor.}
#'     \item{cumulative_var}{Numeric vector: cumulative variance explained.}
#'     \item{pca_object}{The raw prcomp() object for further inspection.}
#'   }
#'
#' @details
#'   The returns matrix is demeaned column-wise before PCA.  Columns (assets)
#'   with fewer than 10 non-NA observations are dropped with a warning.
#'
#' @examples
#' \dontrun{
#'   pca <- pca_factors(returns_df, n_factors = 5)
#' }
pca_factors <- function(returns_df, n_factors = 5) {
  r_mat <- as.matrix(returns_df)

  # Drop columns with too many NAs
  n_obs     <- nrow(r_mat)
  col_valid <- apply(r_mat, 2, function(x) sum(!is.na(x))) >= 10
  if (!all(col_valid)) {
    n_drop <- sum(!col_valid)
    warning(sprintf("Dropping %d asset(s) with fewer than 10 non-NA observations", n_drop))
    r_mat <- r_mat[, col_valid, drop = FALSE]
  }

  # Impute remaining NAs with column means (simple)
  col_means <- colMeans(r_mat, na.rm = TRUE)
  for (j in seq_len(ncol(r_mat))) {
    na_idx <- is.na(r_mat[, j])
    if (any(na_idx)) r_mat[na_idx, j] <- col_means[j]
  }

  n_factors <- min(n_factors, ncol(r_mat), nrow(r_mat) - 1L)
  pca <- stats::prcomp(r_mat, center = TRUE, scale. = FALSE, rank. = n_factors)

  eigs        <- pca$sdev^2
  total_var   <- sum(stats::prcomp(r_mat, center = TRUE, scale. = FALSE)$sdev^2)
  var_exp     <- eigs[seq_len(n_factors)] / total_var
  cum_var     <- cumsum(var_exp)

  list(
    factor_scores      = pca$x[, seq_len(n_factors), drop = FALSE],
    loadings           = pca$rotation[, seq_len(n_factors), drop = FALSE],
    eigenvalues        = eigs[seq_len(n_factors)],
    variance_explained = var_exp,
    cumulative_var     = cum_var,
    pca_object         = pca
  )
}


# ---------------------------------------------------------------------------
# compute_residuals
# ---------------------------------------------------------------------------

#' Compute asset-specific (idiosyncratic) returns from factor model
#'
#' @param returns      Matrix (n_obs x n_assets) of asset returns.
#' @param factor_scores Matrix (n_obs x n_factors) of factor realisations.
#' @param loadings     Matrix (n_assets x n_factors) of factor loadings.
#' @return Matrix (n_obs x n_assets) of residual (specific) returns.
#'
#' @details
#'   The factor model fitted return for asset j is:
#'     r_hat_j = factor_scores %*% loadings[j, ]
#'   The residual is r_j - r_hat_j.
#'
#' @examples
#' \dontrun{
#'   resids <- compute_residuals(returns, pca$factor_scores, pca$loadings)
#' }
compute_residuals <- function(returns, factor_scores, loadings) {
  r_mat <- as.matrix(returns)
  F_mat <- as.matrix(factor_scores)
  L_mat <- as.matrix(loadings)  # n_assets x n_factors

  if (nrow(r_mat) != nrow(F_mat)) {
    stop("returns and factor_scores must have the same number of rows (observations)")
  }
  if (ncol(r_mat) != nrow(L_mat)) {
    stop("ncol(returns) must equal nrow(loadings)")
  }
  if (ncol(F_mat) != ncol(L_mat)) {
    stop("factor_scores and loadings must have the same number of factors")
  }

  # Fitted values: (n_obs x n_factors) %*% (n_factors x n_assets) -> n_obs x n_assets
  fitted  <- F_mat %*% t(L_mat)
  resids  <- r_mat - fitted
  dimnames(resids) <- dimnames(r_mat)
  resids
}


# ---------------------------------------------------------------------------
# mean_reversion_speed
# ---------------------------------------------------------------------------

#' Estimate mean-reversion speed from residual series via AR(1)
#'
#' @param residuals Numeric vector (single asset's residuals) or matrix
#'                  (n_obs x n_assets).  If a matrix, each column is
#'                  processed independently.
#' @return If residuals is a vector: list with ar1_coef, half_life (periods),
#'         and theta.
#'         If residuals is a matrix: data.frame with one row per asset.
#'
#' @details
#'   Fits: resid_t = rho * resid_{t-1} + e_t
#'   theta     = -log(rho)  (mean-reversion speed per period)
#'   half_life = log(2) / theta
#'
#' @examples
#' \dontrun{
#'   mrs <- mean_reversion_speed(resids[, "AAPL"])
#' }
mean_reversion_speed <- function(residuals) {
  .single <- function(x) {
    x  <- x[!is.na(x)]
    if (length(x) < 5) {
      return(list(ar1_coef = NA, half_life = NA, theta = NA))
    }
    lag_x <- c(NA, x[-length(x)])
    fit   <- stats::lm(x ~ lag_x)
    rho   <- stats::coef(fit)["lag_x"]

    if (is.na(rho) || rho <= 0 || rho >= 1) {
      return(list(ar1_coef = rho, half_life = NA, theta = NA))
    }
    theta     <- -log(rho)
    half_life <- log(2) / theta
    list(ar1_coef = rho, half_life = half_life, theta = theta)
  }

  if (is.matrix(residuals) || is.data.frame(residuals)) {
    r_mat    <- as.matrix(residuals)
    n_assets <- ncol(r_mat)
    result   <- data.frame(
      asset     = colnames(r_mat),
      ar1_coef  = numeric(n_assets),
      half_life = numeric(n_assets),
      theta     = numeric(n_assets),
      stringsAsFactors = FALSE
    )
    if (is.null(result$asset)) result$asset <- paste0("A", seq_len(n_assets))

    for (j in seq_len(n_assets)) {
      res_j            <- .single(r_mat[, j])
      result$ar1_coef[j]  <- res_j$ar1_coef
      result$half_life[j] <- res_j$half_life
      result$theta[j]     <- res_j$theta
    }
    return(result)
  }

  .single(as.numeric(residuals))
}


# ---------------------------------------------------------------------------
# stat_arb_signal
# ---------------------------------------------------------------------------

#' Compute z-score based stat arb signal from residuals
#'
#' @param residuals Numeric vector or matrix (n_obs x n_assets) of specific
#'                  returns.
#' @param z_window  Integer rolling window for z-score computation (default 20).
#' @return If residuals is a vector: numeric vector of z-scores.
#'         If residuals is a matrix: matrix of z-scores (same dims).
#'
#' @details
#'   At each time t, z_t = (resid_t - mean(resid_{t-W:t})) / sd(resid_{t-W:t}).
#'   The signal is negative z: a large negative residual (asset has
#'   underperformed the factor model) implies a long signal.
#'
#' @examples
#' \dontrun{
#'   z <- stat_arb_signal(resids, z_window = 20)
#' }
stat_arb_signal <- function(residuals, z_window = 20) {
  .zscore_vec <- function(x) {
    zz    <- zoo::zoo(x)
    r_mu  <- zoo::rollapply(zz, width = z_window, FUN = mean,
                            fill = NA, align = "right", na.rm = TRUE)
    r_sd  <- zoo::rollapply(zz, width = z_window, FUN = sd,
                            fill = NA, align = "right", na.rm = TRUE)
    z_out <- rep(NA_real_, length(x))
    mask  <- !is.na(r_mu) & !is.na(r_sd) & as.numeric(r_sd) > 0
    z_out[mask] <- (x[mask] - as.numeric(r_mu)[mask]) /
                     as.numeric(r_sd)[mask]
    -z_out  # Negate: low residual -> positive (long) signal
  }

  if (is.matrix(residuals) || is.data.frame(residuals)) {
    r_mat  <- as.matrix(residuals)
    result <- apply(r_mat, 2, .zscore_vec)
    dimnames(result) <- dimnames(r_mat)
    return(result)
  }

  .zscore_vec(as.numeric(residuals))
}


# ---------------------------------------------------------------------------
# basket_construction
# ---------------------------------------------------------------------------

#' Construct long/short basket weights from PCA loadings
#'
#' @param pca_result   List returned by pca_factors().
#' @param target_factor Integer: which principal component to use (default 1).
#' @return List with components:
#'   \describe{
#'     \item{long_weights}{Named numeric vector: positive-loading assets,
#'           normalised to sum to 1.}
#'     \item{short_weights}{Named numeric vector: negative-loading assets
#'           (absolute values), normalised to sum to 1.}
#'     \item{net_weights}{Named numeric vector: full long/short weight
#'           vector (longs positive, shorts negative).}
#'     \item{n_long}{Integer: number of long positions.}
#'     \item{n_short}{Integer: number of short positions.}
#'     \item{loading_threshold}{Numeric: threshold used to filter
#'           significant loadings.}
#'   }
#'
#' @details
#'   Only loadings whose absolute value exceeds one standard deviation of
#'   all loadings for the target factor are included.  Weights are
#'   proportional to loading magnitude.
#'
#' @examples
#' \dontrun{
#'   basket <- basket_construction(pca_result, target_factor = 1)
#' }
basket_construction <- function(pca_result, target_factor = 1) {
  if (!is.list(pca_result) || is.null(pca_result$loadings)) {
    stop("pca_result must be the list returned by pca_factors()")
  }
  if (target_factor > ncol(pca_result$loadings)) {
    stop(sprintf("target_factor %d exceeds the number of factors (%d)",
                 target_factor, ncol(pca_result$loadings)))
  }

  loadings    <- pca_result$loadings[, target_factor]
  asset_names <- names(loadings)

  threshold <- sd(loadings, na.rm = TRUE)
  long_mask  <- loadings >  threshold
  short_mask <- loadings < -threshold

  long_load  <- loadings[long_mask]
  short_load <- abs(loadings[short_mask])

  norm <- function(x) {
    s <- sum(x, na.rm = TRUE)
    if (s == 0) return(x)
    x / s
  }

  long_w  <- norm(long_load)
  short_w <- norm(short_load)

  net_weights <- numeric(length(loadings))
  names(net_weights) <- asset_names
  net_weights[long_mask]  <-  long_w
  net_weights[short_mask] <- -short_w

  list(
    long_weights       = long_w,
    short_weights      = short_w,
    net_weights        = net_weights,
    n_long             = sum(long_mask),
    n_short            = sum(short_mask),
    loading_threshold  = threshold
  )
}
