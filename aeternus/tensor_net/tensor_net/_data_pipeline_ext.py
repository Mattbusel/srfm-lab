"""Extension for data_pipeline.py appended programmatically."""

# ---------------------------------------------------------------------------
# Section: Advanced financial feature engineering
# ---------------------------------------------------------------------------

import numpy as np
import warnings


def compute_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Compute Relative Strength Index (RSI) for each asset.

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
        Price series.
    window : int
        RSI lookback window.

    Returns
    -------
    rsi : np.ndarray, shape (T, N)
        RSI values in [0, 100].
    """
    T, N = prices.shape
    delta = np.diff(prices, axis=0)   # (T-1, N)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    rsi = np.zeros((T, N), dtype=np.float32)

    for t in range(window, T):
        avg_gain = gain[t - window:t].mean(axis=0)
        avg_loss = loss[t - window:t].mean(axis=0)
        rs = avg_gain / (avg_loss + 1e-12)
        rsi[t] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def compute_macd(
    prices: np.ndarray,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> tuple:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
    fast_window : int
    slow_window : int
    signal_window : int

    Returns
    -------
    macd_line : np.ndarray, shape (T, N)
    signal_line : np.ndarray, shape (T, N)
    histogram : np.ndarray, shape (T, N)
    """
    T, N = prices.shape

    def ema(arr, span):
        alpha = 2.0 / (span + 1)
        result = np.zeros_like(arr)
        result[0] = arr[0]
        for t in range(1, len(arr)):
            result[t] = alpha * arr[t] + (1 - alpha) * result[t - 1]
        return result

    macd_line = np.zeros((T, N), dtype=np.float32)
    signal_line = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        fast = ema(prices[:, n], fast_window)
        slow = ema(prices[:, n], slow_window)
        macd_line[:, n] = fast - slow
        signal_line[:, n] = ema(macd_line[:, n], signal_window)

    histogram = (macd_line - signal_line).astype(np.float32)
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    prices: np.ndarray,
    window: int = 20,
    n_std: float = 2.0,
) -> tuple:
    """
    Compute Bollinger Bands.

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
    window : int
        Rolling window for mean and std.
    n_std : float
        Number of standard deviations for the bands.

    Returns
    -------
    upper : np.ndarray, shape (T, N)
    middle : np.ndarray, shape (T, N)
    lower : np.ndarray, shape (T, N)
    bandwidth : np.ndarray, shape (T, N)
        (upper - lower) / middle, normalised width.
    """
    T, N = prices.shape
    middle = np.zeros((T, N), dtype=np.float32)
    upper = np.zeros((T, N), dtype=np.float32)
    lower = np.zeros((T, N), dtype=np.float32)

    for t in range(window, T):
        window_prices = prices[t - window:t]
        mu = window_prices.mean(axis=0)
        sigma = window_prices.std(axis=0)
        middle[t] = mu
        upper[t] = mu + n_std * sigma
        lower[t] = mu - n_std * sigma

    bandwidth = (upper - lower) / (middle + 1e-12)
    return upper.astype(np.float32), middle.astype(np.float32), lower.astype(np.float32), bandwidth.astype(np.float32)


def compute_average_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    Compute Average True Range (ATR).

    Parameters
    ----------
    high, low, close : np.ndarray, shape (T, N)
    window : int

    Returns
    -------
    atr : np.ndarray, shape (T, N)
    """
    T, N = high.shape
    prev_close = np.roll(close, 1, axis=0)
    prev_close[0] = close[0]

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )

    atr = np.zeros((T, N), dtype=np.float32)
    for t in range(window, T):
        atr[t] = tr[t - window:t].mean(axis=0)

    return atr


def compute_stochastic_oscillator(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple:
    """
    Compute Stochastic Oscillator (%K and %D).

    Parameters
    ----------
    high, low, close : np.ndarray, shape (T, N)
    k_window : int
    d_window : int

    Returns
    -------
    pct_k : np.ndarray, shape (T, N)
    pct_d : np.ndarray, shape (T, N)
    """
    T, N = high.shape
    pct_k = np.zeros((T, N), dtype=np.float32)

    for t in range(k_window, T):
        h_max = high[t - k_window:t].max(axis=0)
        l_min = low[t - k_window:t].min(axis=0)
        pct_k[t] = 100.0 * (close[t] - l_min) / (h_max - l_min + 1e-12)

    # %D = simple moving average of %K
    pct_d = np.zeros((T, N), dtype=np.float32)
    for t in range(k_window + d_window, T):
        pct_d[t] = pct_k[t - d_window:t].mean(axis=0)

    return pct_k, pct_d


def compute_on_balance_volume(
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Compute On Balance Volume (OBV).

    Parameters
    ----------
    close : np.ndarray, shape (T, N)
    volume : np.ndarray, shape (T, N)

    Returns
    -------
    obv : np.ndarray, shape (T, N)
    """
    T, N = close.shape
    obv = np.zeros((T, N), dtype=np.float64)
    obv[0] = volume[0]
    for t in range(1, T):
        direction = np.sign(close[t] - close[t - 1])
        obv[t] = obv[t - 1] + direction * volume[t]
    return obv.astype(np.float32)


def compute_accumulation_distribution(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """
    Compute Accumulation/Distribution Line.

    Parameters
    ----------
    high, low, close : np.ndarray, shape (T, N)
    volume : np.ndarray, shape (T, N)

    Returns
    -------
    ad : np.ndarray, shape (T, N)
    """
    T, N = high.shape
    clv = ((close - low) - (high - close)) / (high - low + 1e-12)
    mfv = clv * volume
    ad = np.cumsum(mfv, axis=0)
    return ad.astype(np.float32)


def compute_williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14,
) -> np.ndarray:
    """
    Compute Williams %R oscillator.

    Parameters
    ----------
    high, low, close : np.ndarray, shape (T, N)
    window : int

    Returns
    -------
    wr : np.ndarray, shape (T, N)  in [-100, 0]
    """
    T, N = high.shape
    wr = np.zeros((T, N), dtype=np.float32)
    for t in range(window, T):
        h_max = high[t - window:t].max(axis=0)
        l_min = low[t - window:t].min(axis=0)
        wr[t] = -100.0 * (h_max - close[t]) / (h_max - l_min + 1e-12)
    return wr


def compute_commodity_channel_index(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 20,
    constant: float = 0.015,
) -> np.ndarray:
    """
    Compute Commodity Channel Index (CCI).

    Parameters
    ----------
    high, low, close : np.ndarray, shape (T, N)
    window : int
    constant : float

    Returns
    -------
    cci : np.ndarray, shape (T, N)
    """
    T, N = high.shape
    typical = (high + low + close) / 3.0
    cci = np.zeros((T, N), dtype=np.float32)

    for t in range(window, T):
        tp_window = typical[t - window:t]
        mean_tp = tp_window.mean(axis=0)
        mean_dev = np.abs(tp_window - mean_tp).mean(axis=0)
        cci[t] = (typical[t] - mean_tp) / (constant * mean_dev + 1e-12)

    return cci


def build_technical_indicator_tensor(
    prices: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    include_rsi: bool = True,
    include_macd: bool = True,
    include_bb: bool = True,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    bb_window: int = 20,
) -> np.ndarray:
    """
    Construct a tensor of technical indicators from OHLCV data.

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
        Close prices.
    high, low : np.ndarray, shape (T, N), optional
        High and low prices.
    volume : np.ndarray, shape (T, N), optional
        Volume data.
    include_rsi, include_macd, include_bb : bool
        Which indicators to include.

    Returns
    -------
    indicator_tensor : np.ndarray, shape (T, N, K)
        Stacked indicator features.
    """
    T, N = prices.shape
    feature_list = []

    if include_rsi:
        rsi = compute_rsi(prices, window=rsi_window)
        feature_list.append(rsi[:, :, None])

    if include_macd:
        macd_line, signal_line, hist = compute_macd(
            prices, fast_window=macd_fast, slow_window=macd_slow
        )
        feature_list.append(macd_line[:, :, None])
        feature_list.append(signal_line[:, :, None])
        feature_list.append(hist[:, :, None])

    if include_bb and high is not None and low is not None:
        upper, middle, lower, bw = compute_bollinger_bands(prices, window=bb_window)
        feature_list.append(bw[:, :, None])
    elif include_bb:
        upper, middle, lower, bw = compute_bollinger_bands(prices, window=bb_window)
        feature_list.append(bw[:, :, None])

    if not feature_list:
        return prices[:, :, None]

    return np.concatenate(feature_list, axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# Section: Multi-asset return attribution
# ---------------------------------------------------------------------------


def compute_factor_betas(
    returns: np.ndarray,
    factors: np.ndarray,
    window: int = 63,
) -> np.ndarray:
    """
    Rolling OLS factor betas.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    factors : np.ndarray, shape (T, K)
    window : int

    Returns
    -------
    betas : np.ndarray, shape (T, N, K)
        Rolling factor loadings.
    """
    T, N = returns.shape
    K = factors.shape[1]
    betas = np.zeros((T, N, K), dtype=np.float32)

    for t in range(window, T):
        F = factors[t - window:t]       # (window, K)
        R = returns[t - window:t]       # (window, N)
        # OLS: beta = (F^T F)^{-1} F^T R
        FtF = F.T @ F + 1e-6 * np.eye(K)
        FtR = F.T @ R
        beta = np.linalg.solve(FtF, FtR)  # (K, N)
        betas[t] = beta.T  # (N, K)

    return betas


def compute_idiosyncratic_returns(
    returns: np.ndarray,
    factors: np.ndarray,
    betas: np.ndarray,
) -> np.ndarray:
    """
    Compute idiosyncratic (residual) returns after removing factor exposures.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    factors : np.ndarray, shape (T, K)
    betas : np.ndarray, shape (T, N, K)

    Returns
    -------
    idio : np.ndarray, shape (T, N)
    """
    T, N, K = betas.shape
    # systematic = sum_k beta_{t,n,k} * f_{t,k}
    systematic = np.einsum("tnk,tk->tn", betas, factors)
    return (returns - systematic).astype(np.float32)


def build_factor_contribution_tensor(
    returns: np.ndarray,
    factors: np.ndarray,
    factor_names: list | None = None,
    window: int = 63,
) -> dict:
    """
    Decompose asset returns into factor contributions + idiosyncratic.

    Returns a dict with:
    * ``"betas"`` : (T, N, K)
    * ``"factor_returns"`` : (T, N, K) = betas * factor_realizations
    * ``"idiosyncratic"`` : (T, N)
    * ``"r_squared"`` : (T, N)
    * ``"factor_names"`` : list of str
    """
    T, N = returns.shape
    K = factors.shape[1]
    betas = compute_factor_betas(returns, factors, window)
    factor_rets = np.einsum("tnk,tk->tnk", betas, factors)
    idio = compute_idiosyncratic_returns(returns, factors, betas)

    # R-squared
    systematic = factor_rets.sum(axis=2)
    ss_tot = np.var(returns, axis=0, keepdims=True).repeat(T, axis=0) + 1e-12
    ss_res = idio ** 2
    r_sq = 1 - ss_res / ss_tot

    return {
        "betas": betas,
        "factor_returns": factor_rets,
        "idiosyncratic": idio,
        "r_squared": r_sq.astype(np.float32),
        "factor_names": factor_names or [f"f{k}" for k in range(K)],
    }


# ---------------------------------------------------------------------------
# Section: Regime-conditioned data splits
# ---------------------------------------------------------------------------


def split_by_regime(
    returns: np.ndarray,
    regime_labels: np.ndarray,
    n_regimes: int | None = None,
) -> dict:
    """
    Split return tensor by regime label.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    regime_labels : np.ndarray, shape (T,)
        Integer regime assignments.
    n_regimes : int, optional
        Number of regimes. If None, inferred from labels.

    Returns
    -------
    dict mapping regime_id (int) -> np.ndarray of shape (T_r, N)
    """
    if n_regimes is None:
        n_regimes = int(regime_labels.max()) + 1
    result = {}
    for k in range(n_regimes):
        mask = regime_labels == k
        if mask.any():
            result[k] = returns[mask]
    return result


def regime_conditional_statistics(
    returns: np.ndarray,
    regime_labels: np.ndarray,
) -> dict:
    """
    Compute per-regime return statistics.

    Returns dict mapping regime_id -> dict with ``mean``, ``std``,
    ``sharpe``, ``n_obs``.
    """
    splits = split_by_regime(returns, regime_labels)
    stats = {}
    for k, r in splits.items():
        mu = r.mean(axis=0)
        sigma = r.std(axis=0) + 1e-12
        stats[k] = {
            "mean": mu.astype(np.float32),
            "std": sigma.astype(np.float32),
            "sharpe": (mu / sigma * np.sqrt(252)).astype(np.float32),
            "n_obs": r.shape[0],
        }
    return stats


# ---------------------------------------------------------------------------
# Section: Tensor data augmentation — extended
# ---------------------------------------------------------------------------


def augment_time_warp(
    tensor: np.ndarray,
    warp_factor: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Time-warp augmentation: randomly stretch/compress time axis.

    Parameters
    ----------
    tensor : np.ndarray, shape (T, ...) or (T, N)
    warp_factor : float
        Max fractional change in time axis length.
    rng : np.random.Generator, optional

    Returns
    -------
    warped : np.ndarray, same shape as input
    """
    if rng is None:
        rng = np.random.default_rng()
    T = tensor.shape[0]
    new_T = int(T * (1.0 + rng.uniform(-warp_factor, warp_factor)))
    new_T = max(2, new_T)
    from scipy.interpolate import interp1d
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, new_T)
    shape = tensor.shape
    flat = tensor.reshape(T, -1)
    fn = interp1d(old_idx, flat, axis=0, kind="linear", fill_value="extrapolate")
    warped_flat = fn(new_idx)
    # Resize back to original T
    fn2 = interp1d(new_idx, warped_flat, axis=0, kind="linear", fill_value="extrapolate")
    result = fn2(old_idx)
    return result.reshape(shape).astype(np.float32)


def augment_magnitude_warp(
    tensor: np.ndarray,
    warp_std: float = 0.05,
    n_knots: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Magnitude-warp augmentation: smooth random multiplicative distortion.

    Parameters
    ----------
    tensor : np.ndarray, shape (T, N)
    warp_std : float
        Standard deviation of warp noise at knots.
    n_knots : int
        Number of cubic spline knots.
    rng : np.random.Generator, optional
    """
    if rng is None:
        rng = np.random.default_rng()
    T, N = tensor.shape
    knot_locs = np.linspace(0, T - 1, n_knots)
    warp = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        knot_vals = 1.0 + rng.normal(0, warp_std, n_knots)
        warp[:, n] = np.interp(np.arange(T), knot_locs, knot_vals)
    return (tensor * warp).astype(np.float32)


def augment_window_slice(
    tensor: np.ndarray,
    reduce_ratio: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Window-slice augmentation: extract a random sub-window and resize back.

    Parameters
    ----------
    tensor : np.ndarray, shape (T, N)
    reduce_ratio : float
        Fraction of T to sample.
    rng : np.random.Generator, optional
    """
    if rng is None:
        rng = np.random.default_rng()
    T, N = tensor.shape
    window_size = max(2, int(T * reduce_ratio))
    start = rng.integers(0, T - window_size + 1)
    sliced = tensor[start:start + window_size]
    # Resize back to T via interpolation
    old_idx = np.linspace(0, 1, window_size)
    new_idx = np.linspace(0, 1, T)
    result = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        result[:, n] = np.interp(new_idx, old_idx, sliced[:, n])
    return result


def augment_jitter_and_scale(
    tensor: np.ndarray,
    jitter_std: float = 0.001,
    scale_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Combined jitter (additive noise) and scaling augmentation.

    Parameters
    ----------
    tensor : np.ndarray, shape (T, N) or (T, N, M)
    jitter_std : float
        Standard deviation of additive noise.
    scale_std : float
        Standard deviation of multiplicative scale (applied per-asset).
    rng : np.random.Generator, optional
    """
    if rng is None:
        rng = np.random.default_rng()
    jitter = rng.normal(0, jitter_std, tensor.shape).astype(np.float32)
    scale = rng.normal(1.0, scale_std, tensor.shape[1:]).astype(np.float32)
    return ((tensor + jitter) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Section: Cross-asset tensor construction
# ---------------------------------------------------------------------------


def build_cross_asset_return_tensor(
    returns_dict: dict,
    asset_universe: list,
    T: int,
) -> np.ndarray:
    """
    Stack multiple asset return series into a unified (T, N) tensor.

    Parameters
    ----------
    returns_dict : dict mapping asset_id (str) -> np.ndarray, shape (T_i,)
    asset_universe : list of str
        Ordered list of asset IDs.
    T : int
        Target number of time steps (most recent).

    Returns
    -------
    tensor : np.ndarray, shape (T, N)
        NaN-filled where data is missing.
    """
    N = len(asset_universe)
    tensor = np.full((T, N), np.nan, dtype=np.float32)
    for n, asset_id in enumerate(asset_universe):
        if asset_id in returns_dict:
            r = np.array(returns_dict[asset_id], dtype=np.float32)
            length = min(T, len(r))
            tensor[-length:, n] = r[-length:]
    return tensor


def align_multi_frequency_returns(
    daily_returns: np.ndarray,
    weekly_returns: np.ndarray,
    monthly_returns: np.ndarray,
) -> np.ndarray:
    """
    Align daily, weekly, and monthly return series into a 3-D tensor.

    Parameters
    ----------
    daily_returns : np.ndarray, shape (T, N)
    weekly_returns : np.ndarray, shape (T//5, N)
    monthly_returns : np.ndarray, shape (T//21, N)

    Returns
    -------
    tensor : np.ndarray, shape (T, N, 3)
    """
    T, N = daily_returns.shape
    tensor = np.zeros((T, N, 3), dtype=np.float32)
    tensor[:, :, 0] = daily_returns

    # Upsample weekly to daily
    T_w = weekly_returns.shape[0]
    idx_w = np.linspace(0, T - 1, T_w)
    for n in range(N):
        tensor[:, n, 1] = np.interp(np.arange(T), idx_w, weekly_returns[:, n])

    # Upsample monthly to daily
    T_m = monthly_returns.shape[0]
    idx_m = np.linspace(0, T - 1, T_m)
    for n in range(N):
        tensor[:, n, 2] = np.interp(np.arange(T), idx_m, monthly_returns[:, n])

    return tensor


def build_sector_return_tensor(
    returns: np.ndarray,
    sector_ids: np.ndarray,
    n_sectors: int | None = None,
) -> np.ndarray:
    """
    Compute sector-average return tensor.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
    sector_ids : np.ndarray, shape (N,)
        Integer sector labels.
    n_sectors : int, optional

    Returns
    -------
    sector_returns : np.ndarray, shape (T, n_sectors)
    """
    T, N = returns.shape
    if n_sectors is None:
        n_sectors = int(sector_ids.max()) + 1
    sector_ret = np.zeros((T, n_sectors), dtype=np.float32)
    for k in range(n_sectors):
        mask = sector_ids == k
        if mask.any():
            sector_ret[:, k] = returns[:, mask].mean(axis=1)
    return sector_ret


# ---------------------------------------------------------------------------
# Section: Data quality utilities
# ---------------------------------------------------------------------------


def detect_price_jumps(
    prices: np.ndarray,
    threshold: float = 5.0,
    window: int = 20,
) -> np.ndarray:
    """
    Detect abnormal price jumps (returns > threshold * rolling std).

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
    threshold : float
        Z-score threshold for jump detection.
    window : int

    Returns
    -------
    jump_mask : np.ndarray, shape (T, N)  bool
    """
    T, N = prices.shape
    log_ret = np.diff(np.log(np.abs(prices) + 1e-12), axis=0)
    jump_mask = np.zeros((T, N), dtype=bool)

    for t in range(window, T - 1):
        hist_ret = log_ret[t - window:t]
        sigma = hist_ret.std(axis=0) + 1e-12
        z = np.abs(log_ret[t]) / sigma
        jump_mask[t + 1] = z > threshold

    return jump_mask


def fill_price_jumps(
    prices: np.ndarray,
    jump_mask: np.ndarray,
    method: str = "interpolate",
) -> np.ndarray:
    """
    Fill detected price jumps.

    Parameters
    ----------
    prices : np.ndarray, shape (T, N)
    jump_mask : np.ndarray, shape (T, N)  bool
    method : str
        "interpolate" | "prev" | "median"

    Returns
    -------
    filled : np.ndarray, shape (T, N)
    """
    T, N = prices.shape
    filled = prices.copy()

    for n in range(N):
        jump_idx = np.where(jump_mask[:, n])[0]
        if len(jump_idx) == 0:
            continue
        if method == "prev":
            for t in jump_idx:
                if t > 0:
                    filled[t, n] = filled[t - 1, n]
        elif method == "interpolate":
            for t in jump_idx:
                t_prev = t - 1 if t > 0 else 0
                t_next = t + 1 if t < T - 1 else T - 1
                filled[t, n] = (filled[t_prev, n] + filled[t_next, n]) / 2.0
        elif method == "median":
            rolling_median = np.median(prices[:, n])
            for t in jump_idx:
                filled[t, n] = rolling_median

    return filled.astype(np.float32)


def compute_data_quality_report(returns: np.ndarray) -> dict:
    """
    Comprehensive data quality report for a return tensor.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)

    Returns
    -------
    dict with quality metrics per asset and overall summary.
    """
    T, N = returns.shape
    nan_frac = np.isnan(returns).mean(axis=0)
    inf_frac = np.isinf(returns).mean(axis=0)
    mean_ret = np.nanmean(returns, axis=0)
    std_ret = np.nanstd(returns, axis=0)
    min_ret = np.nanmin(returns, axis=0)
    max_ret = np.nanmax(returns, axis=0)
    skew = (np.nanmean((returns - mean_ret) ** 3, axis=0) /
            (std_ret ** 3 + 1e-12))
    kurt = (np.nanmean((returns - mean_ret) ** 4, axis=0) /
            (std_ret ** 4 + 1e-12)) - 3

    return {
        "T": T,
        "N": N,
        "overall_nan_frac": float(nan_frac.mean()),
        "overall_inf_frac": float(inf_frac.mean()),
        "assets_with_nans": int((nan_frac > 0).sum()),
        "per_asset": {
            "nan_frac": nan_frac.tolist(),
            "inf_frac": inf_frac.tolist(),
            "mean": mean_ret.tolist(),
            "std": std_ret.tolist(),
            "min": min_ret.tolist(),
            "max": max_ret.tolist(),
            "skewness": skew.tolist(),
            "excess_kurtosis": kurt.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Section: Pipeline orchestration helpers
# ---------------------------------------------------------------------------


class DataPipelineOrchestrator:
    """
    High-level orchestrator that chains multiple pipeline steps.

    Steps are registered as callables and executed in order.

    Usage::

        orch = DataPipelineOrchestrator()
        orch.register("load", lambda: load_prices())
        orch.register("returns", lambda x: prices_to_returns(x))
        orch.register("normalise", lambda x: (x - x.mean()) / x.std())
        result = orch.run()
    """

    def __init__(self) -> None:
        self._steps: list = []   # list of (name, fn)
        self._outputs: dict = {}
        self._timings: dict = {}

    def register(self, name: str, fn, depends_on: str | None = None) -> None:
        """Register a pipeline step."""
        self._steps.append((name, fn, depends_on))

    def run(self, initial_input=None) -> dict:
        """
        Execute all registered steps in order.

        Returns dict of step_name -> output.
        """
        import time
        current = initial_input
        for name, fn, depends_on in self._steps:
            if depends_on is not None:
                inp = self._outputs.get(depends_on, current)
            else:
                inp = current
            t0 = time.monotonic()
            if inp is None:
                output = fn()
            else:
                output = fn(inp)
            self._timings[name] = time.monotonic() - t0
            self._outputs[name] = output
            current = output
        return self._outputs

    def get_output(self, step_name: str):
        """Return output of a specific step."""
        return self._outputs.get(step_name)

    def timing_report(self) -> dict:
        """Return timing info for each step."""
        return dict(self._timings)

    def reset(self) -> None:
        self._outputs = {}
        self._timings = {}


class DataPipelineCache:
    """
    Simple on-disk cache for pipeline outputs using numpy .npz format.

    Parameters
    ----------
    cache_dir : str
        Directory for cached files.
    """

    def __init__(self, cache_dir: str = "/tmp/tensor_net_cache") -> None:
        import os
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        import os
        return os.path.join(self.cache_dir, f"{key}.npz")

    def exists(self, key: str) -> bool:
        import os
        return os.path.exists(self._path(key))

    def save(self, key: str, arrays: dict) -> None:
        np.savez_compressed(self._path(key), **arrays)

    def load(self, key: str) -> dict:
        data = np.load(self._path(key))
        return dict(data)

    def delete(self, key: str) -> None:
        import os
        p = self._path(key)
        if os.path.exists(p):
            os.remove(p)

    def list_keys(self) -> list:
        import os
        return [
            f.replace(".npz", "")
            for f in os.listdir(self.cache_dir)
            if f.endswith(".npz")
        ]
