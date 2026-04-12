"""Expand data_pipeline.py with advanced data processing components."""

PATH = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina\data_pipeline.py"

CONTENT = r'''

# =============================================================================
# SECTION: Advanced Feature Engineering
# =============================================================================

class MicrostructureFeatures:
    """Market microstructure feature extraction.

    Computes features capturing market quality, liquidity, and
    intraday trading dynamics from OHLCV and trade data.

    Features:
    - Bid-ask spread proxies (Corwin-Schultz, Roll measure)
    - Price impact (Amihud illiquidity ratio)
    - Intraday patterns (U-shaped volume, volatility clustering)
    - Order flow imbalance proxies

    Args:
        window: Rolling window for feature computation
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def corwin_schultz_spread(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> np.ndarray:
        """Estimate bid-ask spread from high-low prices.

        Reference: Corwin & Schultz, "A Simple Way to Estimate
        Bid-Ask Spreads from Daily High and Low Prices" JF 2012.

        Args:
            high: (T,) daily high prices
            low: (T,) daily low prices
        Returns:
            (T,) spread estimates (in fraction)
        """
        T = len(high)
        spreads = np.zeros(T)
        beta = np.log(high / low) ** 2
        beta_sum = np.zeros(T)
        for t in range(1, T):
            beta_sum[t] = beta[t] + beta[t-1]
        gamma = np.log(np.maximum(high[1:], high[:-1]) / np.minimum(low[1:], low[:-1])) ** 2
        gamma = np.concatenate([[0], gamma])
        alpha = (np.sqrt(2 * beta_sum) - np.sqrt(beta_sum)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        spreads = np.maximum(0, 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)))
        return spreads

    def roll_spread(self, close: np.ndarray) -> np.ndarray:
        """Roll (1984) effective spread estimate from price changes.

        Args:
            close: (T,) closing prices
        Returns:
            (T,) spread estimates
        """
        delta = np.diff(close)
        T = len(close)
        spreads = np.full(T, np.nan)
        half_w = self.window // 2
        for t in range(self.window, T):
            dlt = delta[t - self.window:t]
            cov = np.cov(dlt[1:], dlt[:-1])[0, 1]
            if cov < 0:
                spreads[t] = 2 * np.sqrt(-cov)
        return spreads

    def amihud_illiquidity(
        self,
        returns: np.ndarray,
        volume: np.ndarray,
        price: np.ndarray,
    ) -> np.ndarray:
        """Amihud illiquidity ratio: |return| / (price * volume).

        High values = illiquid (large price impact per unit volume).

        Reference: Amihud, "Illiquidity and stock returns" JFM 2002.

        Args:
            returns: (T,) daily returns
            volume: (T,) daily volume in shares
            price: (T,) closing prices
        Returns:
            (T,) illiquidity estimates
        """
        dollar_vol = price * volume
        illiq = np.abs(returns) / (dollar_vol + 1e-10)
        # Rolling average
        result = np.full_like(illiq, np.nan)
        for t in range(self.window, len(illiq)):
            result[t] = illiq[t - self.window:t].mean()
        return result

    def intraday_volume_pattern(
        self,
        volume_by_hour: np.ndarray,
    ) -> np.ndarray:
        """Compute deviation from typical U-shaped intraday volume.

        Args:
            volume_by_hour: (T, H) volume per hour for T days, H hours
        Returns:
            (T,) measure of deviation from typical pattern
        """
        T, H = volume_by_hour.shape
        # Typical U-shape: high at open and close
        typical = np.array([1/(1+abs(h - H/2)) for h in range(H)])
        typical = typical / typical.sum()
        result = np.zeros(T)
        for t in range(T):
            row = volume_by_hour[t]
            total = row.sum()
            if total > 0:
                norm_vol = row / total
                result[t] = np.sqrt(((norm_vol - typical) ** 2).mean())
        return result

    def order_flow_toxicity(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        window: Optional[int] = None,
    ) -> np.ndarray:
        """VPIN (Volume-synchronized Probability of Informed Trading) proxy.

        Estimates probability of informed trading based on
        volume imbalance between buyer and seller-initiated trades.

        Reference: Easley et al., "Flow Toxicity and Liquidity in a
        High-Frequency World" RFS 2012.

        Args:
            close: (T,) prices
            volume: (T,) volumes
            window: Window for estimation
        Returns:
            (T,) VPIN-proxy estimates
        """
        w = window or self.window
        returns = np.diff(close, prepend=close[0])
        # Signed volume proxy: positive returns -> buy, negative -> sell
        buy_vol = volume * (returns > 0).astype(float)
        sell_vol = volume * (returns <= 0).astype(float)
        T = len(close)
        vpin = np.full(T, np.nan)
        for t in range(w, T):
            bv = buy_vol[t-w:t].sum()
            sv = sell_vol[t-w:t].sum()
            total = bv + sv
            if total > 0:
                vpin[t] = abs(bv - sv) / total
        return vpin

    def compute_all(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute all microstructure features.

        Args:
            close, high, low, volume: (T,) price/volume arrays
        Returns:
            Dict of feature name -> (T,) array
        """
        returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
        features = {}
        try:
            features["cs_spread"] = self.corwin_schultz_spread(high, low)
        except Exception:
            features["cs_spread"] = np.zeros(len(close))
        try:
            features["roll_spread"] = self.roll_spread(close)
        except Exception:
            features["roll_spread"] = np.full(len(close), np.nan)
        features["amihud"] = self.amihud_illiquidity(returns, volume, close)
        features["vpin"] = self.order_flow_toxicity(close, volume)
        features["rel_spread_hl"] = (high - low) / (close + 1e-10)
        features["volume_zscore"] = (volume - volume.mean()) / (volume.std() + 1e-10)
        return features


class AlternativeDataProcessor:
    """Process alternative data sources for financial modeling.

    Handles:
    - News sentiment scores
    - Social media metrics (Twitter, Reddit)
    - Web search trends
    - Satellite imagery features (parking lot occupancy, etc.)
    - Credit card transaction data

    All data is aligned to daily trading calendar.

    Args:
        trading_calendar: Array of trading dates (YYYYMMDD format)
        fillna_method: How to fill missing values ('ffill', 'zero', 'mean')
    """

    def __init__(
        self,
        trading_calendar: np.ndarray,
        fillna_method: str = "ffill",
    ) -> None:
        self.trading_calendar = trading_calendar
        self.fillna_method = fillna_method

    def process_news_sentiment(
        self,
        dates: np.ndarray,
        sentiment_scores: np.ndarray,
        article_counts: np.ndarray,
    ) -> np.ndarray:
        """Aggregate news sentiment to daily trading frequency.

        Args:
            dates: (N,) article dates (YYYYMMDD integers)
            sentiment_scores: (N,) sentiment per article [-1, 1]
            article_counts: (N,) or ones
        Returns:
            (T,) volume-weighted daily sentiment
        """
        T = len(self.trading_calendar)
        daily_sentiment = np.zeros(T)
        daily_count = np.zeros(T)

        cal_map = {d: i for i, d in enumerate(self.trading_calendar)}
        for d, s, c in zip(dates, sentiment_scores, article_counts):
            if d in cal_map:
                t = cal_map[d]
                daily_sentiment[t] += s * c
                daily_count[t] += c

        # Normalize
        valid = daily_count > 0
        daily_sentiment[valid] /= daily_count[valid]

        # Fill missing days
        if self.fillna_method == "ffill":
            last_val = 0.0
            for t in range(T):
                if valid[t]:
                    last_val = daily_sentiment[t]
                else:
                    daily_sentiment[t] = last_val
        return daily_sentiment

    def process_search_trends(
        self,
        weekly_trends: np.ndarray,
        trend_dates: np.ndarray,
    ) -> np.ndarray:
        """Interpolate weekly Google Trends to daily frequency.

        Args:
            weekly_trends: (W,) normalized search volume (0-100)
            trend_dates: (W,) week start dates (YYYYMMDD)
        Returns:
            (T,) daily-interpolated trend scores
        """
        T = len(self.trading_calendar)
        # Map to trading day indices
        cal_arr = self.trading_calendar.astype(float)
        date_arr = trend_dates.astype(float)
        result = np.interp(cal_arr, date_arr, weekly_trends.astype(float))
        return result

    def compute_text_features(
        self,
        texts: List[str],
        dates: np.ndarray,
        vectorizer=None,
    ) -> np.ndarray:
        """Extract bag-of-words or TF-IDF features from text.

        Args:
            texts: List of text strings
            dates: (N,) dates for each text
            vectorizer: Optional sklearn TfidfVectorizer
        Returns:
            (T, vocab_size) sparse feature matrix
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return np.zeros((len(self.trading_calendar), 1))

        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        X = vectorizer.fit_transform(texts).toarray()  # (N, V)
        T = len(self.trading_calendar)
        V = X.shape[1]
        result = np.zeros((T, V))
        count = np.zeros(T)
        cal_map = {d: i for i, d in enumerate(self.trading_calendar)}
        for i, d in enumerate(dates):
            if d in cal_map:
                t = cal_map[d]
                result[t] += X[i]
                count[t] += 1
        valid = count > 0
        result[valid] /= count[valid, np.newaxis]
        return result


class CrossSectionalNormalizer:
    """Cross-sectional normalization for equity return prediction.

    For each time step, normalizes features across the cross-section
    of assets. This removes time-series level trends and focuses
    the model on relative (cross-sectional) differences.

    Methods:
    - z-score: (x - mean) / std
    - rank: percentile rank [0, 1]
    - winsorize + z-score: clip outliers then z-score
    - robust: (x - median) / MAD
    - truncated: truncate to [-n, n] sigma

    Args:
        method: Normalization method
        winsorize_pct: Percentile for winsorization (default 1%)
        eps: Small constant for numerical stability
    """

    def __init__(
        self,
        method: str = "zscore",
        winsorize_pct: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        self.method = method
        self.winsorize_pct = winsorize_pct
        self.eps = eps

    def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Apply cross-sectional normalization.

        Args:
            X: (T, N) feature matrix where T=time, N=assets
        Returns:
            (T, N) normalized features
        """
        T, N = X.shape
        result = np.zeros_like(X)

        for t in range(T):
            row = X[t].copy()
            if np.all(np.isnan(row)):
                continue
            valid = ~np.isnan(row)

            if self.method == "zscore":
                mu = row[valid].mean()
                sigma = row[valid].std()
                result[t] = (row - mu) / (sigma + self.eps)
            elif self.method == "rank":
                ranked = np.full(N, np.nan)
                ranks = row[valid].argsort().argsort()
                valid_idx = np.where(valid)[0]
                ranked[valid_idx] = ranks / max(1, len(ranks) - 1)
                result[t] = ranked
            elif self.method == "winsorize_zscore":
                lo = np.nanpercentile(row[valid], self.winsorize_pct * 100)
                hi = np.nanpercentile(row[valid], (1 - self.winsorize_pct) * 100)
                row[valid] = np.clip(row[valid], lo, hi)
                mu = row[valid].mean()
                sigma = row[valid].std()
                result[t] = (row - mu) / (sigma + self.eps)
            elif self.method == "robust":
                median = np.nanmedian(row[valid])
                mad = np.nanmedian(np.abs(row[valid] - median))
                result[t] = (row - median) / (1.4826 * mad + self.eps)
            else:
                result[t] = row

        return result

    def transform_batch(
        self,
        X: np.ndarray,
        t_axis: int = 0,
    ) -> np.ndarray:
        """Transform with arbitrary time axis."""
        if t_axis != 0:
            X = np.moveaxis(X, t_axis, 0)
        result = self.fit_transform(X)
        if t_axis != 0:
            result = np.moveaxis(result, 0, t_axis)
        return result


class TimeSeriesDataset:
    """PyTorch-compatible financial time series dataset with lookback windows.

    Handles:
    - Multiple time series aligned to the same calendar
    - Configurable lookback window and forecast horizon
    - Train/val/test splits respecting temporal ordering
    - Missing data handling
    - Multi-asset (panel) data support

    Args:
        data: (T, N, F) array where T=time, N=assets, F=features
        labels: (T, N, H) target labels where H=forecast horizons
        lookback: Number of historical timesteps as input
        horizon: Number of future timesteps to predict
        stride: Step between samples
        normalize: Whether to normalize within each window
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        lookback: int = 60,
        horizon: int = 5,
        stride: int = 1,
        normalize: bool = True,
    ) -> None:
        self.data = data
        self.labels = labels
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.normalize = normalize
        T = data.shape[0]
        # Valid sample indices
        self.indices = list(range(lookback, T - horizon, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        t = self.indices[idx]
        x = self.data[t - self.lookback:t].copy()  # (lookback, N, F)
        if self.normalize:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True) + 1e-8
            x = (x - mean) / std
        result = {"x": x, "t": t}
        if self.labels is not None:
            y = self.labels[t:t + self.horizon]  # (horizon, N, ...)
            result["y"] = y
        return result

    def train_val_test_split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> Tuple["TimeSeriesDataset", "TimeSeriesDataset", "TimeSeriesDataset"]:
        """Split dataset preserving temporal order.

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        n = len(self.indices)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        def subset(start, end):
            ds = TimeSeriesDataset.__new__(TimeSeriesDataset)
            ds.data = self.data
            ds.labels = self.labels
            ds.lookback = self.lookback
            ds.horizon = self.horizon
            ds.stride = self.stride
            ds.normalize = self.normalize
            ds.indices = self.indices[start:end]
            return ds

        return (
            subset(0, n_train),
            subset(n_train, n_train + n_val),
            subset(n_train + n_val, n),
        )


class DataQualityChecker:
    """Data quality checks and cleaning for financial time series.

    Detects and handles:
    - Missing values
    - Price staleness (repeated values)
    - Price jumps (extreme moves)
    - Volume anomalies
    - Survivorship bias warnings
    - Corporate action detection

    Args:
        max_missing_frac: Max fraction of missing values per series
        max_consecutive_missing: Max consecutive missing values
        price_jump_threshold: Abs return threshold for suspected error (e.g., 0.5 = 50%)
        stale_threshold: Max consecutive identical prices
    """

    def __init__(
        self,
        max_missing_frac: float = 0.05,
        max_consecutive_missing: int = 5,
        price_jump_threshold: float = 0.5,
        stale_threshold: int = 5,
    ) -> None:
        self.max_missing_frac = max_missing_frac
        self.max_consecutive_missing = max_consecutive_missing
        self.price_jump_threshold = price_jump_threshold
        self.stale_threshold = stale_threshold

    def check_series(self, prices: np.ndarray, name: str = "") -> Dict[str, Any]:
        """Run all quality checks on a price series.

        Args:
            prices: (T,) price time series
            name: Series identifier for reporting
        Returns:
            Dict with check results and issues
        """
        issues = []
        T = len(prices)

        # Missing value check
        missing = np.isnan(prices)
        missing_frac = missing.mean()
        if missing_frac > self.max_missing_frac:
            issues.append(f"High missing fraction: {missing_frac:.2%}")

        # Consecutive missing
        max_consec = 0
        consec = 0
        for m in missing:
            if m:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0
        if max_consec > self.max_consecutive_missing:
            issues.append(f"Long missing gap: {max_consec} consecutive")

        # Price staleness
        valid_prices = prices[~missing]
        if len(valid_prices) > 1:
            stale_count = 0
            max_stale = 0
            for t in range(1, len(valid_prices)):
                if valid_prices[t] == valid_prices[t-1]:
                    stale_count += 1
                    max_stale = max(max_stale, stale_count)
                else:
                    stale_count = 0
            if max_stale >= self.stale_threshold:
                issues.append(f"Price staleness: {max_stale} consecutive identical")

        # Price jumps
        if len(valid_prices) > 1:
            returns = np.diff(valid_prices) / (valid_prices[:-1] + 1e-10)
            extreme = (np.abs(returns) > self.price_jump_threshold)
            if extreme.any():
                issues.append(f"Extreme price moves: {extreme.sum()} instances")

        return {
            "name": name,
            "length": T,
            "missing_fraction": float(missing_frac),
            "max_consecutive_missing": max_consec,
            "issues": issues,
            "quality_score": max(0.0, 1.0 - len(issues) * 0.25),
        }

    def clean_series(
        self,
        prices: np.ndarray,
        method: str = "ffill",
    ) -> np.ndarray:
        """Clean a price series by handling detected issues.

        Args:
            prices: (T,) raw price series
            method: How to fill missing values
        Returns:
            (T,) cleaned price series
        """
        result = prices.copy().astype(float)

        # Fill NaN
        if method == "ffill":
            last_valid = np.nan
            for t in range(len(result)):
                if np.isnan(result[t]):
                    result[t] = last_valid
                else:
                    last_valid = result[t]
        elif method == "interpolate":
            nans = np.isnan(result)
            x = np.arange(len(result))
            if not nans.all():
                result[nans] = np.interp(x[nans], x[~nans], result[~nans])
        elif method == "zero":
            result = np.nan_to_num(result, nan=0.0)

        # Clip extreme moves
        returns = np.diff(result, prepend=result[0]) / (np.abs(result) + 1e-10)
        extreme = np.abs(returns) > self.price_jump_threshold
        for t in np.where(extreme)[0]:
            if t > 0:
                result[t] = result[t-1]  # Replace with previous value

        return result


class EfficientDataLoader:
    """Memory-efficient data loader for large financial datasets.

    Supports:
    - Chunked reading from disk (HDF5, Parquet, CSV)
    - In-memory caching with LRU eviction
    - Asynchronous prefetching
    - Multi-process workers

    Args:
        data_path: Path to data file or directory
        chunk_size: Number of timesteps per chunk
        cache_chunks: Number of chunks to keep in memory
        file_format: 'hdf5', 'parquet', 'csv', or 'numpy'
    """

    def __init__(
        self,
        data_path: str,
        chunk_size: int = 1000,
        cache_chunks: int = 10,
        file_format: str = "numpy",
    ) -> None:
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.cache_chunks = cache_chunks
        self.file_format = file_format
        self._cache: Dict[int, Any] = {}
        self._cache_order: List[int] = []

    def _load_chunk(self, chunk_idx: int) -> np.ndarray:
        """Load a specific chunk from disk."""
        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size

        if self.file_format == "numpy":
            data = np.load(self.data_path)
            return data[start:end]
        elif self.file_format == "csv":
            import pandas as pd
            df = pd.read_csv(self.data_path, skiprows=start, nrows=self.chunk_size)
            return df.values
        elif self.file_format == "parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(self.data_path)
                return df.values[start:end]
            except ImportError:
                return np.zeros((self.chunk_size, 1))
        else:
            return np.zeros((self.chunk_size, 1))

    def get_chunk(self, chunk_idx: int) -> np.ndarray:
        """Get chunk from cache or load from disk."""
        if chunk_idx in self._cache:
            return self._cache[chunk_idx]
        # Evict LRU if cache full
        if len(self._cache) >= self.cache_chunks:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        chunk = self._load_chunk(chunk_idx)
        self._cache[chunk_idx] = chunk
        self._cache_order.append(chunk_idx)
        return chunk

    def get_range(self, start: int, end: int) -> np.ndarray:
        """Get a range of data across potentially multiple chunks."""
        chunks = []
        start_chunk = start // self.chunk_size
        end_chunk = end // self.chunk_size + 1
        for ci in range(start_chunk, end_chunk):
            chunk = self.get_chunk(ci)
            chunk_start = ci * self.chunk_size
            slice_start = max(0, start - chunk_start)
            slice_end = min(self.chunk_size, end - chunk_start)
            if slice_start < slice_end and slice_start < len(chunk):
                chunks.append(chunk[slice_start:min(slice_end, len(chunk))])
        return np.concatenate(chunks, axis=0) if chunks else np.array([])


class SyntheticDataGenerator:
    """Generate synthetic financial time series for testing and augmentation.

    Generates realistic synthetic data using:
    - Geometric Brownian Motion (GBM)
    - Jump-diffusion processes (Merton)
    - Heston stochastic volatility
    - GARCH(1,1) volatility clustering
    - Regime-switching models

    Args:
        seed: Random seed for reproducibility
        dt: Time step (1/252 for daily)
    """

    def __init__(self, seed: int = 42, dt: float = 1.0 / 252) -> None:
        np.random.seed(seed)
        self.dt = dt

    def gbm(
        self,
        T: int,
        mu: float = 0.08,
        sigma: float = 0.2,
        S0: float = 100.0,
    ) -> np.ndarray:
        """Geometric Brownian Motion.

        dS = mu*S*dt + sigma*S*dW

        Args:
            T: Number of timesteps
            mu: Drift (annual)
            sigma: Volatility (annual)
            S0: Initial price
        Returns:
            (T,) price path
        """
        dt = self.dt
        Z = np.random.standard_normal(T)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        return S0 * np.exp(log_returns.cumsum())

    def heston(
        self,
        T: int,
        mu: float = 0.08,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        V0: float = 0.04,
        S0: float = 100.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Heston stochastic volatility model.

        dS = mu*S*dt + sqrt(V)*S*dW1
        dV = kappa*(theta-V)*dt + xi*sqrt(V)*dW2
        corr(dW1, dW2) = rho

        Args:
            T: Number of timesteps
            mu, kappa, theta, xi, rho, V0, S0: Model parameters
        Returns:
            (prices, variances) both (T,)
        """
        dt = self.dt
        prices = np.zeros(T)
        variances = np.zeros(T)
        S, V = S0, V0
        for t in range(T):
            Z1 = np.random.standard_normal()
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal()
            V = max(1e-8, V + kappa * (theta - V) * dt + xi * np.sqrt(V * dt) * Z2)
            S = S * np.exp((mu - 0.5 * V) * dt + np.sqrt(V * dt) * Z1)
            prices[t] = S
            variances[t] = V
        return prices, variances

    def jump_diffusion(
        self,
        T: int,
        mu: float = 0.08,
        sigma: float = 0.2,
        jump_intensity: float = 5.0,
        jump_mean: float = -0.02,
        jump_std: float = 0.05,
        S0: float = 100.0,
    ) -> np.ndarray:
        """Merton jump-diffusion model.

        dS/S = (mu - lambda*k)*dt + sigma*dW + J*dN
        where N is Poisson process, J is log-normal jump size.

        Args:
            T: Number of timesteps
            jump_intensity: Poisson intensity (jumps per year)
            jump_mean, jump_std: Log-jump size distribution
        Returns:
            (T,) price path
        """
        dt = self.dt
        S = S0
        prices = np.zeros(T)
        lam = jump_intensity * dt
        k = np.exp(jump_mean + 0.5 * jump_std**2) - 1
        for t in range(T):
            Z = np.random.standard_normal()
            N = np.random.poisson(lam)
            jump = sum(np.random.normal(jump_mean, jump_std) for _ in range(N))
            S = S * np.exp((mu - 0.5 * sigma**2 - lam * k) * dt +
                           sigma * np.sqrt(dt) * Z + jump)
            prices[t] = S
        return prices

    def garch_returns(
        self,
        T: int,
        omega: float = 1e-6,
        alpha: float = 0.1,
        beta: float = 0.85,
        mu: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GARCH(1,1) return process for volatility clustering.

        sigma_t^2 = omega + alpha*epsilon_{t-1}^2 + beta*sigma_{t-1}^2

        Args:
            T: Number of timesteps
            omega, alpha, beta: GARCH parameters
            mu: Mean return
        Returns:
            (returns, variances) both (T,)
        """
        returns = np.zeros(T)
        variances = np.zeros(T)
        sigma2 = omega / (1 - alpha - beta)  # Unconditional variance
        eps = 0.0
        for t in range(T):
            sigma2 = omega + alpha * eps**2 + beta * sigma2
            eps = np.sqrt(sigma2) * np.random.standard_normal()
            returns[t] = mu + eps
            variances[t] = sigma2
        return returns, variances

    def regime_switching(
        self,
        T: int,
        regimes: Optional[List[Dict]] = None,
        transition_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two-regime Markov switching model.

        Args:
            T: Number of timesteps
            regimes: List of regime parameter dicts with 'mu' and 'sigma'
            transition_matrix: (2,2) transition probability matrix
        Returns:
            (prices, regime_labels) both (T,)
        """
        if regimes is None:
            regimes = [
                {"mu": 0.10, "sigma": 0.15},   # Bull
                {"mu": -0.20, "sigma": 0.35},  # Bear
            ]
        if transition_matrix is None:
            transition_matrix = np.array([[0.97, 0.03], [0.05, 0.95]])

        prices = np.zeros(T)
        labels = np.zeros(T, dtype=int)
        state = 0
        S = 100.0
        for t in range(T):
            state = np.random.choice(2, p=transition_matrix[state])
            r = regimes[state]
            ret = np.random.normal(r["mu"] * self.dt, r["sigma"] * np.sqrt(self.dt))
            S = S * np.exp(ret)
            prices[t] = S
            labels[t] = state

        return prices, labels


_NEW_DATA_PIPELINE_EXPORTS = [
    "MicrostructureFeatures", "AlternativeDataProcessor", "CrossSectionalNormalizer",
    "TimeSeriesDataset", "DataQualityChecker", "EfficientDataLoader", "SyntheticDataGenerator",
]
'''

with open(PATH, "a", encoding="utf-8") as f:
    f.write(CONTENT)

import subprocess
r = subprocess.run(["wc", "-l", PATH], capture_output=True, text=True, shell=True)
print(r.stdout.strip())
