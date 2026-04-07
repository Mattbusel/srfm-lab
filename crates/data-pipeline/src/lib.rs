pub mod ohlcv;
pub mod indicators;
pub mod features;
pub mod normalizer;
pub mod splitter;

// ── New extensions ────────────────────────────────────────────────────────────
pub mod bar_normalizer;
pub mod feature_extractor;
pub mod aggregator;
pub mod cross_asset;
pub mod data_quality;
pub mod timeseries_ops;
pub mod pipeline_metrics;

pub use ohlcv::{
    Bar, BarSeries, Frequency, RenkoBar,
    resample, merge, fill_gaps, FillMethod,
    returns, simple_returns, rolling_stats, renko,
};
pub use indicators::{
    sma, ema, rsi, macd, bollinger_bands, atr, adx,
    stochastic, obv, vwap_rolling, donchian_channel,
    keltner_channel, ichimoku, IchimokuPoint, heikin_ashi,
};
pub use features::{
    Matrix,
    price_features, microstructure_features, calendar_features,
    regime_features, cross_sectional_features,
};
pub use normalizer::{
    z_score_rolling, rank_normalize, rank_normalize_rolling,
    min_max_rolling, robust_scale, robust_scale_rolling,
    winsorize, standardize_columns, normalize_columns,
};
pub use splitter::{
    WalkForwardSplit, KFoldSplit, SplitPair, SplitStats,
    walk_forward_splits, purged_kfold, combinatorial_purged_cv,
    train_test_split_with_embargo, walk_forward_stats,
};

pub use bar_normalizer::{BarNormalizer, RawBar, NormalizedBar, BarHistory, NormError};
pub use feature_extractor::{
    FeatureExtractor, FeatureVector, FeatureHistory,
    FEATURE_NAMES, N_FEATURES,
    F_LOG_RETURN, F_HL_RANGE, F_RSI14, F_ATR14_PCT,
};
pub use aggregator::{BarAggregator, Tick, CompletedBar, Timeframe};
pub use cross_asset::{CrossAssetFeatures, CrossAssetFV, CrossAssetHistory, ols_beta, ewma_beta};
pub use data_quality::{
    DataQualityChecker, QualityReport, CheckResult, QualityContext,
    run_all_checks, quality_series,
};
pub use timeseries_ops::{
    resample as ts_resample, align_bars, fill_gaps as ts_fill_gaps,
    rolling_window, exponential_smoothing, seasonal_decompose,
    differencing, autocorrelation,
};
pub use pipeline_metrics::{PipelineMetrics, PipelineSnapshot};
