pub mod ohlcv;
pub mod indicators;
pub mod features;
pub mod normalizer;
pub mod splitter;

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
