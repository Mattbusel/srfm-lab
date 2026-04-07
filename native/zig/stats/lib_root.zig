//! lib_root.zig -- Library root that re-exports both stats and hurst modules.
//!
//! This file is the root_source_file for the shared library build target.
//! It pulls in all exported C-ABI symbols so the linker sees them.

pub const stats = @import("stats.zig");
pub const hurst = @import("hurst.zig");

// Re-export all C ABI symbols explicitly so they appear in the shared lib.
// stats exports
pub const stats_ewma_new         = stats.stats_ewma_new;
pub const stats_ewma_update      = stats.stats_ewma_update;
pub const stats_ewma_get         = stats.stats_ewma_get;
pub const stats_ewma_free        = stats.stats_ewma_free;
pub const stats_ewmvar_new       = stats.stats_ewmvar_new;
pub const stats_ewmvar_update    = stats.stats_ewmvar_update;
pub const stats_variance_get     = stats.stats_variance_get;
pub const stats_ewmvar_volatility = stats.stats_ewmvar_volatility;
pub const stats_ewmvar_free      = stats.stats_ewmvar_free;
pub const stats_runvar_new       = stats.stats_runvar_new;
pub const stats_runvar_update    = stats.stats_runvar_update;
pub const stats_runvar_mean      = stats.stats_runvar_mean;
pub const stats_runvar_variance  = stats.stats_runvar_variance;
pub const stats_runvar_stddev    = stats.stats_runvar_stddev;
pub const stats_runvar_skewness  = stats.stats_runvar_skewness;
pub const stats_runvar_kurtosis  = stats.stats_runvar_kurtosis;
pub const stats_runvar_free      = stats.stats_runvar_free;
pub const stats_corr_new         = stats.stats_corr_new;
pub const stats_corr_update      = stats.stats_corr_update;
pub const stats_correlation_get  = stats.stats_correlation_get;
pub const stats_corr_free        = stats.stats_corr_free;
pub const stats_linreg_new       = stats.stats_linreg_new;
pub const stats_linreg_update    = stats.stats_linreg_update;
pub const stats_linreg_slope     = stats.stats_linreg_slope;
pub const stats_linreg_predict   = stats.stats_linreg_predict;
pub const stats_linreg_free      = stats.stats_linreg_free;

// hurst exports
pub const hurst_new              = hurst.hurst_new;
pub const hurst_update           = hurst.hurst_update;
pub const hurst_get              = hurst.hurst_get;
pub const hurst_free             = hurst.hurst_free;
pub const hurst_is_trending      = hurst.hurst_is_trending;
pub const hurst_is_mean_reverting = hurst.hurst_is_mean_reverting;
pub const hurst_sample_count     = hurst.hurst_sample_count;
