pub mod data;
pub mod engine;
pub mod portfolio;
pub mod strategy;
pub mod analytics;
pub mod optimization;
pub mod report;

pub use data::{
    Bar, BarSeries, Tick, TickSeries, TickSide, Quote,
    MultiAssetData, FillMethod,
    parse_bars_csv, bars_to_csv, serialize_bars, deserialize_bars,
    adjust_for_splits, adjust_for_dividends, fill_missing_bars,
    validate_series, generate_synthetic_bars,
    volume_bars, dollar_bars, tick_imbalance_bars, resample_bars,
    rolling_volatility_bars, rolling_mean_bars, bollinger_bands,
    compute_rsi, compute_macd, compute_atr, compute_obv, compute_cci,
};
pub use engine::{
    BacktestEngine, BacktestConfig, BacktestMode, BacktestEvent,
    Order, OrderType, OrderStatus, TimeInForce,
    SlippageModel, CommissionModel, FillSimulator,
    StrategyCallback,
    WalkForwardRunner, ExpandingWindowRunner,
    monte_carlo_paths,
    Rebalancer, TWAPSlicer, VWAPSlicer, ExecutionAnalysis, EventLogger,
};
pub use portfolio::{
    Portfolio, Position, Trade, TradeSide, TradeStats,
    Roundtrip, extract_roundtrips,
    PortfolioRisk, compute_portfolio_risk,
    PortfolioSnapshot, CurrencyConverter, TaxLotTracker, TaxLot,
};
pub use strategy::{
    Signal, Strategy,
    SMACrossover, RSIStrategy, BollingerStrategy, MomentumStrategy,
    CompositeStrategy, CombinationMethod, RegimeFilteredStrategy,
    PositionSizer, SizingContext, RiskManager,
    signals_to_orders, ema_signal, signal_threshold, signal_discretize,
    compute_turnover,
    MeanReversionStrategy, PairsTradingStrategy, BreakoutStrategy, VolumeStrategy,
    dual_timeframe_signal,
};
pub use analytics::{
    PerformanceMetrics, compute_metrics, compute_metrics_with_benchmark,
    max_drawdown_with_duration, drawdown_series, underwater_curve, ulcer_index,
    rolling_sharpe, rolling_sortino, rolling_volatility, rolling_max_drawdown, rolling_beta,
    regime_conditional_metrics, equity_curve, monthly_returns, annual_returns,
    monte_carlo_bootstrap, monte_carlo_equity_paths,
    cpcv_splits, deflated_sharpe_ratio, probabilistic_sharpe_ratio, min_track_record_length,
    hurst_exponent, stability_of_returns, gain_to_pain, common_sense_ratio,
    drawdown_periods, format_metrics,
    lower_tail_dependence, cornish_fisher_var, expected_tail_loss,
    recovery_factor, payoff_ratio, cagr, consecutive_analysis,
    rolling_correlation, capture_ratios, risk_contribution, tracking_error,
    downside_deviation, sterling_ratio, burke_ratio, pain_index, pain_ratio,
    martin_ratio, kappa_ratio, sharpe_ratio_test,
};
pub use optimization::{
    ParamSpec, ParamSet, OptResult,
    GridSearch, RandomSearch, BayesianOptimizer,
    WalkForwardOptimizer, WFResult, walk_forward_efficiency, walk_forward_aggregate,
    SimulatedAnnealing, GeneticAlgorithm,
    ParamStability, analyze_parameter_stability, robust_param_selection,
    overfitting_probability, deflated_optimal_performance,
};
pub use report::{
    BacktestReport, text_report, html_report, json_report,
    csv_trade_list, csv_equity_curve, comparison_table, roundtrip_report,
    save_report,
};
