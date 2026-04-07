// ============================================================
// types/signals.ts -- TypeScript interfaces for the signal
// research and ML pipeline APIs.
//
// Mirrors research/signal_analytics/ and ml/ Python modules.
// ============================================================

// ---------------------------------------------------------------------------
// Signal metadata / lifecycle
// ---------------------------------------------------------------------------

export type SignalStatus = 'active' | 'probation' | 'retired' | 'research';
export type SignalCategory =
  | 'MOMENTUM'
  | 'MEAN_REVERSION'
  | 'VOLATILITY'
  | 'MICROSTRUCTURE'
  | 'PHYSICS'
  | 'TECHNICAL'
  | 'ONCHAIN'
  | 'MACRO';

export interface SignalMeta {
  id: string;                   // e.g. "mom_roc_20"
  name: string;                 // human-readable
  category: SignalCategory;
  status: SignalStatus;
  universe: string[];           // symbols it trades
  created_at: string;           // ISO8601
  promoted_at: string | null;
  retired_at: string | null;
  description: string;
  lookback_bars: number;
  holding_period_bars: number;
}

// ---------------------------------------------------------------------------
// IC / ICIR time series
// ---------------------------------------------------------------------------

export interface ICPoint {
  date: string;                 // ISO8601 date
  ic: number;                   // cross-sectional IC (Pearson)
  ic_spearman: number;
  ic_t_stat: number;
  p_value: number;
  n_obs: number;
}

export interface ICRollingResult {
  signal_id: string;
  window: number;               // rolling window in bars
  data: ICPoint[];
  mean_ic: number;
  icir: number;                 // IC / std(IC)
  ic_positive_rate: number;     // fraction of periods IC > 0
  t_stat: number;
  p_value: number;
}

// ---------------------------------------------------------------------------
// Alpha decay
// ---------------------------------------------------------------------------

export interface DecayPoint {
  horizon: number;              // bars ahead
  ic: number;
  ic_stderr: number;
}

export interface AlphaDecayResult {
  signal_id: string;
  category: SignalCategory;
  data: DecayPoint[];
  half_life_bars: number;
  decay_rate: number;           // lambda in IC(h) = IC0 * exp(-lambda*h)
  ic_at_zero: number;
  r_squared: number;
  optimal_holding_period: number;
  cost_adjusted_ic: number;
}

// ---------------------------------------------------------------------------
// Regime overlay
// ---------------------------------------------------------------------------

export type RegimeState = 'trending' | 'mean_reverting' | 'volatile' | 'low_vol';

export interface RegimePoint {
  date: string;
  bh_mass: number;              // black-hole mass proxy
  hurst_h: number;              // Hurst exponent (0.5 = random walk)
  regime_state: RegimeState;
  regime_probability: number;   // confidence in current regime
  volatility_regime: number;    // 1=low, 2=mid, 3=high, 4=crisis
  adx: number;                  // Average Directional Index
  trending_score: number;       // 0..1
}

export interface RegimeTimeSeries {
  symbol: string;
  data: RegimePoint[];
  current_regime: RegimeState;
  current_bh_mass: number;
  current_hurst: number;
  regime_duration_bars: number;
}

// ---------------------------------------------------------------------------
// Feature importance (ML pipeline)
// ---------------------------------------------------------------------------

export interface FeatureImportanceRow {
  feature_name: string;
  signal_category: SignalCategory;
  importance_score: number;     // SHAP / permutation importance
  rank: number;
  direction: 'positive' | 'negative' | 'mixed';
  stability: number;            // cross-fold stability 0..1
  model: string;                // which model
}

export interface FeatureImportanceResponse {
  model_id: string;
  model_type: string;           // e.g. "LightGBM", "XGBoost", "RandomForest"
  training_date: string;
  n_features: number;
  n_samples: number;
  features: FeatureImportanceRow[];
  top_categories: Array<{ category: SignalCategory; total_importance: number }>;
  out_of_sample_ic: number;
  validation_sharpe: number;
}

// ---------------------------------------------------------------------------
// Signal library full row (105+ signals)
// ---------------------------------------------------------------------------

export interface SignalLibraryRow {
  meta: SignalMeta;
  ic_rolling: ICRollingResult;
  decay: AlphaDecayResult;
  current_ic: number;
  current_icir: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  turnover_per_day: number;
  win_rate: number;
  profit_factor: number;
  last_signal_value: number;   // normalized -1..+1
  regime_conditioned_ic: Record<RegimeState, number>;
}

// ---------------------------------------------------------------------------
// Backtest equity curve
// ---------------------------------------------------------------------------

export interface BacktestEquityPoint {
  date: string;
  equity: number;
  benchmark: number;
  drawdown: number;            // negative decimal
  rolling_sharpe: number;
  rolling_vol: number;
  position_count: number;
}

export interface BacktestResult {
  signal_id: string;
  start_date: string;
  end_date: string;
  initial_equity: number;
  final_equity: number;
  total_return: number;
  annualized_return: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  max_drawdown: number;
  max_drawdown_duration_bars: number;
  win_rate: number;
  profit_factor: number;
  n_trades: number;
  equity_curve: BacktestEquityPoint[];
}

// ---------------------------------------------------------------------------
// Signal analytics API response
// ---------------------------------------------------------------------------

export interface SignalAnalyticsResponse {
  signals: SignalLibraryRow[];
  total_signals: number;
  active_count: number;
  probation_count: number;
  retired_count: number;
  feature_importance: FeatureImportanceResponse | null;
  regime: RegimeTimeSeries | null;
  top_signals_by_ic: string[];   // signal IDs
  top_signals_by_icir: string[];
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Quantile / turnover per category
// ---------------------------------------------------------------------------

export interface CategorySummary {
  category: SignalCategory;
  n_signals: number;
  mean_ic: number;
  mean_icir: number;
  mean_half_life: number;
  mean_sharpe: number;
  active_count: number;
}
