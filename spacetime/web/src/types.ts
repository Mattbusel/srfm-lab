// ─── Regime ────────────────────────────────────────────────────────────────
export type Regime = 'BULL' | 'BEAR' | 'SIDEWAYS' | 'HIGH_VOL';

// ─── Live WS ───────────────────────────────────────────────────────────────
export interface InstrumentLiveState {
  bh_mass_15m: number;
  bh_mass_1h: number;
  bh_mass_1d: number;
  active_15m: boolean;
  active_1h: boolean;
  active_1d: boolean;
  price: number;
  frac: number;
  regime: Regime;
}

export interface LiveMessage {
  timestamp: string;
  equity: number;
  instruments: Record<string, InstrumentLiveState>;
}

// ─── Replay WS ─────────────────────────────────────────────────────────────
export interface ReplayBar {
  bar_idx: number;
  timestamp: string;
  price: number;
  beta: number;
  is_timelike: boolean;
  bh_mass: number;
  bh_active: boolean;
  bh_dir: number;
  ctl: number;
  regime: Regime;
  position_frac: number;
  pos_floor: number;
  equity: number;
}

// ─── Backtest ───────────────────────────────────────────────────────────────
export interface BacktestParams {
  sym: string;
  source: 'yfinance' | 'alpaca' | 'csv';
  start: string;
  end: string;
  long_only: boolean;
  params: {
    cf?: number;
    bh_form?: number;
    bh_decay?: number;
    bh_collapse?: number;
  };
}

export interface Trade {
  id: string;
  sym: string;
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  pnl_dollar: number;
  pnl_pct: number;
  hold_bars: number;
  mfe: number;
  mae: number;
  tf_score: number;
  regime: Regime;
  bh_mass_at_entry?: number;
  pos_floor?: number;
}

export interface EquityPoint {
  date: string;
  equity: number;
  drawdown?: number;
  regime?: Regime;
  benchmark?: number;
}

export interface BacktestMetrics {
  cagr: number;
  sharpe: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  avg_hold_bars: number;
  total_return: number;
}

export interface BHMassPoint {
  date: string;
  mass_15m: number;
  mass_1h: number;
  mass_1d: number;
  active: boolean;
}

export interface BacktestResult {
  run_id: string;
  sym: string;
  metrics: BacktestMetrics;
  equity_curve: EquityPoint[];
  trades: Trade[];
  bh_mass_series: BHMassPoint[];
  trades_json: string;
}

// ─── Monte Carlo ────────────────────────────────────────────────────────────
export interface MCParams {
  trades_json: string;
  n_sims: number;
  months: number;
  regime_aware: boolean;
}

export interface MCPercentiles {
  p5: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p95: number[];
  dates: string[];
}

export interface MCResult {
  percentiles: MCPercentiles;
  blowup_rate: number;
  kelly_f: number;
  drawdown_dist: number[];
  final_equity_dist: number[];
}

export interface EfficientFrontierPoint {
  vol: number;
  ret: number;
  sharpe: number;
  weights: Record<string, number>;
}

// ─── Sensitivity ────────────────────────────────────────────────────────────
export interface SensitivityParams {
  sym: string;
  source: string;
  params: Record<string, number>;
}

export interface SensitivityCell {
  param: string;
  perturbation: number;
  sharpe: number;
  cagr: number;
  max_dd: number;
}

export interface SensitivityResult {
  cells: SensitivityCell[];
  fragile_params: string[];
  robust_params: string[];
  base_sharpe: number;
}

// ─── Correlation ────────────────────────────────────────────────────────────
export interface CorrelationResult {
  instruments: string[];
  jaccard: number[][];
  pearson: number[][];
  optimal_portfolio: {
    weights: Record<string, number>;
    diversification_score: number;
    expected_correlation: number;
  };
  clusters: Array<Array<string>>;
}

// ─── Archaeology ────────────────────────────────────────────────────────────
export interface ArchaeologyResult {
  run_name: string;
  trades: Trade[];
  summary: {
    total_trades: number;
    win_rate_by_regime: Record<Regime, number>;
    win_rate_by_tfscore: Array<{ score: number; win_rate: number; count: number }>;
    pnl_distribution: number[];
  };
}

// ─── Report ─────────────────────────────────────────────────────────────────
export interface ReportRequest {
  run_names: string[];
  include_mc: boolean;
  include_sensitivity: boolean;
}

// ─── Instrument ─────────────────────────────────────────────────────────────
export interface Instrument {
  sym: string;
  name: string;
  asset_class: string;
}
