// ============================================================
// types/risk.ts -- TypeScript interfaces for the FastAPI risk
// service running on port 8791.
//
// Mirrors execution/risk/risk_api.py, live_var.py, attribution.py,
// limits.py, and correlation_monitor.py.
// ============================================================

// ---------------------------------------------------------------------------
// VaR / CVaR
// ---------------------------------------------------------------------------

export type VaRMethod = 'parametric' | 'historical' | 'monte_carlo';

export interface VaRPoint {
  timestamp: string;        // ISO8601
  var_95: number;           // USD -- positive = loss
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  method: VaRMethod;
}

export interface VaRSummary {
  parametric: VaRPoint;
  historical: VaRPoint;
  monte_carlo: VaRPoint;
  consensus_var99: number;  // USD
  equity: number;
  n_positions: number;
  breach_flag: boolean;
  timestamp: string;
}

export interface VaRTrendPoint {
  timestamp: string;
  parametric_var99: number;
  historical_var99: number;
  mc_var99: number;
  consensus_var99: number;
  equity: number;
}

// ---------------------------------------------------------------------------
// Positions
// ---------------------------------------------------------------------------

export type AssetClass = 'crypto' | 'equity' | 'option' | 'futures' | 'other';
export type PositionSide = 'long' | 'short';

export interface OptionGreeks {
  delta: number;
  gamma: number;
  vega: number;             // USD per 1% IV move
  theta: number;            // USD per day
  rho: number;
  vanna: number;
  charm: number;
}

export interface PositionRow {
  symbol: string;
  asset_class: AssetClass;
  side: PositionSide;
  qty: number;
  entry_price: number;
  current_price: number;
  notional_usd: number;
  weight: number;           // fraction of equity (signed)
  unrealized_pnl: number;   // USD
  realized_pnl: number;     // USD today
  delta_dollars: number;    // USD delta exposure
  greeks: OptionGreeks | null;
  var_contribution: number; // fractional contribution to portfolio VaR
  implied_vol: number | null;
  expiry: string | null;    // ISO8601 for options
  strike: number | null;
}

export interface PortfolioSummary {
  equity: number;
  gross_exposure: number;
  net_exposure: number;
  leverage: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  n_positions: number;
  margin_utilization: number;
}

// ---------------------------------------------------------------------------
// Greeks aggregate
// ---------------------------------------------------------------------------

export interface GreeksSummaryRow {
  symbol: string;
  expiry: string;
  strike: number;
  option_type: 'call' | 'put';
  qty: number;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
  notional_usd: number;
  iv: number;
  underlying_price: number;
}

export interface GreeksAggregate {
  net_delta: number;
  net_gamma: number;
  net_vega: number;
  net_theta: number;        // USD/day
  net_rho: number;
  dollar_delta: number;
  dollar_gamma: number;     // USD per 1% move^2
  dollar_vega: number;      // USD per 1% IV
  positions: GreeksSummaryRow[];
}

// ---------------------------------------------------------------------------
// Correlation
// ---------------------------------------------------------------------------

export interface CorrelationMatrixResponse {
  symbols: string[];
  pearson: number[][];      // NxN matrix
  spearman: number[][];
  ewm: number[][];          // EWMA correlation (lambda=0.94)
  avg_correlation: number;
  is_crowding: boolean;
  is_stress: boolean;
  regime: number;           // 0=normal, 1=crowding, 2=stress
  timestamp: string;
  top_centrality: Array<{ symbol: string; centrality: number }>;
}

// ---------------------------------------------------------------------------
// Circuit breakers / API health
// ---------------------------------------------------------------------------

export type CircuitBreakerState = 'CLOSED' | 'HALF_OPEN' | 'OPEN';
export type BrokerName = 'Alpaca' | 'Binance' | 'Polygon';

export interface BrokerCircuitBreaker {
  broker: BrokerName;
  state: CircuitBreakerState;
  reason: string;
  tripped_at: string | null;
  reset_at: string | null;
  consecutive_errors: number;
  last_latency_ms: number;
  error_rate_1m: number;    // errors per minute
}

export interface ApiHealthResponse {
  service: string;
  status: 'healthy' | 'degraded' | 'down';
  uptime_seconds: number;
  last_update: string;
  brokers: BrokerCircuitBreaker[];
}

// ---------------------------------------------------------------------------
// Limits / breaches
// ---------------------------------------------------------------------------

export type LimitStatus = 'ok' | 'warn' | 'breach';

export interface LimitRow {
  name: string;
  current_value: number;
  limit_value: number;
  utilization: number;      // 0..1
  status: LimitStatus;
  unit: string;
  last_checked: string;
}

export interface LimitsResponse {
  limits: LimitRow[];
  any_breach: boolean;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// VaR breach alerts
// ---------------------------------------------------------------------------

export type AlertSeverity = 'info' | 'warn' | 'critical';

export interface VaRBreachAlert {
  id: string;
  severity: AlertSeverity;
  message: string;
  value: number;
  threshold: number;
  method: VaRMethod;
  timestamp: string;
  acknowledged: boolean;
}

// ---------------------------------------------------------------------------
// P&L attribution
// ---------------------------------------------------------------------------

export interface AttributionFactor {
  name: string;
  contribution_usd: number;
  contribution_pct: number;
  beta: number;
  r_squared: number;
}

export interface AttributionRow {
  symbol: string;
  total_pnl: number;
  systematic_pnl: number;
  idiosyncratic_pnl: number;
  factors: AttributionFactor[];
  residual: number;
  holding_period_bars: number;
}

export interface AttributionResponse {
  date_range: { start: string; end: string };
  total_pnl: number;
  systematic_pnl: number;
  idiosyncratic_pnl: number;
  positions: AttributionRow[];
  factor_summary: AttributionFactor[];
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Full portfolio response shape (GET /risk/portfolio)
// ---------------------------------------------------------------------------

export interface PortfolioRiskResponse {
  summary: PortfolioSummary;
  var_summary: VaRSummary;
  var_trend: VaRTrendPoint[];     // last 30 data points
  positions: PositionRow[];
  greeks: GreeksAggregate | null;
  limits: LimitsResponse;
  alerts: VaRBreachAlert[];
  correlation: CorrelationMatrixResponse | null;
  health: ApiHealthResponse;
  attribution: AttributionResponse | null;
  timestamp: string;
}
