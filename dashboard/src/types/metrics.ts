// ============================================================
// metrics.ts — TypeScript interfaces for observability dashboard data
// ============================================================

// ---- Connection / WebSocket ------------------------------------------

export type WsConnectionStatus = 'connecting' | 'open' | 'closed' | 'error'

// ---- Equity & P&L ---------------------------------------------------

export interface EquityPoint {
  timestamp: string       // ISO8601
  equity: number          // USD value
  drawdown: number        // decimal e.g. -0.05
  dailyPnl: number        // USD
  cumulativePnl: number
}

export interface DrawdownPoint {
  timestamp: string
  drawdown: number        // decimal (negative)
  duration: number        // bars in drawdown
}

// ---- Positions -------------------------------------------------------

export type AssetClass = 'crypto' | 'equity' | 'other'
export type PositionSide = 'long' | 'short' | 'flat'

export interface PositionSizing {
  symbol: string
  assetClass: AssetClass
  notionalUsd: number
  weight: number           // 0..1
  kellyTarget: number      // Kelly-optimal weight
  kellyGap: number         // kellyTarget - weight
  maxWeight: number        // risk limit
}

// ---- BH Signals -------------------------------------------------------

export type SignalStrength = 'strong_long' | 'weak_long' | 'neutral' | 'weak_short' | 'strong_short'
export type SpacetimeType = 'TIMELIKE' | 'SPACELIKE' | 'LIGHTLIKE' | 'NONE'

export interface BHSignal {
  symbol: string
  timeframe: '15m' | '1h' | '4h'
  strength: SignalStrength
  strengthValue: number    // -1..+1
  bhMass: number
  massThreshold: number
  spacetimeType: SpacetimeType
  lastUpdated: string      // ISO8601
  confidence: number       // 0..1
}

export interface BHMassHistory {
  symbol: string
  timestamps: string[]
  massValues: number[]
  threshold: number
}

// ---- Circuit Breaker -------------------------------------------------

export type CircuitBreakerState = 'OPEN' | 'HALF_OPEN' | 'CLOSED'

export interface CircuitBreakerStatus {
  state: CircuitBreakerState
  reason: string
  trippedAt: string | null   // ISO8601
  resetAt: string | null     // ISO8601
  consecutiveLosses: number
  dailyLossUsd: number
  dailyLossLimit: number
  drawdownPct: number
  drawdownLimit: number
}

// ---- Trade frequency heatmap -----------------------------------------

export interface TradeHeatmapCell {
  hour: number             // 0..23
  dayOfWeek: number        // 0=Mon, 6=Sun
  fillCount: number
  avgPnl: number
}

// ---- P&L Attribution -------------------------------------------------

export interface PnlSlice {
  label: string
  value: number            // USD
  pct: number              // 0..100
  assetClass: AssetClass
  color?: string
}

// ---- Greeks ----------------------------------------------------------

export interface GreeksSummary {
  delta: number
  gamma: number
  theta: number            // per day
  vega: number             // per 1% vol move
  rho: number
  netDelta: number         // portfolio-level
  notionalExposure: number
}

// ---- Risk metrics ----------------------------------------------------

export interface RiskMetrics {
  var99_1d: number         // USD
  cvar99_1d: number        // USD
  var99_10d: number        // USD
  sharpeRatio: number
  sortinoRatio: number
  calmarRatio: number
  maxDrawdown: number      // decimal
  currentDrawdown: number  // decimal
  volatilityAnn: number    // annualised decimal
  skewness: number
  excessKurtosis: number
  beta: number
  winRate: number          // decimal 0..1
  profitFactor: number
}

// ---- Portfolio snapshot ----------------------------------------------

export interface PortfolioMetrics {
  timestamp: string
  totalEquity: number
  dailyPnl: number
  dailyPnlPct: number
  weeklyPnl: number
  monthlyPnl: number
  ytdPnl: number
  marginUtilization: number   // decimal 0..1
  openPositionCount: number
  risk: RiskMetrics
}

// ---- Factor attribution ----------------------------------------------

export interface FactorContribution {
  factor: string
  contribution: number     // decimal return
  beta: number
}

export interface FactorAttribution {
  symbol: string
  timestamp: string
  totalReturn: number
  systematicReturn: number
  idiosyncraticReturn: number
  factors: FactorContribution[]
  rSquared: number
}

// ---- Correlation -----------------------------------------------------

export interface CorrelationCell {
  symbolA: string
  symbolB: string
  pearson: number
  spearman: number
  ewm: number
}

export interface CorrelationState {
  timestamp: string
  avgCorrelation: number
  isCrowding: boolean
  isStress: boolean
  regime: number
  cells: CorrelationCell[]
  topCentrality: Array<{ symbol: string; centrality: number }>
}

// ---- Stress tests ----------------------------------------------------

export interface StressTestResult {
  scenario: string
  portfolioShock: number   // USD
  portfolioShockPct: number
  worstPosition: string
  worstShock: number
}

// ---- Dashboard message payload (WebSocket) ---------------------------

export interface DashboardMetricsPayload {
  portfolio: PortfolioMetrics
  equityCurve: EquityPoint[]
  drawdown: DrawdownPoint[]
  positionSizing: PositionSizing[]
  bhSignals: BHSignal[]
  circuitBreaker: CircuitBreakerStatus
  tradeHeatmap: TradeHeatmapCell[]
  pnlBySymbol: PnlSlice[]
  pnlByAssetClass: PnlSlice[]
  greeks: GreeksSummary | null
  correlation: CorrelationState | null
  factorAttribution: FactorAttribution[]
  stressTests: StressTestResult[]
  riskMetrics: RiskMetrics
}

// ---- WebSocket message union -----------------------------------------

export type DashboardWsMessage =
  | { type: 'dashboard_update'; payload: DashboardMetricsPayload }
  | { type: 'equity_update'; payload: { point: EquityPoint } }
  | { type: 'signal_update'; payload: { signals: BHSignal[] } }
  | { type: 'circuit_breaker'; payload: CircuitBreakerStatus }
  | { type: 'risk_update'; payload: RiskMetrics }
  | { type: 'ping'; timestamp: string }
  | { type: 'pong'; timestamp: string }

// ---- Sparkline data --------------------------------------------------

export interface SparklinePoint {
  value: number
}

// ---- Metric card data ------------------------------------------------

export interface MetricCardData {
  label: string
  value: number
  unit?: string
  change24h?: number        // decimal e.g. 0.05 = +5%
  sparkline?: SparklinePoint[]
  threshold?: { warn: number; critical: number }
  higherIsBetter?: boolean
  format?: 'currency' | 'percent' | 'ratio' | 'count'
}
