// ============================================================
// STRATEGY BUILDER TYPES
// ============================================================

export type NodeType =
  | 'indicator'
  | 'signal'
  | 'filter'
  | 'sizer'
  | 'entry'
  | 'exit'
  | 'output'
  | 'logic'
  | 'source'
  | 'transform'

export type NodeCategory =
  | 'Indicators'
  | 'Signals'
  | 'Filters'
  | 'Sizers'
  | 'Logic'
  | 'Sources'
  | 'Transforms'
  | 'Outputs'

export interface NodeHandle {
  id: string
  label: string
  dataType: 'number' | 'boolean' | 'series' | 'signal'
  position: 'top' | 'bottom' | 'left' | 'right'
}

export interface NodeParamDef {
  key: string
  label: string
  type: 'number' | 'string' | 'boolean' | 'select' | 'range'
  default: number | string | boolean
  min?: number
  max?: number
  step?: number
  options?: { label: string; value: string | number }[]
  description?: string
  unit?: string
}

export interface NodeDefinition {
  type: string          // unique identifier e.g. 'ema', 'rsi', 'bh_mass'
  category: NodeCategory
  label: string
  description: string
  icon?: string
  color: string         // hex color for node header
  inputs: NodeHandle[]
  outputs: NodeHandle[]
  params: NodeParamDef[]
  minInputs?: number
  maxInputs?: number
}

export interface StrategyNode {
  id: string
  type: NodeType
  definitionType: string    // maps to NodeDefinition.type
  name: string
  params: Record<string, number | string | boolean>
  position: { x: number; y: number }
  inputs: string[]          // handle ids
  outputs: string[]         // handle ids
  disabled?: boolean
  notes?: string
  lastOutput?: number | boolean | null
  error?: string
}

export interface StrategyEdge {
  id: string
  source: string            // node id
  sourceHandle: string      // handle id
  target: string            // node id
  targetHandle: string      // handle id
  animated?: boolean
  label?: string
}

export interface StrategyGraphMetadata {
  name: string
  description: string
  version: string
  createdAt: number
  updatedAt: number
  author?: string
  tags?: string[]
  symbols?: string[]        // intended symbols
  intervals?: string[]      // intended intervals
}

export interface StrategyGraph {
  id: string
  nodes: StrategyNode[]
  edges: StrategyEdge[]
  metadata: StrategyGraphMetadata
  viewport: {
    x: number
    y: number
    zoom: number
  }
}

export interface BacktestConfig {
  symbol: string
  startDate: string         // ISO date string
  endDate: string
  initialCapital: number
  commission: number        // fraction e.g. 0.001 = 0.1%
  slippage: number          // fraction
  interval: string          // '1d', '1h', etc.
  maxPositionSize?: number  // fraction of capital
  allowShort?: boolean
  reinvestDividends?: boolean
  benchmarkSymbol?: string
}

export interface BacktestTrade {
  entryTime: number
  exitTime: number
  side: 'long' | 'short'
  entryPrice: number
  exitPrice: number
  qty: number
  pnl: number
  pnlPct: number
  holdingBars: number
  entrySignal: string
  exitSignal: string
}

export interface DrawdownPeriod {
  startTime: number
  endTime: number | null
  peakEquity: number
  troughEquity: number
  drawdown: number
  drawdownPct: number
  durationDays: number
  recovered: boolean
}

export interface BacktestEquityPoint {
  time: number
  equity: number
  benchmark?: number
  drawdown: number
  drawdownPct: number
  position: number          // 0=flat, 1=long, -1=short
  signal?: number
}

export interface BacktestMetrics {
  totalReturn: number
  totalReturnPct: number
  annualizedReturn: number
  annualizedVolatility: number
  sharpe: number
  sortino: number
  calmar: number
  maxDrawdown: number
  maxDrawdownPct: number
  maxDrawdownDuration: number
  winRate: number
  numTrades: number
  numWins: number
  numLosses: number
  avgWin: number
  avgLoss: number
  profitFactor: number
  expectancy: number
  avgHoldingPeriod: number    // bars
  bestTrade: number
  worstTrade: number
  alpha?: number
  beta?: number
  benchmarkReturn?: number
  benchmarkVolatility?: number
  informationRatio?: number
  startDate: string
  endDate: string
  initialCapital: number
  finalCapital: number
  totalCommission: number
  totalSlippage: number
}

export interface BacktestResult {
  id: string
  graphId: string
  config: BacktestConfig
  metrics: BacktestMetrics
  equityCurve: BacktestEquityPoint[]
  trades: BacktestTrade[]
  drawdowns: DrawdownPeriod[]
  monthlyReturns: { year: number; month: number; return: number }[]
  runAt: number
  duration: number           // ms to run
  status: 'success' | 'error' | 'running'
  error?: string
}

// Built-in node types registry
export type IndicatorNodeType =
  | 'bh_mass'
  | 'ema'
  | 'sma'
  | 'rsi'
  | 'macd'
  | 'atr'
  | 'bollinger'
  | 'stochastic'
  | 'adx'
  | 'cci'
  | 'obv'
  | 'vwap'
  | 'pivot'
  | 'ichimoku'
  | 'supertrend'
  | 'donchian'

export type SignalNodeType =
  | 'crossover'
  | 'crossunder'
  | 'threshold_cross'
  | 'threshold_hold'
  | 'regime_match'
  | 'bh_formation'
  | 'bh_dir_change'
  | 'momentum'
  | 'reversal'
  | 'breakout'
  | 'pullback'

export type FilterNodeType =
  | 'time_filter'
  | 'volume_filter'
  | 'regime_filter'
  | 'trend_filter'
  | 'volatility_filter'
  | 'gap_filter'
  | 'spread_filter'

export type SizerNodeType =
  | 'fixed_qty'
  | 'fixed_fraction'
  | 'kelly'
  | 'vol_target'
  | 'atr_based'
  | 'equal_weight'
  | 'risk_parity'

export type LogicNodeType =
  | 'and'
  | 'or'
  | 'not'
  | 'delay'
  | 'cooldown'
  | 'consecutive'
  | 'counter'
  | 'latch'

export interface FactorExposure {
  factor: string
  loading: number
  tStat: number
  pValue: number
  significant: boolean
}

export interface StrategyFactorAnalysis {
  graphId: string
  symbol: string
  dateRange: { start: string; end: string }
  factorExposures: FactorExposure[]
  regimePerformance: {
    regime: string
    return: number
    sharpe: number
    numPeriods: number
  }[]
  sensitivityAnalysis: {
    param: string
    nodeId: string
    values: number[]
    sharpes: number[]
    returns: number[]
  }[]
}
