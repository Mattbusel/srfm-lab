export interface Trade {
  id: string
  instrument: string
  side: 'long' | 'short'
  entryTime: string
  exitTime: string | null
  entryPrice: number
  exitPrice: number | null
  quantity: number
  pnl: number
  pnlPct: number
  commission: number
  slippage: number
  regime: RegimeType
  signalStrength: number
  holdingPeriodHours: number
  mae: number   // max adverse excursion
  mfe: number   // max favorable excursion
  strategy: string
  foldId?: number
}

export type RegimeType = 'bull' | 'bear' | 'sideways' | 'ranging' | 'volatile'

export interface TradeFilter {
  instrument?: string
  regime?: RegimeType
  dateFrom?: string
  dateTo?: string
  strategy?: string
  side?: 'long' | 'short'
}

export interface EquityPoint {
  timestamp: string
  equity: number
  drawdown: number
  benchmark?: number
}

export interface PerformanceMetrics {
  totalPnl: number
  totalPnlPct: number
  sharpeRatio: number
  sortinoRatio: number
  maxDrawdown: number
  maxDrawdownPct: number
  winRate: number
  profitFactor: number
  avgWin: number
  avgLoss: number
  avgHoldingHours: number
  totalTrades: number
  totalWins: number
  totalLosses: number
  calmarRatio: number
  annualizedReturn: number
  annualizedVolatility: number
  skewness: number
  kurtosis: number
  var95: number
  cvar95: number
}

export interface ReconciliationRow {
  instrument: string
  regime: RegimeType
  liveEntryPrice: number
  backtestEntryPrice: number
  liveExitPrice: number | null
  backtestExitPrice: number | null
  livePnl: number
  backtestPnl: number
  pnlDiff: number
  slippage: number
  signalDrift: number
  tradeDate: string
  notes: string
}

export interface SlippageStats {
  instrument: string
  avgSlippage: number
  medianSlippage: number
  p95Slippage: number
  worstSlippage: number
  slippagePct: number
  count: number
}
