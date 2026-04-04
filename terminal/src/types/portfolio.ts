// ============================================================
// PORTFOLIO & ORDER MANAGEMENT TYPES
// ============================================================

export type OrderSide = 'buy' | 'sell'
export type OrderType = 'market' | 'limit' | 'stop' | 'stop_limit' | 'trailing_stop'
export type OrderStatus = 'pending' | 'accepted' | 'filled' | 'partial' | 'cancelled' | 'rejected' | 'expired'
export type TimeInForce = 'day' | 'gtc' | 'ioc' | 'fok' | 'opg' | 'cls'
export type PositionSide = 'long' | 'short'

export interface Position {
  symbol: string
  qty: number
  side: PositionSide
  entryPrice: number
  currentPrice: number
  unrealizedPnl: number
  unrealizedPnlPct: number
  realizedPnl: number
  marketValue: number
  costBasis: number
  weight: number          // as fraction of total equity
  beta?: number
  betaExposure?: number
  lastDayPrice?: number
  dayPnl?: number
  dayPnlPct?: number
  assetClass?: string
  exchange?: string
  openedAt?: number
  avgEntryDate?: number
}

export interface Order {
  id: string
  clientOrderId?: string
  symbol: string
  side: OrderSide
  qty: number
  notional?: number       // alternative to qty (dollar amount)
  type: OrderType
  status: OrderStatus
  timeInForce: TimeInForce
  price?: number          // limit price
  stopPrice?: number      // stop trigger price
  trailPrice?: number     // trailing stop amount
  trailPercent?: number   // trailing stop %
  filledQty: number
  avgFillPrice: number
  commission: number
  createdAt: number
  updatedAt: number
  submittedAt?: number
  filledAt?: number
  cancelledAt?: number
  expiredAt?: number
  extendedHours: boolean
  legs?: OrderLeg[]
  source?: 'manual' | 'algo' | 'backtest'
  strategyId?: string
  notes?: string
}

export interface OrderLeg {
  id: string
  symbol: string
  side: OrderSide
  qty: number
  filledQty: number
  price?: number
  type: OrderType
}

export interface OrderRequest {
  symbol: string
  side: OrderSide
  qty?: number
  notional?: number
  type: OrderType
  timeInForce: TimeInForce
  price?: number
  stopPrice?: number
  trailPrice?: number
  trailPercent?: number
  extendedHours?: boolean
  clientOrderId?: string
  notes?: string
}

export interface AccountState {
  id: string
  equity: number
  cash: number
  buyingPower: number
  portfolioValue: number
  longMarketValue: number
  shortMarketValue: number
  marginUsed: number
  marginAvailable: number
  initialMargin: number
  maintenanceMargin: number
  dayPnl: number
  dayPnlPct: number
  totalPnl: number
  totalPnlPct: number
  positions: Position[]
  orders: Order[]
  openOrderCount: number
  daytradingBuyingPower: number
  regTBuyingPower: number
  currency: string
  tradingBlocked: boolean
  accountBlocked: boolean
  patternDayTrader: boolean
  daytradingCount?: number
  lastEquityCheck?: number
}

export interface HistoricalTrade {
  id: string
  orderId: string
  symbol: string
  side: OrderSide
  qty: number
  price: number
  commission: number
  pnl: number
  pnlPct: number
  holdingPeriod: number  // seconds
  entryTime: number
  exitTime: number
  entryPrice: number
  exitPrice: number
  strategyId?: string
  tags?: string[]
}

export interface EquityPoint {
  timestamp: number
  equity: number
  cash: number
  longValue: number
  shortValue: number
  dayPnl: number
  totalPnl: number
}

export interface RiskMetrics {
  portfolioVar95: number
  portfolioVar99: number
  cVar95: number
  maxConcentration: number
  maxConcentrationSymbol: string
  beta: number
  correlation: number       // to SPY
  sharpe: number
  sortino: number
  maxDrawdown: number
  currentDrawdown: number
  volatility: number        // annualized
  winRate: number
  avgWin: number
  avgLoss: number
  profitFactor: number
  expectancy: number
}

export interface PortfolioAnalytics {
  totalReturn: number
  totalReturnPct: number
  annualizedReturn: number
  volatility: number
  sharpe: number
  sortino: number
  calmar: number
  maxDrawdown: number
  maxDrawdownDuration: number  // days
  winRate: number
  avgHoldingPeriod: number     // hours
  totalTrades: number
  profitFactor: number
  expectancy: number
  bestTrade: HistoricalTrade | null
  worstTrade: HistoricalTrade | null
  bySymbol: Record<string, SymbolAnalytics>
  byMonth: MonthlyReturn[]
  byDay: DailyReturn[]
}

export interface SymbolAnalytics {
  symbol: string
  totalPnl: number
  totalPnlPct: number
  trades: number
  winRate: number
  avgHold: number
  avgWin: number
  avgLoss: number
}

export interface MonthlyReturn {
  year: number
  month: number        // 1-12
  return: number
  returnPct: number
  trades: number
}

export interface DailyReturn {
  date: string
  return: number
  returnPct: number
  trades: number
}

export interface OrderFill {
  orderId: string
  fillId: string
  symbol: string
  side: OrderSide
  qty: number
  price: number
  commission: number
  timestamp: number
  exchange: string
  liquidity: 'maker' | 'taker' | 'unknown'
}

export interface PositionTarget {
  symbol: string
  targetQty: number
  targetWeight: number
  currentQty: number
  currentWeight: number
  diffQty: number
  requiredOrder?: Partial<OrderRequest>
}

export interface DailyPnlTarget {
  target: number
  targetPct: number          // of equity
  achieved: number
  achievedPct: number
  remaining: number
  onTrack: boolean
}
