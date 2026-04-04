// ============================================================
// Core domain types for the SRFM Executive Dashboard
// ============================================================

// --- Portfolio & Positions ---

export interface Position {
  symbol: string
  side: 'long' | 'short'
  size: number          // in base asset units
  sizeUsd: number       // USD notional
  entryPrice: number
  currentPrice: number
  unrealizedPnl: number
  unrealizedPnlPct: number
  realizedPnl: number
  weight: number        // % of portfolio
  strategy: string
  openedAt: string      // ISO timestamp
  leverage: number
  stopLoss?: number
  takeProfit?: number
  liquidationPrice?: number
  margin: number
  sector: InstrumentSector
}

export type InstrumentSector = 'L1' | 'DeFi' | 'Exchange' | 'Meme' | 'Stablecoin' | 'Other'

export interface PortfolioSnapshot {
  timestamp: string
  totalEquity: number
  totalUnrealizedPnl: number
  totalRealizedPnl: number
  dailyPnl: number
  dailyPnlPct: number
  weeklyPnl: number
  monthlyPnl: number
  ytdPnl: number
  totalMarginUsed: number
  availableMargin: number
  marginUtilization: number
  winRate: number
  sharpeRatio: number
  calmarRatio: number
  sortinoRatio: number
  maxDrawdown: number
  currentDrawdown: number
  volatility: number
  beta: number
  alpha: number
}

export interface EquityPoint {
  timestamp: string
  equity: number
  drawdown: number
  dailyPnl: number
}

// --- Trades ---

export interface Trade {
  id: string
  symbol: string
  side: 'long' | 'short'
  entryPrice: number
  exitPrice: number
  size: number
  sizeUsd: number
  pnl: number
  pnlPct: number
  fees: number
  strategy: string
  entryTime: string
  exitTime: string
  durationMs: number
  regime: MarketRegime
  bhSignalAtEntry: BHState
  maxFavorableExcursion: number
  maxAdverseExcursion: number
  exitReason: 'tp' | 'sl' | 'manual' | 'signal' | 'liquidation'
  tags: string[]
}

// --- BH Signal Types ---

export type BHState = 'bullish' | 'bearish' | 'neutral'
export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w'
export type MarketRegime = 'trending_up' | 'trending_down' | 'ranging' | 'volatile'

export interface BHFormation {
  id: string
  symbol: string
  timeframe: Timeframe
  state: BHState
  mass: number           // 0-2 scale
  deltaScore: number     // -1 to 1
  entryBar: number
  startTime: string
  activeCount: number    // number of active timeframe confirmations
  confirmedTimeframes: Timeframe[]
  patternType: string
  reliability: number    // 0-1
}

export interface SignalCard {
  symbol: string
  daily: BHState
  hourly: BHState
  m15: BHState
  mass: number
  deltaScore: number
  activeFormations: number
  trend: MarketRegime
  lastUpdate: string
  price: number
  change24h: number
  change24hPct: number
  volume24h: number
}

// --- Risk ---

export interface RiskMetrics {
  var95: number          // 95% VaR in USD
  var99: number          // 99% VaR in USD
  cvar95: number         // Expected Shortfall 95%
  cvar99: number         // Expected Shortfall 99%
  grossExposure: number
  netExposure: number
  leverageRatio: number
  marginUtilization: number
  concentrationHHI: number  // Herfindahl index
  correlationRisk: number   // avg pairwise correlation
  liquidityRisk: number     // 0-1
  betaToMarket: number
  maxSinglePositionPct: number
  limitUtilization: { [limitName: string]: number }  // 0-1
}

export interface DrawdownPoint {
  timestamp: string
  equity: number
  drawdown: number
  drawdownPct: number
}

export interface CorrelationEntry {
  symbolA: string
  symbolB: string
  correlation: number
}

// --- Market Data ---

export interface CoinData {
  symbol: string
  name: string
  price: number
  change24h: number
  change24hPct: number
  volume24h: number
  marketCap: number
  allocation: number    // % in portfolio
  sector: InstrumentSector
  ath: number
  atl: number
  circulatingSupply: number
}

// --- Attribution ---

export interface AttributionEntry {
  label: string
  pnl: number
  pnlPct: number
  contribution: number  // % of total pnl
  trades: number
  winRate: number
}

export interface AttributionData {
  byInstrument: AttributionEntry[]
  byStrategy: AttributionEntry[]
  byTimeframe: AttributionEntry[]
  byRegime: AttributionEntry[]
}

// --- WebSocket ---

export interface WsMessage {
  type: 'portfolio' | 'position' | 'signal' | 'trade' | 'market' | 'risk' | 'ping'
  payload: unknown
  timestamp: string
}

// --- API Response wrappers ---

export interface ApiResponse<T> {
  data: T
  ok: boolean
  error?: string
  timestamp: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  hasMore: boolean
}
