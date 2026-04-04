// ============================================================
// MARKET DATA TYPES
// ============================================================

export interface OHLCV {
  time: number          // Unix timestamp in seconds
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Trade {
  id: string
  symbol: string
  timestamp: number
  price: number
  size: number
  side: 'buy' | 'sell'
  exchange?: string
  conditions?: string[]
}

export interface Quote {
  symbol: string
  timestamp: number
  bidPrice: number
  bidSize: number
  askPrice: number
  askSize: number
  midPrice: number
  spread: number
  spreadBps: number
  lastPrice: number
  lastSize: number
  dayOpen: number
  dayHigh: number
  dayLow: number
  dayClose: number
  dayVolume: number
  dayVwap: number
  dayChange: number
  dayChangePct: number
  prevClose: number
}

export interface OrderBookLevel {
  price: number
  size: number
  total: number      // cumulative from best price
  pct: number        // percentage of total visible liquidity
  numOrders?: number
}

export interface OrderBook {
  symbol: string
  timestamp: number
  bids: OrderBookLevel[]
  asks: OrderBookLevel[]
  midPrice: number
  spread: number
  spreadBps: number
  imbalance: number  // 0-1, >0.5 means more bids
  totalBidSize: number
  totalAskSize: number
}

export interface MarketDepthSnapshot {
  symbol: string
  timestamp: number
  bids: [number, number][]  // [price, size]
  asks: [number, number][]  // [price, size]
}

export interface Bar {
  symbol: string
  interval: string
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
  vwap?: number
  numTrades?: number
}

export type Interval = '1m' | '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '1d' | '1w'

export interface SparklinePoint {
  time: number
  value: number
}

export interface InstrumentMeta {
  symbol: string
  name: string
  exchange: string
  assetClass: 'equity' | 'crypto' | 'forex' | 'futures' | 'options'
  currency: string
  marginable: boolean
  shortable: boolean
  sector?: string
  industry?: string
  marketCap?: number
}

export interface MarketSession {
  isOpen: boolean
  sessionType: 'pre' | 'regular' | 'post' | 'closed'
  nextOpen?: number
  nextClose?: number
  currentOpen?: number
  currentClose?: number
}

export interface TickData {
  symbol: string
  timestamp: number
  price: number
  size: number
  side: 'buy' | 'sell' | 'unknown'
  exchange: string
  conditions: string[]
  isLargeTrade: boolean
  sizeThreshold: number
}

export interface VolumeProfileLevel {
  price: number
  buyVolume: number
  sellVolume: number
  totalVolume: number
  pct: number          // pct of total profile volume
  isPOC: boolean       // point of control (highest volume level)
  isVAH: boolean       // value area high
  isVAL: boolean       // value area low
  isValueArea: boolean // within 70% value area
}

export interface VolumeProfile {
  symbol: string
  startTime: number
  endTime: number
  levels: VolumeProfileLevel[]
  poc: number           // point of control price
  vah: number           // value area high
  val: number           // value area low
  totalVolume: number
  valuePct: number      // 0.70 default
}

export interface MarketStat {
  symbol: string
  name: string
  price: number
  change: number
  changePct: number
  volume: number
  sparkline: SparklinePoint[]
  category: 'equity' | 'crypto' | 'bond' | 'commodity' | 'index'
}

export interface WatchlistItem {
  symbol: string
  addedAt: number
  notes?: string
  tags?: string[]
  alertPrice?: number
}

export type SortableField =
  | 'symbol'
  | 'price'
  | 'change'
  | 'changePct'
  | 'volume'
  | 'bhMass'
  | 'bhActive'
  | 'regime'

export interface SortConfig {
  field: SortableField
  direction: 'asc' | 'desc'
}

// WebSocket message types
export type WSMessageType =
  | 'quote'
  | 'trade'
  | 'bar'
  | 'orderbook'
  | 'orderbook_delta'
  | 'bh_state'
  | 'alert'
  | 'heartbeat'
  | 'error'
  | 'subscribed'
  | 'unsubscribed'

export interface WSMessage<T = unknown> {
  type: WSMessageType
  symbol?: string
  data: T
  timestamp: number
  seq?: number
}

export interface WSSubscription {
  symbols: string[]
  channels: ('quotes' | 'trades' | 'bars' | 'orderbook')[]
  interval?: Interval
}
