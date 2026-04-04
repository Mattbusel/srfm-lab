// ============================================================
// MARKET DATA STORE — Zustand + Immer
// ============================================================
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { subscribeWithSelector } from 'zustand/middleware'
import type {
  Quote,
  OrderBook,
  OHLCV,
  Interval,
  WatchlistItem,
  SortConfig,
  SparklinePoint,
  Trade,
  VolumeProfile,
  MarketSession,
} from '@/types'

const DEFAULT_WATCHLIST = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMZN', 'MSFT', 'BTC/USD', 'ETH/USD']
const MAX_BARS = 500
const MAX_TRADES = 200

interface PriceFlash {
  symbol: string
  direction: 'up' | 'down'
  timestamp: number
}

interface MarketState {
  // ---- Data ----
  quotes: Record<string, Quote>
  orderBooks: Record<string, OrderBook>
  recentBars: Record<string, OHLCV[]>          // last MAX_BARS per symbol
  recentTrades: Record<string, Trade[]>        // last MAX_TRADES per symbol
  sparklines: Record<string, SparklinePoint[]>  // 7d sparkline per symbol
  volumeProfiles: Record<string, VolumeProfile>
  marketSession: MarketSession | null

  // ---- Watchlist ----
  watchlist: WatchlistItem[]
  watchlistSort: SortConfig

  // ---- Selection ----
  selectedSymbol: string
  selectedInterval: Interval
  hoveredPrice: number | null
  priceFlashes: Record<string, PriceFlash>

  // ---- Connection ----
  isConnected: boolean
  lastHeartbeat: number | null
  subscriptions: Set<string>
  errorMessage: string | null
}

interface MarketActions {
  // Quotes
  updateQuote(quote: Quote): void
  batchUpdateQuotes(quotes: Quote[]): void

  // Order Books
  updateOrderBook(book: OrderBook): void
  applyOrderBookDelta(symbol: string, bids: [number, number][], asks: [number, number][]): void

  // Bars
  addBar(symbol: string, bar: OHLCV): void
  setBars(symbol: string, bars: OHLCV[]): void

  // Trades
  addTrade(symbol: string, trade: Trade): void

  // Sparklines
  setSparkline(symbol: string, points: SparklinePoint[]): void

  // Volume Profile
  setVolumeProfile(symbol: string, profile: VolumeProfile): void

  // Watchlist
  addToWatchlist(symbol: string): void
  removeFromWatchlist(symbol: string): void
  reorderWatchlist(symbols: string[]): void
  setWatchlistSort(sort: SortConfig): void

  // Selection
  setSelectedSymbol(symbol: string): void
  setSelectedInterval(interval: Interval): void
  setHoveredPrice(price: number | null): void

  // Session
  setMarketSession(session: MarketSession): void

  // Connection
  setConnected(connected: boolean): void
  setLastHeartbeat(ts: number): void
  addSubscription(channel: string): void
  removeSubscription(channel: string): void
  setError(msg: string | null): void
  clearPriceFlash(symbol: string): void
}

type MarketStore = MarketState & MarketActions

export const useMarketStore = create<MarketStore>()(
  subscribeWithSelector(
    immer((set, _get) => ({
      // ---- Initial State ----
      quotes: {},
      orderBooks: {},
      recentBars: {},
      recentTrades: {},
      sparklines: {},
      volumeProfiles: {},
      marketSession: null,
      watchlist: DEFAULT_WATCHLIST.map((symbol, i) => ({
        symbol,
        addedAt: Date.now() - i * 1000,
      })),
      watchlistSort: { field: 'symbol', direction: 'asc' },
      selectedSymbol: 'SPY',
      selectedInterval: '1d',
      hoveredPrice: null,
      priceFlashes: {},
      isConnected: false,
      lastHeartbeat: null,
      subscriptions: new Set(),
      errorMessage: null,

      // ---- Quote Actions ----
      updateQuote(quote: Quote) {
        set((state) => {
          const prev = state.quotes[quote.symbol]
          state.quotes[quote.symbol] = quote

          // Price flash
          if (prev && prev.lastPrice !== quote.lastPrice) {
            state.priceFlashes[quote.symbol] = {
              symbol: quote.symbol,
              direction: quote.lastPrice > prev.lastPrice ? 'up' : 'down',
              timestamp: Date.now(),
            }
          }
        })
      },

      batchUpdateQuotes(quotes: Quote[]) {
        set((state) => {
          for (const quote of quotes) {
            const prev = state.quotes[quote.symbol]
            state.quotes[quote.symbol] = quote
            if (prev && prev.lastPrice !== quote.lastPrice) {
              state.priceFlashes[quote.symbol] = {
                symbol: quote.symbol,
                direction: quote.lastPrice > prev.lastPrice ? 'up' : 'down',
                timestamp: Date.now(),
              }
            }
          }
        })
      },

      clearPriceFlash(symbol: string) {
        set((state) => {
          delete state.priceFlashes[symbol]
        })
      },

      // ---- Order Book Actions ----
      updateOrderBook(book: OrderBook) {
        set((state) => {
          state.orderBooks[book.symbol] = book
        })
      },

      applyOrderBookDelta(symbol: string, bids: [number, number][], asks: [number, number][]) {
        set((state) => {
          const book = state.orderBooks[symbol]
          if (!book) return

          // Apply bid deltas
          for (const [price, size] of bids) {
            const idx = book.bids.findIndex((l) => l.price === price)
            if (size === 0) {
              if (idx !== -1) book.bids.splice(idx, 1)
            } else if (idx !== -1) {
              book.bids[idx].size = size
            } else {
              book.bids.push({ price, size, total: 0, pct: 0 })
              book.bids.sort((a, b) => b.price - a.price)
            }
          }

          // Apply ask deltas
          for (const [price, size] of asks) {
            const idx = book.asks.findIndex((l) => l.price === price)
            if (size === 0) {
              if (idx !== -1) book.asks.splice(idx, 1)
            } else if (idx !== -1) {
              book.asks[idx].size = size
            } else {
              book.asks.push({ price, size, total: 0, pct: 0 })
              book.asks.sort((a, b) => a.price - b.price)
            }
          }

          // Recompute cumulative totals
          let bidTotal = 0
          for (const level of book.bids) {
            bidTotal += level.size
            level.total = bidTotal
          }
          let askTotal = 0
          for (const level of book.asks) {
            askTotal += level.size
            level.total = askTotal
          }

          const totalSize = bidTotal + askTotal
          for (const level of book.bids) level.pct = totalSize > 0 ? level.size / totalSize : 0
          for (const level of book.asks) level.pct = totalSize > 0 ? level.size / totalSize : 0

          book.totalBidSize = bidTotal
          book.totalAskSize = askTotal
          book.imbalance = totalSize > 0 ? bidTotal / totalSize : 0.5

          if (book.bids[0] && book.asks[0]) {
            book.midPrice = (book.bids[0].price + book.asks[0].price) / 2
            book.spread = book.asks[0].price - book.bids[0].price
            book.spreadBps = book.spread / book.midPrice * 10000
          }

          book.timestamp = Date.now()
        })
      },

      // ---- Bar Actions ----
      addBar(symbol: string, bar: OHLCV) {
        set((state) => {
          if (!state.recentBars[symbol]) {
            state.recentBars[symbol] = []
          }
          const bars = state.recentBars[symbol]

          // Replace last bar if same timestamp, else append
          if (bars.length > 0 && bars[bars.length - 1].time === bar.time) {
            bars[bars.length - 1] = bar
          } else {
            bars.push(bar)
            if (bars.length > MAX_BARS) {
              bars.splice(0, bars.length - MAX_BARS)
            }
          }
        })
      },

      setBars(symbol: string, bars: OHLCV[]) {
        set((state) => {
          state.recentBars[symbol] = bars.slice(-MAX_BARS)
        })
      },

      // ---- Trade Actions ----
      addTrade(symbol: string, trade: Trade) {
        set((state) => {
          if (!state.recentTrades[symbol]) {
            state.recentTrades[symbol] = []
          }
          const trades = state.recentTrades[symbol]
          trades.unshift(trade)
          if (trades.length > MAX_TRADES) {
            trades.splice(MAX_TRADES)
          }
        })
      },

      // ---- Sparkline Actions ----
      setSparkline(symbol: string, points: SparklinePoint[]) {
        set((state) => {
          state.sparklines[symbol] = points
        })
      },

      // ---- Volume Profile Actions ----
      setVolumeProfile(symbol: string, profile: VolumeProfile) {
        set((state) => {
          state.volumeProfiles[symbol] = profile
        })
      },

      // ---- Watchlist Actions ----
      addToWatchlist(symbol: string) {
        set((state) => {
          const existing = state.watchlist.find((w) => w.symbol === symbol)
          if (!existing) {
            state.watchlist.push({ symbol, addedAt: Date.now() })
          }
        })
      },

      removeFromWatchlist(symbol: string) {
        set((state) => {
          const idx = state.watchlist.findIndex((w) => w.symbol === symbol)
          if (idx !== -1) {
            state.watchlist.splice(idx, 1)
          }
        })
      },

      reorderWatchlist(symbols: string[]) {
        set((state) => {
          const map = new Map(state.watchlist.map((w) => [w.symbol, w]))
          state.watchlist = symbols.map((s) => map.get(s) ?? { symbol: s, addedAt: Date.now() })
        })
      },

      setWatchlistSort(sort: SortConfig) {
        set((state) => {
          state.watchlistSort = sort
        })
      },

      // ---- Selection Actions ----
      setSelectedSymbol(symbol: string) {
        set((state) => {
          state.selectedSymbol = symbol
        })
      },

      setSelectedInterval(interval: Interval) {
        set((state) => {
          state.selectedInterval = interval
        })
      },

      setHoveredPrice(price: number | null) {
        set((state) => {
          state.hoveredPrice = price
        })
      },

      // ---- Session Actions ----
      setMarketSession(session: MarketSession) {
        set((state) => {
          state.marketSession = session
        })
      },

      // ---- Connection Actions ----
      setConnected(connected: boolean) {
        set((state) => {
          state.isConnected = connected
        })
      },

      setLastHeartbeat(ts: number) {
        set((state) => {
          state.lastHeartbeat = ts
        })
      },

      addSubscription(channel: string) {
        set((state) => {
          state.subscriptions.add(channel)
        })
      },

      removeSubscription(channel: string) {
        set((state) => {
          state.subscriptions.delete(channel)
        })
      },

      setError(msg: string | null) {
        set((state) => {
          state.errorMessage = msg
        })
      },
    }))
  )
)

// ---- Selectors ----
export const selectQuote = (symbol: string) => (state: MarketStore) => state.quotes[symbol]
export const selectOrderBook = (symbol: string) => (state: MarketStore) => state.orderBooks[symbol]
export const selectBars = (symbol: string) => (state: MarketStore) => state.recentBars[symbol] ?? []
export const selectTrades = (symbol: string) => (state: MarketStore) => state.recentTrades[symbol] ?? []
export const selectSparkline = (symbol: string) => (state: MarketStore) => state.sparklines[symbol] ?? []
export const selectWatchlistSymbols = (state: MarketStore) => state.watchlist.map((w) => w.symbol)

export const selectSortedWatchlist = (state: MarketStore) => {
  const { watchlist, watchlistSort, quotes } = state
  const sorted = [...watchlist].sort((a, b) => {
    const qa = quotes[a.symbol]
    const qb = quotes[b.symbol]
    let valA: number | string = a.symbol
    let valB: number | string = b.symbol

    switch (watchlistSort.field) {
      case 'price':
        valA = qa?.lastPrice ?? 0
        valB = qb?.lastPrice ?? 0
        break
      case 'change':
        valA = qa?.dayChange ?? 0
        valB = qb?.dayChange ?? 0
        break
      case 'changePct':
        valA = qa?.dayChangePct ?? 0
        valB = qb?.dayChangePct ?? 0
        break
      case 'volume':
        valA = qa?.dayVolume ?? 0
        valB = qb?.dayVolume ?? 0
        break
      case 'symbol':
      default:
        valA = a.symbol
        valB = b.symbol
    }

    const cmp =
      typeof valA === 'string' && typeof valB === 'string'
        ? valA.localeCompare(valB)
        : (valA as number) - (valB as number)

    return watchlistSort.direction === 'asc' ? cmp : -cmp
  })
  return sorted
}
