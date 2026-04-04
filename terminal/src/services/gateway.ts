// ============================================================
// GO GATEWAY API CLIENT — market data, order book
// ============================================================
import type {
  Quote,
  OrderBook,
  OrderBookLevel,
  OHLCV,
  Trade,
  Bar,
  Interval,
  MarketDepthSnapshot,
  InstrumentMeta,
  MarketSession,
  ApiResponse,
} from '@/types'
import { useSettingsStore } from '@/store/settingsStore'

const getBaseUrl = () => useSettingsStore.getState().settings.gatewayUrl

class GatewayClient {
  private async fetch<T>(path: string, options: RequestInit = {}): Promise<T> {
    const url = `${getBaseUrl()}${path}`
    const response = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    })

    if (!response.ok) {
      const err = await response.text().catch(() => '')
      throw new Error(`Gateway error ${response.status}: ${err}`)
    }

    return response.json() as T
  }

  // ---- Quotes ----
  async getQuote(symbol: string): Promise<Quote> {
    const resp = await this.fetch<ApiResponse<Quote>>(`/v1/quotes/${encodeURIComponent(symbol)}`)
    if (!resp.data) throw new Error('No quote data')
    return resp.data
  }

  async getQuotes(symbols: string[]): Promise<Record<string, Quote>> {
    const resp = await this.fetch<ApiResponse<Record<string, Quote>>>('/v1/quotes/batch', {
      method: 'POST',
      body: JSON.stringify({ symbols }),
    })
    return resp.data ?? {}
  }

  // ---- Order Book ----
  async getOrderBook(symbol: string, levels = 20): Promise<OrderBook> {
    const resp = await this.fetch<ApiResponse<OrderBook>>(
      `/v1/orderbook/${encodeURIComponent(symbol)}?levels=${levels}`
    )
    if (!resp.data) throw new Error('No order book data')
    return resp.data
  }

  async getOrderBookSnapshot(symbol: string): Promise<MarketDepthSnapshot> {
    const resp = await this.fetch<ApiResponse<MarketDepthSnapshot>>(
      `/v1/orderbook/${encodeURIComponent(symbol)}/snapshot`
    )
    if (!resp.data) throw new Error('No snapshot data')
    return resp.data
  }

  // ---- Recent Trades ----
  async getRecentTrades(symbol: string, limit = 100): Promise<Trade[]> {
    const resp = await this.fetch<ApiResponse<Trade[]>>(
      `/v1/trades/${encodeURIComponent(symbol)}?limit=${limit}`
    )
    return resp.data ?? []
  }

  // ---- Bars ----
  async getBars(symbol: string, interval: Interval, limit = 500): Promise<OHLCV[]> {
    const resp = await this.fetch<ApiResponse<OHLCV[]>>(
      `/v1/bars/${encodeURIComponent(symbol)}?interval=${interval}&limit=${limit}`
    )
    return resp.data ?? []
  }

  async getBarsRange(symbol: string, interval: Interval, start: number, end: number): Promise<OHLCV[]> {
    const params = new URLSearchParams({
      interval,
      start: String(start),
      end: String(end),
    })
    const resp = await this.fetch<ApiResponse<OHLCV[]>>(
      `/v1/bars/${encodeURIComponent(symbol)}/range?${params}`
    )
    return resp.data ?? []
  }

  // ---- Instruments ----
  async getInstrumentMeta(symbol: string): Promise<InstrumentMeta> {
    const resp = await this.fetch<ApiResponse<InstrumentMeta>>(
      `/v1/instruments/${encodeURIComponent(symbol)}`
    )
    if (!resp.data) throw new Error('No instrument data')
    return resp.data
  }

  async searchInstruments(query: string, limit = 20): Promise<InstrumentMeta[]> {
    const resp = await this.fetch<ApiResponse<InstrumentMeta[]>>(
      `/v1/instruments/search?q=${encodeURIComponent(query)}&limit=${limit}`
    )
    return resp.data ?? []
  }

  async getWatchlistData(symbols: string[]): Promise<{
    quotes: Record<string, Quote>
    metas: Record<string, InstrumentMeta>
  }> {
    const resp = await this.fetch<ApiResponse<{
      quotes: Record<string, Quote>
      metas: Record<string, InstrumentMeta>
    }>>('/v1/watchlist', {
      method: 'POST',
      body: JSON.stringify({ symbols }),
    })
    return resp.data ?? { quotes: {}, metas: {} }
  }

  // ---- Market Session ----
  async getMarketSession(): Promise<MarketSession> {
    const resp = await this.fetch<ApiResponse<MarketSession>>('/v1/session')
    if (!resp.data) throw new Error('No session data')
    return resp.data
  }

  // ---- Health ----
  async healthCheck(): Promise<{ status: string; feeds: Record<string, string> }> {
    return this.fetch('/v1/health')
  }

  // ---- Mock Data Generators ----
  generateMockOrderBook(symbol: string, midPrice: number, levels = 20): OrderBook {
    const spread = midPrice * 0.0001
    const bids: OrderBookLevel[] = []
    const asks: OrderBookLevel[] = []

    let bidTotal = 0
    let askTotal = 0

    for (let i = 0; i < levels; i++) {
      const bidPrice = midPrice - spread / 2 - i * spread * 0.8
      const askPrice = midPrice + spread / 2 + i * spread * 0.8
      const bidSize = Math.floor(100 + Math.random() * 1000)
      const askSize = Math.floor(100 + Math.random() * 1000)
      bidTotal += bidSize
      askTotal += askSize
      bids.push({ price: bidPrice, size: bidSize, total: bidTotal, pct: 0 })
      asks.push({ price: askPrice, size: askSize, total: askTotal, pct: 0 })
    }

    const total = bidTotal + askTotal
    for (const l of bids) l.pct = l.size / total
    for (const l of asks) l.pct = l.size / total

    return {
      symbol,
      timestamp: Date.now(),
      bids,
      asks,
      midPrice,
      spread: spread,
      spreadBps: (spread / midPrice) * 10000,
      imbalance: bidTotal / total,
      totalBidSize: bidTotal,
      totalAskSize: askTotal,
    }
  }

  generateMockQuote(symbol: string, basePrice: number): Quote {
    const spread = basePrice * 0.0001
    const change = (Math.random() - 0.5) * basePrice * 0.03
    const prevClose = basePrice - change
    return {
      symbol,
      timestamp: Date.now(),
      bidPrice: basePrice - spread / 2,
      bidSize: Math.floor(100 + Math.random() * 500),
      askPrice: basePrice + spread / 2,
      askSize: Math.floor(100 + Math.random() * 500),
      midPrice: basePrice,
      spread,
      spreadBps: (spread / basePrice) * 10000,
      lastPrice: basePrice,
      lastSize: Math.floor(10 + Math.random() * 200),
      dayOpen: prevClose * (1 + (Math.random() - 0.5) * 0.01),
      dayHigh: basePrice * (1 + Math.random() * 0.02),
      dayLow: basePrice * (1 - Math.random() * 0.02),
      dayClose: basePrice,
      dayVolume: Math.floor(1000000 + Math.random() * 5000000),
      dayVwap: basePrice * (1 + (Math.random() - 0.5) * 0.005),
      dayChange: change,
      dayChangePct: change / prevClose,
      prevClose,
    }
  }

  generateMockBars(symbol: string, interval: Interval, count = 500): OHLCV[] {
    const bars: OHLCV[] = []
    let price = 100 + Math.random() * 200
    const intervalMs: Record<string, number> = {
      '1m': 60000,
      '5m': 300000,
      '15m': 900000,
      '1h': 3600000,
      '4h': 14400000,
      '1d': 86400000,
    }
    const ms = intervalMs[interval] ?? 86400000
    let t = Math.floor((Date.now() - count * ms) / 1000)

    for (let i = 0; i < count; i++) {
      const open = price
      const change = (Math.random() - 0.49) * price * 0.02
      price = Math.max(1, price + change)
      const high = Math.max(open, price) * (1 + Math.random() * 0.005)
      const low = Math.min(open, price) * (1 - Math.random() * 0.005)
      const volume = Math.floor(100000 + Math.random() * 500000)

      bars.push({ time: t, open, high, low, close: price, volume })
      t += Math.floor(ms / 1000)
    }

    return bars
  }
}

export const gatewayService = new GatewayClient()
