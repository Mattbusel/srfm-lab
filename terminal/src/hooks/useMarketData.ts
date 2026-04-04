// ============================================================
// useMarketData — connects to gateway WebSocket
// ============================================================
import { useEffect, useRef, useCallback } from 'react'
import { useMarketStore } from '@/store/marketStore'
import { useSettingsStore } from '@/store/settingsStore'
import { wsManager } from '@/services/ws'
import { gatewayService } from '@/services/gateway'
import type { Quote, Trade, OHLCV, OrderBook, WSMessage, Interval } from '@/types'

const MOCK_SYMBOLS: Record<string, number> = {
  'SPY': 540.25,
  'QQQ': 462.10,
  'AAPL': 189.50,
  'TSLA': 245.30,
  'NVDA': 875.40,
  'AMZN': 198.75,
  'MSFT': 442.80,
  'BTC/USD': 68500,
  'ETH/USD': 3750,
}

export function useMarketData(symbols?: string[]) {
  const store = useMarketStore()
  const wsUrl = useSettingsStore((s) => s.settings.gatewayWsUrl)
  const watchlistSymbols = useMarketStore((s) => s.watchlist.map((w) => w.symbol))
  const targetSymbols = symbols ?? watchlistSymbols
  const mockIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const handleMessage = useCallback((data: unknown) => {
    const msg = data as WSMessage
    if (!msg?.type) return

    switch (msg.type) {
      case 'quote':
        store.updateQuote(msg.data as Quote)
        break
      case 'trade':
        if (msg.symbol) store.addTrade(msg.symbol, msg.data as Trade)
        break
      case 'bar':
        if (msg.symbol) store.addBar(msg.symbol, msg.data as OHLCV)
        break
      case 'orderbook':
        store.updateOrderBook(msg.data as OrderBook)
        break
      case 'heartbeat':
        store.setLastHeartbeat(Date.now())
        break
      case 'error':
        store.setError(String((msg.data as { message?: string })?.message ?? 'Unknown error'))
        break
    }
  }, [store])

  useEffect(() => {
    const ws = wsManager.create('market-data', {
      url: wsUrl,
      reconnectDelay: 1000,
      maxReconnectDelay: 30000,
      onMessage: handleMessage,
      onStatus: (status) => {
        store.setConnected(status === 'connected')
        if (status === 'connected') {
          // Subscribe to target symbols
          ws.send({
            type: 'subscribe',
            channels: ['quotes', 'trades', 'bars'],
            symbols: targetSymbols,
          })
        }
      },
    })

    ws.connect()

    return () => {
      ws.close()
    }
  }, [wsUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  // Update subscriptions when symbols change
  useEffect(() => {
    const ws = wsManager.get('market-data')
    if (ws?.isConnected) {
      ws.send({ type: 'subscribe', channels: ['quotes', 'trades'], symbols: targetSymbols })
    }
  }, [targetSymbols.join(',')])  // eslint-disable-line react-hooks/exhaustive-deps

  // Mock data feed when not connected
  useEffect(() => {
    if (store.isConnected) return

    // Start mock data simulation
    const simulatePrices = () => {
      for (const [symbol, basePrice] of Object.entries(MOCK_SYMBOLS)) {
        if (!targetSymbols.includes(symbol)) continue
        const price = basePrice * (1 + (Math.random() - 0.5) * 0.001)
        MOCK_SYMBOLS[symbol] = price
        const quote = gatewayService.generateMockQuote(symbol, price)
        store.updateQuote(quote)
      }
    }

    mockIntervalRef.current = setInterval(simulatePrices, 500)
    return () => {
      if (mockIntervalRef.current) clearInterval(mockIntervalRef.current)
    }
  }, [store.isConnected, targetSymbols.join(',')]) // eslint-disable-line react-hooks/exhaustive-deps
}

export function useChartDataFeed(symbol: string, interval: Interval) {
  const store = useMarketStore()
  const wsUrl = useSettingsStore((s) => s.settings.gatewayWsUrl)
  const apiUrl = useSettingsStore((s) => s.settings.apiUrl)

  // Load historical bars
  useEffect(() => {
    const existing = store.recentBars[symbol]
    if (existing && existing.length > 100) return  // already loaded

    // Try API first, fall back to mock
    const loadBars = async () => {
      try {
        const bars = await gatewayService.getBars(symbol, interval)
        store.setBars(symbol, bars)
      } catch {
        // Use mock data
        const mockBars = gatewayService.generateMockBars(symbol, interval, 500)
        store.setBars(symbol, mockBars)
      }
    }

    loadBars()
  }, [symbol, interval]) // eslint-disable-line react-hooks/exhaustive-deps

  // Subscribe to real-time bar updates
  useEffect(() => {
    const ws = wsManager.get('market-data')
    if (ws?.isConnected) {
      ws.send({
        type: 'subscribe',
        channels: ['bars'],
        symbols: [symbol],
        interval,
      })
    }
  }, [symbol, interval])
}
