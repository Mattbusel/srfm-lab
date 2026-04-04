// ============================================================
// useOrderBook — live L2 order book
// ============================================================
import { useEffect, useCallback, useRef } from 'react'
import { useMarketStore } from '@/store/marketStore'
import { useSettingsStore } from '@/store/settingsStore'
import { wsManager } from '@/services/ws'
import { gatewayService } from '@/services/gateway'
import type { OrderBook } from '@/types'

export function useOrderBook(symbol: string, levels = 20) {
  const store = useMarketStore()
  const wsUrl = useSettingsStore((s) => s.settings.gatewayWsUrl)
  const orderBook = store.orderBooks[symbol]
  const mockRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const currentPriceRef = useRef<number>(100)

  const handleMessage = useCallback((data: unknown) => {
    const msg = data as { type: string; symbol?: string; data: unknown }
    if (!msg) return

    if (msg.type === 'orderbook' && msg.symbol === symbol) {
      store.updateOrderBook(msg.data as OrderBook)
    } else if (msg.type === 'orderbook_delta' && msg.symbol === symbol) {
      const delta = msg.data as { bids: [number, number][]; asks: [number, number][] }
      store.applyOrderBookDelta(symbol, delta.bids, delta.asks)
    }
  }, [symbol, store])

  useEffect(() => {
    // Subscribe to order book channel
    const ws = wsManager.get('market-data')
    if (ws?.isConnected) {
      ws.send({
        type: 'subscribe',
        channels: ['orderbook'],
        symbols: [symbol],
        levels,
      })
    }

    // Also subscribe via dedicated orderbook WS if available
    const obWs = wsManager.create(`orderbook-${symbol}`, {
      url: wsUrl,
      name: `orderbook-${symbol}`,
      reconnectDelay: 500,
      onMessage: handleMessage,
      onStatus: (status) => {
        if (status === 'connected') {
          obWs.send({ type: 'subscribe', channel: 'orderbook', symbol, levels })
        }
      },
    })

    obWs.connect()

    return () => {
      obWs.close()
    }
  }, [symbol, levels, wsUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  // Mock order book when not connected
  useEffect(() => {
    if (orderBook) {
      currentPriceRef.current = orderBook.midPrice
      return
    }

    // Generate initial mock book
    const quote = store.quotes[symbol]
    const price = quote?.lastPrice ?? currentPriceRef.current

    const mockBook = gatewayService.generateMockOrderBook(symbol, price, levels)
    store.updateOrderBook(mockBook)

    // Simulate updates
    mockRef.current = setInterval(() => {
      const currentPrice = store.quotes[symbol]?.lastPrice ?? currentPriceRef.current
      const updatedBook = gatewayService.generateMockOrderBook(symbol, currentPrice, levels)
      store.updateOrderBook(updatedBook)
    }, 1000)

    return () => {
      if (mockRef.current) clearInterval(mockRef.current)
    }
  }, [symbol, !!orderBook]) // eslint-disable-line react-hooks/exhaustive-deps

  return orderBook ?? null
}
