// ============================================================
// usePortfolio — polls Alpaca for account state
// ============================================================
import { useEffect, useRef, useCallback } from 'react'
import { usePortfolioStore } from '@/store/portfolioStore'
import { useMarketStore } from '@/store/marketStore'
import { useSettingsStore } from '@/store/settingsStore'
import { alpacaService } from '@/services/alpaca'
import { wsManager } from '@/services/ws'

const REFRESH_INTERVAL_MS = 30000  // 30 seconds
const FAST_REFRESH_MS = 5000       // After order submission

export function usePortfolio() {
  const portfolioStore = usePortfolioStore()
  const marketStore = useMarketStore()
  const wsUrl = useSettingsStore((s) => s.settings.wsUrl)
  const alpacaKey = useSettingsStore((s) => s.settings.alpacaApiKey)
  const refreshTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const lastOrderCountRef = useRef(0)

  const refresh = useCallback(async () => {
    try {
      await portfolioStore.refreshAccount()
    } catch {
      // Use mock data if Alpaca not configured
      if (!alpacaKey) {
        const mockAccount = alpacaService.getMockAccount()
        // Set mock positions
        const mockPositions = [
          {
            symbol: 'AAPL',
            qty: 100,
            side: 'long' as const,
            entryPrice: 178.50,
            currentPrice: 189.50,
            unrealizedPnl: 1100,
            unrealizedPnlPct: 0.0616,
            realizedPnl: 0,
            marketValue: 18950,
            costBasis: 17850,
            weight: 0.151,
          },
          {
            symbol: 'NVDA',
            qty: 20,
            side: 'long' as const,
            entryPrice: 820.00,
            currentPrice: 875.40,
            unrealizedPnl: 1108,
            unrealizedPnlPct: 0.0676,
            realizedPnl: 0,
            marketValue: 17508,
            costBasis: 16400,
            weight: 0.140,
          },
          {
            symbol: 'SPY',
            qty: 50,
            side: 'long' as const,
            entryPrice: 525.00,
            currentPrice: 540.25,
            unrealizedPnl: 762.50,
            unrealizedPnlPct: 0.0290,
            realizedPnl: 0,
            marketValue: 27012.50,
            costBasis: 26250,
            weight: 0.215,
          },
          {
            symbol: 'TSLA',
            qty: 30,
            side: 'long' as const,
            entryPrice: 260.00,
            currentPrice: 245.30,
            unrealizedPnl: -441,
            unrealizedPnlPct: -0.0565,
            realizedPnl: 0,
            marketValue: 7359,
            costBasis: 7800,
            weight: 0.0587,
          },
        ]

        // Update market prices from positions
        for (const pos of mockPositions) {
          marketStore.updatePositionPrice?.(pos.symbol, pos.currentPrice)
        }

        // Mock orders
        const mockOrders = [
          {
            id: 'order-001',
            symbol: 'MSFT',
            side: 'buy' as const,
            qty: 10,
            type: 'limit' as const,
            status: 'pending' as const,
            timeInForce: 'day' as const,
            price: 440.00,
            filledQty: 0,
            avgFillPrice: 0,
            commission: 0,
            createdAt: Date.now() - 120000,
            updatedAt: Date.now() - 120000,
            extendedHours: false,
          },
        ]

        // Inject mock data
        const store = usePortfolioStore.getState()
        store['positions' as keyof typeof store]
        ;(usePortfolioStore.setState as Function)({
          account: { ...mockAccount, positions: mockPositions, orders: mockOrders },
          positions: mockPositions,
          orders: mockOrders,
          isLoading: false,
          lastRefresh: Date.now(),
        })
      }
    }
  }, [alpacaKey]) // eslint-disable-line react-hooks/exhaustive-deps

  // Initial load
  useEffect(() => {
    refresh()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Poll for updates
  useEffect(() => {
    refreshTimerRef.current = setInterval(refresh, REFRESH_INTERVAL_MS)
    return () => {
      if (refreshTimerRef.current) clearInterval(refreshTimerRef.current)
    }
  }, [refresh])

  // Subscribe to Alpaca streaming account updates
  useEffect(() => {
    if (!alpacaKey) return

    const ws = wsManager.create('alpaca-stream', {
      url: 'wss://stream.data.alpaca.markets/v2/sip',
      reconnectDelay: 2000,
      onMessage: (data) => {
        const msgs = Array.isArray(data) ? data : [data]
        for (const msg of msgs) {
          const m = msg as { T: string; [key: string]: unknown }
          if (m.T === 'trade_updates') {
            // Refresh positions on fill
            portfolioStore.refreshAccount()
          }
        }
      },
      onStatus: (status) => {
        if (status === 'connected') {
          ws.send({ action: 'auth', key: alpacaKey, secret: useSettingsStore.getState().settings.alpacaSecretKey })
          ws.send({ action: 'listen', data: { streams: ['trade_updates'] } })
        }
      },
    })

    ws.connect()

    return () => {
      ws.close()
    }
  }, [alpacaKey]) // eslint-disable-line react-hooks/exhaustive-deps

  // Update position prices from market data
  useEffect(() => {
    const positions = portfolioStore.positions
    for (const pos of positions) {
      const quote = marketStore.quotes[pos.symbol]
      if (quote && quote.lastPrice !== pos.currentPrice) {
        portfolioStore.updatePositionPrice(pos.symbol, quote.lastPrice)
      }
    }
  }, [marketStore.quotes]) // eslint-disable-line react-hooks/exhaustive-deps

  // Fast refresh after order submissions
  useEffect(() => {
    const orderCount = portfolioStore.orders.length
    if (orderCount > lastOrderCountRef.current) {
      setTimeout(refresh, FAST_REFRESH_MS)
    }
    lastOrderCountRef.current = orderCount
  }, [portfolioStore.orders.length]) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    refresh,
    isLoading: portfolioStore.isLoading,
    lastRefresh: portfolioStore.lastRefresh,
  }
}
