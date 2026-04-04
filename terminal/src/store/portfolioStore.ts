// ============================================================
// PORTFOLIO STORE — Zustand + Immer
// ============================================================
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { subscribeWithSelector, persist } from 'zustand/middleware'
import type {
  AccountState,
  Position,
  Order,
  OrderRequest,
  HistoricalTrade,
  EquityPoint,
  RiskMetrics,
  PortfolioAnalytics,
  DailyPnlTarget,
  OrderFill,
} from '@/types'
import { alpacaService } from '@/services/alpaca'

interface PortfolioState {
  account: AccountState | null
  positions: Position[]
  orders: Order[]
  tradeHistory: HistoricalTrade[]
  equityHistory: EquityPoint[]
  riskMetrics: RiskMetrics | null
  analytics: PortfolioAnalytics | null
  dailyPnlTarget: DailyPnlTarget | null
  fills: OrderFill[]

  isLoading: boolean
  isSubmittingOrder: boolean
  lastRefresh: number | null
  error: string | null
  pendingOrderIds: Set<string>
}

interface PortfolioActions {
  // Order Management
  submitOrder(req: OrderRequest): Promise<Order | null>
  cancelOrder(orderId: string): Promise<boolean>
  cancelAllOrders(): Promise<void>

  // Data Refresh
  refreshAccount(): Promise<void>
  refreshPositions(): Promise<void>
  refreshOrders(): Promise<void>
  refreshTradeHistory(params?: { startDate?: string; endDate?: string; symbol?: string }): Promise<void>
  refreshEquityHistory(days?: number): Promise<void>

  // Local state
  updatePositionPrice(symbol: string, price: number): void
  addFill(fill: OrderFill): void
  setDailyPnlTarget(target: number): void
  clearError(): void
}

type PortfolioStore = PortfolioState & PortfolioActions

export const usePortfolioStore = create<PortfolioStore>()(
  subscribeWithSelector(
    immer((set, get) => ({
      // ---- Initial State ----
      account: null,
      positions: [],
      orders: [],
      tradeHistory: [],
      equityHistory: [],
      riskMetrics: null,
      analytics: null,
      dailyPnlTarget: null,
      fills: [],
      isLoading: false,
      isSubmittingOrder: false,
      lastRefresh: null,
      error: null,
      pendingOrderIds: new Set(),

      // ---- Order Management ----
      async submitOrder(req: OrderRequest) {
        set((state) => {
          state.isSubmittingOrder = true
          state.error = null
        })

        try {
          const order = await alpacaService.submitOrder(req)
          set((state) => {
            state.isSubmittingOrder = false
            // Add to orders list
            state.orders.unshift(order)
          })
          // Refresh account after order
          setTimeout(() => get().refreshAccount(), 1000)
          return order
        } catch (err) {
          set((state) => {
            state.isSubmittingOrder = false
            state.error = err instanceof Error ? err.message : 'Order submission failed'
          })
          return null
        }
      },

      async cancelOrder(orderId: string) {
        set((state) => {
          state.pendingOrderIds.add(orderId)
        })

        try {
          await alpacaService.cancelOrder(orderId)
          set((state) => {
            state.pendingOrderIds.delete(orderId)
            const idx = state.orders.findIndex((o) => o.id === orderId)
            if (idx !== -1) {
              state.orders[idx].status = 'cancelled'
              state.orders[idx].cancelledAt = Date.now()
            }
          })
          return true
        } catch (err) {
          set((state) => {
            state.pendingOrderIds.delete(orderId)
            state.error = err instanceof Error ? err.message : 'Cancel failed'
          })
          return false
        }
      },

      async cancelAllOrders() {
        const openOrders = get().orders.filter(
          (o) => o.status === 'pending' || o.status === 'accepted' || o.status === 'partial'
        )
        await Promise.all(openOrders.map((o) => get().cancelOrder(o.id)))
      },

      // ---- Data Refresh ----
      async refreshAccount() {
        set((state) => { state.isLoading = true })

        try {
          const [account, positions, orders] = await Promise.all([
            alpacaService.getAccount(),
            alpacaService.getPositions(),
            alpacaService.getOrders('open'),
          ])

          set((state) => {
            state.account = account
            state.positions = positions
            state.orders = orders
            state.isLoading = false
            state.lastRefresh = Date.now()
          })
        } catch (err) {
          set((state) => {
            state.isLoading = false
            state.error = err instanceof Error ? err.message : 'Failed to refresh account'
          })
        }
      },

      async refreshPositions() {
        try {
          const positions = await alpacaService.getPositions()
          set((state) => {
            state.positions = positions
          })
        } catch (err) {
          set((state) => {
            state.error = err instanceof Error ? err.message : 'Failed to refresh positions'
          })
        }
      },

      async refreshOrders() {
        try {
          const orders = await alpacaService.getOrders('all')
          set((state) => {
            state.orders = orders
          })
        } catch (err) {
          set((state) => {
            state.error = err instanceof Error ? err.message : 'Failed to refresh orders'
          })
        }
      },

      async refreshTradeHistory(params = {}) {
        try {
          const trades = await alpacaService.getTradeHistory(params)
          set((state) => {
            state.tradeHistory = trades
          })
        } catch (err) {
          set((state) => {
            state.error = err instanceof Error ? err.message : 'Failed to refresh trade history'
          })
        }
      },

      async refreshEquityHistory(days = 30) {
        try {
          const history = await alpacaService.getEquityHistory(days)
          set((state) => {
            state.equityHistory = history
          })
        } catch (err) {
          set((state) => {
            state.error = err instanceof Error ? err.message : 'Failed to refresh equity history'
          })
        }
      },

      // ---- Local State Updates ----
      updatePositionPrice(symbol: string, price: number) {
        set((state) => {
          const pos = state.positions.find((p) => p.symbol === symbol)
          if (pos) {
            pos.currentPrice = price
            const pnlMultiplier = pos.side === 'long' ? 1 : -1
            pos.unrealizedPnl = (price - pos.entryPrice) * pos.qty * pnlMultiplier
            pos.unrealizedPnlPct = pos.unrealizedPnl / (pos.entryPrice * pos.qty)
            pos.marketValue = price * pos.qty
          }

          // Update account equity estimate
          if (state.account) {
            const totalValue = state.positions.reduce((sum, p) => {
              if (p.symbol === symbol) return sum + price * p.qty
              return sum + p.marketValue
            }, 0)
            state.account.longMarketValue = totalValue
          }
        })
      },

      addFill(fill: OrderFill) {
        set((state) => {
          state.fills.unshift(fill)
          if (state.fills.length > 1000) {
            state.fills.splice(1000)
          }

          // Update corresponding order
          const order = state.orders.find((o) => o.id === fill.orderId)
          if (order) {
            order.filledQty += fill.qty
            order.avgFillPrice = fill.price  // simplified
            if (order.filledQty >= order.qty) {
              order.status = 'filled'
              order.filledAt = fill.timestamp
            } else {
              order.status = 'partial'
            }
          }
        })
      },

      setDailyPnlTarget(target: number) {
        set((state) => {
          const equity = state.account?.equity ?? 0
          const achieved = state.account?.dayPnl ?? 0
          state.dailyPnlTarget = {
            target,
            targetPct: equity > 0 ? target / equity : 0,
            achieved,
            achievedPct: equity > 0 ? achieved / equity : 0,
            remaining: target - achieved,
            onTrack: achieved >= target * 0.5,
          }
        })
      },

      clearError() {
        set((state) => {
          state.error = null
        })
      },
    }))
  )
)

// ---- Selectors ----
export const selectOpenPositions = (state: PortfolioStore) =>
  state.positions.filter((p) => p.qty > 0)

export const selectOpenOrders = (state: PortfolioStore) =>
  state.orders.filter((o) => o.status === 'pending' || o.status === 'accepted' || o.status === 'partial')

export const selectPositionForSymbol = (symbol: string) => (state: PortfolioStore) =>
  state.positions.find((p) => p.symbol === symbol) ?? null

export const selectTotalUnrealizedPnl = (state: PortfolioStore) =>
  state.positions.reduce((sum, p) => sum + p.unrealizedPnl, 0)

export const selectPortfolioConcentration = (state: PortfolioStore) => {
  const total = state.positions.reduce((sum, p) => sum + Math.abs(p.marketValue), 0)
  return state.positions.map((p) => ({
    symbol: p.symbol,
    weight: total > 0 ? Math.abs(p.marketValue) / total : 0,
    marketValue: p.marketValue,
  }))
}
