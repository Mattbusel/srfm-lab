import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import type { Trade, EquityPoint, PerformanceMetrics, TradeFilter } from '@/types/trades'

interface TradeState {
  trades: Trade[]
  equityCurve: EquityPoint[]
  metrics: PerformanceMetrics | null
  filter: TradeFilter
  loading: boolean
  error: string | null
  lastUpdated: string | null

  setTrades: (trades: Trade[]) => void
  setEquityCurve: (curve: EquityPoint[]) => void
  setMetrics: (metrics: PerformanceMetrics) => void
  setFilter: (filter: Partial<TradeFilter>) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  reset: () => void
}

export const useTradeStore = create<TradeState>()(
  immer((set) => ({
    trades: [],
    equityCurve: [],
    metrics: null,
    filter: {},
    loading: false,
    error: null,
    lastUpdated: null,

    setTrades: (trades) =>
      set((state) => {
        state.trades = trades
        state.lastUpdated = new Date().toISOString()
      }),

    setEquityCurve: (curve) =>
      set((state) => {
        state.equityCurve = curve
      }),

    setMetrics: (metrics) =>
      set((state) => {
        state.metrics = metrics
      }),

    setFilter: (filter) =>
      set((state) => {
        state.filter = { ...state.filter, ...filter }
      }),

    setLoading: (loading) =>
      set((state) => {
        state.loading = loading
      }),

    setError: (error) =>
      set((state) => {
        state.error = error
      }),

    reset: () =>
      set((state) => {
        state.trades = []
        state.equityCurve = []
        state.metrics = null
        state.error = null
      }),
  }))
)
