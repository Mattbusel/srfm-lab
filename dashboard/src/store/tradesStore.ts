// ============================================================
// tradesStore.ts — Zustand store for trade history
// ============================================================
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import type { Trade, BHState, MarketRegime } from '@/types'

const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'LINKUSDT', 'AVAXUSDT', 'DOGEUSDT', 'UNIUSDT']
const STRATEGIES = ['BH_Trend', 'BH_Swing', 'BH_Scalp', 'BH_Short', 'Momentum', 'DeFi_Rev']
const REGIMES: MarketRegime[] = ['trending_up', 'trending_down', 'ranging', 'volatile']
const STATES: BHState[] = ['bullish', 'bearish', 'neutral']
const REASONS = ['tp', 'sl', 'manual', 'signal'] as const

function generateTrades(n = 200): Trade[] {
  const trades: Trade[] = []
  for (let i = 0; i < n; i++) {
    const sym = SYMBOLS[Math.floor(Math.random() * SYMBOLS.length)]
    const side: 'long' | 'short' = Math.random() > 0.45 ? 'long' : 'short'
    const durationMs = Math.floor(Math.random() * 48 * 3600000) + 60000
    const entryTime = new Date(Date.now() - Math.random() * 90 * 86400000)
    const exitTime = new Date(entryTime.getTime() + durationMs)
    const basePrice = { BTCUSDT: 60000, ETHUSDT: 3000, SOLUSDT: 150, BNBUSDT: 550, LINKUSDT: 14, AVAXUSDT: 35, DOGEUSDT: 0.12, UNIUSDT: 8 }[sym] ?? 100
    const entry = basePrice * (1 + (Math.random() - 0.5) * 0.1)
    const isWin = Math.random() < 0.62
    const moveMultiplier = isWin ? (Math.random() * 0.06 + 0.005) : -(Math.random() * 0.04 + 0.005)
    const exit = side === 'long' ? entry * (1 + moveMultiplier) : entry * (1 - moveMultiplier)
    const size = Math.random() * 2 + 0.1
    const sizeUsd = size * entry
    const pnl = side === 'long' ? (exit - entry) * size : (entry - exit) * size
    const pnlPct = pnl / sizeUsd
    trades.push({
      id: `trade-${i}`,
      symbol: sym,
      side,
      entryPrice: entry,
      exitPrice: exit,
      size,
      sizeUsd,
      pnl,
      pnlPct,
      fees: sizeUsd * 0.00055,
      strategy: STRATEGIES[Math.floor(Math.random() * STRATEGIES.length)],
      entryTime: entryTime.toISOString(),
      exitTime: exitTime.toISOString(),
      durationMs,
      regime: REGIMES[Math.floor(Math.random() * REGIMES.length)],
      bhSignalAtEntry: STATES[Math.floor(Math.random() * STATES.length)],
      maxFavorableExcursion: Math.abs(pnl) * (1 + Math.random() * 0.5),
      maxAdverseExcursion: Math.abs(pnl) * Math.random() * 0.6,
      exitReason: REASONS[Math.floor(Math.random() * REASONS.length)],
      tags: [],
    })
  }
  return trades.sort((a, b) => new Date(b.exitTime).getTime() - new Date(a.exitTime).getTime())
}

interface TradesState {
  trades: Trade[]
  loading: boolean
  filterSymbol: string | null
  filterStrategy: string | null
  filterSide: 'long' | 'short' | null

  setTrades: (t: Trade[]) => void
  appendTrades: (t: Trade[]) => void
  setLoading: (l: boolean) => void
  setFilter: (key: 'filterSymbol' | 'filterStrategy' | 'filterSide', value: string | null) => void
  initMockData: () => void

  getFiltered: () => Trade[]
}

export const useTradesStore = create<TradesState>()(
  immer((set, get) => ({
    trades: [],
    loading: false,
    filterSymbol: null,
    filterStrategy: null,
    filterSide: null,

    setTrades: (t) =>
      set((state) => {
        state.trades = t
      }),

    appendTrades: (t) =>
      set((state) => {
        state.trades.unshift(...t)
      }),

    setLoading: (l) =>
      set((state) => {
        state.loading = l
      }),

    setFilter: (key, value) =>
      set((state) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ;(state as any)[key] = value
      }),

    initMockData: () =>
      set((state) => {
        state.trades = generateTrades(200)
      }),

    getFiltered: () => {
      const { trades, filterSymbol, filterStrategy, filterSide } = get()
      return trades.filter((t) => {
        if (filterSymbol && t.symbol !== filterSymbol) return false
        if (filterStrategy && t.strategy !== filterStrategy) return false
        if (filterSide && t.side !== filterSide) return false
        return true
      })
    },
  })),
)
