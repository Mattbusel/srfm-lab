// ============================================================
// positionsStore.ts — Zustand store for open positions
// ============================================================
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import type { Position, InstrumentSector } from '@/types'

function mockPosition(
  symbol: string,
  side: 'long' | 'short',
  size: number,
  entry: number,
  current: number,
  strategy: string,
  sector: InstrumentSector,
): Position {
  const sizeUsd = size * current
  const unrealizedPnl = side === 'long'
    ? (current - entry) * size
    : (entry - current) * size
  const unrealizedPnlPct = unrealizedPnl / (entry * size)
  return {
    symbol,
    side,
    size,
    sizeUsd,
    entryPrice: entry,
    currentPrice: current,
    unrealizedPnl,
    unrealizedPnlPct,
    realizedPnl: Math.random() * 2000 - 500,
    weight: 0,
    strategy,
    openedAt: new Date(Date.now() - Math.random() * 7 * 86400000).toISOString(),
    leverage: Math.floor(Math.random() * 3) + 1,
    stopLoss: side === 'long' ? entry * 0.95 : entry * 1.05,
    takeProfit: side === 'long' ? entry * 1.12 : entry * 0.88,
    margin: sizeUsd / (Math.floor(Math.random() * 3) + 1),
    sector,
  }
}

function generatePositions(): Position[] {
  const raw: Position[] = [
    mockPosition('BTCUSDT',  'long',  0.35,   62800, 63450, 'BH_Trend', 'L1'),
    mockPosition('ETHUSDT',  'long',  2.8,    3100,  3218,  'BH_Trend', 'L1'),
    mockPosition('SOLUSDT',  'long',  42,     148,   156.5, 'BH_Swing', 'L1'),
    mockPosition('BNBUSDT',  'long',  8,      560,   582,   'BH_Swing', 'Exchange'),
    mockPosition('LINKUSDT', 'long',  120,    14.2,  14.8,  'BH_Scalp', 'DeFi'),
    mockPosition('AVAXUSDT', 'short', 30,     36,    34.2,  'BH_Short', 'L1'),
    mockPosition('DOGEUSDT', 'long',  8500,   0.124, 0.131, 'Momentum', 'Meme'),
    mockPosition('UNIUSDT',  'long',  300,    8.4,   8.7,   'DeFi_Rev', 'DeFi'),
    mockPosition('AAVEUSDT', 'short', 15,     98,    94.5,  'BH_Short', 'DeFi'),
    mockPosition('MATICUSDT','long',  2000,   0.72,  0.78,  'Momentum', 'L1'),
  ]
  const totalUsd = raw.reduce((s, p) => s + p.sizeUsd, 0)
  return raw.map((p) => ({ ...p, weight: p.sizeUsd / totalUsd }))
}

interface PositionsState {
  positions: Position[]
  loading: boolean
  selectedSymbol: string | null

  setPositions: (p: Position[]) => void
  updatePosition: (symbol: string, update: Partial<Position>) => void
  setLoading: (l: boolean) => void
  selectSymbol: (s: string | null) => void
  initMockData: () => void
}

export const usePositionsStore = create<PositionsState>()(
  immer((set) => ({
    positions: [],
    loading: false,
    selectedSymbol: null,

    setPositions: (p) =>
      set((state) => {
        state.positions = p
      }),

    updatePosition: (symbol, update) =>
      set((state) => {
        const idx = state.positions.findIndex((p) => p.symbol === symbol)
        if (idx >= 0) Object.assign(state.positions[idx], update)
      }),

    setLoading: (l) =>
      set((state) => {
        state.loading = l
      }),

    selectSymbol: (s) =>
      set((state) => {
        state.selectedSymbol = s
      }),

    initMockData: () =>
      set((state) => {
        state.positions = generatePositions()
      }),
  })),
)
