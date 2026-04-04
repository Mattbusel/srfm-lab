// ============================================================
// signalsStore.ts — Zustand store for BH signals
// ============================================================
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import type { SignalCard, BHFormation, BHState, MarketRegime } from '@/types'

function mockSignalCard(
  symbol: string,
  daily: BHState,
  hourly: BHState,
  m15: BHState,
  mass: number,
  deltaScore: number,
  trend: MarketRegime,
  price: number,
  change24hPct: number,
): SignalCard {
  return {
    symbol,
    daily,
    hourly,
    m15,
    mass,
    deltaScore,
    activeFormations: Math.floor(mass * 3 + 1),
    trend,
    lastUpdate: new Date(Date.now() - Math.random() * 60000).toISOString(),
    price,
    change24h: price * change24hPct,
    change24hPct,
    volume24h: price * (Math.random() * 50000 + 10000),
  }
}

function generateSignalCards(): SignalCard[] {
  return [
    mockSignalCard('BTCUSDT',  'bullish', 'bullish', 'bullish', 1.8, 0.72, 'trending_up',   63450, 0.021),
    mockSignalCard('ETHUSDT',  'bullish', 'bullish', 'neutral', 1.4, 0.54, 'trending_up',   3218,  0.018),
    mockSignalCard('SOLUSDT',  'bullish', 'neutral', 'bearish', 1.1, 0.22, 'ranging',       156.5, -0.008),
    mockSignalCard('BNBUSDT',  'neutral', 'bullish', 'bullish', 0.9, 0.41, 'ranging',       582,   0.031),
    mockSignalCard('AVAXUSDT', 'bearish', 'bearish', 'neutral', 1.6, -0.6, 'trending_down', 34.2,  -0.042),
    mockSignalCard('LINKUSDT', 'bullish', 'neutral', 'bullish', 1.2, 0.35, 'trending_up',   14.8,  0.015),
    mockSignalCard('DOGEUSDT', 'bullish', 'bullish', 'bullish', 1.9, 0.81, 'trending_up',   0.131, 0.055),
    mockSignalCard('UNIUSDT',  'neutral', 'neutral', 'neutral', 0.5, 0.05, 'ranging',       8.7,   0.003),
    mockSignalCard('AAVEUSDT', 'bearish', 'neutral', 'bearish', 1.3, -0.4, 'trending_down', 94.5,  -0.028),
    mockSignalCard('MATICUSDT','bullish', 'neutral', 'bullish', 1.0, 0.28, 'ranging',       0.78,  0.011),
    mockSignalCard('ARBUSDT',  'bullish', 'bullish', 'neutral', 1.5, 0.62, 'trending_up',   1.12,  0.038),
    mockSignalCard('OPUSDT',   'neutral', 'bearish', 'bearish', 0.7, -0.2, 'ranging',       2.34,  -0.017),
  ]
}

function generateFormations(): BHFormation[] {
  const arr: BHFormation[] = []
  const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT', 'ARBUSDT']
  const tfs = ['15m', '1h', '4h', '1d'] as const
  let id = 1
  for (const sym of symbols) {
    const n = Math.floor(Math.random() * 3) + 1
    for (let i = 0; i < n; i++) {
      const state: BHState = Math.random() > 0.4 ? 'bullish' : Math.random() > 0.5 ? 'bearish' : 'neutral'
      arr.push({
        id: String(id++),
        symbol: sym,
        timeframe: tfs[Math.floor(Math.random() * tfs.length)],
        state,
        mass: Math.random() * 2,
        deltaScore: (Math.random() - 0.3) * 2 - 0.5,
        entryBar: Math.floor(Math.random() * 50),
        startTime: new Date(Date.now() - Math.random() * 3600000 * 24).toISOString(),
        activeCount: Math.floor(Math.random() * 4) + 1,
        confirmedTimeframes: tfs.slice(0, Math.floor(Math.random() * 3) + 1),
        patternType: ['Double Bottom', 'Cup Handle', 'Flag', 'Wedge', 'Triangle'][Math.floor(Math.random() * 5)],
        reliability: Math.random() * 0.4 + 0.6,
      })
    }
  }
  return arr
}

interface SignalsState {
  cards: SignalCard[]
  formations: BHFormation[]
  loading: boolean
  selectedSymbol: string | null

  setCards: (c: SignalCard[]) => void
  updateCard: (symbol: string, update: Partial<SignalCard>) => void
  setFormations: (f: BHFormation[]) => void
  addFormation: (f: BHFormation) => void
  setLoading: (l: boolean) => void
  selectSymbol: (s: string | null) => void
  initMockData: () => void
}

export const useSignalsStore = create<SignalsState>()(
  immer((set) => ({
    cards: [],
    formations: [],
    loading: false,
    selectedSymbol: null,

    setCards: (c) =>
      set((state) => {
        state.cards = c
      }),

    updateCard: (symbol, update) =>
      set((state) => {
        const idx = state.cards.findIndex((c) => c.symbol === symbol)
        if (idx >= 0) Object.assign(state.cards[idx], update)
        else state.cards.push({ ...update, symbol } as SignalCard)
      }),

    setFormations: (f) =>
      set((state) => {
        state.formations = f
      }),

    addFormation: (f) =>
      set((state) => {
        state.formations.unshift(f)
        if (state.formations.length > 200) state.formations.pop()
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
        state.cards = generateSignalCards()
        state.formations = generateFormations()
      }),
  })),
)
