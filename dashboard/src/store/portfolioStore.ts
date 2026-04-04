// ============================================================
// portfolioStore.ts — Zustand store for portfolio state
// ============================================================
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import type { PortfolioSnapshot, EquityPoint, DrawdownPoint } from '@/types'

// ---- Mock data generators ----

function generateEquityCurve(days = 90): EquityPoint[] {
  const points: EquityPoint[] = []
  let equity = 100_000
  const now = Date.now()
  for (let i = days * 24; i >= 0; i--) {
    const ts = new Date(now - i * 3_600_000).toISOString()
    const delta = (Math.random() - 0.47) * 800
    equity = Math.max(equity + delta, 50_000)
    points.push({
      timestamp: ts,
      equity,
      drawdown: 0,
      dailyPnl: delta,
    })
  }
  // compute drawdown
  let peak = points[0].equity
  for (const p of points) {
    if (p.equity > peak) peak = p.equity
    p.drawdown = (p.equity - peak) / peak
  }
  return points
}

function generateSnapshot(): PortfolioSnapshot {
  return {
    timestamp: new Date().toISOString(),
    totalEquity: 127_450.32,
    totalUnrealizedPnl: 3_280.15,
    totalRealizedPnl: 24_170.00,
    dailyPnl: 1_245.80,
    dailyPnlPct: 0.985,
    weeklyPnl: 3_780.00,
    monthlyPnl: 9_640.00,
    ytdPnl: 27_450.32,
    totalMarginUsed: 48_200.00,
    availableMargin: 79_250.32,
    marginUtilization: 0.378,
    winRate: 0.612,
    sharpeRatio: 1.83,
    calmarRatio: 2.14,
    sortinoRatio: 2.51,
    maxDrawdown: -0.142,
    currentDrawdown: -0.031,
    volatility: 0.187,
    beta: 0.74,
    alpha: 0.23,
  }
}

function generateDailyPnl(days = 30): { date: string; pnl: number; pnlPct: number }[] {
  const arr: { date: string; pnl: number; pnlPct: number }[] = []
  const now = Date.now()
  for (let i = days; i >= 0; i--) {
    const d = new Date(now - i * 86_400_000)
    const pnl = (Math.random() - 0.42) * 3_000
    arr.push({
      date: d.toISOString().slice(0, 10),
      pnl,
      pnlPct: pnl / 110_000,
    })
  }
  return arr
}

// ---- Store interface ----

interface PortfolioState {
  snapshot: PortfolioSnapshot | null
  equityCurve: EquityPoint[]
  drawdown: DrawdownPoint[]
  dailyPnl: { date: string; pnl: number; pnlPct: number }[]
  loading: boolean
  lastFetch: Date | null

  setSnapshot: (s: PortfolioSnapshot) => void
  setEquityCurve: (c: EquityPoint[]) => void
  setDailyPnl: (d: { date: string; pnl: number; pnlPct: number }[]) => void
  setLoading: (l: boolean) => void
  initMockData: () => void
  updateRealtime: (s: PortfolioSnapshot) => void
}

export const usePortfolioStore = create<PortfolioState>()(
  immer((set) => ({
    snapshot: null,
    equityCurve: [],
    drawdown: [],
    dailyPnl: [],
    loading: false,
    lastFetch: null,

    setSnapshot: (s) =>
      set((state) => {
        state.snapshot = s
        state.lastFetch = new Date()
      }),

    setEquityCurve: (c) =>
      set((state) => {
        state.equityCurve = c
      }),

    setDailyPnl: (d) =>
      set((state) => {
        state.dailyPnl = d
      }),

    setLoading: (l) =>
      set((state) => {
        state.loading = l
      }),

    initMockData: () =>
      set((state) => {
        state.snapshot = generateSnapshot()
        state.equityCurve = generateEquityCurve()
        state.dailyPnl = generateDailyPnl()
        state.lastFetch = new Date()
      }),

    updateRealtime: (s) =>
      set((state) => {
        state.snapshot = s
        // Append latest equity point
        if (state.equityCurve.length > 0) {
          const last = state.equityCurve[state.equityCurve.length - 1]
          const peak = state.equityCurve.reduce((m, p) => Math.max(m, p.equity), 0)
          state.equityCurve.push({
            timestamp: s.timestamp,
            equity: s.totalEquity,
            drawdown: (s.totalEquity - peak) / peak,
            dailyPnl: s.dailyPnl,
          })
          // Keep last 720 points
          if (state.equityCurve.length > 720) state.equityCurve.shift()
          void last
        }
      }),
  })),
)
