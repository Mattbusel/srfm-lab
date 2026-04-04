import type { RegimeType } from './trades'

export interface RegimeSegment {
  startDate: string
  endDate: string
  regime: RegimeType
  durationDays: number
  priceReturn: number
  volatility: number
  avgVolume: number
}

export interface RegimeTransition {
  from: RegimeType
  to: RegimeType
  probability: number
  avgDurationFrom: number
  count: number
}

export type TransitionMatrix = Record<RegimeType, Record<RegimeType, number>>

export interface RegimePerformance {
  regime: RegimeType
  totalPnl: number
  avgPnl: number
  sharpe: number
  winRate: number
  tradeCount: number
  avgHoldingHours: number
  maxDrawdown: number
  occurrencePct: number
}

export interface RegimeDuration {
  regime: RegimeType
  avgDays: number
  medianDays: number
  minDays: number
  maxDays: number
  p25Days: number
  p75Days: number
  count: number
}

export interface StressScenario {
  id: string
  name: string
  description: string
  category: 'market_crash' | 'vol_spike' | 'liquidity' | 'correlation' | 'tail'
  startDate: string
  endDate: string
  regimeSequence: RegimeType[]
  pnlImpact: number
  pnlImpactPct: number
  maxDrawdown: number
  recoveryDays: number | null
  probability: number
}

export interface RegimeFilter {
  regimes: RegimeType[]
  dateFrom?: string
  dateTo?: string
}
