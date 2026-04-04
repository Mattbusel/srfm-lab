import { api } from './client'
import type { Trade, TradeFilter, EquityPoint, PerformanceMetrics } from '@/types/trades'
import { generateMockTrades, generateMockEquity, generateMockMetrics } from '@/api/mockData'

const USE_MOCK = true // set to false when API is available

export async function fetchTrades(filter?: TradeFilter): Promise<Trade[]> {
  if (USE_MOCK) return generateMockTrades(200, filter)
  return api.get<Trade[]>('/api/trades', filter as Record<string, string>)
}

export async function fetchEquityCurve(
  days = 30,
  instrument?: string
): Promise<EquityPoint[]> {
  if (USE_MOCK) return generateMockEquity(days, 100_000)
  return api.get<EquityPoint[]>('/api/equity', { days, instrument })
}

export async function fetchPerformanceMetrics(
  dateFrom?: string,
  dateTo?: string
): Promise<PerformanceMetrics> {
  if (USE_MOCK) return generateMockMetrics()
  return api.get<PerformanceMetrics>('/api/performance', { dateFrom, dateTo })
}

export async function fetchTopTrades(limit = 10): Promise<Trade[]> {
  const trades = await fetchTrades()
  return [...trades]
    .sort((a, b) => Math.abs(b.pnl) - Math.abs(a.pnl))
    .slice(0, limit)
}
