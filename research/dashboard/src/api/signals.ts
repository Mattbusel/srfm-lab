import { api } from './client'
import type { SignalSnapshot, ICPoint, RollingICPoint, ICByRegime, FactorAttribution, QuintileReturn } from '@/types/signals'
import {
  generateMockSignals,
  generateMockICDecay,
  generateMockRollingIC,
  generateMockICByRegime,
  generateMockFactorAttribution,
  generateMockQuintileReturns,
} from './mockData'

const USE_MOCK = true

export async function fetchSignalSnapshots(): Promise<SignalSnapshot[]> {
  if (USE_MOCK) return generateMockSignals()
  return api.get<SignalSnapshot[]>('/api/signals/snapshot')
}

export async function fetchICDecay(instrument?: string): Promise<ICPoint[]> {
  if (USE_MOCK) return generateMockICDecay()
  return api.get<ICPoint[]>('/api/signals/ic-decay', { instrument })
}

export async function fetchRollingIC(days = 120, instrument?: string): Promise<RollingICPoint[]> {
  if (USE_MOCK) return generateMockRollingIC(days)
  return api.get<RollingICPoint[]>('/api/signals/rolling-ic', { days, instrument })
}

export async function fetchICByRegime(instrument?: string): Promise<ICByRegime[]> {
  if (USE_MOCK) return generateMockICByRegime()
  return api.get<ICByRegime[]>('/api/signals/ic-by-regime', { instrument })
}

export async function fetchFactorAttribution(dateFrom?: string, dateTo?: string): Promise<FactorAttribution[]> {
  if (USE_MOCK) return generateMockFactorAttribution()
  return api.get<FactorAttribution[]>('/api/signals/attribution', { dateFrom, dateTo })
}

export async function fetchQuintileReturns(instrument?: string): Promise<QuintileReturn[]> {
  if (USE_MOCK) return generateMockQuintileReturns()
  return api.get<QuintileReturn[]>('/api/signals/quintiles', { instrument })
}
