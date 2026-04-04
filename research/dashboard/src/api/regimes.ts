import { api } from './client'
import type { RegimeSegment, TransitionMatrix, RegimePerformance, RegimeDuration, StressScenario } from '@/types/regimes'
import {
  generateMockRegimeSegments,
  generateMockTransitionMatrix,
  generateMockRegimePerformance,
  generateMockRegimeDurations,
  generateMockStressScenarios,
} from './mockData'

const USE_MOCK = true

export async function fetchRegimeSegments(days = 180): Promise<RegimeSegment[]> {
  if (USE_MOCK) return generateMockRegimeSegments(days)
  return api.get<RegimeSegment[]>('/api/regimes/segments', { days })
}

export async function fetchTransitionMatrix(): Promise<TransitionMatrix> {
  if (USE_MOCK) return generateMockTransitionMatrix()
  return api.get<TransitionMatrix>('/api/regimes/transitions')
}

export async function fetchRegimePerformance(): Promise<RegimePerformance[]> {
  if (USE_MOCK) return generateMockRegimePerformance()
  return api.get<RegimePerformance[]>('/api/regimes/performance')
}

export async function fetchRegimeDurations(): Promise<RegimeDuration[]> {
  if (USE_MOCK) return generateMockRegimeDurations()
  return api.get<RegimeDuration[]>('/api/regimes/durations')
}

export async function fetchStressScenarios(): Promise<StressScenario[]> {
  if (USE_MOCK) return generateMockStressScenarios()
  return api.get<StressScenario[]>('/api/regimes/stress')
}
