import { api } from './client'
import type { MCResults, MCSimParams } from '@/types/mc'
import { generateMockMCResults } from './mockData'

const USE_MOCK = true

export async function fetchMCResults(params?: Partial<MCSimParams>): Promise<MCResults> {
  if (USE_MOCK) return generateMockMCResults(params?.nDays ?? 252, params?.initialEquity ?? 100_000)
  return api.post<MCResults>('/api/mc/run', params ?? {})
}

export async function fetchMCBenchmark(): Promise<MCResults> {
  if (USE_MOCK) return generateMockMCResults(252, 100_000)
  return api.get<MCResults>('/api/mc/benchmark')
}
