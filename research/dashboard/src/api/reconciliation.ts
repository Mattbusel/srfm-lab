import { api } from './client'
import type { ReconciliationRow, SlippageStats } from '@/types/trades'
import { generateMockReconciliation, generateMockSlippageStats } from './mockData'

const USE_MOCK = true

export async function fetchReconciliation(dateFrom?: string, dateTo?: string): Promise<ReconciliationRow[]> {
  if (USE_MOCK) return generateMockReconciliation(50)
  return api.get<ReconciliationRow[]>('/api/reconciliation', { dateFrom, dateTo })
}

export async function fetchSlippageStats(): Promise<SlippageStats[]> {
  if (USE_MOCK) return generateMockSlippageStats()
  return api.get<SlippageStats[]>('/api/slippage/stats')
}
