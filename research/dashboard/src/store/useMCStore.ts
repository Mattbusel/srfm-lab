import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import type { MCResults, MCSimParams } from '@/types/mc'

interface MCState {
  results: MCResults | null
  params: MCSimParams
  loading: boolean
  error: string | null
  lastRunAt: string | null

  setResults: (results: MCResults) => void
  setParams: (params: Partial<MCSimParams>) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

const DEFAULT_PARAMS: MCSimParams = {
  nPaths: 10_000,
  nDays: 252,
  initialEquity: 100_000,
  useHistoricalReturns: true,
  regimeWeighting: true,
}

export const useMCStore = create<MCState>()(
  immer((set) => ({
    results: null,
    params: DEFAULT_PARAMS,
    loading: false,
    error: null,
    lastRunAt: null,

    setResults: (results) =>
      set((state) => {
        state.results = results
        state.lastRunAt = new Date().toISOString()
      }),

    setParams: (params) =>
      set((state) => {
        state.params = { ...state.params, ...params }
      }),

    setLoading: (loading) =>
      set((state) => { state.loading = loading }),

    setError: (error) =>
      set((state) => { state.error = error }),
  }))
)
