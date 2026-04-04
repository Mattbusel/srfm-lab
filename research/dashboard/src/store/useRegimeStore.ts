import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import type { RegimeSegment, TransitionMatrix, RegimePerformance, StressScenario } from '@/types/regimes'
import type { RegimeType } from '@/types/trades'

interface RegimeState {
  segments: RegimeSegment[]
  transitionMatrix: TransitionMatrix | null
  performance: RegimePerformance[]
  stressScenarios: StressScenario[]
  activeRegime: RegimeType | null
  selectedRegimes: RegimeType[]
  loading: boolean
  error: string | null

  setSegments: (segments: RegimeSegment[]) => void
  setTransitionMatrix: (matrix: TransitionMatrix) => void
  setPerformance: (perf: RegimePerformance[]) => void
  setStressScenarios: (scenarios: StressScenario[]) => void
  setActiveRegime: (regime: RegimeType | null) => void
  toggleRegime: (regime: RegimeType) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export const useRegimeStore = create<RegimeState>()(
  immer((set) => ({
    segments: [],
    transitionMatrix: null,
    performance: [],
    stressScenarios: [],
    activeRegime: null,
    selectedRegimes: ['bull', 'bear', 'sideways', 'ranging', 'volatile'],
    loading: false,
    error: null,

    setSegments: (segments) =>
      set((state) => { state.segments = segments }),

    setTransitionMatrix: (matrix) =>
      set((state) => { state.transitionMatrix = matrix }),

    setPerformance: (perf) =>
      set((state) => { state.performance = perf }),

    setStressScenarios: (scenarios) =>
      set((state) => { state.stressScenarios = scenarios }),

    setActiveRegime: (regime) =>
      set((state) => { state.activeRegime = regime }),

    toggleRegime: (regime) =>
      set((state) => {
        const idx = state.selectedRegimes.indexOf(regime)
        if (idx >= 0) {
          state.selectedRegimes.splice(idx, 1)
        } else {
          state.selectedRegimes.push(regime)
        }
      }),

    setLoading: (loading) =>
      set((state) => { state.loading = loading }),

    setError: (error) =>
      set((state) => { state.error = error }),
  }))
)
