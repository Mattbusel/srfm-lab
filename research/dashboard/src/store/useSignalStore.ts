import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import type { SignalSnapshot, ICPoint, RollingICPoint } from '@/types/signals'

interface SignalState {
  snapshots: SignalSnapshot[]
  icDecay: ICPoint[]
  rollingIC: RollingICPoint[]
  selectedInstrument: string | null
  loading: boolean
  error: string | null
  lastUpdated: string | null

  setSnapshots: (snapshots: SignalSnapshot[]) => void
  setICDecay: (decay: ICPoint[]) => void
  setRollingIC: (rolling: RollingICPoint[]) => void
  setSelectedInstrument: (instrument: string | null) => void
  updateSnapshot: (snapshot: SignalSnapshot) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export const useSignalStore = create<SignalState>()(
  immer((set) => ({
    snapshots: [],
    icDecay: [],
    rollingIC: [],
    selectedInstrument: null,
    loading: false,
    error: null,
    lastUpdated: null,

    setSnapshots: (snapshots) =>
      set((state) => {
        state.snapshots = snapshots
        state.lastUpdated = new Date().toISOString()
      }),

    setICDecay: (decay) =>
      set((state) => { state.icDecay = decay }),

    setRollingIC: (rolling) =>
      set((state) => { state.rollingIC = rolling }),

    setSelectedInstrument: (instrument) =>
      set((state) => { state.selectedInstrument = instrument }),

    updateSnapshot: (snapshot) =>
      set((state) => {
        const idx = state.snapshots.findIndex(s => s.instrument === snapshot.instrument)
        if (idx >= 0) {
          state.snapshots[idx] = snapshot
        } else {
          state.snapshots.push(snapshot)
        }
        state.lastUpdated = new Date().toISOString()
      }),

    setLoading: (loading) =>
      set((state) => { state.loading = loading }),

    setError: (error) =>
      set((state) => { state.error = error }),
  }))
)
