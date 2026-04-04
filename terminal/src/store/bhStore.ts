// ============================================================
// BH PHYSICS STORE — Zustand + Immer
// ============================================================
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { subscribeWithSelector } from 'zustand/middleware'
import type {
  LiveState,
  InstrumentBHState,
  BHState,
  BHMassPoint,
  BHFormationEvent,
  BHHistoryRecord,
  BHScanResult,
  BHTimeframe,
  BHRegime,
} from '@/types'

interface BHStoreState {
  liveState: LiveState | null
  instruments: Record<string, InstrumentBHState>
  history: Record<string, BHHistoryRecord>         // symbol -> history
  formationEvents: BHFormationEvent[]
  scanResults: BHScanResult[]

  isConnected: boolean
  lastUpdate: number | null
  activeFormations: string[]
  selectedInstrument: string | null
  highlightedFormation: string | null

  scanFilter: {
    activeOnly: boolean
    minMass: number
    regimes: BHRegime[]
    timeframes: BHTimeframe[]
  }
}

interface BHStoreActions {
  // Live State Updates
  updateLiveState(state: LiveState): void
  updateInstrumentState(symbol: string, state: InstrumentBHState): void
  updateTFState(symbol: string, tf: BHTimeframe, state: BHState): void

  // History
  appendMassPoint(symbol: string, point: BHMassPoint): void
  setHistory(symbol: string, points: BHMassPoint[], events: BHFormationEvent[]): void
  clearHistory(symbol: string): void

  // Formations
  addFormationEvent(event: BHFormationEvent): void
  acknowledgeFormation(eventId: string): void
  clearFormations(): void

  // Scan
  setScanResults(results: BHScanResult[]): void
  setScanFilter(filter: Partial<BHStoreState['scanFilter']>): void

  // UI
  setSelectedInstrument(symbol: string | null): void
  setHighlightedFormation(eventId: string | null): void
  setConnected(connected: boolean): void
  setLastUpdate(ts: number): void
}

type BHStore = BHStoreState & BHStoreActions

const MAX_HISTORY_POINTS = 1000
const MAX_FORMATION_EVENTS = 500

export const useBHStore = create<BHStore>()(
  subscribeWithSelector(
    immer((set) => ({
      // ---- Initial State ----
      liveState: null,
      instruments: {},
      history: {},
      formationEvents: [],
      scanResults: [],
      isConnected: false,
      lastUpdate: null,
      activeFormations: [],
      selectedInstrument: null,
      highlightedFormation: null,
      scanFilter: {
        activeOnly: false,
        minMass: 0,
        regimes: ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOL'],
        timeframes: ['15m', '1h', '1d'],
      },

      // ---- Live State Updates ----
      updateLiveState(liveState: LiveState) {
        set((state) => {
          state.liveState = liveState
          state.lastUpdate = Date.now()

          // Update individual instruments
          for (const [symbol, instrState] of Object.entries(liveState.instruments)) {
            state.instruments[symbol] = instrState
          }

          // Track active formations
          state.activeFormations = []
          for (const [symbol, instr] of Object.entries(liveState.instruments)) {
            if (instr.tf15m.active || instr.tf1h.active || instr.tf1d.active) {
              state.activeFormations.push(symbol)
            }
          }
        })
      },

      updateInstrumentState(symbol: string, instrState: InstrumentBHState) {
        set((state) => {
          state.instruments[symbol] = instrState
          state.lastUpdate = Date.now()

          // Update active formations list
          const idx = state.activeFormations.indexOf(symbol)
          const isActive = instrState.tf15m.active || instrState.tf1h.active || instrState.tf1d.active
          if (isActive && idx === -1) {
            state.activeFormations.push(symbol)
          } else if (!isActive && idx !== -1) {
            state.activeFormations.splice(idx, 1)
          }
        })
      },

      updateTFState(symbol: string, tf: BHTimeframe, bhState: BHState) {
        set((state) => {
          if (!state.instruments[symbol]) return
          const instr = state.instruments[symbol]
          switch (tf) {
            case '15m': instr.tf15m = bhState; break
            case '1h': instr.tf1h = bhState; break
            case '1d': instr.tf1d = bhState; break
          }
          state.lastUpdate = Date.now()
        })
      },

      // ---- History ----
      appendMassPoint(symbol: string, point: BHMassPoint) {
        set((state) => {
          if (!state.history[symbol]) {
            state.history[symbol] = {
              symbol,
              points: [],
              formationEvents: [],
              lastUpdated: Date.now(),
            }
          }
          state.history[symbol].points.push(point)
          if (state.history[symbol].points.length > MAX_HISTORY_POINTS) {
            state.history[symbol].points.splice(0, state.history[symbol].points.length - MAX_HISTORY_POINTS)
          }
          state.history[symbol].lastUpdated = Date.now()
        })
      },

      setHistory(symbol: string, points: BHMassPoint[], events: BHFormationEvent[]) {
        set((state) => {
          state.history[symbol] = {
            symbol,
            points: points.slice(-MAX_HISTORY_POINTS),
            formationEvents: events,
            lastUpdated: Date.now(),
          }
        })
      },

      clearHistory(symbol: string) {
        set((state) => {
          delete state.history[symbol]
        })
      },

      // ---- Formations ----
      addFormationEvent(event: BHFormationEvent) {
        set((state) => {
          state.formationEvents.unshift(event)
          if (state.formationEvents.length > MAX_FORMATION_EVENTS) {
            state.formationEvents.splice(MAX_FORMATION_EVENTS)
          }
          // Add to instrument history
          if (state.history[event.symbol]) {
            state.history[event.symbol].formationEvents.unshift(event)
          }
        })
      },

      acknowledgeFormation(eventId: string) {
        set((state) => {
          const event = state.formationEvents.find((e) => e.id === eventId)
          if (event) event.acknowledged = true
        })
      },

      clearFormations() {
        set((state) => {
          state.formationEvents = []
        })
      },

      // ---- Scan ----
      setScanResults(results: BHScanResult[]) {
        set((state) => {
          state.scanResults = results
        })
      },

      setScanFilter(filter: Partial<BHStoreState['scanFilter']>) {
        set((state) => {
          Object.assign(state.scanFilter, filter)
        })
      },

      // ---- UI ----
      setSelectedInstrument(symbol: string | null) {
        set((state) => {
          state.selectedInstrument = symbol
        })
      },

      setHighlightedFormation(eventId: string | null) {
        set((state) => {
          state.highlightedFormation = eventId
        })
      },

      setConnected(connected: boolean) {
        set((state) => {
          state.isConnected = connected
        })
      },

      setLastUpdate(ts: number) {
        set((state) => {
          state.lastUpdate = ts
        })
      },
    }))
  )
)

// ---- Selectors ----
export const selectInstrument = (symbol: string) => (state: BHStore) =>
  state.instruments[symbol] ?? null

export const selectInstrumentHistory = (symbol: string) => (state: BHStore) =>
  state.history[symbol] ?? null

export const selectFilteredScanResults = (state: BHStore) => {
  const { scanResults, scanFilter } = state
  return scanResults.filter((r) => {
    if (scanFilter.activeOnly) {
      const instr = r.state
      if (!instr.tf15m.active && !instr.tf1h.active && !instr.tf1d.active) return false
    }
    const maxMass = Math.max(r.state.tf15m.mass, r.state.tf1h.mass, r.state.tf1d.mass)
    if (maxMass < scanFilter.minMass) return false
    return true
  })
}

export const selectUnacknowledgedFormations = (state: BHStore) =>
  state.formationEvents.filter((e) => !e.acknowledged)

export const selectBHGaugeData = (symbol: string) => (state: BHStore) => {
  const instr = state.instruments[symbol]
  if (!instr) return null

  const getColor = (bhState: BHState) => {
    if (!bhState.active) return '#6b7280'
    switch (bhState.regime) {
      case 'BULL': return '#22c55e'
      case 'BEAR': return '#ef4444'
      case 'HIGH_VOL': return '#f59e0b'
      default: return '#6b7280'
    }
  }

  return {
    tf15m: {
      timeframe: '15m' as BHTimeframe,
      ...instr.tf15m,
      maxMass: 3,
      formationActive: instr.tf15m.bh_form > 0,
      color: getColor(instr.tf15m),
      label: '15m',
    },
    tf1h: {
      timeframe: '1h' as BHTimeframe,
      ...instr.tf1h,
      maxMass: 3,
      formationActive: instr.tf1h.bh_form > 0,
      color: getColor(instr.tf1h),
      label: '1h',
    },
    tf1d: {
      timeframe: '1d' as BHTimeframe,
      ...instr.tf1d,
      maxMass: 3,
      formationActive: instr.tf1d.bh_form > 0,
      color: getColor(instr.tf1d),
      label: '1d',
    },
  }
}
