import { create } from 'zustand'
import type {
  Genome,
  Hypothesis,
  Shadow,
  Alert,
  Island,
  WsMessage,
  RegimeChangePayload,
  EvolutionTickPayload,
} from '../types'

// ─── State Shape ──────────────────────────────────────────────────────────────

interface IdeaState {
  // Data
  genomes: Genome[]
  hypotheses: Hypothesis[]
  shadows: Shadow[]
  alerts: Alert[]

  // Regime
  currentRegime: Island
  regimeConfidence: number
  previousRegime: Island | null

  // Evolution
  evolutionGeneration: number
  evolutionStats: Record<Island, { generation: number; bestFitness: number; meanFitness: number; diversityIndex: number }>

  // UI State
  wsConnected: boolean
  lastWsMessage: WsMessage | null
  selectedIsland: Island | 'HOF'
  selectedGenomeId: number | null
  sidebarCollapsed: boolean

  // Actions
  setGenomes: (genomes: Genome[]) => void
  addGenome: (genome: Genome) => void
  updateGenome: (id: number, updates: Partial<Genome>) => void

  addHypothesis: (hyp: Hypothesis) => void
  updateHypothesis: (id: number, updates: Partial<Hypothesis>) => void
  setHypotheses: (hyps: Hypothesis[]) => void

  updateShadow: (shadowId: number, updates: Partial<Shadow>) => void
  setShadows: (shadows: Shadow[]) => void

  setAlerts: (alerts: Alert[]) => void
  acknowledgeAlert: (id: number) => void
  addAlert: (alert: Alert) => void

  setRegime: (regime: Island, confidence?: number) => void
  setWsConnected: (connected: boolean) => void
  setSelectedIsland: (island: Island | 'HOF') => void
  setSelectedGenomeId: (id: number | null) => void
  setSidebarCollapsed: (collapsed: boolean) => void

  handleWsMessage: (msg: WsMessage) => void
}

// ─── Store ────────────────────────────────────────────────────────────────────

export const useIdeaStore = create<IdeaState>((set, get) => ({
  // Initial state
  genomes: [],
  hypotheses: [],
  shadows: [],
  alerts: [],

  currentRegime: 'NEUTRAL',
  regimeConfidence: 0.65,
  previousRegime: null,

  evolutionGeneration: 0,
  evolutionStats: {
    BULL: { generation: 0, bestFitness: 0, meanFitness: 0, diversityIndex: 0 },
    BEAR: { generation: 0, bestFitness: 0, meanFitness: 0, diversityIndex: 0 },
    NEUTRAL: { generation: 0, bestFitness: 0, meanFitness: 0, diversityIndex: 0 },
  },

  wsConnected: false,
  lastWsMessage: null,
  selectedIsland: 'BULL',
  selectedGenomeId: null,
  sidebarCollapsed: false,

  // Genome actions
  setGenomes: (genomes) => set({ genomes }),
  addGenome: (genome) =>
    set((s) => ({ genomes: [...s.genomes, genome] })),
  updateGenome: (id, updates) =>
    set((s) => ({
      genomes: s.genomes.map((g) => (g.id === id ? { ...g, ...updates } : g)),
    })),

  // Hypothesis actions
  setHypotheses: (hypotheses) => set({ hypotheses }),
  addHypothesis: (hyp) =>
    set((s) => ({ hypotheses: [hyp, ...s.hypotheses] })),
  updateHypothesis: (id, updates) =>
    set((s) => ({
      hypotheses: s.hypotheses.map((h) => (h.id === id ? { ...h, ...updates } : h)),
    })),

  // Shadow actions
  setShadows: (shadows) => set({ shadows }),
  updateShadow: (shadowId, updates) =>
    set((s) => ({
      shadows: s.shadows.map((sh) =>
        sh.shadowId === shadowId ? { ...sh, ...updates } : sh
      ),
    })),

  // Alert actions
  setAlerts: (alerts) => set({ alerts }),
  addAlert: (alert) =>
    set((s) => ({ alerts: [alert, ...s.alerts].slice(0, 100) })),
  acknowledgeAlert: (id) =>
    set((s) => ({
      alerts: s.alerts.map((a) =>
        a.id === id ? { ...a, acknowledged: true } : a
      ),
    })),

  // Regime
  setRegime: (regime, confidence) =>
    set((s) => ({
      previousRegime: s.currentRegime,
      currentRegime: regime,
      regimeConfidence: confidence ?? s.regimeConfidence,
    })),

  // UI
  setWsConnected: (wsConnected) => set({ wsConnected }),
  setSelectedIsland: (selectedIsland) => set({ selectedIsland }),
  setSelectedGenomeId: (selectedGenomeId) => set({ selectedGenomeId }),
  setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),

  // WebSocket message handler
  handleWsMessage: (msg) => {
    set({ lastWsMessage: msg })
    const { type, payload } = msg
    const state = get()

    switch (type) {
      case 'genome_update': {
        const genome = payload as Genome
        const existing = state.genomes.find((g) => g.id === genome.id)
        if (existing) {
          state.updateGenome(genome.id, genome)
        } else {
          state.addGenome(genome)
        }
        break
      }
      case 'hypothesis_update': {
        const hyp = payload as Hypothesis
        const existing = state.hypotheses.find((h) => h.id === hyp.id)
        if (existing) {
          state.updateHypothesis(hyp.id, hyp)
        } else {
          state.addHypothesis(hyp)
        }
        break
      }
      case 'shadow_update': {
        const shadow = payload as Shadow
        const existing = state.shadows.find((s) => s.shadowId === shadow.shadowId)
        if (existing) {
          state.updateShadow(shadow.shadowId, shadow)
        } else {
          set((s) => ({ shadows: [...s.shadows, shadow] }))
        }
        break
      }
      case 'alert': {
        state.addAlert(payload as Alert)
        break
      }
      case 'regime_change': {
        const rcp = payload as RegimeChangePayload
        state.setRegime(rcp.regime, rcp.confidence)
        state.addAlert({
          id: Date.now(),
          type: 'regime',
          severity: 'critical',
          message: `Regime change: ${rcp.previousRegime} → ${rcp.regime} (confidence: ${rcp.confidence.toFixed(2)})`,
          acknowledged: false,
          createdAt: msg.timestamp,
        })
        break
      }
      case 'evolution_tick': {
        const tick = payload as EvolutionTickPayload
        set((s) => ({
          evolutionGeneration: Math.max(s.evolutionGeneration, tick.generation),
          evolutionStats: {
            ...s.evolutionStats,
            [tick.island]: {
              generation: tick.generation,
              bestFitness: tick.bestFitness,
              meanFitness: tick.meanFitness,
              diversityIndex: tick.diversityIndex,
            },
          },
        }))
        break
      }
      default:
        break
    }
  },
}))

// ─── Selectors ────────────────────────────────────────────────────────────────

export const selectUnacknowledgedAlerts = (s: IdeaState) =>
  s.alerts.filter((a) => !a.acknowledged)

export const selectCriticalAlerts = (s: IdeaState) =>
  s.alerts.filter((a) => a.severity === 'critical' && !a.acknowledged)

export const selectHallOfFameGenomes = (s: IdeaState) =>
  s.genomes.filter((g) => g.isHallOfFame).sort((a, b) => b.sharpe - a.sharpe)

export const selectTopGenomes = (n: number) => (s: IdeaState) =>
  [...s.genomes].sort((a, b) => b.sharpe - a.sharpe).slice(0, n)

export const selectGenomesByIsland = (island: Island) => (s: IdeaState) =>
  s.genomes.filter((g) => g.island === island)

export const selectPendingHypotheses = (s: IdeaState) =>
  s.hypotheses.filter((h) => h.status === 'pending')

export const selectPositiveAlphaShadows = (s: IdeaState) =>
  s.shadows.filter((sh) => sh.alpha > 0 && !sh.promoted)
