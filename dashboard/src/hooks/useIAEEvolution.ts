// ============================================================
// hooks/useIAEEvolution.ts -- React Query hooks for IAE evolution data
// All hooks poll the IAE API at :8795 with 10s refetch interval.
// ============================================================

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import type {
  GenerationStats,
  GenomeRecord,
  ParameterHistoryPoint,
  RollbackEvent,
  GenomePedigree,
  IaeEvolutionPayload,
  IaeEvolutionConfig,
} from '../types/iae'

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const IAE_API_BASE = 'http://localhost:8795'
const POLL_FAST_MS = 10_000
const POLL_MEDIUM_MS = 30_000

// ---------------------------------------------------------------------------
// Mock data generators -- realistic shapes
// ---------------------------------------------------------------------------

function genId(): string {
  return Math.random().toString(36).slice(2, 10)
}

function mockGenerationStats(gen = 42): GenerationStats {
  const base = 0.6 + gen * 0.004 + (Math.random() - 0.5) * 0.05
  return {
    generation: gen,
    population_size: 64,
    best_fitness: base + 0.08 + Math.random() * 0.05,
    mean_fitness: base + Math.random() * 0.03,
    worst_fitness: base - 0.15 - Math.random() * 0.05,
    diversity_index: Math.max(0.1, 0.7 - gen * 0.008 + Math.random() * 0.1),
    mutation_rate: 0.12 + Math.sin(gen * 0.3) * 0.04,
    crossover_rate: 0.72,
    elite_count: 4,
    stagnation_counter: Math.floor(Math.random() * 5),
    best_genome_id: genId(),
    timestamp: new Date().toISOString(),
  }
}

function mockGenerationHistory(n = 50): GenerationStats[] {
  return Array.from({ length: n }, (_, i) => mockGenerationStats(i + 1))
}

function mockGenome(gen: number, rank: number): GenomeRecord {
  const fitness = 0.85 - rank * 0.03 + (Math.random() - 0.5) * 0.01
  return {
    id: genId(),
    generation: gen,
    fitness,
    sharpe: 1.8 + (1 - rank * 0.08) + Math.random() * 0.2,
    max_drawdown: -(0.08 + rank * 0.01 + Math.random() * 0.02),
    total_return: 0.35 - rank * 0.02 + Math.random() * 0.05,
    win_rate: 0.62 - rank * 0.01 + Math.random() * 0.02,
    calmar: 3.2 - rank * 0.18 + Math.random() * 0.3,
    sortino: 2.4 - rank * 0.12 + Math.random() * 0.2,
    genes: {
      BH_MASS_THRESH: 0.45 + Math.random() * 0.3,
      BH_DECAY: 0.88 + Math.random() * 0.08,
      BH_COLLAPSE: 0.2 + Math.random() * 0.2,
      NAV_OMEGA_SCALE_K: 1.1 + Math.random() * 0.6,
      NAV_GEO_ENTRY_GATE: 0.55 + Math.random() * 0.25,
      HURST_WINDOW: Math.round(80 + Math.random() * 60),
      GARCH_ALPHA: 0.06 + Math.random() * 0.06,
      GARCH_BETA: 0.88 + Math.random() * 0.08,
      MIN_HOLD_BARS: Math.round(2 + Math.random() * 6),
      MAX_POSITION_SIZE: 0.08 + Math.random() * 0.07,
      ENTRY_ZSCORE: 1.6 + Math.random() * 0.8,
      EXIT_ZSCORE: 0.3 + Math.random() * 0.5,
      // extra genes filling up to 31
      VOL_LOOKBACK: Math.round(20 + Math.random() * 20),
      REGIME_THRESHOLD: 0.5 + Math.random() * 0.3,
      SKEW_WEIGHT: 0.1 + Math.random() * 0.4,
      KURT_WEIGHT: 0.1 + Math.random() * 0.3,
      MOMENTUM_WINDOW: Math.round(10 + Math.random() * 20),
      MEAN_REVERSION_K: 0.5 + Math.random() * 0.8,
      CARRY_WEIGHT: 0.05 + Math.random() * 0.2,
      LIQUIDITY_FILTER: 0.3 + Math.random() * 0.5,
      SIGNAL_DECAY: 0.7 + Math.random() * 0.2,
      COOLDOWN_BARS: Math.round(3 + Math.random() * 10),
      PARTIAL_EXIT_FRAC: 0.3 + Math.random() * 0.4,
      TRAIL_STOP_ATR: 1.5 + Math.random() * 1.5,
      ENTRY_CONFIRMATION: Math.round(1 + Math.random() * 3),
      REGIME_HALFLIFE: Math.round(20 + Math.random() * 40),
      SPREAD_FILTER: 0.001 + Math.random() * 0.003,
      FUNDING_THRESHOLD: 0.0005 + Math.random() * 0.001,
      DELTA_HEDGE_RATIO: 0.8 + Math.random() * 0.15,
      RISK_SCALE: 0.7 + Math.random() * 0.5,
      ALPHA_HALFLIFE: Math.round(5 + Math.random() * 20),
    },
    parent_ids: rank === 0 ? [] : [genId(), genId()],
    operator: rank === 0 ? 'elite' : rank < 3 ? 'crossover' : 'mutation',
    created_at: new Date(Date.now() - rank * 60_000).toISOString(),
    is_active: rank === 0,
    rank,
  }
}

function mockTopGenomes(n: number): GenomeRecord[] {
  return Array.from({ length: n }, (_, i) => mockGenome(42, i))
}

function mockParameterHistory(param: string, hours: number): ParameterHistoryPoint[] {
  const now = Date.now()
  const pts = hours * 4 // one point per 15 min
  let val = 0.5 + Math.random() * 0.5
  return Array.from({ length: pts }, (_, i) => {
    val += (Math.random() - 0.5) * 0.02
    return {
      timestamp: new Date(now - (pts - i) * 15 * 60_000).toISOString(),
      value: val,
      fitness_at_time: 0.7 + Math.random() * 0.2,
    }
  })
}

function mockRollbacks(n: number): RollbackEvent[] {
  const reasons: RollbackEvent['reason'][] = [
    'sharpe_degradation',
    'drawdown_breach',
    'fitness_regression',
    'manual_override',
    'circuit_breaker',
  ]
  return Array.from({ length: n }, (_, i) => ({
    id: genId(),
    timestamp: new Date(Date.now() - i * 4 * 3600_000).toISOString(),
    reason: reasons[i % reasons.length],
    reason_detail: `Automatic rollback triggered: ${reasons[i % reasons.length].replace(/_/g, ' ')}`,
    from_generation: 42 - i * 2,
    to_generation: 42 - i * 2 - 3,
    parameter_delta: {
      BH_MASS_THRESH: (Math.random() - 0.5) * 0.1,
      ENTRY_ZSCORE: (Math.random() - 0.5) * 0.3,
      GARCH_ALPHA: (Math.random() - 0.5) * 0.02,
    },
    sharpe_before: 1.2 + Math.random() * 0.3,
    sharpe_after: 1.6 + Math.random() * 0.4,
    fitness_before: 0.55 + Math.random() * 0.1,
    fitness_after: 0.72 + Math.random() * 0.1,
    initiated_by: i === 1 ? 'manual' : 'automatic',
    operator_id: i === 1 ? 'admin' : undefined,
  }))
}

function mockPedigree(root_id: string): GenomePedigree {
  const ids = Array.from({ length: 7 }, () => genId())
  return {
    root_id,
    nodes: [
      // generation n-2 (2 nodes)
      { genome_id: ids[0], generation: 40, fitness: 0.71, sharpe: 1.65, operator: 'elite', parent_ids: [], children_ids: [ids[2], ids[3]] },
      { genome_id: ids[1], generation: 40, fitness: 0.68, sharpe: 1.58, operator: 'mutation', parent_ids: [], children_ids: [ids[2]] },
      // generation n-1 (3 nodes)
      { genome_id: ids[2], generation: 41, fitness: 0.77, sharpe: 1.82, operator: 'crossover', parent_ids: [ids[0], ids[1]], children_ids: [ids[5], ids[6]] },
      { genome_id: ids[3], generation: 41, fitness: 0.74, sharpe: 1.74, operator: 'crossover', parent_ids: [ids[0]], children_ids: [ids[4]] },
      { genome_id: ids[4], generation: 41, fitness: 0.70, sharpe: 1.61, operator: 'mutation', parent_ids: [ids[3]], children_ids: [] },
      // generation n (2 nodes)
      { genome_id: ids[5], generation: 42, fitness: 0.83, sharpe: 1.95, operator: 'crossover', parent_ids: [ids[2]], children_ids: [] },
      { genome_id: root_id, generation: 42, fitness: 0.87, sharpe: 2.08, operator: 'elite', parent_ids: [ids[2]], children_ids: [] },
    ],
    edges: [
      { from: ids[0], to: ids[2] },
      { from: ids[1], to: ids[2] },
      { from: ids[0], to: ids[3] },
      { from: ids[3], to: ids[4] },
      { from: ids[2], to: ids[5] },
      { from: ids[2], to: root_id },
    ],
    depth: 3,
  }
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function fetchJson<T>(url: string, mock: () => T): Promise<T> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(5_000) })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return (await res.json()) as T
  } catch {
    // Fall back to mock data when API is unreachable (dev/demo mode)
    return mock()
  }
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

/**
 * useGenerationStats -- polls /api/iae/generation/stats every 10s.
 * Returns current generation stats plus 50-generation history for charts.
 */
export function useGenerationStats() {
  return useQuery({
    queryKey: ['iae', 'generation', 'stats'],
    queryFn: () =>
      fetchJson<{ current: GenerationStats; history: GenerationStats[] }>(
        `${IAE_API_BASE}/api/iae/generation/stats`,
        () => ({ current: mockGenerationStats(42), history: mockGenerationHistory(50) }),
      ),
    refetchInterval: POLL_FAST_MS,
    staleTime: POLL_FAST_MS / 2,
  })
}

/**
 * useTopGenomes -- fetches top N genomes from /api/iae/genomes/top
 */
export function useTopGenomes(n: number = 10) {
  return useQuery({
    queryKey: ['iae', 'genomes', 'top', n],
    queryFn: () =>
      fetchJson<GenomeRecord[]>(
        `${IAE_API_BASE}/api/iae/genomes/top?n=${n}`,
        () => mockTopGenomes(n),
      ),
    refetchInterval: POLL_FAST_MS,
    staleTime: POLL_FAST_MS / 2,
  })
}

/**
 * useParameterHistory -- fetches value history for a single parameter
 */
export function useParameterHistory(param: string, hours: number = 24) {
  return useQuery({
    queryKey: ['iae', 'params', param, 'history', hours],
    queryFn: () =>
      fetchJson<ParameterHistoryPoint[]>(
        `${IAE_API_BASE}/api/iae/params/${param}/history?hours=${hours}`,
        () => mockParameterHistory(param, hours),
      ),
    refetchInterval: POLL_MEDIUM_MS,
    staleTime: POLL_MEDIUM_MS / 2,
    enabled: !!param,
  })
}

/**
 * useRollbackHistory -- fetches recent rollback events
 */
export function useRollbackHistory(n: number = 10) {
  return useQuery({
    queryKey: ['iae', 'rollbacks', n],
    queryFn: () =>
      fetchJson<RollbackEvent[]>(
        `${IAE_API_BASE}/api/iae/rollbacks?n=${n}`,
        () => mockRollbacks(n),
      ),
    refetchInterval: POLL_MEDIUM_MS,
    staleTime: POLL_MEDIUM_MS / 2,
  })
}

/**
 * useGenomePedigree -- fetches parent-child tree for a specific genome
 */
export function useGenomePedigree(genome_id: string) {
  return useQuery({
    queryKey: ['iae', 'genome', genome_id, 'pedigree'],
    queryFn: () =>
      fetchJson<GenomePedigree>(
        `${IAE_API_BASE}/api/iae/genome/${genome_id}/pedigree`,
        () => mockPedigree(genome_id),
      ),
    refetchInterval: POLL_FAST_MS,
    staleTime: POLL_FAST_MS / 2,
    enabled: !!genome_id,
  })
}

/**
 * useIAEEvolution -- master hook fetching the combined evolution payload.
 * Used by IAEEvolution.tsx as the primary data source.
 */
export function useIAEEvolution() {
  return useQuery({
    queryKey: ['iae', 'evolution'],
    queryFn: () =>
      fetchJson<IaeEvolutionPayload>(`${IAE_API_BASE}/api/iae/evolution`, () => {
        const history = mockGenerationHistory(50)
        return {
          current_stats: history[history.length - 1],
          history,
          top_genomes: mockTopGenomes(10),
          parameters: [],        // populated by IAEEvolution page from separate query
          recent_rollbacks: mockRollbacks(8),
          config: {
            population_size: 64,
            max_generations: 200,
            mutation_sigma: 0.05,
            crossover_prob: 0.72,
            elite_fraction: 0.0625,
            tournament_size: 5,
            diversity_pressure: 0.15,
            fitness_window_bars: 252,
            rollback_sharpe_threshold: 0.3,
            rollback_dd_threshold: 0.15,
            running: true,
            paused: false,
          },
          fetched_at: new Date().toISOString(),
        }
      }),
    refetchInterval: POLL_FAST_MS,
    staleTime: POLL_FAST_MS / 2,
  })
}

/**
 * usePauseEvolution -- mutation to pause/resume the IAE engine
 */
export function usePauseEvolution() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (pause: boolean) => {
      const res = await fetch(`${IAE_API_BASE}/api/iae/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: pause ? 'pause' : 'resume' }),
      })
      if (!res.ok) throw new Error('Control request failed')
      return res.json()
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['iae'] })
    },
  })
}
