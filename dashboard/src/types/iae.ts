// ============================================================
// types/iae.ts -- TypeScript types for IAE (Iterative Adaptation Engine)
// data structures: genomes, generation stats, parameters, rollbacks
// ============================================================

// ---- Genome & Evolution -----------------------------------------------

export type GeneticOperator = 'crossover' | 'mutation' | 'elite'

export interface GenomeRecord {
  id: string
  generation: number
  fitness: number
  sharpe: number
  max_drawdown: number
  total_return: number
  win_rate: number
  calmar: number
  sortino: number
  genes: Record<string, number>
  parent_ids: string[]
  operator: GeneticOperator
  created_at: string
  is_active: boolean
  rank?: number
}

// ---- Generation Statistics --------------------------------------------

export interface GenerationStats {
  generation: number
  population_size: number
  best_fitness: number
  mean_fitness: number
  worst_fitness: number
  diversity_index: number           // 0..1 -- lower means convergence
  mutation_rate: number             // current adaptive mutation rate
  crossover_rate: number
  elite_count: number
  stagnation_counter: number        // generations without improvement
  best_genome_id: string
  timestamp: string
}

export interface GenerationHistory {
  generations: GenerationStats[]
  mutation_rate_history: number[]   // parallel array matching generations
}

// ---- Parameter State --------------------------------------------------

export type IaeParamName =
  | 'BH_MASS_THRESH'
  | 'BH_DECAY'
  | 'BH_COLLAPSE'
  | 'NAV_OMEGA_SCALE_K'
  | 'NAV_GEO_ENTRY_GATE'
  | 'HURST_WINDOW'
  | 'GARCH_ALPHA'
  | 'GARCH_BETA'
  | 'MIN_HOLD_BARS'
  | 'MAX_POSITION_SIZE'
  | 'ENTRY_ZSCORE'
  | 'EXIT_ZSCORE'

export interface ParameterState {
  name: IaeParamName
  current_value: number
  previous_value: number           // value 24h ago
  delta_24h: number                // absolute change
  delta_pct: number                // pct change
  min_allowed: number
  max_allowed: number
  default_value: number
  direction: 'improving' | 'degrading' | 'neutral'
  last_updated: string
  description: string
}

export interface ParameterHistoryPoint {
  timestamp: string
  value: number
  fitness_at_time: number
}

// ---- Rollback Events --------------------------------------------------

export type RollbackReason =
  | 'sharpe_degradation'
  | 'drawdown_breach'
  | 'fitness_regression'
  | 'manual_override'
  | 'circuit_breaker'

export interface RollbackEvent {
  id: string
  timestamp: string
  reason: RollbackReason
  reason_detail: string
  from_generation: number
  to_generation: number
  parameter_delta: Record<string, number>   // param -> change magnitude
  sharpe_before: number
  sharpe_after: number
  fitness_before: number
  fitness_after: number
  initiated_by: 'automatic' | 'manual'
  operator_id?: string
}

// ---- Genome Pedigree (SVG tree) ----------------------------------------

export interface PedigreeNode {
  genome_id: string
  generation: number
  fitness: number
  sharpe: number
  operator: GeneticOperator
  parent_ids: string[]
  children_ids: string[]
  x?: number                        // layout coords set by frontend
  y?: number
}

export interface GenomePedigree {
  root_id: string
  nodes: PedigreeNode[]
  edges: Array<{ from: string; to: string }>
  depth: number
}

// ---- Evolution Config -------------------------------------------------

export interface IaeEvolutionConfig {
  population_size: number
  max_generations: number
  mutation_sigma: number
  crossover_prob: number
  elite_fraction: number
  tournament_size: number
  diversity_pressure: number
  fitness_window_bars: number
  rollback_sharpe_threshold: number
  rollback_dd_threshold: number
  running: boolean
  paused: boolean
}

// ---- API Response Wrappers --------------------------------------------

export interface IaeEvolutionPayload {
  current_stats: GenerationStats
  history: GenerationStats[]
  top_genomes: GenomeRecord[]
  parameters: ParameterState[]
  recent_rollbacks: RollbackEvent[]
  config: IaeEvolutionConfig
  fetched_at: string
}

// ---- Param metadata for display ----------------------------------------

export interface ParamMeta {
  name: IaeParamName
  label: string
  description: string
  unit: string
  precision: number
}

export const PARAM_META: Record<IaeParamName, ParamMeta> = {
  BH_MASS_THRESH: {
    name: 'BH_MASS_THRESH',
    label: 'BH Mass Threshold',
    description: 'Minimum black hole mass to trigger signal',
    unit: '',
    precision: 3,
  },
  BH_DECAY: {
    name: 'BH_DECAY',
    label: 'BH Decay Rate',
    description: 'Exponential decay coefficient for BH signal strength',
    unit: '',
    precision: 4,
  },
  BH_COLLAPSE: {
    name: 'BH_COLLAPSE',
    label: 'BH Collapse Factor',
    description: 'Multiplier applied at signal collapse event',
    unit: '',
    precision: 3,
  },
  NAV_OMEGA_SCALE_K: {
    name: 'NAV_OMEGA_SCALE_K',
    label: 'Nav Omega Scale',
    description: 'Navigation omega scaling constant',
    unit: '',
    precision: 4,
  },
  NAV_GEO_ENTRY_GATE: {
    name: 'NAV_GEO_ENTRY_GATE',
    label: 'Geo Entry Gate',
    description: 'Geodesic entry gate threshold',
    unit: '',
    precision: 4,
  },
  HURST_WINDOW: {
    name: 'HURST_WINDOW',
    label: 'Hurst Window',
    description: 'Lookback window for Hurst exponent calculation',
    unit: 'bars',
    precision: 0,
  },
  GARCH_ALPHA: {
    name: 'GARCH_ALPHA',
    label: 'GARCH Alpha',
    description: 'GARCH(1,1) alpha (ARCH) coefficient',
    unit: '',
    precision: 4,
  },
  GARCH_BETA: {
    name: 'GARCH_BETA',
    label: 'GARCH Beta',
    description: 'GARCH(1,1) beta (GARCH) coefficient',
    unit: '',
    precision: 4,
  },
  MIN_HOLD_BARS: {
    name: 'MIN_HOLD_BARS',
    label: 'Min Hold Bars',
    description: 'Minimum number of bars to hold a position',
    unit: 'bars',
    precision: 0,
  },
  MAX_POSITION_SIZE: {
    name: 'MAX_POSITION_SIZE',
    label: 'Max Position Size',
    description: 'Maximum notional position size as fraction of NAV',
    unit: '%',
    precision: 3,
  },
  ENTRY_ZSCORE: {
    name: 'ENTRY_ZSCORE',
    label: 'Entry Z-Score',
    description: 'Z-score threshold to open a new position',
    unit: 'σ',
    precision: 2,
  },
  EXIT_ZSCORE: {
    name: 'EXIT_ZSCORE',
    label: 'Exit Z-Score',
    description: 'Z-score threshold to close an existing position',
    unit: 'σ',
    precision: 2,
  },
}
