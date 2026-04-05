// ─── Core Domain Types ────────────────────────────────────────────────────────

export type Island = 'BULL' | 'BEAR' | 'NEUTRAL'
export type HypothesisStatus = 'pending' | 'testing' | 'adopted' | 'rejected'
export type HypothesisSource = 'genome' | 'academic' | 'serendipity' | 'causal'
export type HypothesisType =
  | 'entry_signal'
  | 'exit_signal'
  | 'position_sizing'
  | 'risk_filter'
  | 'regime_detection'
  | 'correlation'
export type AlertSeverity = 'info' | 'warning' | 'critical'
export type AlertType =
  | 'evolution'
  | 'hypothesis'
  | 'shadow'
  | 'regime'
  | 'system'
  | 'promotion'
export type SerendipityTechnique =
  | 'domain_borrow'
  | 'inversion'
  | 'combination'
  | 'mutation'
export type Complexity = 'low' | 'medium' | 'high'
export type PaperSource = 'arXiv' | 'SSRN' | 'local'

// ─── Genome ───────────────────────────────────────────────────────────────────

export interface GenomeParams {
  lookback: number
  threshold: number
  stopLoss: number
  takeProfit: number
  positionSize: number
  rsiPeriod: number
  macdFast: number
  macdSlow: number
  atrMultiplier: number
  volFilter: number
}

export interface Genome {
  id: number
  params: GenomeParams
  fitness: number
  sharpe: number
  maxDD: number
  calmar: number
  island: Island
  generation: number
  isHallOfFame: boolean
  parentIds?: number[]
  createdAt: string
  totalTrades?: number
  winRate?: number
}

export interface FitnessHistory {
  generation: number
  fitness: number
  sharpe: number
  maxDD: number
}

// ─── Hypothesis ───────────────────────────────────────────────────────────────

export interface Hypothesis {
  id: number
  type: HypothesisType
  description: string
  params: Record<string, number | string | boolean>
  status: HypothesisStatus
  score: number
  source: HypothesisSource
  createdAt: string
  updatedAt: string
  relatedGenomeId?: number
  relatedPaperId?: number
  testResults?: HypothesisTestResult
}

export interface HypothesisTestResult {
  sharpe: number
  maxDD: number
  calmar: number
  totalTrades: number
  winRate: number
  duration: string
  completedAt: string
}

// ─── Shadow ───────────────────────────────────────────────────────────────────

export interface Shadow {
  shadowId: number
  genomeId: number
  return7d: number
  returnLive7d: number
  alpha: number
  promoted: boolean
  startedAt: string
  alphaDays: number
  equityCurve?: EquityPoint[]
  liveEquityCurve?: EquityPoint[]
}

export interface EquityPoint {
  timestamp: string
  equity: number
}

// ─── Counterfactual ───────────────────────────────────────────────────────────

export interface Counterfactual {
  id: number
  baselineRunId: string
  paramDelta: Record<string, number>
  improvement: number
  sharpe: number
  maxDD: number
  calmar: number
  description: string
  createdAt: string
}

// ─── Academic Papers ──────────────────────────────────────────────────────────

export interface AcademicPaper {
  id: number
  title: string
  abstract: string
  relevanceScore: number
  source: PaperSource
  authors: string[]
  publishedAt: string
  url?: string
  extractedHypotheses?: ExtractedHypothesis[]
  tags: string[]
}

export interface ExtractedHypothesis {
  description: string
  confidence: number
  type: HypothesisType
}

// ─── Serendipity ──────────────────────────────────────────────────────────────

export interface SerendipityIdea {
  id: number
  technique: SerendipityTechnique
  domain: string
  ideaText: string
  rationale: string
  complexity: Complexity
  createdAt: string
  score?: number
  submittedAsHypothesis?: boolean
}

// ─── Alerts ───────────────────────────────────────────────────────────────────

export interface Alert {
  id: number
  type: AlertType
  severity: AlertSeverity
  message: string
  acknowledged: boolean
  createdAt: string
  metadata?: Record<string, unknown>
}

// ─── Genealogy ────────────────────────────────────────────────────────────────

export interface GenealogyNode {
  genomeId: number
  island: Island
  generation: number
  fitness: number
  isHallOfFame: boolean
  parentIds?: number[]
  sharpe?: number
}

export interface GenealogyEdge {
  source: number
  target: number
}

export interface GenealogyGraph {
  nodes: GenealogyNode[]
  edges: GenealogyEdge[]
}

// ─── Evolution Stats ──────────────────────────────────────────────────────────

export interface EvolutionStats {
  generation: number
  bestFitness: number
  meanFitness: number
  diversityIndex: number
  island: Island
  timestamp: string
}

export interface MutationFrequency {
  mutation: string
  count: number
  avgFitnessImprovement: number
}

// ─── Narrative / Report ───────────────────────────────────────────────────────

export interface WeeklyReport {
  id: number
  weekStart: string
  weekEnd: string
  markdownContent: string
  generatedAt: string
  topGenomes: number[]
  hypothesesAdopted: number
  hypothesesRejected: number
  alerts: Alert[]
}

// ─── WebSocket Messages ───────────────────────────────────────────────────────

export type WsMessageType =
  | 'genome_update'
  | 'hypothesis_update'
  | 'shadow_update'
  | 'alert'
  | 'regime_change'
  | 'evolution_tick'
  | 'ping'

export interface WsMessage {
  type: WsMessageType
  payload: unknown
  timestamp: string
}

export interface RegimeChangePayload {
  regime: Island
  confidence: number
  previousRegime: Island
}

export interface EvolutionTickPayload {
  island: Island
  generation: number
  bestFitness: number
  meanFitness: number
  diversityIndex: number
}

// ─── API Response Wrappers ────────────────────────────────────────────────────

export interface ApiResponse<T> {
  data: T
  total?: number
  page?: number
  perPage?: number
}

export interface ApiError {
  error: string
  code: number
  details?: string
}
