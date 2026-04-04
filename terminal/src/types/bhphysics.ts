// ============================================================
// BLACK HOLE PHYSICS STATE TYPES
// ============================================================

export type BHRegime = 'BULL' | 'BEAR' | 'SIDEWAYS' | 'HIGH_VOL'
export type BHDirection = 1 | -1 | 0
export type BHTimeframe = '15m' | '1h' | '1d'

export interface BHState {
  mass: number              // BH mass scalar (0..∞, typically 0..3)
  active: boolean           // is mass above threshold?
  dir: BHDirection          // directional momentum: 1=up, -1=down, 0=neutral
  ctl: number               // counter-trend-lock count
  bh_form: number           // formation index (0=no formation, 1+=active)
  regime: BHRegime
  massThreshold: number     // configurable threshold for 'active'
  massHistory?: number[]    // last N mass values for sparkline
  lastFormationTime?: number
  formationStrength?: number
}

export interface InstrumentBHState {
  symbol: string
  tf15m: BHState
  tf1h: BHState
  tf1d: BHState
  price: number
  frac: number              // fractal dimension
  entryPrice?: number
  positionSide?: 'long' | 'short' | null
  positionSize?: number
  lastUpdated: number
}

export interface LiveState {
  timestamp: string
  equity: number
  instruments: Record<string, InstrumentBHState>
  sessionPnl?: number
  sessionPnlPct?: number
  activeFormations?: string[]   // symbols with active formations
}

export interface BHFormationEvent {
  id: string
  symbol: string
  timeframe: BHTimeframe
  mass: number
  dir: BHDirection
  regime: BHRegime
  timestamp: number
  price: number
  acknowledged: boolean
}

export interface BHMassPoint {
  timestamp: number
  mass15m: number
  mass1h: number
  mass1d: number
  regime: BHRegime
  dir: BHDirection
  price: number
}

export interface BHHistoryRecord {
  symbol: string
  points: BHMassPoint[]
  formationEvents: BHFormationEvent[]
  lastUpdated: number
}

export interface BHAlertRule {
  id: string
  symbol: string
  timeframe: BHTimeframe | 'any'
  condition: 'mass_above' | 'mass_below' | 'formation' | 'dir_change' | 'regime_change'
  threshold?: number
  targetRegime?: BHRegime
  enabled: boolean
  lastTriggered?: number
  cooldownMs: number       // min time between triggers
  sound: boolean
  notify: boolean
  message?: string
}

export interface BHScanResult {
  symbol: string
  state: InstrumentBHState
  score: number            // computed relevance score
  reason: string[]         // why it showed up in scan
  priority: 'critical' | 'high' | 'medium' | 'low'
}

export interface BHRegimePeriod {
  symbol: string
  timeframe: BHTimeframe
  regime: BHRegime
  startTime: number
  endTime: number | null
  duration: number         // seconds
  priceReturn: number      // price change during period
  avgMass: number
}

export interface BHGaugeData {
  timeframe: BHTimeframe
  mass: number
  maxMass: number          // for gauge scale
  active: boolean
  dir: BHDirection
  regime: BHRegime
  ctl: number
  formationActive: boolean
  color: string            // computed color based on state
  label: string
}

// API response types
export interface BHStateResponse {
  status: 'ok' | 'error'
  timestamp: string
  data: LiveState
}

export interface BHHistoryResponse {
  status: 'ok' | 'error'
  symbol: string
  timeframe: BHTimeframe
  points: BHMassPoint[]
  total: number
}

export interface BHScanResponse {
  status: 'ok' | 'error'
  timestamp: string
  results: BHScanResult[]
  total: number
}

// WebSocket BH update
export interface BHStateUpdate {
  type: 'bh_state'
  timestamp: string
  symbol: string
  tf: BHTimeframe
  state: BHState
  price: number
  frac: number
}

export interface BHFormationAlert {
  type: 'bh_formation'
  timestamp: string
  symbol: string
  tf: BHTimeframe
  mass: number
  dir: BHDirection
  regime: BHRegime
  price: number
  strength: number
}
