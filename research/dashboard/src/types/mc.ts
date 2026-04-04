export interface MCPath {
  pathId: number
  equity: number[]
  dates: string[]
  finalEquity: number
  maxDrawdown: number
  sharpe: number
  blowup: boolean
}

export interface MCBands {
  date: string
  p5: number
  p25: number
  p50: number
  p75: number
  p95: number
  mean: number
}

export interface MCResults {
  nPaths: number
  nDays: number
  initialEquity: number
  bands: MCBands[]
  finalEquityDistribution: number[]
  blowupRate: number
  blowupThreshold: number
  kellyFraction: number
  optimalFraction: number
  expectedFinalEquity: number
  p5FinalEquity: number
  p50FinalEquity: number
  p95FinalEquity: number
  annualizedReturn: number
  annualizedVolatility: number
  sharpeEstimate: number
  regimeStratified?: RegimeStratifiedMC[]
}

export interface RegimeStratifiedMC {
  regime: string
  nPaths: number
  blowupRate: number
  p50FinalEquity: number
  p5FinalEquity: number
  p95FinalEquity: number
  expectedReturn: number
}

export interface MCSimParams {
  nPaths: number
  nDays: number
  initialEquity: number
  useHistoricalReturns: boolean
  seed?: number
  regimeWeighting: boolean
}
