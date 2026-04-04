export interface SignalSnapshot {
  instrument: string
  timestamp: string
  rawSignal: number
  normalizedSignal: number
  zscore: number
  ic: number
  momentum1d: number
  momentum5d: number
  momentum21d: number
  meanReversion: number
  volumeSignal: number
  regimeSignal: number
  compositeStrength: number
  direction: 'long' | 'short' | 'neutral'
}

export interface ICPoint {
  lag: number
  ic: number
  icLow: number   // lower confidence band
  icHigh: number  // upper confidence band
  tStat: number
}

export interface RollingICPoint {
  date: string
  ic: number
  icMean: number
  icStd: number
  hitRate: number
}

export interface ICByRegime {
  regime: string
  ic: number
  icStd: number
  sampleSize: number
  tStat: number
  pValue: number
}

export interface FactorAttribution {
  factor: string
  contribution: number
  contributionPct: number
  tStat: number
  active: boolean
}

export interface QuintileReturn {
  quintile: number
  avgReturn: number
  count: number
  sharpe: number
  winRate: number
}

export interface AlphaDecay {
  halfLifeDays: number
  decayCurve: ICPoint[]
  regime: string
}

export interface SignalDriftPoint {
  date: string
  liveSignal: number
  backtestSignal: number
  drift: number
  instrument: string
}
