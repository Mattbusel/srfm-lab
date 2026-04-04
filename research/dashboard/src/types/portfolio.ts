export interface PortfolioWeight {
  instrument: string
  weight: number
  targetWeight: number
  drift: number
  value: number
  pnlContribution: number
  riskContribution: number
}

export interface EfficientFrontierPoint {
  expectedReturn: number
  volatility: number
  sharpe: number
  weights: Record<string, number>
  isOptimal: boolean
}

export interface CorrelationEntry {
  instrumentA: string
  instrumentB: string
  correlation: number
  rollingCorrelation: number[]
  dates: string[]
}

export interface RiskContribution {
  instrument: string
  marginalContribution: number
  componentContribution: number
  pctContribution: number
  beta: number
  specificRisk: number
  systematicRisk: number
}

export interface HRPCluster {
  id: string
  label: string
  children?: HRPCluster[]
  weight?: number
  distance?: number
  instruments?: string[]
}

export interface PortfolioStats {
  totalValue: number
  expectedReturn: number
  volatility: number
  sharpe: number
  maxDrawdown: number
  var95: number
  cvar95: number
  beta: number
  herfindahlIndex: number
  effectiveN: number
}

export interface RollingCorrelationPoint {
  date: string
  correlations: Record<string, number>
}
