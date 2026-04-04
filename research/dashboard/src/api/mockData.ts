import type { Trade, EquityPoint, PerformanceMetrics, ReconciliationRow, SlippageStats, TradeFilter } from '@/types/trades'
import type { SignalSnapshot, ICPoint, RollingICPoint, ICByRegime, FactorAttribution, QuintileReturn } from '@/types/signals'
import type { RegimeSegment, TransitionMatrix, RegimePerformance, RegimeDuration, StressScenario } from '@/types/regimes'
import type { MCResults, MCBands } from '@/types/mc'
import type { PortfolioWeight, EfficientFrontierPoint, RiskContribution } from '@/types/portfolio'
import type { RegimeType } from '@/types/trades'

// ── Seed-based PRNG ───────────────────────────────────────────────────────────

let _seed = 42
function rand(): number {
  _seed = (_seed * 1664525 + 1013904223) & 0xffffffff
  return ((_seed >>> 0) / 0xffffffff)
}
function randn(): number {
  const u = 1 - rand(), v = rand()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}
function randChoice<T>(arr: T[]): T {
  return arr[Math.floor(rand() * arr.length)]
}

export const INSTRUMENTS = [
  'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
  'ADA-USD', 'AVAX-USD', 'DOT-USD', 'MATIC-USD', 'LINK-USD',
  'UNI-USD', 'ATOM-USD', 'LTC-USD', 'BCH-USD', 'FIL-USD',
  'NEAR-USD', 'APT-USD', 'ARB-USD', 'OP-USD', 'INJ-USD',
  'SUI-USD', 'SEI-USD', 'TIA-USD', 'DYDX-USD', 'JTO-USD',
  'WIF-USD', 'BOME-USD',
]

const REGIMES: RegimeType[] = ['bull', 'bear', 'sideways', 'ranging', 'volatile']
const STRATEGIES = ['mean_rev', 'momentum', 'breakout', 'regime_adaptive', 'bh_signal']

// ── Trades ────────────────────────────────────────────────────────────────────

export function generateMockTrades(n = 200, filter?: TradeFilter): Trade[] {
  _seed = 12345
  const trades: Trade[] = []
  const now = Date.now()

  for (let i = 0; i < n; i++) {
    const instrument = filter?.instrument ?? randChoice(INSTRUMENTS)
    const regime = filter?.regime ?? randChoice(REGIMES)
    const side = rand() > 0.5 ? 'long' : 'short'
    const entryDaysAgo = rand() * 90
    const holdingHours = 1 + rand() * 48
    const entryTime = new Date(now - entryDaysAgo * 86_400_000).toISOString()
    const exitTime = rand() > 0.1
      ? new Date(now - entryDaysAgo * 86_400_000 + holdingHours * 3_600_000).toISOString()
      : null
    const entryPrice = 100 + rand() * 50000
    const pnlPct = (randn() * 0.02 + 0.003) * (side === 'long' ? 1 : -1)
    const pnl = pnlPct * entryPrice * (10 + rand() * 90)

    trades.push({
      id: `TRD-${i.toString().padStart(5, '0')}`,
      instrument,
      side,
      entryTime,
      exitTime,
      entryPrice,
      exitPrice: exitTime ? entryPrice * (1 + pnlPct) : null,
      quantity: 10 + rand() * 90,
      pnl,
      pnlPct: pnlPct * 100,
      commission: Math.abs(pnl) * 0.001,
      slippage: randn() * 5 + 2,
      regime,
      signalStrength: randn() * 0.3 + 0.5,
      holdingPeriodHours: holdingHours,
      mae: Math.abs(pnl) * (0.2 + rand() * 0.5),
      mfe: Math.abs(pnl) * (1 + rand()),
      strategy: randChoice(STRATEGIES),
      foldId: Math.floor(rand() * 5),
    })
  }

  return trades
}

// ── Equity curve ──────────────────────────────────────────────────────────────

export function generateMockEquity(days = 90, initial = 100_000): EquityPoint[] {
  _seed = 99
  const points: EquityPoint[] = []
  let equity = initial
  let peak = initial
  const now = Date.now()

  for (let d = days; d >= 0; d--) {
    const ret = randn() * 0.015 + 0.0008
    equity *= 1 + ret
    if (equity > peak) peak = equity
    const drawdown = (peak - equity) / peak

    points.push({
      timestamp: new Date(now - d * 86_400_000).toISOString().slice(0, 10),
      equity: Math.round(equity * 100) / 100,
      drawdown: Math.round(drawdown * 10_000) / 10_000,
      benchmark: initial * (1 + (days - d) * 0.001),
    })
  }

  return points
}

// ── Performance metrics ───────────────────────────────────────────────────────

export function generateMockMetrics(): PerformanceMetrics {
  return {
    totalPnl: 28_450,
    totalPnlPct: 28.45,
    sharpeRatio: 1.84,
    sortinoRatio: 2.31,
    maxDrawdown: 8_200,
    maxDrawdownPct: 8.2,
    winRate: 0.613,
    profitFactor: 1.72,
    avgWin: 680,
    avgLoss: -320,
    avgHoldingHours: 14.3,
    totalTrades: 186,
    totalWins: 114,
    totalLosses: 72,
    calmarRatio: 3.47,
    annualizedReturn: 0.284,
    annualizedVolatility: 0.154,
    skewness: 0.31,
    kurtosis: 0.87,
    var95: 0.024,
    cvar95: 0.038,
  }
}

// ── Reconciliation ────────────────────────────────────────────────────────────

export function generateMockReconciliation(n = 50): ReconciliationRow[] {
  _seed = 7777
  return Array.from({ length: n }, (_, i) => {
    const livePnl = randn() * 500 + 200
    const backtestPnl = livePnl + randn() * 80
    return {
      instrument: randChoice(INSTRUMENTS),
      regime: randChoice(REGIMES),
      liveEntryPrice: 100 + rand() * 10000,
      backtestEntryPrice: 100 + rand() * 10000,
      liveExitPrice: rand() > 0.1 ? 100 + rand() * 10000 : null,
      backtestExitPrice: rand() > 0.1 ? 100 + rand() * 10000 : null,
      livePnl,
      backtestPnl,
      pnlDiff: livePnl - backtestPnl,
      slippage: randn() * 4 + 2,
      signalDrift: randn() * 0.1,
      tradeDate: new Date(Date.now() - rand() * 30 * 86_400_000).toISOString().slice(0, 10),
      notes: i % 10 === 0 ? 'Liquidity gap at entry' : '',
    }
  })
}

export function generateMockSlippageStats(): SlippageStats[] {
  _seed = 5555
  return INSTRUMENTS.slice(0, 12).map(instrument => ({
    instrument,
    avgSlippage: 1 + rand() * 8,
    medianSlippage: 0.5 + rand() * 5,
    p95Slippage: 5 + rand() * 20,
    worstSlippage: 20 + rand() * 50,
    slippagePct: 0.01 + rand() * 0.1,
    count: Math.floor(10 + rand() * 40),
  }))
}

// ── Signals ───────────────────────────────────────────────────────────────────

export function generateMockSignals(): SignalSnapshot[] {
  _seed = 3333
  return INSTRUMENTS.map(instrument => {
    const composite = randn() * 0.4
    return {
      instrument,
      timestamp: new Date().toISOString(),
      rawSignal: randn() * 2,
      normalizedSignal: composite,
      zscore: randn() * 2,
      ic: randn() * 0.15,
      momentum1d: randn() * 0.03,
      momentum5d: randn() * 0.08,
      momentum21d: randn() * 0.15,
      meanReversion: randn() * 0.2,
      volumeSignal: randn() * 0.3,
      regimeSignal: randn() * 0.4,
      compositeStrength: composite,
      direction: composite > 0.1 ? 'long' : composite < -0.1 ? 'short' : 'neutral',
    }
  })
}

export function generateMockICDecay(): ICPoint[] {
  _seed = 9999
  return Array.from({ length: 21 }, (_, lag) => {
    const base = 0.12 * Math.exp(-lag * 0.08)
    const noise = randn() * 0.02
    const ic = base + noise
    return {
      lag,
      ic,
      icLow: ic - 0.025 - rand() * 0.01,
      icHigh: ic + 0.025 + rand() * 0.01,
      tStat: ic / 0.015,
    }
  })
}

export function generateMockRollingIC(days = 120): RollingICPoint[] {
  _seed = 8888
  const now = Date.now()
  return Array.from({ length: days }, (_, i) => {
    const ic = randn() * 0.06 + 0.04
    return {
      date: new Date(now - (days - i) * 86_400_000).toISOString().slice(0, 10),
      ic,
      icMean: 0.04 + randn() * 0.01,
      icStd: 0.02 + rand() * 0.02,
      hitRate: 0.5 + randn() * 0.05,
    }
  })
}

export function generateMockICByRegime(): ICByRegime[] {
  _seed = 6666
  return REGIMES.map(regime => ({
    regime,
    ic: randn() * 0.08 + 0.04,
    icStd: 0.02 + rand() * 0.03,
    sampleSize: 20 + Math.floor(rand() * 80),
    tStat: 2 + randn(),
    pValue: rand() * 0.1,
  }))
}

export function generateMockFactorAttribution(): FactorAttribution[] {
  const factors = ['Momentum', 'MeanRev', 'Volume', 'Regime', 'Volatility', 'Flow', 'Sentiment']
  _seed = 4444
  return factors.map(factor => {
    const contribution = randn() * 3000
    return {
      factor,
      contribution,
      contributionPct: contribution / 300,
      tStat: contribution / 500,
      active: rand() > 0.2,
    }
  })
}

export function generateMockQuintileReturns(): QuintileReturn[] {
  _seed = 2222
  return [1, 2, 3, 4, 5].map(q => ({
    quintile: q,
    avgReturn: (q - 3) * 0.012 + randn() * 0.005,
    count: 30 + Math.floor(rand() * 20),
    sharpe: (q - 3) * 0.4 + randn() * 0.2,
    winRate: 0.3 + q * 0.06 + randn() * 0.03,
  }))
}

// ── Regimes ───────────────────────────────────────────────────────────────────

export function generateMockRegimeSegments(days = 180): RegimeSegment[] {
  _seed = 1111
  const segments: RegimeSegment[] = []
  let cursor = Date.now() - days * 86_400_000

  while (cursor < Date.now()) {
    const duration = 5 + Math.floor(rand() * 25)
    const endMs = Math.min(cursor + duration * 86_400_000, Date.now())
    segments.push({
      startDate: new Date(cursor).toISOString().slice(0, 10),
      endDate: new Date(endMs).toISOString().slice(0, 10),
      regime: randChoice(REGIMES),
      durationDays: duration,
      priceReturn: randn() * 0.08,
      volatility: 0.01 + rand() * 0.04,
      avgVolume: 1e9 + rand() * 5e9,
    })
    cursor = endMs
  }

  return segments
}

export function generateMockTransitionMatrix(): TransitionMatrix {
  const regimes: RegimeType[] = ['bull', 'bear', 'sideways', 'ranging', 'volatile']
  const matrix = {} as TransitionMatrix
  for (const from of regimes) {
    matrix[from] = {} as Record<RegimeType, number>
    const weights = regimes.map(() => rand())
    const total = weights.reduce((a, b) => a + b, 0)
    regimes.forEach((to, i) => {
      matrix[from][to] = weights[i] / total
    })
  }
  return matrix
}

export function generateMockRegimePerformance(): RegimePerformance[] {
  _seed = 1234
  return REGIMES.map(regime => ({
    regime,
    totalPnl: randn() * 10000 + 3000,
    avgPnl: randn() * 300 + 100,
    sharpe: randn() * 0.5 + 1.2,
    winRate: 0.45 + rand() * 0.25,
    tradeCount: 20 + Math.floor(rand() * 60),
    avgHoldingHours: 8 + rand() * 24,
    maxDrawdown: rand() * 0.15,
    occurrencePct: 0.1 + rand() * 0.3,
  }))
}

export function generateMockRegimeDurations(): RegimeDuration[] {
  _seed = 5678
  return REGIMES.map(regime => ({
    regime,
    avgDays: 8 + rand() * 20,
    medianDays: 6 + rand() * 15,
    minDays: 1 + Math.floor(rand() * 3),
    maxDays: 30 + Math.floor(rand() * 60),
    p25Days: 3 + rand() * 8,
    p75Days: 12 + rand() * 20,
    count: 5 + Math.floor(rand() * 20),
  }))
}

export function generateMockStressScenarios(): StressScenario[] {
  return [
    { id: 'covid', name: 'COVID-19 Crash', description: 'Mar 2020 global crash', category: 'market_crash', startDate: '2020-02-20', endDate: '2020-03-23', regimeSequence: ['volatile', 'bear'], pnlImpact: -14200, pnlImpactPct: -14.2, maxDrawdown: 0.28, recoveryDays: 65, probability: 0.02 },
    { id: 'ftx', name: 'FTX Collapse', description: 'Nov 2022 FTX bankruptcy', category: 'market_crash', startDate: '2022-11-07', endDate: '2022-11-14', regimeSequence: ['volatile', 'bear', 'bear'], pnlImpact: -8900, pnlImpactPct: -8.9, maxDrawdown: 0.22, recoveryDays: 120, probability: 0.01 },
    { id: 'luna', name: 'LUNA Implosion', description: 'May 2022 Terra/LUNA collapse', category: 'tail', startDate: '2022-05-07', endDate: '2022-05-12', regimeSequence: ['volatile', 'bear'], pnlImpact: -6300, pnlImpactPct: -6.3, maxDrawdown: 0.18, recoveryDays: 90, probability: 0.02 },
    { id: 'volspike', name: 'Vol Spike', description: 'Sudden volatility regime shift', category: 'vol_spike', startDate: '2024-08-05', endDate: '2024-08-08', regimeSequence: ['volatile'], pnlImpact: -3200, pnlImpactPct: -3.2, maxDrawdown: 0.08, recoveryDays: 14, probability: 0.05 },
    { id: 'correlation', name: 'Correlation Breakdown', description: 'Cross-asset correlation spike', category: 'correlation', startDate: '2023-03-10', endDate: '2023-03-17', regimeSequence: ['ranging', 'volatile'], pnlImpact: -2100, pnlImpactPct: -2.1, maxDrawdown: 0.05, recoveryDays: 21, probability: 0.08 },
    { id: 'liquidity', name: 'Liquidity Crisis', description: 'Bid-ask spreads 10x', category: 'liquidity', startDate: '2022-06-13', endDate: '2022-06-18', regimeSequence: ['bear', 'volatile'], pnlImpact: -4800, pnlImpactPct: -4.8, maxDrawdown: 0.12, recoveryDays: 45, probability: 0.03 },
  ]
}

// ── MC simulation ─────────────────────────────────────────────────────────────

export function generateMockMCResults(nDays = 252, initial = 100_000): MCResults {
  _seed = 77777
  const nPaths = 10_000
  const bands: MCBands[] = []
  const now = Date.now()
  const finalEquities: number[] = []

  for (let d = 0; d <= nDays; d++) {
    const date = new Date(now + d * 86_400_000).toISOString().slice(0, 10)
    const progress = d / nDays
    const mean = initial * (1 + progress * 0.25)
    const vol = initial * 0.15 * Math.sqrt(progress + 0.01)

    bands.push({
      date,
      p5: mean - 1.645 * vol,
      p25: mean - 0.674 * vol,
      p50: mean,
      p75: mean + 0.674 * vol,
      p95: mean + 1.645 * vol,
      mean,
    })
  }

  for (let p = 0; p < 200; p++) {
    finalEquities.push(initial * (1 + randn() * 0.25 + 0.25))
  }

  return {
    nPaths,
    nDays,
    initialEquity: initial,
    bands,
    finalEquityDistribution: finalEquities,
    blowupRate: 0.023,
    blowupThreshold: initial * 0.5,
    kellyFraction: 0.18,
    optimalFraction: 0.12,
    expectedFinalEquity: initial * 1.25,
    p5FinalEquity: initial * 0.82,
    p50FinalEquity: initial * 1.22,
    p95FinalEquity: initial * 1.75,
    annualizedReturn: 0.25,
    annualizedVolatility: 0.18,
    sharpeEstimate: 1.39,
    regimeStratified: REGIMES.map(r => ({
      regime: r,
      nPaths: Math.floor(nPaths / 5),
      blowupRate: rand() * 0.05,
      p50FinalEquity: initial * (0.9 + rand() * 0.6),
      p5FinalEquity: initial * (0.6 + rand() * 0.3),
      p95FinalEquity: initial * (1.3 + rand() * 0.8),
      expectedReturn: randn() * 0.15 + 0.1,
    })),
  }
}

// ── Portfolio ─────────────────────────────────────────────────────────────────

export function generateMockPortfolioWeights(): PortfolioWeight[] {
  _seed = 11111
  const instruments = INSTRUMENTS.slice(0, 10)
  const rawWeights = instruments.map(() => rand())
  const total = rawWeights.reduce((a, b) => a + b, 0)
  const totalValue = 100_000

  return instruments.map((instrument, i) => {
    const weight = rawWeights[i] / total
    return {
      instrument,
      weight,
      targetWeight: rawWeights[i] / total + randn() * 0.02,
      drift: randn() * 0.02,
      value: totalValue * weight,
      pnlContribution: randn() * 1000,
      riskContribution: rawWeights[i] / total + randn() * 0.01,
    }
  })
}

export function generateMockEfficientFrontier(): EfficientFrontierPoint[] {
  _seed = 22222
  return Array.from({ length: 20 }, (_, i) => {
    const vol = 0.08 + i * 0.01
    const ret = 0.05 + i * 0.012 + randn() * 0.01
    return {
      expectedReturn: ret,
      volatility: vol,
      sharpe: ret / vol,
      weights: Object.fromEntries(INSTRUMENTS.slice(0, 5).map(k => [k, rand() * 0.3])),
      isOptimal: i === 12,
    }
  })
}

export function generateMockRiskContribution(): RiskContribution[] {
  _seed = 33333
  const instruments = INSTRUMENTS.slice(0, 8)
  const rawRC = instruments.map(() => rand())
  const total = rawRC.reduce((a, b) => a + b, 0)
  return instruments.map((instrument, i) => ({
    instrument,
    marginalContribution: rawRC[i] / total * 0.15,
    componentContribution: rawRC[i] / total * 0.15,
    pctContribution: rawRC[i] / total,
    beta: 0.6 + rand() * 0.8,
    specificRisk: rand() * 0.1,
    systematicRisk: rand() * 0.08,
  }))
}

export { REGIMES }
