import type { Trade, PerformanceMetrics, EquityPoint } from '@/types/trades'

// ── Basic statistics ──────────────────────────────────────────────────────────

export function mean(arr: number[]): number {
  if (arr.length === 0) return 0
  return arr.reduce((a, b) => a + b, 0) / arr.length
}

export function std(arr: number[], ddof = 1): number {
  if (arr.length < 2) return 0
  const m = mean(arr)
  const variance = arr.reduce((acc, x) => acc + (x - m) ** 2, 0) / (arr.length - ddof)
  return Math.sqrt(variance)
}

export function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b)
  const idx = (p / 100) * (sorted.length - 1)
  const lower = Math.floor(idx)
  const upper = Math.ceil(idx)
  if (lower === upper) return sorted[lower]
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (idx - lower)
}

export function skewness(arr: number[]): number {
  if (arr.length < 3) return 0
  const m = mean(arr)
  const s = std(arr)
  if (s === 0) return 0
  return (
    (arr.reduce((acc, x) => acc + ((x - m) / s) ** 3, 0) / arr.length)
  )
}

export function kurtosis(arr: number[]): number {
  if (arr.length < 4) return 0
  const m = mean(arr)
  const s = std(arr)
  if (s === 0) return 0
  return arr.reduce((acc, x) => acc + ((x - m) / s) ** 4, 0) / arr.length - 3
}

// ── Finance metrics ───────────────────────────────────────────────────────────

export function sharpeRatio(returns: number[], riskFreeRate = 0, annualizeFactor = 252): number {
  const excess = returns.map(r => r - riskFreeRate / annualizeFactor)
  const m = mean(excess)
  const s = std(excess)
  if (s === 0) return 0
  return (m / s) * Math.sqrt(annualizeFactor)
}

export function sortinoRatio(returns: number[], targetReturn = 0, annualizeFactor = 252): number {
  const downside = returns.filter(r => r < targetReturn).map(r => (r - targetReturn) ** 2)
  const downsideDev = Math.sqrt(mean(downside)) * Math.sqrt(annualizeFactor)
  const annReturn = mean(returns) * annualizeFactor
  if (downsideDev === 0) return 0
  return (annReturn - targetReturn) / downsideDev
}

export function maxDrawdown(equityCurve: number[]): { value: number; pct: number; startIdx: number; endIdx: number } {
  let peak = equityCurve[0]
  let maxDD = 0
  let maxDDPct = 0
  let startIdx = 0
  let endIdx = 0
  let peakIdx = 0

  for (let i = 1; i < equityCurve.length; i++) {
    if (equityCurve[i] > peak) {
      peak = equityCurve[i]
      peakIdx = i
    }
    const dd = peak - equityCurve[i]
    const ddPct = dd / peak
    if (ddPct > maxDDPct) {
      maxDD = dd
      maxDDPct = ddPct
      startIdx = peakIdx
      endIdx = i
    }
  }

  return { value: maxDD, pct: maxDDPct, startIdx, endIdx }
}

export function calmarRatio(annualizedReturn: number, maxDrawdownPct: number): number {
  if (maxDrawdownPct === 0) return 0
  return annualizedReturn / maxDrawdownPct
}

export function valueAtRisk(returns: number[], confidence = 0.95): number {
  return -percentile(returns, (1 - confidence) * 100)
}

export function conditionalVaR(returns: number[], confidence = 0.95): number {
  const varThreshold = -valueAtRisk(returns, confidence)
  const tailReturns = returns.filter(r => r <= varThreshold)
  if (tailReturns.length === 0) return 0
  return -mean(tailReturns)
}

// ── Trade-based metrics ───────────────────────────────────────────────────────

export function computeMetrics(trades: Trade[], equity: number[]): PerformanceMetrics {
  const closedTrades = trades.filter(t => t.exitTime !== null && t.pnl !== undefined)
  const returns = equity.length > 1
    ? equity.slice(1).map((e, i) => (e - equity[i]) / equity[i])
    : []

  const pnls = closedTrades.map(t => t.pnl)
  const wins = pnls.filter(p => p > 0)
  const losses = pnls.filter(p => p < 0)
  const totalPnl = pnls.reduce((a, b) => a + b, 0)

  const annFactor = 252
  const annReturn = mean(returns) * annFactor
  const annVol = std(returns) * Math.sqrt(annFactor)

  const { pct: mddPct, value: mddVal } = equity.length > 0
    ? maxDrawdown(equity)
    : { pct: 0, value: 0 }

  return {
    totalPnl,
    totalPnlPct: equity.length > 0 ? (equity[equity.length - 1] - equity[0]) / equity[0] : 0,
    sharpeRatio: sharpeRatio(returns),
    sortinoRatio: sortinoRatio(returns),
    maxDrawdown: mddVal,
    maxDrawdownPct: mddPct,
    winRate: closedTrades.length > 0 ? wins.length / closedTrades.length : 0,
    profitFactor: Math.abs(losses.reduce((a, b) => a + b, 0)) > 0
      ? wins.reduce((a, b) => a + b, 0) / Math.abs(losses.reduce((a, b) => a + b, 0))
      : 0,
    avgWin: wins.length > 0 ? mean(wins) : 0,
    avgLoss: losses.length > 0 ? mean(losses) : 0,
    avgHoldingHours: closedTrades.length > 0
      ? mean(closedTrades.map(t => t.holdingPeriodHours))
      : 0,
    totalTrades: closedTrades.length,
    totalWins: wins.length,
    totalLosses: losses.length,
    calmarRatio: calmarRatio(annReturn, mddPct),
    annualizedReturn: annReturn,
    annualizedVolatility: annVol,
    skewness: skewness(returns),
    kurtosis: kurtosis(returns),
    var95: valueAtRisk(returns, 0.95),
    cvar95: conditionalVaR(returns, 0.95),
  }
}

// ── Equity curve builder ──────────────────────────────────────────────────────

export function buildEquityCurve(
  trades: Trade[],
  initialEquity = 100_000
): EquityPoint[] {
  const sorted = [...trades].sort(
    (a, b) => new Date(a.entryTime).getTime() - new Date(b.entryTime).getTime()
  )

  let equity = initialEquity
  let peak = initialEquity
  const points: EquityPoint[] = [
    { timestamp: sorted[0]?.entryTime ?? new Date().toISOString(), equity, drawdown: 0 },
  ]

  for (const t of sorted) {
    if (t.pnl !== undefined && t.exitTime) {
      equity += t.pnl
      if (equity > peak) peak = equity
      const dd = peak > 0 ? (peak - equity) / peak : 0
      points.push({ timestamp: t.exitTime, equity, drawdown: dd })
    }
  }

  return points
}

// ── IC calculations ───────────────────────────────────────────────────────────

export function rankCorrelation(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length)
  if (n < 3) return 0

  const rankX = rankArray(x.slice(0, n))
  const rankY = rankArray(y.slice(0, n))

  const mx = mean(rankX)
  const my = mean(rankY)
  const cov = rankX.reduce((acc, rx, i) => acc + (rx - mx) * (rankY[i] - my), 0) / n
  const stdX = std(rankX)
  const stdY = std(rankY)

  if (stdX === 0 || stdY === 0) return 0
  return cov / (stdX * stdY)
}

function rankArray(arr: number[]): number[] {
  const indexed = arr.map((v, i) => ({ v, i }))
  indexed.sort((a, b) => a.v - b.v)
  const ranks = new Array<number>(arr.length)
  indexed.forEach((x, rank) => { ranks[x.i] = rank + 1 })
  return ranks
}

// ── Kelly fraction ────────────────────────────────────────────────────────────

export function kellyFraction(winRate: number, avgWin: number, avgLoss: number): number {
  if (avgLoss === 0) return 0
  const b = Math.abs(avgWin / avgLoss)
  return (b * winRate - (1 - winRate)) / b
}

// ── Deflated Sharpe ───────────────────────────────────────────────────────────

export function deflatedSharpe(
  sharpe: number,
  nTrials: number,
  nObs: number,
  skew = 0,
  kurt = 0
): number {
  // Probability that the Sharpe is positive out-of-sample (Bailey & Lopez de Prado)
  const expectedMaxSharpe =
    Math.sqrt(2) *
    (function erf(x: number): number {
      const t = 1 / (1 + 0.3275911 * Math.abs(x))
      const y = 1 - (0.254829592 * t - 0.284496736 * t ** 2 + 1.421413741 * t ** 3 - 1.453152027 * t ** 4 + 1.061405429 * t ** 5) * Math.exp(-x * x)
      return x >= 0 ? y : -y
    })(Math.sqrt(Math.log(nTrials) - Math.log(Math.log(nTrials)) / 2) / Math.sqrt(2))

  const correction = Math.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / (nObs - 1))
  return (sharpe - expectedMaxSharpe) / correction
}
