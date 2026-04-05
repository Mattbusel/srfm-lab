import type {
  Genome,
  Hypothesis,
  Shadow,
  Counterfactual,
  Alert,
  AcademicPaper,
  SerendipityIdea,
  GenealogyGraph,
  EvolutionStats,
  MutationFrequency,
  WeeklyReport,
  Island,
  HypothesisStatus,
  SerendipityTechnique,
  ApiError,
} from '../types'

// ─── Base Config ─────────────────────────────────────────────────────────────

const BASE_URL = '/api'

class ApiClientError extends Error {
  constructor(
    message: string,
    public status: number,
    public details?: string
  ) {
    super(message)
    this.name = 'ApiClientError'
  }
}

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${BASE_URL}${path}`
  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers ?? {}),
    },
    ...options,
  })

  if (!res.ok) {
    let errBody: ApiError | null = null
    try {
      errBody = await res.json()
    } catch {
      // ignore parse error
    }
    throw new ApiClientError(
      errBody?.error ?? `HTTP ${res.status}`,
      res.status,
      errBody?.details
    )
  }

  // 204 No Content
  if (res.status === 204) return undefined as unknown as T

  return res.json() as Promise<T>
}

// ─── Mock Data Generators ─────────────────────────────────────────────────────
// These provide realistic data when the backend is not running.

function mockGenomes(island?: Island): Genome[] {
  const islands: Island[] = island ? [island] : ['BULL', 'BEAR', 'NEUTRAL']
  const genomes: Genome[] = []
  let id = 1
  for (const isl of islands) {
    for (let g = 0; g < 8; g++) {
      const fitness = 0.3 + Math.random() * 0.7
      genomes.push({
        id: id++,
        params: {
          lookback: Math.floor(10 + Math.random() * 50),
          threshold: parseFloat((0.1 + Math.random() * 0.9).toFixed(3)),
          stopLoss: parseFloat((0.01 + Math.random() * 0.05).toFixed(3)),
          takeProfit: parseFloat((0.02 + Math.random() * 0.1).toFixed(3)),
          positionSize: parseFloat((0.05 + Math.random() * 0.2).toFixed(3)),
          rsiPeriod: Math.floor(7 + Math.random() * 21),
          macdFast: Math.floor(8 + Math.random() * 6),
          macdSlow: Math.floor(18 + Math.random() * 8),
          atrMultiplier: parseFloat((1 + Math.random() * 3).toFixed(2)),
          volFilter: parseFloat((0.5 + Math.random() * 2).toFixed(2)),
        },
        fitness,
        sharpe: parseFloat((fitness * 2.5 + (Math.random() - 0.5) * 0.5).toFixed(3)),
        maxDD: parseFloat((-0.05 - Math.random() * 0.15).toFixed(3)),
        calmar: parseFloat((fitness * 5 + Math.random()).toFixed(3)),
        island: isl,
        generation: Math.floor(1 + Math.random() * 20),
        isHallOfFame: fitness > 0.85,
        createdAt: new Date(Date.now() - Math.random() * 7 * 86400000).toISOString(),
        totalTrades: Math.floor(50 + Math.random() * 200),
        winRate: parseFloat((0.45 + Math.random() * 0.2).toFixed(3)),
      })
    }
  }
  return genomes
}

function mockHypotheses(status?: HypothesisStatus): Hypothesis[] {
  const statuses: HypothesisStatus[] = ['pending', 'testing', 'adopted', 'rejected']
  const types = ['entry_signal', 'exit_signal', 'position_sizing', 'risk_filter', 'regime_detection'] as const
  const sources = ['genome', 'academic', 'serendipity', 'causal'] as const
  const descriptions = [
    'RSI divergence on 4h timeframe signals mean reversion with >0.6 confidence',
    'MACD crossover combined with volume spike improves entry timing by 12%',
    'ATR-based position sizing reduces drawdown without sacrificing returns',
    'Funding rate extremes predict 24h reversal with 68% accuracy',
    'Correlation between BTC dominance and altcoin returns during regime shifts',
    'Kelly criterion with half-Kelly adjustment outperforms fixed position sizing',
    'On-chain whale accumulation signals 48h forward momentum',
    'VIX-analog for crypto using realized vs implied vol spread',
  ]
  return descriptions.map((desc, i) => {
    const s = status ?? statuses[i % statuses.length]
    return {
      id: i + 1,
      type: types[i % types.length],
      description: desc,
      params: { period: 14 + i * 2, threshold: 0.5 + i * 0.05 },
      status: s,
      score: parseFloat((0.4 + Math.random() * 0.6).toFixed(3)),
      source: sources[i % sources.length],
      createdAt: new Date(Date.now() - i * 3600000).toISOString(),
      updatedAt: new Date(Date.now() - i * 1800000).toISOString(),
    }
  })
}

function mockShadows(): Shadow[] {
  return Array.from({ length: 6 }, (_, i) => {
    const ret7d = parseFloat((-0.05 + Math.random() * 0.15).toFixed(4))
    const live7d = parseFloat((-0.03 + Math.random() * 0.12).toFixed(4))
    return {
      shadowId: i + 1,
      genomeId: i * 3 + 1,
      return7d: ret7d,
      returnLive7d: live7d,
      alpha: parseFloat((ret7d - live7d).toFixed(4)),
      promoted: i === 0,
      startedAt: new Date(Date.now() - (7 + i) * 86400000).toISOString(),
      alphaDays: Math.floor(1 + Math.random() * 10),
    }
  })
}

function mockCounterfactuals(): Counterfactual[] {
  return Array.from({ length: 5 }, (_, i) => ({
    id: i + 1,
    baselineRunId: `run_${1000 + i}`,
    paramDelta: { lookback: i * 2 - 4, threshold: 0.05 * i },
    improvement: parseFloat((0.01 + Math.random() * 0.15).toFixed(4)),
    sharpe: parseFloat((1.2 + Math.random() * 0.8).toFixed(3)),
    maxDD: parseFloat((-0.08 - Math.random() * 0.05).toFixed(3)),
    calmar: parseFloat((2 + Math.random() * 3).toFixed(3)),
    description: `Adjusted lookback +${i * 2} bars, threshold +${(0.05 * i).toFixed(2)}`,
    createdAt: new Date(Date.now() - i * 7200000).toISOString(),
  }))
}

function mockAlerts(): Alert[] {
  const alerts: Alert[] = [
    { id: 1, type: 'regime', severity: 'critical', message: 'Regime shift detected: BULL → BEAR (confidence: 0.87)', acknowledged: false, createdAt: new Date(Date.now() - 300000).toISOString() },
    { id: 2, type: 'evolution', severity: 'info', message: 'Island BULL reached generation 42 — new hall-of-fame genome #77 (fitness: 0.921)', acknowledged: false, createdAt: new Date(Date.now() - 900000).toISOString() },
    { id: 3, type: 'shadow', severity: 'warning', message: 'Shadow #3 alpha turning negative after 5 positive days — monitor closely', acknowledged: false, createdAt: new Date(Date.now() - 1800000).toISOString() },
    { id: 4, type: 'hypothesis', severity: 'info', message: 'Hypothesis #5 "RSI divergence" adopted with Sharpe 1.84', acknowledged: true, createdAt: new Date(Date.now() - 3600000).toISOString() },
    { id: 5, type: 'promotion', severity: 'info', message: 'Genome #12 promoted from shadow to live after 8 days positive alpha', acknowledged: true, createdAt: new Date(Date.now() - 7200000).toISOString() },
    { id: 6, type: 'system', severity: 'warning', message: 'Academic paper ingestion rate below threshold — check SSRN connector', acknowledged: false, createdAt: new Date(Date.now() - 10800000).toISOString() },
  ]
  return alerts
}

function mockEvolutionStats(): EvolutionStats[] {
  const stats: EvolutionStats[] = []
  const islands: Island[] = ['BULL', 'BEAR', 'NEUTRAL']
  for (const island of islands) {
    for (let gen = 1; gen <= 25; gen++) {
      stats.push({
        generation: gen,
        bestFitness: parseFloat((0.3 + gen * 0.025 + (Math.random() - 0.5) * 0.05).toFixed(4)),
        meanFitness: parseFloat((0.2 + gen * 0.018 + (Math.random() - 0.5) * 0.04).toFixed(4)),
        diversityIndex: parseFloat((0.9 - gen * 0.015 + (Math.random() - 0.5) * 0.1).toFixed(4)),
        island,
        timestamp: new Date(Date.now() - (25 - gen) * 3600000).toISOString(),
      })
    }
  }
  return stats
}

function mockAcademicPapers(): AcademicPaper[] {
  return [
    {
      id: 1,
      title: 'Cross-Sectional Momentum in Cryptocurrency Markets: Risk or Mispricing?',
      abstract: 'We document significant cross-sectional momentum in cryptocurrency returns. A long-short portfolio based on past 1-month returns generates monthly alpha of 4.5% after controlling for common risk factors. The momentum premium is concentrated in small-cap tokens and reverses sharply during market stress periods, suggesting behavioral rather than risk-based explanations.',
      relevanceScore: 0.94,
      source: 'SSRN',
      authors: ['Chen, Y.', 'Zhang, L.', 'Kim, J.'],
      publishedAt: '2025-11-14',
      url: 'https://ssrn.com/abstract=mock',
      tags: ['momentum', 'cross-section', 'crypto', 'alpha'],
      extractedHypotheses: [
        { description: 'Past 1-month return predicts next-month return for small-cap tokens', confidence: 0.82, type: 'entry_signal' },
        { description: 'Momentum reversal during high-volatility regimes — add VIX-analog filter', confidence: 0.71, type: 'risk_filter' },
      ],
    },
    {
      id: 2,
      title: 'Funding Rate Dynamics and Return Predictability in Perpetual Futures',
      abstract: 'Using a dataset of 48 perpetual swap contracts across three major exchanges, we show that extreme funding rate events predict significant price reversals within 24-48 hours. The effect is strongest when funding exceeds 2 standard deviations above the rolling mean, with annualized Sharpe ratios exceeding 2.1 in out-of-sample tests.',
      relevanceScore: 0.91,
      source: 'arXiv',
      authors: ['Patel, R.', 'Nakamura, T.'],
      publishedAt: '2025-12-01',
      url: 'https://arxiv.org/abs/mock',
      tags: ['funding rate', 'perpetuals', 'mean reversion', 'predictability'],
      extractedHypotheses: [
        { description: 'Funding rate > 2σ triggers mean-reversion entry with 24h hold', confidence: 0.88, type: 'entry_signal' },
      ],
    },
    {
      id: 3,
      title: 'Regime-Conditional Factor Exposures in Digital Asset Portfolios',
      abstract: 'We identify three distinct market regimes in Bitcoin using a hidden Markov model calibrated on realized volatility and on-chain metrics. Factor loadings shift dramatically across regimes: momentum strategies excel in trending regimes while mean-reversion dominates in choppy regimes. A regime-conditional allocation increases Sharpe by 0.6 over static exposure.',
      relevanceScore: 0.88,
      source: 'SSRN',
      authors: ['Garcia, M.', 'Weber, K.', 'Liu, S.'],
      publishedAt: '2025-10-22',
      tags: ['regime', 'HMM', 'factor', 'on-chain'],
    },
    {
      id: 4,
      title: 'Optimal Execution in Illiquid Crypto Markets with Adverse Selection',
      abstract: 'We develop an execution algorithm that accounts for adverse selection costs unique to crypto markets, including wash trading, spoofing, and bot activity. Compared to TWAP and VWAP benchmarks, our algorithm reduces implementation shortfall by 23 basis points on average for mid-cap tokens.',
      relevanceScore: 0.72,
      source: 'arXiv',
      authors: ['Alvarez, J.', 'Singh, A.'],
      publishedAt: '2025-09-15',
      tags: ['execution', 'market microstructure', 'adverse selection'],
    },
    {
      id: 5,
      title: 'Deep Reinforcement Learning for Adaptive Position Sizing in High-Frequency Crypto',
      abstract: 'We train a PPO agent to dynamically adjust position sizes based on realized volatility, order book depth, and recent PnL. The agent learns to reduce exposure during adverse selection events and increase it during momentum regimes, achieving a 31% reduction in maximum drawdown versus fixed position sizing with only 8% reduction in total return.',
      relevanceScore: 0.85,
      source: 'local',
      authors: ['Internal Research Team'],
      publishedAt: '2025-12-10',
      tags: ['reinforcement learning', 'position sizing', 'drawdown', 'RL'],
      extractedHypotheses: [
        { description: 'Reduce position size by 50% when realized vol exceeds 2x 30d average', confidence: 0.79, type: 'position_sizing' },
        { description: 'Increase position size during low-spread, high-depth order book conditions', confidence: 0.65, type: 'position_sizing' },
      ],
    },
  ]
}

function mockSerendipityIdeas(): SerendipityIdea[] {
  return [
    {
      id: 1,
      technique: 'domain_borrow',
      domain: 'Epidemiology',
      ideaText: 'Model momentum "contagion" like disease spread — R0 equivalent for trend strength indicates when momentum will self-sustain vs decay',
      rationale: 'Epidemic models capture threshold dynamics that match momentum regime transitions. When R0 > 1 in epidemics, spread accelerates; analogously, when momentum "R0" > 1, trend self-reinforces.',
      complexity: 'medium',
      createdAt: new Date(Date.now() - 3600000).toISOString(),
      score: 0.82,
    },
    {
      id: 2,
      technique: 'inversion',
      domain: 'Market Microstructure',
      ideaText: 'Instead of predicting price direction, predict the failure of existing signals — trade the breakdown of MACD/RSI reliability as the signal itself',
      rationale: 'Signal decay is more predictable than signal direction. When a historically reliable indicator starts failing, it marks regime transition with high precision.',
      complexity: 'high',
      createdAt: new Date(Date.now() - 7200000).toISOString(),
      score: 0.74,
    },
    {
      id: 3,
      technique: 'combination',
      domain: 'Physics / Thermodynamics',
      ideaText: 'Apply entropy to order book state — measure Shannon entropy of bid/ask distribution; low entropy = concentrated liquidity = imminent large move',
      rationale: 'Order book entropy drops before large moves as liquidity concentrates. Low entropy predicts volatility expansion within 15 minutes with ~70% accuracy historically.',
      complexity: 'high',
      createdAt: new Date(Date.now() - 10800000).toISOString(),
      score: 0.91,
    },
    {
      id: 4,
      technique: 'mutation',
      domain: 'Existing Strategy',
      ideaText: 'Replace fixed RSI overbought/oversold thresholds with adaptive thresholds based on rolling percentile of RSI distribution over past 90 days',
      rationale: 'Fixed thresholds (30/70) ignore regime context. Adaptive thresholds using 15th/85th percentile of rolling RSI distribution match current market character.',
      complexity: 'low',
      createdAt: new Date(Date.now() - 14400000).toISOString(),
      score: 0.69,
    },
    {
      id: 5,
      technique: 'domain_borrow',
      domain: 'Ecology / Population Dynamics',
      ideaText: 'Model market maker inventory as predator-prey dynamics (Lotka-Volterra) — predict inventory exhaustion events that cause temporary price dislocations',
      rationale: 'Market makers oscillate between inventory accumulation and distribution cycles. Lotka-Volterra equations model these oscillations and predict equilibrium-breaking events.',
      complexity: 'high',
      createdAt: new Date(Date.now() - 18000000).toISOString(),
      score: 0.77,
    },
    {
      id: 6,
      technique: 'combination',
      domain: 'Social Network Analysis',
      ideaText: 'Build token correlation network; track "centrality" of BTC — when BTC loses centrality (correlation drops), altcoin season begins',
      rationale: 'BTC dominance metrics are lagged. Network centrality measures provide earlier signal of capital rotation. Declining BTC centrality precedes altcoin outperformance by 5-10 days.',
      complexity: 'medium',
      createdAt: new Date(Date.now() - 21600000).toISOString(),
      score: 0.86,
    },
  ]
}

function mockGenealogyGraph(): GenealogyGraph {
  const nodes: import('../types').GenealogyNode[] = []
  const edges: import('../types').GenealogyEdge[] = []
  let id = 1
  const islands: Island[] = ['BULL', 'BEAR', 'NEUTRAL']

  for (const island of islands) {
    // 3 generations, branching
    const gen1 = [id++, id++]
    for (const g of gen1) {
      nodes.push({ genomeId: g, island, generation: 1, fitness: 0.3 + Math.random() * 0.3, isHallOfFame: false, sharpe: 0.8 + Math.random() })
    }
    const gen2 = [id++, id++, id++]
    for (const g of gen2) {
      nodes.push({ genomeId: g, island, generation: 2, fitness: 0.5 + Math.random() * 0.3, isHallOfFame: false, sharpe: 1.2 + Math.random() })
      edges.push({ source: gen1[Math.floor(Math.random() * gen1.length)], target: g })
    }
    const gen3 = [id++, id++, id++, id++]
    for (const g of gen3) {
      const fit = 0.6 + Math.random() * 0.4
      nodes.push({ genomeId: g, island, generation: 3, fitness: fit, isHallOfFame: fit > 0.88, sharpe: 1.5 + Math.random() * 0.8 })
      edges.push({ source: gen2[Math.floor(Math.random() * gen2.length)], target: g })
    }
  }
  return { nodes, edges }
}

function mockMutationFrequencies(): MutationFrequency[] {
  return [
    { mutation: 'lookback_±5', count: 142, avgFitnessImprovement: 0.034 },
    { mutation: 'threshold_scale', count: 118, avgFitnessImprovement: 0.028 },
    { mutation: 'rsi_period_shift', count: 95, avgFitnessImprovement: 0.019 },
    { mutation: 'stop_loss_tighten', count: 87, avgFitnessImprovement: 0.041 },
    { mutation: 'macd_crossover_variant', count: 73, avgFitnessImprovement: 0.015 },
    { mutation: 'position_size_kelly', count: 61, avgFitnessImprovement: 0.052 },
    { mutation: 'atr_multiplier_±0.5', count: 54, avgFitnessImprovement: 0.022 },
    { mutation: 'vol_filter_remove', count: 38, avgFitnessImprovement: -0.008 },
  ]
}

function mockWeeklyReport(): WeeklyReport {
  return {
    id: 1,
    weekStart: '2026-03-30',
    weekEnd: '2026-04-05',
    markdownContent: `# Weekly Research Report — Apr 5, 2026

## Executive Summary

This week the IAE completed **generation 42** across all three islands, producing **3 new Hall of Fame genomes**. The most significant development was a regime shift from BULL → BEAR detected Monday morning with 87% confidence, triggering automatic island rebalancing.

## Evolution Progress

| Island | Generation | Best Fitness | Mean Fitness | Diversity |
|--------|-----------|-------------|-------------|-----------|
| BULL   | 42        | 0.921       | 0.743       | 0.31      |
| BEAR   | 38        | 0.887       | 0.712       | 0.44      |
| NEUTRAL| 45        | 0.863       | 0.698       | 0.52      |

**Observation:** BEAR island diversity remains elevated — the population has not yet converged, suggesting more exploration ahead. BULL island is approaching convergence (diversity 0.31), near-optimal strategy for bull conditions likely identified.

## Hypotheses Pipeline

- **2 adopted** this week: RSI-divergence mean-reversion (Sharpe 1.84) and funding-rate extremes reversal (Sharpe 2.11)
- **1 rejected**: MACD + volume spike combination — insufficient edge after transaction cost adjustment
- **5 pending** in queue — 3 from serendipity engine, 2 from academic papers

## Shadow Performance

Shadow #1 (Genome #12) completed **8 consecutive positive alpha days** and has been promoted to live trading. Shadow #3 is showing concerning alpha decay — under review.

## Notable Findings

### Entropy Signal (Serendipity)
The order-book entropy hypothesis (score 0.91) is the most promising serendipity idea this cycle. Initial backtests show a 70% accuracy rate for predicting volatility expansion within 15 minutes. Scheduling formal hypothesis test next week.

### Funding Rate Paper
The newly ingested SSRN paper on funding rate dynamics aligns with 2 existing genome strategies. Extracted hypotheses submitted to queue with high confidence (0.88).

## Next Week Priorities

1. Formal test of entropy signal hypothesis
2. Run generation 43 with increased mutation rate on BEAR island
3. Ingest 3 pending local research reports
4. Evaluate NEUTRAL island genome #31 for promotion
`,
    generatedAt: new Date().toISOString(),
    topGenomes: [77, 12, 31],
    hypothesesAdopted: 2,
    hypothesesRejected: 1,
    alerts: mockAlerts(),
  }
}

// ─── API Functions with Mock Fallback ─────────────────────────────────────────

async function withMockFallback<T>(
  apiFn: () => Promise<T>,
  mockFn: () => T
): Promise<T> {
  try {
    return await apiFn()
  } catch {
    // Backend not running — return mock data
    return mockFn()
  }
}

export async function fetchGenomes(island?: Island): Promise<Genome[]> {
  const qs = island ? `?island=${island}` : ''
  return withMockFallback(
    () => request<Genome[]>(`/genomes${qs}`),
    () => mockGenomes(island)
  )
}

export async function fetchHypotheses(status?: HypothesisStatus): Promise<Hypothesis[]> {
  const qs = status ? `?status=${status}` : ''
  return withMockFallback(
    () => request<Hypothesis[]>(`/hypotheses${qs}`),
    () => mockHypotheses(status)
  )
}

export async function fetchShadows(): Promise<Shadow[]> {
  return withMockFallback(
    () => request<Shadow[]>('/shadows'),
    mockShadows
  )
}

export async function fetchCounterfactuals(runId?: string): Promise<Counterfactual[]> {
  const qs = runId ? `?runId=${runId}` : ''
  return withMockFallback(
    () => request<Counterfactual[]>(`/counterfactuals${qs}`),
    mockCounterfactuals
  )
}

export async function fetchAlerts(): Promise<Alert[]> {
  return withMockFallback(
    () => request<Alert[]>('/alerts'),
    mockAlerts
  )
}

export async function acknowledgeAlert(id: number): Promise<void> {
  return withMockFallback(
    () => request<void>(`/alerts/${id}/ack`, { method: 'POST' }),
    () => undefined
  )
}

export async function triggerEvolution(island: Island): Promise<{ queued: boolean }> {
  return withMockFallback(
    () => request<{ queued: boolean }>(`/evolution/trigger`, {
      method: 'POST',
      body: JSON.stringify({ island }),
    }),
    () => ({ queued: true })
  )
}

export async function promoteGenome(genomeId: number): Promise<{ promoted: boolean }> {
  return withMockFallback(
    () => request<{ promoted: boolean }>(`/genomes/${genomeId}/promote`, { method: 'POST' }),
    () => ({ promoted: true })
  )
}

export async function fetchAcademicPapers(query?: string): Promise<AcademicPaper[]> {
  const qs = query ? `?q=${encodeURIComponent(query)}` : ''
  return withMockFallback(
    () => request<AcademicPaper[]>(`/academic${qs}`),
    mockAcademicPapers
  )
}

export async function fetchSerendipityIdeas(
  technique?: SerendipityTechnique
): Promise<SerendipityIdea[]> {
  const qs = technique ? `?technique=${technique}` : ''
  return withMockFallback(
    () => request<SerendipityIdea[]>(`/serendipity${qs}`),
    mockSerendipityIdeas
  )
}

export async function generateSerendipityIdeas(): Promise<SerendipityIdea[]> {
  return withMockFallback(
    () => request<SerendipityIdea[]>('/serendipity/generate', { method: 'POST' }),
    () => {
      const ideas = mockSerendipityIdeas()
      return ideas.map(i => ({ ...i, id: i.id + 100, createdAt: new Date().toISOString() }))
    }
  )
}

export async function submitIdeaAsHypothesis(ideaId: number): Promise<Hypothesis> {
  return withMockFallback(
    () => request<Hypothesis>(`/serendipity/${ideaId}/submit`, { method: 'POST' }),
    () => mockHypotheses()[0]
  )
}

export async function fetchGenealogyGraph(): Promise<GenealogyGraph> {
  return withMockFallback(
    () => request<GenealogyGraph>('/genealogy'),
    mockGenealogyGraph
  )
}

export async function fetchEvolutionStats(): Promise<EvolutionStats[]> {
  return withMockFallback(
    () => request<EvolutionStats[]>('/evolution/stats'),
    mockEvolutionStats
  )
}

export async function fetchMutationFrequencies(): Promise<MutationFrequency[]> {
  return withMockFallback(
    () => request<MutationFrequency[]>('/evolution/mutations'),
    mockMutationFrequencies
  )
}

export async function fetchWeeklyReport(): Promise<WeeklyReport> {
  return withMockFallback(
    () => request<WeeklyReport>('/narratives/weekly'),
    mockWeeklyReport
  )
}

export async function submitHypothesisTest(hypothesisId: number): Promise<{ queued: boolean }> {
  return withMockFallback(
    () => request<{ queued: boolean }>(`/hypotheses/${hypothesisId}/test`, { method: 'POST' }),
    () => ({ queued: true })
  )
}
