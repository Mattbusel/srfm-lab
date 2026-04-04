// ============================================================
// SPACETIME API CLIENT — typed API for the spacetime backend
// ============================================================
import type {
  ApiResponse,
  LiveState,
  BHHistoryResponse,
  BHScanResponse,
  BHTimeframe,
  StrategyGraph,
  BacktestConfig,
  BacktestResult,
  BacktestMetrics,
  BacktestEquityPoint,
  BacktestTrade,
  DrawdownPeriod,
  StrategyFactorAnalysis,
  OHLCV,
  VolumeProfile,
} from '@/types'
import { useSettingsStore } from '@/store/settingsStore'

const getBaseUrl = () => useSettingsStore.getState().settings.apiUrl
const generateId = () => Math.random().toString(36).slice(2, 11)

class SpacetimeApiClient {
  private async fetch<T>(path: string, options: RequestInit = {}): Promise<T> {
    const url = `${getBaseUrl()}${path}`
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    })

    if (!response.ok) {
      const errorBody = await response.text().catch(() => '')
      throw new Error(`API error ${response.status}: ${errorBody || response.statusText}`)
    }

    return response.json() as T
  }

  // ---- BH State ----
  async getLiveState(): Promise<LiveState> {
    const resp = await this.fetch<ApiResponse<LiveState>>('/api/v1/bh/live')
    if (!resp.data) throw new Error('No data in response')
    return resp.data
  }

  async getBHHistory(symbol: string, tf: BHTimeframe, limit = 200): Promise<BHHistoryResponse> {
    return this.fetch<BHHistoryResponse>(
      `/api/v1/bh/history?symbol=${encodeURIComponent(symbol)}&tf=${tf}&limit=${limit}`
    )
  }

  async getBHScan(filter?: { activeOnly?: boolean; minMass?: number }): Promise<BHScanResponse> {
    const params = new URLSearchParams()
    if (filter?.activeOnly) params.set('active_only', 'true')
    if (filter?.minMass !== undefined) params.set('min_mass', String(filter.minMass))
    return this.fetch<BHScanResponse>(`/api/v1/bh/scan?${params.toString()}`)
  }

  // ---- Chart Data ----
  async getBars(symbol: string, interval: string, limit = 500): Promise<OHLCV[]> {
    const resp = await this.fetch<ApiResponse<OHLCV[]>>(
      `/api/v1/bars?symbol=${encodeURIComponent(symbol)}&interval=${interval}&limit=${limit}`
    )
    return resp.data ?? []
  }

  async getBarsRange(symbol: string, interval: string, start: string, end: string): Promise<OHLCV[]> {
    const params = new URLSearchParams({ symbol, interval, start, end })
    const resp = await this.fetch<ApiResponse<OHLCV[]>>(`/api/v1/bars/range?${params.toString()}`)
    return resp.data ?? []
  }

  async getVolumeProfile(symbol: string, startTime: number, endTime: number): Promise<VolumeProfile> {
    const params = new URLSearchParams({
      symbol,
      start: String(startTime),
      end: String(endTime),
    })
    const resp = await this.fetch<ApiResponse<VolumeProfile>>(`/api/v1/volume-profile?${params.toString()}`)
    if (!resp.data) throw new Error('No data')
    return resp.data
  }

  // ---- Backtesting ----
  async runBacktest(
    graph: StrategyGraph,
    config: BacktestConfig,
    onProgress?: (pct: number) => void
  ): Promise<BacktestResult> {
    // Poll-based backtest with progress
    const resp = await this.fetch<{ jobId: string }>('/api/v1/backtest/submit', {
      method: 'POST',
      body: JSON.stringify({ graph, config }),
    })

    const { jobId } = resp

    // Poll for completion
    let attempts = 0
    const maxAttempts = 300  // 5 minutes at 1s intervals

    while (attempts < maxAttempts) {
      await new Promise((resolve) => setTimeout(resolve, 1000))
      attempts++

      const status = await this.fetch<{
        status: 'running' | 'success' | 'error'
        progress: number
        result?: BacktestResult
        error?: string
      }>(`/api/v1/backtest/status/${jobId}`)

      onProgress?.(status.progress ?? 0)

      if (status.status === 'success' && status.result) {
        return status.result
      }

      if (status.status === 'error') {
        throw new Error(status.error ?? 'Backtest failed')
      }
    }

    throw new Error('Backtest timed out')
  }

  async getBacktestResults(graphId: string): Promise<BacktestResult[]> {
    const resp = await this.fetch<ApiResponse<BacktestResult[]>>(
      `/api/v1/backtest/results?graph_id=${encodeURIComponent(graphId)}`
    )
    return resp.data ?? []
  }

  // ---- Strategy Analysis ----
  async runFactorAnalysis(
    graph: StrategyGraph,
    symbol: string,
    dateRange: { start: string; end: string }
  ): Promise<StrategyFactorAnalysis> {
    const resp = await this.fetch<ApiResponse<StrategyFactorAnalysis>>('/api/v1/strategy/factor-analysis', {
      method: 'POST',
      body: JSON.stringify({ graph, symbol, startDate: dateRange.start, endDate: dateRange.end }),
    })
    if (!resp.data) throw new Error('No data')
    return resp.data
  }

  // ---- Health ----
  async healthCheck(): Promise<{ status: string; version: string; uptime: number }> {
    return this.fetch('/api/v1/health')
  }

  // ---- Mock/Demo data generators (used when API is unavailable) ----
  generateMockBacktestResult(config: BacktestConfig): BacktestResult {
    const now = Date.now()
    const startMs = new Date(config.startDate).getTime()
    const endMs = new Date(config.endDate).getTime()
    const daysTotal = (endMs - startMs) / 86400000

    // Generate synthetic equity curve
    let equity = config.initialCapital
    const equityCurve: BacktestEquityPoint[] = []
    const trades: BacktestTrade[] = []
    let peakEquity = equity
    let maxDrawdown = 0
    let maxDrawdownPct = 0

    for (let d = 0; d <= daysTotal; d++) {
      const t = startMs + d * 86400000
      const dailyReturn = (Math.random() - 0.48) * 0.02  // slight positive bias
      equity *= 1 + dailyReturn
      peakEquity = Math.max(peakEquity, equity)
      const dd = peakEquity - equity
      const ddPct = dd / peakEquity
      maxDrawdown = Math.max(maxDrawdown, dd)
      maxDrawdownPct = Math.max(maxDrawdownPct, ddPct)

      equityCurve.push({
        time: Math.floor(t / 1000),
        equity,
        drawdown: dd,
        drawdownPct: ddPct,
        position: Math.random() > 0.5 ? 1 : 0,
      })

      // Random trades
      if (Math.random() < 0.05) {
        const entryPrice = 100 + Math.random() * 50
        const exitPrice = entryPrice * (1 + (Math.random() - 0.45) * 0.05)
        const pnl = (exitPrice - entryPrice) * 100
        trades.push({
          entryTime: Math.floor(t / 1000),
          exitTime: Math.floor((t + 86400000 * Math.floor(Math.random() * 5 + 1)) / 1000),
          side: 'long',
          entryPrice,
          exitPrice,
          qty: 100,
          pnl,
          pnlPct: (exitPrice - entryPrice) / entryPrice,
          holdingBars: Math.floor(Math.random() * 10 + 1),
          entrySignal: 'BH Formation',
          exitSignal: 'BH Reversal',
        })
      }
    }

    const wins = trades.filter((t) => t.pnl > 0)
    const losses = trades.filter((t) => t.pnl < 0)
    const totalReturn = (equity - config.initialCapital) / config.initialCapital
    const annualizedReturn = Math.pow(1 + totalReturn, 365 / daysTotal) - 1
    const returns = equityCurve.map((p, i) =>
      i > 0 ? (p.equity - equityCurve[i - 1].equity) / equityCurve[i - 1].equity : 0
    )
    const avgReturn = returns.reduce((s, r) => s + r, 0) / returns.length
    const stdReturn = Math.sqrt(returns.reduce((s, r) => s + Math.pow(r - avgReturn, 2), 0) / returns.length)
    const sharpe = stdReturn > 0 ? (avgReturn / stdReturn) * Math.sqrt(252) : 0

    const metrics: BacktestMetrics = {
      totalReturn: equity - config.initialCapital,
      totalReturnPct: totalReturn,
      annualizedReturn,
      annualizedVolatility: stdReturn * Math.sqrt(252),
      sharpe,
      sortino: sharpe * 1.2,
      calmar: maxDrawdownPct > 0 ? annualizedReturn / maxDrawdownPct : 0,
      maxDrawdown,
      maxDrawdownPct,
      maxDrawdownDuration: 45,
      winRate: trades.length > 0 ? wins.length / trades.length : 0,
      numTrades: trades.length,
      numWins: wins.length,
      numLosses: losses.length,
      avgWin: wins.length > 0 ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0,
      avgLoss: losses.length > 0 ? losses.reduce((s, t) => s + t.pnl, 0) / losses.length : 0,
      profitFactor: losses.length > 0 ? wins.reduce((s, t) => s + t.pnl, 0) / Math.abs(losses.reduce((s, t) => s + t.pnl, 0)) : 2,
      expectancy: trades.length > 0 ? trades.reduce((s, t) => s + t.pnl, 0) / trades.length : 0,
      avgHoldingPeriod: trades.length > 0 ? trades.reduce((s, t) => s + t.holdingBars, 0) / trades.length : 0,
      bestTrade: Math.max(...trades.map((t) => t.pnl), 0),
      worstTrade: Math.min(...trades.map((t) => t.pnl), 0),
      startDate: config.startDate,
      endDate: config.endDate,
      initialCapital: config.initialCapital,
      finalCapital: equity,
      totalCommission: trades.length * config.commission * config.initialCapital,
      totalSlippage: trades.length * config.slippage * config.initialCapital,
    }

    return {
      id: generateId(),
      graphId: '',
      config,
      metrics,
      equityCurve,
      trades,
      drawdowns: [],
      monthlyReturns: [],
      runAt: now,
      duration: 1200,
      status: 'success',
    }
  }
}

export const spacetimeApi = new SpacetimeApiClient()
