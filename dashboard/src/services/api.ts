// ============================================================
// api.ts — REST client for all Spacetime Arena endpoints
// ============================================================
import type {
  ApiResponse,
  PaginatedResponse,
  PortfolioSnapshot,
  Position,
  Trade,
  RiskMetrics,
  AttributionData,
  SignalCard,
  CoinData,
  EquityPoint,
  DrawdownPoint,
  CorrelationEntry,
  BHFormation,
  Timeframe,
} from '@/types'

const BASE_URL = import.meta.env.VITE_API_BASE ?? 'http://localhost:8765'

// ============================================================
// Core fetch wrapper
// ============================================================

async function apiFetch<T>(
  path: string,
  options?: RequestInit,
): Promise<ApiResponse<T>> {
  const url = `${BASE_URL}${path}`
  try {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json', ...options?.headers },
      ...options,
    })
    if (!res.ok) {
      const text = await res.text()
      return { data: null as unknown as T, ok: false, error: text, timestamp: new Date().toISOString() }
    }
    const json = await res.json()
    return { data: json as T, ok: true, timestamp: new Date().toISOString() }
  } catch (err) {
    return {
      data: null as unknown as T,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
      timestamp: new Date().toISOString(),
    }
  }
}

// ============================================================
// Portfolio endpoints
// ============================================================

export const portfolioApi = {
  getSnapshot(): Promise<ApiResponse<PortfolioSnapshot>> {
    return apiFetch('/api/portfolio/snapshot')
  },

  getEquityCurve(params?: {
    from?: string
    to?: string
    interval?: '1h' | '4h' | '1d'
  }): Promise<ApiResponse<EquityPoint[]>> {
    const qs = new URLSearchParams(params as Record<string, string>).toString()
    return apiFetch(`/api/portfolio/equity${qs ? `?${qs}` : ''}`)
  },

  getDrawdown(params?: { from?: string; to?: string }): Promise<ApiResponse<DrawdownPoint[]>> {
    const qs = new URLSearchParams(params as Record<string, string>).toString()
    return apiFetch(`/api/portfolio/drawdown${qs ? `?${qs}` : ''}`)
  },

  getMetrics(period?: '1d' | '7d' | '30d' | 'ytd'): Promise<ApiResponse<PortfolioSnapshot>> {
    return apiFetch(`/api/portfolio/metrics${period ? `?period=${period}` : ''}`)
  },

  getDailyPnl(days?: number): Promise<ApiResponse<{ date: string; pnl: number; pnlPct: number }[]>> {
    return apiFetch(`/api/portfolio/daily-pnl${days != null ? `?days=${days}` : ''}`)
  },
}

// ============================================================
// Positions endpoints
// ============================================================

export const positionsApi = {
  getAll(): Promise<ApiResponse<Position[]>> {
    return apiFetch('/api/positions')
  },

  getBySymbol(symbol: string): Promise<ApiResponse<Position>> {
    return apiFetch(`/api/positions/${symbol}`)
  },

  getConcentration(): Promise<ApiResponse<{ symbol: string; weight: number }[]>> {
    return apiFetch('/api/positions/concentration')
  },
}

// ============================================================
// Trades endpoints
// ============================================================

export const tradesApi = {
  getHistory(params?: {
    page?: number
    pageSize?: number
    symbol?: string
    strategy?: string
    from?: string
    to?: string
    side?: 'long' | 'short'
    exitReason?: string
  }): Promise<ApiResponse<PaginatedResponse<Trade>>> {
    const qs = new URLSearchParams(
      Object.fromEntries(
        Object.entries(params ?? {}).filter(([, v]) => v != null).map(([k, v]) => [k, String(v)]),
      ),
    ).toString()
    return apiFetch(`/api/trades${qs ? `?${qs}` : ''}`)
  },

  getStats(period?: string): Promise<ApiResponse<{
    totalTrades: number
    winRate: number
    avgWin: number
    avgLoss: number
    avgDurationMs: number
    expectancy: number
    profitFactor: number
    longestWinStreak: number
    longestLossStreak: number
  }>> {
    return apiFetch(`/api/trades/stats${period ? `?period=${period}` : ''}`)
  },

  getBestWorst(n?: number): Promise<ApiResponse<{ best: Trade[]; worst: Trade[] }>> {
    return apiFetch(`/api/trades/best-worst${n != null ? `?n=${n}` : ''}`)
  },
}

// ============================================================
// Risk endpoints
// ============================================================

export const riskApi = {
  getMetrics(): Promise<ApiResponse<RiskMetrics>> {
    return apiFetch('/api/risk/metrics')
  },

  getCorrelationMatrix(
    symbols?: string[],
    lookback?: number,
  ): Promise<ApiResponse<CorrelationEntry[]>> {
    const params: Record<string, string> = {}
    if (symbols?.length) params.symbols = symbols.join(',')
    if (lookback != null) params.lookback = String(lookback)
    const qs = new URLSearchParams(params).toString()
    return apiFetch(`/api/risk/correlation${qs ? `?${qs}` : ''}`)
  },

  getLimits(): Promise<ApiResponse<{ name: string; limit: number; used: number; utilization: number }[]>> {
    return apiFetch('/api/risk/limits')
  },
}

// ============================================================
// Attribution endpoints
// ============================================================

export const attributionApi = {
  getAttribution(params?: {
    period?: string
    from?: string
    to?: string
  }): Promise<ApiResponse<AttributionData>> {
    const qs = new URLSearchParams(params as Record<string, string>).toString()
    return apiFetch(`/api/attribution${qs ? `?${qs}` : ''}`)
  },
}

// ============================================================
// Signal / BH endpoints
// ============================================================

export const signalApi = {
  getSignalCards(symbols?: string[]): Promise<ApiResponse<SignalCard[]>> {
    const qs = symbols?.length ? `?symbols=${symbols.join(',')}` : ''
    return apiFetch(`/api/signals/cards${qs}`)
  },

  getFormations(params?: {
    symbol?: string
    timeframe?: Timeframe
    state?: string
    limit?: number
  }): Promise<ApiResponse<BHFormation[]>> {
    const qs = new URLSearchParams(
      Object.fromEntries(
        Object.entries(params ?? {}).filter(([, v]) => v != null).map(([k, v]) => [k, String(v)]),
      ),
    ).toString()
    return apiFetch(`/api/signals/formations${qs ? `?${qs}` : ''}`)
  },

  getActivationHistory(params?: {
    symbol?: string
    from?: string
    to?: string
  }): Promise<ApiResponse<{ date: string; symbol: string; count: number }[]>> {
    const qs = new URLSearchParams(params as Record<string, string>).toString()
    return apiFetch(`/api/signals/activation-history${qs ? `?${qs}` : ''}`)
  },
}

// ============================================================
// Market data endpoints
// ============================================================

export const marketApi = {
  getCoinData(symbols?: string[]): Promise<ApiResponse<CoinData[]>> {
    const qs = symbols?.length ? `?symbols=${symbols.join(',')}` : ''
    return apiFetch(`/api/market/coins${qs}`)
  },

  getCandles(params: {
    symbol: string
    timeframe: Timeframe
    from?: string
    to?: string
    limit?: number
  }): Promise<ApiResponse<{ t: number; o: number; h: number; l: number; c: number; v: number }[]>> {
    const qs = new URLSearchParams(
      Object.fromEntries(
        Object.entries(params).filter(([, v]) => v != null).map(([k, v]) => [k, String(v)]),
      ),
    ).toString()
    return apiFetch(`/api/market/candles?${qs}`)
  },

  getTicker(symbol: string): Promise<ApiResponse<{
    symbol: string
    price: number
    bid: number
    ask: number
    change24h: number
    volume24h: number
    openInterest: number
    fundingRate: number
  }>> {
    return apiFetch(`/api/market/ticker/${symbol}`)
  },
}
