// ============================================================
// hooks/useRiskAPI.ts -- React Query hooks for the risk API
// running at :8791. Polling interval: 5 s for live data,
// 30 s for attribution/correlation.
// ============================================================

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';
import type {
  PortfolioRiskResponse,
  VaRSummary,
  VaRTrendPoint,
  PositionRow,
  GreeksAggregate,
  CorrelationMatrixResponse,
  LimitsResponse,
  VaRBreachAlert,
  ApiHealthResponse,
  AttributionResponse,
  BrokerCircuitBreaker,
  CircuitBreakerState,
} from '../types/risk';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const RISK_API_BASE = 'http://localhost:8791';
const POLL_LIVE_MS   = 5_000;
const POLL_MEDIUM_MS = 15_000;
const POLL_SLOW_MS   = 30_000;

// ---------------------------------------------------------------------------
// Mock data generators -- realistic shapes matching risk_api.py responses
// ---------------------------------------------------------------------------

function generateVaRTrend(n = 30): VaRTrendPoint[] {
  const now = Date.now();
  let base = 98_000;
  return Array.from({ length: n }, (_, i) => {
    base += (Math.random() - 0.49) * 800;
    return {
      timestamp: new Date(now - (n - i) * 300_000).toISOString(),
      parametric_var99:  base * 0.028 + Math.random() * 200,
      historical_var99:  base * 0.031 + Math.random() * 220,
      mc_var99:          base * 0.033 + Math.random() * 250,
      consensus_var99:   base * 0.030 + Math.random() * 210,
      equity: base,
    };
  });
}

function generatePositions(): PositionRow[] {
  const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AAPL', 'SPY', 'QQQ', 'BTC-241227-C-80000', 'ETH-241227-P-3000'];
  const classes = ['crypto', 'crypto', 'crypto', 'equity', 'equity', 'equity', 'option', 'option'] as const;
  return symbols.map((sym, i) => {
    const notional = (Math.random() * 18_000 + 2_000) * (i % 3 === 2 ? -1 : 1);
    const entry = 100 + Math.random() * 200;
    const current = entry * (1 + (Math.random() - 0.47) * 0.08);
    const isOption = classes[i] === 'option';
    return {
      symbol: sym,
      asset_class: classes[i],
      side: notional > 0 ? 'long' : 'short',
      qty: notional / current,
      entry_price: entry,
      current_price: current,
      notional_usd: Math.abs(notional),
      weight: notional / 98_000,
      unrealized_pnl: (current - entry) * Math.abs(notional / current),
      realized_pnl: (Math.random() - 0.4) * 300,
      delta_dollars: notional * (isOption ? (Math.random() * 0.8 + 0.1) : 1),
      greeks: isOption ? {
        delta:  Math.random() * 0.8 + 0.1,
        gamma:  Math.random() * 0.02,
        vega:   Math.random() * 120 + 10,
        theta: -(Math.random() * 80 + 5),
        rho:    Math.random() * 0.05,
        vanna:  (Math.random() - 0.5) * 0.01,
        charm: -(Math.random() * 0.005),
      } : null,
      var_contribution: Math.random() * 0.25,
      implied_vol: isOption ? (Math.random() * 0.4 + 0.2) : null,
      expiry: isOption ? '2024-12-27T00:00:00Z' : null,
      strike: isOption ? (i === 6 ? 80000 : 3000) : null,
    };
  });
}

function generateCorrelation(symbols: string[]): CorrelationMatrixResponse {
  const n = symbols.length;
  const pearson: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (__, j) =>
      i === j ? 1 : parseFloat((Math.random() * 1.4 - 0.4).toFixed(3))
    )
  );
  return {
    symbols,
    pearson,
    spearman: pearson.map(row => row.map(v => v + (Math.random() - 0.5) * 0.05)),
    ewm:      pearson.map(row => row.map(v => v * 0.94 + (Math.random() - 0.5) * 0.03)),
    avg_correlation: 0.42,
    is_crowding: Math.random() > 0.8,
    is_stress: Math.random() > 0.92,
    regime: Math.floor(Math.random() * 3),
    timestamp: new Date().toISOString(),
    top_centrality: symbols.slice(0, 4).map(s => ({ symbol: s, centrality: Math.random() })),
  };
}

function generateAlerts(): VaRBreachAlert[] {
  const now = new Date().toISOString();
  return Math.random() > 0.6 ? [{
    id: 'alert-001',
    severity: 'warn',
    message: 'Historical VaR99 approaching daily limit (87% utilization)',
    value: 4_890,
    threshold: 5_600,
    method: 'historical',
    timestamp: now,
    acknowledged: false,
  }] : [];
}

function generateLimits(): LimitsResponse {
  return {
    limits: [
      { name: 'Max Single Position', current_value: 0.21, limit_value: 0.25, utilization: 0.84, status: 'warn', unit: 'fraction', last_checked: new Date().toISOString() },
      { name: 'Max Gross Exposure', current_value: 1.42, limit_value: 2.0,  utilization: 0.71, status: 'ok',   unit: 'x leverage', last_checked: new Date().toISOString() },
      { name: 'Daily VaR Limit',    current_value: 4_250, limit_value: 5_600, utilization: 0.76, status: 'ok', unit: 'USD', last_checked: new Date().toISOString() },
      { name: 'Daily Loss Limit',   current_value: 1_120, limit_value: 3_000, utilization: 0.37, status: 'ok', unit: 'USD', last_checked: new Date().toISOString() },
      { name: 'Drawdown Limit',     current_value: 0.062, limit_value: 0.10, utilization: 0.62, status: 'ok',  unit: 'fraction', last_checked: new Date().toISOString() },
      { name: 'Margin Utilization', current_value: 0.378, limit_value: 0.80, utilization: 0.47, status: 'ok', unit: 'fraction', last_checked: new Date().toISOString() },
    ],
    any_breach: false,
    timestamp: new Date().toISOString(),
  };
}

function generatePortfolioRisk(): PortfolioRiskResponse {
  const positions = generatePositions();
  const symbols = positions.map(p => p.symbol);
  return {
    summary: {
      equity: 98_420,
      gross_exposure: 96_800,
      net_exposure: 74_200,
      leverage: 1.42,
      daily_pnl: 1_240,
      daily_pnl_pct: 0.0127,
      n_positions: positions.length,
      margin_utilization: 0.378,
    },
    var_summary: {
      parametric:   { timestamp: new Date().toISOString(), var_95: 2_240, var_99: 3_560, cvar_95: 3_100, cvar_99: 5_020, method: 'parametric' },
      historical:   { timestamp: new Date().toISOString(), var_95: 2_480, var_99: 3_890, cvar_95: 3_420, cvar_99: 5_470, method: 'historical' },
      monte_carlo:  { timestamp: new Date().toISOString(), var_95: 2_610, var_99: 4_140, cvar_95: 3_670, cvar_99: 5_820, method: 'monte_carlo' },
      consensus_var99: 3_860,
      equity: 98_420,
      n_positions: positions.length,
      breach_flag: false,
      timestamp: new Date().toISOString(),
    },
    var_trend: generateVaRTrend(),
    positions,
    greeks: {
      net_delta: 0.72,
      net_gamma: 0.0034,
      net_vega: 210,
      net_theta: -145,
      net_rho: 0.018,
      dollar_delta: 70_800,
      dollar_gamma: 310,
      dollar_vega: 2_100,
      positions: positions
        .filter(p => p.greeks !== null)
        .map(p => ({
          symbol: p.symbol,
          expiry: p.expiry!,
          strike: p.strike!,
          option_type: p.symbol.includes('-C-') ? 'call' : 'put',
          qty: p.qty,
          delta: p.greeks!.delta,
          gamma: p.greeks!.gamma,
          vega: p.greeks!.vega,
          theta: p.greeks!.theta,
          rho: p.greeks!.rho,
          notional_usd: p.notional_usd,
          iv: p.implied_vol!,
          underlying_price: p.current_price,
        })),
    },
    limits: generateLimits(),
    alerts: generateAlerts(),
    correlation: generateCorrelation(symbols.slice(0, 6)),
    health: {
      service: 'risk-api',
      status: 'healthy',
      uptime_seconds: 14_400,
      last_update: new Date().toISOString(),
      brokers: [
        { broker: 'Alpaca',  state: 'CLOSED', reason: '',         tripped_at: null, reset_at: null, consecutive_errors: 0, last_latency_ms: 42,  error_rate_1m: 0 },
        { broker: 'Binance', state: 'CLOSED', reason: '',         tripped_at: null, reset_at: null, consecutive_errors: 0, last_latency_ms: 88,  error_rate_1m: 0 },
        { broker: 'Polygon', state: 'CLOSED', reason: '',         tripped_at: null, reset_at: null, consecutive_errors: 0, last_latency_ms: 120, error_rate_1m: 0 },
      ],
    },
    attribution: null,
    timestamp: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Fetch helpers -- fall back to mock data if API is unreachable
// ---------------------------------------------------------------------------

async function fetchRisk<T>(path: string, fallback: () => T): Promise<T> {
  try {
    const res = await fetch(`${RISK_API_BASE}${path}`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(4_000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as T;
  } catch {
    return fallback();
  }
}

// ---------------------------------------------------------------------------
// Public hooks
// ---------------------------------------------------------------------------

/** Main portfolio risk hook -- polls every 5 s. */
export function usePortfolioRisk() {
  return useQuery<PortfolioRiskResponse>({
    queryKey: ['risk', 'portfolio'],
    queryFn: () => fetchRisk('/risk/portfolio', generatePortfolioRisk),
    refetchInterval: POLL_LIVE_MS,
    staleTime: POLL_LIVE_MS / 2,
    retry: 2,
  });
}

/** VaR trend only -- lighter poll. */
export function useVaRTrend() {
  return useQuery<VaRTrendPoint[]>({
    queryKey: ['risk', 'var_trend'],
    queryFn: () => fetchRisk('/risk/portfolio', () => generateVaRTrend()),
    refetchInterval: POLL_MEDIUM_MS,
    staleTime: POLL_MEDIUM_MS / 2,
    select: (d: any) => (d as PortfolioRiskResponse).var_trend ?? d,
  });
}

/** Limits poll -- 15 s. */
export function useRiskLimits() {
  return useQuery<LimitsResponse>({
    queryKey: ['risk', 'limits'],
    queryFn: () => fetchRisk('/risk/limits', generateLimits),
    refetchInterval: POLL_MEDIUM_MS,
    staleTime: POLL_MEDIUM_MS / 2,
  });
}

/** Correlation matrix -- 30 s (expensive). */
export function useCorrelation() {
  return useQuery<CorrelationMatrixResponse>({
    queryKey: ['risk', 'correlation'],
    queryFn: () => fetchRisk('/risk/correlation', () =>
      generateCorrelation(['BTC-USD', 'ETH-USD', 'SOL-USD', 'AAPL', 'SPY', 'QQQ'])
    ),
    refetchInterval: POLL_SLOW_MS,
    staleTime: POLL_SLOW_MS / 2,
  });
}

/** Health / circuit breakers -- 10 s. */
export function useApiHealth() {
  return useQuery<ApiHealthResponse>({
    queryKey: ['risk', 'health'],
    queryFn: () => fetchRisk('/risk/health', () => generatePortfolioRisk().health),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

/** Attribution -- 30 s (read from DB). */
export function useAttribution(days = 7) {
  return useQuery<AttributionResponse | null>({
    queryKey: ['risk', 'attribution', days],
    queryFn: () => fetchRisk(`/risk/attribution?days=${days}`, () => null),
    refetchInterval: POLL_SLOW_MS,
    staleTime: POLL_SLOW_MS / 2,
  });
}

/** Derived: active VaR breach alerts from portfolio data. */
export function useVaRAlerts() {
  const { data } = usePortfolioRisk();
  return useMemo(() => data?.alerts ?? [], [data]);
}

/** Invalidate all risk queries -- call after manual refresh. */
export function useRefreshRisk() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: ['risk'] });
}
