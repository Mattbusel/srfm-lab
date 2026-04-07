// ============================================================
// hooks/useExecutionAPI.ts -- React Query hooks for execution
// analytics data. Falls back to realistic mock data when the
// execution API at :8792 is unreachable.
// ============================================================

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState, useCallback } from 'react';
import { ManagedWebSocket, resolveWsUrl } from '../utils/websocket';
import type {
  OrderRecord,
  TCAResult,
  VenueScore,
  ExecutionSummary,
  ExecutionCostPoint,
  OrderStreamEvent,
  OrdersResponse,
  TCAResponse,
  VenueScorecardResponse,
  ExecutionSummaryResponse,
} from '../types/execution';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const EXEC_API_BASE = 'http://localhost:8792';
const POLL_LIVE_MS  = 5_000;
const POLL_SLOW_MS  = 60_000;
const MAX_STREAM_ORDERS = 200;

// ---------------------------------------------------------------------------
// Mock data generators
// ---------------------------------------------------------------------------

const VENUES   = ['Alpaca', 'Binance', 'Coinbase', 'Kraken', 'NYSE', 'NASDAQ'];
const SYMBOLS  = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AAPL', 'SPY', 'QQQ', 'MSFT', 'TSLA'];
const STRATEGIES = ['momentum', 'mean_reversion', 'stat_arb', 'trend_follow', 'market_make'];
const SIDES    = ['buy', 'sell'] as const;
const STATUSES = ['filled', 'pending', 'partial', 'rejected', 'cancelled'] as const;
const TYPES    = ['market', 'limit', 'twap', 'vwap'] as const;

function randEl<T>(arr: readonly T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randRange(lo: number, hi: number): number {
  return lo + Math.random() * (hi - lo);
}

let _orderSeq = 1000;

function genOrder(minsAgo = 0): OrderRecord {
  const status = randEl(STATUSES);
  const symbol = randEl(SYMBOLS);
  const side   = randEl(SIDES);
  const qty    = parseFloat(randRange(0.01, 10).toFixed(4));
  const oPrice = randRange(20, 50_000);
  const filled = status === 'filled'  ? qty
               : status === 'partial' ? parseFloat((qty * randRange(0.1, 0.9)).toFixed(4))
               : 0;
  const now = Date.now() - minsAgo * 60_000;
  return {
    id:         `ORD-${++_orderSeq}`,
    symbol,
    side,
    orderType:  randEl(TYPES),
    qty,
    filledQty:  filled,
    orderPrice: randEl(TYPES) === 'market' ? null : oPrice,
    fillPrice:  filled > 0 ? parseFloat((oPrice * randRange(0.998, 1.002)).toFixed(4)) : null,
    orderTime:  new Date(now - randRange(100, 10_000)).toISOString(),
    fillTime:   filled > 0 ? new Date(now).toISOString() : null,
    venue:      randEl(VENUES),
    strategy:   randEl(STRATEGIES),
    status,
    notionalUsd:  filled > 0 ? filled * oPrice : undefined,
    slippageBps:  filled > 0 ? parseFloat(randRange(-2, 15).toFixed(2)) : undefined,
  };
}

function genOrders(n: number): OrderRecord[] {
  return Array.from({ length: n }, (_, i) => genOrder(i * 0.1));
}

function genTCA(orderId: string, symbol: string, venue: string, strategy: string): TCAResult {
  const arrival  = randRange(20, 50_000);
  const impact   = randRange(0.5, 8);
  const timing   = randRange(-1, 4);
  const spread   = randRange(0.3, 3);
  const total    = impact + timing + spread;
  const vwap     = arrival * randRange(0.999, 1.003);
  const fill     = arrival * (1 + total / 10_000);
  return {
    orderId,
    symbol,
    venue,
    strategy,
    implShortfallBps:  parseFloat(total.toFixed(2)),
    marketImpactBps:   parseFloat(impact.toFixed(2)),
    timingCostBps:     parseFloat(timing.toFixed(2)),
    spreadCostBps:     parseFloat(spread.toFixed(2)),
    totalCostBps:      parseFloat(total.toFixed(2)),
    vwapSlippageBps:   parseFloat(randRange(-1, 6).toFixed(2)),
    arrivalPrice:      arrival,
    periodVwap:        vwap,
    fillPrice:         fill,
    fillTime:          new Date(Date.now() - randRange(0, 3_600_000)).toISOString(),
    notionalUsd:       parseFloat((randRange(500, 50_000)).toFixed(0)),
  };
}

function genTCAResults(n: number): TCAResult[] {
  return Array.from({ length: n }, (_, i) =>
    genTCA(`ORD-${1000 + i}`, randEl(SYMBOLS), randEl(VENUES), randEl(STRATEGIES))
  );
}

function genVenueScores(): VenueScore[] {
  return VENUES.map(venue => {
    const slipBps = parseFloat(randRange(1, 14).toFixed(2));
    const fillRate = parseFloat(randRange(0.82, 0.99).toFixed(3));
    const fillMs   = parseFloat(randRange(40, 800).toFixed(0));
    // Score: lower slip => higher, higher fill => higher, lower time => higher
    const score = Math.min(100, Math.max(0,
      100 - slipBps * 4 + fillRate * 20 - fillMs * 0.05
    ));
    return {
      venue,
      avgSlippageBps:     slipBps,
      fillRate,
      avgFillTimeMs:      fillMs,
      score:              parseFloat(score.toFixed(1)),
      nTrades:            Math.floor(randRange(20, 400)),
      avgMarketImpactBps: parseFloat(randRange(0.5, 7).toFixed(2)),
      avgSpreadCostBps:   parseFloat(randRange(0.3, 3).toFixed(2)),
      p95FillTimeMs:      parseFloat((fillMs * randRange(2, 4)).toFixed(0)),
    };
  });
}

function genSummary(date: string): ExecutionSummary {
  const total     = Math.floor(randRange(60, 300));
  const filled    = Math.floor(total * randRange(0.82, 0.97));
  const rejected  = Math.floor(total * randRange(0, 0.04));
  const cancelled = total - filled - rejected;
  const scores    = genVenueScores().sort((a, b) => a.avgSlippageBps - b.avgSlippageBps);
  return {
    date,
    totalOrders:      total,
    filledOrders:     filled,
    rejectedOrders:   rejected,
    cancelledOrders:  Math.max(0, cancelled),
    avgSlippageBps:   parseFloat(randRange(2, 10).toFixed(2)),
    totalCostBps:     parseFloat(randRange(3, 15).toFixed(2)),
    totalNotionalUsd: parseFloat(randRange(200_000, 5_000_000).toFixed(0)),
    bestVenue:        scores[0].venue,
    worstVenue:       scores[scores.length - 1].venue,
    fillRate:         filled / total,
    avgFillTimeMs:    parseFloat(randRange(80, 500).toFixed(0)),
  };
}

function genCostTrend(days = 30): ExecutionCostPoint[] {
  let rolling = 6;
  return Array.from({ length: days }, (_, i) => {
    const d = new Date();
    d.setDate(d.getDate() - (days - 1 - i));
    const daily = parseFloat(randRange(2, 14).toFixed(2));
    rolling = parseFloat((rolling * 0.85 + daily * 0.15).toFixed(2));
    return {
      date:            d.toISOString().slice(0, 10),
      avgCostBpsDaily: daily,
      avgCostBps7d:    rolling,
      totalNotionalUsd: parseFloat(randRange(200_000, 3_000_000).toFixed(0)),
      nFills:           Math.floor(randRange(40, 250)),
    };
  });
}

// ---------------------------------------------------------------------------
// Fetch helper -- falls back to mock on any error
// ---------------------------------------------------------------------------

async function fetchExec<T>(path: string, fallback: () => T): Promise<T> {
  try {
    const res = await fetch(`${EXEC_API_BASE}${path}`, {
      headers: { Accept: 'application/json' },
      signal:  AbortSignal.timeout(4_000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as T;
  } catch {
    return fallback();
  }
}

// ---------------------------------------------------------------------------
// useRecentOrders
// ---------------------------------------------------------------------------

/**
 * Fetch the most recent N orders. Polls every 5 s.
 */
export function useRecentOrders(n = 100) {
  return useQuery<OrderRecord[]>({
    queryKey: ['exec', 'orders', n],
    queryFn:  () => fetchExec<OrderRecord[]>(
      `/api/execution/orders?n=${n}`,
      () => genOrders(n),
    ),
    refetchInterval: POLL_LIVE_MS,
    staleTime:       POLL_LIVE_MS / 2,
    retry: 1,
  });
}

// ---------------------------------------------------------------------------
// useTCAResults
// ---------------------------------------------------------------------------

/**
 * Fetch TCA results, optionally filtered by symbol and lookback days.
 */
export function useTCAResults(symbol?: string, days?: number) {
  const params = new URLSearchParams();
  if (symbol) params.set('symbol', symbol);
  if (days)   params.set('days', String(days));
  const qs = params.toString() ? `?${params.toString()}` : '';

  return useQuery<TCAResult[]>({
    queryKey: ['exec', 'tca', symbol ?? 'all', days ?? 7],
    queryFn:  () => fetchExec<TCAResult[]>(
      `/api/execution/tca${qs}`,
      () => genTCAResults(80),
    ),
    refetchInterval: POLL_LIVE_MS,
    staleTime:       POLL_LIVE_MS / 2,
    retry: 1,
  });
}

// ---------------------------------------------------------------------------
// useVenueScorecard
// ---------------------------------------------------------------------------

/**
 * Fetch venue scorecard. Polls every 60 s (changes slowly).
 */
export function useVenueScorecard() {
  return useQuery<VenueScore[]>({
    queryKey: ['exec', 'venues', 'scorecard'],
    queryFn:  () => fetchExec<VenueScore[]>(
      '/api/execution/venues/scorecard',
      genVenueScores,
    ),
    refetchInterval: POLL_SLOW_MS,
    staleTime:       POLL_SLOW_MS / 2,
    retry: 1,
  });
}

// ---------------------------------------------------------------------------
// useExecutionSummary
// ---------------------------------------------------------------------------

/**
 * Fetch the daily execution summary. Defaults to today's date.
 */
export function useExecutionSummary(date?: string) {
  const d = date ?? new Date().toISOString().slice(0, 10);
  return useQuery<ExecutionSummary>({
    queryKey: ['exec', 'summary', d],
    queryFn:  () => fetchExec<ExecutionSummary>(
      `/api/execution/summary?date=${d}`,
      () => genSummary(d),
    ),
    refetchInterval: POLL_LIVE_MS,
    staleTime:       POLL_LIVE_MS / 2,
    retry: 1,
  });
}

// ---------------------------------------------------------------------------
// useExecutionCostTrend
// ---------------------------------------------------------------------------

export function useExecutionCostTrend(days = 30) {
  return useQuery<ExecutionCostPoint[]>({
    queryKey: ['exec', 'cost_trend', days],
    queryFn:  () => fetchExec<ExecutionCostPoint[]>(
      `/api/execution/cost_trend?days=${days}`,
      () => genCostTrend(days),
    ),
    refetchInterval: POLL_SLOW_MS,
    staleTime:       POLL_SLOW_MS / 2,
    retry: 1,
  });
}

// ---------------------------------------------------------------------------
// useOrderStream -- WebSocket hook for real-time order updates
// ---------------------------------------------------------------------------

export interface OrderStreamState {
  orders:      OrderRecord[];
  isConnected: boolean;
  lastUpdate:  Date | null;
}

/**
 * Real-time order stream over WebSocket.
 * Maintains a ring buffer of the last `maxOrders` orders.
 * Reconnects with exponential backoff on disconnect.
 */
export function useOrderStream(maxOrders = 50): OrderStreamState {
  const [orders,      setOrders]      = useState<OrderRecord[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate,  setLastUpdate]  = useState<Date | null>(null);
  const wsRef = useRef<ManagedWebSocket | null>(null);

  const handleMessage = useCallback((raw: unknown) => {
    const evt = raw as OrderStreamEvent;
    setLastUpdate(new Date());

    setOrders(prev => {
      let next = [...prev];

      switch (evt.type) {
        case 'snapshot':
          return evt.orders.slice(-maxOrders);

        case 'order_new': {
          next.unshift(evt.order);
          break;
        }

        case 'order_update':
        case 'order_fill': {
          const idx = next.findIndex(o => o.id === evt.order.id);
          if (idx >= 0) {
            next[idx] = evt.order;
          } else {
            next.unshift(evt.order);
          }
          break;
        }

        case 'order_cancel': {
          next = next.map(o =>
            o.id === evt.orderId ? { ...o, status: 'cancelled' as const } : o
          );
          break;
        }

        default:
          break;
      }

      return next.slice(0, maxOrders);
    });
  }, [maxOrders]);

  useEffect(() => {
    const ws = new ManagedWebSocket(
      resolveWsUrl('/ws/orders'),
      {
        reconnectDelay:      1_000,
        maxReconnectDelay:   30_000,
        onMessage:           handleMessage,
        onConnect:           () => setIsConnected(true),
        onDisconnect:        () => setIsConnected(false),
        heartbeatIntervalMs: 20_000,
      },
    );
    wsRef.current = ws;

    // Seed with mock data while connecting (removed once real stream arrives)
    setOrders(genOrders(maxOrders));

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [handleMessage, maxOrders]);

  return { orders, isConnected, lastUpdate };
}

// ---------------------------------------------------------------------------
// useRefreshExec -- invalidate all exec queries
// ---------------------------------------------------------------------------

export function useRefreshExec() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: ['exec'] });
}
