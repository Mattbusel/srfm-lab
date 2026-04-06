// ============================================================
// useLiveMetrics.ts — WebSocket hook for observability dashboard
// Real-time metrics feed from dashboard_data.py on ws://localhost:8798
// ============================================================

import { useCallback, useEffect, useRef, useState } from 'react'
import type {
  DashboardMetricsPayload,
  DashboardWsMessage,
  BHSignal,
  CircuitBreakerStatus,
  EquityPoint,
  RiskMetrics,
  PortfolioMetrics,
  PositionSizing,
  TradeHeatmapCell,
  PnlSlice,
  CorrelationState,
  FactorAttribution,
  StressTestResult,
  DrawdownPoint,
  GreeksSummary,
} from '@/types/metrics'

// ---- Constants -------------------------------------------------------

const DASHBOARD_WS_URL = 'ws://localhost:8798/ws/dashboard'
const INITIAL_RECONNECT_DELAY_MS = 500
const MAX_RECONNECT_DELAY_MS = 30_000
const RECONNECT_MULTIPLIER = 1.8
const PING_INTERVAL_MS = 20_000
const MAX_EQUITY_HISTORY = 2000
const MAX_DRAWDOWN_HISTORY = 2000

// ---- Types -----------------------------------------------------------

export type LiveMetricsStatus = 'connecting' | 'open' | 'closed' | 'error'

export interface LiveMetricsState {
  status: LiveMetricsStatus
  lastMessageAt: Date | null
  portfolio: PortfolioMetrics | null
  equityCurve: EquityPoint[]
  drawdown: DrawdownPoint[]
  positionSizing: PositionSizing[]
  bhSignals: BHSignal[]
  circuitBreaker: CircuitBreakerStatus | null
  tradeHeatmap: TradeHeatmapCell[]
  pnlBySymbol: PnlSlice[]
  pnlByAssetClass: PnlSlice[]
  greeks: GreeksSummary | null
  correlation: CorrelationState | null
  factorAttribution: FactorAttribution[]
  stressTests: StressTestResult[]
  riskMetrics: RiskMetrics | null
  errorMessage: string | null
}

export interface UseLiveMetricsOptions {
  url?: string
  enabled?: boolean
  onConnected?: () => void
  onDisconnected?: () => void
  onError?: (msg: string) => void
  onFullUpdate?: (payload: DashboardMetricsPayload) => void
}

export interface UseLiveMetricsResult extends LiveMetricsState {
  reconnect: () => void
  send: (msg: unknown) => void
  isConnected: boolean
  reconnectCount: number
}

// ---- Initial state ---------------------------------------------------

const INITIAL_STATE: LiveMetricsState = {
  status: 'closed',
  lastMessageAt: null,
  portfolio: null,
  equityCurve: [],
  drawdown: [],
  positionSizing: [],
  bhSignals: [],
  circuitBreaker: null,
  tradeHeatmap: [],
  pnlBySymbol: [],
  pnlByAssetClass: [],
  greeks: null,
  correlation: null,
  factorAttribution: [],
  stressTests: [],
  riskMetrics: null,
  errorMessage: null,
}

// ---- Hook ------------------------------------------------------------

export function useLiveMetrics(options: UseLiveMetricsOptions = {}): UseLiveMetricsResult {
  const {
    url = DASHBOARD_WS_URL,
    enabled = true,
    onConnected,
    onDisconnected,
    onError,
    onFullUpdate,
  } = options

  const [state, setState] = useState<LiveMetricsState>(INITIAL_STATE)
  const [reconnectCount, setReconnectCount] = useState(0)

  const wsRef = useRef<WebSocket | null>(null)
  const mountedRef = useRef(true)
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY_MS)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Stable callbacks via refs
  const onConnectedRef = useRef(onConnected)
  const onDisconnectedRef = useRef(onDisconnected)
  const onErrorRef = useRef(onError)
  const onFullUpdateRef = useRef(onFullUpdate)

  onConnectedRef.current = onConnected
  onDisconnectedRef.current = onDisconnected
  onErrorRef.current = onError
  onFullUpdateRef.current = onFullUpdate

  // ------------------------------------------------------------------
  // Message processing
  // ------------------------------------------------------------------

  const processMessage = useCallback((msg: DashboardWsMessage) => {
    if (!mountedRef.current) return

    setState(prev => {
      switch (msg.type) {
        case 'dashboard_update': {
          const p = msg.payload
          onFullUpdateRef.current?.(p)

          // Merge equity curve (deduplicate and limit history)
          const existingTs = new Set(prev.equityCurve.map(e => e.timestamp))
          const newPoints = p.equityCurve.filter(e => !existingTs.has(e.timestamp))
          const mergedEquity = [...prev.equityCurve, ...newPoints]
            .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
            .slice(-MAX_EQUITY_HISTORY)

          const existingDdTs = new Set(prev.drawdown.map(d => d.timestamp))
          const newDd = p.drawdown.filter(d => !existingDdTs.has(d.timestamp))
          const mergedDd = [...prev.drawdown, ...newDd]
            .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
            .slice(-MAX_DRAWDOWN_HISTORY)

          return {
            ...prev,
            portfolio: p.portfolio,
            equityCurve: mergedEquity,
            drawdown: mergedDd,
            positionSizing: p.positionSizing,
            bhSignals: p.bhSignals,
            circuitBreaker: p.circuitBreaker,
            tradeHeatmap: p.tradeHeatmap,
            pnlBySymbol: p.pnlBySymbol,
            pnlByAssetClass: p.pnlByAssetClass,
            greeks: p.greeks,
            correlation: p.correlation,
            factorAttribution: p.factorAttribution,
            stressTests: p.stressTests,
            riskMetrics: p.riskMetrics,
            lastMessageAt: new Date(),
          }
        }

        case 'equity_update': {
          const { point } = msg.payload
          const merged = [...prev.equityCurve, point].slice(-MAX_EQUITY_HISTORY)
          return { ...prev, equityCurve: merged, lastMessageAt: new Date() }
        }

        case 'signal_update': {
          // Merge updated signals by (symbol, timeframe) key
          const updated = new Map(
            msg.payload.signals.map(s => [`${s.symbol}:${s.timeframe}`, s])
          )
          const mergedSignals = prev.bhSignals.map(s => {
            const key = `${s.symbol}:${s.timeframe}`
            return updated.get(key) ?? s
          })
          // Add any brand new signals
          for (const [key, sig] of updated) {
            if (!prev.bhSignals.find(s => `${s.symbol}:${s.timeframe}` === key)) {
              mergedSignals.push(sig)
            }
          }
          return { ...prev, bhSignals: mergedSignals, lastMessageAt: new Date() }
        }

        case 'circuit_breaker': {
          return { ...prev, circuitBreaker: msg.payload, lastMessageAt: new Date() }
        }

        case 'risk_update': {
          return { ...prev, riskMetrics: msg.payload, lastMessageAt: new Date() }
        }

        case 'ping':
        case 'pong':
          return prev

        default:
          return prev
      }
    })
  }, [])

  // ------------------------------------------------------------------
  // Ping / keepalive
  // ------------------------------------------------------------------

  const clearPing = useCallback(() => {
    if (pingTimerRef.current != null) {
      clearInterval(pingTimerRef.current)
      pingTimerRef.current = null
    }
  }, [])

  const startPing = useCallback(() => {
    clearPing()
    pingTimerRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }))
      }
    }, PING_INTERVAL_MS)
  }, [clearPing])

  // ------------------------------------------------------------------
  // Reconnect timer
  // ------------------------------------------------------------------

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current != null) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
  }, [])

  // ------------------------------------------------------------------
  // Connect
  // ------------------------------------------------------------------

  const connect = useCallback(() => {
    if (!mountedRef.current || !enabled) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setState(prev => ({ ...prev, status: 'connecting', errorMessage: null }))

    let ws: WebSocket
    try {
      ws = new WebSocket(url)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setState(prev => ({ ...prev, status: 'error', errorMessage: msg }))
      onErrorRef.current?.(msg)
      return
    }

    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) return
      reconnectDelayRef.current = INITIAL_RECONNECT_DELAY_MS
      setState(prev => ({ ...prev, status: 'open', errorMessage: null }))
      setReconnectCount(0)
      onConnectedRef.current?.()
      startPing()
    }

    ws.onmessage = (event: MessageEvent<string>) => {
      if (!mountedRef.current) return
      try {
        const msg = JSON.parse(event.data) as DashboardWsMessage
        processMessage(msg)
      } catch {
        // Ignore malformed frames
      }
    }

    ws.onclose = () => {
      if (!mountedRef.current) return
      clearPing()
      setState(prev => ({ ...prev, status: 'closed' }))
      onDisconnectedRef.current?.()

      // Exponential backoff reconnect
      clearReconnectTimer()
      reconnectTimerRef.current = setTimeout(() => {
        if (mountedRef.current && enabled) {
          reconnectDelayRef.current = Math.min(
            reconnectDelayRef.current * RECONNECT_MULTIPLIER,
            MAX_RECONNECT_DELAY_MS
          )
          setReconnectCount(c => c + 1)
          connect()
        }
      }, reconnectDelayRef.current)
    }

    ws.onerror = () => {
      if (!mountedRef.current) return
      const msg = 'WebSocket error'
      setState(prev => ({ ...prev, status: 'error', errorMessage: msg }))
      onErrorRef.current?.(msg)
    }
  }, [url, enabled, processMessage, startPing, clearPing, clearReconnectTimer])

  // ------------------------------------------------------------------
  // Disconnect
  // ------------------------------------------------------------------

  const disconnect = useCallback(() => {
    clearPing()
    clearReconnectTimer()
    if (wsRef.current) {
      wsRef.current.onclose = null  // Prevent reconnect loop on manual disconnect
      wsRef.current.close()
      wsRef.current = null
    }
  }, [clearPing, clearReconnectTimer])

  // ------------------------------------------------------------------
  // Manual reconnect
  // ------------------------------------------------------------------

  const reconnect = useCallback(() => {
    disconnect()
    reconnectDelayRef.current = INITIAL_RECONNECT_DELAY_MS
    connect()
  }, [disconnect, connect])

  // ------------------------------------------------------------------
  // Send
  // ------------------------------------------------------------------

  const send = useCallback((msg: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg))
    }
  }, [])

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  useEffect(() => {
    mountedRef.current = true
    if (enabled) connect()

    return () => {
      mountedRef.current = false
      disconnect()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, url])

  return {
    ...state,
    reconnect,
    send,
    isConnected: state.status === 'open',
    reconnectCount,
  }
}

// ---- Selector hooks (memoisation helpers) ----------------------------

/**
 * Subscribe only to BH signals for a specific symbol.
 */
export function useBHSignalsForSymbol(
  symbol: string,
  metrics: Pick<UseLiveMetricsResult, 'bhSignals'>
): BHSignal[] {
  return metrics.bhSignals.filter(s => s.symbol === symbol)
}

/**
 * Derive circuit breaker color for UI.
 */
export function circuitBreakerColor(state: CircuitBreakerStatus | null): string {
  if (!state) return '#9ca3af'  // gray
  switch (state.state) {
    case 'OPEN':      return '#ef4444'  // red
    case 'HALF_OPEN': return '#f59e0b'  // amber
    case 'CLOSED':    return '#22c55e'  // green
  }
}

/**
 * Format a signal strength value to display label.
 */
export function formatSignalStrength(strength: BHSignal['strength']): string {
  switch (strength) {
    case 'strong_long':  return '▲▲'
    case 'weak_long':    return '▲'
    case 'neutral':      return '–'
    case 'weak_short':   return '▼'
    case 'strong_short': return '▼▼'
  }
}

/**
 * Map signal strength to a Tailwind color class.
 */
export function signalColor(strength: BHSignal['strength']): string {
  switch (strength) {
    case 'strong_long':  return '#16a34a'   // green-600
    case 'weak_long':    return '#86efac'   // green-300
    case 'neutral':      return '#6b7280'   // gray-500
    case 'weak_short':   return '#fca5a5'   // red-300
    case 'strong_short': return '#dc2626'   // red-600
  }
}
