// ============================================================
// hooks/useAlertsAPI.ts -- Alerts data hooks with WebSocket + React Query
// WebSocket stream from /ws/alerts with polling fallback.
// ============================================================

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState, useCallback } from 'react'

// ---------------------------------------------------------------------------
// Types -- imported inline to avoid circular deps; real types live in iae.ts
// ---------------------------------------------------------------------------

export type AlertSeverity = 'critical' | 'warning' | 'info'
export type AlertStatus = 'active' | 'acknowledged' | 'resolved' | 'snoozed'

export interface Alert {
  id: string
  severity: AlertSeverity
  status: AlertStatus
  title: string
  message: string
  service: string
  timestamp: string
  acknowledged_at?: string
  resolved_at?: string
  snoozed_until?: string
  tags: string[]
  metadata: Record<string, unknown>
}

export type CircuitBreakerState = 'CLOSED' | 'OPEN' | 'HALF_OPEN'

export interface CircuitBreaker {
  name: string
  state: CircuitBreakerState
  failure_count: number
  success_count: number
  last_failure_at: string | null
  last_success_at: string | null
  cooldown_remaining_ms: number
  threshold: number
  timeout_ms: number
}

export type ServiceStatus = 'healthy' | 'degraded' | 'down' | 'unknown'

export interface ServiceHealth {
  name: string
  status: ServiceStatus
  last_check_at: string
  response_time_ms: number
  uptime_pct_24h: number
  endpoint: string
  error?: string
}

export interface AlertRule {
  id: string
  name: string
  condition: string
  severity: AlertSeverity
  service: string
  enabled: boolean
  cooldown_minutes: number
  created_at: string
  last_fired_at: string | null
  fire_count: number
}

export interface AlertTimelineBucket {
  hour: string           // ISO truncated to hour
  critical: number
  warning: number
  info: number
  total: number
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const ALERTS_API_BASE = 'http://localhost:8796'
const ALERTS_WS_URL = 'ws://localhost:8796/ws/alerts'
const POLL_FAST_MS = 5_000
const POLL_MEDIUM_MS = 15_000
const POLL_SLOW_MS = 60_000

// ---------------------------------------------------------------------------
// Mock data generators
// ---------------------------------------------------------------------------

function genId(): string {
  return Math.random().toString(36).slice(2, 10)
}

const MOCK_SERVICES = ['live_trader', 'market_data', 'iae_api', 'coordination', 'risk_engine']
const MOCK_TITLES: Record<AlertSeverity, string[]> = {
  critical: [
    'Position limit breach detected',
    'Circuit breaker OPEN on Alpaca',
    'Drawdown threshold exceeded',
    'WebSocket reconnect failure',
    'Order rejection rate > 10%',
  ],
  warning: [
    'Latency spike on Binance feed',
    'GARCH model diverging',
    'Correlation matrix stale',
    'IAE diversity index low',
    'Memory usage 85%',
  ],
  info: [
    'Daily reset completed',
    'New genome generation started',
    'Backtest job queued',
    'Risk limits recalculated',
    'Config reload successful',
  ],
}

function mockAlerts(n = 20): Alert[] {
  const severities: AlertSeverity[] = ['critical', 'warning', 'info']
  const statuses: AlertStatus[] = ['active', 'acknowledged', 'resolved', 'snoozed']
  return Array.from({ length: n }, (_, i) => {
    const sev = severities[i % 3]
    const titles = MOCK_TITLES[sev]
    const status = i < 3 ? 'active' : statuses[i % 4]
    return {
      id: genId(),
      severity: sev,
      status,
      title: titles[i % titles.length],
      message: `Detected at threshold breach for ${MOCK_SERVICES[i % MOCK_SERVICES.length]}. Automatic escalation pending.`,
      service: MOCK_SERVICES[i % MOCK_SERVICES.length],
      timestamp: new Date(Date.now() - i * 7 * 60_000).toISOString(),
      acknowledged_at: status === 'acknowledged' ? new Date(Date.now() - i * 5 * 60_000).toISOString() : undefined,
      resolved_at: status === 'resolved' ? new Date(Date.now() - i * 3 * 60_000).toISOString() : undefined,
      snoozed_until: status === 'snoozed' ? new Date(Date.now() + 15 * 60_000).toISOString() : undefined,
      tags: [sev, MOCK_SERVICES[i % MOCK_SERVICES.length]],
      metadata: { threshold: 0.05 + Math.random() * 0.1, observed: 0.12 + Math.random() * 0.05 },
    }
  })
}

function mockCircuitBreakers(): CircuitBreaker[] {
  return [
    {
      name: 'Alpaca',
      state: 'CLOSED',
      failure_count: 0,
      success_count: 842,
      last_failure_at: new Date(Date.now() - 3600_000 * 6).toISOString(),
      last_success_at: new Date(Date.now() - 2_000).toISOString(),
      cooldown_remaining_ms: 0,
      threshold: 5,
      timeout_ms: 30_000,
    },
    {
      name: 'Binance',
      state: 'HALF_OPEN',
      failure_count: 3,
      success_count: 1210,
      last_failure_at: new Date(Date.now() - 90_000).toISOString(),
      last_success_at: new Date(Date.now() - 120_000).toISOString(),
      cooldown_remaining_ms: 12_000,
      threshold: 5,
      timeout_ms: 30_000,
    },
    {
      name: 'Polygon',
      state: 'CLOSED',
      failure_count: 1,
      success_count: 2048,
      last_failure_at: new Date(Date.now() - 3600_000 * 2).toISOString(),
      last_success_at: new Date(Date.now() - 1_000).toISOString(),
      cooldown_remaining_ms: 0,
      threshold: 5,
      timeout_ms: 30_000,
    },
  ]
}

function mockServiceHealth(): ServiceHealth[] {
  return [
    { name: 'live_trader', status: 'healthy', last_check_at: new Date().toISOString(), response_time_ms: 14, uptime_pct_24h: 99.8, endpoint: 'http://localhost:8790/health' },
    { name: 'market_data', status: 'degraded', last_check_at: new Date().toISOString(), response_time_ms: 380, uptime_pct_24h: 97.2, endpoint: 'http://localhost:8792/health', error: 'Elevated latency' },
    { name: 'iae_api', status: 'healthy', last_check_at: new Date().toISOString(), response_time_ms: 22, uptime_pct_24h: 99.9, endpoint: 'http://localhost:8795/health' },
    { name: 'coordination', status: 'healthy', last_check_at: new Date().toISOString(), response_time_ms: 8, uptime_pct_24h: 100, endpoint: 'http://localhost:8797/health' },
    { name: 'risk_engine', status: 'healthy', last_check_at: new Date().toISOString(), response_time_ms: 18, uptime_pct_24h: 99.6, endpoint: 'http://localhost:8791/health' },
  ]
}

function mockAlertRules(): AlertRule[] {
  return [
    { id: genId(), name: 'Drawdown Alert', condition: 'portfolio.drawdown > 0.10', severity: 'critical', service: 'live_trader', enabled: true, cooldown_minutes: 15, created_at: '2025-01-01T00:00:00Z', last_fired_at: null, fire_count: 0 },
    { id: genId(), name: 'Position Limit', condition: 'position.size_usd > 50000', severity: 'warning', service: 'live_trader', enabled: true, cooldown_minutes: 5, created_at: '2025-01-01T00:00:00Z', last_fired_at: new Date(Date.now() - 3600_000).toISOString(), fire_count: 3 },
    { id: genId(), name: 'IAE Stagnation', condition: 'iae.stagnation_counter > 10', severity: 'warning', service: 'iae_api', enabled: true, cooldown_minutes: 60, created_at: '2025-01-15T00:00:00Z', last_fired_at: null, fire_count: 0 },
    { id: genId(), name: 'Latency Spike', condition: 'feed.latency_ms > 500', severity: 'warning', service: 'market_data', enabled: true, cooldown_minutes: 2, created_at: '2025-02-01T00:00:00Z', last_fired_at: new Date(Date.now() - 7200_000).toISOString(), fire_count: 12 },
    { id: genId(), name: 'Circuit Open', condition: 'broker.circuit_state == OPEN', severity: 'critical', service: 'live_trader', enabled: true, cooldown_minutes: 1, created_at: '2025-01-01T00:00:00Z', last_fired_at: new Date(Date.now() - 86400_000).toISOString(), fire_count: 1 },
    { id: genId(), name: 'Memory High', condition: 'system.memory_pct > 90', severity: 'warning', service: 'coordination', enabled: false, cooldown_minutes: 10, created_at: '2025-03-01T00:00:00Z', last_fired_at: null, fire_count: 0 },
  ]
}

function mockAlertTimeline(hours = 24): AlertTimelineBucket[] {
  const now = new Date()
  return Array.from({ length: hours }, (_, i) => {
    const h = new Date(now.getTime() - (hours - 1 - i) * 3600_000)
    h.setMinutes(0, 0, 0)
    const critical = Math.floor(Math.random() * 3)
    const warning = Math.floor(Math.random() * 6)
    const info = Math.floor(Math.random() * 8)
    return {
      hour: h.toISOString(),
      critical,
      warning,
      info,
      total: critical + warning + info,
    }
  })
}

// ---------------------------------------------------------------------------
// Fetch helper
// ---------------------------------------------------------------------------

async function fetchJson<T>(url: string, mock: () => T): Promise<T> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(5_000) })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return (await res.json()) as T
  } catch {
    return mock()
  }
}

// ---------------------------------------------------------------------------
// WebSocket hook for live alerts stream
// ---------------------------------------------------------------------------

export interface UseActiveAlertsReturn {
  alerts: Alert[]
  wsConnected: boolean
  isLoading: boolean
  error: Error | null
}

export function useActiveAlerts(): UseActiveAlertsReturn {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)

  // Polling fallback via React Query when WS is not connected
  const { data: polledAlerts, isLoading, error } = useQuery({
    queryKey: ['alerts', 'active'],
    queryFn: () => fetchJson<Alert[]>(`${ALERTS_API_BASE}/api/alerts/active`, () => mockAlerts(20)),
    refetchInterval: wsConnected ? false : POLL_FAST_MS,
    staleTime: POLL_FAST_MS / 2,
  })

  // Merge polled data into local state when WS is down
  useEffect(() => {
    if (!wsConnected && polledAlerts) {
      setAlerts(polledAlerts)
    }
  }, [polledAlerts, wsConnected])

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    try {
      const ws = new WebSocket(ALERTS_WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        if (mountedRef.current) setWsConnected(true)
      }

      ws.onmessage = (ev) => {
        if (!mountedRef.current) return
        try {
          const msg = JSON.parse(ev.data as string) as { type: string; payload: Alert | Alert[] }
          if (msg.type === 'alert_snapshot') {
            setAlerts(Array.isArray(msg.payload) ? msg.payload : [msg.payload])
          } else if (msg.type === 'alert_new') {
            setAlerts((prev) => [msg.payload as Alert, ...prev.slice(0, 99)])
          } else if (msg.type === 'alert_update') {
            const updated = msg.payload as Alert
            setAlerts((prev) => prev.map((a) => (a.id === updated.id ? updated : a)))
          }
        } catch {
          // ignore parse errors
        }
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        setWsConnected(false)
        wsRef.current = null
        reconnectTimer.current = setTimeout(connect, 5_000)
      }

      ws.onerror = () => {
        ws.close()
      }
    } catch {
      // WebSocket not available (e.g., dev mode) -- polling takes over
      setWsConnected(false)
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { alerts, wsConnected, isLoading: isLoading && !wsConnected, error: error as Error | null }
}

// ---------------------------------------------------------------------------
// Circuit breaker hook
// ---------------------------------------------------------------------------

export function useCircuitBreakerStatus() {
  return useQuery({
    queryKey: ['alerts', 'circuit-breakers'],
    queryFn: () =>
      fetchJson<CircuitBreaker[]>(
        `${ALERTS_API_BASE}/api/circuit-breakers`,
        mockCircuitBreakers,
      ),
    refetchInterval: POLL_FAST_MS,
    staleTime: POLL_FAST_MS / 2,
  })
}

// ---------------------------------------------------------------------------
// Service health hook
// ---------------------------------------------------------------------------

export function useServiceHealth() {
  return useQuery({
    queryKey: ['alerts', 'service-health'],
    queryFn: () =>
      fetchJson<ServiceHealth[]>(
        `${ALERTS_API_BASE}/api/services/health`,
        mockServiceHealth,
      ),
    refetchInterval: POLL_MEDIUM_MS,
    staleTime: POLL_MEDIUM_MS / 2,
  })
}

// ---------------------------------------------------------------------------
// Alert rules hook
// ---------------------------------------------------------------------------

export function useAlertRules() {
  return useQuery({
    queryKey: ['alerts', 'rules'],
    queryFn: () =>
      fetchJson<AlertRule[]>(`${ALERTS_API_BASE}/api/alerts/rules`, mockAlertRules),
    refetchInterval: POLL_SLOW_MS,
    staleTime: POLL_SLOW_MS / 2,
  })
}

// ---------------------------------------------------------------------------
// Alert timeline
// ---------------------------------------------------------------------------

export function useAlertTimeline(hours = 24) {
  return useQuery({
    queryKey: ['alerts', 'timeline', hours],
    queryFn: () =>
      fetchJson<AlertTimelineBucket[]>(
        `${ALERTS_API_BASE}/api/alerts/timeline?hours=${hours}`,
        () => mockAlertTimeline(hours),
      ),
    refetchInterval: POLL_MEDIUM_MS,
    staleTime: POLL_MEDIUM_MS / 2,
  })
}

// ---------------------------------------------------------------------------
// Alert action mutations
// ---------------------------------------------------------------------------

export function useAlertActions() {
  const qc = useQueryClient()

  const invalidate = () => qc.invalidateQueries({ queryKey: ['alerts'] })

  const acknowledge = useMutation({
    mutationFn: async (alert_id: string) => {
      const res = await fetch(`${ALERTS_API_BASE}/api/alerts/${alert_id}/acknowledge`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error('Acknowledge failed')
      return res.json()
    },
    onSuccess: invalidate,
  })

  const resolve = useMutation({
    mutationFn: async (alert_id: string) => {
      const res = await fetch(`${ALERTS_API_BASE}/api/alerts/${alert_id}/resolve`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error('Resolve failed')
      return res.json()
    },
    onSuccess: invalidate,
  })

  const snooze = useMutation({
    mutationFn: async ({ alert_id, minutes }: { alert_id: string; minutes: number }) => {
      const res = await fetch(`${ALERTS_API_BASE}/api/alerts/${alert_id}/snooze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ minutes }),
      })
      if (!res.ok) throw new Error('Snooze failed')
      return res.json()
    },
    onSuccess: invalidate,
  })

  const toggleRule = useMutation({
    mutationFn: async ({ rule_id, enabled }: { rule_id: string; enabled: boolean }) => {
      const res = await fetch(`${ALERTS_API_BASE}/api/alerts/rules/${rule_id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      })
      if (!res.ok) throw new Error('Toggle rule failed')
      return res.json()
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts', 'rules'] }),
  })

  return { acknowledge, resolve, snooze, toggleRule }
}
