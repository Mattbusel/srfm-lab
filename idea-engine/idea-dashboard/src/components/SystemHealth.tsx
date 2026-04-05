import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'

// ─── Types ────────────────────────────────────────────────────────────────────

type ServiceState = 'OK' | 'DEGRADED' | 'DOWN'

interface ServiceStatus {
  name: string
  state: ServiceState
  latencyMs?: number
  history: ServiceState[]   // last 24 hours, index 0 = oldest
}

interface Alert {
  id: string
  message: string
  severity: 'CRITICAL' | 'WARN' | 'INFO'
  ts: string
}

interface SystemHealthData {
  services: ServiceStatus[]
  healthScore: number       // 0–100
  alerts: Alert[]
}

// ─── Mock ─────────────────────────────────────────────────────────────────────

function buildHistory(base: ServiceState): ServiceState[] {
  const states: ServiceState[] = ['OK', 'DEGRADED', 'DOWN']
  return Array.from({ length: 24 }, (_, i) => {
    if (base === 'DOWN' && i > 20) return 'DOWN'
    if (base === 'DEGRADED' && (i === 12 || i === 16)) return 'DEGRADED'
    return Math.random() > 0.95 ? states[Math.floor(Math.random() * 2) + 1] : 'OK'
  })
}

const MOCK: SystemHealthData = {
  healthScore: 82,
  services: [
    { name: 'Idea Engine API',     state: 'OK',       latencyMs: 12,  history: buildHistory('OK') },
    { name: 'Coordination Service', state: 'OK',      latencyMs: 8,   history: buildHistory('OK') },
    { name: 'Stats Service',       state: 'OK',       latencyMs: 45,  history: buildHistory('OK') },
    { name: 'Backtester',          state: 'DEGRADED', latencyMs: 340, history: buildHistory('DEGRADED') },
    { name: 'Live Trader',         state: 'OK',       latencyMs: 15,  history: buildHistory('OK') },
    { name: 'WebSocket Feed',      state: 'OK',       latencyMs: 4,   history: buildHistory('OK') },
    { name: 'Data Pipeline',       state: 'OK',       latencyMs: 22,  history: buildHistory('OK') },
    { name: 'Postgres',            state: 'OK',       latencyMs: 3,   history: buildHistory('OK') },
    { name: 'Redis Cache',         state: 'DOWN',     latencyMs: undefined, history: buildHistory('DOWN') },
  ],
  alerts: [
    { id: 'a1', message: 'Backtester latency elevated (340ms avg)', severity: 'WARN',     ts: new Date(Date.now() - 600_000).toISOString() },
    { id: 'a2', message: 'Redis Cache connection lost',             severity: 'CRITICAL', ts: new Date(Date.now() - 120_000).toISOString() },
    { id: 'a3', message: 'Loop cycle completed in 342s (slow)',     severity: 'WARN',     ts: new Date(Date.now() - 900_000).toISOString() },
    { id: 'a4', message: 'New hypothesis applied: signal.threshold', severity: 'INFO',    ts: new Date(Date.now() - 1_800_000).toISOString() },
  ],
}

async function fetchSystemHealth(): Promise<SystemHealthData> {
  try {
    const res = await fetch('/api/system/health')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK
  }
}

// ─── Sub-components ───────────────────────────────────────────────────────────

const STATE_COLORS: Record<ServiceState, string> = {
  OK:       'var(--green)',
  DEGRADED: 'var(--yellow)',
  DOWN:     'var(--red)',
}

interface SparklineHistoryProps {
  history: ServiceState[]
}

const SparklineHistory: React.FC<SparklineHistoryProps> = ({ history }) => {
  const W = 96, H = 18
  const cellW = W / history.length
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: W, height: H }}>
      {history.map((s, i) => (
        <rect
          key={i}
          x={i * cellW + 0.5}
          y={0}
          width={cellW - 1}
          height={H}
          fill={STATE_COLORS[s]}
          opacity={s === 'OK' ? 0.4 : 0.85}
        />
      ))}
    </svg>
  )
}

// ─── Component ────────────────────────────────────────────────────────────────

const SystemHealth: React.FC = () => {
  const [expandedService, setExpandedService] = useState<string | null>(null)

  const { data, isLoading } = useQuery<SystemHealthData>({
    queryKey: ['system-health'],
    queryFn: fetchSystemHealth,
    refetchInterval: 15_000,
  })

  if (isLoading || !data) return (
    <div style={{ padding: '12px 16px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
      Loading system health…
    </div>
  )

  const scoreColor = data.healthScore >= 80 ? 'var(--green)' : data.healthScore >= 60 ? 'var(--yellow)' : 'var(--red)'
  const downCount = data.services.filter(s => s.state === 'DOWN').length
  const degradedCount = data.services.filter(s => s.state === 'DEGRADED').length

  const alertColors: Record<string, string> = {
    CRITICAL: 'var(--red)',
    WARN:     'var(--yellow)',
    INFO:     'var(--blue)',
  }

  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      {/* Header with health score gauge */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        gap: 16,
      }}>
        <div>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>SYSTEM HEALTH</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
            <span style={{ fontSize: '1.4rem', fontWeight: 800, color: scoreColor, fontVariantNumeric: 'tabular-nums' }}>
              {data.healthScore}
            </span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>/100</span>
          </div>
        </div>
        <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--bg-hover)', overflow: 'hidden' }}>
          <div style={{
            width: `${data.healthScore}%`, height: '100%', background: scoreColor,
            transition: 'width 0.4s',
          }} />
        </div>
        <div style={{ display: 'flex', gap: 12, fontSize: '0.72rem' }}>
          {downCount > 0 && (
            <span style={{ color: 'var(--red)', fontWeight: 700 }}>{downCount} DOWN</span>
          )}
          {degradedCount > 0 && (
            <span style={{ color: 'var(--yellow)', fontWeight: 700 }}>{degradedCount} DEGRADED</span>
          )}
          {downCount === 0 && degradedCount === 0 && (
            <span style={{ color: 'var(--green)', fontWeight: 700 }}>ALL OK</span>
          )}
        </div>
      </div>

      {/* Services grid */}
      <div style={{ padding: '10px 12px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 6 }}>
          {data.services.map(svc => {
            const color = STATE_COLORS[svc.state]
            const expanded = expandedService === svc.name
            return (
              <div
                key={svc.name}
                onClick={() => setExpandedService(expanded ? null : svc.name)}
                style={{
                  padding: '8px 10px',
                  borderRadius: 6,
                  background: 'var(--bg-hover)',
                  border: `1px solid ${expanded ? color : 'var(--border)'}`,
                  cursor: 'pointer',
                  transition: 'border-color 0.2s',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: color, flexShrink: 0,
                    boxShadow: svc.state !== 'OK' ? `0 0 5px ${color}` : undefined,
                  }} />
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-primary)', flex: 1 }}>{svc.name}</span>
                  {svc.latencyMs != null && (
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                      {svc.latencyMs}ms
                    </span>
                  )}
                </div>
                {expanded && (
                  <div style={{ marginTop: 6 }}>
                    <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: 3 }}>24h history</div>
                    <SparklineHistory history={svc.history} />
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Recent alerts */}
      <div style={{ borderTop: '1px solid var(--border)' }}>
        <div style={{ padding: '8px 16px', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>
          RECENT ALERTS
        </div>
        {data.alerts.slice(0, 4).map((a, i) => (
          <div key={a.id} style={{
            padding: '6px 16px',
            borderTop: i > 0 ? '1px solid var(--border)' : undefined,
            display: 'flex', alignItems: 'flex-start', gap: 8,
          }}>
            <span style={{
              fontSize: '0.62rem', fontWeight: 700, padding: '1px 5px', borderRadius: 4,
              background: `${alertColors[a.severity]}18`, color: alertColors[a.severity],
              flexShrink: 0, marginTop: 1,
            }}>
              {a.severity}
            </span>
            <span style={{ flex: 1, fontSize: '0.72rem', color: 'var(--text-secondary)' }}>{a.message}</span>
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', flexShrink: 0 }}>
              {new Date(a.ts).toLocaleTimeString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default SystemHealth
