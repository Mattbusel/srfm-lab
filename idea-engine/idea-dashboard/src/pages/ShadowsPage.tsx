import React, { useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { formatDistanceToNow } from 'date-fns'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import { fetchShadows, promoteGenome } from '../api/client'
import type { Shadow } from '../types'

const SHADOW_POLL_INTERVAL = 60_000

// ─── Alpha Chart ──────────────────────────────────────────────────────────────

interface AlphaChartProps {
  shadow: Shadow
}

const AlphaChart: React.FC<AlphaChartProps> = ({ shadow }) => {
  // Build synthetic equity curves if not provided
  const data = useMemo(() => {
    const days = 7
    const points = []
    let shadowEq = 10000
    let liveEq = 10000
    for (let d = 0; d <= days; d++) {
      const dailyShadow = shadow.return7d / days + (Math.random() - 0.5) * 0.002
      const dailyLive = shadow.returnLive7d / days + (Math.random() - 0.5) * 0.002
      shadowEq *= 1 + dailyShadow
      liveEq *= 1 + dailyLive
      points.push({
        day: `D${d}`,
        shadow: parseFloat(shadowEq.toFixed(2)),
        live: parseFloat(liveEq.toFixed(2)),
      })
    }
    return points
  }, [shadow.shadowId, shadow.return7d, shadow.returnLive7d]) // eslint-disable-line react-hooks/exhaustive-deps

  const isPositiveAlpha = shadow.alpha > 0

  return (
    <ResponsiveContainer width="100%" height={120}>
      <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
        <defs>
          <linearGradient id={`sg_${shadow.shadowId}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={isPositiveAlpha ? '#22c55e' : '#ef4444'} stopOpacity={0.3} />
            <stop offset="95%" stopColor={isPositiveAlpha ? '#22c55e' : '#ef4444'} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
        <XAxis dataKey="day" tick={{ fill: 'var(--text-muted)', fontSize: 9 }} />
        <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 9 }} tickFormatter={(v: number) => `$${Math.round(v)}`} />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-emphasis)',
            borderRadius: 6,
            fontSize: 11,
          }}
          formatter={(val: number) => [`$${val.toFixed(0)}`]}
        />
        <Area
          type="monotone"
          dataKey="shadow"
          name="Shadow"
          stroke={isPositiveAlpha ? '#22c55e' : '#ef4444'}
          fill={`url(#sg_${shadow.shadowId})`}
          strokeWidth={2}
        />
        <Area
          type="monotone"
          dataKey="live"
          name="Live"
          stroke="var(--text-muted)"
          fill="none"
          strokeDasharray="4 2"
          strokeWidth={1.5}
        />
        <Legend wrapperStyle={{ fontSize: 10 }} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ─── Shadow Row ───────────────────────────────────────────────────────────────

interface ShadowRowProps {
  shadow: Shadow
  expanded: boolean
  onExpand: () => void
  onPromote: (genomeId: number) => void
  promoting: boolean
}

const ShadowRow: React.FC<ShadowRowProps> = ({
  shadow,
  expanded,
  onExpand,
  onPromote,
  promoting,
}) => {
  const alphaColor = shadow.alpha > 0 ? 'var(--green)' : 'var(--red)'
  const canPromote = shadow.alpha > 0 && shadow.alphaDays >= 7 && !shadow.promoted

  return (
    <>
      <tr onClick={onExpand} style={{ cursor: 'pointer' }}>
        <td>
          <span className="num" style={{ color: 'var(--accent)' }}>
            #{shadow.shadowId}
          </span>
        </td>
        <td>
          <span className="num" style={{ color: 'var(--text-secondary)' }}>
            #{shadow.genomeId}
          </span>
        </td>
        <td>
          <span
            className="num"
            style={{
              color: shadow.return7d >= 0 ? 'var(--green)' : 'var(--red)',
              fontWeight: 600,
            }}
          >
            {shadow.return7d >= 0 ? '+' : ''}
            {(shadow.return7d * 100).toFixed(2)}%
          </span>
        </td>
        <td>
          <span
            className="num"
            style={{
              color: shadow.returnLive7d >= 0 ? 'var(--green)' : 'var(--red)',
            }}
          >
            {shadow.returnLive7d >= 0 ? '+' : ''}
            {(shadow.returnLive7d * 100).toFixed(2)}%
          </span>
        </td>
        <td>
          <span className="num" style={{ color: alphaColor, fontWeight: 700 }}>
            {shadow.alpha >= 0 ? '+' : ''}
            {(shadow.alpha * 100).toFixed(2)}%
          </span>
        </td>
        <td>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {shadow.alphaDays}d positive
          </span>
        </td>
        <td>
          {shadow.promoted ? (
            <StatusBadge value="promoted" size="sm" label="Promoted" />
          ) : shadow.alpha > 0 ? (
            <StatusBadge value="shadow" size="sm" label="Active" />
          ) : (
            <StatusBadge value="rejected" size="sm" label="Weak" />
          )}
        </td>
        <td>
          <div style={{ display: 'flex', gap: 6 }} onClick={(e) => e.stopPropagation()}>
            {canPromote && (
              <button
                className="btn btn-primary btn-sm"
                onClick={() => onPromote(shadow.genomeId)}
                disabled={promoting}
              >
                {promoting ? <LoadingSpinner size={12} /> : null}
                Promote
              </button>
            )}
            <button className="btn btn-ghost btn-sm" onClick={onExpand}>
              {expanded ? '▲' : '▼'}
            </button>
          </div>
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={8} style={{ padding: '12px 16px', background: 'var(--bg-elevated)' }}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr 200px',
                gap: 16,
                alignItems: 'start',
              }}
            >
              <AlphaChart shadow={shadow} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 2 }}>
                    Started
                  </div>
                  <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                    {formatDistanceToNow(new Date(shadow.startedAt), { addSuffix: true })}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 2 }}>
                    Alpha Days
                  </div>
                  <div
                    className="num"
                    style={{
                      fontSize: '1.25rem',
                      fontWeight: 700,
                      color: shadow.alphaDays >= 7 ? 'var(--green)' : 'var(--yellow)',
                    }}
                  >
                    {shadow.alphaDays}
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: 4 }}>
                      / 7 needed
                    </span>
                  </div>
                  <div
                    style={{
                      height: 4,
                      background: 'var(--bg-hover)',
                      borderRadius: 2,
                      marginTop: 4,
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        height: '100%',
                        width: `${Math.min(100, (shadow.alphaDays / 7) * 100)}%`,
                        background: shadow.alphaDays >= 7 ? 'var(--green)' : 'var(--yellow)',
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

// ─── Shadows Page ─────────────────────────────────────────────────────────────

const ShadowsPage: React.FC = () => {
  const [expandedId, setExpandedId] = React.useState<number | null>(null)

  const { data: shadows = [], isLoading, error } = useQuery({
    queryKey: ['shadows'],
    queryFn: fetchShadows,
    refetchInterval: SHADOW_POLL_INTERVAL,
    staleTime: 50_000,
  })

  const qc = useQueryClient()
  const promoteMutation = useMutation({
    mutationFn: (genomeId: number) => promoteGenome(genomeId),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['shadows'] }),
  })

  const stats = useMemo(() => ({
    total: shadows.length,
    positiveAlpha: shadows.filter((s) => s.alpha > 0).length,
    promoted: shadows.filter((s) => s.promoted).length,
    avgAlpha: shadows.length > 0
      ? shadows.reduce((acc, s) => acc + s.alpha, 0) / shadows.length
      : 0,
  }), [shadows])

  if (error) {
    return (
      <div className="empty-state">
        <div className="icon">⚠</div>
        <span>Failed to load shadow runners</span>
      </div>
    )
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Shadow Runners</div>
          <div className="page-subtitle">
            {stats.total} active · {stats.positiveAlpha} positive alpha · {stats.promoted} promoted · 60s refresh
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid-4" style={{ marginBottom: 20 }}>
        {[
          { label: 'Active Shadows', value: stats.total, color: 'var(--accent)' },
          { label: 'Positive Alpha', value: stats.positiveAlpha, color: 'var(--green)' },
          { label: 'Promoted', value: stats.promoted, color: 'var(--gold)' },
          {
            label: 'Avg Alpha (7d)',
            value: `${stats.avgAlpha >= 0 ? '+' : ''}${(stats.avgAlpha * 100).toFixed(2)}%`,
            color: stats.avgAlpha >= 0 ? 'var(--green)' : 'var(--red)',
          },
        ].map(({ label, value, color }) => (
          <div key={label} className="card">
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
              {label}
            </div>
            <div className="num" style={{ fontSize: '1.5rem', fontWeight: 700, color }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {isLoading ? (
        <LoadingSpinner fullPage label="Loading shadow runners…" />
      ) : shadows.length === 0 ? (
        <div className="empty-state">
          <div className="icon">◎</div>
          <span>No shadow runners active</span>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <table>
            <thead>
              <tr>
                <th>Shadow</th>
                <th>Genome</th>
                <th>7d Return</th>
                <th>Live 7d</th>
                <th>Alpha</th>
                <th>Alpha Days</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {shadows.map((shadow) => (
                <ShadowRow
                  key={shadow.shadowId}
                  shadow={shadow}
                  expanded={expandedId === shadow.shadowId}
                  onExpand={() =>
                    setExpandedId((prev) =>
                      prev === shadow.shadowId ? null : shadow.shadowId
                    )
                  }
                  onPromote={promoteMutation.mutate}
                  promoting={promoteMutation.isPending && promoteMutation.variables === shadow.genomeId}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default ShadowsPage
