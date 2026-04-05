import React, { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { formatDistanceToNow } from 'date-fns'
import LoadingSpinner from '../components/LoadingSpinner'
import { fetchCounterfactuals } from '../api/client'
import type { Counterfactual } from '../types'

// ─── Counterfactual Card ──────────────────────────────────────────────────────

interface CFCardProps {
  cf: Counterfactual
}

const CFCard: React.FC<CFCardProps> = ({ cf }) => {
  const improvColor = cf.improvement > 0 ? 'var(--green)' : 'var(--red)'
  const deltas = Object.entries(cf.paramDelta)

  return (
    <div
      className="card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        borderLeft: `3px solid ${improvColor}`,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
            {cf.description}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Baseline: <span className="num" style={{ color: 'var(--accent)' }}>{cf.baselineRunId}</span>
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div
            className="num"
            style={{ fontSize: '1.25rem', fontWeight: 700, color: improvColor }}
          >
            {cf.improvement >= 0 ? '+' : ''}
            {(cf.improvement * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>improvement</div>
        </div>
      </div>

      {/* Metric Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
        {[
          { label: 'Sharpe', value: cf.sharpe.toFixed(3), color: 'var(--green)' },
          { label: 'Max DD', value: `${(cf.maxDD * 100).toFixed(1)}%`, color: 'var(--red)' },
          { label: 'Calmar', value: cf.calmar.toFixed(2), color: 'var(--yellow)' },
        ].map(({ label, value, color }) => (
          <div
            key={label}
            style={{
              background: 'var(--bg-elevated)',
              borderRadius: 'var(--radius)',
              padding: '6px 8px',
            }}
          >
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
            <div className="num" style={{ color, fontWeight: 600, fontSize: '0.875rem' }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Param Deltas */}
      {deltas.length > 0 && (
        <div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Parameter Changes
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {deltas.map(([param, delta]) => (
              <span
                key={param}
                style={{
                  background: 'var(--bg-elevated)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '2px 8px',
                  fontSize: '0.75rem',
                  border: '1px solid var(--border)',
                }}
              >
                <span style={{ color: 'var(--text-muted)' }}>{param}: </span>
                <span
                  className="num"
                  style={{
                    color: (delta as number) > 0 ? 'var(--green)' : 'var(--red)',
                    fontWeight: 600,
                  }}
                >
                  {(delta as number) > 0 ? '+' : ''}
                  {typeof delta === 'number' ? delta.toFixed(3) : String(delta)}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
        {formatDistanceToNow(new Date(cf.createdAt), { addSuffix: true })}
      </div>
    </div>
  )
}

// ─── Improvement Chart ────────────────────────────────────────────────────────

interface ImprovChartProps {
  counterfactuals: Counterfactual[]
}

const ImprovementChart: React.FC<ImprovChartProps> = ({ counterfactuals }) => {
  const data = useMemo(
    () =>
      [...counterfactuals]
        .sort((a, b) => b.improvement - a.improvement)
        .map((cf) => ({
          id: cf.baselineRunId,
          improvement: parseFloat((cf.improvement * 100).toFixed(2)),
          sharpe: cf.sharpe,
        })),
    [counterfactuals]
  )

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
        <XAxis dataKey="id" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
        <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 10 }} tickFormatter={(v: number) => `${v}%`} />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-emphasis)',
            borderRadius: 6,
            fontSize: 12,
          }}
          formatter={(val: number) => [`${val.toFixed(2)}%`, 'Improvement']}
        />
        <Bar dataKey="improvement" radius={[3, 3, 0, 0]}>
          {data.map((entry, i) => (
            <Cell
              key={`cell-${i}`}
              fill={entry.improvement >= 0 ? '#22c55e' : '#ef4444'}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ─── Counterfactuals Page ─────────────────────────────────────────────────────

const CounterfactualsPage: React.FC = () => {
  const { data: cfs = [], isLoading, error } = useQuery({
    queryKey: ['counterfactuals'],
    queryFn: () => fetchCounterfactuals(),
    refetchInterval: 60_000,
    staleTime: 50_000,
  })

  const positive = cfs.filter((c) => c.improvement > 0)

  if (error) {
    return (
      <div className="empty-state">
        <div className="icon">⚠</div>
        <span>Failed to load counterfactuals</span>
      </div>
    )
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Counterfactual Analysis</div>
          <div className="page-subtitle">
            {cfs.length} experiments · {positive.length} positive improvements
          </div>
        </div>
      </div>

      {isLoading ? (
        <LoadingSpinner fullPage label="Loading counterfactuals…" />
      ) : (
        <>
          {cfs.length > 0 && (
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-header">
                <span className="card-title">Improvement vs Baseline</span>
              </div>
              <ImprovementChart counterfactuals={cfs} />
            </div>
          )}

          {cfs.length === 0 ? (
            <div className="empty-state">
              <div className="icon">∅</div>
              <span>No counterfactual experiments yet</span>
            </div>
          ) : (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
                gap: 14,
              }}
            >
              {[...cfs]
                .sort((a, b) => b.improvement - a.improvement)
                .map((cf) => (
                  <CFCard key={cf.id} cf={cf} />
                ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default CounterfactualsPage
