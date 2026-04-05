import React, { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { formatDistanceToNow } from 'date-fns'
import MetricCard from '../components/MetricCard'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import {
  fetchGenomes,
  fetchAlerts,
  fetchEvolutionStats,
  fetchHypotheses,
} from '../api/client'
import { useIdeaStore } from '../store/ideaStore'
import { useAcknowledgeAlert } from '../hooks/useAlerts'
import type { Island } from '../types'

// ─── Evolution Chart ──────────────────────────────────────────────────────────

const ISLAND_COLORS: Record<Island, string> = {
  BULL: '#22c55e',
  BEAR: '#ef4444',
  NEUTRAL: '#3b82f6',
}

const EvolutionChart: React.FC = () => {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['evolution', 'stats'],
    queryFn: fetchEvolutionStats,
    refetchInterval: 30_000,
  })

  const chartData = useMemo(() => {
    if (!stats) return []
    const byGen: Record<number, Record<string, number>> = {}
    for (const s of stats) {
      if (!byGen[s.generation]) byGen[s.generation] = { generation: s.generation }
      byGen[s.generation][`${s.island}_best`] = s.bestFitness
      byGen[s.generation][`${s.island}_mean`] = s.meanFitness
    }
    return Object.values(byGen).sort((a, b) => a.generation - b.generation)
  }, [stats])

  if (isLoading) return <LoadingSpinner fullPage label="Loading evolution data…" />

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
        <XAxis dataKey="generation" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v: number) => v.toFixed(1)}
          tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-emphasis)',
            borderRadius: 6,
            fontSize: 12,
          }}
          formatter={(val: number, name: string) => [val.toFixed(4), name]}
        />
        <Legend
          wrapperStyle={{ fontSize: 11, color: 'var(--text-muted)' }}
        />
        {(['BULL', 'BEAR', 'NEUTRAL'] as Island[]).map((island) => (
          <React.Fragment key={island}>
            <Line
              type="monotone"
              dataKey={`${island}_best`}
              name={`${island} Best`}
              stroke={ISLAND_COLORS[island]}
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey={`${island}_mean`}
              name={`${island} Mean`}
              stroke={ISLAND_COLORS[island]}
              strokeWidth={1}
              strokeDasharray="4 2"
              dot={false}
              connectNulls
            />
          </React.Fragment>
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

// ─── Regime Indicator ─────────────────────────────────────────────────────────

const RegimeIndicator: React.FC = () => {
  const { currentRegime, regimeConfidence, previousRegime } = useIdeaStore()

  const regimeColor: Record<Island, string> = {
    BULL: 'var(--green)',
    BEAR: 'var(--red)',
    NEUTRAL: 'var(--blue)',
  }

  return (
    <div
      className="card"
      style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
    >
      <span className="card-title">Market Regime</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div
          style={{
            width: 12,
            height: 12,
            borderRadius: '50%',
            background: regimeColor[currentRegime],
            boxShadow: `0 0 8px ${regimeColor[currentRegime]}`,
            flexShrink: 0,
          }}
        />
        <span
          style={{
            fontSize: '1.5rem',
            fontWeight: 700,
            color: regimeColor[currentRegime],
            fontFamily: 'var(--font-mono)',
          }}
        >
          {currentRegime}
        </span>
      </div>
      <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
        Confidence:{' '}
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-secondary)',
          }}
        >
          {(regimeConfidence * 100).toFixed(0)}%
        </span>
      </div>
      {previousRegime && previousRegime !== currentRegime && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Previous:{' '}
          <StatusBadge value={previousRegime} size="sm" />
        </div>
      )}
      <div>
        <div
          style={{
            height: 4,
            borderRadius: 2,
            background: 'var(--bg-elevated)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${regimeConfidence * 100}%`,
              background: regimeColor[currentRegime],
              transition: 'width 0.5s ease',
            }}
          />
        </div>
      </div>
    </div>
  )
}

// ─── Alert Feed ───────────────────────────────────────────────────────────────

const AlertFeed: React.FC = () => {
  const { data: alerts = [], isLoading } = useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    refetchInterval: 15_000,
  })

  const { mutate: ack } = useAcknowledgeAlert()

  if (isLoading) return <LoadingSpinner fullPage label="Loading alerts…" />

  const unacked = alerts.filter((a) => !a.acknowledged)
  const recent = [...alerts].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  ).slice(0, 12)

  const severityIcon: Record<string, string> = {
    critical: '⬤',
    warning: '⬤',
    info: '⬤',
  }

  const severityColor: Record<string, string> = {
    critical: 'var(--red)',
    warning: 'var(--yellow)',
    info: 'var(--blue)',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {unacked.length === 0 && recent.length === 0 && (
        <div className="empty-state" style={{ padding: 24 }}>
          <div style={{ fontSize: '1.5rem', opacity: 0.3 }}>✓</div>
          <span>No alerts</span>
        </div>
      )}
      {recent.map((alert) => (
        <div
          key={alert.id}
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 8,
            padding: '8px 0',
            borderBottom: '1px solid var(--border-subtle)',
            opacity: alert.acknowledged ? 0.45 : 1,
            transition: 'opacity 0.2s',
          }}
        >
          <span
            style={{
              color: severityColor[alert.severity],
              fontSize: '0.5rem',
              marginTop: 4,
              flexShrink: 0,
            }}
          >
            {severityIcon[alert.severity]}
          </span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                fontSize: '0.8125rem',
                color: 'var(--text-secondary)',
                lineHeight: 1.4,
              }}
            >
              {alert.message}
            </div>
            <div
              style={{
                fontSize: '0.7rem',
                color: 'var(--text-muted)',
                marginTop: 2,
              }}
            >
              {formatDistanceToNow(new Date(alert.createdAt), {
                addSuffix: true,
              })}
            </div>
          </div>
          {!alert.acknowledged && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => ack(alert.id)}
              style={{ flexShrink: 0, fontSize: '0.7rem', padding: '2px 6px' }}
            >
              Ack
            </button>
          )}
        </div>
      ))}
    </div>
  )
}

// ─── Top Genomes Table ────────────────────────────────────────────────────────

const TopGenomesTable: React.FC = () => {
  const { data: genomes = [], isLoading } = useQuery({
    queryKey: ['genomes', 'all'],
    queryFn: () => fetchGenomes(),
    refetchInterval: 30_000,
  })

  const top5 = useMemo(
    () =>
      [...genomes]
        .sort((a, b) => b.sharpe - a.sharpe)
        .slice(0, 5),
    [genomes]
  )

  if (isLoading) return <LoadingSpinner fullPage label="Loading genomes…" />

  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Island</th>
          <th>Gen</th>
          <th>Sharpe</th>
          <th>MaxDD</th>
          <th>Calmar</th>
          <th>HOF</th>
        </tr>
      </thead>
      <tbody>
        {top5.map((g) => (
          <tr key={g.id}>
            <td>
              <span className="num" style={{ color: 'var(--accent)' }}>
                #{g.id}
              </span>
            </td>
            <td>
              <StatusBadge value={g.island} size="sm" />
            </td>
            <td className="num" style={{ color: 'var(--text-muted)' }}>
              {g.generation}
            </td>
            <td>
              <span
                className="num"
                style={{
                  color:
                    g.sharpe >= 2
                      ? 'var(--green)'
                      : g.sharpe >= 1
                      ? 'var(--accent)'
                      : 'var(--text-secondary)',
                  fontWeight: 600,
                }}
              >
                {g.sharpe.toFixed(3)}
              </span>
            </td>
            <td>
              <span className="num" style={{ color: 'var(--red)' }}>
                {(g.maxDD * 100).toFixed(1)}%
              </span>
            </td>
            <td>
              <span className="num" style={{ color: 'var(--yellow)' }}>
                {g.calmar.toFixed(2)}
              </span>
            </td>
            <td>
              {g.isHallOfFame && (
                <StatusBadge value="hof" size="sm" label="HOF" />
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ─── Dashboard Page ───────────────────────────────────────────────────────────

const DashboardPage: React.FC = () => {
  const { data: genomes = [] } = useQuery({
    queryKey: ['genomes', 'all'],
    queryFn: () => fetchGenomes(),
    refetchInterval: 30_000,
  })
  const { data: hypotheses = [] } = useQuery({
    queryKey: ['hypotheses', 'all'],
    queryFn: () => fetchHypotheses(),
    refetchInterval: 30_000,
  })
  const { data: alerts = [] } = useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    refetchInterval: 15_000,
  })
  const { data: stats = [] } = useQuery({
    queryKey: ['evolution', 'stats'],
    queryFn: fetchEvolutionStats,
    refetchInterval: 30_000,
  })

  const bestSharpe = useMemo(
    () =>
      genomes.length > 0
        ? Math.max(...genomes.map((g) => g.sharpe)).toFixed(3)
        : '—',
    [genomes]
  )
  const pendingCount = useMemo(
    () => hypotheses.filter((h) => h.status === 'pending').length,
    [hypotheses]
  )
  const unackedAlerts = useMemo(
    () => alerts.filter((a) => !a.acknowledged).length,
    [alerts]
  )
  const maxGen = useMemo(
    () => (stats.length > 0 ? Math.max(...stats.map((s) => s.generation)) : 0),
    [stats]
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">IAE Dashboard</div>
          <div className="page-subtitle">
            Idea Automation Engine — Real-Time Research Overview
          </div>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid-4">
        <MetricCard
          label="Total Genomes"
          value={genomes.length}
          subValue={`${genomes.filter((g) => g.isHallOfFame).length} Hall of Fame`}
          color="accent"
          icon="⬡"
        />
        <MetricCard
          label="Best Sharpe"
          value={bestSharpe}
          subValue="All islands"
          color="green"
          icon="▲"
          trend="up"
          trendValue="vs prev gen"
        />
        <MetricCard
          label="Hypotheses Pending"
          value={pendingCount}
          subValue={`${hypotheses.length} total`}
          color={pendingCount > 5 ? 'yellow' : 'default'}
          icon="⧖"
        />
        <MetricCard
          label="Unacked Alerts"
          value={unackedAlerts}
          subValue={`Gen ${maxGen} active`}
          color={unackedAlerts > 0 ? 'red' : 'default'}
          icon="⚑"
        />
      </div>

      {/* Main Content: Chart + Regime */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 200px', gap: 16 }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Evolution Progress — All Islands</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              30s refresh
            </span>
          </div>
          <EvolutionChart />
        </div>
        <RegimeIndicator />
      </div>

      {/* Bottom Row: Alerts + Top Genomes */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              Alert Feed
              {unackedAlerts > 0 && (
                <span
                  style={{
                    marginLeft: 8,
                    background: 'var(--red)',
                    color: '#fff',
                    borderRadius: '50%',
                    width: 18,
                    height: 18,
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                  }}
                >
                  {unackedAlerts}
                </span>
              )}
            </span>
          </div>
          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
            <AlertFeed />
          </div>
        </div>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Top Genomes — By Sharpe</span>
          </div>
          <TopGenomesTable />
        </div>
      </div>
    </div>
  )
}

export default DashboardPage
