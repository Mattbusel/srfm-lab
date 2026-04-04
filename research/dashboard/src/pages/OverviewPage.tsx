import React, { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  TrendingUp, TrendingDown, Activity, DollarSign, Shield, Target, Zap, BarChart2,
  RefreshCw, AlertCircle, CheckCircle,
} from 'lucide-react'
import { MetricCard } from '@/components/ui/MetricCard'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { ErrorBoundary, ErrorDisplay } from '@/components/ui/ErrorBoundary'
import { EquityCurveChart } from '@/components/charts/EquityCurveChart'
import { DrawdownChart } from '@/components/charts/DrawdownChart'
import { RegimeTimelineChart, RegimeLegend } from '@/components/charts/RegimeTimelineChart'
import { TradeTable } from '@/components/tables/TradeTable'
import { PerformanceTable } from '@/components/tables/PerformanceTable'
import { fetchEquityCurve, fetchPerformanceMetrics, fetchTopTrades, fetchTrades } from '@/api/trades'
import { fetchRegimeSegments } from '@/api/regimes'
import { fetchSignalSnapshots } from '@/api/signals'
import { signalHeatmapColor, REGIME_COLORS, REGIME_LABELS, CHART_COLORS } from '@/utils/colors'
import { formatCurrency, formatPct, formatRatio, formatDate, pnlColor } from '@/utils/formatters'
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts'
import type { RegimeType } from '@/types/trades'
import { clsx } from 'clsx'

// ── Signal heatmap (27 instruments) ────────────────────────────────────────���─

function SignalHeatmap() {
  const { data: signals, isLoading, dataUpdatedAt } = useQuery({
    queryKey: ['signals-snapshot'],
    queryFn: fetchSignalSnapshots,
    refetchInterval: 30_000,
  })

  if (isLoading) return <LoadingSpinner size="sm" fullHeight />
  if (!signals?.length) return <div className="text-sm text-research-subtle">No signal data</div>

  const longCount = signals.filter(s => s.direction === 'long').length
  const shortCount = signals.filter(s => s.direction === 'short').length
  const neutralCount = signals.filter(s => s.direction === 'neutral').length

  return (
    <div className="space-y-3">
      {/* Summary bar */}
      <div className="flex items-center gap-4 text-xs font-mono">
        <span className="text-research-bull">▲ Long: {longCount}</span>
        <span className="text-research-bear">▼ Short: {shortCount}</span>
        <span className="text-research-subtle">– Neutral: {neutralCount}</span>
        <span className="ml-auto text-research-subtle/50">
          {dataUpdatedAt ? `@ ${new Date(dataUpdatedAt).toLocaleTimeString()}` : ''}
        </span>
      </div>

      {/* Grid */}
      <div className="grid gap-1" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(88px, 1fr))' }}>
        {signals.map(s => {
          const color = signalHeatmapColor(s.zscore)
          return (
            <div
              key={s.instrument}
              className="rounded p-2 text-center cursor-default transition-all hover:scale-105"
              style={{
                backgroundColor: `${color}33`,
                border: `1px solid ${color}55`,
              }}
              title={`${s.instrument}\nZ: ${s.zscore.toFixed(3)}\nComposite: ${s.compositeStrength.toFixed(3)}\nIC: ${s.ic.toFixed(4)}\nMom1d: ${(s.momentum1d * 100).toFixed(2)}%`}
            >
              <div className="text-[10px] font-mono font-semibold text-research-text leading-tight truncate">
                {s.instrument.replace('-USD', '')}
              </div>
              <div
                className="text-sm font-mono font-bold mt-0.5 leading-tight"
                style={{ color }}
              >
                {s.zscore > 0 ? '+' : ''}{s.zscore.toFixed(2)}σ
              </div>
              <div className="text-[9px] font-mono mt-0.5" style={{ color }}>
                {s.direction === 'long' ? '▲ LONG' : s.direction === 'short' ? '▼ SHORT' : '── NEU'}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Regime breakdown pie chart ────────────────────────────────────────────────

function RegimePieChart({ segments }: { segments: Array<{ regime: RegimeType; durationDays: number }> }) {
  const totals: Partial<Record<RegimeType, number>> = {}
  for (const s of segments) {
    totals[s.regime] = (totals[s.regime] ?? 0) + s.durationDays
  }
  const data = Object.entries(totals).map(([regime, days]) => ({
    name: REGIME_LABELS[regime as RegimeType],
    value: days,
    regime: regime as RegimeType,
  })).sort((a, b) => b.value - a.value)

  const allDays = data.reduce((a, b) => a + b.value, 0)

  return (
    <div>
      <ResponsiveContainer width="100%" height={180}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            outerRadius={68}
            innerRadius={30}
            dataKey="value"
            strokeWidth={1}
            stroke="#0e1220"
          >
            {data.map((entry, i) => (
              <Cell key={i} fill={REGIME_COLORS[entry.regime]} fillOpacity={0.85} />
            ))}
          </Pie>
          <Tooltip
            formatter={(v: number) => [`${v}d (${((v / allDays) * 100).toFixed(0)}%)`, '']}
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }}
          />
        </PieChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="grid grid-cols-2 gap-1 mt-1">
        {data.map(d => (
          <div key={d.regime} className="flex items-center gap-1.5 text-xs">
            <span className="w-2 h-2 rounded-sm shrink-0" style={{ backgroundColor: REGIME_COLORS[d.regime] }} />
            <span className="text-research-subtle truncate">{d.name}</span>
            <span className="ml-auto font-mono text-research-text">{((d.value / allDays) * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Recent activity feed ──────────────────────────────────────────────────────

function ActivityFeed({ trades }: { trades: Array<{ id: string; instrument: string; side: 'long' | 'short'; pnl: number; entryTime: string; regime: RegimeType }> }) {
  const recent = [...trades].sort((a, b) => new Date(b.entryTime).getTime() - new Date(a.entryTime).getTime()).slice(0, 8)

  return (
    <div className="space-y-1.5">
      {recent.map(t => (
        <div key={t.id} className="flex items-center gap-2 py-1 px-2 rounded hover:bg-research-muted/20 transition-colors">
          <span className={clsx('text-xs font-mono font-semibold w-16', t.pnl >= 0 ? 'text-research-bull' : 'text-research-bear')}>
            {t.side === 'long' ? '▲' : '▼'} {t.side.toUpperCase()}
          </span>
          <span className="text-xs font-mono text-research-accent font-medium flex-1">{t.instrument}</span>
          <span className={clsx('text-xs font-mono font-semibold', pnlColor(t.pnl))}>
            {formatCurrency(t.pnl, { sign: true, compact: true })}
          </span>
          <span className="text-[10px] font-mono text-research-subtle">{formatDate(t.entryTime, 'MM-dd HH:mm')}</span>
        </div>
      ))}
    </div>
  )
}

// ── BH activation tracker ────────────────────────────────────────────────────

function BHActivations() {
  // Mock BH (Bid-Hit) activations for the day
  const activations = [
    { time: '09:14:32', instrument: 'BTC-USD', type: 'BH_SPIKE', confidence: 0.87, action: 'reduce' },
    { time: '10:28:15', instrument: 'ETH-USD', type: 'BH_REVERSAL', confidence: 0.74, action: 'hold' },
    { time: '11:55:01', instrument: 'SOL-USD', type: 'BH_MOMENTUM', confidence: 0.91, action: 'increase' },
    { time: '13:02:44', instrument: 'BNB-USD', type: 'BH_SPIKE', confidence: 0.68, action: 'reduce' },
    { time: '14:17:09', instrument: 'BTC-USD', type: 'BH_REGIME_CHG', confidence: 0.95, action: 'exit' },
    { time: '15:33:28', instrument: 'AVAX-USD', type: 'BH_MOMENTUM', confidence: 0.82, action: 'increase' },
    { time: '16:48:17', instrument: 'ETH-USD', type: 'BH_REVERSAL', confidence: 0.79, action: 'hold' },
  ]

  const actionColor = (a: string) => a === 'exit' ? 'text-research-bear' : a === 'reduce' ? 'text-research-warning' : a === 'increase' ? 'text-research-bull' : 'text-research-subtle'

  return (
    <div className="space-y-1">
      {activations.map((a, i) => (
        <div key={i} className="flex items-center gap-2 py-1 px-2 rounded hover:bg-research-muted/20 text-xs">
          <span className="font-mono text-research-subtle w-16">{a.time}</span>
          <span className="font-mono text-research-accent font-medium w-20">{a.instrument}</span>
          <span className="font-mono text-research-subtle flex-1">{a.type}</span>
          <span className="font-mono text-research-info">{(a.confidence * 100).toFixed(0)}%</span>
          <span className={clsx('font-mono font-semibold w-14 text-right uppercase text-[10px]', actionColor(a.action))}>
            {a.action}
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Daily P&L bar chart ───────────────────────────────────────────────────────

function DailyPnLChart({ days = 14 }: { days?: number }) {
  let seed = 555
  function rand() { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff }
  function randn() { return Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand()) }

  const data = Array.from({ length: days }, (_, i) => {
    const pnl = randn() * 1200 + 200
    return {
      date: formatDate(new Date(Date.now() - (days - 1 - i) * 86_400_000), 'MM-dd'),
      pnl,
    }
  })

  return (
    <ResponsiveContainer width="100%" height={130}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} interval={1} />
        <YAxis tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} width={45} tickFormatter={v => `$${(v / 1000).toFixed(1)}K`} />
        <Tooltip
          formatter={(v: number) => [formatCurrency(v, { sign: true }), 'Daily P&L']}
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }}
        />
        <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.pnl >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function OverviewPage() {
  const equityQ = useQuery({ queryKey: ['equity-30d'], queryFn: () => fetchEquityCurve(30), refetchInterval: 60_000 })
  const metricsQ = useQuery({ queryKey: ['metrics'], queryFn: fetchPerformanceMetrics, refetchInterval: 60_000 })
  const topTradesQ = useQuery({ queryKey: ['top-trades'], queryFn: () => fetchTopTrades(10), refetchInterval: 60_000 })
  const allTradesQ = useQuery({ queryKey: ['all-trades-overview'], queryFn: () => fetchTrades(), refetchInterval: 120_000 })
  const regimesQ = useQuery({ queryKey: ['regimes-90d'], queryFn: () => fetchRegimeSegments(90), refetchInterval: 120_000 })

  const metrics = metricsQ.data
  const loading = metricsQ.isLoading
  const bhActivations = 7

  // Derive active positions count
  const activePositions = useMemo(() => {
    return allTradesQ.data?.filter(t => t.exitTime === null).length ?? 0
  }, [allTradesQ.data])

  return (
    <div className="space-y-5 animate-fade-in">
      {/* ── KPI cards ── */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
        <MetricCard
          label="Total P&L"
          value={metrics ? formatCurrency(metrics.totalPnl, { compact: true }) : '–'}
          subvalue={metrics ? formatPct(metrics.totalPnlPct, { sign: true }) : undefined}
          trend={metrics && metrics.totalPnl > 0 ? 'up' : 'down'}
          variant={metrics && metrics.totalPnl > 0 ? 'bull' : 'bear'}
          icon={<DollarSign size={14} />}
          loading={loading}
        />
        <MetricCard
          label="Sharpe Ratio"
          value={metrics ? formatRatio(metrics.sharpeRatio) : '–'}
          subvalue={metrics ? `Sortino ${formatRatio(metrics.sortinoRatio)}` : undefined}
          trend={metrics && metrics.sharpeRatio > 1 ? 'up' : 'down'}
          variant={metrics && metrics.sharpeRatio > 1 ? 'bull' : 'bear'}
          icon={<TrendingUp size={14} />}
          loading={loading}
        />
        <MetricCard
          label="Max Drawdown"
          value={metrics ? formatPct(metrics.maxDrawdownPct) : '–'}
          subvalue={metrics ? formatCurrency(metrics.maxDrawdown, { compact: true }) : undefined}
          trend="down"
          variant="bear"
          icon={<TrendingDown size={14} />}
          loading={loading}
        />
        <MetricCard
          label="Win Rate"
          value={metrics ? formatPct(metrics.winRate * 100) : '–'}
          subvalue={metrics ? `${metrics.totalWins}W/${metrics.totalLosses}L` : undefined}
          trend={metrics && metrics.winRate > 0.5 ? 'up' : 'down'}
          variant={metrics && metrics.winRate > 0.5 ? 'bull' : 'bear'}
          icon={<Target size={14} />}
          loading={loading}
        />
        <MetricCard
          label="Active Positions"
          value={allTradesQ.isLoading ? '…' : String(activePositions)}
          subvalue={metrics ? `of ${metrics.totalTrades} total` : undefined}
          variant="info"
          icon={<Activity size={14} />}
          loading={allTradesQ.isLoading}
        />
        <MetricCard
          label="BH Activations"
          value={String(bhActivations)}
          subvalue="today"
          variant="warning"
          icon={<Zap size={14} />}
        />
      </div>

      {/* ── Equity curve + Drawdown ── */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 space-y-3">
          <div className="bg-research-card border border-research-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-research-text">30-Day Equity Curve</h2>
              <div className="flex gap-3 text-xs font-mono text-research-subtle">
                <span className="flex items-center gap-1">
                  <span className="w-3 h-0.5 inline-block bg-blue-500 rounded" />
                  Equity
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-0.5 inline-block bg-gray-500 rounded opacity-60" style={{ borderTop: '1px dashed' }} />
                  Benchmark
                </span>
              </div>
            </div>
            {equityQ.isLoading ? <LoadingSpinner fullHeight /> :
              equityQ.error ? <ErrorDisplay error={equityQ.error as Error} /> :
              <EquityCurveChart data={equityQ.data ?? []} showBenchmark />}
          </div>

          <div className="bg-research-card border border-research-border rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold text-research-text">Drawdown</h2>
              {metrics && (
                <span className="text-xs font-mono text-research-bear">
                  Max: {formatPct(metrics.maxDrawdownPct)}
                </span>
              )}
            </div>
            {equityQ.isLoading ? <LoadingSpinner size="sm" /> :
              <DrawdownChart data={equityQ.data ?? []} height={120} />}
          </div>
        </div>

        {/* Right column: regime breakdown + performance stats */}
        <div className="space-y-3">
          <div className="bg-research-card border border-research-border rounded-lg p-4">
            <h2 className="text-sm font-semibold text-research-text mb-3">Regime Breakdown (90D)</h2>
            {regimesQ.isLoading ? <LoadingSpinner size="sm" fullHeight /> :
              <RegimePieChart segments={regimesQ.data ?? []} />}
          </div>

          <div className="bg-research-card border border-research-border rounded-lg p-4">
            <h2 className="text-sm font-semibold text-research-text mb-2">Performance Metrics</h2>
            {metricsQ.isLoading ? <LoadingSpinner size="sm" /> :
              metrics ? <PerformanceTable metrics={metrics} compact /> : null}
          </div>
        </div>
      </div>

      {/* ── Regime timeline ── */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-research-text">Regime Timeline (90D)</h2>
          <RegimeLegend />
        </div>
        {regimesQ.isLoading ? <LoadingSpinner size="sm" /> :
          <RegimeTimelineChart segments={regimesQ.data ?? []} />}
      </div>

      {/* ── Daily P&L + BH Activations ── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Daily P&L (14D)</h2>
          <DailyPnLChart days={14} />
        </div>

        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-research-text">BH Activations Today</h2>
            <span className="px-2 py-0.5 text-[10px] font-mono bg-research-warning/15 text-research-warning rounded border border-research-warning/30">
              {bhActivations} EVENTS
            </span>
          </div>
          <BHActivations />
        </div>
      </div>

      {/* ── Top trades + Signal heatmap ── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Top 10 Trades by |P&L|</h2>
          {topTradesQ.isLoading ? <LoadingSpinner fullHeight /> :
            <TradeTable trades={topTradesQ.data ?? []} maxRows={10} compact />}
        </div>

        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-research-text">Live Signal Strength</h2>
            <div className="flex items-center gap-2 text-xs font-mono text-research-subtle">
              <span className="w-16 h-2 rounded" style={{ background: 'linear-gradient(to right, #ef4444, #1e2a3a, #22c55e)' }} />
              <span>-3σ → +3σ</span>
            </div>
          </div>
          <ErrorBoundary>
            <SignalHeatmap />
          </ErrorBoundary>
        </div>
      </div>

      {/* ── Recent activity feed ── */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-research-text">Recent Trade Activity</h2>
          <span className="text-xs text-research-subtle font-mono">
            {allTradesQ.data ? `${allTradesQ.data.length} trades loaded` : ''}
          </span>
        </div>
        {allTradesQ.isLoading ? <LoadingSpinner size="sm" /> :
          <ActivityFeed trades={allTradesQ.data ?? []} />}
      </div>
    </div>
  )
}
