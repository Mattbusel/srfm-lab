import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { ErrorDisplay } from '@/components/ui/ErrorBoundary'
import { DateRangePicker } from '@/components/ui/DateRangePicker'
import { SlippageTable } from '@/components/tables/SlippageTable'
import { fetchReconciliation, fetchSlippageStats } from '@/api/reconciliation'
import { fetchSignalSnapshots } from '@/api/signals'
import { formatCurrency, formatPct, formatDate, pnlColor } from '@/utils/formatters'
import { REGIME_COLORS, REGIME_LABELS } from '@/utils/colors'
import { clsx } from 'clsx'
import type { RegimeType } from '@/types/trades'
import type { ReconciliationRow } from '@/types/trades'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine,
  LineChart, Line,
} from 'recharts'
import { format, subDays } from 'date-fns'

// ── P&L attribution waterfall ─────────────────────────────────────────────────

function WaterfallChart({ data }: { data: Array<{ label: string; value: number; cumulative: number; isTotal?: boolean }> }) {
  const chartData = data.map(d => ({
    ...d,
    base: d.cumulative - d.value,
    bar: d.value,
  }))

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 24, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          angle={-30}
          textAnchor="end"
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={v => `$${(v / 1000).toFixed(1)}K`}
          tickLine={false}
          axisLine={false}
          width={55}
        />
        <Tooltip
          formatter={(v: number) => [formatCurrency(v), '']}
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }}
        />
        <ReferenceLine y={0} stroke="#2d3a4f" />
        {/* transparent base bar */}
        <Bar dataKey="base" stackId="a" fill="transparent" />
        <Bar dataKey="bar" stackId="a" radius={[2, 2, 0, 0]}>
          {chartData.map((entry, i) => (
            <Cell key={i} fill={entry.isTotal ? '#3b82f6' : entry.bar >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Reconciliation table ──────────────────────────────────────────────────────

function ReconcTable({ rows }: { rows: ReconciliationRow[] }) {
  const thClass = "text-left text-[10px] font-medium text-research-subtle uppercase tracking-wide py-2 px-2 whitespace-nowrap"
  const tdClass = "py-1.5 px-2 font-mono text-xs border-b border-research-border/50 whitespace-nowrap"

  return (
    <div className="overflow-x-auto max-h-80 overflow-y-auto">
      <table className="w-full border-collapse">
        <thead className="sticky top-0 bg-research-card z-10">
          <tr className="border-b border-research-border">
            <th className={thClass}>Instrument</th>
            <th className={thClass}>Date</th>
            <th className={thClass}>Regime</th>
            <th className={clsx(thClass, 'text-right')}>Live P&L</th>
            <th className={clsx(thClass, 'text-right')}>BT P&L</th>
            <th className={clsx(thClass, 'text-right')}>Diff</th>
            <th className={clsx(thClass, 'text-right')}>Slippage</th>
            <th className={clsx(thClass, 'text-right')}>Sig Drift</th>
            <th className={thClass}>Notes</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="hover:bg-research-muted/20 transition-colors">
              <td className={clsx(tdClass, 'text-research-accent font-semibold')}>{row.instrument}</td>
              <td className={clsx(tdClass, 'text-research-subtle')}>{row.tradeDate}</td>
              <td className={tdClass}>
                <span style={{ color: REGIME_COLORS[row.regime] }}>{REGIME_LABELS[row.regime]}</span>
              </td>
              <td className={clsx(tdClass, 'text-right', pnlColor(row.livePnl))}>
                {formatCurrency(row.livePnl, { sign: true })}
              </td>
              <td className={clsx(tdClass, 'text-right', pnlColor(row.backtestPnl))}>
                {formatCurrency(row.backtestPnl, { sign: true })}
              </td>
              <td className={clsx(tdClass, 'text-right', Math.abs(row.pnlDiff) > 200 ? 'text-research-warning' : pnlColor(row.pnlDiff))}>
                {formatCurrency(row.pnlDiff, { sign: true })}
              </td>
              <td className={clsx(tdClass, 'text-right text-research-subtle')}>{row.slippage.toFixed(2)}</td>
              <td className={clsx(tdClass, 'text-right', Math.abs(row.signalDrift) > 0.05 ? 'text-research-warning' : 'text-research-subtle')}>
                {row.signalDrift.toFixed(4)}
              </td>
              <td className={clsx(tdClass, 'text-research-subtle text-[10px]')}>{row.notes}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Slippage histogram ─────────────────────────────────────────────────────────

function SlippageHistogram({ data }: { data: number[] }) {
  const bins = 20
  const min = Math.min(...data), max = Math.max(...data)
  const binWidth = (max - min) / bins
  const counts = Array(bins).fill(0)
  for (const v of data) {
    const idx = Math.min(bins - 1, Math.floor((v - min) / binWidth))
    counts[idx]++
  }
  const chartData = counts.map((count, i) => ({
    x: (min + (i + 0.5) * binWidth).toFixed(1),
    count,
  }))

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis dataKey="x" tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} width={30} />
        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px' }} />
        <Bar dataKey="count" fill="#f59e0b" fillOpacity={0.7} radius={[2, 2, 0, 0]} name="Frequency" />
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Per-regime comparison ─────────────────────────────────────────────────────

function RegimeComparisonTable({ rows }: { rows: ReconciliationRow[] }) {
  const regimes: RegimeType[] = ['bull', 'bear', 'sideways', 'ranging', 'volatile']
  const stats = regimes.map(regime => {
    const filtered = rows.filter(r => r.regime === regime)
    const avgDiff = filtered.length > 0
      ? filtered.reduce((a, b) => a + b.pnlDiff, 0) / filtered.length
      : 0
    const avgSlip = filtered.length > 0
      ? filtered.reduce((a, b) => a + b.slippage, 0) / filtered.length
      : 0
    return { regime, count: filtered.length, avgDiff, avgSlip }
  })

  return (
    <table className="w-full border-collapse text-xs">
      <thead>
        <tr className="border-b border-research-border">
          {['Regime', 'Count', 'Avg P&L Diff', 'Avg Slippage'].map(h => (
            <th key={h} className="text-left text-[10px] text-research-subtle uppercase tracking-wide py-2 px-3">{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {stats.map(row => (
          <tr key={row.regime} className="hover:bg-research-muted/20 border-b border-research-border/50">
            <td className="py-1.5 px-3 font-mono">
              <span style={{ color: REGIME_COLORS[row.regime] }}>{REGIME_LABELS[row.regime]}</span>
            </td>
            <td className="py-1.5 px-3 font-mono text-research-text">{row.count}</td>
            <td className={clsx('py-1.5 px-3 font-mono', pnlColor(row.avgDiff))}>
              {formatCurrency(row.avgDiff, { sign: true })}
            </td>
            <td className="py-1.5 px-3 font-mono text-research-warning">{row.avgSlip.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function ReconciliationPage() {
  const [dateRange, setDateRange] = useState({
    from: format(subDays(new Date(), 30), 'yyyy-MM-dd'),
    to: format(new Date(), 'yyyy-MM-dd'),
  })

  const reconcQ = useQuery({
    queryKey: ['reconciliation', dateRange],
    queryFn: () => fetchReconciliation(dateRange.from, dateRange.to),
  })
  const slippageQ = useQuery({ queryKey: ['slippage-stats'], queryFn: fetchSlippageStats })

  const rows = reconcQ.data ?? []
  const slippages = rows.map(r => r.slippage)

  // Build waterfall attribution data
  const waterfallData = [
    { label: 'Momentum', value: 4200, cumulative: 4200 },
    { label: 'MeanRev', value: 2800, cumulative: 7000 },
    { label: 'Regime', value: -1200, cumulative: 5800 },
    { label: 'Volume', value: 1400, cumulative: 7200 },
    { label: 'Slippage', value: -900, cumulative: 6300 },
    { label: 'Commission', value: -350, cumulative: 5950 },
    { label: 'Total', value: 5950, cumulative: 5950, isTotal: true },
  ]

  // Signal drift mock timeline
  const driftData = Array.from({ length: 30 }, (_, i) => {
    let seed = 99 + i * 7
    function rand() { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff }
    function randn() { return Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand()) }
    return {
      date: format(subDays(new Date(), 29 - i), 'MM-dd'),
      drift: randn() * 0.05,
      cumDrift: (i - 15) * 0.003 + randn() * 0.01,
    }
  })

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Header controls */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-research-subtle">
          {rows.length} reconciled trades in period
        </p>
        <DateRangePicker value={dateRange} onChange={setDateRange} />
      </div>

      {/* Live vs Backtest comparison table */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <h2 className="text-sm font-semibold text-research-text mb-3">Live vs Backtest Comparison</h2>
        {reconcQ.isLoading ? <LoadingSpinner fullHeight /> :
          reconcQ.error ? <ErrorDisplay error={reconcQ.error as Error} /> :
          <ReconcTable rows={rows} />}
      </div>

      {/* Charts grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {/* Slippage distribution */}
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Slippage Distribution</h2>
          {slippages.length > 0 ? (
            <SlippageHistogram data={slippages} />
          ) : (
            <LoadingSpinner size="sm" />
          )}
        </div>

        {/* Signal drift timeline */}
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Signal Drift Timeline (30D)</h2>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={driftData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} interval={4} />
              <YAxis tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} width={45} tickFormatter={v => v.toFixed(3)} />
              <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px' }} />
              <ReferenceLine y={0} stroke="#2d3a4f" />
              <Line type="monotone" dataKey="drift" stroke="#8b5cf6" strokeWidth={1.5} dot={false} name="Daily Drift" />
              <Line type="monotone" dataKey="cumDrift" stroke="#f59e0b" strokeWidth={1} dot={false} strokeDasharray="4 4" name="Cum Drift" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* P&L attribution waterfall */}
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">P&L Attribution Waterfall</h2>
          <WaterfallChart data={waterfallData} />
        </div>

        {/* Per-regime comparison */}
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Per-Regime Comparison</h2>
          <RegimeComparisonTable rows={rows} />
        </div>
      </div>

      {/* Slippage stats table */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <h2 className="text-sm font-semibold text-research-text mb-3">Slippage Stats by Instrument</h2>
        {slippageQ.isLoading ? <LoadingSpinner size="sm" /> :
          <SlippageTable data={slippageQ.data ?? []} />}
      </div>
    </div>
  )
}
