import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { MetricCard } from '@/components/ui/MetricCard'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { InstrumentSelector } from '@/components/ui/InstrumentSelector'
import { ICDecayChart } from '@/components/charts/ICDecayChart'
import { QuintileBarChart } from '@/components/charts/QuintileBarChart'
import {
  fetchICDecay, fetchRollingIC, fetchICByRegime,
  fetchFactorAttribution, fetchQuintileReturns,
} from '@/api/signals'
import { formatRatio, formatPct } from '@/utils/formatters'
import { REGIME_COLORS, REGIME_LABELS, CHART_COLORS } from '@/utils/colors'
import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell, ReferenceLine, ComposedChart, Area,
} from 'recharts'
import { clsx } from 'clsx'
import type { FactorAttribution } from '@/types/signals'

// ── Rolling IC chart ──────────────────────────────────────────────────────────

function RollingICChart({ instrument }: { instrument: string | null }) {
  const { data, isLoading } = useQuery({
    queryKey: ['rolling-ic', instrument],
    queryFn: () => fetchRollingIC(120, instrument ?? undefined),
  })

  if (isLoading) return <LoadingSpinner fullHeight />

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <defs>
          <linearGradient id="icRollingGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22c55e" stopOpacity={0.2} />
            <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickFormatter={v => v.slice(5)} tickLine={false} axisLine={false} interval={14} />
        <YAxis tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} width={50} tickFormatter={v => v.toFixed(3)} />
        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }} />
        <ReferenceLine y={0} stroke="#2d3a4f" />
        <Area type="monotone" dataKey="ic" stroke="#22c55e" strokeWidth={1.5} fill="url(#icRollingGrad)" dot={false} name="Rolling IC (60d)" />
        <Line type="monotone" dataKey="icMean" stroke="#f59e0b" strokeWidth={1} dot={false} strokeDasharray="4 4" name="Mean" />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

// ── IC by regime ──────────────────────────────────────────────────────────────

function ICByRegimeChart({ instrument }: { instrument: string | null }) {
  const { data, isLoading } = useQuery({
    queryKey: ['ic-by-regime', instrument],
    queryFn: () => fetchICByRegime(instrument ?? undefined),
  })

  if (isLoading) return <LoadingSpinner size="sm" />

  const chartData = (data ?? []).map(d => ({
    regime: REGIME_LABELS[d.regime as keyof typeof REGIME_LABELS] ?? d.regime,
    ic: d.ic,
    regimeKey: d.regime,
    tStat: d.tStat,
    n: d.sampleSize,
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis dataKey="regime" tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} width={50} tickFormatter={v => v.toFixed(3)} />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const d = payload[0].payload as typeof chartData[0]
            return (
              <div className="bg-research-card border border-research-border rounded p-2 text-xs font-mono shadow-xl">
                <div className="text-research-text font-medium">{d.regime}</div>
                <div className="text-research-subtle">IC: <span className="text-research-text">{d.ic.toFixed(4)}</span></div>
                <div className="text-research-subtle">t-stat: <span className={Math.abs(d.tStat) > 2 ? 'text-research-bull' : 'text-research-warning'}>{d.tStat.toFixed(2)}</span></div>
                <div className="text-research-subtle">n={d.n}</div>
              </div>
            )
          }}
        />
        <ReferenceLine y={0} stroke="#2d3a4f" />
        <Bar dataKey="ic" radius={[3, 3, 0, 0]} name="IC">
          {chartData.map((entry, i) => (
            <Cell key={i} fill={REGIME_COLORS[entry.regimeKey as keyof typeof REGIME_COLORS] ?? '#3b82f6'} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Factor attribution waterfall ──────────────────────────────────────────────

function FactorWaterfall({ data }: { data: FactorAttribution[] }) {
  const sorted = [...data].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
  let cumulative = 0
  const chartData = sorted.map(f => {
    cumulative += f.contribution
    return {
      factor: f.factor,
      base: cumulative - f.contribution,
      bar: f.contribution,
      contribution: f.contribution,
      tStat: f.tStat,
      active: f.active,
    }
  })

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 24, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis dataKey="factor" tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} angle={-20} textAnchor="end" tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} width={55} tickFormatter={v => `$${(v / 1000).toFixed(1)}K`} />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const d = payload[1]?.payload as typeof chartData[0]
            if (!d) return null
            return (
              <div className="bg-research-card border border-research-border rounded p-2 text-xs font-mono shadow-xl">
                <div className="text-research-text font-medium">{d.factor}</div>
                <div className={clsx('font-medium', d.contribution >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                  ${d.contribution.toFixed(0)}
                </div>
                <div className="text-research-subtle">t={d.tStat.toFixed(2)}</div>
              </div>
            )
          }}
        />
        <ReferenceLine y={0} stroke="#2d3a4f" />
        <Bar dataKey="base" stackId="a" fill="transparent" />
        <Bar dataKey="bar" stackId="a" radius={[2, 2, 0, 0]}>
          {chartData.map((entry, i) => (
            <Cell key={i} fill={entry.contribution >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={entry.active ? 0.85 : 0.35} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function SignalAnalyticsPage() {
  const [instrument, setInstrument] = useState<string | null>(null)

  const icDecayQ = useQuery({
    queryKey: ['ic-decay', instrument],
    queryFn: () => fetchICDecay(instrument ?? undefined),
  })
  const quintileQ = useQuery({
    queryKey: ['quintiles', instrument],
    queryFn: () => fetchQuintileReturns(instrument ?? undefined),
  })
  const factorQ = useQuery({
    queryKey: ['factor-attr'],
    queryFn: fetchFactorAttribution,
  })

  const icData = icDecayQ.data ?? []
  const halfLife = icData.length > 1
    ? Math.log(2) / (-Math.log(icData[1]?.ic / (icData[0]?.ic || 1)) || 0.08)
    : 8.7

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Controls */}
      <div className="flex items-center gap-3">
        <InstrumentSelector value={instrument} onChange={setInstrument} />
        <span className="text-xs text-research-subtle">
          {instrument ? `Showing: ${instrument}` : 'All instruments aggregated'}
        </span>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="IC (lag=1)"
          value={icData[1] ? formatRatio(icData[1].ic, 4) : '–'}
          variant={icData[1]?.ic && icData[1].ic > 0 ? 'bull' : 'bear'}
        />
        <MetricCard
          label="Alpha Decay Half-Life"
          value={`${isFinite(halfLife) ? halfLife.toFixed(1) : '8.7'}d`}
          variant="info"
        />
        <MetricCard
          label="IC t-stat (lag=1)"
          value={icData[1] ? formatRatio(icData[1].tStat, 2) : '–'}
          variant={icData[1]?.tStat && Math.abs(icData[1].tStat) > 2 ? 'bull' : 'warning'}
        />
        <MetricCard
          label="Q5–Q1 Spread"
          value={quintileQ.data
            ? formatPct(((quintileQ.data[4]?.avgReturn ?? 0) - (quintileQ.data[0]?.avgReturn ?? 0)) * 100, { sign: true, decimals: 2 })
            : '–'}
          variant="info"
        />
      </div>

      {/* IC decay + Rolling IC */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-research-text">IC Decay Curve</h2>
            <span className="text-xs text-research-subtle font-mono">with 95% confidence bands</span>
          </div>
          {icDecayQ.isLoading ? <LoadingSpinner fullHeight /> : <ICDecayChart data={icData} />}
        </div>

        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Rolling IC Timeline (60-day window)</h2>
          <RollingICChart instrument={instrument} />
        </div>
      </div>

      {/* IC by regime + Quintile chart */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">IC by Regime</h2>
          <ICByRegimeChart instrument={instrument} />
        </div>

        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Quintile Returns (Q1–Q5)</h2>
          {quintileQ.isLoading ? <LoadingSpinner fullHeight /> :
            <QuintileBarChart data={quintileQ.data ?? []} />}
          {quintileQ.data && (
            <div className="mt-2 grid grid-cols-5 gap-1">
              {quintileQ.data.map(q => (
                <div key={q.quintile} className="text-center">
                  <div className="text-[10px] text-research-subtle font-mono">Q{q.quintile}</div>
                  <div className={clsx('text-xs font-mono font-medium', q.avgReturn >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                    {formatPct(q.avgReturn * 100, { sign: true, decimals: 2 })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Factor attribution */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <h2 className="text-sm font-semibold text-research-text mb-3">Factor Attribution Waterfall</h2>
        {factorQ.isLoading ? <LoadingSpinner size="sm" /> : <FactorWaterfall data={factorQ.data ?? []} />}
        {factorQ.data && (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-research-border">
                  {['Factor', 'Contribution ($)', 'Contribution (%)', 't-stat', 'Active'].map(h => (
                    <th key={h} className="text-left text-[10px] text-research-subtle uppercase tracking-wide py-1.5 px-3">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {factorQ.data.map(f => (
                  <tr key={f.factor} className="hover:bg-research-muted/20 border-b border-research-border/50">
                    <td className="py-1.5 px-3 font-mono text-research-text">{f.factor}</td>
                    <td className={clsx('py-1.5 px-3 font-mono', f.contribution >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                      {f.contribution >= 0 ? '+' : ''}${f.contribution.toFixed(0)}
                    </td>
                    <td className={clsx('py-1.5 px-3 font-mono', f.contributionPct >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                      {formatPct(f.contributionPct, { sign: true })}
                    </td>
                    <td className={clsx('py-1.5 px-3 font-mono', Math.abs(f.tStat) > 2 ? 'text-research-bull' : 'text-research-subtle')}>
                      {f.tStat.toFixed(2)}
                    </td>
                    <td className="py-1.5 px-3">
                      <span className={clsx('px-1.5 py-0.5 rounded text-[10px] font-mono', f.active ? 'bg-research-bull/15 text-research-bull' : 'bg-research-muted text-research-subtle')}>
                        {f.active ? 'YES' : 'NO'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
