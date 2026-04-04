import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { MetricCard } from '@/components/ui/MetricCard'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { MCFanChart } from '@/components/charts/MCFanChart'
import { fetchMCResults } from '@/api/mc'
import { formatCurrency, formatPct, formatRatio } from '@/utils/formatters'
import { REGIME_COLORS, REGIME_LABELS, CHART_COLORS } from '@/utils/colors'
import { kellyFraction } from '@/utils/calculations'
import { clsx } from 'clsx'
import { Play } from 'lucide-react'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Cell, RadialBarChart, RadialBar, PolarAngleAxis,
} from 'recharts'
import type { RegimeType } from '@/types/trades'

// ── Final equity distribution histogram ──────────────────────────────────────

function FinalEquityHistogram({ values, initial }: { values: number[]; initial: number }) {
  const bins = 20
  const min = Math.min(...values), max = Math.max(...values)
  const binW = (max - min) / bins
  const counts = Array(bins).fill(0)
  for (const v of values) counts[Math.min(bins - 1, Math.floor((v - min) / binW))]++
  const data = counts.map((count, i) => ({
    x: `${((min + (i + 0.5) * binW) / 1000).toFixed(0)}K`,
    count,
    val: min + (i + 0.5) * binW,
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis dataKey="x" tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} width={30} />
        <Tooltip
          formatter={(v: number) => [v, 'Frequency']}
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px' }}
        />
        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.val >= initial ? '#22c55e' : '#ef4444'} fillOpacity={0.7} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Blowup rate gauge ─────────────────────────────────────────────────────────

function BlowupGauge({ rate }: { rate: number }) {
  const pct = rate * 100
  const color = pct < 2 ? '#22c55e' : pct < 5 ? '#f59e0b' : '#ef4444'
  const data = [{ value: pct, fill: color }]

  return (
    <div className="flex flex-col items-center gap-2">
      <ResponsiveContainer width={160} height={100}>
        <RadialBarChart
          cx="50%"
          cy="85%"
          innerRadius="70%"
          outerRadius="100%"
          startAngle={180}
          endAngle={0}
          data={data}
          barSize={16}
        >
          <PolarAngleAxis type="number" domain={[0, 10]} angleAxisId={0} tick={false} />
          <RadialBar
            background={{ fill: '#1e2a3a' }}
            dataKey="value"
            cornerRadius={4}
            angleAxisId={0}
          />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="text-center -mt-4">
        <div className="text-2xl font-bold font-mono" style={{ color }}>{pct.toFixed(1)}%</div>
        <div className="text-xs text-research-subtle">Blowup Rate</div>
      </div>
    </div>
  )
}

// ── Regime-stratified table ───────────────────────────────────────────────────

function RegimeStratifiedTable({ data }: { data: Array<{ regime: string; blowupRate: number; p50FinalEquity: number; p5FinalEquity: number; p95FinalEquity: number; expectedReturn: number }> }) {
  return (
    <table className="w-full border-collapse text-xs">
      <thead>
        <tr className="border-b border-research-border">
          {['Regime', 'Blowup Rate', 'P50 Final', 'P5 Final', 'P95 Final', 'Exp Return'].map(h => (
            <th key={h} className="text-left text-[10px] text-research-subtle uppercase tracking-wide py-2 px-3 whitespace-nowrap">{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map(row => (
          <tr key={row.regime} className="hover:bg-research-muted/20 border-b border-research-border/50">
            <td className="py-1.5 px-3 font-mono" style={{ color: REGIME_COLORS[row.regime as RegimeType] }}>
              {REGIME_LABELS[row.regime as RegimeType] ?? row.regime}
            </td>
            <td className={clsx('py-1.5 px-3 font-mono', row.blowupRate < 0.02 ? 'text-research-bull' : row.blowupRate < 0.05 ? 'text-research-warning' : 'text-research-bear')}>
              {formatPct(row.blowupRate * 100)}
            </td>
            <td className="py-1.5 px-3 font-mono text-research-text">{formatCurrency(row.p50FinalEquity, { compact: true })}</td>
            <td className="py-1.5 px-3 font-mono text-research-bear">{formatCurrency(row.p5FinalEquity, { compact: true })}</td>
            <td className="py-1.5 px-3 font-mono text-research-bull">{formatCurrency(row.p95FinalEquity, { compact: true })}</td>
            <td className={clsx('py-1.5 px-3 font-mono', row.expectedReturn >= 0 ? 'text-research-bull' : 'text-research-bear')}>
              {formatPct(row.expectedReturn * 100, { sign: true })}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function MCSimPage() {
  const [nDays, setNDays] = useState(252)
  const [nPaths, setNPaths] = useState(10000)

  const mcQ = useQuery({
    queryKey: ['mc-results', nDays, nPaths],
    queryFn: () => fetchMCResults({ nDays, nPaths, initialEquity: 100_000 }),
  })

  const results = mcQ.data
  const kelly = results ? kellyFraction(0.613, 680, 320) : 0

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Controls */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2 text-sm">
          <label className="text-research-subtle text-xs">Paths:</label>
          <select
            value={nPaths}
            onChange={e => setNPaths(Number(e.target.value))}
            className="bg-research-surface border border-research-border rounded px-2 py-1 text-xs font-mono text-research-text focus:outline-none focus:border-research-accent"
          >
            {[1000, 5000, 10000].map(v => <option key={v} value={v}>{v.toLocaleString()}</option>)}
          </select>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <label className="text-research-subtle text-xs">Days:</label>
          <select
            value={nDays}
            onChange={e => setNDays(Number(e.target.value))}
            className="bg-research-surface border border-research-border rounded px-2 py-1 text-xs font-mono text-research-text focus:outline-none focus:border-research-accent"
          >
            {[90, 180, 252, 504].map(v => <option key={v} value={v}>{v}d</option>)}
          </select>
        </div>
        <button
          onClick={() => mcQ.refetch()}
          className="flex items-center gap-2 px-3 py-1.5 text-xs bg-research-accent hover:bg-research-accent-dim text-white rounded transition-colors"
        >
          <Play size={12} />
          Run Simulation
        </button>
        {mcQ.isFetching && <span className="text-xs text-research-subtle animate-pulse">Running {nPaths.toLocaleString()} paths...</span>}
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
        <MetricCard label="P50 Final Equity" value={results ? formatCurrency(results.p50FinalEquity, { compact: true }) : '–'} variant="info" loading={mcQ.isLoading} />
        <MetricCard label="P5 Final Equity" value={results ? formatCurrency(results.p5FinalEquity, { compact: true }) : '–'} variant="bear" loading={mcQ.isLoading} />
        <MetricCard label="P95 Final Equity" value={results ? formatCurrency(results.p95FinalEquity, { compact: true }) : '–'} variant="bull" loading={mcQ.isLoading} />
        <MetricCard label="Expected Return" value={results ? formatPct(results.annualizedReturn * 100) : '–'} loading={mcQ.isLoading} />
        <MetricCard label="Est. Sharpe" value={results ? formatRatio(results.sharpeEstimate) : '–'} variant={results && results.sharpeEstimate > 1 ? 'bull' : 'bear'} loading={mcQ.isLoading} />
        <MetricCard
          label="Kelly Fraction"
          value={`${formatPct(kelly * 100, { decimals: 1 })}`}
          subvalue={results ? `opt ${formatPct(results.optimalFraction * 100, { decimals: 1 })}` : undefined}
          variant="warning"
          loading={mcQ.isLoading}
        />
      </div>

      {/* Fan chart */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-research-text">Monte Carlo Fan Chart ({results?.nPaths.toLocaleString() ?? '–'} paths)</h2>
          <div className="flex gap-3 text-xs font-mono text-research-subtle">
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 inline-block bg-blue-500" /> Median</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 inline-block bg-amber-500 opacity-70" style={{ borderStyle: 'dashed' }} /> Mean</span>
          </div>
        </div>
        {mcQ.isLoading ? <LoadingSpinner fullHeight /> :
          results ? <MCFanChart bands={results.bands} initialEquity={results.initialEquity} height={320} /> : null}
      </div>

      {/* Distribution histogram + Blowup gauge */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2 bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Final Equity Distribution</h2>
          {mcQ.isLoading ? <LoadingSpinner size="sm" /> :
            results ? <FinalEquityHistogram values={results.finalEquityDistribution} initial={results.initialEquity} /> : null}
        </div>

        <div className="bg-research-card border border-research-border rounded-lg p-4 flex flex-col items-center justify-center">
          <h2 className="text-sm font-semibold text-research-text mb-4 self-start">Blowup Rate</h2>
          {results ? <BlowupGauge rate={results.blowupRate} /> : <LoadingSpinner size="md" />}
          {results && (
            <div className="mt-3 text-xs text-research-subtle text-center font-mono">
              Threshold: {formatCurrency(results.blowupThreshold, { compact: true })}
              <br />
              ({formatPct(results.blowupThreshold / results.initialEquity * 100, { decimals: 0 })} of initial)
            </div>
          )}
        </div>
      </div>

      {/* Regime-stratified comparison */}
      {results?.regimeStratified && (
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Regime-Stratified MC Comparison</h2>
          <RegimeStratifiedTable data={results.regimeStratified} />
        </div>
      )}
    </div>
  )
}
