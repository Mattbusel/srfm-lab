import React, { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { MetricCard } from '@/components/ui/MetricCard'
import { ParameterHeatmap, generateMockParameterHeatmap } from '@/components/charts/ParameterHeatmap'
import {
  ISvsOOSScatter,
  FoldEquityCurves,
  generateMockWalkForwardFolds,
  type WalkForwardFold,
} from '@/components/charts/WalkForwardChart'
import { deflatedSharpe } from '@/utils/calculations'
import { formatRatio, formatPct } from '@/utils/formatters'
import { clsx } from 'clsx'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine,
} from 'recharts'
import { CHART_COLORS } from '@/utils/colors'

// ── CPCV path distribution ─────────────────────────────────────────────────────

function CPCVHistogram({ values }: { values: number[] }) {
  const bins = 15
  const min = Math.min(...values), max = Math.max(...values)
  const binW = (max - min) / bins
  const counts = Array(bins).fill(0)
  for (const v of values) counts[Math.min(bins - 1, Math.floor((v - min) / binW))]++
  const data = counts.map((count, i) => ({ x: (min + (i + 0.5) * binW).toFixed(2), count }))

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis dataKey="x" tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} width={28} />
        <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px' }} />
        <ReferenceLine x="0.00" stroke="#2d3a4f" />
        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={parseFloat(d.x) >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={0.75} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Parameter stability table ─────────────────────────────────────────────────

function ParamStabilityTable({ folds }: { folds: WalkForwardFold[] }) {
  const allParams = new Set<string>()
  for (const fold of folds) for (const k of Object.keys(fold.selectedParams)) allParams.add(k)
  const params = [...allParams]

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="border-b border-research-border">
            <th className="text-left text-[10px] text-research-subtle uppercase tracking-wide py-2 px-3">Parameter</th>
            {folds.map(f => (
              <th key={f.foldId} className="text-center text-[10px] text-research-subtle uppercase tracking-wide py-2 px-2">
                Fold {f.foldId + 1}
              </th>
            ))}
            <th className="text-center text-[10px] text-research-subtle uppercase tracking-wide py-2 px-2">Stability</th>
          </tr>
        </thead>
        <tbody>
          {params.map(param => {
            const values = folds.map(f => f.selectedParams[param] ?? null)
            const nonNull = values.filter(v => v !== null) as number[]
            const mean = nonNull.reduce((a, b) => a + b, 0) / nonNull.length
            const std = Math.sqrt(nonNull.reduce((a, b) => a + (b - mean) ** 2, 0) / nonNull.length)
            const cv = mean !== 0 ? std / mean : 1
            return (
              <tr key={param} className="hover:bg-research-muted/20 border-b border-research-border/50">
                <td className="py-1.5 px-3 font-mono text-research-text capitalize">{param}</td>
                {values.map((v, i) => (
                  <td key={i} className="py-1.5 px-2 font-mono text-center text-research-text">
                    {v !== null ? (
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px]"
                        style={{
                          backgroundColor: CHART_COLORS[folds[i].foldId % CHART_COLORS.length] + '22',
                          color: CHART_COLORS[folds[i].foldId % CHART_COLORS.length],
                        }}
                      >
                        {typeof v === 'number' && v < 10 ? v.toFixed(2) : v}
                      </span>
                    ) : '–'}
                  </td>
                ))}
                <td className="py-1.5 px-2 text-center">
                  <div className={clsx('text-[10px] font-mono', cv < 0.3 ? 'text-research-bull' : cv < 0.6 ? 'text-research-warning' : 'text-research-bear')}>
                    {cv < 0.3 ? 'STABLE' : cv < 0.6 ? 'MODERATE' : 'UNSTABLE'}
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function WalkForwardPage() {
  const folds = useMemo(() => generateMockWalkForwardFolds(5), [])
  const heatmapData = useMemo(() => generateMockParameterHeatmap(), [])

  // CPCV: generate many OOS Sharpe paths
  const cpvcPaths = useMemo(() => {
    let seed = 123
    function rand() { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff }
    function randn() { return Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand()) }
    return Array.from({ length: 200 }, () => randn() * 0.4 + 0.3)
  }, [])

  const avgISSharpe = folds.reduce((a, b) => a + b.isSharpe, 0) / folds.length
  const avgOOSSharpe = folds.reduce((a, b) => a + b.oosSharpe, 0) / folds.length
  const overfit = (avgISSharpe - avgOOSSharpe) / avgISSharpe
  const dsr = deflatedSharpe(avgOOSSharpe, folds.length, 252, 0.2, 0.5)

  return (
    <div className="space-y-4 animate-fade-in">
      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Avg IS Sharpe" value={formatRatio(avgISSharpe)} variant="info" />
        <MetricCard
          label="Avg OOS Sharpe"
          value={formatRatio(avgOOSSharpe)}
          variant={avgOOSSharpe > 0.5 ? 'bull' : 'bear'}
        />
        <MetricCard
          label="Overfit Ratio"
          value={formatPct(overfit * 100)}
          subvalue="IS−OOS / IS"
          variant={overfit < 0.3 ? 'bull' : overfit < 0.6 ? 'warning' : 'bear'}
        />
        <MetricCard
          label="Deflated Sharpe"
          value={formatRatio(dsr, 3)}
          subvalue="Bailey-Lopez"
          variant={dsr > 0 ? 'bull' : 'bear'}
        />
      </div>

      {/* IS vs OOS scatter + fold curves */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">IS vs OOS Sharpe (per fold)</h2>
          <ISvsOOSScatter folds={folds} />
        </div>
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Fold-by-Fold Equity Curves (normalized)</h2>
          <FoldEquityCurves folds={folds} />
        </div>
      </div>

      {/* Parameter stability */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <h2 className="text-sm font-semibold text-research-text mb-3">Parameter Stability Across Folds</h2>
        <ParamStabilityTable folds={folds} />
      </div>

      {/* CPCV distribution + Parameter heatmap */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-2">CPCV Path OOS Sharpe Distribution</h2>
          <div className="text-xs text-research-subtle mb-3 font-mono">
            200 paths · mean={formatRatio(avgOOSSharpe)} · p5={formatRatio(Math.min(...cpvcPaths))}
          </div>
          <CPCVHistogram values={cpvcPaths} />
        </div>
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Parameter Heatmap (OOS Sharpe)</h2>
          <ParameterHeatmap data={heatmapData} />
        </div>
      </div>

      {/* Fold summary table */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <h2 className="text-sm font-semibold text-research-text mb-3">Fold Summary</h2>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="border-b border-research-border">
                {['Fold', 'IS Period', 'OOS Period', 'IS Sharpe', 'OOS Sharpe', 'IS Return', 'OOS Return', 'Degradation'].map(h => (
                  <th key={h} className="text-left text-[10px] text-research-subtle uppercase tracking-wide py-2 px-3 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {folds.map(f => {
                const deg = (f.isSharpe - f.oosSharpe) / f.isSharpe
                return (
                  <tr key={f.foldId} className="hover:bg-research-muted/20 border-b border-research-border/50">
                    <td className="py-1.5 px-3 font-mono">
                      <span style={{ color: CHART_COLORS[f.foldId % CHART_COLORS.length] }}>Fold {f.foldId + 1}</span>
                    </td>
                    <td className="py-1.5 px-3 font-mono text-research-subtle text-[10px]">{f.isStartDate}→{f.isEndDate}</td>
                    <td className="py-1.5 px-3 font-mono text-research-subtle text-[10px]">{f.oosStartDate}→{f.oosEndDate}</td>
                    <td className="py-1.5 px-3 font-mono text-research-info">{formatRatio(f.isSharpe)}</td>
                    <td className={clsx('py-1.5 px-3 font-mono', f.oosSharpe > 0.5 ? 'text-research-bull' : 'text-research-bear')}>
                      {formatRatio(f.oosSharpe)}
                    </td>
                    <td className={clsx('py-1.5 px-3 font-mono', f.isReturn >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                      {formatPct(f.isReturn * 100, { sign: true })}
                    </td>
                    <td className={clsx('py-1.5 px-3 font-mono', f.oosReturn >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                      {formatPct(f.oosReturn * 100, { sign: true })}
                    </td>
                    <td className={clsx('py-1.5 px-3 font-mono', deg < 0.3 ? 'text-research-bull' : deg < 0.6 ? 'text-research-warning' : 'text-research-bear')}>
                      {formatPct(deg * 100)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
