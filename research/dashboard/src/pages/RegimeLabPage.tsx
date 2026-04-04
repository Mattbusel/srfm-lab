import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { RegimeTimelineChart, RegimeLegend } from '@/components/charts/RegimeTimelineChart'
import { RegimeBreakdownTable } from '@/components/tables/RegimeBreakdownTable'
import {
  fetchRegimeSegments, fetchTransitionMatrix,
  fetchRegimePerformance, fetchRegimeDurations,
} from '@/api/regimes'
import { REGIME_COLORS, REGIME_LABELS, heatmapColor } from '@/utils/colors'
import { formatPct } from '@/utils/formatters'
import type { RegimeType } from '@/types/trades'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, Legend,
} from 'recharts'
import { clsx } from 'clsx'

const REGIME_ORDER: RegimeType[] = ['bull', 'bear', 'sideways', 'ranging', 'volatile']

// ── Transition matrix heatmap ─────────────────────────────────────────────────

function TransitionMatrixHeatmap({ matrix }: { matrix: Record<RegimeType, Record<RegimeType, number>> }) {
  return (
    <div className="overflow-auto">
      <table className="border-collapse">
        <thead>
          <tr>
            <th className="text-[9px] font-mono text-research-subtle text-right pr-2" style={{ width: 72 }}>From\To</th>
            {REGIME_ORDER.map(to => (
              <th key={to} className="text-[10px] font-mono text-research-subtle font-normal text-center pb-1" style={{ width: 68 }}>
                {REGIME_LABELS[to]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {REGIME_ORDER.map(from => (
            <tr key={from}>
              <td className="text-[10px] font-mono pr-2 text-right" style={{ color: REGIME_COLORS[from] }}>
                {REGIME_LABELS[from]}
              </td>
              {REGIME_ORDER.map(to => {
                const val = matrix[from]?.[to] ?? 0
                const bg = heatmapColor(val, 0, 1)
                const isdiag = from === to
                return (
                  <td
                    key={to}
                    title={`${REGIME_LABELS[from]} → ${REGIME_LABELS[to]}: ${formatPct(val * 100)}`}
                    className="text-center font-mono cursor-default"
                    style={{
                      backgroundColor: bg,
                      width: 68,
                      height: 36,
                      fontSize: 11,
                      color: val > 0.4 ? '#fff' : '#8899aa',
                      border: isdiag ? '1px solid rgba(255,255,255,0.2)' : undefined,
                      fontWeight: isdiag ? 600 : 400,
                    }}
                  >
                    {formatPct(val * 100, { decimals: 1 })}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Regime duration distribution ──────────────────────────────────────────────

function RegimeDurationChart({ data }: { data: Array<{ regime: RegimeType; avgDays: number; p25Days: number; p75Days: number; medianDays: number }> }) {
  const chartData = data.map(d => ({
    regime: REGIME_LABELS[d.regime],
    regimeKey: d.regime,
    avg: d.avgDays,
    p25: d.p25Days,
    iqr: d.p75Days - d.p25Days,
    median: d.medianDays,
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis dataKey="regime" tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} width={35} label={{ value: 'Days', angle: -90, position: 'insideLeft', offset: 8, fontSize: 9, fill: '#8899aa' }} />
        <Tooltip
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }}
          formatter={(v: number) => [`${v.toFixed(1)}d`, '']}
        />
        <Bar dataKey="avg" radius={[3, 3, 0, 0]} name="Avg Duration">
          {chartData.map((entry, i) => (
            <Cell key={i} fill={REGIME_COLORS[entry.regimeKey]} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Per-regime performance bar chart ─────────────────────────────────────────

function RegimePerfChart({ data }: { data: Array<{ regime: RegimeType; sharpe: number; winRate: number }> }) {
  const chartData = data.map(d => ({
    regime: REGIME_LABELS[d.regime],
    regimeKey: d.regime,
    sharpe: d.sharpe,
    winRate: d.winRate * 100,
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis dataKey="regime" tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }} tickLine={false} axisLine={false} width={35} />
        <Tooltip
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: '#8899aa' }} />
        <Bar dataKey="sharpe" fill="#3b82f6" fillOpacity={0.8} name="Sharpe" radius={[2, 2, 0, 0]} />
        <Bar dataKey="winRate" fill="#22c55e" fillOpacity={0.6} name="Win Rate %" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function RegimeLabPage() {
  const segmentsQ = useQuery({ queryKey: ['regime-segments'], queryFn: () => fetchRegimeSegments(180) })
  const matrixQ = useQuery({ queryKey: ['transition-matrix'], queryFn: fetchTransitionMatrix })
  const perfQ = useQuery({ queryKey: ['regime-perf'], queryFn: fetchRegimePerformance })
  const durQ = useQuery({ queryKey: ['regime-dur'], queryFn: fetchRegimeDurations })

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Regime timeline */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-research-text">Regime Timeline (180D)</h2>
          <RegimeLegend />
        </div>
        {segmentsQ.isLoading ? <LoadingSpinner size="sm" /> :
          <RegimeTimelineChart segments={segmentsQ.data ?? []} height={100} />}

        {/* Regime stats row */}
        {segmentsQ.data && (
          <div className="mt-3 grid grid-cols-5 gap-2">
            {REGIME_ORDER.map(regime => {
              const segs = segmentsQ.data.filter(s => s.regime === regime)
              const totalDays = segs.reduce((a, s) => a + s.durationDays, 0)
              const allDays = segmentsQ.data.reduce((a, s) => a + s.durationDays, 0)
              return (
                <div key={regime} className="text-center">
                  <div className="text-[10px] font-mono" style={{ color: REGIME_COLORS[regime] }}>
                    {REGIME_LABELS[regime]}
                  </div>
                  <div className="text-xs font-semibold text-research-text font-mono">
                    {formatPct(totalDays / allDays * 100, { decimals: 0 })}
                  </div>
                  <div className="text-[10px] text-research-subtle font-mono">{segs.length} segs</div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Transition matrix + Duration distribution */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Transition Matrix</h2>
          {matrixQ.isLoading ? <LoadingSpinner size="sm" /> :
            matrixQ.data ? <TransitionMatrixHeatmap matrix={matrixQ.data} /> : null}
        </div>

        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Regime Duration Distribution</h2>
          {durQ.isLoading ? <LoadingSpinner size="sm" /> :
            durQ.data ? <RegimeDurationChart data={durQ.data} /> : null}
        </div>
      </div>

      {/* Per-regime performance */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Per-Regime Performance (Sharpe & Win Rate)</h2>
          {perfQ.isLoading ? <LoadingSpinner size="sm" /> :
            perfQ.data ? <RegimePerfChart data={perfQ.data} /> : null}
        </div>

        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Regime Duration Stats</h2>
          {durQ.data && (
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr className="border-b border-research-border">
                  {['Regime', 'Avg', 'Median', 'P25', 'P75', 'Min', 'Max', 'Count'].map(h => (
                    <th key={h} className="text-left text-[10px] text-research-subtle uppercase tracking-wide py-2 px-2 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {durQ.data.map(d => (
                  <tr key={d.regime} className="hover:bg-research-muted/20 border-b border-research-border/50">
                    <td className="py-1.5 px-2 font-mono" style={{ color: REGIME_COLORS[d.regime] }}>{REGIME_LABELS[d.regime]}</td>
                    <td className="py-1.5 px-2 font-mono text-research-text">{d.avgDays.toFixed(1)}d</td>
                    <td className="py-1.5 px-2 font-mono text-research-text">{d.medianDays.toFixed(1)}d</td>
                    <td className="py-1.5 px-2 font-mono text-research-subtle">{d.p25Days.toFixed(1)}d</td>
                    <td className="py-1.5 px-2 font-mono text-research-subtle">{d.p75Days.toFixed(1)}d</td>
                    <td className="py-1.5 px-2 font-mono text-research-subtle">{d.minDays}d</td>
                    <td className="py-1.5 px-2 font-mono text-research-subtle">{d.maxDays}d</td>
                    <td className="py-1.5 px-2 font-mono text-research-subtle">{d.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Full breakdown table */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <h2 className="text-sm font-semibold text-research-text mb-3">Detailed Regime Performance Breakdown</h2>
        {perfQ.isLoading ? <LoadingSpinner size="sm" /> :
          perfQ.data ? <RegimeBreakdownTable data={perfQ.data} /> : null}
      </div>
    </div>
  )
}
