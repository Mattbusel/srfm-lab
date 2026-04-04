import React, { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { MetricCard } from '@/components/ui/MetricCard'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { CorrelationHeatmap, buildCorrelationMatrix } from '@/components/charts/CorrelationHeatmap'
import { fetchPortfolioWeights, fetchEfficientFrontier, fetchRiskContribution } from '@/api/portfolio'
import { INSTRUMENTS } from '@/api/mockData'
import { CHART_COLORS } from '@/utils/colors'
import { formatCurrency, formatPct, formatRatio } from '@/utils/formatters'
import { clsx } from 'clsx'
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend,
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ZAxis,
  BarChart, Bar,
} from 'recharts'

// ── Weights pie chart ─────────���──────────────────────────────────��────────────

function WeightsPieChart({ weights }: { weights: Array<{ instrument: string; weight: number }> }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <PieChart>
        <Pie
          data={weights}
          cx="50%"
          cy="46%"
          outerRadius={90}
          dataKey="weight"
          nameKey="instrument"
          strokeWidth={1}
          stroke="#0e1220"
          label={({ instrument, weight }) => `${instrument.replace('-USD', '')} ${formatPct(weight * 100, { decimals: 0 })}`}
          labelLine={{ stroke: '#475569', strokeWidth: 0.5 }}
        >
          {weights.map((_, i) => (
            <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.85} />
          ))}
        </Pie>
        <Tooltip
          formatter={(v: number) => [formatPct(v * 100), 'Weight']}
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}

// ── Efficient frontier ────────────────────────────────────────────────────���───

function EfficientFrontierPlot({ data }: { data: Array<{ expectedReturn: number; volatility: number; sharpe: number; isOptimal: boolean }> }) {
  const optimal = data.find(p => p.isOptimal)
  const chartData = data.map(p => ({ ...p, ret: p.expectedReturn * 100, vol: p.volatility * 100 }))

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
        <XAxis
          dataKey="vol"
          name="Volatility"
          type="number"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          label={{ value: 'Volatility %', position: 'insideBottom', offset: -10, fontSize: 10, fill: '#8899aa' }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          dataKey="ret"
          name="Return"
          type="number"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          label={{ value: 'Return %', angle: -90, position: 'insideLeft', offset: 10, fontSize: 10, fill: '#8899aa' }}
          tickLine={false}
          axisLine={false}
        />
        <ZAxis range={[40, 40]} />
        <Tooltip
          cursor={{ strokeDasharray: '3 3' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const d = payload[0].payload as typeof chartData[0]
            return (
              <div className="bg-research-card border border-research-border rounded p-2 text-xs font-mono shadow-xl">
                <div className="text-research-text">Vol: {d.vol.toFixed(1)}% · Ret: {d.ret.toFixed(1)}%</div>
                <div className="text-research-subtle">Sharpe: {d.sharpe.toFixed(2)}</div>
                {d.isOptimal && <div className="text-research-bull font-semibold">★ Optimal</div>}
              </div>
            )
          }}
        />
        <Scatter
          data={chartData.filter(d => !d.isOptimal)}
          fill="#3b82f6"
          fillOpacity={0.6}
          name="Portfolios"
        />
        {optimal && (
          <Scatter
            data={[{ ...optimal, ret: optimal.expectedReturn * 100, vol: optimal.volatility * 100 }]}
            fill="#22c55e"
            fillOpacity={1}
            name="Optimal"
          />
        )}
      </ScatterChart>
    </ResponsiveContainer>
  )
}

// ── Risk contribution chart ────────────────────────────��──────────────────────

function RiskContribChart({ data }: { data: Array<{ instrument: string; pctContribution: number }> }) {
  const sorted = [...data].sort((a, b) => b.pctContribution - a.pctContribution)

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={sorted} layout="vertical" margin={{ top: 4, right: 8, bottom: 0, left: 80 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" horizontal={false} />
        <XAxis
          type="number"
          tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={v => `${(v * 100).toFixed(0)}%`}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          dataKey="instrument"
          type="category"
          tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickLine={false}
          axisLine={false}
          width={75}
          tickFormatter={v => v.replace('-USD', '')}
        />
        <Tooltip
          formatter={(v: number) => [formatPct(v * 100), 'Risk Contribution']}
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #1e2a3a', borderRadius: '6px', fontSize: '11px' }}
        />
        <Bar dataKey="pctContribution" radius={[0, 3, 3, 0]}>
          {sorted.map((_, i) => (
            <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── HRP dendrogram (ASCII SVG) ────────────────────────────────────────────────

function HRPDendrogram({ instruments }: { instruments: string[] }) {
  const n = instruments.length
  let seed = 44444
  function rand() { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff }

  // Simple mock: assign random cluster heights
  const clusters = instruments.map((inst, i) => ({
    label: inst.replace('-USD', ''),
    x: (i + 0.5) / n,
    height: rand(),
  }))

  const width = 420, height = 180, padL = 4, padR = 4, padT = 10, padB = 40

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ maxHeight: 180 }}>
      {clusters.map((c, i) => {
        const x = padL + c.x * (width - padL - padR)
        const y = padT + (1 - c.height * 0.7) * (height - padT - padB)
        const nextC = clusters[(i + 1) % n]
        const nx = padL + nextC.x * (width - padL - padR)
        const ny = padT + (1 - nextC.height * 0.7) * (height - padT - padB)
        return (
          <g key={i}>
            <line
              x1={x} y1={height - padB}
              x2={x} y2={y}
              stroke="#3b82f6"
              strokeWidth={1}
              strokeOpacity={0.6}
            />
            {i < n - 1 && (
              <line
                x1={x} y1={Math.min(y, ny)}
                x2={nx} y2={Math.min(y, ny)}
                stroke="#3b82f6"
                strokeWidth={0.8}
                strokeOpacity={0.4}
              />
            )}
            <text
              x={x}
              y={height - padB + 12}
              textAnchor="middle"
              fontSize={8}
              fill="#8899aa"
              fontFamily="JetBrains Mono, monospace"
              transform={`rotate(-30, ${x}, ${height - padB + 12})`}
            >
              {c.label}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

// ── Main page ──────────────────────────────────────────────────────────���──────

export function PortfolioLabPage() {
  const weightsQ = useQuery({ queryKey: ['portfolio-weights'], queryFn: fetchPortfolioWeights })
  const frontierQ = useQuery({ queryKey: ['efficient-frontier'], queryFn: fetchEfficientFrontier })
  const riskQ = useQuery({ queryKey: ['risk-contribution'], queryFn: fetchRiskContribution })

  const corrInstruments = INSTRUMENTS.slice(0, 8)
  const corrMatrix = useMemo(() => buildCorrelationMatrix(corrInstruments), [])

  const weights = weightsQ.data ?? []
  const optimal = frontierQ.data?.find(p => p.isOptimal)

  return (
    <div className="space-y-4 animate-fade-in">
      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="Total Portfolio Value"
          value={weights.length > 0 ? formatCurrency(weights.reduce((a, b) => a + b.value, 0), { compact: true }) : '–'}
          variant="info"
          loading={weightsQ.isLoading}
        />
        <MetricCard
          label="Optimal Sharpe"
          value={optimal ? formatRatio(optimal.sharpe) : '–'}
          variant={optimal && optimal.sharpe > 1 ? 'bull' : 'bear'}
          loading={frontierQ.isLoading}
        />
        <MetricCard
          label="Effective N"
          value={weights.length > 0
            ? (1 / weights.reduce((a, b) => a + b.weight ** 2, 0)).toFixed(1)
            : '–'}
          variant="info"
          loading={weightsQ.isLoading}
        />
        <MetricCard
          label="Largest Weight"
          value={weights.length > 0
            ? formatPct(Math.max(...weights.map(w => w.weight)) * 100)
            : '–'}
          loading={weightsQ.isLoading}
        />
      </div>

      {/* Weights pie + Efficient frontier */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Current Portfolio Weights</h2>
          {weightsQ.isLoading ? <LoadingSpinner fullHeight /> :
            <WeightsPieChart weights={weights} />}
        </div>

        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Efficient Frontier</h2>
          {frontierQ.isLoading ? <LoadingSpinner fullHeight /> :
            <EfficientFrontierPlot data={frontierQ.data ?? []} />}
        </div>
      </div>

      {/* Risk contribution + Correlation heatmap */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Risk Contribution by Instrument</h2>
          {riskQ.isLoading ? <LoadingSpinner size="sm" /> :
            <RiskContribChart data={riskQ.data ?? []} />}
        </div>

        <div className="bg-research-card border border-research-border rounded-lg p-4">
          <h2 className="text-sm font-semibold text-research-text mb-3">Rolling Correlation Heatmap</h2>
          <CorrelationHeatmap instruments={corrInstruments} matrix={corrMatrix} size="sm" />
        </div>
      </div>

      {/* HRP Dendrogram */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <h2 className="text-sm font-semibold text-research-text mb-3">HRP Dendrogram</h2>
        <HRPDendrogram instruments={corrInstruments} />
      </div>

      {/* Weights breakdown table */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <h2 className="text-sm font-semibold text-research-text mb-3">Portfolio Weights Detail</h2>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="border-b border-research-border">
                {['Instrument', 'Weight', 'Target', 'Drift', 'Value', 'P&L Contrib', 'Risk Contrib'].map(h => (
                  <th key={h} className="text-left text-[10px] text-research-subtle uppercase tracking-wide py-2 px-3 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {weights.map((w, i) => (
                <tr key={w.instrument} className="hover:bg-research-muted/20 border-b border-research-border/50">
                  <td className="py-1.5 px-3 font-mono text-research-accent font-semibold">
                    <span className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: CHART_COLORS[i % CHART_COLORS.length] }} />
                      {w.instrument}
                    </span>
                  </td>
                  <td className="py-1.5 px-3 font-mono text-research-text">{formatPct(w.weight * 100)}</td>
                  <td className="py-1.5 px-3 font-mono text-research-subtle">{formatPct(w.targetWeight * 100)}</td>
                  <td className={clsx('py-1.5 px-3 font-mono', Math.abs(w.drift) > 0.02 ? 'text-research-warning' : 'text-research-subtle')}>
                    {formatPct(w.drift * 100, { sign: true })}
                  </td>
                  <td className="py-1.5 px-3 font-mono text-research-text">{formatCurrency(w.value)}</td>
                  <td className={clsx('py-1.5 px-3 font-mono', w.pnlContribution >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                    {formatCurrency(w.pnlContribution, { sign: true })}
                  </td>
                  <td className="py-1.5 px-3 font-mono text-research-subtle">{formatPct(w.riskContribution * 100)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
