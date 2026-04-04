import React from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts'
import type { EquityPoint } from '@/types/trades'
import { formatCurrency, formatDate } from '@/utils/formatters'
import { EQUITY_CURVE_COLOR, BENCHMARK_COLOR } from '@/utils/colors'

interface EquityCurveChartProps {
  data: EquityPoint[]
  showBenchmark?: boolean
  height?: number
}

function CustomTooltip({ active, payload, label }: {
  active?: boolean
  payload?: Array<{ name: string; value: number; color: string }>
  label?: string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-research-card border border-research-border rounded p-3 text-xs shadow-xl">
      <div className="text-research-subtle mb-2 font-mono">{label}</div>
      {payload.map(p => (
        <div key={p.name} className="flex items-center gap-2 mb-1">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
          <span className="text-research-subtle">{p.name}:</span>
          <span className="font-mono text-research-text font-medium">
            {formatCurrency(p.value, { compact: true })}
          </span>
        </div>
      ))}
    </div>
  )
}

export function EquityCurveChart({ data, showBenchmark = false, height = 260 }: EquityCurveChartProps) {
  const initial = data[0]?.equity ?? 1
  const tickFormatter = (v: number) => formatCurrency(v, { compact: true })

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <defs>
          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={EQUITY_CURVE_COLOR} stopOpacity={0.3} />
            <stop offset="95%" stopColor={EQUITY_CURVE_COLOR} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis
          dataKey="timestamp"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={v => v.slice(5)} // MM-DD
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={tickFormatter}
          tickLine={false}
          axisLine={false}
          width={70}
        />
        <Tooltip content={<CustomTooltip />} />
        {showBenchmark && data[0]?.benchmark !== undefined && (
          <Legend
            wrapperStyle={{ fontSize: '11px', color: '#8899aa' }}
          />
        )}
        <ReferenceLine y={initial} stroke="#2d3a4f" strokeDasharray="4 4" />
        <Area
          type="monotone"
          dataKey="equity"
          stroke={EQUITY_CURVE_COLOR}
          strokeWidth={2}
          fill="url(#equityGradient)"
          dot={false}
          name="Equity"
        />
        {showBenchmark && (
          <Line
            type="monotone"
            dataKey="benchmark"
            stroke={BENCHMARK_COLOR}
            strokeWidth={1}
            dot={false}
            strokeDasharray="4 4"
            name="Benchmark"
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
