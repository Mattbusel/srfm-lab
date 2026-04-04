import React from 'react'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine,
} from 'recharts'
import type { QuintileReturn } from '@/types/signals'
import { formatPct } from '@/utils/formatters'

interface QuintileBarChartProps {
  data: QuintileReturn[]
  height?: number
}

export function QuintileBarChart({ data, height = 200 }: QuintileBarChartProps) {
  const chartData = data.map(q => ({
    name: `Q${q.quintile}`,
    return: q.avgReturn * 100,
    sharpe: q.sharpe,
    winRate: q.winRate * 100,
    count: q.count,
  }))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ fontSize: 11, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={v => `${v.toFixed(1)}%`}
          tickLine={false}
          axisLine={false}
          width={48}
        />
        <Tooltip
          formatter={(v: number, name: string) => [
            name === 'return' ? `${v.toFixed(2)}%` : v.toFixed(2),
            name,
          ]}
          contentStyle={{
            backgroundColor: '#111827',
            border: '1px solid #1e2a3a',
            borderRadius: '6px',
            fontSize: '11px',
            fontFamily: 'JetBrains Mono, monospace',
          }}
        />
        <ReferenceLine y={0} stroke="#2d3a4f" />
        <Bar dataKey="return" radius={[3, 3, 0, 0]} name="Avg Return">
          {chartData.map((entry, i) => (
            <Cell key={i} fill={entry.return >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
