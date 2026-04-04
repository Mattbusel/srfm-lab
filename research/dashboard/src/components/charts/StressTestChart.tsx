import React from 'react'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine,
} from 'recharts'
import type { StressScenario } from '@/types/regimes'
import { formatCurrency, formatPct } from '@/utils/formatters'

interface StressTestChartProps {
  scenarios: StressScenario[]
  height?: number
}

export function StressTestChart({ scenarios, height = 260 }: StressTestChartProps) {
  const sorted = [...scenarios].sort((a, b) => a.pnlImpact - b.pnlImpact)
  const data = sorted.map(s => ({
    name: s.name,
    impact: s.pnlImpact,
    impactPct: s.pnlImpactPct,
    maxDD: s.maxDrawdown * 100,
    recovery: s.recoveryDays,
  }))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 8, bottom: 0, left: 100 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" horizontal={false} />
        <XAxis
          type="number"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={v => `$${(v / 1000).toFixed(0)}K`}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          dataKey="name"
          type="category"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickLine={false}
          axisLine={false}
          width={95}
        />
        <Tooltip
          formatter={(v: number) => [formatCurrency(v), 'P&L Impact']}
          contentStyle={{
            backgroundColor: '#111827',
            border: '1px solid #1e2a3a',
            borderRadius: '6px',
            fontSize: '11px',
            fontFamily: 'JetBrains Mono, monospace',
          }}
        />
        <ReferenceLine x={0} stroke="#2d3a4f" />
        <Bar dataKey="impact" radius={[0, 3, 3, 0]} name="P&L Impact">
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.impact >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
