import React from 'react'
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts'
import type { EquityPoint } from '@/types/trades'
import { formatPct } from '@/utils/formatters'

interface DrawdownChartProps {
  data: EquityPoint[]
  height?: number
}

export function DrawdownChart({ data, height = 140 }: DrawdownChartProps) {
  const chartData = data.map(d => ({
    ...d,
    drawdownNeg: -(d.drawdown * 100),
  }))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <defs>
          <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
            <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis
          dataKey="timestamp"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={v => v.slice(5)}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={v => `${v.toFixed(1)}%`}
          tickLine={false}
          axisLine={false}
          domain={['auto', 0]}
          width={50}
        />
        <Tooltip
          formatter={(v: number) => [`${v.toFixed(2)}%`, 'Drawdown']}
          labelFormatter={l => l}
          contentStyle={{
            backgroundColor: '#111827',
            border: '1px solid #1e2a3a',
            borderRadius: '6px',
            fontSize: '11px',
            fontFamily: 'JetBrains Mono, monospace',
          }}
        />
        <Area
          type="monotone"
          dataKey="drawdownNeg"
          stroke="#ef4444"
          strokeWidth={1.5}
          fill="url(#ddGradient)"
          dot={false}
          name="Drawdown"
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
