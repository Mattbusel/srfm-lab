import React from 'react'
import {
  ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts'
import type { MCBands } from '@/types/mc'
import { formatCurrency } from '@/utils/formatters'

interface MCFanChartProps {
  bands: MCBands[]
  initialEquity: number
  height?: number
  maxPoints?: number
}

export function MCFanChart({ bands, initialEquity, height = 300, maxPoints = 100 }: MCFanChartProps) {
  // Downsample to maxPoints for rendering performance
  const step = Math.max(1, Math.floor(bands.length / maxPoints))
  const data = bands.filter((_, i) => i % step === 0).map(b => ({
    ...b,
    p5_p95: [b.p5, b.p95],
    p25_p75: [b.p25, b.p75],
    label: b.date.slice(5),
  }))

  const tickFmt = (v: number) => formatCurrency(v, { compact: true })

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <defs>
          <linearGradient id="outerBand" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.08} />
            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
          </linearGradient>
          <linearGradient id="innerBand" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={tickFmt}
          tickLine={false}
          axisLine={false}
          width={72}
        />
        <Tooltip
          formatter={(v: number) => [formatCurrency(v), '']}
          contentStyle={{
            backgroundColor: '#111827',
            border: '1px solid #1e2a3a',
            borderRadius: '6px',
            fontSize: '11px',
            fontFamily: 'JetBrains Mono, monospace',
          }}
        />
        {/* P5–P95 outer band */}
        <Area type="monotone" dataKey="p95" stroke="none" fill="url(#outerBand)" name="P95" dot={false} />
        <Area type="monotone" dataKey="p5" stroke="none" fill="#080b12" name="P5" dot={false} />
        {/* P25–P75 inner band */}
        <Area type="monotone" dataKey="p75" stroke="none" fill="url(#innerBand)" name="P75" dot={false} />
        <Area type="monotone" dataKey="p25" stroke="none" fill="#080b12" name="P25" dot={false} />
        {/* Median */}
        <Line type="monotone" dataKey="p50" stroke="#3b82f6" strokeWidth={2} dot={false} name="Median" />
        {/* Mean */}
        <Line type="monotone" dataKey="mean" stroke="#f59e0b" strokeWidth={1} dot={false} strokeDasharray="4 4" name="Mean" />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
