import React from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts'
import type { ICPoint } from '@/types/signals'
import { IC_LINE_COLOR, IC_BAND_COLOR } from '@/utils/colors'

interface ICDecayChartProps {
  data: ICPoint[]
  height?: number
}

export function ICDecayChart({ data, height = 220 }: ICDecayChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <defs>
          <linearGradient id="icBandGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22c55e" stopOpacity={0.2} />
            <stop offset="95%" stopColor="#22c55e" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis
          dataKey="lag"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          label={{ value: 'Lag (days)', position: 'insideBottom', offset: -2, fontSize: 10, fill: '#8899aa' }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={v => v.toFixed(3)}
          tickLine={false}
          axisLine={false}
          width={55}
        />
        <Tooltip
          formatter={(v: number, name: string) => [v.toFixed(4), name]}
          contentStyle={{
            backgroundColor: '#111827',
            border: '1px solid #1e2a3a',
            borderRadius: '6px',
            fontSize: '11px',
            fontFamily: 'JetBrains Mono, monospace',
          }}
        />
        <ReferenceLine y={0} stroke="#2d3a4f" />
        {/* Confidence band */}
        <Area
          type="monotone"
          dataKey="icHigh"
          fill={IC_BAND_COLOR}
          stroke="none"
          activeDot={false}
          name="95% CI High"
        />
        <Area
          type="monotone"
          dataKey="icLow"
          fill={IC_BAND_COLOR}
          stroke="none"
          activeDot={false}
          name="95% CI Low"
        />
        <Line
          type="monotone"
          dataKey="ic"
          stroke={IC_LINE_COLOR}
          strokeWidth={2}
          dot={{ r: 3, fill: IC_LINE_COLOR }}
          activeDot={{ r: 5 }}
          name="IC"
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
