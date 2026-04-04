// ============================================================
// EquityCurve.tsx — Equity curve chart with drawdown overlay
// ============================================================
import React, { useMemo } from 'react'
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { format, parseISO } from 'date-fns'
import type { EquityPoint } from '@/types'

interface EquityCurveProps {
  data: EquityPoint[]
  showDrawdown?: boolean
  height?: number
  className?: string
}

const CustomTooltip: React.FC<{
  active?: boolean
  payload?: { value: number; dataKey: string }[]
  label?: string
}> = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const equity = payload.find((p) => p.dataKey === 'equity')?.value ?? 0
  const drawdown = payload.find((p) => p.dataKey === 'drawdown')?.value ?? 0
  const dailyPnl = payload.find((p) => p.dataKey === 'dailyPnl')?.value ?? 0

  return (
    <div className="bg-[#111318] border border-[#1e2130] rounded-lg p-2.5 text-[10px] font-mono">
      <div className="text-slate-500 mb-1.5">{label}</div>
      <div className="flex flex-col gap-1">
        <div className="flex justify-between gap-4">
          <span className="text-slate-500">Equity</span>
          <span className="text-slate-200">${equity.toLocaleString('en-US', { maximumFractionDigits: 0 })}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-slate-500">Daily P&L</span>
          <span className={dailyPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}>
            {dailyPnl >= 0 ? '+' : ''}${dailyPnl.toFixed(0)}
          </span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-slate-500">Drawdown</span>
          <span className="text-red-400">{(drawdown * 100).toFixed(2)}%</span>
        </div>
      </div>
    </div>
  )
}

export const EquityCurve: React.FC<EquityCurveProps> = ({
  data,
  showDrawdown = true,
  height = 240,
  className,
}) => {
  const formatted = useMemo(
    () =>
      data.map((p) => ({
        ...p,
        dateLabel: format(parseISO(p.timestamp), 'MMM d'),
        drawdown: p.drawdown * 100,  // to percent
      })),
    [data],
  )

  if (!data.length) {
    return (
      <div className="flex items-center justify-center text-slate-600 text-xs font-mono" style={{ height }}>
        No equity data
      </div>
    )
  }

  return (
    <div className={className} style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={formatted} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#1e2130" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="dateLabel"
            tick={{ fill: '#475569', fontSize: 9, fontFamily: 'JetBrains Mono' }}
            axisLine={false}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="equity"
            orientation="left"
            tick={{ fill: '#475569', fontSize: 9, fontFamily: 'JetBrains Mono' }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
            width={40}
          />
          {showDrawdown && (
            <YAxis
              yAxisId="dd"
              orientation="right"
              tick={{ fill: '#475569', fontSize: 9, fontFamily: 'JetBrains Mono' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `${v.toFixed(0)}%`}
              width={36}
              domain={['auto', 0]}
            />
          )}
          <Tooltip content={<CustomTooltip />} />
          {showDrawdown && (
            <Bar
              yAxisId="dd"
              dataKey="drawdown"
              fill="rgba(239,68,68,0.2)"
              stroke="rgba(239,68,68,0.4)"
              strokeWidth={0}
            />
          )}
          <Line
            yAxisId="equity"
            type="monotone"
            dataKey="equity"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3, fill: '#3b82f6' }}
          />
          <ReferenceLine yAxisId="dd" y={0} stroke="#1e2130" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
