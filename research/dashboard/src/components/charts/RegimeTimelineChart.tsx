import React, { useMemo } from 'react'
import { ResponsiveContainer, ComposedChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts'
import type { RegimeSegment } from '@/types/regimes'
import { REGIME_COLORS, REGIME_LABELS } from '@/utils/colors'
import type { RegimeType } from '@/types/trades'

interface RegimeTimelineChartProps {
  segments: RegimeSegment[]
  height?: number
  showPrice?: boolean
}

export function RegimeTimelineChart({ segments, height = 80 }: RegimeTimelineChartProps) {
  const data = segments.map(s => ({
    label: s.startDate.slice(5),
    duration: s.durationDays,
    regime: s.regime,
    startDate: s.startDate,
    endDate: s.endDate,
    priceReturn: s.priceReturn,
  }))

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 8 }} barCategoryGap={1}>
          <XAxis
            dataKey="label"
            tick={{ fontSize: 9, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
            tickLine={false}
            axisLine={false}
            interval={Math.floor(data.length / 8)}
          />
          <YAxis hide />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const d = payload[0].payload as typeof data[0]
              const regime = d.regime as RegimeType
              return (
                <div className="bg-research-card border border-research-border rounded p-2 text-xs shadow-xl">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: REGIME_COLORS[regime] }} />
                    <span className="font-medium text-research-text">{REGIME_LABELS[regime]}</span>
                  </div>
                  <div className="text-research-subtle font-mono">
                    {d.startDate} → {d.endDate}
                  </div>
                  <div className="text-research-subtle font-mono">
                    {d.duration}d | {(d.priceReturn * 100).toFixed(1)}%
                  </div>
                </div>
              )
            }}
          />
          <Bar dataKey="duration" radius={[2, 2, 0, 0]} maxBarSize={40}>
            {data.map((entry, i) => (
              <Cell key={i} fill={REGIME_COLORS[entry.regime as RegimeType]} fillOpacity={0.8} />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Regime legend ─────────────────────────────────────────────────────────────

export function RegimeLegend() {
  const regimes: RegimeType[] = ['bull', 'bear', 'sideways', 'ranging', 'volatile']
  return (
    <div className="flex gap-3 flex-wrap">
      {regimes.map(r => (
        <div key={r} className="flex items-center gap-1.5 text-xs text-research-subtle">
          <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: REGIME_COLORS[r] }} />
          {REGIME_LABELS[r]}
        </div>
      ))}
    </div>
  )
}
