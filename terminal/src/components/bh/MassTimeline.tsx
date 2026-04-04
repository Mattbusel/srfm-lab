// ============================================================
// MassTimeline — multi-instrument BH mass timeline
// ============================================================
import React, { useMemo, useState } from 'react'
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceArea, ReferenceLine, Brush,
} from 'recharts'
import { format } from 'date-fns'
import { useBHStore } from '@/store/bhStore'

const SYMBOL_COLORS = ['#7c3aed', '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#84cc16']

interface MassTimelineProps {
  symbols?: string[]
  height?: number
  showFormationLines?: boolean
  className?: string
}

export const MassTimeline: React.FC<MassTimelineProps> = ({
  symbols,
  height = 300,
  showFormationLines = true,
  className = '',
}) => {
  const allInstruments = useBHStore((s) => s.instruments)
  const allHistory = useBHStore((s) => s.history)
  const formationEvents = useBHStore((s) => s.formationEvents)

  const targetSymbols = symbols ?? Object.keys(allInstruments).slice(0, 4)
  const [selectedTF, setSelectedTF] = useState<'15m' | '1h' | '1d'>('1h')
  const [brushRange, setBrushRange] = useState<[number, number] | null>(null)

  // Merge history points by timestamp
  const chartData = useMemo(() => {
    // Collect all timestamps across all symbols
    const allTimestamps = new Set<number>()
    for (const sym of targetSymbols) {
      const hist = allHistory[sym]
      if (hist) {
        for (const p of hist.points) {
          allTimestamps.add(p.timestamp)
        }
      }
    }

    // Build rows
    const sortedTimestamps = Array.from(allTimestamps).sort((a, b) => a - b)
    const rows: Record<string, number>[] = []

    for (const ts of sortedTimestamps.slice(-200)) {
      const row: Record<string, number> = { timestamp: ts }
      for (const sym of targetSymbols) {
        const hist = allHistory[sym]
        const point = hist?.points.find((p) => Math.abs(p.timestamp - ts) < 120000)  // within 2 min
        if (point) {
          row[`${sym}_${selectedTF}`] = selectedTF === '15m' ? point.mass15m : selectedTF === '1h' ? point.mass1h : point.mass1d
        }
      }
      rows.push(row)
    }

    return rows
  }, [targetSymbols, allHistory, selectedTF])

  // Formation events within time range
  const visibleFormations = useMemo(() => {
    if (!showFormationLines) return []
    return formationEvents.filter((e) => {
      if (!targetSymbols.includes(e.symbol)) return false
      if (e.timeframe !== selectedTF) return false
      return true
    }).slice(0, 20)
  }, [formationEvents, targetSymbols, selectedTF, showFormationLines])

  const formatTs = (ts: number) => format(new Date(ts), 'MM/dd HH:mm')

  return (
    <div className={`flex flex-col bg-terminal-bg ${className}`}>
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border flex-shrink-0">
        <span className="text-terminal-subtle text-xs font-mono uppercase">BH Mass Timeline</span>
        <div className="flex gap-1">
          {(['15m', '1h', '1d'] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => setSelectedTF(tf)}
              className={`text-[10px] font-mono px-2 py-0.5 rounded transition-colors ${
                selectedTF === tf ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 30 }}>
            <CartesianGrid stroke="#1f2937" strokeDasharray="2 4" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTs}
              tick={{ fill: '#9ca3af', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              stroke="#1f2937"
              minTickGap={80}
            />
            <YAxis
              domain={[0, 3]}
              tick={{ fill: '#9ca3af', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
              stroke="#1f2937"
              tickLine={false}
              width={28}
            />
            <Tooltip
              formatter={(v: number, name: string) => [v.toFixed(3), name.split('_')[0]]}
              labelFormatter={(l) => formatTs(l as number)}
              contentStyle={{ backgroundColor: '#111827', border: '1px solid #1f2937', borderRadius: 4, fontSize: 10 }}
            />
            <Legend
              wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
              formatter={(value) => value.split('_')[0]}
            />

            {/* Threshold reference line */}
            <ReferenceLine y={1.2} stroke="#f59e0b" strokeDasharray="4 4" strokeOpacity={0.6} />

            {/* Formation event vertical lines */}
            {visibleFormations.map((evt) => (
              <ReferenceLine
                key={evt.id}
                x={evt.timestamp}
                stroke={evt.dir === 1 ? '#22c55e' : '#ef4444'}
                strokeOpacity={0.5}
                strokeDasharray="2 2"
              />
            ))}

            {/* Mass lines per symbol */}
            {targetSymbols.map((sym, i) => (
              <Area
                key={sym}
                type="monotone"
                dataKey={`${sym}_${selectedTF}`}
                name={`${sym}_${selectedTF}`}
                stroke={SYMBOL_COLORS[i % SYMBOL_COLORS.length]}
                fill={`${SYMBOL_COLORS[i % SYMBOL_COLORS.length]}15`}
                strokeWidth={1.5}
                dot={false}
                connectNulls
              />
            ))}

            {chartData.length > 50 && (
              <Brush
                dataKey="timestamp"
                height={20}
                stroke="#374151"
                fill="#111827"
                tickFormatter={formatTs}
                startIndex={Math.max(0, chartData.length - 100)}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default MassTimeline
