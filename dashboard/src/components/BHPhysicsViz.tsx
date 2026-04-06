// ============================================================
// BHPhysicsViz.tsx — BH mass accumulation visualization
// 3D-style bar chart of per-symbol BH mass, time series, spacetime coloring
// ============================================================

import React, { useMemo, useState, useCallback } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
  LineChart,
  Line,
  Legend,
  ComposedChart,
  Area,
} from 'recharts'
import { clsx } from 'clsx'
import type { BHSignal, BHMassHistory, SpacetimeType } from '@/types/metrics'

// ---- Color maps -----------------------------------------------------

const SPACETIME_COLORS: Record<SpacetimeType, string> = {
  TIMELIKE:  '#3b82f6',   // blue  — signal propagating forward in time
  SPACELIKE: '#f59e0b',   // amber — spatially correlated
  LIGHTLIKE: '#a78bfa',   // purple — boundary / critical
  NONE:      '#4b5563',   // gray
}

const STRENGTH_COLORS = {
  strong_long:  '#16a34a',
  weak_long:    '#86efac',
  neutral:      '#6b7280',
  weak_short:   '#fca5a5',
  strong_short: '#dc2626',
}

const TIMEFRAME_COLORS: Record<string, string> = {
  '15m': '#60a5fa',
  '1h':  '#34d399',
  '4h':  '#f472b6',
}

// ---- Helpers ---------------------------------------------------------

function signalColor(signal: BHSignal): string {
  return SPACETIME_COLORS[signal.spacetimeType]
}

function massBarColor(mass: number, threshold: number): string {
  const ratio = mass / Math.max(threshold, 0.001)
  if (ratio >= 1.5) return '#dc2626'   // above 1.5× threshold → red (overextended)
  if (ratio >= 1.0) return '#16a34a'   // above threshold → green (active)
  if (ratio >= 0.7) return '#f59e0b'   // near threshold → amber
  return '#4b5563'                     // below → gray
}

// ---- Custom tooltip --------------------------------------------------

interface MassTooltipProps {
  active?: boolean
  payload?: Array<{ payload: BHSignal; value: number; name: string }>
  label?: string
}

const MassTooltip: React.FC<MassTooltipProps> = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const sig = payload[0].payload

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs shadow-xl">
      <div className="font-semibold text-white mb-1">{sig.symbol} / {sig.timeframe}</div>
      <div className="text-gray-400">BH Mass: <span className="text-white">{sig.bhMass.toFixed(4)}</span></div>
      <div className="text-gray-400">Threshold: <span className="text-white">{sig.massThreshold.toFixed(4)}</span></div>
      <div className="text-gray-400">Confidence: <span className="text-white">{(sig.confidence * 100).toFixed(1)}%</span></div>
      <div className="mt-1 flex items-center gap-1">
        <span
          className="inline-block w-2 h-2 rounded-full"
          style={{ background: SPACETIME_COLORS[sig.spacetimeType] }}
        />
        <span className="text-gray-300">{sig.spacetimeType}</span>
      </div>
    </div>
  )
}

// ---- Mass Bar Chart (per symbol × timeframe) ------------------------

interface MassBarChartProps {
  signals: BHSignal[]
  onSymbolClick?: (symbol: string) => void
  selectedSymbol?: string
}

const MassBarChart: React.FC<MassBarChartProps> = ({ signals, onSymbolClick, selectedSymbol }) => {
  // Aggregate by symbol: take max mass across timeframes for primary bar
  const bySymbol = useMemo(() => {
    const map = new Map<string, BHSignal[]>()
    for (const sig of signals) {
      if (!map.has(sig.symbol)) map.set(sig.symbol, [])
      map.get(sig.symbol)!.push(sig)
    }
    return Array.from(map.entries()).map(([sym, sigs]) => {
      const dominant = sigs.reduce((a, b) => a.bhMass > b.bhMass ? a : b)
      const mass_15m = sigs.find(s => s.timeframe === '15m')?.bhMass ?? 0
      const mass_1h  = sigs.find(s => s.timeframe === '1h')?.bhMass ?? 0
      const mass_4h  = sigs.find(s => s.timeframe === '4h')?.bhMass ?? 0
      return {
        symbol: sym,
        mass_15m,
        mass_1h,
        mass_4h,
        threshold: dominant.massThreshold,
        spacetimeType: dominant.spacetimeType,
        dominant,
      }
    }).sort((a, b) => (b.mass_15m + b.mass_1h + b.mass_4h) - (a.mass_15m + a.mass_1h + a.mass_4h))
  }, [signals])

  const maxThreshold = useMemo(
    () => Math.max(...bySymbol.map(d => d.threshold), 0.001),
    [bySymbol]
  )

  if (!bySymbol.length) {
    return (
      <div className="flex items-center justify-center h-40 text-gray-500 text-sm">
        No BH signal data
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart
        data={bySymbol}
        barSize={14}
        barCategoryGap="15%"
        onClick={data => {
          if (data?.activePayload?.[0]) {
            onSymbolClick?.(data.activePayload[0].payload.symbol)
          }
        }}
        style={{ cursor: 'pointer' }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
        <XAxis
          dataKey="symbol"
          tick={{ fill: '#9ca3af', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#9ca3af', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={40}
        />
        <Tooltip content={<MassTooltip />} />
        <ReferenceLine
          y={maxThreshold}
          stroke="#f59e0b"
          strokeDasharray="4 2"
          label={{ value: 'Threshold', fill: '#f59e0b', fontSize: 10, position: 'insideTopRight' }}
        />
        <Legend
          wrapperStyle={{ fontSize: 11, color: '#9ca3af' }}
        />
        <Bar dataKey="mass_15m" name="15m" fill={TIMEFRAME_COLORS['15m']} radius={[2, 2, 0, 0]}>
          {bySymbol.map(entry => (
            <Cell
              key={entry.symbol}
              fill={entry.symbol === selectedSymbol ? '#93c5fd' : TIMEFRAME_COLORS['15m']}
              opacity={selectedSymbol && entry.symbol !== selectedSymbol ? 0.4 : 1}
            />
          ))}
        </Bar>
        <Bar dataKey="mass_1h" name="1h" fill={TIMEFRAME_COLORS['1h']} radius={[2, 2, 0, 0]}>
          {bySymbol.map(entry => (
            <Cell
              key={entry.symbol}
              fill={entry.symbol === selectedSymbol ? '#6ee7b7' : TIMEFRAME_COLORS['1h']}
              opacity={selectedSymbol && entry.symbol !== selectedSymbol ? 0.4 : 1}
            />
          ))}
        </Bar>
        <Bar dataKey="mass_4h" name="4h" fill={TIMEFRAME_COLORS['4h']} radius={[2, 2, 0, 0]}>
          {bySymbol.map(entry => (
            <Cell
              key={entry.symbol}
              fill={entry.symbol === selectedSymbol ? '#f9a8d4' : TIMEFRAME_COLORS['4h']}
              opacity={selectedSymbol && entry.symbol !== selectedSymbol ? 0.4 : 1}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ---- Mass time series for selected symbol ---------------------------

interface MassTimeSeriesProps {
  history: BHMassHistory | null
  symbol: string
}

const MassTimeSeries: React.FC<MassTimeSeriesProps> = ({ history, symbol }) => {
  if (!history || !history.timestamps.length) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-500 text-sm">
        Select a symbol to view mass history
      </div>
    )
  }

  const data = history.timestamps.map((ts, i) => ({
    time: ts.slice(11, 16),    // HH:MM
    mass: history.massValues[i],
    threshold: history.threshold,
  }))

  return (
    <ResponsiveContainer width="100%" height={140}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="time" tick={{ fill: '#9ca3af', fontSize: 9 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={{ fill: '#9ca3af', fontSize: 9 }} axisLine={false} tickLine={false} width={35} />
        <Tooltip
          contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 6, fontSize: 11 }}
          labelStyle={{ color: '#d1d5db' }}
        />
        <Area
          type="monotone"
          dataKey="mass"
          fill="#1d4ed820"
          stroke="#3b82f6"
          strokeWidth={1.5}
          dot={false}
          name="BH Mass"
        />
        <Line
          type="monotone"
          dataKey="threshold"
          stroke="#f59e0b"
          strokeWidth={1}
          strokeDasharray="4 2"
          dot={false}
          name="Threshold"
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

// ---- Spacetime type legend badge ------------------------------------

const SpacetimeBadge: React.FC<{ type: SpacetimeType; count: number }> = ({ type, count }) => (
  <div className="flex items-center gap-1.5">
    <span
      className="inline-block w-3 h-3 rounded-sm"
      style={{ background: SPACETIME_COLORS[type] }}
    />
    <span className="text-xs text-gray-400">{type}</span>
    <span className="text-xs text-gray-600">×{count}</span>
  </div>
)

// ---- Signal grid row (symbol × timeframe matrix) --------------------

interface SignalGridProps {
  signals: BHSignal[]
  onSignalClick?: (signal: BHSignal) => void
}

const TIMEFRAME_ORDER = ['15m', '1h', '4h'] as const

const SignalGrid: React.FC<SignalGridProps> = ({ signals, onSignalClick }) => {
  const symbolSet = useMemo(
    () => [...new Set(signals.map(s => s.symbol))].sort(),
    [signals]
  )

  const signalMap = useMemo(() => {
    const map = new Map<string, BHSignal>()
    for (const sig of signals) {
      map.set(`${sig.symbol}:${sig.timeframe}`, sig)
    }
    return map
  }, [signals])

  if (!symbolSet.length) return null

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr>
            <th className="text-left text-gray-500 font-medium py-1 pr-3 w-20">Symbol</th>
            {TIMEFRAME_ORDER.map(tf => (
              <th key={tf} className="text-center text-gray-500 font-medium py-1 px-2" style={{ color: TIMEFRAME_COLORS[tf] }}>
                {tf}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {symbolSet.map(sym => (
            <tr key={sym} className="border-t border-gray-800/50 hover:bg-gray-800/30">
              <td className="py-1 pr-3 font-mono text-gray-300 font-medium">{sym}</td>
              {TIMEFRAME_ORDER.map(tf => {
                const sig = signalMap.get(`${sym}:${tf}`)
                if (!sig) {
                  return (
                    <td key={tf} className="py-1 px-2 text-center text-gray-700">–</td>
                  )
                }
                const bg = SPACETIME_COLORS[sig.spacetimeType] + '22'
                const border = SPACETIME_COLORS[sig.spacetimeType] + '66'
                return (
                  <td
                    key={tf}
                    className="py-1 px-2 text-center cursor-pointer"
                    onClick={() => onSignalClick?.(sig)}
                  >
                    <div
                      className="inline-flex flex-col items-center rounded px-1.5 py-0.5 gap-0.5"
                      style={{ background: bg, border: `1px solid ${border}` }}
                      title={`${sym} ${tf}: mass=${sig.bhMass.toFixed(3)} conf=${(sig.confidence*100).toFixed(0)}%`}
                    >
                      <span
                        className="font-bold text-xs"
                        style={{ color: STRENGTH_COLORS[sig.strength] }}
                      >
                        {sig.strengthValue >= 0 ? '▲' : '▼'}
                        {Math.abs(sig.strengthValue * 100).toFixed(0)}
                      </span>
                      <span className="text-gray-500" style={{ fontSize: 9 }}>
                        {sig.bhMass.toFixed(2)}
                      </span>
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---- Main BHPhysicsViz component ------------------------------------

interface BHPhysicsVizProps {
  signals: BHSignal[]
  massHistory?: Record<string, BHMassHistory>  // symbol → history
  className?: string
}

export const BHPhysicsViz: React.FC<BHPhysicsVizProps> = ({
  signals,
  massHistory = {},
  className,
}) => {
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [view, setView] = useState<'bars' | 'grid'>('bars')

  const selectedHistory = selectedSymbol ? massHistory[selectedSymbol] ?? null : null

  // Count spacetime types
  const typeCounts = useMemo(() => {
    const counts: Record<SpacetimeType, number> = { TIMELIKE: 0, SPACELIKE: 0, LIGHTLIKE: 0, NONE: 0 }
    for (const sig of signals) counts[sig.spacetimeType]++
    return counts
  }, [signals])

  const activeSignals = useMemo(
    () => signals.filter(s => s.spacetimeType !== 'NONE' && Math.abs(s.strengthValue) > 0.1),
    [signals]
  )

  const handleBarClick = useCallback((sym: string) => {
    setSelectedSymbol(prev => prev === sym ? null : sym)
  }, [])

  return (
    <div className={clsx('flex flex-col gap-4 bg-gray-900 rounded-xl border border-gray-800 p-4', className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex flex-col gap-0.5">
          <h3 className="text-sm font-semibold text-gray-200">BH Mass Accumulation</h3>
          <span className="text-xs text-gray-500">{activeSignals.length} active signals</span>
        </div>
        <div className="flex gap-2">
          {(['bars', 'grid'] as const).map(v => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={clsx(
                'text-xs px-3 py-1 rounded-lg font-medium transition-colors',
                view === v
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-gray-200 bg-gray-800'
              )}
            >
              {v === 'bars' ? 'Mass Chart' : 'Signal Grid'}
            </button>
          ))}
        </div>
      </div>

      {/* Spacetime legend */}
      <div className="flex flex-wrap gap-3">
        {(Object.entries(typeCounts) as Array<[SpacetimeType, number]>)
          .filter(([, n]) => n > 0)
          .map(([type, count]) => (
            <SpacetimeBadge key={type} type={type} count={count} />
          ))}
      </div>

      {/* Chart / grid */}
      {view === 'bars' ? (
        <>
          <MassBarChart
            signals={signals}
            onSymbolClick={handleBarClick}
            selectedSymbol={selectedSymbol ?? undefined}
          />
          {selectedSymbol && (
            <div className="border-t border-gray-800 pt-3">
              <div className="text-xs text-gray-400 mb-2">
                BH Mass history — <span className="text-blue-400 font-medium">{selectedSymbol}</span>
              </div>
              <MassTimeSeries history={selectedHistory} symbol={selectedSymbol} />
            </div>
          )}
        </>
      ) : (
        <SignalGrid
          signals={signals}
          onSignalClick={sig => setSelectedSymbol(prev => prev === sig.symbol ? null : sig.symbol)}
        />
      )}

      {/* Selected signal detail */}
      {selectedSymbol && view === 'grid' && (
        <div className="border-t border-gray-800 pt-3">
          <div className="text-xs text-gray-400 mb-2">
            Mass history — <span className="text-blue-400 font-medium">{selectedSymbol}</span>
          </div>
          <MassTimeSeries history={selectedHistory} symbol={selectedSymbol} />
        </div>
      )}
    </div>
  )
}
