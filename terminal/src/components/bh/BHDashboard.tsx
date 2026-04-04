// ============================================================
// BHDashboard — full BH physics dashboard
// ============================================================
import React, { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AreaChart, Area, LineChart, Line, ResponsiveContainer, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import { useBHStore, selectBHGaugeData } from '@/store/bhStore'
import { format } from 'date-fns'
import type { InstrumentBHState, BHState, BHRegime } from '@/types'

// ---- BH Mass Gauge ----
interface GaugeProps {
  mass: number
  maxMass?: number
  active: boolean
  regime: BHRegime
  dir: 0 | 1 | -1
  ctl: number
  label: string
  formationActive: boolean
}

function BHGauge({ mass, maxMass = 3, active, regime, dir, ctl, label, formationActive }: GaugeProps) {
  const pct = Math.min(mass / maxMass, 1)
  const angle = pct * 270 - 135  // -135 to +135 degrees

  const color = !active ? '#6b7280'
    : regime === 'BULL' ? '#22c55e'
    : regime === 'BEAR' ? '#ef4444'
    : regime === 'HIGH_VOL' ? '#f59e0b'
    : '#6b7280'

  const dirSymbol = dir === 1 ? '↑' : dir === -1 ? '↓' : '→'
  const dirColor = dir === 1 ? '#22c55e' : dir === -1 ? '#ef4444' : '#9ca3af'

  const r = 28
  const cx = 36
  const cy = 36
  const startAngle = -225  // degrees
  const sweepAngle = 270
  const valueAngle = startAngle + sweepAngle * pct

  // SVG arc path
  const toRad = (deg: number) => (deg * Math.PI) / 180
  const arcStart = {
    x: cx + r * Math.cos(toRad(startAngle)),
    y: cy + r * Math.sin(toRad(startAngle)),
  }
  const arcEnd = {
    x: cx + r * Math.cos(toRad(startAngle + sweepAngle)),
    y: cy + r * Math.sin(toRad(startAngle + sweepAngle)),
  }
  const valueEnd = {
    x: cx + r * Math.cos(toRad(valueAngle)),
    y: cy + r * Math.sin(toRad(valueAngle)),
  }

  const largeArc = sweepAngle > 180 ? 1 : 0
  const valueLargeArc = (valueAngle - startAngle) > 180 ? 1 : 0

  return (
    <div className={`flex flex-col items-center ${formationActive ? 'ring-1 ring-terminal-warning/50 rounded' : ''}`}>
      <span className="text-[9px] font-mono text-terminal-subtle mb-0.5">{label}</span>
      <div className="relative">
        <svg width="72" height="50" viewBox="0 0 72 72">
          {/* Track */}
          <path
            d={`M ${arcStart.x} ${arcStart.y} A ${r} ${r} 0 ${largeArc} 1 ${arcEnd.x} ${arcEnd.y}`}
            fill="none"
            stroke="#374151"
            strokeWidth="6"
            strokeLinecap="round"
          />
          {/* Value arc */}
          {pct > 0 && (
            <path
              d={`M ${arcStart.x} ${arcStart.y} A ${r} ${r} 0 ${valueLargeArc} 1 ${valueEnd.x} ${valueEnd.y}`}
              fill="none"
              stroke={color}
              strokeWidth="6"
              strokeLinecap="round"
              opacity={active ? 1 : 0.5}
            />
          )}
          {/* Threshold marker at 1.2/3 = 40% */}
          <circle
            cx={cx + r * Math.cos(toRad(startAngle + sweepAngle * 0.4))}
            cy={cy + r * Math.sin(toRad(startAngle + sweepAngle * 0.4))}
            r="2"
            fill="#f59e0b"
          />
          {/* Mass value */}
          <text x={cx} y={cy + 4} textAnchor="middle" fill={color} fontSize="10" fontFamily="monospace" fontWeight="bold">
            {mass.toFixed(2)}
          </text>
        </svg>
      </div>
      <div className="flex items-center gap-1.5 mt-0.5">
        <span style={{ color: dirColor }} className="text-sm leading-none">{dirSymbol}</span>
        {ctl > 0 && <span className="text-[9px] font-mono text-terminal-warning">CTL:{ctl}</span>}
        {formationActive && (
          <span className="text-[9px] font-mono text-terminal-warning animate-pulse">●</span>
        )}
      </div>
    </div>
  )
}

// ---- Instrument Card ----
interface InstrumentCardProps {
  symbol: string
  state: InstrumentBHState
  isExpanded: boolean
  onToggle: () => void
}

function RegimeBadge({ regime }: { regime: BHRegime }) {
  const cfg = {
    BULL: { color: 'bg-terminal-bull/20 text-terminal-bull border-terminal-bull/30', label: 'BULL' },
    BEAR: { color: 'bg-terminal-bear/20 text-terminal-bear border-terminal-bear/30', label: 'BEAR' },
    SIDEWAYS: { color: 'bg-terminal-muted text-terminal-subtle border-terminal-border', label: 'SIDE' },
    HIGH_VOL: { color: 'bg-terminal-warning/20 text-terminal-warning border-terminal-warning/30', label: 'HV' },
  }[regime] ?? { color: '', label: regime }

  return (
    <span className={`text-[9px] font-mono px-1 py-0.5 rounded border ${cfg.color}`}>{cfg.label}</span>
  )
}

function InstrumentCard({ symbol, state, isExpanded, onToggle }: InstrumentCardProps) {
  const history = useBHStore((s) => s.history[symbol])
  const hasFormation = state.tf15m.bh_form > 0 || state.tf1h.bh_form > 0 || state.tf1d.bh_form > 0
  const maxMass = Math.max(state.tf15m.mass, state.tf1h.mass, state.tf1d.mass)
  const dominantRegime = state.tf1d.regime

  const massHistory = useMemo(() => {
    if (!history?.points.length) return []
    return history.points.slice(-50).map((p) => ({
      time: p.timestamp,
      mass1h: p.mass1h,
      mass1d: p.mass1d,
    }))
  }, [history])

  return (
    <motion.div
      layout
      className={`bg-terminal-surface border rounded transition-all ${
        hasFormation
          ? 'border-terminal-warning/50 shadow-[0_0_8px_rgba(245,158,11,0.2)]'
          : 'border-terminal-border'
      }`}
    >
      {/* Card header */}
      <div
        className="flex items-center justify-between p-2.5 cursor-pointer hover:bg-terminal-muted/20 transition-colors"
        onClick={onToggle}
      >
        <div className="flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full ${hasFormation ? 'bg-terminal-warning animate-pulse' : maxMass > 1.2 ? 'bg-terminal-accent' : 'bg-terminal-muted'}`} />
          <span className="font-mono text-xs font-semibold text-terminal-text">{symbol}</span>
          <RegimeBadge regime={dominantRegime} />
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] text-terminal-subtle">${state.price.toFixed(state.price > 100 ? 2 : 4)}</span>
          <span className={`text-[10px] ${isExpanded ? 'rotate-180' : ''} transition-transform`}>▼</span>
        </div>
      </div>

      {/* Gauges */}
      <div className="flex items-center justify-around px-2 pb-2">
        <BHGauge
          mass={state.tf15m.mass}
          active={state.tf15m.active}
          regime={state.tf15m.regime}
          dir={state.tf15m.dir}
          ctl={state.tf15m.ctl}
          label="15m"
          formationActive={state.tf15m.bh_form > 0}
        />
        <BHGauge
          mass={state.tf1h.mass}
          active={state.tf1h.active}
          regime={state.tf1h.regime}
          dir={state.tf1h.dir}
          ctl={state.tf1h.ctl}
          label="1h"
          formationActive={state.tf1h.bh_form > 0}
        />
        <BHGauge
          mass={state.tf1d.mass}
          active={state.tf1d.active}
          regime={state.tf1d.regime}
          dir={state.tf1d.dir}
          ctl={state.tf1d.ctl}
          label="1d"
          formationActive={state.tf1d.bh_form > 0}
        />
      </div>

      {/* Expanded: mass history chart */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-t border-terminal-border"
          >
            <div className="p-2">
              <div className="text-[10px] font-mono text-terminal-subtle mb-1">BH Mass History (24h)</div>
              {massHistory.length > 1 ? (
                <div className="h-20">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={massHistory} margin={{ top: 2, right: 2, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="#1f2937" strokeDasharray="2 4" />
                      <XAxis dataKey="time" hide />
                      <YAxis hide domain={[0, 3]} />
                      <Tooltip
                        formatter={(v: number) => [v.toFixed(3), '']}
                        labelFormatter={(l) => format(new Date(l as number), 'HH:mm')}
                        contentStyle={{ backgroundColor: '#111827', border: '1px solid #1f2937', borderRadius: 4, fontSize: 10 }}
                      />
                      {/* Threshold line at 1.2 */}
                      <Area type="monotone" dataKey="mass1h" stroke="#7c3aed" fill="rgba(124,58,237,0.1)" strokeWidth={1.5} dot={false} name="1h Mass" />
                      <Area type="monotone" dataKey="mass1d" stroke="#3b82f6" fill="rgba(59,130,246,0.1)" strokeWidth={1} dot={false} name="1d Mass" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-20 flex items-center justify-center text-terminal-subtle text-[10px]">
                  No history available
                </div>
              )}

              {/* Stats */}
              <div className="mt-1 grid grid-cols-3 gap-2 text-[9px] font-mono">
                <div>
                  <span className="text-terminal-subtle">Frac: </span>
                  <span className="text-terminal-text">{state.frac.toFixed(3)}</span>
                </div>
                {state.entryPrice && (
                  <div>
                    <span className="text-terminal-subtle">Entry: </span>
                    <span className="text-terminal-text">${state.entryPrice.toFixed(2)}</span>
                  </div>
                )}
                {state.positionSide && (
                  <div>
                    <span className={state.positionSide === 'long' ? 'text-terminal-bull' : 'text-terminal-bear'}>
                      {state.positionSide.toUpperCase()}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// ---- Main Dashboard ----
interface BHDashboardProps {
  className?: string
}

export const BHDashboard: React.FC<BHDashboardProps> = ({ className = '' }) => {
  const instruments = useBHStore((s) => s.instruments)
  const isConnected = useBHStore((s) => s.isConnected)
  const lastUpdate = useBHStore((s) => s.lastUpdate)
  const activeFormations = useBHStore((s) => s.activeFormations)
  const formationEvents = useBHStore((s) => s.formationEvents)

  const [expandedSymbol, setExpandedSymbol] = useState<string | null>(null)
  const [filter, setFilter] = useState<'all' | 'active' | 'formation'>('all')

  const symbolList = useMemo(() => {
    return Object.keys(instruments)
      .filter((sym) => {
        if (filter === 'active') {
          const i = instruments[sym]
          return i.tf15m.active || i.tf1h.active || i.tf1d.active
        }
        if (filter === 'formation') {
          const i = instruments[sym]
          return i.tf15m.bh_form > 0 || i.tf1h.bh_form > 0 || i.tf1d.bh_form > 0
        }
        return true
      })
      .sort((a, b) => {
        // Sort by max mass descending
        const maxA = Math.max(instruments[a]?.tf15m.mass ?? 0, instruments[a]?.tf1h.mass ?? 0, instruments[a]?.tf1d.mass ?? 0)
        const maxB = Math.max(instruments[b]?.tf15m.mass ?? 0, instruments[b]?.tf1h.mass ?? 0, instruments[b]?.tf1d.mass ?? 0)
        return maxB - maxA
      })
  }, [instruments, filter])

  return (
    <div className={`flex flex-col bg-terminal-bg h-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border flex-shrink-0">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-terminal-bull animate-pulse' : 'bg-terminal-bear'}`} />
          <span className="text-terminal-subtle text-xs font-mono uppercase tracking-wider">BH Physics</span>
          {activeFormations.length > 0 && (
            <span className="bg-terminal-warning/20 text-terminal-warning text-[10px] font-mono px-1.5 py-0.5 rounded-full">
              {activeFormations.length} formation{activeFormations.length > 1 ? 's' : ''}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {lastUpdate && (
            <span className="text-[10px] font-mono text-terminal-subtle">
              {format(new Date(lastUpdate), 'HH:mm:ss')}
            </span>
          )}
        </div>
      </div>

      {/* Filter bar */}
      <div className="flex items-center gap-1 px-3 py-1.5 border-b border-terminal-border flex-shrink-0">
        {[
          { value: 'all' as const, label: `All (${Object.keys(instruments).length})` },
          { value: 'active' as const, label: 'Active' },
          { value: 'formation' as const, label: `Formations (${activeFormations.length})` },
        ].map((f) => (
          <button
            key={f.value}
            onClick={() => setFilter(f.value)}
            className={`text-[10px] font-mono px-2 py-1 rounded transition-colors ${
              filter === f.value ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Instrument grid */}
      <div className="flex-1 overflow-y-auto p-2">
        {symbolList.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-terminal-subtle text-sm">
            {filter === 'active' ? 'No active BH formations' : 'No data available'}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {symbolList.map((symbol) => (
              <InstrumentCard
                key={symbol}
                symbol={symbol}
                state={instruments[symbol]}
                isExpanded={expandedSymbol === symbol}
                onToggle={() => setExpandedSymbol(expandedSymbol === symbol ? null : symbol)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Recent formations */}
      {formationEvents.length > 0 && (
        <div className="border-t border-terminal-border flex-shrink-0">
          <div className="px-3 py-1.5 flex items-center justify-between">
            <span className="text-[10px] font-mono text-terminal-subtle uppercase">Recent Formations</span>
          </div>
          <div className="max-h-24 overflow-y-auto">
            {formationEvents.slice(0, 10).map((evt) => (
              <div
                key={evt.id}
                className={`flex items-center gap-2 px-3 py-1 text-[10px] font-mono border-b border-terminal-border/30 ${!evt.acknowledged ? 'bg-terminal-warning/5' : ''}`}
              >
                <span className="text-terminal-subtle">{format(new Date(evt.timestamp), 'HH:mm')}</span>
                <span className="text-terminal-text font-semibold">{evt.symbol}</span>
                <span className="text-terminal-subtle">{evt.timeframe}</span>
                <span className="text-terminal-warning">M:{evt.mass.toFixed(2)}</span>
                <span className={evt.dir === 1 ? 'text-terminal-bull' : 'text-terminal-bear'}>
                  {evt.dir === 1 ? '↑' : '↓'} {evt.regime}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default BHDashboard
