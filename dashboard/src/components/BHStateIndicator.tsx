// ============================================================
// BHStateIndicator.tsx — Visual indicator for BH signal state
// ============================================================
import React from 'react'
import { clsx } from 'clsx'
import type { BHState, Timeframe } from '@/types'

interface BHStateIndicatorProps {
  state: BHState
  timeframe?: Timeframe
  showLabel?: boolean
  size?: 'xs' | 'sm' | 'md'
  className?: string
}

const stateConfig: Record<BHState, { color: string; dot: string; label: string; icon: string }> = {
  bullish:  { color: 'text-emerald-400', dot: 'bg-emerald-400', label: 'Bullish',  icon: '▲' },
  bearish:  { color: 'text-red-400',     dot: 'bg-red-400',     label: 'Bearish',  icon: '▼' },
  neutral:  { color: 'text-slate-500',   dot: 'bg-slate-600',   label: 'Neutral',  icon: '◆' },
}

export const BHStateIndicator: React.FC<BHStateIndicatorProps> = ({
  state,
  timeframe,
  showLabel = true,
  size = 'sm',
  className,
}) => {
  const cfg = stateConfig[state]
  const dotSize = { xs: 'w-1.5 h-1.5', sm: 'w-2 h-2', md: 'w-2.5 h-2.5' }[size]
  const textSize = { xs: 'text-[9px]', sm: 'text-[10px]', md: 'text-xs' }[size]

  return (
    <div className={clsx('flex items-center gap-1', className)}>
      <div className={clsx('rounded-full flex-shrink-0', dotSize, cfg.dot)} />
      {timeframe && (
        <span className={clsx('font-mono text-slate-600', textSize)}>{timeframe}</span>
      )}
      {showLabel && (
        <span className={clsx('font-mono font-semibold', textSize, cfg.color)}>
          {cfg.icon} {cfg.label}
        </span>
      )}
    </div>
  )
}

// ---- Multi-timeframe row ----

interface BHStateRowProps {
  daily: BHState
  hourly: BHState
  m15: BHState
  compact?: boolean
}

export const BHStateRow: React.FC<BHStateRowProps> = ({ daily, hourly, m15, compact = false }) => {
  const tfs: { tf: Timeframe; state: BHState }[] = [
    { tf: '1d', state: daily },
    { tf: '1h', state: hourly },
    { tf: '15m', state: m15 },
  ]
  return (
    <div className="flex items-center gap-2">
      {tfs.map(({ tf, state }) => {
        const cfg = stateConfig[state]
        return (
          <div key={tf} className={clsx(
            'flex items-center gap-1 px-1.5 py-0.5 rounded border',
            state === 'bullish' && 'border-emerald-800/40 bg-emerald-950/30',
            state === 'bearish' && 'border-red-800/40 bg-red-950/30',
            state === 'neutral' && 'border-slate-700/40 bg-slate-900/30',
          )}>
            <span className={clsx('text-[9px] font-mono text-slate-600')}>{tf}</span>
            {!compact && (
              <span className={clsx('text-[9px] font-mono', cfg.color)}>{cfg.icon}</span>
            )}
            <div className={clsx('w-1.5 h-1.5 rounded-full', cfg.dot)} />
          </div>
        )
      })}
    </div>
  )
}

// ---- Mass gauge ----

interface MassGaugeProps {
  mass: number   // 0-2
  size?: number
  className?: string
}

export const MassGauge: React.FC<MassGaugeProps> = ({ mass, size = 48, className }) => {
  const clamped = Math.max(0, Math.min(2, mass))
  const pct = clamped / 2  // 0-1
  const angle = pct * 180  // 0-180 degrees

  // SVG arc math
  const cx = size / 2
  const cy = size / 2
  const r = size * 0.38
  const strokeW = size * 0.08

  function polarToXY(deg: number) {
    const rad = ((deg - 180) * Math.PI) / 180
    return {
      x: cx + r * Math.cos(rad),
      y: cy + r * Math.sin(rad),
    }
  }

  const start = polarToXY(0)
  const end = polarToXY(angle)
  const largeArc = angle > 90 ? 1 : 0

  const color = clamped > 1.5 ? '#22c55e' : clamped > 1.0 ? '#3b82f6' : clamped > 0.5 ? '#f59e0b' : '#475569'

  return (
    <div className={clsx('flex flex-col items-center', className)}>
      <svg width={size} height={size * 0.6} viewBox={`0 0 ${size} ${size * 0.6}`}>
        {/* Track */}
        <path
          d={`M ${start.x} ${start.y} A ${r} ${r} 0 1 1 ${polarToXY(180).x} ${polarToXY(180).y}`}
          fill="none"
          stroke="#1e2130"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />
        {/* Fill */}
        {clamped > 0 && (
          <path
            d={`M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`}
            fill="none"
            stroke={color}
            strokeWidth={strokeW}
            strokeLinecap="round"
          />
        )}
      </svg>
      <span className="text-[10px] font-mono font-bold mt-0.5" style={{ color }}>
        {clamped.toFixed(1)}
      </span>
    </div>
  )
}
