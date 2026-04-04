// ============================================================
// DeltaScoreCard.tsx — Delta score visual card
// ============================================================
import React from 'react'
import { clsx } from 'clsx'
import type { SignalCard } from '@/types'

interface DeltaScoreCardProps {
  card: SignalCard
  onClick?: () => void
  compact?: boolean
}

export const DeltaScoreCard: React.FC<DeltaScoreCardProps> = ({ card, onClick, compact = false }) => {
  const score = Math.max(-1, Math.min(1, card.deltaScore))
  const barWidth = Math.abs(score) * 50  // 0–50%
  const isPositive = score >= 0

  const massColor = card.mass > 1.5
    ? 'text-emerald-400'
    : card.mass > 1.0
      ? 'text-blue-400'
      : card.mass > 0.5
        ? 'text-amber-400'
        : 'text-slate-500'

  return (
    <div
      onClick={onClick}
      className={clsx(
        'bg-[#111318] border border-[#1e2130] rounded-lg transition-all',
        onClick && 'cursor-pointer hover:border-blue-500/40 hover:bg-[#13161e]',
        compact ? 'p-2' : 'p-3',
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-mono font-semibold text-slate-200">{card.symbol.replace('USDT', '')}</span>
        <div className="flex items-center gap-1.5">
          <span className={clsx('text-[10px] font-mono font-bold', massColor)}>
            M{card.mass.toFixed(1)}
          </span>
          <span className="text-[10px] font-mono text-slate-500">{card.activeFormations}f</span>
        </div>
      </div>

      {/* Delta score bar */}
      <div className="mb-2">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[9px] font-mono text-slate-600">DELTA SCORE</span>
          <span className={clsx(
            'text-[10px] font-mono font-semibold',
            isPositive ? 'text-emerald-400' : 'text-red-400',
          )}>
            {score >= 0 ? '+' : ''}{score.toFixed(2)}
          </span>
        </div>
        {/* Bar — center = 0, extends left (bear) or right (bull) */}
        <div className="relative h-2 bg-[#1e2130] rounded-full overflow-hidden">
          <div className="absolute inset-y-0 left-1/2 w-px bg-[#2e3550]" />
          <div
            className={clsx(
              'absolute inset-y-0 rounded-full transition-all duration-300',
              isPositive ? 'bg-emerald-500' : 'bg-red-500',
            )}
            style={{
              left: isPositive ? '50%' : `${50 - barWidth}%`,
              width: `${barWidth}%`,
            }}
          />
        </div>
      </div>

      {/* Price & change */}
      {!compact && (
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-mono text-slate-400">
            ${card.price >= 1000
              ? card.price.toLocaleString('en-US', { maximumFractionDigits: 0 })
              : card.price >= 1
                ? card.price.toFixed(2)
                : card.price.toFixed(4)
            }
          </span>
          <span className={clsx(
            'text-[10px] font-mono',
            card.change24hPct >= 0 ? 'text-emerald-400' : 'text-red-400',
          )}>
            {card.change24hPct >= 0 ? '+' : ''}{(card.change24hPct * 100).toFixed(2)}%
          </span>
        </div>
      )}
    </div>
  )
}
