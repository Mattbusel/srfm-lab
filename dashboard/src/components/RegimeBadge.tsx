// ============================================================
// RegimeBadge.tsx — Market regime badge component
// ============================================================
import React from 'react'
import { clsx } from 'clsx'
import type { MarketRegime } from '@/types'

interface RegimeBadgeProps {
  regime: MarketRegime
  showIcon?: boolean
  size?: 'xs' | 'sm'
  className?: string
}

const regimeConfig: Record<MarketRegime, {
  label: string
  icon: string
  bg: string
  text: string
  border: string
}> = {
  trending_up:   { label: 'Trending ↑', icon: '↗', bg: 'bg-emerald-950/50', text: 'text-emerald-400', border: 'border-emerald-800/40' },
  trending_down: { label: 'Trending ↓', icon: '↘', bg: 'bg-red-950/50',     text: 'text-red-400',     border: 'border-red-800/40' },
  ranging:       { label: 'Ranging',    icon: '↔', bg: 'bg-slate-900/50',   text: 'text-slate-400',   border: 'border-slate-700/40' },
  volatile:      { label: 'Volatile',   icon: '⚡', bg: 'bg-amber-950/50',  text: 'text-amber-400',   border: 'border-amber-800/40' },
}

export const RegimeBadge: React.FC<RegimeBadgeProps> = ({
  regime,
  showIcon = true,
  size = 'sm',
  className,
}) => {
  const cfg = regimeConfig[regime]
  const textSize = size === 'xs' ? 'text-[9px]' : 'text-[10px]'
  const padding = size === 'xs' ? 'px-1 py-0.5' : 'px-1.5 py-0.5'

  return (
    <span className={clsx(
      'inline-flex items-center gap-1 rounded border font-mono',
      textSize, padding,
      cfg.bg, cfg.text, cfg.border,
      className,
    )}>
      {showIcon && <span>{cfg.icon}</span>}
      {cfg.label}
    </span>
  )
}
