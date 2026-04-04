import React from 'react'
import { clsx } from 'clsx'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface MetricCardProps {
  label: string
  value: string
  subvalue?: string
  delta?: number
  deltaLabel?: string
  trend?: 'up' | 'down' | 'neutral'
  variant?: 'default' | 'bull' | 'bear' | 'warning' | 'info'
  icon?: React.ReactNode
  className?: string
  loading?: boolean
}

export function MetricCard({
  label,
  value,
  subvalue,
  delta,
  deltaLabel,
  trend,
  variant = 'default',
  icon,
  className,
  loading = false,
}: MetricCardProps) {
  const variantClasses = {
    default: 'border-research-border',
    bull: 'border-research-bull/30',
    bear: 'border-research-bear/30',
    warning: 'border-research-warning/30',
    info: 'border-research-info/30',
  }

  const trendColor = trend === 'up'
    ? 'text-research-bull'
    : trend === 'down'
      ? 'text-research-bear'
      : 'text-research-subtle'

  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus

  if (loading) {
    return (
      <div className={clsx(
        'bg-research-card border rounded-lg p-4 animate-pulse',
        variantClasses[variant],
        className
      )}>
        <div className="h-3 bg-research-muted rounded w-24 mb-3" />
        <div className="h-7 bg-research-muted rounded w-32 mb-2" />
        <div className="h-3 bg-research-muted rounded w-20" />
      </div>
    )
  }

  return (
    <div className={clsx(
      'bg-research-card border rounded-lg p-4 flex flex-col gap-1 hover:border-research-accent/40 transition-colors',
      variantClasses[variant],
      className
    )}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-research-subtle uppercase tracking-wide">
          {label}
        </span>
        {icon && (
          <span className="text-research-subtle opacity-60">{icon}</span>
        )}
      </div>

      <div className="flex items-end gap-2 mt-1">
        <span className="text-2xl font-semibold font-mono text-research-text leading-none">
          {value}
        </span>
        {subvalue && (
          <span className="text-sm text-research-subtle mb-0.5 font-mono">
            {subvalue}
          </span>
        )}
      </div>

      {(delta !== undefined || deltaLabel) && (
        <div className={clsx('flex items-center gap-1 text-xs', trendColor)}>
          {trend && <TrendIcon size={12} />}
          {delta !== undefined && (
            <span className="font-mono">
              {delta > 0 ? '+' : ''}{delta.toFixed(2)}%
            </span>
          )}
          {deltaLabel && <span className="text-research-subtle">{deltaLabel}</span>}
        </div>
      )}
    </div>
  )
}
