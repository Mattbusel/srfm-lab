// ============================================================
// ui.tsx — Shared primitive UI components
// ============================================================
import React from 'react'
import { clsx } from 'clsx'

// ---- Card ----

interface CardProps {
  children: React.ReactNode
  className?: string
  title?: string
  subtitle?: string
  actions?: React.ReactNode
  padding?: 'sm' | 'md' | 'lg' | 'none'
}

export const Card: React.FC<CardProps> = ({
  children,
  className,
  title,
  subtitle,
  actions,
  padding = 'md',
}) => {
  const paddingClass = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
  }[padding]

  return (
    <div className={clsx('bg-[#111318] border border-[#1e2130] rounded-lg', paddingClass, className)}>
      {(title || actions) && (
        <div className="flex items-center justify-between mb-3">
          <div>
            {title && <h3 className="text-xs font-mono font-semibold text-slate-200 uppercase tracking-wider">{title}</h3>}
            {subtitle && <p className="text-[10px] font-mono text-slate-500 mt-0.5">{subtitle}</p>}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </div>
  )
}

// ---- StatCard ----

interface StatCardProps {
  label: string
  value: string | number
  change?: number
  changeLabel?: string
  className?: string
  valueClass?: string
  icon?: React.ReactNode
  prefix?: string
  suffix?: string
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  change,
  changeLabel,
  className,
  valueClass,
  icon,
  prefix,
  suffix,
}) => {
  const isPositive = change != null && change >= 0
  return (
    <div className={clsx('bg-[#111318] border border-[#1e2130] rounded-lg p-4', className)}>
      <div className="flex items-start justify-between">
        <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">{label}</span>
        {icon && <span className="text-slate-600">{icon}</span>}
      </div>
      <div className={clsx('mt-1.5 text-xl font-mono font-semibold text-slate-100', valueClass)}>
        {prefix}{typeof value === 'number' ? value.toLocaleString() : value}{suffix}
      </div>
      {change != null && (
        <div className={clsx(
          'mt-1 text-[11px] font-mono',
          isPositive ? 'text-emerald-400' : 'text-red-400',
        )}>
          {isPositive ? '+' : ''}{change.toFixed(2)}%
          {changeLabel && <span className="text-slate-600 ml-1">{changeLabel}</span>}
        </div>
      )}
    </div>
  )
}

// ---- Badge ----

type BadgeVariant = 'default' | 'bull' | 'bear' | 'neutral' | 'warning' | 'info'

interface BadgeProps {
  children: React.ReactNode
  variant?: BadgeVariant
  className?: string
}

const badgeClasses: Record<BadgeVariant, string> = {
  default: 'bg-slate-800 text-slate-300 border-slate-700',
  bull: 'bg-emerald-950/60 text-emerald-400 border-emerald-800/50',
  bear: 'bg-red-950/60 text-red-400 border-red-800/50',
  neutral: 'bg-slate-800/60 text-slate-400 border-slate-700/50',
  warning: 'bg-amber-950/60 text-amber-400 border-amber-800/50',
  info: 'bg-cyan-950/60 text-cyan-400 border-cyan-800/50',
}

export const Badge: React.FC<BadgeProps> = ({ children, variant = 'default', className }) => (
  <span className={clsx(
    'inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono border',
    badgeClasses[variant],
    className,
  )}>
    {children}
  </span>
)

// ---- ProgressBar ----

interface ProgressBarProps {
  value: number  // 0-1
  max?: number
  color?: string
  label?: string
  showValue?: boolean
  height?: number
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 1,
  color = '#3b82f6',
  label,
  showValue = false,
  height = 6,
}) => {
  const pct = Math.min(Math.max(value / max, 0), 1) * 100
  return (
    <div>
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-1">
          {label && <span className="text-[10px] font-mono text-slate-500">{label}</span>}
          {showValue && <span className="text-[10px] font-mono text-slate-400">{pct.toFixed(1)}%</span>}
        </div>
      )}
      <div className="w-full bg-[#1e2130] rounded-full overflow-hidden" style={{ height }}>
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  )
}

// ---- Divider ----

export const Divider: React.FC<{ className?: string }> = ({ className }) => (
  <div className={clsx('border-t border-[#1e2130]', className)} />
)

// ---- LoadingSpinner ----

export const LoadingSpinner: React.FC<{ size?: number }> = ({ size = 20 }) => (
  <svg
    className="animate-spin text-blue-500"
    style={{ width: size, height: size }}
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
  >
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
  </svg>
)

// ---- EmptyState ----

interface EmptyStateProps {
  message?: string
  icon?: string
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  message = 'No data available',
  icon = '◎',
}) => (
  <div className="flex flex-col items-center justify-center py-12 text-slate-600">
    <span className="text-3xl mb-3">{icon}</span>
    <span className="text-xs font-mono">{message}</span>
  </div>
)

// ---- Select ----

interface SelectProps {
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
  className?: string
}

export const Select: React.FC<SelectProps> = ({ value, onChange, options, className }) => (
  <select
    value={value}
    onChange={(e) => onChange(e.target.value)}
    className={clsx(
      'bg-[#1a1d26] border border-[#1e2130] text-slate-300 text-xs font-mono rounded px-2 py-1',
      'focus:outline-none focus:border-blue-500/50',
      className,
    )}
  >
    {options.map((o) => (
      <option key={o.value} value={o.value}>{o.label}</option>
    ))}
  </select>
)

// ---- WsStatusDot ----

import type { WsStatus } from '@/hooks/useWebSocket'

export const WsStatusDot: React.FC<{ status: WsStatus }> = ({ status }) => {
  const classes: Record<WsStatus, string> = {
    open: 'bg-emerald-400 animate-pulse',
    connecting: 'bg-amber-400 animate-pulse',
    closed: 'bg-slate-600',
    error: 'bg-red-400',
  }
  const labels: Record<WsStatus, string> = {
    open: 'Live',
    connecting: 'Connecting',
    closed: 'Offline',
    error: 'Error',
  }
  return (
    <div className="flex items-center gap-1.5">
      <div className={clsx('w-1.5 h-1.5 rounded-full', classes[status])} />
      <span className="text-[10px] font-mono text-slate-500">{labels[status]}</span>
    </div>
  )
}
