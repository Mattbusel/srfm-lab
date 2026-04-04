import { format, formatDistanceToNow, parseISO } from 'date-fns'

// ── Currency ──────────────────────────────────────────────────────────────────

export function formatCurrency(
  value: number,
  opts: { decimals?: number; compact?: boolean; sign?: boolean } = {}
): string {
  const { decimals = 2, compact = false, sign = false } = opts
  const abs = Math.abs(value)
  const prefix = sign && value > 0 ? '+' : value < 0 ? '-' : ''

  if (compact) {
    if (abs >= 1_000_000) return `${prefix}$${(abs / 1_000_000).toFixed(1)}M`
    if (abs >= 1_000) return `${prefix}$${(abs / 1_000).toFixed(1)}K`
  }

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
    signDisplay: sign ? 'exceptZero' : 'auto',
  }).format(value)
}

// ── Percentage ────────────────────────────────────────────────────────────────

export function formatPct(
  value: number,
  opts: { decimals?: number; sign?: boolean; multiply?: boolean } = {}
): string {
  const { decimals = 2, sign = false, multiply = false } = opts
  const v = multiply ? value * 100 : value
  const prefix = sign && v > 0 ? '+' : ''
  return `${prefix}${v.toFixed(decimals)}%`
}

// ── Numbers ───────────────────────────────────────────────────────────────────

export function formatNumber(
  value: number,
  opts: { decimals?: number; compact?: boolean } = {}
): string {
  const { decimals = 2, compact = false } = opts
  if (compact) {
    if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
    if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  }
  return value.toFixed(decimals)
}

export function formatRatio(value: number, decimals = 2): string {
  return value.toFixed(decimals)
}

// ── Dates ─────────────────────────────────────────────────────────────────────

export function formatDate(date: string | Date, fmt = 'yyyy-MM-dd'): string {
  const d = typeof date === 'string' ? parseISO(date) : date
  return format(d, fmt)
}

export function formatDateTime(date: string | Date): string {
  const d = typeof date === 'string' ? parseISO(date) : date
  return format(d, 'yyyy-MM-dd HH:mm:ss')
}

export function formatRelativeTime(date: string | Date): string {
  const d = typeof date === 'string' ? parseISO(date) : date
  return formatDistanceToNow(d, { addSuffix: true })
}

export function formatDuration(hours: number): string {
  if (hours < 1) return `${Math.round(hours * 60)}m`
  if (hours < 24) return `${hours.toFixed(1)}h`
  return `${(hours / 24).toFixed(1)}d`
}

// ── Conditional coloring helpers ──────────────────────────────────────────────

export function pnlColor(value: number): string {
  if (value > 0) return 'text-research-bull'
  if (value < 0) return 'text-research-bear'
  return 'text-research-subtle'
}

export function signalColor(value: number): string {
  if (value > 0.5) return 'text-research-bull'
  if (value < -0.5) return 'text-research-bear'
  return 'text-research-subtle'
}
