import type { RegimeType } from '@/types/trades'

// ── Regime colors ─────────────────────────────────────────────────────────────

export const REGIME_COLORS: Record<RegimeType, string> = {
  bull: '#22c55e',
  bear: '#ef4444',
  sideways: '#6b7280',
  ranging: '#8b5cf6',
  volatile: '#f59e0b',
}

export const REGIME_BG_COLORS: Record<RegimeType, string> = {
  bull: 'rgba(34, 197, 94, 0.15)',
  bear: 'rgba(239, 68, 68, 0.15)',
  sideways: 'rgba(107, 114, 128, 0.15)',
  ranging: 'rgba(139, 92, 246, 0.15)',
  volatile: 'rgba(245, 158, 11, 0.15)',
}

export const REGIME_LABELS: Record<RegimeType, string> = {
  bull: 'Bull',
  bear: 'Bear',
  sideways: 'Sideways',
  ranging: 'Ranging',
  volatile: 'Volatile',
}

// ── Chart palette ─────────────────────────────────────────────────────────────

export const CHART_COLORS = [
  '#3b82f6',  // blue
  '#22c55e',  // green
  '#ef4444',  // red
  '#8b5cf6',  // purple
  '#f59e0b',  // amber
  '#06b6d4',  // cyan
  '#f97316',  // orange
  '#ec4899',  // pink
  '#84cc16',  // lime
  '#14b8a6',  // teal
]

export const EQUITY_CURVE_COLOR = '#3b82f6'
export const BENCHMARK_COLOR = '#6b7280'
export const DRAWDOWN_COLOR = '#ef4444'
export const IC_LINE_COLOR = '#22c55e'
export const IC_BAND_COLOR = 'rgba(34, 197, 94, 0.15)'

// ── MC fan chart colors ────────────────────────────────────────────────────────

export const MC_BAND_COLORS = {
  p5_p95: 'rgba(59, 130, 246, 0.1)',
  p25_p75: 'rgba(59, 130, 246, 0.2)',
  median: '#3b82f6',
  mean: '#f59e0b',
}

// ── Heatmap gradient ──────────────────────────────────────────────────────────

export function heatmapColor(value: number, min = -1, max = 1): string {
  // normalize to [0, 1]
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)))
  if (t < 0.5) {
    // red to neutral
    const r = Math.round(239 + (30 - 239) * (t / 0.5))
    const g = Math.round(68 + (41 - 68) * (t / 0.5))
    const b = Math.round(68 + (55 - 68) * (t / 0.5))
    return `rgb(${r},${g},${b})`
  } else {
    // neutral to green
    const r2 = Math.round(30 + (34 - 30) * ((t - 0.5) / 0.5))
    const g2 = Math.round(41 + (197 - 41) * ((t - 0.5) / 0.5))
    const b2 = Math.round(55 + (94 - 55) * ((t - 0.5) / 0.5))
    return `rgb(${r2},${g2},${b2})`
  }
}

export function signalHeatmapColor(zscore: number): string {
  const clamped = Math.max(-3, Math.min(3, zscore))
  return heatmapColor(clamped, -3, 3)
}

export function correlationColor(value: number): string {
  return heatmapColor(value, -1, 1)
}
