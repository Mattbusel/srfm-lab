// ============================================================
// MetricsPanel.tsx — Metric cards with sparklines and 24h change
// ============================================================

import React, { useMemo } from 'react'
import {
  LineChart,
  Line,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  YAxis,
} from 'recharts'
import { clsx } from 'clsx'
import type { MetricCardData, PortfolioMetrics, RiskMetrics, SparklinePoint } from '@/types/metrics'

// ---- Formatting helpers ----------------------------------------------

function formatValue(value: number, format: MetricCardData['format'] = 'ratio'): string {
  switch (format) {
    case 'currency':
      return value >= 1_000_000
        ? `$${(value / 1_000_000).toFixed(2)}M`
        : value >= 1_000
        ? `$${(value / 1_000).toFixed(1)}K`
        : `$${value.toFixed(0)}`
    case 'percent':
      return `${(value * 100).toFixed(2)}%`
    case 'count':
      return value.toFixed(0)
    default:
      return value.toFixed(3)
  }
}

function formatChange(change: number, format: MetricCardData['format'] = 'ratio'): string {
  const sign = change >= 0 ? '+' : ''
  switch (format) {
    case 'currency':
      return `${sign}$${Math.abs(change) >= 1000 ? (change / 1000).toFixed(1) + 'K' : change.toFixed(0)}`
    case 'percent':
      return `${sign}${(change * 100).toFixed(2)}%`
    default:
      return `${sign}${change.toFixed(3)}`
  }
}

// ---- Mini sparkline --------------------------------------------------

interface SparklineProps {
  data: SparklinePoint[]
  color: string
  height?: number
}

const Sparkline: React.FC<SparklineProps> = ({ data, color, height = 40 }) => {
  if (!data || data.length < 2) return null

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <YAxis domain={['auto', 'auto']} hide />
        <Line
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ---- Threshold ring indicator ----------------------------------------

function getThresholdColor(
  value: number,
  threshold: MetricCardData['threshold'],
  higherIsBetter: boolean = true
): 'green' | 'yellow' | 'red' {
  if (!threshold) return 'green'
  const { warn, critical } = threshold
  if (higherIsBetter) {
    if (value >= warn) return 'green'
    if (value >= critical) return 'yellow'
    return 'red'
  } else {
    if (value <= warn) return 'green'
    if (value <= critical) return 'yellow'
    return 'red'
  }
}

const colorMap = {
  green: {
    bg: 'bg-green-900/20',
    border: 'border-green-500/40',
    text: '#22c55e',
    badge: 'bg-green-500/20 text-green-400',
  },
  yellow: {
    bg: 'bg-yellow-900/20',
    border: 'border-yellow-500/40',
    text: '#f59e0b',
    badge: 'bg-yellow-500/20 text-yellow-400',
  },
  red: {
    bg: 'bg-red-900/20',
    border: 'border-red-500/40',
    text: '#ef4444',
    badge: 'bg-red-500/20 text-red-400',
  },
}

// ---- Single MetricCard -----------------------------------------------

interface MetricCardProps {
  data: MetricCardData
  className?: string
}

export const MetricCard: React.FC<MetricCardProps> = ({ data, className }) => {
  const {
    label,
    value,
    unit,
    change24h,
    sparkline,
    threshold,
    higherIsBetter = true,
    format = 'ratio',
  } = data

  const colorKey = getThresholdColor(value, threshold, higherIsBetter)
  const colors = colorMap[colorKey]

  const changePositive = (change24h ?? 0) >= 0
  const changeGood = higherIsBetter ? changePositive : !changePositive
  const changeColor = changeGood ? '#22c55e' : '#ef4444'
  const changeIcon = changePositive ? '▲' : '▼'

  return (
    <div
      className={clsx(
        'rounded-xl border p-4 flex flex-col gap-2 bg-gray-900',
        colors.border,
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          {label}
        </span>
        {threshold && (
          <span className={clsx('text-xs px-2 py-0.5 rounded-full font-medium', colors.badge)}>
            {colorKey === 'green' ? 'OK' : colorKey === 'yellow' ? 'WARN' : 'CRIT'}
          </span>
        )}
      </div>

      {/* Value */}
      <div className="flex items-end gap-2">
        <span
          className="text-2xl font-bold tabular-nums"
          style={{ color: colors.text }}
        >
          {formatValue(value, format)}
        </span>
        {unit && <span className="text-sm text-gray-500 mb-0.5">{unit}</span>}
      </div>

      {/* 24h change */}
      {change24h !== undefined && (
        <div className="flex items-center gap-1">
          <span className="text-xs" style={{ color: changeColor }}>
            {changeIcon} {formatChange(change24h, format)}
          </span>
          <span className="text-xs text-gray-500">24h</span>
        </div>
      )}

      {/* Sparkline */}
      {sparkline && sparkline.length > 1 && (
        <div className="mt-1">
          <Sparkline data={sparkline} color={colors.text} />
        </div>
      )}
    </div>
  )
}

// ---- Tooltip for sparkline context -----------------------------------

const SparklineTooltipContent: React.FC<{ active?: boolean; payload?: Array<{ value: number }> }> = ({
  active,
  payload,
}) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-200">
      {payload[0].value.toFixed(4)}
    </div>
  )
}

// ---- MetricsPanel (full panel of cards) ------------------------------

interface MetricsPanelProps {
  portfolio: PortfolioMetrics | null
  riskMetrics: RiskMetrics | null
  equitySparkline?: SparklinePoint[]
  className?: string
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({
  portfolio,
  riskMetrics,
  equitySparkline,
  className,
}) => {
  const cards = useMemo((): MetricCardData[] => {
    if (!portfolio && !riskMetrics) return []

    const p = portfolio
    const r = riskMetrics

    return [
      {
        label: 'Total Equity',
        value: p?.totalEquity ?? 0,
        change24h: p ? p.dailyPnl : undefined,
        format: 'currency',
        sparkline: equitySparkline,
        higherIsBetter: true,
        threshold: { warn: 900_000, critical: 800_000 },
      },
      {
        label: 'Daily P&L',
        value: p?.dailyPnl ?? 0,
        change24h: p?.weeklyPnl !== undefined ? (p.weeklyPnl - (p.dailyPnl * 5)) : undefined,
        format: 'currency',
        higherIsBetter: true,
      },
      {
        label: 'Monthly P&L',
        value: p?.monthlyPnl ?? 0,
        format: 'currency',
        higherIsBetter: true,
      },
      {
        label: 'Sharpe Ratio',
        value: r?.sharpeRatio ?? 0,
        format: 'ratio',
        higherIsBetter: true,
        threshold: { warn: 1.0, critical: 0.5 },
      },
      {
        label: 'Win Rate',
        value: r?.winRate ?? 0,
        format: 'percent',
        higherIsBetter: true,
        threshold: { warn: 0.5, critical: 0.4 },
      },
      {
        label: 'Profit Factor',
        value: r?.profitFactor ?? 0,
        format: 'ratio',
        higherIsBetter: true,
        threshold: { warn: 1.2, critical: 1.0 },
      },
      {
        label: 'Max Drawdown',
        value: r?.maxDrawdown ?? 0,
        format: 'percent',
        higherIsBetter: false,
        threshold: { warn: -0.10, critical: -0.20 },
      },
      {
        label: 'Curr. Drawdown',
        value: r?.currentDrawdown ?? 0,
        format: 'percent',
        higherIsBetter: false,
        threshold: { warn: -0.05, critical: -0.15 },
      },
      {
        label: 'Ann. Volatility',
        value: r?.volatilityAnn ?? 0,
        format: 'percent',
        higherIsBetter: false,
        threshold: { warn: 0.30, critical: 0.50 },
      },
      {
        label: 'VaR 99% 1D',
        value: r?.var99_1d ?? 0,
        format: 'currency',
        higherIsBetter: false,
        threshold: { warn: 30_000, critical: 50_000 },
      },
      {
        label: 'CVaR 99% 1D',
        value: r?.cvar99_1d ?? 0,
        format: 'currency',
        higherIsBetter: false,
        threshold: { warn: 45_000, critical: 75_000 },
      },
      {
        label: 'Sortino',
        value: r?.sortinoRatio ?? 0,
        format: 'ratio',
        higherIsBetter: true,
        threshold: { warn: 1.5, critical: 0.8 },
      },
      {
        label: 'Calmar',
        value: r?.calmarRatio ?? 0,
        format: 'ratio',
        higherIsBetter: true,
        threshold: { warn: 1.0, critical: 0.5 },
      },
      {
        label: 'Beta',
        value: r?.beta ?? 0,
        format: 'ratio',
        higherIsBetter: false,
        threshold: { warn: 0.8, critical: 1.2 },
      },
      {
        label: 'Margin Used',
        value: p?.marginUtilization ?? 0,
        format: 'percent',
        higherIsBetter: false,
        threshold: { warn: 0.60, critical: 0.80 },
      },
    ]
  }, [portfolio, riskMetrics, equitySparkline])

  if (cards.length === 0) {
    return (
      <div className={clsx('flex items-center justify-center h-32 text-gray-500', className)}>
        <span className="text-sm">Awaiting data…</span>
      </div>
    )
  }

  return (
    <div className={clsx('grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3', className)}>
      {cards.map(card => (
        <MetricCard key={card.label} data={card} />
      ))}
    </div>
  )
}

// ---- Compact one-row summary bar ------------------------------------

interface MetricsSummaryBarProps {
  portfolio: PortfolioMetrics | null
  riskMetrics: RiskMetrics | null
  className?: string
}

export const MetricsSummaryBar: React.FC<MetricsSummaryBarProps> = ({
  portfolio,
  riskMetrics,
  className,
}) => {
  const stats = [
    { label: 'Equity', value: portfolio?.totalEquity ?? 0, format: 'currency' as const },
    { label: 'Daily P&L', value: portfolio?.dailyPnl ?? 0, format: 'currency' as const, signed: true },
    { label: 'Sharpe', value: riskMetrics?.sharpeRatio ?? 0, format: 'ratio' as const },
    { label: 'Win Rate', value: riskMetrics?.winRate ?? 0, format: 'percent' as const },
    { label: 'DD', value: riskMetrics?.currentDrawdown ?? 0, format: 'percent' as const },
  ]

  return (
    <div className={clsx('flex items-center gap-6 bg-gray-900 rounded-lg px-4 py-2 border border-gray-800', className)}>
      {stats.map(({ label, value, format }) => {
        const isNeg = value < 0
        const color = isNeg ? '#ef4444' : '#22c55e'
        return (
          <div key={label} className="flex flex-col items-center">
            <span className="text-xs text-gray-500">{label}</span>
            <span className="text-sm font-semibold tabular-nums" style={{ color }}>
              {formatValue(value, format)}
            </span>
          </div>
        )
      })}
    </div>
  )
}
