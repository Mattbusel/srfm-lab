import React from 'react'
import type { PerformanceMetrics } from '@/types/trades'
import { formatCurrency, formatPct, formatRatio, pnlColor } from '@/utils/formatters'
import { clsx } from 'clsx'

interface PerformanceTableProps {
  metrics: PerformanceMetrics
  compact?: boolean
}

interface MetricRow {
  label: string
  value: string
  colorClass?: string
  group?: string
}

export function PerformanceTable({ metrics, compact = false }: PerformanceTableProps) {
  const rows: MetricRow[] = [
    { group: 'Returns', label: 'Total P&L', value: formatCurrency(metrics.totalPnl, { sign: true }), colorClass: pnlColor(metrics.totalPnl) },
    { label: 'Total Return', value: formatPct(metrics.totalPnlPct, { sign: true }), colorClass: pnlColor(metrics.totalPnlPct) },
    { label: 'Ann. Return', value: formatPct(metrics.annualizedReturn * 100, { sign: true }), colorClass: pnlColor(metrics.annualizedReturn) },
    { label: 'Ann. Volatility', value: formatPct(metrics.annualizedVolatility * 100) },

    { group: 'Risk', label: 'Sharpe Ratio', value: formatRatio(metrics.sharpeRatio), colorClass: metrics.sharpeRatio > 1 ? 'text-research-bull' : 'text-research-bear' },
    { label: 'Sortino Ratio', value: formatRatio(metrics.sortinoRatio), colorClass: metrics.sortinoRatio > 1 ? 'text-research-bull' : 'text-research-bear' },
    { label: 'Calmar Ratio', value: formatRatio(metrics.calmarRatio) },
    { label: 'Max Drawdown', value: formatPct(metrics.maxDrawdownPct, { sign: false }), colorClass: 'text-research-bear' },
    { label: 'VaR (95%)', value: formatPct(metrics.var95 * 100), colorClass: 'text-research-bear' },
    { label: 'CVaR (95%)', value: formatPct(metrics.cvar95 * 100), colorClass: 'text-research-bear' },

    { group: 'Trade Stats', label: 'Total Trades', value: metrics.totalTrades.toString() },
    { label: 'Win Rate', value: formatPct(metrics.winRate * 100), colorClass: metrics.winRate > 0.5 ? 'text-research-bull' : 'text-research-bear' },
    { label: 'Profit Factor', value: formatRatio(metrics.profitFactor, 3), colorClass: metrics.profitFactor > 1 ? 'text-research-bull' : 'text-research-bear' },
    { label: 'Avg Win', value: formatCurrency(metrics.avgWin, { sign: true }), colorClass: 'text-research-bull' },
    { label: 'Avg Loss', value: formatCurrency(metrics.avgLoss, { sign: true }), colorClass: 'text-research-bear' },
    { label: 'Avg Hold', value: `${metrics.avgHoldingHours.toFixed(1)}h` },

    { group: 'Distribution', label: 'Skewness', value: formatRatio(metrics.skewness, 3), colorClass: metrics.skewness > 0 ? 'text-research-bull' : 'text-research-bear' },
    { label: 'Kurtosis', value: formatRatio(metrics.kurtosis, 3) },
  ]

  return (
    <div className="overflow-auto">
      <table className="w-full border-collapse text-xs">
        <tbody>
          {rows.map((row, i) => (
            <React.Fragment key={row.label}>
              {row.group && (
                <tr>
                  <td
                    colSpan={2}
                    className={clsx(
                      'text-[10px] font-semibold uppercase tracking-widest text-research-subtle py-1.5 px-3',
                      i > 0 && 'pt-3'
                    )}
                  >
                    {row.group}
                  </td>
                </tr>
              )}
              <tr className="hover:bg-research-muted/20 transition-colors">
                <td className="py-1 px-3 text-research-subtle">{row.label}</td>
                <td className={clsx('py-1 px-3 text-right font-mono font-medium', row.colorClass ?? 'text-research-text')}>
                  {row.value}
                </td>
              </tr>
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  )
}
