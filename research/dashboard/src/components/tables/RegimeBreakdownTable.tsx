import React from 'react'
import type { RegimePerformance } from '@/types/regimes'
import { REGIME_COLORS, REGIME_LABELS } from '@/utils/colors'
import { formatCurrency, formatPct, formatRatio } from '@/utils/formatters'
import { clsx } from 'clsx'

interface RegimeBreakdownTableProps {
  data: RegimePerformance[]
}

export function RegimeBreakdownTable({ data }: RegimeBreakdownTableProps) {
  const thClass = "text-left text-[10px] font-medium text-research-subtle uppercase tracking-wide py-2 px-3 whitespace-nowrap"
  const tdClass = "py-2 px-3 font-mono text-xs border-b border-research-border/50 whitespace-nowrap"

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-research-border">
            <th className={thClass}>Regime</th>
            <th className={clsx(thClass, 'text-right')}>Trades</th>
            <th className={clsx(thClass, 'text-right')}>Total P&L</th>
            <th className={clsx(thClass, 'text-right')}>Avg P&L</th>
            <th className={clsx(thClass, 'text-right')}>Sharpe</th>
            <th className={clsx(thClass, 'text-right')}>Win %</th>
            <th className={clsx(thClass, 'text-right')}>Max DD</th>
            <th className={clsx(thClass, 'text-right')}>Occurrence</th>
          </tr>
        </thead>
        <tbody>
          {data.map(row => (
            <tr key={row.regime} className="hover:bg-research-muted/20 transition-colors">
              <td className={tdClass}>
                <span
                  className="flex items-center gap-2"
                >
                  <span
                    className="w-2.5 h-2.5 rounded-sm"
                    style={{ backgroundColor: REGIME_COLORS[row.regime] }}
                  />
                  <span style={{ color: REGIME_COLORS[row.regime] }}>
                    {REGIME_LABELS[row.regime]}
                  </span>
                </span>
              </td>
              <td className={clsx(tdClass, 'text-right text-research-text')}>{row.tradeCount}</td>
              <td className={clsx(tdClass, 'text-right', row.totalPnl >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                {formatCurrency(row.totalPnl, { compact: true, sign: true })}
              </td>
              <td className={clsx(tdClass, 'text-right', row.avgPnl >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                {formatCurrency(row.avgPnl, { sign: true })}
              </td>
              <td className={clsx(tdClass, 'text-right', row.sharpe >= 1 ? 'text-research-bull' : row.sharpe < 0 ? 'text-research-bear' : 'text-research-text')}>
                {formatRatio(row.sharpe)}
              </td>
              <td className={clsx(tdClass, 'text-right', row.winRate >= 0.5 ? 'text-research-bull' : 'text-research-bear')}>
                {formatPct(row.winRate * 100)}
              </td>
              <td className={clsx(tdClass, 'text-right text-research-bear')}>
                {formatPct(row.maxDrawdown * 100)}
              </td>
              <td className={clsx(tdClass, 'text-right text-research-subtle')}>
                {formatPct(row.occurrencePct * 100)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
