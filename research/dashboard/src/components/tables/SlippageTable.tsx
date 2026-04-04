import React from 'react'
import type { SlippageStats } from '@/types/trades'
import { formatPct } from '@/utils/formatters'
import { clsx } from 'clsx'

interface SlippageTableProps {
  data: SlippageStats[]
}

export function SlippageTable({ data }: SlippageTableProps) {
  const sorted = [...data].sort((a, b) => b.avgSlippage - a.avgSlippage)

  const thClass = "text-left text-[10px] font-medium text-research-subtle uppercase tracking-wide py-2 px-3 whitespace-nowrap"
  const tdClass = "py-2 px-3 font-mono text-xs border-b border-research-border/50 whitespace-nowrap"
  const maxSlip = Math.max(...sorted.map(s => s.avgSlippage))

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-research-border">
            <th className={thClass}>Instrument</th>
            <th className={clsx(thClass, 'text-right')}>Avg Slip ($)</th>
            <th className={clsx(thClass, 'text-right')}>Median</th>
            <th className={clsx(thClass, 'text-right')}>P95</th>
            <th className={clsx(thClass, 'text-right')}>Worst</th>
            <th className={clsx(thClass, 'text-right')}>Slip %</th>
            <th className={clsx(thClass, 'text-right')}>Count</th>
            <th className={thClass}>Distribution</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(row => (
            <tr key={row.instrument} className="hover:bg-research-muted/20 transition-colors">
              <td className={clsx(tdClass, 'text-research-accent font-semibold')}>{row.instrument}</td>
              <td className={clsx(tdClass, 'text-right text-research-warning')}>{row.avgSlippage.toFixed(2)}</td>
              <td className={clsx(tdClass, 'text-right text-research-text')}>{row.medianSlippage.toFixed(2)}</td>
              <td className={clsx(tdClass, 'text-right text-research-text')}>{row.p95Slippage.toFixed(2)}</td>
              <td className={clsx(tdClass, 'text-right text-research-bear')}>{row.worstSlippage.toFixed(2)}</td>
              <td className={clsx(tdClass, 'text-right text-research-subtle')}>{formatPct(row.slippagePct * 100, { decimals: 3 })}</td>
              <td className={clsx(tdClass, 'text-right text-research-subtle')}>{row.count}</td>
              <td className={tdClass}>
                <div className="flex items-center gap-1">
                  <div className="h-1.5 rounded bg-research-warning/60" style={{ width: `${(row.avgSlippage / maxSlip) * 64}px`, minWidth: 2 }} />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
