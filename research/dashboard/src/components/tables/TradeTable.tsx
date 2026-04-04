import React, { useState, useMemo } from 'react'
import { clsx } from 'clsx'
import { ChevronUp, ChevronDown } from 'lucide-react'
import type { Trade } from '@/types/trades'
import { formatCurrency, formatPct, formatDate, formatDuration, pnlColor } from '@/utils/formatters'
import { REGIME_COLORS, REGIME_LABELS } from '@/utils/colors'

interface TradeTableProps {
  trades: Trade[]
  maxRows?: number
  compact?: boolean
}

type SortKey = keyof Trade
type SortDir = 'asc' | 'desc'

export function TradeTable({ trades, maxRows, compact = false }: TradeTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('entryTime')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [page, setPage] = useState(0)
  const perPage = maxRows ?? 20

  const sorted = useMemo(() => {
    return [...trades].sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (av === null || av === undefined) return 1
      if (bv === null || bv === undefined) return -1
      if (typeof av === 'string' && typeof bv === 'string') {
        return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
      }
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortDir === 'asc' ? av - bv : bv - av
      }
      return 0
    })
  }, [trades, sortKey, sortDir])

  const paged = sorted.slice(page * perPage, (page + 1) * perPage)
  const totalPages = Math.ceil(sorted.length / perPage)

  const onSort = (key: SortKey) => {
    if (key === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const SortIcon = ({ k }: { k: SortKey }) =>
    sortKey === k ? (sortDir === 'asc' ? <ChevronUp size={10} /> : <ChevronDown size={10} />) : null

  const thClass = "text-left text-xs font-medium text-research-subtle uppercase tracking-wide py-2 px-3 cursor-pointer select-none hover:text-research-text transition-colors whitespace-nowrap"
  const tdClass = clsx("py-2 px-3 font-mono text-xs border-b border-research-border/50 whitespace-nowrap", compact && "py-1")

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-research-border">
            <th className={thClass} onClick={() => onSort('id')}>
              <span className="flex items-center gap-1">ID <SortIcon k="id" /></span>
            </th>
            <th className={thClass} onClick={() => onSort('instrument')}>
              <span className="flex items-center gap-1">Instrument <SortIcon k="instrument" /></span>
            </th>
            <th className={thClass} onClick={() => onSort('side')}>Side</th>
            <th className={thClass} onClick={() => onSort('entryTime')}>
              <span className="flex items-center gap-1">Entry <SortIcon k="entryTime" /></span>
            </th>
            <th className={thClass} onClick={() => onSort('pnl')}>
              <span className="flex items-center gap-1">P&L <SortIcon k="pnl" /></span>
            </th>
            <th className={thClass} onClick={() => onSort('pnlPct')}>P&L%</th>
            <th className={thClass} onClick={() => onSort('slippage')}>Slip</th>
            <th className={thClass} onClick={() => onSort('holdingPeriodHours')}>Hold</th>
            <th className={thClass}>Regime</th>
          </tr>
        </thead>
        <tbody>
          {paged.map(trade => (
            <tr key={trade.id} className="hover:bg-research-muted/30 transition-colors group">
              <td className={tdClass}>
                <span className="text-research-subtle">{trade.id}</span>
              </td>
              <td className={tdClass}>
                <span className="text-research-accent font-semibold">{trade.instrument}</span>
              </td>
              <td className={tdClass}>
                <span className={trade.side === 'long' ? 'text-research-bull' : 'text-research-bear'}>
                  {trade.side.toUpperCase()}
                </span>
              </td>
              <td className={tdClass}>
                <span className="text-research-subtle">{formatDate(trade.entryTime, 'MM-dd HH:mm')}</span>
              </td>
              <td className={clsx(tdClass, pnlColor(trade.pnl))}>
                {formatCurrency(trade.pnl, { sign: true })}
              </td>
              <td className={clsx(tdClass, pnlColor(trade.pnlPct))}>
                {formatPct(trade.pnlPct, { sign: true })}
              </td>
              <td className={tdClass}>
                <span className="text-research-subtle">{trade.slippage.toFixed(1)}</span>
              </td>
              <td className={tdClass}>
                <span className="text-research-subtle">{formatDuration(trade.holdingPeriodHours)}</span>
              </td>
              <td className={tdClass}>
                <span
                  className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                  style={{
                    backgroundColor: `${REGIME_COLORS[trade.regime]}22`,
                    color: REGIME_COLORS[trade.regime],
                    borderColor: `${REGIME_COLORS[trade.regime]}44`,
                    border: '1px solid',
                  }}
                >
                  {REGIME_LABELS[trade.regime]}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {totalPages > 1 && (
        <div className="flex items-center justify-between px-3 py-2 border-t border-research-border">
          <span className="text-xs text-research-subtle font-mono">
            {page * perPage + 1}–{Math.min((page + 1) * perPage, sorted.length)} of {sorted.length}
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-2 py-1 text-xs bg-research-surface border border-research-border rounded disabled:opacity-40 hover:border-research-accent/50 transition-colors"
            >
              Prev
            </button>
            <button
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-2 py-1 text-xs bg-research-surface border border-research-border rounded disabled:opacity-40 hover:border-research-accent/50 transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
