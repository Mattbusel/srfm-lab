// ============================================================
// TradeHistory — paginated historical trades with analytics
// ============================================================
import React, { useState, useMemo, useCallback } from 'react'
import { format } from 'date-fns'
import { usePortfolioStore } from '@/store/portfolioStore'
import type { HistoricalTrade } from '@/types'

const PAGE_SIZE = 25

interface TradeHistoryProps {
  className?: string
}

function exportToCsv(trades: HistoricalTrade[]) {
  const headers = ['Date', 'Symbol', 'Side', 'Qty', 'Entry', 'Exit', 'P&L', 'P&L%', 'Hold (hrs)']
  const rows = trades.map((t) => [
    format(new Date(t.entryTime), 'yyyy-MM-dd HH:mm'),
    t.symbol,
    t.side,
    t.qty.toString(),
    t.entryPrice.toFixed(4),
    t.exitPrice.toFixed(4),
    t.pnl.toFixed(2),
    (t.pnlPct * 100).toFixed(2) + '%',
    (t.holdingPeriod / 3600).toFixed(1),
  ])

  const csv = [headers, ...rows].map((r) => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `trade-history-${format(new Date(), 'yyyy-MM-dd')}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export const TradeHistory: React.FC<TradeHistoryProps> = ({ className = '' }) => {
  const tradeHistory = usePortfolioStore((s) => s.tradeHistory)
  const refreshTrades = usePortfolioStore((s) => s.refreshTradeHistory)

  const [page, setPage] = useState(0)
  const [symbolFilter, setSymbolFilter] = useState('')
  const [sideFilter, setSideFilter] = useState<'all' | 'buy' | 'sell'>('all')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [sortKey, setSortKey] = useState<keyof HistoricalTrade>('entryTime')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  // Filter & sort
  const filtered = useMemo(() => {
    let data = [...tradeHistory]

    if (symbolFilter) data = data.filter((t) => t.symbol.includes(symbolFilter.toUpperCase()))
    if (sideFilter !== 'all') data = data.filter((t) => t.side === sideFilter)
    if (startDate) data = data.filter((t) => t.entryTime >= new Date(startDate).getTime())
    if (endDate) data = data.filter((t) => t.entryTime <= new Date(endDate).getTime() + 86400000)

    data.sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      let cmp = 0
      if (typeof av === 'number' && typeof bv === 'number') cmp = av - bv
      else if (typeof av === 'string' && typeof bv === 'string') cmp = av.localeCompare(bv)
      return sortDir === 'asc' ? cmp : -cmp
    })

    return data
  }, [tradeHistory, symbolFilter, sideFilter, startDate, endDate, sortKey, sortDir])

  // Analytics
  const analytics = useMemo(() => {
    if (filtered.length === 0) return null
    const wins = filtered.filter((t) => t.pnl > 0)
    const losses = filtered.filter((t) => t.pnl < 0)
    const totalPnl = filtered.reduce((s, t) => s + t.pnl, 0)
    const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0
    const avgLoss = losses.length > 0 ? losses.reduce((s, t) => s + t.pnl, 0) / losses.length : 0
    const profitFactor = Math.abs(avgLoss) > 0 ? (avgWin * wins.length) / (Math.abs(avgLoss) * losses.length) : Infinity
    const best = filtered.reduce((m, t) => t.pnl > m.pnl ? t : m, filtered[0])
    const worst = filtered.reduce((m, t) => t.pnl < m.pnl ? t : m, filtered[0])

    return {
      total: filtered.length,
      wins: wins.length,
      losses: losses.length,
      winRate: filtered.length > 0 ? wins.length / filtered.length : 0,
      totalPnl,
      avgWin,
      avgLoss,
      profitFactor,
      best,
      worst,
      avgHold: filtered.reduce((s, t) => s + t.holdingPeriod, 0) / filtered.length / 3600,
    }
  }, [filtered])

  const page_count = Math.ceil(filtered.length / PAGE_SIZE)
  const page_data = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const handleSort = useCallback((key: keyof HistoricalTrade) => {
    setSortKey((prev) => {
      setSortDir((prevDir) => prev === key ? (prevDir === 'asc' ? 'desc' : 'asc') : 'desc')
      return key
    })
  }, [])

  // Generate mock trades if none
  const displayTrades = page_data.length > 0 ? page_data : []

  return (
    <div className={`flex flex-col bg-terminal-bg h-full ${className}`}>
      {/* Filters */}
      <div className="px-3 py-2 border-b border-terminal-border flex-shrink-0 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-terminal-subtle text-xs font-mono uppercase">Trade History</span>
          <div className="flex gap-2">
            <button
              onClick={() => exportToCsv(filtered)}
              className="text-[10px] font-mono text-terminal-accent hover:underline"
            >
              Export CSV
            </button>
            <button
              onClick={() => refreshTrades({ startDate, endDate, symbol: symbolFilter || undefined })}
              className="text-[10px] font-mono text-terminal-subtle hover:text-terminal-text"
            >
              Refresh
            </button>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          <input
            type="text"
            placeholder="Symbol..."
            value={symbolFilter}
            onChange={(e) => { setSymbolFilter(e.target.value.toUpperCase()); setPage(0) }}
            className="w-20 bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-[10px] font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
          />
          <div className="flex gap-1">
            {(['all', 'buy', 'sell'] as const).map((s) => (
              <button key={s} onClick={() => { setSideFilter(s); setPage(0) }}
                className={`text-[10px] font-mono px-2 py-1 rounded capitalize transition-colors ${sideFilter === s ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text'}`}
              >{s}</button>
            ))}
          </div>
          <input type="date" value={startDate} onChange={(e) => { setStartDate(e.target.value); setPage(0) }}
            className="bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-[10px] font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
          />
          <input type="date" value={endDate} onChange={(e) => { setEndDate(e.target.value); setPage(0) }}
            className="bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-[10px] font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
          />
        </div>
      </div>

      {/* Analytics summary */}
      {analytics && (
        <div className="px-3 py-2 border-b border-terminal-border bg-terminal-surface/30 flex-shrink-0">
          <div className="flex flex-wrap gap-4 text-xs font-mono">
            <div>
              <span className="text-terminal-subtle">Trades: </span>
              <span className="text-terminal-text">{analytics.total}</span>
            </div>
            <div>
              <span className="text-terminal-subtle">Win Rate: </span>
              <span className={analytics.winRate >= 0.5 ? 'text-terminal-bull' : 'text-terminal-bear'}>
                {(analytics.winRate * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="text-terminal-subtle">Total P&L: </span>
              <span className={analytics.totalPnl >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}>
                {analytics.totalPnl >= 0 ? '+' : ''}${analytics.totalPnl.toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-terminal-subtle">Profit Factor: </span>
              <span className="text-terminal-text">
                {isFinite(analytics.profitFactor) ? analytics.profitFactor.toFixed(2) : '∞'}
              </span>
            </div>
            <div>
              <span className="text-terminal-subtle">Avg Hold: </span>
              <span className="text-terminal-text">{analytics.avgHold.toFixed(1)}h</span>
            </div>
            <div>
              <span className="text-terminal-subtle">Best: </span>
              <span className="text-terminal-bull">${analytics.best?.pnl.toFixed(2)}</span>
            </div>
            <div>
              <span className="text-terminal-subtle">Worst: </span>
              <span className="text-terminal-bear">${analytics.worst?.pnl.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-terminal-surface border-b border-terminal-border z-10">
            <tr>
              {[
                { key: 'entryTime', label: 'Date' },
                { key: 'symbol', label: 'Symbol' },
                { key: 'side', label: 'Side' },
                { key: 'qty', label: 'Qty' },
                { key: 'entryPrice', label: 'Entry' },
                { key: 'exitPrice', label: 'Exit' },
                { key: 'pnl', label: 'P&L' },
                { key: 'pnlPct', label: 'P&L%' },
                { key: 'holdingPeriod', label: 'Hold' },
              ].map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key as keyof HistoricalTrade)}
                  className="px-2 py-1.5 text-left font-mono text-[10px] text-terminal-subtle uppercase cursor-pointer hover:text-terminal-text"
                >
                  {col.label}
                  {sortKey === col.key && <span className="ml-0.5">{sortDir === 'asc' ? '↑' : '↓'}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayTrades.map((trade, i) => (
              <tr
                key={trade.id}
                className={`border-b border-terminal-border/20 hover:bg-terminal-surface transition-colors ${
                  trade.pnl > 0 ? 'bg-terminal-bull/5' : trade.pnl < 0 ? 'bg-terminal-bear/5' : ''
                }`}
              >
                <td className="px-2 py-1 font-mono text-[10px] text-terminal-subtle whitespace-nowrap">
                  {format(new Date(trade.entryTime), 'MM/dd HH:mm')}
                </td>
                <td className="px-2 py-1 font-mono text-xs text-terminal-text">{trade.symbol}</td>
                <td className="px-2 py-1">
                  <span className={`font-mono text-[10px] px-1 rounded ${trade.side === 'buy' ? 'bg-terminal-bull/20 text-terminal-bull' : 'bg-terminal-bear/20 text-terminal-bear'}`}>
                    {trade.side.toUpperCase()}
                  </span>
                </td>
                <td className="px-2 py-1 font-mono text-xs text-terminal-text text-right">{trade.qty.toFixed(0)}</td>
                <td className="px-2 py-1 font-mono text-xs text-terminal-subtle text-right">{trade.entryPrice.toFixed(2)}</td>
                <td className="px-2 py-1 font-mono text-xs text-terminal-subtle text-right">{trade.exitPrice.toFixed(2)}</td>
                <td className={`px-2 py-1 font-mono text-xs text-right font-medium ${trade.pnl >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
                  {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                </td>
                <td className={`px-2 py-1 font-mono text-[10px] text-right ${trade.pnlPct >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
                  {trade.pnlPct >= 0 ? '+' : ''}{(trade.pnlPct * 100).toFixed(2)}%
                </td>
                <td className="px-2 py-1 font-mono text-[10px] text-terminal-subtle text-right">
                  {(trade.holdingPeriod / 3600).toFixed(1)}h
                </td>
              </tr>
            ))}
            {displayTrades.length === 0 && (
              <tr>
                <td colSpan={9} className="text-center py-8 text-terminal-subtle text-xs">
                  No trades found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {page_count > 1 && (
        <div className="flex items-center justify-between px-3 py-2 border-t border-terminal-border flex-shrink-0">
          <span className="text-[10px] font-mono text-terminal-subtle">
            {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filtered.length)} of {filtered.length}
          </span>
          <div className="flex gap-1">
            <button
              onClick={() => setPage(0)}
              disabled={page === 0}
              className="text-[10px] font-mono px-2 py-1 rounded disabled:opacity-30 text-terminal-subtle hover:text-terminal-text"
            >
              ««
            </button>
            <button
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="text-[10px] font-mono px-2 py-1 rounded disabled:opacity-30 text-terminal-subtle hover:text-terminal-text"
            >
              ‹ Prev
            </button>
            <span className="text-[10px] font-mono px-2 py-1 text-terminal-text">
              {page + 1}/{page_count}
            </span>
            <button
              onClick={() => setPage(p => Math.min(page_count - 1, p + 1))}
              disabled={page >= page_count - 1}
              className="text-[10px] font-mono px-2 py-1 rounded disabled:opacity-30 text-terminal-subtle hover:text-terminal-text"
            >
              Next ›
            </button>
            <button
              onClick={() => setPage(page_count - 1)}
              disabled={page >= page_count - 1}
              className="text-[10px] font-mono px-2 py-1 rounded disabled:opacity-30 text-terminal-subtle hover:text-terminal-text"
            >
              »»
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default TradeHistory
