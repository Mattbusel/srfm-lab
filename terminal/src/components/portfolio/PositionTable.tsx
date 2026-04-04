// ============================================================
// PositionTable — full positions grid
// ============================================================
import React, { useState, useCallback, useMemo } from 'react'
import { motion } from 'framer-motion'
import { usePortfolioStore, selectOpenPositions } from '@/store/portfolioStore'
import { useMarketStore } from '@/store/marketStore'
import type { Position } from '@/types'

interface ColumnDef {
  key: keyof Position | 'pnlCalc' | 'actions'
  label: string
  sortable?: boolean
  align?: 'left' | 'right' | 'center'
  width?: string
  render: (pos: Position, ctx: { calcPrice: number; onClose: (symbol: string, qty?: number) => void }) => React.ReactNode
}

const fmt = {
  price: (v: number) => v.toFixed(v > 100 ? 2 : 4),
  pct: (v: number) => (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%',
  currency: (v: number) => (v >= 0 ? '+' : '') + '$' + Math.abs(v).toFixed(2),
  size: (v: number) => v.toFixed(2),
  weight: (v: number) => (v * 100).toFixed(1) + '%',
}

const COLUMNS: ColumnDef[] = [
  {
    key: 'symbol',
    label: 'Symbol',
    sortable: true,
    width: 'w-16',
    render: (pos) => (
      <span className="font-mono text-xs font-semibold text-terminal-text">{pos.symbol}</span>
    ),
  },
  {
    key: 'side',
    label: 'Side',
    sortable: true,
    align: 'center',
    width: 'w-10',
    render: (pos) => (
      <span className={`text-[10px] font-mono rounded px-1 ${pos.side === 'long' ? 'bg-terminal-bull/20 text-terminal-bull' : 'bg-terminal-bear/20 text-terminal-bear'}`}>
        {pos.side.toUpperCase()}
      </span>
    ),
  },
  {
    key: 'qty',
    label: 'Qty',
    sortable: true,
    align: 'right',
    render: (pos) => <span className="font-mono text-xs text-terminal-text">{fmt.size(pos.qty)}</span>,
  },
  {
    key: 'entryPrice',
    label: 'Entry',
    sortable: true,
    align: 'right',
    render: (pos) => <span className="font-mono text-xs text-terminal-subtle">{fmt.price(pos.entryPrice)}</span>,
  },
  {
    key: 'currentPrice',
    label: 'Current',
    sortable: true,
    align: 'right',
    render: (pos) => <span className="font-mono text-xs text-terminal-text">{fmt.price(pos.currentPrice)}</span>,
  },
  {
    key: 'unrealizedPnl',
    label: 'Unreal P&L',
    sortable: true,
    align: 'right',
    render: (pos) => (
      <div className="text-right">
        <div className={`font-mono text-xs font-medium ${pos.unrealizedPnl >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
          {fmt.currency(pos.unrealizedPnl)}
        </div>
        <div className={`font-mono text-[10px] ${pos.unrealizedPnlPct >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
          {fmt.pct(pos.unrealizedPnlPct)}
        </div>
      </div>
    ),
  },
  {
    key: 'marketValue',
    label: 'Mkt Value',
    sortable: true,
    align: 'right',
    render: (pos) => <span className="font-mono text-xs text-terminal-text">${pos.marketValue.toFixed(0)}</span>,
  },
  {
    key: 'weight',
    label: 'Weight',
    sortable: true,
    align: 'right',
    render: (pos) => (
      <div className="flex items-center gap-1 justify-end">
        <div className="w-10 h-1.5 bg-terminal-muted rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${pos.unrealizedPnl >= 0 ? 'bg-terminal-bull' : 'bg-terminal-bear'}`}
            style={{ width: `${Math.min(pos.weight * 10, 100)}%` }}
          />
        </div>
        <span className="font-mono text-[10px] text-terminal-subtle">{fmt.weight(pos.weight)}</span>
      </div>
    ),
  },
  {
    key: 'pnlCalc',
    label: 'If Price =',
    align: 'right',
    render: (pos, { calcPrice }) => {
      if (!calcPrice || calcPrice <= 0) return null
      const hypothetical = (calcPrice - pos.entryPrice) * pos.qty * (pos.side === 'long' ? 1 : -1)
      return (
        <span className={`font-mono text-[10px] ${hypothetical >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
          {fmt.currency(hypothetical)}
        </span>
      )
    },
  },
  {
    key: 'actions',
    label: '',
    align: 'center',
    width: 'w-16',
    render: (pos, { onClose }) => (
      <button
        onClick={(e) => { e.stopPropagation(); onClose(pos.symbol) }}
        className="text-[10px] font-mono px-2 py-0.5 rounded text-terminal-bear border border-terminal-bear/30 hover:bg-terminal-bear/20 transition-colors"
      >
        Close
      </button>
    ),
  },
]

interface PositionTableProps {
  onSelectPosition?: (symbol: string) => void
  className?: string
}

export const PositionTable: React.FC<PositionTableProps> = ({ onSelectPosition, className = '' }) => {
  const positions = usePortfolioStore(selectOpenPositions)
  const account = usePortfolioStore((s) => s.account)
  const submitOrder = usePortfolioStore((s) => s.submitOrder)
  const setSelectedSymbol = useMarketStore((s) => s.setSelectedSymbol)

  const [sortKey, setSortKey] = useState<keyof Position>('symbol')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')
  const [calcPrice, setCalcPrice] = useState<number>(0)
  const [selectedSymbol, setSelected] = useState<string | null>(null)
  const [closingSymbol, setClosingSymbol] = useState<string | null>(null)
  const [showCloseConfirm, setShowCloseConfirm] = useState<{ symbol: string; qty: number } | null>(null)

  const handleSort = useCallback((key: keyof Position) => {
    setSortKey((prev) => {
      setSortDir((prevDir) => prev === key ? (prevDir === 'asc' ? 'desc' : 'asc') : 'asc')
      return key
    })
  }, [])

  const handleClose = useCallback((symbol: string, qty?: number) => {
    const pos = positions.find((p) => p.symbol === symbol)
    if (!pos) return
    setShowCloseConfirm({ symbol, qty: qty ?? pos.qty })
  }, [positions])

  const confirmClose = useCallback(async () => {
    if (!showCloseConfirm) return
    const pos = positions.find((p) => p.symbol === showCloseConfirm.symbol)
    if (!pos) return

    setClosingSymbol(showCloseConfirm.symbol)
    setShowCloseConfirm(null)

    await submitOrder({
      symbol: pos.symbol,
      side: pos.side === 'long' ? 'sell' : 'buy',
      qty: showCloseConfirm.qty,
      type: 'market',
      timeInForce: 'day',
    })

    setClosingSymbol(null)
  }, [showCloseConfirm, positions, submitOrder])

  const sortedPositions = useMemo(() => {
    return [...positions].sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      let cmp = 0
      if (typeof av === 'string' && typeof bv === 'string') {
        cmp = av.localeCompare(bv)
      } else if (typeof av === 'number' && typeof bv === 'number') {
        cmp = av - bv
      }
      return sortDir === 'asc' ? cmp : -cmp
    })
  }, [positions, sortKey, sortDir])

  const totalUnrealizedPnl = positions.reduce((s, p) => s + p.unrealizedPnl, 0)
  const totalMarketValue = positions.reduce((s, p) => s + p.marketValue, 0)

  if (positions.length === 0) {
    return (
      <div className={`flex flex-col h-full bg-terminal-bg ${className}`}>
        <div className="px-3 py-2 border-b border-terminal-border">
          <span className="text-terminal-subtle text-xs font-mono uppercase">Positions</span>
        </div>
        <div className="flex-1 flex items-center justify-center text-terminal-subtle text-sm">
          No open positions
        </div>
      </div>
    )
  }

  return (
    <div className={`flex flex-col h-full bg-terminal-bg ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border flex-shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-terminal-subtle text-xs font-mono uppercase">Positions</span>
          <span className="text-terminal-subtle text-[10px] font-mono">{positions.length} open</span>
        </div>
        <div className="flex items-center gap-3">
          {/* What-if calculator */}
          <div className="flex items-center gap-1">
            <span className="text-[10px] font-mono text-terminal-subtle">If price =</span>
            <input
              type="number"
              value={calcPrice || ''}
              onChange={(e) => setCalcPrice(Number(e.target.value))}
              placeholder="0.00"
              className="w-20 bg-terminal-surface border border-terminal-border rounded px-1.5 py-0.5 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
            />
          </div>
          <div className="flex items-center gap-2">
            <div>
              <span className="text-[10px] font-mono text-terminal-subtle">Total P&L: </span>
              <span className={`font-mono text-xs font-medium ${totalUnrealizedPnl >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
                ${totalUnrealizedPnl >= 0 ? '+' : ''}{totalUnrealizedPnl.toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-[10px] font-mono text-terminal-subtle">Mkt: </span>
              <span className="font-mono text-xs text-terminal-text">${totalMarketValue.toFixed(0)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-terminal-surface border-b border-terminal-border z-10">
            <tr>
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  className={`px-2 py-1.5 font-mono text-terminal-subtle text-[10px] uppercase tracking-wider ${
                    col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'
                  } ${col.sortable ? 'cursor-pointer hover:text-terminal-text' : ''} ${col.width ?? ''}`}
                  onClick={() => col.sortable && handleSort(col.key as keyof Position)}
                >
                  {col.label}
                  {col.sortable && sortKey === col.key && (
                    <span className="ml-0.5">{sortDir === 'asc' ? '↑' : '↓'}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedPositions.map((pos) => {
              const isSelected = selectedSymbol === pos.symbol
              const isClosing = closingSymbol === pos.symbol
              const isProfitable = pos.unrealizedPnl > 0
              const isLosing = pos.unrealizedPnl < 0

              return (
                <motion.tr
                  key={pos.symbol}
                  layout
                  onClick={() => {
                    setSelected(pos.symbol)
                    setSelectedSymbol(pos.symbol)
                    onSelectPosition?.(pos.symbol)
                  }}
                  className={`border-b border-terminal-border/30 cursor-pointer transition-all ${
                    isSelected ? 'bg-terminal-accent/10' : ''
                  } ${isProfitable ? 'hover:bg-terminal-bull/5' : isLosing ? 'hover:bg-terminal-bear/5' : 'hover:bg-terminal-surface'} ${
                    isClosing ? 'opacity-50' : ''
                  }`}
                >
                  {COLUMNS.map((col) => (
                    <td
                      key={col.key}
                      className={`px-2 py-1.5 ${
                        col.align === 'right' ? 'text-right' : col.align === 'center' ? 'text-center' : 'text-left'
                      } ${col.width ?? ''}`}
                    >
                      {col.render(pos, { calcPrice, onClose: handleClose })}
                    </td>
                  ))}
                </motion.tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Close confirmation modal */}
      {showCloseConfirm && (
        <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-terminal-surface border border-terminal-border rounded-lg p-4 shadow-xl max-w-sm w-full mx-4">
            <h3 className="text-terminal-text font-semibold text-sm mb-2">Confirm Close Position</h3>
            <p className="text-terminal-subtle text-xs mb-4">
              Close {showCloseConfirm.qty} shares of {showCloseConfirm.symbol} at market price?
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setShowCloseConfirm(null)}
                className="px-3 py-1.5 text-xs font-mono text-terminal-subtle border border-terminal-border rounded hover:text-terminal-text transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmClose}
                className="px-3 py-1.5 text-xs font-mono bg-terminal-bear text-white rounded hover:bg-terminal-bear/80 transition-colors"
              >
                Close Position
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default PositionTable
