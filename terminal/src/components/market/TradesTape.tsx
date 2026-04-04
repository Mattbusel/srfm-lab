// ============================================================
// TradesTape — scrolling recent trades with whale detection
// ============================================================
import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useMarketStore } from '@/store/marketStore'
import { format } from 'date-fns'
import type { Trade } from '@/types'

interface TradesTapeProps {
  symbol: string
  maxItems?: number
  showFilter?: boolean
  className?: string
}

const WHALE_MULTIPLIER = 5  // trade size > 5x average = whale

function TradeRow({ trade, avgSize }: { trade: Trade; avgSize: number }) {
  const isWhale = trade.size > avgSize * WHALE_MULTIPLIER
  const isBig = trade.size > avgSize * 2

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -4, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }}
      className={`flex items-center justify-between px-2 py-0.5 text-xs font-mono border-b border-terminal-border/20 ${
        isWhale ? 'bg-terminal-warning/10 border-l-2 border-l-terminal-warning' :
        isBig ? 'bg-terminal-muted/10' : ''
      }`}
    >
      <span className="text-terminal-subtle text-[10px] w-16 flex-shrink-0">
        {format(new Date(trade.timestamp), 'HH:mm:ss')}
      </span>
      <span className={`font-semibold w-16 text-right flex-shrink-0 ${
        trade.side === 'buy' ? 'text-terminal-bull' : 'text-terminal-bear'
      }`}>
        {trade.price.toFixed(trade.price > 100 ? 2 : 4)}
      </span>
      <div className="flex items-center gap-1 flex-shrink-0">
        <span className={`text-right w-14 ${
          isWhale ? 'text-terminal-warning font-bold' : isBig ? 'text-terminal-text font-medium' : 'text-terminal-subtle'
        }`}>
          {trade.size >= 1000 ? `${(trade.size / 1000).toFixed(1)}K` : trade.size.toFixed(0)}
        </span>
        {isWhale && <span className="text-terminal-warning text-[10px]">🐋</span>}
        {!isWhale && isBig && <span className="text-[10px]">⬆</span>}
        {!isWhale && !isBig && (
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${trade.side === 'buy' ? 'bg-terminal-bull' : 'bg-terminal-bear'}`} />
        )}
      </div>
    </motion.div>
  )
}

export const TradesTape: React.FC<TradesTapeProps> = ({
  symbol,
  maxItems = 50,
  showFilter = true,
  className = '',
}) => {
  const trades = useMarketStore((s) => s.recentTrades[symbol] ?? [])
  const [minSize, setMinSize] = useState(0)
  const [paused, setPaused] = useState(false)
  const [sideFilter, setSideFilter] = useState<'all' | 'buy' | 'sell'>('all')
  const containerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to top (newest)
  useEffect(() => {
    if (!paused && containerRef.current) {
      containerRef.current.scrollTop = 0
    }
  }, [trades.length, paused])

  const avgSize = trades.length > 0
    ? trades.reduce((s, t) => s + t.size, 0) / trades.length
    : 100

  const filteredTrades = trades
    .filter((t) => t.size >= minSize)
    .filter((t) => sideFilter === 'all' || t.side === sideFilter)
    .slice(0, maxItems)

  // Stats
  const totalBuyVol = trades.filter((t) => t.side === 'buy').reduce((s, t) => s + t.size, 0)
  const totalSellVol = trades.filter((t) => t.side === 'sell').reduce((s, t) => s + t.size, 0)
  const totalVol = totalBuyVol + totalSellVol
  const buySidePct = totalVol > 0 ? (totalBuyVol / totalVol) * 100 : 50

  return (
    <div className={`flex flex-col bg-terminal-bg h-full ${className}`}>
      {/* Header */}
      <div className="px-2 py-2 border-b border-terminal-border flex-shrink-0">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-terminal-subtle text-xs font-mono">Trades Tape</span>
          <button
            onClick={() => setPaused(!paused)}
            className={`text-[10px] font-mono px-2 py-0.5 rounded ${
              paused ? 'bg-terminal-warning/20 text-terminal-warning' : 'text-terminal-subtle hover:text-terminal-text'
            }`}
          >
            {paused ? '▶ Resume' : '⏸ Pause'}
          </button>
        </div>

        {/* Buy/Sell volume imbalance */}
        <div className="space-y-0.5">
          <div className="flex justify-between text-[10px] font-mono">
            <span className="text-terminal-bull">Buy {buySidePct.toFixed(0)}%</span>
            <span className="text-terminal-bear">{(100 - buySidePct).toFixed(0)}% Sell</span>
          </div>
          <div className="h-1 rounded-full bg-terminal-muted overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-terminal-bull to-terminal-bear transition-all duration-500"
              style={{ width: `${buySidePct}%` }}
            />
          </div>
        </div>

        {showFilter && (
          <div className="flex items-center gap-2 mt-1.5">
            {/* Side filter */}
            <div className="flex gap-1">
              {(['all', 'buy', 'sell'] as const).map((side) => (
                <button
                  key={side}
                  onClick={() => setSideFilter(side)}
                  className={`text-[10px] font-mono px-1.5 py-0.5 rounded capitalize transition-colors ${
                    sideFilter === side
                      ? side === 'buy' ? 'bg-terminal-bull/20 text-terminal-bull'
                        : side === 'sell' ? 'bg-terminal-bear/20 text-terminal-bear'
                        : 'bg-terminal-accent/20 text-terminal-accent'
                      : 'text-terminal-subtle hover:text-terminal-text'
                  }`}
                >
                  {side}
                </button>
              ))}
            </div>

            {/* Min size filter */}
            <div className="flex items-center gap-1 ml-auto">
              <span className="text-[10px] font-mono text-terminal-subtle">Min:</span>
              <input
                type="number"
                value={minSize}
                onChange={(e) => setMinSize(Math.max(0, Number(e.target.value)))}
                className="w-14 bg-terminal-surface border border-terminal-border rounded px-1 py-0.5 text-[10px] font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                step={100}
                min={0}
              />
            </div>
          </div>
        )}
      </div>

      {/* Column headers */}
      <div className="flex items-center justify-between px-2 py-0.5 text-[10px] font-mono text-terminal-subtle border-b border-terminal-border/50 flex-shrink-0">
        <span className="w-16">Time</span>
        <span className="w-16 text-right">Price</span>
        <span className="w-20 text-right">Size</span>
      </div>

      {/* Tape */}
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto"
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
      >
        <AnimatePresence initial={false}>
          {filteredTrades.map((trade) => (
            <TradeRow key={trade.id} trade={trade} avgSize={avgSize} />
          ))}
        </AnimatePresence>

        {filteredTrades.length === 0 && (
          <div className="flex items-center justify-center py-8 text-terminal-subtle text-xs">
            {minSize > 0 ? 'No trades above min size' : 'Waiting for trades...'}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-terminal-border px-2 py-1 flex-shrink-0">
        <div className="flex items-center justify-between text-[10px] font-mono text-terminal-subtle">
          <span>{filteredTrades.length} trades</span>
          <span>Avg size: {avgSize.toFixed(0)}</span>
        </div>
      </div>
    </div>
  )
}

export default TradesTape
