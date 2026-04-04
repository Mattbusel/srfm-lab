// ============================================================
// OrderBookChart — L2 order book depth visualization
// ============================================================
import React, { useMemo, useRef, useEffect } from 'react'
import { useOrderBook } from '@/hooks/useOrderBook'
import { useMarketStore } from '@/store/marketStore'
import type { OrderBookLevel, Trade } from '@/types'
import { motion, AnimatePresence } from 'framer-motion'

interface OrderBookChartProps {
  symbol: string
  levels?: number
  onPriceClick?: (price: number) => void
  showTape?: boolean
  className?: string
}

const formatPrice = (p: number) => p.toFixed(p > 100 ? 2 : 4)
const formatSize = (s: number) => s >= 10000 ? `${(s / 1000).toFixed(1)}K` : s.toFixed(0)
const formatTime = (ts: number) => new Date(ts).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })

const BAR_MAX_WIDTH = '100%'

function BookRow({
  level,
  side,
  maxSize,
  onClick,
}: {
  level: OrderBookLevel
  side: 'bid' | 'ask'
  maxSize: number
  onClick?: (price: number) => void
}) {
  const barWidth = maxSize > 0 ? (level.size / maxSize) * 100 : 0
  const isLarge = level.size > maxSize * 0.3

  return (
    <div
      className={`relative flex items-center h-5 cursor-pointer hover:bg-terminal-muted/20 transition-colors group`}
      onClick={() => onClick?.(level.price)}
    >
      {/* Background bar */}
      <div
        className={`absolute top-0 ${side === 'bid' ? 'right-0' : 'left-0'} h-full transition-all duration-300 ${
          side === 'bid' ? 'bg-terminal-bull/15' : 'bg-terminal-bear/15'
        }`}
        style={{ width: `${barWidth}%` }}
      />

      {/* Content */}
      {side === 'ask' ? (
        <div className="relative flex items-center justify-between w-full px-1.5 z-10">
          <span className={`font-mono text-xs ${isLarge ? 'text-terminal-bear font-semibold' : 'text-terminal-bear'}`}>
            {formatPrice(level.price)}
          </span>
          <span className="font-mono text-xs text-terminal-text">{formatSize(level.size)}</span>
          <span className="font-mono text-xs text-terminal-subtle">{formatSize(level.total)}</span>
        </div>
      ) : (
        <div className="relative flex items-center justify-between w-full px-1.5 z-10">
          <span className="font-mono text-xs text-terminal-subtle">{formatSize(level.total)}</span>
          <span className="font-mono text-xs text-terminal-text">{formatSize(level.size)}</span>
          <span className={`font-mono text-xs ${isLarge ? 'text-terminal-bull font-semibold' : 'text-terminal-bull'}`}>
            {formatPrice(level.price)}
          </span>
        </div>
      )}
    </div>
  )
}

function TradeTapeRow({ trade }: { trade: Trade }) {
  const isLarge = trade.size > 1000
  return (
    <motion.div
      initial={{ opacity: 0, x: -4 }}
      animate={{ opacity: 1, x: 0 }}
      className={`flex items-center justify-between px-2 py-0.5 text-xs font-mono ${
        isLarge ? 'bg-terminal-muted/30' : ''
      }`}
    >
      <span className="text-terminal-subtle text-[10px]">{formatTime(trade.timestamp)}</span>
      <span className={trade.side === 'buy' ? 'text-terminal-bull' : 'text-terminal-bear'}>
        {formatPrice(trade.price)}
      </span>
      <span className={`${isLarge ? 'text-terminal-warning font-semibold' : 'text-terminal-text'}`}>
        {formatSize(trade.size)}
        {isLarge && ' 🐋'}
      </span>
    </motion.div>
  )
}

export const OrderBookChart: React.FC<OrderBookChartProps> = ({
  symbol,
  levels = 10,
  onPriceClick,
  showTape = true,
  className = '',
}) => {
  const book = useOrderBook(symbol, levels)
  const trades = useMarketStore((s) => s.recentTrades[symbol] ?? [])
  const tapeRef = useRef<HTMLDivElement>(null)

  // Auto-scroll tape
  useEffect(() => {
    if (tapeRef.current) {
      tapeRef.current.scrollTop = 0
    }
  }, [trades.length])

  const maxBidSize = useMemo(
    () => Math.max(...(book?.bids.map((l) => l.size) ?? [1]), 1),
    [book?.bids]
  )
  const maxAskSize = useMemo(
    () => Math.max(...(book?.asks.map((l) => l.size) ?? [1]), 1),
    [book?.asks]
  )

  const imbalancePct = book ? book.imbalance * 100 : 50

  if (!book) {
    return (
      <div className={`flex items-center justify-center bg-terminal-bg ${className}`}>
        <div className="text-terminal-subtle text-sm">Loading order book...</div>
      </div>
    )
  }

  return (
    <div className={`flex flex-col bg-terminal-bg h-full ${className}`}>
      {/* Header */}
      <div className="px-3 py-2 border-b border-terminal-border flex-shrink-0">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-terminal-subtle text-xs font-mono">Order Book</span>
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="text-terminal-subtle">Spread:</span>
            <span className="text-terminal-text">{book.spread.toFixed(4)}</span>
            <span className="text-terminal-subtle">({book.spreadBps.toFixed(1)}bps)</span>
          </div>
        </div>

        {/* Imbalance meter */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[10px] font-mono">
            <span className="text-terminal-bull">Bids {imbalancePct.toFixed(0)}%</span>
            <span className="text-terminal-subtle">Imbalance</span>
            <span className="text-terminal-bear">Asks {(100 - imbalancePct).toFixed(0)}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-terminal-muted overflow-hidden">
            <div
              className="h-full transition-all duration-500 rounded-full"
              style={{
                width: `${imbalancePct}%`,
                background: imbalancePct > 55
                  ? '#22c55e'
                  : imbalancePct < 45
                  ? '#ef4444'
                  : '#6b7280',
              }}
            />
          </div>
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center justify-between px-1.5 py-1 text-[10px] font-mono text-terminal-subtle border-b border-terminal-border/50 flex-shrink-0">
        <div className="flex-1 grid grid-cols-3 gap-1">
          <span>Total</span>
          <span className="text-center">Size</span>
          <span className="text-right text-terminal-bull">Bid</span>
        </div>
        <div className="w-px bg-terminal-border mx-1 h-3" />
        <div className="flex-1 grid grid-cols-3 gap-1">
          <span className="text-terminal-bear">Ask</span>
          <span className="text-center">Size</span>
          <span className="text-right">Total</span>
        </div>
      </div>

      {/* Order book rows — bid/ask side by side */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <div className="flex h-full">
          {/* Bids */}
          <div className="flex-1 flex flex-col">
            {book.bids.slice(0, levels).map((level, i) => (
              <BookRow
                key={`bid-${i}-${level.price}`}
                level={level}
                side="bid"
                maxSize={maxBidSize}
                onClick={onPriceClick}
              />
            ))}
          </div>

          {/* Mid price separator */}
          <div className="w-px bg-terminal-border flex-shrink-0 flex flex-col items-center justify-start pt-2">
            <div className="w-px flex-1 bg-terminal-border" />
          </div>

          {/* Asks */}
          <div className="flex-1 flex flex-col">
            {book.asks.slice(0, levels).map((level, i) => (
              <BookRow
                key={`ask-${i}-${level.price}`}
                level={level}
                side="ask"
                maxSize={maxAskSize}
                onClick={onPriceClick}
              />
            ))}
          </div>
        </div>

        {/* Mid price */}
        <div className="border-t border-terminal-border bg-terminal-surface px-2 py-1 flex items-center justify-center gap-2">
          <span className="text-terminal-text font-mono text-sm font-semibold">
            {formatPrice(book.midPrice)}
          </span>
          <span className="text-terminal-subtle text-xs font-mono">MID</span>
        </div>
      </div>

      {/* Trades Tape */}
      {showTape && (
        <div className="flex-shrink-0 border-t border-terminal-border">
          <div className="flex items-center justify-between px-2 py-1 border-b border-terminal-border/50">
            <span className="text-[10px] font-mono text-terminal-subtle">Time &nbsp; Trades</span>
            <div className="flex gap-3 text-[10px] font-mono text-terminal-subtle">
              <span>Price</span>
              <span>Size</span>
            </div>
          </div>
          <div ref={tapeRef} className="h-32 overflow-y-auto">
            <AnimatePresence initial={false}>
              {trades.slice(0, 30).map((trade) => (
                <TradeTapeRow key={trade.id} trade={trade} />
              ))}
            </AnimatePresence>
            {trades.length === 0 && (
              <div className="flex items-center justify-center h-full text-terminal-subtle text-xs">
                Waiting for trades...
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default OrderBookChart
