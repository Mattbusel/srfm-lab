// ============================================================
// OrderBook.tsx — Live L2 order book component
// ============================================================
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import { clsx } from 'clsx'

// ---- Types ----

export interface OrderBookLevel {
  price: number
  size: number
  total: number       // cumulative
  orders?: number
}

export interface OrderBookSnapshot {
  symbol: string
  bids: OrderBookLevel[]
  asks: OrderBookLevel[]
  midPrice: number
  spread: number
  spreadBps: number
  lastUpdate: number
}

export interface OrderBookProps {
  symbol?: string
  levels?: number
  onPriceClick?: (price: number, side: 'bid' | 'ask') => void
  flashUpdates?: boolean
  grouping?: number
}

// ---- Mock data generator ----

function generateBook(spot: number, levels: number, grouping: number): OrderBookSnapshot {
  const step = grouping || (spot > 10000 ? 10 : spot > 1000 ? 1 : spot > 100 ? 0.1 : 0.01)
  const atmBid = Math.floor(spot / step) * step
  const atmAsk = atmBid + step

  const bids: OrderBookLevel[] = []
  let bidTotal = 0
  for (let i = 0; i < levels; i++) {
    const price = atmBid - i * step
    const size = Math.random() * 3 + 0.1 + (i === 2 ? 8 : i === 7 ? 12 : 0)
    bidTotal += size
    bids.push({ price, size, total: bidTotal, orders: Math.floor(Math.random() * 8) + 1 })
  }

  const asks: OrderBookLevel[] = []
  let askTotal = 0
  for (let i = 0; i < levels; i++) {
    const price = atmAsk + i * step
    const size = Math.random() * 3 + 0.1 + (i === 3 ? 10 : i === 8 ? 6 : 0)
    askTotal += size
    asks.push({ price, size, total: askTotal, orders: Math.floor(Math.random() * 8) + 1 })
  }

  const spread = atmAsk - atmBid
  return {
    symbol: 'BTCUSDT',
    bids,
    asks,
    midPrice: (atmBid + atmAsk) / 2,
    spread,
    spreadBps: (spread / spot) * 10000,
    lastUpdate: Date.now(),
  }
}

// ---- Flash animation tracker ----

type FlashType = 'up' | 'down' | null

function useFlash(value: number): FlashType {
  const [flash, setFlash] = useState<FlashType>(null)
  const prev = useRef(value)

  useEffect(() => {
    if (value !== prev.current) {
      setFlash(value > prev.current ? 'up' : 'down')
      prev.current = value
      const t = setTimeout(() => setFlash(null), 300)
      return () => clearTimeout(t)
    }
  }, [value])

  return flash
}

// ---- Book row ----

interface BookRowProps {
  level: OrderBookLevel
  maxTotal: number
  side: 'bid' | 'ask'
  isHighlighted: boolean
  showOrders: boolean
  onPriceClick?: (price: number, side: 'bid' | 'ask') => void
  flashUpdates: boolean
}

const BookRow: React.FC<BookRowProps> = ({
  level,
  maxTotal,
  side,
  isHighlighted,
  showOrders,
  onPriceClick,
  flashUpdates,
}) => {
  const barPct = (level.total / maxTotal) * 100
  const flash = useFlash(level.size)
  const isFlashing = flashUpdates && flash

  const formatPrice = (p: number) => {
    if (p >= 10000) return p.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })
    if (p >= 100)   return p.toFixed(2)
    return p.toFixed(4)
  }

  return (
    <div
      onClick={() => onPriceClick?.(level.price, side)}
      className={clsx(
        'relative flex items-center gap-1 px-2 py-[3px] text-[10px] font-mono group',
        'hover:bg-[#13161e] transition-colors',
        onPriceClick && 'cursor-pointer',
        isFlashing === 'up' && 'bg-emerald-950/20',
        isFlashing === 'down' && 'bg-red-950/20',
      )}
    >
      {/* Depth bar */}
      <div
        className="absolute inset-y-0"
        style={{
          width: `${barPct}%`,
          background: side === 'bid' ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)',
          left: side === 'bid' ? 0 : 'auto',
          right: side === 'ask' ? 0 : 'auto',
        }}
      />

      {/* Content */}
      <span className={clsx(
        'flex-1 text-right',
        side === 'bid' ? 'text-emerald-400' : 'text-red-400',
        isHighlighted && 'font-semibold',
      )}>
        {formatPrice(level.price)}
      </span>
      <span className="text-slate-400 w-16 text-right">
        {level.size.toFixed(3)}
      </span>
      <span className="text-slate-600 w-16 text-right">
        {level.total.toFixed(3)}
      </span>
      {showOrders && (
        <span className="text-slate-700 w-6 text-right text-[8px]">
          {level.orders}
        </span>
      )}
    </div>
  )
}

// ---- Spread indicator ----

const SpreadIndicator: React.FC<{ midPrice: number; spread: number; spreadBps: number }> = ({
  midPrice,
  spread,
  spreadBps,
}) => {
  const formatMid = (p: number) => {
    if (p >= 10000) return p.toLocaleString('en-US', { minimumFractionDigits: 1 })
    return p.toFixed(4)
  }

  return (
    <div className="flex items-center justify-between px-2 py-1.5 bg-[#13161e] border-y border-[#1e2130]">
      <span className="text-[9px] font-mono text-slate-600">
        Spread: ${spread.toFixed(2)} ({spreadBps.toFixed(2)} bps)
      </span>
      <span className="text-[11px] font-mono font-bold text-slate-200">
        ${formatMid(midPrice)}
      </span>
      <span className="text-[9px] font-mono text-slate-600">MID</span>
    </div>
  )
}

// ---- Depth summary bar ----

const DepthSummary: React.FC<{ bids: OrderBookLevel[]; asks: OrderBookLevel[] }> = ({ bids, asks }) => {
  const totalBid = bids.reduce((s, l) => s + l.size, 0)
  const totalAsk = asks.reduce((s, l) => s + l.size, 0)
  const total = totalBid + totalAsk
  const bidPct = (totalBid / total) * 100

  return (
    <div className="flex items-center gap-2 px-2 py-1.5 border-b border-[#1e2130]">
      <span className="text-[9px] font-mono text-emerald-400">{totalBid.toFixed(2)}</span>
      <div className="flex-1 h-1.5 bg-[#1e2130] rounded overflow-hidden">
        <div
          className="h-full bg-emerald-500 rounded-l float-left"
          style={{ width: `${bidPct}%` }}
        />
        <div
          className="h-full bg-red-500 rounded-r"
          style={{ width: `${100 - bidPct}%`, float: 'right' }}
        />
      </div>
      <span className="text-[9px] font-mono text-red-400">{totalAsk.toFixed(2)}</span>
    </div>
  )
}

// ---- Main component ----

const SPOT = 63450

export const OrderBook: React.FC<OrderBookProps> = ({
  symbol = 'BTCUSDT',
  levels = 20,
  onPriceClick,
  flashUpdates = true,
  grouping = 10,
}) => {
  const [book, setBook] = useState<OrderBookSnapshot>(() => generateBook(SPOT, levels, grouping))
  const [highlightedBid, setHighlightedBid] = useState<number | null>(null)
  const [highlightedAsk, setHighlightedAsk] = useState<number | null>(null)
  const [showOrders, setShowOrders] = useState(false)
  const [currentGrouping, setCurrentGrouping] = useState(grouping)
  const groupings = SPOT > 10000 ? [1, 5, 10, 50, 100] : SPOT > 1000 ? [0.1, 0.5, 1, 5] : [0.01, 0.05, 0.1, 0.5]

  // Simulate live updates
  useEffect(() => {
    const interval = setInterval(() => {
      setBook(generateBook(SPOT + (Math.random() - 0.5) * 50, levels, currentGrouping))
    }, 800)
    return () => clearInterval(interval)
  }, [levels, currentGrouping])

  const maxTotal = useMemo(
    () => Math.max(
      book.bids[book.bids.length - 1]?.total ?? 0,
      book.asks[book.asks.length - 1]?.total ?? 0,
    ),
    [book],
  )

  const handlePriceClick = useCallback(
    (price: number, side: 'bid' | 'ask') => {
      if (side === 'bid') setHighlightedBid((p) => (p === price ? null : price))
      else setHighlightedAsk((p) => (p === price ? null : price))
      onPriceClick?.(price, side)
    },
    [onPriceClick],
  )

  return (
    <div className="flex flex-col bg-[#0e1017] border border-[#1e2130] rounded-lg overflow-hidden font-mono text-[10px]" style={{ minWidth: 280 }}>
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1.5 bg-[#111318] border-b border-[#1e2130]">
        <span className="text-[10px] font-semibold text-slate-300">{symbol}</span>
        <div className="flex items-center gap-1">
          {groupings.map((g) => (
            <button
              key={g}
              onClick={() => setCurrentGrouping(g)}
              className={clsx(
                'px-1.5 py-0.5 rounded border text-[8px] transition-colors',
                currentGrouping === g
                  ? 'border-blue-500/50 text-blue-400'
                  : 'border-[#1e2130] text-slate-600 hover:text-slate-400',
              )}
            >
              {g}
            </button>
          ))}
          <button
            onClick={() => setShowOrders((p) => !p)}
            className={clsx(
              'ml-1 px-1.5 py-0.5 rounded border text-[8px] transition-colors',
              showOrders ? 'border-blue-500/50 text-blue-400' : 'border-[#1e2130] text-slate-600 hover:text-slate-400',
            )}
          >
            Orders
          </button>
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center gap-1 px-2 py-1 text-[8px] text-slate-700 border-b border-[#1e2130]">
        <span className="flex-1 text-right">Price</span>
        <span className="w-16 text-right">Size</span>
        <span className="w-16 text-right">Total</span>
        {showOrders && <span className="w-6 text-right">#</span>}
      </div>

      {/* Depth summary */}
      <DepthSummary bids={book.bids} asks={book.asks} />

      {/* Asks (reversed — highest at top) */}
      <div className="overflow-y-auto thin-scrollbar">
        {[...book.asks].reverse().map((level) => (
          <BookRow
            key={level.price}
            level={level}
            maxTotal={maxTotal}
            side="ask"
            isHighlighted={level.price === highlightedAsk}
            showOrders={showOrders}
            onPriceClick={handlePriceClick}
            flashUpdates={flashUpdates}
          />
        ))}
      </div>

      {/* Mid / spread */}
      <SpreadIndicator midPrice={book.midPrice} spread={book.spread} spreadBps={book.spreadBps} />

      {/* Bids */}
      <div className="overflow-y-auto thin-scrollbar">
        {book.bids.map((level) => (
          <BookRow
            key={level.price}
            level={level}
            maxTotal={maxTotal}
            side="bid"
            isHighlighted={level.price === highlightedBid}
            showOrders={showOrders}
            onPriceClick={handlePriceClick}
            flashUpdates={flashUpdates}
          />
        ))}
      </div>
    </div>
  )
}
