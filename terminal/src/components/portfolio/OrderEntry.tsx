// ============================================================
// OrderEntry — full order entry form
// ============================================================
import React, { useState, useCallback, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { usePortfolioStore, selectOpenOrders } from '@/store/portfolioStore'
import { useMarketStore } from '@/store/marketStore'
import { useSettingsStore } from '@/store/settingsStore'
import type { OrderSide, OrderType, TimeInForce, OrderRequest, Order } from '@/types'
import { format } from 'date-fns'

interface OrderEntryProps {
  defaultSymbol?: string
  defaultSide?: OrderSide
  defaultPrice?: number
  onOrderSubmitted?: (order: Order) => void
  className?: string
}

type InputMode = 'shares' | 'notional'
type OrderTabType = 'new' | 'recent'

const ORDER_TYPES: { value: OrderType; label: string }[] = [
  { value: 'market', label: 'Market' },
  { value: 'limit', label: 'Limit' },
  { value: 'stop', label: 'Stop' },
  { value: 'stop_limit', label: 'Stop Limit' },
  { value: 'trailing_stop', label: 'Trail Stop' },
]

const TIF_OPTIONS: { value: TimeInForce; label: string }[] = [
  { value: 'day', label: 'Day' },
  { value: 'gtc', label: 'GTC' },
  { value: 'ioc', label: 'IOC' },
  { value: 'fok', label: 'FOK' },
]

function OrderRow({ order }: { order: Order }) {
  const cancelOrder = usePortfolioStore((s) => s.cancelOrder)
  const [cancelling, setCancelling] = useState(false)

  const handleCancel = useCallback(async () => {
    setCancelling(true)
    await cancelOrder(order.id)
    setCancelling(false)
  }, [order.id, cancelOrder])

  const statusColor = {
    pending: 'text-terminal-warning',
    accepted: 'text-terminal-info',
    filled: 'text-terminal-bull',
    partial: 'text-terminal-info',
    cancelled: 'text-terminal-subtle',
    rejected: 'text-terminal-bear',
    expired: 'text-terminal-subtle',
  }[order.status] ?? 'text-terminal-subtle'

  const canCancel = order.status === 'pending' || order.status === 'accepted' || order.status === 'partial'

  return (
    <div className="flex items-center gap-2 px-2 py-1.5 border-b border-terminal-border/30 text-xs">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className={`font-mono font-semibold ${order.side === 'buy' ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
            {order.side.toUpperCase()}
          </span>
          <span className="font-mono text-terminal-text">{order.symbol}</span>
          <span className="font-mono text-terminal-subtle">{order.qty}@{order.price ? `$${order.price.toFixed(2)}` : 'MKT'}</span>
        </div>
        <div className="flex items-center gap-2 text-[10px] mt-0.5">
          <span className={statusColor}>{order.status}</span>
          {order.filledQty > 0 && (
            <span className="text-terminal-subtle">filled: {order.filledQty}</span>
          )}
          <span className="text-terminal-subtle">{format(new Date(order.createdAt), 'HH:mm:ss')}</span>
        </div>
      </div>
      {canCancel && (
        <button
          onClick={handleCancel}
          disabled={cancelling}
          className="text-[10px] font-mono px-1.5 py-0.5 rounded text-terminal-bear border border-terminal-bear/30 hover:bg-terminal-bear/20 transition-colors disabled:opacity-50"
        >
          {cancelling ? '...' : '✕'}
        </button>
      )}
    </div>
  )
}

export const OrderEntry: React.FC<OrderEntryProps> = ({
  defaultSymbol,
  defaultSide = 'buy',
  defaultPrice,
  onOrderSubmitted,
  className = '',
}) => {
  const submitOrder = usePortfolioStore((s) => s.submitOrder)
  const isSubmitting = usePortfolioStore((s) => s.isSubmittingOrder)
  const openOrders = usePortfolioStore(selectOpenOrders)
  const account = usePortfolioStore((s) => s.account)
  const settings = useSettingsStore((s) => s.settings)

  const selectedSymbol = useMarketStore((s) => s.selectedSymbol)
  const getQuote = useMarketStore((s) => (sym: string) => s.quotes[sym])

  const [tab, setTab] = useState<OrderTabType>('new')
  const [symbol, setSymbol] = useState(defaultSymbol ?? selectedSymbol ?? 'SPY')
  const [side, setSide] = useState<OrderSide>(defaultSide)
  const [orderType, setOrderType] = useState<OrderType>((settings.defaultOrderType as OrderType) ?? 'limit')
  const [tif, setTif] = useState<TimeInForce>((settings.defaultTimeInForce as TimeInForce) ?? 'day')
  const [qty, setQty] = useState<string>('100')
  const [price, setPrice] = useState<string>(defaultPrice?.toFixed(2) ?? '')
  const [stopPrice, setStopPrice] = useState<string>('')
  const [trailPct, setTrailPct] = useState<string>('1.0')
  const [inputMode, setInputMode] = useState<InputMode>('shares')
  const [extendedHours, setExtendedHours] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastOrderId, setLastOrderId] = useState<string | null>(null)

  const quote = getQuote(symbol)

  // Sync symbol from market store
  useEffect(() => {
    if (!defaultSymbol) {
      setSymbol(selectedSymbol)
      if (quote) {
        setPrice(quote.lastPrice.toFixed(2))
      }
    }
  }, [selectedSymbol, defaultSymbol])

  // Auto-fill price from quote
  useEffect(() => {
    if (quote && !defaultPrice) {
      setPrice(quote.lastPrice.toFixed(2))
    }
  }, [quote?.lastPrice])

  const numQty = parseFloat(qty) || 0
  const numPrice = parseFloat(price) || quote?.lastPrice || 0
  const estimatedCost = inputMode === 'shares' ? numQty * numPrice : parseFloat(qty) || 0
  const sharesFromNotional = inputMode === 'notional' ? estimatedCost / numPrice : numQty
  const commission = settings.alpacaPaper ? 0 : estimatedCost * 0.0005
  const marginImpact = estimatedCost / (account?.buyingPower ?? 1) * 100

  const validate = useCallback((): string | null => {
    if (!symbol.trim()) return 'Symbol required'
    if (isNaN(numQty) || numQty <= 0) return 'Quantity must be positive'
    if (orderType === 'limit' && (!numPrice || numPrice <= 0)) return 'Limit price required'
    if (orderType === 'stop' && (!parseFloat(stopPrice) || parseFloat(stopPrice) <= 0)) return 'Stop price required'
    if (account && estimatedCost > account.buyingPower) return 'Insufficient buying power'
    return null
  }, [symbol, numQty, numPrice, orderType, stopPrice, account, estimatedCost])

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    const validationError = validate()
    if (validationError) { setError(validationError); return }

    if (settings.confirmOrders) {
      const confirmed = confirm(
        `${side.toUpperCase()} ${sharesFromNotional.toFixed(0)} ${symbol} @ ${orderType === 'market' ? 'MARKET' : `$${numPrice.toFixed(2)}`}\nEstimated: $${estimatedCost.toFixed(2)}`
      )
      if (!confirmed) return
    }

    setError(null)

    const req: OrderRequest = {
      symbol,
      side,
      type: orderType,
      timeInForce: tif,
      extendedHours,
    }

    if (inputMode === 'shares') {
      req.qty = sharesFromNotional
    } else {
      req.notional = estimatedCost
    }

    if (orderType === 'limit' || orderType === 'stop_limit') req.price = numPrice
    if (orderType === 'stop' || orderType === 'stop_limit') req.stopPrice = parseFloat(stopPrice)
    if (orderType === 'trailing_stop') req.trailPercent = parseFloat(trailPct)

    const order = await submitOrder(req)
    if (order) {
      setLastOrderId(order.id)
      onOrderSubmitted?.(order)
      // Reset qty
      setQty('100')
    }
  }, [symbol, side, orderType, tif, inputMode, numQty, numPrice, stopPrice, trailPct, extendedHours, sharesFromNotional, estimatedCost, validate, settings.confirmOrders, submitOrder, onOrderSubmitted])

  const needsPrice = orderType === 'limit' || orderType === 'stop_limit'
  const needsStop = orderType === 'stop' || orderType === 'stop_limit'
  const needsTrail = orderType === 'trailing_stop'

  return (
    <div className={`flex flex-col bg-terminal-bg h-full ${className}`}>
      {/* Tabs */}
      <div className="flex border-b border-terminal-border flex-shrink-0">
        {(['new', 'recent'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 py-2 text-xs font-mono transition-colors ${
              tab === t ? 'text-terminal-text border-b-2 border-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text'
            }`}
          >
            {t === 'new' ? 'New Order' : `Orders (${openOrders.length})`}
          </button>
        ))}
      </div>

      {tab === 'new' ? (
        <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto p-3 space-y-2.5">
          {/* Symbol */}
          <div>
            <label className="text-[10px] font-mono text-terminal-subtle uppercase mb-0.5 block">Symbol</label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-sm font-mono font-bold text-terminal-text focus:outline-none focus:border-terminal-accent"
              placeholder="SPY"
            />
          </div>

          {/* Side */}
          <div className="flex gap-1">
            <button
              type="button"
              onClick={() => setSide('buy')}
              className={`flex-1 py-2 rounded font-mono text-sm font-bold transition-colors ${
                side === 'buy'
                  ? 'bg-terminal-bull text-white'
                  : 'bg-terminal-surface text-terminal-subtle border border-terminal-border hover:border-terminal-bull/50'
              }`}
            >
              BUY
            </button>
            <button
              type="button"
              onClick={() => setSide('sell')}
              className={`flex-1 py-2 rounded font-mono text-sm font-bold transition-colors ${
                side === 'sell'
                  ? 'bg-terminal-bear text-white'
                  : 'bg-terminal-surface text-terminal-subtle border border-terminal-border hover:border-terminal-bear/50'
              }`}
            >
              SELL
            </button>
          </div>

          {/* Order type */}
          <div className="flex flex-wrap gap-1">
            {ORDER_TYPES.map((ot) => (
              <button
                key={ot.value}
                type="button"
                onClick={() => setOrderType(ot.value)}
                className={`px-2 py-1 rounded text-[10px] font-mono transition-colors ${
                  orderType === ot.value
                    ? 'bg-terminal-accent/20 text-terminal-accent border border-terminal-accent/40'
                    : 'text-terminal-subtle border border-terminal-border hover:text-terminal-text'
                }`}
              >
                {ot.label}
              </button>
            ))}
          </div>

          {/* Qty / Notional */}
          <div>
            <div className="flex items-center justify-between mb-0.5">
              <label className="text-[10px] font-mono text-terminal-subtle uppercase">
                {inputMode === 'shares' ? 'Quantity' : 'Notional ($)'}
              </label>
              <button
                type="button"
                onClick={() => setInputMode(m => m === 'shares' ? 'notional' : 'shares')}
                className="text-[10px] font-mono text-terminal-accent hover:underline"
              >
                Switch to {inputMode === 'shares' ? '$' : 'shares'}
              </button>
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setQty(String(Math.max(1, parseFloat(qty) - (inputMode === 'shares' ? 1 : 100))))}
                className="px-2 py-1.5 bg-terminal-surface border border-terminal-border rounded text-terminal-text hover:bg-terminal-muted"
              >
                −
              </button>
              <input
                type="number"
                value={qty}
                onChange={(e) => setQty(e.target.value)}
                className="flex-1 bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-sm font-mono text-terminal-text focus:outline-none focus:border-terminal-accent text-center"
                min="0"
                step={inputMode === 'shares' ? 1 : 100}
              />
              <button
                type="button"
                onClick={() => setQty(String(parseFloat(qty) + (inputMode === 'shares' ? 1 : 100)))}
                className="px-2 py-1.5 bg-terminal-surface border border-terminal-border rounded text-terminal-text hover:bg-terminal-muted"
              >
                +
              </button>
            </div>
          </div>

          {/* Limit price */}
          {needsPrice && (
            <div>
              <label className="text-[10px] font-mono text-terminal-subtle uppercase mb-0.5 block">Limit Price</label>
              <input
                type="number"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-sm font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                step="0.01"
                placeholder="0.00"
              />
            </div>
          )}

          {/* Stop price */}
          {needsStop && (
            <div>
              <label className="text-[10px] font-mono text-terminal-subtle uppercase mb-0.5 block">Stop Price</label>
              <input
                type="number"
                value={stopPrice}
                onChange={(e) => setStopPrice(e.target.value)}
                className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-sm font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                step="0.01"
                placeholder="0.00"
              />
            </div>
          )}

          {/* Trailing stop */}
          {needsTrail && (
            <div>
              <label className="text-[10px] font-mono text-terminal-subtle uppercase mb-0.5 block">Trail %</label>
              <input
                type="number"
                value={trailPct}
                onChange={(e) => setTrailPct(e.target.value)}
                className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-sm font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                step="0.1"
                min="0.1"
                max="50"
              />
            </div>
          )}

          {/* TIF */}
          <div className="flex gap-1">
            {TIF_OPTIONS.map((t) => (
              <button
                key={t.value}
                type="button"
                onClick={() => setTif(t.value)}
                className={`flex-1 py-1 rounded text-[10px] font-mono transition-colors ${
                  tif === t.value ? 'bg-terminal-muted text-terminal-text' : 'text-terminal-subtle hover:text-terminal-text'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Extended hours */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={extendedHours}
              onChange={(e) => setExtendedHours(e.target.checked)}
              className="w-3 h-3 accent-terminal-accent"
            />
            <span className="text-[10px] font-mono text-terminal-subtle">Extended Hours</span>
          </label>

          {/* Order preview */}
          <div className="bg-terminal-surface/50 rounded p-2 border border-terminal-border/50 space-y-1">
            <div className="flex justify-between text-[10px] font-mono">
              <span className="text-terminal-subtle">Est. Cost:</span>
              <span className="text-terminal-text">${estimatedCost.toFixed(2)}</span>
            </div>
            {commission > 0 && (
              <div className="flex justify-between text-[10px] font-mono">
                <span className="text-terminal-subtle">Commission:</span>
                <span className="text-terminal-text">${commission.toFixed(2)}</span>
              </div>
            )}
            <div className="flex justify-between text-[10px] font-mono">
              <span className="text-terminal-subtle">Shares:</span>
              <span className="text-terminal-text">{sharesFromNotional.toFixed(0)}</span>
            </div>
            <div className="flex justify-between text-[10px] font-mono">
              <span className="text-terminal-subtle">Margin Used:</span>
              <span className={`${marginImpact > 50 ? 'text-terminal-warning' : 'text-terminal-text'}`}>
                {marginImpact.toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="text-terminal-bear text-[10px] font-mono bg-terminal-bear/10 rounded px-2 py-1 border border-terminal-bear/30"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Success */}
          <AnimatePresence>
            {lastOrderId && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="text-terminal-bull text-[10px] font-mono bg-terminal-bull/10 rounded px-2 py-1 border border-terminal-bull/30"
              >
                Order submitted: {lastOrderId.slice(0, 8)}...
              </motion.div>
            )}
          </AnimatePresence>

          {/* Submit */}
          <button
            type="submit"
            disabled={isSubmitting}
            className={`w-full py-2.5 rounded font-mono font-bold text-sm transition-colors ${
              isSubmitting ? 'opacity-50 cursor-not-allowed' : ''
            } ${
              side === 'buy'
                ? 'bg-terminal-bull hover:bg-terminal-bull/80 text-white'
                : 'bg-terminal-bear hover:bg-terminal-bear/80 text-white'
            }`}
          >
            {isSubmitting ? (
              <span className="flex items-center justify-center gap-2">
                <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Submitting...
              </span>
            ) : (
              `${side.toUpperCase()} ${symbol}`
            )}
          </button>
        </form>
      ) : (
        // Recent orders
        <div className="flex-1 overflow-y-auto">
          {openOrders.length === 0 ? (
            <div className="flex items-center justify-center py-8 text-terminal-subtle text-xs">
              No open orders
            </div>
          ) : (
            openOrders.map((order) => <OrderRow key={order.id} order={order} />)
          )}
        </div>
      )}
    </div>
  )
}

export default OrderEntry
