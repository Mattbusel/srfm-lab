// ============================================================
// QuoteBar — selected symbol quote display with flash animations
// ============================================================
import React, { useEffect, useRef, useState } from 'react'
import { useMarketStore } from '@/store/marketStore'
import { useBHStore } from '@/store/bhStore'

interface QuoteBarProps {
  symbol: string
  className?: string
}

function FlashValue({ value, format: fmt, className = '' }: {
  value: number | undefined
  format: (v: number) => string
  className?: string
}) {
  const prevRef = useRef<number | undefined>(value)
  const [flashClass, setFlashClass] = useState('')

  useEffect(() => {
    if (value !== undefined && prevRef.current !== undefined && value !== prevRef.current) {
      const dir = value > prevRef.current ? 'up' : 'down'
      setFlashClass(dir === 'up' ? 'animate-flash-green' : 'animate-flash-red')
      const timer = setTimeout(() => setFlashClass(''), 400)
      prevRef.current = value
      return () => clearTimeout(timer)
    }
    prevRef.current = value
  }, [value])

  return (
    <span className={`font-mono ${className} ${flashClass}`}>
      {value !== undefined ? fmt(value) : '—'}
    </span>
  )
}

const formatPrice = (p: number) => p.toFixed(p > 100 ? 2 : 4)
const formatChange = (c: number) => (c >= 0 ? '+' : '') + c.toFixed(2)
const formatPct = (p: number) => (p >= 0 ? '+' : '') + (p * 100).toFixed(2) + '%'
const formatVolume = (v: number) => {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`
  return v.toString()
}

export const QuoteBar: React.FC<QuoteBarProps> = ({ symbol, className = '' }) => {
  const quote = useMarketStore((s) => s.quotes[symbol])
  const marketSession = useMarketStore((s) => s.marketSession)
  const bhInstrument = useBHStore((s) => s.instruments[symbol])

  const isUp = (quote?.dayChangePct ?? 0) >= 0

  // Session indicator
  const sessionLabel = marketSession?.isOpen
    ? marketSession.sessionType === 'regular' ? 'OPEN' : marketSession.sessionType.toUpperCase()
    : 'CLOSED'
  const sessionColor = marketSession?.isOpen ? 'text-terminal-bull' : 'text-terminal-subtle'

  return (
    <div className={`flex items-center gap-4 px-3 py-2 border-b border-terminal-border bg-terminal-surface flex-shrink-0 overflow-x-auto ${className}`}>
      {/* Symbol & session */}
      <div className="flex items-center gap-2 flex-shrink-0">
        <span className="font-mono text-sm font-bold text-terminal-text">{symbol}</span>
        <span className={`text-[10px] font-mono ${sessionColor} border border-current rounded px-1`}>
          {sessionLabel}
        </span>
      </div>

      {/* Last price */}
      <div className="flex items-baseline gap-1.5 flex-shrink-0">
        <FlashValue
          value={quote?.lastPrice}
          format={formatPrice}
          className={`text-base font-bold ${isUp ? 'text-terminal-bull' : 'text-terminal-bear'}`}
        />
        <div className="flex items-center gap-1 text-xs">
          <FlashValue
            value={quote?.dayChange}
            format={formatChange}
            className={isUp ? 'text-terminal-bull' : 'text-terminal-bear'}
          />
          <FlashValue
            value={quote?.dayChangePct}
            format={formatPct}
            className={isUp ? 'text-terminal-bull' : 'text-terminal-bear'}
          />
        </div>
      </div>

      {/* Bid / Ask */}
      <div className="flex items-center gap-2 text-xs flex-shrink-0">
        <div className="flex flex-col items-center">
          <span className="text-[9px] text-terminal-subtle font-mono uppercase">Bid</span>
          <span className="font-mono text-terminal-bull">
            {quote ? formatPrice(quote.bidPrice) : '—'}
          </span>
        </div>
        <div className="text-terminal-muted">×</div>
        <div className="flex flex-col items-center">
          <span className="text-[9px] text-terminal-subtle font-mono uppercase">Ask</span>
          <span className="font-mono text-terminal-bear">
            {quote ? formatPrice(quote.askPrice) : '—'}
          </span>
        </div>
        <div className="flex flex-col items-center ml-1">
          <span className="text-[9px] text-terminal-subtle font-mono uppercase">Sprd</span>
          <span className="font-mono text-terminal-subtle text-[11px]">
            {quote ? quote.spreadBps.toFixed(1) + 'bps' : '—'}
          </span>
        </div>
      </div>

      <div className="w-px h-6 bg-terminal-border flex-shrink-0" />

      {/* Day range */}
      <div className="flex flex-col flex-shrink-0">
        <span className="text-[9px] text-terminal-subtle font-mono uppercase">Day Range</span>
        <div className="flex items-center gap-1 text-xs font-mono">
          <span className="text-terminal-bear">{quote ? formatPrice(quote.dayLow) : '—'}</span>
          <span className="text-terminal-subtle">—</span>
          <span className="text-terminal-bull">{quote ? formatPrice(quote.dayHigh) : '—'}</span>
        </div>
        {quote && quote.dayHigh > quote.dayLow && (
          <div className="w-20 h-1 bg-terminal-muted rounded-full mt-0.5">
            <div
              className="h-full bg-terminal-accent rounded-full"
              style={{
                width: `${((quote.lastPrice - quote.dayLow) / (quote.dayHigh - quote.dayLow)) * 100}%`,
              }}
            />
          </div>
        )}
      </div>

      {/* VWAP */}
      {quote?.dayVwap && (
        <div className="flex flex-col flex-shrink-0">
          <span className="text-[9px] text-terminal-subtle font-mono uppercase">VWAP</span>
          <span className="font-mono text-xs text-terminal-text">{formatPrice(quote.dayVwap)}</span>
        </div>
      )}

      {/* Volume */}
      <div className="flex flex-col flex-shrink-0">
        <span className="text-[9px] text-terminal-subtle font-mono uppercase">Volume</span>
        <span className="font-mono text-xs text-terminal-text">
          {quote ? formatVolume(quote.dayVolume) : '—'}
        </span>
      </div>

      <div className="w-px h-6 bg-terminal-border flex-shrink-0" />

      {/* Prev close */}
      <div className="flex flex-col flex-shrink-0">
        <span className="text-[9px] text-terminal-subtle font-mono uppercase">Prev Close</span>
        <span className="font-mono text-xs text-terminal-text">
          {quote ? formatPrice(quote.prevClose) : '—'}
        </span>
      </div>

      {/* BH State quick view */}
      {bhInstrument && (
        <>
          <div className="w-px h-6 bg-terminal-border flex-shrink-0" />
          <div className="flex items-center gap-2 flex-shrink-0">
            {(['tf15m', 'tf1h', 'tf1d'] as const).map((tf) => {
              const state = bhInstrument[tf]
              const label = tf === 'tf15m' ? '15m' : tf === 'tf1h' ? '1h' : '1d'
              const color = state.regime === 'BULL' ? 'text-terminal-bull' : state.regime === 'BEAR' ? 'text-terminal-bear' : state.regime === 'HIGH_VOL' ? 'text-terminal-warning' : 'text-terminal-subtle'
              return (
                <div key={tf} className="flex flex-col items-center">
                  <span className="text-[9px] text-terminal-subtle font-mono">{label}</span>
                  <span className={`font-mono text-xs ${color} ${state.active ? 'font-bold' : ''}`}>
                    {state.mass.toFixed(2)}
                  </span>
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}

export default QuoteBar
