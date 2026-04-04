// ============================================================
// TapeReader.tsx — Scrolling trade tape with cumulative delta
// ============================================================
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  Tooltip,
} from 'recharts'
import { clsx } from 'clsx'

// ---- Types ----

export interface TapeTrade {
  id: number
  timestamp: number
  price: number
  size: number
  side: 'buy' | 'sell'
  exchange?: string
  aggressor?: 'taker' | 'maker'
}

export interface TapeReaderProps {
  symbol?: string
  maxTrades?: number
  showCumDelta?: boolean
  showStats?: boolean
  filterMinSize?: number
  autoScroll?: boolean
}

// ---- Size classification ----

type SizeClass = 'nano' | 'small' | 'medium' | 'large' | 'block' | 'whale'

function classifySize(size: number, spot: number): SizeClass {
  const usdSize = size * spot
  if (usdSize >= 1_000_000) return 'whale'
  if (usdSize >= 200_000)   return 'block'
  if (usdSize >= 50_000)    return 'large'
  if (usdSize >= 10_000)    return 'medium'
  if (usdSize >= 1_000)     return 'small'
  return 'nano'
}

const SIZE_CLASS_CONFIG: Record<SizeClass, {
  color: string
  bgColor: string
  fontWeight: number
  showBorder: boolean
  borderColor: string
}> = {
  nano:   { color: '#4b5563',     bgColor: 'transparent',               fontWeight: 400, showBorder: false, borderColor: 'transparent' },
  small:  { color: '#6b7280',     bgColor: 'transparent',               fontWeight: 400, showBorder: false, borderColor: 'transparent' },
  medium: { color: '#94a3b8',     bgColor: 'transparent',               fontWeight: 400, showBorder: false, borderColor: 'transparent' },
  large:  { color: '#fbbf24',     bgColor: 'rgba(251,191,36,0.06)',      fontWeight: 600, showBorder: true,  borderColor: 'rgba(251,191,36,0.2)' },
  block:  { color: '#f97316',     bgColor: 'rgba(249,115,22,0.08)',      fontWeight: 700, showBorder: true,  borderColor: 'rgba(249,115,22,0.3)' },
  whale:  { color: '#ec4899',     bgColor: 'rgba(236,72,153,0.12)',      fontWeight: 700, showBorder: true,  borderColor: 'rgba(236,72,153,0.5)' },
}

// ---- Mock trade generator ----

const SPOT = 63450

let nextId = 0

function generateTrade(): TapeTrade {
  const u = Math.random()
  const size = u < 0.7
    ? Math.random() * 0.1 + 0.001
    : u < 0.9
      ? Math.random() * 1 + 0.1
      : u < 0.97
        ? Math.random() * 5 + 1
        : Math.random() * 20 + 5

  return {
    id: nextId++,
    timestamp: Date.now(),
    price: SPOT + (Math.random() - 0.5) * 30,
    size,
    side: Math.random() > 0.5 ? 'buy' : 'sell',
    exchange: ['Binance', 'OKX', 'Bybit', 'Deribit'][Math.floor(Math.random() * 4)],
    aggressor: Math.random() > 0.3 ? 'taker' : 'maker',
  }
}

// ---- Cumulative delta bar chart ----

interface CumDeltaPoint {
  time: string
  delta: number
  cumDelta: number
}

const CumDeltaChart: React.FC<{ points: CumDeltaPoint[] }> = ({ points }) => (
  <ResponsiveContainer width="100%" height={80}>
    <BarChart data={points} margin={{ top: 2, right: 0, bottom: 0, left: -12 }}>
      <XAxis
        dataKey="time"
        tick={{ fill: '#475569', fontSize: 7, fontFamily: 'JetBrains Mono' }}
        axisLine={false}
        tickLine={false}
        interval={5}
      />
      <YAxis
        tick={{ fill: '#475569', fontSize: 7 }}
        axisLine={false}
        tickLine={false}
      />
      <Tooltip
        formatter={(v: number) => [v.toFixed(3), 'Δ']}
        contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 4, fontSize: 8, fontFamily: 'JetBrains Mono' }}
      />
      <ReferenceLine y={0} stroke="#2e3550" />
      <Bar dataKey="delta" radius={[1, 1, 0, 0]}>
        {points.map((p, i) => (
          <Cell key={i} fill={p.delta >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={0.75} />
        ))}
      </Bar>
    </BarChart>
  </ResponsiveContainer>
)

// ---- Stats panel ----

interface TapeStats {
  totalBuys: number
  totalSells: number
  buyVolume: number
  sellVolume: number
  cumDelta: number
  largeOrders: number
  whaleOrders: number
  avgTradeSize: number
}

const StatsPanel: React.FC<{ stats: TapeStats }> = ({ stats }) => {
  const delta = stats.cumDelta
  const ratio = stats.buyVolume / (stats.buyVolume + stats.sellVolume)

  return (
    <div className="grid grid-cols-2 gap-1 p-2 border-b border-[#1e2130]">
      {[
        { label: 'Cum Δ', value: `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}`, color: delta >= 0 ? '#22c55e' : '#ef4444' },
        { label: 'B/S Ratio', value: `${(ratio * 100).toFixed(0)}/${(100 - ratio * 100).toFixed(0)}%`, color: ratio > 0.55 ? '#22c55e' : ratio < 0.45 ? '#ef4444' : '#94a3b8' },
        { label: 'Buy Vol', value: stats.buyVolume.toFixed(2), color: '#22c55e' },
        { label: 'Sell Vol', value: stats.sellVolume.toFixed(2), color: '#ef4444' },
        { label: 'Large', value: stats.largeOrders.toString(), color: '#fbbf24' },
        { label: 'Whale', value: stats.whaleOrders.toString(), color: '#ec4899' },
      ].map((s) => (
        <div key={s.label} className="flex items-center justify-between px-1.5 py-0.5 bg-[#111318] rounded border border-[#1e2130]">
          <span className="text-[8px] font-mono text-slate-700 uppercase">{s.label}</span>
          <span className="text-[9px] font-mono font-semibold" style={{ color: s.color }}>{s.value}</span>
        </div>
      ))}
    </div>
  )
}

// ---- Trade row ----

const TradeRow: React.FC<{
  trade: TapeTrade
  sizeClass: SizeClass
  isNew: boolean
}> = ({ trade, sizeClass, isNew }) => {
  const cfg = SIZE_CLASS_CONFIG[sizeClass]
  const time = new Date(trade.timestamp).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })

  return (
    <div
      className={clsx(
        'flex items-center gap-1 px-2 py-[3px] text-[9px] font-mono transition-all',
        isNew && 'animate-pulse',
      )}
      style={{
        background: cfg.bgColor,
        borderLeft: cfg.showBorder ? `2px solid ${trade.side === 'buy' ? '#22c55e' : '#ef4444'}` : '2px solid transparent',
        borderBottom: cfg.showBorder ? `1px solid ${cfg.borderColor}` : '1px solid rgba(30,33,48,0.3)',
      }}
    >
      <span className="text-slate-700 flex-shrink-0 w-[76px]">{time}</span>
      <span className={clsx(
        'flex-shrink-0 w-8',
        trade.side === 'buy' ? 'text-emerald-400' : 'text-red-400',
      )}>
        {trade.side === 'buy' ? 'B' : 'S'}
      </span>
      <span className="flex-1 text-right" style={{ color: cfg.color, fontWeight: cfg.fontWeight }}>
        {trade.price.toFixed(1)}
      </span>
      <span className="text-right w-16" style={{ color: cfg.color, fontWeight: cfg.fontWeight }}>
        {trade.size.toFixed(3)}
      </span>
      {sizeClass !== 'nano' && sizeClass !== 'small' && (
        <span className="text-[7px] flex-shrink-0 ml-1 uppercase px-1 rounded border" style={{ color: cfg.color, borderColor: cfg.borderColor }}>
          {sizeClass}
        </span>
      )}
    </div>
  )
}

// ---- Main component ----

export const TapeReader: React.FC<TapeReaderProps> = ({
  symbol = 'BTCUSDT',
  maxTrades = 200,
  showCumDelta = true,
  showStats = true,
  filterMinSize = 0,
  autoScroll = true,
}) => {
  const [trades, setTrades] = useState<TapeTrade[]>(() =>
    Array.from({ length: 50 }, () => {
      const t = generateTrade()
      t.timestamp = Date.now() - Math.random() * 60000
      return t
    }).sort((a, b) => b.timestamp - a.timestamp),
  )
  const [newIds, setNewIds] = useState<Set<number>>(new Set())
  const [filterClass, setFilterClass] = useState<SizeClass | 'all'>('all')
  const scrollRef = useRef<HTMLDivElement>(null)

  // Add new trades periodically
  useEffect(() => {
    const interval = setInterval(() => {
      const count = Math.floor(Math.random() * 3) + 1
      const newTrades = Array.from({ length: count }, () => generateTrade())
      const ids = new Set(newTrades.map((t) => t.id))

      setTrades((prev) => [...newTrades, ...prev].slice(0, maxTrades))
      setNewIds(ids)
      setTimeout(() => setNewIds(new Set()), 500)

      if (autoScroll && scrollRef.current) {
        scrollRef.current.scrollTop = 0
      }
    }, 500)
    return () => clearInterval(interval)
  }, [maxTrades, autoScroll])

  const tradeWithClass = useMemo(
    () => trades
      .filter((t) => t.size >= filterMinSize)
      .map((t) => ({ trade: t, sizeClass: classifySize(t.size, SPOT) }))
      .filter(({ sizeClass }) => filterClass === 'all' || sizeClass === filterClass),
    [trades, filterMinSize, filterClass],
  )

  const stats = useMemo((): TapeStats => {
    const buys = trades.filter((t) => t.side === 'buy')
    const sells = trades.filter((t) => t.side === 'sell')
    const buyVol = buys.reduce((s, t) => s + t.size, 0)
    const sellVol = sells.reduce((s, t) => s + t.size, 0)
    return {
      totalBuys: buys.length,
      totalSells: sells.length,
      buyVolume: buyVol,
      sellVolume: sellVol,
      cumDelta: buyVol - sellVol,
      largeOrders: trades.filter((t) => classifySize(t.size, SPOT) === 'large').length,
      whaleOrders: trades.filter((t) => ['block', 'whale'].includes(classifySize(t.size, SPOT))).length,
      avgTradeSize: trades.reduce((s, t) => s + t.size, 0) / trades.length,
    }
  }, [trades])

  // Cumulative delta over last 30 bars (1 min each)
  const cumDeltaPoints = useMemo((): CumDeltaPoint[] => {
    const now = Date.now()
    const buckets: Record<string, { delta: number }> = {}
    for (let i = 29; i >= 0; i--) {
      const t = new Date(now - i * 60000)
      const key = t.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })
      buckets[key] = { delta: 0 }
    }
    for (const trade of trades) {
      const t = new Date(trade.timestamp)
      const key = t.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' })
      if (buckets[key]) {
        buckets[key].delta += trade.side === 'buy' ? trade.size : -trade.size
      }
    }
    let cumDelta = 0
    return Object.entries(buckets).map(([time, { delta }]) => {
      cumDelta += delta
      return { time, delta, cumDelta }
    })
  }, [trades])

  const sizeFilters: (SizeClass | 'all')[] = ['all', 'large', 'block', 'whale']

  return (
    <div className="flex flex-col bg-[#0e1017] border border-[#1e2130] rounded-lg overflow-hidden" style={{ minWidth: 320 }}>
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1.5 bg-[#111318] border-b border-[#1e2130]">
        <span className="text-[10px] font-mono font-semibold text-slate-300">{symbol} TAPE</span>
        <div className="flex items-center gap-1">
          {sizeFilters.map((f) => (
            <button
              key={f}
              onClick={() => setFilterClass(f)}
              className={clsx(
                'px-1.5 py-0.5 rounded border text-[8px] font-mono transition-colors capitalize',
                filterClass === f
                  ? 'border-blue-500/50 text-blue-400 bg-blue-950/30'
                  : 'border-[#1e2130] text-slate-600 hover:text-slate-400',
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Cumulative delta */}
      {showCumDelta && (
        <div className="px-2 pt-2 pb-1 border-b border-[#1e2130]">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[8px] font-mono text-slate-700 uppercase">Cumulative Delta (30m)</span>
            <span className={clsx(
              'text-[9px] font-mono font-semibold',
              stats.cumDelta >= 0 ? 'text-emerald-400' : 'text-red-400',
            )}>
              {stats.cumDelta >= 0 ? '+' : ''}{stats.cumDelta.toFixed(3)}
            </span>
          </div>
          <CumDeltaChart points={cumDeltaPoints} />
        </div>
      )}

      {/* Stats */}
      {showStats && <StatsPanel stats={stats} />}

      {/* Column headers */}
      <div className="flex items-center gap-1 px-2 py-1 text-[7px] font-mono text-slate-700 uppercase border-b border-[#1e2130]">
        <span className="w-[76px]">Time</span>
        <span className="w-8">S</span>
        <span className="flex-1 text-right">Price</span>
        <span className="text-right w-16">Size</span>
      </div>

      {/* Tape */}
      <div ref={scrollRef} className="overflow-y-auto thin-scrollbar" style={{ maxHeight: 360, minHeight: 120 }}>
        {tradeWithClass.map(({ trade, sizeClass }) => (
          <TradeRow
            key={trade.id}
            trade={trade}
            sizeClass={sizeClass}
            isNew={newIds.has(trade.id)}
          />
        ))}
      </div>
    </div>
  )
}
