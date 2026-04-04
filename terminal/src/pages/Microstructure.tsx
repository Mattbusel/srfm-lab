// ============================================================
// Microstructure.tsx — Order flow & microstructure page
// ============================================================
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LineChart,
  Line,
  ReferenceLine,
} from 'recharts'
import { clsx } from 'clsx'
import { format } from 'date-fns'

// ---- Types ----

interface OFIBar {
  id: number
  time: string
  ofi: number       // positive = buy imbalance, negative = sell
  cumulativeDelta: number
  volume: number
}

interface DepthLevel {
  price: number
  size: number
  side: 'bid' | 'ask'
  cumSize: number
}

interface TapeTrade {
  id: number
  time: string
  price: number
  size: number
  side: 'buy' | 'sell'
  sizeClass: 'small' | 'medium' | 'large' | 'block'
}

interface SpreadDecomposition {
  label: string
  value: number
  color: string
}

// ---- Mock data generators ----

const SPOT = 63450

function generateOFIBar(id: number, prevCumDelta: number): OFIBar {
  const ofi = (Math.random() - 0.45) * 200 + (Math.random() > 0.6 ? 50 : -50)
  return {
    id,
    time: format(new Date(Date.now() - (50 - id) * 60000), 'HH:mm'),
    ofi,
    cumulativeDelta: prevCumDelta + ofi,
    volume: Math.abs(ofi) * (Math.random() * 3 + 1),
  }
}

function generateDepth(): DepthLevel[] {
  const levels: DepthLevel[] = []
  const askBase = SPOT * 1.0001
  const bidBase = SPOT * 0.9999

  // Asks (ascending from mid)
  let cumAsk = 0
  for (let i = 0; i < 20; i++) {
    const price = askBase + i * (SPOT * 0.0001)
    const size = Math.random() * 2 + 0.1 + (i === 5 ? 8 : 0)  // wall at level 5
    cumAsk += size
    levels.push({ price, size, side: 'ask', cumSize: cumAsk })
  }

  // Bids (descending from mid)
  let cumBid = 0
  for (let i = 0; i < 20; i++) {
    const price = bidBase - i * (SPOT * 0.0001)
    const size = Math.random() * 2 + 0.1 + (i === 3 ? 10 : 0)  // big bid wall
    cumBid += size
    levels.push({ price, size, side: 'bid', cumSize: cumBid })
  }

  return levels.sort((a, b) => b.price - a.price)
}

function generateTapeTrade(id: number): TapeTrade {
  const side: 'buy' | 'sell' = Math.random() > 0.5 ? 'buy' : 'sell'
  const size = Math.random() < 0.8
    ? Math.random() * 0.5 + 0.01
    : Math.random() < 0.9
      ? Math.random() * 3 + 0.5
      : Math.random() * 10 + 3

  const sizeClass: TapeTrade['sizeClass'] =
    size >= 10 ? 'block' : size >= 3 ? 'large' : size >= 0.5 ? 'medium' : 'small'

  return {
    id,
    time: format(new Date(), 'HH:mm:ss.SSS').slice(0, 11),
    price: SPOT + (Math.random() - 0.5) * 20,
    size,
    side,
    sizeClass,
  }
}

const SPREAD_DECOMP: SpreadDecomposition[] = [
  { label: 'Adverse Selection', value: 42, color: '#ef4444' },
  { label: 'Inventory',         value: 28, color: '#f59e0b' },
  { label: 'Processing',        value: 30, color: '#3b82f6' },
]

// ---- OFI Chart ----

const OFIChart: React.FC<{ bars: OFIBar[] }> = ({ bars }) => (
  <ResponsiveContainer width="100%" height={160}>
    <BarChart data={bars} margin={{ top: 4, right: 4, bottom: 0, left: -8 }}>
      <CartesianGrid stroke="#1a1d26" strokeDasharray="3 3" vertical={false} />
      <XAxis
        dataKey="time"
        tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
        axisLine={false}
        tickLine={false}
        interval={9}
      />
      <YAxis
        tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
        axisLine={false}
        tickLine={false}
      />
      <Tooltip
        formatter={(v: number) => [v.toFixed(0), 'OFI']}
        contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 9, fontFamily: 'JetBrains Mono' }}
      />
      <ReferenceLine y={0} stroke="#2e3550" />
      <Bar dataKey="ofi" radius={[1, 1, 0, 0]}>
        {bars.map((b, i) => (
          <Cell key={i} fill={b.ofi >= 0 ? '#22c55e' : '#ef4444'} fillOpacity={0.8} />
        ))}
      </Bar>
    </BarChart>
  </ResponsiveContainer>
)

// ---- Cumulative delta ----

const CumulativeDeltaChart: React.FC<{ bars: OFIBar[] }> = ({ bars }) => (
  <ResponsiveContainer width="100%" height={120}>
    <LineChart data={bars} margin={{ top: 4, right: 4, bottom: 0, left: -8 }}>
      <CartesianGrid stroke="#1a1d26" strokeDasharray="3 3" vertical={false} />
      <XAxis
        dataKey="time"
        tick={{ fill: '#475569', fontSize: 8 }}
        axisLine={false}
        tickLine={false}
        interval={9}
      />
      <YAxis
        tick={{ fill: '#475569', fontSize: 8 }}
        axisLine={false}
        tickLine={false}
      />
      <Tooltip
        formatter={(v: number) => [v.toFixed(0), 'Cumulative Δ']}
        contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 9, fontFamily: 'JetBrains Mono' }}
      />
      <ReferenceLine y={0} stroke="#2e3550" />
      <Line type="monotone" dataKey="cumulativeDelta" stroke="#8b5cf6" strokeWidth={1.5} dot={false} />
    </LineChart>
  </ResponsiveContainer>
)

// ---- Depth chart ----

const DepthChart: React.FC<{ levels: DepthLevel[] }> = ({ levels }) => {
  const asks = levels.filter((l) => l.side === 'ask').sort((a, b) => a.price - b.price)
  const bids = levels.filter((l) => l.side === 'bid').sort((a, b) => b.price - a.price)

  const maxSize = Math.max(...levels.map((l) => l.size))

  return (
    <div className="flex flex-col gap-0.5 overflow-y-auto" style={{ maxHeight: 400 }}>
      {asks.slice().reverse().map((level) => (
        <div key={level.price} className="flex items-center gap-2 group" style={{ fontSize: 9, fontFamily: 'JetBrains Mono' }}>
          <div className="w-16 text-right text-red-400">{level.price.toFixed(1)}</div>
          <div className="flex-1 relative h-3 bg-[#1a1d26] rounded overflow-hidden">
            <div
              className="absolute right-0 inset-y-0 rounded"
              style={{
                width: `${(level.size / maxSize) * 100}%`,
                background: 'rgba(239,68,68,0.3)',
              }}
            />
          </div>
          <div className="w-12 text-red-400/70">{level.size.toFixed(3)}</div>
        </div>
      ))}
      <div className="text-center py-1 border-y border-[#2e3550] text-[9px] font-mono text-blue-400 font-semibold">
        ${SPOT.toLocaleString()} MID
      </div>
      {bids.map((level) => (
        <div key={level.price} className="flex items-center gap-2" style={{ fontSize: 9, fontFamily: 'JetBrains Mono' }}>
          <div className="w-16 text-right text-emerald-400">{level.price.toFixed(1)}</div>
          <div className="flex-1 relative h-3 bg-[#1a1d26] rounded overflow-hidden">
            <div
              className="absolute left-0 inset-y-0 rounded"
              style={{
                width: `${(level.size / maxSize) * 100}%`,
                background: 'rgba(34,197,94,0.3)',
              }}
            />
          </div>
          <div className="w-12 text-emerald-400/70">{level.size.toFixed(3)}</div>
        </div>
      ))}
    </div>
  )
}

// ---- VPIN Gauge ----

const VPINGauge: React.FC<{ value: number }> = ({ value }) => {
  const pct = Math.max(0, Math.min(1, value))
  const angle = pct * 180
  const cx = 80, cy = 75, r = 60, sw = 12

  function arc(deg: number) {
    const rad = ((deg - 180) * Math.PI) / 180
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
  }

  const largeArc = angle > 90 ? 1 : 0
  const color = pct > 0.7 ? '#ef4444' : pct > 0.5 ? '#f59e0b' : '#22c55e'

  return (
    <div className="flex flex-col items-center">
      <svg width={160} height={95} viewBox="0 0 160 95">
        <path
          d={`M ${arc(0).x} ${arc(0).y} A ${r} ${r} 0 1 1 ${arc(180).x} ${arc(180).y}`}
          fill="none" stroke="#1e2130" strokeWidth={sw} strokeLinecap="round"
        />
        {pct > 0 && (
          <path
            d={`M ${arc(0).x} ${arc(0).y} A ${r} ${r} 0 ${largeArc} 1 ${arc(angle).x} ${arc(angle).y}`}
            fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round"
          />
        )}
        {/* Zone markers */}
        {[{ at: 0.5, label: 'HIGH' }, { at: 0.7, label: '!' }].map(({ at, label }) => {
          const pt = arc(at * 180)
          return (
            <text key={label} x={pt.x} y={pt.y - 4} textAnchor="middle" fill="#2e3550" fontSize={7} fontFamily="JetBrains Mono">
              {label}
            </text>
          )
        })}
        <text x={cx} y={cy} textAnchor="middle" fill={color} fontSize={22} fontFamily="JetBrains Mono" fontWeight={700}>
          {pct.toFixed(2)}
        </text>
        <text x={cx} y={cy + 16} textAnchor="middle" fill="#475569" fontSize={9} fontFamily="JetBrains Mono">
          VPIN TOXICITY
        </text>
      </svg>
      <div className="flex items-center gap-3 mt-1" style={{ fontSize: 8, fontFamily: 'JetBrains Mono' }}>
        {[{ t: 0.3, c: '#22c55e', l: 'Low' }, { t: 0.6, c: '#f59e0b', l: 'Elevated' }, { t: 1.0, c: '#ef4444', l: 'High' }].map((z) => (
          <div key={z.l} className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full" style={{ background: z.c }} />
            <span className="text-slate-600">{z.l}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ---- Trade Tape ----

const TapeComponent: React.FC<{ trades: TapeTrade[]; cumDelta: number }> = ({ trades, cumDelta }) => {
  const sizeColors: Record<TapeTrade['sizeClass'], string> = {
    small:  'text-slate-500',
    medium: 'text-slate-300',
    large:  'text-amber-400',
    block:  'text-red-400 font-bold',
  }

  return (
    <div className="flex flex-col gap-0.5 overflow-y-auto thin-scrollbar" style={{ maxHeight: 340, fontSize: 9, fontFamily: 'JetBrains Mono' }}>
      <div className="flex items-center justify-between px-2 py-1 border-b border-[#1e2130] mb-1">
        <span className="text-[9px] text-slate-600">TAPE</span>
        <span className={clsx('text-[9px] font-semibold', cumDelta >= 0 ? 'text-emerald-400' : 'text-red-400')}>
          CumΔ: {cumDelta >= 0 ? '+' : ''}{cumDelta.toFixed(0)}
        </span>
      </div>
      {trades.map((t) => (
        <div
          key={t.id}
          className="flex items-center gap-2 px-2 py-0.5 hover:bg-[#111318] rounded"
          style={{ borderLeft: `2px solid ${t.side === 'buy' ? '#22c55e' : '#ef4444'}` }}
        >
          <span className="text-slate-600 w-20 flex-shrink-0">{t.time}</span>
          <span className={t.side === 'buy' ? 'text-emerald-400 w-8 flex-shrink-0' : 'text-red-400 w-8 flex-shrink-0'}>
            {t.side === 'buy' ? 'BUY' : 'SELL'}
          </span>
          <span className="text-slate-300 flex-1">{t.price.toFixed(1)}</span>
          <span className={clsx('w-16 text-right flex-shrink-0', sizeColors[t.sizeClass])}>
            {t.size.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  )
}

// ---- Spread Decomposition ----

const SpreadDecompChart: React.FC<{ spread: number; bps: number }> = ({ spread, bps }) => {
  const total = SPREAD_DECOMP.reduce((s, d) => s + d.value, 0)
  let cumulative = 0

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <span className="text-[9px] font-mono text-slate-600 uppercase">Spread Decomposition</span>
        <div className="text-[10px] font-mono">
          <span className="text-slate-400">${spread.toFixed(2)}</span>
          <span className="text-slate-600 ml-1">({bps.toFixed(1)} bps)</span>
        </div>
      </div>
      {/* Stacked bar */}
      <div className="h-6 rounded overflow-hidden flex mb-3">
        {SPREAD_DECOMP.map((d) => (
          <div
            key={d.label}
            style={{ width: `${(d.value / total) * 100}%`, background: d.color + 'cc' }}
            title={`${d.label}: ${d.value}%`}
          />
        ))}
      </div>
      {/* Legend */}
      {SPREAD_DECOMP.map((d) => {
        const start = cumulative
        cumulative += d.value
        return (
          <div key={d.label} className="flex items-center gap-2 mb-1.5">
            <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ background: d.color }} />
            <span className="text-[10px] font-mono text-slate-400 flex-1">{d.label}</span>
            <span className="text-[10px] font-mono" style={{ color: d.color }}>{d.value}%</span>
            <span className="text-[9px] font-mono text-slate-600">
              {((d.value / total) * bps).toFixed(1)} bps
            </span>
          </div>
        )
      })}
      {void start}
    </div>
  )
}

// ---- Main page ----

export const Microstructure: React.FC = () => {
  const [ofiBars, setOfiBars] = useState<OFIBar[]>(() => {
    let cumDelta = 0
    return Array.from({ length: 50 }, (_, i) => {
      const bar = generateOFIBar(i, cumDelta)
      cumDelta = bar.cumulativeDelta
      return bar
    })
  })

  const [depth, setDepth] = useState<DepthLevel[]>(() => generateDepth())
  const [tape, setTape] = useState<TapeTrade[]>(() =>
    Array.from({ length: 60 }, (_, i) => generateTapeTrade(i)),
  )
  const [vpin, setVpin] = useState(0.38)
  const tradeIdRef = useRef(100)

  // Simulate live updates
  useEffect(() => {
    const interval = setInterval(() => {
      // New OFI bar
      setOfiBars((prev) => {
        const cumDelta = prev[prev.length - 1]?.cumulativeDelta ?? 0
        const newBar = generateOFIBar(prev.length, cumDelta)
        const next = [...prev.slice(-49), newBar]
        next.forEach((b, i) => { b.id = i })
        return next
      })

      // New trade
      setTape((prev) => {
        const trade = generateTapeTrade(tradeIdRef.current++)
        return [trade, ...prev.slice(0, 99)]
      })

      // Update VPIN slowly
      setVpin((p) => Math.max(0.1, Math.min(0.95, p + (Math.random() - 0.5) * 0.02)))

      // Refresh depth occasionally
      if (Math.random() > 0.7) setDepth(generateDepth())
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  const currentCumDelta = ofiBars[ofiBars.length - 1]?.cumulativeDelta ?? 0
  const spread = SPOT * 0.0002
  const bps = (spread / SPOT) * 10000

  return (
    <div className="flex flex-col h-full overflow-y-auto" style={{ padding: '12px', gap: '12px' }}>

      {/* Row 1: OFI + Cumulative delta */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
          <div className="text-[10px] font-mono text-slate-500 uppercase mb-2">
            Order Flow Imbalance — Rolling 50 bars
          </div>
          <OFIChart bars={ofiBars} />
        </div>

        <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
          <div className="text-[10px] font-mono text-slate-500 uppercase mb-2">Cumulative Delta</div>
          <CumulativeDeltaChart bars={ofiBars} />
          <div className="mt-2 flex items-center gap-3">
            <span className="text-[9px] font-mono text-slate-600">Current:</span>
            <span className={clsx(
              'text-sm font-mono font-bold',
              currentCumDelta >= 0 ? 'text-emerald-400' : 'text-red-400',
            )}>
              {currentCumDelta >= 0 ? '+' : ''}{currentCumDelta.toFixed(0)}
            </span>
          </div>
        </div>
      </div>

      {/* Row 2: Depth + Tape + VPIN */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
          <div className="text-[10px] font-mono text-slate-500 uppercase mb-2">Bid/Ask Depth</div>
          <DepthChart levels={depth} />
        </div>

        <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
          <TapeComponent trades={tape} cumDelta={currentCumDelta} />
        </div>

        <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
          <div className="text-[10px] font-mono text-slate-500 uppercase mb-2">VPIN — Order Toxicity</div>
          <VPINGauge value={vpin} />
          <div className="mt-3 text-[9px] font-mono text-slate-700 leading-relaxed">
            VPIN (Volume-Synchronized Probability of Informed Trading) measures the probability
            that a trade is from an informed trader. Values above 0.7 indicate elevated adverse
            selection risk.
          </div>
        </div>
      </div>

      {/* Row 3: Spread decomp */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
          <SpreadDecompChart spread={spread} bps={bps} />
        </div>

        {/* OFI stats */}
        <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
          <div className="text-[10px] font-mono text-slate-500 uppercase mb-3">Microstructure Summary</div>
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Bid-Ask Spread', value: `$${spread.toFixed(2)} (${bps.toFixed(1)} bps)` },
              { label: 'VPIN', value: vpin.toFixed(3), color: vpin > 0.7 ? '#ef4444' : vpin > 0.5 ? '#f59e0b' : '#22c55e' },
              { label: 'Cum Delta', value: `${currentCumDelta >= 0 ? '+' : ''}${currentCumDelta.toFixed(0)}`, color: currentCumDelta >= 0 ? '#22c55e' : '#ef4444' },
              { label: 'Buy/Sell Ratio', value: (() => {
                  const b = tape.filter((t) => t.side === 'buy').length
                  const s = tape.filter((t) => t.side === 'sell').length
                  return `${(b / (b + s) * 100).toFixed(0)}% / ${(s / (b + s) * 100).toFixed(0)}%`
                })(),
              },
              { label: 'Block Trades', value: tape.filter((t) => t.sizeClass === 'block').length.toString(), color: '#f59e0b' },
              { label: 'Avg Trade Size', value: `${(tape.reduce((s, t) => s + t.size, 0) / tape.length).toFixed(3)} BTC` },
            ].map((stat) => (
              <div key={stat.label} className="bg-[#111318] rounded border border-[#1e2130] p-2">
                <div className="text-[9px] font-mono text-slate-600 mb-0.5 uppercase">{stat.label}</div>
                <div className="text-[11px] font-mono font-semibold" style={{ color: stat.color ?? '#e2e8f0' }}>
                  {stat.value}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

    </div>
  )
}
