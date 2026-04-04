// ============================================================
// GammaExposure.tsx — GEX bar chart by strike
// ============================================================
import React, { useMemo, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  LineChart,
  Line,
  ComposedChart,
  Area,
} from 'recharts'
import { clsx } from 'clsx'

// ---- Types ----

interface GEXLevel {
  strike: number
  callGEX: number     // positive = dealers long gamma → acts as support/resistance
  putGEX: number      // negative = dealers short gamma → accelerates moves
  netGEX: number
  openInterestCall: number
  openInterestPut: number
  isFlipLevel: boolean
}

export interface GammaExposureProps {
  symbol?: string
  spotPrice?: number
  className?: string
}

// ---- Mock GEX generation ----

function blackScholesGamma(S: number, K: number, T: number, r: number, sigma: number): number {
  if (T <= 0) return 0
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * Math.sqrt(T))
  const nd1 = Math.exp(-0.5 * d1 ** 2) / Math.sqrt(2 * Math.PI)
  return nd1 / (S * sigma * Math.sqrt(T))
}

function generateGEXData(spot: number): GEXLevel[] {
  const step = spot > 50000 ? 500 : spot > 5000 ? 50 : spot > 100 ? 5 : 0.5
  const atmStrike = Math.round(spot / step) * step
  const strikes: number[] = []
  for (let i = -16; i <= 16; i++) {
    strikes.push(atmStrike + i * step)
  }

  const T = 21 / 365
  const r = 0.05
  const atmSigma = 0.38

  let totalNetGEX = 0
  const levels: (GEXLevel & { absNet: number })[] = strikes.map((K) => {
    const m = Math.log(K / spot)
    const sigma = Math.max(0.15, atmSigma + 0.4 * m ** 2 - 0.12 * m)

    const gamma = blackScholesGamma(spot, K, T, r, sigma)

    const callOI = Math.floor(
      Math.random() * 3000 + 200 +
      (Math.abs(K - spot) < step * 1.5 ? 5000 : 0) +  // ATM has high OI
      (K === atmStrike + 4 * step ? 8000 : 0),          // common strike wall
    )
    const putOI = Math.floor(
      Math.random() * 2500 + 200 +
      (Math.abs(K - spot) < step * 1.5 ? 4000 : 0) +
      (K === atmStrike - 3 * step ? 7000 : 0),
    )

    // GEX = gamma * OI * spot^2 * 0.01 (in $M equivalent)
    const callGEX = gamma * callOI * spot ** 2 * 0.01 / 1_000_000
    const putGEX  = -gamma * putOI * spot ** 2 * 0.01 / 1_000_000  // dealers short put gamma
    const netGEX  = callGEX + putGEX

    totalNetGEX += netGEX

    return {
      strike: K,
      callGEX,
      putGEX,
      netGEX,
      openInterestCall: callOI,
      openInterestPut: putOI,
      isFlipLevel: false,
      absNet: Math.abs(netGEX),
    }
  })

  // Find gamma flip level (where net GEX crosses zero)
  for (let i = 0; i < levels.length - 1; i++) {
    if (levels[i].netGEX * levels[i + 1].netGEX < 0) {
      // Mark the one closer to zero as flip
      if (Math.abs(levels[i].netGEX) < Math.abs(levels[i + 1].netGEX)) {
        levels[i].isFlipLevel = true
      } else {
        levels[i + 1].isFlipLevel = true
      }
    }
  }

  return levels
}

// ---- Tooltip ----

const GEXTooltip: React.FC<{
  active?: boolean
  payload?: { value: number; dataKey: string; name: string }[]
  label?: string | number
  spot: number
}> = ({ active, payload, label, spot }) => {
  if (!active || !payload?.length) return null

  const strike = Number(label)
  const pct = ((strike - spot) / spot * 100).toFixed(1)

  return (
    <div className="bg-[#111318] border border-[#1e2130] rounded-lg p-2.5 text-[9px] font-mono">
      <div className="text-slate-400 mb-1.5 font-semibold">
        Strike {strike >= 1000 ? strike.toLocaleString() : strike} ({pct}%)
      </div>
      {payload.map((p) => {
        const isCall = p.dataKey === 'callGEX'
        const isPut = p.dataKey === 'putGEX'
        const isNet = p.dataKey === 'netGEX'
        return (
          <div key={p.dataKey} className="flex justify-between gap-4 mb-0.5">
            <span className={clsx(
              isCall ? 'text-blue-400' : isPut ? 'text-red-400' : isNet ? 'text-slate-300 font-semibold' : 'text-slate-500',
            )}>
              {isCall ? 'Call GEX' : isPut ? 'Put GEX' : 'Net GEX'}
            </span>
            <span className={clsx(
              isCall ? 'text-blue-400' : isPut ? 'text-red-400' : p.value >= 0 ? 'text-emerald-400' : 'text-red-400',
            )}>
              {p.value >= 0 ? '+' : ''}{p.value.toFixed(2)}M$
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ---- Main component ----

const SPOT = 63450

export const GammaExposure: React.FC<GammaExposureProps> = ({
  symbol = 'BTCUSDT',
  spotPrice = SPOT,
  className,
}) => {
  const [view, setView] = useState<'net' | 'breakdown' | 'oi'>('net')
  const data = useMemo(() => generateGEXData(spotPrice), [spotPrice])

  const flipLevel = data.find((d) => d.isFlipLevel)
  const totalNetGEX = data.reduce((s, d) => s + d.netGEX, 0)

  const formatStrike = (v: number) =>
    v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v)

  return (
    <div className={clsx('bg-[#0e1017] border border-[#1e2130] rounded-lg overflow-hidden', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#1e2130]">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono font-semibold text-slate-300">{symbol} GEX</span>
          <span className={clsx(
            'text-[10px] font-mono font-bold',
            totalNetGEX >= 0 ? 'text-emerald-400' : 'text-red-400',
          )}>
            {totalNetGEX >= 0 ? '+' : ''}{totalNetGEX.toFixed(1)}M$
          </span>
          {flipLevel && (
            <span className="text-[9px] font-mono text-amber-400">
              Flip: {flipLevel.strike >= 1000 ? flipLevel.strike.toLocaleString() : flipLevel.strike}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {(['net', 'breakdown', 'oi'] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={clsx(
                'px-2 py-0.5 rounded border text-[8px] font-mono transition-colors capitalize',
                view === v
                  ? 'border-blue-500/50 text-blue-400 bg-blue-950/30'
                  : 'border-[#1e2130] text-slate-600 hover:text-slate-400',
              )}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* Main chart */}
      <div className="p-3">
        {view === 'net' && (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="#1a1d26" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="strike"
                tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={formatStrike}
                interval={3}
              />
              <YAxis
                tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `${v >= 0 ? '' : '-'}${Math.abs(v).toFixed(0)}M`}
              />
              <Tooltip content={<GEXTooltip spot={spotPrice} />} />
              <ReferenceLine y={0} stroke="#2e3550" strokeWidth={1.5} />
              {/* Spot price marker */}
              <ReferenceLine x={Math.round(spotPrice / (spotPrice > 50000 ? 500 : 50)) * (spotPrice > 50000 ? 500 : 50)}
                stroke="#3b82f6" strokeDasharray="4 2" strokeOpacity={0.6} />
              {/* Flip level marker */}
              {flipLevel && (
                <ReferenceLine x={flipLevel.strike} stroke="#f59e0b" strokeDasharray="2 2" strokeOpacity={0.7}
                  label={{ value: 'FLIP', fill: '#f59e0b', fontSize: 7, fontFamily: 'JetBrains Mono' }} />
              )}
              <Bar dataKey="netGEX" radius={[2, 2, 0, 0]}>
                {data.map((d, i) => (
                  <Cell
                    key={i}
                    fill={d.isFlipLevel
                      ? '#f59e0b'
                      : d.netGEX >= 0 ? '#3b82f6' : '#ef4444'}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}

        {view === 'breakdown' && (
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="#1a1d26" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="strike"
                tick={{ fill: '#475569', fontSize: 8 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={formatStrike}
                interval={3}
              />
              <YAxis
                tick={{ fill: '#475569', fontSize: 8 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `${v.toFixed(0)}M`}
              />
              <Tooltip content={<GEXTooltip spot={spotPrice} />} />
              <ReferenceLine y={0} stroke="#2e3550" strokeWidth={1.5} />
              <ReferenceLine x={Math.round(spotPrice / (spotPrice > 50000 ? 500 : 50)) * (spotPrice > 50000 ? 500 : 50)}
                stroke="#3b82f6" strokeDasharray="4 2" strokeOpacity={0.5} />
              <Bar dataKey="callGEX" fill="#3b82f6" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
              <Bar dataKey="putGEX" fill="#ef4444" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
              <Line type="monotone" dataKey="netGEX" stroke="#e2e8f0" strokeWidth={1.5} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        )}

        {view === 'oi' && (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="#1a1d26" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="strike"
                tick={{ fill: '#475569', fontSize: 8 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={formatStrike}
                interval={3}
              />
              <YAxis
                tick={{ fill: '#475569', fontSize: 8 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `${(v / 1000).toFixed(0)}k`}
              />
              <Tooltip
                contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 9, fontFamily: 'JetBrains Mono' }}
                formatter={(v: number, name: string) => [
                  v.toLocaleString(),
                  name === 'openInterestCall' ? 'Call OI' : 'Put OI',
                ]}
              />
              <ReferenceLine x={Math.round(spotPrice / (spotPrice > 50000 ? 500 : 50)) * (spotPrice > 50000 ? 500 : 50)}
                stroke="#3b82f6" strokeDasharray="4 2" strokeOpacity={0.5} />
              <Bar dataKey="openInterestCall" fill="#3b82f6" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
              <Bar dataKey="openInterestPut" fill="#ef4444" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Footer info */}
      <div className="px-3 pb-3 flex items-center gap-3 text-[8px] font-mono text-slate-700">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-sm bg-blue-500/70" />
          <span>Positive GEX (dealers long gamma → support)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-sm bg-red-500/70" />
          <span>Negative GEX (dealers short gamma → volatility)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-sm bg-amber-500/70" />
          <span>Gamma flip level</span>
        </div>
      </div>
    </div>
  )
}
