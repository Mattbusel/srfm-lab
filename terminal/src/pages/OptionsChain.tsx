// ============================================================
// OptionsChain.tsx — Full options chain + analytics page
// ============================================================
import React, { useState, useMemo, useCallback } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  ReferenceLine,
  AreaChart,
  Area,
} from 'recharts'
import { clsx } from 'clsx'

// ---- Types ----

interface OptionContract {
  strike: number
  expiry: string
  dteDisplay: string
  dte: number
  // Call side
  callBid: number
  callAsk: number
  callIV: number
  callDelta: number
  callGamma: number
  callTheta: number
  callVega: number
  callOI: number
  callVol: number
  callMidPrice: number
  // Put side
  putBid: number
  putAsk: number
  putIV: number
  putDelta: number
  putGamma: number
  putTheta: number
  putVega: number
  putOI: number
  putVol: number
  putMidPrice: number
  isATM: boolean
}

interface SelectedOption {
  type: 'call' | 'put'
  strike: number
  expiry: string
  contracts: number
  side: 'long' | 'short'
}

// ---- Mock data generators ----

const EXPIRIES = [
  { label: 'Apr 25', dte: 22 },
  { label: 'May 2',  dte: 29 },
  { label: 'May 16', dte: 43 },
  { label: 'Jun 20', dte: 78 },
  { label: 'Jul 18', dte: 106 },
  { label: 'Sep 19', dte: 169 },
  { label: 'Dec 19', dte: 261 },
]

function blackScholes(
  S: number,
  K: number,
  T: number,   // years
  r: number,
  sigma: number,
  type: 'call' | 'put',
): { price: number; delta: number; gamma: number; theta: number; vega: number } {
  if (T <= 0) {
    const intrinsic = type === 'call' ? Math.max(S - K, 0) : Math.max(K - S, 0)
    return { price: intrinsic, delta: type === 'call' ? (S > K ? 1 : 0) : (S < K ? -1 : 0), gamma: 0, theta: 0, vega: 0 }
  }
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * Math.sqrt(T))
  const d2 = d1 - sigma * Math.sqrt(T)

  function normCDF(x: number): number {
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911
    const sign = x < 0 ? -1 : 1
    const t = 1 / (1 + p * Math.abs(x))
    const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-x * x)
    return 0.5 * (1 + sign * y)
  }

  function normPDF(x: number): number {
    return Math.exp(-0.5 * x ** 2) / Math.sqrt(2 * Math.PI)
  }

  const Nd1 = normCDF(d1)
  const Nd2 = normCDF(d2)
  const nd1 = normPDF(d1)

  const callPrice = S * Nd1 - K * Math.exp(-r * T) * Nd2
  const putPrice  = callPrice - S + K * Math.exp(-r * T)
  const price = type === 'call' ? callPrice : putPrice
  const delta = type === 'call' ? Nd1 : Nd1 - 1
  const gamma = nd1 / (S * sigma * Math.sqrt(T))
  const theta = (type === 'call'
    ? (-S * nd1 * sigma / (2 * Math.sqrt(T)) - r * K * Math.exp(-r * T) * Nd2)
    : (-S * nd1 * sigma / (2 * Math.sqrt(T)) + r * K * Math.exp(-r * T) * (1 - Nd2))) / 365
  const vega = S * nd1 * Math.sqrt(T) / 100

  return { price: Math.max(price, 0.01), delta, gamma, theta, vega }
}

function generateChain(spot: number, expiry: { label: string; dte: number }): OptionContract[] {
  const strikes: number[] = []
  const step = spot > 50000 ? 1000 : spot > 5000 ? 100 : spot > 100 ? 5 : 0.5
  const atmStrike = Math.round(spot / step) * step
  for (let i = -12; i <= 12; i++) {
    strikes.push(atmStrike + i * step)
  }

  const T = expiry.dte / 365
  const r = 0.05
  const atmIV = 0.35 + 0.08 * Math.exp(-expiry.dte / 60)

  return strikes.map((K) => {
    const moneyness = Math.log(K / spot)
    const skew = -0.15 * moneyness + 0.4 * moneyness ** 2
    const sigma = Math.max(0.1, atmIV + skew + (Math.random() - 0.5) * 0.005)
    const spreadMult = 0.005 + 0.02 * Math.abs(moneyness)

    const call = blackScholes(spot, K, T, r, sigma, 'call')
    const put  = blackScholes(spot, K, T, r, sigma, 'put')

    const callMid = call.price
    const putMid  = put.price

    return {
      strike: K,
      expiry: expiry.label,
      dteDisplay: `${expiry.dte}d`,
      dte: expiry.dte,
      callBid:    Math.max(0.01, callMid * (1 - spreadMult)),
      callAsk:    callMid * (1 + spreadMult),
      callIV:     sigma * 100,
      callDelta:  call.delta,
      callGamma:  call.gamma,
      callTheta:  call.theta,
      callVega:   call.vega,
      callOI:     Math.floor(Math.random() * 5000 + 100),
      callVol:    Math.floor(Math.random() * 500 + 10),
      callMidPrice: callMid,
      putBid:     Math.max(0.01, putMid * (1 - spreadMult)),
      putAsk:     putMid * (1 + spreadMult),
      putIV:      sigma * 100,
      putDelta:   put.delta,
      putGamma:   put.gamma,
      putTheta:   put.theta,
      putVega:    put.vega,
      putOI:      Math.floor(Math.random() * 4000 + 100),
      putVol:     Math.floor(Math.random() * 400 + 10),
      putMidPrice: putMid,
      isATM: Math.abs(K - spot) < step * 0.6,
    }
  })
}

function generateTermStructure(spot: number): { expiry: string; dte: number; atmIV: number }[] {
  return EXPIRIES.map((e) => ({
    expiry: e.label,
    dte: e.dte,
    atmIV: (0.35 + 0.08 * Math.exp(-e.dte / 60) + (Math.random() - 0.5) * 0.01) * 100,
  }))
}

// ---- Payoff diagram ----

function computePayoff(
  selectedOptions: SelectedOption[],
  spot: number,
  chain: OptionContract[],
): { price: number; payoff: number; breakeven1?: number; breakeven2?: number }[] {
  const range = spot * 0.3
  const prices = Array.from({ length: 100 }, (_, i) => spot - range + (i / 99) * range * 2)

  return prices.map((S) => {
    let payoff = 0
    for (const opt of selectedOptions) {
      const row = chain.find((r) => r.strike === opt.strike)
      if (!row) continue
      const premium = opt.type === 'call' ? row.callMidPrice : row.putMidPrice
      const mult = opt.side === 'long' ? 1 : -1
      const intrinsic = opt.type === 'call' ? Math.max(S - opt.strike, 0) : Math.max(opt.strike - S, 0)
      payoff += mult * opt.contracts * (intrinsic - premium) * 100
    }
    return { price: S, payoff }
  })
}

// ---- Smile chart ----

const SmileChart: React.FC<{ chain: OptionContract[] }> = ({ chain }) => {
  const data = chain.map((c) => ({ strike: c.strike, callIV: c.callIV, putIV: c.putIV }))

  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
        <CartesianGrid stroke="#1a1d26" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="strike"
          tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v)}
        />
        <YAxis
          tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: number) => `${v.toFixed(0)}%`}
        />
        <Tooltip
          contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 9, fontFamily: 'JetBrains Mono' }}
        />
        <Line type="monotone" dataKey="callIV" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="Call IV" />
        <Line type="monotone" dataKey="putIV" stroke="#22c55e" strokeWidth={1.5} dot={false} name="Put IV" strokeDasharray="4 2" />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ---- Greeks heatmap ----

const GreeksHeatmap: React.FC<{ chains: Record<string, OptionContract[]>; greek: 'delta' | 'gamma' | 'theta' | 'vega'; optType: 'call' | 'put' }> = ({
  chains, greek, optType,
}) => {
  const expiries = Object.keys(chains)
  if (!expiries.length) return null

  const allRows = Object.values(chains)[0]
  if (!allRows?.length) return null

  const strikes = allRows.map((r) => r.strike)

  const getValue = (row: OptionContract): number => {
    const prefix = optType === 'call' ? 'call' : 'put'
    const key = `${prefix}${greek.charAt(0).toUpperCase() + greek.slice(1)}` as keyof OptionContract
    return Number(row[key])
  }

  const allVals = expiries.flatMap((exp) => chains[exp].map(getValue))
  const min = Math.min(...allVals)
  const max = Math.max(...allVals)

  function cellColor(v: number): string {
    const t = max === min ? 0.5 : (v - min) / (max - min)
    if (greek === 'delta') {
      const r = Math.round(30 + t * 9)
      const g = Math.round(30 + t * 167)
      const b = Math.round(80 + t * 166)
      return `rgb(${r},${g},${b})`
    }
    const r = Math.round(30 + t * 209)
    const g = Math.round(30 + t * 100)
    const b = Math.round(80 + (1 - t) * 100)
    return `rgb(${r},${g},${b})`
  }

  const cellW = 52
  const cellH = 28

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: `60px repeat(${expiries.length}, ${cellW}px)`, width: 'fit-content' }}>
        <div />
        {expiries.map((exp) => (
          <div key={exp} style={{ width: cellW, textAlign: 'center', padding: '2px 0', fontSize: 9, fontFamily: 'JetBrains Mono', color: '#475569' }}>
            {exp}
          </div>
        ))}
        {strikes.map((strike, si) => (
          <React.Fragment key={strike}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 6, fontSize: 9, fontFamily: 'JetBrains Mono', color: '#475569' }}>
              {strike >= 1000 ? `${(strike / 1000).toFixed(0)}k` : strike}
            </div>
            {expiries.map((exp) => {
              const row = chains[exp][si]
              if (!row) return <div key={exp} style={{ width: cellW, height: cellH }} />
              const val = getValue(row)
              return (
                <div
                  key={exp}
                  title={`${greek}: ${val.toFixed(4)}`}
                  style={{
                    width: cellW,
                    height: cellH,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: cellColor(val),
                    border: '1px solid rgba(20,22,30,0.5)',
                    fontSize: 8,
                    fontFamily: 'JetBrains Mono',
                    color: 'rgba(226,232,240,0.8)',
                  }}
                >
                  {Math.abs(val) >= 0.01 ? val.toFixed(2) : val.toFixed(4)}
                </div>
              )
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}

// ---- Main page ----

const SPOT = 63450

export const OptionsChain: React.FC = () => {
  const [selectedExpiry, setSelectedExpiry] = useState(EXPIRIES[0].label)
  const [selectedOptions, setSelectedOptions] = useState<SelectedOption[]>([])
  const [greekView, setGreekView] = useState<'delta' | 'gamma' | 'theta' | 'vega'>('delta')
  const [greekOptType, setGreekOptType] = useState<'call' | 'put'>('call')

  const chains = useMemo(() => {
    const result: Record<string, OptionContract[]> = {}
    for (const exp of EXPIRIES) {
      result[exp.label] = generateChain(SPOT, exp)
    }
    return result
  }, [])

  const currentChain = chains[selectedExpiry] ?? []
  const termStructure = useMemo(() => generateTermStructure(SPOT), [])

  const payoffData = useMemo(
    () => computePayoff(selectedOptions, SPOT, currentChain),
    [selectedOptions, currentChain],
  )

  const toggleOption = useCallback((type: 'call' | 'put', strike: number, side: 'long' | 'short') => {
    setSelectedOptions((prev) => {
      const existing = prev.findIndex((o) => o.type === type && o.strike === strike && o.expiry === selectedExpiry)
      if (existing >= 0) {
        return prev.filter((_, i) => i !== existing)
      }
      return [...prev, { type, strike, expiry: selectedExpiry, contracts: 1, side }]
    })
  }, [selectedExpiry])

  const isSelected = (type: 'call' | 'put', strike: number) =>
    selectedOptions.some((o) => o.type === type && o.strike === strike && o.expiry === selectedExpiry)

  const formatPrice = (p: number) => {
    if (p >= 100) return p.toFixed(0)
    if (p >= 1) return p.toFixed(2)
    return p.toFixed(4)
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto" style={{ padding: '12px', gap: '12px', display: 'flex', flexDirection: 'column' }}>

      {/* Top controls */}
      <div className="flex items-center gap-2 flex-wrap" style={{ fontSize: 10, fontFamily: 'JetBrains Mono' }}>
        <span className="text-slate-400">Expiry:</span>
        {EXPIRIES.map((e) => (
          <button
            key={e.label}
            onClick={() => setSelectedExpiry(e.label)}
            className={clsx(
              'px-2 py-0.5 rounded border text-[10px] font-mono transition-colors',
              selectedExpiry === e.label
                ? 'border-blue-500/60 text-blue-400 bg-blue-950/30'
                : 'border-[#1e2130] text-slate-500 hover:text-slate-300',
            )}
          >
            {e.label} ({e.dte}d)
          </button>
        ))}
        <span className="ml-auto text-slate-500">
          Spot: ${SPOT.toLocaleString()}
        </span>
      </div>

      {/* Main chain table */}
      <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table style={{ width: '100%', fontSize: 9, fontFamily: 'JetBrains Mono', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #1e2130' }}>
                {/* Call side */}
                <th className="text-right px-2 py-1.5 text-slate-600 font-normal">OI</th>
                <th className="text-right px-2 py-1.5 text-slate-600 font-normal">Vol</th>
                <th className="text-right px-2 py-1.5 text-slate-600 font-normal">IV</th>
                <th className="text-right px-2 py-1.5 text-slate-600 font-normal">Delta</th>
                <th className="text-right px-2 py-1.5 text-slate-600 font-normal">Gamma</th>
                <th className="text-right px-2 py-1.5 text-slate-600 font-normal">Theta</th>
                <th className="text-right px-2 py-1.5 text-slate-600 font-normal">Vega</th>
                <th className="text-right px-2 py-1.5 text-slate-600 font-normal">Bid</th>
                <th className="text-right px-2 py-1.5 text-blue-400/80 font-semibold">CALL</th>
                <th className="text-center px-3 py-1.5 text-slate-300 font-semibold text-xs bg-[#111318]">STRIKE</th>
                <th className="text-left px-2 py-1.5 text-emerald-400/80 font-semibold">PUT</th>
                <th className="text-left px-2 py-1.5 text-slate-600 font-normal">Ask</th>
                <th className="text-left px-2 py-1.5 text-slate-600 font-normal">IV</th>
                <th className="text-left px-2 py-1.5 text-slate-600 font-normal">Delta</th>
                <th className="text-left px-2 py-1.5 text-slate-600 font-normal">Gamma</th>
                <th className="text-left px-2 py-1.5 text-slate-600 font-normal">Theta</th>
                <th className="text-left px-2 py-1.5 text-slate-600 font-normal">Vol</th>
                <th className="text-left px-2 py-1.5 text-slate-600 font-normal">OI</th>
              </tr>
            </thead>
            <tbody>
              {currentChain.map((row) => (
                <tr
                  key={row.strike}
                  style={{
                    background: row.isATM ? 'rgba(59,130,246,0.05)' : undefined,
                    borderBottom: '1px solid rgba(30,33,48,0.4)',
                  }}
                >
                  {/* Call side */}
                  <td className="text-right px-2 py-1 text-slate-600">{row.callOI.toLocaleString()}</td>
                  <td className="text-right px-2 py-1 text-slate-600">{row.callVol}</td>
                  <td className="text-right px-2 py-1 text-amber-400/80">{row.callIV.toFixed(1)}%</td>
                  <td className="text-right px-2 py-1 text-slate-400">{row.callDelta.toFixed(2)}</td>
                  <td className="text-right px-2 py-1 text-slate-500">{row.callGamma.toFixed(4)}</td>
                  <td className="text-right px-2 py-1 text-red-400/70">{row.callTheta.toFixed(4)}</td>
                  <td className="text-right px-2 py-1 text-slate-500">{row.callVega.toFixed(4)}</td>
                  <td className="text-right px-2 py-1 text-slate-400">{formatPrice(row.callBid)}</td>
                  {/* Call bid/ask + action */}
                  <td
                    className="text-right px-2 py-1 cursor-pointer hover:bg-blue-900/20 transition-colors"
                    onClick={() => toggleOption('call', row.strike, 'long')}
                    style={{
                      color: isSelected('call', row.strike) ? '#60a5fa' : '#94a3b8',
                      fontWeight: isSelected('call', row.strike) ? 700 : 400,
                      background: isSelected('call', row.strike) ? 'rgba(59,130,246,0.12)' : undefined,
                    }}
                  >
                    {formatPrice(row.callAsk)}
                  </td>
                  {/* Strike */}
                  <td className="text-center px-3 py-1 bg-[#111318]" style={{ color: row.isATM ? '#e2e8f0' : '#94a3b8', fontWeight: row.isATM ? 700 : 400 }}>
                    {row.strike >= 1000 ? row.strike.toLocaleString() : row.strike}
                    {row.isATM && <span style={{ color: '#3b82f6', fontSize: 7 }}> ATM</span>}
                  </td>
                  {/* Put bid/ask */}
                  <td
                    className="text-left px-2 py-1 cursor-pointer hover:bg-emerald-900/20 transition-colors"
                    onClick={() => toggleOption('put', row.strike, 'long')}
                    style={{
                      color: isSelected('put', row.strike) ? '#4ade80' : '#94a3b8',
                      fontWeight: isSelected('put', row.strike) ? 700 : 400,
                      background: isSelected('put', row.strike) ? 'rgba(34,197,94,0.1)' : undefined,
                    }}
                  >
                    {formatPrice(row.putBid)}
                  </td>
                  <td className="text-left px-2 py-1 text-slate-400">{formatPrice(row.putAsk)}</td>
                  <td className="text-left px-2 py-1 text-amber-400/80">{row.putIV.toFixed(1)}%</td>
                  <td className="text-left px-2 py-1 text-slate-400">{row.putDelta.toFixed(2)}</td>
                  <td className="text-left px-2 py-1 text-slate-500">{row.putGamma.toFixed(4)}</td>
                  <td className="text-left px-2 py-1 text-red-400/70">{row.putTheta.toFixed(4)}</td>
                  <td className="text-left px-2 py-1 text-slate-600">{row.putVol}</td>
                  <td className="text-left px-2 py-1 text-slate-600">{row.putOI.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">

        {/* IV Smile */}
        <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
          <div className="text-[10px] font-mono text-slate-500 uppercase mb-2">IV Smile — {selectedExpiry}</div>
          <SmileChart chain={currentChain} />
        </div>

        {/* Term Structure */}
        <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
          <div className="text-[10px] font-mono text-slate-500 uppercase mb-2">ATM IV Term Structure</div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={termStructure} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="#1a1d26" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="expiry"
                tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `${v.toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 9, fontFamily: 'JetBrains Mono' }}
                formatter={(v: number) => [`${v.toFixed(2)}%`, 'ATM IV']}
              />
              <Line type="monotone" dataKey="atmIV" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3, fill: '#8b5cf6' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* P&L Diagram */}
        <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono text-slate-500 uppercase">P&L Payoff Diagram</span>
            {selectedOptions.length > 0 && (
              <button
                onClick={() => setSelectedOptions([])}
                className="text-[9px] font-mono text-slate-600 hover:text-red-400 border border-[#1e2130] rounded px-1.5 py-0.5"
              >
                Clear
              </button>
            )}
          </div>
          {selectedOptions.length === 0 ? (
            <div className="flex items-center justify-center h-40 text-[10px] font-mono text-slate-700">
              Click cells above to add options
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={payoffData} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
                <CartesianGrid stroke="#1a1d26" strokeDasharray="3 3" />
                <XAxis
                  dataKey="price"
                  tick={{ fill: '#475569', fontSize: 8 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)}
                />
                <YAxis
                  tick={{ fill: '#475569', fontSize: 8 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v: number) => `$${v.toFixed(0)}`}
                />
                <Tooltip
                  formatter={(v: number) => [`$${v.toFixed(0)}`, 'P&L']}
                  contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 9, fontFamily: 'JetBrains Mono' }}
                />
                <ReferenceLine y={0} stroke="#2e3550" />
                <ReferenceLine x={SPOT} stroke="#3b82f6" strokeDasharray="4 2" strokeOpacity={0.5} />
                <Area
                  type="monotone"
                  dataKey="payoff"
                  stroke="#22c55e"
                  strokeWidth={2}
                  fill="rgba(34,197,94,0.1)"
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
          {/* Selected positions */}
          {selectedOptions.length > 0 && (
            <div className="mt-2 flex flex-col gap-1">
              {selectedOptions.map((o, i) => (
                <div key={i} className="flex items-center gap-2 text-[9px] font-mono text-slate-500">
                  <span className={o.type === 'call' ? 'text-blue-400' : 'text-emerald-400'}>{o.type.toUpperCase()}</span>
                  <span>K={o.strike >= 1000 ? `${(o.strike / 1000).toFixed(0)}k` : o.strike}</span>
                  <span>{o.expiry}</span>
                  <span className={o.side === 'long' ? 'text-emerald-500' : 'text-red-500'}>{o.side}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Greeks heatmap */}
      <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-3">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-[10px] font-mono text-slate-500 uppercase">Greeks Heatmap</span>
          <div className="flex items-center gap-1">
            {(['delta', 'gamma', 'theta', 'vega'] as const).map((g) => (
              <button
                key={g}
                onClick={() => setGreekView(g)}
                className={clsx(
                  'px-2 py-0.5 rounded border text-[9px] font-mono transition-colors capitalize',
                  greekView === g
                    ? 'border-blue-500/60 text-blue-400 bg-blue-950/30'
                    : 'border-[#1e2130] text-slate-600 hover:text-slate-400',
                )}
              >
                {g}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            {(['call', 'put'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setGreekOptType(t)}
                className={clsx(
                  'px-2 py-0.5 rounded border text-[9px] font-mono transition-colors capitalize',
                  greekOptType === t
                    ? t === 'call' ? 'border-blue-500/60 text-blue-400 bg-blue-950/30' : 'border-emerald-500/60 text-emerald-400 bg-emerald-950/30'
                    : 'border-[#1e2130] text-slate-600 hover:text-slate-400',
                )}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
        <GreeksHeatmap chains={chains} greek={greekView} optType={greekOptType} />
      </div>

    </div>
  )
}
