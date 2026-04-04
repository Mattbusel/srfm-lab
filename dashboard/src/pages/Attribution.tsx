// ============================================================
// Attribution.tsx — P&L attribution page
// ============================================================
import React, { useEffect, useMemo, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Legend,
} from 'recharts'
import { clsx } from 'clsx'
import { Card, Select } from '@/components/ui'
import { RegimeBadge } from '@/components/RegimeBadge'
import type { AttributionEntry, MarketRegime } from '@/types'

// ---- Mock data ----

const INSTRUMENT_ATTR: AttributionEntry[] = [
  { label: 'BTC', pnl: 8450, pnlPct: 0.134, contribution: 0.346, trades: 24, winRate: 0.71 },
  { label: 'ETH', pnl: 5230, pnlPct: 0.089, contribution: 0.214, trades: 31, winRate: 0.65 },
  { label: 'SOL', pnl: 3180, pnlPct: 0.054, contribution: 0.130, trades: 18, winRate: 0.67 },
  { label: 'BNB', pnl: 2100, pnlPct: 0.037, contribution: 0.086, trades: 14, winRate: 0.64 },
  { label: 'LINK', pnl: 1850, pnlPct: 0.032, contribution: 0.076, trades: 22, winRate: 0.59 },
  { label: 'AVAX', pnl: -780, pnlPct: -0.013, contribution: -0.032, trades: 12, winRate: 0.42 },
  { label: 'DOGE', pnl: 3640, pnlPct: 0.062, contribution: 0.149, trades: 9, winRate: 0.78 },
  { label: 'UNI', pnl: -420, pnlPct: -0.007, contribution: -0.017, trades: 8, winRate: 0.38 },
  { label: 'AAVE', pnl: 980, pnlPct: 0.017, contribution: 0.040, trades: 11, winRate: 0.55 },
  { label: 'MATIC', pnl: 210, pnlPct: 0.004, contribution: 0.009, trades: 7, winRate: 0.57 },
]

const STRATEGY_ATTR: AttributionEntry[] = [
  { label: 'BH_Trend', pnl: 14200, pnlPct: 0.238, contribution: 0.581, trades: 45, winRate: 0.72 },
  { label: 'BH_Swing', pnl: 5800, pnlPct: 0.097, contribution: 0.237, trades: 38, winRate: 0.63 },
  { label: 'BH_Scalp', pnl: 2100, pnlPct: 0.035, contribution: 0.086, trades: 62, winRate: 0.58 },
  { label: 'BH_Short', pnl: 1900, pnlPct: 0.032, contribution: 0.078, trades: 21, winRate: 0.52 },
  { label: 'Momentum', pnl: 760, pnlPct: 0.013, contribution: 0.031, trades: 17, winRate: 0.53 },
  { label: 'DeFi_Rev', pnl: -380, pnlPct: -0.006, contribution: -0.016, trades: 9, winRate: 0.44 },
]

const TIMEFRAME_ATTR: AttributionEntry[] = [
  { label: '1d',  pnl: 11200, pnlPct: 0.188, contribution: 0.458, trades: 28, winRate: 0.75 },
  { label: '4h',  pnl: 7400,  pnlPct: 0.124, contribution: 0.303, trades: 54, winRate: 0.67 },
  { label: '1h',  pnl: 3800,  pnlPct: 0.064, contribution: 0.155, trades: 78, winRate: 0.60 },
  { label: '15m', pnl: 1800,  pnlPct: 0.030, contribution: 0.074, trades: 96, winRate: 0.55 },
  { label: '5m',  pnl: -760,  pnlPct: -0.013, contribution: -0.031, trades: 112, winRate: 0.48 },
]

const REGIME_ATTR: (AttributionEntry & { regime: MarketRegime })[] = [
  { label: 'Trending Up',   regime: 'trending_up',   pnl: 16800, pnlPct: 0.282, contribution: 0.687, trades: 112, winRate: 0.71 },
  { label: 'Ranging',       regime: 'ranging',        pnl: 4200,  pnlPct: 0.070, contribution: 0.172, trades: 88,  winRate: 0.55 },
  { label: 'Trending Down', regime: 'trending_down',  pnl: 2100,  pnlPct: 0.035, contribution: 0.086, trades: 34,  winRate: 0.62 },
  { label: 'Volatile',      regime: 'volatile',       pnl: -660,  pnlPct: -0.011, contribution: -0.027, trades: 22, winRate: 0.41 },
]

// ---- Waterfall chart for instrument attribution ----

const WaterfallChart: React.FC<{ data: AttributionEntry[] }> = ({ data }) => {
  const chartData = useMemo(() => {
    const sorted = [...data].sort((a, b) => b.pnl - a.pnl)
    let cumulative = 0
    return sorted.map((d) => {
      const base = d.pnl >= 0 ? cumulative : cumulative + d.pnl
      const result = {
        label: d.label,
        pnl: d.pnl,
        base,
        value: Math.abs(d.pnl),
        isPositive: d.pnl >= 0,
        cumulative,
      }
      cumulative += d.pnl
      return result
    })
  }, [data])

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
        <CartesianGrid stroke="#1e2130" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: '#475569', fontSize: 9, fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: number) => `$${v >= 0 ? '' : '-'}${Math.abs(v / 1000).toFixed(1)}k`}
        />
        <Tooltip
          formatter={(v: number, name: string) =>
            name === 'base' ? null : [`$${v >= 0 ? '' : '-'}${Math.abs(v).toFixed(0)}`, 'P&L']
          }
          contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 10, fontFamily: 'JetBrains Mono' }}
        />
        {/* transparent base bar for waterfall effect */}
        <Bar dataKey="base" stackId="a" fill="transparent" />
        <Bar dataKey="value" stackId="a" radius={[3, 3, 0, 0]}>
          {chartData.map((e, i) => (
            <Cell key={i} fill={e.isPositive ? '#22c55e' : '#ef4444'} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ---- Attribution table ----

const AttributionTable: React.FC<{ data: AttributionEntry[] }> = ({ data }) => {
  const sorted = useMemo(() => [...data].sort((a, b) => b.pnl - a.pnl), [data])
  const maxPnl = Math.max(...sorted.map((d) => Math.abs(d.pnl)))

  return (
    <table className="w-full text-[10px] font-mono">
      <thead>
        <tr className="border-b border-[#1e2130]">
          {['Label', 'P&L', 'Contrib', 'Win Rate', 'Trades', 'Bar'].map((h) => (
            <th key={h} className="text-left py-1.5 px-2 text-slate-600 font-normal uppercase tracking-wider whitespace-nowrap">
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map((d) => (
          <tr key={d.label} className="border-b border-[#1a1d26] hover:bg-[#13161e]">
            <td className="py-1.5 px-2 text-slate-200 font-semibold">{d.label}</td>
            <td className={clsx('py-1.5 px-2 font-semibold', d.pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
              {d.pnl >= 0 ? '+' : ''}${d.pnl.toLocaleString('en-US', { maximumFractionDigits: 0 })}
            </td>
            <td className={clsx('py-1.5 px-2', d.contribution >= 0 ? 'text-emerald-400' : 'text-red-400')}>
              {(d.contribution * 100).toFixed(1)}%
            </td>
            <td className="py-1.5 px-2 text-slate-400">
              {(d.winRate * 100).toFixed(0)}%
            </td>
            <td className="py-1.5 px-2 text-slate-500">{d.trades}</td>
            <td className="py-1.5 px-2 w-24">
              <div className="h-1.5 bg-[#1e2130] rounded-full overflow-hidden">
                <div
                  className={clsx('h-full rounded-full', d.pnl >= 0 ? 'bg-emerald-500/70' : 'bg-red-500/70')}
                  style={{ width: `${(Math.abs(d.pnl) / maxPnl) * 100}%` }}
                />
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// ---- Page ----

export const Attribution: React.FC = () => {
  const [period, setPeriod] = useState('30d')

  const regimePieData = useMemo(() =>
    REGIME_ATTR.filter((r) => r.pnl > 0).map((r) => ({
      name: r.label,
      value: r.pnl,
    })),
  [])

  const PIE_COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6']

  return (
    <div className="flex flex-col h-full overflow-y-auto thin-scrollbar p-4 gap-4">

      {/* Period selector */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-mono font-semibold text-slate-200">P&L Attribution</h2>
        <Select
          value={period}
          onChange={setPeriod}
          options={[
            { value: '1d', label: '1 Day' },
            { value: '7d', label: '7 Days' },
            { value: '30d', label: '30 Days' },
            { value: '90d', label: '90 Days' },
            { value: 'ytd', label: 'YTD' },
          ]}
        />
      </div>

      {/* Waterfall — by instrument */}
      <Card title="By Instrument" subtitle="Waterfall P&L contribution">
        <WaterfallChart data={INSTRUMENT_ATTR} />
      </Card>

      {/* Instrument + Strategy tables */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card title="Instrument Attribution" padding="none">
          <div className="px-4 pb-4">
            <AttributionTable data={INSTRUMENT_ATTR} />
          </div>
        </Card>
        <Card title="Strategy Attribution" padding="none">
          <div className="px-4 pb-4">
            <AttributionTable data={STRATEGY_ATTR} />
          </div>
        </Card>
      </div>

      {/* Timeframe + Regime */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card title="By Timeframe" padding="none">
          <div className="px-4 pb-4">
            <AttributionTable data={TIMEFRAME_ATTR} />
          </div>
        </Card>

        <Card title="By Regime">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={regimePieData}
                  cx="50%"
                  cy="50%"
                  innerRadius="45%"
                  outerRadius="70%"
                  dataKey="value"
                  paddingAngle={2}
                >
                  {regimePieData.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} fillOpacity={0.85} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v: number) => [`$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}`, 'P&L']}
                  contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 10, fontFamily: 'JetBrains Mono' }}
                />
                <Legend
                  formatter={(value: string) => <span style={{ fontSize: 9, fontFamily: 'JetBrains Mono', color: '#94a3b8' }}>{value}</span>}
                />
              </PieChart>
            </ResponsiveContainer>

            <div className="flex flex-col gap-2 justify-center">
              {REGIME_ATTR.map((r) => (
                <div key={r.label} className="flex items-center justify-between">
                  <RegimeBadge regime={r.regime} size="xs" />
                  <div className="flex items-center gap-2">
                    <span className={clsx(
                      'text-[10px] font-mono font-semibold',
                      r.pnl >= 0 ? 'text-emerald-400' : 'text-red-400',
                    )}>
                      {r.pnl >= 0 ? '+' : ''}${r.pnl.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </span>
                    <span className="text-[9px] font-mono text-slate-600">
                      ({(r.winRate * 100).toFixed(0)}% WR)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>

      {/* BH Signal contribution */}
      <Card title="BH Signal Contribution" subtitle="P&L by signal state at entry">
        <div className="grid grid-cols-3 gap-4">
          {[
            { state: 'Bullish Entry', pnl: 18400, trades: 156, winRate: 0.69, color: '#22c55e' },
            { state: 'Bearish Entry', pnl: 4200,  trades: 48,  winRate: 0.58, color: '#ef4444' },
            { state: 'Neutral Entry', pnl: 1840,  trades: 62,  winRate: 0.52, color: '#475569' },
          ].map((s) => (
            <div key={s.state} className="bg-[#0e1017] rounded-lg p-3 border border-[#1e2130]">
              <div className="text-[9px] font-mono text-slate-500 mb-2 uppercase">{s.state}</div>
              <div className="text-lg font-mono font-semibold" style={{ color: s.color }}>
                +${s.pnl.toLocaleString()}
              </div>
              <div className="text-[10px] font-mono text-slate-500 mt-1">
                {s.trades} trades · {(s.winRate * 100).toFixed(0)}% WR
              </div>
            </div>
          ))}
        </div>
      </Card>

    </div>
  )
}
