// ============================================================
// Overview.tsx — Portfolio overview page
// ============================================================
import React, { useEffect, useMemo } from 'react'
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
import { usePortfolioStore } from '@/store/portfolioStore'
import { usePositionsStore } from '@/store/positionsStore'
import { Card, StatCard, Badge, ProgressBar, LoadingSpinner } from '@/components/ui'
import { EquityCurve } from '@/components/EquityCurve'
import type { Position } from '@/types'

// ---- Win Rate Gauge ----

const WinRateGauge: React.FC<{ rate: number }> = ({ rate }) => {
  const pct = Math.round(rate * 100)
  const angle = rate * 180
  const cx = 70
  const cy = 70
  const r = 52
  const strokeW = 10

  function arc(deg: number) {
    const rad = ((deg - 180) * Math.PI) / 180
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
  }

  const start = arc(0)
  const fill = arc(angle)
  const largeArc = angle > 90 ? 1 : 0

  const color = rate > 0.65 ? '#22c55e' : rate > 0.5 ? '#3b82f6' : '#f59e0b'

  return (
    <div className="flex flex-col items-center">
      <svg width={140} height={80} viewBox="0 0 140 80">
        {/* Track */}
        <path
          d={`M ${arc(0).x} ${arc(0).y} A ${r} ${r} 0 1 1 ${arc(180).x} ${arc(180).y}`}
          fill="none"
          stroke="#1e2130"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />
        {/* Fill */}
        {rate > 0 && (
          <path
            d={`M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${fill.x} ${fill.y}`}
            fill="none"
            stroke={color}
            strokeWidth={strokeW}
            strokeLinecap="round"
          />
        )}
        <text x={cx} y={cy + 2} textAnchor="middle" fill={color} fontSize={22} fontFamily="JetBrains Mono" fontWeight={700}>
          {pct}%
        </text>
        <text x={cx} y={cy + 16} textAnchor="middle" fill="#475569" fontSize={9} fontFamily="JetBrains Mono">
          WIN RATE
        </text>
      </svg>
    </div>
  )
}

// ---- Positions Table ----

const PositionsTable: React.FC<{ positions: Position[] }> = ({ positions }) => {
  return (
    <div className="overflow-x-auto thin-scrollbar">
      <table className="w-full text-[10px] font-mono">
        <thead>
          <tr className="border-b border-[#1e2130]">
            {['Symbol', 'Side', 'Size (USD)', 'Entry', 'Current', 'P&L', 'P&L %', 'Weight'].map((h) => (
              <th key={h} className="text-left py-1.5 px-2 text-slate-600 font-normal uppercase tracking-wider">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {positions.map((p) => (
            <tr key={p.symbol} className="border-b border-[#1a1d26] hover:bg-[#13161e] transition-colors">
              <td className="py-1.5 px-2 text-slate-200 font-semibold">{p.symbol.replace('USDT', '')}</td>
              <td className="py-1.5 px-2">
                <Badge variant={p.side === 'long' ? 'bull' : 'bear'}>
                  {p.side.toUpperCase()}
                </Badge>
              </td>
              <td className="py-1.5 px-2 text-slate-300">
                ${p.sizeUsd.toLocaleString('en-US', { maximumFractionDigits: 0 })}
              </td>
              <td className="py-1.5 px-2 text-slate-400">
                {p.entryPrice >= 1000
                  ? p.entryPrice.toLocaleString('en-US', { maximumFractionDigits: 0 })
                  : p.entryPrice >= 1
                    ? p.entryPrice.toFixed(3)
                    : p.entryPrice.toFixed(5)
                }
              </td>
              <td className="py-1.5 px-2 text-slate-400">
                {p.currentPrice >= 1000
                  ? p.currentPrice.toLocaleString('en-US', { maximumFractionDigits: 0 })
                  : p.currentPrice >= 1
                    ? p.currentPrice.toFixed(3)
                    : p.currentPrice.toFixed(5)
                }
              </td>
              <td className={clsx('py-1.5 px-2 font-semibold', p.unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                {p.unrealizedPnl >= 0 ? '+' : ''}${p.unrealizedPnl.toFixed(0)}
              </td>
              <td className={clsx('py-1.5 px-2 font-semibold', p.unrealizedPnlPct >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                {p.unrealizedPnlPct >= 0 ? '+' : ''}{(p.unrealizedPnlPct * 100).toFixed(2)}%
              </td>
              <td className="py-1.5 px-2">
                <div className="flex items-center gap-1.5">
                  <div className="w-10 h-1.5 bg-[#1e2130] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-blue-500/70"
                      style={{ width: `${p.weight * 100}%` }}
                    />
                  </div>
                  <span className="text-slate-500">{(p.weight * 100).toFixed(1)}%</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---- Page ----

export const Overview: React.FC = () => {
  const { snapshot, equityCurve, dailyPnl, loading, initMockData } = usePortfolioStore()
  const { positions, initMockData: initPositions } = usePositionsStore()

  useEffect(() => {
    initMockData()
    initPositions()
  }, [initMockData, initPositions])

  const recentEquity = useMemo(
    () => equityCurve.slice(-72),  // last 3 days hourly
    [equityCurve],
  )

  const sparklineData = useMemo(
    () => equityCurve.slice(-24).map((p, i) => ({ i, equity: p.equity })),
    [equityCurve],
  )

  const dailyPnlColored = useMemo(
    () => dailyPnl.map((d) => ({ ...d, color: d.pnl >= 0 ? '#22c55e' : '#ef4444' })),
    [dailyPnl],
  )

  if (loading || !snapshot) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner />
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto thin-scrollbar p-4 gap-4">

      {/* Top stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
        <StatCard
          label="Total Equity"
          value={`$${snapshot.totalEquity.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          change={snapshot.dailyPnlPct}
          changeLabel="24h"
          className="col-span-2"
          valueClass="text-2xl"
        />
        <StatCard
          label="Daily P&L"
          value={`${snapshot.dailyPnl >= 0 ? '+' : ''}$${snapshot.dailyPnl.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          valueClass={clsx('text-xl', snapshot.dailyPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}
        />
        <StatCard
          label="Unrealized P&L"
          value={`${snapshot.totalUnrealizedPnl >= 0 ? '+' : ''}$${snapshot.totalUnrealizedPnl.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          valueClass={clsx('text-xl', snapshot.totalUnrealizedPnl >= 0 ? 'text-emerald-400' : 'text-red-400')}
        />
        <StatCard
          label="Sharpe Ratio"
          value={snapshot.sharpeRatio.toFixed(2)}
          valueClass="text-xl text-blue-400"
        />
        <StatCard
          label="Max Drawdown"
          value={`${(snapshot.maxDrawdown * 100).toFixed(1)}%`}
          valueClass="text-xl text-amber-400"
        />
      </div>

      {/* Equity curve + Daily P&L */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card title="Equity Curve" subtitle="With drawdown overlay" className="xl:col-span-2">
          <EquityCurve data={recentEquity} height={220} showDrawdown />
        </Card>

        <Card title="Daily P&L" subtitle="Last 30 days">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={dailyPnlColored} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="#1e2130" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
                interval={6}
                tickFormatter={(v: string) => v.slice(5)}
              />
              <YAxis
                tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `$${v >= 0 ? '' : '-'}${Math.abs(v / 1000).toFixed(1)}k`}
              />
              <Tooltip
                formatter={(v: number) => [`$${v.toFixed(0)}`, 'P&L']}
                contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 10, fontFamily: 'JetBrains Mono' }}
              />
              <ReferenceLine y={0} stroke="#2e3550" />
              <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
                {dailyPnlColored.map((e, i) => (
                  <Cell key={i} fill={e.color} fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Performance metrics row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card className="flex items-center justify-center">
          <WinRateGauge rate={snapshot.winRate} />
        </Card>
        <StatCard label="Calmar Ratio" value={snapshot.calmarRatio.toFixed(2)} valueClass="text-xl text-blue-400" />
        <StatCard label="Sortino Ratio" value={snapshot.sortinoRatio.toFixed(2)} valueClass="text-xl text-blue-400" />
        <StatCard label="Volatility" value={`${(snapshot.volatility * 100).toFixed(1)}%`} valueClass="text-xl text-amber-400" />
      </div>

      {/* P&L summary row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="7-Day P&L" value={`$${snapshot.weeklyPnl.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          valueClass={clsx('text-lg', snapshot.weeklyPnl >= 0 ? 'text-emerald-400' : 'text-red-400')} />
        <StatCard label="30-Day P&L" value={`$${snapshot.monthlyPnl.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          valueClass={clsx('text-lg', snapshot.monthlyPnl >= 0 ? 'text-emerald-400' : 'text-red-400')} />
        <StatCard label="YTD P&L" value={`$${snapshot.ytdPnl.toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          valueClass={clsx('text-lg', snapshot.ytdPnl >= 0 ? 'text-emerald-400' : 'text-red-400')} />
        <Card>
          <div className="text-[10px] font-mono text-slate-500 mb-2 uppercase tracking-wider">Margin Usage</div>
          <ProgressBar
            value={snapshot.marginUtilization}
            color={snapshot.marginUtilization > 0.8 ? '#ef4444' : snapshot.marginUtilization > 0.6 ? '#f59e0b' : '#3b82f6'}
            showValue
            label={`$${snapshot.totalMarginUsed.toLocaleString('en-US', { maximumFractionDigits: 0 })} / $${(snapshot.totalMarginUsed + snapshot.availableMargin).toLocaleString('en-US', { maximumFractionDigits: 0 })}`}
          />
        </Card>
      </div>

      {/* Equity sparkline */}
      <Card title="24h Equity Sparkline" padding="sm">
        <ResponsiveContainer width="100%" height={60}>
          <LineChart data={sparklineData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <Line
              type="monotone"
              dataKey="equity"
              stroke="#3b82f6"
              strokeWidth={1.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Positions table */}
      <Card title="Open Positions" subtitle={`${positions.length} positions`} padding="none">
        <div className="px-4 pb-4">
          <PositionsTable positions={positions} />
        </div>
      </Card>

    </div>
  )
}
