// ============================================================
// TradeHistory.tsx — Trade log page
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
  ScatterChart,
  Scatter,
  ZAxis,
} from 'recharts'
import { clsx } from 'clsx'
import { format } from 'date-fns'
import { Card, StatCard, Badge, Select } from '@/components/ui'
import { RegimeBadge } from '@/components/RegimeBadge'
import { BHStateIndicator } from '@/components/BHStateIndicator'
import { useTradesStore } from '@/store/tradesStore'

// ---- P&L Histogram ----

const PnLHistogram: React.FC<{ trades: { pnl: number }[] }> = ({ trades }) => {
  const buckets = useMemo(() => {
    const vals = trades.map((t) => t.pnl)
    const min = Math.min(...vals)
    const max = Math.max(...vals)
    const range = max - min
    const n = 20
    const step = range / n
    const data: { bucket: string; count: number; isPositive: boolean }[] = []
    for (let i = 0; i < n; i++) {
      const lo = min + i * step
      const hi = lo + step
      data.push({
        bucket: `${lo >= 0 ? '+' : ''}$${lo.toFixed(0)}`,
        count: vals.filter((v) => v >= lo && v < hi).length,
        isPositive: lo >= 0,
      })
    }
    return data
  }, [trades])

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={buckets} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
        <CartesianGrid stroke="#1e2130" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="bucket"
          tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
          interval={4}
        />
        <YAxis
          tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          formatter={(v: number) => [v, 'Trades']}
          contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 10, fontFamily: 'JetBrains Mono' }}
        />
        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
          {buckets.map((b, i) => (
            <Cell key={i} fill={b.isPositive ? '#22c55e' : '#ef4444'} fillOpacity={0.75} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ---- Trade duration distribution ----

const DurationChart: React.FC<{ trades: { durationMs: number; pnl: number }[] }> = ({ trades }) => {
  const data = useMemo(() =>
    trades.map((t) => ({
      duration: t.durationMs / 3600000,  // hours
      pnl: t.pnl,
      size: Math.abs(t.pnl),
    })),
  [trades])

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ScatterChart margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
        <CartesianGrid stroke="#1e2130" strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="duration"
          name="Duration (h)"
          tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: number) => `${v.toFixed(0)}h`}
        />
        <YAxis
          type="number"
          dataKey="pnl"
          name="P&L"
          tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: number) => `$${v.toFixed(0)}`}
        />
        <ZAxis dataKey="size" range={[20, 200]} />
        <Tooltip
          cursor={{ stroke: '#1e2130' }}
          contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 10, fontFamily: 'JetBrains Mono' }}
          formatter={(v: number, name: string) => [
            name === 'P&L' ? `$${v.toFixed(2)}` : `${v.toFixed(1)}h`,
            name,
          ]}
        />
        <Scatter
          data={data}
          fill="#3b82f6"
          fillOpacity={0.6}
        />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

// ---- Win/Loss streak tracker ----

const StreakTracker: React.FC<{ trades: { pnl: number }[] }> = ({ trades }) => {
  const { streaks, currentStreak, currentStreakType } = useMemo(() => {
    const streaks: { type: 'win' | 'loss'; length: number }[] = []
    let cur = 0
    let curType: 'win' | 'loss' | null = null

    for (const t of [...trades].reverse()) {
      const isWin = t.pnl > 0
      if (curType === null) {
        curType = isWin ? 'win' : 'loss'
        cur = 1
      } else if ((isWin && curType === 'win') || (!isWin && curType === 'loss')) {
        cur++
      } else {
        streaks.push({ type: curType, length: cur })
        curType = isWin ? 'win' : 'loss'
        cur = 1
      }
    }
    if (curType) streaks.push({ type: curType, length: cur })

    const maxWin = Math.max(...streaks.filter((s) => s.type === 'win').map((s) => s.length), 0)
    const maxLoss = Math.max(...streaks.filter((s) => s.type === 'loss').map((s) => s.length), 0)
    const currentStreak = streaks[streaks.length - 1]?.length ?? 0
    const currentStreakType = streaks[streaks.length - 1]?.type ?? 'win'

    return { streaks, currentStreak, currentStreakType, maxWin, maxLoss }
  }, [trades])

  // Visual streak dots — last 30
  const last30 = trades.slice(0, 30).map((t) => t.pnl > 0)

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1">
        {last30.map((win, i) => (
          <div
            key={i}
            title={win ? 'Win' : 'Loss'}
            className={clsx(
              'w-3 h-3 rounded-sm',
              win ? 'bg-emerald-500/70' : 'bg-red-500/70',
            )}
          />
        ))}
      </div>
      <div className="flex items-center gap-4">
        <div className="flex flex-col items-center">
          <span className="text-[9px] font-mono text-slate-600 uppercase">Current</span>
          <span className={clsx(
            'text-lg font-mono font-bold',
            currentStreakType === 'win' ? 'text-emerald-400' : 'text-red-400',
          )}>
            {currentStreak}
          </span>
          <span className={clsx(
            'text-[9px] font-mono',
            currentStreakType === 'win' ? 'text-emerald-600' : 'text-red-600',
          )}>
            {currentStreakType === 'win' ? 'wins' : 'losses'}
          </span>
        </div>
        {streaks.filter((s) => s.type === 'win').length > 0 && (
          <div className="flex flex-col items-center">
            <span className="text-[9px] font-mono text-slate-600 uppercase">Best Win</span>
            <span className="text-lg font-mono font-bold text-emerald-400">
              {Math.max(...streaks.filter((s) => s.type === 'win').map((s) => s.length))}
            </span>
            <span className="text-[9px] font-mono text-emerald-600">streak</span>
          </div>
        )}
        {streaks.filter((s) => s.type === 'loss').length > 0 && (
          <div className="flex flex-col items-center">
            <span className="text-[9px] font-mono text-slate-600 uppercase">Worst Loss</span>
            <span className="text-lg font-mono font-bold text-red-400">
              {Math.max(...streaks.filter((s) => s.type === 'loss').map((s) => s.length))}
            </span>
            <span className="text-[9px] font-mono text-red-600">streak</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ---- Trade table ----

const TradeTable: React.FC<{ trades: ReturnType<typeof useTradesStore>['trades'] }> = ({ trades }) => {
  const formatDuration = (ms: number) => {
    if (ms < 3600000) return `${Math.round(ms / 60000)}m`
    if (ms < 86400000) return `${(ms / 3600000).toFixed(1)}h`
    return `${(ms / 86400000).toFixed(1)}d`
  }

  return (
    <div className="overflow-x-auto thin-scrollbar">
      <table className="w-full text-[10px] font-mono">
        <thead>
          <tr className="border-b border-[#1e2130]">
            {['Symbol', 'Side', 'P&L', 'P&L %', 'Duration', 'Entry', 'Exit', 'Strategy', 'Regime', 'BH State', 'Exit Reason'].map((h) => (
              <th key={h} className="text-left py-1.5 px-2 text-slate-600 font-normal uppercase tracking-wider whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {trades.slice(0, 50).map((t) => (
            <tr key={t.id} className="border-b border-[#1a1d26] hover:bg-[#13161e] transition-colors">
              <td className="py-1.5 px-2 text-slate-200 font-semibold">{t.symbol.replace('USDT', '')}</td>
              <td className="py-1.5 px-2">
                <Badge variant={t.side === 'long' ? 'bull' : 'bear'}>{t.side.toUpperCase()}</Badge>
              </td>
              <td className={clsx('py-1.5 px-2 font-semibold', t.pnl >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                {t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(0)}
              </td>
              <td className={clsx('py-1.5 px-2 font-semibold', t.pnlPct >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                {t.pnlPct >= 0 ? '+' : ''}{(t.pnlPct * 100).toFixed(2)}%
              </td>
              <td className="py-1.5 px-2 text-slate-500">{formatDuration(t.durationMs)}</td>
              <td className="py-1.5 px-2 text-slate-400">{format(new Date(t.entryTime), 'MM/dd HH:mm')}</td>
              <td className="py-1.5 px-2 text-slate-400">{format(new Date(t.exitTime), 'MM/dd HH:mm')}</td>
              <td className="py-1.5 px-2 text-slate-500">{t.strategy}</td>
              <td className="py-1.5 px-2">
                <RegimeBadge regime={t.regime} size="xs" showIcon={false} />
              </td>
              <td className="py-1.5 px-2">
                <BHStateIndicator state={t.bhSignalAtEntry} showLabel={false} size="xs" />
              </td>
              <td className="py-1.5 px-2">
                <Badge variant={t.exitReason === 'tp' ? 'bull' : t.exitReason === 'sl' ? 'bear' : 'neutral'}>
                  {t.exitReason.toUpperCase()}
                </Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---- Page ----

export const TradeHistory: React.FC = () => {
  const { trades, initMockData, getFiltered, setFilter, filterSymbol, filterStrategy, filterSide } = useTradesStore()

  useEffect(() => {
    initMockData()
  }, [initMockData])

  const filtered = getFiltered()

  const stats = useMemo(() => {
    if (!filtered.length) return null
    const wins = filtered.filter((t) => t.pnl > 0)
    const losses = filtered.filter((t) => t.pnl < 0)
    const totalPnl = filtered.reduce((s, t) => s + t.pnl, 0)
    const avgWin = wins.length ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0
    const avgLoss = losses.length ? losses.reduce((s, t) => s + t.pnl, 0) / losses.length : 0
    const avgDuration = filtered.reduce((s, t) => s + t.durationMs, 0) / filtered.length
    const profitFactor = wins.length && losses.length
      ? Math.abs(wins.reduce((s, t) => s + t.pnl, 0) / losses.reduce((s, t) => s + t.pnl, 0))
      : 0
    const expectancy = (wins.length / filtered.length) * avgWin - (losses.length / filtered.length) * Math.abs(avgLoss)
    return { wins: wins.length, losses: losses.length, totalPnl, avgWin, avgLoss, avgDuration, profitFactor, expectancy }
  }, [filtered])

  const bestWorst = useMemo(() => {
    const sorted = [...filtered].sort((a, b) => b.pnl - a.pnl)
    return { best: sorted.slice(0, 5), worst: sorted.slice(-5).reverse() }
  }, [filtered])

  const allSymbols = [...new Set(trades.map((t) => t.symbol))]
  const allStrategies = [...new Set(trades.map((t) => t.strategy))]

  return (
    <div className="flex flex-col h-full overflow-y-auto thin-scrollbar p-4 gap-4">

      {/* Filters */}
      <div className="flex items-center gap-2 flex-wrap">
        <Select
          value={filterSymbol ?? ''}
          onChange={(v) => setFilter('filterSymbol', v || null)}
          options={[{ value: '', label: 'All Symbols' }, ...allSymbols.map((s) => ({ value: s, label: s.replace('USDT', '') }))]}
        />
        <Select
          value={filterStrategy ?? ''}
          onChange={(v) => setFilter('filterStrategy', v || null)}
          options={[{ value: '', label: 'All Strategies' }, ...allStrategies.map((s) => ({ value: s, label: s }))]}
        />
        <Select
          value={filterSide ?? ''}
          onChange={(v) => setFilter('filterSide', v || null)}
          options={[
            { value: '', label: 'Both Sides' },
            { value: 'long', label: 'Long Only' },
            { value: 'short', label: 'Short Only' },
          ]}
        />
        <span className="text-[10px] font-mono text-slate-600 ml-auto">
          {filtered.length} trades
        </span>
      </div>

      {/* Stats row */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-3">
          <StatCard label="Total P&L" value={`${stats.totalPnl >= 0 ? '+' : ''}$${stats.totalPnl.toFixed(0)}`}
            valueClass={clsx('text-lg', stats.totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400')} />
          <StatCard label="Win Rate" value={`${((stats.wins / filtered.length) * 100).toFixed(1)}%`} />
          <StatCard label="Avg Win" value={`+$${stats.avgWin.toFixed(0)}`} valueClass="text-lg text-emerald-400" />
          <StatCard label="Avg Loss" value={`-$${Math.abs(stats.avgLoss).toFixed(0)}`} valueClass="text-lg text-red-400" />
          <StatCard label="Profit Factor" value={stats.profitFactor.toFixed(2)} valueClass="text-lg text-blue-400" />
          <StatCard label="Expectancy" value={`$${stats.expectancy.toFixed(0)}`}
            valueClass={clsx('text-lg', stats.expectancy >= 0 ? 'text-emerald-400' : 'text-red-400')} />
          <StatCard label="Avg Duration" value={`${(stats.avgDuration / 3600000).toFixed(1)}h`} />
        </div>
      )}

      {/* Charts row */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card title="P&L Distribution" subtitle="Trade outcome histogram">
          <PnLHistogram trades={filtered} />
        </Card>
        <Card title="Win/Loss Streak" subtitle="Last 30 trades">
          <StreakTracker trades={filtered} />
        </Card>
        <Card title="Duration vs P&L" subtitle="Scatter plot">
          <DurationChart trades={filtered} />
        </Card>
      </div>

      {/* Best / Worst trades */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card title="Best Trades" padding="none">
          <div className="px-4 pb-4 flex flex-col gap-1.5">
            {bestWorst.best.map((t) => (
              <div key={t.id} className="flex items-center gap-2 py-1.5 border-b border-[#1a1d26] text-[10px] font-mono">
                <span className="text-slate-200 w-12 font-semibold">{t.symbol.replace('USDT', '')}</span>
                <Badge variant={t.side === 'long' ? 'bull' : 'bear'}>{t.side[0].toUpperCase()}</Badge>
                <span className="text-emerald-400 font-semibold flex-1">+${t.pnl.toFixed(0)}</span>
                <span className="text-emerald-500/80">(+{(t.pnlPct * 100).toFixed(1)}%)</span>
                <span className="text-slate-600">{t.strategy}</span>
              </div>
            ))}
          </div>
        </Card>
        <Card title="Worst Trades" padding="none">
          <div className="px-4 pb-4 flex flex-col gap-1.5">
            {bestWorst.worst.map((t) => (
              <div key={t.id} className="flex items-center gap-2 py-1.5 border-b border-[#1a1d26] text-[10px] font-mono">
                <span className="text-slate-200 w-12 font-semibold">{t.symbol.replace('USDT', '')}</span>
                <Badge variant={t.side === 'long' ? 'bull' : 'bear'}>{t.side[0].toUpperCase()}</Badge>
                <span className="text-red-400 font-semibold flex-1">${t.pnl.toFixed(0)}</span>
                <span className="text-red-500/80">({(t.pnlPct * 100).toFixed(1)}%)</span>
                <span className="text-slate-600">{t.strategy}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Full trade log */}
      <Card title="Trade Log" subtitle={`${filtered.length} trades — showing first 50`} padding="none">
        <div className="px-4 pb-4">
          <TradeTable trades={filtered} />
        </div>
      </Card>

    </div>
  )
}
