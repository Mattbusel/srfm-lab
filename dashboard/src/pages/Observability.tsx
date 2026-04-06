// ============================================================
// Observability.tsx — Real-time quantitative trading observability dashboard
// WebSocket: ws://localhost:8798/ws/dashboard
// ============================================================

import React, { useMemo, useState, useCallback } from 'react'
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
  LabelList,
} from 'recharts'
import { clsx } from 'clsx'
import { useLiveMetrics, circuitBreakerColor } from '@/hooks/useLiveMetrics'
import { MetricsPanel } from '@/components/MetricsPanel'
import { BHPhysicsViz } from '@/components/BHPhysicsViz'
import type {
  EquityPoint,
  DrawdownPoint,
  PositionSizing,
  BHSignal,
  TradeHeatmapCell,
  PnlSlice,
  GreeksSummary,
  CircuitBreakerStatus,
  CircuitBreakerState,
  SparklinePoint,
} from '@/types/metrics'

// ---- Layout constants ------------------------------------------------

const WS_URL = 'ws://localhost:8798/ws/dashboard'
const HEATMAP_HOURS = Array.from({ length: 24 }, (_, i) => i)
const DAYS_OF_WEEK = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const PIE_COLORS = [
  '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#a78bfa',
  '#06b6d4', '#f472b6', '#34d399', '#fb923c', '#818cf8',
]

// ---- Formatting helpers ----------------------------------------------

function fmt$(n: number): string {
  return n >= 1e6
    ? `$${(n / 1e6).toFixed(2)}M`
    : n >= 1e3
    ? `$${(n / 1e3).toFixed(1)}K`
    : `$${n.toFixed(0)}`
}

function fmtPct(n: number): string {
  return `${(n * 100).toFixed(2)}%`
}

function fmtPctSigned(n: number): string {
  const sign = n >= 0 ? '+' : ''
  return `${sign}${(n * 100).toFixed(2)}%`
}

// ---- Connection status badge ----------------------------------------

interface StatusBadgeProps {
  status: string
  reconnectCount: number
  lastMessageAt: Date | null
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, reconnectCount, lastMessageAt }) => {
  const colorMap: Record<string, string> = {
    open:       'bg-green-500',
    connecting: 'bg-yellow-500 animate-pulse',
    closed:     'bg-gray-500',
    error:      'bg-red-500',
  }
  const dot = colorMap[status] ?? 'bg-gray-500'

  const age = lastMessageAt
    ? Math.floor((Date.now() - lastMessageAt.getTime()) / 1000)
    : null

  return (
    <div className="flex items-center gap-2 text-xs text-gray-400">
      <span className={clsx('w-2 h-2 rounded-full', dot)} />
      <span className="capitalize">{status}</span>
      {reconnectCount > 0 && <span className="text-gray-600">({reconnectCount} reconnects)</span>}
      {age !== null && <span className="text-gray-600">{age}s ago</span>}
    </div>
  )
}

// ---- Equity curve panel ---------------------------------------------

interface EquityCurveProps {
  data: EquityPoint[]
  isLoading: boolean
}

const EquityCurvePanel: React.FC<EquityCurveProps> = ({ data, isLoading }) => {
  const chartData = useMemo(
    () => data.map(p => ({
      time: p.timestamp.slice(5, 16).replace('T', ' '),
      equity: p.equity,
      dailyPnl: p.dailyPnl,
    })),
    [data]
  )

  const minVal = useMemo(() => Math.min(...data.map(d => d.equity)) * 0.998, [data])
  const maxVal = useMemo(() => Math.max(...data.map(d => d.equity)) * 1.002, [data])

  const isPositive = data.length >= 2
    ? data[data.length - 1].equity >= data[0].equity
    : true

  const gradientId = 'equityGradient'
  const lineColor = isPositive ? '#22c55e' : '#ef4444'
  const fillColor = isPositive ? '#16a34a' : '#dc2626'

  if (isLoading && !data.length) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm">
        Awaiting equity data…
      </div>
    )
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: 8 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={fillColor} stopOpacity={0.25} />
              <stop offset="95%" stopColor={fillColor} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
          <XAxis
            dataKey="time"
            tick={{ fill: '#6b7280', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[minVal, maxVal]}
            tick={{ fill: '#6b7280', fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            width={60}
            tickFormatter={v => fmt$(v)}
          />
          <Tooltip
            contentStyle={{
              background: '#111827',
              border: '1px solid #374151',
              borderRadius: 8,
              fontSize: 11,
            }}
            labelStyle={{ color: '#d1d5db' }}
            formatter={(value: number, name: string) => [
              name === 'equity' ? fmt$(value) : fmt$(value),
              name === 'equity' ? 'Equity' : 'Daily P&L',
            ]}
          />
          <Area
            type="monotone"
            dataKey="equity"
            stroke={lineColor}
            strokeWidth={2}
            fill={`url(#${gradientId})`}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---- Drawdown waterfall chart ----------------------------------------

interface DrawdownChartProps {
  data: DrawdownPoint[]
}

const DrawdownChart: React.FC<DrawdownChartProps> = ({ data }) => {
  const chartData = useMemo(
    () => data.map(p => ({
      time: p.timestamp.slice(5, 16).replace('T', ' '),
      drawdown: p.drawdown * 100,
    })),
    [data]
  )

  if (!chartData.length) return null

  return (
    <ResponsiveContainer width="100%" height={100}>
      <AreaChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: 8 }}>
        <defs>
          <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
            <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
        <XAxis dataKey="time" tick={{ fill: '#6b7280', fontSize: 8 }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={{ fill: '#6b7280', fontSize: 8 }} axisLine={false} tickLine={false} width={40} tickFormatter={v => `${v.toFixed(1)}%`} />
        <Tooltip
          contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 6, fontSize: 10 }}
          formatter={(v: number) => [`${v.toFixed(2)}%`, 'Drawdown']}
        />
        <ReferenceLine y={-10} stroke="#f59e0b" strokeDasharray="4 2" label={{ value: '-10%', fill: '#f59e0b', fontSize: 9, position: 'insideTopRight' }} />
        <ReferenceLine y={-20} stroke="#ef4444" strokeDasharray="4 2" label={{ value: '-20%', fill: '#ef4444', fontSize: 9, position: 'insideTopRight' }} />
        <Area
          type="monotone"
          dataKey="drawdown"
          stroke="#ef4444"
          strokeWidth={1.5}
          fill="url(#ddGradient)"
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ---- Position sizing bars -------------------------------------------

interface PositionSizingProps {
  data: PositionSizing[]
}

const PositionSizingPanel: React.FC<PositionSizingProps> = ({ data }) => {
  const sorted = useMemo(
    () => [...data].sort((a, b) => b.weight - a.weight).slice(0, 15),
    [data]
  )

  if (!sorted.length) {
    return <div className="text-xs text-gray-500 py-4 text-center">No open positions</div>
  }

  return (
    <div className="space-y-2">
      {sorted.map(pos => {
        const kellyGapColor = pos.kellyGap > 0.05 ? '#22c55e' : pos.kellyGap < -0.05 ? '#ef4444' : '#6b7280'
        return (
          <div key={pos.symbol} className="flex items-center gap-2">
            <span className="text-xs font-mono text-gray-300 w-14 flex-shrink-0">{pos.symbol}</span>
            <div className="flex-1 relative h-4 bg-gray-800 rounded overflow-hidden">
              {/* Kelly target */}
              <div
                className="absolute top-0 h-full border-r-2 border-yellow-400/50"
                style={{ width: `${Math.min(pos.kellyTarget * 100, 100)}%` }}
              />
              {/* Current weight */}
              <div
                className="absolute top-0 h-full rounded"
                style={{
                  width: `${Math.min(pos.weight * 100, 100)}%`,
                  background: pos.weight > pos.maxWeight ? '#ef4444' : '#3b82f6',
                  opacity: 0.8,
                }}
              />
            </div>
            <span className="text-xs tabular-nums text-gray-400 w-12 text-right">
              {(pos.weight * 100).toFixed(1)}%
            </span>
            <span
              className="text-xs tabular-nums w-12 text-right"
              style={{ color: kellyGapColor }}
            >
              {pos.kellyGap >= 0 ? '+' : ''}{(pos.kellyGap * 100).toFixed(1)}%
            </span>
          </div>
        )
      })}
      <div className="flex gap-4 pt-1 text-xs text-gray-600">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 border-t-2 border-yellow-400/50" />
          Kelly target
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-2 rounded bg-blue-500/70" />
          Current weight
        </span>
      </div>
    </div>
  )
}

// ---- Circuit breaker status -----------------------------------------

const circuitBreakerLabel: Record<CircuitBreakerState, string> = {
  OPEN:      'TRIPPED',
  HALF_OPEN: 'TESTING',
  CLOSED:    'NORMAL',
}

const CircuitBreakerPanel: React.FC<{ status: CircuitBreakerStatus | null }> = ({ status }) => {
  const color = circuitBreakerColor(status)
  const isTripped = status?.state === 'OPEN'
  const label = status ? circuitBreakerLabel[status.state] : 'UNKNOWN'

  return (
    <div
      className="rounded-xl border p-3 flex flex-col gap-2"
      style={{
        borderColor: color + '66',
        background: color + '11',
      }}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-400">CIRCUIT BREAKER</span>
        <div className="flex items-center gap-2">
          <span
            className={clsx(
              'w-3 h-3 rounded-full',
              isTripped ? 'animate-pulse' : ''
            )}
            style={{ background: color }}
          />
          <span className="text-xs font-bold" style={{ color }}>
            {label}
          </span>
        </div>
      </div>

      {status && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          <span className="text-gray-500">Daily Loss</span>
          <span className="text-right tabular-nums" style={{ color: status.dailyLossUsd > 0 ? '#ef4444' : '#6b7280' }}>
            {fmt$(status.dailyLossUsd)} / {fmt$(status.dailyLossLimit)}
          </span>

          <span className="text-gray-500">Drawdown</span>
          <span className="text-right tabular-nums" style={{ color: Math.abs(status.drawdownPct) > 0.1 ? '#f59e0b' : '#6b7280' }}>
            {fmtPct(status.drawdownPct)} / {fmtPct(status.drawdownLimit)}
          </span>

          <span className="text-gray-500">Consec. Losses</span>
          <span className="text-right tabular-nums" style={{ color: status.consecutiveLosses >= 3 ? '#f59e0b' : '#6b7280' }}>
            {status.consecutiveLosses}
          </span>

          {status.reason && (
            <>
              <span className="text-gray-500">Reason</span>
              <span className="text-right text-gray-300 truncate">{status.reason}</span>
            </>
          )}
        </div>
      )}

      {!status && (
        <span className="text-xs text-gray-600">No status received</span>
      )}
    </div>
  )
}

// ---- Trade frequency heatmap ----------------------------------------

interface TradeHeatmapProps {
  data: TradeHeatmapCell[]
}

const TradeFrequencyHeatmap: React.FC<TradeHeatmapProps> = ({ data }) => {
  const maxFills = useMemo(
    () => Math.max(...data.map(d => d.fillCount), 1),
    [data]
  )

  const cellMap = useMemo(() => {
    const map = new Map<string, TradeHeatmapCell>()
    for (const cell of data) {
      map.set(`${cell.dayOfWeek}:${cell.hour}`, cell)
    }
    return map
  }, [data])

  if (!data.length) {
    return <div className="text-xs text-gray-500 py-4 text-center">No trade data</div>
  }

  return (
    <div className="overflow-x-auto">
      <div className="flex gap-0.5">
        {/* Hour labels */}
        <div className="flex flex-col gap-0.5 mr-1">
          <div className="h-4" />  {/* spacer for day row */}
          {DAYS_OF_WEEK.map(d => (
            <div key={d} className="h-4 text-xs text-gray-600 flex items-center" style={{ fontSize: 9, width: 24 }}>
              {d}
            </div>
          ))}
        </div>

        {/* Grid */}
        <div className="flex flex-col gap-0.5">
          {/* Hour header */}
          <div className="flex gap-0.5 mb-0.5">
            {HEATMAP_HOURS.map(h => (
              <div key={h} className="w-4 text-center text-gray-600" style={{ fontSize: 7 }}>
                {h % 6 === 0 ? h : ''}
              </div>
            ))}
          </div>

          {/* Day rows */}
          {Array.from({ length: 7 }, (_, day) => (
            <div key={day} className="flex gap-0.5">
              {HEATMAP_HOURS.map(hour => {
                const cell = cellMap.get(`${day}:${hour}`)
                const intensity = cell ? cell.fillCount / maxFills : 0
                const pnlPositive = (cell?.avgPnl ?? 0) >= 0

                return (
                  <div
                    key={hour}
                    className="w-4 h-4 rounded-sm cursor-default"
                    style={{
                      background: intensity === 0
                        ? '#111827'
                        : pnlPositive
                        ? `rgba(34, 197, 94, ${0.15 + intensity * 0.75})`
                        : `rgba(239, 68, 68, ${0.15 + intensity * 0.75})`,
                      border: '1px solid #1f2937',
                    }}
                    title={cell ? `${DAYS_OF_WEEK[day]} ${hour}:00 — ${cell.fillCount} fills, avg P&L: ${fmt$(cell.avgPnl)}` : ''}
                  />
                )
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-4 mt-2 text-xs text-gray-600">
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm bg-green-500/50" />
          Profitable hour
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm bg-red-500/50" />
          Loss hour
        </span>
      </div>
    </div>
  )
}

// ---- P&L Attribution pie ---------------------------------------------

interface PnlPieProps {
  data: PnlSlice[]
  title: string
}

const PnlPieChart: React.FC<PnlPieProps> = ({ data, title }) => {
  const [activeIndex, setActiveIndex] = useState<number | null>(null)

  const pieData = useMemo(
    () => data
      .filter(d => Math.abs(d.value) > 0)
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, 10),
    [data]
  )

  if (!pieData.length) {
    return <div className="flex items-center justify-center h-32 text-gray-500 text-xs">No P&L data</div>
  }

  const total = pieData.reduce((s, d) => s + d.value, 0)

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium text-gray-400">{title}</span>
      <div className="flex gap-4 items-center">
        <ResponsiveContainer width={120} height={120}>
          <PieChart>
            <Pie
              data={pieData}
              dataKey="value"
              cx="50%"
              cy="50%"
              innerRadius={30}
              outerRadius={55}
              paddingAngle={2}
              isAnimationActive={false}
              onMouseEnter={(_, i) => setActiveIndex(i)}
              onMouseLeave={() => setActiveIndex(null)}
            >
              {pieData.map((entry, i) => (
                <Cell
                  key={entry.label}
                  fill={entry.color ?? PIE_COLORS[i % PIE_COLORS.length]}
                  opacity={activeIndex === null || activeIndex === i ? 1 : 0.5}
                  stroke="transparent"
                />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 6, fontSize: 10 }}
              formatter={(v: number, name: string, item) => {
                const slice = item.payload as PnlSlice | undefined
                return [
                  `${fmt$(v)}${slice ? ` (${slice.pct.toFixed(1)}%)` : ''}`,
                  slice?.label ?? name,
                ]
              }}
            />
          </PieChart>
        </ResponsiveContainer>

        <div className="flex flex-col gap-1 flex-1 min-w-0">
          {pieData.slice(0, 6).map((d, i) => (
            <div key={d.label} className="flex items-center gap-1.5 min-w-0">
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: d.color ?? PIE_COLORS[i % PIE_COLORS.length] }}
              />
              <span className="text-xs text-gray-400 truncate flex-1">{d.label}</span>
              <span
                className="text-xs tabular-nums flex-shrink-0"
                style={{ color: d.value >= 0 ? '#22c55e' : '#ef4444' }}
              >
                {fmt$(d.value)}
              </span>
            </div>
          ))}
          <div className="mt-1 text-xs text-gray-500 border-t border-gray-800 pt-1">
            Total: <span className={total >= 0 ? 'text-green-400' : 'text-red-400'}>{fmt$(total)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// ---- Greeks summary panel -------------------------------------------

const GreeksPanel: React.FC<{ greeks: GreeksSummary | null }> = ({ greeks }) => {
  if (!greeks) {
    return (
      <div className="text-xs text-gray-600 py-2 text-center">
        No options positions active
      </div>
    )
  }

  const rows = [
    { label: 'Net Delta', value: greeks.netDelta.toFixed(4), color: Math.abs(greeks.netDelta) > 0.5 ? '#f59e0b' : '#22c55e' },
    { label: 'Gamma', value: greeks.gamma.toFixed(4), color: '#6b7280' },
    { label: 'Theta/day', value: `$${greeks.theta.toFixed(0)}`, color: greeks.theta < -500 ? '#ef4444' : '#6b7280' },
    { label: 'Vega/1%', value: `$${greeks.vega.toFixed(0)}`, color: '#6b7280' },
    { label: 'Rho', value: greeks.rho.toFixed(4), color: '#6b7280' },
    { label: 'Notional', value: fmt$(greeks.notionalExposure), color: '#9ca3af' },
  ]

  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
      {rows.map(({ label, value, color }) => (
        <React.Fragment key={label}>
          <span className="text-xs text-gray-500">{label}</span>
          <span className="text-xs tabular-nums font-mono text-right" style={{ color }}>{value}</span>
        </React.Fragment>
      ))}
    </div>
  )
}

// ---- Section card wrapper -------------------------------------------

interface SectionCardProps {
  title: string
  subtitle?: string
  children: React.ReactNode
  className?: string
  rightSlot?: React.ReactNode
}

const SectionCard: React.FC<SectionCardProps> = ({ title, subtitle, children, className, rightSlot }) => (
  <div className={clsx('bg-gray-900 rounded-xl border border-gray-800 p-4 flex flex-col gap-3', className)}>
    <div className="flex items-center justify-between">
      <div>
        <h3 className="text-sm font-semibold text-gray-200">{title}</h3>
        {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
      </div>
      {rightSlot}
    </div>
    {children}
  </div>
)

// ---- Main page -------------------------------------------------------

const Observability: React.FC = () => {
  const {
    status,
    reconnect,
    isConnected,
    reconnectCount,
    lastMessageAt,
    portfolio,
    equityCurve,
    drawdown,
    positionSizing,
    bhSignals,
    circuitBreaker,
    tradeHeatmap,
    pnlBySymbol,
    pnlByAssetClass,
    greeks,
    riskMetrics,
  } = useLiveMetrics({ url: WS_URL })

  const [activeTab, setActiveTab] = useState<'overview' | 'signals' | 'risk' | 'attribution'>('overview')

  // Derive sparkline from equity curve
  const equitySparkline = useMemo(
    (): SparklinePoint[] => equityCurve.slice(-60).map(p => ({ value: p.equity })),
    [equityCurve]
  )

  const handleReconnect = useCallback(() => {
    reconnect()
  }, [reconnect])

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 space-y-4">

      {/* ---- Header ---- */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white">
            Observability
          </h1>
          <p className="text-xs text-gray-500">Real-time portfolio analytics & signal monitoring</p>
        </div>
        <div className="flex items-center gap-4">
          <StatusBadge
            status={status}
            reconnectCount={reconnectCount}
            lastMessageAt={lastMessageAt}
          />
          {!isConnected && (
            <button
              onClick={handleReconnect}
              className="text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors"
            >
              Reconnect
            </button>
          )}
        </div>
      </div>

      {/* ---- Tab navigation ---- */}
      <div className="flex gap-1 bg-gray-900 rounded-xl p-1 border border-gray-800 w-fit">
        {(['overview', 'signals', 'risk', 'attribution'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={clsx(
              'px-4 py-1.5 rounded-lg text-xs font-medium transition-colors capitalize',
              activeTab === tab
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-gray-200'
            )}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* ---- Metrics Panel (always visible) ---- */}
      <MetricsPanel
        portfolio={portfolio}
        riskMetrics={riskMetrics}
        equitySparkline={equitySparkline}
      />

      {/* ============================================================
          OVERVIEW TAB
      ============================================================ */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-12 gap-4">

          {/* Equity curve */}
          <SectionCard
            title="Equity Curve"
            subtitle={portfolio ? `${fmt$(portfolio.totalEquity)} · ${fmtPctSigned(portfolio.dailyPnlPct)} today` : 'Connecting…'}
            className="col-span-12 lg:col-span-8"
          >
            <EquityCurvePanel data={equityCurve} isLoading={!isConnected} />
            {drawdown.length > 0 && (
              <div>
                <p className="text-xs text-gray-600 mb-1">Drawdown</p>
                <DrawdownChart data={drawdown} />
              </div>
            )}
          </SectionCard>

          {/* Circuit breaker + Greeks */}
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-4">
            <CircuitBreakerPanel status={circuitBreaker} />

            <SectionCard title="Options Greeks" subtitle="Portfolio-level exposure">
              <GreeksPanel greeks={greeks} />
            </SectionCard>
          </div>

          {/* Position sizing */}
          <SectionCard
            title="Position Sizing"
            subtitle="Current vs Kelly-optimal weights"
            className="col-span-12 lg:col-span-6"
          >
            <PositionSizingPanel data={positionSizing} />
          </SectionCard>

          {/* Trade heatmap */}
          <SectionCard
            title="Trade Frequency Heatmap"
            subtitle="Fills per hour of day × day of week"
            className="col-span-12 lg:col-span-6"
          >
            <TradeFrequencyHeatmap data={tradeHeatmap} />
          </SectionCard>

          {/* P&L attribution pies */}
          <SectionCard
            title="P&L Attribution"
            subtitle="By symbol and asset class"
            className="col-span-12"
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <PnlPieChart data={pnlBySymbol} title="By Symbol" />
              <PnlPieChart data={pnlByAssetClass} title="By Asset Class" />
            </div>
          </SectionCard>

        </div>
      )}

      {/* ============================================================
          SIGNALS TAB
      ============================================================ */}
      {activeTab === 'signals' && (
        <div className="grid grid-cols-12 gap-4">

          {/* BH Physics Viz */}
          <BHPhysicsViz
            signals={bhSignals}
            className="col-span-12"
          />

          {/* Active signals summary */}
          <SectionCard
            title="Active Signal Summary"
            subtitle={`${bhSignals.filter(s => s.spacetimeType !== 'NONE').length} signals active`}
            className="col-span-12 lg:col-span-6"
          >
            <div className="space-y-2">
              {bhSignals
                .filter(s => Math.abs(s.strengthValue) > 0.3)
                .sort((a, b) => Math.abs(b.strengthValue) - Math.abs(a.strengthValue))
                .slice(0, 12)
                .map(sig => {
                  const isLong = sig.strengthValue > 0
                  return (
                    <div key={`${sig.symbol}:${sig.timeframe}`} className="flex items-center gap-2 text-xs">
                      <span className="font-mono text-gray-300 w-14">{sig.symbol}</span>
                      <span className="text-gray-600 w-8">{sig.timeframe}</span>
                      <div className="flex-1 bg-gray-800 rounded h-1.5 overflow-hidden">
                        <div
                          className="h-full rounded"
                          style={{
                            width: `${Math.abs(sig.strengthValue) * 100}%`,
                            background: isLong ? '#22c55e' : '#ef4444',
                            marginLeft: isLong ? 0 : 'auto',
                          }}
                        />
                      </div>
                      <span
                        className="w-14 text-right tabular-nums"
                        style={{ color: isLong ? '#22c55e' : '#ef4444' }}
                      >
                        {isLong ? '▲' : '▼'} {Math.abs(sig.strengthValue * 100).toFixed(1)}%
                      </span>
                      <span className="text-gray-600 w-16 text-right">
                        {(sig.confidence * 100).toFixed(0)}% conf
                      </span>
                    </div>
                  )
                })}
              {bhSignals.filter(s => Math.abs(s.strengthValue) > 0.3).length === 0 && (
                <p className="text-xs text-gray-500">No strong signals active</p>
              )}
            </div>
          </SectionCard>

          {/* Position sizing alongside signals */}
          <SectionCard
            title="Position Sizing vs Kelly"
            className="col-span-12 lg:col-span-6"
          >
            <PositionSizingPanel data={positionSizing} />
          </SectionCard>
        </div>
      )}

      {/* ============================================================
          RISK TAB
      ============================================================ */}
      {activeTab === 'risk' && (
        <div className="grid grid-cols-12 gap-4">

          {/* Risk gauges */}
          <SectionCard
            title="Tail Risk Metrics"
            subtitle="VaR, CVaR, distribution statistics"
            className="col-span-12 lg:col-span-6"
          >
            {riskMetrics ? (
              <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
                {[
                  { label: 'VaR 99% 1D', value: fmt$(riskMetrics.var99_1d), danger: riskMetrics.var99_1d > 30_000 },
                  { label: 'CVaR 99% 1D', value: fmt$(riskMetrics.cvar99_1d), danger: riskMetrics.cvar99_1d > 50_000 },
                  { label: 'VaR 99% 10D', value: fmt$(riskMetrics.var99_10d), danger: false },
                  { label: 'Ann. Volatility', value: fmtPct(riskMetrics.volatilityAnn), danger: riskMetrics.volatilityAnn > 0.4 },
                  { label: 'Skewness', value: riskMetrics.skewness.toFixed(3), danger: riskMetrics.skewness < -1 },
                  { label: 'Excess Kurtosis', value: riskMetrics.excessKurtosis.toFixed(3), danger: riskMetrics.excessKurtosis > 5 },
                  { label: 'Sharpe (ann)', value: riskMetrics.sharpeRatio.toFixed(3), danger: riskMetrics.sharpeRatio < 0.5 },
                  { label: 'Sortino (ann)', value: riskMetrics.sortinoRatio.toFixed(3), danger: riskMetrics.sortinoRatio < 0.8 },
                  { label: 'Calmar', value: riskMetrics.calmarRatio.toFixed(3), danger: riskMetrics.calmarRatio < 0.5 },
                  { label: 'Max Drawdown', value: fmtPct(riskMetrics.maxDrawdown), danger: riskMetrics.maxDrawdown < -0.15 },
                  { label: 'Beta (vs BTC)', value: riskMetrics.beta.toFixed(3), danger: riskMetrics.beta > 1.2 },
                  { label: 'Win Rate', value: fmtPct(riskMetrics.winRate), danger: riskMetrics.winRate < 0.45 },
                ].map(({ label, value, danger }) => (
                  <React.Fragment key={label}>
                    <span className="text-gray-500">{label}</span>
                    <span className="text-right tabular-nums" style={{ color: danger ? '#f59e0b' : '#d1d5db' }}>
                      {value}
                    </span>
                  </React.Fragment>
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-500">Awaiting risk data…</p>
            )}
          </SectionCard>

          {/* Drawdown */}
          <SectionCard
            title="Drawdown History"
            className="col-span-12 lg:col-span-6"
          >
            <DrawdownChart data={drawdown} />
            {riskMetrics && (
              <div className="flex gap-6 text-xs mt-2">
                <div>
                  <span className="text-gray-500">Max DD: </span>
                  <span className="text-red-400">{fmtPct(riskMetrics.maxDrawdown)}</span>
                </div>
                <div>
                  <span className="text-gray-500">Current: </span>
                  <span className={riskMetrics.currentDrawdown < -0.05 ? 'text-red-400' : 'text-gray-300'}>
                    {fmtPct(riskMetrics.currentDrawdown)}
                  </span>
                </div>
              </div>
            )}
          </SectionCard>

          {/* Circuit breaker */}
          <div className="col-span-12 lg:col-span-4">
            <CircuitBreakerPanel status={circuitBreaker} />
          </div>

        </div>
      )}

      {/* ============================================================
          ATTRIBUTION TAB
      ============================================================ */}
      {activeTab === 'attribution' && (
        <div className="grid grid-cols-12 gap-4">

          <SectionCard
            title="P&L by Symbol"
            subtitle="Realized + unrealized attribution"
            className="col-span-12 lg:col-span-6"
          >
            <PnlPieChart data={pnlBySymbol} title="By Symbol" />
          </SectionCard>

          <SectionCard
            title="P&L by Asset Class"
            className="col-span-12 lg:col-span-6"
          >
            <PnlPieChart data={pnlByAssetClass} title="By Asset Class" />
          </SectionCard>

          {/* P&L bar chart */}
          <SectionCard
            title="Symbol P&L Bar Chart"
            className="col-span-12"
          >
            {pnlBySymbol.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart
                  data={[...pnlBySymbol].sort((a, b) => b.value - a.value).slice(0, 20)}
                  layout="vertical"
                  margin={{ top: 4, right: 16, bottom: 4, left: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fill: '#6b7280', fontSize: 9 }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={v => fmt$(v)}
                  />
                  <YAxis
                    type="category"
                    dataKey="label"
                    tick={{ fill: '#9ca3af', fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                    width={55}
                  />
                  <Tooltip
                    contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 6, fontSize: 10 }}
                    formatter={(v: number) => [fmt$(v), 'P&L']}
                  />
                  <ReferenceLine x={0} stroke="#374151" />
                  <Bar dataKey="value" radius={[0, 3, 3, 0]} isAnimationActive={false}>
                    {pnlBySymbol.map((entry, i) => (
                      <Cell
                        key={entry.label}
                        fill={entry.value >= 0 ? '#16a34a' : '#dc2626'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-xs text-gray-500">No attribution data</p>
            )}
          </SectionCard>

          {/* Trade frequency */}
          <SectionCard
            title="Trade Frequency Heatmap"
            subtitle="P&L color by profitability"
            className="col-span-12"
          >
            <TradeFrequencyHeatmap data={tradeHeatmap} />
          </SectionCard>

        </div>
      )}

    </div>
  )
}

export default Observability
