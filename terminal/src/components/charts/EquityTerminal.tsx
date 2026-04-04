// ============================================================
// EquityTerminal — equity curve chart with analytics
// ============================================================
import React, { useMemo, useState } from 'react'
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
} from 'recharts'
import { format } from 'date-fns'
import { usePortfolioStore } from '@/store/portfolioStore'
import type { EquityPoint } from '@/types'

interface EquityTerminalProps {
  height?: number
  showBenchmark?: boolean
  showDrawdown?: boolean
  showRegimes?: boolean
  className?: string
}

type DateRangePreset = '1W' | '1M' | '3M' | '6M' | '1Y' | 'YTD' | 'ALL'

const DATE_RANGES: DateRangePreset[] = ['1W', '1M', '3M', '6M', '1Y', 'YTD', 'ALL']

const RANGE_DAYS: Record<DateRangePreset, number> = {
  '1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365, 'YTD': 0, 'ALL': 9999,
}

function formatCurrency(v: number): string {
  return v >= 1000 ? `$${(v / 1000).toFixed(1)}K` : `$${v.toFixed(0)}`
}

function formatDate(ts: number): string {
  return format(new Date(ts), 'MMM d')
}

function computeMetrics(points: EquityPoint[]) {
  if (points.length < 2) return null

  const initial = points[0].equity
  const final = points.at(-1)!.equity
  const totalReturn = (final - initial) / initial
  const days = (points.at(-1)!.timestamp - points[0].timestamp) / 86400000
  const annualizedReturn = days > 0 ? Math.pow(1 + totalReturn, 365 / days) - 1 : 0

  // Daily returns
  const dailyReturns: number[] = []
  for (let i = 1; i < points.length; i++) {
    if (points[i - 1].equity > 0) {
      dailyReturns.push((points[i].equity - points[i - 1].equity) / points[i - 1].equity)
    }
  }

  const avgReturn = dailyReturns.reduce((s, r) => s + r, 0) / dailyReturns.length
  const variance = dailyReturns.reduce((s, r) => s + Math.pow(r - avgReturn, 2), 0) / dailyReturns.length
  const stddev = Math.sqrt(variance)
  const downReturns = dailyReturns.filter((r) => r < 0)
  const downVariance = downReturns.reduce((s, r) => s + r * r, 0) / (downReturns.length || 1)
  const downStddev = Math.sqrt(downVariance)
  const sharpe = stddev > 0 ? (avgReturn / stddev) * Math.sqrt(252) : 0
  const sortino = downStddev > 0 ? (avgReturn / downStddev) * Math.sqrt(252) : 0

  // Max drawdown
  let peak = initial
  let maxDD = 0
  let maxDDPct = 0
  for (const p of points) {
    peak = Math.max(peak, p.equity)
    const dd = peak - p.equity
    if (dd > maxDD) {
      maxDD = dd
      maxDDPct = dd / peak
    }
  }

  const calmar = maxDDPct > 0 ? annualizedReturn / maxDDPct : 0

  return { totalReturn, annualizedReturn, sharpe, sortino, maxDD, maxDDPct, calmar, days }
}

interface ChartPoint {
  timestamp: number
  equity: number
  drawdown: number
  drawdownPct: number
  totalPnl: number
}

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean
  payload?: Array<{ value: number; color: string; name: string }>
  label?: number
}) => {
  if (!active || !payload?.length) return null

  return (
    <div className="bg-terminal-surface border border-terminal-border rounded p-2 text-xs font-mono shadow-lg">
      <div className="text-terminal-subtle mb-1">{label ? formatDate(label) : ''}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center justify-between gap-4">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="text-terminal-text">
            {p.name.includes('Drawdown') ? `${(p.value * 100).toFixed(2)}%` : formatCurrency(p.value)}
          </span>
        </div>
      ))}
    </div>
  )
}

export const EquityTerminal: React.FC<EquityTerminalProps> = ({
  height = 300,
  showBenchmark = false,
  showDrawdown = true,
  className = '',
}) => {
  const equityHistory = usePortfolioStore((s) => s.equityHistory)
  const [selectedRange, setSelectedRange] = useState<DateRangePreset>('1M')

  // Filter by date range
  const filteredData = useMemo((): ChartPoint[] => {
    if (!equityHistory.length) {
      // Generate mock equity curve
      const now = Date.now()
      const mockPoints: ChartPoint[] = []
      let equity = 100000
      let peak = equity
      for (let i = 365; i >= 0; i--) {
        const dailyReturn = (Math.random() - 0.48) * 0.015
        equity = Math.max(equity * (1 + dailyReturn), 1)
        peak = Math.max(peak, equity)
        const dd = peak - equity
        mockPoints.push({
          timestamp: now - i * 86400000,
          equity,
          drawdown: dd,
          drawdownPct: dd / peak,
          totalPnl: equity - 100000,
        })
      }
      return mockPoints
    }

    const rangeDays = RANGE_DAYS[selectedRange]
    const cutoff = selectedRange === 'YTD'
      ? new Date(new Date().getFullYear(), 0, 1).getTime()
      : rangeDays < 9999
      ? Date.now() - rangeDays * 86400000
      : 0

    const filtered = equityHistory.filter((p) => p.timestamp >= cutoff)

    // Compute drawdown
    let peak = filtered[0]?.equity ?? 1
    return filtered.map((p) => {
      peak = Math.max(peak, p.equity)
      const dd = peak - p.equity
      return {
        timestamp: p.timestamp,
        equity: p.equity,
        drawdown: dd,
        drawdownPct: dd / peak,
        totalPnl: p.totalPnl,
      }
    })
  }, [equityHistory, selectedRange])

  const metrics = useMemo(() => computeMetrics(
    filteredData.map((d) => ({ ...d, cash: 0, longValue: 0, shortValue: 0, dayPnl: 0 }))
  ), [filteredData])

  const initialEquity = filteredData[0]?.equity ?? 0
  const currentEquity = filteredData.at(-1)?.equity ?? 0
  const totalReturn = initialEquity > 0 ? (currentEquity - initialEquity) / initialEquity : 0
  const maxDrawdownPct = Math.max(...filteredData.map((d) => d.drawdownPct), 0)

  return (
    <div className={`flex flex-col bg-terminal-bg ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border flex-shrink-0">
        <div className="flex items-center gap-4">
          <div>
            <div className="text-terminal-subtle text-[10px] font-mono">Equity</div>
            <div className="text-terminal-text font-mono text-sm font-semibold">
              {formatCurrency(currentEquity)}
            </div>
          </div>
          <div>
            <div className="text-terminal-subtle text-[10px] font-mono">Return</div>
            <div className={`font-mono text-sm font-semibold ${totalReturn >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
              {totalReturn >= 0 ? '+' : ''}{(totalReturn * 100).toFixed(2)}%
            </div>
          </div>
          {metrics && (
            <>
              <div className="hidden md:block">
                <div className="text-terminal-subtle text-[10px] font-mono">Sharpe</div>
                <div className="font-mono text-sm text-terminal-text">{metrics.sharpe.toFixed(2)}</div>
              </div>
              <div className="hidden md:block">
                <div className="text-terminal-subtle text-[10px] font-mono">Max DD</div>
                <div className="font-mono text-sm text-terminal-bear">{(metrics.maxDDPct * 100).toFixed(2)}%</div>
              </div>
              <div className="hidden lg:block">
                <div className="text-terminal-subtle text-[10px] font-mono">Sortino</div>
                <div className="font-mono text-sm text-terminal-text">{metrics.sortino.toFixed(2)}</div>
              </div>
              <div className="hidden lg:block">
                <div className="text-terminal-subtle text-[10px] font-mono">CAGR</div>
                <div className={`font-mono text-sm ${metrics.annualizedReturn >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
                  {(metrics.annualizedReturn * 100).toFixed(1)}%
                </div>
              </div>
            </>
          )}
        </div>

        {/* Range selector */}
        <div className="flex items-center gap-1">
          {DATE_RANGES.map((r) => (
            <button
              key={r}
              onClick={() => setSelectedRange(r)}
              className={`px-2 py-0.5 text-xs rounded font-mono transition-colors ${
                selectedRange === r
                  ? 'bg-terminal-accent text-white'
                  : 'text-terminal-subtle hover:text-terminal-text hover:bg-terminal-muted'
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div style={{ height }} className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={filteredData} margin={{ top: 8, right: 8, left: 0, bottom: showDrawdown ? 60 : 8 }}>
            <CartesianGrid stroke="#1f2937" strokeDasharray="2 4" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatDate}
              tick={{ fill: '#9ca3af', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
              stroke="#1f2937"
              tickLine={false}
              minTickGap={60}
            />
            <YAxis
              yAxisId="equity"
              tickFormatter={formatCurrency}
              tick={{ fill: '#9ca3af', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
              stroke="#1f2937"
              tickLine={false}
              width={55}
            />
            {showDrawdown && (
              <YAxis
                yAxisId="dd"
                orientation="right"
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{ fill: '#9ca3af', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
                stroke="#1f2937"
                tickLine={false}
                width={40}
              />
            )}
            <Tooltip content={<CustomTooltip />} />

            {/* Equity curve */}
            <Area
              yAxisId="equity"
              type="monotone"
              dataKey="equity"
              name="Equity"
              stroke="#3b82f6"
              strokeWidth={2}
              fill="rgba(59, 130, 246, 0.08)"
              dot={false}
              activeDot={{ r: 3, fill: '#3b82f6' }}
            />

            {/* Drawdown shading */}
            {showDrawdown && (
              <Area
                yAxisId="dd"
                type="monotone"
                dataKey="drawdownPct"
                name="Drawdown"
                stroke="#ef4444"
                strokeWidth={1}
                fill="rgba(239, 68, 68, 0.12)"
                dot={false}
              />
            )}

            {/* Zero line */}
            <ReferenceLine yAxisId="equity" y={initialEquity} stroke="#4b5563" strokeDasharray="4 4" />

            {filteredData.length > 50 && (
              <Brush
                dataKey="timestamp"
                height={20}
                stroke="#374151"
                fill="#111827"
                tickFormatter={formatDate}
                startIndex={Math.max(0, filteredData.length - 90)}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default EquityTerminal
