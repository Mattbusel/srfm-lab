// ============================================================
// HeatmapChart — calendar heatmap of daily returns
// ============================================================
import React, { useMemo, useState } from 'react'
import { format, startOfYear, eachDayOfInterval, getDay, getWeek, getYear } from 'date-fns'
import { usePortfolioStore } from '@/store/portfolioStore'

interface HeatmapChartProps {
  year?: number
  className?: string
}

interface DayData {
  date: Date
  returnPct: number
  pnl: number
  trades: number
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * Math.clamp(t, 0, 1)

// Clamp polyfill
Math.clamp = (v: number, min: number, max: number) => Math.min(Math.max(v, min), max)

function getHeatmapColor(returnPct: number, maxAbsReturn: number): string {
  if (maxAbsReturn === 0) return '#1f2937'
  const t = Math.abs(returnPct) / maxAbsReturn
  if (returnPct > 0) {
    // Greens
    const r = Math.round(lerp(17, 34, t))
    const g = Math.round(lerp(24, 197, t))
    const b = Math.round(lerp(39, 94, t))
    return `rgb(${r},${g},${b})`
  } else if (returnPct < 0) {
    // Reds
    const r = Math.round(lerp(17, 239, t))
    const g = Math.round(lerp(24, 68, t))
    const b = Math.round(lerp(39, 68, t))
    return `rgb(${r},${g},${b})`
  }
  return '#1f2937'
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const DAYS_OF_WEEK = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

export const HeatmapChart: React.FC<HeatmapChartProps> = ({
  year = new Date().getFullYear(),
  className = '',
}) => {
  const equityHistory = usePortfolioStore((s) => s.equityHistory)
  const [hoveredDay, setHoveredDay] = useState<DayData | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  // Build daily return map
  const dailyDataMap = useMemo(() => {
    const map = new Map<string, DayData>()

    if (equityHistory.length === 0) {
      // Mock data
      const start = new Date(year, 0, 1)
      const end = new Date(year, 11, 31)
      const days = eachDayOfInterval({ start, end })
      for (const day of days) {
        if (getDay(day) === 0 || getDay(day) === 6) continue
        const returnPct = (Math.random() - 0.48) * 0.04
        map.set(format(day, 'yyyy-MM-dd'), {
          date: day,
          returnPct,
          pnl: returnPct * 125000,
          trades: Math.floor(Math.random() * 8),
        })
      }
      return map
    }

    for (let i = 1; i < equityHistory.length; i++) {
      const prev = equityHistory[i - 1]
      const curr = equityHistory[i]
      const date = new Date(curr.timestamp)
      if (getYear(date) !== year) continue

      const returnPct = prev.equity > 0 ? (curr.equity - prev.equity) / prev.equity : 0
      map.set(format(date, 'yyyy-MM-dd'), {
        date,
        returnPct,
        pnl: curr.totalPnl - prev.totalPnl,
        trades: 0,
      })
    }

    return map
  }, [equityHistory, year])

  // Build grid structure: weeks x days
  const { weeks, monthLabels } = useMemo(() => {
    const start = new Date(year, 0, 1)
    const end = new Date(year, 11, 31)
    const allDays = eachDayOfInterval({ start, end })

    // Group by week
    const weekMap = new Map<number, (DayData | null)[]>()
    for (const day of allDays) {
      const weekNum = getWeek(day)
      if (!weekMap.has(weekNum)) {
        weekMap.set(weekNum, new Array(7).fill(null))
      }
      const dayData = dailyDataMap.get(format(day, 'yyyy-MM-dd')) ?? null
      weekMap.get(weekNum)![getDay(day)] = dayData ? dayData : (getDay(day) !== 0 && getDay(day) !== 6 ? { date: day, returnPct: 0, pnl: 0, trades: 0 } : null)
    }

    // Sort by week number
    const sortedWeeks = Array.from(weekMap.entries()).sort(([a], [b]) => a - b)

    // Month label positions
    const monthLabelList: { month: string; weekIndex: number }[] = []
    let lastMonth = -1
    for (let i = 0; i < sortedWeeks.length; i++) {
      const [, days] = sortedWeeks[i]
      const firstDay = days.find((d) => d !== null)
      if (firstDay && firstDay.date.getMonth() !== lastMonth) {
        lastMonth = firstDay.date.getMonth()
        monthLabelList.push({ month: MONTHS[lastMonth], weekIndex: i })
      }
    }

    return { weeks: sortedWeeks.map(([, d]) => d), monthLabels: monthLabelList }
  }, [dailyDataMap, year])

  const maxAbsReturn = useMemo(() => {
    let max = 0
    for (const d of dailyDataMap.values()) {
      max = Math.max(max, Math.abs(d.returnPct))
    }
    return max
  }, [dailyDataMap])

  // Monthly summary
  const monthlySummary = useMemo(() => {
    return Array.from({ length: 12 }, (_, m) => {
      let totalReturn = 0
      for (const [key, d] of dailyDataMap.entries()) {
        if (new Date(key).getMonth() === m) totalReturn += d.returnPct
      }
      return { month: MONTHS[m], returnPct: totalReturn }
    })
  }, [dailyDataMap])

  return (
    <div className={`flex flex-col gap-3 p-3 bg-terminal-bg ${className}`}>
      {/* Year header */}
      <div className="flex items-center justify-between">
        <span className="text-terminal-text text-sm font-semibold font-mono">{year} Daily Returns</span>
        <div className="flex items-center gap-2 text-[10px] font-mono text-terminal-subtle">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-sm" style={{ background: getHeatmapColor(-0.02, 0.03) }} />
            <span>-2%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-sm bg-terminal-muted" />
            <span>0%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-sm" style={{ background: getHeatmapColor(0.02, 0.03) }} />
            <span>+2%</span>
          </div>
        </div>
      </div>

      {/* Calendar grid */}
      <div className="overflow-x-auto">
        <div className="relative" style={{ minWidth: weeks.length * 14 + 30 }}>
          {/* Month labels */}
          <div className="flex mb-1 ml-8">
            {monthLabels.map(({ month, weekIndex }) => (
              <div
                key={month}
                className="text-[10px] font-mono text-terminal-subtle absolute"
                style={{ left: 32 + weekIndex * 14 }}
              >
                {month}
              </div>
            ))}
          </div>

          <div className="flex gap-0 mt-4">
            {/* Day labels */}
            <div className="flex flex-col gap-0.5 mr-1">
              {DAYS_OF_WEEK.map((day, i) => (
                <div key={day} className="text-[9px] font-mono text-terminal-subtle h-3 flex items-center">
                  {i % 2 === 1 ? day.slice(0, 1) : ''}
                </div>
              ))}
            </div>

            {/* Weeks */}
            {weeks.map((week, wi) => (
              <div key={wi} className="flex flex-col gap-0.5 mr-0.5">
                {week.map((day, di) => {
                  if (!day) {
                    return <div key={di} className="w-3 h-3 rounded-sm bg-transparent" />
                  }

                  const color = getHeatmapColor(day.returnPct, maxAbsReturn)

                  return (
                    <div
                      key={di}
                      className="w-3 h-3 rounded-sm cursor-pointer transition-all hover:ring-1 hover:ring-terminal-text/50"
                      style={{ backgroundColor: color }}
                      onMouseEnter={(e) => {
                        setHoveredDay(day)
                        setTooltipPos({ x: e.clientX, y: e.clientY })
                      }}
                      onMouseLeave={() => setHoveredDay(null)}
                    />
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Monthly summary row */}
      <div className="flex items-center gap-1 mt-1">
        <span className="text-[10px] font-mono text-terminal-subtle w-8">Mo:</span>
        {monthlySummary.map(({ month, returnPct }) => (
          <div
            key={month}
            className="flex-1 text-center rounded text-[9px] font-mono py-0.5 cursor-default"
            style={{
              backgroundColor: getHeatmapColor(returnPct, maxAbsReturn * 5),
              color: Math.abs(returnPct) > 0.01 ? 'white' : '#9ca3af',
            }}
            title={`${month}: ${(returnPct * 100).toFixed(2)}%`}
          >
            {(returnPct * 100).toFixed(0)}%
          </div>
        ))}
      </div>

      {/* Tooltip */}
      {hoveredDay && (
        <div
          className="fixed z-50 bg-terminal-surface border border-terminal-border rounded p-2 text-xs font-mono shadow-lg pointer-events-none"
          style={{ left: tooltipPos.x + 12, top: tooltipPos.y - 60 }}
        >
          <div className="text-terminal-text font-semibold">{format(hoveredDay.date, 'MMM d, yyyy')}</div>
          <div className={`${hoveredDay.returnPct >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
            {hoveredDay.returnPct >= 0 ? '+' : ''}{(hoveredDay.returnPct * 100).toFixed(2)}%
          </div>
          <div className={`${hoveredDay.pnl >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
            ${Math.abs(hoveredDay.pnl).toFixed(0)} {hoveredDay.pnl >= 0 ? 'gain' : 'loss'}
          </div>
          {hoveredDay.trades > 0 && (
            <div className="text-terminal-subtle">{hoveredDay.trades} trades</div>
          )}
        </div>
      )}
    </div>
  )
}

export default HeatmapChart
