import React from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, Scatter, ScatterChart, ZAxis,
} from 'recharts'
import { CHART_COLORS } from '@/utils/colors'

export interface WalkForwardFold {
  foldId: number
  isStartDate: string
  isEndDate: string
  oosStartDate: string
  oosEndDate: string
  isSharpe: number
  oosSharpe: number
  selectedParams: Record<string, number>
  equityCurve: Array<{ date: string; equity: number }>
  isReturn: number
  oosReturn: number
}

interface ISvsOOSScatterProps {
  folds: WalkForwardFold[]
  height?: number
}

export function ISvsOOSScatter({ folds, height = 260 }: ISvsOOSScatterProps) {
  const data = folds.map(f => ({
    isSharpe: f.isSharpe,
    oosSharpe: f.oosSharpe,
    name: `Fold ${f.foldId + 1}`,
    foldId: f.foldId,
  }))

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
        <XAxis
          dataKey="isSharpe"
          name="IS Sharpe"
          type="number"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          label={{ value: 'IS Sharpe', position: 'insideBottom', offset: -10, fontSize: 10, fill: '#8899aa' }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          dataKey="oosSharpe"
          name="OOS Sharpe"
          type="number"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          label={{ value: 'OOS Sharpe', angle: -90, position: 'insideLeft', offset: 10, fontSize: 10, fill: '#8899aa' }}
          tickLine={false}
          axisLine={false}
        />
        <ZAxis range={[60, 60]} />
        <Tooltip
          cursor={{ strokeDasharray: '3 3' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const d = payload[0].payload as typeof data[0]
            return (
              <div className="bg-research-card border border-research-border rounded p-2 text-xs shadow-xl font-mono">
                <div className="text-research-text mb-1">{d.name}</div>
                <div className="text-research-subtle">IS: <span className="text-research-text">{d.isSharpe.toFixed(3)}</span></div>
                <div className="text-research-subtle">OOS: <span className="text-research-text">{d.oosSharpe.toFixed(3)}</span></div>
              </div>
            )
          }}
        />
        <ReferenceLine x={0} stroke="#2d3a4f" />
        <ReferenceLine y={0} stroke="#2d3a4f" />
        {/* 45-degree line */}
        <ReferenceLine
          segment={[{ x: -1, y: -1 }, { x: 3, y: 3 }]}
          stroke="#475569"
          strokeDasharray="4 4"
          label={{ value: 'IS=OOS', fill: '#475569', fontSize: 9 }}
        />
        <Scatter
          data={data}
          fill="#3b82f6"
          fillOpacity={0.8}
          name="Folds"
        />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

interface FoldEquityCurvesProps {
  folds: WalkForwardFold[]
  height?: number
}

export function FoldEquityCurves({ folds, height = 260 }: FoldEquityCurvesProps) {
  // Normalize each fold equity to start at 100
  const allDates = new Set<string>()
  const foldData = folds.map(fold => {
    const init = fold.equityCurve[0]?.equity ?? 1
    return fold.equityCurve.map(p => {
      allDates.add(p.date)
      return { date: p.date, equity: (p.equity / init) * 100 }
    })
  })

  const dates = [...allDates].sort()
  const merged = dates.map(date => {
    const row: Record<string, number | string> = { date }
    folds.forEach((fold, i) => {
      const pt = foldData[i].find(p => p.date === date)
      if (pt) row[`fold${fold.foldId}`] = pt.equity
    })
    return row
  })

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={merged} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={v => String(v).slice(5)}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#8899aa', fontFamily: 'JetBrains Mono, monospace' }}
          tickFormatter={v => `${v.toFixed(0)}`}
          tickLine={false}
          axisLine={false}
          width={40}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#111827',
            border: '1px solid #1e2a3a',
            borderRadius: '6px',
            fontSize: '11px',
            fontFamily: 'JetBrains Mono, monospace',
          }}
        />
        <ReferenceLine y={100} stroke="#2d3a4f" strokeDasharray="4 4" />
        {folds.map((fold, i) => (
          <Line
            key={fold.foldId}
            type="monotone"
            dataKey={`fold${fold.foldId}`}
            stroke={CHART_COLORS[i % CHART_COLORS.length]}
            strokeWidth={1.5}
            dot={false}
            name={`Fold ${fold.foldId + 1}`}
            connectNulls
          />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

// Mock data generator
export function generateMockWalkForwardFolds(n = 5): WalkForwardFold[] {
  let seed = 55555
  function rand() {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff
    return (seed >>> 0) / 0xffffffff
  }
  function randn() {
    return Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand())
  }

  return Array.from({ length: n }, (_, i) => {
    const isSharpe = 1.5 + randn() * 0.5
    const oosSharpe = isSharpe * 0.7 + randn() * 0.3
    const nPts = 60
    let equity = 100
    const equityCurve = Array.from({ length: nPts }, (_, d) => {
      equity *= 1 + randn() * 0.015 + 0.001
      return {
        date: new Date(Date.now() - (n - i) * 90 * 86_400_000 + d * 86_400_000).toISOString().slice(0, 10),
        equity,
      }
    })

    return {
      foldId: i,
      isStartDate: new Date(Date.now() - (n - i + 1) * 90 * 86_400_000).toISOString().slice(0, 10),
      isEndDate: new Date(Date.now() - (n - i) * 90 * 86_400_000).toISOString().slice(0, 10),
      oosStartDate: new Date(Date.now() - (n - i) * 90 * 86_400_000).toISOString().slice(0, 10),
      oosEndDate: new Date(Date.now() - (n - i - 1) * 90 * 86_400_000).toISOString().slice(0, 10),
      isSharpe,
      oosSharpe,
      selectedParams: {
        lookback: [5, 10, 20, 40, 60][Math.floor(rand() * 5)],
        threshold: parseFloat((0.1 + rand() * 0.9).toFixed(2)),
        rebalance: [1, 3, 5][Math.floor(rand() * 3)],
      },
      equityCurve,
      isReturn: randn() * 0.15 + 0.1,
      oosReturn: randn() * 0.1 + 0.05,
    }
  })
}
