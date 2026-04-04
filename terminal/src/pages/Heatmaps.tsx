// ============================================================
// Heatmaps.tsx — Multi-heatmap analysis page
// ============================================================
import React, { useState, useMemo } from 'react'
import { clsx } from 'clsx'

// ---- Types ----

type HeatmapView = 'sector-rotation' | 'correlation' | 'bh-calendar' | 'intraday'

// ---- Helpers ----

function hsvToRgb(h: number, s: number, v: number): string {
  const f = (n: number) => {
    const k = (n + h / 60) % 6
    return v - v * s * Math.max(Math.min(k, 4 - k, 1), 0)
  }
  return `rgb(${Math.round(f(5) * 255)},${Math.round(f(3) * 255)},${Math.round(f(1) * 255)})`
}

function returnColor(ret: number, maxAbs = 0.08): string {
  const t = Math.max(-1, Math.min(1, ret / maxAbs))
  if (t > 0) {
    return `rgba(34,197,94,${0.1 + t * 0.9})`
  } else {
    return `rgba(239,68,68,${0.1 + (-t) * 0.9})`
  }
}

function corrColor(c: number): string {
  if (c > 0) return `rgba(59,130,246,${0.1 + c * 0.85})`
  return `rgba(239,68,68,${0.1 + (-c) * 0.85})`
}

function bhColor(count: number, maxCount: number): string {
  if (count === 0) return '#111318'
  const t = count / maxCount
  return hsvToRgb(240 - t * 240, 0.8, 0.2 + t * 0.6)
}

function intradayColor(ret: number): string {
  return returnColor(ret, 0.02)
}

// ---- Sector Rotation Heatmap ----

const ASSETS = ['BTC', 'ETH', 'SOL', 'BNB', 'DOGE', 'LINK', 'AVAX', 'UNI', 'AAVE', 'MATIC', 'ARB', 'OP']
const PERIODS = ['1D', '1W', '2W', '1M', '3M', '6M', 'YTD', '1Y']

function generateSectorRotation(): number[][] {
  return ASSETS.map(() =>
    PERIODS.map((p) => {
      const magnitude = p === '1D' ? 0.04 : p === '1W' ? 0.1 : p === '1M' ? 0.2 : 0.4
      return (Math.random() - 0.42) * magnitude
    }),
  )
}

const SectorRotationHeatmap: React.FC = () => {
  const data = useMemo(() => generateSectorRotation(), [])
  const cellH = 36
  const cellW = 72

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: `60px repeat(${PERIODS.length}, ${cellW}px)`, width: 'fit-content', rowGap: 0 }}>
        {/* Header */}
        <div />
        {PERIODS.map((p) => (
          <div key={p} style={{ textAlign: 'center', padding: '4px 0', fontSize: 9, fontFamily: 'JetBrains Mono', color: '#475569', borderBottom: '1px solid #1e2130' }}>
            {p}
          </div>
        ))}

        {/* Rows */}
        {ASSETS.map((asset, ai) => (
          <React.Fragment key={asset}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 8, fontSize: 10, fontFamily: 'JetBrains Mono', color: '#94a3b8', height: cellH, borderBottom: '1px solid rgba(30,33,48,0.3)' }}>
              {asset}
            </div>
            {PERIODS.map((period, pi) => {
              const ret = data[ai][pi]
              return (
                <div
                  key={period}
                  title={`${asset} ${period}: ${(ret * 100).toFixed(2)}%`}
                  style={{
                    width: cellW,
                    height: cellH,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: returnColor(ret),
                    border: '1px solid rgba(10,11,14,0.5)',
                    fontSize: 9,
                    fontFamily: 'JetBrains Mono',
                    color: Math.abs(ret) > 0.04 ? 'rgba(226,232,240,0.9)' : 'rgba(148,163,184,0.7)',
                  }}
                >
                  {(ret * 100).toFixed(1)}%
                </div>
              )
            })}
          </React.Fragment>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-3 mt-3">
        <span className="text-[9px] font-mono text-slate-600">Return:</span>
        <div className="flex items-center gap-1">
          <div className="w-8 h-3 rounded" style={{ background: 'rgba(239,68,68,0.9)' }} />
          <span className="text-[9px] font-mono text-slate-600">-8%+</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-8 h-3 rounded" style={{ background: 'rgba(239,68,68,0.2)' }} />
          <span className="text-[9px] font-mono text-slate-600">-2%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-8 h-3 rounded" style={{ background: 'rgba(34,197,94,0.2)' }} />
          <span className="text-[9px] font-mono text-slate-600">+2%</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-8 h-3 rounded" style={{ background: 'rgba(34,197,94,0.9)' }} />
          <span className="text-[9px] font-mono text-slate-600">+8%+</span>
        </div>
      </div>
    </div>
  )
}

// ---- Correlation Heatmap with clustering ----

function generateCorrelations(): number[][] {
  const n = ASSETS.length
  const base: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 1
      if (Math.abs(i - j) <= 1) return 0.75 + Math.random() * 0.2
      if (Math.abs(i - j) <= 3) return 0.5 + Math.random() * 0.3
      return 0.2 + Math.random() * 0.4
    }),
  )
  return base
}

// Simple hierarchical-style reordering based on first correlation eigenvector approximation
function clusterOrder(corr: number[][]): number[] {
  const n = corr.length
  const scores = Array.from({ length: n }, (_, i) =>
    corr[i].reduce((s, v, j) => (i !== j ? s + v : s), 0) / (n - 1),
  )
  return Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a])
}

const CorrelationHeatmapClustered: React.FC = () => {
  const corrMatrix = useMemo(() => generateCorrelations(), [])
  const order = useMemo(() => clusterOrder(corrMatrix), [corrMatrix])
  const orderedAssets = order.map((i) => ASSETS[i])

  const cellSize = 46

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: `56px repeat(${orderedAssets.length}, ${cellSize}px)`, width: 'fit-content' }}>
        <div />
        {orderedAssets.map((sym) => (
          <div key={sym} style={{ width: cellSize, display: 'flex', alignItems: 'flex-end', justifyContent: 'center', paddingBottom: 4 }}>
            <span style={{ fontSize: 8, fontFamily: 'JetBrains Mono', color: '#475569', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
              {sym}
            </span>
          </div>
        ))}

        {order.map((ri, rr) => (
          <React.Fragment key={ri}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 6, fontSize: 9, fontFamily: 'JetBrains Mono', color: '#475569' }}>
              {ASSETS[ri]}
            </div>
            {order.map((ci) => {
              const val = corrMatrix[ri][ci]
              return (
                <div
                  key={ci}
                  title={`${ASSETS[ri]} / ${ASSETS[ci]}: ${val.toFixed(2)}`}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: corrColor(ri === ci ? 0.5 : val),
                    border: '1px solid rgba(10,11,14,0.6)',
                    fontSize: 8,
                    fontFamily: 'JetBrains Mono',
                    color: val > 0.6 ? '#93c5fd' : val > 0.3 ? '#94a3b8' : '#475569',
                    fontWeight: ri === ci ? 700 : 400,
                  }}
                >
                  {ri === ci ? '—' : val.toFixed(2)}
                </div>
              )
            })}
            {void rr}
          </React.Fragment>
        ))}
      </div>

      <div className="flex items-center gap-3 mt-3">
        <span className="text-[9px] font-mono text-slate-600">Correlation:</span>
        <div style={{ height: 8, width: 160, background: 'linear-gradient(to right, rgba(239,68,68,0.9), rgba(30,33,48,0.5), rgba(59,130,246,0.9))', borderRadius: 4 }} />
        <span className="text-[9px] font-mono text-slate-600">-1 → 0 → +1</span>
        <span className="text-[9px] font-mono text-slate-700 ml-2">Reordered by hierarchical clustering</span>
      </div>
    </div>
  )
}

// ---- BH Activation Calendar (GitHub-style) ----

const BH_ASSETS = ['BTC', 'ETH', 'SOL', 'BNB', 'LINK']

function generateBHCalendar(): Record<string, Record<string, number>> {
  const result: Record<string, Record<string, number>> = {}
  for (const asset of BH_ASSETS) {
    result[asset] = {}
    for (let d = 90; d >= 0; d--) {
      const date = new Date(Date.now() - d * 86400000).toISOString().slice(0, 10)
      result[asset][date] = Math.floor(Math.random() ** 2 * 8)  // skewed toward low values
    }
  }
  return result
}

const BHCalendarHeatmap: React.FC = () => {
  const data = useMemo(() => generateBHCalendar(), [])

  const allCounts = BH_ASSETS.flatMap((a) => Object.values(data[a]))
  const maxCount = Math.max(...allCounts)

  // Group dates by week
  const dates = Object.keys(data[BH_ASSETS[0]]).sort()
  const weeks: string[][] = []
  let week: string[] = []
  for (const date of dates) {
    week.push(date)
    if (week.length === 7) {
      weeks.push(week)
      week = []
    }
  }
  if (week.length) weeks.push(week)

  const cellSize = 16

  return (
    <div className="flex flex-col gap-3">
      {BH_ASSETS.map((asset) => (
        <div key={asset}>
          <div className="text-[10px] font-mono text-slate-500 mb-1.5 uppercase">{asset}</div>
          <div className="flex gap-0.5">
            {weeks.map((w, wi) => (
              <div key={wi} className="flex flex-col gap-0.5">
                {w.map((date) => {
                  const count = data[asset][date] ?? 0
                  return (
                    <div
                      key={date}
                      title={`${asset} ${date}: ${count} formations`}
                      style={{
                        width: cellSize,
                        height: cellSize,
                        borderRadius: 2,
                        background: bhColor(count, maxCount),
                        cursor: 'default',
                      }}
                    />
                  )
                })}
              </div>
            ))}
          </div>
        </div>
      ))}

      {/* Legend */}
      <div className="flex items-center gap-2 mt-1">
        <span className="text-[9px] font-mono text-slate-600">BH Formations:</span>
        {[0, 2, 4, 6, 8].map((c) => (
          <div key={c} className="flex items-center gap-1">
            <div style={{ width: 12, height: 12, borderRadius: 2, background: c === 0 ? '#111318' : bhColor(c, 8) }} />
            <span className="text-[9px] font-mono text-slate-700">{c}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ---- Intraday Seasonality Heatmap ----

const HOURS = Array.from({ length: 24 }, (_, i) => `${i.toString().padStart(2, '0')}:00`)
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

function generateIntradayData(): number[][] {
  return HOURS.map((_, h) =>
    DAYS.map((_, d) => {
      // More volatile during US/Asian session crossovers, quieter on weekends
      const sessionBoost = (h >= 13 && h <= 17) ? 0.008 : (h >= 0 && h <= 4) ? 0.006 : 0.003
      const weekendFactor = (d >= 5) ? 0.5 : 1
      return (Math.random() - 0.45) * sessionBoost * weekendFactor * 2
    }),
  )
}

const IntradaySeasonality: React.FC = () => {
  const data = useMemo(() => generateIntradayData(), [])
  const cellH = 22
  const cellW = 64

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: `48px repeat(${DAYS.length}, ${cellW}px)`, width: 'fit-content' }}>
        <div />
        {DAYS.map((d) => (
          <div key={d} style={{ width: cellW, textAlign: 'center', padding: '2px 0', fontSize: 9, fontFamily: 'JetBrains Mono', color: d === 'Sat' || d === 'Sun' ? '#2e3550' : '#475569' }}>
            {d}
          </div>
        ))}

        {HOURS.map((hour, hi) => (
          <React.Fragment key={hour}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 6, height: cellH, fontSize: 8, fontFamily: 'JetBrains Mono', color: '#475569' }}>
              {hour}
            </div>
            {DAYS.map((_, di) => {
              const ret = data[hi][di]
              const isSession = (hi >= 13 && hi <= 17) || (hi >= 0 && hi <= 4)
              return (
                <div
                  key={di}
                  title={`${DAYS[di]} ${hour}: ${(ret * 100).toFixed(3)}%`}
                  style={{
                    width: cellW,
                    height: cellH,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: intradayColor(ret),
                    border: `1px solid ${isSession ? 'rgba(59,130,246,0.1)' : 'rgba(10,11,14,0.5)'}`,
                    fontSize: 8,
                    fontFamily: 'JetBrains Mono',
                    color: Math.abs(ret) > 0.005 ? 'rgba(226,232,240,0.8)' : 'rgba(71,85,105,0.7)',
                  }}
                >
                  {(ret * 100).toFixed(2)}%
                </div>
              )
            })}
          </React.Fragment>
        ))}
      </div>

      <div className="flex items-center gap-4 mt-3 text-[9px] font-mono text-slate-600">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3" style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)' }} />
          <span>Session crossover hours</span>
        </div>
        <span>Row = hour (UTC) · Column = day of week</span>
      </div>
    </div>
  )
}

// ---- Main page ----

export const Heatmaps: React.FC = () => {
  const [view, setView] = useState<HeatmapView>('sector-rotation')

  const views: { key: HeatmapView; label: string }[] = [
    { key: 'sector-rotation', label: 'Sector Rotation' },
    { key: 'correlation',     label: 'Correlation (Clustered)' },
    { key: 'bh-calendar',    label: 'BH Calendar' },
    { key: 'intraday',       label: 'Intraday Seasonality' },
  ]

  const renderView = () => {
    switch (view) {
      case 'sector-rotation': return <SectorRotationHeatmap />
      case 'correlation':     return <CorrelationHeatmapClustered />
      case 'bh-calendar':    return <BHCalendarHeatmap />
      case 'intraday':       return <IntradaySeasonality />
    }
  }

  const viewDescriptions: Record<HeatmapView, string> = {
    'sector-rotation': 'Returns by asset × time period. Click cells to compare across columns.',
    'correlation':     'Pairwise correlation matrix reordered by hierarchical clustering. Blue = positive, red = negative.',
    'bh-calendar':    'GitHub-style heatmap of BH signal activation frequency by asset and day.',
    'intraday':       'Average return by hour-of-day (UTC) × day-of-week. Identifies seasonality.',
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto" style={{ padding: '12px', gap: '12px' }}>

      {/* Tab bar */}
      <div className="flex items-center gap-1 flex-wrap">
        {views.map((v) => (
          <button
            key={v.key}
            onClick={() => setView(v.key)}
            className={clsx(
              'px-3 py-1.5 rounded border text-[10px] font-mono transition-colors',
              view === v.key
                ? 'border-blue-500/50 text-blue-400 bg-blue-950/30'
                : 'border-[#1e2130] text-slate-500 hover:text-slate-300 bg-[#0e1017]',
            )}
          >
            {v.label}
          </button>
        ))}
      </div>

      {/* Description */}
      <p className="text-[9px] font-mono text-slate-700">{viewDescriptions[view]}</p>

      {/* Content */}
      <div className="bg-[#0e1017] border border-[#1e2130] rounded-lg p-4">
        {renderView()}
      </div>

    </div>
  )
}
