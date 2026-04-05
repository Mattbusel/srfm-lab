import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from '../components/LoadingSpinner'
import MetricCard from '../components/MetricCard'

// ─── Types ────────────────────────────────────────────────────────────────────

interface SlippageCell {
  hour: number
  dow: number       // 0=Mon..6=Sun
  slippageBps: number
}

interface Fill {
  ts: number
  symbol: string
  price: number
  vwap: number
  side: 'BUY' | 'SELL'
  slippageBps: number
  qty: number
}

interface CircuitBreaker {
  name: string
  state: 'CLOSED' | 'OPEN' | 'HALF_OPEN'
  trips: number
  lastTrippedAt?: string
  description: string
}

interface DailyReport {
  date: string
  trades: number
  fillRate: number
  avgSlippageBps: number
  totalCommissions: number
}

interface ExecutionData {
  slippageHeatmap: SlippageCell[]
  fills: Fill[]
  circuitBreakers: CircuitBreaker[]
  dailyReports: DailyReport[]
  avgSlippageBps: number
  vwapBeatRate: number
  totalCommissionsToday: number
  fillRateToday: number
}

// ─── Mock ─────────────────────────────────────────────────────────────────────

function buildHeatmap(): SlippageCell[] {
  const cells: SlippageCell[] = []
  for (let dow = 0; dow < 5; dow++) {
    for (let hour = 0; hour < 24; hour++) {
      // High slippage at open/close, low during liquid hours
      const base = 2.0
      const timeNoise = Math.abs(Math.sin(hour * 0.5 + dow * 0.8)) * 3
      const openBoost = (hour === 9 || hour === 10) ? 3 : 0
      const closeBoost = (hour === 15 || hour === 16) ? 2 : 0
      const asianDiscount = hour < 7 ? 1.5 : 0
      cells.push({ hour, dow, slippageBps: Math.max(0.2, base + timeNoise + openBoost + closeBoost + asianDiscount) })
    }
  }
  return cells
}

function buildFills(): Fill[] {
  const symbols = ['BTC', 'ETH', 'SOL', 'BNB']
  return Array.from({ length: 40 }, (_, i) => {
    const price = 1000 + Math.random() * 500
    const slip = (Math.random() - 0.5) * 0.002
    return {
      ts: Date.now() - i * 900_000,
      symbol: symbols[i % symbols.length],
      price,
      vwap: price * (1 + slip),
      side: Math.random() > 0.5 ? 'BUY' : 'SELL',
      slippageBps: Math.abs(slip * 10_000),
      qty: Math.round(Math.random() * 10 + 0.5),
    }
  })
}

const MOCK: ExecutionData = {
  avgSlippageBps: 1.8,
  vwapBeatRate: 0.61,
  totalCommissionsToday: 142.50,
  fillRateToday: 0.993,
  slippageHeatmap: buildHeatmap(),
  fills: buildFills(),
  circuitBreakers: [
    { name: 'Daily Loss Limit',     state: 'CLOSED',    trips: 2,  lastTrippedAt: new Date(Date.now() - 86_400_000 * 12).toISOString(), description: 'Max daily loss exceeded' },
    { name: 'Drawdown Breaker',     state: 'CLOSED',    trips: 1,  description: 'Portfolio drawdown > 6%' },
    { name: 'Slippage Spike',       state: 'HALF_OPEN', trips: 4,  lastTrippedAt: new Date(Date.now() - 3_600_000).toISOString(),      description: 'Avg slippage > 5 bps in window' },
    { name: 'Fill Rate Breaker',    state: 'CLOSED',    trips: 0,  description: 'Fill rate < 95%' },
    { name: 'Concentration Limit',  state: 'CLOSED',    trips: 0,  description: 'Single position > 20% of portfolio' },
    { name: 'API Error Rate',       state: 'OPEN',      trips: 7,  lastTrippedAt: new Date(Date.now() - 600_000).toISOString(),        description: 'Exchange API errors > 5/min' },
  ],
  dailyReports: Array.from({ length: 14 }, (_, i) => ({
    date: new Date(Date.now() - i * 86_400_000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    trades: Math.floor(Math.random() * 30) + 10,
    fillRate: 0.97 + Math.random() * 0.03,
    avgSlippageBps: 1.0 + Math.random() * 2,
    totalCommissions: 80 + Math.random() * 200,
  })),
}

async function fetchExecutionData(): Promise<ExecutionData> {
  try {
    const res = await fetch('/api/execution/analytics')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK
  }
}

// ─── Slippage Heatmap ─────────────────────────────────────────────────────────

const DOW_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
const HOUR_LABELS = ['0', '3', '6', '9', '12', '15', '18', '21']

interface SlippageHeatmapProps {
  data: SlippageCell[]
}

const SlippageHeatmap: React.FC<SlippageHeatmapProps> = ({ data }) => {
  const [hoveredCell, setHoveredCell] = useState<SlippageCell | null>(null)
  const maxSlip = Math.max(...data.map(c => c.slippageBps), 1)

  const cellW = 100 / 24
  const cellH = 100 / 5

  function slipColor(bps: number): string {
    const ratio = bps / maxSlip
    if (ratio < 0.2) return '#22c55e'
    if (ratio < 0.4) return '#86efac'
    if (ratio < 0.6) return '#fde047'
    if (ratio < 0.8) return '#f97316'
    return '#ef4444'
  }

  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox="0 0 100 55" style={{ width: '100%', height: 180 }} preserveAspectRatio="none">
        {data.map((cell) => (
          <rect
            key={`${cell.dow}-${cell.hour}`}
            x={cell.hour * cellW + 0.1}
            y={cell.dow * cellH + 0.1}
            width={cellW - 0.2}
            height={cellH - 0.2}
            fill={slipColor(cell.slippageBps)}
            opacity={0.85}
            onMouseEnter={() => setHoveredCell(cell)}
            onMouseLeave={() => setHoveredCell(null)}
            style={{ cursor: 'pointer' }}
          />
        ))}
        {/* Hour labels */}
        {HOUR_LABELS.map((lbl, i) => (
          <text key={lbl} x={i * (100 / 8) + 1} y={53} fontSize={2.8} fill="var(--text-muted)">
            {lbl}h
          </text>
        ))}
        {/* Dow labels */}
        {DOW_LABELS.map((lbl, i) => (
          <text key={lbl} x={0} y={(i + 0.65) * cellH} fontSize={3} fill="var(--text-muted)">
            {lbl}
          </text>
        ))}
      </svg>
      {hoveredCell && (
        <div style={{
          position: 'absolute', bottom: 0, right: 0,
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 6, padding: '6px 10px', fontSize: '0.72rem', color: 'var(--text-primary)',
          pointerEvents: 'none',
        }}>
          {DOW_LABELS[hoveredCell.dow]} {hoveredCell.hour}:00 UTC —{' '}
          <strong style={{ color: 'var(--yellow)' }}>{hoveredCell.slippageBps.toFixed(2)} bps</strong>
        </div>
      )}
    </div>
  )
}

// ─── VWAP Fill Chart ──────────────────────────────────────────────────────────

interface VwapChartProps {
  fills: Fill[]
}

const VwapChart: React.FC<VwapChartProps> = ({ fills }) => {
  const recent = fills.slice(0, 20)
  if (recent.length < 2) return null

  const prices = [...recent.map(f => f.price), ...recent.map(f => f.vwap)]
  const minP = Math.min(...prices)
  const maxP = Math.max(...prices)
  const rangeP = maxP - minP || 1
  const w = 100
  const h = 80
  const pad = { l: 8, r: 4, t: 6, b: 16 }
  const iw = w - pad.l - pad.r
  const ih = h - pad.t - pad.b

  const toX = (i: number) => pad.l + (i / (recent.length - 1)) * iw
  const toY = (p: number) => pad.t + (1 - (p - minP) / rangeP) * ih

  const vwapPath = recent.map((f, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(f.vwap).toFixed(1)}`).join(' ')

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', height: 140 }} preserveAspectRatio="none">
      {/* VWAP line */}
      <path d={vwapPath} fill="none" stroke="var(--yellow)" strokeWidth={0.8} strokeDasharray="2,1" />
      {/* Fill dots */}
      {recent.map((f, i) => {
        const beatVwap = f.side === 'BUY' ? f.price <= f.vwap : f.price >= f.vwap
        return (
          <circle
            key={i}
            cx={toX(i)}
            cy={toY(f.price)}
            r={1.5}
            fill={beatVwap ? 'var(--green)' : 'var(--red)'}
            opacity={0.9}
          />
        )
      })}
      {/* Legend */}
      <circle cx={4} cy={4} r={1.5} fill="var(--green)" />
      <text x={7} y={5.5} fontSize={2.8} fill="var(--text-muted)">Beat VWAP</text>
      <circle cx={28} cy={4} r={1.5} fill="var(--red)" />
      <text x={31} y={5.5} fontSize={2.8} fill="var(--text-muted)">Missed</text>
      <line x1={50} y1={3} x2={56} y2={3} stroke="var(--yellow)" strokeWidth={0.8} strokeDasharray="1.5,0.8" />
      <text x={57} y={4.5} fontSize={2.8} fill="var(--text-muted)">VWAP</text>
    </svg>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const ExecutionAnalytics: React.FC = () => {
  const { data, isLoading } = useQuery<ExecutionData>({
    queryKey: ['execution-analytics'],
    queryFn: fetchExecutionData,
    refetchInterval: 30_000,
  })

  if (isLoading || !data) return <LoadingSpinner message="Loading execution analytics…" />

  const cbColors: Record<string, string> = {
    CLOSED:    'var(--green)',
    OPEN:      'var(--red)',
    HALF_OPEN: 'var(--yellow)',
  }

  const openBreakers = data.circuitBreakers.filter(cb => cb.state !== 'CLOSED')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        <MetricCard label="Avg Slippage" value={`${data.avgSlippageBps.toFixed(2)} bps`} color={data.avgSlippageBps < 2 ? 'var(--green)' : 'var(--yellow)'} />
        <MetricCard label="VWAP Beat Rate" value={`${(data.vwapBeatRate * 100).toFixed(1)}%`} color={data.vwapBeatRate > 0.55 ? 'var(--green)' : 'var(--yellow)'} />
        <MetricCard label="Fill Rate Today" value={`${(data.fillRateToday * 100).toFixed(1)}%`} color={data.fillRateToday > 0.98 ? 'var(--green)' : 'var(--yellow)'} />
        <MetricCard label="Commissions Today" value={`$${data.totalCommissionsToday.toFixed(2)}`} color="var(--text-muted)" />
      </div>

      {/* Circuit Breaker Alert */}
      {openBreakers.length > 0 && (
        <div style={{
          padding: '12px 16px', borderRadius: 8,
          background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)',
          fontSize: '0.8rem', color: 'var(--red)',
        }}>
          <strong>Circuit breakers active:</strong>{' '}
          {openBreakers.map(cb => cb.name).join(', ')}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 16 }}>
        {/* Slippage Heatmap */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '16px',
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10 }}>
            SLIPPAGE HEATMAP (bps by hour UTC × weekday)
          </div>
          <SlippageHeatmap data={data.slippageHeatmap} />
          <div style={{ display: 'flex', gap: 12, marginTop: 8, fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            {[['Low', '#22c55e'], ['Med', '#fde047'], ['High', '#f97316'], ['Peak', '#ef4444']].map(([lbl, col]) => (
              <div key={lbl} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: col as string }} />
                {lbl}
              </div>
            ))}
          </div>
        </div>

        {/* Circuit Breakers */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, overflow: 'hidden',
        }}>
          <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            CIRCUIT BREAKERS
          </div>
          {data.circuitBreakers.map((cb, i) => {
            const color = cbColors[cb.state]
            return (
              <div key={i} style={{
                padding: '10px 16px',
                borderBottom: i < data.circuitBreakers.length - 1 ? '1px solid var(--border)' : undefined,
                display: 'flex', alignItems: 'flex-start', gap: 10,
              }}>
                <span style={{
                  width: 10, height: 10, borderRadius: '50%',
                  background: color, flexShrink: 0, marginTop: 3,
                  boxShadow: cb.state !== 'CLOSED' ? `0 0 6px ${color}` : undefined,
                }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>{cb.name}</div>
                  <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)', marginTop: 1 }}>{cb.description}</div>
                  {cb.lastTrippedAt && (
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 1 }}>
                      Last: {new Date(cb.lastTrippedAt).toLocaleString()}
                    </div>
                  )}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 }}>
                  <span style={{
                    fontSize: '0.65rem', fontWeight: 700, padding: '1px 6px', borderRadius: 4,
                    background: `${color}18`, color,
                  }}>
                    {cb.state}
                  </span>
                  <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>{cb.trips} trips</span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* VWAP Chart */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '16px',
      }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10 }}>
          FILL PRICES vs VWAP BENCHMARK (last 20 fills)
        </div>
        <VwapChart fills={data.fills} />
      </div>

      {/* Daily Execution Report */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, overflow: 'hidden',
      }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
          DAILY EXECUTION REPORT
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Date', 'Trades', 'Fill Rate', 'Avg Slippage (bps)', 'Commissions'].map(h => (
                <th key={h} style={{ padding: '6px 14px', textAlign: 'left', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.dailyReports.map((r, i) => (
              <tr key={i} style={{ borderBottom: i < data.dailyReports.length - 1 ? '1px solid var(--border)' : undefined }}>
                <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--text-primary)', fontWeight: i === 0 ? 700 : 400 }}>
                  {r.date}{i === 0 && <span style={{ marginLeft: 6, fontSize: '0.62rem', color: 'var(--accent)' }}>TODAY</span>}
                </td>
                <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>{r.trades}</td>
                <td style={{ padding: '7px 14px', fontSize: '0.78rem', fontVariantNumeric: 'tabular-nums', color: r.fillRate > 0.98 ? 'var(--green)' : 'var(--yellow)' }}>
                  {(r.fillRate * 100).toFixed(1)}%
                </td>
                <td style={{ padding: '7px 14px', fontSize: '0.78rem', fontVariantNumeric: 'tabular-nums', color: r.avgSlippageBps < 2 ? 'var(--green)' : 'var(--yellow)' }}>
                  {r.avgSlippageBps.toFixed(2)}
                </td>
                <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                  ${r.totalCommissions.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default ExecutionAnalytics
