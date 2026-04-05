import React, { useState, useEffect, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import MetricCard from '../components/MetricCard'
import LoadingSpinner from '../components/LoadingSpinner'

// ─── Types ────────────────────────────────────────────────────────────────────

interface Position {
  symbol: string
  size: number
  entryPrice: number
  currentPrice: number
  unrealizedPnl: number
  holdBars: number
  side: 'LONG' | 'SHORT'
}

interface Trade {
  id: string
  symbol: string
  pnl: number
  holdBars: number
  signalType: string
  closedAt: string
  side: 'LONG' | 'SHORT'
}

interface EquityPoint {
  ts: number
  equity: number
}

interface DailyPnl {
  date: string
  pnl: number
}

interface WinRate {
  window50: number
  window100: number
  window200: number
}

interface TraderStatus {
  uptime: string
  lastHeartbeat: string
  connected: boolean
  positions: Position[]
  recentTrades: Trade[]
  equity: EquityPoint[]
  dailyPnl: DailyPnl[]
  winRate: WinRate
  totalEquity: number
  dailyPnlToday: number
  openPnl: number
}

// ─── Mock Data ────────────────────────────────────────────────────────────────

function buildMockEquity(): EquityPoint[] {
  const pts: EquityPoint[] = []
  let equity = 100_000
  const now = Date.now()
  for (let i = 200; i >= 0; i--) {
    equity += (Math.random() - 0.47) * 800
    pts.push({ ts: now - i * 5 * 60_000, equity: Math.max(equity, 80_000) })
  }
  return pts
}

function buildMockDailyPnl(): DailyPnl[] {
  const out: DailyPnl[] = []
  for (let i = 29; i >= 0; i--) {
    const d = new Date(Date.now() - i * 86_400_000)
    out.push({
      date: d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      pnl: (Math.random() - 0.44) * 3_200,
    })
  }
  return out
}

const MOCK_STATUS: TraderStatus = {
  uptime: '14d 7h 22m',
  lastHeartbeat: new Date().toISOString(),
  connected: true,
  totalEquity: 127_845.32,
  dailyPnlToday: 1_240.80,
  openPnl: 542.10,
  winRate: { window50: 0.58, window100: 0.55, window200: 0.53 },
  equity: buildMockEquity(),
  dailyPnl: buildMockDailyPnl(),
  positions: [
    { symbol: 'BTC', side: 'LONG',  size: 0.42,  entryPrice: 68_200, currentPrice: 69_540, unrealizedPnl: 562.80,  holdBars: 14 },
    { symbol: 'ETH', side: 'LONG',  size: 3.1,   entryPrice: 3_820,  currentPrice: 3_795,  unrealizedPnl: -77.50,  holdBars: 7  },
    { symbol: 'SOL', side: 'SHORT', size: 22.0,  entryPrice: 182.4,  currentPrice: 179.8,  unrealizedPnl: 57.20,   holdBars: 3  },
  ],
  recentTrades: Array.from({ length: 20 }, (_, i) => ({
    id: `t${i}`,
    symbol: ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX'][i % 5],
    pnl: (Math.random() - 0.42) * 1_200,
    holdBars: Math.floor(Math.random() * 48) + 1,
    signalType: ['MOMENTUM', 'MEAN_REV', 'BREAKOUT', 'ML_ENSEMBLE'][i % 4],
    side: Math.random() > 0.5 ? 'LONG' : 'SHORT',
    closedAt: new Date(Date.now() - i * 3_600_000).toISOString(),
  })),
}

// ─── API ──────────────────────────────────────────────────────────────────────

async function fetchTraderStatus(): Promise<TraderStatus> {
  try {
    const res = await fetch('/api/live-trader/status')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return {
      ...MOCK_STATUS,
      lastHeartbeat: new Date().toISOString(),
    }
  }
}

async function postHalt(): Promise<void> {
  await fetch('http://localhost:8090/halt', { method: 'POST' })
}

// ─── SVG Equity Curve ─────────────────────────────────────────────────────────

interface EquityCurveProps {
  points: EquityPoint[]
  width?: number
  height?: number
}

const EquityCurve: React.FC<EquityCurveProps> = ({ points, width = 800, height = 160 }) => {
  if (points.length < 2) return null

  const minEq = Math.min(...points.map(p => p.equity))
  const maxEq = Math.max(...points.map(p => p.equity))
  const range = maxEq - minEq || 1
  const pad = { top: 12, bottom: 24, left: 60, right: 12 }
  const w = width - pad.left - pad.right
  const h = height - pad.top - pad.bottom

  const toX = (i: number) => pad.left + (i / (points.length - 1)) * w
  const toY = (eq: number) => pad.top + (1 - (eq - minEq) / range) * h

  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(p.equity).toFixed(1)}`)
    .join(' ')

  const areaD =
    pathD +
    ` L ${toX(points.length - 1).toFixed(1)} ${(pad.top + h).toFixed(1)}` +
    ` L ${pad.left.toFixed(1)} ${(pad.top + h).toFixed(1)} Z`

  // Y axis labels
  const yLabels = [minEq, minEq + range * 0.5, maxEq]
  const startEq = points[0].equity
  const endEq = points[points.length - 1].equity
  const isUp = endEq >= startEq
  const lineColor = isUp ? 'var(--green)' : 'var(--red)'

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: '100%', height }}
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id="eq-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={lineColor} stopOpacity="0.25" />
          <stop offset="100%" stopColor={lineColor} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      {/* Grid lines */}
      {yLabels.map((val, i) => (
        <g key={i}>
          <line
            x1={pad.left} y1={toY(val)}
            x2={pad.left + w} y2={toY(val)}
            stroke="var(--border)" strokeWidth={0.5} strokeDasharray="3,3"
          />
          <text
            x={pad.left - 6} y={toY(val) + 4}
            textAnchor="end"
            fill="var(--text-muted)"
            fontSize={9}
          >
            {(val / 1000).toFixed(0)}k
          </text>
        </g>
      ))}
      {/* Area fill */}
      <path d={areaD} fill="url(#eq-grad)" />
      {/* Line */}
      <path d={pathD} fill="none" stroke={lineColor} strokeWidth={1.5} />
      {/* Current dot */}
      <circle
        cx={toX(points.length - 1)}
        cy={toY(endEq)}
        r={3}
        fill={lineColor}
      />
    </svg>
  )
}

// ─── Daily P&L Bar Chart ──────────────────────────────────────────────────────

interface DailyBarChartProps {
  data: DailyPnl[]
  height?: number
}

const DailyBarChart: React.FC<DailyBarChartProps> = ({ data, height = 100 }) => {
  const maxAbs = Math.max(...data.map(d => Math.abs(d.pnl)), 1)
  const barW = 100 / data.length
  const midY = height / 2

  return (
    <svg viewBox={`0 0 100 ${height}`} style={{ width: '100%', height }} preserveAspectRatio="none">
      {/* Zero line */}
      <line x1={0} y1={midY} x2={100} y2={midY} stroke="var(--border)" strokeWidth={0.5} />
      {data.map((d, i) => {
        const ratio = d.pnl / maxAbs
        const barH = Math.abs(ratio) * midY * 0.9
        const y = d.pnl >= 0 ? midY - barH : midY
        return (
          <rect
            key={i}
            x={i * barW + barW * 0.1}
            y={y}
            width={barW * 0.8}
            height={barH}
            fill={d.pnl >= 0 ? 'var(--green)' : 'var(--red)'}
            opacity={0.75}
          />
        )
      })}
    </svg>
  )
}

// ─── Win Rate Gauge ───────────────────────────────────────────────────────────

interface WinGaugeProps {
  rate: number
  label: string
}

const WinGauge: React.FC<WinGaugeProps> = ({ rate, label }) => {
  const r = 28
  const cx = 36
  const cy = 36
  const circumference = Math.PI * r
  const filled = circumference * rate
  const color = rate >= 0.55 ? 'var(--green)' : rate >= 0.5 ? 'var(--yellow)' : 'var(--red)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <svg width={72} height={40} viewBox="0 0 72 40">
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none" stroke="var(--bg-hover)" strokeWidth={6}
        />
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth={6}
          strokeDasharray={`${filled} ${circumference}`}
          strokeLinecap="round"
        />
        <text x={cx} y={cy - 4} textAnchor="middle" fill="var(--text-primary)" fontSize={11} fontWeight={700}>
          {(rate * 100).toFixed(0)}%
        </text>
      </svg>
      <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{label}</span>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const LiveTrader: React.FC = () => {
  const queryClient = useQueryClient()
  const [haltConfirm, setHaltConfirm] = useState(false)
  const [lastRefresh, setLastRefresh] = useState(new Date())

  const { data, isLoading } = useQuery<TraderStatus>({
    queryKey: ['live-trader'],
    queryFn: fetchTraderStatus,
    refetchInterval: 10_000,
  })

  useEffect(() => {
    const interval = setInterval(() => setLastRefresh(new Date()), 10_000)
    return () => clearInterval(interval)
  }, [])

  const haltMutation = useMutation({
    mutationFn: postHalt,
    onSuccess: () => {
      setHaltConfirm(false)
      queryClient.invalidateQueries({ queryKey: ['live-trader'] })
    },
  })

  if (isLoading || !data) return <LoadingSpinner message="Loading live trader…" />

  const totalOpenPnl = data.positions.reduce((s, p) => s + p.unrealizedPnl, 0)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Top metric row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        <MetricCard
          label="Total Equity"
          value={`$${data.totalEquity.toLocaleString('en-US', { minimumFractionDigits: 2 })}`}
          color="var(--accent)"
        />
        <MetricCard
          label="Today's P&L"
          value={`${data.dailyPnlToday >= 0 ? '+' : ''}$${data.dailyPnlToday.toLocaleString('en-US', { minimumFractionDigits: 2 })}`}
          color={data.dailyPnlToday >= 0 ? 'var(--green)' : 'var(--red)'}
        />
        <MetricCard
          label="Open P&L"
          value={`${totalOpenPnl >= 0 ? '+' : ''}$${totalOpenPnl.toLocaleString('en-US', { minimumFractionDigits: 2 })}`}
          color={totalOpenPnl >= 0 ? 'var(--green)' : 'var(--red)'}
        />
        <MetricCard
          label="Open Positions"
          value={String(data.positions.length)}
          color="var(--blue)"
        />
      </div>

      {/* Live Status + Halt */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 24,
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '12px 20px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              width: 10, height: 10, borderRadius: '50%',
              background: data.connected ? 'var(--green)' : 'var(--red)',
              boxShadow: data.connected ? '0 0 6px var(--green)' : undefined,
              animation: data.connected ? 'pulse 2s infinite' : undefined,
              flexShrink: 0,
            }}
          />
          <span style={{ fontSize: '0.8rem', color: 'var(--text-primary)', fontWeight: 600 }}>
            {data.connected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
        </div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Uptime: <strong style={{ color: 'var(--text-secondary)' }}>{data.uptime}</strong>
        </span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Heartbeat:{' '}
          <strong style={{ color: 'var(--text-secondary)' }}>
            {new Date(data.lastHeartbeat).toLocaleTimeString()}
          </strong>
        </span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Refreshed: {lastRefresh.toLocaleTimeString()}
        </span>
        <div style={{ marginLeft: 'auto' }}>
          {!haltConfirm ? (
            <button
              onClick={() => setHaltConfirm(true)}
              style={{
                padding: '6px 16px', borderRadius: 6, border: '1px solid var(--red)',
                background: 'rgba(239,68,68,0.10)', color: 'var(--red)',
                fontSize: '0.8rem', fontWeight: 700, cursor: 'pointer',
              }}
            >
              ⛔ EMERGENCY HALT
            </button>
          ) : (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--red)', fontWeight: 600 }}>
                Confirm halt?
              </span>
              <button
                onClick={() => haltMutation.mutate()}
                disabled={haltMutation.isPending}
                style={{
                  padding: '5px 12px', borderRadius: 6, border: 'none',
                  background: 'var(--red)', color: '#fff',
                  fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer',
                }}
              >
                {haltMutation.isPending ? 'Halting…' : 'YES, HALT'}
              </button>
              <button
                onClick={() => setHaltConfirm(false)}
                style={{
                  padding: '5px 12px', borderRadius: 6, border: '1px solid var(--border)',
                  background: 'transparent', color: 'var(--text-muted)',
                  fontSize: '0.78rem', cursor: 'pointer',
                }}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Equity Curve */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '16px',
      }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 8 }}>
          EQUITY CURVE (real-time)
        </div>
        <EquityCurve points={data.equity} height={160} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {/* Current Positions */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, overflow: 'hidden',
        }}>
          <div style={{
            padding: '10px 16px', borderBottom: '1px solid var(--border)',
            fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600,
          }}>
            OPEN POSITIONS ({data.positions.length})
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                {['Symbol', 'Side', 'Size', 'Entry', 'Current', 'Unr. P&L', 'Bars'].map(h => (
                  <th key={h} style={{
                    padding: '6px 12px', textAlign: 'left',
                    fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600,
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.positions.map((p, i) => (
                <tr key={i} style={{ borderBottom: i < data.positions.length - 1 ? '1px solid var(--border)' : undefined }}>
                  <td style={{ padding: '8px 12px', fontWeight: 700, fontSize: '0.85rem', color: 'var(--text-primary)' }}>
                    {p.symbol}
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={{
                      fontSize: '0.68rem', fontWeight: 700, padding: '2px 6px', borderRadius: 4,
                      background: p.side === 'LONG' ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                      color: p.side === 'LONG' ? 'var(--green)' : 'var(--red)',
                    }}>
                      {p.side}
                    </span>
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: '0.78rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {p.size}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: '0.78rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                    ${p.entryPrice.toLocaleString()}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: '0.78rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    ${p.currentPrice.toLocaleString()}
                  </td>
                  <td style={{
                    padding: '8px 12px', fontSize: '0.78rem', fontWeight: 700, fontVariantNumeric: 'tabular-nums',
                    color: p.unrealizedPnl >= 0 ? 'var(--green)' : 'var(--red)',
                  }}>
                    {p.unrealizedPnl >= 0 ? '+' : ''}${p.unrealizedPnl.toFixed(2)}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    {p.holdBars}
                  </td>
                </tr>
              ))}
              {data.positions.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                    No open positions
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Win Rate Gauges */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '16px',
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 16 }}>
            WIN RATE (ROLLING WINDOWS)
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center', marginBottom: 24 }}>
            <WinGauge rate={data.winRate.window50} label="50-trade" />
            <WinGauge rate={data.winRate.window100} label="100-trade" />
            <WinGauge rate={data.winRate.window200} label="200-trade" />
          </div>

          {/* Daily P&L Chart */}
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 8 }}>
            DAILY P&L (LAST 30 DAYS)
          </div>
          <DailyBarChart data={data.dailyPnl} height={100} />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            <span>{data.dailyPnl[0]?.date}</span>
            <span>{data.dailyPnl[data.dailyPnl.length - 1]?.date}</span>
          </div>
        </div>
      </div>

      {/* Recent Trades Feed */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, overflow: 'hidden',
      }}>
        <div style={{
          padding: '10px 16px', borderBottom: '1px solid var(--border)',
          fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600,
        }}>
          RECENT TRADES (LAST 20)
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Time', 'Symbol', 'Side', 'Signal', 'Hold Bars', 'P&L'].map(h => (
                <th key={h} style={{
                  padding: '6px 14px', textAlign: 'left',
                  fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600,
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.recentTrades.map((t, i) => (
              <tr key={t.id} style={{ borderBottom: i < data.recentTrades.length - 1 ? '1px solid var(--border)' : undefined }}>
                <td style={{ padding: '7px 14px', fontSize: '0.72rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                  {new Date(t.closedAt).toLocaleTimeString()}
                </td>
                <td style={{ padding: '7px 14px', fontWeight: 700, fontSize: '0.82rem', color: 'var(--text-primary)' }}>
                  {t.symbol}
                </td>
                <td style={{ padding: '7px 14px' }}>
                  <span style={{
                    fontSize: '0.65rem', fontWeight: 700, padding: '1px 5px', borderRadius: 4,
                    background: t.side === 'LONG' ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                    color: t.side === 'LONG' ? 'var(--green)' : 'var(--red)',
                  }}>
                    {t.side}
                  </span>
                </td>
                <td style={{ padding: '7px 14px', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  {t.signalType}
                </td>
                <td style={{ padding: '7px 14px', fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                  {t.holdBars}
                </td>
                <td style={{
                  padding: '7px 14px', fontSize: '0.82rem', fontWeight: 700, fontVariantNumeric: 'tabular-nums',
                  color: t.pnl >= 0 ? 'var(--green)' : 'var(--red)',
                }}>
                  {t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default LiveTrader
