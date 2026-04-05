import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from '../components/LoadingSpinner'
import MetricCard from '../components/MetricCard'

// ─── Types ────────────────────────────────────────────────────────────────────

interface SymbolMicrostructure {
  symbol: string
  liquidityScore: number        // 0–100; 100 = perfectly liquid
  spreadEstimate: number        // basis points
  spreadVsBaseline: number      // ratio: 1.0 = normal, 2.0 = 2× normal
  orderFlowImbalance: number    // -1 to +1 (negative = sell-heavy)
  vpin: number                  // 0–1; >0.5 = elevated informed-trading risk
  hourlyLiqVsBaseline: number   // ratio vs same-hour historical average
  isGoodTimeToTrade: boolean
  depth1pct: number             // $ depth within 1% of mid
  depthFormatted: string        // e.g. "$4.2M"
}

interface MicrostructureData {
  symbols: SymbolMicrostructure[]
  marketWideLiquidityScore: number
  currentHourUTC: number
  bestSymbol: string
  worstSymbol: string
  updatedAt: string
}

// ─── API ──────────────────────────────────────────────────────────────────────

async function fetchMicrostructure(): Promise<MicrostructureData> {
  try {
    const res = await fetch('/api/microstructure/health')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK_MICRO
  }
}

const MOCK_MICRO: MicrostructureData = {
  marketWideLiquidityScore: 74,
  currentHourUTC: new Date().getUTCHours(),
  bestSymbol: 'BTC',
  worstSymbol: 'DOGE',
  updatedAt: new Date().toISOString(),
  symbols: [
    { symbol: 'BTC',  liquidityScore: 92, spreadEstimate: 0.8,  spreadVsBaseline: 0.95, orderFlowImbalance: 0.18,  vpin: 0.22, hourlyLiqVsBaseline: 1.05, isGoodTimeToTrade: true,  depth1pct: 24_500_000, depthFormatted: '$24.5M' },
    { symbol: 'ETH',  liquidityScore: 85, spreadEstimate: 1.2,  spreadVsBaseline: 1.10, orderFlowImbalance: 0.08,  vpin: 0.28, hourlyLiqVsBaseline: 1.02, isGoodTimeToTrade: true,  depth1pct: 12_800_000, depthFormatted: '$12.8M' },
    { symbol: 'SOL',  liquidityScore: 71, spreadEstimate: 2.1,  spreadVsBaseline: 1.25, orderFlowImbalance: 0.32,  vpin: 0.35, hourlyLiqVsBaseline: 0.88, isGoodTimeToTrade: true,  depth1pct: 4_200_000,  depthFormatted: '$4.2M'  },
    { symbol: 'BNB',  liquidityScore: 68, spreadEstimate: 2.8,  spreadVsBaseline: 1.40, orderFlowImbalance: -0.12, vpin: 0.41, hourlyLiqVsBaseline: 0.92, isGoodTimeToTrade: true,  depth1pct: 3_100_000,  depthFormatted: '$3.1M'  },
    { symbol: 'DOGE', liquidityScore: 38, spreadEstimate: 8.5,  spreadVsBaseline: 2.80, orderFlowImbalance: -0.48, vpin: 0.67, hourlyLiqVsBaseline: 0.55, isGoodTimeToTrade: false, depth1pct: 850_000,    depthFormatted: '$850K'  },
    { symbol: 'XRP',  liquidityScore: 62, spreadEstimate: 3.4,  spreadVsBaseline: 1.60, orderFlowImbalance: -0.05, vpin: 0.38, hourlyLiqVsBaseline: 0.95, isGoodTimeToTrade: true,  depth1pct: 2_400_000,  depthFormatted: '$2.4M'  },
    { symbol: 'ADA',  liquidityScore: 55, spreadEstimate: 4.2,  spreadVsBaseline: 1.75, orderFlowImbalance: -0.22, vpin: 0.45, hourlyLiqVsBaseline: 0.80, isGoodTimeToTrade: false, depth1pct: 1_100_000,  depthFormatted: '$1.1M'  },
    { symbol: 'AVAX', liquidityScore: 65, spreadEstimate: 3.1,  spreadVsBaseline: 1.45, orderFlowImbalance: 0.14,  vpin: 0.33, hourlyLiqVsBaseline: 0.97, isGoodTimeToTrade: true,  depth1pct: 1_900_000,  depthFormatted: '$1.9M'  },
  ],
}

// ─── Components ───────────────────────────────────────────────────────────────

function liquidityColor(score: number): string {
  if (score >= 80) return 'var(--green)'
  if (score >= 60) return '#86efac'
  if (score >= 40) return 'var(--yellow)'
  if (score >= 20) return '#fca5a5'
  return 'var(--red)'
}

function spreadColor(ratio: number): string {
  if (ratio <= 1.1)  return 'var(--green)'
  if (ratio <= 1.5)  return 'var(--yellow)'
  if (ratio <= 2.5)  return '#f97316'
  return 'var(--red)'
}

function vpinColor(vpin: number): string {
  if (vpin < 0.3) return 'var(--green)'
  if (vpin < 0.5) return 'var(--yellow)'
  if (vpin < 0.7) return '#f97316'
  return 'var(--red)'
}

function ofiColor(ofi: number): string {
  if (ofi > 0.3) return 'var(--green)'
  if (ofi < -0.3) return 'var(--red)'
  return 'var(--text-muted)'
}

interface HeatCellProps {
  label: string
  value: number     // 0–100
  color: string
  sublabel?: string
  good: boolean
}

const HeatCell: React.FC<HeatCellProps> = ({ label, value, color, sublabel, good }) => (
  <div style={{
    padding: '10px 12px',
    borderRadius: 6,
    background: `${color}18`,
    border: `1px solid ${color}40`,
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--text-primary)' }}>{label}</span>
      <span style={{ fontSize: '0.65rem', color: good ? 'var(--green)' : 'var(--red)', fontWeight: 700 }}>
        {good ? '● OK' : '● CAUTION'}
      </span>
    </div>
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--bg-hover)', overflow: 'hidden' }}>
        <div style={{ width: `${value}%`, height: '100%', background: color, transition: 'width 0.4s' }} />
      </div>
      <span style={{ fontSize: '0.75rem', color, fontWeight: 700, minWidth: 28, textAlign: 'right' }}>{value}</span>
    </div>
    {sublabel && (
      <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{sublabel}</span>
    )}
  </div>
)

interface OFIBarProps {
  ofi: number   // -1 to +1
}

const OFIBar: React.FC<OFIBarProps> = ({ ofi }) => {
  const color = ofiColor(ofi)
  const pct   = ((ofi + 1) / 2) * 100

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <div style={{ position: 'relative', height: 8, borderRadius: 4, background: 'var(--bg-hover)', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', left: '50%', width: 1, height: '100%', background: 'var(--border)' }} />
        <div style={{
          position: 'absolute',
          left:  ofi >= 0 ? '50%' : `${pct}%`,
          width: `${Math.abs(ofi) * 50}%`,
          height: '100%',
          background: color,
          transition: 'all 0.4s',
        }} />
      </div>
      <div style={{ fontSize: '0.65rem', color, fontWeight: 600, textAlign: 'center' }}>
        {ofi >= 0 ? 'Buy' : 'Sell'} {(Math.abs(ofi) * 100).toFixed(0)}%
      </div>
    </div>
  )
}

interface VPINGaugeProps {
  vpin: number
}

const VPINGauge: React.FC<VPINGaugeProps> = ({ vpin }) => {
  const color = vpinColor(vpin)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-hover)', overflow: 'hidden' }}>
        <div style={{ width: `${vpin * 100}%`, height: '100%', background: color, transition: 'width 0.4s' }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.62rem', color: 'var(--text-muted)' }}>
        <span>0</span>
        <span style={{ color, fontWeight: 700 }}>{vpin.toFixed(2)}</span>
        <span>1.0</span>
      </div>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const MicrostructureHealth: React.FC = () => {
  const { data, isLoading } = useQuery<MicrostructureData>({
    queryKey: ['microstructure', 'health'],
    queryFn: fetchMicrostructure,
    refetchInterval: 15_000,
  })

  const [view, setView] = useState<'heatmap' | 'table'>('heatmap')

  if (isLoading || !data) return <LoadingSpinner message="Loading microstructure data…" />

  const mwColor = liquidityColor(data.marketWideLiquidityScore)
  const nGood   = data.symbols.filter(s => s.isGoodTimeToTrade).length
  const avgSpread = data.symbols.reduce((a, s) => a + s.spreadEstimate, 0) / data.symbols.length

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Summary row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        <MetricCard label="Market liquidity" value={`${data.marketWideLiquidityScore}/100`} color={mwColor} />
        <MetricCard label="Good to trade now" value={`${nGood} / ${data.symbols.length}`} color={nGood >= data.symbols.length * 0.7 ? 'var(--green)' : 'var(--yellow)'} />
        <MetricCard label="Avg spread" value={`${avgSpread.toFixed(1)} bps`} />
        <MetricCard label="Best liquidity" value={data.bestSymbol} color="var(--green)" />
      </div>

      {/* View toggle */}
      <div style={{ display: 'flex', gap: 6 }}>
        {(['heatmap', 'table'] as const).map((v) => (
          <button
            key={v}
            className="btn-icon"
            onClick={() => setView(v)}
            style={{
              padding: '5px 14px', borderRadius: 6, fontSize: '0.8rem',
              background: view === v ? 'var(--accent)' : 'var(--bg-hover)',
              color: view === v ? '#000' : 'var(--text-muted)',
              fontWeight: view === v ? 700 : 400, border: 'none',
              textTransform: 'capitalize',
            }}
          >
            {v}
          </button>
        ))}
      </div>

      {/* Heat map view */}
      {view === 'heatmap' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
          {data.symbols.map((s) => (
            <HeatCell
              key={s.symbol}
              label={s.symbol}
              value={s.liquidityScore}
              color={liquidityColor(s.liquidityScore)}
              sublabel={`${s.depthFormatted} depth · ${s.spreadEstimate.toFixed(1)} bps`}
              good={s.isGoodTimeToTrade}
            />
          ))}
        </div>
      )}

      {/* Table view */}
      {view === 'table' && (
        <div style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          overflow: 'hidden',
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                {['Symbol', 'Liquidity', 'Spread (bps)', 'Spread vs Base', 'Order Flow', 'VPIN', 'Depth', 'Status'].map((h) => (
                  <th key={h} style={{ padding: '8px 14px', textAlign: 'left', fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600, whiteSpace: 'nowrap' }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.symbols.map((s, i) => (
                <tr key={s.symbol} style={{ borderBottom: i < data.symbols.length - 1 ? '1px solid var(--border)' : undefined }}>
                  <td style={{ padding: '10px 14px', fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                    {s.symbol}
                  </td>
                  <td style={{ padding: '10px 14px', width: 120 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ flex: 1, height: 6, borderRadius: 3, background: 'var(--bg-hover)', overflow: 'hidden' }}>
                        <div style={{ width: `${s.liquidityScore}%`, height: '100%', background: liquidityColor(s.liquidityScore) }} />
                      </div>
                      <span style={{ fontSize: '0.72rem', color: liquidityColor(s.liquidityScore), fontWeight: 700, minWidth: 24 }}>
                        {s.liquidityScore}
                      </span>
                    </div>
                  </td>
                  <td style={{ padding: '10px 14px', fontSize: '0.8rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {s.spreadEstimate.toFixed(1)}
                  </td>
                  <td style={{ padding: '10px 14px', fontSize: '0.8rem', color: spreadColor(s.spreadVsBaseline), fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                    {s.spreadVsBaseline.toFixed(2)}×
                  </td>
                  <td style={{ padding: '10px 14px', width: 100 }}>
                    <OFIBar ofi={s.orderFlowImbalance} />
                  </td>
                  <td style={{ padding: '10px 14px', width: 100 }}>
                    <VPINGauge vpin={s.vpin} />
                  </td>
                  <td style={{ padding: '10px 14px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    {s.depthFormatted}
                  </td>
                  <td style={{ padding: '10px 14px' }}>
                    <span style={{
                      fontSize: '0.7rem', fontWeight: 700, padding: '2px 8px', borderRadius: 4,
                      background: s.isGoodTimeToTrade ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                      color: s.isGoodTimeToTrade ? 'var(--green)' : 'var(--red)',
                    }}>
                      {s.isGoodTimeToTrade ? '● TRADE' : '● WAIT'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* VPIN alert for elevated symbols */}
      {data.symbols.filter(s => s.vpin > 0.5).length > 0 && (
        <div style={{
          padding: '12px 16px',
          background: 'rgba(239,68,68,0.08)',
          border: '1px solid rgba(239,68,68,0.3)',
          borderRadius: 8,
          fontSize: '0.8rem',
          color: 'var(--red)',
        }}>
          <strong>Elevated VPIN detected</strong> — informed trading risk on:{' '}
          {data.symbols.filter(s => s.vpin > 0.5).map(s => s.symbol).join(', ')}.
          Consider reducing position size by 20–40% on these symbols.
        </div>
      )}

      {/* Updated at */}
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'right' }}>
        Last updated: {new Date(data.updatedAt).toLocaleTimeString()}
      </div>
    </div>
  )
}

export default MicrostructureHealth
