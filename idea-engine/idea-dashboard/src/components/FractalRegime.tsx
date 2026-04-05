import React from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from './LoadingSpinner'

// ─── Types ────────────────────────────────────────────────────────────────────

type FractalRegimeType = 'TRENDING' | 'CHOPPY' | 'TRANSITIONING'

interface SymbolFractal {
  symbol: string
  hurstExponent: number       // 0–1; 0.5=random, >0.5=trending, <0.5=mean-reverting
  fractalDimension: number    // D = 2 - H; ~1.5 for random, <1.5 for trending
  regime: FractalRegimeType
  regimeConfidence: number    // 0–1
  signalQuality: number       // 0–100; how reliable BH signals are in current regime
  hurstHistory: number[]      // last 20 rolling Hurst values
}

interface FractalData {
  symbols: SymbolFractal[]
  updatedAt: string
}

// ─── API ──────────────────────────────────────────────────────────────────────

async function fetchFractalData(): Promise<FractalData> {
  try {
    const res = await fetch('/api/fractal/regime')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK_FRACTAL
  }
}

function hurstHistory(base: number): number[] {
  const h: number[] = []
  let v = base
  for (let i = 0; i < 20; i++) {
    v = Math.max(0.2, Math.min(0.9, v + (Math.random() - 0.5) * 0.05))
    h.push(v)
  }
  return h
}

const MOCK_FRACTAL: FractalData = {
  updatedAt: new Date().toISOString(),
  symbols: [
    { symbol: 'BTC',  hurstExponent: 0.64, fractalDimension: 1.36, regime: 'TRENDING',     regimeConfidence: 0.82, signalQuality: 85, hurstHistory: hurstHistory(0.64) },
    { symbol: 'ETH',  hurstExponent: 0.58, fractalDimension: 1.42, regime: 'TRENDING',     regimeConfidence: 0.65, signalQuality: 72, hurstHistory: hurstHistory(0.58) },
    { symbol: 'SOL',  hurstExponent: 0.52, fractalDimension: 1.48, regime: 'TRANSITIONING',regimeConfidence: 0.51, signalQuality: 55, hurstHistory: hurstHistory(0.52) },
    { symbol: 'BNB',  hurstExponent: 0.45, fractalDimension: 1.55, regime: 'CHOPPY',       regimeConfidence: 0.70, signalQuality: 35, hurstHistory: hurstHistory(0.45) },
    { symbol: 'DOGE', hurstExponent: 0.38, fractalDimension: 1.62, regime: 'CHOPPY',       regimeConfidence: 0.78, signalQuality: 22, hurstHistory: hurstHistory(0.38) },
    { symbol: 'XRP',  hurstExponent: 0.55, fractalDimension: 1.45, regime: 'TRANSITIONING',regimeConfidence: 0.48, signalQuality: 60, hurstHistory: hurstHistory(0.55) },
    { symbol: 'ADA',  hurstExponent: 0.41, fractalDimension: 1.59, regime: 'CHOPPY',       regimeConfidence: 0.68, signalQuality: 28, hurstHistory: hurstHistory(0.41) },
    { symbol: 'AVAX', hurstExponent: 0.61, fractalDimension: 1.39, regime: 'TRENDING',     regimeConfidence: 0.74, signalQuality: 78, hurstHistory: hurstHistory(0.61) },
  ],
}

// ─── Component Helpers ────────────────────────────────────────────────────────

const REGIME_STYLE: Record<FractalRegimeType, { color: string; bg: string; label: string }> = {
  TRENDING:      { color: 'var(--green)',  bg: 'rgba(34,197,94,0.12)',  label: 'TRENDING' },
  CHOPPY:        { color: 'var(--red)',    bg: 'rgba(239,68,68,0.12)',  label: 'CHOPPY'   },
  TRANSITIONING: { color: 'var(--yellow)', bg: 'rgba(234,179,8,0.12)', label: 'TRANSIT.' },
}

function hurstColor(h: number): string {
  if (h > 0.6) return 'var(--green)'
  if (h > 0.55) return '#86efac'
  if (h > 0.45) return 'var(--text-muted)'
  if (h > 0.38) return '#fca5a5'
  return 'var(--red)'
}

// ─── Hurst Gauge ──────────────────────────────────────────────────────────────

interface HurstGaugeProps {
  value: number   // 0–1
}

const HurstGauge: React.FC<HurstGaugeProps> = ({ value }) => {
  const cx = 50, cy = 50, r = 36
  const angle = (value * 180) - 90   // map 0→-90deg, 1→+90deg
  const toXY  = (deg: number) => ({
    x: cx + r * Math.cos((deg * Math.PI) / 180),
    y: cy + r * Math.sin((deg * Math.PI) / 180),
  })

  // Arc from -90 to angle
  const start = toXY(-90)
  const end   = toXY(angle)
  const large = value > 0.5 ? 1 : 0
  const color = hurstColor(value)

  return (
    <svg width={90} height={56} viewBox="0 0 100 60">
      {/* Background arc */}
      <path
        d={`M ${toXY(-90).x} ${toXY(-90).y} A ${r} ${r} 0 0 1 ${toXY(90).x} ${toXY(90).y}`}
        fill="none" stroke="var(--border)" strokeWidth={7} strokeLinecap="round"
      />
      {/* Zone colors: red zone 0-0.45, yellow 0.45-0.55, green 0.55-1.0 */}
      {/* Filled arc */}
      <path
        d={`M ${start.x} ${start.y} A ${r} ${r} 0 ${large} 1 ${end.x} ${end.y}`}
        fill="none" stroke={color} strokeWidth={7} strokeLinecap="round"
      />
      {/* Center value */}
      <text x={cx} y={54} textAnchor="middle" fontSize="11" fill="var(--text-primary)" fontWeight="700">
        {value.toFixed(2)}
      </text>
    </svg>
  )
}

// ─── Mini Hurst History Sparkline ─────────────────────────────────────────────

interface HurstSparklineProps {
  data: number[]
  width?: number
  height?: number
}

const HurstSparkline: React.FC<HurstSparklineProps> = ({ data, width = 80, height = 28 }) => {
  if (data.length < 2) return null
  const min = 0.2, max = 0.9, range = max - min
  const step = width / (data.length - 1)

  const points = data.map((v, i) => ({
    x: i * step,
    y: height - ((v - min) / range) * height,
  }))

  // Color gradient: use last value's color
  const lastColor = hurstColor(data[data.length - 1])

  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      {/* 0.5 line */}
      <line
        x1={0} y1={height - ((0.5 - min) / range) * height}
        x2={width} y2={height - ((0.5 - min) / range) * height}
        stroke="var(--border)" strokeWidth={0.8} strokeDasharray="2,2"
      />
      <polyline
        points={points.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')}
        fill="none" stroke={lastColor} strokeWidth={1.5}
        strokeLinejoin="round" strokeLinecap="round"
      />
    </svg>
  )
}

// ─── Symbol Fractal Card ──────────────────────────────────────────────────────

interface SymbolCardProps {
  s: SymbolFractal
}

const SymbolFractalCard: React.FC<SymbolCardProps> = ({ s }) => {
  const rs    = REGIME_STYLE[s.regime]
  const sqColor = s.signalQuality >= 70 ? 'var(--green)'
    : s.signalQuality >= 45 ? 'var(--yellow)'
    : 'var(--red)'

  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: `1px solid ${rs.color}40`,
      borderRadius: 8,
      padding: '14px 16px',
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
    }}>
      {/* Symbol + regime badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
          {s.symbol}
        </span>
        <span style={{
          fontSize: '0.65rem', fontWeight: 700, padding: '2px 7px',
          background: rs.bg, color: rs.color,
          borderRadius: 4, letterSpacing: '0.04em',
        }}>
          {rs.label}
        </span>
        <span style={{ marginLeft: 'auto', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          {(s.regimeConfidence * 100).toFixed(0)}% conf
        </span>
      </div>

      {/* Hurst gauge + sparkline */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <HurstGauge value={s.hurstExponent} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flex: 1 }}>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>Hurst trend</div>
          <HurstSparkline data={s.hurstHistory} />
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, paddingTop: 4, borderTop: '1px solid var(--border)' }}>
        <div>
          <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: 2 }}>
            Fractal dim D
          </div>
          <div style={{ fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
            {s.fractalDimension.toFixed(3)}
          </div>
        </div>
        <div>
          <div style={{ fontSize: '0.62rem', color: 'var(--text-muted)', marginBottom: 2 }}>
            BH signal quality
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ flex: 1, height: 5, borderRadius: 3, background: 'var(--bg-hover)', overflow: 'hidden' }}>
              <div style={{ width: `${s.signalQuality}%`, height: '100%', background: sqColor, transition: 'width 0.4s' }} />
            </div>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: sqColor, minWidth: 24 }}>
              {s.signalQuality}
            </span>
          </div>
        </div>
      </div>

      {/* Signal quality annotation */}
      <div style={{
        fontSize: '0.7rem',
        color: sqColor,
        padding: '4px 8px',
        background: `${sqColor}10`,
        borderRadius: 4,
      }}>
        {s.regime === 'TRENDING'
          ? `BH strategy is reliable — trending regime detected (H=${s.hurstExponent.toFixed(2)})`
          : s.regime === 'CHOPPY'
          ? `BH signals unreliable — choppy market (H=${s.hurstExponent.toFixed(2)}); reduce size`
          : `Regime in flux — wait for H to stabilise before trading`
        }
      </div>
    </div>
  )
}

// ─── Component ────────────────────────────────────────────────────────────────

interface FractalRegimeProps {
  compact?: boolean
}

const FractalRegime: React.FC<FractalRegimeProps> = ({ compact = false }) => {
  const { data, isLoading } = useQuery<FractalData>({
    queryKey: ['fractal', 'regime'],
    queryFn: fetchFractalData,
    refetchInterval: 60_000,
  })

  if (isLoading || !data) return <LoadingSpinner message="Loading fractal regime data…" />

  const nTrending = data.symbols.filter(s => s.regime === 'TRENDING').length
  const nChoppy   = data.symbols.filter(s => s.regime === 'CHOPPY').length
  const avgHurst  = data.symbols.reduce((a, s) => a + s.hurstExponent, 0) / data.symbols.length

  if (compact) {
    // Compact mode: just the grid of symbol badges
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {data.symbols.map((s) => {
            const rs = REGIME_STYLE[s.regime]
            return (
              <div key={s.symbol} title={`H=${s.hurstExponent.toFixed(2)}, ${s.regime}`} style={{
                padding: '4px 10px',
                background: rs.bg,
                border: `1px solid ${rs.color}40`,
                borderRadius: 5,
                display: 'flex', alignItems: 'center', gap: 5,
              }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-primary)' }}>{s.symbol}</span>
                <span style={{ fontSize: '0.7rem', color: rs.color, fontWeight: 700 }}>{s.hurstExponent.toFixed(2)}</span>
              </div>
            )
          })}
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          {nTrending} trending · {nChoppy} choppy · avg H={avgHurst.toFixed(2)}
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px' }}>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: 4 }}>Avg Hurst</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: hurstColor(avgHurst) }}>{avgHurst.toFixed(3)}</div>
        </div>
        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px' }}>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: 4 }}>Trending symbols</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--green)' }}>{nTrending}</div>
        </div>
        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px' }}>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: 4 }}>Choppy symbols</div>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--red)' }}>{nChoppy}</div>
        </div>
        <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 14px' }}>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: 4 }}>Updated</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{new Date(data.updatedAt).toLocaleTimeString()}</div>
        </div>
      </div>

      {/* Hurst scale legend */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '10px 14px',
        display: 'flex', alignItems: 'center', gap: 16,
        fontSize: '0.72rem', color: 'var(--text-muted)',
        flexWrap: 'wrap',
      }}>
        <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>Hurst scale:</span>
        <span style={{ color: 'var(--red)' }}>H&lt;0.45 = Mean-reverting</span>
        <span>0.45–0.55 = Random walk</span>
        <span style={{ color: 'var(--green)' }}>H&gt;0.55 = Persistent/trending</span>
        <span style={{ marginLeft: 'auto' }}>D = 2 − H (fractal dimension)</span>
      </div>

      {/* Symbol grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 10 }}>
        {data.symbols.map((s) => (
          <SymbolFractalCard key={s.symbol} s={s} />
        ))}
      </div>
    </div>
  )
}

export default FractalRegime
