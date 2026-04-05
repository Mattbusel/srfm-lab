import React from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from '../components/LoadingSpinner'
import MetricCard from '../components/MetricCard'

// ─── Types ────────────────────────────────────────────────────────────────────

type MacroRegimeType = 'RISK_ON' | 'RISK_OFF' | 'CRISIS'

interface FactorGauge {
  name: string
  value: number      // raw value
  normalised: number // -1 to +1: negative = bearish, positive = bullish
  label: string      // e.g. "DXY: 104.2 ↑"
  bullish: boolean
}

interface RegimeDay {
  date: string
  regime: MacroRegimeType
  confidence: number
  onChainScore: number
}

interface MacroData {
  currentRegime: MacroRegimeType
  confidence: number
  factors: FactorGauge[]
  positionSizeMultiplier: number
  onChainComposite: number   // 0–100
  regimeHistory: RegimeDay[]
  updatedAt: string
}

// ─── API ──────────────────────────────────────────────────────────────────────

async function fetchMacroData(): Promise<MacroData> {
  try {
    const res = await fetch('/api/macro/regime')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK_MACRO
  }
}

const MOCK_MACRO: MacroData = {
  currentRegime: 'RISK_ON',
  confidence: 0.74,
  positionSizeMultiplier: 1.15,
  onChainComposite: 68,
  updatedAt: new Date().toISOString(),
  factors: [
    { name: 'DXY Trend',       value: 103.4, normalised: -0.45, label: 'DXY 103.4 ↓', bullish: true },
    { name: 'VIX Level',       value: 17.8,  normalised: 0.52,  label: 'VIX 17.8 ↓',  bullish: true },
    { name: 'Yield Curve',     value: 0.32,  normalised: 0.30,  label: '10y-2y +32bp', bullish: true },
    { name: 'BTC Dominance',   value: 54.2,  normalised: 0.20,  label: 'BTC.D 54.2%',  bullish: false },
    { name: 'Funding Rates',   value: 0.012, normalised: 0.60,  label: 'Avg +1.2%/8h', bullish: true },
    { name: 'Stablecoin Inflow', value: 1.8, normalised: 0.55,  label: '+$1.8B/7d',   bullish: true },
  ],
  regimeHistory: Array.from({ length: 60 }, (_, i) => ({
    date: new Date(Date.now() - (59 - i) * 86_400_000).toISOString().split('T')[0],
    regime: i < 10 ? 'RISK_OFF' : i < 15 ? 'CRISIS' : 'RISK_ON',
    confidence: 0.5 + Math.random() * 0.4,
    onChainScore: 40 + Math.random() * 40,
  })),
}

// ─── Components ───────────────────────────────────────────────────────────────

const REGIME_STYLE: Record<MacroRegimeType, { bg: string; color: string; border: string }> = {
  RISK_ON:  { bg: 'rgba(34,197,94,0.12)',  color: 'var(--green)', border: 'rgba(34,197,94,0.4)' },
  RISK_OFF: { bg: 'rgba(234,179,8,0.12)',  color: 'var(--yellow)', border: 'rgba(234,179,8,0.4)' },
  CRISIS:   { bg: 'rgba(239,68,68,0.15)',  color: 'var(--red)',   border: 'rgba(239,68,68,0.5)' },
}

interface RegimeBadgeProps {
  regime: MacroRegimeType
  confidence: number
  large?: boolean
}

const RegimeBadge: React.FC<RegimeBadgeProps> = ({ regime, confidence, large }) => {
  const s = REGIME_STYLE[regime]
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 10,
      padding: large ? '14px 24px' : '6px 14px',
      background: s.bg,
      border: `1px solid ${s.border}`,
      borderRadius: large ? 12 : 20,
    }}>
      <span style={{
        width: large ? 12 : 8, height: large ? 12 : 8,
        borderRadius: '50%', background: s.color,
        boxShadow: `0 0 6px ${s.color}`,
        animation: 'pulse 2s infinite',
        flexShrink: 0,
      }} />
      <span style={{
        fontWeight: 700,
        fontSize: large ? '1.25rem' : '0.8rem',
        color: s.color,
        letterSpacing: '0.04em',
      }}>
        {regime}
      </span>
      <span style={{ fontSize: large ? '0.9rem' : '0.7rem', color: s.color, opacity: 0.8 }}>
        {(confidence * 100).toFixed(0)}%
      </span>
    </div>
  )
}

interface FactorGaugeCardProps {
  factor: FactorGauge
}

const FactorGaugeCard: React.FC<FactorGaugeCardProps> = ({ factor }) => {
  // Normalised: -1 to +1. Map to 0–100% fill.
  const fill    = ((factor.normalised + 1) / 2) * 100
  const color   = factor.bullish ? 'var(--green)' : 'var(--red)'
  const neutral = '#3b82f6'
  const barColor = Math.abs(factor.normalised) < 0.15 ? neutral : color

  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '12px 14px',
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{factor.name}</span>
        <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          {factor.label}
        </span>
      </div>
      {/* Gauge bar: left=bearish, center=neutral, right=bullish */}
      <div style={{ position: 'relative', height: 8, background: 'var(--bg-hover)', borderRadius: 4 }}>
        {/* Center marker */}
        <div style={{
          position: 'absolute', left: '50%', top: 0, width: 1,
          height: '100%', background: 'var(--border)',
        }} />
        {/* Fill */}
        <div style={{
          position: 'absolute',
          left: factor.normalised >= 0 ? '50%' : `${fill}%`,
          width: `${Math.abs(factor.normalised) * 50}%`,
          height: '100%',
          background: barColor,
          borderRadius: 4,
          transition: 'all 0.4s',
        }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
        <span style={{ color: 'var(--red)' }}>BEARISH</span>
        <span>NEUTRAL</span>
        <span style={{ color: 'var(--green)' }}>BULLISH</span>
      </div>
    </div>
  )
}

interface RegimeTimelineProps {
  history: RegimeDay[]
}

const RegimeTimeline: React.FC<RegimeTimelineProps> = ({ history }) => {
  const CELL_W = 10

  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '14px 16px',
    }}>
      <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 10 }}>
        Regime history (last {history.length} days)
      </div>
      <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        {history.map((day) => {
          const s = REGIME_STYLE[day.regime]
          return (
            <div
              key={day.date}
              title={`${day.date}: ${day.regime} (${(day.confidence * 100).toFixed(0)}%)`}
              style={{
                width: CELL_W, height: 20,
                background: s.color,
                opacity: 0.3 + day.confidence * 0.7,
                borderRadius: 2,
              }}
            />
          )
        })}
      </div>
      <div style={{ display: 'flex', gap: 12, marginTop: 8, fontSize: '0.7rem' }}>
        {(['RISK_ON', 'RISK_OFF', 'CRISIS'] as MacroRegimeType[]).map((r) => (
          <span key={r} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: REGIME_STYLE[r].color, display: 'inline-block' }} />
            <span style={{ color: 'var(--text-muted)' }}>{r}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

interface GaugeArcProps {
  value: number   // 0–100
  label: string
  color: string
}

const GaugeArc: React.FC<GaugeArcProps> = ({ value, label, color }) => {
  const angle = (value / 100) * 180 - 90   // -90 = left, +90 = right
  const r     = 38
  const cx    = 50, cy = 50
  const toXY  = (deg: number) => ({
    x: cx + r * Math.cos((deg * Math.PI) / 180),
    y: cy + r * Math.sin((deg * Math.PI) / 180),
  })

  const start = toXY(-90)
  const end   = toXY(angle)
  const large = value > 50 ? 1 : 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <svg width={90} height={55} viewBox="0 0 100 60">
        {/* Background arc */}
        <path
          d={`M ${toXY(-90).x} ${toXY(-90).y} A ${r} ${r} 0 0 1 ${toXY(90).x} ${toXY(90).y}`}
          fill="none" stroke="var(--border)" strokeWidth={7} strokeLinecap="round"
        />
        {/* Filled arc */}
        {value > 0 && (
          <path
            d={`M ${start.x} ${start.y} A ${r} ${r} 0 ${large} 1 ${end.x} ${end.y}`}
            fill="none" stroke={color} strokeWidth={7} strokeLinecap="round"
            style={{ transition: 'd 0.5s' }}
          />
        )}
        <text x={cx} y={54} textAnchor="middle" fontSize="11" fill="var(--text-primary)" fontWeight="700">
          {value}
        </text>
      </svg>
      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{label}</span>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const MacroRegime: React.FC = () => {
  const { data, isLoading } = useQuery<MacroData>({
    queryKey: ['macro', 'regime'],
    queryFn: fetchMacroData,
    refetchInterval: 30_000,
  })

  if (isLoading || !data) return <LoadingSpinner message="Loading macro regime data…" />

  const sizeColor = data.positionSizeMultiplier >= 1.1 ? 'var(--green)'
    : data.positionSizeMultiplier <= 0.8 ? 'var(--red)' : 'var(--yellow)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Current regime */}
      <div style={{
        background: 'var(--bg-surface)',
        border: `1px solid ${REGIME_STYLE[data.currentRegime].border}`,
        borderRadius: 8,
        padding: '20px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: 24,
        flexWrap: 'wrap',
      }}>
        <div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Current Regime
          </div>
          <RegimeBadge regime={data.currentRegime} confidence={data.confidence} large />
        </div>
        <div style={{ height: 60, width: 1, background: 'var(--border)', flexShrink: 0 }} />
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', flex: 1 }}>
          <div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 4 }}>Position Multiplier</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: sizeColor, fontVariantNumeric: 'tabular-nums' }}>
              {data.positionSizeMultiplier.toFixed(2)}×
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 4 }}>On-Chain Composite</div>
            <GaugeArc value={data.onChainComposite} label="On-chain" color="var(--accent)" />
          </div>
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', alignSelf: 'flex-end' }}>
          Updated {new Date(data.updatedAt).toLocaleTimeString()}
        </div>
      </div>

      {/* Factor gauges */}
      <div>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Factor Gauges
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 10 }}>
          {data.factors.map((f) => (
            <FactorGaugeCard key={f.name} factor={f} />
          ))}
        </div>
      </div>

      {/* Summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
        <MetricCard
          label="Bullish factors"
          value={`${data.factors.filter(f => f.bullish).length} / ${data.factors.length}`}
          color="var(--green)"
        />
        <MetricCard
          label="Regime confidence"
          value={`${(data.confidence * 100).toFixed(0)}%`}
          color={REGIME_STYLE[data.currentRegime].color}
        />
        <MetricCard
          label="Size recommendation"
          value={`${(data.positionSizeMultiplier * 100).toFixed(0)}% of normal`}
          color={sizeColor}
        />
      </div>

      {/* Timeline */}
      <RegimeTimeline history={data.regimeHistory} />
    </div>
  )
}

export default MacroRegime
