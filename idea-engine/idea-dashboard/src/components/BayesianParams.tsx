import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from './LoadingSpinner'

// ─── Types ────────────────────────────────────────────────────────────────────

interface ParameterDistribution {
  name: string
  displayName: string
  unit?: string
  prior: {
    mean: number
    std: number
    samples: number[]
  }
  posterior: {
    mean: number
    std: number
    ci95Lower: number
    ci95Upper: number
    samples: number[]
  }
  drifted: boolean
  driftMagnitude: number    // (posterior.mean - prior.mean) / prior.std
  recommendedValue: number
  currentValue: number
}

interface BayesianParamsData {
  parameters: ParameterDistribution[]
  genomeId?: number
  updatedAt: string
}

// ─── API ──────────────────────────────────────────────────────────────────────

async function fetchBayesianParams(genomeId?: number): Promise<BayesianParamsData> {
  try {
    const url = genomeId
      ? `/api/bayesian/params?genome=${genomeId}`
      : '/api/bayesian/params'
    const res = await fetch(url)
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK_PARAMS
  }
}

// Generate synthetic Gaussian samples for mock data
function gaussianSamples(mu: number, sigma: number, n = 200): number[] {
  const out: number[] = []
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(Math.random(), 1e-12)
    const u2 = Math.random()
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2)
    out.push(mu + sigma * z0)
    if (i + 1 < n) out.push(mu + sigma * z1)
  }
  return out
}

const MOCK_PARAMS: BayesianParamsData = {
  updatedAt: new Date().toISOString(),
  parameters: [
    {
      name: 'lookback', displayName: 'Lookback Period', unit: 'bars',
      prior:     { mean: 20, std: 5,   samples: gaussianSamples(20, 5) },
      posterior: { mean: 24, std: 3.1, ci95Lower: 18.1, ci95Upper: 29.9, samples: gaussianSamples(24, 3.1) },
      drifted: true, driftMagnitude: 0.80, recommendedValue: 24, currentValue: 20,
    },
    {
      name: 'threshold', displayName: 'Entry Threshold', unit: '%',
      prior:     { mean: 0.5, std: 0.15, samples: gaussianSamples(0.5, 0.15) },
      posterior: { mean: 0.52, std: 0.11, ci95Lower: 0.305, ci95Upper: 0.735, samples: gaussianSamples(0.52, 0.11) },
      drifted: false, driftMagnitude: 0.13, recommendedValue: 0.52, currentValue: 0.5,
    },
    {
      name: 'stopLoss', displayName: 'Stop Loss', unit: '%',
      prior:     { mean: 1.5, std: 0.4, samples: gaussianSamples(1.5, 0.4) },
      posterior: { mean: 1.2, std: 0.25, ci95Lower: 0.71, ci95Upper: 1.69, samples: gaussianSamples(1.2, 0.25) },
      drifted: true, driftMagnitude: -0.75, recommendedValue: 1.2, currentValue: 1.5,
    },
    {
      name: 'positionSize', displayName: 'Position Size', unit: '% equity',
      prior:     { mean: 10, std: 3,   samples: gaussianSamples(10, 3) },
      posterior: { mean: 12.4, std: 2.1, ci95Lower: 8.3, ci95Upper: 16.5, samples: gaussianSamples(12.4, 2.1) },
      drifted: true, driftMagnitude: 0.80, recommendedValue: 12.4, currentValue: 10,
    },
    {
      name: 'macdFast', displayName: 'MACD Fast', unit: 'bars',
      prior:     { mean: 12, std: 2,   samples: gaussianSamples(12, 2) },
      posterior: { mean: 11.8, std: 1.6, ci95Lower: 8.7, ci95Upper: 14.9, samples: gaussianSamples(11.8, 1.6) },
      drifted: false, driftMagnitude: -0.10, recommendedValue: 12, currentValue: 12,
    },
  ],
}

// ─── Histogram Component ──────────────────────────────────────────────────────

interface HistogramProps {
  priorSamples: number[]
  posteriorSamples: number[]
  ci95Lower: number
  ci95Upper: number
  currentValue: number
  recommendedValue: number
  width?: number
  height?: number
}

const DistributionChart: React.FC<HistogramProps> = ({
  priorSamples,
  posteriorSamples,
  ci95Lower, ci95Upper,
  currentValue, recommendedValue,
  width = 280, height = 80,
}) => {
  const allSamples = [...priorSamples, ...posteriorSamples]
  const min = Math.min(...allSamples)
  const max = Math.max(...allSamples)
  const range = max - min || 1

  const N_BINS = 30
  const binWidth = range / N_BINS

  const binSamples = (samples: number[]) => {
    const bins = Array(N_BINS).fill(0)
    for (const s of samples) {
      const idx = Math.min(Math.floor((s - min) / binWidth), N_BINS - 1)
      bins[idx]++
    }
    return bins
  }

  const priorBins   = binSamples(priorSamples)
  const posterBins  = binSamples(posteriorSamples)
  const maxCount    = Math.max(...priorBins, ...posterBins)

  const toX  = (v: number) => ((v - min) / range) * width
  const toH  = (c: number) => (c / maxCount) * height

  const barW = width / N_BINS - 0.5

  // CI shading rect
  const ci_x1 = toX(ci95Lower)
  const ci_x2 = toX(ci95Upper)

  return (
    <svg width={width} height={height + 14} style={{ overflow: 'visible' }}>
      {/* CI shading */}
      <rect
        x={ci_x1} y={0}
        width={Math.max(0, ci_x2 - ci_x1)} height={height}
        fill="rgba(59,130,246,0.08)" rx={2}
      />

      {/* Prior bars */}
      {priorBins.map((count, i) => {
        const x = i * (width / N_BINS)
        const h = toH(count)
        return (
          <rect key={`p${i}`} x={x} y={height - h} width={barW} height={h}
            fill="rgba(148,163,184,0.25)" rx={1} />
        )
      })}

      {/* Posterior bars */}
      {posterBins.map((count, i) => {
        const x = i * (width / N_BINS)
        const h = toH(count)
        return (
          <rect key={`q${i}`} x={x} y={height - h} width={barW} height={h}
            fill="rgba(59,130,246,0.55)" rx={1} />
        )
      })}

      {/* Current value line */}
      <line
        x1={toX(currentValue)} y1={0} x2={toX(currentValue)} y2={height}
        stroke="var(--yellow)" strokeWidth={1.5} strokeDasharray="3,2"
      />

      {/* Recommended value line */}
      <line
        x1={toX(recommendedValue)} y1={0} x2={toX(recommendedValue)} y2={height}
        stroke="var(--green)" strokeWidth={1.5}
      />

      {/* Axis labels */}
      <text x={0} y={height + 11} fontSize={9} fill="var(--text-muted)">
        {min.toFixed(1)}
      </text>
      <text x={width} y={height + 11} fontSize={9} fill="var(--text-muted)" textAnchor="end">
        {max.toFixed(1)}
      </text>
    </svg>
  )
}

// ─── Single Parameter Card ────────────────────────────────────────────────────

interface ParamCardProps {
  param: ParameterDistribution
}

const ParamCard: React.FC<ParamCardProps> = ({ param }) => {
  const driftColor = Math.abs(param.driftMagnitude) > 0.5
    ? (param.driftMagnitude > 0 ? 'var(--green)' : 'var(--red)')
    : 'var(--text-muted)'

  const changeDir = param.recommendedValue > param.currentValue ? '↑' : '↓'
  const changePct = param.currentValue !== 0
    ? ((param.recommendedValue - param.currentValue) / Math.abs(param.currentValue)) * 100
    : 0

  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: `1px solid ${param.drifted ? 'var(--accent)' : 'var(--border)'}`,
      borderRadius: 8,
      padding: '14px 16px',
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
          {param.displayName}
        </span>
        {param.unit && (
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>({param.unit})</span>
        )}
        {param.drifted && (
          <span style={{
            marginLeft: 'auto',
            fontSize: '0.65rem', fontWeight: 700, padding: '2px 7px',
            background: 'rgba(251,146,60,0.15)',
            color: '#fb923c',
            border: '1px solid rgba(251,146,60,0.3)',
            borderRadius: 4,
          }}>
            DRIFTED
          </span>
        )}
      </div>

      {/* Distribution chart */}
      <DistributionChart
        priorSamples={param.prior.samples}
        posteriorSamples={param.posterior.samples}
        ci95Lower={param.posterior.ci95Lower}
        ci95Upper={param.posterior.ci95Upper}
        currentValue={param.currentValue}
        recommendedValue={param.recommendedValue}
      />

      {/* Legend */}
      <div style={{ display: 'flex', gap: 12, fontSize: '0.68rem', color: 'var(--text-muted)' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <span style={{ display: 'inline-block', width: 10, height: 8, background: 'rgba(148,163,184,0.5)', borderRadius: 1 }} />
          Prior
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <span style={{ display: 'inline-block', width: 10, height: 8, background: 'rgba(59,130,246,0.7)', borderRadius: 1 }} />
          Posterior
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <span style={{ display: 'inline-block', width: 10, height: 2, background: 'var(--yellow)' }} />
          Current
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <span style={{ display: 'inline-block', width: 10, height: 2, background: 'var(--green)' }} />
          Recommended
        </span>
      </div>

      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, paddingTop: 4, borderTop: '1px solid var(--border)' }}>
        <div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 2 }}>Posterior mean</div>
          <div style={{ fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
            {param.posterior.mean.toFixed(2)}
          </div>
        </div>
        <div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 2 }}>95% CI</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
            [{param.posterior.ci95Lower.toFixed(2)}, {param.posterior.ci95Upper.toFixed(2)}]
          </div>
        </div>
        <div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 2 }}>Change</div>
          <div style={{ fontSize: '0.875rem', fontWeight: 700, color: driftColor, fontVariantNumeric: 'tabular-nums' }}>
            {changeDir} {Math.abs(changePct).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Recommendation */}
      {param.drifted && (
        <div style={{
          fontSize: '0.75rem',
          padding: '6px 10px',
          background: 'rgba(59,130,246,0.08)',
          borderRadius: 5,
          color: 'var(--text-secondary)',
          borderLeft: '2px solid var(--accent)',
        }}>
          Update {param.displayName} from <strong>{param.currentValue}</strong> → <strong>{param.recommendedValue.toFixed(2)}</strong> ({param.unit})
        </div>
      )}
    </div>
  )
}

// ─── Component ────────────────────────────────────────────────────────────────

interface BayesianParamsProps {
  genomeId?: number
}

const BayesianParams: React.FC<BayesianParamsProps> = ({ genomeId }) => {
  const { data, isLoading } = useQuery<BayesianParamsData>({
    queryKey: ['bayesian', 'params', genomeId],
    queryFn: () => fetchBayesianParams(genomeId),
    refetchInterval: 60_000,
  })

  const [showOnlyDrifted, setShowOnlyDrifted] = useState(false)

  if (isLoading || !data) return <LoadingSpinner message="Loading Bayesian posteriors…" />

  const params = showOnlyDrifted
    ? data.parameters.filter((p) => p.drifted)
    : data.parameters

  const nDrifted = data.parameters.filter((p) => p.drifted).length

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          {nDrifted} parameter{nDrifted !== 1 ? 's' : ''} drifted from prior
        </span>
        <button
          className="btn-icon"
          onClick={() => setShowOnlyDrifted((v) => !v)}
          style={{
            marginLeft: 'auto',
            padding: '4px 12px', borderRadius: 5, fontSize: '0.75rem', border: 'none',
            background: showOnlyDrifted ? 'var(--accent)' : 'var(--bg-hover)',
            color: showOnlyDrifted ? '#000' : 'var(--text-muted)',
            fontWeight: showOnlyDrifted ? 700 : 400,
          }}
        >
          Show drifted only
        </button>
        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
          Updated {new Date(data.updatedAt).toLocaleTimeString()}
        </span>
      </div>

      {/* Parameter grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 12 }}>
        {params.map((p) => (
          <ParamCard key={p.name} param={p} />
        ))}
      </div>
    </div>
  )
}

export default BayesianParams
