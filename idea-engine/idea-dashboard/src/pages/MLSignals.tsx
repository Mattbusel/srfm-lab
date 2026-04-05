import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from '../components/LoadingSpinner'
import MetricCard from '../components/MetricCard'

// ─── Types ────────────────────────────────────────────────────────────────────

type ModelName = 'LSTM' | 'Transformer' | 'XGBoost' | 'Ensemble'

interface ModelPerf {
  name: ModelName
  icMean: number
  icStd: number
  icIr: number
  sharpe: number
  hitRate: number
  weight: number
  driftAlert: boolean
  icSeries: number[]    // last 60 periods
}

interface FeatureImportance {
  feature: string
  importance: number
}

interface ScatterPoint {
  predicted: number
  actual: number
  symbol: string
}

interface MLSignalData {
  models: ModelPerf[]
  features: FeatureImportance[]
  scatter: ScatterPoint[]
}

// ─── Mock ─────────────────────────────────────────────────────────────────────

function buildICSeries(mean: number, drift: boolean): number[] {
  const out: number[] = []
  let val = mean
  for (let i = 0; i < 60; i++) {
    val += (Math.random() - 0.5) * 0.03
    if (drift && i > 45) val -= 0.005
    out.push(Math.max(-0.3, Math.min(0.3, val)))
  }
  return out
}

const MOCK: MLSignalData = {
  models: [
    { name: 'LSTM',        icMean: 0.082, icStd: 0.031, icIr: 2.65, sharpe: 1.42, hitRate: 0.56, weight: 0.30, driftAlert: false, icSeries: buildICSeries(0.082, false) },
    { name: 'Transformer', icMean: 0.071, icStd: 0.028, icIr: 2.54, sharpe: 1.31, hitRate: 0.54, weight: 0.25, driftAlert: false, icSeries: buildICSeries(0.071, false) },
    { name: 'XGBoost',     icMean: 0.056, icStd: 0.025, icIr: 2.24, sharpe: 1.18, hitRate: 0.53, weight: 0.20, driftAlert: true,  icSeries: buildICSeries(0.056, true)  },
    { name: 'Ensemble',    icMean: 0.091, icStd: 0.022, icIr: 4.14, sharpe: 1.68, hitRate: 0.58, weight: 0.25, driftAlert: false, icSeries: buildICSeries(0.091, false) },
  ],
  features: [
    { feature: 'momentum_14d',        importance: 0.142 },
    { feature: 'volatility_20d',      importance: 0.118 },
    { feature: 'rsi_14',              importance: 0.098 },
    { feature: 'funding_rate',        importance: 0.087 },
    { feature: 'orderbook_imbalance', importance: 0.081 },
    { feature: 'volume_ratio',        importance: 0.074 },
    { feature: 'on_chain_flows',      importance: 0.068 },
    { feature: 'sentiment_score',     importance: 0.062 },
    { feature: 'correlation_btc',     importance: 0.055 },
    { feature: 'macd_signal',         importance: 0.048 },
    { feature: 'bid_ask_spread',      importance: 0.042 },
    { feature: 'open_interest',       importance: 0.038 },
    { feature: 'social_volume',       importance: 0.031 },
    { feature: 'whale_flows',         importance: 0.028 },
    { feature: 'macro_regime',        importance: 0.022 },
  ],
  scatter: Array.from({ length: 80 }, (_, i) => {
    const predicted = (Math.random() - 0.5) * 0.04
    const noise = (Math.random() - 0.5) * 0.03
    return {
      predicted,
      actual: predicted * 0.7 + noise,
      symbol: ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX'][i % 5],
    }
  }),
}

async function fetchMLData(): Promise<MLSignalData> {
  try {
    const res = await fetch('/api/ml-signals')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK
  }
}

// ─── IC Sparkline ─────────────────────────────────────────────────────────────

interface ICSparklineProps {
  data: number[]
  width?: number
  height?: number
  color: string
}

const ICSparkline: React.FC<ICSparklineProps> = ({ data, width = 120, height = 30, color }) => {
  if (data.length < 2) return null
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 0.01
  const toX = (i: number) => (i / (data.length - 1)) * width
  const toY = (v: number) => height - ((v - min) / range) * height * 0.9 - height * 0.05
  const d = data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${toX(i).toFixed(1)} ${toY(v).toFixed(1)}`).join(' ')
  const zeroY = toY(0)
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <line x1={0} y1={zeroY} x2={width} y2={zeroY} stroke="var(--border)" strokeWidth={0.5} />
      <path d={d} fill="none" stroke={color} strokeWidth={1.2} />
    </svg>
  )
}

// ─── Scatter Plot ─────────────────────────────────────────────────────────────

const SYMBOL_COLORS: Record<string, string> = {
  BTC: 'var(--accent)',
  ETH: '#818cf8',
  SOL: 'var(--green)',
  BNB: '#fde047',
  AVAX: '#f97316',
}

interface ScatterProps {
  points: ScatterPoint[]
  width?: number
  height?: number
}

const ScatterPlot: React.FC<ScatterProps> = ({ points, width = 300, height = 220 }) => {
  const pad = { l: 32, r: 12, t: 12, b: 28 }
  const w = width - pad.l - pad.r
  const h = height - pad.t - pad.b

  const allVals = [...points.map(p => p.predicted), ...points.map(p => p.actual)]
  const minV = Math.min(...allVals)
  const maxV = Math.max(...allVals)
  const rangeV = maxV - minV || 0.01

  const toX = (v: number) => pad.l + ((v - minV) / rangeV) * w
  const toY = (v: number) => pad.t + (1 - (v - minV) / rangeV) * h

  // Regression line
  const n = points.length
  const sumX = points.reduce((s, p) => s + p.predicted, 0)
  const sumY = points.reduce((s, p) => s + p.actual, 0)
  const sumXY = points.reduce((s, p) => s + p.predicted * p.actual, 0)
  const sumX2 = points.reduce((s, p) => s + p.predicted * p.predicted, 0)
  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX + 1e-10)
  const intercept = (sumY - slope * sumX) / n
  const rxMin = minV, rxMax = maxV
  const ryMin = slope * rxMin + intercept
  const ryMax = slope * rxMax + intercept

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height }}>
      {/* Axes */}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + h} stroke="var(--border)" strokeWidth={0.5} />
      <line x1={pad.l} y1={pad.t + h} x2={pad.l + w} y2={pad.t + h} stroke="var(--border)" strokeWidth={0.5} />
      {/* Zero reference */}
      <line x1={toX(0)} y1={pad.t} x2={toX(0)} y2={pad.t + h} stroke="var(--border)" strokeWidth={0.5} strokeDasharray="2,2" />
      <line x1={pad.l} y1={toY(0)} x2={pad.l + w} y2={toY(0)} stroke="var(--border)" strokeWidth={0.5} strokeDasharray="2,2" />
      {/* Regression line */}
      <line
        x1={toX(rxMin)} y1={toY(ryMin)}
        x2={toX(rxMax)} y2={toY(ryMax)}
        stroke="var(--accent)" strokeWidth={1} strokeDasharray="3,2" opacity={0.6}
      />
      {/* Points */}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={toX(p.predicted)}
          cy={toY(p.actual)}
          r={2.5}
          fill={SYMBOL_COLORS[p.symbol] ?? 'var(--text-muted)'}
          opacity={0.7}
        />
      ))}
      {/* Labels */}
      <text x={pad.l + w / 2} y={height - 4} textAnchor="middle" fontSize={7} fill="var(--text-muted)">Predicted Return</text>
      <text x={8} y={pad.t + h / 2} textAnchor="middle" fontSize={7} fill="var(--text-muted)"
        transform={`rotate(-90, 8, ${pad.t + h / 2})`}>Actual</text>
    </svg>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const MLSignals: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<ModelName>('Ensemble')

  const { data, isLoading } = useQuery<MLSignalData>({
    queryKey: ['ml-signals'],
    queryFn: fetchMLData,
    refetchInterval: 30_000,
  })

  if (isLoading || !data) return <LoadingSpinner message="Loading ML signal data…" />

  const ensemble = data.models.find(m => m.name === 'Ensemble')!
  const driftModels = data.models.filter(m => m.driftAlert)
  const maxImp = Math.max(...data.features.map(f => f.importance))
  const selectedModelData = data.models.find(m => m.name === selectedModel)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        <MetricCard label="Ensemble IC" value={ensemble.icMean.toFixed(4)} color="var(--green)" />
        <MetricCard label="IC IR" value={ensemble.icIr.toFixed(2)} color="var(--accent)" />
        <MetricCard label="Ensemble Sharpe" value={ensemble.sharpe.toFixed(2)} color="var(--blue)" />
        <MetricCard label="Hit Rate" value={`${(ensemble.hitRate * 100).toFixed(1)}%`} color="var(--green)" />
      </div>

      {/* Drift Alert */}
      {driftModels.length > 0 && (
        <div style={{
          padding: '12px 16px', borderRadius: 8,
          background: 'rgba(234,179,8,0.08)', border: '1px solid rgba(234,179,8,0.3)',
          fontSize: '0.8rem', color: 'var(--yellow)',
        }}>
          <strong>IC Drift Detected</strong> — models showing degraded IC over last 15 periods:{' '}
          {driftModels.map(m => m.name).join(', ')}. Consider retraining.
        </div>
      )}

      {/* Per-model performance table */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, overflow: 'hidden',
      }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
          MODEL PERFORMANCE
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Model', 'IC Mean', 'IC Std', 'IC IR', 'Sharpe', 'Hit Rate', 'Ens. Weight', 'IC Trend', 'Drift'].map(h => (
                <th key={h} style={{ padding: '6px 14px', textAlign: 'left', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.models.map((m, i) => {
              const icColor = m.icMean > 0.07 ? 'var(--green)' : m.icMean > 0.04 ? 'var(--yellow)' : 'var(--red)'
              return (
                <tr
                  key={m.name}
                  onClick={() => setSelectedModel(m.name)}
                  style={{
                    borderBottom: i < data.models.length - 1 ? '1px solid var(--border)' : undefined,
                    cursor: 'pointer',
                    background: selectedModel === m.name ? 'var(--bg-hover)' : undefined,
                  }}
                >
                  <td style={{ padding: '8px 14px', fontWeight: 700, fontSize: '0.85rem', color: 'var(--text-primary)' }}>
                    {m.name}
                    {m.name === 'Ensemble' && (
                      <span style={{ marginLeft: 6, fontSize: '0.62rem', color: 'var(--accent)', fontWeight: 600 }}>MASTER</span>
                    )}
                  </td>
                  <td style={{ padding: '8px 14px', fontSize: '0.8rem', color: icColor, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                    {m.icMean.toFixed(4)}
                  </td>
                  <td style={{ padding: '8px 14px', fontSize: '0.78rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                    {m.icStd.toFixed(4)}
                  </td>
                  <td style={{ padding: '8px 14px', fontSize: '0.78rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {m.icIr.toFixed(2)}
                  </td>
                  <td style={{ padding: '8px 14px', fontSize: '0.78rem', color: m.sharpe > 1.4 ? 'var(--green)' : 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {m.sharpe.toFixed(2)}
                  </td>
                  <td style={{ padding: '8px 14px', fontSize: '0.78rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {(m.hitRate * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '8px 14px', fontSize: '0.78rem', color: 'var(--accent)', fontVariantNumeric: 'tabular-nums' }}>
                    {(m.weight * 100).toFixed(0)}%
                  </td>
                  <td style={{ padding: '8px 14px' }}>
                    <ICSparkline data={m.icSeries} color={icColor} />
                  </td>
                  <td style={{ padding: '8px 14px' }}>
                    {m.driftAlert && (
                      <span style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--yellow)', padding: '2px 6px', borderRadius: 4, background: 'rgba(234,179,8,0.12)' }}>
                        DRIFT
                      </span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        {/* Ensemble Weights */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '16px',
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 14 }}>
            ENSEMBLE WEIGHTS
          </div>
          {data.models.filter(m => m.name !== 'Ensemble').map(m => {
            const totalW = data.models.filter(x => x.name !== 'Ensemble').reduce((s, x) => s + x.weight, 0)
            const pct = m.weight / totalW
            const color = m.name === 'LSTM' ? 'var(--green)' : m.name === 'Transformer' ? 'var(--blue)' : 'var(--yellow)'
            return (
              <div key={m.name} style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: '0.75rem', color, fontWeight: 600 }}>{m.name}</span>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {(pct * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ height: 8, borderRadius: 4, background: 'var(--bg-hover)', overflow: 'hidden' }}>
                  <div style={{ width: `${pct * 100}%`, height: '100%', background: color, transition: 'width 0.4s' }} />
                </div>
              </div>
            )
          })}
        </div>

        {/* Feature Importance */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '16px',
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 14 }}>
            XGBoost FEATURE IMPORTANCE (top 15)
          </div>
          {data.features.slice(0, 15).map((f, i) => (
            <div key={f.feature} style={{ marginBottom: 7 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ fontSize: '0.68rem', color: i < 3 ? 'var(--accent)' : 'var(--text-secondary)', fontFamily: 'monospace' }}>
                  {f.feature}
                </span>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                  {(f.importance * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{ height: 4, borderRadius: 2, background: 'var(--bg-hover)', overflow: 'hidden' }}>
                <div style={{
                  width: `${(f.importance / maxImp) * 100}%`,
                  height: '100%',
                  background: i < 3 ? 'var(--accent)' : 'var(--text-muted)',
                  transition: 'width 0.4s',
                }} />
              </div>
            </div>
          ))}
        </div>

        {/* Scatter plot */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '16px',
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10 }}>
            PREDICTED vs ACTUAL RETURNS
          </div>
          <ScatterPlot points={data.scatter} height={200} />
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 8 }}>
            {Object.entries(SYMBOL_COLORS).map(([sym, col]) => (
              <div key={sym} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                <div style={{ width: 7, height: 7, borderRadius: '50%', background: col }} />
                {sym}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* IC Time Series for selected model */}
      {selectedModelData && (
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '16px',
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10 }}>
            {selectedModelData.name} — IC TIME SERIES (last 60 periods)
            {selectedModelData.driftAlert && (
              <span style={{ marginLeft: 10, fontSize: '0.68rem', color: 'var(--yellow)', fontWeight: 700 }}>
                DRIFT ALERT: IC declining
              </span>
            )}
          </div>
          <ICSparkline data={selectedModelData.icSeries} width={800} height={60} color={selectedModelData.driftAlert ? 'var(--yellow)' : 'var(--green)'} />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            <span>60 periods ago</span>
            <span>Now</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default MLSignals
