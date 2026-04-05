import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from '../components/LoadingSpinner'
import MetricCard from '../components/MetricCard'

// ─── Types ────────────────────────────────────────────────────────────────────

interface ParamSnapshot {
  id: string
  timestamp: string
  label: string
  changedBy: string        // 'LOOP' | 'MANUAL' | 'ROLLBACK'
  params: Record<string, number | string>
  sharpe?: number
  totalReturn?: number
}

interface SensitivityEntry {
  param: string
  avgImpact: number        // % change in Sharpe when this param was changed
  changeCount: number
}

interface ParamHistoryData {
  snapshots: ParamSnapshot[]
  currentParams: Record<string, number | string>
  baselineParams: Record<string, number | string>
  bestParams: Record<string, number | string>
  sensitivity: SensitivityEntry[]
}

// ─── Mock ─────────────────────────────────────────────────────────────────────

const BASE_PARAMS: Record<string, number | string> = {
  'momentum.lookback':          14,
  'momentum.signalThreshold':   1.5,
  'risk.maxDrawdown':           0.08,
  'risk.maxPosition':           0.15,
  'ensemble.lstmWeight':        0.25,
  'ensemble.xgbWeight':         0.35,
  'ensemble.transformerWeight': 0.20,
  'volatility.halflife':        20,
  'portfolio.maxConcentration': 0.30,
  'execution.urgency':          'NORMAL',
}

function perturbParams(base: Record<string, number | string>, seed: number): Record<string, number | string> {
  const out: Record<string, number | string> = {}
  Object.entries(base).forEach(([k, v]) => {
    if (typeof v === 'number') {
      out[k] = Math.round((v * (1 + (Math.sin(seed * k.length) * 0.12))) * 100) / 100
    } else {
      out[k] = v
    }
  })
  return out
}

const MOCK_SNAPSHOTS: ParamSnapshot[] = Array.from({ length: 18 }, (_, i) => ({
  id: `snap-${i}`,
  timestamp: new Date(Date.now() - i * 86_400_000 * 0.6).toISOString(),
  label: i === 0 ? 'Current' : i === 17 ? 'Baseline' : `Version ${18 - i}`,
  changedBy: i % 3 === 0 ? 'MANUAL' : 'LOOP',
  params: i === 17 ? BASE_PARAMS : perturbParams(BASE_PARAMS, i * 7),
  sharpe: 1.12 + (18 - i) * 0.02 + Math.sin(i) * 0.1,
  totalReturn: 18.4 + (18 - i) * 0.3 + Math.sin(i * 2) * 1.2,
}))

const MOCK: ParamHistoryData = {
  snapshots: MOCK_SNAPSHOTS,
  currentParams: MOCK_SNAPSHOTS[0].params,
  baselineParams: BASE_PARAMS,
  bestParams: perturbParams(BASE_PARAMS, 3),
  sensitivity: [
    { param: 'signal.threshold',          avgImpact: 0.18, changeCount: 6 },
    { param: 'momentum.lookback',          avgImpact: 0.14, changeCount: 8 },
    { param: 'risk.maxDrawdown',           avgImpact: 0.12, changeCount: 5 },
    { param: 'ensemble.lstmWeight',        avgImpact: 0.08, changeCount: 4 },
    { param: 'volatility.halflife',        avgImpact: 0.07, changeCount: 7 },
    { param: 'portfolio.maxConcentration', avgImpact: 0.05, changeCount: 3 },
  ],
}

async function fetchParamHistory(): Promise<ParamHistoryData> {
  try {
    const res = await fetch('/api/parameter-history')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK
  }
}

// ─── Diff View ────────────────────────────────────────────────────────────────

interface DiffRowProps {
  param: string
  valA: number | string
  valB: number | string
  labelA: string
  labelB: string
}

const DiffRow: React.FC<DiffRowProps> = ({ param, valA, valB, labelA, labelB }) => {
  const changed = String(valA) !== String(valB)
  if (!changed) return null
  const numA = typeof valA === 'number' ? valA : NaN
  const numB = typeof valB === 'number' ? valB : NaN
  const pctChange = !isNaN(numA) && !isNaN(numB) && numA !== 0
    ? ((numB - numA) / numA) * 100
    : NaN

  return (
    <tr style={{ borderBottom: '1px solid var(--border)', background: 'rgba(59,130,246,0.04)' }}>
      <td style={{ padding: '7px 14px', fontSize: '0.78rem', fontFamily: 'monospace', color: 'var(--accent)' }}>
        {param}
      </td>
      <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
        {String(valA)}
        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginLeft: 4 }}>({labelA})</span>
      </td>
      <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--accent)', fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
        {String(valB)}
        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginLeft: 4 }}>({labelB})</span>
      </td>
      <td style={{ padding: '7px 14px', fontSize: '0.72rem', fontVariantNumeric: 'tabular-nums', color: pctChange > 0 ? 'var(--green)' : 'var(--red)' }}>
        {!isNaN(pctChange) ? `${pctChange >= 0 ? '+' : ''}${pctChange.toFixed(1)}%` : '—'}
      </td>
    </tr>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const ParameterHistory: React.FC = () => {
  const [compareIdA, setCompareIdA] = useState<string>('')
  const [compareIdB, setCompareIdB] = useState<string>('')
  const [rollbackConfirm, setRollbackConfirm] = useState<string | null>(null)

  const { data, isLoading } = useQuery<ParamHistoryData>({
    queryKey: ['param-history'],
    queryFn: fetchParamHistory,
  })

  if (isLoading || !data) return <LoadingSpinner message="Loading parameter history…" />

  const snapA = data.snapshots.find(s => s.id === compareIdA)
  const snapB = data.snapshots.find(s => s.id === compareIdB)

  const allParamKeys = Array.from(new Set([
    ...Object.keys(data.currentParams),
    ...Object.keys(data.baselineParams),
    ...Object.keys(data.bestParams),
  ])).sort()

  const maxSensImpact = Math.max(...data.sensitivity.map(s => s.avgImpact), 0.01)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
        <MetricCard label="Total Snapshots" value={String(data.snapshots.length)} />
        <MetricCard label="Loop-applied Changes" value={String(data.snapshots.filter(s => s.changedBy === 'LOOP').length)} color="var(--accent)" />
        <MetricCard label="Manual Overrides" value={String(data.snapshots.filter(s => s.changedBy === 'MANUAL').length)} color="var(--yellow)" />
      </div>

      {/* Current vs Baseline vs Best */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, overflow: 'hidden',
      }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
          CURRENT vs BASELINE vs ALL-TIME BEST
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Parameter', 'Current', 'Baseline', 'All-Time Best'].map(h => (
                <th key={h} style={{ padding: '7px 14px', textAlign: 'left', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {allParamKeys.map((k, i) => {
              const cur = data.currentParams[k]
              const base = data.baselineParams[k]
              const best = data.bestParams[k]
              const curChanged = String(cur) !== String(base)
              return (
                <tr key={k} style={{ borderBottom: i < allParamKeys.length - 1 ? '1px solid var(--border)' : undefined }}>
                  <td style={{ padding: '7px 14px', fontSize: '0.78rem', fontFamily: 'monospace', color: 'var(--accent)' }}>
                    {k}
                  </td>
                  <td style={{
                    padding: '7px 14px', fontSize: '0.78rem', fontVariantNumeric: 'tabular-nums',
                    color: curChanged ? 'var(--text-primary)' : 'var(--text-muted)',
                    fontWeight: curChanged ? 600 : 400,
                  }}>
                    {String(cur)}
                  </td>
                  <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                    {String(base)}
                  </td>
                  <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--green)', fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>
                    {String(best)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Parameter Comparison (diff) */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, overflow: 'hidden',
      }}>
        <div style={{
          padding: '10px 16px', borderBottom: '1px solid var(--border)',
          display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
        }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>PARAMETER DIFF</span>
          <select
            value={compareIdA}
            onChange={e => setCompareIdA(e.target.value)}
            style={{
              background: 'var(--bg-hover)', border: '1px solid var(--border)', borderRadius: 4,
              color: 'var(--text-primary)', fontSize: '0.75rem', padding: '4px 8px',
            }}
          >
            <option value="">— Select version A —</option>
            {data.snapshots.map(s => (
              <option key={s.id} value={s.id}>
                {s.label} ({new Date(s.timestamp).toLocaleDateString()})
              </option>
            ))}
          </select>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>vs</span>
          <select
            value={compareIdB}
            onChange={e => setCompareIdB(e.target.value)}
            style={{
              background: 'var(--bg-hover)', border: '1px solid var(--border)', borderRadius: 4,
              color: 'var(--text-primary)', fontSize: '0.75rem', padding: '4px 8px',
            }}
          >
            <option value="">— Select version B —</option>
            {data.snapshots.map(s => (
              <option key={s.id} value={s.id}>
                {s.label} ({new Date(s.timestamp).toLocaleDateString()})
              </option>
            ))}
          </select>
        </div>
        {snapA && snapB ? (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)' }}>
                {['Parameter', `Version A: ${snapA.label}`, `Version B: ${snapB.label}`, 'Change'].map(h => (
                  <th key={h} style={{ padding: '6px 14px', textAlign: 'left', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {allParamKeys.map(k => (
                <DiffRow
                  key={k}
                  param={k}
                  valA={snapA.params[k] ?? '—'}
                  valB={snapB.params[k] ?? '—'}
                  labelA={snapA.label}
                  labelB={snapB.label}
                />
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            Select two versions above to see differences
          </div>
        )}
      </div>

      {/* Sensitivity */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '16px',
      }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 14 }}>
          PARAMETER SENSITIVITY (avg Sharpe impact per change)
        </div>
        {data.sensitivity.map((s) => (
          <div key={s.param} style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: 'var(--accent)' }}>{s.param}</span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                {s.changeCount} changes · <span style={{ color: 'var(--green)', fontWeight: 600 }}>+{(s.avgImpact * 100).toFixed(1)}% avg Sharpe</span>
              </span>
            </div>
            <div style={{ height: 7, borderRadius: 3, background: 'var(--bg-hover)', overflow: 'hidden' }}>
              <div style={{
                width: `${(s.avgImpact / maxSensImpact) * 100}%`,
                height: '100%', background: 'var(--accent)', transition: 'width 0.4s',
              }} />
            </div>
          </div>
        ))}
      </div>

      {/* Timeline + Rollback */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, overflow: 'hidden',
      }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
          VERSION TIMELINE
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Version', 'Date', 'Changed By', 'Sharpe', 'Total Return', ''].map(h => (
                <th key={h} style={{ padding: '6px 14px', textAlign: 'left', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.snapshots.map((s, i) => {
              const changedByColor = s.changedBy === 'LOOP' ? 'var(--accent)' : s.changedBy === 'ROLLBACK' ? 'var(--yellow)' : 'var(--blue)'
              return (
                <tr key={s.id} style={{ borderBottom: i < data.snapshots.length - 1 ? '1px solid var(--border)' : undefined }}>
                  <td style={{ padding: '7px 14px', fontWeight: i === 0 ? 700 : 400, fontSize: '0.8rem', color: 'var(--text-primary)' }}>
                    {s.label}
                    {i === 0 && <span style={{ marginLeft: 6, fontSize: '0.62rem', color: 'var(--green)', fontWeight: 700 }}>CURRENT</span>}
                  </td>
                  <td style={{ padding: '7px 14px', fontSize: '0.72rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                    {new Date(s.timestamp).toLocaleString()}
                  </td>
                  <td style={{ padding: '7px 14px' }}>
                    <span style={{
                      fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: 4,
                      background: `${changedByColor}18`, color: changedByColor,
                    }}>
                      {s.changedBy}
                    </span>
                  </td>
                  <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {s.sharpe?.toFixed(3) ?? '—'}
                  </td>
                  <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: s.totalReturn && s.totalReturn > 0 ? 'var(--green)' : 'var(--red)', fontVariantNumeric: 'tabular-nums' }}>
                    {s.totalReturn != null ? `${s.totalReturn >= 0 ? '+' : ''}${s.totalReturn.toFixed(1)}%` : '—'}
                  </td>
                  <td style={{ padding: '7px 14px' }}>
                    {i > 0 && (
                      rollbackConfirm === s.id ? (
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <span style={{ fontSize: '0.65rem', color: 'var(--yellow)' }}>Confirm?</span>
                          <button
                            onClick={() => {
                              // POST /api/parameter-history/rollback { id: s.id }
                              setRollbackConfirm(null)
                            }}
                            style={{
                              padding: '2px 8px', borderRadius: 4, border: 'none',
                              background: 'var(--yellow)', color: '#000',
                              fontSize: '0.65rem', fontWeight: 700, cursor: 'pointer',
                            }}
                          >
                            Yes
                          </button>
                          <button
                            onClick={() => setRollbackConfirm(null)}
                            style={{
                              padding: '2px 8px', borderRadius: 4,
                              border: '1px solid var(--border)', background: 'transparent',
                              color: 'var(--text-muted)', fontSize: '0.65rem', cursor: 'pointer',
                            }}
                          >
                            No
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setRollbackConfirm(s.id)}
                          style={{
                            padding: '3px 10px', borderRadius: 4,
                            border: '1px solid var(--border)', background: 'transparent',
                            color: 'var(--text-muted)', fontSize: '0.68rem', cursor: 'pointer',
                          }}
                        >
                          Rollback
                        </button>
                      )
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default ParameterHistory
