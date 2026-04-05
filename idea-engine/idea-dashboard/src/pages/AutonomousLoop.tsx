import React, { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from '../components/LoadingSpinner'
import MetricCard from '../components/MetricCard'

// ─── Types ────────────────────────────────────────────────────────────────────

type LoopState = 'RUNNING' | 'PAUSED' | 'ERROR'

type HypStage = 'MINED' | 'DEBATE' | 'BACKTEST' | 'VALIDATED' | 'APPLIED'

const PIPELINE_STEPS = [
  'Mine Hypotheses',
  'Literature Check',
  'Debate Chamber',
  'Backtest',
  'Statistical Validation',
  'Risk Filter',
  'Approval Gate',
  'Apply Parameters',
  'Monitor Impact',
]

interface CycleRecord {
  id: string
  startedAt: string
  durationSec: number
  hypothesesFound: number
  changesApplied: number
  status: 'OK' | 'ERROR' | 'PARTIAL'
}

interface QueueCounts {
  MINED: number
  DEBATE: number
  BACKTEST: number
  VALIDATED: number
  APPLIED: number
}

interface AppliedChange {
  id: string
  appliedAt: string
  paramName: string
  oldValue: number | string
  newValue: number | string
  hypothesis: string
  impactPct: number
}

interface LoopStatus {
  state: LoopState
  currentStep: number      // 0-8
  cycleCount: number
  totalChanges: number
  netImpactPct: number
  nextCycleInSec: number
  cycles: CycleRecord[]
  queue: QueueCounts
  applied: AppliedChange[]
  errorMessage?: string
}

// ─── Mock ─────────────────────────────────────────────────────────────────────

const MOCK: LoopStatus = {
  state: 'RUNNING',
  currentStep: 3,
  cycleCount: 142,
  totalChanges: 38,
  netImpactPct: 4.72,
  nextCycleInSec: 180,
  errorMessage: undefined,
  queue: { MINED: 12, DEBATE: 5, BACKTEST: 3, VALIDATED: 2, APPLIED: 38 },
  cycles: Array.from({ length: 20 }, (_, i) => ({
    id: `c${i}`,
    startedAt: new Date(Date.now() - i * 600_000).toISOString(),
    durationSec: 180 + Math.floor(Math.random() * 120),
    hypothesesFound: Math.floor(Math.random() * 8) + 1,
    changesApplied: Math.random() > 0.7 ? Math.floor(Math.random() * 3) + 1 : 0,
    status: i === 2 ? 'ERROR' : Math.random() > 0.9 ? 'PARTIAL' : 'OK',
  })),
  applied: [
    { id: 'a1', appliedAt: new Date(Date.now() - 86_400_000 * 0.5).toISOString(), paramName: 'momentum.lookback', oldValue: 14, newValue: 18, hypothesis: 'Longer lookback reduces noise in BTC', impactPct: 1.2 },
    { id: 'a2', appliedAt: new Date(Date.now() - 86_400_000 * 1.2).toISOString(), paramName: 'risk.maxDrawdown', oldValue: 0.08, newValue: 0.06, hypothesis: 'Tighter drawdown improves Sharpe', impactPct: 0.8 },
    { id: 'a3', appliedAt: new Date(Date.now() - 86_400_000 * 2.1).toISOString(), paramName: 'signal.threshold', oldValue: 1.5, newValue: 1.8, hypothesis: 'Higher threshold filters false breakouts', impactPct: 1.5 },
    { id: 'a4', appliedAt: new Date(Date.now() - 86_400_000 * 3.4).toISOString(), paramName: 'ensemble.lstmWeight', oldValue: 0.25, newValue: 0.35, hypothesis: 'LSTM outperformed in ranging markets', impactPct: -0.3 },
    { id: 'a5', appliedAt: new Date(Date.now() - 86_400_000 * 4.8).toISOString(), paramName: 'portfolio.maxConcentration', oldValue: 0.30, newValue: 0.25, hypothesis: 'Diversification improves risk-adjusted return', impactPct: 0.5 },
    { id: 'a6', appliedAt: new Date(Date.now() - 86_400_000 * 6.2).toISOString(), paramName: 'volatility.halflife', oldValue: 20, newValue: 15, hypothesis: 'Faster vol decay captures regime shifts', impactPct: 1.0 },
  ],
}

async function fetchLoopStatus(): Promise<LoopStatus> {
  try {
    const res = await fetch('/api/autonomous-loop/status')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return { ...MOCK, nextCycleInSec: MOCK.nextCycleInSec - 1 }
  }
}

// ─── Sub-components ───────────────────────────────────────────────────────────

interface PipelineProgressProps {
  currentStep: number
  state: LoopState
}

const PipelineProgress: React.FC<PipelineProgressProps> = ({ currentStep, state }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
    {PIPELINE_STEPS.map((step, i) => {
      const done = i < currentStep
      const active = i === currentStep
      const isErr = state === 'ERROR' && active
      const color = isErr ? 'var(--red)' : done ? 'var(--green)' : active ? 'var(--accent)' : 'var(--border)'
      const textColor = done || active ? 'var(--text-primary)' : 'var(--text-muted)'

      return (
        <React.Fragment key={i}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, flex: 1, minWidth: 0 }}>
            <div style={{
              width: 28, height: 28, borderRadius: '50%',
              background: done ? 'var(--green)' : active ? (isErr ? 'var(--red)' : 'var(--accent)') : 'var(--bg-hover)',
              border: `2px solid ${color}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.65rem', fontWeight: 700, color: (done || active) ? '#000' : 'var(--text-muted)',
              flexShrink: 0,
              animation: active && !isErr ? 'pulse 1.5s infinite' : undefined,
            }}>
              {done ? '✓' : i + 1}
            </div>
            <span style={{
              fontSize: '0.58rem', color: textColor, textAlign: 'center',
              lineHeight: 1.2, maxWidth: 70, overflow: 'hidden',
            }}>
              {step}
            </span>
          </div>
          {i < PIPELINE_STEPS.length - 1 && (
            <div style={{
              height: 2, flex: 0.3, background: i < currentStep ? 'var(--green)' : 'var(--border)',
              marginBottom: 18, flexShrink: 0,
            }} />
          )}
        </React.Fragment>
      )
    })}
  </div>
)

const STAGE_COLORS: Record<HypStage, string> = {
  MINED:     'var(--text-muted)',
  DEBATE:    'var(--yellow)',
  BACKTEST:  'var(--blue)',
  VALIDATED: 'var(--accent)',
  APPLIED:   'var(--green)',
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const AutonomousLoop: React.FC = () => {
  const [countdown, setCountdown] = useState(0)

  const { data, isLoading } = useQuery<LoopStatus>({
    queryKey: ['autonomous-loop'],
    queryFn: fetchLoopStatus,
    refetchInterval: 15_000,
  })

  useEffect(() => {
    if (!data) return
    setCountdown(data.nextCycleInSec)
  }, [data?.nextCycleInSec])

  useEffect(() => {
    const timer = setInterval(() => setCountdown(c => Math.max(0, c - 1)), 1000)
    return () => clearInterval(timer)
  }, [])

  if (isLoading || !data) return <LoadingSpinner message="Loading autonomous loop…" />

  const stateColors: Record<LoopState, string> = {
    RUNNING: 'var(--green)',
    PAUSED:  'var(--yellow)',
    ERROR:   'var(--red)',
  }
  const stateColor = stateColors[data.state]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Status header */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        <div style={{
          background: 'var(--bg-surface)', border: `1px solid ${stateColor}40`,
          borderRadius: 8, padding: '14px 18px',
          display: 'flex', alignItems: 'center', gap: 12,
        }}>
          <span style={{
            width: 12, height: 12, borderRadius: '50%',
            background: stateColor,
            boxShadow: `0 0 8px ${stateColor}`,
            animation: data.state === 'RUNNING' ? 'pulse 1.5s infinite' : undefined,
            flexShrink: 0,
          }} />
          <div>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>LOOP STATE</div>
            <div style={{ fontSize: '1rem', fontWeight: 800, color: stateColor }}>{data.state}</div>
            {data.errorMessage && (
              <div style={{ fontSize: '0.65rem', color: 'var(--red)', marginTop: 2 }}>{data.errorMessage}</div>
            )}
          </div>
        </div>
        <MetricCard label="Cycles Completed" value={String(data.cycleCount)} color="var(--accent)" />
        <MetricCard label="Total Changes Applied" value={String(data.totalChanges)} color="var(--blue)" />
        <MetricCard
          label="Net Loop Impact"
          value={`${data.netImpactPct >= 0 ? '+' : ''}${data.netImpactPct.toFixed(2)}%`}
          color={data.netImpactPct >= 0 ? 'var(--green)' : 'var(--red)'}
        />
      </div>

      {/* Pipeline progress */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '20px',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20,
        }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            CURRENT CYCLE — STEP {data.currentStep + 1} / {PIPELINE_STEPS.length}
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Next cycle in{' '}
            <strong style={{ color: 'var(--accent)', fontVariantNumeric: 'tabular-nums' }}>
              {String(Math.floor(countdown / 60)).padStart(2, '0')}:{String(countdown % 60).padStart(2, '0')}
            </strong>
          </span>
        </div>
        <PipelineProgress currentStep={data.currentStep} state={data.state} />
      </div>

      {/* Queue + Applied changes */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16 }}>

        {/* Queue visualization */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '16px',
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 14 }}>
            HYPOTHESIS QUEUE
          </div>
          {(Object.keys(data.queue) as HypStage[]).map((stage) => {
            const count = data.queue[stage]
            const color = STAGE_COLORS[stage]
            const maxCount = Math.max(...Object.values(data.queue), 1)
            return (
              <div key={stage} style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: '0.72rem', color, fontWeight: 600 }}>{stage}</span>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {count}
                  </span>
                </div>
                <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-hover)', overflow: 'hidden' }}>
                  <div style={{
                    width: `${(count / maxCount) * 100}%`, height: '100%',
                    background: color, transition: 'width 0.4s',
                  }} />
                </div>
              </div>
            )
          })}
        </div>

        {/* Applied changes timeline */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, overflow: 'hidden',
        }}>
          <div style={{
            padding: '10px 16px', borderBottom: '1px solid var(--border)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
              APPLIED CHANGES TIMELINE
            </span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Loop has made {data.totalChanges} changes, net impact:{' '}
              <strong style={{ color: data.netImpactPct >= 0 ? 'var(--green)' : 'var(--red)' }}>
                {data.netImpactPct >= 0 ? '+' : ''}{data.netImpactPct.toFixed(2)}%
              </strong>
            </span>
          </div>
          <div style={{ overflowY: 'auto', maxHeight: 280 }}>
            {data.applied.map((ch, i) => (
              <div key={ch.id} style={{
                padding: '10px 16px',
                borderBottom: i < data.applied.length - 1 ? '1px solid var(--border)' : undefined,
                display: 'flex', alignItems: 'flex-start', gap: 12,
              }}>
                <div style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: ch.impactPct >= 0 ? 'var(--green)' : 'var(--red)',
                  marginTop: 5, flexShrink: 0,
                }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--accent)', fontWeight: 700, fontFamily: 'monospace' }}>
                      {ch.paramName}
                    </span>
                    <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                      {new Date(ch.appliedAt).toLocaleDateString()}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>
                    <span style={{ color: 'var(--red)' }}>{ch.oldValue}</span>
                    {' → '}
                    <span style={{ color: 'var(--green)' }}>{ch.newValue}</span>
                    {' · '}
                    <span style={{ color: 'var(--text-secondary)', fontStyle: 'italic' }}>{ch.hypothesis}</span>
                  </div>
                </div>
                <div style={{
                  fontSize: '0.75rem', fontWeight: 700, flexShrink: 0,
                  color: ch.impactPct >= 0 ? 'var(--green)' : 'var(--red)',
                }}>
                  {ch.impactPct >= 0 ? '+' : ''}{ch.impactPct.toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Cycle history */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, overflow: 'hidden',
      }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
          CYCLE HISTORY (LAST 20)
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Cycle', 'Started', 'Duration', 'Hypotheses Found', 'Changes Applied', 'Status'].map(h => (
                <th key={h} style={{ padding: '6px 14px', textAlign: 'left', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.cycles.map((c, i) => {
              const statusColor = c.status === 'OK' ? 'var(--green)' : c.status === 'ERROR' ? 'var(--red)' : 'var(--yellow)'
              return (
                <tr key={c.id} style={{ borderBottom: i < data.cycles.length - 1 ? '1px solid var(--border)' : undefined }}>
                  <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                    #{data.cycleCount - i}
                  </td>
                  <td style={{ padding: '7px 14px', fontSize: '0.72rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                    {new Date(c.startedAt).toLocaleTimeString()}
                  </td>
                  <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {c.durationSec}s
                  </td>
                  <td style={{ padding: '7px 14px', fontSize: '0.78rem', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {c.hypothesesFound}
                  </td>
                  <td style={{ padding: '7px 14px', fontSize: '0.78rem', fontVariantNumeric: 'tabular-nums', color: c.changesApplied > 0 ? 'var(--accent)' : 'var(--text-muted)' }}>
                    {c.changesApplied}
                  </td>
                  <td style={{ padding: '7px 14px' }}>
                    <span style={{
                      fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: 4,
                      background: `${statusColor}18`, color: statusColor,
                    }}>
                      {c.status}
                    </span>
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

export default AutonomousLoop
