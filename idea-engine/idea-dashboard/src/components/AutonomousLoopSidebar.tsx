import React, { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'

// ─── Types ────────────────────────────────────────────────────────────────────

type LoopState = 'RUNNING' | 'PAUSED' | 'ERROR'

interface LoopSummary {
  state: LoopState
  currentStep: number
  totalSteps: number
  lastApplied?: {
    paramName: string
    appliedAt: string
    impactPct: number
  }
  lastCycleDurationSec: number
  nextCycleInSec: number
  cycleCount: number
}

// ─── Mock ─────────────────────────────────────────────────────────────────────

const MOCK: LoopSummary = {
  state: 'RUNNING',
  currentStep: 3,
  totalSteps: 9,
  lastApplied: {
    paramName: 'signal.threshold',
    appliedAt: new Date(Date.now() - 7_200_000).toISOString(),
    impactPct: 1.5,
  },
  lastCycleDurationSec: 287,
  nextCycleInSec: 120,
  cycleCount: 142,
}

async function fetchLoopSummary(): Promise<LoopSummary> {
  try {
    const res = await fetch('/api/autonomous-loop/summary')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

const STATE_COLORS: Record<LoopState, string> = {
  RUNNING: 'var(--green)',
  PAUSED:  'var(--yellow)',
  ERROR:   'var(--red)',
}

const STEP_LABELS = ['Mine', 'Lit', 'Debate', 'BT', 'Val', 'Risk', 'Gate', 'Apply', 'Mon']

const AutonomousLoopSidebar: React.FC = () => {
  const [countdown, setCountdown] = useState(0)

  const { data } = useQuery<LoopSummary>({
    queryKey: ['autonomous-loop-summary'],
    queryFn: fetchLoopSummary,
    refetchInterval: 15_000,
  })

  useEffect(() => {
    if (data) setCountdown(data.nextCycleInSec)
  }, [data?.nextCycleInSec])

  useEffect(() => {
    const timer = setInterval(() => setCountdown(c => Math.max(0, c - 1)), 1000)
    return () => clearInterval(timer)
  }, [])

  if (!data) return null

  const stateColor = STATE_COLORS[data.state]

  return (
    <div style={{
      margin: '0 8px 12px',
      padding: '10px 12px',
      background: 'var(--bg-hover)',
      border: `1px solid ${stateColor}30`,
      borderRadius: 6,
      fontSize: '0.72rem',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
        <span style={{
          width: 7, height: 7, borderRadius: '50%',
          background: stateColor, flexShrink: 0,
          animation: data.state === 'RUNNING' ? 'pulse 1.5s infinite' : undefined,
        }} />
        <span style={{ color: stateColor, fontWeight: 700, fontSize: '0.68rem' }}>LOOP {data.state}</span>
        <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
          #{data.cycleCount}
        </span>
      </div>

      {/* Mini pipeline */}
      <div style={{ display: 'flex', gap: 2, marginBottom: 8 }}>
        {STEP_LABELS.map((lbl, i) => {
          const done = i < data.currentStep
          const active = i === data.currentStep
          const color = done ? 'var(--green)' : active ? stateColor : 'var(--border)'
          return (
            <div
              key={i}
              title={lbl}
              style={{
                flex: 1, height: 4, borderRadius: 2,
                background: color, opacity: active ? 1 : done ? 0.7 : 0.3,
                transition: 'background 0.3s',
              }}
            />
          )
        })}
      </div>
      <div style={{ color: 'var(--text-muted)', marginBottom: 8 }}>
        Step {data.currentStep + 1}/{data.totalSteps}: <span style={{ color: 'var(--text-secondary)' }}>{STEP_LABELS[data.currentStep]}</span>
      </div>

      {/* Last applied */}
      {data.lastApplied && (
        <div style={{
          padding: '6px 8px', borderRadius: 4,
          background: 'var(--bg-surface)', marginBottom: 8,
        }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.65rem', marginBottom: 2 }}>LAST APPLIED</div>
          <div style={{ color: 'var(--accent)', fontFamily: 'monospace', fontSize: '0.7rem', fontWeight: 600 }}>
            {data.lastApplied.paramName}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.62rem' }}>
              {new Date(data.lastApplied.appliedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
            <span style={{
              fontSize: '0.68rem', fontWeight: 700,
              color: data.lastApplied.impactPct >= 0 ? 'var(--green)' : 'var(--red)',
            }}>
              {data.lastApplied.impactPct >= 0 ? '+' : ''}{data.lastApplied.impactPct.toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {/* Loop health */}
      <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
        <span>Last cycle: <span style={{ color: 'var(--text-secondary)' }}>{data.lastCycleDurationSec}s</span></span>
        <span>Next: <span style={{ color: 'var(--accent)', fontVariantNumeric: 'tabular-nums' }}>
          {String(Math.floor(countdown / 60)).padStart(2, '0')}:{String(countdown % 60).padStart(2, '0')}
        </span></span>
      </div>
    </div>
  )
}

export default AutonomousLoopSidebar
