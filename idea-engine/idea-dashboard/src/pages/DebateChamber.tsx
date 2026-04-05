import React, { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import LoadingSpinner from '../components/LoadingSpinner'
import StatusBadge from '../components/StatusBadge'
import MetricCard from '../components/MetricCard'

// ─── Types ────────────────────────────────────────────────────────────────────

type VoteType = 'FOR' | 'AGAINST' | 'ABSTAIN'

interface AnalystVote {
  analystId: string
  analystName: string
  vote: VoteType
  confidence: number       // 0–1
  argument: string
  timestamp: string
  weight: number           // analyst credibility weight
}

interface DebateHypothesis {
  id: number
  description: string
  type: string
  source: string
  score: number
  createdAt: string
  votes: AnalystVote[]
  weightedFor: number      // sum of (weight * confidence) for FOR votes
  weightedAgainst: number
  weightedAbstain: number
  totalWeight: number
  status: 'pending' | 'approved' | 'rejected' | 'debating'
  debateStartedAt?: string
}

// ─── API helpers (mock until backend wired) ──────────────────────────────────

async function fetchDebateHypotheses(): Promise<DebateHypothesis[]> {
  try {
    const res = await fetch('/api/debate/hypotheses')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    // Dev fallback
    return MOCK_DEBATES
  }
}

async function submitOverride(id: number, decision: 'approved' | 'rejected'): Promise<void> {
  await fetch(`/api/debate/hypotheses/${id}/override`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision }),
  })
}

// ─── Mock data ────────────────────────────────────────────────────────────────

const MOCK_DEBATES: DebateHypothesis[] = [
  {
    id: 101,
    description: 'Increase BTC position size by 15% during RISK_ON regime when VIX < 18',
    type: 'position_sizing',
    source: 'genome',
    score: 0.78,
    createdAt: new Date(Date.now() - 3_600_000).toISOString(),
    status: 'debating',
    debateStartedAt: new Date(Date.now() - 1_800_000).toISOString(),
    weightedFor: 0.72, weightedAgainst: 0.18, weightedAbstain: 0.10, totalWeight: 1.0,
    votes: [
      { analystId: 'a1', analystName: 'Momentum Analyst', vote: 'FOR',     confidence: 0.85, weight: 0.30, timestamp: new Date(Date.now()-1200000).toISOString(), argument: 'Backtests show 22% IR improvement in RISK_ON regimes. The VIX<18 filter correctly reduces drawdown by 31%.' },
      { analystId: 'a2', analystName: 'Risk Manager',     vote: 'AGAINST', confidence: 0.60, weight: 0.25, timestamp: new Date(Date.now()-900000).toISOString(),  argument: 'Concentration risk increases. Max drawdown exceedance probability rises 8% in stress simulations.' },
      { analystId: 'a3', analystName: 'Stats Engine',     vote: 'FOR',     confidence: 0.90, weight: 0.35, timestamp: new Date(Date.now()-600000).toISOString(),  argument: 'Kelly criterion supports 15-18% sizing at current estimated edge (μ/σ² = 0.41). Half-Kelly = 20%.' },
      { analystId: 'a4', analystName: 'Macro Watcher',    vote: 'ABSTAIN', confidence: 0.50, weight: 0.10, timestamp: new Date(Date.now()-300000).toISOString(),  argument: 'Macro regime currently ambiguous. VIX at 19.2 — marginal. Recommend reassessing in 48h.' },
    ],
  },
  {
    id: 102,
    description: 'Exit all positions when bid-ask spread > 3× baseline; resume after 2-hour cooldown',
    type: 'risk_filter',
    source: 'academic',
    score: 0.84,
    createdAt: new Date(Date.now() - 7_200_000).toISOString(),
    status: 'pending',
    weightedFor: 0.81, weightedAgainst: 0.12, weightedAbstain: 0.07, totalWeight: 1.0,
    votes: [
      { analystId: 'a2', analystName: 'Risk Manager',     vote: 'FOR',     confidence: 0.92, weight: 0.25, timestamp: new Date(Date.now()-3600000).toISOString(), argument: 'Liquidity crises account for 38% of all 3σ loss events. This filter would have avoided 5 of the 7 worst drawdowns in backtest.' },
      { analystId: 'a3', analystName: 'Stats Engine',     vote: 'FOR',     confidence: 0.88, weight: 0.35, timestamp: new Date(Date.now()-3200000).toISOString(), argument: 'VPIN analysis confirms spread > 3× is highly predictive of adverse selection. P-value 0.003 on holdout.' },
      { analystId: 'a1', analystName: 'Momentum Analyst', vote: 'AGAINST', confidence: 0.45, weight: 0.30, timestamp: new Date(Date.now()-2800000).toISOString(), argument: 'Cooldown period too long. Missing 18% of profitable momentum continuation moves post-spread spike.' },
    ],
  },
]

// ─── Components ───────────────────────────────────────────────────────────────

const VOTE_COLORS: Record<VoteType, string> = {
  FOR:     'var(--green)',
  AGAINST: 'var(--red)',
  ABSTAIN: 'var(--text-muted)',
}

const VOTE_BG: Record<VoteType, string> = {
  FOR:     'rgba(34,197,94,0.10)',
  AGAINST: 'rgba(239,68,68,0.10)',
  ABSTAIN: 'rgba(148,163,184,0.08)',
}

interface VoteBarProps {
  for_: number
  against: number
  abstain: number
}

const VoteBar: React.FC<VoteBarProps> = ({ for_, against, abstain }) => {
  const total = for_ + against + abstain
  const pFor  = total > 0 ? (for_    / total) * 100 : 0
  const pAg   = total > 0 ? (against / total) * 100 : 0
  const pAb   = total > 0 ? (abstain / total) * 100 : 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', gap: 1 }}>
        <div style={{ width: `${pFor}%`,  background: 'var(--green)', transition: 'width 0.5s' }} />
        <div style={{ width: `${pAg}%`,  background: 'var(--red)',   transition: 'width 0.5s' }} />
        <div style={{ width: `${pAb}%`,  background: 'var(--text-muted)', opacity: 0.5, transition: 'width 0.5s' }} />
      </div>
      <div style={{ display: 'flex', gap: 12, fontSize: '0.7rem', color: 'var(--text-muted)' }}>
        <span style={{ color: 'var(--green)' }}>FOR {pFor.toFixed(0)}%</span>
        <span style={{ color: 'var(--red)'   }}>AGAINST {pAg.toFixed(0)}%</span>
        <span>ABSTAIN {pAb.toFixed(0)}%</span>
      </div>
    </div>
  )
}

interface ConfidenceBarProps {
  value: number  // 0–1
  vote: VoteType
}

const ConfidenceBar: React.FC<ConfidenceBarProps> = ({ value, vote }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
    <div style={{
      flex: 1, height: 4, borderRadius: 2,
      background: 'var(--bg-hover)', overflow: 'hidden',
    }}>
      <div style={{
        width: `${value * 100}%`,
        height: '100%',
        background: VOTE_COLORS[vote],
        transition: 'width 0.4s',
      }} />
    </div>
    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', width: 30, textAlign: 'right' }}>
      {(value * 100).toFixed(0)}%
    </span>
  </div>
)

interface DebateTranscriptProps {
  votes: AnalystVote[]
  animate: boolean
}

const DebateTranscript: React.FC<DebateTranscriptProps> = ({ votes, animate }) => {
  const [visible, setVisible] = useState(animate ? 1 : votes.length)

  useEffect(() => {
    if (!animate || visible >= votes.length) return
    const timer = setTimeout(() => setVisible((v) => v + 1), 900)
    return () => clearTimeout(timer)
  }, [visible, animate, votes.length])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {votes.slice(0, visible).map((v, i) => (
        <div
          key={v.analystId}
          style={{
            padding: '10px 12px',
            borderRadius: 6,
            background: VOTE_BG[v.vote],
            borderLeft: `3px solid ${VOTE_COLORS[v.vote]}`,
            animation: animate && i === visible - 1 ? 'fadeIn 0.4s ease' : undefined,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{
              fontWeight: 700, fontSize: '0.8rem', color: VOTE_COLORS[v.vote],
              textTransform: 'uppercase', letterSpacing: '0.04em',
            }}>
              {v.vote}
            </span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-primary)', fontWeight: 600 }}>
              {v.analystName}
            </span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
              w={v.weight.toFixed(2)}
            </span>
          </div>
          <ConfidenceBar value={v.confidence} vote={v.vote} />
          <p style={{
            margin: '6px 0 0',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
            lineHeight: 1.5,
          }}>
            {v.argument}
          </p>
        </div>
      ))}
      {animate && visible < votes.length && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', paddingLeft: 4 }}>
          <span style={{ animation: 'pulse 1s infinite' }}>● </span>
          Analyst deliberating…
        </div>
      )}
    </div>
  )
}

interface HypothesisDebateCardProps {
  hyp: DebateHypothesis
  onApprove: (id: number) => void
  onReject: (id: number) => void
  isPending: boolean
}

const HypothesisDebateCard: React.FC<HypothesisDebateCardProps> = ({
  hyp, onApprove, onReject, isPending,
}) => {
  const [expanded, setExpanded] = useState(false)
  const isDebating = hyp.status === 'debating'

  const verdict = hyp.weightedFor > hyp.weightedAgainst + 0.15
    ? 'LEAN_FOR'
    : hyp.weightedAgainst > hyp.weightedFor + 0.15
    ? 'LEAN_AGAINST'
    : 'UNDECIDED'

  const verdictColor = {
    LEAN_FOR:     'var(--green)',
    LEAN_AGAINST: 'var(--red)',
    UNDECIDED:    'var(--yellow)',
  }[verdict]

  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: `1px solid ${isDebating ? 'var(--accent)' : 'var(--border)'}`,
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div
        style={{
          padding: '14px 16px',
          cursor: 'pointer',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
        }}
        onClick={() => setExpanded((e) => !e)}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              {isDebating && (
                <span style={{
                  fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.06em',
                  background: 'var(--accent)', color: '#000', padding: '2px 6px',
                  borderRadius: 4, textTransform: 'uppercase',
                }}>
                  LIVE
                </span>
              )}
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                {hyp.type} · {hyp.source} · score {(hyp.score * 100).toFixed(0)}
              </span>
              <span style={{ marginLeft: 'auto', color: verdictColor, fontWeight: 700, fontSize: '0.75rem' }}>
                {verdict.replace('_', ' ')}
              </span>
            </div>
            <p style={{ margin: 0, fontSize: '0.875rem', color: 'var(--text-primary)', lineHeight: 1.4 }}>
              {hyp.description}
            </p>
          </div>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem', flexShrink: 0 }}>
            {expanded ? '▲' : '▼'}
          </span>
        </div>

        <VoteBar
          for_={hyp.weightedFor}
          against={hyp.weightedAgainst}
          abstain={hyp.weightedAbstain}
        />
      </div>

      {/* Expanded: transcript + controls */}
      {expanded && (
        <div style={{ padding: '0 16px 16px', borderTop: '1px solid var(--border)' }}>
          <div style={{ paddingTop: 12, marginBottom: 12 }}>
            <DebateTranscript votes={hyp.votes} animate={isDebating} />
          </div>

          {/* Human override */}
          {(hyp.status === 'pending' || hyp.status === 'debating') && (
            <div style={{
              display: 'flex', gap: 8, paddingTop: 12,
              borderTop: '1px solid var(--border)',
            }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', alignSelf: 'center', flex: 1 }}>
                Human override:
              </span>
              <button
                className="btn-icon"
                disabled={isPending}
                onClick={(e) => { e.stopPropagation(); onApprove(hyp.id) }}
                style={{
                  background: 'rgba(34,197,94,0.15)',
                  color: 'var(--green)',
                  padding: '5px 14px',
                  borderRadius: 5,
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  border: '1px solid rgba(34,197,94,0.3)',
                }}
              >
                ✓ Approve
              </button>
              <button
                className="btn-icon"
                disabled={isPending}
                onClick={(e) => { e.stopPropagation(); onReject(hyp.id) }}
                style={{
                  background: 'rgba(239,68,68,0.15)',
                  color: 'var(--red)',
                  padding: '5px 14px',
                  borderRadius: 5,
                  fontSize: '0.8rem',
                  fontWeight: 700,
                  border: '1px solid rgba(239,68,68,0.3)',
                }}
              >
                ✗ Reject
              </button>
            </div>
          )}

          {(hyp.status === 'approved' || hyp.status === 'rejected') && (
            <div style={{
              marginTop: 12, padding: '6px 12px',
              background: hyp.status === 'approved' ? 'rgba(34,197,94,0.10)' : 'rgba(239,68,68,0.10)',
              borderRadius: 5,
              fontSize: '0.8rem',
              color: hyp.status === 'approved' ? 'var(--green)' : 'var(--red)',
              fontWeight: 700,
            }}>
              {hyp.status === 'approved' ? '✓ APPROVED' : '✗ REJECTED'} (human override)
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const DebateChamber: React.FC = () => {
  const qc = useQueryClient()

  const { data: hypotheses = [], isLoading } = useQuery({
    queryKey: ['debate', 'hypotheses'],
    queryFn: fetchDebateHypotheses,
    refetchInterval: 10_000,
  })

  const override = useMutation({
    mutationFn: ({ id, decision }: { id: number; decision: 'approved' | 'rejected' }) =>
      submitOverride(id, decision),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['debate', 'hypotheses'] }),
  })

  const [filter, setFilter] = useState<'all' | 'pending' | 'debating' | 'decided'>('all')

  const filtered = hypotheses.filter((h) => {
    if (filter === 'all')      return true
    if (filter === 'debating') return h.status === 'debating'
    if (filter === 'pending')  return h.status === 'pending'
    return h.status === 'approved' || h.status === 'rejected'
  })

  const nLive    = hypotheses.filter((h) => h.status === 'debating').length
  const nPending = hypotheses.filter((h) => h.status === 'pending').length

  if (isLoading) return <LoadingSpinner message="Loading debate chamber…" />

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        <MetricCard label="Total in queue" value={hypotheses.length} />
        <MetricCard label="Live debates" value={nLive} color="var(--accent)" />
        <MetricCard label="Pending human review" value={nPending} color="var(--yellow)" />
        <MetricCard
          label="Consensus rate"
          value={`${(hypotheses.length > 0 ? hypotheses.filter(h => {
            const netFor = h.weightedFor - h.weightedAgainst
            return Math.abs(netFor) > 0.2
          }).length / hypotheses.length * 100 : 0).toFixed(0)}%`}
        />
      </div>

      {/* Filter tabs */}
      <div style={{ display: 'flex', gap: 6 }}>
        {(['all', 'debating', 'pending', 'decided'] as const).map((f) => (
          <button
            key={f}
            className="btn-icon"
            onClick={() => setFilter(f)}
            style={{
              padding: '5px 14px',
              borderRadius: 6,
              fontSize: '0.8rem',
              background: filter === f ? 'var(--accent)' : 'var(--bg-hover)',
              color: filter === f ? '#000' : 'var(--text-muted)',
              fontWeight: filter === f ? 700 : 400,
              textTransform: 'capitalize',
              border: 'none',
            }}
          >
            {f}
          </button>
        ))}
        <span style={{ marginLeft: 'auto', fontSize: '0.75rem', color: 'var(--text-muted)', alignSelf: 'center' }}>
          {filtered.length} hypothesis{filtered.length !== 1 ? 'es' : ''}
        </span>
      </div>

      {/* Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {filtered.length === 0 && (
          <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
            No hypotheses in this category
          </div>
        )}
        {filtered.map((h) => (
          <HypothesisDebateCard
            key={h.id}
            hyp={h}
            onApprove={(id) => override.mutate({ id, decision: 'approved' })}
            onReject={(id)  => override.mutate({ id, decision: 'rejected' })}
            isPending={override.isPending}
          />
        ))}
      </div>
    </div>
  )
}

export default DebateChamber
