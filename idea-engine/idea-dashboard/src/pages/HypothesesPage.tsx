import React, { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { formatDistanceToNow } from 'date-fns'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import { fetchHypotheses, submitHypothesisTest } from '../api/client'
import type { Hypothesis, HypothesisStatus } from '../types'

const STATUS_TABS: { value: HypothesisStatus | 'all'; label: string }[] = [
  { value: 'all',      label: 'All' },
  { value: 'pending',  label: 'Pending' },
  { value: 'testing',  label: 'Testing' },
  { value: 'adopted',  label: 'Adopted' },
  { value: 'rejected', label: 'Rejected' },
]

const KANBAN_STATUSES: HypothesisStatus[] = ['pending', 'testing', 'adopted', 'rejected']

const KANBAN_COLORS: Record<HypothesisStatus, string> = {
  pending:  'var(--yellow)',
  testing:  'var(--blue)',
  adopted:  'var(--green)',
  rejected: 'var(--red)',
}

// ─── Hypothesis Card ──────────────────────────────────────────────────────────

interface HypothesisCardProps {
  hypothesis: Hypothesis
  onTest: (id: number) => void
  testingId: number | null
}

const HypothesisCard: React.FC<HypothesisCardProps> = ({
  hypothesis: h,
  onTest,
  testingId,
}) => {
  const [expanded, setExpanded] = useState(false)
  const isSubmitting = testingId === h.id

  const scoreColor =
    h.score >= 0.8
      ? 'var(--green)'
      : h.score >= 0.6
      ? 'var(--accent)'
      : h.score >= 0.4
      ? 'var(--yellow)'
      : 'var(--red)'

  return (
    <div
      className="card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        transition: 'border-color var(--transition)',
        cursor: 'pointer',
        borderLeft: `3px solid ${KANBAN_COLORS[h.status]}`,
      }}
      onClick={() => setExpanded((v) => !v)}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <StatusBadge value={h.status} size="sm" />
          <StatusBadge value={h.type} size="sm" />
          <StatusBadge value={h.source} size="sm" />
        </div>
        <span
          className="num"
          style={{ color: 'var(--text-muted)', fontSize: '0.75rem', flexShrink: 0 }}
        >
          #{h.id}
        </span>
      </div>

      {/* Description */}
      <p
        style={{
          fontSize: '0.875rem',
          color: 'var(--text-primary)',
          lineHeight: 1.5,
          margin: 0,
        }}
      >
        {h.description}
      </p>

      {/* Score Bar */}
      <div>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.75rem',
            color: 'var(--text-muted)',
            marginBottom: 4,
          }}
        >
          <span>Confidence Score</span>
          <span className="num" style={{ color: scoreColor, fontWeight: 700 }}>
            {(h.score * 100).toFixed(0)}%
          </span>
        </div>
        <div
          style={{
            height: 4,
            background: 'var(--bg-elevated)',
            borderRadius: 2,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${h.score * 100}%`,
              background: scoreColor,
              transition: 'width 0.4s ease',
            }}
          />
        </div>
      </div>

      {/* Expanded: Params + Test Results */}
      {expanded && (
        <div
          style={{
            background: 'var(--bg-elevated)',
            borderRadius: 'var(--radius)',
            padding: 10,
            fontSize: '0.8125rem',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div style={{ color: 'var(--text-muted)', marginBottom: 6, fontSize: '0.75rem', fontWeight: 600 }}>
            PARAMETERS
          </div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {Object.entries(h.params).map(([k, v]) => (
              <span key={k} style={{ fontSize: '0.75rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>{k}: </span>
                <span className="num" style={{ color: 'var(--text-primary)' }}>
                  {String(v)}
                </span>
              </span>
            ))}
          </div>

          {h.testResults && (
            <div style={{ marginTop: 10 }}>
              <div style={{ color: 'var(--text-muted)', marginBottom: 6, fontSize: '0.75rem', fontWeight: 600 }}>
                TEST RESULTS
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                {[
                  { label: 'Sharpe', value: h.testResults.sharpe.toFixed(3), color: 'var(--green)' },
                  { label: 'Max DD', value: `${(h.testResults.maxDD * 100).toFixed(1)}%`, color: 'var(--red)' },
                  { label: 'Win Rate', value: `${(h.testResults.winRate * 100).toFixed(0)}%`, color: 'var(--accent)' },
                ].map(({ label, value, color }) => (
                  <div key={label}>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{label}</div>
                    <div className="num" style={{ color, fontWeight: 600 }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <span>{formatDistanceToNow(new Date(h.createdAt), { addSuffix: true })}</span>
        {h.status === 'pending' && (
          <button
            className="btn btn-primary btn-sm"
            onClick={() => onTest(h.id)}
            disabled={isSubmitting}
          >
            {isSubmitting ? <LoadingSpinner size={12} /> : null}
            Test Now
          </button>
        )}
      </div>
    </div>
  )
}

// ─── Kanban Pipeline ──────────────────────────────────────────────────────────

interface KanbanProps {
  hypotheses: Hypothesis[]
  onTest: (id: number) => void
  testingId: number | null
}

const KanbanView: React.FC<KanbanProps> = ({ hypotheses, onTest, testingId }) => {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 12,
        alignItems: 'start',
      }}
    >
      {KANBAN_STATUSES.map((status) => {
        const items = hypotheses.filter((h) => h.status === status)
        return (
          <div key={status}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: 10,
                paddingBottom: 8,
                borderBottom: `2px solid ${KANBAN_COLORS[status]}`,
              }}
            >
              <StatusBadge value={status} />
              <span
                style={{
                  background: 'var(--bg-elevated)',
                  borderRadius: 10,
                  padding: '1px 6px',
                  fontSize: '0.7rem',
                  color: 'var(--text-muted)',
                }}
              >
                {items.length}
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {items.length === 0 ? (
                <div
                  style={{
                    padding: '16px 0',
                    textAlign: 'center',
                    color: 'var(--text-muted)',
                    fontSize: '0.8125rem',
                  }}
                >
                  Empty
                </div>
              ) : (
                items.map((h) => (
                  <HypothesisCard
                    key={h.id}
                    hypothesis={h}
                    onTest={onTest}
                    testingId={testingId}
                  />
                ))
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ─── Hypotheses Page ──────────────────────────────────────────────────────────

const HypothesesPage: React.FC = () => {
  const [statusFilter, setStatusFilter] = useState<HypothesisStatus | 'all'>('all')
  const [viewMode, setViewMode] = useState<'list' | 'kanban'>('kanban')

  const { data: hypotheses = [], isLoading, error } = useQuery({
    queryKey: ['hypotheses', 'all'],
    queryFn: () => fetchHypotheses(),
    refetchInterval: 30_000,
  })

  const qc = useQueryClient()
  const testMutation = useMutation({
    mutationFn: (id: number) => submitHypothesisTest(id),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['hypotheses'] }),
  })

  const filtered = useMemo(() => {
    let result = statusFilter === 'all'
      ? hypotheses
      : hypotheses.filter((h) => h.status === statusFilter)
    return [...result].sort((a, b) => b.score - a.score)
  }, [hypotheses, statusFilter])

  const counts = useMemo(() => {
    const map: Record<string, number> = { all: hypotheses.length }
    for (const s of ['pending', 'testing', 'adopted', 'rejected'] as HypothesisStatus[]) {
      map[s] = hypotheses.filter((h) => h.status === s).length
    }
    return map
  }, [hypotheses])

  if (error) {
    return (
      <div className="empty-state">
        <div className="icon">⚠</div>
        <span>Failed to load hypotheses</span>
      </div>
    )
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Hypothesis Queue</div>
          <div className="page-subtitle">
            {hypotheses.length} hypotheses · {counts.adopted ?? 0} adopted · {counts.pending ?? 0} pending test
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className={`btn btn-sm ${viewMode === 'list' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setViewMode('list')}
          >
            List
          </button>
          <button
            className={`btn btn-sm ${viewMode === 'kanban' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setViewMode('kanban')}
          >
            Kanban
          </button>
        </div>
      </div>

      {/* Status Tabs (list mode only) */}
      {viewMode === 'list' && (
        <div className="tabs">
          {STATUS_TABS.map((tab) => (
            <button
              key={tab.value}
              className={`tab ${statusFilter === tab.value ? 'active' : ''}`}
              onClick={() => setStatusFilter(tab.value)}
            >
              {tab.label}
              <span
                style={{
                  marginLeft: 6,
                  background: 'var(--bg-elevated)',
                  borderRadius: 10,
                  padding: '1px 6px',
                  fontSize: '0.7rem',
                  color: 'var(--text-muted)',
                }}
              >
                {counts[tab.value] ?? 0}
              </span>
            </button>
          ))}
        </div>
      )}

      {isLoading ? (
        <LoadingSpinner fullPage label="Loading hypotheses…" />
      ) : viewMode === 'kanban' ? (
        <KanbanView
          hypotheses={hypotheses}
          onTest={testMutation.mutate}
          testingId={testMutation.isPending ? (testMutation.variables as number) : null}
        />
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <div className="icon">⧖</div>
          <span>No hypotheses in this category</span>
        </div>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
            gap: 14,
          }}
        >
          {filtered.map((h) => (
            <HypothesisCard
              key={h.id}
              hypothesis={h}
              onTest={testMutation.mutate}
              testingId={testMutation.isPending ? (testMutation.variables as number) : null}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default HypothesesPage
