import React, { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import {
  fetchSerendipityIdeas,
  generateSerendipityIdeas,
  submitIdeaAsHypothesis,
} from '../api/client'
import type { SerendipityIdea, SerendipityTechnique } from '../types'

const TECHNIQUE_TABS: { value: SerendipityTechnique | 'all'; label: string }[] = [
  { value: 'all',          label: 'All' },
  { value: 'domain_borrow', label: 'Domain Borrow' },
  { value: 'inversion',    label: 'Inversion' },
  { value: 'combination',  label: 'Combination' },
  { value: 'mutation',     label: 'Mutation' },
]

const TECHNIQUE_DESCRIPTIONS: Record<SerendipityTechnique, string> = {
  domain_borrow: 'Apply concepts from unrelated fields',
  inversion:     'Flip the problem or assumption',
  combination:   'Merge two separate techniques',
  mutation:      'Modify an existing strategy',
}

// ─── Idea Card ────────────────────────────────────────────────────────────────

interface IdeaCardProps {
  idea: SerendipityIdea
  onSubmit: (id: number) => void
  isSubmitting: boolean
}

const IdeaCard: React.FC<IdeaCardProps> = ({ idea, onSubmit, isSubmitting }) => {
  const [expanded, setExpanded] = useState(false)

  const complexityColors = {
    low:    { bg: 'var(--green-bg)',  color: 'var(--green)'  },
    medium: { bg: 'var(--yellow-bg)', color: 'var(--yellow)' },
    high:   { bg: 'var(--red-bg)',    color: 'var(--red)'    },
  }
  const complexStyle = complexityColors[idea.complexity]

  const scoreColor =
    (idea.score ?? 0) >= 0.85
      ? 'var(--gold)'
      : (idea.score ?? 0) >= 0.7
      ? 'var(--green)'
      : (idea.score ?? 0) >= 0.5
      ? 'var(--accent)'
      : 'var(--text-muted)'

  return (
    <div
      className="card fade-in"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        borderTop: `3px solid ${complexStyle.color}`,
      }}
    >
      {/* Header badges */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <StatusBadge value={idea.technique} size="sm" />
          <span
            style={{
              fontSize: '0.7rem',
              background: 'var(--bg-elevated)',
              color: 'var(--text-muted)',
              padding: '2px 8px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)',
              fontStyle: 'italic',
            }}
          >
            {idea.domain}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <StatusBadge value={idea.complexity} size="sm" />
          {idea.score !== undefined && (
            <span className="num" style={{ fontSize: '0.8125rem', fontWeight: 700, color: scoreColor }}>
              {(idea.score * 100).toFixed(0)}
            </span>
          )}
        </div>
      </div>

      {/* Idea Text */}
      <div
        style={{
          fontSize: '0.9375rem',
          color: 'var(--text-primary)',
          lineHeight: 1.5,
          fontWeight: 500,
          cursor: 'pointer',
          borderLeft: `2px solid ${complexStyle.color}`,
          paddingLeft: 12,
        }}
        onClick={() => setExpanded((v) => !v)}
      >
        {idea.ideaText}
      </div>

      {/* Technique description (small) */}
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
        {TECHNIQUE_DESCRIPTIONS[idea.technique]}
      </div>

      {/* Rationale (expanded) */}
      {expanded && (
        <div
          style={{
            background: 'var(--bg-elevated)',
            borderRadius: 'var(--radius)',
            padding: '10px 12px',
            border: '1px solid var(--border)',
            fontSize: '0.8125rem',
            color: 'var(--text-secondary)',
            lineHeight: 1.6,
            animation: 'fadeIn 150ms ease',
          }}
        >
          <div
            style={{
              fontSize: '0.7rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: 'var(--text-muted)',
              marginBottom: 6,
            }}
          >
            Rationale
          </div>
          {idea.rationale}
        </div>
      )}

      {/* Footer */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <button
          className="btn btn-ghost btn-sm"
          style={{ fontSize: '0.75rem', padding: '2px 0', color: 'var(--accent)' }}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? '▲ Hide rationale' : '▼ Show rationale'}
        </button>

        {!idea.submittedAsHypothesis ? (
          <button
            className="btn btn-primary btn-sm"
            onClick={() => onSubmit(idea.id)}
            disabled={isSubmitting}
            style={{ display: 'flex', alignItems: 'center', gap: 4 }}
          >
            {isSubmitting ? <LoadingSpinner size={12} /> : null}
            Test This
          </button>
        ) : (
          <span
            style={{
              fontSize: '0.75rem',
              color: 'var(--green)',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            ✓ In hypothesis queue
          </span>
        )}
      </div>
    </div>
  )
}

// ─── Serendipity Page ─────────────────────────────────────────────────────────

const SerendipityPage: React.FC = () => {
  const [techniqueFilter, setTechniqueFilter] = useState<SerendipityTechnique | 'all'>('all')
  const [isGenerating, setIsGenerating] = useState(false)
  const [submitId, setSubmitId] = useState<number | null>(null)
  const [localIdeas, setLocalIdeas] = useState<SerendipityIdea[]>([])

  const { data: ideas = [], isLoading } = useQuery({
    queryKey: ['serendipity'],
    queryFn: () => fetchSerendipityIdeas(),
    refetchInterval: 120_000,
    staleTime: 60_000,
  })

  const qc = useQueryClient()
  const submitMutation = useMutation({
    mutationFn: (id: number) => submitIdeaAsHypothesis(id),
    onMutate: (id) => setSubmitId(id),
    onSuccess: (_data, id) => {
      void qc.invalidateQueries({ queryKey: ['hypotheses'] })
      setLocalIdeas((prev) =>
        prev.map((i) => (i.id === id ? { ...i, submittedAsHypothesis: true } : i))
      )
    },
    onSettled: () => setSubmitId(null),
  })

  const allIdeas = useMemo(
    () => [...ideas, ...localIdeas],
    [ideas, localIdeas]
  )

  const filtered = useMemo(
    () =>
      techniqueFilter === 'all'
        ? allIdeas
        : allIdeas.filter((i) => i.technique === techniqueFilter),
    [allIdeas, techniqueFilter]
  )

  const counts = useMemo(() => {
    const map: Record<string, number> = { all: allIdeas.length }
    for (const tech of ['domain_borrow', 'inversion', 'combination', 'mutation'] as SerendipityTechnique[]) {
      map[tech] = allIdeas.filter((i) => i.technique === tech).length
    }
    return map
  }, [allIdeas])

  const handleGenerate = async () => {
    setIsGenerating(true)
    try {
      const newIdeas = await generateSerendipityIdeas()
      setLocalIdeas((prev) => [...newIdeas, ...prev])
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Serendipity Engine</div>
          <div className="page-subtitle">
            Wild cross-domain ideas for new strategy hypotheses
          </div>
        </div>
        <button
          className="btn btn-primary"
          onClick={handleGenerate}
          disabled={isGenerating}
          style={{
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {isGenerating ? (
            <>
              <LoadingSpinner size={14} color="#0d0d0d" />
              Generating…
            </>
          ) : (
            <>
              ✦ Generate New Ideas
            </>
          )}
        </button>
      </div>

      {/* Technique Tabs */}
      <div className="tabs">
        {TECHNIQUE_TABS.map((tab) => (
          <button
            key={tab.value}
            className={`tab ${techniqueFilter === tab.value ? 'active' : ''}`}
            onClick={() => setTechniqueFilter(tab.value)}
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

      {isLoading ? (
        <LoadingSpinner fullPage label="Loading ideas…" />
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <div className="icon">✦</div>
          <span>No ideas yet — click Generate to create some</span>
        </div>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
            gap: 14,
          }}
        >
          {filtered.map((idea) => (
            <IdeaCard
              key={idea.id}
              idea={idea}
              onSubmit={submitMutation.mutate}
              isSubmitting={submitId === idea.id}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default SerendipityPage
