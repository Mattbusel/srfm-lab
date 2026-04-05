import React, { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { formatDistanceToNow } from 'date-fns'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import { fetchAcademicPapers, submitHypothesisTest } from '../api/client'
import type { AcademicPaper } from '../types'

// ─── Paper Card ───────────────────────────────────────────────────────────────

interface PaperCardProps {
  paper: AcademicPaper
}

const PaperCard: React.FC<PaperCardProps> = ({ paper }) => {
  const [expanded, setExpanded] = useState(false)
  const qc = useQueryClient()
  const testMutation = useMutation({
    mutationFn: (id: number) => submitHypothesisTest(id),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['hypotheses'] }),
  })

  const scorePercent = Math.round(paper.relevanceScore * 100)
  const scoreColor =
    paper.relevanceScore >= 0.85
      ? 'var(--green)'
      : paper.relevanceScore >= 0.7
      ? 'var(--accent)'
      : 'var(--yellow)'

  return (
    <div
      className="card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
        borderLeft: '3px solid var(--border-emphasis)',
        transition: 'border-color var(--transition)',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.borderLeftColor = 'var(--accent)')}
      onMouseLeave={(e) => (e.currentTarget.style.borderLeftColor = 'var(--border-emphasis)')}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 6 }}>
            <StatusBadge value={paper.source} size="sm" />
            {paper.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                style={{
                  fontSize: '0.65rem',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text-muted)',
                  padding: '2px 6px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border)',
                }}
              >
                {tag}
              </span>
            ))}
          </div>
          <h3
            style={{
              fontSize: '0.9375rem',
              fontWeight: 600,
              color: 'var(--text-primary)',
              lineHeight: 1.4,
              cursor: 'pointer',
            }}
            onClick={() => setExpanded((v) => !v)}
          >
            {paper.title}
          </h3>
        </div>
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <div className="num" style={{ fontSize: '1.25rem', fontWeight: 700, color: scoreColor }}>
            {scorePercent}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>relevance</div>
        </div>
      </div>

      {/* Relevance Bar */}
      <div
        style={{
          height: 3,
          background: 'var(--bg-elevated)',
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${scorePercent}%`,
            background: scoreColor,
            transition: 'width 0.5s ease',
          }}
        />
      </div>

      {/* Authors + Date */}
      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
        <span>{paper.authors.join(', ')}</span>
        <span>{paper.publishedAt}</span>
      </div>

      {/* Abstract Excerpt */}
      <p
        style={{
          fontSize: '0.8125rem',
          color: 'var(--text-secondary)',
          lineHeight: 1.6,
          display: '-webkit-box',
          WebkitLineClamp: expanded ? 'none' : 3,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
          cursor: 'pointer',
          margin: 0,
        }}
        onClick={() => setExpanded((v) => !v)}
      >
        {paper.abstract}
      </p>

      {/* Expand Toggle */}
      <button
        className="btn btn-ghost btn-sm"
        style={{ alignSelf: 'flex-start', fontSize: '0.75rem', padding: '2px 0', color: 'var(--accent)' }}
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? '▲ Show less' : '▼ Read more'}
      </button>

      {/* Extracted Hypotheses */}
      {expanded && paper.extractedHypotheses && paper.extractedHypotheses.length > 0 && (
        <div>
          <div
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              color: 'var(--text-muted)',
              marginBottom: 8,
            }}
          >
            Extracted Hypotheses
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {paper.extractedHypotheses.map((hyp, i) => (
              <div
                key={i}
                style={{
                  background: 'var(--bg-elevated)',
                  borderRadius: 'var(--radius)',
                  padding: '10px 12px',
                  border: '1px solid var(--border)',
                  display: 'flex',
                  gap: 10,
                  alignItems: 'flex-start',
                }}
              >
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', gap: 6, marginBottom: 4 }}>
                    <StatusBadge value={hyp.type} size="sm" />
                    <span
                      className="num"
                      style={{
                        fontSize: '0.75rem',
                        color:
                          hyp.confidence >= 0.8
                            ? 'var(--green)'
                            : hyp.confidence >= 0.6
                            ? 'var(--yellow)'
                            : 'var(--red)',
                      }}
                    >
                      {(hyp.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                  <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
                    {hyp.description}
                  </div>
                </div>
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => testMutation.mutate(paper.id * 100 + i)}
                  disabled={testMutation.isPending}
                  style={{ flexShrink: 0 }}
                >
                  Add to Queue
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Link */}
      {expanded && paper.url && (
        <a
          href={paper.url}
          target="_blank"
          rel="noopener noreferrer"
          style={{ fontSize: '0.8125rem' }}
        >
          View paper ↗
        </a>
      )}
    </div>
  )
}

// ─── Academic Page ────────────────────────────────────────────────────────────

const AcademicPage: React.FC = () => {
  const [query, setQuery] = useState('')
  const [sortMode, setSortMode] = useState<'relevance' | 'date'>('relevance')

  const { data: papers = [], isLoading, error } = useQuery({
    queryKey: ['academic', query],
    queryFn: () => fetchAcademicPapers(query || undefined),
    refetchInterval: 120_000,
    staleTime: 60_000,
  })

  const filtered = useMemo(() => {
    let result = query
      ? papers.filter(
          (p) =>
            p.title.toLowerCase().includes(query.toLowerCase()) ||
            p.abstract.toLowerCase().includes(query.toLowerCase()) ||
            p.tags.some((t) => t.toLowerCase().includes(query.toLowerCase()))
        )
      : papers

    if (sortMode === 'relevance') {
      result = [...result].sort((a, b) => b.relevanceScore - a.relevanceScore)
    } else {
      result = [...result].sort(
        (a, b) =>
          new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime()
      )
    }
    return result
  }, [papers, query, sortMode])

  const totalExtracted = papers.reduce(
    (acc, p) => acc + (p.extractedHypotheses?.length ?? 0),
    0
  )

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Academic Feed</div>
          <div className="page-subtitle">
            {papers.length} papers · {totalExtracted} extracted hypotheses
          </div>
        </div>
      </div>

      {/* Search + Sort */}
      <div
        style={{
          display: 'flex',
          gap: 10,
          marginBottom: 20,
          alignItems: 'center',
        }}
      >
        <input
          type="text"
          placeholder="Search papers, abstracts, tags…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ flex: 1, maxWidth: 400 }}
        />
        <select
          value={sortMode}
          onChange={(e) => setSortMode(e.target.value as 'relevance' | 'date')}
          style={{ fontSize: '0.8125rem', padding: '4px 8px' }}
        >
          <option value="relevance">Sort: Relevance</option>
          <option value="date">Sort: Date</option>
        </select>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {filtered.length} results
        </span>
      </div>

      {isLoading ? (
        <LoadingSpinner fullPage label="Loading papers…" />
      ) : error ? (
        <div className="empty-state">
          <div className="icon">⚠</div>
          <span>Failed to load papers</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <div className="icon">📄</div>
          <span>No papers match your search</span>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {filtered.map((paper) => (
            <PaperCard key={paper.id} paper={paper} />
          ))}
        </div>
      )}
    </div>
  )
}

export default AcademicPage
