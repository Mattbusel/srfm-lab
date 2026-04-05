import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { format, parseISO } from 'date-fns'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import { fetchWeeklyReport } from '../api/client'
import type { Alert } from '../types'

// ─── Simple Markdown Renderer ─────────────────────────────────────────────────

function renderMarkdown(md: string): React.ReactNode[] {
  const lines = md.split('\n')
  const elements: React.ReactNode[] = []
  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i]

    // H1
    if (line.startsWith('# ')) {
      elements.push(
        <h1 key={key++} style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: 8, marginTop: 16, color: 'var(--text-primary)' }}>
          {line.slice(2)}
        </h1>
      )
      i++; continue
    }
    // H2
    if (line.startsWith('## ')) {
      elements.push(
        <h2 key={key++} style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 6, marginTop: 20, color: 'var(--accent)', borderBottom: '1px solid var(--border)', paddingBottom: 4 }}>
          {line.slice(3)}
        </h2>
      )
      i++; continue
    }
    // H3
    if (line.startsWith('### ')) {
      elements.push(
        <h3 key={key++} style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: 4, marginTop: 14, color: 'var(--text-primary)' }}>
          {line.slice(4)}
        </h3>
      )
      i++; continue
    }

    // Table
    if (line.startsWith('|')) {
      const rows: string[][] = []
      while (i < lines.length && lines[i].startsWith('|')) {
        if (!lines[i].includes('---')) {
          rows.push(lines[i].split('|').filter(Boolean).map((c) => c.trim()))
        }
        i++
      }
      if (rows.length > 0) {
        elements.push(
          <div key={key++} style={{ overflowX: 'auto', marginBottom: 12 }}>
            <table>
              <thead>
                <tr>
                  {rows[0].map((cell, ci) => (
                    <th key={ci}>{cell}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.slice(1).map((row, ri) => (
                  <tr key={ri}>
                    {row.map((cell, ci) => (
                      <td key={ci} className="num" style={{ fontSize: '0.8125rem' }}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      }
      continue
    }

    // Bullet list
    if (line.startsWith('- ') || line.startsWith('* ')) {
      const items: string[] = []
      while (i < lines.length && (lines[i].startsWith('- ') || lines[i].startsWith('* '))) {
        items.push(lines[i].slice(2))
        i++
      }
      elements.push(
        <ul key={key++} style={{ paddingLeft: 20, marginBottom: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {items.map((item, ii) => (
            <li key={ii} style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {renderInline(item)}
            </li>
          ))}
        </ul>
      )
      continue
    }

    // Numbered list
    if (/^\d+\.\s/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s/, ''))
        i++
      }
      elements.push(
        <ol key={key++} style={{ paddingLeft: 20, marginBottom: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {items.map((item, ii) => (
            <li key={ii} style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {renderInline(item)}
            </li>
          ))}
        </ol>
      )
      continue
    }

    // Empty line → spacer
    if (line.trim() === '') {
      elements.push(<div key={key++} style={{ height: 6 }} />)
      i++; continue
    }

    // Paragraph
    elements.push(
      <p key={key++} style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: 6 }}>
        {renderInline(line)}
      </p>
    )
    i++
  }

  return elements
}

function renderInline(text: string): React.ReactNode {
  // Bold: **text**
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={i} style={{ background: 'var(--bg-elevated)', padding: '1px 5px', borderRadius: 3, fontSize: '0.85em', color: 'var(--accent)' }}>{part.slice(1, -1)}</code>
    }
    return part
  })
}

// ─── Alert History ────────────────────────────────────────────────────────────

interface AlertHistoryProps {
  alerts: Alert[]
}

const AlertHistory: React.FC<AlertHistoryProps> = ({ alerts }) => {
  const severityColor: Record<string, string> = {
    critical: 'var(--red)',
    warning:  'var(--yellow)',
    info:     'var(--blue)',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {alerts.map((alert) => (
        <div
          key={alert.id}
          style={{
            display: 'flex',
            gap: 10,
            padding: '8px 12px',
            background: 'var(--bg-elevated)',
            borderRadius: 'var(--radius)',
            border: '1px solid var(--border)',
            borderLeft: `3px solid ${severityColor[alert.severity]}`,
            opacity: alert.acknowledged ? 0.55 : 1,
          }}
        >
          <StatusBadge value={alert.severity} size="sm" />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
              {alert.message}
            </div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>
              {format(parseISO(alert.createdAt), 'MMM d, HH:mm')}
              {alert.acknowledged && (
                <span style={{ marginLeft: 8, color: 'var(--green)' }}>✓ acknowledged</span>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

// ─── Narratives Page ──────────────────────────────────────────────────────────

const NarrativesPage: React.FC = () => {
  const [activeSection, setActiveSection] = useState<'report' | 'alerts'>('report')

  const { data: report, isLoading, error } = useQuery({
    queryKey: ['narratives', 'weekly'],
    queryFn: fetchWeeklyReport,
    refetchInterval: 300_000, // 5 min
    staleTime: 240_000,
  })

  const sortedAlerts = report
    ? [...report.alerts].sort(
        (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      )
    : []

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Research Narratives</div>
          <div className="page-subtitle">
            {report
              ? `Week of ${report.weekStart} — ${report.weekEnd}`
              : 'Weekly report & alert history'}
          </div>
        </div>
        {report && (
          <div style={{ display: 'flex', gap: 12, fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
            <span style={{ color: 'var(--green)' }}>
              ✓ {report.hypothesesAdopted} adopted
            </span>
            <span style={{ color: 'var(--red)' }}>
              ✗ {report.hypothesesRejected} rejected
            </span>
          </div>
        )}
      </div>

      {/* Section Tabs */}
      <div className="tabs">
        {[
          { value: 'report', label: 'Weekly Report' },
          { value: 'alerts', label: `Alert History (${sortedAlerts.length})` },
        ].map(({ value, label }) => (
          <button
            key={value}
            className={`tab ${activeSection === value ? 'active' : ''}`}
            onClick={() => setActiveSection(value as 'report' | 'alerts')}
          >
            {label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <LoadingSpinner fullPage label="Loading report…" />
      ) : error ? (
        <div className="empty-state">
          <div className="icon">⚠</div>
          <span>Failed to load report</span>
        </div>
      ) : !report ? null : activeSection === 'report' ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 240px', gap: 20, alignItems: 'start' }}>
          {/* Report */}
          <div className="card">
            <div style={{ lineHeight: 1.6 }}>
              {renderMarkdown(report.markdownContent)}
            </div>
            <div style={{ marginTop: 20, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Generated {format(parseISO(report.generatedAt), 'MMM d, yyyy HH:mm')}
            </div>
          </div>

          {/* Sidebar: Stats */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="card">
              <div className="card-title" style={{ marginBottom: 10 }}>This Week</div>
              {[
                { label: 'Adopted', value: report.hypothesesAdopted, color: 'var(--green)' },
                { label: 'Rejected', value: report.hypothesesRejected, color: 'var(--red)' },
                { label: 'Top Genomes', value: report.topGenomes.length, color: 'var(--gold)' },
              ].map(({ label, value, color }) => (
                <div
                  key={label}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '6px 0',
                    borderBottom: '1px solid var(--border-subtle)',
                  }}
                >
                  <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>{label}</span>
                  <span className="num" style={{ fontWeight: 700, color }}>{value}</span>
                </div>
              ))}
            </div>

            <div className="card">
              <div className="card-title" style={{ marginBottom: 10 }}>Hall of Fame (This Week)</div>
              {report.topGenomes.map((id) => (
                <div
                  key={id}
                  style={{
                    padding: '4px 0',
                    fontSize: '0.8125rem',
                    color: 'var(--gold)',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  ★ Genome #{id}
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <AlertHistory alerts={sortedAlerts} />
      )}
    </div>
  )
}

export default NarrativesPage
