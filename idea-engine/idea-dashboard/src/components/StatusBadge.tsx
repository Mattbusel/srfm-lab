import React from 'react'
import clsx from 'clsx'
import type {
  Island,
  HypothesisStatus,
  AlertSeverity,
  SerendipityTechnique,
  Complexity,
  HypothesisSource,
  PaperSource,
} from '../types'

type BadgeVariant =
  | Island
  | HypothesisStatus
  | AlertSeverity
  | SerendipityTechnique
  | Complexity
  | HypothesisSource
  | PaperSource
  | 'hof'
  | 'promoted'
  | 'live'
  | 'shadow'
  | string

interface StatusBadgeProps {
  value: BadgeVariant
  label?: string
  size?: 'sm' | 'md'
  className?: string
}

const BADGE_STYLES: Record<string, React.CSSProperties> = {
  // Islands
  BULL: { background: 'var(--green-bg)', color: 'var(--green)', borderColor: 'var(--green-dim)' },
  BEAR: { background: 'var(--red-bg)', color: 'var(--red)', borderColor: 'var(--red-dim)' },
  NEUTRAL: { background: 'var(--blue-bg)', color: 'var(--blue)', borderColor: 'var(--blue-dim)' },

  // Hypothesis Status
  pending: { background: 'var(--yellow-bg)', color: 'var(--yellow)', borderColor: 'var(--yellow-dim)' },
  testing: { background: 'var(--blue-bg)', color: 'var(--blue)', borderColor: 'var(--blue-dim)' },
  adopted: { background: 'var(--green-bg)', color: 'var(--green)', borderColor: 'var(--green-dim)' },
  rejected: { background: 'var(--red-bg)', color: 'var(--red)', borderColor: 'var(--red-dim)' },

  // Alert Severity
  critical: { background: 'var(--red-bg)', color: 'var(--red)', borderColor: 'var(--red-dim)' },
  warning: { background: 'var(--yellow-bg)', color: 'var(--yellow)', borderColor: 'var(--yellow-dim)' },
  info: { background: 'var(--blue-bg)', color: 'var(--blue)', borderColor: 'var(--blue-dim)' },

  // Serendipity Technique
  domain_borrow: { background: 'var(--purple-bg)', color: 'var(--purple)', borderColor: 'var(--purple-dim)' },
  inversion: { background: 'var(--yellow-bg)', color: 'var(--yellow)', borderColor: 'var(--yellow-dim)' },
  combination: { background: 'var(--blue-bg)', color: 'var(--blue)', borderColor: 'var(--blue-dim)' },
  mutation: { background: 'var(--green-bg)', color: 'var(--green)', borderColor: 'var(--green-dim)' },

  // Complexity
  low: { background: 'var(--green-bg)', color: 'var(--green)', borderColor: 'var(--green-dim)' },
  medium: { background: 'var(--yellow-bg)', color: 'var(--yellow)', borderColor: 'var(--yellow-dim)' },
  high: { background: 'var(--red-bg)', color: 'var(--red)', borderColor: 'var(--red-dim)' },

  // Hypothesis Source
  genome: { background: 'var(--accent-glow)', color: 'var(--accent)', borderColor: 'var(--accent-dim)' },
  academic: { background: 'var(--blue-bg)', color: 'var(--blue)', borderColor: 'var(--blue-dim)' },
  serendipity: { background: 'var(--purple-bg)', color: 'var(--purple)', borderColor: 'var(--purple-dim)' },
  causal: { background: 'var(--yellow-bg)', color: 'var(--yellow)', borderColor: 'var(--yellow-dim)' },

  // Paper Source
  arXiv: { background: 'var(--red-bg)', color: 'var(--red)', borderColor: 'var(--red-dim)' },
  SSRN: { background: 'var(--blue-bg)', color: 'var(--blue)', borderColor: 'var(--blue-dim)' },
  local: { background: 'var(--bg-elevated)', color: 'var(--text-secondary)', borderColor: 'var(--border)' },

  // Special
  hof: { background: 'var(--gold-bg)', color: 'var(--gold)', borderColor: '#d97706' },
  promoted: { background: 'var(--accent-glow)', color: 'var(--accent)', borderColor: 'var(--accent-dim)' },
  live: { background: 'var(--green-bg)', color: 'var(--green)', borderColor: 'var(--green-dim)' },
  shadow: { background: 'var(--bg-elevated)', color: 'var(--text-muted)', borderColor: 'var(--border)' },
}

const LABEL_MAP: Record<string, string> = {
  domain_borrow: 'Domain Borrow',
  hof: 'Hall of Fame',
  entry_signal: 'Entry Signal',
  exit_signal: 'Exit Signal',
  position_sizing: 'Pos. Sizing',
  risk_filter: 'Risk Filter',
  regime_detection: 'Regime',
  correlation: 'Correlation',
}

const StatusBadge: React.FC<StatusBadgeProps> = ({
  value,
  label,
  size = 'md',
  className,
}) => {
  const style = BADGE_STYLES[value] ?? {
    background: 'var(--bg-elevated)',
    color: 'var(--text-secondary)',
    borderColor: 'var(--border)',
  }

  const displayLabel = label ?? LABEL_MAP[value] ?? value

  return (
    <span
      className={clsx(className)}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: size === 'sm' ? '2px 6px' : '3px 8px',
        borderRadius: 'var(--radius-sm)',
        border: '1px solid',
        fontSize: size === 'sm' ? '0.6875rem' : '0.75rem',
        fontWeight: 600,
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
        whiteSpace: 'nowrap',
        lineHeight: 1.4,
        ...style,
      }}
    >
      {displayLabel}
    </span>
  )
}

export default StatusBadge
