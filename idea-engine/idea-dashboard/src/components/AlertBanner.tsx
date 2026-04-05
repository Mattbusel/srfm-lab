import React, { useState } from 'react'
import type { Alert } from '../types'

interface AlertBannerProps {
  alerts: Alert[]
  onAcknowledge: (id: number) => void
}

const SEVERITY_STYLES = {
  critical: {
    background: 'var(--red-bg)',
    border: 'var(--red-dim)',
    color: 'var(--red)',
    icon: '✕',
  },
  warning: {
    background: 'var(--yellow-bg)',
    border: 'var(--yellow-dim)',
    color: 'var(--yellow)',
    icon: '⚠',
  },
  info: {
    background: 'var(--blue-bg)',
    border: 'var(--blue-dim)',
    color: 'var(--blue)',
    icon: 'ℹ',
  },
}

const AlertBanner: React.FC<AlertBannerProps> = ({ alerts, onAcknowledge }) => {
  const [dismissed, setDismissed] = useState<Set<number>>(new Set())

  const visible = alerts.filter(
    (a) => !a.acknowledged && !dismissed.has(a.id)
  )

  if (visible.length === 0) return null

  // Show only the highest severity
  const sorted = [...visible].sort((a, b) => {
    const order = { critical: 0, warning: 1, info: 2 }
    return order[a.severity] - order[b.severity]
  })

  const top = sorted[0]
  const style = SEVERITY_STYLES[top.severity]
  const extra = sorted.length - 1

  const handleDismiss = () => {
    setDismissed((prev) => new Set([...prev, top.id]))
  }

  const handleAck = () => {
    onAcknowledge(top.id)
    setDismissed((prev) => new Set([...prev, top.id]))
  }

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '8px 16px',
        background: style.background,
        borderBottom: `1px solid ${style.border}`,
        animation: 'fadeIn 200ms ease',
        flexShrink: 0,
      }}
    >
      <span style={{ color: style.color, fontWeight: 700, fontSize: '0.875rem', flexShrink: 0 }}>
        {style.icon}
      </span>
      <span
        style={{
          flex: 1,
          fontSize: '0.8125rem',
          color: 'var(--text-primary)',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        <span style={{ color: style.color, fontWeight: 600, marginRight: 6 }}>
          [{top.type.toUpperCase()}]
        </span>
        {top.message}
        {extra > 0 && (
          <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>
            +{extra} more alert{extra > 1 ? 's' : ''}
          </span>
        )}
      </span>
      <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
        <button
          className="btn btn-sm btn-ghost"
          onClick={handleAck}
          style={{ fontSize: '0.75rem', padding: '2px 8px' }}
        >
          Acknowledge
        </button>
        <button
          className="btn-icon"
          onClick={handleDismiss}
          title="Dismiss"
          style={{ fontSize: '0.875rem', padding: '2px 6px' }}
        >
          ×
        </button>
      </div>
    </div>
  )
}

export default AlertBanner
