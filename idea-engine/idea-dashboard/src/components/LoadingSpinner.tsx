import React from 'react'

interface LoadingSpinnerProps {
  size?: number
  color?: string
  label?: string
  fullPage?: boolean
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 24,
  color = 'var(--accent)',
  label,
  fullPage = false,
}) => {
  const spinner = (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '12px',
      }}
    >
      <div
        style={{
          width: size,
          height: size,
          border: `2px solid var(--border-emphasis)`,
          borderTopColor: color,
          borderRadius: '50%',
          animation: 'spin 600ms linear infinite',
          flexShrink: 0,
        }}
      />
      {label && (
        <span style={{ color: 'var(--text-muted)', fontSize: '0.8125rem' }}>
          {label}
        </span>
      )}
    </div>
  )

  if (fullPage) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: 200,
        }}
      >
        {spinner}
      </div>
    )
  }

  return spinner
}

export default LoadingSpinner
