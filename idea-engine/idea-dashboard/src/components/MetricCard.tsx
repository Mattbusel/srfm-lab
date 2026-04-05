import React from 'react'
import clsx from 'clsx'

interface MetricCardProps {
  label: string
  value: string | number
  subValue?: string
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
  color?: 'default' | 'green' | 'red' | 'yellow' | 'accent' | 'gold'
  icon?: React.ReactNode
  loading?: boolean
  onClick?: () => void
}

const COLOR_MAP = {
  default: 'var(--text-primary)',
  green: 'var(--green)',
  red: 'var(--red)',
  yellow: 'var(--yellow)',
  accent: 'var(--accent)',
  gold: 'var(--gold)',
}

const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subValue,
  trend,
  trendValue,
  color = 'default',
  icon,
  loading = false,
  onClick,
}) => {
  const valueColor = COLOR_MAP[color]
  const trendColor =
    trend === 'up'
      ? 'var(--green)'
      : trend === 'down'
      ? 'var(--red)'
      : 'var(--text-muted)'
  const trendIcon = trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'

  return (
    <div
      className={clsx('card', onClick && 'card-clickable')}
      onClick={onClick}
      style={{
        cursor: onClick ? 'pointer' : undefined,
        transition: 'border-color var(--transition)',
        minWidth: 0,
      }}
      onMouseEnter={(e) => {
        if (onClick) {
          ;(e.currentTarget as HTMLDivElement).style.borderColor =
            'var(--border-emphasis)'
        }
      }}
      onMouseLeave={(e) => {
        if (onClick) {
          ;(e.currentTarget as HTMLDivElement).style.borderColor =
            'var(--border)'
        }
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          marginBottom: '8px',
        }}
      >
        <span
          style={{
            fontSize: '0.75rem',
            fontWeight: 600,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: 'var(--text-muted)',
          }}
        >
          {label}
        </span>
        {icon && (
          <span style={{ color: 'var(--text-muted)', fontSize: '1rem' }}>
            {icon}
          </span>
        )}
      </div>

      {loading ? (
        <div
          className="skeleton"
          style={{ height: 28, width: '60%', borderRadius: 4 }}
        />
      ) : (
        <div
          style={{
            fontSize: '1.75rem',
            fontWeight: 700,
            fontFamily: 'var(--font-mono)',
            fontVariantNumeric: 'tabular-nums',
            color: valueColor,
            lineHeight: 1,
            marginBottom: subValue || trend ? '8px' : 0,
          }}
        >
          {value}
        </div>
      )}

      {(subValue || trend) && !loading && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '0.75rem',
          }}
        >
          {trend && trendValue && (
            <span style={{ color: trendColor, fontWeight: 600 }}>
              {trendIcon} {trendValue}
            </span>
          )}
          {subValue && (
            <span style={{ color: 'var(--text-muted)' }}>{subValue}</span>
          )}
        </div>
      )}
    </div>
  )
}

export default MetricCard
