import React from 'react'
import { clsx } from 'clsx'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  label?: string
  className?: string
  fullHeight?: boolean
}

export function LoadingSpinner({ size = 'md', label, className, fullHeight = false }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-2',
    lg: 'w-12 h-12 border-3',
  }

  return (
    <div className={clsx(
      'flex flex-col items-center justify-center gap-3',
      fullHeight && 'h-full min-h-[200px]',
      className
    )}>
      <div className={clsx(
        'rounded-full border-research-border border-t-research-accent animate-spin',
        sizeClasses[size]
      )} />
      {label && (
        <span className="text-sm text-research-subtle animate-pulse">{label}</span>
      )}
    </div>
  )
}

interface SectionLoaderProps {
  rows?: number
  cols?: number
}

export function SectionLoader({ rows = 3, cols = 1 }: SectionLoaderProps) {
  return (
    <div className="space-y-3 animate-pulse p-4">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className={`grid gap-3 grid-cols-${cols}`}>
          {Array.from({ length: cols }, (_, j) => (
            <div key={j} className="h-4 bg-research-muted rounded" style={{ width: `${60 + Math.random() * 40}%` }} />
          ))}
        </div>
      ))}
    </div>
  )
}
