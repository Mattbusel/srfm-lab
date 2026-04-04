import React, { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback

      return (
        <div className="flex flex-col items-center justify-center p-8 gap-4 text-center">
          <AlertTriangle className="text-research-warning" size={32} />
          <div>
            <p className="text-research-text font-medium">Component Error</p>
            <p className="text-sm text-research-subtle mt-1 font-mono">
              {this.state.error?.message ?? 'Unknown error'}
            </p>
          </div>
          <button
            onClick={this.handleReset}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-research-muted hover:bg-research-border rounded transition-colors"
          >
            <RefreshCw size={14} />
            Try Again
          </button>
        </div>
      )
    }

    return this.props.children
  }
}

// Inline error display for query errors
interface ErrorDisplayProps {
  error: Error | string | null
  onRetry?: () => void
  compact?: boolean
}

export function ErrorDisplay({ error, onRetry, compact = false }: ErrorDisplayProps) {
  if (!error) return null
  const message = typeof error === 'string' ? error : error.message

  if (compact) {
    return (
      <div className="flex items-center gap-2 text-sm text-research-bear p-2">
        <AlertTriangle size={14} />
        <span className="font-mono">{message}</span>
        {onRetry && (
          <button onClick={onRetry} className="ml-auto text-research-subtle hover:text-research-text">
            <RefreshCw size={12} />
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center gap-3 p-6 text-center">
      <AlertTriangle className="text-research-bear" size={24} />
      <p className="text-sm text-research-subtle font-mono">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-2 px-3 py-1.5 text-sm bg-research-muted hover:bg-research-border rounded transition-colors"
        >
          <RefreshCw size={14} />
          Retry
        </button>
      )}
    </div>
  )
}
