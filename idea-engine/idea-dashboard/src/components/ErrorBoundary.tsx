import React, { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, info: ErrorInfo) => void
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.setState({ errorInfo: info })
    this.props.onError?.(error, info)
    console.error('[ErrorBoundary] Caught error:', error, info)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback

      return (
        <div
          style={{
            padding: '24px',
            background: 'var(--bg-card)',
            border: '1px solid var(--red-dim)',
            borderRadius: '8px',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '1.25rem' }}>⚠</span>
            <span style={{ fontWeight: 600, color: 'var(--red)' }}>
              Component Error
            </span>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
            {this.state.error?.message ?? 'An unexpected error occurred.'}
          </p>
          {this.state.errorInfo && (
            <details style={{ cursor: 'pointer' }}>
              <summary
                style={{
                  color: 'var(--text-muted)',
                  fontSize: '0.75rem',
                  marginBottom: '6px',
                }}
              >
                Stack trace
              </summary>
              <pre
                style={{
                  fontSize: '0.7rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  whiteSpace: 'pre-wrap',
                  background: 'var(--bg-elevated)',
                  padding: '8px',
                  borderRadius: '4px',
                  maxHeight: '200px',
                  overflowY: 'auto',
                }}
              >
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}
          <button
            className="btn btn-secondary btn-sm"
            onClick={this.handleReset}
            style={{ alignSelf: 'flex-start' }}
          >
            Try again
          </button>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
