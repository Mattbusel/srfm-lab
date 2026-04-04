import React, { useEffect, useState } from 'react'
import { PanelLeft, Wifi, WifiOff, RefreshCw, Clock } from 'lucide-react'
import { format } from 'date-fns'
import { clsx } from 'clsx'
import { researchWS } from '@/api/client'

interface TopBarProps {
  onToggleSidebar: () => void
  title?: string
}

export function TopBar({ onToggleSidebar, title }: TopBarProps) {
  const [connected, setConnected] = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)

  useEffect(() => {
    const unsub1 = researchWS.on('connected', () => setConnected(true))
    const unsub2 = researchWS.on('disconnected', () => setConnected(false))
    researchWS.connect()

    const timer = setInterval(() => setCurrentTime(new Date()), 1000)

    return () => {
      unsub1()
      unsub2()
      clearInterval(timer)
    }
  }, [])

  const handleRefresh = () => {
    setLastRefresh(new Date())
    window.dispatchEvent(new CustomEvent('research:refresh'))
  }

  return (
    <header className="h-12 bg-research-surface border-b border-research-border flex items-center px-4 gap-4 shrink-0">
      <button
        onClick={onToggleSidebar}
        className="text-research-subtle hover:text-research-text transition-colors"
        title="Toggle sidebar"
      >
        <PanelLeft size={18} />
      </button>

      <div className="flex-1">
        {title && (
          <h1 className="text-sm font-semibold text-research-text">{title}</h1>
        )}
      </div>

      <div className="flex items-center gap-4">
        {/* Last refresh */}
        {lastRefresh && (
          <span className="text-xs text-research-subtle/60 font-mono hidden sm:block">
            Refreshed {format(lastRefresh, 'HH:mm:ss')}
          </span>
        )}

        {/* Refresh button */}
        <button
          onClick={handleRefresh}
          className="text-research-subtle hover:text-research-text transition-colors"
          title="Refresh data"
        >
          <RefreshCw size={15} />
        </button>

        {/* Clock */}
        <div className="flex items-center gap-1.5 text-xs font-mono text-research-subtle">
          <Clock size={12} />
          <span>{format(currentTime, 'HH:mm:ss')}</span>
        </div>

        {/* WS connection status */}
        <div className={clsx(
          'flex items-center gap-1.5 text-xs font-mono',
          connected ? 'text-research-bull' : 'text-research-bear'
        )}>
          {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
          <span className="hidden sm:block">
            {connected ? 'Live' : 'Disconnected'}
          </span>
        </div>

        {/* Env badge */}
        <span className="px-2 py-0.5 text-[10px] font-mono bg-research-warning/15 text-research-warning rounded border border-research-warning/30">
          RESEARCH
        </span>
      </div>
    </header>
  )
}
