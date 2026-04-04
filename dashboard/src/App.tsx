// ============================================================
// App.tsx — SRFM Executive Dashboard root
// ============================================================
import React, { useState, useEffect } from 'react'
import { clsx } from 'clsx'
import { Overview } from '@/pages/Overview'
import { Attribution } from '@/pages/Attribution'
import { RiskDashboard } from '@/pages/RiskDashboard'
import { MarketHeatmap } from '@/pages/MarketHeatmap'
import { SignalMonitor } from '@/pages/SignalMonitor'
import { TradeHistory } from '@/pages/TradeHistory'
import { useWebSocket } from '@/hooks/useWebSocket'
import { usePortfolioStore } from '@/store/portfolioStore'
import { WsStatusDot } from '@/components/ui'
import type { WsStatus } from '@/hooks/useWebSocket'

// ---- Route types ----

type Route =
  | 'overview'
  | 'attribution'
  | 'risk'
  | 'heatmap'
  | 'signals'
  | 'trades'

const ROUTES: { key: Route; label: string; icon: string }[] = [
  { key: 'overview',    label: 'Overview',     icon: '◈' },
  { key: 'attribution', label: 'Attribution',  icon: '⬡' },
  { key: 'risk',        label: 'Risk',         icon: '⬟' },
  { key: 'heatmap',     label: 'Heatmap',      icon: '⬢' },
  { key: 'signals',     label: 'Signals',      icon: '◎' },
  { key: 'trades',      label: 'Trades',       icon: '◆' },
]

// ---- Status bar ----

const StatusBar: React.FC<{ wsStatus: WsStatus }> = ({ wsStatus }) => {
  const [time, setTime] = useState(new Date())
  const snapshot = usePortfolioStore((s) => s.snapshot)

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="flex items-center justify-between px-3 py-1 border-t border-[#1e2130] bg-[#0e1017]/50 flex-shrink-0">
      <div className="flex items-center gap-3">
        <span className="text-[9px] font-mono text-slate-600">SRFM DASHBOARD v1.0</span>
        {snapshot && (
          <>
            <span className="text-[9px] font-mono text-slate-600">|</span>
            <span className={clsx(
              'text-[9px] font-mono font-semibold',
              snapshot.dailyPnl >= 0 ? 'text-emerald-500' : 'text-red-500',
            )}>
              D-PNL: {snapshot.dailyPnl >= 0 ? '+' : ''}${snapshot.dailyPnl.toFixed(0)}
            </span>
            <span className={clsx(
              'text-[9px] font-mono',
              snapshot.currentDrawdown < -0.05 ? 'text-amber-500' : 'text-slate-600',
            )}>
              DD: {(snapshot.currentDrawdown * 100).toFixed(1)}%
            </span>
          </>
        )}
      </div>
      <div className="flex items-center gap-3">
        <WsStatusDot status={wsStatus} />
        <span className="text-[9px] font-mono text-slate-600">
          {time.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
        </span>
        <span className="text-[9px] font-mono text-slate-500">
          {time.toLocaleTimeString('en-US', { hour12: false })} UTC
        </span>
      </div>
    </div>
  )
}

// ---- App ----

const App: React.FC = () => {
  const [route, setRoute] = useState<Route>('overview')

  const { status: wsStatus } = useWebSocket({ enabled: false })  // disabled until backend available

  const renderPage = () => {
    switch (route) {
      case 'overview':    return <Overview />
      case 'attribution': return <Attribution />
      case 'risk':        return <RiskDashboard />
      case 'heatmap':     return <MarketHeatmap />
      case 'signals':     return <SignalMonitor />
      case 'trades':      return <TradeHistory />
    }
  }

  return (
    <div className="flex flex-col h-full bg-[#0a0b0e] overflow-hidden">
      {/* Top navigation bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#0e1017] border-b border-[#1e2130] flex-shrink-0 z-50">
        <div className="flex items-center gap-4">
          {/* Logo */}
          <div className="flex items-center gap-2 mr-2">
            <span className="text-blue-400 font-mono font-bold text-sm tracking-widest">SRFM</span>
            <span className="text-slate-600 font-mono text-[10px]">EXEC</span>
          </div>

          {/* Nav */}
          <nav className="flex items-center gap-0.5">
            {ROUTES.map((r) => (
              <button
                key={r.key}
                onClick={() => setRoute(r.key)}
                className={clsx(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded text-[10px] font-mono transition-colors',
                  route === r.key
                    ? 'text-slate-100 bg-blue-500/15 border border-blue-500/30'
                    : 'text-slate-500 hover:text-slate-300 border border-transparent hover:bg-[#111318]',
                )}
              >
                <span className="text-[11px]">{r.icon}</span>
                {r.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[9px] font-mono text-slate-600 px-2 py-0.5 border border-[#1e2130] rounded">
            Spacetime Arena
          </span>
        </div>
      </div>

      {/* Page content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {renderPage()}
      </div>

      {/* Status bar */}
      <StatusBar wsStatus={wsStatus} />
    </div>
  )
}

export default App
