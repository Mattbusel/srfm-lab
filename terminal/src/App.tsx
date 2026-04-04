// ============================================================
// App.tsx — Root component with navigation and routing
// ============================================================
import React, { useState, useCallback, useEffect } from 'react'
import { useSettingsStore } from '@/store/settingsStore'
import { useBHStore } from '@/store/bhStore'
import { Terminal } from '@/pages/Terminal'
import { StrategyBuilder } from '@/pages/StrategyBuilder'
import { Research } from '@/pages/Research'
import { Scanner } from '@/pages/Scanner'
import { Alerts } from '@/pages/Alerts'
import { Settings } from '@/pages/Settings'

// ============================================================
// Route types
// ============================================================

type Route = 'terminal' | 'strategy' | 'research' | 'scanner' | 'alerts' | 'settings'

const ROUTES: { key: Route; label: string; shortLabel: string; icon: string }[] = [
  { key: 'terminal',  label: 'Terminal',         shortLabel: 'TML', icon: '⬡' },
  { key: 'scanner',   label: 'Market Scanner',   shortLabel: 'SCN', icon: '◈' },
  { key: 'strategy',  label: 'Strategy Builder', shortLabel: 'STR', icon: '⬢' },
  { key: 'research',  label: 'Research',         shortLabel: 'RES', icon: '⬟' },
  { key: 'alerts',    label: 'Alerts',           shortLabel: 'ALT', icon: '◎' },
  { key: 'settings',  label: 'Settings',         shortLabel: 'CFG', icon: '◉' },
]

// ============================================================
// Nav item component
// ============================================================

interface NavItemProps {
  route: typeof ROUTES[number]
  isActive: boolean
  badge?: number
  onClick: () => void
  collapsed: boolean
}

const NavItem: React.FC<NavItemProps> = ({ route, isActive, badge, onClick, collapsed }) => (
  <button
    onClick={onClick}
    title={collapsed ? route.label : undefined}
    className={`
      w-full flex items-center gap-2.5 px-2.5 py-2 rounded transition-colors relative group
      ${isActive
        ? 'bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/25'
        : 'text-terminal-subtle hover:text-terminal-text hover:bg-terminal-surface border border-transparent'
      }
    `}
  >
    <span className="text-sm flex-shrink-0">{route.icon}</span>
    {!collapsed && (
      <span className="text-xs font-mono truncate">{route.label}</span>
    )}
    {badge != null && badge > 0 && (
      <span className={`
        absolute flex items-center justify-center
        text-[9px] font-mono font-bold rounded-full min-w-[14px] h-[14px] px-0.5
        bg-terminal-bear text-white
        ${collapsed ? 'top-0.5 right-0.5' : 'right-2'}
      `}>
        {badge > 99 ? '99+' : badge}
      </span>
    )}
    {collapsed && (
      <span className="
        absolute left-full ml-2 px-2 py-1 bg-terminal-surface border border-terminal-border
        rounded text-xs font-mono text-terminal-text whitespace-nowrap
        opacity-0 group-hover:opacity-100 pointer-events-none z-50
        transition-opacity
      ">
        {route.label}
      </span>
    )}
  </button>
)

// ============================================================
// Notification bell component
// ============================================================

const NotificationBell: React.FC<{ count: number; collapsed: boolean }> = ({ count, collapsed }) => (
  <div
    className={`flex items-center gap-2 px-2.5 py-1.5 text-xs font-mono ${count > 0 ? 'text-terminal-warning' : 'text-terminal-muted'}`}
  >
    <span className="flex-shrink-0">{count > 0 ? '🔔' : '🔕'}</span>
    {!collapsed && <span>{count > 0 ? `${count} alert${count !== 1 ? 's' : ''}` : 'No alerts'}</span>}
  </div>
)

// ============================================================
// Status bar component
// ============================================================

const StatusBar: React.FC = () => {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  const bhFormations = useBHStore((s) => s.formationEvents.filter((e) => !e.acknowledged).length)
  const alerts = useSettingsStore((s) => s.alerts.filter((a) => !a.acknowledged).length)

  return (
    <div className="flex items-center justify-between px-3 py-1 border-t border-terminal-border bg-terminal-surface/50 flex-shrink-0">
      <div className="flex items-center gap-3">
        <span className="text-[9px] font-mono text-terminal-muted">SRFM v0.1.0</span>
        {bhFormations > 0 && (
          <span className="text-[9px] font-mono text-terminal-warning animate-pulse">
            {bhFormations} BH formation{bhFormations !== 1 ? 's' : ''}
          </span>
        )}
        {alerts > 0 && (
          <span className="text-[9px] font-mono text-terminal-bear">
            {alerts} unacknowledged alert{alerts !== 1 ? 's' : ''}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2">
        <span className="text-[9px] font-mono text-terminal-muted">
          {time.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
        </span>
        <span className="text-[9px] font-mono text-terminal-subtle">
          {time.toLocaleTimeString('en-US', { hour12: false })} ET
        </span>
      </div>
    </div>
  )
}

// ============================================================
// App root
// ============================================================

const App: React.FC = () => {
  const [route, setRoute] = useState<Route>('terminal')
  const [navCollapsed, setNavCollapsed] = useState(false)

  const unacknowledgedAlerts = useSettingsStore((s) => s.alerts.filter((a) => !a.acknowledged).length)
  const bhFormations = useBHStore((s) => s.formationEvents.filter((e) => !e.acknowledged).length)

  const navigate = useCallback((r: Route) => {
    setRoute(r)
  }, [])

  const getBadge = useCallback((route: Route): number | undefined => {
    if (route === 'alerts') return unacknowledgedAlerts + bhFormations || undefined
    return undefined
  }, [unacknowledgedAlerts, bhFormations])

  const renderPage = () => {
    switch (route) {
      case 'terminal':  return <Terminal />
      case 'scanner':   return <Scanner />
      case 'strategy':  return <StrategyBuilder />
      case 'research':  return <Research />
      case 'alerts':    return <Alerts />
      case 'settings':  return <Settings />
      default:          return <Terminal />
    }
  }

  return (
    <div className="flex flex-col h-full bg-terminal-bg overflow-hidden">
      {/* Top application bar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-terminal-surface border-b border-terminal-border flex-shrink-0 z-50">
        <div className="flex items-center gap-3">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <span className="text-terminal-accent font-mono font-bold text-base tracking-widest">SRFM</span>
            <span className="text-terminal-muted font-mono text-[10px]">TERMINAL</span>
          </div>

          {/* Nav tabs — horizontal for large screens */}
          <nav className="hidden md:flex items-center gap-0.5 ml-2">
            {ROUTES.map((r) => (
              <button
                key={r.key}
                onClick={() => navigate(r.key)}
                className={`
                  relative px-3 py-1.5 text-xs font-mono rounded transition-colors
                  ${route === r.key
                    ? 'text-terminal-text bg-terminal-accent/15 border border-terminal-accent/25'
                    : 'text-terminal-subtle hover:text-terminal-text hover:bg-terminal-surface'
                  }
                `}
              >
                {r.label}
                {(getBadge(r.key) ?? 0) > 0 && (
                  <span className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-terminal-bear text-white text-[8px] font-mono rounded-full flex items-center justify-center">
                    {getBadge(r.key)}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>

        {/* Right side: build info */}
        <div className="flex items-center gap-3">
          {unacknowledgedAlerts > 0 && (
            <button
              onClick={() => navigate('alerts')}
              className="text-[10px] font-mono text-terminal-bear border border-terminal-bear/30 bg-terminal-bear/10 rounded px-2 py-0.5 hover:bg-terminal-bear/20 transition-colors"
            >
              {unacknowledgedAlerts} alert{unacknowledgedAlerts !== 1 ? 's' : ''}
            </button>
          )}
          {bhFormations > 0 && (
            <button
              onClick={() => navigate('alerts')}
              className="text-[10px] font-mono text-terminal-warning border border-terminal-warning/30 bg-terminal-warning/10 rounded px-2 py-0.5 animate-pulse hover:animate-none hover:bg-terminal-warning/20 transition-colors"
            >
              ⚛ {bhFormations} formation{bhFormations !== 1 ? 's' : ''}
            </button>
          )}
          <button
            onClick={() => navigate('settings')}
            className={`text-[10px] font-mono px-2 py-1 rounded border transition-colors ${
              route === 'settings'
                ? 'border-terminal-accent/30 text-terminal-accent bg-terminal-accent/10'
                : 'border-terminal-border text-terminal-subtle hover:text-terminal-text'
            }`}
          >
            Settings
          </button>
        </div>
      </div>

      {/* Mobile nav — shown on small screens */}
      <div className="flex md:hidden border-b border-terminal-border bg-terminal-surface flex-shrink-0 overflow-x-auto no-scrollbar">
        {ROUTES.map((r) => (
          <button
            key={r.key}
            onClick={() => navigate(r.key)}
            className={`
              relative flex-shrink-0 px-3 py-2 text-[10px] font-mono transition-colors whitespace-nowrap
              ${route === r.key
                ? 'text-terminal-accent border-b-2 border-terminal-accent'
                : 'text-terminal-subtle hover:text-terminal-text'
              }
            `}
          >
            {r.shortLabel}
            {(getBadge(r.key) ?? 0) > 0 && (
              <span className="absolute top-1 right-0.5 w-3 h-3 bg-terminal-bear text-white text-[7px] rounded-full flex items-center justify-center">
                {getBadge(r.key)}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Main content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {renderPage()}
      </div>

      {/* Status bar */}
      <StatusBar />
    </div>
  )
}

export default App
