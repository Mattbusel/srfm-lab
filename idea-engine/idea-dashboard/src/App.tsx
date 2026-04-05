import React, { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import DashboardPage from './pages/DashboardPage'
import GenomesPage from './pages/GenomesPage'
import HypothesesPage from './pages/HypothesesPage'
import ShadowsPage from './pages/ShadowsPage'
import CounterfactualsPage from './pages/CounterfactualsPage'
import AcademicPage from './pages/AcademicPage'
import SerendipityPage from './pages/SerendipityPage'
import GenealogyPage from './pages/GenealogyPage'
import NarrativesPage from './pages/NarrativesPage'
import DebateChamber from './pages/DebateChamber'
import MacroRegime from './pages/MacroRegime'
import SentimentFeed from './pages/SentimentFeed'
import MicrostructureHealth from './pages/MicrostructureHealth'
import AlertBanner from './components/AlertBanner'
import ErrorBoundary from './components/ErrorBoundary'
import { useWebSocket } from './hooks/useWebSocket'
import { useIdeaStore, selectCriticalAlerts } from './store/ideaStore'
import { fetchAlerts, fetchHypotheses } from './api/client'
import { useAcknowledgeAlert } from './hooks/useAlerts'

// ─── Nav Items ────────────────────────────────────────────────────────────────

type RouteId =
  | 'dashboard'
  | 'genomes'
  | 'hypotheses'
  | 'shadows'
  | 'counterfactuals'
  | 'academic'
  | 'serendipity'
  | 'genealogy'
  | 'narratives'
  | 'debate'
  | 'macro'
  | 'sentiment'
  | 'microstructure'

interface NavItem {
  id: RouteId
  label: string
  icon: string
  badge?: number
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────

interface SidebarProps {
  active: RouteId
  onNavigate: (id: RouteId) => void
  collapsed: boolean
  onToggleCollapse: () => void
  wsStatus: 'connecting' | 'connected' | 'disconnected' | 'error'
  navItems: NavItem[]
}

const Sidebar: React.FC<SidebarProps> = ({
  active,
  onNavigate,
  collapsed,
  onToggleCollapse,
  wsStatus,
  navItems,
}) => {
  const { currentRegime, evolutionStats } = useIdeaStore()

  const regimeColors = {
    BULL:    { text: 'var(--green)', dot: '#22c55e' },
    BEAR:    { text: 'var(--red)',   dot: '#ef4444' },
    NEUTRAL: { text: 'var(--blue)',  dot: '#3b82f6' },
  }
  const rc = regimeColors[currentRegime]

  return (
    <nav className={clsx('sidebar', collapsed && 'collapsed')}>
      {/* Logo */}
      <div
        style={{
          padding: '14px 16px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          overflow: 'hidden',
          flexShrink: 0,
        }}
      >
        <span
          style={{
            fontSize: '1.25rem',
            color: 'var(--accent)',
            flexShrink: 0,
            lineHeight: 1,
          }}
        >
          ◈
        </span>
        {!collapsed && (
          <div>
            <div
              style={{
                fontWeight: 800,
                fontSize: '0.9375rem',
                letterSpacing: '-0.02em',
                color: 'var(--text-primary)',
                lineHeight: 1,
              }}
            >
              IAE
            </div>
            <div
              style={{
                fontSize: '0.65rem',
                color: 'var(--text-muted)',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              }}
            >
              Idea Engine
            </div>
          </div>
        )}
        <button
          className="btn-icon"
          onClick={onToggleCollapse}
          style={{ marginLeft: 'auto', flexShrink: 0, fontSize: '0.75rem' }}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? '›' : '‹'}
        </button>
      </div>

      {/* Navigation */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onNavigate(item.id)}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: collapsed ? '10px 16px' : '9px 16px',
              justifyContent: collapsed ? 'center' : undefined,
              background: active === item.id ? 'var(--bg-hover)' : 'transparent',
              color: active === item.id ? 'var(--accent)' : 'var(--text-muted)',
              fontSize: '0.875rem',
              fontWeight: active === item.id ? 600 : 400,
              transition: 'all var(--transition)',
              position: 'relative',
              overflow: 'hidden',
              borderRadius: 0,
              borderLeft: `2px solid ${active === item.id ? 'var(--accent)' : 'transparent'}`,
              borderTop: 'none',
              borderRight: 'none',
              borderBottom: 'none',
            }}
            onMouseEnter={(e) => {
              if (active !== item.id) {
                e.currentTarget.style.background = 'var(--bg-hover)'
                e.currentTarget.style.color = 'var(--text-primary)'
              }
            }}
            onMouseLeave={(e) => {
              if (active !== item.id) {
                e.currentTarget.style.background = 'transparent'
                e.currentTarget.style.color = 'var(--text-muted)'
              }
            }}
            title={collapsed ? item.label : undefined}
          >
            <span style={{ fontSize: '1rem', flexShrink: 0 }}>{item.icon}</span>
            {!collapsed && (
              <>
                <span style={{ flex: 1, textAlign: 'left', whiteSpace: 'nowrap' }}>
                  {item.label}
                </span>
                {item.badge !== undefined && item.badge > 0 && (
                  <span
                    style={{
                      background: 'var(--red)',
                      color: '#fff',
                      borderRadius: 10,
                      padding: '1px 5px',
                      fontSize: '0.65rem',
                      fontWeight: 700,
                      flexShrink: 0,
                    }}
                  >
                    {item.badge}
                  </span>
                )}
              </>
            )}
          </button>
        ))}
      </div>

      {/* Status Footer */}
      {!collapsed && (
        <div
          style={{
            padding: '12px 16px',
            borderTop: '1px solid var(--border)',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            flexShrink: 0,
          }}
        >
          {/* WS Status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem' }}>
            <span className={`ws-dot ${wsStatus}`} />
            <span style={{ color: 'var(--text-muted)' }}>
              {wsStatus === 'connected'
                ? 'Live'
                : wsStatus === 'connecting'
                ? 'Connecting…'
                : 'Offline'}
            </span>
          </div>

          {/* Regime */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem' }}>
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: rc.dot,
                flexShrink: 0,
              }}
            />
            <span style={{ color: 'var(--text-muted)' }}>Regime:</span>
            <span style={{ color: rc.text, fontWeight: 600 }}>{currentRegime}</span>
          </div>

          {/* Best Gen */}
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            Gen{' '}
            <span className="num" style={{ color: 'var(--text-secondary)' }}>
              {Math.max(
                evolutionStats.BULL.generation,
                evolutionStats.BEAR.generation,
                evolutionStats.NEUTRAL.generation
              ) || '—'}
            </span>
          </div>
        </div>
      )}
    </nav>
  )
}

// ─── Header ───────────────────────────────────────────────────────────────────

interface HeaderProps {
  activeRoute: RouteId
  wsStatus: 'connecting' | 'connected' | 'disconnected' | 'error'
  retryCount: number
}

const ROUTE_TITLES: Record<RouteId, string> = {
  dashboard:        'Dashboard',
  genomes:          'Genome Browser',
  hypotheses:       'Hypothesis Queue',
  shadows:          'Shadow Runners',
  counterfactuals:  'Counterfactual Analysis',
  academic:         'Academic Feed',
  serendipity:      'Serendipity Engine',
  genealogy:        'Genealogy',
  narratives:       'Research Narratives',
  debate:           'Debate Chamber',
  macro:            'Macro Regime',
  sentiment:        'Sentiment Feed',
  microstructure:   'Microstructure Health',
}

const Header: React.FC<HeaderProps> = ({ activeRoute, wsStatus, retryCount }) => {
  const now = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })
  const [time, setTime] = React.useState(now)

  useEffect(() => {
    const interval = setInterval(() => {
      setTime(new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }))
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div
      style={{
        height: 'var(--header-height)',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 20px',
        gap: 16,
        background: 'var(--bg-surface)',
        flexShrink: 0,
      }}
    >
      <span
        style={{
          fontWeight: 700,
          fontSize: '0.9375rem',
          color: 'var(--text-primary)',
        }}
      >
        {ROUTE_TITLES[activeRoute]}
      </span>
      <div style={{ flex: 1 }} />
      {wsStatus !== 'connected' && retryCount > 0 && (
        <span style={{ fontSize: '0.75rem', color: 'var(--yellow)' }}>
          Reconnecting (attempt {retryCount})…
        </span>
      )}
      {wsStatus === 'connected' && (
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 4 }}>
          <span className="ws-dot connected" />
          Live Feed
        </span>
      )}
      <span
        className="num"
        style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}
      >
        {time}
      </span>
    </div>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────

const App: React.FC = () => {
  const [activeRoute, setActiveRoute] = useState<RouteId>('dashboard')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const handleWsMessage = useIdeaStore((s) => s.handleWsMessage)
  const setWsConnected = useIdeaStore((s) => s.setWsConnected)
  const setAlerts = useIdeaStore((s) => s.setAlerts)
  const criticalAlerts = useIdeaStore(selectCriticalAlerts)

  // Seed store with initial alert data
  const { data: seedAlerts } = useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    refetchInterval: 15_000,
  })
  useEffect(() => {
    if (seedAlerts) setAlerts(seedAlerts)
  }, [seedAlerts, setAlerts])

  const { status: wsStatus, retryCount } = useWebSocket({
    url: 'ws://localhost:8767/ws',
    onMessage: handleWsMessage,
    onOpen: () => setWsConnected(true),
    onClose: () => setWsConnected(false),
  })

  const { mutate: ackAlert } = useAcknowledgeAlert()

  // Build nav items with dynamic badges
  const { data: alertData } = useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    select: (d) => d.filter((a) => !a.acknowledged).length,
  })
  const { data: hypothesesData } = useQuery({
    queryKey: ['hypotheses', 'all'],
    queryFn: () => fetchHypotheses(),
    select: (d: Awaited<ReturnType<typeof fetchHypotheses>>) => d.filter((h) => h.status === 'pending').length,
  })

  const NAV_ITEMS: NavItem[] = [
    { id: 'dashboard',       label: 'Dashboard',           icon: '⊞' },
    { id: 'genomes',         label: 'Genomes',              icon: '⬡' },
    { id: 'hypotheses',      label: 'Hypotheses',           icon: '⧖', badge: hypothesesData },
    { id: 'debate',          label: 'Debate Chamber',       icon: '⚖' },
    { id: 'shadows',         label: 'Shadow Runners',       icon: '◎' },
    { id: 'counterfactuals', label: 'Counterfactuals',      icon: '⊿' },
    { id: 'academic',        label: 'Academic',             icon: '⚘' },
    { id: 'serendipity',     label: 'Serendipity',          icon: '✦' },
    { id: 'genealogy',       label: 'Genealogy',            icon: '⟠' },
    { id: 'narratives',      label: 'Narratives',           icon: '≡', badge: alertData },
    { id: 'macro',           label: 'Macro Regime',         icon: '◈' },
    { id: 'sentiment',       label: 'Sentiment Feed',       icon: '◉' },
    { id: 'microstructure',  label: 'Microstructure',       icon: '⊙' },
  ]

  const renderPage = () => {
    switch (activeRoute) {
      case 'dashboard':       return <DashboardPage />
      case 'genomes':         return <GenomesPage />
      case 'hypotheses':      return <HypothesesPage />
      case 'debate':          return <DebateChamber />
      case 'shadows':         return <ShadowsPage />
      case 'counterfactuals': return <CounterfactualsPage />
      case 'academic':        return <AcademicPage />
      case 'serendipity':     return <SerendipityPage />
      case 'genealogy':       return <GenealogyPage />
      case 'narratives':      return <NarrativesPage />
      case 'macro':           return <MacroRegime />
      case 'sentiment':       return <SentimentFeed />
      case 'microstructure':  return <MicrostructureHealth />
      default:                return <DashboardPage />
    }
  }

  return (
    <div className="app-layout">
      <Sidebar
        active={activeRoute}
        onNavigate={setActiveRoute}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
        wsStatus={wsStatus}
        navItems={NAV_ITEMS}
      />
      <div className="main-area">
        <Header
          activeRoute={activeRoute}
          wsStatus={wsStatus}
          retryCount={retryCount}
        />
        {/* Critical alert banner */}
        <AlertBanner
          alerts={criticalAlerts}
          onAcknowledge={ackAlert}
        />
        <main className="page-content">
          <ErrorBoundary key={activeRoute}>
            {renderPage()}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}

export default App
