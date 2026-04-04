import { NavLink, useLocation } from 'react-router-dom';
import { useLiveState } from '../hooks/useLiveState';

interface NavItem {
  to: string;
  label: string;
  icon: React.ReactNode;
}

function IconDashboard() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z"
      />
    </svg>
  );
}

function IconBacktest() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z"
      />
    </svg>
  );
}

function IconMC() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6"
      />
    </svg>
  );
}

function IconSensitivity() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M6.429 9.75L2.25 12l4.179 2.25m0-4.5l5.571 3 5.571-3m-11.142 0L2.25 7.5 12 2.25l9.75 5.25-4.179 2.25m0 0L21.75 12l-4.179 2.25m0 0l4.179 2.25L12 21.75 2.25 16.5l4.179-2.25m11.142 0l-5.571 3-5.571-3"
      />
    </svg>
  );
}

function IconCorrelation() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5"
      />
    </svg>
  );
}

function IconArchaeology() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 6.75c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125"
      />
    </svg>
  );
}

function IconReplay() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z"
      />
    </svg>
  );
}

function IconReports() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
      />
    </svg>
  );
}

const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Dashboard', icon: <IconDashboard /> },
  { to: '/backtest', label: 'Backtest', icon: <IconBacktest /> },
  { to: '/montecarlo', label: 'Monte Carlo', icon: <IconMC /> },
  { to: '/sensitivity', label: 'Sensitivity', icon: <IconSensitivity /> },
  { to: '/correlation', label: 'Correlation', icon: <IconCorrelation /> },
  { to: '/archaeology', label: 'Archaeology', icon: <IconArchaeology /> },
  { to: '/replay', label: 'Replay', icon: <IconReplay /> },
  { to: '/reports', label: 'Reports', icon: <IconReports /> },
];

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const { wsStatus, latest, pnlToday, activePositions } = useLiveState();
  const location = useLocation();

  const pageName = NAV_ITEMS.find(n => n.to === location.pathname)?.label ?? 'Spacetime Arena';

  const wsIndicator = wsStatus === 'open'
    ? <span className="w-2 h-2 rounded-full bg-bull animate-pulse" title="Live" />
    : wsStatus === 'connecting'
    ? <span className="w-2 h-2 rounded-full bg-bh-warm animate-pulse" title="Connecting" />
    : <span className="w-2 h-2 rounded-full bg-bear" title="Disconnected" />;

  return (
    <div className="flex h-screen bg-bg-base text-gray-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 bg-bg-card border-r border-bg-border flex flex-col">
        {/* Logo */}
        <div className="px-4 py-5 border-b border-bg-border">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-full bg-accent/20 border border-accent/50 flex items-center justify-center">
              <svg className="w-4 h-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <circle cx="12" cy="12" r="3" />
                <path strokeLinecap="round" d="M12 2v2M12 20v2M2 12h2M20 12h2" />
              </svg>
            </div>
            <div>
              <p className="text-xs font-mono font-bold text-gray-100 leading-none">SPACETIME</p>
              <p className="text-xs font-mono text-gray-500 leading-none">ARENA</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-3 space-y-0.5 px-2 overflow-y-auto">
          {NAV_ITEMS.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-mono transition-all duration-150
                 ${isActive
                   ? 'bg-accent/20 text-accent border border-accent/30'
                   : 'text-gray-400 hover:text-gray-200 hover:bg-bg-elevated border border-transparent'
                 }`
              }
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
        </nav>

        {/* Live status footer */}
        <div className="p-3 border-t border-bg-border space-y-1.5">
          <div className="flex items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-1.5 text-gray-500">
              {wsIndicator}
              <span>{wsStatus === 'open' ? 'Live' : wsStatus}</span>
            </div>
            <span className="text-gray-500">{activePositions} pos</span>
          </div>
          {latest && (
            <div className="text-xs font-mono text-gray-400">
              <span className="text-gray-500">Equity </span>
              <span className="text-gray-200">
                ${(latest.equity / 1000).toFixed(1)}K
              </span>
              <span className={`ml-1 ${pnlToday >= 0 ? 'text-bull' : 'text-bear'}`}>
                {pnlToday >= 0 ? '+' : ''}{(pnlToday / 1000).toFixed(1)}K
              </span>
            </div>
          )}
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-12 border-b border-bg-border bg-bg-card px-6 flex items-center justify-between flex-shrink-0">
          <h1 className="text-sm font-mono font-semibold text-gray-300 uppercase tracking-wider">
            {pageName}
          </h1>
          <div className="flex items-center gap-4 text-xs font-mono text-gray-500">
            <span>{new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
            <span className="text-gray-700">SRFM v2.0</span>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}
