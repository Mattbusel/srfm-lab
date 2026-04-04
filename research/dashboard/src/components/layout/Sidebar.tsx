import React from 'react'
import { NavLink } from 'react-router-dom'
import { clsx } from 'clsx'
import {
  LayoutDashboard,
  GitCompare,
  TrendingUp,
  Radio,
  Brain,
  Briefcase,
  Shuffle,
  Zap,
  Activity,
} from 'lucide-react'

interface NavItem {
  path: string
  label: string
  icon: React.ReactNode
  description: string
}

const NAV_ITEMS: NavItem[] = [
  { path: '/', label: 'Overview', icon: <LayoutDashboard size={16} />, description: 'KPIs & summary' },
  { path: '/reconciliation', label: 'Reconciliation', icon: <GitCompare size={16} />, description: 'Live vs backtest' },
  { path: '/walk-forward', label: 'Walk Forward', icon: <TrendingUp size={16} />, description: 'IS/OOS analysis' },
  { path: '/signals', label: 'Signal Analytics', icon: <Radio size={16} />, description: 'IC & factor analysis' },
  { path: '/regimes', label: 'Regime Lab', icon: <Brain size={16} />, description: 'Regime detection' },
  { path: '/portfolio', label: 'Portfolio Lab', icon: <Briefcase size={16} />, description: 'Weights & frontier' },
  { path: '/mc-sim', label: 'MC Simulation', icon: <Shuffle size={16} />, description: '10K path simulation' },
  { path: '/stress-test', label: 'Stress Test', icon: <Zap size={16} />, description: 'Scenario analysis' },
]

interface SidebarProps {
  collapsed: boolean
}

export function Sidebar({ collapsed }: SidebarProps) {
  return (
    <aside className={clsx(
      'flex flex-col bg-research-surface border-r border-research-border transition-all duration-200',
      collapsed ? 'w-14' : 'w-56'
    )}>
      {/* Logo */}
      <div className={clsx(
        'flex items-center gap-2 px-3 py-4 border-b border-research-border',
        collapsed && 'justify-center'
      )}>
        <div className="w-7 h-7 rounded bg-research-accent flex items-center justify-center shrink-0">
          <Activity size={14} className="text-white" />
        </div>
        {!collapsed && (
          <div>
            <div className="text-xs font-bold text-research-text tracking-wider">SRFM</div>
            <div className="text-[10px] text-research-subtle uppercase tracking-widest">Research</div>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-2 overflow-y-auto">
        {NAV_ITEMS.map(item => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) => clsx(
              'flex items-center gap-3 px-3 py-2.5 mx-1 rounded transition-all text-sm group',
              collapsed && 'justify-center',
              isActive
                ? 'bg-research-accent/15 text-research-accent'
                : 'text-research-subtle hover:text-research-text hover:bg-research-muted/50'
            )}
            title={collapsed ? item.label : undefined}
          >
            <span className="shrink-0">{item.icon}</span>
            {!collapsed && (
              <div className="min-w-0">
                <div className="text-xs font-medium truncate">{item.label}</div>
                <div className="text-[10px] text-research-subtle/60 truncate hidden group-hover:block">
                  {item.description}
                </div>
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className={clsx(
        'px-3 py-3 border-t border-research-border',
        collapsed && 'flex justify-center'
      )}>
        {!collapsed && (
          <div className="text-[10px] text-research-subtle/50 font-mono">
            Research API :8766
            <br />
            Arena :8765
          </div>
        )}
      </div>
    </aside>
  )
}
