import React, { useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'

const PAGE_TITLES: Record<string, string> = {
  '/': 'Overview',
  '/reconciliation': 'Reconciliation — Live vs Backtest',
  '/walk-forward': 'Walk-Forward Analysis',
  '/signals': 'Signal Analytics',
  '/regimes': 'Regime Lab',
  '/portfolio': 'Portfolio Lab',
  '/mc-sim': 'Monte Carlo Simulation',
  '/stress-test': 'Stress Test',
}

export function Layout() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()
  const title = PAGE_TITLES[location.pathname] ?? 'Research'

  return (
    <div className="flex h-screen overflow-hidden bg-research-bg">
      <Sidebar collapsed={collapsed} />

      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <TopBar
          onToggleSidebar={() => setCollapsed(c => !c)}
          title={title}
        />

        <main className="flex-1 overflow-y-auto p-4 md:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
