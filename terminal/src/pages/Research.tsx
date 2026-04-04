// ============================================================
// Research — strategy analysis dashboard
// ============================================================
import React, { useState, useCallback } from 'react'
import { useStrategyStore } from '@/store/strategyStore'
import { EquityTerminal } from '@/components/charts/EquityTerminal'
import { HeatmapChart } from '@/components/charts/HeatmapChart'
import { MassTimeline } from '@/components/bh/MassTimeline'
import { RegimeMap } from '@/components/bh/RegimeMap'
import { useBacktest } from '@/hooks/useBacktest'
import { BacktestPanel } from '@/components/strategy-builder/BacktestPanel'

type ResearchTab = 'performance' | 'bh_analysis' | 'backtest' | 'factors'

export const Research: React.FC = () => {
  const graphs = useStrategyStore((s) => s.graphs)
  const [selectedGraphId, setSelectedGraphId] = useState<string | null>(graphs[0]?.id ?? null)
  const [activeTab, setActiveTab] = useState<ResearchTab>('performance')

  const TABS: { key: ResearchTab; label: string }[] = [
    { key: 'performance', label: 'Performance' },
    { key: 'bh_analysis', label: 'BH Analysis' },
    { key: 'backtest', label: 'Backtesting' },
    { key: 'factors', label: 'Factor Analysis' },
  ]

  return (
    <div className="flex h-full bg-terminal-bg">
      {/* Left: strategy selector */}
      <div className="w-52 flex-shrink-0 border-r border-terminal-border flex flex-col">
        <div className="px-3 py-2 border-b border-terminal-border">
          <span className="text-terminal-subtle text-xs font-mono uppercase">Research</span>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div
            onClick={() => setSelectedGraphId(null)}
            className={`px-2 py-1.5 rounded cursor-pointer text-xs font-mono transition-colors ${!selectedGraphId ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text hover:bg-terminal-surface'}`}
          >
            Portfolio Overview
          </div>
          {graphs.map((g) => (
            <div
              key={g.id}
              onClick={() => setSelectedGraphId(g.id)}
              className={`px-2 py-1.5 rounded cursor-pointer text-xs font-mono transition-colors ${selectedGraphId === g.id ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text hover:bg-terminal-surface'}`}
            >
              {g.metadata.name}
              <div className="text-[9px] text-terminal-muted">{g.nodes.length} nodes</div>
            </div>
          ))}
          {graphs.length === 0 && (
            <div className="text-[10px] font-mono text-terminal-muted text-center py-4">
              No strategies. Create one in the Strategy Builder.
            </div>
          )}
        </div>
      </div>

      {/* Right: analysis panels */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Tab bar */}
        <div className="flex border-b border-terminal-border flex-shrink-0">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-4 py-2 text-xs font-mono transition-colors ${
                activeTab === tab.key
                  ? 'text-terminal-text border-b-2 border-terminal-accent'
                  : 'text-terminal-subtle hover:text-terminal-text'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {activeTab === 'performance' && (
            <div className="space-y-4">
              <div className="bg-terminal-surface rounded border border-terminal-border overflow-hidden">
                <div className="px-3 py-2 border-b border-terminal-border">
                  <span className="text-terminal-text font-mono text-sm font-semibold">Portfolio Equity Curve</span>
                </div>
                <EquityTerminal height={300} showBenchmark showDrawdown />
              </div>
              <div className="bg-terminal-surface rounded border border-terminal-border overflow-hidden">
                <div className="px-3 py-2 border-b border-terminal-border">
                  <span className="text-terminal-text font-mono text-sm font-semibold">Daily Returns Calendar</span>
                </div>
                <HeatmapChart />
              </div>
            </div>
          )}

          {activeTab === 'bh_analysis' && (
            <div className="space-y-4">
              <div className="bg-terminal-surface rounded border border-terminal-border overflow-hidden">
                <MassTimeline height={250} />
              </div>
              <div className="bg-terminal-surface rounded border border-terminal-border overflow-hidden">
                <RegimeMap height={280} />
              </div>
            </div>
          )}

          {activeTab === 'backtest' && selectedGraphId && (
            <div className="h-full max-h-[800px]">
              <div className="bg-terminal-surface rounded border border-terminal-border h-full overflow-hidden">
                <BacktestPanel graphId={selectedGraphId} />
              </div>
            </div>
          )}

          {activeTab === 'backtest' && !selectedGraphId && (
            <div className="flex items-center justify-center h-64 text-terminal-subtle text-sm">
              Select a strategy from the left panel to run backtests
            </div>
          )}

          {activeTab === 'factors' && (
            <div className="flex items-center justify-center h-64 text-terminal-subtle text-sm">
              <div className="text-center">
                <div className="text-2xl mb-2">🔬</div>
                <div>Select a strategy to view factor exposure analysis</div>
                <div className="text-xs mt-1 text-terminal-muted">Coming soon: Fama-French factor loadings, regime attribution</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Research
