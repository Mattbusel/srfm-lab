import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { StressTestChart } from '@/components/charts/StressTestChart'
import { EquityCurveChart } from '@/components/charts/EquityCurveChart'
import { fetchStressScenarios } from '@/api/regimes'
import { generateMockEquity } from '@/api/mockData'
import { REGIME_COLORS, REGIME_LABELS } from '@/utils/colors'
import { formatCurrency, formatPct, formatDate } from '@/utils/formatters'
import { clsx } from 'clsx'
import type { StressScenario } from '@/types/regimes'
import type { RegimeType } from '@/types/trades'

const CATEGORY_LABELS: Record<string, string> = {
  market_crash: 'Market Crash',
  vol_spike: 'Vol Spike',
  liquidity: 'Liquidity',
  correlation: 'Correlation',
  tail: 'Tail Risk',
}

const CATEGORY_COLORS: Record<string, string> = {
  market_crash: '#ef4444',
  vol_spike: '#f59e0b',
  liquidity: '#8b5cf6',
  correlation: '#06b6d4',
  tail: '#ec4899',
}

function ScenarioTable({
  scenarios,
  selected,
  onSelect,
}: {
  scenarios: StressScenario[]
  selected: string | null
  onSelect: (id: string) => void
}) {
  const sorted = [...scenarios].sort((a, b) => a.pnlImpact - b.pnlImpact)

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr className="border-b border-research-border">
            {['Scenario', 'Category', 'Period', 'P&L Impact', 'Impact %', 'Max DD', 'Recovery', 'Prob', 'Regimes'].map(h => (
              <th key={h} className="text-left text-[10px] text-research-subtle uppercase tracking-wide py-2 px-3 whitespace-nowrap">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map(s => (
            <tr
              key={s.id}
              onClick={() => onSelect(s.id)}
              className={clsx(
                'cursor-pointer border-b border-research-border/50 transition-colors',
                selected === s.id ? 'bg-research-accent/10' : 'hover:bg-research-muted/20'
              )}
            >
              <td className="py-2 px-3 font-mono font-semibold text-research-text">{s.name}</td>
              <td className="py-2 px-3">
                <span
                  className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                  style={{ backgroundColor: `${CATEGORY_COLORS[s.category]}22`, color: CATEGORY_COLORS[s.category], border: `1px solid ${CATEGORY_COLORS[s.category]}44` }}
                >
                  {CATEGORY_LABELS[s.category]}
                </span>
              </td>
              <td className="py-2 px-3 font-mono text-research-subtle text-[10px]">
                {s.startDate} → {s.endDate}
              </td>
              <td className={clsx('py-2 px-3 font-mono font-semibold', s.pnlImpact >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                {formatCurrency(s.pnlImpact, { sign: true })}
              </td>
              <td className={clsx('py-2 px-3 font-mono', s.pnlImpactPct >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                {formatPct(s.pnlImpactPct, { sign: true })}
              </td>
              <td className="py-2 px-3 font-mono text-research-bear">
                {formatPct(s.maxDrawdown * 100)}
              </td>
              <td className="py-2 px-3 font-mono text-research-subtle">
                {s.recoveryDays !== null ? `${s.recoveryDays}d` : 'N/A'}
              </td>
              <td className="py-2 px-3 font-mono text-research-subtle">
                {formatPct(s.probability * 100, { decimals: 0 })}
              </td>
              <td className="py-2 px-3">
                <div className="flex gap-0.5 flex-wrap">
                  {s.regimeSequence.map((r, i) => (
                    <span
                      key={i}
                      className="text-[9px] px-1 py-0.5 rounded font-mono"
                      style={{ backgroundColor: `${REGIME_COLORS[r as RegimeType]}22`, color: REGIME_COLORS[r as RegimeType] }}
                    >
                      {REGIME_LABELS[r as RegimeType]}
                    </span>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Generate a "stressed" equity curve for a scenario
function generateStressedEquity(scenario: StressScenario) {
  const normal = generateMockEquity(60, 100_000)
  const stressStart = Math.floor(normal.length * 0.4)
  const stressDuration = 10

  return normal.map((pt, i) => {
    if (i >= stressStart && i < stressStart + stressDuration) {
      const progress = (i - stressStart) / stressDuration
      const stressFactor = 1 + (scenario.pnlImpactPct / 100) * (progress)
      return { ...pt, equity: pt.equity * stressFactor }
    }
    if (i >= stressStart + stressDuration && scenario.recoveryDays) {
      const recProgress = Math.min(1, (i - stressStart - stressDuration) / (scenario.recoveryDays * 0.5))
      const impactFactor = 1 + (scenario.pnlImpactPct / 100)
      const recoveryFactor = impactFactor + (1 - impactFactor) * recProgress
      return { ...pt, equity: pt.equity * recoveryFactor }
    }
    return pt
  })
}

export function StressTestPage() {
  const [selected, setSelected] = useState<string | null>(null)
  const scenariosQ = useQuery({ queryKey: ['stress-scenarios'], queryFn: fetchStressScenarios })

  const scenarios = scenariosQ.data ?? []
  const selectedScenario = scenarios.find(s => s.id === selected)
  const stressedEquity = selectedScenario ? generateStressedEquity(selectedScenario) : null

  // Summary stats
  const worstCase = scenarios.length > 0 ? Math.min(...scenarios.map(s => s.pnlImpact)) : 0
  const totalExpLoss = scenarios.reduce((a, s) => a + s.pnlImpact * s.probability, 0)

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Summary row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-research-card border border-research-border/30 rounded-lg p-4">
          <div className="text-xs text-research-subtle mb-1">Scenarios Analyzed</div>
          <div className="text-2xl font-bold font-mono text-research-text">{scenarios.length}</div>
        </div>
        <div className="bg-research-card border border-research-bear/30 rounded-lg p-4">
          <div className="text-xs text-research-subtle mb-1">Worst Case P&L</div>
          <div className="text-2xl font-bold font-mono text-research-bear">
            {formatCurrency(worstCase, { compact: true })}
          </div>
        </div>
        <div className="bg-research-card border border-research-warning/30 rounded-lg p-4">
          <div className="text-xs text-research-subtle mb-1">Expected Loss (prob-weighted)</div>
          <div className="text-2xl font-bold font-mono text-research-warning">
            {formatCurrency(totalExpLoss, { compact: true })}
          </div>
        </div>
        <div className="bg-research-card border border-research-border/30 rounded-lg p-4">
          <div className="text-xs text-research-subtle mb-1">Max DD (worst scenario)</div>
          <div className="text-2xl font-bold font-mono text-research-bear">
            {scenarios.length > 0 ? formatPct(Math.max(...scenarios.map(s => s.maxDrawdown)) * 100) : '–'}
          </div>
        </div>
      </div>

      {/* Bar chart of scenario impacts */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <h2 className="text-sm font-semibold text-research-text mb-3">Scenario P&L Impact Ranked</h2>
        {scenariosQ.isLoading ? <LoadingSpinner size="sm" /> :
          <StressTestChart scenarios={scenarios} height={280} />}
      </div>

      {/* Scenario table */}
      <div className="bg-research-card border border-research-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-research-text">Scenario Results</h2>
          <span className="text-xs text-research-subtle">Click row to inspect equity behavior</span>
        </div>
        {scenariosQ.isLoading ? <LoadingSpinner size="sm" /> :
          <ScenarioTable scenarios={scenarios} selected={selected} onSelect={setSelected} />}
      </div>

      {/* Historical equity behavior for selected scenario */}
      {selectedScenario && stressedEquity && (
        <div className="bg-research-card border border-research-border rounded-lg p-4 animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-research-text">
              Equity Behavior: {selectedScenario.name}
            </h2>
            <div className="flex gap-2">
              <span
                className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                style={{ backgroundColor: `${CATEGORY_COLORS[selectedScenario.category]}22`, color: CATEGORY_COLORS[selectedScenario.category] }}
              >
                {CATEGORY_LABELS[selectedScenario.category]}
              </span>
              <span className="text-xs text-research-subtle font-mono">
                {selectedScenario.startDate} → {selectedScenario.endDate}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="text-center">
              <div className="text-xs text-research-subtle mb-1">P&L Impact</div>
              <div className={clsx('text-lg font-bold font-mono', selectedScenario.pnlImpact >= 0 ? 'text-research-bull' : 'text-research-bear')}>
                {formatCurrency(selectedScenario.pnlImpact, { sign: true })}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-research-subtle mb-1">Max Drawdown</div>
              <div className="text-lg font-bold font-mono text-research-bear">
                {formatPct(selectedScenario.maxDrawdown * 100)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-research-subtle mb-1">Recovery Days</div>
              <div className="text-lg font-bold font-mono text-research-text">
                {selectedScenario.recoveryDays !== null ? `${selectedScenario.recoveryDays}d` : 'N/A'}
              </div>
            </div>
          </div>

          <p className="text-xs text-research-subtle mb-3 font-mono">{selectedScenario.description}</p>
          <EquityCurveChart data={stressedEquity} height={200} />
        </div>
      )}
    </div>
  )
}
