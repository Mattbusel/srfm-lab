// ============================================================
// AccountSummary — equity, cash, margin, P&L, risk
// ============================================================
import React, { useMemo } from 'react'
import { usePortfolioStore, selectPortfolioConcentration } from '@/store/portfolioStore'
import { useSettingsStore } from '@/store/settingsStore'

function MetricCard({ label, value, subValue, color = 'text-terminal-text', className = '' }: {
  label: string
  value: string
  subValue?: string
  color?: string
  className?: string
}) {
  return (
    <div className={`flex flex-col ${className}`}>
      <span className="text-[9px] font-mono text-terminal-subtle uppercase tracking-wider">{label}</span>
      <span className={`font-mono text-sm font-semibold ${color}`}>{value}</span>
      {subValue && <span className="font-mono text-[10px] text-terminal-subtle">{subValue}</span>}
    </div>
  )
}

function ProgressBar({ value, max, color, label, showPct = true }: {
  value: number
  max: number
  color: string
  label: string
  showPct?: boolean
}) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
  return (
    <div>
      <div className="flex justify-between text-[9px] font-mono mb-0.5">
        <span className="text-terminal-subtle">{label}</span>
        {showPct && <span className="text-terminal-subtle">{pct.toFixed(0)}%</span>}
      </div>
      <div className="h-1 bg-terminal-muted rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

export const AccountSummary: React.FC<{ className?: string }> = ({ className = '' }) => {
  const account = usePortfolioStore((s) => s.account)
  const dailyTarget = usePortfolioStore((s) => s.dailyPnlTarget)
  const concentration = usePortfolioStore(selectPortfolioConcentration)

  const fmt = useMemo(() => ({
    currency: (v: number) => `$${Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + 'K' : v.toFixed(2)}`,
    pct: (v: number) => (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%',
    signed: (v: number) => (v >= 0 ? '+$' : '-$') + Math.abs(v).toFixed(2),
  }), [])

  if (!account) {
    return (
      <div className={`flex items-center justify-center bg-terminal-bg ${className}`}>
        <div className="text-terminal-subtle text-sm">Loading account...</div>
      </div>
    )
  }

  const marginUsedPct = account.initialMargin > 0 ? account.marginUsed / account.initialMargin : 0
  const dayPnlColor = account.dayPnl >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'
  const totalPnlColor = account.totalPnl >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'

  const topConcentration = [...concentration]
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 5)

  return (
    <div className={`flex flex-col bg-terminal-bg h-full ${className}`}>
      {/* Header */}
      <div className="px-3 py-2 border-b border-terminal-border flex-shrink-0">
        <span className="text-terminal-subtle text-xs font-mono uppercase tracking-wider">Account</span>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Equity & Cash row */}
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            label="Portfolio Value"
            value={fmt.currency(account.portfolioValue)}
            subValue={fmt.pct(account.totalPnlPct)}
            color={totalPnlColor}
          />
          <MetricCard
            label="Cash"
            value={fmt.currency(account.cash)}
            subValue="Available"
            color="text-terminal-text"
          />
        </div>

        {/* Day P&L */}
        <div className="bg-terminal-surface rounded p-2.5 border border-terminal-border">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[10px] font-mono text-terminal-subtle uppercase">Day P&L</span>
            <div className={`font-mono text-sm font-bold ${dayPnlColor}`}>
              {fmt.signed(account.dayPnl)}
            </div>
          </div>
          <div className="flex items-center justify-between mb-1">
            <span className={`font-mono text-xs ${dayPnlColor}`}>{fmt.pct(account.dayPnlPct)}</span>
            {dailyTarget && (
              <span className="text-[10px] font-mono text-terminal-subtle">
                Target: {fmt.currency(dailyTarget.target)}
              </span>
            )}
          </div>
          {dailyTarget && (
            <ProgressBar
              value={Math.max(account.dayPnl, 0)}
              max={dailyTarget.target}
              color={dailyTarget.achieved >= dailyTarget.target ? '#22c55e' : '#3b82f6'}
              label="Daily target progress"
            />
          )}
        </div>

        {/* Buying Power */}
        <div className="bg-terminal-surface rounded p-2.5 border border-terminal-border space-y-1.5">
          <span className="text-[10px] font-mono text-terminal-subtle uppercase block">Buying Power</span>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="Available"
              value={fmt.currency(account.buyingPower)}
              color="text-terminal-text"
            />
            <MetricCard
              label="DT Buying Power"
              value={fmt.currency(account.daytradingBuyingPower)}
              color="text-terminal-info"
            />
          </div>
          <ProgressBar
            value={account.marginUsed}
            max={account.initialMargin || account.marginUsed + account.marginAvailable}
            color={marginUsedPct > 0.8 ? '#ef4444' : marginUsedPct > 0.5 ? '#f59e0b' : '#3b82f6'}
            label="Margin used"
          />
          <div className="flex justify-between text-[9px] font-mono text-terminal-subtle">
            <span>Used: {fmt.currency(account.marginUsed)}</span>
            <span>Available: {fmt.currency(account.marginAvailable)}</span>
          </div>
        </div>

        {/* Concentration */}
        {topConcentration.length > 0 && (
          <div className="bg-terminal-surface rounded p-2.5 border border-terminal-border">
            <span className="text-[10px] font-mono text-terminal-subtle uppercase block mb-2">Top Holdings</span>
            <div className="space-y-1.5">
              {topConcentration.map((c) => (
                <div key={c.symbol} className="flex items-center gap-2">
                  <span className="font-mono text-[10px] text-terminal-text w-12 flex-shrink-0">{c.symbol}</span>
                  <div className="flex-1 h-1.5 bg-terminal-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-terminal-accent rounded-full"
                      style={{ width: `${c.weight * 100}%` }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-terminal-subtle w-10 text-right">
                    {(c.weight * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Risk metrics */}
        <div className="grid grid-cols-2 gap-2">
          {account.patternDayTrader && (
            <div className="col-span-2 bg-terminal-warning/10 border border-terminal-warning/30 rounded px-2 py-1 text-[10px] font-mono text-terminal-warning">
              Pattern Day Trader — {account.daytradingCount ?? 0} DT trades
            </div>
          )}
          {account.tradingBlocked && (
            <div className="col-span-2 bg-terminal-bear/10 border border-terminal-bear/30 rounded px-2 py-1 text-[10px] font-mono text-terminal-bear">
              Trading is currently BLOCKED
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default AccountSummary
