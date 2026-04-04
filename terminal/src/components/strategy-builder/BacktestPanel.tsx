// ============================================================
// BacktestPanel — backtest config and results
// ============================================================
import React, { useState, useCallback } from 'react'
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { format } from 'date-fns'
import { useBacktest } from '@/hooks/useBacktest'
import type { BacktestConfig, BacktestResult, BacktestMetrics } from '@/types'

interface BacktestPanelProps {
  graphId: string
  className?: string
}

const DEFAULT_CONFIG: BacktestConfig = {
  symbol: 'SPY',
  startDate: '2020-01-01',
  endDate: '2024-01-01',
  initialCapital: 100000,
  commission: 0.001,
  slippage: 0.0005,
  interval: '1d',
  allowShort: false,
  benchmarkSymbol: 'SPY',
}

function MetricRow({ label, value, color = '' }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between items-center py-0.5">
      <span className="text-[10px] font-mono text-terminal-subtle">{label}</span>
      <span className={`text-[10px] font-mono font-medium ${color || 'text-terminal-text'}`}>{value}</span>
    </div>
  )
}

function MetricsGrid({ metrics }: { metrics: BacktestMetrics }) {
  const pct = (v: number) => (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%'
  const currency = (v: number) => '$' + Math.abs(v).toFixed(2)
  const num = (v: number, dp = 2) => v.toFixed(dp)

  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-0">
      <div>
        <div className="text-[9px] font-mono text-terminal-subtle uppercase mb-1">Returns</div>
        <MetricRow label="Total Return" value={pct(metrics.totalReturnPct)} color={metrics.totalReturnPct >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'} />
        <MetricRow label="Annualized" value={pct(metrics.annualizedReturn)} color={metrics.annualizedReturn >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'} />
        <MetricRow label="Final Equity" value={`$${metrics.finalCapital.toFixed(0)}`} />
        <MetricRow label="Commission" value={currency(metrics.totalCommission)} />
      </div>
      <div>
        <div className="text-[9px] font-mono text-terminal-subtle uppercase mb-1">Risk</div>
        <MetricRow label="Sharpe" value={num(metrics.sharpe)} color={metrics.sharpe > 1 ? 'text-terminal-bull' : metrics.sharpe < 0 ? 'text-terminal-bear' : ''} />
        <MetricRow label="Sortino" value={num(metrics.sortino)} />
        <MetricRow label="Calmar" value={num(metrics.calmar)} />
        <MetricRow label="Max DD" value={pct(metrics.maxDrawdownPct)} color="text-terminal-bear" />
      </div>
      <div className="mt-2">
        <div className="text-[9px] font-mono text-terminal-subtle uppercase mb-1">Trades</div>
        <MetricRow label="Total" value={String(metrics.numTrades)} />
        <MetricRow label="Win Rate" value={pct(metrics.winRate)} color={metrics.winRate >= 0.5 ? 'text-terminal-bull' : 'text-terminal-bear'} />
        <MetricRow label="Profit Factor" value={isFinite(metrics.profitFactor) ? num(metrics.profitFactor) : '∞'} />
        <MetricRow label="Expectancy" value={`$${metrics.expectancy.toFixed(2)}`} color={metrics.expectancy >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'} />
      </div>
      <div className="mt-2">
        <div className="text-[9px] font-mono text-terminal-subtle uppercase mb-1">Trade Stats</div>
        <MetricRow label="Avg Win" value={`$${metrics.avgWin.toFixed(2)}`} color="text-terminal-bull" />
        <MetricRow label="Avg Loss" value={`$${metrics.avgLoss.toFixed(2)}`} color="text-terminal-bear" />
        <MetricRow label="Best Trade" value={`$${metrics.bestTrade.toFixed(2)}`} />
        <MetricRow label="Avg Hold" value={`${metrics.avgHoldingPeriod.toFixed(1)} bars`} />
      </div>
    </div>
  )
}

export const BacktestPanel: React.FC<BacktestPanelProps> = ({ graphId, className = '' }) => {
  const {
    isRunning, progress, error, results, activeResult,
    runBacktest, cancelBacktest, selectResult, deleteResult, exportResult, exportTradesToCsv,
  } = useBacktest(graphId)

  const [config, setConfig] = useState<BacktestConfig>(DEFAULT_CONFIG)
  const [tab, setTab] = useState<'config' | 'results' | 'trades'>('config')

  const handleRun = useCallback(async () => {
    await runBacktest(config)
    setTab('results')
  }, [config, runBacktest])

  const updateConfig = useCallback(<K extends keyof BacktestConfig>(key: K, value: BacktestConfig[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }))
  }, [])

  const chartData = activeResult?.equityCurve.map((p) => ({
    time: p.time * 1000,
    equity: p.equity,
    drawdownPct: p.drawdownPct,
  })) ?? []

  return (
    <div className={`flex flex-col bg-terminal-bg h-full ${className}`}>
      {/* Tabs */}
      <div className="flex border-b border-terminal-border flex-shrink-0">
        {(['config', 'results', 'trades'] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 py-1.5 text-[11px] font-mono capitalize transition-colors ${
              tab === t ? 'text-terminal-text border-b-2 border-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text'
            }`}
          >
            {t === 'results' && results.length > 0 ? `Results (${results.length})` : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'config' && (
        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] font-mono text-terminal-subtle uppercase block mb-0.5">Symbol</label>
              <input type="text" value={config.symbol}
                onChange={(e) => updateConfig('symbol', e.target.value.toUpperCase())}
                className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
              />
            </div>
            <div>
              <label className="text-[10px] font-mono text-terminal-subtle uppercase block mb-0.5">Interval</label>
              <select value={config.interval} onChange={(e) => updateConfig('interval', e.target.value)}
                className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
              >
                {['1m', '5m', '15m', '1h', '4h', '1d'].map((iv) => <option key={iv} value={iv}>{iv}</option>)}
              </select>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] font-mono text-terminal-subtle uppercase block mb-0.5">Start Date</label>
              <input type="date" value={config.startDate} onChange={(e) => updateConfig('startDate', e.target.value)}
                className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
              />
            </div>
            <div>
              <label className="text-[10px] font-mono text-terminal-subtle uppercase block mb-0.5">End Date</label>
              <input type="date" value={config.endDate} onChange={(e) => updateConfig('endDate', e.target.value)}
                className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
              />
            </div>
          </div>
          <div>
            <label className="text-[10px] font-mono text-terminal-subtle uppercase block mb-0.5">Initial Capital ($)</label>
            <input type="number" value={config.initialCapital} onChange={(e) => updateConfig('initialCapital', parseFloat(e.target.value))}
              className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
              step={1000} min={1000}
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] font-mono text-terminal-subtle uppercase block mb-0.5">Commission</label>
              <input type="number" value={config.commission}
                onChange={(e) => updateConfig('commission', parseFloat(e.target.value))}
                className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                step={0.0001} min={0} max={0.01}
              />
            </div>
            <div>
              <label className="text-[10px] font-mono text-terminal-subtle uppercase block mb-0.5">Slippage</label>
              <input type="number" value={config.slippage}
                onChange={(e) => updateConfig('slippage', parseFloat(e.target.value))}
                className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1.5 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                step={0.0001} min={0} max={0.01}
              />
            </div>
          </div>
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={config.allowShort ?? false}
              onChange={(e) => updateConfig('allowShort', e.target.checked)}
              className="accent-terminal-accent"
            />
            <span className="text-[11px] font-mono text-terminal-subtle">Allow Short Selling</span>
          </label>

          {/* Run button */}
          {isRunning ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono">
                <span className="text-terminal-subtle">Running backtest...</span>
                <span className="text-terminal-text">{progress.toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-terminal-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-terminal-accent rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <button onClick={cancelBacktest}
                className="w-full py-1.5 text-xs font-mono text-terminal-bear border border-terminal-bear/30 rounded hover:bg-terminal-bear/20 transition-colors"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button onClick={handleRun}
              className="w-full py-2.5 rounded bg-terminal-accent text-white font-mono text-sm font-bold hover:bg-terminal-accent-dim transition-colors"
            >
              Run Backtest
            </button>
          )}

          {error && (
            <div className="text-terminal-bear text-[10px] font-mono bg-terminal-bear/10 rounded p-2 border border-terminal-bear/30">
              {error}
            </div>
          )}
        </div>
      )}

      {tab === 'results' && (
        <div className="flex-1 overflow-y-auto">
          {/* Result selector */}
          {results.length > 1 && (
            <div className="px-3 py-2 border-b border-terminal-border flex gap-2 overflow-x-auto flex-shrink-0">
              {results.map((r, i) => (
                <button key={r.id} onClick={() => selectResult(r.id)}
                  className={`text-[10px] font-mono px-2 py-1 rounded whitespace-nowrap flex-shrink-0 transition-colors ${r.id === activeResult?.id ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text border border-terminal-border'}`}
                >
                  Run {results.length - i}: {r.config.symbol} {r.config.startDate.slice(0, 4)}
                </button>
              ))}
            </div>
          )}

          {activeResult ? (
            <div className="p-3 space-y-4">
              {/* Equity curve */}
              <div>
                <div className="text-[10px] font-mono text-terminal-subtle uppercase mb-1">Equity Curve</div>
                <div className="h-36">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 4, right: 4, left: 0, bottom: 4 }}>
                      <CartesianGrid stroke="#1f2937" strokeDasharray="2 4" />
                      <XAxis dataKey="time" tickFormatter={(v) => format(new Date(v), 'MM/yy')} tick={{ fill: '#9ca3af', fontSize: 9 }} stroke="#1f2937" minTickGap={50} />
                      <YAxis tickFormatter={(v) => `$${(v/1000).toFixed(0)}K`} tick={{ fill: '#9ca3af', fontSize: 9 }} stroke="#1f2937" width={42} />
                      <YAxis yAxisId="dd" orientation="right" tickFormatter={(v) => `${(v*100).toFixed(0)}%`} tick={{ fill: '#9ca3af', fontSize: 9 }} stroke="#1f2937" width={32} />
                      <Tooltip
                        formatter={(v: number, name: string) => [
                          name === 'drawdownPct' ? `${(v * 100).toFixed(2)}%` : `$${v.toFixed(0)}`,
                          name === 'drawdownPct' ? 'Drawdown' : 'Equity',
                        ]}
                        labelFormatter={(l) => format(new Date(l as number), 'yyyy-MM-dd')}
                        contentStyle={{ backgroundColor: '#111827', border: '1px solid #1f2937', fontSize: 10 }}
                      />
                      <ReferenceLine y={activeResult.config.initialCapital} stroke="#4b5563" strokeDasharray="4 4" />
                      <Area type="monotone" dataKey="equity" stroke="#3b82f6" fill="rgba(59,130,246,0.1)" strokeWidth={1.5} dot={false} />
                      <Area yAxisId="dd" type="monotone" dataKey="drawdownPct" stroke="#ef4444" fill="rgba(239,68,68,0.1)" strokeWidth={1} dot={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Metrics */}
              <div>
                <div className="text-[10px] font-mono text-terminal-subtle uppercase mb-1">Metrics</div>
                <MetricsGrid metrics={activeResult.metrics} />
              </div>

              {/* Actions */}
              <div className="flex gap-2 pt-2 border-t border-terminal-border">
                <button onClick={() => exportResult(activeResult)}
                  className="flex-1 text-[10px] font-mono py-1.5 rounded border border-terminal-border text-terminal-subtle hover:text-terminal-text transition-colors"
                >
                  Export JSON
                </button>
                <button onClick={() => exportTradesToCsv(activeResult)}
                  className="flex-1 text-[10px] font-mono py-1.5 rounded border border-terminal-border text-terminal-subtle hover:text-terminal-text transition-colors"
                >
                  Export Trades CSV
                </button>
                <button onClick={() => deleteResult(activeResult.id)}
                  className="text-[10px] font-mono py-1.5 px-3 rounded border border-terminal-bear/30 text-terminal-bear hover:bg-terminal-bear/20 transition-colors"
                >
                  Delete
                </button>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-terminal-subtle text-sm">
              No results yet — run a backtest
            </div>
          )}
        </div>
      )}

      {tab === 'trades' && (
        <div className="flex-1 overflow-auto">
          {activeResult ? (
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-terminal-surface border-b border-terminal-border z-10">
                <tr>
                  {['Entry', 'Exit', 'Side', 'Entry $', 'Exit $', 'P&L'].map((h) => (
                    <th key={h} className="px-2 py-1.5 text-left font-mono text-[10px] text-terminal-subtle uppercase">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {activeResult.trades.slice(0, 200).map((t, i) => (
                  <tr key={i} className={`border-b border-terminal-border/20 ${t.pnl >= 0 ? 'bg-terminal-bull/5' : 'bg-terminal-bear/5'}`}>
                    <td className="px-2 py-1 font-mono text-[10px] text-terminal-subtle">{format(new Date(t.entryTime * 1000), 'MM/dd')}</td>
                    <td className="px-2 py-1 font-mono text-[10px] text-terminal-subtle">{format(new Date(t.exitTime * 1000), 'MM/dd')}</td>
                    <td className="px-2 py-1"><span className={`text-[10px] font-mono ${t.side === 'long' ? 'text-terminal-bull' : 'text-terminal-bear'}`}>{t.side.toUpperCase()}</span></td>
                    <td className="px-2 py-1 font-mono text-[10px] text-terminal-subtle">{t.entryPrice.toFixed(2)}</td>
                    <td className="px-2 py-1 font-mono text-[10px] text-terminal-subtle">{t.exitPrice.toFixed(2)}</td>
                    <td className={`px-2 py-1 font-mono text-[10px] font-medium ${t.pnl >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
                      {t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="flex items-center justify-center h-full text-terminal-subtle text-sm">
              No backtest results
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default BacktestPanel
