// ============================================================
// useBacktest — runs backtests via spacetime API
// ============================================================
import { useCallback } from 'react'
import { useStrategyStore } from '@/store/strategyStore'
import type { BacktestConfig, BacktestResult } from '@/types'

export function useBacktest(graphId: string) {
  const store = useStrategyStore()

  const isRunning = store.isRunningBacktest
  const progress = store.backtestProgress
  const error = store.error

  const results = store.backtestResults[graphId] ?? []
  const activeResultId = store.activeBacktestId[graphId]
  const activeResult = results.find((r) => r.id === activeResultId) ?? results[0] ?? null

  const runBacktest = useCallback(async (config: BacktestConfig) => {
    await store.runBacktest(graphId, config)
  }, [graphId, store])

  const cancelBacktest = useCallback(() => {
    store.cancelBacktest()
  }, [store])

  const selectResult = useCallback((resultId: string) => {
    store.setActiveBacktest(graphId, resultId)
  }, [graphId, store])

  const deleteResult = useCallback((resultId: string) => {
    store.deleteBacktestResult(graphId, resultId)
  }, [graphId, store])

  const exportResult = useCallback((result: BacktestResult) => {
    const json = JSON.stringify(result, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `backtest-${result.id}-${result.config.symbol}-${result.config.startDate}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [])

  const exportTradesToCsv = useCallback((result: BacktestResult) => {
    const headers = ['Entry Time', 'Exit Time', 'Side', 'Entry Price', 'Exit Price', 'Qty', 'P&L', 'P&L %', 'Entry Signal', 'Exit Signal']
    const rows = result.trades.map((t) => [
      new Date(t.entryTime * 1000).toISOString(),
      new Date(t.exitTime * 1000).toISOString(),
      t.side,
      t.entryPrice.toFixed(4),
      t.exitPrice.toFixed(4),
      t.qty.toString(),
      t.pnl.toFixed(2),
      (t.pnlPct * 100).toFixed(2) + '%',
      t.entrySignal,
      t.exitSignal,
    ])

    const csv = [headers, ...rows].map((row) => row.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `trades-${result.id}-${result.config.symbol}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [])

  return {
    isRunning,
    progress,
    error,
    results,
    activeResult,
    activeResultId,
    runBacktest,
    cancelBacktest,
    selectResult,
    deleteResult,
    exportResult,
    exportTradesToCsv,
  }
}
