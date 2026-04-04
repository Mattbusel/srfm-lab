// ============================================================
// Terminal — main trading terminal layout
// ============================================================
import React, { useState, useCallback, useEffect } from 'react'
import { useMarketStore } from '@/store/marketStore'
import { usePortfolioStore } from '@/store/portfolioStore'
import { useBHStore } from '@/store/bhStore'
import { useSettingsStore } from '@/store/settingsStore'
import { useMarketData } from '@/hooks/useMarketData'
import { useLiveTrader } from '@/hooks/useLiveTrader'
import { usePortfolio } from '@/hooks/usePortfolio'
import { useAlerts } from '@/hooks/useAlerts'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { CandlestickChart } from '@/components/charts/CandlestickChart'
import { OrderBookChart } from '@/components/charts/OrderBookChart'
import { Watchlist } from '@/components/market/Watchlist'
import { QuoteBar } from '@/components/market/QuoteBar'
import { PositionTable } from '@/components/portfolio/PositionTable'
import { OrderEntry } from '@/components/portfolio/OrderEntry'
import { AccountSummary } from '@/components/portfolio/AccountSummary'
import { BHDashboard } from '@/components/bh/BHDashboard'
import type { Interval } from '@/types'

type PaneLayout = 'standard' | 'chart-full' | 'book-focus' | 'bh-focus'

const LAYOUT_LABELS: Record<PaneLayout, string> = {
  standard: 'Standard',
  'chart-full': 'Chart Full',
  'book-focus': 'Book Focus',
  'bh-focus': 'BH Focus',
}

export const Terminal: React.FC = () => {
  const selectedSymbol = useMarketStore((s) => s.selectedSymbol)
  const setSelectedSymbol = useMarketStore((s) => s.setSelectedSymbol)
  const selectedInterval = useMarketStore((s) => s.selectedInterval)
  const setSelectedInterval = useMarketStore((s) => s.setSelectedInterval)
  const isConnected = useMarketStore((s) => s.isConnected)
  const account = usePortfolioStore((s) => s.account)
  const bhFormations = useBHStore((s) => s.formationEvents.filter((e) => !e.acknowledged).length)
  const alerts = useSettingsStore((s) => s.alerts.filter((a) => !a.acknowledged).length)

  const [layout, setLayout] = useState<PaneLayout>('standard')
  const [showOrderEntry, setShowOrderEntry] = useState(false)
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy')
  const [showBHPanel, setShowBHPanel] = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())

  // Initialize data feeds
  useMarketData()
  useLiveTrader()
  usePortfolio()
  useAlerts()

  // Update clock
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  // Keyboard shortcuts
  useKeyboardShortcuts([
    {
      key: 'b',
      description: 'Buy selected symbol',
      action: () => { setOrderSide('buy'); setShowOrderEntry(true) },
    },
    {
      key: 's',
      description: 'Sell selected symbol',
      action: () => { setOrderSide('sell'); setShowOrderEntry(true) },
    },
    {
      key: 'Escape',
      description: 'Close modals',
      action: () => { setShowOrderEntry(false); setShowBHPanel(false) },
    },
    { key: '1', description: '1m interval', action: () => setSelectedInterval('1m' as Interval) },
    { key: '2', description: '5m interval', action: () => setSelectedInterval('5m' as Interval) },
    { key: '3', description: '15m interval', action: () => setSelectedInterval('15m' as Interval) },
    { key: '4', description: '1h interval', action: () => setSelectedInterval('1h' as Interval) },
    { key: '5', description: '4h interval', action: () => setSelectedInterval('4h' as Interval) },
    { key: '6', description: '1d interval', action: () => setSelectedInterval('1d' as Interval) },
  ])

  const dayPnl = account?.dayPnl ?? 0
  const dayPnlPct = account?.dayPnlPct ?? 0
  const equity = account?.equity ?? 0

  return (
    <div className="flex flex-col h-full bg-terminal-bg">
      {/* Top bar */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-terminal-surface border-b border-terminal-border flex-shrink-0">
        {/* Left: brand + connection */}
        <div className="flex items-center gap-3">
          <span className="text-terminal-text font-bold text-sm font-mono tracking-wider">SRFM</span>
          <span className="text-terminal-subtle text-xs font-mono">TERMINAL</span>
          <div className="flex items-center gap-1.5">
            <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? 'bg-terminal-bull animate-pulse' : 'bg-terminal-bear'}`} />
            <span className="text-[10px] font-mono text-terminal-subtle">{isConnected ? 'LIVE' : 'OFFLINE'}</span>
          </div>
        </div>

        {/* Center: account summary */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="text-terminal-subtle">EQ:</span>
            <span className="text-terminal-text font-medium">${equity >= 1000 ? (equity / 1000).toFixed(1) + 'K' : equity.toFixed(0)}</span>
          </div>
          <div className="flex items-center gap-1 text-xs font-mono">
            <span className="text-terminal-subtle">Day:</span>
            <span className={dayPnl >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}>
              {dayPnl >= 0 ? '+' : ''}${Math.abs(dayPnl).toFixed(2)} ({dayPnlPct >= 0 ? '+' : ''}{(dayPnlPct * 100).toFixed(2)}%)
            </span>
          </div>
          {bhFormations > 0 && (
            <button
              onClick={() => setShowBHPanel(!showBHPanel)}
              className="flex items-center gap-1 bg-terminal-warning/20 text-terminal-warning border border-terminal-warning/30 rounded px-2 py-0.5 text-[10px] font-mono animate-pulse hover:animate-none hover:bg-terminal-warning/30 transition-all"
            >
              ⚛ {bhFormations} formation{bhFormations > 1 ? 's' : ''}
            </button>
          )}
          {alerts > 0 && (
            <div className="bg-terminal-bear/20 text-terminal-bear border border-terminal-bear/30 rounded px-2 py-0.5 text-[10px] font-mono">
              {alerts} alert{alerts > 1 ? 's' : ''}
            </div>
          )}
        </div>

        {/* Right: layout + time */}
        <div className="flex items-center gap-3">
          {/* Layout selector */}
          <div className="flex gap-1">
            {(Object.keys(LAYOUT_LABELS) as PaneLayout[]).map((l) => (
              <button
                key={l}
                onClick={() => setLayout(l)}
                className={`text-[10px] font-mono px-1.5 py-0.5 rounded transition-colors ${
                  layout === l ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text'
                }`}
              >
                {LAYOUT_LABELS[l]}
              </button>
            ))}
          </div>

          <div className="font-mono text-xs text-terminal-subtle">
            {currentTime.toLocaleTimeString('en-US', { hour12: false })}
            {' '}
            <span className="text-[10px] text-terminal-muted">ET</span>
          </div>

          {/* Quick order buttons */}
          <div className="flex gap-1">
            <button
              onClick={() => { setOrderSide('buy'); setShowOrderEntry(true) }}
              className="bg-terminal-bull/20 text-terminal-bull border border-terminal-bull/30 rounded px-2 py-0.5 text-[10px] font-mono font-bold hover:bg-terminal-bull/30 transition-colors"
            >
              B
            </button>
            <button
              onClick={() => { setOrderSide('sell'); setShowOrderEntry(true) }}
              className="bg-terminal-bear/20 text-terminal-bear border border-terminal-bear/30 rounded px-2 py-0.5 text-[10px] font-mono font-bold hover:bg-terminal-bear/30 transition-colors"
            >
              S
            </button>
          </div>
        </div>
      </div>

      {/* Quote bar */}
      <QuoteBar symbol={selectedSymbol} />

      {/* Main content */}
      <div className="flex-1 flex min-h-0">
        {/* Watchlist */}
        <div className="w-52 flex-shrink-0 border-r border-terminal-border">
          <Watchlist onSymbolSelect={setSelectedSymbol} />
        </div>

        {/* Center area */}
        <div className="flex-1 flex flex-col min-w-0">
          {layout === 'standard' && (
            <>
              {/* Chart */}
              <div className="flex-1 min-h-0 border-b border-terminal-border">
                <div className="flex h-full">
                  <div className="flex-1 min-w-0">
                    <CandlestickChart
                      symbol={selectedSymbol}
                      interval={selectedInterval}
                      height={400}
                      onIntervalChange={setSelectedInterval}
                    />
                  </div>
                  {/* Order book right of chart */}
                  <div className="w-64 border-l border-terminal-border flex-shrink-0">
                    <OrderBookChart
                      symbol={selectedSymbol}
                      levels={12}
                      onPriceClick={(price) => {}}
                      showTape={false}
                    />
                  </div>
                </div>
              </div>
              {/* Positions table */}
              <div className="h-44 flex-shrink-0">
                <PositionTable />
              </div>
            </>
          )}

          {layout === 'chart-full' && (
            <div className="flex-1 min-h-0">
              <CandlestickChart
                symbol={selectedSymbol}
                interval={selectedInterval}
                height={600}
                onIntervalChange={setSelectedInterval}
              />
            </div>
          )}

          {layout === 'book-focus' && (
            <div className="flex h-full">
              <div className="flex-1">
                <CandlestickChart symbol={selectedSymbol} interval={selectedInterval} height={400} />
              </div>
              <div className="w-80 border-l border-terminal-border">
                <OrderBookChart symbol={selectedSymbol} levels={20} showTape />
              </div>
            </div>
          )}

          {layout === 'bh-focus' && (
            <div className="flex h-full">
              <div className="flex-1">
                <CandlestickChart symbol={selectedSymbol} interval={selectedInterval} height={350} />
              </div>
              <div className="w-96 border-l border-terminal-border">
                <BHDashboard />
              </div>
            </div>
          )}
        </div>

        {/* Right panel: Order entry + account */}
        <div className="w-64 flex-shrink-0 border-l border-terminal-border flex flex-col">
          <div className="flex-1 min-h-0 border-b border-terminal-border">
            <OrderEntry defaultSide={orderSide} />
          </div>
          <div className="h-56 flex-shrink-0">
            <AccountSummary />
          </div>
        </div>
      </div>

      {/* BH Panel overlay */}
      {showBHPanel && (
        <div className="absolute inset-0 z-50 flex">
          <div className="flex-1" onClick={() => setShowBHPanel(false)} />
          <div className="w-96 h-full border-l border-terminal-border bg-terminal-bg shadow-xl">
            <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border">
              <span className="font-mono text-sm text-terminal-text">BH Physics Dashboard</span>
              <button onClick={() => setShowBHPanel(false)} className="text-terminal-subtle hover:text-terminal-text">✕</button>
            </div>
            <BHDashboard />
          </div>
        </div>
      )}

      {/* Floating order entry modal */}
      {showOrderEntry && (
        <div className="absolute inset-0 z-50 bg-black/60 flex items-center justify-center">
          <div className="w-80 h-auto max-h-[80vh] bg-terminal-bg border border-terminal-border rounded-lg shadow-xl overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border">
              <span className="font-mono text-sm text-terminal-text font-semibold">
                {orderSide === 'buy' ? '▲ Buy' : '▼ Sell'} {selectedSymbol}
              </span>
              <button onClick={() => setShowOrderEntry(false)} className="text-terminal-subtle hover:text-terminal-text">✕</button>
            </div>
            <div className="h-[600px]">
              <OrderEntry
                defaultSide={orderSide}
                defaultSymbol={selectedSymbol}
                onOrderSubmitted={() => setShowOrderEntry(false)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Terminal
