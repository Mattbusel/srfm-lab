// ============================================================
// MarketSummary — market overview cards with sparklines
// ============================================================
import React, { useMemo } from 'react'
import { LineChart, Line, ResponsiveContainer, Tooltip } from 'recharts'
import { useMarketStore } from '@/store/marketStore'
import type { MarketStat } from '@/types'

const MARKET_SYMBOLS: MarketStat[] = [
  { symbol: 'SPY', name: 'S&P 500', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'index' },
  { symbol: 'QQQ', name: 'NASDAQ', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'index' },
  { symbol: 'IWM', name: 'Russell 2K', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'index' },
  { symbol: 'DIA', name: 'Dow Jones', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'index' },
  { symbol: 'GLD', name: 'Gold', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'commodity' },
  { symbol: 'TLT', name: '20Y Bonds', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'bond' },
  { symbol: 'UUP', name: 'USD Index', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'equity' },
  { symbol: 'BTC/USD', name: 'Bitcoin', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'crypto' },
  { symbol: 'ETH/USD', name: 'Ethereum', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'crypto' },
  { symbol: 'VIX', name: 'VIX', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'index' },
  { symbol: 'USO', name: 'Crude Oil', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'commodity' },
  { symbol: 'XLF', name: 'Financials', price: 0, change: 0, changePct: 0, volume: 0, sparkline: [], category: 'equity' },
]

const CATEGORY_COLORS: Record<string, string> = {
  index: '#3b82f6',
  equity: '#22c55e',
  crypto: '#f59e0b',
  bond: '#06b6d4',
  commodity: '#d97706',
}

interface MarketCardProps {
  stat: MarketStat & { quote?: ReturnType<typeof useMarketStore>['quotes'][string] }
  onClick?: () => void
}

function MarketCard({ stat, onClick }: MarketCardProps) {
  const quote = useMarketStore((s) => s.quotes[stat.symbol])
  const sparkline = useMarketStore((s) => s.sparklines[stat.symbol] ?? [])

  const price = quote?.lastPrice ?? stat.price
  const changePct = quote?.dayChangePct ?? stat.changePct
  const isUp = changePct >= 0
  const categoryColor = CATEGORY_COLORS[stat.category] ?? '#9ca3af'

  return (
    <div
      className={`bg-terminal-surface border border-terminal-border rounded p-2.5 cursor-pointer hover:border-terminal-accent/50 transition-all group ${
        isUp ? 'hover:border-terminal-bull/40' : 'hover:border-terminal-bear/40'
      }`}
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-1">
        <div>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: categoryColor }} />
            <span className="text-[10px] font-mono text-terminal-subtle uppercase">{stat.symbol}</span>
          </div>
          <div className="text-terminal-subtle text-[10px] truncate max-w-[80px]">{stat.name}</div>
        </div>
        <div className="text-right">
          <div className={`font-mono text-sm font-semibold ${isUp ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
            {price > 0 ? (price > 1000 ? `$${(price / 1000).toFixed(1)}K` : `$${price.toFixed(price > 10 ? 2 : 4)}`) : '—'}
          </div>
          <div className={`font-mono text-[10px] ${isUp ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
            {isUp ? '+' : ''}{(changePct * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Sparkline */}
      {sparkline.length > 1 ? (
        <div className="h-8">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sparkline}>
              <Line
                type="monotone"
                dataKey="value"
                stroke={isUp ? '#22c55e' : '#ef4444'}
                strokeWidth={1.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="h-8 flex items-center justify-center">
          <div className="w-full h-px bg-terminal-border" />
        </div>
      )}
    </div>
  )
}

interface MarketSummaryProps {
  onSymbolClick?: (symbol: string) => void
  className?: string
}

type CategoryFilter = 'all' | 'index' | 'crypto' | 'bond' | 'commodity'

export const MarketSummary: React.FC<MarketSummaryProps> = ({
  onSymbolClick,
  className = '',
}) => {
  const [categoryFilter, setCategoryFilter] = React.useState<CategoryFilter>('all')

  const filteredSymbols = useMemo(
    () => MARKET_SYMBOLS.filter((s) => categoryFilter === 'all' || s.category === categoryFilter),
    [categoryFilter]
  )

  const FILTERS: { label: string; value: CategoryFilter }[] = [
    { label: 'All', value: 'all' },
    { label: 'Index', value: 'index' },
    { label: 'Crypto', value: 'crypto' },
    { label: 'Bonds', value: 'bond' },
    { label: 'Commodities', value: 'commodity' },
  ]

  return (
    <div className={`flex flex-col bg-terminal-bg ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border flex-shrink-0">
        <span className="text-terminal-subtle text-xs font-mono uppercase tracking-wider">Markets</span>
        <div className="flex gap-1">
          {FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => setCategoryFilter(f.value)}
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded transition-colors ${
                categoryFilter === f.value
                  ? 'bg-terminal-accent/20 text-terminal-accent'
                  : 'text-terminal-subtle hover:text-terminal-text'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-2">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-2">
          {filteredSymbols.map((stat) => (
            <MarketCard
              key={stat.symbol}
              stat={stat}
              onClick={() => onSymbolClick?.(stat.symbol)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

export default MarketSummary
