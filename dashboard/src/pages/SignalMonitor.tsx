// ============================================================
// SignalMonitor.tsx — Live BH signal monitor page
// ============================================================
import React, { useEffect, useState, useCallback } from 'react'
import { clsx } from 'clsx'
import { format } from 'date-fns'
import { Card, Select, WsStatusDot } from '@/components/ui'
import { BHStateIndicator, BHStateRow, MassGauge } from '@/components/BHStateIndicator'
import { DeltaScoreCard } from '@/components/DeltaScoreCard'
import { RegimeBadge } from '@/components/RegimeBadge'
import { useSignalsStore } from '@/store/signalsStore'
import { useWebSocket } from '@/hooks/useWebSocket'
import type { WsMessage, SignalCard, BHFormation } from '@/types'

// ---- Signal card expanded ----

const SignalCardExpanded: React.FC<{
  card: SignalCard
  onClick?: () => void
  selected?: boolean
}> = ({ card, onClick, selected }) => {
  const formatPrice = (p: number) => {
    if (p >= 1000) return `$${p.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
    if (p >= 1) return `$${p.toFixed(2)}`
    return `$${p.toFixed(5)}`
  }

  return (
    <div
      onClick={onClick}
      className={clsx(
        'bg-[#111318] border rounded-lg p-3 cursor-pointer transition-all',
        selected
          ? 'border-blue-500/60 bg-[#13161e]'
          : 'border-[#1e2130] hover:border-slate-600',
      )}
    >
      {/* Symbol + regime */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono font-bold text-slate-100">
            {card.symbol.replace('USDT', '')}
          </span>
          <RegimeBadge regime={card.trend} size="xs" />
        </div>
        <div className="flex items-center gap-2">
          <span className={clsx(
            'text-[10px] font-mono',
            card.change24hPct >= 0 ? 'text-emerald-400' : 'text-red-400',
          )}>
            {card.change24hPct >= 0 ? '+' : ''}{(card.change24hPct * 100).toFixed(2)}%
          </span>
          <span className="text-[10px] font-mono text-slate-500">{formatPrice(card.price)}</span>
        </div>
      </div>

      {/* BH state row */}
      <div className="mb-2">
        <BHStateRow daily={card.daily} hourly={card.hourly} m15={card.m15} />
      </div>

      {/* Mass + delta score */}
      <div className="flex items-center gap-3">
        <div className="flex flex-col items-center">
          <MassGauge mass={card.mass} size={44} />
          <span className="text-[9px] font-mono text-slate-600 -mt-1">MASS</span>
        </div>
        <div className="flex-1">
          <div className="text-[9px] font-mono text-slate-600 mb-1">DELTA SCORE</div>
          <div className="relative h-2 bg-[#1e2130] rounded-full overflow-hidden">
            <div className="absolute inset-y-0 left-1/2 w-px bg-[#2e3550]" />
            <div
              className={clsx(
                'absolute inset-y-0 rounded-full',
                card.deltaScore >= 0 ? 'bg-emerald-500' : 'bg-red-500',
              )}
              style={{
                left: card.deltaScore >= 0 ? '50%' : `${50 + card.deltaScore * 50}%`,
                width: `${Math.abs(card.deltaScore) * 50}%`,
              }}
            />
          </div>
          <div className={clsx(
            'text-[10px] font-mono font-semibold mt-0.5',
            card.deltaScore >= 0 ? 'text-emerald-400' : 'text-red-400',
          )}>
            {card.deltaScore >= 0 ? '+' : ''}{card.deltaScore.toFixed(2)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[9px] font-mono text-slate-600 mb-0.5">FORMATIONS</div>
          <div className="text-base font-mono font-bold text-blue-400">{card.activeFormations}</div>
        </div>
      </div>

      {/* Last update */}
      <div className="mt-2 pt-2 border-t border-[#1e2130]">
        <span className="text-[9px] font-mono text-slate-700">
          Updated {format(new Date(card.lastUpdate), 'HH:mm:ss')}
        </span>
      </div>
    </div>
  )
}

// ---- Formation list ----

const FormationList: React.FC<{ formations: BHFormation[] }> = ({ formations }) => (
  <div className="flex flex-col gap-1.5">
    {formations.slice(0, 20).map((f) => (
      <div key={f.id} className="flex items-center gap-2 px-3 py-2 bg-[#0e1017] rounded border border-[#1a1d26] text-[10px] font-mono">
        <BHStateIndicator state={f.state} showLabel={false} size="xs" />
        <span className="text-slate-200 font-semibold w-16 flex-shrink-0">
          {f.symbol.replace('USDT', '')}
        </span>
        <span className="text-slate-600 w-8 flex-shrink-0">{f.timeframe}</span>
        <span className="text-slate-400 flex-1">{f.patternType}</span>
        <span className={clsx(
          'font-semibold',
          f.state === 'bullish' ? 'text-emerald-400' : f.state === 'bearish' ? 'text-red-400' : 'text-slate-500',
        )}>
          M{f.mass.toFixed(1)}
        </span>
        <span className="text-slate-600">{(f.reliability * 100).toFixed(0)}%</span>
      </div>
    ))}
  </div>
)

// ---- Page ----

export const SignalMonitor: React.FC = () => {
  const { cards, formations, initMockData, updateCard } = useSignalsStore()
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [sortBy, setSortBy] = useState<'mass' | 'delta' | 'symbol'>('mass')
  const [filterState, setFilterState] = useState<'all' | 'bullish' | 'bearish' | 'neutral'>('all')

  useEffect(() => {
    initMockData()
  }, [initMockData])

  // Simulate live updates
  useEffect(() => {
    const interval = setInterval(() => {
      const card = cards[Math.floor(Math.random() * cards.length)]
      if (card) {
        updateCard(card.symbol, {
          deltaScore: Math.max(-1, Math.min(1, card.deltaScore + (Math.random() - 0.5) * 0.05)),
          mass: Math.max(0, Math.min(2, card.mass + (Math.random() - 0.5) * 0.05)),
          lastUpdate: new Date().toISOString(),
        })
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [cards, updateCard])

  const handleWsMessage = useCallback(
    (msg: WsMessage) => {
      if (msg.type === 'signal') {
        const payload = msg.payload as Partial<SignalCard> & { symbol: string }
        updateCard(payload.symbol, payload)
      }
    },
    [updateCard],
  )

  const { status: wsStatus } = useWebSocket({ onMessage: handleWsMessage })

  const sorted = [...cards]
    .filter((c) => filterState === 'all' || c.daily === filterState)
    .sort((a, b) => {
      if (sortBy === 'mass') return b.mass - a.mass
      if (sortBy === 'delta') return b.deltaScore - a.deltaScore
      return a.symbol.localeCompare(b.symbol)
    })

  const selectedCard = cards.find((c) => c.symbol === selectedSymbol) ?? null

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left: signal grid */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Controls bar */}
        <div className="flex items-center gap-3 px-4 py-2.5 border-b border-[#1e2130] bg-[#0e1017] flex-shrink-0 flex-wrap gap-y-2">
          <WsStatusDot status={wsStatus} />
          <div className="flex items-center gap-1">
            {(['all', 'bullish', 'bearish', 'neutral'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilterState(f)}
                className={clsx(
                  'px-2 py-0.5 rounded text-[9px] font-mono border transition-colors capitalize',
                  filterState === f
                    ? f === 'bullish' ? 'border-emerald-700/60 text-emerald-400 bg-emerald-950/40'
                      : f === 'bearish' ? 'border-red-700/60 text-red-400 bg-red-950/40'
                        : f === 'neutral' ? 'border-slate-700/60 text-slate-400 bg-slate-900/40'
                          : 'border-blue-500/50 text-blue-400 bg-blue-950/30'
                    : 'border-[#1e2130] text-slate-600 hover:text-slate-400',
                )}
              >
                {f}
              </button>
            ))}
          </div>
          <Select
            value={sortBy}
            onChange={(v) => setSortBy(v as typeof sortBy)}
            options={[
              { value: 'mass', label: 'Sort: Mass' },
              { value: 'delta', label: 'Sort: Delta' },
              { value: 'symbol', label: 'Sort: Symbol' },
            ]}
          />
          <span className="text-[10px] font-mono text-slate-600 ml-auto">
            {sorted.length} signals
          </span>
        </div>

        {/* Grid */}
        <div className="flex-1 overflow-y-auto thin-scrollbar p-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
            {sorted.map((card) => (
              <SignalCardExpanded
                key={card.symbol}
                card={card}
                selected={selectedSymbol === card.symbol}
                onClick={() => setSelectedSymbol((p) => p === card.symbol ? null : card.symbol)}
              />
            ))}
          </div>

          {/* Delta score compact grid */}
          <div className="mt-4">
            <h3 className="text-[10px] font-mono text-slate-500 uppercase tracking-wider mb-2">
              Delta Score Overview
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
              {cards.map((card) => (
                <DeltaScoreCard
                  key={card.symbol}
                  card={card}
                  compact
                  onClick={() => setSelectedSymbol((p) => p === card.symbol ? null : card.symbol)}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Right: formations + detail */}
      <div className="w-80 border-l border-[#1e2130] bg-[#0a0b0e] flex flex-col overflow-hidden">
        {selectedCard && (
          <div className="p-3 border-b border-[#1e2130]">
            <div className="text-[10px] font-mono text-slate-500 mb-2 uppercase">Selected Signal</div>
            <div className="flex items-center gap-2 mb-1">
              <span className="font-mono font-bold text-slate-100">{selectedCard.symbol}</span>
              <RegimeBadge regime={selectedCard.trend} size="xs" />
            </div>
            <div className="grid grid-cols-3 gap-1 text-[10px] font-mono">
              <div className="text-center">
                <div className="text-slate-600 mb-0.5">1D</div>
                <BHStateIndicator state={selectedCard.daily} showLabel={false} size="xs" />
              </div>
              <div className="text-center">
                <div className="text-slate-600 mb-0.5">1H</div>
                <BHStateIndicator state={selectedCard.hourly} showLabel={false} size="xs" />
              </div>
              <div className="text-center">
                <div className="text-slate-600 mb-0.5">15M</div>
                <BHStateIndicator state={selectedCard.m15} showLabel={false} size="xs" />
              </div>
            </div>
          </div>
        )}

        <div className="flex-1 overflow-y-auto thin-scrollbar">
          <div className="p-3">
            <div className="text-[10px] font-mono text-slate-500 uppercase tracking-wider mb-2">
              Active Formations ({formations.length})
            </div>
            <FormationList formations={formations} />
          </div>
        </div>
      </div>
    </div>
  )
}
