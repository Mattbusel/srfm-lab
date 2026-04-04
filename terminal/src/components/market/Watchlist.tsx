// ============================================================
// Watchlist — draggable list with BH status and sparklines
// ============================================================
import React, { useState, useCallback, useRef } from 'react'
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors, DragEndEvent } from '@dnd-kit/core'
import { SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy, useSortable } from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { LineChart, Line, ResponsiveContainer } from 'recharts'
import { motion, AnimatePresence } from 'framer-motion'
import { useMarketStore, selectSortedWatchlist } from '@/store/marketStore'
import { useBHStore } from '@/store/bhStore'
import { useSettingsStore } from '@/store/settingsStore'
import type { Quote, SortableField } from '@/types'

interface WatchlistItemRowProps {
  symbol: string
  isSelected: boolean
  onClick: () => void
}

function WatchlistItemRow({ symbol, isSelected, onClick }: WatchlistItemRowProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: symbol })

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  }

  const quote = useMarketStore((s) => s.quotes[symbol])
  const sparkline = useMarketStore((s) => s.sparklines[symbol] ?? [])
  const flash = useMarketStore((s) => s.priceFlashes[symbol])
  const bhInstrument = useBHStore((s) => s.instruments[symbol])
  const showBH = useSettingsStore((s) => s.settings.showBHOverlay)

  const isUp = (quote?.dayChangePct ?? 0) >= 0
  const hasFlash = flash && Date.now() - flash.timestamp < 500

  const bhActive = bhInstrument && (bhInstrument.tf15m.active || bhInstrument.tf1h.active || bhInstrument.tf1d.active)
  const bhRegime = bhInstrument?.tf1h.regime ?? bhInstrument?.tf1d.regime ?? null

  const bhDotColor = bhActive
    ? bhRegime === 'BULL' ? 'bg-terminal-bull'
    : bhRegime === 'BEAR' ? 'bg-terminal-bear'
    : bhRegime === 'HIGH_VOL' ? 'bg-terminal-warning'
    : 'bg-terminal-subtle'
    : 'bg-terminal-muted'

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`group flex items-center gap-1.5 px-2 py-1.5 cursor-pointer transition-all border-b border-terminal-border/30 ${
        isSelected ? 'bg-terminal-accent/15 border-l-2 border-l-terminal-accent' : 'hover:bg-terminal-surface border-l-2 border-l-transparent'
      } ${hasFlash ? (flash?.direction === 'up' ? 'animate-flash-green' : 'animate-flash-red') : ''}`}
      onClick={onClick}
    >
      {/* Drag handle */}
      <div
        {...attributes}
        {...listeners}
        className="opacity-0 group-hover:opacity-40 cursor-grab active:cursor-grabbing text-terminal-subtle"
      >
        ⠿
      </div>

      {/* BH dot */}
      {showBH && (
        <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${bhDotColor} ${bhActive ? 'animate-pulse' : ''}`} />
      )}

      {/* Symbol */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs font-semibold text-terminal-text truncate">{symbol}</span>
          <span className={`font-mono text-xs font-medium ${isUp ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
            {quote ? (quote.dayChangePct >= 0 ? '+' : '') + (quote.dayChangePct * 100).toFixed(2) + '%' : '—'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className={`font-mono text-[10px] ${hasFlash ? (flash?.direction === 'up' ? 'text-terminal-bull' : 'text-terminal-bear') : 'text-terminal-subtle'}`}>
            {quote ? quote.lastPrice.toFixed(quote.lastPrice > 100 ? 2 : 4) : '—'}
          </span>
          {/* Sparkline */}
          {sparkline.length > 1 && (
            <div className="w-14 h-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sparkline}>
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke={isUp ? '#22c55e' : '#ef4444'}
                    strokeWidth={1}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

interface WatchlistProps {
  className?: string
  onSymbolSelect?: (symbol: string) => void
}

export const Watchlist: React.FC<WatchlistProps> = ({ className = '', onSymbolSelect }) => {
  const store = useMarketStore()
  const sortedWatchlist = useMarketStore(selectSortedWatchlist)
  const selectedSymbol = store.selectedSymbol
  const watchlistSort = store.watchlistSort

  const [searchQuery, setSearchQuery] = useState('')
  const [isAdding, setIsAdding] = useState(false)
  const [newSymbol, setNewSymbol] = useState('')
  const searchRef = useRef<HTMLInputElement>(null)

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  )

  const handleDragEnd = useCallback((event: DragEndEvent) => {
    const { active, over } = event
    if (!over || active.id === over.id) return

    const symbols = sortedWatchlist.map((w) => w.symbol)
    const oldIdx = symbols.indexOf(String(active.id))
    const newIdx = symbols.indexOf(String(over.id))

    if (oldIdx !== -1 && newIdx !== -1) {
      const newOrder = [...symbols]
      newOrder.splice(oldIdx, 1)
      newOrder.splice(newIdx, 0, String(active.id))
      store.reorderWatchlist(newOrder)
    }
  }, [sortedWatchlist, store])

  const handleSelect = useCallback((symbol: string) => {
    store.setSelectedSymbol(symbol)
    onSymbolSelect?.(symbol)
  }, [store, onSymbolSelect])

  const handleAddSymbol = useCallback(() => {
    const sym = newSymbol.toUpperCase().trim()
    if (sym) {
      store.addToWatchlist(sym)
      setNewSymbol('')
      setIsAdding(false)
    }
  }, [newSymbol, store])

  const SORT_OPTIONS: { label: string; field: SortableField }[] = [
    { label: 'A-Z', field: 'symbol' },
    { label: 'Chg%', field: 'changePct' },
    { label: 'Price', field: 'price' },
    { label: 'Vol', field: 'volume' },
  ]

  const filteredWatchlist = searchQuery
    ? sortedWatchlist.filter((w) => w.symbol.toLowerCase().includes(searchQuery.toLowerCase()))
    : sortedWatchlist

  return (
    <div className={`flex flex-col bg-terminal-bg border-r border-terminal-border h-full ${className}`}>
      {/* Header */}
      <div className="px-2 py-2 border-b border-terminal-border flex-shrink-0">
        <div className="flex items-center justify-between mb-2">
          <span className="text-terminal-subtle text-xs font-mono uppercase tracking-wider">Watchlist</span>
          <button
            onClick={() => { setIsAdding(!isAdding); if (!isAdding) setNewSymbol('') }}
            className="text-terminal-subtle hover:text-terminal-accent text-xs transition-colors"
          >
            {isAdding ? '✕' : '+ Add'}
          </button>
        </div>

        {/* Add symbol input */}
        <AnimatePresence>
          {isAdding && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden mb-2"
            >
              <div className="flex gap-1">
                <input
                  ref={searchRef}
                  type="text"
                  value={newSymbol}
                  onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleAddSymbol() }}
                  placeholder="Symbol..."
                  autoFocus
                  className="flex-1 bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                />
                <button
                  onClick={handleAddSymbol}
                  className="bg-terminal-accent text-white px-2 py-1 rounded text-xs hover:bg-terminal-accent-dim transition-colors"
                >
                  Add
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Search */}
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value.toUpperCase())}
          placeholder="Filter..."
          className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text placeholder-terminal-muted focus:outline-none focus:border-terminal-accent mb-1.5"
        />

        {/* Sort buttons */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] font-mono text-terminal-subtle">Sort:</span>
          {SORT_OPTIONS.map((opt) => (
            <button
              key={opt.field}
              onClick={() => {
                store.setWatchlistSort({
                  field: opt.field,
                  direction: watchlistSort.field === opt.field && watchlistSort.direction === 'asc' ? 'desc' : 'asc',
                })
              }}
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded transition-colors ${
                watchlistSort.field === opt.field
                  ? 'text-terminal-accent bg-terminal-accent/10'
                  : 'text-terminal-subtle hover:text-terminal-text'
              }`}
            >
              {opt.label}
              {watchlistSort.field === opt.field && (watchlistSort.direction === 'asc' ? ' ↑' : ' ↓')}
            </button>
          ))}
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
          <SortableContext
            items={filteredWatchlist.map((w) => w.symbol)}
            strategy={verticalListSortingStrategy}
          >
            <AnimatePresence>
              {filteredWatchlist.map((item) => (
                <WatchlistItemRow
                  key={item.symbol}
                  symbol={item.symbol}
                  isSelected={selectedSymbol === item.symbol}
                  onClick={() => handleSelect(item.symbol)}
                />
              ))}
            </AnimatePresence>
          </SortableContext>
        </DndContext>

        {filteredWatchlist.length === 0 && (
          <div className="flex items-center justify-center py-8 text-terminal-subtle text-xs">
            {searchQuery ? `No symbols matching "${searchQuery}"` : 'Watchlist is empty'}
          </div>
        )}
      </div>

      {/* Footer stats */}
      <div className="border-t border-terminal-border px-2 py-1.5 flex-shrink-0">
        <div className="flex items-center justify-between text-[10px] font-mono text-terminal-subtle">
          <span>{filteredWatchlist.length} symbols</span>
          <button
            onClick={() => {
              if (confirm('Remove selected symbol from watchlist?')) {
                store.removeFromWatchlist(selectedSymbol)
              }
            }}
            className="hover:text-terminal-bear transition-colors"
          >
            Remove selected
          </button>
        </div>
      </div>
    </div>
  )
}

export default Watchlist
