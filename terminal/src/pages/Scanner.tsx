// ============================================================
// Scanner — market scanner with BH state filtering
// ============================================================
import React, { useState, useMemo, useCallback } from 'react'
import { useBHStore, selectFilteredScanResults } from '@/store/bhStore'
import { useMarketStore } from '@/store/marketStore'
import type { BHRegime, BHTimeframe } from '@/types'

type SortKey = 'symbol' | 'mass15m' | 'mass1h' | 'mass1d' | 'price' | 'ctl' | 'regime'

export const Scanner: React.FC = () => {
  const instruments = useBHStore((s) => s.instruments)
  const setSelectedSymbol = useMarketStore((s) => s.setSelectedSymbol)
  const addToWatchlist = useMarketStore((s) => s.addToWatchlist)

  const [sortKey, setSortKey] = useState<SortKey>('mass1h')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [filterActive, setFilterActive] = useState(false)
  const [filterFormation, setFilterFormation] = useState(false)
  const [filterRegime, setFilterRegime] = useState<BHRegime | 'all'>('all')
  const [filterMinMass, setFilterMinMass] = useState(0)
  const [search, setSearch] = useState('')

  const rows = useMemo(() => {
    let data = Object.entries(instruments).map(([symbol, instr]) => ({
      symbol,
      price: instr.price,
      mass15m: instr.tf15m.mass,
      mass1h: instr.tf1h.mass,
      mass1d: instr.tf1d.mass,
      regime: instr.tf1d.regime,
      dir: instr.tf1h.dir,
      active15m: instr.tf15m.active,
      active1h: instr.tf1h.active,
      active1d: instr.tf1d.active,
      ctl: instr.tf1h.ctl,
      formation: instr.tf15m.bh_form > 0 || instr.tf1h.bh_form > 0 || instr.tf1d.bh_form > 0,
      frac: instr.frac,
      lastUpdated: instr.lastUpdated,
    }))

    if (search) data = data.filter((d) => d.symbol.includes(search.toUpperCase()))
    if (filterActive) data = data.filter((d) => d.active15m || d.active1h || d.active1d)
    if (filterFormation) data = data.filter((d) => d.formation)
    if (filterRegime !== 'all') data = data.filter((d) => d.regime === filterRegime)
    if (filterMinMass > 0) data = data.filter((d) => Math.max(d.mass15m, d.mass1h, d.mass1d) >= filterMinMass)

    data.sort((a, b) => {
      const av = a[sortKey as keyof typeof a] as number | string
      const bv = b[sortKey as keyof typeof b] as number | string
      let cmp = 0
      if (typeof av === 'number' && typeof bv === 'number') cmp = av - bv
      else if (typeof av === 'string' && typeof bv === 'string') cmp = av.localeCompare(bv)
      return sortDir === 'asc' ? cmp : -cmp
    })

    return data
  }, [instruments, sortKey, sortDir, filterActive, filterFormation, filterRegime, filterMinMass, search])

  const handleSort = useCallback((key: SortKey) => {
    setSortKey((prev) => {
      setSortDir((prevDir) => prev === key ? (prevDir === 'asc' ? 'desc' : 'asc') : 'desc')
      return key
    })
  }, [])

  const REGIME_STYLES: Record<BHRegime, string> = {
    BULL: 'bg-terminal-bull/20 text-terminal-bull',
    BEAR: 'bg-terminal-bear/20 text-terminal-bear',
    SIDEWAYS: 'bg-terminal-muted text-terminal-subtle',
    HIGH_VOL: 'bg-terminal-warning/20 text-terminal-warning',
  }

  const MassCell = ({ mass, active }: { mass: number; active: boolean }) => (
    <span className={`font-mono text-xs ${active ? 'text-terminal-warning font-bold' : mass > 0.8 ? 'text-terminal-text' : 'text-terminal-subtle'}`}>
      {mass.toFixed(3)}
      {active && ' ●'}
    </span>
  )

  return (
    <div className="flex flex-col h-full bg-terminal-bg">
      {/* Header + filters */}
      <div className="px-4 py-3 border-b border-terminal-border flex-shrink-0 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-terminal-text font-mono text-sm font-semibold">Market Scanner</span>
          <span className="text-terminal-subtle text-[10px] font-mono">{rows.length} instruments</span>
        </div>

        <div className="flex flex-wrap gap-3 items-center">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value.toUpperCase())}
            placeholder="Search..."
            className="w-24 bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
          />

          <label className="flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" checked={filterActive} onChange={(e) => setFilterActive(e.target.checked)} className="accent-terminal-accent w-3 h-3" />
            <span className="text-[11px] font-mono text-terminal-subtle">Active BH</span>
          </label>

          <label className="flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" checked={filterFormation} onChange={(e) => setFilterFormation(e.target.checked)} className="accent-terminal-warning w-3 h-3" />
            <span className="text-[11px] font-mono text-terminal-subtle">Formation</span>
          </label>

          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono text-terminal-subtle">Regime:</span>
            <select
              value={filterRegime}
              onChange={(e) => setFilterRegime(e.target.value as BHRegime | 'all')}
              className="bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-[11px] font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
            >
              <option value="all">All</option>
              <option value="BULL">Bull</option>
              <option value="BEAR">Bear</option>
              <option value="SIDEWAYS">Sideways</option>
              <option value="HIGH_VOL">High Vol</option>
            </select>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-mono text-terminal-subtle">Min Mass:</span>
            <input
              type="number"
              value={filterMinMass}
              onChange={(e) => setFilterMinMass(parseFloat(e.target.value) || 0)}
              className="w-16 bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-[11px] font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
              step={0.1} min={0} max={3}
            />
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-terminal-surface border-b border-terminal-border z-10">
            <tr>
              {[
                { key: 'symbol', label: 'Symbol' },
                { key: 'price', label: 'Price' },
                { key: 'regime', label: 'Regime' },
                { key: 'mass15m', label: 'Mass 15m' },
                { key: 'mass1h', label: 'Mass 1h' },
                { key: 'mass1d', label: 'Mass 1d' },
                { key: 'ctl', label: 'CTL' },
              ].map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key as SortKey)}
                  className="px-3 py-2 text-left font-mono text-[10px] text-terminal-subtle uppercase cursor-pointer hover:text-terminal-text transition-colors"
                >
                  {col.label}
                  {sortKey === col.key && <span className="ml-0.5">{sortDir === 'asc' ? '↑' : '↓'}</span>}
                </th>
              ))}
              <th className="px-3 py-2 text-left font-mono text-[10px] text-terminal-subtle uppercase">Formation</th>
              <th className="px-3 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.symbol}
                className={`border-b border-terminal-border/20 hover:bg-terminal-surface transition-colors cursor-pointer ${row.formation ? 'bg-terminal-warning/5' : ''}`}
                onClick={() => setSelectedSymbol(row.symbol)}
              >
                <td className="px-3 py-2">
                  <div className="flex items-center gap-1.5">
                    {row.formation && <div className="w-1.5 h-1.5 rounded-full bg-terminal-warning animate-pulse flex-shrink-0" />}
                    <span className="font-mono text-xs font-semibold text-terminal-text">{row.symbol}</span>
                  </div>
                </td>
                <td className="px-3 py-2 font-mono text-xs text-terminal-text">
                  ${row.price.toFixed(row.price > 100 ? 2 : 4)}
                </td>
                <td className="px-3 py-2">
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${REGIME_STYLES[row.regime]}`}>
                    {row.regime}
                  </span>
                </td>
                <td className="px-3 py-2">
                  <MassCell mass={row.mass15m} active={row.active15m} />
                </td>
                <td className="px-3 py-2">
                  <MassCell mass={row.mass1h} active={row.active1h} />
                </td>
                <td className="px-3 py-2">
                  <MassCell mass={row.mass1d} active={row.active1d} />
                </td>
                <td className="px-3 py-2 font-mono text-xs text-terminal-subtle">
                  {row.ctl > 0 ? <span className="text-terminal-warning">{row.ctl}</span> : '—'}
                </td>
                <td className="px-3 py-2">
                  {row.formation ? (
                    <span className="text-[10px] font-mono text-terminal-warning bg-terminal-warning/10 px-1.5 py-0.5 rounded border border-terminal-warning/30 animate-pulse">
                      ACTIVE
                    </span>
                  ) : '—'}
                </td>
                <td className="px-3 py-2">
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => { e.stopPropagation(); addToWatchlist(row.symbol) }}
                      className="text-[10px] font-mono text-terminal-accent hover:underline"
                    >
                      Watch
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={9} className="text-center py-12 text-terminal-subtle text-sm">
                  No instruments match the current filters
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default Scanner
