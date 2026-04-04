// ============================================================
// MarketHeatmap.tsx — Crypto market heatmap page
// ============================================================
import React, { useState, useMemo } from 'react'
import { clsx } from 'clsx'
import { Card, Badge, Select } from '@/components/ui'
import type { InstrumentSector } from '@/types'

interface CoinTile {
  symbol: string
  name: string
  sector: InstrumentSector
  price: number
  change24hPct: number
  volume24hUsd: number
  marketCap: number
  allocation: number
  dominance: number
}

const COIN_DATA: CoinTile[] = [
  { symbol: 'BTC',  name: 'Bitcoin',       sector: 'L1',       price: 63450,  change24hPct: 0.021,  volume24hUsd: 28_500_000_000, marketCap: 1_248_000_000_000, allocation: 0.21, dominance: 54.2 },
  { symbol: 'ETH',  name: 'Ethereum',      sector: 'L1',       price: 3218,   change24hPct: 0.018,  volume24hUsd: 14_200_000_000, marketCap: 386_000_000_000,  allocation: 0.18, dominance: 16.7 },
  { symbol: 'BNB',  name: 'BNB',           sector: 'Exchange', price: 582,    change24hPct: 0.031,  volume24hUsd: 1_860_000_000,  marketCap: 84_800_000_000,   allocation: 0.09, dominance: 3.67 },
  { symbol: 'SOL',  name: 'Solana',        sector: 'L1',       price: 156.5,  change24hPct: -0.008, volume24hUsd: 2_920_000_000,  marketCap: 71_600_000_000,   allocation: 0.13, dominance: 3.10 },
  { symbol: 'DOGE', name: 'Dogecoin',      sector: 'Meme',     price: 0.131,  change24hPct: 0.055,  volume24hUsd: 1_240_000_000,  marketCap: 19_200_000_000,   allocation: 0.22, dominance: 0.83 },
  { symbol: 'AVAX', name: 'Avalanche',     sector: 'L1',       price: 34.2,   change24hPct: -0.042, volume24hUsd: 680_000_000,    marketCap: 14_100_000_000,   allocation: 0.02, dominance: 0.61 },
  { symbol: 'LINK', name: 'Chainlink',     sector: 'DeFi',     price: 14.8,   change24hPct: 0.015,  volume24hUsd: 520_000_000,    marketCap: 8_640_000_000,    allocation: 0.04, dominance: 0.37 },
  { symbol: 'UNI',  name: 'Uniswap',       sector: 'DeFi',     price: 8.7,    change24hPct: 0.003,  volume24hUsd: 142_000_000,    marketCap: 5_220_000_000,    allocation: 0.05, dominance: 0.23 },
  { symbol: 'AAVE', name: 'Aave',          sector: 'DeFi',     price: 94.5,   change24hPct: -0.028, volume24hUsd: 198_000_000,    marketCap: 1_418_000_000,    allocation: 0.03, dominance: 0.06 },
  { symbol: 'ARB',  name: 'Arbitrum',      sector: 'L1',       price: 1.12,   change24hPct: 0.038,  volume24hUsd: 312_000_000,    marketCap: 2_880_000_000,    allocation: 0.01, dominance: 0.12 },
  { symbol: 'OP',   name: 'Optimism',      sector: 'L1',       price: 2.34,   change24hPct: -0.017, volume24hUsd: 224_000_000,    marketCap: 3_120_000_000,    allocation: 0.00, dominance: 0.14 },
  { symbol: 'MATIC',name: 'Polygon',       sector: 'L1',       price: 0.78,   change24hPct: 0.011,  volume24hUsd: 286_000_000,    marketCap: 7_600_000_000,    allocation: 0.05, dominance: 0.33 },
  { symbol: 'SHIB', name: 'Shiba Inu',     sector: 'Meme',     price: 0.0000182, change24hPct: 0.024, volume24hUsd: 420_000_000, marketCap: 10_800_000_000, allocation: 0.00, dominance: 0.47 },
  { symbol: 'DOT',  name: 'Polkadot',      sector: 'L1',       price: 7.2,    change24hPct: -0.009, volume24hUsd: 186_000_000,    marketCap: 10_240_000_000,   allocation: 0.00, dominance: 0.44 },
  { symbol: 'MKR',  name: 'Maker',         sector: 'DeFi',     price: 2840,   change24hPct: 0.032,  volume24hUsd: 89_000_000,     marketCap: 2_640_000_000,    allocation: 0.00, dominance: 0.11 },
  { symbol: 'CRV',  name: 'Curve',         sector: 'DeFi',     price: 0.48,   change24hPct: -0.041, volume24hUsd: 64_000_000,     marketCap: 428_000_000,      allocation: 0.00, dominance: 0.02 },
  { symbol: 'OKB',  name: 'OKB',           sector: 'Exchange', price: 54.8,   change24hPct: 0.018,  volume24hUsd: 48_000_000,     marketCap: 3_280_000_000,    allocation: 0.00, dominance: 0.14 },
  { symbol: 'FTT',  name: 'FTT',           sector: 'Exchange', price: 1.84,   change24hPct: 0.008,  volume24hUsd: 32_000_000,     marketCap: 580_000_000,      allocation: 0.00, dominance: 0.03 },
]

const SECTORS: (InstrumentSector | 'All')[] = ['All', 'L1', 'DeFi', 'Exchange', 'Meme']

// ---- Color utilities ----

function changeColor(pct: number): string {
  const abs = Math.abs(pct)
  const intensity = Math.min(abs / 0.08, 1)
  if (pct > 0) {
    const g = Math.round(50 + intensity * 205)
    return `rgb(0,${g},60)`
  } else {
    const r = Math.round(80 + intensity * 175)
    return `rgb(${r},0,0)`
  }
}

function changeTextColor(pct: number): string {
  if (pct > 0.02) return '#4ade80'
  if (pct > 0) return '#86efac'
  if (pct > -0.02) return '#fca5a5'
  return '#f87171'
}

// ---- Coin tile ----

const CoinTileComponent: React.FC<{
  coin: CoinTile
  sizeMode: 'allocation' | 'marketcap' | 'volume'
  selected: boolean
  onClick: () => void
}> = ({ coin, sizeMode, selected, onClick }) => {
  const minPx = 60
  const maxPx = 200

  const sizeValue = {
    allocation: coin.allocation,
    marketcap: coin.marketCap,
    volume: coin.volume24hUsd,
  }[sizeMode]

  const maxValues: Record<string, number> = {
    allocation: 0.25,
    marketcap: 1_250_000_000_000,
    volume: 30_000_000_000,
  }

  const sizePx = Math.max(minPx, Math.round((sizeValue / maxValues[sizeMode]) * maxPx))

  const formatPrice = (p: number) => {
    if (p >= 1000) return `$${p.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
    if (p >= 1) return `$${p.toFixed(2)}`
    if (p >= 0.0001) return `$${p.toFixed(4)}`
    return `$${p.toFixed(8)}`
  }

  return (
    <div
      onClick={onClick}
      title={`${coin.name}\n${formatPrice(coin.price)}\n${(coin.change24hPct * 100).toFixed(2)}%`}
      className={clsx(
        'relative flex flex-col items-center justify-center cursor-pointer rounded-lg transition-all border overflow-hidden',
        selected
          ? 'border-blue-500 ring-1 ring-blue-500/50'
          : 'border-transparent hover:border-slate-600',
      )}
      style={{
        width: sizePx,
        height: sizePx,
        background: changeColor(coin.change24hPct),
      }}
    >
      <span className="text-[11px] font-mono font-bold text-white/90">{coin.symbol}</span>
      <span className="text-[9px] font-mono mt-0.5" style={{ color: changeTextColor(coin.change24hPct) }}>
        {coin.change24hPct >= 0 ? '+' : ''}{(coin.change24hPct * 100).toFixed(2)}%
      </span>
      {sizePx >= 80 && (
        <span className="text-[8px] font-mono text-white/50 mt-0.5">{formatPrice(coin.price)}</span>
      )}
      {coin.allocation > 0 && sizePx >= 70 && (
        <div className="absolute top-1 right-1 w-1 h-1 rounded-full bg-blue-400/70" />
      )}
    </div>
  )
}

// ---- Detail panel ----

const CoinDetail: React.FC<{ coin: CoinTile | null }> = ({ coin }) => {
  if (!coin) return (
    <div className="flex items-center justify-center h-full text-slate-600 text-xs font-mono">
      Click a coin to see details
    </div>
  )

  const formatLarge = (n: number) => {
    if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`
    if (n >= 1e9) return `$${(n / 1e9).toFixed(1)}B`
    if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`
    return `$${n.toLocaleString()}`
  }

  return (
    <div className="flex flex-col gap-3 p-3 text-xs font-mono">
      <div className="flex items-center gap-2">
        <span className="text-lg font-bold text-slate-100">{coin.symbol}</span>
        <span className="text-slate-500">{coin.name}</span>
        <Badge variant={coin.sector === 'L1' ? 'info' : coin.sector === 'DeFi' ? 'bull' : coin.sector === 'Meme' ? 'warning' : 'neutral'}>
          {coin.sector}
        </Badge>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {[
          { label: 'Price', value: coin.price >= 1 ? `$${coin.price.toFixed(2)}` : `$${coin.price.toFixed(6)}` },
          { label: '24h Change', value: `${coin.change24hPct >= 0 ? '+' : ''}${(coin.change24hPct * 100).toFixed(2)}%`, positive: coin.change24hPct >= 0 },
          { label: 'Volume 24h', value: formatLarge(coin.volume24hUsd) },
          { label: 'Market Cap', value: formatLarge(coin.marketCap) },
          { label: 'Dominance', value: `${coin.dominance.toFixed(2)}%` },
          { label: 'In Portfolio', value: coin.allocation > 0 ? `${(coin.allocation * 100).toFixed(1)}%` : '—' },
        ].map((row) => (
          <div key={row.label} className="bg-[#0e1017] rounded p-2 border border-[#1e2130]">
            <div className="text-[9px] text-slate-600 uppercase mb-0.5">{row.label}</div>
            <div className={clsx(
              'text-[11px] font-semibold',
              'positive' in row ? (row.positive ? 'text-emerald-400' : 'text-red-400') : 'text-slate-200',
            )}>
              {row.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ---- Page ----

export const MarketHeatmap: React.FC = () => {
  const [sector, setSector] = useState<InstrumentSector | 'All'>('All')
  const [sizeMode, setSizeMode] = useState<'allocation' | 'marketcap' | 'volume'>('marketcap')
  const [selectedCoin, setSelectedCoin] = useState<CoinTile | null>(null)

  const filtered = useMemo(
    () => COIN_DATA.filter((c) => sector === 'All' || c.sector === sector),
    [sector],
  )

  const sectorGroups = useMemo(() => {
    if (sector !== 'All') return { [sector]: filtered }
    const groups: Partial<Record<InstrumentSector, CoinTile[]>> = {}
    for (const coin of COIN_DATA) {
      if (!groups[coin.sector]) groups[coin.sector] = []
      groups[coin.sector]!.push(coin)
    }
    return groups
  }, [sector, filtered])

  return (
    <div className="flex h-full overflow-hidden">
      {/* Main heatmap */}
      <div className="flex-1 flex flex-col overflow-y-auto thin-scrollbar p-4 gap-4">
        {/* Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-1">
            {SECTORS.map((s) => (
              <button
                key={s}
                onClick={() => setSector(s)}
                className={clsx(
                  'px-2.5 py-1 rounded text-[10px] font-mono border transition-colors',
                  sector === s
                    ? 'border-blue-500/50 text-blue-400 bg-blue-950/30'
                    : 'border-[#1e2130] text-slate-500 hover:text-slate-300',
                )}
              >
                {s}
              </button>
            ))}
          </div>
          <Select
            value={sizeMode}
            onChange={(v) => setSizeMode(v as typeof sizeMode)}
            options={[
              { value: 'marketcap', label: 'Size: Market Cap' },
              { value: 'allocation', label: 'Size: Portfolio Alloc' },
              { value: 'volume', label: 'Size: Volume' },
            ]}
          />
        </div>

        {/* Heatmap */}
        {Object.entries(sectorGroups).map(([sectorName, coins]) => (
          <Card key={sectorName} title={sectorName} padding="sm">
            <div className="flex flex-wrap gap-2 items-end">
              {coins!.map((coin) => (
                <CoinTileComponent
                  key={coin.symbol}
                  coin={coin}
                  sizeMode={sizeMode}
                  selected={selectedCoin?.symbol === coin.symbol}
                  onClick={() => setSelectedCoin((prev) => prev?.symbol === coin.symbol ? null : coin)}
                />
              ))}
            </div>
          </Card>
        ))}

        {/* Legend */}
        <div className="flex items-center gap-3">
          <span className="text-[9px] font-mono text-slate-600">Return:</span>
          <div className="flex items-center gap-1">
            <div className="w-6 h-3 rounded" style={{ background: 'rgb(255,0,0)' }} />
            <span className="text-[9px] font-mono text-slate-600">-8%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-6 h-3 rounded" style={{ background: 'rgb(80,0,0)' }} />
            <span className="text-[9px] font-mono text-slate-600">-2%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-6 h-3 rounded" style={{ background: 'rgb(0,50,30)' }} />
            <span className="text-[9px] font-mono text-slate-600">+2%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-6 h-3 rounded" style={{ background: 'rgb(0,255,60)' }} />
            <span className="text-[9px] font-mono text-slate-600">+8%</span>
          </div>
          <div className="ml-2 w-2 h-2 rounded-full bg-blue-400/70" />
          <span className="text-[9px] font-mono text-slate-600">In portfolio</span>
        </div>
      </div>

      {/* Side panel */}
      <div className="w-64 border-l border-[#1e2130] bg-[#0e1017] overflow-y-auto thin-scrollbar">
        <CoinDetail coin={selectedCoin} />
      </div>
    </div>
  )
}
