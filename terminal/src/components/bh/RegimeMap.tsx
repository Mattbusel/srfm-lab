// ============================================================
// RegimeMap — treemap of current regime status
// ============================================================
import React, { useMemo, useState } from 'react'
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts'
import { useBHStore } from '@/store/bhStore'
import { usePortfolioStore, selectPortfolioConcentration } from '@/store/portfolioStore'
import type { BHRegime } from '@/types'

const REGIME_COLORS: Record<BHRegime, { bg: string; text: string; border: string }> = {
  BULL: { bg: 'rgba(34, 197, 94, 0.3)', text: '#22c55e', border: '#16a34a' },
  BEAR: { bg: 'rgba(239, 68, 68, 0.3)', text: '#ef4444', border: '#dc2626' },
  SIDEWAYS: { bg: 'rgba(107, 114, 128, 0.2)', text: '#9ca3af', border: '#6b7280' },
  HIGH_VOL: { bg: 'rgba(245, 158, 11, 0.3)', text: '#f59e0b', border: '#d97706' },
}

interface RegimeMapProps {
  className?: string
  height?: number
}

interface TreeNode {
  name: string
  symbol: string
  value: number
  regime: BHRegime
  mass: number
  price: number
  color: string
}

const CustomContent = ({ x, y, width, height, symbol, regime, mass, value }: {
  x?: number; y?: number; width?: number; height?: number
  symbol?: string; regime?: BHRegime; mass?: number; value?: number
}) => {
  if (!x || !y || !width || !height || width < 20 || height < 20) return null
  const cfg = regime ? REGIME_COLORS[regime] : REGIME_COLORS.SIDEWAYS

  return (
    <g>
      <rect
        x={x + 1} y={y + 1}
        width={width - 2} height={height - 2}
        fill={cfg.bg}
        stroke={cfg.border}
        strokeWidth={1}
        rx={3}
      />
      {width > 50 && height > 30 && (
        <>
          <text x={x + width / 2} y={y + height / 2 - 4} textAnchor="middle" fill={cfg.text} fontSize={Math.min(12, width / 4)} fontFamily="JetBrains Mono, monospace" fontWeight="bold">
            {symbol}
          </text>
          <text x={x + width / 2} y={y + height / 2 + 10} textAnchor="middle" fill={cfg.text} fontSize={Math.min(9, width / 6)} fontFamily="JetBrains Mono, monospace" opacity={0.8}>
            {mass?.toFixed(2)}
          </text>
          {height > 50 && (
            <text x={x + width / 2} y={y + height / 2 + 22} textAnchor="middle" fill={cfg.text} fontSize={Math.min(8, width / 7)} fontFamily="JetBrains Mono, monospace" opacity={0.6}>
              {((value ?? 0) * 100).toFixed(1)}%
            </text>
          )}
        </>
      )}
      {width <= 50 && width > 20 && height > 20 && (
        <text x={x + width / 2} y={y + height / 2 + 4} textAnchor="middle" fill={cfg.text} fontSize={8} fontFamily="JetBrains Mono, monospace" fontWeight="bold">
          {(symbol ?? '').slice(0, 3)}
        </text>
      )}
    </g>
  )
}

const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: TreeNode }> }) => {
  if (!active || !payload?.length) return null
  const node = payload[0].payload
  const cfg = REGIME_COLORS[node.regime]
  return (
    <div className="bg-terminal-surface border border-terminal-border rounded p-2 text-xs font-mono shadow-lg">
      <div className="font-bold text-terminal-text mb-1">{node.symbol}</div>
      <div style={{ color: cfg.text }}>{node.regime}</div>
      <div className="text-terminal-subtle">Mass: {node.mass.toFixed(3)}</div>
      <div className="text-terminal-subtle">Weight: {(node.value * 100).toFixed(1)}%</div>
      <div className="text-terminal-text">${node.price.toFixed(2)}</div>
    </div>
  )
}

export const RegimeMap: React.FC<RegimeMapProps> = ({ className = '', height = 300 }) => {
  const instruments = useBHStore((s) => s.instruments)
  const concentration = usePortfolioStore(selectPortfolioConcentration)
  const [timeframe, setTimeframe] = useState<'15m' | '1h' | '1d'>('1h')

  const concMap = new Map(concentration.map((c) => [c.symbol, c.weight]))

  const treeData = useMemo((): TreeNode[] => {
    return Object.entries(instruments).map(([sym, instr]) => {
      const tfState = timeframe === '15m' ? instr.tf15m : timeframe === '1h' ? instr.tf1h : instr.tf1d
      const weight = concMap.get(sym) ?? 0.05  // default 5% for display
      const cfg = REGIME_COLORS[tfState.regime]

      return {
        name: sym,
        symbol: sym,
        value: Math.max(weight, 0.02),
        regime: tfState.regime,
        mass: tfState.mass,
        price: instr.price,
        color: cfg.bg,
      }
    })
  }, [instruments, timeframe])

  // Summary counts
  const regimeCounts = useMemo(() => {
    const counts: Record<BHRegime, number> = { BULL: 0, BEAR: 0, SIDEWAYS: 0, HIGH_VOL: 0 }
    for (const d of treeData) counts[d.regime]++
    return counts
  }, [treeData])

  return (
    <div className={`flex flex-col bg-terminal-bg ${className}`}>
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border flex-shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-terminal-subtle text-xs font-mono uppercase">Regime Map</span>
          <div className="flex gap-2 text-[10px] font-mono">
            {(Object.entries(regimeCounts) as [BHRegime, number][]).map(([regime, count]) => {
              if (count === 0) return null
              const cfg = REGIME_COLORS[regime]
              return (
                <span key={regime} style={{ color: cfg.text }}>
                  {regime.slice(0, 2)}: {count}
                </span>
              )
            })}
          </div>
        </div>
        <div className="flex gap-1">
          {(['15m', '1h', '1d'] as const).map((tf) => (
            <button key={tf} onClick={() => setTimeframe(tf)}
              className={`text-[10px] font-mono px-2 py-0.5 rounded transition-colors ${timeframe === tf ? 'bg-terminal-accent/20 text-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text'}`}
            >{tf}</button>
          ))}
        </div>
      </div>

      <div style={{ height }}>
        {treeData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <Treemap
              data={treeData}
              dataKey="value"
              content={<CustomContent />}
            >
              <Tooltip content={<CustomTooltip />} />
            </Treemap>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-full text-terminal-subtle text-sm">
            No instrument data
          </div>
        )}
      </div>
    </div>
  )
}

export default RegimeMap
