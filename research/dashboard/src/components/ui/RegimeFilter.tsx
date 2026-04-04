import React from 'react'
import { clsx } from 'clsx'
import { REGIME_COLORS, REGIME_LABELS } from '@/utils/colors'
import type { RegimeType } from '@/types/trades'

const ALL_REGIMES: RegimeType[] = ['bull', 'bear', 'sideways', 'ranging', 'volatile']

interface RegimeFilterProps {
  selected: RegimeType[]
  onChange: (regimes: RegimeType[]) => void
  className?: string
}

export function RegimeFilter({ selected, onChange, className }: RegimeFilterProps) {
  const toggle = (regime: RegimeType) => {
    if (selected.includes(regime)) {
      onChange(selected.filter(r => r !== regime))
    } else {
      onChange([...selected, regime])
    }
  }

  const selectAll = () => onChange([...ALL_REGIMES])
  const clearAll = () => onChange([])

  return (
    <div className={clsx('flex items-center gap-2 flex-wrap', className)}>
      <span className="text-xs text-research-subtle">Regimes:</span>
      {ALL_REGIMES.map(regime => {
        const active = selected.includes(regime)
        return (
          <button
            key={regime}
            onClick={() => toggle(regime)}
            className={clsx(
              'px-2 py-0.5 rounded text-xs font-medium transition-all border',
              active
                ? 'text-white border-transparent'
                : 'text-research-subtle border-research-border bg-transparent hover:border-research-muted'
            )}
            style={active ? { backgroundColor: REGIME_COLORS[regime], borderColor: REGIME_COLORS[regime] } : {}}
          >
            {REGIME_LABELS[regime]}
          </button>
        )
      })}
      <div className="flex gap-1 ml-1">
        <button
          onClick={selectAll}
          className="text-xs text-research-subtle hover:text-research-text px-1"
        >
          All
        </button>
        <span className="text-research-muted">|</span>
        <button
          onClick={clearAll}
          className="text-xs text-research-subtle hover:text-research-text px-1"
        >
          None
        </button>
      </div>
    </div>
  )
}
