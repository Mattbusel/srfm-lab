import React, { useState } from 'react'
import { Calendar, ChevronDown } from 'lucide-react'
import { clsx } from 'clsx'
import { format, subDays, subMonths } from 'date-fns'

interface DateRange {
  from: string
  to: string
}

interface DateRangePickerProps {
  value: DateRange
  onChange: (range: DateRange) => void
  className?: string
}

const PRESETS = [
  { label: '7D', days: 7 },
  { label: '30D', days: 30 },
  { label: '90D', days: 90 },
  { label: '180D', days: 180 },
  { label: '1Y', days: 365 },
]

export function DateRangePicker({ value, onChange, className }: DateRangePickerProps) {
  const [open, setOpen] = useState(false)

  const handlePreset = (days: number) => {
    const to = format(new Date(), 'yyyy-MM-dd')
    const from = format(subDays(new Date(), days), 'yyyy-MM-dd')
    onChange({ from, to })
    setOpen(false)
  }

  return (
    <div className={clsx('relative', className)}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 text-sm bg-research-surface border border-research-border rounded hover:border-research-accent/50 transition-colors text-research-text"
      >
        <Calendar size={14} className="text-research-subtle" />
        <span className="font-mono text-xs">
          {value.from} → {value.to}
        </span>
        <ChevronDown size={12} className="text-research-subtle" />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 bg-research-card border border-research-border rounded-lg shadow-xl p-3 min-w-[260px] animate-fade-in">
          <div className="flex gap-1.5 mb-3">
            {PRESETS.map(p => (
              <button
                key={p.label}
                onClick={() => handlePreset(p.days)}
                className="flex-1 px-2 py-1 text-xs font-mono bg-research-surface hover:bg-research-muted rounded transition-colors text-research-text border border-research-border"
              >
                {p.label}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-research-subtle block mb-1">From</label>
              <input
                type="date"
                value={value.from}
                onChange={e => onChange({ ...value, from: e.target.value })}
                className="w-full bg-research-surface border border-research-border rounded px-2 py-1 text-xs font-mono text-research-text focus:outline-none focus:border-research-accent"
              />
            </div>
            <div>
              <label className="text-xs text-research-subtle block mb-1">To</label>
              <input
                type="date"
                value={value.to}
                onChange={e => onChange({ ...value, to: e.target.value })}
                className="w-full bg-research-surface border border-research-border rounded px-2 py-1 text-xs font-mono text-research-text focus:outline-none focus:border-research-accent"
              />
            </div>
          </div>

          <button
            onClick={() => setOpen(false)}
            className="w-full mt-3 py-1.5 text-xs bg-research-accent hover:bg-research-accent-dim text-white rounded transition-colors"
          >
            Apply
          </button>
        </div>
      )}

      {open && (
        <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
      )}
    </div>
  )
}
