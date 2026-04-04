import React, { useState, useRef, useEffect } from 'react'
import { Search, ChevronDown, X } from 'lucide-react'
import { clsx } from 'clsx'
import { INSTRUMENTS } from '@/api/mockData'

interface InstrumentSelectorProps {
  value: string | null
  onChange: (instrument: string | null) => void
  placeholder?: string
  className?: string
  instruments?: string[]
}

export function InstrumentSelector({
  value,
  onChange,
  placeholder = 'All instruments',
  className,
  instruments = INSTRUMENTS,
}: InstrumentSelectorProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  const filtered = instruments.filter(i =>
    i.toLowerCase().includes(query.toLowerCase())
  )

  return (
    <div className={clsx('relative', className)}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 text-sm bg-research-surface border border-research-border rounded hover:border-research-accent/50 transition-colors text-research-text min-w-[160px]"
      >
        <span className="font-mono text-xs flex-1 text-left">
          {value ?? placeholder}
        </span>
        {value ? (
          <X
            size={12}
            className="text-research-subtle hover:text-research-bear shrink-0"
            onClick={e => { e.stopPropagation(); onChange(null) }}
          />
        ) : (
          <ChevronDown size={12} className="text-research-subtle shrink-0" />
        )}
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-1 z-50 bg-research-card border border-research-border rounded-lg shadow-xl w-56 animate-fade-in">
          <div className="p-2 border-b border-research-border">
            <div className="flex items-center gap-2 bg-research-surface rounded px-2">
              <Search size={12} className="text-research-subtle" />
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Search..."
                className="flex-1 bg-transparent py-1.5 text-xs text-research-text placeholder-research-subtle focus:outline-none font-mono"
              />
            </div>
          </div>

          <div className="max-h-52 overflow-y-auto py-1">
            <button
              onClick={() => { onChange(null); setOpen(false); setQuery('') }}
              className="w-full px-3 py-1.5 text-xs text-left text-research-subtle hover:bg-research-muted transition-colors"
            >
              {placeholder}
            </button>
            {filtered.map(inst => (
              <button
                key={inst}
                onClick={() => { onChange(inst); setOpen(false); setQuery('') }}
                className={clsx(
                  'w-full px-3 py-1.5 text-xs text-left font-mono hover:bg-research-muted transition-colors',
                  inst === value ? 'text-research-accent bg-research-accent/10' : 'text-research-text'
                )}
              >
                {inst}
              </button>
            ))}
          </div>
        </div>
      )}

      {open && <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />}
    </div>
  )
}
