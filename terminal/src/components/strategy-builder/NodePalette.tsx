// ============================================================
// NodePalette — draggable node library
// ============================================================
import React, { useState, useCallback } from 'react'
import { NODE_DEFINITIONS } from '@/hooks/useStrategyBuilder'
import type { NodeCategory, NodeDefinition } from '@/types'

const CATEGORY_ORDER: NodeCategory[] = ['Indicators', 'Signals', 'Filters', 'Sizers', 'Logic', 'Outputs']
const CATEGORY_ICONS: Record<NodeCategory, string> = {
  Indicators: '📊',
  Signals: '⚡',
  Filters: '🔍',
  Sizers: '⚖️',
  Logic: '🔀',
  Sources: '📡',
  Transforms: '🔄',
  Outputs: '🎯',
}

interface NodePaletteProps {
  onNodeDragStart?: (definitionType: string, e: React.DragEvent) => void
  onNodeClick?: (definitionType: string) => void
  className?: string
}

function NodeCard({ def, onDragStart, onClick }: {
  def: NodeDefinition
  onDragStart: (e: React.DragEvent) => void
  onClick: () => void
}) {
  return (
    <div
      draggable
      onDragStart={onDragStart}
      onClick={onClick}
      className="group flex items-start gap-2 p-2 rounded border border-terminal-border bg-terminal-surface hover:border-terminal-accent/50 hover:bg-terminal-surface cursor-grab active:cursor-grabbing transition-all"
      title={def.description}
    >
      <div
        className="w-2 h-2 rounded-sm mt-0.5 flex-shrink-0"
        style={{ backgroundColor: def.color }}
      />
      <div className="flex-1 min-w-0">
        <div className="font-mono text-[11px] text-terminal-text font-medium truncate">{def.label}</div>
        <div className="font-mono text-[9px] text-terminal-subtle truncate">{def.description}</div>
        {def.params.length > 0 && (
          <div className="text-[9px] text-terminal-muted mt-0.5">
            {def.params.slice(0, 2).map((p) => p.key).join(', ')}
            {def.params.length > 2 && ` +${def.params.length - 2}`}
          </div>
        )}
      </div>
    </div>
  )
}

export const NodePalette: React.FC<NodePaletteProps> = ({
  onNodeDragStart,
  onNodeClick,
  className = '',
}) => {
  const [search, setSearch] = useState('')
  const [openCategories, setOpenCategories] = useState<Set<NodeCategory>>(
    new Set(['Indicators', 'Signals'])
  )

  const handleDragStart = useCallback((defType: string, e: React.DragEvent) => {
    e.dataTransfer.setData('application/node-type', defType)
    e.dataTransfer.effectAllowed = 'copy'
    onNodeDragStart?.(defType, e)
  }, [onNodeDragStart])

  const toggleCategory = useCallback((cat: NodeCategory) => {
    setOpenCategories((prev) => {
      const next = new Set(prev)
      if (next.has(cat)) next.delete(cat)
      else next.add(cat)
      return next
    })
  }, [])

  const filteredDefs = search
    ? NODE_DEFINITIONS.filter((d) =>
        d.label.toLowerCase().includes(search.toLowerCase()) ||
        d.description.toLowerCase().includes(search.toLowerCase()) ||
        d.type.includes(search.toLowerCase())
      )
    : null

  return (
    <div className={`flex flex-col bg-terminal-bg border-r border-terminal-border h-full ${className}`}>
      {/* Header */}
      <div className="px-2 py-2 border-b border-terminal-border flex-shrink-0">
        <span className="text-terminal-subtle text-xs font-mono uppercase tracking-wider block mb-1.5">Node Library</span>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search nodes..."
          className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-[11px] font-mono text-terminal-text placeholder-terminal-muted focus:outline-none focus:border-terminal-accent"
        />
      </div>

      {/* Node list */}
      <div className="flex-1 overflow-y-auto">
        {filteredDefs ? (
          // Search results
          <div className="p-2 space-y-1">
            {filteredDefs.length === 0 ? (
              <div className="text-center py-4 text-terminal-subtle text-xs">No nodes found</div>
            ) : (
              filteredDefs.map((def) => (
                <NodeCard
                  key={def.type}
                  def={def}
                  onDragStart={(e) => handleDragStart(def.type, e)}
                  onClick={() => onNodeClick?.(def.type)}
                />
              ))
            )}
          </div>
        ) : (
          // Categories
          CATEGORY_ORDER.map((category) => {
            const defs = NODE_DEFINITIONS.filter((d) => d.category === category)
            if (defs.length === 0) return null
            const isOpen = openCategories.has(category)

            return (
              <div key={category}>
                <button
                  onClick={() => toggleCategory(category)}
                  className="w-full flex items-center justify-between px-2 py-1.5 text-[10px] font-mono text-terminal-subtle hover:text-terminal-text hover:bg-terminal-surface transition-colors border-b border-terminal-border/50"
                >
                  <span className="flex items-center gap-1.5">
                    <span>{CATEGORY_ICONS[category] ?? ''}</span>
                    <span className="uppercase tracking-wider">{category}</span>
                    <span className="text-terminal-muted">({defs.length})</span>
                  </span>
                  <span className={`transition-transform ${isOpen ? 'rotate-90' : ''}`}>›</span>
                </button>

                {isOpen && (
                  <div className="p-2 space-y-1">
                    {defs.map((def) => (
                      <NodeCard
                        key={def.type}
                        def={def}
                        onDragStart={(e) => handleDragStart(def.type, e)}
                        onClick={() => onNodeClick?.(def.type)}
                      />
                    ))}
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>

      {/* Footer hint */}
      <div className="px-2 py-1.5 border-t border-terminal-border flex-shrink-0">
        <p className="text-[9px] font-mono text-terminal-muted">Drag onto canvas to add node</p>
      </div>
    </div>
  )
}

export default NodePalette
