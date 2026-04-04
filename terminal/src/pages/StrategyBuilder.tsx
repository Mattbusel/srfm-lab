// ============================================================
// StrategyBuilder — visual strategy builder page
// ============================================================
import React, { useState, useCallback } from 'react'
import { useStrategyStore, selectActiveGraph } from '@/store/strategyStore'
import { StrategyCanvas } from '@/components/strategy-builder/StrategyCanvas'
import { BacktestPanel } from '@/components/strategy-builder/BacktestPanel'
import { motion, AnimatePresence } from 'framer-motion'

type SidebarMode = 'strategies' | 'backtest'

export const StrategyBuilder: React.FC = () => {
  const store = useStrategyStore()
  const activeGraph = useStrategyStore(selectActiveGraph)
  const graphs = useStrategyStore((s) => s.graphs)

  const [sidebarMode, setSidebarMode] = useState<SidebarMode>('strategies')
  const [newName, setNewName] = useState('')
  const [showNewInput, setShowNewInput] = useState(false)
  const [importJson, setImportJson] = useState('')
  const [showImport, setShowImport] = useState(false)

  const handleCreate = useCallback(() => {
    if (!newName.trim()) return
    store.createGraph(newName.trim())
    setNewName('')
    setShowNewInput(false)
  }, [newName, store])

  const handleExport = useCallback(() => {
    if (!activeGraph) return
    const json = store.exportStrategy(activeGraph.id)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${activeGraph.metadata.name.replace(/\s+/g, '-')}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [activeGraph, store])

  const handleImport = useCallback(() => {
    if (!importJson.trim()) return
    const id = store.importStrategy(importJson)
    if (id) {
      setShowImport(false)
      setImportJson('')
    }
  }, [importJson, store])

  return (
    <div className="flex h-full bg-terminal-bg">
      {/* Left sidebar: strategy list + controls */}
      <div className="w-48 flex-shrink-0 border-r border-terminal-border flex flex-col">
        {/* Header */}
        <div className="px-3 py-2 border-b border-terminal-border flex-shrink-0">
          <div className="flex items-center justify-between mb-2">
            <span className="text-terminal-subtle text-xs font-mono uppercase">Strategies</span>
            <button
              onClick={() => setShowNewInput(!showNewInput)}
              className="text-terminal-subtle hover:text-terminal-accent text-xs transition-colors"
            >
              + New
            </button>
          </div>

          <AnimatePresence>
            {showNewInput && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden mb-2">
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleCreate() }}
                  placeholder="Strategy name..."
                  autoFocus
                  className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent mb-1"
                />
                <button onClick={handleCreate} className="w-full bg-terminal-accent text-white rounded py-1 text-xs font-mono hover:bg-terminal-accent-dim transition-colors">
                  Create
                </button>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Toolbar buttons */}
          <div className="flex flex-wrap gap-1">
            {activeGraph && (
              <>
                <button onClick={handleExport} className="text-[10px] font-mono text-terminal-subtle hover:text-terminal-accent transition-colors">Export</button>
                <span className="text-terminal-muted">·</span>
                <button onClick={() => setShowImport(!showImport)} className="text-[10px] font-mono text-terminal-subtle hover:text-terminal-accent transition-colors">Import</button>
                <span className="text-terminal-muted">·</span>
                <button onClick={() => store.duplicateGraph(activeGraph.id)} className="text-[10px] font-mono text-terminal-subtle hover:text-terminal-accent transition-colors">Dup</button>
              </>
            )}
          </div>

          <AnimatePresence>
            {showImport && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden mt-2">
                <textarea
                  value={importJson}
                  onChange={(e) => setImportJson(e.target.value)}
                  placeholder="Paste strategy JSON..."
                  className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-[10px] font-mono text-terminal-text focus:outline-none focus:border-terminal-accent resize-none h-20"
                />
                <button onClick={handleImport} className="w-full bg-terminal-accent text-white rounded py-1 text-[10px] font-mono hover:bg-terminal-accent-dim transition-colors mt-1">
                  Import
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Strategy list */}
        <div className="flex-1 overflow-y-auto">
          {graphs.length === 0 ? (
            <div className="flex items-center justify-center py-8 text-terminal-subtle text-xs text-center px-3">
              Create a strategy to get started
            </div>
          ) : (
            graphs.map((graph) => (
              <div
                key={graph.id}
                onClick={() => store.setActiveGraph(graph.id)}
                className={`px-3 py-2 cursor-pointer border-b border-terminal-border/30 hover:bg-terminal-surface transition-colors group ${
                  graph.id === activeGraph?.id ? 'bg-terminal-accent/10 border-l-2 border-l-terminal-accent' : 'border-l-2 border-l-transparent'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-terminal-text truncate flex-1">{graph.metadata.name}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); if (confirm(`Delete "${graph.metadata.name}"?`)) store.deleteGraph(graph.id) }}
                    className="opacity-0 group-hover:opacity-60 hover:!opacity-100 text-terminal-bear text-[10px] ml-1 transition-opacity"
                  >
                    ✕
                  </button>
                </div>
                <div className="flex items-center gap-2 mt-0.5 text-[10px] font-mono text-terminal-subtle">
                  <span>{graph.nodes.length} nodes</span>
                  <span>v{graph.metadata.version}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main canvas area */}
      <div className="flex-1 flex flex-col min-w-0">
        {activeGraph ? (
          <>
            {/* Strategy name bar */}
            <div className="flex items-center justify-between px-3 py-1.5 border-b border-terminal-border bg-terminal-surface/50 flex-shrink-0">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm font-semibold text-terminal-text">{activeGraph.metadata.name}</span>
                <span className="text-[10px] font-mono text-terminal-subtle">v{activeGraph.metadata.version}</span>
                {activeGraph.metadata.description && (
                  <span className="text-[10px] font-mono text-terminal-muted">— {activeGraph.metadata.description}</span>
                )}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => setSidebarMode(m => m === 'backtest' ? 'strategies' : 'backtest')}
                  className={`text-[10px] font-mono px-2 py-1 rounded border transition-colors ${
                    sidebarMode === 'backtest'
                      ? 'bg-terminal-accent/20 text-terminal-accent border-terminal-accent/40'
                      : 'text-terminal-subtle border-terminal-border hover:text-terminal-text'
                  }`}
                >
                  Backtest
                </button>
              </div>
            </div>

            <div className="flex-1 flex min-h-0">
              {/* Canvas */}
              <div className="flex-1 min-w-0">
                <StrategyCanvas graphId={activeGraph.id} />
              </div>

              {/* Backtest sidebar */}
              <AnimatePresence>
                {sidebarMode === 'backtest' && (
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: 320 }}
                    exit={{ width: 0 }}
                    className="flex-shrink-0 border-l border-terminal-border overflow-hidden"
                  >
                    <BacktestPanel graphId={activeGraph.id} />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center text-terminal-subtle">
              <div className="text-3xl mb-3">⬡</div>
              <div className="text-sm font-mono">Select or create a strategy to begin</div>
              <button
                onClick={() => { store.createGraph('My First Strategy'); setShowNewInput(false) }}
                className="mt-4 bg-terminal-accent text-white rounded px-4 py-2 text-xs font-mono hover:bg-terminal-accent-dim transition-colors"
              >
                Create New Strategy
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default StrategyBuilder
