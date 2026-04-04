// ============================================================
// NodeEditor — parameter editor for selected strategy node
// ============================================================
import React, { useCallback, useMemo } from 'react'
import { useStrategyBuilder, NODE_DEFINITIONS } from '@/hooks/useStrategyBuilder'
import type { StrategyNode, NodeParamDef } from '@/types'

interface NodeEditorProps {
  graphId: string | null
  selectedNode: StrategyNode | null
  onClose?: () => void
  className?: string
}

function ParamField({ param, value, onChange }: {
  param: NodeParamDef
  value: number | string | boolean
  onChange: (key: string, value: number | string | boolean) => void
}) {
  const handleChange = useCallback((newVal: number | string | boolean) => {
    onChange(param.key, newVal)
  }, [param.key, onChange])

  if (param.type === 'boolean') {
    return (
      <div className="flex items-center justify-between">
        <label className="text-[11px] font-mono text-terminal-subtle flex-1">{param.label}</label>
        <button
          onClick={() => handleChange(!value)}
          className={`w-8 h-4 rounded-full transition-colors relative ${value ? 'bg-terminal-accent' : 'bg-terminal-muted'}`}
        >
          <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform ${value ? 'right-0.5' : 'left-0.5'}`} />
        </button>
      </div>
    )
  }

  if (param.type === 'select') {
    return (
      <div>
        <label className="text-[11px] font-mono text-terminal-subtle block mb-0.5">{param.label}</label>
        <select
          value={String(value)}
          onChange={(e) => handleChange(e.target.value)}
          className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-[11px] font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
        >
          {param.options?.map((opt) => (
            <option key={String(opt.value)} value={String(opt.value)}>{opt.label}</option>
          ))}
        </select>
      </div>
    )
  }

  if (param.type === 'number') {
    const numValue = Number(value)
    const range = param.max !== undefined && param.min !== undefined ? param.max - param.min : 100
    const pct = range > 0 ? ((numValue - (param.min ?? 0)) / range) * 100 : 0

    return (
      <div>
        <div className="flex items-center justify-between mb-0.5">
          <label className="text-[11px] font-mono text-terminal-subtle">{param.label}</label>
          <div className="flex items-center gap-1">
            <input
              type="number"
              value={numValue}
              onChange={(e) => handleChange(parseFloat(e.target.value))}
              min={param.min}
              max={param.max}
              step={param.step ?? 1}
              className="w-16 bg-terminal-surface border border-terminal-border rounded px-1.5 py-0.5 text-[11px] font-mono text-terminal-text focus:outline-none focus:border-terminal-accent text-right"
            />
            {param.unit && <span className="text-[9px] text-terminal-subtle">{param.unit}</span>}
          </div>
        </div>
        {param.min !== undefined && param.max !== undefined && (
          <div className="relative h-1.5 bg-terminal-muted rounded-full">
            <input
              type="range"
              min={param.min}
              max={param.max}
              step={param.step ?? (param.max - param.min) / 100}
              value={numValue}
              onChange={(e) => handleChange(parseFloat(e.target.value))}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div
              className="h-full bg-terminal-accent rounded-full pointer-events-none"
              style={{ width: `${Math.max(0, Math.min(100, pct))}%` }}
            />
            <div
              className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-terminal-accent rounded-full border-2 border-terminal-bg pointer-events-none"
              style={{ left: `calc(${Math.max(0, Math.min(100, pct))}% - 6px)` }}
            />
          </div>
        )}
        {(param.min !== undefined || param.max !== undefined) && (
          <div className="flex justify-between text-[9px] font-mono text-terminal-muted mt-0.5">
            <span>{param.min}</span>
            <span>{param.max}</span>
          </div>
        )}
      </div>
    )
  }

  // String input
  return (
    <div>
      <label className="text-[11px] font-mono text-terminal-subtle block mb-0.5">{param.label}</label>
      <input
        type="text"
        value={String(value)}
        onChange={(e) => handleChange(e.target.value)}
        className="w-full bg-terminal-surface border border-terminal-border rounded px-2 py-1 text-[11px] font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
      />
    </div>
  )
}

export const NodeEditor: React.FC<NodeEditorProps> = ({
  graphId,
  selectedNode,
  onClose,
  className = '',
}) => {
  const { updateNodeParams, getNodeDef } = useStrategyBuilder(graphId)

  const def = useMemo(() => selectedNode ? getNodeDef(selectedNode.definitionType) : null, [selectedNode, getNodeDef])

  const handleParamChange = useCallback((key: string, value: number | string | boolean) => {
    if (!selectedNode || !graphId) return
    updateNodeParams(selectedNode.id, { [key]: value })
  }, [selectedNode, graphId, updateNodeParams])

  const handleReset = useCallback(() => {
    if (!selectedNode || !def) return
    const defaults = Object.fromEntries(def.params.map((p) => [p.key, p.default]))
    updateNodeParams(selectedNode.id, defaults)
  }, [selectedNode, def, updateNodeParams])

  if (!selectedNode || !def) {
    return (
      <div className={`flex flex-col items-center justify-center h-full bg-terminal-bg ${className}`}>
        <div className="text-terminal-subtle text-xs font-mono text-center px-4">
          Select a node on the canvas to edit its parameters
        </div>
      </div>
    )
  }

  return (
    <div className={`flex flex-col bg-terminal-bg h-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border flex-shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: def.color }} />
          <span className="font-mono text-xs font-semibold text-terminal-text">{def.label}</span>
        </div>
        <div className="flex gap-1">
          <button
            onClick={handleReset}
            className="text-[10px] font-mono px-2 py-0.5 rounded text-terminal-subtle border border-terminal-border hover:text-terminal-text transition-colors"
          >
            Reset
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="text-[10px] font-mono px-2 py-0.5 rounded text-terminal-subtle hover:text-terminal-text transition-colors"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Node info */}
      <div className="px-3 py-2 border-b border-terminal-border/50 flex-shrink-0">
        <p className="text-[10px] font-mono text-terminal-subtle">{def.description}</p>
        <div className="flex gap-2 mt-1 text-[9px] font-mono text-terminal-muted">
          {def.inputs.length > 0 && <span>Inputs: {def.inputs.map((i) => i.label).join(', ')}</span>}
          {def.outputs.length > 0 && <span>Outputs: {def.outputs.map((o) => o.label).join(', ')}</span>}
        </div>
      </div>

      {/* Parameters */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {def.params.length === 0 ? (
          <div className="text-center py-4 text-terminal-subtle text-xs font-mono">
            No configurable parameters
          </div>
        ) : (
          def.params.map((param) => (
            <ParamField
              key={param.key}
              param={param}
              value={selectedNode.params[param.key] ?? param.default}
              onChange={handleParamChange}
            />
          ))
        )}

        {/* Node metadata */}
        <div className="border-t border-terminal-border/50 pt-3 space-y-2">
          <div className="text-[10px] font-mono text-terminal-subtle">
            <span>Node ID: </span>
            <span className="text-terminal-muted">{selectedNode.id}</span>
          </div>
          <div className="text-[10px] font-mono text-terminal-subtle">
            <span>Type: </span>
            <span className="text-terminal-muted">{selectedNode.definitionType}</span>
          </div>
          {selectedNode.lastOutput !== undefined && (
            <div className="text-[10px] font-mono text-terminal-subtle">
              <span>Last output: </span>
              <span className={`${typeof selectedNode.lastOutput === 'boolean' ? (selectedNode.lastOutput ? 'text-terminal-bull' : 'text-terminal-bear') : 'text-terminal-text'}`}>
                {String(selectedNode.lastOutput)}
              </span>
            </div>
          )}
          {selectedNode.error && (
            <div className="text-[10px] font-mono text-terminal-bear bg-terminal-bear/10 rounded p-1">
              Error: {selectedNode.error}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default NodeEditor
