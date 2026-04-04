// ============================================================
// StrategyCanvas — visual drag-and-drop node graph editor
// ============================================================
import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { useStrategyStore } from '@/store/strategyStore'
import { useStrategyBuilder, NODE_DEFINITIONS } from '@/hooks/useStrategyBuilder'
import { NodeEditor } from './NodeEditor'
import { NodePalette } from './NodePalette'
import type { StrategyNode, StrategyEdge } from '@/types'

const GRID_SIZE = 20
const NODE_WIDTH = 160
const NODE_HEIGHT = 80

function snapToGrid(v: number): number {
  return Math.round(v / GRID_SIZE) * GRID_SIZE
}

const generateId = () => Math.random().toString(36).slice(2, 11)

// ---- Edge renderer ----
function EdgeSVG({ edges, nodes, hoveredEdge, onEdgeClick }: {
  edges: StrategyEdge[]
  nodes: StrategyNode[]
  hoveredEdge: string | null
  onEdgeClick: (edgeId: string) => void
}) {
  const nodeMap = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes])

  const getHandlePos = (nodeId: string, handle: string, side: 'source' | 'target') => {
    const node = nodeMap.get(nodeId)
    if (!node) return { x: 0, y: 0 }
    const x = side === 'source' ? node.position.x + NODE_WIDTH : node.position.x
    // Find handle index for vertical positioning
    const def = NODE_DEFINITIONS.find((d) => d.type === node.definitionType)
    const handles = side === 'source' ? (def?.outputs ?? []) : (def?.inputs ?? [])
    const idx = handles.findIndex((h) => h.id === handle)
    const total = handles.length || 1
    const y = node.position.y + NODE_HEIGHT * ((idx + 1) / (total + 1))
    return { x, y }
  }

  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ overflow: 'visible' }}>
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#4b5563" />
        </marker>
        <marker id="arrowhead-hover" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#3b82f6" />
        </marker>
      </defs>
      {edges.map((edge) => {
        const src = getHandlePos(edge.source, edge.sourceHandle, 'source')
        const tgt = getHandlePos(edge.target, edge.targetHandle, 'target')
        const cx = (src.x + tgt.x) / 2
        const isHovered = hoveredEdge === edge.id
        const path = `M ${src.x} ${src.y} C ${cx} ${src.y}, ${cx} ${tgt.y}, ${tgt.x} ${tgt.y}`

        return (
          <g key={edge.id} className="pointer-events-auto">
            {/* Invisible wide hit area */}
            <path
              d={path}
              fill="none"
              stroke="transparent"
              strokeWidth={12}
              onClick={() => onEdgeClick(edge.id)}
              className="cursor-pointer"
            />
            {/* Visible edge */}
            <path
              d={path}
              fill="none"
              stroke={isHovered ? '#3b82f6' : '#374151'}
              strokeWidth={isHovered ? 2 : 1.5}
              markerEnd={isHovered ? 'url(#arrowhead-hover)' : 'url(#arrowhead)'}
              strokeDasharray={edge.animated ? '5 3' : undefined}
            />
          </g>
        )
      })}
    </svg>
  )
}

// ---- Node component ----
function CanvasNode({
  node,
  isSelected,
  onSelect,
  onDragEnd,
  onStartConnection,
  zoom,
}: {
  node: StrategyNode
  isSelected: boolean
  onSelect: (id: string) => void
  onDragEnd: (id: string, pos: { x: number; y: number }) => void
  onStartConnection: (nodeId: string, handle: string, side: 'source' | 'target', e: React.MouseEvent) => void
  zoom: number
}) {
  const def = NODE_DEFINITIONS.find((d) => d.type === node.definitionType)
  const dragRef = useRef<{ startX: number; startY: number; nodeX: number; nodeY: number } | null>(null)
  const posRef = useRef(node.position)
  const [isDragging, setIsDragging] = useState(false)
  const [localPos, setLocalPos] = useState(node.position)

  useEffect(() => {
    setLocalPos(node.position)
    posRef.current = node.position
  }, [node.position])

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).dataset.handle) return
    e.stopPropagation()
    onSelect(node.id)
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      nodeX: posRef.current.x,
      nodeY: posRef.current.y,
    }
    setIsDragging(true)

    const handleMouseMove = (me: MouseEvent) => {
      if (!dragRef.current) return
      const dx = (me.clientX - dragRef.current.startX) / zoom
      const dy = (me.clientY - dragRef.current.startY) / zoom
      const newPos = {
        x: snapToGrid(dragRef.current.nodeX + dx),
        y: snapToGrid(dragRef.current.nodeY + dy),
      }
      posRef.current = newPos
      setLocalPos(newPos)
    }

    const handleMouseUp = () => {
      onDragEnd(node.id, posRef.current)
      setIsDragging(false)
      dragRef.current = null
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }, [node.id, onSelect, onDragEnd, zoom])

  const headerColor = def?.color ?? '#374151'
  const isDisabled = node.disabled

  return (
    <div
      onMouseDown={handleMouseDown}
      className={`absolute select-none rounded-lg border-2 overflow-visible bg-terminal-surface transition-shadow ${
        isSelected ? 'border-terminal-accent shadow-lg shadow-terminal-accent/20' : 'border-terminal-border'
      } ${isDragging ? 'shadow-xl opacity-90 cursor-grabbing' : 'cursor-grab'} ${isDisabled ? 'opacity-50' : ''}`}
      style={{
        left: localPos.x,
        top: localPos.y,
        width: NODE_WIDTH,
        zIndex: isSelected ? 10 : 1,
        minHeight: NODE_HEIGHT,
      }}
    >
      {/* Node header */}
      <div
        className="px-2 py-1 rounded-t text-white text-[10px] font-mono font-bold flex items-center justify-between"
        style={{ backgroundColor: headerColor }}
      >
        <span className="truncate">{node.name}</span>
        {node.error && <span className="text-terminal-bear text-[9px]">⚠</span>}
      </div>

      {/* Input handles */}
      {def?.inputs.map((input, i) => {
        const total = def.inputs.length
        const yPct = (i + 1) / (total + 1)
        return (
          <div
            key={input.id}
            data-handle="true"
            className="absolute w-3 h-3 rounded-full bg-terminal-surface border-2 border-terminal-border hover:border-terminal-accent cursor-crosshair transition-colors"
            style={{
              left: -6,
              top: `${yPct * 100}%`,
              transform: 'translateY(-50%)',
            }}
            onMouseDown={(e) => {
              e.stopPropagation()
              onStartConnection(node.id, input.id, 'target', e)
            }}
            title={`${input.label} (${input.dataType})`}
          />
        )
      })}

      {/* Node body */}
      <div className="px-2 py-1.5 min-h-[40px]">
        {def ? (
          <div className="space-y-0.5">
            {Object.entries(node.params).slice(0, 2).map(([k, v]) => (
              <div key={k} className="flex justify-between text-[9px] font-mono">
                <span className="text-terminal-subtle">{k}:</span>
                <span className="text-terminal-text">{String(v)}</span>
              </div>
            ))}
            {Object.keys(node.params).length > 2 && (
              <div className="text-[9px] font-mono text-terminal-muted">+{Object.keys(node.params).length - 2} more</div>
            )}
          </div>
        ) : (
          <div className="text-[9px] font-mono text-terminal-subtle">{node.definitionType}</div>
        )}
      </div>

      {/* Output handles */}
      {def?.outputs.map((output, i) => {
        const total = def.outputs.length
        const yPct = (i + 1) / (total + 1)
        return (
          <div
            key={output.id}
            data-handle="true"
            className="absolute w-3 h-3 rounded-full bg-terminal-accent border-2 border-terminal-accent-dim hover:scale-110 cursor-crosshair transition-all"
            style={{
              right: -6,
              top: `${yPct * 100}%`,
              transform: 'translateY(-50%)',
            }}
            onMouseDown={(e) => {
              e.stopPropagation()
              onStartConnection(node.id, output.id, 'source', e)
            }}
            title={`${output.label} (${output.dataType})`}
          />
        )
      })}
    </div>
  )
}

// ---- Mini-map ----
function MiniMap({ nodes, viewport, canvasSize }: {
  nodes: StrategyNode[]
  viewport: { x: number; y: number; zoom: number }
  canvasSize: { width: number; height: number }
}) {
  const mapWidth = 140
  const mapHeight = 90
  const scale = mapWidth / 3000

  return (
    <div className="absolute bottom-4 right-4 bg-terminal-surface/90 border border-terminal-border rounded overflow-hidden" style={{ width: mapWidth, height: mapHeight }}>
      <svg width={mapWidth} height={mapHeight}>
        {nodes.map((node) => (
          <rect
            key={node.id}
            x={node.position.x * scale}
            y={node.position.y * scale}
            width={NODE_WIDTH * scale}
            height={NODE_HEIGHT * scale}
            fill={NODE_DEFINITIONS.find((d) => d.type === node.definitionType)?.color ?? '#374151'}
            opacity={0.7}
            rx={1}
          />
        ))}
        {/* Viewport rect */}
        <rect
          x={(-viewport.x / viewport.zoom) * scale}
          y={(-viewport.y / viewport.zoom) * scale}
          width={(canvasSize.width / viewport.zoom) * scale}
          height={(canvasSize.height / viewport.zoom) * scale}
          fill="none"
          stroke="#3b82f6"
          strokeWidth={1}
          opacity={0.5}
        />
      </svg>
    </div>
  )
}

// ---- Main Canvas ----
interface StrategyCanvasProps {
  graphId: string | null
  className?: string
}

export const StrategyCanvas: React.FC<StrategyCanvasProps> = ({ graphId, className = '' }) => {
  const { graph, addNodeFromDefinition, connectNodes, moveNode, removeNode, removeEdge, setViewport, validateGraph } = useStrategyBuilder(graphId)
  const canvasRef = useRef<HTMLDivElement>(null)
  const [viewport, setVp] = useState({ x: 0, y: 0, zoom: 1 })
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null)
  const [isPanning, setIsPanning] = useState(false)
  const [panStart, setPanStart] = useState({ x: 0, y: 0 })
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 })
  const [connecting, setConnecting] = useState<{ nodeId: string; handle: string; side: 'source' | 'target' } | null>(null)
  const [showSnapGrid, setShowSnapGrid] = useState(true)
  const pendingEdgeRef = useRef<{ start: { x: number; y: number }; end: { x: number; y: number } } | null>(null)

  const selectedNode = graph?.nodes.find((n) => n.id === selectedNodeId) ?? null

  // Observe canvas size
  useEffect(() => {
    if (!canvasRef.current) return
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) {
        setCanvasSize({ width: entry.contentRect.width, height: entry.contentRect.height })
      }
    })
    ro.observe(canvasRef.current)
    return () => ro.disconnect()
  }, [])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedNodeId && graphId) {
          removeNode(selectedNodeId)
          setSelectedNodeId(null)
        }
      }
      if (e.key === 'Escape') {
        setSelectedNodeId(null)
        setConnecting(null)
      }
      if (e.ctrlKey && e.key === '0') {
        setVp({ x: 0, y: 0, zoom: 1 })
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [selectedNodeId, graphId, removeNode])

  // Drag from palette onto canvas
  const handleCanvasDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const defType = e.dataTransfer.getData('application/node-type')
    if (!defType || !graphId || !canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = snapToGrid((e.clientX - rect.left - viewport.x) / viewport.zoom)
    const y = snapToGrid((e.clientY - rect.top - viewport.y) / viewport.zoom)

    addNodeFromDefinition(defType, { x: Math.max(0, x), y: Math.max(0, y) })
  }, [graphId, viewport, addNodeFromDefinition])

  // Pan
  const handleCanvasMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 1 && !(e.button === 0 && e.altKey)) return
    e.preventDefault()
    setIsPanning(true)
    setPanStart({ x: e.clientX - viewport.x, y: e.clientY - viewport.y })
  }, [viewport])

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isPanning) return
    const newVp = { ...viewport, x: e.clientX - panStart.x, y: e.clientY - panStart.y }
    setVp(newVp)
  }, [isPanning, panStart, viewport])

  const handleCanvasMouseUp = useCallback(() => {
    if (isPanning) {
      setIsPanning(false)
      setViewport(viewport)
    }
    setConnecting(null)
  }, [isPanning, viewport, setViewport])

  // Zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY * -0.001
    const newZoom = Math.max(0.2, Math.min(2, viewport.zoom + delta))
    setVp((prev) => ({ ...prev, zoom: newZoom }))
  }, [viewport.zoom])

  const handleNodeDragEnd = useCallback((nodeId: string, pos: { x: number; y: number }) => {
    moveNode(nodeId, pos)
  }, [moveNode])

  const handleStartConnection = useCallback((nodeId: string, handle: string, side: 'source' | 'target', e: React.MouseEvent) => {
    e.stopPropagation()
    if (connecting) {
      // Complete connection
      if (connecting.side !== side && connecting.nodeId !== nodeId) {
        const [src, tgt] = connecting.side === 'source'
          ? [connecting, { nodeId, handle, side }]
          : [{ nodeId, handle, side }, connecting]
        connectNodes(src.nodeId, src.handle, tgt.nodeId, tgt.handle)
      }
      setConnecting(null)
    } else {
      setConnecting({ nodeId, handle, side })
    }
  }, [connecting, connectNodes])

  const handleEdgeClick = useCallback((edgeId: string) => {
    if (graphId) removeEdge(edgeId)
  }, [graphId, removeEdge])

  const validation = useMemo(() => validateGraph(), [graph, validateGraph])

  const nodeCount = graph?.nodes.length ?? 0
  const edgeCount = graph?.edges.length ?? 0

  return (
    <div className={`flex h-full bg-terminal-bg ${className}`}>
      {/* Node Palette sidebar */}
      <div className="w-48 flex-shrink-0">
        <NodePalette
          onNodeDragStart={() => {}}
          onNodeClick={(defType) => {
            if (!graphId) return
            const pos = {
              x: snapToGrid((-viewport.x + 200) / viewport.zoom),
              y: snapToGrid((-viewport.y + 100) / viewport.zoom),
            }
            addNodeFromDefinition(defType, pos)
          }}
        />
      </div>

      {/* Canvas area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <div className="flex items-center justify-between px-3 py-1.5 border-b border-terminal-border flex-shrink-0 bg-terminal-surface/50">
          <div className="flex items-center gap-3 text-[10px] font-mono text-terminal-subtle">
            <span>{nodeCount} nodes</span>
            <span>{edgeCount} edges</span>
            {!validation.valid && (
              <span className="text-terminal-warning" title={validation.errors.join('\n')}>
                ⚠ {validation.errors.length} issue{validation.errors.length > 1 ? 's' : ''}
              </span>
            )}
            {connecting && (
              <span className="text-terminal-accent animate-pulse">
                Connecting... (click target handle or ESC to cancel)
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setShowSnapGrid(!showSnapGrid)} className={`text-[10px] font-mono px-2 py-0.5 rounded transition-colors ${showSnapGrid ? 'text-terminal-accent' : 'text-terminal-subtle'}`}>
              Grid
            </button>
            <button onClick={() => setVp({ x: 0, y: 0, zoom: 1 })} className="text-[10px] font-mono px-2 py-0.5 rounded text-terminal-subtle hover:text-terminal-text transition-colors">
              Reset View
            </button>
            <span className="text-[10px] font-mono text-terminal-subtle">{(viewport.zoom * 100).toFixed(0)}%</span>
            <button onClick={() => setVp(v => ({ ...v, zoom: Math.min(2, v.zoom + 0.1) }))} className="text-terminal-subtle hover:text-terminal-text w-5 text-center">+</button>
            <button onClick={() => setVp(v => ({ ...v, zoom: Math.max(0.2, v.zoom - 0.1) }))} className="text-terminal-subtle hover:text-terminal-text w-5 text-center">−</button>
          </div>
        </div>

        {/* Canvas */}
        <div
          ref={canvasRef}
          className={`flex-1 relative overflow-hidden ${isPanning ? 'cursor-grabbing' : connecting ? 'cursor-crosshair' : 'cursor-default'}`}
          onDrop={handleCanvasDrop}
          onDragOver={(e) => e.preventDefault()}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onWheel={handleWheel}
          onClick={(e) => {
            if (e.target === e.currentTarget) setSelectedNodeId(null)
          }}
        >
          {/* Background grid */}
          {showSnapGrid && (
            <svg className="absolute inset-0 w-full h-full pointer-events-none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <pattern id="grid" width={GRID_SIZE * viewport.zoom} height={GRID_SIZE * viewport.zoom} x={(viewport.x % (GRID_SIZE * viewport.zoom))} y={(viewport.y % (GRID_SIZE * viewport.zoom))} patternUnits="userSpaceOnUse">
                  <path d={`M ${GRID_SIZE * viewport.zoom} 0 L 0 0 0 ${GRID_SIZE * viewport.zoom}`} fill="none" stroke="#1f2937" strokeWidth={0.5} />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
            </svg>
          )}

          {/* Transformed canvas content */}
          <div
            className="absolute origin-top-left"
            style={{ transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})` }}
          >
            {/* Edges SVG */}
            {graph && (
              <EdgeSVG
                edges={graph.edges}
                nodes={graph.nodes}
                hoveredEdge={hoveredEdge}
                onEdgeClick={handleEdgeClick}
              />
            )}

            {/* Nodes */}
            {graph?.nodes.map((node) => (
              <CanvasNode
                key={node.id}
                node={node}
                isSelected={selectedNodeId === node.id}
                onSelect={setSelectedNodeId}
                onDragEnd={handleNodeDragEnd}
                onStartConnection={handleStartConnection}
                zoom={viewport.zoom}
              />
            ))}

            {/* Empty state */}
            {(!graph || graph.nodes.length === 0) && (
              <div className="absolute inset-0 flex items-center justify-center" style={{ width: 800, height: 600 }}>
                <div className="text-center text-terminal-subtle">
                  <div className="text-2xl mb-2">⬡</div>
                  <div className="text-sm font-mono">Drag nodes from the palette or click to add</div>
                  <div className="text-xs mt-1">Alt+drag to pan · Scroll to zoom · Delete to remove selected</div>
                </div>
              </div>
            )}
          </div>

          {/* Mini-map */}
          {graph && graph.nodes.length > 0 && (
            <MiniMap nodes={graph.nodes} viewport={viewport} canvasSize={canvasSize} />
          )}
        </div>
      </div>

      {/* Node Editor sidebar */}
      <div className="w-56 flex-shrink-0 border-l border-terminal-border">
        <NodeEditor
          graphId={graphId}
          selectedNode={selectedNode}
          onClose={() => setSelectedNodeId(null)}
        />
      </div>
    </div>
  )
}

export default StrategyCanvas
