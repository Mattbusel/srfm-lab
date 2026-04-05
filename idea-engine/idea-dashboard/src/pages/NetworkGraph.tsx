import React, { useState, useEffect, useRef, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from '../components/LoadingSpinner'

// ─── Types ────────────────────────────────────────────────────────────────────

interface GraphNode {
  id: string
  weight: number        // portfolio weight 0–1
  community: number     // community id
  x: number
  y: number
  vx: number
  vy: number
}

interface GraphEdge {
  source: string
  target: string
  correlation: number  // -1 to +1
  isMST: boolean
  leadLag?: number     // >0 means source leads target, <0 target leads
}

type Regime = 'NORMAL' | 'STRESS'

interface NetworkData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

// ─── Mock ─────────────────────────────────────────────────────────────────────

const SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX', 'ADA', 'MATIC', 'DOT', 'LINK', 'UNI']
const WEIGHTS = [0.30, 0.20, 0.12, 0.08, 0.07, 0.05, 0.05, 0.05, 0.04, 0.04]
const COMMUNITIES = [0, 0, 1, 0, 1, 1, 1, 2, 2, 2]

function buildMockData(regime: Regime): NetworkData {
  const nodes: GraphNode[] = SYMBOLS.map((id, i) => ({
    id,
    weight: WEIGHTS[i],
    community: COMMUNITIES[i],
    x: 250 + Math.cos((i / SYMBOLS.length) * Math.PI * 2) * 160,
    y: 200 + Math.sin((i / SYMBOLS.length) * Math.PI * 2) * 140,
    vx: 0, vy: 0,
  }))

  const edges: GraphEdge[] = []
  const normalCorrs: Record<string, number> = {
    'BTC-ETH': 0.82, 'BTC-SOL': 0.68, 'BTC-BNB': 0.74, 'BTC-AVAX': 0.62,
    'ETH-SOL': 0.76, 'ETH-LINK': 0.65, 'ETH-UNI': 0.71, 'SOL-AVAX': 0.55,
    'BNB-ADA': 0.48, 'MATIC-DOT': 0.52, 'DOT-LINK': 0.60, 'ADA-MATIC': 0.45,
    'BTC-LINK': 0.55, 'ETH-MATIC': 0.68, 'BTC-ADA': 0.42, 'SOL-BNB': 0.51,
  }
  const stressCorrs: Record<string, number> = {}
  Object.entries(normalCorrs).forEach(([k, v]) => {
    stressCorrs[k] = Math.min(0.98, v + 0.15 + Math.random() * 0.1)
  })
  const corrs = regime === 'STRESS' ? stressCorrs : normalCorrs

  // MST: greedy by highest correlation
  const mstEdges = new Set<string>()
  const mstNodes = new Set<string>(['BTC'])
  const sortedEdges = Object.entries(corrs).sort((a, b) => b[1] - a[1])
  while (mstNodes.size < SYMBOLS.length) {
    for (const [key] of sortedEdges) {
      const [s, t] = key.split('-')
      if ((mstNodes.has(s) && !mstNodes.has(t)) || (!mstNodes.has(s) && mstNodes.has(t))) {
        mstEdges.add(key)
        mstNodes.add(s)
        mstNodes.add(t)
        break
      }
    }
    break  // avoid infinite loop with incomplete data
  }

  Object.entries(corrs).forEach(([key, correlation]) => {
    const [source, target] = key.split('-')
    edges.push({
      source, target, correlation,
      isMST: mstEdges.has(key),
      leadLag: Math.random() > 0.5 ? (Math.random() - 0.5) * 2 : undefined,
    })
  })

  return { nodes, edges }
}

// ─── Physics Simulation ───────────────────────────────────────────────────────

function runPhysics(nodes: GraphNode[], edges: GraphEdge[], width: number, height: number): GraphNode[] {
  const next = nodes.map(n => ({ ...n }))
  const nodeMap = new Map(next.map(n => [n.id, n]))

  const K_SPRING = 0.004
  const K_REPEL = 800
  const DAMPING = 0.88
  const REST_LEN = 90

  // Spring forces along edges
  edges.forEach(e => {
    const s = nodeMap.get(e.source)
    const t = nodeMap.get(e.target)
    if (!s || !t) return
    const dx = t.x - s.x
    const dy = t.y - s.y
    const dist = Math.sqrt(dx * dx + dy * dy) || 1
    const strength = Math.abs(e.correlation)
    const force = K_SPRING * strength * (dist - REST_LEN)
    const fx = (dx / dist) * force
    const fy = (dy / dist) * force
    s.vx += fx; s.vy += fy
    t.vx -= fx; t.vy -= fy
  })

  // Repulsion between all node pairs
  for (let i = 0; i < next.length; i++) {
    for (let j = i + 1; j < next.length; j++) {
      const a = next[i]; const b = next[j]
      const dx = b.x - a.x
      const dy = b.y - a.y
      const dist2 = dx * dx + dy * dy + 1
      const force = K_REPEL / dist2
      const dist = Math.sqrt(dist2)
      a.vx -= (dx / dist) * force
      a.vy -= (dy / dist) * force
      b.vx += (dx / dist) * force
      b.vy += (dy / dist) * force
    }
  }

  // Center attraction
  next.forEach(n => {
    n.vx += (width / 2 - n.x) * 0.002
    n.vy += (height / 2 - n.y) * 0.002
  })

  // Integrate
  next.forEach(n => {
    n.vx *= DAMPING
    n.vy *= DAMPING
    n.x += n.vx
    n.y += n.vy
    n.x = Math.max(24, Math.min(width - 24, n.x))
    n.y = Math.max(24, Math.min(height - 24, n.y))
  })

  return next
}

// ─── Community colors ─────────────────────────────────────────────────────────

const COMMUNITY_COLORS = ['var(--accent)', '#22c55e', '#a78bfa', '#f97316', '#ec4899']

// ─── Page ─────────────────────────────────────────────────────────────────────

const NetworkGraph: React.FC = () => {
  const [regime, setRegime] = useState<Regime>('NORMAL')
  const [showMSTOnly, setShowMSTOnly] = useState(false)
  const [showLeadLag, setShowLeadLag] = useState(true)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [graphData, setGraphData] = useState<NetworkData>(() => buildMockData('NORMAL'))
  const animRef = useRef<number>()
  const stepRef = useRef(0)
  const W = 500, H = 380

  // Re-seed on regime change
  useEffect(() => {
    setGraphData(buildMockData(regime))
    stepRef.current = 0
  }, [regime])

  // Physics animation
  useEffect(() => {
    const tick = () => {
      if (stepRef.current < 120) {
        stepRef.current++
        setGraphData(prev => ({
          ...prev,
          nodes: runPhysics(prev.nodes, prev.edges, W, H),
        }))
      }
      animRef.current = requestAnimationFrame(tick)
    }
    animRef.current = requestAnimationFrame(tick)
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current) }
  }, [regime])

  const { nodes, edges } = graphData
  const nodeMap = new Map(nodes.map(n => [n.id, n]))

  const visibleEdges = showMSTOnly ? edges.filter(e => e.isMST) : edges.filter(e => Math.abs(e.correlation) > 0.4)

  const minWeight = Math.min(...nodes.map(n => n.weight))
  const maxWeight = Math.max(...nodes.map(n => n.weight))
  const nodeRadius = (w: number) => 6 + ((w - minWeight) / (maxWeight - minWeight + 0.001)) * 14

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 6 }}>
          {(['NORMAL', 'STRESS'] as Regime[]).map(r => (
            <button
              key={r}
              onClick={() => setRegime(r)}
              style={{
                padding: '5px 14px', borderRadius: 6, border: '1px solid var(--border)',
                background: regime === r ? 'var(--accent)' : 'var(--bg-hover)',
                color: regime === r ? '#000' : 'var(--text-muted)',
                fontSize: '0.8rem', fontWeight: regime === r ? 700 : 400, cursor: 'pointer',
              }}
            >
              {r} Regime
            </button>
          ))}
        </div>
        <button
          onClick={() => setShowMSTOnly(v => !v)}
          style={{
            padding: '5px 14px', borderRadius: 6, border: '1px solid var(--border)',
            background: showMSTOnly ? 'var(--blue)' : 'var(--bg-hover)',
            color: showMSTOnly ? '#fff' : 'var(--text-muted)',
            fontSize: '0.8rem', cursor: 'pointer',
          }}
        >
          MST Only
        </button>
        <button
          onClick={() => setShowLeadLag(v => !v)}
          style={{
            padding: '5px 14px', borderRadius: 6, border: '1px solid var(--border)',
            background: showLeadLag ? 'var(--yellow)' : 'var(--bg-hover)',
            color: showLeadLag ? '#000' : 'var(--text-muted)',
            fontSize: '0.8rem', cursor: 'pointer',
          }}
        >
          Lead-Lag Arrows
        </button>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 12, fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          <span>Node size = portfolio weight</span>
          <span>Edge opacity = |correlation|</span>
          <span style={{ color: 'var(--green)' }}>Green = positive corr</span>
          <span style={{ color: 'var(--red)' }}>Red = negative</span>
        </div>
      </div>

      {/* SVG Graph */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, overflow: 'hidden',
      }}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          style={{ width: '100%', height: 380 }}
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <marker id="arrow-pos" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="var(--green)" opacity={0.7} />
            </marker>
            <marker id="arrow-neg" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="var(--yellow)" opacity={0.7} />
            </marker>
          </defs>

          {/* Edges */}
          {visibleEdges.map(e => {
            const s = nodeMap.get(e.source)
            const t = nodeMap.get(e.target)
            if (!s || !t) return null
            const opacity = Math.abs(e.correlation) * 0.8
            const color = e.correlation >= 0 ? '#22c55e' : '#ef4444'
            const isHovered = hoveredNode === e.source || hoveredNode === e.target
            const hasLeadLag = showLeadLag && e.leadLag != null && Math.abs(e.leadLag!) > 0.5

            return (
              <line
                key={`${e.source}-${e.target}`}
                x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                stroke={color}
                strokeWidth={e.isMST ? 1.8 : 0.8}
                opacity={isHovered ? 1 : opacity}
                strokeDasharray={e.isMST ? undefined : '3,2'}
                markerEnd={hasLeadLag && e.leadLag! > 0 ? 'url(#arrow-pos)' : hasLeadLag ? 'url(#arrow-neg)' : undefined}
              />
            )
          })}

          {/* Nodes */}
          {nodes.map(n => {
            const r = nodeRadius(n.weight)
            const color = COMMUNITY_COLORS[n.community % COMMUNITY_COLORS.length]
            const isHovered = hoveredNode === n.id
            return (
              <g
                key={n.id}
                onMouseEnter={() => setHoveredNode(n.id)}
                onMouseLeave={() => setHoveredNode(null)}
                style={{ cursor: 'pointer' }}
              >
                <circle
                  cx={n.x} cy={n.y} r={r + (isHovered ? 3 : 0)}
                  fill={color}
                  opacity={isHovered ? 1 : 0.75}
                  stroke={isHovered ? '#fff' : 'var(--bg-primary)'}
                  strokeWidth={isHovered ? 1.5 : 0.8}
                />
                <text
                  x={n.x} y={n.y + r + 9}
                  textAnchor="middle"
                  fontSize={8}
                  fill="var(--text-secondary)"
                  fontWeight={isHovered ? 700 : 400}
                >
                  {n.id}
                </text>
                {isHovered && (
                  <text x={n.x} y={n.y + 3} textAnchor="middle" fontSize={6} fill="#000" fontWeight={700}>
                    {(n.weight * 100).toFixed(0)}%
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      </div>

      {/* Community Legend */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        {[...new Set(nodes.map(n => n.community))].sort().map(c => {
          const members = nodes.filter(n => n.community === c).map(n => n.id)
          return (
            <div key={c} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem' }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: COMMUNITY_COLORS[c % COMMUNITY_COLORS.length] }} />
              <span style={{ color: 'var(--text-muted)' }}>Cluster {c + 1}:</span>
              <span style={{ color: 'var(--text-secondary)' }}>{members.join(', ')}</span>
            </div>
          )
        })}
      </div>

      {/* Correlation table for hovered node */}
      {hoveredNode && (
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '14px 16px',
        }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, marginBottom: 10 }}>
            {hoveredNode} CORRELATIONS
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {edges
              .filter(e => e.source === hoveredNode || e.target === hoveredNode)
              .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
              .map(e => {
                const peer = e.source === hoveredNode ? e.target : e.source
                const color = e.correlation >= 0 ? 'var(--green)' : 'var(--red)'
                return (
                  <div key={peer} style={{
                    padding: '4px 10px', borderRadius: 5,
                    background: 'var(--bg-hover)', fontSize: '0.75rem',
                    display: 'flex', gap: 6, alignItems: 'center',
                  }}>
                    <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{peer}</span>
                    <span style={{ color, fontWeight: 700 }}>{e.correlation.toFixed(2)}</span>
                    {e.isMST && <span style={{ fontSize: '0.62rem', color: 'var(--accent)' }}>MST</span>}
                  </div>
                )
              })}
          </div>
        </div>
      )}
    </div>
  )
}

export default NetworkGraph
