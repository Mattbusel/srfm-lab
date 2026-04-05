import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react'
import * as d3 from 'd3'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  Cell,
  ResponsiveContainer,
} from 'recharts'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import {
  fetchGenealogyGraph,
  fetchEvolutionStats,
  fetchMutationFrequencies,
} from '../api/client'
import type { GenealogyNode, Island } from '../types'

const ISLAND_COLORS: Record<Island, string> = {
  BULL:    '#22c55e',
  BEAR:    '#ef4444',
  NEUTRAL: '#3b82f6',
}

// ─── D3 Force Graph ───────────────────────────────────────────────────────────

interface D3Node extends d3.SimulationNodeDatum {
  id: number
  island: Island
  generation: number
  fitness: number
  isHallOfFame: boolean
  sharpe: number
}

interface D3Link extends d3.SimulationLinkDatum<D3Node> {
  source: number | D3Node
  target: number | D3Node
}

interface D3GenealogyTreeProps {
  onNodeClick: (node: GenealogyNode) => void
  selectedId: number | null
}

const D3GenealogyTree: React.FC<D3GenealogyTreeProps> = ({
  onNodeClick,
  selectedId,
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const { data: graph, isLoading } = useQuery({
    queryKey: ['genealogy'],
    queryFn: fetchGenealogyGraph,
    staleTime: 60_000,
  })

  const nodeMap = useMemo(() => {
    if (!graph) return new Map<number, GenealogyNode>()
    return new Map(graph.nodes.map((n) => [n.genomeId, n]))
  }, [graph])

  useEffect(() => {
    if (!graph || !svgRef.current) return

    const container = svgRef.current.parentElement
    const width = container?.clientWidth ?? 600
    const height = 420

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    svg.attr('width', width).attr('height', height)

    // Background
    svg
      .append('rect')
      .attr('width', width)
      .attr('height', height)
      .attr('fill', 'var(--bg-elevated)')
      .attr('rx', 8)

    const g = svg.append('g')

    // Zoom
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform.toString())
      })
    svg.call(zoom)

    // Build nodes and links
    const nodes: D3Node[] = graph.nodes.map((n) => ({
      id: n.genomeId,
      island: n.island,
      generation: n.generation,
      fitness: n.fitness,
      isHallOfFame: n.isHallOfFame,
      sharpe: n.sharpe ?? 1,
    }))

    const links: D3Link[] = graph.edges.map((e) => ({
      source: e.source,
      target: e.target,
    }))

    // Simulation
    const sim = d3
      .forceSimulation<D3Node>(nodes)
      .force(
        'link',
        d3
          .forceLink<D3Node, D3Link>(links)
          .id((d) => d.id)
          .distance(60)
          .strength(0.8)
      )
      .force('charge', d3.forceManyBody().strength(-200))
      .force('x', d3.forceX<D3Node>((d) => {
        const islandX: Record<Island, number> = { BULL: width * 0.25, BEAR: width * 0.5, NEUTRAL: width * 0.75 }
        return islandX[d.island]
      }).strength(0.4))
      .force('y', d3.forceY<D3Node>((d) => {
        return 60 + (d.generation / 5) * (height - 120)
      }).strength(0.6))
      .force('collision', d3.forceCollide<D3Node>(18))

    // Links
    const link = g
      .append('g')
      .selectAll<SVGLineElement, D3Link>('line')
      .data(links)
      .join('line')
      .attr('stroke', 'var(--border-emphasis)')
      .attr('stroke-width', 1)
      .attr('stroke-opacity', 0.5)

    // Nodes
    const nodeGroup = g
      .append('g')
      .selectAll<SVGGElement, D3Node>('g')
      .data(nodes)
      .join('g')
      .style('cursor', 'pointer')
      .on('click', (_event, d) => {
        const original = nodeMap.get(d.id)
        if (original) onNodeClick(original)
      })

    // Node circle
    nodeGroup
      .append('circle')
      .attr('r', (d) => Math.max(6, Math.min(16, d.sharpe * 4)))
      .attr('fill', (d) => {
        if (d.isHallOfFame) return '#fbbf24'
        const low = d3.color('#ef4444')!
        const high = d3.color('#22c55e')!
        return d3.interpolateRgb(low.toString(), high.toString())(d.fitness)
      })
      .attr('fill-opacity', 0.85)
      .attr('stroke', (d) =>
        d.isHallOfFame ? '#fbbf24' : ISLAND_COLORS[d.island]
      )
      .attr('stroke-width', (d) => (d.isHallOfFame ? 2.5 : 1.5))

    // HOF glow
    nodeGroup
      .filter((d) => d.isHallOfFame)
      .append('circle')
      .attr('r', (d) => Math.max(8, Math.min(20, d.sharpe * 5)))
      .attr('fill', 'none')
      .attr('stroke', '#fbbf24')
      .attr('stroke-width', 1)
      .attr('stroke-opacity', 0.4)

    // Label
    nodeGroup
      .append('text')
      .text((d) => `#${d.id}`)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', '7px')
      .attr('font-family', 'var(--font-mono)')
      .attr('fill', '#0d0d0d')
      .attr('pointer-events', 'none')

    // Selected highlight
    nodeGroup.selectAll('circle').attr('stroke-width', (d: unknown) => {
      const nd = d as D3Node
      return nd.id === selectedId ? 3 : nd.isHallOfFame ? 2.5 : 1.5
    })

    // Tick
    sim.on('tick', () => {
      link
        .attr('x1', (d) => (d.source as D3Node).x ?? 0)
        .attr('y1', (d) => (d.source as D3Node).y ?? 0)
        .attr('x2', (d) => (d.target as D3Node).x ?? 0)
        .attr('y2', (d) => (d.target as D3Node).y ?? 0)

      nodeGroup.attr(
        'transform',
        (d) => `translate(${d.x ?? 0}, ${d.y ?? 0})`
      )
    })

    return () => {
      sim.stop()
    }
  }, [graph, selectedId, nodeMap, onNodeClick])

  if (isLoading) return <LoadingSpinner fullPage label="Building genealogy graph…" />

  return <svg ref={svgRef} style={{ width: '100%', display: 'block' }} />
}

// ─── Diversity Chart ──────────────────────────────────────────────────────────

const DiversityChart: React.FC = () => {
  const { data: stats } = useQuery({
    queryKey: ['evolution', 'stats'],
    queryFn: fetchEvolutionStats,
    refetchInterval: 30_000,
  })

  const data = useMemo(() => {
    if (!stats) return []
    const bull = stats.filter((s) => s.island === 'BULL').sort((a, b) => a.generation - b.generation)
    return bull.map((s) => ({
      generation: s.generation,
      diversity: parseFloat(s.diversityIndex.toFixed(4)),
    }))
  }, [stats])

  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
        <XAxis dataKey="generation" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
        <YAxis domain={[0, 1]} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-emphasis)',
            borderRadius: 6,
            fontSize: 12,
          }}
        />
        <Line
          type="monotone"
          dataKey="diversity"
          stroke="var(--purple)"
          strokeWidth={2}
          dot={false}
          name="Diversity Index"
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ─── Mutation Frequency ───────────────────────────────────────────────────────

const MutationFrequencyChart: React.FC = () => {
  const { data: mutations = [] } = useQuery({
    queryKey: ['evolution', 'mutations'],
    queryFn: fetchMutationFrequencies,
    refetchInterval: 60_000,
  })

  const data = [...mutations].sort((a, b) => b.count - a.count)

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 4, bottom: 0, left: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
        <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
        <YAxis
          type="category"
          dataKey="mutation"
          tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
          width={60}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-emphasis)',
            borderRadius: 6,
            fontSize: 12,
          }}
        />
        <Bar dataKey="count" radius={[0, 3, 3, 0]}>
          {data.map((entry, i) => (
            <Cell
              key={`cell-${i}`}
              fill={
                entry.avgFitnessImprovement > 0.03
                  ? '#22c55e'
                  : entry.avgFitnessImprovement > 0
                  ? '#00d4aa'
                  : '#ef4444'
              }
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ─── Generation Heatmap ───────────────────────────────────────────────────────

const GenerationHeatmap: React.FC = () => {
  const { data: stats } = useQuery({
    queryKey: ['evolution', 'stats'],
    queryFn: fetchEvolutionStats,
    refetchInterval: 30_000,
  })

  const heatmapData = useMemo(() => {
    if (!stats) return { rows: [], islands: [] as Island[], generations: [] as number[] }
    const islands: Island[] = ['BULL', 'BEAR', 'NEUTRAL']
    const generations = [...new Set(stats.map((s) => s.generation))].sort((a, b) => a - b)
    const rows = generations.slice(-10).map((gen) => {
      const row: Record<string, number | string> = { gen }
      for (const island of islands) {
        const s = stats.find((st) => st.generation === gen && st.island === island)
        row[island] = s ? s.bestFitness : 0
      }
      return row
    })
    return { rows, islands, generations: generations.slice(-10) }
  }, [stats])

  const maxFitness = 1
  const colorScale = (val: number) => {
    const t = val / maxFitness
    return `rgba(0, 212, 170, ${0.1 + t * 0.8})`
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 2, marginBottom: 4 }}>
        {(['BULL', 'BEAR', 'NEUTRAL'] as Island[]).map((island) => (
          <div key={island} style={{ flex: 1, textAlign: 'center', fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            <StatusBadge value={island} size="sm" />
          </div>
        ))}
      </div>
      {heatmapData.rows.map((row) => (
        <div key={row.gen} style={{ display: 'flex', gap: 2, marginBottom: 2, alignItems: 'center' }}>
          <div style={{ width: 28, fontSize: '0.65rem', color: 'var(--text-muted)', textAlign: 'right', paddingRight: 4, flexShrink: 0 }}>
            G{row.gen}
          </div>
          {(['BULL', 'BEAR', 'NEUTRAL'] as Island[]).map((island) => {
            const val = row[island] as number
            return (
              <div
                key={island}
                title={`${island} Gen ${row.gen}: ${val.toFixed(4)}`}
                style={{
                  flex: 1,
                  height: 18,
                  background: colorScale(val),
                  borderRadius: 2,
                  border: '1px solid var(--border-subtle)',
                }}
              />
            )
          })}
        </div>
      ))}
      <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 6, textAlign: 'right' }}>
        darker = higher fitness
      </div>
    </div>
  )
}

// ─── Genealogy Page ───────────────────────────────────────────────────────────

const GenealogyPage: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<GenealogyNode | null>(null)

  const handleNodeClick = useCallback((node: GenealogyNode) => {
    setSelectedNode((prev) => (prev?.genomeId === node.genomeId ? null : node))
  }, [])

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-title">Genealogy</div>
          <div className="page-subtitle">
            Evolution lineage, diversity, and mutation analysis
          </div>
        </div>
        <div style={{ display: 'flex', gap: 12, fontSize: '0.75rem', color: 'var(--text-muted)', alignItems: 'center' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#fbbf24', display: 'inline-block' }} />
            Hall of Fame
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#22c55e', display: 'inline-block' }} />
            High fitness
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#ef4444', display: 'inline-block' }} />
            Low fitness
          </span>
          <span style={{ color: 'var(--text-muted)' }}>Size = Sharpe · Scroll to zoom</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 16, marginBottom: 16 }}>
        {/* Force Graph */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div className="card-header" style={{ padding: '12px 16px' }}>
            <span className="card-title">Lineage Graph — All Islands</span>
          </div>
          <D3GenealogyTree
            onNodeClick={handleNodeClick}
            selectedId={selectedNode?.genomeId ?? null}
          />
        </div>

        {/* Node Detail Panel */}
        <div className="card">
          <div className="card-title" style={{ marginBottom: 12 }}>
            {selectedNode ? `Genome #${selectedNode.genomeId}` : 'Select a Node'}
          </div>
          {selectedNode ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', gap: 6 }}>
                <StatusBadge value={selectedNode.island} />
                {selectedNode.isHallOfFame && <StatusBadge value="hof" />}
              </div>
              {[
                { label: 'Fitness', value: selectedNode.fitness.toFixed(4), color: 'var(--accent)' },
                { label: 'Sharpe', value: (selectedNode.sharpe ?? 0).toFixed(3), color: 'var(--green)' },
                { label: 'Generation', value: selectedNode.generation, color: 'var(--text-primary)' },
              ].map(({ label, value, color }) => (
                <div
                  key={label}
                  style={{
                    background: 'var(--bg-elevated)',
                    borderRadius: 'var(--radius)',
                    padding: '8px 10px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <span style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>{label}</span>
                  <span className="num" style={{ fontSize: '0.875rem', fontWeight: 700, color }}>
                    {value}
                  </span>
                </div>
              ))}
              {selectedNode.parentIds && selectedNode.parentIds.length > 0 && (
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 4 }}>
                    Parent IDs
                  </div>
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                    {selectedNode.parentIds.map((pid) => (
                      <span key={pid} className="num" style={{ color: 'var(--accent)', fontSize: '0.8125rem' }}>
                        #{pid}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8125rem', textAlign: 'center', padding: '24px 0' }}>
              Click a node in the graph to view details
            </div>
          )}
        </div>
      </div>

      {/* Bottom Row: Diversity + Mutations + Heatmap */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Diversity Index (BULL)</span>
          </div>
          <DiversityChart />
        </div>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Mutation Frequency</span>
          </div>
          <MutationFrequencyChart />
        </div>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Generation × Island Fitness</span>
          </div>
          <GenerationHeatmap />
        </div>
      </div>
    </div>
  )
}

export default GenealogyPage
