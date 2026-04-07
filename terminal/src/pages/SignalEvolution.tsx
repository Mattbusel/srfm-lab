// ============================================================
// SignalEvolution.tsx -- Signal genome evolution tracker
// Parent-child lineage, equity curves, gene contributions
// ============================================================
import React, { useState, useMemo, useCallback, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, BarChart, Bar, Cell,
  ScatterChart, Scatter, ZAxis,
} from 'recharts'
import { clsx } from 'clsx'

// ---- Types ----

interface GenomeNode {
  id: string
  parentId: string | null
  fitness: number
  generation: number
  mutationType: 'mutation' | 'crossover' | 'seed'
  genes: number[] // 31 gene values [0,1]
  isActive: boolean
  novelty: number
}

interface EquityCurvePoint {
  bar: number
  [genomeId: string]: number
}

interface ICPoint {
  horizon: number // 1 to 20 bars
  ic: number
}

// ---- Gene metadata: 31 genes across BH/Nav/Hurst/GARCH/Risk ----

const GENE_META: { name: string; type: 'BH' | 'Nav' | 'Hurst' | 'GARCH' | 'Risk' }[] = [
  { name: 'bh_mass_w', type: 'BH' },
  { name: 'bh_radius_w', type: 'BH' },
  { name: 'bh_collapse_gate', type: 'BH' },
  { name: 'bh_formation_thresh', type: 'BH' },
  { name: 'bh_decay_half', type: 'BH' },
  { name: 'bh_min_bars', type: 'BH' },
  { name: 'bh_vol_scale', type: 'BH' },
  { name: 'nav_entry_gate', type: 'Nav' },
  { name: 'nav_exit_gate', type: 'Nav' },
  { name: 'nav_curvature_w', type: 'Nav' },
  { name: 'nav_geo_entry', type: 'Nav' },
  { name: 'nav_tidal_w', type: 'Nav' },
  { name: 'nav_horizon', type: 'Nav' },
  { name: 'hurst_window', type: 'Hurst' },
  { name: 'hurst_min_val', type: 'Hurst' },
  { name: 'hurst_trend_gate', type: 'Hurst' },
  { name: 'hurst_mr_gate', type: 'Hurst' },
  { name: 'hurst_weight', type: 'Hurst' },
  { name: 'garch_alpha', type: 'GARCH' },
  { name: 'garch_beta', type: 'GARCH' },
  { name: 'garch_omega', type: 'GARCH' },
  { name: 'garch_vol_target', type: 'GARCH' },
  { name: 'garch_vol_cap', type: 'GARCH' },
  { name: 'garch_forecast_h', type: 'GARCH' },
  { name: 'risk_drawdown_lim', type: 'Risk' },
  { name: 'risk_pos_size_k', type: 'Risk' },
  { name: 'risk_max_pos', type: 'Risk' },
  { name: 'risk_corr_cap', type: 'Risk' },
  { name: 'risk_stop_atr', type: 'Risk' },
  { name: 'risk_take_atr', type: 'Risk' },
  { name: 'risk_kelly_frac', type: 'Risk' },
]

const GENE_TYPE_COLORS: Record<string, string> = {
  BH: '#8b5cf6',
  Nav: '#3b82f6',
  Hurst: '#22c55e',
  GARCH: '#f59e0b',
  Risk: '#ef4444',
}

// ---- Demo data ----

function makeGenes(seed: number): number[] {
  return GENE_META.map((_, i) => Math.max(0, Math.min(1, 0.5 + Math.sin(seed + i * 1.37) * 0.45)))
}

function buildDemoData() {
  // Build a tree of ~20 genomes across 6 generations
  const nodes: GenomeNode[] = []

  // Seed
  nodes.push({
    id: 'G1', parentId: null, fitness: 0.42, generation: 1,
    mutationType: 'seed', genes: makeGenes(1), isActive: false, novelty: 0.8,
  })
  nodes.push({
    id: 'G2', parentId: null, fitness: 0.38, generation: 1,
    mutationType: 'seed', genes: makeGenes(2), isActive: false, novelty: 0.9,
  })

  const pairs = [
    ['G3', 'G1', 0.51, 2, 'mutation'], ['G4', 'G1', 0.48, 2, 'crossover'],
    ['G5', 'G2', 0.55, 2, 'mutation'], ['G6', 'G2', 0.44, 2, 'mutation'],
    ['G7', 'G3', 0.63, 3, 'mutation'], ['G8', 'G3', 0.59, 3, 'crossover'],
    ['G9', 'G5', 0.68, 3, 'mutation'], ['G10', 'G5', 0.61, 3, 'crossover'],
    ['G11', 'G7', 0.74, 4, 'mutation'], ['G12', 'G7', 0.70, 4, 'crossover'],
    ['G13', 'G9', 0.79, 4, 'mutation'], ['G14', 'G9', 0.75, 4, 'crossover'],
    ['G15', 'G11', 0.83, 5, 'mutation'], ['G16', 'G13', 0.87, 5, 'mutation'],
    ['G17', 'G13', 0.85, 5, 'crossover'], ['G18', 'G16', 0.91, 6, 'mutation'],
    ['G19', 'G16', 0.88, 6, 'crossover'], ['G20', 'G18', 0.94, 6, 'mutation'],
  ] as const

  pairs.forEach(([id, parentId, fitness, gen, mtype], i) => {
    nodes.push({
      id, parentId, fitness, generation: gen,
      mutationType: mtype as GenomeNode['mutationType'],
      genes: makeGenes(i + 3),
      isActive: id === 'G20',
      novelty: Math.max(0.05, 0.9 - fitness * 0.7 + Math.abs(Math.sin(i * 1.1)) * 0.3),
    })
  })

  // Top 5 equity curves
  const TOP5 = ['G20', 'G18', 'G19', 'G16', 'G17']
  const eqCurves: EquityCurvePoint[] = []
  for (let b = 0; b <= 100; b++) {
    const pt: EquityCurvePoint = { bar: b }
    TOP5.forEach((id, rank) => {
      const drift = (0.94 - rank * 0.04) * 0.008
      const prev = b === 0 ? 1.0 : (eqCurves[b - 1][id] as number)
      pt[id] = prev * (1 + drift + (Math.random() - 0.49) * 0.015)
    })
    eqCurves.push(pt)
  }

  // IC decay curve
  const icDecay: ICPoint[] = Array.from({ length: 20 }, (_, i) => ({
    horizon: i + 1,
    ic: Math.max(-0.05, 0.18 * Math.exp(-i * 0.15) + Math.sin(i * 0.8) * 0.02),
  }))

  // Gene contributions for best genome G20
  const geneContribs = GENE_META.map((g, i) => ({
    name: g.name,
    type: g.type,
    contribution: Math.max(-0.05, 0.25 * Math.abs(Math.sin(20 + i * 1.37)) - 0.02),
  })).sort((a, b) => b.contribution - a.contribution)

  // Novelty scatter
  const noveltyScatter = nodes.map((n) => ({
    id: n.id,
    fitness: n.fitness,
    novelty: n.novelty,
    generation: n.generation,
  }))

  return { nodes, eqCurves, icDecay, geneContribs, noveltyScatter, TOP5 }
}

// ---- Tree layout helpers ----

function layoutTree(nodes: GenomeNode[]): Map<string, { x: number; y: number }> {
  // Group by generation
  const byGen = new Map<number, GenomeNode[]>()
  nodes.forEach((n) => {
    if (!byGen.has(n.generation)) byGen.set(n.generation, [])
    byGen.get(n.generation)!.push(n)
  })

  const positions = new Map<string, { x: number; y: number }>()
  const maxGen = Math.max(...nodes.map((n) => n.generation))
  const W = 560; const H = 260

  byGen.forEach((gens, gen) => {
    gens.forEach((node, idx) => {
      positions.set(node.id, {
        x: (gen / maxGen) * (W - 40) + 20,
        y: ((idx + 1) / (gens.length + 1)) * H,
      })
    })
  })
  return positions
}

// ---- Sub-components ----

const SignalTree: React.FC<{
  nodes: GenomeNode[]
  selectedId: string | null
  onSelect: (id: string) => void
}> = ({ nodes, selectedId, onSelect }) => {
  const positions = useMemo(() => layoutTree(nodes), [nodes])
  const W = 560; const H = 260

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Signal Lineage Tree -- click node to inspect
      </h3>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
        {/* Edges */}
        {nodes.filter((n) => n.parentId).map((n) => {
          const from = positions.get(n.parentId!)
          const to = positions.get(n.id)
          if (!from || !to) return null
          const stroke = n.mutationType === 'crossover' ? '#3b82f6' : '#6b7280'
          return (
            <line
              key={`e-${n.id}`}
              x1={from.x} y1={from.y} x2={to.x} y2={to.y}
              stroke={stroke} strokeWidth={n.mutationType === 'crossover' ? 2 : 1}
              strokeDasharray={n.mutationType === 'crossover' ? '4,2' : undefined}
              opacity={0.6}
            />
          )
        })}
        {/* Nodes */}
        {nodes.map((n) => {
          const pos = positions.get(n.id)
          if (!pos) return null
          const r = n.isActive ? 10 : 7
          const fill = fitnessNodeColor(n.fitness)
          const isSelected = n.id === selectedId
          return (
            <g key={n.id} onClick={() => onSelect(n.id)} style={{ cursor: 'pointer' }}>
              <circle
                cx={pos.x} cy={pos.y} r={r}
                fill={fill}
                stroke={isSelected ? '#ffffff' : n.isActive ? '#22c55e' : '#374151'}
                strokeWidth={isSelected ? 2.5 : n.isActive ? 2 : 1}
              />
              <text
                x={pos.x} y={pos.y + 3}
                textAnchor="middle" fill="white" fontSize={7} fontWeight="bold"
              >
                {n.id}
              </text>
            </g>
          )
        })}
        {/* Gen labels at top */}
        {[1,2,3,4,5,6].map((gen) => {
          const x = (gen / 6) * (W - 40) + 20
          return (
            <text key={gen} x={x} y={12} textAnchor="middle" fill="#6b7280" fontSize={9}>
              Gen {gen}
            </text>
          )
        })}
      </svg>
      <div className="flex gap-4 mt-1 text-xs text-gray-500">
        <span><span className="inline-block w-4 h-px bg-gray-500 mr-1 align-middle" />Mutation</span>
        <span><span className="inline-block w-4 h-px bg-blue-500 mr-1 align-middle border-dashed" style={{ borderBottom: '2px dashed #3b82f6' }} />Crossover</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-green-400 mr-1 align-middle" />Active</span>
      </div>
    </div>
  )
}

function fitnessNodeColor(f: number): string {
  const r = Math.round(200 * (1 - f))
  const g = Math.round(200 * f)
  return `rgb(${r},${g},60)`
}

const EQ_COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#a78bfa', '#f472b6']

const SignalBacktestComparison: React.FC<{
  data: EquityCurvePoint[]
  top5: readonly string[]
}> = ({ data, top5 }) => (
  <div className="bg-gray-900 border border-gray-700 rounded p-3">
    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
      Top 5 Genome Equity Curves
    </h3>
    <ResponsiveContainer width="100%" height={170}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="bar" tick={{ fill: '#9ca3af', fontSize: 10 }} label={{ value: 'Bar', fill: '#6b7280', fontSize: 9, position: 'insideRight' }} />
        <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)} />
        <Tooltip
          contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
          formatter={(v: number) => v.toFixed(4)}
        />
        <Legend wrapperStyle={{ fontSize: 10 }} />
        {top5.map((id, i) => (
          <Line key={id} type="monotone" dataKey={id} stroke={EQ_COLORS[i]} dot={false} strokeWidth={1.5} name={id} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  </div>
)

const GeneContributionBar: React.FC<{ data: { name: string; type: string; contribution: number }[] }> = ({ data }) => {
  const top15 = data.slice(0, 15)
  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Gene Fitness Contribution (Top 15)
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={top15} layout="vertical" margin={{ top: 0, right: 30, bottom: 0, left: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
          <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 10 }} domain={[0, 'dataMax']} />
          <YAxis type="category" dataKey="name" tick={{ fill: '#9ca3af', fontSize: 9 }} width={80} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
            formatter={(v: number) => v.toFixed(4)}
          />
          <Bar dataKey="contribution" radius={[0, 2, 2, 0]}>
            {top15.map((entry, i) => (
              <Cell key={i} fill={GENE_TYPE_COLORS[entry.type] ?? '#6b7280'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-2 mt-1">
        {Object.entries(GENE_TYPE_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-sm" style={{ background: color }} />
            <span className="text-xs text-gray-500">{type}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

const NoveltyScatterPlot: React.FC<{
  data: { id: string; fitness: number; novelty: number; generation: number }[]
}> = ({ data }) => {
  const maxGen = Math.max(...data.map((d) => d.generation))
  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Novelty vs Fitness (color = generation)
      </h3>
      <ResponsiveContainer width="100%" height={180}>
        <ScatterChart margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="fitness" type="number" domain={[0.3, 1.0]} name="Fitness"
            tick={{ fill: '#9ca3af', fontSize: 10 }}
            label={{ value: 'Fitness', fill: '#6b7280', fontSize: 9, position: 'insideBottom', offset: -10 }}
          />
          <YAxis
            dataKey="novelty" type="number" domain={[0, 1]} name="Novelty"
            tick={{ fill: '#9ca3af', fontSize: 10 }}
          />
          <ZAxis range={[40, 80]} />
          <Tooltip
            cursor={{ stroke: '#374151' }}
            contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
            content={({ payload }) => {
              if (!payload?.length) return null
              const d = payload[0].payload
              return (
                <div className="bg-gray-900 border border-gray-700 p-2 text-xs">
                  <div className="text-white font-bold">{d.id}</div>
                  <div>Fitness: {d.fitness.toFixed(3)}</div>
                  <div>Novelty: {d.novelty.toFixed(3)}</div>
                  <div>Gen: {d.generation}</div>
                </div>
              )
            }}
          />
          <Scatter data={data} name="Genomes">
            {data.map((d, i) => {
              const t = (d.generation - 1) / (maxGen - 1)
              const r = Math.round(100 + t * 100)
              const b = Math.round(200 - t * 150)
              return <Cell key={i} fill={`rgb(${r},80,${b})`} />
            })}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

const SignalDecayMonitor: React.FC<{ data: ICPoint[] }> = ({ data }) => (
  <div className="bg-gray-900 border border-gray-700 rounded p-3">
    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
      IC Decay -- Best Signal (1-20 bar horizons)
    </h3>
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="horizon" tick={{ fill: '#9ca3af', fontSize: 10 }} label={{ value: 'Bars', fill: '#6b7280', fontSize: 9, position: 'insideRight' }} />
        <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)} />
        <Tooltip
          contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
          formatter={(v: number) => v.toFixed(4)}
        />
        <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="4 2" strokeWidth={1} />
        <ReferenceLine y={0.05} stroke="#22c55e" strokeDasharray="4 2" strokeWidth={1} label={{ value: 'IC=0.05', fill: '#22c55e', fontSize: 9 }} />
        <Line type="monotone" dataKey="ic" stroke="#3b82f6" dot={{ r: 3, fill: '#3b82f6' }} strokeWidth={2} name="IC" />
      </LineChart>
    </ResponsiveContainer>
  </div>
)

const ActiveSignalCard: React.FC<{ genome: GenomeNode | undefined }> = ({ genome }) => {
  if (!genome) return null
  const typeGroups = ['BH', 'Nav', 'Hurst', 'GARCH', 'Risk'] as const
  return (
    <div className="bg-gray-900 border border-green-700 rounded p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold text-green-400 uppercase tracking-wider">
          Active Signal -- {genome.id}
        </h3>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-xs text-green-400">LIVE</span>
          <span className="text-xs text-gray-500">Gen {genome.generation}</span>
          <span className="text-xs text-white font-mono">f={genome.fitness.toFixed(4)}</span>
        </div>
      </div>
      <div className="grid grid-cols-5 gap-2">
        {typeGroups.map((type) => {
          const genes = GENE_META.map((g, i) => ({ ...g, value: genome.genes[i] }))
            .filter((g) => g.type === type)
          return (
            <div key={type} className="bg-gray-800 rounded p-2">
              <div className="text-xs font-semibold mb-1" style={{ color: GENE_TYPE_COLORS[type] }}>
                {type}
              </div>
              {genes.map((g) => (
                <div key={g.name} className="flex justify-between gap-1 text-xs">
                  <span className="text-gray-500 truncate" title={g.name}>{g.name.replace(/^(bh|nav|hurst|garch|risk)_/, '')}</span>
                  <span className="text-white font-mono">{g.value.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ---- Main Page ----

export const SignalEvolution: React.FC = () => {
  const [selectedId, setSelectedId] = useState<string | null>('G20')

  const { nodes, eqCurves, icDecay, geneContribs, noveltyScatter, TOP5 } = useMemo(
    () => buildDemoData(),
    []
  )

  const selectedGenome = useMemo(
    () => nodes.find((n) => n.id === selectedId),
    [nodes, selectedId]
  )

  return (
    <div className="min-h-screen bg-gray-950 text-white p-4">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-lg font-bold tracking-wide">Signal Genome Evolution</h1>
        <div className="flex items-center gap-4 text-xs text-gray-400">
          <span>{nodes.length} genomes tracked</span>
          <span className="text-green-400">{nodes.filter((n) => n.isActive).length} active</span>
          <span>31 genes -- BH/Nav/Hurst/GARCH/Risk</span>
        </div>
      </div>

      {/* Row 1: Tree + Equity curves */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <SignalTree nodes={nodes} selectedId={selectedId} onSelect={setSelectedId} />
        <SignalBacktestComparison data={eqCurves} top5={TOP5} />
      </div>

      {/* Row 2: Gene contributions + Novelty scatter + IC decay */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <GeneContributionBar data={geneContribs} />
        <NoveltyScatterPlot data={noveltyScatter} />
        <SignalDecayMonitor data={icDecay} />
      </div>

      {/* Row 3: Active signal card */}
      <ActiveSignalCard genome={selectedGenome} />
    </div>
  )
}

export default SignalEvolution
