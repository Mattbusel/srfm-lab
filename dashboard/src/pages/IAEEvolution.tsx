// ============================================================
// pages/IAEEvolution.tsx -- IAE (Iterative Adaptation Engine) evolution dashboard
// Shows generation progress, parameter drift, top genomes, pedigree, and rollbacks.
// ============================================================

import React, { useState, useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
  Sparklines,
  SparklinesCurve,
} from 'recharts'
import { clsx } from 'clsx'
import { Card, StatCard, LoadingSpinner } from '@/components/ui'
import {
  useIAEEvolution,
  useTopGenomes,
  useRollbackHistory,
  useGenomePedigree,
  useGenerationStats,
} from '@/hooks/useIAEEvolution'
import type { GenomeRecord, GenerationStats, ParameterState, RollbackEvent, PedigreeNode } from '@/types/iae'
import { PARAM_META } from '@/types/iae'

// ---------------------------------------------------------------------------
// Tiny helpers
// ---------------------------------------------------------------------------

function fmt2(n: number): string {
  return n.toFixed(2)
}

function fmt4(n: number): string {
  return n.toFixed(4)
}

function fmtPct(n: number): string {
  return `${(n * 100).toFixed(1)}%`
}

function fmtDelta(n: number, pct = false): string {
  const s = pct ? fmtPct(Math.abs(n)) : fmt4(Math.abs(n))
  return n >= 0 ? `+${s}` : `-${s}`
}

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`
  if (diff < 3600_000) return `${Math.floor(diff / 60_000)}m ago`
  return `${Math.floor(diff / 3600_000)}h ago`
}

const OPERATOR_COLORS: Record<string, string> = {
  elite: '#f59e0b',
  crossover: '#3b82f6',
  mutation: '#8b5cf6',
}

const FITNESS_GRADIENT = (v: number): string => {
  // 0..1 -> red -> yellow -> green
  if (v >= 0.8) return '#22c55e'
  if (v >= 0.65) return '#84cc16'
  if (v >= 0.5) return '#f59e0b'
  return '#ef4444'
}

// ---------------------------------------------------------------------------
// SparkLine component (simple inline chart using svg)
// ---------------------------------------------------------------------------

const SparkLine: React.FC<{ data: number[]; color?: string; height?: number }> = ({
  data,
  color = '#3b82f6',
  height = 32,
}) => {
  if (!data.length) return null
  const w = 120
  const h = height
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w
      const y = h - ((v - min) / range) * h
      return `${x},${y}`
    })
    .join(' ')
  return (
    <svg width={w} height={h} className="overflow-visible">
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
    </svg>
  )
}

// ---------------------------------------------------------------------------
// GenerationProgress -- fitness history chart + metric cards
// ---------------------------------------------------------------------------

const GenerationProgress: React.FC<{
  history: GenerationStats[]
  current: GenerationStats
}> = ({ history, current }) => {
  const chartData = history.map((g, i) => ({
    gen: g.generation,
    best: g.best_fitness,
    mean: g.mean_fitness,
    worst: g.worst_fitness,
  }))

  const mutRateHistory = history.map((g) => g.mutation_rate)

  return (
    <div className="space-y-4">
      {/* Metric cards row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard
          label="Generation"
          value={current.generation}
          className="!bg-[#0d1117]"
        />
        <StatCard
          label="Population"
          value={current.population_size}
          className="!bg-[#0d1117]"
        />
        <StatCard
          label="Diversity Index"
          value={fmt2(current.diversity_index)}
          change={current.diversity_index - 0.5}
          changeLabel="vs 0.5"
          className="!bg-[#0d1117]"
        />
        <StatCard
          label="Best Fitness"
          value={fmt4(current.best_fitness)}
          change={current.best_fitness - current.mean_fitness}
          changeLabel="vs mean"
          className="!bg-[#0d1117]"
        />
      </div>

      {/* Mutation rate sparkline */}
      <div className="flex items-center gap-4 px-1">
        <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider whitespace-nowrap">
          Mutation Rate History
        </span>
        <SparkLine data={mutRateHistory} color="#8b5cf6" height={28} />
        <span className="text-xs font-mono text-slate-400 ml-2">
          {fmt4(current.mutation_rate)}
        </span>
        <span className="ml-auto text-[10px] font-mono text-slate-600">
          Stagnation: {current.stagnation_counter}
        </span>
      </div>

      {/* Fitness evolution chart */}
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2130" />
            <XAxis
              dataKey="gen"
              tick={{ fill: '#475569', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              label={{ value: 'Generation', position: 'insideBottom', offset: -2, fill: '#475569', fontSize: 10 }}
            />
            <YAxis
              tick={{ fill: '#475569', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              domain={['auto', 'auto']}
              width={48}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip
              contentStyle={{ background: '#111318', border: '1px solid #1e2130', fontSize: 11, fontFamily: 'JetBrains Mono' }}
              formatter={(v: number) => v.toFixed(4)}
              labelFormatter={(v) => `Gen ${v}`}
            />
            <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono' }} />
            <Line type="monotone" dataKey="best" stroke="#22c55e" dot={false} strokeWidth={2} name="Best" />
            <Line type="monotone" dataKey="mean" stroke="#3b82f6" dot={false} strokeWidth={1.5} name="Mean" />
            <Line type="monotone" dataKey="worst" stroke="#ef4444" dot={false} strokeWidth={1} strokeDasharray="4 2" name="Worst" />
            <ReferenceLine y={current.mean_fitness} stroke="#3b82f6" strokeDasharray="2 4" strokeOpacity={0.4} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// ParameterDriftGrid -- 12 parameter cards
// ---------------------------------------------------------------------------

const ParameterCard: React.FC<{ param: ParameterState }> = ({ param }) => {
  const meta = PARAM_META[param.name]
  const rangeWidth = param.max_allowed - param.min_allowed
  const fillPct = rangeWidth > 0
    ? ((param.current_value - param.min_allowed) / rangeWidth) * 100
    : 50
  const clampedFill = Math.max(0, Math.min(100, fillPct))

  const deltaColor =
    param.direction === 'improving'
      ? 'text-emerald-400'
      : param.direction === 'degrading'
      ? 'text-red-400'
      : 'text-slate-500'

  const precision = meta?.precision ?? 4

  return (
    <div className="bg-[#0d1117] border border-[#1e2130] rounded-lg p-3 space-y-2">
      <div className="flex items-start justify-between">
        <div>
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">
            {meta?.label ?? param.name}
          </span>
          <div className="text-sm font-mono font-semibold text-slate-100 mt-0.5">
            {param.current_value.toFixed(precision)}
            {meta?.unit && <span className="text-[10px] text-slate-500 ml-1">{meta.unit}</span>}
          </div>
        </div>
        <span className={clsx('text-[11px] font-mono font-medium', deltaColor)}>
          {fmtDelta(param.delta_24h)}
        </span>
      </div>

      {/* Range bar */}
      <div className="space-y-0.5">
        <div className="h-1.5 rounded-full bg-[#1e2130] overflow-hidden">
          <div
            className="h-full rounded-full bg-blue-500 transition-all duration-300"
            style={{ width: `${clampedFill}%` }}
          />
        </div>
        <div className="flex justify-between text-[9px] font-mono text-slate-600">
          <span>{param.min_allowed.toFixed(precision)}</span>
          <span>{param.max_allowed.toFixed(precision)}</span>
        </div>
      </div>

      <div className="text-[9px] font-mono text-slate-600">
        updated {timeAgo(param.last_updated)}
      </div>
    </div>
  )
}

// Mock parameter states from current stats -- in production these come from API
function buildMockParams(): ParameterState[] {
  const names = Object.keys(PARAM_META) as (keyof typeof PARAM_META)[]
  return names.map((name) => {
    const meta = PARAM_META[name]
    const cur = meta.name === 'HURST_WINDOW' ? 100 : meta.name === 'MIN_HOLD_BARS' ? 4 : 0.45 + Math.random() * 0.4
    const prev = cur + (Math.random() - 0.5) * 0.05
    const delta = cur - prev
    return {
      name,
      current_value: cur,
      previous_value: prev,
      delta_24h: delta,
      delta_pct: prev !== 0 ? delta / prev : 0,
      min_allowed: meta.name === 'HURST_WINDOW' ? 50 : meta.name === 'MIN_HOLD_BARS' ? 1 : 0.01,
      max_allowed: meta.name === 'HURST_WINDOW' ? 200 : meta.name === 'MIN_HOLD_BARS' ? 20 : 1.5,
      default_value: cur,
      direction: delta > 0 ? 'improving' : delta < 0 ? 'degrading' : 'neutral',
      last_updated: new Date(Date.now() - Math.random() * 3600_000).toISOString(),
      description: meta.description,
    } satisfies ParameterState
  })
}

const ParameterDriftGrid: React.FC<{ params?: ParameterState[] }> = ({ params }) => {
  const displayParams = params && params.length > 0 ? params : buildMockParams()
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
      {displayParams.map((p) => (
        <ParameterCard key={p.name} param={p} />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// TopGenomesTable -- sortable table with expandable rows
// ---------------------------------------------------------------------------

type GenomeSortKey = 'rank' | 'fitness' | 'sharpe' | 'max_drawdown'

const GenomeGeneList: React.FC<{ genes: Record<string, number> }> = ({ genes }) => (
  <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-1 p-3 bg-[#0a0d12] border-t border-[#1e2130]">
    {Object.entries(genes).map(([k, v]) => (
      <div key={k} className="text-[9px] font-mono">
        <span className="text-slate-600">{k}: </span>
        <span className="text-slate-300">{typeof v === 'number' ? v.toFixed(4) : v}</span>
      </div>
    ))}
  </div>
)

const TopGenomesTable: React.FC<{
  genomes: GenomeRecord[]
  onSelectGenome: (id: string) => void
  selectedId?: string
}> = ({ genomes, onSelectGenome, selectedId }) => {
  const [sortKey, setSortKey] = useState<GenomeSortKey>('rank')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const sorted = useMemo(() => {
    const arr = [...genomes]
    arr.sort((a, b) => {
      let av: number, bv: number
      if (sortKey === 'rank') { av = a.rank ?? 99; bv = b.rank ?? 99 }
      else if (sortKey === 'fitness') { av = a.fitness; bv = b.fitness }
      else if (sortKey === 'sharpe') { av = a.sharpe; bv = b.sharpe }
      else { av = a.max_drawdown; bv = b.max_drawdown }
      return sortDir === 'asc' ? av - bv : bv - av
    })
    return arr
  }, [genomes, sortKey, sortDir])

  function handleSort(key: GenomeSortKey) {
    if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('asc') }
  }

  const ColHeader: React.FC<{ label: string; k: GenomeSortKey }> = ({ label, k }) => (
    <th
      className="px-3 py-2 text-left text-[10px] font-mono uppercase tracking-wider text-slate-500 cursor-pointer hover:text-slate-300 select-none whitespace-nowrap"
      onClick={() => handleSort(k)}
    >
      {label}
      {sortKey === k && <span className="ml-1">{sortDir === 'asc' ? '↑' : '↓'}</span>}
    </th>
  )

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead className="border-b border-[#1e2130]">
          <tr>
            <ColHeader label="#" k="rank" />
            <th className="px-3 py-2 text-left text-[10px] font-mono uppercase tracking-wider text-slate-500">Genome ID</th>
            <ColHeader label="Fitness" k="fitness" />
            <ColHeader label="Sharpe" k="sharpe" />
            <ColHeader label="Max DD" k="max_drawdown" />
            <th className="px-3 py-2 text-left text-[10px] font-mono uppercase tracking-wider text-slate-500">Operator</th>
            <th className="px-3 py-2 text-left text-[10px] font-mono uppercase tracking-wider text-slate-500">Parent</th>
            <th className="px-3 py-2 text-left text-[10px] font-mono uppercase tracking-wider text-slate-500">Pedigree</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((g, i) => (
            <React.Fragment key={g.id}>
              <tr
                className={clsx(
                  'border-b border-[#1e2130] cursor-pointer transition-colors',
                  selectedId === g.id ? 'bg-blue-950/30' : 'hover:bg-[#1e2130]/40',
                )}
                onClick={() => setExpandedId(expandedId === g.id ? null : g.id)}
              >
                <td className="px-3 py-2 text-slate-400">{(g.rank ?? i) + 1}</td>
                <td className="px-3 py-2 text-slate-300 font-mono">{g.id}</td>
                <td className="px-3 py-2">
                  <span style={{ color: FITNESS_GRADIENT(g.fitness) }}>
                    {fmt4(g.fitness)}
                  </span>
                </td>
                <td className="px-3 py-2 text-blue-300">{fmt2(g.sharpe)}</td>
                <td className="px-3 py-2 text-red-400">{fmtPct(g.max_drawdown)}</td>
                <td className="px-3 py-2">
                  <span
                    className="px-1.5 py-0.5 rounded text-[9px] font-mono"
                    style={{
                      background: OPERATOR_COLORS[g.operator] + '22',
                      color: OPERATOR_COLORS[g.operator],
                    }}
                  >
                    {g.operator}
                  </span>
                </td>
                <td className="px-3 py-2 text-slate-600">
                  {g.parent_ids[0]?.slice(0, 6) ?? '--'}
                </td>
                <td className="px-3 py-2">
                  <button
                    className="text-[10px] text-blue-400 hover:text-blue-200 underline"
                    onClick={(e) => { e.stopPropagation(); onSelectGenome(g.id) }}
                  >
                    View
                  </button>
                </td>
              </tr>
              {expandedId === g.id && (
                <tr>
                  <td colSpan={8} className="p-0">
                    <GenomeGeneList genes={g.genes} />
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// GenomePedigree -- SVG tree diagram
// ---------------------------------------------------------------------------

const PEDIGREE_NODE_W = 110
const PEDIGREE_NODE_H = 48
const PEDIGREE_H_GAP = 50
const PEDIGREE_V_GAP = 70

function layoutPedigree(nodes: PedigreeNode[]): Map<string, { x: number; y: number }> {
  const byGen: Map<number, PedigreeNode[]> = new Map()
  for (const n of nodes) {
    if (!byGen.has(n.generation)) byGen.set(n.generation, [])
    byGen.get(n.generation)!.push(n)
  }
  const gens = Array.from(byGen.keys()).sort((a, b) => a - b)
  const positions = new Map<string, { x: number; y: number }>()
  gens.forEach((gen, row) => {
    const cols = byGen.get(gen)!
    cols.forEach((n, col) => {
      const totalW = cols.length * (PEDIGREE_NODE_W + PEDIGREE_H_GAP) - PEDIGREE_H_GAP
      const startX = -totalW / 2 + col * (PEDIGREE_NODE_W + PEDIGREE_H_GAP)
      positions.set(n.genome_id, {
        x: startX,
        y: row * (PEDIGREE_NODE_H + PEDIGREE_V_GAP),
      })
    })
  })
  return positions
}

const GenomePedigreeView: React.FC<{ pedigree: ReturnType<typeof import('@/hooks/useIAEEvolution').useGenomePedigree>['data'] }> = ({
  pedigree,
}) => {
  if (!pedigree) return null

  const positions = layoutPedigree(pedigree.nodes)
  const maxRows = Math.max(...pedigree.nodes.map((n) => n.generation))
  const minRows = Math.min(...pedigree.nodes.map((n) => n.generation))
  const rowCount = maxRows - minRows + 1
  const svgH = rowCount * (PEDIGREE_NODE_H + PEDIGREE_V_GAP) + 20
  const svgW = 600

  const centerX = svgW / 2
  const centerY = 20

  return (
    <div className="overflow-x-auto">
      <svg
        width={svgW}
        height={svgH}
        className="mx-auto"
        style={{ fontFamily: 'JetBrains Mono' }}
      >
        {/* Edges */}
        {pedigree.edges.map((edge, i) => {
          const from = positions.get(edge.from)
          const to = positions.get(edge.to)
          if (!from || !to) return null
          const x1 = centerX + from.x + PEDIGREE_NODE_W / 2
          const y1 = centerY + from.y + PEDIGREE_NODE_H
          const x2 = centerX + to.x + PEDIGREE_NODE_W / 2
          const y2 = centerY + to.y
          const cy = (y1 + y2) / 2
          return (
            <path
              key={i}
              d={`M ${x1} ${y1} C ${x1} ${cy} ${x2} ${cy} ${x2} ${y2}`}
              fill="none"
              stroke="#1e3a5f"
              strokeWidth={1.5}
            />
          )
        })}

        {/* Nodes */}
        {pedigree.nodes.map((node) => {
          const pos = positions.get(node.genome_id)
          if (!pos) return null
          const nx = centerX + pos.x
          const ny = centerY + pos.y
          const color = FITNESS_GRADIENT(node.fitness)
          const isRoot = node.genome_id === pedigree.root_id
          return (
            <g key={node.genome_id} transform={`translate(${nx}, ${ny})`}>
              <rect
                width={PEDIGREE_NODE_W}
                height={PEDIGREE_NODE_H}
                rx={4}
                fill="#0d1117"
                stroke={isRoot ? color : '#1e2130'}
                strokeWidth={isRoot ? 2 : 1}
              />
              {/* Fitness indicator bar */}
              <rect
                x={0}
                y={PEDIGREE_NODE_H - 3}
                width={PEDIGREE_NODE_W * node.fitness}
                height={3}
                rx={1}
                fill={color}
              />
              <text x={6} y={16} fill="#94a3b8" fontSize={9}>
                {node.genome_id.slice(0, 8)}
              </text>
              <text x={6} y={28} fill={color} fontSize={10} fontWeight={600}>
                f={node.fitness.toFixed(3)}
              </text>
              <text x={6} y={40} fill="#475569" fontSize={9}>
                SR={node.sharpe.toFixed(2)}
              </text>
              <circle
                cx={PEDIGREE_NODE_W - 8}
                cy={12}
                r={4}
                fill={OPERATOR_COLORS[node.operator] ?? '#475569'}
              />
            </g>
          )
        })}
      </svg>

      {/* Legend */}
      <div className="flex gap-4 justify-center mt-2">
        {Object.entries(OPERATOR_COLORS).map(([op, col]) => (
          <div key={op} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full" style={{ background: col }} />
            <span className="text-[10px] font-mono text-slate-500">{op}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// RollbackHistory -- table of rollback events
// ---------------------------------------------------------------------------

const REASON_COLORS: Record<string, string> = {
  sharpe_degradation: 'text-yellow-400',
  drawdown_breach: 'text-red-400',
  fitness_regression: 'text-orange-400',
  manual_override: 'text-blue-400',
  circuit_breaker: 'text-purple-400',
}

const RollbackHistory: React.FC<{ rollbacks: RollbackEvent[] }> = ({ rollbacks }) => {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead className="border-b border-[#1e2130]">
          <tr>
            {['Timestamp', 'Reason', 'From Gen', 'To Gen', 'Sharpe Before', 'Sharpe After', 'Delta', 'By'].map((h) => (
              <th key={h} className="px-3 py-2 text-left text-[10px] font-mono uppercase tracking-wider text-slate-500">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rollbacks.map((r) => {
            const sharpeDelta = r.sharpe_after - r.sharpe_before
            const paramKeys = Object.keys(r.parameter_delta)
            return (
              <tr key={r.id} className="border-b border-[#1e2130] hover:bg-[#1e2130]/30">
                <td className="px-3 py-2 text-slate-500 whitespace-nowrap">
                  {new Date(r.timestamp).toLocaleString()}
                </td>
                <td className={clsx('px-3 py-2 whitespace-nowrap', REASON_COLORS[r.reason] ?? 'text-slate-400')}>
                  {r.reason.replace(/_/g, ' ')}
                </td>
                <td className="px-3 py-2 text-slate-400">{r.from_generation}</td>
                <td className="px-3 py-2 text-slate-400">{r.to_generation}</td>
                <td className="px-3 py-2 text-slate-300">{fmt2(r.sharpe_before)}</td>
                <td className="px-3 py-2 text-slate-300">{fmt2(r.sharpe_after)}</td>
                <td className={clsx('px-3 py-2', sharpeDelta >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                  {fmtDelta(sharpeDelta)}
                </td>
                <td className="px-3 py-2 text-slate-600">
                  {r.initiated_by === 'manual' ? (
                    <span className="text-blue-400">{r.operator_id ?? 'manual'}</span>
                  ) : (
                    <span className="text-slate-600">auto</span>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {rollbacks.length === 0 && (
        <div className="text-center py-8 text-slate-600 text-sm font-mono">
          No rollback events recorded
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section header
// ---------------------------------------------------------------------------

const SectionHeader: React.FC<{ title: string; subtitle?: string; badge?: string }> = ({
  title,
  subtitle,
  badge,
}) => (
  <div className="flex items-center gap-3 mb-4">
    <div>
      <h2 className="text-sm font-mono font-semibold text-slate-200 uppercase tracking-wider">{title}</h2>
      {subtitle && <p className="text-[10px] font-mono text-slate-500 mt-0.5">{subtitle}</p>}
    </div>
    {badge && (
      <span className="ml-auto px-2 py-0.5 text-[10px] font-mono rounded bg-blue-900/40 text-blue-300 border border-blue-800/40">
        {badge}
      </span>
    )}
  </div>
)

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export const IAEEvolution: React.FC = () => {
  const { data: evolution, isLoading, error } = useIAEEvolution()
  const { data: topGenomes } = useTopGenomes(10)
  const { data: rollbacks } = useRollbackHistory(10)
  const [pedigreeGenomeId, setPedigreeGenomeId] = useState<string>('')
  const { data: pedigree } = useGenomePedigree(pedigreeGenomeId)

  // Set default pedigree genome to best genome
  React.useEffect(() => {
    if (evolution?.current_stats?.best_genome_id && !pedigreeGenomeId) {
      setPedigreeGenomeId(evolution.current_stats.best_genome_id)
    }
  }, [evolution?.current_stats?.best_genome_id])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner />
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-red-400 text-sm font-mono p-8">
        Failed to load IAE evolution data: {(error as Error).message}
      </div>
    )
  }

  const stats = evolution!
  const running = stats.config.running && !stats.config.paused

  return (
    <div className="space-y-6 p-4">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-base font-mono font-bold text-slate-100 uppercase tracking-widest">
            IAE Evolution Dashboard
          </h1>
          <p className="text-[11px] font-mono text-slate-500 mt-1">
            Iterative Adaptation Engine -- Genetic optimizer for SRFM parameters
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className={clsx(
            'flex items-center gap-1.5 px-3 py-1.5 rounded text-[11px] font-mono',
            running ? 'bg-emerald-900/30 text-emerald-400 border border-emerald-800/40'
              : 'bg-yellow-900/30 text-yellow-400 border border-yellow-800/40',
          )}>
            <div className={clsx('w-1.5 h-1.5 rounded-full', running ? 'bg-emerald-400 animate-pulse' : 'bg-yellow-400')} />
            {running ? 'RUNNING' : stats.config.paused ? 'PAUSED' : 'STOPPED'}
          </div>
          <div className="text-[10px] font-mono text-slate-600">
            Pop: {stats.config.population_size} | Max Gen: {stats.config.max_generations}
          </div>
        </div>
      </div>

      {/* Section 1 -- Generation Progress */}
      <Card padding="lg">
        <SectionHeader
          title="Generation Progress"
          subtitle="Fitness evolution over generations"
          badge={`Gen ${stats.current_stats.generation}`}
        />
        <GenerationProgress history={stats.history} current={stats.current_stats} />
      </Card>

      {/* Section 2 -- Parameter Drift */}
      <Card padding="lg">
        <SectionHeader
          title="Parameter Drift Grid"
          subtitle="Current gene values vs 24h baseline -- green = improving, red = degrading"
        />
        <ParameterDriftGrid params={stats.parameters} />
      </Card>

      {/* Section 3 -- Top Genomes */}
      <Card padding="lg">
        <SectionHeader
          title="Top Genomes"
          subtitle="Click any row to expand all 31 gene values -- click Pedigree to trace lineage"
          badge="Top 10"
        />
        <TopGenomesTable
          genomes={topGenomes ?? stats.top_genomes}
          onSelectGenome={setPedigreeGenomeId}
          selectedId={pedigreeGenomeId}
        />
      </Card>

      {/* Section 4 -- Genome Pedigree */}
      <Card padding="lg">
        <SectionHeader
          title="Genome Pedigree"
          subtitle={`Lineage tree for genome ${pedigreeGenomeId.slice(0, 8) || '...'} -- last 3 generations`}
        />
        {pedigreeGenomeId ? (
          pedigree ? (
            <GenomePedigreeView pedigree={pedigree} />
          ) : (
            <div className="flex items-center justify-center h-24">
              <LoadingSpinner />
            </div>
          )
        ) : (
          <div className="text-center py-8 text-slate-600 text-sm font-mono">
            Select a genome from the table above to view its pedigree
          </div>
        )}
      </Card>

      {/* Section 5 -- Rollback History */}
      <Card padding="lg">
        <SectionHeader
          title="Rollback History"
          subtitle="Recent automatic and manual rollback events"
          badge={`${(rollbacks ?? stats.recent_rollbacks).length} events`}
        />
        <RollbackHistory rollbacks={rollbacks ?? stats.recent_rollbacks} />
      </Card>
    </div>
  )
}

export default IAEEvolution
