// ============================================================
// IAEDashboard.tsx -- IAE Iterative Adaptation Engine monitor
// Shows genome evolution, parameter drift, fitness landscape
// ============================================================
import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'
import { clsx } from 'clsx'

// ---- Types ----

interface EvolutionPoint {
  generation: number
  best: number
  avg: number
  worst: number
  diversity: number
}

interface ParameterRow {
  name: string
  value: number
  delta24h: number
  min: number
  max: number
  drift: number -- drift as fraction of allowed range
}

interface GenomeEntry {
  id: string
  fitness: number
  params: number[] -- normalized [0,1] per param axis
  generation: number
}

interface RollbackEntry {
  timestamp: string
  param: string
  from: number
  to: number
  reason: string
  magnitude: number
}

interface IAEStatus {
  generation: number
  best_fitness: number
  diversity_score: number
  mutation_rate: number
  mutation_history: number[]
  evolution_history: EvolutionPoint[]
  parameters: ParameterRow[]
  top_genomes: GenomeEntry[]
  fitness_grid: number[][] -- 20x20
  rollback_history: RollbackEntry[]
}

// ---- Demo data generators ----

function buildDemoStatus(seed: number): IAEStatus {
  const gen = 120 + Math.floor(seed * 0.1)
  const history: EvolutionPoint[] = []
  for (let g = 1; g <= gen; g++) {
    const base = 0.3 + (g / gen) * 0.5
    history.push({
      generation: g,
      best: Math.min(0.98, base + 0.15 + Math.sin(g * 0.3) * 0.04),
      avg: base + Math.sin(g * 0.2) * 0.03,
      worst: Math.max(0.01, base - 0.15 + Math.cos(g * 0.25) * 0.05),
      diversity: Math.max(0.05, 0.6 - (g / gen) * 0.4 + Math.sin(g * 0.15) * 0.1),
    })
  }
  const PARAM_NAMES = [
    'bh_mass_threshold', 'nav_entry_gate', 'hurst_min', 'garch_vol_cap',
    'signal_decay_rate', 'position_size_k', 'drawdown_limit', 'momentum_window',
    'vol_target', 'mean_reversion_z', 'regime_confidence', 'correlation_cap',
  ]
  const parameters: ParameterRow[] = PARAM_NAMES.map((name, i) => {
    const min = 0.1 + i * 0.05
    const max = min + 0.5 + (i % 3) * 0.2
    const value = min + (max - min) * (0.4 + Math.sin(seed + i) * 0.3)
    const delta24h = (Math.sin(seed * 0.7 + i) * 0.05)
    const drift = Math.abs(delta24h) / (max - min)
    return { name, value, delta24h, min, max, drift }
  })
  const AXES = ['bh_mass', 'nav_curv', 'hurst', 'garch_vol', 'signal_decay', 'pos_size']
  const top_genomes: GenomeEntry[] = Array.from({ length: 10 }, (_, i) => ({
    id: `G${gen - i}`,
    fitness: 0.98 - i * 0.04 + Math.sin(i * 1.3) * 0.01,
    params: AXES.map((_, j) => Math.max(0, Math.min(1, 0.5 + Math.sin(i * 1.1 + j * 0.9) * 0.4))),
    generation: gen - i * 3,
  }))
  const fitness_grid: number[][] = Array.from({ length: 20 }, (_, r) =>
    Array.from({ length: 20 }, (_, c) => {
      const dx = (c - 10) / 10
      const dy = (r - 10) / 10
      return Math.max(0, Math.min(1, 0.5 + Math.exp(-(dx * dx + dy * dy) * 2) * 0.5 + Math.sin(dx * 4 + seed * 0.01) * 0.1))
    })
  )
  const mutation_history = Array.from({ length: 40 }, (_, i) =>
    Math.max(0.01, 0.08 * Math.exp(-i * 0.03) + Math.abs(Math.sin(i * 0.5)) * 0.02)
  )
  const rollback_history: RollbackEntry[] = [
    { timestamp: '09:14:32', param: 'bh_mass_threshold', from: 0.97, to: 0.92, reason: 'OOS degradation > 15%', magnitude: 0.052 },
    { timestamp: '11:02:17', param: 'nav_entry_gate', from: 3.4, to: 3.0, reason: 'IC decay spike', magnitude: 0.118 },
    { timestamp: '13:45:08', param: 'garch_vol_cap', from: 0.41, to: 0.35, reason: 'Max drawdown breach', magnitude: 0.146 },
    { timestamp: '15:33:51', param: 'signal_decay_rate', from: 0.12, to: 0.08, reason: 'Fitness plateau', magnitude: 0.033 },
  ]
  return {
    generation: gen,
    best_fitness: history[history.length - 1]?.best ?? 0.85,
    diversity_score: history[history.length - 1]?.diversity ?? 0.22,
    mutation_rate: mutation_history[mutation_history.length - 1],
    mutation_history,
    evolution_history: history,
    parameters,
    top_genomes,
    fitness_grid,
    rollback_history,
  }
}

// ---- Helpers ----

function fitnessColor(f: number): string {
  -- green=best, red=worst, interpolate
  const r = Math.round(255 * (1 - f))
  const g = Math.round(255 * f)
  return `rgb(${r},${g},50)`
}

function gridColor(v: number): string {
  const r = Math.round(180 * (1 - v) + 20)
  const g = Math.round(180 * v + 20)
  return `rgb(${r},${g},60)`
}

function fmt3(v: number) { return v.toFixed(3) }
function fmtPct(v: number) { return `${(v * 100).toFixed(1)}%` }

// ---- Sub-components ----

const GenomeEvolutionChart: React.FC<{ data: EvolutionPoint[] }> = ({ data }) => {
  const recent = data.slice(-80)
  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Genome Fitness Evolution
      </h3>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={recent} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="generation" tick={{ fill: '#9ca3af', fontSize: 10 }} />
          <YAxis domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
            labelStyle={{ color: '#e5e7eb' }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="best" stroke="#22c55e" dot={false} strokeWidth={2} name="Best" />
          <Line type="monotone" dataKey="avg" stroke="#3b82f6" dot={false} strokeWidth={1.5} name="Avg" />
          <Line type="monotone" dataKey="worst" stroke="#ef4444" dot={false} strokeWidth={1} name="Worst" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

const ParameterDriftTable: React.FC<{ params: ParameterRow[] }> = ({ params }) => (
  <div className="bg-gray-900 border border-gray-700 rounded p-3">
    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
      Parameter Drift
    </h3>
    <div className="overflow-auto max-h-52">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-gray-500 border-b border-gray-700">
            <th className="text-left py-1 pr-2">Param</th>
            <th className="text-right pr-2">Value</th>
            <th className="text-right pr-2">24h Δ</th>
            <th className="text-right pr-2">Range</th>
            <th className="text-right">Drift%</th>
          </tr>
        </thead>
        <tbody>
          {params.map((p) => (
            <tr key={p.name} className="border-b border-gray-800 hover:bg-gray-800">
              <td className="py-1 pr-2 font-mono text-gray-300">{p.name}</td>
              <td className="text-right pr-2 text-white font-mono">{fmt3(p.value)}</td>
              <td className={clsx('text-right pr-2 font-mono', p.delta24h >= 0 ? 'text-green-400' : 'text-red-400')}>
                {p.delta24h >= 0 ? '+' : ''}{fmt3(p.delta24h)}
              </td>
              <td className="text-right pr-2 text-gray-500">[{fmt3(p.min)},{fmt3(p.max)}]</td>
              <td className={clsx('text-right font-mono', p.drift > 0.2 ? 'text-yellow-400' : 'text-gray-400')}>
                {fmtPct(p.drift)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
)

const AXIS_LABELS = ['bh_mass', 'nav_curv', 'hurst', 'garch_vol', 'signal_decay', 'pos_size']

const GenomeParallelCoords: React.FC<{ genomes: GenomeEntry[] }> = ({ genomes }) => {
  const W = 520
  const H = 200
  const PAD_L = 20
  const PAD_R = 20
  const PAD_T = 30
  const PAD_B = 20
  const n = AXIS_LABELS.length
  const axisX = (i: number) => PAD_L + (i / (n - 1)) * (W - PAD_L - PAD_R)
  const valY = (v: number) => PAD_T + (1 - v) * (H - PAD_T - PAD_B)

  const maxF = Math.max(...genomes.map((g) => g.fitness))
  const minF = Math.min(...genomes.map((g) => g.fitness))

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Genome Parallel Coordinates -- Top 10
      </h3>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
        {/* Axes */}
        {AXIS_LABELS.map((label, i) => (
          <g key={label}>
            <line
              x1={axisX(i)} y1={PAD_T}
              x2={axisX(i)} y2={H - PAD_B}
              stroke="#4b5563" strokeWidth={1}
            />
            <text x={axisX(i)} y={PAD_T - 6} textAnchor="middle" fill="#9ca3af" fontSize={9}>
              {label}
            </text>
            <text x={axisX(i)} y={H - PAD_B + 14} textAnchor="middle" fill="#6b7280" fontSize={8}>0</text>
            <text x={axisX(i)} y={PAD_T - 2} textAnchor="middle" fill="#6b7280" fontSize={7}></text>
          </g>
        ))}
        {/* Genome polylines */}
        {genomes.map((genome) => {
          const t = maxF > minF ? (genome.fitness - minF) / (maxF - minF) : 0.5
          const r = Math.round(255 * (1 - t))
          const g = Math.round(220 * t)
          const color = `rgba(${r},${g},50,0.75)`
          const points = genome.params.map((v, i) => `${axisX(i)},${valY(v)}`).join(' ')
          return (
            <polyline
              key={genome.id}
              points={points}
              fill="none"
              stroke={color}
              strokeWidth={1.5}
            />
          )
        })}
      </svg>
      <div className="flex items-center gap-3 mt-1">
        <span className="text-xs text-gray-500">Fitness:</span>
        <div className="flex items-center gap-1">
          <div className="w-8 h-2 rounded" style={{ background: 'linear-gradient(to right, rgb(255,0,50), rgb(0,220,50))' }} />
          <span className="text-xs text-gray-500">low -- high</span>
        </div>
      </div>
    </div>
  )
}

const MutationRateGauge: React.FC<{ rate: number; history: number[] }> = ({ rate, history }) => {
  const sparkData = history.map((v, i) => ({ i, v }))
  const maxRate = 0.12
  const angle = (rate / maxRate) * 180 -- 0=left, 180=right
  const rad = ((angle - 90) * Math.PI) / 180
  const cx = 60; const cy = 55; const r = 44
  const nx = cx + r * Math.cos(rad)
  const ny = cy + r * Math.sin(rad)

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Mutation Rate
      </h3>
      <div className="flex items-center gap-4">
        <svg width={120} height={70}>
          <path
            d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
            fill="none" stroke="#1f2937" strokeWidth={10}
          />
          {/* colored arc based on rate */}
          <path
            d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${nx} ${ny}`}
            fill="none"
            stroke={rate > 0.07 ? '#ef4444' : rate > 0.04 ? '#eab308' : '#22c55e'}
            strokeWidth={10}
          />
          <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="white" strokeWidth={2} />
          <circle cx={cx} cy={cy} r={3} fill="white" />
          <text x={cx} y={cy + 14} textAnchor="middle" fill="white" fontSize={11} fontWeight="bold">
            {fmtPct(rate)}
          </text>
        </svg>
        <div className="flex-1">
          <div className="text-xs text-gray-500 mb-1">History (40 gens)</div>
          <ResponsiveContainer width="100%" height={50}>
            <LineChart data={sparkData}>
              <Line type="monotone" dataKey="v" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
              <YAxis domain={[0, maxRate]} hide />
              <XAxis dataKey="i" hide />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

const FitnessLandscapeHeatmap: React.FC<{ grid: number[][] }> = ({ grid }) => {
  const CELL = 14
  const N = 20
  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Fitness Landscape (PCA Projection)
      </h3>
      <svg width={N * CELL + 40} height={N * CELL + 30}>
        {grid.map((row, r) =>
          row.map((v, c) => (
            <rect
              key={`${r}-${c}`}
              x={c * CELL + 30}
              y={r * CELL}
              width={CELL - 1}
              height={CELL - 1}
              fill={gridColor(v)}
              opacity={0.9}
            />
          ))
        )}
        {/* Axis labels */}
        <text x={30 + (N * CELL) / 2} y={N * CELL + 16} textAnchor="middle" fill="#6b7280" fontSize={9}>
          PC1
        </text>
        <text
          x={10} y={(N * CELL) / 2}
          textAnchor="middle" fill="#6b7280" fontSize={9}
          transform={`rotate(-90, 10, ${(N * CELL) / 2})`}
        >
          PC2
        </text>
        {/* Color scale */}
        {Array.from({ length: N }, (_, i) => (
          <rect key={i} x={N * CELL + 32} y={i * CELL} width={6} height={CELL}
            fill={gridColor(1 - i / N)} />
        ))}
        <text x={N * CELL + 38} y={8} fill="#6b7280" fontSize={8}>1.0</text>
        <text x={N * CELL + 38} y={N * CELL - 2} fill="#6b7280" fontSize={8}>0.0</text>
      </svg>
    </div>
  )
}

const RollbackHistory: React.FC<{ entries: RollbackEntry[] }> = ({ entries }) => (
  <div className="bg-gray-900 border border-gray-700 rounded p-3">
    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
      Rollback History
    </h3>
    <table className="w-full text-xs">
      <thead>
        <tr className="text-gray-500 border-b border-gray-700">
          <th className="text-left py-1 pr-2">Time</th>
          <th className="text-left pr-2">Param</th>
          <th className="text-right pr-2">From</th>
          <th className="text-right pr-2">To</th>
          <th className="text-right pr-2">Δ</th>
          <th className="text-left">Reason</th>
        </tr>
      </thead>
      <tbody>
        {entries.map((e, i) => (
          <tr key={i} className="border-b border-gray-800">
            <td className="py-1 pr-2 font-mono text-gray-400">{e.timestamp}</td>
            <td className="pr-2 text-yellow-400 font-mono">{e.param}</td>
            <td className="text-right pr-2 text-gray-300">{fmt3(e.from)}</td>
            <td className="text-right pr-2 text-white">{fmt3(e.to)}</td>
            <td className="text-right pr-2 text-red-400">-{fmtPct(e.magnitude)}</td>
            <td className="text-gray-500">{e.reason}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)

// ---- Main Page ----

export const IAEDashboard: React.FC = () => {
  const [status, setStatus] = useState<IAEStatus>(() => buildDemoStatus(0))
  const [tick, setTick] = useState(0)
  const [loading, setLoading] = useState(false)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/iae/evolution/status')
      if (res.ok) {
        const data = await res.json()
        setStatus(data)
      }
    } catch {
      -- API not available, use demo data with small perturbations
      setStatus(buildDemoStatus(Date.now() * 0.001))
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const id = setInterval(() => {
      setTick((t) => t + 1)
      fetchStatus()
    }, 5000)
    return () => clearInterval(id)
  }, [fetchStatus])

  const diversityPct = fmtPct(status.diversity_score)

  return (
    <div className="min-h-screen bg-gray-950 text-white p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-6">
          <h1 className="text-lg font-bold tracking-wide text-white">
            IAE -- Iterative Adaptation Engine
          </h1>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs text-green-400">EVOLVING</span>
          </div>
        </div>
        <div className="flex items-center gap-6 text-sm">
          <div className="text-center">
            <div className="text-gray-500 text-xs">Generation</div>
            <div className="text-white font-bold font-mono">{status.generation}</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500 text-xs">Best Fitness</div>
            <div className="text-green-400 font-bold font-mono">{fmt3(status.best_fitness)}</div>
          </div>
          <div className="text-center">
            <div className="text-gray-500 text-xs">Diversity</div>
            <div className={clsx('font-bold font-mono', status.diversity_score < 0.15 ? 'text-yellow-400' : 'text-blue-400')}>
              {diversityPct}
            </div>
          </div>
          <div className="text-center">
            <div className="text-gray-500 text-xs">Mutation Rate</div>
            <div className="text-purple-400 font-bold font-mono">{fmtPct(status.mutation_rate)}</div>
          </div>
          <div className="text-xs text-gray-600">
            Poll 5s -- tick {tick}
          </div>
        </div>
      </div>

      {/* Row 1: Evolution chart + Parallel coords */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <GenomeEvolutionChart data={status.evolution_history} />
        <GenomeParallelCoords genomes={status.top_genomes} />
      </div>

      {/* Row 2: Mutation gauge + Fitness landscape + Parameter drift */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <MutationRateGauge rate={status.mutation_rate} history={status.mutation_history} />
        <FitnessLandscapeHeatmap grid={status.fitness_grid} />
        <ParameterDriftTable params={status.parameters} />
      </div>

      {/* Row 3: Rollback history */}
      <RollbackHistory entries={status.rollback_history} />
    </div>
  )
}

export default IAEDashboard
