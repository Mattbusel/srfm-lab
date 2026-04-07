// ============================================================
// BHPhysicsPanel.tsx -- Deep BH physics visualization panel
// Minkowski metric, event horizon, geodesics, quaternions,
// spacetime flow field
// ============================================================
import React, { useState, useEffect, useRef, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { clsx } from 'clsx'

// ---- Constants ----

const BH_COLLAPSE = 0.992
const NAV_GEO_ENTRY_GATE = 3.0
const C_LIGHT = 1.0 -- normalized speed of light

// ---- Props ----

export interface BHPhysicsPanelProps {
  bhMass: number
  navCurvature: number
  geodesicDev: number
  quaternion: [number, number, number, number] -- [w, x, y, z]
}

// ---- Helper types ----

interface GeodesicPoint {
  bar: number
  value: number
}

// ---- Demo data helpers ----

function buildGeodesicHistory(current: number): GeodesicPoint[] {
  const pts: GeodesicPoint[] = []
  for (let i = 95; i >= 0; i--) {
    const t = i / 95
    pts.push({
      bar: 96 - i,
      value: Math.max(0, current * t + (1 - t) * 1.2 + Math.sin(i * 0.4) * 0.5),
    })
  }
  return pts
}

// ---- MinkowskiMetricDisplay ----

const MinkowskiMetricDisplay: React.FC<{
  bhMass: number
  navCurvature: number
}> = ({ bhMass, navCurvature }) => {
  const [animPhase, setAnimPhase] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setAnimPhase((p) => (p + 0.05) % (Math.PI * 2)), 50)
    return () => clearInterval(id)
  }, [])

  -- Compute spacetime interval components
  const dt = 1.0
  const dx = navCurvature * 0.1 * (1 + Math.sin(animPhase) * 0.05)
  const dy = bhMass * 0.05 * (1 + Math.cos(animPhase * 1.3) * 0.04)
  const dz = 0.01 + Math.sin(animPhase * 0.7) * 0.005
  const ds2 = -(C_LIGHT * C_LIGHT * dt * dt) + dx * dx + dy * dy + dz * dz
  const timelike = ds2 < 0

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Minkowski Spacetime Metric
      </h3>
      <div className="font-mono text-center mb-3">
        <div className="text-gray-400 text-xs mb-1">ds&#178; = -c&#178;dt&#178; + dx&#178; + dy&#178; + dz&#178;</div>
        <div className="text-white text-sm">
          ds&#178; ={' '}
          <span className={clsx('font-bold', timelike ? 'text-blue-400' : 'text-yellow-400')}>
            {ds2.toFixed(6)}
          </span>
        </div>
        <div className={clsx('text-xs mt-1', timelike ? 'text-blue-400' : 'text-yellow-400')}>
          {timelike ? 'TIMELIKE -- causal' : 'SPACELIKE -- acausal'}
        </div>
      </div>
      {/* SVG spacetime diagram */}
      <svg width="100%" viewBox="0 0 200 120">
        {/* Light cone lines */}
        <line x1={100} y1={60} x2={40} y2={0} stroke="#6b7280" strokeWidth={1} strokeDasharray="3,2" opacity={0.5} />
        <line x1={100} y1={60} x2={160} y2={0} stroke="#6b7280" strokeWidth={1} strokeDasharray="3,2" opacity={0.5} />
        <line x1={100} y1={60} x2={40} y2={120} stroke="#6b7280" strokeWidth={1} strokeDasharray="3,2" opacity={0.5} />
        <line x1={100} y1={60} x2={160} y2={120} stroke="#6b7280" strokeWidth={1} strokeDasharray="3,2" opacity={0.5} />
        {/* Time axis */}
        <line x1={100} y1={0} x2={100} y2={120} stroke="#4b5563" strokeWidth={1} />
        {/* Space axis */}
        <line x1={20} y1={60} x2={180} y2={60} stroke="#4b5563" strokeWidth={1} />
        {/* Animated event */}
        <circle
          cx={100 + dx * 300}
          cy={60 - dt * 30 * (1 + Math.sin(animPhase) * 0.1)}
          r={4}
          fill={timelike ? '#3b82f6' : '#eab308'}
        />
        {/* World line */}
        <line
          x1={100} y1={60}
          x2={100 + dx * 300}
          y2={60 - dt * 30}
          stroke={timelike ? '#3b82f650' : '#eab30850'}
          strokeWidth={2}
        />
        <text x={105} y={10} fill="#6b7280" fontSize={9}>t</text>
        <text x={175} y={58} fill="#6b7280" fontSize={9}>x</text>
        <text x={40} y={25} fill="#6b7280" fontSize={8}>c=1</text>
        {/* Component labels */}
        <text x={4} y={115} fill="#9ca3af" fontSize={8}>
          dt={dt.toFixed(2)} dx={dx.toFixed(3)} dy={dy.toFixed(3)} dz={dz.toFixed(4)}
        </text>
      </svg>
    </div>
  )
}

// ---- EventHorizonGauge ----

const EventHorizonGauge: React.FC<{ bhMass: number }> = ({ bhMass }) => {
  const [animTick, setAnimTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setAnimTick((t) => t + 1), 80)
    return () => clearInterval(id)
  }, [])

  const clampedMass = Math.max(0, Math.min(1.1, bhMass))
  const fillAngle = (clampedMass / 1.1) * 360

  -- SVG arc helpers
  const polarToXY = (angle: number, r: number, cx: number, cy: number) => {
    const rad = ((angle - 90) * Math.PI) / 180
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
  }

  const describeArc = (cx: number, cy: number, r: number, startAngle: number, endAngle: number) => {
    const s = polarToXY(startAngle, r, cx, cy)
    const e = polarToXY(endAngle, r, cx, cy)
    const large = endAngle - startAngle > 180 ? 1 : 0
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`
  }

  const collapseAngle = (BH_COLLAPSE / 1.1) * 360
  const isNearCollapse = bhMass > 0.95

  -- Pulsing alpha near collapse
  const pulseAlpha = isNearCollapse
    ? 0.6 + 0.4 * Math.abs(Math.sin(animTick * 0.15))
    : 1.0

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Event Horizon Gauge
      </h3>
      <div className="flex items-center justify-center">
        <svg width={160} height={160}>
          {/* Background ring */}
          <circle cx={80} cy={80} r={60} fill="none" stroke="#1f2937" strokeWidth={16} />
          {/* Filled arc */}
          {fillAngle > 0 && (
            <path
              d={describeArc(80, 80, 60, 0, Math.min(fillAngle, 359.9))}
              fill="none"
              stroke={bhMass > BH_COLLAPSE ? `rgba(239,68,68,${pulseAlpha})` : bhMass > 0.8 ? '#f59e0b' : '#22c55e'}
              strokeWidth={16}
              strokeLinecap="butt"
              style={{ transition: 'stroke 0.3s' }}
            />
          )}
          {/* Collapse threshold marker */}
          {(() => {
            const p = polarToXY(collapseAngle, 60, 80, 80)
            return <circle cx={p.x} cy={p.y} r={5} fill="#ef4444" stroke="#fff" strokeWidth={1.5} />
          })()}
          {/* Center text */}
          <text x={80} y={74} textAnchor="middle" fill="white" fontSize={20} fontWeight="bold">
            {bhMass.toFixed(3)}
          </text>
          <text x={80} y={92} textAnchor="middle" fill="#9ca3af" fontSize={10}>
            bh_mass
          </text>
          {isNearCollapse && (
            <text x={80} y={108} textAnchor="middle" fill="#ef4444" fontSize={9} opacity={pulseAlpha}>
              NEAR COLLAPSE
            </text>
          )}
          {/* Collapse label */}
          {(() => {
            const p = polarToXY(collapseAngle + 8, 46, 80, 80)
            return <text x={p.x} y={p.y} textAnchor="middle" fill="#ef4444" fontSize={8}>{BH_COLLAPSE}</text>
          })()}
        </svg>
      </div>
      <div className="text-center text-xs text-gray-500 mt-1">
        BH_COLLAPSE threshold = {BH_COLLAPSE} &nbsp;|&nbsp;
        <span className={bhMass > BH_COLLAPSE ? 'text-red-400 font-bold' : 'text-gray-400'}>
          {bhMass > BH_COLLAPSE ? 'COLLAPSED' : `${((BH_COLLAPSE - bhMass) / BH_COLLAPSE * 100).toFixed(1)}% headroom`}
        </span>
      </div>
    </div>
  )
}

// ---- GeodesicDeviationPlot ----

const GeodesicDeviationPlot: React.FC<{ geodesicDev: number }> = ({ geodesicDev }) => {
  const data = useMemo(() => buildGeodesicHistory(geodesicDev), [geodesicDev])

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Geodesic Deviation -- 96 bars
      </h3>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <defs>
            <linearGradient id="geoRedFade" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#ef4444" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="bar" tick={{ fill: '#9ca3af', fontSize: 10 }} />
          <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
            formatter={(v: number) => v.toFixed(4)}
          />
          <ReferenceLine
            y={NAV_GEO_ENTRY_GATE}
            stroke="#ef4444"
            strokeDasharray="4 2"
            label={{ value: `GEO_GATE=${NAV_GEO_ENTRY_GATE}`, fill: '#ef4444', fontSize: 9 }}
          />
          <Line
            type="monotone" dataKey="value" stroke="#a78bfa" dot={false} strokeWidth={2}
            name="geo_dev"
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="text-xs text-gray-500 mt-1">
        Current: <span className={clsx('font-mono', geodesicDev > NAV_GEO_ENTRY_GATE ? 'text-red-400' : 'text-green-400')}>
          {geodesicDev.toFixed(4)}
        </span>
        &nbsp;-- NAV_GEO_ENTRY_GATE = {NAV_GEO_ENTRY_GATE}
        {geodesicDev > NAV_GEO_ENTRY_GATE && (
          <span className="text-red-400 ml-2 font-bold">ABOVE GATE -- signal active</span>
        )}
      </div>
    </div>
  )
}

// ---- QuaternionVisualizer ----

const QuaternionVisualizer: React.FC<{ quaternion: [number, number, number, number] }> = ({ quaternion }) => {
  const [w, x, y, z] = quaternion
  const [animAngle, setAnimAngle] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setAnimAngle((a) => a + 0.01), 60)
    return () => clearInterval(id)
  }, [])

  -- Normalize quaternion for display
  const norm = Math.sqrt(w * w + x * x + y * y + z * z) || 1
  const nw = w / norm; const nx = x / norm; const ny = y / norm; const nz = z / norm

  -- Project 3D sphere with perspective
  const R = 55; const cx = 70; const cy = 70
  const perspective = 200

  -- Generate sphere wireframe points
  const sphereLines: React.ReactNode[] = []

  -- Latitude circles
  for (let lat = -60; lat <= 60; lat += 30) {
    const cosLat = Math.cos((lat * Math.PI) / 180)
    const sinLat = Math.sin((lat * Math.PI) / 180)
    const pts: string[] = []
    for (let lon = 0; lon <= 360; lon += 10) {
      const cosLon = Math.cos(((lon + animAngle * 30) * Math.PI) / 180)
      const sinLon = Math.sin(((lon + animAngle * 30) * Math.PI) / 180)
      const px3 = R * cosLat * cosLon
      const py3 = R * cosLat * sinLon
      const pz3 = R * sinLat
      const scale = perspective / (perspective - pz3)
      pts.push(`${cx + px3 * scale},${cy - py3 * scale}`)
    }
    sphereLines.push(
      <polyline key={`lat${lat}`} points={pts.join(' ')} fill="none" stroke="#1f2937" strokeWidth={1} opacity={0.7} />
    )
  }

  -- Longitude meridians
  for (let lon = 0; lon < 180; lon += 30) {
    const pts: string[] = []
    for (let lat = -90; lat <= 90; lat += 5) {
      const cosLat = Math.cos((lat * Math.PI) / 180)
      const sinLat = Math.sin((lat * Math.PI) / 180)
      const cosLon = Math.cos(((lon + animAngle * 30) * Math.PI) / 180)
      const sinLon = Math.sin(((lon + animAngle * 30) * Math.PI) / 180)
      const px3 = R * cosLat * cosLon
      const py3 = R * cosLat * sinLon
      const pz3 = R * sinLat
      const scale = perspective / (perspective - pz3)
      pts.push(`${cx + px3 * scale},${cy - py3 * scale}`)
    }
    sphereLines.push(
      <polyline key={`lon${lon}`} points={pts.join(' ')} fill="none" stroke="#1f2937" strokeWidth={1} opacity={0.7} />
    )
  }

  -- Project quaternion vector onto sphere surface
  const qTheta = 2 * Math.acos(Math.max(-1, Math.min(1, nw)))
  const sinHalf = Math.sin(qTheta / 2) || 1
  const qx3D = nx / sinHalf * R * Math.sin(qTheta)
  const qy3D = ny / sinHalf * R * Math.sin(qTheta)
  const qz3D = nz / sinHalf * R * Math.cos(qTheta)
  const qScale = perspective / (perspective - qz3D)
  const qSx = cx + qx3D * qScale
  const qSy = cy - qy3D * qScale

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Quaternion Unit Sphere
      </h3>
      <div className="flex items-center gap-3">
        <svg width={140} height={140}>
          <circle cx={cx} cy={cy} r={R} fill="#0f172a" stroke="#374151" strokeWidth={1} />
          {sphereLines}
          {/* Axes */}
          <line x1={cx - R - 5} y1={cy} x2={cx + R + 5} y2={cy} stroke="#4b5563" strokeWidth={1} strokeDasharray="2,2" />
          <line x1={cx} y1={cy - R - 5} x2={cx} y2={cy + R + 5} stroke="#4b5563" strokeWidth={1} strokeDasharray="2,2" />
          {/* Quaternion point */}
          <circle cx={qSx} cy={qSy} r={6} fill="#3b82f6" stroke="white" strokeWidth={1.5} />
          <line x1={cx} y1={cy} x2={qSx} y2={qSy} stroke="#3b82f680" strokeWidth={1.5} />
          <text x={4} y={138} fill="#6b7280" fontSize={8}>Rotating: {(animAngle * 30 % 360).toFixed(0)}&deg;</text>
        </svg>
        <div className="text-xs font-mono space-y-1">
          <div><span className="text-gray-500">w =</span> <span className="text-white">{nw.toFixed(4)}</span></div>
          <div><span className="text-gray-500">x =</span> <span className="text-blue-400">{nx.toFixed(4)}</span></div>
          <div><span className="text-gray-500">y =</span> <span className="text-green-400">{ny.toFixed(4)}</span></div>
          <div><span className="text-gray-500">z =</span> <span className="text-purple-400">{nz.toFixed(4)}</span></div>
          <div className="mt-2 text-gray-500">
            |q| = {norm.toFixed(4)}
          </div>
          <div className="text-gray-500">
            theta = {((2 * Math.acos(Math.max(-1, Math.min(1, nw)))) * 180 / Math.PI).toFixed(2)}&deg;
          </div>
        </div>
      </div>
    </div>
  )
}

// ---- SpacetimeFlowField ----

const SpacetimeFlowField: React.FC<{
  bhMass: number
  navCurvature: number
}> = ({ bhMass, navCurvature }) => {
  const [animTick, setAnimTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setAnimTick((t) => t + 1), 100)
    return () => clearInterval(id)
  }, [])

  const GRID = 8
  const CELL = 32
  const PAD = 10
  const W = GRID * CELL + PAD * 2
  const H = GRID * CELL + PAD * 2

  const arrows = useMemo(() => {
    const t = animTick * 0.05
    return Array.from({ length: GRID }, (_, row) =>
      Array.from({ length: GRID }, (_, col) => {
        -- Market flow field: BH mass creates a sink at center
        const nx = (col - GRID / 2 + 0.5) / (GRID / 2)
        const ny = (row - GRID / 2 + 0.5) / (GRID / 2)
        const dist = Math.sqrt(nx * nx + ny * ny) + 0.01
        -- Gravitational pull toward mass center
        const gravX = -nx / dist * bhMass * 0.6
        const gravY = -ny / dist * bhMass * 0.6
        -- Wave component from nav curvature
        const waveX = Math.cos(nx * 3 + t + navCurvature * 0.5) * 0.3
        const waveY = Math.sin(ny * 3 + t) * 0.3
        const vx = gravX + waveX
        const vy = gravY + waveY
        const mag = Math.sqrt(vx * vx + vy * vy)
        return { cx: PAD + col * CELL + CELL / 2, cy: PAD + row * CELL + CELL / 2, vx, vy, mag }
      })
    )
  }, [animTick, bhMass, navCurvature])

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Spacetime Flow Field (8x8)
      </h3>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
        {arrows.flat().map(({ cx, cy, vx, vy, mag }, i) => {
          const len = Math.min(12, mag * 10)
          const nx2 = vx / (mag || 1)
          const ny2 = vy / (mag || 1)
          const ex = cx + nx2 * len
          const ey = cy + ny2 * len
          -- Arrow head
          const hlen = 4
          const angle = Math.atan2(ny2, nx2)
          const ax1 = ex - hlen * Math.cos(angle - 0.4)
          const ay1 = ey - hlen * Math.sin(angle - 0.4)
          const ax2 = ex - hlen * Math.cos(angle + 0.4)
          const ay2 = ey - hlen * Math.sin(angle + 0.4)
          const intensity = Math.min(1, mag)
          const r = Math.round(150 * intensity)
          const g = Math.round(100 - 50 * intensity)
          const b = Math.round(200 - 100 * intensity)
          const color = `rgb(${r},${g},${b})`
          return (
            <g key={i}>
              <line x1={cx} y1={cy} x2={ex} y2={ey} stroke={color} strokeWidth={1.5} opacity={0.8} />
              <polyline points={`${ax1},${ay1} ${ex},${ey} ${ax2},${ay2}`} fill="none" stroke={color} strokeWidth={1.5} opacity={0.8} />
            </g>
          )
        })}
        {/* BH mass center marker */}
        <circle cx={PAD + GRID * CELL / 2} cy={PAD + GRID * CELL / 2} r={6 * bhMass} fill={`rgba(239,68,68,${bhMass * 0.6})`} stroke="#ef4444" strokeWidth={1} />
      </svg>
      <div className="text-xs text-gray-500 mt-1">
        bh_mass={bhMass.toFixed(3)} &nbsp;|&nbsp; nav_curvature={navCurvature.toFixed(3)} &nbsp;|&nbsp;
        <span className="text-purple-400">arrows = velocity + direction</span>
      </div>
    </div>
  )
}

// ---- Main Panel ----

export const BHPhysicsPanel: React.FC<BHPhysicsPanelProps> = ({
  bhMass,
  navCurvature,
  geodesicDev,
  quaternion,
}) => {
  return (
    <div className="bg-gray-950 text-white p-3 rounded-lg">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-bold tracking-wide text-white">
          BH Physics Panel
        </h2>
        <div className="flex items-center gap-4 text-xs">
          <span className="text-gray-500">bh_mass=<span className={clsx('font-mono', bhMass > BH_COLLAPSE ? 'text-red-400' : 'text-green-400')}>{bhMass.toFixed(4)}</span></span>
          <span className="text-gray-500">nav_curv=<span className="text-blue-400 font-mono">{navCurvature.toFixed(4)}</span></span>
          <span className="text-gray-500">geo_dev=<span className={clsx('font-mono', geodesicDev > NAV_GEO_ENTRY_GATE ? 'text-red-400' : 'text-purple-400')}>{geodesicDev.toFixed(4)}</span></span>
        </div>
      </div>

      {/* Row 1: Minkowski + Event Horizon + Geodesic */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <MinkowskiMetricDisplay bhMass={bhMass} navCurvature={navCurvature} />
        <EventHorizonGauge bhMass={bhMass} />
        <GeodesicDeviationPlot geodesicDev={geodesicDev} />
      </div>

      {/* Row 2: Quaternion + Flow Field */}
      <div className="grid grid-cols-2 gap-3">
        <QuaternionVisualizer quaternion={quaternion} />
        <SpacetimeFlowField bhMass={bhMass} navCurvature={navCurvature} />
      </div>
    </div>
  )
}

// ---- Demo wrapper for standalone page usage ----

export const BHPhysicsPanelDemo: React.FC = () => {
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 2000)
    return () => clearInterval(id)
  }, [])

  const bhMass = 0.87 + Math.sin(tick * 0.1) * 0.08
  const navCurvature = 2.4 + Math.cos(tick * 0.07) * 0.8
  const geodesicDev = 2.8 + Math.sin(tick * 0.13) * 1.2
  const q: [number, number, number, number] = [
    Math.cos(tick * 0.05),
    Math.sin(tick * 0.05) * 0.6,
    Math.sin(tick * 0.08) * 0.5,
    Math.sin(tick * 0.04) * 0.4,
  ]

  return (
    <div className="min-h-screen bg-gray-950 p-4">
      <BHPhysicsPanel
        bhMass={bhMass}
        navCurvature={navCurvature}
        geodesicDev={geodesicDev}
        quaternion={q}
      />
    </div>
  )
}

export default BHPhysicsPanel
