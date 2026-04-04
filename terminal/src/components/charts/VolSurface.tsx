// ============================================================
// VolSurface.tsx — Interactive 3D volatility surface
// ============================================================
import React, { useRef, useMemo, useState } from 'react'
import { Canvas, useFrame, ThreeElements } from '@react-three/fiber'
import { OrbitControls, Text, Grid } from '@react-three/drei'
import * as THREE from 'three'
import { clsx } from 'clsx'

// ---- Surface data ----

export interface VolSurfaceData {
  strikes: number[]       // relative moneyness or absolute strikes
  expiries: number[]      // days to expiry
  ivGrid: number[][]      // [strikeIdx][expiryIdx] implied vol (0-1, not percent)
  spotPrice?: number
}

function generateDefaultSurface(): VolSurfaceData {
  const strikes = [0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15, 1.2, 1.3]
  const expiries = [7, 14, 21, 30, 45, 60, 90, 120, 150, 180, 240, 360]

  const ivGrid = strikes.map((k) =>
    expiries.map((dte) => {
      const m = Math.log(k)
      const smile = 0.45 * m ** 2
      const skew  = -0.12 * m
      const atm   = 0.34 + 0.06 * Math.exp(-dte / 90) + 0.04 * Math.exp(-dte / 30)
      const noise = (Math.random() - 0.5) * 0.003
      return Math.max(0.08, atm + smile + skew + noise)
    }),
  )

  return { strikes, expiries, ivGrid }
}

// ---- Surface mesh component ----

interface SurfaceMeshProps {
  data: VolSurfaceData
  colorMode: 'gradient' | 'monochrome'
  showGrid: boolean
  rotating: boolean
}

function ivToColor(iv: number, minIv: number, maxIv: number, mode: 'gradient' | 'monochrome'): THREE.Color {
  const t = maxIv === minIv ? 0.5 : (iv - minIv) / (maxIv - minIv)
  if (mode === 'monochrome') {
    return new THREE.Color(0.1 + t * 0.5, 0.1 + t * 0.5, 0.4 + t * 0.5)
  }
  // Blue (low IV) → Cyan → Green → Yellow → Red (high IV)
  if (t < 0.25) {
    const u = t / 0.25
    return new THREE.Color(0.0, u * 0.5, 1.0 - u * 0.5)
  } else if (t < 0.5) {
    const u = (t - 0.25) / 0.25
    return new THREE.Color(0.0, 0.5 + u * 0.5, 0.5 - u * 0.5)
  } else if (t < 0.75) {
    const u = (t - 0.5) / 0.25
    return new THREE.Color(u, 1.0, 0.0)
  } else {
    const u = (t - 0.75) / 0.25
    return new THREE.Color(1.0, 1.0 - u * 0.5, 0.0)
  }
}

const SurfaceMesh: React.FC<SurfaceMeshProps> = ({ data, colorMode, rotating }) => {
  const groupRef = useRef<THREE.Group>(null)

  const { geometry } = useMemo(() => {
    const { strikes, expiries, ivGrid } = data
    const nS = strikes.length
    const nE = expiries.length

    const flatIV = ivGrid.flat()
    const minIv = Math.min(...flatIV)
    const maxIv = Math.max(...flatIV)

    const positions: number[] = []
    const colors: number[] = []
    const normals: number[] = []
    const uvs: number[] = []
    const indices: number[] = []

    for (let s = 0; s < nS; s++) {
      for (let e = 0; e < nE; e++) {
        const x = (s / (nS - 1)) * 4 - 2
        const z = (e / (nE - 1)) * 4 - 2
        const iv = ivGrid[s][e]
        const y = ((iv - minIv) / (maxIv - minIv)) * 2.5 - 0.5

        positions.push(x, y, z)
        normals.push(0, 1, 0)
        uvs.push(s / (nS - 1), e / (nE - 1))

        const c = ivToColor(iv, minIv, maxIv, colorMode)
        colors.push(c.r, c.g, c.b)
      }
    }

    for (let s = 0; s < nS - 1; s++) {
      for (let e = 0; e < nE - 1; e++) {
        const a = s * nE + e
        const b = s * nE + e + 1
        const c = (s + 1) * nE + e
        const d = (s + 1) * nE + e + 1
        indices.push(a, c, b, b, c, d)
      }
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3))
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
    geo.setIndex(indices)
    geo.computeVertexNormals()

    return { geometry: geo }
  }, [data, colorMode])

  useFrame((_, delta) => {
    if (rotating && groupRef.current) {
      groupRef.current.rotation.y += delta * 0.4
    }
  })

  return (
    <group ref={groupRef}>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          roughness={0.6}
          metalness={0.1}
          transparent
          opacity={0.92}
        />
      </mesh>
      {/* Wire overlay */}
      <mesh geometry={geometry}>
        <meshBasicMaterial
          vertexColors={false}
          color="#1e2130"
          wireframe
          opacity={0.15}
          transparent
        />
      </mesh>
    </group>
  )
}

// ---- Axis labels ----

const AxisLabels: React.FC<{ data: VolSurfaceData }> = ({ data }) => {
  const { strikes, expiries } = data
  const nS = strikes.length
  const nE = expiries.length

  return (
    <group>
      {/* Strike axis labels (x) */}
      {[0, Math.floor(nS / 4), Math.floor(nS / 2), Math.floor(3 * nS / 4), nS - 1].map((si) => {
        const x = (si / (nS - 1)) * 4 - 2
        return (
          <Text key={si} position={[x, -0.8, 2.4]} fontSize={0.14} color="#475569" anchorX="center">
            {strikes[si].toFixed(2)}
          </Text>
        )
      })}

      {/* Expiry axis labels (z) */}
      {[0, Math.floor(nE / 3), Math.floor(2 * nE / 3), nE - 1].map((ei) => {
        const z = (ei / (nE - 1)) * 4 - 2
        return (
          <Text key={ei} position={[-2.4, -0.8, z]} fontSize={0.14} color="#475569" anchorX="center">
            {expiries[ei]}d
          </Text>
        )
      })}

      {/* Axis titles */}
      <Text position={[0, -1.1, 2.8]} fontSize={0.16} color="#64748b" anchorX="center">
        Strike (moneyness)
      </Text>
      <Text position={[-2.8, -1.1, 0]} fontSize={0.16} color="#64748b" anchorX="center" rotation={[0, Math.PI / 2, 0]}>
        DTE
      </Text>
      <Text position={[-2.5, 1.2, -2.5]} fontSize={0.16} color="#64748b" anchorX="center">
        IV
      </Text>
    </group>
  )
}

// ---- IV color legend ----

const IVLegend: React.FC<{ minIv: number; maxIv: number }> = ({ minIv, maxIv }) => (
  <div className="absolute right-3 top-3 flex flex-col items-end gap-1">
    <span className="text-[9px] font-mono text-slate-500">IV</span>
    <div
      style={{
        width: 12,
        height: 80,
        background: 'linear-gradient(to bottom, #ef4444, #fbbf24, #22c55e, #06b6d4, #3b82f6)',
        borderRadius: 4,
      }}
    />
    <span className="text-[9px] font-mono text-red-400">{(maxIv * 100).toFixed(0)}%</span>
    <span className="text-[9px] font-mono text-blue-400 mt-8">{(minIv * 100).toFixed(0)}%</span>
  </div>
)

// ---- Main component ----

interface VolSurfaceProps {
  data?: VolSurfaceData
  height?: number
  className?: string
}

export const VolSurface: React.FC<VolSurfaceProps> = ({
  data,
  height = 380,
  className,
}) => {
  const surfaceData = useMemo(() => data ?? generateDefaultSurface(), [data])
  const [colorMode, setColorMode] = useState<'gradient' | 'monochrome'>('gradient')
  const [showGrid, setShowGrid] = useState(false)
  const [rotating, setRotating] = useState(true)

  const flatIV = surfaceData.ivGrid.flat()
  const minIv = Math.min(...flatIV)
  const maxIv = Math.max(...flatIV)

  return (
    <div
      className={clsx('relative rounded-lg overflow-hidden bg-[#0a0b0e] border border-[#1e2130]', className)}
      style={{ height }}
    >
      <Canvas camera={{ position: [4.5, 3, 4.5], fov: 42 }} gl={{ antialias: true }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[5, 8, 5]} intensity={0.9} castShadow />
        <pointLight position={[-5, 5, -5]} intensity={0.4} color="#3b82f6" />

        <SurfaceMesh data={surfaceData} colorMode={colorMode} showGrid={showGrid} rotating={rotating} />
        <AxisLabels data={surfaceData} />

        {showGrid && <Grid args={[8, 8]} cellSize={0.5} cellThickness={0.3} cellColor="#1e2130" sectionColor="#2e3550" position={[0, -0.6, 0]} />}

        <OrbitControls
          enableDamping
          dampingFactor={0.06}
          minDistance={3}
          maxDistance={14}
          minPolarAngle={0.2}
          maxPolarAngle={Math.PI / 2}
        />
      </Canvas>

      {/* Controls overlay */}
      <div className="absolute top-3 left-3 flex items-center gap-2">
        {[
          { label: 'Color', value: colorMode, options: ['gradient', 'mono'], onClick: () => setColorMode((p) => p === 'gradient' ? 'monochrome' : 'gradient') },
        ].map((ctrl) => (
          <button
            key={ctrl.label}
            onClick={ctrl.onClick}
            className="text-[9px] font-mono text-slate-500 border border-[#1e2130] bg-[#0a0b0e]/70 rounded px-1.5 py-0.5 hover:text-slate-300 hover:border-slate-600 transition-colors"
          >
            {ctrl.label}: {ctrl.value}
          </button>
        ))}
        <button
          onClick={() => setShowGrid((p) => !p)}
          className={clsx(
            'text-[9px] font-mono border rounded px-1.5 py-0.5 transition-colors',
            showGrid ? 'text-blue-400 border-blue-700/50 bg-blue-950/30' : 'text-slate-500 border-[#1e2130] bg-[#0a0b0e]/70 hover:text-slate-300',
          )}
        >
          Grid
        </button>
        <button
          onClick={() => setRotating((p) => !p)}
          className={clsx(
            'text-[9px] font-mono border rounded px-1.5 py-0.5 transition-colors',
            rotating ? 'text-emerald-400 border-emerald-700/50 bg-emerald-950/30' : 'text-slate-500 border-[#1e2130] bg-[#0a0b0e]/70 hover:text-slate-300',
          )}
        >
          {rotating ? 'Pause' : 'Rotate'}
        </button>
      </div>

      <IVLegend minIv={minIv} maxIv={maxIv} />
    </div>
  )
}
