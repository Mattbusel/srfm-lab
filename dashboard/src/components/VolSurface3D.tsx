// ============================================================
// VolSurface3D.tsx — 3D volatility surface using react-three-fiber
// ============================================================
import React, { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import * as THREE from 'three'

// ---- Helpers ----

function ivColor(iv: number, minIv: number, maxIv: number): string {
  const t = (iv - minIv) / (maxIv - minIv)
  // Gradient: blue (low) -> green -> yellow -> red (high)
  if (t < 0.33) {
    const u = t / 0.33
    return `rgb(${Math.round(59 + u * (34 - 59))},${Math.round(130 + u * (197 - 130))},${Math.round(246 + u * (94 - 246))})`
  } else if (t < 0.66) {
    const u = (t - 0.33) / 0.33
    return `rgb(${Math.round(34 + u * (245 - 34))},${Math.round(197 + u * (158 - 197))},${Math.round(94 + u * (11 - 94))})`
  } else {
    const u = (t - 0.66) / 0.34
    return `rgb(${Math.round(245 + u * (239 - 245))},${Math.round(158 + u * (68 - 158))},${Math.round(11 + u * (68 - 11))})`
  }
}

function hexColor(iv: number, minIv: number, maxIv: number): number {
  const t = (iv - minIv) / (maxIv - minIv)
  const r = Math.min(255, Math.round(59 + t * (239 - 59)))
  const g = Math.min(255, Math.round(130 - t * 62))
  const b = Math.min(255, Math.round(246 - t * 178))
  return (r << 16) | (g << 8) | b
}

// ---- Surface geometry ----

interface SurfaceData {
  strikes: number[]
  expiries: number[]    // days to expiry
  ivGrid: number[][]   // [strikeIdx][expiryIdx]
}

function generateMockSurface(): SurfaceData {
  const strikes = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
  const expiries = [7, 14, 30, 60, 90, 120, 180, 360]

  const ivGrid = strikes.map((k) =>
    expiries.map((dte) => {
      const moneyness = Math.log(k)
      // Skew + smile + term structure
      const smile = 0.4 * moneyness ** 2 + 0.01 * moneyness
      const skew = -0.08 * moneyness
      const termStructure = 0.35 * Math.exp(-dte / 120) + 0.20 * (1 - Math.exp(-dte / 120))
      const noise = (Math.random() - 0.5) * 0.005
      return Math.max(0.05, termStructure + smile + skew + noise)
    }),
  )

  return { strikes, expiries, ivGrid }
}

// ---- React-three-fiber surface mesh ----

const VolSurfaceMesh: React.FC<{ data: SurfaceData }> = ({ data }) => {
  const meshRef = useRef<THREE.Mesh>(null)

  const { geometry, colors } = useMemo(() => {
    const { strikes, expiries, ivGrid } = data
    const nS = strikes.length
    const nE = expiries.length

    const flatIV = ivGrid.flat()
    const minIv = Math.min(...flatIV)
    const maxIv = Math.max(...flatIV)

    const positions: number[] = []
    const colorsArr: number[] = []
    const indices: number[] = []

    for (let s = 0; s < nS; s++) {
      for (let e = 0; e < nE; e++) {
        const x = (s / (nS - 1)) * 4 - 2
        const z = (e / (nE - 1)) * 4 - 2
        const y = (ivGrid[s][e] - minIv) / (maxIv - minIv) * 2 - 0.5

        positions.push(x, y, z)

        const c = new THREE.Color(hexColor(ivGrid[s][e], minIv, maxIv))
        colorsArr.push(c.r, c.g, c.b)
      }
    }

    for (let s = 0; s < nS - 1; s++) {
      for (let e = 0; e < nE - 1; e++) {
        const a = s * nE + e
        const b = s * nE + e + 1
        const c = (s + 1) * nE + e
        const d = (s + 1) * nE + e + 1
        indices.push(a, b, c, b, d, c)
      }
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colorsArr, 3))
    geo.setIndex(indices)
    geo.computeVertexNormals()

    return { geometry: geo, colors: colorsArr }
  }, [data])

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.001
    }
  })

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} />
    </mesh>
  )
}

// ---- Main component ----

interface VolSurface3DProps {
  data?: SurfaceData
  height?: number
  className?: string
}

export const VolSurface3D: React.FC<VolSurface3DProps> = ({
  data,
  height = 320,
  className,
}) => {
  const surfaceData = useMemo(() => data ?? generateMockSurface(), [data])

  void ivColor  // keep reference to avoid unused warning

  return (
    <div className={className} style={{ height, background: '#0a0b0e', borderRadius: 8, overflow: 'hidden' }}>
      <Canvas
        camera={{ position: [4, 3, 4], fov: 45 }}
        gl={{ antialias: true }}
        style={{ background: '#0a0b0e' }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <VolSurfaceMesh data={surfaceData} />
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={3}
          maxDistance={12}
          autoRotate
          autoRotateSpeed={0.5}
        />
        {/* Axis labels */}
        <Text position={[-2.5, -0.7, 0]} fontSize={0.2} color="#475569" anchorX="center">
          Strike
        </Text>
        <Text position={[0, -0.7, 2.5]} fontSize={0.2} color="#475569" anchorX="center">
          Expiry
        </Text>
        <Text position={[-2.5, 0.8, -2.5]} fontSize={0.2} color="#475569" anchorX="center">
          IV
        </Text>
      </Canvas>
    </div>
  )
}
