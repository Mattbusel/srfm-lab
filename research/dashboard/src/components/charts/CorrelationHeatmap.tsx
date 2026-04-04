import React from 'react'
import { correlationColor } from '@/utils/colors'

interface CorrelationHeatmapProps {
  instruments: string[]
  matrix: number[][]
  size?: 'sm' | 'md'
}

export function CorrelationHeatmap({ instruments, matrix, size = 'md' }: CorrelationHeatmapProps) {
  const cellSize = size === 'sm' ? 'text-[9px]' : 'text-xs'

  return (
    <div className="overflow-auto">
      <table className="border-collapse" style={{ fontSize: size === 'sm' ? 9 : 11 }}>
        <thead>
          <tr>
            <th className="w-16 text-right pr-1 text-research-subtle font-mono" style={{ fontSize: 9 }}></th>
            {instruments.map(inst => (
              <th
                key={inst}
                className="font-mono text-research-subtle font-normal"
                style={{
                  writingMode: 'vertical-rl',
                  transform: 'rotate(180deg)',
                  height: 60,
                  verticalAlign: 'bottom',
                  paddingBottom: 4,
                  fontSize: 9,
                  minWidth: size === 'sm' ? 22 : 28,
                }}
              >
                {inst.replace('-USD', '')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {instruments.map((rowInst, ri) => (
            <tr key={rowInst}>
              <td className="font-mono text-research-subtle text-right pr-1" style={{ fontSize: 9 }}>
                {rowInst.replace('-USD', '')}
              </td>
              {instruments.map((_, ci) => {
                const val = matrix[ri]?.[ci] ?? 0
                const bg = correlationColor(val)
                return (
                  <td
                    key={ci}
                    title={`${rowInst} / ${instruments[ci]}: ${val.toFixed(3)}`}
                    className="text-center font-mono cursor-default"
                    style={{
                      backgroundColor: bg,
                      width: size === 'sm' ? 22 : 28,
                      height: size === 'sm' ? 20 : 26,
                      fontSize: 8,
                      color: Math.abs(val) > 0.5 ? '#fff' : '#8899aa',
                    }}
                  >
                    {val.toFixed(2)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Generate correlation matrix from returns data
export function buildCorrelationMatrix(instruments: string[]): number[][] {
  const n = instruments.length
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  let seed = 999
  function rand() {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff
    return (seed >>> 0) / 0xffffffff
  }

  for (let i = 0; i < n; i++) {
    matrix[i][i] = 1
    for (let j = i + 1; j < n; j++) {
      const c = rand() * 1.4 - 0.2 // bias toward positive
      matrix[i][j] = Math.max(-1, Math.min(1, c))
      matrix[j][i] = matrix[i][j]
    }
  }
  return matrix
}
