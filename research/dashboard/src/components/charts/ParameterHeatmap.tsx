import React from 'react'
import { heatmapColor } from '@/utils/colors'

export interface ParameterHeatmapData {
  param1Values: number[]
  param2Values: number[]
  param1Label: string
  param2Label: string
  values: number[][]  // [param1_idx][param2_idx] -> metric value
  metricLabel: string
  minVal?: number
  maxVal?: number
}

interface ParameterHeatmapProps {
  data: ParameterHeatmapData
  size?: 'sm' | 'md'
}

export function ParameterHeatmap({ data, size = 'md' }: ParameterHeatmapProps) {
  const { param1Values, param2Values, param1Label, param2Label, values, metricLabel } = data
  const allVals = values.flat()
  const minVal = data.minVal ?? Math.min(...allVals)
  const maxVal = data.maxVal ?? Math.max(...allVals)
  const cellH = size === 'sm' ? 22 : 30
  const cellW = size === 'sm' ? 36 : 48

  return (
    <div className="overflow-auto">
      <div className="text-xs text-research-subtle mb-2 font-mono">
        {param1Label} vs {param2Label} — {metricLabel}
      </div>
      <table className="border-collapse">
        <thead>
          <tr>
            <th className="text-research-subtle font-mono" style={{ fontSize: 9, width: 40, textAlign: 'right', paddingRight: 4 }}>
              {param1Label}↓ {param2Label}→
            </th>
            {param2Values.map(v => (
              <th key={v} className="text-research-subtle font-mono font-normal text-center" style={{ fontSize: 9, width: cellW }}>
                {v}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {param1Values.map((p1, ri) => (
            <tr key={p1}>
              <td className="text-research-subtle font-mono text-right pr-1" style={{ fontSize: 9 }}>{p1}</td>
              {param2Values.map((_, ci) => {
                const val = values[ri]?.[ci] ?? 0
                const bg = heatmapColor(val, minVal, maxVal)
                const textColor = Math.abs((val - minVal) / (maxVal - minVal) - 0.5) > 0.2 ? '#fff' : '#8899aa'
                return (
                  <td
                    key={ci}
                    title={`${param1Label}=${p1}, ${param2Label}=${param2Values[ci]}: ${val.toFixed(3)}`}
                    className="text-center font-mono cursor-default"
                    style={{
                      backgroundColor: bg,
                      height: cellH,
                      width: cellW,
                      fontSize: 8,
                      color: textColor,
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

export function generateMockParameterHeatmap(): ParameterHeatmapData {
  const param1Values = [5, 10, 20, 40, 60]
  const param2Values = [0.1, 0.2, 0.5, 1.0, 2.0]
  let seed = 42
  function rand() {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff
    return (seed >>> 0) / 0xffffffff
  }
  function randn() {
    return Math.sqrt(-2 * Math.log(rand())) * Math.cos(2 * Math.PI * rand())
  }

  const values = param1Values.map(() => param2Values.map(() => 0.5 + randn() * 0.6))

  return {
    param1Values,
    param2Values,
    param1Label: 'Lookback',
    param2Label: 'Threshold',
    values,
    metricLabel: 'Sharpe',
  }
}
