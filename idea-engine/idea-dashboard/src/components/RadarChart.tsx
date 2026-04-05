import React, { useMemo } from 'react'
import {
  RadarChart as RechartsRadar,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { GenomeParams } from '../types'

interface RadarChartProps {
  params: GenomeParams
  color?: string
  compareParams?: GenomeParams
  compareColor?: string
  height?: number
}

// Normalization bounds for each param
const PARAM_BOUNDS: Record<keyof GenomeParams, [number, number]> = {
  lookback:      [5,  100],
  threshold:     [0,  1],
  stopLoss:      [0.005, 0.1],
  takeProfit:    [0.01, 0.2],
  positionSize:  [0.01, 0.5],
  rsiPeriod:     [5, 50],
  macdFast:      [3, 20],
  macdSlow:      [10, 50],
  atrMultiplier: [0.5, 5],
  volFilter:     [0, 5],
}

const PARAM_LABELS: Record<keyof GenomeParams, string> = {
  lookback:      'Lookback',
  threshold:     'Threshold',
  stopLoss:      'Stop Loss',
  takeProfit:    'Take Profit',
  positionSize:  'Pos. Size',
  rsiPeriod:     'RSI Period',
  macdFast:      'MACD Fast',
  macdSlow:      'MACD Slow',
  atrMultiplier: 'ATR Mult',
  volFilter:     'Vol Filter',
}

function normalize(value: number, min: number, max: number): number {
  return Math.max(0, Math.min(1, (value - min) / (max - min)))
}

interface CustomTooltipProps {
  active?: boolean
  payload?: Array<{ payload: { param: string; raw: number; value: number } }>
}

const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="tooltip-custom">
      <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>
        {d.param}
      </div>
      <div style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: '0.8125rem' }}>
        {typeof d.raw === 'number' ? d.raw.toFixed(4) : d.raw}
      </div>
      <div style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
        Normalized: {(d.value * 100).toFixed(0)}%
      </div>
    </div>
  )
}

const RadarChartComponent: React.FC<RadarChartProps> = ({
  params,
  color = 'var(--accent)',
  compareParams,
  compareColor = 'var(--yellow)',
  height = 220,
}) => {
  const data = useMemo(() => {
    return (Object.keys(PARAM_BOUNDS) as Array<keyof GenomeParams>).map((key) => {
      const [min, max] = PARAM_BOUNDS[key]
      const val = normalize(params[key], min, max)
      return {
        param: PARAM_LABELS[key],
        value: parseFloat((val * 100).toFixed(1)),
        raw: params[key],
        compareValue: compareParams
          ? parseFloat((normalize(compareParams[key], min, max) * 100).toFixed(1))
          : undefined,
      }
    })
  }, [params, compareParams])

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsRadar data={data} cx="50%" cy="50%" outerRadius="70%">
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis
          dataKey="param"
          tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
        />
        <PolarRadiusAxis
          angle={30}
          domain={[0, 100]}
          tick={{ fill: 'var(--text-muted)', fontSize: 9 }}
          tickCount={4}
        />
        <Radar
          name="Genome"
          dataKey="value"
          stroke={color}
          fill={color}
          fillOpacity={0.15}
          strokeWidth={2}
        />
        {compareParams && (
          <Radar
            name="Compare"
            dataKey="compareValue"
            stroke={compareColor}
            fill={compareColor}
            fillOpacity={0.1}
            strokeWidth={1.5}
            strokeDasharray="4 2"
          />
        )}
        <Tooltip content={<CustomTooltip />} />
      </RechartsRadar>
    </ResponsiveContainer>
  )
}

export default RadarChartComponent
