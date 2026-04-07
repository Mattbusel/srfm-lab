// ============================================================
// components/SparkLine.tsx -- Compact inline sparkline chart.
// Used in metric cards, table rows, and summary panels.
// No axes, no tooltips -- pure visual trend indicator.
// ============================================================

import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { clsx } from 'clsx';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SparkDataPoint {
  value: number;
  timestamp?: string;
}

export interface SparkLineProps {
  data: SparkDataPoint[];
  /** Width in pixels -- defaults to 100% of container. */
  width?: number | string;
  /** Height in pixels -- default 40. */
  height?: number;
  /** Color override -- auto-selected from last vs first value if omitted. */
  color?: string;
  /** Show reference line at zero. */
  showZero?: boolean;
  /** Show a minimal tooltip on hover. */
  showTooltip?: boolean;
  /** Stroke width -- default 1.5. */
  strokeWidth?: number;
  /** Whether higher values are better (affects color direction). */
  higherIsBetter?: boolean;
  className?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const POSITIVE_COLOR = '#22c55e';  // green-500
const NEGATIVE_COLOR = '#ef4444';  // red-500
const NEUTRAL_COLOR  = '#94a3b8';  // slate-400

function trendColor(data: SparkDataPoint[], higherIsBetter: boolean): string {
  if (data.length < 2) return NEUTRAL_COLOR;
  const first = data[0].value;
  const last  = data[data.length - 1].value;
  const diff  = last - first;
  if (Math.abs(diff) < 1e-9) return NEUTRAL_COLOR;
  const isGood = higherIsBetter ? diff > 0 : diff < 0;
  return isGood ? POSITIVE_COLOR : NEGATIVE_COLOR;
}

// ---------------------------------------------------------------------------
// Custom minimal tooltip
// ---------------------------------------------------------------------------

function SparkTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const v = payload[0]?.value;
  if (v == null) return null;
  return (
    <div className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 pointer-events-none">
      {typeof v === 'number' ? v.toFixed(4) : v}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const SparkLine: React.FC<SparkLineProps> = ({
  data,
  width = '100%',
  height = 40,
  color,
  showZero = false,
  showTooltip = false,
  strokeWidth = 1.5,
  higherIsBetter = true,
  className,
}) => {
  const chartData = useMemo(
    () => data.map(d => ({ value: d.value, ts: d.timestamp })),
    [data]
  );

  const strokeColor = color ?? trendColor(data, higherIsBetter);

  if (!data.length) {
    return (
      <div
        className={clsx('flex items-center justify-center text-slate-600 text-xs', className)}
        style={{ width, height }}
      >
        --
      </div>
    );
  }

  return (
    <div className={clsx('inline-block', className)} style={{ width, height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          {showZero && (
            <ReferenceLine y={0} stroke="#475569" strokeDasharray="2 2" strokeWidth={1} />
          )}
          {showTooltip && <Tooltip content={<SparkTooltip />} />}
          <Line
            type="monotone"
            dataKey="value"
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Variant: area sparkline with subtle fill
// ---------------------------------------------------------------------------

import { AreaChart, Area } from 'recharts';

export interface SparkAreaProps extends SparkLineProps {
  fillOpacity?: number;
}

export const SparkArea: React.FC<SparkAreaProps> = ({
  data,
  width = '100%',
  height = 40,
  color,
  showZero = false,
  showTooltip = false,
  strokeWidth = 1.5,
  higherIsBetter = true,
  fillOpacity = 0.15,
  className,
}) => {
  const chartData = useMemo(
    () => data.map(d => ({ value: d.value, ts: d.timestamp })),
    [data]
  );
  const strokeColor = color ?? trendColor(data, higherIsBetter);

  if (!data.length) return null;

  return (
    <div className={clsx('inline-block', className)} style={{ width, height }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          {showZero && (
            <ReferenceLine y={0} stroke="#475569" strokeDasharray="2 2" strokeWidth={1} />
          )}
          {showTooltip && <Tooltip content={<SparkTooltip />} />}
          <defs>
            <linearGradient id={`spark-fill-${strokeColor.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={strokeColor} stopOpacity={fillOpacity * 2} />
              <stop offset="95%" stopColor={strokeColor} stopOpacity={0} />
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="value"
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            fill={`url(#spark-fill-${strokeColor.replace('#', '')})`}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
