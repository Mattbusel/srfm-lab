// ============================================================
// components/DrawdownChart.tsx -- Standalone drawdown chart
// with underwater area shading, duration markers, and
// recovery annotations.
// ============================================================

import React, { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from 'recharts';
import { clsx } from 'clsx';
import { format, parseISO } from 'date-fns';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DrawdownPoint {
  date: string;              // ISO date string
  drawdown: number;          // negative decimal e.g. -0.12
  equity?: number;
}

export interface DrawdownPeriod {
  start: string;
  end: string;
  depth: number;             // most negative value
  duration_bars: number;
}

export interface DrawdownChartProps {
  data: DrawdownPoint[];
  height?: number;
  /** Highlight worst drawdown periods. */
  highlightPeriods?: DrawdownPeriod[];
  /** Draw reference line at this threshold (e.g. -0.10 for -10%). */
  alertThreshold?: number;
  title?: string;
  color?: string;
  /** Number of decimal places for y-axis labels. */
  decimals?: number;
  className?: string;
  compact?: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function computeDrawdownPeriods(data: DrawdownPoint[], minDepth = -0.05): DrawdownPeriod[] {
  const periods: DrawdownPeriod[] = [];
  let inDD = false;
  let start = '';
  let depth = 0;
  let barCount = 0;

  for (const d of data) {
    if (d.drawdown < 0) {
      if (!inDD) {
        inDD = true;
        start = d.date;
        depth = d.drawdown;
        barCount = 1;
      } else {
        depth = Math.min(depth, d.drawdown);
        barCount++;
      }
    } else {
      if (inDD) {
        if (depth <= minDepth) {
          periods.push({ start, end: d.date, depth, duration_bars: barCount });
        }
        inDD = false;
        depth = 0;
        barCount = 0;
      }
    }
  }
  if (inDD && depth <= minDepth && data.length) {
    periods.push({ start, end: data[data.length - 1].date, depth, duration_bars: barCount });
  }
  return periods;
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

function DDTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const dd = payload[0]?.value;
  let dateStr = label;
  try { dateStr = format(parseISO(label), 'MMM d, yyyy'); } catch { /* keep */ }
  return (
    <div className="bg-slate-900 border border-slate-700 rounded p-2 text-xs">
      <p className="text-slate-400 mb-1">{dateStr}</p>
      <p className={clsx('font-mono', dd < -0.1 ? 'text-red-400' : dd < -0.05 ? 'text-amber-400' : 'text-slate-300')}>
        {(dd * 100).toFixed(2)}%
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const DrawdownChart: React.FC<DrawdownChartProps> = ({
  data,
  height = 120,
  highlightPeriods,
  alertThreshold = -0.10,
  title,
  color = '#ef4444',
  decimals = 1,
  className,
  compact = false,
}) => {
  const periods = useMemo(
    () => highlightPeriods ?? computeDrawdownPeriods(data, Math.abs(alertThreshold) * 0.6),
    [data, highlightPeriods, alertThreshold]
  );

  const worstDD = useMemo(
    () => data.reduce((m, d) => Math.min(m, d.drawdown), 0),
    [data]
  );

  const stats = useMemo(() => {
    const active = data[data.length - 1]?.drawdown ?? 0;
    const inDD   = active < -0.001;
    const ddBars = inDD
      ? data.slice().reverse().findIndex(d => d.drawdown >= -0.001)
      : 0;
    return { active, inDD, ddBars };
  }, [data]);

  return (
    <div className={clsx('flex flex-col', className)}>
      {(title || !compact) && (
        <div className="flex items-center justify-between mb-1">
          {title && (
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</p>
          )}
          <div className="flex items-center gap-3 text-[11px] font-mono">
            <span className="text-red-400">Max: {(worstDD * 100).toFixed(decimals)}%</span>
            {stats.inDD && (
              <span className="text-amber-400">
                Current: {(stats.active * 100).toFixed(decimals)}% ({stats.ddBars}d)
              </span>
            )}
          </div>
        </div>
      )}

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: compact ? 28 : 36 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={v => { try { return format(parseISO(v), compact ? 'MMM' : 'MMM d'); } catch { return v; } }}
            tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
            minTickGap={compact ? 40 : 50}
          />
          <YAxis
            tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={v => `${(v * 100).toFixed(0)}%`}
            domain={[worstDD * 1.15, 0.005]}
            width={compact ? 28 : 36}
          />
          <Tooltip content={<DDTooltip />} cursor={{ stroke: '#334155' }} />

          {/* Shaded drawdown periods */}
          {periods.slice(0, 5).map((p, i) => (
            <ReferenceArea
              key={`dd-period-${i}`}
              x1={p.start}
              x2={p.end}
              fill={color}
              fillOpacity={0.08}
              strokeOpacity={0}
            />
          ))}

          {/* Alert threshold line */}
          <ReferenceLine
            y={alertThreshold}
            stroke={color}
            strokeDasharray="4 4"
            strokeWidth={1}
            opacity={0.6}
            label={{ value: `${(alertThreshold * 100).toFixed(0)}%`, fill: color, fontSize: 9, position: 'right' }}
          />

          <ReferenceLine y={0} stroke="#334155" strokeWidth={1} />

          <defs>
            <linearGradient id="dd-gradient" x1="0" y1="1" x2="0" y2="0">
              <stop offset="5%"  stopColor={color} stopOpacity={0.35} />
              <stop offset="95%" stopColor={color} stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke={color}
            strokeWidth={1.5}
            fill="url(#dd-gradient)"
            dot={false}
            isAnimationActive={false}
            baseLine={0}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
