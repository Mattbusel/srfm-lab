// ============================================================
// components/TimeSeriesChart.tsx -- Multi-series time series
// chart with configurable axes, reference bands, regime
// shading, and an interactive tooltip. Built on Recharts.
// ============================================================

import React, { useMemo, useState } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
  Brush,
} from 'recharts';
import { clsx } from 'clsx';
import { format, parseISO } from 'date-fns';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SeriesType = 'line' | 'area' | 'bar';
export type YAxisSide = 'left' | 'right';

export interface SeriesConfig {
  /** dataKey within each data point object. */
  key: string;
  label: string;
  color: string;
  type?: SeriesType;
  yAxisId?: YAxisSide;
  strokeWidth?: number;
  strokeDasharray?: string;
  fillOpacity?: number;
  dot?: boolean;
  hidden?: boolean;
}

export interface ReferenceLineConfig {
  y: number;
  yAxisId?: YAxisSide;
  color: string;
  label?: string;
  dashArray?: string;
}

export interface ReferenceBand {
  y1: number;
  y2: number;
  yAxisId?: YAxisSide;
  color: string;
  opacity?: number;
  label?: string;
}

export interface TimeSeriesChartProps {
  data: Record<string, any>[];
  series: SeriesConfig[];
  /** dataKey for the x axis (timestamp or date string). */
  xKey?: string;
  /** Date format string for x-axis labels. */
  xFormat?: string;
  /** Custom x-axis tick formatter. */
  xFormatter?: (value: any) => string;
  leftLabel?: string;
  rightLabel?: string;
  leftDomain?: [number | 'auto' | 'dataMin' | 'dataMax', number | 'auto' | 'dataMin' | 'dataMax'];
  rightDomain?: [number | 'auto' | 'dataMin' | 'dataMax', number | 'auto' | 'dataMin' | 'dataMax'];
  referenceLines?: ReferenceLineConfig[];
  referenceBands?: ReferenceBand[];
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  showBrush?: boolean;
  showTooltip?: boolean;
  /** Compact mode -- smaller fonts and padding. */
  compact?: boolean;
  title?: string;
  subtitle?: string;
  className?: string;
  /** Tooltip value formatter. */
  tooltipFormatter?: (value: number, key: string) => string;
}

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------

function CustomTooltip({
  active,
  payload,
  label,
  series,
  formatter,
  xFormat,
}: any) {
  if (!active || !payload?.length) return null;

  let displayLabel = label;
  try {
    displayLabel = format(parseISO(label), xFormat ?? 'MMM d HH:mm');
  } catch { /* keep raw */ }

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 shadow-xl text-xs min-w-[140px]">
      <p className="text-slate-400 font-medium mb-2">{displayLabel}</p>
      {payload.map((entry: any) => {
        const cfg = series.find((s: SeriesConfig) => s.key === entry.dataKey);
        const label2 = cfg?.label ?? entry.dataKey;
        const val = formatter
          ? formatter(entry.value, entry.dataKey)
          : typeof entry.value === 'number'
          ? entry.value.toLocaleString(undefined, { maximumFractionDigits: 4 })
          : entry.value;
        return (
          <div key={entry.dataKey} className="flex justify-between gap-4">
            <span style={{ color: entry.color }} className="font-medium">{label2}</span>
            <span className="tabular-nums text-slate-200">{val}</span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  data,
  series,
  xKey = 'timestamp',
  xFormat = 'MMM d',
  xFormatter,
  leftLabel,
  rightLabel,
  leftDomain,
  rightDomain,
  referenceLines = [],
  referenceBands = [],
  height = 260,
  showGrid = true,
  showLegend = false,
  showBrush = false,
  showTooltip = true,
  compact = false,
  title,
  subtitle,
  className,
  tooltipFormatter,
}) => {
  const [hiddenSeries, setHiddenSeries] = useState<Set<string>>(
    new Set(series.filter(s => s.hidden).map(s => s.key))
  );

  const tickFormatter = xFormatter ?? ((v: string) => {
    try { return format(parseISO(v), xFormat!); }
    catch { return v; }
  });

  const hasRightAxis = series.some(s => s.yAxisId === 'right');

  const toggleSeries = (key: string) => {
    setHiddenSeries(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const margin = compact
    ? { top: 4, right: 8, bottom: 4, left: 4 }
    : { top: 8, right: hasRightAxis ? 40 : 8, bottom: showBrush ? 32 : 8, left: 8 };

  return (
    <div className={clsx('flex flex-col', className)}>
      {(title || subtitle) && (
        <div className="mb-2">
          {title && <p className="text-xs font-semibold text-slate-300 uppercase tracking-wider">{title}</p>}
          {subtitle && <p className="text-[11px] text-slate-500">{subtitle}</p>}
        </div>
      )}

      {/* Custom legend (so we can toggle) */}
      {showLegend && (
        <div className="flex flex-wrap gap-3 mb-2">
          {series.map(s => (
            <button
              key={s.key}
              onClick={() => toggleSeries(s.key)}
              className={clsx(
                'flex items-center gap-1.5 text-xs transition-opacity',
                hiddenSeries.has(s.key) ? 'opacity-30' : 'opacity-100'
              )}
            >
              <span
                className="inline-block w-4 h-0.5 rounded"
                style={{ backgroundColor: s.color, borderRadius: 2 }}
              />
              <span className="text-slate-400">{s.label}</span>
            </button>
          ))}
        </div>
      )}

      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={margin}>
          {showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          )}

          <XAxis
            dataKey={xKey}
            tickFormatter={tickFormatter}
            tick={{ fill: '#64748b', fontSize: compact ? 9 : 10, fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
            minTickGap={compact ? 30 : 50}
          />

          <YAxis
            yAxisId="left"
            tick={{ fill: '#64748b', fontSize: compact ? 9 : 10, fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
            domain={leftDomain}
            label={leftLabel ? {
              value: leftLabel, angle: -90, position: 'insideLeft',
              fill: '#475569', fontSize: 9,
            } : undefined}
            width={compact ? 36 : 52}
          />

          {hasRightAxis && (
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fill: '#64748b', fontSize: compact ? 9 : 10, fontFamily: 'monospace' }}
              tickLine={false}
              axisLine={false}
              domain={rightDomain}
              label={rightLabel ? {
                value: rightLabel, angle: 90, position: 'insideRight',
                fill: '#475569', fontSize: 9,
              } : undefined}
              width={compact ? 36 : 52}
            />
          )}

          {showTooltip && (
            <Tooltip
              content={
                <CustomTooltip
                  series={series}
                  formatter={tooltipFormatter}
                  xFormat={xFormat}
                />
              }
              cursor={{ stroke: '#334155', strokeWidth: 1, strokeDasharray: '3 3' }}
            />
          )}

          {/* Reference bands */}
          {referenceBands.map((band, i) => (
            <ReferenceArea
              key={`band-${i}`}
              yAxisId={band.yAxisId ?? 'left'}
              y1={band.y1}
              y2={band.y2}
              fill={band.color}
              fillOpacity={band.opacity ?? 0.1}
              strokeOpacity={0}
            />
          ))}

          {/* Reference lines */}
          {referenceLines.map((rl, i) => (
            <ReferenceLine
              key={`rl-${i}`}
              yAxisId={rl.yAxisId ?? 'left'}
              y={rl.y}
              stroke={rl.color}
              strokeDasharray={rl.dashArray ?? '4 4'}
              strokeWidth={1}
              label={rl.label ? {
                value: rl.label, fill: rl.color, fontSize: 9, position: 'right',
              } : undefined}
            />
          ))}

          {/* Series */}
          {series.map(s => {
            if (hiddenSeries.has(s.key)) return null;
            const yId = s.yAxisId ?? 'left';
            const sw = s.strokeWidth ?? 1.5;
            if (s.type === 'bar') {
              return (
                <Bar
                  key={s.key}
                  dataKey={s.key}
                  yAxisId={yId}
                  fill={s.color}
                  opacity={0.6}
                  isAnimationActive={false}
                />
              );
            }
            if (s.type === 'area') {
              return (
                <Area
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  yAxisId={yId}
                  stroke={s.color}
                  strokeWidth={sw}
                  fill={s.color}
                  fillOpacity={s.fillOpacity ?? 0.12}
                  dot={false}
                  strokeDasharray={s.strokeDasharray}
                  isAnimationActive={false}
                />
              );
            }
            return (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                yAxisId={yId}
                stroke={s.color}
                strokeWidth={sw}
                dot={s.dot ?? false}
                strokeDasharray={s.strokeDasharray}
                isAnimationActive={false}
                connectNulls
              />
            );
          })}

          {showBrush && (
            <Brush
              dataKey={xKey}
              height={20}
              stroke="#334155"
              fill="#0f172a"
              travellerWidth={6}
              tickFormatter={tickFormatter}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};
