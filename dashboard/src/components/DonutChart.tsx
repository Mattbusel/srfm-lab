// ============================================================
// components/DonutChart.tsx -- Pure SVG donut chart.
// No canvas, no external chart library. Supports hover
// tooltips (CSS-based), legend, and total in center.
// ============================================================

import React, { useState, useId } from 'react';
import { clsx } from 'clsx';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DonutSlice {
  label: string;
  value: number;
  color: string;
}

export interface DonutChartProps {
  data: DonutSlice[];
  /** Outer diameter in px. Defaults to 200. */
  size?: number;
  /** Inner radius as a fraction of the outer radius. Defaults to 0.55. */
  innerRadius?: number;
  /** Format the total displayed in the center. Defaults to locale string. */
  centerFormatter?: (total: number) => string;
  /** Optional label above the total. */
  centerLabel?: string;
  /** Show legend below the chart. Defaults to true. */
  showLegend?: boolean;
  /** Show percentage labels in slices. Defaults to false. */
  showPercentLabels?: boolean;
  className?: string;
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/** Convert polar (r, theta) to Cartesian, centered at (cx, cy). */
function polar(cx: number, cy: number, r: number, angleDeg: number): [number, number] {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}

/** Build an SVG arc path for a donut slice. */
function arcPath(
  cx: number,
  cy: number,
  outerR: number,
  innerR: number,
  startDeg: number,
  endDeg: number,
): string {
  // Clamp sweep to avoid degenerate full-circle arcs
  const sweep = Math.min(endDeg - startDeg, 359.9999);
  const largeArc = sweep > 180 ? 1 : 0;

  const [ox1, oy1] = polar(cx, cy, outerR, startDeg);
  const [ox2, oy2] = polar(cx, cy, outerR, startDeg + sweep);
  const [ix1, iy1] = polar(cx, cy, innerR, startDeg + sweep);
  const [ix2, iy2] = polar(cx, cy, innerR, startDeg);

  return [
    `M ${ox1} ${oy1}`,
    `A ${outerR} ${outerR} 0 ${largeArc} 1 ${ox2} ${oy2}`,
    `L ${ix1} ${iy1}`,
    `A ${innerR} ${innerR} 0 ${largeArc} 0 ${ix2} ${iy2}`,
    'Z',
  ].join(' ');
}

// ---------------------------------------------------------------------------
// Tooltip (CSS-positioned, no portal needed)
// ---------------------------------------------------------------------------

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  label: string;
  value: number;
  pct: string;
  color: string;
}

const TOOLTIP_INIT: TooltipState = {
  visible: false, x: 0, y: 0, label: '', value: 0, pct: '', color: '',
};

// ---------------------------------------------------------------------------
// DonutChart
// ---------------------------------------------------------------------------

export const DonutChart: React.FC<DonutChartProps> = ({
  data,
  size = 200,
  innerRadius = 0.55,
  centerFormatter,
  centerLabel,
  showLegend = true,
  showPercentLabels = false,
  className,
}) => {
  const uid = useId();
  const [tooltip, setTooltip] = useState<TooltipState>(TOOLTIP_INIT);
  const [hovered, setHovered] = useState<number | null>(null);

  // Filter zero-value slices
  const slices = data.filter(d => d.value > 0);
  const total  = slices.reduce((sum, d) => sum + d.value, 0);

  const cx = size / 2;
  const cy = size / 2;
  const outerR = (size / 2) * 0.88;
  const innerR = outerR * innerRadius;
  const gap    = 1.2; // degrees gap between slices

  const centerText = centerFormatter
    ? centerFormatter(total)
    : total >= 1_000_000
    ? `$${(total / 1_000_000).toFixed(2)}M`
    : total >= 1_000
    ? `$${(total / 1_000).toFixed(1)}K`
    : total.toLocaleString('en-US', { maximumFractionDigits: 2 });

  // Build slices with start/end angles
  type SliceArc = DonutSlice & { startDeg: number; endDeg: number; midDeg: number; pct: number };
  let cursor = 0;
  const arcs: SliceArc[] = slices.map((s, i) => {
    const frac   = s.value / total;
    const sweep  = frac * 360 - (slices.length > 1 ? gap : 0);
    const start  = cursor + gap / 2;
    const end    = start + sweep;
    const mid    = start + sweep / 2;
    cursor += frac * 360;
    return { ...s, startDeg: start, endDeg: end, midDeg: mid, pct: frac * 100, index: i } as SliceArc & { index: number };
  });

  const handleMouseMove = (
    e: React.MouseEvent<SVGPathElement>,
    arc: SliceArc,
  ) => {
    const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement)
      .getBoundingClientRect();
    setTooltip({
      visible: true,
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      label: arc.label,
      value: arc.value,
      pct:   arc.pct.toFixed(1),
      color: arc.color,
    });
  };

  const handleMouseLeave = () => {
    setTooltip(TOOLTIP_INIT);
    setHovered(null);
  };

  return (
    <div className={clsx('flex flex-col items-center', className)}>
      {/* SVG chart */}
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          style={{ overflow: 'visible' }}
        >
          <defs>
            {arcs.map((arc, i) => (
              <filter key={`${uid}-drop-${i}`} id={`${uid}-drop-${i}`} x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="1" stdDeviation="3" floodColor={arc.color} floodOpacity="0.4" />
              </filter>
            ))}
          </defs>

          {/* Empty state ring */}
          {arcs.length === 0 && (
            <circle
              cx={cx} cy={cy}
              r={(outerR + innerR) / 2}
              fill="none"
              stroke="#1e2130"
              strokeWidth={outerR - innerR}
            />
          )}

          {arcs.map((arc, i) => {
            const isHov = hovered === i;
            const expandedR = isHov ? outerR * 1.06 : outerR;
            const path = arcPath(cx, cy, expandedR, innerR, arc.startDeg, arc.endDeg);
            return (
              <path
                key={`${uid}-arc-${i}`}
                d={path}
                fill={arc.color}
                fillOpacity={isHov ? 1 : 0.85}
                stroke="#0d1017"
                strokeWidth={1.5}
                filter={isHov ? `url(#${uid}-drop-${i})` : undefined}
                style={{ transition: 'all 0.15s ease', cursor: 'pointer' }}
                onMouseMove={(e) => { setHovered(i); handleMouseMove(e, arc); }}
                onMouseLeave={handleMouseLeave}
              />
            );
          })}

          {/* Percentage labels in slices */}
          {showPercentLabels && arcs.map((arc, i) => {
            if (arc.pct < 5) return null; // skip tiny slices
            const labelR = (outerR + innerR) / 2;
            const [lx, ly] = polar(cx, cy, labelR, arc.midDeg);
            return (
              <text
                key={`${uid}-label-${i}`}
                x={lx} y={ly}
                textAnchor="middle"
                dominantBaseline="central"
                fill="white"
                fontSize={size * 0.06}
                fontFamily="monospace"
                fontWeight="600"
                style={{ pointerEvents: 'none' }}
              >
                {arc.pct.toFixed(0)}%
              </text>
            );
          })}

          {/* Center total */}
          <text
            x={cx} y={centerLabel ? cy - size * 0.05 : cy}
            textAnchor="middle"
            dominantBaseline="central"
            fill="#e2e8f0"
            fontSize={size * 0.12}
            fontFamily="monospace"
            fontWeight="700"
            style={{ pointerEvents: 'none' }}
          >
            {centerText}
          </text>
          {centerLabel && (
            <text
              x={cx} y={cy + size * 0.1}
              textAnchor="middle"
              dominantBaseline="central"
              fill="#64748b"
              fontSize={size * 0.07}
              fontFamily="monospace"
              style={{ pointerEvents: 'none' }}
            >
              {centerLabel}
            </text>
          )}
        </svg>

        {/* Hover tooltip */}
        {tooltip.visible && (
          <div
            className="absolute z-20 pointer-events-none"
            style={{
              left:      tooltip.x + 10,
              top:       tooltip.y - 10,
              transform: 'translateY(-100%)',
            }}
          >
            <div className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 shadow-xl text-xs font-mono whitespace-nowrap">
              <div className="flex items-center gap-2 mb-1">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: tooltip.color }}
                />
                <span className="text-slate-200 font-semibold">{tooltip.label}</span>
              </div>
              <div className="text-slate-400">
                {tooltip.value.toLocaleString('en-US', { maximumFractionDigits: 2 })}{' '}
                <span className="text-slate-500">({tooltip.pct}%)</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      {showLegend && arcs.length > 0 && (
        <div className="mt-4 flex flex-wrap justify-center gap-x-4 gap-y-2 max-w-xs">
          {arcs.map((arc, i) => (
            <div
              key={`${uid}-legend-${i}`}
              className="flex items-center gap-1.5 cursor-pointer"
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
            >
              <span
                className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                style={{ backgroundColor: arc.color, opacity: hovered === i ? 1 : 0.8 }}
              />
              <span className={clsx(
                'text-[10px] font-mono transition-colors',
                hovered === i ? 'text-slate-200' : 'text-slate-500',
              )}>
                {arc.label}
              </span>
              <span className="text-[10px] font-mono text-slate-600">
                {arc.pct.toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DonutChart;
