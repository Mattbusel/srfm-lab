// ============================================================
// components/GaugeChart.tsx -- Semicircular SVG gauge with
// colored arc bands. Used for VPIN, utilization metrics,
// and any 0..1 or 0..100 reading.
// ============================================================

import React, { useMemo } from 'react';
import { clsx } from 'clsx';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GaugeBand {
  /** Start of the band (same unit as min/max). */
  from: number;
  /** End of the band. */
  to: number;
  /** Tailwind or hex color. */
  color: string;
  label?: string;
}

export interface GaugeChartProps {
  /** Current value to display. */
  value: number;
  /** Minimum of the scale -- default 0. */
  min?: number;
  /** Maximum of the scale -- default 1. */
  max?: number;
  /** Colored arc bands, drawn from min to max. */
  bands?: GaugeBand[];
  /** Displayed unit suffix -- e.g. "%" or "". */
  unit?: string;
  /** Label below the value. */
  label?: string;
  /** Sub-label -- e.g. "percentile: 82". */
  subLabel?: string;
  /** Size in px (total width = height * 2 for semicircle). Default 180. */
  size?: number;
  /** Stroke width of the arc -- default 18. */
  strokeWidth?: number;
  /** Whether to animate the needle. */
  animate?: boolean;
  className?: string;
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

const DEG_START = 180; // left
const DEG_END   = 0;   // right

function norm(v: number, min: number, max: number): number {
  return Math.max(0, Math.min(1, (v - min) / (max - min)));
}

/** Convert [0,1] normalized value to SVG arc angle in degrees (180 = left, 0 = right). */
function normToAngle(t: number): number {
  return DEG_START - t * DEG_START; // 180 -> 0
}

/** Polar to cartesian (cx, cy = centre, r = radius, angle in degrees). */
function polar(cx: number, cy: number, r: number, angleDeg: number): [number, number] {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}

/** SVG arc path between two angles. */
function arcPath(cx: number, cy: number, r: number, startDeg: number, endDeg: number): string {
  const [sx, sy] = polar(cx, cy, r, startDeg + 90);
  const [ex, ey] = polar(cx, cy, r, endDeg + 90);
  const large = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  const sweep = endDeg > startDeg ? 1 : 0;
  return `M ${sx} ${sy} A ${r} ${r} 0 ${large} ${sweep} ${ex} ${ey}`;
}

// ---------------------------------------------------------------------------
// Default bands
// ---------------------------------------------------------------------------

const DEFAULT_BANDS: GaugeBand[] = [
  { from: 0,    to: 0.33, color: '#22c55e', label: 'Low'    },
  { from: 0.33, to: 0.66, color: '#f59e0b', label: 'Medium' },
  { from: 0.66, to: 1,    color: '#ef4444', label: 'High'   },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const GaugeChart: React.FC<GaugeChartProps> = ({
  value,
  min = 0,
  max = 1,
  bands = DEFAULT_BANDS,
  unit = '',
  label,
  subLabel,
  size = 180,
  strokeWidth = 18,
  animate = true,
  className,
}) => {
  const cx   = size / 2;
  const cy   = size / 2;
  const r    = size / 2 - strokeWidth - 4;
  const t    = norm(value, min, max);
  const needleAngle = normToAngle(t); // 180 (left) to 0 (right)

  // SVG arc angles use bottom as 0 so we offset
  // We draw the semicircle from 180deg to 360deg (bottom-left to bottom-right via top)
  // Map: t=0 -> 180deg (left), t=1 -> 0deg (right)
  const bandArcs = useMemo(() =>
    bands.map(band => {
      const startT = norm(band.from, min, max);
      const endT   = norm(band.to,   min, max);
      // Convert to SVG angles for our semicircle layout
      // t=0 -> angle 180, t=1 -> angle 0 (going clockwise when viewed normally)
      // In SVG polar: 0deg = top, 90deg = right, 180deg = bottom, 270deg = left
      // We want left=0, right=1 along top half
      const startAngle = 180 * (1 - startT); // 180 -> 0
      const endAngle   = 180 * (1 - endT);
      // draw arc from startAngle to endAngle in SVG polar space
      // polar() adds 90 so: 0deg(top) maps to input -90, we want leftmost at 180
      // Recalculate directly:
      const toSVGArc = (angle: number) => {
        const rad = (angle * Math.PI) / 180;
        return [cx - r * Math.cos(rad), cy - r * Math.sin(rad)] as [number, number];
      };
      const [sx, sy] = toSVGArc(startT === 0 ? 180 : 180 * (1 - startT));
      const [ex, ey] = toSVGArc(endT   === 1 ? 0   : 180 * (1 - endT));
      const sweepAngle = (endT - startT) * 180;
      const large = sweepAngle > 180 ? 1 : 0;
      // Always sweep left to right (clockwise for top half = sweep=0 in standard orientation)
      return { band, sx, sy, ex, ey, r, large, path: `M ${sx} ${sy} A ${r} ${r} 0 ${large} 0 ${ex} ${ey}` };
    }),
    [bands, min, max, r, cx, cy]
  );

  // Needle
  const needleRad    = (needleAngle * Math.PI) / 180;
  const needleTipX   = cx - (r - 2) * Math.cos(needleRad);
  const needleTipY   = cy - (r - 2) * Math.sin(needleRad);
  const needleBaseL  = [cx - 6 * Math.sin(needleRad), cy - 6 * Math.cos(needleRad)] as const;
  const needleBaseR  = [cx + 6 * Math.sin(needleRad), cy + 6 * Math.cos(needleRad)] as const;

  // Active band color
  const activeBand = bands.find(b => value >= b.from && value <= b.to) ?? bands[bands.length - 1];
  const valueColor = activeBand?.color ?? '#f59e0b';

  const displayValue = typeof value === 'number'
    ? (Math.abs(value) < 10 ? value.toFixed(3) : value.toFixed(1))
    : String(value);

  return (
    <div className={clsx('flex flex-col items-center', className)}>
      <svg
        width={size}
        height={size / 2 + 20}
        viewBox={`0 0 ${size} ${size / 2 + 20}`}
        overflow="visible"
      >
        {/* Background arc (track) */}
        <path
          d={`M ${strokeWidth + 4} ${cy} A ${r} ${r} 0 0 1 ${size - strokeWidth - 4} ${cy}`}
          fill="none"
          stroke="#1e293b"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Colored bands */}
        {bandArcs.map(({ band, path }, i) => (
          <path
            key={i}
            d={path}
            fill="none"
            stroke={band.color}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            opacity={0.7}
          />
        ))}

        {/* Active value arc overlay */}
        <path
          d={`M ${strokeWidth + 4} ${cy} A ${r} ${r} 0 0 1 ${needleTipX} ${needleTipY}`}
          fill="none"
          stroke={valueColor}
          strokeWidth={strokeWidth - 6}
          strokeLinecap="round"
          opacity={0.3}
        />

        {/* Needle */}
        <polygon
          points={`${needleTipX},${needleTipY} ${needleBaseL[0]},${needleBaseL[1]} ${needleBaseR[0]},${needleBaseR[1]}`}
          fill={valueColor}
          opacity={0.95}
        />
        <circle cx={cx} cy={cy} r={6} fill="#1e293b" stroke={valueColor} strokeWidth={2} />

        {/* Min / max labels */}
        <text x={strokeWidth + 4} y={cy + 16} textAnchor="middle" fontSize={9} fill="#475569">
          {min}
        </text>
        <text x={size - strokeWidth - 4} y={cy + 16} textAnchor="middle" fontSize={9} fill="#475569">
          {max}
        </text>
      </svg>

      {/* Value display */}
      <div className="flex flex-col items-center -mt-2">
        <span className="text-2xl font-bold tabular-nums" style={{ color: valueColor }}>
          {displayValue}
          {unit && <span className="text-sm font-normal ml-1 text-slate-400">{unit}</span>}
        </span>
        {label && (
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mt-0.5">
            {label}
          </span>
        )}
        {subLabel && (
          <span className="text-xs text-slate-500 mt-0.5">{subLabel}</span>
        )}
      </div>
    </div>
  );
};
