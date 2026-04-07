// ============================================================
// components/HeatMap.tsx -- Generic 2-D heatmap using SVG.
// Supports symmetric correlation matrices and arbitrary
// rectangular grids. Color scale is interpolated per-value.
// ============================================================

import React, { useMemo, useCallback, useState } from 'react';
import { clsx } from 'clsx';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface HeatMapProps {
  /** Row labels (y axis). */
  rowLabels: string[];
  /** Column labels (x axis). */
  colLabels: string[];
  /** Row-major 2-D value matrix -- rowLabels.length x colLabels.length. */
  values: number[][];
  /** Min value for color scale -- defaults to matrix min. */
  minValue?: number;
  /** Max value for color scale -- defaults to matrix max. */
  maxValue?: number;
  /** Color stops for the scale: [low, mid, high] -- e.g. red/white/green. */
  colorStops?: [string, string, string];
  /** Cell size in px -- default 36. */
  cellSize?: number;
  /** Font size for labels -- default 11. */
  labelFontSize?: number;
  /** Show numeric value inside each cell. */
  showValues?: boolean;
  /** Number of decimal places -- default 2. */
  decimals?: number;
  /** Callback when cell is clicked. */
  onCellClick?: (row: number, col: number, value: number) => void;
  /** Highlight diagonal (correlation matrix). */
  highlightDiagonal?: boolean;
  className?: string;
  title?: string;
}

// ---------------------------------------------------------------------------
// Color interpolation
// ---------------------------------------------------------------------------

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex(r: number, g: number, b: number): string {
  return '#' + [r, g, b].map(v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0')).join('');
}

function lerpColor(c1: string, c2: string, t: number): string {
  const [r1, g1, b1] = hexToRgb(c1);
  const [r2, g2, b2] = hexToRgb(c2);
  return rgbToHex(r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t);
}

function valueToColor(
  v: number,
  min: number,
  max: number,
  stops: [string, string, string]
): string {
  if (max === min) return stops[1];
  const t = (v - min) / (max - min); // 0..1
  if (t <= 0.5) return lerpColor(stops[0], stops[1], t * 2);
  return lerpColor(stops[1], stops[2], (t - 0.5) * 2);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DEFAULT_STOPS: [string, string, string] = ['#ef4444', '#1e293b', '#22c55e'];

export const HeatMap: React.FC<HeatMapProps> = ({
  rowLabels,
  colLabels,
  values,
  minValue,
  maxValue,
  colorStops = DEFAULT_STOPS,
  cellSize = 36,
  labelFontSize = 11,
  showValues = true,
  decimals = 2,
  onCellClick,
  highlightDiagonal = false,
  className,
  title,
}) => {
  const [hovered, setHovered] = useState<[number, number] | null>(null);

  const nRows = rowLabels.length;
  const nCols = colLabels.length;

  const { flatMin, flatMax } = useMemo(() => {
    const flat = values.flat().filter(Number.isFinite);
    return {
      flatMin: minValue ?? Math.min(...flat),
      flatMax: maxValue ?? Math.max(...flat),
    };
  }, [values, minValue, maxValue]);

  const labelOffset = 70; // px for row labels
  const headerOffset = 70; // px for col labels
  const svgWidth  = labelOffset + nCols * cellSize + 4;
  const svgHeight = headerOffset + nRows * cellSize + 4;

  const getCellColor = useCallback(
    (v: number, row: number, col: number) => {
      if (highlightDiagonal && row === col) return '#334155';
      return valueToColor(v, flatMin, flatMax, colorStops);
    },
    [flatMin, flatMax, colorStops, highlightDiagonal]
  );

  const handleMouseEnter = (r: number, c: number) => setHovered([r, c]);
  const handleMouseLeave = () => setHovered(null);
  const handleClick = (r: number, c: number, v: number) => {
    if (onCellClick) onCellClick(r, c, v);
  };

  const isHovered = (r: number, c: number) => hovered?.[0] === r || hovered?.[1] === c;

  return (
    <div className={clsx('overflow-auto', className)}>
      {title && (
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">{title}</p>
      )}
      <svg
        width={svgWidth}
        height={svgHeight}
        style={{ display: 'block', userSelect: 'none' }}
      >
        {/* Column labels */}
        {colLabels.map((label, c) => (
          <text
            key={`col-${c}`}
            x={labelOffset + c * cellSize + cellSize / 2}
            y={headerOffset - 6}
            textAnchor="end"
            fontSize={labelFontSize}
            fill={hovered?.[1] === c ? '#e2e8f0' : '#94a3b8'}
            transform={`rotate(-45, ${labelOffset + c * cellSize + cellSize / 2}, ${headerOffset - 6})`}
            style={{ fontFamily: 'monospace' }}
          >
            {label.length > 8 ? label.slice(0, 7) + '\u2026' : label}
          </text>
        ))}

        {/* Row labels */}
        {rowLabels.map((label, r) => (
          <text
            key={`row-${r}`}
            x={labelOffset - 4}
            y={headerOffset + r * cellSize + cellSize / 2 + 4}
            textAnchor="end"
            fontSize={labelFontSize}
            fill={hovered?.[0] === r ? '#e2e8f0' : '#94a3b8'}
            style={{ fontFamily: 'monospace' }}
          >
            {label.length > 8 ? label.slice(0, 7) + '\u2026' : label}
          </text>
        ))}

        {/* Cells */}
        {values.map((row, r) =>
          row.map((v, c) => {
            const x = labelOffset + c * cellSize;
            const y = headerOffset + r * cellSize;
            const color = getCellColor(v, r, c);
            const isHighlighted = isHovered(r, c);
            const textColor = Math.abs(v) > (flatMax - flatMin) * 0.5 ? '#f8fafc' : '#94a3b8';
            return (
              <g
                key={`cell-${r}-${c}`}
                onMouseEnter={() => handleMouseEnter(r, c)}
                onMouseLeave={handleMouseLeave}
                onClick={() => handleClick(r, c, v)}
                style={{ cursor: onCellClick ? 'pointer' : 'default' }}
              >
                <rect
                  x={x + 1}
                  y={y + 1}
                  width={cellSize - 2}
                  height={cellSize - 2}
                  fill={color}
                  opacity={isHighlighted ? 1 : 0.85}
                  rx={2}
                />
                {isHighlighted && (
                  <rect
                    x={x + 1}
                    y={y + 1}
                    width={cellSize - 2}
                    height={cellSize - 2}
                    fill="none"
                    stroke="#e2e8f0"
                    strokeWidth={1}
                    rx={2}
                  />
                )}
                {showValues && cellSize >= 28 && (
                  <text
                    x={x + cellSize / 2}
                    y={y + cellSize / 2 + 4}
                    textAnchor="middle"
                    fontSize={labelFontSize - 1}
                    fill={textColor}
                    style={{ fontFamily: 'monospace', pointerEvents: 'none' }}
                  >
                    {v.toFixed(decimals)}
                  </text>
                )}
              </g>
            );
          })
        )}
      </svg>

      {/* Color scale legend */}
      <div className="flex items-center gap-2 mt-2">
        <span className="text-xs text-slate-500">{flatMin.toFixed(2)}</span>
        <div
          className="flex-1 h-2 rounded"
          style={{
            background: `linear-gradient(to right, ${colorStops[0]}, ${colorStops[1]}, ${colorStops[2]})`,
          }}
        />
        <span className="text-xs text-slate-500">{flatMax.toFixed(2)}</span>
      </div>
    </div>
  );
};
