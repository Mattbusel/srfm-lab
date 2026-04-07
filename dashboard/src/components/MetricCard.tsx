// ============================================================
// components/MetricCard.tsx -- KPI card with value, delta,
// sparkline, and optional threshold progress bar.
// ============================================================

import React from 'react';
import { clsx } from 'clsx';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { SparkArea } from './SparkLine';
import type { SparkDataPoint } from './SparkLine';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MetricFormat = 'currency' | 'percent' | 'ratio' | 'count' | 'bps' | 'raw';

export interface MetricCardProps {
  label: string;
  value: number | string;
  /** Change relative to previous period -- decimal (0.05 = +5%). */
  change?: number;
  /** Unit suffix displayed after the value. */
  unit?: string;
  /** Format determines how the value is rendered. */
  format?: MetricFormat;
  /** Sparkline data -- last N periods. */
  sparkline?: SparkDataPoint[];
  /** If true, higher values are considered better for color coding. */
  higherIsBetter?: boolean;
  /** Threshold bar: current utilization (0..1). */
  utilization?: number;
  /** Threshold label. */
  thresholdLabel?: string;
  /** Warn at this utilization -- default 0.75. */
  warnAt?: number;
  /** Critical at this utilization -- default 0.90. */
  criticalAt?: number;
  /** Sub-label beneath main value. */
  subValue?: string;
  /** Icon component. */
  icon?: React.ReactNode;
  /** Border color variant. */
  variant?: 'default' | 'warn' | 'critical' | 'positive';
  className?: string;
  onClick?: () => void;
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

function formatValue(value: number | string, format: MetricFormat, unit?: string): string {
  if (typeof value === 'string') return value;
  switch (format) {
    case 'currency':
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: value >= 10_000 ? 0 : 2,
        notation: value >= 1_000_000 ? 'compact' : 'standard',
      }).format(value);
    case 'percent':
      return `${(value * 100).toFixed(2)}%`;
    case 'ratio':
      return value.toFixed(3);
    case 'bps':
      return `${(value * 10_000).toFixed(1)} bps`;
    case 'count':
      return new Intl.NumberFormat('en-US').format(Math.round(value));
    default:
      return unit ? `${value.toFixed(2)} ${unit}` : value.toFixed(4);
  }
}

function formatChange(change: number): string {
  const sign = change >= 0 ? '+' : '';
  return `${sign}${(change * 100).toFixed(2)}%`;
}

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

function UtilBar({ util, warnAt, critAt }: { util: number; warnAt: number; critAt: number }) {
  const color = util >= critAt ? 'bg-red-500'
    : util >= warnAt ? 'bg-amber-500'
    : 'bg-emerald-500';
  return (
    <div className="mt-2 space-y-0.5">
      <div className="flex justify-between text-[10px] text-slate-500">
        <span>utilization</span>
        <span className={clsx(
          util >= critAt ? 'text-red-400' : util >= warnAt ? 'text-amber-400' : 'text-emerald-400'
        )}>
          {(util * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={clsx('h-full rounded-full transition-all duration-500', color)}
          style={{ width: `${Math.min(100, util * 100)}%` }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Variant border
// ---------------------------------------------------------------------------

const VARIANT_BORDER: Record<NonNullable<MetricCardProps['variant']>, string> = {
  default:  'border-slate-700/50',
  warn:     'border-amber-500/40',
  critical: 'border-red-500/60',
  positive: 'border-emerald-500/40',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  change,
  unit,
  format = 'raw',
  sparkline,
  higherIsBetter = true,
  utilization,
  thresholdLabel,
  warnAt = 0.75,
  criticalAt = 0.90,
  subValue,
  icon,
  variant = 'default',
  className,
  onClick,
}) => {
  const changeIsGood = change != null
    ? (higherIsBetter ? change >= 0 : change <= 0)
    : null;

  const changeColor = changeIsGood == null ? 'text-slate-500'
    : changeIsGood ? 'text-emerald-400'
    : 'text-red-400';

  const ChangeIcon = change == null ? Minus
    : change > 0 ? TrendingUp
    : change < 0 ? TrendingDown
    : Minus;

  const displayValue = typeof value === 'number'
    ? formatValue(value, format, unit)
    : value;

  return (
    <div
      className={clsx(
        'bg-slate-800/60 border rounded-lg p-4 flex flex-col gap-1',
        'backdrop-blur-sm transition-all duration-200',
        VARIANT_BORDER[variant],
        onClick && 'cursor-pointer hover:bg-slate-800/80 hover:border-slate-600',
        className
      )}
      onClick={onClick}
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          {icon && <span className="text-slate-400 flex-shrink-0">{icon}</span>}
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider truncate">
            {label}
          </span>
        </div>
        {sparkline && sparkline.length > 0 && (
          <SparkArea
            data={sparkline}
            width={64}
            height={28}
            higherIsBetter={higherIsBetter}
            showZero={false}
          />
        )}
      </div>

      {/* Main value */}
      <div className="flex items-end justify-between gap-2 mt-0.5">
        <div className="min-w-0">
          <span className="text-2xl font-bold tabular-nums text-slate-100 leading-none">
            {displayValue}
          </span>
          {subValue && (
            <span className="block text-xs text-slate-500 mt-0.5">{subValue}</span>
          )}
        </div>

        {change != null && (
          <div className={clsx('flex items-center gap-1 flex-shrink-0', changeColor)}>
            <ChangeIcon size={13} />
            <span className="text-xs font-semibold tabular-nums">
              {formatChange(change)}
            </span>
          </div>
        )}
      </div>

      {/* Threshold utilization bar */}
      {utilization != null && (
        <UtilBar util={utilization} warnAt={warnAt} critAt={criticalAt} />
      )}
      {thresholdLabel && (
        <span className="text-[10px] text-slate-600">{thresholdLabel}</span>
      )}
    </div>
  );
};
