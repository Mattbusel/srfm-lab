// ============================================================
// utils/formatters.ts -- Number, date, and color formatting
// utilities for the SRFM dashboard.
// ============================================================

// ---------------------------------------------------------------------------
// Basis points
// ---------------------------------------------------------------------------

/**
 * Format a basis-point value.
 * @example formatBps(12.345) => "12.3 bps"
 */
export function formatBps(bps: number, decimals = 1): string {
  if (!isFinite(bps)) return '--';
  const sign = bps > 0 ? '+' : '';
  return `${sign}${bps.toFixed(decimals)} bps`;
}

/**
 * Format a raw bps number without sign forcing.
 */
export function formatBpsAbs(bps: number, decimals = 1): string {
  if (!isFinite(bps)) return '--';
  return `${bps.toFixed(decimals)} bps`;
}

// ---------------------------------------------------------------------------
// Percentages
// ---------------------------------------------------------------------------

/**
 * Format a fractional value as a percentage string.
 * @param frac -- fractional value, e.g. 0.1234 => "12.34%"
 */
export function formatPct(frac: number, decimals = 2): string {
  if (!isFinite(frac)) return '--';
  return `${(frac * 100).toFixed(decimals)}%`;
}

/**
 * Format an already-multiplied percentage value.
 * @param pct -- e.g. 12.34 => "12.34%"
 */
export function formatPctRaw(pct: number, decimals = 2): string {
  if (!isFinite(pct)) return '--';
  return `${pct.toFixed(decimals)}%`;
}

// ---------------------------------------------------------------------------
// Currency
// ---------------------------------------------------------------------------

/**
 * Format a USD amount with commas and dollar sign.
 * @example formatUSD(1234567) => "$1,234,567"
 */
export function formatUSD(amount: number, decimals = 0): string {
  if (!isFinite(amount)) return '--';
  const abs = Math.abs(amount);
  const sign = amount < 0 ? '-' : '';
  return `${sign}$${abs.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

/**
 * Compact NAV formatting for large dollar amounts.
 * @example formatNav(1_230_000) => "$1.23M"
 * @example formatNav(980_000)   => "$980K"
 */
export function formatNav(nav: number): string {
  if (!isFinite(nav)) return '--';
  const abs = Math.abs(nav);
  const sign = nav < 0 ? '-' : '';
  if (abs >= 1_000_000_000) return `${sign}$${(abs / 1_000_000_000).toFixed(2)}B`;
  if (abs >= 1_000_000)     return `${sign}$${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000)         return `${sign}$${(abs / 1_000).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

/**
 * Format price with appropriate decimal places.
 * High-price assets (>1000) get 2 dp; micro-prices (<0.01) get 6.
 */
export function formatPrice(price: number): string {
  if (!isFinite(price)) return '--';
  if (price >= 1000) return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  if (price >= 1)    return `$${price.toFixed(4)}`;
  return `$${price.toFixed(6)}`;
}

// ---------------------------------------------------------------------------
// Duration
// ---------------------------------------------------------------------------

/**
 * Format a duration in milliseconds into a human-readable string.
 * @example formatDuration(5_000_000) => "1h 23m"
 * @example formatDuration(45_000)    => "45s"
 * @example formatDuration(123)       => "123ms"
 */
export function formatDuration(ms: number): string {
  if (!isFinite(ms) || ms < 0) return '--';
  if (ms < 1_000) return `${Math.round(ms)}ms`;
  const totalSec = Math.round(ms / 1_000);
  if (totalSec < 60) return `${totalSec}s`;
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  if (min < 60) return sec > 0 ? `${min}m ${sec}s` : `${min}m`;
  const hr  = Math.floor(min / 60);
  const rem = min % 60;
  return rem > 0 ? `${hr}h ${rem}m` : `${hr}h`;
}

// ---------------------------------------------------------------------------
// Timestamps
// ---------------------------------------------------------------------------

const DATE_TIME_FMT = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day:   'numeric',
  hour:  '2-digit',
  minute:'2-digit',
  second:'2-digit',
  hour12: false,
});

const TIME_FMT = new Intl.DateTimeFormat('en-US', {
  hour:   '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
});

/**
 * Format an ISO timestamp or Unix epoch (ms) into a localized datetime string.
 */
export function formatTimestamp(ts: string | number): string {
  if (!ts) return '--';
  try {
    const d = typeof ts === 'number' ? new Date(ts) : new Date(ts);
    if (isNaN(d.getTime())) return '--';
    return DATE_TIME_FMT.format(d);
  } catch {
    return '--';
  }
}

/**
 * Format only the time portion of a timestamp.
 */
export function formatTime(ts: string | number): string {
  if (!ts) return '--';
  try {
    const d = typeof ts === 'number' ? new Date(ts) : new Date(ts);
    if (isNaN(d.getTime())) return '--';
    return TIME_FMT.format(d);
  } catch {
    return '--';
  }
}

/**
 * Format an ISO date string to a compact date label.
 * @example "2024-03-15T..." => "Mar 15"
 */
export function formatDate(ts: string): string {
  if (!ts) return '--';
  try {
    const d = new Date(ts);
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return '--';
  }
}

// ---------------------------------------------------------------------------
// Relative time
// ---------------------------------------------------------------------------

/**
 * Express a past timestamp as a human-readable relative time.
 * @example "3m ago", "just now", "2h ago"
 */
export function formatRelative(ts: string | number | Date): string {
  if (!ts) return '--';
  try {
    const d = ts instanceof Date ? ts : new Date(ts);
    const deltaMs = Date.now() - d.getTime();
    if (deltaMs < 5_000)      return 'just now';
    if (deltaMs < 60_000)     return `${Math.floor(deltaMs / 1_000)}s ago`;
    if (deltaMs < 3_600_000)  return `${Math.floor(deltaMs / 60_000)}m ago`;
    if (deltaMs < 86_400_000) return `${Math.floor(deltaMs / 3_600_000)}h ago`;
    return `${Math.floor(deltaMs / 86_400_000)}d ago`;
  } catch {
    return '--';
  }
}

// ---------------------------------------------------------------------------
// Quantity formatting
// ---------------------------------------------------------------------------

/**
 * Format an asset quantity with context-appropriate precision.
 */
export function formatQty(qty: number, symbol?: string): string {
  if (!isFinite(qty)) return '--';
  // For whole-number-ish quantities (equities, small counts)
  if (Math.abs(qty) >= 100 && qty === Math.floor(qty)) return qty.toLocaleString('en-US');
  if (Math.abs(qty) >= 1)   return qty.toFixed(4);
  return qty.toFixed(8);
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

/**
 * Return a CSS color string based on P&L value.
 * Green for positive, red for negative, neutral gray at zero.
 */
export function colorForPnl(pnl: number): string {
  if (pnl > 0)  return '#34d399'; // emerald-400
  if (pnl < 0)  return '#f87171'; // red-400
  return '#94a3b8'; // slate-400
}

/**
 * Return a CSS color string based on P&L as a fraction.
 * Uses magnitude thresholds for color intensity.
 */
export function colorForPnlFrac(frac: number): string {
  if (frac >  0.02) return '#10b981'; // emerald-500
  if (frac >  0)    return '#34d399'; // emerald-400
  if (frac < -0.02) return '#ef4444'; // red-500
  if (frac <  0)    return '#f87171'; // red-400
  return '#94a3b8';
}

/**
 * Return a CSS color string for a composite venue/execution score [0,100].
 * >80 => green, 60-80 => yellow, <60 => red.
 */
export function colorForScore(score: number): string {
  if (score >= 80) return '#34d399'; // emerald-400
  if (score >= 60) return '#fbbf24'; // amber-400
  return '#f87171'; // red-400
}

/**
 * Return a CSS background color for a score badge.
 */
export function bgForScore(score: number): string {
  if (score >= 80) return 'rgba(16, 185, 129, 0.15)';
  if (score >= 60) return 'rgba(245, 158, 11, 0.15)';
  return 'rgba(239, 68, 68, 0.15)';
}

/**
 * Return a CSS color for slippage in bps.
 * Lower slippage (fewer bps) is better.
 */
export function colorForSlippage(bps: number): string {
  if (bps <= 2)  return '#34d399';
  if (bps <= 8)  return '#fbbf24';
  return '#f87171';
}

/**
 * Return a CSS color for a utilization ratio [0,1].
 * <70% green, 70-90% yellow, >90% red.
 */
export function colorForUtilization(utilization: number): string {
  if (utilization < 0.70) return '#34d399';
  if (utilization < 0.90) return '#fbbf24';
  return '#f87171';
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

/**
 * Format a leverage multiplier.
 * @example 1.42 => "1.42x"
 */
export function formatLeverage(lev: number): string {
  if (!isFinite(lev)) return '--';
  return `${lev.toFixed(2)}x`;
}

/**
 * Format a Sharpe or other ratio to 2 dp.
 */
export function formatRatio(r: number): string {
  if (!isFinite(r)) return '--';
  return r.toFixed(2);
}

/**
 * Clamp a number to [min, max].
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
