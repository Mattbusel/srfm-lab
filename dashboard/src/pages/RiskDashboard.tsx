// ============================================================
// pages/RiskDashboard.tsx -- Live risk monitoring dashboard.
//
// Sections:
//   1. Portfolio summary KPI bar
//   2. VaR panel -- parametric / historical / MC with trend sparklines
//   3. Per-instrument positions table (delta, P&L, VaR contribution)
//   4. Greeks aggregate table for options
//   5. Correlation heatmap
//   6. Circuit breaker + API health status
//   7. VaR breach alert panel
//   8. Limit utilization bars
//
// Data from FastAPI risk service at :8791 via usePortfolioRisk hook.
// Falls back to mock data when API is unreachable.
// ============================================================

import React, { useState, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import { clsx } from 'clsx';
import {
  RefreshCw, Shield, AlertTriangle, Activity, Zap,
  CheckCircle2, XCircle, MinusCircle,
  ChevronUp, ChevronDown,
} from 'lucide-react';
import { format, parseISO } from 'date-fns';

import { MetricCard }       from '../components/MetricCard';
import { HeatMap }          from '../components/HeatMap';
import { AlertPanel }       from '../components/AlertPanel';
import { TimeSeriesChart }  from '../components/TimeSeriesChart';

import {
  usePortfolioRisk,
  useRefreshRisk,
  useVaRAlerts,
} from '../hooks/useRiskAPI';

import type {
  PositionRow,
  GreeksSummaryRow,
  BrokerCircuitBreaker,
  LimitRow,
  CircuitBreakerState,
} from '../types/risk';

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

const fmt$ = (v: number) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(v);

const fmtPct = (v: number, dp = 2) => `${(v * 100).toFixed(dp)}%`;

function pnlColor(v: number) {
  return v > 0 ? 'text-emerald-400' : v < 0 ? 'text-red-400' : 'text-slate-400';
}

function limitBarColor(util: number) {
  return util >= 0.90 ? 'bg-red-500' : util >= 0.75 ? 'bg-amber-500' : 'bg-emerald-500';
}

function limitTextColor(util: number) {
  return util >= 0.90 ? 'text-red-400' : util >= 0.75 ? 'text-amber-400' : 'text-emerald-400';
}

// ---------------------------------------------------------------------------
// VaR method badge
// ---------------------------------------------------------------------------

function VaRBadge({ method, varValue, cvarValue, label }: {
  method: string; varValue: number; cvarValue: number; label: string;
}) {
  const colors: Record<string, string> = {
    parametric:  'border-blue-500/40 bg-blue-500/10 text-blue-300',
    historical:  'border-purple-500/40 bg-purple-500/10 text-purple-300',
    monte_carlo: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
  };
  const cls = colors[method] ?? 'border-slate-700 bg-slate-800 text-slate-300';
  return (
    <div className={clsx('flex flex-col rounded-lg border px-4 py-3 gap-1 min-w-[130px]', cls)}>
      <span className="text-[10px] font-bold uppercase tracking-wider opacity-70">{label}</span>
      <span className="text-xl font-bold tabular-nums">{fmt$(varValue)}</span>
      <span className="text-[10px] opacity-50 font-mono">CVaR: {fmt$(cvarValue)}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Circuit breaker row
// ---------------------------------------------------------------------------

const STATE_ICON: Record<CircuitBreakerState, React.ReactNode> = {
  CLOSED:    <CheckCircle2 size={14} className="text-emerald-400" />,
  HALF_OPEN: <MinusCircle  size={14} className="text-amber-400" />,
  OPEN:      <XCircle      size={14} className="text-red-400" />,
};

const STATE_COLOR: Record<CircuitBreakerState, string> = {
  CLOSED:    'text-emerald-400',
  HALF_OPEN: 'text-amber-400',
  OPEN:      'text-red-400',
};

function CircuitBreakerRow({ cb }: { cb: BrokerCircuitBreaker }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-700/50 last:border-0">
      <div className="flex items-center gap-2">
        {STATE_ICON[cb.state]}
        <span className="text-sm font-semibold text-slate-200">{cb.broker}</span>
      </div>
      <div className="flex items-center gap-4 text-xs font-mono">
        <span className="text-slate-500">{cb.last_latency_ms}ms</span>
        <span className="text-slate-500">{cb.error_rate_1m.toFixed(1)}/min</span>
        <span className={clsx('font-bold', STATE_COLOR[cb.state])}>{cb.state}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Positions table
// ---------------------------------------------------------------------------

type SortKey = 'symbol' | 'notional_usd' | 'unrealized_pnl' | 'delta_dollars' | 'var_contribution';
type SortDir = 'asc' | 'desc';

function PositionsTable({ positions }: { positions: PositionRow[] }) {
  const [sortKey, setSortKey] = useState<SortKey>('notional_usd');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const sorted = useMemo(() => {
    return [...positions].sort((a, b) => {
      const av = a[sortKey] as number | string;
      const bv = b[sortKey] as number | string;
      const cmp = typeof av === 'string'
        ? av.localeCompare(bv as string)
        : (av as number) - (bv as number);
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [positions, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  function SortTH({ col, label }: { col: SortKey; label: string }) {
    const active = sortKey === col;
    return (
      <th
        className="px-3 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-500 cursor-pointer hover:text-slate-300 whitespace-nowrap select-none"
        onClick={() => toggleSort(col)}
      >
        <span className="flex items-center gap-1">
          {label}
          {active && (sortDir === 'asc' ? <ChevronUp size={10} /> : <ChevronDown size={10} />)}
        </span>
      </th>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-slate-700">
            <SortTH col="symbol"           label="Symbol" />
            <th className="px-3 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-500">Class</th>
            <th className="px-3 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-500">Side</th>
            <SortTH col="notional_usd"     label="Notional" />
            <th className="px-3 py-2 text-right text-[10px] font-bold uppercase tracking-wider text-slate-500">Wt%</th>
            <th className="px-3 py-2 text-right text-[10px] font-bold uppercase tracking-wider text-slate-500">Entry</th>
            <th className="px-3 py-2 text-right text-[10px] font-bold uppercase tracking-wider text-slate-500">Mark</th>
            <SortTH col="unrealized_pnl"   label="Unreal P&L" />
            <SortTH col="delta_dollars"    label="Delta $" />
            <SortTH col="var_contribution" label="VaR %" />
          </tr>
        </thead>
        <tbody>
          {sorted.map(pos => (
            <tr key={pos.symbol} className="border-b border-slate-800 hover:bg-slate-800/40 transition-colors">
              <td className="px-3 py-2 font-mono font-semibold text-slate-200 whitespace-nowrap">{pos.symbol}</td>
              <td className="px-3 py-2">
                <span className="text-[10px] bg-slate-700/60 text-slate-400 px-1.5 py-0.5 rounded font-mono uppercase">
                  {pos.asset_class}
                </span>
              </td>
              <td className="px-3 py-2">
                <span className={clsx('text-[10px] font-bold', pos.side === 'long' ? 'text-emerald-400' : 'text-red-400')}>
                  {pos.side.toUpperCase()}
                </span>
              </td>
              <td className="px-3 py-2 font-mono text-slate-200 text-right tabular-nums">
                {fmt$(pos.notional_usd)}
              </td>
              <td className="px-3 py-2 font-mono text-right tabular-nums text-slate-400">
                {fmtPct(Math.abs(pos.weight))}
              </td>
              <td className="px-3 py-2 font-mono text-right tabular-nums text-slate-400">
                {pos.entry_price < 1
                  ? pos.entry_price.toFixed(4)
                  : pos.entry_price.toLocaleString('en-US', { maximumFractionDigits: 2 })}
              </td>
              <td className="px-3 py-2 font-mono text-right tabular-nums text-slate-300">
                {pos.current_price < 1
                  ? pos.current_price.toFixed(4)
                  : pos.current_price.toLocaleString('en-US', { maximumFractionDigits: 2 })}
              </td>
              <td className={clsx('px-3 py-2 font-mono text-right tabular-nums', pnlColor(pos.unrealized_pnl))}>
                {pos.unrealized_pnl >= 0 ? '+' : ''}{fmt$(pos.unrealized_pnl)}
              </td>
              <td className={clsx('px-3 py-2 font-mono text-right tabular-nums', pnlColor(pos.delta_dollars))}>
                {fmt$(pos.delta_dollars)}
              </td>
              <td className="px-3 py-2 font-mono text-right tabular-nums text-slate-400">
                {fmtPct(pos.var_contribution)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Greeks table
// ---------------------------------------------------------------------------

function GreeksTable({ rows }: { rows: GreeksSummaryRow[] }) {
  if (!rows.length) {
    return <p className="text-xs text-slate-500 py-4 text-center">No options positions.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-slate-700">
            {['Symbol', 'Type', 'Strike', 'Expiry', 'Qty', 'Delta', 'Gamma', 'Vega $', 'Theta $/d', 'Rho', 'IV'].map(h => (
              <th key={h} className="px-3 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-500 whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={`${row.symbol}-${i}`} className="border-b border-slate-800 hover:bg-slate-800/40">
              <td className="px-3 py-2 font-mono text-slate-200 whitespace-nowrap">{row.symbol}</td>
              <td className="px-3 py-2">
                <span className={clsx('text-[10px] font-bold', row.option_type === 'call' ? 'text-emerald-400' : 'text-red-400')}>
                  {row.option_type.toUpperCase()}
                </span>
              </td>
              <td className="px-3 py-2 font-mono tabular-nums text-slate-300">{row.strike.toLocaleString()}</td>
              <td className="px-3 py-2 font-mono text-slate-400 whitespace-nowrap">
                {row.expiry ? format(parseISO(row.expiry), 'dd MMM yy') : '--'}
              </td>
              <td className="px-3 py-2 font-mono tabular-nums text-slate-400">{row.qty.toFixed(2)}</td>
              <td className="px-3 py-2 font-mono tabular-nums text-slate-200">{row.delta.toFixed(3)}</td>
              <td className="px-3 py-2 font-mono tabular-nums text-slate-400">{row.gamma.toFixed(4)}</td>
              <td className={clsx('px-3 py-2 font-mono tabular-nums', row.vega >= 0 ? 'text-emerald-400' : 'text-red-400')}>
                {fmt$(row.vega)}
              </td>
              <td className={clsx('px-3 py-2 font-mono tabular-nums', row.theta < 0 ? 'text-red-400' : 'text-emerald-400')}>
                {row.theta.toFixed(2)}
              </td>
              <td className="px-3 py-2 font-mono tabular-nums text-slate-500">{row.rho.toFixed(4)}</td>
              <td className="px-3 py-2 font-mono tabular-nums text-slate-400">{fmtPct(row.iv)}</td>
            </tr>
          ))}
        </tbody>

        {/* Net aggregate row */}
        <tfoot className="border-t-2 border-slate-600">
          <tr className="bg-slate-800/40">
            <td colSpan={5} className="px-3 py-2 text-xs font-bold text-slate-300 uppercase">Net Portfolio</td>
            <td className="px-3 py-2 font-mono tabular-nums text-slate-100 font-bold">
              {rows.reduce((s, r) => s + r.delta * r.qty, 0).toFixed(3)}
            </td>
            <td className="px-3 py-2 font-mono tabular-nums text-slate-100 font-bold">
              {rows.reduce((s, r) => s + r.gamma * r.qty, 0).toFixed(5)}
            </td>
            <td className={clsx('px-3 py-2 font-mono tabular-nums font-bold',
              rows.reduce((s, r) => s + r.vega, 0) >= 0 ? 'text-emerald-400' : 'text-red-400')}>
              {fmt$(rows.reduce((s, r) => s + r.vega, 0))}
            </td>
            <td className={clsx('px-3 py-2 font-mono tabular-nums font-bold',
              rows.reduce((s, r) => s + r.theta, 0) < 0 ? 'text-red-400' : 'text-emerald-400')}>
              {rows.reduce((s, r) => s + r.theta, 0).toFixed(2)}
            </td>
            <td colSpan={2} />
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Limits panel
// ---------------------------------------------------------------------------

function LimitsPanel({ limits }: { limits: LimitRow[] }) {
  return (
    <div className="space-y-3">
      {limits.map(lim => (
        <div key={lim.name}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-slate-400">{lim.name}</span>
            <div className="flex items-center gap-2">
              {lim.status !== 'ok' && (
                <span className={clsx(
                  'text-[9px] font-bold px-1 py-0.5 rounded border',
                  lim.status === 'breach'
                    ? 'bg-red-500/10 text-red-400 border-red-500/30'
                    : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                )}>
                  {lim.status.toUpperCase()}
                </span>
              )}
              <span className={clsx('text-xs font-mono font-bold', limitTextColor(lim.utilization))}>
                {fmtPct(lim.utilization)}
              </span>
            </div>
          </div>
          <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div
              className={clsx('h-full rounded-full transition-all duration-500', limitBarColor(lim.utilization))}
              style={{ width: `${Math.min(100, lim.utilization * 100)}%` }}
            />
          </div>
          <div className="flex justify-between mt-0.5">
            <span className="text-[10px] text-slate-600 font-mono">
              {lim.current_value < 10 ? fmtPct(lim.current_value) : fmt$(lim.current_value)}
            </span>
            <span className="text-[10px] text-slate-700 font-mono">
              limit: {lim.limit_value < 10 ? fmtPct(lim.limit_value) : fmt$(lim.limit_value)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function RiskDashboard() {
  const { data, isLoading, isError, dataUpdatedAt } = usePortfolioRisk();
  const refresh = useRefreshRisk();
  const rawAlerts = useVaRAlerts();

  const [activeTab, setActiveTab] = useState<'positions' | 'greeks'>('positions');

  const alertItems = useMemo(() =>
    rawAlerts.map(a => ({
      id: a.id,
      severity: a.severity,
      title: `VaR breach -- ${a.method}`,
      message: a.message,
      timestamp: a.timestamp,
      acknowledged: a.acknowledged,
      value: a.value,
      threshold: a.threshold,
      source: 'risk-api:8791',
    })),
    [rawAlerts]
  );

  const corrData = useMemo(() => {
    const c = data?.correlation;
    if (!c) return { labels: [] as string[], values: [] as number[][] };
    return { labels: c.symbols, values: c.pearson };
  }, [data]);

  const varTrendData = useMemo(() =>
    (data?.var_trend ?? []).map(p => ({
      timestamp: p.timestamp,
      parametric: p.parametric_var99,
      historical: p.historical_var99,
      mc:         p.mc_var99,
      consensus:  p.consensus_var99,
    })),
    [data]
  );

  const lastUpdate = dataUpdatedAt ? format(new Date(dataUpdatedAt), 'HH:mm:ss') : '--';

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500 gap-2">
        <Activity size={18} className="animate-pulse" />
        <span className="text-sm">Loading risk data...</span>
      </div>
    );
  }

  const summary    = data?.summary;
  const varSummary = data?.var_summary;
  const greeks     = data?.greeks;
  const health     = data?.health;
  const limits     = data?.limits?.limits ?? [];
  const positions  = data?.positions ?? [];

  return (
    <div className="flex flex-col gap-5 p-4 min-h-screen bg-slate-950 text-slate-100">

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <Shield size={20} className="text-blue-400" />
          <div>
            <h1 className="text-lg font-bold text-slate-100">Risk Dashboard</h1>
            <p className="text-xs text-slate-500">
              Live risk monitoring -- risk-api:8791
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {isError && (
            <span className="text-xs text-amber-400 flex items-center gap-1">
              <AlertTriangle size={12} /> API unreachable -- using mock data
            </span>
          )}
          <span className="text-[11px] text-slate-600 font-mono">Updated {lastUpdate}</span>
          <button
            onClick={refresh}
            className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded border border-slate-700 transition-colors"
          >
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      </div>

      {/* KPI bar */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          label="Portfolio Equity"
          value={summary?.equity ?? 0}
          format="currency"
          change={summary?.daily_pnl_pct}
          higherIsBetter={true}
          subValue={`Daily P&L: ${fmt$(summary?.daily_pnl ?? 0)}`}
          variant={(summary?.daily_pnl ?? 0) >= 0 ? 'positive' : 'default'}
        />
        <MetricCard
          label="VaR 99 (consensus)"
          value={varSummary?.consensus_var99 ?? 0}
          format="currency"
          higherIsBetter={false}
          subValue={`${fmtPct((varSummary?.consensus_var99 ?? 0) / (summary?.equity ?? 1))} of equity`}
          variant={varSummary?.breach_flag ? 'critical' : 'default'}
        />
        <MetricCard
          label="Gross Exposure"
          value={summary?.gross_exposure ?? 0}
          format="currency"
          subValue={`${(summary?.leverage ?? 0).toFixed(2)}x leverage`}
          utilization={(summary?.gross_exposure ?? 0) / ((summary?.equity ?? 1) * 2)}
          warnAt={0.65}
          criticalAt={0.85}
        />
        <MetricCard
          label="Net Delta $"
          value={greeks?.dollar_delta ?? 0}
          format="currency"
          subValue={`Net delta: ${(greeks?.net_delta ?? 0).toFixed(3)}`}
        />
        <MetricCard
          label="Theta / Day"
          value={greeks?.net_theta ?? 0}
          format="currency"
          higherIsBetter={false}
          subValue={`Vega: ${fmt$(greeks?.dollar_vega ?? 0)} per 1%`}
        />
        <MetricCard
          label="Margin Utilization"
          value={summary?.margin_utilization ?? 0}
          format="percent"
          higherIsBetter={false}
          utilization={summary?.margin_utilization ?? 0}
          warnAt={0.60}
          criticalAt={0.80}
          variant={
            (summary?.margin_utilization ?? 0) >= 0.80 ? 'critical' :
            (summary?.margin_utilization ?? 0) >= 0.60 ? 'warn' :
            'default'
          }
        />
      </div>

      {/* VaR methods + trend */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* VaR comparison */}
        <div className="bg-slate-900/60 border border-slate-700/50 rounded-lg p-4 flex flex-col gap-3">
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Zap size={12} /> VaR Comparison (99%)
          </p>
          <div className="flex flex-wrap gap-2">
            {varSummary && (
              <>
                <VaRBadge method="parametric"  varValue={varSummary.parametric.var_99}  cvarValue={varSummary.parametric.cvar_99}  label="Parametric" />
                <VaRBadge method="historical"  varValue={varSummary.historical.var_99}  cvarValue={varSummary.historical.cvar_99}  label="Historical" />
                <VaRBadge method="monte_carlo" varValue={varSummary.monte_carlo.var_99} cvarValue={varSummary.monte_carlo.cvar_99} label="Monte Carlo" />
              </>
            )}
          </div>
          <div className="border-t border-slate-700/50 pt-2">
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <p className="text-[10px] text-slate-600 uppercase font-bold mb-1">Consensus VaR99</p>
                <p className="text-lg font-bold font-mono text-slate-100">{fmt$(varSummary?.consensus_var99 ?? 0)}</p>
              </div>
              <div>
                <p className="text-[10px] text-slate-600 uppercase font-bold mb-1">Breach Status</p>
                <div className="flex items-center gap-1.5 mt-1">
                  {varSummary?.breach_flag
                    ? <><XCircle size={14} className="text-red-400" /><span className="text-red-400 font-bold text-xs">BREACH</span></>
                    : <><CheckCircle2 size={14} className="text-emerald-400" /><span className="text-emerald-400 font-bold text-xs">WITHIN LIMITS</span></>
                  }
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* VaR trend chart */}
        <div className="lg:col-span-2 bg-slate-900/60 border border-slate-700/50 rounded-lg p-4">
          <TimeSeriesChart
            data={varTrendData}
            xKey="timestamp"
            xFormat="HH:mm"
            title="VaR 99 Intraday Trend"
            height={160}
            showLegend={true}
            showGrid
            series={[
              { key: 'parametric', label: 'Parametric', color: '#60a5fa', type: 'line', strokeWidth: 1.5 },
              { key: 'historical', label: 'Historical',  color: '#a78bfa', type: 'line', strokeWidth: 1.5 },
              { key: 'mc',         label: 'Monte Carlo', color: '#f59e0b', type: 'line', strokeWidth: 1 },
              { key: 'consensus',  label: 'Consensus',   color: '#22c55e', type: 'area', strokeWidth: 2, fillOpacity: 0.08 },
            ]}
            tooltipFormatter={(v: number) => fmt$(v)}
          />
        </div>
      </div>

      {/* Positions / Greeks tabs */}
      <div className="bg-slate-900/60 border border-slate-700/50 rounded-lg p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex gap-1">
            {(['positions', 'greeks'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={clsx(
                  'px-3 py-1.5 rounded text-xs font-semibold uppercase tracking-wider transition-colors',
                  activeTab === tab
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                )}
              >
                {tab === 'positions' ? `Positions (${positions.length})` : `Options Greeks (${greeks?.positions.length ?? 0})`}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-3 text-xs font-mono text-slate-500">
            <span>Long: {positions.filter(p => p.side === 'long').length}</span>
            <span>Short: {positions.filter(p => p.side === 'short').length}</span>
            <span>Options: {positions.filter(p => p.asset_class === 'option').length}</span>
          </div>
        </div>
        {activeTab === 'positions'
          ? <PositionsTable positions={positions} />
          : <GreeksTable rows={greeks?.positions ?? []} />
        }
      </div>

      {/* Correlation + Limits + Circuit breakers */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Correlation heatmap */}
        <div className="bg-slate-900/60 border border-slate-700/50 rounded-lg p-4 overflow-auto">
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
            Correlation Matrix (EWMA)
          </p>
          {corrData.labels.length > 0 ? (
            <HeatMap
              rowLabels={corrData.labels}
              colLabels={corrData.labels}
              values={corrData.values}
              minValue={-1}
              maxValue={1}
              cellSize={38}
              highlightDiagonal={true}
              colorStops={['#ef4444', '#1e293b', '#22c55e']}
              decimals={2}
            />
          ) : (
            <p className="text-xs text-slate-600">No correlation data.</p>
          )}
          {data?.correlation && (
            <div className="flex flex-wrap items-center gap-3 mt-2 text-[11px] font-mono">
              <span className={data.correlation.is_crowding ? 'text-amber-400' : 'text-slate-600'}>
                {data.correlation.is_crowding ? 'CROWDING DETECTED' : 'No crowding'}
              </span>
              <span className={data.correlation.is_stress ? 'text-red-400' : 'text-slate-600'}>
                {data.correlation.is_stress ? 'STRESS REGIME' : 'Normal regime'}
              </span>
              <span className="text-slate-600">avg r={data.correlation.avg_correlation.toFixed(2)}</span>
            </div>
          )}
        </div>

        {/* Limits */}
        <div className="bg-slate-900/60 border border-slate-700/50 rounded-lg p-4">
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Risk Limits</p>
          <LimitsPanel limits={limits} />
        </div>

        {/* Circuit breakers */}
        <div className="bg-slate-900/60 border border-slate-700/50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Activity size={13} className="text-slate-400" />
            <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">API Circuit Breakers</p>
            <span className={clsx(
              'ml-auto text-[10px] font-bold px-1.5 py-0.5 rounded border',
              health?.status === 'healthy'  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' :
              health?.status === 'degraded' ? 'bg-amber-500/10  text-amber-400  border-amber-500/30'  :
              'bg-red-500/10 text-red-400 border-red-500/30'
            )}>
              {health?.status?.toUpperCase() ?? 'UNKNOWN'}
            </span>
          </div>
          {(health?.brokers ?? []).map(cb => (
            <CircuitBreakerRow key={cb.broker} cb={cb} />
          ))}
          {health && (
            <p className="text-[10px] text-slate-600 font-mono mt-3">
              Uptime: {Math.floor(health.uptime_seconds / 3600)}h{' '}
              {Math.floor((health.uptime_seconds % 3600) / 60)}m
            </p>
          )}
        </div>
      </div>

      {/* Alert panel */}
      {alertItems.length > 0 && (
        <div className="bg-slate-900/60 border border-amber-500/20 rounded-lg p-4">
          <AlertPanel
            alerts={alertItems}
            title="VaR Breach Alerts"
            showHeader={true}
            maxVisible={6}
          />
        </div>
      )}

      {/* P&L attribution bar chart */}
      {positions.length > 0 && (
        <div className="bg-slate-900/60 border border-slate-700/50 rounded-lg p-4">
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
            P&L Attribution by Position
          </p>
          <ResponsiveContainer width="100%" height={140}>
            <BarChart
              data={positions.slice(0, 12).map(p => ({
                symbol: p.symbol.length > 12 ? p.symbol.slice(0, 11) + '\u2026' : p.symbol,
                pnl: p.unrealized_pnl + p.realized_pnl,
              }))}
              margin={{ top: 4, right: 8, bottom: 4, left: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis
                dataKey="symbol"
                tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'monospace' }}
                tickLine={false}
                axisLine={{ stroke: '#334155' }}
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'monospace' }}
                tickLine={false}
                axisLine={false}
                tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
                width={36}
              />
              <Tooltip
                formatter={(v: number) => [fmt$(v), 'P&L']}
                contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 11, borderRadius: 6 }}
                labelStyle={{ color: '#94a3b8' }}
              />
              <Bar dataKey="pnl" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                {positions.slice(0, 12).map((p, i) => (
                  <Cell
                    key={i}
                    fill={(p.unrealized_pnl + p.realized_pnl) >= 0 ? '#22c55e' : '#ef4444'}
                    opacity={0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

    </div>
  );
}
