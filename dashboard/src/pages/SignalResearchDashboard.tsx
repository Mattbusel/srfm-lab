// ============================================================
// pages/SignalResearchDashboard.tsx -- Signal analytics and
// ML pipeline dashboard.
//
// Sections:
//   1. Signal library table with lifecycle status filter
//   2. IC/ICIR rolling charts for selected signal
//   3. Alpha decay curves per signal category
//   4. Regime overlay -- BH mass, Hurst H, state timeline
//   5. Feature importance from ML pipeline (top 20)
//   6. Backtest equity curve + drawdown for selected signal
//   7. Category summary cards
// ============================================================

import React, { useState, useMemo } from 'react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import { clsx } from 'clsx';
import {
  RefreshCw, Search, Filter, TrendingUp, TrendingDown,
  Activity, BarChart2, Cpu, Layers, ChevronDown, ChevronUp,
  CheckCircle, Clock, Archive, Beaker,
} from 'lucide-react';
import { format, parseISO } from 'date-fns';

import { TimeSeriesChart }  from '../components/TimeSeriesChart';
import { DrawdownChart }    from '../components/DrawdownChart';
import { MetricCard }       from '../components/MetricCard';
import { SparkArea }        from '../components/SparkLine';

import {
  useSignalAnalytics,
  useBacktest,
  useCategorySummaries,
  useRefreshSignals,
} from '../hooks/useSignalAPI';

import type {
  SignalLibraryRow,
  SignalStatus,
  SignalCategory,
  AlphaDecayResult,
} from '../types/signals';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const CATEGORY_COLORS: Record<SignalCategory, string> = {
  MOMENTUM:       '#60a5fa',
  MEAN_REVERSION: '#a78bfa',
  VOLATILITY:     '#f59e0b',
  MICROSTRUCTURE: '#34d399',
  PHYSICS:        '#fb7185',
  TECHNICAL:      '#94a3b8',
  ONCHAIN:        '#22d3ee',
  MACRO:          '#fbbf24',
};

const STATUS_CONFIG: Record<SignalStatus, { label: string; color: string; Icon: any }> = {
  active:    { label: 'Active',    color: 'text-emerald-400', Icon: CheckCircle },
  probation: { label: 'Probation', color: 'text-amber-400',   Icon: Clock       },
  retired:   { label: 'Retired',   color: 'text-slate-500',   Icon: Archive     },
  research:  { label: 'Research',  color: 'text-blue-400',    Icon: Beaker      },
};

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

const fmtIC = (v: number) => (v >= 0 ? '+' : '') + v.toFixed(4);
const fmtPct = (v: number, dp = 1) => `${(v * 100).toFixed(dp)}%`;
const fmt$ = (v: number) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(v);

// ---------------------------------------------------------------------------
// Signal status badge
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: SignalStatus }) {
  const cfg = STATUS_CONFIG[status];
  return (
    <span className={clsx('text-[10px] font-bold flex items-center gap-0.5', cfg.color)}>
      <cfg.Icon size={10} />
      {cfg.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Category chip
// ---------------------------------------------------------------------------

function CategoryChip({ category }: { category: SignalCategory }) {
  const color = CATEGORY_COLORS[category] ?? '#94a3b8';
  return (
    <span
      className="text-[9px] font-bold px-1.5 py-0.5 rounded border font-mono uppercase"
      style={{ color, borderColor: color + '40', background: color + '15' }}
    >
      {category.slice(0, 4)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Signal library table
// ---------------------------------------------------------------------------

type LibSortKey = 'name' | 'current_ic' | 'current_icir' | 'sharpe' | 'max_drawdown' | 'win_rate';

function SignalLibraryTable({
  rows,
  selectedId,
  onSelect,
}: {
  rows: SignalLibraryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const [sortKey, setSortKey]   = useState<LibSortKey>('current_icir');
  const [sortDir, setSortDir]   = useState<'asc' | 'desc'>('desc');
  const [search,  setSearch]    = useState('');
  const [filter,  setFilter]    = useState<SignalStatus | 'all'>('all');
  const [catFilter, setCatFilter] = useState<SignalCategory | 'all'>('all');

  const filtered = useMemo(() => {
    return rows
      .filter(r =>
        (filter === 'all' || r.meta.status === filter) &&
        (catFilter === 'all' || r.meta.category === catFilter) &&
        (search === '' || r.meta.id.includes(search.toLowerCase()) || r.meta.name.toLowerCase().includes(search.toLowerCase()))
      )
      .sort((a, b) => {
        let av: number | string = 0;
        let bv: number | string = 0;
        if (sortKey === 'name') { av = a.meta.name; bv = b.meta.name; }
        else if (sortKey === 'current_ic')   { av = a.current_ic;   bv = b.current_ic; }
        else if (sortKey === 'current_icir') { av = a.current_icir; bv = b.current_icir; }
        else if (sortKey === 'sharpe')       { av = a.sharpe;       bv = b.sharpe; }
        else if (sortKey === 'max_drawdown') { av = a.max_drawdown; bv = b.max_drawdown; }
        else if (sortKey === 'win_rate')     { av = a.win_rate;     bv = b.win_rate; }
        const cmp = typeof av === 'string' ? av.localeCompare(bv as string) : (av as number) - (bv as number);
        return sortDir === 'asc' ? cmp : -cmp;
      });
  }, [rows, filter, catFilter, search, sortKey, sortDir]);

  function SortTH({ col, label }: { col: LibSortKey; label: string }) {
    const active = sortKey === col;
    return (
      <th
        className="px-2 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-500 cursor-pointer hover:text-slate-300 whitespace-nowrap select-none"
        onClick={() => { if (sortKey === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc'); else { setSortKey(col); setSortDir('desc'); } }}
      >
        <span className="flex items-center gap-0.5">
          {label}
          {active && (sortDir === 'asc' ? <ChevronUp size={9} /> : <ChevronDown size={9} />)}
        </span>
      </th>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search size={12} className="absolute left-2 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search signals..."
            className="bg-slate-800 border border-slate-700 text-slate-300 text-xs pl-7 pr-3 py-1.5 rounded w-44 focus:outline-none focus:border-blue-500"
          />
        </div>
        <select
          value={filter}
          onChange={e => setFilter(e.target.value as any)}
          className="bg-slate-800 border border-slate-700 text-slate-300 text-xs px-2 py-1.5 rounded focus:outline-none"
        >
          <option value="all">All Statuses</option>
          <option value="active">Active</option>
          <option value="probation">Probation</option>
          <option value="retired">Retired</option>
          <option value="research">Research</option>
        </select>
        <select
          value={catFilter}
          onChange={e => setCatFilter(e.target.value as any)}
          className="bg-slate-800 border border-slate-700 text-slate-300 text-xs px-2 py-1.5 rounded focus:outline-none"
        >
          <option value="all">All Categories</option>
          {Object.keys(CATEGORY_COLORS).map(cat => (
            <option key={cat} value={cat}>{cat}</option>
          ))}
        </select>
        <span className="text-xs text-slate-500 font-mono ml-auto">
          {filtered.length} / {rows.length} signals
        </span>
      </div>

      {/* Table */}
      <div className="overflow-auto max-h-72">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-slate-900 z-10">
            <tr className="border-b border-slate-700">
              <SortTH col="name"         label="Signal" />
              <th className="px-2 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-500">Cat</th>
              <th className="px-2 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-500">Status</th>
              <SortTH col="current_ic"   label="IC" />
              <SortTH col="current_icir" label="ICIR" />
              <SortTH col="sharpe"       label="Sharpe" />
              <SortTH col="max_drawdown" label="MaxDD" />
              <SortTH col="win_rate"     label="Win%" />
              <th className="px-2 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-500">Decay T1/2</th>
              <th className="px-2 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-500">IC 30d</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(row => {
              const selected = selectedId === row.meta.id;
              const icData = row.ic_rolling.data.slice(-20).map(d => ({ value: d.ic }));
              return (
                <tr
                  key={row.meta.id}
                  onClick={() => onSelect(row.meta.id)}
                  className={clsx(
                    'border-b border-slate-800 cursor-pointer transition-colors',
                    selected ? 'bg-blue-900/30 border-blue-500/30' : 'hover:bg-slate-800/40'
                  )}
                >
                  <td className="px-2 py-1.5 font-mono text-slate-200 whitespace-nowrap">
                    <span className={clsx('text-xs', selected ? 'text-blue-300 font-semibold' : '')}>
                      {row.meta.id}
                    </span>
                  </td>
                  <td className="px-2 py-1.5">
                    <CategoryChip category={row.meta.category} />
                  </td>
                  <td className="px-2 py-1.5">
                    <StatusBadge status={row.meta.status} />
                  </td>
                  <td className={clsx('px-2 py-1.5 font-mono tabular-nums text-right',
                    row.current_ic > 0 ? 'text-emerald-400' : row.current_ic < 0 ? 'text-red-400' : 'text-slate-500')}>
                    {fmtIC(row.current_ic)}
                  </td>
                  <td className={clsx('px-2 py-1.5 font-mono tabular-nums text-right',
                    Math.abs(row.current_icir) > 0.5 ? 'text-emerald-300' : 'text-slate-400')}>
                    {row.current_icir.toFixed(2)}
                  </td>
                  <td className={clsx('px-2 py-1.5 font-mono tabular-nums text-right',
                    row.sharpe > 1 ? 'text-emerald-400' : row.sharpe < 0 ? 'text-red-400' : 'text-slate-400')}>
                    {row.sharpe.toFixed(2)}
                  </td>
                  <td className="px-2 py-1.5 font-mono tabular-nums text-right text-red-400">
                    {fmtPct(row.max_drawdown)}
                  </td>
                  <td className="px-2 py-1.5 font-mono tabular-nums text-right text-slate-400">
                    {fmtPct(row.win_rate)}
                  </td>
                  <td className="px-2 py-1.5 font-mono tabular-nums text-right text-slate-400">
                    {row.decay.half_life_bars.toFixed(1)}b
                  </td>
                  <td className="px-2 py-1.5">
                    <SparkArea
                      data={icData}
                      width={60}
                      height={22}
                      showZero
                      higherIsBetter
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// IC rolling chart
// ---------------------------------------------------------------------------

function ICRollingPanel({ signal }: { signal: SignalLibraryRow }) {
  const rolling = signal.ic_rolling;
  const chartData = rolling.data.map(d => ({
    date: d.date,
    ic: d.ic,
    ic_spearman: d.ic_spearman,
  }));

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
        <span>Mean IC: <span className={rolling.mean_ic > 0 ? 'text-emerald-400' : 'text-red-400'}>{fmtIC(rolling.mean_ic)}</span></span>
        <span>ICIR: <span className={Math.abs(rolling.icir) > 0.5 ? 'text-emerald-300' : 'text-slate-400'}>{rolling.icir.toFixed(3)}</span></span>
        <span>t-stat: <span className="text-slate-300">{rolling.t_stat.toFixed(2)}</span></span>
        <span>p-value: <span className={rolling.p_value < 0.05 ? 'text-emerald-400' : 'text-amber-400'}>{rolling.p_value.toFixed(4)}</span></span>
        <span>IC+ rate: <span className="text-slate-300">{fmtPct(rolling.ic_positive_rate)}</span></span>
      </div>
      <TimeSeriesChart
        data={chartData}
        xKey="date"
        xFormat="MMM d"
        height={160}
        showGrid
        showLegend
        series={[
          { key: 'ic',         label: 'IC (Pearson)',  color: '#60a5fa', type: 'area', strokeWidth: 1.5, fillOpacity: 0.1 },
          { key: 'ic_spearman', label: 'IC (Spearman)', color: '#a78bfa', type: 'line', strokeWidth: 1, strokeDasharray: '3 3' },
        ]}
        referenceLines={[{ y: 0, color: '#475569', dashArray: '2 2' }]}
        tooltipFormatter={(v: number) => v.toFixed(4)}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Alpha decay panel
// ---------------------------------------------------------------------------

function AlphaDecayPanel({ signals }: { signals: SignalLibraryRow[] }) {
  const [selectedCat, setSelectedCat] = useState<SignalCategory | 'all'>('all');

  const filteredDecays = useMemo(() => {
    const top = signals
      .filter(s => s.meta.status === 'active' && (selectedCat === 'all' || s.meta.category === selectedCat))
      .slice(0, 8);
    return top.map(s => ({
      id: s.meta.id,
      category: s.meta.category,
      decay: s.decay,
      color: CATEGORY_COLORS[s.meta.category] ?? '#94a3b8',
    }));
  }, [signals, selectedCat]);

  const maxHorizon = 20;
  const chartData = Array.from({ length: maxHorizon }, (_, i) => {
    const point: Record<string, number> = { horizon: i + 1 };
    filteredDecays.forEach(d => {
      const dp = d.decay.data.find(x => x.horizon === i + 1);
      if (dp) point[d.id] = dp.ic;
    });
    return point;
  });

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 flex-wrap">
        <select
          value={selectedCat}
          onChange={e => setSelectedCat(e.target.value as any)}
          className="bg-slate-800 border border-slate-700 text-slate-300 text-xs px-2 py-1 rounded focus:outline-none"
        >
          <option value="all">All Categories</option>
          {Object.keys(CATEGORY_COLORS).map(cat => <option key={cat} value={cat}>{cat}</option>)}
        </select>
        <span className="text-[11px] text-slate-500 font-mono">Top 8 active signals by ICIR</span>
      </div>

      <TimeSeriesChart
        data={chartData}
        xKey="horizon"
        xFormatter={(v: number) => `${v}b`}
        height={160}
        showGrid
        showLegend
        series={filteredDecays.map(d => ({
          key: d.id,
          label: d.id,
          color: d.color,
          type: 'line' as const,
          strokeWidth: 1.5,
        }))}
        referenceLines={[{ y: 0, color: '#475569', dashArray: '2 2' }]}
        tooltipFormatter={(v: number) => v.toFixed(4)}
        leftLabel="IC"
      />

      {/* Half-life summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {filteredDecays.slice(0, 4).map(d => (
          <div key={d.id} className="bg-slate-800/40 rounded p-2 border border-slate-700/40">
            <p className="text-[10px] font-mono text-slate-500 truncate">{d.id}</p>
            <p className="text-sm font-bold font-mono" style={{ color: d.color }}>
              T1/2: {d.decay.half_life_bars.toFixed(1)}b
            </p>
            <p className="text-[10px] text-slate-600 font-mono">IC0: {d.decay.ic_at_zero.toFixed(4)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Regime overlay panel
// ---------------------------------------------------------------------------

function RegimePanel({ signals }: { signals: SignalLibraryRow[] }) {
  const { data } = useSignalAnalytics();
  const regime = data?.regime;

  if (!regime) return <p className="text-xs text-slate-500">No regime data.</p>;

  const recent = regime.data.slice(-60);
  const regimeColors: Record<string, string> = {
    trending:      '#60a5fa',
    mean_reverting: '#a78bfa',
    volatile:      '#ef4444',
    low_vol:       '#22c55e',
  };

  const chartData = recent.map(d => ({
    date: d.date,
    bh_mass: d.bh_mass,
    hurst_h: d.hurst_h,
    trending_score: d.trending_score,
    adx: d.adx,
  }));

  return (
    <div className="flex flex-col gap-3">
      {/* Current state */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Regime:</span>
          <span
            className="text-sm font-bold capitalize"
            style={{ color: regimeColors[regime.current_regime] ?? '#94a3b8' }}
          >
            {regime.current_regime.replace('_', ' ')}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">BH Mass:</span>
          <span className="text-sm font-bold font-mono text-amber-400">
            {regime.current_bh_mass.toFixed(3)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Hurst H:</span>
          <span className={clsx(
            'text-sm font-bold font-mono',
            regime.current_hurst > 0.55 ? 'text-blue-400' :
            regime.current_hurst < 0.45 ? 'text-purple-400' :
            'text-slate-400'
          )}>
            {regime.current_hurst.toFixed(3)}
            <span className="text-[10px] text-slate-500 ml-1 font-normal">
              {regime.current_hurst > 0.55 ? '(trending)' : regime.current_hurst < 0.45 ? '(mean-rev)' : '(random)'}
            </span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Duration:</span>
          <span className="text-sm font-bold font-mono text-slate-300">{regime.regime_duration_bars}b</span>
        </div>
      </div>

      {/* BH mass + Hurst chart */}
      <TimeSeriesChart
        data={chartData}
        xKey="date"
        xFormat="MMM d"
        height={150}
        showGrid
        showLegend
        series={[
          { key: 'bh_mass',       label: 'BH Mass',        color: '#f59e0b', type: 'area', yAxisId: 'left',  strokeWidth: 2, fillOpacity: 0.1 },
          { key: 'hurst_h',       label: 'Hurst H',         color: '#60a5fa', type: 'line', yAxisId: 'right', strokeWidth: 1.5 },
          { key: 'trending_score', label: 'Trending Score', color: '#34d399', type: 'line', yAxisId: 'right', strokeWidth: 1, strokeDasharray: '3 3' },
        ]}
        referenceLines={[
          { y: 0.55, yAxisId: 'right', color: '#60a5fa', dashArray: '4 4', label: 'H=0.55' },
          { y: 0.45, yAxisId: 'right', color: '#a78bfa', dashArray: '4 4', label: 'H=0.45' },
        ]}
        leftDomain={[0, 2]}
        rightDomain={[0.3, 0.8]}
        tooltipFormatter={(v: number, k: string) =>
          k === 'bh_mass' ? v.toFixed(3) : k === 'adx' ? v.toFixed(1) : v.toFixed(4)
        }
      />

      {/* IC conditioned on regime */}
      {signals.length > 0 && (
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-2">
            ICIR by Regime (top 5 active signals)
          </p>
          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="px-2 py-1 text-left text-[10px] text-slate-500 font-bold uppercase">Signal</th>
                  {['trending', 'mean_reverting', 'volatile', 'low_vol'].map(r => (
                    <th key={r} className="px-2 py-1 text-right text-[10px] uppercase font-bold"
                        style={{ color: regimeColors[r] ?? '#94a3b8' }}>
                      {r.replace('_', ' ')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {signals.filter(s => s.meta.status === 'active').slice(0, 5).map(sig => (
                  <tr key={sig.meta.id} className="border-b border-slate-800">
                    <td className="px-2 py-1 font-mono text-slate-300">{sig.meta.id}</td>
                    {(['trending', 'mean_reverting', 'volatile', 'low_vol'] as const).map(r => {
                      const ic = sig.regime_conditioned_ic[r];
                      return (
                        <td key={r} className={clsx('px-2 py-1 font-mono tabular-nums text-right',
                          ic > 0.03 ? 'text-emerald-400' : ic < -0.03 ? 'text-red-400' : 'text-slate-500')}>
                          {fmtIC(ic)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Feature importance
// ---------------------------------------------------------------------------

function FeatureImportancePanel() {
  const { data } = useSignalAnalytics();
  const fi = data?.feature_importance;
  if (!fi) return <p className="text-xs text-slate-500">No feature importance data.</p>;

  const top20 = fi.features.slice(0, 20);

  return (
    <div className="flex flex-col gap-3">
      {/* Model header */}
      <div className="flex items-center gap-4 flex-wrap text-xs font-mono">
        <span className="text-slate-400">Model: <span className="text-blue-300">{fi.model_id}</span></span>
        <span className="text-slate-400">Type: <span className="text-slate-200">{fi.model_type}</span></span>
        <span className="text-slate-400">OOS IC: <span className="text-emerald-400">{fi.out_of_sample_ic.toFixed(4)}</span></span>
        <span className="text-slate-400">Val Sharpe: <span className="text-emerald-400">{fi.validation_sharpe.toFixed(2)}</span></span>
        <span className="text-slate-400">N={fi.n_samples.toLocaleString()}</span>
      </div>

      {/* Importance bars */}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={top20.map(f => ({
            feature: f.feature_name.length > 18 ? f.feature_name.slice(0, 17) + '\u2026' : f.feature_name,
            importance: f.importance_score,
            category: f.signal_category,
          }))}
          layout="vertical"
          margin={{ top: 2, right: 16, bottom: 2, left: 140 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fill: '#94a3b8', fontSize: 9, fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
            width={136}
          />
          <Tooltip
            formatter={(v: number) => [v.toFixed(4), 'Importance']}
            contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 11, borderRadius: 6 }}
            labelStyle={{ color: '#94a3b8' }}
          />
          <Bar dataKey="importance" radius={[0, 3, 3, 0]} isAnimationActive={false}>
            {top20.map((f, i) => (
              <Cell key={i} fill={CATEGORY_COLORS[f.signal_category] ?? '#94a3b8'} opacity={0.8} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Category totals radar */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-2">Category Importance</p>
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart data={fi.top_categories.slice(0, 8).map(c => ({
              category: c.category.slice(0, 5),
              importance: parseFloat((c.total_importance * 100).toFixed(1)),
            }))}>
              <PolarGrid stroke="#1e293b" />
              <PolarAngleAxis dataKey="category" tick={{ fill: '#64748b', fontSize: 9 }} />
              <PolarRadiusAxis angle={30} tick={{ fill: '#64748b', fontSize: 8 }} />
              <Radar
                name="Importance"
                dataKey="importance"
                stroke="#60a5fa"
                fill="#60a5fa"
                fillOpacity={0.2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-2">Top Categories</p>
          <div className="space-y-1.5">
            {fi.top_categories.slice(0, 6).map(c => {
              const pct = c.total_importance / fi.top_categories.reduce((s, x) => s + x.total_importance, 0);
              return (
                <div key={c.category}>
                  <div className="flex justify-between text-[11px] mb-0.5">
                    <span style={{ color: CATEGORY_COLORS[c.category] ?? '#94a3b8' }}>{c.category}</span>
                    <span className="font-mono text-slate-400">{fmtPct(pct)}</span>
                  </div>
                  <div className="h-1 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${pct * 100}%`, backgroundColor: CATEGORY_COLORS[c.category] ?? '#94a3b8' }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Selected signal detail
// ---------------------------------------------------------------------------

function SignalDetailPanel({ signalId, signal }: { signalId: string; signal: SignalLibraryRow }) {
  const { data: bt, isLoading: btLoading } = useBacktest(signalId);
  const [activePanel, setActivePanel] = useState<'ic' | 'backtest'>('ic');

  return (
    <div className="flex flex-col gap-3">
      {/* Signal header */}
      <div className="flex items-start justify-between flex-wrap gap-2">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-mono font-bold text-slate-100">{signal.meta.id}</span>
            <CategoryChip category={signal.meta.category} />
            <StatusBadge status={signal.meta.status} />
          </div>
          <p className="text-xs text-slate-500 mt-0.5">{signal.meta.description}</p>
        </div>
        <div className="flex gap-1">
          {(['ic', 'backtest'] as const).map(p => (
            <button
              key={p}
              onClick={() => setActivePanel(p)}
              className={clsx(
                'px-2 py-1 text-[11px] rounded font-semibold uppercase tracking-wider transition-colors',
                activePanel === p ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 hover:text-slate-200'
              )}
            >
              {p === 'ic' ? 'IC/ICIR' : 'Backtest'}
            </button>
          ))}
        </div>
      </div>

      {activePanel === 'ic' && <ICRollingPanel signal={signal} />}

      {activePanel === 'backtest' && (
        btLoading ? (
          <div className="flex items-center gap-2 py-8 text-slate-500 text-xs">
            <Activity size={14} className="animate-pulse" /> Loading backtest...
          </div>
        ) : bt ? (
          <div className="flex flex-col gap-2">
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
              {[
                { label: 'Total Return', value: fmtPct(bt.total_return) },
                { label: 'Ann. Return',  value: fmtPct(bt.annualized_return) },
                { label: 'Sharpe',       value: bt.sharpe.toFixed(2) },
                { label: 'Max DD',       value: fmtPct(bt.max_drawdown) },
                { label: 'Win Rate',     value: fmtPct(bt.win_rate) },
                { label: 'N Trades',     value: bt.n_trades.toString() },
              ].map(m => (
                <div key={m.label} className="bg-slate-800/40 rounded p-2 border border-slate-700/40">
                  <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">{m.label}</p>
                  <p className="text-sm font-bold font-mono text-slate-100 mt-0.5">{m.value}</p>
                </div>
              ))}
            </div>
            {/* Equity curve */}
            <div className="bg-slate-800/20 rounded-lg p-3">
              <TimeSeriesChart
                data={bt.equity_curve.map(p => ({
                  date: p.date,
                  equity: p.equity,
                  benchmark: p.benchmark,
                }))}
                xKey="date"
                xFormat="MMM yy"
                height={160}
                showGrid
                showLegend
                series={[
                  { key: 'equity',    label: 'Strategy',  color: '#22c55e', type: 'area', strokeWidth: 2, fillOpacity: 0.08 },
                  { key: 'benchmark', label: 'Benchmark', color: '#94a3b8', type: 'line', strokeWidth: 1, strokeDasharray: '3 3' },
                ]}
                tooltipFormatter={(v: number) => fmt$(v)}
                title="Equity Curve"
              />
              <DrawdownChart
                data={bt.equity_curve.map(p => ({ date: p.date, drawdown: p.drawdown }))}
                height={70}
                compact
              />
            </div>
          </div>
        ) : (
          <p className="text-xs text-slate-500">No backtest data.</p>
        )
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function SignalResearchDashboard() {
  const { data, isLoading, isError } = useSignalAnalytics();
  const refresh = useRefreshSignals();
  const categorySummaries = useCategorySummaries();

  const [selectedSignalId, setSelectedSignalId] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<'library' | 'decay' | 'regime' | 'features'>('library');

  const signals  = data?.signals ?? [];
  const selected = selectedSignalId ? signals.find(s => s.meta.id === selectedSignalId) ?? null : null;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500 gap-2">
        <Activity size={18} className="animate-pulse" />
        <span className="text-sm">Loading signal analytics...</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 p-4 min-h-screen bg-slate-950 text-slate-100">

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <BarChart2 size={20} className="text-purple-400" />
          <div>
            <h1 className="text-lg font-bold text-slate-100">Signal Research</h1>
            <p className="text-xs text-slate-500">
              {data?.total_signals ?? 0} signals -- {data?.active_count} active, {data?.probation_count} probation
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {isError && <span className="text-xs text-amber-400">API unreachable -- using mock</span>}
          <button
            onClick={refresh}
            className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded border border-slate-700 transition-colors"
          >
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      </div>

      {/* Category summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
        {categorySummaries.map(cat => (
          <div
            key={cat.category}
            className="bg-slate-900/60 border border-slate-700/40 rounded-lg p-3 flex flex-col gap-1"
            style={{ borderTopColor: (CATEGORY_COLORS[cat.category] ?? '#94a3b8') + '60' }}
          >
            <span className="text-[9px] font-bold uppercase tracking-wider" style={{ color: CATEGORY_COLORS[cat.category] ?? '#94a3b8' }}>
              {cat.category.slice(0, 6)}
            </span>
            <span className="text-xs font-bold font-mono text-slate-200">{cat.active_count}/{cat.n_signals}</span>
            <span className="text-[10px] font-mono text-slate-500">ICIR {cat.mean_icir.toFixed(2)}</span>
            <span className="text-[10px] font-mono text-slate-600">SR {cat.mean_sharpe.toFixed(2)}</span>
          </div>
        ))}
      </div>

      {/* Section tabs */}
      <div className="flex gap-1 flex-wrap">
        {[
          { key: 'library',  label: 'Signal Library',    Icon: Layers },
          { key: 'decay',    label: 'Alpha Decay',        Icon: TrendingDown },
          { key: 'regime',   label: 'Regime Overlay',     Icon: Activity },
          { key: 'features', label: 'Feature Importance', Icon: Cpu },
        ].map(({ key, label, Icon }) => (
          <button
            key={key}
            onClick={() => setActiveSection(key as any)}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-2 rounded text-xs font-semibold uppercase tracking-wider transition-colors',
              activeSection === key
                ? 'bg-purple-700 text-white'
                : 'bg-slate-800 text-slate-400 hover:text-slate-200'
            )}
          >
            <Icon size={12} /> {label}
          </button>
        ))}
      </div>

      {/* Main content */}
      <div className={clsx('grid gap-4', selected ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1')}>
        {/* Left panel -- section content */}
        <div className="bg-slate-900/60 border border-slate-700/50 rounded-lg p-4">
          {activeSection === 'library' && (
            <SignalLibraryTable
              rows={signals}
              selectedId={selectedSignalId}
              onSelect={setSelectedSignalId}
            />
          )}
          {activeSection === 'decay' && <AlphaDecayPanel signals={signals} />}
          {activeSection === 'regime' && <RegimePanel signals={signals} />}
          {activeSection === 'features' && <FeatureImportancePanel />}
        </div>

        {/* Right panel -- signal detail (when selected) */}
        {selected && (
          <div className="bg-slate-900/60 border border-blue-500/20 rounded-lg p-4">
            <SignalDetailPanel signalId={selected.meta.id} signal={selected} />
          </div>
        )}
      </div>

    </div>
  );
}
