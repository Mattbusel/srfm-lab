// ============================================================
// WalkForward.tsx -- Walk-forward optimization analysis
// IS vs OOS Sharpe heatmap, fold timeline, stability
// ============================================================
import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, Legend,
} from 'recharts';
import { clsx } from 'clsx';

// ---- Types ----

interface WFFold {
  fold: number;
  is_start: string;
  is_end: string;
  oos_start: string;
  oos_end: string;
  is_sharpe: number;
  oos_sharpe: number;
  is_cagr: number;
  oos_cagr: number;
  n_trades: number;
  param_set: Record<string, number>;
}

interface WFResults {
  folds: WFFold[];
  is_mean_sharpe: number;
  oos_mean_sharpe: number;
  consistency_ratio: number;
  degradation_factor: number;
  param_stability: { name: string; values: number[]; std: number; cv: number }[];
}

// ---- Demo data ----

function buildDemoResults(): WFResults {
  const N = 12;
  const folds: WFFold[] = Array.from({ length: N }, (_, i) => {
    const is_sharpe = 0.8 + Math.sin(i * 0.8) * 0.6 + Math.random() * 0.3;
    const oos_sharpe = is_sharpe * (0.65 + Math.random() * 0.25);
    return {
      fold: i + 1,
      is_start: `202${Math.floor(i / 4)}-${String((i % 12) + 1).padStart(2, '0')}-01`,
      is_end: `202${Math.floor(i / 4)}-${String((i % 12) + 3).padStart(2, '0')}-30`,
      oos_start: `202${Math.floor(i / 4)}-${String((i % 12) + 4).padStart(2, '0')}-01`,
      oos_end: `202${Math.floor(i / 4)}-${String((i % 12) + 5).padStart(2, '0')}-30`,
      is_sharpe,
      oos_sharpe,
      is_cagr: 0.12 + Math.sin(i * 0.6) * 0.08,
      oos_cagr: 0.08 + Math.cos(i * 0.5) * 0.06,
      n_trades: Math.floor(40 + Math.random() * 60),
      param_set: {
        bh_threshold: 0.88 + Math.sin(i * 0.3) * 0.05,
        nav_gate: 2.8 + Math.cos(i * 0.4) * 0.3,
        hurst_min: 0.52 + Math.sin(i * 0.5) * 0.04,
        vol_target: 0.12 + Math.cos(i * 0.6) * 0.02,
      },
    };
  });

  const is_mean = folds.reduce((s, f) => s + f.is_sharpe, 0) / N;
  const oos_mean = folds.reduce((s, f) => s + f.oos_sharpe, 0) / N;
  const consistency_ratio = folds.filter((f) => f.oos_sharpe > 0).length / N;
  const degradation_factor = oos_mean / is_mean;

  const PARAM_NAMES = ['bh_threshold', 'nav_gate', 'hurst_min', 'vol_target'];
  const param_stability = PARAM_NAMES.map((name) => {
    const values = folds.map((f) => f.param_set[name] ?? 0);
    const mean = values.reduce((s, v) => s + v, 0) / N;
    const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / N);
    const cv = std / mean;
    return { name, values, std, cv };
  });

  return { folds, is_mean_sharpe: is_mean, oos_mean_sharpe: oos_mean, consistency_ratio, degradation_factor, param_stability };
}

// ---- Helpers ----

function sharpeColor(s: number): string {
  if (s > 1.5) return 'rgba(34,197,94,0.9)';
  if (s > 1.0) return 'rgba(34,197,94,0.6)';
  if (s > 0.5) return 'rgba(234,179,8,0.7)';
  if (s > 0.0) return 'rgba(234,179,8,0.4)';
  return 'rgba(239,68,68,0.7)';
}

function fmt2(v: number) { return v.toFixed(2); }
function fmtPct(v: number) { return `${(v * 100).toFixed(1)}%`; }

// ---- Sub-components ----

const WalkForwardGrid: React.FC<{ folds: WFFold[] }> = ({ folds }) => {
  const [hovered, setHovered] = useState<number | null>(null);
  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        IS vs OOS Sharpe Grid
      </h3>
      <div className="overflow-x-auto">
        <table className="text-xs w-full">
          <thead>
            <tr className="text-gray-500 border-b border-gray-700">
              <th className="text-left py-1 pr-3">Fold</th>
              <th className="text-center pr-2" colSpan={2}>In-Sample</th>
              <th className="text-center" colSpan={2}>Out-of-Sample</th>
            </tr>
            <tr className="text-gray-600 border-b border-gray-800">
              <th className="py-1 pr-3"></th>
              <th className="text-center pr-2">Sharpe</th>
              <th className="text-center pr-2">CAGR</th>
              <th className="text-center pr-2">Sharpe</th>
              <th className="text-center">CAGR</th>
            </tr>
          </thead>
          <tbody>
            {folds.map((f) => (
              <tr
                key={f.fold}
                className={clsx('border-b border-gray-800 cursor-pointer', hovered === f.fold ? 'bg-gray-800' : 'hover:bg-gray-800')}
                onMouseEnter={() => setHovered(f.fold)}
                onMouseLeave={() => setHovered(null)}
              >
                <td className="py-1 pr-3 text-gray-400 font-mono">F{f.fold}</td>
                <td className="text-center pr-2">
                  <span
                    className="px-2 py-0.5 rounded text-white font-mono font-bold"
                    style={{ background: sharpeColor(f.is_sharpe) }}
                  >
                    {fmt2(f.is_sharpe)}
                  </span>
                </td>
                <td className="text-center pr-2 text-gray-300 font-mono">{fmtPct(f.is_cagr)}</td>
                <td className="text-center pr-2">
                  <span
                    className="px-2 py-0.5 rounded text-white font-mono font-bold"
                    style={{ background: sharpeColor(f.oos_sharpe) }}
                  >
                    {fmt2(f.oos_sharpe)}
                  </span>
                </td>
                <td className="text-center text-gray-300 font-mono">{fmtPct(f.oos_cagr)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const ConsistencyMetrics: React.FC<{
  is_mean: number;
  oos_mean: number;
  consistency_ratio: number;
  degradation_factor: number;
}> = ({ is_mean, oos_mean, consistency_ratio, degradation_factor }) => {
  const metrics = [
    { label: 'IS Mean Sharpe', value: fmt2(is_mean), color: 'text-blue-400' },
    { label: 'OOS Mean Sharpe', value: fmt2(oos_mean), color: oos_mean > 0.5 ? 'text-green-400' : 'text-yellow-400' },
    { label: 'Consistency Ratio', value: fmtPct(consistency_ratio), color: consistency_ratio > 0.7 ? 'text-green-400' : 'text-yellow-400' },
    { label: 'Degradation Factor', value: fmt2(degradation_factor), color: degradation_factor > 0.6 ? 'text-green-400' : 'text-red-400' },
  ];

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
        Consistency Metrics
      </h3>
      <div className="grid grid-cols-2 gap-3">
        {metrics.map((m) => (
          <div key={m.label} className="bg-gray-800 rounded p-2 text-center">
            <div className="text-gray-500 text-xs mb-1">{m.label}</div>
            <div className={clsx('text-xl font-bold font-mono', m.color)}>{m.value}</div>
          </div>
        ))}
      </div>
      <div className="mt-2 text-xs text-gray-600">
        Consistency = fraction of OOS folds with Sharpe &gt; 0 -- Degradation = OOS/IS Sharpe ratio
      </div>
    </div>
  );
};

const FoldTimeline: React.FC<{ folds: WFFold[] }> = ({ folds }) => {
  const data = folds.map((f) => ({
    fold: `F${f.fold}`,
    is: 90, -- IS bar length in units
    oos: 30, -- OOS bar length
  }));

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Fold Timeline (Train / Test)
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 10, bottom: 0, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
          <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 10 }} label={{ value: 'Days', fill: '#6b7280', fontSize: 9, position: 'insideRight' }} />
          <YAxis type="category" dataKey="fold" tick={{ fill: '#9ca3af', fontSize: 9 }} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="is" name="In-Sample" stackId="a" fill="#3b82f6" opacity={0.8} />
          <Bar dataKey="oos" name="Out-of-Sample" stackId="a" fill="#22c55e" opacity={0.8} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

const ParameterStabilityChart: React.FC<{
  stability: { name: string; values: number[]; std: number; cv: number }[];
}> = ({ stability }) => {
  -- Build chart data: one series per param, x = fold index
  const N = stability[0]?.values.length ?? 0;
  const data = Array.from({ length: N }, (_, i) => {
    const pt: Record<string, number | string> = { fold: `F${i + 1}` };
    stability.forEach((s) => { pt[s.name] = s.values[i]; });
    return pt;
  });

  const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#a78bfa'];

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Parameter Stability Across Folds
      </h3>
      <ResponsiveContainer width="100%" height={170}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="fold" tick={{ fill: '#9ca3af', fontSize: 10 }} />
          <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
            formatter={(v: number) => v.toFixed(4)}
          />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {stability.map((s, i) => (
            <Line key={s.name} type="monotone" dataKey={s.name} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={1.5} name={s.name} />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="grid grid-cols-4 gap-2 mt-2">
        {stability.map((s, i) => (
          <div key={s.name} className="text-xs bg-gray-800 rounded px-2 py-1">
            <div style={{ color: COLORS[i % COLORS.length] }} className="font-mono truncate">{s.name}</div>
            <div className="text-gray-400">CV={fmtPct(s.cv)}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

const OOSSharpeDistribution: React.FC<{ folds: WFFold[] }> = ({ folds }) => {
  -- Build histogram with 10 bins from -0.5 to 2.5
  const bins = 12;
  const minS = -0.5; const maxS = 2.5;
  const binW = (maxS - minS) / bins;
  const counts = Array(bins).fill(0);
  folds.forEach((f) => {
    const b = Math.floor((f.oos_sharpe - minS) / binW);
    if (b >= 0 && b < bins) counts[b]++;
  });
  const histData = counts.map((count, i) => ({
    range: `${(minS + i * binW).toFixed(1)}`,
    count,
    above1: minS + i * binW >= 1.0,
  }));

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        OOS Sharpe Distribution
      </h3>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={histData} margin={{ top: 4, right: 8, bottom: 10, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="range" tick={{ fill: '#9ca3af', fontSize: 9 }} label={{ value: 'Sharpe', fill: '#6b7280', fontSize: 9, position: 'insideBottom', offset: -4 }} />
          <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} allowDecimals={false} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
          />
          <Bar dataKey="count" name="Folds" radius={[2, 2, 0, 0]}>
            {histData.map((d, i) => (
              <Cell key={i} fill={d.above1 ? '#22c55e' : d.range >= '0' ? '#eab308' : '#ef4444'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

// ---- Main Page ----

export function WalkForward() {
  const { data, isLoading, error } = useQuery<WFResults>({
    queryKey: ['walkforward-results'],
    queryFn: async () => {
      const res = await fetch('/api/walkforward/results');
      if (!res.ok) throw new Error('API unavailable');
      return res.json();
    },
    retry: false,
  });

  -- Fall back to demo data if API not available
  const results = useMemo(() => {
    if (data) return data;
    return buildDemoResults();
  }, [data]);

  return (
    <div className="min-h-screen bg-gray-950 text-white p-4">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-lg font-bold tracking-wide">Walk-Forward Optimization</h1>
        <div className="flex items-center gap-4 text-xs">
          {isLoading && <span className="text-yellow-400 animate-pulse">Loading...</span>}
          {error && <span className="text-orange-400">Demo data (API offline)</span>}
          {data && <span className="text-green-400">Live data</span>}
          <span className="text-gray-500">{results.folds.length} folds</span>
        </div>
      </div>

      {/* Row 1: Grid + Metrics */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <WalkForwardGrid folds={results.folds} />
        <div className="flex flex-col gap-3">
          <ConsistencyMetrics
            is_mean={results.is_mean_sharpe}
            oos_mean={results.oos_mean_sharpe}
            consistency_ratio={results.consistency_ratio}
            degradation_factor={results.degradation_factor}
          />
          <OOSSharpeDistribution folds={results.folds} />
        </div>
      </div>

      {/* Row 2: Timeline + Parameter stability */}
      <div className="grid grid-cols-2 gap-3">
        <FoldTimeline folds={results.folds} />
        <ParameterStabilityChart stability={results.param_stability} />
      </div>
    </div>
  );
}

export default WalkForward;
