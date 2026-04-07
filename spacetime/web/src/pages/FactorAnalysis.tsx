// ============================================================
// FactorAnalysis.tsx -- Factor attribution analysis
// BH/Nav/Hurst/Vol/Momentum/Value/Size factor decomposition
// ============================================================
import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, AreaChart, Area, LineChart,
  Line, Legend,
} from 'recharts';
import { clsx } from 'clsx';

// ---- Types ----

type FactorName = 'BH' | 'Nav' | 'Hurst' | 'Vol' | 'Momentum' | 'Value' | 'Size';

interface FactorExposure {
  factor: FactorName;
  exposure: number;
  t_stat: number;
  p_value: number;
  r2_contribution: number;
}

interface FactorReturnPoint {
  date: string;
  BH: number;
  Nav: number;
  Hurst: number;
  Vol: number;
  Momentum: number;
  Value: number;
  Size: number;
  Total: number;
}

interface ICPoint {
  date: string;
  BH: number;
  Nav: number;
  Hurst: number;
  Vol: number;
  Momentum: number;
}

interface FamaMacBethResult {
  factor: FactorName;
  lambda: number;     -- risk premium estimate
  t_stat: number;
  p_value: number;
  significant: boolean;
}

interface FactorAttributionData {
  exposures: FactorExposure[];
  return_decomp: FactorReturnPoint[];
  ic_series: ICPoint[];
  factor_corr: number[][];  -- 7x7 matrix
  fama_macbeth: FamaMacBethResult[];
}

// ---- Constants ----

const FACTORS: FactorName[] = ['BH', 'Nav', 'Hurst', 'Vol', 'Momentum', 'Value', 'Size'];

const FACTOR_COLORS: Record<FactorName, string> = {
  BH: '#8b5cf6',
  Nav: '#3b82f6',
  Hurst: '#22c55e',
  Vol: '#ef4444',
  Momentum: '#f59e0b',
  Value: '#06b6d4',
  Size: '#ec4899',
};

// ---- Demo data ----

function buildDemoData(): FactorAttributionData {
  const exposures: FactorExposure[] = FACTORS.map((f, i) => {
    const t_stat = (Math.sin(i * 1.3 + 0.5) * 3.5) + 0.2;
    return {
      factor: f,
      exposure: 0.05 + Math.sin(i * 0.9) * 0.15,
      t_stat,
      p_value: Math.max(0.001, 2 * (1 - Math.min(0.9999, Math.abs(t_stat) * 0.15))),
      r2_contribution: Math.max(0.01, Math.abs(Math.sin(i * 1.1)) * 0.12),
    };
  });

  -- Build 120 daily return decomposition points
  const return_decomp: FactorReturnPoint[] = [];
  const cumRets: Record<string, number> = {};
  FACTORS.forEach((f) => { cumRets[f] = 0; });
  cumRets['Total'] = 0;

  for (let d = 0; d < 120; d++) {
    const date = `2025-${String(Math.floor(d / 30) + 1).padStart(2, '0')}-${String((d % 30) + 1).padStart(2, '0')}`;
    FACTORS.forEach((f, i) => {
      const daily = (Math.sin(d * 0.1 + i * 0.7) * 0.006 + 0.0003);
      cumRets[f] += daily;
    });
    cumRets['Total'] = FACTORS.reduce((s, f) => s + cumRets[f], 0);
    return_decomp.push({
      date,
      BH: cumRets['BH'],
      Nav: cumRets['Nav'],
      Hurst: cumRets['Hurst'],
      Vol: cumRets['Vol'],
      Momentum: cumRets['Momentum'],
      Value: cumRets['Value'],
      Size: cumRets['Size'],
      Total: cumRets['Total'],
    });
  }

  -- IC series for top 5 factors
  const ic_series: ICPoint[] = Array.from({ length: 60 }, (_, d) => ({
    date: `2025-${String(Math.floor(d / 20) + 2).padStart(2, '0')}-${String((d % 20) + 1).padStart(2, '0')}`,
    BH: 0.08 + Math.sin(d * 0.3) * 0.06,
    Nav: 0.06 + Math.cos(d * 0.25) * 0.05,
    Hurst: 0.04 + Math.sin(d * 0.4) * 0.04,
    Vol: -0.02 + Math.cos(d * 0.35) * 0.04,
    Momentum: 0.05 + Math.sin(d * 0.2) * 0.05,
  }));

  -- 7x7 correlation matrix
  const factor_corr: number[][] = Array.from({ length: 7 }, (_, r) =>
    Array.from({ length: 7 }, (_, c) => {
      if (r === c) return 1.0;
      const corr = Math.sin((r + 1) * (c + 1) * 0.5) * 0.6;
      return parseFloat(corr.toFixed(3));
    })
  );

  const fama_macbeth: FamaMacBethResult[] = FACTORS.map((f, i) => {
    const t_stat = Math.sin(i * 1.7 + 1.2) * 3.0;
    const p_value = Math.max(0.001, 0.3 * Math.exp(-Math.abs(t_stat) * 0.5));
    return {
      factor: f,
      lambda: Math.sin(i * 0.9) * 0.008,
      t_stat,
      p_value,
      significant: Math.abs(t_stat) > 1.96,
    };
  });

  return { exposures, return_decomp, ic_series, factor_corr, fama_macbeth };
}

// ---- Helpers ----

function fmt3(v: number) { return v.toFixed(3); }
function fmt4(v: number) { return v.toFixed(4); }
function corrColor(c: number): string {
  if (c >= 0) {
    const alpha = Math.min(1, c * 0.9 + 0.1);
    return `rgba(34,197,94,${alpha})`;
  }
  const alpha = Math.min(1, -c * 0.9 + 0.1);
  return `rgba(239,68,68,${alpha})`;
}

// ---- Sub-components ----

const FactorExposureBar: React.FC<{ exposures: FactorExposure[] }> = ({ exposures }) => {
  const sorted = [...exposures].sort((a, b) => b.exposure - a.exposure);
  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Factor Exposures (beta)
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={sorted} layout="vertical" margin={{ top: 0, right: 40, bottom: 0, left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
          <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 10 }} />
          <YAxis type="category" dataKey="factor" tick={{ fill: '#9ca3af', fontSize: 10 }} width={60} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
            formatter={(v: number, name: string) => [v.toFixed(4), name]}
          />
          <Bar dataKey="exposure" name="Exposure" radius={[0, 3, 3, 0]}>
            {sorted.map((d) => (
              <Cell key={d.factor} fill={FACTOR_COLORS[d.factor]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

const FactorReturnDecomposition: React.FC<{ data: FactorReturnPoint[] }> = ({ data }) => (
  <div className="bg-gray-900 border border-gray-700 rounded p-3">
    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
      Cumulative Return Decomposition by Factor
    </h3>
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <defs>
          {FACTORS.map((f) => (
            <linearGradient key={f} id={`grad-${f}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={FACTOR_COLORS[f]} stopOpacity={0.4} />
              <stop offset="95%" stopColor={FACTOR_COLORS[f]} stopOpacity={0.05} />
            </linearGradient>
          ))}
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 9 }} tickCount={6} />
        <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
        <Tooltip
          contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
          formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
        />
        <Legend wrapperStyle={{ fontSize: 10 }} />
        {FACTORS.map((f) => (
          <Area
            key={f}
            type="monotone"
            dataKey={f}
            stroke={FACTOR_COLORS[f]}
            fill={`url(#grad-${f})`}
            strokeWidth={1.5}
            dot={false}
            stackId="factors"
            name={f}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  </div>
);

const ICTimeSeries: React.FC<{ data: ICPoint[] }> = ({ data }) => {
  const IC_FACTORS: (keyof ICPoint)[] = ['BH', 'Nav', 'Hurst', 'Vol', 'Momentum'];
  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        IC Time Series -- Top 5 Factors
      </h3>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 9 }} tickCount={5} />
          <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)} />
          <Tooltip
            contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
            formatter={(v: number) => v.toFixed(4)}
          />
          <Legend wrapperStyle={{ fontSize: 10 }} />
          {IC_FACTORS.map((f) => (
            <Line
              key={f}
              type="monotone"
              dataKey={f}
              stroke={FACTOR_COLORS[f as FactorName]}
              dot={false}
              strokeWidth={1.5}
              name={f}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

const FactorCorrelationMatrix: React.FC<{ corr: number[][]; factors: FactorName[] }> = ({ corr, factors }) => {
  const [hovered, setHovered] = useState<[number, number] | null>(null);
  const CELL = 38;
  const PAD = 50;
  const W = factors.length * CELL + PAD;
  const H = factors.length * CELL + PAD;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Factor Correlation Matrix
      </h3>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
        {/* Column labels */}
        {factors.map((f, i) => (
          <text
            key={`col-${f}`}
            x={PAD + i * CELL + CELL / 2}
            y={PAD - 4}
            textAnchor="middle"
            fill={FACTOR_COLORS[f]}
            fontSize={9}
            fontWeight="bold"
          >
            {f}
          </text>
        ))}
        {/* Row labels */}
        {factors.map((f, i) => (
          <text
            key={`row-${f}`}
            x={PAD - 4}
            y={PAD + i * CELL + CELL / 2 + 3}
            textAnchor="end"
            fill={FACTOR_COLORS[f]}
            fontSize={9}
            fontWeight="bold"
          >
            {f}
          </text>
        ))}
        {/* Cells */}
        {corr.map((row, r) =>
          row.map((val, c) => {
            const isHovered = hovered && hovered[0] === r && hovered[1] === c;
            return (
              <g
                key={`${r}-${c}`}
                onMouseEnter={() => setHovered([r, c])}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: 'default' }}
              >
                <rect
                  x={PAD + c * CELL}
                  y={PAD + r * CELL}
                  width={CELL - 1}
                  height={CELL - 1}
                  fill={corrColor(val)}
                  stroke={isHovered ? '#ffffff' : 'transparent'}
                  strokeWidth={1.5}
                />
                <text
                  x={PAD + c * CELL + CELL / 2}
                  y={PAD + r * CELL + CELL / 2 + 3}
                  textAnchor="middle"
                  fill={Math.abs(val) > 0.5 ? 'white' : '#9ca3af'}
                  fontSize={8}
                  fontWeight={r === c ? 'bold' : 'normal'}
                >
                  {val.toFixed(2)}
                </text>
              </g>
            );
          })
        )}
      </svg>
      <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 inline-block rounded" style={{ background: 'rgba(34,197,94,0.9)' }} />
          +1.0
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 inline-block rounded" style={{ background: 'rgba(239,68,68,0.9)' }} />
          -1.0
        </span>
        {hovered && (
          <span className="text-white ml-2">
            {factors[hovered[0]]} / {factors[hovered[1]]}: {corr[hovered[0]][hovered[1]].toFixed(4)}
          </span>
        )}
      </div>
    </div>
  );
};

const FamaMacBethResults: React.FC<{ results: FamaMacBethResult[] }> = ({ results }) => (
  <div className="bg-gray-900 border border-gray-700 rounded p-3">
    <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
      Fama-MacBeth Regression Results
    </h3>
    <table className="w-full text-xs">
      <thead>
        <tr className="text-gray-500 border-b border-gray-700">
          <th className="text-left py-1 pr-3">Factor</th>
          <th className="text-right pr-3">Lambda (risk premium)</th>
          <th className="text-right pr-3">t-stat</th>
          <th className="text-right pr-3">p-value</th>
          <th className="text-center">Sig.</th>
        </tr>
      </thead>
      <tbody>
        {results.map((r) => (
          <tr key={r.factor} className="border-b border-gray-800 hover:bg-gray-800">
            <td className="py-1.5 pr-3 font-bold" style={{ color: FACTOR_COLORS[r.factor] }}>
              {r.factor}
            </td>
            <td className={clsx('text-right pr-3 font-mono', r.lambda >= 0 ? 'text-green-400' : 'text-red-400')}>
              {r.lambda >= 0 ? '+' : ''}{fmt4(r.lambda)}
            </td>
            <td className={clsx('text-right pr-3 font-mono', Math.abs(r.t_stat) > 1.96 ? 'text-white' : 'text-gray-400')}>
              {fmt3(r.t_stat)}
            </td>
            <td className={clsx('text-right pr-3 font-mono', r.p_value < 0.05 ? 'text-green-400' : 'text-gray-400')}>
              {r.p_value < 0.001 ? '<0.001' : fmt3(r.p_value)}
            </td>
            <td className="text-center">
              {r.significant ? (
                <span className="text-green-400 font-bold">***</span>
              ) : (
                <span className="text-gray-600">ns</span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
    <div className="mt-2 text-xs text-gray-600">
      *** p&lt;0.05 -- lambda = avg monthly risk premium -- t-stat uses Newey-West std errors
    </div>
  </div>
);

// ---- Main Page ----

export function FactorAnalysis() {
  const { data, isLoading, error } = useQuery<FactorAttributionData>({
    queryKey: ['factor-attribution'],
    queryFn: async () => {
      const res = await fetch('/api/factors/attribution');
      if (!res.ok) throw new Error('API unavailable');
      return res.json();
    },
    retry: false,
    staleTime: 30_000,
  });

  const results = useMemo(() => {
    if (data) return data;
    return buildDemoData();
  }, [data]);

  return (
    <div className="min-h-screen bg-gray-950 text-white p-4">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-lg font-bold tracking-wide">Factor Attribution Analysis</h1>
        <div className="flex items-center gap-4 text-xs">
          {isLoading && <span className="text-yellow-400 animate-pulse">Loading...</span>}
          {error && <span className="text-orange-400">Demo data (API offline)</span>}
          {data && <span className="text-green-400">Live data</span>}
          <div className="flex items-center gap-2">
            {FACTORS.map((f) => (
              <div key={f} className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full" style={{ background: FACTOR_COLORS[f] }} />
                <span className="text-gray-400">{f}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Row 1: Exposure bar + Return decomp */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <FactorExposureBar exposures={results.exposures} />
        <FactorReturnDecomposition data={results.return_decomp} />
      </div>

      {/* Row 2: IC time series + Correlation matrix */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <ICTimeSeries data={results.ic_series} />
        <FactorCorrelationMatrix corr={results.factor_corr} factors={FACTORS} />
      </div>

      {/* Row 3: Fama-MacBeth */}
      <FamaMacBethResults results={results.fama_macbeth} />
    </div>
  );
}

export default FactorAnalysis;
