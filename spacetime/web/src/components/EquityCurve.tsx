import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import type { EquityPoint, Regime } from '../types';

interface Props {
  data: EquityPoint[];
  height?: number;
  showBenchmark?: boolean;
  showRegimeColor?: boolean;
  showDrawdown?: boolean;
  startEquity?: number;
}

const REGIME_COLORS: Record<Regime, string> = {
  BULL: '#22c55e',
  BEAR: '#ef4444',
  SIDEWAYS: '#9ca3af',
  HIGH_VOL: '#a855f7',
};

function fmtDate(ts: string) {
  try {
    return new Date(ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return ts;
  }
}

function fmtCurrency(v: number) {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}K`;
  return `$${v.toFixed(0)}`;
}

// Custom dot to color by regime
function RegimeDot(props: {
  cx?: number; cy?: number; payload?: EquityPoint; showRegimeColor?: boolean;
}) {
  const { cx, cy, payload, showRegimeColor } = props;
  if (!showRegimeColor || !payload?.regime) return null;
  const color = REGIME_COLORS[payload.regime];
  return <circle cx={cx} cy={cy} r={3} fill={color} opacity={0.7} />;
}

function CustomTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number; name: string; color: string; payload: EquityPoint }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const eq = payload.find(p => p.name === 'equity');
  const dd = payload.find(p => p.name === 'drawdown');
  const bm = payload.find(p => p.name === 'benchmark');
  const regime = eq?.payload?.regime;

  return (
    <div className="bg-bg-elevated border border-bg-border rounded-lg p-3 text-xs font-mono shadow-lg">
      <p className="text-gray-400 mb-1">{fmtDate(label ?? '')}</p>
      {eq && <p style={{ color: regime ? REGIME_COLORS[regime] : '#6366f1' }}>
        Equity: {fmtCurrency(eq.value)}
      </p>}
      {dd && <p className="text-bear">Drawdown: {dd.value.toFixed(2)}%</p>}
      {bm && <p className="text-sideways">Benchmark: {fmtCurrency(bm.value)}</p>}
      {regime && (
        <p style={{ color: REGIME_COLORS[regime] }} className="mt-1">
          Regime: {regime}
        </p>
      )}
    </div>
  );
}

export function EquityCurve({
  data, height = 300, showBenchmark = false, showRegimeColor = true, showDrawdown = false,
}: Props) {
  if (!data.length) {
    return (
      <div
        className="flex items-center justify-center text-gray-600 text-sm font-mono"
        style={{ height }}
      >
        No equity data
      </div>
    );
  }

  // Thin data for perf if too many points
  const display = data.length > 500
    ? data.filter((_, i) => i % Math.ceil(data.length / 500) === 0)
    : data;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={display} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
        <XAxis
          dataKey="date"
          tickFormatter={fmtDate}
          tick={{ fill: '#6b7280', fontSize: 11, fontFamily: 'JetBrains Mono' }}
          axisLine={{ stroke: '#2a2a3a' }}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tickFormatter={fmtCurrency}
          tick={{ fill: '#6b7280', fontSize: 11, fontFamily: 'JetBrains Mono' }}
          axisLine={{ stroke: '#2a2a3a' }}
          tickLine={false}
          width={70}
        />
        {showDrawdown && (
          <YAxis
            yAxisId="dd"
            orientation="right"
            tickFormatter={v => `${v.toFixed(1)}%`}
            tick={{ fill: '#6b7280', fontSize: 11, fontFamily: 'JetBrains Mono' }}
            axisLine={{ stroke: '#2a2a3a' }}
            tickLine={false}
            width={55}
          />
        )}
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 11, fontFamily: 'JetBrains Mono', color: '#9ca3af' }}
        />
        <ReferenceLine y={display[0]?.equity ?? 0} stroke="#2a2a3a" strokeDasharray="4 4" />
        <Line
          type="monotone"
          dataKey="equity"
          stroke="#6366f1"
          strokeWidth={1.5}
          dot={showRegimeColor ? <RegimeDot showRegimeColor /> : false}
          activeDot={{ r: 4, fill: '#6366f1' }}
          isAnimationActive={false}
        />
        {showBenchmark && (
          <Line
            type="monotone"
            dataKey="benchmark"
            stroke="#9ca3af"
            strokeWidth={1}
            strokeDasharray="4 4"
            dot={false}
            isAnimationActive={false}
          />
        )}
        {showDrawdown && (
          <Line
            type="monotone"
            dataKey="drawdown"
            yAxisId="dd"
            stroke="#ef4444"
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
