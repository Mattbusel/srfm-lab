interface Props {
  label: string;
  value: string | number;
  sub?: string;
  trend?: 'up' | 'down' | 'neutral';
  accent?: boolean;
}

export function MetricsCard({ label, value, sub, trend, accent }: Props) {
  const trendIcon = trend === 'up' ? '▲' : trend === 'down' ? '▼' : null;
  const trendColor = trend === 'up' ? 'text-bull' : trend === 'down' ? 'text-bear' : 'text-sideways';

  return (
    <div className={`rounded-lg border p-4 ${accent ? 'border-accent/50 bg-accent/5' : 'border-bg-border bg-bg-card'}`}>
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">{label}</p>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-mono font-semibold text-gray-100">{value}</span>
        {trend && trendIcon && (
          <span className={`text-sm font-mono mb-0.5 ${trendColor}`}>{trendIcon}</span>
        )}
      </div>
      {sub && <p className="text-xs text-gray-500 mt-1 font-mono">{sub}</p>}
    </div>
  );
}
