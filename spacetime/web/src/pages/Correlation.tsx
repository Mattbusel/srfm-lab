import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import type { CorrelationResult } from '../types';

function corrToColor(v: number): string {
  // v in [-1, 1] → blue (low/neg) to red (high/pos)
  const normalized = (v + 1) / 2; // 0..1
  const r = Math.round(normalized * 239 + (1 - normalized) * 59);
  const g = Math.round((1 - Math.abs(v)) * 80);
  const b = Math.round((1 - normalized) * 239 + normalized * 68);
  return `rgb(${r},${g},${b})`;
}

function textOnColor(v: number): string {
  return Math.abs(v) > 0.5 ? '#ffffff' : '#9ca3af';
}

function HeatmapMatrix({
  instruments,
  matrix,
  title,
}: {
  instruments: string[];
  matrix: number[][];
  title: string;
}) {
  if (!matrix.length || !instruments.length) return null;

  return (
    <div className="bg-bg-card border border-bg-border rounded-xl p-4">
      <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">{title}</h3>
      <div className="overflow-x-auto">
        <table className="text-xs font-mono border-collapse">
          <thead>
            <tr>
              <th className="w-12" />
              {instruments.map(sym => (
                <th key={sym} className="px-1 pb-2 text-center text-gray-400 font-semibold" style={{ minWidth: '48px' }}>
                  {sym}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {instruments.map((rowSym, ri) => (
              <tr key={rowSym}>
                <td className="pr-2 py-0.5 text-right text-gray-400 font-semibold">{rowSym}</td>
                {instruments.map((colSym, ci) => {
                  const val = matrix[ri]?.[ci] ?? 0;
                  return (
                    <td key={colSym} className="p-0.5">
                      <div
                        className="w-11 h-9 rounded flex items-center justify-center font-mono text-xs font-semibold"
                        style={{
                          backgroundColor: corrToColor(val),
                          color: textOnColor(val),
                        }}
                        title={`${rowSym} vs ${colSym}: ${val.toFixed(3)}`}
                      >
                        {val.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Color scale */}
      <div className="mt-3 flex items-center gap-2">
        <span className="text-xs font-mono text-gray-600">-1.0</span>
        <div className="flex-1 h-2 rounded" style={{
          background: 'linear-gradient(to right, rgb(59,68,239), rgb(80,80,80), rgb(239,68,68))',
        }} />
        <span className="text-xs font-mono text-gray-600">+1.0</span>
      </div>
    </div>
  );
}

// Simple SVG dendrogram
function Dendrogram({ clusters, instruments }: { clusters: string[][], instruments: string[] }) {
  if (!clusters.length) return null;

  const allMapped = clusters.flat();
  const unmapped = instruments.filter(i => !allMapped.includes(i));
  const allClusters = [...clusters, ...unmapped.map(i => [i])];

  return (
    <div className="bg-bg-card border border-bg-border rounded-xl p-4">
      <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">Instrument Clusters</h3>
      <div className="space-y-2">
        {allClusters.map((cluster, ci) => (
          <div key={ci} className="flex items-center gap-2 flex-wrap">
            <div className="w-2 h-2 rounded-full" style={{
              backgroundColor: `hsl(${(ci * 137.5) % 360}, 60%, 55%)`,
            }} />
            {cluster.map(sym => (
              <span key={sym} className="text-xs font-mono bg-bg-elevated border border-bg-border rounded px-2 py-0.5 text-gray-300">
                {sym}
              </span>
            ))}
            {cluster.length > 1 && (
              <span className="text-xs font-mono text-gray-600">← highly correlated</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function Correlation() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['correlation'],
    queryFn: api.correlation,
    staleTime: 5 * 60_000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-3 text-gray-500 font-mono text-sm">
          <span className="w-5 h-5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          Loading correlation matrix…
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="bg-bear/10 border border-bear/30 rounded-xl p-6 text-center">
        <p className="text-bear font-mono text-sm">Failed to load: {(error as Error).message}</p>
      </div>
    );
  }

  if (!data) return null;

  const optPort = data.optimal_portfolio;

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Heatmaps */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <HeatmapMatrix
          instruments={data.instruments}
          matrix={data.jaccard}
          title="Jaccard Similarity — BH Activation Co-occurrence"
        />
        <HeatmapMatrix
          instruments={data.instruments}
          matrix={data.pearson}
          title="Pearson Correlation — Return Series"
        />
      </div>

      {/* Dendrogram + Optimal Portfolio */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Dendrogram clusters={data.clusters} instruments={data.instruments} />

        {/* Optimal portfolio */}
        <div className="bg-bg-card border border-bg-border rounded-xl p-4">
          <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">
            Minimum Correlation Portfolio
          </h3>
          <div className="space-y-2 mb-4">
            {Object.entries(optPort.weights)
              .sort(([, a], [, b]) => b - a)
              .map(([sym, w]) => (
                <div key={sym} className="space-y-0.5">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-gray-300 font-semibold">{sym}</span>
                    <span className="text-accent">{(w * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1.5 bg-bg-border rounded-full">
                    <div className="h-full rounded-full bg-accent" style={{ width: `${w * 100}%` }} />
                  </div>
                </div>
              ))}
          </div>

          <div className="pt-3 border-t border-bg-border grid grid-cols-2 gap-3">
            <div className="bg-bg-elevated rounded-lg p-2.5">
              <p className="text-xs font-mono text-gray-500">Diversification Score</p>
              <p className="text-lg font-mono font-bold text-bull mt-1">
                {(optPort.diversification_score * 100).toFixed(1)}
              </p>
            </div>
            <div className="bg-bg-elevated rounded-lg p-2.5">
              <p className="text-xs font-mono text-gray-500">Expected Correlation</p>
              <p className="text-lg font-mono font-bold text-gray-200 mt-1">
                {optPort.expected_correlation.toFixed(3)}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
