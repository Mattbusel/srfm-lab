import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '../api';
import type { SensitivityResult } from '../types';

const PERTURBATIONS = [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5];
const PERTURBATION_LABELS = ['-50%', '-25%', '-10%', 'Base', '+10%', '+25%', '+50%'];

function sharpeToColor(sharpe: number, base: number): string {
  const normalized = sharpe / Math.max(Math.abs(base) * 2, 1);
  if (sharpe >= base * 1.1) return `rgba(34,197,94,${Math.min(0.8, normalized)})`;
  if (sharpe >= base * 0.9) return `rgba(99,102,241,0.5)`;
  if (sharpe >= 0) return `rgba(234,179,8,${0.3 + normalized * 0.3})`;
  return `rgba(239,68,68,${Math.min(0.8, Math.abs(normalized))})`;
}

function sharpeTextColor(sharpe: number, base: number): string {
  if (sharpe >= base * 1.1) return '#22c55e';
  if (sharpe >= base * 0.9) return '#e5e7eb';
  if (sharpe >= 0) return '#eab308';
  return '#ef4444';
}

export function Sensitivity() {
  const instrumentsQuery = useQuery({
    queryKey: ['instruments'],
    queryFn: api.instruments,
    staleTime: 60_000,
  });

  const [sym, setSym] = useState('SPY');
  const [source, setSource] = useState('yfinance');
  const [paramValues, setParamValues] = useState({
    cf: 0.5,
    bh_form: 1.35,
    bh_decay: 0.95,
    bh_collapse: 0.7,
  });
  const [result, setResult] = useState<SensitivityResult | null>(null);

  const sensMut = useMutation({
    mutationFn: api.sensitivity,
    onSuccess: setResult,
  });

  const syms = instrumentsQuery.data?.map(i => i.sym) ?? ['SPY', 'QQQ', 'BTC', 'ETH'];

  function handleRun() {
    sensMut.mutate({ sym, source, params: paramValues });
  }

  const params = Object.keys(paramValues) as Array<keyof typeof paramValues>;

  // Build heatmap matrix: rows = perturbations, cols = params
  function getCellValue(param: string, perturbation: number): SensitivityResult['cells'][0] | undefined {
    return result?.cells.find(c => c.param === param && Math.abs(c.perturbation - perturbation) < 0.001);
  }

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Config */}
      <div className="bg-bg-card border border-bg-border rounded-xl p-5">
        <h2 className="text-sm font-mono font-semibold text-gray-400 uppercase tracking-wider mb-4">
          Parameter Sensitivity Analysis
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Instrument</label>
            <select
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={sym}
              onChange={e => setSym(e.target.value)}
            >
              {syms.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Data Source</label>
            <select
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={source}
              onChange={e => setSource(e.target.value)}
            >
              {['yfinance', 'alpaca', 'csv'].map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>

        {/* Param inputs */}
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
          {params.map(p => (
            <div key={p}>
              <label className="block text-xs font-mono text-gray-500 mb-1">{p} (base)</label>
              <input
                type="number" step="0.01"
                className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
                value={paramValues[p]}
                onChange={e => setParamValues(prev => ({ ...prev, [p]: parseFloat(e.target.value) || 0 }))}
              />
            </div>
          ))}
        </div>

        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={handleRun}
            disabled={sensMut.isPending}
            className="px-6 py-2.5 bg-accent hover:bg-accent-hover disabled:opacity-50 rounded-lg text-sm font-mono font-semibold text-white transition-colors"
          >
            {sensMut.isPending ? (
              <span className="flex items-center gap-2">
                <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Analyzing…
              </span>
            ) : '▶ Run Sensitivity'}
          </button>
          {sensMut.isError && (
            <span className="text-xs font-mono text-bear">{(sensMut.error as Error).message}</span>
          )}
        </div>
      </div>

      {result && (
        <>
          {/* Heatmap */}
          <div className="bg-bg-card border border-bg-border rounded-xl p-5">
            <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-4">
              Sharpe Ratio Heatmap — Base Sharpe: {result.base_sharpe.toFixed(3)}
            </h3>
            <div className="overflow-x-auto">
              <table className="text-xs font-mono">
                <thead>
                  <tr>
                    <th className="text-right pr-4 pb-2 text-gray-500 font-normal">Perturbation</th>
                    {params.map(p => (
                      <th key={p} className="px-4 pb-2 text-center text-gray-400 font-semibold">
                        {p}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {PERTURBATIONS.map((pert, pi) => (
                    <tr key={pert}>
                      <td className={`text-right pr-4 py-1.5 font-mono ${pert === 0 ? 'text-accent font-semibold' : 'text-gray-500'}`}>
                        {PERTURBATION_LABELS[pi]}
                      </td>
                      {params.map(param => {
                        const cell = getCellValue(param, pert);
                        const sharpe = cell?.sharpe ?? result.base_sharpe;
                        return (
                          <td key={param} className="px-1 py-1">
                            <div
                              className="w-24 h-10 rounded flex flex-col items-center justify-center"
                              style={{ backgroundColor: sharpeToColor(sharpe, result.base_sharpe) }}
                            >
                              <span style={{ color: sharpeTextColor(sharpe, result.base_sharpe) }} className="font-semibold">
                                {sharpe.toFixed(3)}
                              </span>
                              {cell && (
                                <span className="text-gray-500 text-xs">
                                  {(cell.cagr * 100).toFixed(1)}%
                                </span>
                              )}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Legend */}
            <div className="mt-4 flex items-center gap-4 text-xs font-mono text-gray-500">
              <span>Color scale:</span>
              {[
                { color: 'rgba(239,68,68,0.7)', label: 'Fragile (Sharpe ↓)' },
                { color: 'rgba(234,179,8,0.5)', label: 'Sensitive' },
                { color: 'rgba(99,102,241,0.5)', label: 'Neutral (base)' },
                { color: 'rgba(34,197,94,0.7)', label: 'Robust (Sharpe ↑)' },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-1.5">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: color }} />
                  <span>{label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Fragility report */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-bg-card border border-bg-border rounded-xl p-5">
              <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">
                Fragile Parameters
              </h3>
              {result.fragile_params.length === 0 ? (
                <p className="text-sm font-mono text-bull">No fragile parameters detected</p>
              ) : (
                <div className="space-y-2">
                  {result.fragile_params.map(p => (
                    <div key={p} className="flex items-center gap-2 bg-bear/10 border border-bear/20 rounded-lg px-3 py-2">
                      <span className="text-bear text-lg">⚠</span>
                      <div>
                        <p className="text-sm font-mono font-semibold text-bear">{p}</p>
                        <p className="text-xs text-gray-500">High sensitivity — small perturbations degrade performance</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-bg-card border border-bg-border rounded-xl p-5">
              <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">
                Robust Parameters
              </h3>
              {result.robust_params.length === 0 ? (
                <p className="text-sm font-mono text-gray-500">No parameters classified as robust</p>
              ) : (
                <div className="space-y-2">
                  {result.robust_params.map(p => (
                    <div key={p} className="flex items-center gap-2 bg-bull/10 border border-bull/20 rounded-lg px-3 py-2">
                      <span className="text-bull text-lg">✓</span>
                      <div>
                        <p className="text-sm font-mono font-semibold text-bull">{p}</p>
                        <p className="text-xs text-gray-500">Low sensitivity — edge is robust to perturbation</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Edge summary */}
          <div className="bg-accent/5 border border-accent/30 rounded-xl p-5">
            <h3 className="text-sm font-mono font-semibold text-accent mb-2">Edge Analysis Summary</h3>
            <p className="text-sm font-mono text-gray-300">
              Base Sharpe: <span className="text-accent font-semibold">{result.base_sharpe.toFixed(3)}</span>
              {' · '}
              {result.robust_params.length > 0
                ? <>The edge lives primarily in <span className="text-bull">{result.robust_params.join(', ')}</span> — these are stable across parameter perturbations.</>
                : 'No clearly dominant parameters identified. Consider running on more data.'
              }
              {result.fragile_params.length > 0 && (
                <> Parameters <span className="text-bear">{result.fragile_params.join(', ')}</span> are fragile — use caution in live trading.</>
              )}
            </p>
          </div>
        </>
      )}

      {!result && !sensMut.isPending && (
        <div className="bg-bg-card border border-bg-border rounded-xl p-16 text-center">
          <div className="text-4xl mb-3 opacity-30">🔬</div>
          <p className="text-gray-500 font-mono text-sm">Run sensitivity analysis to see parameter heatmap</p>
        </div>
      )}
    </div>
  );
}
