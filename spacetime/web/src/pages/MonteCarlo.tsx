import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { api } from '../api';
import type { MCResult } from '../types';
import { MetricsCard } from '../components/MetricsCard';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

function buildFanData(result: MCResult): Array<Record<string, number>> {
  const { percentiles } = result;
  return percentiles.dates.map((date, i) => ({
    date: i,
    p5: percentiles.p5[i] ?? 0,
    p25: percentiles.p25[i] ?? 0,
    p50: percentiles.p50[i] ?? 0,
    p75: percentiles.p75[i] ?? 0,
    p95: percentiles.p95[i] ?? 0,
  }));
}

function buildDDHistogram(result: MCResult): Array<{ bucket: string; count: number }> {
  const dd = result.drawdown_dist;
  if (!dd.length) return [];
  const min = Math.min(...dd);
  const max = Math.max(...dd);
  const buckets = 20;
  const step = (max - min) / buckets || 0.01;
  const counts = new Array(buckets).fill(0);
  dd.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / step), buckets - 1);
    counts[idx]++;
  });
  return counts.map((count, i) => ({
    bucket: `${((min + i * step) * 100).toFixed(1)}%`,
    count,
  }));
}

export function MonteCarlo() {
  const [tradesJson, setTradesJson] = useState('');
  const [nSims, setNSims] = useState(10000);
  const [months, setMonths] = useState(12);
  const [regimeAware, setRegimeAware] = useState(true);
  const [result, setResult] = useState<MCResult | null>(null);

  const mcMut = useMutation({
    mutationFn: api.monteCarlo,
    onSuccess: setResult,
  });

  function handleRun() {
    if (!tradesJson.trim()) return;
    mcMut.mutate({ trades_json: tradesJson, n_sims: nSims, months, regime_aware: regimeAware });
  }

  const fanData = result ? buildFanData(result) : [];
  const ddHist = result ? buildDDHistogram(result) : [];

  function fmtK(v: number) {
    return v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v);
  }

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Config */}
      <div className="bg-bg-card border border-bg-border rounded-xl p-5">
        <h2 className="text-sm font-mono font-semibold text-gray-400 uppercase tracking-wider mb-4">
          Monte Carlo Simulation
        </h2>

        {/* Trades JSON */}
        <div className="mb-4">
          <label className="block text-xs font-mono text-gray-500 mb-1">Trades JSON (from backtest run)</label>
          <textarea
            className="w-full h-24 bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-xs font-mono text-gray-300 focus:outline-none focus:border-accent resize-none"
            placeholder='Paste trades JSON here, or run a backtest first and use "Run MC" from results…'
            value={tradesJson}
            onChange={e => setTradesJson(e.target.value)}
          />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Simulations</label>
            <input
              type="number"
              min={1000} max={100000} step={1000}
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={nSims}
              onChange={e => setNSims(Math.max(1000, parseInt(e.target.value) || 10000))}
            />
          </div>
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Months Forward</label>
            <input
              type="range" min={1} max={60}
              className="w-full mt-2 accent-accent"
              value={months}
              onChange={e => setMonths(parseInt(e.target.value))}
            />
            <p className="text-xs font-mono text-accent mt-0.5">{months} months</p>
          </div>
          <div className="flex items-center gap-3 pt-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <div
                className={`w-10 h-5 rounded-full relative transition-colors ${regimeAware ? 'bg-accent' : 'bg-bg-border'}`}
                onClick={() => setRegimeAware(!regimeAware)}
              >
                <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${regimeAware ? 'translate-x-5' : 'translate-x-0.5'}`} />
              </div>
              <span className="text-sm font-mono text-gray-300">Regime Aware</span>
            </label>
          </div>
          <div className="flex items-end">
            <button
              onClick={handleRun}
              disabled={mcMut.isPending || !tradesJson.trim()}
              className="w-full px-4 py-2.5 bg-accent hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-mono font-semibold text-white transition-colors"
            >
              {mcMut.isPending ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Simulating…
                </span>
              ) : `▶ Run ${fmtK(nSims)} Sims`}
            </button>
          </div>
        </div>
        {mcMut.isError && (
          <p className="mt-2 text-xs font-mono text-bear">{(mcMut.error as Error).message}</p>
        )}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Key metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricsCard label="Blowup Rate" value={`${(result.blowup_rate * 100).toFixed(2)}%`} trend={result.blowup_rate < 0.05 ? 'up' : 'down'} />
            <MetricsCard label="Kelly f*" value={result.kelly_f.toFixed(4)} accent />
            <MetricsCard label="Median Return" value={`${((result.percentiles.p50.at(-1) ?? 1) - 1) * 100 > 0 ? '+' : ''}${(((result.percentiles.p50.at(-1) ?? 1) - 1) * 100).toFixed(1)}%`} />
            <MetricsCard label="95th Pct Return" value={`${(((result.percentiles.p95.at(-1) ?? 1) - 1) * 100).toFixed(1)}%`} trend="up" />
          </div>

          {/* Fan chart */}
          <div className="bg-bg-card border border-bg-border rounded-xl p-4">
            <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">
              Equity Fan Chart — {fmtK(nSims)} Simulations × {months} Months
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={fanData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                <XAxis dataKey="date" tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} label={{ value: 'Months', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 11 }} />
                <YAxis tickFormatter={v => `${((v - 1) * 100).toFixed(0)}%`} tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
                <Tooltip
                  formatter={(v: number, name: string) => [`${((v - 1) * 100).toFixed(1)}%`, name]}
                  contentStyle={{ background: '#1a1a24', border: '1px solid #2a2a3a', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                />
                <Area type="monotone" dataKey="p95" stroke="#22c55e" fill="#22c55e" fillOpacity={0.08} strokeDasharray="3 3" strokeWidth={1} />
                <Area type="monotone" dataKey="p75" stroke="#6366f1" fill="#6366f1" fillOpacity={0.1} strokeWidth={1} />
                <Area type="monotone" dataKey="p50" stroke="#6366f1" fill="#6366f1" fillOpacity={0.15} strokeWidth={2} />
                <Area type="monotone" dataKey="p25" stroke="#f97316" fill="#f97316" fillOpacity={0.1} strokeWidth={1} />
                <Area type="monotone" dataKey="p5" stroke="#ef4444" fill="#ef4444" fillOpacity={0.08} strokeDasharray="3 3" strokeWidth={1} />
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono', paddingTop: '8px' }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* DD histogram */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-bg-card border border-bg-border rounded-xl p-4">
              <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">Max Drawdown Distribution</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={ddHist} margin={{ top: 5, right: 10, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                  <XAxis dataKey="bucket" tick={{ fill: '#6b7280', fontSize: 9 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} interval={4} angle={-45} textAnchor="end" />
                  <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#1a1a24', border: '1px solid #2a2a3a', fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                  <Bar dataKey="count" fill="#ef4444" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Kelly box */}
            <div className="bg-bg-card border border-bg-border rounded-xl p-4">
              <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">Kelly Criterion Analysis</h3>
              <div className="space-y-3 mt-4">
                <div>
                  <div className="flex justify-between text-xs font-mono mb-1">
                    <span className="text-gray-500">Full Kelly</span>
                    <span className="text-gray-200">{(result.kelly_f * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-bg-border rounded-full">
                    <div className="h-full rounded-full bg-accent" style={{ width: `${Math.min(result.kelly_f * 100, 100)}%` }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs font-mono mb-1">
                    <span className="text-gray-500">Half Kelly (recommended)</span>
                    <span className="text-accent">{(result.kelly_f * 50).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-bg-border rounded-full">
                    <div className="h-full rounded-full bg-accent/60" style={{ width: `${Math.min(result.kelly_f * 50, 100)}%` }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs font-mono mb-1">
                    <span className="text-gray-500">Quarter Kelly (conservative)</span>
                    <span className="text-sideways">{(result.kelly_f * 25).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-bg-border rounded-full">
                    <div className="h-full rounded-full bg-gray-500/60" style={{ width: `${Math.min(result.kelly_f * 25, 100)}%` }} />
                  </div>
                </div>

                <div className="pt-3 border-t border-bg-border space-y-2 text-xs font-mono">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Blowup probability</span>
                    <span className={result.blowup_rate < 0.05 ? 'text-bull' : 'text-bear'}>
                      {(result.blowup_rate * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">5th pct final equity</span>
                    <span className="text-bear">{(((result.percentiles.p5.at(-1) ?? 1) - 1) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">95th pct final equity</span>
                    <span className="text-bull">{(((result.percentiles.p95.at(-1) ?? 1) - 1) * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {!result && !mcMut.isPending && (
        <div className="bg-bg-card border border-bg-border rounded-xl p-16 text-center">
          <div className="text-4xl mb-3 opacity-30">🎲</div>
          <p className="text-gray-500 font-mono text-sm">Paste trades JSON and run simulations to see distribution</p>
        </div>
      )}
    </div>
  );
}
