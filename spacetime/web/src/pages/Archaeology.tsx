import { useState, useCallback } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '../api';
import type { Trade, Regime } from '../types';
import { TradeTable } from '../components/TradeTable';
import { RegimeBadge } from '../components/RegimeBadge';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from 'recharts';

const REGIMES: Regime[] = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOL'];
const REGIME_COLORS: Record<Regime, string> = {
  BULL: '#22c55e', BEAR: '#ef4444', SIDEWAYS: '#9ca3af', HIGH_VOL: '#a855f7',
};

function buildPnLHistogram(trades: Trade[]): Array<{ bucket: string; count: number; color: string }> {
  if (!trades.length) return [];
  const pnls = trades.map(t => t.pnl_pct);
  const min = Math.min(...pnls);
  const max = Math.max(...pnls);
  const buckets = 20;
  const step = (max - min) / buckets || 1;
  const counts = new Array(buckets).fill(0);
  pnls.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / step), buckets - 1);
    counts[idx]++;
  });
  return counts.map((count, i) => {
    const mid = min + (i + 0.5) * step;
    return {
      bucket: `${(mid * 100).toFixed(1)}%`,
      count,
      color: mid >= 0 ? '#22c55e' : '#ef4444',
    };
  });
}

export function Archaeology() {
  const [csvPath, setCsvPath] = useState('');
  const [runName, setRunName] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  // Filters for trade DB
  const [filterSym, setFilterSym] = useState('');
  const [filterRegime, setFilterRegime] = useState('');
  const [filterFromDate, setFilterFromDate] = useState('');
  const [filterToDate, setFilterToDate] = useState('');
  const [filterMinTF, setFilterMinTF] = useState('');

  const tradesQuery = useQuery({
    queryKey: ['trades', filterSym, filterRegime, filterFromDate, filterToDate, filterMinTF],
    queryFn: () => api.trades({
      sym: filterSym || undefined,
      regime: filterRegime || undefined,
      from_date: filterFromDate || undefined,
      to_date: filterToDate || undefined,
      min_tf_score: filterMinTF ? parseFloat(filterMinTF) : undefined,
    }),
    staleTime: 30_000,
  });

  const archMut = useMutation({
    mutationFn: ({ path, name }: { path: string; name: string }) =>
      api.archaeology(path, name),
  });

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      setCsvPath(file.name);
      if (!runName) setRunName(file.name.replace('.csv', ''));
    }
  }, [runName]);

  const trades = tradesQuery.data ?? [];

  // Analytics
  const winRateByRegime = REGIMES.map(r => {
    const rt = trades.filter(t => t.regime === r);
    const wins = rt.filter(t => t.pnl_pct > 0).length;
    return { regime: r, win_rate: rt.length ? wins / rt.length : 0, count: rt.length };
  }).filter(r => r.count > 0);

  const winRateByTFScore = Array.from({ length: 10 }, (_, i) => {
    const minScore = i * 0.1;
    const maxScore = minScore + 0.1;
    const bucket = trades.filter(t => t.tf_score >= minScore && t.tf_score < maxScore);
    const wins = bucket.filter(t => t.pnl_pct > 0).length;
    return {
      score: `${minScore.toFixed(1)}-${maxScore.toFixed(1)}`,
      win_rate: bucket.length ? (wins / bucket.length) * 100 : 0,
      count: bucket.length,
    };
  }).filter(b => b.count > 0);

  const pnlHist = buildPnLHistogram(trades);
  const totalWins = trades.filter(t => t.pnl_pct > 0).length;
  const totalWinRate = trades.length ? totalWins / trades.length : 0;
  const avgPnl = trades.length ? trades.reduce((s, t) => s + t.pnl_pct, 0) / trades.length : 0;

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Drop zone */}
      <div className="grid grid-cols-3 gap-4">
        <div
          className={`col-span-1 border-2 border-dashed rounded-xl p-6 text-center transition-colors cursor-pointer
            ${isDragging ? 'border-accent bg-accent/10' : 'border-bg-border hover:border-gray-600'}`}
          onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <div className="text-3xl mb-2 opacity-50">📂</div>
          <p className="text-sm font-mono text-gray-400">Drop CSV trade file here</p>
          <p className="text-xs font-mono text-gray-600 mt-1">QuantConnect format</p>
          {csvPath && (
            <p className="text-xs font-mono text-accent mt-2 truncate">{csvPath}</p>
          )}
        </div>

        <div className="col-span-2 bg-bg-card border border-bg-border rounded-xl p-4">
          <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">Load Trade Archive</h3>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-mono text-gray-500 mb-1">CSV Path</label>
              <input
                className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
                placeholder="/path/to/trades.csv"
                value={csvPath}
                onChange={e => setCsvPath(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-xs font-mono text-gray-500 mb-1">Run Name</label>
              <input
                className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
                placeholder="e.g. SPY_2020_2024"
                value={runName}
                onChange={e => setRunName(e.target.value)}
              />
            </div>
          </div>
          <button
            onClick={() => archMut.mutate({ path: csvPath, name: runName })}
            disabled={archMut.isPending || !csvPath || !runName}
            className="mt-3 px-4 py-2 bg-accent hover:bg-accent-hover disabled:opacity-40 rounded-lg text-sm font-mono font-semibold text-white transition-colors"
          >
            {archMut.isPending ? 'Loading…' : '▶ Load Archive'}
          </button>
          {archMut.isSuccess && (
            <p className="text-xs font-mono text-bull mt-2">
              Loaded {archMut.data.trades.length} trades as "{archMut.data.run_name}"
            </p>
          )}
        </div>
      </div>

      {/* Filters */}
      <div className="bg-bg-card border border-bg-border rounded-xl p-4">
        <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">Trade Database Filters</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Symbol</label>
            <input
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              placeholder="All"
              value={filterSym}
              onChange={e => setFilterSym(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Regime</label>
            <select
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={filterRegime}
              onChange={e => setFilterRegime(e.target.value)}
            >
              <option value="">All</option>
              {REGIMES.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">From</label>
            <input type="date"
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={filterFromDate}
              onChange={e => setFilterFromDate(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">To</label>
            <input type="date"
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={filterToDate}
              onChange={e => setFilterToDate(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Min TF Score</label>
            <input
              type="number" step="0.01" min="0" max="1"
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              placeholder="0.0"
              value={filterMinTF}
              onChange={e => setFilterMinTF(e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Statistical profile */}
      {trades.length > 0 && (
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-accent/5 border border-accent/30 rounded-xl p-4 col-span-3">
            <h3 className="text-xs font-mono text-accent uppercase tracking-wider mb-2">Statistical Profile — When LARSA Makes Money</h3>
            <div className="grid grid-cols-4 gap-4">
              <div>
                <p className="text-xs font-mono text-gray-500">Total Trades</p>
                <p className="text-xl font-mono font-bold text-gray-200">{trades.length}</p>
              </div>
              <div>
                <p className="text-xs font-mono text-gray-500">Overall Win Rate</p>
                <p className={`text-xl font-mono font-bold ${totalWinRate >= 0.5 ? 'text-bull' : 'text-bear'}`}>
                  {(totalWinRate * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-xs font-mono text-gray-500">Avg P&L per Trade</p>
                <p className={`text-xl font-mono font-bold ${avgPnl >= 0 ? 'text-bull' : 'text-bear'}`}>
                  {avgPnl >= 0 ? '+' : ''}{(avgPnl * 100).toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-xs font-mono text-gray-500">Best Regime</p>
                <p className="text-xl font-mono font-bold text-gray-200">
                  {winRateByRegime.sort((a, b) => b.win_rate - a.win_rate)[0]?.regime ?? '—'}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Analytics charts */}
      {trades.length > 0 && (
        <div className="grid grid-cols-3 gap-4">
          {/* Win rate by TF score */}
          <div className="bg-bg-card border border-bg-border rounded-xl p-4">
            <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">Win Rate by TF Score</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={winRateByTFScore} margin={{ top: 5, right: 5, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                <XAxis dataKey="score" tick={{ fill: '#6b7280', fontSize: 9 }} angle={-45} textAnchor="end" />
                <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} tickFormatter={v => `${v.toFixed(0)}%`} />
                <Tooltip
                  formatter={(v: number) => [`${v.toFixed(1)}%`, 'Win Rate']}
                  contentStyle={{ background: '#1a1a24', border: '1px solid #2a2a3a', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                />
                <Bar dataKey="win_rate" fill="#6366f1" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Win rate by regime */}
          <div className="bg-bg-card border border-bg-border rounded-xl p-4">
            <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">Win Rate by Regime</h3>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={winRateByRegime}
                  dataKey="win_rate"
                  nameKey="regime"
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  label={({ regime, win_rate }: { regime: Regime; win_rate: number }) => `${regime}: ${(win_rate * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {winRateByRegime.map(entry => (
                    <Cell key={entry.regime} fill={REGIME_COLORS[entry.regime]} fillOpacity={0.8} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, 'Win Rate']}
                  contentStyle={{ background: '#1a1a24', border: '1px solid #2a2a3a', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* P&L Distribution */}
          <div className="bg-bg-card border border-bg-border rounded-xl p-4">
            <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">P&L Distribution</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={pnlHist} margin={{ top: 5, right: 5, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                <XAxis dataKey="bucket" tick={{ fill: '#6b7280', fontSize: 8 }} angle={-45} textAnchor="end" interval={4} />
                <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#1a1a24', border: '1px solid #2a2a3a', fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                <Bar dataKey="count" isAnimationActive={false} radius={[2, 2, 0, 0]}>
                  {pnlHist.map((entry, i) => (
                    <Cell key={i} fill={entry.color} fillOpacity={0.7} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Trade table */}
      <div className="bg-bg-card border border-bg-border rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-bg-border flex items-center justify-between">
          <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider">
            Trade Database {trades.length > 0 && `— ${trades.length} records`}
          </h3>
          {tradesQuery.isLoading && (
            <span className="text-xs font-mono text-gray-500 flex items-center gap-2">
              <span className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              Loading…
            </span>
          )}
        </div>
        {trades.length > 0 ? (
          <TradeTable trades={trades} maxRows={500} />
        ) : (
          <div className="p-12 text-center text-gray-600 font-mono text-sm">
            {tradesQuery.isLoading ? 'Loading…' : 'No trades match current filters'}
          </div>
        )}
      </div>
    </div>
  );
}
