import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api } from '../api';
import type { BacktestResult, BacktestParams } from '../types';
import { EquityCurve } from '../components/EquityCurve';
import { TradeTable } from '../components/TradeTable';
import { MetricsCard } from '../components/MetricsCard';
import { RegimeBadge } from '../components/RegimeBadge';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';
import type { Regime } from '../types';

const SOURCES = ['yfinance', 'alpaca', 'csv'] as const;

function fmt(v: number, decimals = 2) { return v.toFixed(decimals); }
function fmtPct(v: number) { return `${(v * 100).toFixed(2)}%`; }

export function Backtest() {
  const instrumentsQuery = useQuery({
    queryKey: ['instruments'],
    queryFn: api.instruments,
    staleTime: 60_000,
  });

  const [params, setParams] = useState<BacktestParams>({
    sym: 'SPY',
    source: 'yfinance',
    start: '2018-01-01',
    end: '2024-12-31',
    long_only: false,
    params: { cf: 0.5, bh_form: 1.35, bh_decay: 0.95, bh_collapse: 0.7 },
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [activeTab, setActiveTab] = useState<'curve' | 'trades' | 'mass'>('curve');

  const backtestMut = useMutation({
    mutationFn: api.backtest,
    onSuccess: setResult,
  });

  const syms = instrumentsQuery.data?.map(i => i.sym) ?? ['SPY', 'QQQ', 'BTC', 'ETH', 'GLD', 'TLT'];

  function handleRun() {
    backtestMut.mutate(params);
  }

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Config panel */}
      <div className="bg-bg-card border border-bg-border rounded-xl p-5">
        <h2 className="text-sm font-mono font-semibold text-gray-400 uppercase tracking-wider mb-4">
          Backtest Configuration
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Symbol */}
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Instrument</label>
            <select
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={params.sym}
              onChange={e => setParams(p => ({ ...p, sym: e.target.value }))}
            >
              {syms.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          {/* Source */}
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Data Source</label>
            <select
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={params.source}
              onChange={e => setParams(p => ({ ...p, source: e.target.value as typeof params.source }))}
            >
              {SOURCES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          {/* Start */}
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Start Date</label>
            <input
              type="date"
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={params.start}
              onChange={e => setParams(p => ({ ...p, start: e.target.value }))}
            />
          </div>

          {/* End */}
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">End Date</label>
            <input
              type="date"
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={params.end}
              onChange={e => setParams(p => ({ ...p, end: e.target.value }))}
            />
          </div>
        </div>

        {/* Long only toggle */}
        <div className="mt-4 flex items-center gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <div
              className={`w-10 h-5 rounded-full relative transition-colors ${params.long_only ? 'bg-accent' : 'bg-bg-border'}`}
              onClick={() => setParams(p => ({ ...p, long_only: !p.long_only }))}
            >
              <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${params.long_only ? 'translate-x-5' : 'translate-x-0.5'}`} />
            </div>
            <span className="text-sm font-mono text-gray-300">Long Only</span>
          </label>

          <button
            className="text-xs font-mono text-gray-500 hover:text-gray-300 transition-colors"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? '▲' : '▼'} Advanced Params
          </button>
        </div>

        {/* Advanced params */}
        {showAdvanced && (
          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-bg-border">
            {(Object.entries(params.params) as [string, number][]).map(([key, val]) => (
              <div key={key}>
                <label className="block text-xs font-mono text-gray-500 mb-1">{key}</label>
                <input
                  type="number"
                  step="0.01"
                  className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
                  value={val}
                  onChange={e => setParams(p => ({
                    ...p,
                    params: { ...p.params, [key]: parseFloat(e.target.value) || 0 },
                  }))}
                />
              </div>
            ))}
          </div>
        )}

        {/* Run button */}
        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={handleRun}
            disabled={backtestMut.isPending}
            className="px-6 py-2.5 bg-accent hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed
                       rounded-lg text-sm font-mono font-semibold text-white transition-colors"
          >
            {backtestMut.isPending ? (
              <span className="flex items-center gap-2">
                <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Running…
              </span>
            ) : '▶ Run Backtest'}
          </button>
          {backtestMut.isError && (
            <span className="text-xs font-mono text-bear">
              Error: {(backtestMut.error as Error).message}
            </span>
          )}
        </div>
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Metrics row */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
            {(
              [
                { label: 'CAGR', value: fmtPct(result.metrics.cagr), trend: result.metrics.cagr > 0 ? 'up' : 'down' },
                { label: 'Sharpe', value: fmt(result.metrics.sharpe), trend: result.metrics.sharpe > 1 ? 'up' : 'neutral' },
                { label: 'Max DD', value: fmtPct(result.metrics.max_drawdown), trend: 'down' },
                { label: 'Win Rate', value: fmtPct(result.metrics.win_rate), trend: result.metrics.win_rate > 0.5 ? 'up' : 'down' },
                { label: 'Prof Factor', value: fmt(result.metrics.profit_factor), trend: result.metrics.profit_factor > 1 ? 'up' : 'down' },
                { label: 'Total Trades', value: String(result.metrics.total_trades), trend: undefined },
                { label: 'Total Return', value: fmtPct(result.metrics.total_return), trend: result.metrics.total_return > 0 ? 'up' : 'down' },
              ] as Array<{ label: string; value: string; trend?: 'up' | 'down' | 'neutral' }>
            ).map(m => (
              <MetricsCard key={m.label} label={m.label} value={m.value} trend={m.trend} />
            ))}
          </div>

          {/* Chart tabs */}
          <div className="bg-bg-card border border-bg-border rounded-xl overflow-hidden">
            <div className="flex border-b border-bg-border">
              {(['curve', 'trades', 'mass'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-5 py-3 text-xs font-mono uppercase tracking-wider transition-colors
                    ${activeTab === tab
                      ? 'text-accent border-b-2 border-accent bg-accent/5'
                      : 'text-gray-500 hover:text-gray-300'
                    }`}
                >
                  {tab === 'curve' ? 'Equity Curve' : tab === 'trades' ? 'Trade List' : 'BH Mass Series'}
                </button>
              ))}
            </div>

            <div className="p-4">
              {activeTab === 'curve' && (
                <EquityCurve
                  data={result.equity_curve}
                  height={350}
                  showBenchmark
                  showRegimeColor
                  showDrawdown
                />
              )}

              {activeTab === 'trades' && (
                <TradeTable trades={result.trades} />
              )}

              {activeTab === 'mass' && (
                <BHMassSeries data={result.bh_mass_series} />
              )}
            </div>
          </div>

          {/* Run ID */}
          <p className="text-xs font-mono text-gray-600">
            Run ID: {result.run_id} · {result.sym} · {result.trades.length} trades
          </p>
        </>
      )}

      {/* Empty state */}
      {!result && !backtestMut.isPending && (
        <div className="bg-bg-card border border-bg-border rounded-xl p-16 text-center">
          <div className="text-4xl mb-3 opacity-30">📊</div>
          <p className="text-gray-500 font-mono text-sm">Configure parameters and run a backtest to see results</p>
        </div>
      )}
    </div>
  );
}

function BHMassSeries({ data }: { data: BacktestResult['bh_mass_series'] }) {
  const REGIME_COLORS: Record<Regime, string> = {
    BULL: '#22c55e', BEAR: '#ef4444', SIDEWAYS: '#9ca3af', HIGH_VOL: '#a855f7',
  };

  function fmtDate(ts: string) {
    try { return new Date(ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' }); }
    catch { return ts; }
  }

  const display = data.length > 500 ? data.filter((_, i) => i % Math.ceil(data.length / 500) === 0) : data;

  return (
    <ResponsiveContainer width="100%" height={350}>
      <LineChart data={display} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
        <XAxis dataKey="date" tickFormatter={fmtDate} tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
        <YAxis domain={[0, 2.0]} tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
        <Tooltip
          contentStyle={{ background: '#1a1a24', border: '1px solid #2a2a3a', fontFamily: 'JetBrains Mono' }}
          labelStyle={{ color: '#9ca3af' }}
        />
        <Legend wrapperStyle={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
        <Line type="monotone" dataKey="mass_15m" stroke="#6366f1" strokeWidth={1} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="mass_1h" stroke="#eab308" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="mass_1d" stroke="#f97316" strokeWidth={2} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
