// ============================================================
// pages/OnChainDashboard.tsx -- On-chain and microstructure
// analytics dashboard.
//
// Sections:
//   1. MVRV Z-score time series + cycle phase
//   2. Funding rates per exchange -- Binance / Bybit / OKX / dYdX
//   3. VPIN gauge (informed trading probability)
//   4. Kyle's Lambda trend (market impact coefficient)
//   5. BTC dominance + network sentiment composite
//   6. Amihud illiquidity ratio per instrument
//   7. Whale exchange flows
// ============================================================

import React, { useState, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
  RadialBarChart, RadialBar, PolarAngleAxis,
} from 'recharts';
import { clsx } from 'clsx';
import {
  RefreshCw, Link2, AlertTriangle, TrendingUp, TrendingDown,
  Droplets, Zap, Activity, Eye, DollarSign,
} from 'lucide-react';

import { MetricCard }       from '../components/MetricCard';
import { GaugeChart }       from '../components/GaugeChart';
import { TimeSeriesChart }  from '../components/TimeSeriesChart';
import { SparkArea }        from '../components/SparkLine';

import {
  useOnChain,
  useRefreshOnChain,
  useMVRV,
  useFundingRates,
  useVPIN,
  useKyleLambda,
  useNetworkSentiment,
  useAmihud,
} from '../hooks/useOnChainAPI';

import type {
  ExchangeName,
  MVRVResponse,
  FundingRatesResponse,
  VPINResponse,
  KyleLambdaResponse,
  NetworkSentimentResponse,
  AmihudResponse,
} from '../types/onchain';

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

const fmtPct   = (v: number, dp = 3) => `${(v * 100).toFixed(dp)}%`;
const fmtBps   = (v: number)         => `${(v * 10_000).toFixed(2)} bps`;
const fmtSci   = (v: number)         => v.toExponential(3);
const fmtB     = (v: number)         => `$${(v / 1e9).toFixed(2)}B`;
const fmtM     = (v: number)         => `$${(v / 1e6).toFixed(1)}M`;
const fmt$     = (v: number)         =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(v);

function signalColor(signal: string): string {
  switch (signal) {
    case 'strong_buy':  return 'text-emerald-300';
    case 'buy':         return 'text-emerald-500';
    case 'neutral':     return 'text-slate-400';
    case 'sell':        return 'text-red-500';
    case 'strong_sell': return 'text-red-300';
    default:            return 'text-slate-400';
  }
}

function regimeBadgeColor(regime: string): string {
  if (regime.includes('extreme')) return 'bg-red-500/10 text-red-400 border-red-500/30';
  if (regime.includes('contango')) return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
  if (regime.includes('backwardation')) return 'bg-blue-500/10 text-blue-400 border-blue-500/30';
  return 'bg-slate-700/40 text-slate-400 border-slate-600/30';
}

function cyclePhaseColor(phase: string): string {
  switch (phase) {
    case 'accumulation': return 'text-blue-400';
    case 'early_bull':   return 'text-emerald-400';
    case 'late_bull':    return 'text-amber-400';
    case 'distribution': return 'text-orange-400';
    case 'bear':         return 'text-red-400';
    default:             return 'text-slate-400';
  }
}

const SENTIMENT_COLORS: Record<string, string> = {
  extreme_fear:  '#ef4444',
  fear:          '#f97316',
  neutral:       '#94a3b8',
  greed:         '#84cc16',
  extreme_greed: '#22c55e',
};

const EXCHANGE_COLORS: Record<ExchangeName, string> = {
  Binance: '#f0b90b',
  Bybit:   '#e6612c',
  OKX:     '#60a5fa',
  dYdX:    '#a78bfa',
  GMX:     '#34d399',
};

// ---------------------------------------------------------------------------
// MVRV section
// ---------------------------------------------------------------------------

function MVRVSection({ mvrv }: { mvrv: MVRVResponse }) {
  const recentData = mvrv.data.slice(-90).map(d => ({
    date: d.date,
    mvrv: d.mvrv,
    z_score: d.mvrv_z_score,
  }));

  return (
    <div className="flex flex-col gap-3">
      {/* Stats row */}
      <div className="flex items-center gap-4 flex-wrap">
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold">MVRV</p>
          <p className="text-xl font-bold font-mono text-slate-100">{mvrv.current_mvrv.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold">Z-score</p>
          <p className={clsx(
            'text-xl font-bold font-mono',
            mvrv.current_z_score > 2 ? 'text-red-400' :
            mvrv.current_z_score > 1 ? 'text-amber-400' :
            mvrv.current_z_score < -1 ? 'text-emerald-400' :
            'text-slate-200'
          )}>
            {mvrv.current_z_score.toFixed(2)}
          </p>
        </div>
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold">Percentile</p>
          <p className="text-xl font-bold font-mono text-slate-200">{mvrv.z_score_percentile.toFixed(0)}th</p>
        </div>
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold">Cycle Phase</p>
          <p className={clsx('text-sm font-bold capitalize', cyclePhaseColor(mvrv.cycle_phase))}>
            {mvrv.cycle_phase.replace('_', ' ')}
          </p>
        </div>
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold">Signal</p>
          <p className={clsx('text-sm font-bold uppercase', signalColor(mvrv.signal))}>
            {mvrv.signal.replace('_', ' ')}
          </p>
        </div>
      </div>

      {/* Chart */}
      <TimeSeriesChart
        data={recentData}
        xKey="date"
        xFormat="MMM d"
        height={150}
        showGrid
        showLegend
        series={[
          { key: 'mvrv',    label: 'MVRV',    color: '#f59e0b', type: 'area', yAxisId: 'left',  strokeWidth: 2, fillOpacity: 0.12 },
          { key: 'z_score', label: 'Z-score', color: '#60a5fa', type: 'line', yAxisId: 'right', strokeWidth: 1.5 },
        ]}
        referenceLines={[
          { y: 3,  yAxisId: 'right', color: '#ef4444', dashArray: '4 4', label: 'Z=3' },
          { y: -2, yAxisId: 'right', color: '#22c55e', dashArray: '4 4', label: 'Z=-2' },
          { y: 0,  yAxisId: 'right', color: '#475569', dashArray: '2 2' },
        ]}
        leftLabel="MVRV"
        rightLabel="Z-score"
        leftDomain={[0, 6]}
        rightDomain={[-4, 8]}
        tooltipFormatter={(v: number, k: string) => k === 'z_score' ? v.toFixed(3) : v.toFixed(3)}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Funding rates section
// ---------------------------------------------------------------------------

function FundingRatesSection({ funding }: { funding: FundingRatesResponse }) {
  const [selectedSymbol] = useState('BTC-USDT');

  const chartData = useMemo(() => {
    // Merge all exchange histories by timestamp
    const byTs: Record<string, Record<string, number>> = {};
    for (const exch of funding.exchanges) {
      for (const h of exch.history.slice(-48)) {
        if (!byTs[h.timestamp]) byTs[h.timestamp] = {};
        byTs[h.timestamp][exch.exchange] = h.rate_annualized;
      }
    }
    return Object.entries(byTs)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([ts, vals]) => ({ timestamp: ts, ...vals }));
  }, [funding]);

  return (
    <div className="flex flex-col gap-3">
      {/* Current rates table */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="px-3 py-2 text-left text-[10px] uppercase font-bold text-slate-500">Exchange</th>
              <th className="px-3 py-2 text-right text-[10px] uppercase font-bold text-slate-500">8h Rate</th>
              <th className="px-3 py-2 text-right text-[10px] uppercase font-bold text-slate-500">Annualized</th>
              <th className="px-3 py-2 text-right text-[10px] uppercase font-bold text-slate-500">Predicted</th>
              <th className="px-3 py-2 text-right text-[10px] uppercase font-bold text-slate-500">OI</th>
              <th className="px-3 py-2 text-left text-[10px] uppercase font-bold text-slate-500">Regime</th>
            </tr>
          </thead>
          <tbody>
            {funding.exchanges.map(exch => {
              const ann = exch.current_annualized;
              return (
                <tr key={exch.exchange} className="border-b border-slate-800 hover:bg-slate-800/30">
                  <td className="px-3 py-2 font-semibold" style={{ color: EXCHANGE_COLORS[exch.exchange] ?? '#94a3b8' }}>
                    {exch.exchange}
                  </td>
                  <td className={clsx('px-3 py-2 font-mono tabular-nums text-right', ann > 0 ? 'text-amber-400' : 'text-blue-400')}>
                    {fmtPct(exch.current_rate_8h, 4)}
                  </td>
                  <td className={clsx('px-3 py-2 font-mono tabular-nums text-right font-semibold', ann > 0.3 ? 'text-red-400' : ann > 0 ? 'text-amber-400' : 'text-blue-400')}>
                    {fmtPct(ann, 2)}
                  </td>
                  <td className="px-3 py-2 font-mono tabular-nums text-right text-slate-400">
                    {fmtPct(exch.predicted_rate, 4)}
                  </td>
                  <td className="px-3 py-2 font-mono tabular-nums text-right text-slate-400">
                    {fmtB(exch.open_interest_usd)}
                  </td>
                  <td className="px-3 py-2">
                    <span className={clsx('text-[9px] font-bold px-1.5 py-0.5 rounded border', regimeBadgeColor(exch.regime))}>
                      {exch.regime.replace(/_/g, ' ').toUpperCase()}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot className="border-t border-slate-600">
            <tr>
              <td className="px-3 py-2 text-xs font-bold text-slate-300">Composite</td>
              <td className="px-3 py-2" />
              <td className={clsx(
                'px-3 py-2 font-mono tabular-nums text-right font-bold',
                funding.composite_annualized > 0 ? 'text-amber-300' : 'text-blue-300'
              )}>
                {fmtPct(funding.composite_annualized, 2)}
              </td>
              <td colSpan={3} className="px-3 py-2 text-xs font-mono text-slate-500">
                Divergence: {funding.divergence.toFixed(1)} bps
                {funding.arb_opportunity && (
                  <span className="ml-2 text-emerald-400 font-bold">ARB OPPORTUNITY</span>
                )}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      {/* Funding rate history chart */}
      <TimeSeriesChart
        data={chartData}
        xKey="timestamp"
        xFormat="MMM d HH:mm"
        height={130}
        showGrid
        showLegend
        series={funding.exchanges.map(exch => ({
          key: exch.exchange,
          label: exch.exchange,
          color: EXCHANGE_COLORS[exch.exchange] ?? '#94a3b8',
          type: 'line' as const,
          strokeWidth: 1.5,
        }))}
        referenceLines={[{ y: 0, color: '#475569', dashArray: '2 2' }]}
        tooltipFormatter={(v: number) => fmtPct(v, 3)}
        leftLabel="Annualized"
        compact
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// VPIN gauge section
// ---------------------------------------------------------------------------

function VPINSection({ vpin }: { vpin: VPINResponse }) {
  const recentHistory = vpin.history.slice(-48).map(h => ({
    timestamp: h.timestamp,
    vpin:      h.vpin,
    imbalance: h.imbalance,
  }));

  const trendColor = vpin.trend === 'rising' ? '#ef4444' : vpin.trend === 'falling' ? '#22c55e' : '#94a3b8';

  return (
    <div className="flex flex-col gap-3">
      {/* Gauge + stats */}
      <div className="flex items-start gap-6 flex-wrap">
        <GaugeChart
          value={vpin.current_vpin}
          min={0}
          max={1}
          label="VPIN"
          subLabel={`${vpin.vpin_percentile.toFixed(0)}th percentile`}
          size={160}
          bands={[
            { from: 0,    to: 0.45, color: '#22c55e', label: 'Low' },
            { from: 0.45, to: 0.65, color: '#f59e0b', label: 'Medium' },
            { from: 0.65, to: 0.85, color: '#ef4444', label: 'High' },
            { from: 0.85, to: 1,    color: '#991b1b', label: 'Extreme' },
          ]}
        />
        <div className="flex flex-col gap-2 text-xs">
          <div className="flex items-center gap-2">
            {vpin.is_alert && (
              <span className="flex items-center gap-1 text-red-400 font-bold text-[11px] border border-red-500/30 bg-red-500/10 px-2 py-1 rounded">
                <AlertTriangle size={11} /> ELEVATED INFORMED TRADING
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono">
            <span className="text-slate-500">Trend:</span>
            <span style={{ color: trendColor }} className="font-semibold capitalize">{vpin.trend}</span>
            <span className="text-slate-500">Threshold:</span>
            <span className="text-slate-300">{vpin.alert_threshold.toFixed(2)}</span>
            <span className="text-slate-500">Symbol:</span>
            <span className="text-slate-300">{vpin.symbol}</span>
          </div>
          <p className="text-[10px] text-slate-600 max-w-xs">
            VPIN measures the probability of informed trading using volume-synchronized order imbalance.
            Values above {vpin.alert_threshold.toFixed(2)} suggest elevated adverse selection risk.
          </p>
        </div>
      </div>

      {/* VPIN time series */}
      <TimeSeriesChart
        data={recentHistory}
        xKey="timestamp"
        xFormat="MMM d HH:mm"
        height={100}
        compact
        showGrid
        series={[
          { key: 'vpin',      label: 'VPIN',      color: '#ef4444', type: 'area', strokeWidth: 1.5, fillOpacity: 0.12 },
          { key: 'imbalance', label: 'Imbalance', color: '#f59e0b', type: 'line', strokeWidth: 1, yAxisId: 'right' },
        ]}
        referenceLines={[
          { y: vpin.alert_threshold, color: '#ef4444', dashArray: '4 4' },
        ]}
        leftDomain={[0, 1]}
        rightDomain={[-0.5, 0.5]}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Kyle's Lambda section
// ---------------------------------------------------------------------------

function KyleLambdaSection({ lambda: kl }: { lambda: KyleLambdaResponse }) {
  const recentHistory = kl.history.slice(-48).map(h => ({
    timestamp: h.timestamp,
    lambda: h.lambda,
    price_impact: h.price_impact_10k_usd,
  }));

  const lambdaSparkData = recentHistory.map(d => ({ value: d.lambda }));

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-start gap-6 flex-wrap">
        <div className="flex flex-col gap-1">
          <p className="text-[10px] text-slate-500 uppercase font-bold tracking-wider">Current lambda</p>
          <p className={clsx(
            'text-2xl font-bold font-mono',
            kl.regime === 'crisis'   ? 'text-red-400' :
            kl.regime === 'illiquid' ? 'text-amber-400' :
            kl.regime === 'normal'   ? 'text-slate-200' :
            'text-emerald-400'
          )}>
            {fmtSci(kl.current_lambda)}
          </p>
          <p className="text-xs text-slate-500 font-mono">USD/unit-flow</p>
        </div>
        <div className="flex flex-col gap-2 text-xs font-mono">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <span className="text-slate-500">Regime:</span>
            <span className={clsx('font-bold capitalize',
              kl.regime === 'crisis' ? 'text-red-400' :
              kl.regime === 'illiquid' ? 'text-amber-400' :
              kl.regime === 'normal' ? 'text-slate-300' :
              'text-emerald-400'
            )}>{kl.regime}</span>
            <span className="text-slate-500">Trend:</span>
            <span className={clsx('font-semibold',
              kl.trend === 'increasing' ? 'text-red-400' :
              kl.trend === 'decreasing' ? 'text-emerald-400' :
              'text-slate-400'
            )}>{kl.trend}</span>
            <span className="text-slate-500">Percentile:</span>
            <span className="text-slate-200">{kl.lambda_percentile.toFixed(0)}th</span>
            <span className="text-slate-500">$10k impact:</span>
            <span className="text-slate-300">
              ${(kl.history[kl.history.length - 1]?.price_impact_10k_usd ?? 0).toFixed(2)}
            </span>
          </div>
        </div>
        <div className="ml-auto">
          <SparkArea
            data={lambdaSparkData}
            width={120}
            height={50}
            higherIsBetter={false}
          />
        </div>
      </div>

      {/* Lambda trend chart */}
      <TimeSeriesChart
        data={recentHistory}
        xKey="timestamp"
        xFormat="MMM d HH:mm"
        height={100}
        compact
        showGrid
        series={[
          { key: 'lambda',       label: "Kyle's Lambda", color: '#34d399', type: 'area', strokeWidth: 1.5, fillOpacity: 0.12 },
          { key: 'price_impact', label: '$10k Impact',   color: '#f59e0b', type: 'line', strokeWidth: 1,   yAxisId: 'right' },
        ]}
        tooltipFormatter={(v: number, k: string) => k === 'lambda' ? fmtSci(v) : `$${v.toFixed(4)}`}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Network sentiment section
// ---------------------------------------------------------------------------

function NetworkSentimentSection({ ns }: { ns: NetworkSentimentResponse }) {
  const recent = ns.sentiment.slice(-60);
  const domRecent = ns.btc_dominance.slice(-60);

  const chartData = recent.map((s, i) => ({
    date: s.date,
    composite: s.sentiment_composite,
    fear_greed: s.fear_greed_index / 100,
    nvt: s.nvt_signal,
    sopr: s.sopr,
    dominance: domRecent[i]?.dominance,
    alt_season: domRecent[i] ? domRecent[i].alt_season_index / 100 : undefined,
  }));

  const fgColor = SENTIMENT_COLORS[ns.sentiment_regime] ?? '#94a3b8';

  return (
    <div className="flex flex-col gap-3">
      {/* Sentiment summary */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="text-center">
          <p className="text-[10px] text-slate-500 uppercase font-bold">Fear & Greed</p>
          <p className="text-3xl font-bold tabular-nums" style={{ color: fgColor }}>
            {ns.current_fear_greed.toFixed(0)}
          </p>
          <p className="text-xs font-semibold capitalize" style={{ color: fgColor }}>
            {ns.sentiment_regime.replace(/_/g, ' ')}
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono">
            <span className="text-slate-500">Composite:</span>
            <span className={clsx('font-semibold', ns.current_sentiment_composite > 0 ? 'text-emerald-400' : 'text-red-400')}>
              {ns.current_sentiment_composite.toFixed(3)}
            </span>
            <span className="text-slate-500">NVT:</span>
            <span className="text-slate-300">{ns.current_nvt.toFixed(1)}</span>
            <span className="text-slate-500">SOPR:</span>
            <span className={clsx('font-semibold', ns.current_sopr > 1 ? 'text-emerald-400' : 'text-red-400')}>
              {ns.current_sopr.toFixed(3)}
            </span>
          </div>
        </div>

        {/* Radial gauge for fear & greed */}
        <div className="ml-auto">
          <ResponsiveContainer width={80} height={80}>
            <RadialBarChart
              cx={40} cy={40} innerRadius={26} outerRadius={36}
              data={[{ value: ns.current_fear_greed, fill: fgColor }]}
              startAngle={180} endAngle={0}
            >
              <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
              <RadialBar background={{ fill: '#1e293b' }} dataKey="value" cornerRadius={4} angleAxisId={0} />
            </RadialBarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart */}
      <TimeSeriesChart
        data={chartData}
        xKey="date"
        xFormat="MMM d"
        height={150}
        showGrid
        showLegend
        series={[
          { key: 'composite',  label: 'Sentiment',    color: '#60a5fa', type: 'area', strokeWidth: 2,   fillOpacity: 0.1 },
          { key: 'fear_greed', label: 'F&G (norm)',   color: '#f59e0b', type: 'line', strokeWidth: 1.5, yAxisId: 'right' },
          { key: 'dominance',  label: 'BTC Dom',      color: '#f97316', type: 'line', strokeWidth: 1,   yAxisId: 'right', strokeDasharray: '3 3' },
        ]}
        referenceLines={[{ y: 0, color: '#475569', dashArray: '2 2' }]}
        leftDomain={[-1, 1]}
        rightDomain={[0, 1]}
        tooltipFormatter={(v: number, k: string) =>
          k === 'dominance' ? fmtPct(v) : v.toFixed(3)
        }
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Amihud illiquidity section
// ---------------------------------------------------------------------------

function AmihudSection({ instruments }: { instruments: string[] }) {
  const { data: btc }  = useAmihud('BTC-USD');
  const { data: eth }  = useAmihud('ETH-USD');
  const { data: sol }  = useAmihud('SOL-USD');

  const datasets: Array<{ symbol: string; data: AmihudResponse | undefined; color: string }> = [
    { symbol: 'BTC-USD', data: btc, color: '#f59e0b' },
    { symbol: 'ETH-USD', data: eth, color: '#60a5fa' },
    { symbol: 'SOL-USD', data: sol, color: '#a78bfa' },
  ];

  const chartData = useMemo(() => {
    const byDate: Record<string, Record<string, number>> = {};
    datasets.forEach(({ symbol, data: d }) => {
      if (!d) return;
      d.history.forEach(h => {
        if (!byDate[h.date]) byDate[h.date] = {};
        byDate[h.date][symbol] = h.illiquidity;
      });
    });
    return Object.entries(byDate).sort(([a], [b]) => a.localeCompare(b)).slice(-45)
      .map(([date, vals]) => ({ date, ...vals }));
  }, [btc, eth, sol]);

  return (
    <div className="flex flex-col gap-3">
      {/* Current readings */}
      <div className="grid grid-cols-3 gap-3">
        {datasets.map(({ symbol, data: d, color }) => (
          <div key={symbol} className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/40">
            <p className="text-[10px] text-slate-500 uppercase font-bold font-mono">{symbol}</p>
            {d ? (
              <>
                <p className="text-base font-bold font-mono mt-1" style={{ color }}>
                  {d.current_illiquidity.toExponential(2)}
                </p>
                <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                  {d.percentile_30d.toFixed(0)}th pct (30d)
                </p>
                <p className={clsx(
                  'text-[10px] font-semibold capitalize mt-0.5',
                  d.trend === 'increasing' ? 'text-red-400' :
                  d.trend === 'decreasing' ? 'text-emerald-400' :
                  'text-slate-500'
                )}>
                  {d.trend}
                </p>
              </>
            ) : (
              <p className="text-xs text-slate-600 mt-1">Loading...</p>
            )}
          </div>
        ))}
      </div>

      {/* Illiquidity chart */}
      <TimeSeriesChart
        data={chartData}
        xKey="date"
        xFormat="MMM d"
        height={130}
        compact
        showGrid
        showLegend
        series={datasets.map(({ symbol, color }) => ({
          key: symbol,
          label: symbol,
          color,
          type: 'line' as const,
          strokeWidth: 1.5,
        }))}
        tooltipFormatter={(v: number) => v.toExponential(3)}
        leftLabel="Illiquidity"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Whale flows section
// ---------------------------------------------------------------------------

function WhaleFlowsSection() {
  const { data } = useOnChain();
  const wf = data?.whale_flows;
  if (!wf) return <p className="text-xs text-slate-500">No whale flow data.</p>;

  const recentHistory = wf.history.slice(-24).map(h => ({
    timestamp: h.timestamp,
    inflow:  h.inflow_usd  / 1e6,
    outflow: -h.outflow_usd / 1e6,
    net:     h.net_flow_usd / 1e6,
    large_txns: h.large_txns_count,
  }));

  return (
    <div className="flex flex-col gap-3">
      {/* Stats */}
      <div className="flex items-center gap-4 flex-wrap text-xs font-mono">
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold">Net Flow 24h</p>
          <p className={clsx(
            'text-lg font-bold',
            wf.current_net_flow_24h > 0 ? 'text-red-400' : 'text-emerald-400'
          )}>
            {wf.current_net_flow_24h > 0 ? '+' : ''}{fmtM(Math.abs(wf.current_net_flow_24h))}
          </p>
          <p className="text-[10px] text-slate-600">{wf.current_net_flow_24h > 0 ? '(bearish -- deposits)' : '(bullish -- withdrawals)'}</p>
        </div>
        <div>
          <p className="text-[10px] text-slate-500 uppercase font-bold">Z-score (30d)</p>
          <p className={clsx(
            'text-lg font-bold',
            Math.abs(wf.net_flow_zscore_30d) > 1.5 ? 'text-amber-400' : 'text-slate-200'
          )}>
            {wf.net_flow_zscore_30d.toFixed(2)}
          </p>
        </div>
        {wf.alert && (
          <span className="flex items-center gap-1 text-amber-400 font-bold border border-amber-500/30 bg-amber-500/10 px-2 py-1 rounded text-[11px]">
            <AlertTriangle size={11} /> WHALE ALERT
          </span>
        )}
      </div>

      {/* Flow chart */}
      <ResponsiveContainer width="100%" height={130}>
        <BarChart data={recentHistory} margin={{ top: 4, right: 4, bottom: 4, left: 36 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="timestamp"
            tickFormatter={v => { try { const d = new Date(v); return `${d.getHours()}:00`; } catch { return v; } }}
            tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={{ stroke: '#334155' }}
            minTickGap={30}
          />
          <YAxis
            tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={v => `$${Math.abs(v).toFixed(0)}M`}
            width={36}
          />
          <Tooltip
            formatter={(v: number) => [`$${Math.abs(v).toFixed(1)}M`, '']}
            contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 11, borderRadius: 6 }}
          />
          <ReferenceLine y={0} stroke="#334155" />
          <Bar dataKey="inflow"  fill="#ef4444" opacity={0.7} radius={[2, 2, 0, 0]} isAnimationActive={false} />
          <Bar dataKey="outflow" fill="#22c55e" opacity={0.7} radius={[0, 0, 2, 2]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-[10px] text-slate-600 text-center">Red = inflows (bearish), Green = outflows (bullish)</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function OnChainDashboard() {
  const { data, isLoading, isError } = useOnChain();
  const refresh = useRefreshOnChain();

  const [activeSection, setActiveSection] = useState<
    'mvrv' | 'funding' | 'vpin' | 'kyle' | 'sentiment' | 'amihud' | 'whales'
  >('mvrv');

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500 gap-2">
        <Activity size={18} className="animate-pulse" />
        <span className="text-sm">Loading on-chain data...</span>
      </div>
    );
  }

  const mvrv     = data?.mvrv;
  const funding  = data?.funding_rates;
  const vpin     = data?.vpin;
  const kyleLam  = data?.kyle_lambda;
  const netSent  = data?.network_sentiment;

  return (
    <div className="flex flex-col gap-5 p-4 min-h-screen bg-slate-950 text-slate-100">

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <Link2 size={20} className="text-cyan-400" />
          <div>
            <h1 className="text-lg font-bold text-slate-100">On-Chain & Microstructure</h1>
            <p className="text-xs text-slate-500">
              BTC/ETH/SOL -- live on-chain and microstructure signals
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {isError && <span className="text-xs text-amber-400">API unreachable -- using mock data</span>}
          <button
            onClick={refresh}
            className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded border border-slate-700 transition-colors"
          >
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      </div>

      {/* KPI bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        {mvrv && (
          <MetricCard
            label="MVRV Z-score"
            value={mvrv.current_z_score}
            format="ratio"
            higherIsBetter={false}
            subValue={`MVRV: ${mvrv.current_mvrv.toFixed(2)}`}
            variant={mvrv.current_z_score > 3 ? 'critical' : mvrv.current_z_score > 2 ? 'warn' : 'default'}
          />
        )}
        {funding && (
          <MetricCard
            label="Funding Rate"
            value={funding.composite_annualized}
            format="percent"
            higherIsBetter={false}
            subValue={`Div: ${funding.divergence.toFixed(1)} bps`}
            variant={Math.abs(funding.composite_annualized) > 0.5 ? 'warn' : 'default'}
          />
        )}
        {vpin && (
          <MetricCard
            label="VPIN"
            value={vpin.current_vpin}
            format="ratio"
            higherIsBetter={false}
            subValue={`${vpin.vpin_percentile.toFixed(0)}th percentile`}
            variant={vpin.is_alert ? 'warn' : 'default'}
          />
        )}
        {kyleLam && (
          <MetricCard
            label="Kyle Lambda"
            value={kyleLam.lambda_percentile}
            format="count"
            unit="th pct"
            higherIsBetter={false}
            subValue={`${kyleLam.regime} liquidity`}
            variant={kyleLam.regime === 'crisis' ? 'critical' : kyleLam.regime === 'illiquid' ? 'warn' : 'default'}
          />
        )}
        {netSent && (
          <>
            <MetricCard
              label="Fear & Greed"
              value={netSent.current_fear_greed}
              format="count"
              higherIsBetter={true}
              subValue={netSent.sentiment_regime.replace(/_/g, ' ')}
            />
            <MetricCard
              label="SOPR"
              value={netSent.current_sopr}
              format="ratio"
              higherIsBetter={true}
              subValue={netSent.current_sopr > 1 ? 'Profit-taking' : 'Capitulation'}
              variant={netSent.current_sopr < 0.97 ? 'warn' : 'default'}
            />
            <MetricCard
              label="NVT Signal"
              value={netSent.current_nvt}
              format="ratio"
              higherIsBetter={false}
              subValue="Valuation proxy"
              variant={netSent.current_nvt > 90 ? 'warn' : 'default'}
            />
          </>
        )}
      </div>

      {/* Section tabs */}
      <div className="flex gap-1 flex-wrap">
        {[
          { key: 'mvrv',      label: 'MVRV Z-score',   Icon: TrendingUp   },
          { key: 'funding',   label: 'Funding Rates',   Icon: DollarSign   },
          { key: 'vpin',      label: 'VPIN',            Icon: Eye          },
          { key: 'kyle',      label: "Kyle's Lambda",   Icon: Zap          },
          { key: 'sentiment', label: 'Sentiment',       Icon: Activity     },
          { key: 'amihud',    label: 'Amihud',          Icon: Droplets     },
          { key: 'whales',    label: 'Whale Flows',     Icon: TrendingDown },
        ].map(({ key, label, Icon }) => (
          <button
            key={key}
            onClick={() => setActiveSection(key as any)}
            className={clsx(
              'flex items-center gap-1.5 px-3 py-2 rounded text-xs font-semibold uppercase tracking-wider transition-colors',
              activeSection === key
                ? 'bg-cyan-700 text-white'
                : 'bg-slate-800 text-slate-400 hover:text-slate-200'
            )}
          >
            <Icon size={12} /> {label}
          </button>
        ))}
      </div>

      {/* Section content */}
      <div className="bg-slate-900/60 border border-slate-700/50 rounded-lg p-4">
        {activeSection === 'mvrv'      && mvrv    && <MVRVSection mvrv={mvrv} />}
        {activeSection === 'funding'   && funding  && <FundingRatesSection funding={funding} />}
        {activeSection === 'vpin'      && vpin     && <VPINSection vpin={vpin} />}
        {activeSection === 'kyle'      && kyleLam  && <KyleLambdaSection lambda={kyleLam} />}
        {activeSection === 'sentiment' && netSent  && <NetworkSentimentSection ns={netSent} />}
        {activeSection === 'amihud'    && <AmihudSection instruments={['BTC-USD', 'ETH-USD', 'SOL-USD']} />}
        {activeSection === 'whales'    && <WhaleFlowsSection />}
        {!data && <p className="text-xs text-slate-500 py-4">No data available.</p>}
      </div>

    </div>
  );
}
