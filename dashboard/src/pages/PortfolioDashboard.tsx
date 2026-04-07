// ============================================================
// pages/PortfolioDashboard.tsx -- Portfolio overview dashboard
//
// Sections:
//   1. Metrics bar (NAV, daily P&L, YTD, leverage, Sharpe)
//   2. Positions table (sortable, expandable rows)
//   3. Sector exposure donut chart (pure SVG)
//   4. P&L attribution (by symbol, by sector, by strategy)
//   5. Risk meters (VaR util, drawdown, margin)
// ============================================================

import React, { useState, useMemo } from 'react';
import { clsx } from 'clsx';
import { useQuery } from '@tanstack/react-query';
import { SortableTable } from '../components/SortableTable';
import type { ColumnDef } from '../components/SortableTable';
import { DonutChart } from '../components/DonutChart';
import type { DonutSlice } from '../components/DonutChart';
import { Card, LoadingSpinner, ProgressBar } from '../components/ui';
import {
  formatUSD,
  formatNav,
  formatPct,
  formatPctRaw,
  formatLeverage,
  formatRatio,
  formatQty,
  formatTimestamp,
  colorForPnl,
  colorForPnlFrac,
  colorForUtilization,
  colorForScore,
} from '../utils/formatters';

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

export interface PortfolioPosition {
  symbol:        string;
  qty:           number;
  avgCost:       number;
  marketValue:   number;
  unrealizedPnl: number;
  unrealizedPct: number;
  dailyPnl:      number;
  sector:        string;
  beta:          number;
  side:          'long' | 'short';
  strategy:      string;
  weight:        number;
}

export interface PortfolioMetrics {
  totalNav:       number;
  cashBalance:    number;
  grossExposure:  number;
  netExposure:    number;
  leverage:       number;
  dailyPnl:       number;
  dailyPnlPct:    number;
  ytdPnl:         number;
  ytdPnlPct:      number;
  sharpeYTD:      number;
  maxDrawdown:    number;
  currentDrawdown:number;
  varUtil:        number;
  marginUtil:     number;
  weeklyPnl:      number;
  monthlyPnl:     number;
}

interface SectorExposure {
  sector: string;
  marketValue: number;
  pct: number;
}

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const SECTORS   = ['L1 Crypto', 'DeFi', 'Equity', 'ETF', 'Options', 'Stablecoin'];
const STRATEGIES= ['momentum', 'mean_reversion', 'stat_arb', 'trend_follow'];
const SYMS      = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AAPL', 'SPY', 'QQQ', 'MSFT', 'TSLA'];
const SECTOR_COLORS = ['#3b82f6','#10b981','#8b5cf6','#f59e0b','#ef4444','#06b6d4'];

function randRange(lo: number, hi: number): number {
  return lo + Math.random() * (hi - lo);
}

function genPositions(): PortfolioPosition[] {
  return SYMS.map((symbol, i) => {
    const avgCost = randRange(10, 60_000);
    const price   = avgCost * randRange(0.88, 1.18);
    const qty     = randRange(0.01, 20);
    const mv      = qty * price;
    const upnl    = (price - avgCost) * qty;
    return {
      symbol,
      qty,
      avgCost,
      marketValue:   mv,
      unrealizedPnl: upnl,
      unrealizedPct: (price - avgCost) / avgCost,
      dailyPnl:      randRange(-500, 800),
      sector:        SECTORS[i % SECTORS.length],
      beta:          parseFloat(randRange(0.4, 1.8).toFixed(2)),
      side:          (Math.random() > 0.2 ? 'long' : 'short') as 'long' | 'short',
      strategy:      STRATEGIES[i % STRATEGIES.length],
      weight:        0,
    };
  }).map((p, _, arr) => {
    const totalMv = arr.reduce((s, x) => s + Math.abs(x.marketValue), 0);
    return { ...p, weight: Math.abs(p.marketValue) / totalMv };
  });
}

function genMetrics(): PortfolioMetrics {
  const nav = randRange(90_000, 120_000);
  return {
    totalNav:        nav,
    cashBalance:     nav * randRange(0.05, 0.20),
    grossExposure:   nav * randRange(1.1, 1.8),
    netExposure:     nav * randRange(0.5, 1.0),
    leverage:        parseFloat(randRange(1.1, 1.8).toFixed(2)),
    dailyPnl:        randRange(-1500, 2000),
    dailyPnlPct:     randRange(-0.015, 0.020),
    ytdPnl:          randRange(-5000, 15_000),
    ytdPnlPct:       randRange(-0.05, 0.18),
    sharpeYTD:       parseFloat(randRange(0.4, 2.8).toFixed(2)),
    maxDrawdown:     parseFloat(randRange(0.05, 0.18).toFixed(3)),
    currentDrawdown: parseFloat(randRange(0, 0.08).toFixed(3)),
    varUtil:         parseFloat(randRange(0.3, 0.85).toFixed(3)),
    marginUtil:      parseFloat(randRange(0.2, 0.7).toFixed(3)),
    weeklyPnl:       randRange(-3000, 4000),
    monthlyPnl:      randRange(-8000, 18_000),
  };
}

// ---------------------------------------------------------------------------
// usePortfolioAPI hook
// ---------------------------------------------------------------------------

const PORTFOLIO_API = 'http://localhost:8791';

async function fetchPortfolio<T>(path: string, fallback: () => T): Promise<T> {
  try {
    const res = await fetch(`${PORTFOLIO_API}${path}`, {
      headers: { Accept: 'application/json' },
      signal:  AbortSignal.timeout(4_000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as T;
  } catch {
    return fallback();
  }
}

export function usePositions() {
  return useQuery<PortfolioPosition[]>({
    queryKey: ['portfolio', 'positions'],
    queryFn:  () => fetchPortfolio('/portfolio/positions', genPositions),
    refetchInterval: 5_000,
    staleTime:       2_500,
  });
}

export function usePortfolioMetrics() {
  return useQuery<PortfolioMetrics>({
    queryKey: ['portfolio', 'metrics'],
    queryFn:  () => fetchPortfolio('/portfolio/metrics', genMetrics),
    refetchInterval: 5_000,
    staleTime:       2_500,
  });
}

export function useExposure() {
  const { data: positions } = usePositions();
  return useMemo((): SectorExposure[] => {
    if (!positions?.length) return [];
    const sectorMap: Record<string, number> = {};
    for (const p of positions) {
      sectorMap[p.sector] = (sectorMap[p.sector] ?? 0) + Math.abs(p.marketValue);
    }
    const total = Object.values(sectorMap).reduce((s, v) => s + v, 0);
    return Object.entries(sectorMap)
      .map(([sector, mv]) => ({ sector, marketValue: mv, pct: mv / total }))
      .sort((a, b) => b.marketValue - a.marketValue);
  }, [positions]);
}

// ---------------------------------------------------------------------------
// Section 1: Metrics bar
// ---------------------------------------------------------------------------

const MetricsBar: React.FC<{ m: PortfolioMetrics }> = ({ m }) => {
  const items = [
    { label: 'NAV',            value: formatNav(m.totalNav),            color: '#e2e8f0' },
    { label: 'Cash',           value: formatNav(m.cashBalance),         color: '#94a3b8' },
    { label: 'Daily P&L',      value: formatUSD(m.dailyPnl),            color: colorForPnl(m.dailyPnl), sub: formatPct(m.dailyPnlPct) },
    { label: 'Weekly P&L',     value: formatUSD(m.weeklyPnl),           color: colorForPnl(m.weeklyPnl) },
    { label: 'Monthly P&L',    value: formatUSD(m.monthlyPnl),          color: colorForPnl(m.monthlyPnl) },
    { label: 'YTD P&L',        value: formatUSD(m.ytdPnl),              color: colorForPnl(m.ytdPnl), sub: formatPct(m.ytdPnlPct) },
    { label: 'Gross Exposure', value: formatNav(m.grossExposure),        color: '#94a3b8' },
    { label: 'Net Exposure',   value: formatNav(m.netExposure),          color: '#94a3b8' },
    { label: 'Leverage',       value: formatLeverage(m.leverage),        color: m.leverage > 1.5 ? '#fbbf24' : '#94a3b8' },
    { label: 'Sharpe YTD',     value: formatRatio(m.sharpeYTD),          color: m.sharpeYTD >= 1 ? '#34d399' : m.sharpeYTD >= 0.5 ? '#fbbf24' : '#f87171' },
  ];

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-5 lg:grid-cols-10">
      {items.map(item => (
        <div
          key={item.label}
          className="bg-[#111318] border border-[#1e2130] rounded-lg px-3 py-2"
        >
          <div className="text-[9px] font-mono text-slate-600 uppercase tracking-wider mb-1">
            {item.label}
          </div>
          <div className="text-sm font-mono font-bold" style={{ color: item.color }}>
            {item.value}
          </div>
          {item.sub && (
            <div className="text-[9px] font-mono mt-0.5" style={{ color: item.color }}>
              {item.sub}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Section 2: PositionsTable
// ---------------------------------------------------------------------------

const POSITION_COLS: ColumnDef<PortfolioPosition>[] = [
  {
    key:    'symbol',
    header: 'Symbol',
    render: (v) => <span className="text-slate-200 font-semibold">{v as string}</span>,
  },
  {
    key:    'side',
    header: 'Side',
    width:  '54px',
    render: (v) => (
      <span className={clsx(
        'text-[10px] font-semibold uppercase',
        v === 'long' ? 'text-emerald-400' : 'text-red-400',
      )}>
        {v as string}
      </span>
    ),
  },
  {
    key:    'qty',
    header: 'Qty',
    align:  'right',
    render: (v) => <span>{formatQty(v as number)}</span>,
  },
  {
    key:    'avgCost',
    header: 'Avg Cost',
    align:  'right',
    render: (v) => <span className="text-slate-400">${(v as number).toFixed(2)}</span>,
  },
  {
    key:    'marketValue',
    header: 'Mkt Value',
    align:  'right',
    render: (v) => <span>{formatUSD(v as number)}</span>,
  },
  {
    key:    'unrealizedPnl',
    header: 'Unreal. P&L',
    align:  'right',
    render: (v) => (
      <span style={{ color: colorForPnl(v as number) }}>
        {formatUSD(v as number)}
      </span>
    ),
  },
  {
    key:    'unrealizedPct',
    header: '%',
    align:  'right',
    width:  '68px',
    render: (v) => (
      <span style={{ color: colorForPnlFrac(v as number) }}>
        {formatPct(v as number)}
      </span>
    ),
  },
  {
    key:    'dailyPnl',
    header: 'Daily P&L',
    align:  'right',
    render: (v) => (
      <span style={{ color: colorForPnl(v as number) }}>
        {formatUSD(v as number)}
      </span>
    ),
  },
  {
    key:    'weight',
    header: 'Weight',
    align:  'right',
    width:  '72px',
    render: (v) => <span className="text-slate-400">{formatPct(v as number)}</span>,
  },
  {
    key:    'beta',
    header: 'Beta',
    align:  'right',
    width:  '54px',
    render: (v) => <span className="text-slate-400">{(v as number).toFixed(2)}</span>,
  },
  {
    key:    'sector',
    header: 'Sector',
    render: (v) => <span className="text-slate-500 text-[10px]">{v as string}</span>,
  },
  {
    key:    'strategy',
    header: 'Strategy',
    render: (v) => <span className="text-slate-600 text-[10px]">{v as string}</span>,
  },
];

interface PositionDetailProps {
  pos:     PortfolioPosition;
  onClose: () => void;
}

const PositionDetail: React.FC<PositionDetailProps> = ({ pos, onClose }) => (
  <div className="mt-3 p-4 rounded-lg bg-[#0d1017] border border-blue-900/40 text-xs font-mono">
    <div className="flex items-center justify-between mb-3">
      <span className="text-slate-200 font-bold">{pos.symbol} -- Position Detail</span>
      <button onClick={onClose} className="text-slate-600 hover:text-slate-400">close x</button>
    </div>
    <div className="grid grid-cols-3 gap-4">
      <div className="space-y-2">
        <h4 className="text-[9px] text-slate-600 uppercase tracking-wide">Position</h4>
        {[
          ['Qty',        formatQty(pos.qty)],
          ['Avg Cost',   `$${pos.avgCost.toFixed(4)}`],
          ['Mkt Value',  formatUSD(pos.marketValue)],
          ['Weight',     formatPct(pos.weight)],
          ['Side',       pos.side.toUpperCase()],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between">
            <span className="text-slate-600">{k}</span>
            <span className="text-slate-300">{v}</span>
          </div>
        ))}
      </div>
      <div className="space-y-2">
        <h4 className="text-[9px] text-slate-600 uppercase tracking-wide">P&amp;L</h4>
        {[
          ['Unreal. P&L',  formatUSD(pos.unrealizedPnl)],
          ['Unreal. %',    formatPct(pos.unrealizedPct)],
          ['Daily P&L',    formatUSD(pos.dailyPnl)],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between">
            <span className="text-slate-600">{k}</span>
            <span style={{ color: colorForPnl(
              k === 'Unreal. P&L' ? pos.unrealizedPnl :
              k === 'Daily P&L'   ? pos.dailyPnl : 0
            ) }}>{v}</span>
          </div>
        ))}
      </div>
      <div className="space-y-2">
        <h4 className="text-[9px] text-slate-600 uppercase tracking-wide">Meta</h4>
        {[
          ['Sector',    pos.sector],
          ['Strategy',  pos.strategy],
          ['Beta',      pos.beta.toFixed(2)],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between">
            <span className="text-slate-600">{k}</span>
            <span className="text-slate-300">{v}</span>
          </div>
        ))}
      </div>
    </div>
  </div>
);

const PositionsTable: React.FC<{
  positions: PortfolioPosition[];
  loading:   boolean;
}> = ({ positions, loading }) => {
  const [selected, setSelected] = useState<PortfolioPosition | null>(null);

  const handleRowClick = (row: PortfolioPosition) => {
    setSelected(prev => prev?.symbol === row.symbol ? null : row);
  };

  return (
    <Card
      title="Positions"
      subtitle={`${positions.length} open`}
      padding="sm"
    >
      {loading ? (
        <div className="flex items-center justify-center h-24"><LoadingSpinner /></div>
      ) : (
        <>
          <SortableTable<PortfolioPosition>
            data={positions}
            columns={POSITION_COLS}
            defaultSortKey="marketValue"
            defaultSortDir="desc"
            rowKey={(r) => r.symbol}
            onRowClick={handleRowClick}
            maxHeight="360px"
            showFooter
            compact
          />
          {selected && (
            <PositionDetail pos={selected} onClose={() => setSelected(null)} />
          )}
        </>
      )}
    </Card>
  );
};

// ---------------------------------------------------------------------------
// Section 3: ExposureChart
// ---------------------------------------------------------------------------

const ExposureChart: React.FC<{
  exposure: SectorExposure[];
}> = ({ exposure }) => {
  const slices: DonutSlice[] = exposure.map((e, i) => ({
    label: e.sector,
    value: e.marketValue,
    color: SECTOR_COLORS[i % SECTOR_COLORS.length],
  }));

  return (
    <Card title="Sector Exposure" padding="sm">
      <div className="flex justify-center">
        <DonutChart
          data={slices}
          size={220}
          innerRadius={0.52}
          centerLabel="Gross"
          showLegend
          showPercentLabels
        />
      </div>
    </Card>
  );
};

// ---------------------------------------------------------------------------
// Section 4: PnLAttribution
// ---------------------------------------------------------------------------

interface AttributionEntry {
  label:       string;
  pnl:         number;
  pct:         number;
  dailyPnl:    number;
}

function buildAttribution(positions: PortfolioPosition[], groupBy: keyof PortfolioPosition): AttributionEntry[] {
  const map: Record<string, { pnl: number; daily: number }> = {};
  for (const p of positions) {
    const key = String(p[groupBy]);
    if (!map[key]) map[key] = { pnl: 0, daily: 0 };
    map[key].pnl   += p.unrealizedPnl;
    map[key].daily += p.dailyPnl;
  }
  const total = Object.values(map).reduce((s, v) => s + Math.abs(v.pnl), 0) || 1;
  return Object.entries(map)
    .map(([label, v]) => ({
      label,
      pnl:      v.pnl,
      pct:      v.pnl / total,
      dailyPnl: v.daily,
    }))
    .sort((a, b) => b.pnl - a.pnl);
}

const AttributionBar: React.FC<{
  entry:    AttributionEntry;
  maxAbs:   number;
  barWidth: number;
}> = ({ entry, maxAbs, barWidth }) => {
  const pct   = maxAbs > 0 ? Math.abs(entry.pnl) / maxAbs : 0;
  const color = colorForPnl(entry.pnl);
  return (
    <div className="flex items-center gap-2 py-1">
      <div className="w-20 text-[10px] font-mono text-slate-400 truncate flex-shrink-0">
        {entry.label}
      </div>
      <div className="flex-1 relative h-4">
        <div
          className="absolute left-0 top-0 h-full rounded-sm opacity-70"
          style={{
            width:      `${pct * 100}%`,
            background: color,
            transition: 'width 0.3s ease',
          }}
        />
      </div>
      <div className="w-20 text-right text-[10px] font-mono flex-shrink-0" style={{ color }}>
        {formatUSD(entry.pnl)}
      </div>
      <div className="w-14 text-right text-[10px] font-mono text-slate-600 flex-shrink-0">
        {formatUSD(entry.dailyPnl)}/d
      </div>
    </div>
  );
};

type AttribGroupBy = 'symbol' | 'sector' | 'strategy';

const PnLAttribution: React.FC<{ positions: PortfolioPosition[] }> = ({ positions }) => {
  const [groupBy, setGroupBy] = useState<AttribGroupBy>('symbol');

  const entries = useMemo(() => {
    if (!positions.length) return [];
    return buildAttribution(positions, groupBy as keyof PortfolioPosition);
  }, [positions, groupBy]);

  const maxAbs = useMemo(() => Math.max(...entries.map(e => Math.abs(e.pnl)), 1), [entries]);

  return (
    <Card
      title="P&L Attribution"
      actions={
        <div className="flex gap-1">
          {(['symbol', 'sector', 'strategy'] as AttribGroupBy[]).map(g => (
            <button
              key={g}
              onClick={() => setGroupBy(g)}
              className={clsx(
                'px-2 py-0.5 text-[9px] font-mono rounded uppercase tracking-wide transition-colors',
                groupBy === g
                  ? 'bg-blue-600/30 border border-blue-600/40 text-blue-300'
                  : 'text-slate-600 hover:text-slate-400',
              )}
            >
              {g}
            </button>
          ))}
        </div>
      }
      padding="sm"
    >
      <div className="space-y-0.5">
        {entries.map(e => (
          <AttributionBar
            key={e.label}
            entry={e}
            maxAbs={maxAbs}
            barWidth={100}
          />
        ))}
        {entries.length === 0 && (
          <p className="text-slate-600 text-xs font-mono text-center py-8">No data</p>
        )}
      </div>
    </Card>
  );
};

// ---------------------------------------------------------------------------
// Section 5: Risk meters
// ---------------------------------------------------------------------------

const RiskMeters: React.FC<{ m: PortfolioMetrics }> = ({ m }) => {
  const meters = [
    {
      label:       'VaR Utilization',
      value:       m.varUtil,
      color:       colorForUtilization(m.varUtil),
      description: `${formatPct(m.varUtil)} of daily VaR limit`,
    },
    {
      label:       'Current Drawdown',
      value:       m.currentDrawdown,
      color:       colorForUtilization(m.currentDrawdown / 0.10),
      description: `${formatPct(m.currentDrawdown)} (max: ${formatPct(m.maxDrawdown)})`,
    },
    {
      label:       'Margin Utilization',
      value:       m.marginUtil,
      color:       colorForUtilization(m.marginUtil),
      description: `${formatPct(m.marginUtil)} of available margin`,
    },
    {
      label:       'Leverage',
      value:       Math.min(m.leverage / 2.0, 1),
      color:       m.leverage > 1.5 ? '#fbbf24' : '#34d399',
      description: `${formatLeverage(m.leverage)} (limit: 2.0x)`,
    },
  ];

  return (
    <Card title="Risk Meters" padding="sm">
      <div className="space-y-4">
        {meters.map(meter => (
          <div key={meter.label}>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-mono text-slate-400">{meter.label}</span>
              <span className="text-[10px] font-mono" style={{ color: meter.color }}>
                {meter.description}
              </span>
            </div>
            <div className="w-full bg-[#1e2130] rounded-full overflow-hidden h-2">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width:      `${Math.min(meter.value * 100, 100)}%`,
                  background: meter.color,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Summary row */}
      <div className="mt-4 pt-3 border-t border-[#1e2130] grid grid-cols-2 gap-3">
        {[
          { label: 'Max Drawdown',  value: formatPct(m.maxDrawdown),  color: '#f87171' },
          { label: 'Sharpe YTD',   value: formatRatio(m.sharpeYTD),  color: m.sharpeYTD >= 1 ? '#34d399' : '#fbbf24' },
        ].map(item => (
          <div key={item.label} className="text-center">
            <div className="text-[9px] font-mono text-slate-600 uppercase tracking-wide">{item.label}</div>
            <div className="text-sm font-mono font-bold mt-0.5" style={{ color: item.color }}>{item.value}</div>
          </div>
        ))}
      </div>
    </Card>
  );
};

// ---------------------------------------------------------------------------
// PortfolioDashboard (main page)
// ---------------------------------------------------------------------------

const PortfolioDashboard: React.FC = () => {
  const { data: positions, isLoading: posLoading } = usePositions();
  const { data: metrics,   isLoading: metLoading } = usePortfolioMetrics();
  const exposure = useExposure();

  const safePositions = positions ?? [];
  const safeMetrics   = metrics ?? {
    totalNav:        0, cashBalance:    0, grossExposure:  0,
    netExposure:     0, leverage:       0, dailyPnl:       0,
    dailyPnlPct:     0, ytdPnl:         0, ytdPnlPct:      0,
    sharpeYTD:       0, maxDrawdown:    0, currentDrawdown:0,
    varUtil:         0, marginUtil:     0, weeklyPnl:      0,
    monthlyPnl:      0,
  };

  return (
    <div className="min-h-screen bg-[#0d1017] text-slate-200 p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-base font-mono font-bold text-slate-100 uppercase tracking-widest">
            Portfolio Overview
          </h1>
          <p className="text-[10px] font-mono text-slate-600 mt-0.5">
            Positions &bull; Exposure &bull; Attribution &bull; Risk
          </p>
        </div>
        {metLoading && <LoadingSpinner size={16} />}
      </div>

      {/* Section 1: Metrics bar */}
      {!metLoading && <MetricsBar m={safeMetrics} />}

      {/* Section 2+3: Positions table + Exposure chart */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2">
          <PositionsTable positions={safePositions} loading={posLoading} />
        </div>
        <div className="flex flex-col gap-4">
          <ExposureChart exposure={exposure} />
          <RiskMeters m={safeMetrics} />
        </div>
      </div>

      {/* Section 4: P&L attribution */}
      <PnLAttribution positions={safePositions} />
    </div>
  );
};

export default PortfolioDashboard;
