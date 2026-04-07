// ============================================================
// pages/ExecutionDashboard.tsx -- Execution analytics dashboard
//
// Sections:
//   1. Real-time order feed (WebSocket, last 50 orders)
//   2. TCA metrics panel (IS bps, VWAP slippage, spread cost)
//   3. Venue scorecard table (sortable, color-coded scores)
//   4. 30-day rolling cost trend chart
// ============================================================

import React, { useState, useMemo, useCallback } from 'react';
import { clsx } from 'clsx';
import {
  useRecentOrders,
  useTCAResults,
  useVenueScorecard,
  useExecutionSummary,
  useExecutionCostTrend,
  useOrderStream,
  useRefreshExec,
} from '../hooks/useExecutionAPI';
import { SortableTable } from '../components/SortableTable';
import type { ColumnDef } from '../components/SortableTable';
import { WebSocketStatus } from '../components/WebSocketStatus';
import type { WSConnectionState } from '../components/WebSocketStatus';
import { TimeSeriesChart } from '../components/TimeSeriesChart';
import { Card, Badge, LoadingSpinner } from '../components/ui';
import {
  formatBps,
  formatBpsAbs,
  formatUSD,
  formatTimestamp,
  formatTime,
  formatDuration,
  formatPct,
  colorForPnl,
  colorForScore,
  bgForScore,
  colorForSlippage,
  formatQty,
} from '../utils/formatters';
import type {
  OrderRecord,
  TCAResult,
  VenueScore,
  OrderStatus,
} from '../types/execution';

// ---------------------------------------------------------------------------
// Color helpers for order status
// ---------------------------------------------------------------------------

function statusColor(status: OrderStatus): string {
  switch (status) {
    case 'filled':   return '#34d399'; // green
    case 'pending':  return '#fbbf24'; // yellow
    case 'partial':  return '#60a5fa'; // blue
    case 'rejected': return '#f87171'; // red
    case 'cancelled':return '#94a3b8'; // slate
    default:         return '#94a3b8';
  }
}

function statusBadgeVariant(status: OrderStatus): 'bull' | 'bear' | 'warning' | 'neutral' | 'info' {
  switch (status) {
    case 'filled':    return 'bull';
    case 'pending':   return 'warning';
    case 'partial':   return 'info';
    case 'rejected':  return 'bear';
    case 'cancelled': return 'neutral';
    default:          return 'neutral';
  }
}

// ---------------------------------------------------------------------------
// Section 1: OrderFeedPanel
// ---------------------------------------------------------------------------

const ORDER_COLS: ColumnDef<OrderRecord>[] = [
  {
    key:    'orderTime',
    header: 'Time',
    width:  '80px',
    render: (v) => (
      <span className="text-slate-400">{formatTime(v as string)}</span>
    ),
  },
  {
    key:    'symbol',
    header: 'Symbol',
    width:  '90px',
    render: (v) => (
      <span className="text-slate-200 font-semibold">{v as string}</span>
    ),
  },
  {
    key:    'side',
    header: 'Side',
    width:  '56px',
    render: (v) => (
      <span className={clsx(
        'font-semibold uppercase text-[10px]',
        v === 'buy' ? 'text-emerald-400' : 'text-red-400',
      )}>
        {v as string}
      </span>
    ),
  },
  {
    key:    'qty',
    header: 'Qty',
    align:  'right',
    width:  '80px',
    render: (v, row) => (
      <span>
        {formatQty(row.filledQty)}/{formatQty(v as number)}
      </span>
    ),
  },
  {
    key:    'fillPrice',
    header: 'Fill Px',
    align:  'right',
    width:  '80px',
    render: (v) => v == null
      ? <span className="text-slate-600">--</span>
      : <span>${(v as number).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 })}</span>,
  },
  {
    key:    'slippageBps',
    header: 'Slip',
    align:  'right',
    width:  '70px',
    render: (v) => {
      if (v == null) return <span className="text-slate-600">--</span>;
      const bps = v as number;
      return (
        <span style={{ color: colorForSlippage(Math.abs(bps)) }}>
          {formatBps(bps)}
        </span>
      );
    },
  },
  {
    key:    'venue',
    header: 'Venue',
    width:  '80px',
    render: (v) => <span className="text-slate-400">{v as string}</span>,
  },
  {
    key:    'strategy',
    header: 'Strategy',
    width:  '100px',
    render: (v) => <span className="text-slate-500 text-[10px]">{v as string}</span>,
  },
  {
    key:    'status',
    header: 'Status',
    width:  '80px',
    sortable: true,
    render: (v) => (
      <Badge variant={statusBadgeVariant(v as OrderStatus)}>
        {(v as string).toUpperCase()}
      </Badge>
    ),
  },
];

interface OrderFeedPanelProps {
  streamOrders: OrderRecord[];
  isConnected:  boolean;
  lastUpdate:   Date | null;
}

const OrderFeedPanel: React.FC<OrderFeedPanelProps> = ({
  streamOrders,
  isConnected,
  lastUpdate,
}) => {
  const [selectedOrder, setSelectedOrder] = useState<OrderRecord | null>(null);

  const wsState: WSConnectionState = isConnected ? 'connected' : 'reconnecting';

  return (
    <Card
      title="Live Order Feed"
      subtitle={`${streamOrders.length} orders`}
      actions={
        <WebSocketStatus
          state={wsState}
          lastUpdate={lastUpdate}
          compact
        />
      }
      padding="sm"
    >
      <SortableTable<OrderRecord>
        data={streamOrders}
        columns={ORDER_COLS}
        defaultSortKey="orderTime"
        defaultSortDir="desc"
        rowKey={(row) => row.id}
        onRowClick={setSelectedOrder}
        maxHeight="320px"
        compact
        emptyMessage="Waiting for orders..."
      />

      {selectedOrder && (
        <OrderDetailDrawer
          order={selectedOrder}
          onClose={() => setSelectedOrder(null)}
        />
      )}
    </Card>
  );
};

// ---------------------------------------------------------------------------
// Order detail drawer (expanded row)
// ---------------------------------------------------------------------------

const OrderDetailDrawer: React.FC<{
  order:   OrderRecord;
  onClose: () => void;
}> = ({ order, onClose }) => (
  <div className="mt-3 p-3 rounded-lg bg-[#0d1017] border border-[#1e2130] text-xs font-mono">
    <div className="flex items-center justify-between mb-2">
      <span className="text-slate-300 font-semibold">{order.id}</span>
      <button
        onClick={onClose}
        className="text-slate-600 hover:text-slate-400 text-[11px]"
      >
        close
      </button>
    </div>
    <div className="grid grid-cols-3 gap-x-6 gap-y-1 text-[10px]">
      {[
        ['Symbol',    order.symbol],
        ['Side',      order.side.toUpperCase()],
        ['Type',      order.orderType],
        ['Qty',       formatQty(order.qty)],
        ['Filled',    formatQty(order.filledQty)],
        ['Order Px',  order.orderPrice != null ? `$${order.orderPrice.toFixed(4)}` : '--'],
        ['Fill Px',   order.fillPrice  != null ? `$${order.fillPrice.toFixed(4)}`  : '--'],
        ['Venue',     order.venue],
        ['Strategy',  order.strategy],
        ['Slippage',  order.slippageBps != null ? formatBps(order.slippageBps) : '--'],
        ['Order Time',formatTimestamp(order.orderTime)],
        ['Fill Time', order.fillTime ? formatTimestamp(order.fillTime) : '--'],
      ].map(([label, val]) => (
        <div key={label} className="flex justify-between gap-2 border-b border-[#151820] py-0.5">
          <span className="text-slate-600">{label}</span>
          <span className="text-slate-300">{val}</span>
        </div>
      ))}
    </div>
    {order.rejectReason && (
      <div className="mt-2 text-red-400 text-[10px]">
        Reject: {order.rejectReason}
      </div>
    )}
  </div>
);

// ---------------------------------------------------------------------------
// Section 2: TCAMetricsPanel
// ---------------------------------------------------------------------------

interface TCAMetricsPanelProps {
  tca:     TCAResult[];
  loading: boolean;
}

const TCAMetricsPanel: React.FC<TCAMetricsPanelProps> = ({ tca, loading }) => {
  const metrics = useMemo(() => {
    if (!tca.length) return null;
    const avg = (fn: (t: TCAResult) => number) =>
      tca.reduce((s, t) => s + fn(t), 0) / tca.length;
    return {
      avgIS:      avg(t => t.implShortfallBps),
      avgVwap:    avg(t => t.vwapSlippageBps),
      avgSpread:  avg(t => t.spreadCostBps),
      avgImpact:  avg(t => t.marketImpactBps),
      avgTotal:   avg(t => t.totalCostBps),
      totalNotional: tca.reduce((s, t) => s + t.notionalUsd, 0),
    };
  }, [tca]);

  if (loading) return (
    <Card title="TCA Metrics" padding="sm">
      <div className="flex items-center justify-center h-24">
        <LoadingSpinner />
      </div>
    </Card>
  );

  if (!metrics) return null;

  const cards = [
    { label: 'Avg Impl. Shortfall', value: formatBpsAbs(metrics.avgIS),     color: colorForSlippage(metrics.avgIS) },
    { label: 'Avg VWAP Slippage',   value: formatBpsAbs(metrics.avgVwap),   color: colorForSlippage(Math.abs(metrics.avgVwap)) },
    { label: 'Avg Spread Cost',     value: formatBpsAbs(metrics.avgSpread),  color: colorForSlippage(metrics.avgSpread) },
    { label: 'Avg Market Impact',   value: formatBpsAbs(metrics.avgImpact),  color: colorForSlippage(metrics.avgImpact) },
    { label: 'Avg Total Cost',      value: formatBpsAbs(metrics.avgTotal),   color: colorForSlippage(metrics.avgTotal) },
    { label: 'Total Notional',      value: formatUSD(metrics.totalNotional), color: '#94a3b8' },
  ];

  return (
    <Card
      title="TCA Metrics"
      subtitle={`${tca.length} fills analyzed`}
      padding="sm"
    >
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
        {cards.map(c => (
          <div
            key={c.label}
            className="bg-[#0d1017] rounded-lg p-3 border border-[#1e2130]"
          >
            <div className="text-[10px] font-mono text-slate-500 mb-1 leading-tight">
              {c.label}
            </div>
            <div
              className="text-sm font-mono font-bold"
              style={{ color: c.color }}
            >
              {c.value}
            </div>
          </div>
        ))}
      </div>

      {/* Sparkline breakdown */}
      <div className="mt-3 grid grid-cols-3 gap-2">
        {[
          { label: 'IS by Venue',    key: 'implShortfallBps' },
          { label: 'VWAP Slip',      key: 'vwapSlippageBps' },
          { label: 'Spread Cost',    key: 'spreadCostBps' },
        ].map(({ label, key }) => {
          const sorted = [...tca]
            .sort((a, b) => new Date(a.fillTime).getTime() - new Date(b.fillTime).getTime())
            .slice(-30)
            .map(t => ({
              fillTime: t.fillTime,
              [key]:    (t as any)[key] as number,
            }));
          return (
            <div key={key} className="bg-[#0d1017] rounded border border-[#151820] p-2">
              <p className="text-[9px] font-mono text-slate-500 uppercase tracking-wide mb-1">{label}</p>
              <TimeSeriesChart
                data={sorted}
                xKey="fillTime"
                series={[{
                  key:         key,
                  label:       label,
                  color:       '#60a5fa',
                  type:        'area',
                  fillOpacity: 0.1,
                  strokeWidth: 1,
                }]}
                height={50}
                showGrid={false}
                showLegend={false}
                showTooltip
                compact
                xFormat="HH:mm"
              />
            </div>
          );
        })}
      </div>
    </Card>
  );
};

// ---------------------------------------------------------------------------
// Section 3: VenueScorecardTable
// ---------------------------------------------------------------------------

const VENUE_COLS: ColumnDef<VenueScore>[] = [
  {
    key:    'venue',
    header: 'Venue',
    width:  '100px',
    render: (v) => <span className="text-slate-200 font-semibold">{v as string}</span>,
  },
  {
    key:    'avgSlippageBps',
    header: 'Avg Slip',
    align:  'right',
    render: (v) => (
      <span style={{ color: colorForSlippage(v as number) }}>
        {formatBpsAbs(v as number)}
      </span>
    ),
  },
  {
    key:    'fillRate',
    header: 'Fill Rate',
    align:  'right',
    render: (v) => (
      <span style={{ color: (v as number) >= 0.95 ? '#34d399' : (v as number) >= 0.85 ? '#fbbf24' : '#f87171' }}>
        {formatPct(v as number)}
      </span>
    ),
  },
  {
    key:    'avgFillTimeMs',
    header: 'Avg Time',
    align:  'right',
    render: (v) => <span className="text-slate-400">{formatDuration(v as number)}</span>,
  },
  {
    key:    'p95FillTimeMs',
    header: 'P95 Time',
    align:  'right',
    render: (v) => <span className="text-slate-500">{formatDuration(v as number)}</span>,
  },
  {
    key:    'avgMarketImpactBps',
    header: 'Impact',
    align:  'right',
    render: (v) => (
      <span style={{ color: colorForSlippage(v as number) }}>
        {formatBpsAbs(v as number)}
      </span>
    ),
  },
  {
    key:    'nTrades',
    header: 'n Trades',
    align:  'right',
    render: (v) => <span className="text-slate-400">{(v as number).toLocaleString()}</span>,
  },
  {
    key:    'score',
    header: 'Score',
    align:  'center',
    width:  '70px',
    render: (v) => {
      const s = v as number;
      return (
        <span
          className="inline-block px-2 py-0.5 rounded text-[10px] font-mono font-bold"
          style={{
            color:           colorForScore(s),
            backgroundColor: bgForScore(s),
          }}
        >
          {s.toFixed(0)}
        </span>
      );
    },
  },
];

const VenueScorecardTable: React.FC<{
  venues:  VenueScore[];
  loading: boolean;
}> = ({ venues, loading }) => (
  <Card
    title="Venue Scorecard"
    subtitle="30-day lookback"
    padding="sm"
  >
    {loading ? (
      <div className="flex items-center justify-center h-20">
        <LoadingSpinner />
      </div>
    ) : (
      <SortableTable<VenueScore>
        data={venues}
        columns={VENUE_COLS}
        defaultSortKey="score"
        defaultSortDir="desc"
        rowKey={(r) => r.venue}
        showFooter
        emptyMessage="No venue data"
        compact
      />
    )}
  </Card>
);

// ---------------------------------------------------------------------------
// Section 4: CostTrendChart
// ---------------------------------------------------------------------------

const CostTrendChart: React.FC<{
  loading: boolean;
  data:    Array<{ date: string; avgCostBpsDaily: number; avgCostBps7d: number; nFills: number }>;
}> = ({ loading, data }) => (
  <Card
    title="30-Day TCA Cost Trend"
    subtitle="Daily avg cost vs 7-day rolling avg (bps)"
    padding="sm"
  >
    {loading ? (
      <div className="flex items-center justify-center h-32">
        <LoadingSpinner />
      </div>
    ) : (
      <TimeSeriesChart
        data={data}
        xKey="date"
        xFormat="MMM d"
        series={[
          {
            key:         'avgCostBpsDaily',
            label:       'Daily Avg Cost',
            color:       '#60a5fa',
            type:        'bar',
            yAxisId:     'left',
          },
          {
            key:         'avgCostBps7d',
            label:       '7d Rolling Avg',
            color:       '#f59e0b',
            type:        'line',
            strokeWidth: 2,
            yAxisId:     'left',
          },
          {
            key:         'nFills',
            label:       'Fill Count',
            color:       '#8b5cf6',
            type:        'line',
            strokeWidth: 1,
            strokeDasharray: '3 3',
            yAxisId:     'right',
            hidden:      true,
          },
        ]}
        leftLabel="bps"
        rightLabel="fills"
        height={220}
        showGrid
        showLegend
        showTooltip
        tooltipFormatter={(val, key) =>
          key === 'nFills' ? val.toFixed(0) : `${val.toFixed(2)} bps`
        }
      />
    )}
  </Card>
);

// ---------------------------------------------------------------------------
// Top summary bar
// ---------------------------------------------------------------------------

const SummaryBar: React.FC<{
  date:          string;
  totalOrders:   number;
  filledOrders:  number;
  avgSlippageBps:number;
  totalCostBps:  number;
  bestVenue:     string;
  worstVenue:    string;
  fillRate:      number;
  loading:       boolean;
}> = ({ date, totalOrders, filledOrders, avgSlippageBps, totalCostBps,
        bestVenue, worstVenue, fillRate, loading }) => {
  if (loading) return null;
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
      {[
        { label: 'Date',         value: date },
        { label: 'Total Orders', value: totalOrders.toLocaleString() },
        { label: 'Filled',       value: filledOrders.toLocaleString() },
        { label: 'Fill Rate',    value: formatPct(fillRate), color: fillRate >= 0.9 ? '#34d399' : '#fbbf24' },
        { label: 'Avg Slip',     value: formatBpsAbs(avgSlippageBps), color: colorForSlippage(avgSlippageBps) },
        { label: 'Best Venue',   value: bestVenue,  color: '#34d399' },
        { label: 'Worst Venue',  value: worstVenue, color: '#f87171' },
      ].map(item => (
        <div
          key={item.label}
          className="bg-[#111318] border border-[#1e2130] rounded-lg px-3 py-2"
        >
          <div className="text-[9px] font-mono text-slate-600 uppercase tracking-wider mb-1">
            {item.label}
          </div>
          <div
            className="text-sm font-mono font-bold"
            style={{ color: (item as any).color ?? '#e2e8f0' }}
          >
            {item.value}
          </div>
        </div>
      ))}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Symbol / days filter controls
// ---------------------------------------------------------------------------

const SYMBOL_OPTIONS = ['', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'AAPL', 'SPY', 'QQQ', 'MSFT', 'TSLA'];
const DAYS_OPTIONS   = [1, 3, 7, 14, 30];

// ---------------------------------------------------------------------------
// ExecutionDashboard (main page)
// ---------------------------------------------------------------------------

const ExecutionDashboard: React.FC = () => {
  const [filterSymbol, setFilterSymbol] = useState<string>('');
  const [filterDays,   setFilterDays]   = useState<number>(7);

  const { orders:    streamOrders, isConnected, lastUpdate } = useOrderStream(50);
  const { data: tca, isLoading: tcaLoading }   = useTCAResults(filterSymbol || undefined, filterDays);
  const { data: venues, isLoading: venLoading } = useVenueScorecard();
  const { data: summary, isLoading: sumLoading }= useExecutionSummary();
  const { data: costTrend, isLoading: trendLoading } = useExecutionCostTrend(30);
  const refresh = useRefreshExec();

  const filteredTCA = useMemo(() => {
    if (!tca) return [];
    if (!filterSymbol) return tca;
    return tca.filter(t => t.symbol === filterSymbol);
  }, [tca, filterSymbol]);

  return (
    <div className="min-h-screen bg-[#0d1017] text-slate-200 p-4 space-y-4">
      {/* Page header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-base font-mono font-bold text-slate-100 uppercase tracking-widest">
            Execution Analytics
          </h1>
          <p className="text-[10px] font-mono text-slate-600 mt-0.5">
            TCA &bull; Venue Scorecard &bull; Order Flow
          </p>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <label className="text-[10px] font-mono text-slate-500 uppercase">Symbol</label>
            <select
              value={filterSymbol}
              onChange={(e) => setFilterSymbol(e.target.value)}
              className="bg-[#1a1d26] border border-[#1e2130] text-slate-300 text-xs font-mono rounded px-2 py-1 focus:outline-none focus:border-blue-500/50"
            >
              {SYMBOL_OPTIONS.map(s => (
                <option key={s} value={s}>{s || 'All'}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-[10px] font-mono text-slate-500 uppercase">Days</label>
            <select
              value={filterDays}
              onChange={(e) => setFilterDays(Number(e.target.value))}
              className="bg-[#1a1d26] border border-[#1e2130] text-slate-300 text-xs font-mono rounded px-2 py-1 focus:outline-none focus:border-blue-500/50"
            >
              {DAYS_OPTIONS.map(d => (
                <option key={d} value={d}>{d}d</option>
              ))}
            </select>
          </div>

          <button
            onClick={refresh}
            className="px-3 py-1 text-[10px] font-mono rounded bg-blue-600/20 border border-blue-600/30 text-blue-400 hover:bg-blue-600/30 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Summary bar */}
      <SummaryBar
        date={summary?.date ?? '--'}
        totalOrders={summary?.totalOrders ?? 0}
        filledOrders={summary?.filledOrders ?? 0}
        avgSlippageBps={summary?.avgSlippageBps ?? 0}
        totalCostBps={summary?.totalCostBps ?? 0}
        bestVenue={summary?.bestVenue ?? '--'}
        worstVenue={summary?.worstVenue ?? '--'}
        fillRate={summary?.fillRate ?? 0}
        loading={sumLoading}
      />

      {/* Main grid: 2 columns on large screens */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {/* Left column: order feed + TCA metrics */}
        <div className="space-y-4">
          <OrderFeedPanel
            streamOrders={streamOrders}
            isConnected={isConnected}
            lastUpdate={lastUpdate}
          />
          <TCAMetricsPanel tca={filteredTCA} loading={tcaLoading} />
        </div>

        {/* Right column: venue scorecard + cost trend */}
        <div className="space-y-4">
          <VenueScorecardTable venues={venues ?? []} loading={venLoading} />
          <CostTrendChart
            loading={trendLoading}
            data={costTrend ?? []}
          />
        </div>
      </div>
    </div>
  );
};

export default ExecutionDashboard;
