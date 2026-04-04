import { useState, useMemo } from 'react';
import type { Trade } from '../types';
import { RegimeBadge } from './RegimeBadge';

interface Props {
  trades: Trade[];
  maxRows?: number;
}

type SortKey = keyof Trade;
type SortDir = 'asc' | 'desc';

function fmtDate(s: string) {
  try { return new Date(s).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' }); }
  catch { return s; }
}

function fmtPrice(v: number) { return `$${v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`; }

function PnLCell({ value }: { value: number }) {
  const cls = value >= 0 ? 'text-bull' : 'text-bear';
  return <span className={`font-mono ${cls}`}>{value >= 0 ? '+' : ''}{value.toFixed(2)}</span>;
}

function ExpandedRow({ trade }: { trade: Trade }) {
  return (
    <tr className="bg-bg-base">
      <td colSpan={11} className="px-4 py-2">
        <div className="flex gap-6 text-xs font-mono text-gray-400">
          <span>BH Mass at Entry: <span className="text-bh-warm">{trade.bh_mass_at_entry?.toFixed(3) ?? '—'}</span></span>
          <span>Pos Floor: <span className="text-accent">{trade.pos_floor?.toFixed(4) ?? '—'}</span></span>
          <span>Entry: {fmtPrice(trade.entry_price)}</span>
          <span>Exit: {fmtPrice(trade.exit_price)}</span>
          <span>Hold: {trade.hold_bars} bars</span>
        </div>
      </td>
    </tr>
  );
}

export function TradeTable({ trades, maxRows = 200 }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('entry_date');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const sorted = useMemo(() => {
    return [...trades].sort((a, b) => {
      const av = a[sortKey] as number | string;
      const bv = b[sortKey] as number | string;
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortDir === 'asc' ? cmp : -cmp;
    }).slice(0, maxRows);
  }, [trades, sortKey, sortDir, maxRows]);

  function handleSort(key: SortKey) {
    if (key === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  }

  function toggleExpand(id: string) {
    setExpanded(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function SortHeader({ k, label }: { k: SortKey; label: string }) {
    const active = sortKey === k;
    return (
      <th
        className={`px-3 py-2 text-left text-xs font-mono uppercase tracking-wider cursor-pointer select-none whitespace-nowrap
          ${active ? 'text-accent' : 'text-gray-500'} hover:text-gray-300 transition-colors`}
        onClick={() => handleSort(k)}
      >
        {label} {active ? (sortDir === 'asc' ? '↑' : '↓') : ''}
      </th>
    );
  }

  if (!trades.length) {
    return <p className="text-center py-8 text-gray-600 text-sm font-mono">No trades</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="border-b border-bg-border">
          <tr>
            <th className="w-6" />
            <SortHeader k="entry_date" label="Date" />
            <SortHeader k="sym" label="Sym" />
            <SortHeader k="entry_price" label="Entry" />
            <SortHeader k="exit_price" label="Exit" />
            <SortHeader k="pnl_dollar" label="P&L $" />
            <SortHeader k="pnl_pct" label="P&L %" />
            <SortHeader k="hold_bars" label="Bars" />
            <SortHeader k="mfe" label="MFE%" />
            <SortHeader k="mae" label="MAE%" />
            <SortHeader k="tf_score" label="TF" />
            <th className="px-3 py-2 text-left text-xs font-mono uppercase tracking-wider text-gray-500">Regime</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(trade => (
            <>
              <tr
                key={trade.id}
                className="border-b border-bg-border/50 hover:bg-bg-elevated cursor-pointer transition-colors"
                onClick={() => toggleExpand(trade.id)}
              >
                <td className="pl-3 text-gray-600 text-xs">
                  {expanded.has(trade.id) ? '▼' : '▶'}
                </td>
                <td className="px-3 py-2 font-mono text-xs text-gray-400">{fmtDate(trade.entry_date)}</td>
                <td className="px-3 py-2 font-mono text-sm font-semibold text-gray-200">{trade.sym}</td>
                <td className="px-3 py-2 font-mono text-xs text-gray-400">{fmtPrice(trade.entry_price)}</td>
                <td className="px-3 py-2 font-mono text-xs text-gray-400">{fmtPrice(trade.exit_price)}</td>
                <td className="px-3 py-2"><PnLCell value={trade.pnl_dollar} /></td>
                <td className="px-3 py-2"><PnLCell value={trade.pnl_pct} /></td>
                <td className="px-3 py-2 font-mono text-xs text-gray-400">{trade.hold_bars}</td>
                <td className="px-3 py-2 font-mono text-xs text-bull">{trade.mfe.toFixed(1)}%</td>
                <td className="px-3 py-2 font-mono text-xs text-bear">{trade.mae.toFixed(1)}%</td>
                <td className="px-3 py-2 font-mono text-xs text-accent">{trade.tf_score.toFixed(2)}</td>
                <td className="px-3 py-2"><RegimeBadge regime={trade.regime} /></td>
              </tr>
              {expanded.has(trade.id) && <ExpandedRow key={`${trade.id}-exp`} trade={trade} />}
            </>
          ))}
        </tbody>
      </table>
      {trades.length > maxRows && (
        <p className="text-center py-2 text-xs text-gray-600 font-mono">
          Showing {maxRows} of {trades.length} trades
        </p>
      )}
    </div>
  );
}
