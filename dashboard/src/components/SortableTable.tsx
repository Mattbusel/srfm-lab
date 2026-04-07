// ============================================================
// components/SortableTable.tsx -- Generic sortable data table.
// Supports click-to-sort on any column, alternating row colors,
// optional row click handler, sticky header, and custom cell
// renderers via ColumnDef.
// ============================================================

import React, { useState, useMemo, useCallback, ReactNode } from 'react';
import { clsx } from 'clsx';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ColumnDef<T> {
  /** Property key on T (also used for sort). */
  key: keyof T;
  /** Column header label. */
  header: string;
  /** Custom cell renderer. Receives the raw value and full row. */
  render?: (value: T[keyof T], row: T) => ReactNode;
  /** Whether this column is sortable. Defaults to true. */
  sortable?: boolean;
  /** Optional CSS width string, e.g. "120px" or "15%". */
  width?: string;
  /** Text alignment for the column. Defaults to "left". */
  align?: 'left' | 'center' | 'right';
}

export type SortDir = 'asc' | 'desc';

export interface SortableTableProps<T> {
  data: T[];
  columns: ColumnDef<T>[];
  defaultSortKey?: keyof T;
  defaultSortDir?: SortDir;
  onRowClick?: (row: T) => void;
  /** Unique key extractor for rows. Defaults to row index. */
  rowKey?: (row: T, index: number) => string | number;
  /** Additional class on the outer wrapper. */
  className?: string;
  /** Max height before the table scrolls. */
  maxHeight?: string;
  /** Show row count in the footer. */
  showFooter?: boolean;
  /** Placeholder when data is empty. */
  emptyMessage?: string;
  /** Compact row height. */
  compact?: boolean;
}

// ---------------------------------------------------------------------------
// Sort helpers
// ---------------------------------------------------------------------------

function defaultCompare<T>(a: T, b: T, key: keyof T, dir: SortDir): number {
  const va = a[key];
  const vb = b[key];

  if (va == null && vb == null) return 0;
  if (va == null) return dir === 'asc' ? -1 : 1;
  if (vb == null) return dir === 'asc' ? 1 : -1;

  if (typeof va === 'number' && typeof vb === 'number') {
    return dir === 'asc' ? va - vb : vb - va;
  }

  const sa = String(va).toLowerCase();
  const sb = String(vb).toLowerCase();
  const cmp = sa < sb ? -1 : sa > sb ? 1 : 0;
  return dir === 'asc' ? cmp : -cmp;
}

// ---------------------------------------------------------------------------
// Sort direction indicator
// ---------------------------------------------------------------------------

const SortIndicator: React.FC<{ dir: SortDir | null }> = ({ dir }) => {
  if (!dir) return (
    <span className="ml-1 text-slate-700 text-[10px] select-none">&#x25B4;&#x25BE;</span>
  );
  return (
    <span className="ml-1 text-blue-400 text-[10px] select-none">
      {dir === 'asc' ? '\u25B4' : '\u25BE'}
    </span>
  );
};

// ---------------------------------------------------------------------------
// SortableTable
// ---------------------------------------------------------------------------

export function SortableTable<T extends object>({
  data,
  columns,
  defaultSortKey,
  defaultSortDir = 'desc',
  onRowClick,
  rowKey,
  className,
  maxHeight = '480px',
  showFooter = false,
  emptyMessage = 'No data',
  compact = false,
}: SortableTableProps<T>): React.ReactElement {
  const [sortKey, setSortKey] = useState<keyof T | null>(defaultSortKey ?? null);
  const [sortDir, setSortDir] = useState<SortDir>(defaultSortDir);

  const handleHeaderClick = useCallback((key: keyof T, sortable: boolean) => {
    if (!sortable) return;
    setSortKey(prev => {
      if (prev === key) {
        setSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
        return key;
      }
      setSortDir('desc');
      return key;
    });
  }, []);

  const sorted = useMemo(() => {
    if (!sortKey) return data;
    return [...data].sort((a, b) => defaultCompare(a, b, sortKey, sortDir));
  }, [data, sortKey, sortDir]);

  const rowH = compact ? 'py-1.5 px-3' : 'py-2.5 px-4';

  return (
    <div className={clsx('flex flex-col', className)}>
      <div
        className="overflow-auto rounded-lg border border-[#1e2130]"
        style={{ maxHeight }}
      >
        <table className="w-full border-collapse text-xs font-mono">
          {/* Sticky header */}
          <thead className="sticky top-0 z-10">
            <tr className="bg-[#0d1017] border-b border-[#1e2130]">
              {columns.map(col => {
                const isSortable = col.sortable !== false;
                const isActive   = sortKey === col.key;
                const align      = col.align ?? 'left';
                return (
                  <th
                    key={String(col.key)}
                    style={{ width: col.width }}
                    className={clsx(
                      'select-none whitespace-nowrap',
                      compact ? 'py-1.5 px-3' : 'py-2.5 px-4',
                      align === 'right'  && 'text-right',
                      align === 'center' && 'text-center',
                      align === 'left'   && 'text-left',
                      isSortable
                        ? 'cursor-pointer text-slate-400 hover:text-slate-200 transition-colors'
                        : 'text-slate-500 cursor-default',
                      isActive && 'text-slate-200',
                      'text-[10px] uppercase tracking-wider font-semibold',
                    )}
                    onClick={() => handleHeaderClick(col.key, isSortable)}
                  >
                    {col.header}
                    {isSortable && (
                      <SortIndicator dir={isActive ? sortDir : null} />
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>

          <tbody>
            {sorted.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  className="text-center py-12 text-slate-600 font-mono text-xs"
                >
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              sorted.map((row, idx) => {
                const key = rowKey ? rowKey(row, idx) : idx;
                const isEven = idx % 2 === 0;
                return (
                  <tr
                    key={key}
                    onClick={() => onRowClick?.(row)}
                    className={clsx(
                      'border-b border-[#151820] transition-colors',
                      isEven ? 'bg-[#0f1219]' : 'bg-[#111318]',
                      onRowClick
                        ? 'cursor-pointer hover:bg-[#1a2035]'
                        : 'hover:bg-[#141824]',
                    )}
                  >
                    {columns.map(col => {
                      const val    = row[col.key];
                      const align  = col.align ?? 'left';
                      const rendered = col.render
                        ? col.render(val, row)
                        : val == null
                        ? <span className="text-slate-600">--</span>
                        : String(val);

                      return (
                        <td
                          key={String(col.key)}
                          className={clsx(
                            rowH,
                            'text-slate-300 whitespace-nowrap',
                            align === 'right'  && 'text-right',
                            align === 'center' && 'text-center',
                            align === 'left'   && 'text-left',
                          )}
                        >
                          {rendered}
                        </td>
                      );
                    })}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {showFooter && sorted.length > 0 && (
        <div className="mt-1.5 text-[10px] font-mono text-slate-600 text-right">
          {sorted.length} row{sorted.length !== 1 ? 's' : ''}
          {data.length !== sorted.length && ` (of ${data.length})`}
        </div>
      )}
    </div>
  );
}

export default SortableTable;
