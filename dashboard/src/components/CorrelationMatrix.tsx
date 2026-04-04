// ============================================================
// CorrelationMatrix.tsx — Custom correlation heatmap
// ============================================================
import React, { useMemo } from 'react'
import type { CorrelationEntry } from '@/types'

interface CorrelationMatrixProps {
  data: CorrelationEntry[]
  symbols?: string[]
  cellSize?: number
  className?: string
}

function colorForCorrelation(c: number): string {
  // -1 = red, 0 = dark, +1 = blue
  const r = c < 0 ? Math.round(239 * Math.abs(c)) : 30
  const g = 20
  const b = c > 0 ? Math.round(96 + 159 * c) : 30
  const a = 0.15 + Math.abs(c) * 0.85
  return `rgba(${r},${g},${b},${a})`
}

function textColorForCorrelation(c: number): string {
  if (Math.abs(c) > 0.6) return c > 0 ? '#93c5fd' : '#fca5a5'
  if (Math.abs(c) > 0.3) return c > 0 ? '#60a5fa' : '#f87171'
  return '#475569'
}

export const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({
  data,
  symbols: propSymbols,
  cellSize = 44,
  className,
}) => {
  const { symbols, matrix } = useMemo(() => {
    const symSet = new Set<string>()
    for (const e of data) {
      symSet.add(e.symbolA)
      symSet.add(e.symbolB)
    }
    const syms = propSymbols ?? Array.from(symSet).sort()

    const map = new Map<string, number>()
    for (const e of data) {
      map.set(`${e.symbolA}|${e.symbolB}`, e.correlation)
      map.set(`${e.symbolB}|${e.symbolA}`, e.correlation)
    }

    const mat: number[][] = syms.map((a) =>
      syms.map((b) => (a === b ? 1 : (map.get(`${a}|${b}`) ?? 0))),
    )

    return { symbols: syms, matrix: mat }
  }, [data, propSymbols])

  const labelWidth = 52

  if (!symbols.length) {
    return (
      <div className="text-slate-600 text-xs font-mono text-center py-8">No correlation data</div>
    )
  }

  return (
    <div className={className} style={{ overflowX: 'auto' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `${labelWidth}px repeat(${symbols.length}, ${cellSize}px)`,
          gridTemplateRows: `${cellSize * 0.5}px repeat(${symbols.length}, ${cellSize}px)`,
          width: 'fit-content',
        }}
      >
        {/* Top-left empty cell */}
        <div />
        {/* Column headers */}
        {symbols.map((sym) => (
          <div
            key={`col-${sym}`}
            style={{
              width: cellSize,
              display: 'flex',
              alignItems: 'flex-end',
              justifyContent: 'center',
              paddingBottom: 4,
            }}
          >
            <span
              style={{
                fontSize: 9,
                fontFamily: 'JetBrains Mono',
                color: '#475569',
                writingMode: 'vertical-rl',
                textOrientation: 'mixed',
                transform: 'rotate(180deg)',
                lineHeight: 1,
              }}
            >
              {sym.replace('USDT', '')}
            </span>
          </div>
        ))}

        {/* Rows */}
        {symbols.map((rowSym, r) => (
          <React.Fragment key={rowSym}>
            {/* Row label */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                paddingRight: 6,
                justifyContent: 'flex-end',
              }}
            >
              <span style={{ fontSize: 9, fontFamily: 'JetBrains Mono', color: '#475569' }}>
                {rowSym.replace('USDT', '')}
              </span>
            </div>
            {/* Cells */}
            {symbols.map((colSym, c) => {
              const val = matrix[r][c]
              return (
                <div
                  key={`${rowSym}-${colSym}`}
                  title={`${rowSym} / ${colSym}: ${val.toFixed(2)}`}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: colorForCorrelation(val),
                    border: '1px solid rgba(30,33,48,0.4)',
                  }}
                >
                  <span
                    style={{
                      fontSize: 9,
                      fontFamily: 'JetBrains Mono',
                      fontWeight: 600,
                      color: textColorForCorrelation(val),
                    }}
                  >
                    {val.toFixed(2)}
                  </span>
                </div>
              )
            })}
          </React.Fragment>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2 mt-3">
        <span className="text-[9px] font-mono text-slate-600">-1.0</span>
        <div
          className="h-2 flex-1 rounded"
          style={{
            background: 'linear-gradient(to right, rgba(239,30,30,0.9), rgba(30,20,30,0.5), rgba(30,30,255,0.9))',
          }}
        />
        <span className="text-[9px] font-mono text-slate-600">+1.0</span>
      </div>
    </div>
  )
}
