import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from '../components/LoadingSpinner'

// ─── Types ────────────────────────────────────────────────────────────────────

type Impact = 'HIGH' | 'MEDIUM' | 'LOW'
type EventCategory = 'TOKEN_UNLOCK' | 'PROTOCOL_UPGRADE' | 'MACRO' | 'LISTING' | 'GOVERNANCE'

interface CryptoEvent {
  id: string
  name: string
  symbol?: string
  date: string           // ISO string
  impact: Impact
  category: EventCategory
  description: string
}

interface PastEvent {
  id: string
  name: string
  symbol?: string
  date: string
  impact: Impact
  pricePct24h: number    // price change 24h after event
  pricePct7d: number
}

interface TokenUnlock {
  symbol: string
  amount: number
  valueUsd: number
  unlocksAt: string
}

interface CalendarData {
  events: CryptoEvent[]
  pastEvents: PastEvent[]
  nextUnlock: TokenUnlock
}

// ─── Mock ─────────────────────────────────────────────────────────────────────

const today = new Date()

function daysFromNow(d: number): string {
  const dt = new Date(today)
  dt.setDate(dt.getDate() + d)
  return dt.toISOString()
}

const MOCK: CalendarData = {
  nextUnlock: {
    symbol: 'ARB',
    amount: 1_100_000_000,
    valueUsd: 1_250_000_000,
    unlocksAt: daysFromNow(2),
  },
  events: [
    { id: 'e1',  name: 'FOMC Meeting',           date: daysFromNow(3),  impact: 'HIGH',   category: 'MACRO',           symbol: undefined, description: 'Federal Reserve rate decision — elevated crypto volatility expected' },
    { id: 'e2',  name: 'ARB Token Unlock',        date: daysFromNow(2),  impact: 'HIGH',   category: 'TOKEN_UNLOCK',    symbol: 'ARB',     description: '1.1B ARB unlocks to team/investors — potential sell pressure' },
    { id: 'e3',  name: 'ETH Pectra Upgrade',      date: daysFromNow(5),  impact: 'HIGH',   category: 'PROTOCOL_UPGRADE', symbol: 'ETH',    description: 'Major Ethereum upgrade including EIP-7251' },
    { id: 'e4',  name: 'CPI Data Release',        date: daysFromNow(1),  impact: 'HIGH',   category: 'MACRO',           symbol: undefined, description: 'US Consumer Price Index — market-moving macro data' },
    { id: 'e5',  name: 'SOL Breakpoint Conference',date: daysFromNow(8), impact: 'MEDIUM', category: 'PROTOCOL_UPGRADE', symbol: 'SOL',   description: 'Annual Solana developer conference — announcements expected' },
    { id: 'e6',  name: 'AVAX Token Unlock',       date: daysFromNow(10), impact: 'MEDIUM', category: 'TOKEN_UNLOCK',    symbol: 'AVAX',    description: '50M AVAX vesting cliff' },
    { id: 'e7',  name: 'NFP Report',              date: daysFromNow(4),  impact: 'MEDIUM', category: 'MACRO',           symbol: undefined, description: 'Non-Farm Payrolls — labor market data' },
    { id: 'e8',  name: 'BNB Governance Vote',     date: daysFromNow(6),  impact: 'LOW',    category: 'GOVERNANCE',      symbol: 'BNB',     description: 'BEP-48 fee structure proposal vote' },
    { id: 'e9',  name: 'LINK Mainnet Update',     date: daysFromNow(12), impact: 'LOW',    category: 'PROTOCOL_UPGRADE', symbol: 'LINK',  description: 'Chainlink CCIP v1.5 launch' },
    { id: 'e10', name: 'Binance Listing: New Token', date: daysFromNow(0), impact: 'MEDIUM', category: 'LISTING',       symbol: 'NEW',     description: 'Major exchange listing typically causes 20–50% initial spike' },
  ],
  pastEvents: [
    { id: 'p1', name: 'OP Token Unlock', symbol: 'OP', date: new Date(today.getTime() - 5 * 86_400_000).toISOString(), impact: 'HIGH',   pricePct24h: -8.2,  pricePct7d: -5.1 },
    { id: 'p2', name: 'Fed Minutes',                   date: new Date(today.getTime() - 8 * 86_400_000).toISOString(), impact: 'HIGH',   pricePct24h:  4.1,  pricePct7d:  7.3 },
    { id: 'p3', name: 'BTC Halving',     symbol: 'BTC', date: new Date(today.getTime() - 12* 86_400_000).toISOString(), impact: 'HIGH',   pricePct24h:  2.3,  pricePct7d: 14.2 },
    { id: 'p4', name: 'SOL v1.18 Launch', symbol: 'SOL', date: new Date(today.getTime() - 16* 86_400_000).toISOString(), impact: 'MEDIUM', pricePct24h:  6.8,  pricePct7d:  9.4 },
    { id: 'p5', name: 'CPI Release',                   date: new Date(today.getTime() - 20* 86_400_000).toISOString(), impact: 'HIGH',   pricePct24h: -3.4,  pricePct7d: -1.2 },
  ],
}

async function fetchCalendarData(): Promise<CalendarData> {
  try {
    const res = await fetch('/api/events/calendar')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const IMPACT_COLORS: Record<Impact, string> = {
  HIGH:   'var(--red)',
  MEDIUM: 'var(--yellow)',
  LOW:    'var(--green)',
}

const CATEGORY_ICONS: Record<EventCategory, string> = {
  TOKEN_UNLOCK:    '🔓',
  PROTOCOL_UPGRADE: '⬆',
  MACRO:           '📊',
  LISTING:         '⊕',
  GOVERNANCE:      '⚖',
}

function formatCountdown(target: string): string {
  const diff = new Date(target).getTime() - Date.now()
  if (diff <= 0) return 'Now'
  const h = Math.floor(diff / 3_600_000)
  const m = Math.floor((diff % 3_600_000) / 60_000)
  if (h >= 24) return `${Math.floor(h / 24)}d ${h % 24}h`
  return `${h}h ${m}m`
}

// ─── Calendar Grid ────────────────────────────────────────────────────────────

interface CalendarGridProps {
  events: CryptoEvent[]
  year: number
  month: number
}

const CalendarGrid: React.FC<CalendarGridProps> = ({ events, year, month }) => {
  const firstDay = new Date(year, month, 1).getDay()
  const daysInMonth = new Date(year, month + 1, 0).getDate()
  const cells: (number | null)[] = [
    ...Array(firstDay).fill(null),
    ...Array.from({ length: daysInMonth }, (_, i) => i + 1),
  ]
  // pad to complete grid
  while (cells.length % 7 !== 0) cells.push(null)

  function eventsOnDay(day: number): CryptoEvent[] {
    return events.filter(e => {
      const d = new Date(e.date)
      return d.getFullYear() === year && d.getMonth() === month && d.getDate() === day
    })
  }

  const todayDate = today.getDate()
  const isCurrentMonth = today.getFullYear() === year && today.getMonth() === month

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 1, marginBottom: 1 }}>
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(d => (
          <div key={d} style={{ padding: '4px', textAlign: 'center', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            {d}
          </div>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 1 }}>
        {cells.map((day, i) => {
          const evs = day ? eventsOnDay(day) : []
          const isToday = isCurrentMonth && day === todayDate
          return (
            <div
              key={i}
              style={{
                minHeight: 52,
                padding: '4px 5px',
                background: isToday ? 'rgba(59,130,246,0.08)' : 'var(--bg-hover)',
                border: isToday ? '1px solid var(--accent)' : '1px solid var(--border)',
                borderRadius: 4,
                opacity: day ? 1 : 0.2,
              }}
            >
              {day && (
                <>
                  <div style={{
                    fontSize: '0.7rem', fontWeight: isToday ? 700 : 400,
                    color: isToday ? 'var(--accent)' : 'var(--text-muted)',
                    marginBottom: 2,
                  }}>
                    {day}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    {evs.slice(0, 3).map(e => (
                      <div
                        key={e.id}
                        title={`${e.name}: ${e.description}`}
                        style={{
                          width: 8, height: 8, borderRadius: '50%',
                          background: IMPACT_COLORS[e.impact],
                          display: 'inline-block',
                        }}
                      />
                    ))}
                  </div>
                </>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const EventCalendar: React.FC = () => {
  const [calMonth, setCalMonth] = useState({ year: today.getFullYear(), month: today.getMonth() })

  const { data, isLoading } = useQuery<CalendarData>({
    queryKey: ['event-calendar'],
    queryFn: fetchCalendarData,
    refetchInterval: 60_000,
  })

  if (isLoading || !data) return <LoadingSpinner message="Loading event calendar…" />

  const next7Days = data.events
    .filter(e => {
      const diff = new Date(e.date).getTime() - Date.now()
      return diff >= 0 && diff <= 7 * 86_400_000
    })
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())

  const macroEvents = data.events.filter(e => e.category === 'MACRO')
  const unlockEvents = data.events.filter(e => e.category === 'TOKEN_UNLOCK')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Token unlock countdown */}
      <div style={{
        padding: '14px 20px', borderRadius: 8,
        background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.3)',
        display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
      }}>
        <span style={{ fontSize: '1.2rem' }}>🔓</span>
        <div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>NEXT MAJOR TOKEN UNLOCK</div>
          <div style={{ fontSize: '1rem', fontWeight: 800, color: 'var(--text-primary)' }}>
            {data.nextUnlock.symbol} — ${(data.nextUnlock.valueUsd / 1e9).toFixed(2)}B
          </div>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>COUNTDOWN</div>
          <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--red)', fontVariantNumeric: 'tabular-nums' }}>
            {formatCountdown(data.nextUnlock.unlocksAt)}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: 16 }}>
        {/* Calendar */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '16px',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {new Date(calMonth.year, calMonth.month).toLocaleString('en-US', { month: 'long', year: 'numeric' })}
            </span>
            <div style={{ display: 'flex', gap: 4 }}>
              <button
                onClick={() => setCalMonth(p => {
                  const m = p.month === 0 ? 11 : p.month - 1
                  const y = p.month === 0 ? p.year - 1 : p.year
                  return { year: y, month: m }
                })}
                style={{ padding: '3px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.8rem' }}
              >
                ‹
              </button>
              <button
                onClick={() => setCalMonth(p => {
                  const m = p.month === 11 ? 0 : p.month + 1
                  const y = p.month === 11 ? p.year + 1 : p.year
                  return { year: y, month: m }
                })}
                style={{ padding: '3px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '0.8rem' }}
              >
                ›
              </button>
            </div>
          </div>
          <CalendarGrid events={data.events} year={calMonth.year} month={calMonth.month} />
          <div style={{ display: 'flex', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
            {(['HIGH', 'MEDIUM', 'LOW'] as Impact[]).map(imp => (
              <div key={imp} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: IMPACT_COLORS[imp] }} />
                {imp}
              </div>
            ))}
          </div>
        </div>

        {/* Next 7 days */}
        <div style={{
          background: 'var(--bg-surface)', border: '1px solid var(--border)',
          borderRadius: 8, overflow: 'hidden',
        }}>
          <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
            NEXT 7 DAYS
          </div>
          <div style={{ overflowY: 'auto', maxHeight: 380 }}>
            {next7Days.map((e, i) => {
              const isMacro = e.category === 'MACRO'
              return (
                <div key={e.id} style={{
                  padding: '10px 16px',
                  borderBottom: i < next7Days.length - 1 ? '1px solid var(--border)' : undefined,
                  background: isMacro && e.impact === 'HIGH' ? 'rgba(239,68,68,0.04)' : undefined,
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ fontSize: '0.9rem' }}>{CATEGORY_ICONS[e.category]}</span>
                        <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)' }}>{e.name}</span>
                        {e.symbol && (
                          <span style={{ fontSize: '0.68rem', color: 'var(--accent)', fontWeight: 700 }}>{e.symbol}</span>
                        )}
                      </div>
                      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 3 }}>{e.description}</div>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 3, flexShrink: 0 }}>
                      <span style={{
                        fontSize: '0.65rem', fontWeight: 700, padding: '2px 7px', borderRadius: 4,
                        background: `${IMPACT_COLORS[e.impact]}18`, color: IMPACT_COLORS[e.impact],
                      }}>
                        {e.impact}
                      </span>
                      <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                        {formatCountdown(e.date)}
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
            {next7Days.length === 0 && (
              <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                No events in next 7 days
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Historical Event Outcomes */}
      <div style={{
        background: 'var(--bg-surface)', border: '1px solid var(--border)',
        borderRadius: 8, overflow: 'hidden',
      }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
          HISTORICAL EVENT OUTCOMES
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Event', 'Symbol', 'Date', 'Impact', '+24h Price', '+7d Price'].map(h => (
                <th key={h} style={{ padding: '6px 14px', textAlign: 'left', fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.pastEvents.map((e, i) => (
              <tr key={e.id} style={{ borderBottom: i < data.pastEvents.length - 1 ? '1px solid var(--border)' : undefined }}>
                <td style={{ padding: '7px 14px', fontSize: '0.8rem', color: 'var(--text-primary)', fontWeight: 600 }}>{e.name}</td>
                <td style={{ padding: '7px 14px', fontSize: '0.75rem', color: 'var(--accent)' }}>{e.symbol ?? '—'}</td>
                <td style={{ padding: '7px 14px', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  {new Date(e.date).toLocaleDateString()}
                </td>
                <td style={{ padding: '7px 14px' }}>
                  <span style={{
                    fontSize: '0.65rem', fontWeight: 700, padding: '2px 6px', borderRadius: 4,
                    background: `${IMPACT_COLORS[e.impact]}18`, color: IMPACT_COLORS[e.impact],
                  }}>
                    {e.impact}
                  </span>
                </td>
                <td style={{
                  padding: '7px 14px', fontSize: '0.8rem', fontWeight: 700, fontVariantNumeric: 'tabular-nums',
                  color: e.pricePct24h >= 0 ? 'var(--green)' : 'var(--red)',
                }}>
                  {e.pricePct24h >= 0 ? '+' : ''}{e.pricePct24h.toFixed(1)}%
                </td>
                <td style={{
                  padding: '7px 14px', fontSize: '0.8rem', fontWeight: 700, fontVariantNumeric: 'tabular-nums',
                  color: e.pricePct7d >= 0 ? 'var(--green)' : 'var(--red)',
                }}>
                  {e.pricePct7d >= 0 ? '+' : ''}{e.pricePct7d.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default EventCalendar
