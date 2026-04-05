import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import LoadingSpinner from '../components/LoadingSpinner'
import MetricCard from '../components/MetricCard'

// ─── Types ────────────────────────────────────────────────────────────────────

interface SymbolSentiment {
  symbol: string
  score: number       // -1 to +1
  change24h: number   // delta vs 24h ago
  volume: number      // mention volume (normalised 0–100)
  sparkline: number[] // last 12 data points of mention volume
  fundingRate: number // annualised % funding rate
}

interface NewsItem {
  id: string
  headline: string
  source: string
  timestamp: string
  sentiment: number  // -1 to +1
  symbols: string[]
}

interface SentimentData {
  fearGreedIndex: number     // 0–100
  fearGreedLabel: string
  fearGreedChange: number    // vs yesterday
  symbols: SymbolSentiment[]
  recentNews: NewsItem[]
  updatedAt: string
}

// ─── API ──────────────────────────────────────────────────────────────────────

async function fetchSentiment(): Promise<SentimentData> {
  try {
    const res = await fetch('/api/sentiment/feed')
    if (!res.ok) throw new Error('not ok')
    return res.json()
  } catch {
    return MOCK_SENTIMENT
  }
}

function rndSparkline(base: number): number[] {
  const s: number[] = []
  let v = base
  for (let i = 0; i < 12; i++) {
    v = Math.max(0, Math.min(100, v + (Math.random() - 0.5) * 20))
    s.push(v)
  }
  return s
}

const MOCK_SENTIMENT: SentimentData = {
  fearGreedIndex: 72,
  fearGreedLabel: 'Greed',
  fearGreedChange: +5,
  updatedAt: new Date().toISOString(),
  symbols: [
    { symbol: 'BTC',  score: 0.62,  change24h: +0.08, volume: 88, sparkline: rndSparkline(80), fundingRate: 18.2 },
    { symbol: 'ETH',  score: 0.44,  change24h: +0.02, volume: 65, sparkline: rndSparkline(60), fundingRate: 12.4 },
    { symbol: 'SOL',  score: 0.71,  change24h: +0.15, volume: 72, sparkline: rndSparkline(65), fundingRate: 22.1 },
    { symbol: 'BNB',  score: 0.18,  change24h: -0.05, volume: 40, sparkline: rndSparkline(38), fundingRate: 8.7  },
    { symbol: 'DOGE', score: -0.12, change24h: -0.18, volume: 55, sparkline: rndSparkline(52), fundingRate: -2.4 },
    { symbol: 'XRP',  score: 0.35,  change24h: +0.04, volume: 48, sparkline: rndSparkline(45), fundingRate: 6.1  },
    { symbol: 'ADA',  score: -0.08, change24h: -0.09, volume: 30, sparkline: rndSparkline(28), fundingRate: 1.2  },
    { symbol: 'AVAX', score: 0.52,  change24h: +0.10, volume: 44, sparkline: rndSparkline(40), fundingRate: 14.8 },
  ],
  recentNews: [
    { id: 'n1', headline: 'BlackRock BTC ETF sees record $800M inflow in single session', source: 'Bloomberg', timestamp: new Date(Date.now() - 1800000).toISOString(), sentiment: 0.85, symbols: ['BTC'] },
    { id: 'n2', headline: 'Ethereum Pectra upgrade testnet goes live; validators celebrate low gas', source: 'CoinDesk', timestamp: new Date(Date.now() - 3600000).toISOString(), sentiment: 0.70, symbols: ['ETH'] },
    { id: 'n3', headline: 'US CPI comes in hotter than expected; rate cut odds drop sharply', source: 'Reuters', timestamp: new Date(Date.now() - 5400000).toISOString(), sentiment: -0.55, symbols: ['BTC', 'ETH', 'SOL'] },
    { id: 'n4', headline: 'Solana DEX volume surpasses Ethereum for 3rd consecutive day', source: 'The Block', timestamp: new Date(Date.now() - 7200000).toISOString(), sentiment: 0.65, symbols: ['SOL'] },
    { id: 'n5', headline: 'DOGE whale moves $240M to unknown wallet; community speculates dump', source: 'CryptoSlate', timestamp: new Date(Date.now() - 9000000).toISOString(), sentiment: -0.45, symbols: ['DOGE'] },
  ],
}

// ─── Components ───────────────────────────────────────────────────────────────

const SENTIMENT_COLOR = (score: number) => {
  if (score > 0.4) return 'var(--green)'
  if (score > 0.1) return '#86efac'
  if (score > -0.1) return 'var(--text-muted)'
  if (score > -0.4) return '#fca5a5'
  return 'var(--red)'
}

interface BullBearGaugeProps {
  score: number  // -1 to +1
  size?: 'sm' | 'lg'
}

const BullBearGauge: React.FC<BullBearGaugeProps> = ({ score, size = 'sm' }) => {
  const pct    = ((score + 1) / 2) * 100
  const color  = SENTIMENT_COLOR(score)
  const h      = size === 'lg' ? 10 : 6
  const radius = size === 'lg' ? 4 : 2

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <div style={{
        width: size === 'lg' ? 120 : 80,
        height: h, borderRadius: radius,
        background: 'var(--bg-hover)',
        position: 'relative', overflow: 'hidden',
      }}>
        {/* Center line */}
        <div style={{ position: 'absolute', left: '50%', top: 0, width: 1, height: '100%', background: 'var(--border)' }} />
        <div style={{
          position: 'absolute',
          left:  score >= 0 ? '50%' : `${pct}%`,
          width: `${Math.abs(score) * 50}%`,
          height: '100%',
          background: color,
          transition: 'all 0.4s',
        }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.6rem', color: 'var(--text-muted)', width: size === 'lg' ? 120 : 80 }}>
        <span style={{ color: 'var(--red)' }}>B</span>
        <span style={{ color, fontWeight: 700 }}>{score >= 0 ? '+' : ''}{score.toFixed(2)}</span>
        <span style={{ color: 'var(--green)' }}>B</span>
      </div>
    </div>
  )
}

interface SparklineProps {
  data: number[]
  color?: string
  width?: number
  height?: number
}

const Sparkline: React.FC<SparklineProps> = ({ data, color = 'var(--accent)', width = 60, height = 24 }) => {
  if (data.length < 2) return null
  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1
  const step  = width / (data.length - 1)

  const points = data.map((v, i) => ({
    x: i * step,
    y: height - ((v - min) / range) * height,
  }))

  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')

  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      <polyline
        points={points.map(p => `${p.x},${p.y}`).join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}

interface FundingHeatMapProps {
  symbols: SymbolSentiment[]
}

const FundingHeatMap: React.FC<FundingHeatMapProps> = ({ symbols }) => {
  const max = Math.max(...symbols.map(s => Math.abs(s.fundingRate)))

  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '14px 16px',
    }}>
      <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Funding Rate Heat Map (annualised %)
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {symbols.map((s) => {
          const intensity = Math.abs(s.fundingRate) / max
          const isPos     = s.fundingRate >= 0
          const bg        = isPos
            ? `rgba(34,197,94,${0.1 + intensity * 0.5})`
            : `rgba(239,68,68,${0.1 + intensity * 0.5})`
          const textColor = isPos ? 'var(--green)' : 'var(--red)'

          return (
            <div key={s.symbol} style={{
              padding: '8px 12px',
              borderRadius: 6,
              background: bg,
              border: `1px solid ${isPos ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)'}`,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 2,
              minWidth: 60,
            }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {s.symbol}
              </span>
              <span style={{ fontSize: '0.8rem', fontWeight: 700, color: textColor, fontVariantNumeric: 'tabular-nums' }}>
                {s.fundingRate >= 0 ? '+' : ''}{s.fundingRate.toFixed(1)}%
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

const SentimentFeed: React.FC = () => {
  const { data, isLoading } = useQuery<SentimentData>({
    queryKey: ['sentiment', 'feed'],
    queryFn: fetchSentiment,
    refetchInterval: 20_000,
  })

  const [sortBy, setSortBy] = useState<'score' | 'volume' | 'symbol'>('score')

  if (isLoading || !data) return <LoadingSpinner message="Loading sentiment feed…" />

  const fgColor =
    data.fearGreedIndex > 75 ? 'var(--red)'
    : data.fearGreedIndex > 55 ? 'var(--yellow)'
    : data.fearGreedIndex > 35 ? 'var(--text-muted)'
    : 'var(--green)'

  const sortedSymbols = [...data.symbols].sort((a, b) => {
    if (sortBy === 'score')  return b.score - a.score
    if (sortBy === 'volume') return b.volume - a.volume
    return a.symbol.localeCompare(b.symbol)
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Fear & Greed */}
      <div style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        padding: '20px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: 28,
        flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Fear & Greed Index
          </span>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
            <span style={{ fontSize: '3rem', fontWeight: 800, color: fgColor, fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>
              {data.fearGreedIndex}
            </span>
            <span style={{ fontSize: '1.1rem', color: fgColor, fontWeight: 700 }}>
              {data.fearGreedLabel}
            </span>
            <span style={{ fontSize: '0.8rem', color: data.fearGreedChange > 0 ? 'var(--green)' : 'var(--red)' }}>
              {data.fearGreedChange > 0 ? '▲' : '▼'} {Math.abs(data.fearGreedChange)} vs yesterday
            </span>
          </div>
          {/* Bar */}
          <div style={{ width: 240, height: 8, borderRadius: 4, background: 'linear-gradient(90deg, var(--green) 0%, var(--yellow) 50%, var(--red) 100%)', position: 'relative' }}>
            <div style={{
              position: 'absolute',
              left: `${data.fearGreedIndex}%`,
              top: -3, width: 3, height: 14,
              background: '#fff',
              borderRadius: 2,
              transform: 'translateX(-50%)',
              boxShadow: '0 0 4px rgba(0,0,0,0.5)',
            }} />
          </div>
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          <MetricCard label="Bullish signals" value={`${data.symbols.filter(s => s.score > 0.2).length}`} color="var(--green)" />
          <MetricCard label="Bearish signals" value={`${data.symbols.filter(s => s.score < -0.2).length}`} color="var(--red)" />
          <MetricCard label="Neutral"         value={`${data.symbols.filter(s => Math.abs(s.score) <= 0.2).length}`} />
        </div>
      </div>

      {/* Per-symbol sentiment table */}
      <div style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        overflow: 'hidden',
      }}>
        <div style={{
          padding: '10px 16px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Symbol Sentiment
          </span>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
            {(['score', 'volume', 'symbol'] as const).map((s) => (
              <button
                key={s}
                className="btn-icon"
                onClick={() => setSortBy(s)}
                style={{
                  fontSize: '0.7rem', padding: '3px 10px', borderRadius: 4,
                  background: sortBy === s ? 'var(--accent)' : 'var(--bg-hover)',
                  color: sortBy === s ? '#000' : 'var(--text-muted)',
                  fontWeight: sortBy === s ? 700 : 400,
                  border: 'none',
                  textTransform: 'capitalize',
                }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Symbol', 'Sentiment', '24h Δ', 'Mention Vol', 'Trend', 'Funding'].map((h) => (
                <th key={h} style={{ padding: '8px 16px', textAlign: 'left', fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedSymbols.map((s, i) => (
              <tr key={s.symbol} style={{
                borderBottom: i < sortedSymbols.length - 1 ? '1px solid var(--border)' : undefined,
              }}>
                <td style={{ padding: '10px 16px', fontWeight: 700, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                  {s.symbol}
                </td>
                <td style={{ padding: '10px 16px' }}>
                  <BullBearGauge score={s.score} />
                </td>
                <td style={{ padding: '10px 16px', fontSize: '0.8rem', color: s.change24h >= 0 ? 'var(--green)' : 'var(--red)', fontVariantNumeric: 'tabular-nums' }}>
                  {s.change24h >= 0 ? '+' : ''}{s.change24h.toFixed(2)}
                </td>
                <td style={{ padding: '10px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ width: 60, height: 6, borderRadius: 3, background: 'var(--bg-hover)', overflow: 'hidden' }}>
                      <div style={{ width: `${s.volume}%`, height: '100%', background: 'var(--accent)', transition: 'width 0.4s' }} />
                    </div>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{s.volume}</span>
                  </div>
                </td>
                <td style={{ padding: '10px 16px' }}>
                  <Sparkline data={s.sparkline} color={SENTIMENT_COLOR(s.score)} />
                </td>
                <td style={{ padding: '10px 16px', fontSize: '0.8rem', fontVariantNumeric: 'tabular-nums', color: s.fundingRate >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                  {s.fundingRate >= 0 ? '+' : ''}{s.fundingRate.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Funding heat map */}
      <FundingHeatMap symbols={data.symbols} />

      {/* News feed */}
      <div style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        overflow: 'hidden',
      }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', fontSize: '0.78rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Recent News
        </div>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {data.recentNews.map((n, i) => {
            const sc = SENTIMENT_COLOR(n.sentiment)
            return (
              <div key={n.id} style={{
                padding: '12px 16px',
                borderBottom: i < data.recentNews.length - 1 ? '1px solid var(--border)' : undefined,
                display: 'flex',
                gap: 12,
                alignItems: 'flex-start',
              }}>
                <div style={{
                  width: 4, borderRadius: 2, alignSelf: 'stretch', flexShrink: 0,
                  background: sc,
                }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-primary)', lineHeight: 1.4, marginBottom: 4 }}>
                    {n.headline}
                  </div>
                  <div style={{ display: 'flex', gap: 10, alignItems: 'center', fontSize: '0.72rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>{n.source}</span>
                    <span style={{ color: 'var(--text-muted)' }}>
                      {new Date(n.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    {n.symbols.map((sym) => (
                      <span key={sym} style={{
                        padding: '1px 6px', borderRadius: 4,
                        background: 'var(--bg-hover)',
                        color: 'var(--text-secondary)',
                        fontSize: '0.68rem',
                      }}>
                        {sym}
                      </span>
                    ))}
                    <span style={{ marginLeft: 'auto', color: sc, fontWeight: 700 }}>
                      {n.sentiment >= 0 ? '+' : ''}{n.sentiment.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export default SentimentFeed
