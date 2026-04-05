import React, { useState, useMemo, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import StatusBadge from '../components/StatusBadge'
import RadarChartComponent from '../components/RadarChart'
import LoadingSpinner from '../components/LoadingSpinner'
import { fetchGenomes, triggerEvolution } from '../api/client'
import type { Genome, Island } from '../types'

type SortKey = 'fitness' | 'sharpe' | 'calmar' | 'maxDD' | 'generation'
type TabValue = Island | 'HOF'

const TABS: { value: TabValue; label: string }[] = [
  { value: 'BULL', label: 'Bull' },
  { value: 'BEAR', label: 'Bear' },
  { value: 'NEUTRAL', label: 'Neutral' },
  { value: 'HOF', label: 'Hall of Fame' },
]

// ─── Genome Card ──────────────────────────────────────────────────────────────

interface GenomeCardProps {
  genome: Genome
  onClick: (g: Genome) => void
  selected: boolean
}

const GenomeCard: React.FC<GenomeCardProps> = React.memo(
  ({ genome, onClick, selected }) => {
    return (
      <div
        className="card"
        onClick={() => onClick(genome)}
        style={{
          cursor: 'pointer',
          borderColor: selected ? 'var(--accent)' : undefined,
          boxShadow: selected ? 'var(--shadow-accent)' : undefined,
          transition: 'border-color var(--transition), box-shadow var(--transition)',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontWeight: 700,
              color: 'var(--accent)',
              fontSize: '0.9375rem',
            }}
          >
            #{genome.id}
          </span>
          <div style={{ display: 'flex', gap: 4 }}>
            <StatusBadge value={genome.island} size="sm" />
            {genome.isHallOfFame && <StatusBadge value="hof" size="sm" label="HOF" />}
          </div>
        </div>

        {/* Radar */}
        <RadarChartComponent params={genome.params} height={180} />

        {/* Metrics Grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 8,
          }}
        >
          {[
            { label: 'Sharpe', value: genome.sharpe.toFixed(3), color: genome.sharpe >= 2 ? 'var(--green)' : genome.sharpe >= 1 ? 'var(--accent)' : 'var(--text-secondary)' },
            { label: 'Fitness', value: genome.fitness.toFixed(4), color: 'var(--text-primary)' },
            { label: 'Max DD', value: `${(genome.maxDD * 100).toFixed(1)}%`, color: 'var(--red)' },
            { label: 'Calmar', value: genome.calmar.toFixed(2), color: 'var(--yellow)' },
          ].map(({ label, value, color }) => (
            <div
              key={label}
              style={{
                background: 'var(--bg-elevated)',
                borderRadius: 'var(--radius)',
                padding: '6px 8px',
              }}
            >
              <div style={{ fontSize: '0.6875rem', color: 'var(--text-muted)', marginBottom: 2 }}>
                {label}
              </div>
              <div
                className="num"
                style={{ fontSize: '0.875rem', fontWeight: 600, color }}
              >
                {value}
              </div>
            </div>
          ))}
        </div>

        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Gen {genome.generation}
          {genome.winRate !== undefined && (
            <span style={{ marginLeft: 8 }}>
              Win rate:{' '}
              <span className="num" style={{ color: 'var(--text-secondary)' }}>
                {(genome.winRate * 100).toFixed(0)}%
              </span>
            </span>
          )}
        </div>
      </div>
    )
  }
)

// ─── Genome Detail Modal ──────────────────────────────────────────────────────

interface GenomeDetailProps {
  genome: Genome
  onClose: () => void
}

const GenomeDetail: React.FC<GenomeDetailProps> = ({ genome, onClose }) => {
  // Synthetic fitness history from genome generation
  const fitnessHistory = useMemo(
    () =>
      Array.from({ length: genome.generation }, (_, i) => ({
        gen: i + 1,
        fitness: parseFloat(
          (genome.fitness * (0.5 + (0.5 * i) / genome.generation) +
            (Math.random() - 0.5) * 0.05).toFixed(4)
        ),
      })),
    [genome]
  )

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 800 }}
      >
        <div className="modal-header">
          <div>
            <h2 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span
                className="num"
                style={{ color: 'var(--accent)' }}
              >
                Genome #{genome.id}
              </span>
              <StatusBadge value={genome.island} />
              {genome.isHallOfFame && <StatusBadge value="hof" />}
            </h2>
            <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: 4 }}>
              Generation {genome.generation}
            </div>
          </div>
          <button className="btn-icon" onClick={onClose} style={{ fontSize: '1.2rem' }}>
            ×
          </button>
        </div>

        {/* Performance Metrics */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 10,
            marginBottom: 20,
          }}
        >
          {[
            { label: 'Fitness', value: genome.fitness.toFixed(4), color: 'var(--accent)' },
            { label: 'Sharpe', value: genome.sharpe.toFixed(3), color: genome.sharpe >= 2 ? 'var(--green)' : 'var(--text-primary)' },
            { label: 'Max DD', value: `${(genome.maxDD * 100).toFixed(1)}%`, color: 'var(--red)' },
            { label: 'Calmar', value: genome.calmar.toFixed(2), color: 'var(--yellow)' },
          ].map(({ label, value, color }) => (
            <div
              key={label}
              style={{
                background: 'var(--bg-elevated)',
                borderRadius: 'var(--radius)',
                padding: '10px 12px',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 4 }}>
                {label}
              </div>
              <div className="num" style={{ fontSize: '1.1rem', fontWeight: 700, color }}>
                {value}
              </div>
            </div>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
          {/* Radar Chart */}
          <div>
            <div className="card-title" style={{ marginBottom: 10 }}>
              Parameter Profile
            </div>
            <RadarChartComponent params={genome.params} height={200} />
          </div>

          {/* Fitness History */}
          <div>
            <div className="card-title" style={{ marginBottom: 10 }}>
              Fitness History
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={fitnessHistory} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
                <XAxis dataKey="gen" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                <YAxis domain={[0, 1]} tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    background: 'var(--bg-elevated)',
                    border: '1px solid var(--border-emphasis)',
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                />
                <Line type="monotone" dataKey="fitness" stroke="var(--accent)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Parameter Breakdown */}
        <div className="card-title" style={{ marginBottom: 10 }}>
          Parameters
        </div>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(5, 1fr)',
            gap: 8,
          }}
        >
          {Object.entries(genome.params).map(([key, val]) => (
            <div
              key={key}
              style={{
                background: 'var(--bg-elevated)',
                borderRadius: 'var(--radius)',
                padding: '6px 8px',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 2 }}>
                {key}
              </div>
              <div className="num" style={{ fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                {typeof val === 'number' ? val.toFixed(4) : String(val)}
              </div>
            </div>
          ))}
        </div>

        {/* Parent IDs */}
        {genome.parentIds && genome.parentIds.length > 0 && (
          <div style={{ marginTop: 16, fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
            Parents:{' '}
            {genome.parentIds.map((pid) => (
              <span
                key={pid}
                className="num"
                style={{ color: 'var(--accent)', marginRight: 6 }}
              >
                #{pid}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Trigger Evolution Button ─────────────────────────────────────────────────

interface TriggerBtnProps {
  island: Island
}

const TriggerEvolutionButton: React.FC<TriggerBtnProps> = ({ island }) => {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<string | null>(null)

  const handleTrigger = async () => {
    setLoading(true)
    setResult(null)
    try {
      const res = await triggerEvolution(island)
      setResult(res.queued ? 'Queued!' : 'Failed')
    } catch {
      setResult('Error')
    } finally {
      setLoading(false)
      setTimeout(() => setResult(null), 3000)
    }
  }

  return (
    <button
      className="btn btn-secondary btn-sm"
      onClick={handleTrigger}
      disabled={loading}
    >
      {loading ? <LoadingSpinner size={14} /> : '⟳'}
      {result ?? `Trigger ${island} Evolution`}
    </button>
  )
}

// ─── Genomes Page ─────────────────────────────────────────────────────────────

const GenomesPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabValue>('BULL')
  const [sortKey, setSortKey] = useState<SortKey>('sharpe')
  const [selectedGenome, setSelectedGenome] = useState<Genome | null>(null)

  const { data: genomes = [], isLoading, error } = useQuery({
    queryKey: ['genomes', 'all'],
    queryFn: () => fetchGenomes(),
    refetchInterval: 30_000,
  })

  const filtered = useMemo(() => {
    let result = activeTab === 'HOF'
      ? genomes.filter((g) => g.isHallOfFame)
      : genomes.filter((g) => g.island === activeTab)

    result = [...result].sort((a, b) => {
      switch (sortKey) {
        case 'fitness':   return b.fitness - a.fitness
        case 'sharpe':    return b.sharpe - a.sharpe
        case 'calmar':    return b.calmar - a.calmar
        case 'maxDD':     return a.maxDD - b.maxDD // less negative = better
        case 'generation': return b.generation - a.generation
        default:          return b.sharpe - a.sharpe
      }
    })
    return result
  }, [genomes, activeTab, sortKey])

  const handleCardClick = useCallback((g: Genome) => {
    setSelectedGenome((prev) => (prev?.id === g.id ? null : g))
  }, [])

  if (error) {
    return (
      <div className="empty-state">
        <div className="icon">⚠</div>
        <span>Failed to load genomes</span>
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <div className="page-title">Genome Browser</div>
          <div className="page-subtitle">
            {genomes.length} genomes · {genomes.filter((g) => g.isHallOfFame).length} Hall of Fame
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {/* Sort Select */}
          <select
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value as SortKey)}
            style={{ fontSize: '0.8125rem', padding: '4px 8px' }}
          >
            <option value="sharpe">Sort: Sharpe</option>
            <option value="fitness">Sort: Fitness</option>
            <option value="calmar">Sort: Calmar</option>
            <option value="maxDD">Sort: Max DD</option>
            <option value="generation">Sort: Generation</option>
          </select>
          {activeTab !== 'HOF' && (
            <TriggerEvolutionButton island={activeTab as Island} />
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="tabs">
        {TABS.map((tab) => {
          const count =
            tab.value === 'HOF'
              ? genomes.filter((g) => g.isHallOfFame).length
              : genomes.filter((g) => g.island === tab.value).length
          return (
            <button
              key={tab.value}
              className={`tab ${activeTab === tab.value ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.value)}
            >
              {tab.label}
              <span
                style={{
                  marginLeft: 6,
                  background: 'var(--bg-elevated)',
                  borderRadius: 10,
                  padding: '1px 6px',
                  fontSize: '0.7rem',
                  color: 'var(--text-muted)',
                }}
              >
                {count}
              </span>
            </button>
          )
        })}
      </div>

      {/* Grid */}
      {isLoading ? (
        <LoadingSpinner fullPage label="Loading genomes…" />
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <div className="icon">⬡</div>
          <span>No genomes in this category</span>
        </div>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
            gap: 16,
          }}
        >
          {filtered.map((g) => (
            <GenomeCard
              key={g.id}
              genome={g}
              onClick={handleCardClick}
              selected={selectedGenome?.id === g.id}
            />
          ))}
        </div>
      )}

      {/* Detail Modal */}
      {selectedGenome && (
        <GenomeDetail
          genome={selectedGenome}
          onClose={() => setSelectedGenome(null)}
        />
      )}
    </div>
  )
}

export default GenomesPage
