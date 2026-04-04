// ============================================================
// VolumeProfile — horizontal volume histogram
// ============================================================
import React, { useMemo } from 'react'
import { useVolumeProfile } from '@/hooks/useChartData'
import { useMarketStore } from '@/store/marketStore'
import type { VolumeProfileLevel } from '@/types'

interface VolumeProfileProps {
  symbol: string
  width?: number
  height?: number
  showValueArea?: boolean
  showPOC?: boolean
  className?: string
  style?: React.CSSProperties
}

const formatVolume = (v: number): string => {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`
  return v.toFixed(0)
}

export const VolumeProfile: React.FC<VolumeProfileProps> = ({
  symbol,
  width = 80,
  height = 400,
  showValueArea = true,
  showPOC = true,
  className = '',
  style,
}) => {
  const bars = useMarketStore((s) => s.recentBars[symbol] ?? [])
  const startTime = bars[0]?.time ?? 0
  const endTime = bars.at(-1)?.time ?? 0

  const { profile, isLoading } = useVolumeProfile({
    symbol,
    startTime,
    endTime,
    enabled: bars.length > 10,
  })

  const maxVolume = useMemo(
    () => Math.max(...(profile?.levels.map((l) => l.totalVolume) ?? [1]), 1),
    [profile?.levels]
  )

  if (isLoading || !profile) {
    return (
      <div
        className={`bg-terminal-bg/50 flex items-center justify-center ${className}`}
        style={{ width, height, ...style }}
      >
        <div className="w-3 h-3 border border-terminal-accent border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const levelHeight = height / profile.levels.length

  return (
    <div
      className={`relative bg-terminal-bg/50 ${className}`}
      style={{ width, height, ...style }}
      title="Volume Profile"
    >
      {/* Stats overlay */}
      <div className="absolute top-1 left-1 z-10 text-[9px] font-mono text-terminal-subtle space-y-0.5">
        <div>
          <span className="text-terminal-warning">POC</span>{' '}
          <span className="text-terminal-text">{profile.poc.toFixed(2)}</span>
        </div>
        <div>
          <span className="text-terminal-bull">VAH</span>{' '}
          <span className="text-terminal-text">{profile.vah.toFixed(2)}</span>
        </div>
        <div>
          <span className="text-terminal-bear">VAL</span>{' '}
          <span className="text-terminal-text">{profile.val.toFixed(2)}</span>
        </div>
      </div>

      {/* Bars */}
      {profile.levels.map((level, i) => {
        const barWidth = (level.totalVolume / maxVolume) * (width - 4)
        const y = height - (i + 1) * levelHeight

        let color = 'rgba(107, 114, 128, 0.35)'
        if (level.isPOC) color = 'rgba(245, 158, 11, 0.8)'
        else if (level.isValueArea) color = 'rgba(59, 130, 246, 0.35)'

        return (
          <div
            key={i}
            className="absolute right-0 transition-all duration-200"
            style={{
              bottom: i * levelHeight,
              height: Math.max(levelHeight - 0.5, 1),
              width: barWidth,
              backgroundColor: color,
              borderRight: level.isPOC ? '2px solid #f59e0b' : level.isVAH ? '1px solid #22c55e' : level.isVAL ? '1px solid #ef4444' : 'none',
            }}
            title={`Price: ${level.price.toFixed(2)}\nVolume: ${formatVolume(level.totalVolume)}\nPct: ${(level.pct * 100).toFixed(1)}%`}
          />
        )
      })}

      {/* POC line */}
      {showPOC && profile.poc && (
        <div
          className="absolute right-0 left-0 border-t border-terminal-warning/80 z-10"
          style={{
            bottom: ((profile.poc - (profile.levels[0]?.price ?? 0)) / ((profile.levels.at(-1)?.price ?? 1) - (profile.levels[0]?.price ?? 0))) * height,
          }}
        />
      )}

      {/* VAH line */}
      {showValueArea && profile.vah && (
        <div
          className="absolute right-0 left-0 border-t border-terminal-bull/50 z-10 border-dashed"
          style={{
            bottom: ((profile.vah - (profile.levels[0]?.price ?? 0)) / ((profile.levels.at(-1)?.price ?? 1) - (profile.levels[0]?.price ?? 0))) * height,
          }}
        />
      )}

      {/* VAL line */}
      {showValueArea && profile.val && (
        <div
          className="absolute right-0 left-0 border-t border-terminal-bear/50 z-10 border-dashed"
          style={{
            bottom: ((profile.val - (profile.levels[0]?.price ?? 0)) / ((profile.levels.at(-1)?.price ?? 1) - (profile.levels[0]?.price ?? 0))) * height,
          }}
        />
      )}
    </div>
  )
}

export default VolumeProfile
