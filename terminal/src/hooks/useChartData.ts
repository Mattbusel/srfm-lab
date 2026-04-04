// ============================================================
// useChartData — fetches and manages chart OHLCV data
// ============================================================
import { useEffect, useRef, useCallback, useState } from 'react'
import { useMarketStore } from '@/store/marketStore'
import { gatewayService } from '@/services/gateway'
import type { OHLCV, Interval, VolumeProfile } from '@/types'

interface UseChartDataOptions {
  symbol: string
  interval: Interval
  limit?: number
  enabled?: boolean
}

interface UseChartDataReturn {
  bars: OHLCV[]
  isLoading: boolean
  error: string | null
  refetch: () => void
  append: (bar: OHLCV) => void
}

export function useChartData({ symbol, interval, limit = 500, enabled = true }: UseChartDataOptions): UseChartDataReturn {
  const store = useMarketStore()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const bars = store.recentBars[symbol] ?? []

  const fetchBars = useCallback(async () => {
    if (!enabled || !symbol) return

    setIsLoading(true)
    setError(null)

    // Cancel previous request
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    try {
      const data = await gatewayService.getBars(symbol, interval, limit)
      store.setBars(symbol, data)
      setIsLoading(false)
    } catch (err) {
      if ((err as Error).name === 'AbortError') return

      // Fall back to mock data
      try {
        const mockBars = gatewayService.generateMockBars(symbol, interval, limit)
        store.setBars(symbol, mockBars)
        setIsLoading(false)
      } catch (mockErr) {
        setError(mockErr instanceof Error ? mockErr.message : 'Failed to load chart data')
        setIsLoading(false)
      }
    }
  }, [symbol, interval, limit, enabled])

  useEffect(() => {
    fetchBars()
    return () => { abortRef.current?.abort() }
  }, [symbol, interval, limit, enabled])

  const append = useCallback((bar: OHLCV) => {
    store.addBar(symbol, bar)
  }, [symbol])

  return { bars, isLoading, error, refetch: fetchBars, append }
}

interface UseVolumeProfileOptions {
  symbol: string
  startTime: number
  endTime: number
  enabled?: boolean
}

export function useVolumeProfile({ symbol, startTime, endTime, enabled = true }: UseVolumeProfileOptions) {
  const store = useMarketStore()
  const [isLoading, setIsLoading] = useState(false)

  const profile = store.volumeProfiles[symbol] ?? null

  useEffect(() => {
    if (!enabled || !symbol || !startTime || !endTime) return

    setIsLoading(true)
    import('@/services/api').then(({ spacetimeApi }) => {
      spacetimeApi.getVolumeProfile(symbol, startTime, endTime)
        .then((p) => {
          store.setVolumeProfile(symbol, p)
          setIsLoading(false)
        })
        .catch(() => {
          // Generate mock volume profile from bars
          const bars = store.recentBars[symbol] ?? []
          if (bars.length > 0) {
            const profile = computeVolumeProfile(bars)
            store.setVolumeProfile(symbol, { ...profile, symbol, startTime, endTime, valuePct: 0.7 })
          }
          setIsLoading(false)
        })
    })
  }, [symbol, startTime, endTime, enabled])

  return { profile, isLoading }
}

function computeVolumeProfile(bars: OHLCV[]): Omit<VolumeProfile, 'symbol' | 'startTime' | 'endTime' | 'valuePct'> {
  if (bars.length === 0) {
    return { levels: [], poc: 0, vah: 0, val: 0, totalVolume: 0 }
  }

  const minPrice = Math.min(...bars.map((b) => b.low))
  const maxPrice = Math.max(...bars.map((b) => b.high))
  const numLevels = 50
  const priceStep = (maxPrice - minPrice) / numLevels

  const volumeByLevel = new Array(numLevels).fill(0) as number[]

  for (const bar of bars) {
    const barRange = bar.high - bar.low
    if (barRange === 0) continue

    // Distribute volume proportionally across price range
    const startLevel = Math.floor((bar.low - minPrice) / priceStep)
    const endLevel = Math.min(Math.ceil((bar.high - minPrice) / priceStep), numLevels - 1)

    for (let i = startLevel; i <= endLevel; i++) {
      const levelPct = 1 / (endLevel - startLevel + 1)
      volumeByLevel[i] = (volumeByLevel[i] ?? 0) + bar.volume * levelPct
    }
  }

  const totalVolume = volumeByLevel.reduce((s, v) => s + v, 0)

  // Find POC
  const pocIdx = volumeByLevel.indexOf(Math.max(...volumeByLevel))
  const poc = minPrice + pocIdx * priceStep + priceStep / 2

  // Find Value Area (70% of volume around POC)
  let vaVolume = volumeByLevel[pocIdx] ?? 0
  let vaHigh = pocIdx
  let vaLow = pocIdx
  const targetVaVolume = totalVolume * 0.7

  while (vaVolume < targetVaVolume && (vaHigh < numLevels - 1 || vaLow > 0)) {
    const upVol = vaHigh < numLevels - 1 ? (volumeByLevel[vaHigh + 1] ?? 0) : 0
    const downVol = vaLow > 0 ? (volumeByLevel[vaLow - 1] ?? 0) : 0

    if (upVol >= downVol) {
      vaHigh++
      vaVolume += upVol
    } else {
      vaLow--
      vaVolume += downVol
    }
  }

  const vah = minPrice + vaHigh * priceStep + priceStep
  const val = minPrice + vaLow * priceStep

  const levels = volumeByLevel.map((vol, i) => ({
    price: minPrice + i * priceStep + priceStep / 2,
    buyVolume: vol * 0.5,
    sellVolume: vol * 0.5,
    totalVolume: vol,
    pct: totalVolume > 0 ? vol / totalVolume : 0,
    isPOC: i === pocIdx,
    isVAH: i === vaHigh,
    isVAL: i === vaLow,
    isValueArea: i >= vaLow && i <= vaHigh,
  }))

  return { levels, poc, vah, val, totalVolume }
}
