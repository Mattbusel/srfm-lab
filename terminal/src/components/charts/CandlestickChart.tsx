// ============================================================
// CandlestickChart — TradingView Lightweight Charts integration
// ============================================================
import React, { useEffect, useRef, useState, useCallback } from 'react'
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  ColorType,
  CrosshairMode,
  LineStyle,
  Time,
} from 'lightweight-charts'
import { useMarketStore } from '@/store/marketStore'
import { useBHStore } from '@/store/bhStore'
import { useSettingsStore } from '@/store/settingsStore'
import { useChartData } from '@/hooks/useChartData'
import type { OHLCV, Interval, BHRegime } from '@/types'

interface CandlestickChartProps {
  symbol: string
  interval: Interval
  height?: number
  showVolume?: boolean
  showEMA20?: boolean
  showEMA50?: boolean
  showEMA200?: boolean
  showBHOverlay?: boolean
  showRegimeColors?: boolean
  showVolumeProfile?: boolean
  onPriceHover?: (price: number | null) => void
  onIntervalChange?: (interval: Interval) => void
  className?: string
}

const INTERVALS: Interval[] = ['1m', '5m', '15m', '1h', '4h', '1d']
const INTERVAL_LABELS: Record<Interval, string> = {
  '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
  '1h': '1H', '2h': '2H', '4h': '4H', '1d': '1D', '1w': '1W',
}

// EMA calculation
function computeEMA(data: OHLCV[], period: number): { time: number; value: number }[] {
  if (data.length < period) return []
  const k = 2 / (period + 1)
  const result: { time: number; value: number }[] = []
  let ema = data.slice(0, period).reduce((s, d) => s + d.close, 0) / period

  for (let i = period - 1; i < data.length; i++) {
    if (i === period - 1) {
      ema = data.slice(0, period).reduce((s, d) => s + d.close, 0) / period
    } else {
      ema = data[i].close * k + ema * (1 - k)
    }
    result.push({ time: data[i].time as number, value: ema })
  }
  return result
}

const REGIME_COLORS: Record<BHRegime, string> = {
  BULL: 'rgba(34, 197, 94, 0.08)',
  BEAR: 'rgba(239, 68, 68, 0.08)',
  SIDEWAYS: 'rgba(107, 114, 128, 0.05)',
  HIGH_VOL: 'rgba(245, 158, 11, 0.08)',
}

export const CandlestickChart: React.FC<CandlestickChartProps> = ({
  symbol,
  interval,
  height = 500,
  showVolume: showVolumeProp,
  showEMA20: showEMA20Prop,
  showEMA50: showEMA50Prop,
  showEMA200: showEMA200Prop,
  showBHOverlay: showBHOverlayProp,
  showRegimeColors: showRegimeColorsProp,
  onPriceHover,
  onIntervalChange,
  className = '',
}) => {
  const settings = useSettingsStore((s) => s.settings)
  const showVolume = showVolumeProp ?? settings.showVolume
  const showEMA20 = showEMA20Prop ?? settings.showEMA20
  const showEMA50 = showEMA50Prop ?? settings.showEMA50
  const showEMA200 = showEMA200Prop ?? settings.showEMA200
  const showBHOverlay = showBHOverlayProp ?? settings.showBHOverlay
  const showRegimeColors = showRegimeColorsProp ?? settings.showRegimeColors

  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const ema20Ref = useRef<ISeriesApi<'Line'> | null>(null)
  const ema50Ref = useRef<ISeriesApi<'Line'> | null>(null)
  const ema200Ref = useRef<ISeriesApi<'Line'> | null>(null)
  const bhMassRef = useRef<ISeriesApi<'Line'> | null>(null)

  const [crosshairData, setCrosshairData] = useState<OHLCV | null>(null)
  const [selectedInterval, setSelectedInterval] = useState<Interval>(interval)

  const { bars, isLoading } = useChartData({ symbol, interval: selectedInterval })
  const realtimeBar = useMarketStore((s) => (s.recentBars[symbol] ?? []).at(-1))
  const bhInstrument = useBHStore((s) => s.instruments[symbol])

  // Initialize chart
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0a0e17' },
        textColor: '#9ca3af',
        fontSize: 11,
        fontFamily: 'JetBrains Mono, Fira Code, monospace',
      },
      grid: {
        vertLines: { color: '#1f2937', style: LineStyle.Dotted },
        horzLines: { color: '#1f2937', style: LineStyle.Dotted },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#4b5563', style: LineStyle.Dashed, labelBackgroundColor: '#1f2937' },
        horzLine: { color: '#4b5563', style: LineStyle.Dashed, labelBackgroundColor: '#1f2937' },
      },
      rightPriceScale: {
        borderColor: '#1f2937',
        textColor: '#9ca3af',
      },
      timeScale: {
        borderColor: '#1f2937',
        timeVisible: true,
        secondsVisible: selectedInterval === '1m' || selectedInterval === '5m',
      },
      handleScroll: { vertTouchDrag: false },
      handleScale: { axisPressedMouseMove: { time: true, price: false } },
    })

    chartRef.current = chart

    // Candlestick series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      priceLineVisible: true,
      priceLineWidth: 1,
      priceLineColor: '#4b5563',
      priceLineStyle: LineStyle.Dashed,
      lastValueVisible: true,
    })
    candleSeriesRef.current = candleSeries

    // Volume series (pane below)
    if (showVolume) {
      const volSeries = chart.addSeries(HistogramSeries, {
        color: '#4b5563',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })
      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
        borderVisible: false,
      })
      volumeSeriesRef.current = volSeries
    }

    // EMA overlays
    if (showEMA20) {
      ema20Ref.current = chart.addSeries(LineSeries, {
        color: '#3b82f6',
        lineWidth: 1,
        crosshairMarkerVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
        title: 'EMA20',
      })
    }

    if (showEMA50) {
      ema50Ref.current = chart.addSeries(LineSeries, {
        color: '#f59e0b',
        lineWidth: 1,
        crosshairMarkerVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
        title: 'EMA50',
      })
    }

    if (showEMA200) {
      ema200Ref.current = chart.addSeries(LineSeries, {
        color: '#ef4444',
        lineWidth: 1,
        crosshairMarkerVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
        title: 'EMA200',
      })
    }

    // BH Mass overlay (secondary axis)
    if (showBHOverlay) {
      bhMassRef.current = chart.addSeries(LineSeries, {
        color: '#7c3aed',
        lineWidth: 2,
        crosshairMarkerVisible: false,
        lastValueVisible: true,
        priceLineVisible: false,
        priceScaleId: 'bh_mass',
        title: 'BH Mass',
      })
      chart.priceScale('bh_mass').applyOptions({
        scaleMargins: { top: 0.7, bottom: 0.1 },
        borderVisible: false,
        textColor: '#7c3aed',
      })
    }

    // Crosshair move handler
    chart.subscribeCrosshairMove((param) => {
      if (param.time && candleSeries) {
        const data = param.seriesData.get(candleSeries) as {
          time: Time; open: number; high: number; low: number; close: number
        } | undefined

        if (data) {
          setCrosshairData({
            time: Number(data.time),
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            volume: 0,
          })
          onPriceHover?.(data.close)
        }
      } else {
        setCrosshairData(null)
        onPriceHover?.(null)
      }
    })

    // Resize observer
    const resizeObserver = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    })
    if (containerRef.current) resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
      chartRef.current = null
      candleSeriesRef.current = null
      volumeSeriesRef.current = null
      ema20Ref.current = null
      ema50Ref.current = null
      ema200Ref.current = null
      bhMassRef.current = null
    }
  }, [selectedInterval, showVolume, showEMA20, showEMA50, showEMA200, showBHOverlay]) // eslint-disable-line react-hooks/exhaustive-deps

  // Load bars into chart
  useEffect(() => {
    if (!candleSeriesRef.current || bars.length === 0) return

    const candleData = bars.map((b) => ({
      time: b.time as Time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }))

    candleSeriesRef.current.setData(candleData)

    // Volume
    if (volumeSeriesRef.current) {
      const volData = bars.map((b) => ({
        time: b.time as Time,
        value: b.volume,
        color: b.close >= b.open ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)',
      }))
      volumeSeriesRef.current.setData(volData)
    }

    // EMAs
    if (ema20Ref.current) {
      const data = computeEMA(bars, 20)
      ema20Ref.current.setData(data.map((d) => ({ time: d.time as Time, value: d.value })))
    }
    if (ema50Ref.current) {
      const data = computeEMA(bars, 50)
      ema50Ref.current.setData(data.map((d) => ({ time: d.time as Time, value: d.value })))
    }
    if (ema200Ref.current) {
      const data = computeEMA(bars, 200)
      ema200Ref.current.setData(data.map((d) => ({ time: d.time as Time, value: d.value })))
    }

    // BH Mass overlay (use mock for now)
    if (bhMassRef.current) {
      const massData = bars.map((b, i) => ({
        time: b.time as Time,
        value: 0.5 + Math.abs(Math.sin(i * 0.1)) * 2,
      }))
      bhMassRef.current.setData(massData)
    }

    chartRef.current?.timeScale().fitContent()
  }, [bars])

  // Realtime bar updates
  useEffect(() => {
    if (!realtimeBar || !candleSeriesRef.current) return
    candleSeriesRef.current.update({
      time: realtimeBar.time as Time,
      open: realtimeBar.open,
      high: realtimeBar.high,
      low: realtimeBar.low,
      close: realtimeBar.close,
    })
    if (volumeSeriesRef.current) {
      volumeSeriesRef.current.update({
        time: realtimeBar.time as Time,
        value: realtimeBar.volume,
        color: realtimeBar.close >= realtimeBar.open
          ? 'rgba(34, 197, 94, 0.4)'
          : 'rgba(239, 68, 68, 0.4)',
      })
    }
  }, [realtimeBar])

  const handleIntervalChange = useCallback((iv: Interval) => {
    setSelectedInterval(iv)
    onIntervalChange?.(iv)
  }, [onIntervalChange])

  const pctChange = bars.length >= 2
    ? ((bars.at(-1)!.close - bars.at(-2)!.close) / bars.at(-2)!.close * 100)
    : 0

  return (
    <div className={`flex flex-col h-full bg-terminal-bg ${className}`}>
      {/* ---- Toolbar ---- */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-terminal-text font-semibold font-mono text-sm">{symbol}</span>
          {bars.at(-1) && (
            <>
              <span className="font-mono text-sm text-terminal-text">
                {bars.at(-1)!.close.toFixed(2)}
              </span>
              <span className={`text-xs font-mono ${pctChange >= 0 ? 'text-terminal-bull' : 'text-terminal-bear'}`}>
                {pctChange >= 0 ? '+' : ''}{pctChange.toFixed(2)}%
              </span>
            </>
          )}
        </div>

        {/* Crosshair tooltip */}
        {crosshairData && (
          <div className="flex items-center gap-3 text-xs font-mono text-terminal-subtle">
            <span>O:<span className="text-terminal-text">{crosshairData.open.toFixed(2)}</span></span>
            <span>H:<span className="text-terminal-bull">{crosshairData.high.toFixed(2)}</span></span>
            <span>L:<span className="text-terminal-bear">{crosshairData.low.toFixed(2)}</span></span>
            <span>C:<span className={crosshairData.close >= crosshairData.open ? 'text-terminal-bull' : 'text-terminal-bear'}>
              {crosshairData.close.toFixed(2)}
            </span></span>
          </div>
        )}

        {/* Interval selector */}
        <div className="flex items-center gap-1">
          {INTERVALS.map((iv) => (
            <button
              key={iv}
              onClick={() => handleIntervalChange(iv)}
              className={`px-2 py-0.5 text-xs rounded font-mono transition-colors ${
                selectedInterval === iv
                  ? 'bg-terminal-accent text-white'
                  : 'text-terminal-subtle hover:text-terminal-text hover:bg-terminal-muted'
              }`}
            >
              {INTERVAL_LABELS[iv]}
            </button>
          ))}
        </div>
      </div>

      {/* BH Regime indicator */}
      {showBHOverlay && bhInstrument && (
        <div className="flex items-center gap-2 px-3 py-1 border-b border-terminal-border bg-terminal-surface text-xs">
          {(['tf15m', 'tf1h', 'tf1d'] as const).map((tf) => {
            const state = bhInstrument[tf]
            const label = tf === 'tf15m' ? '15m' : tf === 'tf1h' ? '1h' : '1d'
            const regimeColor = state.regime === 'BULL' ? 'text-terminal-bull' : state.regime === 'BEAR' ? 'text-terminal-bear' : state.regime === 'HIGH_VOL' ? 'text-terminal-warning' : 'text-terminal-subtle'
            return (
              <div key={tf} className="flex items-center gap-1">
                <span className="text-terminal-subtle">{label}</span>
                <span className={`font-mono ${regimeColor}`}>{state.mass.toFixed(2)}</span>
                {state.active && <span className={`w-1.5 h-1.5 rounded-full ${state.regime === 'BULL' ? 'bg-terminal-bull' : 'bg-terminal-bear'} animate-pulse`} />}
                <span className="text-terminal-muted">|</span>
              </div>
            )
          })}
        </div>
      )}

      {/* Chart container */}
      <div className="relative flex-1 min-h-0">
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-terminal-bg/80 z-10">
            <div className="flex items-center gap-2 text-terminal-subtle text-sm">
              <div className="w-4 h-4 border-2 border-terminal-accent border-t-transparent rounded-full animate-spin" />
              Loading chart data...
            </div>
          </div>
        )}
        <div ref={containerRef} style={{ height: `${height}px`, width: '100%' }} />
      </div>
    </div>
  )
}

export default CandlestickChart
