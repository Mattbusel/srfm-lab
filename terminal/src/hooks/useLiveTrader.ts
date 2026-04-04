// ============================================================
// useLiveTrader — connects to spacetime API WebSocket for BH state
// ============================================================
import { useEffect, useCallback, useRef } from 'react'
import { useBHStore } from '@/store/bhStore'
import { useSettingsStore } from '@/store/settingsStore'
import { useMarketStore } from '@/store/marketStore'
import { wsManager } from '@/services/ws'
import { spacetimeApi } from '@/services/api'
import type {
  LiveState,
  BHState,
  BHTimeframe,
  BHFormationEvent,
  InstrumentBHState,
} from '@/types'

const generateId = () => Math.random().toString(36).slice(2, 11)

// Mock BH state generator for demo
function generateMockBHState(mass?: number): BHState {
  const m = mass ?? Math.random() * 2.5
  const regime = m > 1.5
    ? (Math.random() > 0.5 ? 'BULL' : 'BEAR') as const
    : m > 0.8
    ? 'HIGH_VOL' as const
    : 'SIDEWAYS' as const

  return {
    mass: m,
    active: m > 1.2,
    dir: (Math.random() > 0.5 ? 1 : Math.random() > 0.5 ? -1 : 0) as 1 | -1 | 0,
    ctl: Math.floor(Math.random() * 5),
    bh_form: m > 1.5 ? Math.floor(Math.random() * 3 + 1) : 0,
    regime,
    massThreshold: 1.2,
    massHistory: Array.from({ length: 20 }, () => Math.random() * 2.5),
  }
}

const MOCK_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMZN', 'MSFT', 'BTC/USD', 'ETH/USD']
const MOCK_PRICES: Record<string, number> = {
  'SPY': 540.25,
  'QQQ': 462.10,
  'AAPL': 189.50,
  'TSLA': 245.30,
  'NVDA': 875.40,
  'AMZN': 198.75,
  'MSFT': 442.80,
  'BTC/USD': 68500,
  'ETH/USD': 3750,
}

function generateMockLiveState(): LiveState {
  const instruments: Record<string, InstrumentBHState> = {}
  for (const sym of MOCK_SYMBOLS) {
    instruments[sym] = {
      symbol: sym,
      tf15m: generateMockBHState(),
      tf1h: generateMockBHState(),
      tf1d: generateMockBHState(),
      price: MOCK_PRICES[sym] ?? 100,
      frac: 1.2 + Math.random() * 0.6,
      lastUpdated: Date.now(),
    }
  }

  return {
    timestamp: new Date().toISOString(),
    equity: 125432.89,
    instruments,
    sessionPnl: 1234.56,
    sessionPnlPct: 0.0099,
    activeFormations: Object.keys(instruments).filter(
      (s) => instruments[s].tf1h.active || instruments[s].tf1d.active
    ),
  }
}

export function useLiveTrader() {
  const bhStore = useBHStore()
  const wsUrl = useSettingsStore((s) => s.settings.wsUrl)
  const mockIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const handleMessage = useCallback((data: unknown) => {
    const msg = data as { type: string; data?: unknown; timestamp?: string }
    if (!msg?.type) return

    switch (msg.type) {
      case 'live_state': {
        const state = msg.data as LiveState
        bhStore.updateLiveState(state)
        bhStore.setLastUpdate(Date.now())
        break
      }

      case 'bh_state': {
        const update = msg.data as {
          symbol: string
          tf: BHTimeframe
          state: BHState
          price: number
          frac: number
        }
        bhStore.updateTFState(update.symbol, update.tf, update.state)

        // Append mass point
        const instr = bhStore.instruments[update.symbol]
        if (instr) {
          bhStore.appendMassPoint(update.symbol, {
            timestamp: Date.now(),
            mass15m: instr.tf15m.mass,
            mass1h: instr.tf1h.mass,
            mass1d: instr.tf1d.mass,
            regime: update.state.regime,
            dir: update.state.dir,
            price: update.price,
          })
        }
        break
      }

      case 'bh_formation': {
        const evt = msg.data as Omit<BHFormationEvent, 'id' | 'acknowledged'>
        const formationEvent: BHFormationEvent = {
          ...evt,
          id: generateId(),
          acknowledged: false,
        }
        bhStore.addFormationEvent(formationEvent)
        break
      }

      case 'heartbeat': {
        bhStore.setLastUpdate(Date.now())
        break
      }
    }
  }, [bhStore])

  useEffect(() => {
    // Try to connect to spacetime API WebSocket
    const ws = wsManager.create('live-trader', {
      url: wsUrl,
      reconnectDelay: 2000,
      maxReconnectDelay: 60000,
      onMessage: handleMessage,
      onStatus: (status) => {
        bhStore.setConnected(status === 'connected')
        if (status === 'connected') {
          ws.send({ type: 'subscribe', channels: ['live_state', 'bh_formations'] })
        }
      },
    })

    ws.connect()

    return () => {
      ws.close()
    }
  }, [wsUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  // Mock BH state when not connected
  useEffect(() => {
    if (bhStore.isConnected) return

    // Generate initial state
    const initialState = generateMockLiveState()
    bhStore.updateLiveState(initialState)

    // Simulate updates
    mockIntervalRef.current = setInterval(() => {
      // Mutate prices slightly
      for (const sym of MOCK_SYMBOLS) {
        MOCK_PRICES[sym] = (MOCK_PRICES[sym] ?? 100) * (1 + (Math.random() - 0.5) * 0.002)
      }

      // Update random instrument
      const sym = MOCK_SYMBOLS[Math.floor(Math.random() * MOCK_SYMBOLS.length)]
      const tf = (['15m', '1h', '1d'] as BHTimeframe[])[Math.floor(Math.random() * 3)]
      const newState = generateMockBHState()
      bhStore.updateTFState(sym, tf, newState)

      // Occasionally fire a formation event
      if (newState.active && newState.bh_form > 0 && Math.random() < 0.05) {
        bhStore.addFormationEvent({
          id: generateId(),
          symbol: sym,
          timeframe: tf,
          mass: newState.mass,
          dir: newState.dir,
          regime: newState.regime,
          timestamp: Date.now(),
          price: MOCK_PRICES[sym] ?? 100,
          acknowledged: false,
        })
      }
    }, 2000)

    return () => {
      if (mockIntervalRef.current) clearInterval(mockIntervalRef.current)
    }
  }, [bhStore.isConnected]) // eslint-disable-line react-hooks/exhaustive-deps

  // Load BH history for watched symbols
  const loadBHHistory = useCallback(async (symbol: string) => {
    try {
      const resp = await spacetimeApi.getBHHistory(symbol, '1h', 200)
      if (resp.status === 'ok') {
        bhStore.setHistory(symbol, resp.points, [])
      }
    } catch {
      // Use mock history
      const points = Array.from({ length: 200 }, (_, i) => ({
        timestamp: Date.now() - (200 - i) * 3600000,
        mass15m: Math.random() * 2.5,
        mass1h: Math.random() * 2.5,
        mass1d: Math.random() * 2.5,
        regime: 'SIDEWAYS' as const,
        dir: 0 as const,
        price: MOCK_PRICES[symbol] ?? 100,
      }))
      bhStore.setHistory(symbol, points, [])
    }
  }, [bhStore])

  return { loadBHHistory }
}
