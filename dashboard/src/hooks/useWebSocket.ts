// ============================================================
// useWebSocket.ts — Real-time data hook for Spacetime Arena
// ============================================================
import { useEffect, useRef, useCallback, useState } from 'react'
import type { WsMessage } from '@/types'

const WS_URL = import.meta.env.VITE_WS_URL ?? 'ws://localhost:8765/ws/live'
const RECONNECT_DELAY_MS = 3000
const MAX_RECONNECT_DELAY_MS = 30000
const PING_INTERVAL_MS = 15000

export type WsStatus = 'connecting' | 'open' | 'closed' | 'error'

export interface UseWebSocketOptions {
  onMessage?: (msg: WsMessage) => void
  onOpen?: () => void
  onClose?: () => void
  onError?: (e: Event) => void
  enabled?: boolean
  url?: string
}

export interface UseWebSocketResult {
  status: WsStatus
  send: (msg: unknown) => void
  reconnect: () => void
  lastMessageAt: Date | null
}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketResult {
  const {
    onMessage,
    onOpen,
    onClose,
    onError,
    enabled = true,
    url = WS_URL,
  } = options

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const reconnectDelayRef = useRef(RECONNECT_DELAY_MS)
  const mountedRef = useRef(true)
  const onMessageRef = useRef(onMessage)
  const onOpenRef = useRef(onOpen)
  const onCloseRef = useRef(onClose)
  const onErrorRef = useRef(onError)

  // Keep refs up to date without triggering re-connects
  onMessageRef.current = onMessage
  onOpenRef.current = onOpen
  onCloseRef.current = onClose
  onErrorRef.current = onError

  const [status, setStatus] = useState<WsStatus>('closed')
  const [lastMessageAt, setLastMessageAt] = useState<Date | null>(null)

  const clearPing = useCallback(() => {
    if (pingIntervalRef.current != null) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }
  }, [])

  const clearReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current != null) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setStatus('connecting')

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountedRef.current) return
        reconnectDelayRef.current = RECONNECT_DELAY_MS
        setStatus('open')
        onOpenRef.current?.()

        // Start ping keep-alive
        clearPing()
        pingIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }))
          }
        }, PING_INTERVAL_MS)
      }

      ws.onmessage = (event) => {
        if (!mountedRef.current) return
        setLastMessageAt(new Date())
        try {
          const msg = JSON.parse(event.data as string) as WsMessage
          if (msg.type !== 'ping') {
            onMessageRef.current?.(msg)
          }
        } catch {
          // silently ignore non-JSON frames
        }
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        clearPing()
        setStatus('closed')
        onCloseRef.current?.()

        // Exponential backoff reconnect
        clearReconnect()
        reconnectTimeoutRef.current = setTimeout(() => {
          if (mountedRef.current && enabled) {
            reconnectDelayRef.current = Math.min(
              reconnectDelayRef.current * 1.5,
              MAX_RECONNECT_DELAY_MS,
            )
            connect()
          }
        }, reconnectDelayRef.current)
      }

      ws.onerror = (e) => {
        if (!mountedRef.current) return
        setStatus('error')
        onErrorRef.current?.(e)
      }
    } catch {
      setStatus('error')
    }
  }, [url, enabled, clearPing, clearReconnect])

  const disconnect = useCallback(() => {
    clearPing()
    clearReconnect()
    wsRef.current?.close()
    wsRef.current = null
  }, [clearPing, clearReconnect])

  const reconnect = useCallback(() => {
    disconnect()
    connect()
  }, [disconnect, connect])

  const send = useCallback((msg: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg))
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    if (enabled) connect()
    return () => {
      mountedRef.current = false
      disconnect()
    }
  }, [enabled, connect, disconnect])

  return { status, send, reconnect, lastMessageAt }
}

// ============================================================
// Typed subscription hook — subscribe to a specific message type
// ============================================================

export function useWsSubscription<T>(
  messageType: WsMessage['type'],
  handler: (payload: T) => void,
  enabled = true,
): WsStatus {
  const handlerRef = useRef(handler)
  handlerRef.current = handler

  const { status } = useWebSocket({
    enabled,
    onMessage: useCallback(
      (msg: WsMessage) => {
        if (msg.type === messageType) {
          handlerRef.current(msg.payload as T)
        }
      },
      [messageType],
    ),
  })

  return status
}
