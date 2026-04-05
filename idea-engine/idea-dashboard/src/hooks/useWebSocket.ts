import { useEffect, useRef, useState, useCallback } from 'react'
import type { WsMessage } from '../types'

export type WsStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

interface UseWebSocketOptions {
  url: string
  onMessage?: (msg: WsMessage) => void
  onOpen?: () => void
  onClose?: () => void
  onError?: (event: Event) => void
  reconnect?: boolean
  maxRetries?: number
  initialDelay?: number
  maxDelay?: number
}

interface UseWebSocketReturn {
  status: WsStatus
  lastMessage: WsMessage | null
  send: (data: unknown) => void
  connect: () => void
  disconnect: () => void
  retryCount: number
}

export function useWebSocket({
  url,
  onMessage,
  onOpen,
  onClose,
  onError,
  reconnect = true,
  maxRetries = 10,
  initialDelay = 500,
  maxDelay = 30000,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [status, setStatus] = useState<WsStatus>('disconnected')
  const [lastMessage, setLastMessage] = useState<WsMessage | null>(null)
  const [retryCount, setRetryCount] = useState(0)

  const wsRef = useRef<WebSocket | null>(null)
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const retryCountRef = useRef(0)
  const mountedRef = useRef(true)
  const manualDisconnectRef = useRef(false)

  const clearRetryTimer = () => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current)
      retryTimerRef.current = null
    }
  }

  const getDelay = (attempt: number): number => {
    // Exponential backoff with jitter
    const base = Math.min(initialDelay * Math.pow(2, attempt), maxDelay)
    return base + Math.random() * 1000
  }

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    manualDisconnectRef.current = false
    setStatus('connecting')

    let ws: WebSocket
    try {
      ws = new WebSocket(url)
    } catch {
      setStatus('error')
      return
    }

    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) return
      setStatus('connected')
      setRetryCount(0)
      retryCountRef.current = 0
      onOpen?.()
    }

    ws.onmessage = (event: MessageEvent) => {
      if (!mountedRef.current) return
      try {
        const msg = JSON.parse(event.data as string) as WsMessage
        setLastMessage(msg)
        onMessage?.(msg)
      } catch {
        // Ignore unparseable messages
      }
    }

    ws.onerror = (event: Event) => {
      if (!mountedRef.current) return
      setStatus('error')
      onError?.(event)
    }

    ws.onclose = () => {
      if (!mountedRef.current) return
      wsRef.current = null
      setStatus('disconnected')
      onClose?.()

      if (
        reconnect &&
        !manualDisconnectRef.current &&
        retryCountRef.current < maxRetries
      ) {
        const delay = getDelay(retryCountRef.current)
        retryCountRef.current++
        setRetryCount(retryCountRef.current)

        retryTimerRef.current = setTimeout(() => {
          if (mountedRef.current && !manualDisconnectRef.current) {
            connect()
          }
        }, delay)
      }
    }
  }, [url]) // eslint-disable-line react-hooks/exhaustive-deps

  const disconnect = useCallback(() => {
    manualDisconnectRef.current = true
    clearRetryTimer()
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setStatus('disconnected')
  }, [])

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    } else {
      console.warn('[useWebSocket] Cannot send — not connected')
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()

    return () => {
      mountedRef.current = false
      manualDisconnectRef.current = true
      clearRetryTimer()
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])

  return { status, lastMessage, send, connect, disconnect, retryCount }
}
