// ============================================================
// useWebSocket — generic reconnecting WebSocket hook
// ============================================================
import { useEffect, useRef, useCallback, useState } from 'react'
import { ManagedWebSocket } from '@/services/ws'

export type WSStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

interface UseWebSocketOptions {
  url: string
  enabled?: boolean
  reconnectDelay?: number
  maxReconnectDelay?: number
  heartbeatIntervalMs?: number
  onMessage?: (data: unknown) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (err: Error) => void
}

interface UseWebSocketReturn {
  status: WSStatus
  send: (data: unknown) => void
  reconnect: () => void
  disconnect: () => void
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    url,
    enabled = true,
    reconnectDelay = 1000,
    maxReconnectDelay = 30000,
    heartbeatIntervalMs = 30000,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options

  const [status, setStatus] = useState<WSStatus>('disconnected')
  const wsRef = useRef<ManagedWebSocket | null>(null)
  const onMessageRef = useRef(onMessage)
  const onConnectRef = useRef(onConnect)
  const onDisconnectRef = useRef(onDisconnect)
  const onErrorRef = useRef(onError)

  useEffect(() => { onMessageRef.current = onMessage }, [onMessage])
  useEffect(() => { onConnectRef.current = onConnect }, [onConnect])
  useEffect(() => { onDisconnectRef.current = onDisconnect }, [onDisconnect])
  useEffect(() => { onErrorRef.current = onError }, [onError])

  useEffect(() => {
    if (!enabled || !url) return

    setStatus('connecting')

    const ws = new ManagedWebSocket({
      url,
      name: `hook-${url}`,
      reconnectDelay,
      maxReconnectDelay,
      heartbeatIntervalMs,
      onMessage: (data) => onMessageRef.current?.(data),
      onStatus: (st, err) => {
        if (st === 'connected') {
          setStatus('connected')
          onConnectRef.current?.()
        } else if (st === 'disconnected') {
          setStatus('disconnected')
          onDisconnectRef.current?.()
        } else if (st === 'error') {
          setStatus('error')
          onErrorRef.current?.(err ?? new Error('WebSocket error'))
        }
      },
    })

    wsRef.current = ws
    ws.connect()

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [url, enabled, reconnectDelay, maxReconnectDelay, heartbeatIntervalMs])

  const send = useCallback((data: unknown) => {
    wsRef.current?.send(data)
  }, [])

  const reconnect = useCallback(() => {
    wsRef.current?.reconnect()
  }, [])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    setStatus('disconnected')
  }, [])

  return { status, send, reconnect, disconnect }
}
