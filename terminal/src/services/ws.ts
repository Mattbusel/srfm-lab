// ============================================================
// WEBSOCKET MANAGER — reconnecting, multi-connection
// ============================================================

type MessageHandler = (data: unknown) => void
type StatusHandler = (status: 'connected' | 'disconnected' | 'error', error?: Error) => void

interface WSOptions {
  url: string
  name: string
  reconnectDelay?: number
  maxReconnectDelay?: number
  reconnectMultiplier?: number
  maxReconnectAttempts?: number
  heartbeatIntervalMs?: number
  onMessage?: MessageHandler
  onStatus?: StatusHandler
  protocols?: string[]
}

export class ManagedWebSocket {
  private ws: WebSocket | null = null
  private options: Required<WSOptions>
  private reconnectAttempts = 0
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null
  private isClosed = false
  private messageHandlers: Set<MessageHandler> = new Set()
  private statusHandlers: Set<StatusHandler> = new Set()
  private messageQueue: string[] = []

  constructor(options: WSOptions) {
    this.options = {
      reconnectDelay: 1000,
      maxReconnectDelay: 30000,
      reconnectMultiplier: 1.5,
      maxReconnectAttempts: Infinity,
      heartbeatIntervalMs: 30000,
      onMessage: options.onMessage ?? (() => {}),
      onStatus: options.onStatus ?? (() => {}),
      protocols: options.protocols ?? [],
      ...options,
    }

    if (options.onMessage) this.messageHandlers.add(options.onMessage)
    if (options.onStatus) this.statusHandlers.add(options.onStatus)
  }

  connect() {
    if (this.isClosed) return
    try {
      const ws = new WebSocket(this.options.url, this.options.protocols.length > 0 ? this.options.protocols : undefined)
      this.ws = ws

      ws.onopen = () => {
        console.log(`[WS:${this.options.name}] Connected`)
        this.reconnectAttempts = 0
        this.startHeartbeat()
        this.notifyStatus('connected')

        // Flush queued messages
        while (this.messageQueue.length > 0) {
          const msg = this.messageQueue.shift()!
          ws.send(msg)
        }
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          for (const handler of this.messageHandlers) {
            handler(data)
          }
        } catch {
          // Raw string message
          for (const handler of this.messageHandlers) {
            handler(event.data)
          }
        }
      }

      ws.onerror = (event) => {
        console.error(`[WS:${this.options.name}] Error`, event)
        this.notifyStatus('error', new Error('WebSocket error'))
      }

      ws.onclose = (event) => {
        console.log(`[WS:${this.options.name}] Closed`, event.code, event.reason)
        this.stopHeartbeat()
        this.notifyStatus('disconnected')

        if (!this.isClosed && this.reconnectAttempts < this.options.maxReconnectAttempts) {
          this.scheduleReconnect()
        }
      }
    } catch (err) {
      console.error(`[WS:${this.options.name}] Failed to connect`, err)
      this.scheduleReconnect()
    }
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)

    const delay = Math.min(
      this.options.reconnectDelay * Math.pow(this.options.reconnectMultiplier, this.reconnectAttempts),
      this.options.maxReconnectDelay
    )

    console.log(`[WS:${this.options.name}] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})`)

    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++
      this.connect()
    }, delay)
  }

  private startHeartbeat() {
    this.stopHeartbeat()
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }))
      }
    }, this.options.heartbeatIntervalMs)
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  private notifyStatus(status: 'connected' | 'disconnected' | 'error', error?: Error) {
    for (const handler of this.statusHandlers) {
      handler(status, error)
    }
  }

  send(data: unknown) {
    const msg = typeof data === 'string' ? data : JSON.stringify(data)
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(msg)
    } else {
      this.messageQueue.push(msg)
    }
  }

  addMessageHandler(handler: MessageHandler) {
    this.messageHandlers.add(handler)
    return () => this.messageHandlers.delete(handler)
  }

  addStatusHandler(handler: StatusHandler) {
    this.statusHandlers.add(handler)
    return () => this.statusHandlers.delete(handler)
  }

  get isConnected() {
    return this.ws?.readyState === WebSocket.OPEN
  }

  get readyState() {
    return this.ws?.readyState ?? WebSocket.CLOSED
  }

  close() {
    this.isClosed = true
    this.stopHeartbeat()
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    if (this.ws) {
      this.ws.close(1000, 'Client closed')
      this.ws = null
    }
  }

  reconnect() {
    this.isClosed = false
    this.reconnectAttempts = 0
    if (this.ws) {
      this.ws.close()
    }
    this.connect()
  }
}

// ---- Global WS Connection Manager ----
class WSManager {
  private connections: Map<string, ManagedWebSocket> = new Map()

  create(name: string, options: Omit<WSOptions, 'name'>): ManagedWebSocket {
    const existing = this.connections.get(name)
    if (existing) {
      existing.close()
    }

    const ws = new ManagedWebSocket({ ...options, name })
    this.connections.set(name, ws)
    return ws
  }

  get(name: string): ManagedWebSocket | undefined {
    return this.connections.get(name)
  }

  closeAll() {
    for (const ws of this.connections.values()) {
      ws.close()
    }
    this.connections.clear()
  }

  close(name: string) {
    const ws = this.connections.get(name)
    if (ws) {
      ws.close()
      this.connections.delete(name)
    }
  }

  getStatus() {
    const status: Record<string, string> = {}
    for (const [name, ws] of this.connections.entries()) {
      const states = ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED']
      status[name] = states[ws.readyState] ?? 'UNKNOWN'
    }
    return status
  }
}

export const wsManager = new WSManager()
