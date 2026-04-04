const RESEARCH_API = 'http://localhost:8766'
const ARENA_WS = 'ws://localhost:8765'
const RESEARCH_WS = 'ws://localhost:8766'

// ── HTTP client ───────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    message: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

interface FetchOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>
}

async function apiFetch<T>(path: string, opts: FetchOptions = {}): Promise<T> {
  const { params, ...rest } = opts
  let url = `${RESEARCH_API}${path}`

  if (params) {
    const qs = new URLSearchParams(
      Object.entries(params)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => [k, String(v)])
    ).toString()
    if (qs) url += `?${qs}`
  }

  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...rest.headers },
    ...rest,
  })

  if (!response.ok) {
    const body = await response.text().catch(() => '')
    throw new ApiError(response.status, response.statusText, body || `HTTP ${response.status}`)
  }

  return response.json() as Promise<T>
}

export const api = {
  get: <T>(path: string, params?: FetchOptions['params']) =>
    apiFetch<T>(path, { method: 'GET', params }),
  post: <T>(path: string, body: unknown) =>
    apiFetch<T>(path, { method: 'POST', body: JSON.stringify(body) }),
}

// ── WebSocket manager ─────────────────────────────────────────────────────────

type WSMessageHandler = (data: unknown) => void

export class ResearchWebSocket {
  private ws: WebSocket | null = null
  private handlers = new Map<string, WSMessageHandler[]>()
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private reconnectDelay = 1000
  private maxReconnectDelay = 30_000
  private url: string

  constructor(endpoint: 'research' | 'arena' = 'research') {
    this.url = endpoint === 'arena' ? ARENA_WS : RESEARCH_WS
  }

  connect(): void {
    try {
      this.ws = new WebSocket(this.url)
      this.ws.onopen = () => {
        this.reconnectDelay = 1000
        this.emit('connected', null)
      }
      this.ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data as string) as { type: string; data: unknown }
          this.emit(msg.type, msg.data)
        } catch {
          // non-JSON frames ignored
        }
      }
      this.ws.onclose = () => {
        this.emit('disconnected', null)
        this.scheduleReconnect()
      }
      this.ws.onerror = () => {
        this.emit('error', null)
      }
    } catch {
      this.scheduleReconnect()
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = setTimeout(() => {
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay)
      this.connect()
    }, this.reconnectDelay)
  }

  on(type: string, handler: WSMessageHandler): () => void {
    if (!this.handlers.has(type)) this.handlers.set(type, [])
    this.handlers.get(type)!.push(handler)
    return () => {
      const arr = this.handlers.get(type)
      if (arr) {
        const idx = arr.indexOf(handler)
        if (idx >= 0) arr.splice(idx, 1)
      }
    }
  }

  private emit(type: string, data: unknown): void {
    const handlers = this.handlers.get(type) ?? []
    for (const h of handlers) h(data)
  }

  send(type: string, data: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, data }))
    }
  }

  disconnect(): void {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.ws?.close()
  }

  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED
  }
}

// Singletons
export const researchWS = new ResearchWebSocket('research')
export const arenaWS = new ResearchWebSocket('arena')
