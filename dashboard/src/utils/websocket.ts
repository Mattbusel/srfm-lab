// ============================================================
// utils/websocket.ts -- Auto-reconnecting WebSocket wrapper
// with exponential backoff, message queuing, and jitter.
// ============================================================

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type WSReadyState = 'connecting' | 'open' | 'closed' | 'error';

export interface WSOptions {
  /** Base reconnect delay in ms (default: 1000). */
  reconnectDelay: number;
  /** Maximum reconnect delay in ms (default: 30000). */
  maxReconnectDelay: number;
  /** Called with the parsed JSON payload of each message. */
  onMessage: (data: unknown) => void;
  /** Called when the socket successfully connects. */
  onConnect: () => void;
  /** Called when the socket disconnects (for any reason). */
  onDisconnect: () => void;
  /** Optional: called on each state change. */
  onStateChange?: (state: WSReadyState) => void;
  /** Optional: max queued messages while disconnected (default: 50). */
  maxQueueSize?: number;
  /** Optional: heartbeat interval in ms; 0 = disabled (default: 20000). */
  heartbeatIntervalMs?: number;
}

// ---------------------------------------------------------------------------
// ManagedWebSocket
// ---------------------------------------------------------------------------

/**
 * Auto-reconnecting WebSocket wrapper with:
 *   - Exponential backoff with +-20% jitter on reconnect delays
 *   - Outbound message queue drains on reconnect
 *   - Graceful close (no reconnect) via close()
 *   - Heartbeat ping to detect stale connections
 */
export class ManagedWebSocket {
  private readonly url: string;
  private readonly opts: Required<WSOptions>;

  private ws: WebSocket | null = null;
  private _state: WSReadyState = 'closed';
  private _reconnectCount = 0;
  private currentDelay: number;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private messageQueue: string[] = [];
  private intentionallyClosed = false;

  constructor(url: string, options: WSOptions) {
    this.url = url;
    this.opts = {
      reconnectDelay:     options.reconnectDelay,
      maxReconnectDelay:  options.maxReconnectDelay,
      onMessage:          options.onMessage,
      onConnect:          options.onConnect,
      onDisconnect:       options.onDisconnect,
      onStateChange:      options.onStateChange ?? (() => undefined),
      maxQueueSize:       options.maxQueueSize ?? 50,
      heartbeatIntervalMs: options.heartbeatIntervalMs ?? 20_000,
    };
    this.currentDelay = options.reconnectDelay;
    this._connect();
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /** Send a JSON-serialisable object. Queues the message if not connected. */
  send(data: object): void {
    const payload = JSON.stringify(data);
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(payload);
    } else {
      if (this.messageQueue.length < this.opts.maxQueueSize) {
        this.messageQueue.push(payload);
      }
      // Silently drop when queue is full to avoid unbounded memory
    }
  }

  /**
   * Gracefully close the socket. No reconnect will be attempted after this.
   */
  close(): void {
    this.intentionallyClosed = true;
    this._clearTimers();
    if (this.ws) {
      this.ws.onclose = null; // Suppress the onclose reconnect logic
      this.ws.close(1000, 'client_close');
      this.ws = null;
    }
    this._setState('closed');
  }

  get isConnected(): boolean {
    return this._state === 'open';
  }

  get reconnectCount(): number {
    return this._reconnectCount;
  }

  get state(): WSReadyState {
    return this._state;
  }

  // -------------------------------------------------------------------------
  // Internal: connection
  // -------------------------------------------------------------------------

  private _connect(): void {
    if (this.intentionallyClosed) return;
    this._setState('connecting');
    try {
      const ws = new WebSocket(this.url);
      this.ws = ws;

      ws.onopen = () => {
        if (this.intentionallyClosed) { ws.close(); return; }
        this._reconnectCount = 0;
        this.currentDelay = this.opts.reconnectDelay;
        this._setState('open');
        this.opts.onConnect();
        this._drainQueue();
        this._startHeartbeat();
      };

      ws.onmessage = (evt: MessageEvent) => {
        try {
          const data = JSON.parse(evt.data as string) as unknown;
          this.opts.onMessage(data);
        } catch {
          // Non-JSON frames (e.g. raw text pings) are silently ignored
        }
      };

      ws.onerror = () => {
        // onerror always precedes onclose -- let onclose handle reconnect
        this._setState('error');
      };

      ws.onclose = () => {
        this._clearHeartbeat();
        if (this.intentionallyClosed) {
          this._setState('closed');
          return;
        }
        this._setState('closed');
        this.opts.onDisconnect();
        this._scheduleReconnect();
      };
    } catch {
      this._setState('error');
      this._scheduleReconnect();
    }
  }

  // -------------------------------------------------------------------------
  // Internal: reconnect backoff
  // -------------------------------------------------------------------------

  private _scheduleReconnect(): void {
    if (this.intentionallyClosed) return;
    this._reconnectCount += 1;

    // Add +-20% jitter to avoid thundering herd
    const jitter = this.currentDelay * 0.2 * (Math.random() * 2 - 1);
    const delay  = Math.min(
      this.currentDelay + jitter,
      this.opts.maxReconnectDelay,
    );

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this._connect();
    }, Math.max(0, delay));

    // Double delay for next attempt (exponential backoff)
    this.currentDelay = Math.min(this.currentDelay * 2, this.opts.maxReconnectDelay);
  }

  // -------------------------------------------------------------------------
  // Internal: heartbeat
  // -------------------------------------------------------------------------

  private _startHeartbeat(): void {
    if (this.opts.heartbeatIntervalMs <= 0) return;
    this._clearHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify({ type: 'ping', t: Date.now() }));
        } catch {
          // If send throws the socket is likely already closing
        }
      }
    }, this.opts.heartbeatIntervalMs);
  }

  private _clearHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // -------------------------------------------------------------------------
  // Internal: queue drain
  // -------------------------------------------------------------------------

  private _drainQueue(): void {
    while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      const msg = this.messageQueue.shift()!;
      try {
        this.ws.send(msg);
      } catch {
        // Put back at front if send fails
        this.messageQueue.unshift(msg);
        break;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Internal: state management
  // -------------------------------------------------------------------------

  private _setState(state: WSReadyState): void {
    if (this._state === state) return;
    this._state = state;
    this.opts.onStateChange(state);
  }

  private _clearTimers(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this._clearHeartbeat();
  }
}

// ---------------------------------------------------------------------------
// Utility: build a managed WS URL from env or default
// ---------------------------------------------------------------------------

export function resolveWsUrl(path: string, defaultBase = 'ws://localhost:8792'): string {
  const base = (import.meta as any).env?.VITE_EXEC_WS_URL ?? defaultBase;
  return `${base}${path}`;
}
