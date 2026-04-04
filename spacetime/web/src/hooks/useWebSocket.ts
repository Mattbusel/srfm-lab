import { useEffect, useRef, useCallback, useState } from 'react';

export type WSStatus = 'connecting' | 'open' | 'closed' | 'error';

interface UseWebSocketOptions<T> {
  url: string;
  enabled?: boolean;
  onMessage?: (data: T) => void;
  reconnectDelay?: number;
}

export function useWebSocket<T>({
  url,
  enabled = true,
  onMessage,
  reconnectDelay = 3000,
}: UseWebSocketOptions<T>) {
  const [status, setStatus] = useState<WSStatus>('closed');
  const [lastMessage, setLastMessage] = useState<T | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    if (!mountedRef.current || !enabled) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus('connecting');
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      setStatus('open');
    };

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return;
      try {
        const data = JSON.parse(evt.data) as T;
        setLastMessage(data);
        onMessageRef.current?.(data);
      } catch {
        // ignore parse errors
      }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setStatus('error');
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setStatus('closed');
      if (enabled) {
        reconnectTimer.current = setTimeout(connect, reconnectDelay);
      }
    };
  }, [url, enabled, reconnectDelay]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    if (enabled) connect();
    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [enabled, connect, disconnect]);

  return { status, lastMessage, send, disconnect, reconnect: connect };
}
