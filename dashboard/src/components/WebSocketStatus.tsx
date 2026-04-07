// ============================================================
// components/WebSocketStatus.tsx -- Connection status indicator
// for WebSocket streams. Shows connected / reconnecting /
// disconnected state with last-message timestamp and a manual
// reconnect button.
// ============================================================

import React, { useState, useEffect } from 'react';
import { clsx } from 'clsx';
import { formatRelative } from '../utils/formatters';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type WSConnectionState = 'connected' | 'reconnecting' | 'disconnected' | 'error';

export interface WebSocketStatusProps {
  /** Current connection state. */
  state: WSConnectionState;
  /** Timestamp of the last received message. */
  lastUpdate?: Date | null;
  /** Called when the user clicks the reconnect button. */
  onReconnect?: () => void;
  /** Label displayed alongside the status dot. */
  label?: string;
  /** Number of reconnect attempts so far. */
  reconnectCount?: number;
  /** Compact mode -- hides label and timestamp. */
  compact?: boolean;
  className?: string;
}

// ---------------------------------------------------------------------------
// Dot component
// ---------------------------------------------------------------------------

const StateDot: React.FC<{ state: WSConnectionState }> = ({ state }) => {
  const dotClass: Record<WSConnectionState, string> = {
    connected:    'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]',
    reconnecting: 'bg-amber-400 animate-pulse shadow-[0_0_6px_rgba(251,191,36,0.5)]',
    disconnected: 'bg-slate-600',
    error:        'bg-red-500 animate-pulse',
  };
  return (
    <span
      className={clsx('inline-block w-2 h-2 rounded-full flex-shrink-0', dotClass[state])}
    />
  );
};

// ---------------------------------------------------------------------------
// Label text
// ---------------------------------------------------------------------------

const STATE_LABEL: Record<WSConnectionState, string> = {
  connected:    'Live',
  reconnecting: 'Reconnecting',
  disconnected: 'Disconnected',
  error:        'Error',
};

const STATE_COLOR: Record<WSConnectionState, string> = {
  connected:    'text-emerald-400',
  reconnecting: 'text-amber-400',
  disconnected: 'text-slate-500',
  error:        'text-red-400',
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const WebSocketStatus: React.FC<WebSocketStatusProps> = ({
  state,
  lastUpdate,
  onReconnect,
  label,
  reconnectCount,
  compact = false,
  className,
}) => {
  // Force a re-render every 15 s so the relative timestamp stays fresh
  const [, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 15_000);
    return () => clearInterval(id);
  }, []);

  const showReconnect =
    (state === 'disconnected' || state === 'error') && onReconnect != null;

  if (compact) {
    return (
      <div className={clsx('flex items-center gap-1.5', className)}>
        <StateDot state={state} />
        <span className={clsx('text-[10px] font-mono', STATE_COLOR[state])}>
          {STATE_LABEL[state]}
        </span>
        {showReconnect && (
          <button
            onClick={onReconnect}
            className="text-[10px] font-mono text-blue-400 hover:text-blue-300 underline ml-1"
          >
            retry
          </button>
        )}
      </div>
    );
  }

  return (
    <div
      className={clsx(
        'flex items-center gap-3 px-3 py-2 rounded-lg',
        'bg-[#111318] border border-[#1e2130]',
        className,
      )}
    >
      <StateDot state={state} />

      <div className="flex flex-col min-w-0">
        <div className="flex items-center gap-2">
          <span className={clsx('text-xs font-mono font-semibold', STATE_COLOR[state])}>
            {label ?? 'Order Stream'}: {STATE_LABEL[state]}
          </span>
          {reconnectCount != null && reconnectCount > 0 && (
            <span className="text-[10px] font-mono text-slate-600">
              ({reconnectCount} reconnect{reconnectCount !== 1 ? 's' : ''})
            </span>
          )}
        </div>

        {lastUpdate && (
          <span className="text-[10px] font-mono text-slate-600 mt-0.5">
            Last msg: {formatRelative(lastUpdate)}
          </span>
        )}

        {!lastUpdate && state === 'connected' && (
          <span className="text-[10px] font-mono text-slate-600 mt-0.5">
            Waiting for data...
          </span>
        )}
      </div>

      {state === 'reconnecting' && (
        <ReconnectingSpinner />
      )}

      {showReconnect && (
        <button
          onClick={onReconnect}
          className={clsx(
            'ml-auto flex-shrink-0 px-2.5 py-1 rounded text-[10px] font-mono font-semibold',
            'bg-blue-600/20 border border-blue-600/40 text-blue-400',
            'hover:bg-blue-600/30 hover:text-blue-300 transition-colors',
            'focus:outline-none focus:ring-1 focus:ring-blue-500/50',
          )}
        >
          Reconnect
        </button>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Reconnecting spinner
// ---------------------------------------------------------------------------

const ReconnectingSpinner: React.FC = () => (
  <svg
    className="animate-spin text-amber-400 flex-shrink-0"
    width={14}
    height={14}
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <circle
      className="opacity-25"
      cx="12" cy="12" r="10"
      stroke="currentColor" strokeWidth="4"
    />
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
    />
  </svg>
);

// ---------------------------------------------------------------------------
// Compact inline variant -- just the dot + label, no card chrome
// ---------------------------------------------------------------------------

export const WSStatusInline: React.FC<{
  state: WSConnectionState;
  label?: string;
  className?: string;
}> = ({ state, label, className }) => (
  <div className={clsx('flex items-center gap-1.5', className)}>
    <StateDot state={state} />
    <span className={clsx('text-[10px] font-mono', STATE_COLOR[state])}>
      {label ?? STATE_LABEL[state]}
    </span>
  </div>
);

export default WebSocketStatus;
