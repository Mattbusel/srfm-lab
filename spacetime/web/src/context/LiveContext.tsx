import React, { createContext, useReducer, useCallback, useMemo } from 'react';
import type { LiveMessage, EquityPoint } from '../types';
import { useWebSocket } from '../hooks/useWebSocket';

interface LiveState {
  latest: LiveMessage | null;
  equityHistory: EquityPoint[];
  wsStatus: 'connecting' | 'open' | 'closed' | 'error';
  equityAtDayStart: number | null;
}

type LiveAction =
  | { type: 'MESSAGE'; payload: LiveMessage }
  | { type: 'WS_STATUS'; payload: LiveState['wsStatus'] };

const MAX_EQUITY_HISTORY = 2000;

function liveReducer(state: LiveState, action: LiveAction): LiveState {
  switch (action.type) {
    case 'MESSAGE': {
      const msg = action.payload;
      const newPoint: EquityPoint = {
        date: msg.timestamp,
        equity: msg.equity,
      };
      const equityHistory = [...state.equityHistory, newPoint].slice(-MAX_EQUITY_HISTORY);

      // Track day start equity (first message of day)
      const todayStr = new Date().toDateString();
      const dayStartPoint = state.equityHistory.find(
        p => new Date(p.date).toDateString() === todayStr
      );
      const equityAtDayStart = dayStartPoint?.equity ?? state.equityAtDayStart ?? msg.equity;

      return { ...state, latest: msg, equityHistory, equityAtDayStart };
    }
    case 'WS_STATUS':
      return { ...state, wsStatus: action.payload };
    default:
      return state;
  }
}

const initialState: LiveState = {
  latest: null,
  equityHistory: [],
  wsStatus: 'closed',
  equityAtDayStart: null,
};

interface LiveContextValue extends LiveState {
  pnlToday: number;
  activePositions: number;
}

export const LiveContext = createContext<LiveContextValue | null>(null);

export function LiveProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(liveReducer, initialState);

  const handleMessage = useCallback((msg: LiveMessage) => {
    dispatch({ type: 'MESSAGE', payload: msg });
  }, []);

  const { status } = useWebSocket<LiveMessage>({
    url: 'ws://localhost:8000/ws/live',
    onMessage: handleMessage,
    reconnectDelay: 5000,
  });

  // Sync WS status to state
  React.useEffect(() => {
    dispatch({ type: 'WS_STATUS', payload: status });
  }, [status]);

  const value = useMemo<LiveContextValue>(() => {
    const pnlToday = state.latest && state.equityAtDayStart !== null
      ? state.latest.equity - state.equityAtDayStart
      : 0;

    const activePositions = state.latest
      ? Object.values(state.latest.instruments).filter(
          i => i.active_15m || i.active_1h || i.active_1d
        ).length
      : 0;

    return { ...state, pnlToday, activePositions };
  }, [state]);

  return <LiveContext.Provider value={value}>{children}</LiveContext.Provider>;
}
