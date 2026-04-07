// ============================================================
// types/execution.ts -- Domain types for execution analytics,
// TCA (transaction cost analysis), and venue scoring.
// ============================================================

// ---------------------------------------------------------------------------
// Order lifecycle
// ---------------------------------------------------------------------------

export type OrderSide = 'buy' | 'sell';
export type OrderStatus = 'pending' | 'partial' | 'filled' | 'cancelled' | 'rejected';
export type OrderType = 'market' | 'limit' | 'stop' | 'stop_limit' | 'twap' | 'vwap';

export interface OrderRecord {
  /** Unique order identifier. */
  id: string;
  symbol: string;
  side: OrderSide;
  orderType: OrderType;
  /** Quantity ordered in base asset units. */
  qty: number;
  /** Quantity filled so far. */
  filledQty: number;
  /** Limit or stop price; null for market orders. */
  orderPrice: number | null;
  /** Average fill price (null if unfilled). */
  fillPrice: number | null;
  /** ISO timestamp of order submission. */
  orderTime: string;
  /** ISO timestamp of last fill (null if unfilled). */
  fillTime: string | null;
  /** Exchange or broker identifier. */
  venue: string;
  /** Originating strategy name. */
  strategy: string;
  status: OrderStatus;
  /** Any rejection reason. */
  rejectReason?: string;
  /** Notional value in USD at fill price. */
  notionalUsd?: number;
  /** Estimated slippage in basis points at time of fill. */
  slippageBps?: number;
}

// ---------------------------------------------------------------------------
// Transaction Cost Analysis
// ---------------------------------------------------------------------------

/**
 * TCA result computed post-fill.
 * All cost components expressed in basis points (bps).
 * Positive = cost, negative = savings vs benchmark.
 */
export interface TCAResult {
  orderId: string;
  symbol: string;
  venue: string;
  strategy: string;
  /** Implementation shortfall vs arrival price. */
  implShortfallBps: number;
  /** Market impact component of IS. */
  marketImpactBps: number;
  /** Timing cost component of IS. */
  timingCostBps: number;
  /** Half-spread paid. */
  spreadCostBps: number;
  /** Sum of all components. */
  totalCostBps: number;
  /** Fill price vs VWAP over fill window. */
  vwapSlippageBps: number;
  /** Arrival price used as benchmark. */
  arrivalPrice: number;
  /** VWAP over the fill period. */
  periodVwap: number;
  /** Fill price. */
  fillPrice: number;
  /** Fill timestamp. */
  fillTime: string;
  /** Order size in USD. */
  notionalUsd: number;
}

// ---------------------------------------------------------------------------
// Venue scorecard
// ---------------------------------------------------------------------------

export interface VenueScore {
  /** Exchange/broker identifier. */
  venue: string;
  /** Average slippage vs arrival in bps (positive = cost). */
  avgSlippageBps: number;
  /** Fraction of orders fully filled [0,1]. */
  fillRate: number;
  /** Median fill time from submission to last fill (ms). */
  avgFillTimeMs: number;
  /** Composite score [0, 100], higher = better. */
  score: number;
  /** Number of trades used in computation. */
  nTrades: number;
  /** Average market impact in bps. */
  avgMarketImpactBps: number;
  /** Average spread cost in bps. */
  avgSpreadCostBps: number;
  /** P95 fill time in ms. */
  p95FillTimeMs: number;
}

// ---------------------------------------------------------------------------
// Daily execution summary
// ---------------------------------------------------------------------------

export interface ExecutionSummary {
  /** ISO date string: YYYY-MM-DD. */
  date: string;
  totalOrders: number;
  filledOrders: number;
  rejectedOrders: number;
  cancelledOrders: number;
  /** Average implementation shortfall across all fills. */
  avgSlippageBps: number;
  /** Sum of all TCA costs in bps (notional-weighted). */
  totalCostBps: number;
  /** Total notional traded in USD. */
  totalNotionalUsd: number;
  /** Venue with the lowest avg slippage today. */
  bestVenue: string;
  /** Venue with the highest avg slippage today. */
  worstVenue: string;
  /** Fill rate across all orders. */
  fillRate: number;
  /** Average fill time in ms. */
  avgFillTimeMs: number;
}

// ---------------------------------------------------------------------------
// Execution cost trend (for rolling chart)
// ---------------------------------------------------------------------------

export interface ExecutionCostPoint {
  /** ISO date string: YYYY-MM-DD. */
  date: string;
  /** Rolling 7-day average TCA cost in bps. */
  avgCostBps7d: number;
  /** Daily average TCA cost in bps. */
  avgCostBpsDaily: number;
  /** Total notional traded that day. */
  totalNotionalUsd: number;
  /** Number of fills. */
  nFills: number;
}

// ---------------------------------------------------------------------------
// WebSocket order stream
// ---------------------------------------------------------------------------

export type OrderStreamEvent =
  | { type: 'order_new';    order: OrderRecord }
  | { type: 'order_update'; order: OrderRecord }
  | { type: 'order_fill';   order: OrderRecord; fillQty: number; fillPrice: number }
  | { type: 'order_cancel'; orderId: string }
  | { type: 'snapshot';     orders: OrderRecord[] };

// ---------------------------------------------------------------------------
// API response wrappers
// ---------------------------------------------------------------------------

export interface OrdersResponse {
  orders: OrderRecord[];
  total: number;
  timestamp: string;
}

export interface TCAResponse {
  results: TCAResult[];
  total: number;
  avgCostBps: number;
  timestamp: string;
}

export interface VenueScorecardResponse {
  venues: VenueScore[];
  asOf: string;
  lookbackDays: number;
}

export interface ExecutionSummaryResponse {
  summary: ExecutionSummary;
  costTrend: ExecutionCostPoint[];
  timestamp: string;
}
