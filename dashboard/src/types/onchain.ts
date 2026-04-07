// ============================================================
// types/onchain.ts -- TypeScript interfaces for on-chain and
// microstructure data APIs.
//
// Mirrors research/onchain/ Python modules.
// ============================================================

// ---------------------------------------------------------------------------
// MVRV
// ---------------------------------------------------------------------------

export interface MVRVPoint {
  date: string;
  mvrv: number;               // Market Value / Realized Value
  mvrv_z_score: number;       // z-score vs rolling mean/std
  market_cap_usd: number;
  realized_cap_usd: number;
  cycle_phase: 'accumulation' | 'early_bull' | 'late_bull' | 'distribution' | 'bear';
}

export interface MVRVResponse {
  asset: string;
  data: MVRVPoint[];
  current_mvrv: number;
  current_z_score: number;
  z_score_percentile: number; // 0..100 vs 4yr history
  cycle_phase: string;
  signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell';
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Funding rates
// ---------------------------------------------------------------------------

export type ExchangeName = 'Binance' | 'Bybit' | 'OKX' | 'dYdX' | 'GMX';

export interface FundingRatePoint {
  timestamp: string;
  rate_8h: number;            // 8-hour rate (decimal)
  rate_annualized: number;    // annualized (x1095)
  predicted_rate: number;     // next period
  open_interest_usd: number;
}

export interface ExchangeFundingData {
  exchange: ExchangeName;
  symbol: string;
  current_rate_8h: number;
  current_annualized: number;
  predicted_rate: number;
  open_interest_usd: number;
  history: FundingRatePoint[];
  regime: 'contango_extreme' | 'contango' | 'neutral' | 'backwardation' | 'backwardation_extreme';
}

export interface FundingRatesResponse {
  symbol: string;
  exchanges: ExchangeFundingData[];
  composite_rate: number;     // weighted by OI
  composite_annualized: number;
  divergence: number;         // max - min across exchanges (bps)
  arb_opportunity: boolean;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// VPIN (Volume-Synchronized Probability of Informed Trading)
// ---------------------------------------------------------------------------

export interface VPINPoint {
  timestamp: string;
  vpin: number;               // 0..1 (higher = more informed)
  buy_volume: number;
  sell_volume: number;
  imbalance: number;          // (buy - sell) / total
  bucket_index: number;
}

export interface VPINResponse {
  symbol: string;
  current_vpin: number;
  vpin_percentile: number;    // 0..100 vs 30-day history
  alert_threshold: number;    // e.g. 0.70
  is_alert: boolean;
  trend: 'rising' | 'falling' | 'stable';
  history: VPINPoint[];
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Kyle's Lambda (market impact coefficient)
// ---------------------------------------------------------------------------

export interface KyleLambdaPoint {
  timestamp: string;
  lambda: number;             // price impact per unit order flow
  r_squared: number;
  n_trades: number;
  avg_trade_size: number;
  price_impact_10k_usd: number; // estimated impact for $10k order
}

export interface KyleLambdaResponse {
  symbol: string;
  current_lambda: number;
  lambda_percentile: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  history: KyleLambdaPoint[];
  regime: 'liquid' | 'normal' | 'illiquid' | 'crisis';
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Amihud illiquidity
// ---------------------------------------------------------------------------

export interface AmihudPoint {
  date: string;
  illiquidity: number;        // |return| / dollar_volume * 1e6
  dollar_volume: number;
  abs_return: number;
  rolling_30d_avg: number;
}

export interface AmihudResponse {
  symbol: string;
  current_illiquidity: number;
  percentile_30d: number;     // 0..100
  percentile_1yr: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  history: AmihudPoint[];
  timestamp: string;
}

// ---------------------------------------------------------------------------
// BTC dominance and network sentiment
// ---------------------------------------------------------------------------

export interface BtcDominancePoint {
  date: string;
  dominance: number;          // 0..1
  alt_season_index: number;   // 0..100
}

export interface NetworkSentimentPoint {
  date: string;
  sentiment_composite: number; // -1..+1
  nvt_signal: number;
  sopr: number;               // Spent Output Profit Ratio
  active_addresses_zscore: number;
  exchange_netflow_zscore: number;
  miner_outflow_zscore: number;
  fear_greed_index: number;   // 0..100
}

export interface NetworkSentimentResponse {
  asset: string;
  btc_dominance: BtcDominancePoint[];
  sentiment: NetworkSentimentPoint[];
  current_sentiment_composite: number;
  current_fear_greed: number;
  current_nvt: number;
  current_sopr: number;
  sentiment_regime: 'extreme_fear' | 'fear' | 'neutral' | 'greed' | 'extreme_greed';
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Whale exchange flows
// ---------------------------------------------------------------------------

export interface WhaleFlowPoint {
  timestamp: string;
  inflow_usd: number;
  outflow_usd: number;
  net_flow_usd: number;       // positive = net deposit (bearish)
  flow_ratio: number;         // inflow / (inflow + outflow)
  large_txns_count: number;   // transactions > $1M
}

export interface WhaleFlowResponse {
  asset: string;
  exchange: string;
  current_net_flow_24h: number;
  net_flow_zscore_30d: number;
  alert: boolean;
  history: WhaleFlowPoint[];
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Full on-chain API response
// ---------------------------------------------------------------------------

export interface OnChainResponse {
  mvrv: MVRVResponse;
  funding_rates: FundingRatesResponse;
  vpin: VPINResponse;
  kyle_lambda: KyleLambdaResponse;
  amihud: Record<string, AmihudResponse>;   // keyed by symbol
  network_sentiment: NetworkSentimentResponse;
  whale_flows: WhaleFlowResponse;
  timestamp: string;
}
