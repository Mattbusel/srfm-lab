// ============================================================
// hooks/useOnChainAPI.ts -- React Query hooks for on-chain
// and microstructure data. Mock data mirrors research/onchain/
// Python module schemas.
// ============================================================

import { useQuery, useQueryClient } from '@tanstack/react-query';
import type {
  OnChainResponse,
  MVRVResponse,
  FundingRatesResponse,
  VPINResponse,
  KyleLambdaResponse,
  AmihudResponse,
  NetworkSentimentResponse,
  WhaleFlowResponse,
  ExchangeName,
} from '../types/onchain';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const ONCHAIN_API_BASE = 'http://localhost:8793';
const POLL_LIVE_MS     = 15_000;
const POLL_MEDIUM_MS   = 30_000;
const POLL_SLOW_MS     = 60_000;

// ---------------------------------------------------------------------------
// Seeded pseudo-random (deterministic per session shape, noisy per call)
// ---------------------------------------------------------------------------

function s(seed: number) {
  return (Math.sin(seed * 91.3 + 17.9) * 0.5 + 0.5);
}

// ---------------------------------------------------------------------------
// Mock data generators
// ---------------------------------------------------------------------------

function generateMVRV(asset = 'BTC'): MVRVResponse {
  const now = Date.now();
  const n = 365;
  let mvrv = 1.8 + Math.random() * 0.4;
  const data = Array.from({ length: n }, (_, i) => {
    mvrv += (Math.random() - 0.49) * 0.04;
    mvrv = Math.max(0.6, Math.min(5.5, mvrv));
    const z = (mvrv - 2.1) / 0.9;
    const phase = mvrv > 3.5 ? 'distribution'
      : mvrv > 2.5 ? 'late_bull'
      : mvrv > 1.5 ? 'early_bull'
      : mvrv > 1.0 ? 'accumulation'
      : 'bear';
    return {
      date: new Date(now - (n - i) * 86_400_000).toISOString().slice(0, 10),
      mvrv,
      mvrv_z_score: z,
      market_cap_usd: mvrv * 450_000_000_000,
      realized_cap_usd: 450_000_000_000,
      cycle_phase: phase as any,
    };
  });
  const cur = data[n - 1];
  return {
    asset,
    data,
    current_mvrv: cur.mvrv,
    current_z_score: cur.mvrv_z_score,
    z_score_percentile: Math.min(99, Math.max(1, (cur.mvrv_z_score + 3) * 16.7)),
    cycle_phase: cur.cycle_phase,
    signal: cur.mvrv_z_score > 2 ? 'strong_sell'
           : cur.mvrv_z_score > 1 ? 'sell'
           : cur.mvrv_z_score < -1 ? 'buy'
           : cur.mvrv_z_score < -2 ? 'strong_buy'
           : 'neutral',
    timestamp: new Date().toISOString(),
  };
}

function generateFundingRates(symbol = 'BTC-USDT'): FundingRatesResponse {
  const exchanges: ExchangeName[] = ['Binance', 'Bybit', 'OKX', 'dYdX'];
  const now = Date.now();
  const n = 96; // 32 days of 8h data
  const exchangeData = exchanges.map(exch => {
    let rate = (Math.random() - 0.48) * 0.0003;
    const history = Array.from({ length: n }, (_, i) => {
      rate += (Math.random() - 0.49) * 0.00005;
      rate = Math.max(-0.001, Math.min(0.003, rate));
      return {
        timestamp: new Date(now - (n - i) * 8 * 3600_000).toISOString(),
        rate_8h: rate,
        rate_annualized: rate * 1095,
        predicted_rate: rate + (Math.random() - 0.5) * 0.00002,
        open_interest_usd: 2_000_000_000 + Math.random() * 1_000_000_000,
      };
    });
    const cur = history[n - 1];
    const annualized = cur.rate_annualized;
    return {
      exchange: exch,
      symbol,
      current_rate_8h: cur.rate_8h,
      current_annualized: annualized,
      predicted_rate: cur.predicted_rate,
      open_interest_usd: cur.open_interest_usd,
      history,
      regime: annualized > 0.5  ? 'contango_extreme'
            : annualized > 0.15 ? 'contango'
            : annualized < -0.5  ? 'backwardation_extreme'
            : annualized < -0.1  ? 'backwardation'
            : 'neutral' as any,
    };
  });
  const compositeRate = exchangeData.reduce((s, e) => s + e.current_rate_8h * e.open_interest_usd, 0)
    / exchangeData.reduce((s, e) => s + e.open_interest_usd, 0);
  const rates = exchangeData.map(e => e.current_annualized);
  return {
    symbol,
    exchanges: exchangeData,
    composite_rate: compositeRate,
    composite_annualized: compositeRate * 1095,
    divergence: (Math.max(...rates) - Math.min(...rates)) * 10_000, // bps
    arb_opportunity: Math.abs(Math.max(...rates) - Math.min(...rates)) > 0.15,
    timestamp: new Date().toISOString(),
  };
}

function generateVPIN(symbol = 'BTC-USDT'): VPINResponse {
  const now = Date.now();
  const n = 200;
  let vpin = 0.35 + Math.random() * 0.2;
  const history = Array.from({ length: n }, (_, i) => {
    vpin += (Math.random() - 0.49) * 0.015;
    vpin = Math.max(0.05, Math.min(0.98, vpin));
    const buyVol = 500_000 + Math.random() * 1_500_000;
    const sellVol = buyVol * (1 - (vpin - 0.5) * 0.4);
    return {
      timestamp: new Date(now - (n - i) * 3_600_000).toISOString(),
      vpin,
      buy_volume: buyVol,
      sell_volume: sellVol,
      imbalance: (buyVol - sellVol) / (buyVol + sellVol),
      bucket_index: i,
    };
  });
  const cur = history[n - 1];
  const prev = history[n - 11];
  return {
    symbol,
    current_vpin: cur.vpin,
    vpin_percentile: Math.min(99, cur.vpin * 110),
    alert_threshold: 0.70,
    is_alert: cur.vpin > 0.70,
    trend: cur.vpin > prev.vpin + 0.02 ? 'rising'
         : cur.vpin < prev.vpin - 0.02 ? 'falling'
         : 'stable',
    history,
    timestamp: new Date().toISOString(),
  };
}

function generateKyleLambda(symbol = 'BTC-USDT'): KyleLambdaResponse {
  const now = Date.now();
  const n = 100;
  let lambda = 1.2e-9 + Math.random() * 4e-10;
  const history = Array.from({ length: n }, (_, i) => {
    lambda += (Math.random() - 0.49) * 1e-10;
    lambda = Math.max(2e-10, Math.min(8e-9, lambda));
    return {
      timestamp: new Date(now - (n - i) * 3_600_000).toISOString(),
      lambda,
      r_squared: 0.55 + Math.random() * 0.35,
      n_trades: 800 + Math.floor(Math.random() * 400),
      avg_trade_size: 8_000 + Math.random() * 4_000,
      price_impact_10k_usd: lambda * 10_000 * 65_000, // price * order flow
    };
  });
  const cur = history[n - 1];
  const prev = history[n - 13];
  return {
    symbol,
    current_lambda: cur.lambda,
    lambda_percentile: Math.min(99, (cur.lambda / 8e-9) * 100),
    trend: cur.lambda > prev.lambda * 1.05 ? 'increasing'
         : cur.lambda < prev.lambda * 0.95 ? 'decreasing'
         : 'stable',
    history,
    regime: cur.lambda > 5e-9  ? 'crisis'
          : cur.lambda > 3e-9  ? 'illiquid'
          : cur.lambda > 1.5e-9 ? 'normal'
          : 'liquid',
    timestamp: new Date().toISOString(),
  };
}

function generateAmihud(symbol: string): AmihudResponse {
  const now = Date.now();
  const n = 60;
  let illiq = 0.0004 + Math.random() * 0.0003;
  const history = Array.from({ length: n }, (_, i) => {
    illiq += (Math.random() - 0.49) * 0.00005;
    illiq = Math.max(0.00005, Math.min(0.005, illiq));
    const dollarVol = 500_000_000 + Math.random() * 800_000_000;
    const absRet = 0.008 + Math.random() * 0.015;
    return {
      date: new Date(now - (n - i) * 86_400_000).toISOString().slice(0, 10),
      illiquidity: (absRet / dollarVol) * 1e6,
      dollar_volume: dollarVol,
      abs_return: absRet,
      rolling_30d_avg: illiq,
    };
  });
  const cur = history[n - 1];
  const prev = history[n - 8];
  return {
    symbol,
    current_illiquidity: cur.illiquidity,
    percentile_30d: Math.min(99, cur.illiquidity / 0.005 * 100),
    percentile_1yr: Math.min(99, cur.illiquidity / 0.005 * 85),
    trend: cur.illiquidity > prev.illiquidity * 1.1 ? 'increasing'
         : cur.illiquidity < prev.illiquidity * 0.9 ? 'decreasing'
         : 'stable',
    history,
    timestamp: new Date().toISOString(),
  };
}

function generateNetworkSentiment(): NetworkSentimentResponse {
  const now = Date.now();
  const n = 90;
  let dom = 0.52;
  let fgi = 50;
  const btcDom = Array.from({ length: n }, (_, i) => {
    dom += (Math.random() - 0.49) * 0.004;
    dom = Math.max(0.35, Math.min(0.72, dom));
    return {
      date: new Date(now - (n - i) * 86_400_000).toISOString().slice(0, 10),
      dominance: dom,
      alt_season_index: Math.round((1 - dom) * 100 + Math.random() * 15 - 7),
    };
  });
  const sentiment = Array.from({ length: n }, (_, i) => {
    fgi += (Math.random() - 0.49) * 3;
    fgi = Math.max(5, Math.min(95, fgi));
    return {
      date: new Date(now - (n - i) * 86_400_000).toISOString().slice(0, 10),
      sentiment_composite: (fgi / 50) - 1,
      nvt_signal: 60 + Math.random() * 40,
      sopr: 0.95 + Math.random() * 0.12,
      active_addresses_zscore: (Math.random() - 0.5) * 3,
      exchange_netflow_zscore: (Math.random() - 0.5) * 2.5,
      miner_outflow_zscore: (Math.random() - 0.5) * 2,
      fear_greed_index: fgi,
    };
  });
  const curSent = sentiment[n - 1];
  return {
    asset: 'BTC',
    btc_dominance: btcDom,
    sentiment,
    current_sentiment_composite: curSent.sentiment_composite,
    current_fear_greed: curSent.fear_greed_index,
    current_nvt: curSent.nvt_signal,
    current_sopr: curSent.sopr,
    sentiment_regime: curSent.fear_greed_index > 75 ? 'extreme_greed'
      : curSent.fear_greed_index > 55 ? 'greed'
      : curSent.fear_greed_index < 25 ? 'extreme_fear'
      : curSent.fear_greed_index < 45 ? 'fear'
      : 'neutral',
    timestamp: new Date().toISOString(),
  };
}

function generateWhaleFlows(): WhaleFlowResponse {
  const now = Date.now();
  const n = 48;
  const history = Array.from({ length: n }, (_, i) => {
    const inflow  = 50_000_000 + Math.random() * 200_000_000;
    const outflow = 40_000_000 + Math.random() * 190_000_000;
    return {
      timestamp: new Date(now - (n - i) * 3_600_000).toISOString(),
      inflow_usd: inflow,
      outflow_usd: outflow,
      net_flow_usd: inflow - outflow,
      flow_ratio: inflow / (inflow + outflow),
      large_txns_count: 2 + Math.floor(Math.random() * 12),
    };
  });
  const net24h = history.slice(-24).reduce((s, h) => s + h.net_flow_usd, 0);
  return {
    asset: 'BTC',
    exchange: 'all_exchanges',
    current_net_flow_24h: net24h,
    net_flow_zscore_30d: net24h / 500_000_000,
    alert: Math.abs(net24h) > 800_000_000,
    history,
    timestamp: new Date().toISOString(),
  };
}

function generateOnChain(): OnChainResponse {
  const instruments = ['BTC-USD', 'ETH-USD', 'SOL-USD'];
  return {
    mvrv: generateMVRV(),
    funding_rates: generateFundingRates(),
    vpin: generateVPIN(),
    kyle_lambda: generateKyleLambda(),
    amihud: Object.fromEntries(instruments.map(sym => [sym, generateAmihud(sym)])),
    network_sentiment: generateNetworkSentiment(),
    whale_flows: generateWhaleFlows(),
    timestamp: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function fetchOnChain<T>(path: string, fallback: () => T): Promise<T> {
  try {
    const res = await fetch(`${ONCHAIN_API_BASE}${path}`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(5_000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as T;
  } catch {
    return fallback();
  }
}

// ---------------------------------------------------------------------------
// Public hooks
// ---------------------------------------------------------------------------

/** Full on-chain dashboard -- 15 s poll. */
export function useOnChain(symbol = 'BTC-USD') {
  return useQuery<OnChainResponse>({
    queryKey: ['onchain', 'full', symbol],
    queryFn: () => fetchOnChain(`/onchain?symbol=${symbol}`, generateOnChain),
    refetchInterval: POLL_LIVE_MS,
    staleTime: POLL_LIVE_MS / 2,
  });
}

/** MVRV Z-score -- 30 s. */
export function useMVRV(asset = 'BTC') {
  return useQuery<MVRVResponse>({
    queryKey: ['onchain', 'mvrv', asset],
    queryFn: () => fetchOnChain(`/onchain/mvrv?asset=${asset}`, () => generateMVRV(asset)),
    refetchInterval: POLL_MEDIUM_MS,
    staleTime: POLL_MEDIUM_MS / 2,
  });
}

/** Funding rates -- 15 s (changes every 8 h but we want fresh data). */
export function useFundingRates(symbol = 'BTC-USDT') {
  return useQuery<FundingRatesResponse>({
    queryKey: ['onchain', 'funding', symbol],
    queryFn: () => fetchOnChain(`/onchain/funding?symbol=${symbol}`, () => generateFundingRates(symbol)),
    refetchInterval: POLL_LIVE_MS,
    staleTime: POLL_LIVE_MS / 2,
  });
}

/** VPIN -- 15 s (near real-time microstructure). */
export function useVPIN(symbol = 'BTC-USDT') {
  return useQuery<VPINResponse>({
    queryKey: ['onchain', 'vpin', symbol],
    queryFn: () => fetchOnChain(`/onchain/vpin?symbol=${symbol}`, () => generateVPIN(symbol)),
    refetchInterval: POLL_LIVE_MS,
    staleTime: POLL_LIVE_MS / 2,
  });
}

/** Kyle's Lambda -- 30 s. */
export function useKyleLambda(symbol = 'BTC-USDT') {
  return useQuery<KyleLambdaResponse>({
    queryKey: ['onchain', 'kyle_lambda', symbol],
    queryFn: () => fetchOnChain(`/onchain/kyle_lambda?symbol=${symbol}`, () => generateKyleLambda(symbol)),
    refetchInterval: POLL_MEDIUM_MS,
    staleTime: POLL_MEDIUM_MS / 2,
  });
}

/** Network sentiment + BTC dominance -- 60 s. */
export function useNetworkSentiment() {
  return useQuery<NetworkSentimentResponse>({
    queryKey: ['onchain', 'sentiment'],
    queryFn: () => fetchOnChain('/onchain/sentiment', generateNetworkSentiment),
    refetchInterval: POLL_SLOW_MS,
    staleTime: POLL_SLOW_MS / 2,
  });
}

/** Amihud illiquidity per instrument -- 60 s. */
export function useAmihud(symbol: string) {
  return useQuery<AmihudResponse>({
    queryKey: ['onchain', 'amihud', symbol],
    queryFn: () => fetchOnChain(`/onchain/amihud?symbol=${symbol}`, () => generateAmihud(symbol)),
    refetchInterval: POLL_SLOW_MS,
    staleTime: POLL_SLOW_MS / 2,
    enabled: !!symbol,
  });
}

export function useRefreshOnChain() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: ['onchain'] });
}
