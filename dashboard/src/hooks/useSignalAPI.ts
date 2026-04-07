// ============================================================
// hooks/useSignalAPI.ts -- React Query hooks for the signal
// research and ML pipeline. Mock data matches research/
// signal_analytics/ Python structures.
// ============================================================

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';
import type {
  SignalAnalyticsResponse,
  SignalLibraryRow,
  ICRollingResult,
  AlphaDecayResult,
  FeatureImportanceResponse,
  RegimeTimeSeries,
  BacktestResult,
  CategorySummary,
  SignalCategory,
  RegimeState,
} from '../types/signals';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const SIGNAL_API_BASE = 'http://localhost:8792';
const POLL_LIVE_MS    = 10_000;
const POLL_SLOW_MS    = 60_000;

// ---------------------------------------------------------------------------
// Mock generators
// ---------------------------------------------------------------------------

const SIGNAL_CATEGORIES: SignalCategory[] = [
  'MOMENTUM', 'MEAN_REVERSION', 'VOLATILITY',
  'MICROSTRUCTURE', 'PHYSICS', 'TECHNICAL', 'ONCHAIN', 'MACRO',
];

const SIGNAL_IDS = [
  // MOMENTUM
  'mom_roc_5', 'mom_roc_20', 'mom_roc_60', 'mom_ema_crossover', 'mom_macd_hist',
  'mom_rsi_div', 'mom_dual_momentum', 'mom_52w_hi_proximity', 'mom_acc_dist',
  'mom_breakout_volume', 'mom_price_accel', 'mom_tsmom_12_1', 'mom_cross_sec',
  'mom_idio_momentum', 'mom_vol_adj_mom', 'mom_long_run', 'mom_earnings_drift',
  'mom_gap_fade', 'mom_overnight_gap', 'mom_intraday_rev',
  // MEAN_REVERSION
  'rev_zscore_20', 'rev_zscore_60', 'rev_bollinger_pct', 'rev_rsi_extreme',
  'rev_pairs_btc_eth', 'rev_pairs_spy_qqq', 'rev_kalman_spread', 'rev_ou_params',
  'rev_cointegration_index', 'rev_half_life_opt', 'rev_bid_ask_bounce',
  'rev_overnight_gap_fade', 'rev_vol_mean_rev', 'rev_leverage_cycle',
  'rev_funding_rate_rev', 'rev_basis_trade', 'rev_carry_mean_rev',
  'rev_sentiment_reversal', 'rev_order_flow_imb', 'rev_microstructure_rev',
  // VOLATILITY
  'vol_garch_signal', 'vol_realized_var', 'vol_implied_realized_spread',
  'vol_vix_regime', 'vol_term_structure', 'vol_surface_skew', 'vol_variance_risk_prem',
  'vol_vol_of_vol', 'vol_clustering', 'vol_regime_switch', 'vol_breakout_vol',
  'vol_compression', 'vol_parkinson', 'vol_garman_klass', 'vol_yang_zhang',
  // MICROSTRUCTURE
  'micro_vpin', 'micro_kyle_lambda', 'micro_amihud', 'micro_bid_ask_spread',
  'micro_order_flow', 'micro_depth_imbalance', 'micro_trade_size_dist',
  'micro_tick_rule', 'micro_price_impact', 'micro_roll_spread',
  'micro_high_freq_mom', 'micro_order_toxicity', 'micro_pin_estimate',
  'micro_adverse_selection', 'micro_resiliency',
  // PHYSICS
  'bh_mass_signal', 'bh_accretion_rate', 'bh_hawking_temp', 'bh_entropy',
  'srfm_field_tension', 'srfm_curvature', 'hurst_signal', 'lyapunov_exp',
  'fractal_dim', 'phase_space_vol', 'entropy_rate', 'information_flow',
  'bh_tidal_force', 'ergosphere_signal', 'srfm_geodesic',
  // TECHNICAL
  'tech_ichimoku', 'tech_supertrend', 'tech_adx', 'tech_williams_r',
  'tech_mfi', 'tech_trix', 'tech_dpo', 'tech_cci', 'tech_stochastic',
  'tech_ultimate_osc', 'tech_elder_ray', 'tech_force_index',
  'tech_pvt', 'tech_obv', 'tech_chaikin_mf',
];

function seeded(i: number, scale = 1): number {
  return (Math.sin(i * 127.1 + 311.7) * 0.5 + 0.5) * scale;
}

function generateICRolling(signalId: string, n = 60): ICRollingResult {
  const now = Date.now();
  const baseIC = (seeded(signalId.length) - 0.5) * 0.14;
  return {
    signal_id: signalId,
    window: 20,
    data: Array.from({ length: n }, (_, i) => ({
      date: new Date(now - (n - i) * 86_400_000).toISOString().slice(0, 10),
      ic: baseIC + (Math.random() - 0.5) * 0.08,
      ic_spearman: baseIC * 0.92 + (Math.random() - 0.5) * 0.07,
      ic_t_stat: (baseIC / 0.04) + (Math.random() - 0.5) * 0.8,
      p_value: Math.max(0.001, Math.min(0.99, 0.15 - Math.abs(baseIC) * 1.8 + Math.random() * 0.1)),
      n_obs: 80 + Math.floor(Math.random() * 20),
    })),
    mean_ic: baseIC,
    icir: baseIC / 0.04,
    ic_positive_rate: 0.5 + baseIC * 3,
    t_stat: (baseIC / 0.04) * Math.sqrt(n),
    p_value: 0.04 + Math.random() * 0.06,
  };
}

function generateDecay(signalId: string): AlphaDecayResult {
  const cat = SIGNAL_CATEGORIES[signalId.length % SIGNAL_CATEGORIES.length];
  const ic0 = seeded(signalId.length + 1) * 0.12 + 0.01;
  const halfLife = 3 + seeded(signalId.length + 2) * 20;
  const lambda = Math.LN2 / halfLife;
  return {
    signal_id: signalId,
    category: cat,
    data: Array.from({ length: 20 }, (_, h) => ({
      horizon: h + 1,
      ic: ic0 * Math.exp(-lambda * (h + 1)),
      ic_stderr: 0.01 + Math.random() * 0.008,
    })),
    half_life_bars: halfLife,
    decay_rate: lambda,
    ic_at_zero: ic0,
    r_squared: 0.7 + seeded(signalId.length + 3) * 0.28,
    optimal_holding_period: Math.ceil(halfLife * 1.5),
    cost_adjusted_ic: ic0 * 0.7,
  };
}

function generateSignalLibrary(): SignalLibraryRow[] {
  return SIGNAL_IDS.map((id, i) => {
    const cat = SIGNAL_CATEGORIES[i % SIGNAL_CATEGORIES.length];
    const statusRoll = seeded(i + 99);
    const status = statusRoll > 0.85 ? 'retired'
      : statusRoll > 0.72 ? 'probation'
      : statusRoll > 0.05 ? 'active'
      : 'research';
    const baseIC = (seeded(i) - 0.5) * 0.14;
    return {
      meta: {
        id,
        name: id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        category: cat,
        status,
        universe: ['BTC-USD', 'ETH-USD', 'SOL-USD'].slice(0, 1 + (i % 3)),
        created_at: new Date(Date.now() - seeded(i) * 365 * 86_400_000 * 2).toISOString(),
        promoted_at: status !== 'research' ? new Date(Date.now() - seeded(i + 5) * 180 * 86_400_000).toISOString() : null,
        retired_at: status === 'retired' ? new Date(Date.now() - seeded(i + 6) * 30 * 86_400_000).toISOString() : null,
        description: `${cat} signal based on ${id.split('_').slice(1).join(' ')}`,
        lookback_bars: 5 + i % 60,
        holding_period_bars: 1 + i % 10,
      },
      ic_rolling: generateICRolling(id),
      decay: generateDecay(id),
      current_ic: baseIC + (Math.random() - 0.5) * 0.02,
      current_icir: baseIC / 0.04,
      sharpe: (seeded(i + 10) - 0.2) * 2.5,
      sortino: (seeded(i + 11) - 0.2) * 3.2,
      max_drawdown: -(seeded(i + 12) * 0.35 + 0.05),
      turnover_per_day: seeded(i + 13) * 0.8 + 0.05,
      win_rate: 0.42 + seeded(i + 14) * 0.2,
      profit_factor: 0.9 + seeded(i + 15) * 1.5,
      last_signal_value: (seeded(i + 16) - 0.5) * 2,
      regime_conditioned_ic: {
        trending:      baseIC * (1 + seeded(i + 17) * 0.5),
        mean_reverting: baseIC * (1 - seeded(i + 18) * 0.3),
        volatile:      baseIC * (0.6 + seeded(i + 19) * 0.4),
        low_vol:       baseIC * (0.8 + seeded(i + 20) * 0.3),
      },
    };
  });
}

function generateRegime(): RegimeTimeSeries {
  const now = Date.now();
  const n = 90;
  let mass = 0.6;
  let hurst = 0.52;
  const regimes: RegimeState[] = ['trending', 'mean_reverting', 'volatile', 'low_vol'];
  return {
    symbol: 'BTC-USD',
    data: Array.from({ length: n }, (_, i) => {
      mass  += (Math.random() - 0.49) * 0.02;
      hurst += (Math.random() - 0.5)  * 0.01;
      mass  = Math.max(0.1, Math.min(1.5, mass));
      hurst = Math.max(0.3, Math.min(0.8, hurst));
      const regimeState: RegimeState = hurst > 0.55 ? 'trending'
        : hurst < 0.46 ? 'mean_reverting'
        : mass > 1.0   ? 'volatile'
        : 'low_vol';
      return {
        date: new Date(now - (n - i) * 86_400_000).toISOString().slice(0, 10),
        bh_mass: mass,
        hurst_h: hurst,
        regime_state: regimeState,
        regime_probability: 0.6 + Math.random() * 0.35,
        volatility_regime: mass > 1.2 ? 4 : mass > 0.9 ? 3 : mass > 0.6 ? 2 : 1,
        adx: 15 + mass * 25 + Math.random() * 8,
        trending_score: hurst > 0.5 ? (hurst - 0.5) * 2 : 0,
      };
    }),
    current_regime: 'trending',
    current_bh_mass: mass,
    current_hurst: hurst,
    regime_duration_bars: 12,
  };
}

function generateFeatureImportance(): FeatureImportanceResponse {
  const features = SIGNAL_IDS.slice(0, 40).map((id, i) => ({
    feature_name: id,
    signal_category: SIGNAL_CATEGORIES[i % SIGNAL_CATEGORIES.length],
    importance_score: Math.max(0.001, seeded(i + 30) * 0.15),
    rank: i + 1,
    direction: (['positive', 'negative', 'mixed'] as const)[i % 3],
    stability: 0.5 + seeded(i + 31) * 0.5,
    model: 'LightGBM',
  }));
  features.sort((a, b) => b.importance_score - a.importance_score);
  features.forEach((f, i) => { f.rank = i + 1; });
  return {
    model_id: 'lgbm-alpha-v4',
    model_type: 'LightGBM',
    training_date: new Date(Date.now() - 3 * 86_400_000).toISOString(),
    n_features: features.length,
    n_samples: 48_000,
    features,
    top_categories: SIGNAL_CATEGORIES.map(cat => ({
      category: cat,
      total_importance: features
        .filter(f => f.signal_category === cat)
        .reduce((s, f) => s + f.importance_score, 0),
    })).sort((a, b) => b.total_importance - a.total_importance),
    out_of_sample_ic: 0.048,
    validation_sharpe: 1.62,
  };
}

function generateBacktest(signalId: string): BacktestResult {
  const n = 252;
  const now = Date.now();
  let eq = 100_000;
  let peak = eq;
  const data = Array.from({ length: n }, (_, i) => {
    const ret = (Math.random() - 0.47) * 0.018;
    eq *= (1 + ret);
    peak = Math.max(peak, eq);
    const dd = (eq - peak) / peak;
    return {
      date: new Date(now - (n - i) * 86_400_000).toISOString().slice(0, 10),
      equity: eq,
      benchmark: 100_000 * (1 + i * 0.00035 + Math.random() * 0.004 - 0.002),
      drawdown: dd,
      rolling_sharpe: 0.8 + Math.random() * 1.4,
      rolling_vol: 0.12 + Math.random() * 0.08,
      position_count: 1 + Math.floor(Math.random() * 3),
    };
  });
  return {
    signal_id: signalId,
    start_date: data[0].date,
    end_date: data[n - 1].date,
    initial_equity: 100_000,
    final_equity: eq,
    total_return: (eq - 100_000) / 100_000,
    annualized_return: ((eq / 100_000) ** (1 / (n / 252))) - 1,
    sharpe: 0.8 + seeded(signalId.length) * 1.6,
    sortino: 1.1 + seeded(signalId.length + 1) * 2.0,
    calmar: 0.4 + seeded(signalId.length + 2) * 1.2,
    max_drawdown: Math.min(...data.map(d => d.drawdown)),
    max_drawdown_duration_bars: 12 + Math.floor(seeded(signalId.length + 3) * 40),
    win_rate: 0.44 + seeded(signalId.length + 4) * 0.18,
    profit_factor: 1.1 + seeded(signalId.length + 5) * 0.8,
    n_trades: 80 + Math.floor(seeded(signalId.length + 6) * 120),
    equity_curve: data,
  };
}

function generateSignalAnalytics(): SignalAnalyticsResponse {
  const signals = generateSignalLibrary();
  return {
    signals,
    total_signals: signals.length,
    active_count:    signals.filter(s => s.meta.status === 'active').length,
    probation_count: signals.filter(s => s.meta.status === 'probation').length,
    retired_count:   signals.filter(s => s.meta.status === 'retired').length,
    feature_importance: generateFeatureImportance(),
    regime: generateRegime(),
    top_signals_by_ic:   signals.slice().sort((a, b) => Math.abs(b.current_ic) - Math.abs(a.current_ic)).slice(0, 10).map(s => s.meta.id),
    top_signals_by_icir: signals.slice().sort((a, b) => Math.abs(b.current_icir) - Math.abs(a.current_icir)).slice(0, 10).map(s => s.meta.id),
    timestamp: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

async function fetchSignal<T>(path: string, fallback: () => T): Promise<T> {
  try {
    const res = await fetch(`${SIGNAL_API_BASE}${path}`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(4_000),
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

/** Full signal analytics response -- 10 s poll. */
export function useSignalAnalytics() {
  return useQuery<SignalAnalyticsResponse>({
    queryKey: ['signals', 'analytics'],
    queryFn: () => fetchSignal('/signals/analytics', generateSignalAnalytics),
    refetchInterval: POLL_LIVE_MS,
    staleTime: POLL_LIVE_MS / 2,
  });
}

/** Regime time series -- 30 s. */
export function useRegime(symbol = 'BTC-USD') {
  return useQuery<RegimeTimeSeries>({
    queryKey: ['signals', 'regime', symbol],
    queryFn: () => fetchSignal(`/signals/regime?symbol=${symbol}`, generateRegime),
    refetchInterval: POLL_SLOW_MS,
    staleTime: POLL_SLOW_MS / 2,
  });
}

/** Feature importance -- 60 s (expensive ML call). */
export function useFeatureImportance() {
  return useQuery<FeatureImportanceResponse>({
    queryKey: ['signals', 'feature_importance'],
    queryFn: () => fetchSignal('/signals/feature_importance', generateFeatureImportance),
    refetchInterval: POLL_SLOW_MS,
    staleTime: POLL_SLOW_MS / 2,
  });
}

/** Backtest for a single signal -- cached, no polling. */
export function useBacktest(signalId: string) {
  return useQuery<BacktestResult>({
    queryKey: ['signals', 'backtest', signalId],
    queryFn: () => fetchSignal(`/signals/backtest/${signalId}`, () => generateBacktest(signalId)),
    staleTime: 300_000,   // 5 min -- backtests don't change in real time
    enabled: !!signalId,
  });
}

/** Category summary derived from library. */
export function useCategorySummaries(): CategorySummary[] {
  const { data } = useSignalAnalytics();
  return useMemo(() => {
    if (!data) return [];
    return SIGNAL_CATEGORIES.map(cat => {
      const sigs = data.signals.filter(s => s.meta.category === cat);
      if (!sigs.length) return { category: cat, n_signals: 0, mean_ic: 0, mean_icir: 0, mean_half_life: 0, mean_sharpe: 0, active_count: 0 };
      return {
        category: cat,
        n_signals: sigs.length,
        mean_ic:        sigs.reduce((s, x) => s + x.current_ic, 0) / sigs.length,
        mean_icir:      sigs.reduce((s, x) => s + x.current_icir, 0) / sigs.length,
        mean_half_life: sigs.reduce((s, x) => s + x.decay.half_life_bars, 0) / sigs.length,
        mean_sharpe:    sigs.reduce((s, x) => s + x.sharpe, 0) / sigs.length,
        active_count:   sigs.filter(x => x.meta.status === 'active').length,
      };
    });
  }, [data]);
}

export function useRefreshSignals() {
  const qc = useQueryClient();
  return () => qc.invalidateQueries({ queryKey: ['signals'] });
}
