import type {
  BacktestParams, BacktestResult,
  MCParams, MCResult,
  SensitivityParams, SensitivityResult,
  CorrelationResult,
  Trade,
  Instrument,
  ArchaeologyResult,
  ReportRequest,
  EfficientFrontierPoint,
} from './types';

const BASE = '/api';

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  instruments: (): Promise<Instrument[]> =>
    apiFetch('/instruments'),

  backtest: (params: BacktestParams): Promise<BacktestResult> =>
    apiFetch('/backtest', { method: 'POST', body: JSON.stringify(params) }),

  monteCarlo: (params: MCParams): Promise<MCResult> =>
    apiFetch('/mc', { method: 'POST', body: JSON.stringify(params) }),

  portfolioMC: (syms: string[], tradesJson: string): Promise<{
    frontier: EfficientFrontierPoint[];
    mc_results: Record<string, MCResult>;
  }> =>
    apiFetch('/mc/portfolio', {
      method: 'POST',
      body: JSON.stringify({ syms, trades_json: tradesJson }),
    }),

  sensitivity: (params: SensitivityParams): Promise<SensitivityResult> =>
    apiFetch('/sensitivity', { method: 'POST', body: JSON.stringify(params) }),

  correlation: (): Promise<CorrelationResult> =>
    apiFetch('/correlation'),

  trades: (filters?: {
    sym?: string;
    from_date?: string;
    to_date?: string;
    regime?: string;
    min_tf_score?: number;
  }): Promise<Trade[]> => {
    const params = new URLSearchParams();
    if (filters?.sym) params.set('sym', filters.sym);
    if (filters?.from_date) params.set('from_date', filters.from_date);
    if (filters?.to_date) params.set('to_date', filters.to_date);
    if (filters?.regime) params.set('regime', filters.regime);
    if (filters?.min_tf_score !== undefined) params.set('min_tf_score', String(filters.min_tf_score));
    const qs = params.toString();
    return apiFetch(`/trades${qs ? `?${qs}` : ''}`);
  },

  archaeology: (csv_path: string, run_name: string): Promise<ArchaeologyResult> =>
    apiFetch('/archaeology', { method: 'POST', body: JSON.stringify({ csv_path, run_name }) }),

  report: (req: ReportRequest): Promise<Blob> =>
    fetch(`${BASE}/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }).then(r => {
      if (!r.ok) throw new Error(`Report error ${r.status}`);
      return r.blob();
    }),
};
