// ============================================================
// ALPACA API CLIENT — orders, positions, account
// ============================================================
import type {
  AccountState,
  Position,
  Order,
  OrderRequest,
  HistoricalTrade,
  EquityPoint,
  PositionSide,
  OrderStatus,
  OrderType,
  TimeInForce,
} from '@/types'
import { useSettingsStore } from '@/store/settingsStore'

const getAlpacaConfig = () => {
  const settings = useSettingsStore.getState().settings
  return {
    apiKey: settings.alpacaApiKey,
    secretKey: settings.alpacaSecretKey,
    paper: settings.alpacaPaper,
    baseUrl: settings.alpacaPaper
      ? 'https://paper-api.alpaca.markets'
      : 'https://api.alpaca.markets',
    dataUrl: 'https://data.alpaca.markets',
  }
}

class AlpacaClient {
  private async fetch<T>(path: string, options: RequestInit = {}): Promise<T> {
    const { apiKey, secretKey, baseUrl } = getAlpacaConfig()
    const url = `${baseUrl}${path}`

    const response = await fetch(url, {
      ...options,
      headers: {
        'APCA-API-KEY-ID': apiKey,
        'APCA-API-SECRET-KEY': secretKey,
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })

    if (!response.ok) {
      const body = await response.json().catch(() => ({})) as { message?: string; code?: number }
      throw new Error(`Alpaca API ${response.status}: ${body.message ?? response.statusText}`)
    }

    if (response.status === 204) return {} as T
    return response.json() as T
  }

  // ---- Account ----
  async getAccount(): Promise<AccountState> {
    interface AlpacaAccount {
      id: string
      equity: string
      cash: string
      buying_power: string
      portfolio_value: string
      long_market_value: string
      short_market_value: string
      initial_margin: string
      maintenance_margin: string
      daytrading_buying_power: string
      regt_buying_power: string
      daytrade_count: number
      pattern_day_trader: boolean
      trading_blocked: boolean
      account_blocked: boolean
      currency: string
    }

    const raw = await this.fetch<AlpacaAccount>('/v2/account')
    const equity = parseFloat(raw.equity)
    const cash = parseFloat(raw.cash)
    const portfolioValue = parseFloat(raw.portfolio_value)
    const prevEquity = equity  // simplified

    return {
      id: raw.id,
      equity,
      cash,
      buyingPower: parseFloat(raw.buying_power),
      portfolioValue,
      longMarketValue: parseFloat(raw.long_market_value),
      shortMarketValue: parseFloat(raw.short_market_value),
      marginUsed: parseFloat(raw.initial_margin),
      marginAvailable: parseFloat(raw.buying_power),
      initialMargin: parseFloat(raw.initial_margin),
      maintenanceMargin: parseFloat(raw.maintenance_margin),
      dayPnl: 0,  // computed separately
      dayPnlPct: 0,
      totalPnl: portfolioValue - 100000,  // simplified
      totalPnlPct: (portfolioValue - 100000) / 100000,
      positions: [],
      orders: [],
      openOrderCount: 0,
      daytradingBuyingPower: parseFloat(raw.daytrading_buying_power),
      regTBuyingPower: parseFloat(raw.regt_buying_power),
      currency: raw.currency,
      tradingBlocked: raw.trading_blocked,
      accountBlocked: raw.account_blocked,
      patternDayTrader: raw.pattern_day_trader,
      daytradingCount: raw.daytrade_count,
    }
  }

  // ---- Positions ----
  async getPositions(): Promise<Position[]> {
    interface AlpacaPosition {
      symbol: string
      qty: string
      side: string
      avg_entry_price: string
      current_price: string
      unrealized_pl: string
      unrealized_plpc: string
      market_value: string
      cost_basis: string
      unrealized_intraday_pl: string
      unrealized_intraday_plpc: string
      lastday_price: string
      change_today: string
    }

    const raw = await this.fetch<AlpacaPosition[]>('/v2/positions')

    return raw.map((p) => ({
      symbol: p.symbol,
      qty: parseFloat(p.qty),
      side: p.side as PositionSide,
      entryPrice: parseFloat(p.avg_entry_price),
      currentPrice: parseFloat(p.current_price),
      unrealizedPnl: parseFloat(p.unrealized_pl),
      unrealizedPnlPct: parseFloat(p.unrealized_plpc),
      realizedPnl: 0,
      marketValue: parseFloat(p.market_value),
      costBasis: parseFloat(p.cost_basis),
      weight: 0,  // computed after all positions loaded
      dayPnl: parseFloat(p.unrealized_intraday_pl),
      dayPnlPct: parseFloat(p.unrealized_intraday_plpc),
      lastDayPrice: parseFloat(p.lastday_price),
    }))
  }

  async closePosition(symbol: string, qty?: number): Promise<Order> {
    const path = `/v2/positions/${encodeURIComponent(symbol)}`
    const options: RequestInit = { method: 'DELETE' }
    if (qty !== undefined) {
      options.body = JSON.stringify({ qty: qty.toString() })
    }
    return this.parseOrder(await this.fetch<Record<string, unknown>>(path, options))
  }

  async closeAllPositions(): Promise<void> {
    await this.fetch('/v2/positions', { method: 'DELETE' })
  }

  // ---- Orders ----
  async getOrders(status: 'open' | 'closed' | 'all' = 'open', limit = 100): Promise<Order[]> {
    const raw = await this.fetch<Record<string, unknown>[]>(
      `/v2/orders?status=${status}&limit=${limit}&direction=desc`
    )
    return raw.map((o) => this.parseOrder(o))
  }

  async getOrder(orderId: string): Promise<Order> {
    const raw = await this.fetch<Record<string, unknown>>(`/v2/orders/${encodeURIComponent(orderId)}`)
    return this.parseOrder(raw)
  }

  async submitOrder(req: OrderRequest): Promise<Order> {
    interface AlpacaOrderRequest {
      symbol: string
      qty?: string
      notional?: string
      side: string
      type: string
      time_in_force: string
      limit_price?: string
      stop_price?: string
      trail_price?: string
      trail_percent?: string
      extended_hours?: boolean
      client_order_id?: string
    }

    const body: AlpacaOrderRequest = {
      symbol: req.symbol,
      side: req.side,
      type: req.type,
      time_in_force: req.timeInForce,
    }

    if (req.qty !== undefined) body.qty = req.qty.toString()
    if (req.notional !== undefined) body.notional = req.notional.toString()
    if (req.price !== undefined) body.limit_price = req.price.toString()
    if (req.stopPrice !== undefined) body.stop_price = req.stopPrice.toString()
    if (req.trailPrice !== undefined) body.trail_price = req.trailPrice.toString()
    if (req.trailPercent !== undefined) body.trail_percent = req.trailPercent.toString()
    if (req.extendedHours) body.extended_hours = req.extendedHours
    if (req.clientOrderId) body.client_order_id = req.clientOrderId

    const raw = await this.fetch<Record<string, unknown>>('/v2/orders', {
      method: 'POST',
      body: JSON.stringify(body),
    })
    return this.parseOrder(raw)
  }

  async cancelOrder(orderId: string): Promise<void> {
    await this.fetch(`/v2/orders/${encodeURIComponent(orderId)}`, { method: 'DELETE' })
  }

  async cancelAllOrders(): Promise<void> {
    await this.fetch('/v2/orders', { method: 'DELETE' })
  }

  async replaceOrder(orderId: string, updates: { qty?: number; price?: number; stopPrice?: number; timeInForce?: TimeInForce }): Promise<Order> {
    const body: Record<string, string> = {}
    if (updates.qty !== undefined) body.qty = updates.qty.toString()
    if (updates.price !== undefined) body.limit_price = updates.price.toString()
    if (updates.stopPrice !== undefined) body.stop_price = updates.stopPrice.toString()
    if (updates.timeInForce) body.time_in_force = updates.timeInForce

    const raw = await this.fetch<Record<string, unknown>>(`/v2/orders/${encodeURIComponent(orderId)}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    })
    return this.parseOrder(raw)
  }

  // ---- Portfolio History ----
  async getEquityHistory(days = 30): Promise<EquityPoint[]> {
    interface AlpacaPortfolioHistory {
      timestamp: number[]
      equity: number[]
      profit_loss: number[]
      profit_loss_pct: number[]
      base_value: number
    }

    const raw = await this.fetch<AlpacaPortfolioHistory>(
      `/v2/account/portfolio/history?period=${days}D&timeframe=1D`
    )

    return raw.timestamp.map((ts, i) => ({
      timestamp: ts * 1000,
      equity: raw.equity[i] ?? 0,
      cash: 0,
      longValue: raw.equity[i] ?? 0,
      shortValue: 0,
      dayPnl: raw.profit_loss[i] ?? 0,
      totalPnl: (raw.equity[i] ?? 0) - raw.base_value,
    }))
  }

  // ---- Trade History ----
  async getTradeHistory(params: {
    startDate?: string
    endDate?: string
    symbol?: string
    limit?: number
  } = {}): Promise<HistoricalTrade[]> {
    const query = new URLSearchParams()
    if (params.startDate) query.set('after', params.startDate)
    if (params.endDate) query.set('until', params.endDate)
    if (params.limit) query.set('limit', params.limit.toString())

    const activities = await this.fetch<Record<string, unknown>[]>(
      `/v2/account/activities/FILL?${query.toString()}`
    )

    // Group by order, build trade records
    const trades: HistoricalTrade[] = []
    for (const act of activities) {
      if (params.symbol && act['symbol'] !== params.symbol) continue
      trades.push({
        id: String(act['id'] ?? Math.random()),
        orderId: String(act['order_id'] ?? ''),
        symbol: String(act['symbol'] ?? ''),
        side: String(act['side'] ?? '') === 'buy' ? 'buy' : 'sell',
        qty: parseFloat(String(act['qty'] ?? '0')),
        price: parseFloat(String(act['price'] ?? '0')),
        commission: 0,
        pnl: 0,  // computed post-hoc
        pnlPct: 0,
        holdingPeriod: 0,
        entryTime: new Date(String(act['transaction_time'] ?? Date.now())).getTime(),
        exitTime: new Date(String(act['transaction_time'] ?? Date.now())).getTime(),
        entryPrice: parseFloat(String(act['price'] ?? '0')),
        exitPrice: parseFloat(String(act['price'] ?? '0')),
      })
    }

    return trades
  }

  // ---- Helpers ----
  private parseOrder(raw: Record<string, unknown>): Order {
    return {
      id: String(raw['id'] ?? ''),
      clientOrderId: raw['client_order_id'] as string | undefined,
      symbol: String(raw['symbol'] ?? ''),
      side: String(raw['side'] ?? 'buy') === 'buy' ? 'buy' : 'sell',
      qty: parseFloat(String(raw['qty'] ?? '0')),
      type: (raw['type'] as OrderType) ?? 'market',
      status: this.parseOrderStatus(String(raw['status'] ?? 'pending')),
      timeInForce: (raw['time_in_force'] as TimeInForce) ?? 'day',
      price: raw['limit_price'] ? parseFloat(String(raw['limit_price'])) : undefined,
      stopPrice: raw['stop_price'] ? parseFloat(String(raw['stop_price'])) : undefined,
      filledQty: parseFloat(String(raw['filled_qty'] ?? '0')),
      avgFillPrice: parseFloat(String(raw['filled_avg_price'] ?? '0')),
      commission: 0,
      createdAt: new Date(String(raw['created_at'] ?? Date.now())).getTime(),
      updatedAt: new Date(String(raw['updated_at'] ?? Date.now())).getTime(),
      submittedAt: raw['submitted_at'] ? new Date(String(raw['submitted_at'])).getTime() : undefined,
      filledAt: raw['filled_at'] ? new Date(String(raw['filled_at'])).getTime() : undefined,
      cancelledAt: raw['canceled_at'] ? new Date(String(raw['canceled_at'])).getTime() : undefined,
      extendedHours: Boolean(raw['extended_hours']),
    }
  }

  private parseOrderStatus(raw: string): OrderStatus {
    const map: Record<string, OrderStatus> = {
      new: 'pending',
      accepted: 'accepted',
      pending_new: 'pending',
      accepted_for_bidding: 'accepted',
      stopped: 'pending',
      rejected: 'rejected',
      suspended: 'cancelled',
      calculated: 'pending',
      filled: 'filled',
      partially_filled: 'partial',
      canceled: 'cancelled',
      expired: 'expired',
      replaced: 'cancelled',
    }
    return map[raw] ?? 'pending'
  }

  // ---- Mock Account (paper/demo mode) ----
  getMockAccount(): AccountState {
    return {
      id: 'mock-account-001',
      equity: 125432.89,
      cash: 45678.12,
      buyingPower: 91356.24,
      portfolioValue: 125432.89,
      longMarketValue: 79754.77,
      shortMarketValue: 0,
      marginUsed: 15234.56,
      marginAvailable: 76121.68,
      initialMargin: 15234.56,
      maintenanceMargin: 11425.92,
      dayPnl: 1234.56,
      dayPnlPct: 0.00994,
      totalPnl: 25432.89,
      totalPnlPct: 0.2543,
      positions: [],
      orders: [],
      openOrderCount: 3,
      daytradingBuyingPower: 91356.24,
      regTBuyingPower: 45678.12,
      currency: 'USD',
      tradingBlocked: false,
      accountBlocked: false,
      patternDayTrader: false,
      daytradingCount: 1,
    }
  }
}

export const alpacaService = new AlpacaClient()
