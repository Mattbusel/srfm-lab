// ============================================================
// TYPE BARREL EXPORT
// ============================================================

export * from './market'
export * from './portfolio'
export * from './strategy'
export * from './bhphysics'

// ============================================================
// SHARED UTILITY TYPES
// ============================================================

export interface ApiResponse<T> {
  status: 'ok' | 'error'
  data?: T
  error?: string
  message?: string
  timestamp?: number
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  hasMore: boolean
}

export interface DateRange {
  start: string
  end: string
}

export interface TimeRange {
  start: number
  end: number
}

export interface SelectOption<T = string> {
  label: string
  value: T
  disabled?: boolean
  description?: string
}

export interface TableColumn<T> {
  key: keyof T | string
  label: string
  sortable?: boolean
  width?: number | string
  align?: 'left' | 'center' | 'right'
  render?: (value: unknown, row: T) => React.ReactNode
  className?: string
}

export interface Alert {
  id: string
  type: 'info' | 'success' | 'warning' | 'error' | 'bh'
  title: string
  message: string
  timestamp: number
  symbol?: string
  acknowledged: boolean
  persistent?: boolean
  sound?: boolean
  actions?: AlertAction[]
}

export interface AlertAction {
  label: string
  onClick: () => void
  variant?: 'primary' | 'secondary' | 'danger'
}

export interface AlertRule {
  id: string
  name: string
  type: 'price' | 'bh_mass' | 'bh_formation' | 'pnl' | 'equity' | 'regime'
  symbol?: string
  condition: 'above' | 'below' | 'crosses_above' | 'crosses_below' | 'equals' | 'changes'
  threshold?: number
  enabled: boolean
  sound: boolean
  persistent: boolean
  cooldownMs: number
  lastTriggered?: number
  triggerCount: number
  message?: string
  createdAt: number
}

export interface UserSettings {
  theme: 'dark' | 'light'
  layout: 'standard' | 'compact' | 'wide'
  defaultSymbol: string
  chartInterval: string
  apiUrl: string
  wsUrl: string
  gatewayUrl: string
  gatewayWsUrl: string
  alpacaApiKey: string
  alpacaSecretKey: string
  alpacaPaper: boolean
  showBHOverlay: boolean
  showRegimeColors: boolean
  showPosFloorLine: boolean
  alertSoundEnabled: boolean
  alertVolume: number
  showVolume: boolean
  showEMA20: boolean
  showEMA50: boolean
  showEMA200: boolean
  showVolumeProfile: boolean
  orderBookLevels: number
  watchlistSortField: string
  watchlistSortDir: 'asc' | 'desc'
  dailyPnlTarget: number
  maxPositionSize: number
  defaultOrderType: string
  defaultTimeInForce: string
  confirmOrders: boolean
  hotkeysEnabled: boolean
}

export interface Notification {
  id: string
  alertId?: string
  ruleId?: string
  type: 'info' | 'success' | 'warning' | 'error' | 'bh_formation'
  title: string
  body: string
  timestamp: number
  read: boolean
  symbol?: string
  actions?: { label: string; url?: string }[]
}

export interface LayoutPane {
  id: string
  component: string
  title: string
  x: number
  y: number
  w: number
  h: number
  minW?: number
  minH?: number
  props?: Record<string, unknown>
}

export interface Layout {
  id: string
  name: string
  panes: LayoutPane[]
}

export type ThemeColor =
  | 'bg'
  | 'surface'
  | 'border'
  | 'muted'
  | 'text'
  | 'subtle'
  | 'accent'
  | 'bull'
  | 'bear'
  | 'sideways'
  | 'warning'
  | 'info'
