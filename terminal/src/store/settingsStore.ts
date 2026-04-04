// ============================================================
// SETTINGS STORE — Zustand + Persist
// ============================================================
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import type { UserSettings, Alert, AlertRule, Notification } from '@/types'

const DEFAULT_SETTINGS: UserSettings = {
  theme: 'dark',
  layout: 'standard',
  defaultSymbol: 'SPY',
  chartInterval: '1d',
  apiUrl: 'http://localhost:8080',
  wsUrl: 'ws://localhost:8080/ws',
  gatewayUrl: 'http://localhost:9090',
  gatewayWsUrl: 'ws://localhost:9090/ws',
  alpacaApiKey: '',
  alpacaSecretKey: '',
  alpacaPaper: true,
  showBHOverlay: true,
  showRegimeColors: true,
  showPosFloorLine: true,
  alertSoundEnabled: true,
  alertVolume: 0.6,
  showVolume: true,
  showEMA20: true,
  showEMA50: true,
  showEMA200: false,
  showVolumeProfile: false,
  orderBookLevels: 10,
  watchlistSortField: 'symbol',
  watchlistSortDir: 'asc',
  dailyPnlTarget: 500,
  maxPositionSize: 0.1,
  defaultOrderType: 'limit',
  defaultTimeInForce: 'day',
  confirmOrders: true,
  hotkeysEnabled: true,
}

interface SettingsState {
  settings: UserSettings
  alerts: Alert[]
  alertRules: AlertRule[]
  notifications: Notification[]
  unreadCount: number
  isSoundEnabled: boolean
  audioContext: AudioContext | null
}

interface SettingsActions {
  updateSettings(updates: Partial<UserSettings>): void
  resetSettings(): void

  // Alerts
  addAlert(alert: Omit<Alert, 'id' | 'timestamp' | 'acknowledged'>): void
  acknowledgeAlert(alertId: string): void
  clearAlerts(): void
  removeAlert(alertId: string): void

  // Alert Rules
  addAlertRule(rule: Omit<AlertRule, 'id' | 'createdAt' | 'triggerCount'>): string
  updateAlertRule(id: string, updates: Partial<AlertRule>): void
  deleteAlertRule(id: string): void
  toggleAlertRule(id: string): void
  recordAlertTrigger(id: string): void

  // Notifications
  addNotification(notif: Omit<Notification, 'id' | 'timestamp' | 'read'>): void
  markNotificationRead(id: string): void
  markAllNotificationsRead(): void
  clearNotifications(): void

  // Sound
  playSoundAlert(type: 'info' | 'warning' | 'critical'): void
  setSoundEnabled(enabled: boolean): void
}

type SettingsStore = SettingsState & SettingsActions

const generateId = () => Math.random().toString(36).slice(2, 11)

export const useSettingsStore = create<SettingsStore>()(
  persist(
    immer((set, get) => ({
      // ---- Initial State ----
      settings: { ...DEFAULT_SETTINGS },
      alerts: [],
      alertRules: [],
      notifications: [],
      unreadCount: 0,
      isSoundEnabled: true,
      audioContext: null,

      // ---- Settings ----
      updateSettings(updates: Partial<UserSettings>) {
        set((state) => {
          Object.assign(state.settings, updates)
        })
      },

      resetSettings() {
        set((state) => {
          state.settings = { ...DEFAULT_SETTINGS }
        })
      },

      // ---- Alerts ----
      addAlert(alert: Omit<Alert, 'id' | 'timestamp' | 'acknowledged'>) {
        set((state) => {
          const newAlert: Alert = {
            ...alert,
            id: generateId(),
            timestamp: Date.now(),
            acknowledged: false,
          }
          state.alerts.unshift(newAlert)
          if (state.alerts.length > 200) state.alerts.splice(200)

          // Also create a notification
          state.notifications.unshift({
            id: generateId(),
            alertId: newAlert.id,
            type: alert.type === 'bh' ? 'bh_formation' : alert.type,
            title: alert.title,
            body: alert.message,
            timestamp: Date.now(),
            read: false,
            symbol: alert.symbol,
          })
          state.unreadCount = state.notifications.filter((n) => !n.read).length
        })

        // Play sound
        if (get().settings.alertSoundEnabled && get().isSoundEnabled) {
          get().playSoundAlert(
            alert.type === 'error' ? 'critical' : alert.type === 'warning' ? 'warning' : 'info'
          )
        }
      },

      acknowledgeAlert(alertId: string) {
        set((state) => {
          const alert = state.alerts.find((a) => a.id === alertId)
          if (alert) alert.acknowledged = true
        })
      },

      clearAlerts() {
        set((state) => {
          state.alerts = []
        })
      },

      removeAlert(alertId: string) {
        set((state) => {
          const idx = state.alerts.findIndex((a) => a.id === alertId)
          if (idx !== -1) state.alerts.splice(idx, 1)
        })
      },

      // ---- Alert Rules ----
      addAlertRule(rule: Omit<AlertRule, 'id' | 'createdAt' | 'triggerCount'>) {
        const id = generateId()
        set((state) => {
          state.alertRules.push({
            ...rule,
            id,
            createdAt: Date.now(),
            triggerCount: 0,
          })
        })
        return id
      },

      updateAlertRule(id: string, updates: Partial<AlertRule>) {
        set((state) => {
          const rule = state.alertRules.find((r) => r.id === id)
          if (rule) Object.assign(rule, updates)
        })
      },

      deleteAlertRule(id: string) {
        set((state) => {
          const idx = state.alertRules.findIndex((r) => r.id === id)
          if (idx !== -1) state.alertRules.splice(idx, 1)
        })
      },

      toggleAlertRule(id: string) {
        set((state) => {
          const rule = state.alertRules.find((r) => r.id === id)
          if (rule) rule.enabled = !rule.enabled
        })
      },

      recordAlertTrigger(id: string) {
        set((state) => {
          const rule = state.alertRules.find((r) => r.id === id)
          if (rule) {
            rule.triggerCount++
            rule.lastTriggered = Date.now()
          }
        })
      },

      // ---- Notifications ----
      addNotification(notif: Omit<Notification, 'id' | 'timestamp' | 'read'>) {
        set((state) => {
          state.notifications.unshift({
            ...notif,
            id: generateId(),
            timestamp: Date.now(),
            read: false,
          })
          if (state.notifications.length > 500) state.notifications.splice(500)
          state.unreadCount = state.notifications.filter((n) => !n.read).length
        })
      },

      markNotificationRead(id: string) {
        set((state) => {
          const notif = state.notifications.find((n) => n.id === id)
          if (notif) notif.read = true
          state.unreadCount = state.notifications.filter((n) => !n.read).length
        })
      },

      markAllNotificationsRead() {
        set((state) => {
          state.notifications.forEach((n) => { n.read = true })
          state.unreadCount = 0
        })
      },

      clearNotifications() {
        set((state) => {
          state.notifications = []
          state.unreadCount = 0
        })
      },

      // ---- Sound ----
      playSoundAlert(type: 'info' | 'warning' | 'critical') {
        try {
          const ctx = new AudioContext()
          const oscillator = ctx.createOscillator()
          const gainNode = ctx.createGain()

          oscillator.connect(gainNode)
          gainNode.connect(ctx.destination)

          const volume = get().settings.alertVolume

          if (type === 'critical') {
            oscillator.frequency.setValueAtTime(880, ctx.currentTime)
            oscillator.frequency.setValueAtTime(440, ctx.currentTime + 0.1)
            oscillator.frequency.setValueAtTime(880, ctx.currentTime + 0.2)
          } else if (type === 'warning') {
            oscillator.frequency.setValueAtTime(660, ctx.currentTime)
            oscillator.frequency.setValueAtTime(550, ctx.currentTime + 0.15)
          } else {
            oscillator.frequency.setValueAtTime(523.25, ctx.currentTime)
          }

          gainNode.gain.setValueAtTime(volume * 0.3, ctx.currentTime)
          gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4)

          oscillator.start(ctx.currentTime)
          oscillator.stop(ctx.currentTime + 0.4)
        } catch {
          // Audio not available
        }
      },

      setSoundEnabled(enabled: boolean) {
        set((state) => {
          state.isSoundEnabled = enabled
          state.settings.alertSoundEnabled = enabled
        })
      },
    })),
    {
      name: 'settings-store',
      partialize: (state) => ({
        settings: state.settings,
        alertRules: state.alertRules,
        isSoundEnabled: state.isSoundEnabled,
      }),
    }
  )
)

// ---- Selectors ----
export const selectSetting = <K extends keyof UserSettings>(key: K) =>
  (state: SettingsStore) => state.settings[key]

export const selectActiveAlertRules = (state: SettingsStore) =>
  state.alertRules.filter((r) => r.enabled)

export const selectUnreadAlerts = (state: SettingsStore) =>
  state.alerts.filter((a) => !a.acknowledged)
