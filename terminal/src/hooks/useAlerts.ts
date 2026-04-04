// ============================================================
// useAlerts — manages alert rules and notifications
// ============================================================
import { useEffect, useRef, useCallback } from 'react'
import { useSettingsStore } from '@/store/settingsStore'
import { useMarketStore } from '@/store/marketStore'
import { usePortfolioStore } from '@/store/portfolioStore'
import { useBHStore } from '@/store/bhStore'
import type { AlertRule } from '@/types'

export function useAlerts() {
  const settingsStore = useSettingsStore()
  const marketStore = useMarketStore()
  const portfolioStore = usePortfolioStore()
  const bhStore = useBHStore()

  const alertRules = settingsStore.alertRules
  const prevValuesRef = useRef<Record<string, number>>({})
  const checkIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const checkRule = useCallback((rule: AlertRule) => {
    if (!rule.enabled) return

    // Cooldown check
    if (rule.lastTriggered && Date.now() - rule.lastTriggered < rule.cooldownMs) return

    let currentValue: number | undefined
    let prevValue: number | undefined
    const valueKey = `${rule.id}-value`

    switch (rule.type) {
      case 'price': {
        if (!rule.symbol) return
        const quote = marketStore.quotes[rule.symbol]
        if (!quote) return
        currentValue = quote.lastPrice
        prevValue = prevValuesRef.current[valueKey]
        break
      }

      case 'bh_mass': {
        if (!rule.symbol) return
        const instr = bhStore.instruments[rule.symbol]
        if (!instr) return
        currentValue = Math.max(instr.tf15m.mass, instr.tf1h.mass, instr.tf1d.mass)
        prevValue = prevValuesRef.current[valueKey]
        break
      }

      case 'pnl': {
        currentValue = portfolioStore.account?.dayPnl ?? 0
        prevValue = prevValuesRef.current[valueKey]
        break
      }

      case 'equity': {
        currentValue = portfolioStore.account?.equity ?? 0
        prevValue = prevValuesRef.current[valueKey]
        break
      }

      case 'bh_formation': {
        if (!rule.symbol) return
        const instr = bhStore.instruments[rule.symbol]
        if (!instr) return
        const hasFormation = instr.tf15m.bh_form > 0 || instr.tf1h.bh_form > 0 || instr.tf1d.bh_form > 0
        if (hasFormation) {
          trigger(rule, `BH Formation detected for ${rule.symbol}`)
        }
        return
      }

      default:
        return
    }

    if (currentValue === undefined) return

    // Update prev value for next check
    const prevValueStored = prevValuesRef.current[valueKey]
    prevValuesRef.current[valueKey] = currentValue

    if (rule.threshold === undefined) return

    let triggered = false
    let message = ''

    switch (rule.condition) {
      case 'above':
        triggered = currentValue > rule.threshold
        message = `${rule.symbol ?? ''} ${rule.type} is above ${rule.threshold} (currently ${currentValue.toFixed(2)})`
        break
      case 'below':
        triggered = currentValue < rule.threshold
        message = `${rule.symbol ?? ''} ${rule.type} is below ${rule.threshold} (currently ${currentValue.toFixed(2)})`
        break
      case 'crosses_above':
        triggered = prevValueStored !== undefined && prevValueStored <= rule.threshold && currentValue > rule.threshold
        message = `${rule.symbol ?? ''} ${rule.type} crossed above ${rule.threshold}`
        break
      case 'crosses_below':
        triggered = prevValueStored !== undefined && prevValueStored >= rule.threshold && currentValue < rule.threshold
        message = `${rule.symbol ?? ''} ${rule.type} crossed below ${rule.threshold}`
        break
    }

    if (triggered) {
      trigger(rule, rule.message ?? message)
    }
  }, [marketStore.quotes, bhStore.instruments, portfolioStore.account])

  const trigger = useCallback((rule: AlertRule, message: string) => {
    settingsStore.addAlert({
      type: rule.type === 'bh_mass' || rule.type === 'bh_formation' ? 'bh' : 'warning',
      title: rule.name,
      message,
      symbol: rule.symbol,
      sound: rule.sound,
      persistent: rule.persistent,
    })
    settingsStore.recordAlertTrigger(rule.id)
  }, [settingsStore])

  // Periodic alert check
  useEffect(() => {
    checkIntervalRef.current = setInterval(() => {
      for (const rule of alertRules) {
        checkRule(rule)
      }
    }, 2000)

    return () => {
      if (checkIntervalRef.current) clearInterval(checkIntervalRef.current)
    }
  }, [alertRules, checkRule])

  // BH formation alerts — react to store changes
  useEffect(() => {
    const unackFormations = bhStore.formationEvents.filter((e) => !e.acknowledged)
    for (const formation of unackFormations) {
      // Check if there's a formation alert rule for this symbol
      const rule = alertRules.find(
        (r) =>
          r.type === 'bh_formation' &&
          (r.symbol === formation.symbol || !r.symbol) &&
          r.enabled
      )

      if (rule) {
        const cooldownOk = !rule.lastTriggered || Date.now() - rule.lastTriggered > rule.cooldownMs
        if (cooldownOk) {
          trigger(rule, `BH Formation on ${formation.symbol} (${formation.timeframe}): mass=${formation.mass.toFixed(2)}, dir=${formation.dir > 0 ? 'UP' : 'DOWN'}, regime=${formation.regime}`)
        }
      }

      bhStore.acknowledgeFormation(formation.id)
    }
  }, [bhStore.formationEvents.length]) // eslint-disable-line react-hooks/exhaustive-deps

  const createPriceAlert = useCallback((symbol: string, price: number, condition: 'above' | 'below') => {
    return settingsStore.addAlertRule({
      name: `${symbol} price ${condition} ${price}`,
      type: 'price',
      symbol,
      condition,
      threshold: price,
      enabled: true,
      sound: settingsStore.settings.alertSoundEnabled,
      persistent: false,
      cooldownMs: 60000,
    })
  }, [settingsStore])

  const createBHFormationAlert = useCallback((symbol?: string) => {
    return settingsStore.addAlertRule({
      name: symbol ? `BH Formation — ${symbol}` : 'BH Formation — All',
      type: 'bh_formation',
      symbol,
      condition: 'above',
      enabled: true,
      sound: true,
      persistent: true,
      cooldownMs: 300000,  // 5 min cooldown per symbol
    })
  }, [settingsStore])

  return {
    alertRules,
    alerts: settingsStore.alerts,
    notifications: settingsStore.notifications,
    unreadCount: settingsStore.unreadCount,
    createPriceAlert,
    createBHFormationAlert,
    acknowledgeAlert: settingsStore.acknowledgeAlert,
    clearAlerts: settingsStore.clearAlerts,
  }
}
