// ============================================================
// Alerts — alert management page
// ============================================================
import React, { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSettingsStore } from '@/store/settingsStore'
import { format } from 'date-fns'
import type { AlertRule, Alert } from '@/types'

type AlertTab = 'active' | 'rules' | 'history'

export const Alerts: React.FC = () => {
  const store = useSettingsStore()
  const alerts = store.alerts
  const alertRules = store.alertRules
  const [tab, setTab] = useState<AlertTab>('active')
  const [showNewRule, setShowNewRule] = useState(false)

  // New rule form state
  const [newRule, setNewRule] = useState<Partial<AlertRule>>({
    name: '',
    type: 'price',
    symbol: '',
    condition: 'above',
    threshold: 0,
    enabled: true,
    sound: true,
    persistent: false,
    cooldownMs: 60000,
    message: '',
  })

  const handleCreateRule = useCallback(() => {
    if (!newRule.name?.trim() || !newRule.type) return
    store.addAlertRule({
      name: newRule.name!,
      type: newRule.type as AlertRule['type'],
      symbol: newRule.symbol || undefined,
      condition: newRule.condition as AlertRule['condition'],
      threshold: newRule.threshold,
      enabled: newRule.enabled ?? true,
      sound: newRule.sound ?? true,
      persistent: newRule.persistent ?? false,
      cooldownMs: newRule.cooldownMs ?? 60000,
      message: newRule.message || undefined,
    })
    setShowNewRule(false)
    setNewRule({ name: '', type: 'price', condition: 'above', threshold: 0, enabled: true, sound: true, persistent: false, cooldownMs: 60000 })
  }, [newRule, store])

  const activeAlerts = alerts.filter((a) => !a.acknowledged)
  const acknowledgedAlerts = alerts.filter((a) => a.acknowledged)

  return (
    <div className="flex flex-col h-full bg-terminal-bg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-terminal-border flex-shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-terminal-text font-mono text-sm font-semibold">Alerts</span>
          {activeAlerts.length > 0 && (
            <span className="bg-terminal-bear/20 text-terminal-bear text-[10px] font-mono px-2 py-0.5 rounded-full">
              {activeAlerts.length} unacknowledged
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => store.setSoundEnabled(!store.isSoundEnabled)}
            className={`text-[11px] font-mono px-2 py-1 rounded border transition-colors ${
              store.isSoundEnabled ? 'text-terminal-bull border-terminal-bull/30 hover:bg-terminal-bull/10' : 'text-terminal-subtle border-terminal-border hover:text-terminal-text'
            }`}
          >
            {store.isSoundEnabled ? '🔔 Sound On' : '🔕 Sound Off'}
          </button>
          <button
            onClick={() => store.clearAlerts()}
            className="text-[11px] font-mono px-2 py-1 rounded border border-terminal-border text-terminal-subtle hover:text-terminal-text transition-colors"
          >
            Clear All
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-terminal-border flex-shrink-0">
        {([
          { key: 'active', label: `Active (${activeAlerts.length})` },
          { key: 'rules', label: `Rules (${alertRules.length})` },
          { key: 'history', label: `History (${alerts.length})` },
        ] as const).map((t) => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-xs font-mono transition-colors ${tab === t.key ? 'text-terminal-text border-b-2 border-terminal-accent' : 'text-terminal-subtle hover:text-terminal-text'}`}
          >{t.label}</button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Active alerts */}
        {tab === 'active' && (
          <div className="p-3 space-y-2">
            <AnimatePresence>
              {activeAlerts.length === 0 ? (
                <div className="flex items-center justify-center py-12 text-terminal-subtle text-sm">
                  No active alerts
                </div>
              ) : (
                activeAlerts.map((alert) => (
                  <motion.div
                    key={alert.id}
                    layout
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className={`p-3 rounded border ${
                      alert.type === 'error' ? 'bg-terminal-bear/10 border-terminal-bear/30' :
                      alert.type === 'warning' ? 'bg-terminal-warning/10 border-terminal-warning/30' :
                      alert.type === 'bh' ? 'bg-terminal-accent/10 border-terminal-accent/30' :
                      alert.type === 'success' ? 'bg-terminal-bull/10 border-terminal-bull/30' :
                      'bg-terminal-surface border-terminal-border'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-0.5">
                          <span className="font-mono text-xs font-semibold text-terminal-text">{alert.title}</span>
                          {alert.symbol && (
                            <span className="text-[10px] font-mono text-terminal-subtle bg-terminal-muted rounded px-1">{alert.symbol}</span>
                          )}
                          <span className="text-[9px] font-mono text-terminal-muted">{format(new Date(alert.timestamp), 'HH:mm:ss')}</span>
                        </div>
                        <p className="text-[11px] font-mono text-terminal-subtle">{alert.message}</p>
                      </div>
                      <button
                        onClick={() => store.acknowledgeAlert(alert.id)}
                        className="text-[10px] font-mono text-terminal-subtle hover:text-terminal-text transition-colors flex-shrink-0"
                      >
                        ✓ Ack
                      </button>
                    </div>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </div>
        )}

        {/* Alert rules */}
        {tab === 'rules' && (
          <div className="p-3 space-y-3">
            {/* Create new rule button */}
            <button
              onClick={() => setShowNewRule(!showNewRule)}
              className="w-full py-2 text-xs font-mono text-terminal-accent border border-terminal-accent/30 rounded hover:bg-terminal-accent/10 transition-colors"
            >
              + Create Alert Rule
            </button>

            <AnimatePresence>
              {showNewRule && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="bg-terminal-surface border border-terminal-border rounded p-3 space-y-2">
                    <div className="text-xs font-mono text-terminal-text font-semibold mb-2">New Alert Rule</div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="text-[10px] font-mono text-terminal-subtle block mb-0.5">Name</label>
                        <input type="text" value={newRule.name ?? ''} onChange={(e) => setNewRule(r => ({ ...r, name: e.target.value }))}
                          className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] font-mono text-terminal-subtle block mb-0.5">Type</label>
                        <select value={newRule.type} onChange={(e) => setNewRule(r => ({ ...r, type: e.target.value as AlertRule['type'] }))}
                          className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                        >
                          {['price', 'bh_mass', 'bh_formation', 'pnl', 'equity'].map(t => <option key={t} value={t}>{t.replace('_', ' ')}</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="text-[10px] font-mono text-terminal-subtle block mb-0.5">Symbol (optional)</label>
                        <input type="text" value={newRule.symbol ?? ''} onChange={(e) => setNewRule(r => ({ ...r, symbol: e.target.value.toUpperCase() }))}
                          className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                          placeholder="SPY"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] font-mono text-terminal-subtle block mb-0.5">Condition</label>
                        <select value={newRule.condition} onChange={(e) => setNewRule(r => ({ ...r, condition: e.target.value as AlertRule['condition'] }))}
                          className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                        >
                          {['above', 'below', 'crosses_above', 'crosses_below'].map(c => <option key={c} value={c}>{c.replace('_', ' ')}</option>)}
                        </select>
                      </div>
                      <div>
                        <label className="text-[10px] font-mono text-terminal-subtle block mb-0.5">Threshold</label>
                        <input type="number" value={newRule.threshold ?? 0} onChange={(e) => setNewRule(r => ({ ...r, threshold: parseFloat(e.target.value) }))}
                          className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                          step={0.01}
                        />
                      </div>
                      <div>
                        <label className="text-[10px] font-mono text-terminal-subtle block mb-0.5">Cooldown (min)</label>
                        <input type="number" value={(newRule.cooldownMs ?? 60000) / 60000} onChange={(e) => setNewRule(r => ({ ...r, cooldownMs: parseFloat(e.target.value) * 60000 }))}
                          className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
                          min={1} step={1}
                        />
                      </div>
                    </div>
                    <div className="flex gap-3">
                      <label className="flex items-center gap-1.5 cursor-pointer">
                        <input type="checkbox" checked={newRule.sound ?? true} onChange={(e) => setNewRule(r => ({ ...r, sound: e.target.checked }))} className="accent-terminal-accent w-3 h-3" />
                        <span className="text-[10px] font-mono text-terminal-subtle">Sound</span>
                      </label>
                      <label className="flex items-center gap-1.5 cursor-pointer">
                        <input type="checkbox" checked={newRule.persistent ?? false} onChange={(e) => setNewRule(r => ({ ...r, persistent: e.target.checked }))} className="accent-terminal-accent w-3 h-3" />
                        <span className="text-[10px] font-mono text-terminal-subtle">Persistent</span>
                      </label>
                    </div>
                    <div className="flex gap-2">
                      <button onClick={handleCreateRule} className="flex-1 py-1.5 bg-terminal-accent text-white rounded text-xs font-mono hover:bg-terminal-accent-dim transition-colors">Create</button>
                      <button onClick={() => setShowNewRule(false)} className="px-3 py-1.5 border border-terminal-border rounded text-xs font-mono text-terminal-subtle hover:text-terminal-text transition-colors">Cancel</button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {alertRules.length === 0 && !showNewRule && (
              <div className="text-center py-8 text-terminal-subtle text-sm">No alert rules configured</div>
            )}

            {alertRules.map((rule) => (
              <div key={rule.id} className={`p-3 rounded border ${rule.enabled ? 'border-terminal-border bg-terminal-surface' : 'border-terminal-border/50 bg-terminal-surface/50 opacity-60'}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <button onClick={() => store.toggleAlertRule(rule.id)} className={`w-8 h-4 rounded-full transition-colors relative ${rule.enabled ? 'bg-terminal-accent' : 'bg-terminal-muted'}`}>
                      <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-transform ${rule.enabled ? 'right-0.5' : 'left-0.5'}`} />
                    </button>
                    <span className="font-mono text-xs text-terminal-text">{rule.name}</span>
                    {rule.symbol && <span className="text-[10px] font-mono text-terminal-subtle bg-terminal-muted rounded px-1">{rule.symbol}</span>}
                  </div>
                  <div className="flex items-center gap-2">
                    {rule.triggerCount > 0 && (
                      <span className="text-[10px] font-mono text-terminal-subtle">{rule.triggerCount}× fired</span>
                    )}
                    <button onClick={() => store.deleteAlertRule(rule.id)} className="text-[10px] font-mono text-terminal-bear hover:text-terminal-bear/80 transition-colors">✕</button>
                  </div>
                </div>
                <div className="text-[10px] font-mono text-terminal-subtle mt-0.5">
                  {rule.type} {rule.condition.replace('_', ' ')} {rule.threshold ?? ''} · cooldown {rule.cooldownMs / 60000}min
                  {rule.sound && ' · 🔔'}
                </div>
                {rule.lastTriggered && (
                  <div className="text-[9px] font-mono text-terminal-muted mt-0.5">
                    Last: {format(new Date(rule.lastTriggered), 'MM/dd HH:mm:ss')}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Alert history */}
        {tab === 'history' && (
          <div className="p-3 space-y-1.5">
            {alerts.length === 0 ? (
              <div className="text-center py-12 text-terminal-subtle text-sm">No alert history</div>
            ) : (
              alerts.map((alert) => (
                <div
                  key={alert.id}
                  className={`flex items-center justify-between px-2 py-1.5 rounded text-xs border ${
                    alert.acknowledged ? 'border-terminal-border/50 opacity-60' : 'border-terminal-border'
                  } bg-terminal-surface/50`}
                >
                  <div className="flex items-center gap-2 flex-1">
                    <span className={`font-mono ${alert.type === 'error' ? 'text-terminal-bear' : alert.type === 'success' ? 'text-terminal-bull' : alert.type === 'warning' || alert.type === 'bh' ? 'text-terminal-warning' : 'text-terminal-subtle'}`}>●</span>
                    <span className="font-mono text-terminal-text text-[11px]">{alert.title}</span>
                    {alert.symbol && <span className="text-[10px] font-mono text-terminal-subtle">[{alert.symbol}]</span>}
                    <span className="text-[10px] font-mono text-terminal-muted flex-1 truncate">{alert.message}</span>
                  </div>
                  <span className="text-[9px] font-mono text-terminal-muted flex-shrink-0">{format(new Date(alert.timestamp), 'HH:mm:ss')}</span>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default Alerts
