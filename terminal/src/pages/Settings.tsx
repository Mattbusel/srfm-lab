// ============================================================
// Settings — application settings and configuration
// ============================================================
import React, { useState, useCallback } from 'react'
import { useSettingsStore } from '@/store/settingsStore'
import type { UserSettings } from '@/types'

type SettingsTab = 'general' | 'api' | 'shortcuts' | 'display' | 'notifications'

const KEYBOARD_SHORTCUTS = [
  { key: 'B', description: 'Buy selected symbol', category: 'Trading' },
  { key: 'S', description: 'Sell selected symbol', category: 'Trading' },
  { key: 'Escape', description: 'Close modals / overlays', category: 'Navigation' },
  { key: '1', description: '1-minute interval', category: 'Chart' },
  { key: '2', description: '5-minute interval', category: 'Chart' },
  { key: '3', description: '15-minute interval', category: 'Chart' },
  { key: '4', description: '1-hour interval', category: 'Chart' },
  { key: '5', description: '4-hour interval', category: 'Chart' },
  { key: '6', description: '1-day interval', category: 'Chart' },
]

const SHORTCUT_CATEGORIES = ['Trading', 'Navigation', 'Chart']

const ToggleSwitch: React.FC<{
  value: boolean
  onChange: (v: boolean) => void
  label: string
  description?: string
}> = ({ value, onChange, label, description }) => (
  <div className="flex items-center justify-between py-2.5 border-b border-terminal-border/30 last:border-0">
    <div>
      <div className="text-xs font-mono text-terminal-text">{label}</div>
      {description && <div className="text-[10px] font-mono text-terminal-muted mt-0.5">{description}</div>}
    </div>
    <button
      onClick={() => onChange(!value)}
      className={`w-10 h-5 rounded-full transition-colors relative flex-shrink-0 ml-4 ${value ? 'bg-terminal-accent' : 'bg-terminal-muted'}`}
    >
      <div className={`absolute top-0.5 w-4 h-4 bg-white rounded-full transition-transform shadow-sm ${value ? 'right-0.5' : 'left-0.5'}`} />
    </button>
  </div>
)

const NumberInput: React.FC<{
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  suffix?: string
  description?: string
}> = ({ label, value, onChange, min, max, step = 1, suffix, description }) => (
  <div className="flex items-center justify-between py-2.5 border-b border-terminal-border/30 last:border-0">
    <div>
      <div className="text-xs font-mono text-terminal-text">{label}</div>
      {description && <div className="text-[10px] font-mono text-terminal-muted mt-0.5">{description}</div>}
    </div>
    <div className="flex items-center gap-1.5 ml-4 flex-shrink-0">
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step}
        className="w-20 bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent text-right"
      />
      {suffix && <span className="text-[10px] font-mono text-terminal-muted">{suffix}</span>}
    </div>
  </div>
)

const SelectInput: React.FC<{
  label: string
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
  description?: string
}> = ({ label, value, onChange, options, description }) => (
  <div className="flex items-center justify-between py-2.5 border-b border-terminal-border/30 last:border-0">
    <div>
      <div className="text-xs font-mono text-terminal-text">{label}</div>
      {description && <div className="text-[10px] font-mono text-terminal-muted mt-0.5">{description}</div>}
    </div>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-terminal-bg border border-terminal-border rounded px-2 py-1 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent ml-4 flex-shrink-0"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  </div>
)

const TextInput: React.FC<{
  label: string
  value: string
  onChange: (v: string) => void
  placeholder?: string
  type?: string
  description?: string
}> = ({ label, value, onChange, placeholder, type = 'text', description }) => (
  <div className="py-2.5 border-b border-terminal-border/30 last:border-0">
    <div className="text-xs font-mono text-terminal-text mb-1">{label}</div>
    {description && <div className="text-[10px] font-mono text-terminal-muted mb-1.5">{description}</div>}
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full bg-terminal-bg border border-terminal-border rounded px-2 py-1.5 text-xs font-mono text-terminal-text focus:outline-none focus:border-terminal-accent"
    />
  </div>
)

export const Settings: React.FC = () => {
  const store = useSettingsStore()
  const settings = store.settings
  const [tab, setTab] = useState<SettingsTab>('general')
  const [saved, setSaved] = useState(false)

  const update = useCallback(
    <K extends keyof UserSettings>(key: K, value: UserSettings[K]) => {
      store.updateSettings({ [key]: value } as Partial<UserSettings>)
    },
    [store]
  )

  const handleSave = useCallback(() => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }, [])

  const handleReset = useCallback(() => {
    if (confirm('Reset all settings to defaults?')) {
      store.resetSettings()
    }
  }, [store])

  const TABS: { key: SettingsTab; label: string }[] = [
    { key: 'general', label: 'General' },
    { key: 'api', label: 'API Config' },
    { key: 'display', label: 'Display' },
    { key: 'notifications', label: 'Notifications' },
    { key: 'shortcuts', label: 'Shortcuts' },
  ]

  return (
    <div className="flex h-full bg-terminal-bg">
      {/* Left nav */}
      <div className="w-44 flex-shrink-0 border-r border-terminal-border flex flex-col">
        <div className="px-3 py-2 border-b border-terminal-border">
          <span className="text-terminal-subtle text-xs font-mono uppercase">Settings</span>
        </div>
        <div className="flex-1 p-2 space-y-0.5">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`w-full text-left px-2 py-1.5 rounded text-xs font-mono transition-colors ${
                tab === t.key
                  ? 'bg-terminal-accent/20 text-terminal-accent'
                  : 'text-terminal-subtle hover:text-terminal-text hover:bg-terminal-surface'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="p-3 border-t border-terminal-border space-y-2">
          <button
            onClick={handleSave}
            className={`w-full py-1.5 rounded text-xs font-mono transition-colors ${
              saved
                ? 'bg-terminal-bull/20 text-terminal-bull border border-terminal-bull/30'
                : 'bg-terminal-accent text-white hover:opacity-90'
            }`}
          >
            {saved ? '✓ Saved' : 'Save'}
          </button>
          <button
            onClick={handleReset}
            className="w-full py-1.5 rounded text-xs font-mono border border-terminal-border text-terminal-subtle hover:text-terminal-bear hover:border-terminal-bear/30 transition-colors"
          >
            Reset All
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        {tab === 'general' && (
          <div className="max-w-xl space-y-6">
            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">General Settings</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <SelectInput
                  label="Default Symbol"
                  value={settings.defaultSymbol}
                  onChange={(v) => update('defaultSymbol', v)}
                  options={['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'GOOGL'].map((s) => ({
                    value: s,
                    label: s,
                  }))}
                  description="Symbol shown on startup"
                />
                <SelectInput
                  label="Default Interval"
                  value={settings.chartInterval}
                  onChange={(v) => update('chartInterval', v)}
                  options={[
                    { value: '1m', label: '1 Minute' },
                    { value: '5m', label: '5 Minutes' },
                    { value: '15m', label: '15 Minutes' },
                    { value: '1h', label: '1 Hour' },
                    { value: '4h', label: '4 Hours' },
                    { value: '1d', label: '1 Day' },
                  ]}
                  description="Default chart timeframe"
                />
                <ToggleSwitch
                  label="Enable Keyboard Shortcuts"
                  value={settings.hotkeysEnabled}
                  onChange={(v) => update('hotkeysEnabled', v)}
                  description="Enable global keyboard shortcuts (B/S for orders, etc.)"
                />
                <ToggleSwitch
                  label="Confirm Orders"
                  value={settings.confirmOrders}
                  onChange={(v) => update('confirmOrders', v)}
                  description="Show confirmation dialog before submitting orders"
                />
              </div>
            </div>

            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">Trading Defaults</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <SelectInput
                  label="Default Order Type"
                  value={settings.defaultOrderType}
                  onChange={(v) => update('defaultOrderType', v)}
                  options={[
                    { value: 'market', label: 'Market' },
                    { value: 'limit', label: 'Limit' },
                    { value: 'stop', label: 'Stop' },
                    { value: 'stop_limit', label: 'Stop Limit' },
                  ]}
                />
                <SelectInput
                  label="Default Time-in-Force"
                  value={settings.defaultTimeInForce}
                  onChange={(v) => update('defaultTimeInForce', v)}
                  options={[
                    { value: 'day', label: 'Day' },
                    { value: 'gtc', label: 'GTC' },
                    { value: 'ioc', label: 'IOC' },
                    { value: 'fok', label: 'FOK' },
                  ]}
                />
                <ToggleSwitch
                  label="Paper Trading"
                  value={settings.alpacaPaper}
                  onChange={(v) => update('alpacaPaper', v)}
                  description="Use Alpaca paper trading account (simulated orders)"
                />
                <NumberInput
                  label="Daily P&L Target"
                  value={settings.dailyPnlTarget}
                  onChange={(v) => update('dailyPnlTarget', v)}
                  min={0}
                  step={100}
                  suffix="USD"
                  description="Daily profit target for progress tracking"
                />
                <NumberInput
                  label="Max Position Size"
                  value={Math.round(settings.maxPositionSize * 100)}
                  onChange={(v) => update('maxPositionSize', v / 100)}
                  min={1}
                  max={100}
                  step={1}
                  suffix="%"
                  description="Maximum single position as percentage of equity"
                />
              </div>
            </div>
          </div>
        )}

        {tab === 'api' && (
          <div className="max-w-xl space-y-6">
            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">Alpaca API</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <TextInput
                  label="API Key ID"
                  value={settings.alpacaApiKey}
                  onChange={(v) => update('alpacaApiKey', v)}
                  placeholder="PKXXXXXXXXXXXXXXXXXX"
                  description="Found in Alpaca dashboard under API Keys"
                />
                <TextInput
                  label="API Secret Key"
                  value={settings.alpacaSecretKey}
                  onChange={(v) => update('alpacaSecretKey', v)}
                  type="password"
                  placeholder="Enter secret key"
                  description="Keep this secret — never share"
                />
                <ToggleSwitch
                  label="Paper Trading"
                  value={settings.alpacaPaper}
                  onChange={(v) => update('alpacaPaper', v)}
                  description="Use paper.alpaca.markets instead of api.alpaca.markets"
                />
                <div className="pt-2">
                  <button
                    onClick={() => alert('Connection test — check console for status')}
                    className="text-[10px] font-mono px-3 py-1.5 rounded border border-terminal-accent/30 text-terminal-accent hover:bg-terminal-accent/10 transition-colors"
                  >
                    Test Connection
                  </button>
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">Spacetime Analytics Server</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <TextInput
                  label="REST API Endpoint"
                  value={settings.apiUrl}
                  onChange={(v) => update('apiUrl', v)}
                  placeholder="http://localhost:8080"
                  description="URL of the Spacetime analytics server"
                />
                <TextInput
                  label="WebSocket Endpoint"
                  value={settings.wsUrl}
                  onChange={(v) => update('wsUrl', v)}
                  placeholder="ws://localhost:8080/ws"
                  description="WebSocket URL for live BH state streaming"
                />
              </div>
            </div>

            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">Market Data Gateway</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <TextInput
                  label="Gateway REST Endpoint"
                  value={settings.gatewayUrl}
                  onChange={(v) => update('gatewayUrl', v)}
                  placeholder="http://localhost:9090"
                  description="URL of the Go market data gateway"
                />
                <TextInput
                  label="Gateway WebSocket"
                  value={settings.gatewayWsUrl}
                  onChange={(v) => update('gatewayWsUrl', v)}
                  placeholder="ws://localhost:9090/ws"
                />
              </div>
            </div>

            <div className="bg-terminal-warning/5 border border-terminal-warning/20 rounded p-3">
              <div className="text-[10px] font-mono text-terminal-warning font-semibold mb-1">Security Notice</div>
              <div className="text-[10px] font-mono text-terminal-muted leading-relaxed">
                API keys are stored in localStorage. Do not use this terminal on shared computers.
                For production deployments, set keys via environment variables and disable localStorage persistence.
              </div>
            </div>
          </div>
        )}

        {tab === 'display' && (
          <div className="max-w-xl space-y-6">
            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">Chart Overlays</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <ToggleSwitch
                  label="Show EMA 20"
                  value={settings.showEMA20}
                  onChange={(v) => update('showEMA20', v)}
                />
                <ToggleSwitch
                  label="Show EMA 50"
                  value={settings.showEMA50}
                  onChange={(v) => update('showEMA50', v)}
                />
                <ToggleSwitch
                  label="Show EMA 200"
                  value={settings.showEMA200}
                  onChange={(v) => update('showEMA200', v)}
                />
                <ToggleSwitch
                  label="Show Volume"
                  value={settings.showVolume}
                  onChange={(v) => update('showVolume', v)}
                />
                <ToggleSwitch
                  label="Show Volume Profile"
                  value={settings.showVolumeProfile}
                  onChange={(v) => update('showVolumeProfile', v)}
                />
                <ToggleSwitch
                  label="Show BH Mass Overlay"
                  value={settings.showBHOverlay}
                  onChange={(v) => update('showBHOverlay', v)}
                  description="Overlay BH mass scalar line on price chart"
                />
                <ToggleSwitch
                  label="Show Regime Colors"
                  value={settings.showRegimeColors}
                  onChange={(v) => update('showRegimeColors', v)}
                  description="Color candles by BH regime (BULL/BEAR/SIDEWAYS/HIGH_VOL)"
                />
                <ToggleSwitch
                  label="Show Position Floor Line"
                  value={settings.showPosFloorLine}
                  onChange={(v) => update('showPosFloorLine', v)}
                  description="Draw horizontal line at average cost basis for open positions"
                />
              </div>
            </div>

            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">Order Book</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <NumberInput
                  label="Book Depth Levels"
                  value={settings.orderBookLevels}
                  onChange={(v) => update('orderBookLevels', v)}
                  min={5}
                  max={50}
                  step={5}
                  description="Number of price levels to display in the order book"
                />
              </div>
            </div>

            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">Watchlist Defaults</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <SelectInput
                  label="Sort Field"
                  value={settings.watchlistSortField}
                  onChange={(v) => update('watchlistSortField', v)}
                  options={[
                    { value: 'symbol', label: 'Symbol' },
                    { value: 'price', label: 'Price' },
                    { value: 'change', label: 'Change %' },
                    { value: 'volume', label: 'Volume' },
                  ]}
                />
                <SelectInput
                  label="Sort Direction"
                  value={settings.watchlistSortDir}
                  onChange={(v) => update('watchlistSortDir', v as 'asc' | 'desc')}
                  options={[
                    { value: 'asc', label: 'Ascending' },
                    { value: 'desc', label: 'Descending' },
                  ]}
                />
              </div>
            </div>

            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">Theme</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <div className="flex items-center justify-between py-2.5">
                  <span className="text-xs font-mono text-terminal-text">Color Theme</span>
                  <div className="flex gap-2 ml-4">
                    {(['dark', 'light'] as const).map((t) => (
                      <button
                        key={t}
                        onClick={() => update('theme', t)}
                        className={`px-3 py-1 rounded text-xs font-mono border transition-colors capitalize ${
                          settings.theme === t
                            ? 'bg-terminal-accent/20 text-terminal-accent border-terminal-accent/30'
                            : 'text-terminal-subtle border-terminal-border hover:text-terminal-text'
                        }`}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="py-2.5">
                  <div className="text-[10px] font-mono text-terminal-muted">
                    Light theme is planned for a future update. Currently only dark theme is supported.
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {tab === 'notifications' && (
          <div className="max-w-xl space-y-6">
            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">Sound Alerts</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <ToggleSwitch
                  label="Enable Sound Alerts"
                  value={store.isSoundEnabled}
                  onChange={(v) => store.setSoundEnabled(v)}
                  description="Play audio beeps for triggered alerts"
                />
                <ToggleSwitch
                  label="Alert Sound Setting"
                  value={settings.alertSoundEnabled}
                  onChange={(v) => update('alertSoundEnabled', v)}
                  description="Persist sound preference across sessions"
                />
                <NumberInput
                  label="Alert Volume"
                  value={Math.round(settings.alertVolume * 100)}
                  onChange={(v) => update('alertVolume', Math.min(1, Math.max(0, v / 100)))}
                  min={0}
                  max={100}
                  step={5}
                  suffix="%"
                />
                <div className="py-2.5">
                  <button
                    onClick={() => store.playSoundAlert('info')}
                    className="text-[10px] font-mono px-3 py-1.5 rounded border border-terminal-border text-terminal-subtle hover:text-terminal-text hover:border-terminal-accent transition-colors mr-2"
                  >
                    Test Info Sound
                  </button>
                  <button
                    onClick={() => store.playSoundAlert('warning')}
                    className="text-[10px] font-mono px-3 py-1.5 rounded border border-terminal-warning/30 text-terminal-warning hover:bg-terminal-warning/10 transition-colors mr-2"
                  >
                    Test Warning
                  </button>
                  <button
                    onClick={() => store.playSoundAlert('critical')}
                    className="text-[10px] font-mono px-3 py-1.5 rounded border border-terminal-bear/30 text-terminal-bear hover:bg-terminal-bear/10 transition-colors"
                  >
                    Test Critical
                  </button>
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-3">Alert Management</h2>
              <div className="bg-terminal-surface border border-terminal-border rounded p-3">
                <div className="flex items-center justify-between py-2.5 border-b border-terminal-border/30">
                  <div>
                    <div className="text-xs font-mono text-terminal-text">Active Alert Rules</div>
                    <div className="text-[10px] font-mono text-terminal-muted mt-0.5">Configure rules in the Alerts page</div>
                  </div>
                  <span className="text-xs font-mono text-terminal-accent ml-4">
                    {store.alertRules.filter(r => r.enabled).length} / {store.alertRules.length} enabled
                  </span>
                </div>
                <div className="flex items-center justify-between py-2.5 border-b border-terminal-border/30">
                  <div>
                    <div className="text-xs font-mono text-terminal-text">Pending Alerts</div>
                    <div className="text-[10px] font-mono text-terminal-muted mt-0.5">Unacknowledged alerts in queue</div>
                  </div>
                  <span className={`text-xs font-mono ml-4 ${store.alerts.filter(a => !a.acknowledged).length > 0 ? 'text-terminal-bear' : 'text-terminal-subtle'}`}>
                    {store.alerts.filter(a => !a.acknowledged).length}
                  </span>
                </div>
                <div className="py-2.5 flex gap-2">
                  <button
                    onClick={() => store.clearAlerts()}
                    className="text-[10px] font-mono px-3 py-1.5 rounded border border-terminal-border text-terminal-subtle hover:text-terminal-bear hover:border-terminal-bear/30 transition-colors"
                  >
                    Clear All Alerts
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {tab === 'shortcuts' && (
          <div className="max-w-xl space-y-6">
            <div>
              <h2 className="text-sm font-mono font-semibold text-terminal-text mb-1">Keyboard Shortcuts</h2>
              <p className="text-[10px] font-mono text-terminal-muted mb-4">
                Shortcuts are active when not typing in a form field. Enable/disable all shortcuts in General settings.
              </p>

              {SHORTCUT_CATEGORIES.map((cat) => {
                const shortcuts = KEYBOARD_SHORTCUTS.filter((s) => s.category === cat)
                return (
                  <div key={cat} className="mb-5">
                    <div className="text-[10px] font-mono text-terminal-accent uppercase mb-2 font-semibold">{cat}</div>
                    <div className="bg-terminal-surface border border-terminal-border rounded overflow-hidden">
                      {shortcuts.map((s, i) => (
                        <div
                          key={s.key}
                          className={`flex items-center justify-between px-3 py-2.5 ${
                            i !== shortcuts.length - 1 ? 'border-b border-terminal-border/30' : ''
                          }`}
                        >
                          <span className="text-xs font-mono text-terminal-text">{s.description}</span>
                          <kbd className="text-[10px] font-mono bg-terminal-bg border border-terminal-border/80 rounded px-2 py-0.5 text-terminal-subtle shadow-sm">
                            {s.key}
                          </kbd>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}

              <div>
                <div className="text-[10px] font-mono text-terminal-accent uppercase mb-2 font-semibold">Strategy Canvas</div>
                <div className="bg-terminal-surface border border-terminal-border rounded overflow-hidden">
                  {[
                    ['Mouse Wheel', 'Zoom in / out'],
                    ['Alt + Drag', 'Pan canvas'],
                    ['Middle Click Drag', 'Pan canvas'],
                    ['Click Node', 'Select and inspect node'],
                    ['Drag Node', 'Move node (20px grid snap)'],
                    ['Click Output Handle', 'Start connection'],
                    ['Click Input Handle', 'Complete connection'],
                    ['Drag from Palette', 'Drop new node onto canvas'],
                  ].map(([k, v], i, arr) => (
                    <div
                      key={k}
                      className={`flex items-center justify-between px-3 py-2.5 ${
                        i !== arr.length - 1 ? 'border-b border-terminal-border/30' : ''
                      }`}
                    >
                      <span className="text-xs font-mono text-terminal-text">{v}</span>
                      <kbd className="text-[10px] font-mono bg-terminal-bg border border-terminal-border/80 rounded px-2 py-0.5 text-terminal-subtle shadow-sm">
                        {k}
                      </kbd>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Settings
