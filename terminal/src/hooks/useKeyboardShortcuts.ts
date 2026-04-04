// ============================================================
// useKeyboardShortcuts — global hotkey registration
// ============================================================
import { useEffect, useRef, useCallback } from 'react'
import { useSettingsStore } from '@/store/settingsStore'

export interface Shortcut {
  key: string              // 'b', 'Escape', 'F1', etc.
  ctrl?: boolean
  alt?: boolean
  shift?: boolean
  meta?: boolean
  description: string
  action: () => void
  when?: () => boolean     // optional condition
  preventDefault?: boolean
}

type ShortcutMap = Record<string, Shortcut>

const normalizeKey = (e: KeyboardEvent): string => {
  const parts: string[] = []
  if (e.ctrlKey) parts.push('ctrl')
  if (e.altKey) parts.push('alt')
  if (e.shiftKey) parts.push('shift')
  if (e.metaKey) parts.push('meta')
  parts.push(e.key.toLowerCase())
  return parts.join('+')
}

const shortcutKey = (s: Shortcut): string => {
  const parts: string[] = []
  if (s.ctrl) parts.push('ctrl')
  if (s.alt) parts.push('alt')
  if (s.shift) parts.push('shift')
  if (s.meta) parts.push('meta')
  parts.push(s.key.toLowerCase())
  return parts.join('+')
}

// Global registry (singleton pattern)
const globalShortcuts: Map<string, Shortcut> = new Map()
let globalListenerAttached = false

function attachGlobalListener() {
  if (globalListenerAttached) return
  globalListenerAttached = true

  document.addEventListener('keydown', (e) => {
    // Don't fire when typing in inputs
    const target = e.target as HTMLElement
    const isTyping =
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.tagName === 'SELECT' ||
      target.isContentEditable

    if (isTyping) return

    const key = normalizeKey(e)
    const shortcut = globalShortcuts.get(key)
    if (!shortcut) return

    const hotkeysEnabled = useSettingsStore.getState().settings.hotkeysEnabled
    if (!hotkeysEnabled) return

    if (shortcut.when && !shortcut.when()) return

    if (shortcut.preventDefault !== false) {
      e.preventDefault()
    }

    shortcut.action()
  })
}

export function useKeyboardShortcuts(shortcuts: Shortcut[], deps: unknown[] = []) {
  const shortcutsRef = useRef(shortcuts)
  shortcutsRef.current = shortcuts

  const registeredKeys = useRef<string[]>([])

  useEffect(() => {
    attachGlobalListener()

    // Remove old shortcuts
    for (const key of registeredKeys.current) {
      globalShortcuts.delete(key)
    }
    registeredKeys.current = []

    // Register new shortcuts
    for (const shortcut of shortcutsRef.current) {
      const key = shortcutKey(shortcut)
      globalShortcuts.set(key, shortcut)
      registeredKeys.current.push(key)
    }

    return () => {
      for (const key of registeredKeys.current) {
        globalShortcuts.delete(key)
      }
      registeredKeys.current = []
    }
  }, deps)  // eslint-disable-line react-hooks/exhaustive-deps
}

export function useGlobalShortcut(
  key: string,
  options: Omit<Shortcut, 'key' | 'action'>,
  action: () => void
) {
  const shortcut: Shortcut = { key, ...options, action }
  useKeyboardShortcuts([shortcut], [key])
}

// Get all registered shortcuts for the help overlay
export function getAllShortcuts(): Shortcut[] {
  return Array.from(globalShortcuts.values())
}

// Terminal-specific shortcuts
export const TERMINAL_SHORTCUTS: Omit<Shortcut, 'action'>[] = [
  { key: 'b', description: 'Buy selected symbol' },
  { key: 's', description: 'Sell selected symbol' },
  { key: 'escape', description: 'Cancel / Close modal' },
  { key: '1', description: 'Chart interval: 1m' },
  { key: '2', description: 'Chart interval: 5m' },
  { key: '3', description: 'Chart interval: 15m' },
  { key: '4', description: 'Chart interval: 1h' },
  { key: '5', description: 'Chart interval: 4h' },
  { key: '6', description: 'Chart interval: 1d' },
  { key: 'o', ctrl: true, description: 'Open order entry' },
  { key: 'h', description: 'Toggle chart crosshair' },
  { key: 'f', description: 'Focus symbol search' },
  { key: '?', shift: true, description: 'Show keyboard shortcuts' },
  { key: 'r', ctrl: true, description: 'Refresh account data' },
  { key: 'p', ctrl: true, description: 'Go to portfolio' },
  { key: 't', ctrl: true, description: 'Go to terminal' },
  { key: 'm', ctrl: true, description: 'Go to scanner' },
]
