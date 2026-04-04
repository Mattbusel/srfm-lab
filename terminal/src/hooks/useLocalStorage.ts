// ============================================================
// useLocalStorage — persistent state helper
// ============================================================
import { useState, useEffect, useCallback, useRef } from 'react'

function tryParse<T>(value: string, fallback: T): T {
  try {
    return JSON.parse(value) as T
  } catch {
    return fallback
  }
}

export function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T | ((prev: T) => T)) => void, () => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key)
      return item !== null ? tryParse(item, initialValue) : initialValue
    } catch {
      return initialValue
    }
  })

  // Keep a ref to detect key changes
  const keyRef = useRef(key)

  useEffect(() => {
    // Re-read from storage when key changes
    if (keyRef.current !== key) {
      keyRef.current = key
      try {
        const item = window.localStorage.getItem(key)
        setStoredValue(item !== null ? tryParse(item, initialValue) : initialValue)
      } catch {
        setStoredValue(initialValue)
      }
    }
  }, [key, initialValue])

  // Listen for storage events (cross-tab sync)
  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.key === key) {
        try {
          const newValue = event.newValue !== null
            ? tryParse(event.newValue, initialValue)
            : initialValue
          setStoredValue(newValue)
        } catch {
          // ignore
        }
      }
    }

    window.addEventListener('storage', handleStorage)
    return () => window.removeEventListener('storage', handleStorage)
  }, [key, initialValue])

  const setValue = useCallback((value: T | ((prev: T) => T)) => {
    try {
      setStoredValue((prev) => {
        const newValue = typeof value === 'function' ? (value as (p: T) => T)(prev) : value
        window.localStorage.setItem(key, JSON.stringify(newValue))
        return newValue
      })
    } catch (err) {
      console.error(`useLocalStorage: failed to set key "${key}"`, err)
    }
  }, [key])

  const removeValue = useCallback(() => {
    try {
      window.localStorage.removeItem(key)
      setStoredValue(initialValue)
    } catch (err) {
      console.error(`useLocalStorage: failed to remove key "${key}"`, err)
    }
  }, [key, initialValue])

  return [storedValue, setValue, removeValue]
}

// Simpler version that just syncs with localStorage without React state overhead
export function useLocalStorageValue<T>(key: string, defaultValue: T): T {
  try {
    const item = window.localStorage.getItem(key)
    return item !== null ? (JSON.parse(item) as T) : defaultValue
  } catch {
    return defaultValue
  }
}

export function setLocalStorage<T>(key: string, value: T): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value))
  } catch (err) {
    console.error(`setLocalStorage failed for key "${key}"`, err)
  }
}

export function getLocalStorage<T>(key: string, defaultValue: T): T {
  try {
    const item = window.localStorage.getItem(key)
    return item !== null ? (JSON.parse(item) as T) : defaultValue
  } catch {
    return defaultValue
  }
}
