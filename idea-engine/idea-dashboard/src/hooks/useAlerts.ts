import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { fetchAlerts, acknowledgeAlert } from '../api/client'
import type { AlertSeverity } from '../types'

const ALERTS_POLL_INTERVAL = 15_000 // 15s

export function useAlerts() {
  return useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    refetchInterval: ALERTS_POLL_INTERVAL,
    staleTime: 10_000,
  })
}

export function useUnacknowledgedAlerts() {
  return useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    refetchInterval: ALERTS_POLL_INTERVAL,
    staleTime: 10_000,
    select: (data) => data.filter((a) => !a.acknowledged),
  })
}

export function useCriticalAlerts() {
  return useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    refetchInterval: ALERTS_POLL_INTERVAL,
    staleTime: 10_000,
    select: (data) =>
      data.filter((a) => a.severity === 'critical' && !a.acknowledged),
  })
}

export function useAlertsBySeverity(severity: AlertSeverity) {
  return useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    select: (data) => data.filter((a) => a.severity === severity),
  })
}

export function useAcknowledgeAlert() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: number) => acknowledgeAlert(id),
    onMutate: async (id) => {
      // Optimistic update
      await queryClient.cancelQueries({ queryKey: ['alerts'] })
      const prev = queryClient.getQueryData(['alerts'])
      queryClient.setQueryData(['alerts'], (old: ReturnType<typeof fetchAlerts> | undefined) => {
        if (!old) return old
        // Handle both promise and array
        return old
      })
      // Direct optimistic patch
      queryClient.setQueryData(
        ['alerts'],
        (old: Awaited<ReturnType<typeof fetchAlerts>> | undefined) => {
          if (!old) return old
          return old.map((a) =>
            a.id === id ? { ...a, acknowledged: true } : a
          )
        }
      )
      return { prev }
    },
    onError: (_err, _id, context) => {
      if (context?.prev) {
        queryClient.setQueryData(['alerts'], context.prev)
      }
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ['alerts'] })
    },
  })
}

export function useAlertCounts() {
  return useQuery({
    queryKey: ['alerts'],
    queryFn: fetchAlerts,
    select: (data) => ({
      total: data.length,
      unacknowledged: data.filter((a) => !a.acknowledged).length,
      critical: data.filter((a) => a.severity === 'critical' && !a.acknowledged).length,
      warning: data.filter((a) => a.severity === 'warning' && !a.acknowledged).length,
      info: data.filter((a) => a.severity === 'info' && !a.acknowledged).length,
    }),
  })
}
