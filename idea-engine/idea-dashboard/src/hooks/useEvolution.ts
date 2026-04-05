import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  fetchEvolutionStats,
  fetchGenomes,
  fetchMutationFrequencies,
  triggerEvolution,
} from '../api/client'
import type { Island } from '../types'

const EVOLUTION_POLL_INTERVAL = 30_000 // 30s

export function useEvolutionStats() {
  return useQuery({
    queryKey: ['evolution', 'stats'],
    queryFn: fetchEvolutionStats,
    refetchInterval: EVOLUTION_POLL_INTERVAL,
    staleTime: 20_000,
    select: (data) => {
      // Group by island for easy chart consumption
      const byIsland: Record<Island, typeof data> = {
        BULL: [],
        BEAR: [],
        NEUTRAL: [],
      }
      for (const stat of data) {
        byIsland[stat.island].push(stat)
      }
      // Sort each island by generation
      for (const island of Object.keys(byIsland) as Island[]) {
        byIsland[island].sort((a, b) => a.generation - b.generation)
      }
      return { raw: data, byIsland }
    },
  })
}

export function useGenomes(island?: Island) {
  return useQuery({
    queryKey: ['genomes', island ?? 'all'],
    queryFn: () => fetchGenomes(island),
    refetchInterval: EVOLUTION_POLL_INTERVAL,
    staleTime: 20_000,
  })
}

export function useMutationFrequencies() {
  return useQuery({
    queryKey: ['evolution', 'mutations'],
    queryFn: fetchMutationFrequencies,
    refetchInterval: 60_000,
    staleTime: 50_000,
  })
}

export function useTriggerEvolution() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (island: Island) => triggerEvolution(island),
    onSuccess: (_data, island) => {
      // Invalidate genomes and stats for the triggered island
      void queryClient.invalidateQueries({ queryKey: ['genomes', island] })
      void queryClient.invalidateQueries({ queryKey: ['evolution', 'stats'] })
    },
  })
}

export function useEvolutionSummary() {
  const { data: stats, isLoading, error } = useEvolutionStats()

  const summary = stats
    ? {
        maxGeneration: Math.max(
          ...stats.raw.map((s) => s.generation)
        ),
        globalBestFitness: Math.max(...stats.raw.map((s) => s.bestFitness)),
        avgDiversity:
          stats.raw.reduce((acc, s) => acc + s.diversityIndex, 0) /
          stats.raw.length,
        islandSummaries: (['BULL', 'BEAR', 'NEUTRAL'] as Island[]).map(
          (island) => {
            const islandStats = stats.byIsland[island]
            const latest = islandStats[islandStats.length - 1]
            return {
              island,
              generation: latest?.generation ?? 0,
              bestFitness: latest?.bestFitness ?? 0,
              meanFitness: latest?.meanFitness ?? 0,
              diversityIndex: latest?.diversityIndex ?? 0,
            }
          }
        ),
      }
    : null

  return { summary, isLoading, error }
}
