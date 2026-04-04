// ============================================================
// STRATEGY STORE — Zustand + Immer
// ============================================================
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { persist } from 'zustand/middleware'
import type {
  StrategyGraph,
  StrategyNode,
  StrategyEdge,
  BacktestConfig,
  BacktestResult,
  StrategyFactorAnalysis,
} from '@/types'
import { spacetimeApi } from '@/services/api'

const generateId = () => Math.random().toString(36).slice(2, 11)

interface StrategyState {
  graphs: StrategyGraph[]
  activeGraphId: string | null
  backtestResults: Record<string, BacktestResult[]>   // graphId -> results
  activeBacktestId: Record<string, string | null>     // graphId -> active result id
  factorAnalyses: Record<string, StrategyFactorAnalysis>

  isRunningBacktest: boolean
  backtestProgress: number
  loadingGraphId: string | null
  error: string | null
}

interface StrategyActions {
  // Graph management
  createGraph(name: string, description?: string): string
  deleteGraph(graphId: string): void
  duplicateGraph(graphId: string): string
  setActiveGraph(graphId: string | null): void
  updateGraphMetadata(graphId: string, meta: Partial<StrategyGraph['metadata']>): void

  // Node operations
  addNode(graphId: string, node: StrategyNode): void
  removeNode(graphId: string, nodeId: string): void
  updateNode(graphId: string, nodeId: string, updates: Partial<StrategyNode>): void
  updateNodeParams(graphId: string, nodeId: string, params: Record<string, number | string | boolean>): void
  moveNode(graphId: string, nodeId: string, position: { x: number; y: number }): void
  toggleNode(graphId: string, nodeId: string): void

  // Edge operations
  addEdge(graphId: string, edge: StrategyEdge): void
  removeEdge(graphId: string, edgeId: string): void
  removeEdgesForNode(graphId: string, nodeId: string): void

  // Viewport
  setViewport(graphId: string, viewport: { x: number; y: number; zoom: number }): void

  // Backtest
  runBacktest(graphId: string, config: BacktestConfig): Promise<void>
  cancelBacktest(): void
  setActiveBacktest(graphId: string, resultId: string | null): void
  deleteBacktestResult(graphId: string, resultId: string): void

  // Import/Export
  exportStrategy(graphId: string): string
  importStrategy(json: string): string

  // Factor Analysis
  runFactorAnalysis(graphId: string, symbol: string, dateRange: { start: string; end: string }): Promise<void>

  clearError(): void
}

type StrategyStore = StrategyState & StrategyActions

const EMPTY_GRAPH = (): StrategyGraph => ({
  id: generateId(),
  nodes: [],
  edges: [],
  metadata: {
    name: 'Untitled Strategy',
    description: '',
    version: '1.0.0',
    createdAt: Date.now(),
    updatedAt: Date.now(),
  },
  viewport: { x: 0, y: 0, zoom: 1 },
})

export const useStrategyStore = create<StrategyStore>()(
  persist(
    immer((set, get) => ({
      // ---- Initial State ----
      graphs: [],
      activeGraphId: null,
      backtestResults: {},
      activeBacktestId: {},
      factorAnalyses: {},
      isRunningBacktest: false,
      backtestProgress: 0,
      loadingGraphId: null,
      error: null,

      // ---- Graph Management ----
      createGraph(name: string, description = '') {
        const graph = EMPTY_GRAPH()
        graph.metadata.name = name
        graph.metadata.description = description
        set((state) => {
          state.graphs.push(graph)
          state.activeGraphId = graph.id
        })
        return graph.id
      },

      deleteGraph(graphId: string) {
        set((state) => {
          const idx = state.graphs.findIndex((g) => g.id === graphId)
          if (idx !== -1) {
            state.graphs.splice(idx, 1)
          }
          if (state.activeGraphId === graphId) {
            state.activeGraphId = state.graphs[0]?.id ?? null
          }
          delete state.backtestResults[graphId]
          delete state.activeBacktestId[graphId]
        })
      },

      duplicateGraph(graphId: string) {
        const orig = get().graphs.find((g) => g.id === graphId)
        if (!orig) return graphId

        const copy: StrategyGraph = JSON.parse(JSON.stringify(orig))
        copy.id = generateId()
        copy.metadata.name = orig.metadata.name + ' (Copy)'
        copy.metadata.createdAt = Date.now()
        copy.metadata.updatedAt = Date.now()

        set((state) => {
          state.graphs.push(copy)
          state.activeGraphId = copy.id
        })

        return copy.id
      },

      setActiveGraph(graphId: string | null) {
        set((state) => {
          state.activeGraphId = graphId
        })
      },

      updateGraphMetadata(graphId: string, meta: Partial<StrategyGraph['metadata']>) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            Object.assign(graph.metadata, meta)
            graph.metadata.updatedAt = Date.now()
          }
        })
      },

      // ---- Node Operations ----
      addNode(graphId: string, node: StrategyNode) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            graph.nodes.push(node)
            graph.metadata.updatedAt = Date.now()
          }
        })
      },

      removeNode(graphId: string, nodeId: string) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            const idx = graph.nodes.findIndex((n) => n.id === nodeId)
            if (idx !== -1) graph.nodes.splice(idx, 1)
            // Remove connected edges
            graph.edges = graph.edges.filter(
              (e) => e.source !== nodeId && e.target !== nodeId
            )
            graph.metadata.updatedAt = Date.now()
          }
        })
      },

      updateNode(graphId: string, nodeId: string, updates: Partial<StrategyNode>) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            const node = graph.nodes.find((n) => n.id === nodeId)
            if (node) {
              Object.assign(node, updates)
              graph.metadata.updatedAt = Date.now()
            }
          }
        })
      },

      updateNodeParams(graphId: string, nodeId: string, params: Record<string, number | string | boolean>) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            const node = graph.nodes.find((n) => n.id === nodeId)
            if (node) {
              Object.assign(node.params, params)
              graph.metadata.updatedAt = Date.now()
            }
          }
        })
      },

      moveNode(graphId: string, nodeId: string, position: { x: number; y: number }) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            const node = graph.nodes.find((n) => n.id === nodeId)
            if (node) {
              node.position = position
            }
          }
        })
      },

      toggleNode(graphId: string, nodeId: string) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            const node = graph.nodes.find((n) => n.id === nodeId)
            if (node) {
              node.disabled = !node.disabled
            }
          }
        })
      },

      // ---- Edge Operations ----
      addEdge(graphId: string, edge: StrategyEdge) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            // Prevent duplicate edges on same target handle
            graph.edges = graph.edges.filter(
              (e) => !(e.target === edge.target && e.targetHandle === edge.targetHandle)
            )
            graph.edges.push(edge)
            graph.metadata.updatedAt = Date.now()
          }
        })
      },

      removeEdge(graphId: string, edgeId: string) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            const idx = graph.edges.findIndex((e) => e.id === edgeId)
            if (idx !== -1) graph.edges.splice(idx, 1)
            graph.metadata.updatedAt = Date.now()
          }
        })
      },

      removeEdgesForNode(graphId: string, nodeId: string) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            graph.edges = graph.edges.filter(
              (e) => e.source !== nodeId && e.target !== nodeId
            )
          }
        })
      },

      setViewport(graphId: string, viewport: { x: number; y: number; zoom: number }) {
        set((state) => {
          const graph = state.graphs.find((g) => g.id === graphId)
          if (graph) {
            graph.viewport = viewport
          }
        })
      },

      // ---- Backtest ----
      async runBacktest(graphId: string, config: BacktestConfig) {
        const graph = get().graphs.find((g) => g.id === graphId)
        if (!graph) return

        set((state) => {
          state.isRunningBacktest = true
          state.backtestProgress = 0
          state.error = null
        })

        try {
          const result = await spacetimeApi.runBacktest(graph, config, (progress) => {
            set((state) => { state.backtestProgress = progress })
          })

          set((state) => {
            if (!state.backtestResults[graphId]) {
              state.backtestResults[graphId] = []
            }
            state.backtestResults[graphId].unshift(result)
            state.activeBacktestId[graphId] = result.id
            state.isRunningBacktest = false
            state.backtestProgress = 100
          })
        } catch (err) {
          set((state) => {
            state.isRunningBacktest = false
            state.backtestProgress = 0
            state.error = err instanceof Error ? err.message : 'Backtest failed'
          })
        }
      },

      cancelBacktest() {
        set((state) => {
          state.isRunningBacktest = false
          state.backtestProgress = 0
        })
      },

      setActiveBacktest(graphId: string, resultId: string | null) {
        set((state) => {
          state.activeBacktestId[graphId] = resultId
        })
      },

      deleteBacktestResult(graphId: string, resultId: string) {
        set((state) => {
          const results = state.backtestResults[graphId]
          if (results) {
            const idx = results.findIndex((r) => r.id === resultId)
            if (idx !== -1) results.splice(idx, 1)
          }
          if (state.activeBacktestId[graphId] === resultId) {
            state.activeBacktestId[graphId] =
              state.backtestResults[graphId]?.[0]?.id ?? null
          }
        })
      },

      // ---- Import/Export ----
      exportStrategy(graphId: string) {
        const graph = get().graphs.find((g) => g.id === graphId)
        if (!graph) return ''
        return JSON.stringify(graph, null, 2)
      },

      importStrategy(json: string) {
        try {
          const graph: StrategyGraph = JSON.parse(json)
          const newId = generateId()
          graph.id = newId
          graph.metadata.name = graph.metadata.name + ' (Imported)'
          graph.metadata.createdAt = Date.now()
          graph.metadata.updatedAt = Date.now()

          set((state) => {
            state.graphs.push(graph)
            state.activeGraphId = newId
          })

          return newId
        } catch {
          set((state) => {
            state.error = 'Failed to import strategy: invalid JSON'
          })
          return ''
        }
      },

      // ---- Factor Analysis ----
      async runFactorAnalysis(graphId: string, symbol: string, dateRange: { start: string; end: string }) {
        try {
          const graph = get().graphs.find((g) => g.id === graphId)
          if (!graph) return

          const analysis = await spacetimeApi.runFactorAnalysis(graph, symbol, dateRange)
          set((state) => {
            state.factorAnalyses[`${graphId}-${symbol}`] = analysis
          })
        } catch (err) {
          set((state) => {
            state.error = err instanceof Error ? err.message : 'Factor analysis failed'
          })
        }
      },

      clearError() {
        set((state) => {
          state.error = null
        })
      },
    })),
    {
      name: 'strategy-store',
      partialize: (state) => ({
        graphs: state.graphs,
        activeGraphId: state.activeGraphId,
        backtestResults: state.backtestResults,
        activeBacktestId: state.activeBacktestId,
      }),
    }
  )
)

// ---- Selectors ----
export const selectActiveGraph = (state: StrategyStore) =>
  state.graphs.find((g) => g.id === state.activeGraphId) ?? null

export const selectGraphById = (id: string) => (state: StrategyStore) =>
  state.graphs.find((g) => g.id === id) ?? null

export const selectActiveBacktestResult = (graphId: string) => (state: StrategyStore) => {
  const results = state.backtestResults[graphId] ?? []
  const activeId = state.activeBacktestId[graphId]
  return results.find((r) => r.id === activeId) ?? results[0] ?? null
}

export const selectAllBacktestResults = (graphId: string) => (state: StrategyStore) =>
  state.backtestResults[graphId] ?? []
