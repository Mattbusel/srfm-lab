// ============================================================
// useStrategyBuilder — all strategy builder operations
// ============================================================
import { useCallback, useMemo } from 'react'
import { useStrategyStore } from '@/store/strategyStore'
import type {
  StrategyNode,
  StrategyEdge,
  NodeDefinition,
  NodeType,
  NodeCategory,
  NodeParamDef,
} from '@/types'

const generateId = () => Math.random().toString(36).slice(2, 11)

// ---- Node Definitions Registry ----
export const NODE_DEFINITIONS: NodeDefinition[] = [
  // === INDICATORS ===
  {
    type: 'bh_mass',
    category: 'Indicators',
    label: 'BH Mass',
    description: 'Black Hole mass scalar — primary BH physics indicator',
    color: '#7c3aed',
    inputs: [{ id: 'price', label: 'Price', dataType: 'series', position: 'left' }],
    outputs: [
      { id: 'mass', label: 'Mass', dataType: 'number', position: 'right' },
      { id: 'active', label: 'Active', dataType: 'boolean', position: 'right' },
      { id: 'dir', label: 'Direction', dataType: 'number', position: 'right' },
    ],
    params: [
      { key: 'timeframe', label: 'Timeframe', type: 'select', default: '1h', options: [
        { label: '15m', value: '15m' }, { label: '1h', value: '1h' }, { label: '1d', value: '1d' }
      ]},
      { key: 'threshold', label: 'Active Threshold', type: 'number', default: 1.2, min: 0.1, max: 5, step: 0.1 },
    ],
  },
  {
    type: 'ema',
    category: 'Indicators',
    label: 'EMA',
    description: 'Exponential Moving Average',
    color: '#2563eb',
    inputs: [{ id: 'source', label: 'Source', dataType: 'series', position: 'left' }],
    outputs: [{ id: 'value', label: 'EMA', dataType: 'series', position: 'right' }],
    params: [
      { key: 'period', label: 'Period', type: 'number', default: 20, min: 1, max: 500, step: 1 },
      { key: 'source', label: 'Source', type: 'select', default: 'close', options: [
        { label: 'Close', value: 'close' }, { label: 'Open', value: 'open' },
        { label: 'High', value: 'high' }, { label: 'Low', value: 'low' }, { label: 'HL2', value: 'hl2' }
      ]},
    ],
  },
  {
    type: 'sma',
    category: 'Indicators',
    label: 'SMA',
    description: 'Simple Moving Average',
    color: '#2563eb',
    inputs: [{ id: 'source', label: 'Source', dataType: 'series', position: 'left' }],
    outputs: [{ id: 'value', label: 'SMA', dataType: 'series', position: 'right' }],
    params: [
      { key: 'period', label: 'Period', type: 'number', default: 20, min: 2, max: 500, step: 1 },
    ],
  },
  {
    type: 'rsi',
    category: 'Indicators',
    label: 'RSI',
    description: 'Relative Strength Index (0-100)',
    color: '#059669',
    inputs: [{ id: 'source', label: 'Source', dataType: 'series', position: 'left' }],
    outputs: [
      { id: 'value', label: 'RSI', dataType: 'number', position: 'right' },
      { id: 'overbought', label: 'Overbought', dataType: 'boolean', position: 'right' },
      { id: 'oversold', label: 'Oversold', dataType: 'boolean', position: 'right' },
    ],
    params: [
      { key: 'period', label: 'Period', type: 'number', default: 14, min: 2, max: 100, step: 1 },
      { key: 'overbought', label: 'Overbought Level', type: 'number', default: 70, min: 50, max: 100, step: 1 },
      { key: 'oversold', label: 'Oversold Level', type: 'number', default: 30, min: 0, max: 50, step: 1 },
    ],
  },
  {
    type: 'macd',
    category: 'Indicators',
    label: 'MACD',
    description: 'Moving Average Convergence Divergence',
    color: '#059669',
    inputs: [{ id: 'source', label: 'Source', dataType: 'series', position: 'left' }],
    outputs: [
      { id: 'macd', label: 'MACD', dataType: 'number', position: 'right' },
      { id: 'signal', label: 'Signal', dataType: 'number', position: 'right' },
      { id: 'histogram', label: 'Histogram', dataType: 'number', position: 'right' },
    ],
    params: [
      { key: 'fast', label: 'Fast Period', type: 'number', default: 12, min: 2, max: 100 },
      { key: 'slow', label: 'Slow Period', type: 'number', default: 26, min: 2, max: 200 },
      { key: 'signal', label: 'Signal Period', type: 'number', default: 9, min: 2, max: 50 },
    ],
  },
  {
    type: 'atr',
    category: 'Indicators',
    label: 'ATR',
    description: 'Average True Range — volatility measure',
    color: '#d97706',
    inputs: [{ id: 'price', label: 'OHLC', dataType: 'series', position: 'left' }],
    outputs: [{ id: 'value', label: 'ATR', dataType: 'number', position: 'right' }],
    params: [
      { key: 'period', label: 'Period', type: 'number', default: 14, min: 1, max: 100 },
    ],
  },
  {
    type: 'bollinger',
    category: 'Indicators',
    label: 'Bollinger Bands',
    description: 'Bollinger Bands — price envelope',
    color: '#6366f1',
    inputs: [{ id: 'source', label: 'Source', dataType: 'series', position: 'left' }],
    outputs: [
      { id: 'upper', label: 'Upper', dataType: 'series', position: 'right' },
      { id: 'middle', label: 'Middle', dataType: 'series', position: 'right' },
      { id: 'lower', label: 'Lower', dataType: 'series', position: 'right' },
      { id: 'width', label: 'Width', dataType: 'number', position: 'right' },
      { id: 'pct_b', label: '%B', dataType: 'number', position: 'right' },
    ],
    params: [
      { key: 'period', label: 'Period', type: 'number', default: 20, min: 2, max: 200 },
      { key: 'stddev', label: 'Std Dev', type: 'number', default: 2, min: 0.5, max: 5, step: 0.1 },
    ],
  },
  {
    type: 'stochastic',
    category: 'Indicators',
    label: 'Stochastic',
    description: 'Stochastic Oscillator',
    color: '#ec4899',
    inputs: [{ id: 'price', label: 'OHLC', dataType: 'series', position: 'left' }],
    outputs: [
      { id: 'k', label: '%K', dataType: 'number', position: 'right' },
      { id: 'd', label: '%D', dataType: 'number', position: 'right' },
    ],
    params: [
      { key: 'k_period', label: '%K Period', type: 'number', default: 14, min: 1, max: 100 },
      { key: 'd_period', label: '%D Period', type: 'number', default: 3, min: 1, max: 50 },
      { key: 'smooth', label: 'Smooth', type: 'number', default: 3, min: 1, max: 10 },
    ],
  },
  {
    type: 'adx',
    category: 'Indicators',
    label: 'ADX',
    description: 'Average Directional Index — trend strength',
    color: '#0891b2',
    inputs: [{ id: 'price', label: 'OHLC', dataType: 'series', position: 'left' }],
    outputs: [
      { id: 'adx', label: 'ADX', dataType: 'number', position: 'right' },
      { id: 'plus_di', label: '+DI', dataType: 'number', position: 'right' },
      { id: 'minus_di', label: '-DI', dataType: 'number', position: 'right' },
      { id: 'trending', label: 'Trending', dataType: 'boolean', position: 'right' },
    ],
    params: [
      { key: 'period', label: 'Period', type: 'number', default: 14, min: 2, max: 100 },
      { key: 'trend_threshold', label: 'Trend Threshold', type: 'number', default: 25, min: 10, max: 50 },
    ],
  },

  // === SIGNALS ===
  {
    type: 'crossover',
    category: 'Signals',
    label: 'Crossover',
    description: 'Fires when series A crosses above series B',
    color: '#16a34a',
    inputs: [
      { id: 'a', label: 'Series A', dataType: 'series', position: 'left' },
      { id: 'b', label: 'Series B', dataType: 'series', position: 'left' },
    ],
    outputs: [
      { id: 'cross_above', label: 'Cross Above', dataType: 'boolean', position: 'right' },
      { id: 'cross_below', label: 'Cross Below', dataType: 'boolean', position: 'right' },
    ],
    params: [],
  },
  {
    type: 'threshold_cross',
    category: 'Signals',
    label: 'Threshold Cross',
    description: 'Detects when a value crosses a threshold',
    color: '#16a34a',
    inputs: [{ id: 'value', label: 'Value', dataType: 'number', position: 'left' }],
    outputs: [
      { id: 'cross_above', label: 'Cross Above', dataType: 'boolean', position: 'right' },
      { id: 'cross_below', label: 'Cross Below', dataType: 'boolean', position: 'right' },
      { id: 'above', label: 'Is Above', dataType: 'boolean', position: 'right' },
    ],
    params: [
      { key: 'threshold', label: 'Threshold', type: 'number', default: 50, min: -1000, max: 1000, step: 0.01 },
    ],
  },
  {
    type: 'bh_formation',
    category: 'Signals',
    label: 'BH Formation',
    description: 'Fires on BH mass formation event',
    color: '#7c3aed',
    inputs: [{ id: 'mass', label: 'Mass', dataType: 'number', position: 'left' }],
    outputs: [
      { id: 'long', label: 'Long Signal', dataType: 'boolean', position: 'right' },
      { id: 'short', label: 'Short Signal', dataType: 'boolean', position: 'right' },
      { id: 'any', label: 'Any Formation', dataType: 'boolean', position: 'right' },
    ],
    params: [
      { key: 'min_mass', label: 'Min Mass', type: 'number', default: 1.2, min: 0.5, max: 5, step: 0.1 },
      { key: 'require_dir', label: 'Require Direction', type: 'boolean', default: true },
    ],
  },
  {
    type: 'regime_match',
    category: 'Signals',
    label: 'Regime Match',
    description: 'Fires when current BH regime matches target',
    color: '#7c3aed',
    inputs: [],
    outputs: [{ id: 'match', label: 'Match', dataType: 'boolean', position: 'right' }],
    params: [
      { key: 'regime', label: 'Target Regime', type: 'select', default: 'BULL', options: [
        { label: 'Bull', value: 'BULL' }, { label: 'Bear', value: 'BEAR' },
        { label: 'Sideways', value: 'SIDEWAYS' }, { label: 'High Vol', value: 'HIGH_VOL' }
      ]},
    ],
  },

  // === FILTERS ===
  {
    type: 'time_filter',
    category: 'Filters',
    label: 'Time Filter',
    description: 'Allow signals only during specified hours',
    color: '#78716c',
    inputs: [{ id: 'signal', label: 'Signal', dataType: 'boolean', position: 'left' }],
    outputs: [{ id: 'filtered', label: 'Filtered', dataType: 'boolean', position: 'right' }],
    params: [
      { key: 'start_hour', label: 'Start Hour', type: 'number', default: 9, min: 0, max: 23 },
      { key: 'end_hour', label: 'End Hour', type: 'number', default: 16, min: 0, max: 23 },
      { key: 'timezone', label: 'Timezone', type: 'select', default: 'US/Eastern', options: [
        { label: 'US Eastern', value: 'US/Eastern' }, { label: 'UTC', value: 'UTC' }
      ]},
    ],
  },
  {
    type: 'volume_filter',
    category: 'Filters',
    label: 'Volume Filter',
    description: 'Allow signals only when volume is above average',
    color: '#78716c',
    inputs: [{ id: 'signal', label: 'Signal', dataType: 'boolean', position: 'left' }],
    outputs: [{ id: 'filtered', label: 'Filtered', dataType: 'boolean', position: 'right' }],
    params: [
      { key: 'min_vol_multiplier', label: 'Min Vol Multiplier', type: 'number', default: 1.0, min: 0.1, max: 10, step: 0.1 },
      { key: 'lookback', label: 'Lookback', type: 'number', default: 20, min: 5, max: 200 },
    ],
  },
  {
    type: 'regime_filter',
    category: 'Filters',
    label: 'Regime Filter',
    description: 'Allow signals only in specific BH regime',
    color: '#78716c',
    inputs: [{ id: 'signal', label: 'Signal', dataType: 'boolean', position: 'left' }],
    outputs: [{ id: 'filtered', label: 'Filtered', dataType: 'boolean', position: 'right' }],
    params: [
      { key: 'allow_bull', label: 'Allow Bull', type: 'boolean', default: true },
      { key: 'allow_bear', label: 'Allow Bear', type: 'boolean', default: false },
      { key: 'allow_sideways', label: 'Allow Sideways', type: 'boolean', default: false },
      { key: 'allow_high_vol', label: 'Allow High Vol', type: 'boolean', default: true },
    ],
  },

  // === SIZERS ===
  {
    type: 'fixed_fraction',
    category: 'Sizers',
    label: 'Fixed Fraction',
    description: 'Size position as fixed % of equity',
    color: '#b45309',
    inputs: [{ id: 'signal', label: 'Signal', dataType: 'boolean', position: 'left' }],
    outputs: [{ id: 'qty', label: 'Quantity', dataType: 'number', position: 'right' }],
    params: [
      { key: 'fraction', label: 'Fraction', type: 'number', default: 0.05, min: 0.001, max: 1.0, step: 0.001, unit: '%' },
      { key: 'max_shares', label: 'Max Shares', type: 'number', default: 0, min: 0, max: 10000 },
    ],
  },
  {
    type: 'kelly',
    category: 'Sizers',
    label: 'Kelly Criterion',
    description: 'Optimal sizing using Kelly formula',
    color: '#b45309',
    inputs: [{ id: 'signal', label: 'Signal', dataType: 'boolean', position: 'left' }],
    outputs: [{ id: 'qty', label: 'Quantity', dataType: 'number', position: 'right' }],
    params: [
      { key: 'win_rate', label: 'Win Rate', type: 'number', default: 0.55, min: 0.01, max: 0.99, step: 0.01 },
      { key: 'win_loss_ratio', label: 'Win/Loss Ratio', type: 'number', default: 1.5, min: 0.1, max: 10, step: 0.1 },
      { key: 'kelly_fraction', label: 'Kelly Fraction', type: 'number', default: 0.25, min: 0.01, max: 1.0, step: 0.01 },
      { key: 'max_fraction', label: 'Max Fraction', type: 'number', default: 0.1, min: 0.01, max: 0.5, step: 0.01 },
    ],
  },
  {
    type: 'vol_target',
    category: 'Sizers',
    label: 'Vol Target',
    description: 'Size to target portfolio volatility',
    color: '#b45309',
    inputs: [
      { id: 'signal', label: 'Signal', dataType: 'boolean', position: 'left' },
      { id: 'atr', label: 'ATR', dataType: 'number', position: 'left' },
    ],
    outputs: [{ id: 'qty', label: 'Quantity', dataType: 'number', position: 'right' }],
    params: [
      { key: 'target_vol', label: 'Target Annual Vol', type: 'number', default: 0.15, min: 0.01, max: 1.0, step: 0.01 },
      { key: 'lookback', label: 'Vol Lookback', type: 'number', default: 20, min: 5, max: 100 },
    ],
  },

  // === LOGIC ===
  {
    type: 'and',
    category: 'Logic',
    label: 'AND',
    description: 'True when all inputs are true',
    color: '#374151',
    inputs: [
      { id: 'a', label: 'A', dataType: 'boolean', position: 'left' },
      { id: 'b', label: 'B', dataType: 'boolean', position: 'left' },
      { id: 'c', label: 'C (opt)', dataType: 'boolean', position: 'left' },
    ],
    outputs: [{ id: 'result', label: 'Result', dataType: 'boolean', position: 'right' }],
    params: [],
    minInputs: 2,
  },
  {
    type: 'or',
    category: 'Logic',
    label: 'OR',
    description: 'True when any input is true',
    color: '#374151',
    inputs: [
      { id: 'a', label: 'A', dataType: 'boolean', position: 'left' },
      { id: 'b', label: 'B', dataType: 'boolean', position: 'left' },
    ],
    outputs: [{ id: 'result', label: 'Result', dataType: 'boolean', position: 'right' }],
    params: [],
    minInputs: 2,
  },
  {
    type: 'not',
    category: 'Logic',
    label: 'NOT',
    description: 'Inverts boolean input',
    color: '#374151',
    inputs: [{ id: 'input', label: 'Input', dataType: 'boolean', position: 'left' }],
    outputs: [{ id: 'result', label: 'Result', dataType: 'boolean', position: 'right' }],
    params: [],
  },
  {
    type: 'cooldown',
    category: 'Logic',
    label: 'Cooldown',
    description: 'Block signals within N bars of last trigger',
    color: '#374151',
    inputs: [{ id: 'signal', label: 'Signal', dataType: 'boolean', position: 'left' }],
    outputs: [{ id: 'result', label: 'Result', dataType: 'boolean', position: 'right' }],
    params: [
      { key: 'bars', label: 'Cooldown Bars', type: 'number', default: 5, min: 1, max: 500 },
    ],
  },
  {
    type: 'delay',
    category: 'Logic',
    label: 'Delay',
    description: 'Delay signal by N bars',
    color: '#374151',
    inputs: [{ id: 'signal', label: 'Signal', dataType: 'boolean', position: 'left' }],
    outputs: [{ id: 'result', label: 'Result', dataType: 'boolean', position: 'right' }],
    params: [
      { key: 'bars', label: 'Delay Bars', type: 'number', default: 1, min: 1, max: 100 },
    ],
  },

  // === OUTPUT ===
  {
    type: 'entry_long',
    category: 'Outputs',
    label: 'Enter Long',
    description: 'Execute long entry when signal fires',
    color: '#16a34a',
    inputs: [
      { id: 'signal', label: 'Signal', dataType: 'boolean', position: 'left' },
      { id: 'qty', label: 'Qty', dataType: 'number', position: 'left' },
    ],
    outputs: [],
    params: [
      { key: 'order_type', label: 'Order Type', type: 'select', default: 'market', options: [
        { label: 'Market', value: 'market' }, { label: 'Limit', value: 'limit' }
      ]},
      { key: 'limit_offset', label: 'Limit Offset', type: 'number', default: 0, min: -10, max: 10, step: 0.01 },
    ],
  },
  {
    type: 'exit_long',
    category: 'Outputs',
    label: 'Exit Long',
    description: 'Exit long position when signal fires',
    color: '#dc2626',
    inputs: [
      { id: 'signal', label: 'Signal', dataType: 'boolean', position: 'left' },
    ],
    outputs: [],
    params: [
      { key: 'order_type', label: 'Order Type', type: 'select', default: 'market', options: [
        { label: 'Market', value: 'market' }, { label: 'Limit', value: 'limit' }
      ]},
      { key: 'stop_loss_pct', label: 'Stop Loss %', type: 'number', default: 0.02, min: 0, max: 0.5, step: 0.001 },
      { key: 'take_profit_pct', label: 'Take Profit %', type: 'number', default: 0.05, min: 0, max: 1.0, step: 0.001 },
    ],
  },
]

const NODE_DEF_MAP = new Map(NODE_DEFINITIONS.map((d) => [d.type, d]))

export function useStrategyBuilder(graphId: string | null) {
  const store = useStrategyStore()
  const graph = graphId ? store.graphs.find((g) => g.id === graphId) ?? null : null

  const addNodeFromDefinition = useCallback((defType: string, position: { x: number; y: number }) => {
    if (!graphId) return null
    const def = NODE_DEF_MAP.get(defType)
    if (!def) return null

    const node: StrategyNode = {
      id: generateId(),
      type: def.category.toLowerCase() as NodeType,
      definitionType: defType,
      name: def.label,
      params: Object.fromEntries(def.params.map((p) => [p.key, p.default])),
      position,
      inputs: def.inputs.map((i) => i.id),
      outputs: def.outputs.map((o) => o.id),
    }

    store.addNode(graphId, node)
    return node.id
  }, [graphId, store])

  const connectNodes = useCallback((
    sourceNodeId: string,
    sourceHandle: string,
    targetNodeId: string,
    targetHandle: string
  ) => {
    if (!graphId) return null

    // Validate connection (type checking)
    const sourceNode = graph?.nodes.find((n) => n.id === sourceNodeId)
    const targetNode = graph?.nodes.find((n) => n.id === targetNodeId)
    if (!sourceNode || !targetNode) return null

    // Prevent self-connection
    if (sourceNodeId === targetNodeId) return null

    const edge: StrategyEdge = {
      id: generateId(),
      source: sourceNodeId,
      sourceHandle,
      target: targetNodeId,
      targetHandle,
    }

    store.addEdge(graphId, edge)
    return edge.id
  }, [graphId, graph, store])

  const nodeDefinitions = useMemo(() => NODE_DEFINITIONS, [])
  const nodesByCategory = useMemo(() => {
    const map = new Map<NodeCategory, NodeDefinition[]>()
    for (const def of NODE_DEFINITIONS) {
      if (!map.has(def.category)) map.set(def.category, [])
      map.get(def.category)!.push(def)
    }
    return map
  }, [])

  const getNodeDef = useCallback((type: string) => NODE_DEF_MAP.get(type) ?? null, [])

  const validateGraph = useCallback(() => {
    if (!graph) return { valid: false, errors: ['No graph selected'] }
    const errors: string[] = []

    // Check for output nodes
    const hasOutput = graph.nodes.some((n) => n.type === 'output' || n.definitionType.startsWith('entry') || n.definitionType.startsWith('exit'))
    if (!hasOutput) errors.push('Graph has no entry/exit nodes')

    // Check disconnected nodes
    for (const node of graph.nodes) {
      const def = NODE_DEF_MAP.get(node.definitionType)
      if (!def) continue

      const requiredInputs = def.inputs.slice(0, def.minInputs ?? 0)
      for (const input of requiredInputs) {
        const hasConnection = graph.edges.some((e) => e.target === node.id && e.targetHandle === input.id)
        if (!hasConnection) {
          errors.push(`Node "${node.name}" missing required input "${input.label}"`)
        }
      }
    }

    return { valid: errors.length === 0, errors }
  }, [graph])

  return {
    graph,
    nodeDefinitions,
    nodesByCategory,
    getNodeDef,
    addNodeFromDefinition,
    connectNodes,
    validateGraph,
    moveNode: (nodeId: string, position: { x: number; y: number }) =>
      graphId && store.moveNode(graphId, nodeId, position),
    removeNode: (nodeId: string) =>
      graphId && store.removeNode(graphId, nodeId),
    updateNodeParams: (nodeId: string, params: Record<string, number | string | boolean>) =>
      graphId && store.updateNodeParams(graphId, nodeId, params),
    removeEdge: (edgeId: string) =>
      graphId && store.removeEdge(graphId, edgeId),
    setViewport: (viewport: { x: number; y: number; zoom: number }) =>
      graphId && store.setViewport(graphId, viewport),
  }
}
