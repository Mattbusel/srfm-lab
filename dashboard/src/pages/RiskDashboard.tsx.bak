// ============================================================
// RiskDashboard.tsx — Risk view page
// ============================================================
import React, { useMemo, useEffect } from 'react'
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ReferenceLine,
} from 'recharts'
import { clsx } from 'clsx'
import { format, parseISO } from 'date-fns'
import { Card, StatCard, ProgressBar } from '@/components/ui'
import { CorrelationMatrix } from '@/components/CorrelationMatrix'
import { usePositionsStore } from '@/store/positionsStore'
import { usePortfolioStore } from '@/store/portfolioStore'
import type { CorrelationEntry } from '@/types'

// ---- Mock risk data ----

const RISK_METRICS = {
  var95: 3240,
  var99: 5180,
  cvar95: 4620,
  cvar99: 7340,
  grossExposure: 96800,
  netExposure: 74200,
  leverageRatio: 1.42,
  marginUtilization: 0.378,
  concentrationHHI: 0.156,
  correlationRisk: 0.61,
  liquidityRisk: 0.18,
  betaToMarket: 0.74,
  maxSinglePositionPct: 0.21,
  limitUtilization: {
    'Max Single Position': 0.42,
    'Max Gross Exposure': 0.64,
    'Max Net Exposure': 0.58,
    'Daily Loss Limit': 0.22,
    'Drawdown Limit': 0.31,
    'Margin Utilization': 0.378,
    'Concentration Limit': 0.52,
    'Beta Limit': 0.74,
  },
}

function generateCorrelationData(): CorrelationEntry[] {
  const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'LINKUSDT', 'AVAXUSDT', 'DOGEUSDT']
  const entries: CorrelationEntry[] = []
  const baseCorrelations: Record<string, number> = {
    'BTCUSDT|ETHUSDT': 0.88,
    'BTCUSDT|SOLUSDT': 0.72,
    'BTCUSDT|BNBUSDT': 0.79,
    'BTCUSDT|LINKUSDT': 0.65,
    'BTCUSDT|AVAXUSDT': 0.71,
    'BTCUSDT|DOGEUSDT': 0.58,
    'ETHUSDT|SOLUSDT': 0.76,
    'ETHUSDT|BNBUSDT': 0.82,
    'ETHUSDT|LINKUSDT': 0.78,
    'ETHUSDT|AVAXUSDT': 0.74,
    'ETHUSDT|DOGEUSDT': 0.53,
    'SOLUSDT|BNBUSDT': 0.69,
    'SOLUSDT|LINKUSDT': 0.61,
    'SOLUSDT|AVAXUSDT': 0.67,
    'SOLUSDT|DOGEUSDT': 0.49,
    'BNBUSDT|LINKUSDT': 0.64,
    'BNBUSDT|AVAXUSDT': 0.68,
    'BNBUSDT|DOGEUSDT': 0.47,
    'LINKUSDT|AVAXUSDT': 0.72,
    'LINKUSDT|DOGEUSDT': 0.43,
    'AVAXUSDT|DOGEUSDT': 0.51,
  }
  for (let i = 0; i < symbols.length; i++) {
    for (let j = i + 1; j < symbols.length; j++) {
      const key = `${symbols[i]}|${symbols[j]}`
      const corr = (baseCorrelations[key] ?? 0.5) + (Math.random() - 0.5) * 0.05
      entries.push({ symbolA: symbols[i], symbolB: symbols[j], correlation: corr })
    }
  }
  return entries
}

function generateDrawdownTimeline(): { timestamp: string; drawdown: number; equity: number }[] {
  const points = []
  let equity = 100000
  let peak = equity
  const now = Date.now()
  for (let i = 90 * 24; i >= 0; i--) {
    const delta = (Math.random() - 0.47) * 600
    equity = Math.max(equity + delta, 70000)
    if (equity > peak) peak = equity
    points.push({
      timestamp: new Date(now - i * 3600000).toISOString(),
      equity,
      drawdown: ((equity - peak) / peak) * 100,
    })
  }
  return points
}

// ---- VaR Gauge ----

const VaRGauge: React.FC<{
  label: string
  value: number
  equity: number
  color: string
}> = ({ label, value, equity, color }) => {
  const pct = value / equity
  const angle = Math.min(pct / 0.1, 1) * 180  // 10% = full

  const cx = 70
  const cy = 65
  const r = 52
  const sw = 10

  function arc(deg: number) {
    const rad = ((deg - 180) * Math.PI) / 180
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
  }
  const largeArc = angle > 90 ? 1 : 0

  return (
    <div className="flex flex-col items-center">
      <svg width={140} height={80} viewBox="0 0 140 80">
        <path
          d={`M ${arc(0).x} ${arc(0).y} A ${r} ${r} 0 1 1 ${arc(180).x} ${arc(180).y}`}
          fill="none"
          stroke="#1e2130"
          strokeWidth={sw}
          strokeLinecap="round"
        />
        <path
          d={`M ${arc(0).x} ${arc(0).y} A ${r} ${r} 0 ${largeArc} 1 ${arc(angle).x} ${arc(angle).y}`}
          fill="none"
          stroke={color}
          strokeWidth={sw}
          strokeLinecap="round"
        />
        <text x={cx} y={cy - 4} textAnchor="middle" fill="#e2e8f0" fontSize={14} fontFamily="JetBrains Mono" fontWeight={700}>
          ${(value / 1000).toFixed(1)}k
        </text>
        <text x={cx} y={cy + 10} textAnchor="middle" fill="#475569" fontSize={9} fontFamily="JetBrains Mono">
          {(pct * 100).toFixed(2)}%
        </text>
      </svg>
      <span className="text-[10px] font-mono text-slate-500">{label}</span>
    </div>
  )
}

// ---- Page ----

export const RiskDashboard: React.FC = () => {
  const { positions, initMockData } = usePositionsStore()
  const { snapshot, equityCurve, initMockData: initPortfolio } = usePortfolioStore()

  useEffect(() => {
    initMockData()
    initPortfolio()
  }, [initMockData, initPortfolio])

  const correlationData = useMemo(() => generateCorrelationData(), [])
  const drawdownTimeline = useMemo(() => generateDrawdownTimeline(), [])
  const drawdownDisplay = useMemo(
    () => drawdownTimeline
      .filter((_, i) => i % 24 === 0)  // daily
      .map((p) => ({
        ...p,
        dateLabel: format(parseISO(p.timestamp), 'MMM d'),
      })),
    [drawdownTimeline],
  )

  const concentrationData = useMemo(
    () => positions.map((p) => ({ name: p.symbol.replace('USDT', ''), value: p.sizeUsd })),
    [positions],
  )

  const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#14b8a6']

  const limitEntries = Object.entries(RISK_METRICS.limitUtilization)

  return (
    <div className="flex flex-col h-full overflow-y-auto thin-scrollbar p-4 gap-4">

      {/* Risk summary stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Gross Exposure" value={`$${(RISK_METRICS.grossExposure / 1000).toFixed(1)}k`} />
        <StatCard label="Net Exposure" value={`$${(RISK_METRICS.netExposure / 1000).toFixed(1)}k`} />
        <StatCard label="Leverage" value={`${RISK_METRICS.leverageRatio.toFixed(2)}x`} valueClass="text-xl text-amber-400" />
        <StatCard label="Portfolio Beta" value={RISK_METRICS.betaToMarket.toFixed(2)} />
      </div>

      {/* VaR/CVaR gauges */}
      <Card title="Value at Risk / Expected Shortfall">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <VaRGauge label="VaR 95%" value={RISK_METRICS.var95} equity={snapshot?.totalEquity ?? 127450} color="#f59e0b" />
          <VaRGauge label="VaR 99%" value={RISK_METRICS.var99} equity={snapshot?.totalEquity ?? 127450} color="#ef4444" />
          <VaRGauge label="CVaR 95%" value={RISK_METRICS.cvar95} equity={snapshot?.totalEquity ?? 127450} color="#f97316" />
          <VaRGauge label="CVaR 99%" value={RISK_METRICS.cvar99} equity={snapshot?.totalEquity ?? 127450} color="#dc2626" />
        </div>
      </Card>

      {/* Position concentration + Drawdown */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card title="Position Concentration" subtitle="Allocation by notional">
          <div className="flex items-center gap-4">
            <ResponsiveContainer width={180} height={180}>
              <PieChart>
                <Pie
                  data={concentrationData}
                  cx="50%"
                  cy="50%"
                  innerRadius="45%"
                  outerRadius="75%"
                  dataKey="value"
                  paddingAngle={1}
                >
                  {concentrationData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} fillOpacity={0.85} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v: number) => [`$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}`, 'Size']}
                  contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 10, fontFamily: 'JetBrains Mono' }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-col gap-1.5 flex-1">
              {concentrationData.map((d, i) => (
                <div key={d.name} className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                  <span className="text-[10px] font-mono text-slate-400 flex-1">{d.name}</span>
                  <span className="text-[10px] font-mono text-slate-500">
                    ${(d.value / 1000).toFixed(1)}k
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Card>

        <Card title="Drawdown Timeline" subtitle="90 days">
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={drawdownDisplay} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="#1e2130" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="dateLabel"
                tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
                interval={6}
              />
              <YAxis
                tick={{ fill: '#475569', fontSize: 8, fontFamily: 'JetBrains Mono' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                domain={['auto', 0]}
              />
              <Tooltip
                formatter={(v: number) => [`${v.toFixed(2)}%`, 'Drawdown']}
                contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 10, fontFamily: 'JetBrains Mono' }}
              />
              <ReferenceLine y={-10} stroke="#ef4444" strokeDasharray="4 2" strokeOpacity={0.4} label={{ value: '-10%', fill: '#ef4444', fontSize: 8, fontFamily: 'JetBrains Mono' }} />
              <Area
                type="monotone"
                dataKey="drawdown"
                stroke="#ef4444"
                strokeWidth={1.5}
                fill="rgba(239,68,68,0.15)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Correlation heatmap */}
      <Card title="Correlation Matrix" subtitle="30-day rolling pairwise correlations">
        <CorrelationMatrix data={correlationData} cellSize={48} />
      </Card>

      {/* Limit utilization bars */}
      <Card title="Limit Utilization" subtitle="Current usage vs configured limits">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3">
          {limitEntries.map(([name, util]) => (
            <ProgressBar
              key={name}
              label={name}
              value={util}
              color={util > 0.8 ? '#ef4444' : util > 0.65 ? '#f59e0b' : '#3b82f6'}
              showValue
            />
          ))}
        </div>
      </Card>

      {/* Risk factor bar chart */}
      <Card title="Risk Factor Summary">
        <ResponsiveContainer width="100%" height={160}>
          <BarChart
            data={[
              { name: 'Corr Risk', value: RISK_METRICS.correlationRisk * 100 },
              { name: 'Liq Risk', value: RISK_METRICS.liquidityRisk * 100 },
              { name: 'Conc HHI', value: RISK_METRICS.concentrationHHI * 100 },
              { name: 'Margin %', value: RISK_METRICS.marginUtilization * 100 },
              { name: 'Beta', value: RISK_METRICS.betaToMarket * 100 },
            ]}
            layout="vertical"
            margin={{ top: 4, right: 8, bottom: 0, left: 64 }}
          >
            <CartesianGrid stroke="#1e2130" strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fill: '#475569', fontSize: 8 }} axisLine={false} tickLine={false} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
            <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 9, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
            <Tooltip
              formatter={(v: number) => [`${v.toFixed(1)}%`, 'Value']}
              contentStyle={{ background: '#111318', border: '1px solid #1e2130', borderRadius: 6, fontSize: 10, fontFamily: 'JetBrains Mono' }}
            />
            <Bar dataKey="value" fill="#3b82f6" fillOpacity={0.75} radius={[0, 3, 3, 0]}>
              {[RISK_METRICS.correlationRisk, RISK_METRICS.liquidityRisk, RISK_METRICS.concentrationHHI, RISK_METRICS.marginUtilization, RISK_METRICS.betaToMarket].map((v, i) => (
                <Cell key={i} fill={v > 0.7 ? '#ef4444' : v > 0.5 ? '#f59e0b' : '#3b82f6'} fillOpacity={0.75} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

    </div>
  )
}
