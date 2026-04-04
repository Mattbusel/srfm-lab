import { useState } from 'react';
import { useLiveState } from '../hooks/useLiveState';
import { BHMassGauge } from '../components/BHMassGauge';
import { FormationRadar } from '../components/FormationRadar';
import { EquityCurve } from '../components/EquityCurve';
import { RegimeBadge } from '../components/RegimeBadge';
import type { InstrumentLiveState, Regime } from '../types';

function dotColor(inst: InstrumentLiveState): string {
  const active = inst.active_15m || inst.active_1h || inst.active_1d;
  const maxMass = Math.max(inst.bh_mass_15m, inst.bh_mass_1h, inst.bh_mass_1d);
  if (active) return '#22c55e';
  if (maxMass >= 1.2) return '#eab308';
  return '#4b5563';
}

function buildDemoInstruments(): Record<string, InstrumentLiveState> {
  const regimes: Regime[] = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOL'];
  const syms = ['BTC', 'ETH', 'SPY', 'QQQ', 'GLD', 'TLT'];
  const result: Record<string, InstrumentLiveState> = {};
  syms.forEach((sym, i) => {
    result[sym] = {
      bh_mass_15m: Math.max(0.4, 0.8 + Math.sin(i * 1.3) * 0.7),
      bh_mass_1h: Math.max(0.4, 1.0 + Math.cos(i * 0.9) * 0.6),
      bh_mass_1d: Math.max(0.4, 1.2 + Math.sin(i * 0.5) * 0.5),
      active_15m: i === 0,
      active_1h: i === 0 || i === 2,
      active_1d: i < 3,
      price: 83500 / Math.pow(1.8, i),
      frac: 0.05 * (i + 1),
      regime: regimes[i % 4],
    };
  });
  return result;
}

export function Dashboard() {
  const { latest, equityHistory, pnlToday, activePositions, wsStatus } = useLiveState();
  const [selectedSym, setSelectedSym] = useState<string | null>(null);

  const instruments = latest?.instruments ?? buildDemoInstruments();
  const equity = latest?.equity ?? 105234.50;
  const instrumentEntries = Object.entries(instruments);
  const recentEquity = equityHistory.slice(-4320);

  const gaugeSize = 96;

  return (
    <div className="flex gap-4" style={{ height: 'calc(100vh - 112px)' }}>
      {/* Sidebar */}
      <aside className="w-52 flex-shrink-0 flex flex-col gap-3 overflow-hidden">
        <div className="bg-bg-card border border-bg-border rounded-lg p-3 flex-1 overflow-y-auto">
          <p className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-2">Instruments</p>
          <div className="space-y-0.5">
            {instrumentEntries
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([sym, inst]) => {
                const maxMass = Math.max(inst.bh_mass_15m, inst.bh_mass_1h, inst.bh_mass_1d);
                return (
                  <button
                    key={sym}
                    onClick={() => setSelectedSym(sym === selectedSym ? null : sym)}
                    className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left transition-all text-sm
                      ${selectedSym === sym
                        ? 'bg-accent/20 border border-accent/30'
                        : 'hover:bg-bg-elevated border border-transparent'
                      }`}
                  >
                    <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: dotColor(inst) }} />
                    <span className="font-mono font-semibold text-gray-200 flex-1">{sym}</span>
                    <span className="text-xs font-mono text-gray-500">{maxMass.toFixed(2)}</span>
                  </button>
                );
              })}
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col gap-3 min-w-0 overflow-hidden">
        {/* KPI bar */}
        <div className="grid grid-cols-4 gap-3 flex-shrink-0">
          {[
            {
              label: 'Portfolio Equity',
              value: `$${equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
              cls: 'text-gray-100',
            },
            {
              label: 'P&L Today',
              value: `${pnlToday >= 0 ? '+' : ''}$${Math.abs(pnlToday).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
              cls: pnlToday >= 0 ? 'text-bull' : 'text-bear',
            },
            {
              label: 'Active Positions',
              value: String(activePositions),
              cls: 'text-accent',
            },
            {
              label: 'Data Feed',
              value: wsStatus === 'open' ? 'LIVE' : wsStatus.toUpperCase(),
              cls: wsStatus === 'open' ? 'text-bull' : 'text-bear',
            },
          ].map(({ label, value, cls }) => (
            <div key={label} className="bg-bg-card border border-bg-border rounded-lg px-4 py-3">
              <p className="text-xs font-mono text-gray-500 uppercase tracking-wider">{label}</p>
              <p className={`text-xl font-mono font-bold mt-1 ${cls}`}>{value}</p>
            </div>
          ))}
        </div>

        {/* Gauge grid */}
        <div className="bg-bg-card border border-bg-border rounded-lg p-4 flex-1 overflow-y-auto">
          <p className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-4">
            BH Mass Monitor — All Timeframes
          </p>
          <div
            className="grid gap-4"
            style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))' }}
          >
            {instrumentEntries.map(([sym, inst]) => (
              <div
                key={sym}
                className={`bg-bg-elevated rounded-xl p-3 border cursor-pointer transition-all
                  ${selectedSym === sym
                    ? 'border-accent/50 shadow shadow-accent/20'
                    : 'border-bg-border hover:border-gray-600'
                  }`}
                onClick={() => setSelectedSym(sym === selectedSym ? null : sym)}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono font-bold text-gray-200">{sym}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-gray-500">
                      ${inst.price.toLocaleString('en-US', { maximumFractionDigits: 2 })}
                    </span>
                    <RegimeBadge regime={inst.regime as Regime} />
                  </div>
                </div>
                <div className="flex justify-around">
                  <BHMassGauge mass={inst.bh_mass_15m} timeframe="15m" active={inst.active_15m} size={gaugeSize} />
                  <BHMassGauge mass={inst.bh_mass_1h} timeframe="1h" active={inst.active_1h} size={gaugeSize} />
                  <BHMassGauge mass={inst.bh_mass_1d} timeframe="1d" active={inst.active_1d} size={gaugeSize} />
                </div>
                <div className="mt-2 flex justify-between text-xs font-mono text-gray-600">
                  <span>Frac {(inst.frac * 100).toFixed(1)}%</span>
                  <span className={inst.active_15m || inst.active_1h || inst.active_1d ? 'text-bull' : ''}>
                    {inst.active_15m || inst.active_1h || inst.active_1d ? '● IN POSITION' : '○ FLAT'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom strip */}
        <div className="grid grid-cols-3 gap-3 flex-shrink-0" style={{ height: '200px' }}>
          <div className="col-span-2 bg-bg-card border border-bg-border rounded-lg p-3">
            <p className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-1">
              Equity — Last 30 Days
            </p>
            <EquityCurve data={recentEquity} height={155} showRegimeColor={false} />
          </div>
          <div className="bg-bg-card border border-bg-border rounded-lg p-3 overflow-y-auto">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-1.5 h-1.5 rounded-full bg-bh-warm animate-pulse" />
              <p className="text-xs font-mono text-gray-500 uppercase tracking-wider">Formation Radar</p>
            </div>
            <FormationRadar instruments={instruments} />
          </div>
        </div>
      </div>
    </div>
  );
}
