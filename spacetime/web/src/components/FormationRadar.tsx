import type { InstrumentLiveState } from '../types';

interface Props {
  instruments: Record<string, InstrumentLiveState>;
}

function getMaxMass(inst: InstrumentLiveState): { mass: number; tf: string } {
  const vals = [
    { mass: inst.bh_mass_15m, tf: '15m' },
    { mass: inst.bh_mass_1h, tf: '1h' },
    { mass: inst.bh_mass_1d, tf: '1d' },
  ];
  return vals.reduce((a, b) => (b.mass > a.mass ? b : a));
}

function getMassBarColor(mass: number): string {
  if (mass >= 1.8) return 'bg-bear';
  if (mass >= 1.5) return 'bg-bh-hot';
  if (mass >= 1.2) return 'bg-bh-warm';
  return 'bg-bh-cold';
}

function getMassTextColor(mass: number): string {
  if (mass >= 1.8) return 'text-bear';
  if (mass >= 1.5) return 'text-bh-hot';
  if (mass >= 1.2) return 'text-bh-warm';
  return 'text-bh-cold';
}

export function FormationRadar({ instruments }: Props) {
  const entries = Object.entries(instruments)
    .map(([sym, inst]) => ({ sym, inst, ...getMaxMass(inst) }))
    .filter(e => e.mass >= 1.0)
    .sort((a, b) => b.mass - a.mass);

  if (entries.length === 0) {
    return (
      <div className="text-center py-8 text-gray-600 text-sm font-mono">
        No formations approaching threshold
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {entries.map(({ sym, inst, mass, tf }) => {
        const frac = Math.min(mass / 2.0, 1);
        const isActive = inst.active_15m || inst.active_1h || inst.active_1d;
        return (
          <div key={sym} className="space-y-1">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div
                  className={`w-2 h-2 rounded-full ${isActive ? 'animate-pulse' : ''}`}
                  style={{
                    backgroundColor: mass >= 1.8 ? '#ef4444' : mass >= 1.5 ? '#f97316' : mass >= 1.2 ? '#eab308' : '#6b7280',
                  }}
                />
                <span className="text-sm font-mono text-gray-200 font-semibold">{sym}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-mono ${getMassTextColor(mass)}`}>
                  {mass.toFixed(3)}
                </span>
                <span className="text-xs font-mono text-gray-500 bg-bg-elevated px-1 rounded">
                  {tf}
                </span>
              </div>
            </div>
            <div className="h-1.5 bg-bg-border rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-700 ease-out ${getMassBarColor(mass)}`}
                style={{ width: `${frac * 100}%` }}
              />
            </div>
            {/* Threshold markers */}
            <div className="relative h-0">
              {[1.2, 1.5, 1.8].map(threshold => (
                <div
                  key={threshold}
                  className="absolute top-[-6px] w-px h-3 bg-gray-600 opacity-50"
                  style={{ left: `${(threshold / 2.0) * 100}%` }}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
