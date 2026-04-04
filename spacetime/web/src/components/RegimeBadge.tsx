import type { Regime } from '../types';

interface Props {
  regime: Regime;
  size?: 'sm' | 'md';
}

const CONFIG: Record<Regime, { label: string; cls: string }> = {
  BULL:     { label: 'BULL',     cls: 'bg-bull/20 text-bull border-bull/40' },
  BEAR:     { label: 'BEAR',     cls: 'bg-bear/20 text-bear border-bear/40' },
  SIDEWAYS: { label: 'SIDEWAYS', cls: 'bg-sideways/20 text-sideways border-sideways/40' },
  HIGH_VOL: { label: 'HIGH VOL', cls: 'bg-highvol/20 text-highvol border-highvol/40' },
};

export function RegimeBadge({ regime, size = 'sm' }: Props) {
  const { label, cls } = CONFIG[regime] ?? CONFIG['SIDEWAYS'];
  const sz = size === 'sm' ? 'text-xs px-1.5 py-0.5' : 'text-sm px-2.5 py-1';
  return (
    <span className={`inline-flex items-center rounded border font-mono font-semibold ${sz} ${cls}`}>
      {label}
    </span>
  );
}
