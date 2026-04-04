import { useEffect, useRef } from 'react';

interface Props {
  mass: number;
  timeframe: '15m' | '1h' | '1d';
  active: boolean;
  sym?: string;
  size?: number;
}

function getMassColor(mass: number): { stroke: string; glow: string; fill: string } {
  if (mass >= 1.8) return { stroke: '#ef4444', glow: 'rgba(239,68,68,0.6)', fill: '#ef444420' };
  if (mass >= 1.5) return { stroke: '#f97316', glow: 'rgba(249,115,22,0.5)', fill: '#f9731620' };
  if (mass >= 1.2) return { stroke: '#eab308', glow: 'rgba(234,179,8,0.4)', fill: '#eab30820' };
  return { stroke: '#6b7280', glow: 'rgba(107,114,128,0.2)', fill: '#6b728010' };
}

export function BHMassGauge({ mass, timeframe, active, sym, size = 120 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const currentMassRef = useRef<number>(mass);
  const targetMassRef = useRef<number>(mass);

  useEffect(() => {
    targetMassRef.current = mass;
  }, [mass]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size / 2;
    const r = size * 0.38;
    const startAngle = Math.PI * 0.75;
    const totalAngle = Math.PI * 1.5;
    const maxMass = 2.0;

    function draw(m: number) {
      ctx.clearRect(0, 0, size, size);

      // Background track
      ctx.beginPath();
      ctx.arc(cx, cy, r, startAngle, startAngle + totalAngle);
      ctx.strokeStyle = '#2a2a3a';
      ctx.lineWidth = size * 0.08;
      ctx.lineCap = 'round';
      ctx.stroke();

      // Zone arcs
      const zones = [
        { from: 0, to: 1.2 / maxMass, color: '#6b7280' },
        { from: 1.2 / maxMass, to: 1.5 / maxMass, color: '#eab308' },
        { from: 1.5 / maxMass, to: 1.8 / maxMass, color: '#f97316' },
        { from: 1.8 / maxMass, to: 1.0, color: '#ef4444' },
      ];

      zones.forEach(({ from, to, color }) => {
        ctx.beginPath();
        ctx.arc(cx, cy, r, startAngle + from * totalAngle, startAngle + to * totalAngle);
        ctx.strokeStyle = color + '30';
        ctx.lineWidth = size * 0.04;
        ctx.lineCap = 'butt';
        ctx.stroke();
      });

      // Value arc
      const frac = Math.min(m / maxMass, 1);
      if (frac > 0) {
        const colors = getMassColor(m);

        // Glow effect
        ctx.save();
        ctx.shadowColor = colors.glow;
        ctx.shadowBlur = size * 0.12;
        ctx.beginPath();
        ctx.arc(cx, cy, r, startAngle, startAngle + frac * totalAngle);
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = size * 0.08;
        ctx.lineCap = 'round';
        ctx.stroke();
        ctx.restore();
      }

      // Needle
      const needleAngle = startAngle + frac * totalAngle;
      const needleColors = getMassColor(m);
      const nx = cx + Math.cos(needleAngle) * (r - size * 0.02);
      const ny = cy + Math.sin(needleAngle) * (r - size * 0.02);

      ctx.save();
      ctx.shadowColor = needleColors.glow;
      ctx.shadowBlur = size * 0.08;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(nx, ny);
      ctx.strokeStyle = needleColors.stroke;
      ctx.lineWidth = size * 0.025;
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.restore();

      // Center circle
      ctx.beginPath();
      ctx.arc(cx, cy, size * 0.08, 0, Math.PI * 2);
      ctx.fillStyle = '#1a1a24';
      ctx.fill();
      ctx.strokeStyle = '#2a2a3a';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Mass value text
      ctx.font = `bold ${size * 0.16}px "JetBrains Mono", monospace`;
      ctx.fillStyle = needleColors.stroke;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(m.toFixed(2), cx, cy + size * 0.02);

      // Bezel decoration — tick marks
      for (let i = 0; i <= 8; i++) {
        const tickFrac = i / 8;
        const tickAngle = startAngle + tickFrac * totalAngle;
        const innerR = r - size * 0.12;
        const outerR = r - size * 0.06;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(tickAngle) * innerR, cy + Math.sin(tickAngle) * innerR);
        ctx.lineTo(cx + Math.cos(tickAngle) * outerR, cy + Math.sin(tickAngle) * outerR);
        ctx.strokeStyle = '#3a3a4a';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    function animate() {
      const diff = targetMassRef.current - currentMassRef.current;
      if (Math.abs(diff) > 0.001) {
        currentMassRef.current += diff * 0.15;
      } else {
        currentMassRef.current = targetMassRef.current;
      }
      draw(currentMassRef.current);
      animRef.current = requestAnimationFrame(animate);
    }

    animRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animRef.current);
  }, [size]);

  const colors = getMassColor(mass);

  return (
    <div className="flex flex-col items-center gap-1">
      <div
        className="relative rounded-full"
        style={{
          background: 'radial-gradient(circle at 30% 30%, #2a2a3a, #0f0f14)',
          boxShadow: `0 0 0 2px #2a2a3a, inset 0 2px 4px rgba(0,0,0,0.5)`,
          padding: '4px',
        }}
      >
        <canvas
          ref={canvasRef}
          style={{ width: size, height: size, display: 'block' }}
        />
        {active && (
          <div
            className="absolute inset-0 rounded-full pointer-events-none animate-pulse-glow"
            style={{ boxShadow: `0 0 ${size * 0.15}px ${colors.glow}` }}
          />
        )}
      </div>
      <div className="flex items-center gap-1.5">
        <div
          className={`w-1.5 h-1.5 rounded-full ${active ? 'animate-pulse' : ''}`}
          style={{ backgroundColor: active ? colors.stroke : '#4b5563' }}
        />
        <span className="text-xs font-mono text-gray-400">{timeframe}</span>
        {sym && <span className="text-xs font-mono text-gray-500">· {sym}</span>}
      </div>
    </div>
  );
}
