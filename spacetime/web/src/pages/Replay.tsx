import { useState, useEffect, useRef, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import type { ReplayBar, Regime } from '../types';
import { useWebSocket } from '../hooks/useWebSocket';
import { RegimeBadge } from '../components/RegimeBadge';
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

const FAMOUS_EVENTS = [
  { label: 'Volmageddon', date: '2018-02-01', description: 'Feb 2018 Vol Spike' },
  { label: 'COVID Crash', date: '2020-02-20', description: 'Feb 2020 Pandemic Crash' },
  { label: '2022 Bear Market', date: '2022-01-03', description: '2022 Rate-Hike Bear Market' },
  { label: 'FTX Collapse', date: '2022-11-01', description: 'Nov 2022 FTX Implosion' },
];

const SPEEDS = [1, 5, 20, 100];

function BHFormationMeter({ mass, active }: { mass: number; active: boolean }) {
  const frac = Math.min(mass / 2.0, 1);
  const color = mass >= 1.8 ? '#ef4444' : mass >= 1.5 ? '#f97316' : mass >= 1.2 ? '#eab308' : '#6b7280';
  const glowColor = mass >= 1.8 ? 'rgba(239,68,68,0.5)' : mass >= 1.5 ? 'rgba(249,115,22,0.4)' : 'rgba(234,179,8,0.3)';

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs font-mono">
        <span className="text-gray-500">BH Formation Meter</span>
        <div className="flex items-center gap-1.5">
          <span style={{ color }}>{mass.toFixed(3)}</span>
          {active && <span className="text-bull animate-pulse text-xs">● ACTIVE</span>}
        </div>
      </div>
      <div className="h-4 bg-bg-border rounded-full overflow-hidden relative">
        {/* Zone markers */}
        {[0.6, 0.75, 0.9].map(frac => (
          <div
            key={frac}
            className="absolute top-0 w-px h-full bg-gray-700 opacity-50"
            style={{ left: `${frac * 100}%` }}
          />
        ))}
        <div
          className="h-full rounded-full transition-all duration-300 ease-out"
          style={{
            width: `${frac * 100}%`,
            backgroundColor: color,
            boxShadow: mass >= 1.2 ? `0 0 8px ${glowColor}` : 'none',
          }}
        />
      </div>
      <div className="flex justify-between text-xs font-mono text-gray-700">
        <span>0.0</span>
        <span className="text-gray-600">1.2</span>
        <span className="text-bh-warm">1.5</span>
        <span className="text-bh-hot">1.8</span>
        <span className="text-bear">2.0</span>
      </div>
    </div>
  );
}

export function Replay() {
  const instrumentsQuery = useQuery({
    queryKey: ['instruments'],
    queryFn: api.instruments,
    staleTime: 60_000,
  });

  const syms = instrumentsQuery.data?.map(i => i.sym) ?? ['SPY', 'QQQ', 'BTC', 'ETH'];

  const [sym, setSym] = useState('SPY');
  const [startDate, setStartDate] = useState('2020-02-20');
  const [endDate, setEndDate] = useState('2020-04-01');
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [connected, setConnected] = useState(false);
  const [bars, setBars] = useState<ReplayBar[]>([]);
  const [currentBarIdx, setCurrentBarIdx] = useState(0);
  const [totalBars, setTotalBars] = useState(0);

  const handleBar = useCallback((bar: ReplayBar) => {
    setBars(prev => {
      const next = [...prev, bar].slice(-300);
      return next;
    });
    setCurrentBarIdx(bar.bar_idx);
    if (bar.bar_idx === 0) setTotalBars(0);
  }, []);

  const { status: wsStatus, send } = useWebSocket<ReplayBar>({
    url: 'ws://localhost:8000/ws/replay',
    enabled: connected,
    onMessage: handleBar,
  });

  function handleConnect() {
    setBars([]);
    setCurrentBarIdx(0);
    setConnected(true);
    setTimeout(() => {
      send({ sym, start: startDate, end: endDate, speed });
    }, 500);
  }

  function handlePlayPause() {
    setPlaying(!playing);
    send({ command: playing ? 'pause' : 'play' });
  }

  function handleReset() {
    setBars([]);
    setCurrentBarIdx(0);
    setPlaying(false);
    send({ command: 'reset' });
  }

  function handleFamousEvent(evt: typeof FAMOUS_EVENTS[0]) {
    setStartDate(evt.date);
    const end = new Date(evt.date);
    end.setMonth(end.getMonth() + 2);
    setEndDate(end.toISOString().split('T')[0]);
  }

  function handleScrub(barIdx: number) {
    setCurrentBarIdx(barIdx);
    send({ command: 'seek', bar_idx: barIdx });
  }

  const latestBar = bars[bars.length - 1];
  const displayBars = bars.slice(-150);

  // Build candlestick-compatible data
  const chartData = displayBars.map(b => ({
    time: new Date(b.timestamp).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
    price: b.price,
    bh_mass: b.bh_mass,
    active: b.bh_active ? b.price : null,
    regime: b.regime,
    position: b.position_frac > 0 ? b.price : null,
  }));

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Controls */}
      <div className="bg-bg-card border border-bg-border rounded-xl p-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Instrument</label>
            <select
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={sym}
              onChange={e => setSym(e.target.value)}
            >
              {syms.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Start Date</label>
            <input type="date"
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">End Date</label>
            <input type="date"
              className="w-full bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-xs font-mono text-gray-500 mb-1">Speed</label>
            <div className="flex gap-1">
              {SPEEDS.map(s => (
                <button
                  key={s}
                  onClick={() => { setSpeed(s); send({ command: 'speed', value: s }); }}
                  className={`flex-1 py-2 rounded-lg text-xs font-mono font-semibold transition-colors
                    ${speed === s ? 'bg-accent text-white' : 'bg-bg-elevated text-gray-400 hover:text-gray-200'}`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Famous events */}
        <div className="flex flex-wrap gap-2 mb-4">
          <span className="text-xs font-mono text-gray-600 self-center">Famous Events:</span>
          {FAMOUS_EVENTS.map(evt => (
            <button
              key={evt.label}
              onClick={() => handleFamousEvent(evt)}
              className="text-xs font-mono px-2.5 py-1 bg-bg-elevated border border-bg-border hover:border-gray-600 rounded-lg text-gray-400 hover:text-gray-200 transition-colors"
              title={evt.description}
            >
              {evt.label}
            </button>
          ))}
        </div>

        {/* Play controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={handleConnect}
            disabled={wsStatus === 'connecting'}
            className="px-4 py-2 bg-accent hover:bg-accent-hover disabled:opacity-50 rounded-lg text-sm font-mono font-semibold text-white transition-colors"
          >
            {wsStatus === 'connecting' ? 'Connecting…' : 'Connect'}
          </button>

          <button
            onClick={handlePlayPause}
            disabled={wsStatus !== 'open'}
            className={`px-4 py-2 rounded-lg text-sm font-mono font-semibold transition-colors disabled:opacity-40
              ${playing ? 'bg-bh-warm text-bg-base' : 'bg-bull/80 hover:bg-bull text-white'}`}
          >
            {playing ? '⏸ Pause' : '▶ Play'}
          </button>

          <button
            onClick={handleReset}
            disabled={wsStatus !== 'open'}
            className="px-4 py-2 bg-bg-elevated hover:bg-bg-border disabled:opacity-40 rounded-lg text-sm font-mono text-gray-400 transition-colors"
          >
            ⏮ Reset
          </button>

          <div className="flex items-center gap-1.5 ml-2">
            <div className={`w-2 h-2 rounded-full ${wsStatus === 'open' ? 'bg-bull animate-pulse' : wsStatus === 'connecting' ? 'bg-bh-warm animate-pulse' : 'bg-gray-600'}`} />
            <span className="text-xs font-mono text-gray-500">{wsStatus}</span>
          </div>
        </div>
      </div>

      {bars.length > 0 && (
        <>
          {/* Main chart + side panel */}
          <div className="grid grid-cols-4 gap-4">
            {/* Chart */}
            <div className="col-span-3 bg-bg-card border border-bg-border rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-mono font-bold text-gray-200">{sym}</span>
                  {latestBar && <RegimeBadge regime={latestBar.regime as Regime} />}
                </div>
                {latestBar && (
                  <span className="text-sm font-mono font-bold text-gray-200">
                    ${latestBar.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </span>
                )}
              </div>

              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3a" />
                  <XAxis dataKey="time" tick={{ fill: '#6b7280', fontSize: 9 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} interval={Math.max(1, Math.floor(chartData.length / 10))} />
                  <YAxis yAxisId="price" orientation="left" tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
                  <YAxis yAxisId="mass" orientation="right" domain={[0, 2]} tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} width={35} />
                  <Tooltip
                    contentStyle={{ background: '#1a1a24', border: '1px solid #2a2a3a', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                    labelStyle={{ color: '#9ca3af' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                  <Line yAxisId="price" type="monotone" dataKey="price" stroke="#6366f1" strokeWidth={1.5} dot={false} isAnimationActive={false} name="Price" />
                  <Line yAxisId="mass" type="monotone" dataKey="bh_mass" stroke="#eab308" strokeWidth={1.5} dot={false} isAnimationActive={false} name="BH Mass" strokeDasharray="4 2" />
                </ComposedChart>
              </ResponsiveContainer>

              {/* BH Formation Meter */}
              <div className="mt-3 px-2">
                <BHFormationMeter mass={latestBar?.bh_mass ?? 0} active={latestBar?.bh_active ?? false} />
              </div>

              {/* Timeline scrubber */}
              {totalBars > 0 && (
                <div className="mt-3 px-2">
                  <input
                    type="range"
                    min={0}
                    max={totalBars}
                    value={currentBarIdx}
                    className="w-full accent-accent"
                    onChange={e => handleScrub(parseInt(e.target.value))}
                  />
                  <div className="flex justify-between text-xs font-mono text-gray-600">
                    <span>Bar 0</span>
                    <span className="text-accent">Bar {currentBarIdx}</span>
                    <span>Bar {totalBars}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Side panel */}
            <div className="bg-bg-card border border-bg-border rounded-xl p-4">
              <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">Live Stats</h3>
              {latestBar ? (
                <div className="space-y-3">
                  {[
                    { label: 'Bar Index', value: String(latestBar.bar_idx) },
                    { label: 'Price', value: `$${latestBar.price.toLocaleString('en-US', { maximumFractionDigits: 2 })}` },
                    { label: 'Beta', value: latestBar.beta.toFixed(4) },
                    { label: 'Is Timelike', value: latestBar.is_timelike ? '✓ YES' : '✗ NO', cls: latestBar.is_timelike ? 'text-bull' : 'text-bear' },
                    { label: 'BH Mass', value: latestBar.bh_mass.toFixed(4), cls: latestBar.bh_mass >= 1.8 ? 'text-bear' : latestBar.bh_mass >= 1.2 ? 'text-bh-warm' : 'text-gray-300' },
                    { label: 'BH Active', value: latestBar.bh_active ? '✓ YES' : '○ NO', cls: latestBar.bh_active ? 'text-bull' : 'text-gray-500' },
                    { label: 'BH Dir', value: latestBar.bh_dir > 0 ? '↑ LONG' : latestBar.bh_dir < 0 ? '↓ SHORT' : '— FLAT', cls: latestBar.bh_dir > 0 ? 'text-bull' : latestBar.bh_dir < 0 ? 'text-bear' : 'text-gray-500' },
                    { label: 'CTL', value: latestBar.ctl.toFixed(4) },
                    { label: 'Position Frac', value: `${(latestBar.position_frac * 100).toFixed(1)}%`, cls: latestBar.position_frac > 0 ? 'text-accent' : 'text-gray-500' },
                    { label: 'Pos Floor', value: latestBar.pos_floor.toFixed(4) },
                    { label: 'Equity', value: `$${latestBar.equity.toLocaleString('en-US', { maximumFractionDigits: 0 })}` },
                  ].map(({ label, value, cls = 'text-gray-200' }) => (
                    <div key={label} className="flex items-center justify-between">
                      <span className="text-xs font-mono text-gray-500">{label}</span>
                      <span className={`text-xs font-mono font-semibold ${cls}`}>{value}</span>
                    </div>
                  ))}
                  <div className="pt-2 border-t border-bg-border">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono text-gray-500">Regime</span>
                      <RegimeBadge regime={latestBar.regime as Regime} />
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-xs font-mono text-gray-600">No data yet…</p>
              )}
            </div>
          </div>
        </>
      )}

      {bars.length === 0 && wsStatus !== 'connecting' && (
        <div className="bg-bg-card border border-bg-border rounded-xl p-16 text-center">
          <div className="text-4xl mb-3 opacity-30">▶</div>
          <p className="text-gray-500 font-mono text-sm">Select an instrument, date range, and press Connect to begin replay</p>
        </div>
      )}
    </div>
  );
}
