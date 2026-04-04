package dashboard

// DashboardHTML is the self-contained monitoring dashboard.
// It uses vanilla JS + Chart.js from CDN, no build step required.
const DashboardHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SRFM Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --orange: #db6d28;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace; font-size: 13px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 16px; font-weight: 600; color: var(--accent); letter-spacing: 0.5px; }
  #status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; margin-right: 6px; }
  #status-dot.offline { background: var(--red); }
  .grid { display: grid; gap: 16px; padding: 16px; }
  .grid-2 { grid-template-columns: repeat(2, 1fr); }
  .grid-4 { grid-template-columns: repeat(4, 1fr); }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 16px; }
  .card h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 12px; }
  .stat-val { font-size: 24px; font-weight: 600; font-family: monospace; }
  .stat-val.positive { color: var(--green); }
  .stat-val.negative { color: var(--red); }
  .stat-label { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
  .chart-wrap { position: relative; height: 200px; }
  .chart-wrap.tall { height: 300px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; padding: 6px 8px; color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d; }
  tr:last-child td { border-bottom: none; }
  .badge { display: inline-block; padding: 2px 6px; border-radius: 10px; font-size: 10px; font-weight: 600; }
  .badge.info { background: #1f4068; color: var(--accent); }
  .badge.warning { background: #3d2a00; color: var(--yellow); }
  .badge.critical { background: #3d0000; color: var(--red); }
  .mass-bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-top: 4px; }
  .mass-bar-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.5s ease; }
  .mass-bar-fill.high { background: var(--red); }
  .mass-bar-fill.med { background: var(--yellow); }
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }
  #last-updated { font-size: 11px; color: var(--text-muted); }
  .no-data { color: var(--text-muted); font-style: italic; padding: 12px 0; text-align: center; }
</style>
</head>
<body>
<header>
  <div>
    <span id="status-dot"></span>
    <h1 style="display:inline">SRFM Monitor</h1>
  </div>
  <span id="last-updated">connecting...</span>
</header>

<div class="grid grid-4">
  <div class="card" id="card-equity">
    <h2>Equity</h2>
    <div class="stat-val" id="equity-val">—</div>
    <div class="stat-label">Portfolio value</div>
  </div>
  <div class="card">
    <h2>Daily P&amp;L</h2>
    <div class="stat-val" id="pnl-val">—</div>
    <div class="stat-label">Since open</div>
  </div>
  <div class="card">
    <h2>Intraday Drawdown</h2>
    <div class="stat-val" id="dd-val">—</div>
    <div class="stat-label">From daily high</div>
  </div>
  <div class="card">
    <h2>Open Positions</h2>
    <div class="stat-val" id="pos-count">—</div>
    <div class="stat-label">Active positions</div>
  </div>
</div>

<div class="grid grid-2">
  <div class="card">
    <h2>Equity Curve (30d)</h2>
    <div class="chart-wrap tall"><canvas id="equityChart"></canvas></div>
  </div>
  <div class="card">
    <h2>BH Mass States</h2>
    <div id="bh-table-wrap">
      <p class="no-data">Loading...</p>
    </div>
  </div>
</div>

<div class="grid grid-2">
  <div class="card">
    <h2>Positions</h2>
    <div id="positions-wrap">
      <p class="no-data">No open positions</p>
    </div>
  </div>
  <div class="card">
    <h2>Recent Alerts</h2>
    <div id="alerts-wrap">
      <p class="no-data">No alerts</p>
    </div>
  </div>
</div>

<script>
const fmt = {
  currency: v => v == null ? '—' : '$' + v.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}),
  pct: v => v == null ? '—' : v.toFixed(2) + '%',
  num: v => v == null ? '—' : v.toLocaleString(),
  ts: s => s ? new Date(s).toLocaleTimeString() : '—',
};

// --- Equity Chart ---
const equityCtx = document.getElementById('equityChart').getContext('2d');
const equityChart = new Chart(equityCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Equity',
      data: [],
      borderColor: '#58a6ff',
      backgroundColor: 'rgba(88,166,255,0.08)',
      fill: true,
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 1.5,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { display: false },
      y: {
        grid: { color: '#21262d' },
        ticks: { color: '#8b949e', font: { size: 11 } }
      }
    }
  }
});

function updateEquityChart(curve) {
  if (!curve || curve.length === 0) return;
  const last = curve.slice(-500);
  equityChart.data.labels = last.map(p => new Date(p.timestamp).toLocaleDateString());
  equityChart.data.datasets[0].data = last.map(p => p.equity);
  equityChart.update('none');
}

function updateState(data) {
  document.getElementById('last-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();

  // Portfolio stats.
  const p = data.portfolio;
  if (p) {
    const equityEl = document.getElementById('equity-val');
    equityEl.textContent = fmt.currency(p.equity || p.Equity);
    equityEl.className = 'stat-val';

    const pnl = p.daily_pnl ?? p.DailyPnL;
    const pnlEl = document.getElementById('pnl-val');
    pnlEl.textContent = fmt.currency(pnl);
    pnlEl.className = 'stat-val ' + (pnl >= 0 ? 'positive' : 'negative');

    const dd = p.intraday_dd ?? p.IntradayDD;
    const ddEl = document.getElementById('dd-val');
    ddEl.textContent = fmt.pct(dd);
    ddEl.className = 'stat-val ' + (dd > 1 ? 'negative' : '');

    const positions = p.positions ?? p.Positions ?? {};
    const posCount = Object.keys(positions).length;
    document.getElementById('pos-count').textContent = posCount;

    // Positions table.
    if (posCount > 0) {
      let html = '<table><thead><tr><th>Symbol</th><th>Qty</th><th>Avg Cost</th><th>Mkt Val</th><th>Unreal P&L</th><th>Side</th></tr></thead><tbody>';
      for (const [sym, pos] of Object.entries(positions)) {
        const unrealPnl = pos.unreal_pnl ?? pos.UnrealPnL ?? 0;
        html += '<tr>' +
          '<td>' + sym + '</td>' +
          '<td>' + (pos.qty ?? pos.Qty ?? 0) + '</td>' +
          '<td>' + fmt.currency(pos.avg_cost ?? pos.AvgCost) + '</td>' +
          '<td>' + fmt.currency(pos.market_val ?? pos.MarketVal) + '</td>' +
          '<td class="' + (unrealPnl >= 0 ? 'pnl-pos' : 'pnl-neg') + '">' + fmt.currency(unrealPnl) + '</td>' +
          '<td>' + (pos.side ?? pos.Side ?? '') + '</td>' +
          '</tr>';
      }
      html += '</tbody></table>';
      document.getElementById('positions-wrap').innerHTML = html;
    } else {
      document.getElementById('positions-wrap').innerHTML = '<p class="no-data">No open positions</p>';
    }
  }

  // Equity curve.
  if (data.equity_curve) {
    updateEquityChart(data.equity_curve);
  }

  // BH masses.
  const masses = data.bh_masses;
  if (masses && Object.keys(masses).length > 0) {
    let html = '<table><thead><tr><th>Symbol|TF</th><th>Mass</th><th>Bar</th></tr></thead><tbody>';
    const sorted = Object.entries(masses).sort((a, b) => b[1] - a[1]);
    for (const [key, mass] of sorted) {
      const cls = mass >= 0.7 ? 'high' : (mass >= 0.4 ? 'med' : '');
      html += '<tr>' +
        '<td>' + key + '</td>' +
        '<td>' + mass.toFixed(4) + '</td>' +
        '<td><div class="mass-bar"><div class="mass-bar-fill ' + cls + '" style="width:' + Math.min(100, mass * 100).toFixed(1) + '%"></div></div></td>' +
        '</tr>';
    }
    html += '</tbody></table>';
    document.getElementById('bh-table-wrap').innerHTML = html;
  } else {
    document.getElementById('bh-table-wrap').innerHTML = '<p class="no-data">No BH data</p>';
  }

  // Alerts.
  const alerts = data.recent_alerts;
  if (alerts && alerts.length > 0) {
    let html = '<table><thead><tr><th>Time</th><th>Level</th><th>Rule</th><th>Symbol</th><th>Value</th></tr></thead><tbody>';
    const recent = [...alerts].reverse().slice(0, 20);
    for (const a of recent) {
      const level = a.level ?? a.Level ?? 'info';
      html += '<tr>' +
        '<td>' + fmt.ts(a.fired_at ?? a.FiredAt) + '</td>' +
        '<td><span class="badge ' + level + '">' + level.toUpperCase() + '</span></td>' +
        '<td>' + (a.rule?.name ?? a.Rule?.Name ?? '') + '</td>' +
        '<td>' + (a.symbol ?? a.Symbol ?? '') + '</td>' +
        '<td>' + ((a.value ?? a.Value ?? 0).toFixed(4)) + '</td>' +
        '</tr>';
    }
    html += '</tbody></table>';
    document.getElementById('alerts-wrap').innerHTML = html;
  } else {
    document.getElementById('alerts-wrap').innerHTML = '<p class="no-data">No alerts</p>';
  }
}

// --- SSE connection ---
let evtSource = null;
const statusDot = document.getElementById('status-dot');

function connect() {
  evtSource = new EventSource('/events');

  evtSource.onopen = () => {
    statusDot.className = '';
    console.log('SSE connected');
  };

  evtSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      updateState(data);
    } catch (err) {
      console.warn('SSE parse error', err);
    }
  };

  evtSource.onerror = () => {
    statusDot.className = 'offline';
    document.getElementById('last-updated').textContent = 'Disconnected — retrying...';
    evtSource.close();
    setTimeout(connect, 5000);
  };
}

connect();
</script>
</body>
</html>`
