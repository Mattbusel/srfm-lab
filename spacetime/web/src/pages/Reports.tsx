import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { api } from '../api';

interface ReportMeta {
  generatedAt: string;
  runs: string[];
  includedMC: boolean;
  includedSensitivity: boolean;
  fileSizeKB: number;
}

export function Reports() {
  const [runNames, setRunNames] = useState<string[]>([]);
  const [newRunName, setNewRunName] = useState('');
  const [includeMC, setIncludeMC] = useState(true);
  const [includeSensitivity, setIncludeSensitivity] = useState(true);
  const [lastReport, setLastReport] = useState<ReportMeta | null>(null);

  const reportMut = useMutation({
    mutationFn: () => api.report({
      run_names: runNames,
      include_mc: includeMC,
      include_sensitivity: includeSensitivity,
    }),
    onSuccess: async (blob) => {
      // Auto-download
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `spacetime_report_${new Date().toISOString().split('T')[0]}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setLastReport({
        generatedAt: new Date().toISOString(),
        runs: runNames,
        includedMC: includeMC,
        includedSensitivity: includeSensitivity,
        fileSizeKB: Math.round(blob.size / 1024),
      });
    },
  });

  function addRunName() {
    if (newRunName.trim() && !runNames.includes(newRunName.trim())) {
      setRunNames(prev => [...prev, newRunName.trim()]);
      setNewRunName('');
    }
  }

  function removeRunName(name: string) {
    setRunNames(prev => prev.filter(r => r !== name));
  }

  const sections = [
    { id: 'executive_summary', label: 'Executive Summary', always: true },
    { id: 'equity_curve', label: 'Equity Curve', always: true },
    { id: 'trade_list', label: 'Trade List', always: true },
    { id: 'metrics_table', label: 'Performance Metrics', always: true },
    { id: 'mc', label: 'Monte Carlo Analysis', state: includeMC, setState: setIncludeMC },
    { id: 'sensitivity', label: 'Parameter Sensitivity', state: includeSensitivity, setState: setIncludeSensitivity },
  ];

  return (
    <div className="space-y-4 max-w-4xl mx-auto">
      {/* Header */}
      <div className="bg-accent/5 border border-accent/30 rounded-xl p-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-accent/20 border border-accent/40 flex items-center justify-center">
            <svg className="w-4 h-4 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
              />
            </svg>
          </div>
          <div>
            <h2 className="text-sm font-mono font-semibold text-gray-200">PDF Report Generator</h2>
            <p className="text-xs font-mono text-gray-500">Generate investor-ready research reports for LARSA backtest runs</p>
          </div>
        </div>
      </div>

      {/* Run selection */}
      <div className="bg-bg-card border border-bg-border rounded-xl p-5">
        <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">Select Backtest Runs</h3>

        <div className="flex gap-2 mb-3">
          <input
            className="flex-1 bg-bg-elevated border border-bg-border rounded-lg px-3 py-2 text-sm font-mono text-gray-200 focus:outline-none focus:border-accent"
            placeholder="Enter run ID or name (e.g. BT_SPY_2020)"
            value={newRunName}
            onChange={e => setNewRunName(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && addRunName()}
          />
          <button
            onClick={addRunName}
            className="px-4 py-2 bg-accent hover:bg-accent-hover rounded-lg text-sm font-mono font-semibold text-white transition-colors"
          >
            + Add
          </button>
        </div>

        {runNames.length === 0 ? (
          <p className="text-xs font-mono text-gray-600 text-center py-4">
            No runs selected. Add backtest run IDs above.
          </p>
        ) : (
          <div className="space-y-1">
            {runNames.map(name => (
              <div key={name} className="flex items-center justify-between bg-bg-elevated rounded-lg px-3 py-2">
                <span className="text-sm font-mono text-gray-200">{name}</span>
                <button
                  onClick={() => removeRunName(name)}
                  className="text-gray-600 hover:text-bear transition-colors text-xs"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Section checklist */}
      <div className="bg-bg-card border border-bg-border rounded-xl p-5">
        <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-3">Report Sections</h3>
        <div className="space-y-2">
          {sections.map(section => (
            <div key={section.id} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-bg-elevated">
              <div
                className={`w-4 h-4 rounded border flex items-center justify-center cursor-pointer transition-colors ${
                  section.always || section.state
                    ? 'bg-accent border-accent'
                    : 'bg-transparent border-bg-border hover:border-gray-500'
                }`}
                onClick={() => !section.always && section.setState?.(!section.state)}
              >
                {(section.always || section.state) && (
                  <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                  </svg>
                )}
              </div>
              <span className="text-sm font-mono text-gray-200">{section.label}</span>
              {section.always && <span className="text-xs font-mono text-gray-600 ml-auto">Always included</span>}
            </div>
          ))}
        </div>
      </div>

      {/* Generate button */}
      <div className="bg-bg-card border border-bg-border rounded-xl p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-mono text-gray-300">
              {runNames.length} run{runNames.length !== 1 ? 's' : ''} selected ·{' '}
              {sections.filter(s => s.always || s.state).length} sections
            </p>
            <p className="text-xs font-mono text-gray-500 mt-0.5">
              Output: PDF with embedded charts and tables
            </p>
          </div>
          <button
            onClick={() => reportMut.mutate()}
            disabled={reportMut.isPending || runNames.length === 0}
            className="px-6 py-3 bg-accent hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-mono font-semibold text-white transition-colors"
          >
            {reportMut.isPending ? (
              <span className="flex items-center gap-2">
                <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Generating…
              </span>
            ) : '⬇ Generate Report PDF'}
          </button>
        </div>
        {reportMut.isError && (
          <p className="mt-2 text-xs font-mono text-bear">{(reportMut.error as Error).message}</p>
        )}
      </div>

      {/* Last report preview */}
      {lastReport && (
        <div className="bg-bull/5 border border-bull/30 rounded-xl p-5">
          <h3 className="text-xs font-mono text-bull uppercase tracking-wider mb-3">Last Generated Report</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Generated', value: new Date(lastReport.generatedAt).toLocaleString() },
              { label: 'Runs Included', value: String(lastReport.runs.length) },
              { label: 'File Size', value: `~${lastReport.fileSizeKB} KB` },
              { label: 'Sections', value: [
                'Exec Summary', 'Equity', 'Trades', 'Metrics',
                lastReport.includedMC ? 'MC' : null,
                lastReport.includedSensitivity ? 'Sensitivity' : null,
              ].filter(Boolean).join(', ') },
            ].map(({ label, value }) => (
              <div key={label}>
                <p className="text-xs font-mono text-gray-500">{label}</p>
                <p className="text-sm font-mono text-gray-200 mt-0.5">{value}</p>
              </div>
            ))}
          </div>
          {lastReport.runs.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {lastReport.runs.map(r => (
                <span key={r} className="text-xs font-mono bg-bg-elevated border border-bg-border rounded px-2 py-0.5 text-gray-300">
                  {r}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
