// ============================================================
// components/AlertPanel.tsx -- Scrollable alert list with
// severity badges, timestamps, and acknowledge actions.
// Used for VaR breach alerts and circuit breaker events.
// ============================================================

import React, { useState, useCallback } from 'react';
import { clsx } from 'clsx';
import { AlertTriangle, Info, XCircle, CheckCircle, X, Bell, BellOff } from 'lucide-react';
import { formatDistanceToNow, parseISO } from 'date-fns';
import type { AlertSeverity } from '../types/risk';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AlertItem {
  id: string;
  severity: AlertSeverity;
  title: string;
  message: string;
  timestamp: string;   // ISO8601
  acknowledged: boolean;
  value?: number;
  threshold?: number;
  source?: string;
}

export interface AlertPanelProps {
  alerts: AlertItem[];
  /** Called when user acknowledges an alert. */
  onAcknowledge?: (id: string) => void;
  /** Called when user dismisses (removes) an alert. */
  onDismiss?: (id: string) => void;
  /** Max items shown before scrolling -- default 8. */
  maxVisible?: number;
  /** Show header with count badge. */
  showHeader?: boolean;
  title?: string;
  className?: string;
  emptyMessage?: string;
}

// ---------------------------------------------------------------------------
// Severity config
// ---------------------------------------------------------------------------

type SeverityConfig = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  icon: React.ComponentType<any>;
  badgeClass: string;
  borderClass: string;
  bgClass: string;
  label: string;
};

const SEVERITY_CONFIG: Record<AlertSeverity, SeverityConfig> = {
  info: {
    icon: Info,
    badgeClass: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    borderClass: 'border-l-blue-500',
    bgClass: 'bg-blue-500/5',
    label: 'INFO',
  },
  warn: {
    icon: AlertTriangle,
    badgeClass: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    borderClass: 'border-l-amber-500',
    bgClass: 'bg-amber-500/5',
    label: 'WARN',
  },
  critical: {
    icon: XCircle,
    badgeClass: 'bg-red-500/20 text-red-400 border-red-500/30',
    borderClass: 'border-l-red-500',
    bgClass: 'bg-red-500/5',
    label: 'CRIT',
  },
};

// ---------------------------------------------------------------------------
// Single alert row
// ---------------------------------------------------------------------------

interface AlertRowProps {
  alert: AlertItem;
  onAcknowledge?: (id: string) => void;
  onDismiss?: (id: string) => void;
}

const AlertRow: React.FC<AlertRowProps> = ({ alert, onAcknowledge, onDismiss }) => {
  const cfg = SEVERITY_CONFIG[alert.severity];
  const Icon = cfg.icon;

  let timeAgo: string;
  try {
    timeAgo = formatDistanceToNow(parseISO(alert.timestamp), { addSuffix: true });
  } catch {
    timeAgo = alert.timestamp;
  }

  return (
    <div
      className={clsx(
        'relative flex gap-3 p-3 rounded-r border-l-2 transition-opacity',
        cfg.borderClass,
        cfg.bgClass,
        alert.acknowledged && 'opacity-50'
      )}
    >
      <div className="flex-shrink-0 mt-0.5">
        <Icon size={15} className={clsx('', cfg.badgeClass.split(' ')[1])} />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={clsx('text-[10px] font-bold px-1.5 py-0.5 rounded border', cfg.badgeClass)}>
              {cfg.label}
            </span>
            {alert.source && (
              <span className="text-[10px] text-slate-500 font-mono">{alert.source}</span>
            )}
          </div>
          <div className="flex items-center gap-1 flex-shrink-0">
            {!alert.acknowledged && onAcknowledge && (
              <button
                onClick={() => onAcknowledge(alert.id)}
                title="Acknowledge"
                className="p-0.5 hover:text-green-400 text-slate-500 transition-colors"
              >
                <CheckCircle size={13} />
              </button>
            )}
            {onDismiss && (
              <button
                onClick={() => onDismiss(alert.id)}
                title="Dismiss"
                className="p-0.5 hover:text-slate-300 text-slate-600 transition-colors"
              >
                <X size={13} />
              </button>
            )}
          </div>
        </div>

        {alert.title && (
          <p className="text-xs font-semibold text-slate-200 mt-1 leading-snug">{alert.title}</p>
        )}
        <p className="text-xs text-slate-400 mt-0.5 leading-snug">{alert.message}</p>

        {(alert.value != null || alert.threshold != null) && (
          <div className="flex gap-3 mt-1.5 text-[11px] font-mono">
            {alert.value != null && (
              <span className="text-slate-300">
                value: <span className="text-amber-400">{alert.value.toLocaleString()}</span>
              </span>
            )}
            {alert.threshold != null && (
              <span className="text-slate-300">
                limit: <span className="text-slate-400">{alert.threshold.toLocaleString()}</span>
              </span>
            )}
          </div>
        )}

        <p className="text-[10px] text-slate-600 mt-1">{timeAgo}</p>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Panel component
// ---------------------------------------------------------------------------

export const AlertPanel: React.FC<AlertPanelProps> = ({
  alerts,
  onAcknowledge,
  onDismiss,
  maxVisible = 8,
  showHeader = true,
  title = 'Alerts',
  className,
  emptyMessage = 'No active alerts',
}) => {
  const [localAlerts, setLocalAlerts] = useState<AlertItem[]>(alerts);

  // Sync external updates
  React.useEffect(() => {
    setLocalAlerts(alerts);
  }, [alerts]);

  const handleAck = useCallback((id: string) => {
    setLocalAlerts(prev => prev.map(a => a.id === id ? { ...a, acknowledged: true } : a));
    onAcknowledge?.(id);
  }, [onAcknowledge]);

  const handleDismiss = useCallback((id: string) => {
    setLocalAlerts(prev => prev.filter(a => a.id !== id));
    onDismiss?.(id);
  }, [onDismiss]);

  const critical = localAlerts.filter(a => a.severity === 'critical' && !a.acknowledged).length;
  const warn     = localAlerts.filter(a => a.severity === 'warn'     && !a.acknowledged).length;
  const total    = localAlerts.filter(a => !a.acknowledged).length;

  return (
    <div className={clsx('flex flex-col', className)}>
      {showHeader && (
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {total > 0 ? (
              <Bell size={14} className="text-amber-400" />
            ) : (
              <BellOff size={14} className="text-slate-600" />
            )}
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              {title}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            {critical > 0 && (
              <span className="text-[10px] font-bold bg-red-500/20 text-red-400 border border-red-500/30 px-1.5 py-0.5 rounded">
                {critical} CRIT
              </span>
            )}
            {warn > 0 && (
              <span className="text-[10px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30 px-1.5 py-0.5 rounded">
                {warn} WARN
              </span>
            )}
          </div>
        </div>
      )}

      {localAlerts.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-6 text-slate-600 gap-2">
          <CheckCircle size={20} />
          <span className="text-xs">{emptyMessage}</span>
        </div>
      ) : (
        <div
          className="flex flex-col gap-1.5 overflow-y-auto"
          style={{ maxHeight: maxVisible * 76 }}
        >
          {localAlerts
            .slice()
            .sort((a, b) => {
              // Critical first, then warn, then info. Within same severity: unacknowledged first.
              const sevOrder = { critical: 0, warn: 1, info: 2 };
              const sev = sevOrder[a.severity] - sevOrder[b.severity];
              if (sev !== 0) return sev;
              if (a.acknowledged !== b.acknowledged) return a.acknowledged ? 1 : -1;
              return b.timestamp.localeCompare(a.timestamp);
            })
            .map(alert => (
              <AlertRow
                key={alert.id}
                alert={alert}
                onAcknowledge={onAcknowledge ? handleAck : undefined}
                onDismiss={onDismiss ? handleDismiss : undefined}
              />
            ))}
        </div>
      )}
    </div>
  );
};
