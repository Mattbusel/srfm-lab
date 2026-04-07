// cmd/alerter/engine/rules.go -- Alert rules engine for SRFM Lab.
//
// Defines the Rule interface, AlertContext, and all concrete rule implementations.
// The RuleEngine runs every 60 seconds, deduplicates within a 30-minute window,
// and manages alert lifecycle: firing -> resolved.

package engine

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Severity levels
// ---------------------------------------------------------------------------

// Severity classifies the urgency of a fired alert.
type Severity string

const (
	SeverityInfo      Severity = "INFO"
	SeverityWarning   Severity = "WARNING"
	SeverityCritical  Severity = "CRITICAL"
	SeverityEmergency Severity = "EMERGENCY"
)

// ---------------------------------------------------------------------------
// AlertContext -- snapshot of all system state passed to every rule
// ---------------------------------------------------------------------------

// PositionInfo describes a single open position.
type PositionInfo struct {
	Symbol    string
	Side      string  // "long" | "short"
	SizePct   float64 // fraction of total portfolio value
	Signal    string  // "up" | "down" | "flat"
	BHActive  bool    // buy-and-hold hold signal active
}

// CircuitBreakerState mirrors the CBState enum from the main engine.
type CircuitBreakerState int

const (
	CBOff CircuitBreakerState = 0
	CBOn  CircuitBreakerState = 1
)

// ServiceRestartInfo tracks recent restart events for one service.
type ServiceRestartInfo struct {
	Name        string
	Restarts    int       // restarts in the trailing hour
	LastRestart time.Time
	Healthy     bool
}

// FeedInfo holds liveness info for one market data feed.
type FeedInfo struct {
	Symbol   string
	LastSeen time.Time
	Healthy  bool
}

// AlertContext is the full snapshot of system state handed to every Rule.
type AlertContext struct {
	// Timestamps
	CollectedAt time.Time

	// Portfolio / P&L
	Equity          float64 // current portfolio equity in USD
	PeakEquity      float64 // peak equity since inception / last reset
	DrawdownPct     float64 // (equity - peak) / peak, always <= 0
	DayPnlPct       float64 // today's P&L as fraction of start-of-day equity
	RealizedLossPct float64 // realized loss today as fraction of start-of-day equity

	// VaR
	VaR95Day float64 // 1-day 95% VaR as positive dollar amount
	// EquityDropBars maps bar-count -> equity drop fraction over that many 15-min bars.
	// Key "1" means the most recent 15-min bar.
	EquityDropBars map[int]float64

	// Positions
	Positions []PositionInfo

	// Correlation
	BTCETHCorrelation float64 // rolling 24h Pearson correlation

	// Circuit breaker
	CircuitBreaker CircuitBreakerState

	// IAE parameter freshness
	LastParamUpdateAt time.Time

	// Signal counts
	SignalBullish int // instruments with bullish signal
	SignalBearish int // instruments with bearish signal
	BHActiveCount int // buy-and-hold positions currently active

	// Services
	Services []ServiceRestartInfo

	// Market data feeds
	Feeds []FeedInfo
}

// ---------------------------------------------------------------------------
// Alert -- the output of a Rule.Evaluate call
// ---------------------------------------------------------------------------

// AlertState tracks the lifecycle of an alert.
type AlertState string

const (
	AlertFiring   AlertState = "FIRING"
	AlertResolved AlertState = "RESOLVED"
)

// Alert is produced when a rule condition is satisfied.
type Alert struct {
	ID          string     `json:"id"`
	RuleName    string     `json:"rule_name"`
	Severity    Severity   `json:"severity"`
	State       AlertState `json:"state"`
	Message     string     `json:"message"`
	Detail      string     `json:"detail,omitempty"`
	Value       float64    `json:"value"`
	Threshold   float64    `json:"threshold"`
	FiredAt     time.Time  `json:"fired_at"`
	ResolvedAt  *time.Time `json:"resolved_at,omitempty"`
	Duration    string     `json:"duration,omitempty"`
}

// ---------------------------------------------------------------------------
// Rule interface
// ---------------------------------------------------------------------------

// Rule is the interface every alert rule must implement.
type Rule interface {
	// Name returns the unique identifier for this rule.
	Name() string
	// Evaluate checks the context and returns an Alert if the condition is
	// satisfied, or nil if the condition is clear.
	Evaluate(ctx AlertContext) *Alert
}

// ---------------------------------------------------------------------------
// Helper: build an alert with a generated ID
// ---------------------------------------------------------------------------

func newAlert(name string, sev Severity, msg, detail string, value, threshold float64) *Alert {
	return &Alert{
		ID:        fmt.Sprintf("%s-%d", name, time.Now().UnixMilli()),
		RuleName:  name,
		Severity:  sev,
		State:     AlertFiring,
		Message:   msg,
		Detail:    detail,
		Value:     value,
		Threshold: threshold,
		FiredAt:   time.Now(),
	}
}

// ---------------------------------------------------------------------------
// DrawdownRule
// ---------------------------------------------------------------------------

// DrawdownRule fires at three severity tiers based on portfolio drawdown.
// 5% -> WARNING, 10% -> CRITICAL, 15% -> EMERGENCY.
type DrawdownRule struct{}

func (r *DrawdownRule) Name() string { return "drawdown" }

func (r *DrawdownRule) Evaluate(ctx AlertContext) *Alert {
	dd := ctx.DrawdownPct // negative value, e.g. -0.12 means 12% drawdown

	switch {
	case dd <= -0.15:
		return newAlert(r.Name(), SeverityEmergency,
			fmt.Sprintf("EMERGENCY: portfolio drawdown %.1f%% exceeds 15%% limit", -dd*100),
			fmt.Sprintf("equity=%.2f peak=%.2f", ctx.Equity, ctx.PeakEquity),
			dd, -0.15,
		)
	case dd <= -0.10:
		return newAlert(r.Name(), SeverityCritical,
			fmt.Sprintf("CRITICAL: portfolio drawdown %.1f%% exceeds 10%% limit", -dd*100),
			fmt.Sprintf("equity=%.2f peak=%.2f", ctx.Equity, ctx.PeakEquity),
			dd, -0.10,
		)
	case dd <= -0.05:
		return newAlert(r.Name(), SeverityWarning,
			fmt.Sprintf("WARNING: portfolio drawdown %.1f%% exceeds 5%% limit", -dd*100),
			fmt.Sprintf("equity=%.2f peak=%.2f", ctx.Equity, ctx.PeakEquity),
			dd, -0.05,
		)
	}
	return nil
}

// ---------------------------------------------------------------------------
// VaRBreachRule
// ---------------------------------------------------------------------------

// VaRBreachRule fires when the realized loss today exceeds the 95% 1-day VaR.
type VaRBreachRule struct{}

func (r *VaRBreachRule) Name() string { return "var_breach" }

func (r *VaRBreachRule) Evaluate(ctx AlertContext) *Alert {
	if ctx.VaR95Day <= 0 || ctx.Equity <= 0 {
		return nil
	}
	// RealizedLossPct is negative when losing. Convert to dollar loss.
	realizedLoss := -ctx.RealizedLossPct * ctx.Equity
	if realizedLoss > ctx.VaR95Day {
		return newAlert(r.Name(), SeverityCritical,
			fmt.Sprintf("VaR breach: realized loss $%.0f exceeds 95%% VaR $%.0f",
				realizedLoss, ctx.VaR95Day),
			fmt.Sprintf("realized_loss_pct=%.2f%% var95=%.2f",
				-ctx.RealizedLossPct*100, ctx.VaR95Day),
			realizedLoss, ctx.VaR95Day,
		)
	}
	return nil
}

// ---------------------------------------------------------------------------
// PositionConcentrationRule
// ---------------------------------------------------------------------------

// PositionConcentrationRule fires when any single position exceeds 25% of portfolio.
type PositionConcentrationRule struct{}

func (r *PositionConcentrationRule) Name() string { return "position_concentration" }

func (r *PositionConcentrationRule) Evaluate(ctx AlertContext) *Alert {
	const threshold = 0.25
	for _, pos := range ctx.Positions {
		if pos.SizePct > threshold {
			return newAlert(r.Name(), SeverityWarning,
				fmt.Sprintf("Concentration risk: %s is %.1f%% of portfolio (limit 25%%)",
					pos.Symbol, pos.SizePct*100),
				fmt.Sprintf("symbol=%s side=%s size_pct=%.2f%%",
					pos.Symbol, pos.Side, pos.SizePct*100),
				pos.SizePct, threshold,
			)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// ParameterStaleRule
// ---------------------------------------------------------------------------

// ParameterStaleRule fires when the last IAE parameter update is older than 8 hours.
// This may indicate the genome engine is stuck or disconnected.
type ParameterStaleRule struct{}

func (r *ParameterStaleRule) Name() string { return "parameter_stale" }

func (r *ParameterStaleRule) Evaluate(ctx AlertContext) *Alert {
	const maxAge = 8 * time.Hour
	if ctx.LastParamUpdateAt.IsZero() {
		return nil // no data yet; don't false-fire on startup
	}
	age := time.Since(ctx.LastParamUpdateAt)
	if age > maxAge {
		return newAlert(r.Name(), SeverityWarning,
			fmt.Sprintf("IAE parameters stale: last update %.1f hours ago (limit 8h)", age.Hours()),
			fmt.Sprintf("last_update=%s age=%s",
				ctx.LastParamUpdateAt.Format(time.RFC3339), age.Truncate(time.Minute)),
			age.Seconds(), maxAge.Seconds(),
		)
	}
	return nil
}

// ---------------------------------------------------------------------------
// SignalDivergenceRule
// ---------------------------------------------------------------------------

// SignalDivergenceRule fires when 3+ instruments have conflicting signals while
// all BH positions are active -- half pointing up, half pointing down.
type SignalDivergenceRule struct{}

func (r *SignalDivergenceRule) Name() string { return "signal_divergence" }

func (r *SignalDivergenceRule) Evaluate(ctx AlertContext) *Alert {
	if ctx.BHActiveCount < 3 {
		return nil // not enough active positions to matter
	}
	bullish := ctx.SignalBullish
	bearish := ctx.SignalBearish
	total := bullish + bearish
	if total < 3 {
		return nil
	}
	// Divergence: both sides have at least 30% representation.
	if bullish == 0 || bearish == 0 {
		return nil
	}
	minSide := bullish
	if bearish < minSide {
		minSide = bearish
	}
	divergencePct := float64(minSide) / float64(total)
	if divergencePct >= 0.30 {
		return newAlert(r.Name(), SeverityWarning,
			fmt.Sprintf("Signal divergence: %d bullish vs %d bearish across %d BH positions",
				bullish, bearish, ctx.BHActiveCount),
			fmt.Sprintf("bullish=%d bearish=%d total=%d divergence_pct=%.1f%%",
				bullish, bearish, total, divergencePct*100),
			divergencePct, 0.30,
		)
	}
	return nil
}

// ---------------------------------------------------------------------------
// VelocityRule
// ---------------------------------------------------------------------------

// VelocityRule fires when equity drops more than 2% in a single 15-minute bar.
type VelocityRule struct{}

func (r *VelocityRule) Name() string { return "velocity" }

func (r *VelocityRule) Evaluate(ctx AlertContext) *Alert {
	const threshold = 0.02 // 2% single-bar drop
	if ctx.EquityDropBars == nil {
		return nil
	}
	drop, ok := ctx.EquityDropBars[1] // most recent 15-min bar
	if !ok {
		return nil
	}
	// drop is negative when falling
	if drop <= -threshold {
		return newAlert(r.Name(), SeverityCritical,
			fmt.Sprintf("Velocity alert: equity dropped %.2f%% in last 15-min bar (limit 2%%)",
				-drop*100),
			fmt.Sprintf("equity=%.2f bar_drop_pct=%.4f", ctx.Equity, drop),
			drop, -threshold,
		)
	}
	return nil
}

// ---------------------------------------------------------------------------
// CorrelationBreakdownRule
// ---------------------------------------------------------------------------

// CorrelationBreakdownRule fires when BTC-ETH rolling correlation drops below 0.5.
// A correlation this low is unusual and may indicate regime change or data issues.
type CorrelationBreakdownRule struct{}

func (r *CorrelationBreakdownRule) Name() string { return "correlation_breakdown" }

func (r *CorrelationBreakdownRule) Evaluate(ctx AlertContext) *Alert {
	const threshold = 0.5
	// Only fire when we have a meaningful sample (non-zero correlation present).
	if ctx.BTCETHCorrelation == 0 {
		return nil
	}
	if ctx.BTCETHCorrelation < threshold {
		return newAlert(r.Name(), SeverityWarning,
			fmt.Sprintf("BTC-ETH correlation %.3f below threshold 0.5 -- unusual decoupling",
				ctx.BTCETHCorrelation),
			fmt.Sprintf("btc_eth_corr=%.4f threshold=%.1f", ctx.BTCETHCorrelation, threshold),
			ctx.BTCETHCorrelation, threshold,
		)
	}
	return nil
}

// ---------------------------------------------------------------------------
// FeedStalenessRule
// ---------------------------------------------------------------------------

// FeedStalenessRule fires when any market data feed has been silent for > 30 minutes.
type FeedStalenessRule struct{}

func (r *FeedStalenessRule) Name() string { return "feed_staleness" }

func (r *FeedStalenessRule) Evaluate(ctx AlertContext) *Alert {
	const maxSilence = 30 * time.Minute
	for _, feed := range ctx.Feeds {
		if feed.LastSeen.IsZero() {
			continue // never seen; don't false-fire
		}
		silence := time.Since(feed.LastSeen)
		if silence > maxSilence {
			return newAlert(r.Name(), SeverityCritical,
				fmt.Sprintf("Market data feed %q silent for %.0f minutes (limit 30)",
					feed.Symbol, silence.Minutes()),
				fmt.Sprintf("feed=%s last_seen=%s silence=%s",
					feed.Symbol,
					feed.LastSeen.Format(time.RFC3339),
					silence.Truncate(time.Second)),
				silence.Seconds(), maxSilence.Seconds(),
			)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// ServiceCrashLoopRule
// ---------------------------------------------------------------------------

// ServiceCrashLoopRule fires when any tracked service has restarted more than
// 3 times in the trailing hour.
type ServiceCrashLoopRule struct{}

func (r *ServiceCrashLoopRule) Name() string { return "service_crash_loop" }

func (r *ServiceCrashLoopRule) Evaluate(ctx AlertContext) *Alert {
	const maxRestarts = 3
	for _, svc := range ctx.Services {
		if svc.Restarts > maxRestarts {
			return newAlert(r.Name(), SeverityCritical,
				fmt.Sprintf("Crash loop: service %q restarted %d times in last hour (limit %d)",
					svc.Name, svc.Restarts, maxRestarts),
				fmt.Sprintf("service=%s restarts=%d last_restart=%s healthy=%v",
					svc.Name, svc.Restarts,
					svc.LastRestart.Format(time.RFC3339),
					svc.Healthy),
				float64(svc.Restarts), float64(maxRestarts),
			)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// RuleEngine
// ---------------------------------------------------------------------------

const (
	engineInterval = 60 * time.Second
	dedupWindow    = 30 * time.Minute
	histCap        = 500
)

// activeEntry tracks a currently firing alert plus when it was last emitted.
type activeEntry struct {
	alert      *Alert
	lastFired  time.Time
}

// RuleEngine runs all registered rules on a fixed interval, deduplicates alerts
// within a 30-minute window, and manages the firing -> resolved lifecycle.
type RuleEngine struct {
	mu       sync.RWMutex
	rules    []Rule
	active   map[string]*activeEntry // rule name -> entry
	history  []*Alert
	histIdx  int
	histFull bool
	logger   *slog.Logger
	dispatch func(ctx context.Context, alert *Alert)
}

// NewRuleEngine creates a RuleEngine with all built-in rules registered.
// The dispatch function is called once per alert event (fire or resolve).
func NewRuleEngine(logger *slog.Logger, dispatch func(context.Context, *Alert)) *RuleEngine {
	e := &RuleEngine{
		active:   make(map[string]*activeEntry),
		history:  make([]*Alert, histCap),
		logger:   logger,
		dispatch: dispatch,
	}
	e.registerBuiltins()
	return e
}

// registerBuiltins registers all nine concrete rules.
func (e *RuleEngine) registerBuiltins() {
	e.rules = []Rule{
		&DrawdownRule{},
		&VaRBreachRule{},
		&PositionConcentrationRule{},
		&ParameterStaleRule{},
		&SignalDivergenceRule{},
		&VelocityRule{},
		&CorrelationBreakdownRule{},
		&FeedStalenessRule{},
		&ServiceCrashLoopRule{},
	}
}

// AddRule appends a custom rule to the engine.
func (e *RuleEngine) AddRule(r Rule) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.rules = append(e.rules, r)
}

// Run starts the evaluation loop. It blocks until ctx is cancelled.
func (e *RuleEngine) Run(ctx context.Context, source func() AlertContext) {
	ticker := time.NewTicker(engineInterval)
	defer ticker.Stop()

	e.logger.Info("rule engine started", "rules", len(e.rules), "interval", engineInterval)

	for {
		select {
		case <-ctx.Done():
			e.logger.Info("rule engine stopped")
			return
		case <-ticker.C:
			ac := source()
			e.EvaluateOnce(ctx, ac)
		}
	}
}

// EvaluateOnce runs a single evaluation pass -- useful for testing.
func (e *RuleEngine) EvaluateOnce(ctx context.Context, ac AlertContext) {
	e.mu.Lock()
	defer e.mu.Unlock()

	now := time.Now()

	for _, rule := range e.rules {
		alert := rule.Evaluate(ac)
		name := rule.Name()

		if alert != nil {
			// Rule is currently firing.
			entry, already := e.active[name]
			if already {
				// Dedup: suppress re-notification within the dedup window.
				if now.Sub(entry.lastFired) < dedupWindow {
					continue
				}
				// Update the stored alert in place.
				entry.alert = alert
				entry.lastFired = now
			} else {
				e.active[name] = &activeEntry{alert: alert, lastFired: now}
			}
			e.appendHistory(alert)
			e.logger.Warn("alert firing",
				"rule", name,
				"severity", alert.Severity,
				"message", alert.Message,
			)
			if e.dispatch != nil {
				e.dispatch(ctx, alert)
			}
		} else {
			// Rule is clear. Resolve if previously firing.
			entry, firing := e.active[name]
			if !firing {
				continue
			}
			t := now
			resolved := *entry.alert
			resolved.State = AlertResolved
			resolved.ResolvedAt = &t
			dur := now.Sub(entry.alert.FiredAt).Truncate(time.Second)
			resolved.Duration = dur.String()
			delete(e.active, name)
			e.appendHistory(&resolved)
			e.logger.Info("alert resolved",
				"rule", name,
				"duration", resolved.Duration,
			)
			if e.dispatch != nil {
				e.dispatch(ctx, &resolved)
			}
		}
	}
}

// appendHistory adds an alert to the ring buffer (caller must hold mu).
func (e *RuleEngine) appendHistory(a *Alert) {
	cp := *a
	e.history[e.histIdx] = &cp
	e.histIdx = (e.histIdx + 1) % histCap
	if e.histIdx == 0 {
		e.histFull = true
	}
}

// ActiveAlerts returns a snapshot of currently firing alerts.
func (e *RuleEngine) ActiveAlerts() []*Alert {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]*Alert, 0, len(e.active))
	for _, entry := range e.active {
		cp := *entry.alert
		out = append(out, &cp)
	}
	return out
}

// History returns all stored alert events in chronological order.
func (e *RuleEngine) History() []*Alert {
	e.mu.RLock()
	defer e.mu.RUnlock()
	var out []*Alert
	if e.histFull {
		for i := e.histIdx; i < histCap; i++ {
			if e.history[i] != nil {
				out = append(out, e.history[i])
			}
		}
	}
	for i := 0; i < e.histIdx; i++ {
		if e.history[i] != nil {
			out = append(out, e.history[i])
		}
	}
	return out
}

// RuleNames returns the names of all registered rules.
func (e *RuleEngine) RuleNames() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	names := make([]string, len(e.rules))
	for i, r := range e.rules {
		names[i] = r.Name()
	}
	return names
}
