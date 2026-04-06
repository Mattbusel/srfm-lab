// alert_engine.go — Rule-based alert evaluation engine.
//
// Evaluates alert rules every 10s against a MetricSnapshot.
// Implements firing → resolved state machine with 5-minute dedup.

package main

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// ─── Types ─────────────────────────────────────────────────────────────────────

// Severity of an alert.
type Severity string

const (
	SeverityInfo     Severity = "info"
	SeverityWarning  Severity = "warning"
	SeverityCritical Severity = "critical"
	SeverityError    Severity = "error"
)

// RuleType determines how the rule condition is evaluated.
type RuleType string

const (
	RuleTypeThreshold    RuleType = "threshold"
	RuleTypeRateOfChange RuleType = "rate_of_change"
	RuleTypeSustained    RuleType = "sustained"
	RuleTypePattern      RuleType = "pattern"
)

// AlertState tracks the current firing state of a rule.
type AlertState string

const (
	AlertStatePending  AlertState = "pending"
	AlertStateFiring   AlertState = "firing"
	AlertStateResolved AlertState = "resolved"
)

// AlertRule defines when and how an alert should be generated.
type AlertRule struct {
	Name        string    `yaml:"name"`
	Enabled     bool      `yaml:"enabled"`
	Severity    Severity  `yaml:"severity"`
	Type        RuleType  `yaml:"type"`
	// Threshold rule fields.
	Metric      string    `yaml:"metric"`
	Operator    string    `yaml:"operator"` // ">", "<", ">=", "<=", "==", "!="
	Threshold   float64   `yaml:"threshold"`
	// Rate of change: alert if |Δmetric/Δt| > threshold.
	Window      int       `yaml:"window"` // number of eval cycles for rate calculation
	// Sustained: alert if condition holds for N consecutive cycles.
	ConsecutiveN int      `yaml:"consecutive_n"`
	// Message template.
	Message     string    `yaml:"message"`
	// Dedup: minimum seconds between repeated alerts for this rule.
	DedupSecs   int       `yaml:"dedup_secs"`
}

// Alert is a fired or resolved alert event.
type Alert struct {
	ID          string     `json:"id"`
	RuleName    string     `json:"rule_name"`
	Severity    Severity   `json:"severity"`
	State       AlertState `json:"state"`
	Message     string     `json:"message"`
	MetricName  string     `json:"metric_name"`
	MetricValue float64    `json:"metric_value"`
	Threshold   float64    `json:"threshold"`
	FiredAt     time.Time  `json:"fired_at"`
	ResolvedAt  *time.Time `json:"resolved_at,omitempty"`
	Duration    string     `json:"duration,omitempty"`
}

// MetricSnapshot is a point-in-time view of all tracked metrics.
type MetricSnapshot struct {
	Timestamp           time.Time
	Equity              float64
	DrawdownPct         float64
	OpenPositions       int
	CircuitBreakerState int // 0=off, 1=on
	BhActiveCount       int
	DayPnlPct           float64
	TraderAlive         bool
	HealthChecks        map[string]bool
}

// ─── RuleState ─────────────────────────────────────────────────────────────────

// ruleState holds runtime state for a single rule.
type ruleState struct {
	rule          AlertRule
	state         AlertState
	consecutiveHits int
	lastFiredAt   time.Time
	prevValues    []float64 // for rate-of-change calculation
	activeAlertID string
}

func newRuleState(rule AlertRule) *ruleState {
	return &ruleState{rule: rule, state: AlertStateResolved}
}

// ─── AlertEngine ───────────────────────────────────────────────────────────────

const historyCapacity = 1000

// AlertEngine evaluates rules and dispatches alerts.
type AlertEngine struct {
	mu         sync.RWMutex
	rules      []*ruleState
	active     map[string]*Alert // rule name → active alert
	history    []*Alert          // ring buffer
	histIdx    int
	histFull   bool
	notifier   CompositeNotifier
	logger     *slog.Logger
	evalCount  uint64
	prevSnap   *MetricSnapshot
}

// NewAlertEngine creates an engine from a slice of rules.
func NewAlertEngine(rules []AlertRule, notifier CompositeNotifier, logger *slog.Logger) *AlertEngine {
	states := make([]*ruleState, 0, len(rules))
	for _, r := range rules {
		if r.Enabled {
			states = append(states, newRuleState(r))
		}
	}
	return &AlertEngine{
		rules:    states,
		active:   make(map[string]*Alert),
		history:  make([]*Alert, historyCapacity),
		notifier: notifier,
		logger:   logger,
	}
}

// Evaluate runs all rules against the current snapshot.
func (e *AlertEngine) Evaluate(ctx context.Context, snap MetricSnapshot) {
	snap.Timestamp = time.Now()
	e.mu.Lock()
	defer e.mu.Unlock()
	e.evalCount++

	for _, rs := range e.rules {
		e.evalRule(ctx, rs, snap)
	}
	e.prevSnap = &snap
}

func (e *AlertEngine) evalRule(ctx context.Context, rs *ruleState, snap MetricSnapshot) {
	if !rs.rule.Enabled { return }

	var triggered bool
	var value float64

	switch rs.rule.Type {
	case RuleTypeThreshold, "": // default
		value = e.metricValue(rs.rule.Metric, snap)
		triggered = compare(value, rs.rule.Operator, rs.rule.Threshold)

	case RuleTypeRateOfChange:
		value = e.metricValue(rs.rule.Metric, snap)
		rs.prevValues = append(rs.prevValues, value)
		if len(rs.prevValues) > rs.rule.Window+1 {
			rs.prevValues = rs.prevValues[1:]
		}
		if len(rs.prevValues) >= 2 {
			rate := value - rs.prevValues[0]
			triggered = compare(math.Abs(rate), rs.rule.Operator, rs.rule.Threshold)
		}

	case RuleTypeSustained:
		value = e.metricValue(rs.rule.Metric, snap)
		if compare(value, rs.rule.Operator, rs.rule.Threshold) {
			rs.consecutiveHits++
			n := rs.rule.ConsecutiveN
			if n <= 0 { n = 3 }
			triggered = rs.consecutiveHits >= n
		} else {
			rs.consecutiveHits = 0
		}

	case RuleTypePattern:
		// Pattern rules use built-in pattern checks.
		triggered, value = e.evalPattern(rs.rule, snap)
	}

	if triggered {
		e.handleFiring(ctx, rs, snap, value)
	} else {
		e.handleResolved(ctx, rs)
	}
}

func (e *AlertEngine) handleFiring(ctx context.Context, rs *ruleState, snap MetricSnapshot, value float64) {
	dedupSecs := rs.rule.DedupSecs
	if dedupSecs <= 0 { dedupSecs = 300 } // 5 minutes default

	now := time.Now()
	sinceLastFire := now.Sub(rs.lastFiredAt)

	if rs.state == AlertStateFiring && sinceLastFire < time.Duration(dedupSecs)*time.Second {
		return // within dedup window, skip
	}

	rs.state = AlertStateFiring
	rs.lastFiredAt = now

	alert := &Alert{
		ID:          fmt.Sprintf("%s-%d", rs.rule.Name, now.UnixMilli()),
		RuleName:    rs.rule.Name,
		Severity:    rs.rule.Severity,
		State:       AlertStateFiring,
		Message:     e.formatMessage(rs.rule.Message, snap, value),
		MetricName:  rs.rule.Metric,
		MetricValue: value,
		Threshold:   rs.rule.Threshold,
		FiredAt:     now,
	}
	rs.activeAlertID = alert.ID
	e.active[rs.rule.Name] = alert
	e.addHistory(alert)

	e.logger.Warn("alert firing",
		"rule", rs.rule.Name,
		"severity", rs.rule.Severity,
		"metric", rs.rule.Metric,
		"value", value,
		"threshold", rs.rule.Threshold,
	)

	if err := e.notifier.Notify(ctx, alert); err != nil {
		e.logger.Error("notification failed", "rule", rs.rule.Name, "err", err)
	}
}

func (e *AlertEngine) handleResolved(ctx context.Context, rs *ruleState) {
	if rs.state != AlertStateFiring { return }
	rs.state = AlertStateResolved
	rs.consecutiveHits = 0

	now := time.Now()
	if a, ok := e.active[rs.rule.Name]; ok {
		a.State = AlertStateResolved
		a.ResolvedAt = &now
		dur := now.Sub(a.FiredAt).Truncate(time.Second).String()
		a.Duration = dur
		delete(e.active, rs.rule.Name)
		e.addHistory(a)

		e.logger.Info("alert resolved",
			"rule", rs.rule.Name,
			"duration", dur,
		)
		if err := e.notifier.Notify(ctx, a); err != nil {
			e.logger.Error("resolve notification failed", "rule", rs.rule.Name, "err", err)
		}
	}
}

// metricValue extracts a named metric from the snapshot.
func (e *AlertEngine) metricValue(name string, snap MetricSnapshot) float64 {
	switch strings.ToLower(name) {
	case "equity":
		return snap.Equity
	case "drawdown_pct":
		return snap.DrawdownPct
	case "open_positions":
		return float64(snap.OpenPositions)
	case "circuit_breaker_state":
		return float64(snap.CircuitBreakerState)
	case "bh_active_count":
		return float64(snap.BhActiveCount)
	case "day_pnl_pct":
		return snap.DayPnlPct
	case "trader_alive":
		if snap.TraderAlive { return 1.0 }
		return 0.0
	default:
		return 0.0
	}
}

// evalPattern checks named composite patterns.
func (e *AlertEngine) evalPattern(rule AlertRule, snap MetricSnapshot) (bool, float64) {
	switch rule.Metric {
	case "circuit_breaker_and_drawdown":
		if snap.CircuitBreakerState == 1 && snap.DrawdownPct < -0.10 {
			return true, snap.DrawdownPct
		}
	case "trader_down":
		if !snap.TraderAlive {
			return true, 0.0
		}
	}
	return false, 0.0
}

func (e *AlertEngine) formatMessage(tmpl string, snap MetricSnapshot, value float64) string {
	r := strings.NewReplacer(
		"{value:.4f}", fmt.Sprintf("%.4f", value),
		"{value:.2f}", fmt.Sprintf("%.2f", value),
		"{value:.1%}", fmt.Sprintf("%.1f%%", value*100),
		"{equity}", fmt.Sprintf("%.0f", snap.Equity),
		"{drawdown_pct:.1%}", fmt.Sprintf("%.1f%%", snap.DrawdownPct*100),
		"{day_pnl_pct:.1%}", fmt.Sprintf("%.1f%%", snap.DayPnlPct*100),
		"{open_positions}", fmt.Sprintf("%d", snap.OpenPositions),
		"{bh_active_count}", fmt.Sprintf("%d", snap.BhActiveCount),
	)
	result := r.Replace(tmpl)
	if result == tmpl {
		// Fallback: append value.
		result = fmt.Sprintf("%s [value=%.4f]", tmpl, value)
	}
	return result
}

func (e *AlertEngine) addHistory(a *Alert) {
	// Copy to avoid mutation.
	cp := *a
	e.history[e.histIdx] = &cp
	e.histIdx = (e.histIdx + 1) % historyCapacity
	if e.histIdx == 0 { e.histFull = true }
}

// ActiveAlerts returns currently firing alerts.
func (e *AlertEngine) ActiveAlerts() []*Alert {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]*Alert, 0, len(e.active))
	for _, a := range e.active {
		cp := *a
		out = append(out, &cp)
	}
	return out
}

// History returns the last ≤1000 alert events in chronological order.
func (e *AlertEngine) History() []*Alert {
	e.mu.RLock()
	defer e.mu.RUnlock()
	var out []*Alert
	if e.histFull {
		for i := e.histIdx; i < historyCapacity; i++ {
			if e.history[i] != nil { out = append(out, e.history[i]) }
		}
	}
	for i := 0; i < e.histIdx; i++ {
		if e.history[i] != nil { out = append(out, e.history[i]) }
	}
	return out
}

// Rules returns a copy of the configured rule definitions.
func (e *AlertEngine) Rules() []AlertRule {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]AlertRule, len(e.rules))
	for i, rs := range e.rules { out[i] = rs.rule }
	return out
}

// ─── Rule Loading ─────────────────────────────────────────────────────────────

type alertRulesFile struct {
	Rules map[string]struct {
		Enabled      bool     `yaml:"enabled"`
		Level        string   `yaml:"level"`
		Condition    string   `yaml:"condition"`
		Message      string   `yaml:"message"`
		CooldownMins int      `yaml:"cooldown_minutes"`
	} `yaml:"alerts"`
}

// LoadAlertRules parses the YAML file into AlertRule slices.
// The existing alert_rules.yaml format is adapted to the engine's rule types.
func LoadAlertRules(path string) ([]AlertRule, error) {
	data, err := os.ReadFile(path)
	if err != nil { return nil, err }

	var raw alertRulesFile
	if err := yaml.Unmarshal(data, &raw); err != nil { return nil, err }

	rules := make([]AlertRule, 0, len(raw.Rules))
	for name, r := range raw.Rules {
		if !r.Enabled { continue }
		sev := SeverityInfo
		switch strings.ToUpper(r.Level) {
		case "WARNING": sev = SeverityWarning
		case "CRITICAL": sev = SeverityCritical
		case "ERROR": sev = SeverityError
		}
		dedupSecs := r.CooldownMins * 60
		if dedupSecs <= 0 { dedupSecs = 0 } // 0 = always fire

		rule := AlertRule{
			Name:      name,
			Enabled:   r.Enabled,
			Severity:  sev,
			Type:      RuleTypeThreshold,
			Message:   r.Message,
			DedupSecs: dedupSecs,
		}
		// Map well-known rule names to metrics.
		switch name {
		case "drawdown_10pct":
			rule.Metric, rule.Operator, rule.Threshold = "drawdown_pct", "<", -0.10
		case "drawdown_circuit_breaker":
			rule.Metric, rule.Operator, rule.Threshold = "drawdown_pct", "<", -0.15
		case "daily_loss_limit":
			rule.Metric, rule.Operator, rule.Threshold = "day_pnl_pct", "<", -0.02
		case "circuit_breaker_and_drawdown":
			rule.Type, rule.Metric = RuleTypePattern, "circuit_breaker_and_drawdown"
		default:
			rule.Metric = name
		}
		rules = append(rules, rule)
	}
	return rules, nil
}

// DefaultAlertRules returns a minimal safe set of rules if config load fails.
func DefaultAlertRules() []AlertRule {
	return []AlertRule{
		{
			Name: "drawdown_warning", Enabled: true, Severity: SeverityWarning,
			Type: RuleTypeThreshold, Metric: "drawdown_pct", Operator: "<", Threshold: -0.10,
			Message: "Drawdown at {drawdown_pct:.1%}", DedupSecs: 3600,
		},
		{
			Name: "drawdown_critical", Enabled: true, Severity: SeverityCritical,
			Type: RuleTypeThreshold, Metric: "drawdown_pct", Operator: "<", Threshold: -0.15,
			Message: "CRITICAL drawdown {drawdown_pct:.1%}", DedupSecs: 0,
		},
		{
			Name: "trader_down", Enabled: true, Severity: SeverityCritical,
			Type: RuleTypePattern, Metric: "trader_down",
			Message: "Trader heartbeat lost — system may be halted", DedupSecs: 300,
		},
		{
			Name: "circuit_breaker", Enabled: true, Severity: SeverityCritical,
			Type: RuleTypeThreshold, Metric: "circuit_breaker_state", Operator: "==", Threshold: 1,
			Message: "Circuit breaker ACTIVE", DedupSecs: 0,
		},
	}
}

// compare evaluates: left op right.
func compare(left float64, op string, right float64) bool {
	switch op {
	case ">":  return left > right
	case "<":  return left < right
	case ">=": return left >= right
	case "<=": return left <= right
	case "==": return math.Abs(left-right) < 1e-9
	case "!=": return math.Abs(left-right) >= 1e-9
	default:   return false
	}
}
