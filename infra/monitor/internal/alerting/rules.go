// Package alerting provides the alert rule engine for the monitoring service.
package alerting

import (
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// Bar is a minimal bar type used within the alerting package.
// It mirrors feed.Bar but avoids the import cycle.
type Bar struct {
	Symbol    string
	Timestamp time.Time
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	Source    string
}

// PortfolioState represents a snapshot of the paper trading account.
type PortfolioState struct {
	Timestamp      time.Time
	Equity         float64
	Cash           float64
	DailyPnL       float64
	IntradayDD     float64 // intraday drawdown from daily high
	HighWaterMark  float64
	Positions      map[string]Position
	TotalExposure  float64
}

// Position represents a single open position.
type Position struct {
	Symbol   string
	Qty      float64
	AvgCost  float64
	MarketVal float64
	UnrealPnL float64
	Side     string // "long" | "short"
}

// BHEvent represents a black-hole engine state transition.
type BHEvent struct {
	Symbol    string
	Timeframe string
	EventType string // "formation", "collapse", "mass_spike"
	Mass      float64
	Timestamp time.Time
	Extra     map[string]interface{}
}

// AlertLevel classifies alert severity.
type AlertLevel string

const (
	AlertInfo     AlertLevel = "info"
	AlertWarning  AlertLevel = "warning"
	AlertCritical AlertLevel = "critical"
)

// Alert is a fired alert instance.
type Alert struct {
	ID        string
	Rule      AlertRule
	Symbol    string
	Metric    string
	Value     float64
	Threshold float64
	Level     AlertLevel
	Message   string
	Timestamp time.Time
	FiredAt   time.Time
}

// AlertRule defines a condition that triggers an alert.
type AlertRule struct {
	Name      string        `yaml:"name"`
	Symbol    string        `yaml:"symbol"`    // "" = all symbols
	Metric    string        `yaml:"metric"`    // see MetricXxx constants
	Operator  string        `yaml:"operator"`  // ">", "<", ">=", "<=", "crosses_above", "crosses_below"
	Threshold float64       `yaml:"threshold"`
	Level     AlertLevel    `yaml:"level"`
	Cooldown  time.Duration `yaml:"cooldown"`
	Message   string        `yaml:"message"` // optional template
}

// Metric constants — what value to measure.
const (
	MetricPriceChange = "price_change"    // % change of close vs prev close
	MetricVolSpike    = "vol_spike"       // volume / rolling avg volume
	MetricBHMass      = "bh_mass"         // BH engine mass
	MetricDrawdown    = "drawdown"        // intraday drawdown %
	MetricPnL         = "pnl"            // daily P&L $
	MetricExposure    = "exposure"        // total $ exposure
	MetricBarVolume   = "bar_volume"      // raw bar volume
	MetricBidAskSpread = "bid_ask_spread" // spread in bps
)

// ruleState holds per-rule firing state.
type ruleState struct {
	lastFired  time.Time
	prevValue  float64
	prevAbove  bool // for crosses_above / crosses_below
}

// AlertManager evaluates alert rules against incoming data.
type AlertManager struct {
	log      *zap.Logger
	mu       sync.RWMutex
	rules    []AlertRule
	state    map[string]*ruleState // key = rule name
	notifiers []NotifyChannel
}

// NewAlertManager creates an AlertManager.
func NewAlertManager(log *zap.Logger) *AlertManager {
	return &AlertManager{
		log:   log,
		state: make(map[string]*ruleState),
	}
}

// AddRule registers an alert rule.
func (am *AlertManager) AddRule(rule AlertRule) {
	am.mu.Lock()
	defer am.mu.Unlock()
	am.rules = append(am.rules, rule)
	if _, ok := am.state[rule.Name]; !ok {
		am.state[rule.Name] = &ruleState{}
	}
	am.log.Info("alert rule added",
		zap.String("rule", rule.Name),
		zap.String("metric", rule.Metric),
		zap.String("operator", rule.Operator),
		zap.Float64("threshold", rule.Threshold))
}

// RemoveRule removes a rule by name.
func (am *AlertManager) RemoveRule(name string) {
	am.mu.Lock()
	defer am.mu.Unlock()
	filtered := am.rules[:0]
	for _, r := range am.rules {
		if r.Name != name {
			filtered = append(filtered, r)
		}
	}
	am.rules = filtered
}

// Rules returns a snapshot of all current rules.
func (am *AlertManager) Rules() []AlertRule {
	am.mu.RLock()
	defer am.mu.RUnlock()
	out := make([]AlertRule, len(am.rules))
	copy(out, am.rules)
	return out
}

// AddNotifier registers a notification channel.
func (am *AlertManager) AddNotifier(n NotifyChannel) {
	am.mu.Lock()
	defer am.mu.Unlock()
	am.notifiers = append(am.notifiers, n)
}

// EvaluateBar evaluates bar-based metrics against all rules.
func (am *AlertManager) EvaluateBar(b Bar, prevClose float64) []Alert {
	am.mu.Lock()
	defer am.mu.Unlock()

	metrics := map[string]float64{
		MetricPriceChange: priceChangePct(prevClose, b.Close),
		MetricBarVolume:   b.Volume,
	}

	return am.evaluateMetrics(b.Symbol, b.Timestamp, metrics)
}

// EvaluatePortfolio evaluates portfolio-based metrics against all rules.
func (am *AlertManager) EvaluatePortfolio(state PortfolioState) []Alert {
	am.mu.Lock()
	defer am.mu.Unlock()

	metrics := map[string]float64{
		MetricPnL:      state.DailyPnL,
		MetricDrawdown: state.IntradayDD,
		MetricExposure: state.TotalExposure,
	}

	return am.evaluateMetrics("PORTFOLIO", state.Timestamp, metrics)
}

// EvaluateBHEvent evaluates BH-based metrics against rules.
func (am *AlertManager) EvaluateBHEvent(evt BHEvent) []Alert {
	am.mu.Lock()
	defer am.mu.Unlock()

	metrics := map[string]float64{
		MetricBHMass: evt.Mass,
	}

	return am.evaluateMetrics(evt.Symbol, evt.Timestamp, metrics)
}

// SendAlert dispatches an alert to all registered notifiers.
func (am *AlertManager) SendAlert(alert Alert) {
	am.mu.RLock()
	notifiers := am.notifiers
	am.mu.RUnlock()

	for _, n := range notifiers {
		if err := n.Send(alert); err != nil {
			am.log.Warn("notifier send failed",
				zap.String("notifier", n.Name()),
				zap.Error(err))
		}
	}
}

// evaluateMetrics checks all rules against the supplied metric map.
// Caller must hold am.mu.Lock().
func (am *AlertManager) evaluateMetrics(symbol string, ts time.Time, metrics map[string]float64) []Alert {
	var alerts []Alert
	now := time.Now()

	for i := range am.rules {
		rule := &am.rules[i]
		// Symbol filter.
		if rule.Symbol != "" && rule.Symbol != symbol {
			continue
		}
		// Metric must be available.
		value, ok := metrics[rule.Metric]
		if !ok {
			continue
		}

		st := am.state[rule.Name]
		if st == nil {
			st = &ruleState{}
			am.state[rule.Name] = st
		}

		// Cooldown check.
		if !st.lastFired.IsZero() && now.Sub(st.lastFired) < rule.Cooldown {
			st.prevValue = value
			continue
		}

		fired := false
		switch rule.Operator {
		case ">":
			fired = value > rule.Threshold
		case "<":
			fired = value < rule.Threshold
		case ">=":
			fired = value >= rule.Threshold
		case "<=":
			fired = value <= rule.Threshold
		case "crosses_above":
			fired = st.prevValue <= rule.Threshold && value > rule.Threshold
		case "crosses_below":
			fired = st.prevValue >= rule.Threshold && value < rule.Threshold
		}

		prevAbove := st.prevAbove
		st.prevAbove = value > rule.Threshold
		st.prevValue = value
		_ = prevAbove

		if fired {
			st.lastFired = now
			msg := rule.Message
			if msg == "" {
				msg = fmt.Sprintf("[%s] %s %s %.4f %s %.4f",
					rule.Level, rule.Name, rule.Metric, value, rule.Operator, rule.Threshold)
			}
			level := rule.Level
			if level == "" {
				level = AlertWarning
			}
			alert := Alert{
				ID:        fmt.Sprintf("%s-%d", rule.Name, now.UnixNano()),
				Rule:      *rule,
				Symbol:    symbol,
				Metric:    rule.Metric,
				Value:     value,
				Threshold: rule.Threshold,
				Level:     level,
				Message:   msg,
				Timestamp: ts,
				FiredAt:   now,
			}
			alerts = append(alerts, alert)
			am.log.Info("alert fired",
				zap.String("rule", rule.Name),
				zap.String("symbol", symbol),
				zap.String("level", string(level)),
				zap.Float64("value", value),
				zap.Float64("threshold", rule.Threshold))
		}
	}

	return alerts
}

func priceChangePct(prev, current float64) float64 {
	if prev == 0 {
		return 0
	}
	return (current - prev) / prev * 100
}

// DefaultRules returns a sensible default set of alert rules.
func DefaultRules() []AlertRule {
	return []AlertRule{
		{
			Name:      "large_price_move",
			Metric:    MetricPriceChange,
			Operator:  ">",
			Threshold: 3.0,
			Level:     AlertWarning,
			Cooldown:  5 * time.Minute,
			Message:   "Price moved more than 3% in a single bar",
		},
		{
			Name:      "large_price_drop",
			Metric:    MetricPriceChange,
			Operator:  "<",
			Threshold: -3.0,
			Level:     AlertWarning,
			Cooldown:  5 * time.Minute,
			Message:   "Price dropped more than 3% in a single bar",
		},
		{
			Name:      "daily_loss_limit",
			Metric:    MetricPnL,
			Operator:  "<",
			Threshold: -500.0,
			Level:     AlertCritical,
			Cooldown:  30 * time.Minute,
			Message:   "Daily P&L below -$500",
		},
		{
			Name:      "intraday_drawdown",
			Metric:    MetricDrawdown,
			Operator:  ">",
			Threshold: 2.0,
			Level:     AlertWarning,
			Cooldown:  15 * time.Minute,
			Message:   "Intraday drawdown exceeded 2%",
		},
		{
			Name:      "bh_mass_spike",
			Metric:    MetricBHMass,
			Operator:  ">",
			Threshold: 0.8,
			Level:     AlertInfo,
			Cooldown:  10 * time.Minute,
			Message:   "BH mass crossed 0.8 threshold",
		},
	}
}
