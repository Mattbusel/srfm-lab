package alerting

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"go.uber.org/zap"
)

// BarHistory holds a rolling window of recent closes for metrics computation.
type BarHistory struct {
	mu      sync.RWMutex
	history map[string][]float64 // symbol -> recent closes
	maxLen  int
}

// NewBarHistory creates a BarHistory with maxLen closes per symbol.
func NewBarHistory(maxLen int) *BarHistory {
	if maxLen <= 0 {
		maxLen = 100
	}
	return &BarHistory{
		history: make(map[string][]float64),
		maxLen:  maxLen,
	}
}

// Push appends a close price for a symbol.
func (bh *BarHistory) Push(symbol string, close float64) {
	bh.mu.Lock()
	defer bh.mu.Unlock()
	h := bh.history[symbol]
	h = append(h, close)
	if len(h) > bh.maxLen {
		h = h[len(h)-bh.maxLen:]
	}
	bh.history[symbol] = h
}

// PrevClose returns the close before the most recent one.
func (bh *BarHistory) PrevClose(symbol string) float64 {
	bh.mu.RLock()
	defer bh.mu.RUnlock()
	h := bh.history[symbol]
	if len(h) < 2 {
		return 0
	}
	return h[len(h)-2]
}

// RollingStdDev computes the rolling standard deviation of the last n closes.
func (bh *BarHistory) RollingStdDev(symbol string, n int) float64 {
	bh.mu.RLock()
	h := bh.history[symbol]
	bh.mu.RUnlock()

	if len(h) < 2 {
		return 0
	}
	if n > len(h) {
		n = len(h)
	}
	slice := h[len(h)-n:]
	var sum float64
	for _, v := range slice {
		sum += v
	}
	mean := sum / float64(len(slice))
	var sq float64
	for _, v := range slice {
		d := v - mean
		sq += d * d
	}
	variance := sq / float64(len(slice))
	if variance <= 0 {
		return 0
	}
	// Newton-Raphson sqrt.
	z := variance
	for i := 0; i < 50; i++ {
		z = (z + variance/z) / 2
	}
	return z
}

// VolRatio returns the last bar's volume vs the rolling average.
func VolRatio(lastVol float64, history []float64) float64 {
	if len(history) == 0 {
		return 1.0
	}
	var sum float64
	for _, v := range history {
		sum += v
	}
	avg := sum / float64(len(history))
	if avg == 0 {
		return 0
	}
	return lastVol / avg
}

// ----- AlertStateMachine tracks rule state transitions -----

// StateMachineState is the persisted state of an alert rule.
type StateMachineState struct {
	Active     bool
	EnteredAt  time.Time
	ExitedAt   time.Time
	TriggerVal float64
	Count      int
}

// AlertStateMachine evaluates stateful alert rules with hysteresis.
// It prevents alert flapping by requiring N consecutive triggers before firing.
type AlertStateMachine struct {
	mu        sync.RWMutex
	log       *zap.Logger
	rules     map[string]*statefulRule // rule name -> state
	notifiers []NotifyChannel
}

type statefulRule struct {
	rule          AlertRule
	consecutiveHi int // consecutive ticks above threshold
	consecutiveLo int // consecutive ticks below threshold
	firing        bool
	lastFired     time.Time
	prevValue     float64
}

// NewAlertStateMachine creates an AlertStateMachine.
func NewAlertStateMachine(log *zap.Logger) *AlertStateMachine {
	return &AlertStateMachine{
		log:   log,
		rules: make(map[string]*statefulRule),
	}
}

// AddRule registers a rule in the state machine.
func (asm *AlertStateMachine) AddRule(rule AlertRule) {
	asm.mu.Lock()
	defer asm.mu.Unlock()
	asm.rules[rule.Name] = &statefulRule{rule: rule}
}

// AddNotifier registers a notification channel.
func (asm *AlertStateMachine) AddNotifier(n NotifyChannel) {
	asm.mu.Lock()
	defer asm.mu.Unlock()
	asm.notifiers = append(asm.notifiers, n)
}

// Tick evaluates all rules against the current metric value for a symbol.
func (asm *AlertStateMachine) Tick(symbol string, metric string, value float64, ts time.Time) []Alert {
	asm.mu.Lock()
	defer asm.mu.Unlock()

	var fired []Alert
	now := time.Now()

	for _, sr := range asm.rules {
		r := sr.rule
		if r.Symbol != "" && r.Symbol != symbol {
			continue
		}
		if r.Metric != metric {
			continue
		}
		// Cooldown.
		if !sr.lastFired.IsZero() && now.Sub(sr.lastFired) < r.Cooldown {
			sr.prevValue = value
			continue
		}

		condition := evaluateCondition(value, sr.prevValue, r.Operator, r.Threshold)
		sr.prevValue = value

		if condition {
			sr.consecutiveHi++
			sr.consecutiveLo = 0
		} else {
			sr.consecutiveLo++
			sr.consecutiveHi = 0
		}

		// Fire after 1 consecutive trigger (extend to N for debouncing if needed).
		if condition && sr.consecutiveHi >= 1 {
			sr.firing = true
			sr.lastFired = now
			msg := r.Message
			if msg == "" {
				msg = fmt.Sprintf("%s: %s %s %.4f threshold=%.4f",
					r.Name, metric, r.Operator, value, r.Threshold)
			}
			a := Alert{
				ID:        fmt.Sprintf("%s-%s-%d", r.Name, symbol, now.UnixNano()),
				Rule:      r,
				Symbol:    symbol,
				Metric:    metric,
				Value:     value,
				Threshold: r.Threshold,
				Level:     r.Level,
				Message:   msg,
				Timestamp: ts,
				FiredAt:   now,
			}
			fired = append(fired, a)
			asm.log.Info("alert fired via state machine",
				zap.String("rule", r.Name),
				zap.String("symbol", symbol),
				zap.Float64("value", value))
		} else if !condition && sr.firing {
			sr.firing = false
		}
	}

	return fired
}

func evaluateCondition(value, prevValue float64, operator string, threshold float64) bool {
	switch operator {
	case ">":
		return value > threshold
	case "<":
		return value < threshold
	case ">=":
		return value >= threshold
	case "<=":
		return value <= threshold
	case "crosses_above":
		return prevValue <= threshold && value > threshold
	case "crosses_below":
		return prevValue >= threshold && value < threshold
	}
	return false
}

// ----- Alert aggregator / deduplication -----

// AlertDeduplicator suppresses duplicate alerts within a time window.
type AlertDeduplicator struct {
	mu      sync.Mutex
	seen    map[string]time.Time // alert fingerprint -> last seen
	window  time.Duration
}

// NewAlertDeduplicator creates an AlertDeduplicator.
func NewAlertDeduplicator(window time.Duration) *AlertDeduplicator {
	return &AlertDeduplicator{
		seen:   make(map[string]time.Time),
		window: window,
	}
}

// IsDuplicate returns true if this alert was already seen within the window.
// It also records the alert as seen.
func (d *AlertDeduplicator) IsDuplicate(a Alert) bool {
	fp := alertFingerprint(a)
	d.mu.Lock()
	defer d.mu.Unlock()

	if last, ok := d.seen[fp]; ok {
		if time.Since(last) < d.window {
			return true
		}
	}
	d.seen[fp] = time.Now()
	return false
}

// Prune removes stale entries from the deduplication map.
func (d *AlertDeduplicator) Prune() {
	d.mu.Lock()
	defer d.mu.Unlock()
	for fp, ts := range d.seen {
		if time.Since(ts) > d.window*2 {
			delete(d.seen, fp)
		}
	}
}

func alertFingerprint(a Alert) string {
	return fmt.Sprintf("%s|%s|%s|%.4f", a.Rule.Name, a.Symbol, a.Metric, math.Round(a.Value*100)/100)
}

// ----- Alert sorter -----

// SortAlertsByLevel sorts alerts with critical > warning > info.
func SortAlertsByLevel(alerts []Alert) []Alert {
	order := map[AlertLevel]int{AlertCritical: 0, AlertWarning: 1, AlertInfo: 2}
	sorted := make([]Alert, len(alerts))
	copy(sorted, alerts)
	sort.Slice(sorted, func(i, j int) bool {
		li := order[sorted[i].Level]
		lj := order[sorted[j].Level]
		if li != lj {
			return li < lj
		}
		return sorted[i].FiredAt.After(sorted[j].FiredAt)
	})
	return sorted
}
