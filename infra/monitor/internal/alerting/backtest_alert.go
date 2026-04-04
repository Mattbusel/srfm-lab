package alerting

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// BacktestAlert replays historical bar data through alert rules and returns
// all alerts that would have fired. This is used for alert rule validation
// and sensitivity analysis before deploying rules to production.

// HistoricalBar is a simplified bar for backtesting alerting logic.
type HistoricalBar struct {
	Symbol    string
	Timestamp time.Time
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

// AlertBacktestResult describes one alert that would have fired during replay.
type AlertBacktestResult struct {
	Rule      AlertRule
	Symbol    string
	Timestamp time.Time
	Value     float64
	Message   string
}

// AlertBacktester replays historical bars through an AlertManager.
type AlertBacktester struct {
	rules []AlertRule
}

// NewAlertBacktester creates a backtester with the given rules.
func NewAlertBacktester(rules []AlertRule) *AlertBacktester {
	return &AlertBacktester{rules: rules}
}

// Run replays bars (sorted chronologically) and returns every alert
// that would have fired, respecting cooldown periods.
func (ab *AlertBacktester) Run(bars []HistoricalBar) []AlertBacktestResult {
	// Sort bars by timestamp.
	sorted := make([]HistoricalBar, len(bars))
	copy(sorted, bars)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Timestamp.Before(sorted[j].Timestamp)
	})

	// Track per-rule last fired time to respect cooldowns.
	type ruleKey struct {
		rule   string
		symbol string
	}
	lastFired := make(map[ruleKey]time.Time)

	// Rolling window of closes per symbol for derived metrics.
	type window struct{ closes []float64 }
	windows := make(map[string]*window)

	var results []AlertBacktestResult

	for _, bar := range sorted {
		w := windows[bar.Symbol]
		if w == nil {
			w = &window{}
			windows[bar.Symbol] = w
		}
		w.closes = append(w.closes, bar.Close)
		if len(w.closes) > 100 {
			w.closes = w.closes[len(w.closes)-100:]
		}

		for _, rule := range ab.rules {
			// Only match rules for this symbol or wildcard.
			if rule.Symbol != "" && rule.Symbol != bar.Symbol {
				continue
			}

			value := extractMetric(rule.Metric, bar, w.closes)
			if !evaluateCondition(rule.Operator, value, rule.Threshold) {
				continue
			}

			// Check cooldown.
			key := ruleKey{rule.Name, bar.Symbol}
			if !lastFired[key].IsZero() {
				if bar.Timestamp.Sub(lastFired[key]) < rule.Cooldown {
					continue
				}
			}
			lastFired[key] = bar.Timestamp

			results = append(results, AlertBacktestResult{
				Rule:      rule,
				Symbol:    bar.Symbol,
				Timestamp: bar.Timestamp,
				Value:     value,
				Message:   fmt.Sprintf("[BACKTEST] %s: %s=%g threshold=%g", rule.Name, rule.Metric, value, rule.Threshold),
			})
		}
	}
	return results
}

// BacktestSummary holds aggregate statistics from an alert backtest run.
type BacktestSummary struct {
	TotalAlerts    int                    `json:"total_alerts"`
	AlertsByRule   map[string]int         `json:"alerts_by_rule"`
	AlertsBySymbol map[string]int         `json:"alerts_by_symbol"`
	FirstAlert     time.Time              `json:"first_alert,omitempty"`
	LastAlert      time.Time              `json:"last_alert,omitempty"`
	AvgPerDay      float64                `json:"avg_per_day"`
	PeakDay        string                 `json:"peak_day"`
	PeakDayCount   int                    `json:"peak_day_count"`
	Precision      map[string]float64     `json:"precision,omitempty"` // if known signals provided
}

// Summarise computes a BacktestSummary from a slice of results.
func Summarise(results []AlertBacktestResult) BacktestSummary {
	if len(results) == 0 {
		return BacktestSummary{AlertsByRule: map[string]int{}, AlertsBySymbol: map[string]int{}}
	}

	byRule := make(map[string]int)
	bySymbol := make(map[string]int)
	byDay := make(map[string]int)

	var first, last time.Time
	for _, r := range results {
		byRule[r.Rule.Name]++
		bySymbol[r.Symbol]++
		day := r.Timestamp.Format("2006-01-02")
		byDay[day]++
		if first.IsZero() || r.Timestamp.Before(first) {
			first = r.Timestamp
		}
		if last.IsZero() || r.Timestamp.After(last) {
			last = r.Timestamp
		}
	}

	// Find peak day.
	peakDay, peakCount := "", 0
	for d, cnt := range byDay {
		if cnt > peakCount {
			peakCount = cnt
			peakDay = d
		}
	}

	days := last.Sub(first).Hours() / 24
	avgPerDay := 0.0
	if days > 0 {
		avgPerDay = float64(len(results)) / days
	}

	return BacktestSummary{
		TotalAlerts:    len(results),
		AlertsByRule:   byRule,
		AlertsBySymbol: bySymbol,
		FirstAlert:     first,
		LastAlert:      last,
		AvgPerDay:      avgPerDay,
		PeakDay:        peakDay,
		PeakDayCount:   peakCount,
	}
}

// extractMetric computes the metric value for a bar.
func extractMetric(metric string, bar HistoricalBar, closes []float64) float64 {
	switch metric {
	case "price":
		return bar.Close
	case "price_change":
		if len(closes) < 2 {
			return 0
		}
		prev := closes[len(closes)-2]
		if prev == 0 {
			return 0
		}
		return (bar.Close - prev) / prev * 100
	case "volume":
		return bar.Volume
	case "high":
		return bar.High
	case "low":
		return bar.Low
	case "range_pct":
		if bar.Close == 0 {
			return 0
		}
		return (bar.High - bar.Low) / bar.Close * 100
	case "vol_spike":
		// Ratio of current close change to rolling stddev.
		return rollingVolSpike(closes)
	case "rsi":
		return backtestRSI(closes, 14)
	case "drawdown":
		return rollingDrawdown(closes)
	default:
		return 0
	}
}

// evaluateCondition checks if value satisfies the operator+threshold condition.
func evaluateCondition(operator string, value, threshold float64) bool {
	switch operator {
	case ">":
		return value > threshold
	case ">=":
		return value >= threshold
	case "<":
		return value < threshold
	case "<=":
		return value <= threshold
	case "==", "=":
		return value == threshold
	case "!=":
		return value != threshold
	default:
		return false
	}
}

func rollingVolSpike(closes []float64) float64 {
	if len(closes) < 10 {
		return 0
	}
	n := 20
	if len(closes) < n+1 {
		n = len(closes) - 1
	}
	window := closes[len(closes)-n-1 : len(closes)-1]
	rets := make([]float64, len(window)-1)
	for i := 1; i < len(window); i++ {
		if window[i-1] > 0 {
			rets[i-1] = (window[i] - window[i-1]) / window[i-1]
		}
	}
	mean := 0.0
	for _, r := range rets {
		mean += r
	}
	mean /= float64(len(rets))
	variance := 0.0
	for _, r := range rets {
		d := r - mean
		variance += d * d
	}
	stddev := math.Sqrt(variance / float64(len(rets)))
	if stddev == 0 {
		return 0
	}
	lastRet := 0.0
	if closes[len(closes)-2] > 0 {
		lastRet = (closes[len(closes)-1] - closes[len(closes)-2]) / closes[len(closes)-2]
	}
	return math.Abs(lastRet) / stddev
}

func backtestRSI(closes []float64, period int) float64 {
	if len(closes) < period+1 {
		return 50
	}
	gains, losses := 0.0, 0.0
	for i := len(closes) - period; i < len(closes); i++ {
		d := closes[i] - closes[i-1]
		if d > 0 {
			gains += d
		} else {
			losses -= d
		}
	}
	avgGain := gains / float64(period)
	avgLoss := losses / float64(period)
	if avgLoss == 0 {
		return 100
	}
	return 100 - 100/(1+avgGain/avgLoss)
}

func rollingDrawdown(closes []float64) float64 {
	if len(closes) == 0 {
		return 0
	}
	peak := closes[0]
	maxDD := 0.0
	for _, c := range closes {
		if c > peak {
			peak = c
		}
		if peak > 0 {
			dd := (peak - c) / peak * 100
			if dd > maxDD {
				maxDD = dd
			}
		}
	}
	return maxDD
}

// SensitivityReport shows how alert counts change across threshold values.
type SensitivityReport struct {
	Metric     string             `json:"metric"`
	Thresholds []float64          `json:"thresholds"`
	Counts     []int              `json:"counts"`
	Optimal    float64            `json:"optimal_threshold"` // fewest alerts while still catching spikes
}

// ThresholdSensitivity sweeps threshold values for a given rule and returns
// how many alerts fire at each level.
func ThresholdSensitivity(bars []HistoricalBar, baseRule AlertRule, thresholds []float64) SensitivityReport {
	counts := make([]int, len(thresholds))
	for i, thresh := range thresholds {
		rule := baseRule
		rule.Threshold = thresh
		bt := NewAlertBacktester([]AlertRule{rule})
		results := bt.Run(bars)
		counts[i] = len(results)
	}

	// Find "elbow" — threshold where additional strictness doesn't cut alerts much.
	optimal := thresholds[0]
	if len(thresholds) > 2 {
		maxDrop := 0.0
		for i := 1; i < len(counts); i++ {
			drop := float64(counts[i-1] - counts[i])
			if drop > maxDrop {
				maxDrop = drop
				optimal = thresholds[i]
			}
		}
	}

	return SensitivityReport{
		Metric:     baseRule.Metric,
		Thresholds: thresholds,
		Counts:     counts,
		Optimal:    optimal,
	}
}
