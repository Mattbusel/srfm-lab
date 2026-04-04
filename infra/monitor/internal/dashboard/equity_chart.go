package dashboard

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// ChartPoint is a generic (x, y) data point for chart rendering.
type ChartPoint struct {
	X time.Time
	Y float64
}

// EquityChartData holds processed data for the equity chart panel.
type EquityChartData struct {
	Labels   []string  `json:"labels"`
	Values   []float64 `json:"values"`
	// Benchmark is an optional second series (e.g. SPY) for comparison.
	Benchmark []float64 `json:"benchmark,omitempty"`
	// DrawdownSeries tracks the running drawdown from peak equity.
	DrawdownSeries []float64 `json:"drawdown,omitempty"`
	// Stats is a map of summary statistics.
	Stats map[string]float64 `json:"stats,omitempty"`
}

// BuildEquityChartData converts raw EquityPoints to chart-ready data.
func BuildEquityChartData(points []EquityPoint, maxPoints int) *EquityChartData {
	if len(points) == 0 {
		return &EquityChartData{
			Labels: []string{},
			Values: []float64{},
			Stats:  map[string]float64{},
		}
	}

	// Sort by timestamp.
	sorted := make([]EquityPoint, len(points))
	copy(sorted, points)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Timestamp.Before(sorted[j].Timestamp)
	})

	// Downsample to maxPoints if needed.
	if maxPoints > 0 && len(sorted) > maxPoints {
		sorted = downsampleEquity(sorted, maxPoints)
	}

	labels := make([]string, len(sorted))
	values := make([]float64, len(sorted))
	dd := make([]float64, len(sorted))

	peak := sorted[0].Equity
	for i, p := range sorted {
		labels[i] = p.Timestamp.Format("2006-01-02 15:04")
		values[i] = p.Equity
		if p.Equity > peak {
			peak = p.Equity
		}
		if peak > 0 {
			dd[i] = (peak - p.Equity) / peak * 100
		}
	}

	// Compute stats.
	stats := computeEquityStats(sorted)

	return &EquityChartData{
		Labels:         labels,
		Values:         values,
		DrawdownSeries: dd,
		Stats:          stats,
	}
}

// computeEquityStats computes summary statistics from equity points.
func computeEquityStats(points []EquityPoint) map[string]float64 {
	if len(points) < 2 {
		return map[string]float64{}
	}

	first := points[0].Equity
	last := points[len(points)-1].Equity
	totalReturn := 0.0
	if first > 0 {
		totalReturn = (last - first) / first * 100
	}

	// Daily returns.
	rets := make([]float64, 0, len(points)-1)
	for i := 1; i < len(points); i++ {
		prev := points[i-1].Equity
		curr := points[i].Equity
		if prev > 0 {
			rets = append(rets, (curr-prev)/prev)
		}
	}

	var sumRet float64
	for _, r := range rets {
		sumRet += r
	}
	meanRet := 0.0
	if len(rets) > 0 {
		meanRet = sumRet / float64(len(rets))
	}

	var sqSum float64
	for _, r := range rets {
		d := r - meanRet
		sqSum += d * d
	}
	vol := 0.0
	if len(rets) > 1 {
		vol = math.Sqrt(sqSum/float64(len(rets)-1)) * math.Sqrt(252) * 100
	}

	// Max drawdown.
	peak := points[0].Equity
	maxDD := 0.0
	for _, p := range points[1:] {
		if p.Equity > peak {
			peak = p.Equity
		}
		if peak > 0 {
			dd := (peak - p.Equity) / peak * 100
			if dd > maxDD {
				maxDD = dd
			}
		}
	}

	// Sharpe.
	sharpe := 0.0
	if vol > 0 {
		sharpe = meanRet * 252 / (vol / 100)
	}

	return map[string]float64{
		"total_return_pct":  totalReturn,
		"max_drawdown_pct":  maxDD,
		"annualized_vol_pct": vol,
		"sharpe_ratio":      sharpe,
		"current_equity":    last,
		"starting_equity":   first,
		"num_points":        float64(len(points)),
	}
}

// downsampleEquity reduces a point series to at most maxPoints by
// using a Largest-Triangle-Three-Buckets approximation.
func downsampleEquity(points []EquityPoint, maxPoints int) []EquityPoint {
	if len(points) <= maxPoints {
		return points
	}

	// Simple bucket downsampling: take the point with max equity per bucket.
	bucketSize := len(points) / maxPoints
	result := make([]EquityPoint, 0, maxPoints)

	for i := 0; i < maxPoints; i++ {
		start := i * bucketSize
		end := start + bucketSize
		if end > len(points) {
			end = len(points)
		}
		bucket := points[start:end]

		// Pick the point with max absolute deviation from the mean.
		var sumEq float64
		for _, p := range bucket {
			sumEq += p.Equity
		}
		meanEq := sumEq / float64(len(bucket))

		best := bucket[0]
		bestDev := math.Abs(bucket[0].Equity - meanEq)
		for _, p := range bucket[1:] {
			dev := math.Abs(p.Equity - meanEq)
			if dev > bestDev {
				bestDev = dev
				best = p
			}
		}
		result = append(result, best)
	}
	return result
}

// PositionChart holds data for the position exposure bar chart.
type PositionChart struct {
	Symbols    []string  `json:"symbols"`
	Exposures  []float64 `json:"exposures"`  // $ exposure (positive = long, negative = short)
	Colors     []string  `json:"colors"`
}

// BuildPositionChart converts a position map to chart data.
func BuildPositionChart(positions map[string]struct {
	MarketVal float64
	Side      string
}) *PositionChart {
	type entry struct {
		Symbol string
		Val    float64
		Side   string
	}
	entries := make([]entry, 0, len(positions))
	for sym, pos := range positions {
		v := pos.MarketVal
		if pos.Side == "short" {
			v = -v
		}
		entries = append(entries, entry{sym, v, pos.Side})
	}
	sort.Slice(entries, func(i, j int) bool {
		return math.Abs(entries[i].Val) > math.Abs(entries[j].Val)
	})

	chart := &PositionChart{
		Symbols:   make([]string, len(entries)),
		Exposures: make([]float64, len(entries)),
		Colors:    make([]string, len(entries)),
	}
	for i, e := range entries {
		chart.Symbols[i] = e.Symbol
		chart.Exposures[i] = e.Val
		if e.Val >= 0 {
			chart.Colors[i] = "#3fb950" // green for long
		} else {
			chart.Colors[i] = "#f85149" // red for short
		}
	}
	return chart
}

// AlertLogEntry is a formatted alert for display.
type AlertLogEntry struct {
	ID        string `json:"id"`
	Time      string `json:"time"`
	Level     string `json:"level"`
	Symbol    string `json:"symbol"`
	Rule      string `json:"rule"`
	Metric    string `json:"metric"`
	Value     string `json:"value"`
	Message   string `json:"message"`
	LevelColor string `json:"level_color"`
}

// FormatAlertLog formats a slice of alerts for the dashboard log panel.
func FormatAlertLog(alerts []interface{ GetLevel() string; GetSymbol() string; GetRule() string; GetMetric() string; GetValue() float64; GetMessage() string; GetFiredAt() time.Time; GetID() string }, maxEntries int) []AlertLogEntry {
	if maxEntries > 0 && len(alerts) > maxEntries {
		alerts = alerts[len(alerts)-maxEntries:]
	}
	result := make([]AlertLogEntry, 0, len(alerts))
	levelColors := map[string]string{
		"info":     "#58a6ff",
		"warning":  "#d29922",
		"critical": "#f85149",
	}
	for i := len(alerts) - 1; i >= 0; i-- {
		a := alerts[i]
		level := a.GetLevel()
		result = append(result, AlertLogEntry{
			ID:         a.GetID(),
			Time:       a.GetFiredAt().Format("15:04:05"),
			Level:      level,
			Symbol:     a.GetSymbol(),
			Rule:       a.GetRule(),
			Metric:     a.GetMetric(),
			Value:      fmt.Sprintf("%.4f", a.GetValue()),
			Message:    a.GetMessage(),
			LevelColor: levelColors[level],
		})
	}
	return result
}
