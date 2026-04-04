package aggregator

import (
	"fmt"
	"sort"
	"time"

	"github.com/srfm/gateway/internal/feed"
)

// TimeframeDuration maps well-known timeframe names to their durations.
var TimeframeDuration = map[string]time.Duration{
	"1m":  time.Minute,
	"5m":  5 * time.Minute,
	"15m": 15 * time.Minute,
	"30m": 30 * time.Minute,
	"1h":  time.Hour,
	"2h":  2 * time.Hour,
	"4h":  4 * time.Hour,
	"6h":  6 * time.Hour,
	"12h": 12 * time.Hour,
	"1d":  24 * time.Hour,
	"1w":  7 * 24 * time.Hour,
}

// ValidTimeframes returns the sorted list of supported timeframe names.
func ValidTimeframes() []string {
	names := make([]string, 0, len(TimeframeDuration))
	for k := range TimeframeDuration {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// IsValidTimeframe returns true if name is a known timeframe.
func IsValidTimeframe(name string) bool {
	_, ok := TimeframeDuration[name]
	return ok
}

// WindowStart returns the start of the timeframe window containing t.
// For daily and larger timeframes, the window is anchored to UTC midnight.
func WindowStart(t time.Time, dur time.Duration) time.Time {
	if dur >= 24*time.Hour {
		// Align to UTC date boundary.
		y, m, d := t.UTC().Date()
		return time.Date(y, m, d, 0, 0, 0, 0, time.UTC)
	}
	return t.UTC().Truncate(dur)
}

// NextWindowStart returns the start of the next timeframe window after t.
func NextWindowStart(t time.Time, dur time.Duration) time.Time {
	return WindowStart(t, dur).Add(dur)
}

// ResampleBars resamples a slice of source bars (in any order) to target timeframe.
// Missing windows can be forward-filled or omitted based on forwardFill.
func ResampleBars(bars []feed.Bar, targetDur time.Duration, forwardFill bool) []feed.Bar {
	if len(bars) == 0 {
		return nil
	}

	// Sort by timestamp.
	sorted := make([]feed.Bar, len(bars))
	copy(sorted, bars)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Timestamp.Before(sorted[j].Timestamp)
	})

	type window struct {
		start time.Time
		bar   feed.Bar
		has   bool
	}

	windows := make(map[time.Time]*window)
	order := make([]time.Time, 0)

	for _, b := range sorted {
		ws := WindowStart(b.Timestamp, targetDur)
		w, ok := windows[ws]
		if !ok {
			w = &window{start: ws}
			windows[ws] = w
			order = append(order, ws)
		}
		if !w.has {
			w.bar = feed.Bar{
				Symbol:    b.Symbol,
				Timestamp: ws,
				Open:      b.Open,
				High:      b.High,
				Low:       b.Low,
				Close:     b.Close,
				Volume:    b.Volume,
				Source:    b.Source,
			}
			w.has = true
		} else {
			if b.High > w.bar.High {
				w.bar.High = b.High
			}
			if b.Low < w.bar.Low {
				w.bar.Low = b.Low
			}
			w.bar.Close = b.Close
			w.bar.Volume += b.Volume
		}
	}

	sort.Slice(order, func(i, j int) bool { return order[i].Before(order[j]) })

	out := make([]feed.Bar, 0, len(order))
	var lastBar *feed.Bar

	for _, ws := range order {
		w := windows[ws]
		if w.has {
			out = append(out, w.bar)
			lastBar = &out[len(out)-1]
		} else if forwardFill && lastBar != nil {
			// Forward-fill: create a flat bar using last close.
			fb := feed.Bar{
				Symbol:    lastBar.Symbol,
				Timestamp: ws,
				Open:      lastBar.Close,
				High:      lastBar.Close,
				Low:       lastBar.Close,
				Close:     lastBar.Close,
				Volume:    0,
				Source:    lastBar.Source + "_fwd",
			}
			out = append(out, fb)
			lastBar = &out[len(out)-1]
		}
	}

	return out
}

// OHLCV aggregates a slice of bars into a single bar.
func OHLCV(bars []feed.Bar) (feed.Bar, error) {
	if len(bars) == 0 {
		return feed.Bar{}, fmt.Errorf("empty bars slice")
	}
	result := feed.Bar{
		Symbol:    bars[0].Symbol,
		Timestamp: bars[0].Timestamp,
		Open:      bars[0].Open,
		High:      bars[0].High,
		Low:       bars[0].Low,
		Close:     bars[len(bars)-1].Close,
		Source:    bars[0].Source,
	}
	for _, b := range bars {
		if b.High > result.High {
			result.High = b.High
		}
		if b.Low < result.Low {
			result.Low = b.Low
		}
		result.Volume += b.Volume
	}
	return result, nil
}

// FillGaps identifies gaps in a bar series and optionally fills them.
// A gap is defined as two consecutive bars more than gapThreshold apart.
func FillGaps(bars []feed.Bar, tf time.Duration, fill bool) []feed.Bar {
	if len(bars) < 2 {
		return bars
	}

	var out []feed.Bar
	out = append(out, bars[0])

	for i := 1; i < len(bars); i++ {
		prev := bars[i-1]
		curr := bars[i]
		gap := curr.Timestamp.Sub(prev.Timestamp)

		// How many bars should fit in this gap?
		expectedBars := int(gap / tf)
		if expectedBars > 1 && fill {
			// Insert forward-filled bars.
			for j := 1; j < expectedBars; j++ {
				ts := WindowStart(prev.Timestamp.Add(time.Duration(j)*tf), tf)
				out = append(out, feed.Bar{
					Symbol:    prev.Symbol,
					Timestamp: ts,
					Open:      prev.Close,
					High:      prev.Close,
					Low:       prev.Close,
					Close:     prev.Close,
					Volume:    0,
					Source:    prev.Source + "_gap",
				})
			}
		}
		out = append(out, curr)
	}
	return out
}

// Returns returns the slice of bar-over-bar returns for a series.
func Returns(bars []feed.Bar) []float64 {
	if len(bars) < 2 {
		return nil
	}
	out := make([]float64, len(bars)-1)
	for i := 1; i < len(bars); i++ {
		prev := bars[i-1].Close
		curr := bars[i].Close
		if prev == 0 {
			out[i-1] = 0
		} else {
			out[i-1] = (curr - prev) / prev
		}
	}
	return out
}

// RollingMean computes a rolling mean with window w over x.
func RollingMean(x []float64, w int) []float64 {
	if len(x) < w {
		return nil
	}
	out := make([]float64, len(x)-w+1)
	var sum float64
	for i := 0; i < w; i++ {
		sum += x[i]
	}
	out[0] = sum / float64(w)
	for i := w; i < len(x); i++ {
		sum += x[i] - x[i-w]
		out[i-w+1] = sum / float64(w)
	}
	return out
}

// RollingStdDev computes a rolling standard deviation with window w.
func RollingStdDev(x []float64, w int) []float64 {
	if len(x) < w {
		return nil
	}
	out := make([]float64, len(x)-w+1)
	for i := 0; i <= len(x)-w; i++ {
		slice := x[i : i+w]
		var sum, sumSq float64
		for _, v := range slice {
			sum += v
			sumSq += v * v
		}
		n := float64(w)
		variance := (sumSq - (sum*sum)/n) / (n - 1)
		if variance < 0 {
			variance = 0
		}
		out[i] = sqrt(variance)
	}
	return out
}

// Sharpe computes the annualised Sharpe ratio from a returns slice.
// barsPerYear is the number of bars in a year (e.g., 252*390 for 1m equity).
func Sharpe(returns []float64, barsPerYear float64) float64 {
	if len(returns) == 0 {
		return 0
	}
	var sum float64
	for _, r := range returns {
		sum += r
	}
	mean := sum / float64(len(returns))

	var sq float64
	for _, r := range returns {
		d := r - mean
		sq += d * d
	}
	variance := sq / float64(len(returns))
	std := sqrt(variance)
	if std == 0 {
		return 0
	}
	return mean / std * sqrt(barsPerYear)
}

// MaxDrawdown returns the maximum percentage drawdown in a return series.
func MaxDrawdown(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}
	peak := 1.0
	equity := 1.0
	maxDD := 0.0
	for _, r := range returns {
		equity *= (1 + r)
		if equity > peak {
			peak = equity
		}
		dd := (peak - equity) / peak
		if dd > maxDD {
			maxDD = dd
		}
	}
	return maxDD
}

// sqrt is a pure-Go Newton-Raphson square root (avoids importing math).
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 50; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}
