package aggregator

import (
	"fmt"
	"sync"
	"time"

	"github.com/srfm/gateway/internal/feed"
	"go.uber.org/zap"
)

// Timeframe represents a bar aggregation frequency.
type Timeframe struct {
	Name     string
	Duration time.Duration
}

// ParseTimeframe converts a string like "5m", "1h", "4h", "1d" into a Timeframe.
func ParseTimeframe(s string) (Timeframe, error) {
	switch s {
	case "1m":
		return Timeframe{Name: "1m", Duration: time.Minute}, nil
	case "5m":
		return Timeframe{Name: "5m", Duration: 5 * time.Minute}, nil
	case "15m":
		return Timeframe{Name: "15m", Duration: 15 * time.Minute}, nil
	case "30m":
		return Timeframe{Name: "30m", Duration: 30 * time.Minute}, nil
	case "1h":
		return Timeframe{Name: "1h", Duration: time.Hour}, nil
	case "4h":
		return Timeframe{Name: "4h", Duration: 4 * time.Hour}, nil
	case "1d":
		return Timeframe{Name: "1d", Duration: 24 * time.Hour}, nil
	default:
		return Timeframe{}, fmt.Errorf("unknown timeframe %q", s)
	}
}

// BarHandler is a callback invoked when a completed aggregated bar is ready.
type BarHandler func(timeframe Timeframe, bar feed.Bar)

// partialBar tracks an in-progress aggregated bar.
type partialBar struct {
	bar      feed.Bar
	barStart time.Time
}

// symbolAgg tracks aggregation state for one symbol + timeframe combination.
type symbolAgg struct {
	mu      sync.Mutex
	partial *partialBar
}

// BarAggregator aggregates incoming 1-minute bars to configurable timeframes.
// It is safe for concurrent use: multiple goroutines may call Push simultaneously.
type BarAggregator struct {
	log        *zap.Logger
	timeframes []Timeframe

	// state[symbol][tf.Name] = *symbolAgg
	mu    sync.RWMutex
	state map[string]map[string]*symbolAgg

	// handlers are called (in a new goroutine) when a bar completes.
	handlerMu sync.RWMutex
	handlers  []BarHandler
}

// NewBarAggregator creates a BarAggregator for the given timeframe strings.
func NewBarAggregator(timeframeNames []string, log *zap.Logger) (*BarAggregator, error) {
	tfs := make([]Timeframe, 0, len(timeframeNames))
	for _, name := range timeframeNames {
		tf, err := ParseTimeframe(name)
		if err != nil {
			return nil, err
		}
		tfs = append(tfs, tf)
	}
	return &BarAggregator{
		log:        log,
		timeframes: tfs,
		state:      make(map[string]map[string]*symbolAgg),
	}, nil
}

// AddHandler registers a callback that is called for each completed bar.
// Handlers are called serially in the order they were registered, but the
// invocation itself happens in the Push caller's goroutine.
func (a *BarAggregator) AddHandler(h BarHandler) {
	a.handlerMu.Lock()
	defer a.handlerMu.Unlock()
	a.handlers = append(a.handlers, h)
}

// Push feeds a completed 1m bar into the aggregator.
// Partial (IsPartial=true) bars are ignored.
func (a *BarAggregator) Push(b feed.Bar) {
	if b.IsPartial {
		return
	}

	for _, tf := range a.timeframes {
		a.pushForTimeframe(b, tf)
	}
}

func (a *BarAggregator) pushForTimeframe(b feed.Bar, tf Timeframe) {
	agg := a.getOrCreate(b.Symbol, tf.Name)
	agg.mu.Lock()
	defer agg.mu.Unlock()

	barStart := b.Timestamp.Truncate(tf.Duration)

	if agg.partial == nil {
		// Start a new partial bar.
		agg.partial = &partialBar{
			barStart: barStart,
			bar: feed.Bar{
				Symbol:    b.Symbol,
				Timestamp: barStart,
				Open:      b.Open,
				High:      b.High,
				Low:       b.Low,
				Close:     b.Close,
				Volume:    b.Volume,
				Source:    b.Source,
			},
		}
		return
	}

	if barStart == agg.partial.barStart {
		// Same window: merge.
		pb := &agg.partial.bar
		if b.High > pb.High {
			pb.High = b.High
		}
		if b.Low < pb.Low {
			pb.Low = b.Low
		}
		pb.Close = b.Close
		pb.Volume += b.Volume
		return
	}

	if barStart.After(agg.partial.barStart) {
		// New window: emit completed bar.
		completed := agg.partial.bar
		// Start fresh with the current incoming bar.
		agg.partial = &partialBar{
			barStart: barStart,
			bar: feed.Bar{
				Symbol:    b.Symbol,
				Timestamp: barStart,
				Open:      b.Open,
				High:      b.High,
				Low:       b.Low,
				Close:     b.Close,
				Volume:    b.Volume,
				Source:    b.Source,
			},
		}
		// Notify handlers outside the lock would require copying — we call
		// them after returning from this function.  Use a deferred closure.
		a.emit(tf, completed)
		return
	}

	// Out-of-order bar (barStart < partial.barStart): discard.
	a.log.Warn("bar aggregator: out-of-order bar discarded",
		zap.String("symbol", b.Symbol),
		zap.String("timeframe", tf.Name),
		zap.Time("bar_time", b.Timestamp),
		zap.Time("window_start", agg.partial.barStart))
}

// Flush forces emission of any pending partial bars for all symbols.
// Useful at shutdown or end-of-day.
func (a *BarAggregator) Flush() {
	a.mu.RLock()
	defer a.mu.RUnlock()

	for symbol, tfs := range a.state {
		for tfName, agg := range tfs {
			agg.mu.Lock()
			if agg.partial != nil {
				completed := agg.partial.bar
				agg.partial = nil
				agg.mu.Unlock()
				tf, err := ParseTimeframe(tfName)
				if err != nil {
					continue
				}
				a.log.Debug("bar aggregator flush",
					zap.String("symbol", symbol),
					zap.String("timeframe", tfName))
				a.emit(tf, completed)
			} else {
				agg.mu.Unlock()
			}
		}
	}
}

func (a *BarAggregator) getOrCreate(symbol, tfName string) *symbolAgg {
	a.mu.RLock()
	if tfs, ok := a.state[symbol]; ok {
		if agg, ok := tfs[tfName]; ok {
			a.mu.RUnlock()
			return agg
		}
	}
	a.mu.RUnlock()

	a.mu.Lock()
	defer a.mu.Unlock()
	if _, ok := a.state[symbol]; !ok {
		a.state[symbol] = make(map[string]*symbolAgg)
	}
	if agg, ok := a.state[symbol][tfName]; ok {
		return agg
	}
	agg := &symbolAgg{}
	a.state[symbol][tfName] = agg
	return agg
}

func (a *BarAggregator) emit(tf Timeframe, bar feed.Bar) {
	a.handlerMu.RLock()
	hs := a.handlers
	a.handlerMu.RUnlock()
	for _, h := range hs {
		h(tf, bar)
	}
}

// Timeframes returns the list of configured timeframes.
func (a *BarAggregator) Timeframes() []Timeframe {
	return a.timeframes
}
