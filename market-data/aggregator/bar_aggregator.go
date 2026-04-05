package aggregator

import (
	"context"
	"log"
	"sync"
	"time"

	"srfm/market-data/monitoring"
)

// Timeframes supported by the aggregator.
var Timeframes = []string{"1m", "5m", "15m", "1h", "4h", "1d"}

// timeframeDuration maps timeframe strings to durations.
var timeframeDuration = map[string]time.Duration{
	"1m":  1 * time.Minute,
	"5m":  5 * time.Minute,
	"15m": 15 * time.Minute,
	"1h":  60 * time.Minute,
	"4h":  4 * 60 * time.Minute,
	"1d":  24 * 60 * time.Minute,
}

// RawTick is a single market event from any feed.
type RawTick struct {
	Symbol     string
	Open       float64
	High       float64
	Low        float64
	Close      float64
	Volume     float64
	Timestamp  time.Time
	Source     string
	IsBar      bool  // true = 1m OHLCV bar, false = individual trade tick
	IsComplete bool  // for feeds that signal bar completion (e.g. Binance)
}

// BarEvent is an aggregated bar ready for storage/broadcast.
type BarEvent struct {
	Symbol     string    `json:"symbol"`
	Timeframe  string    `json:"timeframe"`
	Open       float64   `json:"open"`
	High       float64   `json:"high"`
	Low        float64   `json:"low"`
	Close      float64   `json:"close"`
	Volume     float64   `json:"volume"`
	Timestamp  time.Time `json:"timestamp"`
	IsComplete bool      `json:"is_complete"`
	Source     string    `json:"source"`
}

// barState holds the in-progress bar for a (symbol, timeframe) pair.
type barState struct {
	symbol      string
	timeframe   string
	open        float64
	high        float64
	low         float64
	close       float64
	volume      float64
	windowStart time.Time
	hasData     bool
}

func (bs *barState) reset(windowStart time.Time) {
	bs.windowStart = windowStart
	bs.hasData = false
	bs.open = 0
	bs.high = 0
	bs.low = 0
	bs.close = 0
	bs.volume = 0
}

func (bs *barState) update(tick RawTick) {
	if !bs.hasData {
		bs.open = tick.Open
		if tick.IsBar {
			bs.high = tick.High
			bs.low = tick.Low
		} else {
			bs.high = tick.Close
			bs.low = tick.Close
		}
		bs.hasData = true
	} else {
		if tick.IsBar {
			if tick.High > bs.high {
				bs.high = tick.High
			}
			if tick.Low < bs.low {
				bs.low = tick.Low
			}
		} else {
			if tick.Close > bs.high {
				bs.high = tick.Close
			}
			if tick.Close < bs.low {
				bs.low = tick.Close
			}
		}
	}
	bs.close = tick.Close
	bs.volume += tick.Volume
}

func (bs *barState) toEvent(complete bool) BarEvent {
	return BarEvent{
		Symbol:     bs.symbol,
		Timeframe:  bs.timeframe,
		Open:       bs.open,
		High:       bs.high,
		Low:        bs.low,
		Close:      bs.close,
		Volume:     bs.volume,
		Timestamp:  bs.windowStart,
		IsComplete: complete,
		Source:     "aggregator",
	}
}

// BarStorer is the interface for persisting bars.
type BarStorer interface {
	InsertBar(evt BarEvent) error
}

// BarCacher is the interface for caching bars.
type BarCacher interface {
	Put(evt BarEvent)
}

// BarAggregator aggregates 1m ticks into multiple timeframes.
type BarAggregator struct {
	inCh      chan RawTick
	store     BarStorer
	cache     BarCacher
	metrics   *monitoring.Metrics
	broadcast func(BarEvent)

	mu     sync.Mutex
	states map[string]*barState // key: "symbol:timeframe"
	buf    *TickBuffer

	batchMu  sync.Mutex
	batch    []BarEvent
	batchMax int
}

// NewBarAggregator creates a BarAggregator with buffered input channel.
func NewBarAggregator(store BarStorer, cache BarCacher, metrics *monitoring.Metrics) *BarAggregator {
	return &BarAggregator{
		inCh:     make(chan RawTick, 4096),
		store:    store,
		cache:    cache,
		metrics:  metrics,
		states:   make(map[string]*barState),
		buf:      NewTickBuffer(1000),
		batchMax: 100,
	}
}

// EventChan returns the input channel for raw ticks.
func (a *BarAggregator) EventChan() chan<- RawTick {
	return a.inCh
}

// SetBroadcastFunc sets the function called for each completed/updated bar.
func (a *BarAggregator) SetBroadcastFunc(fn func(BarEvent)) {
	a.broadcast = fn
}

// Run processes ticks until ctx is cancelled.
func (a *BarAggregator) Run(ctx context.Context) {
	// Periodic flush ticker for in-progress bars
	flushTicker := time.NewTicker(15 * time.Second)
	defer flushTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return

		case tick, ok := <-a.inCh:
			if !ok {
				return
			}
			a.processTick(tick)

		case <-flushTicker.C:
			a.flushBatch()
		}
	}
}

// Flush forces any pending batch to be written. Called on shutdown.
func (a *BarAggregator) Flush() {
	a.flushBatch()
}

const gracePeriod = 5 * time.Second

func (a *BarAggregator) processTick(tick RawTick) {
	// Validate tick
	n := NewNormalizer()
	if !n.ValidateTick(tick) {
		return
	}
	tick = n.NormalizeTick(tick)

	// Store in ring buffer
	a.buf.Push(tick)

	// For each timeframe, determine the window start this tick belongs to
	for _, tf := range Timeframes {
		dur := timeframeDuration[tf]
		windowStart := tick.Timestamp.Truncate(dur)
		key := tick.Symbol + ":" + tf

		a.mu.Lock()
		state, exists := a.states[key]

		if !exists {
			state = &barState{
				symbol:      tick.Symbol,
				timeframe:   tf,
				windowStart: windowStart,
			}
			a.states[key] = state
		}

		// Check if tick belongs to a new window
		if windowStart.After(state.windowStart) {
			// Late tick grace period: if tick is only slightly behind, absorb it
			tickAge := tick.Timestamp.Sub(state.windowStart.Add(dur))
			if tickAge < 0 {
				tickAge = 0
			}
			if tickAge <= gracePeriod && state.hasData {
				// Late tick for current window - absorb
				state.update(tick)
				evt := state.toEvent(false)
				a.mu.Unlock()
				a.emitBar(evt, false)
				continue
			}

			// Emit completed bar for previous window
			if state.hasData {
				completed := state.toEvent(true)
				state.reset(windowStart)
				state.update(tick)
				currentEvt := state.toEvent(false)
				a.mu.Unlock()
				a.emitBar(completed, true)
				a.emitBar(currentEvt, false)
				continue
			}

			// No data in previous window - just reset
			state.reset(windowStart)
		} else if windowStart.Before(state.windowStart) {
			// Tick from a past window - check grace period
			age := state.windowStart.Sub(windowStart)
			if age <= gracePeriod {
				state.update(tick)
				evt := state.toEvent(false)
				a.mu.Unlock()
				a.emitBar(evt, false)
				continue
			}
			// Too old, discard
			a.mu.Unlock()
			log.Printf("[agg] discarding late tick for %s %s (age=%v)", tick.Symbol, tf, age)
			continue
		}

		// Tick belongs to current window
		state.update(tick)
		evt := state.toEvent(false)
		a.mu.Unlock()
		a.emitBar(evt, false)
	}
}

func (a *BarAggregator) emitBar(evt BarEvent, complete bool) {
	// Update cache with latest state
	a.cache.Put(evt)

	// Broadcast to WebSocket clients
	if a.broadcast != nil {
		a.broadcast(evt)
	}

	// If complete, queue for storage
	if complete {
		a.metrics.BarStored(evt.Timeframe)
		a.batchMu.Lock()
		a.batch = append(a.batch, evt)
		shouldFlush := len(a.batch) >= a.batchMax
		a.batchMu.Unlock()

		if shouldFlush {
			a.flushBatch()
		}
	}
}

func (a *BarAggregator) flushBatch() {
	a.batchMu.Lock()
	if len(a.batch) == 0 {
		a.batchMu.Unlock()
		return
	}
	batch := a.batch
	a.batch = nil
	a.batchMu.Unlock()

	for _, evt := range batch {
		if err := a.store.InsertBar(evt); err != nil {
			log.Printf("[agg] store error for %s/%s: %v", evt.Symbol, evt.Timeframe, err)
		}
	}
}
