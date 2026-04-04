package feed

import (
	"context"
	"sync"
	"time"

	"go.uber.org/zap"
)

// Feed is the interface implemented by all market data feeds.
type Feed interface {
	Start(ctx context.Context)
	Stop()
}

// FeedManager manages multiple feeds, tracks their status, and exposes
// aggregate diagnostics.
type FeedManager struct {
	log    *zap.Logger
	feeds  []namedFeed
	statMu sync.RWMutex
	stats  map[string]*FeedStat
}

type namedFeed struct {
	name string
	feed Feed
}

// FeedStat holds runtime statistics for a single feed.
type FeedStat struct {
	Name       string
	Running    bool
	StartedAt  time.Time
	StoppedAt  time.Time
	BarsRcvd   int64
	TradesRcvd int64
	Errors     int64
}

// NewFeedManager creates a FeedManager.
func NewFeedManager(log *zap.Logger) *FeedManager {
	return &FeedManager{
		log:   log,
		stats: make(map[string]*FeedStat),
	}
}

// Register adds a feed with a human-readable name.
func (fm *FeedManager) Register(name string, f Feed) {
	fm.statMu.Lock()
	fm.feeds = append(fm.feeds, namedFeed{name: name, feed: f})
	fm.stats[name] = &FeedStat{Name: name}
	fm.statMu.Unlock()
}

// StartAll starts all registered feeds.
func (fm *FeedManager) StartAll(ctx context.Context) {
	fm.statMu.Lock()
	for _, nf := range fm.feeds {
		fm.stats[nf.name].Running = true
		fm.stats[nf.name].StartedAt = time.Now()
	}
	fm.statMu.Unlock()

	for _, nf := range fm.feeds {
		nf.feed.Start(ctx)
		fm.log.Info("feed started", zap.String("name", nf.name))
	}
}

// StopAll stops all registered feeds.
func (fm *FeedManager) StopAll() {
	for _, nf := range fm.feeds {
		nf.feed.Stop()
		fm.statMu.Lock()
		if st, ok := fm.stats[nf.name]; ok {
			st.Running = false
			st.StoppedAt = time.Now()
		}
		fm.statMu.Unlock()
		fm.log.Info("feed stopped", zap.String("name", nf.name))
	}
}

// Stats returns a copy of all feed statistics.
func (fm *FeedManager) Stats() []FeedStat {
	fm.statMu.RLock()
	defer fm.statMu.RUnlock()
	out := make([]FeedStat, 0, len(fm.stats))
	for _, st := range fm.stats {
		out = append(out, *st)
	}
	return out
}

// IncrBars increments the bar counter for a feed.
func (fm *FeedManager) IncrBars(name string) {
	fm.statMu.Lock()
	if st, ok := fm.stats[name]; ok {
		st.BarsRcvd++
	}
	fm.statMu.Unlock()
}

// IncrErrors increments the error counter for a feed.
func (fm *FeedManager) IncrErrors(name string) {
	fm.statMu.Lock()
	if st, ok := fm.stats[name]; ok {
		st.Errors++
	}
	fm.statMu.Unlock()
}

// EventDispatcher is a helper that routes events from a channel to
// handler functions. It is safe for concurrent use.
type EventDispatcher struct {
	barHandlers   []func(Bar)
	tradeHandlers []func(Trade)
	quoteHandlers []func(Quote)
	mu            sync.RWMutex
}

// NewEventDispatcher creates an EventDispatcher.
func NewEventDispatcher() *EventDispatcher {
	return &EventDispatcher{}
}

// OnBar registers a bar handler.
func (d *EventDispatcher) OnBar(h func(Bar)) {
	d.mu.Lock()
	d.barHandlers = append(d.barHandlers, h)
	d.mu.Unlock()
}

// OnTrade registers a trade handler.
func (d *EventDispatcher) OnTrade(h func(Trade)) {
	d.mu.Lock()
	d.tradeHandlers = append(d.tradeHandlers, h)
	d.mu.Unlock()
}

// OnQuote registers a quote handler.
func (d *EventDispatcher) OnQuote(h func(Quote)) {
	d.mu.Lock()
	d.quoteHandlers = append(d.quoteHandlers, h)
	d.mu.Unlock()
}

// Dispatch processes events from ch until ctx is cancelled.
func (d *EventDispatcher) Dispatch(ctx context.Context, ch <-chan Event) {
	for {
		select {
		case <-ctx.Done():
			return
		case evt, ok := <-ch:
			if !ok {
				return
			}
			d.dispatch(evt)
		}
	}
}

func (d *EventDispatcher) dispatch(evt Event) {
	d.mu.RLock()
	bhs := d.barHandlers
	ths := d.tradeHandlers
	qhs := d.quoteHandlers
	d.mu.RUnlock()

	switch evt.Kind {
	case EventBar:
		if evt.Bar != nil {
			for _, h := range bhs {
				h(*evt.Bar)
			}
		}
	case EventTrade:
		if evt.Trade != nil {
			for _, h := range ths {
				h(*evt.Trade)
			}
		}
	case EventQuote:
		if evt.Quote != nil {
			for _, h := range qhs {
				h(*evt.Quote)
			}
		}
	}
}

// ChannelSplitter fans a single event channel out to N output channels.
// This allows multiple independent processors to each receive all events.
type ChannelSplitter struct {
	outputs []chan<- Event
	log     *zap.Logger
}

// NewChannelSplitter creates a ChannelSplitter with the given output channels.
func NewChannelSplitter(log *zap.Logger, outputs ...chan<- Event) *ChannelSplitter {
	return &ChannelSplitter{outputs: outputs, log: log}
}

// Run reads from in and forwards each event to all outputs until ctx is done.
func (cs *ChannelSplitter) Run(ctx context.Context, in <-chan Event) {
	for {
		select {
		case <-ctx.Done():
			return
		case evt, ok := <-in:
			if !ok {
				return
			}
			for _, out := range cs.outputs {
				select {
				case out <- evt:
				default:
					cs.log.Warn("channel splitter: output full, dropping event")
				}
			}
		}
	}
}
