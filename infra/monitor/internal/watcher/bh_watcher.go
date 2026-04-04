package watcher

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/srfm/monitor/internal/alerting"
	"go.uber.org/zap"
)

// BHStateFile is the default path for the live BH engine state.
const BHStateFile = "/c/Users/Matthew/srfm-lab/spacetime/cache/live_state.json"

// BHConfig configures the BH state watcher.
type BHConfig struct {
	StateFilePath   string
	PollEvery       time.Duration
	MassThreshold   float64 // mass above which to emit "formation"
	CollapseThreshold float64 // mass below which (after being above) = "collapse"
}

// DefaultBHConfig returns a sensible default.
func DefaultBHConfig() BHConfig {
	return BHConfig{
		StateFilePath:   BHStateFile,
		PollEvery:       10 * time.Second,
		MassThreshold:   0.7,
		CollapseThreshold: 0.3,
	}
}

// bhSymbolState is the per-symbol state JSON structure written by the Spacetime engine.
type bhSymbolState struct {
	Symbol       string             `json:"symbol"`
	Timeframe    string             `json:"timeframe"`
	Mass         float64            `json:"mass"`
	Active       bool               `json:"active"`
	LastUpdated  string             `json:"last_updated"`
	Indicators   map[string]float64 `json:"indicators,omitempty"`
}

// bhLiveState is the top-level structure of live_state.json.
type bhLiveState struct {
	UpdatedAt string          `json:"updated_at"`
	States    []bhSymbolState `json:"states"`
}

// BHEventHandler is called when a BH event is detected.
type BHEventHandler func(event alerting.BHEvent)

// BHWatcher polls the BH engine live_state.json and emits BHEvents.
type BHWatcher struct {
	cfg      BHConfig
	log      *zap.Logger
	handlers []BHEventHandler

	mu       sync.Mutex
	prevMass map[string]float64 // symbol+timeframe -> last known mass
	prevAbove map[string]bool   // symbol+timeframe -> was above threshold
}

// NewBHWatcher creates a BHWatcher.
func NewBHWatcher(cfg BHConfig, log *zap.Logger) *BHWatcher {
	if cfg.PollEvery == 0 {
		cfg.PollEvery = 10 * time.Second
	}
	if cfg.StateFilePath == "" {
		cfg.StateFilePath = BHStateFile
	}
	return &BHWatcher{
		cfg:       cfg,
		log:       log,
		prevMass:  make(map[string]float64),
		prevAbove: make(map[string]bool),
	}
}

// AddHandler registers a callback for BHEvent notifications.
func (bw *BHWatcher) AddHandler(h BHEventHandler) {
	bw.mu.Lock()
	defer bw.mu.Unlock()
	bw.handlers = append(bw.handlers, h)
}

// Run starts polling in the current goroutine. Cancel ctx to stop.
func (bw *BHWatcher) Run(ctx context.Context) {
	ticker := time.NewTicker(bw.cfg.PollEvery)
	defer ticker.Stop()

	// Poll immediately.
	bw.poll()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			bw.poll()
		}
	}
}

func (bw *BHWatcher) poll() {
	state, err := bw.readStateFile()
	if err != nil {
		if !os.IsNotExist(err) {
			bw.log.Warn("bh watcher: read state file", zap.Error(err))
		}
		return
	}

	now := time.Now()
	var updatedAt time.Time
	if state.UpdatedAt != "" {
		updatedAt, _ = time.Parse(time.RFC3339, state.UpdatedAt)
	}
	if updatedAt.IsZero() {
		updatedAt = now
	}

	bw.mu.Lock()
	defer bw.mu.Unlock()

	for _, s := range state.States {
		key := s.Symbol + "|" + s.Timeframe
		prevMass := bw.prevMass[key]
		wasAbove := bw.prevAbove[key]
		nowAbove := s.Mass >= bw.cfg.MassThreshold

		bw.prevMass[key] = s.Mass

		var events []alerting.BHEvent

		if !wasAbove && nowAbove {
			// Crossed above threshold: formation.
			events = append(events, alerting.BHEvent{
				Symbol:    s.Symbol,
				Timeframe: s.Timeframe,
				EventType: "formation",
				Mass:      s.Mass,
				Timestamp: updatedAt,
				Extra: map[string]interface{}{
					"indicators": s.Indicators,
				},
			})
			bw.log.Info("BH formation detected",
				zap.String("symbol", s.Symbol),
				zap.String("timeframe", s.Timeframe),
				zap.Float64("mass", s.Mass))
		} else if wasAbove && s.Mass < bw.cfg.CollapseThreshold {
			// Mass dropped below collapse threshold: collapse.
			events = append(events, alerting.BHEvent{
				Symbol:    s.Symbol,
				Timeframe: s.Timeframe,
				EventType: "collapse",
				Mass:      s.Mass,
				Timestamp: updatedAt,
				Extra: map[string]interface{}{
					"prev_mass":  prevMass,
					"indicators": s.Indicators,
				},
			})
			bw.log.Info("BH collapse detected",
				zap.String("symbol", s.Symbol),
				zap.String("timeframe", s.Timeframe),
				zap.Float64("mass", s.Mass))
		} else if nowAbove && s.Mass > prevMass*1.2 && prevMass > 0 {
			// Mass spiked by > 20%: mass_spike.
			events = append(events, alerting.BHEvent{
				Symbol:    s.Symbol,
				Timeframe: s.Timeframe,
				EventType: "mass_spike",
				Mass:      s.Mass,
				Timestamp: updatedAt,
				Extra: map[string]interface{}{
					"prev_mass":  prevMass,
					"pct_change": (s.Mass - prevMass) / prevMass * 100,
				},
			})
		}

		bw.prevAbove[key] = nowAbove

		if len(events) > 0 {
			hs := bw.handlers
			for _, evt := range events {
				for _, h := range hs {
					h(evt)
				}
			}
		}
	}
}

func (bw *BHWatcher) readStateFile() (*bhLiveState, error) {
	f, err := os.Open(bw.cfg.StateFilePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var state bhLiveState
	if err := json.NewDecoder(f).Decode(&state); err != nil {
		return nil, fmt.Errorf("decode bh state: %w", err)
	}
	return &state, nil
}

// CurrentMasses returns a snapshot of all current mass values.
func (bw *BHWatcher) CurrentMasses() map[string]float64 {
	bw.mu.Lock()
	defer bw.mu.Unlock()
	out := make(map[string]float64, len(bw.prevMass))
	for k, v := range bw.prevMass {
		out[k] = v
	}
	return out
}
