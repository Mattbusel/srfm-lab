package streaming

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"srfm/market-data/aggregator"
)

// ReplayRequest configures a historical replay session.
type ReplayRequest struct {
	Symbol    string    `json:"symbol"`
	Timeframe string    `json:"timeframe"`
	From      time.Time `json:"from"`
	To        time.Time `json:"to"`
	Speed     float64   `json:"speed"` // 1.0 = real-time, 10.0 = 10x, etc.
}

// BarQuerier can retrieve historical bars for replay.
type BarQuerier interface {
	QueryBars(symbol, timeframe string, start, end time.Time, limit int) ([]aggregator.BarEvent, error)
}

// Replayer manages historical replay sessions.
type Replayer struct {
	store      BarQuerier
	hub        *WebSocketHub
	subManager *SubscriptionManager

	mu       sync.Mutex
	sessions map[string]context.CancelFunc
}

// NewReplayer creates a Replayer.
func NewReplayer(store BarQuerier, hub *WebSocketHub, subManager *SubscriptionManager) *Replayer {
	return &Replayer{
		store:      store,
		hub:        hub,
		subManager: subManager,
		sessions:   make(map[string]context.CancelFunc),
	}
}

// StartSession begins a replay session and returns its session ID.
func (r *Replayer) StartSession(req ReplayRequest) (string, error) {
	if req.Speed <= 0 {
		req.Speed = 1.0
	}
	if req.Timeframe == "" {
		req.Timeframe = "1m"
	}
	if req.To.IsZero() {
		req.To = time.Now().UTC()
	}
	if req.From.IsZero() {
		req.From = req.To.Add(-24 * time.Hour)
	}
	if req.From.After(req.To) {
		return "", fmt.Errorf("from must be before to")
	}
	if req.Symbol == "" {
		return "", fmt.Errorf("symbol required")
	}

	sessionID := fmt.Sprintf("replay-%s-%d", req.Symbol, time.Now().UnixNano())

	ctx, cancel := context.WithCancel(context.Background())
	r.mu.Lock()
	r.sessions[sessionID] = cancel
	r.mu.Unlock()

	go r.run(ctx, sessionID, req)
	log.Printf("[replay] started session %s: %s/%s from %s speed %.1fx",
		sessionID, req.Symbol, req.Timeframe, req.From.Format(time.RFC3339), req.Speed)

	return sessionID, nil
}

// StopSession cancels a running replay.
func (r *Replayer) StopSession(sessionID string) bool {
	r.mu.Lock()
	cancel, ok := r.sessions[sessionID]
	if ok {
		cancel()
		delete(r.sessions, sessionID)
	}
	r.mu.Unlock()
	return ok
}

func (r *Replayer) run(ctx context.Context, sessionID string, req ReplayRequest) {
	defer func() {
		r.mu.Lock()
		delete(r.sessions, sessionID)
		r.mu.Unlock()
		log.Printf("[replay] session %s complete", sessionID)
	}()

	bars, err := r.store.QueryBars(req.Symbol, req.Timeframe, req.From, req.To, 5000)
	if err != nil {
		log.Printf("[replay] %s: query error: %v", sessionID, err)
		return
	}
	if len(bars) == 0 {
		log.Printf("[replay] %s: no bars found", sessionID)
		return
	}

	log.Printf("[replay] %s: replaying %d bars", sessionID, len(bars))

	// Determine interval between bars based on timeframe
	tfDurations := map[string]time.Duration{
		"1m": time.Minute, "5m": 5 * time.Minute, "15m": 15 * time.Minute,
		"1h": time.Hour, "4h": 4 * time.Hour, "1d": 24 * time.Hour,
	}
	interval, ok := tfDurations[req.Timeframe]
	if !ok {
		interval = time.Minute
	}
	delay := time.Duration(float64(interval) / req.Speed)

	key := SubscriptionKey{Symbol: req.Symbol, Timeframe: req.Timeframe}

	for i, bar := range bars {
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Mark as replay event
		bar.Source = "replay:" + sessionID

		// Attach sequence number
		type replayMsg struct {
			aggregator.BarEvent
			ReplaySeq   int    `json:"replay_seq"`
			ReplayTotal int    `json:"replay_total"`
			SessionID   string `json:"session_id"`
		}
		msg := replayMsg{
			BarEvent:    bar,
			ReplaySeq:   i + 1,
			ReplayTotal: len(bars),
			SessionID:   sessionID,
		}
		data, err := json.Marshal(msg)
		if err == nil {
			r.hub.BroadcastFiltered(key, data)
		}

		if i < len(bars)-1 && delay > 0 {
			select {
			case <-ctx.Done():
				return
			case <-time.After(delay):
			}
		}
	}
}
