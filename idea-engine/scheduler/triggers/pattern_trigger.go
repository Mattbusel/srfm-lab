package triggers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"go.uber.org/zap"
)

// PatternBatch accumulates discovered patterns and fires a callback when the
// batch reaches a configurable threshold.
//
// PatternTrigger subscribes to the bus patterns.discovered topic via an HTTP
// long-poll or webhook. When at least minPatterns new patterns have accumulated
// since the last trigger, it calls the registered callback.
type PatternTrigger struct {
	// Callback is invoked when the batch threshold is reached.
	// patternIDs is the list of pattern IDs in the current batch.
	Callback func(ctx context.Context, patternIDs []string)
	// MinPatterns is the minimum number of patterns required to fire.
	MinPatterns int
	// BusURL is the base URL of the bus service.
	BusURL string

	mu         sync.Mutex
	pending    []string // accumulated pattern IDs since last trigger
	lastFired  time.Time
	httpClient *http.Client
	log        *zap.Logger
}

// NewPatternTrigger constructs a PatternTrigger.
// minPatterns defaults to 5 if <= 0.
func NewPatternTrigger(
	callback func(ctx context.Context, patternIDs []string),
	busURL string,
	minPatterns int,
	log *zap.Logger,
) *PatternTrigger {
	if minPatterns <= 0 {
		minPatterns = 5
	}
	return &PatternTrigger{
		Callback:    callback,
		MinPatterns: minPatterns,
		BusURL:      busURL,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
		log: log,
	}
}

// HandleBusEvent is the handler invoked by the bus subscription (or by a
// direct HTTP callback from the bus adapter). It unmarshals the event payload,
// accumulates pattern IDs, and fires the callback if the threshold is reached.
func (t *PatternTrigger) HandleBusEvent(ctx context.Context, eventPayload []byte) error {
	var payload struct {
		Patterns []struct {
			PatternID string `json:"pattern_id"`
		} `json:"patterns"`
		PatternCount int    `json:"pattern_count"`
		RunID        string `json:"run_id"`
	}
	if err := json.Unmarshal(eventPayload, &payload); err != nil {
		return fmt.Errorf("pattern_trigger: unmarshal payload: %w", err)
	}

	t.mu.Lock()
	for _, p := range payload.Patterns {
		if p.PatternID != "" {
			t.pending = append(t.pending, p.PatternID)
		}
	}
	count := len(t.pending)
	var toFire []string
	if count >= t.MinPatterns {
		toFire = make([]string, len(t.pending))
		copy(toFire, t.pending)
		t.pending = t.pending[:0]
		t.lastFired = time.Now().UTC()
	}
	t.mu.Unlock()

	if len(toFire) > 0 {
		t.log.Info("pattern_trigger: threshold reached, firing callback",
			zap.Int("pattern_count", len(toFire)),
			zap.Int("threshold", t.MinPatterns),
		)
		go func() {
			cbCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
			defer cancel()
			t.Callback(cbCtx, toFire)
		}()
	} else {
		t.log.Debug("pattern_trigger: accumulating patterns",
			zap.Int("pending", count),
			zap.Int("threshold", t.MinPatterns),
		)
	}
	return nil
}

// PollBus runs a polling loop that fetches new patterns.discovered events from
// the bus replay endpoint and processes them. It fires once per poll interval.
// Use this when the bus does not support push delivery.
func (t *PatternTrigger) PollBus(ctx context.Context, pollInterval time.Duration) {
	if pollInterval <= 0 {
		pollInterval = 2 * time.Minute
	}

	t.log.Info("pattern_trigger: starting bus poll loop",
		zap.Duration("interval", pollInterval),
		zap.String("bus_url", t.BusURL),
	)

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	since := time.Now().UTC().Add(-24 * time.Hour) // start from 24h ago on first run

	for {
		select {
		case <-ctx.Done():
			t.log.Info("pattern_trigger: poll loop stopped")
			return
		case <-ticker.C:
			events, err := t.fetchEvents(ctx, since)
			if err != nil {
				t.log.Warn("pattern_trigger: fetch failed", zap.Error(err))
				continue
			}
			since = time.Now().UTC()
			for _, evt := range events {
				if err := t.HandleBusEvent(ctx, evt); err != nil {
					t.log.Warn("pattern_trigger: handle event failed", zap.Error(err))
				}
			}
		}
	}
}

// fetchEvents queries the bus replay endpoint for patterns.discovered events
// since the given timestamp.
func (t *PatternTrigger) fetchEvents(ctx context.Context, since time.Time) ([]json.RawMessage, error) {
	url := fmt.Sprintf("%s/replay?topic=patterns.discovered&since=%s",
		t.BusURL,
		since.UTC().Format(time.RFC3339),
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("pattern_trigger: build request: %w", err)
	}

	resp, err := t.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("pattern_trigger: GET replay: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("pattern_trigger: replay returned %d", resp.StatusCode)
	}

	var body struct {
		Events []struct {
			Payload json.RawMessage `json:"payload"`
		} `json:"events"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("pattern_trigger: decode response: %w", err)
	}

	payloads := make([]json.RawMessage, 0, len(body.Events))
	for _, e := range body.Events {
		payloads = append(payloads, e.Payload)
	}
	return payloads, nil
}

// PendingCount returns the number of pattern IDs accumulated since the last trigger.
// Useful for health/stats endpoints.
func (t *PatternTrigger) PendingCount() int {
	t.mu.Lock()
	n := len(t.pending)
	t.mu.Unlock()
	return n
}

// LastFired returns the time the callback was last invoked (zero if never).
func (t *PatternTrigger) LastFired() time.Time {
	t.mu.Lock()
	ts := t.lastFired
	t.mu.Unlock()
	return ts
}
