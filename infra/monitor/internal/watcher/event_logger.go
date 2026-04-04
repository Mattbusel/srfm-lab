package watcher

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"go.uber.org/zap"
)

// EventRecord is a persisted event entry.
type EventRecord struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	Symbol    string                 `json:"symbol"`
	Timeframe string                 `json:"timeframe,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data,omitempty"`
}

// EventLogger writes structured events to a rotating JSONL log file.
type EventLogger struct {
	log     *zap.Logger
	mu      sync.Mutex
	file    *os.File
	path    string
	maxSize int64 // max file size before rotation (bytes)
	written int64
}

// NewEventLogger creates an EventLogger that writes to path.
func NewEventLogger(path string, maxSizeMB int, log *zap.Logger) (*EventLogger, error) {
	if maxSizeMB <= 0 {
		maxSizeMB = 50
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("mkdir: %w", err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open event log: %w", err)
	}
	fi, _ := f.Stat()
	written := int64(0)
	if fi != nil {
		written = fi.Size()
	}
	return &EventLogger{
		log:     log,
		file:    f,
		path:    path,
		maxSize: int64(maxSizeMB) * 1024 * 1024,
		written: written,
	}, nil
}

// Log writes an event record to the log file.
func (el *EventLogger) Log(rec EventRecord) error {
	if rec.ID == "" {
		rec.ID = fmt.Sprintf("%s-%d", rec.Type, time.Now().UnixNano())
	}
	if rec.Timestamp.IsZero() {
		rec.Timestamp = time.Now()
	}

	data, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("marshal event: %w", err)
	}
	data = append(data, '\n')

	el.mu.Lock()
	defer el.mu.Unlock()

	// Rotate if needed.
	if el.written+int64(len(data)) > el.maxSize {
		if err := el.rotate(); err != nil {
			el.log.Warn("event log rotation failed", zap.Error(err))
		}
	}

	n, err := el.file.Write(data)
	el.written += int64(n)
	return err
}

// Close closes the log file.
func (el *EventLogger) Close() error {
	el.mu.Lock()
	defer el.mu.Unlock()
	if el.file != nil {
		return el.file.Close()
	}
	return nil
}

// rotate renames the current file and opens a new one.
func (el *EventLogger) rotate() error {
	el.file.Close()
	rotated := el.path + "." + time.Now().UTC().Format("20060102_150405")
	if err := os.Rename(el.path, rotated); err != nil {
		el.log.Warn("rename event log", zap.Error(err))
	}
	f, err := os.Create(el.path)
	if err != nil {
		return fmt.Errorf("create new event log: %w", err)
	}
	el.file = f
	el.written = 0
	el.log.Info("event log rotated", zap.String("rotated_to", rotated))
	return nil
}

// ReadEvents reads the most recent n events from the log file.
func (el *EventLogger) ReadEvents(n int) ([]EventRecord, error) {
	el.mu.Lock()
	path := el.path
	el.mu.Unlock()

	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var records []EventRecord
	dec := json.NewDecoder(f)
	for dec.More() {
		var rec EventRecord
		if err := dec.Decode(&rec); err != nil {
			continue
		}
		records = append(records, rec)
	}

	if n > 0 && len(records) > n {
		records = records[len(records)-n:]
	}
	return records, nil
}

// EventType constants for the event log.
const (
	EventTypeBHFormation = "bh_formation"
	EventTypeBHCollapse  = "bh_collapse"
	EventTypeBHMassSpike = "bh_mass_spike"
	EventTypeAlertFired  = "alert_fired"
	EventTypePortfolioDD = "portfolio_drawdown"
	EventTypeTradeEntry  = "trade_entry"
	EventTypeTradeExit   = "trade_exit"
	EventTypeConnected   = "feed_connected"
	EventTypeDisconnected = "feed_disconnected"
)

// LogBHEvent writes a BH event record.
func (el *EventLogger) LogBHEvent(symbol, timeframe, eventType string, mass float64, extra map[string]interface{}) {
	data := map[string]interface{}{
		"mass":       mass,
		"event_type": eventType,
	}
	for k, v := range extra {
		data[k] = v
	}
	if err := el.Log(EventRecord{
		Type:      eventType,
		Symbol:    symbol,
		Timeframe: timeframe,
		Timestamp: time.Now(),
		Data:      data,
	}); err != nil {
		el.log.Warn("log bh event", zap.Error(err))
	}
}

// LogAlertFired writes an alert-fired record.
func (el *EventLogger) LogAlertFired(id, rule, symbol, metric, level string, value, threshold float64) {
	if err := el.Log(EventRecord{
		ID:        id,
		Type:      EventTypeAlertFired,
		Symbol:    symbol,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"rule":      rule,
			"metric":    metric,
			"level":     level,
			"value":     value,
			"threshold": threshold,
		},
	}); err != nil {
		el.log.Warn("log alert", zap.Error(err))
	}
}
