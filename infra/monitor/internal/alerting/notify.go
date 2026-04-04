package alerting

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"go.uber.org/zap"
)

// NotifyChannel is the interface for all notification backends.
type NotifyChannel interface {
	Name() string
	Send(alert Alert) error
}

// ---- LogNotifier ----

// LogNotifier writes alerts to a zap logger.
type LogNotifier struct {
	log *zap.Logger
}

// NewLogNotifier creates a LogNotifier.
func NewLogNotifier(log *zap.Logger) *LogNotifier {
	return &LogNotifier{log: log}
}

func (n *LogNotifier) Name() string { return "log" }

func (n *LogNotifier) Send(a Alert) error {
	n.log.Info("ALERT",
		zap.String("id", a.ID),
		zap.String("rule", a.Rule.Name),
		zap.String("level", string(a.Level)),
		zap.String("symbol", a.Symbol),
		zap.String("metric", a.Metric),
		zap.Float64("value", a.Value),
		zap.Float64("threshold", a.Threshold),
		zap.String("message", a.Message),
		zap.Time("fired_at", a.FiredAt))
	return nil
}

// ---- WebhookNotifier ----

// WebhookNotifier POSTs alerts as JSON to a configured URL.
type WebhookNotifier struct {
	url    string
	client *http.Client
	log    *zap.Logger
}

// NewWebhookNotifier creates a WebhookNotifier.
func NewWebhookNotifier(url string, log *zap.Logger) *WebhookNotifier {
	return &WebhookNotifier{
		url: url,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
		log: log,
	}
}

func (n *WebhookNotifier) Name() string { return "webhook" }

func (n *WebhookNotifier) Send(a Alert) error {
	payload := map[string]interface{}{
		"id":        a.ID,
		"rule":      a.Rule.Name,
		"level":     string(a.Level),
		"symbol":    a.Symbol,
		"metric":    a.Metric,
		"value":     a.Value,
		"threshold": a.Threshold,
		"message":   a.Message,
		"timestamp": a.Timestamp.Format(time.RFC3339),
		"fired_at":  a.FiredAt.Format(time.RFC3339),
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal webhook payload: %w", err)
	}

	resp, err := n.client.Post(n.url, "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("post webhook: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("webhook returned %d", resp.StatusCode)
	}
	return nil
}

// ---- FileNotifier ----

// FileNotifier appends alerts to a CSV file.
type FileNotifier struct {
	mu   sync.Mutex
	path string
	log  *zap.Logger
}

// NewFileNotifier creates a FileNotifier.
func NewFileNotifier(path string, log *zap.Logger) (*FileNotifier, error) {
	n := &FileNotifier{path: path, log: log}
	// Write header if file does not exist.
	if _, err := os.Stat(path); os.IsNotExist(err) {
		f, err := os.Create(path)
		if err != nil {
			return nil, fmt.Errorf("create alert file: %w", err)
		}
		w := csv.NewWriter(f)
		_ = w.Write([]string{
			"fired_at", "id", "rule", "level", "symbol",
			"metric", "value", "threshold", "message",
		})
		w.Flush()
		f.Close()
	}
	return n, nil
}

func (n *FileNotifier) Name() string { return "file" }

func (n *FileNotifier) Send(a Alert) error {
	n.mu.Lock()
	defer n.mu.Unlock()

	f, err := os.OpenFile(n.path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open alert file: %w", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	err = w.Write([]string{
		a.FiredAt.Format(time.RFC3339),
		a.ID,
		a.Rule.Name,
		string(a.Level),
		a.Symbol,
		a.Metric,
		strconv.FormatFloat(a.Value, 'f', 6, 64),
		strconv.FormatFloat(a.Threshold, 'f', 6, 64),
		a.Message,
	})
	if err != nil {
		return fmt.Errorf("write csv: %w", err)
	}
	w.Flush()
	return w.Error()
}

// ---- MultiNotifier ----

// MultiNotifier dispatches an alert to several channels, collecting all errors.
type MultiNotifier struct {
	channels []NotifyChannel
}

// NewMultiNotifier creates a MultiNotifier.
func NewMultiNotifier(channels ...NotifyChannel) *MultiNotifier {
	return &MultiNotifier{channels: channels}
}

func (m *MultiNotifier) Name() string { return "multi" }

func (m *MultiNotifier) Send(a Alert) error {
	var errs []error
	for _, ch := range m.channels {
		if err := ch.Send(a); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", ch.Name(), err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("multi-notify errors: %v", errs)
	}
	return nil
}

// Add appends a new channel to the multi-notifier at runtime.
func (m *MultiNotifier) Add(ch NotifyChannel) {
	m.channels = append(m.channels, ch)
}
