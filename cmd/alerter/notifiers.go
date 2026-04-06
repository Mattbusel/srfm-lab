// notifiers.go — Alert notification backends.
//
// SlackNotifier: POST to Slack Blocks API with ASCII sparkline.
// WebhookNotifier: configurable HTTP POST with JSON payload.
// LogNotifier: structured JSON to alerts.log.
// CompositeNotifier: fan-out with retry (3 attempts, exponential backoff).

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"os"
	"strings"
	"time"
)

// ─── Notifier Interface ───────────────────────────────────────────────────────

// Notifier dispatches a single alert event.
type Notifier interface {
	Notify(ctx context.Context, alert *Alert) error
	Name() string
}

// ─── Retry Helper ─────────────────────────────────────────────────────────────

const (
	maxRetries    = 3
	baseBackoffMs = 200
)

func withRetry(ctx context.Context, name string, logger *slog.Logger, fn func() error) error {
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(baseBackoffMs*(1<<attempt)) * time.Millisecond
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}
		if err := fn(); err != nil {
			lastErr = err
			logger.Warn("notifier retry",
				"notifier", name, "attempt", attempt+1, "err", err)
			continue
		}
		return nil
	}
	return fmt.Errorf("%s failed after %d attempts: %w", name, maxRetries, lastErr)
}

// ─── SlackNotifier ────────────────────────────────────────────────────────────

// SlackNotifier sends alerts to a Slack incoming webhook using the Blocks API.
type SlackNotifier struct {
	webhookURL string
	client     *http.Client
	logger     *slog.Logger
}

func NewSlackNotifier(webhookURL string, logger *slog.Logger) *SlackNotifier {
	return &SlackNotifier{
		webhookURL: webhookURL,
		client:     &http.Client{Timeout: 10 * time.Second},
		logger:     logger,
	}
}

func (s *SlackNotifier) Name() string { return "slack" }

func (s *SlackNotifier) Notify(ctx context.Context, alert *Alert) error {
	payload := s.buildPayload(alert)
	return withRetry(ctx, s.Name(), s.logger, func() error {
		body, err := json.Marshal(payload)
		if err != nil { return err }
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.webhookURL, bytes.NewReader(body))
		if err != nil { return err }
		req.Header.Set("Content-Type", "application/json")
		resp, err := s.client.Do(req)
		if err != nil { return err }
		defer resp.Body.Close()
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return fmt.Errorf("slack returned HTTP %d", resp.StatusCode)
		}
		return nil
	})
}

// buildPayload constructs a Slack Blocks API message.
func (s *SlackNotifier) buildPayload(alert *Alert) map[string]any {
	emoji := alertEmoji(alert.Severity, alert.State)
	stateLabel := strings.ToUpper(string(alert.State))
	color := alertColor(alert.Severity, alert.State)

	sparkline := trendSparkline(alert.MetricValue, alert.Threshold)

	headerText := fmt.Sprintf("%s [%s] %s", emoji, stateLabel, alert.RuleName)
	bodyText := fmt.Sprintf(
		"*Message:* %s\n*Metric:* `%s` = `%.4f`  (threshold: `%.4f`)\n*Trend:* `%s`\n*Severity:* %s\n*Fired:* %s",
		alert.Message,
		alert.MetricName, alert.MetricValue, alert.Threshold,
		sparkline,
		string(alert.Severity),
		alert.FiredAt.Format(time.RFC3339),
	)
	if alert.ResolvedAt != nil {
		bodyText += fmt.Sprintf("\n*Resolved:* %s  _(duration: %s)_",
			alert.ResolvedAt.Format(time.RFC3339), alert.Duration)
	}

	return map[string]any{
		"attachments": []map[string]any{
			{
				"color": color,
				"blocks": []map[string]any{
					{
						"type": "header",
						"text": map[string]any{
							"type": "plain_text",
							"text": headerText,
						},
					},
					{
						"type": "section",
						"text": map[string]any{
							"type": "mrkdwn",
							"text": bodyText,
						},
					},
					{
						"type": "context",
						"elements": []map[string]any{
							{
								"type": "mrkdwn",
								"text": fmt.Sprintf("Alert ID: `%s`", alert.ID),
							},
						},
					},
				},
			},
		},
	}
}

func alertEmoji(sev Severity, state AlertState) string {
	if state == AlertStateResolved { return "✅" }
	switch sev {
	case SeverityCritical: return "🔴"
	case SeverityWarning:  return "🟡"
	case SeverityError:    return "🟠"
	default:               return "🔵"
	}
}

func alertColor(sev Severity, state AlertState) string {
	if state == AlertStateResolved { return "good" }
	switch sev {
	case SeverityCritical: return "danger"
	case SeverityWarning:  return "warning"
	default:               return "#439FE0"
	}
}

// trendSparkline returns an ASCII representation of value vs threshold.
// Uses block elements to give a quick visual impression.
func trendSparkline(value, threshold float64) string {
	const width = 10
	if math.IsNaN(value) || math.IsInf(value, 0) || threshold == 0 {
		return "──────────"
	}
	ratio := value / threshold
	// Map ratio to bar length.
	filled := int(math.Round(math.Abs(ratio) * width))
	if filled > width { filled = width }
	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	sign := "+"
	if value < 0 { sign = "-" }
	return fmt.Sprintf("[%s] %s%.4f / %.4f", bar, sign, math.Abs(value), math.Abs(threshold))
}

// ─── WebhookNotifier ──────────────────────────────────────────────────────────

// WebhookNotifier POSTs JSON alert payloads to a configurable URL.
type WebhookNotifier struct {
	url    string
	client *http.Client
	logger *slog.Logger
}

func NewWebhookNotifier(url string, logger *slog.Logger) *WebhookNotifier {
	return &WebhookNotifier{
		url:    url,
		client: &http.Client{Timeout: 10 * time.Second},
		logger: logger,
	}
}

func (w *WebhookNotifier) Name() string { return "webhook" }

func (w *WebhookNotifier) Notify(ctx context.Context, alert *Alert) error {
	return withRetry(ctx, w.Name(), w.logger, func() error {
		payload := map[string]any{
			"version":      "1.0",
			"source":       "srfm-alerter",
			"alert_id":     alert.ID,
			"rule":         alert.RuleName,
			"severity":     string(alert.Severity),
			"state":        string(alert.State),
			"message":      alert.Message,
			"metric_name":  alert.MetricName,
			"metric_value": alert.MetricValue,
			"threshold":    alert.Threshold,
			"fired_at":     alert.FiredAt.Format(time.RFC3339),
			"resolved_at":  alert.ResolvedAt,
		}
		body, err := json.Marshal(payload)
		if err != nil { return err }
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, w.url, bytes.NewReader(body))
		if err != nil { return err }
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-SRFM-Source", "alerter")
		resp, err := w.client.Do(req)
		if err != nil { return err }
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			return fmt.Errorf("webhook returned HTTP %d", resp.StatusCode)
		}
		return nil
	})
}

// ─── LogNotifier ─────────────────────────────────────────────────────────────

// LogNotifier writes structured JSON alert lines to alerts.log.
type LogNotifier struct {
	path   string
	logger *slog.Logger
	fileLogger *slog.Logger
}

func NewLogNotifier(path string, logger *slog.Logger) *LogNotifier {
	n := &LogNotifier{path: path, logger: logger}
	if f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); err == nil {
		n.fileLogger = slog.New(slog.NewJSONHandler(f, nil))
	} else {
		logger.Warn("could not open alerts log, falling back to stdout", "path", path, "err", err)
		n.fileLogger = logger
	}
	return n
}

func (l *LogNotifier) Name() string { return "log" }

func (l *LogNotifier) Notify(_ context.Context, alert *Alert) error {
	level := slog.LevelInfo
	switch alert.Severity {
	case SeverityWarning:  level = slog.LevelWarn
	case SeverityCritical, SeverityError: level = slog.LevelError
	}
	l.fileLogger.Log(context.Background(), level, alert.Message,
		"alert_id",     alert.ID,
		"rule",         alert.RuleName,
		"severity",     string(alert.Severity),
		"state",        string(alert.State),
		"metric",       alert.MetricName,
		"value",        alert.MetricValue,
		"threshold",    alert.Threshold,
		"fired_at",     alert.FiredAt.Format(time.RFC3339),
	)
	return nil
}

// ─── StdoutNotifier ───────────────────────────────────────────────────────────

// StdoutNotifier writes human-readable alert lines to stdout.
type StdoutNotifier struct {
	logger *slog.Logger
}

func NewStdoutNotifier(logger *slog.Logger) *StdoutNotifier {
	return &StdoutNotifier{logger: logger}
}

func (s *StdoutNotifier) Name() string { return "stdout" }

func (s *StdoutNotifier) Notify(_ context.Context, alert *Alert) error {
	stateStr := strings.ToUpper(string(alert.State))
	sevStr := strings.ToUpper(string(alert.Severity))
	fmt.Printf("[ALERT][%s][%s] %s | %s=%.4f (threshold %.4f) at %s\n",
		stateStr, sevStr,
		alert.Message,
		alert.MetricName, alert.MetricValue, alert.Threshold,
		alert.FiredAt.Format("2006-01-02 15:04:05"),
	)
	return nil
}

// ─── CompositeNotifier ────────────────────────────────────────────────────────

// CompositeNotifier fans out to multiple notifiers, continuing on partial failures.
type CompositeNotifier struct {
	notifiers []Notifier
	logger    *slog.Logger
}

func NewCompositeNotifier(notifiers []Notifier) CompositeNotifier {
	return CompositeNotifier{notifiers: notifiers}
}

func (c *CompositeNotifier) SetLogger(l *slog.Logger) { c.logger = l }

func (c CompositeNotifier) Notify(ctx context.Context, alert *Alert) error {
	var firstErr error
	for _, n := range c.notifiers {
		if err := n.Notify(ctx, alert); err != nil {
			if firstErr == nil { firstErr = err }
			if c.logger != nil {
				c.logger.Error("notifier error", "notifier", n.Name(), "err", err)
			}
		}
	}
	return firstErr
}
