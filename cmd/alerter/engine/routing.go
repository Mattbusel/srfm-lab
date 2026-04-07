// cmd/alerter/engine/routing.go -- Alert routing and notification dispatch.
//
// AlertRouter routes each alert by severity to the appropriate set of backends:
//   EMERGENCY -> PagerDuty + Slack + SMS (stub)
//   CRITICAL  -> PagerDuty + Slack
//   WARNING   -> Slack only
//   INFO      -> log only
//
// Rate limiting: max 5 Slack messages per minute per channel.
// Maintenance windows: suppress all non-EMERGENCY alerts when active.

package engine

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Notifier interface
// ---------------------------------------------------------------------------

// Notifier is implemented by all downstream notification backends.
type Notifier interface {
	Send(ctx context.Context, alert *Alert) error
	BackendName() string
}

// ---------------------------------------------------------------------------
// MaintenanceWindow
// ---------------------------------------------------------------------------

// MaintenanceWindow represents a scheduled suppression period.
type MaintenanceWindow struct {
	Label     string
	StartsAt  time.Time
	EndsAt    time.Time
}

// Active returns true if the window is currently in effect.
func (mw MaintenanceWindow) Active() bool {
	now := time.Now()
	return now.After(mw.StartsAt) && now.Before(mw.EndsAt)
}

// ---------------------------------------------------------------------------
// AlertRouter
// ---------------------------------------------------------------------------

// routerConfig holds all injectable configuration for the router.
type routerConfig struct {
	slackRateLimit int // messages per minute per channel (default 5)
}

// AlertRouter dispatches alerts to the correct notifiers based on severity,
// enforces rate limits, and respects maintenance windows.
type AlertRouter struct {
	mu          sync.RWMutex
	slackN      *SlackNotifier
	pagerN      *PagerDutyNotifier
	sentryN     *SentryNotifier
	maintenance []MaintenanceWindow
	cfg         routerConfig
	logger      *slog.Logger
	// silenced rule names -> expiry
	silenced map[string]time.Time
}

// RouterOption is a functional option for NewAlertRouter.
type RouterOption func(*AlertRouter)

// WithSlack attaches a SlackNotifier.
func WithSlack(n *SlackNotifier) RouterOption {
	return func(r *AlertRouter) { r.slackN = n }
}

// WithPagerDuty attaches a PagerDutyNotifier.
func WithPagerDuty(n *PagerDutyNotifier) RouterOption {
	return func(r *AlertRouter) { r.pagerN = n }
}

// WithSentry attaches a SentryNotifier.
func WithSentry(n *SentryNotifier) RouterOption {
	return func(r *AlertRouter) { r.sentryN = n }
}

// WithSlackRateLimit sets a custom Slack rate limit (messages per minute).
func WithSlackRateLimit(n int) RouterOption {
	return func(r *AlertRouter) { r.cfg.slackRateLimit = n }
}

// NewAlertRouter constructs an AlertRouter with optional backends.
func NewAlertRouter(logger *slog.Logger, opts ...RouterOption) *AlertRouter {
	r := &AlertRouter{
		logger:   logger,
		silenced: make(map[string]time.Time),
		cfg: routerConfig{
			slackRateLimit: 5,
		},
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Route dispatches a single alert to the appropriate notifiers.
func (r *AlertRouter) Route(ctx context.Context, alert *Alert) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Check silence.
	if exp, ok := r.silenced[alert.RuleName]; ok {
		if time.Now().Before(exp) {
			r.logger.Debug("alert silenced", "rule", alert.RuleName, "expires", exp)
			return nil
		}
	}

	// Check maintenance windows.
	if alert.Severity != SeverityEmergency && r.inMaintenance() {
		r.logger.Info("alert suppressed by maintenance window",
			"rule", alert.RuleName, "severity", alert.Severity)
		return nil
	}

	var errs []string

	switch alert.Severity {
	case SeverityEmergency:
		// All channels.
		if r.pagerN != nil {
			if err := r.pagerN.Send(ctx, alert); err != nil {
				errs = append(errs, fmt.Sprintf("pagerduty: %v", err))
				r.logger.Error("pagerduty send failed", "err", err)
			}
		}
		if r.slackN != nil {
			if err := r.slackN.Send(ctx, alert); err != nil {
				errs = append(errs, fmt.Sprintf("slack: %v", err))
				r.logger.Error("slack send failed", "err", err)
			}
		}
		// SMS stub -- log as would integrate with Twilio or AWS SNS.
		r.logger.Warn("SMS notification triggered (stub)",
			"rule", alert.RuleName, "severity", alert.Severity)

	case SeverityCritical:
		if r.pagerN != nil {
			if err := r.pagerN.Send(ctx, alert); err != nil {
				errs = append(errs, fmt.Sprintf("pagerduty: %v", err))
				r.logger.Error("pagerduty send failed", "err", err)
			}
		}
		if r.slackN != nil {
			if err := r.slackN.Send(ctx, alert); err != nil {
				errs = append(errs, fmt.Sprintf("slack: %v", err))
			}
		}

	case SeverityWarning:
		if r.slackN != nil {
			if err := r.slackN.Send(ctx, alert); err != nil {
				errs = append(errs, fmt.Sprintf("slack: %v", err))
			}
		}

	default: // INFO
		r.logger.Info("info-level alert", "rule", alert.RuleName, "msg", alert.Message)
	}

	// Always relay CRITICAL+ to Sentry for debugging.
	if r.sentryN != nil && (alert.Severity == SeverityCritical || alert.Severity == SeverityEmergency) {
		if err := r.sentryN.Send(ctx, alert); err != nil {
			errs = append(errs, fmt.Sprintf("sentry: %v", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("routing partial failure: %s", strings.Join(errs, "; "))
	}
	return nil
}

// Silence suppresses a rule for the given duration.
func (r *AlertRouter) Silence(ruleName string, duration time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.silenced[ruleName] = time.Now().Add(duration)
	r.logger.Info("rule silenced", "rule", ruleName, "duration", duration)
}

// SetMaintenance replaces the current maintenance window list.
func (r *AlertRouter) SetMaintenance(windows []MaintenanceWindow) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.maintenance = windows
	r.logger.Info("maintenance windows updated", "count", len(windows))
}

// inMaintenance returns true when any configured window is currently active.
// Caller must hold at least a read lock.
func (r *AlertRouter) inMaintenance() bool {
	for _, mw := range r.maintenance {
		if mw.Active() {
			return true
		}
	}
	return false
}

// ActiveMaintenance returns a copy of all currently active maintenance windows.
func (r *AlertRouter) ActiveMaintenance() []MaintenanceWindow {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var out []MaintenanceWindow
	for _, mw := range r.maintenance {
		if mw.Active() {
			out = append(out, mw)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// SlackNotifier
// ---------------------------------------------------------------------------

// slackRateBucket is a per-channel token bucket for rate limiting.
type slackRateBucket struct {
	mu        sync.Mutex
	count     int
	windowEnd time.Time
	limit     int // per minute
}

func (b *slackRateBucket) allow() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	now := time.Now()
	if now.After(b.windowEnd) {
		b.count = 0
		b.windowEnd = now.Add(time.Minute)
	}
	if b.count >= b.limit {
		return false
	}
	b.count++
	return true
}

// SlackNotifier formats and delivers rich Slack messages with context.
type SlackNotifier struct {
	webhookURL string
	channel    string
	client     *http.Client
	logger     *slog.Logger
	bucket     *slackRateBucket
}

// NewSlackNotifier creates a SlackNotifier.
// rateLimit is the max messages per minute allowed on this channel.
func NewSlackNotifier(webhookURL, channel string, rateLimit int, logger *slog.Logger) *SlackNotifier {
	if rateLimit <= 0 {
		rateLimit = 5
	}
	return &SlackNotifier{
		webhookURL: webhookURL,
		channel:    channel,
		client:     &http.Client{Timeout: 10 * time.Second},
		logger:     logger,
		bucket: &slackRateBucket{
			limit:     rateLimit,
			windowEnd: time.Now().Add(time.Minute),
		},
	}
}

func (s *SlackNotifier) BackendName() string { return "slack" }

// Send delivers an alert notification to Slack, respecting the per-channel rate limit.
func (s *SlackNotifier) Send(ctx context.Context, alert *Alert) error {
	if !s.bucket.allow() {
		s.logger.Warn("slack rate limit reached, dropping notification",
			"channel", s.channel, "rule", alert.RuleName)
		return fmt.Errorf("slack rate limit exceeded for channel %q", s.channel)
	}

	payload := s.buildBlocks(alert)
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal slack payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.webhookURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("slack HTTP: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("slack returned HTTP %d", resp.StatusCode)
	}
	return nil
}

// buildBlocks constructs a Slack Block Kit payload with rich context.
func (s *SlackNotifier) buildBlocks(alert *Alert) map[string]any {
	emoji := severityEmoji(alert.Severity, alert.State)
	color := severityColor(alert.Severity, alert.State)

	header := fmt.Sprintf("%s [%s] %s", emoji, string(alert.State), alert.RuleName)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("*%s*\n", alert.Message))
	if alert.Detail != "" {
		sb.WriteString(fmt.Sprintf("```%s```\n", alert.Detail))
	}
	sb.WriteString(fmt.Sprintf("*Severity:* %s  |  *Value:* `%.4f`  |  *Threshold:* `%.4f`\n",
		alert.Severity, alert.Value, alert.Threshold))
	sb.WriteString(fmt.Sprintf("*Fired:* %s", alert.FiredAt.Format(time.RFC3339)))
	if alert.ResolvedAt != nil {
		sb.WriteString(fmt.Sprintf("  |  *Resolved:* %s  _(duration: %s)_",
			alert.ResolvedAt.Format(time.RFC3339), alert.Duration))
	}

	return map[string]any{
		"attachments": []map[string]any{
			{
				"color": color,
				"blocks": []map[string]any{
					{
						"type": "header",
						"text": map[string]any{"type": "plain_text", "text": header},
					},
					{
						"type": "section",
						"text": map[string]any{"type": "mrkdwn", "text": sb.String()},
					},
					{
						"type": "context",
						"elements": []map[string]any{
							{"type": "mrkdwn", "text": fmt.Sprintf("ID: `%s`", alert.ID)},
						},
					},
				},
			},
		},
	}
}

func severityEmoji(sev Severity, state AlertState) string {
	if state == AlertResolved {
		return "✅"
	}
	switch sev {
	case SeverityEmergency:
		return "🚨"
	case SeverityCritical:
		return "🔴"
	case SeverityWarning:
		return "🟡"
	default:
		return "🔵"
	}
}

func severityColor(sev Severity, state AlertState) string {
	if state == AlertResolved {
		return "good"
	}
	switch sev {
	case SeverityEmergency, SeverityCritical:
		return "danger"
	case SeverityWarning:
		return "warning"
	default:
		return "#439FE0"
	}
}

// ---------------------------------------------------------------------------
// PagerDutyNotifier
// ---------------------------------------------------------------------------

// pdEvent is the PagerDuty Events API v2 payload.
type pdEvent struct {
	RoutingKey  string         `json:"routing_key"`
	EventAction string         `json:"event_action"` // "trigger" | "resolve"
	DedupKey    string         `json:"dedup_key"`
	Payload     pdEventPayload `json:"payload"`
}

type pdEventPayload struct {
	Summary   string            `json:"summary"`
	Source    string            `json:"source"`
	Severity  string            `json:"severity"` // critical | error | warning | info
	Timestamp string            `json:"timestamp"`
	CustomDetails map[string]any `json:"custom_details,omitempty"`
}

// PagerDutyNotifier creates and resolves PagerDuty incidents via Events API v2.
type PagerDutyNotifier struct {
	routingKey string
	client     *http.Client
	logger     *slog.Logger
	apiURL     string
}

// NewPagerDutyNotifier creates a PagerDutyNotifier.
// routingKey is the integration key from a PagerDuty service.
func NewPagerDutyNotifier(routingKey string, logger *slog.Logger) *PagerDutyNotifier {
	return &PagerDutyNotifier{
		routingKey: routingKey,
		client:     &http.Client{Timeout: 15 * time.Second},
		logger:     logger,
		apiURL:     "https://events.pagerduty.com/v2/enqueue",
	}
}

func (p *PagerDutyNotifier) BackendName() string { return "pagerduty" }

// Send triggers or resolves a PagerDuty incident for the given alert.
func (p *PagerDutyNotifier) Send(ctx context.Context, alert *Alert) error {
	action := "trigger"
	if alert.State == AlertResolved {
		action = "resolve"
	}

	pdSev := "critical"
	switch alert.Severity {
	case SeverityWarning:
		pdSev = "warning"
	case SeverityInfo:
		pdSev = "info"
	}

	event := pdEvent{
		RoutingKey:  p.routingKey,
		EventAction: action,
		DedupKey:    fmt.Sprintf("srfm-%s", alert.RuleName),
		Payload: pdEventPayload{
			Summary:   alert.Message,
			Source:    "srfm-alerter",
			Severity:  pdSev,
			Timestamp: alert.FiredAt.Format(time.RFC3339),
			CustomDetails: map[string]any{
				"alert_id":  alert.ID,
				"rule":      alert.RuleName,
				"value":     alert.Value,
				"threshold": alert.Threshold,
				"detail":    alert.Detail,
			},
		},
	}

	body, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("marshal pagerduty event: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.apiURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("pagerduty HTTP: %w", err)
	}
	defer resp.Body.Close()

	// PagerDuty returns 202 on success.
	if resp.StatusCode != http.StatusAccepted && resp.StatusCode != http.StatusOK {
		return fmt.Errorf("pagerduty returned HTTP %d", resp.StatusCode)
	}

	p.logger.Info("pagerduty event sent",
		"action", action, "rule", alert.RuleName, "dedup_key", event.DedupKey)
	return nil
}

// ---------------------------------------------------------------------------
// SentryNotifier
// ---------------------------------------------------------------------------

// SentryNotifier logs alert events to Sentry as error-level events.
// Uses the Sentry store endpoint directly so we avoid a heavy SDK dependency.
type SentryNotifier struct {
	dsn    string
	client *http.Client
	logger *slog.Logger
	// Parsed DSN fields.
	sentryURL string
	publicKey string
}

// NewSentryNotifier creates a SentryNotifier from a Sentry DSN.
// DSN format: https://PUBLIC_KEY@sentry.io/PROJECT_ID
func NewSentryNotifier(dsn string, logger *slog.Logger) *SentryNotifier {
	n := &SentryNotifier{
		dsn:    dsn,
		client: &http.Client{Timeout: 10 * time.Second},
		logger: logger,
	}
	n.parseDSN(dsn)
	return n
}

func (s *SentryNotifier) BackendName() string { return "sentry" }

func (s *SentryNotifier) parseDSN(dsn string) {
	// Best-effort parse: extract key and build store URL.
	// Format: https://{key}@{host}/{project_id}
	if dsn == "" {
		return
	}
	// Strip scheme.
	rest := strings.TrimPrefix(dsn, "https://")
	rest = strings.TrimPrefix(rest, "http://")
	atIdx := strings.Index(rest, "@")
	if atIdx < 0 {
		return
	}
	s.publicKey = rest[:atIdx]
	hostAndPath := rest[atIdx+1:]
	slashIdx := strings.LastIndex(hostAndPath, "/")
	if slashIdx < 0 {
		return
	}
	host := hostAndPath[:slashIdx]
	projectID := hostAndPath[slashIdx+1:]
	s.sentryURL = fmt.Sprintf("https://%s/api/%s/store/", host, projectID)
}

// Send posts an alert as a Sentry event.
func (s *SentryNotifier) Send(ctx context.Context, alert *Alert) error {
	if s.sentryURL == "" || s.publicKey == "" {
		s.logger.Warn("sentry DSN not configured, skipping", "rule", alert.RuleName)
		return nil
	}

	payload := map[string]any{
		"event_id":  strings.ReplaceAll(alert.ID, "-", ""),
		"timestamp": alert.FiredAt.Format(time.RFC3339),
		"level":     "error",
		"logger":    "srfm-alerter",
		"message":   alert.Message,
		"extra": map[string]any{
			"rule":      alert.RuleName,
			"severity":  string(alert.Severity),
			"state":     string(alert.State),
			"value":     alert.Value,
			"threshold": alert.Threshold,
			"detail":    alert.Detail,
		},
		"tags": map[string]string{
			"service":  "srfm-alerter",
			"severity": string(alert.Severity),
			"rule":     alert.RuleName,
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal sentry event: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.sentryURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Sentry-Auth",
		fmt.Sprintf("Sentry sentry_version=7, sentry_key=%s, sentry_client=srfm-alerter/1.0",
			s.publicKey))

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("sentry HTTP: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("sentry returned HTTP %d", resp.StatusCode)
	}
	return nil
}
