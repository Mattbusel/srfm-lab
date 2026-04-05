package monitoring

import (
	"fmt"
	"log"
	"sync"
	"time"
)

const (
	disconnectAlertThreshold = 60 * time.Second
	barGapThreshold          = 5 * time.Minute
	clientPileupThreshold    = 50
	alertCooldown            = 5 * time.Minute
)

// AlertEvent records an alert occurrence.
type AlertEvent struct {
	Level     string    `json:"level"`
	Category  string    `json:"category"`
	Message   string    `json:"message"`
	Timestamp time.Time `json:"timestamp"`
}

// Alerter monitors system state and fires alerts.
type Alerter struct {
	mu           sync.Mutex
	lastAlerts   map[string]time.Time
	recentAlerts []AlertEvent
	maxHistory   int

	// Channel for feeding alerts to external sinks
	alertCh chan AlertEvent
}

// NewAlerter creates an Alerter.
func NewAlerter() *Alerter {
	return &Alerter{
		lastAlerts: make(map[string]time.Time),
		maxHistory: 200,
		alertCh:    make(chan AlertEvent, 64),
	}
}

// Run starts the alerter's log drain loop. Blocks until ctx is cancelled.
func (a *Alerter) Run(ctx interface{ Done() <-chan struct{} }, provider interface{}) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	// Drain alert channel and log
	go func() {
		for evt := range a.alertCh {
			log.Printf("[ALERT][%s][%s] %s", evt.Level, evt.Category, evt.Message)
		}
	}()

	for {
		select {
		case <-ctx.Done():
			close(a.alertCh)
			return
		case <-ticker.C:
			// Periodic health checks driven by FeedManager calls
		}
	}
}

// FeedDisconnect fires a disconnection alert if cooldown has passed.
func (a *Alerter) FeedDisconnect(feed string) {
	key := "disconnect:" + feed
	a.fire(key, "warning", "feed_disconnect",
		feed+" feed disconnected or not receiving data for >"+disconnectAlertThreshold.String())
}

// FeedFailover fires a failover alert.
func (a *Alerter) FeedFailover(from, to string) {
	key := "failover:" + from + ":" + to
	a.fire(key, "critical", "feed_failover",
		"Feed failover: "+from+" -> "+to)
}

// BarGap fires a bar-gap alert for a symbol.
func (a *Alerter) BarGap(symbol, timeframe string, gap time.Duration) {
	key := "bargap:" + symbol + ":" + timeframe
	a.fire(key, "warning", "bar_gap",
		symbol+"/"+timeframe+" has bar gap of "+gap.Round(time.Second).String())
}

// StorageError fires a storage error alert.
func (a *Alerter) StorageError(err error) {
	a.fire("storage_error", "critical", "storage_error", "Storage error: "+err.Error())
}

// ClientPileup fires when WebSocket client count exceeds threshold.
func (a *Alerter) ClientPileup(count int) {
	key := "client_pileup"
	a.fire(key, "warning", "ws_pileup",
		"WebSocket client count high: "+fmt.Sprintf("%d", count))
}

func (a *Alerter) fire(key, level, category, message string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	last, ok := a.lastAlerts[key]
	if ok && time.Since(last) < alertCooldown {
		return // still in cooldown
	}
	a.lastAlerts[key] = time.Now()

	evt := AlertEvent{
		Level:     level,
		Category:  category,
		Message:   message,
		Timestamp: time.Now().UTC(),
	}

	a.recentAlerts = append(a.recentAlerts, evt)
	if len(a.recentAlerts) > a.maxHistory {
		a.recentAlerts = a.recentAlerts[len(a.recentAlerts)-a.maxHistory:]
	}

	select {
	case a.alertCh <- evt:
	default:
	}
}

// RecentAlerts returns the most recent alert events.
func (a *Alerter) RecentAlerts(n int) []AlertEvent {
	a.mu.Lock()
	defer a.mu.Unlock()
	if n <= 0 || n >= len(a.recentAlerts) {
		out := make([]AlertEvent, len(a.recentAlerts))
		copy(out, a.recentAlerts)
		return out
	}
	out := make([]AlertEvent, n)
	copy(out, a.recentAlerts[len(a.recentAlerts)-n:])
	return out
}
