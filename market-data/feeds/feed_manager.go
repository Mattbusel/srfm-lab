package feeds

import (
	"context"
	"log"
	"os"
	"time"

	"srfm/market-data/monitoring"
)

const (
	healthCheckInterval  = 10 * time.Second
	alpacaSilenceTimeout = 30 * time.Second
)

// FeedHealth describes the current state of a feed.
type FeedHealth struct {
	Name         string
	IsConnected  bool
	LastBarTime  time.Time
	BarsReceived int64
	ErrorsCount  int64
	LatencyMs    int64
	Uptime       time.Duration
}

// Feed is the interface both feeds implement.
type Feed interface {
	Start()
	Stop()
	Health() FeedHealth
}

// FeedManager manages Alpaca (primary) and Binance (secondary) feeds,
// handles automatic failover, and exposes aggregated health.
type FeedManager struct {
	alpaca  *AlpacaFeed
	binance *BinanceFeed
	alerter *monitoring.Alerter
	metrics *monitoring.Metrics

	primaryIsBinance bool
}

// NewFeedManager creates a FeedManager.
func NewFeedManager(alpaca *AlpacaFeed, binance *BinanceFeed, alerter *monitoring.Alerter, metrics *monitoring.Metrics) *FeedManager {
	return &FeedManager{
		alpaca:  alpaca,
		binance: binance,
		alerter: alerter,
		metrics: metrics,
	}
}

// Run starts both feeds and the health monitor loop. Blocks until ctx is cancelled.
func (m *FeedManager) Run(ctx context.Context) {
	// Inject Alpaca credentials from environment
	key := os.Getenv("ALPACA_API_KEY")
	secret := os.Getenv("ALPACA_API_SECRET")
	m.alpaca.SetCredentials(key, secret)

	// Start both feeds concurrently
	go m.alpaca.Start()
	go m.binance.Start()

	ticker := time.NewTicker(healthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			m.alpaca.Stop()
			m.binance.Stop()
			return
		case <-ticker.C:
			m.checkHealth()
		}
	}
}

func (m *FeedManager) checkHealth() {
	ah := m.alpaca.Health()
	bh := m.binance.Health()

	// Determine if Alpaca has been silent
	alpacaSilent := false
	if !ah.LastBarTime.IsZero() && time.Since(ah.LastBarTime) > alpacaSilenceTimeout {
		alpacaSilent = true
	} else if ah.LastBarTime.IsZero() && ah.BarsReceived == 0 {
		// Never received a bar; check connection age
		if ah.Uptime > alpacaSilenceTimeout {
			alpacaSilent = true
		}
	}

	// Log health
	log.Printf("[feed-mgr] alpaca: connected=%v lastBar=%v bars=%d errs=%d latency=%dms",
		ah.IsConnected, ah.LastBarTime.Format(time.RFC3339), ah.BarsReceived, ah.ErrorsCount, ah.LatencyMs)
	log.Printf("[feed-mgr] binance: connected=%v lastBar=%v bars=%d errs=%d latency=%dms",
		bh.IsConnected, bh.LastBarTime.Format(time.RFC3339), bh.BarsReceived, bh.ErrorsCount, bh.LatencyMs)

	// Failover logic
	if alpacaSilent && !m.primaryIsBinance {
		m.primaryIsBinance = true
		log.Printf("[feed-mgr] FAILOVER: Alpaca silent for >%v, promoting Binance to primary", alpacaSilenceTimeout)
		m.alerter.FeedFailover("alpaca", "binance")
		m.metrics.RecordFailover("alpaca", "binance")
	} else if !alpacaSilent && m.primaryIsBinance {
		m.primaryIsBinance = false
		log.Printf("[feed-mgr] RECOVERY: Alpaca resumed, demoting Binance back to secondary")
		m.alerter.FeedFailover("binance", "alpaca")
		m.metrics.RecordFailover("binance", "alpaca")
	}

	// Alert on feed disconnect
	if !ah.IsConnected {
		m.alerter.FeedDisconnect("alpaca")
	}
	if !bh.IsConnected {
		m.alerter.FeedDisconnect("binance")
	}
}

// GetHealth returns health snapshots for both feeds.
func (m *FeedManager) GetHealth() (alpaca FeedHealth, binance FeedHealth) {
	return m.alpaca.Health(), m.binance.Health()
}

// PrimaryFeed returns the name of the current primary feed.
func (m *FeedManager) PrimaryFeed() string {
	if m.primaryIsBinance {
		return "binance"
	}
	return "alpaca"
}
