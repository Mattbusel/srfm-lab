// Package feed — reconnect.go provides a generic exponential-backoff reconnector.
package feed

import (
	"context"
	"math"
	"math/rand"
	"time"

	"go.uber.org/zap"
)

// ReconnectConfig controls backoff behaviour.
type ReconnectConfig struct {
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	Multiplier     float64
	JitterFraction float64 // 0-1: random jitter applied to backoff
	MaxAttempts    int     // 0 = unlimited
}

// DefaultReconnectConfig returns a sensible reconnect configuration.
func DefaultReconnectConfig() ReconnectConfig {
	return ReconnectConfig{
		InitialBackoff: time.Second,
		MaxBackoff:     2 * time.Minute,
		Multiplier:     2.0,
		JitterFraction: 0.2,
		MaxAttempts:    0,
	}
}

// Reconnector manages exponential backoff reconnect loops.
type Reconnector struct {
	cfg     ReconnectConfig
	log     *zap.Logger
	source  string
	rng     *rand.Rand
}

// NewReconnector creates a Reconnector.
func NewReconnector(cfg ReconnectConfig, source string, log *zap.Logger) *Reconnector {
	return &Reconnector{
		cfg:    cfg,
		log:    log,
		source: source,
		rng:    rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Run calls connectFn in a loop, backing off on each failure.
// It stops when ctx is cancelled, connectFn returns nil (clean stop),
// or MaxAttempts is exceeded.
// reconnectCounter is called after each reconnect (for metrics); may be nil.
func (r *Reconnector) Run(
	ctx context.Context,
	connectFn func(ctx context.Context) error,
	reconnectCounter func(),
) {
	backoff := r.cfg.InitialBackoff
	attempts := 0

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		err := connectFn(ctx)
		if err == nil {
			// Clean disconnect — reset backoff.
			backoff = r.cfg.InitialBackoff
			select {
			case <-ctx.Done():
				return
			default:
				continue
			}
		}

		attempts++
		if r.cfg.MaxAttempts > 0 && attempts >= r.cfg.MaxAttempts {
			r.log.Error("reconnect max attempts reached",
				zap.String("source", r.source),
				zap.Int("attempts", attempts))
			return
		}

		jitter := time.Duration(float64(backoff) * r.cfg.JitterFraction * r.rng.Float64())
		wait := backoff + jitter

		r.log.Warn("feed disconnected, reconnecting",
			zap.String("source", r.source),
			zap.Error(err),
			zap.Duration("wait", wait),
			zap.Int("attempt", attempts))

		if reconnectCounter != nil {
			reconnectCounter()
		}

		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}

		// Advance backoff.
		next := float64(backoff) * r.cfg.Multiplier
		backoff = time.Duration(math.Min(next, float64(r.cfg.MaxBackoff)))
	}
}

// ConnectionState tracks the live state of a feed connection.
type ConnectionState int

const (
	StateDisconnected ConnectionState = iota
	StateConnecting
	StateConnected
	StateError
)

func (s ConnectionState) String() string {
	switch s {
	case StateDisconnected:
		return "disconnected"
	case StateConnecting:
		return "connecting"
	case StateConnected:
		return "connected"
	case StateError:
		return "error"
	default:
		return "unknown"
	}
}

// FeedStatus holds diagnostics for a running feed.
type FeedStatus struct {
	Source        string
	State         ConnectionState
	ConnectedAt   time.Time
	LastBarAt     time.Time
	BarCount      int64
	ErrorCount    int64
	LastError     string
	Reconnects    int64
}
