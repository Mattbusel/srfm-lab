package scheduler

import (
	"math"
	"time"

	"go.uber.org/zap"
)

const (
	// maxRetries is the maximum number of retry attempts before an experiment
	// is permanently marked as failed.
	maxRetries = 3
	// baseBackoff is the initial backoff duration.
	baseBackoff = 1 * time.Minute
	// maxBackoff is the ceiling on the exponential backoff.
	maxBackoff = 1 * time.Hour
)

// RetryPolicy encapsulates retry logic for failed experiments.
// It uses truncated exponential backoff: delay = min(base * 2^attempt, maxBackoff).
type RetryPolicy struct {
	MaxRetries  int
	BaseBackoff time.Duration
	MaxBackoff  time.Duration
	log         *zap.Logger
}

// NewRetryPolicy constructs a RetryPolicy with the package-level constants.
func NewRetryPolicy(log *zap.Logger) *RetryPolicy {
	return &RetryPolicy{
		MaxRetries:  maxRetries,
		BaseBackoff: baseBackoff,
		MaxBackoff:  maxBackoff,
		log:         log,
	}
}

// ShouldRetry returns true if the experiment should be retried based on the
// current retry count.
func (p *RetryPolicy) ShouldRetry(retryCount int) bool {
	return retryCount < p.MaxRetries
}

// NextDelay computes the backoff duration for the next retry attempt.
// attempt is the zero-based retry index (0 = first retry).
func (p *RetryPolicy) NextDelay(attempt int) time.Duration {
	if attempt < 0 {
		attempt = 0
	}
	// delay = base * 2^attempt
	exp := math.Pow(2, float64(attempt))
	delay := time.Duration(float64(p.BaseBackoff) * exp)
	if delay > p.MaxBackoff {
		delay = p.MaxBackoff
	}
	return delay
}

// RetryAfter returns the absolute time after which the next retry should fire.
func (p *RetryPolicy) RetryAfter(retryCount int) time.Time {
	return time.Now().Add(p.NextDelay(retryCount))
}

// RetryDecision holds the outcome of a retry evaluation.
type RetryDecision struct {
	// ShouldRetry is true when another attempt should be scheduled.
	ShouldRetry bool
	// Delay is how long to wait before the next attempt.
	Delay time.Duration
	// RetryAt is the absolute time for the next attempt.
	RetryAt time.Time
}

// Evaluate decides whether and when to retry an experiment.
func (p *RetryPolicy) Evaluate(experimentID string, retryCount int, errMsg string) RetryDecision {
	if !p.ShouldRetry(retryCount) {
		p.log.Warn("experiment exhausted retries",
			zap.String("experiment_id", experimentID),
			zap.Int("retry_count", retryCount),
			zap.Int("max_retries", p.MaxRetries),
			zap.String("last_error", errMsg),
		)
		return RetryDecision{ShouldRetry: false}
	}

	delay := p.NextDelay(retryCount)
	retryAt := time.Now().Add(delay)

	p.log.Info("experiment scheduled for retry",
		zap.String("experiment_id", experimentID),
		zap.Int("attempt", retryCount+1),
		zap.Int("max_retries", p.MaxRetries),
		zap.Duration("delay", delay),
		zap.Time("retry_at", retryAt),
		zap.String("error", errMsg),
	)

	return RetryDecision{
		ShouldRetry: true,
		Delay:       delay,
		RetryAt:     retryAt,
	}
}

// RetryItem wraps an ExperimentItem with retry metadata so the scheduler can
// sleep-and-requeue it at the right time.
type RetryItem struct {
	Item    *ExperimentItem
	RetryAt time.Time
}

// RetryQueue is a simple in-memory store for experiments waiting to be
// re-queued after a backoff delay. A background goroutine drains it.
type RetryQueue struct {
	ch chan RetryItem
}

// NewRetryQueue constructs a RetryQueue with a buffer for up to 256 items.
func NewRetryQueue() *RetryQueue {
	return &RetryQueue{ch: make(chan RetryItem, 256)}
}

// Enqueue adds item to the retry queue. It is non-blocking; items are dropped
// if the buffer is full (this is logged by the scheduler).
func (rq *RetryQueue) Enqueue(item RetryItem) bool {
	select {
	case rq.ch <- item:
		return true
	default:
		return false
	}
}

// Drain calls fn for each item once its RetryAt time has elapsed. It blocks
// until the provided done channel is closed.
func (rq *RetryQueue) Drain(fn func(*ExperimentItem), done <-chan struct{}) {
	for {
		select {
		case <-done:
			return
		case ri := <-rq.ch:
			// Sleep until it is time to retry, then hand it back to the scheduler.
			wait := time.Until(ri.RetryAt)
			if wait > 0 {
				timer := time.NewTimer(wait)
				select {
				case <-timer.C:
				case <-done:
					timer.Stop()
					return
				}
			}
			fn(ri.Item)
		}
	}
}
