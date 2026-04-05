// Package triggers implements scheduled event triggers for the scheduler.
package triggers

import (
	"context"
	"time"

	"go.uber.org/zap"
)

// MarketCloseTrigger fires daily at 17:00 ET (22:00 UTC) to kick off the
// full ingestion → pattern mining → hypothesis generation cycle.
// On Sundays it also fires a weekly batch processing cycle.
//
// The trigger calls the registered callbacks in goroutines so that slow
// downstream work does not delay the next tick.
type MarketCloseTrigger struct {
	// DailyCallback is invoked every day at the market-close time.
	DailyCallback func(ctx context.Context, isWeekly bool)
	// marketCloseUTC is the hour (UTC) at which to fire (default 22).
	marketCloseUTC int
	log            *zap.Logger
}

// NewMarketCloseTrigger constructs a MarketCloseTrigger.
// dailyCallback is called with isWeekly=true on Sundays and isWeekly=false
// on all other weekdays.
func NewMarketCloseTrigger(dailyCallback func(ctx context.Context, isWeekly bool), log *zap.Logger) *MarketCloseTrigger {
	return &MarketCloseTrigger{
		DailyCallback:  dailyCallback,
		marketCloseUTC: 22,
		log:            log,
	}
}

// Run blocks and fires the callback at the appropriate UTC wall-clock time
// each day. It exits when ctx is cancelled.
func (t *MarketCloseTrigger) Run(ctx context.Context) {
	t.log.Info("market_close_trigger: started", zap.Int("fire_hour_utc", t.marketCloseUTC))

	for {
		next := t.nextFireTime()
		wait := time.Until(next)
		t.log.Debug("market_close_trigger: sleeping until next fire",
			zap.Time("next_fire", next),
			zap.Duration("wait", wait),
		)

		select {
		case <-ctx.Done():
			t.log.Info("market_close_trigger: stopped")
			return
		case <-time.After(wait):
		}

		now := time.Now().UTC()
		isWeekly := now.Weekday() == time.Sunday

		t.log.Info("market_close_trigger: firing",
			zap.Time("ts", now),
			zap.Bool("is_weekly", isWeekly),
		)

		// Run in a goroutine so the trigger loop is not blocked.
		go func(isW bool) {
			cbCtx, cancel := context.WithTimeout(ctx, 4*time.Hour)
			defer cancel()
			t.DailyCallback(cbCtx, isW)
		}(isWeekly)
	}
}

// nextFireTime computes the next UTC timestamp at which the trigger should fire.
// If the fire time for today has already passed, it returns tomorrow's fire time.
func (t *MarketCloseTrigger) nextFireTime() time.Time {
	now := time.Now().UTC()
	candidate := time.Date(now.Year(), now.Month(), now.Day(),
		t.marketCloseUTC, 0, 0, 0, time.UTC)
	if candidate.Before(now) || candidate.Equal(now) {
		// Today's window has passed; schedule for tomorrow.
		candidate = candidate.Add(24 * time.Hour)
	}
	return candidate
}

// FireNow triggers the daily callback immediately, bypassing the schedule.
// Primarily used for manual back-fill runs during development.
func (t *MarketCloseTrigger) FireNow(ctx context.Context) {
	isWeekly := time.Now().UTC().Weekday() == time.Sunday
	t.log.Info("market_close_trigger: manual fire", zap.Bool("is_weekly", isWeekly))
	go func() {
		cbCtx, cancel := context.WithTimeout(ctx, 4*time.Hour)
		defer cancel()
		t.DailyCallback(cbCtx, isWeekly)
	}()
}
