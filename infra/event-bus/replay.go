// replay.go — Event replay from Redis Streams by time range or offset.
package eventbus

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// Replay configuration
// ─────────────────────────────────────────────────────────────────────────────

// ReplayConfig configures an event replay operation.
type ReplayConfig struct {
	// Topics to replay. Required.
	Topics []Topic

	// StartTime is the earliest event timestamp to include.
	// If zero, replays from the beginning of the stream.
	StartTime time.Time

	// EndTime is the latest event timestamp to include.
	// If zero, replays up to the latest entry.
	EndTime time.Time

	// StartOffset is the Redis Stream entry ID to start from.
	// Overrides StartTime when set. Use "0" to replay from the beginning.
	StartOffset string

	// EndOffset is the Redis Stream entry ID to stop at.
	// Overrides EndTime when set.
	EndOffset string

	// BatchSize is the number of entries to fetch per XRANGE call.
	BatchSize int64

	// MaxEvents caps the total number of events replayed. 0 = no limit.
	MaxEvents int64

	// IncludePartitions is the set of partitions to replay.
	// If empty, all partitions are included.
	IncludePartitions []int

	// FilterTypes filters events by event type (empty = all types).
	FilterTypes []string

	// SpeedMultiplier replays at N× real time (0 = as fast as possible).
	SpeedMultiplier float64
}

// ReplayResult summarises a completed replay.
type ReplayResult struct {
	EventsReplayed int64
	EventsFiltered int64
	Duration       time.Duration
	LastOffset     string
}

// ─────────────────────────────────────────────────────────────────────────────
// Replayer
// ─────────────────────────────────────────────────────────────────────────────

// Replayer reads events from Redis Streams and re-delivers them to a handler.
type Replayer struct {
	rdb *redis.Client
	ser Serializer
	log *zap.Logger
}

// NewReplayer creates a Replayer.
func NewReplayer(rdb *redis.Client, ser Serializer, log *zap.Logger) *Replayer {
	return &Replayer{rdb: rdb, ser: ser, log: log}
}

// Replay reads events matching ReplayConfig and delivers them to handler.
// It blocks until replay is complete or ctx is cancelled.
func (r *Replayer) Replay(ctx context.Context, cfg ReplayConfig, handler EventHandler) (*ReplayResult, error) {
	if len(cfg.Topics) == 0 {
		return nil, fmt.Errorf("at least one topic is required")
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 500
	}

	// Build start/end IDs.
	start := cfg.StartOffset
	if start == "" {
		start = timeToStreamID(cfg.StartTime)
	}
	end := cfg.EndOffset
	if end == "" {
		end = timeToEndStreamID(cfg.EndTime)
	}

	result := &ReplayResult{}
	replayStart := time.Now()

	// Filter set.
	typeFilter := make(map[string]bool)
	for _, t := range cfg.FilterTypes {
		typeFilter[t] = true
	}

	for _, topic := range cfg.Topics {
		if err := ctx.Err(); err != nil {
			break
		}

		meta := GetTopicMeta(topic)
		partitions := []int{0}
		if meta != nil && meta.Partitions > 1 {
			if len(cfg.IncludePartitions) > 0 {
				partitions = cfg.IncludePartitions
			} else {
				partitions = make([]int, meta.Partitions)
				for i := range partitions {
					partitions[i] = i
				}
			}
		}

		for _, partition := range partitions {
			streamKey := StreamKey(topic, partition)
			lastID, count, filtered, err := r.replayStream(ctx, streamKey, start, end, cfg, typeFilter, handler)
			if err != nil {
				r.log.Warn("replay stream error", zap.String("stream", streamKey), zap.Error(err))
			}
			result.EventsReplayed += count
			result.EventsFiltered += filtered
			result.LastOffset = lastID

			if cfg.MaxEvents > 0 && result.EventsReplayed >= cfg.MaxEvents {
				goto done
			}
		}
	}

done:
	result.Duration = time.Since(replayStart)
	return result, nil
}

// replayStream replays events from a single stream key.
func (r *Replayer) replayStream(
	ctx context.Context,
	streamKey, start, end string,
	cfg ReplayConfig,
	typeFilter map[string]bool,
	handler EventHandler,
) (lastID string, replayed, filtered int64, err error) {

	// Track wall-clock time for speed multiplier replay.
	var firstEventTime time.Time
	var replayWallStart time.Time
	if cfg.SpeedMultiplier > 0 {
		replayWallStart = time.Now()
	}

	cursor := start
	for {
		if ctx.Err() != nil {
			return cursor, replayed, filtered, ctx.Err()
		}

		// Fetch a batch of entries.
		entries, err := r.rdb.XRange(ctx, streamKey, cursor, end).Result()
		if err != nil {
			return cursor, replayed, filtered, fmt.Errorf("XRANGE %s: %w", streamKey, err)
		}
		if len(entries) == 0 {
			break
		}

		for _, entry := range entries {
			cursor = entry.ID

			payloadStr, _ := entry.Values["payload"].(string)
			if payloadStr == "" {
				filtered++
				continue
			}

			evt, err := r.ser.Unmarshal([]byte(payloadStr))
			if err != nil {
				r.log.Warn("unmarshal replay entry failed", zap.String("id", entry.ID), zap.Error(err))
				filtered++
				continue
			}

			// Apply type filter.
			if len(typeFilter) > 0 && !typeFilter[evt.Type] {
				filtered++
				continue
			}

			// Speed-controlled replay: sleep to match original timing.
			if cfg.SpeedMultiplier > 0 && !evt.Timestamp.IsZero() {
				if firstEventTime.IsZero() {
					firstEventTime = evt.Timestamp
				}
				eventElapsed := evt.Timestamp.Sub(firstEventTime)
				wallElapsed := time.Since(replayWallStart)
				targetWall := time.Duration(float64(eventElapsed) / cfg.SpeedMultiplier)
				if targetWall > wallElapsed {
					select {
					case <-ctx.Done():
						return cursor, replayed, filtered, ctx.Err()
					case <-time.After(targetWall - wallElapsed):
					}
				}
			}

			if err := handler(ctx, evt); err != nil {
				r.log.Warn("replay handler error",
					zap.String("event_id", evt.ID),
					zap.Error(err))
				// Continue replaying; caller handles errors.
			}

			replayed++
			if cfg.MaxEvents > 0 && replayed >= cfg.MaxEvents {
				return cursor, replayed, filtered, nil
			}
		}

		// If we got a full batch, use the last ID+1 as the next cursor.
		if int64(len(entries)) < cfg.BatchSize {
			break // end of stream
		}
		// Advance past the last ID by appending "+1" at the millisecond.
		cursor = advanceStreamID(cursor)
	}

	return cursor, replayed, filtered, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// DLQ Replayer
// ─────────────────────────────────────────────────────────────────────────────

// DLQEntry represents a message in the dead-letter queue.
type DLQEntry struct {
	EventID   string
	StreamKey string
	MsgID     string
	Retries   int64
	Payload   []byte
	FailedAt  time.Time
}

// DLQReplayer reads from dead-letter queues and allows manual reprocessing.
type DLQReplayer struct {
	rdb *redis.Client
	ser Serializer
	log *zap.Logger
}

// NewDLQReplayer creates a DLQReplayer.
func NewDLQReplayer(rdb *redis.Client, ser Serializer, log *zap.Logger) *DLQReplayer {
	return &DLQReplayer{rdb: rdb, ser: ser, log: log}
}

// ListDLQ returns a batch of entries from the dead-letter queue for a topic.
func (r *DLQReplayer) ListDLQ(ctx context.Context, topic Topic, count int64) ([]*DLQEntry, error) {
	dlqKey := DLQKey(topic)
	entries, err := r.rdb.XRange(ctx, dlqKey, "-", "+").Result()
	if err != nil {
		return nil, fmt.Errorf("XRANGE %s: %w", dlqKey, err)
	}
	if count > 0 && int64(len(entries)) > count {
		entries = entries[:count]
	}

	var result []*DLQEntry
	for _, entry := range entries {
		e := &DLQEntry{}
		e.EventID, _ = entry.Values["event_id"].(string)
		e.StreamKey, _ = entry.Values["stream_key"].(string)
		e.MsgID, _ = entry.Values["msg_id"].(string)
		if retriesStr, ok := entry.Values["retries"].(string); ok {
			e.Retries, _ = strconv.ParseInt(retriesStr, 10, 64)
		}
		if payloadStr, ok := entry.Values["payload"].(string); ok {
			e.Payload = []byte(payloadStr)
		}
		if nsStr, ok := entry.Values["failed_at"].(string); ok {
			ns, _ := strconv.ParseInt(nsStr, 10, 64)
			e.FailedAt = time.Unix(0, ns).UTC()
		}
		result = append(result, e)
	}
	return result, nil
}

// ReplayDLQ re-delivers all entries from the dead-letter queue to handler.
// On success, acknowledges and removes the entry from the DLQ.
func (r *DLQReplayer) ReplayDLQ(ctx context.Context, topic Topic, handler EventHandler) (int64, error) {
	entries, err := r.ListDLQ(ctx, topic, 0)
	if err != nil {
		return 0, err
	}

	dlqKey := DLQKey(topic)
	var replayed int64
	var ids []string

	for _, entry := range entries {
		if len(entry.Payload) == 0 {
			continue
		}
		evt, err := r.ser.Unmarshal(entry.Payload)
		if err != nil {
			r.log.Warn("DLQ unmarshal failed", zap.String("event_id", entry.EventID), zap.Error(err))
			continue
		}

		if err := handler(ctx, evt); err != nil {
			r.log.Warn("DLQ handler error, leaving in queue",
				zap.String("event_id", entry.EventID), zap.Error(err))
			continue
		}

		replayed++
		ids = append(ids, entry.MsgID)
	}

	// Remove successfully processed entries from DLQ.
	if len(ids) > 0 {
		if err := r.rdb.XDel(ctx, dlqKey, ids...).Err(); err != nil {
			r.log.Warn("DLQ XDEL failed", zap.Error(err))
		}
	}
	return replayed, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Time → Stream ID conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

// timeToStreamID converts a time.Time to a Redis Stream ID (milliseconds-0).
func timeToStreamID(t time.Time) string {
	if t.IsZero() {
		return "0"
	}
	return fmt.Sprintf("%d-0", t.UnixMilli())
}

// timeToEndStreamID converts an end time to a Stream ID.
// If t is zero, returns "+" (latest).
func timeToEndStreamID(t time.Time) string {
	if t.IsZero() {
		return "+"
	}
	return fmt.Sprintf("%d-9999999999", t.UnixMilli())
}

// advanceStreamID increments a Stream ID to just past the given ID.
func advanceStreamID(id string) string {
	// Parse "ms-seq" format.
	for i := len(id) - 1; i >= 0; i-- {
		if id[i] == '-' {
			ms := id[:i]
			seq, err := strconv.ParseInt(id[i+1:], 10, 64)
			if err != nil {
				break
			}
			return fmt.Sprintf("%s-%d", ms, seq+1)
		}
	}
	// Fallback: treat as ms and add 1.
	ms, err := strconv.ParseInt(id, 10, 64)
	if err != nil {
		return id
	}
	return fmt.Sprintf("%d-0", ms+1)
}

// StreamIDToTime converts a Redis Stream ID to its approximate time.Time.
func StreamIDToTime(id string) time.Time {
	for i := 0; i < len(id); i++ {
		if id[i] == '-' {
			ms, err := strconv.ParseInt(id[:i], 10, 64)
			if err != nil {
				return time.Time{}
			}
			return time.UnixMilli(ms).UTC()
		}
	}
	ms, err := strconv.ParseInt(id, 10, 64)
	if err != nil {
		return time.Time{}
	}
	return time.UnixMilli(ms).UTC()
}
