// persistence.go — Redis Streams durable event log with consumer groups.
package eventbus

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// StreamWriter — appends events to Redis Streams
// ─────────────────────────────────────────────────────────────────────────────

// StreamWriter writes events to a Redis Stream with configurable maxlen and TTL.
type StreamWriter struct {
	rdb    *redis.Client
	ser    Serializer
	log    *zap.Logger
	cfg    StreamWriterConfig
}

// StreamWriterConfig configures a StreamWriter.
type StreamWriterConfig struct {
	// MaxLen is the approximate max number of entries to keep per stream.
	MaxLen int64
	// PartitionCount divides a topic stream into N shards.
	PartitionCount int
	// KeyTTL sets an expiry on the stream key itself (0 = never expire).
	KeyTTL time.Duration
}

// DefaultStreamWriterConfig returns sensible defaults.
func DefaultStreamWriterConfig() StreamWriterConfig {
	return StreamWriterConfig{
		MaxLen:         500_000,
		PartitionCount: 1,
		KeyTTL:         0,
	}
}

// NewStreamWriter creates a StreamWriter.
func NewStreamWriter(rdb *redis.Client, ser Serializer, cfg StreamWriterConfig, log *zap.Logger) *StreamWriter {
	return &StreamWriter{rdb: rdb, ser: ser, cfg: cfg, log: log}
}

// Write appends an event to the stream for its topic.
// Returns the Redis Stream entry ID on success.
func (w *StreamWriter) Write(ctx context.Context, evt *Event) (string, error) {
	data, err := w.ser.Marshal(evt)
	if err != nil {
		return "", fmt.Errorf("serialize: %w", err)
	}

	// Partition selection: hash event ID into partition.
	partition := 0
	if w.cfg.PartitionCount > 1 {
		h := 0
		for _, c := range evt.ID {
			h = (h*31 + int(c)) % w.cfg.PartitionCount
		}
		partition = h
	}

	streamKey := StreamKey(evt.Topic, partition)
	args := &redis.XAddArgs{
		Stream: streamKey,
		Values: map[string]interface{}{
			"id":            evt.ID,
			"type":          evt.Type,
			"source":        evt.Source,
			"topic":         string(evt.Topic),
			"sequence":      evt.Sequence,
			"timestamp_ns":  evt.Timestamp.UnixNano(),
			"schema_version": evt.SchemaVersion,
			"payload":       string(data),
		},
	}
	if w.cfg.MaxLen > 0 {
		args.MaxLen = w.cfg.MaxLen
		args.Approx = true
	}

	entryID, err := w.rdb.XAdd(ctx, args).Result()
	if err != nil {
		return "", fmt.Errorf("XADD to %s: %w", streamKey, err)
	}

	// Set TTL on stream key if configured.
	if w.cfg.KeyTTL > 0 {
		_ = w.rdb.Expire(ctx, streamKey, w.cfg.KeyTTL).Err()
	}

	return entryID, nil
}

// WriteBatch writes multiple events in a Redis pipeline for efficiency.
func (w *StreamWriter) WriteBatch(ctx context.Context, events []*Event) error {
	pipe := w.rdb.Pipeline()
	for _, evt := range events {
		data, err := w.ser.Marshal(evt)
		if err != nil {
			w.log.Warn("serialize failed, skipping event", zap.String("id", evt.ID), zap.Error(err))
			continue
		}
		streamKey := StreamKey(evt.Topic, 0)
		pipe.XAdd(ctx, &redis.XAddArgs{
			Stream: streamKey,
			MaxLen: w.cfg.MaxLen,
			Approx: true,
			Values: map[string]interface{}{
				"id":           evt.ID,
				"type":         evt.Type,
				"source":       evt.Source,
				"topic":        string(evt.Topic),
				"sequence":     evt.Sequence,
				"timestamp_ns": evt.Timestamp.UnixNano(),
				"payload":      string(data),
			},
		})
	}
	_, err := pipe.Exec(ctx)
	return err
}

// ─────────────────────────────────────────────────────────────────────────────
// ConsumerGroup — reads from Redis Streams with at-least-once semantics
// ─────────────────────────────────────────────────────────────────────────────

// ConsumerGroupConfig configures a ConsumerGroup reader.
type ConsumerGroupConfig struct {
	// GroupName is the Redis consumer group name.
	GroupName string
	// ConsumerName uniquely identifies this consumer within the group.
	ConsumerName string
	// BatchSize is how many messages to fetch per XREADGROUP call.
	BatchSize int64
	// BlockDuration is how long to block waiting for new messages.
	BlockDuration time.Duration
	// MaxPendingAge is the max age of a pending message before it is claimed.
	MaxPendingAge time.Duration
	// MaxDeliveries is the max number of times a message is delivered before DLQ.
	MaxDeliveries int64
}

// DefaultConsumerGroupConfig returns sensible defaults.
func DefaultConsumerGroupConfig(groupName, consumerName string) ConsumerGroupConfig {
	return ConsumerGroupConfig{
		GroupName:     groupName,
		ConsumerName:  consumerName,
		BatchSize:     100,
		BlockDuration: 2 * time.Second,
		MaxPendingAge: 30 * time.Second,
		MaxDeliveries: 3,
	}
}

// ConsumerGroup wraps Redis consumer group operations.
type ConsumerGroup struct {
	rdb     *redis.Client
	ser     Serializer
	cfg     ConsumerGroupConfig
	log     *zap.Logger
	handler EventHandler

	mu     sync.Mutex
	topics []streamTopicEntry

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

type streamTopicEntry struct {
	topic      Topic
	streamKeys []string // one or more shards
}

// NewConsumerGroup creates a ConsumerGroup for the given topics.
// It creates the consumer groups in Redis if they don't already exist.
func NewConsumerGroup(
	ctx context.Context,
	rdb *redis.Client,
	ser Serializer,
	topics []Topic,
	handler EventHandler,
	cfg ConsumerGroupConfig,
	log *zap.Logger,
) (*ConsumerGroup, error) {
	cg := &ConsumerGroup{
		rdb:     rdb,
		ser:     ser,
		cfg:     cfg,
		log:     log,
		handler: handler,
	}
	cg.ctx, cg.cancel = context.WithCancel(context.Background())

	for _, topic := range topics {
		meta := GetTopicMeta(topic)
		partitions := 1
		if meta != nil && meta.Partitions > 1 {
			partitions = meta.Partitions
		}

		var streamKeys []string
		for p := 0; p < partitions; p++ {
			streamKey := StreamKey(topic, p)
			streamKeys = append(streamKeys, streamKey)

			// Create stream + consumer group (ignore BUSYGROUP if already exists).
			err := rdb.XGroupCreateMkStream(ctx, streamKey, cfg.GroupName, "0").Err()
			if err != nil && err.Error() != "BUSYGROUP Consumer Group name already exists" {
				return nil, fmt.Errorf("XGroupCreateMkStream %s: %w", streamKey, err)
			}
		}
		cg.topics = append(cg.topics, streamTopicEntry{topic: topic, streamKeys: streamKeys})
	}
	return cg, nil
}

// Start launches background goroutines to consume messages from Redis Streams.
func (cg *ConsumerGroup) Start() {
	for _, entry := range cg.topics {
		for _, streamKey := range entry.streamKeys {
			cg.wg.Add(1)
			go cg.readLoop(streamKey)
		}
		// Also launch a pending messages reclaimer.
		for _, streamKey := range entry.streamKeys {
			cg.wg.Add(1)
			go cg.pendingReclaimer(streamKey)
		}
	}
}

// Stop gracefully shuts down the consumer group.
func (cg *ConsumerGroup) Stop() {
	cg.cancel()
	cg.wg.Wait()
}

// readLoop continuously reads new messages from a single stream key.
func (cg *ConsumerGroup) readLoop(streamKey string) {
	defer cg.wg.Done()

	for {
		select {
		case <-cg.ctx.Done():
			return
		default:
		}

		messages, err := cg.rdb.XReadGroup(cg.ctx, &redis.XReadGroupArgs{
			Group:    cg.cfg.GroupName,
			Consumer: cg.cfg.ConsumerName,
			Streams:  []string{streamKey, ">"},
			Count:    cg.cfg.BatchSize,
			Block:    cg.cfg.BlockDuration,
		}).Result()

		if err != nil {
			if err == redis.Nil || err == context.Canceled {
				continue
			}
			cg.log.Warn("XREADGROUP error", zap.String("stream", streamKey), zap.Error(err))
			time.Sleep(time.Second)
			continue
		}

		for _, stream := range messages {
			for _, msg := range stream.Messages {
				cg.processMessage(streamKey, &msg)
			}
		}
	}
}

// processMessage deserializes and delivers a single stream message.
func (cg *ConsumerGroup) processMessage(streamKey string, msg *redis.XMessage) {
	payloadStr, _ := msg.Values["payload"].(string)
	if payloadStr == "" {
		// Acknowledge and skip malformed messages.
		_ = cg.rdb.XAck(cg.ctx, streamKey, cg.cfg.GroupName, msg.ID).Err()
		return
	}

	evt, err := cg.ser.Unmarshal([]byte(payloadStr))
	if err != nil {
		cg.log.Warn("stream message unmarshal failed",
			zap.String("stream", streamKey),
			zap.String("msg_id", msg.ID),
			zap.Error(err))
		_ = cg.rdb.XAck(cg.ctx, streamKey, cg.cfg.GroupName, msg.ID).Err()
		return
	}

	if err := cg.handler(cg.ctx, evt); err != nil {
		cg.log.Warn("handler returned error, message not acknowledged",
			zap.String("stream", streamKey),
			zap.String("event_id", evt.ID),
			zap.Error(err))
		// Do not ACK — message will be re-delivered.
		return
	}

	// Acknowledge successful delivery.
	if err := cg.rdb.XAck(cg.ctx, streamKey, cg.cfg.GroupName, msg.ID).Err(); err != nil {
		cg.log.Warn("XACK failed", zap.String("msg_id", msg.ID), zap.Error(err))
	}
}

// pendingReclaimer periodically claims stale pending messages and re-delivers them.
func (cg *ConsumerGroup) pendingReclaimer(streamKey string) {
	defer cg.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-cg.ctx.Done():
			return
		case <-ticker.C:
			cg.reclaimPending(streamKey)
		}
	}
}

// reclaimPending uses XAUTOCLAIM to re-assign stale pending messages.
func (cg *ConsumerGroup) reclaimPending(streamKey string) {
	minIdle := cg.cfg.MaxPendingAge

	res, err := cg.rdb.XAutoClaim(cg.ctx, &redis.XAutoClaimArgs{
		Stream:   streamKey,
		Group:    cg.cfg.GroupName,
		Consumer: cg.cfg.ConsumerName,
		MinIdle:  minIdle,
		Start:    "0-0",
		Count:    cg.cfg.BatchSize,
	}).Result()
	if err != nil {
		if err != redis.Nil {
			cg.log.Warn("XAUTOCLAIM error", zap.String("stream", streamKey), zap.Error(err))
		}
		return
	}

	for _, msg := range res.Messages {
		// Check delivery count — send to DLQ if exceeded.
		pending, err := cg.rdb.XPendingExt(cg.ctx, &redis.XPendingExtArgs{
			Stream: streamKey,
			Group:  cg.cfg.GroupName,
			Start:  msg.ID,
			Stop:   msg.ID,
			Count:  1,
		}).Result()
		if err == nil && len(pending) > 0 && pending[0].RetryCount >= cg.cfg.MaxDeliveries {
			// Too many retries — send to DLQ and acknowledge.
			payloadStr, _ := msg.Values["payload"].(string)
			if payloadStr != "" {
				evt, _ := cg.ser.Unmarshal([]byte(payloadStr))
				if evt != nil {
					dlqKey := DLQKey(evt.Topic)
					_ = cg.rdb.XAdd(cg.ctx, &redis.XAddArgs{
						Stream: dlqKey,
						MaxLen: 10_000,
						Approx: true,
						Values: map[string]interface{}{
							"event_id":    evt.ID,
							"stream_key":  streamKey,
							"msg_id":      msg.ID,
							"retries":     pending[0].RetryCount,
							"payload":     payloadStr,
							"failed_at":   time.Now().UnixNano(),
						},
					}).Err()
				}
			}
			_ = cg.rdb.XAck(cg.ctx, streamKey, cg.cfg.GroupName, msg.ID).Err()
			continue
		}
		cg.processMessage(streamKey, &msg)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream info and management helpers
// ─────────────────────────────────────────────────────────────────────────────

// StreamInfo holds information about a Redis Stream.
type StreamInfo struct {
	StreamKey    string
	Length       int64
	FirstEntryID string
	LastEntryID  string
	Groups       int64
}

// GetStreamInfo fetches metadata about a stream.
func GetStreamInfo(ctx context.Context, rdb *redis.Client, streamKey string) (*StreamInfo, error) {
	info, err := rdb.XInfoStream(ctx, streamKey).Result()
	if err != nil {
		return nil, fmt.Errorf("XINFO STREAM %s: %w", streamKey, err)
	}
	return &StreamInfo{
		StreamKey: streamKey,
		Length:    info.Length,
		Groups:    info.Groups,
	}, nil
}

// TrimStream trims a stream to the given max length.
func TrimStream(ctx context.Context, rdb *redis.Client, streamKey string, maxLen int64) error {
	return rdb.XTrimMaxLen(ctx, streamKey, maxLen).Err()
}

// StreamLag returns the consumer group lag (number of unread messages) for a stream.
func StreamLag(ctx context.Context, rdb *redis.Client, streamKey, groupName string) (int64, error) {
	groups, err := rdb.XInfoGroups(ctx, streamKey).Result()
	if err != nil {
		return 0, err
	}
	for _, g := range groups {
		if g.Name == groupName {
			return g.Lag, nil
		}
	}
	return 0, fmt.Errorf("group %q not found on stream %s", groupName, streamKey)
}

// PendingCount returns the number of pending (unacknowledged) messages for a group.
func PendingCount(ctx context.Context, rdb *redis.Client, streamKey, groupName string) (int64, error) {
	pending, err := rdb.XPending(ctx, streamKey, groupName).Result()
	if err != nil {
		return 0, err
	}
	return pending.Count, nil
}

// parseStreamTimestamp converts a Redis Stream entry timestamp_ns field to time.Time.
func parseStreamTimestamp(values map[string]interface{}) time.Time {
	v, ok := values["timestamp_ns"]
	if !ok {
		return time.Time{}
	}
	var ns int64
	switch x := v.(type) {
	case string:
		ns, _ = strconv.ParseInt(x, 10, 64)
	case int64:
		ns = x
	}
	if ns == 0 {
		return time.Time{}
	}
	return time.Unix(0, ns).UTC()
}
