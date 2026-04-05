package bus

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"go.uber.org/zap"

	"srfm-lab/idea-engine/bus/events"
)

// EventStore manages the SQLite-backed persistent event log. It implements
// the Persister interface consumed by the Router and also provides replay
// capabilities for crash recovery.
type EventStore struct {
	db  *sql.DB
	log *zap.Logger
}

// NewEventStore opens (or creates) the SQLite database at dbPath, enables
// WAL mode for concurrent access, and ensures the event_log table exists.
func NewEventStore(dbPath string, log *zap.Logger) (*EventStore, error) {
	dsn := fmt.Sprintf(
		"file:%s?_journal_mode=WAL&_busy_timeout=5000&_foreign_keys=on&cache=shared",
		dbPath,
	)
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("eventstore: open sqlite3 %q: %w", dbPath, err)
	}

	// SQLite is single-writer; serialise writes but allow many readers via WAL.
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(4)
	db.SetConnMaxLifetime(time.Hour)
	db.SetConnMaxIdleTime(15 * time.Minute)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("eventstore: ping sqlite3 %q: %w", dbPath, err)
	}

	es := &EventStore{db: db, log: log}
	if err := es.migrate(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("eventstore: migrate: %w", err)
	}
	return es, nil
}

// migrate creates the event_log table if it does not already exist.
func (es *EventStore) migrate(ctx context.Context) error {
	const ddl = `
CREATE TABLE IF NOT EXISTS event_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT    NOT NULL UNIQUE,
    topic        TEXT    NOT NULL,
    payload      TEXT    NOT NULL,
    produced_at  TEXT    NOT NULL,
    producer_name TEXT   NOT NULL,
    persisted_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_event_log_topic        ON event_log(topic);
CREATE INDEX IF NOT EXISTS idx_event_log_produced_at  ON event_log(produced_at);
CREATE INDEX IF NOT EXISTS idx_event_log_topic_ts     ON event_log(topic, produced_at);
`
	_, err := es.db.ExecContext(ctx, ddl)
	return err
}

// PersistEvent writes evt to the event_log table. It satisfies the Persister
// interface so that the Router can call it after every Publish.
func (es *EventStore) PersistEvent(evt events.Event) error {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	const q = `
INSERT INTO event_log (event_id, topic, payload, produced_at, producer_name)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(event_id) DO NOTHING
`
	_, err := es.db.ExecContext(ctx, q,
		evt.EventID,
		evt.Topic,
		string(evt.Payload),
		evt.ProducedAt.UTC().Format(time.RFC3339Nano),
		evt.ProducerName,
	)
	if err != nil {
		return fmt.Errorf("eventstore: insert event %q: %w", evt.EventID, err)
	}
	return nil
}

// ReplayTopic returns all events for the given topic that were produced at or
// after since, in ascending chronological order. It is used for crash recovery
// so that services can catch up on missed events after a restart.
func (es *EventStore) ReplayTopic(topic string, since time.Time) ([]events.Event, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const q = `
SELECT event_id, topic, payload, produced_at, producer_name
FROM   event_log
WHERE  topic = ?
  AND  produced_at >= ?
ORDER  BY produced_at ASC
`
	rows, err := es.db.QueryContext(ctx, q, topic, since.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return nil, fmt.Errorf("eventstore: replay query: %w", err)
	}
	defer rows.Close()

	var result []events.Event
	for rows.Next() {
		var (
			e         events.Event
			producedS string
			payloadS  string
		)
		if err := rows.Scan(&e.EventID, &e.Topic, &payloadS, &producedS, &e.ProducerName); err != nil {
			return nil, fmt.Errorf("eventstore: replay scan: %w", err)
		}
		t, err := time.Parse(time.RFC3339Nano, producedS)
		if err != nil {
			// Fall back to RFC3339 without nanoseconds.
			t, err = time.Parse(time.RFC3339, producedS)
			if err != nil {
				return nil, fmt.Errorf("eventstore: parse produced_at %q: %w", producedS, err)
			}
		}
		e.ProducedAt = t.UTC()
		e.Payload = []byte(payloadS)
		result = append(result, e)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("eventstore: replay iterate: %w", err)
	}
	return result, nil
}

// EventCount returns the total number of persisted events for topic.
// Pass an empty string to count all events across all topics.
func (es *EventStore) EventCount(topic string) (int64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var (
		q    string
		args []interface{}
	)
	if topic == "" {
		q = `SELECT COUNT(*) FROM event_log`
	} else {
		q = `SELECT COUNT(*) FROM event_log WHERE topic = ?`
		args = append(args, topic)
	}

	var n int64
	if err := es.db.QueryRowContext(ctx, q, args...).Scan(&n); err != nil {
		return 0, fmt.Errorf("eventstore: count: %w", err)
	}
	return n, nil
}

// Close closes the underlying database connection pool.
func (es *EventStore) Close() error {
	return es.db.Close()
}
