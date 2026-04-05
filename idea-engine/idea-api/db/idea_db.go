// Package db provides SQLite database access helpers for the idea-api service.
// It mirrors the pattern established in infra/research-api/db/sqlite.go.
package db

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// IdeaDB wraps database/sql with helper methods tailored for the idea-api.
type IdeaDB struct {
	db   *sql.DB
	path string
}

// NewIdeaDB opens (or creates) a SQLite database at path, enables WAL mode
// for concurrent access, and configures the connection pool.
func NewIdeaDB(path string) (*IdeaDB, error) {
	dsn := fmt.Sprintf(
		"file:%s?_journal_mode=WAL&_busy_timeout=5000&_foreign_keys=on&cache=shared",
		path,
	)
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite3 %q: %w", path, err)
	}

	// Allow a single writer, multiple readers via WAL.
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(4)
	db.SetConnMaxLifetime(time.Hour)
	db.SetConnMaxIdleTime(15 * time.Minute)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping sqlite3 %q: %w", path, err)
	}

	return &IdeaDB{db: db, path: path}, nil
}

// Close closes the underlying database connection pool.
func (d *IdeaDB) Close() error {
	return d.db.Close()
}

// Path returns the file-system path of the database.
func (d *IdeaDB) Path() string {
	return d.path
}

// DB returns the raw *sql.DB for callers that need direct access.
func (d *IdeaDB) DB() *sql.DB {
	return d.db
}

// Row is a single result row represented as a map of column name → value.
type Row map[string]any

// QueryRows executes query with args and returns all rows as []Row.
func (d *IdeaDB) QueryRows(ctx context.Context, query string, args ...any) ([]Row, error) {
	rows, err := d.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("QueryRows: %w", err)
	}
	defer rows.Close()

	cols, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("QueryRows columns: %w", err)
	}

	var result []Row
	for rows.Next() {
		vals := make([]any, len(cols))
		ptrs := make([]any, len(cols))
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return nil, fmt.Errorf("QueryRows scan: %w", err)
		}
		row := make(Row, len(cols))
		for i, col := range cols {
			if b, ok := vals[i].([]byte); ok {
				row[col] = string(b)
			} else {
				row[col] = vals[i]
			}
		}
		result = append(result, row)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("QueryRows iterate: %w", err)
	}
	return result, nil
}

// QueryRow executes query with args and returns a single Row.
// Returns nil, nil when no row is found.
func (d *IdeaDB) QueryRow(ctx context.Context, query string, args ...any) (Row, error) {
	rows, err := d.QueryRows(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}
	return rows[0], nil
}

// QueryRowInto executes query and scans a single result into dest pointers.
// Returns sql.ErrNoRows when no row is found.
func (d *IdeaDB) QueryRowInto(ctx context.Context, query string, args []any, dest ...any) error {
	row := d.db.QueryRowContext(ctx, query, args...)
	return row.Scan(dest...)
}

// Execute runs a statement that does not return rows (INSERT, UPDATE, DELETE).
// Returns the number of rows affected.
func (d *IdeaDB) Execute(ctx context.Context, query string, args ...any) (int64, error) {
	res, err := d.db.ExecContext(ctx, query, args...)
	if err != nil {
		return 0, fmt.Errorf("Execute: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("Execute rows affected: %w", err)
	}
	return n, nil
}

// WithTx runs fn inside a transaction. If fn returns an error the transaction
// is rolled back; otherwise it is committed.
func (d *IdeaDB) WithTx(ctx context.Context, fn func(*sql.Tx) error) error {
	tx, err := d.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	if err := fn(tx); err != nil {
		_ = tx.Rollback()
		return err
	}
	return tx.Commit()
}

// Ping checks that the database is still reachable.
func (d *IdeaDB) Ping(ctx context.Context) error {
	return d.db.PingContext(ctx)
}

// Stats returns connection pool statistics.
func (d *IdeaDB) Stats() sql.DBStats {
	return d.db.Stats()
}
