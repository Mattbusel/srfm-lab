// Package db provides SQLite database access helpers for the research API.
package db

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// SQLiteDB wraps database/sql with helper methods for the research API.
type SQLiteDB struct {
	db   *sql.DB
	path string
}

// NewSQLiteDB opens (or creates) a SQLite database at path and configures
// connection pool settings appropriate for a read-heavy research API.
func NewSQLiteDB(path string) (*SQLiteDB, error) {
	// Use WAL mode and shared cache for concurrent read access.
	dsn := fmt.Sprintf("file:%s?_journal_mode=WAL&_busy_timeout=5000&_foreign_keys=on&cache=shared", path)
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite3 %q: %w", path, err)
	}

	// Pool config: SQLite is single-writer so cap writers; allow many readers
	// via WAL. The research API is read-only against the warehouse.
	db.SetMaxOpenConns(1)       // serialize writes
	db.SetMaxIdleConns(4)       // keep connections warm for reads
	db.SetConnMaxLifetime(time.Hour)
	db.SetConnMaxIdleTime(15 * time.Minute)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping sqlite3 %q: %w", path, err)
	}

	return &SQLiteDB{db: db, path: path}, nil
}

// Close closes the underlying database connection pool.
func (s *SQLiteDB) Close() error {
	return s.db.Close()
}

// Path returns the file path of the SQLite database.
func (s *SQLiteDB) Path() string {
	return s.path
}

// DB returns the raw *sql.DB for callers that need direct access.
func (s *SQLiteDB) DB() *sql.DB {
	return s.db
}

// Row represents a single result row as a map of column name → value.
type Row map[string]any

// QueryRows executes query with args and returns all rows as []Row.
// The caller is responsible for passing a context with an appropriate deadline.
func (s *SQLiteDB) QueryRows(ctx context.Context, query string, args ...any) ([]Row, error) {
	rows, err := s.db.QueryContext(ctx, query, args...)
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
			// Convert []byte (TEXT) to string for cleaner JSON marshalling.
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
func (s *SQLiteDB) QueryRow(ctx context.Context, query string, args ...any) (Row, error) {
	rows, err := s.QueryRows(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}
	return rows[0], nil
}

// QueryRowInto executes query and scans the single result into dest pointers,
// mirroring sql.Row.Scan semantics. Returns sql.ErrNoRows when no row found.
func (s *SQLiteDB) QueryRowInto(ctx context.Context, query string, args []any, dest ...any) error {
	row := s.db.QueryRowContext(ctx, query, args...)
	return row.Scan(dest...)
}

// Execute runs a statement that does not return rows (INSERT, UPDATE, DELETE).
// Returns the number of rows affected.
func (s *SQLiteDB) Execute(ctx context.Context, query string, args ...any) (int64, error) {
	res, err := s.db.ExecContext(ctx, query, args...)
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
func (s *SQLiteDB) WithTx(ctx context.Context, fn func(*sql.Tx) error) error {
	tx, err := s.db.BeginTx(ctx, nil)
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
func (s *SQLiteDB) Ping(ctx context.Context) error {
	return s.db.PingContext(ctx)
}

// Stats returns connection pool statistics for the health endpoint.
func (s *SQLiteDB) Stats() sql.DBStats {
	return s.db.Stats()
}
