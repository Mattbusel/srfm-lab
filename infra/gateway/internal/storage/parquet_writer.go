// Package storage handles persisting bars to parquet files on disk.
// Because the Go parquet ecosystem is large and version-sensitive, we implement
// a lightweight columnar CSV-parquet approximation using a well-structured CSV
// with parquet-compatible column names.  Full Apache Parquet binary encoding
// would require a heavyweight dependency; this module provides the same interface
// and can be swapped for a real parquet library without changing callers.
package storage

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/srfm/gateway/internal/feed"
	"go.uber.org/zap"
)

// ParquetWriter writes bars to daily columnar files per symbol + timeframe.
// Files are named: <root>/<symbol>/<timeframe>/<YYYY-MM-DD>.csv
// (The .csv extension is used here as a parquet-compatible stand-in.)
type ParquetWriter struct {
	log     *zap.Logger
	rootDir string

	mu      sync.Mutex
	writers map[fileKey]*dailyWriter
}

type fileKey struct {
	symbol    string
	timeframe string
	date      string // "2006-01-02"
}

// dailyWriter wraps a CSV writer for one symbol+timeframe+date file.
type dailyWriter struct {
	f   *os.File
	csv *csv.Writer
}

// NewParquetWriter creates a ParquetWriter that stores files under rootDir.
func NewParquetWriter(rootDir string, log *zap.Logger) *ParquetWriter {
	return &ParquetWriter{
		rootDir: rootDir,
		writers: make(map[fileKey]*dailyWriter),
		log:     log,
	}
}

// WriteBar appends a bar to the appropriate daily file.
func (pw *ParquetWriter) WriteBar(symbol, timeframe string, b feed.Bar) error {
	date := b.Timestamp.UTC().Format("2006-01-02")
	key := fileKey{symbol, timeframe, date}

	pw.mu.Lock()
	dw, ok := pw.writers[key]
	if !ok {
		var err error
		dw, err = pw.openWriter(key)
		if err != nil {
			pw.mu.Unlock()
			return fmt.Errorf("open writer for %v: %w", key, err)
		}
		pw.writers[key] = dw
	}
	pw.mu.Unlock()

	row := []string{
		b.Timestamp.UTC().Format(time.RFC3339),
		b.Symbol,
		strconv.FormatFloat(b.Open, 'f', 6, 64),
		strconv.FormatFloat(b.High, 'f', 6, 64),
		strconv.FormatFloat(b.Low, 'f', 6, 64),
		strconv.FormatFloat(b.Close, 'f', 6, 64),
		strconv.FormatFloat(b.Volume, 'f', 2, 64),
		b.Source,
	}

	pw.mu.Lock()
	err := dw.csv.Write(row)
	if err == nil {
		dw.csv.Flush()
		err = dw.csv.Error()
	}
	pw.mu.Unlock()

	return err
}

// FlushAll flushes and closes all open file handles.
// Call this at market close or shutdown.
func (pw *ParquetWriter) FlushAll() {
	pw.mu.Lock()
	defer pw.mu.Unlock()

	for key, dw := range pw.writers {
		dw.csv.Flush()
		if err := dw.f.Close(); err != nil {
			pw.log.Warn("close parquet file", zap.String("key", fmt.Sprintf("%v", key)), zap.Error(err))
		}
		delete(pw.writers, key)
	}
}

// FlushDate closes writers for the specified date so they can be picked up
// by downstream readers.
func (pw *ParquetWriter) FlushDate(date string) {
	pw.mu.Lock()
	defer pw.mu.Unlock()

	for key, dw := range pw.writers {
		if key.date == date {
			dw.csv.Flush()
			dw.f.Close()
			delete(pw.writers, key)
		}
	}
}

// ReadBars reads bars from the daily file for symbol + timeframe + date.
func (pw *ParquetWriter) ReadBars(symbol, timeframe, date string) ([]feed.Bar, error) {
	key := fileKey{symbol, timeframe, date}
	path := pw.filePath(key)

	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	rows, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read csv: %w", err)
	}

	var bars []feed.Bar
	for _, row := range rows {
		if len(row) < 8 {
			continue
		}
		// Skip header row.
		if row[0] == "timestamp" {
			continue
		}
		ts, err := time.Parse(time.RFC3339, row[0])
		if err != nil {
			continue
		}
		parseF := func(s string) float64 {
			v, _ := strconv.ParseFloat(s, 64)
			return v
		}
		bars = append(bars, feed.Bar{
			Timestamp: ts,
			Symbol:    row[1],
			Open:      parseF(row[2]),
			High:      parseF(row[3]),
			Low:       parseF(row[4]),
			Close:     parseF(row[5]),
			Volume:    parseF(row[6]),
			Source:    row[7],
		})
	}
	return bars, nil
}

// openWriter creates (or appends to) the daily CSV file for the given key.
func (pw *ParquetWriter) openWriter(key fileKey) (*dailyWriter, error) {
	path := pw.filePath(key)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}

	needHeader := false
	if _, err := os.Stat(path); os.IsNotExist(err) {
		needHeader = true
	}

	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}

	w := csv.NewWriter(f)
	if needHeader {
		_ = w.Write([]string{"timestamp", "symbol", "open", "high", "low", "close", "volume", "source"})
		w.Flush()
	}

	pw.log.Debug("opened parquet writer", zap.String("path", path))
	return &dailyWriter{f: f, csv: w}, nil
}

func (pw *ParquetWriter) filePath(key fileKey) string {
	return filepath.Join(pw.rootDir, key.symbol, key.timeframe, key.date+".csv")
}
