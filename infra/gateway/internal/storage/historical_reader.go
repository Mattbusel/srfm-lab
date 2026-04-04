package storage

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	"github.com/srfm/gateway/internal/feed"
	"go.uber.org/zap"
)

// HistoricalReader reads historical bar data from the parquet/CSV storage.
type HistoricalReader struct {
	rootDir string
	log     *zap.Logger
}

// NewHistoricalReader creates a HistoricalReader.
func NewHistoricalReader(rootDir string, log *zap.Logger) *HistoricalReader {
	return &HistoricalReader{rootDir: rootDir, log: log}
}

// ReadRange reads all bars for symbol+timeframe in [from, to].
// It scans daily files that overlap the range.
func (hr *HistoricalReader) ReadRange(symbol, timeframe string, from, to time.Time) ([]feed.Bar, error) {
	dir := filepath.Join(hr.rootDir, symbol, timeframe)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("readdir %s: %w", dir, err)
	}

	var allBars []feed.Bar
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".csv" {
			continue
		}
		// Parse date from filename (YYYY-MM-DD.csv).
		datePart := entry.Name()[:len(entry.Name())-4]
		fileDate, err := time.Parse("2006-01-02", datePart)
		if err != nil {
			continue
		}
		// Skip files entirely outside the range.
		fileEnd := fileDate.Add(24 * time.Hour)
		if fileEnd.Before(from) || fileDate.After(to) {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		bars, err := hr.readFile(path, symbol)
		if err != nil {
			hr.log.Warn("read historical file", zap.String("path", path), zap.Error(err))
			continue
		}
		allBars = append(allBars, bars...)
	}

	// Filter to exact range.
	var result []feed.Bar
	for _, b := range allBars {
		if !b.Timestamp.Before(from) && !b.Timestamp.After(to) {
			result = append(result, b)
		}
	}

	// Sort by timestamp.
	sort.Slice(result, func(i, j int) bool {
		return result[i].Timestamp.Before(result[j].Timestamp)
	})

	return result, nil
}

// readFile reads a single CSV bar file.
func (hr *HistoricalReader) readFile(path, symbol string) ([]feed.Bar, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	rows, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("csv read: %w", err)
	}
	if len(rows) == 0 {
		return nil, nil
	}

	// Find column indices from header.
	header := rows[0]
	idx := make(map[string]int)
	for i, h := range header {
		idx[h] = i
	}

	pf := func(s string) float64 {
		v, _ := strconv.ParseFloat(s, 64)
		return v
	}

	var bars []feed.Bar
	for _, row := range rows[1:] {
		if len(row) < len(header) {
			continue
		}
		ts, err := time.Parse(time.RFC3339, row[idx["timestamp"]])
		if err != nil {
			continue
		}
		sym := symbol
		if si, ok := idx["symbol"]; ok && row[si] != "" {
			sym = row[si]
		}
		src := "storage"
		if si, ok := idx["source"]; ok {
			src = row[si]
		}
		bars = append(bars, feed.Bar{
			Timestamp: ts,
			Symbol:    sym,
			Open:      pf(row[idx["open"]]),
			High:      pf(row[idx["high"]]),
			Low:       pf(row[idx["low"]]),
			Close:     pf(row[idx["close"]]),
			Volume:    pf(row[idx["volume"]]),
			Source:    src,
		})
	}
	return bars, nil
}

// ListSymbols returns the list of symbols available in storage.
func (hr *HistoricalReader) ListSymbols() ([]string, error) {
	entries, err := os.ReadDir(hr.rootDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("readdir: %w", err)
	}
	var symbols []string
	for _, e := range entries {
		if e.IsDir() {
			symbols = append(symbols, e.Name())
		}
	}
	sort.Strings(symbols)
	return symbols, nil
}

// ListTimeframes returns the timeframes available for a symbol.
func (hr *HistoricalReader) ListTimeframes(symbol string) ([]string, error) {
	dir := filepath.Join(hr.rootDir, symbol)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("readdir: %w", err)
	}
	var tfs []string
	for _, e := range entries {
		if e.IsDir() {
			tfs = append(tfs, e.Name())
		}
	}
	sort.Strings(tfs)
	return tfs, nil
}

// AvailableDates returns the list of dates with data for symbol+timeframe.
func (hr *HistoricalReader) AvailableDates(symbol, timeframe string) ([]string, error) {
	dir := filepath.Join(hr.rootDir, symbol, timeframe)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("readdir: %w", err)
	}
	var dates []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".csv" {
			date := e.Name()[:len(e.Name())-4]
			if _, err := time.Parse("2006-01-02", date); err == nil {
				dates = append(dates, date)
			}
		}
	}
	sort.Strings(dates)
	return dates, nil
}

// StorageStats returns summary statistics about the storage directory.
func (hr *HistoricalReader) StorageStats() (map[string]interface{}, error) {
	symbols, err := hr.ListSymbols()
	if err != nil {
		return nil, err
	}

	stats := map[string]interface{}{
		"root_dir":     hr.rootDir,
		"symbol_count": len(symbols),
		"symbols":      symbols,
	}

	totalFiles := 0
	totalSize := int64(0)
	for _, sym := range symbols {
		tfs, _ := hr.ListTimeframes(sym)
		for _, tf := range tfs {
			dates, _ := hr.AvailableDates(sym, tf)
			for _, date := range dates {
				path := filepath.Join(hr.rootDir, sym, tf, date+".csv")
				if fi, err := os.Stat(path); err == nil {
					totalFiles++
					totalSize += fi.Size()
				}
			}
		}
	}

	stats["total_files"] = totalFiles
	stats["total_size_mb"] = float64(totalSize) / 1024 / 1024
	return stats, nil
}
