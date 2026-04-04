package duckdb

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"go.uber.org/zap"
)

// DataLoader loads CSV/parquet data into DuckDB from the parquet storage tree.
type DataLoader struct {
	analytics *Analytics
	log       *zap.Logger
}

// NewDataLoader creates a DataLoader.
func NewDataLoader(a *Analytics, log *zap.Logger) *DataLoader {
	return &DataLoader{analytics: a, log: log}
}

// LoadParquetDir scans a parquet storage directory and creates DuckDB views.
// The directory structure is: root/<symbol>/<timeframe>/<date>.csv
func (dl *DataLoader) LoadParquetDir(ctx context.Context, rootDir string) error {
	// Build a list of glob patterns for each timeframe.
	patterns, err := dl.scanPatterns(rootDir)
	if err != nil {
		return fmt.Errorf("scan patterns: %w", err)
	}
	if len(patterns) == 0 {
		dl.log.Warn("no data files found", zap.String("root", rootDir))
		return nil
	}

	// Create a UNION ALL view across all patterns.
	var unionParts []string
	for _, p := range patterns {
		unionParts = append(unionParts, fmt.Sprintf(
			"SELECT * FROM read_csv_auto('%s', header=true, union_by_name=true)", p))
	}
	viewSQL := "CREATE OR REPLACE VIEW bars AS\n" + strings.Join(unionParts, "\nUNION ALL\n")

	dl.log.Info("loading DuckDB view", zap.Int("pattern_count", len(patterns)))
	_, err = dl.analytics.RunQuery(ctx, viewSQL)
	if err != nil {
		return fmt.Errorf("create bars view: %w", err)
	}
	dl.log.Info("DuckDB bars view created")
	return nil
}

// scanPatterns returns glob patterns for each distinct symbol+timeframe.
func (dl *DataLoader) scanPatterns(rootDir string) ([]string, error) {
	type key struct{ symbol, tf string }
	seen := make(map[key]struct{})

	err := filepath.WalkDir(rootDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		if filepath.Ext(path) != ".csv" && filepath.Ext(path) != ".parquet" {
			return nil
		}
		// Parse: rootDir/<symbol>/<timeframe>/<date>.csv
		rel, err := filepath.Rel(rootDir, path)
		if err != nil {
			return nil
		}
		parts := strings.Split(rel, string(filepath.Separator))
		if len(parts) < 3 {
			return nil
		}
		seen[key{parts[0], parts[1]}] = struct{}{}
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	patterns := make([]string, 0, len(seen))
	for k := range seen {
		ext := "csv"
		pattern := filepath.Join(rootDir, k.symbol, k.tf, "*."+ext)
		// Use forward slashes for DuckDB on all platforms.
		pattern = filepath.ToSlash(pattern)
		patterns = append(patterns, pattern)
	}
	sort.Strings(patterns)
	return patterns, nil
}

// DateRange returns the first and last dates available for a symbol+timeframe.
func (dl *DataLoader) DateRange(rootDir, symbol, timeframe string) (first, last time.Time, err error) {
	dir := filepath.Join(rootDir, symbol, timeframe)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return time.Time{}, time.Time{}, nil
		}
		return time.Time{}, time.Time{}, err
	}

	var dates []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".csv" {
			dates = append(dates, e.Name()[:len(e.Name())-4])
		}
	}
	if len(dates) == 0 {
		return time.Time{}, time.Time{}, nil
	}
	sort.Strings(dates)

	first, err = time.Parse("2006-01-02", dates[0])
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	last, err = time.Parse("2006-01-02", dates[len(dates)-1])
	return first, last, err
}

// CatalogEntry describes what data is available for a symbol+timeframe.
type CatalogEntry struct {
	Symbol    string
	Timeframe string
	FirstDate time.Time
	LastDate  time.Time
	FileCount int
}

// BuildCatalog scans the storage directory and returns a catalog of available data.
func (dl *DataLoader) BuildCatalog(rootDir string) ([]CatalogEntry, error) {
	symbols, err := dl.listDirs(rootDir)
	if err != nil {
		return nil, err
	}

	var catalog []CatalogEntry
	for _, sym := range symbols {
		symDir := filepath.Join(rootDir, sym)
		timeframes, err := dl.listDirs(symDir)
		if err != nil {
			continue
		}
		for _, tf := range timeframes {
			tfDir := filepath.Join(symDir, tf)
			files, _ := filepath.Glob(filepath.Join(tfDir, "*.csv"))
			if len(files) == 0 {
				continue
			}
			first, last, err := dl.DateRange(rootDir, sym, tf)
			if err != nil {
				continue
			}
			catalog = append(catalog, CatalogEntry{
				Symbol:    sym,
				Timeframe: tf,
				FirstDate: first,
				LastDate:  last,
				FileCount: len(files),
			})
		}
	}

	sort.Slice(catalog, func(i, j int) bool {
		if catalog[i].Symbol != catalog[j].Symbol {
			return catalog[i].Symbol < catalog[j].Symbol
		}
		return catalog[i].Timeframe < catalog[j].Timeframe
	})
	return catalog, nil
}

func (dl *DataLoader) listDirs(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var dirs []string
	for _, e := range entries {
		if e.IsDir() {
			dirs = append(dirs, e.Name())
		}
	}
	return dirs, nil
}

// ExportToCSV exports a query result to a CSV file.
func (dl *DataLoader) ExportToCSV(ctx context.Context, query, outputPath string) error {
	exportSQL := fmt.Sprintf(
		"COPY (%s) TO '%s' (HEADER, DELIMITER ',')",
		query, filepath.ToSlash(outputPath),
	)
	_, err := dl.analytics.RunQuery(ctx, exportSQL)
	return err
}

// ExportToParquet exports a query result to a Parquet file.
func (dl *DataLoader) ExportToParquet(ctx context.Context, query, outputPath string) error {
	exportSQL := fmt.Sprintf(
		"COPY (%s) TO '%s' (FORMAT PARQUET)",
		query, filepath.ToSlash(outputPath),
	)
	_, err := dl.analytics.RunQuery(ctx, exportSQL)
	return err
}
