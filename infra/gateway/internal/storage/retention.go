package storage

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"go.uber.org/zap"
)

// RetentionPolicy defines how long to keep data files.
type RetentionPolicy struct {
	// MaxAgeDays is the maximum age in days to keep CSV files.
	// Files older than this are deleted. 0 means keep all.
	MaxAgeDays int
	// MaxFilesPerSeries is the max number of daily files per symbol+timeframe.
	// Oldest files are removed first. 0 means unlimited.
	MaxFilesPerSeries int
	// MinFreeBytes is the minimum free disk space to maintain.
	// If free space drops below this, oldest files are pruned.
	MinFreeBytes int64
}

// DefaultRetentionPolicy returns a sensible default retention policy.
func DefaultRetentionPolicy() RetentionPolicy {
	return RetentionPolicy{
		MaxAgeDays:        365,
		MaxFilesPerSeries: 0,
		MinFreeBytes:      500 * 1024 * 1024, // 500 MB
	}
}

// RetentionManager prunes old data files according to a RetentionPolicy.
type RetentionManager struct {
	rootDir string
	policy  RetentionPolicy
	log     *zap.Logger
}

// NewRetentionManager creates a RetentionManager.
func NewRetentionManager(rootDir string, policy RetentionPolicy, log *zap.Logger) *RetentionManager {
	return &RetentionManager{rootDir: rootDir, policy: policy, log: log}
}

// PruneResult describes the outcome of a pruning run.
type PruneResult struct {
	FilesDeleted  int
	BytesReclaimed int64
	Errors        []string
}

// Prune scans the storage directory and removes files exceeding the policy.
func (rm *RetentionManager) Prune() PruneResult {
	var result PruneResult
	cutoff := time.Now().AddDate(0, 0, -rm.policy.MaxAgeDays)

	type seriesFile struct {
		path string
		date time.Time
		size int64
	}
	bySeriesFiles := make(map[string][]seriesFile)

	// Walk the directory tree.
	err := filepath.WalkDir(rm.rootDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		if filepath.Ext(path) != ".csv" {
			return nil
		}
		rel, _ := filepath.Rel(rm.rootDir, path)
		parts := strings.Split(rel, string(filepath.Separator))
		if len(parts) < 3 {
			return nil
		}
		// parts[0]=symbol, parts[1]=timeframe, parts[2]=date.csv
		seriesKey := parts[0] + "/" + parts[1]
		base := strings.TrimSuffix(parts[2], ".csv")
		t, parseErr := time.Parse("2006-01-02", base)
		if parseErr != nil {
			return nil
		}
		info, statErr := d.Info()
		size := int64(0)
		if statErr == nil {
			size = info.Size()
		}
		bySeriesFiles[seriesKey] = append(bySeriesFiles[seriesKey], seriesFile{path, t, size})
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		result.Errors = append(result.Errors, fmt.Sprintf("walk error: %v", err))
		return result
	}

	for series, files := range bySeriesFiles {
		// Sort oldest first.
		sort.Slice(files, func(i, j int) bool {
			return files[i].date.Before(files[j].date)
		})

		for _, f := range files {
			shouldDelete := false
			reason := ""

			// Age check.
			if rm.policy.MaxAgeDays > 0 && f.date.Before(cutoff) {
				shouldDelete = true
				reason = "exceeded max age"
			}

			// Count check.
			if !shouldDelete && rm.policy.MaxFilesPerSeries > 0 && len(files) > rm.policy.MaxFilesPerSeries {
				shouldDelete = true
				reason = "exceeded max files per series"
				files = files[1:] // remove from count
			}

			if shouldDelete {
				rm.log.Info("pruning data file",
					zap.String("series", series),
					zap.String("path", f.path),
					zap.String("reason", reason),
				)
				if delErr := os.Remove(f.path); delErr != nil {
					result.Errors = append(result.Errors, fmt.Sprintf("delete %s: %v", f.path, delErr))
				} else {
					result.FilesDeleted++
					result.BytesReclaimed += f.size
				}
			}
		}
	}

	rm.log.Info("retention prune complete",
		zap.Int("files_deleted", result.FilesDeleted),
		zap.Int64("bytes_reclaimed", result.BytesReclaimed),
	)
	return result
}

// DiskUsage returns the total bytes used under rootDir by CSV files.
func (rm *RetentionManager) DiskUsage() (int64, error) {
	var total int64
	err := filepath.WalkDir(rm.rootDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		if filepath.Ext(path) != ".csv" {
			return nil
		}
		info, statErr := d.Info()
		if statErr == nil {
			total += info.Size()
		}
		return nil
	})
	return total, err
}

// FileInventory returns a map of series -> file count.
func (rm *RetentionManager) FileInventory() (map[string]int, error) {
	inventory := make(map[string]int)
	err := filepath.WalkDir(rm.rootDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		if filepath.Ext(path) != ".csv" {
			return nil
		}
		rel, _ := filepath.Rel(rm.rootDir, path)
		parts := strings.Split(rel, string(filepath.Separator))
		if len(parts) < 3 {
			return nil
		}
		seriesKey := parts[0] + "/" + parts[1]
		inventory[seriesKey]++
		return nil
	})
	return inventory, err
}

// ScheduledPrune wraps Prune to be called on a schedule via a ticker.
// It logs results and errors. Call it in a goroutine.
func (rm *RetentionManager) ScheduledPrune(interval time.Duration, stop <-chan struct{}) {
	rm.log.Info("retention manager started", zap.Duration("interval", interval))
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			rm.log.Info("retention manager stopped")
			return
		case <-ticker.C:
			result := rm.Prune()
			if len(result.Errors) > 0 {
				for _, e := range result.Errors {
					rm.log.Error("prune error", zap.String("err", e))
				}
			}
		}
	}
}
