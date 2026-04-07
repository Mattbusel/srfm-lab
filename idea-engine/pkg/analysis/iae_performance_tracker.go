package analysis

import (
	"database/sql"
	"fmt"
	"math"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// ---------------------------------------------------------------------------
// CycleRecord
// ---------------------------------------------------------------------------

// CycleRecord stores one IAE cycle's outcome, capturing the live Sharpe ratio
// before and after the parameter update so the tracker can measure whether
// the IAE's adaptations are actually improving live performance.
type CycleRecord struct {
	CycleID        string
	Timestamp      time.Time
	Generation     int
	BestFitness    float64
	LiveSharpePre  float64 // Sharpe ratio in the window before the param update
	LiveSharpePost float64 // Sharpe ratio in the window after the param update
	ParamDelta     float64 // L2 norm of the parameter change
	RollbackFlag   bool    // true if the system rolled back this update
}

// ---------------------------------------------------------------------------
// IAEPerformanceTracker
// ---------------------------------------------------------------------------

// IAEPerformanceTracker records IAE cycle outcomes to measure whether the
// adaptation engine is actually improving live trading performance.
//
// All public methods are safe for concurrent use.
type IAEPerformanceTracker struct {
	mu     sync.RWMutex
	DB     *sql.DB
	Window int // rolling window for rolling-rate metrics
}

// NewIAEPerformanceTracker creates a tracker backed by the given SQLite DB.
// It runs the schema migration automatically.
func NewIAEPerformanceTracker(db *sql.DB, window int) (*IAEPerformanceTracker, error) {
	t := &IAEPerformanceTracker{DB: db, Window: window}
	if err := t.migrate(); err != nil {
		return nil, fmt.Errorf("IAEPerformanceTracker migrate: %w", err)
	}
	return t, nil
}

// migrate ensures the iae_cycles table exists.
func (t *IAEPerformanceTracker) migrate() error {
	_, err := t.DB.Exec(`
		CREATE TABLE IF NOT EXISTS iae_cycles (
			cycle_id        TEXT PRIMARY KEY,
			timestamp       INTEGER NOT NULL,
			generation      INTEGER NOT NULL,
			best_fitness    REAL    NOT NULL,
			sharpe_pre      REAL    NOT NULL,
			sharpe_post     REAL    NOT NULL,
			param_delta     REAL    NOT NULL,
			rollback_flag   INTEGER NOT NULL DEFAULT 0
		)
	`)
	return err
}

// RecordCycle persists one CycleRecord. Returns an error if the insert fails.
func (t *IAEPerformanceTracker) RecordCycle(record CycleRecord) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	rollbackInt := 0
	if record.RollbackFlag {
		rollbackInt = 1
	}
	_, err := t.DB.Exec(`
		INSERT OR REPLACE INTO iae_cycles
			(cycle_id, timestamp, generation, best_fitness, sharpe_pre, sharpe_post, param_delta, rollback_flag)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`,
		record.CycleID,
		record.Timestamp.Unix(),
		record.Generation,
		record.BestFitness,
		record.LiveSharpePre,
		record.LiveSharpePost,
		record.ParamDelta,
		rollbackInt,
	)
	if err != nil {
		return fmt.Errorf("RecordCycle: %w", err)
	}
	return nil
}

// GetRecentCycles returns up to n CycleRecords ordered by timestamp descending.
func (t *IAEPerformanceTracker) GetRecentCycles(n int) []CycleRecord {
	t.mu.RLock()
	defer t.mu.RUnlock()

	rows, err := t.DB.Query(`
		SELECT cycle_id, timestamp, generation, best_fitness, sharpe_pre, sharpe_post, param_delta, rollback_flag
		FROM iae_cycles
		ORDER BY timestamp DESC
		LIMIT ?
	`, n)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var out []CycleRecord
	for rows.Next() {
		var r CycleRecord
		var ts int64
		var rollbackInt int
		if err := rows.Scan(
			&r.CycleID, &ts, &r.Generation, &r.BestFitness,
			&r.LiveSharpePre, &r.LiveSharpePost, &r.ParamDelta, &rollbackInt,
		); err != nil {
			continue
		}
		r.Timestamp = time.Unix(ts, 0)
		r.RollbackFlag = rollbackInt != 0
		out = append(out, r)
	}
	return out
}

// ComputeImprovementRate returns the fraction of cycles (over all recorded
// cycles) where LiveSharpePost > LiveSharpePre. Returns 0.0 if no cycles
// are recorded.
func (t *IAEPerformanceTracker) ComputeImprovementRate() float64 {
	cycles := t.GetRecentCycles(10000) // effectively all
	return improvementRate(cycles)
}

// ComputeAverageLift returns the mean (LiveSharpePost - LiveSharpePre)
// over all recorded cycles. Returns 0.0 if no cycles exist.
func (t *IAEPerformanceTracker) ComputeAverageLift() float64 {
	cycles := t.GetRecentCycles(10000)
	return averageLift(cycles)
}

// IsIAEEffective returns true when the improvement rate exceeds 55% AND the
// average Sharpe lift is positive. Both conditions must hold simultaneously.
func (t *IAEPerformanceTracker) IsIAEEffective() bool {
	cycles := t.GetRecentCycles(10000)
	return improvementRate(cycles) > 0.55 && averageLift(cycles) > 0.0
}

// DetectRegression returns true when the rolling improvement rate over the
// last 10 cycles falls below 30 %. This signals that recent updates are
// more likely to hurt than help and a human review is warranted.
func (t *IAEPerformanceTracker) DetectRegression() bool {
	cycles := t.GetRecentCycles(10)
	return len(cycles) >= 10 && improvementRate(cycles) < 0.30
}

// ---------------------------------------------------------------------------
// IAEReport
// ---------------------------------------------------------------------------

// IAEReport is a full performance summary produced by GenerateReport.
type IAEReport struct {
	TotalCycles       int
	ImprovementRate   float64 // fraction of cycles with post > pre Sharpe
	AverageLift       float64 // mean(post - pre) Sharpe
	MaxLift           float64 // best single-cycle lift
	MinLift           float64 // worst single-cycle lift
	RollbackRate      float64 // fraction of cycles that were rolled back
	AvgParamDelta     float64 // mean L2 norm of parameter changes
	AvgBestFitness    float64 // mean best-fitness across cycles
	IsEffective       bool    // improvement_rate > 0.55 && avg_lift > 0
	RegressionWarning bool    // rolling 10-cycle improvement_rate < 0.30
	ReportTime        time.Time
}

// GenerateReport produces a full IAEReport from all recorded cycles.
func (t *IAEPerformanceTracker) GenerateReport() IAEReport {
	cycles := t.GetRecentCycles(10000)
	n := len(cycles)
	report := IAEReport{
		TotalCycles: n,
		ReportTime:  time.Now(),
	}
	if n == 0 {
		return report
	}

	improved := 0
	rolledBack := 0
	totalLift := 0.0
	totalDelta := 0.0
	totalFitness := 0.0
	maxLift := math.Inf(-1)
	minLift := math.Inf(1)

	for _, c := range cycles {
		lift := c.LiveSharpePost - c.LiveSharpePre
		totalLift += lift
		totalDelta += c.ParamDelta
		totalFitness += c.BestFitness
		if lift > 0 {
			improved++
		}
		if c.RollbackFlag {
			rolledBack++
		}
		if lift > maxLift {
			maxLift = lift
		}
		if lift < minLift {
			minLift = lift
		}
	}

	report.ImprovementRate = float64(improved) / float64(n)
	report.AverageLift = totalLift / float64(n)
	report.MaxLift = maxLift
	report.MinLift = minLift
	report.RollbackRate = float64(rolledBack) / float64(n)
	report.AvgParamDelta = totalDelta / float64(n)
	report.AvgBestFitness = totalFitness / float64(n)
	report.IsEffective = report.ImprovementRate > 0.55 && report.AverageLift > 0.0
	report.RegressionWarning = t.DetectRegression()
	return report
}

// ---------------------------------------------------------------------------
// Rolling window helpers
// ---------------------------------------------------------------------------

// RollingImprovementRate returns the improvement rate over the last n cycles.
func (t *IAEPerformanceTracker) RollingImprovementRate(n int) float64 {
	return improvementRate(t.GetRecentCycles(n))
}

// RollingAverageLift returns the average lift over the last n cycles.
func (t *IAEPerformanceTracker) RollingAverageLift(n int) float64 {
	return averageLift(t.GetRecentCycles(n))
}

// ---------------------------------------------------------------------------
// Pure helpers (no receiver needed)
// ---------------------------------------------------------------------------

func improvementRate(cycles []CycleRecord) float64 {
	if len(cycles) == 0 {
		return 0.0
	}
	improved := 0
	for _, c := range cycles {
		if c.LiveSharpePost > c.LiveSharpePre {
			improved++
		}
	}
	return float64(improved) / float64(len(cycles))
}

func averageLift(cycles []CycleRecord) float64 {
	if len(cycles) == 0 {
		return 0.0
	}
	total := 0.0
	for _, c := range cycles {
		total += c.LiveSharpePost - c.LiveSharpePre
	}
	return total / float64(len(cycles))
}
