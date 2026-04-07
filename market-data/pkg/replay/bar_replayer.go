// Package replay provides historical bar replay at configurable speeds.
package replay

import (
	"context"
	"database/sql"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// OHLCV mirrors the aggregator type for use within the replay package.
type OHLCV struct {
	Symbol    string
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	Timestamp time.Time
}

// ReplayConfig controls how the replayer loads and emits bars.
type ReplayConfig struct {
	// SpeedMultiplier controls playback rate.
	// 0 = as fast as possible, 1.0 = real-time, 60 = 60x real-time.
	SpeedMultiplier float64

	StartTime time.Time
	EndTime   time.Time
	Symbols   []string

	// DataDir is searched for <symbol>.csv and <symbol>.db files.
	DataDir string
}

// ReplayEvent is emitted for each bar during replay.
type ReplayEvent struct {
	Symbol        string
	Bar           OHLCV
	SimulatedTime time.Time
	IsLast        bool
}

// pendingBar is an internal wrapper used for priority-queue ordering.
type pendingBar struct {
	event      ReplayEvent
	sourceIdx  int
	nextBar    *pendingBar // linked within the same source
}

// barSource holds all bars for one symbol sorted by timestamp.
type barSource struct {
	symbol string
	bars   []OHLCV
	pos    int
}

func (s *barSource) done() bool {
	return s.pos >= len(s.bars)
}

func (s *barSource) peek() *OHLCV {
	if s.done() {
		return nil
	}
	return &s.bars[s.pos]
}

func (s *barSource) pop() OHLCV {
	b := s.bars[s.pos]
	s.pos++
	return b
}

// BarReplayer replays historical bars at configurable speed.
type BarReplayer struct {
	cfg     ReplayConfig
	sources []*barSource

	// progress tracking -- atomically updated
	totalBars   int64
	emittedBars int64

	mu      sync.Mutex
	running bool
}

// NewBarReplayer constructs a BarReplayer. Call Start to begin replay.
func NewBarReplayer(cfg ReplayConfig) *BarReplayer {
	return &BarReplayer{cfg: cfg}
}

// Start loads all bar data, then begins emitting ReplayEvents on the returned channel.
// The channel is closed when replay is complete or ctx is cancelled.
func (r *BarReplayer) Start(ctx context.Context) <-chan ReplayEvent {
	ch := make(chan ReplayEvent, 512)

	r.mu.Lock()
	r.running = true
	r.mu.Unlock()

	go func() {
		defer close(ch)
		defer func() {
			r.mu.Lock()
			r.running = false
			r.mu.Unlock()
		}()

		if err := r.loadSources(); err != nil {
			return
		}

		r.countTotal()
		r.replay(ctx, ch)
	}()

	return ch
}

// Progress returns a value in [0, 1] representing replay completion.
func (r *BarReplayer) Progress() float64 {
	total := atomic.LoadInt64(&r.totalBars)
	emitted := atomic.LoadInt64(&r.emittedBars)
	if total == 0 {
		return 0
	}
	p := float64(emitted) / float64(total)
	return math.Min(1.0, p)
}

// IsRunning reports whether a replay session is active.
func (r *BarReplayer) IsRunning() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.running
}

// loadSources populates r.sources by scanning DataDir for each symbol.
func (r *BarReplayer) loadSources() error {
	r.sources = r.sources[:0]

	for _, sym := range r.cfg.Symbols {
		var bars []OHLCV
		var err error

		// Try SQLite first.
		dbPath := filepath.Join(r.cfg.DataDir, sym+".db")
		if _, statErr := os.Stat(dbPath); statErr == nil {
			bars, err = loadSQLite(dbPath, sym, r.cfg.StartTime, r.cfg.EndTime)
		} else {
			// Fall back to CSV.
			csvPath := filepath.Join(r.cfg.DataDir, sym+".csv")
			bars, err = loadCSV(csvPath, sym, r.cfg.StartTime, r.cfg.EndTime)
		}

		if err != nil {
			// skip missing symbol files gracefully
			continue
		}

		sort.Slice(bars, func(i, j int) bool {
			return bars[i].Timestamp.Before(bars[j].Timestamp)
		})

		r.sources = append(r.sources, &barSource{
			symbol: sym,
			bars:   bars,
		})
	}

	if len(r.sources) == 0 {
		return fmt.Errorf("no bar data found for any symbol in %s", r.cfg.DataDir)
	}
	return nil
}

// countTotal sums bars across all sources for progress tracking.
func (r *BarReplayer) countTotal() {
	var total int64
	for _, src := range r.sources {
		total += int64(len(src.bars))
	}
	atomic.StoreInt64(&r.totalBars, total)
	atomic.StoreInt64(&r.emittedBars, 0)
}

// replay emits bars interleaved by timestamp across all sources.
func (r *BarReplayer) replay(ctx context.Context, ch chan<- ReplayEvent) {
	speedMult := r.cfg.SpeedMultiplier
	realtime := speedMult > 0

	// Track the simulated time of the last emitted event.
	var simPrev time.Time
	var wallPrev time.Time

	for {
		if ctx.Err() != nil {
			return
		}

		// Find source with earliest next bar.
		src := r.earliest()
		if src == nil {
			break
		}

		bar := src.pop()
		atomic.AddInt64(&r.emittedBars, 1)

		isLast := r.allDone()

		ev := ReplayEvent{
			Symbol:        src.symbol,
			Bar:           bar,
			SimulatedTime: bar.Timestamp,
			IsLast:        isLast,
		}

		if realtime && !simPrev.IsZero() {
			// Sleep proportional to simulated time gap / speedMult.
			simDelta := bar.Timestamp.Sub(simPrev)
			wallDelta := time.Duration(float64(simDelta) / speedMult)
			wallTarget := wallPrev.Add(wallDelta)
			now := time.Now()
			if wallTarget.After(now) {
				sleepTimer := time.NewTimer(wallTarget.Sub(now))
				select {
				case <-ctx.Done():
					sleepTimer.Stop()
					return
				case <-sleepTimer.C:
				}
			}
		}

		simPrev = bar.Timestamp
		wallPrev = time.Now()

		select {
		case <-ctx.Done():
			return
		case ch <- ev:
		}

		if isLast {
			break
		}
	}
}

// earliest returns the barSource with the earliest next bar, or nil when all are done.
func (r *BarReplayer) earliest() *barSource {
	var best *barSource
	for _, src := range r.sources {
		if src.done() {
			continue
		}
		if best == nil || src.peek().Timestamp.Before(best.peek().Timestamp) {
			best = src
		}
	}
	return best
}

// allDone returns true when every source is exhausted.
func (r *BarReplayer) allDone() bool {
	for _, src := range r.sources {
		if !src.done() {
			return false
		}
	}
	return true
}

// -- CSV loader --

// loadCSV reads bars from a CSV file with header: timestamp,open,high,low,close,volume
// Timestamps must be RFC3339 or Unix seconds.
func loadCSV(path, symbol string, start, end time.Time) ([]OHLCV, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	rdr := csv.NewReader(f)
	rdr.TrimLeadingSpace = true

	// read header
	hdr, err := rdr.Read()
	if err != nil {
		return nil, fmt.Errorf("csv header: %w", err)
	}
	colIdx := make(map[string]int, len(hdr))
	for i, h := range hdr {
		colIdx[strings.ToLower(strings.TrimSpace(h))] = i
	}

	required := []string{"timestamp", "open", "high", "low", "close", "volume"}
	for _, col := range required {
		if _, ok := colIdx[col]; !ok {
			return nil, fmt.Errorf("csv missing column: %s", col)
		}
	}

	var bars []OHLCV
	for {
		row, err := rdr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		ts, err := parseTimestamp(row[colIdx["timestamp"]])
		if err != nil {
			continue
		}
		if !start.IsZero() && ts.Before(start) {
			continue
		}
		if !end.IsZero() && ts.After(end) {
			continue
		}

		open, _ := strconv.ParseFloat(strings.TrimSpace(row[colIdx["open"]]), 64)
		high, _ := strconv.ParseFloat(strings.TrimSpace(row[colIdx["high"]]), 64)
		low, _ := strconv.ParseFloat(strings.TrimSpace(row[colIdx["low"]]), 64)
		close, _ := strconv.ParseFloat(strings.TrimSpace(row[colIdx["close"]]), 64)
		volume, _ := strconv.ParseFloat(strings.TrimSpace(row[colIdx["volume"]]), 64)

		bars = append(bars, OHLCV{
			Symbol:    symbol,
			Open:      open,
			High:      high,
			Low:       low,
			Close:     close,
			Volume:    volume,
			Timestamp: ts,
		})
	}
	return bars, nil
}

// parseTimestamp handles RFC3339 strings and Unix second integers.
func parseTimestamp(s string) (time.Time, error) {
	s = strings.TrimSpace(s)
	// Try RFC3339
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t, nil
	}
	// Try unix seconds
	if sec, err := strconv.ParseInt(s, 10, 64); err == nil {
		return time.Unix(sec, 0).UTC(), nil
	}
	return time.Time{}, fmt.Errorf("unparseable timestamp: %s", s)
}

// -- SQLite loader --

// loadSQLite reads bars from a SQLite database.
// Expected schema: CREATE TABLE bars (timestamp INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL)
func loadSQLite(path, symbol string, start, end time.Time) ([]OHLCV, error) {
	db, err := sql.Open("sqlite3", path+"?mode=ro")
	if err != nil {
		return nil, err
	}
	defer db.Close()

	var startUnix, endUnix int64
	startUnix = 0
	endUnix = math.MaxInt32

	if !start.IsZero() {
		startUnix = start.Unix()
	}
	if !end.IsZero() {
		endUnix = end.Unix()
	}

	rows, err := db.Query(
		`SELECT timestamp, open, high, low, close, volume FROM bars WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC`,
		startUnix, endUnix,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var bars []OHLCV
	for rows.Next() {
		var ts int64
		var b OHLCV
		if err := rows.Scan(&ts, &b.Open, &b.High, &b.Low, &b.Close, &b.Volume); err != nil {
			continue
		}
		b.Symbol = symbol
		b.Timestamp = time.Unix(ts, 0).UTC()
		bars = append(bars, b)
	}
	return bars, rows.Err()
}
