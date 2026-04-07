// Package storage provides storage and caching utilities for market data.
package storage

import (
	"bytes"
	"compress/zlib"
	"database/sql"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"sync"
	"time"

	"srfm/market-data/aggregator"
)

// Bar is a convenience alias for the canonical bar type used across the
// storage package. It mirrors aggregator.BarEvent so callers that import
// only the storage pkg do not need a transitive import.
type Bar = aggregator.BarEvent

// BarCompressor compresses OHLCV bar data for efficient archival storage.
// It uses delta encoding followed by zlib compression.
//
//   Encoding layout (per bar, big-endian int64):
//     [0] timestamp Unix seconds
//     [1] open  ticks  (price * 10^PrecisionDecimals, delta from prev)
//     [2] high  ticks  (delta from prev open-ticks, NOT prev high)
//     [3] low   ticks  (delta from prev bar)
//     [4] close ticks  (delta from prev bar)
//     [5] volume * 100 (rounded integer, delta from prev)
//
// The first bar is always stored as absolute values.
type BarCompressor struct {
	// PrecisionDecimals controls how many decimal places of price are
	// preserved. Use 2 for equities/FX pairs quoted to cents, 8 for crypto.
	PrecisionDecimals int
}

// NewBarCompressor returns a BarCompressor with the given precision.
func NewBarCompressor(precision int) *BarCompressor {
	if precision < 0 {
		precision = 0
	}
	return &BarCompressor{PrecisionDecimals: precision}
}

// tickFactor returns 10^PrecisionDecimals as a float64.
func (bc *BarCompressor) tickFactor() float64 {
	f := 1.0
	for i := 0; i < bc.PrecisionDecimals; i++ {
		f *= 10.0
	}
	return f
}

// priceToTick quantises a price to an integer tick.
func (bc *BarCompressor) priceToTick(price float64) int64 {
	return int64(math.Round(price * bc.tickFactor()))
}

// tickToPrice converts a tick back to a float64 price.
func (bc *BarCompressor) tickToPrice(tick int64) float64 {
	return float64(tick) / bc.tickFactor()
}

// volToInt rounds a volume value to an integer scaled by 100.
func volToInt(v float64) int64 { return int64(math.Round(v * 100)) }

// intToVol converts the scaled integer back to float64 volume.
func intToVol(i int64) float64 { return float64(i) / 100.0 }

// fieldsPerBar is the number of int64 fields encoded per bar.
const fieldsPerBar = 6

// Compress encodes bars using delta encoding and then compresses the
// resulting byte stream with zlib. Returns an error if bars is empty.
func (bc *BarCompressor) Compress(bars []Bar) ([]byte, error) {
	if len(bars) == 0 {
		return nil, fmt.Errorf("bar_compressor: cannot compress empty slice")
	}

	// Allocate int64 array: len(bars) * fieldsPerBar.
	ints := make([]int64, len(bars)*fieldsPerBar)

	// First bar stored absolutely.
	b0 := bars[0]
	ints[0] = b0.Timestamp.UTC().Unix()
	ints[1] = bc.priceToTick(b0.Open)
	ints[2] = bc.priceToTick(b0.High)
	ints[3] = bc.priceToTick(b0.Low)
	ints[4] = bc.priceToTick(b0.Close)
	ints[5] = volToInt(b0.Volume)

	// Subsequent bars stored as deltas from previous.
	for i := 1; i < len(bars); i++ {
		prev := bars[i-1]
		curr := bars[i]
		base := i * fieldsPerBar

		ints[base+0] = curr.Timestamp.UTC().Unix() - prev.Timestamp.UTC().Unix()
		ints[base+1] = bc.priceToTick(curr.Open) - bc.priceToTick(prev.Open)
		ints[base+2] = bc.priceToTick(curr.High) - bc.priceToTick(prev.High)
		ints[base+3] = bc.priceToTick(curr.Low) - bc.priceToTick(prev.Low)
		ints[base+4] = bc.priceToTick(curr.Close) - bc.priceToTick(prev.Close)
		ints[base+5] = volToInt(curr.Volume) - volToInt(prev.Volume)
	}

	// Encode int64 array to bytes (big-endian).
	raw := make([]byte, len(ints)*8)
	for i, v := range ints {
		binary.BigEndian.PutUint64(raw[i*8:], uint64(v))
	}

	// zlib compress.
	var buf bytes.Buffer
	w, err := zlib.NewWriterLevel(&buf, zlib.BestCompression)
	if err != nil {
		return nil, fmt.Errorf("bar_compressor: init zlib writer: %w", err)
	}
	if _, err := w.Write(raw); err != nil {
		return nil, fmt.Errorf("bar_compressor: zlib write: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("bar_compressor: zlib close: %w", err)
	}

	// Prepend a 4-byte header: [precision byte] [bar count 3 bytes big-endian].
	header := []byte{
		byte(bc.PrecisionDecimals),
		byte(len(bars) >> 16),
		byte(len(bars) >> 8),
		byte(len(bars)),
	}
	return append(header, buf.Bytes()...), nil
}

// Decompress reverses the Compress encoding, returning the original bar slice.
// The precision embedded in the header is used -- the receiver's
// PrecisionDecimals field is ignored during decompression.
func (bc *BarCompressor) Decompress(data []byte) ([]Bar, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("bar_compressor: data too short")
	}

	// Parse header.
	precision := int(data[0])
	nBars := int(data[1])<<16 | int(data[2])<<8 | int(data[3])
	if nBars == 0 {
		return nil, fmt.Errorf("bar_compressor: header reports 0 bars")
	}

	dec := &BarCompressor{PrecisionDecimals: precision}

	// Decompress zlib payload.
	r, err := zlib.NewReader(bytes.NewReader(data[4:]))
	if err != nil {
		return nil, fmt.Errorf("bar_compressor: init zlib reader: %w", err)
	}
	defer r.Close()

	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("bar_compressor: zlib read: %w", err)
	}

	expectedBytes := nBars * fieldsPerBar * 8
	if len(raw) != expectedBytes {
		return nil, fmt.Errorf("bar_compressor: decompressed %d bytes, expected %d",
			len(raw), expectedBytes)
	}

	// Decode int64 array.
	ints := make([]int64, nBars*fieldsPerBar)
	for i := range ints {
		ints[i] = int64(binary.BigEndian.Uint64(raw[i*8:]))
	}

	// Reconstruct bars.
	bars := make([]Bar, nBars)

	// First bar is absolute.
	bars[0].Timestamp = time.Unix(ints[0], 0).UTC()
	bars[0].Open = dec.tickToPrice(ints[1])
	bars[0].High = dec.tickToPrice(ints[2])
	bars[0].Low = dec.tickToPrice(ints[3])
	bars[0].Close = dec.tickToPrice(ints[4])
	bars[0].Volume = intToVol(ints[5])

	// Subsequent bars are cumulative sums of deltas.
	prevTs := ints[0]
	prevOpen := ints[1]
	prevHigh := ints[2]
	prevLow := ints[3]
	prevClose := ints[4]
	prevVol := ints[5]

	for i := 1; i < nBars; i++ {
		base := i * fieldsPerBar
		tsAbs := prevTs + ints[base+0]
		openAbs := prevOpen + ints[base+1]
		highAbs := prevHigh + ints[base+2]
		lowAbs := prevLow + ints[base+3]
		closeAbs := prevClose + ints[base+4]
		volAbs := prevVol + ints[base+5]

		bars[i].Timestamp = time.Unix(tsAbs, 0).UTC()
		bars[i].Open = dec.tickToPrice(openAbs)
		bars[i].High = dec.tickToPrice(highAbs)
		bars[i].Low = dec.tickToPrice(lowAbs)
		bars[i].Close = dec.tickToPrice(closeAbs)
		bars[i].Volume = intToVol(volAbs)

		prevTs = tsAbs
		prevOpen = openAbs
		prevHigh = highAbs
		prevLow = lowAbs
		prevClose = closeAbs
		prevVol = volAbs
	}

	return bars, nil
}

// EstimateCompressionRatio estimates the compression ratio without performing
// full compression. The estimate is based on the empirical observation that
// delta-encoded OHLCV data compresses to roughly 15-25% of raw float64 size.
// Returns the ratio as compressed_size / raw_size (lower is better).
func (bc *BarCompressor) EstimateCompressionRatio(bars []Bar) float64 {
	if len(bars) == 0 {
		return 1.0
	}
	// Raw float64: 5 float64 fields + 1 time.Time (8 bytes) = 48 bytes/bar.
	rawBytes := float64(len(bars)) * 48.0
	// Encoded int64 size (before zlib): fieldsPerBar * 8 bytes/bar.
	encodedBytes := float64(len(bars)) * float64(fieldsPerBar) * 8.0
	// Empirical zlib ratio for delta-encoded integer time series: ~0.30.
	zlibFactor := 0.30
	compressedBytes := encodedBytes * zlibFactor
	return compressedBytes / rawBytes
}

// ---------------------------------------------------------------------------
// BarArchiver -- DuckDB-backed archival of compressed bar chunks.
// ---------------------------------------------------------------------------

// archiveRow represents one row in the bar_archives DuckDB table.
type archiveRow struct {
	symbol string
	date   time.Time // truncated to the day
	data   []byte    // compressed blob
	nBars  int
}

// BarArchiver stores and retrieves compressed bar chunks in a DuckDB (or
// SQLite -- same driver interface) database. Each row holds one symbol-day
// worth of bars as a compressed blob.
type BarArchiver struct {
	db         *sql.DB
	compressor *BarCompressor
	mu         sync.Mutex
}

// archiveSchema is the DDL for the bar archives table.
const archiveSchema = `
CREATE TABLE IF NOT EXISTS bar_archives (
    symbol    TEXT    NOT NULL,
    date      INTEGER NOT NULL,  -- Unix timestamp of midnight UTC
    n_bars    INTEGER NOT NULL,
    data      BLOB    NOT NULL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_bar_archives_symbol ON bar_archives (symbol, date);
`

// NewBarArchiver opens (or creates) the archive database at dbPath and
// initialises the schema.
func NewBarArchiver(db *sql.DB, precision int) (*BarArchiver, error) {
	if _, err := db.Exec(archiveSchema); err != nil {
		return nil, fmt.Errorf("bar_archiver: create schema: %w", err)
	}
	return &BarArchiver{
		db:         db,
		compressor: NewBarCompressor(precision),
	}, nil
}

// Archive compresses bars for symbol on date and upserts into the archive
// table. Existing rows for the same (symbol, date) are replaced.
func (a *BarArchiver) Archive(symbol string, bars []Bar, date time.Time) error {
	if len(bars) == 0 {
		return fmt.Errorf("bar_archiver: no bars to archive for %s on %s",
			symbol, date.Format("2006-01-02"))
	}

	compressed, err := a.compressor.Compress(bars)
	if err != nil {
		return fmt.Errorf("bar_archiver: compress: %w", err)
	}

	day := truncateToDay(date)

	a.mu.Lock()
	defer a.mu.Unlock()

	_, err = a.db.Exec(
		`INSERT OR REPLACE INTO bar_archives (symbol, date, n_bars, data) VALUES (?, ?, ?, ?)`,
		symbol, day.Unix(), len(bars), compressed,
	)
	if err != nil {
		return fmt.Errorf("bar_archiver: upsert %s/%s: %w", symbol,
			date.Format("2006-01-02"), err)
	}
	return nil
}

// Load retrieves all bars for symbol within [start, end] (inclusive dates).
// Bars from multiple days are concatenated in chronological order.
func (a *BarArchiver) Load(symbol string, start, end time.Time) ([]Bar, error) {
	startDay := truncateToDay(start).Unix()
	endDay := truncateToDay(end).Unix()

	a.mu.Lock()
	defer a.mu.Unlock()

	rows, err := a.db.Query(
		`SELECT data FROM bar_archives
		 WHERE symbol = ? AND date >= ? AND date <= ?
		 ORDER BY date ASC`,
		symbol, startDay, endDay,
	)
	if err != nil {
		return nil, fmt.Errorf("bar_archiver: query %s: %w", symbol, err)
	}
	defer rows.Close()

	var all []Bar
	for rows.Next() {
		var blob []byte
		if err := rows.Scan(&blob); err != nil {
			return nil, fmt.Errorf("bar_archiver: scan: %w", err)
		}
		bars, err := a.compressor.Decompress(blob)
		if err != nil {
			return nil, fmt.Errorf("bar_archiver: decompress: %w", err)
		}
		all = append(all, bars...)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("bar_archiver: rows: %w", err)
	}
	return all, nil
}

// ListDates returns all dates with archived data for symbol, in ascending
// chronological order.
func (a *BarArchiver) ListDates(symbol string) ([]time.Time, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	rows, err := a.db.Query(
		`SELECT date FROM bar_archives WHERE symbol = ? ORDER BY date ASC`,
		symbol,
	)
	if err != nil {
		return nil, fmt.Errorf("bar_archiver: list dates: %w", err)
	}
	defer rows.Close()

	var dates []time.Time
	for rows.Next() {
		var unix int64
		if err := rows.Scan(&unix); err != nil {
			return nil, err
		}
		dates = append(dates, time.Unix(unix, 0).UTC())
	}
	return dates, rows.Err()
}

// truncateToDay returns midnight UTC for the given time.
func truncateToDay(t time.Time) time.Time {
	y, m, d := t.UTC().Date()
	return time.Date(y, m, d, 0, 0, 0, 0, time.UTC)
}
