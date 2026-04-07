// Package normalizer converts raw price bars from multiple venues into a
// consistent internal format (USD-denominated, split-adjusted, fixed precision).
package normalizer

import (
	"database/sql"
	"fmt"
	"math"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// RawBar is the exchange-native bar before any normalization.
type RawBar struct {
	Symbol    string  `json:"symbol"`
	Open      float64 `json:"open"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Close     float64 `json:"close"`
	Volume    float64 `json:"volume"`
	Timestamp int64   `json:"timestamp_ms"` // Unix milliseconds
	Source    string  `json:"source"`
	Currency  string  `json:"currency"` // e.g. "USD", "USDT", "EUR"
}

// NormalizedBar is the canonical internal representation after normalization.
type NormalizedBar struct {
	Symbol    string     `json:"symbol"`
	OHLCV     [5]float64 `json:"ohlcv"` // [open, high, low, close, volume]
	Timestamp int64      `json:"timestamp_ms"`
	AdjFactor float64    `json:"adj_factor"` // cumulative split/dividend factor applied
}

// adjustmentRecord persists a price adjustment factor in SQLite.
type adjustmentRecord struct {
	Symbol        string
	Factor        float64
	EffectiveDate time.Time
}

// currencyRate maps currency codes to their USD conversion rate.
// USDT is treated 1:1 with USD; other currencies would be populated
// at runtime from a market feed.  This table covers common crypto
// quote currencies.
var defaultCurrencyRates = map[string]float64{
	"USD":  1.0,
	"USDT": 1.0, // treated as dollar-pegged
	"USDC": 1.0,
	"EUR":  0.0, // populated from live feed at runtime
	"GBP":  0.0,
}

// PriceNormalizer converts RawBar values from various venues to NormalizedBar.
// Adjustments (splits, dividends) are persisted in SQLite and applied per bar.
type PriceNormalizer struct {
	db            *sql.DB
	currencyRates map[string]float64 // symbol -> USD rate
	precision     int                // decimal places to round prices to
}

// NewPriceNormalizer opens (or creates) the SQLite database at dbPath and
// returns a ready-to-use normalizer with the given decimal precision.
func NewPriceNormalizer(dbPath string, precision int) (*PriceNormalizer, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	n := &PriceNormalizer{
		db:            db,
		currencyRates: make(map[string]float64),
		precision:     precision,
	}
	// Copy defaults so callers can override without mutating package-level state.
	for k, v := range defaultCurrencyRates {
		n.currencyRates[k] = v
	}
	if err := n.migrate(); err != nil {
		db.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return n, nil
}

// Close releases the underlying database connection.
func (n *PriceNormalizer) Close() error {
	return n.db.Close()
}

// SetCurrencyRate registers a USD conversion rate for the given currency code.
// This allows callers to update rates from a live forex feed.
func (n *PriceNormalizer) SetCurrencyRate(currency string, usdRate float64) {
	n.currencyRates[strings.ToUpper(currency)] = usdRate
}

// RegisterAdjustment stores a price adjustment factor effective from effectiveDate.
// All bars with Timestamp >= effectiveDate.UnixMilli() will have the factor applied.
func (n *PriceNormalizer) RegisterAdjustment(symbol string, factor float64, effectiveDate time.Time) error {
	_, err := n.db.Exec(`
		INSERT INTO price_adjustments (symbol, factor, effective_ms)
		VALUES (?, ?, ?)
		ON CONFLICT(symbol, effective_ms) DO UPDATE SET factor = excluded.factor`,
		strings.ToUpper(symbol), factor, effectiveDate.UnixMilli())
	return err
}

// Normalize converts a RawBar from the named source into a NormalizedBar.
// Steps applied in order:
//  1. Currency conversion to USD.
//  2. Cumulative split/adjustment factor lookup.
//  3. Price multiplication by the adj factor.
//  4. Rounding to the configured decimal precision.
func (n *PriceNormalizer) Normalize(raw RawBar, source string) (NormalizedBar, error) {
	sym := strings.ToUpper(raw.Symbol)

	// 1. Currency conversion.
	fxRate, err := n.usdRate(raw.Currency)
	if err != nil {
		return NormalizedBar{}, fmt.Errorf("currency %s: %w", raw.Currency, err)
	}

	// 2. Adjustment factor for this symbol at this bar's timestamp.
	adjFactor, err := n.adjustmentFactor(sym, raw.Timestamp)
	if err != nil {
		return NormalizedBar{}, fmt.Errorf("adj factor for %s: %w", sym, err)
	}

	combined := fxRate * adjFactor

	round := func(v float64) float64 {
		return roundTo(v*combined, n.precision)
	}

	nb := NormalizedBar{
		Symbol: sym,
		OHLCV: [5]float64{
			round(raw.Open),
			round(raw.High),
			round(raw.Low),
			round(raw.Close),
			raw.Volume, // volume is not price-adjusted
		},
		Timestamp: raw.Timestamp,
		AdjFactor: combined,
	}
	return nb, nil
}

// NormalizeBatch normalizes a slice of bars, returning only successful results
// and a slice of per-bar errors (nil where successful).
func (n *PriceNormalizer) NormalizeBatch(raws []RawBar, source string) ([]NormalizedBar, []error) {
	out := make([]NormalizedBar, 0, len(raws))
	errs := make([]error, len(raws))
	for i, r := range raws {
		nb, err := n.Normalize(r, source)
		errs[i] = err
		if err == nil {
			out = append(out, nb)
		}
	}
	return out, errs
}

// -- internal helpers --

func (n *PriceNormalizer) migrate() error {
	const ddl = `
CREATE TABLE IF NOT EXISTS price_adjustments (
    symbol       TEXT NOT NULL,
    factor       REAL NOT NULL,
    effective_ms INTEGER NOT NULL,
    PRIMARY KEY (symbol, effective_ms)
);`
	_, err := n.db.Exec(ddl)
	return err
}

// usdRate returns the USD conversion rate for the given currency code.
func (n *PriceNormalizer) usdRate(currency string) (float64, error) {
	if currency == "" || strings.EqualFold(currency, "USD") {
		return 1.0, nil
	}
	rate, ok := n.currencyRates[strings.ToUpper(currency)]
	if !ok {
		return 0, fmt.Errorf("unknown currency %q", currency)
	}
	if rate == 0 {
		return 0, fmt.Errorf("rate for %q is zero (not populated)", currency)
	}
	return rate, nil
}

// adjustmentFactor looks up the most recent adjustment factor for sym that
// is effective on or before timestampMs.  Returns 1.0 (no adjustment) if none
// is found.
func (n *PriceNormalizer) adjustmentFactor(symbol string, timestampMs int64) (float64, error) {
	var factor float64
	err := n.db.QueryRow(`
		SELECT factor FROM price_adjustments
		WHERE symbol = ? AND effective_ms <= ?
		ORDER BY effective_ms DESC
		LIMIT 1`, symbol, timestampMs).Scan(&factor)
	if err == sql.ErrNoRows {
		return 1.0, nil
	}
	if err != nil {
		return 0, err
	}
	return factor, nil
}

// roundTo rounds v to d decimal places.
func roundTo(v float64, d int) float64 {
	if d < 0 {
		return v
	}
	pow := math.Pow10(d)
	return math.Round(v*pow) / pow
}
