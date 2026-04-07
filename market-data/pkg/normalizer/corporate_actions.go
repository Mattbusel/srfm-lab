// Package normalizer converts raw price bars from multiple venues into a
// consistent internal format.
// This file handles stock splits, reverse splits, dividends and rights offerings.
package normalizer

import (
	"database/sql"
	"fmt"
	"sort"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// ActionType enumerates the supported corporate action types.
type ActionType string

const (
	ActionSplit         ActionType = "SPLIT"
	ActionReverseSplit  ActionType = "REVERSE_SPLIT"
	ActionDividend      ActionType = "DIVIDEND"
	ActionRightsOffering ActionType = "RIGHTS_OFFERING"
)

// CorporateAction describes a single corporate event that affects historical prices.
type CorporateAction struct {
	Symbol  string     `json:"symbol"`
	Type    ActionType `json:"type"`
	ExDate  time.Time  `json:"ex_date"` // first trading day the adjustment applies
	Factor  float64    `json:"factor"`  // split ratio, dividend per share, etc.
}

// CorporateActionHandler loads pending actions from SQLite and applies them
// to historical bar arrays or to individual real-time bars.
type CorporateActionHandler struct {
	db *sql.DB
}

// NewCorporateActionHandler opens (or creates) the SQLite database at dbPath.
func NewCorporateActionHandler(dbPath string) (*CorporateActionHandler, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	h := &CorporateActionHandler{db: db}
	if err := h.migrate(); err != nil {
		db.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return h, nil
}

// Close releases the database connection.
func (h *CorporateActionHandler) Close() error {
	return h.db.Close()
}

// AddAction inserts a corporate action into the database.
// Duplicate (symbol, type, ex_date) entries are silently replaced.
func (h *CorporateActionHandler) AddAction(a CorporateAction) error {
	_, err := h.db.Exec(`
		INSERT INTO corporate_actions (symbol, action_type, ex_date_ms, factor)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(symbol, action_type, ex_date_ms) DO UPDATE SET factor = excluded.factor`,
		strings.ToUpper(a.Symbol), string(a.Type), a.ExDate.UnixMilli(), a.Factor)
	return err
}

// PendingActions returns all corporate actions whose ex_date is in the future
// (i.e. not yet applied to live prices).
func (h *CorporateActionHandler) PendingActions() ([]CorporateAction, error) {
	nowMs := time.Now().UnixMilli()
	rows, err := h.db.Query(`
		SELECT symbol, action_type, ex_date_ms, factor
		FROM corporate_actions
		WHERE ex_date_ms > ?
		ORDER BY ex_date_ms`, nowMs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return h.scanActions(rows)
}

// ActionsForSymbol returns all corporate actions for the given symbol,
// ordered by ex_date ascending.
func (h *CorporateActionHandler) ActionsForSymbol(symbol string) ([]CorporateAction, error) {
	rows, err := h.db.Query(`
		SELECT symbol, action_type, ex_date_ms, factor
		FROM corporate_actions
		WHERE symbol = ?
		ORDER BY ex_date_ms`, strings.ToUpper(symbol))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return h.scanActions(rows)
}

// AdjustHistory applies all relevant corporate actions to a slice of historical bars.
// Bars are assumed to be in ascending timestamp order.  The function returns a new
// slice with adjusted OHLCV values; the originals are not mutated.
//
// Algorithm: for each action with ex_date D, all bars *before* D are adjusted
// backwards by the action factor (standard backward-adjustment convention).
//
// Supported adjustments:
//   - SPLIT / REVERSE_SPLIT: prices divided by factor, volume multiplied by factor.
//   - DIVIDEND: prices reduced by dividend amount (additive).
//   - RIGHTS_OFFERING: prices reduced by factor.
func (h *CorporateActionHandler) AdjustHistory(bars []NormalizedBar, actions []CorporateAction) []NormalizedBar {
	if len(bars) == 0 || len(actions) == 0 {
		return bars
	}

	// Sort actions by ex_date descending so we can apply them from newest to oldest.
	sorted := make([]CorporateAction, len(actions))
	copy(sorted, actions)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].ExDate.After(sorted[j].ExDate)
	})

	// Copy bars to avoid mutating caller data.
	out := make([]NormalizedBar, len(bars))
	copy(out, bars)

	for _, act := range sorted {
		exMs := act.ExDate.UnixMilli()
		for i := range out {
			if out[i].Timestamp < exMs {
				out[i] = applyActionToBar(out[i], act)
			}
		}
	}
	return out
}

// ApplyRealtime adjusts a single current bar for any corporate actions that are
// effective as of the bar's timestamp.  Use this on live streaming bars.
func (h *CorporateActionHandler) ApplyRealtime(bar NormalizedBar) (NormalizedBar, error) {
	sym := strings.ToUpper(bar.Symbol)
	rows, err := h.db.Query(`
		SELECT symbol, action_type, ex_date_ms, factor
		FROM corporate_actions
		WHERE symbol = ? AND ex_date_ms <= ?
		ORDER BY ex_date_ms`, sym, bar.Timestamp)
	if err != nil {
		return bar, err
	}
	defer rows.Close()

	actions, err := h.scanActions(rows)
	if err != nil {
		return bar, err
	}

	result := bar
	for _, act := range actions {
		result = applyActionToBar(result, act)
	}
	return result, nil
}

// -- internal helpers --

func (h *CorporateActionHandler) migrate() error {
	const ddl = `
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol       TEXT NOT NULL,
    action_type  TEXT NOT NULL,
    ex_date_ms   INTEGER NOT NULL,
    factor       REAL NOT NULL,
    PRIMARY KEY (symbol, action_type, ex_date_ms)
);`
	_, err := h.db.Exec(ddl)
	return err
}

func (h *CorporateActionHandler) scanActions(rows *sql.Rows) ([]CorporateAction, error) {
	var out []CorporateAction
	for rows.Next() {
		var (
			sym       string
			actType   string
			exDateMs  int64
			factor    float64
		)
		if err := rows.Scan(&sym, &actType, &exDateMs, &factor); err != nil {
			return nil, err
		}
		out = append(out, CorporateAction{
			Symbol:  sym,
			Type:    ActionType(actType),
			ExDate:  time.UnixMilli(exDateMs).UTC(),
			Factor:  factor,
		})
	}
	return out, rows.Err()
}

// applyActionToBar applies one corporate action to one bar.
// OHLCV[4] is volume; indices 0-3 are prices.
func applyActionToBar(bar NormalizedBar, act CorporateAction) NormalizedBar {
	result := bar
	switch act.Type {
	case ActionSplit, ActionReverseSplit:
		if act.Factor == 0 {
			return result
		}
		for i := 0; i < 4; i++ {
			result.OHLCV[i] = bar.OHLCV[i] / act.Factor
		}
		result.OHLCV[4] = bar.OHLCV[4] * act.Factor // volume inverse of price
		result.AdjFactor = bar.AdjFactor / act.Factor

	case ActionDividend:
		// Subtract the dividend from pre-ex prices.
		for i := 0; i < 4; i++ {
			result.OHLCV[i] = bar.OHLCV[i] - act.Factor
		}

	case ActionRightsOffering:
		// Rights offerings are similar to dividends for price adjustment purposes.
		if act.Factor == 0 {
			return result
		}
		for i := 0; i < 4; i++ {
			result.OHLCV[i] = bar.OHLCV[i] / act.Factor
		}
	}
	return result
}
