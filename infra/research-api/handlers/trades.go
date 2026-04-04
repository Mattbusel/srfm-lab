// Package handlers implements HTTP handlers for the research API.
package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/srfm/research-api/db"
)

// TradeRecord represents a single trade row as returned by the API.
type TradeRecord struct {
	ID          int64   `json:"id"`
	RunID       int64   `json:"run_id"`
	Symbol      string  `json:"symbol"`
	Side        string  `json:"side"`
	Qty         float64 `json:"qty"`
	EntryPrice  float64 `json:"entry_price"`
	ExitPrice   float64 `json:"exit_price,omitempty"`
	EntryTime   string  `json:"entry_time"`
	ExitTime    string  `json:"exit_time,omitempty"`
	PnL         float64 `json:"pnl,omitempty"`
	PnLPct      float64 `json:"pnl_pct,omitempty"`
	Commission  float64 `json:"commission"`
	Source      string  `json:"source"` // "live" or "backtest"
	ExitReason  string  `json:"exit_reason,omitempty"`
	Timeframe   string  `json:"timeframe,omitempty"`
	StrategyTag string  `json:"strategy_tag,omitempty"`
}

// TradeStats aggregates key performance metrics for a filtered trade set.
type TradeStats struct {
	TotalTrades   int     `json:"total_trades"`
	WinCount      int     `json:"win_count"`
	LossCount     int     `json:"loss_count"`
	WinRate       float64 `json:"win_rate"`
	TotalPnL      float64 `json:"total_pnl"`
	AvgPnL        float64 `json:"avg_pnl"`
	AvgWin        float64 `json:"avg_win"`
	AvgLoss       float64 `json:"avg_loss"`
	ProfitFactor  float64 `json:"profit_factor"`
	Sharpe        float64 `json:"sharpe"`
	CAGR          float64 `json:"cagr"`
	MaxDrawdown   float64 `json:"max_drawdown"`
	AvgHoldBars   float64 `json:"avg_hold_bars"`
	TotalCommission float64 `json:"total_commission"`
}

// TradesHandler serves trade data from the warehouse SQLite database.
type TradesHandler struct {
	warehouse *db.SQLiteDB // warehouse.db — contains strategy_runs, trades, etc.
	live      *db.SQLiteDB // live_trades.db — live execution fills
}

// NewTradesHandler creates a TradesHandler with database connections.
func NewTradesHandler(warehouse, live *db.SQLiteDB) *TradesHandler {
	return &TradesHandler{warehouse: warehouse, live: live}
}

// tradeFilter holds parsed query parameters.
type tradeFilter struct {
	source    string    // "live", "backtest", or "" (both)
	from      time.Time
	to        time.Time
	sym       string
	runID     int64
	limit     int
	offset    int
}

// parseTradeFilter extracts and validates query params from the request.
func parseTradeFilter(r *http.Request) (tradeFilter, error) {
	q := r.URL.Query()
	f := tradeFilter{
		source: q.Get("source"),
		sym:    strings.ToUpper(q.Get("sym")),
		limit:  200,
	}

	if f.source != "" && f.source != "live" && f.source != "backtest" {
		return f, fmt.Errorf("source must be 'live' or 'backtest'")
	}

	if fs := q.Get("from"); fs != "" {
		t, err := time.Parse(time.RFC3339, fs)
		if err != nil {
			return f, fmt.Errorf("invalid from: %w", err)
		}
		f.from = t
	}
	if ts := q.Get("to"); ts != "" {
		t, err := time.Parse(time.RFC3339, ts)
		if err != nil {
			return f, fmt.Errorf("invalid to: %w", err)
		}
		f.to = t
	}
	if rs := q.Get("run_id"); rs != "" {
		id, err := strconv.ParseInt(rs, 10, 64)
		if err != nil {
			return f, fmt.Errorf("invalid run_id: %w", err)
		}
		f.runID = id
	}
	if ls := q.Get("limit"); ls != "" {
		n, err := strconv.Atoi(ls)
		if err != nil || n < 1 || n > 5000 {
			return f, fmt.Errorf("limit must be 1-5000")
		}
		f.limit = n
	}
	if os := q.Get("offset"); os != "" {
		n, err := strconv.Atoi(os)
		if err != nil || n < 0 {
			return f, fmt.Errorf("invalid offset")
		}
		f.offset = n
	}
	return f, nil
}

// GetTrades handles GET /api/v1/trades
// Query params: source=live|backtest, from=RFC3339, to=RFC3339, sym=AAPL, run_id=, limit=, offset=
func (h *TradesHandler) GetTrades(w http.ResponseWriter, r *http.Request) {
	f, err := parseTradeFilter(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	trades, err := h.queryTrades(ctx, f)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "query failed: "+err.Error())
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"trades": trades,
		"count":  len(trades),
		"filter": map[string]any{
			"source": f.source,
			"sym":    f.sym,
			"from":   nullableTime(f.from),
			"to":     nullableTime(f.to),
			"run_id": f.runID,
			"limit":  f.limit,
			"offset": f.offset,
		},
	})
}

// GetTradeStats handles GET /api/v1/trades/stats
// Returns aggregate statistics for the filtered trade set.
func (h *TradesHandler) GetTradeStats(w http.ResponseWriter, r *http.Request) {
	f, err := parseTradeFilter(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	f.limit = 10000 // raise limit for stats computation

	ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
	defer cancel()

	trades, err := h.queryTrades(ctx, f)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "query failed: "+err.Error())
		return
	}

	stats := computeTradeStats(trades)
	writeJSON(w, http.StatusOK, stats)
}

// queryTrades builds a dynamic SQL query and returns matching TradeRecords.
func (h *TradesHandler) queryTrades(ctx context.Context, f tradeFilter) ([]TradeRecord, error) {
	// Determine which DB to query based on source filter.
	// For simplicity, the warehouse DB holds both live fills (joined to strategy_runs)
	// and backtest trades. Adjust the table/join logic to match your schema.
	var (
		whereClauses []string
		args         []any
	)

	if f.source == "live" {
		whereClauses = append(whereClauses, "t.source = 'live'")
	} else if f.source == "backtest" {
		whereClauses = append(whereClauses, "t.source = 'backtest'")
	}
	if f.sym != "" {
		whereClauses = append(whereClauses, "t.symbol = ?")
		args = append(args, f.sym)
	}
	if !f.from.IsZero() {
		whereClauses = append(whereClauses, "t.entry_time >= ?")
		args = append(args, f.from.UTC().Format(time.RFC3339))
	}
	if !f.to.IsZero() {
		whereClauses = append(whereClauses, "t.entry_time <= ?")
		args = append(args, f.to.UTC().Format(time.RFC3339))
	}
	if f.runID > 0 {
		whereClauses = append(whereClauses, "t.run_id = ?")
		args = append(args, f.runID)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			t.id,
			t.run_id,
			t.symbol,
			t.side,
			t.qty,
			t.entry_price,
			COALESCE(t.exit_price, 0.0)   AS exit_price,
			t.entry_time,
			COALESCE(t.exit_time, '')      AS exit_time,
			COALESCE(t.pnl, 0.0)          AS pnl,
			COALESCE(t.pnl_pct, 0.0)      AS pnl_pct,
			COALESCE(t.commission, 0.0)    AS commission,
			t.source,
			COALESCE(t.exit_reason, '')    AS exit_reason,
			COALESCE(t.timeframe, '')      AS timeframe,
			COALESCE(t.strategy_tag, '')   AS strategy_tag
		FROM trades t
		%s
		ORDER BY t.entry_time DESC
		LIMIT ? OFFSET ?
	`, where)

	args = append(args, f.limit, f.offset)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		return nil, err
	}

	records := make([]TradeRecord, 0, len(rows))
	for _, row := range rows {
		rec := TradeRecord{
			ID:          toInt64(row["id"]),
			RunID:       toInt64(row["run_id"]),
			Symbol:      toString(row["symbol"]),
			Side:        toString(row["side"]),
			Qty:         toFloat64(row["qty"]),
			EntryPrice:  toFloat64(row["entry_price"]),
			ExitPrice:   toFloat64(row["exit_price"]),
			EntryTime:   toString(row["entry_time"]),
			ExitTime:    toString(row["exit_time"]),
			PnL:         toFloat64(row["pnl"]),
			PnLPct:      toFloat64(row["pnl_pct"]),
			Commission:  toFloat64(row["commission"]),
			Source:      toString(row["source"]),
			ExitReason:  toString(row["exit_reason"]),
			Timeframe:   toString(row["timeframe"]),
			StrategyTag: toString(row["strategy_tag"]),
		}
		records = append(records, rec)
	}
	return records, nil
}

// computeTradeStats derives aggregate statistics from a slice of TradeRecords.
func computeTradeStats(trades []TradeRecord) TradeStats {
	if len(trades) == 0 {
		return TradeStats{}
	}

	stats := TradeStats{TotalTrades: len(trades)}
	var totalWin, totalLoss float64
	dailyPnL := make(map[string]float64)

	for _, t := range trades {
		stats.TotalPnL += t.PnL
		stats.TotalCommission += t.Commission

		date := ""
		if len(t.EntryTime) >= 10 {
			date = t.EntryTime[:10]
		}
		dailyPnL[date] += t.PnL

		if t.PnL > 0 {
			stats.WinCount++
			totalWin += t.PnL
		} else if t.PnL < 0 {
			stats.LossCount++
			totalLoss += t.PnL
		}
	}

	if stats.TotalTrades > 0 {
		stats.WinRate = float64(stats.WinCount) / float64(stats.TotalTrades)
		stats.AvgPnL = stats.TotalPnL / float64(stats.TotalTrades)
	}
	if stats.WinCount > 0 {
		stats.AvgWin = totalWin / float64(stats.WinCount)
	}
	if stats.LossCount > 0 {
		stats.AvgLoss = totalLoss / float64(stats.LossCount)
	}
	if totalLoss != 0 {
		stats.ProfitFactor = totalWin / -totalLoss
	}

	// Compute Sharpe from daily PnL series.
	if len(dailyPnL) > 1 {
		var returns []float64
		for _, v := range dailyPnL {
			returns = append(returns, v)
		}
		mean, std := meanStd(returns)
		if std > 0 {
			stats.Sharpe = (mean / std) * 16 // annualise sqrt(252) ≈ 15.87
		}
		// Drawdown
		var peak, dd float64
		equity := 0.0
		for _, v := range returns {
			equity += v
			if equity > peak {
				peak = equity
			}
			if peak > 0 {
				d := (equity - peak) / peak
				if d < dd {
					dd = d
				}
			}
		}
		stats.MaxDrawdown = dd
	}

	return stats
}

// meanStd returns the mean and population standard deviation of a float slice.
func meanStd(xs []float64) (mean, std float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	for _, x := range xs {
		mean += x
	}
	mean /= float64(len(xs))
	for _, x := range xs {
		d := x - mean
		std += d * d
	}
	std = sqrt(std / float64(len(xs)))
	return mean, std
}

// sqrt is a simple float square root to avoid importing math in one extra place.
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Newton-Raphson
	z := x
	for i := 0; i < 50; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}

// nullableTime returns an RFC3339 string or nil for a zero time.
func nullableTime(t time.Time) any {
	if t.IsZero() {
		return nil
	}
	return t.UTC().Format(time.RFC3339)
}

// --- shared JSON helpers used across all handlers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// --- type coercion helpers for db.Row values ---

func toInt64(v any) int64 {
	switch x := v.(type) {
	case int64:
		return x
	case float64:
		return int64(x)
	case int:
		return int64(x)
	case string:
		n, _ := strconv.ParseInt(x, 10, 64)
		return n
	}
	return 0
}

func toFloat64(v any) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case int64:
		return float64(x)
	case string:
		f, _ := strconv.ParseFloat(x, 64)
		return f
	}
	return 0
}

func toString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

func toBool(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case int64:
		return x != 0
	case string:
		return x == "1" || strings.EqualFold(x, "true")
	}
	return false
}
