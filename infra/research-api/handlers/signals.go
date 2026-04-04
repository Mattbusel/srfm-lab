package handlers

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/srfm/research-api/db"
)

// ICSeries represents a row in the ic_series table.
type ICSeries struct {
	ID           int64   `json:"id"`
	RunID        int64   `json:"run_id"`
	InstrumentID int64   `json:"instrument_id"`
	Symbol       string  `json:"symbol"`
	SignalName   string  `json:"signal_name"`
	Timeframe    string  `json:"timeframe"`
	ComputedAt   string  `json:"computed_at"`
	WindowBars   int64   `json:"window_bars"`
	IC           float64 `json:"ic"`
	ICIR         float64 `json:"ic_ir,omitempty"`
	ICTStat      float64 `json:"ic_t_stat,omitempty"`
	HitRate      float64 `json:"hit_rate,omitempty"`
	ObsCount     int64   `json:"obs_count"`
	FwdHorizon   int64   `json:"fwd_horizon"`
}

// FactorReturn represents a row in the factor_returns table.
type FactorReturn struct {
	ID          int64   `json:"id"`
	RunID       int64   `json:"run_id"`
	SignalName  string  `json:"signal_name"`
	Timeframe   string  `json:"timeframe"`
	PeriodStart string  `json:"period_start"`
	PeriodEnd   string  `json:"period_end"`
	Universe    string  `json:"universe,omitempty"`
	Q1Return    float64 `json:"q1_return"`
	Q2Return    float64 `json:"q2_return"`
	Q3Return    float64 `json:"q3_return"`
	Q4Return    float64 `json:"q4_return"`
	Q5Return    float64 `json:"q5_return"`
	LSReturn    float64 `json:"ls_return"`
	LSSharpe    float64 `json:"ls_sharpe,omitempty"`
	Turnover    float64 `json:"turnover,omitempty"`
}

// AlphaDecay represents a row in the alpha_decay table.
type AlphaDecay struct {
	ID            int64   `json:"id"`
	RunID         int64   `json:"run_id"`
	Symbol        string  `json:"symbol,omitempty"`
	SignalName    string  `json:"signal_name"`
	ComputedAt    string  `json:"computed_at"`
	HorizonBars   int64   `json:"horizon_bars"`
	IC            float64 `json:"ic"`
	ICSE          float64 `json:"ic_se,omitempty"`
	TStat         float64 `json:"t_stat,omitempty"`
	IsSignificant bool    `json:"is_significant"`
	HalfLifeBars  float64 `json:"half_life_bars,omitempty"`
	DecayRate     float64 `json:"decay_rate,omitempty"`
	ObsCount      int64   `json:"obs_count,omitempty"`
}

// SignalsHandler serves signal analytics data.
type SignalsHandler struct {
	warehouse *db.SQLiteDB
}

// NewSignalsHandler creates a SignalsHandler.
func NewSignalsHandler(warehouse *db.SQLiteDB) *SignalsHandler {
	return &SignalsHandler{warehouse: warehouse}
}

// GetICHistory handles GET /api/v1/signals/ic?instrument=&signal=&window=&from=&to=&limit=
// Returns rolling IC series for a given signal / instrument combination.
func (h *SignalsHandler) GetICHistory(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	instrument := strings.ToUpper(q.Get("instrument"))
	if instrument != "" {
		whereClauses = append(whereClauses, "i.symbol = ?")
		args = append(args, instrument)
	}
	if signal := q.Get("signal"); signal != "" {
		whereClauses = append(whereClauses, "ic.signal_name = ?")
		args = append(args, signal)
	}
	if window := q.Get("window"); window != "" {
		whereClauses = append(whereClauses, "ic.window_bars = ?")
		args = append(args, window)
	}
	if from := q.Get("from"); from != "" {
		whereClauses = append(whereClauses, "ic.computed_at >= ?")
		args = append(args, from)
	}
	if to := q.Get("to"); to != "" {
		whereClauses = append(whereClauses, "ic.computed_at <= ?")
		args = append(args, to)
	}
	if runID := q.Get("run_id"); runID != "" {
		whereClauses = append(whereClauses, "ic.run_id = ?")
		args = append(args, runID)
	}

	limit := 1000
	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			ic.id, ic.run_id, ic.instrument_id,
			COALESCE(i.symbol, '')           AS symbol,
			ic.signal_name, ic.timeframe, ic.computed_at,
			ic.window_bars,
			ic.ic,
			COALESCE(ic.ic_ir, 0.0)         AS ic_ir,
			COALESCE(ic.ic_t_stat, 0.0)     AS ic_t_stat,
			COALESCE(ic.hit_rate, 0.0)      AS hit_rate,
			ic.obs_count, ic.fwd_horizon
		FROM ic_series ic
		LEFT JOIN instruments i ON i.id = ic.instrument_id
		%s
		ORDER BY ic.computed_at DESC
		LIMIT %d
	`, where, limit)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	series := make([]ICSeries, 0, len(rows))
	for _, row := range rows {
		series = append(series, ICSeries{
			ID:           toInt64(row["id"]),
			RunID:        toInt64(row["run_id"]),
			InstrumentID: toInt64(row["instrument_id"]),
			Symbol:       toString(row["symbol"]),
			SignalName:   toString(row["signal_name"]),
			Timeframe:    toString(row["timeframe"]),
			ComputedAt:   toString(row["computed_at"]),
			WindowBars:   toInt64(row["window_bars"]),
			IC:           toFloat64(row["ic"]),
			ICIR:         toFloat64(row["ic_ir"]),
			ICTStat:      toFloat64(row["ic_t_stat"]),
			HitRate:      toFloat64(row["hit_rate"]),
			ObsCount:     toInt64(row["obs_count"]),
			FwdHorizon:   toInt64(row["fwd_horizon"]),
		})
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"ic_series": series,
		"count":     len(series),
	})
}

// GetFactorReturns handles GET /api/v1/signals/factor-returns?signal=&from=&to=
func (h *SignalsHandler) GetFactorReturns(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	if signal := q.Get("signal"); signal != "" {
		whereClauses = append(whereClauses, "fr.signal_name = ?")
		args = append(args, signal)
	}
	if from := q.Get("from"); from != "" {
		whereClauses = append(whereClauses, "fr.period_start >= ?")
		args = append(args, from)
	}
	if to := q.Get("to"); to != "" {
		whereClauses = append(whereClauses, "fr.period_end <= ?")
		args = append(args, to)
	}
	if runID := q.Get("run_id"); runID != "" {
		whereClauses = append(whereClauses, "fr.run_id = ?")
		args = append(args, runID)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			fr.id, fr.run_id, fr.signal_name, fr.timeframe,
			fr.period_start, fr.period_end,
			COALESCE(fr.universe, '')        AS universe,
			COALESCE(fr.q1_return, 0.0)     AS q1_return,
			COALESCE(fr.q2_return, 0.0)     AS q2_return,
			COALESCE(fr.q3_return, 0.0)     AS q3_return,
			COALESCE(fr.q4_return, 0.0)     AS q4_return,
			COALESCE(fr.q5_return, 0.0)     AS q5_return,
			COALESCE(fr.ls_return, 0.0)     AS ls_return,
			COALESCE(fr.ls_sharpe, 0.0)     AS ls_sharpe,
			COALESCE(fr.turnover, 0.0)      AS turnover
		FROM factor_returns fr
		%s
		ORDER BY fr.period_start DESC
		LIMIT 500
	`, where)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	results := make([]FactorReturn, 0, len(rows))
	for _, row := range rows {
		results = append(results, FactorReturn{
			ID:          toInt64(row["id"]),
			RunID:       toInt64(row["run_id"]),
			SignalName:  toString(row["signal_name"]),
			Timeframe:   toString(row["timeframe"]),
			PeriodStart: toString(row["period_start"]),
			PeriodEnd:   toString(row["period_end"]),
			Universe:    toString(row["universe"]),
			Q1Return:    toFloat64(row["q1_return"]),
			Q2Return:    toFloat64(row["q2_return"]),
			Q3Return:    toFloat64(row["q3_return"]),
			Q4Return:    toFloat64(row["q4_return"]),
			Q5Return:    toFloat64(row["q5_return"]),
			LSReturn:    toFloat64(row["ls_return"]),
			LSSharpe:    toFloat64(row["ls_sharpe"]),
			Turnover:    toFloat64(row["turnover"]),
		})
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"factor_returns": results,
		"count":          len(results),
	})
}

// GetAlphaDecay handles GET /api/v1/signals/alpha-decay?signal=&instrument=
// Returns the full IC-vs-horizon decay curve for a signal.
func (h *SignalsHandler) GetAlphaDecay(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	if signal := q.Get("signal"); signal != "" {
		whereClauses = append(whereClauses, "ad.signal_name = ?")
		args = append(args, signal)
	}
	if inst := q.Get("instrument"); inst != "" {
		whereClauses = append(whereClauses, "i.symbol = ?")
		args = append(args, strings.ToUpper(inst))
	}
	if runID := q.Get("run_id"); runID != "" {
		whereClauses = append(whereClauses, "ad.run_id = ?")
		args = append(args, runID)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			ad.id, ad.run_id,
			COALESCE(i.symbol, '')           AS symbol,
			ad.signal_name, ad.computed_at,
			ad.horizon_bars,
			COALESCE(ad.ic, 0.0)            AS ic,
			COALESCE(ad.ic_se, 0.0)         AS ic_se,
			COALESCE(ad.t_stat, 0.0)        AS t_stat,
			ad.is_significant,
			COALESCE(ad.half_life_bars, 0.0) AS half_life_bars,
			COALESCE(ad.decay_rate, 0.0)    AS decay_rate,
			COALESCE(ad.obs_count, 0)       AS obs_count
		FROM alpha_decay ad
		LEFT JOIN instruments i ON i.id = ad.instrument_id
		%s
		ORDER BY ad.signal_name, ad.horizon_bars ASC
		LIMIT 500
	`, where)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	results := make([]AlphaDecay, 0, len(rows))
	for _, row := range rows {
		results = append(results, AlphaDecay{
			ID:            toInt64(row["id"]),
			RunID:         toInt64(row["run_id"]),
			Symbol:        toString(row["symbol"]),
			SignalName:    toString(row["signal_name"]),
			ComputedAt:    toString(row["computed_at"]),
			HorizonBars:   toInt64(row["horizon_bars"]),
			IC:            toFloat64(row["ic"]),
			ICSE:          toFloat64(row["ic_se"]),
			TStat:         toFloat64(row["t_stat"]),
			IsSignificant: toBool(row["is_significant"]),
			HalfLifeBars:  toFloat64(row["half_life_bars"]),
			DecayRate:     toFloat64(row["decay_rate"]),
			ObsCount:      toInt64(row["obs_count"]),
		})
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"alpha_decay": results,
		"count":       len(results),
	})
}
