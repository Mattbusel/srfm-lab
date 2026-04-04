package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/srfm/research-api/db"
)

// RegimeDetection represents a single regime period.
type RegimeDetection struct {
	ID            int64   `json:"id"`
	InstrumentID  int64   `json:"instrument_id"`
	Symbol        string  `json:"symbol"`
	Method        string  `json:"method"`
	StartedAt     string  `json:"started_at"`
	EndedAt       string  `json:"ended_at,omitempty"`
	Regime        string  `json:"regime"`
	RegimeIndex   int64   `json:"regime_index,omitempty"`
	Confidence    float64 `json:"confidence,omitempty"`
	NStates       int64   `json:"n_states,omitempty"`
	Timeframe     string  `json:"timeframe"`
	Volatility    float64 `json:"volatility,omitempty"`
	TrendStrength float64 `json:"trend_strength,omitempty"`
	DurationBars  int64   `json:"duration_bars,omitempty"`
	IsCurrent     bool    `json:"is_current"`
	ModelVersion  string  `json:"model_version,omitempty"`
}

// TransitionRow represents one cell of the regime transition matrix.
type TransitionRow struct {
	FromRegime            string  `json:"from_regime"`
	ToRegime              string  `json:"to_regime"`
	Probability           float64 `json:"probability"`
	SampleCount           int64   `json:"sample_count"`
	ExpectedDurationBars  float64 `json:"expected_duration_bars,omitempty"`
}

// RegimesHandler serves regime detection and transition matrix data.
type RegimesHandler struct {
	warehouse *db.SQLiteDB
}

// NewRegimesHandler creates a RegimesHandler.
func NewRegimesHandler(warehouse *db.SQLiteDB) *RegimesHandler {
	return &RegimesHandler{warehouse: warehouse}
}

// GetCurrentRegime handles GET /api/v1/regimes/current?instrument=&method=
// Returns the latest active regime detection for each instrument/method combination.
func (h *RegimesHandler) GetCurrentRegime(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	whereClauses = append(whereClauses, "rd.is_current = 1")

	if inst := q.Get("instrument"); inst != "" {
		whereClauses = append(whereClauses, "i.symbol = ?")
		args = append(args, strings.ToUpper(inst))
	}
	if method := q.Get("method"); method != "" {
		whereClauses = append(whereClauses, "rd.method = ?")
		args = append(args, method)
	}

	where := "WHERE " + strings.Join(whereClauses, " AND ")

	query := fmt.Sprintf(`
		SELECT
			rd.id, rd.instrument_id,
			COALESCE(i.symbol, '')           AS symbol,
			rd.method, rd.started_at,
			COALESCE(rd.ended_at, '')        AS ended_at,
			rd.regime,
			COALESCE(rd.regime_index, 0)     AS regime_index,
			COALESCE(rd.confidence, 0.0)     AS confidence,
			COALESCE(rd.n_states, 0)         AS n_states,
			rd.timeframe,
			COALESCE(rd.volatility, 0.0)     AS volatility,
			COALESCE(rd.trend_strength, 0.0) AS trend_strength,
			COALESCE(rd.duration_bars, 0)    AS duration_bars,
			rd.is_current,
			COALESCE(rd.model_version, '')   AS model_version
		FROM regime_detections rd
		LEFT JOIN instruments i ON i.id = rd.instrument_id
		%s
		ORDER BY rd.started_at DESC
	`, where)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	regimes := rowsToRegimeDetections(rows)

	// If no DB rows, fall back to regime_log (legacy table used by some strategies).
	if len(regimes) == 0 {
		regimes, err = h.queryRegimeLog(ctx, q.Get("instrument"), true, "", "")
		if err != nil {
			writeError(w, http.StatusInternalServerError, "fallback query: "+err.Error())
			return
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"current_regimes": regimes,
		"count":           len(regimes),
		"as_of":           time.Now().UTC().Format(time.RFC3339),
	})
}

// GetRegimeHistory handles GET /api/v1/regimes/history?instrument=&method=&from=&to=
func (h *RegimesHandler) GetRegimeHistory(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	if inst := q.Get("instrument"); inst != "" {
		whereClauses = append(whereClauses, "i.symbol = ?")
		args = append(args, strings.ToUpper(inst))
	}
	if method := q.Get("method"); method != "" {
		whereClauses = append(whereClauses, "rd.method = ?")
		args = append(args, method)
	}
	if from := q.Get("from"); from != "" {
		whereClauses = append(whereClauses, "rd.started_at >= ?")
		args = append(args, from)
	}
	if to := q.Get("to"); to != "" {
		whereClauses = append(whereClauses, "rd.started_at <= ?")
		args = append(args, to)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			rd.id, rd.instrument_id,
			COALESCE(i.symbol, '')           AS symbol,
			rd.method, rd.started_at,
			COALESCE(rd.ended_at, '')        AS ended_at,
			rd.regime,
			COALESCE(rd.regime_index, 0)     AS regime_index,
			COALESCE(rd.confidence, 0.0)     AS confidence,
			COALESCE(rd.n_states, 0)         AS n_states,
			rd.timeframe,
			COALESCE(rd.volatility, 0.0)     AS volatility,
			COALESCE(rd.trend_strength, 0.0) AS trend_strength,
			COALESCE(rd.duration_bars, 0)    AS duration_bars,
			rd.is_current,
			COALESCE(rd.model_version, '')   AS model_version
		FROM regime_detections rd
		LEFT JOIN instruments i ON i.id = rd.instrument_id
		%s
		ORDER BY rd.started_at DESC
		LIMIT 1000
	`, where)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	regimes := rowsToRegimeDetections(rows)

	// Fall back to legacy regime_log if the new table has no data.
	if len(regimes) == 0 {
		regimes, err = h.queryRegimeLog(ctx, q.Get("instrument"), false, q.Get("from"), q.Get("to"))
		if err != nil {
			writeError(w, http.StatusInternalServerError, "fallback query: "+err.Error())
			return
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"regimes": regimes,
		"count":   len(regimes),
	})
}

// GetTransitionMatrix handles GET /api/v1/regimes/transition-matrix?instrument=&method=
func (h *RegimesHandler) GetTransitionMatrix(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	if inst := q.Get("instrument"); inst != "" {
		whereClauses = append(whereClauses, "i.symbol = ?")
		args = append(args, strings.ToUpper(inst))
	}
	if method := q.Get("method"); method != "" {
		whereClauses = append(whereClauses, "tm.method = ?")
		args = append(args, method)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			tm.from_regime, tm.to_regime,
			tm.probability,
			tm.sample_count,
			COALESCE(tm.expected_duration_bars, 0.0) AS expected_duration_bars
		FROM transition_matrices tm
		LEFT JOIN instruments i ON i.id = tm.instrument_id
		%s
		ORDER BY tm.from_regime, tm.probability DESC
	`, where)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		// Fall back to the legacy regime_transition_matrix table.
		rows, err = h.legacyTransitionMatrix(ctx, q.Get("instrument"))
		if err != nil {
			writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
			return
		}
	}

	transitions := make([]TransitionRow, 0, len(rows))
	for _, row := range rows {
		transitions = append(transitions, TransitionRow{
			FromRegime:           toString(row["from_regime"]),
			ToRegime:             toString(row["to_regime"]),
			Probability:          toFloat64(row["probability"]),
			SampleCount:          toInt64(row["sample_count"]),
			ExpectedDurationBars: toFloat64(row["expected_duration_bars"]),
		})
	}

	// Build a matrix representation as well.
	matrix := buildTransitionMatrix(transitions)

	writeJSON(w, http.StatusOK, map[string]any{
		"transitions": transitions,
		"matrix":      matrix,
		"count":       len(transitions),
	})
}

// queryRegimeLog falls back to the legacy regime_periods table when
// regime_detections has no data.
func (h *RegimesHandler) queryRegimeLog(
	ctx context.Context,
	instrument string,
	currentOnly bool,
	from, to string,
) ([]RegimeDetection, error) {
	var whereClauses []string
	var args []any

	if instrument != "" {
		whereClauses = append(whereClauses, "i.symbol = ?")
		args = append(args, strings.ToUpper(instrument))
	}
	if currentOnly {
		whereClauses = append(whereClauses, "rp.ended_at IS NULL")
	}
	if from != "" {
		whereClauses = append(whereClauses, "rp.started_at >= ?")
		args = append(args, from)
	}
	if to != "" {
		whereClauses = append(whereClauses, "rp.started_at <= ?")
		args = append(args, to)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			rp.id, rp.instrument_id,
			COALESCE(i.symbol, '')     AS symbol,
			'bh_state'                 AS method,
			rp.started_at,
			COALESCE(rp.ended_at, '') AS ended_at,
			rp.regime,
			0                          AS regime_index,
			0.0                        AS confidence,
			0                          AS n_states,
			COALESCE(rp.timeframe, '1d') AS timeframe,
			0.0                        AS volatility,
			0.0                        AS trend_strength,
			COALESCE(rp.duration_bars, 0) AS duration_bars,
			CASE WHEN rp.ended_at IS NULL THEN 1 ELSE 0 END AS is_current,
			''                         AS model_version
		FROM regime_periods rp
		LEFT JOIN instruments i ON i.id = rp.instrument_id
		%s
		ORDER BY rp.started_at DESC
		LIMIT 500
	`, where)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	return rowsToRegimeDetections(rows), nil
}

// legacyTransitionMatrix falls back to the regime_transition_matrix table.
func (h *RegimesHandler) legacyTransitionMatrix(ctx context.Context, instrument string) ([]db.Row, error) {
	var whereClauses []string
	var args []any
	if instrument != "" {
		whereClauses = append(whereClauses, "i.symbol = ?")
		args = append(args, strings.ToUpper(instrument))
	}
	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}
	query := fmt.Sprintf(`
		SELECT
			rtm.from_regime, rtm.to_regime,
			rtm.probability,
			COALESCE(rtm.count, 0)         AS sample_count,
			COALESCE(rtm.avg_duration, 0.0) AS expected_duration_bars
		FROM regime_transition_matrix rtm
		LEFT JOIN instruments i ON i.id = rtm.instrument_id
		%s
		ORDER BY rtm.from_regime, rtm.probability DESC
	`, where)
	return h.warehouse.QueryRows(ctx, query, args...)
}

func rowsToRegimeDetections(rows []db.Row) []RegimeDetection {
	out := make([]RegimeDetection, 0, len(rows))
	for _, row := range rows {
		out = append(out, RegimeDetection{
			ID:            toInt64(row["id"]),
			InstrumentID:  toInt64(row["instrument_id"]),
			Symbol:        toString(row["symbol"]),
			Method:        toString(row["method"]),
			StartedAt:     toString(row["started_at"]),
			EndedAt:       toString(row["ended_at"]),
			Regime:        toString(row["regime"]),
			RegimeIndex:   toInt64(row["regime_index"]),
			Confidence:    toFloat64(row["confidence"]),
			NStates:       toInt64(row["n_states"]),
			Timeframe:     toString(row["timeframe"]),
			Volatility:    toFloat64(row["volatility"]),
			TrendStrength: toFloat64(row["trend_strength"]),
			DurationBars:  toInt64(row["duration_bars"]),
			IsCurrent:     toBool(row["is_current"]),
			ModelVersion:  toString(row["model_version"]),
		})
	}
	return out
}

// buildTransitionMatrix converts a flat list of transitions into a
// JSON-serialisable map[from][to]=probability structure.
func buildTransitionMatrix(rows []TransitionRow) map[string]map[string]float64 {
	m := make(map[string]map[string]float64)
	for _, r := range rows {
		if m[r.FromRegime] == nil {
			m[r.FromRegime] = make(map[string]float64)
		}
		m[r.FromRegime][r.ToRegime] = r.Probability
	}
	return m
}

// Ensure json.RawMessage is available in this file.
var _ = json.RawMessage{}
