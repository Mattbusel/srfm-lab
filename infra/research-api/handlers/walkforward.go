package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/srfm/research-api/db"
)

// WFRun represents metadata for a walk-forward or CPCV analysis run.
type WFRun struct {
	ID              int64   `json:"id"`
	RunID           int64   `json:"run_id"`
	StartedAt       string  `json:"started_at"`
	FinishedAt      string  `json:"finished_at,omitempty"`
	Status          string  `json:"status"`
	WFType          string  `json:"wf_type"`
	Symbol          string  `json:"symbol,omitempty"`
	Universe        string  `json:"universe,omitempty"`
	InSampleBars    int64   `json:"in_sample_bars"`
	OOSBars         int64   `json:"oos_bars"`
	NFolds          int64   `json:"n_folds"`
	NPaths          int64   `json:"n_paths,omitempty"`
	Metric          string  `json:"metric"`
	TotalISSharpe   float64 `json:"total_is_sharpe,omitempty"`
	TotalOOSSharpe  float64 `json:"total_oos_sharpe,omitempty"`
	EfficiencyRatio float64 `json:"efficiency_ratio,omitempty"`
	PBOScore        float64 `json:"pbo_score,omitempty"`
	BestParams      json.RawMessage `json:"best_params,omitempty"`
	Notes           string  `json:"notes,omitempty"`
}

// WFFold represents per-fold results within a walk-forward run.
type WFFold struct {
	ID          int64   `json:"id"`
	WFRunID     int64   `json:"wf_run_id"`
	FoldIndex   int64   `json:"fold_index"`
	ISStart     string  `json:"is_start"`
	ISEnd       string  `json:"is_end"`
	OOSStart    string  `json:"oos_start"`
	OOSEnd      string  `json:"oos_end"`
	ISSharpe    float64 `json:"is_sharpe,omitempty"`
	OOSSharpe   float64 `json:"oos_sharpe,omitempty"`
	ISCAGR      float64 `json:"is_cagr,omitempty"`
	OOSCAGR     float64 `json:"oos_cagr,omitempty"`
	ISMaxDD     float64 `json:"is_max_dd,omitempty"`
	OOSMaxDD    float64 `json:"oos_max_dd,omitempty"`
	ISTrades    int64   `json:"is_trades,omitempty"`
	OOSTrades   int64   `json:"oos_trades,omitempty"`
	BestParams  json.RawMessage `json:"best_params,omitempty"`
}

// CPCVPath represents one synthetic backtest path from CPCV analysis.
type CPCVPath struct {
	ID          int64   `json:"id"`
	WFRunID     int64   `json:"wf_run_id"`
	PathIndex   int64   `json:"path_index"`
	Sharpe      float64 `json:"sharpe,omitempty"`
	CAGR        float64 `json:"cagr,omitempty"`
	MaxDrawdown float64 `json:"max_drawdown,omitempty"`
	Sortino     float64 `json:"sortino,omitempty"`
	WinRate     float64 `json:"win_rate,omitempty"`
	TradeCount  int64   `json:"trade_count,omitempty"`
}

// WalkForwardHandler serves walk-forward analysis data.
type WalkForwardHandler struct {
	warehouse *db.SQLiteDB
}

// NewWalkForwardHandler creates a WalkForwardHandler.
func NewWalkForwardHandler(warehouse *db.SQLiteDB) *WalkForwardHandler {
	return &WalkForwardHandler{warehouse: warehouse}
}

// GetRuns handles GET /api/v1/walkforward/runs?status=&wf_type=&run_id=&limit=
func (h *WalkForwardHandler) GetRuns(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	if status := q.Get("status"); status != "" {
		whereClauses = append(whereClauses, "wr.status = ?")
		args = append(args, status)
	}
	if wfType := q.Get("wf_type"); wfType != "" {
		whereClauses = append(whereClauses, "wr.wf_type = ?")
		args = append(args, wfType)
	}
	if runID := q.Get("run_id"); runID != "" {
		whereClauses = append(whereClauses, "wr.run_id = ?")
		args = append(args, runID)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			wr.id, wr.run_id, wr.started_at,
			COALESCE(wr.finished_at, '')         AS finished_at,
			wr.status, wr.wf_type,
			COALESCE(i.symbol, '')               AS symbol,
			COALESCE(wr.universe, '')            AS universe,
			wr.in_sample_bars, wr.oos_bars,
			wr.n_folds,
			COALESCE(wr.n_paths, 0)              AS n_paths,
			wr.metric,
			COALESCE(wr.total_is_sharpe, 0.0)   AS total_is_sharpe,
			COALESCE(wr.total_oos_sharpe, 0.0)  AS total_oos_sharpe,
			COALESCE(wr.efficiency_ratio, 0.0)  AS efficiency_ratio,
			COALESCE(wr.pbo_score, 0.0)         AS pbo_score,
			wr.best_params,
			COALESCE(wr.notes, '')              AS notes
		FROM wf_runs wr
		LEFT JOIN instruments i ON i.id = wr.instrument_id
		%s
		ORDER BY wr.started_at DESC
		LIMIT 100
	`, where)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	runs := make([]WFRun, 0, len(rows))
	for _, row := range rows {
		run := WFRun{
			ID:              toInt64(row["id"]),
			RunID:           toInt64(row["run_id"]),
			StartedAt:       toString(row["started_at"]),
			FinishedAt:      toString(row["finished_at"]),
			Status:          toString(row["status"]),
			WFType:          toString(row["wf_type"]),
			Symbol:          toString(row["symbol"]),
			Universe:        toString(row["universe"]),
			InSampleBars:    toInt64(row["in_sample_bars"]),
			OOSBars:         toInt64(row["oos_bars"]),
			NFolds:          toInt64(row["n_folds"]),
			NPaths:          toInt64(row["n_paths"]),
			Metric:          toString(row["metric"]),
			TotalISSharpe:   toFloat64(row["total_is_sharpe"]),
			TotalOOSSharpe:  toFloat64(row["total_oos_sharpe"]),
			EfficiencyRatio: toFloat64(row["efficiency_ratio"]),
			PBOScore:        toFloat64(row["pbo_score"]),
			Notes:           toString(row["notes"]),
		}
		if raw := toString(row["best_params"]); raw != "" && raw != "{}" {
			run.BestParams = json.RawMessage(raw)
		}
		runs = append(runs, run)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"runs":  runs,
		"count": len(runs),
	})
}

// GetRunDetail handles GET /api/v1/walkforward/runs/{id}
// Returns the WFRun along with its folds and (if CPCV) path distribution.
func (h *WalkForwardHandler) GetRunDetail(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "missing id")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()

	// Fetch the run header.
	runRow, err := h.warehouse.QueryRow(ctx, `
		SELECT
			wr.id, wr.run_id, wr.started_at,
			COALESCE(wr.finished_at, '') AS finished_at,
			wr.status, wr.wf_type,
			COALESCE(i.symbol, '')       AS symbol,
			COALESCE(wr.universe, '')    AS universe,
			wr.in_sample_bars, wr.oos_bars,
			wr.n_folds,
			COALESCE(wr.n_paths, 0)     AS n_paths,
			wr.metric,
			COALESCE(wr.total_is_sharpe, 0.0)  AS total_is_sharpe,
			COALESCE(wr.total_oos_sharpe, 0.0) AS total_oos_sharpe,
			COALESCE(wr.efficiency_ratio, 0.0) AS efficiency_ratio,
			COALESCE(wr.pbo_score, 0.0)        AS pbo_score,
			wr.best_params,
			COALESCE(wr.notes, '')             AS notes
		FROM wf_runs wr
		LEFT JOIN instruments i ON i.id = wr.instrument_id
		WHERE wr.id = ?
	`, id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}
	if runRow == nil {
		writeError(w, http.StatusNotFound, fmt.Sprintf("wf_run %s not found", id))
		return
	}

	run := WFRun{
		ID:              toInt64(runRow["id"]),
		RunID:           toInt64(runRow["run_id"]),
		StartedAt:       toString(runRow["started_at"]),
		FinishedAt:      toString(runRow["finished_at"]),
		Status:          toString(runRow["status"]),
		WFType:          toString(runRow["wf_type"]),
		Symbol:          toString(runRow["symbol"]),
		Universe:        toString(runRow["universe"]),
		InSampleBars:    toInt64(runRow["in_sample_bars"]),
		OOSBars:         toInt64(runRow["oos_bars"]),
		NFolds:          toInt64(runRow["n_folds"]),
		NPaths:          toInt64(runRow["n_paths"]),
		Metric:          toString(runRow["metric"]),
		TotalISSharpe:   toFloat64(runRow["total_is_sharpe"]),
		TotalOOSSharpe:  toFloat64(runRow["total_oos_sharpe"]),
		EfficiencyRatio: toFloat64(runRow["efficiency_ratio"]),
		PBOScore:        toFloat64(runRow["pbo_score"]),
		Notes:           toString(runRow["notes"]),
	}
	if raw := toString(runRow["best_params"]); raw != "" && raw != "{}" {
		run.BestParams = json.RawMessage(raw)
	}

	// Fetch folds.
	foldRows, err := h.warehouse.QueryRows(ctx, `
		SELECT
			id, wf_run_id, fold_index,
			is_start, is_end, oos_start, oos_end,
			COALESCE(is_sharpe, 0.0)  AS is_sharpe,
			COALESCE(oos_sharpe, 0.0) AS oos_sharpe,
			COALESCE(is_cagr, 0.0)   AS is_cagr,
			COALESCE(oos_cagr, 0.0)  AS oos_cagr,
			COALESCE(is_max_dd, 0.0) AS is_max_dd,
			COALESCE(oos_max_dd, 0.0) AS oos_max_dd,
			COALESCE(is_trades, 0)   AS is_trades,
			COALESCE(oos_trades, 0)  AS oos_trades,
			best_params
		FROM wf_folds
		WHERE wf_run_id = ?
		ORDER BY fold_index ASC
	`, id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "folds query: "+err.Error())
		return
	}

	folds := make([]WFFold, 0, len(foldRows))
	for _, row := range foldRows {
		fold := WFFold{
			ID:        toInt64(row["id"]),
			WFRunID:   toInt64(row["wf_run_id"]),
			FoldIndex: toInt64(row["fold_index"]),
			ISStart:   toString(row["is_start"]),
			ISEnd:     toString(row["is_end"]),
			OOSStart:  toString(row["oos_start"]),
			OOSEnd:    toString(row["oos_end"]),
			ISSharpe:  toFloat64(row["is_sharpe"]),
			OOSSharpe: toFloat64(row["oos_sharpe"]),
			ISCAGR:    toFloat64(row["is_cagr"]),
			OOSCAGR:   toFloat64(row["oos_cagr"]),
			ISMaxDD:   toFloat64(row["is_max_dd"]),
			OOSMaxDD:  toFloat64(row["oos_max_dd"]),
			ISTrades:  toInt64(row["is_trades"]),
			OOSTrades: toInt64(row["oos_trades"]),
		}
		if raw := toString(row["best_params"]); raw != "" && raw != "{}" {
			fold.BestParams = json.RawMessage(raw)
		}
		folds = append(folds, fold)
	}

	// Fetch CPCV paths if applicable.
	var paths []CPCVPath
	if run.WFType == "cpcv" {
		pathRows, err := h.warehouse.QueryRows(ctx, `
			SELECT
				id, wf_run_id, path_index,
				COALESCE(sharpe, 0.0)       AS sharpe,
				COALESCE(cagr, 0.0)         AS cagr,
				COALESCE(max_drawdown, 0.0) AS max_drawdown,
				COALESCE(sortino, 0.0)      AS sortino,
				COALESCE(win_rate, 0.0)     AS win_rate,
				COALESCE(trade_count, 0)    AS trade_count
			FROM cpcv_paths
			WHERE wf_run_id = ?
			ORDER BY path_index ASC
		`, id)
		if err == nil {
			for _, row := range pathRows {
				paths = append(paths, CPCVPath{
					ID:          toInt64(row["id"]),
					WFRunID:     toInt64(row["wf_run_id"]),
					PathIndex:   toInt64(row["path_index"]),
					Sharpe:      toFloat64(row["sharpe"]),
					CAGR:        toFloat64(row["cagr"]),
					MaxDrawdown: toFloat64(row["max_drawdown"]),
					Sortino:     toFloat64(row["sortino"]),
					WinRate:     toFloat64(row["win_rate"]),
					TradeCount:  toInt64(row["trade_count"]),
				})
			}
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"run":   run,
		"folds": folds,
		"paths": paths,
	})
}
