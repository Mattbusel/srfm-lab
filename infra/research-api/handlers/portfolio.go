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

// PortfolioConstruction represents a solved portfolio weight set.
type PortfolioConstruction struct {
	ID              int64              `json:"id"`
	RunID           int64              `json:"run_id,omitempty"`
	ConstructedAt   string             `json:"constructed_at"`
	Method          string             `json:"method"`
	Universe        string             `json:"universe,omitempty"`
	NAssets         int64              `json:"n_assets"`
	Weights         map[string]float64 `json:"weights"`
	ExpectedReturn  float64            `json:"expected_return,omitempty"`
	ExpectedVol     float64            `json:"expected_vol,omitempty"`
	ExpectedSharpe  float64            `json:"expected_sharpe,omitempty"`
	ExpectedVaR95   float64            `json:"expected_var95,omitempty"`
	ExpectedCVaR95  float64            `json:"expected_cvar95,omitempty"`
	TurnoverVsPrev  float64            `json:"turnover_vs_prev,omitempty"`
	MaxWeight       float64            `json:"max_weight,omitempty"`
	Concentration   float64            `json:"concentration,omitempty"`
	SolverStatus    string             `json:"solver_status"`
	IsLive          bool               `json:"is_live"`
}

// RiskMetrics represents a risk metrics snapshot for a portfolio construction.
type RiskMetrics struct {
	ID                   int64   `json:"id"`
	ConstructionID       int64   `json:"construction_id"`
	ComputedAt           string  `json:"computed_at"`
	PeriodDays           int64   `json:"period_days"`
	Sharpe               float64 `json:"sharpe,omitempty"`
	Sortino              float64 `json:"sortino,omitempty"`
	Calmar               float64 `json:"calmar,omitempty"`
	CAGR                 float64 `json:"cagr,omitempty"`
	TotalReturn          float64 `json:"total_return,omitempty"`
	VolatilityAnn        float64 `json:"volatility_ann,omitempty"`
	MaxDrawdown          float64 `json:"max_drawdown,omitempty"`
	MaxDrawdownDuration  int64   `json:"max_drawdown_duration,omitempty"`
	VaR951D              float64 `json:"var_95_1d,omitempty"`
	VaR991D              float64 `json:"var_99_1d,omitempty"`
	CVaR951D             float64 `json:"cvar_95_1d,omitempty"`
	CVaR991D             float64 `json:"cvar_99_1d,omitempty"`
	VaRMethod            string  `json:"var_method"`
	BetaToSPY            float64 `json:"beta_to_spy,omitempty"`
	AlphaAnn             float64 `json:"alpha_ann,omitempty"`
	InfoRatio            float64 `json:"info_ratio,omitempty"`
	Skewness             float64 `json:"skewness,omitempty"`
	Kurtosis             float64 `json:"kurtosis,omitempty"`
	FactorExposures      json.RawMessage `json:"factor_exposures,omitempty"`
}

// CorrelationSnapshot represents a stored correlation matrix.
type CorrelationSnapshot struct {
	ID             int64              `json:"id"`
	ConstructionID int64              `json:"construction_id,omitempty"`
	ComputedAt     string             `json:"computed_at"`
	Universe       string             `json:"universe,omitempty"`
	NAssets        int64              `json:"n_assets"`
	LookbackDays   int64              `json:"lookback_days"`
	Method         string             `json:"method"`
	Symbols        []string           `json:"symbols"`
	Matrix         json.RawMessage    `json:"matrix"`
	AvgCorrelation float64            `json:"avg_correlation,omitempty"`
	MaxCorrelation float64            `json:"max_correlation,omitempty"`
	IsPositiveDef  bool               `json:"is_positive_definite"`
}

// PortfolioHandler serves portfolio construction and risk data.
type PortfolioHandler struct {
	warehouse *db.SQLiteDB
}

// NewPortfolioHandler creates a PortfolioHandler.
func NewPortfolioHandler(warehouse *db.SQLiteDB) *PortfolioHandler {
	return &PortfolioHandler{warehouse: warehouse}
}

// GetWeights handles GET /api/v1/portfolio/weights?method=&live_only=1&limit=
func (h *PortfolioHandler) GetWeights(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	if method := q.Get("method"); method != "" {
		whereClauses = append(whereClauses, "pc.method = ?")
		args = append(args, method)
	}
	if q.Get("live_only") == "1" {
		whereClauses = append(whereClauses, "pc.is_live = 1")
	}
	if runID := q.Get("run_id"); runID != "" {
		whereClauses = append(whereClauses, "pc.run_id = ?")
		args = append(args, runID)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			pc.id,
			COALESCE(pc.run_id, 0)              AS run_id,
			pc.constructed_at, pc.method,
			COALESCE(pc.universe, '')           AS universe,
			pc.n_assets,
			pc.weights,
			COALESCE(pc.expected_return, 0.0)   AS expected_return,
			COALESCE(pc.expected_vol, 0.0)      AS expected_vol,
			COALESCE(pc.expected_sharpe, 0.0)   AS expected_sharpe,
			COALESCE(pc.expected_var95, 0.0)    AS expected_var95,
			COALESCE(pc.expected_cvar95, 0.0)   AS expected_cvar95,
			COALESCE(pc.turnover_vs_prev, 0.0)  AS turnover_vs_prev,
			COALESCE(pc.max_weight, 0.0)        AS max_weight,
			COALESCE(pc.concentration, 0.0)     AS concentration,
			pc.solver_status,
			pc.is_live
		FROM portfolio_constructions pc
		%s
		ORDER BY pc.constructed_at DESC
		LIMIT 50
	`, where)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	constructions := make([]PortfolioConstruction, 0, len(rows))
	for _, row := range rows {
		pc := PortfolioConstruction{
			ID:             toInt64(row["id"]),
			RunID:          toInt64(row["run_id"]),
			ConstructedAt:  toString(row["constructed_at"]),
			Method:         toString(row["method"]),
			Universe:       toString(row["universe"]),
			NAssets:        toInt64(row["n_assets"]),
			ExpectedReturn: toFloat64(row["expected_return"]),
			ExpectedVol:    toFloat64(row["expected_vol"]),
			ExpectedSharpe: toFloat64(row["expected_sharpe"]),
			ExpectedVaR95:  toFloat64(row["expected_var95"]),
			ExpectedCVaR95: toFloat64(row["expected_cvar95"]),
			TurnoverVsPrev: toFloat64(row["turnover_vs_prev"]),
			MaxWeight:      toFloat64(row["max_weight"]),
			Concentration:  toFloat64(row["concentration"]),
			SolverStatus:   toString(row["solver_status"]),
			IsLive:         toBool(row["is_live"]),
		}

		// Parse weights JSON blob.
		if raw := toString(row["weights"]); raw != "" {
			_ = json.Unmarshal([]byte(raw), &pc.Weights)
		}
		if pc.Weights == nil {
			pc.Weights = make(map[string]float64)
		}

		constructions = append(constructions, pc)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"constructions": constructions,
		"count":         len(constructions),
	})
}

// GetRiskMetrics handles GET /api/v1/portfolio/risk?construction_id=&latest=1
func (h *PortfolioHandler) GetRiskMetrics(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	if cid := q.Get("construction_id"); cid != "" {
		whereClauses = append(whereClauses, "rm.construction_id = ?")
		args = append(args, cid)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	limit := "100"
	if q.Get("latest") == "1" {
		limit = "1"
	}

	query := fmt.Sprintf(`
		SELECT
			rm.id, rm.construction_id, rm.computed_at,
			rm.period_days,
			COALESCE(rm.sharpe, 0.0)                  AS sharpe,
			COALESCE(rm.sortino, 0.0)                 AS sortino,
			COALESCE(rm.calmar, 0.0)                  AS calmar,
			COALESCE(rm.cagr, 0.0)                    AS cagr,
			COALESCE(rm.total_return, 0.0)            AS total_return,
			COALESCE(rm.volatility_ann, 0.0)          AS volatility_ann,
			COALESCE(rm.max_drawdown, 0.0)            AS max_drawdown,
			COALESCE(rm.max_drawdown_duration, 0)     AS max_drawdown_duration,
			COALESCE(rm.var_95_1d, 0.0)               AS var_95_1d,
			COALESCE(rm.var_99_1d, 0.0)               AS var_99_1d,
			COALESCE(rm.cvar_95_1d, 0.0)              AS cvar_95_1d,
			COALESCE(rm.cvar_99_1d, 0.0)              AS cvar_99_1d,
			rm.var_method,
			COALESCE(rm.beta_to_spy, 0.0)             AS beta_to_spy,
			COALESCE(rm.alpha_ann, 0.0)               AS alpha_ann,
			COALESCE(rm.info_ratio, 0.0)              AS info_ratio,
			COALESCE(rm.skewness, 0.0)                AS skewness,
			COALESCE(rm.kurtosis, 0.0)                AS kurtosis,
			rm.factor_exposures
		FROM risk_metrics rm
		%s
		ORDER BY rm.computed_at DESC
		LIMIT %s
	`, where, limit)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	metrics := make([]RiskMetrics, 0, len(rows))
	for _, row := range rows {
		rm := RiskMetrics{
			ID:                  toInt64(row["id"]),
			ConstructionID:      toInt64(row["construction_id"]),
			ComputedAt:          toString(row["computed_at"]),
			PeriodDays:          toInt64(row["period_days"]),
			Sharpe:              toFloat64(row["sharpe"]),
			Sortino:             toFloat64(row["sortino"]),
			Calmar:              toFloat64(row["calmar"]),
			CAGR:                toFloat64(row["cagr"]),
			TotalReturn:         toFloat64(row["total_return"]),
			VolatilityAnn:       toFloat64(row["volatility_ann"]),
			MaxDrawdown:         toFloat64(row["max_drawdown"]),
			MaxDrawdownDuration: toInt64(row["max_drawdown_duration"]),
			VaR951D:             toFloat64(row["var_95_1d"]),
			VaR991D:             toFloat64(row["var_99_1d"]),
			CVaR951D:            toFloat64(row["cvar_95_1d"]),
			CVaR991D:            toFloat64(row["cvar_99_1d"]),
			VaRMethod:           toString(row["var_method"]),
			BetaToSPY:           toFloat64(row["beta_to_spy"]),
			AlphaAnn:            toFloat64(row["alpha_ann"]),
			InfoRatio:           toFloat64(row["info_ratio"]),
			Skewness:            toFloat64(row["skewness"]),
			Kurtosis:            toFloat64(row["kurtosis"]),
		}
		if raw := toString(row["factor_exposures"]); raw != "" && raw != "{}" {
			rm.FactorExposures = json.RawMessage(raw)
		}
		metrics = append(metrics, rm)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"risk_metrics": metrics,
		"count":        len(metrics),
	})
}

// GetCorrelation handles GET /api/v1/portfolio/correlation?construction_id=&latest=1
func (h *PortfolioHandler) GetCorrelation(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	if cid := q.Get("construction_id"); cid != "" {
		whereClauses = append(whereClauses, "cs.construction_id = ?")
		args = append(args, cid)
	}
	if method := q.Get("method"); method != "" {
		whereClauses = append(whereClauses, "cs.method = ?")
		args = append(args, method)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	limit := "20"
	if q.Get("latest") == "1" {
		limit = "1"
	}

	query := fmt.Sprintf(`
		SELECT
			cs.id,
			COALESCE(cs.construction_id, 0) AS construction_id,
			cs.computed_at,
			COALESCE(cs.universe, '')        AS universe,
			cs.n_assets, cs.lookback_days, cs.method,
			cs.symbols, cs.matrix,
			COALESCE(cs.avg_correlation, 0.0) AS avg_correlation,
			COALESCE(cs.max_correlation, 0.0) AS max_correlation,
			cs.is_positive_definite
		FROM correlation_snapshots cs
		%s
		ORDER BY cs.computed_at DESC
		LIMIT %s
	`, where, limit)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	snapshots := make([]CorrelationSnapshot, 0, len(rows))
	for _, row := range rows {
		snap := CorrelationSnapshot{
			ID:             toInt64(row["id"]),
			ConstructionID: toInt64(row["construction_id"]),
			ComputedAt:     toString(row["computed_at"]),
			Universe:       toString(row["universe"]),
			NAssets:        toInt64(row["n_assets"]),
			LookbackDays:   toInt64(row["lookback_days"]),
			Method:         toString(row["method"]),
			AvgCorrelation: toFloat64(row["avg_correlation"]),
			MaxCorrelation: toFloat64(row["max_correlation"]),
			IsPositiveDef:  toBool(row["is_positive_definite"]),
		}
		// Parse symbols array (stored as TEXT[] in Postgres but TEXT in SQLite).
		if raw := toString(row["symbols"]); raw != "" {
			// Try JSON array first, fall back to comma-separated.
			if err := json.Unmarshal([]byte(raw), &snap.Symbols); err != nil {
				snap.Symbols = strings.Split(strings.Trim(raw, "{}"), ",")
			}
		}
		if raw := toString(row["matrix"]); raw != "" {
			snap.Matrix = json.RawMessage(raw)
		}
		snapshots = append(snapshots, snap)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"correlation_snapshots": snapshots,
		"count":                 len(snapshots),
	})
}
