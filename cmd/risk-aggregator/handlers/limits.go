// Package handlers provides HTTP and WebSocket handlers for the risk-aggregator service.
// This file implements CRUD endpoints for position/risk limits with SQLite persistence.
package handlers

import (
	"database/sql"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	_ "github.com/mattn/go-sqlite3"
	"github.com/rs/zerolog/log"
)

// Limit defines the risk thresholds for a single symbol (or the portfolio
// when Symbol == "__portfolio__").
type Limit struct {
	Symbol           string    `json:"symbol" binding:"required"`
	MaxPositionUSD   float64   `json:"max_position_usd"`
	MaxDailyLossBps  float64   `json:"max_daily_loss_bps"`
	MaxDrawdownPct   float64   `json:"max_drawdown_pct"`
	SectorLimit      float64   `json:"sector_limit_usd"`
	UpdatedAt        time.Time `json:"updated_at"`
}

// LimitBreach describes a limit that has been violated or is close to violation.
type LimitBreach struct {
	Symbol       string  `json:"symbol"`
	LimitType    string  `json:"limit_type"`
	Message      string  `json:"message"`
	CurrentValue float64 `json:"current_value"`
	LimitValue   float64 `json:"limit_value"`
	// Severity is "warning" (>80% of limit) or "critical" (>100% of limit).
	Severity string `json:"severity"`
}

// OrderCheckRequest is the body for POST /limits/check.
type OrderCheckRequest struct {
	Symbol        string  `json:"symbol" binding:"required"`
	Quantity      float64 `json:"quantity" binding:"required"`
	Price         float64 `json:"price" binding:"required"`
	CurrentPosUSD float64 `json:"current_position_usd"`
}

// OrderCheckResponse reports whether the hypothetical order would breach limits.
type OrderCheckResponse struct {
	Symbol   string        `json:"symbol"`
	Allowed  bool          `json:"allowed"`
	Breaches []LimitBreach `json:"breaches,omitempty"`
}

// PositionProvider is a narrow interface for fetching current exposure data
// so that LimitsHandler can be tested without the full aggregator.
type PositionProvider interface {
	CurrentPositions() (map[string]float64, error) // symbol -> USD value
	DailyPnLBps() (map[string]float64, error)      // symbol -> bps P&L today
	DrawdownPct() (map[string]float64, error)       // symbol -> drawdown %
}

// LimitsHandler manages per-symbol and portfolio-level risk limits.
// Limits are persisted in a local SQLite database so they survive restarts.
type LimitsHandler struct {
	db       *sql.DB
	provider PositionProvider
}

// NewLimitsHandler opens (or creates) the SQLite database at dbPath and
// returns a ready-to-use handler.
func NewLimitsHandler(dbPath string, provider PositionProvider) (*LimitsHandler, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	h := &LimitsHandler{db: db, provider: provider}
	if err := h.migrate(); err != nil {
		db.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return h, nil
}

// Close releases the underlying database connection.
func (h *LimitsHandler) Close() error {
	return h.db.Close()
}

// migrate ensures the schema is up to date.
func (h *LimitsHandler) migrate() error {
	const ddl = `
CREATE TABLE IF NOT EXISTS limits (
    symbol            TEXT PRIMARY KEY,
    max_position_usd  REAL NOT NULL DEFAULT 0,
    max_daily_loss_bps REAL NOT NULL DEFAULT 0,
    max_drawdown_pct  REAL NOT NULL DEFAULT 0,
    sector_limit_usd  REAL NOT NULL DEFAULT 0,
    updated_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);`
	_, err := h.db.Exec(ddl)
	return err
}

// RegisterRoutes attaches all limit routes to a gin RouterGroup.
// Example:
//
//	h.RegisterRoutes(r.Group("/limits"))
func (h *LimitsHandler) RegisterRoutes(rg *gin.RouterGroup) {
	rg.GET("", h.ListLimits)
	rg.GET("/breaches", h.GetBreaches)
	rg.POST("/check", h.CheckOrder)
	rg.GET("/:symbol", h.GetLimit)
	rg.POST("/:symbol", h.UpsertLimit)
	rg.DELETE("/:symbol", h.DeleteLimit)
}

// ListLimits handles GET /limits -- returns all configured limits.
func (h *LimitsHandler) ListLimits(c *gin.Context) {
	limits, err := h.fetchAllLimits()
	if err != nil {
		log.Error().Err(err).Msg("list limits")
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"limits": limits, "count": len(limits)})
}

// GetLimit handles GET /limits/:symbol -- returns limits for one symbol.
func (h *LimitsHandler) GetLimit(c *gin.Context) {
	symbol := strings.ToUpper(c.Param("symbol"))
	lim, err := h.fetchLimit(symbol)
	if err == sql.ErrNoRows {
		c.JSON(http.StatusNotFound, gin.H{"error": "no limit configured for " + symbol})
		return
	}
	if err != nil {
		log.Error().Err(err).Str("symbol", symbol).Msg("fetch limit")
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, lim)
}

// UpsertLimit handles POST /limits/:symbol -- creates or updates a limit.
func (h *LimitsHandler) UpsertLimit(c *gin.Context) {
	symbol := strings.ToUpper(c.Param("symbol"))
	var body Limit
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	body.Symbol = symbol

	if err := h.upsert(body); err != nil {
		log.Error().Err(err).Str("symbol", symbol).Msg("upsert limit")
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	log.Info().Str("symbol", symbol).
		Float64("max_pos_usd", body.MaxPositionUSD).
		Msg("limit upserted")
	c.JSON(http.StatusOK, gin.H{"status": "ok", "symbol": symbol})
}

// DeleteLimit handles DELETE /limits/:symbol -- removes a symbol-specific limit.
func (h *LimitsHandler) DeleteLimit(c *gin.Context) {
	symbol := strings.ToUpper(c.Param("symbol"))
	result, err := h.db.Exec(`DELETE FROM limits WHERE symbol = ?`, symbol)
	if err != nil {
		log.Error().Err(err).Str("symbol", symbol).Msg("delete limit")
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	rows, _ := result.RowsAffected()
	if rows == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "limit not found for " + symbol})
		return
	}
	c.JSON(http.StatusOK, gin.H{"status": "deleted", "symbol": symbol})
}

// GetBreaches handles GET /limits/breaches -- returns all currently active breaches.
func (h *LimitsHandler) GetBreaches(c *gin.Context) {
	breaches, err := h.computeBreaches()
	if err != nil {
		log.Error().Err(err).Msg("compute breaches")
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"breaches": breaches, "count": len(breaches)})
}

// CheckOrder handles POST /limits/check -- evaluates a hypothetical order
// against configured limits without executing it.
func (h *LimitsHandler) CheckOrder(c *gin.Context) {
	var req OrderCheckRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	req.Symbol = strings.ToUpper(req.Symbol)

	lim, err := h.fetchLimit(req.Symbol)
	if err == sql.ErrNoRows {
		// No limit configured -- order is implicitly allowed.
		c.JSON(http.StatusOK, OrderCheckResponse{Symbol: req.Symbol, Allowed: true})
		return
	}
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	orderUSD := req.Quantity * req.Price
	newPositionUSD := req.CurrentPosUSD + orderUSD

	var breaches []LimitBreach

	if lim.MaxPositionUSD > 0 {
		if newPositionUSD > lim.MaxPositionUSD {
			breaches = append(breaches, LimitBreach{
				Symbol:       req.Symbol,
				LimitType:    "max_position_usd",
				Message:      fmt.Sprintf("order would push position to $%.2f, limit is $%.2f", newPositionUSD, lim.MaxPositionUSD),
				CurrentValue: newPositionUSD,
				LimitValue:   lim.MaxPositionUSD,
				Severity:     "critical",
			})
		} else if newPositionUSD > lim.MaxPositionUSD*0.8 {
			breaches = append(breaches, LimitBreach{
				Symbol:       req.Symbol,
				LimitType:    "max_position_usd",
				Message:      fmt.Sprintf("order would push position to $%.2f, >80%% of limit $%.2f", newPositionUSD, lim.MaxPositionUSD),
				CurrentValue: newPositionUSD,
				LimitValue:   lim.MaxPositionUSD,
				Severity:     "warning",
			})
		}
	}

	allowed := true
	for _, b := range breaches {
		if b.Severity == "critical" {
			allowed = false
			break
		}
	}

	c.JSON(http.StatusOK, OrderCheckResponse{
		Symbol:   req.Symbol,
		Allowed:  allowed,
		Breaches: breaches,
	})
}

// -- internal helpers --

func (h *LimitsHandler) fetchAllLimits() ([]Limit, error) {
	rows, err := h.db.Query(`
		SELECT symbol, max_position_usd, max_daily_loss_bps,
		       max_drawdown_pct, sector_limit_usd, updated_at
		FROM limits ORDER BY symbol`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Limit
	for rows.Next() {
		var l Limit
		if err := rows.Scan(&l.Symbol, &l.MaxPositionUSD, &l.MaxDailyLossBps,
			&l.MaxDrawdownPct, &l.SectorLimit, &l.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, l)
	}
	return out, rows.Err()
}

func (h *LimitsHandler) fetchLimit(symbol string) (Limit, error) {
	var l Limit
	err := h.db.QueryRow(`
		SELECT symbol, max_position_usd, max_daily_loss_bps,
		       max_drawdown_pct, sector_limit_usd, updated_at
		FROM limits WHERE symbol = ?`, symbol).
		Scan(&l.Symbol, &l.MaxPositionUSD, &l.MaxDailyLossBps,
			&l.MaxDrawdownPct, &l.SectorLimit, &l.UpdatedAt)
	return l, err
}

func (h *LimitsHandler) upsert(l Limit) error {
	_, err := h.db.Exec(`
		INSERT INTO limits (symbol, max_position_usd, max_daily_loss_bps,
		                    max_drawdown_pct, sector_limit_usd, updated_at)
		VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(symbol) DO UPDATE SET
		    max_position_usd   = excluded.max_position_usd,
		    max_daily_loss_bps = excluded.max_daily_loss_bps,
		    max_drawdown_pct   = excluded.max_drawdown_pct,
		    sector_limit_usd   = excluded.sector_limit_usd,
		    updated_at         = CURRENT_TIMESTAMP`,
		l.Symbol, l.MaxPositionUSD, l.MaxDailyLossBps,
		l.MaxDrawdownPct, l.SectorLimit)
	return err
}

// computeBreaches fetches live positions and compares against stored limits.
func (h *LimitsHandler) computeBreaches() ([]LimitBreach, error) {
	if h.provider == nil {
		return nil, nil
	}

	positions, err := h.provider.CurrentPositions()
	if err != nil {
		return nil, fmt.Errorf("fetch positions: %w", err)
	}
	dailyBps, err := h.provider.DailyPnLBps()
	if err != nil {
		return nil, fmt.Errorf("fetch daily pnl: %w", err)
	}
	drawdowns, err := h.provider.DrawdownPct()
	if err != nil {
		return nil, fmt.Errorf("fetch drawdowns: %w", err)
	}

	limits, err := h.fetchAllLimits()
	if err != nil {
		return nil, err
	}

	var breaches []LimitBreach

	for _, lim := range limits {
		sym := lim.Symbol

		// Position size check.
		if lim.MaxPositionUSD > 0 {
			pos := positions[sym]
			if pos > lim.MaxPositionUSD {
				breaches = append(breaches, LimitBreach{
					Symbol:       sym,
					LimitType:    "max_position_usd",
					Message:      fmt.Sprintf("position $%.2f exceeds limit $%.2f", pos, lim.MaxPositionUSD),
					CurrentValue: pos,
					LimitValue:   lim.MaxPositionUSD,
					Severity:     severityFor(pos, lim.MaxPositionUSD),
				})
			}
		}

		// Daily loss check.
		if lim.MaxDailyLossBps > 0 {
			bps := dailyBps[sym]
			// Loss is negative bps; compare absolute value.
			if bps < 0 && -bps > lim.MaxDailyLossBps {
				breaches = append(breaches, LimitBreach{
					Symbol:       sym,
					LimitType:    "max_daily_loss_bps",
					Message:      fmt.Sprintf("daily loss %.1f bps exceeds limit %.1f bps", -bps, lim.MaxDailyLossBps),
					CurrentValue: -bps,
					LimitValue:   lim.MaxDailyLossBps,
					Severity:     severityFor(-bps, lim.MaxDailyLossBps),
				})
			}
		}

		// Drawdown check.
		if lim.MaxDrawdownPct > 0 {
			dd := drawdowns[sym]
			if dd > lim.MaxDrawdownPct {
				breaches = append(breaches, LimitBreach{
					Symbol:       sym,
					LimitType:    "max_drawdown_pct",
					Message:      fmt.Sprintf("drawdown %.2f%% exceeds limit %.2f%%", dd, lim.MaxDrawdownPct),
					CurrentValue: dd,
					LimitValue:   lim.MaxDrawdownPct,
					Severity:     severityFor(dd, lim.MaxDrawdownPct),
				})
			}
		}
	}

	return breaches, nil
}

// severityFor returns "warning" when current is 80-100% of limit,
// "critical" when current exceeds limit.
func severityFor(current, limit float64) string {
	if limit <= 0 {
		return "warning"
	}
	ratio := current / limit
	if ratio >= 1.0 {
		return "critical"
	}
	if ratio >= 0.8 {
		return "warning"
	}
	return "ok"
}
