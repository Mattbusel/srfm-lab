// cmd/webhook/handlers/alpaca_fills.go -- Alpaca fill webhook handler.
//
// Receives POST /webhook/fills from Alpaca on every order fill event.
// Actions:
//   1. Parses and validates the fill payload.
//   2. Updates the SQLite positions table with weighted average cost.
//   3. Calculates realized P&L on reduce/close fills.
//   4. Writes the fill to the audit trail in SQLite.
//   5. Forwards a fill event to the Elixir EventBus (:8781).
//   6. On FULL_FILL: triggers position manager update.
//   7. On PARTIAL_FILL: logs partial state for order tracking.

package handlers

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Fill event types
// ---------------------------------------------------------------------------

// FillType distinguishes full and partial fills.
type FillType string

const (
	FillFull    FillType = "fill"
	FillPartial FillType = "partial_fill"
)

// FillSide is buy or sell.
type FillSide string

const (
	SideBuy  FillSide = "buy"
	SideSell FillSide = "sell"
)

// AlpacaFillEvent is the parsed representation of an Alpaca fill webhook.
type AlpacaFillEvent struct {
	EventType   string   `json:"event"`           // "fill" | "partial_fill"
	OrderID     string   `json:"order_id"`
	Symbol      string   `json:"symbol"`
	Qty         float64  `json:"-"`               // parsed from string field
	Price       float64  `json:"-"`               // parsed from string field
	Side        FillSide `json:"side"`
	Timestamp   time.Time `json:"timestamp"`
	OrderStatus string   `json:"order_status"`
	// String fields from Alpaca JSON.
	QtyStr       string `json:"qty"`
	FilledQtyStr string `json:"filled_qty"`
	FilledAvgStr string `json:"filled_avg_price"`
}

// fillType converts EventType to our FillType enum.
func (e *AlpacaFillEvent) fillType() FillType {
	switch strings.ToLower(e.EventType) {
	case "fill":
		return FillFull
	case "partial_fill":
		return FillPartial
	default:
		return FillFull
	}
}

// elixirFillEvent is the payload forwarded to the Elixir EventBus.
type elixirFillEvent struct {
	Source      string    `json:"source"`
	EventType   string    `json:"event_type"`
	OrderID     string    `json:"order_id"`
	Symbol      string    `json:"symbol"`
	Qty         float64   `json:"qty"`
	Price       float64   `json:"price"`
	Side        string    `json:"side"`
	FillType    string    `json:"fill_type"`
	RealizedPnL float64   `json:"realized_pnl"`
	Timestamp   time.Time `json:"timestamp"`
}

// ---------------------------------------------------------------------------
// AlpacaFillsHandler
// ---------------------------------------------------------------------------

// AlpacaFillsHandler processes Alpaca fill webhooks.
type AlpacaFillsHandler struct {
	db          *sql.DB
	elixirAddr  string
	alerterAddr string
	client      *http.Client
	logger      *slog.Logger
}

// NewAlpacaFillsHandler constructs an AlpacaFillsHandler.
func NewAlpacaFillsHandler(db *sql.DB, elixirAddr, alerterAddr string, logger *slog.Logger) *AlpacaFillsHandler {
	return &AlpacaFillsHandler{
		db:          db,
		elixirAddr:  elixirAddr,
		alerterAddr: alerterAddr,
		client:      &http.Client{Timeout: 10 * time.Second},
		logger:      logger,
	}
}

// HandleFill is the HTTP handler for POST /webhook/fills.
func (h *AlpacaFillsHandler) HandleFill(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20)) // 1 MB limit
	if err != nil {
		h.logger.Error("fill: read body", "err", err)
		http.Error(w, "read error", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	fill, err := h.parseFill(body)
	if err != nil {
		h.logger.Warn("fill: parse failed", "err", err)
		http.Error(w, "parse error: "+err.Error(), http.StatusBadRequest)
		return
	}

	h.logger.Info("fill received",
		"order_id", fill.OrderID,
		"symbol", fill.Symbol,
		"qty", fill.Qty,
		"price", fill.Price,
		"side", fill.Side,
		"type", fill.fillType(),
	)

	// Calculate realized P&L then update position.
	realizedPnL, err := h.processPosition(r.Context(), fill)
	if err != nil {
		h.logger.Error("fill: position update", "err", err, "order_id", fill.OrderID)
		// Continue -- don't block acknowledgement.
	}

	// Write audit trail.
	if err := h.writeFillAudit(r.Context(), fill, realizedPnL, string(body)); err != nil {
		h.logger.Error("fill: audit write", "err", err)
	}

	// Forward to Elixir EventBus.
	go h.forwardToElixir(context.Background(), fill, realizedPnL)

	// Acknowledge immediately; downstream work is async.
	w.WriteHeader(http.StatusNoContent)
}

// parseFill deserializes and normalizes an Alpaca fill JSON payload.
func (h *AlpacaFillsHandler) parseFill(data []byte) (*AlpacaFillEvent, error) {
	var ev AlpacaFillEvent
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}

	if ev.OrderID == "" {
		return nil, fmt.Errorf("missing order_id")
	}
	if ev.Symbol == "" {
		return nil, fmt.Errorf("missing symbol")
	}

	// Alpaca sends numeric fields as strings.
	if ev.FilledAvgStr != "" {
		if _, err := fmt.Sscanf(ev.FilledAvgStr, "%f", &ev.Price); err != nil {
			return nil, fmt.Errorf("parse filled_avg_price %q: %w", ev.FilledAvgStr, err)
		}
	}
	if ev.FilledQtyStr != "" {
		if _, err := fmt.Sscanf(ev.FilledQtyStr, "%f", &ev.Qty); err != nil {
			return nil, fmt.Errorf("parse filled_qty %q: %w", ev.FilledQtyStr, err)
		}
	} else if ev.QtyStr != "" {
		if _, err := fmt.Sscanf(ev.QtyStr, "%f", &ev.Qty); err != nil {
			return nil, fmt.Errorf("parse qty %q: %w", ev.QtyStr, err)
		}
	}

	if ev.Qty <= 0 {
		return nil, fmt.Errorf("qty must be positive, got %f", ev.Qty)
	}
	if ev.Price <= 0 {
		return nil, fmt.Errorf("price must be positive, got %f", ev.Price)
	}
	if ev.Timestamp.IsZero() {
		ev.Timestamp = time.Now().UTC()
	}

	return &ev, nil
}

// processPosition updates the positions table and returns realized P&L.
func (h *AlpacaFillsHandler) processPosition(ctx context.Context, fill *AlpacaFillEvent) (float64, error) {
	tx, err := h.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	// Fetch current position.
	var (
		currentQty      float64
		avgCost         float64
		currentSide     string
		realizedPnL     float64
	)
	row := tx.QueryRowContext(ctx,
		`SELECT qty, avg_cost, side FROM positions WHERE symbol = ?`, fill.Symbol)
	err = row.Scan(&currentQty, &avgCost, &currentSide)
	if err != nil && err != sql.ErrNoRows {
		return 0, fmt.Errorf("query position: %w", err)
	}

	var newQty, newAvgCost float64
	var newSide string

	switch fill.Side {
	case SideBuy:
		if currentSide == "short" && currentQty > 0 {
			// Closing / reducing short position.
			realizedPnL = (avgCost - fill.Price) * math.Min(fill.Qty, currentQty)
			remainQty := currentQty - fill.Qty
			if remainQty > 0 {
				newQty = remainQty
				newAvgCost = avgCost
				newSide = "short"
			} else {
				// Flipped to long.
				newQty = fill.Qty - currentQty
				if newQty > 0 {
					newAvgCost = fill.Price
					newSide = "long"
				} else {
					newQty = 0
					newAvgCost = 0
					newSide = "flat"
				}
			}
		} else {
			// Adding to or initiating long position.
			totalCost := avgCost*currentQty + fill.Price*fill.Qty
			newQty = currentQty + fill.Qty
			newAvgCost = totalCost / newQty
			newSide = "long"
		}

	case SideSell:
		if currentSide == "long" && currentQty > 0 {
			// Closing / reducing long position.
			realizedPnL = (fill.Price - avgCost) * math.Min(fill.Qty, currentQty)
			remainQty := currentQty - fill.Qty
			if remainQty > 0 {
				newQty = remainQty
				newAvgCost = avgCost
				newSide = "long"
			} else {
				// Flipped to short.
				newQty = fill.Qty - currentQty
				if newQty > 0 {
					newAvgCost = fill.Price
					newSide = "short"
				} else {
					newQty = 0
					newAvgCost = 0
					newSide = "flat"
				}
			}
		} else {
			// Adding to or initiating short position.
			totalCost := avgCost*currentQty + fill.Price*fill.Qty
			newQty = currentQty + fill.Qty
			newAvgCost = totalCost / newQty
			newSide = "short"
		}
	}

	// Upsert position.
	_, err = tx.ExecContext(ctx,
		`INSERT INTO positions (symbol, qty, avg_cost, side, realized_pnl, updated_at)
		 VALUES (?, ?, ?, ?, ?, datetime('now'))
		 ON CONFLICT(symbol) DO UPDATE SET
		   qty          = excluded.qty,
		   avg_cost     = excluded.avg_cost,
		   side         = excluded.side,
		   realized_pnl = realized_pnl + excluded.realized_pnl,
		   updated_at   = excluded.updated_at`,
		fill.Symbol, newQty, newAvgCost, newSide, realizedPnL,
	)
	if err != nil {
		return 0, fmt.Errorf("upsert position: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("commit: %w", err)
	}

	h.logger.Info("position updated",
		"symbol", fill.Symbol,
		"new_qty", newQty,
		"new_avg_cost", newAvgCost,
		"new_side", newSide,
		"realized_pnl", realizedPnL,
	)
	return realizedPnL, nil
}

// writeFillAudit persists a fill record to the audit trail.
func (h *AlpacaFillsHandler) writeFillAudit(ctx context.Context, fill *AlpacaFillEvent, realizedPnL float64, raw string) error {
	_, err := h.db.ExecContext(ctx,
		`INSERT INTO fills (order_id, symbol, qty, price, side, fill_type, realized_pnl, timestamp, raw_payload)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		fill.OrderID,
		fill.Symbol,
		fill.Qty,
		fill.Price,
		string(fill.Side),
		string(fill.fillType()),
		realizedPnL,
		fill.Timestamp.Format(time.RFC3339),
		raw,
	)
	if err != nil {
		return fmt.Errorf("insert fill audit: %w", err)
	}
	return nil
}

// forwardToElixir sends a fill event to the Elixir EventBus.
func (h *AlpacaFillsHandler) forwardToElixir(ctx context.Context, fill *AlpacaFillEvent, realizedPnL float64) {
	event := elixirFillEvent{
		Source:      "alpaca-webhook",
		EventType:   "fill",
		OrderID:     fill.OrderID,
		Symbol:      fill.Symbol,
		Qty:         fill.Qty,
		Price:       fill.Price,
		Side:        string(fill.Side),
		FillType:    string(fill.fillType()),
		RealizedPnL: realizedPnL,
		Timestamp:   fill.Timestamp,
	}

	body, err := json.Marshal(event)
	if err != nil {
		h.logger.Error("elixir forward: marshal", "err", err)
		return
	}

	url := h.elixirAddr + "/fills"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		h.logger.Error("elixir forward: build request", "err", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-SRFM-Source", "webhook-service")

	resp, err := h.client.Do(req)
	if err != nil {
		h.logger.Warn("elixir forward: request failed", "err", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		h.logger.Warn("elixir forward: non-2xx", "status", resp.StatusCode)
		return
	}

	h.logger.Debug("fill forwarded to elixir",
		"order_id", fill.OrderID, "symbol", fill.Symbol)

	// For full fills: also notify the performance tracker endpoint.
	if fill.fillType() == FillFull {
		h.notifyPerformanceTracker(ctx, fill, realizedPnL)
	}
}

// notifyPerformanceTracker sends a fill notification to the performance tracker.
func (h *AlpacaFillsHandler) notifyPerformanceTracker(ctx context.Context, fill *AlpacaFillEvent, realizedPnL float64) {
	payload := map[string]any{
		"event":        "full_fill",
		"order_id":     fill.OrderID,
		"symbol":       fill.Symbol,
		"qty":          fill.Qty,
		"price":        fill.Price,
		"side":         string(fill.Side),
		"realized_pnl": realizedPnL,
		"timestamp":    fill.Timestamp.Format(time.RFC3339),
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return
	}

	url := h.elixirAddr + "/performance/fill"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url,
		bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.client.Do(req)
	if err != nil {
		h.logger.Debug("performance tracker unreachable", "err", err)
		return
	}
	defer resp.Body.Close()
}
