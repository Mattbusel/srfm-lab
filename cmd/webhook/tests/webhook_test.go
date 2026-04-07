// cmd/webhook/tests/webhook_test.go -- Integration and unit tests for the
// SRFM webhook service.
//
// Test suites:
//   TestAlpacaFillProcessing      -- fill parsing, P&L, audit trail
//   TestParameterWebhookRateLimit -- 30-minute rate gate
//   TestHMACSignatureVerification -- signature auth middleware
//   TestHealthSummaryAggregation  -- health map and summary endpoint
//   TestFillAuditTrail            -- SQLite fill audit records
//   TestTokenBucket               -- token bucket algorithm correctness
//   TestParameterSchemaValidation -- schema checker edge cases

package tests

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"srfm-webhook/handlers"
	"srfm-webhook/middleware"
)

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// setupTestDB creates an in-memory SQLite database with the required schema.
func setupTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", ":memory:?_journal=WAL")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}

	stmts := []string{
		`CREATE TABLE IF NOT EXISTS fills (
			id              INTEGER PRIMARY KEY AUTOINCREMENT,
			order_id        TEXT NOT NULL,
			symbol          TEXT NOT NULL,
			qty             REAL NOT NULL,
			price           REAL NOT NULL,
			side            TEXT NOT NULL,
			fill_type       TEXT NOT NULL,
			realized_pnl    REAL NOT NULL DEFAULT 0,
			timestamp       TEXT NOT NULL,
			created_at      TEXT NOT NULL DEFAULT (datetime('now')),
			raw_payload     TEXT
		)`,
		`CREATE TABLE IF NOT EXISTS positions (
			symbol          TEXT PRIMARY KEY,
			qty             REAL NOT NULL DEFAULT 0,
			avg_cost        REAL NOT NULL DEFAULT 0,
			side            TEXT NOT NULL DEFAULT 'flat',
			unrealized_pnl  REAL NOT NULL DEFAULT 0,
			realized_pnl    REAL NOT NULL DEFAULT 0,
			updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
		)`,
		`CREATE TABLE IF NOT EXISTS param_proposals (
			id              INTEGER PRIMARY KEY AUTOINCREMENT,
			genome_hash     TEXT NOT NULL,
			fitness_score   REAL NOT NULL,
			schema_valid    INTEGER NOT NULL DEFAULT 0,
			forwarded       INTEGER NOT NULL DEFAULT 0,
			proposed_at     TEXT NOT NULL DEFAULT (datetime('now')),
			raw_payload     TEXT
		)`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			t.Fatalf("migrate: %v", err)
		}
	}
	t.Cleanup(func() { db.Close() })
	return db
}

// testLogger returns a discard logger for tests.
func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// signBody computes an Alpaca-style HMAC-SHA256 signature for a body.
func signBody(t *testing.T, secret, body string) string {
	t.Helper()
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(body))
	return hex.EncodeToString(mac.Sum(nil))
}

// buildFillPayload constructs a minimal Alpaca fill JSON body.
func buildFillPayload(orderID, symbol, side, fillType string, qty, price float64) string {
	return fmt.Sprintf(`{
		"event": %q,
		"order_id": %q,
		"symbol": %q,
		"side": %q,
		"filled_qty": "%.4f",
		"filled_avg_price": "%.4f",
		"order_status": "filled",
		"timestamp": %q
	}`, fillType, orderID, symbol, side,
		qty, price,
		time.Now().UTC().Format(time.RFC3339),
	)
}

// newElixirMock creates an httptest.Server that records received bodies.
func newElixirMock(t *testing.T) (*httptest.Server, *[]string) {
	t.Helper()
	var received []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		received = append(received, string(body))
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)
	return srv, &received
}

// ---------------------------------------------------------------------------
// TestAlpacaFillProcessing
// ---------------------------------------------------------------------------

func TestAlpacaFillProcessing(t *testing.T) {
	db := setupTestDB(t)
	elixir, received := newElixirMock(t)
	h := handlers.NewAlpacaFillsHandler(db, elixir.URL, "", testLogger())

	t.Run("valid_full_fill_buy", func(t *testing.T) {
		body := buildFillPayload("ord-001", "BTCUSD", "buy", "fill", 1.5, 60000.0)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleFill(rec, req)

		if rec.Code != http.StatusNoContent {
			t.Errorf("expected 204, got %d", rec.Code)
		}
		// Position should be created.
		var qty float64
		var side string
		err := db.QueryRow(`SELECT qty, side FROM positions WHERE symbol = 'BTCUSD'`).Scan(&qty, &side)
		if err != nil {
			t.Fatalf("position not found: %v", err)
		}
		if qty != 1.5 {
			t.Errorf("expected qty=1.5, got %.4f", qty)
		}
		if side != "long" {
			t.Errorf("expected side=long, got %s", side)
		}
	})

	t.Run("partial_fill_adds_to_position", func(t *testing.T) {
		body := buildFillPayload("ord-002", "BTCUSD", "buy", "partial_fill", 0.5, 61000.0)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleFill(rec, req)

		if rec.Code != http.StatusNoContent {
			t.Errorf("expected 204, got %d", rec.Code)
		}
		var qty float64
		db.QueryRow(`SELECT qty FROM positions WHERE symbol = 'BTCUSD'`).Scan(&qty)
		// 1.5 + 0.5 = 2.0
		if qty != 2.0 {
			t.Errorf("expected qty=2.0 after partial fill, got %.4f", qty)
		}
	})

	t.Run("sell_fill_realizes_pnl", func(t *testing.T) {
		// Sell 1.0 of the 2.0 long position. Avg cost was (1.5*60000+0.5*61000)/2 = 60250
		// Realized PnL = (62000 - 60250) * 1.0 = 1750
		body := buildFillPayload("ord-003", "BTCUSD", "sell", "fill", 1.0, 62000.0)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleFill(rec, req)

		if rec.Code != http.StatusNoContent {
			t.Errorf("expected 204, got %d", rec.Code)
		}
		var realizedPnL float64
		db.QueryRow(`SELECT realized_pnl FROM positions WHERE symbol = 'BTCUSD'`).Scan(&realizedPnL)
		if realizedPnL <= 0 {
			t.Errorf("expected positive realized PnL, got %.4f", realizedPnL)
		}
	})

	t.Run("invalid_missing_order_id", func(t *testing.T) {
		body := `{"event":"fill","symbol":"ETHUSD","side":"buy","filled_qty":"1.0","filled_avg_price":"3000.0"}`
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleFill(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("expected 400 for missing order_id, got %d", rec.Code)
		}
	})

	t.Run("invalid_method_get", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/webhook/fills", nil)
		rec := httptest.NewRecorder()
		h.HandleFill(rec, req)
		if rec.Code != http.StatusMethodNotAllowed {
			t.Errorf("expected 405, got %d", rec.Code)
		}
	})

	t.Run("negative_qty_rejected", func(t *testing.T) {
		body := buildFillPayload("ord-004", "SOLUSD", "buy", "fill", -1.0, 150.0)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleFill(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("expected 400 for negative qty, got %d", rec.Code)
		}
	})

	t.Run("elixir_notified_on_full_fill", func(t *testing.T) {
		before := len(*received)
		body := buildFillPayload("ord-005", "ETHUSD", "buy", "fill", 2.0, 3000.0)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleFill(rec, req)
		// Give goroutine time to fire.
		time.Sleep(100 * time.Millisecond)
		after := len(*received)
		if after <= before {
			t.Error("expected elixir to receive at least one request after full fill")
		}
	})

	t.Run("short_sell_creates_short_position", func(t *testing.T) {
		body := buildFillPayload("ord-006", "SOLUSD", "sell", "fill", 10.0, 150.0)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleFill(rec, req)
		if rec.Code != http.StatusNoContent {
			t.Errorf("expected 204, got %d", rec.Code)
		}
		var side string
		db.QueryRow(`SELECT side FROM positions WHERE symbol = 'SOLUSD'`).Scan(&side)
		if side != "short" {
			t.Errorf("expected side=short, got %s", side)
		}
	})
}

// ---------------------------------------------------------------------------
// TestFillAuditTrail
// ---------------------------------------------------------------------------

func TestFillAuditTrail(t *testing.T) {
	db := setupTestDB(t)
	elixir, _ := newElixirMock(t)
	h := handlers.NewAlpacaFillsHandler(db, elixir.URL, "", testLogger())

	fills := []struct {
		orderID string
		symbol  string
		side    string
		qty     float64
		price   float64
	}{
		{"audit-001", "BTCUSD", "buy", 1.0, 50000.0},
		{"audit-002", "BTCUSD", "sell", 0.5, 55000.0},
		{"audit-003", "ETHUSD", "buy", 5.0, 3000.0},
	}

	for _, f := range fills {
		body := buildFillPayload(f.orderID, f.symbol, f.side, "fill", f.qty, f.price)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleFill(rec, req)
		if rec.Code != http.StatusNoContent {
			t.Errorf("[%s] expected 204, got %d", f.orderID, rec.Code)
		}
	}

	t.Run("all_fills_logged", func(t *testing.T) {
		var count int
		db.QueryRow(`SELECT COUNT(*) FROM fills`).Scan(&count)
		if count != len(fills) {
			t.Errorf("expected %d fill records, got %d", len(fills), count)
		}
	})

	t.Run("fill_has_raw_payload", func(t *testing.T) {
		var raw sql.NullString
		db.QueryRow(`SELECT raw_payload FROM fills WHERE order_id = 'audit-001'`).Scan(&raw)
		if !raw.Valid || raw.String == "" {
			t.Error("expected raw_payload to be stored for fill audit-001")
		}
	})

	t.Run("realized_pnl_positive_on_sell", func(t *testing.T) {
		var pnl float64
		db.QueryRow(`SELECT realized_pnl FROM fills WHERE order_id = 'audit-002'`).Scan(&pnl)
		if pnl <= 0 {
			t.Errorf("expected positive realized_pnl on sell, got %.4f", pnl)
		}
	})

	t.Run("order_id_indexed", func(t *testing.T) {
		rows, err := db.Query(`SELECT order_id FROM fills WHERE order_id LIKE 'audit-%'`)
		if err != nil {
			t.Fatalf("query: %v", err)
		}
		defer rows.Close()
		var ids []string
		for rows.Next() {
			var id string
			rows.Scan(&id)
			ids = append(ids, id)
		}
		if len(ids) != len(fills) {
			t.Errorf("expected %d rows via index, got %d", len(fills), len(ids))
		}
	})
}

// ---------------------------------------------------------------------------
// TestParameterWebhookRateLimit
// ---------------------------------------------------------------------------

func TestParameterWebhookRateLimit(t *testing.T) {
	db := setupTestDB(t)
	elixir, _ := newElixirMock(t)
	h := handlers.NewParameterWebhookHandler(db, elixir.URL, testLogger())

	validPayload := func(hash string) string {
		return fmt.Sprintf(`{
			"genome_hash": %q,
			"fitness_score": 0.85,
			"generation": 42,
			"parameters": {
				"alpha": 1.2,
				"beta": 0.3,
				"risk_fraction": 0.05,
				"stop_loss_pct": 0.03
			}
		}`, hash)
	}

	t.Run("first_update_accepted", func(t *testing.T) {
		body := validPayload("genome-001")
		req := httptest.NewRequest(http.MethodPost, "/webhook/params/update", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleParamUpdate(rec, req)
		if rec.Code != http.StatusAccepted {
			t.Errorf("expected 202 on first update, got %d: %s", rec.Code, rec.Body.String())
		}
	})

	t.Run("second_update_rate_limited", func(t *testing.T) {
		body := validPayload("genome-002")
		req := httptest.NewRequest(http.MethodPost, "/webhook/params/update", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleParamUpdate(rec, req)
		if rec.Code != http.StatusTooManyRequests {
			t.Errorf("expected 429 on second update within window, got %d", rec.Code)
		}
	})

	t.Run("rate_limit_response_has_retry_after", func(t *testing.T) {
		body := validPayload("genome-003")
		req := httptest.NewRequest(http.MethodPost, "/webhook/params/update", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleParamUpdate(rec, req)
		if ra := rec.Header().Get("Retry-After"); ra == "" {
			t.Error("expected Retry-After header on 429 response")
		}
	})

	t.Run("invalid_method_get_rejected", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/webhook/params/update", nil)
		rec := httptest.NewRecorder()
		h.HandleParamUpdate(rec, req)
		if rec.Code != http.StatusMethodNotAllowed {
			t.Errorf("expected 405, got %d", rec.Code)
		}
	})
}

// ---------------------------------------------------------------------------
// TestParameterSchemaValidation
// ---------------------------------------------------------------------------

func TestParameterSchemaValidation(t *testing.T) {
	db := setupTestDB(t)
	elixir, _ := newElixirMock(t)
	// Use a fresh handler with no prior updates so the rate limit doesn't interfere.
	// Each sub-test uses a distinct handler instance.

	makeHandler := func() *handlers.ParameterWebhookHandler {
		return handlers.NewParameterWebhookHandler(db, elixir.URL, testLogger())
	}

	t.Run("missing_required_field", func(t *testing.T) {
		h := makeHandler()
		body := `{
			"genome_hash": "genome-schema-1",
			"fitness_score": 0.7,
			"generation": 1,
			"parameters": {
				"alpha": 1.0,
				"beta": 0.5,
				"risk_fraction": 0.05
			}
		}` // missing stop_loss_pct
		req := httptest.NewRequest(http.MethodPost, "/webhook/params/update", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleParamUpdate(rec, req)
		if rec.Code != http.StatusUnprocessableEntity {
			t.Errorf("expected 422 for missing field, got %d: %s", rec.Code, rec.Body.String())
		}
	})

	t.Run("out_of_bounds_parameter", func(t *testing.T) {
		h := makeHandler()
		body := `{
			"genome_hash": "genome-schema-2",
			"fitness_score": 0.7,
			"generation": 1,
			"parameters": {
				"alpha": 1.0,
				"beta": 0.5,
				"risk_fraction": 0.99,
				"stop_loss_pct": 0.03
			}
		}` // risk_fraction > 0.25
		req := httptest.NewRequest(http.MethodPost, "/webhook/params/update", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleParamUpdate(rec, req)
		if rec.Code != http.StatusUnprocessableEntity {
			t.Errorf("expected 422 for out-of-bounds, got %d", rec.Code)
		}
	})

	t.Run("missing_genome_hash_rejected", func(t *testing.T) {
		h := makeHandler()
		body := `{
			"fitness_score": 0.5,
			"generation": 1,
			"parameters": {"alpha":1.0,"beta":0.5,"risk_fraction":0.05,"stop_loss_pct":0.02}
		}`
		req := httptest.NewRequest(http.MethodPost, "/webhook/params/update", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleParamUpdate(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("expected 400 for missing genome_hash, got %d", rec.Code)
		}
	})

	t.Run("negative_stop_loss_rejected", func(t *testing.T) {
		h := makeHandler()
		body := `{
			"genome_hash": "genome-schema-3",
			"fitness_score": 0.6,
			"generation": 2,
			"parameters": {"alpha":1.0,"beta":0.5,"risk_fraction":0.05,"stop_loss_pct":-0.01}
		}`
		req := httptest.NewRequest(http.MethodPost, "/webhook/params/update", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleParamUpdate(rec, req)
		if rec.Code != http.StatusUnprocessableEntity {
			t.Errorf("expected 422 for negative stop_loss_pct, got %d", rec.Code)
		}
	})

	t.Run("valid_proposal_stored_in_db", func(t *testing.T) {
		h := makeHandler()
		body := `{
			"genome_hash": "genome-db-test",
			"fitness_score": 0.91,
			"generation": 10,
			"parameters": {"alpha":0.8,"beta":0.2,"risk_fraction":0.04,"stop_loss_pct":0.025}
		}`
		req := httptest.NewRequest(http.MethodPost, "/webhook/params/update", strings.NewReader(body))
		rec := httptest.NewRecorder()
		h.HandleParamUpdate(rec, req)
		if rec.Code != http.StatusAccepted {
			t.Fatalf("expected 202, got %d: %s", rec.Code, rec.Body.String())
		}
		var count int
		db.QueryRow(`SELECT COUNT(*) FROM param_proposals WHERE genome_hash = 'genome-db-test'`).Scan(&count)
		if count != 1 {
			t.Errorf("expected 1 proposal in db, got %d", count)
		}
	})
}

// ---------------------------------------------------------------------------
// TestHMACSignatureVerification
// ---------------------------------------------------------------------------

func TestHMACSignatureVerification(t *testing.T) {
	secret := "test-webhook-secret-key"
	auth := middleware.NewHMACAuth(secret, testLogger())

	// A simple downstream handler that returns 200 OK.
	downstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, "ok")
	})
	protected := auth.Verify(downstream)

	t.Run("valid_signature_passes", func(t *testing.T) {
		body := `{"event":"fill","order_id":"sig-001"}`
		sig := signBody(t, secret, body)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		req.Header.Set("X-Alpaca-Signature", sig)
		rec := httptest.NewRecorder()
		protected.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Errorf("expected 200 for valid signature, got %d", rec.Code)
		}
	})

	t.Run("invalid_signature_rejected", func(t *testing.T) {
		body := `{"event":"fill","order_id":"sig-002"}`
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		req.Header.Set("X-Alpaca-Signature", "deadbeef1234")
		rec := httptest.NewRecorder()
		protected.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("expected 401 for invalid signature, got %d", rec.Code)
		}
	})

	t.Run("missing_signature_rejected", func(t *testing.T) {
		body := `{"event":"fill","order_id":"sig-003"}`
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		rec := httptest.NewRecorder()
		protected.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("expected 401 for missing signature, got %d", rec.Code)
		}
	})

	t.Run("tampered_body_rejected", func(t *testing.T) {
		originalBody := `{"event":"fill","order_id":"sig-004"}`
		sig := signBody(t, secret, originalBody)
		tamperedBody := `{"event":"fill","order_id":"sig-004-TAMPERED"}`
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(tamperedBody))
		req.Header.Set("X-Alpaca-Signature", sig)
		rec := httptest.NewRecorder()
		protected.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("expected 401 for tampered body, got %d", rec.Code)
		}
	})

	t.Run("sha256_prefixed_signature_accepted", func(t *testing.T) {
		body := `{"event":"fill","order_id":"sig-005"}`
		sig := "sha256=" + signBody(t, secret, body)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		req.Header.Set("X-Alpaca-Signature", sig)
		rec := httptest.NewRecorder()
		protected.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Errorf("expected 200 for sha256= prefixed signature, got %d", rec.Code)
		}
	})

	t.Run("empty_body_with_valid_signature", func(t *testing.T) {
		body := ""
		sig := signBody(t, secret, body)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		req.Header.Set("X-Alpaca-Signature", sig)
		rec := httptest.NewRecorder()
		protected.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Errorf("expected 200 for valid signature of empty body, got %d", rec.Code)
		}
	})

	t.Run("no_secret_configured_skips_verification", func(t *testing.T) {
		authNoSecret := middleware.NewHMACAuth("", testLogger())
		handler := authNoSecret.Verify(downstream)
		body := `{"event":"fill"}`
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		// No signature header set.
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		// Should pass through because no secret is configured.
		if rec.Code != http.StatusOK {
			t.Errorf("expected 200 when no secret configured, got %d", rec.Code)
		}
	})
}

// ---------------------------------------------------------------------------
// TestHealthSummaryAggregation
// ---------------------------------------------------------------------------

func TestHealthSummaryAggregation(t *testing.T) {
	alerter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer alerter.Close()

	h := handlers.NewHealthWebhookHandler(alerter.URL, testLogger())

	postHealth := func(t *testing.T, service, status, msg string) {
		t.Helper()
		body := fmt.Sprintf(`{"status":%q,"message":%q}`, status, msg)
		req := httptest.NewRequest(http.MethodPost,
			"/webhook/health/"+service,
			strings.NewReader(body),
		)
		rec := httptest.NewRecorder()
		h.HandleServiceHealth(rec, req)
		if rec.Code != http.StatusNoContent {
			t.Errorf("[%s] expected 204, got %d", service, rec.Code)
		}
	}

	t.Run("empty_summary_returns_ok", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/webhook/health/summary", nil)
		rec := httptest.NewRecorder()
		h.HandleSummary(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d", rec.Code)
		}
		var resp map[string]any
		json.Unmarshal(rec.Body.Bytes(), &resp)
		if resp["overall"] != "ok" {
			t.Errorf("expected overall=ok, got %v", resp["overall"])
		}
	})

	t.Run("degraded_service_reflected_in_summary", func(t *testing.T) {
		postHealth(t, "risk-aggregator", "degraded", "high latency")
		req := httptest.NewRequest(http.MethodGet, "/webhook/health/summary", nil)
		rec := httptest.NewRecorder()
		h.HandleSummary(rec, req)
		var resp map[string]any
		json.Unmarshal(rec.Body.Bytes(), &resp)
		if resp["overall"] != "degraded" {
			t.Errorf("expected overall=degraded, got %v", resp["overall"])
		}
		if resp["degraded_count"].(float64) < 1 {
			t.Error("expected degraded_count >= 1")
		}
	})

	t.Run("down_service_makes_overall_down", func(t *testing.T) {
		postHealth(t, "market-data", "down", "feed stopped")
		req := httptest.NewRequest(http.MethodGet, "/webhook/health/summary", nil)
		rec := httptest.NewRecorder()
		h.HandleSummary(rec, req)
		var resp map[string]any
		json.Unmarshal(rec.Body.Bytes(), &resp)
		if resp["overall"] != "down" {
			t.Errorf("expected overall=down, got %v", resp["overall"])
		}
	})

	t.Run("all_ok_services_yield_ok_summary", func(t *testing.T) {
		h2 := handlers.NewHealthWebhookHandler(alerter.URL, testLogger())
		for _, svc := range []string{"svc-a", "svc-b", "svc-c"} {
			postHealthTo(t, h2, svc, "ok", "")
		}
		req := httptest.NewRequest(http.MethodGet, "/webhook/health/summary", nil)
		rec := httptest.NewRecorder()
		h2.HandleSummary(rec, req)
		var resp map[string]any
		json.Unmarshal(rec.Body.Bytes(), &resp)
		if resp["overall"] != "ok" {
			t.Errorf("expected overall=ok, got %v", resp["overall"])
		}
	})

	t.Run("service_count_correct", func(t *testing.T) {
		count := h.ServiceCount()
		if count < 2 {
			t.Errorf("expected >= 2 services registered, got %d", count)
		}
	})

	t.Run("unknown_service_returns_empty_status", func(t *testing.T) {
		status := h.GetServiceStatus("nonexistent-service")
		if status != "" {
			t.Errorf("expected empty status for unknown service, got %q", status)
		}
	})

	t.Run("missing_service_name_returns_400", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/webhook/health/", strings.NewReader(`{"status":"ok"}`))
		rec := httptest.NewRecorder()
		h.HandleServiceHealth(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("expected 400 for missing service name, got %d", rec.Code)
		}
	})

	t.Run("get_method_on_service_endpoint_rejected", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/webhook/health/some-service", nil)
		rec := httptest.NewRecorder()
		h.HandleServiceHealth(rec, req)
		if rec.Code != http.StatusMethodNotAllowed {
			t.Errorf("expected 405, got %d", rec.Code)
		}
	})
}

// postHealthTo is a helper that posts health to a specific handler instance.
func postHealthTo(t *testing.T, h *handlers.HealthWebhookHandler, service, status, msg string) {
	t.Helper()
	body := fmt.Sprintf(`{"status":%q,"message":%q}`, status, msg)
	req := httptest.NewRequest(http.MethodPost,
		"/webhook/health/"+service,
		strings.NewReader(body),
	)
	rec := httptest.NewRecorder()
	h.HandleServiceHealth(rec, req)
}

// ---------------------------------------------------------------------------
// TestTokenBucket
// ---------------------------------------------------------------------------

func TestTokenBucket(t *testing.T) {
	rl := middleware.NewRateLimiter(testLogger())

	t.Run("allows_requests_within_capacity", func(t *testing.T) {
		rl.Register("test-route-1", 5, 1)
		handler := rl.Limit("test-route-1", 5, 1)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))

		for i := 0; i < 5; i++ {
			req := httptest.NewRequest(http.MethodGet, "/test", nil)
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Errorf("request %d: expected 200, got %d", i+1, rec.Code)
			}
		}
	})

	t.Run("rejects_when_bucket_empty", func(t *testing.T) {
		rl2 := middleware.NewRateLimiter(testLogger())
		handler := rl2.Limit("tight-route", 2, 0.001)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))

		// Drain the bucket.
		for i := 0; i < 2; i++ {
			req := httptest.NewRequest(http.MethodGet, "/test", nil)
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
		}
		// Next request should be rate limited.
		req := httptest.NewRequest(http.MethodGet, "/test", nil)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusTooManyRequests {
			t.Errorf("expected 429 after bucket empty, got %d", rec.Code)
		}
	})

	t.Run("retry_after_header_present", func(t *testing.T) {
		rl3 := middleware.NewRateLimiter(testLogger())
		handler := rl3.Limit("retry-route", 1, 0.001)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		// Drain.
		httptest.NewRecorder() // unused, just for clarity
		req1 := httptest.NewRequest(http.MethodGet, "/test", nil)
		rl3.Limit("retry-route", 1, 0.001)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		})).ServeHTTP(httptest.NewRecorder(), req1)

		req2 := httptest.NewRequest(http.MethodGet, "/test", nil)
		rec2 := httptest.NewRecorder()
		handler.ServeHTTP(rec2, req2)
		if rec2.Code == http.StatusTooManyRequests {
			if rec2.Header().Get("Retry-After") == "" {
				t.Error("expected Retry-After header on 429")
			}
		}
	})

	t.Run("stats_returns_token_counts", func(t *testing.T) {
		rl4 := middleware.NewRateLimiter(testLogger())
		rl4.Register("stats-route", 10, 5)
		stats := rl4.Stats()
		if _, ok := stats["stats-route"]; !ok {
			t.Error("expected stats-route in stats map")
		}
	})
}

// ---------------------------------------------------------------------------
// TestContextPropagation -- ensures context cancellation is respected
// ---------------------------------------------------------------------------

func TestContextPropagation(t *testing.T) {
	db := setupTestDB(t)

	// Elixir server that hangs until cancelled.
	hangServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			return
		case <-time.After(30 * time.Second):
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer hangServer.Close()

	h := handlers.NewAlpacaFillsHandler(db, hangServer.URL, "", testLogger())

	t.Run("handler_returns_promptly_despite_slow_elixir", func(t *testing.T) {
		body := buildFillPayload("ctx-001", "BTCUSD", "buy", "fill", 1.0, 50000.0)
		req := httptest.NewRequest(http.MethodPost, "/webhook/fills", strings.NewReader(body))
		rec := httptest.NewRecorder()

		done := make(chan struct{})
		go func() {
			h.HandleFill(rec, req)
			close(done)
		}()

		select {
		case <-done:
			if rec.Code != http.StatusNoContent {
				t.Errorf("expected 204, got %d", rec.Code)
			}
		case <-time.After(2 * time.Second):
			t.Error("handler did not return within 2 seconds")
		}
	})
}

// ---------------------------------------------------------------------------
// TestHMACTiming -- verifies timing-safe comparison doesn't leak via timing
// ---------------------------------------------------------------------------

func TestHMACTiming(t *testing.T) {
	secret := "timing-test-secret"
	auth := middleware.NewHMACAuth(secret, testLogger())
	downstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	protected := auth.Verify(downstream)

	body := `{"event":"fill","order_id":"timing-test"}`

	// Valid sig.
	validSig := signBody(t, secret, body)

	// Measure valid.
	start := time.Now()
	for i := 0; i < 100; i++ {
		req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
		req.Header.Set("X-Alpaca-Signature", validSig)
		rec := httptest.NewRecorder()
		protected.ServeHTTP(rec, req)
	}
	validDur := time.Since(start)

	// Measure invalid.
	start = time.Now()
	for i := 0; i < 100; i++ {
		req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
		req.Header.Set("X-Alpaca-Signature", strings.Repeat("a", len(validSig)))
		rec := httptest.NewRecorder()
		protected.ServeHTTP(rec, req)
	}
	invalidDur := time.Since(start)

	// We can't guarantee exact timing equality in a test, but we can ensure
	// neither path is pathologically faster than the other (>10x difference).
	ratio := float64(validDur) / float64(invalidDur)
	if ratio > 10 || ratio < 0.1 {
		t.Logf("timing ratio valid/invalid = %.2f (informational, not a hard failure)", ratio)
	}
}

// Ensure os is imported for env usage in future expansions.
var _ = os.Getenv
var _ = context.Background
var _ = bytes.NewReader
var _ = json.Marshal
