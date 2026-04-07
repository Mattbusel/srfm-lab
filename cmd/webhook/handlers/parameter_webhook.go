// cmd/webhook/handlers/parameter_webhook.go -- IAE parameter update webhook.
//
// POST /webhook/params/update
//   Receives a proposed parameter set from the IAE genome engine.
//   Actions:
//     1. Validates schema compliance locally (required fields + type checks).
//     2. Enforces rate limit: max 1 parameter update per 30 minutes.
//     3. Forwards valid proposals to Elixir coordination layer (:8781/params/propose).
//     4. Logs proposal with fitness score and genome hash to SQLite.

package handlers

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Parameter proposal types
// ---------------------------------------------------------------------------

// ParamProposal is the expected JSON payload from the IAE genome engine.
type ParamProposal struct {
	GenomeHash   string             `json:"genome_hash"`
	FitnessScore float64            `json:"fitness_score"`
	Generation   int                `json:"generation"`
	Parameters   map[string]float64 `json:"parameters"`
	Metadata     map[string]any     `json:"metadata,omitempty"`
	ProposedAt   time.Time          `json:"proposed_at,omitempty"`
}

// requiredParamFields lists fields that must be present in every proposal.
var requiredParamFields = []string{
	"alpha",
	"beta",
	"risk_fraction",
	"stop_loss_pct",
}

// paramBounds defines [min, max] for numeric parameters.
var paramBounds = map[string][2]float64{
	"alpha":         {0.0, 10.0},
	"beta":          {-5.0, 5.0},
	"risk_fraction": {0.001, 0.25},
	"stop_loss_pct": {0.001, 0.20},
}

// ---------------------------------------------------------------------------
// ParameterWebhookHandler
// ---------------------------------------------------------------------------

// ParameterWebhookHandler processes parameter update notifications from IAE.
type ParameterWebhookHandler struct {
	db          *sql.DB
	elixirAddr  string
	client      *http.Client
	logger      *slog.Logger
	rateMu      sync.Mutex
	lastUpdate  time.Time
	rateWindow  time.Duration
}

// NewParameterWebhookHandler constructs a ParameterWebhookHandler.
func NewParameterWebhookHandler(db *sql.DB, elixirAddr string, logger *slog.Logger) *ParameterWebhookHandler {
	return &ParameterWebhookHandler{
		db:         db,
		elixirAddr: elixirAddr,
		client:     &http.Client{Timeout: 10 * time.Second},
		logger:     logger,
		rateWindow: 30 * time.Minute,
	}
}

// HandleParamUpdate handles POST /webhook/params/update.
func (h *ParameterWebhookHandler) HandleParamUpdate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Rate limit check.
	if !h.allowUpdate() {
		retryAfter := h.retryAfterSeconds()
		w.Header().Set("Retry-After", fmt.Sprintf("%d", retryAfter))
		http.Error(w, "rate limit: max 1 parameter update per 30 minutes", http.StatusTooManyRequests)
		h.logger.Warn("param update rate limited", "retry_after_secs", retryAfter)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		http.Error(w, "read error", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	proposal, err := h.parseProposal(body)
	if err != nil {
		h.logger.Warn("param proposal parse error", "err", err)
		http.Error(w, "parse error: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Schema validation.
	if errs := validateProposal(proposal); len(errs) > 0 {
		h.logger.Warn("param proposal schema invalid",
			"genome_hash", proposal.GenomeHash,
			"errors", errs,
		)
		writeJSONHandler(w, http.StatusUnprocessableEntity, map[string]any{
			"error":            "schema validation failed",
			"validation_errors": errs,
		})
		return
	}

	h.markUpdated()

	// Log to SQLite.
	if err := h.logProposal(r.Context(), proposal, string(body)); err != nil {
		h.logger.Error("param log failed", "err", err)
	}

	h.logger.Info("param proposal accepted",
		"genome_hash", proposal.GenomeHash,
		"fitness", proposal.FitnessScore,
		"generation", proposal.Generation,
	)

	// Forward to Elixir asynchronously.
	go h.forwardToElixir(context.Background(), proposal)

	writeJSONHandler(w, http.StatusAccepted, map[string]any{
		"status":      "accepted",
		"genome_hash": proposal.GenomeHash,
		"fitness":     proposal.FitnessScore,
	})
}

// parseProposal deserializes and sets default fields on a ParamProposal.
func (h *ParameterWebhookHandler) parseProposal(data []byte) (*ParamProposal, error) {
	var p ParamProposal
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}
	if p.GenomeHash == "" {
		return nil, fmt.Errorf("genome_hash is required")
	}
	if p.Parameters == nil {
		return nil, fmt.Errorf("parameters map is required")
	}
	if p.ProposedAt.IsZero() {
		p.ProposedAt = time.Now().UTC()
	}
	return &p, nil
}

// validateProposal checks schema compliance: required fields, bounds, fitness.
func validateProposal(p *ParamProposal) []string {
	var errs []string

	// Required field presence.
	for _, field := range requiredParamFields {
		if _, ok := p.Parameters[field]; !ok {
			errs = append(errs, fmt.Sprintf("missing required parameter: %q", field))
		}
	}

	// Bounds checking.
	for field, bounds := range paramBounds {
		val, ok := p.Parameters[field]
		if !ok {
			continue // already flagged as missing above
		}
		lo, hi := bounds[0], bounds[1]
		if val < lo || val > hi {
			errs = append(errs, fmt.Sprintf(
				"parameter %q = %.6f out of bounds [%.4f, %.4f]",
				field, val, lo, hi,
			))
		}
	}

	// Fitness score sanity.
	if p.FitnessScore < -1e6 || p.FitnessScore > 1e6 {
		errs = append(errs, fmt.Sprintf(
			"fitness_score %.4f outside plausible range [-1e6, 1e6]", p.FitnessScore,
		))
	}

	return errs
}

// logProposal writes the proposal to the param_proposals table.
func (h *ParameterWebhookHandler) logProposal(ctx context.Context, p *ParamProposal, raw string) error {
	_, err := h.db.ExecContext(ctx,
		`INSERT INTO param_proposals (genome_hash, fitness_score, schema_valid, proposed_at, raw_payload)
		 VALUES (?, ?, 1, ?, ?)`,
		p.GenomeHash,
		p.FitnessScore,
		p.ProposedAt.Format(time.RFC3339),
		raw,
	)
	if err != nil {
		return fmt.Errorf("insert param_proposal: %w", err)
	}
	return nil
}

// forwardToElixir POSTs the proposal to the Elixir coordination layer.
func (h *ParameterWebhookHandler) forwardToElixir(ctx context.Context, p *ParamProposal) {
	payload := map[string]any{
		"genome_hash":   p.GenomeHash,
		"fitness_score": p.FitnessScore,
		"generation":    p.Generation,
		"parameters":    p.Parameters,
		"metadata":      p.Metadata,
		"proposed_at":   p.ProposedAt.Format(time.RFC3339),
		"source":        "srfm-webhook",
	}
	body, err := json.Marshal(payload)
	if err != nil {
		h.logger.Error("param elixir forward: marshal", "err", err)
		return
	}

	url := h.elixirAddr + "/params/propose"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		h.logger.Error("param elixir forward: build request", "err", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-SRFM-Source", "webhook-service")

	resp, err := h.client.Do(req)
	if err != nil {
		h.logger.Warn("param elixir forward: request failed", "err", err)
		// Mark as not forwarded in DB (best effort).
		_, _ = h.db.Exec(
			`UPDATE param_proposals SET forwarded = 0 WHERE genome_hash = ? ORDER BY id DESC LIMIT 1`,
			p.GenomeHash,
		)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		h.logger.Warn("param elixir forward: non-2xx", "status", resp.StatusCode,
			"genome_hash", p.GenomeHash)
		return
	}

	// Mark as forwarded.
	_, _ = h.db.Exec(
		`UPDATE param_proposals SET forwarded = 1 WHERE genome_hash = ? ORDER BY id DESC LIMIT 1`,
		p.GenomeHash,
	)
	h.logger.Info("param proposal forwarded to elixir",
		"genome_hash", p.GenomeHash, "fitness", p.FitnessScore)
}

// allowUpdate returns true if a new update is permitted under the rate limit.
func (h *ParameterWebhookHandler) allowUpdate() bool {
	h.rateMu.Lock()
	defer h.rateMu.Unlock()
	return time.Since(h.lastUpdate) >= h.rateWindow
}

// markUpdated records the current time as the last accepted update.
func (h *ParameterWebhookHandler) markUpdated() {
	h.rateMu.Lock()
	defer h.rateMu.Unlock()
	h.lastUpdate = time.Now()
}

// retryAfterSeconds returns how many seconds until the next update is allowed.
func (h *ParameterWebhookHandler) retryAfterSeconds() int {
	h.rateMu.Lock()
	defer h.rateMu.Unlock()
	remaining := h.rateWindow - time.Since(h.lastUpdate)
	if remaining < 0 {
		return 0
	}
	return int(remaining.Seconds()) + 1
}

// writeJSONHandler is a local helper used by this package's handlers.
func writeJSONHandler(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
