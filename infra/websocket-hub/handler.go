// handler.go — HTTP upgrade handler, CORS, compression (permessage-deflate).
package wshub

import (
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// Upgrader configuration
// ─────────────────────────────────────────────────────────────────────────────

// HandlerConfig configures the HTTP upgrade handler.
type HandlerConfig struct {
	// AllowedOrigins is the list of allowed CORS origins.
	// Use "*" to allow all origins (not recommended in production).
	AllowedOrigins []string

	// CheckOrigin is a custom origin validator. If nil, AllowedOrigins is used.
	CheckOrigin func(r *http.Request) bool

	// EnableCompression enables permessage-deflate compression.
	EnableCompression bool

	// ReadBufferSize and WriteBufferSize control the underlying WebSocket buffers.
	ReadBufferSize  int
	WriteBufferSize int

	// HandshakeTimeout is the deadline for the WebSocket handshake.
	HandshakeTimeout time.Duration

	// RequireAuth: if true, unauthenticated upgrade requests are rejected with 401.
	RequireAuth bool
}

// DefaultHandlerConfig returns secure defaults.
func DefaultHandlerConfig() HandlerConfig {
	return HandlerConfig{
		AllowedOrigins:    []string{"*"},
		EnableCompression: true,
		ReadBufferSize:    4096,
		WriteBufferSize:   4096,
		HandshakeTimeout:  10 * time.Second,
		RequireAuth:       false,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// UpgradeHandler
// ─────────────────────────────────────────────────────────────────────────────

// UpgradeHandler handles HTTP → WebSocket upgrades.
type UpgradeHandler struct {
	hub      *Hub
	auth     *Authenticator
	rl       *HubRateLimiter
	upgrader websocket.Upgrader
	cfg      HandlerConfig
	log      *zap.Logger
}

// NewUpgradeHandler creates an UpgradeHandler.
func NewUpgradeHandler(hub *Hub, auth *Authenticator, rl *HubRateLimiter, cfg HandlerConfig, log *zap.Logger) *UpgradeHandler {
	checkOrigin := cfg.CheckOrigin
	if checkOrigin == nil {
		allowedOrigins := cfg.AllowedOrigins
		checkOrigin = func(r *http.Request) bool {
			if len(allowedOrigins) == 0 {
				return false
			}
			origin := r.Header.Get("Origin")
			if origin == "" {
				return true // non-browser clients
			}
			for _, allowed := range allowedOrigins {
				if allowed == "*" || strings.EqualFold(allowed, origin) {
					return true
				}
			}
			return false
		}
	}

	upgrader := websocket.Upgrader{
		ReadBufferSize:    cfg.ReadBufferSize,
		WriteBufferSize:   cfg.WriteBufferSize,
		HandshakeTimeout:  cfg.HandshakeTimeout,
		EnableCompression: cfg.EnableCompression,
		CheckOrigin:       checkOrigin,
	}

	return &UpgradeHandler{
		hub:      hub,
		auth:     auth,
		rl:       rl,
		upgrader: upgrader,
		cfg:      cfg,
		log:      log,
	}
}

// ServeHTTP handles a WebSocket upgrade request.
func (h *UpgradeHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// CORS preflight.
	if r.Method == http.MethodOptions {
		h.setCORSHeaders(w, r)
		w.WriteHeader(http.StatusNoContent)
		return
	}
	h.setCORSHeaders(w, r)

	// Authenticate before upgrading (saves a connection slot if creds are bad).
	var claims *WSClaims
	if h.auth != nil {
		var err error
		claims, err = h.auth.AuthenticateRequest(r)
		if err != nil {
			if h.cfg.RequireAuth {
				http.Error(w, "unauthorized: "+err.Error(), http.StatusUnauthorized)
				hubRejectedTotal.WithLabelValues("auth").Inc()
				return
			}
			// Allow anonymous.
			claims = &WSClaims{Subject: "anonymous", Roles: []string{"viewer"}}
		}
	}

	// Check global rate limit for new connections.
	if h.rl != nil && !h.rl.AllowInbound("upgrade") {
		http.Error(w, "too many requests", http.StatusTooManyRequests)
		hubRejectedTotal.WithLabelValues("rate_limit").Inc()
		return
	}

	// Upgrade.
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.log.Warn("WebSocket upgrade failed", zap.Error(err))
		hubRejectedTotal.WithLabelValues("upgrade_error").Inc()
		return
	}

	// Build client.
	c := newClient(h.hub, conn, h.hub.cfg.ClientCfg, h.log)
	if claims != nil {
		c.UserID = claims.Subject
		c.AccountID = claims.AccountID
		c.Roles = claims.Roles
	}

	// Register with hub.
	if err := h.hub.RegisterClient(c); err != nil {
		h.log.Warn("client registration failed", zap.Error(err))
		_ = conn.WriteJSON(NewErrorMessage("", ErrCodeServiceUnavail, "server at capacity", err.Error()))
		_ = conn.Close()
		hubRejectedTotal.WithLabelValues("capacity").Inc()
		return
	}

	hubUpgradesTotal.Inc()
	h.log.Info("client connected",
		zap.String("client", c.ID),
		zap.String("user", c.UserID),
		zap.String("remote", r.RemoteAddr))

	// Run the client pumps (blocks until disconnect).
	c.run()
}

// setCORSHeaders writes CORS response headers.
func (h *UpgradeHandler) setCORSHeaders(w http.ResponseWriter, r *http.Request) {
	origin := r.Header.Get("Origin")
	if origin == "" {
		return
	}
	for _, allowed := range h.cfg.AllowedOrigins {
		if allowed == "*" || strings.EqualFold(allowed, origin) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Credentials", "true")
			w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Authorization, X-API-Key, Content-Type")
			w.Header().Set("Vary", "Origin")
			return
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Admin HTTP handler
// ─────────────────────────────────────────────────────────────────────────────

// AdminHandler serves a JSON endpoint with hub statistics.
type AdminHandler struct {
	hub *Hub
	log *zap.Logger
}

// NewAdminHandler creates an AdminHandler.
func NewAdminHandler(hub *Hub, log *zap.Logger) *AdminHandler {
	return &AdminHandler{hub: hub, log: log}
}

// ServeHTTP handles admin API requests.
func (h *AdminHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch r.URL.Path {
	case "/admin/stats":
		h.handleStats(w, r)
	case "/admin/rooms":
		h.handleRooms(w, r)
	case "/admin/clients":
		h.handleClients(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (h *AdminHandler) handleStats(w http.ResponseWriter, r *http.Request) {
	stats := h.hub.Stats()
	writeJSON(w, stats)
}

func (h *AdminHandler) handleRooms(w http.ResponseWriter, r *http.Request) {
	rooms := h.hub.ListRooms()
	writeJSON(w, rooms)
}

func (h *AdminHandler) handleClients(w http.ResponseWriter, r *http.Request) {
	h.hub.clientsMu.RLock()
	clients := make([]ClientStats, 0, len(h.hub.clients))
	for _, c := range h.hub.clients {
		clients = append(clients, c.Stats())
	}
	h.hub.clientsMu.RUnlock()
	writeJSON(w, clients)
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	enc := newJSONEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		http.Error(w, "encode error", http.StatusInternalServerError)
	}
}
