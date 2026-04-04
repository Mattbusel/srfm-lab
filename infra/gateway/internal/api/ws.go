package api

import (
	"net/http"

	"github.com/srfm/gateway/internal/hub"
	"go.uber.org/zap"
)

// WSHandler wraps the hub and exposes an HTTP handler for WebSocket upgrades.
type WSHandler struct {
	hub *hub.Hub
	log *zap.Logger
}

// NewWSHandler creates a WSHandler.
func NewWSHandler(h *hub.Hub, log *zap.Logger) *WSHandler {
	return &WSHandler{hub: h, log: log}
}

// ServeHTTP upgrades the connection and hands it to the hub.
// The caller should mount this at e.g. GET /ws.
func (wsh *WSHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	wsh.log.Debug("ws connection incoming",
		zap.String("remote", r.RemoteAddr),
		zap.String("user_agent", r.UserAgent()))
	wsh.hub.HandleConn(w, r)
}
