// Package dashboard provides a minimal Go web server for the monitoring dashboard.
package dashboard

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/srfm/monitor/internal/alerting"
	"go.uber.org/zap"
)

// EquityPoint is a single (time, equity) data point for the equity curve.
type EquityPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Equity    float64   `json:"equity"`
	DailyPnL  float64   `json:"daily_pnl"`
}

// DashboardState is the shared state object for the dashboard server.
type DashboardState struct {
	mu sync.RWMutex

	EquityCurve    []EquityPoint
	ActiveAlerts   []alerting.Alert
	BHMasses       map[string]float64
	PortfolioState *alerting.PortfolioState
	RecentAlerts   []alerting.Alert
	MaxEquityPoints int
}

// NewDashboardState creates a DashboardState.
func NewDashboardState(maxEquityPoints int) *DashboardState {
	if maxEquityPoints <= 0 {
		maxEquityPoints = 43200 // 30 days of 1-min points
	}
	return &DashboardState{
		BHMasses:        make(map[string]float64),
		MaxEquityPoints: maxEquityPoints,
	}
}

// AddEquityPoint appends an equity point to the curve (bounded by max).
func (ds *DashboardState) AddEquityPoint(pt EquityPoint) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.EquityCurve = append(ds.EquityCurve, pt)
	if len(ds.EquityCurve) > ds.MaxEquityPoints {
		ds.EquityCurve = ds.EquityCurve[len(ds.EquityCurve)-ds.MaxEquityPoints:]
	}
}

// AddAlert adds an alert, keeping the last 500.
func (ds *DashboardState) AddAlert(a alerting.Alert) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.RecentAlerts = append(ds.RecentAlerts, a)
	if len(ds.RecentAlerts) > 500 {
		ds.RecentAlerts = ds.RecentAlerts[len(ds.RecentAlerts)-500:]
	}
}

// UpdatePortfolio updates the current portfolio state.
func (ds *DashboardState) UpdatePortfolio(state alerting.PortfolioState) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.PortfolioState = &state
}

// UpdateBHMasses replaces the BH mass map.
func (ds *DashboardState) UpdateBHMasses(masses map[string]float64) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.BHMasses = masses
}

// Snapshot returns a deep copy of the current state for serving.
func (ds *DashboardState) Snapshot() map[string]interface{} {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	curve := make([]EquityPoint, len(ds.EquityCurve))
	copy(curve, ds.EquityCurve)

	alerts := make([]alerting.Alert, len(ds.RecentAlerts))
	copy(alerts, ds.RecentAlerts)

	masses := make(map[string]float64, len(ds.BHMasses))
	for k, v := range ds.BHMasses {
		masses[k] = v
	}

	var portfolio interface{}
	if ds.PortfolioState != nil {
		cp := *ds.PortfolioState
		portfolio = cp
	}

	return map[string]interface{}{
		"equity_curve":    curve,
		"recent_alerts":   alerts,
		"bh_masses":       masses,
		"portfolio":       portfolio,
		"as_of":           time.Now(),
	}
}

// Server is the dashboard HTTP server.
type Server struct {
	state     *DashboardState
	log       *zap.Logger
	sseClients map[chan string]struct{}
	sseMu      sync.Mutex
}

// NewServer creates a dashboard Server.
func NewServer(state *DashboardState, log *zap.Logger) *Server {
	return &Server{
		state:      state,
		log:        log,
		sseClients: make(map[chan string]struct{}),
	}
}

// Routes returns the chi router.
func (s *Server) Routes() http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.Recoverer)
	r.Get("/", s.serveIndex)
	r.Get("/api/state", s.apiState)
	r.Get("/api/equity", s.apiEquity)
	r.Get("/api/alerts", s.apiAlerts)
	r.Get("/api/bh", s.apiBH)
	r.Get("/api/portfolio", s.apiPortfolio)
	r.Get("/events", s.serveSSE)
	return r
}

// serveIndex serves the embedded dashboard HTML.
func (s *Server) serveIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, DashboardHTML)
}

func (s *Server) apiState(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.state.Snapshot())
}

func (s *Server) apiEquity(w http.ResponseWriter, r *http.Request) {
	s.state.mu.RLock()
	curve := make([]EquityPoint, len(s.state.EquityCurve))
	copy(curve, s.state.EquityCurve)
	s.state.mu.RUnlock()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"equity_curve": curve,
		"count":        len(curve),
	})
}

func (s *Server) apiAlerts(w http.ResponseWriter, r *http.Request) {
	s.state.mu.RLock()
	alerts := make([]alerting.Alert, len(s.state.RecentAlerts))
	copy(alerts, s.state.RecentAlerts)
	s.state.mu.RUnlock()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"alerts": alerts,
		"count":  len(alerts),
	})
}

func (s *Server) apiBH(w http.ResponseWriter, r *http.Request) {
	s.state.mu.RLock()
	masses := make(map[string]float64, len(s.state.BHMasses))
	for k, v := range s.state.BHMasses {
		masses[k] = v
	}
	s.state.mu.RUnlock()
	writeJSON(w, http.StatusOK, masses)
}

func (s *Server) apiPortfolio(w http.ResponseWriter, r *http.Request) {
	s.state.mu.RLock()
	var ps interface{}
	if s.state.PortfolioState != nil {
		cp := *s.state.PortfolioState
		ps = cp
	}
	s.state.mu.RUnlock()
	if ps == nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "no portfolio data"})
		return
	}
	writeJSON(w, http.StatusOK, ps)
}

// serveSSE handles Server-Sent Events connections for real-time dashboard updates.
func (s *Server) serveSSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "SSE not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	msgCh := make(chan string, 64)
	s.sseMu.Lock()
	s.sseClients[msgCh] = struct{}{}
	s.sseMu.Unlock()

	defer func() {
		s.sseMu.Lock()
		delete(s.sseClients, msgCh)
		s.sseMu.Unlock()
	}()

	// Send initial state.
	snapshot, _ := json.Marshal(s.state.Snapshot())
	fmt.Fprintf(w, "data: %s\n\n", snapshot)
	flusher.Flush()

	heartbeat := time.NewTicker(15 * time.Second)
	defer heartbeat.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case msg := <-msgCh:
			fmt.Fprintf(w, "data: %s\n\n", msg)
			flusher.Flush()
		case <-heartbeat.C:
			fmt.Fprintf(w, ": ping\n\n")
			flusher.Flush()
		}
	}
}

// PushSSE sends a message to all connected SSE clients.
func (s *Server) PushSSE(payload interface{}) {
	raw, err := json.Marshal(payload)
	if err != nil {
		s.log.Warn("sse marshal", zap.Error(err))
		return
	}
	msg := string(raw)
	s.sseMu.Lock()
	defer s.sseMu.Unlock()
	for ch := range s.sseClients {
		select {
		case ch <- msg:
		default:
		}
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
