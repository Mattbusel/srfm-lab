package replay

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// sessionState holds the mutable state of a replay HTTP session.
type sessionState struct {
	mu            sync.RWMutex
	running       bool
	cfg           ReplayConfig
	replayer      *BarReplayer
	cancel        context.CancelFunc
	simTime       time.Time
	progress      float64
	speedMult     float64
}

// ReplayServer exposes replay functionality via HTTP and WebSocket.
// It presents the same data interface as the live market-data server,
// allowing backtesting code to consume historical data transparently.
type ReplayServer struct {
	addr    string
	session *sessionState
	upgrader websocket.Upgrader
}

// NewReplayServer creates a ReplayServer that listens on addr (e.g. ":8090").
func NewReplayServer(addr string) *ReplayServer {
	return &ReplayServer{
		addr: addr,
		session: &sessionState{
			speedMult: 1.0,
		},
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}
}

// ListenAndServe registers routes and starts the HTTP server.
// It blocks until the server returns an error.
func (s *ReplayServer) ListenAndServe() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/replay/start", s.handleStart)
	mux.HandleFunc("/replay/status", s.handleStatus)
	mux.HandleFunc("/replay/speed", s.handleSpeed)
	mux.HandleFunc("/replay/stop", s.handleStop)
	mux.HandleFunc("/replay/stream", s.handleStream)

	srv := &http.Server{Addr: s.addr, Handler: mux}
	log.Printf("replay-server listening on %s", s.addr)
	return srv.ListenAndServe()
}

// -- /replay/start --

// startRequest is the JSON body expected by POST /replay/start.
type startRequest struct {
	SpeedMultiplier float64  `json:"speed_multiplier"`
	StartTime       string   `json:"start_time"` // RFC3339
	EndTime         string   `json:"end_time"`   // RFC3339
	Symbols         []string `json:"symbols"`
	DataDir         string   `json:"data_dir"`
}

func (s *ReplayServer) handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	s.session.mu.Lock()
	defer s.session.mu.Unlock()

	if s.session.running {
		http.Error(w, "replay already running", http.StatusConflict)
		return
	}

	var req startRequest
	if r.Method == http.MethodPost {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		q := r.URL.Query()
		req.DataDir = q.Get("data_dir")
		req.Symbols = splitComma(q.Get("symbols"))
		if v, err := strconv.ParseFloat(q.Get("speed"), 64); err == nil {
			req.SpeedMultiplier = v
		} else {
			req.SpeedMultiplier = 1.0
		}
		req.StartTime = q.Get("start_time")
		req.EndTime = q.Get("end_time")
	}

	cfg := ReplayConfig{
		SpeedMultiplier: req.SpeedMultiplier,
		Symbols:         req.Symbols,
		DataDir:         req.DataDir,
	}
	if req.StartTime != "" {
		if t, err := time.Parse(time.RFC3339, req.StartTime); err == nil {
			cfg.StartTime = t
		}
	}
	if req.EndTime != "" {
		if t, err := time.Parse(time.RFC3339, req.EndTime); err == nil {
			cfg.EndTime = t
		}
	}

	s.session.cfg = cfg
	s.session.speedMult = cfg.SpeedMultiplier
	s.session.progress = 0
	s.session.running = true

	ctx, cancel := context.WithCancel(context.Background())
	s.session.cancel = cancel
	rep := NewBarReplayer(cfg)
	s.session.replayer = rep

	// Run replay in background, updating session state.
	go func() {
		ch := rep.Start(ctx)
		for ev := range ch {
			s.session.mu.Lock()
			s.session.simTime = ev.SimulatedTime
			s.session.progress = rep.Progress()
			s.session.mu.Unlock()
		}
		s.session.mu.Lock()
		s.session.running = false
		s.session.progress = 1.0
		s.session.mu.Unlock()
	}()

	writeJSON(w, http.StatusOK, map[string]string{"status": "started"})
}

// -- /replay/status --

type statusResponse struct {
	Running       bool      `json:"running"`
	SimulatedTime time.Time `json:"simulated_time"`
	Progress      float64   `json:"progress"`
	SpeedMult     float64   `json:"speed_multiplier"`
}

func (s *ReplayServer) handleStatus(w http.ResponseWriter, r *http.Request) {
	s.session.mu.RLock()
	defer s.session.mu.RUnlock()

	writeJSON(w, http.StatusOK, statusResponse{
		Running:       s.session.running,
		SimulatedTime: s.session.simTime,
		Progress:      s.session.progress,
		SpeedMult:     s.session.speedMult,
	})
}

// -- /replay/speed --

type speedRequest struct {
	SpeedMultiplier float64 `json:"speed_multiplier"`
}

func (s *ReplayServer) handleSpeed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req speedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if req.SpeedMultiplier < 0 {
		http.Error(w, "speed_multiplier must be >= 0", http.StatusBadRequest)
		return
	}

	s.session.mu.Lock()
	s.session.speedMult = req.SpeedMultiplier
	if s.session.replayer != nil {
		// The BarReplayer reads SpeedMultiplier from cfg at start; we update in place.
		s.session.cfg.SpeedMultiplier = req.SpeedMultiplier
	}
	s.session.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"speed_multiplier": req.SpeedMultiplier,
	})
}

// -- /replay/stop --

func (s *ReplayServer) handleStop(w http.ResponseWriter, r *http.Request) {
	s.session.mu.Lock()
	defer s.session.mu.Unlock()

	if !s.session.running {
		writeJSON(w, http.StatusOK, map[string]string{"status": "not_running"})
		return
	}

	if s.session.cancel != nil {
		s.session.cancel()
		s.session.cancel = nil
	}
	s.session.running = false
	s.session.replayer = nil
	s.session.progress = 0
	s.session.simTime = time.Time{}

	writeJSON(w, http.StatusOK, map[string]string{"status": "stopped"})
}

// -- /replay/stream (WebSocket) --

// wsEvent is the JSON shape streamed to WebSocket clients.
type wsEvent struct {
	Symbol        string    `json:"symbol"`
	Timestamp     time.Time `json:"timestamp"`
	Open          float64   `json:"open"`
	High          float64   `json:"high"`
	Low           float64   `json:"low"`
	Close         float64   `json:"close"`
	Volume        float64   `json:"volume"`
	SimulatedTime time.Time `json:"simulated_time"`
	IsLast        bool      `json:"is_last"`
	Progress      float64   `json:"progress"`
}

func (s *ReplayServer) handleStream(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("ws upgrade: %v", err)
		return
	}
	defer conn.Close()

	// Parse query params for a new isolated replay session on this connection.
	q := r.URL.Query()
	cfg := ReplayConfig{
		DataDir: q.Get("data_dir"),
		Symbols: splitComma(q.Get("symbols")),
	}
	if v, err := strconv.ParseFloat(q.Get("speed"), 64); err == nil {
		cfg.SpeedMultiplier = v
	} else {
		cfg.SpeedMultiplier = 1.0
	}
	if st := q.Get("start_time"); st != "" {
		if t, err := time.Parse(time.RFC3339, st); err == nil {
			cfg.StartTime = t
		}
	}
	if et := q.Get("end_time"); et != "" {
		if t, err := time.Parse(time.RFC3339, et); err == nil {
			cfg.EndTime = t
		}
	}

	// If no symbol list provided, fall back to shared session config.
	if len(cfg.Symbols) == 0 {
		s.session.mu.RLock()
		cfg = s.session.cfg
		s.session.mu.RUnlock()
	}

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	rep := NewBarReplayer(cfg)
	ch := rep.Start(ctx)

	// Read pump -- handle client disconnect.
	go func() {
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				cancel()
				return
			}
		}
	}()

	for ev := range ch {
		msg := wsEvent{
			Symbol:        ev.Symbol,
			Timestamp:     ev.Bar.Timestamp,
			Open:          ev.Bar.Open,
			High:          ev.Bar.High,
			Low:           ev.Bar.Low,
			Close:         ev.Bar.Close,
			Volume:        ev.Bar.Volume,
			SimulatedTime: ev.SimulatedTime,
			IsLast:        ev.IsLast,
			Progress:      rep.Progress(),
		}
		if err := conn.WriteJSON(msg); err != nil {
			log.Printf("ws write: %v", err)
			return
		}
	}
}

// -- helpers --

func writeJSON(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("writeJSON: %v", err)
	}
}

func splitComma(s string) []string {
	if s == "" {
		return nil
	}
	var out []string
	for _, part := range splitOn(s, ',') {
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func splitOn(s string, sep rune) []string {
	var parts []string
	start := 0
	for i, r := range s {
		if r == sep {
			parts = append(parts, s[start:i])
			start = i + 1
		}
	}
	parts = append(parts, s[start:])
	return parts
}

// ReplayEventToJSON serializes a ReplayEvent for use by non-WebSocket consumers.
func ReplayEventToJSON(ev ReplayEvent) ([]byte, error) {
	v := struct {
		Symbol        string    `json:"symbol"`
		Open          float64   `json:"open"`
		High          float64   `json:"high"`
		Low           float64   `json:"low"`
		Close         float64   `json:"close"`
		Volume        float64   `json:"volume"`
		Timestamp     time.Time `json:"timestamp"`
		SimulatedTime time.Time `json:"simulated_time"`
		IsLast        bool      `json:"is_last"`
	}{
		Symbol:        ev.Symbol,
		Open:          ev.Bar.Open,
		High:          ev.Bar.High,
		Low:           ev.Bar.Low,
		Close:         ev.Bar.Close,
		Volume:        ev.Bar.Volume,
		Timestamp:     ev.Bar.Timestamp,
		SimulatedTime: ev.SimulatedTime,
		IsLast:        ev.IsLast,
	}
	return json.Marshal(v)
}

