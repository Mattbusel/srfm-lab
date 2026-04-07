package main

import (
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// FIXAPIServer exposes the FIX gateway via a REST HTTP interface.
type FIXAPIServer struct {
	session *FIXSession
	router  *FIXOrderRouter
	addr    string
}

// NewFIXAPIServer creates an API server.
func NewFIXAPIServer(addr string, session *FIXSession, router *FIXOrderRouter) *FIXAPIServer {
	return &FIXAPIServer{
		addr:    addr,
		session: session,
		router:  router,
	}
}

// ListenAndServe registers all routes and starts the HTTP server.
func (a *FIXAPIServer) ListenAndServe() error {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", a.handleHealth)
	mux.HandleFunc("/fix/status", a.handleFIXStatus)
	mux.HandleFunc("/fix/order", a.handleOrder)
	mux.HandleFunc("/fix/orders/pending", a.handlePending)
	mux.HandleFunc("/fix/orders/history", a.handleHistory)

	// /fix/order/:clOrdID -- mux doesn't support path params, so we strip prefix manually.
	mux.HandleFunc("/fix/order/", a.handleOrderByID)

	log.Printf("fix-api: listening on %s", a.addr)
	return http.ListenAndServe(a.addr, mux)
}

// -- GET /health --

type healthResponse struct {
	Status        string `json:"status"`
	SessionState  string `json:"session_state"`
	Timestamp     string `json:"timestamp"`
}

func (a *FIXAPIServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	status := "ok"
	if !a.session.IsActive() {
		status = "degraded"
	}
	writeJSONAPI(w, http.StatusOK, healthResponse{
		Status:       status,
		SessionState: a.session.State().String(),
		Timestamp:    time.Now().UTC().Format(time.RFC3339),
	})
}

// -- GET /fix/status --

type fixStatusResponse struct {
	SessionState string `json:"session_state"`
	SeqNumOut    int64  `json:"seq_num_out"`
	SeqNumIn     int64  `json:"seq_num_in"`
	SenderCompID string `json:"sender_comp_id"`
	TargetCompID string `json:"target_comp_id"`
	PendingCount int    `json:"pending_orders"`
	SubmitTotal  int64  `json:"submitted_total"`
}

func (a *FIXAPIServer) handleFIXStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSONAPI(w, http.StatusOK, fixStatusResponse{
		SessionState: a.session.State().String(),
		SeqNumOut:    a.session.SeqNumOut(),
		SeqNumIn:     a.session.SeqNumIn(),
		SenderCompID: a.session.cfg.SenderCompID,
		TargetCompID: a.session.cfg.TargetCompID,
		PendingCount: len(a.router.PendingOrders()),
		SubmitTotal:  a.router.SubmitCount(),
	})
}

// -- POST /fix/order --

type orderRequest struct {
	Symbol  string `json:"symbol"`
	Side    string `json:"side"`    // "buy" or "sell"
	Qty     string `json:"qty"`
	Price   string `json:"price"`
	OrdType string `json:"ord_type"` // "market" or "limit"
}

type orderResponse struct {
	ClOrdID string `json:"cl_ord_id"`
	Status  string `json:"status"`
}

func (a *FIXAPIServer) handleOrder(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		a.handleSubmitOrder(w, r)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (a *FIXAPIServer) handleSubmitOrder(w http.ResponseWriter, r *http.Request) {
	var req orderRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	side := "1" // buy
	if strings.EqualFold(req.Side, "sell") {
		side = "2"
	}
	ordType := "2" // limit
	if strings.EqualFold(req.OrdType, "market") {
		ordType = "1"
	}

	order := FIXOrder{
		Symbol:  req.Symbol,
		Side:    side,
		Qty:     req.Qty,
		Price:   req.Price,
		OrdType: ordType,
	}

	clOrdID, err := a.router.Submit(order)
	if err != nil {
		http.Error(w, "submit error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSONAPI(w, http.StatusCreated, orderResponse{
		ClOrdID: clOrdID,
		Status:  "submitted",
	})
}

// -- DELETE /fix/order/:clOrdID --

func (a *FIXAPIServer) handleOrderByID(w http.ResponseWriter, r *http.Request) {
	// Path: /fix/order/<clOrdID>
	clOrdID := strings.TrimPrefix(r.URL.Path, "/fix/order/")
	if clOrdID == "" {
		http.Error(w, "missing clOrdID", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodDelete:
		if err := a.router.Cancel(clOrdID); err != nil {
			http.Error(w, "cancel error: "+err.Error(), http.StatusBadRequest)
			return
		}
		writeJSONAPI(w, http.StatusOK, map[string]string{
			"cl_ord_id": clOrdID,
			"status":    "cancel_sent",
		})
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// -- GET /fix/orders/pending --

func (a *FIXAPIServer) handlePending(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	orders := a.router.PendingOrders()
	type pendingItem struct {
		ClOrdID     string    `json:"cl_ord_id"`
		Symbol      string    `json:"symbol"`
		Side        string    `json:"side"`
		Qty         float64   `json:"qty"`
		Price       float64   `json:"price"`
		Status      string    `json:"status"`
		SubmittedAt time.Time `json:"submitted_at"`
		AgeSeconds  float64   `json:"age_seconds"`
		CumQty      float64   `json:"cum_qty"`
		AvgPx       float64   `json:"avg_px"`
	}
	now := time.Now()
	out := make([]pendingItem, 0, len(orders))
	for _, o := range orders {
		out = append(out, pendingItem{
			ClOrdID:     o.ClOrdID,
			Symbol:      o.Symbol,
			Side:        o.Side,
			Qty:         o.Qty,
			Price:       o.Price,
			Status:      o.Status,
			SubmittedAt: o.SubmittedAt,
			AgeSeconds:  now.Sub(o.SubmittedAt).Seconds(),
			CumQty:      o.CumQty,
			AvgPx:       o.AvgPx,
		})
	}
	writeJSONAPI(w, http.StatusOK, map[string]interface{}{
		"pending": out,
		"count":   len(out),
	})
}

// -- GET /fix/orders/history?n=100 --

func (a *FIXAPIServer) handleHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	n := 100
	if nStr := r.URL.Query().Get("n"); nStr != "" {
		if parsed, err := strconv.Atoi(nStr); err == nil && parsed > 0 {
			n = parsed
		}
	}
	entries := a.router.RecentHistory(n)
	writeJSONAPI(w, http.StatusOK, map[string]interface{}{
		"history": entries,
		"count":   len(entries),
	})
}

// writeJSONAPI is the API-layer variant of writeJSON.
func writeJSONAPI(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("fix-api: response encode error: %v", err)
	}
}
