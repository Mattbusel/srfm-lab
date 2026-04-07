package main

import (
	"fmt"
	"log"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// PendingOrder tracks an order from submission through terminal state.
type PendingOrder struct {
	ClOrdID     string
	Symbol      string
	Side        string
	Qty         float64
	Price       float64
	SubmittedAt time.Time
	Status      string // "pending", "partial", "filled", "cancelled", "rejected"
	CumQty      float64
	AvgPx       float64
}

// historyEntry captures a completed ExecutionReport for the order history log.
type historyEntry struct {
	ClOrdID   string
	Symbol    string
	ExecType  string
	OrdStatus string
	LastQty   float64
	LastPx    float64
	CumQty    float64
	AvgPx     float64
	ReceivedAt time.Time
}

// FIXOrderRouter routes orders through a FIXSession and tracks their lifecycle.
type FIXOrderRouter struct {
	session *FIXSession

	pending sync.Map // map[string]*PendingOrder (keyed by ClOrdID)

	histMu  sync.Mutex
	history []historyEntry

	// orderTimeout is how long before a pending order triggers an alert.
	orderTimeout time.Duration

	// alertFn is called when an order has been pending too long.
	alertFn func(order PendingOrder)

	// counter for submitted orders (atomic)
	submitCount int64

	stopCh chan struct{}
	once   sync.Once
}

// NewFIXOrderRouter creates an order router using the given session.
func NewFIXOrderRouter(session *FIXSession) *FIXOrderRouter {
	r := &FIXOrderRouter{
		session:      session,
		orderTimeout: 30 * time.Second,
		stopCh:       make(chan struct{}),
	}
	session.SetExecutionReportHandler(r.OnExecutionReport)
	return r
}

// SetOrderTimeout overrides the default 30s pending-order timeout.
func (r *FIXOrderRouter) SetOrderTimeout(d time.Duration) {
	r.orderTimeout = d
}

// SetAlertHandler registers a function called when an order exceeds the timeout.
func (r *FIXOrderRouter) SetAlertHandler(fn func(order PendingOrder)) {
	r.alertFn = fn
}

// Start begins the background timeout scanner. Call Stop when done.
func (r *FIXOrderRouter) Start() {
	r.once.Do(func() {
		go r.timeoutScanner()
	})
}

// Stop halts the timeout scanner.
func (r *FIXOrderRouter) Stop() {
	close(r.stopCh)
}

// Submit sends a FIXOrder through the session and records it as pending.
// Returns the ClOrdID assigned to the order.
func (r *FIXOrderRouter) Submit(order FIXOrder) (string, error) {
	clOrdID, err := r.session.SendOrder(order)
	if err != nil {
		return "", fmt.Errorf("submit: %w", err)
	}

	qty, _ := strconv.ParseFloat(order.Qty, 64)
	price, _ := strconv.ParseFloat(order.Price, 64)

	pending := &PendingOrder{
		ClOrdID:     clOrdID,
		Symbol:      order.Symbol,
		Side:        order.Side,
		Qty:         qty,
		Price:       price,
		SubmittedAt: time.Now(),
		Status:      "pending",
	}

	r.pending.Store(clOrdID, pending)
	atomic.AddInt64(&r.submitCount, 1)

	log.Printf("order-router: submitted clOrdID=%s symbol=%s side=%s qty=%s price=%s",
		clOrdID, order.Symbol, order.Side, order.Qty, order.Price)

	return clOrdID, nil
}

// Cancel sends a cancel request for an existing pending order.
func (r *FIXOrderRouter) Cancel(clOrdID string) error {
	val, ok := r.pending.Load(clOrdID)
	if !ok {
		return fmt.Errorf("unknown clOrdID: %s", clOrdID)
	}
	order := val.(*PendingOrder)
	if err := r.session.CancelOrder(clOrdID, order.Symbol, order.Side); err != nil {
		return fmt.Errorf("cancel: %w", err)
	}
	return nil
}

// OnExecutionReport is called by the FIXSession when a 35=8 arrives.
// It updates the pending order state and records the report in history.
func (r *FIXOrderRouter) OnExecutionReport(report ExecutionReport) {
	entry := historyEntry{
		ClOrdID:    report.ClOrdID,
		Symbol:     report.Symbol,
		ExecType:   report.ExecType,
		OrdStatus:  report.OrdStatus,
		LastQty:    report.LastQty,
		LastPx:     report.LastPx,
		CumQty:     report.CumQty,
		AvgPx:      report.AvgPx,
		ReceivedAt: time.Now(),
	}

	r.histMu.Lock()
	r.history = append(r.history, entry)
	r.histMu.Unlock()

	val, ok := r.pending.Load(report.ClOrdID)
	if !ok {
		// May be a cancel ack for an already-removed order -- log and skip.
		log.Printf("order-router: exec report for unknown clOrdID=%s execType=%s", report.ClOrdID, report.ExecType)
		return
	}

	order := val.(*PendingOrder)
	order.CumQty = report.CumQty
	order.AvgPx = report.AvgPx

	// OrdStatus values: "0"=new "1"=partial "2"=filled "4"=cancelled "8"=rejected
	switch report.OrdStatus {
	case "0":
		order.Status = "acknowledged"
	case "1":
		order.Status = "partial"
	case "2":
		order.Status = "filled"
		r.pending.Delete(report.ClOrdID)
		log.Printf("order-router: filled clOrdID=%s cumQty=%.4f avgPx=%.4f",
			report.ClOrdID, report.CumQty, report.AvgPx)
	case "4":
		order.Status = "cancelled"
		r.pending.Delete(report.ClOrdID)
		log.Printf("order-router: cancelled clOrdID=%s", report.ClOrdID)
	case "8":
		order.Status = "rejected"
		r.pending.Delete(report.ClOrdID)
		log.Printf("order-router: rejected clOrdID=%s", report.ClOrdID)
	}
}

// PendingOrders returns a snapshot of all currently pending orders sorted by submission time.
func (r *FIXOrderRouter) PendingOrders() []PendingOrder {
	var out []PendingOrder
	r.pending.Range(func(_, val interface{}) bool {
		o := val.(*PendingOrder)
		cp := *o
		out = append(out, cp)
		return true
	})
	sort.Slice(out, func(i, j int) bool {
		return out[i].SubmittedAt.Before(out[j].SubmittedAt)
	})
	return out
}

// RecentHistory returns the last n execution report entries.
func (r *FIXOrderRouter) RecentHistory(n int) []historyEntry {
	r.histMu.Lock()
	defer r.histMu.Unlock()
	if n <= 0 || n > len(r.history) {
		n = len(r.history)
	}
	start := len(r.history) - n
	out := make([]historyEntry, n)
	copy(out, r.history[start:])
	return out
}

// SubmitCount returns the total number of orders submitted.
func (r *FIXOrderRouter) SubmitCount() int64 {
	return atomic.LoadInt64(&r.submitCount)
}

// timeoutScanner runs every 5 seconds and calls alertFn for stale pending orders.
func (r *FIXOrderRouter) timeoutScanner() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-r.stopCh:
			return
		case <-ticker.C:
			r.scanTimeouts()
		}
	}
}

func (r *FIXOrderRouter) scanTimeouts() {
	now := time.Now()
	r.pending.Range(func(_, val interface{}) bool {
		order := val.(*PendingOrder)
		if now.Sub(order.SubmittedAt) > r.orderTimeout {
			log.Printf("order-router: ALERT stale order clOrdID=%s age=%v",
				order.ClOrdID, now.Sub(order.SubmittedAt).Truncate(time.Second))
			if r.alertFn != nil {
				cp := *order
				r.alertFn(cp)
			}
		}
		return true
	})
}
