package main

import (
	"strconv"
	"strings"
	"testing"
	"time"
)

// -- FIX message encoding/decoding tests --

func TestEncodeDecode_Roundtrip(t *testing.T) {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeNewOrderSingle)
	msg.Set(TagSenderCompID, "SENDER")
	msg.Set(TagTargetCompID, "TARGET")
	msg.Set(TagMsgSeqNum, "1")
	msg.Set(TagSendingTime, "20240101-12:00:00.000")
	msg.Set(TagClOrdID, "ORD001")
	msg.Set(TagSymbol, "AAPL")
	msg.Set(TagSide, "1")
	msg.Set(TagOrderQty, "100")
	msg.Set(TagOrdType, "2")
	msg.Set(TagPrice, "150.50")
	msg.Set(TagTimeInForce, "0")
	msg.Set(TagTransactTime, "20240101-12:00:00")
	msg.Set(TagHandlInst, "1")

	wire := Encode(msg)
	decoded, err := Decode(wire)
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if decoded.MsgType != MsgTypeNewOrderSingle {
		t.Errorf("expected MsgType %s got %s", MsgTypeNewOrderSingle, decoded.MsgType)
	}
	if decoded.Get(TagSymbol) != "AAPL" {
		t.Errorf("expected symbol AAPL got %s", decoded.Get(TagSymbol))
	}
	if decoded.Get(TagPrice) != "150.50" {
		t.Errorf("expected price 150.50 got %s", decoded.Get(TagPrice))
	}
}

func TestEncode_ContainsSOH(t *testing.T) {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeHeartbeat)
	msg.Set(TagSenderCompID, "A")
	msg.Set(TagTargetCompID, "B")
	msg.Set(TagMsgSeqNum, "1")
	msg.Set(TagSendingTime, "20240101-00:00:00.000")

	wire := Encode(msg)
	if !strings.Contains(string(wire), SOH) {
		t.Error("encoded message missing SOH delimiter")
	}
}

func TestEncode_ChecksumFormat(t *testing.T) {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeHeartbeat)
	msg.Set(TagSenderCompID, "SENDER")
	msg.Set(TagTargetCompID, "TARGET")
	msg.Set(TagMsgSeqNum, "2")
	msg.Set(TagSendingTime, "20240101-00:00:00.000")

	wire := Encode(msg)
	s := string(wire)

	// Last field before trailing SOH should be 10=NNN
	trimmed := strings.TrimRight(s, SOH)
	parts := strings.Split(trimmed, SOH)
	last := parts[len(parts)-1]
	if !strings.HasPrefix(last, "10=") {
		t.Errorf("expected last field to be checksum (10=), got: %q", last)
	}
	csVal := strings.TrimPrefix(last, "10=")
	if len(csVal) != 3 {
		t.Errorf("checksum value should be 3 digits, got %q", csVal)
	}
}

func TestDecode_ChecksumMismatch(t *testing.T) {
	// Build a valid message then corrupt the checksum.
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeLogon)
	msg.Set(TagSenderCompID, "X")
	msg.Set(TagTargetCompID, "Y")
	msg.Set(TagMsgSeqNum, "1")
	msg.Set(TagSendingTime, "20240101-00:00:00.000")
	msg.Set(TagHeartBtInt, "30")

	wire := Encode(msg)
	// Replace checksum value with 000.
	s := string(wire)
	parts := strings.Split(s, SOH)
	for i, p := range parts {
		if strings.HasPrefix(p, "10=") {
			parts[i] = "10=000"
		}
	}
	corrupted := strings.Join(parts, SOH)

	_, err := Decode([]byte(corrupted))
	if err == nil {
		t.Error("expected checksum mismatch error, got nil")
	}
}

func TestNewOrderSingle_Fields(t *testing.T) {
	msg := NewOrderSingle("C001", "MSFT", "1", "50", "300.00", "2", "20240101-09:30:00")
	if msg.Get(TagClOrdID) != "C001" {
		t.Errorf("ClOrdID mismatch")
	}
	if msg.Get(TagSymbol) != "MSFT" {
		t.Errorf("Symbol mismatch")
	}
	if msg.Get(TagSide) != "1" {
		t.Errorf("Side mismatch")
	}
	if msg.Get(TagOrdType) != "2" {
		t.Errorf("OrdType mismatch")
	}
	if msg.Get(TagPrice) != "300.00" {
		t.Errorf("Price mismatch")
	}
	if msg.MsgType != MsgTypeNewOrderSingle {
		t.Errorf("MsgType mismatch")
	}
}

func TestNewOrderSingle_MarketOrder_NoPrice(t *testing.T) {
	msg := NewOrderSingle("C002", "TSLA", "2", "10", "", "1", "20240101-09:30:00")
	if msg.Get(TagPrice) != "" {
		t.Errorf("market order should not have Price tag, got %q", msg.Get(TagPrice))
	}
	if msg.Get(TagOrdType) != "1" {
		t.Errorf("expected OrdType=1 (market)")
	}
}

func TestNewOrderCancelRequest_Fields(t *testing.T) {
	msg := NewOrderCancelRequest("C003", "C001", "MSFT", "1", "20240101-09:35:00")
	if msg.MsgType != MsgTypeOrderCancelRequest {
		t.Errorf("expected OrderCancelRequest MsgType")
	}
	if msg.Get(TagOrigClOrdID) != "C001" {
		t.Errorf("OrigClOrdID mismatch")
	}
	if msg.Get(TagClOrdID) != "C003" {
		t.Errorf("ClOrdID mismatch")
	}
}

func TestParseExecutionReport_Filled(t *testing.T) {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeExecutionReport)
	msg.Set(TagClOrdID, "C001")
	msg.Set(TagOrderID, "ORD-123")
	msg.Set(TagExecID, "EXEC-001")
	msg.Set(TagExecType, "F")
	msg.Set(TagOrdStatus, "2")
	msg.Set(TagSymbol, "AAPL")
	msg.Set(TagSide, "1")
	msg.Set(TagLastQty, "100")
	msg.Set(TagLastPx, "150.25")
	msg.Set(TagCumQty, "100")
	msg.Set(TagAvgPx, "150.25")
	msg.Set(TagLeavesQty, "0")

	report := ParseExecutionReport(msg)
	if report.ClOrdID != "C001" {
		t.Errorf("ClOrdID: got %s", report.ClOrdID)
	}
	if report.OrdStatus != "2" {
		t.Errorf("OrdStatus: got %s", report.OrdStatus)
	}
	if report.LastQty != 100 {
		t.Errorf("LastQty: got %f", report.LastQty)
	}
	if report.AvgPx != 150.25 {
		t.Errorf("AvgPx: got %f", report.AvgPx)
	}
}

func TestComputeChecksum(t *testing.T) {
	// Known FIX 4.2 checksum example.
	// "8=FIX.4.2\x019=5\x0135=D\x0110=050\x01"
	// checksum is sum of all bytes mod 256.
	data := []byte("8=FIX.4.2\x0149=SENDER\x01")
	cs := ComputeChecksum(data)
	if cs < 0 || cs > 255 {
		t.Errorf("checksum out of range: %d", cs)
	}
}

func TestDecode_MalformedField(t *testing.T) {
	badData := []byte("notafield\x01")
	_, err := Decode(badData)
	if err == nil {
		t.Error("expected error for malformed field, got nil")
	}
}

func TestLogon_ResetFlag(t *testing.T) {
	msg := NewLogon(30, true)
	if msg.MsgType != MsgTypeLogon {
		t.Errorf("expected Logon MsgType")
	}
	if msg.Get(TagResetSeqNumFlag) != "Y" {
		t.Errorf("expected ResetSeqNumFlag=Y")
	}
	if msg.Get(TagHeartBtInt) != "30" {
		t.Errorf("expected HeartBtInt=30")
	}
}

func TestLogon_NoResetFlag(t *testing.T) {
	msg := NewLogon(60, false)
	if msg.Get(TagResetSeqNumFlag) != "" {
		t.Errorf("expected no ResetSeqNumFlag when reset=false")
	}
}

func TestHeartbeat_WithTestReqID(t *testing.T) {
	msg := NewHeartbeat("TR-001")
	if msg.MsgType != MsgTypeHeartbeat {
		t.Errorf("expected Heartbeat MsgType")
	}
	if msg.Get(TagTestReqID) != "TR-001" {
		t.Errorf("expected TestReqID=TR-001")
	}
}

func TestHeartbeat_Spontaneous(t *testing.T) {
	msg := NewHeartbeat("")
	if msg.Get(TagTestReqID) != "" {
		t.Errorf("spontaneous heartbeat should not have TestReqID")
	}
}

// -- Order router state machine tests --

// mockSession is a FIXSession stand-in that captures sent orders.
type mockSession struct {
	orders  []FIXOrder
	handler func(ExecutionReport)
}

func (m *mockSession) IsActive() bool { return true }

// TestOrderRouter_Submit verifies that pending orders are tracked correctly.
func TestOrderRouter_Submit_Pending(t *testing.T) {
	// We need to test the router logic without a real TCP session.
	// Use a thin wrapper that exercises the router's state tracking directly.
	session := NewFIXSession(FIXConfig{
		SenderCompID: "TEST",
		TargetCompID: "BROKER",
		Host:         "127.0.0.1",
		Port:         19999, // nothing listening
	})
	router := NewFIXOrderRouter(session)

	// Inject a pending order directly to bypass TCP.
	pending := &PendingOrder{
		ClOrdID:     "C-MANUAL",
		Symbol:      "BTC",
		Side:        "1",
		Qty:         1.0,
		Price:       50000.0,
		SubmittedAt: time.Now(),
		Status:      "pending",
	}
	router.pending.Store("C-MANUAL", pending)

	orders := router.PendingOrders()
	if len(orders) != 1 {
		t.Fatalf("expected 1 pending order, got %d", len(orders))
	}
	if orders[0].ClOrdID != "C-MANUAL" {
		t.Errorf("wrong ClOrdID: %s", orders[0].ClOrdID)
	}
}

// TestOrderRouter_OnExecutionReport_Filled verifies filled orders are removed.
func TestOrderRouter_OnExecutionReport_Filled(t *testing.T) {
	session := NewFIXSession(FIXConfig{SenderCompID: "T", TargetCompID: "B", Host: "127.0.0.1", Port: 19999})
	router := NewFIXOrderRouter(session)

	pending := &PendingOrder{
		ClOrdID:     "C-FILL",
		Symbol:      "ETH",
		Side:        "1",
		Qty:         5.0,
		SubmittedAt: time.Now(),
		Status:      "pending",
	}
	router.pending.Store("C-FILL", pending)

	report := ExecutionReport{
		ClOrdID:   "C-FILL",
		OrdStatus: "2", // filled
		CumQty:    5.0,
		AvgPx:     2000.0,
	}
	router.OnExecutionReport(report)

	orders := router.PendingOrders()
	if len(orders) != 0 {
		t.Errorf("expected 0 pending orders after fill, got %d", len(orders))
	}

	history := router.RecentHistory(10)
	if len(history) != 1 {
		t.Errorf("expected 1 history entry, got %d", len(history))
	}
	if history[0].CumQty != 5.0 {
		t.Errorf("history CumQty mismatch: %f", history[0].CumQty)
	}
}

// TestOrderRouter_OnExecutionReport_Cancelled verifies cancelled orders are removed.
func TestOrderRouter_OnExecutionReport_Cancelled(t *testing.T) {
	session := NewFIXSession(FIXConfig{SenderCompID: "T", TargetCompID: "B", Host: "127.0.0.1", Port: 19999})
	router := NewFIXOrderRouter(session)

	pending := &PendingOrder{
		ClOrdID:     "C-CANCEL",
		Symbol:      "SOL",
		Side:        "2",
		Qty:         10.0,
		SubmittedAt: time.Now(),
		Status:      "pending",
	}
	router.pending.Store("C-CANCEL", pending)

	router.OnExecutionReport(ExecutionReport{ClOrdID: "C-CANCEL", OrdStatus: "4"})

	if len(router.PendingOrders()) != 0 {
		t.Error("cancelled order should not remain pending")
	}
}

// TestOrderRouter_Timeout_Alert verifies the alert fires for stale orders.
func TestOrderRouter_Timeout_Alert(t *testing.T) {
	session := NewFIXSession(FIXConfig{SenderCompID: "T", TargetCompID: "B", Host: "127.0.0.1", Port: 19999})
	router := NewFIXOrderRouter(session)
	router.SetOrderTimeout(0) // 0 = always stale

	alerted := make(chan string, 1)
	router.SetAlertHandler(func(o PendingOrder) {
		alerted <- o.ClOrdID
	})

	router.pending.Store("C-STALE", &PendingOrder{
		ClOrdID:     "C-STALE",
		SubmittedAt: time.Now().Add(-1 * time.Minute),
		Status:      "pending",
	})

	router.scanTimeouts()

	select {
	case id := <-alerted:
		if id != "C-STALE" {
			t.Errorf("wrong clOrdID in alert: %s", id)
		}
	default:
		t.Error("expected alert for stale order")
	}
}

// TestSortInts verifies the internal sort helper.
func TestSortInts(t *testing.T) {
	a := []int{50, 35, 8, 9, 10, 49}
	sortInts(a)
	for i := 1; i < len(a); i++ {
		if a[i] < a[i-1] {
			t.Errorf("not sorted at index %d: %v", i, a)
		}
	}
}

// TestEncodeBodyLength verifies that the 9= field reflects body byte count.
func TestEncodeBodyLength(t *testing.T) {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeHeartbeat)
	msg.Set(TagSenderCompID, "SRFM")
	msg.Set(TagTargetCompID, "BRKR")
	msg.Set(TagMsgSeqNum, "5")
	msg.Set(TagSendingTime, "20240101-00:00:00.000")

	wire := Encode(msg)
	s := string(wire)

	// Parse 9= field.
	parts := strings.Split(s, SOH)
	var bodyLen int
	for _, p := range parts {
		if strings.HasPrefix(p, "9=") {
			bodyLen, _ = strconv.Atoi(strings.TrimPrefix(p, "9="))
			break
		}
	}
	if bodyLen <= 0 {
		t.Fatalf("could not parse BodyLength from encoded message")
	}
	// The body is everything between end of 9= field and start of 10= field.
	bodyStart := strings.Index(s, "9=")
	bodyStart = strings.Index(s[bodyStart:], SOH) + bodyStart + 1
	checksumIdx := strings.LastIndex(s, "10=")
	if checksumIdx < 0 {
		t.Fatal("no checksum field found")
	}
	actualBodyLen := len(s[bodyStart:checksumIdx])
	if actualBodyLen != bodyLen {
		t.Errorf("BodyLength=%d but actual body bytes=%d", bodyLen, actualBodyLen)
	}
}
