package main

import (
	"fmt"
	"strconv"
	"strings"
)

// SOH is the FIX field delimiter (ASCII 0x01).
const SOH = "\x01"

// FIX tag constants referenced throughout the gateway.
const (
	TagBeginString  = 8
	TagBodyLength   = 9
	TagMsgType      = 35
	TagSenderCompID = 49
	TagTargetCompID = 56
	TagMsgSeqNum    = 34
	TagSendingTime  = 52
	TagCheckSum     = 10

	// Session-level tags
	TagHeartBtInt    = 108
	TagResetSeqNumFlag = 141
	TagTestReqID     = 112
	TagText          = 58
	TagRefSeqNum     = 45
	TagGapFillFlag   = 123
	TagNewSeqNo      = 36

	// Order tags
	TagClOrdID      = 11
	TagOrigClOrdID  = 41
	TagSymbol       = 55
	TagSide         = 54
	TagOrderQty     = 38
	TagOrdType      = 40
	TagPrice        = 44
	TagTimeInForce  = 59
	TagTransactTime = 60
	TagHandlInst    = 21

	// Execution report tags
	TagOrderID      = 37
	TagExecID       = 17
	TagExecType     = 150
	TagOrdStatus    = 39
	TagLastQty      = 32
	TagLastPx       = 31
	TagCumQty       = 14
	TagAvgPx        = 6
	TagLeavesQty    = 151

	// Message types
	MsgTypeHeartbeat      = "0"
	MsgTypeTestRequest    = "1"
	MsgTypeResendRequest  = "2"
	MsgTypeReject         = "3"
	MsgTypeSequenceReset  = "4"
	MsgTypeLogout         = "5"
	MsgTypeExecutionReport = "8"
	MsgTypeLogon          = "A"
	MsgTypeNewOrderSingle = "D"
	MsgTypeOrderCancelRequest = "F"
)

// FIXMessage is the in-memory representation of a FIX message.
// Fields maps tag numbers to string values.
type FIXMessage struct {
	MsgType string
	Fields  map[int]string
}

// Get returns the string value of a FIX tag, or "" if absent.
func (m *FIXMessage) Get(tag int) string {
	return m.Fields[tag]
}

// Set sets a FIX tag value, initializing Fields if needed.
func (m *FIXMessage) Set(tag int, value string) {
	if m.Fields == nil {
		m.Fields = make(map[int]string)
	}
	m.Fields[tag] = value
	if tag == TagMsgType {
		m.MsgType = value
	}
}

// Encode serializes a FIXMessage to SOH-delimited wire bytes including
// BeginString(8), BodyLength(9), all body fields, and CheckSum(10).
// The caller must have populated all required session header tags.
func Encode(msg FIXMessage) []byte {
	// Build body: all fields except 8, 9, 10 in ascending tag order.
	var body strings.Builder

	// Collect and sort tags for deterministic output.
	tags := make([]int, 0, len(msg.Fields))
	for tag := range msg.Fields {
		if tag == TagBeginString || tag == TagBodyLength || tag == TagCheckSum {
			continue
		}
		tags = append(tags, tag)
	}
	sortInts(tags)

	for _, tag := range tags {
		body.WriteString(strconv.Itoa(tag))
		body.WriteByte('=')
		body.WriteString(msg.Fields[tag])
		body.WriteString(SOH)
	}

	bodyStr := body.String()

	// Prepend header fields 8 and 9.
	beginStr := "8=FIX.4.2" + SOH
	bodyLenStr := fmt.Sprintf("9=%d%s", len(bodyStr), SOH)

	// Compute checksum over 8= ... 9= ... body fields (excluding 10=).
	var sum int
	for _, b := range []byte(beginStr) {
		sum += int(b)
	}
	for _, b := range []byte(bodyLenStr) {
		sum += int(b)
	}
	for _, b := range []byte(bodyStr) {
		sum += int(b)
	}
	checksum := sum % 256

	checksumStr := fmt.Sprintf("10=%03d%s", checksum, SOH)

	return []byte(beginStr + bodyLenStr + bodyStr + checksumStr)
}

// Decode parses SOH-delimited FIX bytes into a FIXMessage.
// Returns an error if checksum validation fails.
func Decode(data []byte) (FIXMessage, error) {
	msg := FIXMessage{
		Fields: make(map[int]string),
	}

	parts := strings.Split(string(data), SOH)
	var checksumFromMsg int
	checksumTagSeen := false

	// Running checksum over all bytes except the "10=NNN" field itself.
	var checksumBytes int

	for _, part := range parts {
		if part == "" {
			continue
		}
		eqIdx := strings.IndexByte(part, '=')
		if eqIdx < 1 {
			return msg, fmt.Errorf("malformed field: %q", part)
		}
		tagStr := part[:eqIdx]
		value := part[eqIdx+1:]

		tag, err := strconv.Atoi(tagStr)
		if err != nil {
			return msg, fmt.Errorf("non-numeric tag: %q", tagStr)
		}

		if tag == TagCheckSum {
			checksumTagSeen = true
			checksumFromMsg, _ = strconv.Atoi(value)
			// Do not include checksum field in running sum.
		} else {
			// Add field bytes including the trailing SOH delimiter.
			fieldBytes := []byte(part + SOH)
			for _, b := range fieldBytes {
				checksumBytes += int(b)
			}
		}

		msg.Fields[tag] = value
		if tag == TagMsgType {
			msg.MsgType = value
		}
	}

	if checksumTagSeen {
		computed := checksumBytes % 256
		if computed != checksumFromMsg {
			return msg, fmt.Errorf("checksum mismatch: got %03d want %03d", checksumFromMsg, computed)
		}
	}

	return msg, nil
}

// NewOrderSingle builds a FIX 4.2 NewOrderSingle (35=D) message.
// side: "1"=buy, "2"=sell. ordType: "1"=market, "2"=limit.
func NewOrderSingle(clOrdID, symbol, side, qty, price, ordType, transactTime string) FIXMessage {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeNewOrderSingle)
	msg.Set(TagClOrdID, clOrdID)
	msg.Set(TagHandlInst, "1") // automated execution
	msg.Set(TagSymbol, symbol)
	msg.Set(TagSide, side)
	msg.Set(TagTransactTime, transactTime)
	msg.Set(TagOrderQty, qty)
	msg.Set(TagOrdType, ordType)
	if price != "" && ordType == "2" {
		msg.Set(TagPrice, price)
	}
	msg.Set(TagTimeInForce, "0") // DAY
	return msg
}

// NewOrderCancelRequest builds a FIX 4.2 OrderCancelRequest (35=F).
func NewOrderCancelRequest(clOrdID, origClOrdID, symbol, side, transactTime string) FIXMessage {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeOrderCancelRequest)
	msg.Set(TagClOrdID, clOrdID)
	msg.Set(TagOrigClOrdID, origClOrdID)
	msg.Set(TagSymbol, symbol)
	msg.Set(TagSide, side)
	msg.Set(TagTransactTime, transactTime)
	return msg
}

// NewLogon builds a FIX 4.2 Logon (35=A) message.
func NewLogon(heartbeatInterval int, resetSeqNum bool) FIXMessage {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeLogon)
	msg.Set(TagHeartBtInt, strconv.Itoa(heartbeatInterval))
	if resetSeqNum {
		msg.Set(TagResetSeqNumFlag, "Y")
	}
	return msg
}

// NewLogout builds a FIX 4.2 Logout (35=5).
func NewLogout(text string) FIXMessage {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeLogout)
	if text != "" {
		msg.Set(TagText, text)
	}
	return msg
}

// NewHeartbeat builds a FIX Heartbeat (35=0).
// Pass testReqID="" for spontaneous heartbeats.
func NewHeartbeat(testReqID string) FIXMessage {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeHeartbeat)
	if testReqID != "" {
		msg.Set(TagTestReqID, testReqID)
	}
	return msg
}

// NewTestRequest builds a FIX TestRequest (35=1).
func NewTestRequest(testReqID string) FIXMessage {
	msg := FIXMessage{Fields: make(map[int]string)}
	msg.Set(TagMsgType, MsgTypeTestRequest)
	msg.Set(TagTestReqID, testReqID)
	return msg
}

// ExecutionReport holds the parsed contents of a FIX ExecutionReport (35=8).
type ExecutionReport struct {
	ClOrdID   string
	OrderID   string
	ExecID    string
	ExecType  string
	OrdStatus string
	Symbol    string
	Side      string
	LastQty   float64
	LastPx    float64
	CumQty    float64
	AvgPx     float64
	LeavesQty float64
}

// ParseExecutionReport extracts an ExecutionReport from a FIX message.
func ParseExecutionReport(msg FIXMessage) ExecutionReport {
	return ExecutionReport{
		ClOrdID:   msg.Get(TagClOrdID),
		OrderID:   msg.Get(TagOrderID),
		ExecID:    msg.Get(TagExecID),
		ExecType:  msg.Get(TagExecType),
		OrdStatus: msg.Get(TagOrdStatus),
		Symbol:    msg.Get(TagSymbol),
		Side:      msg.Get(TagSide),
		LastQty:   parseFloat(msg.Get(TagLastQty)),
		LastPx:    parseFloat(msg.Get(TagLastPx)),
		CumQty:    parseFloat(msg.Get(TagCumQty)),
		AvgPx:     parseFloat(msg.Get(TagAvgPx)),
		LeavesQty: parseFloat(msg.Get(TagLeavesQty)),
	}
}

// parseFloat parses a string as float64, returning 0 on error.
func parseFloat(s string) float64 {
	if s == "" {
		return 0
	}
	v, _ := strconv.ParseFloat(s, 64)
	return v
}

// ComputeChecksum computes the FIX checksum (sum of all bytes mod 256)
// for the given raw message bytes.
func ComputeChecksum(data []byte) int {
	var sum int
	for _, b := range data {
		sum += int(b)
	}
	return sum % 256
}

// sortInts is a minimal in-place integer sort (insertion sort for small n).
func sortInts(a []int) {
	for i := 1; i < len(a); i++ {
		key := a[i]
		j := i - 1
		for j >= 0 && a[j] > key {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = key
	}
}
