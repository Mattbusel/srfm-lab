// Package wshub implements a scalable WebSocket broadcasting hub.
// protocol.go — Message protocol: typed message structs and JSON codec.
package wshub

import (
	"encoding/json"
	"fmt"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// Message types
// ─────────────────────────────────────────────────────────────────────────────

// MessageType discriminates the message kind.
type MessageType string

const (
	// Client → Server messages.
	MsgTypeSubscribe   MessageType = "subscribe"
	MsgTypeUnsubscribe MessageType = "unsubscribe"
	MsgTypePing        MessageType = "ping"
	MsgTypeAuth        MessageType = "auth"

	// Server → Client messages.
	MsgTypeSnapshot  MessageType = "snapshot"
	MsgTypeUpdate    MessageType = "update"
	MsgTypeError     MessageType = "error"
	MsgTypePong      MessageType = "pong"
	MsgTypeAck       MessageType = "ack"
	MsgTypeConnected MessageType = "connected"
	MsgTypeDisconnected MessageType = "disconnected"
	MsgTypeAlert     MessageType = "alert"

	// Bidirectional.
	MsgTypeHeartbeat MessageType = "heartbeat"
)

// ─────────────────────────────────────────────────────────────────────────────
// Envelope — the universal message wrapper
// ─────────────────────────────────────────────────────────────────────────────

// Message is the wire format for all WebSocket messages.
type Message struct {
	// Type identifies the message kind.
	Type MessageType `json:"type"`

	// ID is an optional client-provided request ID for correlation.
	ID string `json:"id,omitempty"`

	// Room is the room/channel this message targets.
	Room string `json:"room,omitempty"`

	// Sequence is a monotonic counter for ordered delivery detection.
	Sequence int64 `json:"seq,omitempty"`

	// Timestamp is the server-side message creation time.
	Timestamp time.Time `json:"ts"`

	// Payload contains the typed message body.
	Payload json.RawMessage `json:"payload,omitempty"`

	// Error contains error details (for MsgTypeError).
	Error *ErrorPayload `json:"error,omitempty"`
}

// ─────────────────────────────────────────────────────────────────────────────
// Typed payloads
// ─────────────────────────────────────────────────────────────────────────────

// SubscribePayload is sent by clients to subscribe to a room.
type SubscribePayload struct {
	Rooms  []string          `json:"rooms"`
	Params map[string]string `json:"params,omitempty"`
}

// UnsubscribePayload is sent by clients to leave a room.
type UnsubscribePayload struct {
	Rooms []string `json:"rooms"`
}

// AuthPayload is sent by clients for post-connect authentication.
type AuthPayload struct {
	Token  string `json:"token,omitempty"`
	APIKey string `json:"api_key,omitempty"`
}

// AckPayload confirms a client message was received.
type AckPayload struct {
	RequestID string `json:"request_id"`
	Success   bool   `json:"success"`
	Message   string `json:"message,omitempty"`
}

// ConnectedPayload is sent to clients on successful connection.
type ConnectedPayload struct {
	ClientID string   `json:"client_id"`
	ServerID string   `json:"server_id"`
	Rooms    []string `json:"rooms,omitempty"`
	Features []string `json:"features"`
}

// SnapshotPayload is a full data snapshot sent on subscription.
type SnapshotPayload struct {
	Room      string          `json:"room"`
	DataType  string          `json:"data_type"`
	Data      json.RawMessage `json:"data"`
	Timestamp time.Time       `json:"timestamp"`
}

// UpdatePayload is an incremental data update.
type UpdatePayload struct {
	Room      string          `json:"room"`
	DataType  string          `json:"data_type"`
	Data      json.RawMessage `json:"data"`
	IsDelta   bool            `json:"is_delta,omitempty"`
	Sequence  int64           `json:"sequence"`
	Timestamp time.Time       `json:"timestamp"`
}

// ErrorPayload contains error details.
type ErrorPayload struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Detail  string `json:"detail,omitempty"`
}

// HeartbeatPayload is exchanged to detect dead connections.
type HeartbeatPayload struct {
	ServerTime time.Time `json:"server_time"`
	ClientTime time.Time `json:"client_time,omitempty"`
	RTT        int64     `json:"rtt_us,omitempty"` // round-trip time if echoed
}

// AlertPayload is a server-push alert message.
type AlertPayload struct {
	Level   string `json:"level"` // "info", "warning", "critical"
	Title   string `json:"title"`
	Body    string `json:"body"`
	Symbol  string `json:"symbol,omitempty"`
	RuleID  string `json:"rule_id,omitempty"`
}

// ─────────────────────────────────────────────────────────────────────────────
// Market data payloads (echoed from event bus)
// ─────────────────────────────────────────────────────────────────────────────

// BarUpdate is sent to clients subscribed to a symbol's bar room.
type BarUpdate struct {
	Symbol    string    `json:"symbol"`
	Timeframe string    `json:"timeframe"`
	Open      float64   `json:"o"`
	High      float64   `json:"h"`
	Low       float64   `json:"l"`
	Close     float64   `json:"c"`
	Volume    float64   `json:"v"`
	Vwap      float64   `json:"vwap,omitempty"`
	Timestamp time.Time `json:"t"`
	IsPartial bool      `json:"partial,omitempty"`
}

// QuoteUpdate is sent to clients subscribed to a symbol's quote room.
type QuoteUpdate struct {
	Symbol    string    `json:"symbol"`
	Bid       float64   `json:"bid"`
	Ask       float64   `json:"ask"`
	BidSize   float64   `json:"bid_sz"`
	AskSize   float64   `json:"ask_sz"`
	Mid       float64   `json:"mid"`
	SpreadBps float64   `json:"spread_bps"`
	Timestamp time.Time `json:"t"`
}

// SignalUpdate is sent to clients subscribed to a strategy signal room.
type SignalUpdate struct {
	Symbol          string    `json:"symbol"`
	StrategyID      string    `json:"strategy_id"`
	Timeframe       string    `json:"timeframe"`
	Direction       string    `json:"direction"`
	Confidence      float64   `json:"confidence"`
	Strength        string    `json:"strength"`
	BHBullScore     float64   `json:"bh_bull"`
	BHBearScore     float64   `json:"bh_bear"`
	BHHawkScore     float64   `json:"bh_hawk"`
	BHDoveScore     float64   `json:"bh_dove"`
	Regime          string    `json:"regime"`
	Timestamp       time.Time `json:"t"`
}

// RiskBreachUpdate is sent to clients subscribed to a risk room.
type RiskBreachUpdate struct {
	AccountID    string    `json:"account_id"`
	RuleID       string    `json:"rule_id"`
	EventType    string    `json:"event_type"`
	Severity     string    `json:"severity"`
	Symbol       string    `json:"symbol,omitempty"`
	Description  string    `json:"description"`
	CurrentValue float64   `json:"current_value"`
	LimitValue   float64   `json:"limit_value"`
	OccurredAt   time.Time `json:"occurred_at"`
}

// ─────────────────────────────────────────────────────────────────────────────
// Room naming conventions
// ─────────────────────────────────────────────────────────────────────────────

// Room name helpers.
func BarRoom(symbol, timeframe string) string {
	return fmt.Sprintf("bars:%s:%s", symbol, timeframe)
}

func QuoteRoom(symbol string) string {
	return fmt.Sprintf("quotes:%s", symbol)
}

func SignalRoom(symbol, strategyID string) string {
	return fmt.Sprintf("signals:%s:%s", symbol, strategyID)
}

func RiskRoom(accountID string) string {
	return fmt.Sprintf("risk:%s", accountID)
}

func PortfolioRoom(accountID string) string {
	return fmt.Sprintf("portfolio:%s", accountID)
}

func OrderBookRoom(symbol string) string {
	return fmt.Sprintf("book:%s", symbol)
}

// ─────────────────────────────────────────────────────────────────────────────
// Codec — encode/decode Message
// ─────────────────────────────────────────────────────────────────────────────

// Codec encodes and decodes wire messages.
type Codec struct{}

// NewCodec creates a Codec.
func NewCodec() *Codec { return &Codec{} }

// Encode serialises a Message to JSON bytes.
func (c *Codec) Encode(msg *Message) ([]byte, error) {
	if msg.Timestamp.IsZero() {
		msg.Timestamp = time.Now().UTC()
	}
	data, err := json.Marshal(msg)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	return data, nil
}

// Decode deserialises JSON bytes into a Message.
func (c *Codec) Decode(data []byte) (*Message, error) {
	var msg Message
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &msg, nil
}

// NewMessage constructs a Message with a typed payload.
func NewMessage(msgType MessageType, room, id string, payload interface{}) (*Message, error) {
	var raw json.RawMessage
	if payload != nil {
		b, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal payload: %w", err)
		}
		raw = b
	}
	return &Message{
		Type:      msgType,
		ID:        id,
		Room:      room,
		Timestamp: time.Now().UTC(),
		Payload:   raw,
	}, nil
}

// NewErrorMessage constructs an error Message.
func NewErrorMessage(requestID string, code int, msg, detail string) *Message {
	return &Message{
		Type:      MsgTypeError,
		ID:        requestID,
		Timestamp: time.Now().UTC(),
		Error:     &ErrorPayload{Code: code, Message: msg, Detail: detail},
	}
}

// DecodePayload unmarshals a message's Payload into a typed struct.
func DecodePayload[T any](msg *Message) (*T, error) {
	if len(msg.Payload) == 0 {
		return nil, fmt.Errorf("empty payload")
	}
	var t T
	if err := json.Unmarshal(msg.Payload, &t); err != nil {
		return nil, fmt.Errorf("decode payload as %T: %w", t, err)
	}
	return &t, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Error codes
// ─────────────────────────────────────────────────────────────────────────────

const (
	ErrCodeBadRequest     = 400
	ErrCodeUnauthorized   = 401
	ErrCodeForbidden      = 403
	ErrCodeNotFound       = 404
	ErrCodeRateLimit      = 429
	ErrCodeInternal       = 500
	ErrCodeServiceUnavail = 503
)
