package hub

import (
	"encoding/json"
	"fmt"
	"time"
)

// MessageType identifies the kind of WS message.
type MessageType string

const (
	// Outbound message types.
	MsgTypeBar     MessageType = "bar"
	MsgTypeTrade   MessageType = "trade"
	MsgTypeQuote   MessageType = "quote"
	MsgTypeWelcome MessageType = "welcome"
	MsgTypeError   MessageType = "error"
	MsgTypePong    MessageType = "pong"
	MsgTypeStats   MessageType = "stats"

	// Inbound control message types.
	MsgTypeSubscribe   MessageType = "subscribe"
	MsgTypeUnsubscribe MessageType = "unsubscribe"
	MsgTypePing        MessageType = "ping"
	MsgTypeGetStats    MessageType = "get_stats"
)

// BaseMessage is the common envelope for all WS messages.
type BaseMessage struct {
	Type      MessageType `json:"type"`
	Timestamp time.Time   `json:"ts"`
	ID        string      `json:"id,omitempty"`
}

// BarMessage carries OHLCV data.
type BarMessage struct {
	BaseMessage
	Symbol    string  `json:"symbol"`
	Timeframe string  `json:"timeframe"`
	Open      float64 `json:"o"`
	High      float64 `json:"h"`
	Low       float64 `json:"l"`
	Close     float64 `json:"c"`
	Volume    float64 `json:"v"`
	Source    string  `json:"src,omitempty"`
	Partial   bool    `json:"partial,omitempty"`
}

// TradeMessage carries trade execution data.
type TradeMessage struct {
	BaseMessage
	Symbol string  `json:"symbol"`
	Price  float64 `json:"price"`
	Size   float64 `json:"size"`
	Side   string  `json:"side,omitempty"`
	Source string  `json:"src,omitempty"`
}

// QuoteMessage carries best bid/ask data.
type QuoteMessage struct {
	BaseMessage
	Symbol   string  `json:"symbol"`
	BidPrice float64 `json:"bid"`
	BidSize  float64 `json:"bidsz"`
	AskPrice float64 `json:"ask"`
	AskSize  float64 `json:"asksz"`
	Source   string  `json:"src,omitempty"`
}

// WelcomeMessage is sent immediately after a client connects.
type WelcomeMessagePayload struct {
	BaseMessage
	SubscriberID  string   `json:"subscriber_id"`
	ActiveSymbols []string `json:"active_symbols"`
	ServerTime    time.Time `json:"server_time"`
}

// ErrorMessage signals an error to the client.
type ErrorMessage struct {
	BaseMessage
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// StatsMessage carries subscription statistics.
type StatsMessage struct {
	BaseMessage
	Subscribers int     `json:"subscribers"`
	BarsPerSec  float64 `json:"bars_per_sec"`
}

// ControlMessage is an inbound subscription control message.
type ControlMessage struct {
	Type       MessageType `json:"type"`
	Symbols    []string    `json:"symbols,omitempty"`
	Timeframes []string    `json:"timeframes,omitempty"`
	EventTypes []string    `json:"event_types,omitempty"`
	Partial    bool        `json:"partial,omitempty"`
	ID         string      `json:"id,omitempty"`
}

// MarshalBarMessage serialises a BarMessage.
func MarshalBarMessage(msg BarMessage) ([]byte, error) {
	return json.Marshal(msg)
}

// MarshalTradeMessage serialises a TradeMessage.
func MarshalTradeMessage(msg TradeMessage) ([]byte, error) {
	return json.Marshal(msg)
}

// MarshalQuoteMessage serialises a QuoteMessage.
func MarshalQuoteMessage(msg QuoteMessage) ([]byte, error) {
	return json.Marshal(msg)
}

// MarshalError serialises an ErrorMessage.
func MarshalError(code int, message string) []byte {
	msg := ErrorMessage{
		BaseMessage: BaseMessage{Type: MsgTypeError, Timestamp: time.Now()},
		Code:        code,
		Message:     message,
	}
	data, _ := json.Marshal(msg)
	return data
}

// MarshalWelcome serialises a WelcomeMessage.
func MarshalWelcome(subID uint64, symbols []string) []byte {
	msg := WelcomeMessagePayload{
		BaseMessage:   BaseMessage{Type: MsgTypeWelcome, Timestamp: time.Now()},
		SubscriberID:  fmt.Sprintf("%d", subID),
		ActiveSymbols: symbols,
		ServerTime:    time.Now(),
	}
	data, _ := json.Marshal(msg)
	return data
}

// ParseControlMessage parses an inbound client message.
func ParseControlMessage(raw []byte) (*ControlMessage, error) {
	var msg ControlMessage
	if err := json.Unmarshal(raw, &msg); err != nil {
		return nil, fmt.Errorf("parse control message: %w", err)
	}
	return &msg, nil
}

// NormalizeEventTypes ensures event_types defaults to ["bar"] if empty.
func NormalizeEventTypes(types []string) []string {
	if len(types) == 0 {
		return []string{string(MsgTypeBar)}
	}
	return types
}
