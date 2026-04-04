package dashboard

import (
	"encoding/json"
	"time"

	"github.com/srfm/monitor/internal/alerting"
)

// EventType classifies SSE event payloads.
type EventType string

const (
	EventTypePortfolio EventType = "portfolio"
	EventTypeAlert     EventType = "alert"
	EventTypeBHUpdate  EventType = "bh_update"
	EventTypeHeartbeat EventType = "heartbeat"
	EventTypeState     EventType = "state"
)

// SSEEvent is the top-level envelope for SSE messages.
type SSEEvent struct {
	Type      EventType   `json:"type"`
	Timestamp time.Time   `json:"timestamp"`
	Data      interface{} `json:"data"`
}

// PortfolioEvent carries a portfolio state snapshot.
type PortfolioEvent struct {
	Equity        float64                       `json:"equity"`
	Cash          float64                       `json:"cash"`
	DailyPnL      float64                       `json:"daily_pnl"`
	IntradayDD    float64                       `json:"intraday_dd"`
	HighWaterMark float64                       `json:"high_water_mark"`
	TotalExposure float64                       `json:"total_exposure"`
	Positions     map[string]alerting.Position  `json:"positions"`
}

// AlertEvent carries a fired alert.
type AlertEvent struct {
	ID        string             `json:"id"`
	Rule      string             `json:"rule"`
	Level     alerting.AlertLevel `json:"level"`
	Symbol    string             `json:"symbol"`
	Metric    string             `json:"metric"`
	Value     float64            `json:"value"`
	Threshold float64            `json:"threshold"`
	Message   string             `json:"message"`
	FiredAt   time.Time          `json:"fired_at"`
}

// BHUpdateEvent carries a BH mass update.
type BHUpdateEvent struct {
	Symbol    string  `json:"symbol"`
	Timeframe string  `json:"timeframe"`
	Mass      float64 `json:"mass"`
	EventType string  `json:"event_type"` // "formation", "collapse", "mass_spike", "update"
}

// BuildPortfolioEvent creates an SSEEvent from a PortfolioState.
func BuildPortfolioEvent(state alerting.PortfolioState) SSEEvent {
	return SSEEvent{
		Type:      EventTypePortfolio,
		Timestamp: state.Timestamp,
		Data: PortfolioEvent{
			Equity:        state.Equity,
			Cash:          state.Cash,
			DailyPnL:      state.DailyPnL,
			IntradayDD:    state.IntradayDD,
			HighWaterMark: state.HighWaterMark,
			TotalExposure: state.TotalExposure,
			Positions:     state.Positions,
		},
	}
}

// BuildAlertEvent creates an SSEEvent from a fired Alert.
func BuildAlertEvent(a alerting.Alert) SSEEvent {
	return SSEEvent{
		Type:      EventTypeAlert,
		Timestamp: a.FiredAt,
		Data: AlertEvent{
			ID:        a.ID,
			Rule:      a.Rule.Name,
			Level:     a.Level,
			Symbol:    a.Symbol,
			Metric:    a.Metric,
			Value:     a.Value,
			Threshold: a.Threshold,
			Message:   a.Message,
			FiredAt:   a.FiredAt,
		},
	}
}

// BuildBHEvent creates an SSEEvent from a BHEvent.
func BuildBHEvent(evt alerting.BHEvent) SSEEvent {
	return SSEEvent{
		Type:      EventTypeBHUpdate,
		Timestamp: evt.Timestamp,
		Data: BHUpdateEvent{
			Symbol:    evt.Symbol,
			Timeframe: evt.Timeframe,
			Mass:      evt.Mass,
			EventType: evt.EventType,
		},
	}
}

// BuildHeartbeat creates a heartbeat SSE event.
func BuildHeartbeat() SSEEvent {
	return SSEEvent{
		Type:      EventTypeHeartbeat,
		Timestamp: time.Now(),
		Data:      map[string]string{"status": "alive"},
	}
}

// MarshalSSEEvent serialises an SSEEvent to a JSON string.
func MarshalSSEEvent(e SSEEvent) (string, error) {
	b, err := json.Marshal(e)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// SSEBroadcaster manages SSE client connections and delivers typed events.
type SSEBroadcaster struct {
	server *Server
}

// NewSSEBroadcaster creates an SSEBroadcaster backed by the given Server.
func NewSSEBroadcaster(s *Server) *SSEBroadcaster {
	return &SSEBroadcaster{server: s}
}

// Portfolio delivers a portfolio update to all SSE clients.
func (b *SSEBroadcaster) Portfolio(state alerting.PortfolioState) {
	evt := BuildPortfolioEvent(state)
	b.server.PushSSE(evt)
}

// Alert delivers an alert to all SSE clients.
func (b *SSEBroadcaster) Alert(a alerting.Alert) {
	evt := BuildAlertEvent(a)
	b.server.PushSSE(evt)
}

// BHUpdate delivers a BH event to all SSE clients.
func (b *SSEBroadcaster) BHUpdate(evt alerting.BHEvent) {
	e := BuildBHEvent(evt)
	b.server.PushSSE(e)
}

// Heartbeat sends a keep-alive to all SSE clients.
func (b *SSEBroadcaster) Heartbeat() {
	b.server.PushSSE(BuildHeartbeat())
}
