// publisher.go — Typed publishers for each domain event category.
package eventbus

import (
	"context"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// Typed event payloads
// ─────────────────────────────────────────────────────────────────────────────

// BarEvent is published when an OHLCV bar closes (or updates mid-bar).
type BarEvent struct {
	Symbol    string    `json:"symbol" msgpack:"symbol"`
	Timeframe string    `json:"timeframe" msgpack:"timeframe"`
	Timestamp time.Time `json:"timestamp" msgpack:"timestamp"`
	Open      float64   `json:"open" msgpack:"open"`
	High      float64   `json:"high" msgpack:"high"`
	Low       float64   `json:"low" msgpack:"low"`
	Close     float64   `json:"close" msgpack:"close"`
	Volume    float64   `json:"volume" msgpack:"volume"`
	Vwap      float64   `json:"vwap,omitempty" msgpack:"vwap,omitempty"`
	Trades    int64     `json:"trades,omitempty" msgpack:"trades,omitempty"`
	IsPartial bool      `json:"is_partial,omitempty" msgpack:"is_partial,omitempty"`
}

// QuoteEvent is published on NBBO quote changes.
type QuoteEvent struct {
	Symbol    string    `json:"symbol" msgpack:"symbol"`
	Bid       float64   `json:"bid" msgpack:"bid"`
	Ask       float64   `json:"ask" msgpack:"ask"`
	BidSize   float64   `json:"bid_size" msgpack:"bid_size"`
	AskSize   float64   `json:"ask_size" msgpack:"ask_size"`
	Mid       float64   `json:"mid" msgpack:"mid"`
	SpreadBps float64   `json:"spread_bps" msgpack:"spread_bps"`
	Last      float64   `json:"last" msgpack:"last"`
	Timestamp time.Time `json:"timestamp" msgpack:"timestamp"`
}

// SignalEvent is published when a BH signal is computed.
type SignalEvent struct {
	Symbol          string    `json:"symbol" msgpack:"symbol"`
	StrategyID      string    `json:"strategy_id" msgpack:"strategy_id"`
	Timeframe       string    `json:"timeframe" msgpack:"timeframe"`
	Direction       string    `json:"direction" msgpack:"direction"` // "long", "short", "flat"
	Confidence      float64   `json:"confidence" msgpack:"confidence"`
	Strength        string    `json:"strength" msgpack:"strength"`
	EntryPrice      float64   `json:"entry_price" msgpack:"entry_price"`
	StopLoss        float64   `json:"stop_loss" msgpack:"stop_loss"`
	TakeProfit      float64   `json:"take_profit" msgpack:"take_profit"`
	PositionSizePct float64   `json:"position_size_pct" msgpack:"position_size_pct"`
	BHBullScore     float64   `json:"bh_bull_score" msgpack:"bh_bull_score"`
	BHBearScore     float64   `json:"bh_bear_score" msgpack:"bh_bear_score"`
	BHHawkScore     float64   `json:"bh_hawk_score" msgpack:"bh_hawk_score"`
	BHDoveScore     float64   `json:"bh_dove_score" msgpack:"bh_dove_score"`
	Regime          string    `json:"regime" msgpack:"regime"`
	Rationale       string    `json:"rationale" msgpack:"rationale"`
	Timestamp       time.Time `json:"timestamp" msgpack:"timestamp"`
}

// TradeEvent is published when a trade is executed.
type TradeEvent struct {
	ExecutionID string    `json:"execution_id" msgpack:"execution_id"`
	OrderID     string    `json:"order_id" msgpack:"order_id"`
	AccountID   string    `json:"account_id" msgpack:"account_id"`
	Symbol      string    `json:"symbol" msgpack:"symbol"`
	Side        string    `json:"side" msgpack:"side"` // "buy", "sell", "sell_short"
	Quantity    float64   `json:"quantity" msgpack:"quantity"`
	FillPrice   float64   `json:"fill_price" msgpack:"fill_price"`
	Commission  float64   `json:"commission" msgpack:"commission"`
	StrategyID  string    `json:"strategy_id" msgpack:"strategy_id"`
	ExecutedAt  time.Time `json:"executed_at" msgpack:"executed_at"`
}

// RiskBreachEvent is published when a risk limit is breached.
type RiskBreachEvent struct {
	EventID      string    `json:"event_id" msgpack:"event_id"`
	AccountID    string    `json:"account_id" msgpack:"account_id"`
	RuleID       string    `json:"rule_id" msgpack:"rule_id"`
	EventType    string    `json:"event_type" msgpack:"event_type"`
	Severity     string    `json:"severity" msgpack:"severity"`
	Symbol       string    `json:"symbol,omitempty" msgpack:"symbol,omitempty"`
	Description  string    `json:"description" msgpack:"description"`
	CurrentValue float64   `json:"current_value" msgpack:"current_value"`
	LimitValue   float64   `json:"limit_value" msgpack:"limit_value"`
	BreachPct    float64   `json:"breach_pct" msgpack:"breach_pct"`
	OccurredAt   time.Time `json:"occurred_at" msgpack:"occurred_at"`
}

// PortfolioUpdateEvent is published on position / equity changes.
type PortfolioUpdateEvent struct {
	AccountID     string    `json:"account_id" msgpack:"account_id"`
	Symbol        string    `json:"symbol,omitempty" msgpack:"symbol,omitempty"`
	EventType     string    `json:"event_type" msgpack:"event_type"` // "position_opened", "position_closed", etc.
	Equity        float64   `json:"equity" msgpack:"equity"`
	UnrealizedPnL float64   `json:"unrealized_pnl" msgpack:"unrealized_pnl"`
	RealizedPnL   float64   `json:"realized_pnl" msgpack:"realized_pnl"`
	DayPnL        float64   `json:"day_pnl" msgpack:"day_pnl"`
	UpdatedAt     time.Time `json:"updated_at" msgpack:"updated_at"`
}

// OrderBookEvent is published on L2 order book changes.
type OrderBookEvent struct {
	Symbol    string           `json:"symbol" msgpack:"symbol"`
	Timestamp time.Time        `json:"timestamp" msgpack:"timestamp"`
	MidPrice  float64          `json:"mid_price" msgpack:"mid_price"`
	SpreadBps float64          `json:"spread_bps" msgpack:"spread_bps"`
	Bids      []OrderBookLevel `json:"bids" msgpack:"bids"`
	Asks      []OrderBookLevel `json:"asks" msgpack:"asks"`
	IsDelta   bool             `json:"is_delta" msgpack:"is_delta"`
}

// OrderBookLevel is a single price level in the order book.
type OrderBookLevel struct {
	Price  float64 `json:"price" msgpack:"price"`
	Size   float64 `json:"size" msgpack:"size"`
	Orders int32   `json:"orders" msgpack:"orders"`
}

// ─────────────────────────────────────────────────────────────────────────────
// BarPublisher
// ─────────────────────────────────────────────────────────────────────────────

// BarPublisher publishes OHLCV bar events to the bus.
type BarPublisher struct {
	bus    *EventBus
	source string
}

// NewBarPublisher creates a BarPublisher for the given service name.
func NewBarPublisher(bus *EventBus, source string) *BarPublisher {
	return &BarPublisher{bus: bus, source: source}
}

// PublishBar publishes a single bar event.
func (p *BarPublisher) PublishBar(ctx context.Context, bar *BarEvent) error {
	topic := ForSymbol(TopicMarketBars, bar.Symbol)
	evt, err := NewEvent(topic, "bar.closed", bar, p.source)
	if err != nil {
		return err
	}
	evt.Metadata = map[string]string{
		"symbol":    bar.Symbol,
		"timeframe": bar.Timeframe,
	}
	if bar.IsPartial {
		evt.Type = "bar.partial"
	}
	return p.bus.Publish(ctx, evt)
}

// PublishBars publishes a batch of bar events.
func (p *BarPublisher) PublishBars(ctx context.Context, bars []*BarEvent) error {
	for _, bar := range bars {
		if err := p.PublishBar(ctx, bar); err != nil {
			return err
		}
	}
	return nil
}

// PublishQuote publishes an NBBO quote event.
func (p *BarPublisher) PublishQuote(ctx context.Context, q *QuoteEvent) error {
	topic := ForSymbol(TopicMarketQuotes, q.Symbol)
	evt, err := NewEvent(topic, "quote.updated", q, p.source)
	if err != nil {
		return err
	}
	return p.bus.Publish(ctx, evt)
}

// PublishOrderBook publishes an order book snapshot/delta.
func (p *BarPublisher) PublishOrderBook(ctx context.Context, book *OrderBookEvent) error {
	topic := ForSymbol(TopicMarketBook, book.Symbol)
	evtType := "orderbook.snapshot"
	if book.IsDelta {
		evtType = "orderbook.delta"
	}
	evt, err := NewEvent(topic, evtType, book, p.source)
	if err != nil {
		return err
	}
	return p.bus.Publish(ctx, evt)
}

// ─────────────────────────────────────────────────────────────────────────────
// SignalPublisher
// ─────────────────────────────────────────────────────────────────────────────

// SignalPublisher publishes BH signals and state events.
type SignalPublisher struct {
	bus    *EventBus
	source string
}

// NewSignalPublisher creates a SignalPublisher.
func NewSignalPublisher(bus *EventBus, source string) *SignalPublisher {
	return &SignalPublisher{bus: bus, source: source}
}

// PublishSignal publishes a signal event.
func (p *SignalPublisher) PublishSignal(ctx context.Context, sig *SignalEvent) error {
	topic := ForSymbol(TopicBHSignal, sig.Symbol)
	evt, err := NewEvent(topic, "signal.generated", sig, p.source)
	if err != nil {
		return err
	}
	evt.Metadata = map[string]string{
		"symbol":      sig.Symbol,
		"strategy_id": sig.StrategyID,
		"timeframe":   sig.Timeframe,
		"direction":   sig.Direction,
	}
	return p.bus.Publish(ctx, evt)
}

// PublishSignalAsync publishes a signal event asynchronously.
func (p *SignalPublisher) PublishSignalAsync(sig *SignalEvent) {
	topic := ForSymbol(TopicBHSignal, sig.Symbol)
	evt, err := NewEvent(topic, "signal.generated", sig, p.source)
	if err != nil {
		return
	}
	p.bus.PublishAsync(evt)
}

// ─────────────────────────────────────────────────────────────────────────────
// TradePublisher
// ─────────────────────────────────────────────────────────────────────────────

// TradePublisher publishes trade execution events.
type TradePublisher struct {
	bus    *EventBus
	source string
}

// NewTradePublisher creates a TradePublisher.
func NewTradePublisher(bus *EventBus, source string) *TradePublisher {
	return &TradePublisher{bus: bus, source: source}
}

// PublishTradeExecuted publishes a trade executed event.
func (p *TradePublisher) PublishTradeExecuted(ctx context.Context, trade *TradeEvent) error {
	evt, err := NewEvent(TopicTradeExecuted, "trade.executed", trade, p.source)
	if err != nil {
		return err
	}
	evt.Metadata = map[string]string{
		"account_id":   trade.AccountID,
		"symbol":       trade.Symbol,
		"side":         trade.Side,
		"execution_id": trade.ExecutionID,
	}
	return p.bus.Publish(ctx, evt)
}

// PublishOrderFilled publishes an order filled event.
func (p *TradePublisher) PublishOrderFilled(ctx context.Context, trade *TradeEvent) error {
	evt, err := NewEvent(TopicOrderFilled, "order.filled", trade, p.source)
	if err != nil {
		return err
	}
	return p.bus.Publish(ctx, evt)
}

// PublishOrderSubmitted publishes an order submitted event.
func (p *TradePublisher) PublishOrderSubmitted(ctx context.Context, trade *TradeEvent) error {
	evt, err := NewEvent(TopicOrderSubmitted, "order.submitted", trade, p.source)
	if err != nil {
		return err
	}
	return p.bus.Publish(ctx, evt)
}

// ─────────────────────────────────────────────────────────────────────────────
// RiskPublisher
// ─────────────────────────────────────────────────────────────────────────────

// RiskPublisher publishes risk-related events.
type RiskPublisher struct {
	bus    *EventBus
	source string
}

// NewRiskPublisher creates a RiskPublisher.
func NewRiskPublisher(bus *EventBus, source string) *RiskPublisher {
	return &RiskPublisher{bus: bus, source: source}
}

// PublishRiskBreach publishes a risk breach event.
func (p *RiskPublisher) PublishRiskBreach(ctx context.Context, breach *RiskBreachEvent) error {
	evt, err := NewEvent(TopicRiskBreach, "risk.breach", breach, p.source)
	if err != nil {
		return err
	}
	evt.Metadata = map[string]string{
		"account_id": breach.AccountID,
		"rule_id":    breach.RuleID,
		"severity":   breach.Severity,
	}
	// Also publish to per-account topic.
	acctTopic := ForAccount(TopicRiskEvent, breach.AccountID)
	acctEvt, err := NewEvent(acctTopic, "risk.event", breach, p.source)
	if err != nil {
		return err
	}

	if err := p.bus.Publish(ctx, evt); err != nil {
		return err
	}
	return p.bus.Publish(ctx, acctEvt)
}

// PublishMarginCall publishes a margin call event.
func (p *RiskPublisher) PublishMarginCall(ctx context.Context, breach *RiskBreachEvent) error {
	evt, err := NewEvent(TopicMarginCall, "risk.margin_call", breach, p.source)
	if err != nil {
		return err
	}
	return p.bus.Publish(ctx, evt)
}

// PublishPortfolioUpdate publishes a portfolio update event.
func (p *RiskPublisher) PublishPortfolioUpdate(ctx context.Context, update *PortfolioUpdateEvent) error {
	topic := ForAccount(TopicPortfolioUpdate, update.AccountID)
	evt, err := NewEvent(topic, "portfolio.updated", update, p.source)
	if err != nil {
		return err
	}
	return p.bus.Publish(ctx, evt)
}
