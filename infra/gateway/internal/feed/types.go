package feed

import "time"

// Bar represents an OHLCV candlestick bar.
type Bar struct {
	Symbol    string
	Timestamp time.Time
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	Source    string
	// IsPartial indicates the bar is still in progress (live candle).
	IsPartial bool
}

// Trade represents a single executed trade.
type Trade struct {
	Symbol    string
	Timestamp time.Time
	Price     float64
	Size      float64
	Side      string // "buy" | "sell" | ""
	Source    string
}

// Quote represents a best bid/ask snapshot.
type Quote struct {
	Symbol    string
	Timestamp time.Time
	BidPrice  float64
	BidSize   float64
	AskPrice  float64
	AskSize   float64
	Source    string
}

// Event is a discriminated union emitted by feeds.
type EventKind int

const (
	EventBar EventKind = iota
	EventTrade
	EventQuote
)

type Event struct {
	Kind  EventKind
	Bar   *Bar
	Trade *Trade
	Quote *Quote
}
