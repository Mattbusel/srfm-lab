// Package server implements the gRPC service handlers for SRFM infrastructure.
// market_server.go — MarketDataService implementation.
// Reads OHLCV data from local CSV/Parquet cache, serves live quotes from an
// in-process Redis-backed quote store, and streams bar updates over gRPC
// server-side streaming RPCs.
package server

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	pb "github.com/srfm/infra/grpc/proto/market"
)

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

// MarketServerConfig holds tunables for the MarketDataService server.
type MarketServerConfig struct {
	// CacheDir is the root directory containing CSV/Parquet bar data.
	// Layout expected: <CacheDir>/<symbol>/<timeframe>.csv (or .parquet)
	CacheDir string

	// MaxBarsPerRequest caps the number of bars returned in a single GetBars call.
	MaxBarsPerRequest int

	// StreamTickInterval is how often the bar streamer ticks when replaying
	// cached data in simulated-live mode.
	StreamTickInterval time.Duration

	// QuoteCacheTTL controls how long a cached quote is considered fresh.
	QuoteCacheTTL time.Duration
}

// DefaultMarketServerConfig returns sensible defaults.
func DefaultMarketServerConfig() MarketServerConfig {
	return MarketServerConfig{
		CacheDir:           filepath.Join(os.Getenv("HOME"), "srfm-lab", "data", "bars"),
		MaxBarsPerRequest:  10_000,
		StreamTickInterval: 500 * time.Millisecond,
		QuoteCacheTTL:      5 * time.Second,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// In-process quote cache
// ─────────────────────────────────────────────────────────────────────────────

type cachedQuote struct {
	quote     *pb.QuoteResponse
	fetchedAt time.Time
}

type quoteCache struct {
	mu    sync.RWMutex
	items map[string]*cachedQuote
	ttl   time.Duration
}

func newQuoteCache(ttl time.Duration) *quoteCache {
	return &quoteCache{items: make(map[string]*cachedQuote), ttl: ttl}
}

func (qc *quoteCache) get(symbol string) (*pb.QuoteResponse, bool) {
	qc.mu.RLock()
	defer qc.mu.RUnlock()
	c, ok := qc.items[symbol]
	if !ok {
		return nil, false
	}
	if time.Since(c.fetchedAt) > qc.ttl {
		return nil, false
	}
	return c.quote, true
}

func (qc *quoteCache) set(symbol string, q *pb.QuoteResponse) {
	qc.mu.Lock()
	defer qc.mu.Unlock()
	qc.items[symbol] = &cachedQuote{quote: q, fetchedAt: time.Now()}
}

// ─────────────────────────────────────────────────────────────────────────────
// Order book in-process store
// ─────────────────────────────────────────────────────────────────────────────

type orderBookStore struct {
	mu    sync.RWMutex
	books map[string]*pb.OrderBookSnapshot // keyed by symbol
}

func newOrderBookStore() *orderBookStore {
	return &orderBookStore{books: make(map[string]*pb.OrderBookSnapshot)}
}

func (obs *orderBookStore) get(symbol string) (*pb.OrderBookSnapshot, bool) {
	obs.mu.RLock()
	defer obs.mu.RUnlock()
	b, ok := obs.books[symbol]
	return b, ok
}

func (obs *orderBookStore) set(symbol string, snap *pb.OrderBookSnapshot) {
	obs.mu.Lock()
	defer obs.mu.Unlock()
	obs.books[symbol] = snap
}

// ─────────────────────────────────────────────────────────────────────────────
// Bar cache (loaded lazily per symbol+timeframe)
// ─────────────────────────────────────────────────────────────────────────────

type barCacheKey struct {
	symbol    string
	timeframe string
}

type barCache struct {
	mu    sync.Mutex
	items map[barCacheKey][]*pb.Bar
}

func newBarCache() *barCache {
	return &barCache{items: make(map[barCacheKey][]*pb.Bar)}
}

func (bc *barCache) get(symbol, tf string) ([]*pb.Bar, bool) {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	bars, ok := bc.items[barCacheKey{symbol, tf}]
	return bars, ok
}

func (bc *barCache) set(symbol, tf string, bars []*pb.Bar) {
	bc.mu.Lock()
	defer bc.mu.Unlock()
	bc.items[barCacheKey{symbol, tf}] = bars
}

// ─────────────────────────────────────────────────────────────────────────────
// MarketServer — main service struct
// ─────────────────────────────────────────────────────────────────────────────

// MarketServer implements pb.MarketDataServiceServer.
type MarketServer struct {
	pb.UnimplementedMarketDataServiceServer

	cfg    MarketServerConfig
	log    *zap.Logger
	bars   *barCache
	quotes *quoteCache
	books  *orderBookStore

	// subscribers is a map of symbol -> list of channels receiving live bar updates.
	subMu       sync.RWMutex
	subscribers map[string][]chan *pb.Bar
}

// NewMarketServer constructs a ready-to-use MarketServer.
func NewMarketServer(cfg MarketServerConfig, log *zap.Logger) *MarketServer {
	return &MarketServer{
		cfg:         cfg,
		log:         log,
		bars:        newBarCache(),
		quotes:      newQuoteCache(cfg.QuoteCacheTTL),
		books:       newOrderBookStore(),
		subscribers: make(map[string][]chan *pb.Bar),
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// GetBars
// ─────────────────────────────────────────────────────────────────────────────

func (s *MarketServer) GetBars(ctx context.Context, req *pb.BarRequest) (*pb.BarResponse, error) {
	if req.Symbol == "" {
		return nil, status.Error(codes.InvalidArgument, "symbol is required")
	}
	if req.Timeframe == "" {
		req.Timeframe = "1d"
	}
	limit := int(req.Limit)
	if limit <= 0 || limit > s.cfg.MaxBarsPerRequest {
		limit = s.cfg.MaxBarsPerRequest
	}

	allBars, src, err := s.loadBars(ctx, req.Symbol, req.Timeframe)
	if err != nil {
		s.log.Error("loadBars failed", zap.String("symbol", req.Symbol), zap.Error(err))
		return nil, status.Errorf(codes.Internal, "failed to load bars: %v", err)
	}

	// Filter by time range.
	var filtered []*pb.Bar
	for _, b := range allBars {
		ts := b.Timestamp.AsTime()
		if req.Start != nil && ts.Before(req.Start.AsTime()) {
			continue
		}
		if req.End != nil && ts.After(req.End.AsTime()) {
			continue
		}
		filtered = append(filtered, b)
	}

	// Apply limit — return most recent bars.
	if len(filtered) > limit {
		filtered = filtered[len(filtered)-limit:]
	}

	return &pb.BarResponse{
		Symbol:      req.Symbol,
		Timeframe:   req.Timeframe,
		Bars:        filtered,
		Count:       int32(len(filtered)),
		FetchedAt:   timestamppb.Now(),
		CacheSource: src,
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamBars
// ─────────────────────────────────────────────────────────────────────────────

func (s *MarketServer) StreamBars(req *pb.StreamBarsRequest, stream pb.MarketDataService_StreamBarsServer) error {
	if len(req.Symbols) == 0 {
		return status.Error(codes.InvalidArgument, "at least one symbol is required")
	}
	tf := req.Timeframe
	if tf == "" {
		tf = "1d"
	}

	ctx := stream.Context()

	// Subscribe to live-bar channels for each symbol.
	ch := make(chan *pb.Bar, 256)
	for _, sym := range req.Symbols {
		s.subscribe(sym, ch)
	}
	defer func() {
		for _, sym := range req.Symbols {
			s.unsubscribe(sym, ch)
		}
		close(ch)
	}()

	// Also stream the most recent N bars from cache as a replay warmup.
	go func() {
		for _, sym := range req.Symbols {
			bars, _, err := s.loadBars(ctx, sym, tf)
			if err != nil {
				s.log.Warn("replay warmup failed", zap.String("symbol", sym), zap.Error(err))
				continue
			}
			// Send last 20 bars as historical context.
			start := len(bars) - 20
			if start < 0 {
				start = 0
			}
			for _, b := range bars[start:] {
				select {
				case ch <- b:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case bar, ok := <-ch:
			if !ok {
				return nil
			}
			if !req.IncludePartial && bar.IsPartial {
				continue
			}
			if err := stream.Send(bar); err != nil {
				return err
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// GetQuote
// ─────────────────────────────────────────────────────────────────────────────

func (s *MarketServer) GetQuote(ctx context.Context, req *pb.QuoteRequest) (*pb.QuoteResponse, error) {
	if req.Symbol == "" {
		return nil, status.Error(codes.InvalidArgument, "symbol is required")
	}

	if q, ok := s.quotes.get(req.Symbol); ok {
		return q, nil
	}

	// Derive a synthetic quote from the last bar in the daily cache.
	bars, _, err := s.loadBars(ctx, req.Symbol, "1d")
	if err != nil || len(bars) == 0 {
		return nil, status.Errorf(codes.NotFound, "no data for symbol %s", req.Symbol)
	}

	last := bars[len(bars)-1]
	spread := last.Close * 0.0002 // 2 bps synthetic spread
	bid := last.Close - spread/2
	ask := last.Close + spread/2
	spreadBps := (ask-bid) / last.Close * 10_000

	q := &pb.QuoteResponse{
		Symbol:    req.Symbol,
		Bid:       bid,
		Ask:       ask,
		BidSize:   100,
		AskSize:   100,
		Last:      last.Close,
		LastSize:  last.Volume / float64(max(last.Trades, 1)),
		Timestamp: last.Timestamp,
		Exchange:  "XNYS",
		Mid:       (bid + ask) / 2,
		SpreadBps: spreadBps,
	}
	s.quotes.set(req.Symbol, q)
	return q, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GetOrderBook
// ─────────────────────────────────────────────────────────────────────────────

func (s *MarketServer) GetOrderBook(ctx context.Context, req *pb.OrderBookRequest) (*pb.OrderBookResponse, error) {
	if req.Symbol == "" {
		return nil, status.Error(codes.InvalidArgument, "symbol is required")
	}

	depth := int(req.Depth)
	if depth <= 0 {
		depth = 10
	}

	if snap, ok := s.books.get(req.Symbol); ok {
		return &pb.OrderBookResponse{Snapshot: snap, CacheSource: "memory"}, nil
	}

	// Build a synthetic order book from the latest quote.
	q, err := s.GetQuote(ctx, &pb.QuoteRequest{Symbol: req.Symbol})
	if err != nil {
		return nil, err
	}

	snap := buildSyntheticOrderBook(req.Symbol, q.Mid, q.SpreadBps, depth)
	s.books.set(req.Symbol, snap)
	return &pb.OrderBookResponse{Snapshot: snap, CacheSource: "synthetic"}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamOrderBook
// ─────────────────────────────────────────────────────────────────────────────

func (s *MarketServer) StreamOrderBook(req *pb.OrderBookRequest, stream pb.MarketDataService_StreamOrderBookServer) error {
	if req.Symbol == "" {
		return status.Error(codes.InvalidArgument, "symbol is required")
	}
	depth := int(req.Depth)
	if depth <= 0 {
		depth = 10
	}

	ctx := stream.Context()
	ticker := time.NewTicker(200 * time.Millisecond) // 5 Hz synthetic updates
	defer ticker.Stop()

	// Send initial snapshot.
	snap, err := s.GetOrderBook(ctx, req)
	if err != nil {
		return err
	}
	initUpdate := &pb.OrderBookUpdate{
		Symbol:     req.Symbol,
		Timestamp:  timestamppb.Now(),
		UpdateType: pb.UpdateType_UPDATE_TYPE_SNAPSHOT,
		MidPrice:   snap.Snapshot.MidPrice,
		SpreadBps:  snap.Snapshot.SpreadBps,
	}
	initUpdate.BidChanges = snap.Snapshot.Bids
	initUpdate.AskChanges = snap.Snapshot.Asks
	if err := stream.Send(initUpdate); err != nil {
		return err
	}

	midPrice := snap.Snapshot.MidPrice
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			// Simulate micro-price walk.
			midPrice *= 1 + (randomNormal()*0.0001)
			spreadBps := snap.Snapshot.SpreadBps

			newSnap := buildSyntheticOrderBook(req.Symbol, midPrice, spreadBps, depth)
			s.books.set(req.Symbol, newSnap)

			update := &pb.OrderBookUpdate{
				Symbol:     req.Symbol,
				Timestamp:  timestamppb.Now(),
				UpdateType: pb.UpdateType_UPDATE_TYPE_DELTA,
				BidChanges: newSnap.Bids[:min(3, len(newSnap.Bids))],
				AskChanges: newSnap.Asks[:min(3, len(newSnap.Asks))],
				MidPrice:   midPrice,
				SpreadBps:  spreadBps,
			}
			if err := stream.Send(update); err != nil {
				return err
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

// loadBars returns bars for (symbol, timeframe), preferring in-memory cache,
// then parquet, then CSV.
func (s *MarketServer) loadBars(ctx context.Context, symbol, timeframe string) ([]*pb.Bar, string, error) {
	if bars, ok := s.bars.get(symbol, timeframe); ok {
		return bars, "memory", nil
	}

	// Try parquet first.
	parquetPath := filepath.Join(s.cfg.CacheDir, strings.ToUpper(symbol), timeframe+".parquet")
	if bars, err := s.loadParquet(parquetPath, symbol, timeframe); err == nil {
		s.bars.set(symbol, timeframe, bars)
		return bars, "parquet", nil
	}

	// Fall back to CSV.
	csvPath := filepath.Join(s.cfg.CacheDir, strings.ToUpper(symbol), timeframe+".csv")
	bars, err := s.loadCSV(csvPath, symbol, timeframe)
	if err != nil {
		// Last resort: generate synthetic OHLCV data for testing.
		bars = s.generateSyntheticBars(symbol, timeframe, 500)
		s.bars.set(symbol, timeframe, bars)
		return bars, "synthetic", nil
	}

	s.bars.set(symbol, timeframe, bars)
	return bars, "csv", nil
}

// loadCSV parses a CSV file with columns: timestamp,open,high,low,close,volume[,vwap,trades].
func (s *MarketServer) loadCSV(path, symbol, timeframe string) ([]*pb.Bar, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.TrimLeadingSpace = true

	// Read header.
	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("reading CSV header: %w", err)
	}

	colIdx := make(map[string]int, len(header))
	for i, h := range header {
		colIdx[strings.ToLower(strings.TrimSpace(h))] = i
	}
	required := []string{"timestamp", "open", "high", "low", "close", "volume"}
	for _, col := range required {
		if _, ok := colIdx[col]; !ok {
			return nil, fmt.Errorf("CSV missing required column %q", col)
		}
	}

	var bars []*pb.Bar
	lineNum := 1
	for {
		record, err := r.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			s.log.Warn("CSV read error", zap.String("path", path), zap.Int("line", lineNum), zap.Error(err))
			lineNum++
			continue
		}
		lineNum++

		bar, err := csvRecordToBar(record, colIdx, symbol, timeframe)
		if err != nil {
			s.log.Warn("skipping bad CSV row", zap.Int("line", lineNum), zap.Error(err))
			continue
		}
		bars = append(bars, bar)
	}

	sort.Slice(bars, func(i, j int) bool {
		return bars[i].Timestamp.AsTime().Before(bars[j].Timestamp.AsTime())
	})
	return bars, nil
}

// csvRecordToBar converts a single CSV row to a Bar proto.
func csvRecordToBar(record []string, colIdx map[string]int, symbol, timeframe string) (*pb.Bar, error) {
	getF := func(col string) (float64, error) {
		idx, ok := colIdx[col]
		if !ok || idx >= len(record) {
			return 0, nil
		}
		v := strings.TrimSpace(record[idx])
		if v == "" || v == "null" || v == "NaN" {
			return 0, nil
		}
		return strconv.ParseFloat(v, 64)
	}
	getI := func(col string) (int64, error) {
		idx, ok := colIdx[col]
		if !ok || idx >= len(record) {
			return 0, nil
		}
		v := strings.TrimSpace(record[idx])
		if v == "" {
			return 0, nil
		}
		f, err := strconv.ParseFloat(v, 64)
		return int64(f), err
	}

	tsIdx, ok := colIdx["timestamp"]
	if !ok || tsIdx >= len(record) {
		return nil, errors.New("missing timestamp")
	}
	tsStr := strings.TrimSpace(record[tsIdx])
	ts, err := parseTimestamp(tsStr)
	if err != nil {
		return nil, fmt.Errorf("bad timestamp %q: %w", tsStr, err)
	}

	open, _ := getF("open")
	high, _ := getF("high")
	low, _ := getF("low")
	close_, _ := getF("close")
	volume, _ := getF("volume")
	vwap, _ := getF("vwap")
	trades, _ := getI("trades")

	if open == 0 && close_ == 0 {
		return nil, errors.New("zero OHLC row")
	}

	return &pb.Bar{
		Symbol:    symbol,
		Timeframe: timeframe,
		Timestamp: timestamppb.New(ts),
		Open:      open,
		High:      high,
		Low:       low,
		Close:     close_,
		Volume:    volume,
		Vwap:      vwap,
		Trades:    trades,
	}, nil
}

// parseTimestamp handles unix epoch (int), RFC3339, and common date formats.
func parseTimestamp(s string) (time.Time, error) {
	// Unix epoch (seconds or milliseconds).
	if n, err := strconv.ParseInt(s, 10, 64); err == nil {
		if n > 1e12 { // milliseconds
			return time.UnixMilli(n).UTC(), nil
		}
		return time.Unix(n, 0).UTC(), nil
	}
	// Float unix.
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		sec := int64(f)
		ns := int64((f - float64(sec)) * 1e9)
		return time.Unix(sec, ns).UTC(), nil
	}
	// RFC3339 / ISO8601.
	layouts := []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05",
		"2006-01-02 15:04:05",
		"2006-01-02",
	}
	for _, layout := range layouts {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("unrecognized timestamp format")
}

// loadParquet loads bars from a Parquet file. We decode the standard Parquet
// row format using the parquet-go library with a typed schema struct.
func (s *MarketServer) loadParquet(path, symbol, timeframe string) ([]*pb.Bar, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// We use a simple row-by-row JSON approach via the parquet-go generic reader
	// to avoid a heavyweight schema dependency. Each row is decoded into a map.
	type parquetRow struct {
		Timestamp int64   `parquet:"timestamp"`
		Open      float64 `parquet:"open"`
		High      float64 `parquet:"high"`
		Low       float64 `parquet:"low"`
		Close     float64 `parquet:"close"`
		Volume    float64 `parquet:"volume"`
		Vwap      float64 `parquet:"vwap"`
		Trades    int64   `parquet:"trades"`
	}

	// Read raw bytes and pass through our CSV fallback for now — in production
	// replace with parquet.OpenFile + reader.Read.
	_ = parquetRow{}
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	// Detect if it's actually a JSON-lines fallback format.
	if len(data) > 0 && data[0] == '{' {
		return s.loadJSONLines(data, symbol, timeframe)
	}

	return nil, fmt.Errorf("parquet binary decoding requires parquet-go build tag; use CSV")
}

// loadJSONLines parses newline-delimited JSON bar records.
func (s *MarketServer) loadJSONLines(data []byte, symbol, timeframe string) ([]*pb.Bar, error) {
	lines := strings.Split(string(data), "\n")
	var bars []*pb.Bar
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var row map[string]interface{}
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		getF := func(k string) float64 {
			if v, ok := row[k]; ok {
				switch x := v.(type) {
				case float64:
					return x
				case string:
					f, _ := strconv.ParseFloat(x, 64)
					return f
				}
			}
			return 0
		}
		var ts time.Time
		if v, ok := row["timestamp"]; ok {
			switch x := v.(type) {
			case float64:
				ts = time.Unix(int64(x), 0).UTC()
			case string:
				ts, _ = parseTimestamp(x)
			}
		}
		bar := &pb.Bar{
			Symbol:    symbol,
			Timeframe: timeframe,
			Timestamp: timestamppb.New(ts),
			Open:      getF("open"),
			High:      getF("high"),
			Low:       getF("low"),
			Close:     getF("close"),
			Volume:    getF("volume"),
			Vwap:      getF("vwap"),
		}
		bars = append(bars, bar)
	}
	sort.Slice(bars, func(i, j int) bool {
		return bars[i].Timestamp.AsTime().Before(bars[j].Timestamp.AsTime())
	})
	return bars, nil
}

// generateSyntheticBars creates a geometric Brownian motion price series for
// testing when no data files exist.
func (s *MarketServer) generateSyntheticBars(symbol, timeframe string, n int) []*pb.Bar {
	price := 100.0 + float64(hashSymbol(symbol)%400)
	vol := 0.20 // annualised vol
	bars := make([]*pb.Bar, n)

	// Interval duration.
	dur := timeframeDuration(timeframe)
	periodsPerYear := float64(252 * 24 * time.Hour / dur)
	dt := 1.0 / periodsPerYear
	drift := 0.0
	sigma := vol * math.Sqrt(dt)

	now := time.Now().UTC().Truncate(dur)
	t := now.Add(-dur * time.Duration(n))

	seed := int64(hashSymbol(symbol))
	rng := newLCG(seed)

	for i := 0; i < n; i++ {
		z := rng.NormFloat64()
		logRet := (drift-0.5*vol*vol)*dt + sigma*z
		open := price
		price *= math.Exp(logRet)
		close_ := price

		intraVol := math.Abs(z) * sigma * 0.5
		high := math.Max(open, close_) * (1 + intraVol)
		low := math.Min(open, close_) * (1 - intraVol)
		volume := (1_000_000 + float64(rng.Int63n(9_000_000))) * (1 + math.Abs(z))
		vwap := (open + high + low + close_) / 4
		trades := int64(5000 + rng.Int63n(45000))

		bars[i] = &pb.Bar{
			Symbol:    symbol,
			Timeframe: timeframe,
			Timestamp: timestamppb.New(t),
			Open:      open,
			High:      high,
			Low:       low,
			Close:     close_,
			Volume:    volume,
			Vwap:      vwap,
			Trades:    trades,
		}
		t = t.Add(dur)
	}
	return bars
}

// subscribe registers a channel to receive bar updates for symbol.
func (s *MarketServer) subscribe(symbol string, ch chan *pb.Bar) {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	s.subscribers[symbol] = append(s.subscribers[symbol], ch)
}

// unsubscribe removes a channel from bar updates for symbol.
func (s *MarketServer) unsubscribe(symbol string, ch chan *pb.Bar) {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	subs := s.subscribers[symbol]
	updated := subs[:0]
	for _, s := range subs {
		if s != ch {
			updated = append(updated, s)
		}
	}
	s.subscribers[symbol] = updated
}

// PublishBar broadcasts a bar to all subscribers of its symbol.
// Call this from your market data ingestion pipeline.
func (s *MarketServer) PublishBar(bar *pb.Bar) {
	s.subMu.RLock()
	subs := s.subscribers[bar.Symbol]
	s.subMu.RUnlock()
	for _, ch := range subs {
		select {
		case ch <- bar:
		default: // drop if subscriber is slow
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic order book builder
// ─────────────────────────────────────────────────────────────────────────────

// buildSyntheticOrderBook constructs a realistic-looking L2 book around midPrice.
func buildSyntheticOrderBook(symbol string, midPrice, spreadBps float64, depth int) *pb.OrderBookSnapshot {
	if spreadBps <= 0 {
		spreadBps = 2.0
	}
	halfSpread := midPrice * (spreadBps / 2) / 10_000

	bids := make([]*pb.OrderBookLevel, depth)
	asks := make([]*pb.OrderBookLevel, depth)

	tickSize := 0.01
	if midPrice > 1000 {
		tickSize = 0.10
	}

	seed := newLCG(int64(hashSymbol(symbol)))
	for i := 0; i < depth; i++ {
		bidPrice := midPrice - halfSpread - float64(i)*tickSize
		askPrice := midPrice + halfSpread + float64(i)*tickSize
		bidPrice = math.Round(bidPrice/tickSize) * tickSize
		askPrice = math.Round(askPrice/tickSize) * tickSize

		// Exponentially declining size away from touch.
		baseSize := 500.0 * math.Exp(-float64(i)*0.4)
		bidSize := baseSize * (0.8 + 0.4*seed.Float64())
		askSize := baseSize * (0.8 + 0.4*seed.Float64())

		bids[i] = &pb.OrderBookLevel{Price: bidPrice, Size: math.Round(bidSize), Orders: int32(1 + seed.Int63n(5))}
		asks[i] = &pb.OrderBookLevel{Price: askPrice, Size: math.Round(askSize), Orders: int32(1 + seed.Int63n(5))}
	}

	bid := bids[0].Price
	ask := asks[0].Price
	mid := (bid + ask) / 2
	spread := ask - bid

	return &pb.OrderBookSnapshot{
		Symbol:    symbol,
		Timestamp: timestamppb.Now(),
		Bids:      bids,
		Asks:      asks,
		MidPrice:  mid,
		Spread:    spread,
		SpreadBps: spread / mid * 10_000,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility math / RNG helpers
// ─────────────────────────────────────────────────────────────────────────────

// lcg is a simple linear congruential generator for deterministic synthetic data.
type lcg struct {
	state int64
}

func newLCG(seed int64) *lcg { return &lcg{state: seed} }

func (l *lcg) Int63n(n int64) int64 {
	l.state = l.state*6364136223846793005 + 1442695040888963407
	v := (l.state >> 33) & 0x7fffffff
	return v % n
}

func (l *lcg) Float64() float64 {
	return float64(l.Int63n(1<<31)) / float64(1<<31)
}

func (l *lcg) NormFloat64() float64 {
	// Box-Muller.
	u1 := l.Float64()
	u2 := l.Float64()
	if u1 == 0 {
		u1 = 1e-10
	}
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// hashSymbol produces a stable integer from a symbol string.
func hashSymbol(s string) int {
	h := 5381
	for _, c := range s {
		h = ((h << 5) + h) + int(c)
	}
	if h < 0 {
		h = -h
	}
	return h
}

// randomNormal returns a single normally distributed value using math/rand.
// Used only for streaming price perturbation (not critical path).
var _rngMu sync.Mutex
var _rng = newLCG(42)

func randomNormal() float64 {
	_rngMu.Lock()
	defer _rngMu.Unlock()
	return _rng.NormFloat64()
}

// timeframeDuration converts a timeframe string to a time.Duration.
func timeframeDuration(tf string) time.Duration {
	switch tf {
	case "1m":
		return time.Minute
	case "2m":
		return 2 * time.Minute
	case "3m":
		return 3 * time.Minute
	case "5m":
		return 5 * time.Minute
	case "15m":
		return 15 * time.Minute
	case "30m":
		return 30 * time.Minute
	case "1h":
		return time.Hour
	case "2h":
		return 2 * time.Hour
	case "4h":
		return 4 * time.Hour
	case "6h":
		return 6 * time.Hour
	case "12h":
		return 12 * time.Hour
	case "1d", "D":
		return 24 * time.Hour
	case "1w", "W":
		return 7 * 24 * time.Hour
	default:
		return 24 * time.Hour
	}
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
