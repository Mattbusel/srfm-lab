// strategy_server.go — StrategyService gRPC server.
// Implements BH (Bull-Hawk) state computation, delta scoring, and signal
// generation in pure Go, mirroring the Rust/Python BH engine logic.
package server

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	pbm  "github.com/srfm/infra/grpc/proto/market"
	pb   "github.com/srfm/infra/grpc/proto/strategy"
)

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

// StrategyServerConfig holds parameters for the BH computation engine.
type StrategyServerConfig struct {
	// Default lookback windows (in bars) for each timeframe.
	MomentumWindow    int
	VolatilityWindow  int
	TrendWindow       int
	DeltaBucketWindow int

	// BH scoring weights.
	MomentumWeight   float64
	TrendWeight      float64
	VolatilityWeight float64
	VolumeWeight     float64

	// Signal confidence threshold below which we emit FLAT.
	MinConfidence float64

	// Maximum history states to keep per symbol/timeframe.
	MaxHistoryLen int
}

func DefaultStrategyServerConfig() StrategyServerConfig {
	return StrategyServerConfig{
		MomentumWindow:    14,
		VolatilityWindow:  20,
		TrendWindow:       50,
		DeltaBucketWindow: 10,
		MomentumWeight:    0.35,
		TrendWeight:       0.30,
		VolatilityWeight:  0.20,
		VolumeWeight:      0.15,
		MinConfidence:     0.40,
		MaxHistoryLen:     200,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Strategy metadata registry
// ─────────────────────────────────────────────────────────────────────────────

type strategyMeta struct {
	id          string
	name        string
	version     string
	description string
	timeframes  []string
	features    []string
	params      map[string]string
	createdAt   time.Time
}

var defaultStrategies = []strategyMeta{
	{
		id: "bh_v1", name: "Bull-Hawk v1", version: "1.0.0",
		description: "Core BH directional model using momentum, trend, vol regime, and volume.",
		timeframes:  []string{"1m", "5m", "15m", "1h", "4h", "1d"},
		features:    []string{"momentum_z", "trend_slope", "vol_percentile", "volume_z", "rsi_14", "atr_14"},
		params:      map[string]string{"momentum_window": "14", "trend_window": "50", "vol_window": "20"},
		createdAt:   time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
	},
	{
		id: "bh_v2", name: "Bull-Hawk v2", version: "2.0.0",
		description: "Enhanced BH with hawk/dove sub-scores and regime detection.",
		timeframes:  []string{"5m", "15m", "1h", "4h", "1d"},
		features:    []string{"momentum_z", "trend_slope", "vol_percentile", "volume_z", "rsi_14", "atr_14", "adx", "bb_width"},
		params:      map[string]string{"momentum_window": "21", "trend_window": "50", "vol_window": "20", "adx_window": "14"},
		createdAt:   time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC),
	},
	{
		id: "delta_mean_revert", name: "Delta Mean Reversion", version: "1.0.0",
		description: "Delta-score based mean reversion on intraday timeframes.",
		timeframes:  []string{"1m", "5m", "15m"},
		features:    []string{"delta_score", "momentum_z", "rsi_14", "bb_pct_b"},
		params:      map[string]string{"bucket_window": "10", "zscore_threshold": "2.0"},
		createdAt:   time.Date(2024, 3, 1, 0, 0, 0, 0, time.UTC),
	},
}

// ─────────────────────────────────────────────────────────────────────────────
// BH state history store
// ─────────────────────────────────────────────────────────────────────────────

type bhHistoryKey struct {
	symbol    string
	timeframe string
}

type bhHistoryStore struct {
	mu      sync.RWMutex
	history map[bhHistoryKey][]*pb.BHState
	maxLen  int
}

func newBHHistoryStore(maxLen int) *bhHistoryStore {
	return &bhHistoryStore{history: make(map[bhHistoryKey][]*pb.BHState), maxLen: maxLen}
}

func (h *bhHistoryStore) append(symbol, tf string, state *pb.BHState) {
	h.mu.Lock()
	defer h.mu.Unlock()
	k := bhHistoryKey{symbol, tf}
	h.history[k] = append(h.history[k], state)
	if len(h.history[k]) > h.maxLen {
		h.history[k] = h.history[k][len(h.history[k])-h.maxLen:]
	}
}

func (h *bhHistoryStore) get(symbol, tf string, n int) []*pb.BHState {
	h.mu.RLock()
	defer h.mu.RUnlock()
	k := bhHistoryKey{symbol, tf}
	states := h.history[k]
	if n <= 0 || n >= len(states) {
		cp := make([]*pb.BHState, len(states))
		copy(cp, states)
		return cp
	}
	cp := make([]*pb.BHState, n)
	copy(cp, states[len(states)-n:])
	return cp
}

// ─────────────────────────────────────────────────────────────────────────────
// StrategyServer
// ─────────────────────────────────────────────────────────────────────────────

// StrategyServer implements pb.StrategyServiceServer.
type StrategyServer struct {
	pb.UnimplementedStrategyServiceServer

	cfg       StrategyServerConfig
	log       *zap.Logger
	marketSrv *MarketServer
	history   *bhHistoryStore

	// signal subscriber channels: symbol -> []chan
	subMu      sync.RWMutex
	sigSubs    map[string][]chan *pb.SignalEvent
	seqCounter int64
	seqMu      sync.Mutex
}

// NewStrategyServer constructs a ready-to-use StrategyServer.
func NewStrategyServer(cfg StrategyServerConfig, market *MarketServer, log *zap.Logger) *StrategyServer {
	return &StrategyServer{
		cfg:       cfg,
		log:       log,
		marketSrv: market,
		history:   newBHHistoryStore(cfg.MaxHistoryLen),
		sigSubs:   make(map[string][]chan *pb.SignalEvent),
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// ComputeSignal
// ─────────────────────────────────────────────────────────────────────────────

func (s *StrategyServer) ComputeSignal(ctx context.Context, req *pb.SignalRequest) (*pb.SignalResponse, error) {
	if req.Symbol == "" {
		return nil, status.Error(codes.InvalidArgument, "symbol is required")
	}
	if req.StrategyId == "" {
		req.StrategyId = "bh_v2"
	}
	if req.Timeframe == "" {
		req.Timeframe = "1d"
	}

	start := time.Now()

	bhState, err := s.computeBHState(ctx, req.Symbol, req.Timeframe, req.AsOf, req.Features)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "BH computation failed: %v", err)
	}

	signal := bhStateToSignal(bhState, req.StrategyId, req.Timeframe, s.cfg.MinConfidence)
	s.history.append(req.Symbol, req.Timeframe, bhState)

	// Broadcast to stream subscribers.
	s.broadcastSignal(signal)

	return &pb.SignalResponse{
		Signal:    signal,
		ComputeUs: time.Since(start).Microseconds(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GetBHState
// ─────────────────────────────────────────────────────────────────────────────

func (s *StrategyServer) GetBHState(ctx context.Context, req *pb.BHStateRequest) (*pb.BHStateResponse, error) {
	if req.Symbol == "" {
		return nil, status.Error(codes.InvalidArgument, "symbol is required")
	}
	if req.Timeframe == "" {
		req.Timeframe = "1d"
	}

	start := time.Now()
	bhState, err := s.computeBHState(ctx, req.Symbol, req.Timeframe, req.AsOf, nil)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "BH computation failed: %v", err)
	}
	s.history.append(req.Symbol, req.Timeframe, bhState)

	resp := &pb.BHStateResponse{
		Current:   bhState,
		ComputeUs: time.Since(start).Microseconds(),
	}
	if req.IncludeHistory {
		n := int(req.HistoryBars)
		resp.History = &pb.BHStateHistory{States: s.history.get(req.Symbol, req.Timeframe, n)}
	}
	return resp, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GetDeltaScore
// ─────────────────────────────────────────────────────────────────────────────

func (s *StrategyServer) GetDeltaScore(ctx context.Context, req *pb.DeltaScoreRequest) (*pb.DeltaScoreResponse, error) {
	if req.Symbol == "" {
		return nil, status.Error(codes.InvalidArgument, "symbol is required")
	}
	if req.BucketId == "" {
		req.BucketId = "momentum_1d"
	}
	if req.Timeframe == "" {
		req.Timeframe = "1d"
	}

	start := time.Now()

	barResp, err := s.marketSrv.GetBars(ctx, &pbm.BarRequest{
		Symbol:    req.Symbol,
		Timeframe: req.Timeframe,
		Limit:     int32(s.cfg.DeltaBucketWindow * 5),
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to fetch bars: %v", err)
	}

	closes := extractCloses(barResp.Bars)
	if len(closes) < s.cfg.DeltaBucketWindow {
		return nil, status.Errorf(codes.FailedPrecondition,
			"insufficient data: need %d bars, got %d", s.cfg.DeltaBucketWindow, len(closes))
	}

	score := computeDeltaScore(closes, req.BucketId, s.cfg.DeltaBucketWindow)
	score.Symbol = req.Symbol
	score.Timestamp = timestamppb.Now()

	return &pb.DeltaScoreResponse{
		Score:     score,
		ComputeUs: time.Since(start).Microseconds(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamSignals
// ─────────────────────────────────────────────────────────────────────────────

func (s *StrategyServer) StreamSignals(req *pb.StreamSignalsRequest, stream pb.StrategyService_StreamSignalsServer) error {
	ctx := stream.Context()
	ch := make(chan *pb.SignalEvent, 128)

	symbols := req.Symbols
	if len(symbols) == 0 {
		symbols = []string{"*"}
	}
	for _, sym := range symbols {
		s.addSigSub(sym, ch)
	}
	defer func() {
		for _, sym := range symbols {
			s.removeSigSub(sym, ch)
		}
		close(ch)
	}()

	minConf := req.MinConfidence

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case evt, ok := <-ch:
			if !ok {
				return nil
			}
			if len(req.StrategyIds) > 0 && !containsStr(req.StrategyIds, evt.Signal.StrategyId) {
				continue
			}
			if evt.Signal.Confidence < minConf {
				continue
			}
			if req.Timeframe != "" && evt.Signal.Timeframe != req.Timeframe {
				continue
			}
			if err := stream.Send(evt); err != nil {
				return err
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchComputeSignals
// ─────────────────────────────────────────────────────────────────────────────

func (s *StrategyServer) BatchComputeSignals(ctx context.Context, req *pb.BatchSignalRequest) (*pb.BatchSignalResponse, error) {
	if len(req.Requests) == 0 {
		return &pb.BatchSignalResponse{}, nil
	}

	start := time.Now()
	results := make([]*pb.SignalResponse, len(req.Requests))
	var errMsgs []string
	errCount := 0

	if req.Parallel {
		var wg sync.WaitGroup
		var mu sync.Mutex
		wg.Add(len(req.Requests))
		for i, r := range req.Requests {
			i, r := i, r
			go func() {
				defer wg.Done()
				resp, err := s.ComputeSignal(ctx, r)
				mu.Lock()
				defer mu.Unlock()
				if err != nil {
					errCount++
					errMsgs = append(errMsgs, fmt.Sprintf("%s: %v", r.Symbol, err))
					results[i] = &pb.SignalResponse{}
				} else {
					results[i] = resp
				}
			}()
		}
		wg.Wait()
	} else {
		for i, r := range req.Requests {
			resp, err := s.ComputeSignal(ctx, r)
			if err != nil {
				errCount++
				errMsgs = append(errMsgs, fmt.Sprintf("%s: %v", r.Symbol, err))
				results[i] = &pb.SignalResponse{}
			} else {
				results[i] = resp
			}
		}
	}

	return &pb.BatchSignalResponse{
		Results:      results,
		TotalCount:   int32(len(req.Requests)),
		ErrorCount:   int32(errCount),
		Errors:       errMsgs,
		TotalComputeUs: time.Since(start).Microseconds(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GetStrategyInfo
// ─────────────────────────────────────────────────────────────────────────────

func (s *StrategyServer) GetStrategyInfo(ctx context.Context, req *pb.StrategyInfoRequest) (*pb.StrategyInfoResponse, error) {
	for _, meta := range defaultStrategies {
		if meta.id == req.StrategyId {
			return &pb.StrategyInfoResponse{
				StrategyId:          meta.id,
				Name:                meta.name,
				Version:             meta.version,
				Description:         meta.description,
				SupportedTimeframes: meta.timeframes,
				RequiredFeatures:    meta.features,
				Parameters:          meta.params,
				CreatedAt:           timestamppb.New(meta.createdAt),
			}, nil
		}
	}
	return nil, status.Errorf(codes.NotFound, "strategy %q not found", req.StrategyId)
}

// ─────────────────────────────────────────────────────────────────────────────
// Core BH computation engine
// ─────────────────────────────────────────────────────────────────────────────

// computeBHState fetches bars and computes the full BH state vector.
func (s *StrategyServer) computeBHState(
	ctx context.Context,
	symbol, timeframe string,
	asOf *timestamppb.Timestamp,
	overrideFeatures map[string]float64,
) (*pb.BHState, error) {
	lookback := s.cfg.TrendWindow + 10
	barReq := &pbm.BarRequest{
		Symbol:    symbol,
		Timeframe: timeframe,
		Limit:     int32(lookback),
	}
	if asOf != nil {
		barReq.End = asOf
	}

	barResp, err := s.marketSrv.GetBars(ctx, barReq)
	if err != nil {
		return nil, err
	}
	if len(barResp.Bars) < 20 {
		return nil, fmt.Errorf("insufficient bars: %d", len(barResp.Bars))
	}

	bars := barResp.Bars
	closes := extractCloses(bars)
	volumes := extractVolumes(bars)
	highs := extractHighs(bars)
	lows := extractLows(bars)
	lastBar := bars[len(bars)-1]

	// ── Feature computation ──────────────────────────────────────────────────

	n := len(closes)
	mw := min(s.cfg.MomentumWindow, n-1)
	vw := min(s.cfg.VolatilityWindow, n-1)
	tw := min(s.cfg.TrendWindow, n-1)

	// Momentum: rate of change over momentum window.
	roc := (closes[n-1] - closes[n-1-mw]) / closes[n-1-mw]
	momentumZ := zScoreCloses(closes, mw)

	// Trend: linear regression slope z-score.
	trendSlope := linearRegressionSlope(closes[n-tw:])
	trendZ := trendSlope / (stdDev(closes[n-tw:]) + 1e-10)

	// Volatility: realized vol percentile rank.
	realizedVol := stdDev(logReturns(closes[n-vw:]))
	volHistory := rollingStdDev(logReturns(closes), vw)
	volPctRank := percentileRank(volHistory, realizedVol)

	// RSI.
	rsi := computeRSI(closes, mw)

	// ATR.
	atr := computeATR(highs, lows, closes, mw)
	atrPct := atr / closes[n-1]

	// Volume z-score.
	volZ := zScoreValues(volumes, vw)

	// ADX.
	adx := computeADX(highs, lows, closes, 14)

	// Bollinger Band width.
	bbWidth := computeBBWidth(closes, 20, 2.0)

	// Apply overrides.
	features := map[string]float64{
		"momentum_z":     momentumZ,
		"trend_slope":    trendZ,
		"vol_percentile": volPctRank,
		"volume_z":       volZ,
		"rsi_14":         rsi,
		"atr_14":         atrPct,
		"adx":            adx,
		"bb_width":       bbWidth,
		"roc":            roc,
	}
	for k, v := range overrideFeatures {
		features[k] = v
	}

	// ── BH score computation ────────────────────────────────────────────────

	// Bull score: positive momentum + uptrend + above-average volume on up days.
	bullMomentum := sigmoid(momentumZ * 2)
	bullTrend := sigmoid(trendZ * 3)
	bullVolume := 0.5
	if roc > 0 {
		bullVolume = sigmoid(volZ * 1.5)
	}
	bullScore := (bullMomentum*s.cfg.MomentumWeight +
		bullTrend*s.cfg.TrendWeight +
		0.5*s.cfg.VolatilityWeight + // neutral vol contribution
		bullVolume*s.cfg.VolumeWeight)

	// Bear score: inverse of bull pressure.
	bearMomentum := sigmoid(-momentumZ * 2)
	bearTrend := sigmoid(-trendZ * 3)
	bearVolume := 0.5
	if roc < 0 {
		bearVolume = sigmoid(volZ * 1.5)
	}
	bearScore := (bearMomentum*s.cfg.MomentumWeight +
		bearTrend*s.cfg.TrendWeight +
		0.5*s.cfg.VolatilityWeight +
		bearVolume*s.cfg.VolumeWeight)

	// Hawk score: breakout / aggressive momentum energy.
	hawkScore := sigmoid(adx/25.0-1.0) * sigmoid(volZ)
	if bbWidth > 0.05 { // expanding bands = breakout energy
		hawkScore = math.Min(1.0, hawkScore*1.3)
	}

	// Dove score: mean reversion / exhaustion.
	rsiExhaustion := 0.0
	if rsi > 70 {
		rsiExhaustion = (rsi - 70) / 30
	} else if rsi < 30 {
		rsiExhaustion = (30 - rsi) / 30
	}
	doveScore := (rsiExhaustion*0.5 + sigmoid(-adx/25.0)*0.3 + (1-bbWidth*10)*0.2)
	doveScore = math.Max(0, math.Min(1, doveScore))

	bhRatio := bullScore / (bullScore + bearScore + 1e-10)

	// ── Regime classification ───────────────────────────────────────────────

	regime := classifyRegime(trendZ, volPctRank, adx, rsi, roc, bbWidth)

	// ── Direction and confidence ────────────────────────────────────────────

	rawConf := math.Abs(bullScore - bearScore)
	confidence := math.Min(1.0, rawConf*2)

	direction := pb.Direction_DIRECTION_FLAT
	if confidence >= s.cfg.MinConfidence {
		if bullScore > bearScore {
			direction = pb.Direction_DIRECTION_LONG
		} else {
			direction = pb.Direction_DIRECTION_SHORT
		}
	}

	return &pb.BHState{
		Symbol:       symbol,
		Timeframe:    timeframe,
		Timestamp:    lastBar.Timestamp,
		BullScore:    bullScore,
		HawkScore:    hawkScore,
		BearScore:    bearScore,
		DoveScore:    doveScore,
		BhRatio:      bhRatio,
		MomentumZ:    momentumZ,
		VolRegime:    volPctRank,
		Regime:       regime,
		Direction:    direction,
		Confidence:   confidence,
		FactorScores: features,
		ActivePatterns: detectPatterns(closes, highs, lows, rsi, adx, bbWidth),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Delta score computation
// ─────────────────────────────────────────────────────────────────────────────

func computeDeltaScore(closes []float64, bucketID string, window int) *pb.DeltaScore {
	n := len(closes)
	if n < window+1 {
		return &pb.DeltaScore{BucketId: bucketID, IsStale: true}
	}

	recent := closes[n-window:]
	rawDelta := recent[len(recent)-1] - recent[0]
	pctDelta := rawDelta / (recent[0] + 1e-10)

	// Compute rolling deltas for z-score.
	deltas := make([]float64, n-window)
	for i := 0; i < len(deltas); i++ {
		deltas[i] = (closes[i+window] - closes[i]) / (closes[i] + 1e-10)
	}

	mean := meanFloat(deltas)
	std := stdDev(deltas)
	normalized := (pctDelta - mean) / (std + 1e-10)

	pctRank := percentileRank(deltas, pctDelta)

	// Decay factor: more recent signal gets higher weight.
	decayFactor := math.Exp(-0.1 * float64(window))

	// Signal value in [-1, 1].
	signalVal := math.Tanh(normalized)

	return &pb.DeltaScore{
		BucketId:    bucketID,
		RawDelta:    rawDelta,
		Normalized:  normalized,
		Percentile:  pctRank,
		SignalValue: signalVal,
		DecayFactor: decayFactor,
		SampleCount: int32(n),
		IsStale:     false,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal synthesis from BH state
// ─────────────────────────────────────────────────────────────────────────────

func bhStateToSignal(bh *pb.BHState, strategyID, timeframe string, minConf float64) *pb.Signal {
	strength := pb.SignalStrength_SIGNAL_STRENGTH_WEAK
	switch {
	case bh.Confidence >= 0.85:
		strength = pb.SignalStrength_SIGNAL_STRENGTH_CONVICTION
	case bh.Confidence >= 0.70:
		strength = pb.SignalStrength_SIGNAL_STRENGTH_STRONG
	case bh.Confidence >= 0.55:
		strength = pb.SignalStrength_SIGNAL_STRENGTH_MODERATE
	}

	// Derive entry/stop/target from ATR if available.
	atr := bh.FactorScores["atr_14"]
	entryPrice := 0.0 // caller should fill in current market price
	stopLoss := 0.0
	takeProfit := 0.0
	if atr > 0 {
		stopLoss = 2.0 * atr
		takeProfit = 3.0 * atr
	}

	// Suggested position size based on confidence.
	posSize := bh.Confidence * 0.10 // up to 10% of portfolio at full conviction

	rationale := fmt.Sprintf(
		"BH[bull=%.2f bear=%.2f hawk=%.2f dove=%.2f] regime=%s mom_z=%.2f rsi=%.1f",
		bh.BullScore, bh.BearScore, bh.HawkScore, bh.DoveScore,
		bh.Regime.String(), bh.MomentumZ,
		bh.FactorScores["rsi_14"],
	)

	return &pb.Signal{
		Symbol:          bh.Symbol,
		StrategyId:      strategyID,
		Timeframe:       timeframe,
		Timestamp:       bh.Timestamp,
		Direction:       bh.Direction,
		Strength:        strength,
		Confidence:      bh.Confidence,
		EntryPrice:      entryPrice,
		StopLoss:        stopLoss,
		TakeProfit:      takeProfit,
		PositionSizePct: posSize,
		Rationale:       rationale,
		Scores:          bh.FactorScores,
		BhState:         bh,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Pattern detection
// ─────────────────────────────────────────────────────────────────────────────

func detectPatterns(closes, highs, lows []float64, rsi, adx, bbWidth float64) []string {
	var patterns []string
	n := len(closes)
	if n < 5 {
		return patterns
	}

	// RSI oversold / overbought.
	if rsi > 70 {
		patterns = append(patterns, "rsi_overbought")
	} else if rsi < 30 {
		patterns = append(patterns, "rsi_oversold")
	}

	// Strong trend.
	if adx > 25 {
		patterns = append(patterns, "strong_trend")
	} else if adx < 20 {
		patterns = append(patterns, "ranging")
	}

	// BB squeeze then expand.
	if bbWidth < 0.02 {
		patterns = append(patterns, "bb_squeeze")
	} else if bbWidth > 0.08 {
		patterns = append(patterns, "bb_expansion")
	}

	// Higher highs / lower lows (last 5 bars).
	if n >= 5 {
		hh := highs[n-1] > highs[n-2] && highs[n-2] > highs[n-3]
		hl := lows[n-1] > lows[n-2] && lows[n-2] > lows[n-3]
		lh := highs[n-1] < highs[n-2] && highs[n-2] < highs[n-3]
		ll := lows[n-1] < lows[n-2] && lows[n-2] < lows[n-3]

		if hh && hl {
			patterns = append(patterns, "higher_highs_lows")
		}
		if lh && ll {
			patterns = append(patterns, "lower_highs_lows")
		}
	}

	// Doji-like bar.
	if n >= 1 {
		body := math.Abs(closes[n-1] - closes[n-2])
		range_ := highs[n-1] - lows[n-1]
		if range_ > 0 && body/range_ < 0.1 {
			patterns = append(patterns, "doji")
		}
	}

	// Engulfing.
	if n >= 2 {
		prevBody := math.Abs(closes[n-2] - closes[n-3])
		currBody := math.Abs(closes[n-1] - closes[n-2])
		bullEngulf := closes[n-1] > closes[n-2] && closes[n-2] < closes[n-3] && currBody > prevBody
		bearEngulf := closes[n-1] < closes[n-2] && closes[n-2] > closes[n-3] && currBody > prevBody
		if bullEngulf {
			patterns = append(patterns, "bullish_engulfing")
		}
		if bearEngulf {
			patterns = append(patterns, "bearish_engulfing")
		}
	}

	return patterns
}

// ─────────────────────────────────────────────────────────────────────────────
// Regime classification
// ─────────────────────────────────────────────────────────────────────────────

func classifyRegime(trendZ, volPctRank, adx, rsi, roc, bbWidth float64) pb.RegimeType {
	isTrending := adx > 22
	isBreakout := bbWidth > 0.06 && math.Abs(roc) > 0.02
	isVolatile := volPctRank > 0.75
	isRanging := adx < 18 && bbWidth < 0.03

	switch {
	case isBreakout && roc > 0:
		return pb.RegimeType_REGIME_BREAKOUT
	case isBreakout && roc < 0:
		return pb.RegimeType_REGIME_BREAKDOWN
	case isVolatile:
		return pb.RegimeType_REGIME_VOLATILE
	case isTrending && trendZ > 0:
		return pb.RegimeType_REGIME_TRENDING_UP
	case isTrending && trendZ < 0:
		return pb.RegimeType_REGIME_TRENDING_DOWN
	case isRanging:
		return pb.RegimeType_REGIME_RANGING
	default:
		if trendZ > 0.5 {
			return pb.RegimeType_REGIME_TRENDING_UP
		} else if trendZ < -0.5 {
			return pb.RegimeType_REGIME_TRENDING_DOWN
		}
		return pb.RegimeType_REGIME_RANGING
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Technical indicator implementations
// ─────────────────────────────────────────────────────────────────────────────

func computeRSI(closes []float64, period int) float64 {
	n := len(closes)
	if n <= period {
		return 50.0
	}
	returns := make([]float64, n-1)
	for i := 1; i < n; i++ {
		returns[i-1] = closes[i] - closes[i-1]
	}
	// Use Wilder's smoothed method.
	gains, losses := 0.0, 0.0
	for _, r := range returns[:period] {
		if r > 0 {
			gains += r
		} else {
			losses += -r
		}
	}
	avgGain := gains / float64(period)
	avgLoss := losses / float64(period)

	for _, r := range returns[period:] {
		if r > 0 {
			avgGain = (avgGain*float64(period-1) + r) / float64(period)
			avgLoss = avgLoss * float64(period-1) / float64(period)
		} else {
			avgGain = avgGain * float64(period-1) / float64(period)
			avgLoss = (avgLoss*float64(period-1) + (-r)) / float64(period)
		}
	}

	if avgLoss == 0 {
		return 100.0
	}
	rs := avgGain / avgLoss
	return 100.0 - 100.0/(1+rs)
}

func computeATR(highs, lows, closes []float64, period int) float64 {
	n := len(closes)
	if n <= 1 {
		return 0
	}
	if period > n-1 {
		period = n - 1
	}
	trs := make([]float64, n-1)
	for i := 1; i < n; i++ {
		hl := highs[i] - lows[i]
		hc := math.Abs(highs[i] - closes[i-1])
		lc := math.Abs(lows[i] - closes[i-1])
		trs[i-1] = math.Max(hl, math.Max(hc, lc))
	}
	if len(trs) == 0 {
		return 0
	}
	// Wilder smoothed ATR.
	atr := meanFloat(trs[:min(period, len(trs))])
	for _, tr := range trs[min(period, len(trs)):] {
		atr = (atr*float64(period-1) + tr) / float64(period)
	}
	return atr
}

func computeADX(highs, lows, closes []float64, period int) float64 {
	n := len(closes)
	if n <= period+1 {
		return 20.0
	}
	dmPlus := make([]float64, n-1)
	dmMinus := make([]float64, n-1)
	trs := make([]float64, n-1)
	for i := 1; i < n; i++ {
		up := highs[i] - highs[i-1]
		down := lows[i-1] - lows[i]
		if up > down && up > 0 {
			dmPlus[i-1] = up
		}
		if down > up && down > 0 {
			dmMinus[i-1] = down
		}
		hl := highs[i] - lows[i]
		hpc := math.Abs(highs[i] - closes[i-1])
		lpc := math.Abs(lows[i] - closes[i-1])
		trs[i-1] = math.Max(hl, math.Max(hpc, lpc))
	}

	smoothTR := sumFloat(trs[:period])
	smoothPlus := sumFloat(dmPlus[:period])
	smoothMinus := sumFloat(dmMinus[:period])

	diPlusArr := make([]float64, 0, len(trs)-period)
	diMinusArr := make([]float64, 0, len(trs)-period)
	dxArr := make([]float64, 0, len(trs)-period)

	for i := period; i < len(trs); i++ {
		smoothTR = smoothTR - smoothTR/float64(period) + trs[i]
		smoothPlus = smoothPlus - smoothPlus/float64(period) + dmPlus[i]
		smoothMinus = smoothMinus - smoothMinus/float64(period) + dmMinus[i]
		diPlus := 100 * smoothPlus / (smoothTR + 1e-10)
		diMinus := 100 * smoothMinus / (smoothTR + 1e-10)
		dx := 100 * math.Abs(diPlus-diMinus) / (diPlus + diMinus + 1e-10)
		diPlusArr = append(diPlusArr, diPlus)
		diMinusArr = append(diMinusArr, diMinus)
		dxArr = append(dxArr, dx)
	}

	if len(dxArr) == 0 {
		return 20.0
	}
	// Smooth DX into ADX.
	adx := meanFloat(dxArr[:min(period, len(dxArr))])
	for _, dx := range dxArr[min(period, len(dxArr)):] {
		adx = (adx*float64(period-1) + dx) / float64(period)
	}
	_ = diPlusArr
	_ = diMinusArr
	return adx
}

func computeBBWidth(closes []float64, period int, numStd float64) float64 {
	n := len(closes)
	if n < period {
		return 0
	}
	window := closes[n-period:]
	ma := meanFloat(window)
	std := stdDev(window)
	upper := ma + numStd*std
	lower := ma - numStd*std
	if ma == 0 {
		return 0
	}
	return (upper - lower) / ma
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics helpers
// ─────────────────────────────────────────────────────────────────────────────

func extractCloses(bars []*pbm.Bar) []float64 {
	out := make([]float64, len(bars))
	for i, b := range bars {
		out[i] = b.Close
	}
	return out
}

func extractVolumes(bars []*pbm.Bar) []float64 {
	out := make([]float64, len(bars))
	for i, b := range bars {
		out[i] = b.Volume
	}
	return out
}

func extractHighs(bars []*pbm.Bar) []float64 {
	out := make([]float64, len(bars))
	for i, b := range bars {
		out[i] = b.High
	}
	return out
}

func extractLows(bars []*pbm.Bar) []float64 {
	out := make([]float64, len(bars))
	for i, b := range bars {
		out[i] = b.Low
	}
	return out
}

func logReturns(closes []float64) []float64 {
	if len(closes) < 2 {
		return nil
	}
	out := make([]float64, len(closes)-1)
	for i := 1; i < len(closes); i++ {
		out[i-1] = math.Log(closes[i] / (closes[i-1] + 1e-10))
	}
	return out
}

func meanFloat(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func sumFloat(xs []float64) float64 {
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s
}

func stdDev(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := meanFloat(xs)
	v := 0.0
	for _, x := range xs {
		d := x - m
		v += d * d
	}
	return math.Sqrt(v / float64(len(xs)-1))
}

func zScoreCloses(closes []float64, window int) float64 {
	n := len(closes)
	if n < window+1 {
		return 0
	}
	rets := logReturns(closes)
	recent := rets[len(rets)-window:]
	return (recent[len(recent)-1] - meanFloat(recent)) / (stdDev(recent) + 1e-10)
}

func zScoreValues(vals []float64, window int) float64 {
	n := len(vals)
	if n < window {
		return 0
	}
	recent := vals[n-window:]
	return (vals[n-1] - meanFloat(recent)) / (stdDev(recent) + 1e-10)
}

func percentileRank(sorted []float64, val float64) float64 {
	if len(sorted) == 0 {
		return 0.5
	}
	cp := make([]float64, len(sorted))
	copy(cp, sorted)
	sort.Float64s(cp)
	below := 0
	for _, v := range cp {
		if v < val {
			below++
		}
	}
	return float64(below) / float64(len(cp))
}

func linearRegressionSlope(xs []float64) float64 {
	n := float64(len(xs))
	if n < 2 {
		return 0
	}
	sumX, sumY, sumXY, sumX2 := 0.0, 0.0, 0.0, 0.0
	for i, y := range xs {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}
	denom := n*sumX2 - sumX*sumX
	if denom == 0 {
		return 0
	}
	return (n*sumXY - sumX*sumY) / denom
}

func rollingStdDev(xs []float64, window int) []float64 {
	if len(xs) < window {
		return nil
	}
	out := make([]float64, len(xs)-window+1)
	for i := 0; i <= len(xs)-window; i++ {
		out[i] = stdDev(xs[i : i+window])
	}
	return out
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// ─────────────────────────────────────────────────────────────────────────────
// Subscriber management
// ─────────────────────────────────────────────────────────────────────────────

func (s *StrategyServer) addSigSub(symbol string, ch chan *pb.SignalEvent) {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	s.sigSubs[symbol] = append(s.sigSubs[symbol], ch)
}

func (s *StrategyServer) removeSigSub(symbol string, ch chan *pb.SignalEvent) {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	subs := s.sigSubs[symbol]
	updated := subs[:0]
	for _, c := range subs {
		if c != ch {
			updated = append(updated, c)
		}
	}
	s.sigSubs[symbol] = updated
}

func (s *StrategyServer) broadcastSignal(sig *pb.Signal) {
	s.seqMu.Lock()
	s.seqCounter++
	seq := s.seqCounter
	s.seqMu.Unlock()

	evt := &pb.SignalEvent{
		Signal:   sig,
		EventId:  fmt.Sprintf("sig-%d", seq),
		Sequence: seq,
	}

	s.subMu.RLock()
	defer s.subMu.RUnlock()

	send := func(chs []chan *pb.SignalEvent) {
		for _, ch := range chs {
			select {
			case ch <- evt:
			default:
			}
		}
	}
	send(s.sigSubs[sig.Symbol])
	send(s.sigSubs["*"]) // wildcard subscribers
}

func containsStr(ss []string, s string) bool {
	for _, v := range ss {
		if v == s {
			return true
		}
	}
	return false
}
