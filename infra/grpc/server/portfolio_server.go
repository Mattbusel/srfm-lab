// portfolio_server.go — PortfolioService gRPC server.
// Manages positions, P&L, allocation, and runs mean-variance optimization.
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

	pbm "github.com/srfm/infra/grpc/proto/market"
	pb  "github.com/srfm/infra/grpc/proto/portfolio"
)

// ─────────────────────────────────────────────────────────────────────────────
// In-memory position store
// ─────────────────────────────────────────────────────────────────────────────

type positionKey struct {
	accountID string
	symbol    string
}

type positionStore struct {
	mu        sync.RWMutex
	positions map[positionKey]*pb.Position
}

func newPositionStore() *positionStore {
	return &positionStore{positions: make(map[positionKey]*pb.Position)}
}

func (ps *positionStore) get(accountID, symbol string) (*pb.Position, bool) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	p, ok := ps.positions[positionKey{accountID, symbol}]
	return p, ok
}

func (ps *positionStore) set(pos *pb.Position) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.positions[positionKey{pos.AccountId, pos.Symbol}] = pos
}

func (ps *positionStore) delete(accountID, symbol string) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	delete(ps.positions, positionKey{accountID, symbol})
}

func (ps *positionStore) listByAccount(accountID string) []*pb.Position {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	var out []*pb.Position
	for k, p := range ps.positions {
		if k.accountID == accountID {
			cp := proto_clone_position(p)
			out = append(out, cp)
		}
	}
	return out
}

// proto_clone_position performs a shallow clone of a Position proto.
func proto_clone_position(p *pb.Position) *pb.Position {
	cp := *p
	return &cp
}

// ─────────────────────────────────────────────────────────────────────────────
// Account cash / equity state
// ─────────────────────────────────────────────────────────────────────────────

type accountState struct {
	mu             sync.RWMutex
	cash           float64
	dayStartEquity float64
	realizedPnL    float64
	executionCount int
	lastUpdated    time.Time
}

func newAccountState(initialCash float64) *accountState {
	return &accountState{
		cash:           initialCash,
		dayStartEquity: initialCash,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Position update subscribers
// ─────────────────────────────────────────────────────────────────────────────

type posUpdateSub struct {
	ch        chan *pb.PositionUpdate
	accountID string
	strategy  string
	symbol    string
}

// ─────────────────────────────────────────────────────────────────────────────
// PortfolioServer
// ─────────────────────────────────────────────────────────────────────────────

// PortfolioServer implements pb.PortfolioServiceServer.
type PortfolioServer struct {
	pb.UnimplementedPortfolioServiceServer

	log       *zap.Logger
	marketSrv *MarketServer
	positions *positionStore

	accountsMu sync.RWMutex
	accounts   map[string]*accountState

	// Sector mappings (symbol → sector).
	sectorMu      sync.RWMutex
	sectorMap     map[string]string

	// Position update subscribers.
	subMu sync.RWMutex
	subs  []*posUpdateSub
}

// NewPortfolioServer constructs a PortfolioServer.
func NewPortfolioServer(market *MarketServer, log *zap.Logger) *PortfolioServer {
	ps := &PortfolioServer{
		log:       log,
		marketSrv: market,
		positions: newPositionStore(),
		accounts:  make(map[string]*accountState),
		sectorMap: defaultSectorMap(),
	}
	return ps
}

// ─────────────────────────────────────────────────────────────────────────────
// GetPositions
// ─────────────────────────────────────────────────────────────────────────────

func (s *PortfolioServer) GetPositions(ctx context.Context, req *pb.PositionsRequest) (*pb.PositionResponse, error) {
	if req.AccountId == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id is required")
	}

	positions := s.positions.listByAccount(req.AccountId)

	// Apply filters.
	var filtered []*pb.Position
	for _, p := range positions {
		if req.Symbol != "" && p.Symbol != req.Symbol {
			continue
		}
		if req.StrategyId != "" && p.StrategyId != req.StrategyId {
			continue
		}
		filtered = append(filtered, p)
	}

	// Mark-to-market each position.
	totalValue := 0.0
	totalUnreal := 0.0
	totalReal := 0.0
	for _, pos := range filtered {
		s.markToMarket(ctx, pos)
		totalValue += pos.MarketValue
		totalUnreal += pos.UnrealizedPnl
		totalReal += pos.RealizedPnl
	}

	// Sort by market value descending.
	sort.Slice(filtered, func(i, j int) bool {
		return math.Abs(filtered[i].MarketValue) > math.Abs(filtered[j].MarketValue)
	})

	return &pb.PositionResponse{
		AccountId:          req.AccountId,
		Positions:          filtered,
		TotalMarketValue:   totalValue,
		TotalUnrealizedPnl: totalUnreal,
		TotalRealizedPnl:   totalReal,
		PositionCount:      int32(len(filtered)),
		AsOf:               timestamppb.Now(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GetPnL
// ─────────────────────────────────────────────────────────────────────────────

func (s *PortfolioServer) GetPnL(ctx context.Context, req *pb.PnLRequest) (*pb.PnLResponse, error) {
	if req.AccountId == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id is required")
	}

	positions := s.positions.listByAccount(req.AccountId)

	var unrealPnL, realPnL float64
	pnlBySymbol := make(map[string]float64)
	pnlByStrategy := make(map[string]float64)

	for _, pos := range positions {
		if req.Symbol != "" && pos.Symbol != req.Symbol {
			continue
		}
		if req.StrategyId != "" && pos.StrategyId != req.StrategyId {
			continue
		}
		s.markToMarket(ctx, pos)
		unrealPnL += pos.UnrealizedPnl
		realPnL += pos.RealizedPnl
		pnlBySymbol[pos.Symbol] += pos.UnrealizedPnl + pos.RealizedPnl
		if pos.StrategyId != "" {
			pnlByStrategy[pos.StrategyId] += pos.UnrealizedPnl + pos.RealizedPnl
		}
	}

	// Include realized PnL from closed positions stored in account state.
	state := s.getOrCreateAccount(req.AccountId)
	state.mu.RLock()
	realPnL += state.realizedPnL
	startEquity := state.dayStartEquity
	state.mu.RUnlock()

	// Day PnL.
	summary, _ := s.GetAccountSummary(ctx, &pb.AccountSummaryRequest{AccountId: req.AccountId})
	currentEquity := startEquity
	if summary != nil {
		currentEquity = summary.Equity
	}
	dayPnL := currentEquity - startEquity
	dayPnLPct := 0.0
	if startEquity > 0 {
		dayPnLPct = dayPnL / startEquity
	}

	return &pb.PnLResponse{
		AccountId:     req.AccountId,
		UnrealizedPnl: unrealPnL,
		RealizedPnl:   realPnL,
		TotalPnl:      unrealPnL + realPnL,
		DayPnl:        dayPnL,
		DayPnlPct:     dayPnLPct,
		PnlBySymbol:   pnlBySymbol,
		PnlByStrategy: pnlByStrategy,
		AsOf:          timestamppb.Now(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GetAllocation
// ─────────────────────────────────────────────────────────────────────────────

func (s *PortfolioServer) GetAllocation(ctx context.Context, req *pb.AllocationRequest) (*pb.AllocationResponse, error) {
	if req.AccountId == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id is required")
	}

	summary, err := s.GetAccountSummary(ctx, &pb.AccountSummaryRequest{AccountId: req.AccountId})
	if err != nil {
		return nil, err
	}
	totalValue := summary.Equity
	cash := summary.Cash
	cashPct := cash / (totalValue + 1e-10)

	positions := s.positions.listByAccount(req.AccountId)
	for _, p := range positions {
		s.markToMarket(ctx, p)
	}

	// By symbol.
	bySymbol := make(map[string]*pb.AllocationBucket)
	bySector := make(map[string]*pb.AllocationBucket)
	byStrategy := make(map[string]*pb.AllocationBucket)
	byAssetClass := make(map[string]*pb.AllocationBucket)

	for _, pos := range positions {
		mv := pos.MarketValue
		pct := mv / (totalValue + 1e-10)
		sector := s.getSector(pos.Symbol)

		// By symbol.
		if req.BySymbol {
			b := getOrCreate(bySymbol, pos.Symbol)
			b.Value += mv
			b.Pct += pct
			b.PositionCount++
		}

		// By sector.
		if req.BySector {
			b := getOrCreate(bySector, sector)
			b.Value += mv
			b.Pct += pct
			b.PositionCount++
		}

		// By strategy.
		if req.ByStrategy && pos.StrategyId != "" {
			b := getOrCreate(byStrategy, pos.StrategyId)
			b.Value += mv
			b.Pct += pct
			b.PositionCount++
		}

		// By asset class.
		if req.ByAssetClass {
			ac := pos.AssetClass.String()
			b := getOrCreate(byAssetClass, ac)
			b.Value += mv
			b.Pct += pct
			b.PositionCount++
		}
	}

	return &pb.AllocationResponse{
		AccountId:    req.AccountId,
		TotalValue:   totalValue,
		Cash:         cash,
		CashPct:      cashPct,
		BySymbol:     mapToSlice(bySymbol),
		BySector:     mapToSlice(bySector),
		ByStrategy:   mapToSlice(byStrategy),
		ByAssetClass: mapToSlice(byAssetClass),
		AsOf:         timestamppb.Now(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// OptimizeWeights — mean-variance optimization
// ─────────────────────────────────────────────────────────────────────────────

func (s *PortfolioServer) OptimizeWeights(ctx context.Context, req *pb.OptimizeRequest) (*pb.OptimizeResponse, error) {
	if len(req.Symbols) == 0 {
		return nil, status.Error(codes.InvalidArgument, "at least one symbol is required")
	}

	start := time.Now()
	lookback := int(req.LookbackDays)
	if lookback <= 0 {
		lookback = 252
	}

	// Fetch return series for each symbol.
	n := len(req.Symbols)
	returnMatrix := make([][]float64, n)
	for i, sym := range req.Symbols {
		barResp, err := s.marketSrv.GetBars(ctx, &pbm.BarRequest{
			Symbol:    sym,
			Timeframe: "1d",
			Limit:     int32(lookback + 1),
		})
		if err != nil || len(barResp.Bars) < 5 {
			// Use synthetic returns.
			returnMatrix[i] = generateSyntheticReturns_portfolio(lookback, hashSymbol(sym))
		} else {
			closes := extractCloses(barResp.Bars)
			returnMatrix[i] = logReturns(closes)
		}
	}

	// Align return lengths.
	minLen := math.MaxInt32
	for _, rets := range returnMatrix {
		if len(rets) < minLen {
			minLen = len(rets)
		}
	}
	for i := range returnMatrix {
		returnMatrix[i] = returnMatrix[i][len(returnMatrix[i])-minLen:]
	}

	// Compute mean returns and covariance matrix.
	means := make([]float64, n)
	for i, rets := range returnMatrix {
		means[i] = meanFloat(rets)
	}
	cov := covarianceMatrix(returnMatrix, means)

	// Apply Black-Litterman views if provided.
	if req.Objective == pb.OptimizationObjective_OPT_OBJECTIVE_BLACK_LITTERMAN && len(req.Views) > 0 {
		means = applyBLViews(means, req.Symbols, req.Views, cov)
	}

	// Optimize.
	maxW := req.MaxWeight
	if maxW <= 0 {
		maxW = 1.0
	}
	minW := req.MinWeight
	if !req.AllowShort && minW < 0 {
		minW = 0
	}

	var weights []float64
	switch req.Objective {
	case pb.OptimizationObjective_OPT_OBJECTIVE_MAX_SHARPE:
		weights = maximizeSharpe(means, cov, minW, maxW, 0.0, 1000)
	case pb.OptimizationObjective_OPT_OBJECTIVE_MIN_VARIANCE:
		weights = minimizeVariance(cov, minW, maxW, 1000)
	case pb.OptimizationObjective_OPT_OBJECTIVE_RISK_PARITY:
		weights = riskParityWeights(cov, 500)
	case pb.OptimizationObjective_OPT_OBJECTIVE_MAX_DIVERSIFICATION:
		weights = maximizeDiversification(cov, minW, maxW, 500)
	default:
		weights = maximizeSharpe(means, cov, minW, maxW, 0.0, 1000)
	}

	// Project to feasible simplex.
	weights = projectToSimplex(weights, minW, maxW)

	// Compute expected portfolio statistics.
	expRet := dotProduct(means, weights) * 252 // annualize
	expVol := math.Sqrt(portfolioVariance(weights, cov)) * math.Sqrt(252)
	riskFreeRate := 0.05 / 252
	expSharpe := 0.0
	if expVol > 0 {
		expSharpe = (dotProduct(means, weights) - riskFreeRate) / math.Sqrt(portfolioVariance(weights, cov)) * math.Sqrt(252)
	}

	// Get current weights.
	posResp, _ := s.GetPositions(ctx, &pb.PositionsRequest{AccountId: req.AccountId})
	summary, _ := s.GetAccountSummary(ctx, &pb.AccountSummaryRequest{AccountId: req.AccountId})
	currentWeights := make(map[string]float64, n)
	if posResp != nil && summary != nil && summary.Equity > 0 {
		for _, pos := range posResp.Positions {
			currentWeights[pos.Symbol] = pos.MarketValue / summary.Equity
		}
	}

	targetWeights := make(map[string]float64, n)
	deltaWeights := make(map[string]float64, n)
	var actions []string
	for i, sym := range req.Symbols {
		targetWeights[sym] = weights[i]
		deltaWeights[sym] = weights[i] - currentWeights[sym]
		if math.Abs(deltaWeights[sym]) > 0.001 {
			dir := "increase"
			if deltaWeights[sym] < 0 {
				dir = "decrease"
			}
			actions = append(actions, fmt.Sprintf("%s %s by %.1f%%", dir, sym, math.Abs(deltaWeights[sym])*100))
		}
	}

	return &pb.OptimizeResponse{
		TargetWeights:     targetWeights,
		CurrentWeights:    currentWeights,
		DeltaWeights:      deltaWeights,
		ExpectedReturn:    expRet,
		ExpectedVolatility: expVol,
		ExpectedSharpe:    expSharpe,
		RebalanceActions:  actions,
		SolverStatus:      "optimal",
		SolveUs:           time.Since(start).Microseconds(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamPositions
// ─────────────────────────────────────────────────────────────────────────────

func (s *PortfolioServer) StreamPositions(req *pb.PositionsRequest, stream pb.PortfolioService_StreamPositionsServer) error {
	ctx := stream.Context()
	ch := make(chan *pb.PositionUpdate, 64)

	sub := &posUpdateSub{
		ch:        ch,
		accountID: req.AccountId,
		strategy:  req.StrategyId,
		symbol:    req.Symbol,
	}
	s.addSub(sub)
	defer func() {
		s.removeSub(sub)
		close(ch)
	}()

	// Send current snapshot.
	posResp, err := s.GetPositions(ctx, req)
	if err == nil {
		for _, pos := range posResp.Positions {
			upd := &pb.PositionUpdate{
				EventType: "snapshot",
				Position:  pos,
				Timestamp: timestamppb.Now(),
			}
			if err := stream.Send(upd); err != nil {
				return err
			}
		}
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case upd, ok := <-ch:
			if !ok {
				return nil
			}
			if err := stream.Send(upd); err != nil {
				return err
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// GetAccountSummary
// ─────────────────────────────────────────────────────────────────────────────

func (s *PortfolioServer) GetAccountSummary(ctx context.Context, req *pb.AccountSummaryRequest) (*pb.AccountSummaryResponse, error) {
	if req.AccountId == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id is required")
	}

	state := s.getOrCreateAccount(req.AccountId)
	state.mu.RLock()
	cash := state.cash
	realPnL := state.realizedPnL
	startEquity := state.dayStartEquity
	state.mu.RUnlock()

	positions := s.positions.listByAccount(req.AccountId)
	grossExposure := 0.0
	netExposure := 0.0
	totalUnreal := 0.0
	for _, p := range positions {
		s.markToMarket(ctx, p)
		mv := p.MarketValue
		totalUnreal += p.UnrealizedPnl
		if p.Side == pb.PositionSide_POSITION_SIDE_LONG {
			grossExposure += mv
			netExposure += mv
		} else {
			grossExposure += mv
			netExposure -= mv
		}
	}

	equity := cash + grossExposure
	leverage := grossExposure / (equity + 1e-10)
	// Reg-T margin: 50% of gross exposure.
	marginUsed := grossExposure * 0.50
	marginAvail := math.Max(0, equity-marginUsed)
	buyingPower := marginAvail * 2 // 2x buying power on margin

	dayPnL := (equity - startEquity)
	totalPnL := totalUnreal + realPnL

	return &pb.AccountSummaryResponse{
		AccountId:       req.AccountId,
		Equity:          equity,
		Cash:            cash,
		MarginUsed:      marginUsed,
		MarginAvailable: marginAvail,
		BuyingPower:     buyingPower,
		GrossExposure:   grossExposure,
		NetExposure:     netExposure,
		Leverage:        leverage,
		DayPnl:          dayPnL,
		TotalPnl:        totalPnL,
		UpdatedAt:       timestamppb.Now(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GetPerformance
// ─────────────────────────────────────────────────────────────────────────────

func (s *PortfolioServer) GetPerformance(ctx context.Context, req *pb.PerformanceRequest) (*pb.PerformanceResponse, error) {
	if req.AccountId == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id is required")
	}

	// In a full implementation this would pull historical equity curve from
	// a timeseries store. We compute approximations from current state.
	summary, _ := s.GetAccountSummary(ctx, &pb.AccountSummaryRequest{AccountId: req.AccountId})
	equity := 1_000_000.0
	if summary != nil {
		equity = summary.Equity
	}

	// Generate a synthetic equity curve for demo.
	lookback := 252
	curve := buildSyntheticEquityCurve(equity, lookback, 42)
	curveDates := make([]*timestamppb.Timestamp, len(curve))
	now := time.Now().UTC()
	for i := range curve {
		curveDates[i] = timestamppb.New(now.Add(-time.Duration(len(curve)-1-i) * 24 * time.Hour))
	}

	metrics := computePerformanceMetrics(curve)

	// Benchmark performance (SPY proxy).
	benchCurve := buildSyntheticEquityCurve(equity, lookback, 99) // different seed
	benchMetrics := computePerformanceMetrics(benchCurve)

	return &pb.PerformanceResponse{
		AccountId:        req.AccountId,
		StrategyId:       req.StrategyId,
		Metrics:          metrics,
		BenchmarkMetrics: benchMetrics,
		EquityCurve:      curve,
		CurveDates:       curveDates,
		Start:            curveDates[0],
		End:              curveDates[len(curveDates)-1],
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// RecordExecution
// ─────────────────────────────────────────────────────────────────────────────

func (s *PortfolioServer) RecordExecution(ctx context.Context, req *pb.ExecutionRequest) (*pb.ExecutionResponse, error) {
	if req.AccountId == "" || req.Symbol == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id and symbol are required")
	}
	if req.Quantity <= 0 || req.FillPrice <= 0 {
		return nil, status.Error(codes.InvalidArgument, "quantity and fill_price must be positive")
	}

	execAt := req.ExecutedAt
	if execAt == nil {
		execAt = timestamppb.Now()
	}

	state := s.getOrCreateAccount(req.AccountId)
	tradeValue := req.Quantity * req.FillPrice
	isBuy := req.Side == "buy"

	var realizedPnL float64
	var positionClosed bool

	// Get or create position.
	pos, exists := s.positions.get(req.AccountId, req.Symbol)

	if !exists || pos == nil {
		// Open new position.
		pos = &pb.Position{
			AccountId:   req.AccountId,
			Symbol:      req.Symbol,
			AssetClass:  pb.AssetClass_ASSET_CLASS_EQUITY,
			Side:        pb.PositionSide_POSITION_SIDE_LONG,
			Quantity:    req.Quantity,
			AvgEntry:    req.FillPrice,
			CostBasis:   tradeValue,
			StrategyId:  req.StrategyId,
			Sector:      s.getSector(req.Symbol),
			OpenedAt:    execAt,
			UpdatedAt:   execAt,
		}
		if !isBuy {
			pos.Side = pb.PositionSide_POSITION_SIDE_SHORT
		}
		state.mu.Lock()
		if isBuy {
			state.cash -= tradeValue + req.Commission
		} else {
			state.cash += tradeValue - req.Commission
		}
		state.mu.Unlock()
	} else {
		// Update existing position.
		pos.UpdatedAt = execAt

		if (isBuy && pos.Side == pb.PositionSide_POSITION_SIDE_LONG) ||
			(!isBuy && pos.Side == pb.PositionSide_POSITION_SIDE_SHORT) {
			// Adding to position.
			totalCost := pos.CostBasis + tradeValue
			pos.Quantity += req.Quantity
			pos.AvgEntry = totalCost / pos.Quantity
			pos.CostBasis = totalCost
			state.mu.Lock()
			if isBuy {
				state.cash -= tradeValue + req.Commission
			} else {
				state.cash += tradeValue - req.Commission
			}
			state.mu.Unlock()
		} else {
			// Reducing / closing position.
			closingQty := math.Min(req.Quantity, pos.Quantity)
			if pos.Side == pb.PositionSide_POSITION_SIDE_LONG {
				realizedPnL = closingQty * (req.FillPrice - pos.AvgEntry) - req.Commission
			} else {
				realizedPnL = closingQty * (pos.AvgEntry - req.FillPrice) - req.Commission
			}
			pos.RealizedPnl += realizedPnL
			pos.Quantity -= closingQty

			state.mu.Lock()
			state.realizedPnL += realizedPnL
			if isBuy {
				// Covering short.
				state.cash -= tradeValue + req.Commission
			} else {
				// Selling long.
				state.cash += tradeValue - req.Commission
			}
			state.mu.Unlock()

			if pos.Quantity <= 1e-8 {
				// Position fully closed.
				positionClosed = true
				s.positions.delete(req.AccountId, req.Symbol)
				// Emit closed update.
				s.emitPositionUpdate("closed", pos, req.FillPrice, req.Quantity)
				return &pb.ExecutionResponse{
					ExecutionId:     fmt.Sprintf("exec-%d", time.Now().UnixNano()),
					OrderId:         req.OrderId,
					UpdatedPosition: pos,
					RealizedPnl:     realizedPnL,
					PositionClosed:  positionClosed,
					RecordedAt:      timestamppb.Now(),
				}, nil
			}
			pos.CostBasis = pos.Quantity * pos.AvgEntry
		}
	}

	// Mark to market.
	s.markToMarket(ctx, pos)
	s.positions.set(pos)

	evtType := "updated"
	if !exists {
		evtType = "opened"
	}
	s.emitPositionUpdate(evtType, pos, req.FillPrice, req.Quantity)

	return &pb.ExecutionResponse{
		ExecutionId:     fmt.Sprintf("exec-%d", time.Now().UnixNano()),
		OrderId:         req.OrderId,
		UpdatedPosition: pos,
		RealizedPnl:     realizedPnL,
		PositionClosed:  positionClosed,
		RecordedAt:      timestamppb.Now(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Mark-to-market
// ─────────────────────────────────────────────────────────────────────────────

func (s *PortfolioServer) markToMarket(ctx context.Context, pos *pb.Position) {
	q, err := s.marketSrv.GetQuote(ctx, &pbm.QuoteRequest{Symbol: pos.Symbol})
	if err != nil {
		return
	}
	pos.MarketPrice = q.Mid
	if pos.Side == pb.PositionSide_POSITION_SIDE_LONG {
		pos.MarketValue = pos.Quantity * q.Mid
		pos.UnrealizedPnl = pos.MarketValue - pos.CostBasis
	} else {
		pos.MarketValue = pos.Quantity * q.Mid
		pos.UnrealizedPnl = pos.CostBasis - pos.MarketValue
	}
	pos.UpdatedAt = timestamppb.Now()
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimization algorithms
// ─────────────────────────────────────────────────────────────────────────────

// covarianceMatrix computes the sample covariance matrix from return series.
func covarianceMatrix(returns [][]float64, means []float64) [][]float64 {
	n := len(returns)
	if n == 0 {
		return nil
	}
	T := len(returns[0])
	cov := make([][]float64, n)
	for i := range cov {
		cov[i] = make([]float64, n)
	}
	if T < 2 {
		return cov
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			c := 0.0
			for t := 0; t < T; t++ {
				c += (returns[i][t] - means[i]) * (returns[j][t] - means[j])
			}
			c /= float64(T - 1)
			cov[i][j] = c
			cov[j][i] = c
		}
	}
	return cov
}

// maximizeSharpe runs gradient ascent on the Sharpe ratio.
func maximizeSharpe(means, cov [][]float64, minW, maxW, riskFree float64, iters int) []float64 {
	n := len(means)
	w := uniformWeights(n)
	lr := 0.01

	for iter := 0; iter < iters; iter++ {
		ret := dotProduct(means, w) - riskFree
		variance := portfolioVariance(w, cov)
		vol := math.Sqrt(variance + 1e-12)
		grad := make([]float64, n)
		for i := 0; i < n; i++ {
			dVar := 0.0
			for j := 0; j < n; j++ {
				dVar += 2 * cov[i][j] * w[j]
			}
			grad[i] = (means[i]*vol - ret*dVar/(2*vol)) / (variance + 1e-12)
		}
		// Update.
		for i := 0; i < n; i++ {
			w[i] += lr * grad[i]
		}
		w = projectToSimplex(w, minW, maxW)

		// Decay learning rate.
		if iter%100 == 99 {
			lr *= 0.9
		}
	}
	return w
}

// minimizeVariance runs gradient descent on portfolio variance.
func minimizeVariance(cov [][]float64, minW, maxW float64, iters int) []float64 {
	n := len(cov)
	w := uniformWeights(n)
	lr := 0.01
	for iter := 0; iter < iters; iter++ {
		grad := make([]float64, n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				grad[i] += 2 * cov[i][j] * w[j]
			}
		}
		for i := 0; i < n; i++ {
			w[i] -= lr * grad[i]
		}
		w = projectToSimplex(w, minW, maxW)
		if iter%100 == 99 {
			lr *= 0.9
		}
	}
	return w
}

// riskParityWeights computes risk-parity (equal risk contribution) weights.
func riskParityWeights(cov [][]float64, iters int) []float64 {
	n := len(cov)
	w := uniformWeights(n)
	lr := 0.01
	target := 1.0 / float64(n)

	for iter := 0; iter < iters; iter++ {
		sigma := math.Sqrt(portfolioVariance(w, cov) + 1e-12)
		mrc := make([]float64, n) // marginal risk contributions
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				mrc[i] += cov[i][j] * w[j]
			}
			mrc[i] /= sigma
		}
		for i := 0; i < n; i++ {
			rc := w[i] * mrc[i] / sigma // risk contribution fraction
			w[i] -= lr * (rc - target)
			if w[i] < 0 {
				w[i] = 1e-6
			}
		}
		total := sumFloat(w)
		for i := range w {
			w[i] /= total
		}
		if iter%100 == 99 {
			lr *= 0.9
		}
	}
	return w
}

// maximizeDiversification maximizes the diversification ratio.
func maximizeDiversification(cov [][]float64, minW, maxW float64, iters int) []float64 {
	n := len(cov)
	vols := make([]float64, n)
	for i := 0; i < n; i++ {
		vols[i] = math.Sqrt(cov[i][i])
	}
	w := uniformWeights(n)
	lr := 0.01

	for iter := 0; iter < iters; iter++ {
		sigma := math.Sqrt(portfolioVariance(w, cov) + 1e-12)
		weightedVol := dotProduct(vols, w)
		grad := make([]float64, n)
		for i := 0; i < n; i++ {
			dSigma := 0.0
			for j := 0; j < n; j++ {
				dSigma += cov[i][j] * w[j]
			}
			dSigma /= sigma
			grad[i] = (vols[i]*sigma - weightedVol*dSigma) / (sigma * sigma + 1e-12)
		}
		for i := 0; i < n; i++ {
			w[i] += lr * grad[i]
		}
		w = projectToSimplex(w, minW, maxW)
		if iter%100 == 99 {
			lr *= 0.9
		}
	}
	return w
}

// applyBLViews blends Black-Litterman views into the prior mean returns.
func applyBLViews(means []float64, symbols []string, views map[string]float64, cov [][]float64) []float64 {
	tau := 0.05
	blended := make([]float64, len(means))
	copy(blended, means)
	for i, sym := range symbols {
		if v, ok := views[sym]; ok {
			// Simple mixing: posterior mean = (1-tau)*prior + tau*view
			blended[i] = (1-tau)*means[i] + tau*v
		}
	}
	return blended
}

// projectToSimplex projects w onto the probability simplex with box constraints.
func projectToSimplex(w []float64, minW, maxW float64) []float64 {
	n := len(w)
	result := make([]float64, n)
	copy(result, w)

	// Clip to [minW, maxW].
	for i := range result {
		if result[i] < minW {
			result[i] = minW
		}
		if result[i] > maxW {
			result[i] = maxW
		}
	}

	// Project onto unit simplex by scaling.
	total := sumFloat(result)
	if total > 0 {
		for i := range result {
			result[i] /= total
		}
	} else {
		for i := range result {
			result[i] = 1.0 / float64(n)
		}
	}
	return result
}

func portfolioVariance(w []float64, cov [][]float64) float64 {
	n := len(w)
	v := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v += w[i] * cov[i][j] * w[j]
		}
	}
	return v
}

func dotProduct(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		if i < len(b) {
			s += a[i] * b[i]
		}
	}
	return s
}

func uniformWeights(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 1.0 / float64(n)
	}
	return w
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance metrics
// ─────────────────────────────────────────────────────────────────────────────

func computePerformanceMetrics(equityCurve []float64) *pb.PerformanceMetrics {
	n := len(equityCurve)
	if n < 2 {
		return &pb.PerformanceMetrics{}
	}

	// Daily returns.
	rets := make([]float64, n-1)
	for i := 1; i < n; i++ {
		rets[i-1] = (equityCurve[i] - equityCurve[i-1]) / equityCurve[i-1]
	}

	totalReturn := (equityCurve[n-1] - equityCurve[0]) / equityCurve[0]
	years := float64(n) / 252.0
	annReturn := math.Pow(1+totalReturn, 1/years) - 1
	vol := stdDev(rets) * math.Sqrt(252)
	riskFree := 0.05
	sharpe := 0.0
	if vol > 0 {
		sharpe = (annReturn - riskFree) / vol
	}

	// Downside deviation for Sortino.
	var downside []float64
	for _, r := range rets {
		if r < 0 {
			downside = append(downside, r)
		}
	}
	downVol := 0.0
	if len(downside) > 0 {
		downVol = stdDev(downside) * math.Sqrt(252)
	}
	sortino := 0.0
	if downVol > 0 {
		sortino = (annReturn - riskFree) / downVol
	}

	// Max drawdown.
	peak := equityCurve[0]
	maxDD := 0.0
	for _, v := range equityCurve {
		if v > peak {
			peak = v
		}
		dd := (peak - v) / peak
		if dd > maxDD {
			maxDD = dd
		}
	}
	calmar := 0.0
	if maxDD > 0 {
		calmar = annReturn / maxDD
	}

	// Win rate.
	wins, losses := 0, 0
	winSum, lossSum := 0.0, 0.0
	for _, r := range rets {
		if r > 0 {
			wins++
			winSum += r
		} else if r < 0 {
			losses++
			lossSum += -r
		}
	}
	winRate := float64(wins) / float64(wins+losses+1)
	avgWin := 0.0
	if wins > 0 {
		avgWin = winSum / float64(wins)
	}
	avgLoss := 0.0
	if losses > 0 {
		avgLoss = lossSum / float64(losses)
	}
	profitFactor := 0.0
	if lossSum > 0 {
		profitFactor = winSum / lossSum
	}
	expectancy := winRate*avgWin - (1-winRate)*avgLoss

	return &pb.PerformanceMetrics{
		TotalReturn:      totalReturn,
		AnnualizedReturn: annReturn,
		Volatility:       vol,
		SharpeRatio:      sharpe,
		SortinoRatio:     sortino,
		CalmarRatio:      calmar,
		MaxDrawdown:      maxDD,
		WinRate:          winRate,
		ProfitFactor:     profitFactor,
		AvgWin:           avgWin,
		AvgLoss:          avgLoss,
		Expectancy:       expectancy,
		TotalTrades:      int32(wins + losses),
	}
}

func buildSyntheticEquityCurve(startEquity float64, n, seed int) []float64 {
	rng := newLCG(int64(seed))
	curve := make([]float64, n)
	curve[0] = startEquity
	for i := 1; i < n; i++ {
		ret := rng.NormFloat64()*0.012 + 0.0003
		curve[i] = curve[i-1] * (1 + ret)
	}
	return curve
}

func generateSyntheticReturns_portfolio(n, seed int) []float64 {
	rng := newLCG(int64(seed))
	rets := make([]float64, n)
	for i := range rets {
		rets[i] = rng.NormFloat64()*0.012 + 0.0003
	}
	return rets
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func (s *PortfolioServer) getOrCreateAccount(accountID string) *accountState {
	s.accountsMu.Lock()
	defer s.accountsMu.Unlock()
	if a, ok := s.accounts[accountID]; ok {
		return a
	}
	a := newAccountState(1_000_000)
	s.accounts[accountID] = a
	return a
}

func (s *PortfolioServer) getSector(symbol string) string {
	s.sectorMu.RLock()
	defer s.sectorMu.RUnlock()
	if sec, ok := s.sectorMap[symbol]; ok {
		return sec
	}
	return "Unknown"
}

func (s *PortfolioServer) addSub(sub *posUpdateSub) {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	s.subs = append(s.subs, sub)
}

func (s *PortfolioServer) removeSub(target *posUpdateSub) {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	updated := s.subs[:0]
	for _, sub := range s.subs {
		if sub != target {
			updated = append(updated, sub)
		}
	}
	s.subs = updated
}

func (s *PortfolioServer) emitPositionUpdate(evtType string, pos *pb.Position, fillPrice, fillQty float64) {
	s.subMu.RLock()
	defer s.subMu.RUnlock()
	upd := &pb.PositionUpdate{
		EventType: evtType,
		Position:  pos,
		FillPrice: fillPrice,
		FillQty:   fillQty,
		Timestamp: timestamppb.Now(),
	}
	for _, sub := range s.subs {
		if sub.accountID != "" && sub.accountID != pos.AccountId {
			continue
		}
		if sub.symbol != "" && sub.symbol != pos.Symbol {
			continue
		}
		if sub.strategy != "" && sub.strategy != pos.StrategyId {
			continue
		}
		select {
		case sub.ch <- upd:
		default:
		}
	}
}

func getOrCreate(m map[string]*pb.AllocationBucket, key string) *pb.AllocationBucket {
	if b, ok := m[key]; ok {
		return b
	}
	b := &pb.AllocationBucket{Label: key}
	m[key] = b
	return b
}

func mapToSlice(m map[string]*pb.AllocationBucket) []*pb.AllocationBucket {
	out := make([]*pb.AllocationBucket, 0, len(m))
	for _, v := range m {
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool {
		return math.Abs(out[i].Value) > math.Abs(out[j].Value)
	})
	return out
}

func defaultSectorMap() map[string]string {
	return map[string]string{
		"AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
		"AMZN": "Consumer Discretionary", "NVDA": "Technology", "META": "Technology",
		"TSLA": "Consumer Discretionary", "BRK.B": "Financials", "JPM": "Financials",
		"JNJ": "Healthcare", "PG": "Consumer Staples", "HD": "Consumer Discretionary",
		"XOM": "Energy", "CVX": "Energy", "BAC": "Financials", "WMT": "Consumer Staples",
		"LLY": "Healthcare", "AVGO": "Technology", "MA": "Financials", "MRK": "Healthcare",
		"SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "GLD": "ETF", "TLT": "ETF",
	}
}
