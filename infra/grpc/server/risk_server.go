// risk_server.go — RiskService gRPC server.
// Implements pre-trade checks, VaR computation (historical simulation,
// parametric, Monte Carlo), drawdown tracking, and risk event streaming.
package server

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	pbm  "github.com/srfm/infra/grpc/proto/market"
	pbp  "github.com/srfm/infra/grpc/proto/portfolio"
	pb   "github.com/srfm/infra/grpc/proto/risk"
)

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

// RiskServerConfig holds global risk parameters.
type RiskServerConfig struct {
	// Default risk limits applied to all accounts unless overridden.
	DefaultLimits pb.RiskLimits

	// VaR computation defaults.
	DefaultVaRConfidence  float64
	DefaultVaRHorizon     int
	DefaultVaRMethod      string
	DefaultVaRLookback    int
	MonteCarloSimulations int

	// Drawdown tracking.
	DrawdownLookbackDays int
}

func DefaultRiskServerConfig() RiskServerConfig {
	return RiskServerConfig{
		DefaultLimits: pb.RiskLimits{
			MaxPositionSize:        500_000,
			MaxPositionPct:         0.20,
			MaxDrawdownPct:         0.15,
			MaxDailyLoss:           50_000,
			MaxPortfolioVar:        100_000,
			MaxLeverage:            2.0,
			MaxConcentration:       0.20,
			MaxSectorConcentration: 0.35,
		},
		DefaultVaRConfidence:  0.99,
		DefaultVaRHorizon:     1,
		DefaultVaRMethod:      "historical",
		DefaultVaRLookback:    252,
		MonteCarloSimulations: 10_000,
		DrawdownLookbackDays:  365,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// In-process state
// ─────────────────────────────────────────────────────────────────────────────

type accountRiskState struct {
	mu           sync.RWMutex
	dailyPnL     float64
	peakEquity   float64
	currentEquity float64
	dayStartEquity float64
	lastUpdated  time.Time
}

func newAccountRiskState(initialEquity float64) *accountRiskState {
	return &accountRiskState{
		peakEquity:     initialEquity,
		currentEquity:  initialEquity,
		dayStartEquity: initialEquity,
	}
}

func (a *accountRiskState) updateEquity(equity float64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.currentEquity = equity
	if equity > a.peakEquity {
		a.peakEquity = equity
	}
	a.dailyPnL = equity - a.dayStartEquity
	a.lastUpdated = time.Now()
}

func (a *accountRiskState) drawdownPct() float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	if a.peakEquity == 0 {
		return 0
	}
	return (a.peakEquity - a.currentEquity) / a.peakEquity
}

type riskEventSub struct {
	ch         chan *pb.RiskEvent
	types      map[pb.RiskEventType]bool
	minSeverity pb.Severity
}

// ─────────────────────────────────────────────────────────────────────────────
// RiskServer
// ─────────────────────────────────────────────────────────────────────────────

// RiskServer implements pb.RiskServiceServer.
type RiskServer struct {
	pb.UnimplementedRiskServiceServer

	cfg        RiskServerConfig
	log        *zap.Logger
	marketSrv  *MarketServer
	portfolioSrv *PortfolioServer

	stateMu  sync.RWMutex
	accounts map[string]*accountRiskState

	// Risk limits per account (overrides defaults).
	limitsMu sync.RWMutex
	limits   map[string]*pb.RiskLimits

	// Event stream subscribers.
	subMu sync.RWMutex
	subs  map[string][]*riskEventSub // account_id -> subs

	// Event ID counter.
	eventSeq   int64
	eventSeqMu sync.Mutex
}

// NewRiskServer constructs a RiskServer.
func NewRiskServer(cfg RiskServerConfig, market *MarketServer, portfolio *PortfolioServer, log *zap.Logger) *RiskServer {
	return &RiskServer{
		cfg:          cfg,
		log:          log,
		marketSrv:    market,
		portfolioSrv: portfolio,
		accounts:     make(map[string]*accountRiskState),
		limits:       make(map[string]*pb.RiskLimits),
		subs:         make(map[string][]*riskEventSub),
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckPreTrade
// ─────────────────────────────────────────────────────────────────────────────

func (s *RiskServer) CheckPreTrade(ctx context.Context, req *pb.PreTradeRequest) (*pb.PreTradeResponse, error) {
	if req.AccountId == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id is required")
	}
	if req.Symbol == "" {
		return nil, status.Error(codes.InvalidArgument, "symbol is required")
	}
	if req.Quantity <= 0 {
		return nil, status.Error(codes.InvalidArgument, "quantity must be positive")
	}

	start := time.Now()
	limits := s.getLimits(req.AccountId)

	// Get portfolio state.
	portfolioResp, err := s.portfolioSrv.GetAccountSummary(ctx, &pbp.AccountSummaryRequest{AccountId: req.AccountId})
	if err != nil {
		// Allow trade if portfolio service unavailable, but warn.
		s.log.Warn("portfolio service unavailable for pre-trade check", zap.Error(err))
		portfolioResp = &pbp.AccountSummaryResponse{
			AccountId:       req.AccountId,
			Equity:          1_000_000, // assume 1M fallback
			BuyingPower:     500_000,
			MarginAvailable: 500_000,
		}
	}

	// Get current price if not provided.
	price := req.Price
	if price <= 0 {
		q, err := s.marketSrv.GetQuote(ctx, &pbm.QuoteRequest{Symbol: req.Symbol})
		if err != nil {
			return nil, status.Errorf(codes.Internal, "cannot determine price: %v", err)
		}
		if req.Side == "buy" || req.Side == "sell_short" {
			price = q.Ask
		} else {
			price = q.Bid
		}
	}

	tradeValue := req.Quantity * price
	portfolioValue := portfolioResp.Equity

	var breaches []*pb.RiskBreach

	// ── Check 1: Position size ───────────────────────────────────────────────
	if tradeValue > limits.MaxPositionSize {
		breaches = append(breaches, &pb.RiskBreach{
			EventType:    pb.RiskEventType_RISK_EVENT_POSITION_LIMIT,
			RuleId:       "max_position_size",
			Description:  fmt.Sprintf("Trade value $%.0f exceeds max position size $%.0f", tradeValue, limits.MaxPositionSize),
			CurrentValue: tradeValue,
			LimitValue:   limits.MaxPositionSize,
			BreachPct:    (tradeValue - limits.MaxPositionSize) / limits.MaxPositionSize * 100,
		})
	}

	// ── Check 2: Position % of portfolio ────────────────────────────────────
	positionPct := tradeValue / (portfolioValue + 1e-10)
	if positionPct > limits.MaxPositionPct {
		breaches = append(breaches, &pb.RiskBreach{
			EventType:    pb.RiskEventType_RISK_EVENT_POSITION_LIMIT,
			RuleId:       "max_position_pct",
			Description:  fmt.Sprintf("Position %.1f%% of portfolio exceeds max %.1f%%", positionPct*100, limits.MaxPositionPct*100),
			CurrentValue: positionPct,
			LimitValue:   limits.MaxPositionPct,
			BreachPct:    (positionPct - limits.MaxPositionPct) / limits.MaxPositionPct * 100,
		})
	}

	// ── Check 3: Available margin ────────────────────────────────────────────
	marginRequired := tradeValue * s.marginRequirement(req.Symbol, req.Side)
	marginAvail := portfolioResp.MarginAvailable
	if marginRequired > marginAvail {
		breaches = append(breaches, &pb.RiskBreach{
			EventType:    pb.RiskEventType_RISK_EVENT_MARGIN_CALL,
			RuleId:       "margin_available",
			Description:  fmt.Sprintf("Margin required $%.0f exceeds available $%.0f", marginRequired, marginAvail),
			CurrentValue: marginRequired,
			LimitValue:   marginAvail,
			BreachPct:    (marginRequired - marginAvail) / (marginAvail + 1) * 100,
		})
	}

	// ── Check 4: Daily loss limit ────────────────────────────────────────────
	state := s.getOrCreateState(req.AccountId, portfolioValue)
	state.mu.RLock()
	dailyPnL := state.dailyPnL
	state.mu.RUnlock()
	if dailyPnL < -limits.MaxDailyLoss {
		breaches = append(breaches, &pb.RiskBreach{
			EventType:    pb.RiskEventType_RISK_EVENT_DRAWDOWN_LIMIT,
			RuleId:       "max_daily_loss",
			Description:  fmt.Sprintf("Daily loss $%.0f exceeds limit $%.0f", -dailyPnL, limits.MaxDailyLoss),
			CurrentValue: -dailyPnL,
			LimitValue:   limits.MaxDailyLoss,
			BreachPct:    (-dailyPnL - limits.MaxDailyLoss) / limits.MaxDailyLoss * 100,
		})
	}

	// ── Check 5: Drawdown ────────────────────────────────────────────────────
	ddPct := state.drawdownPct()
	if ddPct > limits.MaxDrawdownPct {
		breaches = append(breaches, &pb.RiskBreach{
			EventType:    pb.RiskEventType_RISK_EVENT_DRAWDOWN_LIMIT,
			RuleId:       "max_drawdown",
			Description:  fmt.Sprintf("Drawdown %.1f%% exceeds limit %.1f%%", ddPct*100, limits.MaxDrawdownPct*100),
			CurrentValue: ddPct,
			LimitValue:   limits.MaxDrawdownPct,
			BreachPct:    (ddPct - limits.MaxDrawdownPct) / limits.MaxDrawdownPct * 100,
		})
	}

	// ── Check 6: Leverage ────────────────────────────────────────────────────
	newExposure := portfolioResp.GrossExposure + tradeValue
	newLeverage := newExposure / (portfolioValue + 1e-10)
	if newLeverage > limits.MaxLeverage {
		breaches = append(breaches, &pb.RiskBreach{
			EventType:    pb.RiskEventType_RISK_EVENT_CONCENTRATION,
			RuleId:       "max_leverage",
			Description:  fmt.Sprintf("Leverage %.2fx would exceed max %.2fx", newLeverage, limits.MaxLeverage),
			CurrentValue: newLeverage,
			LimitValue:   limits.MaxLeverage,
			BreachPct:    (newLeverage - limits.MaxLeverage) / limits.MaxLeverage * 100,
		})
	}

	// ── Make decision ────────────────────────────────────────────────────────
	decision := pb.RiskDecision_RISK_DECISION_APPROVED
	approvedQty := req.Quantity
	rejectReason := ""

	// Hard stops: drawdown limit or daily loss → reject.
	for _, b := range breaches {
		if b.RuleId == "max_drawdown" || b.RuleId == "max_daily_loss" {
			decision = pb.RiskDecision_RISK_DECISION_REJECTED
			rejectReason = b.Description
			approvedQty = 0
			break
		}
	}

	// Soft stops: size reduction.
	if decision == pb.RiskDecision_RISK_DECISION_APPROVED && len(breaches) > 0 {
		decision = pb.RiskDecision_RISK_DECISION_REDUCED
		// Scale down to the most binding constraint.
		maxAllowed := math.Min(
			limits.MaxPositionSize,
			portfolioValue*limits.MaxPositionPct,
		)
		if price > 0 {
			approvedQty = math.Floor(maxAllowed / price)
		}
		if approvedQty <= 0 {
			decision = pb.RiskDecision_RISK_DECISION_REJECTED
			rejectReason = "reduced quantity would be zero"
		}
	}

	// Emit risk events for breaches.
	for _, b := range breaches {
		sev := pb.Severity_SEVERITY_WARNING
		if decision == pb.RiskDecision_RISK_DECISION_REJECTED {
			sev = pb.Severity_SEVERITY_CRITICAL
		}
		s.emitRiskEvent(req.AccountId, b.EventType, sev, req.Symbol, b.Description, b.CurrentValue, b.LimitValue)
	}

	// Estimate post-trade VaR.
	varAfter := 0.0
	if decision != pb.RiskDecision_RISK_DECISION_REJECTED {
		varAfter = portfolioValue * 0.02 // simplified: 2% of portfolio
	}

	return &pb.PreTradeResponse{
		Decision:          decision,
		Breaches:          breaches,
		ApprovedQuantity:  approvedQty,
		RejectReason:      rejectReason,
		AvailableMargin:   marginAvail,
		RequiredMargin:    marginRequired,
		PortfolioVarAfter: varAfter,
		CheckUs:           time.Since(start).Microseconds(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GetVaR
// ─────────────────────────────────────────────────────────────────────────────

func (s *RiskServer) GetVaR(ctx context.Context, req *pb.VaRRequest) (*pb.VaRResponse, error) {
	if req.AccountId == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id is required")
	}

	conf := req.Confidence
	if conf <= 0 {
		conf = s.cfg.DefaultVaRConfidence
	}
	horizon := int(req.HorizonDays)
	if horizon <= 0 {
		horizon = s.cfg.DefaultVaRHorizon
	}
	method := req.Method
	if method == "" {
		method = s.cfg.DefaultVaRMethod
	}
	lookback := int(req.LookbackDays)
	if lookback <= 0 {
		lookback = s.cfg.DefaultVaRLookback
	}

	// Determine portfolio value.
	portfolioResp, err := s.portfolioSrv.GetAccountSummary(ctx, &pbp.AccountSummaryRequest{AccountId: req.AccountId})
	portfolioValue := 1_000_000.0
	if err == nil {
		portfolioValue = portfolioResp.Equity
	}

	// If symbol-level VaR, fetch returns for that symbol.
	var returns []float64
	if req.Symbol != "" {
		barResp, err := s.marketSrv.GetBars(ctx, &pbm.BarRequest{
			Symbol:    req.Symbol,
			Timeframe: "1d",
			Limit:     int32(lookback + 1),
		})
		if err == nil && len(barResp.Bars) > 1 {
			closes := extractCloses(barResp.Bars)
			returns = logReturns(closes)
		}
	} else {
		// Portfolio-level: use a blended synthetic return series.
		returns = generateSyntheticReturns(lookback, 0.0003, 0.012, 42)
	}

	if len(returns) < 30 {
		// Fall back to parametric with assumed params.
		returns = generateSyntheticReturns(lookback, 0.0003, 0.012, 0)
	}

	var varValue, cvarValue float64

	switch method {
	case "historical":
		varValue, cvarValue = historicalVaR(returns, conf, horizon, portfolioValue)
	case "parametric":
		varValue, cvarValue = parametricVaR(returns, conf, horizon, portfolioValue)
	case "monte_carlo":
		varValue, cvarValue = monteCarloVaR(returns, conf, horizon, portfolioValue, s.cfg.MonteCarloSimulations)
	default:
		varValue, cvarValue = historicalVaR(returns, conf, horizon, portfolioValue)
	}

	return &pb.VaRResponse{
		AccountId:      req.AccountId,
		Symbol:         req.Symbol,
		VarValue:       varValue,
		VarPct:         varValue / portfolioValue,
		Confidence:     conf,
		HorizonDays:    int32(horizon),
		Method:         method,
		CvarValue:      cvarValue,
		PortfolioValue: portfolioValue,
		CalculatedAt:   timestamppb.Now(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GetPositionRisk
// ─────────────────────────────────────────────────────────────────────────────

func (s *RiskServer) GetPositionRisk(ctx context.Context, req *pb.PositionRiskRequest) (*pb.PositionRiskResponse, error) {
	if req.AccountId == "" || req.Symbol == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id and symbol are required")
	}

	// Fetch position from portfolio.
	posResp, err := s.portfolioSrv.GetPositions(ctx, &pbp.PositionsRequest{
		AccountId: req.AccountId,
		Symbol:    req.Symbol,
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "portfolio fetch failed: %v", err)
	}

	// Find matching position.
	var pos *pbp.Position
	for _, p := range posResp.Positions {
		if p.Symbol == req.Symbol {
			pos = p
			break
		}
	}
	if pos == nil {
		return nil, status.Errorf(codes.NotFound, "no position found for %s in account %s", req.Symbol, req.AccountId)
	}

	// Compute VaR for this position.
	barResp, err := s.marketSrv.GetBars(ctx, &pbm.BarRequest{
		Symbol:    req.Symbol,
		Timeframe: "1d",
		Limit:     253,
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "bar fetch failed: %v", err)
	}

	closes := extractCloses(barResp.Bars)
	rets := logReturns(closes)
	var95, _ := historicalVaR(rets, 0.95, 1, pos.MarketValue)
	var99, _ := historicalVaR(rets, 0.99, 1, pos.MarketValue)

	// 30-day realized vol.
	vol30 := 0.0
	if len(rets) >= 21 {
		vol30 = stdDev(rets[len(rets)-21:]) * math.Sqrt(252)
	}

	// Beta (approximate using correlation with a market proxy).
	beta := estimateBeta(rets)

	// Days-to-cover.
	avgVol := 0.0
	if len(barResp.Bars) >= 20 {
		vols := extractVolumes(barResp.Bars[len(barResp.Bars)-20:])
		avgVol = meanFloat(vols)
	}
	daysToCover := 0.0
	if avgVol > 0 && pos.Quantity > 0 {
		daysToCover = pos.Quantity / avgVol
	}

	posRisk := &pb.PositionRisk{
		Symbol:           req.Symbol,
		Quantity:         pos.Quantity,
		MarketValue:      pos.MarketValue,
		UnrealizedPnl:    pos.UnrealizedPnl,
		RealizedPnl:      pos.RealizedPnl,
		CostBasis:        pos.CostBasis,
		Var_1D_95:        var95,
		Var_1D_99:        var99,
		Beta:             beta,
		Delta:            1.0, // linear position
		Volatility_30D:   vol30,
		ConcentrationPct: pos.PctOfPortfolio,
		DaysToCover:      daysToCover,
		EntryTime:        pos.OpenedAt,
	}

	limits := s.getLimits(req.AccountId)
	var activeBreaches []*pb.RiskBreach
	if pos.PctOfPortfolio > limits.MaxConcentration {
		activeBreaches = append(activeBreaches, &pb.RiskBreach{
			EventType:    pb.RiskEventType_RISK_EVENT_CONCENTRATION,
			RuleId:       "max_concentration",
			Description:  fmt.Sprintf("Position %.1f%% of portfolio exceeds max %.1f%%", pos.PctOfPortfolio*100, limits.MaxConcentration*100),
			CurrentValue: pos.PctOfPortfolio,
			LimitValue:   limits.MaxConcentration,
		})
	}

	return &pb.PositionRiskResponse{
		Risk:           posRisk,
		Limits:         limits,
		ActiveBreaches: activeBreaches,
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamRiskEvents
// ─────────────────────────────────────────────────────────────────────────────

func (s *RiskServer) StreamRiskEvents(req *pb.StreamRiskRequest, stream pb.RiskService_StreamRiskEventsServer) error {
	if req.AccountId == "" {
		return status.Error(codes.InvalidArgument, "account_id is required")
	}

	ctx := stream.Context()
	ch := make(chan *pb.RiskEvent, 64)

	typeSet := make(map[pb.RiskEventType]bool)
	for _, t := range req.EventTypes {
		typeSet[t] = true
	}
	minSev := req.MinSeverity
	if minSev == pb.Severity_SEVERITY_UNSPECIFIED {
		minSev = pb.Severity_SEVERITY_INFO
	}

	sub := &riskEventSub{ch: ch, types: typeSet, minSeverity: minSev}
	s.addSub(req.AccountId, sub)
	defer func() {
		s.removeSub(req.AccountId, sub)
		close(ch)
	}()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case evt, ok := <-ch:
			if !ok {
				return nil
			}
			if err := stream.Send(evt); err != nil {
				return err
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// GetPortfolioRisk
// ─────────────────────────────────────────────────────────────────────────────

func (s *RiskServer) GetPortfolioRisk(ctx context.Context, req *pb.PortfolioRiskRequest) (*pb.PortfolioRiskResponse, error) {
	if req.AccountId == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id is required")
	}

	summary, err := s.portfolioSrv.GetAccountSummary(ctx, &pbp.AccountSummaryRequest{AccountId: req.AccountId})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "account summary failed: %v", err)
	}

	posResp, err := s.portfolioSrv.GetPositions(ctx, &pbp.PositionsRequest{AccountId: req.AccountId})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "positions failed: %v", err)
	}

	// Portfolio VaR.
	varResp, _ := s.GetVaR(ctx, &pb.VaRRequest{
		AccountId:  req.AccountId,
		Confidence: 0.99,
		HorizonDays: 1,
		Method:     "historical",
	})

	portfolioVar := 0.0
	if varResp != nil {
		portfolioVar = varResp.VarValue
	}

	// Build per-position risk.
	var posRisks []*pb.PositionRisk
	portfolioBeta := 0.0
	totalWeight := 0.0
	for _, pos := range posResp.Positions {
		pr := &pb.PositionRisk{
			Symbol:           pos.Symbol,
			Quantity:         pos.Quantity,
			MarketValue:      pos.MarketValue,
			UnrealizedPnl:    pos.UnrealizedPnl,
			RealizedPnl:      pos.RealizedPnl,
			ConcentrationPct: pos.PctOfPortfolio,
			EntryTime:        pos.OpenedAt,
		}
		posRisks = append(posRisks, pr)
		// Approximate portfolio beta.
		w := math.Abs(pos.PctOfPortfolio)
		portfolioBeta += w * 1.0 // default beta 1.0 if not available
		totalWeight += w
	}
	if totalWeight > 0 {
		portfolioBeta /= totalWeight
	}

	// Stress tests.
	var stressTests []*pb.StressScenario
	if req.IncludeStress {
		stressTests = buildStressScenarios(summary.Equity)
	}

	// Check active portfolio-level breaches.
	limits := s.getLimits(req.AccountId)
	var breaches []*pb.RiskBreach
	if summary.Leverage > limits.MaxLeverage {
		breaches = append(breaches, &pb.RiskBreach{
			EventType:    pb.RiskEventType_RISK_EVENT_CONCENTRATION,
			RuleId:       "max_leverage",
			Description:  fmt.Sprintf("Portfolio leverage %.2fx exceeds max %.2fx", summary.Leverage, limits.MaxLeverage),
			CurrentValue: summary.Leverage,
			LimitValue:   limits.MaxLeverage,
		})
	}

	// Portfolio vol (simplified).
	portfolioVol := 0.15 // assume 15% annualized
	sharpe := 0.0
	if portfolioVol > 0 {
		sharpe = (summary.TotalPnl/summary.Equity) / portfolioVol
	}

	return &pb.PortfolioRiskResponse{
		AccountId:           req.AccountId,
		TotalValue:          summary.Equity,
		TotalExposure:       summary.GrossExposure,
		NetExposure:         summary.NetExposure,
		GrossLeverage:       summary.Leverage,
		NetLeverage:         summary.NetExposure / (summary.Equity + 1e-10),
		PortfolioVar_1D99:   portfolioVar,
		PortfolioBeta:       portfolioBeta,
		PortfolioVolatility: portfolioVol,
		SharpeRatio_30D:     sharpe,
		Positions:           posRisks,
		StressTests:         stressTests,
		ActiveBreaches:      breaches,
		CalculatedAt:        timestamppb.Now(),
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GetDrawdown
// ─────────────────────────────────────────────────────────────────────────────

func (s *RiskServer) GetDrawdown(ctx context.Context, req *pb.DrawdownRequest) (*pb.DrawdownResponse, error) {
	if req.AccountId == "" {
		return nil, status.Error(codes.InvalidArgument, "account_id is required")
	}

	state := s.getOrCreateState(req.AccountId, 1_000_000)
	state.mu.RLock()
	defer state.mu.RUnlock()

	ddPct := 0.0
	if state.peakEquity > 0 {
		ddPct = (state.peakEquity - state.currentEquity) / state.peakEquity
	}

	limits := s.getLimits(req.AccountId)

	return &pb.DrawdownResponse{
		AccountId:       req.AccountId,
		CurrentDrawdown: ddPct,
		MaxDrawdown:     ddPct, // simplified — in prod track rolling max
		PeakValue:       state.peakEquity,
		TroughValue:     state.currentEquity,
		DrawdownLimit:   limits.MaxDrawdownPct,
		LimitBreached:   ddPct > limits.MaxDrawdownPct,
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// VaR methods
// ─────────────────────────────────────────────────────────────────────────────

// historicalVaR computes VaR using the historical simulation method.
func historicalVaR(returns []float64, confidence float64, horizonDays int, portfolioValue float64) (float64, float64) {
	if len(returns) == 0 {
		return portfolioValue * 0.02, portfolioValue * 0.03
	}

	scaledReturns := make([]float64, len(returns))
	sqrtH := math.Sqrt(float64(horizonDays))
	for i, r := range returns {
		scaledReturns[i] = r * sqrtH
	}
	sort.Float64s(scaledReturns)

	// VaR = loss at (1-confidence) quantile.
	varIdx := int(math.Floor((1 - confidence) * float64(len(scaledReturns))))
	if varIdx >= len(scaledReturns) {
		varIdx = len(scaledReturns) - 1
	}
	if varIdx < 0 {
		varIdx = 0
	}
	varReturn := -scaledReturns[varIdx] // make positive (loss)
	varValue := varReturn * portfolioValue

	// CVaR (expected shortfall) = mean of returns beyond VaR.
	cvarSum := 0.0
	cvarCount := 0
	for _, r := range scaledReturns[:varIdx+1] {
		cvarSum += r
		cvarCount++
	}
	cvarReturn := 0.0
	if cvarCount > 0 {
		cvarReturn = -cvarSum / float64(cvarCount)
	}
	cvarValue := cvarReturn * portfolioValue

	return math.Max(0, varValue), math.Max(0, cvarValue)
}

// parametricVaR computes VaR using the variance-covariance (parametric) method.
func parametricVaR(returns []float64, confidence float64, horizonDays int, portfolioValue float64) (float64, float64) {
	mu := meanFloat(returns)
	sigma := stdDev(returns)

	// Scale to horizon.
	sqrtH := math.Sqrt(float64(horizonDays))
	scaledSigma := sigma * sqrtH
	scaledMu := mu * float64(horizonDays)

	// Normal quantile.
	z := normalQuantile(confidence)
	varReturn := -scaledMu + z*scaledSigma
	varValue := varReturn * portfolioValue

	// CVaR for normal: mu - sigma * phi(z) / (1 - confidence)
	phiZ := normalPDF(z)
	cvarReturn := -scaledMu + scaledSigma*phiZ/(1-confidence)
	cvarValue := cvarReturn * portfolioValue

	return math.Max(0, varValue), math.Max(0, cvarValue)
}

// monteCarloVaR computes VaR using Monte Carlo simulation.
func monteCarloVaR(returns []float64, confidence float64, horizonDays int, portfolioValue float64, numSims int) (float64, float64) {
	mu := meanFloat(returns)
	sigma := stdDev(returns)

	rng := rand.New(rand.NewSource(12345))
	simReturns := make([]float64, numSims)

	for i := 0; i < numSims; i++ {
		totalReturn := 0.0
		for d := 0; d < horizonDays; d++ {
			totalReturn += rng.NormFloat64()*sigma + mu
		}
		simReturns[i] = totalReturn
	}

	sort.Float64s(simReturns)
	varIdx := int(math.Floor((1 - confidence) * float64(numSims)))
	if varIdx >= numSims {
		varIdx = numSims - 1
	}
	varReturn := -simReturns[varIdx]
	varValue := varReturn * portfolioValue

	cvarSum := 0.0
	for _, r := range simReturns[:varIdx+1] {
		cvarSum += r
	}
	cvarReturn := 0.0
	if varIdx > 0 {
		cvarReturn = -cvarSum / float64(varIdx)
	}
	cvarValue := cvarReturn * portfolioValue

	return math.Max(0, varValue), math.Max(0, cvarValue)
}

// normalQuantile approximates the inverse CDF of the standard normal.
func normalQuantile(p float64) float64 {
	// Rational approximation (Beasley-Springer-Moro).
	a := []float64{2.515517, 0.802853, 0.010328}
	b := []float64{1.432788, 0.189269, 0.001308}
	t := math.Sqrt(-2 * math.Log(1-p))
	num := a[0] + t*(a[1]+t*a[2])
	den := 1 + t*(b[0]+t*(b[1]+t*b[2]))
	return t - num/den
}

func normalPDF(x float64) float64 {
	return math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
}

// estimateBeta approximates beta vs a market proxy using the return series.
func estimateBeta(returns []float64) float64 {
	// Without actual market returns, use a rule-of-thumb based on vol.
	vol := stdDev(returns) * math.Sqrt(252)
	// Assume market vol ≈ 15%. Beta ≈ asset_vol / market_vol * correlation.
	return vol / 0.15 * 0.7 // assume 0.7 correlation
}

// ─────────────────────────────────────────────────────────────────────────────
// Stress testing
// ─────────────────────────────────────────────────────────────────────────────

func buildStressScenarios(portfolioValue float64) []*pb.StressScenario {
	return []*pb.StressScenario{
		{
			ScenarioName: "2020_covid_crash",
			Description:  "March 2020 COVID-19 market crash (-34% in 33 days)",
			PnlImpact:    portfolioValue * -0.34,
			PctImpact:    -0.34,
			FactorShocks: map[string]float64{"equity": -0.34, "vol_spike": 3.5, "credit_spread": 2.0},
		},
		{
			ScenarioName: "2008_gfc",
			Description:  "2008 Global Financial Crisis (-57% peak-to-trough)",
			PnlImpact:    portfolioValue * -0.57,
			PctImpact:    -0.57,
			FactorShocks: map[string]float64{"equity": -0.57, "vol_spike": 4.0, "credit_spread": 5.0, "liquidity": -0.8},
		},
		{
			ScenarioName: "2000_dotcom",
			Description:  "2000-2002 dot-com bust (-49%)",
			PnlImpact:    portfolioValue * -0.49,
			PctImpact:    -0.49,
			FactorShocks: map[string]float64{"equity": -0.49, "tech_sector": -0.78, "vol_spike": 2.0},
		},
		{
			ScenarioName: "rate_shock_200bps",
			Description:  "Instantaneous +200 bps rate shock",
			PnlImpact:    portfolioValue * -0.12,
			PctImpact:    -0.12,
			FactorShocks: map[string]float64{"rates": 0.02, "equity": -0.08, "duration": -0.12},
		},
		{
			ScenarioName: "vol_spike_3x",
			Description:  "Volatility spike to 3x current level",
			PnlImpact:    portfolioValue * -0.08,
			PctImpact:    -0.08,
			FactorShocks: map[string]float64{"vol_spike": 3.0, "equity": -0.05},
		},
	}
}

// generateSyntheticReturns creates a return series for testing.
func generateSyntheticReturns(n int, mu, sigma float64, seed int64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	rets := make([]float64, n)
	for i := range rets {
		rets[i] = rng.NormFloat64()*sigma + mu
	}
	return rets
}

// ─────────────────────────────────────────────────────────────────────────────
// State and limit helpers
// ─────────────────────────────────────────────────────────────────────────────

func (s *RiskServer) getLimits(accountID string) *pb.RiskLimits {
	s.limitsMu.RLock()
	defer s.limitsMu.RUnlock()
	if l, ok := s.limits[accountID]; ok {
		return l
	}
	defLimits := s.cfg.DefaultLimits
	return &defLimits
}

// SetLimits allows updating per-account risk limits at runtime.
func (s *RiskServer) SetLimits(accountID string, limits *pb.RiskLimits) {
	s.limitsMu.Lock()
	defer s.limitsMu.Unlock()
	s.limits[accountID] = limits
}

func (s *RiskServer) getOrCreateState(accountID string, initialEquity float64) *accountRiskState {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if state, ok := s.accounts[accountID]; ok {
		return state
	}
	state := newAccountRiskState(initialEquity)
	s.accounts[accountID] = state
	return state
}

// UpdateEquity updates the tracked equity for an account (call on every PnL update).
func (s *RiskServer) UpdateEquity(accountID string, equity float64) {
	state := s.getOrCreateState(accountID, equity)
	state.updateEquity(equity)
}

func (s *RiskServer) marginRequirement(symbol, side string) float64 {
	// Return reg-T margin requirement fraction.
	if side == "sell_short" {
		return 1.5 // 150% for short selling
	}
	// Standard equities: 50% initial margin.
	return 0.50
}

// ─────────────────────────────────────────────────────────────────────────────
// Event emission and subscriber management
// ─────────────────────────────────────────────────────────────────────────────

func (s *RiskServer) emitRiskEvent(
	accountID string,
	evtType pb.RiskEventType,
	severity pb.Severity,
	symbol, description string,
	value, limit float64,
) {
	s.eventSeqMu.Lock()
	s.eventSeq++
	seq := s.eventSeq
	s.eventSeqMu.Unlock()

	evt := &pb.RiskEvent{
		EventId:     fmt.Sprintf("risk-%d", seq),
		AccountId:   accountID,
		EventType:   evtType,
		Severity:    severity,
		Symbol:      symbol,
		Description: description,
		Value:       value,
		Limit:       limit,
		OccurredAt:  timestamppb.Now(),
	}

	s.subMu.RLock()
	defer s.subMu.RUnlock()

	for _, sub := range s.subs[accountID] {
		if len(sub.types) > 0 && !sub.types[evtType] {
			continue
		}
		if severity < sub.minSeverity {
			continue
		}
		select {
		case sub.ch <- evt:
		default:
			s.log.Warn("risk event subscriber slow, dropping event", zap.String("account", accountID))
		}
	}
}

func (s *RiskServer) addSub(accountID string, sub *riskEventSub) {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	s.subs[accountID] = append(s.subs[accountID], sub)
}

func (s *RiskServer) removeSub(accountID string, target *riskEventSub) {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	subs := s.subs[accountID]
	updated := subs[:0]
	for _, sub := range subs {
		if sub != target {
			updated = append(updated, sub)
		}
	}
	s.subs[accountID] = updated
}
