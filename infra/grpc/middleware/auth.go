// Package middleware provides gRPC server interceptors for auth, rate limiting,
// Prometheus metrics, and OpenTelemetry distributed tracing.
package middleware

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// ─────────────────────────────────────────────────────────────────────────────
// JWT Auth Interceptor
// ─────────────────────────────────────────────────────────────────────────────

// contextKey is a package-local key type for context values.
type contextKey string

const (
	ctxClaims contextKey = "auth_claims"
	ctxAPIKey contextKey = "api_key"
)

// Claims holds the validated JWT payload.
type Claims struct {
	Subject   string   `json:"sub"`
	AccountID string   `json:"account_id"`
	Roles     []string `json:"roles"`
	jwt.RegisteredClaims
}

// AuthConfig configures the authentication interceptor.
type AuthConfig struct {
	JWTSecret    []byte
	APIKeys      map[string]string // api_key -> account_id
	SkipPaths    []string          // gRPC full method names to skip auth
	AllowAnonymous bool
}

// AuthInterceptor returns a unary gRPC interceptor that validates JWT or API key auth.
func AuthInterceptor(cfg AuthConfig, log *zap.Logger) grpc.UnaryServerInterceptor {
	skipSet := make(map[string]bool)
	for _, p := range cfg.SkipPaths {
		skipSet[p] = true
	}

	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		if skipSet[info.FullMethod] || cfg.AllowAnonymous {
			return handler(ctx, req)
		}

		ctx, err := authenticate(ctx, cfg, log)
		if err != nil {
			return nil, err
		}
		return handler(ctx, req)
	}
}

// AuthStreamInterceptor returns a streaming gRPC interceptor that validates auth.
func AuthStreamInterceptor(cfg AuthConfig, log *zap.Logger) grpc.StreamServerInterceptor {
	skipSet := make(map[string]bool)
	for _, p := range cfg.SkipPaths {
		skipSet[p] = true
	}

	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if skipSet[info.FullMethod] || cfg.AllowAnonymous {
			return handler(srv, ss)
		}

		ctx, err := authenticate(ss.Context(), cfg, log)
		if err != nil {
			return err
		}
		return handler(srv, &wrappedStream{ServerStream: ss, ctx: ctx})
	}
}

// authenticate attempts JWT or API key authentication.
func authenticate(ctx context.Context, cfg AuthConfig, log *zap.Logger) (context.Context, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, status.Error(codes.Unauthenticated, "missing metadata")
	}

	// Try Bearer JWT first.
	authVals := md.Get("authorization")
	if len(authVals) > 0 {
		bearer := authVals[0]
		token, found := strings.CutPrefix(bearer, "Bearer ")
		if !found {
			token, found = strings.CutPrefix(bearer, "bearer ")
		}
		if found && token != "" {
			claims, err := validateJWT(token, cfg.JWTSecret)
			if err != nil {
				log.Warn("JWT validation failed", zap.Error(err))
				return nil, status.Errorf(codes.Unauthenticated, "invalid token: %v", err)
			}
			ctx = context.WithValue(ctx, ctxClaims, claims)
			return ctx, nil
		}
	}

	// Try API key.
	apiKeyVals := md.Get("x-api-key")
	if len(apiKeyVals) > 0 && cfg.APIKeys != nil {
		apiKey := apiKeyVals[0]
		if accountID, ok := cfg.APIKeys[apiKey]; ok {
			ctx = context.WithValue(ctx, ctxAPIKey, apiKey)
			ctx = context.WithValue(ctx, ctxClaims, &Claims{
				Subject:   accountID,
				AccountID: accountID,
				Roles:     []string{"trader"},
			})
			return ctx, nil
		}
		return nil, status.Error(codes.Unauthenticated, "invalid API key")
	}

	return nil, status.Error(codes.Unauthenticated, "no credentials provided")
}

// validateJWT parses and validates a JWT token, returning Claims on success.
func validateJWT(tokenStr string, secret []byte) (*Claims, error) {
	claims := &Claims{}
	token, err := jwt.ParseWithClaims(tokenStr, claims, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, status.Errorf(codes.Unauthenticated, "unexpected signing method: %v", t.Header["alg"])
		}
		return secret, nil
	})
	if err != nil {
		return nil, err
	}
	if !token.Valid {
		return nil, jwt.ErrTokenInvalidClaims
	}
	return claims, nil
}

// ClaimsFromContext extracts the authenticated Claims from a context.
func ClaimsFromContext(ctx context.Context) (*Claims, bool) {
	c, ok := ctx.Value(ctxClaims).(*Claims)
	return c, ok
}

// RequireRole returns a gRPC unary interceptor that enforces a required role.
func RequireRole(role string) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		claims, ok := ClaimsFromContext(ctx)
		if !ok {
			return nil, status.Error(codes.Unauthenticated, "not authenticated")
		}
		for _, r := range claims.Roles {
			if r == role || r == "admin" {
				return handler(ctx, req)
			}
		}
		return nil, status.Errorf(codes.PermissionDenied, "role %q required", role)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// wrappedStream replaces the context on a grpc.ServerStream.
// ─────────────────────────────────────────────────────────────────────────────

type wrappedStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (w *wrappedStream) Context() context.Context { return w.ctx }

// ─────────────────────────────────────────────────────────────────────────────
// Rate Limiter Interceptor
// ─────────────────────────────────────────────────────────────────────────────

// RateLimiterConfig configures per-method and per-client rate limits.
type RateLimiterConfig struct {
	// GlobalRPS is the maximum requests per second across all clients.
	GlobalRPS float64
	// PerClientRPS is the max RPS per authenticated client (identified by Claims.Subject).
	PerClientRPS float64
	// PerMethodRPS overrides GlobalRPS for specific methods.
	PerMethodRPS map[string]float64
	// BurstMultiplier allows temporary bursts above the RPS limit.
	BurstMultiplier float64
}

// tokenBucket implements a simple token-bucket rate limiter.
type tokenBucket struct {
	mu         sync.Mutex
	tokens     float64
	maxTokens  float64
	refillRate float64 // tokens per second
	lastRefill time.Time
}

func newTokenBucket(rps, burst float64) *tokenBucket {
	return &tokenBucket{
		tokens:     burst,
		maxTokens:  burst,
		refillRate: rps,
		lastRefill: time.Now(),
	}
}

// allow returns true if a token is available (and consumes it).
func (tb *tokenBucket) allow() bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tb.tokens += elapsed * tb.refillRate
	if tb.tokens > tb.maxTokens {
		tb.tokens = tb.maxTokens
	}
	tb.lastRefill = now

	if tb.tokens < 1 {
		return false
	}
	tb.tokens--
	return true
}

// rateLimiterState holds per-client buckets.
type rateLimiterState struct {
	mu      sync.Mutex
	buckets map[string]*tokenBucket
	cfg     RateLimiterConfig
}

func newRateLimiterState(cfg RateLimiterConfig) *rateLimiterState {
	return &rateLimiterState{
		buckets: make(map[string]*tokenBucket),
		cfg:     cfg,
	}
}

func (s *rateLimiterState) getOrCreate(clientID string) *tokenBucket {
	s.mu.Lock()
	defer s.mu.Unlock()
	if b, ok := s.buckets[clientID]; ok {
		return b
	}
	burst := s.cfg.PerClientRPS * s.cfg.BurstMultiplier
	if burst <= 0 {
		burst = s.cfg.PerClientRPS * 3
	}
	b := newTokenBucket(s.cfg.PerClientRPS, burst)
	s.buckets[clientID] = b
	return b
}

// RateLimiterInterceptor returns a unary gRPC interceptor that enforces rate limits.
func RateLimiterInterceptor(cfg RateLimiterConfig) grpc.UnaryServerInterceptor {
	globalBurst := cfg.GlobalRPS * 5
	if cfg.BurstMultiplier > 0 {
		globalBurst = cfg.GlobalRPS * cfg.BurstMultiplier
	}
	globalBucket := newTokenBucket(cfg.GlobalRPS, globalBurst)
	clientState := newRateLimiterState(cfg)

	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		// Global limit.
		if !globalBucket.allow() {
			return nil, status.Error(codes.ResourceExhausted, "server rate limit exceeded")
		}

		// Per-method override.
		if rps, ok := cfg.PerMethodRPS[info.FullMethod]; ok {
			methodBucket := newTokenBucket(rps, rps*3)
			if !methodBucket.allow() {
				return nil, status.Errorf(codes.ResourceExhausted, "method rate limit exceeded for %s", info.FullMethod)
			}
		}

		// Per-client limit.
		if cfg.PerClientRPS > 0 {
			clientID := "anonymous"
			if claims, ok := ClaimsFromContext(ctx); ok {
				clientID = claims.Subject
			}
			bucket := clientState.getOrCreate(clientID)
			if !bucket.allow() {
				return nil, status.Error(codes.ResourceExhausted, "client rate limit exceeded")
			}
		}

		return handler(ctx, req)
	}
}

// RateLimiterStreamInterceptor applies rate limiting to streaming RPCs.
func RateLimiterStreamInterceptor(cfg RateLimiterConfig) grpc.StreamServerInterceptor {
	globalBurst := cfg.GlobalRPS
	globalBucket := newTokenBucket(cfg.GlobalRPS/10, globalBurst) // lower limit for stream opens
	clientState := newRateLimiterState(cfg)

	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if !globalBucket.allow() {
			return status.Error(codes.ResourceExhausted, "server stream rate limit exceeded")
		}
		if cfg.PerClientRPS > 0 {
			clientID := "anonymous"
			if claims, ok := ClaimsFromContext(ss.Context()); ok {
				clientID = claims.Subject
			}
			bucket := clientState.getOrCreate(clientID)
			if !bucket.allow() {
				return status.Error(codes.ResourceExhausted, "client stream rate limit exceeded")
			}
		}
		return handler(srv, ss)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Chain helper — composes multiple unary interceptors.
// ─────────────────────────────────────────────────────────────────────────────

// ChainUnary composes multiple UnaryServerInterceptors left to right.
func ChainUnary(interceptors ...grpc.UnaryServerInterceptor) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		h := handler
		for i := len(interceptors) - 1; i >= 0; i-- {
			i, curr := i, interceptors[i]
			next := h
			_ = i
			h = func(ctx context.Context, req interface{}) (interface{}, error) {
				return curr(ctx, req, info, next)
			}
		}
		return h(ctx, req)
	}
}

// ChainStream composes multiple StreamServerInterceptors left to right.
func ChainStream(interceptors ...grpc.StreamServerInterceptor) grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		h := handler
		for i := len(interceptors) - 1; i >= 0; i-- {
			curr := interceptors[i]
			next := h
			h = func(srv interface{}, ss grpc.ServerStream) error {
				return curr(srv, ss, info, next)
			}
		}
		return h(srv, ss)
	}
}
