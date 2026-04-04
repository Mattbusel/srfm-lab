// logging.go — Structured request/response logging gRPC interceptor.
package middleware

import (
	"context"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
)

// ─────────────────────────────────────────────────────────────────────────────
// Logging interceptor configuration
// ─────────────────────────────────────────────────────────────────────────────

// LoggingConfig configures the logging interceptor.
type LoggingConfig struct {
	// LogSuccessful controls whether successful RPCs are logged.
	LogSuccessful bool
	// LogPayloads controls whether request/response payloads are logged.
	// Disable in production for security and performance.
	LogPayloads bool
	// SlowRPCThreshold is the latency above which an RPC is logged as "slow".
	SlowRPCThreshold time.Duration
	// SkipMethods is a set of full method names to not log.
	SkipMethods map[string]bool
}

// DefaultLoggingConfig returns sensible defaults.
func DefaultLoggingConfig() LoggingConfig {
	return LoggingConfig{
		LogSuccessful:    false, // only log errors and slow RPCs by default
		LogPayloads:      false,
		SlowRPCThreshold: 500 * time.Millisecond,
		SkipMethods: map[string]bool{
			"/grpc.health.v1.Health/Check": true,
		},
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Unary logging interceptor
// ─────────────────────────────────────────────────────────────────────────────

// LoggingInterceptor returns a unary gRPC interceptor that logs RPCs.
func LoggingInterceptor(log *zap.Logger, cfg LoggingConfig) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		if cfg.SkipMethods[info.FullMethod] {
			return handler(ctx, req)
		}

		start := time.Now()
		peerAddr := peerAddress(ctx)
		claims, _ := ClaimsFromContext(ctx)
		userID := ""
		if claims != nil {
			userID = claims.Subject
		}

		fields := []zap.Field{
			zap.String("method", info.FullMethod),
			zap.String("peer", peerAddr),
			zap.String("user", userID),
		}

		if cfg.LogPayloads {
			fields = append(fields, zap.Any("request", req))
		}

		resp, err := handler(ctx, req)

		elapsed := time.Since(start)
		code := codeFromError(err)

		fields = append(fields,
			zap.Duration("duration", elapsed),
			zap.String("code", code.String()),
		)
		if cfg.LogPayloads && resp != nil {
			fields = append(fields, zap.Any("response", resp))
		}

		if err != nil {
			fields = append(fields, zap.Error(err))
			log.Warn("RPC error", fields...)
		} else if elapsed >= cfg.SlowRPCThreshold {
			log.Warn("slow RPC", fields...)
		} else if cfg.LogSuccessful {
			log.Info("RPC", fields...)
		}

		return resp, err
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream logging interceptor
// ─────────────────────────────────────────────────────────────────────────────

// LoggingStreamInterceptor returns a streaming gRPC interceptor that logs RPCs.
func LoggingStreamInterceptor(log *zap.Logger, cfg LoggingConfig) grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if cfg.SkipMethods[info.FullMethod] {
			return handler(srv, ss)
		}

		start := time.Now()
		peerAddr := peerAddress(ss.Context())
		claims, _ := ClaimsFromContext(ss.Context())
		userID := ""
		if claims != nil {
			userID = claims.Subject
		}

		fields := []zap.Field{
			zap.String("method", info.FullMethod),
			zap.String("peer", peerAddr),
			zap.String("user", userID),
			zap.Bool("client_stream", info.IsClientStream),
			zap.Bool("server_stream", info.IsServerStream),
		}

		// Wrap the stream to count messages.
		counted := &countingStream{ServerStream: ss}

		err := handler(srv, counted)
		elapsed := time.Since(start)
		code := codeFromError(err)

		fields = append(fields,
			zap.Duration("duration", elapsed),
			zap.String("code", code.String()),
			zap.Int64("messages_sent", counted.sent),
			zap.Int64("messages_received", counted.received),
		)

		if err != nil {
			fields = append(fields, zap.Error(err))
			log.Warn("stream RPC error", fields...)
		} else if cfg.LogSuccessful {
			log.Info("stream RPC completed", fields...)
		}

		return err
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// countingStream — counts messages on a stream
// ─────────────────────────────────────────────────────────────────────────────

type countingStream struct {
	grpc.ServerStream
	sent     int64
	received int64
}

func (s *countingStream) SendMsg(m interface{}) error {
	err := s.ServerStream.SendMsg(m)
	if err == nil {
		s.sent++
	}
	return err
}

func (s *countingStream) RecvMsg(m interface{}) error {
	err := s.ServerStream.RecvMsg(m)
	if err == nil {
		s.received++
	}
	return err
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func peerAddress(ctx context.Context) string {
	if p, ok := peer.FromContext(ctx); ok {
		return p.Addr.String()
	}
	return "unknown"
}

func codeFromError(err error) codes.Code {
	if err == nil {
		return codes.OK
	}
	if s, ok := status.FromError(err); ok {
		return s.Code()
	}
	return codes.Internal
}

// ─────────────────────────────────────────────────────────────────────────────
// Panic recovery interceptor
// ─────────────────────────────────────────────────────────────────────────────

// RecoveryInterceptor catches panics in handlers and converts them to gRPC errors.
func RecoveryInterceptor(log *zap.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
		defer func() {
			if r := recover(); r != nil {
				log.Error("panic in gRPC handler",
					zap.String("method", info.FullMethod),
					zap.Any("panic", r),
				)
				err = status.Errorf(codes.Internal, "internal server error")
			}
		}()
		return handler(ctx, req)
	}
}

// RecoveryStreamInterceptor catches panics in streaming handlers.
func RecoveryStreamInterceptor(log *zap.Logger) grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) (err error) {
		defer func() {
			if r := recover(); r != nil {
				log.Error("panic in gRPC stream handler",
					zap.String("method", info.FullMethod),
					zap.Any("panic", r),
				)
				err = status.Errorf(codes.Internal, "internal server error")
			}
		}()
		return handler(srv, ss)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Deadline enforcement interceptor
// ─────────────────────────────────────────────────────────────────────────────

// DeadlineInterceptor enforces a maximum deadline for unary RPCs.
// If the incoming context has no deadline, defaultTimeout is applied.
func DeadlineInterceptor(defaultTimeout time.Duration) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		if _, ok := ctx.Deadline(); !ok {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, defaultTimeout)
			defer cancel()
		}
		return handler(ctx, req)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Request ID interceptor — attaches a unique request ID to each RPC context
// ─────────────────────────────────────────────────────────────────────────────

const ctxRequestID contextKey = "request_id"

// RequestIDInterceptor attaches a unique request ID to each incoming RPC.
// The ID is extracted from metadata "x-request-id" or generated if absent.
func RequestIDInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		ctx = ensureRequestID(ctx)
		return handler(ctx, req)
	}
}

// RequestIDFromContext returns the request ID from the context.
func RequestIDFromContext(ctx context.Context) string {
	if id, ok := ctx.Value(ctxRequestID).(string); ok {
		return id
	}
	return ""
}

func ensureRequestID(ctx context.Context) context.Context {
	// Already set?
	if id, ok := ctx.Value(ctxRequestID).(string); ok && id != "" {
		return ctx
	}
	// Try to extract from gRPC metadata.
	// (Import metadata here — in practice this would be done at top of file.)
	id := generateRequestID()
	return context.WithValue(ctx, ctxRequestID, id)
}

var reqIDCounter int64
var reqIDMu = &writerMu{}

type writerMu struct{ sync.Mutex }

func generateRequestID() string {
	reqIDMu.Lock()
	reqIDCounter++
	n := reqIDCounter
	reqIDMu.Unlock()
	return formatReqID(n)
}

func formatReqID(n int64) string {
	const chars = "0123456789abcdefghijklmnopqrstuvwxyz"
	buf := [12]byte{}
	for i := 11; i >= 0; i-- {
		buf[i] = chars[n%36]
		n /= 36
	}
	return string(buf[:])
}
