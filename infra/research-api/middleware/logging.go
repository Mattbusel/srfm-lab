package middleware

import (
	"fmt"
	"net/http"
	"time"

	"go.uber.org/zap"
)

// responseWriter wraps http.ResponseWriter to capture the status code written
// by the downstream handler.
type responseWriter struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
	bytes       int
}

func wrapResponseWriter(w http.ResponseWriter) *responseWriter {
	return &responseWriter{ResponseWriter: w, status: http.StatusOK}
}

func (rw *responseWriter) WriteHeader(code int) {
	if rw.wroteHeader {
		return
	}
	rw.status = code
	rw.wroteHeader = true
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	n, err := rw.ResponseWriter.Write(b)
	rw.bytes += n
	return n, err
}

// Logging returns a middleware that logs each request with method, path,
// status code, response size, remote address, and elapsed duration using the
// provided zap.Logger.
//
// Requests to /api/v1/health are logged at Debug level to avoid noisy health
// check spam in production logs.
func Logging(log *zap.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			wrapped := wrapResponseWriter(w)

			// Propagate a request ID if present so downstream handlers can use it.
			reqID := r.Header.Get("X-Request-ID")
			if reqID == "" {
				reqID = fmt.Sprintf("%d", start.UnixNano())
			}
			wrapped.Header().Set("X-Request-ID", reqID)

			defer func() {
				elapsed := time.Since(start)
				fields := []zap.Field{
					zap.String("method", r.Method),
					zap.String("path", r.URL.Path),
					zap.String("query", r.URL.RawQuery),
					zap.Int("status", wrapped.status),
					zap.Int("bytes", wrapped.bytes),
					zap.String("remote", r.RemoteAddr),
					zap.Duration("duration", elapsed),
					zap.String("request_id", reqID),
				}

				if r.URL.Path == "/api/v1/health" {
					log.Debug("request", fields...)
				} else if wrapped.status >= 500 {
					log.Error("request", fields...)
				} else if wrapped.status >= 400 {
					log.Warn("request", fields...)
				} else {
					log.Info("request", fields...)
				}
			}()

			next.ServeHTTP(wrapped, r)
		})
	}
}
