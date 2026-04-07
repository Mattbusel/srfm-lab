// cmd/webhook/middleware/auth.go -- HMAC-SHA256 webhook signature verification.
//
// Verifies the X-Alpaca-Signature header against the request body using a
// shared secret from the ALPACA_WEBHOOK_SECRET environment variable.
// Uses crypto/subtle.ConstantTimeCompare to prevent timing attacks.
// Returns HTTP 401 on invalid or missing signature.

package middleware

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

const (
	alpacaSignatureHeader = "X-Alpaca-Signature"
	signaturePrefix       = "sha256="
)

// HMACAuth provides HMAC-SHA256 webhook signature verification middleware.
type HMACAuth struct {
	secret []byte
	logger *slog.Logger
}

// NewHMACAuth creates an HMACAuth middleware.
// If secret is empty, verification is skipped with a warning (dev mode).
func NewHMACAuth(secret string, logger *slog.Logger) *HMACAuth {
	return &HMACAuth{
		secret: []byte(secret),
		logger: logger,
	}
}

// Verify wraps the given handler with HMAC-SHA256 signature verification.
// The request body is read, verified, and then re-injected as a new ReadCloser
// so downstream handlers can also read it.
func (a *HMACAuth) Verify(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// If no secret is configured, skip verification (dev/test mode).
		if len(a.secret) == 0 {
			a.logger.Warn("HMAC verification skipped: no secret configured",
				"path", r.URL.Path, "remote", r.RemoteAddr)
			next.ServeHTTP(w, r)
			return
		}

		sig := r.Header.Get(alpacaSignatureHeader)
		if sig == "" {
			a.logger.Warn("missing signature header",
				"header", alpacaSignatureHeader, "path", r.URL.Path)
			http.Error(w, "missing "+alpacaSignatureHeader+" header", http.StatusUnauthorized)
			return
		}

		// Strip optional "sha256=" prefix.
		sigHex := strings.TrimPrefix(sig, signaturePrefix)

		// Read body.
		body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
		if err != nil {
			http.Error(w, "body read error", http.StatusBadRequest)
			return
		}
		r.Body.Close()

		// Compute expected HMAC.
		mac := hmac.New(sha256.New, a.secret)
		mac.Write(body)
		expectedMAC := mac.Sum(nil)

		// Decode provided signature.
		providedMAC, err := hex.DecodeString(sigHex)
		if err != nil {
			a.logger.Warn("invalid signature encoding",
				"path", r.URL.Path, "sig", sig)
			http.Error(w, "invalid signature encoding", http.StatusUnauthorized)
			return
		}

		// Timing-safe comparison.
		if subtle.ConstantTimeCompare(expectedMAC, providedMAC) != 1 {
			a.logger.Warn("signature mismatch",
				"path", r.URL.Path,
				"remote", r.RemoteAddr,
			)
			http.Error(w, "invalid signature", http.StatusUnauthorized)
			return
		}

		// Re-inject body for downstream handlers.
		r.Body = io.NopCloser(strings.NewReader(string(body)))

		next.ServeHTTP(w, r)
	})
}
