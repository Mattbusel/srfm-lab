// Package middleware — auth.go
//
// AuthMiddleware provides two authentication strategies for webhook endpoints:
//
//  1. HMAC-SHA256 — validates the X-Webhook-Signature header against the
//     request body using a shared secret.  Used for TradingView webhooks.
//
//  2. API key — validates the X-API-Key header or Bearer token in
//     Authorization against a configured key.  Used for custom webhooks.
//
// Middleware variants:
//
//   HMACOrAPIKey — requires EITHER a valid HMAC signature OR a valid API key.
//                  If hmacSecret is empty, falls back to API-key-only.
//   APIKeyOrOpen  — requires a valid API key if one is configured; otherwise
//                   allows the request through (for development/testing).
//   HMACRequired  — strictly requires a valid HMAC signature.
//   APIKeyRequired — strictly requires a valid API key.
package middleware

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"io"
	"net/http"
	"strings"

	"go.uber.org/zap"
)

// AuthMiddleware holds the shared secrets used for authentication.
type AuthMiddleware struct {
	hmacSecret []byte
	apiKey     string
	logger     *zap.Logger
}

// NewAuthMiddleware creates a new AuthMiddleware.
// Pass empty strings to disable the respective authentication method.
func NewAuthMiddleware(hmacSecret, apiKey string, logger *zap.Logger) *AuthMiddleware {
	return &AuthMiddleware{
		hmacSecret: []byte(hmacSecret),
		apiKey:     apiKey,
		logger:     logger,
	}
}

// ---------------------------------------------------------------------------
// Middleware variants
// ---------------------------------------------------------------------------

// HMACOrAPIKey allows requests with a valid HMAC signature OR a valid API key.
// If neither method is configured, all requests are allowed through (useful
// for local development).
func (a *AuthMiddleware) HMACOrAPIKey(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(a.hmacSecret) == 0 && a.apiKey == "" {
			next.ServeHTTP(w, r)
			return
		}

		// Try HMAC first (requires reading body, so we buffer it)
		if len(a.hmacSecret) > 0 {
			body, err := bufferBody(r)
			if err != nil {
				authError(w, "could not read request body")
				return
			}
			sig := r.Header.Get("X-Webhook-Signature")
			if sig != "" && a.verifyHMAC(body, sig) {
				next.ServeHTTP(w, r)
				return
			}
		}

		// Try API key
		if a.apiKey != "" {
			if key := extractAPIKey(r); key != "" && a.verifyAPIKey(key) {
				next.ServeHTTP(w, r)
				return
			}
		}

		a.logger.Warn("auth failed: HMACOrAPIKey",
			zap.String("path", r.URL.Path),
			zap.String("remote", r.RemoteAddr),
		)
		authError(w, "authentication required: provide HMAC signature or API key")
	})
}

// APIKeyOrOpen allows requests with a valid API key, or all requests if no
// API key is configured.
func (a *AuthMiddleware) APIKeyOrOpen(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if a.apiKey == "" {
			// No key configured — open access (dev mode)
			next.ServeHTTP(w, r)
			return
		}
		key := extractAPIKey(r)
		if key == "" || !a.verifyAPIKey(key) {
			a.logger.Warn("auth failed: APIKeyOrOpen",
				zap.String("path", r.URL.Path),
				zap.String("remote", r.RemoteAddr),
			)
			authError(w, "valid API key required")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// HMACRequired strictly requires a valid HMAC-SHA256 signature.
// Returns 401 if the signature is missing or invalid.
func (a *AuthMiddleware) HMACRequired(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if len(a.hmacSecret) == 0 {
			// No secret configured — allow (misconfiguration warning)
			a.logger.Warn("HMACRequired middleware configured but hmacSecret is empty")
			next.ServeHTTP(w, r)
			return
		}
		body, err := bufferBody(r)
		if err != nil {
			authError(w, "could not read request body")
			return
		}
		sig := r.Header.Get("X-Webhook-Signature")
		if sig == "" {
			authError(w, "missing X-Webhook-Signature header")
			return
		}
		if !a.verifyHMAC(body, sig) {
			a.logger.Warn("HMAC verification failed",
				zap.String("path", r.URL.Path),
			)
			authError(w, "invalid HMAC signature")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// APIKeyRequired strictly requires a valid API key.
func (a *AuthMiddleware) APIKeyRequired(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if a.apiKey == "" {
			a.logger.Warn("APIKeyRequired middleware configured but apiKey is empty")
			next.ServeHTTP(w, r)
			return
		}
		key := extractAPIKey(r)
		if key == "" {
			authError(w, "missing API key: provide X-API-Key header or Bearer token")
			return
		}
		if !a.verifyAPIKey(key) {
			a.logger.Warn("API key verification failed",
				zap.String("path", r.URL.Path),
			)
			authError(w, "invalid API key")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------

// verifyHMAC checks that sig == hex(HMAC-SHA256(secret, body)).
// Uses constant-time comparison to prevent timing attacks.
func (a *AuthMiddleware) verifyHMAC(body []byte, sig string) bool {
	mac := hmac.New(sha256.New, a.hmacSecret)
	mac.Write(body)
	expected := hex.EncodeToString(mac.Sum(nil))
	// Constant-time compare to prevent timing side-channels
	return subtle.ConstantTimeCompare([]byte(expected), []byte(sig)) == 1
}

// verifyAPIKey checks the provided key against the configured key.
// Constant-time compare prevents timing attacks.
func (a *AuthMiddleware) verifyAPIKey(key string) bool {
	return subtle.ConstantTimeCompare([]byte(a.apiKey), []byte(key)) == 1
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

// extractAPIKey looks for an API key in:
//  1. X-API-Key header
//  2. Authorization: Bearer <token> header
//  3. api_key query parameter (less secure; for convenience)
func extractAPIKey(r *http.Request) string {
	if k := r.Header.Get("X-API-Key"); k != "" {
		return k
	}
	if auth := r.Header.Get("Authorization"); auth != "" {
		const prefix = "Bearer "
		if strings.HasPrefix(auth, prefix) {
			return auth[len(prefix):]
		}
	}
	// Query parameter fallback (not recommended for production)
	if k := r.URL.Query().Get("api_key"); k != "" {
		return k
	}
	return ""
}

// bufferBody reads the full request body into a byte slice and restores it
// so subsequent handlers can still read it.
func bufferBody(r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20)) // 1 MB limit
	if err != nil {
		return nil, err
	}
	// Restore body for downstream handlers
	r.Body = io.NopCloser(bytes.NewReader(body))
	return body, nil
}

// ---------------------------------------------------------------------------
// Response helper
// ---------------------------------------------------------------------------

func authError(w http.ResponseWriter, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("WWW-Authenticate", `Bearer realm="webhook-service"`)
	w.WriteHeader(http.StatusUnauthorized)
	_, _ = w.Write([]byte(`{"error":"` + msg + `"}`))
}
