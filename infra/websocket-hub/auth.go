// auth.go — JWT and API key authentication for WebSocket upgrades.
package wshub

import (
	"crypto/subtle"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/golang-jwt/jwt/v5"
)

// ─────────────────────────────────────────────────────────────────────────────
// AuthConfig
// ─────────────────────────────────────────────────────────────────────────────

// AuthConfig configures the authenticator.
type AuthConfig struct {
	// JWTSecret is the HMAC secret for JWT validation.
	JWTSecret []byte

	// APIKeys maps API key → account ID.
	APIKeys map[string]string

	// AllowAnonymous allows unauthenticated connections.
	// Authenticated clients get extra roles; anonymous clients get "viewer" role.
	AllowAnonymous bool

	// AnonymousRooms lists rooms that anonymous clients may join.
	AnonymousRooms []string
}

// ─────────────────────────────────────────────────────────────────────────────
// WSClaims — JWT payload
// ─────────────────────────────────────────────────────────────────────────────

// WSClaims holds the validated JWT payload for a WebSocket client.
type WSClaims struct {
	Subject   string   `json:"sub"`
	AccountID string   `json:"account_id"`
	Roles     []string `json:"roles"`
	jwt.RegisteredClaims
}

// ─────────────────────────────────────────────────────────────────────────────
// Authenticator
// ─────────────────────────────────────────────────────────────────────────────

// Authenticator validates credentials for WebSocket connections.
type Authenticator struct {
	cfg AuthConfig
}

// NewAuthenticator creates an Authenticator.
func NewAuthenticator(cfg AuthConfig) *Authenticator {
	return &Authenticator{cfg: cfg}
}

// AuthenticateRequest extracts and validates credentials from an HTTP request.
// Called during the WebSocket upgrade.
//
// Credential sources (in priority order):
//  1. Authorization: Bearer <JWT>
//  2. X-API-Key: <key>
//  3. ?token=<JWT> query parameter
//  4. ?api_key=<key> query parameter
func (a *Authenticator) AuthenticateRequest(r *http.Request) (*WSClaims, error) {
	// Bearer JWT in Authorization header.
	if auth := r.Header.Get("Authorization"); auth != "" {
		token, found := strings.CutPrefix(auth, "Bearer ")
		if !found {
			token, found = strings.CutPrefix(auth, "bearer ")
		}
		if found && token != "" {
			return a.validateJWT(token)
		}
	}

	// API key in header.
	if apiKey := r.Header.Get("X-API-Key"); apiKey != "" {
		return a.validateAPIKey(apiKey)
	}

	// JWT in query string.
	if token := r.URL.Query().Get("token"); token != "" {
		return a.validateJWT(token)
	}

	// API key in query string.
	if apiKey := r.URL.Query().Get("api_key"); apiKey != "" {
		return a.validateAPIKey(apiKey)
	}

	if a.cfg.AllowAnonymous {
		return &WSClaims{
			Subject: "anonymous",
			Roles:   []string{"viewer"},
		}, nil
	}

	return nil, errors.New("no credentials provided")
}

// Authenticate validates credentials from a post-connect auth message.
func (a *Authenticator) Authenticate(token, apiKey string) (*WSClaims, error) {
	if token != "" {
		return a.validateJWT(token)
	}
	if apiKey != "" {
		return a.validateAPIKey(apiKey)
	}
	if a.cfg.AllowAnonymous {
		return &WSClaims{Subject: "anonymous", Roles: []string{"viewer"}}, nil
	}
	return nil, errors.New("no credentials")
}

// validateJWT parses and validates a JWT token.
func (a *Authenticator) validateJWT(tokenStr string) (*WSClaims, error) {
	if len(a.cfg.JWTSecret) == 0 {
		return nil, errors.New("JWT auth not configured")
	}

	claims := &WSClaims{}
	tok, err := jwt.ParseWithClaims(tokenStr, claims, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", t.Header["alg"])
		}
		return a.cfg.JWTSecret, nil
	})
	if err != nil {
		return nil, fmt.Errorf("JWT validation: %w", err)
	}
	if !tok.Valid {
		return nil, errors.New("token is invalid")
	}
	return claims, nil
}

// validateAPIKey checks the API key against the configured map.
func (a *Authenticator) validateAPIKey(key string) (*WSClaims, error) {
	if a.cfg.APIKeys == nil {
		return nil, errors.New("API key auth not configured")
	}
	// Constant-time comparison to prevent timing attacks.
	for k, accountID := range a.cfg.APIKeys {
		if subtle.ConstantTimeCompare([]byte(k), []byte(key)) == 1 {
			return &WSClaims{
				Subject:   accountID,
				AccountID: accountID,
				Roles:     []string{"trader"},
			}, nil
		}
	}
	return nil, errors.New("invalid API key")
}

// CanJoinRoom checks whether claims allow joining a specific room.
func (a *Authenticator) CanJoinRoom(claims *WSClaims, roomName string) (bool, string) {
	if claims.Subject == "anonymous" {
		for _, allowed := range a.cfg.AnonymousRooms {
			if allowed == roomName || allowed == "*" {
				return true, ""
			}
		}
		return false, "anonymous access not allowed for this room"
	}
	return true, ""
}

// HasRole checks if the claims include a specific role.
func HasRole(claims *WSClaims, role string) bool {
	for _, r := range claims.Roles {
		if r == role || r == "admin" {
			return true
		}
	}
	return false
}
